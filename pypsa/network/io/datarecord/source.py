# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Network-to-record export sources of the PyPSA tool."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from datarecord.duck import empty_relation, null_safe
from datarecord.protocols import Flags, LazyRelations, RecordLike
from datarecord.resolve.owner_map import cast_declared
from datarecord.schema.schema import Schema as RecordSchema
from datarecord.schema.specs import (
    AttributeSpec,
)
from datarecord.settle import settle
from duckdb import CoalesceOperator as coalesce
from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import (
    DuckDBPyConnection,
    DuckDBPyRelation,
    sqltypes,
)
from duckdb import StarExpression as star

if TYPE_CHECKING:
    import pypsa
from pypsa.network.io._common import (
    _collect_network_attributes,
)
from pypsa.network.io.datarecord.schema import (
    _AXIS_TYPES,
    _CUSTOM,
    _DERIVED,
    _PROPERTIES,
    _WEIGHTINGS,
    BUS,
    CARRIER,
    CONNECTION,
    CTYPE,
    ENTITY,
    PERIOD,
    SCENARIO,
    SNAPSHOT,
    SUB_NETWORK,
    TIMESTEP,
    _connection_struct,
    _dim_of,
    _geometries,
    _input_names,
    _is_output,
    _output_attributes,
    _port_attribute,
    _port_names,
    _port_role,
    _port_stems,
    _registry_schema,
    _result_attributes,
    _static_dims,
)
from pypsa.network.io.datarecord.shape import (
    Index,
    _snapshot_frame,
    _snapshot_properties,
)


def _exported(c: pypsa.Components) -> bool:
    """Whether `c`'s members belong in a record at all.

    Empty types have nothing to write. Standard types (`LineType`,
    `TransformerType`) are excluded because PyPSA prepopulates them on every
    fresh `Network` and `_broadcast_standard_types` restores them on build -
    writing them would import each row a second time on the way back.
    """
    return not c.static.empty and c.name not in c.n.standard_type_components  # type: ignore[union-attr]


def _as_entities(c: pypsa.Components) -> bool:
    """Whether `c`'s members are written as entities, not as labels of their own dim."""
    return _exported(c) and c.name not in _AXIS_TYPES


def _keyed_as_entity(rel: DuckDBPyRelation, dim: str) -> DuckDBPyRelation:
    """`rel` with its `dim` column named `entity`, the key PyPSA's frames are built on."""
    if dim == ENTITY:
        return rel
    return rel.project(
        *(col(c).alias(ENTITY) if c == dim else col(c) for c in rel.columns)
    )


def _none_for_empty(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with an empty `carrier`, PyPSA's "none", as NULL: it names no label."""
    if CARRIER not in frame:
        return frame
    return frame.assign(**{CARRIER: frame[CARRIER].where(frame[CARRIER] != "", None)})


def _empty_for_none(rel: DuckDBPyRelation) -> DuckDBPyRelation:
    """`rel` with a NULL `carrier` as PyPSA's `""`, the inverse of `_none_for_empty`."""
    if CARRIER not in rel.columns:
        return rel
    return rel.project(
        *(
            coalesce(col(c), lit("")).alias(c) if c == CARRIER else col(c)
            for c in rel.columns
        )
    )


def _colliding_names(n: pypsa.Network) -> frozenset[str]:
    """Names more than one exported entity type claims.

    Only the types written as entities count, since only those become member
    rows: a bus is a label of `bus`, so a Bus and a Load may share a name. Read off the
    `name` level rather than the index: a stochastic network is keyed
    `(scenario, name)`, and comparing tuples would read one component in two
    scenarios as two names while missing a real collision.

    Notes
    -----
    - [a label is unique within its dim](https://energy-models.github.io/datarecord/design/storage/format/#a-label-is-unique-within-its-dim)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

    """
    seen: dict[str, str] = {}
    clashing: set[str] = set()
    for c in n.components:
        if not _as_entities(c):
            continue
        index = c.static.index
        names = (
            index.get_level_values("name") if "name" in (index.names or []) else index
        )
        for value in names:
            name = str(value)
            if seen.setdefault(name, c.name) != c.name:
                clashing.add(name)
    return frozenset(clashing)


def _repeated_buses(n: pypsa.Network) -> list[str]:
    """One line per component with the same bus on two ports.

    A `connection` is keyed `(entity, bus)`, so the second port of such a
    component would vanish from the record.
    """
    found: list[str] = []
    for c in n.components:
        if not _exported(c):
            continue
        ports = [col for col in c.static.columns if re.fullmatch(r"bus\d*", str(col))]
        if len(ports) < 2:
            continue
        buses = c.static[ports].astype(str)
        unset = c.static[ports].isna() | buses.eq("")
        for (index, row), (_, gone) in zip(
            buses.iterrows(), unset.iterrows(), strict=True
        ):
            name = index[-1] if isinstance(index, tuple) else index
            by_bus: dict[str, list[str]] = {}
            for port, bus in row.items():
                if not gone[port]:
                    by_bus.setdefault(bus, []).append(str(port))
            found.extend(
                f"{c.name} '{name}' has bus '{bus}' on {', '.join(at)}"
                for bus, at in by_bus.items()
                if len(at) > 1
            )
    return found


def _custom_names(c: pypsa.Components) -> list[str]:
    """`c`'s static columns PyPSA's registry has no entry for, but `_DERIVED`."""
    derived = _DERIVED.get(c.name, frozenset())
    return [
        name
        for name in c.static.columns
        if name not in c.defaults.index and name not in derived
    ]


def _custom_columns(n: pypsa.Network) -> dict[str, dict[str, str]]:
    """Exported type -> its custom static columns (`_custom_names`), typed.

    A network may carry its own (`ac_dc_meshed`'s `Bus.country`). Each is an
    attribute the network's schema declares over the type's dim (`_dim_of`),
    an entity type's taken by that type alone (`_taken`).
    """
    found: dict[str, dict[str, str]] = {}
    for c in n.components:
        if not _exported(c):
            continue
        columns = {name: _column_dtype(c.static[name]) for name in _custom_names(c)}
        if columns:
            found[c.name] = columns
    return found


def _clashing_columns(n: pypsa.Network) -> frozenset[tuple[str, str]]:
    """`(type, column)` of `_custom_columns` whose attribute is declared over other dims.

    A custom column is an attribute over its type's dim, and one name is one
    spec: so it clashes with one the registry
    declares otherwise, and with a custom column of the same name on a type of
    another dim (`Bus.country` beside `Generator.country`). A property of the
    same name is its subject's column and does not clash.

    Notes
    -----
    - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)

    """
    declared = _registry_schema()
    custom = _custom_columns(n)
    dims: dict[str, set[str]] = {}
    for ctype, columns in custom.items():
        for name in columns:
            dims.setdefault(name, set()).add(_dim_of(ctype))
    return frozenset(
        (ctype, name)
        for ctype, columns in custom.items()
        for name in columns
        if len(dims[name]) > 1
        or (
            (spec := declared.attributes.get(name)) is not None
            and spec.dims != _static_dims(name, _dim_of(ctype))
        )
    )


def _column_dtype(values: pd.Series) -> str:
    """Return the mathspec dtype to declare a custom static column with: `str` unless it is bool, int or float."""
    if pd.api.types.is_bool_dtype(values):
        return "bool"
    if pd.api.types.is_integer_dtype(values):
        return "int"
    if pd.api.types.is_float_dtype(values):
        return "float"
    return "str"


def _weightings(
    record: RecordLike, dim: str, table: pd.DataFrame, con: DuckDBPyConnection
) -> pd.DataFrame:
    """`table`, PyPSA's weightings of `dim` by label, with each one `record` stores in place.

    A label reached by no row keeps `table`'s value.

    Notes
    -----
    - [the broadcast rule](https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule)

    """
    restored = table.copy()
    labels = {dim: Index.of(table.index.rename(dim), con).rel}
    overstated = Flags(frozenset({dim}), frozenset({dim}))
    for attribute, weighting in _WEIGHTINGS.items():
        frame = record.attributes.get(attribute)
        if weighting.dim != dim or frame is None:
            continue
        rows = settle(frame, labels, overstated, filter=False)
        rows = rows.project(col(dim), col("value")).df()
        values = rows.set_index(dim)["value"].reindex(table.index)
        restored[weighting.column] = values.fillna(table[weighting.column])
    return restored


def _connection_rows(c: pypsa.Components) -> pd.DataFrame:
    """One type's connections as `(name, bus, role)` rows, from its port columns.

    Undoes PyPSA's positional encoding: `c.ports` is the authoritative port
    list (`["0", "1"]` plus `additional_ports`), so the port count comes from
    the network rather than a guessed bound. `role` is the row's one
    property; every per-port input, `delay` included, is long rows of
    `attributes/` keyed by the connection.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

    """
    static = c.static
    frames = []
    for port in c.ports:
        column = _port_attribute("bus", port, c.defaults.index)
        if column not in static.columns:
            continue
        attached = static[static[column].astype(str) != ""]
        if attached.empty:
            continue
        rows = attached[[column]].rename(columns={column: "bus"})
        rows = rows.reset_index().rename(columns={"name": "entity"})
        rows["role"] = _port_role(port)
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=["entity", "bus", "role"])
    return pd.concat(frames, ignore_index=True)


def _from_wkt(values: pd.Series, crs: Any) -> pd.Series:
    """A WKT column as the GeoSeries PyPSA keeps, in `crs`."""
    import geopandas as gpd

    return gpd.GeoSeries.from_wkt(values, crs=crs)


def _drop_default_rows(rows: pd.DataFrame, default: Any) -> pd.DataFrame:
    """Remove scalar rows that just repeat the attribute's default value.

    Mirrors `pypsa.components.array._drop_default_rows`.
    """
    if pd.isnull(default):
        return rows[rows["value"].notna()]
    return rows[rows["value"] != default]


def _scalar_rows(c: pypsa.Components, attr: str, exclude: pd.Index) -> pd.DataFrame:
    """Build long rows from the static scalars of `attr`, less their NULL `snapshot`.

    Mirrors `pypsa.components.array.Components._scalar_rows`, with its
    columns in long order (`entity, scenario, value`) and
    built in one constructor rather than by `reset_index` and column inserts:
    it runs once per attribute and type, where pandas' per-call overhead
    dominated an export.
    """
    static = c.static[attr]
    if attr in _geometries(c.name):
        static = static.to_wkt()
    if len(exclude):
        static = static[~static.index.isin(exclude)]
    index = static.index
    stochastic = isinstance(index, pd.MultiIndex)
    return pd.DataFrame(
        {
            "entity": index.get_level_values("name") if stochastic else index,
            "scenario": index.get_level_values(SCENARIO) if stochastic else np.nan,
            "value": static.array,
        }
    )


def _shared_across_scenarios(rows: pd.DataFrame) -> pd.DataFrame:
    """`rows` with an entity's per-scenario repeats of one value folded to a NULL scenario.

    A stochastic network repeats its static frame per scenario; a value the
    same in every scenario is one row broadcasting over `scenario`, which an
    attribute not addressed by it needs, while one differing keeps its rows.

    Notes
    -----
    - [the broadcast rule](https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule)

    """
    if rows["scenario"].isna().all():
        return rows
    shared = rows.groupby("entity")["value"].transform(
        lambda values: values.nunique(dropna=False) == 1
    )
    folded = rows[shared.astype(bool)].drop_duplicates("entity").assign(scenario=None)
    return pd.concat([folded, rows[~shared.astype(bool)]], ignore_index=True)


@dataclass(frozen=True)
class _NetworkSource:
    """A `pypsa.Network` presented as a `Record`.

    Every `LazyRelations` here is built from key sets read off the network and
    its registry, so nothing is unpivoted until a key is looked up.

    Notes
    -----
    - [the record format](https://energy-models.github.io/datarecord/design/storage/format/)

    """

    n: pypsa.Network
    #: The connection the wide->long melt runs on.
    con: DuckDBPyConnection

    @cached_property
    def schema(self) -> RecordSchema:
        """The layer's schema: the registry-derived schema, widened to this network.

        The dims and the registry's attributes are the same for
        every network - `_registry_schema` derives them once from PyPSA's whole
        component registry. What varies is `meta`, which carries this network's
        own attributes and `n.meta` for read-back and, per entity type, the
        names of its custom static columns (`_custom_columns`), each declared
        over its type's dim, and, where the network has investment periods,
        the properties of `snapshot` (`_snapshot_properties`).

        Notes
        -----
        - [the schema](https://energy-models.github.io/datarecord/design/model/schema/)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

        """
        base = _registry_schema()
        dimensions = dict(base.dimensions)
        properties = dict(base.properties)
        if PERIOD in self._snapshots:
            properties[SNAPSHOT] = _snapshot_properties(self._snapshots)
            if TIMESTEP in self._snapshots:
                dimensions[SNAPSHOT] = dimensions[SNAPSHOT].model_copy(
                    update={"dtype": "int"}
                )
        attributes = dict(base.attributes)
        custom = _custom_columns(self.n)
        for ctype, columns in custom.items():
            dim = _dim_of(ctype)
            for name, dtype in columns.items():
                attributes.setdefault(
                    name,
                    AttributeSpec(
                        dtype=dtype,  # type: ignore[arg-type]
                        dims=_static_dims(name, dim),
                    ),
                )
        return RecordSchema(
            **{
                **dict(base),
                "dimensions": dimensions,
                "properties": properties,
                "attributes": attributes,
                "meta": {
                    "format": "pypsa-parquet",
                    "attributes": _collect_network_attributes(self.n, skip={"srid"}),
                    _CUSTOM: {
                        ctype: sorted(columns)
                        for ctype, columns in custom.items()
                        if _dim_of(ctype) == ENTITY
                    },
                    "meta": dict(self.n.meta),
                    "crs": self.n.crs.to_string(),
                },
            }
        )

    @cached_property
    def _snapshots(self) -> pd.DataFrame:
        """`n.snapshots` as the `snapshot` axis with its properties (`_snapshot_frame`)."""
        return _snapshot_frame(self.n.snapshots)

    @cached_property
    def _snapshot_labels(self) -> pd.Index:
        """The labels of `_snapshots`, in PyPSA's order."""
        return pd.Index(self._snapshots[SNAPSHOT], name=SNAPSHOT)

    @property
    def dims(self) -> LazyRelations:
        """Every axis this network holds, `entity`, `bus` and `carrier` among them, as keys.

        Each labelled axis is its keys; its properties are in `properties`.
        `timestep` holds the timesteps a multi-period network's positional
        snapshots name, and the multi-dim `connection`
        every type's attachments, single-attachment ones included: a
        Generator's one `bus` is as much a connection as a Link's `bus0`.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/storage/format/#where-a-value-lives)

        """
        axes: dict[str, Callable[[], pd.DataFrame]] = {
            SNAPSHOT: lambda: self._snapshots[[SNAPSHOT]],
            TIMESTEP: lambda: self._timesteps,
            PERIOD: lambda: self.n.investment_periods.to_frame(index=False),
            SCENARIO: lambda: self.n.scenario_weightings.index.to_frame(index=False),
            BUS: lambda: self._buses[[BUS]],
            CARRIER: lambda: self._carriers[[CARRIER]],
            SUB_NETWORK: lambda: self._sub_networks[[SUB_NETWORK]],
        }
        present = tuple(
            d
            for d in (SNAPSHOT, TIMESTEP, PERIOD, SCENARIO, BUS, CARRIER, SUB_NETWORK)
            if len(axes[d]()) > 0
        )
        axes[ENTITY] = lambda: self._entity_frame()[[ENTITY]]

        def axis(dim: str) -> DuckDBPyRelation:
            if dim == CONNECTION:
                return self._connection_frame().project(col(CONNECTION))
            return self.con.from_df(axes[dim]())

        return LazyRelations((*present, ENTITY, CONNECTION), axis)

    @cached_property
    def _timesteps(self) -> pd.DataFrame:
        """The `timestep` axis: the distinct timesteps `snapshot.timestep` names, in order."""
        if TIMESTEP not in self._snapshots:
            return pd.DataFrame(columns=[TIMESTEP])
        return self._snapshots[[TIMESTEP]].drop_duplicates(ignore_index=True)

    @cached_property
    def _buses(self) -> pd.DataFrame:
        """`n.buses` as the `bus` axis: each label with its properties, in PyPSA's order.

        Notes
        -----
        - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)

        """
        return self._labels("Bus")

    @cached_property
    def _sub_networks(self) -> pd.DataFrame:
        """`n.sub_networks` as the `sub_network` axis, each label with its carrier."""
        return self._labels("SubNetwork")

    @cached_property
    def _carriers(self) -> pd.DataFrame:
        """`n.carriers` as the `carrier` axis, then every carrier named but not listed.

        A component or bus may name a carrier `n.carriers` lacks, which PyPSA
        tolerates; a property that refers to `carrier` holds a live label, so
        it is added, in the order first named, its properties NULL.

        Notes
        -----
        - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)

        """
        listed = self._labels("Carrier")
        named = [
            frame[CARRIER]
            for frame in (self._buses, self._sub_networks, self._entity_frame())
            if CARRIER in frame
        ]
        unlisted = (
            pd.concat(named).dropna().drop_duplicates()
            if named
            else pd.Series([], dtype=object)
        )
        unlisted = unlisted[~unlisted.isin(listed[CARRIER])]
        added = pd.DataFrame({CARRIER: unlisted.to_numpy()})
        return pd.concat([listed, added], ignore_index=True)

    def _labels(self, ctype: str) -> pd.DataFrame:
        """One `_AXIS_TYPES` type's members as labels of its dim, with their properties."""
        dim = _AXIS_TYPES[ctype]
        static = self.n.c[ctype].static
        columns = [p for p in _PROPERTIES[dim] if p in static.columns]
        frame = static[columns].rename_axis(index={"name": dim}).reset_index()
        return self._tagged(_none_for_empty(frame), dim)

    def _entity_frame(self) -> pd.DataFrame:
        """Every exported type's members and their properties.

        One row per entity, typed by the `ctype` column; a property
        one type lacks is NULL on that type's rows.

        Notes
        -----
        - [a label is unique within its dim](https://energy-models.github.io/datarecord/design/storage/format/#a-label-is-unique-within-its-dim)
        - [where a value lives](https://energy-models.github.io/datarecord/design/storage/format/#where-a-value-lives)

        """
        frames = [
            self._static_frame(c.name).assign(**{CTYPE: c.name})
            for c in self.n.components
            if _as_entities(c)
        ]
        if not frames:
            return pd.DataFrame(columns=[ENTITY, CTYPE])
        merged = _none_for_empty(pd.concat(frames, ignore_index=True))
        lead = [ENTITY, CTYPE]
        return merged[[*lead, *(c for c in merged.columns if c not in lead)]]

    def _connection_frame(self) -> DuckDBPyRelation:
        """Every type's connections in one frame, `(connection, role)`.

        Notes
        -----
        - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

        """
        types = [c.name for c in self.n.components if _as_entities(c) and c.ports]
        frames = [_connection_rows(self.n.c[ctype]) for ctype in types]
        rows = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["entity", "bus", "role"])
        )
        # A stochastic network repeats its connections per scenario, which
        # collapse to one row: an attachment exists or it does not, so nothing
        # scopes it per value of an axis (`_tagged` says the same of members).
        if SCENARIO in rows.columns:
            rows = rows.drop(columns=[SCENARIO]).drop_duplicates(
                subset=["entity", "bus"]
            )
        flat = self.con.from_df(rows)
        return flat.project(
            _connection_struct(col(ENTITY), col(BUS)), star(exclude=[ENTITY, BUS])
        )

    def _as_long_rel(
        self, c: pypsa.Components, attr: str, *, drop_defaults: bool = False
    ) -> DuckDBPyRelation | None:
        """`attr` as a long relation `(entity, snapshot, scenario, value)`.

        `None` where the attribute contributes no rows on this network, so a
        caller unions only the types that do rather than empty relations.

        The wide `snapshots x entities` (or
        `snapshots x (scenario, entity)`) dynamic frame is melted by `UNPIVOT`
        rather than `pandas.stack`, so the relation stays a lazy plan
        `write_layer` writes without a copy. Static scalars land as null-snapshot
        rows exactly as `_scalar_rows` builds them; `drop_defaults` omits a scalar
        that only repeats the attribute's default.

        Parameters
        ----------
        drop_defaults
            Omit static scalar rows whose value equals the attribute's default.

        Notes
        -----
        - [wide and long rows](https://energy-models.github.io/datarecord/design/model/attributes/#wide-and-long-rows)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

        """
        con = self.con
        varying = attr in c.dynamic or (
            attr in c.defaults.index and bool(c.defaults.at[attr, "varying"])
        )
        if (
            not varying
            and not _is_output(c.defaults, attr)
            and attr not in c.static.columns
        ):
            msg = f"'{attr}' of '{c.name}' components has no long representation."
            raise AttributeError(msg)

        parts: list[DuckDBPyRelation] = []
        excluded: pd.Index = pd.Index([])
        if varying:
            melted = self._melt_dynamic(con, c, attr)
            if melted is not None:
                parts.append(melted)
            excluded = c.dynamic[attr].columns
        static_backed = attr in c.defaults.index and bool(c.defaults.at[attr, "static"])
        if not varying or static_backed:
            scalars = self._scalar_rel(
                con, c, attr, excluded, drop_defaults=drop_defaults
            )
            if scalars is not None:
                parts.append(scalars)
        if not parts:
            return None
        rel = parts[0]
        for other in parts[1:]:
            rel = rel.union(other)
        return rel

    def _melt_dynamic(
        self, con: DuckDBPyConnection, c: pypsa.Components, attr: str
    ) -> DuckDBPyRelation | None:
        """Melt one varying attribute's wide dynamic frame to long rows.

        `None` where the frame has no entity columns - a degenerate
        `UNPIVOT ... IN ()` and an empty relation the caller would only union
        away.

        The frame is transposed before the melt, so the rows come out entity by
        entity: a parquet row group then holds a handful of entity names, which
        its dictionary encodes, where snapshot-major rows put every name in
        each row group and leave them unencoded.

        The snapshots are melted by position and looked up in a typed list of
        the labels, so a label never becomes an SQL identifier and keeps its
        dtype.
        """
        wide = c.dynamic[attr]
        stochastic = isinstance(wide.columns, pd.MultiIndex)
        # One flat value row per melted column: `scenario\x1fentity` where a
        # scenario level exists, split back out after the UNPIVOT. `reset_index`
        # returns a fresh frame, so relabelling it never mutates the dynamic
        # frame PyPSA owns - no defensive copy needed.
        valcols = (
            [f"{s}\x1f{e}" for s, e in wide.columns]
            if stochastic
            else list(wide.columns)
        )
        if not valcols or wide.empty:
            return None
        positional = [f"s{i}" for i in range(len(wide.index))]
        w = (
            wide.T.set_axis(positional, axis=1)
            .set_axis(pd.Index(valcols, name="colname"))
            .reset_index()
        )
        # `from_df` captures the frame into the relation's own plan, so a later
        # call reusing the frame variable cannot rebind this one (unlike a named
        # registered view).
        wide_rel = con.from_df(w)  # noqa: F841 - bound by name for the query below
        inlist = ", ".join(f"{p} AS {lit(str(i))}" for i, p in enumerate(positional))
        # Two steps rather than one nested query: the local-relation replacement
        # scan does not compose through a subquery, so melt first, then project.
        melted = con.sql(  # noqa: F841 - bound by name for the query below
            "SELECT colname, CAST(pos AS INTEGER) AS pos, value FROM wide_rel "
            f"UNPIVOT INCLUDE NULLS (value FOR pos IN ({inlist}))"
        )
        if stochastic:
            entity = "split_part(colname, chr(31), 2)"
            scenario = "split_part(colname, chr(31), 1)"
        else:
            entity, scenario = "colname", "CAST(NULL AS VARCHAR)"
        labels = lit(self._snapshot_labels.tolist())
        return con.sql(
            f"SELECT {entity} AS entity, ({labels})[pos + 1] AS {SNAPSHOT}, "
            f"{scenario} AS scenario, value FROM melted"
        )

    def _scalar_rel(
        self,
        con: DuckDBPyConnection,
        c: pypsa.Components,
        attr: str,
        exclude: pd.Index,
        *,
        drop_defaults: bool = False,
    ) -> DuckDBPyRelation | None:
        """Null-snapshot long rows from `attr`'s static scalars (`_scalar_rows`).

        The `snapshot` is an untyped NULL, so a union with the melted rows
        takes their type rather than widening them.

        `None` where no scalar rows remain - every entity excluded, an empty
        static frame, or (with `drop_defaults`) every scalar at its default - a
        relation the caller would only union away. The default drop happens here
        in pandas, where the emptiness is decidable without executing a relation.

        The reshape stays in pandas: it is one `reset_index` on a Series, cheaper
        than selecting a single column out of a wide registered relation per
        attribute (measured), and the frame lands in DuckDB the same way the melt
        does.
        """
        static = _shared_across_scenarios(_scalar_rows(c, attr, exclude))
        if drop_defaults and attr in c.defaults.index:
            static = _drop_default_rows(static, c.defaults.at[attr, "default"])
        if static.empty:
            return None
        return con.sql("SELECT entity, NULL AS snapshot, scenario, value FROM static")

    @property
    def attributes(self) -> LazyRelations:
        """Every input that is no property, the weightings of each axis the network holds among them, and every result with rows."""
        names = self._attribute_names
        results = self._result_rels

        def frame(attr: str) -> DuckDBPyRelation:
            if attr in results:
                return results[attr]
            return self._long_frame(attr, names[attr])

        return LazyRelations((*names, *results), frame)

    @cached_property
    def _attribute_names(self) -> dict[str, list[str]]:
        """Input attribute -> the exported types carrying it.

        Read once per source: `write_layer` asks for `attributes` once per key,
        and recomputing it walks every type's registry each time. The source is
        a view taken for one write, so the network does not change under it.
        """
        names: dict[str, list[str]] = {}
        for c in self.n.components:
            if not _exported(c):
                continue
            for attr in self._record_attributes(c):
                names.setdefault(attr, []).append(c.name)
        for attribute, weighting in _WEIGHTINGS.items():
            if len(self._weighting_table(weighting.dim)):
                names[attribute] = []
        return names

    def _weighting_table(self, dim: str) -> pd.DataFrame:
        """PyPSA's weighting frame of `dim` by label, empty where the network lacks the axis."""
        if dim == SNAPSHOT:
            return self.n.snapshot_weightings.set_axis(self._snapshot_labels)
        if dim == PERIOD:
            return self.n.investment_period_weightings
        return self.n.scenario_weightings

    def _weighting_frame(self, attribute: str) -> DuckDBPyRelation:
        """One axis weighting as long rows over its axis."""
        weighting = _WEIGHTINGS[attribute]
        table = self._weighting_table(weighting.dim)
        weights = (  # noqa: F841 - bound by name for the query below
            table[weighting.column].rename("value").reset_index()
        )
        return self.con.sql(
            f"SELECT * EXCLUDE (value), {lit(attribute)} AS attribute, "
            "NULL AS breakpoint, CAST(value AS DOUBLE) AS value FROM weights"
        )

    @cached_property
    def _output_names(self) -> dict[str, list[str]]:
        """Result attribute -> the exported types declaring it, as the schema does."""
        names: dict[str, list[str]] = {}
        for ctype, results in self._results.items():
            for attr in sorted(set(results.values())):
                names.setdefault(attr, []).append(ctype)
        return names

    #: Whether a result's static rows still at their default are left out.
    _DROP_RESULT_DEFAULTS: ClassVar[bool] = True

    @cached_property
    def _result_rels(self) -> dict[str, DuckDBPyRelation]:
        """Result attribute -> its long rows across types, where any type has rows.

        Built when listed, unlike an input, since a result the network holds
        no rows of has no file: a network never solved writes none.

        Notes
        -----
        - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)

        """
        return {
            attr: rel
            for attr, ctypes in self._output_names.items()
            if (rel := self._output_rel(attr, ctypes)) is not None
        }

    @cached_property
    def _results(self) -> dict[str, dict[str, str]]:
        """Exported type -> its `_result_attributes`, read once per source.

        Every result's frame needs its types' mapping, and deriving it reads
        the type's whole registry.
        """
        inputs = _input_names()
        return {
            c.name: _result_attributes(c, inputs)
            for c in self.n.components
            if _exported(c)
        }

    @property
    def properties(self) -> LazyRelations:
        """The keys and property columns of the entities, buses, carriers, sub-networks, connections and multi-period snapshots."""
        frames = {
            ENTITY: lambda: self.con.from_df(self._entity_frame()),
            BUS: lambda: self.con.from_df(self._buses),
            CARRIER: lambda: self.con.from_df(self._carriers),
            SUB_NETWORK: lambda: self.con.from_df(self._sub_networks),
            CONNECTION: self._connection_frame,
            SNAPSHOT: lambda: self.con.from_df(self._snapshots),
        }
        labels = {
            BUS: self._buses,
            CARRIER: self._carriers,
            SUB_NETWORK: self._sub_networks,
        }
        labelled = [d for d, frame in labels.items() if len(frame)]
        timed = [SNAPSHOT] if PERIOD in self._snapshots else []
        subjects = (ENTITY, *labelled, CONNECTION, *timed)

        def frame(subject: str) -> DuckDBPyRelation:
            rel = frames[subject]()
            declared = {subject, *self.schema.properties[subject]}
            return rel.project(*(col(c) for c in rel.columns if c in declared))

        return LazyRelations(subjects, frame)

    def flags(self, **categories: str | Sequence[str]) -> dict[str, Flags]:
        """Never consulted: `write_layer` persists frames, not flags.

        A network-backed source exists to be written, and the write path reads
        only the frame mappings. Answering properly is easy - `c.static` and
        `c.dynamic` are PyPSA's own split on the snapshot axis, so it needs no
        scan - but an implementation no caller reaches is one no test pins, so
        the honest answer is the empty one.

        Notes
        -----
        - [writing a whole record](https://energy-models.github.io/datarecord/design/use/writing/)

        """
        return {}

    # -- key sets -----------------------------------------------------------

    def _record_attributes(self, c: pypsa.Components) -> list[str]:
        """`c`'s input attributes, in the record's vocabulary: every input but a property.

        Each goes to `attributes/` whatever it varies over. Per-port attributes
        collapse to their stem (`efficiency2` -> `efficiency`), a record keying
        them by bus rather than position; `bus` is the connection itself, and
        outputs are `_result_rels`. A custom column the registry has no
        entry for (`ac_dc_meshed`'s `Bus.country`) is an attribute too.

        Notes
        -----
        - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
        - [where a value lives](https://energy-models.github.io/datarecord/design/storage/format/#where-a-value-lives)

        """
        defaults = c.defaults
        skipped = {"name", *_output_attributes(c), *_PROPERTIES[_dim_of(c.name)]}
        per_port = self._port_stems(c)
        stems: list[str] = []
        for attr in [*defaults.index, *_custom_names(c)]:
            if attr in skipped:
                continue
            if attr in defaults.index and not (
                defaults.at[attr, "varying"] or attr in c.static.columns
            ):
                continue
            stem = per_port.get(attr, attr)
            if stem != BUS and stem not in stems:
                stems.append(stem)
        return stems

    def _port_stems(self, c: pypsa.Components) -> dict[str, str]:
        """Per-port attribute name -> its stem, for this type's actual ports."""
        return _port_stems(c)

    # -- frames -------------------------------------------------------------

    def _static_frame(self, ctype: str) -> pd.DataFrame:
        """Members of one type with the properties of `entity` it has columns for.

        Notes
        -----
        - [the Record protocol](https://energy-models.github.io/datarecord/design/use/reading/)

        """
        c = self.n.c[ctype]
        columns = [p for p in _PROPERTIES[ENTITY] if p in c.static.columns]
        frame = c.static[columns].reset_index().rename(columns={"name": "entity"})
        return self._tagged(frame, ENTITY)

    @staticmethod
    def _tagged(frame: pd.DataFrame, dim: str) -> pd.DataFrame:
        """A `dims/` frame, one row per label.

        No `scenario`: an entity exists or it does not, so nothing scopes
        membership per value of an axis. A stochastic network repeats its
        static frame per scenario, which is a *value* being per-scenario
        rather than the component - so the repeats collapse to one row here
        rather than reaching the record as duplicate entities.

        Notes
        -----
        - [deletion](https://energy-models.github.io/datarecord/design/layers/removal/)
        - [the record format](https://energy-models.github.io/datarecord/design/storage/format/)

        """
        if SCENARIO in frame.columns:
            frame = frame.drop(columns=[SCENARIO]).drop_duplicates(subset=[dim])
        return frame

    def _tagged_long_rel(
        self,
        c: pypsa.Components,
        attribute: str,
        source_attr: str,
        *,
        drop_defaults: bool,
    ) -> DuckDBPyRelation | None:
        """`source_attr`'s long relation, tagged with `attribute` and a NULL breakpoint.

        `attribute` is the record-facing name and `source_attr` PyPSA's - the two
        differ for a per-port attribute, whose ports (`p0`, `p1`, ...) each
        contribute under the one stem, and for a renamed result (`p_internal`).
        A NULL `breakpoint` marks the value a scalar (https://energy-models.github.io/datarecord/design/storage/format/#the-long-schema).
        An `_AXIS_TYPES` type's rows are keyed by its dim in place of
        `entity`. `None` where `source_attr` contributes no rows.
        """
        rel = self._as_long_rel(c, source_attr, drop_defaults=drop_defaults)
        if rel is None:
            return None
        dim = _dim_of(c.name)
        return rel.select(
            *(
                col(name).alias(dim) if name == ENTITY else col(name)
                for name in rel.columns
            ),
            lit(attribute).alias("attribute"),
            lit(None).alias("breakpoint"),
        )

    def _long_frame(self, attribute: str, ctypes: list[str]) -> DuckDBPyRelation:
        """One attribute's long rows, across every type that has it.

        A per-connection attribute contributes rows carrying the `connection`
        struct each port is, in place of `entity`; a component-level one carries
        `entity` alone. The rows are a DuckDB relation melted by `_as_long_rel`, so the
        frame stays a lazy plan `write_layer` writes without a copy.

        Notes
        -----
        - [the Record protocol](https://energy-models.github.io/datarecord/design/use/reading/)
        - [wide and long rows](https://energy-models.github.io/datarecord/design/model/attributes/#wide-and-long-rows)

        """
        # This attribute's own columns, from its spec: one file is one attribute,
        # so it hands over no dim it is not addressed by (https://energy-models.github.io/datarecord/design/storage/format/#the-long-schema).
        if attribute in _WEIGHTINGS:
            return self._weighting_frame(attribute)
        columns = list(self.schema.long_columns_for(attribute))
        rels = []
        for ctype in ctypes:
            c = self.n.c[ctype]
            if attribute in self._port_stems(c).values():
                rels.extend(self._per_port_long_rels(c, attribute))
            elif attribute in c.defaults.index or attribute in c.static.columns:
                rel = self._tagged_long_rel(c, attribute, attribute, drop_defaults=True)
                if rel is not None:
                    rels.append(rel)
        return self._union_select(rels, columns)

    def _per_port_long_rels(
        self, c: pypsa.Components, stem: str
    ) -> list[DuckDBPyRelation]:
        """A per-connection attribute's long relations, one per port.

        The inverse of the positional collapse: `efficiency`/`efficiency2` become
        one attribute whose rows carry the connection each port is.

        Notes
        -----
        - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

        """
        rels = []
        for column, (of, port) in _port_names(c).items():
            if of != stem:
                continue
            rel = self._port_long_rel(c, stem, column, port, drop_defaults=False)
            if rel is not None:
                rels.append(rel)
        return rels

    def _port_long_rel(
        self,
        c: pypsa.Components,
        attribute: str,
        column: str,
        port: str,
        *,
        drop_defaults: bool,
    ) -> DuckDBPyRelation | None:
        """`column`'s long relation at `port`, tagged `attribute`, keyed by connection.

        The `connection` struct packs the entity with the bus joined from the
        port's static bus column rather than mapped in pandas, so the rows stay a
        relation. `None` where the port has no bus column or `column`
        contributes no rows.

        Notes
        -----
        - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

        """
        con = self.con
        static = c.static
        bus_column = _port_attribute(BUS, port, c.defaults.index)
        if bus_column not in static.columns:
            return None
        # `static` is `(scenario, name)`-indexed on a stochastic network; a
        # component's bus does not vary by scenario, so the name level
        # deduplicates to one bus per component.
        buses = static[bus_column]
        if isinstance(buses.index, pd.MultiIndex):
            buses = buses.droplevel(SCENARIO)
            buses = buses[~buses.index.duplicated()]
        buses = buses[buses.astype(str) != ""]
        bus_df = buses.rename("bus").rename_axis("entity").reset_index()  # noqa: F841
        bus_rel = con.sql(  # noqa: F841 - bound by name for the join below
            "SELECT CAST(entity AS VARCHAR) entity, bus FROM bus_df"
        )
        v = self._tagged_long_rel(c, attribute, column, drop_defaults=drop_defaults)
        if v is None:
            return None
        return (
            v.set_alias("v")
            .join(bus_rel.set_alias("b"), col("v", ENTITY) == col("b", ENTITY))
            .project(
                *(col("v", name) for name in v.columns if name != ENTITY),
                _connection_struct(col("v", ENTITY), col("b", BUS)),
            )
        )

    def _output_rel(self, attribute: str, ctypes: list[str]) -> DuckDBPyRelation | None:
        """One result attribute's long rows across types, `None` where no type contributes."""
        rels = self._output_rels(attribute, ctypes)
        return self._union_rel(rels, list(self.schema.long_columns_for(attribute)))

    def _output_rels(self, attribute: str, ctypes: list[str]) -> list[DuckDBPyRelation]:
        """One result attribute's per-type long relations, empties dropped.

        Each type contributes the PyPSA outputs `_result_attributes` lands in
        `attribute`: every port's `p0`, `p1`, ... under `p` with its bus, or a
        single entity-level one, possibly renamed (`Link.p` as `p_internal`).
        """
        rels = []
        for ctype in ctypes:
            c = self.n.c[ctype]
            ports = _port_names(c)
            for column, name in sorted(self._results[ctype].items()):
                if name != attribute:
                    continue
                drop = self._DROP_RESULT_DEFAULTS
                try:
                    rel = (
                        self._port_long_rel(
                            c, name, column, ports[column][1], drop_defaults=drop
                        )
                        if column in ports
                        else self._tagged_long_rel(c, name, column, drop_defaults=drop)
                    )
                except (AttributeError, KeyError):
                    # Not every registry output has a container on every network.
                    continue
                if rel is None:
                    continue
                rels.append(rel)
        return rels

    def _union_rel(
        self, rels: list[DuckDBPyRelation], columns: list[str]
    ) -> DuckDBPyRelation | None:
        """Union `rels` and project `columns`; `None` where none contribute."""
        if not rels:
            return None
        rel = rels[0]
        for other in rels[1:]:
            rel = rel.union(other)
        return rel.select(*(col(name) for name in columns))

    def _union_select(
        self, rels: list[DuckDBPyRelation], columns: list[str]
    ) -> DuckDBPyRelation:
        """Union `rels` and project them to `columns`.

        An empty list is an empty relation with the right columns, each of its
        declared type: `write_layer` checks a multi-dim column is a struct before it
        casts.
        """
        rel = self._union_rel(rels, columns)
        if rel is not None:
            return rel
        schema = _registry_schema()
        return empty_relation(
            self.con, **{c: schema.column_type(c) or sqltypes.VARCHAR for c in columns}
        )


@dataclass(frozen=True)
class _ResultsSource(_NetworkSource):
    """A solved `pypsa.Network`'s results, as a thin layer over `base`.

    Its result attributes and this solve's SubNetworks, the `sub_network`
    axis restated whole, so a re-solve numbering them differently, or finding
    none, leaves none of `base`'s behind. A result's rows at their default are written too, and
    `base`'s result rows it does not write again as NULL, each standing over
    `base`'s value.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)

    """

    #: The record the network was built from.
    base: RecordLike

    _DROP_RESULT_DEFAULTS: ClassVar[bool] = False

    @cached_property
    def schema(self) -> RecordSchema:
        """`base`'s schema: a results layer declares nothing the record does not."""
        return self.base.schema

    @property
    def dims(self) -> LazyRelations:
        """The `sub_network` axis: this solve's SubNetworks, as `_restates` says."""
        return LazyRelations(
            self._restates, lambda _: self._sub_network_rel().project(col(SUB_NETWORK))
        )

    @property
    def properties(self) -> LazyRelations:
        """This solve's SubNetworks' properties."""
        return LazyRelations(self._restates, lambda _: self._sub_network_rel())

    @cached_property
    def _restates(self) -> tuple[str, ...]:
        """The axes this layer restates: `sub_network` where the solve or `base` has one.

        A solve without SubNetworks over a `base` with a `sub_network` axis
        restates it empty, so the record lists none of `base`'s.
        """
        held = len(self._sub_networks) > 0 or SUB_NETWORK in self.base.dims
        return (SUB_NETWORK,) if held else ()

    def _sub_network_rel(self) -> DuckDBPyRelation:
        """`_sub_networks` with its columns typed as the schema declares them, not as an empty frame infers them."""
        rel = self.con.from_df(self._sub_networks)
        return cast_declared(self.schema, rel, SUB_NETWORK)

    def _dead_sub_networks(self) -> DuckDBPyRelation | None:
        """`base`'s SubNetworks this solve does not number, aliased `d`; `None` where `base` has no axis."""
        axis = self.base.dims.get(SUB_NETWORK)
        if axis is None:
            return None
        live = self._sub_network_rel().set_alias("n")
        return (
            axis.set_alias("a")
            .join(live, col("a", SUB_NETWORK) == col("n", SUB_NETWORK), how="anti")
            .set_alias("d")
        )

    @property
    def attributes(self) -> LazyRelations:
        """Every result with rows, `base`'s included."""
        results = self._restated
        return LazyRelations(tuple(results), results.__getitem__)

    @cached_property
    def _restated(self) -> dict[str, DuckDBPyRelation]:
        """This solve's results, then each row of `base`'s it leaves out, as NULL.

        A row the solve does not write would otherwise keep `base`'s value by
        inheritance, though the network holds none there. Rows of a
        SubNetwork of `base` this solve does not number are left out with it.
        """
        written = self._result_rels
        outputs = _registry_schema().attributes.keys() - _input_names()
        dead = self._dead_sub_networks()
        restated = dict(written)
        for attr in sorted(outputs & set(self.base.attributes)):
            columns = self.schema.long_columns_for(attr)
            keys = [c for c in columns if c != "value"]
            stale = self.base.attributes[attr].set_alias("b")
            if attr in written:
                stale = stale.join(
                    written[attr].set_alias("w"), null_safe("b", "w", keys), how="anti"
                )
            if dead is not None and SUB_NETWORK in keys:
                stale = stale.set_alias("s").join(
                    dead, col("s", SUB_NETWORK) == col("d", SUB_NETWORK), how="anti"
                )
            null = lit(None).cast(self.schema.duck_type(self.schema.attributes[attr]))
            rows = stale.project(*map(col, keys), null.alias("value"))
            restated[attr] = rows if attr not in written else written[attr].union(rows)
        return restated
