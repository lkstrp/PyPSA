# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Network-to-record export sources of the PyPSA tool."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, partial, reduce
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
from datarecord.duck import empty_relation, ex_all, fn, join_on, null_safe
from datarecord.protocols import Flags, LazyRelations, RecordLike
from datarecord.schema.schema import Schema as RecordSchema
from datarecord.schema.schema import cast_declared, packed, unpacked
from datarecord.schema.specs import (
    AttributeSpec,
    Dimension,
    PropertySpec,
)
from datarecord.select import Selection
from datarecord.settle import settle, uncovered
from duckdb import CoalesceOperator as coalesce
from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import (
    DuckDBPyConnection,
    DuckDBPyRelation,
    Expression,
    sqltypes,
)
from duckdb import StarExpression as star

if TYPE_CHECKING:
    import pypsa
from pypsa.components.components import _UNREAD
from pypsa.definitions.structures import LazyDict
from pypsa.network.io._common import (
    _collect_network_attributes,
)
from pypsa.network.io.datarecord.schema import (
    _AXIS_TYPES,
    _CATALOG_TYPES,
    _CUSTOM,
    _DERIVED,
    _GEOMETRY_DESCRIPTION,
    _KEYS,
    _SHAPE,
    _SHAPE_DIM,
    _SHAPE_PREFIX,
    _WEIGHTINGS,
    BUS,
    CARRIER,
    CONNECTION,
    CTYPE,
    ENTITY,
    GLOBAL_CONSTRAINT,
    PERIOD,
    PORT,
    SCENARIO,
    SNAPSHOT,
    SUB_NETWORK,
    TIMESTEP,
    _connection_struct,
    _dim_of,
    _input_names,
    _is_output,
    _output_attributes,
    _port_names,
    _port_stems,
    _registry_schema,
    _result_attributes,
    _static_dims,
    pypsa_attribute,
    record_input,
)
from pypsa.network.io.datarecord.shape import _multiperiod

logger = logging.getLogger(__name__)

#: The values in one batch of a melted series, about a parquet row group.
_MELT_BATCH = 122_880


def _own_static(c: pypsa.Components) -> pd.DataFrame:
    """`c.static` less the rows named as in PyPSA's standard type library, edited or not, its custom columns of dtype `object` cast to the dtype their values give (`_column_dtype`), an all-NULL one left NULL for the declared dtype to take."""
    library = c.standard_types
    static = c.static
    if library is not None:
        level = "name" if isinstance(static.index, pd.MultiIndex) else None
        static = static.drop(library.index, level=level, errors="ignore")
    typed = {
        name: _cast_column(static[name])
        for name in _custom_names(c)
        if static[name].dtype == object
    }
    return static.assign(**typed) if typed else static


def _own_names(c: pypsa.Components) -> pd.Index:
    """`c`'s member names less PyPSA's library, read from `_static_names` so an unread static stays unread."""
    names = c._static_names()
    library = c.standard_types
    return names if library is None else names.drop(library.index, errors="ignore")


def _exported(c: pypsa.Components, untouched: Collection[str] = ()) -> bool:
    """Whether `c` holds members beyond PyPSA's library (`_own_static`), which belong in a record.

    A type in `untouched` is answered from its names, its static frame unread.
    """
    if c.name in untouched:
        return len(_own_names(c)) > 0
    return not _own_static(c).empty


def untouched_types(loaded: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """The types that held no attribute loaded, so an export leaves them unread."""
    return frozenset(name for name, held in loaded.items() if not held)


def _as_entities(c: pypsa.Components, untouched: Collection[str] = ()) -> bool:
    """Whether `c`'s members are written as entities, not as labels of their own dim or as shapes."""
    return c.name not in {*_AXIS_TYPES, _SHAPE} and _exported(c, untouched)


def _names(index: pd.Index) -> pd.Index:
    """The `name` level of a static index, `(scenario, name)` on a stochastic network."""
    return index.get_level_values("name") if isinstance(index, pd.MultiIndex) else index


def _keyed_as_entity(rel: DuckDBPyRelation, dim: str) -> DuckDBPyRelation:
    """`rel` with its `dim` column named `entity`, the key PyPSA's frames are built on."""
    if dim == ENTITY:
        return rel
    return rel.project(
        *(col(c).alias(ENTITY) if c == dim else col(c) for c in rel.columns)
    )


#: The columns that name a label of another dim, PyPSA's `""` meaning none.
_REFERENCES = (CARRIER, BUS)


def _none_for_empty(frame: pd.DataFrame, dim: str) -> pd.DataFrame:
    """`frame` with an empty `carrier` or `bus`, PyPSA's none, as NULL, since it names no label.

    `dim` is the frame's own dim, whose column holds labels, not references.
    """
    named = [c for c in _REFERENCES if c in frame and c != dim]
    return frame.assign(**{c: frame[c].where(frame[c] != "", None) for c in named})


def _empty_for_none(rel: DuckDBPyRelation, dim: str) -> DuckDBPyRelation:
    """`rel` with a NULL `carrier` or `bus` as PyPSA's `""`, the inverse of `_none_for_empty`."""
    return rel.project(
        *(
            coalesce(col(c), lit("")).alias(c)
            if c in _REFERENCES and c != dim
            else col(c)
            for c in rel.columns
        )
    )


def _colliding_names(
    n: pypsa.Network, untouched: Collection[str] = ()
) -> frozenset[str]:
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
        if not _as_entities(c, untouched):
            continue
        members = _own_names(c) if c.name in untouched else _names(_own_static(c).index)
        for value in members:
            name = str(value)
            if seen.setdefault(name, c.name) != c.name:
                clashing.add(name)
    return frozenset(clashing)


def _custom_names(c: pypsa.Components) -> list[str]:
    """`c`'s static columns PyPSA's registry has no entry for, but `_DERIVED`."""
    derived = _DERIVED.get(c.name, frozenset())
    return [
        name
        for name in c.static.columns
        if name not in c.defaults.index and name not in derived
    ]


@dataclass(frozen=True)
class _Custom:
    """A network's custom static columns, as the schema declares them.

    A text column, one that as an attribute would hold one name over two
    dims, or any column of a `_CATALOG_TYPES` type, is a property of its
    type's dim, any other an attribute over it. A column takes its dtype
    from the types holding values in it, every type carrying it declared
    with that one dtype.
    """

    #: Exported type -> its custom columns and their dtypes.
    columns: dict[str, dict[str, str]]
    #: Dim -> its custom properties and their dtypes.
    properties: dict[str, dict[str, str]]
    #: The custom attributes, each over its type's dim.
    attributes: dict[str, AttributeSpec]

    @classmethod
    def of(
        cls, n: pypsa.Network, given: Mapping[str, dict[str, str]] | None = None
    ) -> _Custom:
        """Place `n`'s custom columns.

        A type in `given` is not read: its columns and dtypes are the ones given,
        each counted as holding values. A type whose column is all NULL takes the
        dtype of the types holding values, or of those holding text when none
        does; with neither, their one dtype, else `str`.

        Raises
        ------
        ValueError
            If the types of one dim holding values in a custom column carry it
            with different dtypes, property or attribute,
            a custom column is named like a shape's geometry (`shape_`) or a
            dim keying the record's rows (`_KEYS`), or a Shape carries one.

        """
        columns: dict[str, dict[str, str]] = {}
        nulls: dict[str, set[str]] = {}
        for c in n.components:
            if given is not None and c.name in given:
                if given[c.name]:
                    columns[c.name] = dict(given[c.name])
                continue
            if not (_exported(c) and (names := _custom_names(c))):
                continue
            geometric = [name for name in names if name.startswith(_SHAPE_PREFIX)]
            if c.name == _SHAPE or geometric:
                msg = (
                    f"The custom columns {geometric or names} of {c.name} have no "
                    f"place: a shape is the property `{_SHAPE_PREFIX}{{type}}` of the "
                    "label it outlines, so a Shape takes PyPSA's columns alone and "
                    f"no other column is named `{_SHAPE_PREFIX}...`"
                )
                raise ValueError(msg)
            keyed = sorted(_KEYS.intersection(names))
            if keyed:
                msg = (
                    f"The custom columns {keyed} of {c.name} have no place: "
                    f"{sorted(_KEYS)} key a record's rows, so no property or "
                    "attribute is named like one"
                )
                raise ValueError(msg)
            static = _own_static(c)
            columns[c.name] = {name: _column_dtype(static[name]) for name in names}
            nulls[c.name] = {name for name in names if static[name].isna().all()}
        declared = _registry_schema().attributes
        groups: dict[tuple[str, str], dict[str, str]] = {}
        for ctype, typed in columns.items():
            for name, dtype in typed.items():
                groups.setdefault((_dim_of(ctype), name), {})[ctype] = dtype
        dims: dict[str, set[str]] = {}
        for dim, name in groups:
            dims.setdefault(name, set()).add(dim)
        properties: dict[str, dict[str, str]] = {}
        attributes: dict[str, AttributeSpec] = {}
        for (dim, name), by_type in groups.items():
            valued = {t: d for t, d in by_type.items() if name not in nulls.get(t, ())}
            texts = {t: d for t, d in by_type.items() if d == "str"}
            nulled = (
                by_type
                if len(set(by_type.values())) == 1
                else dict.fromkeys(by_type, "str")
            )
            held = valued or texts or nulled
            placed = (
                "str" in held.values()
                or any(ctype in _CATALOG_TYPES for ctype in by_type)
                or len(dims[name]) > 1
                or (name in declared and declared[name].dims != _static_dims(name, dim))
            )
            if len(set(held.values())) > 1:
                kind = (
                    f"a property of `{dim}`" if placed else f"an attribute over `{dim}`"
                )
                msg = (
                    f"The custom column `{name}` is {kind} with one dtype, but its "
                    f"types carry it as {held}: write values of one dtype in each "
                    "type's column, or leave a column NULL"
                )
                raise ValueError(msg)
            dtype = next(iter(held.values()))
            for ctype in by_type:
                columns[ctype][name] = dtype
            if placed:
                properties.setdefault(dim, {})[name] = dtype
            else:
                attributes[name] = AttributeSpec(
                    dtype=dtype,  # type: ignore[arg-type]
                    dims=_static_dims(name, dim),
                )
        return cls(columns, properties, attributes)


def _outlined(
    n: pypsa.Network, con: DuckDBPyConnection, untouched: Collection[str] = ()
) -> dict[str, pd.DataFrame]:
    """`n`'s shapes as the properties of the labels they outline, by dim, each label with its `shape_{type}` geometries as WKT, in network order.

    A shape's `component` names its dim, a PyPSA type its dim (`_dim_of`),
    any other a dim of its own, its labels the shapes' `idx`.

    Raises
    ------
    ValueError
        If a shape lacks `component`, `idx` or `type`, two share `(component,
        idx, type)`, one on a PyPSA type names no member the network writes, or
        one names a dim PyPSA's schema declares as a dim of its own.

    Notes
    -----
    - [geometry](https://energy-models.github.io/datarecord/reference/tools/#geometry)

    """
    static = _own_static(n.c[_SHAPE])
    keys = ["component", "idx", "type"]
    wkt = pd.DataFrame(static[keys]).assign(geometry=static.geometry.to_wkt())
    frame = _unscoped(wkt.reset_index(), ["name"], _SHAPE, con)
    frame[keys] = frame[keys].replace("", None)
    lacking = frame[frame[keys].isna().any(axis=1)]
    if len(lacking):
        msg = (
            f"Shapes {lacking['name'].tolist()} lack a `component`, `idx` or "
            "`type`: a "
            "shape is the property `shape_{type}` of label `idx` of the dim its "
            "`component` names"
        )
        raise ValueError(msg)
    twice = frame[frame.duplicated(keys, keep=False)]
    if len(twice):
        msg = (
            f"Shapes {twice['name'].tolist()} share a `(component, idx, type)`: a "
            "label holds one value of the property `shape_{type}`"
        )
        raise ValueError(msg)
    pypsa_types = n.all_components - {_SHAPE}
    declared = _registry_schema().dimensions
    dims = {}
    for component, rows in frame.groupby("component", sort=False):
        if component in pypsa_types:
            held = n.c[component]
            members = set(
                _own_names(held)
                if component in untouched
                else _names(_own_static(held).index)
            )
            unknown = sorted(set(rows["idx"]) - members)
            if unknown:
                msg = (
                    f"Shapes on {component} name {unknown}, no {component} the "
                    "network writes, PyPSA's standard types and carriers it does "
                    "not list aside: a shape is a property of the label its `idx` "
                    "names"
                )
                raise ValueError(msg)
        elif component in declared:
            msg = (
                f"Shapes name `component` {component!r}, a dim PyPSA's schema "
                "declares: a `component` that is no PyPSA type names a dim of its own"
            )
            raise ValueError(msg)
        dims[component] = _dim_of(component) if component in pypsa_types else component
    frame["dim"] = frame["component"].map(dims)
    outlined = {}
    for dim, rows in frame.groupby("dim", sort=False):
        wide = rows.pivot(index="idx", columns="type", values="geometry")
        wide = wide.reindex(index=rows["idx"].unique(), columns=rows["type"].unique())
        wide.columns = [f"{_SHAPE_PREFIX}{t}" for t in wide.columns]
        outlined[str(dim)] = wide.rename_axis(dim).reset_index()
    return outlined


#: `pandas.api.types.infer_dtype` -> the mathspec dtype stored, `str` for anything unlisted.
_INFERRED = {
    "boolean": "bool",
    "integer": "int",
    "floating": "float",
    "mixed-integer-float": "float",
}


def _column_dtype(values: pd.Series) -> str:
    """Return the mathspec dtype to declare a custom static column with, by its non-null values, `str` unless they are bool, int or float."""
    return _INFERRED.get(pd.api.types.infer_dtype(values, skipna=True), "str")


#: A mathspec dtype -> the pandas dtype holding it with a null, `str` kept as `object`.
_PRESENTED = {"bool": "boolean", "int": "Int64", "float": "float64"}


def _cast_column(values: pd.Series) -> pd.Series:
    """`values` as the dtype `_column_dtype` declares them with, a null kept null."""
    dtype = _column_dtype(values)
    if dtype == "str":
        return values.where(values.isna(), values.astype(str))
    return values.astype(_PRESENTED[dtype])


#: How close two floats held by one label agree, as `numpy.isclose`'s defaults.
_RTOL, _ATOL = 1e-5, 1e-8
#: The DuckDB types `_clash` compares within that tolerance.
_FLOATS = (sqltypes.DOUBLE, sqltypes.FLOAT)


def _unscoped(
    frame: pd.DataFrame, keys: list[str], what: str, con: DuckDBPyConnection
) -> pd.DataFrame:
    """`frame` less `scenario`, one row per `keys`, as a member or connection is not scoped per scenario.

    Raises
    ------
    ValueError
        If a column's values at one row of `keys` differ by scenario (`_agreed`).

    """
    if SCENARIO not in frame.columns:
        return frame
    unscoped = frame.drop(columns=[SCENARIO]).drop_duplicates(subset=keys)
    if unscoped.empty:
        return unscoped
    scoped = frame.set_index([SCENARIO, *keys])
    for column in scoped.columns:
        by = scoped[column].groupby(level=SCENARIO, sort=False)
        values = {s: v.droplevel(SCENARIO) for s, v in by}
        _agreed(what, column, values, SCENARIO, con, "the scenarios")
    return unscoped


def _clash(rel: DuckDBPyRelation) -> Expression:
    """The aggregate over rows of `rel` at one label, true where their `value`s clash.

    Floats agree within `_RTOL` and `_ATOL` of their largest magnitude, an
    infinity only with one of its sign, any other type where equal, and missing
    clashes with held.
    """
    value = col("value")
    unequal = value.n_unique() > lit(1)
    if rel.types[rel.columns.index("value")] in _FLOATS:
        spread = fn.max(value) - fn.min(value)
        tolerance = lit(_ATOL) + lit(_RTOL) * fn.max(fn.abs(value))
        unequal &= fn.bool_or(fn.isinf(value)) | (spread > tolerance)
    held = fn.count(value)
    return ((lit(0) < held) & (held < fn.count())) | unequal


def _agreed(
    what: str,
    attr: str,
    values: Mapping[Any, pd.Series],
    by: str,
    con: DuckDBPyConnection,
    across: str = "the members holding it",
) -> pd.Series:
    """One value per label, from each member's `values` that holds it, in the order first held.

    Parameters
    ----------
    values
        Per member label of `by`, its values by label, on one or more index levels.
    across
        What the refusal names the value as one across.

    Raises
    ------
    ValueError
        If two members hold values for one label that do not agree (`_clash`).

    """
    long = pd.concat(values, names=[by])
    keys = list(range(1, long.index.nlevels))
    group = long.groupby(level=keys, sort=False, dropna=False).ngroup().to_numpy()
    rel = con.from_df(pd.DataFrame({"g": group, "value": long.to_numpy()}))
    clashing = rel.group_by(col("g")).agg(_clash(rel).alias("clash"))
    [(first,)] = clashing.filter(col("clash")).aggregate([fn.min(col("g"))]).fetchall()
    if first is not None:
        label, held = _first_apart(long, group == first)
        name = label[0] if len(label) == 1 else label
        msg = (
            f"{what} {name!r} has `{attr}` {held} by {by}: a value that does not "
            f"vary over `{by}` is one across {across}"
        )
        raise ValueError(msg)
    return long.groupby(level=keys, sort=False).first()


def _first_apart(long: pd.Series, apart: np.ndarray) -> tuple[tuple, dict[Any, Any]]:
    """The label of `long`'s rows `apart` marks, and each member's value there."""
    values = long[apart]
    return values.index[0][1:], {key[0]: value for key, value in values.items()}


def _weightings(record: RecordLike, dim: str, table: pd.DataFrame) -> pd.DataFrame:
    """`table`, PyPSA's weightings of `dim` by label, with each one `record` stores in place.

    A label reached by no row keeps `table`'s value. A multi-dim's label is
    matched by its coordinates, as PyPSA's index levels.

    Notes
    -----
    - [the broadcast rule](https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule)

    """
    restored = table.copy()
    labels = {dim: record.dimensions[dim]}
    overstated = Flags(frozenset({dim}), frozenset({dim}))
    for attribute, weighting in _WEIGHTINGS.items():
        frame = record.attributes.get(attribute)
        if weighting.dim != dim or frame is None:
            continue
        rows = settle(frame, labels, overstated, filter=False)
        rows = unpacked(record.schema, dim, rows.project(col(dim), col("value"))).df()
        keys = [c for c in rows.columns if c != "value"]
        values = rows.set_index(keys)["value"].reindex(table.index)
        restored[weighting.column] = values.fillna(table[weighting.column])
    return restored


def _connection_rows(c: pypsa.Components, con: DuckDBPyConnection) -> pd.DataFrame:
    """One type's connections as `(entity, port, bus)` rows, from its port columns.

    Undoes PyPSA's positional encoding: the table's `buses` lists the ports
    the live component has (`0` and `1` plus `additional_ports`), so the port
    count comes from the network rather than a guessed bound. `bus` is the
    row's one property; every per-port input, `delay` included, is long rows
    of `attributes/` keyed by the connection.

    Raises
    ------
    ValueError
        If an entity's bus at a port differs by scenario (`_unscoped`).

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

    """
    static = c.static
    frames = []
    for port, column in c.port_table.buses(c).items():
        if column not in static.columns:
            continue
        rows = static[[column]].reset_index().rename(columns={"name": ENTITY})
        # Agree on the unfiltered column, so a bus set in one scenario and empty
        # in another is refused rather than left a lone value.
        rows = _unscoped(rows, [ENTITY], c.name, con).rename(columns={column: BUS})
        rows = rows[rows[BUS].astype(str) != ""]
        if rows.empty:
            continue
        frames.append(rows.assign(port=port))
    if not frames:
        return pd.DataFrame(columns=[ENTITY, PORT, BUS])
    return pd.concat(frames, ignore_index=True)[[ENTITY, PORT, BUS]]


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


def _series_labels(columns: pd.Index) -> dict[str, pd.Index]:
    """The entity of each column of a dynamic frame, and its scenario on a stochastic network, by the key they are."""
    labels = {ENTITY: columns.get_level_values(-1)}
    if isinstance(columns, pd.MultiIndex):
        labels[SCENARIO] = columns.get_level_values(0).astype(str)
    return labels


def _union_all(rels: Iterable[DuckDBPyRelation | None]) -> DuckDBPyRelation | None:
    """`rels` but each `None` as one positional `UNION ALL`, `None` where there are none."""
    held = [r for r in rels if r is not None]
    return reduce(DuckDBPyRelation.union, held) if held else None


def _scalar_rows(static: pd.Series, exclude: pd.Index) -> pd.DataFrame:
    """Build long rows from the static scalars of `attr`, less their NULL `snapshot`.

    Mirrors `pypsa.components.array.Components._scalar_rows`, with its
    columns in long order (`entity, scenario, value`) and
    built in one constructor rather than by `reset_index` and column inserts:
    it runs once per attribute and type, where pandas' per-call overhead
    dominated an export.
    """
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
    #: The record this export is written over, `None` for one over nothing.
    base: RecordLike | None = None

    #: Whether a static value or series repeating what the record resolves without it is left out (`_changing`, `_differing`).
    _LEAVES_OUT_REPEATS: ClassVar[bool] = True

    @cached_property
    def schema(self) -> RecordSchema:
        """The layer's schema: the registry-derived schema, widened to this network.

        The dims and the registry's attributes are the same for
        every network - `_registry_schema` derives them once from PyPSA's whole
        component registry. What varies is `meta`, which carries this network's
        own attributes and `n.meta` for read-back and, per entity type, the
        names of its custom static columns (`_Custom.of`), each a property or an
        attribute of its type's dim, and, where the network has investment periods,
        `snapshot` as the multi-dim `(period, timestep)`.

        Notes
        -----
        - [the schema](https://energy-models.github.io/datarecord/design/model/schema/)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

        """
        base = _registry_schema()
        dimensions = dict(base.dimensions)
        properties = dict(base.properties)
        if PERIOD in self._snapshots:
            dimensions[SNAPSHOT] = Dimension(
                over={PERIOD: PERIOD, TIMESTEP: TIMESTEP},
                description="A timestep of a multi-period network in its period.",
            )
        for dim, frame in self._shapes.items():
            if dim not in dimensions:
                dimensions[dim] = Dimension(dtype="str", description=_SHAPE_DIM)
            geometry = PropertySpec(dtype="str", description=_GEOMETRY_DESCRIPTION)
            outlines = {name: geometry for name in frame.columns if name != dim}
            properties[dim] = properties.get(dim, {}) | outlines
        custom = self.custom
        for dim, typed in custom.properties.items():
            properties[dim] = properties[dim] | {
                name: PropertySpec(one_of=name)
                if dtype == "str" and name in dimensions
                else PropertySpec(dtype=dtype)  # type: ignore[arg-type]
                for name, dtype in typed.items()
                if name not in properties[dim]
            }
        return RecordSchema(
            **{
                **base.model_dump(),
                "dimensions": dimensions,
                "properties": properties,
                "attributes": custom.attributes | dict(base.attributes),
                "meta": {
                    "format": "pypsa-parquet",
                    "attributes": _collect_network_attributes(self.n, skip={"srid"}),
                    _CUSTOM: {
                        ctype: sorted(columns)
                        for ctype, columns in custom.columns.items()
                        if _dim_of(ctype) == ENTITY
                    },
                    "meta": dict(self.n.meta),
                    "crs": self.n.crs.to_string(),
                },
            }
        )

    @cached_property
    def _own_statics(self) -> dict[str, pd.DataFrame]:
        """Each type's `_own_static` by its name, as `_own` fills it."""
        return {}

    def _own(self, c: pypsa.Components) -> pd.DataFrame:
        """`c`'s `_own_static`, built once per type, as an export reads it per attribute."""
        if c.name not in self._own_statics:
            self._own_statics[c.name] = _own_static(c)
        return self._own_statics[c.name]

    @cached_property
    def custom(self) -> _Custom:
        """The network's custom static columns, placed (`_Custom.of`)."""
        return _Custom.of(self.n)

    @cached_property
    def _shapes(self) -> dict[str, pd.DataFrame]:
        """The network's shapes as properties of the labels they outline (`_outlined`)."""
        return _outlined(self.n, self.con)

    @cached_property
    def _shape_dims(self) -> list[str]:
        """The dims the network's shapes make, those PyPSA's schema lacks."""
        declared = _registry_schema().dimensions
        return [dim for dim in self._shapes if dim not in declared]

    def check(self) -> None:
        """Refuse a network a full export cannot write, such as a property differing by scenario (`_unscoped`), shapes `_outlined` refuses, or a custom property naming no label of its dim.

        Raises
        ------
        ValueError
            If any is found.

        """
        _ = self._labelled, self._entity_frame, self._connection_table
        schema = self.schema
        self._refuse_missing_buses()
        for subject, typed in self.custom.properties.items():
            for name in typed:
                dim = schema.properties[subject][name].one_of
                if dim is None:
                    continue
                values = self.properties[subject].project(col(name)).df()[name]
                axis = self.dimensions.get(dim)
                labels = (
                    schema.dimensions[dim].one_of or ()
                    if axis is None
                    else {label for (label,) in axis.fetchall()}
                )
                unknown = sorted(set(values.dropna()) - set(labels))
                if unknown:
                    msg = (
                        f"`{subject}.{name}` names {unknown}, no label of `{dim}`: "
                        "a custom text column named like a dim refers to it"
                    )
                    raise ValueError(msg)

    def _refuse_missing_buses(self) -> None:
        """Refuse a `GlobalConstraint` at a bus `n.buses` lacks.

        Raises
        ------
        ValueError
            If one is found.

        """
        constraints = self._labelled[GLOBAL_CONSTRAINT]
        if BUS not in constraints:
            return
        at = constraints[[GLOBAL_CONSTRAINT, BUS]].dropna()
        lost = at[~at[BUS].isin(self._buses[BUS])]
        if lost.empty:
            return
        found = sorted(
            f"'{name}' at '{bus}'"
            for name, bus in zip(lost[GLOBAL_CONSTRAINT], lost[BUS], strict=True)
        )
        msg = f"GlobalConstraint {', '.join(found)} names a bus `n.buses` lacks"
        raise ValueError(msg)

    def _series_checked(self, c: pypsa.Components) -> list[str]:
        """The series of `c` this export reads, which `_stray_series` compares against its members."""
        return [attr for attr in c.dynamic if not self._pending(c, attr)]

    def _stray_series(self) -> tuple[str, str, list[Any]] | None:
        """The first `(type, attribute, columns)` whose series has columns naming no member of its type, else `None`.

        Compared by `get_indexer`, by `(scenario, name)` on a stochastic network.
        A type whose static frame is unread is compared against its names, so it
        stays unread.
        """
        for c in self.n.components.values():
            if c.name == _SHAPE:
                continue
            keys: pd.Index | None = None
            for attr in self._series_checked(c):
                frame = c.dynamic[attr]
                if not len(frame.columns):
                    continue
                if keys is None:
                    keys = self._member_keys(c)
                stray = frame.columns[keys.get_indexer(frame.columns) == -1]
                if len(stray):
                    return c.name, attr, list(stray)
        return None

    def _member_keys(self, c: pypsa.Components) -> pd.Index:
        """The labels a series of `c` may name: its static index, or its names crossed with the scenarios while static is unread."""
        if c._static is not _UNREAD:
            return c.static.index
        names = c._static_names()
        if not self.n.has_scenarios:
            return names
        return pd.MultiIndex.from_product(
            [self.n.scenarios, names], names=["scenario", names.name]
        )

    @cached_property
    def _snapshots(self) -> pd.DataFrame:
        """`n.snapshots` in PyPSA's order, a column per level, `snapshot` or a multi-period network's `period` and `timestep`."""
        return self.n.snapshots.to_frame(index=False)

    @cached_property
    def _snapshot_rel(self) -> DuckDBPyRelation:
        """`_snapshots` as the `snapshot` axis, a multi-dim typed as declared, a single-period label as PyPSA holds it."""
        return self._snapshot_positions.project(col(SNAPSHOT))

    @cached_property
    def _snapshot_positions(self) -> DuckDBPyRelation:
        """`_snapshot_rel` with each label's 1-based position in `n.snapshots` as `_pos`."""
        positions = np.arange(1, len(self._snapshots) + 1)
        rel = self.con.from_df(self._snapshots.assign(_pos=positions))
        keyed = rel.project(self._snapshot_key, col("_pos"))
        if SNAPSHOT not in self.schema.multi_dimensions:
            return keyed
        return cast_declared(self.schema, keyed, SNAPSHOT)

    @cached_property
    def _snapshot_labels(self) -> DuckDBPyRelation:
        """One row holding the labels of `_snapshot_rel` in PyPSA's order as the list `_labels`."""
        return self._snapshot_positions.aggregate(
            f"list({SNAPSHOT} ORDER BY _pos) AS _labels"
        )

    @cached_property
    def _snapshot_key(self) -> Expression:
        """The one `snapshot` column the schema declares, over the columns of `_snapshots`."""
        if SNAPSHOT not in self.schema.multi_dimensions:
            return col(SNAPSHOT)
        return packed(self.schema, SNAPSHOT, self._snapshots.columns, {}).alias(
            SNAPSHOT
        )

    @property
    def dimensions(self) -> LazyRelations:
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
            SNAPSHOT: lambda: self._snapshots,
            TIMESTEP: lambda: self._timesteps,
            PERIOD: lambda: self.n.investment_periods.to_frame(index=False),
            SCENARIO: lambda: self.n.scenario_weightings.index.to_frame(index=False),
            PORT: lambda: self._ports,
        }
        for dim in self._labelled:
            axes[dim] = partial(self._labelled[dim].__getitem__, [dim])
        present = tuple(d for d, frame in axes.items() if len(frame()) > 0)
        axes[ENTITY] = lambda: self._entity_frame[[ENTITY]]

        def axis(dim: str) -> DuckDBPyRelation:
            if dim == CONNECTION:
                return self._connection_frame().project(col(CONNECTION))
            if dim == SNAPSHOT:
                return self._snapshot_rel
            return cast_declared(self.schema, self.con.from_df(axes[dim]()), dim)

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
            for frame in (self._buses, self._sub_networks, self._entity_frame)
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

    @cached_property
    def _labelled(self) -> dict[str, pd.DataFrame]:
        """Every labelled axis with its properties, by dim, PyPSA's then those shapes make."""
        return {
            BUS: self._buses,
            CARRIER: self._carriers,
            SUB_NETWORK: self._sub_networks,
            GLOBAL_CONSTRAINT: self._labels("GlobalConstraint"),
            **{dim: self._labels(ctype) for ctype, dim in _CATALOG_TYPES.items()},
            **{dim: self._shapes[dim] for dim in self._shape_dims},
        }

    def _labels(self, ctype: str) -> pd.DataFrame:
        """One `_AXIS_TYPES` type's members beyond PyPSA's library (`_own_static`) as labels of its dim, with their properties."""
        dim = _AXIS_TYPES[ctype]
        static = self._own(self.n.c[ctype])
        columns = [p for p in self.schema.properties[dim] if p in static.columns]
        frame = static[columns].rename_axis(index={"name": dim}).reset_index()
        return self._with_shapes(
            _unscoped(_none_for_empty(frame, dim), [dim], ctype, self.con), dim
        )

    def _with_shapes(self, frame: pd.DataFrame, dim: str) -> pd.DataFrame:
        """`frame`, labels of `dim` in order, with the geometries of those shapes outline (`_outlined`)."""
        shapes = self._shapes.get(dim)
        return frame if shapes is None else frame.merge(shapes, on=dim, how="left")

    @cached_property
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
            if _as_entities(c, self._untouched)
        ]
        if not frames:
            return pd.DataFrame(columns=[ENTITY, CTYPE])
        merged = _none_for_empty(pd.concat(frames, ignore_index=True), ENTITY)
        merged = self._with_shapes(merged, ENTITY)
        lead = [ENTITY, CTYPE]
        return merged[[*lead, *(c for c in merged.columns if c not in lead)]]

    @property
    def _untouched(self) -> Collection[str]:
        """The types whose frames an export leaves unread."""
        return ()

    def _connections_of(self, c: pypsa.Components) -> pd.DataFrame:
        """`c`'s connections as `(entity, port, bus)` rows."""
        return _connection_rows(c, self.con)

    @cached_property
    def _connection_table(self) -> pd.DataFrame:
        """Every type's connections as `(entity, port, bus)` rows."""
        types = [
            c.name
            for c in self.n.components
            if _as_entities(c, self._untouched)
            and (c.name in self._untouched or c.ports)
        ]
        frames = [self._connections_of(self.n.c[ctype]) for ctype in types]
        if not frames:
            return pd.DataFrame(columns=[ENTITY, PORT, BUS])
        return pd.concat(frames, ignore_index=True)

    @cached_property
    def _ports(self) -> pd.DataFrame:
        """The `port` axis: `0` up to the highest port a connection is at, `1` at least."""
        ports = self._connection_table[PORT]
        highest = max(1, int(ports.max())) if len(ports) else 1
        return pd.DataFrame({PORT: np.arange(highest + 1)})

    def _connection_frame(self) -> DuckDBPyRelation:
        """Every type's connections in one frame, `(connection, bus)`."""
        flat = self.con.from_df(self._connection_table)
        return flat.project(
            _connection_struct(col(ENTITY), col(PORT)), star(exclude=[ENTITY, PORT])
        )

    def _as_long_rel(
        self,
        c: pypsa.Components,
        attr: str,
        *,
        unchanged: str | None,
        ports: pd.Series | None,
        drop_defaults: bool,
    ) -> DuckDBPyRelation | None:
        """`attr` as long rows `(entity, snapshot, scenario, value)`, `None` where it has none.

        A series is melted, a static scalar a NULL-snapshot row, each left
        out where it repeats what record attribute `unchanged` resolves to
        without it (`_differing`, `_changing`).

        Parameters
        ----------
        ports
            Each entity's port `attr` is at, which keys `unchanged`, `None`
            where `attr` is at no port.
        drop_defaults
            Whether a value is compared with its type's default where `base`
            holds none for it.

        Raises
        ------
        AttributeError
            If `attr` is neither varying, an output nor a static column of `c`.

        Notes
        -----
        - [wide and long rows](https://energy-models.github.io/datarecord/reference/data-model/#wide-and-long-rows)

        """
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

        parts: list[DuckDBPyRelation | None] = []
        excluded: pd.Index = pd.Index([])
        curved = self._curved(c, attr)
        self._warn_shadowed(c, attr, curved)
        if varying and self._pending(c, attr):
            excluded = self._series_held(c, unchanged, ports)
        elif varying:
            parts.append(
                self._melt_dynamic(
                    c, attr, unchanged=unchanged, ports=ports, exclude=curved
                )
            )
            excluded = c.dynamic[attr].columns
        if self._piecewise_pending(c, attr):
            excluded = excluded.union(
                self._curves_held(c, unchanged, ports), sort=False
            )
        excluded = excluded.union(curved, sort=False)
        static_backed = attr in c.defaults.index and bool(c.defaults.at[attr, "static"])
        if not varying or static_backed:
            parts.append(
                self._scalar_rel(
                    c,
                    attr,
                    excluded,
                    unchanged=unchanged,
                    ports=ports,
                    drop_defaults=drop_defaults,
                )
            )
        return _union_all(parts)

    def _curve_frame(self, c: pypsa.Components, attr: str) -> pd.DataFrame | None:
        """`c`'s breakpoint frame of `attr`, `None` where it holds none."""
        if self._piecewise_pending(c, attr):
            return None
        frame = c.piecewise.get(attr)
        if frame is None or frame.empty:
            return None
        return frame

    def _curved(self, c: pypsa.Components, attr: str) -> pd.Index:
        """The names of `c` holding a curve on `attr`."""
        frame = self._curve_frame(c, attr)
        if frame is None:
            return pd.Index([])
        return frame.columns.get_level_values("name").unique()

    def _warn_shadowed(self, c: pypsa.Components, attr: str, curved: pd.Index) -> None:
        """Warn once where a curved name also holds a non-default scalar or series of `attr`."""
        if not len(curved) or attr not in c.defaults.index:
            return
        default = c.defaults.at[attr, "default"]
        shadowed: set[str] = set()
        if attr in c.static.columns:
            static = c.static[attr]
            level = static.index.get_level_values(-1)
            held = static[level.isin(curved)]
            differing = held[~((held == default) | (held.isna() & pd.isna(default)))]
            shadowed.update(map(str, differing.index.get_level_values(-1)))
        if attr in c.dynamic:
            wide = c.dynamic[attr]
            wide = wide.loc[:, wide.columns.get_level_values(-1).isin(curved)]
            if not wide.empty:
                moved = (wide.notna() & (wide != default)).any()
                shadowed.update(
                    map(str, wide.columns[moved.to_numpy()].get_level_values(-1))
                )
        if shadowed:
            logger.warning(
                "%s %s hold a piecewise curve on '%s' and a scalar or series; "
                "the export skips the scalar or series because the optimiser "
                "ignores it under a curve.",
                c.name,
                sorted(shadowed),
                attr,
            )

    def _null_snapshot(self) -> Expression:
        """A NULL `snapshot`, typed as the label where it is a struct (a NULL of no type is not one)."""
        snapshot = lit(None)
        if SNAPSHOT in self.schema.multi_dimensions:
            snapshot = snapshot.cast(
                self.schema.duck_type(self.schema.dimensions[SNAPSHOT])
            )
        return snapshot

    def _curve_rel(
        self,
        c: pypsa.Components,
        source_attr: str,
        *,
        unchanged: str | None,
        ports: pd.Series | None,
    ) -> DuckDBPyRelation | None:
        """One row per filled breakpoint of `c`'s curves on `source_attr`, `None` where it has none."""
        frame = self._curve_frame(c, source_attr)
        if frame is None:
            return None
        x = c._piecewise_schema(source_attr).x
        names = frame.columns.get_level_values("name").unique()
        rows = pd.concat(
            [
                pd.DataFrame(
                    {
                        ENTITY: name,
                        "value": frame[(name, source_attr)].to_numpy(dtype=float),
                        "breakpoint": frame[(name, x)].to_numpy(dtype=float),
                    }
                )
                for name in names
            ],
            ignore_index=True,
        ).dropna(subset=["value", "breakpoint"])
        rows = self._changed_curves(c, rows, unchanged, ports)
        if rows.empty:
            return None
        return self.con.from_df(rows).project(
            col(ENTITY),
            self._null_snapshot().alias(SNAPSHOT),
            lit(None).cast(sqltypes.VARCHAR).alias(SCENARIO),
            col("value"),
            col("breakpoint"),
        )

    def _pending(self, c: pypsa.Components, attr: str) -> bool:
        """Whether `c`'s series `attr` is left unread, its static part alone written."""
        return False

    def _piecewise_pending(self, c: pypsa.Components, attr: str) -> bool:
        """Whether `c`'s curves on `attr` are left unread and unwritten."""
        return False

    def _base_curves(
        self, c: pypsa.Components, attribute: str | None, ports: pd.Series | None
    ) -> pd.DataFrame | None:
        """`base`'s curve points of `attribute` at no scenario, `None` where it holds none."""
        rel = None if attribute is None else self._base_attribute(attribute)
        dim = _dim_of(c.name) if ports is None else CONNECTION
        if rel is None or dim not in rel.columns or "breakpoint" not in rel.columns:
            return None
        held = rel.filter(col(dim).isnotnull() & col("breakpoint").isnotnull())
        if SCENARIO in rel.columns:
            held = held.filter(col(SCENARIO).isnull())
        if ports is None:
            keys = [col(dim).cast(sqltypes.VARCHAR).alias(ENTITY)]
        else:
            keys = [
                fn.struct_extract(col(dim), lit(ENTITY))
                .cast(sqltypes.VARCHAR)
                .alias(ENTITY),
                fn.struct_extract(col(dim), lit(PORT))
                .cast(sqltypes.BIGINT)
                .alias(PORT),
            ]
        frame = held.project(
            *keys, col("breakpoint").alias("breakpoint"), col("value")
        ).df()
        if ports is not None:
            frame = frame[frame[PORT] == frame[ENTITY].map(ports)]
        return frame[[ENTITY, "breakpoint", "value"]]

    def _curves_held(
        self, c: pypsa.Components, attribute: str | None, ports: pd.Series | None
    ) -> pd.Index:
        """The entries of `c`'s static index whose entity `base` holds a curve of `attribute` for."""
        frame = self._base_curves(c, attribute, ports)
        static = self._own(c)
        if frame is None:
            return static.index[:0]
        level = static.index.get_level_values(-1)
        return static.index[level.isin(frame[ENTITY])]

    def _changed_curves(
        self,
        c: pypsa.Components,
        rows: pd.DataFrame,
        attribute: str | None,
        ports: pd.Series | None,
    ) -> pd.DataFrame:
        """`rows` without the entities whose curve equals the one `base` holds of `attribute`."""
        held = self._base_curves(c, attribute, ports)
        if held is None or rows.empty:
            return rows

        def points(frame: pd.DataFrame) -> dict[str, frozenset]:
            return {
                name: frozenset(zip(g["breakpoint"], g["value"], strict=True))
                for name, g in frame.groupby(ENTITY)
            }

        base = points(held)
        same = {k for k, v in points(rows).items() if base.get(k) == v}
        return rows[~rows[ENTITY].isin(same)]

    def _series_held(
        self, c: pypsa.Components, attribute: str | None, ports: pd.Series | None
    ) -> pd.Index:
        """The entries of `c`'s static index whose entity `base` holds a series of `attribute` for, at the port `ports` gives."""
        rel = None if attribute is None else self._base_attribute(attribute)
        dim = _dim_of(c.name) if ports is None else CONNECTION
        static = self._own(c)
        if rel is None or dim not in rel.columns or SNAPSHOT not in rel.columns:
            return static.index[:0]
        held = rel.filter(col(dim).isnotnull() & col(SNAPSHOT).isnotnull())
        if ports is None:
            names = held.project(col(dim).cast(sqltypes.VARCHAR).alias(ENTITY))
        else:
            names = held.project(
                fn.struct_extract(col(dim), lit(ENTITY))
                .cast(sqltypes.VARCHAR)
                .alias(ENTITY),
                fn.struct_extract(col(dim), lit(PORT))
                .cast(sqltypes.BIGINT)
                .alias(PORT),
            )
        frame = names.distinct().df()
        if ports is not None:
            frame = frame[frame[PORT] == frame[ENTITY].map(ports)]
        level = static.index.get_level_values(-1)
        return static.index[level.isin(frame[ENTITY])]

    def _melt_dynamic(
        self,
        c: pypsa.Components,
        attr: str,
        *,
        unchanged: str | None,
        ports: pd.Series | None,
        exclude: Collection[str] = (),
    ) -> DuckDBPyRelation | None:
        """Melt one varying attribute's wide dynamic frame to long rows, `None` where it has no columns left.

        A column repeating record attribute `unchanged`'s series is left out
        (`_differing`), keyed by its entity's connection to its bus in
        `ports` where given.

        Each column is one arrow row holding its series as a list, unnested
        entity by entity so a parquet row group holds few entity names, which
        its dictionary encodes.

        A value's position in its list indexes `_snapshot_labels`, so a label
        never becomes an SQL identifier or literal. A NaN cell is a NULL row.
        """
        wide = c.dynamic[attr]
        if len(exclude):
            wide = wide.loc[:, ~wide.columns.get_level_values(-1).isin(exclude)]
        if unchanged is not None and not wide.empty:
            keys = self._series_keys(wide.columns, _dim_of(c.name), ports)
            wide = wide.loc[:, self._differing(unchanged, wide.to_numpy(), keys)]
        if wide.empty:
            return None
        labels = _series_labels(wide.columns)
        steps = len(wide.index)
        # A zero-copy view of PyPSA's one F-contiguous float64 block.
        values = pa.array(wide.to_numpy().T.reshape(-1), from_pandas=True)
        offsets = pa.array(np.arange(0, len(values) + 1, steps))
        table = pa.table(
            {
                **{k: pa.array(v) for k, v in labels.items()},
                "values": pa.LargeListArray.from_arrays(offsets, values),
            }
        )
        batches = table.to_batches(max_chunksize=max(1, _MELT_BATCH // steps))
        melted = self.con.from_arrow(pa.Table.from_batches(batches)).project(
            *map(col, labels),
            fn.unnest(col("values")).alias("value"),
            fn.generate_subscripts(col("values"), lit(1)).alias("_pos"),
        )
        scenario = col(SCENARIO) if SCENARIO in labels else lit(None)
        return melted.cross(self._snapshot_labels).project(
            col(ENTITY).cast(sqltypes.VARCHAR).alias(ENTITY),
            fn.list_extract(col("_labels"), col("_pos")).alias(SNAPSHOT),
            scenario.cast(sqltypes.VARCHAR).alias(SCENARIO),
            col("value"),
        )

    def _series_keys(
        self, columns: pd.Index, dim: str, ports: pd.Series | None
    ) -> DuckDBPyRelation:
        """The keys of each column of a dynamic frame of `dim`'s type, its 1-based position as `_col`, its label of `dim` or its entity's connection at its port in `ports`, and its scenario where stochastic."""
        labels = _series_labels(columns)
        frame = pd.DataFrame({**labels, "_col": np.arange(1, len(columns) + 1)})
        entity = col(ENTITY).cast(sqltypes.VARCHAR)
        key = entity.alias(dim)
        if ports is not None:
            frame[PORT] = frame[ENTITY].map(ports)
            key = _connection_struct(entity, col(PORT))
        scenario = [col(SCENARIO)] if SCENARIO in labels else []
        return self.con.from_df(frame).project(key, *scenario, col("_col"))

    def _differing(
        self, attribute: str, values: np.ndarray, keys: DuckDBPyRelation
    ) -> np.ndarray:
        """Whether each column of `values`, a series over `n.snapshots` in order, differs from `base`'s series of `attribute`.

        A column is matched on its row of `keys`, by its 1-based position
        `_col`, NULL equal to NULL. It differs where `base` holds no row at
        one of the network's snapshots or another value there, NULL equal to
        NaN, and wholly where `base` holds no such keys. `base`'s rows at
        other snapshots, or set on a key `keys` lacks, are not compared.
        """
        steps, width = values.shape
        held = self._base_attribute(attribute)
        on = [k for k in keys.columns if k != "_col"]
        if held is None or not {SNAPSHOT, *on} <= set(held.columns):
            return np.ones(width, dtype=bool)
        unkeyed = [
            col(c).isnull()
            for c in held.columns
            if c not in {*on, SNAPSHOT, "attribute", "value"}
        ]
        series = held.filter(ex_all([col(SNAPSHOT).isnotnull(), *unkeyed]))
        series = series.set_alias("h")
        matched = join_on(
            series, keys.set_alias("k"), [null_safe("h", "k", on)] if on else []
        )
        positions = self._snapshot_positions.set_alias("s")
        placed = matched.join(
            positions, col("h", SNAPSHOT) == col("s", SNAPSHOT)
        ).project(
            (col("s", "_pos") - lit(1)).cast(sqltypes.INTEGER).alias("_pos"),
            (col("k", "_col") - lit(1)).cast(sqltypes.INTEGER).alias("_col"),
            col("h", "value").cast(sqltypes.DOUBLE).alias("value"),
        )
        seen = np.zeros(width, dtype=np.int64)
        apart = np.zeros(width, dtype=bool)
        for batch in placed.to_arrow_reader():
            at = batch["_col"].to_numpy()
            network = values[batch["_pos"].to_numpy(), at]
            held_values = batch["value"].to_numpy(zero_copy_only=False)
            nan = np.isnan(network) & np.isnan(held_values)
            apart[at[(network != held_values) & ~nan]] = True
            seen += np.bincount(at, minlength=width)
        return apart | (seen < steps)

    def _scalar_rel(
        self,
        c: pypsa.Components,
        attr: str,
        exclude: pd.Index,
        *,
        unchanged: str | None,
        ports: pd.Series | None,
        drop_defaults: bool,
    ) -> DuckDBPyRelation | None:
        """NULL-snapshot long rows from `attr`'s static scalars (`_scalar_rows`), `None` where none remain.

        A custom column has no default of its type to compare with.
        """
        static = _shared_across_scenarios(_scalar_rows(self._own(c)[attr], exclude))
        if unchanged is not None:
            undeclared = static
            if drop_defaults and attr in c.defaults.index:
                undeclared = _drop_default_rows(static, c.defaults.at[attr, "default"])
            static = self._changing(
                unchanged, _dim_of(c.name), static, undeclared.index, ports
            )
        if static.empty:
            return None
        return self.con.from_df(static).project(
            col(ENTITY),
            self._null_snapshot().alias(SNAPSHOT),
            col(SCENARIO),
            col("value"),
        )

    def _changing(
        self,
        attribute: str,
        dim: str,
        rows: pd.DataFrame,
        undeclared: pd.Index,
        ports: pd.Series | None,
    ) -> pd.DataFrame:
        """`rows` but those repeating what `attribute` resolves to without them.

        A label of `dim`, or an entity's connection at its port in `ports`,
        that `base` holds rows of `attribute` for is compared with each of
        them at the row's scenario or none, at any for a row at none, and left
        out only where each is static and equal, NULL equal to NULL. A row at
        none is also left out only where those reach every scenario
        (`_held_everywhere`) or it is at its default. Any other row is kept where its index is in
        `undeclared`, those not at their declared default
        (`_drop_default_rows`).
        """
        keep = pd.Series(rows.index.isin(undeclared), index=rows.index)
        on = [ENTITY]
        if ports is not None:
            rows = rows.assign(**{PORT: rows[ENTITY].map(ports)})
            on.append(PORT)
        key = (attribute, dim if ports is None else CONNECTION)
        if key not in self._base_statics:
            self._base_statics[key] = self._base_static(*key)
        held = self._base_statics[key]
        if held is None:
            return rows[keep]
        pairs = rows.reset_index().merge(held, on=on, suffixes=("", "_base"))
        scenario, scenario_base = pairs[SCENARIO], pairs[f"{SCENARIO}_base"]
        at = scenario.isna() | scenario_base.isna() | (scenario == scenario_base)
        pairs = pairs[at]
        value, value_base = pairs["value"], pairs["value_base"]
        equal = (value == value_base) | (value.isna() & value_base.isna())
        repeats = (pairs["static"] & equal).groupby(pairs["index"]).all()
        shared = rows.loc[repeats.index, SCENARIO].isna()
        at_default = ~repeats.index.isin(undeclared)
        everywhere = self._held_everywhere(pairs[f"{SCENARIO}_base"], pairs["index"])
        keep[repeats.index] = ~(repeats & (~shared | at_default | everywhere))
        return rows[keep]

    def _held_everywhere(self, scenario: pd.Series, by: pd.Series) -> pd.Series:
        """Whether, for each group of `by`, `base`'s rows at `scenario` reach every scenario of the network, one at none or one at each.

        A row `_shared_across_scenarios` folds repeats `base` only there, as at
        a scenario `base` holds no row it resolves to the default.
        """
        labels = self.n.scenarios.astype(str) if self.n.has_scenarios else []
        held = scenario[scenario.isin(labels)].groupby(by).nunique()
        each = held.reindex(by.unique(), fill_value=0).eq(len(labels))
        return scenario.isna().groupby(by).any() | each

    @cached_property
    def _base_attributes(self) -> dict[str, DuckDBPyRelation | None]:
        """`_base_attribute` by its argument, resolved once for each comparison with it."""
        return {}

    def _base_attribute(self, attribute: str) -> DuckDBPyRelation | None:
        """`base`'s rows of `attribute`, `None` where it holds none or there is no `base`."""
        if attribute not in self._base_attributes:
            rel = None if self.base is None else self.base.attributes.get(attribute)
            self._base_attributes[attribute] = rel
        return self._base_attributes[attribute]

    @cached_property
    def _base_statics(self) -> dict[tuple[str, str], pd.DataFrame | None]:
        """`_base_static` by its arguments, read once for the component types sharing an attribute."""
        return {}

    def _base_static(self, attribute: str, dim: str) -> pd.DataFrame | None:
        """`base`'s rows of `attribute` keyed by `dim`, one `(entity, scenario, value, static)` per entity and scenario, or `None`.

        A row is static where no key but `dim` and `scenario` is set, and an
        entity at a scenario where each of its rows is, and only then is `value`
        its one row's. A connection's rows carry `port` after `entity`. It is aggregated in DuckDB so a series' rows never reach pandas.
        """
        rel = self._base_attribute(attribute)
        if rel is None or dim not in rel.columns:
            return None
        others = [
            c for c in rel.columns if c not in (dim, SCENARIO, "attribute", "value")
        ]
        keys = {ENTITY: col(dim)}
        if dim == CONNECTION:
            keys = {k: fn.struct_extract(col(dim), lit(k)) for k in (ENTITY, PORT)}
        scenario = col(SCENARIO) if SCENARIO in rel.columns else lit(None)
        static = ex_all([lit(True), *(col(c).isnull() for c in others)])
        keyed = rel.filter(col(dim).isnotnull()).project(
            *(
                key.cast(sqltypes.BIGINT if k == PORT else sqltypes.VARCHAR).alias(k)
                for k, key in keys.items()
            ),
            scenario.cast(sqltypes.VARCHAR).alias(SCENARIO),
            col("value"),
            static.alias("static"),
        )
        return keyed.aggregate(
            [
                *map(col, [*keys, SCENARIO]),
                fn.any_value(col("value")).alias("value"),
                fn.bool_and(col("static")).alias("static"),
            ]
        ).df()

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
            if not _exported(c) or c.name == _SHAPE:
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
            levels = pd.MultiIndex.from_frame(self._snapshots)
            return self.n.snapshot_weightings.set_axis(levels)
        if dim == PERIOD:
            return self.n.investment_period_weightings
        return self.n.scenario_weightings

    def _weighting_frame(self, attribute: str) -> DuckDBPyRelation:
        """One axis weighting as long rows over its axis, none where it is a series repeating `base`'s (`_differing`)."""
        weighting = _WEIGHTINGS[attribute]
        table = self._weighting_table(weighting.dim)
        weights = table[weighting.column].rename("value").reset_index()
        if weighting.dim == SNAPSHOT:
            values = weights[["value"]].to_numpy(dtype=float)
            one = self.con.from_df(pd.DataFrame({"_col": [1]}))
            if not self._differing(attribute, values, one)[0]:
                weights = weights.iloc[:0]
        keys = (
            self._snapshot_key if weighting.dim == SNAPSHOT else star(exclude=["value"])
        )
        return self.con.from_df(weights).project(
            keys,
            lit(attribute).alias("attribute"),
            lit(None).alias("breakpoint"),
            col("value").cast(sqltypes.DOUBLE).alias("value"),
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
            if _exported(c) and c.name != _SHAPE
        }

    @property
    def properties(self) -> LazyRelations:
        """The keys and property columns of the entities, buses, carriers, sub-networks and connections."""
        frames: dict[str, Callable[[], DuckDBPyRelation]] = {
            ENTITY: lambda: self.con.from_df(self._entity_frame),
            CONNECTION: self._connection_frame,
        }
        for dim, labels in self._labelled.items():
            frames[dim] = partial(self.con.from_df, labels)
        labelled = [d for d, labels in self._labelled.items() if len(labels)]
        subjects = (ENTITY, *labelled, CONNECTION)

        def frame(subject: str) -> DuckDBPyRelation:
            rel = frames[subject]()
            declared = {subject, *self.schema.properties[subject]}
            kept = rel.project(*(col(c) for c in rel.columns if c in declared))
            return cast_declared(self.schema, kept, subject)

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
        skipped = {
            "name",
            *_output_attributes(c),
            *self.schema.properties[_dim_of(c.name)],
        }
        per_port = self._port_stems(c)
        stems: list[str] = []
        for attr in [*defaults.index, *_custom_names(c)]:
            if attr in skipped:
                continue
            if attr in defaults.index and not (
                defaults.at[attr, "varying"] or attr in c.static.columns
            ):
                continue
            stem = record_input(c.name, per_port.get(attr, attr))
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
        placed = self.schema.properties[ENTITY]
        static = self._own(c)
        columns = [p for p in placed if p in static.columns]
        frame = static[columns].reset_index().rename(columns={"name": "entity"})
        return _unscoped(frame, [ENTITY], ctype, self.con)

    def _tagged_long_rel(
        self,
        c: pypsa.Components,
        attribute: str,
        source_attr: str,
        *,
        drop_defaults: bool,
        ports: pd.Series | None = None,
    ) -> DuckDBPyRelation | None:
        """`source_attr`'s long relation, tagged with `attribute` and a NULL breakpoint.

        `attribute` is the record-facing name and `source_attr` PyPSA's - the two
        differ for a per-port attribute, whose ports (`p0`, `p1`, ...) each
        contribute under the one stem, and for a renamed result (`p_internal`).
        A NULL `breakpoint` marks the value a scalar (https://energy-models.github.io/datarecord/design/storage/format/#the-long-schema).
        An `_AXIS_TYPES` type's rows are keyed by its dim in place of
        `entity`. `None` where `source_attr` contributes no rows.
        """
        unchanged = attribute if self._LEAVES_OUT_REPEATS else None
        rel = self._as_long_rel(
            c,
            source_attr,
            unchanged=unchanged,
            ports=ports,
            drop_defaults=drop_defaults,
        )
        if rel is not None:
            rel = rel.select(*map(col, rel.columns), lit(None).alias("breakpoint"))
        curves = self._curve_rel(c, source_attr, unchanged=unchanged, ports=ports)
        rel = _union_all([rel, curves])
        if rel is None:
            return None
        dim = _dim_of(c.name)
        return rel.select(
            *(
                col(name).alias(dim) if name == ENTITY else col(name)
                for name in rel.columns
                if name != "breakpoint"
            ),
            lit(attribute).alias("attribute"),
            col("breakpoint"),
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
            elif (source := pypsa_attribute(c.name, attribute)) in c.defaults.index or (
                source in c.static.columns
            ):
                rel = self._tagged_long_rel(c, attribute, source, drop_defaults=True)
                if rel is not None:
                    rels.append(rel)
        rel = self._union_select(rels, columns)
        if attribute in self.custom.attributes:
            # A type's all-NULL column unions as its own dtype, not the declared one.
            rel = cast_declared(self.schema, rel, None, attribute)
        return rel

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
        port: int,
        *,
        drop_defaults: bool,
    ) -> DuckDBPyRelation | None:
        """`column`'s long relation at `port`, tagged `attribute`, keyed by connection.

        The `connection` struct packs the entity with `port`. A port holds
        rows only for the entities with a bus there, joined from the port's
        static bus column rather than mapped in pandas, so the rows stay a
        relation. `None` where the port has no bus column or `column`
        contributes no rows.

        Notes
        -----
        - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

        """
        con = self.con
        static = c.static
        bus_column = c.port_table.column(BUS, port)
        if bus_column not in static.columns:
            return None
        # `static` is `(scenario, name)`-indexed on a stochastic network; a
        # component's bus does not vary by scenario, so it is agreed on the
        # unfiltered column and then one bus per component remains.
        names = static[[bus_column]].reset_index().rename(columns={"name": ENTITY})
        names = _unscoped(names, [ENTITY], c.name, con)
        attached = names.set_index(ENTITY)[bus_column]
        attached = attached[attached.astype(str) != ""]
        bus_df = attached.rename("bus").rename_axis("entity").reset_index()  # noqa: F841
        bus_rel = con.sql(  # noqa: F841 - bound by name for the join below
            "SELECT CAST(entity AS VARCHAR) entity FROM bus_df"
        )
        v = self._tagged_long_rel(
            c,
            attribute,
            column,
            drop_defaults=drop_defaults,
            ports=pd.Series(port, index=attached.index),
        )
        if v is None:
            return None
        return (
            v.set_alias("v")
            .join(bus_rel.set_alias("b"), col("v", ENTITY) == col("b", ENTITY))
            .project(
                *(col("v", name) for name in v.columns if name != ENTITY),
                _connection_struct(col("v", ENTITY), lit(port)),
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
        schema = self.schema
        return empty_relation(
            self.con, **{c: schema.column_type(c) or sqltypes.VARCHAR for c in columns}
        )


#: PyPSA's `build_year` and `lifetime` where a record declares no default.
_ACTIVE_FALLBACK = {"build_year": 0, "lifetime": math.inf}


#: The period a `GlobalConstraint` applies in, NULL applying in every period.
_INVESTMENT_PERIOD = "investment_period"


def _resolved(
    record: RecordLike, fallbacks: Mapping[str, float], dim: str = ENTITY
) -> pd.DataFrame:
    """Each attribute of `fallbacks` of each label of `dim`, a column each, indexed by label, its row, else the declared default, else its fallback.

    Over `entity` the declared default follows the label's `ctype`
    (`default_by`), over `global_constraint` it is the attribute's plain one.

    Raises
    ------
    ValueError
        If a label holds several values of one because the selection keeps a dim it varies over.

    """
    schema = record.schema
    if dim == ENTITY:
        frame = record.properties[ENTITY].project(col(ENTITY), col(CTYPE)).df()
    else:
        frame = record.properties[dim].project(col(dim)).df()
    labels = {dim: record.dimensions[dim]}
    over = Flags(frozenset(labels), frozenset(labels))
    columns = {}
    for attr, fallback in fallbacks.items():
        declared = pd.Series(None, index=frame.index, dtype=object)
        if attr in schema.attributes and dim == ENTITY:
            declared = frame[CTYPE].map(
                lambda c, attr=attr: schema.default_for(attr, **{CTYPE: c})
            )
        elif attr in schema.attributes:
            declared = pd.Series(schema.default_for(attr), index=frame.index)
        default = declared.where(declared.notna(), fallback).astype(float)
        rows = record.attributes.get(attr)
        held = pd.DataFrame(columns=[dim, "value"])
        if rows is not None:
            held = settle(rows, labels, over).df()
        kept = [c for c in held.columns if c in schema.dimensions and c != dim]
        if held[kept].notna().any(axis=None):
            msg = f"`{attr}` varies over a dim the selection keeps, of {kept}: select one label of it"
            raise ValueError(msg)
        by_label = held.set_index(dim)["value"]
        columns[attr] = frame[dim].map(by_label).astype(float).fillna(default)
    return pd.DataFrame(columns).set_axis(frame[dim])


def _applying(record: RecordLike, periods: list[Any]) -> set[Any]:
    """The `global_constraint` labels applying in one of `periods`, those whose `investment_period` is NULL or one of them.

    Raises
    ------
    ValueError
        If a constraint holds several `investment_period` because the selection keeps a dim it varies over.

    """
    resolved = _resolved(record, {_INVESTMENT_PERIOD: math.nan}, GLOBAL_CONSTRAINT)
    period = resolved[_INVESTMENT_PERIOD]
    return set(period.index[period.isna() | period.isin(periods)])


def _active(record: RecordLike, periods: list[Any]) -> set[Any]:
    """The entities PyPSA holds active in one of `periods` (`_active_in`), by `build_year` and `lifetime`.

    An entity with no row takes the declared default, else PyPSA's. A
    constraint's `investment_period` is not read here (`_applying`).

    Raises
    ------
    ValueError
        If an entity holds several values of one because the selection keeps a dim it varies over.

    """
    resolved = _resolved(record, _ACTIVE_FALLBACK)
    kept = _active_in(resolved["build_year"], resolved["lifetime"], periods)
    return set(kept.index[kept])


def _root(record: RecordLike) -> RecordLike:
    """The record `record` selects from, itself where it is no `Selection`."""
    return record.base if isinstance(record, Selection) else record


def _axis_labels(record: RecordLike, dim: str) -> list[Any]:
    """`dim`'s labels in axis order, none where the record holds no axis of it."""
    axis = record.dimensions.get(dim)
    return [] if axis is None else [label for (label,) in axis.fetchall()]


def _active_in(
    start: pd.Series, lifetime: pd.Series, periods: Iterable[Any]
) -> pd.Series:
    """Whether PyPSA holds each active in one of `periods`, by `build_year <= period < build_year + lifetime`."""
    return reduce(
        lambda a, b: a | b, ((start <= p) & (p < start + lifetime) for p in periods)
    )


def _live(c: pypsa.Components, periods: Iterable[Any]) -> pd.Index:
    """The names of `c` PyPSA holds active in one of `periods` (`_active_in`), every name of a type without `build_year` and `lifetime`."""
    static = c.static
    if not {*_ACTIVE_FALLBACK} <= set(static.columns):
        return _names(static.index).unique()
    active = _active_in(static["build_year"], static["lifetime"], periods)
    return _names(static.index[active.to_numpy()]).unique()


@dataclass(frozen=True)
class _OpenedSource(_NetworkSource):
    """A network opened from the record it is written over, listing only what it materialised.

    A type is touched when `loaded` held an attribute of it at construction. Only a
    touched type's attributes and results are listed, `_differing` still decides the rows.
    """

    #: Type name -> its `loaded_attributes`, snapshotted before any check reads a frame.
    loaded: Mapping[str, frozenset[str]] = field(kw_only=True)

    def _pending(self, c: pypsa.Components, attr: str) -> bool:
        """Whether `attr` is a series announced but not loaded."""
        return isinstance(c.dynamic, LazyDict) and attr in c.dynamic.pending

    def _piecewise_pending(self, c: pypsa.Components, attr: str) -> bool:
        """Whether `attr`'s curves are announced but not loaded."""
        return isinstance(c.piecewise, LazyDict) and attr in c.piecewise.pending

    def _touched(self, c: pypsa.Components) -> bool:
        return bool(self.loaded[c.name])

    @property
    def _record(self) -> RecordLike:
        """The record the network was opened from."""
        assert self.base is not None  # noqa: S101
        return self.base

    @cached_property
    def _untouched_types(self) -> frozenset[str]:
        return untouched_types(self.loaded)

    @property
    def _untouched(self) -> Collection[str]:
        return self._untouched_types

    @cached_property
    def _attribute_names(self) -> dict[str, list[str]]:
        """Input attribute -> the touched types carrying it loaded."""
        names: dict[str, list[str]] = {}
        for c in self.n.components:
            if not self._touched(c) or c.name == _SHAPE or not _exported(c):
                continue
            per_port = self._port_stems(c)
            loaded = {
                record_input(c.name, per_port.get(a, a)) for a in self.loaded[c.name]
            }
            for attr in self._record_attributes(c):
                if attr in loaded:
                    names.setdefault(attr, []).append(c.name)
        for attribute, weighting in _WEIGHTINGS.items():
            if len(self._weighting_table(weighting.dim)):
                names[attribute] = []
        return names

    @cached_property
    def _results(self) -> dict[str, dict[str, str]]:
        """Touched exported type -> its `_result_attributes` among the loaded ones."""
        inputs = _input_names()
        return {
            c.name: {
                column: name
                for column, name in _result_attributes(c, inputs).items()
                if column in self.loaded[c.name]
            }
            for c in self.n.components
            if self._touched(c) and c.name != _SHAPE and _exported(c)
        }

    # -- an untouched type's rows come from `base`, its frame unread --------

    @cached_property
    def _shapes(self) -> dict[str, pd.DataFrame]:
        """The network's shapes as properties of the labels they outline."""
        shape = self.n.c[_SHAPE]
        if _SHAPE in self._untouched and not len(shape._static_names()):
            return {}
        return _outlined(self.n, self.con, self._untouched)

    @cached_property
    def _carried(self) -> dict[str, dict[str, str]]:
        """Each untouched type's custom columns and dtypes, as `base`'s schema declares them."""
        schema = self._record.schema
        registry = _registry_schema()
        meta = schema.meta.get(_CUSTOM) or {}
        carried: dict[str, dict[str, str]] = {}
        for c in self.n.components:
            if c.name not in self._untouched:
                continue
            dim = _dim_of(c.name)
            properties = schema.properties.get(dim, {})
            if c.name == _SHAPE:
                names: list[str] = []
            elif dim == ENTITY:
                names = list(meta.get(c.name, []))
            else:
                names = [
                    name
                    for name in properties
                    if name not in registry.properties.get(dim, {})
                    and not name.startswith(_SHAPE_PREFIX)
                ] + [
                    name
                    for name, spec in schema.attributes.items()
                    if name not in registry.attributes and dim in spec.dims
                ]
            carried[c.name] = {
                name: (
                    properties[name].dtype
                    if name in properties
                    else schema.attributes[name].dtype
                )
                or "str"
                for name in names
            }
        return carried

    @cached_property
    def custom(self) -> _Custom:
        """The network's custom static columns, an untouched type's from `base`."""
        return _Custom.of(self.n, self._carried)

    @cached_property
    def _base_entities(self) -> pd.DataFrame:
        """`base`'s entities with their `ctype` and properties."""
        return self._record.properties[ENTITY].df()

    @cached_property
    def _base_connections(self) -> pd.DataFrame:
        """`base`'s connections as `(entity, port, bus)` rows with the `ctype` of each entity."""
        rel = self._record.properties.get(CONNECTION)
        if rel is None:
            return pd.DataFrame(columns=[ENTITY, PORT, BUS, CTYPE])
        rows = rel.project(
            fn.struct_extract(col(CONNECTION), lit(ENTITY)).alias(ENTITY),
            fn.struct_extract(col(CONNECTION), lit(PORT)).alias(PORT),
            col(BUS),
        ).df()
        owner = self._base_entities.set_index(ENTITY)[CTYPE]
        return rows.assign(**{CTYPE: rows[ENTITY].map(owner)})

    def _connections_of(self, c: pypsa.Components) -> pd.DataFrame:
        if c.name not in self._untouched:
            return super()._connections_of(c)
        rows = self._base_connections
        return rows[rows[CTYPE] == c.name][[ENTITY, PORT, BUS]]

    def _static_frame(self, ctype: str) -> pd.DataFrame:
        if ctype not in self._untouched:
            return super()._static_frame(ctype)
        c = self.n.c[ctype]
        held = {*c.defaults.index, *self._carried[ctype]}
        rows = self._base_entities
        rows = rows[rows[CTYPE] == ctype]
        columns = [
            p for p in self.schema.properties[ENTITY] if p in held and p in rows.columns
        ]
        return rows[[ENTITY, *columns]].reset_index(drop=True)

    def _labels(self, ctype: str) -> pd.DataFrame:
        if ctype not in self._untouched:
            return super()._labels(ctype)
        dim = _AXIS_TYPES[ctype]
        rel = self._record.properties.get(dim)
        rows = pd.DataFrame(columns=[dim]) if rel is None else rel.df()
        columns = [
            p
            for p in self.schema.properties[dim]
            if p in rows.columns and not p.startswith(_SHAPE_PREFIX)
        ]
        return self._with_shapes(_none_for_empty(rows[[dim, *columns]], dim), dim)


@dataclass(frozen=True)
class _ResultsSource(_NetworkSource):
    """A solved `pypsa.Network`'s results, as a thin layer over `base`.

    Its results, at their default too, `base`'s result rows at labels it
    holds that it does not write again as NULL, and the `sub_network` axis
    restated whole, unless `_left_to_base`. A series is restated whole, as
    `snapshot` is outside `partial`, `base`'s values standing at the snapshots
    the network lacks (`_elsewhere`); over a multi-period `base` a value over
    no snapshot is written for the entities active in the network's periods
    alone (`_live_entities`). Over a selection, every row is at the labels
    it squeezes of PyPSA's dims (`_lifted`).

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/reference/tools/#writing-results)
    - [lifting rows back](https://energy-models.github.io/datarecord/reference/selecting/#lift)

    """

    #: The record the network was built from, or a selection of it at every snapshot (`_untimed`).
    base: RecordLike
    #: The period `base`'s selection squeezed, at which a single-period network is written.
    period: Any = None

    _DROP_RESULT_DEFAULTS: ClassVar[bool] = False
    _LEAVES_OUT_REPEATS: ClassVar[bool] = False

    @cached_property
    def schema(self) -> RecordSchema:
        """The schema of the record `base` selects from, as a results layer declares nothing the record does not."""
        return _root(self.base).schema

    def _series_checked(self, c: pypsa.Components) -> list[str]:
        """The result series of `c`, as a results layer writes no input series."""
        return [attr for attr in self._results.get(c.name, ()) if attr in c.dynamic]

    @cached_property
    def _shapes(self) -> dict[str, pd.DataFrame]:
        """None, as a results layer writes no shape."""
        return {}

    @cached_property
    def _periods(self) -> list[Any]:
        """The investment periods of a multi-period `base` the network solves, its own else `period`.

        Raises
        ------
        ValueError
            If it carries none and is written at none.

        """
        if len(self.n.investment_periods) > 0:
            return list(self.n.investment_periods)
        if self.period is None:
            periods = _axis_labels(_root(self.base), PERIOD)
            msg = (
                f"A network written over a multi-period `base` of periods {periods} "
                "carries investment periods of it, or is written over the "
                "selection at one period it was built from, "
                "`record.select(period=...)`; else set them with "
                "`n.set_investment_periods`"
            )
            raise ValueError(msg)
        return [self.period]

    @cached_property
    def _snapshots(self) -> pd.DataFrame:
        """`n.snapshots` a column per level of `base`'s `snapshot`, over a multi-period `base` a single-period network's timesteps at its period (`_periods`).

        Raises
        ------
        ValueError
            If `n` is multi-period and `base` is not.

        """
        snapshots = self.n.snapshots
        levels = isinstance(snapshots, pd.MultiIndex)
        if levels and not _multiperiod(self.base):
            msg = "A multi-period network is written over a single-period `base`"
            raise ValueError(msg)
        if levels or not _multiperiod(self.base):
            return snapshots.to_frame(index=False)
        return pd.DataFrame({PERIOD: self._periods[0], TIMESTEP: snapshots})

    def check(self) -> None:
        """Refuse a network holding a snapshot, bus or entity `base` lacks, or first-stage results of fewer scenarios than the record, at the call rather than the write.

        Raises
        ------
        ValueError
            If it holds one.

        """
        self._refuse_first_stage()
        snapshots = _lacking(
            self._snapshot_rel, self.base.dimensions[SNAPSHOT], SNAPSHOT
        )
        held = {
            BUS: self._buses[BUS],
            ENTITY: self._entity_frame[ENTITY],
            GLOBAL_CONSTRAINT: self._labelled[GLOBAL_CONSTRAINT][GLOBAL_CONSTRAINT],
        }
        unknown = {
            SNAPSHOT: [label for (label,) in snapshots.fetchall()],
            **{
                dim: labels[~labels.isin(_axis_labels(self.base, dim))].tolist()
                for dim, labels in held.items()
            },
        }
        for dim, labels in unknown.items():
            if labels:
                msg = f"The network has {dim} labels `base` lacks: {labels}"
                raise ValueError(msg)

    def _refuse_first_stage(self) -> None:
        """Refuse results over no `scenario`, such as `p_nom_opt`, of a network holding fewer scenarios than the record.

        Such a result is the decision for the scenarios solved alone, which
        would stand beside the other scenarios' results solved jointly.

        Raises
        ------
        ValueError
            If the network writes one.

        """
        scenarios = _axis_labels(_root(self.base), SCENARIO)
        held = len(self.n.scenarios) if self.n.has_scenarios else 1
        if len(scenarios) <= held:
            return
        attributes = self.schema.attributes
        first = sorted(a for a in self._restated if SCENARIO not in attributes[a].dims)
        if first:
            msg = (
                f"The network holds {held} of the record's scenarios {scenarios}, "
                f"so its first-stage results {first}, over no `scenario`, would "
                "stand for every scenario: solve every scenario together, or "
                "write a subset solved apart as a collection over a dim of its own"
            )
            raise ValueError(msg)

    @cached_property
    def _elsewhere(self) -> DuckDBPyRelation | None:
        """`base`'s snapshots the network lacks, at which `base`'s series stand, `None` where it holds them all."""
        rest = _lacking(self.base.dimensions[SNAPSHOT], self._snapshot_rel, SNAPSHOT)
        return rest if rest.limit(1).fetchall() else None

    @cached_property
    def _live_entities(self) -> DuckDBPyRelation | None:
        """The entities active in `_periods` (`_live`), to which a value over no snapshot is written, `None` over a single-period `base`."""
        if not _multiperiod(self.base):
            return None
        names = [
            name
            for c in self.n.components
            if _as_entities(c)
            for name in _live(c, self._periods)
        ]
        frame = pd.DataFrame({ENTITY: pd.Series(names, dtype=object)})
        return cast_declared(self.schema, self.con.from_df(frame))

    @cached_property
    def _holds(self) -> dict[str, DuckDBPyRelation]:
        """The labels the network holds of each dim a result is keyed by, to which it restates `base`'s results.

        The `global_constraint` labels are those `base` holds, over a
        selection those it keeps: a constraint the network lacks among them
        is left NULL, one the selection drops stays as `base` holds it.
        """
        frames = {
            ENTITY: self._entity_frame[[ENTITY]],
            BUS: self._buses[[BUS]],
            PORT: self._ports,
        }
        if self.n.has_scenarios:
            frames[SCENARIO] = self.n.scenarios.to_frame(index=False)
        labels = {
            d: cast_declared(self.schema, self.con.from_df(f))
            for d, f in frames.items()
        }
        held = {**labels, CONNECTION: self._connection_frame().project(col(CONNECTION))}
        kept = self._selected_constraints
        if kept is not None:
            frame = pd.DataFrame({GLOBAL_CONSTRAINT: pd.Series(kept, dtype=object)})
            held[GLOBAL_CONSTRAINT] = cast_declared(
                self.schema, self.con.from_df(frame)
            )
        return held

    @cached_property
    def _selected_constraints(self) -> list[Any] | None:
        """The `global_constraint` labels `base` holds that apply in the network's periods, none where `base` holds no axis of them.

        Over a multi-period `base` a constraint whose `investment_period` is
        none of them is one a period selection drops (`_applying`),
        which raises where the constraint's `investment_period` varies over a
        dim `base` keeps.
        """
        if GLOBAL_CONSTRAINT not in self.base.dimensions:
            return None
        labels = _axis_labels(self.base, GLOBAL_CONSTRAINT)
        if not _multiperiod(self.base):
            return labels
        if set(self._periods) >= set(_axis_labels(_root(self.base), PERIOD)):
            return labels
        applying = _applying(self.base, self._periods)
        return [g for g in labels if g in applying]

    @property
    def dimensions(self) -> LazyRelations:
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
        """The axes this layer restates, `sub_network` where the solve or the record has one and it is not left to `base` (`_left_to_base`)."""
        axis = _root(self.base).dimensions.get(SUB_NETWORK)
        held = len(self._sub_networks) > 0 or (
            axis is not None and bool(axis.limit(1).fetchall())
        )
        return (SUB_NETWORK,) if held and not self._left_to_base else ()

    @cached_property
    def _left_to_base(self) -> frozenset[str]:
        """The results this layer leaves to `base`, the `sub_network` axis's where the record holds a bus or entity the network lacks.

        A selection's SubNetworks are numbered over the selection alone, so
        they name no SubNetwork of the whole network that `base`'s rows at
        the labels it does not hold refer to.
        """
        root = _root(self.base)
        held = {BUS: self._buses[BUS], ENTITY: self._entity_frame[ENTITY]}
        if all(set(_axis_labels(root, d)) <= set(labels) for d, labels in held.items()):
            return frozenset()
        return _sub_network_results(self.schema)

    def _sub_network_rel(self) -> DuckDBPyRelation:
        """`_sub_networks` with its columns typed as the schema declares them, not as an empty frame infers them."""
        rel = self.con.from_df(self._sub_networks)
        return cast_declared(self.schema, rel, SUB_NETWORK)

    def _dead_sub_networks(self) -> DuckDBPyRelation | None:
        """The record's SubNetworks this solve does not number, aliased `d`, `None` where it has no axis."""
        axis = _root(self.base).dimensions.get(SUB_NETWORK)
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
        """This solve's results but those `_left_to_base`, then each row of `base`'s at labels it holds (`_holds`) it leaves out as NULL, but those of a SubNetwork it drops, and `base`'s series settled at `_elsewhere`.

        A value over no snapshot is kept to `_live_entities` on both sides,
        and every row is at the labels `base` squeezes (`_lifted`).
        """
        kept = self._left_to_base
        written = {
            a: self._live_only(a, self._lifted(a, r))
            for a, r in self._result_rels.items()
            if a not in kept
        }
        outputs = _registry_schema().attributes.keys() - _input_names() - kept
        dead = self._dead_sub_networks()
        restated = dict(written)
        for attr in sorted(outputs & set(self.base.attributes)):
            columns = self.schema.long_columns_for(attr)
            keys = [c for c in columns if c != "value"]
            rows = self._live_only(attr, self._base_rows(attr))
            for dim in self._holds.keys() & set(keys):
                rows = _null_or_in(rows, col(dim), self._holds[dim], dim)
            parts = [written[attr]] if attr in written else []
            elsewhere = self._elsewhere
            if SNAPSHOT in keys and elsewhere is not None:
                over = frozenset({SNAPSHOT})
                parts.append(settle(rows, {SNAPSHOT: elsewhere}, Flags(over, over)))
                rows = rows.set_alias("r").join(
                    elsewhere.set_alias("e"),
                    col("r", SNAPSHOT) == col("e", SNAPSHOT),
                    how="anti",
                )
            stale = _stale(rows, written.get(attr), keys)
            if dead is not None and SUB_NETWORK in keys:
                stale = stale.set_alias("s").join(
                    dead, col("s", SUB_NETWORK) == col("d", SUB_NETWORK), how="anti"
                )
            null = lit(None).cast(self.schema.duck_type(self.schema.attributes[attr]))
            parts.append(
                stale.project(
                    *(null.alias(c) if c == "value" else col(c) for c in columns)
                )
            )
            restated[attr] = reduce(DuckDBPyRelation.union, parts)
        return restated

    def _lifted(self, attr: str, rows: DuckDBPyRelation) -> DuckDBPyRelation:
        """The network's `rows` of `attr` at the labels `base` squeezes of PyPSA's dims, such as `scenario`, rather than NULL there.

        A dim of the record's own stays NULL (`_as_declared`).
        """
        if not isinstance(self.base, Selection):
            return rows
        columns = self.base.schema.long_columns_for(attr)
        lifted = self.base.lift(attr, rows.select(*map(col, columns)))
        return lifted.select(
            *self._as_declared(attr, self.schema.long_columns_for(attr))
        )

    def _base_rows(self, attr: str) -> DuckDBPyRelation:
        """`base`'s rows of `attr`, as `_lifted` puts the network's."""
        rows = self.base.attributes[attr]
        if isinstance(self.base, Selection):
            rows = self.base.lift(attr, rows)
        return rows.select(*self._as_declared(attr, self.schema.long_columns_for(attr)))

    def _as_declared(self, attribute: str, columns: Iterable[str]) -> list[Expression]:
        """`columns` of `attribute`'s rows, each PyPSA's schema lacks NULL: the network's value at every label."""
        held = _registry_schema().long_columns_for(attribute)

        def everywhere(dim: str) -> Expression:
            declared = self.schema.duck_type(self.schema.dimensions[dim])
            return lit(None).cast(declared).alias(dim)

        return [col(c) if c in held else everywhere(c) for c in columns]

    def _live_only(self, attr: str, rows: DuckDBPyRelation) -> DuckDBPyRelation:
        """`rows` of `attr`, a value over no snapshot only at `_live_entities` or NULL on `entity`."""
        live = self._live_entities
        if live is None or SNAPSHOT in rows.columns:
            return rows
        if ENTITY in rows.columns:
            on = col(ENTITY)
        elif CONNECTION in rows.columns:
            on = fn.struct_extract(col(CONNECTION), lit(ENTITY))
        else:
            return rows
        return _null_or_in(rows, on, live, ENTITY)


def _null_or_in(
    rows: DuckDBPyRelation, on: Expression, among: DuckDBPyRelation, key: str
) -> DuckDBPyRelation:
    """The `rows` whose `on` is NULL or among the values of `among`'s `key`, as SQL's `IN` keeps them."""
    listed = among.project(col(key).alias("_among")).distinct()
    joined = rows.join(listed, on == col("_among"), how="left")
    return joined.filter(on.isnull() | col("_among").isnotnull()).project(
        star(exclude=["_among"])
    )


def _stale(
    rows: DuckDBPyRelation, written: DuckDBPyRelation | None, keys: Sequence[str]
) -> DuckDBPyRelation:
    """The distinct `keys` of `rows` that `written` does not write again.

    A row is written again where a row of `written` covers it (`uncovered`),
    as a row NULL on `scenario` covers `base`'s at every scenario. A row
    over no snapshot is stale where no row over no snapshot covers it. A row
    at a snapshot is stale where none covers it but on `snapshot`, since a
    melted series writes every snapshot of each key it holds.
    """
    if written is None:
        return rows.select(*map(col, keys)).distinct()
    if SNAPSHOT not in keys:
        return uncovered(rows.select(*map(col, keys)).distinct(), written, keys)
    whole = [k for k in keys if k != SNAPSHOT]
    over_none = col(SNAPSHOT).isnull()
    held = rows.select(*map(col, whole), over_none.alias("_over_none")).distinct()
    statics = written.filter(over_none).select(*map(col, whole)).distinct()
    series = written.select(*map(col, whole)).distinct()
    stale = uncovered(held.filter(col("_over_none")), statics, whole).union(
        uncovered(held.filter(~col("_over_none")), series, whole)
    )
    on = null_safe("r", "s", whole) & (
        col("r", SNAPSHOT).isnull() == col("s", "_over_none")
    )
    return (
        rows.set_alias("r")
        .join(stale.set_alias("s"), on, how="semi")
        .select(*map(col, keys))
        .distinct()
    )


def _sub_network_results(schema: RecordSchema) -> frozenset[str]:
    """The results naming a SubNetwork, a line's `sub_network` and every attribute over the `sub_network` axis."""
    return frozenset(
        name
        for name, spec in schema.attributes.items()
        if name == SUB_NETWORK or SUB_NETWORK in spec.dims
    )


def _lacking(
    rel: DuckDBPyRelation, other: DuckDBPyRelation, dim: str
) -> DuckDBPyRelation:
    """The rows of `rel` whose label of `dim` `other` lacks."""
    return rel.set_alias("a").join(
        other.set_alias("b"), col("a", dim) == col("b", dim), how="anti"
    )


def _held(
    relations: Mapping[Any, LazyRelations], key: str
) -> dict[Any, DuckDBPyRelation]:
    """`key`'s relation in each member's `relations` that holds it."""
    return {label: rels[key] for label, rels in relations.items() if key in rels}


def _keys(relations: Mapping[Any, LazyRelations]) -> tuple[str, ...]:
    """Every member's keys of `relations`, in the order first held."""
    return tuple(dict.fromkeys(k for rels in relations.values() for k in rels))


def _differs(held: Mapping[Any, DuckDBPyRelation], labels: Sequence[Any]) -> bool:
    """Whether an attribute's rows, `held` by each member holding it, differ across the members' `labels`.

    Each member is stacked with the first in DuckDB, stopping at the first
    that differs, where a row is held by one alone or its values clash
    (`_clash`). A missing key is a label like any other.
    """
    if any(label not in held for label in labels):
        return True
    first, *rest = held.values()
    keys = [col(c) for c in first.columns if c != "value"]

    def apart(other: DuckDBPyRelation) -> bool:
        stacked = first.union(other.select(*map(col, first.columns)))
        clash = ((fn.count() < lit(2)) | _clash(stacked)).alias("clash")
        clashing = stacked.group_by(*keys).agg(clash).filter(col("clash"))
        return bool(clashing.limit(1).fetchall())

    return any(apart(other) for other in rest)


def _widened(schema: RecordSchema, dim: str, differing: Container[str]) -> RecordSchema:
    """`schema` with each of `differing` over `dim`."""
    attributes = {
        name: spec.model_copy(update={"dims": spec.dims | {dim}})
        if name in differing
        else spec
        for name, spec in schema.attributes.items()
    }
    return RecordSchema(**{**schema.model_dump(), "attributes": attributes})


def _labelled(
    schema: RecordSchema, attr: str, held: Mapping[Any, DuckDBPyRelation], dim: str
) -> DuckDBPyRelation:
    """Each member's rows of `attr`, an attribute over `dim`, at its label."""
    columns = schema.long_columns_for(attr)
    label_type = schema.duck_type(schema.dimensions[dim])

    def at(label: Any, rel: DuckDBPyRelation) -> DuckDBPyRelation:
        labelled = lit(label).cast(label_type).alias(dim)
        rows = rel.select(*(labelled if c == dim else col(c) for c in columns))
        return cast_declared(schema, rows, attribute=attr)

    return reduce(DuckDBPyRelation.union, (at(*item) for item in held.items()))


@dataclass(frozen=True)
class _StackedSource:
    """A network written over a record with a dim of its own, its rows at the label it was built from (`of`)."""

    schema: RecordSchema
    dimensions: Mapping[str, DuckDBPyRelation]
    properties: Mapping[str, DuckDBPyRelation]
    attributes: Mapping[str, DuckDBPyRelation]
    con: DuckDBPyConnection

    @classmethod
    def of(
        cls,
        members: Mapping[Any, _NetworkSource],
        dim: str,
        con: DuckDBPyConnection,
        base: RecordLike | None,
    ) -> _StackedSource:
        """The one member of `members`, its source by its label of `dim`, as a record over `base`.

        Over a `base` holding labels of `dim` the member lacks, every attribute
        is at its label, the `dim` axis and the results naming a SubNetwork
        (`_sub_network_results`) left to `base`, since the member numbers its
        SubNetworks over itself alone. An attribute whose rows differ from the
        labels' (`_differs`) is over `dim`, any other is held once.
        """
        ((label, member),) = members.items()
        schema = member.schema
        labels = list(members)
        whole = [] if base is None else _axis_labels(_root(base), dim)
        apart = len(labels) < len(whole)
        left = _sub_network_results(schema) if apart else frozenset()
        held_dims = {label: member.dimensions}
        dimensions = {
            d: member.dimensions[d]
            for d in _keys(held_dims)
            if d != dim and d not in left
        }
        properties = {
            s: member.properties[s]
            for s in _keys({label: member.properties})
            if s not in left
        }
        attributes = {label: member.attributes}
        names = [a for a in _keys(attributes) if a not in left]
        held = {a: _held(attributes, a) for a in names}
        if apart:
            differing = set(names)
        else:
            differing = {a for a in names if _differs(held[a], labels)}
        schema = _widened(schema, dim, differing)
        if not apart:
            axis = con.from_df(pd.DataFrame({dim: labels}))
            dimensions[dim] = cast_declared(schema, axis, dim)
        rows = {
            attr: _labelled(schema, attr, held[attr], dim)
            if attr in differing
            else next(iter(held[attr].values()))
            for attr in names
        }
        return cls(schema, dimensions, properties, rows, con)

    def flags(self, **categories: str | Sequence[str]) -> dict[str, Flags]:
        """Never consulted, as `_NetworkSource.flags`."""
        return {}
