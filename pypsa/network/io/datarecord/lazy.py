# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""A record opened as the lazy source of a network's components."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from datarecord.protocols import Flags, RecordLike
from datarecord.schema.categories import with_defaults
from datarecord.tools.base import Schema
from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from duckdb import StarExpression as star

from pypsa.components.source import InMemorySource
from pypsa.network.io._common import _coerce_string_dtypes
from pypsa.network.io.datarecord.build import (
    _assign_static,
    _collapse_connections,
    _one_labelled,
    _port_columns,
    _series_frame,
    _squeezed,
    _tuples_of,
    _TypeMembers,
    _unpack_connection,
)
from pypsa.network.io.datarecord.schema import (
    _AXIS_TYPES,
    _CATALOG_TYPES,
    CONNECTION,
    ENTITY,
    GLOBAL_CONSTRAINT,
    PERIOD,
    SCENARIO,
    SNAPSHOT,
    _shape_properties,
    _taken,
    pypsa_attribute,
)
from pypsa.network.io.datarecord.shape import Index, NetworkShape
from pypsa.network.io.datarecord.source import (
    _INVESTMENT_PERIOD,
    _empty_for_none,
    _keyed_as_entity,
    _names,
)
from pypsa.network.io.mixin import _ordered_static, _typed_static

if TYPE_CHECKING:
    import pypsa
    from pypsa.definitions.components import ComponentType

_Attributes = dict[str, tuple[DuckDBPyRelation, Flags]]


class RecordSource:
    """The frames of one opened record, each assembled on first request.

    Keeps unmaterialised DuckDB plans per component type, and turns one into a
    conformed pandas frame per `static` or `dynamic` call.
    """

    def __init__(
        self,
        record: RecordLike,
        n: pypsa.Network,
        shape: NetworkShape,
        con: DuckDBPyConnection,
        members: Mapping[str, _TypeMembers],
        flagged: Mapping[str, Flags],
        schema: Schema,
        *,
        constraintless: bool = False,
    ) -> None:
        """Open `record` over `n`, whose axes are already applied.

        `constraintless` holds none of the record's global constraints, as a
        selection cannot name no label of the axis (`_narrow`).
        """
        self.record = record
        self._constraintless = constraintless
        self._n = weakref.ref(n)
        self._shape = shape
        self._con = con
        self._members = members
        self._flagged = flagged
        self._schema = schema
        self._cache: dict[str, tuple[DuckDBPyRelation, _Attributes]] = {}
        #: Series-only columns a record supplied as static-valued rows, by type,
        #: left for `dynamic` to broadcast.
        self._broadcast: dict[str, pd.DataFrame] = {}
        #: The index `static` supplied, by type: the members the record holds.
        self._held: dict[str, pd.Index] = {}
        #: Curve rows by type and PyPSA column, read only by `piecewise`.
        self._curves: dict[str, dict[str, DuckDBPyRelation]] = {}

    # -- the Source protocol -------------------------------------------------

    def types(self) -> frozenset[str]:
        """Component type names the record has members of, axis types included."""
        axes = {
            ctype
            for ctype, dim in _AXIS_TYPES.items()
            if self.record.properties.get(dim) is not None
        }
        entities = {
            ctype for ctype, m in self._members.items() if m.members is not None
        }
        return frozenset(axes | entities)

    def static(self, ctype: ComponentType) -> pd.DataFrame:
        """Return the conformed static frame of `ctype`."""
        self._check(ctype)
        n = self._network()
        static, attributes = self._attributes(ctype.name)
        raw = _assign_static(
            static,
            attributes,
            self._shape,
            Index.of(n.scenarios, self._con),
            self._con,
        )
        attrs = ctype.defaults
        typed, series = _typed_static(raw, attrs)
        self._held[ctype.name] = typed.index
        if len(series.columns):
            self._broadcast[ctype.name] = series
        ordered = _ordered_static(typed, attrs, ctype.name, n.crs)
        if ctype.name in _CATALOG_TYPES:
            return self._with_library(ordered, ctype, n)
        if ctype.name == "SubNetwork":
            return self._with_objects(ordered, n)
        return ordered

    @staticmethod
    def _with_library(
        own: pd.DataFrame, ctype: ComponentType, n: pypsa.Network
    ) -> pd.DataFrame:
        """`own` after PyPSA's standard type library, which a record never holds, in every scenario."""
        library = ctype.standard_types
        if library is None:
            return own
        # `n.add` takes a library's missing value as the attribute's default.
        defaults = ctype.defaults["default"]
        library = library.fillna(
            {col: defaults[col] for col in library.columns if col in defaults.index}
        )
        if not n.has_scenarios:
            both = pd.concat([library, own])
            both.index = both.index.astype(object)
            return both
        scenarios = list(n.scenarios)
        both = pd.concat(
            [pd.concat(dict.fromkeys(scenarios, library), names=[SCENARIO]), own]
        )
        rank = both.index.get_level_values(SCENARIO).map(scenarios.index)
        return both.iloc[np.argsort(np.asarray(rank), kind="stable")]

    @staticmethod
    def _with_objects(static: pd.DataFrame, n: pypsa.Network) -> pd.DataFrame:
        """`static` with each SubNetwork's `obj`, the `_DERIVED` column PyPSA's topology code reads on every row."""
        import pypsa

        names = _names(static.index)
        return static.assign(obj=[pypsa.SubNetwork(n, name) for name in names])

    def dynamic(self, ctype: ComponentType, attr: str) -> pd.DataFrame:
        """Return the time-varying frame of `attr`, one column per component with data."""
        self._check(ctype)
        n = self._network()
        _, attributes = self._attributes(ctype.name)
        defaults = ctype.defaults
        registered = attr in defaults.index
        series_only = registered and not bool(defaults.at[attr, "static"])

        static_index = n.c[ctype.name].static.index
        held = self._held.get(ctype.name)
        if held is not None:
            # Only the record's own members: a component added after open has
            # no output column, as after `build` then `add`.
            static_index = static_index[static_index.isin(held)]
        snapshots = Index.of(n.snapshots, self._con, self._shape.snapshots)
        carried = attributes.get(attr)
        if carried is not None and SNAPSHOT in carried[1].varies:
            rows = carried[0].filter(
                col(SNAPSHOT).isnotnull() & col("value").isnotnull()
            )
            # Partial columns keep non-series components static.
            wide = _series_frame(rows, self._shape, snapshots, static_index, self._con)
        else:
            wide = pd.DataFrame(index=snapshots.index, columns=static_index[:0])
        wide = wide.astype(float) if wide.shape[1] == 0 else wide

        frame = wide
        if series_only and carried is not None:
            fill = float(defaults.at[attr, "default"])
            frame = pd.DataFrame(
                fill, index=snapshots.index, columns=static_index, dtype=float
            )
            broadcast = self._broadcast.get(ctype.name)
            if broadcast is not None and attr in broadcast:
                frame.loc[:, :] = np.tile(
                    broadcast[attr].reindex(static_index).to_numpy(dtype=float),
                    (len(snapshots.index), 1),
                )
            if wide.shape[1]:
                frame.loc[:, wide.columns] = wide.to_numpy()
        frame.columns.names = (
            ["scenario", "name"]
            if isinstance(frame.columns, pd.MultiIndex)
            else ["name"]
        )
        return _coerce_string_dtypes(frame)

    def piecewise(self, ctype: ComponentType, attr: str) -> pd.DataFrame:
        """Return the breakpoint frame of `attr`, read from the record's curve rows."""
        self._check(ctype)
        self._attributes(ctype.name)
        rows = self._curves.get(ctype.name, {}).get(attr)
        if rows is None:
            return InMemorySource().piecewise(ctype, attr)
        n = self._network()
        c = n.c[ctype.name]
        order = c.static.index
        long = rows.project(f"{ENTITY}, breakpoint, value").df()
        # Rows of another type's entities.
        long = long[long[ENTITY].isin(order)]
        if long.empty:
            return InMemorySource().piecewise(ctype, attr)
        schema_row = c._piecewise_schema(attr)
        curves = {
            name: group.sort_values("breakpoint", kind="mergesort")
            for name, group in long.groupby(ENTITY, sort=False)
        }
        depth = max(len(g) for g in curves.values())

        def padded(values: pd.Series) -> np.ndarray:
            out = np.full(depth, np.nan)
            out[: len(values)] = values.to_numpy(dtype=float)
            return out

        columns: dict[tuple[str, str], np.ndarray] = {}
        for name in order[order.isin(list(curves))]:
            columns[(name, schema_row.x)] = padded(curves[name]["breakpoint"])
            columns[(name, attr)] = padded(curves[name]["value"])
        frame = pd.DataFrame(columns, index=pd.RangeIndex(depth), dtype=float)
        frame.columns = pd.MultiIndex.from_tuples(
            frame.columns, names=["name", "attribute"]
        )
        return n._normalize_breakpoints(frame, schema_row)

    def curve_columns(self, ctype: str) -> list[str]:
        """The PyPSA columns of `ctype` that hold curve rows, `efficiency2` for a bus2 curve."""
        self._attributes(ctype)
        curves = self._curves.get(ctype, {})
        if not curves:
            return []
        probes = [
            rows.project(ENTITY).limit(1).project(lit(column).alias("column"))
            for column, rows in curves.items()
        ]
        found = probes[0]
        for probe in probes[1:]:
            found = found.union(probe)
        held = {column for (column,) in found.distinct().fetchall()}
        return [column for column in curves if column in held]

    def varying_attrs(self, ctype: ComponentType) -> list[str]:
        """Return the registry's varying attributes, then the record's own that vary over `snapshot`."""
        self._check(ctype)
        defaults = ctype.defaults
        _, attributes = self._attributes(ctype.name)
        extra = sorted(
            attr
            for attr, (_, flags) in attributes.items()
            if SNAPSHOT in flags.varies and attr not in defaults.index
        )
        return [*defaults.index[defaults.varying], *extra]

    def names(self, ctype: ComponentType) -> pd.Index:
        """Return `ctype`'s member names from the record's entity axis, no static frame built."""
        self._check(ctype)
        # `_check` admits only types whose members are not None.
        rel = (
            self._attributes(ctype.name)[0]
            if ctype.name in _AXIS_TYPES
            else cast("DuckDBPyRelation", self._members[ctype.name].members)
        )
        names = rel.project(ENTITY).fetchnumpy()[ENTITY]
        own = pd.Index(names, name="name", dtype=object).drop_duplicates()
        library = ctype.standard_types if ctype.name in _CATALOG_TYPES else None
        return own if library is None else library.index.append(own)

    # -- internals -------------------------------------------------------------

    def _check(self, ctype: ComponentType) -> None:
        if ctype.name not in self.types():
            raise KeyError(ctype.name)

    def _network(self) -> pypsa.Network:
        n = self._n()
        if n is None:
            msg = "the network this source belongs to no longer exists"
            raise ReferenceError(msg)
        return n

    def _attributes(self, ctype: str) -> tuple[DuckDBPyRelation, _Attributes]:
        """`ctype`'s static relation and `{pypsa attribute: (long rows, flags)}`, cached."""
        if ctype not in self._cache:
            build = self._axis if ctype in _AXIS_TYPES else self._entity
            self._cache[ctype] = build(ctype)
        return self._cache[ctype]

    def _axis(self, ctype: str) -> tuple[DuckDBPyRelation, _Attributes]:
        """An `_AXIS_TYPES` type: its dim's labels with defaults, and the attributes over the dim."""
        dim = _AXIS_TYPES[ctype]
        record, schema = self.record, self.record.schema
        outlines = _shape_properties(schema, dim)
        names = [name for name in schema.properties[dim] if name not in outlines]
        labels = with_defaults(schema, dim, record.properties[dim], names, self._con)
        if self._constraintless and dim == GLOBAL_CONSTRAINT:
            labels = labels.limit(0)
        static = _keyed_as_entity(_empty_for_none(labels, dim), dim).project(
            "*, row_number() OVER () AS _pos"
        )
        over = {**self._settled, dim: labels}
        # A single-period network has no period for `investment_period` to name.
        periodless = {_INVESTMENT_PERIOD} if PERIOD in _squeezed(record) else set()
        attributes = {
            pypsa_attribute(ctype, attr): (
                _keyed_as_entity(
                    self._schema.resolve(
                        record, ctype, attr, over, self._flagged, filter=False
                    ),
                    dim,
                ),
                flags,
            )
            for attr, flags in self._flagged.items()
            if attr in schema.attributes
            and dim in schema.attributes[attr].dims
            and attr not in periodless
        }
        return static, attributes

    def _entity(self, ctype: str) -> tuple[DuckDBPyRelation, _Attributes]:
        """An entity type: its members with `_pos`, connections collapsed, and its attributes."""
        record, schema = self.record, self.record.schema
        members = self._members[ctype]
        # `_check` admits only types whose members are not None.
        static = cast("DuckDBPyRelation", members.members)
        labels = {**self._settled, ENTITY: static}
        tuples = record.dimensions.get(CONNECTION)
        if tuples is not None:
            labels[CONNECTION] = _tuples_of(tuples, static)
        if not self._shape.stochastic and SCENARIO in static.columns:
            static = static.project(star(exclude=[SCENARIO]))
        # `_pos` records member order before joins scramble it, `_assign_static` sorts by it (https://energy-models.github.io/datarecord/design/layers/resolution/#one-fold-for-every-axis).
        static = static.project("*, row_number() OVER () AS _pos")
        connections = members.connections
        static = _collapse_connections(static, connections, ctype, self._con)

        over_entity = [a for a in schema.attributes if schema.addresses(a, ENTITY)]
        carried = _taken(schema, ctype, over_entity)
        attributes: _Attributes = {}
        for attr, flags in self._flagged.items():
            if attr not in carried:
                continue
            # A named row of another type's entity is left to the joins on
            # `entity` (https://energy-models.github.io/datarecord/design/storage/format/#a-label-is-unique-within-its-dim).
            long = _unpack_connection(
                self._schema.resolve(
                    record, ctype, attr, labels, self._flagged, filter=False
                )
            )
            curves = None
            if flags.breakpoints:
                curves = long.filter(col("breakpoint").isnotnull())
                long = long.filter(col("breakpoint").isnull())
            if CONNECTION in schema.attributes[attr].dims:
                for column, rows in _port_columns(
                    long, attr, connections, ctype, self._con
                ).items():
                    attributes[column] = (rows, flags)
                if curves is not None:
                    self._curves.setdefault(ctype, {}).update(
                        _port_columns(curves, attr, connections, ctype, self._con)
                    )
            else:
                column = pypsa_attribute(ctype, attr)
                attributes[column] = (long, flags)
                if curves is not None:
                    self._curves.setdefault(ctype, {})[column] = curves
        return static, attributes

    @cached_property
    def _settled(self) -> dict[str, DuckDBPyRelation]:
        """The labels rows are settled over, every dim of one label (`_one_labelled`) and the scenarios of a stochastic network."""
        settled = _one_labelled(self.record)
        if self._shape.stochastic:
            settled[SCENARIO] = self.record.dimensions[SCENARIO]
        return settled
