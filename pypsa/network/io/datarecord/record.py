# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""`NetworkRecord`: a `Network` presented lazily as a datarecord `Record`.

The export side of the datarecord format: `NetworkRecord` implements the
`datarecord.Record` protocol over a live `Network`, undoing PyPSA's
static/dynamic split into the record's wide member frames and long rows.
Validated eagerly on construction, so a network that cannot be represented
(name collisions, a component on the same bus twice, unsupported snapshots)
fails before any frame is built.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np
import pandas as pd
from datarecord.record import Flags, LazyFrames

from pypsa.network.index import _validate_level_dtype
from pypsa.network.io.datarecord.schema import (
    _EXCLUDED_TYPES,
    ENTITY_TYPE,
    PERIOD,
    PERIOD_WEIGHTINGS,
    PORT,
    SCENARIO,
    SCENARIO_WEIGHTINGS,
    TIMESTEP,
    build_schema,
    port_columns,
    record_name,
)

if TYPE_CHECKING:
    from datarecord.schema import Schema

    from pypsa import Network
    from pypsa.components.components import Components

# The network-attribute allow-list the datarecord path round-trips; anything
# else PyPSA carries stays off the record (no `dir(n)` scan).
NETWORK_ATTRS = ("name", "pypsa_version", "_multi_invest")

_ENTITY, _BUS, _CONNECTION = "entity", "bus", "connection"


class DatarecordExportError(ValueError):
    """A network cannot be exported to the datarecord format as-is."""


def _exported_components(n: Network) -> list[Components]:
    """Component types with data to write: non-empty, standard types excluded.

    `_EXCLUDED_TYPES` (templates and derived, non-schema types) are never
    exported: the schema grants no entity-type label for them.
    """
    return [
        c
        for c in n.components
        if not c.static.empty
        and c.name not in _EXCLUDED_TYPES
        and c.name not in n.standard_type_components
    ]


def _names(c: Components) -> pd.Index:
    """Return a type's component names, one per entity regardless of scenario."""
    index = c.static.index
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values("name").unique()
    return index


def _check_collisions(components: list[Components]) -> None:
    """Raise if a name is claimed by more than one exported type."""
    owners: dict[str, set[str]] = {}
    for c in components:
        for name in _names(c):
            owners.setdefault(str(name), set()).add(c.name)
    clashing = {name: sorted(types) for name, types in owners.items() if len(types) > 1}
    if clashing:
        detail = "; ".join(
            f"{name}: {', '.join(types)}" for name, types in sorted(clashing.items())
        )
        msg = f"names claimed by more than one component type: {detail}"
        raise DatarecordExportError(msg)


def _check_same_bus_twice(components: list[Components]) -> None:
    """Raise if a component attaches to the same bus on two of its ports."""
    offenders: list[str] = []
    for c in components:
        if not c.ports:
            continue
        static = c.static
        bus_cols = [f"bus{p}" for p in c.ports if f"bus{p}" in static.columns]
        if len(bus_cols) < 2:
            continue
        buses = static[bus_cols]
        if isinstance(buses.index, pd.MultiIndex):
            buses = buses.droplevel(SCENARIO)
            buses = buses[~buses.index.duplicated()]
        for name, row in buses.iterrows():
            seen: set[str] = set()
            for bus in row:
                bus = str(bus)
                if bus == "":
                    continue
                if bus in seen:
                    offenders.append(f"{c.name} {name} -> {bus}")
                seen.add(bus)
    if offenders:
        msg = f"component attached to the same bus on two ports: {'; '.join(offenders)}"
        raise DatarecordExportError(msg)


def _timestep_dtype_name(n: Network) -> str:
    """`Int64` or `Datetime`: the narwhals dtype name for the timestep level."""
    snapshots = n.snapshots
    level = (
        snapshots.get_level_values(TIMESTEP)
        if isinstance(snapshots, pd.MultiIndex)
        else snapshots
    )
    if level.dtype.kind in "iu":
        return "Int64"
    if isinstance(level, pd.DatetimeIndex):
        return "Datetime"
    msg = f"unsupported snapshot dtype {level.dtype!r}; must be integer- or datetime-typed"
    raise DatarecordExportError(msg)


def _check_snapshots(n: Network) -> None:
    """Raise if a snapshot or timestep level is neither integer nor datetime."""
    snapshots = n.snapshots
    try:
        if isinstance(snapshots, pd.MultiIndex):
            _validate_level_dtype(
                snapshots.get_level_values(PERIOD), PERIOD, datetime_allowed=False
            )
            _validate_level_dtype(
                snapshots.get_level_values(TIMESTEP), TIMESTEP, datetime_allowed=True
            )
        else:
            _validate_level_dtype(snapshots, TIMESTEP, datetime_allowed=True)
    except ValueError as e:
        raise DatarecordExportError(str(e)) from e


def _scalar(value: Any) -> Any:
    """Return a numpy scalar as a plain Python one, JSON-storable in `schema.meta`."""
    return value.item() if hasattr(value, "item") else value


def _default(value: Any) -> Any:
    """One default cell as JSON-storable, NaN as absent."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _drop_default(rows: pd.DataFrame, default: Any) -> pd.DataFrame:
    """Drop scalar rows whose value equals the attribute's default."""
    if pd.isnull(default):
        return rows[rows["value"].notna()]
    return rows[rows["value"] != default]


def _scenario_varying(c: Components, columns: list[str]) -> set[str]:
    """Which static columns hold more than one value across scenarios."""
    index = c.static.index
    if not isinstance(index, pd.MultiIndex) or SCENARIO not in (index.names or []):
        return set()
    by_entity = c.static.groupby(level="name")
    return {x for x in columns if (by_entity[x].nunique(dropna=False) > 1).any()}


class NetworkRecord:
    """A `Network` presented as a datarecord `Record` (export only).

    Validates eagerly on construction and raises `DatarecordExportError` for a
    shape the record cannot hold: names claimed by more than one exported
    type, a component on the same bus twice, or snapshots that are neither
    integer- nor datetime-typed.

    Parameters
    ----------
    n
        The network to present.

    Examples
    --------
    >>> n = pypsa.Network()  # doctest: +SKIP
    >>> record = n.to_datarecord()  # doctest: +SKIP

    """

    def __init__(self, n: Network) -> None:
        """Present `n` as a `Record`, validating eagerly."""
        self.n = n
        self._components = _exported_components(n)
        _check_collisions(self._components)
        _check_same_bus_twice(self._components)
        _check_snapshots(n)

    # -- Record protocol ----------------------------------------------------

    @cached_property
    def schema(self) -> Schema:
        """The canonical schema, with the network's own attributes as `meta`."""
        n = self.n
        schema = build_schema(
            multiperiod=n.has_periods, timestep_dtype=_timestep_dtype_name(n)
        )
        schema.meta["pypsa"] = {
            "attributes": {k: _scalar(getattr(n, k)) for k in NETWORK_ATTRS},
            "crs": n.crs.to_wkt() if n.crs is not None else None,
            "meta": dict(n.meta),
        }
        return schema

    @cached_property
    def dims(self) -> LazyFrames:
        """Axis frames, keyed by dim: `timestep`, `period`, `scenario`, `entity`."""
        n = self.n
        axes: dict[str, Any] = {}
        timestep = n.snapshot_weightings.reset_index()
        if "snapshot" in timestep.columns:
            timestep = timestep.rename(columns={"snapshot": TIMESTEP})
        axes[TIMESTEP] = timestep

        if n.has_periods:
            period = n.investment_period_weightings.reset_index().rename(
                columns={"period": PERIOD, **PERIOD_WEIGHTINGS}
            )
        else:
            period = pd.DataFrame(columns=[PERIOD, *PERIOD_WEIGHTINGS.values()])
        axes[PERIOD] = period

        if n.has_scenarios:
            scenario = n.scenario_weightings.reset_index().rename(
                columns={"scenario": SCENARIO, **SCENARIO_WEIGHTINGS}
            )
        else:
            scenario = pd.DataFrame(columns=[SCENARIO, *SCENARIO_WEIGHTINGS.values()])
        axes[SCENARIO] = scenario

        keys = tuple(d for d in (TIMESTEP, PERIOD, SCENARIO) if not axes[d].empty)
        return LazyFrames((*keys, _ENTITY), lambda key: self._dim_frame(key, axes))

    def _dim_frame(self, key: str, axes: dict[str, pd.DataFrame]) -> Any:
        if key == _ENTITY:
            return nw.from_native(self._entity_axis_frame()).lazy()
        return nw.from_native(axes[key]).lazy()

    def _entity_axis_frame(self) -> pd.DataFrame:
        """`(entity, entity_type, deleted)` across every exported type."""
        frames = []
        for c in self._components:
            names = _names(c)
            frames.append(
                pd.DataFrame(
                    {
                        _ENTITY: names.astype(str),
                        ENTITY_TYPE: c.name,
                        "deleted": False,
                    }
                )
            )
        if not frames:
            return pd.DataFrame(columns=[_ENTITY, ENTITY_TYPE, "deleted"])
        return pd.concat(frames, ignore_index=True)

    @cached_property
    def entity_types(self) -> LazyFrames:
        """Wide member frames, keyed by component type."""
        names = tuple(c.name for c in self._components)
        by_name = {c.name: c for c in self._components}
        return LazyFrames(
            names,
            lambda ctype: nw.from_native(self._member_frame(by_name[ctype])).lazy(),
        )

    def _member_frame(self, c: Components) -> pd.DataFrame:
        """One type's non-port, non-output, non-varying static columns, defaults dropped."""
        defaults = c.defaults
        ports = port_columns(c)
        columns = []
        for col in c.static.columns:
            if col in ports or col in ("g_pu", "b_pu"):
                continue
            if col not in defaults.index:
                columns.append(col)
                continue
            if str(defaults.at[col, "status"]).startswith("Output"):
                continue
            if defaults.at[col, "varying"] or c.has_piecewise(col):
                # Addressed beyond `entity` too, so it lives in its long
                # file only (`inputs/<attr>.parquet`), not here as well.
                continue
            columns.append(col)
        columns = [x for x in columns if x not in _scenario_varying(c, columns)]

        static = c.static[columns]
        if isinstance(static.index, pd.MultiIndex):
            static = static.droplevel(SCENARIO)
            static = static[~static.index.duplicated()]
        frame = static.reset_index().rename(columns={"name": _ENTITY})

        for col in frame.columns:
            if col == _ENTITY or col not in defaults.index:
                continue
            default = _default(defaults.at[col, "default"])
            if default is None:
                keep = frame[col].notna()
            else:
                keep = frame[col] != default
            if (~keep).any():
                frame[col] = frame[col].astype(object)
                frame.loc[~keep, col] = np.nan

        if c.name == "Shape" and "geometry" in frame.columns:
            # Plain `pd.DataFrame` before the WKT swap: `frame` is still a
            # `GeoDataFrame` here, and assigning text into its geometry column
            # warns that the column no longer holds geometries.
            wkt = frame["geometry"].to_wkt()
            frame = pd.DataFrame(frame)
            frame["geometry"] = wkt
        if c.name == "SubNetwork" and "obj" in frame.columns:
            frame = frame.drop(columns=["obj"])
        return frame

    @cached_property
    def groups(self) -> LazyFrames:
        """The `connection` group's rows, one frame across every type."""
        return LazyFrames(
            (_CONNECTION,), lambda _: nw.from_native(self._connection_frame()).lazy()
        )

    def _connection_frame(self) -> pd.DataFrame:
        frames = []
        for c in self._components:
            if not c.ports:
                continue
            static = c.static
            for port in c.ports:
                col = f"bus{port}"
                if col not in static.columns:
                    continue
                buses = static[col]
                if isinstance(buses.index, pd.MultiIndex):
                    buses = buses.droplevel(SCENARIO)
                    buses = buses[~buses.index.duplicated()]
                attached = buses[buses.astype(str) != ""]
                if attached.empty:
                    continue
                rows = (
                    attached.rename(_BUS)
                    .reset_index()
                    .rename(columns={"name": _ENTITY})
                )
                rows[_ENTITY] = rows[_ENTITY].astype(str)
                frames.append(rows[[_ENTITY, _BUS]])
        if not frames:
            return pd.DataFrame(columns=[_ENTITY, _BUS])
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    @cached_property
    def attributes(self) -> LazyFrames:
        """Long input frames, keyed by record-wide attribute name."""
        names: dict[str, list[Components]] = {}
        for c in self._components:
            for attr in self._input_attrs(c):
                names.setdefault(attr, []).append(c)
            if c.ports:
                names.setdefault(PORT, []).append(c)
        return LazyFrames(
            tuple(names),
            lambda attr: nw.from_native(self._long_frame(attr, names[attr])).lazy(),
        )

    @cached_property
    def outputs(self) -> LazyFrames:
        """Long result frames, keyed by record-wide attribute name."""
        names: dict[str, list[Components]] = {}
        for c in self._components:
            for attr in self._output_attrs(c):
                names.setdefault(attr, []).append(c)
        return LazyFrames(
            tuple(names),
            lambda attr: nw.from_native(self._long_frame(attr, names[attr])).lazy(),
        )

    def flags(self, ctype: str) -> dict[str, Flags]:
        """Which axes each attribute of `ctype` actually varies over."""
        by_name = {c.name: c for c in self._components}
        c = by_name.get(ctype)
        if c is None:
            return {}
        multiperiod = isinstance(self.n.snapshots, pd.MultiIndex)
        result: dict[str, Flags] = {}
        for attr in (*self._input_attrs(c), *self._output_attrs(c)):
            varies: set[str] = set()
            if attr in c.dynamic and not c.dynamic[attr].empty:
                varies.add(TIMESTEP)
                if multiperiod:
                    varies.add(PERIOD)
            broadcast = {TIMESTEP} if attr in c.static.columns else set()
            breakpoints = attr in c.piecewise and not c.piecewise[attr].empty
            result[attr] = Flags(
                varies=frozenset(varies),
                broadcast=frozenset(broadcast),
                breakpoints=breakpoints,
            )
        return result

    # -- key sets -------------------------------------------------------

    def _input_attrs(self, c: Components) -> list[str]:
        """Record-wide input attribute names this type carries, port stems collapsed."""
        defaults = c.defaults
        ports = port_columns(c)
        diverging = _scenario_varying(c, list(c.static.columns))
        seen: list[str] = []
        for attr in defaults.index:
            if attr == "name":
                continue
            if str(defaults.at[attr, "status"]).startswith("Output"):
                continue
            stem, port = ports.get(attr, (attr, None))
            name = record_name(c.name, stem) if port is None else stem
            if (
                not defaults.at[attr, "varying"]
                and attr not in diverging
                and not c.has_piecewise(attr)
            ):
                continue
            if name not in seen:
                seen.append(name)
        return seen

    def _output_attrs(self, c: Components) -> list[str]:
        defaults = c.defaults
        ports = port_columns(c)
        names: list[str] = []
        for attr in defaults.index:
            if not str(defaults.at[attr, "status"]).startswith("Output"):
                continue
            stem, port = ports.get(attr, (attr, None))
            name = record_name(c.name, stem) if port is None else stem
            if name not in names:
                names.append(name)
        return names

    # -- long frames ------------------------------------------------------

    def _long_frame(self, attribute: str, components: list[Components]) -> pd.DataFrame:
        columns = list(self.schema.long_columns_for(attribute))
        frames = []
        for c in components:
            if attribute == PORT:
                rows = self._port_rows(c)
            else:
                ports = port_columns(c)
                per_port = [
                    col for col, (stem, port) in ports.items() if stem == attribute
                ]
                if per_port:
                    rows = self._per_port_rows(c, attribute, per_port)
                else:
                    rows = self._entity_rows(c, attribute)
            if rows is not None and not rows.empty:
                frames.append(rows)
        if not frames:
            return pd.DataFrame(columns=columns)
        long = pd.concat(frames, ignore_index=True)
        for col in columns:
            if col not in long.columns:
                long[col] = None
        if PERIOD in long.columns:
            # Scalar rows lack `period`; the concat upcasts the column to
            # float64 unless it is cast back to a nullable integer here.
            long[PERIOD] = long[PERIOD].astype("Int64")
        return long[columns]

    def _port_rows(self, c: Components) -> pd.DataFrame:
        """`(entity, bus, attribute="port", breakpoint=None, value=<port label>)`."""
        frames = []
        static = c.static
        for port in c.ports:
            col = f"bus{port}"
            if col not in static.columns:
                continue
            buses = static[col]
            if isinstance(buses.index, pd.MultiIndex):
                buses = buses.droplevel(SCENARIO)
                buses = buses[~buses.index.duplicated()]
            attached = buses[buses.astype(str) != ""]
            if attached.empty:
                continue
            rows = attached.rename(_BUS).reset_index().rename(columns={"name": _ENTITY})
            rows[_ENTITY] = rows[_ENTITY].astype(str)
            rows["value"] = port
            frames.append(rows[[_ENTITY, _BUS, "value"]])
        if not frames:
            return pd.DataFrame(columns=[_ENTITY, _BUS, "value"])
        rows = pd.concat(frames, ignore_index=True)
        rows["attribute"] = PORT
        rows["breakpoint"] = None
        return rows

    def _entity_rows(self, c: Components, attribute: str) -> pd.DataFrame | None:
        """Component-addressed long rows for one attribute, series and scalar."""
        attr = self._source_attr(c, attribute)
        if attr is None:
            return None
        long = self._stack_column(c, attr)
        if long is None:
            return None
        long["attribute"] = attribute
        return long

    def _per_port_rows(
        self, c: Components, attribute: str, columns: list[str]
    ) -> pd.DataFrame:
        """Connection-addressed long rows for a per-port attribute."""
        ports = port_columns(c)
        static = c.static
        frames = []
        for col in columns:
            _stem, port = ports[col]
            bus_col = f"bus{port}"
            if bus_col not in static.columns:
                continue
            rows = self._stack_column(c, col)
            if rows is None or rows.empty:
                continue
            buses = static[bus_col]
            if isinstance(buses.index, pd.MultiIndex):
                buses = buses.droplevel(SCENARIO)
                buses = buses[~buses.index.duplicated()]
            buses.index = buses.index.astype(str)
            rows[_BUS] = rows[_ENTITY].map(buses)
            rows = rows[rows[_BUS].astype(str) != ""]
            rows["attribute"] = attribute
            frames.append(rows)
        if not frames:
            return pd.DataFrame(
                columns=[_ENTITY, _BUS, "attribute", "breakpoint", "value"]
            )
        return pd.concat(frames, ignore_index=True)

    def _stack_column(self, c: Components, attr: str) -> pd.DataFrame | None:
        """`(entity, breakpoint, value)` rows for one raw PyPSA column: series, static and piecewise data, stacked."""
        defaults = c.defaults
        frames = []
        series = c.dynamic.get(attr)
        if series is not None and not series.empty:
            frames.append(self._stack_series(c, series))
        if attr in c.static.columns:
            if series is not None and isinstance(series.columns, pd.MultiIndex):
                exclude = series.columns.get_level_values("name")
            elif series is not None:
                exclude = series.columns
            else:
                exclude = pd.Index([])
            static = c.static[attr]
            if isinstance(static.index, pd.MultiIndex):
                scalars = static.reset_index().rename(
                    columns={"name": _ENTITY, "scenario": SCENARIO, attr: "value"}
                )
            else:
                scalars = static.reset_index().rename(
                    columns={"name": _ENTITY, attr: "value"}
                )
            if len(exclude):
                scalars = scalars[~scalars[_ENTITY].isin(exclude)]
            if attr in defaults.index:
                default = _default(defaults.at[attr, "default"])
                scalars = _drop_default(scalars, default)
            frames.append(scalars)
        piecewise = self._piecewise_long(c, attr)
        if piecewise is not None and not piecewise.empty:
            frames.append(piecewise)
        if not frames:
            return None
        long = pd.concat(frames, ignore_index=True)
        long[_ENTITY] = long[_ENTITY].astype(str)
        if "breakpoint" not in long.columns:
            long["breakpoint"] = None
        return long

    def _piecewise_long(self, c: Components, attr: str) -> pd.DataFrame | None:
        """`(entity, breakpoint, value)` rows from one piecewise curve, NaN padding dropped."""
        pw = c.piecewise.get(attr)
        if pw is None or pw.empty:
            return None
        x_attr = c._piecewise_schema(attr).x
        x = pw.xs(x_attr, level="attribute", axis=1).rename_axis(index="_row")
        y = pw.xs(attr, level="attribute", axis=1).rename_axis(index="_row")
        x_long = x.stack(future_stack=True).rename("breakpoint")
        y_long = y.stack(future_stack=True).rename("value")
        long = pd.concat([x_long, y_long], axis=1).reset_index().drop(columns="_row")
        long = long.rename(columns={"name": _ENTITY}).dropna(
            subset=["breakpoint", "value"]
        )
        long[_ENTITY] = long[_ENTITY].astype(str)
        return long

    def _source_attr(self, c: Components, record_attr: str) -> str | None:
        """Which of `c`'s own columns writes `record_attr` (undo `record_name`)."""
        defaults = c.defaults
        ports = port_columns(c)
        for attr in defaults.index:
            stem, port = ports.get(attr, (attr, None))
            if port is not None:
                continue
            if record_name(c.name, stem) == record_attr:
                return attr
        return None

    def _stack_series(self, c: Components, wide: pd.DataFrame) -> pd.DataFrame:
        """Melt a `snapshots x components` frame into long rows."""
        stochastic = isinstance(wide.columns, pd.MultiIndex)
        wide = wide.rename_axis(columns=[SCENARIO, _ENTITY] if stochastic else _ENTITY)
        stacked = wide.stack(level=list(range(wide.columns.nlevels)), future_stack=True)
        long = stacked.rename("value").reset_index()
        if isinstance(c.snapshots, pd.MultiIndex):
            long = long.rename(columns={"timestep": TIMESTEP, "period": PERIOD})
        else:
            long = long.rename(columns={"snapshot": TIMESTEP})
        return long
