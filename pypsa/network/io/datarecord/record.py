# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""`NetworkRecord`: a `Network` presented lazily as a datarecord `Record`.

The export side of the datarecord format: `NetworkRecord` implements the
`datarecord.Record` protocol over a live `Network`, undoing PyPSA's
static/dynamic split into the record's wide member frames and long rows.
Validated eagerly on construction for shapes it cannot represent (name
collisions, a component on the same bus twice); unsupported snapshot dtypes
surface lazily, the first time `schema` is accessed.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np
import pandas as pd
from datarecord.record import Flags, LazyFrames

from pypsa.network.io.datarecord.schema import (
    _BUS,
    _CONNECTION,
    _CUSTOM_ATTRS_META_KEY,
    _DIM_ATTRS,
    _ENTITY,
    _EXCLUDED_TYPES,
    _TOPOLOGY_OUTPUTS,
    DIM_TYPES,
    ENTITY_TYPE,
    PERIOD,
    PERIOD_WEIGHTINGS,
    PORT,
    SCENARIO,
    SCENARIO_WEIGHTINGS,
    SHAPE,
    TIMESTEP,
    build_schema,
    custom_dim_attr_name,
    declare_custom,
    port_columns,
    pypsa_name,
    record_name,
)

if TYPE_CHECKING:
    from datarecord.schema import Schema

    from pypsa import Network
    from pypsa.components.components import Components

# The network-attribute allow-list the datarecord path round-trips, anything
# else PyPSA carries stays off the record (no `dir(n)` scan). `_objective` and
# `_objective_constant` are kept as user-visible `optimize()` results, like
# every other export format, defaulting to None so an unsolved network
# round-trips them as absent.
NETWORK_ATTRS = (
    "name",
    "pypsa_version",
    "_multi_invest",
    "_objective",
    "_objective_constant",
)


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


def _attached_buses(c: Components, port: str) -> pd.Series:
    """One port's bus attachments, indexed by entity name.

    The scenario level is dropped and scenario-broadcast duplicates removed
    (a connection's bus never varies by scenario), and unattached (empty
    string) entries are filtered out. Empty if the port has no `bus<port>`
    column at all.
    """
    col = f"bus{port}"
    static = c.static
    if col not in static.columns:
        return pd.Series(dtype=object)
    buses = static[col]
    if isinstance(buses.index, pd.MultiIndex):
        buses = buses.droplevel(SCENARIO)
        buses = buses[~buses.index.duplicated()]
    return buses[buses.astype(str) != ""]


def _check_same_bus_twice(components: list[Components]) -> None:
    """Raise if a component attaches to the same bus on two of its ports."""
    offenders: list[str] = []
    for c in components:
        if len(c.ports) < 2:
            continue
        buses = {
            port: s for port in c.ports if not (s := _attached_buses(c, port)).empty
        }
        if len(buses) < 2:
            continue
        frame = pd.DataFrame(buses)
        ports = list(frame.columns)
        for i, left in enumerate(ports):
            for right in ports[i + 1 :]:
                dup = frame[left].notna() & (frame[left] == frame[right])
                offenders += [
                    f"{c.name} {name} -> {bus}"
                    for name, bus in frame.loc[dup, left].items()
                ]
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


def _drop_default_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], defaults: pd.DataFrame
) -> None:
    """Replace each column's default-valued cells with NaN, in place.

    Shared by `_member_frame` and `_carrier_shape_dim_frame`: both write a
    wide frame where a cell equal to the registry default is left unwritten.
    Skips a column absent from `frame` or `defaults`.
    """
    for col in columns:
        if col not in frame.columns or col not in defaults.index:
            continue
        default = _default(defaults.at[col, "default"])
        keep = frame[col].notna() if default is None else frame[col] != default
        if (~keep).any():
            frame[col] = frame[col].astype(object)
            frame.loc[~keep, col] = np.nan


def _scenario_varying(c: Components, columns: list[str]) -> set[str]:
    """Which static columns hold more than one value across scenarios."""
    index = c.static.index
    if not isinstance(index, pd.MultiIndex) or SCENARIO not in (index.names or []):
        return set()
    by_entity = c.static.groupby(level="name")
    return {x for x in columns if (by_entity[x].nunique(dropna=False) > 1).any()}


def _custom_dtype(values: pd.Series | np.ndarray) -> nw.dtypes.DType:
    """Map a pandas column's dtype to the narwhals dtype a custom attribute takes.

    Float becomes Float64, int Int64, bool Boolean, anything else String.
    """
    if pd.api.types.is_bool_dtype(values):
        return nw.Boolean()
    if pd.api.types.is_float_dtype(values):
        return nw.Float64()
    if pd.api.types.is_integer_dtype(values):
        return nw.Int64()
    return nw.String()


def _custom_static_attrs(c: Components) -> dict[str, nw.dtypes.DType]:
    """Collect static columns `c` carries that the registry does not declare."""
    defaults = c.defaults
    return {
        col: _custom_dtype(c.static[col])
        for col in c.static.columns
        if col not in defaults.index
    }


def _custom_series_attrs(c: Components) -> dict[str, nw.dtypes.DType]:
    """Time-varying keys `c.dynamic` carries that the registry does not declare."""
    defaults = c.defaults
    return {
        attr: _custom_dtype(df.to_numpy())
        for attr, df in c.dynamic.items()
        if attr not in defaults.index and not df.empty
    }


def _cast_custom_string_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Cast each listed column to text where its values aren't numeric or boolean, in place.

    `astype(str)` turns NaN into the literal "nan", so null cells are masked
    back to stay null.
    """
    for col in columns:
        if col not in frame.columns:
            continue
        if isinstance(_custom_dtype(frame[col]), nw.String):
            frame[col] = frame[col].where(frame[col].isna(), frame[col].astype(str))


class NetworkRecord:
    """A `Network` presented as a datarecord `Record` (export only).

    Validates eagerly on construction and raises `DatarecordExportError` for a
    shape the record cannot hold: names claimed by more than one exported
    type, or a component on the same bus twice. Snapshots that are neither
    integer- nor datetime-typed raise the same error, but only once `schema`
    is accessed.

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
        self._dim_components = {
            dim: n.components[ctype]
            for dim, ctype in DIM_TYPES.items()
            if not n.components[ctype].static.empty
        }
        _check_collisions(self._components)
        _check_same_bus_twice(self._components)

    # -- Record protocol ----------------------------------------------------

    @cached_property
    def schema(self) -> Schema:
        """The canonical schema, with custom attributes applied.

        The network's own attributes are carried as `meta`.
        """
        n = self.n
        schema = build_schema(
            multiperiod=n.has_periods,
            timestep_dtype=_timestep_dtype_name(n),
            stochastic=n.has_scenarios,
        )
        self._declare_custom_attrs(schema)
        schema.meta["pypsa"] = {
            "attributes": {k: _scalar(getattr(n, k)) for k in NETWORK_ATTRS},
            "crs": n.crs.to_wkt() if n.crs is not None else None,
            "meta": dict(n.meta),
        }
        return schema

    def _declare_custom_attrs(self, schema: Schema) -> None:
        """Grant every custom static or time-varying attribute to its type(s).

        Carrier and Shape's custom static columns are declared too, over
        their dim rather than granted to a type (`declare_custom`).
        """
        multiperiod = self.n.has_periods
        for c in (*self._components, *self._dim_components.values()):
            for attr, dtype in _custom_static_attrs(c).items():
                declare_custom(
                    schema, c.name, attr, dtype, varying=False, multiperiod=multiperiod
                )
            for attr, dtype in _custom_series_attrs(c).items():
                declare_custom(
                    schema, c.name, attr, dtype, varying=True, multiperiod=multiperiod
                )
        schema.meta.pop(_CUSTOM_ATTRS_META_KEY, None)

    @cached_property
    def dims(self) -> LazyFrames:
        """Axis frames, keyed by dim: `timestep`, `period`, `scenario`, `entity`, `carrier`, `shape`."""
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
        keys = (*keys, _ENTITY, *self._dim_components)
        return LazyFrames(keys, lambda key: self._dim_frame(key, axes))

    def _dim_frame(self, key: str, axes: dict[str, pd.DataFrame]) -> Any:
        if key == _ENTITY:
            return nw.from_native(self._entity_axis_frame()).lazy()
        if key in self._dim_components:
            return nw.from_native(self._carrier_shape_dim_frame(key)).lazy()
        return nw.from_native(axes[key]).lazy()

    def _carrier_shape_dim_frame(self, dim: str) -> pd.DataFrame:
        """One row per carrier/shape name, non-stochastic attribute columns included.

        A stochastic network carries only the name column here: its
        attributes vary by scenario and are written as long rows instead
        (`_dim_attr_long_frame`). Default values are dropped as for a
        component's static columns, and columns are named by each
        attribute's record-wide name (`record_name`), not its PyPSA one.
        """
        c = self._dim_components[dim]
        defaults = c.defaults
        static = c.static
        if isinstance(static.index, pd.MultiIndex):
            static = static.droplevel(SCENARIO)
            static = static[~static.index.duplicated()]
        frame = static.reset_index().rename(columns={"name": dim})
        if self.n.has_scenarios:
            return frame[[dim]]

        _drop_default_columns(frame, _DIM_ATTRS[dim], defaults)

        if dim == SHAPE and "geometry" in frame.columns:
            # Cast to plain DataFrame before the WKT swap. Assigning text into
            # a GeoDataFrame's geometry column warns that it no longer holds
            # geometries.
            wkt = frame["geometry"].to_wkt()
            frame = pd.DataFrame(frame)
            frame["geometry"] = wkt
        custom = list(_custom_static_attrs(c))
        _cast_custom_string_columns(frame, custom)
        frame = frame.rename(
            columns={
                **{col: record_name(c.name, col) for col in _DIM_ATTRS[dim]},
                **{col: custom_dim_attr_name(dim, col) for col in custom},
            }
        )
        return frame[
            [
                dim,
                *(record_name(c.name, col) for col in _DIM_ATTRS[dim]),
                *(custom_dim_attr_name(dim, col) for col in custom),
            ]
        ]

    def _dim_attr_long_frame(self, dim: str, attr: str) -> pd.DataFrame:
        """`(scenario, dim, attribute, breakpoint, value)` rows for one stochastic carrier/shape attribute.

        `attr` is the record-wide attribute name; default values are dropped,
        as for a component's long input rows.
        """
        c = self._dim_components[dim]
        defaults = c.defaults
        pypsa_attr = pypsa_name(c.name, attr)
        static = c.static[pypsa_attr]
        if dim == SHAPE and pypsa_attr == "geometry":
            static = static.to_wkt()
        rows = static.rename("value").reset_index().rename(columns={"name": dim})
        default = _default(defaults.at[pypsa_attr, "default"])
        rows = _drop_default(rows, default)
        rows["attribute"] = attr
        rows["breakpoint"] = None
        return rows[[SCENARIO, dim, "attribute", "breakpoint", "value"]]

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
        custom_series = _custom_series_attrs(c)
        custom: list[str] = []
        columns = []
        for col in c.static.columns:
            if col in ports or col in ("g_pu", "b_pu"):
                continue
            if col not in defaults.index:
                if col in custom_series:
                    # Also time-varying: lives in its long file only, like a
                    # registry attribute whose `varying` flag routes it there.
                    continue
                columns.append(col)
                custom.append(col)
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
        _drop_default_columns(frame, tuple(frame.columns), defaults)
        _cast_custom_string_columns(frame, custom)
        return frame

    @cached_property
    def groups(self) -> LazyFrames:
        """The `connection` group's rows, one frame across every type."""
        return LazyFrames(
            (_CONNECTION,), lambda _: nw.from_native(self._connection_frame()).lazy()
        )

    def _connection_frame(self) -> pd.DataFrame:
        frames = [
            self._port_rows(c)[[_ENTITY, _BUS]] for c in self._components if c.ports
        ]
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
        dim_attrs = self._stochastic_dim_attrs()
        keys = (*names, *dim_attrs)

        def _build(attr: str) -> Any:
            if attr in dim_attrs:
                dim = dim_attrs[attr]
                return nw.from_native(self._dim_attr_long_frame(dim, attr)).lazy()
            return nw.from_native(self._long_frame(attr, names[attr])).lazy()

        return LazyFrames(keys, _build)

    def _stochastic_dim_attrs(self) -> dict[str, str]:
        """Carrier/shape attribute's record-wide name -> its dim, for a stochastic network only.

        Empty otherwise: a non-stochastic network's carrier/shape attributes
        are columns of their axis file, not long input rows.
        """
        if not self.n.has_scenarios:
            return {}
        return {
            record_name(self._dim_components[dim].name, attr): dim
            for dim in self._dim_components
            for attr in _DIM_ATTRS[dim]
        }

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
        ports = port_columns(c)
        result: dict[str, Flags] = {}
        for attr in (*self._input_attrs(c), *self._output_attrs(c)):
            per_port = [col for col, (stem, _port) in ports.items() if stem == attr]
            source = per_port or [
                x for x in [self._source_attr(c, attr)] if x is not None
            ]
            varies: set[str] = set()
            broadcast: set[str] = set()
            breakpoints = False
            for col in source:
                if col in c.dynamic and not c.dynamic[col].empty:
                    varies.add(TIMESTEP)
                    if multiperiod:
                        varies.add(PERIOD)
                if col in c.static.columns:
                    broadcast.add(TIMESTEP)
                if col in c.piecewise and not c.piecewise[col].empty:
                    breakpoints = True
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
        for attr in _custom_series_attrs(c):
            if attr not in seen:
                seen.append(attr)
        return seen

    def _output_attrs(self, c: Components) -> list[str]:
        defaults = c.defaults
        ports = port_columns(c)
        names: list[str] = []
        for attr in defaults.index:
            if not str(defaults.at[attr, "status"]).startswith("Output"):
                continue
            if (c.name, attr) in _TOPOLOGY_OUTPUTS:
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
            # Scalar rows lack `period`, and the concat upcasts the column to
            # float64 unless it is cast back to a nullable integer here.
            long[PERIOD] = long[PERIOD].astype("Int64")
        return long[columns]

    def _port_rows(self, c: Components) -> pd.DataFrame:
        """`(entity, bus, attribute="port", breakpoint=None, value=<port label>)`."""
        frames = []
        for port in c.ports:
            attached = _attached_buses(c, port)
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
        frames = []
        for col in columns:
            _stem, port = ports[col]
            buses = _attached_buses(c, port)
            if buses.empty:
                continue
            rows = self._stack_column(c, col)
            if rows is None or rows.empty:
                continue
            buses = buses.copy()
            buses.index = buses.index.astype(str)
            rows[_BUS] = rows[_ENTITY].map(buses)
            rows = rows[rows[_BUS].notna()]
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
        """Which of `c`'s own columns writes `record_attr` (undo `record_name`).

        A custom attribute carries no override, so its record-wide name is
        the PyPSA column itself.
        """
        defaults = c.defaults
        ports = port_columns(c)
        for attr in defaults.index:
            stem, port = ports.get(attr, (attr, None))
            if port is not None:
                continue
            if record_name(c.name, stem) == record_attr:
                return attr
        if record_attr not in defaults.index and record_attr in c.dynamic:
            return record_attr
        return None

    def _stack_series(self, c: Components, wide: pd.DataFrame) -> pd.DataFrame:
        """Melt a `snapshots x components` frame into long rows."""
        stochastic = isinstance(wide.columns, pd.MultiIndex)
        wide = wide.rename_axis(columns=[SCENARIO, _ENTITY] if stochastic else _ENTITY)
        stacked = wide.stack(level=list(range(wide.columns.nlevels)), future_stack=True)
        long = stacked.rename("value").reset_index()
        if not isinstance(c.snapshots, pd.MultiIndex):
            long = long.rename(columns={"snapshot": TIMESTEP})
        return long
