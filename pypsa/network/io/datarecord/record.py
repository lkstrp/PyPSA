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
    _ENTITY,
    _EXCLUDED_TYPES,
    _TOPOLOGY_OUTPUTS,
    CARRIER,
    CARRIER_ATTR,
    ENTITY_TYPE,
    PERIOD,
    PERIOD_WEIGHTINGS,
    PORT,
    SCENARIO,
    SCENARIO_WEIGHTINGS,
    SHAPE,
    SHAPE_GEOMETRY,
    SHAPE_NAME,
    SHAPE_TYPE,
    TIMESTEP,
    DatarecordExportError,
    build_schema,
    declare_custom,
    port_columns,
    record_name,
)
from pypsa.network.names import _name_level, format_clashes

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


def _exported_components(n: Network) -> list[Components]:
    """Component types with data to write: non-empty, standard types excluded.

    `_EXCLUDED_TYPES` (templates, derived non-schema types, and Shape, which
    is the `shape` group) are never exported as entity types: the schema
    grants no entity-type label for them.
    """
    return [
        c
        for c in n.components
        if not c.static.empty
        and c.name not in _EXCLUDED_TYPES
        and c.name not in n.standard_type_components
    ]


def _check_collisions(components: list[Components]) -> None:
    """Raise if a name is claimed by more than one exported type."""
    owners: dict[str, set[str]] = {}
    for c in components:
        for name in _name_level(c.static.index):
            owners.setdefault(str(name), set()).add(c.name)
    clashing = {name: sorted(types) for name, types in owners.items() if len(types) > 1}
    if clashing:
        raise DatarecordExportError(format_clashes(clashing))


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


def _static_once(c: Components) -> pd.DataFrame:
    """`c.static` with the scenario level dropped and one row per name."""
    static = c.static
    if isinstance(static.index, pd.MultiIndex):
        static = static.droplevel(SCENARIO)
        static = static[~static.index.duplicated()]
    return static


def _check_carrier_references(n: Network, components: list[Components]) -> None:
    """Raise if a component names a carrier the network does not declare, or one that varies by scenario.

    An empty carrier, or one equal to the type's registry default (Bus's
    `"AC"`), is not a reference when undeclared: no `carrier` row is written
    and import restores the default. Any other undeclared name is an error,
    since the `carrier` group cannot point at an entity the record lacks.
    """
    declared = set(_name_level(n.components["Carrier"].static.index).astype(str))
    dangling: dict[str, set[str]] = {}
    for c in components:
        if CARRIER_ATTR not in c.static.columns:
            continue
        if _scenario_varying(c, [CARRIER_ATTR]):
            msg = (
                f"{c.name} carrier varies by scenario, which the datarecord format "
                f"cannot represent: a component has one carrier"
            )
            raise DatarecordExportError(msg)
        default = c.defaults.at[CARRIER_ATTR, "default"]
        carriers = c.static[CARRIER_ATTR].astype(str)
        missing = carriers[~carriers.isin(declared) & (carriers != "")]
        missing = missing[missing != default] if isinstance(default, str) else missing
        for name in missing.unique():
            dangling.setdefault(str(name), set()).add(c.name)
    if dangling:
        names = ", ".join(
            f"{k} ({', '.join(sorted(v))})" for k, v in sorted(dangling.items())
        )
        add = ", ".join(repr(k) for k in sorted(dangling))
        msg = (
            f"carriers referenced but not defined in n.carriers: {names}; "
            f'declare them first with n.add("Carrier", [{add}])'
        )
        raise DatarecordExportError(msg)


def _check_shapes(n: Network, components: list[Components]) -> None:
    """Raise for a shape the `shape` group cannot key.

    A shape is `(component, type) -> geometry`, so it must name an exported
    component, no two shapes may describe one component with the same type,
    and no shape column may differ across scenarios.
    """
    c = n.components["Shape"]
    if c.static.empty:
        return
    varying = _scenario_varying(c, list(c.static.columns))
    if varying:
        msg = (
            f"Shape columns {sorted(varying)} vary by scenario, which the "
            f"datarecord format cannot represent: geography has one value"
        )
        raise DatarecordExportError(msg)
    static = _static_once(c)
    names_by_type = {
        x.name: set(_name_level(x.static.index).astype(str)) for x in components
    }
    unattached = [
        str(name)
        for name, row in static.iterrows()
        if str(row["idx"]) not in names_by_type.get(str(row["component"]), set())
    ]
    if unattached:
        msg = (
            f"shapes must describe an exported component through `component` and "
            f"`idx`, but these do not: {', '.join(unattached)}"
        )
        raise DatarecordExportError(msg)
    key = static[["component", "idx", "type"]].astype(str)
    dup = key.duplicated(keep=False)
    if dup.any():
        offenders = ", ".join(
            f"{name} ({row['component']} {row['idx']}, type {row['type']!r})"
            for name, row in key[dup].iterrows()
        )
        msg = f"two shapes describe the same component with the same type: {offenders}"
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

    A cell equal to the registry default is left unwritten in a wide frame.
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
    """Time-varying keys `c.dynamic` carries that the registry does not declare.

    Reads the dtype straight off `df.dtypes` where every column shares one,
    the common case, rather than materialising the frame with `to_numpy()`
    just to inspect it. Falls back to `to_numpy()` for a frame whose columns
    genuinely differ, to keep numpy's own dtype promotion.
    """
    defaults = c.defaults
    result: dict[str, nw.dtypes.DType] = {}
    for attr, df in c.dynamic.items():
        if attr in defaults.index or df.empty:
            continue
        dtypes = df.dtypes
        values = dtypes.iloc[0] if (dtypes == dtypes.iloc[0]).all() else df.to_numpy()
        result[attr] = _custom_dtype(values)
    return result


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
    type, a component on the same bus twice, a carrier referenced but not
    declared, or a shape that describes no exported component. Snapshots that
    are neither integer- nor datetime-typed raise the same error, but only
    once `schema` is accessed.

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
        self._shapes = n.components["Shape"]
        # (type, PyPSA column) <-> record-wide name for custom attributes,
        # filled when `schema` declares them.
        self._custom_names: dict[tuple[str, str], str] = {}
        self._custom_columns: dict[tuple[str, str], str] = {}
        _check_collisions(self._components)
        _check_same_bus_twice(self._components)
        _check_carrier_references(n, self._components)
        _check_shapes(n, self._components)

    # -- Record protocol ----------------------------------------------------

    @cached_property
    def schema(self) -> Schema:
        """The canonical schema, with custom attributes applied.

        The network's own attributes are carried as `meta`.
        """
        n = self.n
        schema = build_schema(
            multiperiod=n.has_periods, timestep_dtype=_timestep_dtype_name(n)
        )
        custom = self._declare_custom_attrs(schema)
        schema.meta["pypsa"] = {
            "attributes": {k: _scalar(getattr(n, k)) for k in NETWORK_ATTRS},
            "crs": n.crs.to_wkt() if n.crs is not None else None,
            "meta": dict(n.meta),
            "custom_attributes": custom,
        }
        return schema

    def _declare_custom_attrs(self, schema: Schema) -> dict[str, dict[str, str]]:
        """Grant every custom static or time-varying attribute to its type(s).

        Shape's custom static columns are declared too, as payload of the
        `shape` group (`declare_custom`). Fills `_custom_names` and returns
        `{type: {record name: PyPSA column}}` for the manifest's meta, static
        columns first in the order the network holds them, so import can
        both undo a `custom_fallback_name` and restore the column order.
        """
        multiperiod = self.n.has_periods
        shapes = [self._shapes] if not self._shapes.static.empty else []
        custom: dict[str, dict[str, str]] = {}
        for c in (*self._components, *shapes):
            declared = [
                (attr, dtype, False) for attr, dtype in _custom_static_attrs(c).items()
            ] + [(attr, dtype, True) for attr, dtype in _custom_series_attrs(c).items()]
            for attr, dtype, varying in declared:
                name = declare_custom(
                    schema,
                    c.name,
                    attr,
                    dtype,
                    varying=varying,
                    multiperiod=multiperiod,
                )
                self._custom_names[(c.name, attr)] = name
                self._custom_columns[(c.name, name)] = attr
                custom.setdefault(c.name, {})[name] = attr
        return custom

    def _custom_record_name(self, c: Components, attr: str) -> str:
        """Record-wide name of one of `c`'s custom columns, declaring the schema first."""
        self.schema  # noqa: B018
        return self._custom_names[(c.name, attr)]

    def _custom_column(self, c: Components, record_attr: str) -> str | None:
        """`c`'s custom column written as `record_attr`, or None if it is not one."""
        self.schema  # noqa: B018
        return self._custom_columns.get((c.name, record_attr))

    @cached_property
    def dims(self) -> LazyFrames:
        """Axis frames, keyed by dim: `timestep`, `period`, `scenario`, `entity`, `shape_type`."""
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
        keys = (*keys, _ENTITY)
        if not self._shapes.static.empty:
            keys = (*keys, SHAPE_TYPE)
        return LazyFrames(keys, lambda key: self._dim_frame(key, axes))

    def _dim_frame(self, key: str, axes: dict[str, pd.DataFrame]) -> Any:
        if key == _ENTITY:
            return nw.from_native(self._entity_axis_frame()).lazy()
        if key == SHAPE_TYPE:
            types = _static_once(self._shapes)["type"].astype(str).unique()
            return nw.from_native(pd.DataFrame({SHAPE_TYPE: types})).lazy()
        return nw.from_native(axes[key]).lazy()

    def _entity_axis_frame(self) -> pd.DataFrame:
        """`(entity, entity_type, deleted)` across every exported type."""
        frames = []
        for c in self._components:
            names = _name_level(c.static.index)
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
            if col in ports or col in ("g_pu", "b_pu", CARRIER_ATTR):
                continue
            if col not in defaults.index:
                if col in custom_series or self._custom_in_long_file(c, col):
                    # Also time-varying, or declared so record-wide, so it
                    # lives in its long file only, like a registry attribute
                    # whose `varying` flag routes it there.
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
        return frame.rename(
            columns={col: self._custom_record_name(c, col) for col in custom}
        )

    @cached_property
    def groups(self) -> LazyFrames:
        """The `connection`, `carrier` and `shape` groups' rows, one frame each across every type.

        `carrier` and `shape` are present only when they have rows.
        """
        builders = {
            _CONNECTION: self._connection_frame,
            CARRIER: lambda: self._carrier_rows,
            SHAPE: self._shape_frame,
        }
        keys = [_CONNECTION]
        if not self._carrier_rows.empty:
            keys.append(CARRIER)
        if not self._shapes.static.empty:
            keys.append(SHAPE)
        return LazyFrames(
            tuple(keys), lambda key: nw.from_native(builders[key]()).lazy()
        )

    def _connection_frame(self) -> pd.DataFrame:
        frames = [
            self._port_rows(c)[[_ENTITY, _BUS]] for c in self._components if c.ports
        ]
        if not frames:
            return pd.DataFrame(columns=[_ENTITY, _BUS])
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    @cached_property
    def _carrier_rows(self) -> pd.DataFrame:
        """`(entity, carrier)` for every component naming a declared carrier.

        An undeclared carrier is an empty string or the type's default
        (`_check_carrier_references` refused anything else), and gets no row.
        """
        declared = set(
            _name_level(self.n.components["Carrier"].static.index).astype(str)
        )
        frames = []
        for c in self._components:
            if CARRIER_ATTR not in c.static.columns:
                continue
            carriers = _static_once(c)[CARRIER_ATTR].astype(str)
            carriers = carriers[carriers.isin(declared)]
            if carriers.empty:
                continue
            rows = (
                carriers.rename(CARRIER).reset_index().rename(columns={"name": _ENTITY})
            )
            rows[_ENTITY] = rows[_ENTITY].astype(str)
            frames.append(rows[[_ENTITY, CARRIER]])
        if not frames:
            return pd.DataFrame(columns=[_ENTITY, CARRIER])
        return pd.concat(frames, ignore_index=True)

    def _shape_frame(self) -> pd.DataFrame:
        """`(entity, shape_type, geometry, shape_name, <custom>...)`, one row per shape.

        `idx` is the entity, `type` the kind, `component` is implied by the
        entity's type. Geometry is written as WKT.
        """
        c = self._shapes
        static = _static_once(c)
        custom = list(_custom_static_attrs(c))
        frame = pd.DataFrame(
            {
                _ENTITY: static["idx"].astype(str).to_numpy(),
                SHAPE_TYPE: static["type"].astype(str).to_numpy(),
                SHAPE_GEOMETRY: static["geometry"].to_wkt().to_numpy(),
                SHAPE_NAME: static.index.astype(str).to_numpy(),
            }
        )
        for col in custom:
            frame[col] = static[col].to_numpy()
        _cast_custom_string_columns(frame, custom)
        return frame

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
            if attr in ("name", CARRIER_ATTR):
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
        for attr in _custom_static_attrs(c):
            name = self._custom_record_name(c, attr)
            if (attr in diverging or self._custom_in_long_file(c, attr)) and (
                name not in seen
            ):
                seen.append(name)
        for attr in _custom_series_attrs(c):
            name = self._custom_record_name(c, attr)
            if name not in seen:
                seen.append(name)
        return seen

    def _custom_in_long_file(self, c: Components, attr: str) -> bool:
        """Whether a custom static column's record-wide spec is time-varying.

        True when it joined a registry attribute some other type varies in
        time, so the column's values are timestep-NULL rows of that long file
        rather than a member-frame column.
        """
        spec = self.schema.attributes[self._custom_record_name(c, attr)]
        return TIMESTEP in spec.dims

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

        A custom attribute is looked up through the names `schema` declared
        it under, whether it lives in `c.dynamic` (time-varying) or `c.static`
        (a scenario-diverging static column, or one routed to a long file).
        """
        defaults = c.defaults
        ports = port_columns(c)
        for attr in defaults.index:
            stem, port = ports.get(attr, (attr, None))
            if port is not None:
                continue
            if record_name(c.name, stem) == record_attr:
                return attr
        return self._custom_column(c, record_attr)

    def _stack_series(self, c: Components, wide: pd.DataFrame) -> pd.DataFrame:
        """Melt a `snapshots x components` frame into long rows."""
        stochastic = isinstance(wide.columns, pd.MultiIndex)
        wide = wide.rename_axis(columns=[SCENARIO, _ENTITY] if stochastic else _ENTITY)
        stacked = wide.stack(level=list(range(wide.columns.nlevels)), future_stack=True)
        long = stacked.rename("value").reset_index()
        if not isinstance(c.snapshots, pd.MultiIndex):
            long = long.rename(columns={"snapshot": TIMESTEP})
        return long
