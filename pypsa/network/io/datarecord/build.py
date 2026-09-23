# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""`network_from_record`: fill an empty `Network` from a datarecord `Record`.

The import side of the datarecord format, undoing `NetworkRecord`: the
record's wide member frames and long rows are reassembled into PyPSA's
static/dynamic split, one component type at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from pyproj import CRS

from pypsa.descriptors import _update_ports_component_attrs
from pypsa.network.io.datarecord.record import NETWORK_ATTRS
from pypsa.network.io.datarecord.schema import (
    ENTITY_TYPE,
    PERIOD,
    PERIOD_WEIGHTINGS,
    PORT,
    SCENARIO,
    SCENARIO_WEIGHTINGS,
    SNAPSHOT_WEIGHTINGS,
    TIMESTEP,
    port_columns,
    record_name,
)

if TYPE_CHECKING:
    from datarecord.record import RecordLike as Record

    from pypsa import Network
    from pypsa.components.components import Components

_ENTITY, _BUS = "entity", "bus"


def _collect(frame: Any) -> pd.DataFrame:
    """Collect a record frame (a lazy narwhals frame) into a pandas DataFrame."""
    return frame.collect(backend="pandas").to_native()


def _apply_meta(record: Record, n: Network) -> None:
    """Set `NETWORK_ATTRS`, `crs` and `meta` from `schema.meta["pypsa"]`."""
    meta = record.schema.meta.get("pypsa") or {}
    attrs = meta.get("attributes") or {}
    for key in NETWORK_ATTRS:
        if key not in attrs or attrs[key] is None:
            continue
        # `pypsa_version` has no public setter. Every other allow-listed
        # attribute (`name`, `_multi_invest`, `_objective`, `_objective_constant`)
        # does, or is itself a plain private attribute.
        if key == "pypsa_version":
            n._pypsa_version = attrs[key]
        else:
            setattr(n, key, attrs[key])
    crs = meta.get("crs")
    if crs is not None:
        n._crs = CRS.from_wkt(crs)
    n.meta = dict(meta.get("meta") or {})


def _apply_axes(record: Record, n: Network) -> tuple[bool, bool]:
    """Set snapshots, investment periods and scenarios, and report their shape."""
    dims = record.dims
    timestep = _collect(dims[TIMESTEP])
    multiperiod = PERIOD in dims and not _collect(dims[PERIOD]).empty

    index_cols = [PERIOD, TIMESTEP] if multiperiod else [TIMESTEP]
    snapshots = (
        pd.MultiIndex.from_frame(timestep[index_cols])
        if multiperiod
        else pd.Index(timestep[TIMESTEP])
    )
    n.set_snapshots(snapshots)
    weight_cols = [col for col in SNAPSHOT_WEIGHTINGS if col in timestep.columns]
    if weight_cols:
        weights = timestep.set_index(index_cols)[weight_cols]
        weights.index.names = n.snapshot_weightings.index.names
        n.snapshot_weightings = weights.reindex(
            index=n.snapshots, columns=list(n.snapshot_weightings.columns)
        )

    if multiperiod:
        inverse = {v: k for k, v in PERIOD_WEIGHTINGS.items()}
        periods = _collect(dims[PERIOD]).rename(columns=inverse).set_index(PERIOD)
        n.periods = periods.index
        n._investment_periods_data = periods.reindex(n.investment_periods)

    stochastic = SCENARIO in dims and not _collect(dims[SCENARIO]).empty
    if stochastic:
        scenarios = _collect(dims[SCENARIO])
        weight_col = SCENARIO_WEIGHTINGS["weight"]
        weights = scenarios.set_index(SCENARIO)[weight_col].rename("weight")
        weights.index = weights.index.astype(str).rename("scenario")
        n.set_scenarios(weights)

    return multiperiod, stochastic


def _bus_columns(c: Components) -> dict[str, str]:
    """Port label -> PyPSA `bus0`/`bus1`/... column name, for one type."""
    return {port: col for col, (stem, port) in port_columns(c).items() if stem == _BUS}


def _pivot_buses(
    static: pd.DataFrame, ports: pd.DataFrame, c: Components
) -> pd.DataFrame:
    """Add `bus0`/`bus1`/... columns to `static` from the `port` connection rows."""
    bus_cols = _bus_columns(c)
    if not bus_cols or ports.empty:
        return static
    mine = ports[ports[_ENTITY].isin(static[_ENTITY])]
    if mine.empty:
        return static
    # A connection's bus never varies by scenario, but a stochastic record's
    # read path broadcasts a scenario-partial row once per scenario. Collapse
    # back to one row per (entity, port) before pivoting.
    mine = mine.drop_duplicates(subset=[_ENTITY, "value"])
    wide = mine.pivot(index=_ENTITY, columns="value", values=_BUS)
    wide = wide.rename(columns=bus_cols).reset_index()
    keep = [_ENTITY, *(col for col in bus_cols.values() if col in wide.columns)]
    return static.merge(wide[keep], on=_ENTITY, how="left")


def _broadcast_scenarios(static: pd.DataFrame, scenarios: pd.Index) -> pd.DataFrame:
    """One row per entity -> one row per `(scenario, entity)`, scenario-major."""
    broadcast = pd.concat(dict.fromkeys(scenarios, static), names=["scenario"])
    return broadcast.reset_index("scenario").reset_index(drop=True)


def _piecewise_wide(rows: pd.DataFrame, c: Components, attr: str) -> pd.DataFrame:
    """`(entity, breakpoint, value)` rows as the MultiIndex frame breakpoint import expects."""
    rows = rows.sort_values([_ENTITY, "breakpoint"], kind="stable")
    rows = rows.assign(_seg=rows.groupby(_ENTITY).cumcount())
    x_attr = c._piecewise_schema(attr).x
    x_wide = rows.pivot(index="_seg", columns=_ENTITY, values="breakpoint")
    y_wide = rows.pivot(index="_seg", columns=_ENTITY, values="value")
    x_wide.columns = pd.MultiIndex.from_product(
        [x_wide.columns, [x_attr]], names=["name", "attribute"]
    )
    y_wide.columns = pd.MultiIndex.from_product(
        [y_wide.columns, [attr]], names=["name", "attribute"]
    )
    return pd.concat([x_wide, y_wide], axis=1)


def _is_stochastic(rows: pd.DataFrame) -> bool:
    """Whether `rows` is scenario-keyed, raise if it mixes keyed and null rows."""
    if "scenario" not in rows.columns:
        return False
    keyed = rows["scenario"].notna()
    if keyed.any() and not keyed.all():
        msg = (
            "rows mix scenario-keyed and scenario-null values in the 'scenario' column"
        )
        raise ValueError(msg)
    return bool(keyed.all())


def _series_wide(
    rows: pd.DataFrame, n: Network, c: Components, *, multiperiod: bool
) -> pd.DataFrame:
    """Long rows pivoted to `n.snapshots x entity`, columns ordered by the static index.

    `pivot` sorts its result columns alphabetically, and the format guarantees
    member order only for axes and groups, never attribute files, so the
    column order is recovered from `c.static.index` instead (the entity and,
    for a stochastic network, the `(scenario, name)` order).
    """
    stochastic = _is_stochastic(rows)
    key_cols = [SCENARIO, _ENTITY] if stochastic else [_ENTITY]
    if multiperiod:
        wide = rows.pivot(index=[PERIOD, TIMESTEP], columns=key_cols, values="value")
        wide.index = wide.index.set_names(["period", "timestep"])
    else:
        wide = rows.pivot(index=TIMESTEP, columns=key_cols, values="value")
        wide.index = wide.index.rename("snapshot")
    wide = wide.reindex(n.snapshots)
    wide.columns = wide.columns.set_names(
        ["scenario", "name"] if stochastic else "name"
    )
    order = c.static.index[c.static.index.isin(wide.columns)]
    return wide.reindex(columns=order)


def _scalar_values(rows: pd.DataFrame, *, stochastic: bool) -> pd.Series:
    """Scalar rows as an `entity`- or `(scenario, entity)`-indexed value series."""
    if stochastic and _is_stochastic(rows):
        indexed = rows.set_index(["scenario", _ENTITY])["value"]
    else:
        indexed = rows.set_index(_ENTITY)["value"]
    return indexed


def _split_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One attribute's long rows split into (scalar, series, piecewise)."""
    is_series = (
        rows["timestep"].notna()
        if "timestep" in rows.columns
        else pd.Series(False, rows.index)
    )
    is_piecewise = (
        rows["breakpoint"].notna()
        if "breakpoint" in rows.columns
        else pd.Series(False, rows.index)
    )
    return rows[~is_series & ~is_piecewise], rows[is_series], rows[is_piecewise]


def _attr_rows(
    cache: dict[str, pd.DataFrame], record: Record, kind: str, record_attr: str
) -> pd.DataFrame:
    """Fetch cached long rows for `record_attr`, once regardless of type."""
    key = f"{kind}:{record_attr}"
    if key not in cache:
        frames = record.attributes if kind == "inputs" else record.outputs
        cache[key] = _collect(frames[record_attr])
    return cache[key]


def _filter_rows(
    rows: pd.DataFrame, entities: pd.Series, bus: pd.Series | None
) -> pd.DataFrame:
    """Scope long rows to one type's entities, and one port's bus where given."""
    rows = rows[rows[_ENTITY].isin(entities)]
    if bus is None:
        return rows
    # `entities`/`bus` may be scenario-broadcast (one row per (scenario, entity)),
    # repeating each (entity, bus) pair once per scenario. A connection's bus
    # itself never varies by scenario, so drop the repeats before joining.
    wanted = pd.DataFrame(
        {_ENTITY: entities.to_numpy(), _BUS: bus.astype(str).to_numpy()}
    ).drop_duplicates()
    return rows.merge(wanted, on=[_ENTITY, _BUS])


def _add_component_type(
    n: Network,
    c: Components,
    static: pd.DataFrame,
    cache: dict[str, pd.DataFrame],
    record: Record,
    *,
    multiperiod: bool,
    stochastic: bool,
) -> None:
    """Assign one type's static frame (inputs), then its series and piecewise data."""
    ctype = c.name
    defaults = c.defaults
    ports = port_columns(c)
    bus_cols = _bus_columns(c)

    deferred: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for attr in defaults.index:
        if attr == "name" or attr in static.columns:
            continue
        if str(defaults.at[attr, "status"]).startswith("Output"):
            continue
        stem, port = ports.get(attr, (attr, None))
        record_attr = record_name(ctype, stem) if port is None else stem
        if record_attr not in record.attributes:
            continue
        rows = _attr_rows(cache, record, "inputs", record_attr)
        bus = static[bus_cols[port]] if port is not None and port in bus_cols else None
        rows = _filter_rows(rows, static[_ENTITY], bus)
        if rows.empty:
            continue
        scalar, series, piecewise = _split_rows(rows)
        if not scalar.empty:
            values = _scalar_values(scalar, stochastic=stochastic)
            key = (
                pd.MultiIndex.from_frame(static[["scenario", _ENTITY]])
                if isinstance(values.index, pd.MultiIndex)
                else static[_ENTITY]
            )
            static[attr] = key.map(values)
        if not series.empty or not piecewise.empty:
            deferred.append((attr, series, piecewise))

    if stochastic:
        static = static.set_index(["scenario", _ENTITY])
        static.index.names = ["scenario", "name"]
    else:
        static = static.set_index(_ENTITY)
        static.index.name = "name"
    n._import_components_from_df(static, ctype)

    for attr, series, piecewise in deferred:
        if not series.empty:
            wide = _series_wide(series, n, c, multiperiod=multiperiod)
            n._import_series_from_df(wide, ctype, attr)
        if not piecewise.empty:
            wide = _piecewise_wide(piecewise, c, attr)
            n._import_piecewise_from_df(wide, ctype, attr)


def _add_outputs(
    n: Network,
    c: Components,
    cache: dict[str, pd.DataFrame],
    record: Record,
    *,
    multiperiod: bool,
) -> None:
    """Assign one type's output attributes, after its inputs are in place."""
    ctype = c.name
    defaults = c.defaults
    ports = port_columns(c)
    bus_cols = _bus_columns(c)
    static = c.static
    entities = pd.Series(
        static.index.get_level_values("name")
        if isinstance(static.index, pd.MultiIndex)
        else static.index,
    ).reset_index(drop=True)

    for attr in defaults.index:
        if not str(defaults.at[attr, "status"]).startswith("Output"):
            continue
        stem, port = ports.get(attr, (attr, None))
        record_attr = record_name(ctype, stem) if port is None else stem
        if record_attr not in record.outputs:
            continue
        rows = _attr_rows(cache, record, "outputs", record_attr)
        bus = (
            pd.Series(static[bus_cols[port]].to_numpy())
            if port is not None and port in bus_cols
            else None
        )
        rows = _filter_rows(rows, entities, bus)
        if rows.empty:
            continue
        scalar, series, _piecewise = _split_rows(rows)
        if not scalar.empty:
            values = _scalar_values(scalar, stochastic=n.has_scenarios)
            key = (
                pd.MultiIndex.from_frame(
                    static.index.to_frame(index=False)[["scenario", "name"]]
                )
                if isinstance(values.index, pd.MultiIndex)
                else pd.Index(entities)
            )
            default = defaults.at[attr, "default"]
            c.static[attr] = key.map(values).to_numpy()
            c.static[attr] = c.static[attr].fillna(default)
        if not series.empty:
            wide = _series_wide(series, n, c, multiperiod=multiperiod)
            n._import_series_from_df(wide, ctype, attr, overwrite=True)


def network_from_record(record: Record, n: Network) -> None:
    """Fill an empty `n` in place from `record`.

    Parameters
    ----------
    record
        A datarecord `Record` built by `NetworkRecord` (or an equivalent one).
    n
        A freshly constructed `Network`, filled in place.

    """
    _apply_meta(record, n)
    multiperiod, stochastic = _apply_axes(record, n)

    ports = (
        _collect(record.attributes[PORT])
        if PORT in record.attributes
        else pd.DataFrame(columns=[_ENTITY, _BUS, "value"])
    )
    cache: dict[str, pd.DataFrame] = {}

    remaining = sorted(set(record.entity_types) - {"Bus", "Carrier"})
    ctypes = [t for t in ("Bus", "Carrier") if t in record.entity_types] + remaining

    for ctype in ctypes:
        c = n.components[ctype]
        static = _collect(record.entity_types[ctype])
        if ENTITY_TYPE in static.columns:
            static = static.drop(columns=[ENTITY_TYPE])
        if not stochastic and "scenario" in static.columns:
            static = static.drop(columns=["scenario"])
        # An all-default column is written all-NaN, with no concrete dtype for
        # parquet to record. Dropping it here lets `_import_components_from_df`
        # recreate it from the registry default, with the right dtype.
        all_null = [
            col for col in static.columns if col != _ENTITY and static[col].isna().all()
        ]
        static = static.drop(columns=all_null)
        static = _pivot_buses(static, ports, c)
        if stochastic:
            static = _broadcast_scenarios(static, n.scenarios)

        if ctype in ("Link", "Process"):
            _update_ports_component_attrs(n, where=static.columns, c_name=ctype)

        _add_component_type(
            n, c, static, cache, record, multiperiod=multiperiod, stochastic=stochastic
        )
        _add_outputs(n, c, cache, record, multiperiod=multiperiod)

    n._broadcast_standard_types()
