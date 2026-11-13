# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Duration attributes in snapshots or hours for the optimisation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pypsa._options import options
from pypsa.optimization.window import snapshot_array

if TYPE_CHECKING:
    import pandas as pd
    from xarray import DataArray

    from pypsa import Components, Network

LEGACY_DURATION_ATTRS = frozenset(
    {"ramp_limit_up", "ramp_limit_down", "min_up_time", "min_down_time"}
)
_TOL = 1e-9
_RAMP_ATTRS = (
    "ramp_limit_up",
    "ramp_limit_down",
    "ramp_limit_start_up",
    "ramp_limit_shut_down",
)


def in_hours(attr: str) -> bool:
    """Whether `attr` is read in hours (the option is on or `attr` is new)."""
    return bool(options.params.optimize.durations_in_hours) or (
        attr not in LEGACY_DURATION_ATTRS
    )


def flagged_names(mask: DataArray) -> list[str]:
    """Names of the assets for which `mask` holds in any scenario."""
    flagged = mask.any([d for d in mask.dims if d != "name"])
    return flagged.indexes["name"][flagged.values].tolist()


def _hours_n_lags(weights: np.ndarray, max_duration: float) -> int:
    """Upper bound of the lags to test, safe for zero weightings."""
    if not len(weights) or max_duration <= 0:
        return 1
    # lags into the network, plus those before it (spaced by the first weighting)
    cap = len(weights) + 1
    if weights[0] > 0:
        cap += int(np.ceil(max_duration / weights[0]))
    positive = weights[weights > 0]
    if not len(positive):
        return cap
    # each zero weighting adds a lag without adding hours
    n_zero = int((weights == 0).sum())
    return min(int(np.ceil(max_duration / positive.min())) + 1 + n_zero, cap)


def duration_lookback(
    n: Network,
    c: Components,
    duration: DataArray,
    sns: pd.Index,
    *,
    hours: bool,
) -> DataArray:
    """Mask of the earlier snapshots that count for each snapshot of the window.

    Parameters
    ----------
    n : Network
        Network whose snapshots and weightings define the time axis.
    c : Components
        Component the duration belongs to.
    duration : DataArray
        Duration per component, in hours if `hours` else in snapshots.
    sns : pd.Index
        Snapshots of the window.
    hours : bool
        Count hours of the 'generators' weighting instead of snapshots.

    Returns
    -------
    DataArray
        Bool array with dims `snapshot`, `lag` and the dims of `duration`.
        Lag k refers to the snapshot k steps before the snapshot. Lags beyond
        its position in the window reach into the history before the window.

    """
    max_duration = float(duration.max()) if duration.size else 0.0
    if not hours:
        n_lags = max(int(np.ceil(max_duration)), 1)
        lag = xr.DataArray(
            np.arange(n_lags), dims="lag", coords={"lag": np.arange(n_lags)}
        )
        mask = lag < duration
        mask = mask & snapshot_array(np.ones(len(sns), dtype=bool), sns)
        return mask.transpose("snapshot", "lag", ...)

    weights = n.snapshot_weightings.generators.to_numpy(dtype=float)
    n_lags = _hours_n_lags(weights, max_duration)
    # start hour of every position, extended before the first snapshot
    starts = np.concatenate([[0.0], np.cumsum(weights)])[:-1]
    pos = n.snapshots.get_indexer(sns)
    k = np.arange(n_lags)
    src = pos[:, None] - k[None, :]
    start_src = np.where(src >= 0, starts[np.clip(src, 0, None)], src * weights[0])
    elapsed = starts[pos][:, None] - start_src
    ok = np.ones_like(src, dtype=bool)
    if n.has_periods:
        periods = np.asarray(n.snapshots.get_level_values("period"))
        src_period = periods[np.clip(src, 0, None)]
        same = src_period == periods[pos][:, None]
        # time before the network belongs to the first period
        ok = np.where(src < 0, periods[pos][:, None] == periods[0], same)
    elapsed_da = xr.DataArray(elapsed, coords=[sns, k], dims=("snapshot", "lag"))
    ok_da = xr.DataArray(ok, coords=[sns, k], dims=("snapshot", "lag"))
    mask = (elapsed_da < duration - _TOL) & ok_da
    # drop trailing lags that count for no snapshot and no component
    used = mask.any([d for d in mask.dims if d != "lag"]).to_numpy()
    keep = int(np.flatnonzero(used).max()) + 1 if used.any() else 1
    return mask.isel(lag=slice(keep)).transpose("snapshot", "lag", ...)


def history_length(lookback: DataArray, sns: pd.Index) -> int:
    """Largest number of snapshots before the window that any snapshot counts."""
    counted = lookback.any([d for d in lookback.dims if d not in ("snapshot", "lag")])
    deepest = (counted * (counted.lag + 1)).max("lag").to_numpy()
    return max(int((deepest - np.arange(len(sns))).max(initial=0)), 0)


def ramp_weighting(n: Network, sns: pd.Index) -> DataArray:
    """Factor that scales the ramp limit rows of each snapshot.

    Parameters
    ----------
    n : Network
        Network whose 'generators' snapshot weightings are used.
    sns : pd.Index
        Snapshots of the window.

    Returns
    -------
    DataArray
        Array over `snapshot` labelled by `sns`: the 'generators' weighting if
        ramp limits are read in hours, else ones.

    """
    if in_hours("ramp_limit_up"):
        weights = n.snapshot_weightings.generators.loc[sns].to_numpy(dtype=float)
    else:
        weights = np.ones(len(sns))
    return snapshot_array(weights, sns)


def _has_duration_dependent_limits(n: Network, sns: pd.Index) -> bool:
    """Whether an active Generator, Link or Process has a ramp limit or minimum time."""
    for name in ("Generator", "Link", "Process"):
        c = n.components[name]
        if c.empty:
            continue
        active = c.da.active.sel(snapshot=sns).any("snapshot")
        for attr in _RAMP_ATTRS:
            if attr in c.static.columns:
                limit = c.da[attr].sel(snapshot=sns)
                if bool((np.isfinite(limit) & active).any()):
                    return True
        if "committable" in c.static.columns:
            on = active & c.da.committable
            for attr in ("min_up_time", "min_down_time"):
                if attr in c.static.columns and bool((on & (c.da[attr] > 0)).any()):
                    return True
    return False


def warn_legacy_durations(n: Network, sns: pd.Index) -> None:
    """Warn if ramp limits or minimum times are read in snapshots but matter.

    Parameters
    ----------
    n : Network
        Network to be optimised.
    sns : pd.Index
        Snapshots of the optimisation.

    """
    if options.params.optimize.durations_in_hours:
        return
    weights = n.snapshot_weightings.generators.loc[sns]
    if (weights == 1).all():
        return
    if not _has_duration_dependent_limits(n, sns):
        return
    warnings.warn(
        "Ramp limits and minimum up/down times are currently read per snapshot, "
        "ignoring the snapshot weightings. With PyPSA 2.0 they will be read per "
        "hour of the 'generators' snapshot weighting. Set "
        "`pypsa.options.params.optimize.durations_in_hours = True` to adopt this "
        "now and silence this warning.",
        FutureWarning,
        stacklevel=3,
    )
