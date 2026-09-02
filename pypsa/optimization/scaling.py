# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling helper."""

from __future__ import annotations

import logging
import re
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from linopy import Constraint, Model
    from linopy.matrices import MatrixAccessor

    from pypsa import Network

logger = logging.getLogger(__name__)

# - `energy` (MW / MWh), default 1024
# - `cost` (€), default 1024
# - `emissions` (tCO2), default 2**20
#
# (energy, cost, emissions)
_NOM = re.compile(r"(?!v_nom)(\w+_nom(_min|_max|_set|_mod)?|nom_(min|max)_\w+)")
_COLUMN_EXPONENTS = {
    # power flows (MW), energy stocks (MWh), capacities (*_nom, MW/MWh)
    "p_set": (-1, 0, 0),
    "q_set": (-1, 0, 0),
    "inflow": (-1, 0, 0),
    "p_dispatch_set": (-1, 0, 0),
    "p_store_set": (-1, 0, 0),
    "e_initial": (-1, 0, 0),
    "e_set": (-1, 0, 0),
    "state_of_charge_initial": (-1, 0, 0),
    "state_of_charge_set": (-1, 0, 0),
    "max_growth": (-1, 0, 0),
    "e_sum_min": (-1, 0, 0),
    "e_sum_max": (-1, 0, 0),
    _NOM: (-1, 0, 0),
    # cost per quantity (€/MW, €/MWh)
    "marginal_cost": (1, -1, 0),
    "spill_cost": (1, -1, 0),
    "marginal_cost_storage": (1, -1, 0),
    "capital_cost": (1, -1, 0),
    "overnight_cost": (1, -1, 0),
    "fom_cost": (1, -1, 0),
    # cost on a dimensionless binaries
    "start_up_cost": (0, -1, 0),
    "shut_down_cost": (0, -1, 0),
    "stand_by_cost": (0, -1, 0),
    # quadratic cost
    "marginal_cost_quadratic": (2, -1, 0),
    # carrier emissions (tCO2/MWh)
    "co2_emissions": (1, 0, -1),
}


class Scaler(NamedTuple):
    """Energy, cost and emissions base-unit factors plus unscaling rules.

    `rows`/`columns` additionally enable pow2 Ruiz equilibration of the
    built linopy model around the solve (see `equilibrated`).
    """

    energy: float
    cost: float
    emissions: float
    rows: bool = False
    columns: bool = False

    @classmethod
    def resolve(cls, scaling: bool | dict | None) -> Scaler | None:
        """Resolve the `scaling` argument into a `Scaler` or `None`."""
        if scaling is False or scaling is None:
            return None
        defaults: dict = {
            "energy": 1024.0,
            "cost": 1024.0,
            "emissions": 2.0**20,
        }
        if scaling is True:
            return cls(**defaults)
        if not isinstance(scaling, dict):
            msg = f"scaling must be a bool or dict, got {type(scaling).__name__}"
            raise TypeError(msg)
        flags = {"rows": False, "columns": False}
        unknown = set(scaling) - set(defaults) - set(flags)
        if unknown:
            msg = (
                f"unknown scaling factor(s) {sorted(unknown)}; "
                f"valid keys are {sorted(defaults) + sorted(flags)}"
            )
            raise ValueError(msg)
        for k in flags:
            if k in scaling:
                if not isinstance(scaling[k], bool):
                    msg = f"scaling flag {k!r} must be a bool, got {scaling[k]!r}"
                    raise TypeError(msg)
                flags[k] = scaling[k]
        for k in defaults:
            if k in scaling and not isinstance(scaling[k], (int, float)):
                msg = f"scaling factor {k!r} must be numeric, got {scaling[k]!r}"
                raise TypeError(msg)
        merged = {k: scaling.get(k, d) for k, d in defaults.items()}
        for k, v in merged.items():
            if v <= 0:
                msg = f"scaling factor {k!r} must be positive, got {v!r}"
                raise ValueError(msg)
        # Round to powers of two so scaling and unscaling are bit-exact.
        return cls(
            energy=2.0 ** round(np.log2(float(merged["energy"]))),
            cost=2.0 ** round(np.log2(float(merged["cost"]))),
            emissions=2.0 ** round(np.log2(float(merged["emissions"]))),
            **flags,
        )

    def _factor(self, exponents: tuple[int, int, int]) -> float:
        """Product of the base units raised to the given exponents."""
        e, c, m = exponents
        return self.energy**e * self.cost**c * self.emissions**m

    def _column_factor(self, col: str) -> float | None:
        """Multiplicative factor for an input column, or None to leave unchanged."""
        exponents = _COLUMN_EXPONENTS.get(col) or (
            _COLUMN_EXPONENTS[_NOM] if _NOM.fullmatch(col) else None
        )
        return self._factor(exponents) if exponents else None

    def variable_factor(self, attr: str) -> float:
        """Factor that converts a scaled solution variable back to original units."""
        # Dimensionless variables (commitment status, module counts, angles)
        # stay unscaled.
        dimensionless = {"status", "start_up", "shut_down", "n_mod", "phase_shift"}
        return 1.0 if attr in dimensionless else self.energy

    def dual_factor(self, prefix: str, suffix: str, n: Network) -> float:
        """Output factor for a constraint dual."""
        if prefix == "GlobalConstraint":
            gc = n.global_constraints
            if suffix in gc.index:
                gc_type = gc.at[suffix, "type"]
                if gc_type == "transmission_expansion_cost_limit":
                    return 1.0
                if gc_type == "primary_energy":
                    return self.cost / self.emissions
        return self.cost / self.energy

    @contextmanager
    def applied(self, n: Network) -> Iterator[None]:
        """Scale `n`'s inputs in place for the block, restoring exact originals on exit."""
        static, dynamic, constant = {}, {}, None
        for c in n.components:
            for col in c.static.columns:
                f = self._column_factor(col)
                if f is not None:
                    static[c.name, col] = c.static[col].copy()
                    c.static[col] = c.static[col] * f
            for col, df in c.dynamic.items():
                f = self._column_factor(col)
                if f is not None and df.shape[1]:
                    dynamic[c.name, col] = df.copy()
                    c.dynamic[col] = df * f

        # Scale each GlobalConstraint RHS budget by its unit (as in dual_factor)
        gc = n.global_constraints
        if len(gc):
            constant = gc["constant"].copy()
            is_cost = gc["type"] == "transmission_expansion_cost_limit"
            is_em = gc["type"] == "primary_energy"
            gc.loc[is_cost, "constant"] /= self.cost
            gc.loc[is_em, "constant"] /= self.emissions
            gc.loc[~(is_cost | is_em), "constant"] /= self.energy

        # Expose the active factors so raw constants can be scaled (e.g. p_init)
        n._scaling = self._asdict()
        try:
            yield
        finally:
            n._scaling = {"energy": 1.0, "cost": 1.0, "emissions": 1.0}
            for (cname, col), original in static.items():
                n.components[cname].static[col] = original
            for (cname, col), original in dynamic.items():
                n.components[cname].dynamic[col] = original
            if constant is not None:
                n.global_constraints["constant"] = constant


def _ruiz_pow2_exponents(
    mat: MatrixAccessor, rows: bool, columns: bool, max_iter: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Pow2 Ruiz equilibration exponents per constraint/variable label.

    Returns (rexp, cexp) arrays indexed by label, one slot longer than the
    largest label so filler entries (label -1) map to exponent 0.
    """
    A = mat.A
    rexp = np.zeros((mat.clabels.max() + 2) if len(mat.clabels) else 1, dtype=np.int64)
    cexp = np.zeros((mat.vlabels.max() + 2) if len(mat.vlabels) else 1, dtype=np.int64)
    if A is None:
        return rexp, cexp

    W = abs(A.tocsr(copy=True))
    nr, nc = W.shape
    row_of_nnz = np.repeat(np.arange(nr), np.diff(W.indptr))
    er = np.zeros(nr, dtype=np.int64)
    ec = np.zeros(nc, dtype=np.int64)
    # only continuous columns are scaled, integral domains stay intact
    col_ok = mat.vtypes == "C" if columns else np.zeros(nc, dtype=bool)

    for _ in range(max_iter):
        step_r = np.zeros(nr, dtype=np.int64)
        step_c = np.zeros(nc, dtype=np.int64)
        if rows:
            rmax = W.max(axis=1).toarray().ravel()
            nz = rmax > 0
            step_r[nz] = -np.round(np.log2(rmax[nz]) / 2).astype(np.int64)
        if col_ok.any():
            cmax = W.max(axis=0).toarray().ravel()
            nz = col_ok & (cmax > 0)
            step_c[nz] = -np.round(np.log2(cmax[nz]) / 2).astype(np.int64)
        if not step_r.any() and not step_c.any():
            break
        W.data *= np.exp2(step_r[row_of_nnz] + step_c[W.indices])
        er += step_r
        ec += step_c

    rexp[mat.clabels] = er
    cexp[mat.vlabels] = ec
    return rexp, cexp


def _apply_equilibration(
    m: Model, rexp: np.ndarray, cexp: np.ndarray, sign: int
) -> None:
    """Scale the model by 2^(sign*exponents); sign=-1 restores bit-exactly."""

    def fac(template: xr.DataArray, exp: np.ndarray) -> xr.DataArray:
        return template.copy(data=np.exp2(sign * exp[template.values]))

    for con in m.constraints.values():
        ds = con.data
        rfac = fac(ds["labels"], rexp)
        ds["coeffs"] = ds["coeffs"] * rfac * fac(ds["vars"], cexp)
        ds["rhs"] = ds["rhs"] * rfac
        if "dual" in ds:
            ds["dual"] = ds["dual"] * fac(ds["labels"], -rexp)
    for var in m.variables.values():
        ds = var.data
        cfac = fac(ds["labels"], -cexp)
        ds["lower"] = ds["lower"] * cfac
        ds["upper"] = ds["upper"] * cfac
        if "solution" in ds:
            # x = C x': the solution maps back with the same factor as bounds
            ds["solution"] = ds["solution"] * cfac
    obj = m.objective.expression.data
    obj["coeffs"] = obj["coeffs"] * fac(obj["vars"], cexp)


@contextmanager
def equilibrated(m: Model, rows: bool = True, columns: bool = True) -> Iterator[None]:
    """Pow2 Ruiz-equilibrate the built linopy model in place around a solve.

    Coefficients, rhs, bounds and objective are scaled by per-row/per-column
    powers of two (bit-exact, integral variables untouched) and restored on
    exit; solution and dual values are mapped back to original units. Call
    `m.constraints.sanitize_zeros()` before and solve with
    `sanitize_zeros=False` so the zero-drop never sees scaled coefficients.
    """
    mat = m.matrices
    if mat.indicator_A is not None:
        logger.warning("equilibration skipped: model has indicator constraints")
        yield
        return
    rexp, cexp = _ruiz_pow2_exponents(mat, rows, columns)
    _apply_equilibration(m, rexp, cexp, +1)
    logger.info(
        "equilibrated model: row exponents [%d, %d], column exponents [%d, %d]",
        rexp.min(),
        rexp.max(),
        cexp.min(),
        cexp.max(),
    )
    try:
        yield
    finally:
        _apply_equilibration(m, rexp, cexp, -1)


def _valid_abs_coeffs(con: Constraint) -> xr.DataArray:
    """|coeffs| with linopy filler terms (vars == -1) and genuine zeros masked."""
    data = con.data
    return abs(data["coeffs"]).where((data["vars"] != -1) & (data["coeffs"] != 0))


def scaling_report(m: Model) -> pd.DataFrame:
    """Absolute nonzero numerical ranges of a linopy model, per group.

    One row per constraint group (`coeff_min/coeff_max/rhs_min/rhs_max`),
    per variable group (`bound_min/bound_max`, infinities excluded) and one
    for the objective coefficients. Filler terms and masked rows are
    excluded; linopy's `coefficientrange` is signed and filler-polluted.
    """
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN groups
        for name, con in m.constraints.items():
            live = con.data["labels"] != -1
            absc = _valid_abs_coeffs(con).where(live)
            absr = abs(con.data["rhs"])
            absr = absr.where(live & np.isfinite(absr) & (absr != 0))
            rows[("constraint", name)] = {
                "coeff_min": float(absc.min()),
                "coeff_max": float(absc.max()),
                "rhs_min": float(absr.min()),
                "rhs_max": float(absr.max()),
            }
        for name, var in m.variables.items():
            bounds = xr.concat(
                [abs(var.data["lower"]), abs(var.data["upper"])], dim="_bound"
            )
            bounds = bounds.where(
                (var.data["labels"] != -1) & np.isfinite(bounds) & (bounds != 0)
            )
            rows[("variable", name)] = {
                "bound_min": float(bounds.min()),
                "bound_max": float(bounds.max()),
            }
        # objective split per variable group: one unit each, so mixed groups
        # (e.g. the objective-constant variable) don't hide the true range
        flat = m.objective.expression.flat
        flat = flat[(flat["coeffs"] != 0) & (flat["vars"] != -1)]
        absc = flat["coeffs"].abs()
        for name, var in m.variables.items():
            labels = var.data["labels"].to_numpy().ravel()
            sel = flat["vars"].isin(labels[labels != -1])
            if sel.any():
                rows[("objective", name)] = {
                    "coeff_min": float(absc[sel].min()),
                    "coeff_max": float(absc[sel].max()),
                }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["kind", "name"])
    return df
