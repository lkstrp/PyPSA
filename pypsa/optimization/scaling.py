# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Numerical scaling helper."""

from __future__ import annotations

import logging
import re
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from scipy.optimize import Bounds, LinearConstraint, milp

if TYPE_CHECKING:
    from collections.abc import Iterator

    from linopy import Constraint, Model

    from pypsa import Network

logger = logging.getLogger(__name__)

ColumnClass = Literal["energy", "cost", "none"]

# log2 range every scaled quantity is pulled into, per category
WINDOW: dict[str, tuple[float, float]] = {
    "matrix": (1e-3, 1e6),
    "cost": (1e-2, 1e6),
    "bound": (1e-2, 1e6),
    "rhs": (1e-2, 1e6),
}
WEIGHT: dict[str, float] = {"matrix": 2.0, "cost": 1.0, "bound": 1.0, "rhs": 1.0}
VIOL_WEIGHT = 10.0
EPS_ONE = 1e-3
G_MAX = 40

_DIMENSIONLESS_SUFFIXES = (
    "-status",
    "-start_up",
    "-shut_down",
    "-maintenance",
    "-maintenance_start",
    "-n_mod",
)
_COST_COLUMNS = {"CVaR-a", "CVaR-theta", "CVaR"}


class ScalingSpec(NamedTuple):
    """Resolved `scaling` argument. `None` pins mean "let the ILP choose"."""

    energy: int | None
    cost: int | None
    rows: bool


class ScalingExponents(NamedTuple):
    """Chosen log2 exponents: one per column class and one per constraint group."""

    energy: int
    cost: int
    rows: dict[str, int]


def resolve_scaling(scaling: bool | dict | None) -> ScalingSpec | None:
    """Turn the user-facing `scaling` argument into a `ScalingSpec` or `None`."""
    if scaling is False or scaling is None:
        return None
    if scaling is True:
        return ScalingSpec(None, None, True)
    if not isinstance(scaling, dict):
        msg = f"scaling must be a bool or dict, got {type(scaling).__name__}"
        raise TypeError(msg)
    valid = ["cost", "energy", "rows"]
    unknown = set(scaling) - set(valid)
    if unknown:
        msg = f"unknown scaling key(s) {sorted(unknown)}; valid keys are {valid}"
        raise ValueError(msg)
    rows = scaling.get("rows", True)
    if not isinstance(rows, bool):
        msg = f"scaling key 'rows' must be a bool, got {rows!r}"
        raise TypeError(msg)
    pins: dict[str, int | None] = {}
    for k in ("energy", "cost"):
        v = scaling.get(k)
        if v is None:
            pins[k] = None
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            msg = f"scaling factor {k!r} must be numeric, got {v!r}"
            raise TypeError(msg)
        if v <= 0:
            msg = f"scaling factor {k!r} must be positive, got {v!r}"
            raise ValueError(msg)
        pins[k] = int(round(np.log2(float(v))))
    return ScalingSpec(pins["energy"], pins["cost"], rows)


def classify_columns(m: Model) -> dict[str, ColumnClass]:
    """Classify every variable group as energy, cost or dimensionless."""
    classes: dict[str, ColumnClass] = {}
    for name, var in m.variables.items():
        attrs = var.attrs
        if (
            attrs.get("integer")
            or attrs.get("binary")
            or name.endswith(_DIMENSIONLESS_SUFFIXES)
            or name == "Transformer-phase_shift"
        ):
            classes[name] = "none"
        elif name in _COST_COLUMNS:
            classes[name] = "cost"
        else:
            classes[name] = "energy"
    return classes


def _label_classes(m: Model, classes: dict[str, ColumnClass]) -> np.ndarray:
    """Per-variable-label class code (0 energy, 1 cost, 2 none), filler -1 -> 2."""
    codes = {"energy": 0, "cost": 1, "none": 2}
    size = max((int(v.labels.max()) for _, v in m.variables.items()), default=-1) + 2
    out = np.full(size, 2, dtype=np.int8)
    for name, var in m.variables.items():
        labels = var.labels.values.ravel()
        out[labels[labels != -1]] = codes[classes[name]]
    return out


def _group_ranges(m: Model, classes: dict[str, ColumnClass] | None) -> pd.DataFrame:
    """Nonzero |coeff| range per constraint group, split by column class if given.

    Index `(name, cls)` with `cls` in energy/cost/none, or `"all"` unsplit.
    """
    label_cls = _label_classes(m, classes) if classes else None
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, con in m.constraints.items():
            absc = _valid_abs_coeffs(con).where(con.data["labels"] != -1)
            if label_cls is None:
                parts = {"all": absc}
            else:
                cls_of = label_cls[con.data["vars"].values]
                parts = {
                    c: absc.where(cls_of == i)
                    for i, c in enumerate(("energy", "cost", "none"))
                }
            for cls, part in parts.items():
                lo, hi = float(part.min()), float(part.max())
                if np.isfinite(lo):
                    rows[(name, cls)] = {"coeff_min": lo, "coeff_max": hi}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["name", "cls"])
    return df


def _quantities(m: Model) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """ILP quantities: log2 values, exponent matrix over (g_e, g_c, r_k), categories.

    Row i of the matrix says how quantity i moves under the unknowns; the
    scaled quantity is `log2 v + A @ g`.
    """
    classes = classify_columns(m)
    report = scaling_report(m)
    names = list(m.constraints)
    kidx = {name: 2 + k for k, name in enumerate(names)}
    ncol = 2 + len(names)
    col_vec = {"energy": (1, 0), "cost": (0, 1), "none": (0, 0)}

    logv: list[float] = []
    arows: list[np.ndarray] = []
    cats: list[str] = []

    def add(v: float, a: np.ndarray, cat: str) -> None:
        logv.append(float(np.log2(v)))
        arows.append(a)
        cats.append(cat)

    for (name, cls), r in _group_ranges(m, classes).iterrows():
        a = np.zeros(ncol)
        a[:2] = col_vec[cls]
        a[kidx[name]] = 1
        add(r["coeff_min"], a, "matrix")
        add(r["coeff_max"], a, "matrix")
    con_rep = report.xs("constraint")
    for name, r in con_rep.iterrows():
        if np.isfinite(r["rhs_min"]):
            a = np.zeros(ncol)
            a[kidx[name]] = 1
            add(r["rhs_min"], a, "rhs")
            add(r["rhs_max"], a, "rhs")
    for name, r in report.xs("variable").iterrows():
        if np.isfinite(r["bound_min"]):
            a = np.zeros(ncol)
            a[:2] = -np.array(col_vec[classes[name]])
            add(r["bound_min"], a, "bound")
            add(r["bound_max"], a, "bound")
    if "objective" in report.index.get_level_values("kind"):
        for name, r in report.xs("objective").iterrows():
            a = np.zeros(ncol)
            a[:2] = col_vec[classes[name]]
            a[1] -= 1
            add(r["coeff_min"], a, "cost")
            add(r["coeff_max"], a, "cost")
    return np.array(logv), np.array(arows).reshape(len(logv), ncol), cats


def choose_exponents(m: Model, spec: ScalingSpec) -> ScalingExponents:
    """Pick pow2 exponents by an ILP over the model's per-group ranges.

    Minimises the weighted log2 spread per category plus window violations,
    with a small pull of every exponent towards zero. Pins in `spec` fix the
    corresponding unknowns.
    """
    names = list(m.constraints)
    zero = ScalingExponents(0, 0, dict.fromkeys(names, 0))
    if m.matrices.indicator_A is not None:
        logger.warning("scaling skipped: model has indicator constraints")
        return zero
    logv, A, cats = _quantities(m)
    if not len(logv):
        return zero

    ng = A.shape[1]
    nq = len(logv)
    cat_list = list(WINDOW)
    cat_idx = np.array([cat_list.index(c) for c in cats])
    scaled = A.any(axis=1)
    # unknown layout: g (ng) | t=|g| (ng) | lo/hi per category (2*ncat) | s_lo, s_hi (2*nq)
    ncat = len(cat_list)
    i_t = ng
    i_lo = 2 * ng
    i_hi = i_lo + ncat
    i_slo = i_hi + ncat
    i_shi = i_slo + nq
    nvar = i_shi + nq

    c = np.zeros(nvar)
    c[i_t : i_t + ng] = EPS_ONE
    for j, cat in enumerate(cat_list):
        c[i_lo + j] = -WEIGHT[cat]
        c[i_hi + j] = WEIGHT[cat]
    w = np.array([VIOL_WEIGHT * WEIGHT[cat] for cat in cats])
    c[i_slo : i_slo + nq] = w
    c[i_shi : i_shi + nq] = w

    rows_A: list[sp.spmatrix] = []
    lb: list[np.ndarray] = []
    ub: list[np.ndarray] = []
    q = np.arange(nq)
    A_sp = sp.csr_matrix(A)
    e_lo = sp.csr_matrix((np.ones(nq), (q, i_lo + cat_idx)), shape=(nq, nvar))
    e_hi = sp.csr_matrix((np.ones(nq), (q, i_hi + cat_idx)), shape=(nq, nvar))
    Ag = sp.hstack([A_sp, sp.csr_matrix((nq, nvar - ng))]).tocsr()
    # lo <= logv + A g  and  logv + A g <= hi
    rows_A.append(Ag - e_lo)
    lb.append(-logv)
    ub.append(np.full(nq, np.inf))
    rows_A.append(Ag - e_hi)
    lb.append(np.full(nq, -np.inf))
    ub.append(-logv)
    # window slacks on scaled quantities only
    if scaled.any():
        qs = q[scaled]
        ns = len(qs)
        e_slo = sp.csr_matrix(
            (np.ones(ns), (np.arange(ns), i_slo + qs)), shape=(ns, nvar)
        )
        e_shi = sp.csr_matrix(
            (np.ones(ns), (np.arange(ns), i_shi + qs)), shape=(ns, nvar)
        )
        wlo = np.log2([WINDOW[cats[i]][0] for i in qs])
        whi = np.log2([WINDOW[cats[i]][1] for i in qs])
        rows_A.append(Ag[qs] + e_slo)
        lb.append(wlo - logv[qs])
        ub.append(np.full(ns, np.inf))
        rows_A.append(Ag[qs] - e_shi)
        lb.append(np.full(ns, -np.inf))
        ub.append(whi - logv[qs])
    # t >= g and t >= -g
    I = sp.identity(ng, format="csr")
    pad = sp.csr_matrix((ng, nvar - 2 * ng))
    rows_A.append(sp.hstack([-I, I, pad]))
    rows_A.append(sp.hstack([I, I, pad]))
    lb += [np.zeros(ng), np.zeros(ng)]
    ub += [np.full(ng, np.inf), np.full(ng, np.inf)]

    vlb = np.concatenate([np.full(ng, -G_MAX), np.zeros(nvar - ng)])
    vub = np.concatenate([np.full(ng, G_MAX), np.full(nvar - ng, np.inf)])
    vlb[i_lo:i_slo] = -np.inf
    for j in range(ncat):
        if not (cat_idx == j).any():  # unused category: pin its spread to 0
            vlb[i_lo + j] = vub[i_lo + j] = vlb[i_hi + j] = vub[i_hi + j] = 0
    if spec.energy is not None:
        vlb[0] = vub[0] = spec.energy
    if spec.cost is not None:
        vlb[1] = vub[1] = spec.cost
    if not spec.rows:
        vlb[2:ng] = vub[2:ng] = 0
    integrality = np.zeros(nvar)
    integrality[:ng] = 1

    res = milp(
        c,
        constraints=LinearConstraint(
            sp.vstack(rows_A).tocsr(), np.concatenate(lb), np.concatenate(ub)
        ),
        integrality=integrality,
        bounds=Bounds(vlb, vub),
    )
    if not res.success:
        msg = f"scaling ILP failed: {res.message}"
        raise RuntimeError(msg)
    g = np.rint(res.x[:ng]).astype(int)
    return ScalingExponents(
        int(g[0]), int(g[1]), {name: int(g[2 + k]) for k, name in enumerate(names)}
    )


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


def _apply_scaling(
    m: Model, rexp: np.ndarray, cexp: np.ndarray, oexp: int, sign: int
) -> None:
    """Scale the model by 2^(sign*exponents); sign=-1 restores bit-exactly.

    Convention: `x_j = 2**cexp_j * x'_j`, row i is multiplied by `2**rexp_i`,
    the objective by `2**-oexp`. On restore the solution maps back with
    `2**cexp`, duals with `2**(rexp + oexp)` and the objective value with
    `2**oexp`.
    """
    ofac = 2.0 ** (-sign * oexp)

    def fac(template: xr.DataArray, exp: np.ndarray) -> xr.DataArray:
        return template.copy(data=np.exp2(sign * exp[template.values]))

    for _, con in m.constraints.items():  # noqa: PERF102
        ds = con.data
        rfac = fac(ds["labels"], rexp)
        ds["coeffs"] = ds["coeffs"] * rfac * fac(ds["vars"], cexp)
        ds["rhs"] = ds["rhs"] * rfac
        if "dual" in ds:
            ds["dual"] = ds["dual"] * fac(ds["labels"], -rexp) * ofac
    for _, var in m.variables.items():  # noqa: PERF102
        ds = var.data
        cfac = fac(ds["labels"], -cexp)
        ds["lower"] = ds["lower"] * cfac
        ds["upper"] = ds["upper"] * cfac
        if "solution" in ds:
            ds["solution"] = ds["solution"] * cfac
    obj = m.objective.expression.data
    obj["coeffs"] = obj["coeffs"] * fac(obj["vars"], cexp) * ofac
    if sign == -1 and m.objective.value is not None:
        m.objective.set_value(m.objective.value * 2.0**oexp)


@contextmanager
def scaled(m: Model, exps: ScalingExponents) -> Iterator[None]:
    """Scale the built model in place around a solve, restoring on exit.

    Energy columns get `2**exps.energy`, cost columns `2**exps.cost`, each
    constraint group its row exponent and the objective `2**-exps.cost`.
    Solution, duals and objective value come back in original units. Call
    `m.constraints.sanitize_zeros()` before and solve with
    `sanitize_zeros=False` so the zero-drop never sees scaled coefficients.
    """
    if exps.energy == 0 and exps.cost == 0 and not any(exps.rows.values()):
        logger.info("scaling: nothing to scale")
        yield
        return
    classes = classify_columns(m)
    col_exp = {"energy": exps.energy, "cost": exps.cost, "none": 0}
    nvar = max((int(v.labels.max()) for _, v in m.variables.items()), default=-1) + 2
    ncon = max((int(c.labels.max()) for _, c in m.constraints.items()), default=-1) + 2
    cexp = np.zeros(nvar, dtype=np.int64)
    rexp = np.zeros(ncon, dtype=np.int64)
    for name, var in m.variables.items():
        labels = var.labels.values.ravel()
        cexp[labels[labels != -1]] = col_exp[classes[name]]
    for name, con in m.constraints.items():
        labels = con.labels.values.ravel()
        rexp[labels[labels != -1]] = exps.rows.get(name, 0)
    rvals = list(exps.rows.values())
    _apply_scaling(m, rexp, cexp, exps.cost, +1)
    logger.info(
        "scaling: energy 2^%d, cost 2^%d, row exponents [%d, %d] over %d groups",
        exps.energy,
        exps.cost,
        min(rvals, default=0),
        max(rvals, default=0),
        len(rvals),
    )
    try:
        yield
    finally:
        _apply_scaling(m, rexp, cexp, exps.cost, -1)


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
    coeffs = _group_ranges(m, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN groups
        for name, con in m.constraints.items():
            live = con.data["labels"] != -1
            absr = abs(con.data["rhs"])
            absr = absr.where(live & np.isfinite(absr) & (absr != 0))
            cr = coeffs.loc[(name, "all")] if (name, "all") in coeffs.index else None
            rows[("constraint", name)] = {
                "coeff_min": float(cr["coeff_min"]) if cr is not None else np.nan,
                "coeff_max": float(cr["coeff_max"]) if cr is not None else np.nan,
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
