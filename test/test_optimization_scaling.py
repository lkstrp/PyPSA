# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for numerical scaling in n.optimize(scaling=...)."""

from unittest.mock import patch

import numpy as np
import pytest

NETWORKS = ["ac_dc_network", "storage_hvdc_network"]


def _solve(n, **kw):
    n.optimize(**kw)
    return n


@pytest.mark.parametrize("network", NETWORKS)
@pytest.mark.parametrize(
    "scaling",
    [
        True,
        {"energy": 100, "cost": 1e3},
        {"energy": 1, "cost": 1, "emissions": 1},
    ],
    ids=["default", "custom", "identity"],
)
def test_scaling_equivalence(request, network, scaling):
    """Scaled and unscaled solves must agree in original units."""
    n = request.getfixturevalue(network)
    ref = _solve(n.copy(), scaling=False)
    got = _solve(n, scaling=scaling)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-5)
    np.testing.assert_allclose(
        got.objective_constant, ref.objective_constant, rtol=1e-5
    )

    for c, attr in [
        ("Generator", "p_nom_opt"),
        ("Link", "p_nom_opt"),
        ("Line", "s_nom_opt"),
    ]:
        a = got.components[c].static.get(attr)
        b = ref.components[c].static.get(attr)
        if a is not None and len(a):
            np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    for c, attr in [
        ("Generator", "p"),
        ("Link", "p"),
        ("Line", "s"),
        ("StorageUnit", "p"),
    ]:
        a = got.components[c].dynamic.get(attr)
        b = ref.components[c].dynamic.get(attr)
        if a is not None and a.shape[1]:
            if c == "StorageUnit":
                # individual dispatch is degenerate across identical units
                np.testing.assert_allclose(
                    a.values.sum(axis=1), b.values.sum(axis=1), rtol=1e-5, atol=1e-4
                )
            else:
                np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(
        got.c.buses.dynamic.marginal_price.values,
        ref.c.buses.dynamic.marginal_price.values,
        rtol=1e-5,
        atol=1e-6,
    )

    # mu_upper dual on a passive branch
    a = got.components["Line"].dynamic.get("mu_upper")
    b = ref.components["Line"].dynamic.get("mu_upper")
    if a is not None and a.shape[1]:
        np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    # global constraint mu
    gc_a, gc_b = got.global_constraints, ref.global_constraints
    if len(gc_a) and "mu" in gc_a:
        np.testing.assert_allclose(
            gc_a["mu"].values, gc_b["mu"].values, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("network", NETWORKS)
def test_scaling_inputs_restored(request, network):
    """Scaled input columns must be byte-identical after the call."""
    n = request.getfixturevalue(network)
    before = n.c.generators.static["capital_cost"].copy()
    before_mc = n.c.generators.dynamic["marginal_cost"].copy()
    n.optimize(scaling=True)
    np.testing.assert_array_equal(
        n.c.generators.static["capital_cost"].values, before.values
    )
    if before_mc.shape[1]:
        np.testing.assert_array_equal(
            n.c.generators.dynamic["marginal_cost"].values, before_mc.values
        )


@pytest.mark.parametrize("network", NETWORKS)
def test_scaling_exception_safety(request, network):
    """A solve failure must still restore the scaled inputs."""
    n = request.getfixturevalue(network)
    before = n.c.generators.static["capital_cost"].copy()
    n.optimize.create_model(scaling=True)
    with (
        patch("linopy.Model.solve", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError),
    ):
        n.optimize.solve_model()
    np.testing.assert_array_equal(
        n.c.generators.static["capital_cost"].values, before.values
    )


def test_scaling_unit_commitment():
    """Commitment costs (1/cost) multiply dimensionless binaries, not a quantity."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(6))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[50, 200, 60, 210, 40, 180])
        n.add("Generator", "base", bus="b", p_nom=100, marginal_cost=20)
        n.add(
            "Generator",
            "peak",
            bus="b",
            p_nom=300,
            marginal_cost=80,
            committable=True,
            p_min_pu=0.3,
            start_up_cost=5000,
            shut_down_cost=2000,
            stand_by_cost=100,
        )
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_array_equal(
        got.components["Generator"].dynamic["status"].values,
        ref.components["Generator"].dynamic["status"].values,
    )


def test_scaling_modular():
    """Modular size columns (*_nom_mod, MW/MWh) must scale by 1/energy."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(4))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[400, 600, 800, 500])
        n.add(
            "Generator",
            "modular_gas",
            bus="b",
            p_nom_extendable=True,
            committable=True,
            p_nom_mod=200,
            p_nom_max=1000,
            p_min_pu=0.3,
            marginal_cost=50,
            capital_cost=50000,
            start_up_cost=100,
            shut_down_cost=50,
        )
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-5)
    np.testing.assert_allclose(
        got.c["Generator"].static["p_nom_opt"].values,
        ref.c["Generator"].static["p_nom_opt"].values,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        got.model.variables["Generator-n_mod"].solution.values,
        ref.model.variables["Generator-n_mod"].solution.values,
    )


def test_scaling_emissions():
    """A primary_energy CO2 cap: objective, shadow price (cost/emissions) and the
    binding-cap dispatch must match the unscaled solve."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(4))
        n.add("Bus", "b")
        n.add("Carrier", "gas", co2_emissions=0.2)
        n.add("Carrier", "clean", co2_emissions=0.0)
        n.add("Load", "l", bus="b", p_set=[100, 120, 90, 110])
        n.add("Generator", "gas", bus="b", carrier="gas", p_nom=200, marginal_cost=20)
        n.add(
            "Generator", "clean", bus="b", carrier="clean", p_nom=200, marginal_cost=80
        )
        # Cap forces some clean dispatch (gas-only would emit ~84 tCO2).
        n.add("GlobalConstraint", "co2", type="primary_energy", constant=40.0)
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_allclose(
        got.global_constraints["mu"].values,
        ref.global_constraints["mu"].values,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        got.c.generators.dynamic.p.values,
        ref.c.generators.dynamic.p.values,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scaling_storage():
    """A storage network (StorageUnit + Store, with >1 snapshot weighting and
    standing losses) exercises max_hours and standing_loss. SoC, store e, bus
    prices and the store energy-balance dual must round-trip."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(6))
        n.snapshot_weightings.loc[:, :] = 3.0  # >1 hour weighting
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[100, 300, 150, 350, 80, 250])
        n.add("Generator", "g", bus="b", p_nom=250, marginal_cost=50)
        n.add("Generator", "peak", bus="b", p_nom=300, marginal_cost=200)
        n.add(
            "StorageUnit",
            "su",
            bus="b",
            p_nom=120,
            max_hours=8,
            marginal_cost=2,
            standing_loss=0.01,  # nonlinear in the (rescaled) stores weighting
            state_of_charge_initial=200,
        )
        n.add("Bus", "e_bus", carrier="energy")
        n.add("Link", "chg", bus0="b", bus1="e_bus", p_nom=150, efficiency=0.9)
        n.add(
            "Store",
            "st",
            bus="e_bus",
            e_nom=2000,
            marginal_cost=1,
            standing_loss=0.02,
            e_initial=1000,
        )
        return n

    ref = _solve(build(), scaling=False, assign_all_duals=True)
    got = _solve(build(), scaling={"energy": 1e3, "cost": 1e6}, assign_all_duals=True)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-5)
    np.testing.assert_allclose(
        got.c.storage_units.dynamic.state_of_charge.values,
        ref.c.storage_units.dynamic.state_of_charge.values,
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        got.c.stores.dynamic.e.values,
        ref.c.stores.dynamic.e.values,
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        got.c.buses.dynamic.marginal_price.values,
        ref.c.buses.dynamic.marginal_price.values,
        rtol=1e-5,
        atol=1e-5,
    )
    # Store energy-balance dual verifies the cost/energy factor. (The StorageUnit
    # SoC-balance dual is skipped: it is degenerate whenever the unit sits at
    # SoC=0, so scaled/unscaled land on equivalent alternate dual vertices.)
    np.testing.assert_allclose(
        got.c.stores.dynamic.mu_energy_balance.values,
        ref.c.stores.dynamic.mu_energy_balance.values,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("network", NETWORKS)
def test_scaling_two_step_matches_one_shot(request, network):
    """`create_model(scaling=...) + solve_model()` must match `optimize(scaling=...)`."""
    one = request.getfixturevalue(network).copy()
    one.optimize(scaling=True, assign_all_duals=True)

    two = request.getfixturevalue(network).copy()
    before = two.c.generators.static["capital_cost"].copy()
    two.optimize.create_model(scaling=True)
    # Inputs restored once the model is built (context exited).
    np.testing.assert_array_equal(
        two.c.generators.static["capital_cost"].values, before.values
    )
    two.optimize.solve_model(assign_all_duals=True)
    # And still restored after solving.
    np.testing.assert_array_equal(
        two.c.generators.static["capital_cost"].values, before.values
    )

    np.testing.assert_allclose(two.objective, one.objective, rtol=1e-5)
    np.testing.assert_allclose(
        two.objective_constant, one.objective_constant, rtol=1e-5
    )

    for c, attr in [
        ("Generator", "p_nom_opt"),  # extendable
        ("Link", "p_nom_opt"),  # fixed-nominal in these fixtures
        ("Line", "s_nom_opt"),
    ]:
        a = two.components[c].static.get(attr)
        b = one.components[c].static.get(attr)
        if a is not None and len(a):
            np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    for c, attr in [("Generator", "p"), ("Link", "p"), ("Line", "s")]:
        a = two.components[c].dynamic.get(attr)
        b = one.components[c].dynamic.get(attr)
        if a is not None and a.shape[1]:
            np.testing.assert_allclose(a.values, b.values, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(
        two.c.buses.dynamic.marginal_price.values,
        one.c.buses.dynamic.marginal_price.values,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scaling_two_step_fixed_nominal():
    """A non-extendable component gets the right p_nom_opt via the two-step path."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[100, 150, 120])
        n.add("Generator", "fixed", bus="b", p_nom=200, marginal_cost=30)
        n.add(
            "Generator",
            "ext",
            bus="b",
            p_nom_extendable=True,
            capital_cost=1e5,
            marginal_cost=10,
        )
        return n

    one = build()
    one.optimize(scaling=True)
    two = build()
    two.optimize.create_model(scaling=True)
    two.optimize.solve_model()

    np.testing.assert_allclose(
        two.c.generators.static["p_nom_opt"].values,
        one.c.generators.static["p_nom_opt"].values,
        rtol=1e-5,
        atol=1e-6,
    )
    # Fixed generator keeps its nominal capacity exactly.
    assert two.c.generators.static.at["fixed", "p_nom_opt"] == 200


def _build_transformer_network(variable=False):
    import pypsa

    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add("Carrier", "AC")
    n.add("Bus", "A", v_nom=1.0, carrier="AC")
    n.add("Bus", "B", v_nom=1.0, carrier="AC")
    n.add("Generator", "gen_A", bus="A", p_nom=100, marginal_cost=10.0, carrier="AC")
    n.add("Load", "load_B", bus="B", p_set=[50.0, 50.0])
    n.add("Line", "L1", bus0="A", bus1="B", x=0.01, r=1e-6, s_nom=100, carrier="AC")
    bounds = {"phase_shift_min": -20.0, "phase_shift_max": 20.0} if variable else {}
    n.add(
        "Transformer",
        "T1",
        bus0="A",
        bus1="B",
        x=1.0,  # x_pu = x / s_nom = 0.01
        r=1e-6,
        s_nom=100,
        phase_shift=10.0,
        **bounds,
    )
    return n


@pytest.mark.parametrize("variable", [False, True], ids=["fixed", "variable"])
def test_scaling_transformer_phase_shift(variable):
    """Transformer KVL terms and phase_shift readback must survive scaling.

    x_pu is derived from s_nom inside the scaled context and the phase-shift
    angle term is a raw constant, so both need explicit scaling handling."""
    ref = _solve(_build_transformer_network(variable), scaling=False)
    got = _solve(_build_transformer_network(variable), scaling=True)

    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    for c in ("lines", "transformers"):
        np.testing.assert_allclose(
            got.c[c].dynamic["p0"].values,
            ref.c[c].dynamic["p0"].values,
            rtol=1e-5,
            atol=1e-6,
        )
    if variable:
        np.testing.assert_allclose(
            got.c.transformers.dynamic["phase_shift_opt"].values,
            ref.c.transformers.dynamic["phase_shift_opt"].values,
            rtol=1e-5,
            atol=1e-6,
        )
    # Persisted per-unit impedances must be the true values, not scaled ones.
    np.testing.assert_allclose(
        got.c.transformers.static["x_pu"].values,
        ref.c.transformers.static["x_pu"].values,
        rtol=1e-12,
    )


def test_scaling_overnight_cost():
    """overnight_cost and fom_cost enter the objective and must scale like
    capital_cost."""
    import pypsa

    def build():
        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", "b")
        n.add("Load", "l", bus="b", p_set=[100, 150, 120])
        n.add("Generator", "fixed", bus="b", p_nom=100, marginal_cost=30)
        n.add(
            "Generator",
            "ext",
            bus="b",
            p_nom_extendable=True,
            overnight_cost=1e6,
            discount_rate=0.07,
            lifetime=25,
            fom_cost=2e4,
            marginal_cost=10,
        )
        return n

    ref = _solve(build(), scaling=False)
    got = _solve(build(), scaling=True)
    np.testing.assert_allclose(got.objective, ref.objective, rtol=1e-6)
    np.testing.assert_allclose(
        got.c.generators.static["p_nom_opt"].values,
        ref.c.generators.static["p_nom_opt"].values,
        rtol=1e-5,
        atol=1e-6,
    )


def _exps_for(m, energy=9, cost=17):
    from pypsa.optimization.scaling import ScalingExponents

    return ScalingExponents(
        energy, cost, {name: (i % 5) - 2 for i, name in enumerate(m.constraints)}
    )


def _model_snapshot(m):
    return {
        "coeffs": {k: c.data["coeffs"].copy() for k, c in m.constraints.items()},
        "rhs": {k: c.data["rhs"].copy() for k, c in m.constraints.items()},
        "lower": {k: v.data["lower"].copy() for k, v in m.variables.items()},
        "upper": {k: v.data["upper"].copy() for k, v in m.variables.items()},
        "objective": m.objective.expression.data["coeffs"].copy(),
    }


def _assert_snapshot_equal(m, snap):
    for k, c in m.constraints.items():
        assert c.data["coeffs"].equals(snap["coeffs"][k])
        assert c.data["rhs"].equals(snap["rhs"][k])
    for k, v in m.variables.items():
        assert v.data["lower"].equals(snap["lower"][k])
        assert v.data["upper"].equals(snap["upper"][k])
    assert m.objective.expression.data["coeffs"].equals(snap["objective"])


def test_scaled_context_applier_restores_bit_exact(ac_dc_network):
    from pypsa.optimization.scaling import scaled

    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    m = n.model
    snap = _model_snapshot(m)
    with scaled(m, _exps_for(m)):
        assert not m.objective.expression.data["coeffs"].equals(snap["objective"])
        assert any(
            not c.data["coeffs"].equals(snap["coeffs"][k])
            for k, c in m.constraints.items()
        )
    _assert_snapshot_equal(m, snap)


def test_scaled_context_solve_matches_plain(ac_dc_network):
    from pypsa.optimization.scaling import scaled

    n = ac_dc_network
    # a zero p_nom_opt makes the bound duals degenerate, floor it
    n.generators["p_nom_min"] = 10.0
    ref = n.copy()
    ref.optimize.create_model(include_objective_constant=False)
    ref.model.solve()

    n.optimize.create_model(include_objective_constant=False)
    m = n.model
    m.constraints.sanitize_zeros()
    with scaled(m, _exps_for(m)):
        m.solve(sanitize_zeros=False)

    np.testing.assert_allclose(m.objective.value, ref.model.objective.value, rtol=1e-6)
    for name, var in m.variables.items():
        np.testing.assert_allclose(
            var.solution.values,
            ref.model.variables[name].solution.values,
            rtol=1e-6,
            atol=1e-6,
        )
    for name, con in m.constraints.items():
        np.testing.assert_allclose(
            con.dual.values,
            ref.model.constraints[name].dual.values,
            rtol=1e-6,
            atol=1e-6,
        )


def test_scaled_context_zero_exponents_untouched(ac_dc_network):
    from pypsa.optimization.scaling import ScalingExponents, scaled

    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    m = n.model
    before = {k: c.data["coeffs"].values for k, c in m.constraints.items()}
    with scaled(m, ScalingExponents(0, 0, dict.fromkeys(m.constraints, 0))):
        pass
    for k, c in m.constraints.items():
        assert np.shares_memory(c.data["coeffs"].values, before[k])


def test_scaling_resolver_errors(ac_dc_network):
    n = ac_dc_network
    with pytest.raises(TypeError):  # not a bool/dict
        n.optimize(scaling="big")
    with pytest.raises(TypeError):  # non-numeric value
        n.optimize(scaling={"energy": "big"})
    with pytest.raises(TypeError):  # non-bool equilibration flag
        n.optimize(scaling={"rows": 1024})
    for bad in ({"power": 100}, {"money": 1e6}, {"enrgy": 100}):  # old/typo keys
        with pytest.raises(ValueError):
            n.optimize(scaling=bad)


def test_scaling_mga(ac_dc_network):
    """MGA with unequal energy/cost factors matches the unscaled MGA optimum."""
    ref = ac_dc_network.copy()
    ref.optimize()
    ref.optimize.optimize_mga(slack=0.05)

    got = ac_dc_network.copy()
    scaling = {"energy": 100, "cost": 1e6}
    got.optimize(scaling=scaling)
    got.optimize.optimize_mga(slack=0.05, model_kwargs={"scaling": scaling})

    # The MGA objective (total generator capacity) is unique even when the
    # individual capacities are degenerate.
    np.testing.assert_allclose(
        got.c.generators.static["p_nom_opt"].sum(),
        ref.c.generators.static["p_nom_opt"].sum(),
        rtol=1e-5,
    )


def test_scaling_report(ac_dc_network):
    n = ac_dc_network
    with pytest.raises(ValueError, match="no model"):
        n.optimize.scaling_report()
    n.optimize.create_model()
    rep = n.optimize.scaling_report()
    cols = {"coeff_min", "coeff_max", "rhs_min", "rhs_max", "bound_min", "bound_max"}
    assert cols <= set(rep.columns)
    assert "objective" in rep.index.get_level_values("kind")
    con = rep.xs("constraint").dropna(subset=["coeff_min"])
    assert (con["coeff_min"] > 0).all()
    assert (con["coeff_max"] >= con["coeff_min"]).all()


# --- resolver, column classifier, exponent chooser ----------------------------


def test_resolve_scaling():
    from pypsa.optimization.scaling import ScalingSpec, resolve_scaling

    assert resolve_scaling(False) is None
    assert resolve_scaling(None) is None
    assert resolve_scaling(True) == ScalingSpec(None, None, True)
    assert resolve_scaling({"energy": 1000}) == ScalingSpec(10, None, True)
    assert resolve_scaling({"rows": False}) == ScalingSpec(None, None, False)
    assert resolve_scaling({"cost": 65536.0}).cost == 16
    for bad in ("big", {"energy": "x"}, {"rows": 1}):
        with pytest.raises(TypeError):
            resolve_scaling(bad)
    for bad in ({"emissions": 1}, {"columns": True}, {"energy": 0}):
        with pytest.raises(ValueError):
            resolve_scaling(bad)
    with pytest.raises(ValueError, match="energy"):
        resolve_scaling({"power": 2})


def _build_uc_modular_network():
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=[400, 600, 800, 500])
    n.add(
        "Generator",
        "modular_gas",
        bus="b",
        p_nom_extendable=True,
        committable=True,
        p_nom_mod=200,
        p_nom_max=1000,
        p_min_pu=0.3,
        marginal_cost=50,
        capital_cost=50000,
        start_up_cost=100,
        shut_down_cost=50,
    )
    return n


def test_classify_columns():
    from pypsa.optimization.scaling import classify_columns

    n = _build_uc_modular_network()
    n.optimize.create_model()
    classes = classify_columns(n.model)
    assert classes["Generator-status"] == "none"
    assert classes["Generator-n_mod"] == "none"
    assert classes["Generator-p"] == "energy"
    assert classes["Generator-p_nom"] == "energy"


def test_classify_columns_cvar(stochastic_network):
    from pypsa.optimization.scaling import classify_columns

    n = stochastic_network
    n.set_risk_preference(alpha=0.2, omega=0.5)
    n.optimize.create_model()
    classes = classify_columns(n.model)
    assert classes["CVaR-a"] == "cost"
    assert classes["CVaR"] == "cost"


def _window_violation(m, exps):
    """Sum of log2 window violations of every ILP quantity under `exps`."""
    from pypsa.optimization.scaling import WINDOW, _quantities

    logv, A, cats = _quantities(m)
    g = np.array([exps.energy, exps.cost, *exps.rows.values()], dtype=float)
    scaled = logv + A @ g
    lo = np.log2([WINDOW[c][0] for c in cats])
    hi = np.log2([WINDOW[c][1] for c in cats])
    return float(np.maximum(lo - scaled, 0).sum() + np.maximum(scaled - hi, 0).sum())


def test_choose_exponents(ac_dc_network):
    from pypsa.optimization.scaling import (
        ScalingExponents,
        ScalingSpec,
        choose_exponents,
    )

    n = ac_dc_network
    n.optimize.create_model(include_objective_constant=False)
    m = n.model

    pinned = choose_exponents(m, ScalingSpec(9, 17, False))
    assert pinned.energy == 9
    assert pinned.cost == 17
    assert set(pinned.rows) == set(m.constraints)
    assert all(v == 0 for v in pinned.rows.values())

    auto = choose_exponents(m, ScalingSpec(None, None, True))
    assert auto == choose_exponents(m, ScalingSpec(None, None, True))
    assert isinstance(auto, ScalingExponents)
    assert all(
        isinstance(v, int) for v in (auto.energy, auto.cost, *auto.rows.values())
    )
    zero = ScalingExponents(0, 0, dict.fromkeys(m.constraints, 0))
    viol_auto, viol_zero = _window_violation(m, auto), _window_violation(m, zero)
    assert viol_auto == 0 or viol_auto < viol_zero


def test_choose_exponents_indicator(caplog):
    import pypsa
    from pypsa.optimization.scaling import ScalingSpec, choose_exponents

    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=[50, 80])
    n.add("Generator", "g", bus="b", p_nom=100, committable=True, marginal_cost=10)
    n.optimize.create_model()
    m = n.model
    status = m.variables["Generator-status"]
    p = m.variables["Generator-p"]
    m.add_indicator_constraints(status, 1, 1 * p, ">=", 0, name="ind")
    with caplog.at_level("WARNING"):
        exps = choose_exponents(m, ScalingSpec(None, None, True))
    assert exps.energy == 0
    assert exps.cost == 0
    assert all(v == 0 for v in exps.rows.values())
    assert "indicator" in caplog.text
