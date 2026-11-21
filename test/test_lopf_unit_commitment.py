# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa
from pypsa._linopy_compat import linopy_uses_v1, v1_semantics


def test_unit_commitment():
    """
    This test is based on https://docs.pypsa.org/latest/examples/unit-commitment/
    and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    expected_status = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array([[4000, 6000, 5000, 0], [0, 0, 0, 800]], dtype=float).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_minimum_up_time():
    """
    This test is based on https://docs.pypsa.org/latest/examples/unit-commitment/
    and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=0,
        min_up_time=3,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 800, 5000, 3000])

    n.optimize()

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


@pytest.mark.parametrize("offset", [0, 2], ids=["full", "subset"])
def test_minimum_up_time_up_time_before(offset):
    """
    This test is based on https://docs.pypsa.org/latest/examples/unit-commitment/
    and is not very comprehensive.
    """
    n = pypsa.Network()

    n.set_snapshots(range(-offset, 4))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        up_time_before=1,
        min_up_time=4,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[0] * offset + [4000, 800, 5000, 3000])

    n.optimize(snapshots=range(4))

    expected_status = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.iloc[offset:].values, expected_status)

    expected_dispatch = np.array(
        [[3900, 0, 4900, 3000], [100, 800, 100, 0]], dtype=float
    ).T

    equal(n.c.generators.dynamic.p.iloc[offset:].values, expected_dispatch)


def test_minimum_down_time():
    """
    This test is based on https://docs.pypsa.org/latest/examples/unit-commitment/
    and is not very comprehensive.
    """
    n = pypsa.Network()

    n.set_snapshots(range(4))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=2,
        down_time_before=1,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    n.add("Load", "load", bus="bus", p_set=[3000, 800, 3000, 8000])

    n.optimize()

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


@pytest.mark.parametrize("offset", [0, 2], ids=["full", "subset"])
def test_minimum_down_time_up_time_before(offset):
    """
    This test is based on https://docs.pypsa.org/latest/examples/unit-commitment/
    and is not very comprehensive.
    """
    n = pypsa.Network()

    n.set_snapshots(range(-offset, 4))

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        min_down_time=3,
        down_time_before=2,
        up_time_before=0,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.1,
        p_nom=4000,
    )

    n.add("Load", "load", bus="bus", p_set=[0] * offset + [3000, 800, 3000, 8000])

    n.optimize(snapshots=range(4))

    expected_status = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.status.iloc[offset:].values, expected_status)

    expected_dispatch = np.array([[0, 0, 3000, 8000], [3000, 800, 0, 0]], dtype=float).T

    equal(n.c.generators.dynamic.p.iloc[offset:].values, expected_dispatch)


def test_start_up_costs():
    n = pypsa.Network()

    n.snapshots = range(4)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        up_time_before=0,
        marginal_cost=20,
        start_up_cost=3000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.0,
        up_time_before=1,
        start_up_cost=1000,
        p_nom=10000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    assert n.objective == 359000


def test_shut_down_costs():
    n = pypsa.Network()

    n.snapshots = range(4)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=20,
        shut_down_cost=1000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        shut_down_cost=1000,
        p_min_pu=0.1,
        p_nom=1000,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 5000, 800])

    n.optimize()

    assert n.objective == 358000


@pytest.mark.parametrize("cost_attr", ["start_up_cost", "shut_down_cost"])
def test_start_up_and_shut_down_costs_multi_investment_period(cost_attr):
    """Start-up and shut-down costs follow the investment period objective weighting."""
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.investment_period_weightings["objective"] = [1, 10]

    n.add("Bus", "bus")
    n.add(
        "Generator",
        "committable",
        bus="bus",
        committable=True,
        p_nom=100,
        p_min_pu=0.5,
        up_time_before=0,
        marginal_cost=1,
        **{cost_attr: 1000},
    )
    n.add("Generator", "backup", bus="bus", p_nom=1000, marginal_cost=100)
    n.add(
        "Load",
        "load",
        bus="bus",
        p_set=pd.Series([60.0, 0.0, 60.0, 0.0], index=n.snapshots),
    )

    status, _ = n.optimize(solver_name="highs", multi_investment_periods=True)
    assert status == "ok"

    # Per period the committable generator runs for one snapshot at 60 MW
    # (marginal cost 60) and changes state once (cost 1000). Periods are
    # weighted 1 and 10: (60 + 1000) * (1 + 10) = 11660.
    expected_status = pd.Series(
        [1.0, 0.0, 1.0, 0.0], index=n.snapshots, name="committable"
    )
    pd.testing.assert_series_equal(
        n.c.generators.dynamic.status["committable"],
        expected_status,
        check_dtype=False,
        check_index_type=False,
    )
    assert n.objective == pytest.approx(11660)


def test_unit_commitment_rolling_horizon():
    n = pypsa.Network()
    n.snapshots = range(7)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "coal",
        bus="bus",
        committable=True,
        p_min_pu=0.1,
        up_time_before=1,
        marginal_cost=20,
        min_up_time=2,
        min_down_time=2,
        start_up_cost=10000,
        p_nom=10000,
    )

    n.add(
        "Generator",
        "gas",
        bus="bus",
        committable=True,
        marginal_cost=70,
        p_min_pu=0.01,
        start_up_cost=100,
        up_time_before=0,
        p_nom=10000,
    )
    n.add("Load", "load", bus="bus", p_set=[4000, 6000, 800, 5000, 3000, 950, 800])

    n.optimize(snapshots=[0, 1, 2])
    n.optimize(snapshots=[2, 3, 4])
    n.optimize(snapshots=[4, 5, 6])

    expected_status = np.array(
        [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1]], dtype=float
    ).T
    equal(n.c.generators.dynamic.status.values, expected_status)

    expected_dispatch = np.array(
        [[4000, 6000, 0, 0, 0, 0, 0], [0, 0, 800, 5000, 3000, 950, 800]]
    ).T

    equal(n.c.generators.dynamic.p.values, expected_dispatch)


def test_linearized_unit_commitment():
    n = pypsa.Network()
    n.snapshots = pd.date_range("2022-01-01", "2022-02-09", freq="D")

    load = np.zeros(len(n.snapshots))
    load[:5] = 5
    load[5:10] = 6
    load[10:15] = 8
    load[15:20] = 10
    load[20:30] = 7
    load[30:40] = 6
    load *= 100

    n.add("Bus", "bus")

    for seed, i in enumerate(range(40), start=1):
        rng = np.random.default_rng(seed)  # Create a random number generator
        p_min_pu = rng.integers(1, 5) / 10
        marginal_cost = rng.integers(1, 11) * 10
        min_up_time = rng.integers(0, 6)
        min_down_time = rng.integers(0, 6)
        p_nom = rng.integers(1, 10) * 5
        start_up_cost = rng.integers(1, 5) * 100

        # the constraint tightening proposed in Baldick et al. depends on start_up_cost
        # and shut_down_cost being equal therefore, we force them to be equal for first
        # 20 generators
        shut_down_cost = rng.integers(1, 5) * 100 if i >= 20 else start_up_cost

        n.add(
            "Generator",
            f"{i}",
            bus="bus",
            committable=True,
            up_time_before=0,
            p_min_pu=p_min_pu,
            marginal_cost=marginal_cost,
            min_up_time=min_up_time,
            min_down_time=min_down_time,
            p_nom=p_nom,
            start_up_cost=start_up_cost,
            shut_down_cost=shut_down_cost,
        )
    n.add("Load", "load", bus="bus", p_set=load)

    n.optimize(linearized_unit_commitment=True)

    MILP_objective = 1241100
    assert round(n.objective / MILP_objective, 2) == 1


def test_linearized_unit_commitment_committable_extendable():
    # linearized relaxation must not exclude the integer optimum for a
    # committable extendable generator
    n = pypsa.Network()
    n.set_snapshots([0, 1, 2, 3])
    n.add("Bus", "hub")
    n.add(
        "Generator",
        "flex",
        bus="hub",
        committable=True,
        p_nom_extendable=True,
        p_min_pu=0.4,
        p_nom_max=100.0,
        marginal_cost=40.0,
        capital_cost=100.0,
    )
    n.add(
        "Generator",
        "peak",
        bus="hub",
        p_nom_extendable=True,
        p_nom_max=100.0,
        marginal_cost=120.0,
        capital_cost=50.0,
    )
    n.add("Load", "demand", bus="hub", p_set=[75, 30, 75, 0])

    n.optimize()
    integer_objective = n.objective

    n.optimize(linearized_unit_commitment=True)
    assert n.objective == pytest.approx(integer_objective)


def test_link_unit_commitment():
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        up_time_before=0,
        min_up_time=3,
        start_up_cost=3333,
        p_nom=12000,
    )

    n.add(
        "Generator",
        "wind",
        bus="electricity",
        p_nom=800,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 6000, 800, 5000])

    n.optimize()

    expected_status = [1.0, 1.0, 1.0, 1.0]

    equal(n.c.links.dynamic.status["OCGT"].values, expected_status)

    expected_dispatch = [3200.0, 5200.0, 600.0, 4200.0]

    equal(-n.c.links.dynamic.p1["OCGT"].values, expected_dispatch)

    assert round(n.objective, 1) == 267333.0


def test_link_ramp_limits():
    """
    Test that ramp limits work for Links.
    """
    n = pypsa.Network()

    snapshots = range(6)
    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        p_min_pu=0.1,
        efficiency=0.5,
        p_nom=10000,
        ramp_limit_up=0.3,  # 30% of p_nom per timestep = 3000 MW
        ramp_limit_down=0.4,  # 40% of p_nom per timestep = 4000 MW
        marginal_cost=20,
    )

    n.add("Generator", "backup", bus="electricity", marginal_cost=100, p_nom=10000)

    # Varying load to induce ramping
    n.add("Load", "load", bus="electricity", p_set=[2000, 7000, 1500, 5500, 5000, 2500])

    n.optimize()

    # Check that ramp limits are respected
    # For Links, use p0 (the power at bus0) which is the optimization variable
    p_diff = n.c.links.dynamic.p0["OCGT"].diff()
    max_ramp_up = 0.3 * 10000  # 3000 MW
    max_ramp_down = 0.4 * 10000  # 4000 MW

    # Check ramp up (positive changes)
    ramp_ups = p_diff[p_diff > 0]
    if not ramp_ups.empty:
        assert ramp_ups.max() <= max_ramp_up + 1e-4, (
            f"Ramp up limit violated: {ramp_ups.max()} > {max_ramp_up}"
        )

    # Check ramp down (negative changes)
    ramp_downs = p_diff[p_diff < 0].abs()
    if not ramp_downs.empty:
        assert ramp_downs.max() <= max_ramp_down + 1e-4, (
            f"Ramp down limit violated: {ramp_downs.max()} > {max_ramp_down}"
        )


def test_link_ramp_limits_rolling_horizon():
    """
    Test that ramp limits work for Links in rolling horizon optimization.
    This specifically tests the historical data retrieval for p0 in Links
    when sns[0] != n.snapshots[0].
    """
    n = pypsa.Network()

    snapshots = range(12)
    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        p_min_pu=0.1,
        efficiency=0.5,
        p_nom=10000,
        ramp_limit_up=0.3,  # 30% of p_nom per timestep = 3000 MW
        ramp_limit_down=0.4,  # 40% of p_nom per timestep = 4000 MW
        marginal_cost=20,
    )

    n.add("Generator", "backup", bus="electricity", marginal_cost=100, p_nom=10000)

    # Varying load to induce ramping with jumps > ramp limits (3000 up, 4000 down)
    n.add(
        "Load",
        "load",
        bus="electricity",
        p_set=[2000, 6000, 1000, 5500, 9000, 3000, 7500, 2500, 7000, 1500, 6000, 2000],
    )

    n.optimize.optimize_with_rolling_horizon(horizon=4, overlap=1)

    # Check that ramp limits are respected across all snapshots
    # For Links, use p0 (the power at bus0)
    p_diff = n.c.links.dynamic.p0["OCGT"].diff()
    max_ramp_up = 0.3 * 10000  # 3000 MW
    max_ramp_down = 0.4 * 10000  # 4000 MW

    # Check ramp up (positive changes)
    ramp_ups = p_diff[p_diff > 0]
    if not ramp_ups.empty:
        assert ramp_ups.max() <= max_ramp_up + 1e-4, (
            f"Ramp up limit violated: {ramp_ups.max()} > {max_ramp_up}"
        )

    # Check ramp down (negative changes)
    ramp_downs = p_diff[p_diff < 0].abs()
    if not ramp_downs.empty:
        assert ramp_downs.max() <= max_ramp_down + 1e-4, (
            f"Ramp down limit violated: {ramp_downs.max()} > {max_ramp_down}"
        )


def test_dynamic_ramp_rates():
    """
    This test checks that dynamic ramp rates are correctly applied when
    considering a unit outage represented by p_max_pu.
    """
    n = pypsa.Network()

    snapshots = range(15)
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)

    # vary marginal price of gen1 to induce ramping
    gen1_marginal = pd.Series(100, index=n.snapshots)
    gen1_marginal[[4, 5, 6, 10, 11, 12]] = 200

    static_ramp_up = 0.8
    static_ramp_down = 1
    p_max_pu = pd.Series(1, index=n.snapshots).astype(float)
    p_max_pu.loc[n.snapshots[0:6]] = 0.5  # 50% capacity outage for 6 periods

    n.add(
        "Generator",
        "gen1",
        bus="bus",
        p_nom=100,
        p_max_pu=p_max_pu,
        ramp_limit_up=static_ramp_up * p_max_pu,
        ramp_limit_down=static_ramp_down * p_max_pu,
        marginal_cost=gen1_marginal,
    )

    n.add("Generator", "gen2", bus="bus", p_nom=100, marginal_cost=150)

    n.optimize()

    assert (n.c.generators.dynamic.p.diff().loc[0:6, "gen1"]).max() <= 0.5 * 80
    assert (n.c.generators.dynamic.p.diff().loc[0:6, "gen1"]).min() >= -0.5 * 100
    assert (n.c.generators.dynamic.p.diff().loc[6:, "gen1"]).max() <= 80
    assert (n.c.generators.dynamic.p.diff().loc[6:, "gen1"]).min() >= -100


@pytest.mark.parametrize("direction", ["up", "down"])
def test_generator_ramp_constraints_unsorted_names(direction):
    """
    See https://github.com/PyPSA/PyPSA/issues/1675

    With one fixed and one extendable generator on the same component,
    a non-alphabetical insertion order used to leak the extendable's
    p_nom variable into the fixed generator's ramp constraint.
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=3, freq="h"))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)

    # Fixed generator's name sorts AFTER the extendable's; this ordering
    # is what triggered the bug.
    n.add(
        "Generator",
        "fixed_lignite",
        bus="bus",
        p_nom=944,
        marginal_cost=30,
        **{f"ramp_limit_{direction}": 0.04},
    )
    n.add(
        "Generator",
        "ext_gas",
        bus="bus",
        p_nom_extendable=True,
        marginal_cost=95,
        capital_cost=52_000,
    )

    n.optimize.create_model()

    con = n.model.constraints[f"Generator-p-ramp_limit_{direction}"]
    p_nom_ext_label = int(
        n.model.variables["Generator-p_nom"].labels.sel(name="ext_gas")
    )
    vars_for_fixed = con.vars.sel(name="fixed_lignite").values
    assert (vars_for_fixed != p_nom_ext_label).all(), (
        f"ramp_{direction} for 'fixed_lignite' references the extendable's "
        f"p_nom variable (issue #1675)"
    )


@pytest.mark.parametrize("direction", ["up", "down"])
def test_generator_ramp_constraints_mask_nan(direction):
    """
    See https://github.com/PyPSA/PyPSA/issues/1493
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=2, freq="h"))

    n.add("Bus", "bus")
    # Add load with sudden change to trigger ramping up/down
    p_set = [0.0, 50.0] if direction == "up" else [50.0, 10.0]
    n.add(
        "Load",
        "load",
        bus="bus",
        p_set=pd.Series(p_set, index=n.snapshots),
    )

    ramp_kw = {f"ramp_limit_{direction}": 0.5}

    # Generator with ramp limits
    n.add(
        "Generator",
        "gen_limited",
        bus="bus",
        p_nom=10,
        marginal_cost=1,
        **ramp_kw,
    )

    # Generator without ramp limits
    n.add(
        "Generator",
        "gen_unlimited",
        bus="bus",
        p_nom=1000,
        marginal_cost=10,
    )

    n.optimize(solver_name="highs")

    # Check labels to see which generators have active ramp constraints applied
    # Generator with undefined ramp limits should have label -1 (no constraint)
    # Limited generators should not have label -1 except for the very first snapshot
    key = f"Generator-p-ramp_limit_{direction}"
    constraints = n.model.constraints[key]
    labels_gen_limited = constraints.data["labels"].sel(name="gen_limited").to_pandas()
    assert (labels_gen_limited.loc[n.snapshots[1:]] != -1).all(), (
        f"ramp_{direction} constraint should be active for 'gen_limited'."
    )

    labels_gen_unlimited = (
        constraints.data["labels"].sel(name="gen_unlimited").to_pandas()
    )
    assert (labels_gen_unlimited == -1).all(), (
        f"ramp_{direction} constraint should be masked for 'gen_unlimited'."
    )


@pytest.mark.parametrize("direction", ["up", "down"])
def test_link_ramp_constraints_mask_nan(direction):
    """
    See https://github.com/PyPSA/PyPSA/issues/1493
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=2, freq="h"))

    n.add("Bus", "bus0")
    n.add("Bus", "bus1")

    # Add load with sudden change to trigger ramping up/down
    p_set = [0.0, 50.0] if direction == "up" else [50.0, 10.0]
    n.add(
        "Load",
        "load",
        bus="bus1",
        p_set=pd.Series(p_set, index=n.snapshots),
    )

    ramp_kw = {
        f"ramp_limit_{direction}": 0.5,
    }

    # Link with ramp limits
    n.add(
        "Link",
        "link_limited",
        bus0="bus0",
        bus1="bus1",
        p_nom=10,
        efficiency=1.0,
        marginal_cost=1,
        **ramp_kw,
    )

    # Link without ramp limits
    n.add(
        "Link",
        "link_unlimited",
        bus0="bus0",
        bus1="bus1",
        p_nom=1000,
        efficiency=1.0,
        marginal_cost=10,
    )

    n.add(
        "Generator",
        "generator",
        bus="bus0",
        p_nom=1000,
        marginal_cost=0.0,
    )

    n.optimize(solver_name="highs")

    # Check labels to see which links have active ramp constraints applied
    # Link with undefined ramp limits should have label -1 (no constraint)
    key = f"Link-p-ramp_limit_{direction}"
    constraints = n.model.constraints[key]
    labels_link_limited = (
        constraints.data["labels"].sel(name="link_limited").to_pandas()
    )
    assert (labels_link_limited.loc[n.snapshots[1:]] != -1).all(), (
        f"ramp_{direction} constraint should be active for 'link_limited'."
    )

    labels_link_unlimited = (
        constraints.data["labels"].sel(name="link_unlimited").to_pandas()
    )
    assert (labels_link_unlimited == -1).all(), (
        f"ramp_{direction} constraint should be masked for 'link_unlimited'."
    )


def test_dynamic_start_up_rates_for_commitables():
    """
    This test checks that start up ramp rate constraints within unit commitment functionality runs through and is considered correctly.
    """
    n = pypsa.Network()

    snapshots = range(15)
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)

    # vary marginal price of gen1 to induce ramping
    gen1_marginal = pd.Series(100, index=n.snapshots)
    gen1_marginal[[4, 5, 6, 10, 11, 12]] = 200

    n.add(
        "Generator",
        "gen1",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.3,
        p_max_pu=1,
        ramp_limit_up=1,
        ramp_limit_start_up=0.3,
        ramp_limit_shut_down=1,
        start_up_cost=10,
        shut_down_cost=10,
        marginal_cost=gen1_marginal,
    )

    n.add("Generator", "gen2", bus="bus", p_nom=100, marginal_cost=150)

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    # Check that ramp_limit_start_up constraint is respected
    gen1_status = n.c.generators.dynamic.status["gen1"]
    gen1_p = n.c.generators.dynamic.p["gen1"]

    # Find startup events (status changes from 0 to 1)
    startup_snapshots = gen1_status[
        (gen1_status == 1) & (gen1_status.shift(1) == 0)
    ].index

    for snapshot in startup_snapshots:
        expected_max_startup = 0.3 * 100  # ramp_limit_start_up * p_nom
        assert gen1_p[snapshot] <= expected_max_startup, (
            f"Startup ramp limit violated at snapshot {snapshot}: {gen1_p[snapshot]} > {expected_max_startup}"
        )


def test_ramp_limits_multi_investment_period():
    n = pypsa.Network()
    sns = pd.date_range("2025-01-01", periods=5, freq="h").append(
        pd.date_range("2026-01-01", periods=5, freq="h")
    )
    n.set_snapshots(pd.MultiIndex.from_arrays([sns.year, sns]))
    n.investment_periods = [2025, 2026]
    n.investment_period_weightings["years"] = 5

    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        ramp_limit_up=0.5,
        marginal_cost=10,
    )
    n.add("Load", "load", bus="bus", p_set=100)

    status, _ = n.optimize()
    assert status == "ok"


def test_committable_start_up_only_ramp_limit():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus")

    load_profile = [0, 50, 100, 100, 50, 0]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    n.add(
        "Generator",
        "gen_commit",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.1,
        ramp_limit_start_up=0.4,
        marginal_cost=10,
    )
    n.add("Generator", "gen_backup", bus="bus", p_nom=200, marginal_cost=100)

    status, _ = n.optimize()
    assert status == "ok"

    gen_status = n.c.generators.dynamic.status["gen_commit"]
    gen_p = n.c.generators.dynamic.p["gen_commit"]
    startup_snapshots = gen_status[(gen_status == 1) & (gen_status.shift(1) == 0)].index

    for snapshot in startup_snapshots:
        expected_max = 0.4 * 100
        assert gen_p[snapshot] <= expected_max + 1e-5


def test_committable_shut_down_only_ramp_limit():
    n = pypsa.Network()
    n.set_snapshots(range(6))
    n.add("Bus", "bus")

    load_profile = [100, 100, 50, 20, 10, 0]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    n.add(
        "Generator",
        "gen_commit",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.1,
        up_time_before=1,
        ramp_limit_shut_down=0.3,
        marginal_cost=10,
    )
    n.add("Generator", "gen_backup", bus="bus", p_nom=200, marginal_cost=100)

    status, _ = n.optimize()
    assert status == "ok"

    gen_status = n.c.generators.dynamic.status["gen_commit"]
    gen_p = n.c.generators.dynamic.p["gen_commit"]
    shutdown_snapshots = gen_status[
        (gen_status == 0) & (gen_status.shift(1) == 1)
    ].index

    for snapshot in shutdown_snapshots:
        prev_snapshot = snapshot - 1
        if prev_snapshot >= 0:
            expected_max_prev = 0.3 * 100
            assert gen_p[prev_snapshot] <= expected_max_prev + 1e-5


def test_committable_ramp_limits_multi_investment_period():
    n = pypsa.Network()
    sns = pd.date_range("2025-01-01", periods=5, freq="h").append(
        pd.date_range("2026-01-01", periods=5, freq="h")
    )
    n.set_snapshots(pd.MultiIndex.from_arrays([sns.year, sns]))
    n.investment_periods = [2025, 2026]
    n.investment_period_weightings["years"] = 5

    n.add("Bus", "bus")

    load_profile = [50, 80, 100, 80, 50, 50, 80, 100, 80, 50]
    n.add("Load", "load", bus="bus", p_set=load_profile)

    n.add(
        "Generator",
        "gen_commit",
        bus="bus",
        p_nom=100,
        committable=True,
        p_min_pu=0.3,
        ramp_limit_up=0.5,
        ramp_limit_start_up=0.4,
        ramp_limit_down=0.5,
        ramp_limit_shut_down=0.3,
        marginal_cost=10,
    )
    n.add("Generator", "gen_backup", bus="bus", p_nom=200, marginal_cost=100)

    status, _ = n.optimize()
    assert status == "ok"

    gen_p = n.c.generators.dynamic.p["gen_commit"]
    gen_status = n.c.generators.dynamic.status["gen_commit"]

    for i in range(1, len(n.snapshots)):
        curr_sns = n.snapshots[i]
        prev_sns = n.snapshots[i - 1]
        curr_status = gen_status[curr_sns]
        prev_status = gen_status[prev_sns]
        curr_p = gen_p[curr_sns]
        prev_p = gen_p[prev_sns]

        if curr_status == 1 and prev_status == 1:
            assert curr_p - prev_p <= 0.5 * 100 + 1e-5
            assert prev_p - curr_p <= 0.5 * 100 + 1e-5


def test_ramp_limit_start_up_binary_uc():
    """
    Test that ramp_limit_start_up parameter works correctly in binary unit commitment.
    """
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", ["gas", "electricity"])
    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    # Committable link with ramp_limit_start_up set, but ramp_limit_up is NaN
    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        ramp_limit_start_up=0.4,  # 40% of p_nom on start-up
        p_nom=10000,
        up_time_before=0,  # Starts from OFF state
        start_up_cost=3333,
    )

    # Backstop generator to ensure feasibility
    n.add(
        "Generator",
        "expensive_backstop",
        bus="electricity",
        p_nom=5000,
        marginal_cost=1e5,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 5000, 2000, 5000])

    status, condition = n.optimize()

    assert status == "ok", f"Optimization failed with status {status}"

    # Get the link output (convert p1 to positive power output)
    link_output = -n.c.links.dynamic.p1.loc[:, "OCGT"].values

    # Expected ramp limit on start-up: 0.4 * 10000 = 4000 MW
    max_start_up_output = 0.4 * 10000

    # First snapshot: unit starts from OFF, should respect ramp_limit_start_up
    assert link_output[0] <= max_start_up_output * 1.01, (
        f"First snapshot output {link_output[0]} exceeds start-up ramp limit {max_start_up_output}"
    )

    # Verify unit is committed (status = 1)
    assert n.c.links.dynamic.status.loc[0, "OCGT"] == 1, (
        "Link should be committed at first snapshot"
    )

    # Verify start_up variable catches the start-up event
    assert n.c.links.dynamic.start_up.loc[0, "OCGT"] == 1, (
        "Link should show start-up at first snapshot"
    )


def test_infeasible_start_up_limit():
    """Test that introduction of infeasible start-up limit results in infeasible solution"""
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])

    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        min_up_time=3,
        start_up_cost=3333,
        p_nom=12000,
        up_time_before=0,
    )

    n.add(
        "Generator",
        "wind",
        bus="electricity",
        p_nom=800,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 6000, 800, 5000])

    status, condition = n.optimize()
    assert status == "ok"

    n.c.links.static.loc["OCGT", "ramp_limit_start_up"] = 0.01

    status, condition = n.optimize()
    assert status == "warning"
    assert condition == "infeasible"


def test_ramp_limit_shut_down_binary_uc():
    """
    Test that ramp_limit_shut_down parameter works correctly in binary unit commitment (excluding first period as special case).
    """
    n = pypsa.Network()
    snapshots = range(4)
    n.set_snapshots(snapshots)

    n.add("Bus", ["gas", "electricity"])
    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        ramp_limit_start_up=0.4,
        start_up_cost=3333,
        p_nom=10000,
        status=1,
        ramp_limit_shut_down=0.1,  # 10% of p_nom on shut-down
    )

    n.add(
        "Generator",
        "expensive_backstop",
        bus="electricity",
        p_nom=5000,
        marginal_cost=1e5,
    )

    n.add(
        "Generator",
        "slack_dump",
        bus="electricity",
        p_nom=20000,
        marginal_cost=1e6,
        sign=-1,
    )

    n.add("Load", "load", bus="electricity", p_set=[4000, 5000, 2000, 0])

    status, condition = n.optimize()

    assert status == "ok", f"Optimization failed with status {status}"

    # Get generator outputs
    gas_output = n.c.generators.dynamic.p.loc[:, "gas"].values
    backstop_output = n.c.generators.dynamic.p.loc[:, "expensive_backstop"].values

    # Expected pattern based on ramp_limit_shut_down constraint
    # Snapshot 0: 9000.0 MW (gas), 0.0 MW (backstop), 500.0 MW (slack)
    # Snapshot 1: 10000.0 MW (gas), 0.0 MW (backstop), 0.0 MW (slack)
    # Snapshot 2: 1000.0 MW (gas), 1500.0 MW (backstop), 0.0 MW (slack)
    # Snapshot 3: 0.0 MW (gas), 0.0 MW (backstop), 0.0 MW (slack)

    assert abs(gas_output[1] - 10000.0) < 1e-3, (
        f"Snapshot 1: Expected gas output 10000.0, got {gas_output[1]}"
    )
    assert abs(backstop_output[1] - 0.0) < 1e-3, (
        f"Snapshot 1: Expected backstop output 0.0, got {backstop_output[1]}"
    )

    assert abs(gas_output[2] - 1000.0) < 1e-3, (
        f"Snapshot 2: Expected gas output 1000.0, got {gas_output[2]}"
    )
    assert abs(backstop_output[2] - 1500.0) < 1e-3, (
        f"Snapshot 2: Expected backstop output 1500.0, got {backstop_output[2]}"
    )

    assert abs(gas_output[3] - 0.0) < 1e-3, (
        f"Snapshot 3: Expected gas output 0.0, got {gas_output[3]}"
    )
    assert abs(backstop_output[3] - 0.0) < 1e-3, (
        f"Snapshot 3: Expected backstop output 0.0, got {backstop_output[3]}"
    )


def test_ramp_limit_shut_down_first_snapshot_with_slack():
    """
    Test that ramp_limit_down and ramp_limit_shut_down constrain dispatch
    at snapshot 0 when unit starts ON at full capacity with zero demand.
    """
    n = pypsa.Network()
    n.set_snapshots(range(4))

    n.add("Bus", ["gas", "electricity"])
    n.add("Generator", "gas", bus="gas", marginal_cost=10, p_nom=20000)

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="electricity",
        committable=True,
        p_min_pu=0.1,
        efficiency=0.5,
        ramp_limit_start_up=0.4,
        start_up_cost=3333,
        p_nom=10000,
        p_init=10000,
        ramp_limit_down=0.1,
        ramp_limit_shut_down=0.1,
    )

    n.add(
        "Generator",
        "expensive_backstop",
        bus="electricity",
        p_nom=5000,
        marginal_cost=1e5,
    )
    n.add(
        "Generator",
        "slack_dump",
        bus="electricity",
        p_nom=20000,
        marginal_cost=1e6,
        sign=-1,
    )
    n.add("Load", "load", bus="electricity", p_set=[0, 5000, 2000, 0])

    status, condition = n.optimize()

    assert status == "ok", f"Optimization failed with status {status}"

    gas_output = n.c.generators.dynamic.p.loc[:, "gas"].values
    slack_output = n.c.generators.dynamic.p.loc[:, "slack_dump"].values

    assert abs(gas_output[0] - 9000.0) < 1e-3, (
        f"Snapshot 0: Expected gas output 9000.0, got {gas_output[0]}"
    )
    assert abs(slack_output[0] - 4500.0) < 1e-3, (
        f"Snapshot 0: Expected slack output 4500.0, got {slack_output[0]}"
    )


@pytest.mark.parametrize(
    "generator_kwargs",
    [
        {},
        {"p_nom_extendable": True, "p_nom_mod": 50},
        {"ramp_limit_up": 0.5},
    ],
    ids=["fixed", "extendable-modular", "ramp-limited"],
)
@pytest.mark.parametrize("committable", [False, True], ids=["sole", "alongside-active"])
def test_inactive_committable_generator(generator_kwargs, committable):
    """Inactive committables get no status variable, so must not be constrained."""
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "active",
        bus="bus",
        p_nom=100,
        marginal_cost=50,
        committable=committable,
        **(generator_kwargs if committable else {}),
    )
    n.add(
        "Generator",
        "inactive",
        bus="bus",
        p_nom=100,
        committable=True,
        active=False,
        marginal_cost=60,
        **generator_kwargs,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    n.optimize.create_model(include_objective_constant=False)

    if committable:
        assert "inactive" not in n.model.variables["Generator-status"].indexes["name"]
    else:
        assert "Generator-status" not in n.model.variables


def test_inactive_committable_generator_multi_investment():
    """Narrowing to active assets must not drop assets active in only some periods."""
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "active",
        bus="bus",
        p_nom=100,
        marginal_cost=50,
        committable=True,
        ramp_limit_up=0.5,
    )
    n.add(
        "Generator",
        "later",
        bus="bus",
        p_nom=100,
        marginal_cost=70,
        committable=True,
        ramp_limit_up=0.5,
        build_year=2030,
        lifetime=20,
    )
    n.add(
        "Generator",
        "inactive",
        bus="bus",
        p_nom=100,
        marginal_cost=60,
        committable=True,
        active=False,
    )
    n.add("Load", "load", bus="bus", p_set=50)

    n.optimize.create_model(
        include_objective_constant=False, multi_investment_periods=True
    )

    names = n.model.variables["Generator-status"].indexes["name"]
    assert "inactive" not in names
    assert "later" in names


def test_inactive_asset_with_modular_committable_ramp():
    """Ramp limits on a modular extendable committable, alongside any inactive asset."""
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "bus")
    n.add("Generator", "backup", bus="bus", p_nom=500, marginal_cost=200)
    n.add(
        "Generator",
        "com_mod_ext",
        bus="bus",
        p_nom_extendable=True,
        p_nom_mod=50,
        p_nom_max=200,
        committable=True,
        marginal_cost=12,
        ramp_limit_up=0.5,
    )
    n.add("Generator", "inactive", bus="bus", p_nom=100, active=False, marginal_cost=99)
    n.add("Load", "load", bus="bus", p_set=120)

    n.optimize.create_model(include_objective_constant=False)

    ramp_up = n.model.constraints["Generator-p-ramp_limit_up"]
    assert "inactive" not in ramp_up.indexes["name"]


@pytest.mark.parametrize("min_down_time", [0, 1, 2])
def test_modular_min_down_time_keeps_other_modules_available(min_down_time):
    """Min down time only holds back the shut-down modules, not the whole unit."""
    n = pypsa.Network(snapshots=range(4))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 150, 150, 100])
    n.add("Generator", "shed", bus="bus", p_nom=1e3, marginal_cost=1e3)
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=150,
        p_nom_mod=50,
        p_min_pu=1,
        committable=True,
        marginal_cost=10,
        min_down_time=min_down_time,
        up_time_before=0,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    assert n.generators_t.status["gen"].tolist() == [2, 3, 3, 2]
    assert n.generators_t.p["shed"].round(6).eq(0).all()


@pytest.mark.parametrize("extendable", [False, True])
def test_modular_min_down_time_holds_back_shut_down_module(extendable):
    """A shut-down module stays off for ``min_down_time`` while the others run."""
    n = pypsa.Network(snapshots=range(4))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[150, 100, 150, 150])
    # shedding at the first snapshot is dearer, so the solution is unique
    n.add("Generator", "shed", bus="bus", p_nom=1e3, marginal_cost=[2e3, 1e3, 1e3, 1e3])
    capacity = (
        {"p_nom_extendable": True, "p_nom_max": 150, "capital_cost": 1}
        if extendable
        else {"p_nom": 150}
    )
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_mod=50,
        p_min_pu=1,
        committable=True,
        marginal_cost=10,
        min_down_time=2,
        up_time_before=0,
        **capacity,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    assert n.c.generators.static.p_nom_opt["gen"] == 150
    assert n.generators_t.status["gen"].tolist() == [3, 2, 2, 3]
    assert n.generators_t.p["shed"].round(6).tolist() == [0, 0, 50, 0]
    if extendable:
        n_mod_label = int(n.model.variables["Generator-n_mod"].labels.sel(name="gen"))
        con = n.model.constraints["Generator-com-down-time"]
        assert (con.lhs.vars == n_mod_label).any()


@pytest.mark.parametrize("linearized", [False, True])
def test_time_varying_start_up_and_shut_down_costs(linearized):
    """Cheap start-up before and cheap shut-down after the peak extend the commitment."""
    n = pypsa.Network(snapshots=range(5))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "cheap",
        bus="bus",
        committable=True,
        p_nom=10,
        p_min_pu=0.01,
        stand_by_cost=10,
        up_time_before=0,
        start_up_cost=pd.Series([0.0, 0.0, 100.0, 100.0, 100.0], index=n.snapshots),
        shut_down_cost=pd.Series([100.0, 100.0, 100.0, 100.0, 0.0], index=n.snapshots),
    )
    n.add("Generator", "expensive", bus="bus", p_nom=10, marginal_cost=100)
    n.add("Load", "load", bus="bus", p_set=[0.0, 0.1, 10.0, 0.1, 0.1])

    status, _ = n.optimize(solver_name="highs", linearized_unit_commitment=linearized)
    assert status == "ok"
    if linearized:
        return
    expected = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0], index=n.snapshots, name="cheap")
    pd.testing.assert_series_equal(n.c.generators.dynamic.status["cheap"], expected)
    assert n.objective == pytest.approx(10 * 3 + 0.1 * 100)


def test_linearized_uc_tightening_with_time_varying_costs(caplog):
    """Tightening applies per unit only if costs are equal in every snapshot."""
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "bus")
    equal = pd.Series([10.0, 20.0, 30.0], index=n.snapshots)
    n.add(
        "Generator",
        "equal",
        bus="bus",
        committable=True,
        p_nom=100,
        start_up_cost=equal,
        shut_down_cost=equal,
    )
    n.add(
        "Generator",
        "unequal",
        bus="bus",
        committable=True,
        p_nom=100,
        start_up_cost=equal,
        shut_down_cost=equal + [0.0, 1.0, 0.0],
    )
    n.add("Load", "load", bus="bus", p_set=50)

    n.optimize.create_model(linearized_unit_commitment=True)

    names = n.model.constraints["Generator-com-p-current"].indexes["name"]
    assert list(names) == ["equal"]
    assert "cannot be tightened" in caplog.text


@pytest.mark.filterwarnings("ignore::linopy.constants.PerformanceWarning")
@pytest.mark.parametrize("semantics", ["legacy", "v1"])
@pytest.mark.parametrize("short_time", [None, 0, 1])
def test_minimum_time_rows_at_window_boundary(
    semantics: str, short_time: int | None
) -> None:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2026-01-01", periods=6, freq="h"))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "generator",
        bus="bus",
        p_nom=10,
        committable=True,
        marginal_cost=1,
        min_up_time=3,
        min_down_time=3,
        up_time_before=0,
    )
    if short_time is not None:
        n.add(
            "Generator",
            "short",
            bus="bus",
            p_nom=10,
            committable=True,
            marginal_cost=2,
            min_up_time=short_time,
            min_down_time=short_time,
            up_time_before=0,
        )
    n.add("Load", "load", bus="bus", p_set=1)
    if semantics == "legacy" and linopy_uses_v1():
        pytest.skip("linopy's legacy arithmetic semantics are not available")
    v1 = semantics == "v1"
    with v1_semantics() if v1 else nullcontext():
        model = n.optimize.create_model(sparse=v1)
        status = model.variables["Generator-status"].labels.values
        for kind, sign, transition in [
            ("up", -1, "start_up"),
            ("down", 1, "shut_down"),
        ]:
            values = model.variables[f"Generator-{transition}"].labels.values
            data = model.constraints[f"Generator-com-{kind}-time"].data
            np.testing.assert_array_equal(
                data.labels.values >= 0,
                np.tile(
                    [True, short_time != 0] if short_time is not None else [True],
                    (6, 1),
                ),
            )
            for generator, duration in enumerate(
                [3, short_time] if short_time is not None else [3]
            ):
                assert duration is not None
                if duration == 0:
                    continue
                for snapshot in range(6):
                    variables = data.vars.values[snapshot, generator]
                    coefficients = data.coeffs.values[snapshot, generator]
                    actual = {
                        int(label): float(coefficient)
                        for label, coefficient in zip(variables, coefficients)
                        if label >= 0 and coefficient != 0
                    }
                    expected = {int(status[snapshot, generator]): float(sign)} | {
                        int(values[t, generator]): 1.0
                        for t in range(max(0, snapshot - duration + 1), snapshot + 1)
                    }
                    assert actual == expected


@pytest.mark.parametrize(
    "flex_kwargs",
    [{"p_nom": 10}, {"p_nom_extendable": True, "capital_cost": 1}],
    ids=["fixed", "extendable"],
)
@pytest.mark.parametrize("with_committable", [False, True])
def test_noncommittable_first_snapshot_ramp(flex_kwargs, with_committable):
    """Regression test for issue #1943.

    The first-snapshot ramp limit of a non-committable unit neither reads
    `up_time_before` nor depends on other units being committable.
    """
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=5)
    n.add(
        "Generator",
        "flex",
        bus="bus",
        marginal_cost=1,
        ramp_limit_up=0.5,
        up_time_before=0,
        **flex_kwargs,
    )
    if with_committable:
        n.add(
            "Generator", "unit", bus="bus", p_nom=10, marginal_cost=2, committable=True
        )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.dynamic.p.loc[0, "flex"] == pytest.approx(5)


def _ramp_series_network(ramp_attr, series, load, extendable=False):
    n = pypsa.Network(snapshots=range(len(load)))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=load)
    n.add("Generator", "backstop", bus="bus", p_nom=1000, marginal_cost=1000)
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=100,
        p_nom_extendable=extendable,
        p_nom_max=100,
        committable=True,
        p_min_pu=0.1,
        marginal_cost=1,
        **{ramp_attr: pd.Series(series, index=n.snapshots)},
    )
    return n


@pytest.mark.parametrize("extendable", [False, True])
def test_ramp_limit_start_up_series(extendable):
    series = [1, 1, 1, 0.4, 1]
    n = _ramp_series_network(
        "ramp_limit_start_up", series, [0, 80, 0, 80, 0], extendable
    )
    status, _ = n.optimize()
    assert status == "ok"
    p = n.c.generators.dynamic.p["gen"]
    assert p.iloc[1] == pytest.approx(80)
    assert p.iloc[3] == pytest.approx(40)


@pytest.mark.parametrize("extendable", [False, True])
def test_ramp_limit_shut_down_series(extendable):
    series = [1, 0.5, 1, 1]
    n = _ramp_series_network("ramp_limit_shut_down", series, [80, 0, 80, 0], extendable)
    status, _ = n.optimize()
    assert status == "ok"
    p = n.c.generators.dynamic.p["gen"]
    assert p.iloc[0] == pytest.approx(50)
    assert p.iloc[2] == pytest.approx(80)


def test_ramp_limit_start_up_series_equals_static():
    load = [0, 80, 0, 80, 0]
    n_series = _ramp_series_network("ramp_limit_start_up", [0.4] * 5, load)
    n_static = _ramp_series_network("ramp_limit_start_up", [1] * 5, load)
    n_static.c.generators.dynamic.ramp_limit_start_up = pd.DataFrame(
        index=n_static.snapshots
    )
    n_static.c.generators.static.loc["gen", "ramp_limit_start_up"] = 0.4
    n_static.optimize()
    n_series.optimize()
    assert n_static.objective == pytest.approx(n_series.objective)


def _min_time_network(
    weights, min_up, load_at, periods=None, kind="min_up_time"
) -> pypsa.Network:
    """Committable unit that is only needed at one snapshot, staying on costs."""
    steps = len(weights)
    if periods:
        n = pypsa.Network(snapshots=range(steps // 2))
        n.investment_periods = periods
    else:
        n = pypsa.Network(snapshots=range(steps))
    for col in n.snapshot_weightings.columns:
        n.snapshot_weightings[col] = np.array(weights, dtype=float)
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=10,
        committable=True,
        stand_by_cost=1,
        up_time_before=0,
        **{kind: min_up},
    )
    n.add("Generator", "backup", bus="bus", p_nom=10, marginal_cost=100)
    load = np.zeros(steps)
    load[load_at] = 5
    n.add("Load", "load", bus="bus", p_set=load)
    return n


def _status(n: pypsa.Network) -> list[int]:
    return n.c.generators.dynamic.status["gen"].round().astype(int).tolist()


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(True, [0, 1, 1, 0, 0, 0]), (False, [0, 1, 1, 1, 0, 0])],
)
def test_min_up_time_hours_vs_snapshots(hours, expected):
    n = _min_time_network([2] * 6, 3, 1)
    with (
        pypsa.option_context("params.optimize.durations_in_hours", hours),
        nullcontext() if hours else pytest.warns(FutureWarning, match="2.0"),
    ):
        n.optimize()
    assert _status(n) == expected


def test_min_up_time_uneven_weightings():
    n = _min_time_network([1, 1, 3, 1, 1, 1], 3, 1)
    with pypsa.option_context("params.optimize.durations_in_hours", True):
        n.optimize()
    assert _status(n) == [0, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("hours", [True, False])
def test_min_up_time_hourly_weightings_identical(hours):
    n = _min_time_network([1] * 6, 3, 1)
    with pypsa.option_context("params.optimize.durations_in_hours", hours):
        n.optimize()
    assert _status(n) == [0, 1, 1, 1, 0, 0]


@pytest.mark.parametrize("hours", [True, False])
def test_float_min_up_time_accepted(hours):
    n = _min_time_network([1] * 6, 2.5, 1)
    with pypsa.option_context("params.optimize.durations_in_hours", hours):
        n.optimize()
    assert _status(n) == [0, 1, 1, 1, 0, 0]


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(True, [0, 0, 1, 0, 0, 0]), (False, [0, 0, 1, 1, 1, 0])],
)
def test_min_up_time_hours_stop_at_period_start(hours, expected):
    n = _min_time_network([1] * 6, 3, 2, periods=[2020, 2030])
    # starting early costs more, so the start is at the end of the first period
    n.c.generators.dynamic.stand_by_cost["gen"] = [5, 5, 1, 1, 1, 1]
    with pypsa.option_context("params.optimize.durations_in_hours", hours):
        n.optimize(multi_investment_periods=True)
    assert _status(n) == expected


def test_min_up_time_hours_zero_weighted_snapshot():
    n = _min_time_network([1, 0, 1, 1, 1, 1], 3, 2)
    with pypsa.option_context("params.optimize.durations_in_hours", True):
        status, _ = n.optimize()
    assert status == "ok"
    assert _status(n)[2] == 1


@pytest.mark.parametrize("zeros", [1, 2, 3])
def test_min_up_time_hours_many_zero_weighted_snapshots(zeros):
    weights = [1, *[0] * zeros, 1, 1]
    n = _min_time_network(weights, 2, 0)
    with pypsa.option_context("params.optimize.durations_in_hours", True):
        n.optimize()
    # the zero-weighted snapshots add no hours, so the unit stays on through
    # the first snapshot after them
    assert _status(n) == [1] * (zeros + 2) + [0]


def _weighted_ramp_network(kind, weight=2, **gen_kwargs):
    n = pypsa.Network(snapshots=range(4))
    n.snapshot_weightings = pd.DataFrame(
        weight, index=n.snapshots, columns=n.snapshot_weightings.columns
    )
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0, 100, 100, 100])
    n.add("Generator", "backstop", bus="bus", p_nom=1000, marginal_cost=1000)
    kw = {"p_nom": 100, "marginal_cost": 1, "ramp_limit_up": 0.1}
    if kind in ("committable", "linearized", "com_ext"):
        kw |= {
            "committable": True,
            "up_time_before": 0,
            "ramp_limit_start_up": 0.1,
        }
    if kind == "com_ext":
        kw |= {"p_nom_extendable": True, "p_nom_max": 100, "capital_cost": 0}
    n.add("Generator", "gen", bus="bus", **(kw | gen_kwargs))
    return n


def _ramp_run(n, hours, **kwargs):
    weighted = bool((n.snapshot_weightings.generators != 1).any())
    with (
        pypsa.option_context("params.optimize.durations_in_hours", hours),
        pytest.warns(FutureWarning, match="2.0")
        if weighted and not hours
        else nullcontext(),
    ):
        status, _ = n.optimize(**kwargs)
    assert status == "ok"
    return n.c.generators.dynamic.p["gen"]


@pytest.mark.parametrize("kind", ["fixed", "committable", "linearized", "com_ext"])
@pytest.mark.parametrize(("hours", "step"), [(True, 20), (False, 10)])
def test_ramp_limit_up_scaled_by_weighting(kind, hours, step):
    n = _weighted_ramp_network(kind)
    kwargs = {"linearized_unit_commitment": True} if kind == "linearized" else {}
    p = _ramp_run(n, hours, **kwargs)
    # the unit is on from the second snapshot and may ramp up from there
    start = p.iloc[1]
    assert p.iloc[2] - start == pytest.approx(step, abs=1e-4)


@pytest.mark.parametrize("kind", ["fixed", "committable", "com_ext"])
def test_ramp_limit_hourly_weightings_same_objective(kind):
    objs = []
    for hours in (True, False):
        n = _weighted_ramp_network(kind, weight=1)
        _ramp_run(n, hours)
        objs.append(n.objective)
    assert objs[0] == pytest.approx(objs[1])


@pytest.mark.parametrize(("hours", "start"), [(True, 40), (False, 20)])
def test_ramp_limit_start_up_scaled_by_weighting(hours, start):
    n = _weighted_ramp_network(
        "committable", ramp_limit_start_up=0.2, ramp_limit_up=1, p_min_pu=0.1
    )
    p = _ramp_run(n, hours)
    assert p.iloc[1] == pytest.approx(start, abs=1e-4)


@pytest.mark.parametrize("kind", ["fixed", "committable", "linearized", "com_ext"])
@pytest.mark.parametrize(("hours", "step"), [(True, 20), (False, 10)])
def test_ramp_limit_down_scaled_by_weighting(kind, hours, step):
    kw = {"ramp_limit_down": 0.1, "ramp_limit_up": 1}
    if kind != "fixed":
        # a tight shut-down limit makes shutting down no way around ramp_limit_down
        kw |= {
            "up_time_before": 1,
            "ramp_limit_start_up": 1,
            "ramp_limit_shut_down": 0.1,
            "p_min_pu": 0.05,
        }
    n = _weighted_ramp_network(kind, **kw)
    n.loads_t.p_set["load"] = [100, 100, 100, 50]
    kwargs = {"linearized_unit_commitment": True} if kind == "linearized" else {}
    p = _ramp_run(n, hours, **kwargs)
    # the load drops in the last snapshot, so the one before is one step above
    assert p.iloc[2] == pytest.approx(50 + step, abs=1e-4)


@pytest.mark.parametrize("kind", ["committable", "com_ext"])
@pytest.mark.parametrize(("hours", "step"), [(True, 20), (False, 10)])
def test_ramp_limit_shut_down_scaled_by_weighting(kind, hours, step):
    n = _weighted_ramp_network(
        kind,
        ramp_limit_down=1,
        ramp_limit_up=1,
        ramp_limit_shut_down=0.1,
        ramp_limit_start_up=1,
        p_min_pu=0.1,
        up_time_before=1,
    )
    n.loads_t.p_set["load"] = [100, 100, 100, 0]
    p = _ramp_run(n, hours)
    assert p.iloc[2] == pytest.approx(step, abs=1e-4)


def _warn_network(weight=2, **gen_kwargs) -> pypsa.Network:
    n = pypsa.Network(snapshots=range(3))
    n.snapshot_weightings = pd.DataFrame(
        weight, index=n.snapshots, columns=n.snapshot_weightings.columns
    )
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=10)
    n.add("Generator", "gen", bus="bus", p_nom=100, marginal_cost=1, **gen_kwargs)
    return n


def _optimize_collecting_warnings(n, hours=False):
    with (
        pypsa.option_context("params.optimize.durations_in_hours", hours),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        n.optimize()
    return [w for w in caught if "durations_in_hours" in str(w.message)]


def test_warn_legacy_durations_all_conditions():
    n = _warn_network(ramp_limit_up=0.5)
    with pytest.warns(FutureWarning, match="2.0.*durations_in_hours") as record:
        with pypsa.option_context("params.optimize.durations_in_hours", False):
            n.optimize()
    assert len([w for w in record if "durations_in_hours" in str(w.message)]) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ramp_limit_down": 0.5},
        {"ramp_limit_start_up": 0.5, "committable": True},
        {"ramp_limit_shut_down": 0.5, "committable": True},
        {"committable": True, "min_up_time": 2},
        {"committable": True, "min_down_time": 2},
    ],
)
def test_warn_legacy_durations_each_attribute(kwargs):
    n = _warn_network(**kwargs)
    assert len(_optimize_collecting_warnings(n)) == 1


def test_warn_legacy_durations_dynamic_ramp_limit():
    n = _warn_network()
    n.c.generators.dynamic.ramp_limit_up["gen"] = [1, 0.5, 1]
    assert len(_optimize_collecting_warnings(n)) == 1


def test_warn_no_warning_with_option_on():
    n = _warn_network(ramp_limit_up=0.5)
    assert _optimize_collecting_warnings(n, hours=True) == []


def test_warn_no_warning_with_unit_weightings():
    n = _warn_network(weight=1, ramp_limit_up=0.5)
    assert _optimize_collecting_warnings(n) == []


def test_warn_no_warning_without_ramp_limits_or_min_times():
    n = _warn_network(committable=True)
    assert _optimize_collecting_warnings(n) == []


def test_warn_no_warning_for_inactive_component():
    n = _warn_network(ramp_limit_up=0.5, active=False)
    n.add("Generator", "backup", bus="bus", p_nom=100, marginal_cost=5)
    assert _optimize_collecting_warnings(n) == []


def test_warn_no_warning_for_min_time_without_committable():
    n = _warn_network(min_up_time=2)
    assert _optimize_collecting_warnings(n) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"ramp_limit_up": 0.5}, {"committable": True, "min_up_time": 2}],
)
def test_warn_legacy_durations_stochastic(kwargs):
    n = _warn_network(**kwargs)
    n.add("Generator", "gen2", bus="bus", p_nom=100, marginal_cost=2)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    with pypsa.option_context("params.optimize.durations_in_hours", False):
        with pytest.warns(FutureWarning, match="durations_in_hours"):
            n.optimize.create_model()


def test_warn_no_warning_stochastic_attributes_in_different_scenarios():
    n = _warn_network(committable=True)
    n.add("Generator", "gen2", bus="bus", p_nom=100, marginal_cost=2)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.generators.static.loc[("high", "gen2"), "min_up_time"] = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pypsa.option_context("params.optimize.durations_in_hours", False):
            n.optimize.create_model()
    assert [w for w in caught if "durations_in_hours" in str(w.message)] == []
