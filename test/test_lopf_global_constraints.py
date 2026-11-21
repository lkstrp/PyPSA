# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa


def test_operational_limit_n_ac_dc_meshed(ac_dc_network):
    n = ac_dc_network

    limit = 30_000

    n.remove("GlobalConstraint", n.c.global_constraints.static.index)

    n.add(
        "GlobalConstraint",
        "gas_limit",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=limit,
    )

    n.optimize()
    assert n.statistics.energy_balance().loc[:, "gas"].sum().round(3) == limit


def test_operational_limit_storage_hvdc(storage_hvdc_network):
    n = storage_hvdc_network

    limit = 5_000

    n.remove("GlobalConstraint", n.c.global_constraints.static.index)

    n.add(
        "GlobalConstraint",
        "battery_limit",
        type="operational_limit",
        carrier_attribute="battery",
        sense="<=",
        constant=limit,
    )

    n.c.storage_units.static["state_of_charge_initial"] = 1_000
    n.c.storage_units.static.p_nom_extendable = True
    n.c.storage_units.static.cyclic_state_of_charge = False

    n.optimize()

    c = n.c.storage_units
    soc_diff = (
        c.static.state_of_charge_initial.sum()
        - c.dynamic.state_of_charge.sum(axis=1).iloc[-1]
    )
    net_inflow = c._as_dynamic("inflow") - c.dynamic.spill
    net_inflow = net_inflow.mul(n.snapshot_weightings.stores, axis=0).sum().sum()
    assert net_inflow > 0
    assert soc_diff + net_inflow == pytest.approx(limit)


@pytest.mark.parametrize("assign", [True, False])
def test_assign_all_duals(ac_dc_network, assign):
    n = ac_dc_network

    limit = 30_000

    m = n.optimize.create_model()

    transmission = m.variables["Link-p"]
    m.add_constraints(
        transmission.sum() <= limit, name="GlobalConstraint-generation_limit"
    )
    m.add_constraints(
        transmission.sum(dim="name") <= limit,
        name="GlobalConstraint-generation_limit_dynamic",
    )

    n.optimize.solve_model(assign_all_duals=assign)

    assert ("generation_limit" in n.c.global_constraints.static.index) == assign
    assert ("mu_generation_limit_dynamic" in n.c.global_constraints.dynamic) == assign


def test_assign_duals_noname(ac_dc_network):
    """Test that dual values are correctly assigned back to network,
    also for a special case of constraints without component dimension."""
    n = ac_dc_network

    limit = 10000
    m = n.optimize.create_model()
    investment = m.variables["Generator-p_nom"]
    m.add_constraints(
        investment.sum() == limit, name="GlobalConstraint-investment_limit"
    )
    n.optimize.solve_model(assign_all_duals=True)

    dual_model_investment = float(
        n.model.constraints["GlobalConstraint-investment_limit"].dual
    )
    dual_network_investment = float(
        n.c.global_constraints.static.mu.loc["investment_limit"]
    )
    assert dual_model_investment == pytest.approx(
        dual_network_investment, rel=1e-8, abs=1e-10
    )

    dual_model_co2 = float(n.model.constraints["GlobalConstraint-co2_limit"].dual)
    dual_network_co2 = float(n.c.global_constraints.static.mu.loc["co2_limit"])
    assert dual_model_co2 == pytest.approx(dual_network_co2, rel=1e-8, abs=1e-10)


def test_assign_custom_variable(ac_dc_network):
    """Test that adding custom variables does not raise an error when optimizing."""
    n = ac_dc_network
    m = n.optimize.create_model()

    m.add_variables(name="custom_var")

    n.optimize.solve_model()


def test_custom_variable_warnings(ac_dc_network, caplog):
    """Test that appropriate warnings are logged for custom variables."""
    import logging

    caplog.set_level(logging.INFO)

    m = ac_dc_network.optimize.create_model()
    m.add_variables(name="custom_var_no_dash")
    m.add_variables(name="NonExistentComponent-attr")
    ac_dc_network.optimize.solve_model()

    assert all(
        text in caplog.text
        for text in [
            "could not be mapped to the network component because it does not include the symbol '-'",
            "custom_var_no_dash",
            "could not be mapped to the network component because the component 'NonExistentComponent' does not exist",
        ]
    )


def test_1449():
    """
    Inactive components with global carrier constraint should not break.
    See https://github.com/PyPSA/PyPSA/issues/1449.
    """
    snapshots = pd.date_range("2025-01-01", freq="h", periods=5)

    n = pypsa.Network(snapshots=snapshots)

    n.add("Carrier", name="gird_with_emissions", co2_emissions=0.5)

    n.add("Bus", name="b_electricity", unit="MW")

    n.add(
        "Generator",
        name="grid",
        bus="b_electricity",
        p_nom=10,
        marginal_cost=5,
    )

    n.add(
        "Generator",
        name="grid_2",
        bus="b_electricity",
        carrier="gird_with_emissions",
        p_nom_extendable=True,
        marginal_cost=5,
        active=False,
    )

    n.add(
        "Load",
        name="demand",
        bus="b_electricity",
        p_set=pd.Series([0, 2, 0, 0, 0], index=snapshots),
    )

    n.add(
        "GlobalConstraint",
        name="CO2_limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=50,
    )

    status, _ = n.optimize()
    assert status == "ok"


def test_1884():
    """
    A tech_capacity_expansion_limit with no investment_period must still apply.
    See https://github.com/PyPSA/PyPSA/issues/1884.
    """
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "b")
    n.add("Carrier", ["wind", "gas"])
    n.add("Load", "l", bus="b", p_set=100)
    # cheap wind is capped at 50, so expensive gas must cover the rest
    n.add(
        "Generator",
        ["wind", "gas"],
        bus="b",
        carrier=["wind", "gas"],
        p_nom_extendable=True,
        capital_cost=1,
        marginal_cost=[1, 100],
    )
    n.add(
        "GlobalConstraint",
        "wind_limit",
        type="tech_capacity_expansion_limit",
        carrier_attribute="wind",
        sense="<=",
        constant=50,
    )

    n.optimize()
    assert n.c.generators.static.p_nom_opt["wind"] == pytest.approx(50)


@pytest.mark.parametrize(
    ("glc_type", "carrier_attribute"),
    [("primary_energy", "co2_emissions"), ("operational_limit", "gas")],
)
def test_investment_period_without_multi_invest_raises(glc_type, carrier_attribute):
    """
    An investment_period raises if the network has no investment periods.
    See https://github.com/PyPSA/PyPSA/issues/1954.
    """
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "b")
    n.add("Carrier", ["gas", "wind"], co2_emissions=[1, 0])
    n.add("Load", "l", bus="b", p_set=50)
    n.add(
        "Generator",
        ["gas", "wind"],
        bus="b",
        carrier=["gas", "wind"],
        p_nom=100,
        marginal_cost=[1, 100],
    )
    n.add(
        "GlobalConstraint",
        "gas_limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=100,
        investment_period=0,
    )

    with pytest.raises(ValueError, match="has no investment periods"):
        n.optimize()


def test_transmission_cost_limit_overnight_cost():
    n = pypsa.Network()
    n.snapshot_weightings.loc[:, :] = 8760.0
    n.add("Bus", ["a", "b"])
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.01,
        s_nom_extendable=True,
        overnight_cost=1000,
        discount_rate=0,
        lifetime=10,
    )
    n.add(
        "GlobalConstraint",
        "cost_limit",
        type="transmission_expansion_cost_limit",
        constant=10000,
        sense="<=",
        carrier_attribute="AC",
    )
    m = n.optimize.create_model()
    con = m.constraints["GlobalConstraint-cost_limit"]
    assert (con.coeffs == 100.0).all()


def _constraint_rows(m, name):
    """Terms, sign and rhs of each row of a constraint, keyed by scenario."""
    data = m.constraints[name].data
    scenarios = data.indexes["scenario"] if "scenario" in data.dims else [None]
    rows = {}
    for scenario in scenarios:
        row = data if scenario is None else data.sel(scenario=scenario)
        terms = {}
        for label, coeff in zip(
            row.vars.values.ravel(), row.coeffs.values.ravel(), strict=True
        ):
            if label < 0:
                continue
            var, pos = m.variables.get_label_position(int(label))
            assert pos.get("scenario") == scenario
            key = (var, pos["name"], pos["snapshot"])
            terms[key] = terms.get(key, 0.0) + float(coeff)
        rows[scenario] = (terms, str(row.sign.item()), float(row.rhs.item()))
    return rows


@pytest.mark.parametrize(
    "mode", ["single", "continuous", "per_period", "stochastic_per_period"]
)
@pytest.mark.parametrize("component", ["StorageUnit", "Store"])
@pytest.mark.parametrize("kind", ["primary_energy", "operational_limit"])
def test_storage_carry_over_terms(kind, component, mode):
    """Non-cyclic storage enters the global constraint as its window carry-over."""
    emission = 0.5
    constant = 3.0
    initial = {"low": 6.0, "high": 4.0}
    multi_invest = mode != "single"
    per_period = mode in ("per_period", "stochastic_per_period")
    years = {2030: 10.0, 2040: 20.0} if per_period else {2030: 1.0, 2040: 1.0}

    n = pypsa.Network()
    if multi_invest:
        n.snapshots = pd.MultiIndex.from_product([[2030, 2040], [0, 1]])
        n.investment_periods = [2030, 2040]
        n.investment_period_weightings["years"] = pd.Series(years)
    else:
        n.snapshots = range(3)
    n.add("Carrier", "gas", co2_emissions=emission)
    n.add("Carrier", "wind")
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=1)
    n.add("Generator", "gen", bus="bus", carrier="wind", p_nom=2, marginal_cost=10)
    if component == "StorageUnit":
        var, initial_attr = "StorageUnit-state_of_charge", "state_of_charge_initial"
        n.add(
            "StorageUnit",
            "storage",
            bus="bus",
            carrier="gas",
            p_nom=10,
            max_hours=1,
            state_of_charge_initial=initial["low"],
            state_of_charge_initial_per_period=per_period,
        )
    else:
        var, initial_attr = "Store-e", "e_initial"
        n.add(
            "Store",
            "storage",
            bus="bus",
            carrier="gas",
            e_nom=10,
            e_initial=initial["low"],
            e_initial_per_period=per_period,
        )
    n.add(
        "GlobalConstraint",
        "limit",
        type=kind,
        carrier_attribute="co2_emissions" if kind == "primary_energy" else "gas",
        sense="<=",
        constant=constant,
    )
    if mode == "stochastic_per_period":
        n.set_scenarios({"low": 0.5, "high": 0.5})
        static = n.c[component].static
        static.loc[("high", "storage"), initial_attr] = initial["high"]
        scenarios = ["low", "high"]
    else:
        scenarios = [None]

    m = n.optimize.create_model(multi_investment_periods=multi_invest)

    factor = emission if kind == "primary_energy" else 1.0
    if per_period:
        weights = {(period, 1): w for period, w in years.items()}
    elif multi_invest:
        weights = {(2040, 1): 1.0}
    else:
        weights = {2: 1.0}
    expected = {}
    for scenario in scenarios:
        level = initial[scenario or "low"]
        terms = {(var, "storage", sn): -factor * w for sn, w in weights.items()}
        rhs = constant - factor * level * sum(weights.values())
        expected[scenario] = (terms, "<=", rhs)

    rows = _constraint_rows(m, "GlobalConstraint-limit")
    assert rows.keys() == expected.keys()
    for scenario, (terms, sign, rhs) in expected.items():
        got_terms, got_sign, got_rhs = rows[scenario]
        assert got_terms.keys() == terms.keys()
        for key, coeff in terms.items():
            assert got_terms[key] == pytest.approx(coeff)
        assert got_sign == sign
        assert got_rhs == pytest.approx(rhs)
