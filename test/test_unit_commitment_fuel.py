# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the unit-commitment fuel attributes stand_by_fuel and start_up_fuel."""

from typing import Any

import pandas as pd
import pytest

import pypsa

ATTRS = ["stand_by_fuel", "start_up_fuel"]


def _make_network(component: str, **kwargs) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "b0")
    n.add("Bus", "b1")
    if component == "Generator":
        n.add("Generator", "c", bus="b0", **kwargs)
    else:
        n.add(component, "c", bus0="b0", bus1="b1", **kwargs)
    return n


@pytest.mark.parametrize("component", ["Generator", "Link", "Process"])
def test_fuel_attributes_default(component):
    n = _make_network(component)
    c = n.components[component]
    for attr in ATTRS:
        assert c.static.loc["c", attr] == 0


@pytest.mark.parametrize("component", ["Generator", "Link", "Process"])
def test_fuel_attributes_scalar_round_trip(component):
    n = _make_network(component, stand_by_fuel=1.5, start_up_fuel=7.0)
    c = n.components[component]
    assert c.static.loc["c", "stand_by_fuel"] == 1.5
    assert c.static.loc["c", "start_up_fuel"] == 7.0


@pytest.mark.parametrize("component", ["Generator", "Link", "Process"])
def test_fuel_attributes_series_round_trip(component):
    sb = pd.Series([1.0, 2.0, 3.0], index=n_snaps())
    su = pd.Series([4.0, 5.0, 6.0], index=n_snaps())
    n = _make_network(component, stand_by_fuel=sb, start_up_fuel=su)
    c = n.components[component]
    pd.testing.assert_series_equal(c.dynamic.stand_by_fuel["c"], sb, check_names=False)
    pd.testing.assert_series_equal(c.dynamic.start_up_fuel["c"], su, check_names=False)


def n_snaps():
    return pd.Index(range(3), name="snapshot")


def _gas_network(
    component: str = "Link", load=(0, 80, 80, 0), weight: float = 1.0, **kwargs
) -> pypsa.Network:
    """Gas generator feeding a committable plant that serves a load."""
    n = pypsa.Network()
    n.set_snapshots(range(len(load)))
    n.snapshot_weightings.loc[:, :] = weight
    n.add("Bus", "gas", carrier="gas")
    n.add("Bus", "elec", carrier="elec")
    n.add("Generator", "supply", bus="gas", p_nom=1000, marginal_cost=1)
    n.add("Load", "load", bus="elec", p_set=list(load))
    plant: dict[str, Any] = {
        "bus0": "gas",
        "bus1": "elec",
        "p_nom": 100,
        "p_min_pu": 0.4,
        "committable": True,
        "up_time_before": 0,
    }
    if component == "Link":
        plant["efficiency"] = 1.0
    else:
        plant["rate0"] = -1.0
        plant["rate1"] = 1.0
    plant.update(kwargs)
    n.add(component, "plant", **plant)
    return n


@pytest.mark.parametrize("component", ["Link", "Process"])
def test_stand_by_fuel_draws_from_bus0(component):
    n = _gas_network(component, stand_by_fuel=3.0)
    n.optimize()
    c = n.components[component]
    p0 = c.dynamic.p0["plant"]
    assert p0.tolist() == pytest.approx([0, 83, 83, 0])
    assert n.c.generators.dynamic.p["supply"].tolist() == pytest.approx([0, 83, 83, 0])
    assert (p0 - c.dynamic.p["plant"]).tolist() == pytest.approx([0, 3, 3, 0])


@pytest.mark.parametrize("component", ["Link", "Process"])
def test_start_up_fuel_divided_by_weighting(component):
    n = _gas_network(component, weight=2.0, start_up_fuel=10.0)
    n.optimize()
    c = n.components[component]
    p0 = c.dynamic.p0["plant"]
    assert p0.tolist() == pytest.approx([0, 85, 80, 0])
    assert n.c.generators.dynamic.p["supply"].tolist() == pytest.approx([0, 85, 80, 0])


def test_p0_minus_p_equals_fuel_terms():
    n = _gas_network(weight=2.0, stand_by_fuel=3.0, start_up_fuel=10.0)
    n.optimize()
    c = n.c.links
    diff = c.dynamic.p0["plant"] - c.dynamic.p["plant"]
    expected = 3.0 * c.dynamic.status["plant"] + 5.0 * c.dynamic.start_up["plant"]
    assert diff.tolist() == pytest.approx(expected.tolist())
    assert diff.tolist() == pytest.approx([0, 8, 3, 0])


def test_energy_balance_closes_with_fuel():
    n = _gas_network(weight=2.0, stand_by_fuel=3.0, start_up_fuel=10.0)
    n.optimize()
    eb = n.statistics.energy_balance(groupby=["carrier", "bus_carrier"])
    assert eb.groupby("bus_carrier").sum().abs().max() < 1e-4


def test_modular_link_scales_stand_by_fuel_with_modules():
    n = _gas_network(
        load=(100, 100, 100),
        p_nom_extendable=True,
        p_nom_mod=50,
        p_nom_min=100,
        p_nom_max=100,
        p_min_pu=0.0,
        stand_by_fuel=3.0,
    )
    n.optimize()
    c = n.c.links
    assert c.dynamic.status["plant"].tolist() == pytest.approx([2, 2, 2])
    diff = c.dynamic.p0["plant"] - c.dynamic.p["plant"]
    assert diff.tolist() == pytest.approx([6, 6, 6])


def test_rolling_horizon_start_up_fuel_burns_once():
    load = (0, 0, 80, 80, 0)
    n = _gas_network(load=load, stand_by_fuel=3.0, start_up_fuel=10.0)
    n.optimize()
    full = n.c.links.dynamic.p0["plant"].copy()
    assert full.sum() == pytest.approx(160 + 2 * 3 + 10)

    n_rh = _gas_network(load=load, stand_by_fuel=3.0, start_up_fuel=10.0)
    n_rh.optimize.optimize_with_rolling_horizon(horizon=2, overlap=1)
    rh = n_rh.c.links.dynamic.p0["plant"]
    assert rh.tolist() == pytest.approx(full.tolist())
    assert n_rh.c.generators.dynamic.p["supply"].sum() == pytest.approx(
        160 + 2 * 3 + 10
    )


def test_rolling_horizon_ramp_seam_ignores_fuel():
    load = (0, 80, 80, 80, 40, 40)

    def build():
        n = _gas_network(
            load=load,
            p_min_pu=0,
            stand_by_fuel=20.0,
            ramp_limit_down=0.4,
        )
        n.add("Generator", "backup", bus="elec", p_nom=1000, marginal_cost=100)
        return n

    n = build()
    n.optimize()
    n_rh = build()
    n_rh.optimize.optimize_with_rolling_horizon(horizon=4)
    assert n.c.links.dynamic.p["plant"].tolist() == pytest.approx(
        [0, 80, 80, 80, 40, 40]
    )
    assert n_rh.c.links.dynamic.p["plant"].tolist() == pytest.approx(
        n.c.links.dynamic.p["plant"].tolist()
    )
    assert n_rh.c.generators.dynamic.p["backup"].sum() == 0


def test_fix_optimal_dispatch_excludes_fuel():
    n = _gas_network(stand_by_fuel=5.0)
    n.optimize()
    p = n.c.links.dynamic.p["plant"].copy()
    assert (n.c.links.dynamic.p0["plant"] - p).abs().sum() > 0
    n.optimize.fix_optimal_dispatch()
    pd.testing.assert_series_equal(
        n.c.links.dynamic.p_set["plant"], p, check_names=False
    )


def _co2_network(
    limit: float, load=(0, 80, 80, 0), weight: float = 1.0, **kwargs
) -> pypsa.Network:
    """Committable gas Generator and a clean backup under a CO2 limit."""
    n = pypsa.Network()
    n.set_snapshots(range(len(load)))
    n.snapshot_weightings.loc[:, :] = weight
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add("Carrier", "clean")
    n.add("Bus", "elec")
    n.add("Load", "load", bus="elec", p_set=list(load))
    gas: dict[str, Any] = {
        "bus": "elec",
        "carrier": "gas",
        "p_nom": 100,
        "p_min_pu": 0.4,
        "efficiency": 0.5,
        "marginal_cost": 1,
        "committable": True,
        "up_time_before": 0,
        "start_up_cost": 0.01,
    }
    gas.update(kwargs)
    n.add("Generator", "gas", **gas)
    n.add("Generator", "backup", bus="elec", carrier="clean", p_nom=1000)
    n.c.generators.static.loc["backup", "marginal_cost"] = 100
    n.add(
        "GlobalConstraint",
        "co2",
        sense="<=",
        constant=limit,
        type="primary_energy",
        carrier_attribute="co2_emissions",
    )
    return n


def test_generator_fuel_counts_in_primary_energy_limit():
    n = _co2_network(limit=100, stand_by_fuel=3.0, start_up_fuel=10.0)
    n.optimize()
    assert n.c.generators.dynamic.p["gas"].tolist() == pytest.approx([0, 80, 80, 0])
    activity = n.model.constraints["GlobalConstraint-co2"].lhs.solution.item()
    assert activity == pytest.approx((320 + 2 * 3 + 10) * 0.2)


def test_generator_fuel_shifts_dispatch_to_clean_unit():
    n = _co2_network(limit=66)
    n.optimize()
    assert n.c.generators.dynamic.p["backup"].sum() == pytest.approx(0)

    n = _co2_network(limit=66, stand_by_fuel=3.0, start_up_fuel=10.0)
    n.optimize()
    assert n.c.generators.dynamic.p["backup"].sum() > 0
    activity = n.model.constraints["GlobalConstraint-co2"].lhs.solution.item()
    assert activity <= 66 + 1e-4


def test_generator_start_up_fuel_multi_invest_weighted_by_years_only():
    n = _co2_network(limit=1e5, stand_by_fuel=3.0, start_up_fuel=10.0)
    n.set_snapshots(range(3))
    n.snapshot_weightings.loc[:, :] = 2.0
    n.investment_periods = [2020, 2030]
    n.investment_period_weightings.loc[:, "years"] = 10
    n.loads_t.p_set = pd.DataFrame({"load": [0, 80, 80] * 2}, index=n.snapshots)
    n.optimize(multi_investment_periods=True)
    activity = n.model.constraints["GlobalConstraint-co2"].lhs.solution.item()
    per_period = 2 * (320 + 2 * 3) + 10
    assert activity == pytest.approx(2 * 10 * per_period * 0.2)


def test_generator_fuel_counts_per_scenario():
    n = _co2_network(limit=1000, stand_by_fuel=3.0, start_up_fuel=10.0)
    n.set_scenarios({"a": 0.5, "b": 0.5})
    n.c.generators.static.loc[("b", "gas"), ["stand_by_fuel", "start_up_fuel"]] = 0
    n.optimize()
    activity = n.model.constraints["GlobalConstraint-co2"].lhs.solution
    assert sorted(activity.values.ravel().tolist()) == pytest.approx([64.0, 67.2])


def test_link_fuel_per_scenario():
    n = _gas_network(stand_by_fuel=3.0, start_up_fuel=10.0)
    n.set_scenarios({"a": 0.5, "b": 0.5})
    n.c.links.static.loc[("b", "plant"), ["stand_by_fuel", "start_up_fuel"]] = 0
    n.optimize()
    c = n.c.links
    diff = c.dynamic.p0 - c.dynamic.p
    assert diff[("a", "plant")].tolist() == pytest.approx([0, 13, 3, 0])
    assert diff[("b", "plant")].tolist() == pytest.approx([0, 0, 0, 0])
