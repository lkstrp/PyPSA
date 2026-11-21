# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Model-build tests of the primary-energy and operational-limit global constraints.

The tests build the model without solving and read back the constraint's dims,
sign, right-hand side, scaling and the summed coefficient of each variable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pypsa

PERIODS = [2020, 2030]
YEARS = [10.0, 20.0]
EMISSIONS = {"gas": 0.2, "coal": 0.3}
EFFICIENCY = {"gas": 0.5, "coal": 0.4}


@dataclass
class Pinned:
    dims: tuple[str, ...]
    sign: str
    rhs: float
    scaling: float
    coeffs: dict[tuple[str, str, Any], float]


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    return value.item() if isinstance(value, np.generic) else value


def pinned(m: Any, name: str, **sel: Any) -> Pinned:
    """Read `GlobalConstraint-<name>` back as dims, sign, rhs and coefficients."""
    con = m.constraints[f"GlobalConstraint-{name}"]
    dims = tuple(d for d in con.vars.dims if d != "_term")
    labels = con.vars.sel(**sel).values.ravel()
    values = con.coeffs.sel(**sel).values.ravel()
    coeffs: dict[tuple[str, str, Any], float] = defaultdict(float)
    for label, coeff in zip(labels, values, strict=True):
        if label == -1:
            continue
        var, coords = m.variables.get_label_position(int(label))
        coeffs[var, coords["name"], _plain(coords["snapshot"])] += float(coeff)
    return Pinned(
        dims=dims,
        sign=str(con.sign.sel(**sel).item()),
        rhs=float(con.rhs.sel(**sel).item()),
        scaling=float(con.scaling.sel(**sel).item()),
        coeffs=dict(coeffs),
    )


def network(*, multi: bool = False, gas_efficiency: Any = None) -> pypsa.Network:
    n = pypsa.Network()
    if multi:
        n.snapshots = pd.MultiIndex.from_product([PERIODS, [0, 1]])
        n.investment_periods = PERIODS
        n.investment_period_weightings.loc[:, "years"] = YEARS
    else:
        n.set_snapshots(range(3))
    n.snapshot_weightings.loc[:, "generators"] = np.arange(1.0, len(n.snapshots) + 1)
    n.add("Bus", "b", carrier="AC")
    n.add("Carrier", ["AC", "wind"])
    for carrier, emission in EMISSIONS.items():
        n.add("Carrier", carrier, co2_emissions=emission)
        n.add(
            "Generator",
            carrier,
            bus="b",
            carrier=carrier,
            p_nom=10,
            efficiency=gas_efficiency
            if carrier == "gas" and gas_efficiency is not None
            else EFFICIENCY[carrier],
            marginal_cost=1,
        )
    n.add("Generator", "wind", bus="b", carrier="wind", p_nom=10)
    n.add("Load", "load", bus="b", p_set=5)
    return n


def add_limits(n: pypsa.Network, **kwargs: Any) -> None:
    n.add(
        "GlobalConstraint",
        "pe",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=100,
        **kwargs,
    )
    n.add(
        "GlobalConstraint",
        "ol",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=50,
        **kwargs,
    )


def build(n: pypsa.Network, **kwargs: Any) -> Any:
    return n.optimize.create_model(
        include_objective_constant=False,
        multi_investment_periods=n.has_periods,
        **kwargs,
    )


def weights(n: pypsa.Network, snapshots: pd.Index | None = None) -> pd.Series:
    """Generator snapshot weighting times the years of each snapshot's period."""
    w = n.snapshot_weightings.generators
    if n.has_periods:
        years = n.investment_period_weightings.years
        w = w * years.reindex(w.index.get_level_values(0)).to_numpy()
    return w if snapshots is None else w.loc[snapshots]


def expected_pe(n: pypsa.Network, snapshots: pd.Index | None = None) -> dict:
    return {
        ("Generator-p", gen, _plain(sn)): w * EMISSIONS[gen] / EFFICIENCY[gen]
        for gen in EMISSIONS
        for sn, w in weights(n, snapshots).items()
    }


def expected_ol(n: pypsa.Network, snapshots: pd.Index | None = None) -> dict:
    return {
        ("Generator-p", "gas", _plain(sn)): w for sn, w in weights(n, snapshots).items()
    }


def assert_coeffs(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value), key


def test_single_period():
    n = network()
    add_limits(n)
    m = build(n)

    pe = pinned(m, "pe")
    assert (pe.dims, pe.sign, pe.rhs, pe.scaling) == ((), "<=", 100, 1)
    assert_coeffs(pe.coeffs, expected_pe(n))

    ol = pinned(m, "ol")
    assert (ol.dims, ol.sign, ol.rhs, ol.scaling) == ((), "<=", 50, 1)
    assert_coeffs(ol.coeffs, expected_ol(n))


def test_multi_period_without_investment_period():
    n = network(multi=True)
    add_limits(n)
    m = build(n)

    assert_coeffs(pinned(m, "pe").coeffs, expected_pe(n))
    assert_coeffs(pinned(m, "ol").coeffs, expected_ol(n))


def test_multi_period_with_investment_period_in_window():
    n = network(multi=True)
    add_limits(n, investment_period=2030)
    m = build(n)

    period_sns = n.snapshots[n.snapshots.get_level_values(0) == 2030]
    assert_coeffs(pinned(m, "pe").coeffs, expected_pe(n, period_sns))
    assert_coeffs(pinned(m, "ol").coeffs, expected_ol(n, period_sns))


def test_multi_period_with_investment_period_outside_window():
    n = network(multi=True)
    add_limits(n, investment_period=2030)
    m = build(n, snapshots=n.snapshots[n.snapshots.get_level_values(0) == 2020])

    assert "GlobalConstraint-pe" not in m.constraints
    assert "GlobalConstraint-ol" not in m.constraints


@pytest.mark.parametrize("name", ["pe", "ol"])
def test_investment_period_without_periods_raises(name):
    n = network()
    add_limits(n)
    n.c.global_constraints.static.loc[name, "investment_period"] = 2030

    with pytest.raises(ValueError, match="network has no investment periods"):
        build(n)


def test_weighting_uses_years_of_each_period():
    n = network(multi=True)
    add_limits(n)
    m = build(n)

    w = n.snapshot_weightings.generators
    for (_, gen, sn), coeff in pinned(m, "pe").coeffs.items():
        years = YEARS[PERIODS.index(sn[0])]
        expected = w.loc[sn] * years * EMISSIONS[gen] / EFFICIENCY[gen]
        assert coeff == pytest.approx(expected)
    for (_, _, sn), coeff in pinned(m, "ol").coeffs.items():
        assert coeff == pytest.approx(w.loc[sn] * YEARS[PERIODS.index(sn[0])])


def test_stochastic_per_scenario_emissions():
    n = network()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.carriers.static.loc[("high", "gas"), "co2_emissions"] = 0.4
    add_limits(n)
    m = build(n)

    for scenario, gas_emission in [("low", 0.2), ("high", 0.4)]:
        pe = pinned(m, "pe", scenario=scenario)
        assert pe.dims == ("scenario",)
        assert pe.rhs == 100
        expected = expected_pe(n)
        for key in expected:
            if key[1] == "gas":
                expected[key] *= gas_emission / EMISSIONS["gas"]
        assert_coeffs(pe.coeffs, expected)

        ol = pinned(m, "ol", scenario=scenario)
        assert ol.dims == ("scenario",)
        assert_coeffs(ol.coeffs, expected_ol(n))


def test_stochastic_zero_emissions_in_one_scenario_raises():
    """Today's outcome, the zero filter drops the scenario before its lookup."""
    n = network()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.carriers.static.loc["high", "co2_emissions"] = 0.0
    add_limits(n)

    with pytest.raises(KeyError, match="high"):
        build(n)


def test_storage_withdrawal_enters_both_constraints():
    n = network()
    n.add(
        "StorageUnit", "su", bus="b", carrier="gas", p_nom=1, state_of_charge_initial=10
    )
    n.add("Store", "st", bus="b", carrier="gas", e_nom=10, e_initial=5)
    add_limits(n)
    m = build(n)
    last = _plain(n.snapshots[-1])

    pe = pinned(m, "pe")
    expected = expected_pe(n)
    expected["StorageUnit-state_of_charge", "su", last] = -EMISSIONS["gas"]
    expected["Store-e", "st", last] = -EMISSIONS["gas"]
    assert_coeffs(pe.coeffs, expected)
    assert pe.rhs == pytest.approx(100 - EMISSIONS["gas"] * (10 + 5))

    ol = pinned(m, "ol")
    expected = expected_ol(n)
    expected["StorageUnit-state_of_charge", "su", last] = -1.0
    expected["Store-e", "st", last] = -1.0
    assert_coeffs(ol.coeffs, expected)
    assert ol.rhs == pytest.approx(50 - (10 + 5))


def test_piecewise_efficiency_uses_aux_variable():
    n = network(gas_efficiency={0.0: 0.0, 0.5: 0.4, 1.0: 0.5})
    add_limits(n)
    m = build(n)

    expected = {key: v for key, v in expected_pe(n).items() if key[1] != "gas"}
    for sn, w in weights(n).items():
        expected["Generator-p_primary_piecewise", "gas", _plain(sn)] = (
            w * EMISSIONS["gas"]
        )
    assert_coeffs(pinned(m, "pe").coeffs, expected)
    assert_coeffs(pinned(m, "ol").coeffs, expected_ol(n))


def test_zero_emissions_omit_primary_energy():
    n = network()
    n.c.carriers.static["co2_emissions"] = 0.0
    add_limits(n)
    m = build(n)

    assert "GlobalConstraint-pe" not in m.constraints
    assert "GlobalConstraint-ol" in m.constraints


def test_operational_limit_without_assets_is_omitted():
    n = network()
    n.add("Carrier", "nuclear")
    n.add(
        "GlobalConstraint",
        "ol",
        type="operational_limit",
        carrier_attribute="nuclear",
        sense="<=",
        constant=50,
    )
    m = build(n)

    assert "GlobalConstraint-ol" not in m.constraints


def test_scaling_is_applied():
    n = network()
    add_limits(n, scaling=1e-3)
    m = build(n)

    for name, rhs, expected in [
        ("pe", 100, expected_pe(n)),
        ("ol", 50, expected_ol(n)),
    ]:
        con = pinned(m, name)
        assert con.scaling == pytest.approx(1e-3)
        assert con.rhs == rhs
        assert_coeffs(con.coeffs, expected)


@pytest.mark.parametrize("multi", [False, True])
def test_primary_energy_with_unit_factor_equals_operational_limit(multi):
    n = network(multi=multi)
    n.c.carriers.static["unit"] = 0.0
    n.add("Store", "st", bus="b", carrier="gas", e_nom=10, e_initial=5, e_cyclic=False)
    n.c.carriers.static.loc["gas", "unit"] = 1.0
    n.c.generators.static.loc["gas", "efficiency"] = 1.0
    n.add(
        "GlobalConstraint",
        "pe",
        type="primary_energy",
        carrier_attribute="unit",
        sense="<=",
        constant=40,
    )
    n.add(
        "GlobalConstraint",
        "ol",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=40,
    )
    if multi:
        n.c.stores.static["e_initial_per_period"] = True
    m = build(n)

    pe, ol = pinned(m, "pe"), pinned(m, "ol")
    assert (pe.dims, pe.sign) == (ol.dims, ol.sign)
    assert pe.rhs == pytest.approx(ol.rhs)
    assert pe.coeffs.keys() == ol.coeffs.keys()
    for key, value in ol.coeffs.items():
        assert pe.coeffs[key] == pytest.approx(value), key
