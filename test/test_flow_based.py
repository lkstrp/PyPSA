# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for flow-based market coupling on `flow_based` global constraints."""

import numpy as np
import pandas as pd
import pytest

import pypsa

CNECS = ["AB (+)", "AB (-)", "BC (+)", "BC (-)", "AC (+)", "AC (-)"]

# Equal-reactance triangle A-B-C with C as reference, signs flipped for (-).
_PTDF_POS = {
    "AB": {"A": 1 / 3, "B": -1 / 3, "C": 0.0},
    "BC": {"A": 1 / 3, "B": 2 / 3, "C": 0.0},
    "AC": {"A": 2 / 3, "B": 1 / 3, "C": 0.0},
}
_RAM = {"AB": 1000.0, "BC": 1500.0, "AC": 2000.0}


def triangle_domain() -> pd.DataFrame:
    """Six-CNEC triangle domain, PTDF columns A, B, C plus RAM."""
    rows = {}
    for line, ptdf in _PTDF_POS.items():
        rows[f"{line} (+)"] = {**ptdf, "RAM": _RAM[line]}
        rows[f"{line} (-)"] = {**{z: -v for z, v in ptdf.items()}, "RAM": _RAM[line]}
    return pd.DataFrame.from_dict(rows, orient="index").loc[CNECS]


def zone_network(snapshots=range(3)) -> pypsa.Network:
    n = pypsa.Network(snapshots=snapshots)
    n.add("Bus", ["A", "B", "C"])
    return n


# Data model


def test_defaults():
    c = zone_network().c.global_constraints
    assert np.isnan(c.defaults.loc["ram", "default"])
    assert "mu_ram" in c.static
    c.add("x", type="operational_limit")
    assert c.static.loc["x", "mu_ram"] == 0


def test_bulk_static_add():
    n = zone_network()
    c = n.c.global_constraints
    domain = triangle_domain()
    # Drop C on purpose: a missing zone column reads 0.
    c.add_flow_based(domain[["A", "B"]], domain["RAM"])

    assert (c.static.loc[CNECS, "type"] == "flow_based").all()
    assert (c.static.loc[CNECS, "sense"] == "<=").all()
    pd.testing.assert_series_equal(
        c.static.loc[CNECS, "ram"],
        domain["RAM"],
        check_names=False,
        check_index_type=False,
    )
    assert {"ptdf_A", "ptdf_B"} <= set(c.static.columns)
    pd.testing.assert_frame_equal(
        c.zonal_ptdf,
        domain[["A", "B"]],
        check_names=False,
        check_index_type=False,
        check_column_type=False,
    )
    assert c.da.zonal_ptdf.dims == ("name", "bus")

    c.add("other", type="flow_based", ram=100, ptdf_C=1.0)
    assert c.zonal_ptdf.loc["AB (+)", "C"] == 0
    assert c.zonal_ptdf.loc["other", "A"] == 0
    assert c.zonal_ptdf.loc["other", "C"] == 1


def test_single_cnec_series():
    n = zone_network()
    c = n.c.global_constraints
    c.add_flow_based(pd.Series({"A": 0.5, "B": -0.5}, name="AB (+)"), ram=800)
    assert c.static.loc["AB (+)", "ram"] == 800
    assert c.zonal_ptdf.loc["AB (+)"].to_dict() == {"A": 0.5, "B": -0.5}


def test_time_varying_add():
    n = zone_network()
    c = n.c.global_constraints
    domain = triangle_domain()[["A", "B", "C"]]
    frames = {sn: domain * (1 + sn) for sn in n.snapshots}
    tv = pd.concat(frames, names=["snapshot", "name"])
    c.add_flow_based(tv, 1000)

    assert list(c.dynamic["ptdf_A"].columns) == CNECS
    assert c.dynamic["ptdf_A"].loc[2, "AB (+)"] == pytest.approx(1.0)
    z = c.zonal_ptdf
    assert z.index.names == ["snapshot", "name"]
    assert z.loc[(1, "AC (+)"), "A"] == pytest.approx(4 / 3)
    assert c.da.zonal_ptdf.dims == ("snapshot", "name", "bus")


def test_incremental_add_static():
    n = zone_network()
    c = n.c.global_constraints
    c.add_flow_based(pd.Series({"A": 0.5}, name="r1"), ram=1)
    c.add_flow_based(pd.Series({"B": 0.25}, name="r2"), ram=2)
    z = c.zonal_ptdf
    assert list(z.columns) == ["A", "B"]
    assert z.loc["r1", "B"] == 0
    assert z.loc["r2", "A"] == 0
    assert z.loc["r2", "B"] == 0.25


def test_incremental_add_dynamic():
    n = zone_network()
    c = n.c.global_constraints
    for cnec, zone in [("r1", "A"), ("r2", "B")]:
        tv = pd.DataFrame(
            {zone: [1.0, 2.0, 3.0]},
            index=pd.MultiIndex.from_product(
                [n.snapshots, [cnec]], names=["snapshot", "name"]
            ),
        )
        c.add_flow_based(tv, 1)
    assert list(c.dynamic["ptdf_A"].columns) == ["r1"]
    assert list(c.dynamic["ptdf_B"].columns) == ["r2"]
    z = c.zonal_ptdf
    assert z.loc[(2, "r1"), "A"] == 3
    assert z.loc[(2, "r1"), "B"] == 0
    assert z.loc[(1, "r2"), "B"] == 2
    assert z.loc[(1, "r2"), "A"] == 0


def test_overwrite_replaces_row():
    n = zone_network()
    c = n.c.global_constraints
    c.add_flow_based(pd.Series({"A": 0.5, "B": -0.5}, name="AB (+)"), ram=1000)
    c.add_flow_based(pd.Series({"A": 0.25}, name="AC (+)"), ram=900)
    c.add_flow_based(pd.Series({"C": 0.75}, name="AB (+)"), ram=1500, overwrite=True)
    assert len(c.static) == 2
    assert c.static.loc["AB (+)", "ram"] == 1500
    assert c.zonal_ptdf.loc["AB (+)"].to_dict() == {"A": 0, "B": 0, "C": 0.75}


def test_add_flow_based_validation():
    n = zone_network()
    c = n.c.global_constraints
    domain = triangle_domain()
    with pytest.raises(ValueError, match="misses the CNECs"):
        c.add_flow_based(domain[["A", "B"]], domain["RAM"].iloc[1:])
    ram = pd.DataFrame(1000.0, index=range(5), columns=CNECS)
    with pytest.raises(ValueError, match="must equal `n.snapshots`"):
        c.add_flow_based(domain[["A", "B"]], ram)
    tv = pd.concat({sn: domain[["A"]] for sn in range(5)}, names=["snapshot", "name"])
    with pytest.raises(ValueError, match="must equal `n.snapshots`"):
        c.add_flow_based(tv, 1000)
    with pytest.raises(ValueError, match="non-empty string"):
        c.add_flow_based(pd.Series({"A": 0.5}), ram=1)
    assert c.static.empty


def test_zonal_ptdf_excludes_other_types():
    n = zone_network()
    c = n.c.global_constraints
    c.add("limit", type="operational_limit", ptdf_A=7.0)
    assert c.da.zonal_ptdf.sizes["name"] == 0
    c.add_flow_based(pd.Series({"A": 0.5}, name="r1"), ram=1)
    assert list(c.da.zonal_ptdf.indexes["name"]) == ["r1"]


def _assert_domain_equal(n, m):
    a, b = n.c.global_constraints, m.c.global_constraints
    pd.testing.assert_frame_equal(
        a.zonal_ptdf,
        b.zonal_ptdf,
        check_names=False,
        check_index_type=False,
        check_column_type=False,
    )
    pd.testing.assert_series_equal(
        a.static["ram"],
        b.static["ram"],
        check_names=False,
        check_index_type=False,
    )


def _mixed_domain_network() -> pypsa.Network:
    n = zone_network()
    domain = triangle_domain()
    n.c.global_constraints.add_flow_based(domain[["A", "B"]], domain["RAM"])
    tv = pd.concat(
        {
            sn: domain[["C"]].iloc[:2].add_suffix(" t", axis=0) + sn
            for sn in n.snapshots
        },
        names=["snapshot", "name"],
    )
    n.c.global_constraints.add_flow_based(tv, 300)
    return n


def test_netcdf_round_trip(tmp_path):
    n = _mixed_domain_network()
    path = tmp_path / "n.nc"
    n.export_to_netcdf(path)
    _assert_domain_equal(n, pypsa.Network(path))


def test_csv_round_trip(tmp_path):
    n = _mixed_domain_network()
    n.export_to_csv_folder(tmp_path / "csv")
    _assert_domain_equal(n, pypsa.Network(tmp_path / "csv"))


def test_copy_keeps_domain():
    n = _mixed_domain_network()
    _assert_domain_equal(n, n.copy())
    m = n.copy(snapshots=n.snapshots[:2])
    a = n.c.global_constraints.zonal_ptdf
    pd.testing.assert_frame_equal(
        m.c.global_constraints.zonal_ptdf,
        a.loc[a.index.get_level_values("snapshot") < 2],
        check_names=False,
    )


# Optimisation


def triangle(ram_overrides=None, snapshots=1, domain=True) -> pypsa.Network:
    """Triangle toy: cheap A exports to B and C, limited by the `AB (+)` CNEC."""
    n = zone_network(snapshots=range(snapshots))
    n.add("Load", ["load A", "load B", "load C"], bus=["A", "B", "C"])
    n.c.loads.static["p_set"] = [500.0, 1500.0, 1000.0]
    n.add(
        "Generator",
        ["gen A", "gen B", "gen C"],
        bus=["A", "B", "C"],
        p_nom=4000,
        marginal_cost=[10, 80, 50],
    )
    if domain:
        d = triangle_domain()
        for cnec, value in (ram_overrides or {}).items():
            d.loc[cnec, "RAM"] = value
        n.c.global_constraints.add_flow_based(d[["A", "B", "C"]], d["RAM"])
    return n


def net_position(n) -> pd.DataFrame:
    return n.model["FlowBased-net_position"].solution.to_pandas()


def test_triangle_optimum():
    n = triangle()
    status, _ = n.optimize()
    assert status == "ok"
    np.testing.assert_allclose(net_position(n).iloc[0], [2000, -1000, -1000])
    np.testing.assert_allclose(n.buses_t.marginal_price.iloc[0], [10, 80, 45])
    assert n.objective == pytest.approx(65000)
    mu = n.global_constraints_t.mu_ram.iloc[0]
    assert mu["AB (+)"] == pytest.approx(-105)
    np.testing.assert_allclose(mu.drop("AB (+)"), 0, atol=1e-8)


def test_triangle_one_at_a_time():
    n = triangle(domain=False)
    d = triangle_domain()
    for cnec in CNECS:
        n.c.global_constraints.add_flow_based(
            d.loc[cnec, ["A", "B", "C"]], d.loc[cnec, "RAM"]
        )
    n.optimize()
    np.testing.assert_allclose(net_position(n).iloc[0], [2000, -1000, -1000])
    np.testing.assert_allclose(n.buses_t.marginal_price.iloc[0], [10, 80, 45])
    assert n.global_constraints_t.mu_ram.iloc[0]["AB (+)"] == pytest.approx(-105)


def test_lower_ram_shifts_optimum():
    n = triangle({"AB (+)": 500})
    n.optimize()
    assert net_position(n).iloc[0]["A"] < 2000 - 1
    assert n.objective > 65000 + 1


def test_kkt_price_spread():
    n = triangle({"AB (+)": 800, "AC (+)": 900})
    n.optimize()
    price = n.buses_t.marginal_price.iloc[0]
    mu = n.global_constraints_t.mu_ram.iloc[0]
    ptdf = n.c.global_constraints.zonal_ptdf
    assert (mu <= 1e-8).all()
    assert (mu < -1e-6).any()
    for z1, z2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        spread = (mu[ptdf.index] * (ptdf[z1] - ptdf[z2])).sum()
        assert price[z1] - price[z2] == pytest.approx(spread)


def test_no_junk_outputs():
    n = triangle()
    names = n.c.global_constraints.static.index.copy()
    before = set(n.c.global_constraints.dynamic.keys())
    n.optimize()
    for c in n.components:
        assert "net_position" not in c.dynamic
        assert "net_position" not in c.static
    assert set(n.c.global_constraints.dynamic.keys()) == before
    assert list(n.c.global_constraints.dynamic["mu_ram"].columns) == CNECS
    assert n.c.global_constraints.static.index.equals(names)
    assert "FlowBased-balance" in n.model.constraints
    assert not any("balance" in k for k in n.c.global_constraints.dynamic.keys())


@pytest.mark.parametrize(
    ("component", "kwargs"),
    [
        ("Line", {"x": 0.1, "s_nom": 100}),
        ("Transformer", {"x": 0.1, "s_nom": 100}),
        ("Link", {"p_nom": 100}),
    ],
)
def test_cross_zone_branch_raises(component, kwargs):
    n = triangle()
    n.add(component, "AB", bus0="A", bus1="B", **kwargs)
    with pytest.raises(ValueError, match="connect two zone buses"):
        n.optimize.create_model()


def test_link_column_without_zone_raises():
    n = triangle()
    n.add("Bus", ["D", "E"])
    n.add("Link", "DE", bus0="D", bus1="E", p_nom=100)
    n.add("GlobalConstraint", "DE (+)", type="flow_based", ram=100, ptdf_DE=1.0)
    with pytest.raises(ValueError, match="touch no zone bus"):
        n.optimize.create_model()


def test_unknown_column_raises():
    n = triangle()
    n.add("GlobalConstraint", "Z (+)", type="flow_based", ram=100, ptdf_Z=1.0)
    with pytest.raises(ValueError, match="Unknown PTDF columns"):
        n.optimize.create_model()


def test_bus_link_clash_raises():
    n = triangle()
    n.add("Bus", "D")
    n.add("Link", "A", bus0="C", bus1="D", p_nom=100)
    with pytest.raises(ValueError, match="both a Bus and a Link"):
        n.optimize.create_model()


def test_isolated_zone_raises():
    n = triangle()
    n.add("Bus", "D")
    n.add("GlobalConstraint", "D (+)", type="flow_based", ram=100, ptdf_D=1.0)
    with pytest.raises(ValueError, match="no attached components"):
        n.optimize.create_model()


def test_nan_ram_raises():
    n = triangle()
    n.add("GlobalConstraint", "AB (x)", type="flow_based", ptdf_A=0.1)
    with pytest.raises(ValueError, match="have no `ram`"):
        n.optimize.create_model()


def test_investment_period_without_periods_raises():
    n = triangle()
    n.c.global_constraints.static.loc["AB (+)", "investment_period"] = 2020
    with pytest.raises(ValueError, match="no investment periods"):
        n.optimize.create_model()


def test_non_zone_link_allowed():
    n = triangle()
    n.add("Bus", "D")
    n.add("Load", "load D", bus="D", p_set=100)
    n.add("Link", "CD", bus0="C", bus1="D", p_nom=1000)
    status, _ = n.optimize()
    assert status == "ok"
    np.testing.assert_allclose(n.links_t.p0["CD"], 100)


@pytest.mark.parametrize("order", [["A", "B", "C"], ["C", "A", "B"]])
def test_net_position_equals_bus_injection(order):
    n = triangle(domain=False)
    d = triangle_domain()
    n.c.global_constraints.add_flow_based(d[order], d["RAM"])
    n.optimize()
    np_ = net_position(n)
    for zone in order:
        np.testing.assert_allclose(np_[zone], n.buses_t.p[zone], atol=1e-6)
    np.testing.assert_allclose(
        np_.loc[:, ["A", "B", "C"]].iloc[0], [2000, -1000, -1000]
    )


def test_meshed_thresholds_split():
    n = triangle()
    n.optimize(meshed_thresholds=[1])
    np.testing.assert_allclose(net_position(n).iloc[0], [2000, -1000, -1000])


def test_no_domain_unchanged():
    # Without a domain and without lines, every zone supplies its own load.
    n = triangle(domain=False)
    n.optimize()
    assert n.objective == pytest.approx(175000)
    assert not any(k.startswith("FlowBased") for k in n.model.variables)
    assert not any(k.startswith("FlowBased") for k in n.model.constraints)
    assert "GlobalConstraint-ram" not in n.model.constraints


def test_copy_resolves():
    n = triangle(snapshots=2)
    for m in (n.copy(), n.copy(snapshots=n.snapshots[:1])):
        pd.testing.assert_frame_equal(
            m.c.global_constraints.zonal_ptdf, n.c.global_constraints.zonal_ptdf
        )
        m.optimize()
        np.testing.assert_allclose(net_position(m)["A"], 2000)


def test_overwrite_moves_optimum():
    n = triangle()
    d = triangle_domain()
    n.c.global_constraints.add_flow_based(
        d.loc["AB (+)", ["A", "B", "C"]], 1500, overwrite=True
    )
    n.optimize()
    assert net_position(n).iloc[0]["A"] > 2000 + 1
    assert n.objective < 65000 - 1


# Corridors, time variation, investment periods, scenarios


def test_corridor_link_binds_at_ram():
    n = triangle()
    n.add("Link", "L", bus0="A", bus1="B", p_nom=3000, p_min_pu=-1)
    c = n.c.global_constraints
    c.static["ptdf_L"] = 0.0
    corridor = pd.DataFrame(
        {"A": 0.0, "B": 0.0, "C": 0.0, "L": [1.0, -1.0]}, index=["L (+)", "L (-)"]
    )
    c.add_flow_based(corridor, 400)
    n.optimize()

    p_l = n.links_t.p0["L"].iloc[0]
    assert p_l == pytest.approx(400)
    assert n.global_constraints_t.mu_ram.iloc[0]["L (+)"] == pytest.approx(-70)
    assert n.generators_t.p["gen B"].iloc[0] == pytest.approx(100)

    ptdf = c.zonal_ptdf
    zones = ["A", "B", "C"]
    loading = ptdf[zones] @ net_position(n).iloc[0][zones] + ptdf["L"] * p_l
    assert loading["L (+)"] == pytest.approx(p_l)
    assert loading["AB (+)"] == pytest.approx(1000)


def test_advanced_hybrid_coupling_import():
    n = triangle()
    n.add("Bus", "X")
    n.add("Generator", "gen X", bus="X", p_nom=6000, marginal_cost=20)
    n.add("Link", "H", bus0="X", bus1="B", p_nom=10000)
    c = n.c.global_constraints
    # Hub sensitivity h = 0 (the reference zone's column), so ptdf_H = -PTDF_B.
    c.static["ptdf_H"] = -c.static["ptdf_B"]
    n.optimize()

    ref = triangle()
    ref.add("Generator", "gen X", bus="C", p_nom=6000, marginal_cost=20)
    ref.optimize()

    assert n.objective == pytest.approx(40000)
    assert ref.objective == pytest.approx(n.objective)
    gen = n.generators_t.p.iloc[0]
    np.testing.assert_allclose(gen[["gen A", "gen B", "gen X"]], [2000, 0, 1000])
    np.testing.assert_allclose(
        ref.generators_t.p.iloc[0][["gen A", "gen B", "gen X"]], [2000, 0, 1000]
    )

    p_h = n.links_t.p1["H"].iloc[0] * -1
    np_ = net_position(n).iloc[0]
    ptdf = c.zonal_ptdf
    zones = ["A", "B", "C"]
    loading = ptdf[zones] @ np_[zones] + ptdf["H"] * p_h
    expected = ptdf["A"] * np_["A"] + ptdf["B"] * (np_["B"] - p_h)
    pd.testing.assert_series_equal(loading, expected, check_names=False)
    assert np_["B"] == pytest.approx(gen["gen B"] - 1500 + p_h)


def test_corridor_export_enters_net_position():
    n = triangle()
    n.add("Bus", "X")
    n.add("Load", "load X", bus="X", p_set=300)
    n.add("Link", "H", bus0="A", bus1="X", p_nom=10000)
    c = n.c.global_constraints
    c.static["ptdf_H"] = c.static["ptdf_A"]
    n.optimize()
    np_ = net_position(n).iloc[0]
    gen_a = n.generators_t.p["gen A"].iloc[0]
    assert n.links_t.p0["H"].iloc[0] == pytest.approx(300)
    assert np_["A"] == pytest.approx(gen_a - 500 - 300)
    np.testing.assert_allclose(np_, n.buses_t.p.iloc[0][np_.index], atol=1e-6)


def _single_snapshot_solve(ram_ab, b_scale):
    n = triangle(domain=False)
    d = triangle_domain()
    d["B"] *= b_scale
    d.loc["AB (+)", "RAM"] = ram_ab
    n.c.global_constraints.add_flow_based(d[["A", "B", "C"]], d["RAM"])
    n.optimize()
    return net_position(n).iloc[0], n.objective


def test_time_varying_matches_static_solves():
    n = triangle(domain=False, snapshots=2)
    d = triangle_domain()
    second = d.copy()
    second["B"] *= 0.5
    tv = pd.concat({0: d[["A", "B", "C"]], 1: second[["A", "B", "C"]]})
    tv.index.names = ["snapshot", "name"]
    ram = pd.DataFrame([d["RAM"], d["RAM"]], index=n.snapshots)
    ram.loc[1, "AB (+)"] = 500
    n.c.global_constraints.add_flow_based(tv, ram)
    n.optimize()

    objective = 0
    for sn, (ram_ab, b_scale) in enumerate([(1000, 1), (500, 0.5)]):
        np_ref, obj_ref = _single_snapshot_solve(ram_ab, b_scale)
        np.testing.assert_allclose(net_position(n).iloc[sn], np_ref, atol=1e-6)
        objective += obj_ref
    assert n.objective == pytest.approx(objective)


def test_mixed_static_and_dynamic_columns():
    n = triangle(domain=False, snapshots=2)
    d = triangle_domain()
    c = n.c.global_constraints
    c.add_flow_based(d[["B", "C"]], d["RAM"])
    c.dynamic["ptdf_A"] = pd.DataFrame(
        [d["A"], d["A"]], index=n.snapshots, columns=CNECS
    )
    assert "ptdf_A" not in c.static
    n.optimize()

    ref = triangle(snapshots=2)
    ref.optimize()
    np.testing.assert_allclose(net_position(n), net_position(ref)[["B", "C", "A"]])
    assert n.objective == pytest.approx(ref.objective)


def _multi_period_triangle() -> pypsa.Network:
    n = triangle(domain=False)
    n.snapshots = pd.MultiIndex.from_product([[2020, 2030], [0]])
    n.investment_periods = [2020, 2030]
    d = triangle_domain()
    for period, ram_ab in [(2020, 1000), (2030, 500)]:
        dp = d.add_suffix(f" {period}", axis=0)
        dp.loc[f"AB (+) {period}", "RAM"] = ram_ab
        n.c.global_constraints.add_flow_based(
            dp[["A", "B", "C"]], dp["RAM"], investment_period=period
        )
    return n


def test_investment_period_selects_domain():
    n = _multi_period_triangle()
    n.optimize(multi_investment_periods=True)
    np_a = net_position(n)["A"]
    # In 2030 the tighter AB (+) binds together with gen C at zero.
    np.testing.assert_allclose(np_a, [2000, 1250])
    mu = n.global_constraints_t.mu_ram
    assert mu["AB (+) 2020"].iloc[0] == pytest.approx(-105)
    assert mu["AB (+) 2020"].iloc[1] == 0
    assert mu["AB (+) 2030"].iloc[0] == 0
    assert mu["AB (+) 2030"].iloc[1] < 0


def test_unknown_investment_period_raises():
    n = _multi_period_triangle()
    n.c.global_constraints.static.loc["AB (+) 2030", "investment_period"] = 2040
    with pytest.raises(ValueError, match="investment_period outside"):
        n.optimize.create_model(multi_investment_periods=True)


def test_stochastic_smoke():
    n = triangle()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.loads.static.loc[("high", "load B"), "p_set"] = 1600
    status, _ = n.optimize()
    assert status == "ok"
    sol = n.model["FlowBased-net_position"].solution
    assert "scenario" in sol.dims
    # `buses_t.p` is not filled on stochastic networks, so rebuild the injection.
    load = {"low": [500, 1500, 1000], "high": [500, 1600, 1000]}
    for scenario, p_set in load.items():
        gen = n.generators_t.p[scenario].iloc[0].to_numpy()
        np.testing.assert_allclose(
            sol.sel(scenario=scenario).to_numpy()[0], gen - p_set, atol=1e-6
        )
