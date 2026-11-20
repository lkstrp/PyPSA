# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for deriving branch impedances from standard types."""

import numpy as np
import pytest

import pypsa

OVERRIDE_MSG = "overridden by the standard type"


@pytest.fixture(params=["Transformer", "Line"])
def typed_branch(request):
    """Network with a single typed branch named 'branch'."""
    component = request.param
    n = pypsa.Network()
    if component == "Transformer":
        n.add("Bus", ["b0", "b1"], v_nom=[20.0, 0.4])
        n.add("Transformer", "branch", bus0="b0", bus1="b1", type="0.25 MVA 20/0.4 kV")
    else:
        n.add("Bus", ["b0", "b1"], v_nom=380.0)
        type_name = "Al/St 240/40 4-bundle 380.0"
        n.add("Line", "branch", bus0="b0", bus1="b1", length=10.0, type=type_name)
    return n, n.components[component].static


def test_type_derives_impedance_silently(typed_branch, caplog):
    n, static = typed_branch
    n.calculate_dependent_values()
    assert static.loc["branch", "x"] > 0
    assert OVERRIDE_MSG not in caplog.text


def test_manual_impedance_override_warns(typed_branch, caplog):
    """See https://github.com/PyPSA/PyPSA/issues/1054."""
    n, static = typed_branch
    n.calculate_dependent_values()

    static.loc["branch", ["r", "x"]] = 0.01
    n.calculate_dependent_values()
    assert OVERRIDE_MSG in caplog.text
    assert static.loc["branch", "r"] != 0.01

    caplog.clear()
    n.calculate_dependent_values()
    assert OVERRIDE_MSG not in caplog.text


TRAFO_TYPE = "0.25 MVA 20/0.4 kV"


def _trafo_network(num_parallel=1.0, count=1, **kwargs):
    """Slack bus at 20 kV, a load on the 0.4 kV bus and typed transformers."""
    n = pypsa.Network()
    n.add("Bus", "hv", v_nom=20.0)
    n.add("Bus", "lv", v_nom=0.4)
    n.add("Generator", "slack", bus="hv", control="Slack")
    n.add("Load", "load", bus="lv", p_set=0.2, q_set=0.05)
    names = [f"trafo{i}" for i in range(count)]
    n.add(
        "Transformer",
        names,
        bus0="hv",
        bus1="lv",
        type=TRAFO_TYPE,
        num_parallel=num_parallel,
        **kwargs,
    )
    return n


def test_parallel_transformer_type_scales_s_nom():
    single = _trafo_network(num_parallel=1)
    bank = _trafo_network(num_parallel=2)
    single.calculate_dependent_values()
    bank.calculate_dependent_values()
    t1 = single.c.transformers.static.loc["trafo0"]
    t2 = bank.c.transformers.static.loc["trafo0"]

    assert t2.s_nom == pytest.approx(2 * t1.s_nom)
    for attr in ["r_pu", "x_pu"]:
        assert t2[attr] == pytest.approx(t1[attr] / 2)
    for attr in ["g_pu", "b_pu"]:
        assert t2[attr] == pytest.approx(2 * t1[attr])


def test_single_transformer_type_parameters_unchanged():
    n = _trafo_network(num_parallel=1)
    n.calculate_dependent_values()
    trafo = n.c.transformers.static.loc["trafo0"]
    typ = n.c.transformer_types.static.loc[TRAFO_TYPE]

    r = typ.vscr / 100.0
    x = np.sqrt((typ.vsc / 100.0) ** 2 - r**2)
    g = typ.pfe / (1000.0 * typ.s_nom)
    b = -np.sqrt(max((typ.i0 / 100.0) ** 2 - g**2, 0))

    assert trafo.s_nom == pytest.approx(typ.s_nom)
    assert trafo.r == pytest.approx(r)
    assert trafo.x == pytest.approx(x)
    assert trafo.g == pytest.approx(g)
    assert trafo.b == pytest.approx(b)


def test_parallel_transformer_type_power_flow_matches_explicit_units():
    bank = _trafo_network(num_parallel=2)
    explicit = _trafo_network(num_parallel=1, count=2)
    bank.pf()
    explicit.pf()

    for attr in ["v_mag_pu", "v_ang"]:
        np.testing.assert_allclose(
            bank.c.buses.dynamic[attr], explicit.c.buses.dynamic[attr]
        )
    for attr in ["p0", "q0", "p1", "q1"]:
        np.testing.assert_allclose(
            bank.c.transformers.dynamic[attr].sum(axis=1),
            explicit.c.transformers.dynamic[attr].sum(axis=1),
        )


def test_parallel_transformer_bank_rating_survives_pf(caplog):
    n = _trafo_network(num_parallel=2)
    type_s_nom = n.c.transformer_types.static.at[TRAFO_TYPE, "s_nom"]
    n.c.transformers.static.loc["trafo0", "s_nom"] = 2 * type_s_nom
    n.pf()
    assert n.c.transformers.static.at["trafo0", "s_nom"] == pytest.approx(
        2 * type_s_nom
    )
    assert OVERRIDE_MSG not in caplog.text
