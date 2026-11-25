# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Writing a network opened from a record back as a thin child layer."""

# ruff: noqa: F811

from __future__ import annotations

import pandas as pd
import pytest

import pypsa

pytest.importorskip("datarecord", reason="needs the datarecord extra")

from datarecord_helpers import assert_networks_equal, export_network  # noqa: E402
from test_io_datarecord import (  # noqa: E402, F401
    _north,
    _shaped,
    ac_dc,
    single_revision,
    store,
    workspace,
)
from test_io_datarecord_lazy import _count, stochastic_revision  # noqa: E402, F401

from pypsa.network.io.datarecord import PyPSA  # noqa: E402
from pypsa.network.io.datarecord.source import _NetworkSource  # noqa: E402


def _rows(layer, attr) -> int:
    if attr not in layer.attributes:
        return 0
    return len(layer.attributes[attr].df())


def _open(rev):
    return pypsa.Network(rev.store.uri, revision=str(rev.id))


def _layer(n, base=None):
    base = n.record if base is None else base
    return PyPSA.to_datarecord(n, base.con, base=base)


def _child(n, meta=None):
    n.commit(meta)
    return n._revision


def test_a_changed_series_writes_only_that_attribute(single_revision):
    n = _open(single_revision)
    counters = _count(n)
    n.generators_t.p_max_pu["Manchester Wind"] *= 0.9
    layer = _layer(n)
    assert _rows(layer, "p_max_pu") > 0
    assert _rows(layer, "marginal_cost") == 0
    assert set(counters["Generator"].dynamic_calls) == {("Generator", "p_max_pu")}
    child = _child(n)
    assert_networks_equal(PyPSA.build(child.record), n)


def test_a_changed_static_value(single_revision):
    n = _open(single_revision)
    n.c.Generator.static.at["Manchester Wind", "marginal_cost"] += 1.0
    layer = _layer(n)
    assert _rows(layer, "marginal_cost") == 1
    child = PyPSA.build(_child(n).record)
    assert_networks_equal(child, n)


def test_a_read_alone_writes_no_rows(single_revision):
    n = _open(single_revision)
    _ = n.generators_t.p_max_pu.sum()
    _ = n.statistics.energy_balance()
    layer = _layer(n)
    assert _rows(layer, "p_max_pu") == 0
    assert _rows(layer, "marginal_cost") == 0
    assert _rows(layer, "p_nom") == 0


def test_a_dropped_series_column_lands_as_its_static_value(single_revision):
    n = _open(single_revision)
    frame = n.generators_t.p_max_pu
    n.generators_t.p_max_pu = frame.drop(columns="Manchester Wind")
    n.c.Generator.static.at["Manchester Wind", "p_max_pu"] = 0.5
    child = PyPSA.build(_child(n).record)
    assert "Manchester Wind" not in child.generators_t.p_max_pu
    assert child.c.Generator.static.at["Manchester Wind", "p_max_pu"] == 0.5


def test_nothing_changed_still_commits(single_revision):
    n = _open(single_revision)
    child = _child(n)
    assert_networks_equal(
        PyPSA.build(child.record), PyPSA.build(single_revision.record)
    )


def test_solved_results_land_with_a_changed_input(single_revision):
    n = _open(single_revision)
    n.c.Generator.static.loc["Manchester Wind", "marginal_cost"] += 1.0
    n.optimize(solver_name="highs")
    layer = _layer(n)
    assert _rows(layer, "marginal_cost") == 1
    assert _rows(layer, "p") > 0
    child = PyPSA.build(_child(n).record)
    assert (
        child.c.Generator.static.at["Manchester Wind", "marginal_cost"]
        == (n.c.Generator.static.at["Manchester Wind", "marginal_cost"])
    )
    pd.testing.assert_frame_equal(
        child.generators_t.p, n.generators_t.p, check_freq=False, atol=1e-6
    )


def test_a_foreign_base_compares_every_attribute(single_revision):
    other = PyPSA.build(single_revision.record)
    other.c.Generator.static.at["Manchester Wind", "p_nom_min"] = 7.0
    foreign = single_revision.store.commit(
        _layer(other, single_revision.record),
        parent=single_revision,
    )
    n = _open(single_revision)
    _ = n.c.Generator.static
    layer = PyPSA.to_datarecord(n, n.record.con, base=foreign.record)
    assert _rows(layer, "p_nom_min") > 0


def test_an_in_memory_network_over_a_base_compares_every_attribute(single_revision):
    n = PyPSA.build(single_revision.record)
    n.c.Generator.static.at["Manchester Wind", "p_nom_min"] = 7.0
    layer = _layer(n, single_revision.record)
    assert _rows(layer, "p_nom_min") == 1


def test_only_a_touched_types_attributes_are_listed(single_revision):
    n = _open(single_revision)
    n.generators_t.p_max_pu["Manchester Wind"] *= 0.9
    layer = _layer(n)
    attributes = set(layer.attributes)
    assert "p_max_pu" in attributes
    assert not attributes & {"s_nom", "r", "length"}


def test_a_static_edit_of_a_pending_series_keeps_the_parent_series(single_revision):
    seeded = PyPSA.build(single_revision.record)
    held, plain = seeded.c.Generator.static.index[:2]
    seeded.generators_t.marginal_cost[held] = 3.0
    parent = single_revision.store.commit(
        _layer(seeded, single_revision.record), parent=single_revision
    )
    n = _open(parent)
    counters = _count(n)
    n.c.Generator.static.at[plain, "marginal_cost"] += 1.0
    child = PyPSA.build(_child(n).record)
    assert not counters["Generator"].dynamic_calls
    assert (
        child.c.Generator.static.at[plain, "marginal_cost"]
        == (n.c.Generator.static.at[plain, "marginal_cost"])
    )
    assert (child.generators_t.marginal_cost[held] == 3.0).all()


def _read_types(counters) -> tuple[list[str], list[str]]:
    static = [t for c in counters.values() for t in c.static_calls]
    dynamic = [t for c in counters.values() for t, _ in c.dynamic_calls]
    return static, dynamic


def test_an_untouched_type_is_not_read_at_export(single_revision):
    n = _open(single_revision)
    counters = _count(n)
    n.generators_t.p_max_pu["Manchester Wind"] *= 0.9
    layer = _layer(n)
    child = _child(n)
    static, dynamic = _read_types(counters)
    assert static == ["Generator"]
    assert set(dynamic) == {"Generator"}
    assert _rows(layer, "p_max_pu") > 0
    assert_networks_equal(PyPSA.build(child.record), n)


def _edit_and_export(rev, edit):
    n = _open(rev)
    counters = _count(n)
    edit(n)
    before = _calls(counters)
    layer = _layer(n)
    committed = _child(n)
    read = {k for k, v in _calls(counters).items() if v != before[k]}
    return n, layer, read, PyPSA.build(committed.record)


def _calls(counters) -> dict[str, tuple[int, int]]:
    return {k: (len(c.static_calls), len(c.dynamic_calls)) for k, c in counters.items()}


def test_an_added_component_leaves_other_types_unread(single_revision):
    def edit(n):
        n.add("Load", "new load", bus="Manchester", p_set=7.0)

    n, layer, read, child = _edit_and_export(single_revision, edit)
    assert read <= {"Load"}
    assert "new load" in child.c.Load.static.index
    assert child.c.Load.static.at["new load", "p_set"] == 7.0
    assert_networks_equal(child, n)


def test_a_removed_component_is_gone_from_the_child(single_revision):
    def edit(n):
        n.remove("Generator", "Norway Wind")

    n, layer, read, child = _edit_and_export(single_revision, edit)
    assert read <= {"Generator"}
    assert "Norway Wind" not in child.c.Generator.static.index
    assert_networks_equal(child, n)


def test_a_removed_snapshot_is_gone_from_the_child(single_revision):
    def edit(n):
        n.set_snapshots(n.snapshots[:-1])

    n, layer, read, child = _edit_and_export(single_revision, edit)
    assert len(child.snapshots) == len(n.snapshots)
    assert_networks_equal(child, n)


def test_a_selection_opened_network_is_refused_for_an_input_export(single_revision):
    first = single_revision.record.dimensions["snapshot"].fetchall()[0][0]
    view = single_revision.record.select(snapshot=[first])
    n = pypsa.Network.from_datarecord(view)
    with pytest.raises(ValueError, match="results_only=True"):
        _layer(n)


def test_a_stochastic_network_leaves_untouched_types_unread(stochastic_revision):
    revision, _ = stochastic_revision

    def edit(n):
        n.c.Generator.static.loc[(slice(None), "Manchester Wind"), "marginal_cost"] += (
            1.0
        )

    n, layer, read, child = _edit_and_export(revision, edit)
    assert read <= {"Generator"}
    assert _rows(layer, "marginal_cost") > 0
    assert_networks_equal(child, n)


def test_untouched_global_constraints_are_served_from_the_base(store, ac_dc):
    n = ac_dc.copy()
    n.add("GlobalConstraint", "plain", sense="<=", constant=1.0)
    n.add("GlobalConstraint", "at_bus", bus="Manchester", sense="<=", constant=2.0)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    revision = export_network(n, store)

    def edit(n):
        n.c.Generator.static.loc[(slice(None), "Manchester Wind"), "marginal_cost"] += (
            1.0
        )

    got, layer, read, child = _edit_and_export(revision, edit)
    assert read <= {"Generator"}
    assert _rows(layer, "constant") == 0
    assert set(child.c.GlobalConstraint.static["bus"]) == {"", "Manchester"}
    assert_networks_equal(child, got)


def test_a_stray_series_never_read_is_not_compared(store, monkeypatch):
    """A series left to `base=n.record` is never read, so its stray column goes unseen.

    The base holds a Generator `p_max_pu` column named after the Load, written
    with the check off. Editing only a Load leaves the Generator type unread.
    The reader keys a series on its own type's members and drops that column
    on load, so this shows the cost property: the check reads nothing.
    """
    n = _north()
    n.add("Load", "demand", bus="north", p_set=1.0)
    n.c.Generator.dynamic.p_max_pu["demand"] = [0.5, 0.6]
    with monkeypatch.context() as patch:
        patch.setattr(_NetworkSource, "_stray_series", lambda self: None)
        root = export_network(n, store)
    opened = _open(root)
    counters = _count(opened)
    opened.c.Load.static.loc["demand", "p_set"] = 2.0
    layer = _layer(opened)
    assert _rows(layer, "p_set") > 0
    static, dynamic = _read_types(counters)
    assert "Generator" not in static
    assert "Generator" not in dynamic


def test_an_untouched_shape_type_is_not_read_at_export(store):
    root = export_network(_shaped(), store)
    n = _open(root)
    counters = _count(n)
    n.generators_t.p_max_pu["gen"] = [0.5, 0.6]
    child = _child(n)
    assert "Shape" not in _read_types(counters)[0]
    assert_networks_equal(PyPSA.build(child.record), n)


CURVE = {0.0: 0.0, 0.5: 20.0, 1.0: 35.0}


@pytest.fixture
def curve_revision(store):
    """A record with a `marginal_cost` curve on `gas` and an `efficiency` curve on `conv`."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", ["a", "b"])
    n.add("Generator", "gas", bus="a", p_nom=10, marginal_cost=CURVE)
    n.add("Generator", "coal", bus="a", p_nom=10, marginal_cost=20.0)
    n.add("Link", "conv", bus0="a", bus1="b", p_nom=5, efficiency={0.0: 0.0, 1.0: 0.8})
    return export_network(n, store)


def _curve_rows(layer, attr) -> pd.DataFrame:
    if attr not in layer.attributes:
        return pd.DataFrame()
    rows = layer.attributes[attr].df()
    return rows[rows["breakpoint"].notna()]


def _curve_of(n, name, attr="marginal_cost"):
    frame = n.c.Generator.piecewise.get(attr)
    if frame is None or name not in frame.columns.get_level_values("name"):
        return None
    return frame[name].dropna()


def test_an_untouched_curve_writes_no_rows_and_stays_unread(curve_revision):
    n = _open(curve_revision)
    counters = _count(n)
    n.c.Generator.static.at["coal", "p_nom"] += 1.0
    layer = _layer(n)
    assert _curve_rows(layer, "marginal_cost").empty
    assert counters["Generator"].piecewise_calls == []
    child = pypsa.Network.from_datarecord(_child(n).record)
    assert _curve_of(child, "gas")["marginal_cost"].to_list() == [0.0, 20.0, 35.0]


def test_a_curve_read_but_unchanged_writes_no_rows(curve_revision):
    n = _open(curve_revision)
    _ = n.c.Generator.piecewise["marginal_cost"]
    layer = _layer(n)
    assert _curve_rows(layer, "marginal_cost").empty
    child = pypsa.Network.from_datarecord(_child(n).record)
    assert _curve_of(child, "gas")["marginal_cost"].to_list() == [0.0, 20.0, 35.0]


def test_a_changed_curve_is_written_whole(curve_revision):
    n = _open(curve_revision)
    n.c.Generator.piecewise["marginal_cost"].loc[1, ("gas", "marginal_cost")] = 25.0
    layer = _layer(n)
    assert len(_curve_rows(layer, "marginal_cost")) == 3
    child = pypsa.Network.from_datarecord(_child(n).record)
    assert _curve_of(child, "gas")["marginal_cost"].to_list() == [0.0, 25.0, 35.0]


def test_a_removed_curve_writes_the_default_scalar(curve_revision):
    n = _open(curve_revision)
    frame = n.c.Generator.piecewise["marginal_cost"]
    n.c.Generator.piecewise["marginal_cost"] = frame.drop(columns="gas", level="name")
    layer = _layer(n)
    assert _curve_rows(layer, "marginal_cost").empty
    child = pypsa.Network.from_datarecord(_child(n).record)
    assert _curve_of(child, "gas") is None
    default = child.c.Generator.defaults.at["marginal_cost", "default"]
    assert child.c.Generator.static.at["gas", "marginal_cost"] == default


def test_an_added_curve_shadows_the_base_scalar(curve_revision):
    n = _open(curve_revision)
    n.add("Generator", "tmp", bus="a", p_nom=1, marginal_cost={0.0: 0.0, 1.0: 30.0})
    frame = n.c.Generator.piecewise["marginal_cost"]
    n.c.Generator.piecewise["marginal_cost"] = frame.rename(columns={"tmp": "coal"})
    n.remove("Generator", "tmp")
    child = pypsa.Network.from_datarecord(_child(n).record)
    assert _curve_of(child, "coal")["marginal_cost"].to_list() == [0.0, 30.0]


def test_an_unread_link_curve_survives_an_edit_of_another_attribute(curve_revision):
    n = _open(curve_revision)
    n.c.Link.static.at["conv", "p_nom"] += 1.0
    child = pypsa.Network.from_datarecord(_child(n).record)
    frame = child.c.Link.piecewise["efficiency"]
    assert frame[("conv", "efficiency")].dropna().to_list() == [0.0, 0.8]


@pytest.mark.parametrize("kwargs", [{"base": None}, {"results_only": True}])
def test_to_datarecord_takes_no_base_or_results_only(single_revision, kwargs):
    n = _open(single_revision)
    with pytest.raises(TypeError):
        n.to_datarecord(**kwargs)
