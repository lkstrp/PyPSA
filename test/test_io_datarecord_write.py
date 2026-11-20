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
    _shaped,
    ac_dc,
    single_revision,
    store,
    workspace,
)
from test_io_datarecord_lazy import _count, stochastic_revision  # noqa: E402, F401

from pypsa.network.io.datarecord import PyPSA  # noqa: E402


def _rows(layer, attr) -> int:
    if attr not in layer.attributes:
        return 0
    return len(layer.attributes[attr].df())


def _child(rev, layer):
    return rev.store.commit(layer, parent=rev)


def test_a_changed_series_writes_only_that_attribute(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    counters = _count(n)
    n.generators_t.p_max_pu["Manchester Wind"] *= 0.9
    layer = n.to_datarecord(base=n.record)
    assert _rows(layer, "p_max_pu") > 0
    assert _rows(layer, "marginal_cost") == 0
    assert set(counters["Generator"].dynamic_calls) == {("Generator", "p_max_pu")}
    child = _child(single_revision, layer)
    assert_networks_equal(PyPSA.build(child.record), n)


def test_a_changed_static_value(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    n.c.Generator.static.at["Manchester Wind", "marginal_cost"] += 1.0
    layer = n.to_datarecord(base=n.record)
    assert _rows(layer, "marginal_cost") == 1
    child = PyPSA.build(_child(single_revision, layer).record)
    assert_networks_equal(child, n)


def test_a_read_alone_writes_no_rows(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    _ = n.generators_t.p_max_pu.sum()
    _ = n.statistics.energy_balance()
    layer = n.to_datarecord(base=n.record)
    assert _rows(layer, "p_max_pu") == 0
    assert _rows(layer, "marginal_cost") == 0
    assert _rows(layer, "p_nom") == 0


def test_a_dropped_series_column_lands_as_its_static_value(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    frame = n.generators_t.p_max_pu
    n.generators_t.p_max_pu = frame.drop(columns="Manchester Wind")
    n.c.Generator.static.at["Manchester Wind", "p_max_pu"] = 0.5
    layer = n.to_datarecord(base=n.record)
    child = PyPSA.build(_child(single_revision, layer).record)
    assert "Manchester Wind" not in child.generators_t.p_max_pu
    assert child.c.Generator.static.at["Manchester Wind", "p_max_pu"] == 0.5


def test_nothing_changed_still_commits(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    layer = n.to_datarecord(base=n.record)
    child = _child(single_revision, layer)
    assert_networks_equal(
        PyPSA.build(child.record), PyPSA.build(single_revision.record)
    )


def test_solved_results_land_with_a_changed_input(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    n.c.Generator.static.loc["Manchester Wind", "marginal_cost"] += 1.0
    n.optimize(solver_name="highs")
    layer = n.to_datarecord(base=n.record)
    assert _rows(layer, "marginal_cost") == 1
    assert _rows(layer, "p") > 0
    child = PyPSA.build(_child(single_revision, layer).record)
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
        other.to_datarecord(single_revision.record.con, base=single_revision.record),
        parent=single_revision,
    )
    n = pypsa.Network.from_datarecord(single_revision.record)
    _ = n.c.Generator.static
    layer = n.to_datarecord(n.record.con, base=foreign.record)
    assert _rows(layer, "p_nom_min") > 0


def test_an_in_memory_network_over_a_base_compares_every_attribute(single_revision):
    n = PyPSA.build(single_revision.record)
    n.c.Generator.static.at["Manchester Wind", "p_nom_min"] = 7.0
    layer = n.to_datarecord(base=single_revision.record)
    assert _rows(layer, "p_nom_min") == 1


def test_only_a_touched_types_attributes_are_listed(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    n.generators_t.p_max_pu["Manchester Wind"] *= 0.9
    layer = n.to_datarecord(base=n.record)
    attributes = set(layer.attributes)
    assert "p_max_pu" in attributes
    assert not attributes & {"s_nom", "r", "length"}


def test_a_static_edit_of_a_pending_series_keeps_the_parent_series(single_revision):
    seeded = PyPSA.build(single_revision.record)
    held, plain = seeded.c.Generator.static.index[:2]
    seeded.generators_t.marginal_cost[held] = 3.0
    parent = single_revision.store.commit(
        seeded.to_datarecord(base=single_revision.record), parent=single_revision
    )
    n = pypsa.Network.from_datarecord(parent.record)
    counters = _count(n)
    n.c.Generator.static.at[plain, "marginal_cost"] += 1.0
    layer = n.to_datarecord(base=n.record)
    child = PyPSA.build(_child(parent, layer).record)
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
    n = pypsa.Network.from_datarecord(single_revision.record)
    counters = _count(n)
    n.generators_t.p_max_pu["Manchester Wind"] *= 0.9
    layer = n.to_datarecord(base=n.record)
    child = _child(single_revision, layer)
    static, dynamic = _read_types(counters)
    assert static == ["Generator"]
    assert set(dynamic) == {"Generator"}
    assert _rows(layer, "p_max_pu") > 0
    assert_networks_equal(PyPSA.build(child.record), n)


def _edit_and_export(rev, edit):
    n = pypsa.Network.from_datarecord(rev.record)
    counters = _count(n)
    edit(n)
    before = _calls(counters)
    layer = n.to_datarecord(base=n.record)
    committed = _child(rev, layer)
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
    first = single_revision.record.dims["snapshot"].fetchall()[0][0]
    view = single_revision.record.select(snapshot=[first])
    n = pypsa.Network.from_datarecord(view)
    with pytest.raises(ValueError, match="results_only=True"):
        n.to_datarecord(base=n.record)


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


def test_an_untouched_shape_type_is_not_read_at_export(store):
    root = export_network(_shaped(), store)
    n = pypsa.Network.from_datarecord(root.record)
    counters = _count(n)
    n.generators_t.p_max_pu["gen"] = [0.5, 0.6]
    layer = n.to_datarecord(base=n.record)
    child = _child(root, layer)
    assert "Shape" not in _read_types(counters)[0]
    assert_networks_equal(PyPSA.build(child.record), n)
