# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Opening a record as a network's lazy source."""

# ruff: noqa: F811

from __future__ import annotations

import copy

import pandas as pd
import pytest

import pypsa

pytest.importorskip("datarecord", reason="needs the datarecord extra")

from datarecord import Workspace  # noqa: E402
from datarecord_helpers import assert_networks_equal, export_network  # noqa: E402
from test_io_datarecord import (  # noqa: E402, F401
    _curve_child,
    _three_port_link,
    ac_dc,
    exported,
    layout,
    multiperiod,
    multiperiod_revision,
    single_revision,
    solved,
    store,
    workspace,
)

from pypsa.network.io.datarecord import PyPSA, RecordSource  # noqa: E402
from pypsa.network.names import unchecked_names  # noqa: E402


class _Counting:
    """Wraps a source and counts its calls."""

    def __init__(self, source):
        self.source = source
        self.static_calls = []
        self.dynamic_calls = []
        self.names_calls = []
        self.piecewise_calls = []

    def static(self, ctype):
        self.static_calls.append(ctype.name)
        return self.source.static(ctype)

    def dynamic(self, ctype, attr):
        self.dynamic_calls.append((ctype.name, attr))
        return self.source.dynamic(ctype, attr)

    def piecewise(self, ctype, attr):
        self.piecewise_calls.append((ctype.name, attr))
        return self.source.piecewise(ctype, attr)

    def varying_attrs(self, ctype):
        return self.source.varying_attrs(ctype)

    def names(self, ctype):
        self.names_calls.append(ctype.name)
        return self.source.names(ctype)


def _count(n):
    """Install a counting wrapper on every record-backed component."""
    counters = {}
    for c in n.components:
        if isinstance(c._source, RecordSource):
            counters[c.name] = counter = _Counting(c._source)
            c._set_source(counter)
    return counters


def _touch_everything(n):
    for c in n.components:
        c._materialize()


@pytest.fixture(scope="module")
def stochastic_revision(tmp_path_factory, ac_dc):
    n = ac_dc.copy()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    workspace = Workspace()
    yield (
        export_network(n, workspace.store(str(tmp_path_factory.mktemp("stochastic")))),
        n,
    )
    workspace.close()


def test_open_leaves_every_frame_unread(single_revision):
    n = PyPSA.open(single_revision.record)
    for c in n.components:
        if isinstance(c._source, RecordSource):
            assert c.dynamic.pending == set(c._source.varying_attrs(c.ctype))
    assert n.record is single_revision.record


def test_a_dynamic_read_removes_exactly_its_key(single_revision):
    n = PyPSA.open(single_revision.record)
    counters = _count(n)
    gens = n.c["Generator"]
    before = set(gens.dynamic.pending)
    _ = gens.dynamic["p_max_pu"]
    assert gens.dynamic.pending == before - {"p_max_pu"}
    assert counters["Generator"].dynamic_calls == [("Generator", "p_max_pu")]
    assert all(
        not calls.static_calls
        for name, calls in counters.items()
        if name != "Generator"
    )


def test_open_then_touch_equals_build(single_revision, ac_dc):
    n = PyPSA.open(single_revision.record)
    _touch_everything(n)
    assert_networks_equal(n, PyPSA.build(single_revision.record))
    assert_networks_equal(n, ac_dc)


def test_stochastic_open_equals_build(stochastic_revision):
    revision, original = stochastic_revision
    n = PyPSA.open(revision.record)
    _touch_everything(n)
    assert_networks_equal(n, PyPSA.build(revision.record))
    assert list(n.c["Generator"].static.index.unique("scenario")) == list(
        original.c["Generator"].static.index.unique("scenario")
    )


def _with_constraints(n):
    n = n.copy()
    n.add("GlobalConstraint", "plain", sense="<=", constant=1.0)
    n.add("GlobalConstraint", "at_bus", bus="Manchester", sense="<=", constant=2.0)
    return n


def test_global_constraints_open_equals_build(store, ac_dc):
    n = _with_constraints(ac_dc)
    record = export_network(n, store).record
    opened = PyPSA.open(record)
    _touch_everything(opened)
    assert_networks_equal(opened, PyPSA.build(record))
    assert_networks_equal(opened, n)


def test_stochastic_global_constraints_open_equals_build(store, ac_dc):
    n = _with_constraints(ac_dc)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.GlobalConstraint.static.loc[("high", "plain"), "constant"] = 9.0
    record = export_network(n, store).record
    opened = PyPSA.open(record)
    _touch_everything(opened)
    assert_networks_equal(opened, PyPSA.build(record))


def test_multiperiod_open_equals_build(multiperiod, multiperiod_revision):
    n = PyPSA.open(multiperiod_revision.record)
    _touch_everything(n)
    assert_networks_equal(n, PyPSA.build(multiperiod_revision.record))
    assert_networks_equal(n, multiperiod)


def test_identity_contract_and_mutation_persist(single_revision):
    n = PyPSA.open(single_revision.record)
    with pypsa.option_context("api.new_components_api", False):
        assert n.buses is n.c.buses.static
    name = n.c.generators.static.index[0]
    n.c.generators.static.loc[name, "p_nom"] = 1
    assert n.c.generators.static.loc[name, "p_nom"] == 1


@pytest.mark.parametrize("stochastic", [False, True])
def test_an_attribute_without_rows_is_an_empty_frame(
    single_revision, stochastic_revision, stochastic
):
    record = (stochastic_revision[0] if stochastic else single_revision).record
    n = PyPSA.open(record)
    frame = n.c["Generator"].dynamic["p_min_pu"]
    assert frame.index.equals(n.snapshots)
    assert frame.columns.names == (["scenario", "name"] if stochastic else ["name"])
    assert frame.shape[1] == 0


def test_a_three_port_link_is_served_lazily(store):
    n = _three_port_link()
    n.c["Link"].dynamic["p2"] = pd.DataFrame({"split": [1.0, 2.0]}, index=n.snapshots)
    record = export_network(n, store).record
    opened = PyPSA.open(record)
    link = opened.c["Link"]
    assert {"bus2", "efficiency2", "p2"} <= set(link.defaults.index)
    assert "p2" in link.dynamic.pending
    assert opened.c.links.static.loc["split", "bus2"] == "c"
    assert link.dynamic["p2"]["split"].to_list() == [1.0, 2.0]


def test_set_snapshots_materialises_every_dynamic_frame(single_revision):
    n = PyPSA.open(single_revision.record)
    n.set_snapshots(n.snapshots[:2])
    for c in n.components:
        assert not c.dynamic.pending


def test_add_reads_no_dynamic_frame(single_revision):
    """`add` reads `Generator`'s static, and other statics only for the cross-type name check."""
    n = PyPSA.open(single_revision.record)
    counters = _count(n)
    n.add("Generator", "new", bus=n.c.buses.static.index[0])
    assert counters["Generator"].static_calls
    assert not any(c.dynamic_calls for c in counters.values())
    assert n.c["Load"].dynamic.pending


def test_add_reads_only_the_names_of_other_types(single_revision):
    n = PyPSA.open(single_revision.record)
    bus = n.c.buses.static.index[0]
    counters = _count(n)
    n.add("Generator", "new", bus=bus)
    unread = {
        name: c.static_calls
        for name, c in counters.items()
        if name not in ("Generator", "Bus")
    }
    assert not any(unread.values()), unread
    assert counters["Line"].names_calls


@pytest.mark.parametrize("revision", ["single_revision", "stochastic_revision"])
def test_record_names_match_the_static_index(revision, request):
    revision = request.getfixturevalue(revision)
    if isinstance(revision, tuple):
        revision = revision[0]
    n = PyPSA.open(revision.record)
    for c in n.components:
        if isinstance(c._source, RecordSource) and c.name in c._source.types():
            assert set(c._source.names(c.ctype)) == set(c.names), c.name


def test_a_name_of_an_unread_type_is_refused(single_revision):
    n = PyPSA.open(single_revision.record)
    lines = n.c["Line"]
    taken = lines._source.names(lines.ctype)[0]
    with pytest.raises(ValueError, match="unique across entity types"):
        n.add("Generator", taken, bus=n.c.buses.static.index[0])


def test_statistics_reads_only_what_it_needs(solved, workspace, tmp_path):
    revision = export_network(solved, workspace.store(str(tmp_path)))
    n = PyPSA.open(revision.record)
    counters = _count(n)
    n.statistics.supply()
    for name, counter in counters.items():
        read = {a for _, a in counter.dynamic_calls}
        assert all(a.startswith("p") and a[1:].isdigit() or a == "p" for a in read), (
            name,
            read,
        )


def test_record_is_exposed_and_dropped_by_copy(single_revision):
    n = PyPSA.open(single_revision.record)
    assert n.record is single_revision.record
    copied = copy.deepcopy(n)
    assert copied.record is None
    for c in copied.components:
        assert isinstance(c._static, pd.DataFrame)
        assert not getattr(c.dynamic, "pending", set())


def test_an_opened_network_round_trips(single_revision, workspace, tmp_path, ac_dc):
    n = PyPSA.open(single_revision.record)
    again = export_network(n, workspace.store(str(tmp_path)))
    assert_networks_equal(PyPSA.open(again.record), ac_dc)


def test_an_opened_network_exports_to_netcdf(single_revision, tmp_path):
    n = PyPSA.open(single_revision.record)
    path = tmp_path / "opened.nc"
    n.export_to_netcdf(path)
    again = pypsa.Network(path)
    pd.testing.assert_frame_equal(
        again.c.generators.static.sort_index(axis=1),
        n.c.generators.static.sort_index(axis=1),
        check_dtype=False,
    )
    for attr in n.c.generators.dynamic:
        pd.testing.assert_frame_equal(
            again.c.generators.dynamic[attr].sort_index(axis=1),
            n.c.generators.dynamic[attr].sort_index(axis=1),
            check_dtype=False,
            check_freq=False,
            check_index_type=False,
            check_names=False,
        )


def test_build_is_fully_materialised(single_revision):
    n = PyPSA.build(single_revision.record)
    for c in n.components:
        assert not c.dynamic.pending


def test_added_component_gets_no_output_column(solved, workspace, tmp_path):
    """A series-only frame spans the record's own members, as after `build` then `add`."""
    revision = export_network(solved, workspace.store(str(tmp_path)))
    built = PyPSA.build(revision.record)
    opened = PyPSA.open(revision.record)
    for n in (built, opened):
        n.add("Generator", "new", bus=n.c.buses.static.index[0])
    assert opened.c.generators.dynamic.p.shape == built.c.generators.dynamic.p.shape
    assert "new" not in opened.c.generators.dynamic.p.columns


def test_open_into_rejects_populated_network(single_revision):
    into = pypsa.Network()
    into.add("Bus", "kept")
    with pytest.raises(ValueError, match="already holds"):
        PyPSA.open(single_revision.record, into=into)
    assert list(into.c.buses.static.index) == ["kept"]
    assert into.record is None


def test_open_into_empty_network(single_revision):
    into = pypsa.Network()
    n = PyPSA.open(single_revision.record, into=into)
    assert n is into
    assert n.record is single_revision.record


def test_rename_keeps_pending_dynamic_data(single_revision):
    built = PyPSA.build(single_revision.record)
    n = PyPSA.open(single_revision.record)
    gens = n.c["Generator"]
    attr = next(
        a
        for a in sorted(gens.dynamic.pending)
        if len(built.c["Generator"].dynamic[a].columns)
    )
    name = built.c["Generator"].dynamic[attr].columns[0]
    n.rename_component_names("Generator", **{name: "renamed"})
    pd.testing.assert_series_equal(
        n.c["Generator"].dynamic[attr]["renamed"],
        built.c["Generator"].dynamic[attr][name],
        check_names=False,
    )


def test_set_scenarios_on_an_opened_network_equals_build(single_revision):
    built = PyPSA.build(single_revision.record)
    built.set_scenarios({"low": 0.5, "high": 0.5})
    n = PyPSA.open(single_revision.record)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    _touch_everything(n)
    assert_networks_equal(n, built)


def test_network_constructor_opens_lazily(ac_dc, tmp_path):
    ac_dc.export_to_datarecord(tmp_path / "record")
    n = pypsa.Network(tmp_path / "record")
    assert n.record is not None
    assert any(c.dynamic.pending for c in n.components)
    _touch_everything(n)
    assert_networks_equal(n, ac_dc)


def test_from_datarecord_opens_lazily(single_revision):
    n = pypsa.Network.from_datarecord(single_revision.record)
    assert n.record is single_revision.record
    assert any(c.dynamic.pending for c in n.components)


def test_a_selection_opens_lazily_as_a_smaller_network(single_revision, ac_dc):
    """`Network.from_datarecord` takes a selection, narrows it before reading, and reads no attribute it does not touch."""
    view = single_revision.record.select(bus=["London", "Norwich"])
    n = pypsa.Network.from_datarecord(view)
    counters = _count(n)
    assert not any(calls.static_calls for calls in counters.values()), "open reads none"

    assert n.c.Bus.static.index.tolist() == ["London", "Norwich"]
    assert n.c.Generator.static.empty
    assert len(n.c.Line.static) < len(ac_dc.c.Line.static)
    assert counters["Bus"].static_calls == ["Bus"]
    assert not counters["Bus"].dynamic_calls, "no series of the buses is read"
    assert not counters["Load"].dynamic_calls
    assert "p_set" in n.c.Load.dynamic.pending, "the loads' series stay unread"


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_a_bus_selection_drops_the_constraint_at_a_dropped_bus(store, ac_dc, lazy):
    """A constraint at a bus the selection drops goes, one at a kept bus and one at none stay."""
    n = ac_dc.copy()
    n.add("GlobalConstraint", "at_gone", bus="Manchester", sense="<=", constant=1.0)
    n.add("GlobalConstraint", "at_kept", bus="London", sense="<=", constant=2.0)
    n.add("GlobalConstraint", "bus_less", sense="<=", constant=3.0)
    record = export_network(n, store).record
    view = record.select(bus=["London", "Norwich"])

    built = PyPSA.open(view) if lazy else PyPSA.build(view)

    assert built.c.GlobalConstraint.static.index.tolist() == [
        "co2_limit",
        "at_kept",
        "bus_less",
    ]


def test_import_into_empty_network_opens_lazily(ac_dc, tmp_path):
    ac_dc.export_to_datarecord(tmp_path / "record")
    n = pypsa.Network()
    n.import_from_datarecord(tmp_path / "record")
    assert n.record is not None
    assert any(c.dynamic.pending for c in n.components)


def test_import_into_populated_network_merges_and_deduplicates(ac_dc, tmp_path):
    ac_dc.export_to_datarecord(tmp_path / "record")
    n = pypsa.Network()
    n.set_snapshots(ac_dc.snapshots)
    n.add("Bus", "extra")
    n.add("Generator", "extra gen", bus="extra")
    clash = ac_dc.c.generators.static.index[0]
    with unchecked_names(n):
        n.add("Load", clash, bus="extra")
    n.import_from_datarecord(tmp_path / "record")
    assert "extra gen" in n.c.generators.static.index
    assert set(ac_dc.c.generators.static.index) <= set(n.c.generators.static.index)
    assert n.deduplicated_names
    assert clash in n.c.generators.static.index
    assert clash not in n.c.loads.static.index
    assert not any(c.dynamic.pending for c in n.components)


def test_a_curve_is_read_on_first_access_only(exported, monkeypatch):
    """Opening a record with curves reads none, the first access reads one."""
    revision = _curve_child(
        exported,
        "marginal_cost",
        curves={"Manchester Wind": ((0.0, 0.0), (0.5, 20.0), (1.0, 35.0))},
        scalar=False,
    )
    calls = []
    read = RecordSource.piecewise

    def counting(self, ctype, attr):
        calls.append((ctype.name, attr))
        return read(self, ctype, attr)

    monkeypatch.setattr(RecordSource, "piecewise", counting)
    n = PyPSA.open(revision.record)
    gens = n.c["Generator"]
    assert calls == []
    assert "marginal_cost" in gens.piecewise.pending
    _ = gens.piecewise["marginal_cost"]
    assert calls == [("Generator", "marginal_cost")]
    assert "marginal_cost" not in gens.piecewise.pending


def _linked():
    """A three-port Link and a same-bus Link beside a Generator, with `p` on each port."""
    n = _three_port_link()
    n.add("Generator", "gen", bus="a")
    n.add("Link", "same", bus0="a", bus1="a", bus2="c", efficiency=0.9, efficiency2=0.5)
    for attr, values in {
        "p0": [3.0, 4.0],
        "p1": [-1.0, -2.0],
        "p2": [-1.5, -1.5],
    }.items():
        n.c["Link"].dynamic[attr] = pd.DataFrame(
            {"split": values, "same": values}, index=n.snapshots
        ).rename_axis(columns="name")
    return n


def test_three_port_and_same_bus_links_open_equal_to_build(store):
    """A lazy open serves every port's bus, efficiency and `p` as the eager build does."""
    record = export_network(_linked(), store).record
    opened = PyPSA.open(record)
    _touch_everything(opened)
    assert_networks_equal(opened, PyPSA.build(record))
    assert opened.c["Link"].static.loc["same", ["bus0", "bus1", "bus2"]].tolist() == [
        "a",
        "a",
        "c",
    ]


def test_an_edit_keeps_the_links_connections_on_re_export(store):
    """Re-exporting an opened network after a Generator edit leaves the Links' ports and values."""
    revision = export_network(_linked(), store)
    opened = pypsa.Network.from_datarecord(revision.record)
    opened.c["Generator"].static.loc["gen", "p_nom"] = 5.0

    child = store.commit(opened.to_datarecord(base=opened.record), parent=revision)

    built = PyPSA.build(child.record)
    assert built.c["Generator"].static.at["gen", "p_nom"] == 5.0
    links = built.c["Link"]
    cols = ["bus0", "bus1", "bus2", "efficiency", "efficiency2"]
    assert links.static.loc["same", cols].tolist() == ["a", "a", "c", 0.9, 0.5]
    assert links.static.loc["split", ["bus0", "bus1", "bus2"]].tolist() == list("abc")
    assert links.dynamic["p2"]["same"].tolist() == [-1.5, -1.5]
    assert links.dynamic["p1"]["split"].tolist() == [-1.0, -2.0]
