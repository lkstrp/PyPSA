# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Opening a datarecord store through `Network(path)` and `revision=`."""

from __future__ import annotations

import copy
import pickle

import pytest

import pypsa

pytest.importorskip("datarecord", reason="needs the datarecord extra")

from datarecord import Workspace  # noqa: E402
from datarecord_helpers import assert_networks_equal, export_network  # noqa: E402

from pypsa.network.io.datarecord import PyPSA  # noqa: E402


@pytest.fixture
def workspace():
    ws = Workspace()
    yield ws
    ws.close()


@pytest.fixture
def root(workspace, tmp_path):
    """A store holding `ac_dc` as one root revision."""
    n = pypsa.examples.ac_dc_meshed()
    store = workspace.store(str(tmp_path / "runs"))
    return export_network(n, store)


def _child(rev, p_nom, meta):
    n = pypsa.Network.from_datarecord(rev.record)
    n.c["Generator"].static.iloc[
        0, n.c["Generator"].static.columns.get_loc("p_nom")
    ] = p_nom
    layer = PyPSA.to_datarecord(n, rev.store.workspace.con, base=n.record)
    return rev.store.commit(layer, parent=rev, meta=meta)


@pytest.fixture
def branched(root):
    a = _child(root, 11.0, {"scenario": "a"})
    b = _child(root, 22.0, {"scenario": "b"})
    return root, a, b


def _p_nom(n):
    return n.c["Generator"].static["p_nom"].iloc[0]


def test_single_leaf_opens(root):
    child = _child(root, 11.0, {"rule": "x"})
    store_path = root.store.uri
    n = pypsa.Network(store_path)
    assert n.revision == str(child.id)
    assert _p_nom(n) == 11.0


def test_root_only_store_opens_the_root(root):
    n = pypsa.Network(root.store.uri)
    assert n.revision == str(root.id)


def test_several_leaves_raise_with_metas(branched):
    root, a, b = branched
    with pytest.raises(ValueError, match="leaf") as exc:
        pypsa.Network(root.store.uri)
    msg = str(exc.value)
    assert str(a.id) in msg
    assert str(b.id) in msg
    assert '"scenario": "a"' in msg
    assert '"scenario": "b"' in msg


def test_revision_by_full_id_and_prefix(branched):
    root, a, b = branched
    uri = root.store.uri
    assert _p_nom(pypsa.Network(uri, revision=str(a.id))) == 11.0
    assert _p_nom(pypsa.Network(uri, revision=b.id.hex[:12])) == 22.0
    assert pypsa.Network.from_datarecord(uri, revision=str(a.id)).revision == str(a.id)
    n = pypsa.Network()
    n.import_from_datarecord(uri, revision=str(b.id)[:10])
    assert n.revision == str(b.id)


def test_unknown_prefix_lists_all_ids(branched):
    root, a, b = branched
    with pytest.raises(ValueError, match="no revision") as exc:
        pypsa.Network(root.store.uri, revision="zzzz")
    for rev in (root, a, b):
        assert str(rev.id) in str(exc.value)


def test_ambiguous_prefix_lists_matches(branched):
    root, a, b = branched
    ids = [str(r.id) for r in (root, a, b)]
    # The empty prefix matches every revision.
    with pytest.raises(ValueError, match="ambiguous") as exc:
        pypsa.Network(root.store.uri, revision="")
    for i in ids:
        assert i in str(exc.value)


def test_layer_folder_path_resolves_through_parents(branched):
    root, a, _ = branched
    n = pypsa.Network(root.store.layer_dir(a.id))
    assert n.revision == str(a.id)
    assert _p_nom(n) == 11.0
    n_root = pypsa.Network(root.store.layer_dir(root.id))
    assert n_root.revision == str(root.id)
    assert len(n_root.c["Generator"].static) > 0


def test_pending_layer_raises(root):
    claim = root.store.claim(parent=root)
    with pytest.raises(ValueError, match="not committed"):
        pypsa.Network(root.store.layer_dir(claim.id))


def test_revision_refused_on_non_store_path(root, tmp_path):
    ac_dc = pypsa.examples.ac_dc_meshed()
    nc = tmp_path / "n.nc"
    ac_dc.export_to_netcdf(nc)
    with pytest.raises(ValueError, match="store"):
        pypsa.Network(nc, revision="abc")
    plain = tmp_path / "plain"
    Workspace.process().write(ac_dc.to_datarecord(), str(plain))
    with pytest.raises(ValueError, match="store"):
        pypsa.Network(plain, revision="abc")
    with pytest.raises(ValueError, match="store"):
        pypsa.Network(root.store.layer_dir(root.id), revision="abc")


def test_plain_record_folder_still_opens(tmp_path):
    ac_dc = pypsa.examples.ac_dc_meshed()
    plain = tmp_path / "plain"
    Workspace.process().write(ac_dc.to_datarecord(), str(plain))
    n = pypsa.Network(plain)
    assert n.revision is None
    assert n.record is not None


def test_revision_none_without_store(tmp_path):
    assert pypsa.Network().revision is None
    nc = tmp_path / "n.nc"
    pypsa.examples.ac_dc_meshed().export_to_netcdf(nc)
    assert pypsa.Network(nc).revision is None


def test_pickle_and_deepcopy_drop_revision(root):
    n = pypsa.Network(root.store.uri)
    assert n.revision is not None
    assert pickle.loads(pickle.dumps(n)).revision is None
    assert copy.deepcopy(n).revision is None
    assert n.revision is not None


def test_export_writes_a_store_with_one_root_revision(tmp_path):
    n = pypsa.examples.ac_dc_meshed()
    path = tmp_path / "runs"
    n.export_to_datarecord(path)
    assert (path / "layers").is_dir()
    assert not (path / "manifest.json").exists()
    revisions = list(Workspace.process().store(str(path)).revisions())
    assert len(revisions) == 1
    assert revisions[0].parent is None
    reopened = pypsa.Network(path)
    assert reopened.revision == str(revisions[0].id)
    assert_networks_equal(reopened, n)


def test_export_to_existing_path_raises(tmp_path):
    n = pypsa.examples.ac_dc_meshed()
    with pytest.raises(FileExistsError):
        n.export_to_datarecord(tmp_path)


def _store_network(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    n = pypsa.examples.ac_dc_meshed()
    path = tmp_path / "runs"
    n.export_to_datarecord(path)
    return path, pypsa.Network(path)


def test_commit_stores_edit_and_unread_falls_through(tmp_path):
    path, n = _store_network(tmp_path)
    parent_id = n.revision
    old_cap = pypsa.Network(path).c["Line"].static["s_nom"].copy()
    n.c["Generator"].static.iloc[
        0, n.c["Generator"].static.columns.get_loc("p_nom")
    ] = 123.0
    rid = n.commit(meta={"rule": "edit"})
    assert n.revision == rid != parent_id
    child = pypsa.Network(path, revision=rid)
    assert _p_nom(child) == 123.0
    assert (child.c["Line"].static["s_nom"] == old_cap).all()
    rev = Workspace.process().store(str(path)).revision(__import__("uuid").UUID(rid))
    assert rev.meta == {"rule": "edit"}
    assert str(rev.parent) == parent_id


def test_commit_after_optimize_stores_results_and_edit(tmp_path):
    path, n = _store_network(tmp_path)
    n.c["Generator"].static.iloc[
        0, n.c["Generator"].static.columns.get_loc("p_nom")
    ] = 5000.0
    n.optimize(solver_name="highs")
    rid = n.commit()
    child = pypsa.Network(path, revision=rid)
    assert _p_nom(child) == 5000.0
    assert not child.c["Generator"].dynamic.p.empty
    assert child.objective == pytest.approx(n.objective)


def test_two_stacked_commits(tmp_path):
    path, n = _store_network(tmp_path)
    first = n.commit(meta={"i": 1})
    assert n.revision == first
    n.c["Generator"].static.iloc[
        0, n.c["Generator"].static.columns.get_loc("p_nom")
    ] = 77.0
    second = n.commit(meta={"i": 2})
    assert n.revision == second
    rev = Workspace.process().store(str(path)).revision(__import__("uuid").UUID(second))
    assert str(rev.parent) == first
    assert _p_nom(pypsa.Network(path)) == 77.0


def test_reading_unloaded_series_after_commit(tmp_path):
    path, n = _store_network(tmp_path)
    n.commit()
    ref = pypsa.Network(path)
    assert not n.c["Load"].dynamic.p_set.empty
    assert_networks_equal(n, ref)


def test_empty_commit_writes_revision(tmp_path):
    path, n = _store_network(tmp_path)
    rid = n.commit(meta={"empty": True})
    revisions = list(Workspace.process().store(str(path)).revisions())
    assert len(revisions) == 2
    assert pypsa.Network(path).revision == rid


def test_sibling_commits_then_open_raises(tmp_path):
    path, _ = _store_network(tmp_path)
    a, b = pypsa.Network(path), pypsa.Network(path)
    a.commit(meta={"scenario": "a"})
    b.commit(meta={"scenario": "b"})
    with pytest.raises(ValueError, match="scenario") as exc:
        pypsa.Network(path)
    assert '"a"' in str(exc.value)
    assert '"b"' in str(exc.value)


def test_commit_refusals(tmp_path):
    ac_dc = pypsa.examples.ac_dc_meshed()
    nc = tmp_path / "n.nc"
    ac_dc.export_to_netcdf(nc)
    plain = tmp_path / "plain"
    Workspace.process().write(ac_dc.to_datarecord(), str(plain))
    for n in (pypsa.Network(nc), pypsa.Network()):
        with pytest.raises(ValueError, match="export_to_datarecord"):
            n.commit()
    plain_n = pypsa.Network(plain)
    with pytest.raises(ValueError, match="plain record folder.*export_to_datarecord"):
        plain_n.commit()
    store_path, _ = _store_network(tmp_path / "s")
    rev = next(iter(Workspace.process().store(str(store_path)).revisions()))
    rec = pypsa.Network.from_datarecord(rev.record)
    with pytest.raises(ValueError, match="export_to_datarecord"):
        rec.commit()
    sel = pypsa.Network.from_datarecord(rev.record.select(bus=["Frankfurt"]))
    with pytest.raises(ValueError, match="no store revision"):
        sel.commit()


def test_reading_after_commit_survives_closing_the_parent_record(tmp_path):
    path, n = _store_network(tmp_path)
    parent_record = n.record
    n.commit()
    parent_record.close()
    assert not n.c["Load"].dynamic.p_set.empty
    assert_networks_equal(n, pypsa.Network(path))


def test_reading_after_commit_rebuilds_settled_rows(tmp_path):
    path, n = _store_network(tmp_path)
    _ = n.c["Generator"].static  # settles rows over the parent's relations
    parent_record = n.record
    n.commit()
    parent_record.close()
    assert not n.c["Load"].dynamic.p_set.empty
    assert not n.c["Generator"].dynamic.p_max_pu.empty
