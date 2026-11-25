# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests of the benchmark harness in `test/benchmarks`."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCH = ROOT / "test" / "benchmarks"
# `test` is a namespace package here and the stdlib ships its own, so the
# harness is a script and its modules are imported from its directory.
sys.path.insert(0, str(BENCH))

import relative  # noqa: E402
import run  # noqa: E402
from relative import Side, interleave, relative_results  # noqa: E402
from run import compare  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_host(monkeypatch):
    """Keep the load retry out of tests that do not exercise it."""
    monkeypatch.setattr(os, "getloadavg", lambda: (0.5, 0.5, 0.5))


no_wait4 = pytest.mark.skipif(sys.platform == "win32", reason="harness needs os.wait4")


def _results(w3: float, w4: float = 1.0, w1: float = 1.0) -> dict:
    return {
        "snapshots": 8760,
        "results": {
            "w1_record_one_attribute": {"wall_s": w1, "peak_rss_mb": 1.0},
            "w3_create_model_scigrid": {"wall_s": w3, "peak_rss_mb": 1.0},
            "w4_add_10k": {"wall_s": w4, "peak_rss_mb": 1.0},
        },
    }


def test_compare_fails_above_five_percent_on_w3():
    _, failed, _ = compare(_results(1.10), _results(1.0))
    assert failed == ["w3_create_model_scigrid"]


def test_compare_passes_below_five_percent_on_w3():
    rows, failed, _ = compare(_results(1.04), _results(1.0))
    assert failed == []
    assert rows["w3_create_model_scigrid"] == pytest.approx(1.04)


def test_compare_reports_w1_without_gating():
    rows, failed, _ = compare(_results(1.0, w1=9.0), _results(1.0))
    assert failed == []
    assert rows["w1_record_one_attribute"] == pytest.approx(9.0)


def _spec_results(record_wall: float) -> dict:
    def res(wall, rss=1.0, attr=None):
        out = {"wall_s": wall, "peak_rss_mb": rss}
        if attr is not None:
            out["attribute_mb"] = attr
        return out

    return {
        "snapshots": 8760,
        "results": {
            "w1_record_one_attribute": res(record_wall, rss=40.0, attr=10.0),
            "w1_nc_one_attribute": res(10.0),
            "w1_record_eight": res(record_wall * 8),
            "w1_nc_eight": res(80.0),
            "w2_record_open_touch_export_nc": res(record_wall),
            "w2_nc_import_export": res(2.0),
        },
    }


def test_compare_reports_spec_ratios_without_failing():
    data = _spec_results(4.0)
    _, failed, spec = compare(data, data)
    assert failed == []
    assert spec["w1_record_one_attribute"] == pytest.approx((0.4, 0.2))
    assert spec["w1_record_eight"] == pytest.approx((0.4, 0.2))
    assert "w1_record_one_attribute peak_rss/attribute (net of import)" not in spec
    assert spec["w2_record_open_touch_export_nc"] == pytest.approx((2.0, 1.5))


def test_compare_memory_ratio_is_net_of_the_import():
    data = _spec_results(4.0)
    data["results"]["w0_import_pypsa"] = {"wall_s": 1.0, "peak_rss_mb": 30.0}
    _, _, spec = compare(data, data)
    assert spec["w1_record_one_attribute peak_rss/attribute (net of import)"] == (
        pytest.approx((1.0, 3.0))
    )


def test_compare_memory_ratio_uses_import_rss_stored_beside_results():
    data = _spec_results(4.0)
    data["import_peak_rss_mb"] = 30.0
    _, _, spec = compare(data, data)
    assert spec["w1_record_one_attribute peak_rss/attribute (net of import)"] == (
        pytest.approx((1.0, 3.0))
    )


def test_compare_reports_w5_against_twice_the_w1_wall():
    data = _spec_results(4.0)
    data["results"]["w5_record_writeback"] = {"wall_s": 6.0, "peak_rss_mb": 1.0}
    _, _, spec = compare(data, data)
    assert spec["w5_record_writeback"] == pytest.approx((1.5, 2.0))


def test_compare_memory_ratio_is_net_of_open_only():
    data = _spec_results(4.0)
    data["results"]["w0_import_pypsa"] = {"wall_s": 1.0, "peak_rss_mb": 30.0}
    data["results"]["w1_record_open_only"] = {"wall_s": 1.0, "peak_rss_mb": 35.0}
    _, _, spec = compare(data, data)
    key = "w1_record_one_attribute peak_rss/attribute (net of open-only)"
    assert spec[key] == pytest.approx((0.5, 3.0))
    assert "w1_record_one_attribute peak_rss/attribute (net of import)" not in spec


def test_compare_skips_spec_ratios_for_missing_workloads():
    _, _, spec = compare(_results(1.0), _results(1.0))
    assert spec == {}


def test_check_exit_code(tmp_path):
    cur, base = tmp_path / "cur.json", tmp_path / "base.json"
    base.write_text(json.dumps(_results(1.0)))
    for w3, code in [(1.10, 1), (1.04, 0)]:
        cur.write_text(json.dumps(_results(w3)))
        out = subprocess.run(
            [sys.executable, str(BENCH / "run.py"), "--compare", str(cur)]
            + ["--baseline", str(base)],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        assert out.returncode == code, out.stderr


BASE_SIDE = Side("base", "base-python", Path("/base"))
HEAD_SIDE = Side("head", "head-python", None)
W3 = "w3_create_model_scigrid"
W1 = "w1_record_one_attribute"


def _fake_run(walls: dict, fail: frozenset = frozenset(), calls: list | None = None):
    """Return a child runner timing `walls[(label, name)]`, failing on `fail`."""
    labels = {"base-python": "base", "head-python": "head"}

    def run(name, paths, python, cwd):
        label = labels[python]
        if calls is not None:
            calls.append((label, name))
        if (label, name) in fail:
            msg = f"workload {name} failed"
            raise RuntimeError(msg)
        return walls[label, name], 1.0, None

    return run


def _walls(base_w0=1.0, head_w0=1.0, base=2.0, head=2.0, name=W3) -> dict:
    return {
        ("base", "w0_import_pypsa"): base_w0,
        ("head", "w0_import_pypsa"): head_w0,
        ("base", name): base,
        ("head", name): head,
    }


def test_interleave_alternates_base_and_head_per_round():
    calls: list = []
    interleave([W3], {}, BASE_SIDE, HEAD_SIDE, 2, run=_fake_run(_walls(), calls=calls))
    round_ = [
        ("base", "w0_import_pypsa"),
        ("head", "w0_import_pypsa"),
        ("base", W3),
        ("head", W3),
    ]
    assert calls == round_ * 2


@pytest.mark.parametrize(("head", "failed"), [(2.1, [W3]), (2.04, [])])
def test_relative_gates_w3_at_five_percent(head, failed):
    # net of W0 (1.0) the ratio is 1.10 and 1.04
    runs = interleave(
        [W3], {}, BASE_SIDE, HEAD_SIDE, 1, run=_fake_run(_walls(head=head))
    )
    res = relative_results(runs)
    assert res["failed"] == failed


def test_relative_subtracts_each_sides_own_w0():
    walls = _walls(base_w0=1.0, head_w0=3.0, base=2.0, head=4.0)
    runs = interleave([W3], {}, BASE_SIDE, HEAD_SIDE, 3, run=_fake_run(walls))
    res = relative_results(runs)
    assert res["base"][W3]["wall_s"] == pytest.approx(1.0)
    assert res["head"][W3]["wall_s"] == pytest.approx(1.0)
    assert res["ratios"][W3] == pytest.approx(1.0)


def test_relative_takes_min():
    # each round: base W0, head W0, base W3, head W3
    seq = iter([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 9.0, 3.0, 1.0, 1.0, 3.0, 2.0])

    def run(name, paths, python, cwd):
        return next(seq), 1.0, None

    runs = interleave([W3], {}, BASE_SIDE, HEAD_SIDE, 3, run=run)
    res = relative_results(runs)
    assert res["ratios"][W3] == pytest.approx(1.0)


def _loads(monkeypatch, loads):
    seq = iter(loads)
    monkeypatch.setattr(os, "getloadavg", lambda: (next(seq), 0.0, 0.0))


def _counting_run(calls):
    def run(name, paths, python, cwd):
        calls.append(name)
        return 1.0, 1.0, None

    return run


def test_measure_does_not_retry_at_the_load_limit(monkeypatch):
    _loads(monkeypatch, [8.0])
    calls: list = []
    runs, loads = run.measure("x", {}, "py", None, run=_counting_run(calls))
    assert len(calls) == 1
    assert len(runs) == 1
    assert loads == [8.0]


def test_measure_retries_once_and_is_clean_when_the_retry_is_quiet(monkeypatch):
    _loads(monkeypatch, [9.0, 2.0])
    calls: list = []
    runs, loads = run.measure("x", {}, "py", None, run=_counting_run(calls))
    assert len(calls) == 2
    assert len(runs) == 2
    assert loads == [9.0, 2.0]
    assert run.high_load(loads) is False


def test_measure_marks_a_retry_that_is_still_loaded(monkeypatch):
    _loads(monkeypatch, [9.0, 9.5, 9.9])
    calls: list = []
    runs, loads = run.measure("x", {}, "py", None, run=_counting_run(calls))
    assert len(calls) == 2
    assert loads == [9.0, 9.5]
    assert run.high_load(loads) is True


def test_high_load_is_false_for_a_single_run():
    assert run.high_load([2.0]) is False


class _NoPaths:
    @staticmethod
    def to_dict():
        return {}


def _patch_paths(monkeypatch):
    import fixtures

    monkeypatch.setattr(fixtures, "build_paths", lambda root, snapshots: _NoPaths())


def test_run_all_is_round_robin(tmp_path, monkeypatch):
    _patch_paths(monkeypatch)
    calls: list = []
    names = ["w1_record_one_attribute", "w1_nc_one_attribute"]
    data = run.run_all(names, 1, 2, tmp_path, run=_counting_run(calls))
    assert calls == ["w0_import_pypsa", *names] * 2
    entry = data["results"]["w1_record_one_attribute"]
    assert entry["load_avg"] == [0.5, 0.5]
    assert "high_load" not in entry


def test_run_all_marks_high_load(tmp_path, monkeypatch):
    _patch_paths(monkeypatch)
    monkeypatch.setattr(os, "getloadavg", lambda: (9.0, 0.0, 0.0))
    data = run.run_all(["w1_record_eight"], 1, 1, tmp_path, run=_counting_run([]))
    assert data["results"]["w1_record_eight"]["high_load"] is True


def test_table_shows_load_and_marks_high_load():
    data = {
        "snapshots": 1,
        "results": {
            "a": {"wall_s": 1.0, "peak_rss_mb": 1.0, "load_avg": [1.0, 3.5]},
            "b": {
                "wall_s": 1.0,
                "peak_rss_mb": 1.0,
                "load_avg": [9.0, 9.5],
                "high_load": True,
            },
        },
    }
    lines = run.table(data).splitlines()
    assert "load" in lines[2]
    row_a = next(x for x in lines if x.startswith("| a "))
    row_b = next(x for x in lines if x.startswith("| b "))
    assert "3.5" in row_a
    assert "*" not in row_a
    assert "*" in row_b
    assert "high load" in lines[-1]


def test_relative_base_failure_on_w1_is_not_applicable():
    walls = _walls(name=W1)
    run = _fake_run(walls, fail=frozenset({("base", W1)}))
    res = relative_results(interleave([W1], {}, BASE_SIDE, HEAD_SIDE, 1, run=run))
    assert res["ratios"][W1] is None
    assert res["failed"] == []
    assert W1 in res["head"]


def test_relative_base_failure_on_w3_raises():
    run = _fake_run(_walls(), fail=frozenset({("base", W3)}))
    with pytest.raises(RuntimeError, match=W3):
        interleave([W3], {}, BASE_SIDE, HEAD_SIDE, 1, run=run)


def test_checkout_removes_worktree_when_sync_fails(monkeypatch):
    calls: list[list[str]] = []

    def fake(cmd, **kwargs):
        calls.append(cmd)
        code = 1 if cmd[:2] == ["uv", "sync"] else 0
        return subprocess.CompletedProcess(cmd, code, "", "no lockfile")

    monkeypatch.setattr(relative.subprocess, "run", fake)
    with (
        pytest.raises(RuntimeError, match="deadbeef"),
        relative.checkout("deadbeef"),
    ):
        pass
    added = next(c for c in calls if c[1:3] == ["worktree", "add"])
    removed = next(c for c in calls if c[1:3] == ["worktree", "remove"])
    assert removed[-1] == added[-2]
    assert calls[-1] is removed
    assert not any("prune" in c for c in calls)


def test_check_with_compare_is_a_parser_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run.main(["--check", "--compare", str(tmp_path / "x.json")])
    assert exc.value.code == 2


def test_base_without_check_is_a_parser_error():
    with pytest.raises(SystemExit) as exc:
        run.main(["--base", "HEAD"])
    assert exc.value.code == 2


def test_base_follows_the_pr_base_branch(monkeypatch):
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    assert run.resolve_base(None) == "origin/master"
    monkeypatch.setenv("GITHUB_BASE_REF", "v2")
    assert run.resolve_base(None) == "origin/v2"
    assert run.resolve_base("abc123") == "abc123"


@pytest.mark.parametrize(("failed", "code"), [([W3], 1), ([], 0)])
def test_check_exit_code_follows_the_relative_gate(monkeypatch, failed, code):
    def fake(names, snapshots, repeat, root, base_rev, head_rev):
        assert base_rev == "origin/master"
        res = {"wall_s": 1.0, "peak_rss_mb": 1.0}
        return {
            "snapshots": snapshots,
            "repeat": repeat,
            "revs": {"base": "a" * 40, "head": "b" * 40},
            "load_avg": {"start": [1.0, 1.0, 1.0], "end": [1.0, 1.0, 1.0]},
            "import_peak_rss_mb": None,
            "base": {W3: res},
            "head": {W3: res},
            "ratios": {W3: 1.1 if failed else 1.0},
            "failed": failed,
        }

    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    monkeypatch.setattr(run, "run_relative", fake)
    assert run.main(["--check", "--workloads", W3]) == code


def _worktrees() -> str:
    return subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.slow
@no_wait4
def test_check_against_head_end_to_end(tmp_path):
    before = _worktrees()
    out = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, str(BENCH / "run.py"), "--check", "--base", "HEAD"]
        + ["--repeat", "1", "--workloads", "w0_import_pypsa"]
        + ["--snapshots", "24", "--root", str(tmp_path / "root"), "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert _worktrees() == before
    data = json.loads(out.read_text())
    assert data["revs"]["head"].startswith(data["revs"]["base"])
    assert len(data["load_avg"]["start"]) == 3
    assert data["ratios"]["w0_import_pypsa"] > 0


@pytest.mark.slow
@no_wait4
def test_carbon_management_fixture(tmp_path, monkeypatch):
    pytest.importorskip("datarecord")
    from platformdirs import user_cache_dir

    import pypsa

    # conftest serves the in-repo examples, which lack carbon_management
    monkeypatch.setattr(
        pypsa.examples,
        "_cache_root",
        lambda: Path(user_cache_dir("pypsa")) / "examples",
    )
    from fixtures import carbon_management_record

    from pypsa.network.names import EXEMPT_TYPES, name_owners

    nc, record = carbon_management_record(tmp_path, 48)
    assert nc.exists()
    assert record.is_dir()
    n = pypsa.Network(nc)
    assert len(n.snapshots) == 48
    for c in n.components:
        if c.name not in EXEMPT_TYPES:
            assert name_owners(n, c.static.index, exclude=c.name) == {}, c.name


@pytest.mark.slow
@no_wait4
def test_smoke_every_workload(tmp_path):
    pytest.importorskip("datarecord")
    from workloads import WORKLOADS

    out = tmp_path / "out.json"
    subprocess.run(
        [sys.executable, str(BENCH / "run.py"), "--snapshots", "24"]
        + ["--repeat", "1", "--root", str(tmp_path / "root"), "--out", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    results = json.loads(out.read_text())["results"]
    assert set(results) == set(WORKLOADS)
    for res in results.values():
        assert res["wall_s"] > 0
        assert res["peak_rss_mb"] > 0


@pytest.mark.slow
@no_wait4
def test_breakdown_reports_every_step(tmp_path):
    pytest.importorskip("datarecord")
    out = tmp_path / "breakdown.json"
    proc = subprocess.run(
        [sys.executable, str(BENCH / "breakdown.py"), "--snapshots", "24"]
        + ["--top", "5", "--root", str(tmp_path / "root"), "--out", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    steps = json.loads(out.read_text())["steps"]
    assert set(steps) == {"open", "static_read", "dynamic_read", "export", "writeback"}
    for res in steps.values():
        assert res["wall_s"] > 0
        assert 0 < res["share"] <= 1
        assert res["load_avg"] >= 0
        assert res["top_self"]
        assert res["top_cumulative"]
        assert {"function", "ncalls", "tottime", "cumtime", "self_share"} <= set(
            res["top_self"][0]
        )
    assert "| step |" in proc.stdout
