# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests of the benchmark harness in `test/benchmarks`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BENCH = ROOT / "test" / "benchmarks"
# `test` is a namespace package here and the stdlib ships its own, so the
# harness is a script and its modules are imported from its directory.
sys.path.insert(0, str(BENCH))

from run import compare  # noqa: E402

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
    _, failed = compare(_results(1.10), _results(1.0))
    assert failed == ["w3_create_model_scigrid"]


def test_compare_passes_below_five_percent_on_w3():
    rows, failed = compare(_results(1.04), _results(1.0))
    assert failed == []
    assert rows["w3_create_model_scigrid"] == pytest.approx(1.04)


def test_compare_reports_w1_without_gating():
    rows, failed = compare(_results(1.0, w1=9.0), _results(1.0))
    assert failed == []
    assert rows["w1_record_one_attribute"] == pytest.approx(9.0)


def test_check_exit_code(tmp_path):
    cur, base = tmp_path / "cur.json", tmp_path / "base.json"
    base.write_text(json.dumps(_results(1.0)))
    for w3, code in [(1.10, 1), (1.04, 0)]:
        cur.write_text(json.dumps(_results(w3)))
        out = subprocess.run(
            [sys.executable, str(BENCH / "run.py"), "--compare", str(cur)]
            + ["--check", str(base)],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        assert out.returncode == code, out.stderr


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
