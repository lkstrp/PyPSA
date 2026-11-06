# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Benchmark CLI: `python test/benchmarks/run.py`. See README.md."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

GATED_PREFIXES = ("w3_", "w4_")
TOLERANCE = 1.05
BASE = "w0_import_pypsa"
# These never import pypsa, so there is no import time to subtract.
NO_PYPSA = ("w1_record_one_attribute", "w1_record_eight")


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def host() -> dict:
    """Describe the machine and package versions."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "pypsa": _version("pypsa"),
        "datarecord": _version("datarecord"),
        "duckdb": _version("duckdb"),
    }


def compare(
    current: dict, baseline: dict, tolerance: float = TOLERANCE
) -> tuple[dict[str, float], list[str]]:
    """Compare two result files by wall time.

    Returns
    -------
    ratios : dict[str, float]
        Current over baseline wall time per workload present in both.
    failed : list[str]
        Gated workloads (W3, W4) whose ratio exceeds `tolerance`.

    """
    ratios = {
        name: res["wall_s"] / baseline["results"][name]["wall_s"]
        for name, res in current["results"].items()
        if name in baseline["results"]
    }
    failed = [
        name
        for name, ratio in ratios.items()
        if name.startswith(GATED_PREFIXES) and ratio > tolerance
    ]
    return ratios, failed


def _run_child(name: str, paths: dict) -> tuple[float, float, float | None]:
    """Run one workload in a fresh interpreter; return wall s, peak RSS MB, attribute MB."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", name]
    cmd += ["--paths", json.dumps(paths)]
    start = time.perf_counter()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:  # noqa: S603
        out = proc.stdout.read() if proc.stdout else ""
        # wait4 reports this child's own usage; RUSAGE_CHILDREN is a running maximum
        _, status, usage = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - start
        proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode:
        msg = f"workload {name} failed with exit code {proc.returncode}"
        raise RuntimeError(msg)
    scale = 1 if sys.platform == "darwin" else 1024
    peak = usage.ru_maxrss * scale / 2**20
    attribute = json.loads(out.strip().splitlines()[-1]).get("attribute_mb")
    return wall, peak, attribute


def run_all(names: list[str], snapshots: int, repeat: int, root: Path) -> dict:
    """Run the workloads and return the result dict."""
    from fixtures import build_paths  # noqa: PLC0415

    paths = build_paths(root, snapshots).to_dict()
    results: dict[str, dict] = {}
    for name in [BASE, *[n for n in names if n != BASE]]:
        runs = [_run_child(name, paths) for _ in range(repeat)]
        entry: dict = {
            "wall_s": statistics.median(r[0] for r in runs),
            "peak_rss_mb": max(r[1] for r in runs),
        }
        if runs[0][2] is not None:
            entry["attribute_mb"] = runs[0][2]
        results[name] = entry
        print(f"{name}: {entry}", file=sys.stderr, flush=True)
    base = results[BASE]["wall_s"]
    for name, entry in results.items():
        if name != BASE and name not in NO_PYPSA:
            entry["wall_s"] = max(entry["wall_s"] - base, 1e-3)
    if BASE not in names:
        del results[BASE]
    return {"host": host(), "snapshots": snapshots, "results": results}


def table(data: dict) -> str:
    """Render the results as a Markdown table."""
    lines = [
        f"Snapshots: {data['snapshots']}. Wall time minus `import pypsa` where the workload imports it.",
        "",
        "| workload | wall_s | peak_rss_mb | attribute_mb |",
        "|---|---:|---:|---:|",
    ]
    for name, res in data["results"].items():
        attr = f"{res['attribute_mb']:.1f}" if "attribute_mb" in res else ""
        lines.append(
            f"| {name} | {res['wall_s']:.2f} | {res['peak_rss_mb']:.0f} | {attr} |"
        )
    return "\n".join(lines)


def _one(name: str, paths: dict) -> None:
    from fixtures import Paths  # noqa: PLC0415
    from workloads import WORKLOADS  # noqa: PLC0415

    value = WORKLOADS[name].fn(Paths(**paths))
    attribute = None if value is None else value / 2**20
    print(json.dumps({"attribute_mb": attribute}))


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(prog="python test/benchmarks/run.py")
    parser.add_argument("--snapshots", type=int, default=8760)
    parser.add_argument("--workloads", help="comma-separated names, default all")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path, help="baseline JSON to compare with")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache" / "pypsa" / "benchmarks",
    )
    parser.add_argument("--one", help="internal: run one workload in this process")
    parser.add_argument("--paths", help="internal: fixture paths as JSON")
    parser.add_argument("--compare", type=Path, help="compare this JSON, run nothing")
    args = parser.parse_args(argv)

    if args.one:
        _one(args.one, json.loads(args.paths))
        return 0

    if args.compare:
        data = json.loads(args.compare.read_text())
    else:
        from workloads import WORKLOADS  # noqa: PLC0415

        names = args.workloads.split(",") if args.workloads else list(WORKLOADS)
        unknown = set(names) - set(WORKLOADS)
        if unknown:
            parser.error(f"unknown workloads: {sorted(unknown)}")
        data = run_all(names, args.snapshots, args.repeat, args.root)
        print(table(data))
        if args.out:
            args.out.write_text(json.dumps(data, indent=2) + "\n")

    if args.check:
        baseline = json.loads(args.check.read_text())
        ratios, failed = compare(data, baseline)
        print("\n| workload | ratio to baseline |\n|---|---:|")
        for name, ratio in ratios.items():
            gate = " (gated)" if name.startswith(GATED_PREFIXES) else ""
            print(f"| {name}{gate} | {ratio:.2f} |")
        if failed:
            print(f"\nFAILED (above {TOLERANCE}x): {', '.join(failed)}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
