# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Benchmark CLI: `python test/benchmarks/run.py`. See README.md."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

GATED_PREFIXES = ("w3_", "w4_")
TOLERANCE = 1.05
BASE = "w0_import_pypsa"
# (record workload, `.nc` workload, spec bound on their wall-time ratio)
SPEC_PAIRS = (
    ("w1_record_one_attribute", "w1_nc_one_attribute", 0.2),
    ("w1_record_eight", "w1_nc_eight", 0.2),
    ("w2_record_open_touch_export_nc", "w2_nc_import_export", 1.5),
    ("w5_record_writeback", "w1_record_one_attribute", 2.0),
)
# A run that starts above this 1-minute load average is repeated once.
LOAD_LIMIT = 8.0
MEMORY_BOUND = 3.0
OPEN_ONLY = "w1_record_open_only"


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
) -> tuple[dict[str, float], list[str], dict[str, tuple[float, float]]]:
    """Compare two result files by wall time.

    Returns
    -------
    ratios : dict[str, float]
        Current over baseline wall time per workload present in both.
    failed : list[str]
        Gated workloads (W3, W4) whose ratio exceeds `tolerance`.
    spec : dict[str, tuple[float, float]]
        Informational W1, W2 and W5 rows of `current` alone: the measured
        ratio and its spec bound. Never gates. The RSS row nets out the peak
        RSS of `w1_record_open_only` when present, else that of the `pypsa`
        import (`import_peak_rss_mb`, else W0's row).

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
    return ratios, failed, spec_rows(current)


def spec_rows(current: dict) -> dict[str, tuple[float, float]]:
    """Return the informational W1, W2 and W5 rows of one result dict, never gated.

    The W5 row is its wall over the W1 one-attribute wall. The RSS row nets
    out `w1_record_open_only` when present, else the `pypsa` import.
    """
    results = current["results"]
    spec: dict[str, tuple[float, float]] = {}
    for record, nc, bound in SPEC_PAIRS:
        if record in results and nc in results:
            spec[record] = (results[record]["wall_s"] / results[nc]["wall_s"], bound)
    one = results.get("w1_record_one_attribute", {})
    open_only = results.get(OPEN_ONLY, {}).get("peak_rss_mb")
    if open_only is not None:
        base_rss, net = open_only, "open-only"
    else:
        base_rss = current.get(
            "import_peak_rss_mb", results.get(BASE, {}).get("peak_rss_mb")
        )
        net = "import"
    if "attribute_mb" in one and base_rss is not None:
        spec[f"w1_record_one_attribute peak_rss/attribute (net of {net})"] = (
            (one["peak_rss_mb"] - base_rss) / one["attribute_mb"],
            MEMORY_BOUND,
        )
    return spec


Run = tuple[float, float, float | None]


def _run_child(
    name: str, paths: dict, python: str = sys.executable, cwd: Path | None = None
) -> tuple[float, float, float | None]:
    """Run one workload in a fresh interpreter; return wall s, peak RSS MB, attribute MB.

    `python` picks the interpreter, and with it the installed `pypsa`; the
    harness and workloads are always this checkout's.
    """
    cmd = [python, str(Path(__file__).resolve()), "--one", name]
    cmd += ["--paths", json.dumps(paths)]
    start = time.perf_counter()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, cwd=cwd) as proc:  # noqa: S603
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


def high_load(loads: list[float]) -> bool:
    """Return whether a retried measurement still started above `LOAD_LIMIT`."""
    return len(loads) > 1 and loads[-1] > LOAD_LIMIT


def measure(
    name: str,
    paths: dict,
    python: str,
    cwd: Path | None,
    run: Callable[..., Run] = _run_child,
) -> tuple[list[Run], list[float]]:
    """Run one workload, once more when it started above `LOAD_LIMIT`.

    Returns the runs and the 1-minute load average read before each started.
    """
    runs: list[Run] = []
    loads: list[float] = []
    for _ in range(2):
        loads.append(os.getloadavg()[0])
        runs.append(run(name, paths, python=python, cwd=cwd))
        if loads[-1] <= LOAD_LIMIT:
            break
    return runs, loads


def summarize(runs: list[tuple], net: float = 0.0) -> dict:
    """Reduce the runs of one workload to a result entry.

    Each run is `(wall, peak_rss, attribute_mb, start_load, high_load)`. Wall
    is the min, less `net` unless that is zero (W0 itself).
    """
    wall = min(r[0] for r in runs)
    entry: dict = {
        "wall_s": max(wall - net, 1e-3) if net else wall,
        "peak_rss_mb": max(r[1] for r in runs),
    }
    if runs[0][2] is not None:
        entry["attribute_mb"] = runs[0][2]
    entry["load_avg"] = [r[3] for r in runs]
    if any(r[4] for r in runs):
        entry["high_load"] = True
    return entry


def tag(runs: list[Run], loads: list[float]) -> list[tuple]:
    """Attach the start load and the high-load mark to the runs of one `measure`."""
    flag = high_load(loads)
    return [(*r, load, flag) for r, load in zip(runs, loads, strict=True)]


def run_all(
    names: list[str],
    snapshots: int,
    repeat: int,
    root: Path,
    run: Callable[..., Run] = _run_child,
) -> dict:
    """Run the workloads in `repeat` rounds and return the result dict.

    A round runs W0, then every workload once, so related workloads sit next
    to each other in time.
    """
    from fixtures import build_paths  # noqa: PLC0415

    paths = build_paths(root, snapshots).to_dict()
    order = [BASE, *[n for n in names if n != BASE]]
    runs: dict[str, list[tuple]] = {name: [] for name in order}
    for _ in range(repeat):
        for name in order:
            measured = measure(name, paths, sys.executable, None, run=run)
            runs[name].extend(tag(*measured))
    results: dict[str, dict] = {BASE: summarize(runs[BASE])}
    w0 = results[BASE]["wall_s"]
    for name in order[1:]:
        results[name] = summarize(runs[name], net=w0)
    for name, entry in results.items():
        print(f"{name}: {entry}", file=sys.stderr, flush=True)
    import_rss = results[BASE]["peak_rss_mb"]
    if BASE not in names:
        del results[BASE]
    return {
        "host": host(),
        "snapshots": snapshots,
        "import_peak_rss_mb": import_rss,
        "results": results,
    }


def table(data: dict) -> str:
    """Render the results as a Markdown table."""
    lines = [
        f"Snapshots: {data['snapshots']}. Wall time minus `import pypsa` where the workload imports it.",
        "",
        "| workload | wall_s | peak_rss_mb | attribute_mb | load |",
        "|---|---:|---:|---:|---:|",
    ]
    marked = False
    for name, res in data["results"].items():
        attr = f"{res['attribute_mb']:.1f}" if "attribute_mb" in res else ""
        load = f"{max(res['load_avg']):.1f}" if res.get("load_avg") else ""
        mark = "*" if res.get("high_load") else ""
        marked = marked or bool(mark)
        lines.append(
            f"| {name} | {res['wall_s']:.2f}{mark} | {res['peak_rss_mb']:.0f} | {attr} | {load} |"
        )
    if marked:
        lines += [
            "",
            f"\\* high load: a repeat still started above load {LOAD_LIMIT:g} after one retry.",
        ]
    return "\n".join(lines)


def _one(name: str, paths: dict) -> None:
    from fixtures import Paths  # noqa: PLC0415
    from workloads import WORKLOADS  # noqa: PLC0415

    value = WORKLOADS[name].fn(Paths(**paths))
    attribute = None if value is None else value / 2**20
    print(json.dumps({"attribute_mb": attribute}))


def resolve_base(rev: str | None) -> str:
    """Return `rev`, else the PR's base branch on `origin`, else `origin/master`."""
    if rev:
        return rev
    ref = os.environ.get("GITHUB_BASE_REF")
    return f"origin/{ref}" if ref else "origin/master"


def run_relative(
    names: list[str],
    snapshots: int,
    repeat: int,
    root: Path,
    base_rev: str,
    head_rev: str | None,
) -> dict:
    """Measure `names` alternately on `base_rev` and the head, and return the result dict.

    The head is this working tree unless `head_rev` is given, in which case it
    is checked out like the base.
    """
    from contextlib import ExitStack  # noqa: PLC0415

    from fixtures import build_paths  # noqa: PLC0415
    from relative import (  # noqa: PLC0415
        checkout,
        current_side,
        interleave,
        is_dirty,
        relative_results,
        rev_sha,
    )

    revs = {"base": rev_sha(base_rev), "head": rev_sha(head_rev or "HEAD")}
    if head_rev is None and is_dirty():
        revs["head"] += "-dirty"
    load_start = os.getloadavg()
    print(f"load_avg at start: {_load(load_start)}", file=sys.stderr, flush=True)
    # absolute, since the base side's children run inside its worktree
    paths = build_paths(root.resolve(), snapshots).to_dict()
    with ExitStack() as stack:
        base = stack.enter_context(checkout(revs["base"]))
        head = (
            stack.enter_context(checkout(revs["head"], label="head"))
            if head_rev
            else current_side()
        )
        runs = interleave(names, paths, base, head, repeat)
    load_end = os.getloadavg()
    print(f"load_avg at end: {_load(load_end)}", file=sys.stderr, flush=True)
    result = relative_results(runs)
    import_rss = result["head"][BASE]["peak_rss_mb"]
    if BASE not in names:
        for key in ("base", "head", "ratios"):
            result[key].pop(BASE, None)
    return {
        "host": host(),
        "snapshots": snapshots,
        "repeat": repeat,
        "revs": revs,
        "load_avg": {"start": list(load_start), "end": list(load_end)},
        "import_peak_rss_mb": import_rss,
        **result,
    }


def _load(avg: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{x:.2f}" for x in avg)


def _print_absolute(data: dict, baseline: dict) -> int:
    ratios, failed, spec = compare(data, baseline)
    print("\n| workload | ratio to baseline |\n|---|---:|")
    for name, ratio in ratios.items():
        gate = " (gated)" if name.startswith(GATED_PREFIXES) else ""
        print(f"| {name}{gate} | {ratio:.2f} |")
    _print_spec(spec)
    if failed:
        print(f"\nFAILED (above {TOLERANCE}x): {', '.join(failed)}")
        return 1
    return 0


def _print_spec(spec: dict[str, tuple[float, float]]) -> None:
    if spec:
        print("\n| workload (informational) | ratio | spec bound |\n|---|---:|---:|")
        for name, (ratio, bound) in spec.items():
            print(f"| {name} | {ratio:.2f} | {bound:g} |")


def _print_relative(data: dict) -> int:
    revs, ratios = data["revs"], data["ratios"]
    print(
        f"Base {revs['base'][:10]}, head {revs['head'][:10]}"
        f"{'-dirty' if revs['head'].endswith('-dirty') else ''}, "
        f"{data['repeat']} rounds, {data['snapshots']} snapshots. "
        f"load_avg {_load(data['load_avg']['start'])} -> {_load(data['load_avg']['end'])}."
    )

    def rows(names: list[str]) -> None:
        print(
            "| workload | base wall_s | head wall_s | head/base |\n|---|---:|---:|---:|"
        )
        for name in names:
            base = data["base"].get(name, {}).get("wall_s")
            head = data["head"].get(name, {}).get("wall_s")
            cells = [
                "n/a" if v is None else f"{v:.2f}" for v in (base, head, ratios[name])
            ]
            print(f"| {name} | {' | '.join(cells)} |")

    gated = [n for n in ratios if n.startswith(GATED_PREFIXES)]
    other = [n for n in ratios if n not in gated]
    if gated:
        print(f"\nGated (fails above {TOLERANCE}):\n")
        rows(gated)
    if other:
        print("\nInformational:\n")
        rows(other)
    _print_spec(
        spec_rows(
            {"results": data["head"], "import_peak_rss_mb": data["import_peak_rss_mb"]}
        )
    )
    if data["failed"]:
        print(f"\nFAILED (above {TOLERANCE}x): {', '.join(data['failed'])}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(prog="python test/benchmarks/run.py")
    parser.add_argument("--snapshots", type=int, default=8760)
    parser.add_argument("--workloads", help="comma-separated names, default all")
    parser.add_argument(
        "--repeat", type=int, default=3, help="runs, or rounds with --check"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="gate head against --base, both measured alternately in this run",
    )
    parser.add_argument(
        "--base",
        help="base rev for --check, default origin/$GITHUB_BASE_REF or origin/master",
    )
    parser.add_argument("--head", help="head rev for --check, default the working tree")
    parser.add_argument(
        "--baseline", type=Path, help="result JSON to compare absolute wall times with"
    )
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
    if args.check and args.compare:
        parser.error("--check measures a run; use --compare with --baseline")
    if (args.base or args.head) and not args.check:
        parser.error("--base and --head need --check")
    if args.check and args.baseline:
        parser.error("--check compares against --base, not --baseline")

    if args.compare:
        data = json.loads(args.compare.read_text())
    else:
        from workloads import WORKLOADS  # noqa: PLC0415

        names = args.workloads.split(",") if args.workloads else list(WORKLOADS)
        unknown = set(names) - set(WORKLOADS)
        if unknown:
            parser.error(f"unknown workloads: {sorted(unknown)}")
        if args.check:
            data = run_relative(
                names,
                args.snapshots,
                args.repeat,
                args.root,
                resolve_base(args.base),
                args.head,
            )
        else:
            data = run_all(names, args.snapshots, args.repeat, args.root)
            print(table(data))
        if args.out:
            args.out.write_text(json.dumps(data, indent=2) + "\n")

    if args.check:
        return _print_relative(data)
    if args.baseline:
        return _print_absolute(data, json.loads(args.baseline.read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
