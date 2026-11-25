# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Per-step cProfile breakdown: `python test/benchmarks/breakdown.py`. See README.md."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

STEPS = ("open", "static_read", "dynamic_read", "export", "writeback")


def _profiled(step: str, paths) -> tuple[cProfile.Profile, float]:
    """Do the step's setup unprofiled, then profile only the step."""
    import pypsa  # noqa: PLC0415

    action: Callable[[], object]
    if step == "open":
        action = lambda: pypsa.Network(paths.record)  # noqa: E731
    elif step == "static_read":
        n = pypsa.Network(paths.record)
        action = lambda: n.c.generators.static  # noqa: E731
    else:
        n = pypsa.Network(paths.record)
        n.c.generators.static  # noqa: B018
        if step == "dynamic_read":
            action = lambda: n.c.generators.dynamic["p_max_pu"]  # noqa: E731
        elif step == "export":
            tmp = tempfile.TemporaryDirectory()
            action = lambda: n.export_to_netcdf(f"{tmp.name}/out.nc")  # noqa: E731
        else:
            c = n.c.generators
            df = c.dynamic["p_max_pu"]
            df.iloc[0, 0] += 0.01
            c.dynamic["p_max_pu"] = df
            # Times the call incl. its `check()`, not the base comparison or a write.
            action = lambda: n.to_datarecord(base=n.record)  # noqa: E731
    prof = cProfile.Profile()
    start = time.perf_counter()
    prof.enable()
    try:
        action()
    finally:
        prof.disable()
    return prof, time.perf_counter() - start


def _top(stats: pstats.Stats, key: str, top: int, wall: float) -> list[dict]:
    rows = []
    for (file, line, func), (_, ncalls, tottime, cumtime, _) in stats.stats.items():  # type: ignore[attr-defined]
        rows.append(
            {
                "function": f"{file}:{line}({func})",
                "ncalls": ncalls,
                "tottime": tottime,
                "cumtime": cumtime,
                "self_share": tottime / wall if wall else 0.0,
            }
        )
    rows.sort(key=lambda r: r[key], reverse=True)
    return rows[:top]


def _one(step: str, paths: dict, top: int) -> None:
    from fixtures import Paths  # noqa: PLC0415

    load = os.getloadavg()[0]
    prof, wall = _profiled(step, Paths(**paths))
    stats = pstats.Stats(prof)
    res = {
        "wall_s": wall,
        "load_avg": load,
        "top_self": _top(stats, "tottime", top, wall),
        "top_cumulative": _top(stats, "cumtime", top, wall),
    }
    print(json.dumps(res))


def _run_child(step: str, paths: dict, top: int) -> dict:
    """Run one step in a fresh interpreter and return its result."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", step]
    cmd += ["--paths", json.dumps(paths), "--top", str(top)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode:
        msg = f"step {step} failed with exit code {proc.returncode}:\n{proc.stderr}"
        raise RuntimeError(msg)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def breakdown(paths: dict, top: int) -> dict:
    """Profile every step in its own child and add each step's share of the total."""
    steps = {step: _run_child(step, paths, top) for step in STEPS}
    total = sum(s["wall_s"] for s in steps.values())
    for s in steps.values():
        s["share"] = s["wall_s"] / total
    return steps


def table(steps: dict) -> str:
    """Return a Markdown table: step, wall, share, top self-time function."""
    lines = [
        "| step | wall_s | share | top self-time function | self share |",
        "|---|---|---|---|---|",
    ]
    for step, s in steps.items():
        t = s["top_self"][0]
        lines.append(
            f"| {step} | {s['wall_s']:.3f} | {s['share']:.0%} "
            f"| `{t['function']}` | {t['self_share']:.0%} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the breakdown CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=2190)
    parser.add_argument(
        "--root", type=Path, default=Path.home() / ".cache" / "pypsa" / "benchmarks"
    )
    parser.add_argument("--out", type=Path, default=Path("breakdown.json"))
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--one", help="internal: profile one step in this process")
    parser.add_argument("--paths", help="internal: fixture paths as JSON")
    args = parser.parse_args(argv)

    if args.one:
        _one(args.one, json.loads(args.paths), args.top)
        return 0

    from fixtures import build_paths  # noqa: PLC0415

    paths = build_paths(args.root, args.snapshots).to_dict()
    steps = breakdown(paths, args.top)
    args.out.write_text(
        json.dumps({"snapshots": args.snapshots, "steps": steps}, indent=2)
    )
    print(table(steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
