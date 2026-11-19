# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Relative benchmark: base and head measured alternately in one run.

Both sides run this checkout's harness and workloads; only the interpreter,
and with it the installed `pypsa`, differs.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from run import BASE as W0
from run import GATED_PREFIXES, TOLERANCE, _run_child

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
# Variables that would point `uv sync` at the running environment instead of
# the worktree's own `.venv`.
_UV_ENV_DROP = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")

Run = tuple[float, float, float | None]


@dataclass(frozen=True)
class Side:
    """One side of the comparison: an interpreter and the directory it runs in."""

    label: str
    python: str
    cwd: Path | None


def current_side() -> Side:
    """Return the running interpreter as the head side."""
    return Side("head", sys.executable, None)


def _git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def rev_sha(rev: str) -> str:
    """Resolve `rev` to a commit SHA in this repository."""
    out = _git("rev-parse", "--verify", f"{rev}^{{commit}}")
    if out.returncode:
        msg = f"cannot resolve {rev}: {out.stderr.strip()}"
        raise RuntimeError(msg)
    return out.stdout.strip()


def is_dirty() -> bool:
    """Return whether tracked files in this checkout differ from `HEAD`."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    if out.returncode:
        msg = f"git status failed: {out.stderr.strip()}"
        raise RuntimeError(msg)
    return bool(out.stdout.strip())


@contextmanager
def checkout(rev: str, label: str = "base") -> Iterator[Side]:
    """Check `rev` out into a temporary worktree with its own environment."""
    wt = Path(tempfile.mkdtemp(prefix=f"pypsa-bench-{label}-"))
    add = _git("worktree", "add", "--detach", str(wt), rev)
    if add.returncode:
        shutil.rmtree(wt, ignore_errors=True)
        msg = f"git worktree add of {rev} failed: {add.stderr.strip()}"
        raise RuntimeError(msg)
    try:
        env = {k: v for k, v in os.environ.items() if k not in _UV_ENV_DROP}
        sync = subprocess.run(
            ["uv", "sync", "--extra", "dev", "--locked", "--python", sys.executable],  # noqa: S607
            cwd=wt,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if sync.returncode:
            tail = "\n".join(sync.stderr.strip().splitlines()[-5:])
            msg = f"uv sync of {rev} failed:\n{tail}"
            raise RuntimeError(msg)
        yield Side(label, str(wt / ".venv" / "bin" / "python"), wt)
    finally:
        remove = _git("worktree", "remove", "--force", str(wt))
        if remove.returncode:
            print(
                f"git worktree remove of {wt} failed: {remove.stderr.strip()}",
                file=sys.stderr,
            )


def interleave(
    names: list[str],
    paths: dict,
    base: Side,
    head: Side,
    repeat: int,
    run: Callable[..., Run] = _run_child,
) -> dict[str, dict[str, list[Run | None]]]:
    """Run every workload `repeat` rounds, base then head within each round.

    A failing child raises on W0 and on gated workloads; elsewhere the run is
    recorded as `None`.
    """
    order = [W0, *[n for n in names if n != W0]]
    runs: dict[str, dict[str, list[Run | None]]] = {
        side.label: {name: [] for name in order} for side in (base, head)
    }
    for _ in range(repeat):
        for name in order:
            for side in (base, head):
                try:
                    result = run(name, paths, python=side.python, cwd=side.cwd)
                except RuntimeError:
                    if name == W0 or name.startswith(GATED_PREFIXES):
                        raise
                    result = None
                runs[side.label][name].append(result)
    return runs


def _side_results(runs: dict[str, list[Run | None]]) -> dict[str, dict]:
    complete = {
        name: [r for r in rs if r is not None]
        for name, rs in runs.items()
        if None not in rs
    }
    w0 = statistics.median(r[0] for r in complete[W0])
    results: dict[str, dict] = {}
    for name, rs in complete.items():
        wall = statistics.median(r[0] for r in rs)
        entry: dict = {
            "wall_s": wall if name == W0 else max(wall - w0, 1e-3),
            "peak_rss_mb": max(r[1] for r in rs),
        }
        if rs[0][2] is not None:
            entry["attribute_mb"] = rs[0][2]
        results[name] = entry
    return results


def relative_results(
    runs: dict[str, dict[str, list[Run | None]]], tolerance: float = TOLERANCE
) -> dict:
    """Reduce interleaved runs to per-side results and head-over-base ratios.

    `ratios` maps each workload to the median of head over the median of
    base, both net of their own side's W0, or `None` when either side failed.
    `failed` lists the gated workloads whose ratio exceeds `tolerance`.
    """
    base, head = _side_results(runs["base"]), _side_results(runs["head"])
    ratios: dict[str, float | None] = {}
    for name in runs["head"]:
        if name in base and name in head:
            ratios[name] = head[name]["wall_s"] / base[name]["wall_s"]
        else:
            ratios[name] = None
    failed = [
        name
        for name, ratio in ratios.items()
        if name.startswith(GATED_PREFIXES) and ratio is not None and ratio > tolerance
    ]
    return {"base": base, "head": head, "ratios": ratios, "failed": failed}
