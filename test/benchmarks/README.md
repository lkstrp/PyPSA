# Benchmarks

Timing harness for the datarecord backend. Not collected by pytest (no `test_`
prefix; `test/conftest.py` also ignores the folder without `datarecord`).

## Workloads

Every workload runs in a fresh interpreter. A run does `--repeat` rounds; a
round runs W0, then every workload once in the table's order, so the record and
`.nc` variants of a workload run next to each other. Wall time is the min of
`--repeat`, minus the time of `w0_import_pypsa` (the import of `pypsa`, which
every workload does, so it is subtracted from all). Peak RSS is the
child's own maximum resident set.

Each run reads the 1-minute load average before it starts. A run that starts
above load 8 is repeated once, and the min covers both. When the repeat also
starts above 8 the result carries `high_load: true` and the table marks its row
with `*`. Every result lists the start loads in `load_avg`; the table shows
their maximum.

| workload | what it does |
|---|---|
| `w0_import_pypsa` | `import pypsa` |
| `w1_record_open_only` | `pypsa.Network(record)`, read nothing; its peak RSS is the baseline of the W1 memory bound |
| `w1_record_one_attribute` | `pypsa.Network(record)` (lazy open), read `p_max_pu`, sum |
| `w1_nc_one_attribute` | `pypsa.Network(nc)`, sum `p_max_pu` |
| `w5_record_writeback` | `pypsa.Network(record)`, read `p_max_pu`, change one value, `n.to_datarecord(base=n.record)` (times its `check()`, not the base comparison or a layer write) |
| `w1_record_eight`, `w1_nc_eight` | the same over eight copies |
| `w2_record_open_touch_export_nc` | `pypsa.Network(record)`, touch every static and dynamic frame, `export_to_netcdf` |
| `w2_nc_import_export` | `pypsa.Network(nc)` and `export_to_netcdf` |
| `w3_create_model_scigrid` | `pypsa.Network(scigrid_de.nc)`, `n.optimize.create_model()` |
| `w4_add_10k` | `n.add("Load", ...)` 10 000 times |

W1, W2 and W5 use the carbon-management example tiled over an hourly index of
`--snapshots` steps (default 8760). W3 and W4 do not depend on `--snapshots`.
`attribute_mb` in the JSON is the size of the `p_max_pu` frame, for the memory
bound.

## Bounds

| workload | bound |
|---|---|
| W1 open record, one statistic on one attribute; also over 8 records | at least 5x faster than the `.nc` path, peak memory at most 3x the attribute's size, net of the open-only workload's peak RSS (reported, not gated, pending datarecord epic #46) |
| W2 open record, touch everything, export `.nc` | at most 1.5x today's `.nc` import plus export (reported, not gated, pending datarecord epic #46) |
| W5 open record, change one series value, `to_datarecord(base=n.record)` (incl. its `check()`, no comparison or write) | at most 2x the W1 one-attribute wall of the same run (reported, not gated) |
| W3 `create_model` on SciGrid-DE | at most 5% slower than master |
| W4 `n.add` of 10k components | at most 5% slower than master |

W1 and W2 are reported, not gated (datarecord epic #46 provides the
per-(type, attribute) read path). Both modes below print their ratios (record
over `.nc`, and peak RSS over `attribute_mb`, net of `w1_record_open_only`'s peak RSS,
or of the `pypsa` import's `import_peak_rss_mb` when that workload is absent) next to the spec bound in an informational table. Only W3
and W4 above a ratio of 1.05 make the gate exit non-zero. W1 is the criterion
of the checkpoint before phase 3.

## Running

```sh
uv sync --extra dev        # Python 3.12 or newer for datarecord
uv run python test/benchmarks/run.py --snapshots 8760 --repeat 3 --out results.json
```

Fixtures are cached under `--root` (default `~/.cache/pypsa/benchmarks/<snapshots>/`)
and rebuilt when missing. `pypsa.examples.carbon_management()` is downloaded on
first use.

The smoke test runs the slow benchmark cases with
`uv run pytest test/test_benchmarks_smoke.py --run-slow`.

## Profile

`breakdown.py` profiles each step of the record path with `cProfile` at one
fixture size and says where the wall goes:

```sh
uv run python test/benchmarks/breakdown.py --snapshots 2190 --out breakdown.json --top 15
```

Every step runs in a fresh interpreter. The child does the step's setup
unprofiled and profiles only the step: `open` (`pypsa.Network(record)`),
`static_read` (first `n.c.generators.static`), `dynamic_read`
(`n.c.generators.dynamic["p_max_pu"]`), `export` (`export_to_netcdf`, the W2
path) and `writeback` (change one `p_max_pu` value, `to_datarecord(base=n.record)`,
the W5 path; times its `check()`, not the base comparison or a layer write). `--out` holds, per step, `wall_s`, its `share` of the total over
all steps, the start `load_avg`, and the top `--top` functions by self time
(`top_self`) and by cumulative time (`top_cumulative`) as `file:line(func)` with
`ncalls`, `tottime`, `cumtime` and `self_share` (`tottime` over the step's wall).
Times include cProfile's overhead. A Markdown table of the top self-time
function per step goes to stdout.

## Gating: relative `--check`

```sh
uv run python test/benchmarks/run.py --workloads w3_create_model_scigrid,w4_add_10k --check
uv run python test/benchmarks/run.py --check --base ddf9443b --head 65111719 --workloads w4_add_10k --repeat 5
```

`--check` measures base and head in the same run on the same host. The base
(`--base`, default `origin/$GITHUB_BASE_REF` when set, else `origin/master`) is
checked out into a temporary git worktree and given its own environment with
`uv sync --extra dev --locked`. The head is this working tree, or `--head <rev>`
checked out the same way. Both sides run this checkout's harness and workloads,
so only `pypsa` and its locked dependencies differ.

Each of `--repeat` rounds runs every workload once on the base, then once on
the head, so drifting load hits both sides. Each side subtracts its own W0
min, and the gated figure is the min of head over the min of base. The same
load retry applies to every run.
A workload that fails on one side gates the run when it is W3 or W4 and reads
`n/a` otherwise (W1 on a base without the record reader, for example). The
output carries both SHAs and the load average at start and end; `--out` stores
them in the JSON with the host block. The worktrees are removed on exit.

A gate verdict needs both sides from the same host and the same run. Under
heavy load, raise `--repeat`, and read the ratio together with the load average.

## Reference: absolute `--compare`

```sh
uv run python test/benchmarks/run.py --compare results.json --baseline test/benchmarks/baseline.json
```

`--baseline PATH` compares absolute wall times of a result JSON (`--compare`,
or a fresh run) with another, with the same 1.05 bound and exit code. It is for
by-hand use: absolute times do not transfer between machines, so a verdict
holds only when both files come from the same host.

`baseline.json` is the reference table of the host that recorded it (its host
block says which); `--check` does not compare against it. Refresh it on that
host with

```sh
uv run python test/benchmarks/run.py --snapshots 8760 --out test/benchmarks/baseline.json
```

The baseline mixes runs (see `partial_refresh` in the file): W1, W2, W5 and the
open-only row were re-measured 2026-11-25 at 2190 snapshots on datarecord 0.1.0a2
(the lockfile pin), min of 3 rounds (2 for the W2 pair), one bound group per
invocation with its own `--root`. The host was loaded throughout, with start loads
of 14 to 49, so every re-measured row carries `high_load`; read those absolute
times with that in mind. W0, W3 and W4 come from the run on 2026-11-05 on the same
machine and do not depend on the snapshot count. Records cached under another
datarecord pin do not open, so refresh these rows with a fresh `--root`, not the
default cache.

## Nightly

The nightly Forgejo workflow (`.forgejo/workflows/benchmarks.yml`) runs
`--check --base origin/master` at 2190 snapshots with 3 rounds and fails the
job when W3 or W4 is more than 5% slower than master on the runner. The run's
JSON is uploaded as the `benchmarks` artifact.
