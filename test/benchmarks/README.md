# Benchmarks

Timing harness for the datarecord backend. Not collected by pytest (no `test_`
prefix; `test/conftest.py` also ignores the folder without `datarecord`).

## Workloads

Every workload runs in a fresh interpreter. Wall time is the median of
`--repeat`, minus the time of `w0_import_pypsa` (the import of `pypsa`, which
`w1_record_*` never does and so is not subtracted from). Peak RSS is the
child's own maximum resident set.

| workload | what it does |
|---|---|
| `w0_import_pypsa` | `import pypsa` |
| `w1_record_one_attribute` | open the record with `Workspace.process().at`, read `p_max_pu` over all snapshots, sum per entity; no network build |
| `w1_nc_one_attribute` | `pypsa.Network(nc)`, sum `p_max_pu` |
| `w1_record_eight`, `w1_nc_eight` | the same over eight copies |
| `w2_record_open_touch_export_nc` | `pypsa.Network(record)`, touch every static and dynamic frame, `export_to_netcdf` |
| `w2_nc_import_export` | `pypsa.Network(nc)` and `export_to_netcdf` |
| `w3_create_model_scigrid` | `pypsa.Network(scigrid_de.nc)`, `n.optimize.create_model()` |
| `w4_add_10k` | `n.add("Load", ...)` 10 000 times |

W1 to W2 use the carbon-management example tiled over an hourly index of
`--snapshots` steps (default 8760). W3 and W4 do not depend on `--snapshots`.
`attribute_mb` in the JSON is the size of the `p_max_pu` frame, for the memory
bound.

## Bounds

| workload | bound |
|---|---|
| W1 open record, one statistic on one attribute; also over 8 records | at least 5x faster than the `.nc` path, peak memory at most 3x the attribute's size |
| W2 open record, touch everything, export `.nc` | at most 1.5x today's `.nc` import plus export |
| W3 `create_model` on SciGrid-DE | at most 5% slower than master |
| W4 `n.add` of 10k components | at most 5% slower than master |

W1 is the criterion of the checkpoint before phase 3 and is not gated in phase
1. `--check` exits non-zero only for W3 and W4 above a ratio of 1.05; W1 and W2
ratios are printed only.

## Running

```sh
uv sync --extra dev        # Python 3.12 or newer for datarecord
uv run python test/benchmarks/run.py --snapshots 8760 --repeat 3 --out results.json
uv run python test/benchmarks/run.py --workloads w3_create_model_scigrid,w4_add_10k --check test/benchmarks/baseline.json
```

Fixtures are cached under `--root` (default `~/.cache/pypsa/benchmarks/<snapshots>/`)
and rebuilt when missing. `pypsa.examples.carbon_management()` is downloaded on
first use.

The smoke test runs the slow benchmark cases with
`uv run pytest test/test_benchmarks_smoke.py --run-slow`.

## Refreshing the baseline

```sh
uv run python test/benchmarks/run.py --snapshots 8760 --out test/benchmarks/baseline.json
```

The baseline mixes runs: W0, W3 and W4 were re-measured after the `name_owners`
fast path (see `partial_refresh` in the file), W1 and W2 come from the earlier run on
the same machine.

The baseline carries the host block it was measured on; compare only runs from
comparable hardware. The nightly Forgejo workflow (`.forgejo/workflows/benchmarks.yml`)
runs at 2190 snapshots and checks W3 and W4, which are size independent.
The check is informational (`continue-on-error`). The committed baseline is this
branch's macOS run and the W3/W4 bounds are against master, so a Linux runner
is not compared like for like; replace the baseline with the runner's first
`bench.json` artifact.
