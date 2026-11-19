# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Benchmark workloads. `pypsa` is imported inside each one, so W0 measures it."""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from fixtures import Paths

Workload = namedtuple("Workload", "name fn fixture")  # noqa: PYI024


def w0_import_pypsa(paths: Paths) -> None:
    """Import pypsa; its time is subtracted from every other workload."""
    import pypsa  # noqa: F401, PLC0415


def _attribute_sum(path: str) -> int:
    import pypsa  # noqa: PLC0415

    n = pypsa.Network(path)
    df = n.c.generators.dynamic["p_max_pu"]
    df.sum()
    return int(df.memory_usage(deep=True).sum())


def w1_record_one_attribute(paths: Paths) -> int:
    """Open the record as a network, read `p_max_pu` and sum it."""
    return _attribute_sum(paths.record)


def w1_nc_one_attribute(paths: Paths) -> int:
    """Import the `.nc` and sum `p_max_pu` per generator."""
    return _attribute_sum(paths.nc)


def w1_record_eight(paths: Paths) -> int:
    """`w1_record_one_attribute` over eight records."""
    return sum(_attribute_sum(p) for p in paths.record_copies)


def w1_nc_eight(paths: Paths) -> int:
    """`w1_nc_one_attribute` over eight files."""
    return sum(_attribute_sum(p) for p in paths.nc_copies)


def w2_record_open_touch_export_nc(paths: Paths) -> None:
    """Open the record, touch every frame, export to `.nc`."""
    import tempfile  # noqa: PLC0415

    import pypsa  # noqa: PLC0415

    n = pypsa.Network(paths.record)
    for c in n.components:
        c.static.shape  # noqa: B018
        for df in c.dynamic.values():
            df.shape  # noqa: B018
    with tempfile.TemporaryDirectory() as tmp:
        n.export_to_netcdf(f"{tmp}/out.nc")


def w2_nc_import_export(paths: Paths) -> None:
    """Import the `.nc` and export it again."""
    import tempfile  # noqa: PLC0415

    import pypsa  # noqa: PLC0415

    n = pypsa.Network(paths.nc)
    with tempfile.TemporaryDirectory() as tmp:
        n.export_to_netcdf(f"{tmp}/out.nc")


def w3_create_model_scigrid(paths: Paths) -> None:
    """Build the linopy model of SciGrid-DE."""
    import pypsa  # noqa: PLC0415

    n = pypsa.Network(paths.scigrid_nc)
    n.optimize.create_model()


def w4_add_10k(paths: Paths) -> None:
    """Add 10 000 loads in a loop."""
    import pypsa  # noqa: PLC0415

    n = pypsa.Network()
    n.add("Bus", "bus")
    for i in range(10_000):
        n.add("Load", f"l{i}", bus="bus")


_FUNCTIONS: list[tuple[Callable, str]] = [
    (w0_import_pypsa, "none"),
    (w1_record_one_attribute, "carbon"),
    (w1_nc_one_attribute, "carbon"),
    (w1_record_eight, "carbon"),
    (w1_nc_eight, "carbon"),
    (w2_record_open_touch_export_nc, "carbon"),
    (w2_nc_import_export, "carbon"),
    (w3_create_model_scigrid, "scigrid"),
    (w4_add_10k, "none"),
]
WORKLOADS: dict[str, Workload] = {
    fn.__name__: Workload(fn.__name__, fn, fixture) for fn, fixture in _FUNCTIONS
}
