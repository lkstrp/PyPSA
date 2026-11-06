# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cached benchmark inputs: a netCDF file and a datarecord folder per network."""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    import pypsa

logger = logging.getLogger(__name__)

COPIES = 8
START = "2013-01-01"


@dataclass
class Paths:
    """Inputs the workloads read."""

    nc: str
    record: str
    scigrid_nc: str
    nc_copies: list[str]
    record_copies: list[str]

    def to_dict(self) -> dict:
        """Return the paths as a JSON-able dict."""
        return asdict(self)


def _tile(n, snapshots: int) -> None:
    """Tile every time series of `n` over an hourly index of `snapshots` steps."""
    index = pd.date_range(START, periods=snapshots, freq="h", name="snapshot")
    series = {
        (c.name, key): df
        for c in n.components
        for key, df in c.dynamic.items()
        if not df.empty
    }
    n.set_snapshots(index)
    for (c_name, key), df in series.items():
        values = df.to_numpy()[np.arange(snapshots) % len(df)]
        n.components[c_name].dynamic[key] = pd.DataFrame(
            values, index=n.snapshots, columns=df.columns
        )
    n.snapshot_weightings.loc[:, :] = 1.0


def carbon_management_record(root: Path, snapshots: int) -> tuple[Path, Path]:
    """Build the carbon-management `.nc` and record under `root`, once per size.

    Parameters
    ----------
    root : Path
        Cache directory.
    snapshots : int
        Number of hourly snapshots the shipped series are tiled over.

    Returns
    -------
    tuple[Path, Path]
        The netCDF file and the record directory.

    """
    import pypsa  # noqa: PLC0415

    nc, record = root / "carbon_management.nc", root / "carbon_management_record"
    if nc.exists() and record.exists():
        return nc, record
    root.mkdir(parents=True, exist_ok=True)
    n = pypsa.examples.carbon_management()
    renamed = {c: len(m) for c, m in n.deduplicated_names.items()}
    logger.info("carbon_management: renamed clashing names on import: %s", renamed)
    _tile(n, snapshots)
    _export(n, nc, record)
    return nc, record


def scigrid_record(root: Path) -> tuple[Path, Path]:
    """Build the `scigrid_de` `.nc` and record under `root` at its 24 snapshots.

    Parameters
    ----------
    root : Path
        Cache directory.

    Returns
    -------
    tuple[Path, Path]
        The netCDF file and the record directory.

    """
    from pathlib import Path  # noqa: PLC0415

    import pypsa  # noqa: PLC0415

    nc, record = root / "scigrid_de.nc", root / "scigrid_de_record"
    if nc.exists() and record.exists():
        return nc, record
    root.mkdir(parents=True, exist_ok=True)
    # The in-repo file, as the published one lacks the de-clashed Bus columns.
    n = pypsa.Network(
        Path(__file__).parents[2]
        / "examples"
        / "networks"
        / "scigrid-de"
        / "scigrid-de.nc"
    )
    _export(n, nc, record)
    return nc, record


def _export(n: pypsa.Network, nc: Path, record: Path) -> None:
    """Export under temporary names and rename, so a failure leaves no cache hit."""
    tmp_nc, tmp_record = (
        nc.with_name(f"tmp_{nc.name}"),
        record.with_name(f"tmp_{record.name}"),
    )
    tmp_nc.unlink(missing_ok=True)
    shutil.rmtree(tmp_record, ignore_errors=True)
    shutil.rmtree(record, ignore_errors=True)
    n.export_to_netcdf(tmp_nc)
    n.export_to_datarecord(tmp_record)
    tmp_nc.rename(nc)
    tmp_record.rename(record)


def build_paths(root: Path, snapshots: int) -> Paths:
    """Build every fixture under `root/<snapshots>` unless cached."""
    root = root / str(snapshots)
    nc, record = carbon_management_record(root, snapshots)
    scigrid_nc, _ = scigrid_record(root)
    nc_copies, record_copies = [], []
    for i in range(COPIES):
        nc_i, record_i = root / f"copy{i}.nc", root / f"copy{i}_record"
        if not nc_i.exists() or nc_i.stat().st_mtime < nc.stat().st_mtime:
            tmp_nc = root / f"tmp_copy{i}.nc"
            shutil.copy(nc, tmp_nc)
            tmp_nc.replace(nc_i)
        if not record_i.exists() or record_i.stat().st_mtime < record.stat().st_mtime:
            tmp_record = root / f"tmp_copy{i}_record"
            shutil.rmtree(tmp_record, ignore_errors=True)
            shutil.copytree(record, tmp_record)
            shutil.rmtree(record_i, ignore_errors=True)
            tmp_record.rename(record_i)
        nc_copies.append(str(nc_i))
        record_copies.append(str(record_i))
    return Paths(str(nc), str(record), str(scigrid_nc), nc_copies, record_copies)
