# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Importer and exporter for HDF5 files."""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd
import validators

from pypsa.common import check_optional_dependency
from pypsa.network.io._common import _Exporter, _Importer, _retrieve_from_url

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
logger = logging.getLogger(__name__)


class _ImporterHDF5(_Importer):
    """Importer class for HDF5 files."""

    def __init__(self, path: str | pd.HDFStore) -> None:
        """Initialize the importer for HDF5 files.

        Parameters
        ----------
        path : str | pd.HDFStore
            Path to the HDF5 file or an hdfstore object.

        """
        check_optional_dependency(
            "tables",
            "Missing optional dependencies to use HDF5 files. Install them via "
            "`pip install pypsa[hdf5]` or `conda install -c conda-forge pypsa[hdf5]`.",
        )
        self.path = path
        self.ds: pd.HDFStore
        if isinstance(path, (str | Path)):
            reader = partial(pd.HDFStore, mode="r")
            if validators.url(str(path)):
                self.ds = _retrieve_from_url(str(path), reader)
            else:
                self.ds = reader(Path(path))

        self.index: dict = {}

    def get_attributes(self) -> dict:
        """Get generic network attributes."""
        return dict(self.ds["/network"].reset_index().iloc[0])

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        return json.loads(self.ds["/meta"][0] if "/meta" in self.ds else "{}")

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        return json.loads(self.ds["/crs"][0] if "/crs" in self.ds else "{}")

    def get_snapshots(self) -> pd.Series:
        """Get snapshots data."""
        return self.ds["/snapshots"] if "/snapshots" in self.ds else None  # noqa: SIM401

    def get_investment_periods(self) -> pd.Series:
        """Get investment periods data."""
        return (
            self.ds["/investment_periods"] if "/investment_periods" in self.ds else None  # noqa: SIM401
        )

    def get_static(self, list_name: str) -> pd.DataFrame:
        """Get static components data."""
        if "/" + list_name not in self.ds:
            return None

        df = self.ds["/" + list_name].set_index("name")

        self.index[list_name] = df.index
        return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get dynamic components data."""
        for tab in self.ds:
            if tab.startswith("/" + list_name + "_t/"):
                attr = tab[len("/" + list_name + "_t/") :]
                df = self.ds[tab]
                df.columns = self.index[list_name][df.columns]
                yield attr, df

    def get_piecewise(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get piecewise component data."""
        for tab in self.ds:
            if tab.startswith("/" + list_name + "_p/"):
                attr = tab[len("/" + list_name + "_p/") :]
                df = self.ds[tab]
                df = df.unstack(["name", "attribute"])
                yield attr, df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterHDF5(_Exporter):
    """Exporter class for HDF5 files."""

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        """Initialize exporter for HDF5 files.

        Parameters
        ----------
        path : str | Path
            Path to save the HDF5 file.
        **kwargs : Any
            Additional keyword arguments for the HDFStore.

        """
        check_optional_dependency(
            "tables",
            "Missing optional dependencies to use HDF5 files. Install them via "
            "`pip install pypsa[hdf5]` or `conda install -c conda-forge pypsa[hdf5]`.",
        )
        self.path = Path(path)
        self._hdf5_handle = self.path.open("w")
        self.ds = pd.HDFStore(self.path, mode="w", **kwargs)
        self.index: dict = {}

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit the context manager."""
        super().__exit__(exc_type, exc_val, exc_tb)

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        name = attrs.pop("name")
        self.ds.put(
            "/network",
            pd.DataFrame(attrs, index=pd.Index([name], name="name")),
            format="table",
            index=False,
        )

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        self.ds.put("/meta", pd.Series(json.dumps(meta)))

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        self.ds.put("/crs", pd.Series(json.dumps(crs)))

    def save_snapshots(self, snapshots: Sequence) -> None:
        """Save snapshots data."""
        self.ds.put("/snapshots", snapshots, format="table", index=False)

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        self.ds.put(
            "/investment_periods",
            investment_periods,
            format="table",
            index=False,
        )

    def save_scenarios(self, scenarios: pd.DataFrame) -> None:
        """Save scenarios data."""
        msg = "Stochastic networks are not supported in the HDF5 exporter. Use netcdf instead."
        raise NotImplementedError(msg)

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save a static components data."""
        df = df.rename_axis(index="name")
        self.index[list_name] = df.index
        df = df.reset_index()
        self.ds.put("/" + list_name, df, format="table", index=False)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""
        df = df.set_axis(self.index[list_name].get_indexer(df.columns), axis="columns")
        self.ds.put("/" + list_name + "_t/" + attr, df, format="table", index=False)

    def save_piecewise(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save piecewise component data."""
        df_stack = df.stack(df.columns.names, future_stack=True)
        self.ds.put(
            "/" + list_name + "_p/" + attr, df_stack, format="table", index=False
        )

    def finish(self) -> None:
        """Postprocessing of exporting process."""
        self._hdf5_handle.close()
