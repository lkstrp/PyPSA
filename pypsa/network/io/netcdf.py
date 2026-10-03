# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Importer and exporter for netCDF files."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import pandas as pd
import validators
import xarray as xr

from pypsa.network.io._common import (
    _coerce_string_dtypes,
    _Exporter,
    _Importer,
    _retrieve_from_url,
)

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self
logger = logging.getLogger(__name__)


def _open_netcdf(path: Path) -> xr.Dataset:
    """Open a netCDF file, reading variable-length strings as object arrays.

    xarray pads them to the longest entry and stores them as UTF-32, which
    explodes memory for length-skewed columns such as geometry WKT.

    Cloud paths (via cloudpathlib) resolve to their local cache file so that netCDF4
    reads the downloaded file instead of treating the URI as a remote URL.
    """
    local_path = os.fspath(path)
    with netCDF4.Dataset(local_path) as nc:
        strings = {
            name: (var.dimensions, var[:])
            for name, var in nc.variables.items()
            if var.dtype == str
        }
    return xr.open_dataset(local_path, drop_variables=strings).assign(strings)


class _ImporterNetCDF(_Importer):
    """Importer class for netCDF files."""

    ds: xr.Dataset

    def __init__(self, path: str | Path | xr.Dataset) -> None:
        """Initialize the importer for netCDF files.

        Parameters
        ----------
        path : str | Path | xr.Dataset
            Path to the netCDF file or an xarray.Dataset.

        """
        self.path = path
        if isinstance(path, (str | Path)):
            if validators.url(str(path)):
                self.ds = _retrieve_from_url(str(path), _open_netcdf)
            else:
                self.ds = _open_netcdf(Path(path))
        else:
            self.ds = path

    def __enter__(self) -> Self:
        """Enter the context manager."""
        if isinstance(self.path, (str | Path)):
            super().__init__()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit the context manager."""
        if isinstance(self.path, (str | Path)):
            super().__exit__(exc_type, exc_val, exc_tb)

    def get_attributes(self) -> dict:
        """Get generic network attributes."""
        return {
            attr[len("network_") :]: val
            for attr, val in self.ds.attrs.items()
            if attr.startswith("network_")
        }

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        return json.loads(self.ds.attrs.get("meta", "{}"))

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        return json.loads(self.ds.attrs.get("crs", "{}"))

    def get_snapshots(self) -> pd.DataFrame:
        """Get snapshots data."""
        return self.get_static("snapshots", "snapshots")

    def get_investment_periods(self) -> pd.DataFrame:
        """Get investment periods data."""
        return self.get_static("investment_periods", "investment_periods")

    def get_scenarios(self) -> pd.DataFrame:
        """Get scenarios data."""
        if "scenario_weight" in self.ds:
            df = self.ds["scenario_weight"].to_pandas().rename("weight").to_frame()
            df.index.name = "scenario"
            return df

    def get_static(self, list_name: str, index_name: str | None = None) -> pd.DataFrame:
        """Get static components data."""
        if index_name is None:
            index_name = list_name + "_i"
        if index_name not in self.ds.coords:
            return None
        df = pd.DataFrame()
        for attr, data_var in self.ds.data_vars.items():
            attr = str(attr)
            match = re.match(rf"^{list_name}_(?!t_|pw_)(.+)$", attr)
            if match:
                loaded_df = data_var.to_pandas()
                if isinstance(loaded_df, pd.DataFrame):
                    loaded_df = loaded_df.stack()
                df[match.group(1)] = loaded_df

        if df.empty:
            index = self.ds.coords[index_name].to_index().rename("name")
            if "scenario" in self.ds.coords:
                scenario_index = self.ds.coords["scenario"].to_index()
                index = pd.MultiIndex.from_product([scenario_index, index])
            df = pd.DataFrame(index=index)
        return _coerce_string_dtypes(df)

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get dynamic components data."""
        t = list_name + "_t_"
        for attr in self.ds.data_vars.keys():
            attr = str(attr)
            if attr.startswith(t):
                df = (
                    self.ds[attr]
                    .rename({attr + "_i": "name"})
                    .to_series()
                    .unstack("snapshots")
                    .T
                )
                yield attr[len(t) :], _coerce_string_dtypes(df)

    def get_piecewise(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get piecewise component data."""
        for attr, data_var in self.ds.data_vars.items():
            match = re.match(rf"^{list_name}_pw_(.+)$", str(attr))
            if match:
                df = data_var.stack(
                    combined=(f"{attr}_i", f"{attr}_attr_i")
                ).to_pandas()
                df.columns.names = ["name", "attribute"]

                yield match.group(1), df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterNetCDF(_Exporter):
    """Exporter class for netCDF files."""

    def __init__(
        self,
        path: Path | str | None,
        compression: dict | None = None,
        float32: bool = False,
    ) -> None:
        """Initialize exporter for netCDF files.

        Parameters
        ----------
        path : str | None
            Path to save the netCDF file.
        compression : dict | None, default None
            Compression settings for the netCDF file.
        float32 : bool, default False
            If True, typecast float64 to float32.

        """
        self.path = path
        if compression is None:
            compression = {"zlib": True, "complevel": 4}
        self.compression = compression
        self.float32 = float32
        self.ds = xr.Dataset()

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        self.ds.attrs.update(("network_" + attr, val) for attr, val in attrs.items())

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        self.ds.attrs["meta"] = json.dumps(meta)

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        self.ds.attrs["crs"] = json.dumps(crs)

    def save_snapshots(self, snapshots: pd.Index) -> None:
        """Save snapshots data."""
        snapshots = snapshots.rename_axis(index="snapshots")
        for attr in snapshots.columns:
            self.ds["snapshots_" + attr] = snapshots[attr]

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        investment_periods = investment_periods.rename_axis(index="investment_periods")
        for attr in investment_periods.columns:
            self.ds["investment_periods_" + attr] = investment_periods[attr]

    def save_scenarios(self, scenarios: pd.Index) -> None:
        """Save scenarios data."""
        for attr in scenarios.columns:
            self.ds["scenario_" + attr] = scenarios[attr]

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save a static components data."""
        df = df.rename_axis(index={"name": list_name + "_i"})
        names = df.index.get_level_values(list_name + "_i").drop_duplicates()
        self.ds[list_name + "_i"] = names

        if not df.columns.empty:
            df_array = df.to_xarray().rename(
                {attr: list_name + "_" + attr for attr in df.columns}
            )
            if isinstance(df.index, pd.MultiIndex):
                # `to_xarray` lays a MultiIndex out in level order, which a
                # rename leaves sorted rather than in row order. Keep the
                # rows' own order so a round trip reads them back as written.
                df_array = df_array.reindex({list_name + "_i": names})
            self.ds = self.ds.merge(df_array, overwrite_vars=True)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save a dynamic components data."""
        new_col_name = list_name + "_t_" + attr + "_i"
        snapshots = ("snapshots", df.index.values)
        if isinstance(df.columns, pd.MultiIndex):  # stochastic
            scenarios = np.sort(df.columns.get_level_values(0).unique().values)
            names = np.sort(df.columns.get_level_values(1).unique().values)
            grid = pd.MultiIndex.from_product([scenarios, names])
            values = df.reindex(columns=grid).values.reshape(
                len(df.index), len(scenarios), len(names)
            )
            dims = ["snapshots", "scenario", new_col_name]
            coords = {
                "snapshots": snapshots,
                "scenario": ("scenario", scenarios),
                new_col_name: (new_col_name, names),
            }
        else:
            values = df.values
            dims = ["snapshots", new_col_name]
            coords = {
                "snapshots": snapshots,
                new_col_name: (new_col_name, df.columns.values),
            }

        self.ds[list_name + "_t_" + attr] = xr.DataArray(
            values, dims=dims, coords=coords
        )

    def save_piecewise(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save piecewise component data."""
        data_var_name = f"{list_name}_pw_{attr}"
        df = df.rename_axis(
            columns={
                "name": f"{data_var_name}_i",
                "attribute": f"{data_var_name}_attr_i",
            }
        )
        self.ds[data_var_name] = df.stack(
            level=df.columns.names, future_stack=True
        ).to_xarray()

    def set_compression_encoding(self) -> None:
        """Set compression encoding for all variables."""
        logger.debug("Setting compression encodings: %s", self.compression)
        for v in self.ds.data_vars:
            if self.ds[v].dtype.kind not in ["U", "O"]:
                self.ds[v].encoding.update(self.compression)

    def typecast_float32(self) -> None:
        """Typecast float64 to float32 for all variables."""
        logger.debug("Typecasting float64 to float32.")
        for v in self.ds.data_vars:
            if self.ds[v].dtype == np.float64:
                self.ds[v] = self.ds[v].astype(np.float32)

    def finish(self) -> None:
        """Finish the export process.

        Runs post-processing, compression and saving to disk.
        """
        # pandas>=3 infer_string strings aren't netCDF-writable. Cast to object and
        # write with the option off to keep NaNs. https://github.com/pydata/xarray/issues/10301
        for name in list(self.ds.variables):
            if isinstance(self.ds[name].dtype, pd.StringDtype):
                self.ds[name] = self.ds[name].astype(object)
        if self.float32:
            self.typecast_float32()
        if self.compression:
            self.set_compression_encoding()
        if self.path is not None:
            _path = Path(self.path)
            with _path.open("w"), pd.option_context("future.infer_string", False):
                self.ds.to_netcdf(_path)
