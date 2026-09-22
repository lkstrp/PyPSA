# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Importer and exporter for Excel files."""

from __future__ import annotations

import json
import logging
import re
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd
import validators
from pandas.errors import ParserError

from pypsa.common import check_optional_dependency
from pypsa.network.io._common import (
    _Exporter,
    _get_safe_excel_sheet_name,
    _Importer,
    _retrieve_from_url,
)

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
if TYPE_CHECKING:
    from collections.abc import Iterable
logger = logging.getLogger(__name__)


class _ImporterExcel(_Importer):
    """Importer class for Excel files."""

    def __init__(self, path: str | Path, engine: str = "calamine") -> None:
        """Initialize the importer for Excel files.

        Parameters
        ----------
        path : str | Path
            Path to the Excel file.
        engine : str
            Engine to use for the Excel file.

        """
        if engine == "calamine":
            check_optional_dependency(
                "python_calamine",
                "Missing optional dependencies to use Excel files. Install them via "
                "`pip install pypsa[excel]`. If you passed any other engine, "
                "make sure it is installed.",
            )
        if not isinstance(path, (str | Path)):
            msg = f"Invalid path type. Expected str or Path, got {type(path)}."
            raise TypeError(msg)

        path = Path(path)
        if not path.is_file():
            msg = f"Excel file {path} does not exist."
            raise FileNotFoundError(msg)
        self.engine = engine

        reader = partial(pd.read_excel, sheet_name=None, engine=self.engine)
        if validators.url(str(path)):
            self.sheets = _retrieve_from_url(str(path), reader)
        else:
            self.sheets = reader(path)
        self.index: dict = {}

    def get_attributes(self) -> dict | None:
        """Get generic network attributes."""
        try:
            # Ensure name and pypsa_version are read as strings to prevent
            # automatic type conversion (e.g., numeric names like "123")
            df = self.sheets["network"]
            if "name" in df.columns:
                df["name"] = df["name"].astype(str)

                df["name"] = df["name"].replace("nan", "")
            if "pypsa_version" in df.columns:
                df["pypsa_version"] = df["pypsa_version"].astype(str)
            return dict(df.iloc[0])
        except (ValueError, KeyError):
            return None

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        try:
            df = self.sheets["meta"]
            if not df.empty:
                meta = {}
                for _, row in df.iterrows():
                    key = row["Key"]
                    value = row["Value"]

                    # Try to parse JSON strings back into dictionaries
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass

                    meta[key] = value
                return meta
        except (ValueError, KeyError):
            return {}
        else:
            return {}

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        try:
            df = self.sheets["crs"]
            if not df.empty:
                # Assuming first column is keys and second column is values
                return dict(zip(df.iloc[:, 0], df.iloc[:, 1], strict=False))
        except (ValueError, KeyError):
            return {}
        else:
            return {}

    def get_snapshots(self) -> pd.Index:
        """Get snapshots data."""
        try:
            df = self.sheets["snapshots"]
        except KeyError:
            return None
        df = df.set_index(df.columns[0])
        # Convert snapshot and timestep to datetime (if possible), unless already integer
        if "snapshot" in df and df.snapshot.dtype.kind != "i":
            try:
                df["snapshot"] = pd.to_datetime(df.snapshot)
            except (ValueError, ParserError):
                pass
        if "timestep" in df and df.timestep.dtype.kind != "i":
            try:
                df["timestep"] = pd.to_datetime(df.timestep)
            except (ValueError, ParserError):
                pass
        return df

    def get_investment_periods(self) -> pd.Series:
        """Get investment periods data."""
        try:
            df = self.sheets["investment_periods"]
            df = df.set_index(df.columns[0])
            df.index = df.index.astype(int)
        except (ValueError, KeyError):
            return None
        else:
            return df

    def get_static(self, list_name: str) -> pd.DataFrame:
        """Get static components data."""
        try:
            df = self.sheets[list_name]
            df = df.set_index(df.columns[0])

            # Handle DataFrames with only index values that were exported from PyPSA
            # Otherwise the column row is read in as a component
            if len(df.columns) == 0 and len(df.index) > 0 and df.index[0] == "name":
                df = df.iloc[1:]  # Remove the first row which contains the index name

            # Convert NaN to empty strings for string columns to handle custom attributes
            str_cols = [
                col
                for col in df.columns
                if df[col].dtype == "object"
                or isinstance(df[col].dtype, pd.StringDtype)
            ]
            if str_cols:
                df[str_cols] = df[str_cols].fillna("")

        except (ValueError, KeyError):
            return None
        else:
            return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get dynamic components data."""
        for sheet_name, df in self.sheets.items():
            sheet_name = _get_safe_excel_sheet_name(sheet_name)
            match = re.match(rf"^{list_name}-(.+)(?<!-pw)$", sheet_name)
            if match:
                attr = match.group(1)
                df = df.set_index(df.columns[0])
                yield attr, df

    def get_piecewise(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get piecewise component data."""
        for sheet_name, df in self.sheets.items():
            sheet_name = _get_safe_excel_sheet_name(sheet_name)
            match = re.match(rf"^{list_name}-(.+)-pw$", sheet_name)
            if match:
                attr = match.group(1)
                df = (
                    df.set_index(["breakpoint", "attribute"])
                    .rename_axis(columns="name")
                    .unstack("attribute", sort=False)
                    .reorder_levels(["name", "attribute"], axis=1)
                )
                yield attr, df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterExcel(_Exporter):
    """Exporter class for Excel files."""

    def __init__(self, path: Path | str, engine: str = "openpyxl") -> None:
        """Initialize the exporter for Excel files.

        Parameters
        ----------
        path : Path | str
            Path to save the Excel file.
        engine : str
            Engine to use for the Excel file.

        """
        if engine == "openpyxl":
            check_optional_dependency(
                "openpyxl",
                "Missing optional dependencies to use Excel files. Install them via "
                "`pip install pypsa[excel]`. If you passed any other engine, "
                "make sure it is installed.",
            )
        self.engine = engine
        self.path = Path(path)
        # Create an empty Excel file if it doesn't exist
        if not self.path.exists():
            logger.warning("Excel file %s does not exist, creating it", path)
            with pd.ExcelWriter(self.path, engine=self.engine) as writer:
                pd.DataFrame().to_excel(writer, sheet_name="_temp")

        # Keep track of sheets to avoid overwriting
        self._writer = None

    @property
    def writer(self) -> pd.ExcelWriter:
        """Get the Excel writer object.

        If the writer object is not already created, create it.
        """
        if self._writer is None:
            self._writer = pd.ExcelWriter(
                self.path,
                engine=self.engine,
                mode="a" if self.path.exists() else "w",
                if_sheet_exists="replace",
            )
        return self._writer

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        name = attrs.pop("name")
        df = pd.DataFrame(attrs, index=pd.Index([name], name="name"))
        df.to_excel(self.writer, sheet_name="network")

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        # Convert meta dictionary to DataFrame with proper handling of nested dicts
        meta_items = []
        for key, value in meta.items():
            # If value is a dict, serialize it as JSON
            if isinstance(value, dict):
                value = json.dumps(value)
            meta_items.append([key, value])

        df = pd.DataFrame(meta_items, columns=["Key", "Value"])
        df.to_excel(self.writer, sheet_name="meta", index=False)

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        df = pd.DataFrame(list(crs.items()), columns=["Key", "Value"])
        df.to_excel(self.writer, sheet_name="crs", index=False)

    def save_snapshots(self, snapshots: pd.Index) -> None:
        """Save snapshots data."""
        snapshots.to_excel(self.writer, sheet_name="snapshots")

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        investment_periods.to_excel(self.writer, sheet_name="investment_periods")

    def save_scenarios(self, scenarios: pd.DataFrame) -> None:
        """Save scenarios data."""
        msg = "Stochastic networks are not supported in the Excel exporter. Use netcdf instead."
        raise NotImplementedError(msg)

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save static components data."""
        df.to_excel(self.writer, sheet_name=list_name)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""
        sheet_name = f"{list_name}-{attr}"
        sheet_name = _get_safe_excel_sheet_name(sheet_name)
        df.to_excel(self.writer, sheet_name=sheet_name)

    def save_piecewise(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save piecewise component data."""
        sheet_name = f"{list_name}-{attr}-pw"
        sheet_name = _get_safe_excel_sheet_name(sheet_name)
        df_stack = df.stack("attribute", future_stack=True).reset_index()
        df_stack.to_excel(self.writer, sheet_name=sheet_name, index=False)

    def remove_static(self, list_name: str) -> None:
        """Remove static components data.

        Needed to not have stale sheets for empty components.

        """
        if list_name in self.writer.book.sheetnames:
            del self.writer.book[list_name]
            logger.warning("Stale sheet %s removed", list_name)

    def remove_series(self, list_name: str, attr: str) -> None:
        """Remove dynamic components data.

        Needed to not have stale sheets for empty components.
        """
        sheet_name = f"{list_name}-{attr}"
        sheet_name = _get_safe_excel_sheet_name(sheet_name)
        if sheet_name in self.writer.book.sheetnames:
            del self.writer.book[sheet_name]
            logger.warning("Stale sheet %s removed", sheet_name)

    def remove_piecewise(self, list_name: str, attr: str) -> None:
        """Remove piecewise component data.

        Needed to not have stale sheets for empty components.
        """
        sheet_name = f"{list_name}-{attr}-pw"
        sheet_name = _get_safe_excel_sheet_name(sheet_name)
        if sheet_name in self.writer.book.sheetnames:
            del self.writer.book[sheet_name]
            logger.warning("Stale sheet %s removed", sheet_name)

    def finish(self) -> None:
        """Postprocessing of exporting process."""
        # Remove temp sheet if it exists
        if "_temp" in self.writer.book.sheetnames:
            del self.writer.book["_temp"]
        # Close writer
        if self.writer is not None:
            self.writer.close()
