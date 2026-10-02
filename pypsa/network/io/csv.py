# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Importer and exporter for CSV files."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import pandas as pd

from pypsa.network.io._common import _Exporter, _Importer

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
if TYPE_CHECKING:
    from collections.abc import Iterable
logger = logging.getLogger(__name__)


class _ImporterCSV(_Importer):
    """Importer class for CSV files."""

    def __init__(self, path: str | Path, encoding: str | None, quotechar: str) -> None:
        """Initialize the importer for CSV files.

        Parameters
        ----------
        path : str | Path
            Path to the CSV folder.
        encoding : str | None
            Encoding to use for the CSV files.
        quotechar : str
            Quote character to use for the CSV files.

        """
        self.path = Path(path)
        self.encoding = encoding
        self.quotechar = quotechar

        if not self.path.is_dir():
            msg = f"Directory {path} does not exist."
            raise FileNotFoundError(msg)

    def get_attributes(self) -> dict | None:
        """Get generic network attributes."""
        fn = self.path.joinpath("network.csv")
        if not fn.is_file():
            return None

        dtypes = {"pypsa_version": str, "name": str}
        return dict(
            pd.read_csv(
                fn, encoding=self.encoding, dtype=dtypes, quotechar=self.quotechar
            ).iloc[0]
        )

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        fn = self.path.joinpath("meta.json")
        return {} if not fn.is_file() else json.loads(fn.open().read())

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        fn = self.path.joinpath("crs.json")
        return {} if not fn.is_file() else json.loads(fn.open().read())

    def get_snapshots(self) -> pd.Index:
        """Get snapshots data."""
        fn = self.path.joinpath("snapshots.csv")
        if not fn.is_file():
            return None
        return pd.read_csv(
            fn,
            index_col=0,
            encoding=self.encoding,
            quotechar=self.quotechar,
        )

    def get_investment_periods(self) -> pd.Series:
        """Get investment periods data."""
        fn = self.path.joinpath("investment_periods.csv")
        if not fn.is_file():
            return None
        return pd.read_csv(
            fn, index_col=0, encoding=self.encoding, quotechar=self.quotechar
        )

    def get_static(self, list_name: str) -> pd.DataFrame:
        """Get static components data."""
        fn = self.path.joinpath(list_name + ".csv")
        if not fn.is_file():
            return None

        df = pd.read_csv(
            fn, index_col=0, encoding=self.encoding, quotechar=self.quotechar
        )

        # Convert NaN to empty strings for string columns to handle custom attributes
        str_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype)
        ]
        if str_cols:
            df[str_cols] = df[str_cols].fillna("")

        return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get dynamic components data."""
        for fn in self.path.iterdir():
            match = re.match(rf"^{list_name}-(.+)(?<!-pw)\.csv$", fn.name)
            if match:
                attr = match.group(1)
                df = pd.read_csv(
                    self.path.joinpath(fn.name),
                    index_col=0,
                    encoding=self.encoding,
                    quotechar=self.quotechar,
                )
                yield attr, df

    def get_piecewise(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get piecewise component data."""
        for fn in self.path.iterdir():
            match = re.match(rf"^{list_name}-(.+)-pw\.csv$", fn.name)
            if match:
                attr = match.group(1)
                df = pd.read_csv(
                    self.path.joinpath(fn.name),
                    index_col=0,
                    header=[0, 1],
                    encoding=self.encoding,
                    quotechar=self.quotechar,
                )
                yield attr, df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterCSV(_Exporter):
    """Exporter class for CSV files."""

    def __init__(self, path: Path | str, encoding: str | None, quotechar: str) -> None:
        """Initialize the exporter for CSV files.

        Parameters
        ----------
        path : Path | str
            Path to the CSV folder.
        encoding : str | None
            Encoding to use for the CSV files.
        quotechar : str
            Quote character to use for the CSV files.

        """
        self.path = Path(path)
        self.encoding = encoding
        self.quotechar = quotechar

        # make sure directory exists
        if not self.path.is_dir():
            logger.warning("Directory %s does not exist, creating it", path)
            self.path.mkdir()

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        name = attrs.pop("name")
        df = pd.DataFrame(attrs, index=pd.Index([name], name="name"))
        fn = self.path.joinpath("network.csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        fn = self.path.joinpath("meta.json")
        fn.open("w").write(json.dumps(meta))

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        fn = self.path.joinpath("crs.json")
        fn.open("w").write(json.dumps(crs))

    def save_snapshots(self, snapshots: pd.Index) -> None:
        """Save snapshots data."""
        fn = self.path.joinpath("snapshots.csv")
        with fn.open("w"):
            snapshots.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        fn = self.path.joinpath("investment_periods.csv")
        with fn.open("w"):
            investment_periods.to_csv(
                fn, encoding=self.encoding, quotechar=self.quotechar
            )

    def save_scenarios(self, scenarios: pd.DataFrame) -> None:
        """Save scenarios data."""
        msg = "Stochastic networks are not supported in the CSV exporter. Use netcdf instead."
        raise NotImplementedError(msg)

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save static components data."""
        fn = self.path.joinpath(list_name + ".csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""
        fn = self.path.joinpath(list_name + "-" + attr + ".csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def save_piecewise(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save piecewise component data."""
        fn = self.path.joinpath(f"{list_name}-{attr}-pw.csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def remove_static(self, list_name: str) -> None:
        """Remove static components data.

        Needed to not have stale sheets for empty components.
        """
        if fns := list(self.path.joinpath(list_name).glob("*.csv")):
            for fn in fns:
                fn.unlink()
            logger.warning("Stale csv file(s) %s removed", ", ".join(fns))

    def remove_series(self, list_name: str, attr: str) -> None:
        """Remove dynamic components data.

        Needed to not have stale sheets for empty components.
        """
        fn = self.path.joinpath(list_name + "-" + attr + ".csv")
        if fn.exists():
            fn.unlink()

    def remove_piecewise(self, list_name: str, attr: str) -> None:
        """Remove piecewise component data.

        Needed to not have stale sheets for empty components.
        """
        fn = self.path.joinpath(f"{list_name}-{attr}-pw.csv")
        if fn.exists():
            fn.unlink()

    def finish(self) -> None:
        """Finish the export process."""
