# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Functions for importing and exporting data."""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import parse as parse_version
from pyproj import CRS

from pypsa.common import _check_for_update, check_optional_dependency
from pypsa.consistency import check_for_unknown_buses
from pypsa.constants import piecewise_attrs
from pypsa.descriptors import (
    _update_ports_component_attrs,
)
from pypsa.network.abstract import _NetworkABC
from pypsa.network.io._common import _coerce_string_dtypes, _Exporter, _sort_attrs
from pypsa.network.io.csv import _ExporterCSV, _ImporterCSV
from pypsa.network.io.excel import _ExporterExcel, _ImporterExcel
from pypsa.network.io.hdf5 import _ExporterHDF5, _ImporterHDF5
from pypsa.network.io.netcdf import _ExporterNetCDF, _ImporterNetCDF
from pypsa.version import __version_base__

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
if TYPE_CHECKING:
    from datarecord.record import RecordLike
    from pandapower.auxiliary import pandapowerNet

    from pypsa import Network
logger = logging.getLogger(__name__)


class NetworkIOMixin(_NetworkABC):
    """Mixin class for network I/O methods.

    <!-- md:guide import-export.md -->

    Class inherits to [pypsa.Network][]. All attributes and methods can be used
    within any Network instance.
    """

    def _export_to_exporter(
        self,
        exporter: _Exporter,
        quotechar: str = '"',
        export_standard_types: bool = False,
    ) -> None:
        """Export to exporter.

        Both static and series attributes of components are exported, but only
        if they have non-default values.

        Parameters
        ----------
        exporter : _Exporter
            Initialized exporter instance
        quotechar : str, default '"'
            String of length 1. Character used to denote the start and end of a
            quoted item. Quoted items can include "," and it will be ignored
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the netowrk).

        """
        # exportable component types
        allowed_types = (float, int, bool, str) + tuple(np.sctypeDict.values())
        skip_attrs = {
            "component_attrs",
            "df",
            "pnl",
            "static",
            "dynamic",
            "iterate_components",
            "_name",
            "_pypsa_version",
        }

        _attrs = {}
        for attr in dir(self):
            if attr.startswith("__") or attr in skip_attrs:
                continue
            # Skip read-only properties (except pypsa_version) without invoking
            # their getters, which may emit warnings (e.g. model, objective).
            prop = getattr(self.__class__, attr, None)
            if (
                isinstance(prop, property)
                and prop.fset is None
                and attr != "pypsa_version"
            ):
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*component_attrs is deprecated as of 1\.0 and will be removed in 2\.0\..*",
                    category=DeprecationWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r".*the API for how to access components data has changed.*",
                    category=DeprecationWarning,
                )
                value = getattr(self, attr)
            if isinstance(value, allowed_types):
                _attrs[attr] = value
        exporter.save_attributes(_attrs)

        crs = {}
        if self.crs is not None:
            crs["_crs"] = self.crs.to_wkt()
        exporter.save_crs(crs)

        exporter.save_meta(self.meta)

        # export snapshots
        snapshots = self.snapshot_weightings.reset_index()
        exporter.save_snapshots(snapshots)

        # export investment period weightings
        if self.has_periods:
            investment_periods = self.investment_period_weightings
            exporter.save_investment_periods(investment_periods)

        # export scenarios
        if self.has_scenarios:
            exporter.save_scenarios(self.scenario_weightings)

        exported_components = []
        for component in self.all_components:
            c = self.components[component]
            list_name = c["list_name"]
            attrs = c["defaults"]

            static = c.static
            dynamic = c.dynamic
            piecewise = c.piecewise

            if component == "Shape":
                static = pd.DataFrame(static).assign(
                    geometry=static["geometry"].to_wkt()
                )

            if not export_standard_types and component in self.standard_type_components:
                if isinstance(static.index, pd.MultiIndex):
                    static = static.drop(c["standard_types"].index, level="name")
                else:
                    static = static.drop(c["standard_types"].index)

            col_export = []
            for col in static.columns:
                # do not export derived attributes and object column of subnetwork
                if col in ["g_pu", "b_pu"]:
                    continue
                if (
                    col in attrs.index
                    and pd.isnull(attrs.at[col, "default"])
                    and pd.isnull(static[col]).all()
                ):
                    continue
                if (
                    col in attrs.index
                    and static[col].dtype == attrs.at[col, "dtype"]
                    and (static[col] == attrs.at[col, "default"]).all()
                ):
                    continue

                col_export.append(col)

            # first do static attributes
            if static.empty:
                exporter.remove_static(list_name)
                continue

            static_export = static[col_export].copy()
            # Stored SubNetwork obj column is not serializable
            if "obj" in col_export and component == "SubNetwork":
                static_export["obj"] = np.nan

            exporter.save_static(list_name, static_export)

            # now do varying attributes
            for attr in dynamic:
                if attr not in attrs.index:
                    col_export = dynamic[attr].columns
                else:
                    default = attrs.at[attr, "default"]

                    if pd.isnull(default):
                        col_export = dynamic[attr].columns[
                            (~pd.isnull(dynamic[attr])).any()
                        ]
                    else:
                        col_export = dynamic[attr].columns[
                            (dynamic[attr] != default).any()
                        ]

                if len(col_export) > 0:
                    static = dynamic[attr].reset_index()[col_export]
                    exporter.save_series(list_name, attr, static)
                else:
                    exporter.remove_series(list_name, attr)

            # now do piecewise attributes
            for attr, pw_df in piecewise.items():
                if not pw_df.empty:
                    exporter.save_piecewise(list_name, attr, pw_df)
                else:
                    exporter.remove_piecewise(list_name, attr)
            exported_components.append(list_name)

        logger.info(
            "Exported network '%s'%s contains: %s",
            self.name,
            f" saved to '{exporter.path}" if exporter.path else "",
            ", ".join(exported_components),
        )

    def _import_from_importer(
        self, importer: Any, basename: str, skip_time: bool = False
    ) -> None:
        """Import network data from importer.

        Parameters
        ----------
        importer : Any
            Importer to import from.
        basename : str
            Name of the network.
        skip_time : bool
            Skip importing time

        """
        # n.meta
        self.meta = importer.get_meta()

        # n.crs
        crs = importer.get_crs()
        crs = crs.pop("_crs", None)
        if crs is not None:
            crs = CRS.from_wkt(crs)
            self._crs = crs

        # other network attributes
        attrs = importer.get_attributes() or {}
        if "name" in attrs:
            name = attrs.pop("name")
            if pd.notna(name):
                self.name = name

        if "pypsa_version" in attrs:
            pypsa_version = parse_version(attrs.pop("pypsa_version", "0.0.0"))
        else:
            pypsa_version = parse_version("0.0.0")

        for attr, val in attrs.items():
            if attr in ["model", "objective", "objective_constant"]:
                setattr(self, f"_{attr}", val)
            else:
                setattr(self, attr, val)

        ## https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
        if pypsa_version < parse_version(__version_base__):
            pypsa_version_str = str(pypsa_version)
            logger.warning(
                "Importing network from PyPSA version v%s while current version is v%s. Read the "
                "release notes at `https://go.pypsa.org/release-notes` "
                "to prepare your network for import.",
                pypsa_version_str,
                __version_base__,
            )

        # Check for newer PyPSA version available
        update_msg = _check_for_update(__version_base__, "PyPSA", "pypsa")
        if update_msg:
            logger.info(update_msg)

        if pypsa_version < parse_version("0.18.0"):
            self._multi_invest = 0

        # if there is snapshots.csv, read in snapshot data
        df = importer.get_snapshots()

        if df is not None:
            if snapshot_levels := {"period", "timestep", "snapshot"}.intersection(
                df.columns
            ):
                df = df.set_index(sorted(snapshot_levels))
            self.set_snapshots(df.index)

            cols = ["objective", "stores", "generators"]
            if not df.columns.intersection(cols).empty:
                # Preserve the default column order from Network.__init__
                existing_cols = [col for col in cols if col in df.columns]
                self.snapshot_weightings = df.reindex(
                    index=self.snapshots, columns=existing_cols
                )
            elif "weightings" in df.columns:
                self.snapshot_weightings = df["weightings"].reindex(self.snapshots)

        # read in investment period weightings
        periods = importer.get_investment_periods()

        if periods is not None and not periods.empty:
            self.periods = periods.index

            self._investment_periods_data = periods.reindex(self.investment_periods)

        scenarios = importer.get_scenarios()
        if scenarios is not None:
            self.scenarios = scenarios

        imported_components = []

        # now read in other components; make sure buses and carriers come first
        for component in ["Bus", "Carrier"] + sorted(
            self.all_components - {"Bus", "Carrier"}
        ):
            list_name = self.components[component]["list_name"]

            df = importer.get_static(list_name)
            if df is None:
                if component == "Bus":
                    logger.error("Error, no buses found")
                    return
                continue

            if component in ("Link", "Process"):
                _update_ports_component_attrs(self, where=df, c_name=component)

            self._import_components_from_df(df, component)

            if not skip_time:
                for attr, df in importer.get_series(list_name):
                    df.set_index(self.snapshots, inplace=True)
                    self._import_series_from_df(df, component, attr)

            for attr, df in importer.get_piecewise(list_name):
                self._import_piecewise_from_df(df, component, attr)

            logger.debug(getattr(self, list_name))

            imported_components.append(list_name)

        self._broadcast_standard_types()

        logger.info(
            "Imported network '%s' has %s",
            self.name,
            ", ".join(imported_components),
        )

    def _broadcast_standard_types(self) -> None:
        """Broadcast each standard-type static table across scenarios, once per import."""
        for component in self.standard_type_components:
            comp = self.components[component]
            if self.has_scenarios and not isinstance(comp.static.index, pd.MultiIndex):
                comp.static = pd.concat(
                    dict.fromkeys(self.scenarios, comp.static), names=["scenario"]
                )

    def import_from_csv_folder(
        self,
        path: str | Path,
        encoding: str | None = None,
        quotechar: str = '"',
        skip_time: bool = False,
    ) -> None:
        """Import network data from CSVs in a folder.

        The CSVs must follow the standard form, see `pypsa/examples`.

        Parameters
        ----------
        path : string
            Name of folder
        encoding : str, default None
            Encoding to use for UTF when reading (ex. 'utf-8'). See [List of Python
            standard encodings](https://docs.python.org/3/library/codecs.html#standard-encodings)
        quotechar : str, default '"'
            String of length 1. Character used to denote the start and end of a
            quoted item. Quoted items can include "," and it will be ignored
        skip_time : bool, default False
            Skip reading in time dependent attributes

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_csv_folder"./my_network") # doctest: +SKIP

        """
        basename = Path(path).name
        with _ImporterCSV(path, encoding=encoding, quotechar=quotechar) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    def export_to_csv_folder(
        self,
        path: Path | str,
        encoding: str | None = None,
        quotechar: str = '"',
        export_standard_types: bool = False,
    ) -> None:
        """Export network and components to a folder of CSVs.

        Both static and series attributes of all components are exported, but only
        if they have non-default values.

        If `path` does not already exist, it is created.

        `path` may also be a cloud object storage URI if cloudpathlib is installed.

        Static attributes are exported in one CSV file per component,
        e.g. `generators.csv`.

        Series attributes are exported in one CSV file per component per
        attribute, e.g. `generators-p_set.csv`.

        Parameters
        ----------
        path : Path | str
            Name of folder to which to export.
        encoding : str, default None
            Encoding to use for UTF when reading (ex. 'utf-8'). See [List of Python
            standard encodings](https://docs.python.org/3/library/codecs.html#standard-encodings)
        quotechar : str, default '"'
            String of length 1. Character used to quote fields.
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the network).

        Examples
        --------
        >>> n.export_to_csv_folder("my_network") # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.export_to_netcdf][], [pypsa.Network.export_to_hdf5][],
        [pypsa.Network.export_to_excel][]

        """
        with _ExporterCSV(
            path=path, encoding=encoding, quotechar=quotechar
        ) as exporter:
            self._export_to_exporter(
                exporter, export_standard_types=export_standard_types
            )

    def import_from_excel(
        self,
        path: str | Path,
        skip_time: bool = False,
        engine: str = "calamine",
    ) -> None:
        """Import network data from an Excel file.

        The Excel file must follow the standard form with appropriate sheets.

        Parameters
        ----------
        path : string or Path
            Path to the Excel file
        skip_time : bool, default False
            Skip reading in time dependent attributes
        engine : string, default "calamine"
            The engine to use for reading the Excel file. See [pandas.read_excel
            ](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html).

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_excel("my_network.xlsx") # doctest: +SKIP

        """
        basename = Path(path).stem
        with _ImporterExcel(path, engine=engine) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    def export_to_excel(
        self,
        path: str | Path,
        export_standard_types: bool = False,
        engine: str = "openpyxl",
    ) -> None:
        """Export network and components to an Excel file.

        It is recommended to only use the Excel format if needed and for small networks.
        Excel files are not as efficient as other formats and can be slow to read/write.

        Both static and series attributes of all components are exported, but only
        if they have non-default values.

        If `path` does not already exist, it is created.

        Static attributes are exported in one sheet per component,
        e.g. a sheet named `generators`.

        Series attributes are exported in one sheet per component per
        attribute, e.g. a sheet named `generators-p_set`.

        Parameters
        ----------
        path : string or Path
            Path to the Excel file to which to export.
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the network).
        engine : string, default "openpyxl"
            The engine to use for writing the Excel file. See [pandas.ExcelWriter
            ](https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html).

        Examples
        --------
        >>> n.export_to_excel("my_network.xlsx") # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.export_to_netcdf][], [pypsa.Network.export_to_hdf5][],
        [pypsa.Network.export_to_csv_folder][]

        """
        with _ExporterExcel(path, engine=engine) as exporter:
            self._export_to_exporter(
                exporter, export_standard_types=export_standard_types
            )

    def import_from_hdf5(self, path: str | Path, skip_time: bool = False) -> None:
        """Import network data from HDF5 store at `path`.

        Parameters
        ----------
        path : string, Path
            Name of HDF5 store. The string could be a URL.
        skip_time : bool, default False
            Skip reading in time dependent attributes

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_hdf5("my_network.h5") # doctest: +SKIP

        """
        basename = Path(path).name

        with _ImporterHDF5(path) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    def export_to_hdf5(
        self,
        path: Path | str,
        export_standard_types: bool = False,
        **kwargs: Any,
    ) -> None:
        """Export network and components to an HDF store.

        Both static and series attributes of components are exported, but only
        if they have non-default values.

        If path does not already exist, it is created.

        `path` may also be a cloud object storage URI if cloudpathlib is installed.

        Parameters
        ----------
        path : string
            Name of hdf5 file to which to export (if it exists, it is overwritten)
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the network).
        **kwargs
            Extra arguments for pd.HDFStore to specify f.i. compression
            (default: complevel=4)

        Examples
        --------
        >>> n.export_to_hdf5("my_network.h5") # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.export_to_netcdf][], [pypsa.Network.export_to_csv_folder][],
        [pypsa.Network.export_to_excel][]

        """
        kwargs.setdefault("complevel", 4)

        with _ExporterHDF5(path, **kwargs) as exporter:
            self._export_to_exporter(
                exporter,
                export_standard_types=export_standard_types,
            )

    def import_from_netcdf(
        self, path: str | Path | xr.Dataset, skip_time: bool = False
    ) -> None:
        """Import network data from netCDF file or xarray Dataset at `path`.

        `path` may also be a cloud object storage URI if cloudpathlib is installed.

        Parameters
        ----------
        path : string | Path | xr.Dataset
            Path to netCDF dataset or instance of xarray Dataset.
            The string could be a URL.
        skip_time : bool, default False
            Skip reading in time dependent attributes

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_netcdf("my_network.nc") # doctest: +SKIP

        """
        basename = "" if isinstance(path, xr.Dataset) else Path(path).name
        with _ImporterNetCDF(path=path) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    def export_to_netcdf(
        self,
        path: Path | str | None = None,
        export_standard_types: bool = False,
        compression: dict | None = None,
        float32: bool = False,
    ) -> xr.Dataset:
        r"""Export network and components to a netCDF file.

        Both static and series attributes of components are exported, but only
        if they have non-default values.

        If path does not already exist, it is created.

        If no path is passed, no file is exported, but the xarray.Dataset
        is still returned.

        Be aware that this cannot export boolean attributes on the Network
        class, e.g. n.my_bool = False is not supported by netCDF.

        Parameters
        ----------
        path : Path | string | None
            Name of netCDF file to which to export (if it exists, it is overwritten);
            if None is passed, no file is exported and only the xarray.Dataset is returned.
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the network).
        compression : dict|None
            Compression level to use for all features which are being prepared.
            The compression is handled via xarray.Dataset.to_netcdf(...). For details see:
            [xarray.Dataset.to\_netcdf](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html)
            An example compression directive is `{'zlib': True, 'complevel': 4}`.
            The default is None which disables compression.
        float32 : boolean, default False
            If True, typecasts values to float32.

        Returns
        -------
        ds : xarray.Dataset

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.export_to_netcdf("my_file.nc") # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.export_to_hdf5][], [pypsa.Network.export_to_csv_folder][],
        [pypsa.Network.export_to_excel][]

        """
        with _ExporterNetCDF(path, compression, float32) as exporter:
            self._export_to_exporter(
                exporter, export_standard_types=export_standard_types
            )
            return exporter.ds

    def _import_components_from_df(
        self, df: pd.DataFrame, cls_name: str, overwrite: bool = False
    ) -> None:
        """Import components from a pandas DataFrame.

        If columns are missing then defaults are used.

        If extra columns are added, these are left in the resulting component dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame whose index is the names of the components and
            whose columns are the non-default attributes.
        cls_name : string
            Name of class of component, e.g. `"Line", "Bus", "Generator", "StorageUnit"`
        overwrite : bool, default False
            If True, overwrite existing components.

        """
        if cls_name in ("Link", "Process"):
            _update_ports_component_attrs(self, where=df, c_name=cls_name)

        attrs = self.components[cls_name]["defaults"]
        static_attrs = attrs[attrs.static].drop("name")
        non_static_attrs = attrs[~attrs.static]

        # Clean dataframe and ensure correct types
        df = pd.DataFrame(df)
        # Handle single and multi-index
        df.index = (
            df.index.astype(str)
            if not isinstance(df.index, pd.MultiIndex)
            else df.index.set_levels([level.astype(str) for level in df.index.levels])
        )

        # Fill nan values with default values
        df = df.fillna(attrs["default"].to_dict())

        for k in static_attrs.index:
            if k not in df.columns:
                df[k] = static_attrs.at[k, "default"]
            else:
                if static_attrs.at[k, "type"] == "string":
                    df[k] = df[k].replace({np.nan: ""})
                if static_attrs.at[k, "type"] == "int":
                    df[k] = df[k].fillna(0)
                if df[k].dtype != static_attrs.at[k, "typ"]:
                    if static_attrs.at[k, "type"] == "geometry":
                        geometry = df[k].replace({"": None, np.nan: None})
                        from shapely.geometry.base import BaseGeometry  # noqa: PLC0415

                        if geometry.apply(lambda x: isinstance(x, BaseGeometry)).all():
                            df[k] = gpd.GeoSeries(geometry)
                        else:
                            df[k] = gpd.GeoSeries.from_wkt(geometry)
                    else:
                        df[k] = df[k].astype(static_attrs.at[k, "typ"])

        non_static_attrs_in_df = non_static_attrs.index.intersection(df.columns)
        old_static = self.c[cls_name].static
        new_static = df.drop(non_static_attrs_in_df, axis=1)

        # Handle duplicates
        duplicated_components = old_static.index.intersection(new_static.index)
        if len(duplicated_components) > 0:
            if not overwrite:
                logger.warning(
                    "The following %s are already defined and will be skipped "
                    "(use overwrite=True to overwrite): %s",
                    self.components[cls_name]["list_name"],
                    ", ".join(duplicated_components),
                )
                new_static = new_static.drop(duplicated_components)
            else:
                self.remove(cls_name, duplicated_components)

        # Concatenate to new dataframe
        if not old_static.empty:
            new_static = pd.concat((old_static, new_static), sort=False)

        if cls_name == "Shape":
            new_static = gpd.GeoDataFrame(new_static, crs=self.crs)

        # Align index (component names) and columns (attributes)
        ordered_columns = _sort_attrs(new_static.columns, attrs.index)
        if not new_static.columns.equals(ordered_columns):
            if isinstance(new_static.columns, pd.MultiIndex):
                new_static = new_static.loc[:, ordered_columns]
            else:
                indexer = new_static.columns.get_indexer(ordered_columns)
                if (indexer >= 0).all():
                    new_static = new_static.iloc[:, indexer]
                else:
                    new_static = new_static.loc[:, ordered_columns]

        new_static.index.names = (
            ["name"]
            if not isinstance(new_static.index, pd.MultiIndex)
            else ["scenario", "name"]
        )
        new_static = _coerce_string_dtypes(new_static)
        self.components[cls_name].static = new_static

        # Now deal with time-dependent properties

        dynamic = self.c[cls_name].dynamic
        for k in non_static_attrs_in_df:
            # If reading in outputs, fill the outputs
            dynamic[k] = dynamic[k].reindex(
                columns=new_static.index, fill_value=non_static_attrs.at[k, "default"]
            )
            if overwrite:
                dynamic[k].loc[:, df.index] = df.loc[:, k].values
            else:
                new_components = df.index.difference(duplicated_components)
                dynamic[k].loc[:, new_components] = df.loc[new_components, k].values

        self.components[cls_name].dynamic = dynamic

        # Run consistency checks
        check_for_unknown_buses(self, self.c[cls_name])

    def _import_series_from_df(
        self,
        df: pd.DataFrame,
        cls_name: str,
        attr: str,
        overwrite: bool = False,
    ) -> None:
        """Import time series from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame whose index is `n.snapshots` and
            whose columns are a subset of the relevant components.
        cls_name : string
            Name of class of component
        attr : string
            Name of time-varying series attribute
        overwrite : bool, default False
            If True, overwrite existing time series.

        """
        static = self.c[cls_name].static
        dynamic = self.c[cls_name].dynamic
        list_name = self.components[cls_name]["list_name"]

        if not overwrite:
            try:
                df = df.drop(df.columns.intersection(dynamic[attr].columns), axis=1)
            except KeyError:
                pass  # Don't drop any columns if the data doesn't exist yet

        # df.columns.names = ["name"] if not isinstance(df.index, pd.MultiIndex) else ["scenario", "name"]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns.names = ["scenario", "name"]
        else:
            df.columns.names = ["name"]
        df = _coerce_string_dtypes(df)

        # Check if components exist in static df
        diff = df.columns.difference(static.index)
        if len(diff) > 0:
            logger.warning(
                "Components %s for attribute %s of %s are not in main components dataframe %s",
                diff,
                attr,
                cls_name,
                list_name,
            )

        # Get all attributes for the component
        attrs = self.components[cls_name]["defaults"]

        # Add all unknown attributes to the dataframe without any checks
        expected_attrs = attrs[lambda ds: ds.type.str.contains("series")].index
        if attr not in expected_attrs:
            if overwrite or attr not in dynamic:
                dynamic[attr] = df
            return

        # Check if any snapshots are missing
        diff = self.snapshots.difference(df.index)
        if len(diff):
            logger.warning(
                "Snapshots %s are missing from %s of %s. Filling with default value '%s'",
                diff,
                attr,
                cls_name,
                attrs.loc[attr].default,
            )
            df = df.reindex(self.snapshots, fill_value=attrs.loc[attr].default)

        if not attrs.loc[attr].static:
            # Preserve static component order for consistency
            ordered_columns = _sort_attrs(
                df.columns.union(static.index),
                static.index,
            )
            dynamic[attr] = dynamic[attr].reindex(
                columns=ordered_columns,
                fill_value=attrs.loc[attr].default,
            )
        else:
            # Preserve existing dynamic order for static attrs
            ordered_columns = _sort_attrs(
                df.columns.union(dynamic[attr].columns),
                dynamic[attr].columns,
            )
            dynamic[attr] = dynamic[attr].reindex(columns=ordered_columns)

        dynamic[attr].loc[self.snapshots, df.columns] = df.loc[
            self.snapshots, df.columns
        ]

    @staticmethod
    def _normalize_breakpoints(
        piecewise_df: pd.DataFrame, piecewise_attrs: pd.Series
    ) -> pd.DataFrame:
        """Sort segment rows by x-coordinate and align ragged curves with trailing NaNs."""
        x_attr, y_attr = piecewise_attrs.x, piecewise_attrs.y

        def __normalize(curve: pd.DataFrame) -> pd.DataFrame:
            filled = curve.notna().any()
            has_later = filled.iloc[::-1].cummax().iloc[::-1]
            if (gap := ~filled & has_later).any():
                msg = (
                    f"Piecewise '{y_attr}' segments for component '{curve.name}' contain "
                    f"non-trailing missing breakpoint rows: {gap[gap].index.tolist()}."
                )
                raise ValueError(msg)
            if (partial := filled & curve.isna().any()).any():
                msg = (
                    f"Piecewise '{y_attr}' segments for component '{curve.name}' have "
                    f"incomplete breakpoint data at rows: {partial[partial].index.tolist()}."
                )
                raise ValueError(msg)
            return curve.loc[:, filled].sort_values(
                (curve.name, x_attr), axis=1, kind="mergesort"
            )

        return (
            piecewise_df.T.groupby(level="name", group_keys=False)
            .apply(__normalize)
            .T.reset_index(drop=True)
            .rename_axis(index="breakpoint")
        )

    def _import_piecewise_from_df(
        self,
        df: pd.DataFrame,
        cls_name: str,
        attr: str,
        is_extendable: pd.Series | None = None,
        overwrite: bool = False,
    ) -> None:
        """Import piecewise breakpoint data from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with MultiIndex columns ``(name, attribute)`` where the
            attribute level holds ``[x_attr, attr]`` (the x-axis coordinate and
            the y-axis attribute) and whose index is the breakpoint number.
        cls_name : str
            Component class name, e.g. ``"Generator"``.
        attr : str
            Piecewise y-axis attribute name, e.g. ``"efficiency"``.
        is_extendable : pd.Series or None, default None
            If Series, a boolean series where True = extendable.
            if None, it is inferred from the component's extendables.
        overwrite : bool, default False
            If True, replace existing breakpoint data for the component.

        """
        idx_name = "breakpoint"
        col_names = ["name", "attribute"]
        c = self.c[cls_name]
        pw_attr = c._piecewise_schema(attr)
        if pw_attr.empty:
            valid_attrs = piecewise_attrs(cls_name).y.unique().tolist()
            msg = (
                f"'{attr}' is not a recognised piecewise attribute for {cls_name}. "
                f"Known piecewise attributes: {valid_attrs}."
            )
            raise ValueError(msg)

        x_attr = pw_attr.x

        # df must have MultiIndex columns (name, attribute) with x_attr and attr present
        if not isinstance(df.columns, pd.MultiIndex):
            msg = (
                "Pass a MultiIndex-columned DataFrame to _import_piecewise_from_df "
                "with levels ['name', 'attribute']. Use network.add() for the "
                "user-facing interface."
            )
            raise TypeError(msg)

        df.index = df.index.set_names(idx_name)
        df.columns = df.columns.set_names(col_names)

        df = self._normalize_breakpoints(df, pw_attr)
        if not pw_attr.allow_extendable:
            new_names = df.columns.unique("name")
            if is_extendable is not None:
                extendable_i = is_extendable[is_extendable].index
            else:
                extendable_i = self.c[cls_name].extendables
            bad = new_names.intersection(extendable_i)
            if not bad.empty:
                msg = (
                    f"Piecewise '{attr}' breakpoints are not supported for extendable "
                    f"components (fixed p_nom required). Extendable components: "
                    f"{bad.tolist()}."
                )
                raise ValueError(msg)
        if pw_attr.y in ("rate", "efficiency"):
            curve = df.xs(attr, level="attribute", axis=1)
            is_pos = ((curve >= 0) | curve.isna()).all()
            is_neg = ((curve <= 0) | curve.isna()).all()
            if not (bad := curve.columns[~(is_pos | is_neg)]).empty:
                msg = f"Cannot mix positive and negative values for piecewise {attr} curves of {c} components {bad.tolist()}"
                raise NotImplementedError(msg)

        if (bad := df.loc[0].loc[pd.IndexSlice[:, x_attr]] > 0).any():
            equivalents = {"p_pu": "p_min_pu", "p_nom": "p_nom_min"}
            msg = (
                f"Piecewise '{attr}' curves must start at {x_attr}=0. "
                f"Otherwise, you implicitly bound {x_attr} from below (equivalent to setting {equivalents.get(x_attr)}). "
                f"Even if you are explicitly bounding from below, you should still set a breakpoint at (0, 0)."
                f"Affected components: {bad[bad].index.tolist()}."
            )
            raise ValueError(msg)
        if (
            x_attr.endswith("_pu")
            and (bad := df.iloc[-1].loc[pd.IndexSlice[:, x_attr]].ffill() < 1).any()
        ):
            equivalents = {"p_pu": "p_max_pu"}
            msg = (
                f"Piecewise '{attr}' curves must end at {x_attr}=1. "
                f"Otherwise, you implicitly bound {x_attr} from above (equivalent to setting {equivalents.get(x_attr)}). "
                f"Even if you are explicitly bounding from above, you should still set a breakpoint at (1, <y-value>)."
                f"Affected components: {bad[bad].index.tolist()}."
            )
            raise ValueError(msg)
        if (bad := df.loc[0].loc[pd.IndexSlice[:, attr]] > 0).any():
            preamble: str
            if "cost" in attr:
                preamble = (
                    "Piecewise '%s' values price the increment from the previous breakpoint, so the y-value at x=0 spans zero width and will be ignored. "
                    "To set the y-value for the first segment, set the value on the second breakpoint instead, e.g. {0.0: 0.0, 0.5: 40.0} for a '%s' of 40 up to 0.5 '%s'."
                )
            else:
                preamble = (
                    "A non-zero y value at x=0 for piecewise '%s' will be ignored when the piecewise constraint is defined since y values are denormalised using x values. "
                    "To approximate '%s' as non-zero at '%s'=0, consider setting it at a very small x value instead, e.g. {0.0: 0.0, 0.001: 10, 0.5: 40.0}."
                )
            msg = preamble + " Affected components: %s."
            logger.warning(msg, attr, attr, x_attr, bad[bad].index.tolist())
        piecewise = self.c[cls_name].piecewise

        if attr not in piecewise:
            piecewise[attr] = pd.DataFrame(
                index=pd.Index([], name=idx_name, dtype=int),
                columns=pd.MultiIndex.from_tuples([], names=col_names),
                dtype=float,
            )

        existing = piecewise[attr]

        attribute_values = df.columns.unique("attribute")
        if not attribute_values.symmetric_difference([x_attr, attr]).empty:
            msg = (
                f"DataFrame for piecewise attribute '{attr}' must have attribute "
                f"level values ['{x_attr}', '{attr}']. Got: {sorted(attribute_values)}."
            )
            raise ValueError(msg)

        if overwrite:
            # Drop existing entries for the new components
            new_names = df.columns.unique("name")
            keep = ~existing.columns.get_level_values("name").isin(new_names)
            existing = existing.loc[:, keep]

        # Align piecewise indices: union of existing and new
        all_piecewise = existing.index.union(df.index)
        existing = existing.reindex(all_piecewise)
        df = df.reindex(all_piecewise)

        piecewise[attr] = pd.concat([existing, df], axis=1)

    def import_from_pypower_ppc(
        self, ppc: dict, overwrite_zero_s_nom: float | None = None
    ) -> None:
        """Import network from PYPOWER PPC dictionary format version 2.

        Converts all baseMVA to base power of 1 MVA.

        For the meaning of the pypower indices, see also pypower/idx_*.

        Parameters
        ----------
        ppc : PYPOWER PPC dict
            PYPOWER PPC dictionary to import from.
        overwrite_zero_s_nom : Float or None, default None
            If a float, all branches with s_nom of 0 will be set to this value.

        Examples
        --------
        >>> from pypower.api import case30 # doctest: +SKIP
        >>> ppc = case30() # doctest: +SKIP
        >>> n.import_from_pypower_ppc(ppc) # doctest: +SKIP

        """
        version = ppc["version"]
        if int(version) != 2:
            logger.warning(
                "Warning, importing from PYPOWER may not work if PPC version is not 2!"
            )

        logger.warning(
            "Warning: Note that when importing from PYPOWER, some PYPOWER features not supported: areas, gencosts, component status"
        )

        baseMVA = ppc["baseMVA"]

        # add buses

        # integer numbering will be bus names
        index = np.array(ppc["bus"][:, 0], dtype=int)

        columns = [
            "type",
            "Pd",
            "Qd",
            "Gs",
            "Bs",
            "area",
            "v_mag_pu_set",
            "v_ang_set",
            "v_nom",
            "zone",
            "v_mag_pu_max",
            "v_mag_pu_min",
        ]

        pdf = {
            "buses": pd.DataFrame(
                index=index,
                columns=columns,
                data=ppc["bus"][:, 1 : len(columns) + 1],
            )
        }
        if (pdf["buses"]["v_nom"] == 0.0).any():
            logger.warning(
                "Warning, some buses have nominal voltage of 0., setting the nominal voltage of these to 1."
            )
            pdf["buses"].loc[pdf["buses"]["v_nom"] == 0.0, "v_nom"] = 1.0

        # rename controls
        controls = ["", "PQ", "PV", "Slack"]
        pdf["buses"]["control"] = (
            pdf["buses"].pop("type").map(lambda i: controls[int(i)])
        )

        # add loads for any buses with Pd or Qd
        pdf["loads"] = pdf["buses"].loc[
            pdf["buses"][["Pd", "Qd"]].any(axis=1), ["Pd", "Qd"]
        ]
        pdf["loads"]["bus"] = pdf["loads"].index
        pdf["loads"].rename(columns={"Qd": "q_set", "Pd": "p_set"}, inplace=True)
        pdf["loads"].index = [f"L{str(i)}" for i in range(len(pdf["loads"]))]

        # add shunt impedances for any buses with Gs or Bs

        shunt = pdf["buses"].loc[
            pdf["buses"][["Gs", "Bs"]].any(axis=1), ["v_nom", "Gs", "Bs"]
        ]

        # base power for shunt is 1 MVA, so no need to rebase here
        shunt["g"] = shunt["Gs"] / shunt["v_nom"] ** 2
        shunt["b"] = shunt["Bs"] / shunt["v_nom"] ** 2
        pdf["shunt_impedances"] = shunt.reindex(columns=["g", "b"])
        pdf["shunt_impedances"]["bus"] = pdf["shunt_impedances"].index
        pdf["shunt_impedances"].index = [
            f"S{str(i)}" for i in range(len(pdf["shunt_impedances"]))
        ]

        # add gens

        # it is assumed that the pypower p_max is the p_nom

        # could also do gen.p_min_pu = p_min/p_nom

        columns = [
            "bus",
            "p_set",
            "q_set",
            "q_max",
            "q_min",
            "v_set_pu",
            "mva_base",
            "status",
            "p_nom",
            "p_min",
            "Pc1",
            "Pc2",
            "Qc1min",
            "Qc1max",
            "Qc2min",
            "Qc2max",
            "ramp_agc",
            "ramp_10",
            "ramp_30",
            "ramp_q",
            "apf",
        ]

        index_list = [f"G{str(i)}" for i in range(len(ppc["gen"]))]

        pdf["generators"] = pd.DataFrame(
            index=index_list, columns=columns, data=ppc["gen"][:, : len(columns)]
        )

        # make sure bus name is an integer
        pdf["generators"]["bus"] = np.array(ppc["gen"][:, 0], dtype=int)

        # add branchs
        ## branch data
        # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax

        columns = [
            "bus0",
            "bus1",
            "r",
            "x",
            "b",
            "s_nom",
            "rateB",
            "rateC",
            "tap_ratio",
            "phase_shift",
            "status",
            "v_ang_min",
            "v_ang_max",
        ]

        pdf["branches"] = pd.DataFrame(
            columns=columns, data=ppc["branch"][:, : len(columns)]
        )

        pdf["branches"]["original_index"] = pdf["branches"].index

        pdf["branches"]["bus0"] = pdf["branches"]["bus0"].astype(int)
        pdf["branches"]["bus1"] = pdf["branches"]["bus1"].astype(int)

        # s_nom = 0 indicates an unconstrained line
        zero_s_nom = pdf["branches"]["s_nom"] == 0.0
        if zero_s_nom.any():
            if overwrite_zero_s_nom is not None:
                pdf["branches"].loc[zero_s_nom, "s_nom"] = overwrite_zero_s_nom
            else:
                logger.warning(
                    "Warning: there are %d branches with s_nom equal to zero, they will probably lead to infeasibilities and should be replaced with a high value using the `overwrite_zero_s_nom` argument.",
                    zero_s_nom.sum(),
                )

        # determine bus voltages of branches to detect transformers
        v_nom = pdf["branches"].bus0.map(pdf["buses"].v_nom)
        v_nom_1 = pdf["branches"].bus1.map(pdf["buses"].v_nom)

        # split branches into transformers and lines
        transformers = (
            (v_nom != v_nom_1)
            | (
                (pdf["branches"].tap_ratio != 0.0) & (pdf["branches"].tap_ratio != 1.0)
            )  # NB: PYPOWER has strange default of 0. for tap ratio
            | (pdf["branches"].phase_shift != 0)
        )
        pdf["transformers"] = pd.DataFrame(pdf["branches"][transformers])
        pdf["lines"] = pdf["branches"][~transformers].drop(
            ["tap_ratio", "phase_shift"], axis=1
        )

        # convert transformers from base baseMVA to base s_nom
        pdf["transformers"]["r"] = (
            pdf["transformers"]["r"] * pdf["transformers"]["s_nom"] / baseMVA
        )
        pdf["transformers"]["x"] = (
            pdf["transformers"]["x"] * pdf["transformers"]["s_nom"] / baseMVA
        )
        pdf["transformers"]["b"] = (
            pdf["transformers"]["b"] * baseMVA / pdf["transformers"]["s_nom"]
        )

        # correct per unit impedances
        pdf["lines"]["r"] = v_nom**2 * pdf["lines"]["r"] / baseMVA
        pdf["lines"]["x"] = v_nom**2 * pdf["lines"]["x"] / baseMVA
        pdf["lines"]["b"] = pdf["lines"]["b"] * baseMVA / v_nom**2

        if (pdf["transformers"]["tap_ratio"] == 0.0).any():
            logger.warning(
                "Warning, some transformers have a tap ratio of 0., setting the tap ratio of these to 1."
            )
            pdf["transformers"].loc[
                pdf["transformers"]["tap_ratio"] == 0.0, "tap_ratio"
            ] = 1.0

        # name them nicely
        pdf["transformers"].index = [
            f"T{str(i)}" for i in range(len(pdf["transformers"]))
        ]
        pdf["lines"].index = [f"L{str(i)}" for i in range(len(pdf["lines"]))]

        # TODO

        ##-----  OPF Data  -----##
        ## generator cost data
        # 1 startup shutdown n x1 y1 ... xn yn
        # 2 startup shutdown n c(n-1) ... c0

        for component in [
            "Bus",
            "Load",
            "Generator",
            "Line",
            "Transformer",
            "ShuntImpedance",
        ]:
            self.add(
                component,
                pdf[self.components[component]["list_name"]].index,
                **pdf[self.components[component]["list_name"]],
            )

        self.c.generators.static["control"] = self.c.generators.static.bus.map(
            self.c.buses.static["control"]
        )

        # for consistency with pypower, take the v_mag set point from the generators
        self.c.buses.static.loc[self.c.generators.static.bus, "v_mag_pu_set"] = (
            np.asarray(self.c.generators.static["v_set_pu"])
        )

    def import_from_pandapower_net(
        self,
        net: pandapowerNet,
        extra_line_data: bool = False,
        use_pandapower_index: bool = False,
    ) -> None:
        """Import PyPSA network from pandapower net.

        Importing from pandapower is still in beta;
        not all pandapower components are supported.

        Unsupported features include:
        - three-winding transformers
        - switches
        - in_service status and
        - tap positions of transformers

        Parameters
        ----------
        net : pandapower network
            pandapower network to import from.
        extra_line_data : boolean, default: False
            if True, the line data for all parameters is imported instead of only the type
        use_pandapower_index : boolean, default: False
            if True, use integer numbers which is the pandapower index standard
            if False, use any net.name as index (e.g. 'Bus 1' (str) or 1 (int))

        Examples
        --------
        >>> n.import_from_pandapower_net(net) # doctest: +SKIP
        OR

        >>> import pandapower as pp # doctest: +SKIP
        >>> import pandapower.networks as pn # doctest: +SKIP
        >>> net = pn.create_cigre_network_mv(with_der='all') # doctest: +SKIP
        >>> n = pypsa.Network()
        >>> n.import_from_pandapower_net(net, extra_line_data=True)  # doctest: +SKIP

        """
        logger.warning(
            "Warning: Importing from pandapower is still in beta; not all pandapower data is supported.\nUnsupported features include: three-winding transformers, switches, in_service status, shunt impedances and tap positions of transformers."
        )

        d = {
            "Bus": pd.DataFrame(
                {"v_nom": net.bus.vn_kv.values, "v_mag_pu_set": 1.0},
                index=net.bus.name,
            )
        }

        d["Bus"].loc[net.bus.name.loc[net.gen.bus].values, "v_mag_pu_set"] = (
            net.gen.vm_pu.values  # fmt: skip
        )

        d["Bus"].loc[net.bus.name.loc[net.ext_grid.bus].values, "v_mag_pu_set"] = (
            net.ext_grid.vm_pu.values  # fmt: skip
        )

        d["Load"] = pd.DataFrame(
            {
                "p_set": (net.load.scaling * net.load.p_mw).values,
                "q_set": (net.load.scaling * net.load.q_mvar).values,
                "bus": net.bus.name.loc[net.load.bus].values,
            },
            index=net.load.name,
        )

        # deal with PV generators
        _tmp_gen = pd.DataFrame(
            {
                "p_set": (net.gen.scaling * net.gen.p_mw).values,
                "q_set": 0.0,
                "bus": net.bus.name.loc[net.gen.bus].values,
                "control": "PV",
            },
            index=net.gen.name,
        )

        # deal with PQ "static" generators
        _tmp_sgen = pd.DataFrame(
            {
                "p_set": (net.sgen.scaling * net.sgen.p_mw).values,
                "q_set": (net.sgen.scaling * net.sgen.q_mvar).values,
                "bus": net.bus.name.loc[net.sgen.bus].values,
                "control": "PQ",
            },
            index=net.sgen.name,
        )

        _tmp_ext_grid = pd.DataFrame(
            {
                "control": "Slack",
                "p_set": 0.0,
                "q_set": 0.0,
                "bus": net.bus.name.loc[net.ext_grid.bus].values,
            },
            index=net.ext_grid.name.fillna("External Grid"),
        )

        # concat all generators and index according to option
        d["Generator"] = pd.concat(
            [_tmp_gen, _tmp_sgen, _tmp_ext_grid], ignore_index=use_pandapower_index
        )

        if extra_line_data is False:
            d["Line"] = pd.DataFrame(
                {
                    "type": net.line.std_type.values,
                    "bus0": net.bus.name.loc[net.line.from_bus].values,
                    "bus1": net.bus.name.loc[net.line.to_bus].values,
                    "length": net.line.length_km.values,
                    "num_parallel": net.line.parallel.values,
                },
                index=net.line.name,
            )
        else:
            r = net.line.r_ohm_per_km.values * net.line.length_km.values
            x = net.line.x_ohm_per_km.values * net.line.length_km.values
            # capacitance values from pandapower in nF; transformed here:
            f = net.f_hz
            b = net.line.c_nf_per_km.values * net.line.length_km.values * 1e-9
            b = b * 2 * math.pi * f

            u = net.bus.vn_kv.loc[net.line.from_bus].values
            s_nom = u * net.line.max_i_ka.values

            d["Line"] = pd.DataFrame(
                {
                    "r": r,
                    "x": x,
                    "b": b,
                    "s_nom": s_nom,
                    "bus0": net.bus.name.loc[net.line.from_bus].values,
                    "bus1": net.bus.name.loc[net.line.to_bus].values,
                    "length": net.line.length_km.values,
                    "num_parallel": net.line.parallel.values,
                },
                index=net.line.name,
            )

        # check, if the trafo is based on a standard-type:
        if net.trafo.std_type.any():
            d["Transformer"] = pd.DataFrame(
                {
                    "type": net.trafo.std_type.values,
                    "bus0": net.bus.name.loc[net.trafo.hv_bus].values,
                    "bus1": net.bus.name.loc[net.trafo.lv_bus].values,
                    "tap_position": net.trafo.tap_pos.values,
                },
                index=net.trafo.name,
            )
        else:
            s_nom = net.trafo.sn_mva.values

            # documented at https://pandapower.readthedocs.io/en/develop/elements/trafo.html?highlight=transformer#impedance-values
            z = net.trafo.vk_percent.values / 100.0 / net.trafo.sn_mva.values
            r = net.trafo.vkr_percent.values / 100.0 / net.trafo.sn_mva.values
            x = np.sqrt(z**2 - r**2)

            y = net.trafo.i0_percent.values / 100.0
            g = (
                net.trafo.pfe_kw.values
                / net.trafo.sn_mva.values
                / 1000
                / net.trafo.sn_mva.values
            )
            b = np.sqrt(y**2 - g**2)

            d["Transformer"] = pd.DataFrame(
                {
                    "phase_shift": net.trafo.shift_degree.values,
                    "s_nom": s_nom,
                    "bus0": net.bus.name.loc[net.trafo.hv_bus].values,
                    "bus1": net.bus.name.loc[net.trafo.lv_bus].values,
                    "r": r,
                    "x": x,
                    "g": g,
                    "b": b,
                    "tap_position": net.trafo.tap_pos.values,
                },
                index=net.trafo.name,
            )
        d["Transformer"] = d["Transformer"].fillna(0)

        # documented at https://docs.pypsa.org/latest/user-guide/components/shunt-impedances
        g_shunt = net.shunt.p_mw.values / net.shunt.vn_kv.values**2
        b_shunt = -net.shunt.q_mvar.values / net.shunt.vn_kv.values**2

        d["ShuntImpedance"] = pd.DataFrame(
            {
                "bus": net.bus.name.loc[net.shunt.bus].values,
                "g": g_shunt,
                "b": b_shunt,
            },
            index=net.shunt.name,
        )
        d["ShuntImpedance"] = d["ShuntImpedance"].fillna(0)

        for component_name in [
            "Bus",
            "Load",
            "Generator",
            "Line",
            "Transformer",
            "ShuntImpedance",
        ]:
            self.add(component_name, d[component_name].index, **d[component_name])

        # amalgamate buses connected by closed switches

        bus_switches = net.switch[(net.switch.et == "b") & net.switch.closed]

        bus_switches["stays"] = bus_switches.bus.map(net.bus.name)
        bus_switches["goes"] = bus_switches.element.map(net.bus.name)

        to_replace = pd.Series(bus_switches.stays.values, bus_switches.goes.values)

        for i in to_replace.index:
            self.remove("Bus", i)

        for component in self.components[["Generator", "Load", "ShuntImpedance"]]:
            if component.empty:
                continue
            component.static.replace({"bus": to_replace}, inplace=True)

        for component in self.components[["Line", "Transformer"]]:
            if component.empty:
                continue
            component.static.replace({"bus0": to_replace}, inplace=True)
            component.static.replace({"bus1": to_replace}, inplace=True)

    @staticmethod
    def _require_datarecord() -> None:
        """Check the `datarecord` extra is installed and warn once it is experimental.

        `stacklevel=3` points the warning at the public `*_datarecord` call
        that invoked this helper, not at this helper itself.
        """
        check_optional_dependency(
            "datarecord",
            "Install with `pip install pypsa[datarecord]` (Python 3.12+).",
        )
        warnings.warn(
            "The datarecord format is experimental and its layout may change.",
            UserWarning,
            stacklevel=3,
        )

    def to_datarecord(self) -> Any:
        """Present this network as a datarecord `Record` (export only).

        <!-- md:badge-version v2.0.0 -->

        !!! warning "Experimental"
            The datarecord format is experimental and its layout may change.

        Requires the `datarecord` extra (Python 3.12+,
        `pip install pypsa[datarecord]`). Names must be unique across
        component types and snapshots must be integer- or datetime-typed.

        Returns
        -------
        pypsa.network.io.datarecord.record.NetworkRecord
            A lazy `Record` view over this network.

        Examples
        --------
        >>> n = pypsa.Network()  # doctest: +SKIP
        >>> record = n.to_datarecord()  # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.export_to_datarecord][], [pypsa.Network.from_datarecord][],
        [pypsa.Network.import_from_datarecord][]

        """
        self._require_datarecord()
        from pypsa.network.io.datarecord.record import NetworkRecord  # noqa: PLC0415

        return NetworkRecord(cast("Network", self))

    def export_to_datarecord(self, path: str | Path) -> None:
        """Export this network to the datarecord format.

        <!-- md:badge-version v2.0.0 -->

        !!! warning "Experimental"
            The datarecord format is experimental and its layout may change.

        Requires the `datarecord` extra (Python 3.12+,
        `pip install pypsa[datarecord]`). Names must be unique across
        component types and snapshots must be integer- or datetime-typed.

        Parameters
        ----------
        path : str | Path
            Directory to write the record to. Remote URIs work through
            datarecord's own connection.

        Examples
        --------
        >>> n = pypsa.Network()  # doctest: +SKIP
        >>> n.export_to_datarecord("network")  # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.import_from_datarecord][], [pypsa.Network.to_datarecord][],
        [pypsa.Network.from_datarecord][]

        """
        self._require_datarecord()
        from datarecord.duck import connect  # noqa: PLC0415
        from datarecord.layered.write import write_record  # noqa: PLC0415

        from pypsa.network.io.datarecord.record import NetworkRecord  # noqa: PLC0415

        record = NetworkRecord(cast("Network", self))
        con = connect()
        write_record(None, record, con, uri=str(path))

    @classmethod
    def from_datarecord(cls, record: RecordLike) -> Network:
        """Build a network from a datarecord `Record` (import only).

        <!-- md:badge-version v2.0.0 -->

        !!! warning "Experimental"
            The datarecord format is experimental and its layout may change.

        Requires the `datarecord` extra (Python 3.12+,
        `pip install pypsa[datarecord]`).

        Parameters
        ----------
        record : datarecord.Record
            A record built by `Network.to_datarecord` (or an equivalent one).

        Returns
        -------
        Network
            The network the record describes. A time-series frame's columns
            follow the component's static index order, which can differ from
            the order they had in the network the record was built from.

        Examples
        --------
        >>> record = n.to_datarecord()  # doctest: +SKIP
        >>> n2 = pypsa.Network.from_datarecord(record)  # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.to_datarecord][], [pypsa.Network.import_from_datarecord][],
        [pypsa.Network.export_to_datarecord][]

        """
        cls._require_datarecord()
        from pypsa.network.io.datarecord.build import (  # noqa: PLC0415
            network_from_record,
        )

        n = cast("Network", cls())
        network_from_record(record, n)
        return n

    def import_from_datarecord(self, path: str | Path) -> None:
        """Import a network from the datarecord format.

        <!-- md:badge-version v2.0.0 -->

        !!! warning "Experimental"
            The datarecord format is experimental and its layout may change.

        Requires the `datarecord` extra (Python 3.12+,
        `pip install pypsa[datarecord]`).

        Parameters
        ----------
        path : str | Path
            Directory to read the record from. Remote URIs work through
            datarecord's own connection.

        Notes
        -----
        A time-series frame's columns follow the component's static index
        order, which can differ from the order they had in the source network.

        Examples
        --------
        >>> n.import_from_datarecord("network")  # doctest: +SKIP

        See Also
        --------
        [pypsa.Network.export_to_datarecord][], [pypsa.Network.to_datarecord][],
        [pypsa.Network.from_datarecord][]

        """
        self._require_datarecord()
        from datarecord import Record, connect  # noqa: PLC0415

        from pypsa.network.io.datarecord.build import (  # noqa: PLC0415
            network_from_record,
        )

        record = Record.at(str(path), connect())
        network_from_record(record, cast("Network", self))
