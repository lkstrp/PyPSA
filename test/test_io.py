# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import sys

import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from numpy.testing import assert_array_almost_equal as equal

import pypsa

try:
    import tables  # noqa: F401

    tables_installed = True
except ImportError:
    tables_installed = False

try:
    import openpyxl  # noqa: F401
    import python_calamine  # noqa: F401

    excel_installed = True
except ImportError:
    excel_installed = False


# `pypsa.examples`' cached networks still carry a stray `now` scalar from an
# older PyPSA version. CSV/netCDF preserve it via `dir(n)` reflection but the
# datarecord network-attribute allow-list does not, by design.
_LEGACY_EXAMPLE_ATTRS = ["now"]


def custom_equals(n1, n2, ignore_attrs=None):
    """
    Custom equality check that allows certain attributes to be different.
    Parameters
    ----------
    n1, n2 : pypsa.Network
        Networks to compare
    ignore_attrs : list of str, optional
        List of attribute names that are allowed to be different.
    """
    if not ignore_attrs:
        return n1.equals(n2, log_mode="strict")

    # Copy networks to avoid modifying originals
    n1 = n1.copy()
    n2 = n2.copy()

    def _resolve(net, parts):
        obj = net
        for part in parts[:-1]:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    for attr in ignore_attrs:
        parts = attr.split(".")
        last = parts[-1]
        obj1, obj2 = _resolve(n1, parts), _resolve(n2, parts)
        for obj in (obj1, obj2):
            if obj is None:
                continue
            # A stray attribute one side carries and the other never set (e.g.
            # a legacy example network's ad hoc scalar) is dropped outright.
            # Nulling it would still leave the key missing from the other
            # side's `__dict__`, which `equals` compares key by key.
            if last in getattr(obj, "__dict__", {}):
                del obj.__dict__[last]
            elif hasattr(obj, last):
                setattr(obj, last, None)

    return n1.equals(n2, log_mode="strict")


# TODO classes could be further parametrized
class TestCSVDir:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_csv_io(self, scipy_network, tmpdir, meta, no_warnings):
        fn = tmpdir / "csv_export"
        scipy_network.meta = meta
        scipy_network.export_to_csv_folder(fn)
        pypsa.Network(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_csv_io_quotes(self, scipy_network, tmpdir, meta):
        fn = tmpdir / "csv_export"
        scipy_network.meta = meta
        scipy_network.export_to_csv_folder(fn, quotechar="'")
        imported = pypsa.Network()
        imported.import_from_csv_folder(fn, quotechar="'")
        assert imported.meta == scipy_network.meta

    def test_csv_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "csv_export"
        scipy_network.export_to_csv_folder(fn)
        pypsa.Network(fn)

    def test_csv_io_piecewise(self, tmpdir, piecewise_network):
        fn = tmpdir / "csv_piecewise"
        piecewise_network.export_to_csv_folder(fn)
        imported = pypsa.Network(fn)
        assert custom_equals(piecewise_network, imported)

    def test_csv_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "csv_export"
        ac_dc_periods.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
            check_index_type=False,
            check_column_type=False,
        )

    def test_csv_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "csv_export"
        ac_dc_shapes.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_csv_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "csv_export"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_csv_folder(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    def test_csv_io_deduplicates_clashing_shape_reference(self, tmp_path):
        from shapely.geometry import Point

        n = pypsa.Network()
        n.add("Bus", ["bus0", "bus1"])
        n.add("Line", "bus0", bus0="bus0", bus1="bus1")
        n.add("Shape", "shape1", geometry=Point(0, 0), component="Line", idx="bus0")
        fn = tmp_path / "csv_export"
        n.export_to_csv_folder(fn)

        m = pypsa.Network(fn)

        assert "bus0-Line" in m.c.lines.static.index
        assert m.c.shapes.static.loc["shape1", "idx"] == "bus0-Line"

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using CSV format.
        """
        n = networks_including_solved
        if n.has_scenarios:
            with pytest.raises(
                NotImplementedError,
                match="Stochastic networks are not supported*",
            ):
                n.export_to_csv_folder(tmp_path / "network")
            return
        n.export_to_csv_folder(tmp_path / "network")
        n3 = pypsa.Network(tmp_path / "network")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n3, ignore_attrs=ignore)


class TestNetcdf:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_netcdf_io(self, scipy_network, tmpdir, meta, no_warnings):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.meta = meta
        scipy_network.export_to_netcdf(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    def test_netcdf_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn)
        pypsa.Network(fn)

    def test_netcdf_io_piecewise(self, tmpdir, piecewise_network):
        ds = piecewise_network.export_to_netcdf(path=None)
        imported = pypsa.Network()
        imported.import_from_netcdf(ds)
        assert custom_equals(piecewise_network, imported)

    def test_netcdf_io_datetime(self, tmpdir):
        fn = tmpdir / "temp.nc"
        exported_sns = pd.date_range(start="2013-03-01", end="2013-03-02", freq="h")
        n = pypsa.Network()
        n.set_snapshots(exported_sns)
        n.export_to_netcdf(fn)
        imported_sns = pypsa.Network(fn).snapshots

        assert (imported_sns == exported_sns).all()

    def test_netcdf_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        ac_dc_periods.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
            check_index_type=False,
            check_column_type=False,
        )
        pd.testing.assert_frame_equal(
            m.snapshot_weightings,
            ac_dc_periods.snapshot_weightings[
                m.snapshot_weightings.columns
            ],  # reset order
        )

    def test_netcdf_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        ac_dc_shapes.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_netcdf_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_netcdf(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    def test_netcdf_io_shapes_memory(self, tmpdir):
        import tracemalloc

        import numpy as np
        from shapely.geometry import Polygon

        angles = np.linspace(0, 2 * np.pi, 200000, endpoint=False)
        coords = np.round(np.c_[100 * np.cos(angles), 100 * np.sin(angles)], 6)
        detailed = Polygon(coords)
        geometry = [Polygon([(0, 0), (1, 0), (1, 1)])] * 150 + [detailed]
        names = [f"s{i}" for i in range(len(geometry))]

        n = pypsa.Network()
        n.add("Bus", names)
        n.add("Shape", names, geometry=geometry, component="Bus", idx=names)
        fn = tmpdir / "netcdf_export.nc"
        n.export_to_netcdf(fn)

        tracemalloc.start()
        m = pypsa.Network(fn)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 500e6
        assert_geodataframe_equal(
            m.c.shapes.static, n.c.shapes.static, check_less_precise=True
        )

    def test_netcdf_io_deduplicates_clashing_names(self, tmp_path):
        n = pypsa.Network()
        n.add("Bus", ["x", "x-Load"])
        n.add("Load", "x", bus="x")
        fn = tmp_path / "netcdf_export.nc"
        n.export_to_netcdf(fn)

        m = pypsa.Network(fn)

        assert set(m.c.buses.static.index) == {"x", "x-Load"}
        assert "x-Load-2" in m.c.loads.static.index

    def test_netcdf_from_url(self):
        url = "https://data.pypsa.org/networks/examples/latest/scigrid_de.nc"
        pypsa.Network(url)

    def test_netcdf_io_no_compression(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn, float32=False, compression=None)
        scipy_network_compressed = pypsa.Network(fn)
        assert (
            (
                scipy_network.c.loads.dynamic.p_set
                == scipy_network_compressed.c.loads.dynamic.p_set
            )
            .all()
            .all()
        )

    def test_netcdf_io_custom_compression(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        digits = 5
        compression = {"zlib": True, "complevel": 9, "least_significant_digit": digits}
        scipy_network.export_to_netcdf(fn, compression=compression)
        scipy_network_compressed = pypsa.Network(fn)
        assert (
            (
                (
                    scipy_network.c.loads.dynamic.p_set
                    - scipy_network_compressed.c.loads.dynamic.p_set
                ).abs()
                < 1 / 10**digits
            )
            .all()
            .all()
        )

    def test_netcdf_io_typecast(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn, float32=True, compression=None)
        pypsa.Network(fn)

    def test_netcdf_io_typecast_and_compression(self, scipy_network, tmpdir):
        fn = tmpdir / "netcdf_export.nc"
        scipy_network.export_to_netcdf(fn, float32=True)
        pypsa.Network(fn)

    @pytest.mark.parametrize("legacy", [True, False])
    def test_netcdf_io_string_dtypes(self, tmpdir, legacy):
        """String data follows `api.legacy_string_dtype` (#1585)."""
        fn = tmpdir / "string_dtypes.nc"

        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2025-01-01", periods=4, freq="h"))
        n.add("Bus", "bus0")
        n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=10)
        n.add("Load", "load0", bus="bus0", p_set=50)
        n.export_to_netcdf(fn)

        with pypsa.option_context("api.legacy_string_dtype", legacy):
            m = pypsa.Network(fn)

        for c in m.components:
            df = c.static
            assert pd.api.types.is_object_dtype(df.index.dtype) == legacy, (
                f"{c.name}.static.index is {df.index.dtype}"
            )
            for col, dtype in df.dtypes.items():
                if c.defaults.at[col, "type"] == "string":
                    assert pd.api.types.is_object_dtype(dtype) == legacy, (
                        f"{c.name}.static[{col!r}] is {dtype}"
                    )

    def test_string_dtypes_reach_xarray_as_object(self):
        """Canary: drop `_strings_to_object` once xarray accepts `StringDtype`.

        xarray rejects string extension arrays (`TypeError: Invalid array type`), so
        `_as_xarray` casts them to object. When a future xarray fixes this, the
        `pytest.raises` fails. See https://github.com/pydata/xarray/issues/10301.
        """
        import xarray as xr  # noqa: PLC0415

        import pypsa.examples  # noqa: PLC0415

        with pypsa.option_context("api.legacy_string_dtype", False):
            n = pypsa.examples.ac_dc_meshed()

        bus = n.c.generators.static.bus
        assert isinstance(bus.dtype, pd.StringDtype)

        with pytest.raises(TypeError, match="Invalid array type"):
            xr.DataArray(bus).sel(name=bus.index[:1])

        assert n.c.generators._as_xarray("bus").dtype == object

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using netCDF format.
        """
        n = networks_including_solved
        n.export_to_netcdf(tmp_path / "network.nc")
        n2 = pypsa.Network(tmp_path / "network.nc")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n2, ignore_attrs=ignore)

    def test_779(self):
        """
        Importing from xarray dataset.
        See https://github.com/PyPSA/PyPSA/issues/779.
        """
        n1 = pypsa.Network()
        n1.add("Bus", "bus")
        xarr = n1.export_to_netcdf()
        n2 = pypsa.Network()
        n2.import_from_netcdf(xarr)

    def test_1522(self, tmp_path):
        """
        NetCDF export corrupts dynamic attributes with direct DataFrame assignment.
        See https://github.com/PyPSA/PyPSA/issues/1522.
        """
        fn = tmp_path / "test.nc"

        n = pypsa.Network()
        n.set_snapshots(range(3))
        n.add("Bus", ["bus0", "bus1"])
        n.add("Generator", ["gen0", "gen1"], bus=["bus0", "bus1"], p_nom=100)
        n.add("Link", ["link0", "link1"], bus0=["bus0", "bus1"], bus1=["bus1", "bus0"])

        # Direct assignment without proper column name
        n.generators_t.marginal_cost = pd.DataFrame(
            {"gen0": [10.0, 20.0, 30.0], "gen1": [15.0, 25.0, 35.0]},
            index=n.snapshots,
        )
        n.links_t.marginal_cost = pd.DataFrame(
            {"link0": [1.0, 2.0, 3.0], "link1": [0.5, 1.5, 2.5]},
            index=n.snapshots,
        )

        n.export_to_netcdf(fn)
        m = pypsa.Network(fn)

        assert set(m.generators_t.marginal_cost.columns) == {"gen0", "gen1"}
        assert set(m.links_t.marginal_cost.columns) == {"link0", "link1"}


@pytest.mark.skipif(not tables_installed, reason="PyTables not installed")
class TestHDF5:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_hdf5_io(self, scipy_network, tmpdir, meta, no_warnings):
        fn = tmpdir / "hdf5_export.h5"
        scipy_network.meta = meta
        scipy_network.export_to_hdf5(fn)
        pypsa.Network(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    def test_hdf5_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        scipy_network.export_to_hdf5(fn)
        pypsa.Network(fn)

    def test_hdf5_io_piecewise(self, tmpdir, piecewise_network):
        fn = tmpdir / "hdf5_piecewise.h5"
        piecewise_network.export_to_hdf5(fn)
        imported = pypsa.Network(fn)
        assert custom_equals(piecewise_network, imported)

    def test_hdf5_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        ac_dc_periods.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
            check_index_type=False,
            check_column_type=False,
        )

    def test_hdf5_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        ac_dc_shapes.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_hdf5_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "hdf5_export.h5"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_hdf5(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using HDF5 format.
        """
        n = networks_including_solved
        if n.has_scenarios:
            with pytest.raises(
                NotImplementedError,
                match="Stochastic networks are not supported*",
            ):
                n.export_to_hdf5(tmp_path / "network.h5")
            return
        n.export_to_hdf5(tmp_path / "network.h5")
        n5 = pypsa.Network(tmp_path / "network.h5")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n5, ignore_attrs=ignore)


@pytest.mark.skipif(not excel_installed, reason="openpyxl not installed")
class TestExcelIO:
    @pytest.mark.parametrize(
        "meta",
        [
            {"test": "test"},
            {"test": "test", "test2": "test2"},
            {"test": {"test": "test", "test2": "test2"}},
        ],
    )
    def test_excel_io(self, scipy_network, tmpdir, meta, no_warnings):
        fn = tmpdir / "excel_export.xlsx"
        scipy_network.meta = meta
        scipy_network.export_to_excel(fn)
        reloaded = pypsa.Network(fn)
        assert reloaded.meta == scipy_network.meta

    def test_excel_io_Path(self, scipy_network, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        scipy_network.export_to_excel(fn)
        pypsa.Network(fn)

    def test_excel_io_piecewise(self, tmpdir, piecewise_network):
        fn = tmpdir / "excel_piecewise.xlsx"
        piecewise_network.export_to_excel(fn)
        imported = pypsa.Network(fn)
        assert custom_equals(piecewise_network, imported)

    def test_excel_io_datetime(self, tmpdir):
        fn = tmpdir / "temp.xlsx"
        exported_sns = pd.date_range(start="2013-03-01", end="2013-03-02", freq="h")
        n = pypsa.Network()
        n.set_snapshots(exported_sns)
        n.export_to_excel(fn)
        imported_sns = pypsa.Network(fn).snapshots
        assert (imported_sns == exported_sns).all()

    def test_excel_io_multiindexed(self, ac_dc_periods, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        ac_dc_periods.export_to_excel(fn)
        m = pypsa.Network(fn)
        pd.testing.assert_frame_equal(
            m.c.generators.dynamic.p,
            ac_dc_periods.c.generators.dynamic.p,
            check_index_type=False,
            check_column_type=False,
        )
        pd.testing.assert_frame_equal(
            m.snapshot_weightings,
            ac_dc_periods.snapshot_weightings[m.snapshot_weightings.columns],
            check_dtype=False,  # TODO Remove once validation layer leads to safer types
            check_index_type=False,
        )

    def test_excel_io_shapes(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        ac_dc_shapes.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            ac_dc_shapes.c.shapes.static,
            check_less_precise=True,
        )

    def test_excel_io_shapes_with_missing(self, ac_dc_shapes, tmpdir):
        fn = tmpdir / "excel_export.xlsx"
        n = ac_dc_shapes.copy()
        n.c.shapes.static.loc["Manchester", "geometry"] = None
        n.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert_geodataframe_equal(
            m.c.shapes.static,
            n.c.shapes.static,
            check_less_precise=True,
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
        reason="Unstable test in CI. Remove with 1.0",
    )
    def test_io_equality(self, networks_including_solved, tmp_path):
        """
        Test if the network is equal after export and import using Excel format.
        """
        n = networks_including_solved
        if n.has_scenarios:
            with pytest.raises(
                NotImplementedError,
                match="Stochastic networks are not supported*",
            ):
                n.export_to_excel(tmp_path / "network.xlsx")
            return
        n.export_to_excel(tmp_path / "network.xlsx")
        n4 = pypsa.Network(tmp_path / "network.xlsx")
        # Allow difference for solved networks
        # TODO: Remove _components.links with #1128
        ignore = (
            [
                "_components.sub_networks.static.obj",
                "_components.links",
                "_components.lines",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n4, ignore_attrs=ignore)

    def test_io_time_dependent_efficiencies_excel(self, tmpdir):
        n = pypsa.Network()
        s = [1, 0.95, 0.99]
        n.snapshots = range(len(s))
        n.add("Bus", "bus")
        n.add("Generator", "gen", bus="bus", efficiency=s)
        n.add("Store", "sto", bus="bus", standing_loss=s)
        n.add(
            "StorageUnit",
            "su",
            bus="bus",
            efficiency_store=s,
            efficiency_dispatch=s,
            standing_loss=s,
        )
        fn = tmpdir / "network-time-eff.xlsx"
        n.export_to_excel(fn)
        m = pypsa.Network(fn)
        assert not m.c.stores.dynamic.standing_loss.empty
        assert not m.c.storage_units.dynamic.standing_loss.empty
        assert not m.c.generators.dynamic.efficiency.empty
        assert not m.c.storage_units.dynamic.efficiency_store.empty
        assert not m.c.storage_units.dynamic.efficiency_dispatch.empty
        equal(m.c.stores.dynamic.standing_loss, n.c.stores.dynamic.standing_loss)
        equal(
            m.c.storage_units.dynamic.standing_loss,
            n.c.storage_units.dynamic.standing_loss,
        )
        equal(m.c.generators.dynamic.efficiency, n.c.generators.dynamic.efficiency)
        equal(
            m.c.storage_units.dynamic.efficiency_store,
            n.c.storage_units.dynamic.efficiency_store,
        )
        equal(
            m.c.storage_units.dynamic.efficiency_dispatch,
            n.c.storage_units.dynamic.efficiency_dispatch,
        )

    def test_1268(self, tmpdir):
        """
        Excel import without snapshots sheet should not raise KeyError.
        See https://github.com/PyPSA/PyPSA/issues/1268.
        """
        fn = str(tmpdir / "no_snapshots.xlsx")

        buses = pd.DataFrame({"v_nom": [132]}, index=["bus1"])
        with pd.ExcelWriter(fn, engine="openpyxl") as writer:
            buses.to_excel(writer, sheet_name="buses")

        n = pypsa.Network()
        n.import_from_excel(fn)
        assert len(n.c.buses.static) == 1


@pytest.mark.skipif(
    sys.version_info < (3, 13) or sys.platform not in ["linux", "darwin"],
    reason="Unstable test in CI. Remove with 1.0",
)
def test_io_equality(networks_including_solved, tmp_path):
    """
    Test if the network is equal after export and import.
    """
    n = networks_including_solved
    n.export_to_netcdf(tmp_path / "network.nc")
    n2 = pypsa.Network(tmp_path / "network.nc")
    # Allow difference for solved networks
    # TODO: Remove _components.links with #1128
    ignore = (
        [
            "_components.sub_networks.static.obj",
            "_components.links",
            "_components.lines",
        ]
        if n.model is not None
        else []
    )
    assert custom_equals(n, n2, ignore_attrs=ignore)

    # Only check with supported io formats
    if n.has_scenarios:
        return

    n.export_to_csv_folder(tmp_path / "network")
    n3 = pypsa.Network(tmp_path / "network")
    assert custom_equals(n, n3, ignore_attrs=ignore)

    if excel_installed:
        n.export_to_excel(tmp_path / "network.xlsx")
        n4 = pypsa.Network(tmp_path / "network.xlsx")
        assert custom_equals(n, n4, ignore_attrs=ignore)

    if tables_installed:
        n.export_to_hdf5(tmp_path / "network.h5")
        n5 = pypsa.Network(tmp_path / "network.h5")
        assert custom_equals(n, n5, ignore_attrs=ignore)


@pytest.mark.parametrize(
    "example_network",
    [
        "ac_dc_meshed",
        "storage_hvdc",
        "scigrid_de",
        "model_energy",
    ],
)
def test_examples_consistency(tmp_path, example_network):
    # Round-trip example networks through CSV export/import to catch schema drift
    n = getattr(pypsa.examples, example_network)()
    n.export_to_csv_folder(tmp_path / "network")
    n2 = pypsa.Network(tmp_path / "network")
    assert n.equals(n2, log_mode="strict")


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Test requires Python 3.12 or higher"
)
@pytest.mark.parametrize("use_pandapower_index", [True, False])
@pytest.mark.parametrize("extra_line_data", [True, False])
def test_import_from_pandapower_network(
    pandapower_custom_network,
    pandapower_cigre_network,
    extra_line_data,
    use_pandapower_index,
):
    nets = [pandapower_custom_network, pandapower_cigre_network]
    for net in nets:
        n = pypsa.Network()
        n.import_from_pandapower_net(
            net,
            use_pandapower_index=use_pandapower_index,
            extra_line_data=extra_line_data,
        )
        assert len(n.c.buses.static) == len(net.bus)
        assert len(n.c.generators.static) == (
            len(net.gen) + len(net.sgen) + len(net.ext_grid)
        )
        assert len(n.loads) == len(net.load)
        assert len(n.c.transformers.static) == len(net.trafo)
        assert len(n.c.shunt_impedances.static) == len(net.shunt)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Test requires Python 3.12 or higher"
)
def test_import_from_pandapower_network_deduplicates_clashing_names(caplog):
    pp = pytest.importorskip("pandapower", reason="pandapower not installed")

    net = pp.create_empty_network()
    bus0 = pp.create_bus(net, vn_kv=20.0, name="clash")
    bus1 = pp.create_bus(net, vn_kv=0.4, name="other")
    pp.create_ext_grid(net, bus=bus0, vm_pu=1.02)
    pp.create_load(net, bus=bus1, p_mw=0.1, q_mvar=0.05, name="clash")
    pp.create_line(
        net,
        from_bus=bus0,
        to_bus=bus1,
        length_km=0.1,
        std_type="NAYY 4x50 SE",
        name="line",
    )

    n = pypsa.Network()
    with caplog.at_level("WARNING"):
        n.import_from_pandapower_net(net)

    assert len(n.c.buses.static.index) == len(set(n.c.buses.static.index)) == 2
    assert len(n.c.loads.static.index) == len(set(n.c.loads.static.index)) == 1
    assert "clash" in n.c.buses.static.index
    assert "clash" not in n.c.loads.static.index
    assert "clash-Load" in n.c.loads.static.index
    assert any(
        "Renamed 1 component names" in record.message for record in caplog.records
    )


def test_io_time_dependent_efficiencies(tmpdir):
    n = pypsa.Network()
    s = [1, 0.95, 0.99]
    n.snapshots = range(len(s))
    n.add("Bus", "bus")
    n.add("Generator", "gen", bus="bus", efficiency=s)
    n.add("Store", "sto", bus="bus", standing_loss=s)
    n.add(
        "StorageUnit",
        "su",
        bus="bus",
        efficiency_store=s,
        efficiency_dispatch=s,
        standing_loss=s,
    )

    fn = tmpdir / "network-time-eff.nc"
    n.export_to_netcdf(fn)
    m = pypsa.Network(fn)

    assert not m.c.stores.dynamic.standing_loss.empty
    assert not m.c.storage_units.dynamic.standing_loss.empty
    assert not m.c.generators.dynamic.efficiency.empty
    assert not m.c.storage_units.dynamic.efficiency_store.empty
    assert not m.c.storage_units.dynamic.efficiency_dispatch.empty

    equal(m.c.stores.dynamic.standing_loss, n.c.stores.dynamic.standing_loss)
    equal(
        m.c.storage_units.dynamic.standing_loss, n.c.storage_units.dynamic.standing_loss
    )
    equal(m.c.generators.dynamic.efficiency, n.c.generators.dynamic.efficiency)
    equal(
        m.c.storage_units.dynamic.efficiency_store,
        n.c.storage_units.dynamic.efficiency_store,
    )
    equal(
        m.c.storage_units.dynamic.efficiency_dispatch,
        n.c.storage_units.dynamic.efficiency_dispatch,
    )


def test_sort_attrs():
    """Ensure _sort_attrs preserves attribute order semantics."""
    from pypsa.network.io import _sort_attrs

    axis_labels = pd.Index(["c", "a", "b", "d"])
    attrs_list = ["a", "b", "c"]
    ordered = _sort_attrs(axis_labels, attrs_list)
    assert list(ordered) == ["a", "b", "c", "d"]

    # Ignore attributes that are not present on the axis
    attrs_list = ["a", "x", "b", "y"]
    ordered = _sort_attrs(axis_labels, attrs_list)
    assert list(ordered) == ["a", "b", "c", "d"]

    # Missing attrs_list should leave order untouched
    ordered = _sort_attrs(axis_labels, [])
    assert ordered.equals(axis_labels)

    # Empty axis behaves like a no-op
    empty_axis = pd.Index([])
    ordered = _sort_attrs(empty_axis, ["a", "b"])
    assert ordered.equals(empty_axis)

    # Works with non-unique Index types (e.g. MultiIndex)
    axis_labels = pd.MultiIndex.from_product([["a", "b"], ["x", "y"]])
    attrs_list = pd.MultiIndex.from_product([["b", "a"], ["y"]])
    ordered = _sort_attrs(axis_labels, attrs_list)
    assert list(ordered) == [
        ("b", "y"),
        ("a", "y"),
        ("a", "x"),
        ("b", "x"),
    ]


def test_version_warning(caplog):
    # Assert no info logged with "version"
    n = pypsa.examples.ac_dc_meshed()
    assert "Importing network from PyPSA version" not in caplog.text

    n._pypsa_version = "0.10.0"
    n.export_to_netcdf("test.nc")
    pypsa.Network("test.nc")
    assert "Importing network from PyPSA version v0.10.0" in caplog.text


def _canonical_dynamic_order(n: pypsa.Network) -> pypsa.Network:
    """Reindex every component's dynamic frame columns to its static index order, in place.

    Dynamic column order carries no PyPSA semantics, but the datarecord
    import recovers it from the static index order, so a network is
    normalised this way before comparing it against a re-imported twin.
    """
    for c in n.components:
        order = c.static.index
        for attr, series in c.dynamic.items():
            if series.empty:
                continue
            c.dynamic[attr] = series.reindex(columns=order[order.isin(series.columns)])
    return n


def _record_twin(n: pypsa.Network) -> pypsa.Network:
    """A copy of `n` with every cross-type name collision renamed away.

    A record scopes names across every component type, while PyPSA scopes
    them per type, so a name two types share is renamed on every claiming
    type, `<Type> <name>`, before the network can be exported. `rename` can
    also turn a plain-index dtype (e.g. object) into pandas' `str` dtype,
    which is restored here too.
    """
    n = n.copy()
    dtypes = {
        c.name: c.static.index.dtype
        for c in n.components
        if not c.static.empty and not isinstance(c.static.index, pd.MultiIndex)
    }
    owners: dict[str, list[str]] = {}
    for c in n.components:
        if c.static.empty:
            continue
        index = c.static.index
        names = (
            index.get_level_values("name")
            if isinstance(index, pd.MultiIndex)
            else index
        )
        for name in names.unique():
            owners.setdefault(str(name), []).append(c.name)
    for name, ctypes in owners.items():
        if len(ctypes) < 2:
            continue
        for ctype in ctypes:
            n.rename_component_names(ctype, **{name: f"{ctype} {name}"})

    for c in n.components:
        if c.name in dtypes and c.static.index.dtype != dtypes[c.name]:
            c.static.index = c.static.index.astype(dtypes[c.name])
    return _canonical_dynamic_order(n)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="datarecord requires Python 3.12+"
)
class TestDatarecord:
    """Export to the datarecord format: errors, warnings and the schema shape."""

    @pytest.fixture(autouse=True)
    def _require_datarecord(self):
        pytest.importorskip("datarecord", reason="datarecord not installed")

    def test_collision_raises(self):
        from pypsa.network.io.datarecord.record import DatarecordExportError

        n = pypsa.Network()
        n.add("Bus", "b0")
        n.add("Load", "b0", bus="b0")

        with pytest.warns(UserWarning, match="experimental"):
            with pytest.raises(DatarecordExportError, match="Bus, Load"):
                n.to_datarecord()

    def test_same_bus_twice_raises(self):
        from pypsa.network.io.datarecord.record import DatarecordExportError

        n = pypsa.Network()
        n.add("Bus", "b0")
        n.add("Link", "l0", bus0="b0", bus1="b0")
        with pytest.warns(UserWarning, match="experimental"):
            with pytest.raises(DatarecordExportError, match="b0"):
                n.to_datarecord()

    def test_experimental_warning_once_per_call(self, ac_dc_network):
        n = _record_twin(ac_dc_network)
        with pytest.warns(UserWarning, match="experimental") as record:
            n.to_datarecord()
        matches = [w for w in record if "experimental" in str(w.message)]
        assert len(matches) == 1

    def test_export_opens_as_directory_record(self, ac_dc_network, tmp_path):
        from datarecord import Record, connect

        from pypsa.network.io.datarecord.schema import ENTITY_TYPE

        n = _record_twin(ac_dc_network)
        path = tmp_path / "record"
        with pytest.warns(UserWarning, match="experimental"):
            n.export_to_datarecord(path)
        assert (path / "manifest.json").exists()
        opened = Record.at(str(path), connect())
        assert set(opened.schema.types) == set(
            opened.schema.dimensions[ENTITY_TYPE].dtype.categories
        )

    def test_outputs_and_per_port_attributes_are_written(self, ac_dc_solved):
        # Non-default efficiency, so the scalar rows are not dropped as defaults.
        n = _record_twin(ac_dc_solved)
        n.c.links.static["efficiency"] = 0.9
        with pytest.warns(UserWarning, match="experimental"):
            rec = n.to_datarecord()

        p = rec.outputs["p"].to_native()
        gen = n.c.generators.static.index[0]
        assert not p[p["entity"] == gen].empty

        link = n.c.links.static.index[0]
        link_rows = p[p["entity"] == link]
        assert not link_rows.empty
        expected_buses = {
            n.c.links.static.at[link, "bus0"],
            n.c.links.static.at[link, "bus1"],
        }
        assert set(link_rows["bus"]) == expected_buses

        efficiency = rec.attributes["efficiency"].to_native()
        assert not efficiency.empty

        from pypsa.network.io.datarecord.schema import TIMESTEP

        assert rec.flags("Link")["p"].varies == frozenset({TIMESTEP})

    def test_multiperiod_series_carry_period_column(self):
        n = pypsa.examples.ac_dc_meshed()
        n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots])
        n.investment_periods = [2020, 2030]
        n = _record_twin(n)
        with pytest.warns(UserWarning, match="experimental"):
            rec = n.to_datarecord()

        p_max_pu = rec.attributes["p_max_pu"].to_native()
        assert "period" in p_max_pu.columns
        assert set(p_max_pu["period"].dropna().unique()) == {2020, 2030}

    def test_multiperiod_network_exports_to_disk_and_reopens(self, tmp_path):
        from datarecord import Record, connect

        n = pypsa.examples.ac_dc_meshed()
        n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots])
        n.investment_periods = [2020, 2030]
        n = _record_twin(n)

        path = tmp_path / "record"
        with pytest.warns(UserWarning, match="experimental"):
            n.export_to_datarecord(path)
        assert (path / "manifest.json").exists()

        opened = Record.at(str(path), connect())
        assert "period" in opened.dims["timestep"].to_native().columns

    def test_piecewise_breakpoints_are_written(self, piecewise_network):
        n = _record_twin(piecewise_network)
        with pytest.warns(UserWarning, match="experimental"):
            rec = n.to_datarecord()

        marginal_cost = rec.attributes["marginal_cost"].to_native()
        assert marginal_cost["breakpoint"].notna().any()

        efficiency = rec.attributes["efficiency"].to_native()
        assert efficiency["breakpoint"].notna().any()

    def test_stochastic_static_and_series_do_not_overlap(self, stochastic_network):
        n = _record_twin(stochastic_network)
        gen = n.c.generators.static.index.get_level_values("name").unique()[0]
        scenarios = n.c.generators.static.index.get_level_values("scenario").unique()
        cols = pd.MultiIndex.from_product(
            [scenarios, [gen]], names=["scenario", "name"]
        )
        n.c.generators.dynamic["marginal_cost"] = pd.DataFrame(
            1.0, index=n.snapshots, columns=cols
        )
        with pytest.warns(UserWarning, match="experimental"):
            rec = n.to_datarecord()

        marginal_cost = rec.attributes["marginal_cost"].to_native()
        entity_rows = marginal_cost[marginal_cost["entity"] == gen]
        assert not entity_rows.empty
        assert not entity_rows["timestep"].isna().any()

    def test_entity_types_omit_varying_and_deleted_columns(self, ac_dc_network):
        n = _record_twin(ac_dc_network)
        with pytest.warns(UserWarning, match="experimental"):
            rec = n.to_datarecord()

        generators = rec.entity_types["Generator"].to_native()
        assert "marginal_cost" not in generators.columns
        assert "p_max_pu" not in generators.columns
        assert "deleted" not in generators.columns

    def test_piecewise_on_non_varying_attribute_is_written(self):
        # capital_cost is not a `varying` attribute, but a piecewise curve on
        # it must still land in a long file, not silently as a scalar.
        n = pypsa.Network()
        n.add("Bus", "b0")
        n.add(
            "StorageUnit", "s1", bus="b0", capital_cost={0.0: 0.0, 10: 10.0, 20: 15.0}
        )
        with pytest.warns(UserWarning, match="experimental"):
            rec = n.to_datarecord()

        assert "capital_cost" in rec.attributes
        rows = rec.attributes["capital_cost"].to_native()
        assert rows["breakpoint"].notna().any()
        assert "capital_cost" not in rec.entity_types["StorageUnit"].to_native().columns

    # -- import: round-trip parity ---------------------------------------

    def _round_trip(self, n, tmp_path):
        """Export `n`'s renamed twin to disk and reopen it via `pypsa.Network`."""
        n = _record_twin(n)
        path = tmp_path / "record"
        with pytest.warns(UserWarning, match="experimental"):
            n.export_to_datarecord(path)
        with pytest.warns(UserWarning, match="experimental"):
            n2 = pypsa.Network(path)
        return n, n2

    def test_round_trip_networks_including_solved(
        self, networks_including_solved, tmp_path
    ):
        n = networks_including_solved
        n, n2 = self._round_trip(n, tmp_path)
        # Derived data (`SubNetwork`) and optimizer-internal scalars are out of
        # scope for the datarecord network-attribute allow-list.
        ignore = _LEGACY_EXAMPLE_ATTRS + (
            [
                "_components.sub_networks",
                "_linearized_uc",
                "_committable_big_m",
            ]
            if n.model is not None
            else []
        )
        assert custom_equals(n, n2, ignore_attrs=ignore)

    def test_round_trip_ac_dc_periods(self, ac_dc_periods, tmp_path):
        n, n2 = self._round_trip(ac_dc_periods, tmp_path)
        assert custom_equals(n, n2, ignore_attrs=_LEGACY_EXAMPLE_ATTRS)

    def test_round_trip_two_period(self, tmp_path):
        n = pypsa.examples.ac_dc_meshed()
        n.snapshots = pd.MultiIndex.from_product([[2020, 2030], n.snapshots])
        n.investment_periods = [2020, 2030]
        n, n2 = self._round_trip(n, tmp_path)
        assert custom_equals(n, n2, ignore_attrs=_LEGACY_EXAMPLE_ATTRS)

    def test_round_trip_ac_dc_shapes(self, ac_dc_shapes, tmp_path):
        n, n2 = self._round_trip(ac_dc_shapes, tmp_path)
        assert n.crs == n2.crs
        # WKT round-trips at limited precision (also true of CSV/Excel/HDF5).
        assert_geodataframe_equal(
            n.c.shapes.static,
            n2.c.shapes.static,
            check_less_precise=True,
        )
        n.c.shapes.static["geometry"] = n2.c.shapes.static["geometry"]
        assert custom_equals(n, n2, ignore_attrs=_LEGACY_EXAMPLE_ATTRS)

    def test_round_trip_piecewise_network(self, piecewise_network, tmp_path):
        n, n2 = self._round_trip(piecewise_network, tmp_path)
        assert custom_equals(n, n2)

    def test_round_trip_stochastic(self, stochastic_network, tmp_path):
        n, n2 = self._round_trip(stochastic_network, tmp_path)
        assert (n.scenario_weightings == n2.scenario_weightings).all().all()
        assert custom_equals(n, n2)

    def test_from_datarecord_without_disk(self, ac_dc_network):
        n = _record_twin(ac_dc_network)
        with pytest.warns(UserWarning, match="experimental"):
            record = n.to_datarecord()
        with pytest.warns(UserWarning, match="experimental"):
            n2 = pypsa.Network.from_datarecord(record)
        assert custom_equals(n, n2, ignore_attrs=_LEGACY_EXAMPLE_ATTRS)

    def test_network_constructor_dispatches_datarecord_directory(
        self, ac_dc_network, tmp_path
    ):
        n = _record_twin(ac_dc_network)
        path = tmp_path / "record"
        with pytest.warns(UserWarning, match="experimental"):
            n.export_to_datarecord(path)
        with pytest.warns(UserWarning, match="experimental"):
            n2 = pypsa.Network(path)
        assert custom_equals(n, n2, ignore_attrs=_LEGACY_EXAMPLE_ATTRS)

    def test_csv_folder_without_manifest_still_imports_as_csv(
        self, ac_dc_network, tmp_path
    ):
        path = tmp_path / "csv"
        ac_dc_network.export_to_csv_folder(path)
        assert not (path / "manifest.json").exists()
        n2 = pypsa.Network(path)
        assert n2.name == ac_dc_network.name

    def test_meta_and_name_survive(self, tmp_path):
        n = pypsa.Network()
        n.name = "custom-name"
        n.meta = {"foo": "bar"}
        n.add("Bus", "b0")
        n, n2 = self._round_trip(n, tmp_path)
        assert n2.name == "custom-name"
        assert n2.meta == {"foo": "bar"}

    def test_custom_attribute_does_not_survive(self, tmp_path):
        n = pypsa.Network()
        n.add("Bus", "b0")
        n.my_custom_attr = "some-value"
        _n, n2 = self._round_trip(n, tmp_path)
        assert not hasattr(n2, "my_custom_attr")

    def test_round_trip_ac_dc_meshed_example(self, tmp_path):
        """The shipped example, renamed but not collapsed to `_record_twin`'s
        per-scenario handling, round-trips strictly once its own dynamic
        column order is normalised to the datarecord import's contract.

        Solving triggers `n.determine_network_topology()`, so `SubNetwork`
        (out of scope for the datarecord allow-list, like the legacy example's
        stray `now` attribute) is ignored the same way
        `test_round_trip_networks_including_solved` already does for a
        solved network.
        """
        n = pypsa.examples.ac_dc_meshed()
        n.optimize(solver_name="highs", log_to_console=False)
        n.model.solver_model = None
        for ctype in ("Line", "Link", "Generator", "Load"):
            c = n.components[ctype]
            n.rename_component_names(
                ctype, **{name: f"{ctype} {name}" for name in c.static.index}
            )
        _canonical_dynamic_order(n)
        ignore = [
            *_LEGACY_EXAMPLE_ATTRS,
            "_components.sub_networks",
            "_linearized_uc",
            "_committable_big_m",
        ]

        path = tmp_path / "record"
        with pytest.warns(UserWarning, match="experimental"):
            n.export_to_datarecord(path)
        with pytest.warns(UserWarning, match="experimental"):
            n2 = pypsa.Network(path)
        assert custom_equals(n, n2, ignore_attrs=ignore)

        with pytest.warns(UserWarning, match="experimental"):
            record = n.to_datarecord()
        with pytest.warns(UserWarning, match="experimental"):
            n3 = pypsa.Network.from_datarecord(record)
        assert custom_equals(n, n3, ignore_attrs=ignore)

        # Importing must not leak into the default snapshot index.
        assert list(pypsa.Network().snapshots) == [0]
