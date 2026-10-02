# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Test snapshot dtype handling."""

import pandas as pd
import pytest

import pypsa


def test_fresh_network_has_integer_snapshot():
    n = pypsa.Network()
    assert n.snapshots.equals(pd.Index([0], name="snapshot"))
    assert n.snapshots.dtype == "int64"


def test_set_snapshots_rejects_string_labels():
    n = pypsa.Network()
    with pytest.raises(ValueError, match="snapshot"):
        n.set_snapshots(["a", "b"])


def test_set_snapshots_accepts_integers():
    n = pypsa.Network()
    n.set_snapshots([0, 1])
    assert list(n.snapshots) == [0, 1]
    assert n.snapshots.dtype == "int64"


def test_set_snapshots_accepts_datetime_index():
    n = pypsa.Network()
    idx = pd.date_range("2015-01-01", freq="h", periods=3)
    n.set_snapshots(idx)
    assert isinstance(n.snapshots, pd.DatetimeIndex)


def test_set_snapshots_rejects_string_timestep_in_multiindex():
    n = pypsa.Network()
    mi = pd.MultiIndex.from_tuples([(2020, "a"), (2020, "b")])
    with pytest.raises(ValueError, match="timestep"):
        n.set_snapshots(mi)


def test_set_snapshots_accepts_int_period_datetime_timestep_multiindex():
    n = pypsa.Network()
    mi = pd.MultiIndex.from_tuples(
        [
            (2020, pd.Timestamp("2020-01-01")),
            (2020, pd.Timestamp("2020-01-02")),
        ]
    )
    n.set_snapshots(mi)
    assert n.snapshots.get_level_values("period").dtype.kind == "i"
    assert n.snapshots.get_level_values("timestep").dtype.kind == "M"


def test_set_snapshots_rejects_string_period_in_multiindex():
    n = pypsa.Network()
    mi = pd.MultiIndex.from_tuples(
        [
            ("a", pd.Timestamp("2020-01-01")),
            ("a", pd.Timestamp("2020-01-02")),
        ]
    )
    with pytest.raises(ValueError, match="period"):
        n.set_snapshots(mi)


def test_csv_import_converts_legacy_now_snapshot(tmp_path):
    # Legacy PyPSA exported a `now` default snapshot label. Importing it
    # must convert it to the integer snapshot `0`.
    (tmp_path / "snapshots.csv").write_text(
        ",snapshot,objective,stores,generators\n0,now,1.0,1.0,1.0\n"
    )
    n = pypsa.Network()
    n.import_from_csv_folder(tmp_path)
    assert list(n.snapshots) == [0]
    assert n.snapshots.dtype == "int64"


def test_csv_roundtrip_keeps_integer_snapshots(tmp_path):
    n = pypsa.Network()
    n.set_snapshots([0, 1, 2])
    n.export_to_csv_folder(tmp_path)

    n2 = pypsa.Network()
    n2.import_from_csv_folder(tmp_path)

    assert n2.snapshots.dtype == "int64"
    assert list(n2.snapshots) == [0, 1, 2]


def _legacy_network(index: pd.Index) -> pypsa.Network:
    """Build a network with a snapshot index that bypasses `set_snapshots`.

    Used to construct files that mimic what an older PyPSA version wrote,
    without hand-crafting the binary netCDF/HDF5 layout.
    """
    n = pypsa.Network()
    n._snapshots_data = pd.DataFrame(
        {"objective": 1.0, "stores": 1.0, "generators": 1.0}, index=index
    )
    return n


def test_netcdf_import_converts_legacy_now_snapshot(tmp_path):
    n = _legacy_network(pd.Index(["now"], name="snapshot"))
    fn = tmp_path / "network.nc"
    n.export_to_netcdf(fn)

    n2 = pypsa.Network(fn)
    assert list(n2.snapshots) == [0]
    assert n2.snapshots.dtype == "int64"


def test_hdf5_import_converts_legacy_now_snapshot(tmp_path):
    pytest.importorskip("tables")
    n = _legacy_network(pd.Index(["now"], name="snapshot"))
    fn = tmp_path / "network.h5"
    n.export_to_hdf5(fn)

    n2 = pypsa.Network(fn)
    assert list(n2.snapshots) == [0]
    assert n2.snapshots.dtype == "int64"


def test_csv_import_parses_legacy_string_date_labels(tmp_path):
    (tmp_path / "snapshots.csv").write_text(
        ",snapshot,objective,stores,generators\n"
        "0,2020-01-01 00:00,1.0,1.0,1.0\n"
        "1,2020-01-01 01:00,1.0,1.0,1.0\n"
    )
    n = pypsa.Network()
    n.import_from_csv_folder(tmp_path)
    assert isinstance(n.snapshots, pd.DatetimeIndex)
    assert list(n.snapshots) == [
        pd.Timestamp("2020-01-01 00:00"),
        pd.Timestamp("2020-01-01 01:00"),
    ]


def test_csv_import_converts_legacy_arbitrary_labels_to_positions(tmp_path, caplog):
    (tmp_path / "snapshots.csv").write_text(
        ",snapshot,objective,stores,generators\n"
        "0,a,1.0,1.0,1.0\n"
        "1,b,1.0,1.0,1.0\n"
        "2,c,1.0,1.0,1.0\n"
    )
    n = pypsa.Network()
    with caplog.at_level("WARNING"):
        n.import_from_csv_folder(tmp_path)

    assert list(n.snapshots) == [0, 1, 2]
    assert n.snapshots.dtype == "int64"
    assert "a" in caplog.text
    assert "b" in caplog.text
    assert "c" in caplog.text


def test_csv_import_converts_legacy_multiperiod_string_timesteps(tmp_path):
    (tmp_path / "snapshots.csv").write_text(
        ",period,timestep,objective,stores,generators\n"
        "0,2020,a,1.0,1.0,1.0\n"
        "1,2020,b,1.0,1.0,1.0\n"
        "2,2021,a,1.0,1.0,1.0\n"
        "3,2021,b,1.0,1.0,1.0\n"
    )
    n = pypsa.Network()
    n.import_from_csv_folder(tmp_path)
    assert list(n.snapshots.get_level_values("period")) == [2020, 2020, 2021, 2021]
    assert list(n.snapshots.get_level_values("timestep")) == [0, 1, 0, 1]


def test_csv_import_still_rejects_legacy_non_integer_period(tmp_path):
    (tmp_path / "snapshots.csv").write_text(
        ",period,timestep,objective,stores,generators\n"
        "0,a,2020-01-01 00:00,1.0,1.0,1.0\n"
        "1,a,2020-01-01 01:00,1.0,1.0,1.0\n"
    )
    n = pypsa.Network()
    with pytest.raises(ValueError, match="period"):
        n.import_from_csv_folder(tmp_path)


def test_legacy_now_network_attribute_dropped_on_import(tmp_path):
    n = _legacy_network(pd.Index(["now"], name="snapshot"))
    n.now = "now"
    n.export_to_csv_folder(tmp_path)

    n2 = pypsa.Network(tmp_path)
    assert not hasattr(n2, "now")


def test_set_snapshots_still_rejects_now_label():
    n = pypsa.Network()
    with pytest.raises(ValueError, match="snapshot"):
        n.set_snapshots(["now"])
