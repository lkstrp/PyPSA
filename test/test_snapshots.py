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


def test_csv_import_rejects_legacy_now_snapshot(tmp_path):
    # Legacy PyPSA exported a `now` default snapshot label. Importing it
    # must raise, not silently resolve to the current wall-clock time.
    (tmp_path / "snapshots.csv").write_text(
        ",snapshot,objective,stores,generators\n0,now,1.0,1.0,1.0\n"
    )
    n = pypsa.Network()
    with pytest.raises(ValueError, match="snapshot"):
        n.import_from_csv_folder(tmp_path)


def test_csv_roundtrip_keeps_integer_snapshots(tmp_path):
    n = pypsa.Network()
    n.set_snapshots([0, 1, 2])
    n.export_to_csv_folder(tmp_path)

    n2 = pypsa.Network()
    n2.import_from_csv_folder(tmp_path)

    assert n2.snapshots.dtype == "int64"
    assert list(n2.snapshots) == [0, 1, 2]
