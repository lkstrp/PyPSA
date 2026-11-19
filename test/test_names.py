# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging

import pytest

import pypsa
from pypsa.network.names import deduplicate_names


def _base():
    n = pypsa.Network()
    n.add("Bus", "b")
    return n


def test_add_clash_raises_and_changes_nothing():
    n = _base()
    n.add("Generator", "x", bus="b")
    with pytest.raises(ValueError, match="Generator"):
        n.add("Load", "x", bus="b")
    assert n.c.loads.static.empty


def test_bus_name_exempt():
    n = _base()
    n.add("Generator", "x", bus="b")
    n.add("Bus", "x")


def test_overwrite_same_type():
    n = _base()
    n.add("Generator", "x", bus="b")
    n.add("Generator", "x", bus="b", overwrite=True)
    n.add("Generator", "x", bus="b", overwrite=True)


def test_csv_import_deduplicates(tmp_path, caplog):
    (tmp_path / "buses.csv").write_text("name\nb\n")
    (tmp_path / "loads.csv").write_text("name,bus\n0,b\n")
    (tmp_path / "lines.csv").write_text("name,bus0,bus1\n0,b,b\n")
    (tmp_path / "snapshots.csv").write_text("snapshot\nnow\n")
    n = pypsa.Network()
    with caplog.at_level(logging.WARNING):
        n.import_from_csv_folder(tmp_path)
    warns = [r for r in caplog.records if r.name == "pypsa.network.names"]
    assert len(warns) == 1
    assert list(n.c.loads.static.index) == ["0-Load"]
    assert n.c.loads.static.loc["0-Load", "bus"] == "b"
    assert n.deduplicated_names == {"Load": {"0": "0-Load"}}


def test_netcdf_roundtrip_clean(tmp_path, caplog):
    n = _base()
    n.add("Generator", "g", bus="b")
    n.add("Load", "l", bus="b")
    n.export_to_netcdf(tmp_path / "n.nc")
    m = pypsa.Network()
    with caplog.at_level(logging.WARNING):
        m.import_from_netcdf(tmp_path / "n.nc")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert m.deduplicated_names == {}
    assert pypsa.Network().deduplicated_names == {}


def test_merge_clash_raises_before_mutation():
    a = _base()
    a.add("Generator", "g", bus="b")
    o = pypsa.Network()
    o.add("Bus", "b2")
    o.add("Load", "g", bus="b2")
    with pytest.raises(ValueError, match="g"):
        a.merge(o, inplace=True)
    assert list(a.c.buses.static.index) == ["b"]
    assert a.c.loads.static.empty


def test_rename_clash_raises():
    n = _base()
    n.add("Generator", "g", bus="b")
    n.add("Load", "a", bus="b")
    with pytest.raises(ValueError, match="g"):
        n.c.loads.rename_component_names(a="g")


def test_deduplicate_names_three_types():
    n = _base()
    with pypsa.network.names.unchecked_names(n):
        n.add("Line", "s", bus0="b", bus1="b")
        n.add("Generator", "s", bus="b")
        n.add("Load", "s", bus="b")
        n.add("Load", "s-Load", bus="b")
    res = deduplicate_names(n)
    assert res == {"Generator": {"s": "s-Generator"}, "Load": {"s": "s-Load-2"}}
    assert list(n.c.lines.static.index) == ["s"]


def test_clustering_with_clashing_names_deduplicates():
    n = pypsa.Network()
    n.add("Bus", ["A0", "A1", "B0", "B1"], x=[0, 1, 5, 6], y=0)
    n.add(
        "Line",
        ["l0", "l1", "l2"],
        bus0=["A0", "A1", "A1"],
        bus1=["A1", "B0", "B1"],
        x=0.1,
        r=0.1,
        s_nom=10,
    )
    n.add("Load", ["0", "1", "2", "3"], bus=["A0", "A1", "B0", "B1"], p_set=1)
    busmap = {"A0": "A", "A1": "A", "B0": "B", "B1": "B"}
    clustered = n.cluster.spatial.cluster_by_busmap(busmap)
    names = [
        name
        for c in ("Line", "Load")
        for name in clustered.components[c].static.index.astype(str)
    ]
    assert len(names) == len(set(names))
    assert "0" in clustered.c.lines.static.index
    assert len(clustered.c.loads.static) == 4
    assert clustered.deduplicated_names


def test_link_add_clash_does_not_mutate():
    n = _base()
    n.add("Generator", "x", bus="b")
    cols = set(n.c.links.static.columns)
    with pytest.raises(ValueError, match="Generator"):
        n.add("Link", "x", bus0="b", bus1="b", bus2="b", bus3="b")
    assert set(n.c.links.static.columns) == cols


def test_name_owners_small_and_large_lookups_agree():
    import pandas as pd  # noqa: PLC0415

    from pypsa.network.names import (  # noqa: PLC0415
        SMALL_LOOKUP,
        name_owners,
        unchecked_names,
    )

    n = pypsa.Network()
    n.add("Bus", ["b0", "b1"])
    n.add("Load", ["x", "y"], bus="b0")
    with unchecked_names(n):
        n.add("Generator", ["y", "z"], bus="b0")
    wanted = ["x", "y", "z", "y", "none"]
    small = name_owners(n, pd.Index(wanted))
    assert small == {"x": ["Load"], "y": ["Generator", "Load"], "z": ["Generator"]}
    padded = wanted + [f"n{i}" for i in range(SMALL_LOOKUP)]
    assert name_owners(n, pd.Index(padded)) == small
