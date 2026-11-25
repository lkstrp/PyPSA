# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import dataclasses
import re

import pandas as pd
import pytest

import pypsa
from pypsa.components import PortTable, port_table, types
from pypsa.constants import PYPSA_DATA_DIR

_COMPONENTS_CSV = pd.read_csv(PYPSA_DATA_DIR / "components.csv", index_col=0)


def _expected(list_name: str) -> dict:
    """The table's facts for a type, read straight off the attribute CSVs."""
    name = _COMPONENTS_CSV.index[_COMPONENTS_CSV.list_name == list_name][0]
    category = _COMPONENTS_CSV.at[name, "category"]
    attrs = pd.read_csv(
        PYPSA_DATA_DIR / "component_attrs" / f"{list_name}.csv", index_col=0
    ).index
    one_port = category in ("controllable_one_port", "passive_one_port")
    unsuffixed = {"efficiency", "delay", "cyclic_delay"} if name == "Link" else set()
    ports: tuple[int, ...]
    if one_port:
        ports, stems = (0,), set()
    else:
        ports = tuple(sorted(int(a[3:]) for a in attrs if re.fullmatch(r"bus\d+", a)))
        stems = {
            m[1]
            for a in attrs
            if (m := re.fullmatch(r"(.+?)(\d+)", a)) and int(m[2]) in ports
        } | unsuffixed
    return {
        "ports": ports,
        "one_port": one_port,
        "extends": category == "controllable_branch",
        "unsuffixed": unsuffixed,
        "stems": stems,
    }


@pytest.mark.parametrize("list_name", list(types.all_components))
def test_table_matches_registry_csvs(list_name):
    table = port_table(list_name)
    expected = _expected(list_name)
    assert table.ports == expected["ports"]
    assert table.one_port == expected["one_port"]
    assert table.extends == expected["extends"]
    assert table.unsuffixed == expected["unsuffixed"]
    assert table.stems == expected["stems"]


def test_column():
    link = port_table("Link")
    assert link.column("efficiency", 1) == "efficiency"
    assert link.column("efficiency", 2) == "efficiency2"
    assert link.column("bus", 1) == "bus1"
    generator = port_table("Generator")
    assert generator.column("bus", 0) == "bus"
    assert generator.column("p", 0) == "p"
    assert port_table("Process").column("rate", 1) == "rate1"


def test_carries():
    line, link = port_table("Line"), port_table("Link")
    assert not line.carries("efficiency", 0)
    assert not link.carries("efficiency", 0)
    assert link.carries("efficiency", 2)
    assert not line.carries("bus", 3)
    assert link.carries("bus", 3)


def test_cached_and_frozen():
    assert port_table("Link") is port_table("Link")
    assert port_table("links") is port_table("Link")
    with pytest.raises(dataclasses.FrozenInstanceError):
        port_table("Link").extends = False


def test_ports_of_and_instance_property():
    n = pypsa.Network()
    n.add("Bus", ["a", "b"])
    n.add("Generator", "g", bus="a")
    n.add("GlobalConstraint", "gc", bus="a")
    n.add("Link", "l", bus0="a", bus1="b")
    assert port_table("GlobalConstraint").ports_of(n.c.global_constraints) == []
    assert port_table("Generator").ports_of(n.c.generators) == [0]
    assert port_table("Generator").buses(n.c.generators) == {0: "bus"}
    assert n.c.links.port_table is port_table("Link")
    assert isinstance(n.c.links.port_table, PortTable)


def test_public_names():
    assert "PortTable" in pypsa.components.__all__
    assert "port_table" in pypsa.components.__all__


@pytest.mark.parametrize("name", ["Link", "Process"])
def test_runtime_extension_follows_table(name):
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", ["a", "b", "c"])
    c = n.c[name]
    before_static = set(c.static.columns)
    before_dynamic = set(c.dynamic.keys())
    n.add(name, "x", bus0="a", bus1="b", bus2="c")

    table = c.port_table
    at_port_2 = {table.column(stem, 2) for stem in table.stems}
    new_rows = set(c.defaults.index) - set(types.get(name).defaults.index)
    new_static = set(c.static.columns) - before_static
    new_dynamic = set(c.dynamic.keys()) - before_dynamic
    assert new_rows
    assert new_rows <= at_port_2
    assert new_static <= at_port_2
    assert new_dynamic <= at_port_2
    assert "bus2" in new_rows
    assert table.ports_of(c) == [0, 1, 2]
    assert table.buses(c)[2] == "bus2"


def test_port_suffix_reads_table():
    n = pypsa.Network()
    assert n.c.links._port_suffix("1") == ""
    assert n.c.links._port_suffix("2") == "2"
    assert n.c.processes._port_suffix("1") == "1"
    assert n.c.processes._port_suffix("0") == "0"


def test_custom_type_has_a_table():
    defaults = pd.DataFrame(
        {
            "attribute": ["name", "bus0", "bus1", "p0", "p1"],
            "type": ["string"] * 3 + ["series"] * 2,
            "unit": ["n/a"] * 5,
            "default": ["n/a", "", "", 0.0, 0.0],
            "description": ["name", "bus 0", "bus 1", "p 0", "p 1"],
            "status": ["Input (required)"] * 3 + ["Output"] * 2,
        }
    )
    types.add_component_type(
        name="PortTableCustom",
        list_name="port_table_customs",
        description="A custom two-port type",
        category="custom",
        defaults_df=defaults,
    )
    try:
        table = port_table("port_table_customs")
        assert table.name == "PortTableCustom"
        assert table.ports == (0, 1)
        assert table.stems == {"bus", "p"}
        assert table.carries("p", 1)
    finally:
        del types.all_components["port_table_customs"]
