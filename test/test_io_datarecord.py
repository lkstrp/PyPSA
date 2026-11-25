# SPDX-FileCopyrightText: datarecord contributors
#
# SPDX-License-Identifier: MIT

"""The datarecord tool: verify a record, build a network, read it back.

Notes
-----
- [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
"""

import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datarecord")

import duckdb
from datarecord import Revision, Workspace
from datarecord.schema.specs import Dimension, PropertySpec
from datarecord.tools.base import Requirements, Schema, UnsupportedRecordError
from datarecord_helpers import (
    HandBuilt,
    as_written_at_8889d623,
    assert_networks_equal,
    empty_child,
    empty_root,
    export_network,
    relation,
    schema,
    unpacked,
    write_attribute,
    write_component,
)
from duckdb import DuckDBPyRelation

import pypsa
from pypsa.network.io.datarecord import PyPSA
from pypsa.network.io.datarecord.schema import (
    _AXIS_TYPES,
    CTYPE,
    _input_names,
    _registry_schema,
)
from pypsa.network.io.datarecord.source import _colliding_names, _weightings
from pypsa.network.names import EXEMPT_TYPES, unchecked_names


@pytest.fixture
def workspace():
    ws = Workspace()
    yield ws
    ws.close()


@pytest.fixture
def store(workspace, tmp_path):
    return workspace.store(str(tmp_path))


@pytest.fixture(scope="session")
def ac_dc():
    """PyPSA's `ac_dc_meshed`, from the in-repo examples."""
    import pypsa

    return pypsa.examples.ac_dc_meshed()


@pytest.fixture(scope="module")
def single_revision(tmp_path_factory, ac_dc):
    """`ac_dc` exported once for the module, for tests that only read it."""
    workspace = Workspace()
    yield export_network(ac_dc, workspace.store(str(tmp_path_factory.mktemp("ac_dc"))))
    workspace.close()


@pytest.fixture(scope="module")
def solved(single_revision):
    """The network `single_revision` builds, optimised once for the module."""
    n = PyPSA.build(single_revision.record)
    n.optimize(solver_name="highs")
    return n


@pytest.fixture(scope="module")
def results(single_revision, solved) -> Revision:
    """`solved`'s results, written once as a thin layer over `single_revision`."""
    return _results_over(single_revision, solved)


@pytest.fixture(scope="module")
def results_built(results):
    """The network `results` builds, once for the module, so a test changes a copy."""
    return PyPSA.build(results.record)


@pytest.fixture
def exported(single_revision, workspace, tmp_path) -> HandBuilt:
    """`single_revision`'s files in a hand-built root of a new store, under its schema.

    For a test to add files or redeclare the schema before it commits.
    """
    store = workspace.store(str(tmp_path / "ac_dc"))
    layer = empty_root(store, single_revision.manifest.schema)
    source = Path(single_revision.store.layer_dir(single_revision.id))
    for kind in ("dimensions", "attributes"):
        shutil.copytree(source / kind, layer.dir / kind)
    return layer


def _results_over(revision, n, **labels):
    """`n`'s results, written as a thin layer over `revision`, over its selection at `labels` where given."""
    record = revision.record
    layer = PyPSA.to_datarecord(
        n,
        record.con,
        results_only=True,
        base=record.select(**labels) if labels else record,
    )
    return revision.store.commit(layer, parent=revision)


def _assert_results_equal(got, expected, *series):
    """`got` holds `expected`'s capacities, dispatch and prices, and each of `series`."""
    pd.testing.assert_series_equal(
        got.c.Generator.static["p_nom_opt"],
        expected.c.Generator.static["p_nom_opt"],
        check_names=False,
    )
    for ctype, attr in (("Generator", "p"), ("Bus", "marginal_price"), *series):
        pd.testing.assert_frame_equal(
            got.c[ctype].dynamic[attr],
            expected.c[ctype].dynamic[attr],
            check_names=False,
            check_index_type=False,
            check_column_type=False,
            obj=attr,
        )


def _sub_networks(revision) -> list[str]:
    return _sub_networks_of(revision.record.dimensions["sub_network"])


def _sub_networks_of(axis) -> list[str]:
    return sorted(axis.df()["sub_network"])


def _prices(revision) -> pd.DataFrame:
    price = revision.record.attributes["marginal_price"].df()
    return price.pivot(index="snapshot", columns="bus", values="value")


def _flat(rel: DuckDBPyRelation) -> pd.DataFrame:
    """A connection attribute's rows with the `connection` struct spread to `entity` and `port`."""
    return rel.project("*, connection.entity AS entity, connection.port AS port").df()


def test_requirements_falsy_when_empty():
    """`bool(Requirements())` distinguishes "nothing missing" from "nothing checked"."""
    assert not Requirements()
    assert Requirements(dimensions=frozenset({"snapshot"}))
    assert (
        "dimensions ['snapshot']"
        in Requirements(dimensions=frozenset({"snapshot"})).describe()
    )
    assert Requirements().describe() == "nothing"


def test_build_refuses_its_own_network_source(workspace, ac_dc):
    """The tool's own network source answers no flags, so `build` refuses it."""
    with pytest.raises(TypeError, match="builds from a Record"):
        PyPSA.build(PyPSA.to_datarecord(ac_dc, workspace.con))


def test_verify_accepts_a_complete_record(single_revision):
    """A record written by `export_to_parquet` supplies everything PyPSA needs."""
    assert not PyPSA.verify(single_revision.record)


def test_the_tool_writes_no_type_axis(single_revision):
    """The schema lists every type, so the record reads them from it.

    Notes
    -----
    - [a dim with one_of](https://energy-models.github.io/datarecord/design/model/dims/#a-dim-with-one_of)
    """
    record = single_revision.record
    layer = Path(single_revision.store.layer_dir(single_revision.id))
    assert not (layer / "dimensions" / "ctype.parquet").exists()
    one_of = record.schema.dimensions["ctype"].one_of
    assert one_of is not None
    assert record.dimensions["ctype"].df()["ctype"].tolist() == list(one_of), (
        "every label of `one_of`, those the network leaves unused included"
    )


def test_the_tool_writes_no_role_dim(single_revision):
    """A connection is keyed by its port, so the record holds no `role` dim."""
    record = single_revision.record
    layer = Path(single_revision.store.layer_dir(single_revision.id))
    assert "role" not in record.schema.dimensions
    assert not (layer / "dimensions" / "role.parquet").exists()


def test_the_attributes_reaching_a_connection_differ_by_port(single_revision):
    """A Link's `delay` starts at port 1, so no connection at port 0 carries one.

    `port` is a coordinate of `connection`, not a category, so `flags` cannot
    select by it: the rows themselves say which attributes reach each port.

    Notes
    -----
    - [multi-dims](https://energy-models.github.io/datarecord/design/model/multi-dims/)
    """
    record = single_revision.record
    over = [
        a for a in record.attributes if "connection" in record.schema.dimensions_of(a)
    ]

    def reaching(port: int) -> set[str]:
        tuples = record.dimensions["connection"].filter(
            f"struct_extract(connection, 'port') = {port}"
        )
        return {
            a
            for a in over
            if record.attributes[a].join(tuples, "connection").limit(1).fetchall()
        }

    assert {"delay", "cyclic_delay"} <= reaching(1)
    assert not {"delay", "cyclic_delay"} & reaching(0)


def test_buses_and_carriers_are_labels_of_their_dims(single_revision, ac_dc):
    """`n.buses` and `n.carriers` are the `bus` and `carrier` axes, under PyPSA's names.

    Notes
    -----
    - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)
    """
    record = single_revision.record
    assert not {"Bus", "Carrier"} & set(record.schema.dimensions["ctype"].one_of or ())
    assert record.properties["bus"].columns == [
        "bus",
        "carrier",
        "type",
        "x",
        "y",
        "unit",
        "location",
        "country",
    ]
    assert record.properties["carrier"].columns == [
        "carrier",
        "color",
        "nice_name",
    ]
    assert record.schema.attributes["v_nom"].dims == {"bus", "scenario"}
    assert record.schema.attributes["co2_emissions"].dims == {"carrier", "scenario"}

    built = PyPSA.build(record)
    for ctype in ("Bus", "Carrier"):
        original = ac_dc.c[ctype].static
        pd.testing.assert_frame_equal(
            built.c[ctype].static[original.columns],
            original,
            check_dtype=False,
            check_index_type=False,
        )


def test_an_unlisted_carrier_is_added_and_an_empty_one_is_none(store):
    """A carrier named but not in `n.carriers` becomes a label; `""` names none.

    Notes
    -----
    - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)
    """
    n = _north()
    n.add("Carrier", "AC", color="black")
    n.add("Generator", "wind", bus="north", carrier="wind")

    record = export_network(n, store).record
    assert record.dimensions["carrier"].df()["carrier"].tolist() == ["AC", "wind"]
    carriers = record.properties["entity"].df()
    assert carriers.set_index("entity")["carrier"].isna().to_dict() == {
        "gen": True,
        "wind": False,
    }, "an empty carrier is stored as NULL"

    built = PyPSA.build(record)
    assert built.c.Carrier.static.index.to_list() == ["AC", "wind"]
    assert built.c.Generator.static["carrier"].to_dict() == {"gen": "", "wind": "wind"}


def test_a_property_is_addressed_by_its_subject(single_revision):
    """`x` is a line's reactance and `bus.x` a bus's coordinate; `carrier` alone is neither.

    Notes
    -----
    - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)
    - [set](https://energy-models.github.io/datarecord/design/use/editing/#set)
    """
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(single_revision.record)
    w.set("x", 0.5, entity="0")
    w.set("bus.x", 10.2, bus="London")

    x = w.attributes["x"].df().set_index("entity")["value"]
    assert x["0"] == 0.5
    assert "London" not in x.index, "`x` reaches no bus"
    buses = w.properties["bus"].df().set_index("bus")
    assert buses.at["London", "x"] == 10.2
    with pytest.raises(KeyError, match=r"`bus\.carrier` or `entity\.carrier`"):
        w.set("carrier", "AC")


def test_a_carrier_is_a_live_label(single_revision):
    """`entity.carrier` names a label, and a label named is not removed.

    Notes
    -----
    - [add / remove](https://energy-models.github.io/datarecord/design/use/editing/#add-remove)
    """
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(single_revision.record)
    with pytest.raises(KeyError, match="coal"):
        w.set("entity.carrier", "coal", entity="Manchester Gas")
    with pytest.raises(ValueError, match="entity.carrier"):
        w.remove("carrier", ["gas"])


def test_a_value_its_type_does_not_take_is_reported_and_left_out(single_revision):
    """A broadcast reaches every entity; the build keeps it to the types taking it.

    Notes
    -----
    - [a value its type does not take](https://energy-models.github.io/datarecord/design/tools/pypsa/#a-value-its-type-does-not-take)
    """
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(single_revision.record)
    w.set("p_max_pu", 0.9)

    report = PyPSA.verify(w)
    assert not report, "a value its type does not take fails no build"
    assert {("Load", "p_max_pu"), ("Line", "p_max_pu")} <= report.ignored
    assert ("Link", "p_max_pu") not in report.ignored, "a link takes p_max_pu"
    n = PyPSA.build(w)
    assert (n.c.Link.static["p_max_pu"] == 0.9).all()
    assert "p_max_pu" not in n.c.Load.static
    assert "p_max_pu" not in n.c.Load.dynamic


def _add_generator(w, name: str) -> None:
    w.add("entity", [name], ctype="Generator")
    w.add("connection", [{"entity": name, "port": 0}], bus="b")


@pytest.mark.parametrize(
    ("layers", "built"),
    [
        (
            [
                lambda w: (
                    w.set("p_max_pu", 0.5, entity="g1"),
                    w.set("p_max_pu", 0.6, entity="g2"),
                ),
                lambda w: w.set("p_max_pu", 0.9, scenario="high"),
            ],
            {
                ("low", "g1"): 0.5,
                ("low", "g2"): 0.6,
                ("high", "g1"): 0.9,
                ("high", "g2"): 0.9,
            },
        ),
        (
            [
                lambda w: w.set("p_max_pu", 0.9, scenario="high"),
                lambda w: w.set("p_max_pu", 0.5, entity="g1"),
            ],
            {
                ("low", "g1"): 0.5,
                ("low", "g2"): 1.0,
                ("high", "g1"): 0.5,
                ("high", "g2"): 0.9,
            },
        ),
        (
            [
                lambda w: w.set("p_max_pu", 0.9),
                lambda w: _add_generator(w, "g3"),
            ],
            {
                ("low", "g1"): 0.9,
                ("low", "g2"): 0.9,
                ("low", "g3"): 0.9,
                ("high", "g1"): 0.9,
                ("high", "g2"): 0.9,
                ("high", "g3"): 0.9,
            },
        ),
    ],
    ids=["scenario-crosses-entities", "entity-crosses-a-scenario", "added-member"],
)
def test_a_build_settles_rows_of_several_layers_by_the_broadcast_rule(
    store, layers, built
):
    """Rows as written, a deeper layer crossing a shallower one, build each member's value.

    Notes
    -----
    - [the broadcast rule](https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule)
    """
    from datarecord.mutable import WorkingRecord

    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", ["g1", "g2"], bus="b")
    n.set_scenarios({"low": 0.5, "high": 0.5})
    revision = export_network(n, store)
    for edit in layers:
        w = WorkingRecord(revision.record)
        edit(w)
        revision = revision.store.commit(w)

    static = PyPSA.build(revision.record).c.Generator.static
    assert static["p_max_pu"].to_dict() == built


def test_verify_reports_an_entity_attached_to_no_bus(single_revision):
    """Removing a bus's connections, then the bus, leaves what was attached to it unbuildable.

    Notes
    -----
    - [deletion](https://energy-models.github.io/datarecord/design/layers/removal/)
    """
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(single_revision.record)
    held = unpacked(single_revision.record.properties["connection"])
    at = held[held["bus"] == "Manchester"]
    w.remove(
        "connection",
        [
            {"entity": entity, "port": port}
            for entity, port in zip(at["entity"], at["port"], strict=True)
        ],
    )
    w.remove("bus", ["Manchester"])

    missing = PyPSA.verify(w).attributes
    assert {("Generator", "bus"), ("Load", "bus")} <= missing
    assert ("Link", "bus0") not in missing, "no link is at Manchester"


def _small_record(store):
    """A bus `b`, a generator, a line and a link, over two snapshots."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", ["b", "c"], v_nom=1.0)
    n.add("Generator", "g", bus="b")
    n.add("Line", "l", bus0="b", bus1="c", x=0.1, r=0.1)
    n.add("Link", "k", bus0="b", bus1="c")
    return export_network(n, store).record


@pytest.mark.parametrize(
    ("entity", "port", "expected"),
    [
        ("g", 1, {("Generator", "bus")}),
        ("l", 2, {("Line", "bus")}),
    ],
)
def test_verify_reports_a_connection_at_a_port_its_type_lacks(
    store, entity, port, expected
):
    """A one-port or two-port type has no bus column at that port, so `build` refuses."""
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(_small_record(store))
    w.add("port", [port])
    w.add("connection", [{"entity": entity, "port": port}], bus="c")

    assert PyPSA.verify(w).unsupported_values == expected
    with pytest.raises(UnsupportedRecordError):
        PyPSA.build(w)


def test_verify_reports_both_types_with_a_bad_port(store):
    """A generator at port 1 and a line at port 2 are both reported."""
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(_small_record(store))
    w.add("port", [2])
    w.add("connection", [{"entity": "g", "port": 1}], bus="c")
    w.add("connection", [{"entity": "l", "port": 2}], bus="c")

    assert PyPSA.verify(w).unsupported_values == {
        ("Generator", "bus"),
        ("Line", "bus"),
    }
    with pytest.raises(UnsupportedRecordError):
        PyPSA.build(w)


def test_a_value_at_a_port_without_a_connection_is_left_out(store):
    """A Link `efficiency` row at port 3 with no connection there builds without it."""
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(_small_record(store))
    w.add("port", [3])
    w.add("connection", [{"entity": "k", "port": 3}])
    w.set("efficiency", 0.5, entity="k", port=3)

    assert not PyPSA.verify(w)
    n = PyPSA.build(w)
    assert "efficiency3" not in n.c.Link.static.columns
    assert n.c.Link.static.loc["k", "efficiency"] == 1.0


@pytest.mark.parametrize("series", [False, True], ids=["static", "series"])
@pytest.mark.parametrize(
    ("port", "efficiency"), [(0, 1.0), (1, 0.7)], ids=["port-0", "port-1"]
)
def test_a_value_at_a_port_without_its_column_is_left_out(
    store, series, port, efficiency
):
    """A Link has no `efficiency0`, so `build` leaves a value at port 0 out."""
    from datarecord.mutable import WorkingRecord

    n = _north()
    n.add("Bus", "south")
    n.add("Link", "in_south", bus0="south", bus1="north")
    w = WorkingRecord(export_network(n, store).record)
    at = {"snapshot": list(n.snapshots)} if series else {}
    w.set(
        "efficiency",
        [0.7, 0.7] if series else 0.7,
        entity="in_south",
        port=port,
        **at,
    )

    assert not PyPSA.verify(w), "a value PyPSA has no column for fails no build"
    built = PyPSA.build(w).c.Link
    values = (
        built.dynamic["efficiency"]["in_south"].tolist()
        if series and port == 1
        else [built.static.at["in_south", "efficiency"]]
    )
    assert set(values) == {efficiency}


@pytest.mark.parametrize(
    ("ctype", "entity", "port", "dropped"),
    [
        ("Generator", "gen", 1, None),
        ("Generator", "gen", 1, 0),
        ("Line", "line", 2, None),
    ],
    ids=["both", "port-1", "line-port-2"],
)
def test_verify_reports_a_bus_at_a_port_its_type_lacks(
    store, ctype, entity, port, dropped
):
    """A Generator has its `bus` at port 0 alone and a Line at ports 0 and 1, so a connection at another port blocks the build."""
    from datarecord.mutable import WorkingRecord

    n = _north()
    n.add("Bus", "south")
    n.add("Line", "line", bus0="north", bus1="south", x=0.1)
    n.add("Link", "link", bus0="north", bus1="south", bus2="south")
    w = WorkingRecord(export_network(n, store).record)
    if dropped is not None:
        w.remove("connection", [{"entity": entity, "port": dropped}])
    w.add("connection", [{"entity": entity, "port": port}], bus="south")

    assert (ctype, "bus") in PyPSA.verify(w).unsupported_values
    with pytest.raises(UnsupportedRecordError, match=ctype):
        PyPSA.build(w)


def _connections_at(record, bus):
    held = unpacked(record.properties["connection"])
    at = held[held["bus"] == bus]
    return [
        {"entity": e, "port": int(p)}
        for e, p in zip(at["entity"], at["port"], strict=True)
    ]


def test_removing_a_bus_a_connection_names_is_refused(store):
    """Datarecord core refuses it, so the tool adds no code for it."""
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(_small_record(store))
    with pytest.raises(ValueError, match=r"remove\('bus'.*connection\.bus"):
        w.remove("bus", ["b"])


def test_removing_the_connections_then_the_bus_builds(store):
    """Removing the connections at the bus, then the bus, still builds."""
    from datarecord.mutable import WorkingRecord

    record = _small_record(store)
    w = WorkingRecord(record)
    w.remove("connection", _connections_at(record, "c"))
    w.remove("bus", ["c"])
    w.remove("entity", ["l", "k"])

    assert not PyPSA.verify(w)
    n = PyPSA.build(w)
    assert list(n.c.Bus.static.index) == ["b"]


def test_requires_reports_the_records_own_types(single_revision):
    """`requires` is record-dependent: PyPSA's axes plus this record's types."""
    req = PyPSA.requires(single_revision.record)
    assert {"snapshot", "period", "scenario"} <= req.dimensions
    assert "Generator" in req.categories
    # Required attributes come from PyPSA's registry, not a list we maintain.
    assert ("Generator", "bus") in req.attributes
    assert ("Line", "x") in req.attributes
    # `name` is the member identity, never an attribute row.
    assert not [a for a in req.attributes if a[1] == "entity"]


def test_verify_reports_a_missing_dim(exported):
    """A schema that declares no `scenario` dim cannot build a network.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    revision = _with_schema(exported, dims={"snapshot": "datetime"}, partial=set())

    missing = PyPSA.verify(revision.record)
    assert missing
    assert missing.dimensions == {"period", "scenario"}
    # A build refuses rather than failing deep inside PyPSA.
    with pytest.raises(UnsupportedRecordError, match="scenario"):
        PyPSA.build(revision.record)


def test_verify_reports_a_type_the_tool_does_not_know(exported):
    """A type outside PyPSA's registry is reported by `verify`, not raised in the fold.

    The type must be one the *schema* declares - `ctype` has `one_of` and
    its list is upheld everywhere, so a record cannot hold a type its own
    schema excludes. What it can hold is a type this tool has no
    registry entry for, which reads back fine and is this tool's business rather
    than the record layer's. `Requirements.categories` is what carries it.

    Notes
    -----
    - [entity types](https://energy-models.github.io/datarecord/design/model/properties/#categories)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    # Declared alongside PyPSA's own types, since the `one_of` labels pin the
    # vocabulary.
    declared = exported.schema
    assert declared.categorizing("entity") == (CTYPE,)
    dim = declared.dimensions[CTYPE]
    assert dim.one_of is not None, "PyPSA lists its types on the dim"
    layer = replace(
        exported,
        schema=declared.model_copy(
            update={
                "dimensions": {
                    **declared.dimensions,
                    CTYPE: dim.model_copy(update={"one_of": (*dim.one_of, "Widget")}),
                }
            }
        ),
    )
    write_component(layer, "Widget", [{"entity": "w1"}])
    revision = layer.commit()

    missing = PyPSA.verify(revision.record)
    assert missing.categories == {"Widget"}
    # The types PyPSA does know are not reported.
    assert "Generator" not in missing.categories
    with pytest.raises(UnsupportedRecordError, match="Widget"):
        PyPSA.build(revision.record)


_DIMS = {
    "snapshot": "datetime",
    "period": "int",
    "scenario": "str",
}


def _with_schema(layer: HandBuilt, **kwargs) -> Revision:
    """`layer` committed under a redeclared schema, keeping the attributes it already declares.

    Hand-built rather than written through `store.commit`, which would reject
    an incompatible redeclaration, and the tool must report on such a schema.

    Notes
    -----
    - [versioning](https://energy-models.github.io/datarecord/design/model/schema/#versioning)
    """
    was = layer.schema
    now = schema(**kwargs)
    labelled = {d: spec for d, spec in was.dimensions.items() if d not in _DIMS}
    redeclared = {d: s for d, s in now.dimensions.items() if d not in labelled}
    now = now.model_copy(update={"dimensions": {**labelled, **redeclared}})
    # Keep the old attributes, but drop any dim a redeclaration removed from
    # their address - an attribute over a dim the schema no longer declares is
    # rejected at construction, and the point here is the redeclared dims, not a
    # stale attribute reference to a gone one.
    addressable = set(now.dimensions)
    attributes = {
        name: pruned
        for name, spec in was.attributes.items()
        if (pruned := spec.model_copy(update={"dims": spec.dims & addressable})).dims
    }
    now = now.model_copy(
        update={
            "attributes": attributes,
            "properties": was.properties,
            "meta": was.meta,
        }
    )
    return replace(layer, schema=now).commit()


def test_verify_reports_snapshot_in_partial(exported):
    """`snapshot` as a key is reported, not crashed on.

    A limit of this tool's *representation*, not of the layer format: PyPSA's
    static/series split needs a component's whole series for an attribute to
    come from one layer, so a stored `snapshot = NULL` broadcast row and a
    descendant's per-snapshot row would coexist with no single container to
    put the result in. The record layer therefore permits the
    declaration - every file does carry the column - and the tool catches it.

    Notes
    -----
    - [partial](https://energy-models.github.io/datarecord/design/layers/tree/#partial-the-granularity-of-an-override)
    """
    revision = _with_schema(exported, partial={"scenario", "snapshot"})

    missing = PyPSA.verify(revision.record)
    assert "snapshot" in missing.unsupported_partial
    assert "snapshot" in missing.describe()
    with pytest.raises(UnsupportedRecordError, match="snapshot"):
        PyPSA.build(revision.record)


def test_a_record_of_the_earlier_layout_is_refused(exported):
    """A `snapshot` dim carrying properties, the layout of pin `365f7a50`, is not migrated."""
    was = exported.schema
    properties = {
        **was.properties,
        "snapshot": {"period": PropertySpec(one_of="period")},
    }
    old = replace(exported, schema=was.model_copy(update={"properties": properties}))
    record = old.commit().record

    with pytest.raises(UnsupportedRecordError, match="earlier datarecord pin"):
        PyPSA.open(record)
    with pytest.raises(UnsupportedRecordError, match="earlier datarecord pin"):
        pypsa.Network.from_datarecord(record)
    with pytest.raises(UnsupportedRecordError, match="earlier datarecord pin"):
        PyPSA.verify(record)


def test_a_record_written_at_8889d623_is_refused_at_the_path(ac_dc, tmp_path):
    """Its manifest cannot become a record at this pin, so the path is checked first."""
    path = tmp_path / "record"
    Workspace.process().write(ac_dc.to_datarecord(), str(path))
    as_written_at_8889d623(path)

    with pytest.raises(UnsupportedRecordError, match="earlier datarecord pin"):
        pypsa.Network(path)
    with pytest.raises(UnsupportedRecordError, match="earlier datarecord pin"):
        pypsa.Network().import_from_datarecord(path)


def test_a_record_written_at_this_pin_opens_from_its_path(ac_dc, tmp_path):
    """The path check leaves a fresh export alone."""
    path = tmp_path / "record"
    ac_dc.export_to_datarecord(path)

    assert_networks_equal(pypsa.Network(path), ac_dc)


def _constraint_record(store):
    """A record of `_north` with one constraint, `cap`, as a committed revision."""
    n = _north()
    n.add("GlobalConstraint", "cap", sense="<=", constant=1.0)
    return export_network(n, store)


def test_a_constraint_label_named_like_an_entity_opens(store):
    """A `global_constraint` label may repeat an entity's name."""
    from datarecord.mutable import WorkingRecord

    w = WorkingRecord(_constraint_record(store).record)
    w.add("entity", ["cap"], ctype="Generator")
    w.add("connection", [{"entity": "cap", "port": 0}], bus="north")
    record = store.commit(w).record

    assert not PyPSA.verify(record).names
    for n in (
        PyPSA.open(record),
        pypsa.Network.from_datarecord(record),
        PyPSA.build(record),
    ):
        assert "cap" in n.c.generators.static.index
        assert "cap" in n.c.global_constraints.static.index

    merged = pypsa.Network()
    merged.add("Bus", "elsewhere")
    merged.import_from_datarecord(record)
    assert "cap" in merged.c.generators.static.index
    assert "cap" in merged.c.global_constraints.static.index
    assert "cap" not in merged.deduplicated_names.get("GlobalConstraint", {})


def test_a_record_with_a_constraint_entity_is_refused_as_an_earlier_layout(store):
    """`GlobalConstraint` members on the entity axis are the layout before the `global_constraint` dim."""
    child = empty_child(_constraint_record(store))
    write_component(child, "GlobalConstraint", [{"entity": "old"}])
    record = child.commit().record

    with pytest.raises(UnsupportedRecordError, match="earlier layout of this tool"):
        PyPSA.open(record)
    with pytest.raises(UnsupportedRecordError, match="earlier layout of this tool"):
        pypsa.Network.from_datarecord(record)
    with pytest.raises(UnsupportedRecordError, match="earlier layout of this tool"):
        PyPSA.verify(record)


def test_a_record_with_a_role_dim_is_refused_as_an_earlier_layout(store):
    """A `role` dim is the layout before connections were keyed by port."""
    old = schema(
        dims={
            "snapshot": "datetime",
            "period": "int",
            "scenario": "str",
            "role": "str",
        },
        multi_dims={
            "connection": {"entity": "entity", "bus": "entity", "role": "role"}
        },
    )
    record = empty_root(store, old).commit().record

    with pytest.raises(UnsupportedRecordError, match="earlier layout of this tool"):
        pypsa.Network.from_datarecord(record)
    with pytest.raises(UnsupportedRecordError, match="earlier layout of this tool"):
        PyPSA.verify(record)


def test_verify_reports_a_missing_required_attribute(exported, ac_dc):
    """A component type with no `bus` anywhere - no connection row supplies it.

    A port's bus is the `connection` tuple itself, declared as no attribute, so
    no default can stand in for it.
    """
    # A Generator member with no connection row supplying its bus
    # (https://energy-models.github.io/datarecord/design/tools/pypsa/#ports).
    write_component(exported, "Generator", [{"entity": "g1"}])
    # One file across every type, so the Generators' rows are dropped from it
    # rather than a per-type file being unlinked.
    path = exported.dir / "dimensions" / "connection.parquet"
    rows = pd.read_parquet(path)
    generators = set(ac_dc.c["Generator"].static.index) | {"g1"}
    entities = rows["connection"].map(lambda c: c["entity"])
    rows[~entities.isin(generators)].to_parquet(path, index=False)

    write_component(exported, "Line", [{"entity": "l1"}])

    missing = PyPSA.verify(exported.commit().record)
    assert ("Generator", "bus") in missing.attributes
    assert ("Line", "x") not in missing.attributes, (
        "PyPSA declares the required `x` with a default, resolvable with no row"
    )


def _curve_child(exported, attribute, extra=None, curves=None, scalar=True):
    """A child layer of `exported` holding curves on `attribute`."""
    curves = curves or {"Manchester Wind": ((0.0, 20.0), (50.0, 35.0))}
    if scalar:
        write_attribute(
            exported,
            "marginal_cost",
            [{"entity": "Manchester Wind", "value": 20.0}],
        )
    revision = exported.commit()
    child = empty_child(revision)
    write_attribute(
        child,
        attribute,
        [
            {"entity": name, "breakpoint": x, "value": v, **(extra or {})}
            for name, points in curves.items()
            for x, v in points
        ],
    )
    return child.commit()


def test_verify_accepts_a_curve_on_a_piecewise_pair(exported):
    """A curve on a pair of PyPSA's piecewise table is expressible, so not reported."""
    revision = _curve_child(exported, "marginal_cost")
    assert not PyPSA.verify(revision.record).unsupported_values


def test_verify_reports_a_curve_off_the_piecewise_table(exported):
    """`p_nom_max` has no piecewise counterpart, so its curve stays reported."""
    revision = _curve_child(exported, "p_nom_max")
    missing = PyPSA.verify(revision.record)
    assert ("Generator", "p_nom_max") in missing.unsupported_values
    assert "unsupported values" in missing.describe()
    with pytest.raises(UnsupportedRecordError):
        PyPSA.build(revision.record)


def test_verify_reports_a_curve_row_with_a_snapshot(exported):
    """PyPSA's piecewise frames have no snapshot axis, so a snapshot row stays reported."""
    revision = _curve_child(
        exported, "marginal_cost", {"snapshot": pd.Timestamp("2030-01-01")}
    )
    missing = PyPSA.verify(revision.record)
    assert ("Generator", "marginal_cost") in missing.unsupported_values


def test_registry_schema_declares_curves_on_piecewise_attributes():
    """Every y attribute of the piecewise table, per-port stems included, takes breakpoints."""
    attributes = _registry_schema().attributes
    for name in ("marginal_cost", "capital_cost", "efficiency", "rate"):
        assert attributes[name].breakpoints, name
    assert not attributes["p_nom"].breakpoints


def _north():
    """A bus `north` and a generator at it, over two snapshots."""
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "north")
    n.add("Generator", "gen", bus="north")
    return n


def test_to_datarecord_rejects_a_cross_type_name_collision(workspace):
    """PyPSA scopes names per type; a record scopes them across every entity type.

    Reported rather than repaired: renaming to `Generator:north` would hand back
    a network whose components PyPSA can no longer find by their own names, so
    the mismatch is the caller's to reconcile.

    Notes
    -----
    - [a label is unique within its dim](https://energy-models.github.io/datarecord/design/storage/format/#a-label-is-unique-within-its-dim)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    n = _north()
    n.add("Generator", "north", bus="north")
    with unchecked_names(n):
        n.add("Load", "north", bus="north")

    with pytest.raises(
        UnsupportedRecordError, match="more than one component type"
    ) as raised:
        PyPSA.to_datarecord(n, workspace.con)
    assert raised.value.missing.names == {"north"}
    assert "Generator, Load" in str(raised.value)
    assert "deduplicate_names" in str(raised.value)


def test_a_constraint_may_be_named_like_a_generator(store):
    """A `GlobalConstraint` is a label of `global_constraint`, so it may share an entity's name."""
    n = _north()
    n.add("GlobalConstraint", "gen", sense="<=", constant=1.0)

    built = PyPSA.build(export_network(n, store).record)
    for c in ("Generator", "GlobalConstraint"):
        pd.testing.assert_frame_equal(
            built.c[c].static, n.c[c].static, check_dtype=False
        )


def test_a_constraint_at_a_missing_bus_is_refused(workspace):
    n = _north()
    n.add("GlobalConstraint", "cap", bus="nowhere", sense="<=", constant=1.0)

    with pytest.raises(ValueError, match="cap.*nowhere|nowhere.*cap"):
        PyPSA.to_datarecord(n, workspace.con)


@pytest.mark.parametrize(
    ("name", "bus", "period"),
    [
        ("cap", "", None),
        ("cap", "north", None),
        ("cap", "", 2040),
        ("cap", "north", 2030),
        ("gen", "", None),
    ],
    ids=["plain", "at-bus", "in-period", "both", "named-like-an-entity"],
)
def test_a_global_constraint_is_a_label_of_its_dim(store, name, bus, period):
    n = _north()
    if period is not None:
        n.set_investment_periods([2030, 2040])
    n.add(
        "GlobalConstraint",
        name,
        bus=bus,
        investment_period=np.nan if period is None else period,
        constant=1.0,
    )
    revision = export_network(n, store)
    record = revision.record

    assert "global_constraint" in record.schema.dimensions
    assert "GlobalConstraint" not in record.schema.dimensions[CTYPE].one_of
    assert set(record.schema.properties["global_constraint"]) == {"type", "bus"}
    labels = [label for (label,) in record.dimensions["global_constraint"].fetchall()]
    assert labels == [name]
    assert "GlobalConstraint" not in set(_ctypes_written(revision))

    static = PyPSA.build(record).c.GlobalConstraint.static
    assert list(static.index) == [name]
    assert static.at[name, "bus"] == bus
    if period is None:
        assert np.isnan(static.at[name, "investment_period"])
    else:
        assert static.at[name, "investment_period"] == period


def test_the_exempt_types_are_the_records_own_dim_types():
    """The names rule exempts exactly the types the record keeps off the entity axis."""
    assert set(_AXIS_TYPES) == EXEMPT_TYPES


def _constraints(periods=False):
    """`_north` with constraints, one with no bus or period, one at a bus, and with periods one for a period and one with both."""
    n = _north()
    n.add("Bus", "south")
    if periods:
        n.set_investment_periods([2030, 2040])
    n.add("GlobalConstraint", "plain", sense="<=", constant=1.0)
    n.add("GlobalConstraint", "at_bus", bus="south", sense="<=", constant=2.0)
    if periods:
        n.add("GlobalConstraint", "in_period", investment_period=2040, constant=3.0)
        n.add(
            "GlobalConstraint",
            "both",
            bus="north",
            investment_period=2030,
            sense=">=",
            constant=4.0,
        )
    return n


@pytest.mark.parametrize("periods", [False, True], ids=["single", "periods"])
def test_global_constraints_round_trip_with_their_bus_and_period(store, periods):
    """The empty bus comes back as `""`, a constraint for no period as NaN."""
    n = _constraints(periods)
    record = export_network(n, store).record
    expected = n.c.GlobalConstraint.static

    for built in (PyPSA.build(record), PyPSA.open(record)):
        static = built.c.GlobalConstraint.static
        assert list(static.index) == list(expected.index)
        assert static["bus"].to_dict() == expected["bus"].to_dict()
        assert static["bus"]["plain"] == ""
        assert static["bus"]["at_bus"] == "south"
        assert static["constant"].to_dict() == expected["constant"].to_dict()
        if periods:
            got = static["investment_period"]
            assert got["both"] == 2030
            assert got["in_period"] == 2040
            assert np.isnan(got["plain"])
            assert static["sense"]["both"] == ">="


def test_a_constraint_added_to_an_opened_network_keeps_the_others(store):
    """A new `global_constraint` label is written without resetting the existing ones."""
    revision = export_network(_constraints(), store)
    n = pypsa.Network(revision.store.uri, revision=str(revision.id))
    n.add("GlobalConstraint", "extra", constant=2.0)
    n.commit()
    child = n._revision

    static = PyPSA.build(child.record).c.GlobalConstraint.static
    assert list(static.index) == ["plain", "at_bus", "extra"]
    assert static.at["plain", "constant"] == 1.0
    assert static.at["plain", "sense"] == "<="
    assert static.at["at_bus", "bus"] == "south"
    assert static.at["extra", "constant"] == 2.0


def test_a_carrier_added_to_an_opened_network_keeps_the_others(store):
    n = _north()
    n.add("Carrier", "gas", co2_emissions=0.2)
    revision = export_network(n, store)
    opened = pypsa.Network(revision.store.uri, revision=str(revision.id))
    opened.add("Carrier", "wind", co2_emissions=0.1)
    opened.commit()
    child = opened._revision
    static = PyPSA.build(child.record).c.Carrier.static
    assert static.at["gas", "co2_emissions"] == 0.2
    assert static.at["wind", "co2_emissions"] == 0.1


def test_a_bus_added_to_an_opened_network_keeps_a_pending_series(store):
    n = _north()
    n.add("Bus", "south")
    n.c.Bus.dynamic.v_mag_pu_set["north"] = [1.05, 1.1]
    revision = export_network(n, store)
    opened = pypsa.Network(revision.store.uri, revision=str(revision.id))
    opened.add("Bus", "east", v_nom=5.0)
    opened.commit()
    child = opened._revision
    built = PyPSA.build(child.record).c.Bus
    assert built.dynamic.v_mag_pu_set["north"].tolist() == [1.05, 1.1]
    assert built.static.at["east", "v_nom"] == 5.0


def test_stochastic_global_constraints_differ_per_scenario(store):
    n = _north()
    n.add("GlobalConstraint", ["co2", "gas"], sense="<=", constant=100.0)
    n.set_scenarios(["a", "b"])
    static = n.c.GlobalConstraint.static
    static.loc[("b", "co2"), "carrier_attribute"] = "gas_emissions"
    static.loc[("b", "co2"), "sense"] = ">="
    static.loc[("b", "co2"), "constant"] = 7.0
    static["mu"] = [1.0, 2.0, 3.0, 4.0]
    record = export_network(n, store).record

    for built in (PyPSA.build(record), PyPSA.open(record)):
        got = built.c.GlobalConstraint.static
        for column in ("carrier_attribute", "sense", "constant", "mu"):
            assert got[column].to_dict() == static[column].to_dict(), column


def test_a_re_solve_writes_a_constraint_result_per_scenario(store):
    """`mu` over `global_constraint`, different per scenario, reaches each scenario."""
    n = _north()
    n.add("GlobalConstraint", "co2", sense="<=", constant=100.0)
    n.set_scenarios(["a", "b"])
    revision = export_network(n, store)
    n = PyPSA.build(revision.record)
    n.c.GlobalConstraint.static["mu"] = [5.0, 6.0]
    revision = _results_over(revision, n)

    built = PyPSA.build(revision.record)
    assert built.c.GlobalConstraint.static["mu"].tolist() == [5.0, 6.0]


def test_flow_based_global_constraints_round_trip(store):
    """`ram` as a series, `scaling` beside a bus's own, `domain` and `ptdf_*` columns."""
    n = _north()
    n.add("Bus", "south")
    n.add(
        "GlobalConstraint",
        "flow",
        sense="<=",
        scaling=5.0,
        domain="nodal",
        ptdf_north=0.25,
        ptdf_south=-0.75,
    )
    n.c.GlobalConstraint.dynamic["ram"] = pd.DataFrame(
        {"flow": [10.0, 20.0]}, index=n.snapshots
    )
    n.c.GlobalConstraint.dynamic["mu_ram"] = pd.DataFrame(
        {"flow": [0.5, 0.0]}, index=n.snapshots
    )
    n.c.Bus.static.loc["south", "scaling"] = 2.0
    record = export_network(n, store).record

    for built in (PyPSA.build(record), PyPSA.open(record)):
        got = built.c.GlobalConstraint
        assert got.static.at["flow", "scaling"] == 5.0
        assert built.c.Bus.static.at["south", "scaling"] == 2.0
        assert got.static.at["flow", "domain"] == "nodal"
        assert got.static.at["flow", "ptdf_north"] == 0.25
        assert got.static.at["flow", "ptdf_south"] == -0.75
        assert got.dynamic["ram"]["flow"].tolist() == [10.0, 20.0]
        assert got.dynamic["mu_ram"]["flow"].tolist() == [0.5, 0.0]


def test_a_name_clash_message_caps_the_names_at_ten(workspace):
    n = _north()
    with unchecked_names(n):
        for i in range(12):
            n.add("Bus", f"x{i}")
            n.add("Generator", f"x{i}", bus=f"x{i}")
            n.add("Load", f"x{i}", bus=f"x{i}")
    with pytest.raises(UnsupportedRecordError, match="and 2 more"):
        PyPSA.to_datarecord(n, workspace.con)


def test_a_bus_and_a_generator_may_share_a_name(store):
    """A bus is a label of `bus`, not an entity, so `north` is one of each."""
    n = _north()
    n.add("Generator", "north", bus="north")

    record = export_network(n, store).record
    assert record.dimensions["bus"].df()["bus"].tolist() == ["north"]
    assert "north" in record.dimensions["entity"].df()["entity"].tolist()
    assert_networks_equal(PyPSA.build(record), n)


@pytest.mark.parametrize(
    ("ctype", "dim", "branch", "standard", "changed", "extra"),
    [
        (
            "LineType",
            "line_type",
            "Line",
            "Al/St 240/40 4-bundle 380.0",
            {"x_per_length": 0.5},
            {"length": 10.0},
        ),
        (
            "TransformerType",
            "transformer_type",
            "Transformer",
            "160 MVA 380/110 kV",
            {"vsc": 20.0},
            {"s_nom": [10.0, 20.0], "phase_shift": [1.0, 2.0], "tap_side": 1},
        ),
    ],
)
def test_a_networks_own_type_round_trips(
    store, ctype, dim, branch, standard, changed, extra
):
    """A type the network adds to PyPSA's library is a label of its own dim, its columns properties, and the library is not written.

    The branch typed by it shares its name, and a Transformer carries its
    own `s_nom`, `phase_shift` and `tap_side` beside the type's, none of
    which clash. Each branch builds back with the same `r` and `x`, and a
    results layer over the record writes no type.
    """
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add(
        "Bus",
        ["a", "b"],
        v_nom=[380.0, 110.0 if branch == "Transformer" else 380.0],
    )
    n.add(
        ctype,
        "own type",
        **{**n.c[ctype].static.loc[standard].to_dict(), **changed},
    )
    n.add(
        branch,
        ["std", "own type"],
        bus0="a",
        bus1="b",
        type=[standard, "own type"],
        **extra,
    )

    root = export_network(n, store)
    record = root.record
    assert record.dimensions[dim].df()[dim].tolist() == ["own type"], (
        "only the type the library lacks is written"
    )
    assert set(record.schema.properties[dim]) == set(n.c[ctype].static), (
        "every column of a type is a property of its dim"
    )
    held = record.properties[dim].df().set_index(dim).loc["own type"]
    assert held[list(changed)].to_dict() == changed
    assert set(_ctypes_written(root)) == {branch}, "a type is no entity"

    built = PyPSA.build(record)
    assert_networks_equal(built, n)
    built.calculate_dependent_values()
    n.calculate_dependent_values()
    pd.testing.assert_frame_equal(
        built.c[branch].static[["r", "x"]], n.c[branch].static[["r", "x"]]
    )

    child = _results_over(root, built)
    layer = Path(store.layer_dir(child.id)) / "dimensions"
    assert not (layer / "entity.parquet").exists()
    assert not (layer / f"{dim}.parquet").exists(), "a results layer writes no type"


def test_a_network_of_library_types_writes_none(single_revision):
    """`ac_dc` holds PyPSA's standard types alone, so the record holds no type axis."""
    held = set(single_revision.manifest.files.dimensions)
    assert not held & {"line_type", "transformer_type"}


@pytest.mark.parametrize(
    ("columns", "kind"),
    [
        ({"Bus": ("country", "DE")}, "property"),
        ({"Bus": ("nom_max_onwind_2050", 5.0)}, "attribute"),
        ({"Generator": ("country", "DE")}, "property"),
        ({"Generator": ("location", "site")}, "property"),
        ({"Generator": ("e_nom", 5.0)}, "attribute"),
        ({"Generator": ("v_nom", 1.0)}, "property"),
        ({"Carrier": ("efficiency", 0.9)}, "property"),
        (
            {"Bus": ("country", "DE"), "Generator": ("country", "DE")},
            "property",
        ),
        ({"Bus": ("share", 0.5), "Generator": ("share", 0.5)}, "property"),
    ],
    ids=[
        "text-bus",
        "number-bus",
        "text-entity",
        "text-entity-named-like-a-bus-property",
        "number-entity-named-like-another-types-input",
        "number-entity-named-like-a-bus-attribute",
        "number-carrier-named-like-a-connection-attribute",
        "text-on-two-dims",
        "number-on-two-dims",
    ],
)
def test_a_custom_column_is_a_property_if_text_or_clashing(store, columns, kind):
    """A text column describes its label, and a clashing name would be one attribute over two dims.

    Either is a property of its subject, any other custom column is an
    attribute over its type's dim. Every case is built back onto the types
    that carried it alone, a load included to receive nothing.
    """
    n = _north()
    n.add("Carrier", "AC")
    n.add("Load", "demand", bus="north")
    for ctype, (name, value) in columns.items():
        n.c[ctype].static[name] = value

    record = export_network(n, store).record
    for ctype, (name, _) in columns.items():
        dim = {"Bus": "bus", "Carrier": "carrier"}.get(ctype, "entity")
        as_property = name in record.schema.properties[dim]
        assert as_property == (kind == "property"), f"{name} is a {kind} of {dim}"
        if kind == "attribute":
            assert record.schema.attributes[name].dims == {dim, "scenario"}
    assert_networks_equal(PyPSA.build(record), n)


@pytest.mark.parametrize(
    ("name", "values", "match"),
    [
        (
            "country",
            ("DE", 1.0),
            "`country` is a property of `entity` with one dtype, but its types "
            "carry it as {'Generator': 'str', 'Load': 'float'}",
        ),
        (
            "tag",
            (True, 0.5),
            "`tag` is an attribute over `entity` with one dtype, but its types "
            "carry it as {'Generator': 'bool', 'Load': 'float'}",
        ),
    ],
    ids=["property", "attribute"],
)
def test_a_custom_column_carried_with_two_dtypes_is_refused(
    workspace, name, values, match
):
    """One property of `entity`, or one attribute over it, is one dtype, whichever types carry it."""
    n = _north()
    n.add("Load", "demand", bus="north")
    n.c.Generator.static[name] = values[0]
    n.c.Load.static[name] = values[1]

    with pytest.raises(ValueError, match=re.escape(match)):
        PyPSA.to_datarecord(n, workspace.con)


@pytest.mark.parametrize(
    ("columns", "kind", "dtype", "kept"),
    [
        ({"Line": [np.nan, True], "Link": np.nan}, "attribute", "bool", True),
        ({"Line": [np.nan, True], "Link": None}, "attribute", "bool", True),
        (
            {"Line": [np.nan, True], "Link": np.nan, "Bus": False},
            "property",
            "bool",
            True,
        ),
        (
            {"Line": [np.nan, "cable"], "Link": np.nan},
            "property",
            "str",
            "cable",
        ),
    ],
    ids=["float", "object", "bool-property", "text"],
)
def test_a_null_custom_column_takes_the_dtype_of_the_others(
    store, columns, kind, dtype, kept
):
    """A custom column all NULL on one type is declared with the dtype the types holding values give it, and read back NULL there."""
    n = _north()
    n.add("Bus", "south")
    n.add("Line", ["l1", "l2"], bus0="north", bus1="south", x=0.1)
    n.add("Link", "k", bus0="north", bus1="south")
    for ctype, value in columns.items():
        static = n.c[ctype].static
        if isinstance(value, list):
            value = pd.Series(value, index=static.index, dtype=object)
        static["underground"] = value

    record = export_network(n, store).record
    if kind == "property":
        declared = record.schema.properties["entity"]["underground"]
    else:
        assert "underground" not in record.schema.properties["entity"]
        declared = record.schema.attributes["underground"]
    assert declared.dtype == dtype
    built = PyPSA.build(record)
    held = [
        built.c.Link.static.at["k", "underground"],
        *built.c.Line.static["underground"],
    ]
    held = [None if pd.isna(value) else value for value in held]
    assert held == [None, None, kept]


@pytest.mark.parametrize("name", ["scenario", "period", "entity"])
def test_a_custom_column_named_like_a_key_is_refused(workspace, name):
    """A key of the record's rows names no property or attribute, text or not.

    Notes
    -----
    - [custom columns](https://energy-models.github.io/datarecord/reference/tools/#custom-columns)
    """
    n = _north()
    _with_bus_column(n, name, "x")
    with pytest.raises(ValueError, match="key a record's rows"):
        PyPSA.to_datarecord(n, workspace.con)


@pytest.mark.parametrize(
    ("ctype", "values", "dtype", "kept"),
    [
        ("Bus", [True, None], "bool", [True, None]),
        ("Bus", [3, None], "int", [3, None]),
        ("Bus", [np.int64(3), np.nan], "int", [3, None]),
        ("Bus", [1, 2.5], "float", [1.0, 2.5]),
        ("Generator", [1.5, None], "float", [1.5, None]),
        ("Generator", [False, True], "bool", [False, True]),
        ("Generator", [np.bool_(True), np.nan], "bool", [True, None]),
        ("Generator", [np.int64(3), np.int64(4)], "int", [3, 4]),
        ("Generator", ["a", None], "str", ["a", None]),
        ("Generator", [True, 2], "str", ["True", "2"]),
        ("Generator", [None, None], "str", [None, None]),
    ],
)
def test_a_custom_object_column_is_typed_by_its_values(
    store, ctype, values, dtype, kept
):
    """A custom column of dtype `object`, as a NaN leaves a column of bools or ints, is declared as its non-null values say, and written as declared, one of mixed types as text."""
    n = _north()
    n.add("Bus", "south")
    n.add("Generator", "south gen", bus="south")
    static = n.c[ctype].static
    static["extra"] = pd.Series(values, index=static.index, dtype=object)

    record = export_network(n, store).record
    dim = "bus" if ctype == "Bus" else "entity"
    spec = record.schema.properties[dim].get("extra")
    declared = record.schema.attributes["extra"] if spec is None else spec
    assert declared.dtype == dtype
    built = PyPSA.build(record).c[ctype].static["extra"]
    held = [None if pd.isna(value) else value for value in built]
    assert held == kept, "the values round-trip as declared"


def test_a_shape_is_a_property_of_the_label_it_outlines(store):
    """A shape is `{dim}.shape_{type}` as WKT, a `component` that is no PyPSA type a dim of its own, its CRS in the schema's `meta`.

    A custom `bus.country` refers to the `country` dim the shapes make, and
    a results layer writes none of it.
    """
    from geopandas.testing import assert_geodataframe_equal

    n = _shaped()
    revision = export_network(n, store)
    record = revision.record
    schema = record.schema
    assert schema.meta["crs"] == "EPSG:3035"
    assert "geometry" not in schema.attributes
    assert "Shape" not in set(_ctypes_written(revision)), "a shape is no entity"
    assert record.dimensions["country"].df()["country"].tolist() == ["DE", "FR"], (
        "a shape-made dim's labels are the shapes' `idx` in network order"
    )
    assert schema.properties["bus"]["country"].one_of == "country"
    bus = record.properties["bus"].df().set_index("bus")
    assert bus.loc["north", "shape_onshore"] == ("POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))")
    assert set(schema.properties["country"]) == {"shape_country", "shape_offshore"}
    assert "shape_site" in schema.properties["entity"]

    built = PyPSA.build(record)
    assert built.crs == n.crs
    assert "country DE offshore" in built.c.Shape.static.index, (
        "a built shape is named `{component} {idx} {type}`"
    )
    expected = _shapes(n)
    assert_geodataframe_equal(_shapes(built), expected, check_like=True)

    child = _results_over(revision, built)
    layer = Path(store.layer_dir(child.id)) / "dimensions"
    assert not (layer / "country.parquet").exists(), "a results layer writes no shape"
    rebuilt = PyPSA.build(child.record)
    assert_geodataframe_equal(_shapes(rebuilt), expected, check_like=True)

    for network in (built, n):
        network.remove("Shape", network.c.Shape.static.index)
    assert_networks_equal(built, n)


def test_two_components_outline_one_idx_with_one_type(store):
    """A `Bus` and a `country` both labelled `north` each keep their `onshore` shape, the component naming them apart."""
    n = _shaped()
    _add_shape(n, idx="north", component="country", type="onshore")
    built = PyPSA.build(export_network(n, store).record)
    assert {"Bus north onshore", "country north onshore"} <= set(
        built.c.Shape.static.index
    )


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        pytest.param(
            lambda n: _add_shape(n, idx="north", component="Bus", type="onshore"),
            r"share a `\(component, idx, type\)`",
            id="duplicate",
        ),
        pytest.param(
            lambda n: _add_shape(n, idx="north", type="onshore"),
            "lack a `component`, `idx` or `type`",
            id="no-component",
        ),
        pytest.param(
            lambda n: _add_shape(n, idx="north", component="Bus"),
            "lack a `component`, `idx` or `type`",
            id="no-type",
        ),
        pytest.param(
            lambda n: _add_shape(n, component="country", type="area"),
            "lack a `component`, `idx` or `type`",
            id="no-idx",
        ),
        pytest.param(
            lambda n: _add_shape(n, idx="Atlantis", component="Bus", type="onshore"),
            r"name \['Atlantis'\], no Bus",
            id="not-a-label",
        ),
        pytest.param(
            lambda n: _add_shape(n, idx="north", component="bus", type="onshore"),
            "a dim PyPSA's schema declares",
            id="pypsa-dim",
        ),
        pytest.param(
            lambda n: _with_bus_column(n, "shape_own", "x"),
            "no place",
            id="custom-shape-column",
        ),
        pytest.param(
            lambda n: _with_bus_column(n, "country", ["DE", "XX"]),
            r"names \['XX'\], no label of `country`",
            id="unknown-country",
        ),
    ],
)
def test_a_shape_naming_no_label_is_refused(workspace, edit, match):
    """A shape is one value of a property of one label, so `to_datarecord` refuses one that names none."""
    n = _shaped()
    edit(n)
    with pytest.raises(ValueError, match=match):
        PyPSA.to_datarecord(n, workspace.con)


def test_a_stochastic_network_holds_its_shapes_per_scenario(store):
    """A stochastic network's shapes are indexed by `(scenario, name)`, as its other components are."""
    from geopandas.testing import assert_geodataframe_equal

    n = _shaped()
    n.set_scenarios(["a", "b"])
    built = PyPSA.build(export_network(n, store).record)
    assert built.c.Shape.static.index.names == ["scenario", "name"]
    assert_geodataframe_equal(_shapes(built), _shapes(n), check_like=True)


def test_a_bus_selection_keeps_every_label_a_shape_made(store):
    """Selecting buses keeps every country shape, though `bus.country` refers to `country`, and drops the other bus's, as a network squeezes no `country`."""
    record = export_network(_shaped(), store).record
    built = PyPSA.build(record.select(bus=["north"]))
    outlined = _shapes(built).index.droplevel("type").unique().tolist()
    assert outlined == [
        ("Bus", "north"),
        ("Generator", "gen"),
        ("country", "DE"),
        ("country", "FR"),
    ]
    with pytest.raises(ValueError, match=r"not \['country'\]"):
        PyPSA.build(record.select(country="DE"))


def test_a_custom_attribute_differing_per_scenario_survives_a_round_trip(store):
    """A numeric custom column is an attribute over `scenario`, as `p_nom` is."""
    n = _north()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.Generator.static["score"] = [1.0, 2.0]

    built = PyPSA.build(export_network(n, store).record)
    assert built.c.Generator.static["score"].to_dict() == {
        ("low", "gen"): 1.0,
        ("high", "gen"): 2.0,
    }, "each scenario keeps its value"


def test_a_custom_property_differing_per_scenario_is_refused(workspace):
    """A text custom column is a property, which has no `scenario` dim."""
    n = _north()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.Bus.static["country"] = ["DE", "FR"]

    with pytest.raises(ValueError, match="'north' has `country` .* by scenario"):
        PyPSA.to_datarecord(n, workspace.con)


def test_a_shape_made_dim_is_no_dim_of_its_own(store):
    """The `country` dim the shapes make is no dim of the record's own, so it is a network's alone."""
    record = export_network(_shaped(), store).record
    assert "country" in record.dimensions
    assert not PyPSA.verify(record).unsupported_dimensions, (
        "`country` is no dim of the record's own"
    )


def test_sub_networks_are_labels_of_their_own_dim(store, ac_dc):
    """PyPSA numbers SubNetworks `0`, `1`, ... as it numbers Lines, which a dim of their own keeps apart.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    """
    n = ac_dc.copy()
    n.determine_network_topology()
    assert set(n.c["SubNetwork"].static.index) & set(n.c["Line"].static.index)

    revision = export_network(n, store)
    record = revision.record
    names = _sub_networks(revision)
    assert names == [str(i) for i in range(4)]
    pointed = set(record.attributes["sub_network"].df()["value"])
    assert pointed <= set(names), "a `sub_network` value names a `sub_network` label"
    carriers = record.properties["sub_network"].df()["carrier"]
    assert set(carriers) <= set(record.dimensions["carrier"].df()["carrier"])

    built = PyPSA.build(record)
    assert sorted(built.c["SubNetwork"].static.index) == ["0", "1", "2", "3"]


_PER_UNIT = ("x_pu", "r_pu", "g_pu", "b_pu", "x_pu_eff", "r_pu_eff")


def test_a_derived_v_nom_stays_on_the_bus(store, ac_dc):
    """`calculate_dependent_values` copies the bus's `v_nom` onto lines and shunts.

    The record keeps `v_nom` over `bus` alone and no per-unit impedance, and
    `build` gets both back once PyPSA recomputes them.
    """
    n = ac_dc.copy()
    n.add("ShuntImpedance", "shunt", bus="London", b=0.1)
    n.add("Bus", "London LV", v_nom=110.0, carrier="AC")
    n.add("Transformer", "trafo", bus0="London", bus1="London LV", x=0.1, s_nom=100)
    n.calculate_dependent_values()

    record = export_network(n, store).record
    assert record.schema.attributes["v_nom"].dims == {"bus", "scenario"}
    assert not set(_PER_UNIT) & set(record.schema.attributes)

    built = PyPSA.build(record)
    built.calculate_dependent_values()
    assert_networks_equal(built, n)


def test_a_solved_network_exports(store, solved):
    """`optimize` leaves the derived `Line.v_nom` behind, which `build` recomputes."""
    built = PyPSA.build(export_network(solved, store).record)
    built.calculate_dependent_values()
    pd.testing.assert_series_equal(
        built.c.Line.static.v_nom, solved.c.Line.static.v_nom
    )


def test_a_results_layer_holds_no_per_unit_impedance(results):
    """A solve sets the per-unit impedances, yet its results layer restates none of them."""
    record = results.record
    assert not set(_PER_UNIT) & set(record.schema.attributes)
    assert not set(_PER_UNIT) & set(record.attributes)


@pytest.fixture
def per_unit_written(store, ac_dc, monkeypatch):
    """`ac_dc` with a transformer, exported as a record written before the per-unit impedances were derived."""
    from pypsa.network.io.datarecord import schema

    n = ac_dc.copy()
    n.add("Bus", "London LV", v_nom=110.0, carrier="AC")
    n.add("Transformer", "trafo", bus0="London", bus1="London LV", x=0.1, s_nom=100)
    n.calculate_dependent_values()

    schema._registry.cache_clear()
    with monkeypatch.context() as patch:
        for ctype in ("Line", "ShuntImpedance", "Transformer"):
            patch.setitem(schema._DERIVED, ctype, frozenset({"v_nom"}))
        try:
            record = export_network(n, store).record
        finally:
            schema._registry.cache_clear()
    return n, record


def test_a_record_holding_per_unit_impedances_still_opens(per_unit_written):
    """A record that holds the per-unit impedances verifies and builds, and PyPSA recomputes them."""
    n, record = per_unit_written
    assert {"x_pu", "x_pu_eff"} <= set(record.schema.attributes)

    assert not PyPSA.verify(record)
    built = PyPSA.build(record)
    built.calculate_dependent_values()
    for ctype in ("Line", "Transformer"):
        pd.testing.assert_frame_equal(
            built.c[ctype].static[list(_PER_UNIT)],
            n.c[ctype].static[list(_PER_UNIT)],
            check_dtype=False,
        )
    assert "x_pu" not in built.c.Generator.static


def test_a_per_unit_impedance_is_recomputed_not_read(store):
    """A line exported before `calculate_dependent_values` gets its `x_pu_eff` from PyPSA, not a `0`."""
    import pypsa

    n = pypsa.Network(snapshots=pd.date_range("2030", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Bus", "c")
    n.add("Line", "ln", bus0="b", bus1="c", x=0.1, s_nom=10)

    record = export_network(n, store).record
    assert "x_pu_eff" not in record.schema.attributes

    built = PyPSA.build(record)
    built.calculate_dependent_values()
    assert built.c.lines.static.at["ln", "x_pu_eff"] != 0


def test_a_stochastic_network_is_not_a_false_collision():
    """A `(scenario, name)` index is one name per component, not one per scenario.

    Comparing the index tuples would report every scenario's copy as a distinct
    name while missing a real cross-type clash.
    """
    import pypsa

    n = pypsa.Network()
    n.set_scenarios(["a", "b"])
    n.add("Bus", "north")
    n.add("Generator", "north gen", bus="north")

    assert not _colliding_names(n)


def test_pypsa_schema_is_the_identity(single_revision):
    """PyPSA defines the record vocabulary, so nothing is renamed.

    The seam still routes every attribute, so an entry added later takes
    effect with no change to `build`.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    assert PyPSA.schema.attrs == {}
    assert PyPSA.schema.sources("Generator", "p_max_pu") == ("p_max_pu",)
    record = single_revision.record
    identity = PyPSA.schema.resolve(record, "Generator", "p_max_pu", {}, record.flags())
    assert identity.fetchall() == relation(single_revision, "p_max_pu").fetchall()


def test_schema_renames_and_computes():
    """A rename maps one source; a computed attribute maps several (`Attr.compute`)."""
    from datarecord.tools.base import Attr

    schema = Schema(
        {
            "Generator": (
                Attr(name="p_max_pu", source=("availability",)),
                Attr(
                    name="marginal_cost",
                    source=("fuel_cost", "efficiency"),
                    compute=lambda fuel, eff: fuel,
                ),
            )
        }
    )
    assert schema.sources("Generator", "p_max_pu") == ("availability",)
    assert schema.sources("Generator", "marginal_cost") == ("fuel_cost", "efficiency")
    # Anything unmapped, and any other type, stays the identity.
    assert schema.sources("Generator", "p_nom") == ("p_nom",)
    assert schema.sources("Link", "p_max_pu") == ("p_max_pu",)
    # A multi-source rename with no way to combine them is a declaration bug.
    with pytest.raises(ValueError, match="exactly one"):
        Attr(name="x", source=("a", "b"))


def test_a_solved_network_exports_its_results_long_form(single_revision, solved):
    """A solved network's results are attributes, keyed by name, long-form.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    n = solved

    results = PyPSA.to_datarecord(n, single_revision.record.con).attributes
    assert "p" in results
    assert "p_nom_opt" in results

    # Relations, lazy so a tool may fetch on demand (https://energy-models.github.io/datarecord/design/tools/).
    assert isinstance(results["p"], DuckDBPyRelation)
    p = results["p"].df()
    # The long schema's columns (https://energy-models.github.io/datarecord/design/use/reading/), so the write path can persist it as-is -
    # and no `ctype`, an attribute row being keyed by `name` (https://energy-models.github.io/datarecord/design/storage/format/#a-label-is-unique-within-its-dim).
    assert {"connection", "snapshot", "scenario", "value"} <= set(p.columns)
    assert "ctype" not in p.columns
    assert "period" not in p.columns, "a series over `snapshot` reaches its period"
    assert set(p["attribute"].tolist()) == {"p"}
    # Series output: one row per (name, snapshot), for the Generator rows -
    # selected by name, since the frame no longer carries the type.
    gens = set(n.c["Generator"].static.index)
    gen = _flat(results["p"]).query("entity in @gens")
    assert len(gen) == len(n.snapshots) * len(gens)

    # A static output has no snapshot, and only the components whose value
    # differs from the default appear (some generators solve to p_nom_opt=0).
    nom = results["p_nom_opt"].df()
    nom = nom[nom["entity"].isin(gens)]
    assert "snapshot" not in nom.columns
    nonzero = n.c["Generator"].static["p_nom_opt"] != 0.0
    assert set(nom["entity"]) == set(n.c["Generator"].static.index[nonzero])


def test_an_export_skips_results_still_at_their_default(single_revision):
    """An unsolved network yields no `p`/`p_nom_opt` rows.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    """
    n = PyPSA.build(single_revision.record)
    results = PyPSA.to_datarecord(n, single_revision.record.con).attributes
    assert "p" not in results
    assert "p_nom_opt" not in results


def test_a_solved_p_is_one_row_per_connection(single_revision, solved):
    """`p` holds a row per `(entity, port)` per snapshot, each port's own sign.

    Notes
    -----
    - [results on connections](https://energy-models.github.io/datarecord/design/tools/pypsa/#results-on-connections)
    """
    n = solved

    p = _flat(PyPSA.to_datarecord(n, single_revision.record.con).attributes["p"])
    connections = unpacked(single_revision.record.dimensions["connection"])
    per_connection = p.groupby(["entity", "port"]).size()
    assert set(per_connection.index) == set(
        zip(connections["entity"], connections["port"])
    ), "every connection carries `p`, and nothing else does"
    assert (per_connection == len(n.snapshots)).all(), "one row per snapshot"

    lines = n.c["Line"]
    line = lines.static.index[0]
    for port in (0, 1):
        rows = p[(p["entity"] == line) & (p["port"] == port)].set_index("snapshot")
        expected = lines.dynamic[f"p{port}"][line]
        assert np.allclose(rows["value"].reindex(expected.index), expected), (
            f"`p{port}` lands on the line's bus{port} connection"
        )


def test_bus_results_are_keyed_by_bus(exported, solved):
    """`marginal_price`, `v_mag_pu` and `v_ang` are over `bus`; `Bus.p` is `p_bus`, off the connection-level `p`.

    `build` copies a result into the network like any attribute.

    Notes
    -----
    - [results on connections](https://energy-models.github.io/datarecord/design/tools/pypsa/#results-on-connections)
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    """
    revision = _results_over(exported.commit(), solved)
    schema = revision.record.schema
    over_bus = {
        a for a, spec in _registry_schema().attributes.items() if "bus" in spec.dims
    }
    assert over_bus - _input_names() == {
        "marginal_price",
        "v_mag_pu",
        "v_ang",
        "net_position",
        "p_bus",
    }
    assert not solved.c["Bus"].dynamic["p"].empty, "PyPSA did compute a `Bus.p`"
    assert "bus" not in schema.attributes["p"].dims
    assert "bus" in schema.attributes["p_bus"].dims

    expected = solved.c["Bus"].dynamic["marginal_price"]
    wide = _prices(revision)
    assert np.allclose(wide[expected.columns].reindex(expected.index), expected)

    built = PyPSA.build(revision.record).c["Bus"].dynamic["marginal_price"]
    pd.testing.assert_frame_equal(
        built[expected.columns],
        expected,
        check_names=False,
        check_freq=False,
        check_index_type=False,
    )


def test_a_child_reads_its_parents_results_and_the_latest_wins(exported, solved):
    """A result is an attribute: it inherits through layers, a later one standing over it.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    """
    revision = _results_over(exported.commit(), solved)
    expected = _prices(revision)
    pd.testing.assert_frame_equal(_prices(empty_child(revision).commit()), expected)

    n = PyPSA.build(revision.record)
    n.c["Bus"].dynamic["marginal_price"] *= 2
    resolved = _results_over(revision, n)
    pd.testing.assert_frame_equal(_prices(empty_child(resolved).commit()), expected * 2)


def test_a_re_solve_leaves_no_stale_sub_network(exported, solved):
    """A results layer replaces `base`'s SubNetworks rather than adding to them.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    """
    root = exported.commit()
    revision = _results_over(root, solved)
    *kept, dropped = sorted(solved.c["SubNetwork"].static.index)
    assert _sub_networks(revision) == sorted([*kept, dropped])
    entities = revision.record.dimensions["entity"].df()
    assert entities.equals(root.record.dimensions["entity"].df()), (
        "a results layer leaves the entity axis alone"
    )

    n = PyPSA.build(revision.record)
    n.remove("SubNetwork", dropped)
    resolved = _results_over(revision, n)
    assert _sub_networks(resolved) == sorted(kept)
    slack = resolved.record.attributes["slack_bus"].df()
    assert dropped not in set(slack["sub_network"]), "its results went with it"


@pytest.mark.parametrize("over_solved", [False, True], ids=["unsolved", "solved"])
def test_results_without_sub_networks_list_none(exported, solved, over_solved):
    """A solve without SubNetworks restates a `sub_network` axis `base` has empty, and writes none where it has none.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    """
    root = exported.commit()
    base = _results_over(root, solved) if over_solved else root
    n = PyPSA.build(base.record)
    n.remove("SubNetwork", n.c["SubNetwork"].static.index)
    n.c["Bus"].dynamic["marginal_price"] = pd.DataFrame(
        1.0, index=n.snapshots, columns=n.c["Bus"].static.index
    )
    revision = _results_over(base, n)
    record = revision.record
    assert _sub_networks_of(record.dimensions["sub_network"]) == []
    assert ("sub_network" in revision.manifest.files.dimensions) == over_solved, (
        "the axis is restated only where `base` has one"
    )
    assert "marginal_price" in record.attributes, "the results are written"
    slack = record.attributes.get("slack_bus")
    assert slack is None or slack.df().empty, "no stale SubNetwork keeps a result"


@pytest.mark.parametrize(
    ("ctype", "attr"), [("Line", "mu_upper"), ("Bus", "marginal_price")]
)
def test_a_result_a_re_solve_leaves_out_reads_as_null(exported, solved, ctype, attr):
    """A results layer restates `base`'s results it does not write as NULL.

    Built back, the network holds PyPSA's own default there.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    """
    root = exported.commit()
    n = PyPSA.build(_results_over(root, solved).record)
    n.c[ctype].dynamic[attr] = pd.DataFrame(
        1.0, index=n.snapshots, columns=n.c[ctype].static.index
    )
    base = _results_over(root, n)
    n = PyPSA.build(base.record)
    n.c[ctype].dynamic[attr] = n.c[ctype].dynamic[attr].iloc[:, :0]
    resolved = _results_over(base, n)

    def read(revision, name):
        df = revision.record.attributes[name].df()
        by = [c for c in df.columns if c not in ("breakpoint", "value")]
        return df.sort_values(by).reset_index(drop=True)

    stale = read(base, attr)
    keys = [c for c in stale.columns if c not in ("breakpoint", "value")]
    values = stale[keys].merge(read(resolved, attr), on=keys)["value"]
    assert (stale["value"] == 1.0).all()
    assert len(values) == len(stale)
    assert values.isna().all(), "the stale value is gone"
    c = PyPSA.build(resolved.record).c[ctype]
    assert (c.dynamic[attr] == c.defaults.at[attr, "default"]).all().all(), (
        "PyPSA fills its own default"
    )
    pd.testing.assert_frame_equal(read(resolved, "p_nom"), read(base, "p_nom"))


def test_a_series_a_re_solve_leaves_out_reads_as_null(results, results_built):
    """A bus's series `base` holds and a re-solve writing the others leaves out is NULL at every snapshot, while one `base` holds over no snapshot takes the re-solve's series.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/reference/tools/#writing-results)
    """
    from datarecord.mutable import WorkingRecord

    n = results_built.copy()
    prices = n.c.Bus.dynamic["marginal_price"] + 1.0
    dropped, static = prices.columns[:2]
    w = WorkingRecord(results.record)
    w.set("marginal_price", 3.0, bus=static)
    base = results.store.commit(w)
    kept = prices.drop(columns=dropped)
    n.c.Bus.dynamic["marginal_price"] = kept
    got = _prices(_results_over(base, n))
    assert got[dropped].isna().all(), "the stale series is gone"
    assert got[got.index.isna()].isna().all().all(), "the static row is NULL"
    pd.testing.assert_frame_equal(
        got.reindex(index=kept.index, columns=kept.columns),
        kept,
        check_names=False,
        check_freq=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param([10.0, 20.0], [5.0, 5.0], id="to-shared"),
        pytest.param([5.0, 5.0], [10.0, 20.0], id="to-each"),
    ],
)
def test_a_re_solve_writes_a_result_shared_across_scenarios_over_each(
    store, first, second
):
    """A re-solve's result equal at every scenario, written once over `base`'s per-scenario rows, reaches each scenario, and the reverse.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/reference/tools/#writing-results)
    """
    n = _north()
    n.add("GlobalConstraint", "co2", sense="<=", constant=100.0)
    n.set_scenarios(["a", "b"])
    revision = export_network(n, store)
    for mu in (first, second):
        n = PyPSA.build(revision.record)
        n.c.GlobalConstraint.static["mu"] = mu
        revision = _results_over(revision, n)
    built = PyPSA.build(revision.record)
    assert built.c.GlobalConstraint.static["mu"].tolist() == second


def _stored_mu(record):
    """The `mu` rows of `record` by `global_constraint`, NULL as NaN, as `build` reads it as 0."""
    rows = record.attributes["mu"].df()
    return rows.set_index("global_constraint")["value"]


def test_a_results_only_export_without_a_constraint_leaves_its_result_null(store):
    """A re-solve lacking one of `base`'s constraints leaves that constraint's `mu` NULL, and the other's as solved."""
    n = _north()
    n.add("GlobalConstraint", ["co2", "gas"], sense="<=", constant=100.0)
    n.c.GlobalConstraint.static["mu"] = [5.0, 6.0]
    revision = export_network(n, store)
    solved = PyPSA.build(revision.record)
    solved.remove("GlobalConstraint", "gas")
    solved.c.GlobalConstraint.static["mu"] = [7.0]
    mu = _stored_mu(_results_over(revision, solved).record)
    assert mu["co2"] == 7.0
    assert pd.isna(mu["gas"])


def test_a_results_only_export_adding_a_constraint_is_refused(store):
    """A network holding a constraint `base` lacks is refused at the call."""
    n = _north()
    n.add("GlobalConstraint", "co2", sense="<=", constant=100.0)
    revision = export_network(n, store)
    grown = PyPSA.build(revision.record)
    grown.add("GlobalConstraint", "extra", sense="<=", constant=1.0)
    with pytest.raises(
        ValueError, match="global_constraint labels `base` lacks: \\['extra'\\]"
    ):
        _results_over(revision, grown)


@pytest.mark.slow
@pytest.mark.parametrize(
    "select",
    [
        pytest.param({"bus": ["London", "Norwich"]}, id="bus"),
        pytest.param({"entity": ["Manchester Wind", "0"]}, id="entity"),
    ],
)
def test_a_re_solve_of_a_selection_restates_only_what_it_holds(
    exported, solved, select
):
    """A selection's results replace `base`'s at the labels it holds, and leave every other label's, and the SubNetworks it numbers over itself alone, to `base`."""
    base = _results_over(exported.commit(), solved)
    n = PyPSA.build(base.record.select(**select))
    n.determine_network_topology()
    n.c.Bus.dynamic["marginal_price"] += 1.0
    resolved = _results_over(base, n, **select)

    before, after = (PyPSA.build(r.record) for r in (base, resolved))
    topology = (r.c.SubNetwork.static.drop(columns="obj") for r in (after, before))
    pd.testing.assert_frame_equal(*topology)
    pd.testing.assert_series_equal(
        after.c.Line.static["sub_network"], before.c.Line.static["sub_network"]
    )
    for ctype, attr in (
        ("Bus", "marginal_price"),
        ("Generator", "p"),
        ("Line", "p0"),
    ):
        got, was = after.c[ctype].dynamic[attr], before.c[ctype].dynamic[attr]
        held = n.c[ctype].dynamic[attr]
        inside = pd.DataFrame(False, index=got.index, columns=got.columns)
        inside.loc[held.index, held.columns] = True
        pd.testing.assert_frame_equal(got.mask(inside), was.mask(inside), obj=attr)
        pd.testing.assert_frame_equal(
            got.loc[held.index, held.columns],
            held,
            check_names=False,
            check_freq=False,
            obj=attr,
        )


@pytest.mark.slow
def test_a_re_solve_of_some_snapshots_keeps_bases_series_elsewhere(exported, solved):
    """Over a single-period `base`, a re-solve of some snapshots takes them and leaves `base`'s dispatch at the rest, while one of snapshots `base` lacks is refused at the call.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/reference/tools/#writing-results)
    """
    base = _results_over(exported.commit(), solved)
    n = PyPSA.build(base.record.select(snapshot=list(solved.snapshots[:2])))
    n.c.Generator.dynamic["p"] += 100.0
    built = PyPSA.build(_results_over(base, n).record).c.Generator.dynamic["p"]
    expected = solved.c.Generator.dynamic["p"].copy()
    expected.loc[n.snapshots] = n.c.Generator.dynamic["p"]
    pd.testing.assert_frame_equal(
        built[expected.columns], expected, check_names=False, check_freq=False
    )
    n.set_snapshots(n.snapshots + pd.DateOffset(years=100))
    with pytest.raises(ValueError, match="snapshot labels `base` lacks"):
        PyPSA.to_datarecord(n, base.record.con, results_only=True, base=base.record)


@pytest.mark.parametrize(
    ("labels", "over", "match"),
    [
        pytest.param(
            {"scenario": "hi"},
            "record",
            "over the selection it was built from",
            id="deterministic",
        ),
        pytest.param(
            {"scenario": "hi"}, "selection", "first-stage results", id="scenario"
        ),
        pytest.param(
            {"scenario": ["hi"]}, "selection", "first-stage results", id="scenario-list"
        ),
    ],
)
@pytest.mark.slow
def test_results_of_some_labels_reaching_every_label_are_refused(
    stochastic, labels, over, match
):
    """Results of a network built at some labels that would stand for every label are refused.

    A deterministic network's are NULL on `scenario`, a subset of scenarios
    decides its `p_nom_opt` for itself alone, though it is one value over
    every scenario.

    Notes
    -----
    - [writing results](https://energy-models.github.io/datarecord/reference/tools/#writing-results)
    """
    view = stochastic.record.select(**labels)
    n = PyPSA.build(view)
    written = view if over == "selection" else stochastic.record
    with pytest.raises(ValueError, match=match):
        PyPSA.to_datarecord(n, view.con, results_only=True, base=written)


def test_results_only_takes_base(single_revision, solved):
    """A results layer is written over the record it was built from."""
    with pytest.raises(ValueError, match="takes `base` with `results_only=True`"):
        PyPSA.to_datarecord(solved, single_revision.record.con, results_only=True)


def test_a_full_export_is_written_over_a_record_not_a_selection(single_revision, ac_dc):
    """A selection's rows would miss the labels it squeezes, so a full export over one is refused at the call."""
    record = single_revision.record
    view = record.select(
        snapshot=record.dimensions["snapshot"].df()["snapshot"].tolist()[:2]
    )
    with pytest.raises(ValueError, match="not over a selection"):
        PyPSA.to_datarecord(ac_dc, record.con, base=view)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("attr", "value"), [("p_nom_extendable", False), ("marginal_cost", 0.0)]
)
def test_a_full_export_over_base_writes_a_return_to_the_default(exported, attr, value):
    """A value back at PyPSA's default is written over a `base` holding another, which would show through."""
    revision = exported.commit()
    n = PyPSA.build(revision.record)
    assert n.c.Generator.static.at["Manchester Wind", attr] != value
    n.c.Generator.static.at["Manchester Wind", attr] = value
    layer = PyPSA.to_datarecord(n, revision.record.con, base=revision.record)
    child = PyPSA.build(revision.store.commit(layer, parent=revision).record)
    assert child.c.Generator.static.at["Manchester Wind", attr] == value


@pytest.mark.slow
@pytest.mark.parametrize("attr", ["p_nom_extendable", "marginal_cost", "p_nom_min"])
@pytest.mark.parametrize(
    "entity", ["Manchester Wind", "Newcomer"], ids=["unchanged", "new-at-default"]
)
def test_a_full_export_over_base_leaves_out_what_it_repeats(exported, attr, entity):
    """A static value equal to what `base` resolves, its row or the default, is not written."""
    revision = exported.commit()
    n = PyPSA.build(revision.record)
    n.add("Generator", "Newcomer", bus="Manchester")
    layer = PyPSA.to_datarecord(n, revision.record.con, base=revision.record)
    rows = layer.attributes[attr].df()
    assert entity not in set(rows["entity"]), "the layer stays thin"
    child = PyPSA.build(revision.store.commit(layer, parent=revision).record)
    expected = n.c.Generator.static.at[entity, attr]
    assert child.c.Generator.static.at[entity, attr] == expected


def _unedited(n) -> None:
    pass


def _nudged(n) -> None:
    n.c.Generator.dynamic["p_max_pu"].iloc[3, 1] += 0.1


def _one_snapshot_later(n) -> None:
    later = pd.DatetimeIndex([n.snapshots[-1] + pd.Timedelta("1h")])
    n.set_snapshots(n.snapshots.append(later))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scenarios", "edit", "written"),
    [
        pytest.param(None, _unedited, slice(0), id="nan-repeats-null"),
        pytest.param(None, _one_snapshot_later, slice(None), id="snapshot-new"),
        pytest.param(None, _nudged, slice(1, 2), id="nudged"),
        pytest.param(["lo", "hi"], _nudged, slice(1, 2), id="nudged-scenario"),
    ],
)
def test_a_full_export_over_base_compares_a_series_at_each_snapshot(
    store, ac_dc, scenarios, edit, written
):
    """Over `base`, a series NaN where `base`'s is NULL repeats it, and one differing at a snapshot, or at one `base` lacks, is written whole for its entity and scenario.

    A port value or custom value repeating `base`'s is left out, and so is
    a weighting where the snapshots stay.
    """
    n = ac_dc.copy()
    n.c.Generator.static["subsidy"] = 2.0
    if scenarios is not None:
        n.set_scenarios(scenarios)
    n.c.Generator.dynamic["p_max_pu"].iloc[0, 0] = np.nan
    revision = export_network(n, store)
    n = PyPSA.build(revision.record)
    edit(n)
    wide = n.c.Generator.dynamic["p_max_pu"]
    layer = PyPSA.to_datarecord(n, revision.record.con, base=revision.record)
    rows = layer.attributes["p_max_pu"].df()
    keys = ["scenario", "entity"] if scenarios is not None else ["entity"]
    assert set(rows.set_index(keys).index) == set(wide.columns[written])
    assert len(rows) == len(wide.columns[written]) * len(n.snapshots), (
        "a differing series is written whole"
    )
    repeated = ["efficiency", "subsidy"]
    if edit is not _one_snapshot_later:
        repeated.append("snapshot_weighting_objective")
    written_again = [a for a in repeated if not layer.attributes[a].df().empty]
    assert written_again == [], "a repeated port, custom or weighting value is left out"
    child = PyPSA.build(revision.store.commit(layer, parent=revision).record)
    pd.testing.assert_frame_equal(
        child.c.Generator.dynamic["p_max_pu"][wide.columns], wide, check_freq=False
    )


@pytest.mark.parametrize(
    ("held", "edited", "written"),
    [
        pytest.param([5.0, 0.0], [5.0, 5.0], True, id="held-at-one"),
        pytest.param([5.0, 5.0], [5.0, 5.0], False, id="held-at-none"),
        pytest.param([5.0, 0.0], [0.0, 0.0], True, id="to-the-default"),
    ],
)
def test_a_full_export_over_base_writes_a_value_shared_across_scenarios(
    store, held, edited, written
):
    """A value equal at every scenario repeats `base` only where `base` resolves it at each, not where it holds it at one and the default at another."""
    n = _north()
    n.set_scenarios(["a", "b"])
    n.c.Generator.static["p_nom"] = held
    revision = export_network(n, store)
    n = PyPSA.build(revision.record)
    n.c.Generator.static["p_nom"] = edited
    layer = PyPSA.to_datarecord(n, revision.record.con, base=revision.record)
    assert layer.attributes["p_nom"].df().empty != written
    child = PyPSA.build(revision.store.commit(layer, parent=revision).record)
    assert child.c.Generator.static["p_nom"].tolist() == edited


@pytest.mark.parametrize("results_only", [False, True])
def test_a_network_is_not_written_over_a_record_with_a_dim_of_its_own(
    swept, results_only
):
    """A network over the whole of a record with a dim of its own would reach every label, refused at the call."""
    with pytest.raises(ValueError, match=r"with a dim of its own, \['weather_year'\]"):
        PyPSA.to_datarecord(
            _north(), swept.record.con, results_only=results_only, base=swept.record
        )


def test_link_internal_power_is_p_internal(single_revision, solved):
    """`Link.p` is power at no bus, an entity-level `p_internal`, no `bus`."""
    n = solved

    results = PyPSA.to_datarecord(n, single_revision.record.con).attributes
    internal = results["p_internal"].df()
    links = list(n.c["Link"].static.index)
    assert "bus" not in internal.columns
    assert set(internal["entity"]) == set(links)
    assert len(internal) == len(n.snapshots) * len(links)
    assert "p_internal" in _registry_schema().attributes


def _ctypes_written(revision) -> pd.Series:
    return revision.record.properties["entity"].df().set_index("entity")["ctype"]


def _add_geometry(n, names, **attrs) -> None:
    """`n.add("Shape", names, **attrs)`, though PyPSA warns that a shape beside others lacks the CRS."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CRS not set")
        n.add("Shape", names, **attrs)


def _shaped():
    """`_north` and a bus `south`, each with onshore shapes, `country` shapes beside and a generator's site.

    `north` also has an offshore shape, and `DE` one, a custom `Bus.country`
    names each bus's country.
    """
    import shapely

    n = _north()
    n.to_crs("EPSG:3035")
    n.add("Bus", "south")
    n.c.Bus.static["country"] = ["DE", "FR"]
    _add_geometry(
        n,
        ["north onshore", "north offshore", "south onshore"],
        geometry=[
            shapely.box(0, 0, 1, 1),
            shapely.Point(2, 3),
            shapely.box(1, 1, 2, 2),
        ],
        idx=["north", "north", "south"],
        component="Bus",
        type=["onshore", "offshore", "onshore"],
    )
    _add_geometry(
        n,
        ["DE country", "FR country", "DE offshore"],
        geometry=[
            shapely.box(0, 0, 3, 3),
            shapely.box(3, 3, 4, 4),
            shapely.Point(5, 5),
        ],
        idx=["DE", "FR", "DE"],
        component="country",
        type=["country", "country", "offshore"],
    )
    _add_geometry(
        n,
        ["gen site"],
        geometry=[shapely.Point(0.5, 0.5)],
        idx="gen",
        component="Generator",
        type="site",
    )
    return n


def _add_shape(n, **attrs) -> None:
    """Add the shape `extra`, a point, with `attrs`."""
    import shapely

    _add_geometry(n, ["extra"], geometry=[shapely.Point(0, 0)], **attrs)


def _with_bus_column(n, name: str, value) -> None:
    """Set the custom column `name` of every bus to `value`."""
    n.c.Bus.static[name] = value


def _shapes(n) -> pd.DataFrame:
    """`n.shapes` keyed by the label each outlines and its `type`, not by name."""
    return n.c.Shape.static.set_index(["component", "idx", "type"]).sort_index()


def _three_port_link(bus1="b"):
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=2, freq="h"))
    for bus in ("a", "b", "c"):
        n.add("Bus", bus)
    n.add("Link", "split", bus0="a", bus1=bus1, bus2="c")
    return n


def test_a_third_port_result_lands_on_its_connection(workspace):
    """`p2` is `p` on the bus2 connection, declared though the bare registry has no `p2`."""
    n = _three_port_link()
    n.c["Link"].dynamic["p2"] = pd.DataFrame({"split": [1.0, 2.0]}, index=n.snapshots)

    results = PyPSA.to_datarecord(n, workspace.con).attributes
    assert "p2" not in results
    assert "p" in _registry_schema().attributes, "the bare registry declares `p`"
    p = _flat(results["p"])
    assert p["port"].to_list() == [2, 2], "a third port's flow is on bus2"
    assert p["value"].to_list() == [1.0, 2.0]


@pytest.mark.parametrize("bus1", ["b", "a"], ids=["own-buses", "one-bus"])
def test_a_three_port_record_writes_p_on_each_connection(store, bus1):
    """Writing a three-port network puts every port's flow on its connection."""
    n = _three_port_link(bus1)
    flows = {"p0": [3.0, 4.0], "p1": [-1.0, -2.0], "p2": [-1.5, -1.5]}
    for attr, values in flows.items():
        n.c["Link"].dynamic[attr] = pd.DataFrame({"split": values}, index=n.snapshots)

    revision = export_network(n, store)

    assert revision.record.dimensions["port"].df()["port"].tolist() == [0, 1, 2], (
        "the registry's ports 0 and 1, and the link's port 2"
    )
    p = _flat(relation(revision, "p")).sort_values(["port", "snapshot"])
    assert p.groupby("port")["value"].apply(list).to_dict() == {
        0: flows["p0"],
        1: flows["p1"],
        2: flows["p2"],
    }


@pytest.mark.parametrize("stochastic", [False, True], ids=["one", "scenarios"])
@pytest.mark.parametrize("bus1", ["north", "mid"], ids=["one-bus", "own-buses"])
def test_ports_at_one_bus_survive_a_round_trip(store, bus1, stochastic):
    """Two ports of a Link at one bus are two connections, each with its own efficiency."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", ["north", "mid", "south"])
    n.add(
        "Link",
        "heat_pump",
        bus0="north",
        bus1=bus1,
        bus2="south",
        efficiency=3.0,
        efficiency2=-1.0,
        p_nom=10.0,
    )
    if stochastic:
        n.set_scenarios({"low": 0.5, "high": 0.5})
        n.c["Link"].static.loc[("high", "heat_pump"), "efficiency"] = 2.0

    built = PyPSA.build(export_network(n, store).record)
    columns = ["bus0", "bus1", "bus2", "efficiency", "efficiency2"]
    links = built.c["Link"].static[columns].to_dict("index")
    assert links == n.c["Link"].static[columns].to_dict("index"), (
        "each port keeps its bus and its efficiency"
    )
    key = ("high", "heat_pump") if stochastic else "heat_pump"
    assert links[key]["efficiency"] == (2.0 if stochastic else 3.0)
    if not stochastic:
        assert_networks_equal(built, n)


def test_network_meta_keeps_non_finite_floats(store):
    """`n.meta` round-trips `inf`, `-inf` and `nan` as the bare words, not as null."""
    n = _north()
    n.meta = {"cap": np.inf, "floor": -np.inf, "unknown": np.nan, "none": None}
    revision = export_network(n, store)
    raw = Path(store.layer_dir(revision.id), "manifest.json").read_text()
    assert '"cap":Infinity' in raw, "the manifest keeps the bare word"
    fresh = Workspace()
    try:
        for reader in (store, fresh.store(store.uri)):
            built = PyPSA.build(reader.revision(revision.id).record)
            assert repr(built.meta) == repr(n.meta)
    finally:
        fresh.close()


def test_network_meta_year_keys_come_back_as_strings(store):
    """An int key, as PyPSA-Eur's planning horizons are, reads back as the string PyPSA's own files hold."""
    n = _north()
    n.meta = {"co2_budget": {2030: 0.4, 2040: 0.1}}
    revision = export_network(n, store)
    built = PyPSA.build(store.revision(revision.id).record)
    assert built.meta == {"co2_budget": {"2030": 0.4, "2040": 0.1}}


def test_network_meta_that_is_no_json_is_refused_with_its_path(store):
    """A tuple is refused before anything is written."""
    n = _north()
    n.meta = {"co2_budget": (0.4, 0.1)}
    with pytest.raises(
        TypeError, match=r"meta\['meta'\]\['co2_budget'\] is \(0\.4, 0\.1\)"
    ):
        export_network(n, store)
    assert store.revisions() == []


def test_a_stochastic_third_port_efficiency_survives_a_round_trip(store):
    """Per-scenario port values are placed by connection like any other.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
    """
    n = _three_port_link()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c["Link"].static["efficiency"] = 0.5
    n.c["Link"].static.loc[("low", "split"), "efficiency2"] = 0.2
    n.c["Link"].static.loc[("high", "split"), "efficiency2"] = 0.4

    built = PyPSA.build(export_network(n, store).record)

    static = built.c["Link"].static
    assert static[["efficiency", "efficiency2"]].to_dict("index") == {
        ("low", "split"): {"efficiency": 0.5, "efficiency2": 0.2},
        ("high", "split"): {"efficiency": 0.5, "efficiency2": 0.4},
    }, "each scenario's port values land in their own columns"


def test_a_static_attribute_differing_per_scenario_survives_a_round_trip(store):
    """A fixed `p_nom` is an input and may differ per scenario, as PyPSA allows.

    Notes
    -----
    - [entities and their properties](https://energy-models.github.io/datarecord/design/tools/pypsa/#entities-and-their-properties)
    """
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "g", bus="b", p_nom=10.0, marginal_cost=1.0)
    n.add("Load", "l", bus="b", p_set=5.0)
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.generators.static.loc[("high", "g"), "p_nom"] = 20.0
    n.consistency_check(strict=["scenario_invariant_attrs"])

    static = PyPSA.build(export_network(n, store).record).c.generators.static
    assert static[["p_nom", "marginal_cost"]].to_dict("index") == {
        ("low", "g"): {"p_nom": 10.0, "marginal_cost": 1.0},
        ("high", "g"): {"p_nom": 20.0, "marginal_cost": 1.0},
    }, "p_nom keeps its scenario, a shared marginal_cost reaches both"


@pytest.mark.parametrize(
    ("attr", "per_scenario"),
    [
        ("p_nom", True),
        ("p_nom_extendable", False),
        ("p_nom_opt", False),
    ],
)
def test_only_pypsas_scenario_invariant_attributes_lack_the_scenario_dim(
    attr, per_scenario
):
    """Every stored attribute may differ per scenario, but those PyPSA keeps invariant."""
    dims = _registry_schema().attributes[attr].dims
    assert ("scenario" in dims) == per_scenario, dims


def _three_port_process():
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=2, freq="h"))
    for bus in ("a", "b", "c"):
        n.add("Bus", bus)
    n.add("Process", "dri", bus0="a", bus1="b", bus2="c")
    return n


def test_process_rate_is_one_attribute_on_its_connections(workspace):
    """`rate0`, `rate1`, `rate2` are one `rate`, a row per connection.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
    """
    n = _three_port_process()
    n.c["Process"].static.loc["dri", ["rate0", "rate1", "rate2"]] = [-2.0, 1.5, 0.3]

    inputs = PyPSA.to_datarecord(n, workspace.con).attributes
    assert not {"rate0", "rate1", "rate2"} & set(inputs), "no positional names"
    rate = _flat(inputs["rate"])
    assert rate.set_index("port")["value"].to_dict() == {0: -2.0, 1: 1.5, 2: 0.3}
    spec = _registry_schema().attributes["rate"]
    assert "connection" in spec.dims
    assert spec.default is None, "rate0 and rate1 default differently"


@pytest.mark.parametrize(
    ("network", "ctype", "name", "columns", "values"),
    [
        (
            _three_port_link,
            "Link",
            "split",
            ["efficiency", "efficiency2"],
            [0.5, 0.3],
        ),
        (
            _three_port_process,
            "Process",
            "dri",
            ["rate0", "rate1", "rate2"],
            [-2.0, 1.5, 0.3],
        ),
    ],
    ids=["link-efficiency", "process-rate"],
)
def test_per_port_values_survive_a_round_trip(
    store, network, ctype, name, columns, values
):
    """Each port's value comes back from its connection, not as PyPSA's default.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
    """
    n = network()
    n.c[ctype].static.loc[name, columns] = values

    built = PyPSA.build(export_network(n, store).record)

    assert built.c[ctype].static.loc[name, columns].tolist() == values, (
        "each port's value lands in its own column"
    )
    assert_networks_equal(built, n)


@pytest.mark.parametrize(
    ("network", "ctype", "name", "static", "series"),
    [
        (
            _three_port_link,
            "Link",
            "split",
            ("efficiency", 0.5),
            ("efficiency2", [0.2, 0.4]),
        ),
        (_three_port_process, "Process", "dri", ("rate0", -2.0), ("rate1", [0.8, 1.2])),
    ],
    ids=["link-efficiency", "process-rate"],
)
def test_a_time_varying_port_survives_a_round_trip(
    store, network, ctype, name, static, series
):
    """A series on one port is that port's series, beside another port's static value.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
    """
    (static_attr, static_value), (series_attr, values) = static, series
    n = network()
    n.c[ctype].static.loc[name, static_attr] = static_value
    n.c[ctype].dynamic[series_attr] = pd.DataFrame(
        {name: values}, index=n.snapshots
    ).rename_axis(columns="name")

    built = PyPSA.build(export_network(n, store).record)

    assert built.c[ctype].dynamic[static_attr].empty, "the static port stays static"
    assert built.c[ctype].static.at[name, static_attr] == static_value
    assert built.c[ctype].dynamic[series_attr][name].to_list() == values
    assert_networks_equal(built, n)


def test_per_type_defaults_round_trip_with_nothing_written(store):
    """A Load and a StorageUnit added without `sign` or `p_min_pu` build with their own defaults.

    A default PyPSA differs per type is declared through `default_by`.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)
    """
    from datarecord.mutable import WorkingRecord

    import pypsa

    sign = _registry_schema().properties["entity"]["sign"]
    assert sign.default_by["entity.ctype"]["Load"] == -1.0, (
        "PyPSA declares a load's sign through `default_by`"
    )

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    root = export_network(n, store)

    staged = WorkingRecord(root.record)
    staged.add("entity", ["l1"], ctype="Load")
    staged.add("entity", ["su1"], ctype="StorageUnit")
    staged.add(
        "connection",
        [{"entity": e, "port": 0} for e in ("l1", "su1")],
        bus="b",
    )
    child = root.store.commit(staged)

    inputs = Path(store.layer_dir(child.id)) / "attributes"
    written = {p.stem for p in inputs.glob("*.parquet")} if inputs.exists() else set()
    assert not {"sign", "p_min_pu"} & written, "a default is never stored"

    built = PyPSA.build(child.record)
    assert built.c["Load"].static.at["l1", "sign"] == -1, "a Load's own sign"
    assert built.c["StorageUnit"].static.at["su1", "sign"] == 1
    assert built.c["StorageUnit"].static.at["su1", "p_min_pu"] == -1, (
        "a StorageUnit's own p_min_pu"
    )


def test_delays_are_rows_keyed_by_their_connections(workspace, store):
    """A Process's `delay0`/`delay1` and a Link's `delay`/`delay2` are one `delay` per port.

    A Link leaves `delay` unsuffixed at bus1, so no suffix marks it per-port.
    Never varying, it is still an attribute: long rows keyed by the connection,
    where `bus` is the connection's one property.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
    """
    n = _three_port_process()
    n.c["Process"].static.loc["dri", ["delay0", "delay1"]] = [1, 2]
    n.c["Process"].static.loc["dri", "cyclic_delay1"] = True
    n.add("Link", "split", bus0="a", bus1="b", bus2="c", delay=3, delay2=4)

    record = PyPSA.to_datarecord(n, workspace.con)
    members = set(record.dimensions["entity"].columns)
    assert not {"delay", "delay0", "delay2", "cyclic_delay1"} & members, (
        "no positional names on the members"
    )
    assert set(record.properties["connection"].columns) == {"connection", "bus"}, (
        "a connection row carries its key and its property"
    )
    rows = unpacked(record.attributes["delay"])
    delay = rows.set_index(["entity", "port"])["value"].dropna().to_dict()
    assert delay == {
        ("dri", 0): 1,
        ("dri", 1): 2,
        ("dri", 2): 0,
        ("split", 1): 3,
        ("split", 2): 4,
    }, "bus0 of a Link has no delay"
    assert _registry_schema().attributes["cyclic_delay"].dims == {
        "connection",
        "scenario",
    }

    built = PyPSA.build(export_network(n, store).record)
    assert built.c["Link"].static.loc["split", ["delay", "delay2"]].tolist() == [3, 4]
    assert built.c["Process"].static.loc["dri", "cyclic_delay1"]
    assert_networks_equal(built, n)


def test_a_nan_in_a_series_survives_the_melt(workspace):
    """`UNPIVOT` drops nulls unless told otherwise; a NaN cell is still a row."""
    import numpy as np

    n = _three_port_link()
    n.c["Link"].dynamic["p_max_pu"] = pd.DataFrame(
        {"split": [1.0, np.nan]}, index=n.snapshots
    )

    rows = PyPSA.to_datarecord(n, workspace.con).attributes["p_max_pu"]
    split = rows.filter("entity = 'split'")
    assert len(split) == 2, "one row per snapshot, NaN included"


def test_a_name_is_never_an_identifier_between_long_and_wide(store):
    """Each name is stored and comes back as its own series.

    `wind "north"` would end a quoted identifier, `snapshot` names a key
    column, and `Gen` and `gen` one case-insensitive identifier.

    Notes
    -----
    - [wide and long rows](https://energy-models.github.io/datarecord/design/model/attributes/#wide-and-long-rows)
    """
    import pypsa

    names = ['wind "north"', "snapshot", "Gen", "gen"]
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "north")
    n.add("Generator", names, bus="north")
    series = pd.DataFrame(
        {name: [0.1 * i, 0.1 * i + 0.05] for i, name in enumerate(names, 1)},
        index=n.snapshots,
    )
    n.c["Generator"].dynamic["p_max_pu"] = series

    revision = store.commit(PyPSA.to_datarecord(n, store.workspace.con), parent=None)
    rows = revision.record.attributes["p_max_pu"]
    assert sorted(rows.project("entity, value").fetchall()) == sorted(
        (name, value) for name in names for value in series[name]
    ), "the melt stores each name's own series"
    built = PyPSA.build(revision.record).c["Generator"].dynamic["p_max_pu"]
    pd.testing.assert_frame_equal(
        built[names],
        series,
        check_names=False,
        check_freq=False,
        check_column_type=False,
    )


def test_a_second_tool_needs_no_record_change():
    """A tool is a plain object taking a record; adding one changes nothing here.

    No registration and no name dispatch: conformance to `Tool` is structural,
    so a second framework's module defines its own singleton and callers import
    it.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
    """
    from datarecord.tools.base import Tool

    class FakeTool:
        name = "fake"
        schema = Schema()

        def requires(self, record):
            return Requirements(dimensions=frozenset({"snapshot"}))

        def verify(self, record):
            return Requirements(categories=frozenset({"Nope"}))

        def build(self, record):
            return "fake-model"

        def to_datarecord(self, model, con, *, results_only=False, base=None):
            return f"layer-of-{model}"

    assert isinstance(FakeTool(), Tool), "conformance to `Tool` is structural"


def test_schema_dims_stay_generic(exported):
    """`Coords` carries a dim PyPSA knows nothing about, and axis names live in the tool, which reports it as a dim of the record's own.

    Declared but not `partial`, so the record's files need no column for it.
    """
    revision = _with_schema(exported, dims={**_DIMS, "vintage": "str"})
    record = revision.record
    assert "vintage" in record.schema.dimensions
    assert record.dimensions["vintage"].fetchall() == [], "no axis rows anywhere"
    missing = PyPSA.verify(revision.record)
    assert missing.unsupported_dimensions == {"vintage"}, (
        "a collection's, not a network's"
    )
    assert not missing.dimensions, "PyPSA's own dims are satisfied"
    assert "vintage" in record.schema.long_columns


def test_a_null_scenario_weight_reaches_every_scenario_its_own_row_does_not():
    rows = pd.DataFrame({"scenario": [None, "high"], "value": [0.25, 0.5]})
    con = duckdb.connect()
    index = pd.Index(["low", "mid", "high"], name="scenario")
    record = SimpleNamespace(
        schema=_registry_schema(),
        dimensions={"scenario": con.from_df(index.to_frame(index=False))},
        attributes={"scenario_weight": con.from_df(rows)},
    )
    equal = pd.DataFrame({"weight": 1 / 3}, index=index)
    weights = _weightings(record, "scenario", equal)
    assert weights["weight"].to_dict() == {"low": 0.25, "mid": 0.25, "high": 0.5}, (
        "a NULL-scenario row broadcasts; a scenario's own row wins"
    )


def test_snapshot_weightings_survive_a_round_trip(store, ac_dc):
    """Each column of `snapshot_weightings` is a `snapshot_weighting_*` attribute."""
    n = ac_dc.copy()
    steps = np.arange(len(n.snapshots), dtype=float)
    n.snapshot_weightings = pd.DataFrame(
        {"objective": 2.0 + steps, "stores": 3.0, "generators": 0.5 * steps},
        index=n.snapshots,
    )

    built = PyPSA.build(export_network(n, store).record)
    pd.testing.assert_frame_equal(
        built.snapshot_weightings,
        n.snapshot_weightings,
        check_freq=False,
        check_index_type=False,
    )


@pytest.fixture(scope="module", params=["repeated", "distinct"])
def layout(request) -> str:
    """How `multiperiod` lays its timesteps over the periods."""
    return request.param


def _over_periods(ac_dc, layout: str):
    """`ac_dc` over two investment periods, with weightings of its own in every column of both axes.

    `repeated` runs every timestep in both periods, as `set_investment_periods`
    does, `distinct` splits the timesteps between them.
    """
    n = ac_dc.copy()
    if layout == "repeated":
        n.set_investment_periods([2030, 2040])
    else:
        periods = np.repeat([2030, 2040], len(n.snapshots) // 2)
        n.set_snapshots(
            pd.MultiIndex.from_arrays(
                [periods, n.snapshots], names=["period", "timestep"]
            )
        )
    n.investment_period_weightings = pd.DataFrame(
        {"objective": [1.0, 0.6], "years": [10.0, 5.0]}, index=n.investment_periods
    )
    steps = np.arange(len(n.snapshots), dtype=float)
    n.snapshot_weightings = pd.DataFrame(
        {"objective": 2.0 + steps, "stores": 3.0, "generators": 0.5 * steps},
        index=n.snapshots,
    )
    return n


@pytest.fixture(scope="module")
def multiperiod(ac_dc, layout):
    return _over_periods(ac_dc, layout)


@pytest.fixture(scope="module")
def multiperiod_revision(tmp_path_factory, multiperiod):
    """`multiperiod` exported once for the module."""
    workspace = Workspace()
    yield export_network(
        multiperiod, workspace.store(str(tmp_path_factory.mktemp("multiperiod")))
    )
    workspace.close()


@pytest.fixture(scope="module")
def multiperiod_built(multiperiod_revision):
    """The network `multiperiod_revision` builds."""
    return PyPSA.build(multiperiod_revision.record)


def test_a_multi_period_snapshot_is_a_multi_dim(multiperiod, multiperiod_revision):
    """A multi-period snapshot is a `(period, timestep)` tuple, its timesteps repeating or not.

    Notes
    -----
    - [multi-dims](https://energy-models.github.io/datarecord/reference/data-model/#multi-dims)
    - [snapshots and weightings](https://energy-models.github.io/datarecord/reference/tools/#snapshots-and-weightings)
    """
    schema = multiperiod_revision.record.schema
    timesteps = multiperiod.snapshots.get_level_values("timestep")
    labels = multiperiod_revision.record.dimensions["snapshot"].df()["snapshot"]
    assert [(s["period"], s["timestep"]) for s in labels] == list(
        multiperiod.snapshots
    ), "PyPSA's index, in its order"
    assert schema.over("snapshot") == {"period": "period", "timestep": "timestep"}
    assert "snapshot" not in schema.properties
    steps = multiperiod_revision.record.dimensions["timestep"].df()["timestep"]
    assert steps.tolist() == list(timesteps.unique()), "each timestep once"
    assert schema.attributes["p_max_pu"].dims == {
        "entity",
        "snapshot",
        "scenario",
    }, "dims as in a single-period network"


def test_multi_period_weightings_survive_a_round_trip(multiperiod, multiperiod_built):
    pd.testing.assert_frame_equal(
        multiperiod_built.snapshot_weightings,
        multiperiod.snapshot_weightings,
        check_index_type=False,
    )
    pd.testing.assert_frame_equal(
        multiperiod_built.investment_period_weightings,
        multiperiod.investment_period_weightings,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(lambda n: pd.DataFrame(index=n.snapshots), id="snapshots"),
        pytest.param(lambda n: pd.DataFrame(index=n.investment_periods), id="periods"),
        pytest.param(lambda n: n.snapshot_weightings, id="snapshot_weightings"),
        pytest.param(lambda n: n.investment_period_weightings, id="period_weightings"),
    ],
)
def test_a_multi_period_axis_survives_a_round_trip(
    multiperiod, multiperiod_built, axis
):
    pd.testing.assert_frame_equal(
        axis(multiperiod_built), axis(multiperiod), check_index_type=False
    )


def test_a_multi_period_network_survives_a_round_trip(multiperiod, multiperiod_built):
    assert_networks_equal(multiperiod_built, multiperiod)


@pytest.mark.parametrize("layout", ["repeated"], indirect=True)
def test_a_set_on_some_snapshots_keeps_the_rest_of_the_series(
    multiperiod, multiperiod_revision
):
    """A snapshot is set by its `period` and `timestep`, some of them by a frame of their tuples.

    Notes
    -----
    - [set](https://energy-models.github.io/datarecord/reference/editing/#set)
    """
    from datarecord.mutable import WorkingRecord

    record = multiperiod_revision.record
    snapshots = record.dimensions["snapshot"]
    first = snapshots.limit(1).fetchall()[0][0]
    in_2040 = snapshots.filter("snapshot.period = 2040").project(
        "snapshot, 0.5 AS value"
    )
    periods = multiperiod.snapshots.get_level_values("period")
    selected = {
        "Manchester Wind": ((in_2040, {}), periods == 2040),
        "Norway Wind": ((0.5, first), np.arange(len(periods)) == 0),
    }
    staged = WorkingRecord(record)
    for entity, ((value, selector), _) in selected.items():
        staged.set("p_max_pu", value, entity=entity, **selector)

    built = PyPSA.build(multiperiod_revision.store.commit(staged).record)
    for entity, (_, changed) in selected.items():
        series = multiperiod.c["Generator"].dynamic["p_max_pu"][entity]
        pd.testing.assert_series_equal(
            built.c["Generator"].dynamic["p_max_pu"][entity],
            series.where(~changed, 0.5),
            check_index_type=False,
        )


@pytest.fixture(scope="module")
def multiperiod_solved(ac_dc, layout):
    """`multiperiod`, solved over every period at once, its solver model dropped so it copies.

    Built afresh, as PyPSA's copy of a multi-period network loses the snapshot
    index's name, which its optimisation needs.
    """
    n = _over_periods(ac_dc, layout)
    n.optimize(solver_name="highs", multi_investment_periods=True)
    n.model.solver_model = None
    return n


def _myopic(n) -> dict[int, pypsa.Network]:
    """`n` split into one network per investment period, as a myopic solve writes it."""
    periods = n.snapshots.get_level_values("period")
    return {
        p: n.copy(snapshots=n.snapshots[periods == p], investment_periods=[p])
        for p in n.investment_periods
    }


def _staggered(layout: str) -> pypsa.Network:
    """Three periods of four hours over one bus, a generator built in each, the first retiring before the second period.

    `repeated` runs the same hours in every period, `distinct` moves them into
    the period's year.
    """
    periods = [2030, 2040, 2050]
    hours = pd.date_range("2030-01-01", periods=4, freq="h")
    moved = layout == "distinct"
    n = pypsa.Network()
    n.set_snapshots(
        pd.MultiIndex.from_tuples(
            [(p, t.replace(year=p) if moved else t) for p in periods for t in hours],
            names=["period", "timestep"],
        )
    )
    n.investment_periods = periods
    n.add("Bus", "bus")
    n.add("Carrier", "gas")
    n.add(
        "Generator",
        [f"gen-{p}" for p in periods],
        bus="bus",
        carrier="gas",
        p_nom_extendable=True,
        build_year=periods,
        lifetime=[10, 30, 30],
        capital_cost=[1.0, 100.0, 200.0],
        marginal_cost=10.0,
    )
    load = pd.Series(np.repeat([50.0, 50.0, 100.0], len(hours)), index=n.snapshots)
    n.add("Load", "load", bus="bus", p_set=load)
    return n


@pytest.fixture(scope="module")
def staggered_revision(tmp_path_factory, layout):
    """`_staggered` exported once for the module."""
    workspace = Workspace()
    yield export_network(
        _staggered(layout), workspace.store(str(tmp_path_factory.mktemp("staggered")))
    )
    workspace.close()


def _solve(n):
    """`n` solved, over its periods at once where it has any, its solver model dropped."""
    n.optimize(
        solver_name="highs", multi_investment_periods=bool(n.investment_periods.size)
    )
    n.model.solver_model = None
    return n


def _assert_dispatch_at(built, n, period: int) -> None:
    """`built` dispatches `n`'s generators as `n` does at its snapshots, those of `period` where `n` has no period."""
    p = built.c.Generator.dynamic["p"]
    at = p if n.investment_periods.size else p.loc[period]
    pd.testing.assert_frame_equal(
        at.loc[n.snapshots, n.c.Generator.static.index],
        n.c.Generator.dynamic["p"],
        check_names=False,
        check_index_type=False,
        check_freq=False,
        obj=f"dispatch of {period}",
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("periods", "period", "results_only"),
    [
        pytest.param("both", 2030, True, id="both-results-first"),
        pytest.param("both", 2040, True, id="both-results-second"),
        pytest.param("one", 2030, True, id="one-results"),
        pytest.param("one", 2030, False, id="one-export"),
    ],
)
def test_a_network_of_one_period_writes_over_a_multi_period_base(
    store,
    multiperiod,
    multiperiod_revision,
    multiperiod_solved,
    periods,
    period,
    results_only,
):
    """A network carrying one of `base`'s investment periods writes its results at that period."""
    member = _myopic(multiperiod_solved)[period]
    base = (
        multiperiod_revision
        if periods == "both"
        else export_network(_myopic(multiperiod)[period], store)
    )
    layer = PyPSA.to_datarecord(
        member, base.record.con, results_only=results_only, base=base.record
    )
    built = PyPSA.build(
        base.store.commit(layer, parent=base).record.select(period=[period])
    )
    _assert_results_equal(built, member)


@pytest.mark.slow
@pytest.mark.parametrize("period", [2030, 2040])
def test_a_period_built_as_a_list_solves_and_writes_back(multiperiod_revision, period):
    """A period selected as a list keeps its investment period, so its solve writes back over `base` as is."""
    base = multiperiod_revision
    view = base.record.select(period=[period])
    n = PyPSA.build(view)
    n.optimize(solver_name="highs", multi_investment_periods=True)
    n.model.solver_model = None
    layer = PyPSA.to_datarecord(n, view.con, results_only=True, base=view)
    written = base.store.commit(layer, parent=base).record
    _assert_results_equal(PyPSA.build(written.select(period=[period])), n)


@pytest.mark.slow
def test_results_of_selected_snapshots_land_on_them(multiperiod_revision):
    """A network of some snapshots of every period writes its results at `base`'s labels of them, and one of snapshots `base` lacks is refused at the call."""
    base = multiperiod_revision
    n = PyPSA.build(
        base.record.select(timestep=slice("2015-01-01 02:00", "2015-01-01 03:00"))
    )
    held = n.c.Generator.static.index
    p = pd.DataFrame(
        np.arange(len(n.snapshots) * len(held), dtype=float).reshape(-1, len(held)) + 1,
        index=n.snapshots,
        columns=held,
    )
    n.c.Generator.dynamic["p"] = p
    layer = PyPSA.to_datarecord(n, base.record.con, results_only=True, base=base.record)
    built = PyPSA.build(
        base.store.commit(layer, parent=base).record
    ).c.Generator.dynamic["p"]
    pd.testing.assert_frame_equal(
        built.loc[p.index, p.columns], p, check_names=False, check_freq=False
    )
    others = built.drop(index=p.index).fillna(0)
    assert (others == 0).all().all(), "no other snapshot holds one"
    periods, timesteps = (n.snapshots.get_level_values(i) for i in range(2))
    n.set_snapshots(
        pd.MultiIndex.from_arrays([periods, timesteps + pd.DateOffset(years=100)])
    )
    with pytest.raises(ValueError, match="snapshot labels `base` lacks"):
        PyPSA.to_datarecord(n, base.record.con, results_only=True, base=base.record)


@pytest.mark.slow
@pytest.mark.parametrize("listed", [True, False], ids=["list", "label"])
def test_per_period_solves_written_in_turn_keep_each_others_results(
    staggered_revision, listed
):
    """Each period built alone, solved and written over the last leaves the other periods' dispatch and the idle generators' capacities standing."""
    revision, solves = staggered_revision, {}
    for period in (2030, 2040, 2050):
        labels = {"period": [period] if listed else period}
        solves[period] = _solve(PyPSA.build(revision.record.select(**labels)))
        revision = _results_over(revision, solves[period], **labels)
    built = PyPSA.build(revision.record)
    for period, n in solves.items():
        _assert_dispatch_at(built, n, period)
    capacities = [n.c.Generator.static["p_nom_opt"] for n in solves.values()]
    latest = pd.concat(capacities).groupby(level=0).last()
    assert latest["gen-2030"] > 0, "the first period's solve builds its generator"
    pd.testing.assert_series_equal(
        built.c.Generator.static["p_nom_opt"],
        latest,
        check_names=False,
        check_index_type=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "select",
    [{"period": [2040, 2050]}, {"period": [2040]}, {"period": 2040}],
    ids=str,
)
def test_a_solve_of_some_periods_leaves_the_rest_to_base(staggered_revision, select):
    """Re-solved at double the load, the selected periods take the re-solve's results, the other periods' dispatch and the capacities of generators idle in the selected ones are `base`'s."""
    whole = _solve(PyPSA.build(staggered_revision.record))
    base = _results_over(staggered_revision, whole)
    n = PyPSA.build(base.record.select(**select))
    n.c.Load.dynamic["p_set"] *= 2
    built = PyPSA.build(_results_over(base, _solve(n), **select).record)

    _assert_dispatch_at(built, n, 2040)
    periods = n.investment_periods if n.investment_periods.size else [2040]
    p = built.c.Generator.dynamic["p"]
    others = ~p.index.get_level_values("period").isin(periods)
    pd.testing.assert_frame_equal(
        p[others], whole.c.Generator.dynamic["p"][others], check_names=False
    )
    expected = whole.c.Generator.static["p_nom_opt"].copy()
    held = n.c.Generator.static["p_nom_opt"]
    expected[held.index] = held
    pd.testing.assert_series_equal(
        built.c.Generator.static["p_nom_opt"],
        expected,
        check_names=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("layout", ["repeated"], indirect=True)
def test_a_multi_period_network_over_a_single_period_base_is_refused(
    single_revision, multiperiod
):
    record = single_revision.record
    with pytest.raises(ValueError, match="over a single-period `base`"):
        PyPSA.to_datarecord(multiperiod, record.con, results_only=True, base=record)


@pytest.mark.parametrize(
    ("network", "results_only", "match"),
    [
        pytest.param(
            "plain", False, "with `results_only=True` only", id="no-period-export"
        ),
        pytest.param(
            "plain", True, "carries investment periods of it", id="no-period-results"
        ),
        pytest.param(
            "member", False, "with `results_only=True` only", id="one-period-export"
        ),
    ],
)
def test_a_network_over_a_multi_period_base_names_its_periods(
    multiperiod_revision, multiperiod_solved, solved, network, results_only, match
):
    """A network not carrying all of `base`'s periods carries some, or was built at one, and is written as results, refused at the call."""
    n = solved if network == "plain" else _myopic(multiperiod_solved)[2030]
    record = multiperiod_revision.record
    with pytest.raises(ValueError, match=match) as raised:
        PyPSA.to_datarecord(n, record.con, results_only=results_only, base=record)
    assert "[2030, 2040]" in str(raised.value)


@pytest.mark.slow
def test_a_child_whose_timesteps_repeat_otherwise_builds_back(
    staggered_revision, layout
):
    """A child whose timesteps repeat otherwise than its parent's restates the `snapshot` tuples."""
    other = _staggered("distinct" if layout == "repeated" else "repeated")
    record = staggered_revision.record
    child = staggered_revision.store.commit(
        PyPSA.to_datarecord(other, record.con, base=record), parent=staggered_revision
    )
    pd.testing.assert_index_equal(
        PyPSA.build(child.record).snapshots, other.snapshots, exact=False
    )


def test_scalar_and_series_rows_union_over_integer_snapshots(workspace):
    """A scalar's NULL snapshot takes the type of the series rows it unions with."""
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.Index([0, 1, 2]))
    n.add("Bus", "b")
    n.add(
        "Load", "series", bus="b", p_set=pd.Series([1.0, 2.0, 3.0], index=n.snapshots)
    )
    n.add("Load", "scalar", bus="b", p_set=5.0)

    p_set = PyPSA.to_datarecord(n, workspace.con).attributes["p_set"].df()
    scalar = p_set[p_set["snapshot"].isna()]
    assert scalar[["entity", "value"]].values.tolist() == [["scalar", 5.0]]
    series = p_set[p_set["entity"] == "series"].sort_values("snapshot")
    assert series["snapshot"].tolist() == [0, 1, 2]


def _round_trip(n, tmp_path):
    import pypsa

    n.export_to_datarecord(tmp_path / "record")
    return pypsa.Network(tmp_path / "record")


def test_an_unsolved_network_round_trips_through_a_folder(ac_dc, tmp_path):
    assert_networks_equal(_round_trip(ac_dc, tmp_path), ac_dc)


def test_a_solved_network_round_trips_through_a_folder(solved, tmp_path):
    got = _round_trip(solved, tmp_path)
    # `Bus.control`, `generator` and `sub_network` are derived, so a build does not set them.
    for column in ("control", "generator", "sub_network"):
        got.c.buses.static[column] = solved.c.buses.static[column]
    got.calculate_dependent_values()
    assert_networks_equal(got, solved)


def test_storage_hvdc_round_trips_through_a_folder(tmp_path):
    import pypsa

    n = pypsa.examples.storage_hvdc()
    got = _round_trip(n, tmp_path)
    # A carrier a bus names but `n.carriers` lacks comes back as a carrier.
    extra = set(n.c.buses.static.carrier) - set(n.c.carriers.static.index)
    assert extra <= set(got.c.carriers.static.index)
    for carrier in extra:
        n.add("Carrier", carrier)
    assert_networks_equal(got, n)


def test_bus_p_survives_a_round_trip(solved, tmp_path):
    got = _round_trip(solved, tmp_path)
    expected = solved.c.buses.dynamic["p"]
    assert not expected.empty
    pd.testing.assert_frame_equal(
        got.c.buses.dynamic["p"][expected.columns],
        expected,
        check_dtype=False,
        check_freq=False,
        check_column_type=False,
        check_index_type=False,
    )


def test_a_missing_path_names_the_datarecord_folder(tmp_path):
    import pypsa

    with pytest.raises(ValueError, match="datarecord folder"):
        pypsa.Network(tmp_path / "missing")


def test_a_csv_folder_still_dispatches_to_csv(ac_dc, tmp_path):
    import pypsa

    ac_dc.export_to_csv_folder(tmp_path / "csv")
    got = pypsa.Network(tmp_path / "csv")
    assert set(got.c.buses.static.index) == set(ac_dc.c.buses.static.index)


def test_export_to_an_existing_directory_raises(ac_dc, tmp_path):
    with pytest.raises(FileExistsError):
        ac_dc.export_to_datarecord(tmp_path)


def test_import_from_datarecord_fills_an_existing_network(ac_dc, tmp_path):
    import pypsa

    ac_dc.export_to_datarecord(tmp_path / "record")
    n = pypsa.Network()
    n.import_from_datarecord(tmp_path / "record")
    assert_networks_equal(n, ac_dc)


def test_a_stochastic_network_round_trips_through_a_folder(tmp_path):
    import pypsa

    n = pypsa.examples.stochastic_network()
    if len(n.snapshots) == 1:
        n.set_snapshots(pd.date_range("2030-01-01", periods=1, freq="h"))
    n.export_to_datarecord(tmp_path / "record")
    got = pypsa.Network(tmp_path / "record")
    assert got.has_scenarios
    pd.testing.assert_series_equal(
        got.scenario_weightings["weight"],
        n.scenario_weightings["weight"],
        check_names=False,
    )
    other = pypsa.Network()
    other.import_from_datarecord(tmp_path / "record")
    assert list(other.scenarios) == list(n.scenarios)


def test_a_link_on_one_bus_twice_round_trips(store):
    """A connection is keyed by its port, so a Link on one bus at two ports keeps both."""
    n = _north()
    n.add("Bus", "south")
    n.add(
        "Link",
        "loop",
        bus0="north",
        bus1="north",
        bus2="south",
        efficiency=0.9,
        efficiency2=0.5,
    )
    flows = {"p0": [3.0, 4.0], "p1": [-2.7, -3.6], "p2": [-0.3, -0.4]}
    for attr, values in flows.items():
        n.c["Link"].dynamic[attr] = pd.DataFrame({"loop": values}, index=n.snapshots)

    revision = export_network(n, store)

    tuples = unpacked(revision.record.properties["connection"])
    held = set(zip(tuples["entity"], tuples["port"], tuples["bus"], strict=True))
    assert held == {
        ("loop", 0, "north"),
        ("loop", 1, "north"),
        ("loop", 2, "south"),
        ("gen", 0, "north"),
    }
    built = PyPSA.build(revision.record)
    link = built.c["Link"].static.loc["loop"]
    assert link[["bus0", "bus1", "bus2"]].tolist() == ["north", "north", "south"]
    assert link[["efficiency", "efficiency2"]].tolist() == [0.9, 0.5]
    for attr, values in flows.items():
        assert built.c["Link"].dynamic[attr]["loop"].tolist() == values


def _flows(n, name, ctype, ports=3):
    """Set a distinct `p` series on each of a component's ports."""
    for i in range(ports):
        n.c[ctype].dynamic[f"p{i}"] = pd.DataFrame(
            {name: [1.0 + i, 2.0 + i]}, index=n.snapshots
        ).rename_axis(columns="name")


@pytest.mark.parametrize("stochastic", [False, True], ids=["single", "stochastic"])
def test_a_same_bus_link_keeps_its_per_port_efficiencies(store, stochastic):
    """A Link with bus0 == bus1 and a third port round-trips every port's bus and efficiency."""
    n = _three_port_link()
    n.add("Link", "same", bus0="a", bus1="a", bus2="c", efficiency=0.9, efficiency2=0.5)
    if stochastic:
        n.set_scenarios({"low": 0.5, "high": 0.5})

    built = PyPSA.build(export_network(n, store).record)

    cols = ["bus0", "bus1", "bus2", "efficiency", "efficiency2"]
    got = (
        built.c["Link"].static[cols].xs("same", level=-1)
        if stochastic
        else (built.c["Link"].static.loc[["same"], cols])
    )
    assert got.iloc[0].tolist() == ["a", "a", "c", 0.9, 0.5]


@pytest.mark.parametrize("shared", [False, True], ids=["own-buses", "one-bus"])
@pytest.mark.parametrize(
    ("network", "ctype", "name"),
    [(_three_port_link, "Link", "split"), (_three_port_process, "Process", "dri")],
    ids=["link", "process"],
)
def test_p_on_every_connection_round_trips(store, network, ctype, name, shared):
    """A three-port Link or Process keeps `p` on each port, on its own buses or all on one."""
    n = network()
    if shared:
        n.c[ctype].static.loc[name, ["bus0", "bus1", "bus2"]] = "a"
    _flows(n, name, ctype)

    revision = export_network(n, store)
    built = PyPSA.build(revision.record)

    for i in range(3):
        assert built.c[ctype].dynamic[f"p{i}"][name].tolist() == [1.0 + i, 2.0 + i]
    assert built.c[ctype].static.loc[name, ["bus0", "bus1", "bus2"]].tolist() == (
        ["a"] * 3 if shared else ["a", "b", "c"]
    )
    assert_networks_equal(built, n)


def test_a_generator_sits_at_port_zero_and_its_p_round_trips(store):
    """A one-port type's bus and `p` are keyed at port 0."""
    n = _north()
    n.c["Generator"].dynamic["p"] = pd.DataFrame({"gen": [1.0, 2.0]}, index=n.snapshots)

    revision = export_network(n, store)

    tuples = unpacked(revision.record.properties["connection"])
    assert list(zip(tuples["entity"], tuples["port"], tuples["bus"], strict=True)) == [
        ("gen", 0, "north")
    ]
    assert _flat(relation(revision, "p"))["port"].to_list() == [0, 0]
    assert PyPSA.build(revision.record).c["Generator"].dynamic["p"]["gen"].tolist() == [
        1.0,
        2.0,
    ]


def test_a_link_without_bus2_writes_no_default_efficiency2(store):
    """A Link with an empty `bus2` has no port-2 connection or efficiency, though the type has a `bus2` column."""
    n = _north()
    for bus in ("south", "east"):
        n.add("Bus", bus)
    n.add("Link", "two", bus0="north", bus1="south")
    n.add("Link", "three", bus0="north", bus1="south", bus2="east", efficiency2=0.5)
    assert "bus2" in n.c["Link"].static.columns
    assert n.c["Link"].static.at["two", "bus2"] == ""

    revision = export_network(n, store)

    efficiency = _flat(revision.record.attributes["efficiency"])
    two = efficiency[efficiency["entity"] == "two"]
    assert 2 not in two["port"].tolist()
    assert (
        efficiency[(efficiency["entity"] == "three") & (efficiency["port"] == 2)].shape[
            0
        ]
        > 0
    )
    tuples = unpacked(revision.record.properties["connection"])
    held = set(zip(tuples["entity"], tuples["port"], tuples["bus"], strict=True))
    assert held == {
        ("gen", 0, "north"),
        ("two", 0, "north"),
        ("two", 1, "south"),
        ("three", 0, "north"),
        ("three", 1, "south"),
        ("three", 2, "east"),
    }
    built = PyPSA.build(revision.record)
    assert built.c["Link"].static.loc["two", ["bus0", "bus1", "bus2"]].tolist() == [
        "north",
        "south",
        "",
    ]


def _acceptance_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Bus", "c")
    n.add("Generator", "gen", bus="b", p_nom=10.0, marginal_cost=1.0)
    n.add(
        "Link",
        "l",
        bus0="b",
        bus1="b",
        bus2="c",
        efficiency=0.9,
        efficiency2=0.5,
        p_nom=10.0,
    )
    return n


def test_a_same_bus_link_with_flows_round_trips_eagerly_and_lazily(store):
    """bus0, bus1, bus2, both efficiencies and p0, p1, p2 come back from a build and a lazy open."""
    n = _acceptance_network()
    _flows(n, "l", "Link")
    revision = export_network(n, store)

    tuples = unpacked(revision.record.properties["connection"])
    rows = set(zip(tuples["entity"], tuples["port"], tuples["bus"], strict=True))
    assert rows == {("l", 0, "b"), ("l", 1, "b"), ("l", 2, "c"), ("gen", 0, "b")}
    assert "role" not in revision.record.dimensions

    for got in (
        pypsa.Network.from_datarecord(revision.record),
        PyPSA.build(revision.record),
    ):
        link, source = got.c["Link"], n.c["Link"]
        for attr in ("bus0", "bus1", "bus2", "efficiency", "efficiency2"):
            assert link.static.at["l", attr] == source.static.at["l", attr]
        for attr in ("p0", "p1", "p2"):
            pd.testing.assert_series_equal(
                link.dynamic[attr]["l"], source.dynamic[attr]["l"], check_freq=False
            )


def _stochastic_link(bus1_by_scenario):
    n = _three_port_link()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    for scenario, bus in bus1_by_scenario.items():
        n.c["Link"].static.loc[(scenario, "split"), "bus1"] = bus
    return n


def test_a_link_bus_differing_per_scenario_is_refused(workspace):
    """A connection's bus has no `scenario` dim, so a Link's `bus1` that differs by scenario is refused."""
    n = _stochastic_link({"low": "b", "high": "a"})

    with pytest.raises(ValueError, match="Link 'split' has `bus1` .*'b'.*'a'"):
        PyPSA.to_datarecord(n, workspace.con)


def test_a_link_bus2_set_in_one_scenario_and_empty_in_another_is_refused(workspace):
    """A port attached in one scenario and unattached in another differs by scenario, in either order."""
    for first, second in (("c", ""), ("", "c")):
        held = {"low": first, "high": second}
        n = _stochastic_link({"low": "b", "high": "b"})
        n.c["Link"].static.loc[("low", "split"), "bus2"] = first
        n.c["Link"].static.loc[("high", "split"), "bus2"] = second

        with pytest.raises(
            ValueError, match=re.escape(f"Link 'split' has `bus2` {held}")
        ):
            PyPSA.to_datarecord(n, workspace.con)


def test_a_bus_agreeing_across_scenarios_writes_one_row_per_port(store):
    """A stochastic Link whose buses agree writes a single connection row per `(entity, port)`."""
    n = _stochastic_link({"low": "b", "high": "b"})

    revision = export_network(n, store)

    tuples = unpacked(revision.record.properties["connection"])
    assert sorted(
        zip(tuples["entity"], tuples["port"], tuples["bus"], strict=True)
    ) == [
        ("split", 0, "a"),
        ("split", 1, "b"),
        ("split", 2, "c"),
    ]


def test_default_snapshots_are_refused(workspace):
    import pypsa

    n = pypsa.Network()
    n.add("Bus", "north")
    n.add("Generator", "gen", bus="north")
    with pytest.raises(UnsupportedRecordError, match="set_snapshots"):
        PyPSA.to_datarecord(n, workspace.con)


def test_a_series_naming_no_component_of_its_type_is_refused(workspace):
    n = _north()
    n.c.Generator.dynamic.p_max_pu["ghost"] = [0.5, 0.6]
    with pytest.raises(UnsupportedRecordError) as refused:
        PyPSA.to_datarecord(n, workspace.con)
    message = str(refused.value)
    for word in ("Generator", "p_max_pu", "ghost", "check_time_series"):
        assert word in message


def test_a_series_of_a_type_without_members_is_refused(workspace):
    n = _north()
    n.c.Load.dynamic.p_set["ghost"] = [1.0, 2.0]
    with pytest.raises(UnsupportedRecordError, match=r"Load\.p_set`.*ghost"):
        PyPSA.to_datarecord(n, workspace.con)


def test_a_clean_series_is_not_refused(workspace):
    n = _north()
    n.c.Generator.dynamic.p_max_pu["gen"] = [0.5, 0.6]
    PyPSA.to_datarecord(n, workspace.con)


def test_a_stray_result_column_is_refused_on_a_results_only_export(store):
    revision = store.commit(
        PyPSA.to_datarecord(_north(), store.workspace.con), parent=None
    )
    n = PyPSA.build(revision.record)
    n.c.Generator.dynamic["p"] = pd.DataFrame(
        {"gen": [1.0, 2.0], "ghost": [3.0, 4.0]}, index=n.snapshots
    )
    with pytest.raises(UnsupportedRecordError, match=r"Generator\.p`.*ghost"):
        PyPSA.to_datarecord(
            n, revision.record.con, results_only=True, base=revision.record
        )


def test_a_stochastic_series_under_another_scenario_is_refused(workspace):
    n = _north()
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.Generator.dynamic.p_max_pu[("other", "gen")] = [0.5, 0.6]
    with pytest.raises(UnsupportedRecordError, match="other"):
        PyPSA.to_datarecord(n, workspace.con)


def test_the_stray_columns_listed_are_capped_at_ten(workspace):
    n = _north()
    for i in range(12):
        n.c.Generator.dynamic.p_max_pu[f"ghost{i:02}"] = [0.5, 0.6]
    with pytest.raises(UnsupportedRecordError) as refused:
        PyPSA.to_datarecord(n, workspace.con)
    message = str(refused.value)
    assert "ghost09" in message
    assert "ghost10" not in message
    assert "and 2 more" in message


def test_from_datarecord_without_the_package_names_the_extra(
    single_revision, monkeypatch
):
    import pypsa

    monkeypatch.setitem(sys.modules, "datarecord", None)
    with pytest.raises(ImportError, match=r"pypsa\[datarecord\]"):
        pypsa.Network.from_datarecord(single_revision.record)


def test_importing_pypsa_imports_no_duckdb():
    code = (
        "import sys\nimport pypsa\n"
        "assert 'duckdb' not in sys.modules\n"
        "assert 'datarecord' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_objective_survives_a_folder_round_trip(solved, tmp_path):
    import pypsa

    solved.export_to_datarecord(tmp_path / "record")
    got = pypsa.Network(tmp_path / "record")
    assert got.objective == pytest.approx(solved.objective)
    assert got.objective_constant == pytest.approx(solved.objective_constant)


def test_record_attributes_leave_out_names_unchecked(ac_dc):
    from pypsa.network.io._common import _collect_network_attributes
    from pypsa.network.names import unchecked_names

    with unchecked_names(ac_dc):
        attrs = _collect_network_attributes(ac_dc, skip={"srid"})
    assert "_names_unchecked" not in attrs
    assert "deduplicated_names" not in attrs


# -- selections ----------------------------------------------------------------

_SNAPSHOT_SELECTIONS = pytest.mark.parametrize(
    ("value", "hours"),
    [
        pytest.param("2015-01", list(range(10)), id="partial-scalar"),
        pytest.param(
            [pd.Timestamp("2015-01-01 05:00"), "2015-01-01 01:00"], [1, 5], id="list"
        ),
        pytest.param(slice("2015-01-01 02", "2015-01-01 04"), [2, 3, 4], id="slice"),
        pytest.param(slice(None, "2015-01-01 01"), [0, 1], id="open-start"),
        pytest.param(slice("2015-01", None), list(range(10)), id="partial-month"),
        pytest.param(slice("2015-01-01 09", "2015-01-01 09"), [9], id="one-slice"),
    ],
)


def _redeclared_over(store, n, attr: str, dim: str, rows: list[dict]) -> Revision:
    """`n` written as a root of `store` declaring `attr` static and over `dim` too, its rows `rows`.

    Static, since an attribute over `snapshot` is over its coordinate `period` already.
    A `dim` the schema lacks is declared too, an integer one of the labels `rows` name.
    """
    revision = export_network(n, store)
    declared = revision.manifest.schema
    spec = declared.attributes[attr]
    over = spec.model_copy(update={"dims": spec.dims - {"snapshot"} | {dim}})
    attributes = {**declared.attributes, attr: over}
    update: dict = {"attributes": attributes}
    own = dim not in declared.dimensions
    if own:
        update["dimensions"] = {**declared.dimensions, dim: Dimension(dtype="int")}
    layer = empty_root(store, declared.model_copy(update=update))
    source = Path(store.layer_dir(revision.id))
    for kind in ("dimensions", "attributes"):
        shutil.copytree(source / kind, layer.dir / kind)
    write_attribute(layer, attr, rows)
    if own:
        labels = sorted({r[dim] for r in rows})
        pd.DataFrame({dim: pd.Series(labels, dtype="Int64")}).to_parquet(
            layer.dir / "dimensions" / f"{dim}.parquet", index=False
        )
        path = layer.dir / "attributes" / f"{attr}.parquet"
        frame = pd.read_parquet(path)
        frame[dim] = pd.Series([r[dim] for r in rows], dtype="Int64")
        frame.to_parquet(path, index=False)
    return layer.commit()


@pytest.mark.parametrize(
    ("spread", "dim", "attr", "values"),
    [
        pytest.param(
            lambda n: n.set_investment_periods([2030, 2040]),
            "period",
            "marginal_cost",
            {2030: 3.0, 2040: 4.0},
            id="period",
        ),
        pytest.param(
            lambda n: n.set_scenarios({"lo": 0.5, "hi": 0.5}),
            "scenario",
            "p_nom_mod",
            {"lo": 1.0, "hi": 2.0},
            id="scenario",
        ),
    ],
)
def test_a_value_apart_over_a_dim_pypsa_holds_it_without_is_refused(
    store, spread, dim, attr, values
):
    """A network holds one value along a dim PyPSA's registry declares the attribute without, so rows naming several of its labels are reported and refused, and a selection at one label builds its value."""
    n = _north()
    spread(n)
    rows = [{"entity": "gen", dim: label, "value": v} for label, v in values.items()]
    record = _redeclared_over(store, n, attr, dim, rows).record

    assert PyPSA.verify(record).unsupported_values == {("Generator", attr)}
    with pytest.raises(ValueError, match=rf"`{attr}` over \['{dim}'\].*list of one"):
        PyPSA.build(record)
    with pytest.raises(ValueError, match="list of one"):
        PyPSA.open(record)
    views = {label: record.select(**{dim: [label]}) for label in values}
    assert not any(PyPSA.verify(view) for view in views.values())
    built = {
        label: PyPSA.build(view).c.Generator.static[attr].tolist()
        for label, view in views.items()
    }
    assert built == {label: [v] for label, v in values.items()}, (
        "each label builds its own value"
    )


def test_a_selected_period_builds_its_row_over_the_broadcast_one(store):
    """At one kept period, the row naming it is the more specific than the row leaving it NULL, so the component holds one value, the period's."""
    n = _north()
    n.set_investment_periods([2030, 2040])
    rows: list[dict] = [
        {"entity": "gen", "period": None, "value": 10.0},
        {"entity": "gen", "period": 2030, "value": 30.0},
    ]
    record = _redeclared_over(store, n, "marginal_cost", "period", rows).record

    built = PyPSA.build(record.select(period=[2030]))

    assert built.c.Generator.static["marginal_cost"].tolist() == [30.0]


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        pytest.param(
            {"bus": ["London", "Manchester", "Norwich"]},
            {
                "Bus": ["London", "Norwich", "Manchester"],
                "Line": ["0", "1", "5"],
                "Link": [],
                "Generator": ["Manchester Wind", "Manchester Gas"],
                "Load": ["London", "Norwich", "Manchester"],
            },
            id="bus",
        ),
        pytest.param(
            {"carrier": ["AC", "wind", "load"]},
            {
                "Bus": [
                    "London",
                    "Norwich",
                    "Manchester",
                    "Bremen",
                    "Frankfurt",
                    "Norway",
                ],
                "Line": ["0", "1", "5", "6"],
                "Link": [],
                "Generator": ["Manchester Wind", "Norway Wind", "Frankfurt Wind"],
                "Carrier": ["wind", "load", "AC"],
            },
            id="carrier",
        ),
        pytest.param(
            {"entity": ["Manchester Wind", "0"]},
            {"Line": ["0"], "Generator": ["Manchester Wind"], "Load": []},
            id="entity",
        ),
    ],
)
def test_a_selection_drops_every_label_referring_to_a_dropped_one(
    single_revision, select, expected
):
    """A bus or entity of another carrier goes, and an entity attached to a dropped bus with it."""
    built = PyPSA.build(single_revision.record.select(**select))
    held = {ctype: built.c[ctype].static.index.tolist() for ctype in expected}
    assert held == expected
    built.consistency_check(strict=["unknown_buses", "unknown_carriers"])


@_SNAPSHOT_SELECTIONS
def test_selected_snapshots_are_the_networks(single_revision, ac_dc, value, hours):
    """Snapshots are selected by timestamp and kept, as a list, a partial date or a slice."""
    built = PyPSA.build(single_revision.record.select(snapshot=value))
    assert list(built.snapshots) == list(ac_dc.snapshots[hours])
    pd.testing.assert_frame_equal(
        built.c.Generator.dynamic["p_max_pu"],
        ac_dc.c.Generator.dynamic["p_max_pu"].iloc[hours],
        check_index_type=False,
        check_names=False,
        check_like=True,
    )


@_SNAPSHOT_SELECTIONS
def test_a_timestamp_selects_its_snapshot_in_every_period(
    multiperiod, multiperiod_revision, ac_dc, value, hours
):
    """A timestamp selects a multi-period snapshot through `timestep`, in each period holding it, and a period holding none goes."""
    built = PyPSA.build(multiperiod_revision.record.select(timestep=value))
    timesteps = multiperiod.snapshots.get_level_values("timestep")
    expected = multiperiod.snapshots[timesteps.isin(ac_dc.snapshots[hours])]
    pd.testing.assert_index_equal(built.snapshots, expected, exact=False)


@pytest.mark.parametrize(
    ("select", "hours"),
    [
        pytest.param({"timestep": ["2015-01-01 07:00"]}, [7], id="timestamp"),
        pytest.param(
            {"timestep": slice("2015-01-01 06", None)}, [6, 7, 8, 9], id="slice"
        ),
    ],
)
def test_one_period_takes_snapshots_by_timestamp(
    multiperiod_revision, ac_dc, select, hours
):
    """A snapshot selected beside one period is one of that period's, named by its timestep."""
    built = PyPSA.build(multiperiod_revision.record.select(period=2040, **select))
    assert list(built.snapshots) == list(ac_dc.snapshots[hours])


@pytest.mark.parametrize("layout", ["distinct"], indirect=True)
@pytest.mark.parametrize("period", [2040, [2040]], ids=["label", "list"])
def test_a_selection_leaving_no_snapshot_is_refused(multiperiod_revision, period):
    """Split between the periods, the first hour is 2030's alone."""
    view = multiperiod_revision.record.select(
        period=period, timestep=["2015-01-01 00:00"]
    )
    with pytest.raises(ValueError, match="holding no snapshot"):
        PyPSA.build(view)


@pytest.mark.slow
def test_dims_selected_together_combine(store, multiperiod):
    """One period, one scenario and two carriers, a deterministic single-period network of the AC buses and their wind generators."""
    n = multiperiod.copy()
    n.set_scenarios({"lo": 0.2, "hi": 0.8})
    record = export_network(n, store).record
    view = record.select(period=2030, scenario="hi", carrier=["AC", "wind"])

    assert not PyPSA.verify(view)
    built = PyPSA.build(view)

    assert not built.has_scenarios
    assert not built.investment_periods.size
    assert set(built.c.Bus.static["carrier"]) == {"AC"}
    assert built.c.Generator.static.index.tolist() == [
        "Manchester Wind",
        "Norway Wind",
        "Frankfurt Wind",
    ]


def test_a_null_carrier_refers_to_no_dropped_carrier(store, ac_dc):
    """PyPSA's `""` carrier is NULL, so a load without one stays under any carrier selected."""
    n = ac_dc.copy()
    n.c.Load.static.loc["London", "carrier"] = ""
    record = export_network(n, store).record
    built = PyPSA.build(record.select(carrier=["AC"]))
    assert built.c.Load.static.index.tolist() == ["London"]


@pytest.mark.parametrize("period", [2030, 2040])
@pytest.mark.slow
def test_one_period_is_a_single_period_network_of_its_active_entities(
    store, ac_dc, layout, period
):
    """Over one period the network is indexed by its timesteps, without the period weightings, holding the generators PyPSA holds active in it."""
    n = _over_periods(ac_dc, layout)
    n.c.Generator.static.loc["Manchester Gas", "build_year"] = 2040
    n.c.Generator.static.loc["Norway Gas", ["build_year", "lifetime"]] = [2000, 35]
    record = export_network(n, store).record

    built = PyPSA.build(record.select(period=period))

    in_period = n.snapshots.get_level_values("period") == period
    assert not built.investment_periods.size, "single-period"
    assert list(built.snapshots) == list(
        n.snapshots[in_period].get_level_values("timestep")
    ), "indexed by the period's timesteps"
    assert list(built.snapshot_weightings["objective"]) == list(
        n.snapshot_weightings["objective"][in_period]
    )
    active = n.c.Generator.get_active_assets(period)
    assert list(built.c.Generator.static.index) == list(active.index[active])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2030, {"co2_limit": np.nan, "cap_2030": np.nan}),
        (2040, {"co2_limit": np.nan, "cap_2040": np.nan}),
        ([2040], {"co2_limit": np.nan, "cap_2040": 2040.0}),
    ],
    ids=str,
)
@pytest.mark.slow
def test_selected_periods_keep_the_global_constraints_applying_in_them(
    store, ac_dc, value, expected
):
    """A constraint for a period the selection drops goes, one period leaves the kept ones applying in every period, as a single-period network has no period to name."""
    n = _over_periods(ac_dc, "repeated")
    n.add(
        "GlobalConstraint",
        ["cap_2030", "cap_2040"],
        investment_period=[2030, 2040],
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=1.0,
    )
    record = export_network(n, store).record

    built = PyPSA.build(record.select(period=value))

    pd.testing.assert_series_equal(
        built.c.GlobalConstraint.static["investment_period"].sort_index(),
        pd.Series(expected, name="investment_period").sort_index(),
        check_index_type=False,
        check_names=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("value", [2040, [2040]], ids=str)
def test_a_period_selection_dropping_every_constraint_builds_without_any(
    store, ac_dc, value
):
    """A selection whose periods none of the constraints applies in builds a network with no constraint."""
    n = _over_periods(ac_dc, "repeated")
    n.remove("GlobalConstraint", "co2_limit")
    n.add(
        "GlobalConstraint",
        "cap_2030",
        investment_period=2030,
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=1.0,
    )
    record = export_network(n, store).record

    view = record.select(period=value)
    for built in (PyPSA.build(view), PyPSA.open(view)):
        assert built.c.GlobalConstraint.static.empty


@pytest.mark.slow
def test_results_over_a_period_selection_keep_the_mu_of_the_constraints_it_dropped(
    store, ac_dc
):
    """Results written over a period selection leave the `mu` of a constraint the selection dropped as `base` holds it, and one the network lacks within the selection NULL."""
    n = _over_periods(ac_dc, "repeated")
    n.add(
        "GlobalConstraint",
        ["cap_2030", "cap_2040"],
        investment_period=[2030, 2040],
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=1.0,
    )
    n.c.GlobalConstraint.static["mu"] = 3.0
    revision = export_network(n, store)
    solved = PyPSA.build(revision.record.select(period=2040))
    assert set(solved.c.GlobalConstraint.static.index) == {"co2_limit", "cap_2040"}
    solved.remove("GlobalConstraint", "co2_limit")
    solved.c.GlobalConstraint.static["mu"] = 9.0
    layer = _results_over(revision, solved, period=2040)
    mu = _stored_mu(layer.record)
    assert mu["cap_2040"] == 9.0
    assert mu["cap_2030"] == 3.0, "dropped by the selection, so left as `base` holds it"
    assert pd.isna(mu["co2_limit"]), "in the selection yet lacked by the network"


@pytest.mark.slow
def test_an_investment_period_varying_by_scenario_is_refused_in_a_period_selection(
    store, ac_dc
):
    """A network holds one `investment_period` of a constraint, so a selection keeping the scenarios it differs over is refused."""
    n = _over_periods(ac_dc, "repeated")
    n.set_scenarios({"lo": 0.5, "hi": 0.5})
    static = n.c.GlobalConstraint.static
    static.loc[("lo", "co2_limit"), "investment_period"] = 2030.0
    static.loc[("hi", "co2_limit"), "investment_period"] = 2040.0
    record = export_network(n, store).record

    with pytest.raises(ValueError, match="`investment_period` varies over a dim"):
        PyPSA.build(record.select(period=[2040]))


@pytest.mark.slow
def test_results_of_a_period_over_scenario_varying_constraints_are_refused(
    store, ac_dc
):
    """A network of one period over a `base` whose `investment_period` varies by scenario cannot tell which constraints the period drops, so it is refused rather than written NULL."""
    n = _over_periods(ac_dc, "repeated")
    n.set_scenarios({"lo": 0.5, "hi": 0.5})
    static = n.c.GlobalConstraint.static
    static.loc[("lo", "co2_limit"), "investment_period"] = 2030.0
    static.loc[("hi", "co2_limit"), "investment_period"] = 2040.0
    static["mu"] = 3.0
    revision = export_network(n, store)
    member = _over_periods(ac_dc, "repeated")
    periods = member.snapshots.get_level_values("period")
    member = member.copy(
        snapshots=member.snapshots[periods == 2040], investment_periods=[2040]
    )
    member.set_scenarios({"lo": 0.5, "hi": 0.5})

    with pytest.raises(ValueError, match="`investment_period` varies over a dim"):
        _results_over(revision, member)


@pytest.mark.parametrize(
    "value", [[2040], slice(2035, None), slice(2040, 2040)], ids=str
)
def test_periods_selected_keep_a_multi_period_network(
    multiperiod, multiperiod_revision, value
):
    """A list or a slice of periods keeps the dim, their snapshots, weightings and every entity."""
    built = PyPSA.build(multiperiod_revision.record.select(period=value))
    in_period = multiperiod.snapshots.get_level_values("period") == 2040
    pd.testing.assert_index_equal(
        built.snapshots, multiperiod.snapshots[in_period], exact=False
    )
    pd.testing.assert_frame_equal(
        built.investment_period_weightings,
        multiperiod.investment_period_weightings.loc[[2040]],
        check_index_type=False,
    )
    pd.testing.assert_index_equal(
        built.c.Generator.static.index, multiperiod.c.Generator.static.index
    )


@pytest.fixture
def stochastic(store, ac_dc):
    """`ac_dc` over three weighted scenarios, `hi` with a marginal cost of its own."""
    n = ac_dc.copy()
    n.set_scenarios({"lo": 0.2, "mid": 0.3, "hi": 0.5})
    n.c.Generator.static.loc[("hi", "Norway Gas"), "marginal_cost"] = 99.0
    return export_network(n, store)


def test_verify_reports_a_curve_of_a_stochastic_network(stochastic):
    """A curve frame has no scenario axis, so a curve over scenarios stays reported."""
    child = empty_child(stochastic)
    write_attribute(
        child,
        "marginal_cost",
        [
            {"entity": "Norway Gas", "breakpoint": x, "value": v}
            for x, v in ((0.0, 20.0), (50.0, 35.0))
        ],
    )
    missing = PyPSA.verify(child.commit().record)
    assert ("Generator", "marginal_cost") in missing.unsupported_values


def test_one_scenario_is_a_deterministic_network_of_its_values(stochastic):
    built = PyPSA.build(stochastic.record.select(scenario="hi"))
    assert not built.has_scenarios
    assert built.c.Generator.static.loc["Norway Gas", "marginal_cost"] == 99.0


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(
            {"lo": 0.2, "mid": 0.3}, {"lo": 0.2 / 0.5, "mid": 0.3 / 0.5}, id="stored"
        ),
        pytest.param(
            {"lo": 0.2},
            {"lo": 0.2 / (0.2 + 1 / 3), "mid": (1 / 3) / (0.2 + 1 / 3)},
            id="unstored",
        ),
    ],
)
def test_selected_scenarios_are_renormalised(stochastic, stored, expected):
    """A scenario no `scenario_weight` row reaches takes its equal share of the whole axis before the kept weights are renormalised."""
    path = Path(
        stochastic.store.layer_dir(stochastic.id),
        "attributes",
        "scenario_weight.parquet",
    )
    rows = pd.read_parquet(path)
    rows["value"] = rows["scenario"].map(stored)
    rows.dropna(subset="value").to_parquet(path, index=False)

    built = PyPSA.build(stochastic.record.select(scenario=["lo", "mid"]))

    weights = built.scenario_weightings["weight"].to_dict()
    assert weights == pytest.approx(expected)


def test_selected_scenarios_weighing_nothing_are_refused(store, ac_dc):
    n = ac_dc.copy()
    n.set_scenarios({"lo": 0.0, "mid": 0.0, "hi": 1.0})
    record = export_network(n, store).record
    with pytest.raises(ValueError, match="weighing nothing"):
        PyPSA.build(record.select(scenario=["lo", "mid"]))


@pytest.mark.parametrize(
    "labels",
    [
        {"snapshot": "2015-01-01 03:00"},
        {"bus": "London"},
        {"entity": "Manchester Wind"},
        {"carrier": "AC"},
        {"ctype": "Generator"},
        {"port": 0},
    ],
    ids=lambda labels: next(iter(labels)),
)
@pytest.mark.parametrize("method", [PyPSA.build, PyPSA.verify], ids=["build", "verify"])
def test_a_selection_squeezing_a_dim_the_network_keeps_is_refused(
    single_revision, labels, method
):
    """A network is built from a selection squeezing `period`, `scenario` or a dim of the record's own alone, any other dim takes a list of one."""
    (dim,) = labels
    with pytest.raises(ValueError, match=rf"not \['{dim}'\]"):
        method(single_revision.record.select(**labels))


@pytest.mark.parametrize("method", [PyPSA.build, PyPSA.open], ids=["build", "open"])
def test_a_port_selection_drops_the_entities_that_lose_a_port(store, method):
    """Keeping ports 0 and 1 drops the three-port Link, two-port Links and one-port types stay."""
    n = _three_port_link()
    n.add("Link", "pair", bus0="a", bus1="b")
    n.add("Generator", "gen", bus="a")
    record = export_network(n, store).record
    built = method(record.select(port=[0, 1]))
    assert built.c.Link.static.index.tolist() == ["pair"]
    assert built.c.Generator.static.index.tolist() == ["gen"]
    assert set(built.c.Bus.static.index) == {"a", "b", "c"}


@pytest.fixture
def swept(store):
    """`_north` over two weather years, `marginal_cost` apart between them, a record with a dim of its own."""
    rows = [
        {"entity": "gen", "weather_year": 2012, "value": 1.0},
        {"entity": "gen", "weather_year": 2013, "value": 2.0},
    ]
    return _redeclared_over(store, _north(), "marginal_cost", "weather_year", rows)


def test_a_dim_of_its_own_is_unsupported_unless_selected_at_one_label(swept):
    """One network has no dim of its own, whole it is reported and refused, and one label builds the member there."""
    record = swept.record
    assert PyPSA.verify(record).unsupported_dimensions == {"weather_year"}
    with pytest.raises(UnsupportedRecordError, match="weather_year"):
        PyPSA.build(record)
    kept = record.select(weather_year=[2013])
    assert PyPSA.verify(kept).unsupported_dimensions == {"weather_year"}, (
        "a list keeps it"
    )
    view = record.select(weather_year=2013)
    assert not PyPSA.verify(view)
    built = PyPSA.build(view)
    assert built.c.Generator.static["marginal_cost"].tolist() == [2.0]
    assert PyPSA.build(record.select(weather_year=2012)).c.Generator.static[
        "marginal_cost"
    ].tolist() == [1.0]


@pytest.mark.slow
def test_results_over_a_dim_of_its_own_land_at_the_selected_label(swept):
    """A network's results over a record with a dim of its own are written at the label it was built from, and no other label sees them."""
    record = swept.record
    n = _solve(PyPSA.build(record.select(weather_year=2013)))
    with pytest.raises(ValueError, match="over the selection it was built from"):
        PyPSA.to_datarecord(n, record.con, results_only=True, base=record)
    written = _results_over(swept, n, weather_year=2013)

    def held(year):
        p = written.record.select(weather_year=year).attributes["p"].df()
        return int(p["value"].notna().sum())

    assert held(2013) == len(n.snapshots), "the selected label holds the solve"
    assert held(2012) == 0, "no other label sees it"


def test_import_from_datarecord_takes_a_selection(single_revision):
    import pypsa

    n = pypsa.Network()
    n.import_from_datarecord(single_revision.record.select(bus=["London", "Norwich"]))
    assert n.c.Bus.static.index.tolist() == ["London", "Norwich"]


def test_a_piecewise_attribute_builds(exported):
    """A curve written by hand builds into `piecewise`, its scalar left at the default."""
    revision = _curve_child(
        exported,
        "marginal_cost",
        curves={"Manchester Wind": ((0.0, 0.0), (0.5, 20.0), (1.0, 35.0))},
        scalar=False,
    )
    n = PyPSA.build(revision.record)
    frame = n.c.Generator.piecewise["marginal_cost"]
    name = "Manchester Wind"
    assert frame.index.to_list() == [0, 1, 2]
    assert frame.index.name == "breakpoint"
    assert frame.columns.to_list() == [(name, "p_pu"), (name, "marginal_cost")]
    assert frame.columns.names == ["name", "attribute"]
    assert (frame.dtypes == "float64").all()
    assert frame[name].to_dict("list") == {
        "p_pu": [0.0, 0.5, 1.0],
        "marginal_cost": [0.0, 20.0, 35.0],
    }
    default = n.c.Generator.defaults.at["marginal_cost", "default"]
    assert n.c.Generator.static.at[name, "marginal_cost"] == default


def test_a_ragged_pair_of_curves_reads_back_nan_padded(exported):
    """The shorter curve ends in NaN, as `n.add` lays ragged curves out."""
    revision = _curve_child(
        exported,
        "marginal_cost",
        curves={
            "Manchester Wind": ((0.0, 0.0), (0.5, 20.0), (1.0, 35.0)),
            "Manchester Gas": ((0.0, 0.0), (1.0, 10.0)),
        },
        scalar=False,
    )
    frame = PyPSA.build(revision.record).c.Generator.piecewise["marginal_cost"]
    assert frame.index.to_list() == [0, 1, 2]
    assert frame["Manchester Wind"]["p_pu"].to_list() == [0.0, 0.5, 1.0]
    assert frame["Manchester Gas"]["p_pu"].to_list()[:2] == [0.0, 1.0]
    assert frame["Manchester Gas"]["marginal_cost"].to_list()[:2] == [0.0, 10.0]
    assert frame["Manchester Gas"].iloc[2].isna().all()


def test_a_link_curve_on_the_bus2_connection_reads_as_efficiency2(store):
    """The port key `efficiency2` is announced and read from the bus2 connection."""
    revision = export_network(_three_port_link(), store)
    child = empty_child(revision)
    write_attribute(
        child,
        "efficiency",
        [
            {"entity": "split", "port": 2, "breakpoint": x, "value": v}
            for x, v in ((0.0, 0.0), (1.0, 0.5))
        ],
    )
    n = PyPSA.open(child.commit().record)
    piecewise = n.c.Link.piecewise
    assert "efficiency2" in piecewise.pending
    assert piecewise["efficiency"].empty
    frame = piecewise["efficiency2"]
    assert frame[("split", "efficiency2")].to_list() == [0.0, 0.5]
    assert frame[("split", "p_pu")].to_list() == [0.0, 1.0]


def _curve_rows(path, attribute):
    rows = pd.read_parquet(Path(path) / "attributes" / f"{attribute}.parquet")
    return rows[rows["breakpoint"].notna()]


def test_piecewise_network_round_trips_through_a_record(piecewise_network, tmp_path):
    # A record's snapshots are datetimes, the fixture's are integers.
    n = piecewise_network.copy()
    n.add("Carrier", "AC")  # a record lists the carriers components name
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.c.loads.dynamic.p_set = pd.DataFrame(
        {"load": [10.0, 20.0]},
        index=n.snapshots,
        columns=pd.Index(["load"], name="name"),
    )
    n.export_to_datarecord(tmp_path / "record")
    got = pypsa.Network(tmp_path / "record")
    assert n.equals(got.copy(), log_mode="strict")
    n.optimize(solver_name="highs")
    got.optimize(solver_name="highs")
    assert got.objective == pytest.approx(n.objective)


def test_an_efficiency2_curve_round_trips(tmp_path):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", ["a", "b", "c"])
    n.add(
        "Link",
        "split",
        bus0="a",
        bus1="b",
        bus2="c",
        p_nom=10,
        efficiency2={0.0: 0.0, 0.5: 0.4, 1.0: 0.5},
    )
    n.export_to_datarecord(tmp_path / "record")
    got = pypsa.Network(tmp_path / "record")
    assert n.equals(got.copy(), log_mode="strict")
    assert len(got.c.links.piecewise["efficiency2"]) == 3


def test_ragged_curves_round_trip_without_nan_breakpoints(tmp_path):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "long",
        bus="bus",
        p_nom=10,
        marginal_cost={0.0: 1.0, 0.5: 2.0, 1.0: 4.0},
    )
    n.add(
        "Generator",
        "short",
        bus="bus",
        p_nom=10,
        marginal_cost={0.0: 1.0, 1.0: 3.0},
    )
    n.export_to_datarecord(tmp_path / "record")
    got = pypsa.Network(tmp_path / "record")
    assert n.equals(got.copy(), log_mode="strict")
    paths = list((tmp_path / "record").rglob("marginal_cost.parquet"))
    assert paths
    curve = pd.concat([_curve_rows(p.parent.parent, "marginal_cost") for p in paths])
    assert len(curve) == 5
    assert not curve["breakpoint"].isna().any()
    assert not curve["value"].isna().any()


def test_a_scalar_beside_a_curve_is_skipped_with_one_warning(tmp_path, caplog):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=10,
        marginal_cost={0.0: 1.0, 1.0: 3.0},
    )
    n.c.generators.static.loc["gen", "marginal_cost"] = 5.0
    with caplog.at_level("WARNING"):
        n.export_to_datarecord(tmp_path / "record")
    warned = [
        r.getMessage()
        for r in caplog.records
        if "skips" in r.getMessage()
        and "gen" in r.getMessage()
        and "marginal_cost" in r.getMessage()
    ]
    assert len(warned) == 1
    got = pypsa.Network(tmp_path / "record")
    assert got.c.generators.static.loc["gen", "marginal_cost"] == 0.0
    assert len(got.c.generators.piecewise["marginal_cost"]) == 2
