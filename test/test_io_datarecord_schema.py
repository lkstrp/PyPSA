# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the canonical datarecord schema built from the component registry."""

import pytest

datarecord = pytest.importorskip("datarecord")

import pypsa  # noqa: E402
from pypsa.components.types import all_components  # noqa: E402
from pypsa.network.io.datarecord.schema import (  # noqa: E402
    TIMESTEP_DTYPES,
    _port_stems,
    build_schema,
    declare_custom,
    port_columns,
    pypsa_name,
    record_name,
)

_EXCLUDED_TYPES = {
    "LineType",
    "TransformerType",
    "SubNetwork",
    "Network",
    "Shape",
}
_TYPE_NAMES = sorted(
    {ct.name for ct in all_components.values() if ct.name not in _EXCLUDED_TYPES}
)


@pytest.mark.parametrize("multiperiod", [False, True])
@pytest.mark.parametrize("timestep_dtype", TIMESTEP_DTYPES)
def test_build_schema_constructs(multiperiod: bool, timestep_dtype: str) -> None:
    """All four (multiperiod, timestep_dtype) variants validate cleanly."""
    schema = build_schema(multiperiod=multiperiod, timestep_dtype=timestep_dtype)
    assert schema.entity_types == frozenset(_TYPE_NAMES)
    assert "Carrier" in schema.entity_types
    assert "Shape" not in schema.entity_types


def test_carrier_is_a_group_not_an_attribute() -> None:
    """A component's `carrier` is the `carrier` group over `(entity, carrier)`,
    both drawing on the entity axis, so no type is granted a `carrier` attribute.
    """
    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    assert "carrier" not in schema.attributes
    assert schema.groups["carrier"].over == {"entity": "entity", "carrier": "entity"}
    for ctype in ("Bus", "Generator", "Link"):
        assert "carrier" not in schema.types[ctype].attributes
    assert schema.attributes["co2_emissions"].dims == frozenset({"entity", "scenario"})
    assert "co2_emissions" in schema.types["Carrier"].attributes


def test_shape_is_a_group_keyed_by_component_and_kind() -> None:
    """Shape rows are the `shape` group over `(entity, shape_type)`, carrying
    `geometry` and `shape_name` as payload granted to every type.
    """
    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    assert schema.groups["shape"].over == {
        "entity": "entity",
        "shape_type": "shape_type",
    }
    assert schema.attributes["geometry"].dims == frozenset({"shape"})
    assert schema.attributes["shape_name"].dims == frozenset({"shape"})
    for ctype in _TYPE_NAMES:
        assert "geometry" in schema.types[ctype].attributes
    assert "shape_type" in schema.dimensions


def test_every_input_attribute_is_granted() -> None:
    """Every registry input attribute of every entity type is granted by that type."""
    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    for ct in all_components.values():
        if ct.name in _EXCLUDED_TYPES:
            continue
        granted = set(schema.types[ct.name].attributes)
        stems = _port_stems(ct.name, ct.defaults)
        for attr, row in ct.defaults.iterrows():
            if attr in ("name", "carrier") or row["status"].startswith("Output"):
                continue
            stem, _port = stems.get(attr, (attr, None))
            name = record_name(ct.name, stem)
            assert name in granted, f"{ct.name}.{attr} ({name}) not granted"


def test_port_columns_link_with_bus2() -> None:
    n = pypsa.Network()
    n.add("Bus", ["b0", "b1", "b2"])
    n.add("Link", "l0", bus0="b0", bus1="b1", bus2="b2")
    c = n.components["Link"]
    cols = port_columns(c)
    assert cols["efficiency2"] == ("efficiency", "2")
    assert cols["p1"] == ("p", "1")


def test_port_columns_generator() -> None:
    n = pypsa.Network()
    c = n.components["Generator"]
    cols = port_columns(c)
    assert cols["bus"] == ("bus", "")


def test_record_name_round_trips() -> None:
    for ct in all_components.values():
        if ct.name in _EXCLUDED_TYPES:
            continue
        for attr in ct.defaults.index:
            if attr == "name":
                continue
            assert pypsa_name(ct.name, record_name(ct.name, attr)) == attr


def test_link_process_p_stays_connection_addressed() -> None:
    """Link/Process's per-port `p` joins the record-wide, connection-addressed
    `p` spec; only their own entity-wide `p` is renamed to `p_activity`.
    """
    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    assert schema.results["p"].dims == frozenset({"connection", "scenario", "timestep"})
    assert schema.results["p_activity"].dims == frozenset(
        {"entity", "scenario", "timestep"}
    )


def test_efficiency_is_connection_addressed_across_types() -> None:
    """Generator's single-port `efficiency` and Link's per-port `efficiency`
    resolve to one connection-addressed spec, not Generator's entity-addressed
    one winning by declaration order.
    """
    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    spec = schema.attributes["efficiency"]
    assert spec.dims == frozenset({"connection", "scenario", "timestep"})


def test_varying_attributes_carry_period_when_multiperiod() -> None:
    """A varying attribute's `dims` gains `period` once the schema is
    multiperiod, so its long rows carry both coordinates.
    """
    schema = build_schema(multiperiod=True, timestep_dtype="Int64")
    spec = schema.attributes["p_max_pu"]
    assert spec.dims == frozenset({"entity", "scenario", "timestep", "period"})

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    spec = schema.attributes["p_max_pu"]
    assert spec.dims == frozenset({"entity", "scenario", "timestep"})


def test_snapshot_weightings_stay_on_the_timestep_axis_when_multiperiod() -> None:
    """A snapshot weighting is addressed by `timestep` alone in both schema
    variants. `timestep` already carries `period` through `within` (the axis
    key is `(period, timestep)`), so the weighting stays a declared column of
    `dims/timestep.parquet` rather than becoming a long attribute.
    """
    for multiperiod in (True, False):
        schema = build_schema(multiperiod=multiperiod, timestep_dtype="Int64")
        spec = schema.attributes["objective"]
        assert spec.dims == frozenset({"timestep"})
        assert not spec.varying


def test_conflicting_record_name_spec_raises() -> None:
    """A name declared with two different shapes is a schema bug, not a
    silent first-wins.
    """
    import narwhals as nw
    from datarecord.schema import AttributeSpec

    from pypsa.network.io.datarecord.schema import _register

    attributes: dict = {}
    _register(
        attributes, "x", AttributeSpec(dtype=nw.Float64(), dims=frozenset({"entity"}))
    )
    with pytest.raises(ValueError, match="x"):
        _register(
            attributes,
            "x",
            AttributeSpec(dtype=nw.String(), dims=frozenset({"entity"})),
        )


def test_declare_custom_registers_and_grants() -> None:
    """A custom attribute is granted to its type with no default."""
    import narwhals as nw

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    declare_custom(
        schema, "Generator", "foo", nw.Float64(), varying=False, multiperiod=False
    )
    assert schema.attributes["foo"].dims == frozenset({"entity", "scenario"})
    assert schema.types["Generator"].attributes["foo"].default is None


def test_declare_custom_varying_adds_timestep() -> None:
    import narwhals as nw

    schema = build_schema(multiperiod=True, timestep_dtype="Int64")
    declare_custom(
        schema, "Generator", "foo", nw.Float64(), varying=True, multiperiod=True
    )
    assert schema.attributes["foo"].dims == frozenset(
        {"entity", "scenario", "timestep", "period"}
    )


def test_declare_custom_merges_dims_across_types() -> None:
    """The same custom name static on one type and time-varying on another
    merges by unioning their dims, rather than clashing.
    """
    import narwhals as nw

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    declare_custom(
        schema, "Load", "foo", nw.Float64(), varying=False, multiperiod=False
    )
    declare_custom(
        schema, "Generator", "foo", nw.Float64(), varying=True, multiperiod=False
    )
    assert schema.attributes["foo"].dims == frozenset(
        {"entity", "scenario", "timestep"}
    )
    assert "foo" in schema.types["Load"].attributes
    assert "foo" in schema.types["Generator"].attributes


def test_declare_custom_dtype_mismatch_raises_for_a_shared_file() -> None:
    """A dtype mismatch raises when the merged attribute would need one
    shared long file (one declaration time-varying).
    """
    import narwhals as nw

    from pypsa.network.io.datarecord.record import DatarecordExportError

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    declare_custom(schema, "Load", "foo", nw.Float64(), varying=True, multiperiod=False)
    with pytest.raises(DatarecordExportError, match="foo"):
        declare_custom(
            schema, "Generator", "foo", nw.String(), varying=False, multiperiod=False
        )


def test_declare_custom_dtype_mismatch_raises_even_when_both_static() -> None:
    """A dtype mismatch raises even for two purely static custom columns.

    One attribute has one dtype, and the manifest cannot say otherwise.
    """
    import narwhals as nw

    from pypsa.network.io.datarecord.record import DatarecordExportError

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    declare_custom(
        schema, "Load", "foo", nw.Float64(), varying=False, multiperiod=False
    )
    with pytest.raises(DatarecordExportError, match="foo"):
        declare_custom(
            schema, "Generator", "foo", nw.String(), varying=False, multiperiod=False
        )


def test_declare_custom_registry_clash_falls_back_to_type_prefixed_name() -> None:
    """A custom declaration that would widen a registry attribute's shape (a
    time-varying `p_nom` on Bus) is declared under `custom_fallback_name`
    instead of rewriting the registry spec.
    """
    import narwhals as nw

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    name = declare_custom(
        schema, "Bus", "p_nom", nw.Float64(), varying=True, multiperiod=False
    )
    assert name == "bus_p_nom"
    assert schema.attributes["p_nom"].dims == frozenset({"entity", "scenario"})
    assert schema.attributes["bus_p_nom"].dims == frozenset(
        {"entity", "scenario", "timestep"}
    )
    assert "bus_p_nom" in schema.types["Bus"].attributes


def test_declare_custom_static_joins_a_varying_registry_attribute() -> None:
    """A static custom column on a name the registry varies in time elsewhere
    (ac_dc_meshed's Carrier `marginal_cost`) joins that spec, its values
    being timestep-NULL rows of the same long file.
    """
    import narwhals as nw

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    name = declare_custom(
        schema,
        "Carrier",
        "marginal_cost",
        nw.Float64(),
        varying=False,
        multiperiod=False,
    )
    assert name == "marginal_cost"
    assert schema.attributes["marginal_cost"].dims == frozenset(
        {"entity", "scenario", "timestep"}
    )
    assert "marginal_cost" in schema.types["Carrier"].attributes


def test_declare_custom_entity_column_cannot_share_a_connection_file() -> None:
    """A static Carrier `efficiency` is entity-addressed, the registry's is
    connection-addressed, so it takes the fallback name.
    """
    import narwhals as nw

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    name = declare_custom(
        schema, "Carrier", "efficiency", nw.Float64(), varying=False, multiperiod=False
    )
    assert name == "carrier_efficiency"
    assert schema.attributes["efficiency"].dims == frozenset(
        {"connection", "scenario", "timestep"}
    )
    assert schema.attributes["carrier_efficiency"].dims == frozenset(
        {"entity", "scenario"}
    )


def test_declare_custom_shape_goes_on_the_shape_group() -> None:
    """A custom Shape column is payload of the `shape` group, granted to
    every type like `geometry`.
    """
    import narwhals as nw

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    name = declare_custom(
        schema, "Shape", "area_km2", nw.Float64(), varying=False, multiperiod=False
    )
    assert name == "area_km2"
    assert schema.attributes["area_km2"].dims == frozenset({"shape"})
    assert "area_km2" in schema.types["Bus"].attributes


def test_declare_custom_shape_varying_raises() -> None:
    """The shape group has no `timestep` axis to carry a time-varying column."""
    import narwhals as nw

    from pypsa.network.io.datarecord.record import DatarecordExportError

    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    with pytest.raises(DatarecordExportError, match="varying"):
        declare_custom(
            schema, "Shape", "my_ts", nw.Float64(), varying=True, multiperiod=False
        )
