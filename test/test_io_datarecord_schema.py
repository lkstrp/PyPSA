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
    "Carrier",
    "Shape",
}
_TYPE_NAMES = sorted(
    {ct.name for ct in all_components.values() if ct.name not in _EXCLUDED_TYPES}
)


@pytest.mark.parametrize("multiperiod", [False, True])
@pytest.mark.parametrize("timestep_dtype", TIMESTEP_DTYPES)
def test_build_schema_constructs(multiperiod: bool, timestep_dtype: str) -> None:
    """All four (multiperiod, timestep_dtype) variants validate cleanly."""
    schema = build_schema(
        multiperiod=multiperiod, timestep_dtype=timestep_dtype, stochastic=False
    )
    assert schema.entity_types == frozenset(_TYPE_NAMES)
    assert "Carrier" not in schema.entity_types
    assert "Shape" not in schema.entity_types


def test_build_schema_stochastic_constructs() -> None:
    """A stochastic schema also validates cleanly, carrier/shape dims included."""
    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=True)
    assert schema.entity_types == frozenset(_TYPE_NAMES)
    assert schema.attributes["co2_emissions"].dims == frozenset({"carrier", "scenario"})
    assert schema.attributes["geometry"].dims == frozenset({"shape", "scenario"})


def test_every_input_attribute_is_granted() -> None:
    """Every registry input attribute of every entity type is granted by that type."""
    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    for ct in all_components.values():
        if ct.name in _EXCLUDED_TYPES:
            continue
        granted = set(schema.types[ct.name].attributes)
        stems = _port_stems(ct.name, ct.defaults)
        for attr, row in ct.defaults.iterrows():
            if attr == "name" or row["status"].startswith("Output"):
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
    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    assert schema.results["p"].dims == frozenset({"connection", "scenario", "timestep"})
    assert schema.results["p_activity"].dims == frozenset(
        {"entity", "scenario", "timestep"}
    )


def test_efficiency_is_connection_addressed_across_types() -> None:
    """Generator's single-port `efficiency` and Link's per-port `efficiency`
    resolve to one connection-addressed spec, not Generator's entity-addressed
    one winning by declaration order.
    """
    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    spec = schema.attributes["efficiency"]
    assert spec.dims == frozenset({"connection", "scenario", "timestep"})


def test_varying_attributes_carry_period_when_multiperiod() -> None:
    """A varying attribute's `dims` gains `period` once the schema is
    multiperiod, so its long rows carry both coordinates.
    """
    schema = build_schema(multiperiod=True, timestep_dtype="Int64", stochastic=False)
    spec = schema.attributes["p_max_pu"]
    assert spec.dims == frozenset({"entity", "scenario", "timestep", "period"})

    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    spec = schema.attributes["p_max_pu"]
    assert spec.dims == frozenset({"entity", "scenario", "timestep"})


def test_snapshot_weightings_stay_on_the_timestep_axis_when_multiperiod() -> None:
    """A snapshot weighting is addressed by `timestep` alone in both schema
    variants. `timestep` already carries `period` through `within` (the axis
    key is `(period, timestep)`), so the weighting stays a declared column of
    `dims/timestep.parquet` rather than becoming a long attribute.
    """
    for multiperiod in (True, False):
        schema = build_schema(
            multiperiod=multiperiod, timestep_dtype="Int64", stochastic=False
        )
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

    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    declare_custom(
        schema, "Generator", "foo", nw.Float64(), varying=False, multiperiod=False
    )
    assert schema.attributes["foo"].dims == frozenset({"entity", "scenario"})
    assert schema.types["Generator"].attributes["foo"].default is None


def test_declare_custom_varying_adds_timestep() -> None:
    import narwhals as nw

    schema = build_schema(multiperiod=True, timestep_dtype="Int64", stochastic=False)
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

    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
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

    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    declare_custom(schema, "Load", "foo", nw.Float64(), varying=True, multiperiod=False)
    with pytest.raises(DatarecordExportError, match="foo"):
        declare_custom(
            schema, "Generator", "foo", nw.String(), varying=False, multiperiod=False
        )


def test_declare_custom_dtype_mismatch_raises_even_when_both_static() -> None:
    """A dtype mismatch raises even for two purely static custom columns:
    one attribute has one dtype, and the manifest cannot say otherwise.
    """
    import narwhals as nw

    from pypsa.network.io.datarecord.record import DatarecordExportError

    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    declare_custom(
        schema, "Load", "foo", nw.Float64(), varying=False, multiperiod=False
    )
    with pytest.raises(DatarecordExportError, match="foo"):
        declare_custom(
            schema, "Generator", "foo", nw.String(), varying=False, multiperiod=False
        )


def test_declare_custom_registry_clash_raises() -> None:
    """Widening a registry attribute's shape through a custom declaration of
    the same name is rejected rather than silently applied.
    """
    import narwhals as nw

    from pypsa.network.io.datarecord.record import DatarecordExportError

    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    with pytest.raises(DatarecordExportError, match="Bus") as exc_info:
        declare_custom(
            schema, "Bus", "p_nom", nw.Float64(), varying=True, multiperiod=False
        )
    assert "p_nom" in str(exc_info.value)


def test_declare_custom_carrier_goes_over_its_dim() -> None:
    """A custom Carrier attribute is namespaced by its dim, so it cannot
    clash with an unrelated per-component registry attribute of the same
    name (`marginal_cost` is also Generator's).
    """
    import narwhals as nw

    schema = build_schema(multiperiod=False, timestep_dtype="Int64", stochastic=False)
    declare_custom(
        schema,
        "Carrier",
        "marginal_cost",
        nw.Float64(),
        varying=False,
        multiperiod=False,
    )
    assert schema.attributes["carrier_marginal_cost"].dims == frozenset({"carrier"})
    assert schema.attributes["marginal_cost"].dims != frozenset({"carrier"})
    assert "Carrier" not in schema.types
