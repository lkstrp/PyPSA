# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the canonical datarecord schema built from the component registry."""

import sys

import pytest

datarecord = pytest.importorskip("datarecord")

if sys.version_info < (3, 12):
    pytest.skip("datarecord requires Python 3.12+", allow_module_level=True)

from pypsa.components.types import all_components  # noqa: E402
from pypsa.network.io.datarecord.schema import (  # noqa: E402
    TIMESTEP_DTYPES,
    _port_stems,
    build_schema,
    port_columns,
    pypsa_name,
    record_name,
)

_EXCLUDED_TYPES = {"LineType", "TransformerType", "SubNetwork", "Network"}
_TYPE_NAMES = sorted(
    {ct.name for ct in all_components.values() if ct.name not in _EXCLUDED_TYPES}
)


@pytest.mark.parametrize("multiperiod", [False, True])
@pytest.mark.parametrize("timestep_dtype", TIMESTEP_DTYPES)
def test_build_schema_constructs(multiperiod: bool, timestep_dtype: str) -> None:
    """All four (multiperiod, timestep_dtype) variants validate cleanly."""
    schema = build_schema(multiperiod=multiperiod, timestep_dtype=timestep_dtype)
    assert schema.entity_types == frozenset(_TYPE_NAMES)


def test_every_input_attribute_is_granted() -> None:
    """Every registry input attribute of every entity type is granted by that type."""
    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
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


def test_entity_type_enum_matches_registry() -> None:
    schema = build_schema(multiperiod=True, timestep_dtype="Datetime")
    assert schema.entity_types == frozenset(_TYPE_NAMES)


def test_port_columns_link_with_bus2() -> None:
    import pypsa

    n = pypsa.Network()
    n.add("Bus", ["b0", "b1", "b2"])
    n.add("Link", "l0", bus0="b0", bus1="b1", bus2="b2")
    c = n.components["Link"]
    cols = port_columns(c)
    assert cols["efficiency2"] == ("efficiency", "2")
    assert cols["p1"] == ("p", "1")


def test_port_columns_generator() -> None:
    import pypsa

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
