# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import geopandas as gpd
import pandas as pd
from pyproj import CRS

import pypsa
from pypsa.components.components import _UNREAD
from pypsa.components.source import InMemorySource, Source
from pypsa.components.types import get as get_component_type
from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP


def test_in_memory_source_satisfies_protocol():
    assert isinstance(InMemorySource(), Source)


def test_static_matches_fresh_network():
    ctype = get_component_type("Generator")
    static = InMemorySource().static(ctype)
    expected = pypsa.Network().c.generators.static
    assert type(static) is pd.DataFrame
    assert list(static.columns) == list(expected.columns)
    pd.testing.assert_series_equal(static.dtypes, expected.dtypes)
    assert static.index.name == "name"
    assert static.empty


def test_names_match_the_static_index():
    ctype = get_component_type("Generator")
    source = InMemorySource()
    assert source.names(ctype).equals(source.static(ctype).index)


def test_unread_static_answers_names_from_the_source():
    gens = pypsa.Network().c.generators
    gens._set_source(InMemorySource())
    assert gens._static_names().empty
    assert gens._static is _UNREAD


def test_static_shape_is_geodataframe_with_default_crs():
    static = InMemorySource().static(get_component_type("Shape"))
    assert isinstance(static, gpd.GeoDataFrame)
    assert static.crs == CRS.from_epsg(DEFAULT_EPSG)
    assert static.index.name == "name"


def test_dynamic_frame_shape():
    df = InMemorySource().dynamic(get_component_type("Generator"), "p_max_pu")
    assert list(df.index) == [DEFAULT_TIMESTAMP]
    assert df.columns.empty
    assert df.index.name == "snapshot"
    assert df.columns.name == "name"


def test_varying_attrs():
    attrs = InMemorySource().varying_attrs(get_component_type("Generator"))
    assert isinstance(attrs, list)
    assert "p_max_pu" in attrs
    assert "p_nom" not in attrs


def test_piecewise_frame_shape():
    df = InMemorySource().piecewise(get_component_type("Generator"), "efficiency")
    assert df.empty
    assert df.index.name == "breakpoint"
    assert list(df.columns.names) == ["name", "attribute"]


def test_calls_return_new_frames():
    src = InMemorySource()
    ctype = get_component_type("Generator")
    assert src.static(ctype) is not src.static(ctype)
    assert src.dynamic(ctype, "p_max_pu") is not src.dynamic(ctype, "p_max_pu")
    assert src.piecewise(ctype, "efficiency") is not src.piecewise(ctype, "efficiency")


class _StubSource:
    def __init__(self):
        self.calls = []
        self._mem = InMemorySource()

    def static(self, ctype):
        self.calls.append(("static", ctype.name))
        return self._mem.static(ctype)

    def dynamic(self, ctype, attr):
        self.calls.append(("dynamic", ctype.name, attr))
        return self._mem.dynamic(ctype, attr)

    def piecewise(self, ctype, attr):
        self.calls.append(("piecewise", ctype.name, attr))
        return self._mem.piecewise(ctype, attr)

    def varying_attrs(self, ctype):
        return ["p_max_pu"]


def test_set_source_resets_containers():
    gens = pypsa.Network().c.generators
    _ = gens.static, gens.dynamic, gens.piecewise
    stub = _StubSource()
    gens._set_source(stub)
    assert stub.calls == []
    assert gens.dynamic.pending == frozenset({"p_max_pu"})
    _ = gens.static
    _ = gens.dynamic["p_max_pu"]
    _ = gens.piecewise["efficiency"]
    assert ("static", "Generator") in stub.calls
    assert ("dynamic", "Generator", "p_max_pu") in stub.calls
    assert ("piecewise", "Generator", "efficiency") in stub.calls
