# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import geopandas as gpd
import numpy as np
import pandas as pd

import pypsa
from pypsa.network.io.mixin import _ordered_static, _typed_static


def _attrs(n, name):
    return n.components[name]["defaults"]


def test_typed_static_defaults_and_casts():
    n = pypsa.Network()
    attrs = _attrs(n, "Generator")
    df = pd.DataFrame(
        {"p_nom": [1, 2], "carrier": ["a", np.nan], "p": [0.5, 0.6]},
        index=["g1", "g2"],
    )
    static, series_cols = _typed_static(df, attrs)
    assert static["p_nom"].dtype == attrs.at["p_nom", "typ"]
    assert static["carrier"].tolist() == ["a", ""]
    assert (
        static["marginal_cost"].tolist() == [attrs.at["marginal_cost", "default"]] * 2
    )
    assert "p" not in static.columns
    assert list(series_cols.columns) == ["p"]


def test_ordered_static_order_and_index_names():
    n = pypsa.Network()
    attrs = _attrs(n, "Generator")
    static, _ = _typed_static(pd.DataFrame({"p_nom": [1.0]}, index=["g"]), attrs)
    out = _ordered_static(static[static.columns[::-1]], attrs, "Generator", None)
    assert list(out.columns) == list(n.c.generators.static.columns)
    assert out.index.names == ["name"]

    mi = pd.MultiIndex.from_tuples([("s", "g")])
    static, _ = _typed_static(pd.DataFrame({"p_nom": [1.0]}, index=mi), attrs)
    out = _ordered_static(static, attrs, "Generator", None)
    assert list(out.index.names) == ["scenario", "name"]


def test_ordered_static_shape_geodataframe():
    n = pypsa.Network()
    attrs = _attrs(n, "Shape")
    static, _ = _typed_static(pd.DataFrame(index=["s"]), attrs)
    out = _ordered_static(static, attrs, "Shape", "EPSG:4326")
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == "EPSG:4326"
