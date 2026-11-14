# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging

import pytest

ccrs = pytest.importorskip("cartopy.crs")

from pypsa.geo import get_projection_from_crs  # noqa: E402


@pytest.mark.parametrize("crs", [999999, "+proj=merc", "EPSG:3035"])
def test_unresolvable_crs_falls_back_to_plate_carree(crs, caplog):
    with caplog.at_level(logging.WARNING, logger="pypsa.geo"):
        projection = get_projection_from_crs(crs)

    assert isinstance(projection, ccrs.PlateCarree)
    assert "Could not find projection" in caplog.text


def test_known_projection_is_returned_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="pypsa.geo"):
        projection = get_projection_from_crs(3035)

    assert not isinstance(projection, ccrs.PlateCarree)
    assert "Could not find projection" not in caplog.text


def test_geographic_crs_falls_back_to_plate_carree_silently(caplog):
    with caplog.at_level(logging.WARNING, logger="pypsa.geo"):
        projection = get_projection_from_crs(4326)

    assert isinstance(projection, ccrs.PlateCarree)
    assert "Could not find projection" not in caplog.text
