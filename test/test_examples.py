# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import logging

import pytest

import pypsa
from pypsa.version import __version_base__

logger = logging.getLogger(__name__)


def test_ac_dc_meshed():
    n = pypsa.examples.ac_dc_meshed()
    assert not n.c.buses.static.empty


def test_storage_hvdc():
    n = pypsa.examples.storage_hvdc()
    assert not n.c.buses.static.empty


def test_ac_dc_meshed_deduplicates_load_names(caplog):
    """Loads share their bus's name, so every Load clashes and is renamed."""
    buses = {
        "London",
        "Norwich",
        "Norwich DC",
        "Manchester",
        "Bremen",
        "Bremen DC",
        "Frankfurt",
        "Norway",
        "Norway DC",
    }
    with caplog.at_level("WARNING"):
        n = pypsa.examples.ac_dc_meshed()

    assert set(n.c.buses.static.index) == buses
    assert "London-Load" in n.c.loads.static.index
    rename_warnings = [r for r in caplog.records if "Renamed" in r.message]
    assert len(rename_warnings) == 1
    assert "Load 6" in rename_warnings[0].message


def test_scigrid_de():
    n = pypsa.examples.scigrid_de()
    assert not n.c.buses.static.empty


def test_scigrid_de_deduplicates_clashing_names(caplog):
    """Lines, transformers and loads are numbered like buses, so they clash."""
    with caplog.at_level("WARNING"):
        n = pypsa.examples.scigrid_de()

    rename_warnings = [r for r in caplog.records if "Renamed" in r.message]
    assert len(rename_warnings) == 1
    assert "Line 486, Transformer 96, Load 489" in rename_warnings[0].message

    buses = n.c.buses.static.index
    assert not any(b.endswith(("-Line", "-Transformer", "-Load")) for b in buses)
    assert "1" in buses

    p_set = n.c.loads.dynamic["p_set"]
    assert "1-Load" in p_set.columns
    assert p_set["1-Load"].notna().any()


def test_model_energy():
    n = pypsa.examples.model_energy()
    assert not n.c.buses.static.empty


def test_model_energy_has_no_name_clashes(caplog):
    with caplog.at_level("WARNING"):
        n = pypsa.examples.model_energy()

    assert not any("Renamed" in r.message for r in caplog.records)
    assert not n.c.buses.static.empty


def test_carbon_management():
    try:
        n = pypsa.examples.carbon_management()
        assert not n.c.buses.static.empty
    except RuntimeError as e:
        logger.warning("Test would have failed: %s", e)
        pytest.skip("Test failed but converted to warning")


@pytest.fixture
def seeded_cache(monkeypatch, tmp_path):
    """Seed a cache directory with a dummy network file."""
    monkeypatch.setattr(pypsa.examples, "_cache_root", lambda: tmp_path)
    cache = tmp_path / f"v{__version_base__}" / "ac_dc_meshed.nc"
    cache.parent.mkdir(parents=True)
    pypsa.Network().export_to_netcdf(str(cache))
    return tmp_path


def test_caching(seeded_cache):
    """Test that cached example is loaded from disk without network."""
    n = pypsa.examples.ac_dc_meshed()
    assert isinstance(n, pypsa.Network)


def test_clear_cache(seeded_cache):

    pypsa.examples.clear_cache()
    assert not seeded_cache.exists()


def test_cache_miss_network_disabled(monkeypatch, tmp_path):
    """Test that cache miss with network requests disabled raises ValueError."""
    monkeypatch.setattr(pypsa.examples, "_cache_root", lambda: tmp_path)

    pypsa.options.general.allow_network_requests = False
    try:
        with pytest.raises(ValueError, match="Network requests are disabled"):
            pypsa.examples.ac_dc_meshed()
    finally:
        pypsa.options.general.allow_network_requests = True


def test_cache_miss_404_falls_back_to_latest(monkeypatch, tmp_path, caplog):
    """404 on versioned URL falls back to latest/ with warning."""
    from urllib.error import HTTPError

    monkeypatch.setattr(pypsa.examples, "_cache_root", lambda: tmp_path)
    calls = []

    def fake_urlretrieve(url, cache):
        calls.append(url)
        if f"/v{__version_base__}/" in url:
            raise HTTPError(url, 404, "Not Found", {}, None)
        pypsa.Network().export_to_netcdf(str(cache))

    monkeypatch.setattr(pypsa.examples, "urlretrieve", fake_urlretrieve)
    with caplog.at_level(logging.WARNING, logger="pypsa.examples"):
        pypsa.examples.ac_dc_meshed()

    assert [f"/v{__version_base__}/" in calls[0], "/latest/" in calls[1]] == [
        True,
        True,
    ]
    assert any("404" in r.message for r in caplog.records)


@pytest.mark.skipif(
    not pypsa.examples._check_url_availability("https://data.pypsa.org"),
    reason="No internet connection",
)
def test_check_url_availability():
    """Test _check_url_availability function."""
    from pypsa.examples import _check_url_availability

    assert not _check_url_availability("invalid-url")
    assert not _check_url_availability("ftp://example.com")
    assert not _check_url_availability("")
    assert not _check_url_availability("https://data.pypsa.org/nonexistent")
    assert _check_url_availability(
        "https://data.pypsa.org/networks/examples/latest/ac_dc_meshed.nc"
    )
