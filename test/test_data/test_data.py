"""
Unit testing for ensemble averaging
"""

from code import data
import numpy as np
import pytest


def _member(T, P, lat, lon):
    return {'T': T, 'P': P, 'lat': lat, 'lon': lon, 'years': T.shape[0]}


def test_ensemble_mean():
    """Test that ensemble_mean averages T and P across members sharing inst/model/scenario."""
    rng = np.random.RandomState(0)
    shape = (3, 12, 4, 6)
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(0, 360, 6)

    T1 = rng.randn(*shape) + 15
    P1 = np.abs(rng.randn(*shape) * 10 + 80)
    T2 = rng.randn(*shape) + 15
    P2 = np.abs(rng.randn(*shape) * 10 + 80)

    loaded = {
        'MOHC_HadGEM3-GC31-LL_4K_r1i1p1f1': _member(T1, P1, lat, lon),
        'MOHC_HadGEM3-GC31-LL_4K_r2i1p1f1': _member(T2, P2, lat, lon),
    }

    result = data.ensemble_mean(loaded)

    assert len(result) == 1
    key = 'MOHC_HadGEM3-GC31-LL_4K'
    assert key in result

    np.testing.assert_allclose(result[key]['T'], (T1 + T2) / 2)
    np.testing.assert_allclose(result[key]['P'], (P1 + P2) / 2)
    np.testing.assert_array_equal(result[key]['lat'], lat)
    np.testing.assert_array_equal(result[key]['lon'], lon)
    assert result[key]['years'] == 3


def test_ensemble_mean_truncates_year_mismatch():
    """Test that ensemble_mean truncates members to the shortest year count and averages all of them."""
    rng = np.random.RandomState(1)
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(0, 360, 6)

    T1 = rng.randn(3, 12, 4, 6) + 15
    P1 = np.abs(rng.randn(3, 12, 4, 6) * 10 + 80)
    # Different number of years
    T2 = rng.randn(5, 12, 4, 6) + 15
    P2 = np.abs(rng.randn(5, 12, 4, 6) * 10 + 80)

    loaded = {
        'MOHC_HadGEM3-GC31-LL_4K_r1i1p1f1': _member(T1, P1, lat, lon),
        'MOHC_HadGEM3-GC31-LL_4K_r2i1p1f1': _member(T2, P2, lat, lon),
    }

    result = data.ensemble_mean(loaded)

    key = 'MOHC_HadGEM3-GC31-LL_4K'
    # Both members contribute, the longer one truncated to 3 years
    np.testing.assert_allclose(result[key]['T'], (T1 + T2[:3]) / 2)
    np.testing.assert_allclose(result[key]['P'], (P1 + P2[:3]) / 2)
    assert result[key]['years'] == 3


def test_ensemble_mean_averages_q():
    """Test that ensemble_mean averages Q alongside T and P when present."""
    rng = np.random.RandomState(3)
    shape = (2, 12, 4, 6)
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(0, 360, 6)

    members = {}
    Qs = []
    for ens in ('r1i1p1f1', 'r2i1p1f1'):
        m = _member(rng.randn(*shape) + 15, np.abs(rng.randn(*shape) * 10 + 80), lat, lon)
        m['Q'] = np.abs(rng.randn(*shape) * 1e-3 + 0.01)
        Qs.append(m['Q'])
        members[f'MOHC_HadGEM3-GC31-LL_0K_{ens}'] = m

    result = data.ensemble_mean(members)
    np.testing.assert_allclose(result['MOHC_HadGEM3-GC31-LL_0K']['Q'], (Qs[0] + Qs[1]) / 2)


def test_ensemble_mean_rejects_grid_mismatch():
    """Test that ensemble_mean raises when members are on different spatial grids."""
    rng = np.random.RandomState(2)
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(0, 360, 6)

    loaded = {
        'MOHC_HadGEM3-GC31-LL_4K_r1i1p1f1': _member(rng.randn(3, 12, 4, 6) + 15, np.abs(rng.randn(3, 12, 4, 6) * 10 + 80), lat, lon),
        # Different spatial grid
        'MOHC_HadGEM3-GC31-LL_4K_r2i1p1f1': _member(rng.randn(3, 12, 8, 12) + 15, np.abs(rng.randn(3, 12, 8, 12) * 10 + 80), lat, lon),
    }

    with pytest.raises(ValueError, match='different grids'):
        data.ensemble_mean(loaded)
