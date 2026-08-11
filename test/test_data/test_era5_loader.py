"""
Unit testing for the ERA5 loader functionality
"""

from code import data
import numpy as np
import xarray as xr
import pandas as pd
import pytest
from code.data.era5_loader import _to_arrays
import earth2grid

def test_era5_loader_invalid_path():
    with pytest.raises(FileNotFoundError, match="Path not found"):
        data.load_era5('/nonexistent/path')

def test_to_arrays_regridding():
    # Create synthetic data (2 years of monthly data on a 4x8 grid)
    n_months = 24
    lat_src = np.array([-45.0, -15.0, 15.0, 45.0])
    lon_src = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    times = pd.date_range('2000-01-01', periods=n_months, freq='MS')

    np.random.seed(42)
    tas_vals = 280.0 + np.random.randn(n_months, len(lat_src), len(lon_src))
    pr_vals = 1e-5 + np.abs(np.random.randn(n_months, len(lat_src), len(lon_src))) * 1e-6

    tas = xr.DataArray(tas_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'K'})
    pr = xr.DataArray(pr_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'kg m-2 s-1'})

    # Target grid: 6x12, a different lat/lon spacing from the source so regridding actually runs
    target_lat = np.linspace(-40, 40, 6)
    target_lon = np.linspace(10, 310, 12)

    result = _to_arrays({'tas': tas, 'pr': pr}, target_lat=target_lat, target_lon=target_lon)

    assert result['T'].shape == (2, 12, 6, 12), f"Expected (2, 12, 6, 12), got {result['T'].shape}."
    assert result['P'].shape == (2, 12, 6, 12), f"Expected (2, 12, 6, 12), got {result['P'].shape}."
    assert np.array_equal(result['lat'], target_lat)
    assert np.array_equal(result['lon'], target_lon)
    assert result['years'] == 2


def test_to_arrays_lon_convention_neg180_to_0_360():
    """Test that source on -180-180 is converted to 0-360 before regridding."""
    n_months = 12
    lat_src = np.array([-30.0, 0.0, 30.0])
    # Source on -180-180
    lon_src = np.array([-135.0, -45.0, 45.0, 135.0])
    times = pd.date_range('2000-01-01', periods=n_months, freq='MS')

    # Temperature field with a longitude gradient so we can verify correct mapping
    lon_field = lon_src[np.newaxis, np.newaxis, :]  # (1, 1, 4)
    tas_vals = np.broadcast_to(280.0 + lon_field * 0.01, (n_months, len(lat_src), len(lon_src))).copy()
    pr_vals = np.full((n_months, len(lat_src), len(lon_src)), 1e-5)

    tas = xr.DataArray(tas_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'K'})
    pr = xr.DataArray(pr_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'kg m-2 s-1'})

    # Target on 0-360, within the interior of the source range so interp doesn't extrapolate
    target_lat = np.array([0.0])
    target_lon = np.array([45.0, 135.0, 225.0, 315.0])

    result = _to_arrays({'tas': tas, 'pr': pr}, target_lat=target_lat, target_lon=target_lon)

    assert result['T'].shape == (1, 12, 1, 4), f"Expected (1, 12, 1, 4), got {result['T'].shape}."
    # The interior points (45 and 135 on 0-360) should have valid (non-NaN) values
    assert not np.isnan(result['T'][:, :, :, 0]).any(), "lon=45 should not be NaN."
    assert not np.isnan(result['T'][:, :, :, 1]).any(), "lon=135 should not be NaN."


def test_to_arrays_lon_convention_0_360_to_neg180():
    """Test that source on 0-360 is converted to -180-180 before regridding."""
    n_months = 12
    lat_src = np.array([-30.0, 0.0, 30.0])
    # Source on 0..360
    lon_src = np.array([45.0, 135.0, 225.0, 315.0])
    times = pd.date_range('2000-01-01', periods=n_months, freq='MS')

    tas_vals = np.full((n_months, len(lat_src), len(lon_src)), 280.0)
    pr_vals = np.full((n_months, len(lat_src), len(lon_src)), 1e-5)

    tas = xr.DataArray(tas_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'K'})
    pr = xr.DataArray(pr_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'kg m-2 s-1'})

    # Target on -180-180, within the interior of the source range
    target_lat = np.array([0.0])
    target_lon = np.array([-135.0, -45.0, 45.0, 135.0])

    result = _to_arrays({'tas': tas, 'pr': pr}, target_lat=target_lat, target_lon=target_lon)

    assert result['T'].shape == (1, 12, 1, 4), f"Expected (1, 12, 1, 4), got {result['T'].shape}."
    # The interior points (-135 and -45 on -180-180) should have valid (non-NaN) values
    assert not np.isnan(result['T'][:, :, :, 0]).any(), "lon=-135 should not be NaN"
    assert not np.isnan(result['T'][:, :, :, 1]).any(), "lon=-45 should not be NaN"


def test_to_arrays_no_regridding():
    """Test that _to_arrays preserves original grid when no target is provided."""
    n_months = 12
    lat_src = np.array([-45.0, -15.0, 15.0, 45.0])
    lon_src = np.array([0.0, 90.0, 180.0, 270.0])
    times = pd.date_range('2000-01-01', periods=n_months, freq='MS')

    tas_vals = 280.0 + np.zeros((n_months, len(lat_src), len(lon_src)))
    pr_vals = 1e-5 + np.zeros((n_months, len(lat_src), len(lon_src)))

    tas = xr.DataArray(tas_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'K'})
    pr = xr.DataArray(pr_vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'kg m-2 s-1'})

    result = _to_arrays({'tas': tas, 'pr': pr})

    assert result['T'].shape == (1, 12, 4, 4), f"Expected (1, 12, 4, 4), got {result['T'].shape}."
    assert np.array_equal(result['lat'], lat_src)
    assert np.array_equal(result['lon'], lon_src)


def test_to_arrays_healpix_grid():
    n_months = 12
    # Nside=2 (level=1) -> 48 pixels. This is the smallest practical HEALPix grid
    nside = 2
    n_pixels = 12 * nside * nside  # 48
    times = pd.date_range('2000-01-01', periods=n_months, freq='MS')

    # Get pixel centre coordinates from earth2grid in RING order
    hpx = earth2grid.healpix.Grid(level=1, pixel_order=earth2grid.healpix.PixelOrder.RING)
    pix_lat, pix_lon = hpx.lat, hpx.lon  # degrees, shape (48,)

    # Uniform temperature field (300 K everywhere)
    tas_vals = np.full((n_months, n_pixels), 300.0)
    pr_vals = np.full((n_months, n_pixels), 1e-5)

    tas = xr.DataArray(tas_vals, dims=['time', 'i'], coords={'time': times, 'latitude': ('i', pix_lat), 'longitude': ('i', pix_lon)}, attrs={'units': 'K'})
    pr = xr.DataArray(pr_vals, dims=['time', 'i'], coords={'time': times, 'latitude': ('i', pix_lat), 'longitude': ('i', pix_lon)}, attrs={'units': 'kg m-2 s-1'})

    target_lat = np.array([-30.0, 0.0, 30.0])
    target_lon = np.array([60.0, 180.0, 300.0])

    result = _to_arrays({'tas': tas, 'pr': pr}, target_lat=target_lat, target_lon=target_lon)

    assert result['T'].shape == (1, 12, 3, 3), f"Expected (1, 12, 3, 3), got {result['T'].shape}."
    assert result['P'].shape == (1, 12, 3, 3), f"Expected (1, 12, 3, 3), got {result['P'].shape}."
    assert np.array_equal(result['lat'], target_lat)
    assert np.array_equal(result['lon'], target_lon)
    # Uniform 300K -> ~26.85 C everywhere
    assert np.allclose(result['T'], 300.0 - 273.15, atol=0.5), f"Expected ~26.85 C, got mean {result['T'].mean():.2f}."


def test_to_arrays_healpix_nonuniform():
    """HEALPix regrid must map a spatially varying field to the right hemisphere, not just pass a uniform value.

    A field that is warm in the north and cold in the south must, after regridding,
    be warm at the northern target latitude and cold at the southern one. A uniform
    field would pass under any pixel permutation; this catches a RING/NEST mixup.
    """
    n_months = 12
    nside = 16
    n_pixels = 12 * nside * nside
    times = pd.date_range('2000-01-01', periods=n_months, freq='MS')

    hpx = earth2grid.healpix.Grid(level=4, pixel_order=earth2grid.healpix.PixelOrder.RING)
    pix_lat, pix_lon = hpx.lat, hpx.lon

    # Temperature increases linearly with latitude: -30 C at south pole, +30 C at north pole.
    lat_field_C = 30.0 * pix_lat / 90.0
    tas_vals = np.broadcast_to((lat_field_C + 273.15)[None, :], (n_months, n_pixels))
    tas = xr.DataArray(tas_vals, dims=['time', 'i'], coords={'time': times, 'latitude': ('i', pix_lat), 'longitude': ('i', pix_lon)}, attrs={'units': 'K'})
    pr = xr.DataArray(np.full((n_months, n_pixels), 1e-5), dims=['time', 'i'], coords=tas.coords, attrs={'units': 'kg m-2 s-1'})

    target_lat = np.array([-60.0, 0.0, 60.0])
    target_lon = np.array([0.0, 120.0, 240.0])

    result = _to_arrays({'tas': tas, 'pr': pr}, target_lat=target_lat, target_lon=target_lon)

    row_means = np.nanmean(result['T'][0], axis=(0, 2))  # mean over months and lon, per target lat
    assert row_means[0] < row_means[1] < row_means[2], f"North-warm gradient not preserved after regrid: {row_means}"
    assert row_means[0] < -10 and row_means[2] > 10, f"Hemispheres not resolved: {row_means}"


def _year_stamped(times, lat, lon, units):
    """Field whose value at every point equals its own calendar year, so misalignment is visible."""
    year_field = np.array([t.year for t in pd.DatetimeIndex(times)], dtype=np.float64)
    vals = np.broadcast_to(year_field[:, None, None], (len(times), len(lat), len(lon))).copy()
    return xr.DataArray(vals, dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat, 'lon': lon}, attrs={'units': units})


def test_to_arrays_aligns_tas_and_pr_by_timestamp():
    """tas and pr come from different file sets, so they must be intersected on timestamps, not sliced positionally.

    Regression test: pr here starts in 1940 while tas starts in 1979. Positional slicing
    would silently pair 1940-1942 precipitation with 1979-1981 temperature.
    """
    lat, lon = np.array([-30.0, 0.0, 30.0]), np.array([0.0, 90.0, 180.0, 270.0])
    tas_times = pd.date_range('1979-01-01', periods=36, freq='MS')  # 1979-1981
    pr_times = pd.date_range('1940-01-01', periods=504, freq='MS')  # 1940-1981, back-extended

    tas = xr.DataArray(280.0 + np.zeros((36, 3, 4)), dims=['time', 'lat', 'lon'], coords={'time': tas_times, 'lat': lat, 'lon': lon}, attrs={'units': 'K'})
    pr = _year_stamped(pr_times, lat, lon, 'mm/month')

    result = _to_arrays({'tas': tas, 'pr': pr})

    assert result['years'] == 3, f"Expected the 3 shared years, got {result['years']}."
    assert result['P'][0].min() == 1979, f"First P year should be 1979, got {result['P'][0].min()} (positional slicing would give 1940)."
    for i, year in enumerate((1979, 1980, 1981)):
        assert np.all(result['P'][i] == year), f"P year index {i} should be {year}, got {np.unique(result['P'][i])}."


def test_to_arrays_intersection_trims_both_ends():
    """Timestamps present in only one of tas or pr are dropped from both."""
    lat, lon = np.array([0.0]), np.array([0.0, 180.0])
    tas_times = pd.date_range('1979-01-01', periods=84, freq='MS')  # 1979-1985
    pr_times = pd.date_range('1982-01-01', periods=108, freq='MS')  # 1982-1990

    tas_years = np.array([t.year for t in pd.DatetimeIndex(tas_times)], dtype=np.float64)
    tas_vals = np.broadcast_to((tas_years + 273.15)[:, None, None], (84, 1, 2)).copy()
    tas = xr.DataArray(tas_vals, dims=['time', 'lat', 'lon'], coords={'time': tas_times, 'lat': lat, 'lon': lon}, attrs={'units': 'K'})
    pr = _year_stamped(pr_times, lat, lon, 'mm/month')

    result = _to_arrays({'tas': tas, 'pr': pr})

    assert result['years'] == 4, f"Expected the 4 overlapping years 1982-1985, got {result['years']}."
    for i, year in enumerate((1982, 1983, 1984, 1985)):
        assert np.all(result['P'][i] == year), f"P year index {i} should be {year}."
        assert np.allclose(result['T'][i], year), f"T year index {i} should be {year}."


def test_to_arrays_year_filter():
    """Test that start_year/end_year keep only whole calendar years inside the range."""
    n_months = 42  # 1999-07 through 2002-12
    lat_src = np.array([-30.0, 0.0, 30.0])
    lon_src = np.array([0.0, 90.0, 180.0, 270.0])
    times = pd.date_range('1999-07-01', periods=n_months, freq='MS')

    tas = xr.DataArray(280.0 + np.zeros((n_months, 3, 4)), dims=['time', 'lat', 'lon'], coords={'time': times, 'lat': lat_src, 'lon': lon_src}, attrs={'units': 'K'})
    pr = xr.DataArray(np.full((n_months, 3, 4), 1e-5), dims=['time', 'lat', 'lon'], coords=tas.coords, attrs={'units': 'kg m-2 s-1'})

    result = _to_arrays({'tas': tas, 'pr': pr}, start_year=2000, end_year=2001)
    assert result['years'] == 2, f"Expected 2 whole years, got {result['years']}."

    # No filter: the partial 1999 must be dropped, leaving 2000-2002.
    result = _to_arrays({'tas': tas, 'pr': pr})
    assert result['years'] == 3, f"Expected 3 whole years, got {result['years']}."
