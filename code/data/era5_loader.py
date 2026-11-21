"""
ERA5 loading and variable conversion to standardized arrays.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from .loader_utils import concat_time, load_land_mask, regrid_to_target


def load_era5(path, resolution='1x1', start_year=None, end_year=None, land_only=True):
    """Load ERA5 monthly tas and pr from {path}/{var}/mon/*.nc."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    land_mask = None
    if land_only:
        land_mask = load_land_mask(data_path / 'lsm' / 'mon' / f'landseamask_{resolution}.nc').values

    datasets = {}
    for var in ('tas', 'pr'):
        files = [f for f in sorted((data_path / var / 'mon').glob('*.nc')) if resolution in f.name]
        if not files:
            raise ValueError(f"No ERA5 {var} monthly files at resolution '{resolution}' in {path}")
        datasets[var] = concat_time([xr.open_dataset(f)[var] for f in files])

    return _to_arrays(datasets, start_year=start_year, end_year=end_year, land_mask=land_mask)


def _time_index(da):
    """(years, months) integer arrays for a time axis; handles cftime."""
    times = da.time.values
    try:
        idx = pd.DatetimeIndex(times)
        return np.asarray(idx.year), np.asarray(idx.month)
    except TypeError:
        return np.array([t.year for t in times]), np.array([t.month for t in times])


def _days_in_month(da):
    """Days per month for a time axis; handles cftime."""
    times = da.time.values
    try:
        return pd.DatetimeIndex(times).days_in_month.values.astype(np.float64)
    except TypeError:
        return np.array([t.daysinmonth for t in times], dtype=np.float64)


def _surface_humidity(datasets, tas):
    """Surface specific humidity: huss directly, derived from tdas+ps, else NaN."""
    if 'huss' in datasets:
        return datasets['huss']
    if 'tdas' in datasets and 'ps' in datasets:
        # tdas and ps can come from different file sets so combine only shared timestamps;
        # the caller then label-aligns the result to tas/pr, never positionally.
        common = pd.Index(datasets['tdas'].time.values).intersection(pd.Index(datasets['ps'].time.values))
        tdas, ps_da = datasets['tdas'].sel(time=common), datasets['ps'].sel(time=common)
        # Tetens e_sat at the dew point (ECMWF constants), then q = eps*e / (ps - (1-eps)*e)
        td = tdas.values
        if np.nanmean(td) < 100:  # Celsius
            td = td + 273.15
        ps = ps_da.values
        if np.nanmean(ps) < 2000:  # hPa
            ps = ps * 100
        eps = 287.0597 / 461.5250
        e = 611.21 * np.exp(17.502 * (td - 273.16) / (td - 32.19))
        q = eps * e / (ps - (1 - eps) * e)
        return xr.DataArray(q, dims=tdas.dims, coords=tdas.coords, attrs={'units': 'kg kg-1'})
    return xr.full_like(tas, np.nan)


def _to_arrays(datasets, start_year=None, end_year=None, land_mask=None, target_lat=None, target_lon=None):
    """Standardize tas/pr(/humidity) DataArrays into T (deg C), P (mm/month), and Q (kg/kg,
    NaN if unavailable), each (years, 12, lat, lon) over whole calendar years."""
    tas, pr, huss = datasets['tas'], datasets['pr'], _surface_humidity(datasets, datasets['tas'])

    # tas and pr can come from different file sets so we retain only shared timestamps.
    common = pd.Index(tas.time.values).intersection(pd.Index(pr.time.values))
    fields = [tas.sel(time=common), pr.sel(time=common), huss.reindex(time=common)]

    # Rename to lat/lon and regrid
    for i, da in enumerate(fields):
        rename = {k: v for k, v in (('latitude', 'lat'), ('longitude', 'lon')) if k in da.dims}
        if rename:
            da = da.rename(rename)
        if target_lat is not None and target_lon is not None:
            da = regrid_to_target(da, target_lat, target_lon)
        fields[i] = da

    # Year filter then cut to whole calendar years (first January to last December)
    years, months = _time_index(fields[0])
    if start_year is not None or end_year is not None:
        keep = np.ones(years.size, dtype=bool)
        if start_year is not None:
            keep &= years >= start_year
        if end_year is not None:
            keep &= years <= end_year
        fields = [da.isel(time=keep) for da in fields]
        months = _time_index(fields[0])[1]
    jan, dec = np.flatnonzero(months == 1), np.flatnonzero(months == 12)
    if jan.size == 0 or dec.size == 0 or dec[-1] < jan[0]:
        raise ValueError("No complete calendar years in the requested range")
    tas, pr, huss = (da.isel(time=slice(jan[0], dec[-1] + 1)).transpose('time', 'lat', 'lon') for da in fields)
    if len(tas.time) % 12:
        raise ValueError(f"{len(tas.time)} months left after cutting to whole years; time axis has gaps")

    # K to deg C; m/day or kg/m2/s to mm/month, with calendar-aware days per month
    T = tas.values
    if tas.attrs.get('units', '') == 'K' or np.nanmean(T) > 100:
        T = T - 273.15
    P = pr.values
    pr_units = pr.attrs.get('units', '')
    if pr_units == 'm' or 'kg' in pr_units or 's-1' in pr_units or np.nanmean(P) < 0.01:
        days = _days_in_month(pr)[:, None, None]
        P = P * (1000 * days if pr_units == 'm' else 86400 * days)

    n_years = len(tas.time) // 12
    shape = (n_years, 12) + T.shape[1:]
    T, P, Q = T.reshape(shape).copy(), P.reshape(shape).copy(), huss.values.reshape(shape).copy()

    if land_mask is not None:
        ocean = land_mask < 0.5
        T[:, :, ocean] = np.nan
        P[:, :, ocean] = np.nan
        Q[:, :, ocean] = np.nan

    return {'T': T, 'P': P, 'Q': Q, 'lat': tas.lat.values, 'lon': tas.lon.values, 'years': n_years}
