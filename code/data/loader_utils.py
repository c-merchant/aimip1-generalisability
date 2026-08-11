"""
Shared loader helpers: time concatenation, land-mask loading, regridding, and humidity fallback.
"""

import math
import numpy as np
import xarray as xr
import healpy as hp

# Dims that are not vertical levels
SURFACE_DROP_DIMS = ('time', 'lat', 'lon', 'latitude', 'longitude')


def concat_time(chunks):
    """Concatenate per-file chunks along time, sort, and drop duplicate timestamps."""
    out = chunks[0] if len(chunks) == 1 else xr.concat(chunks, dim='time')
    return out.sortby('time').drop_duplicates(dim='time')


def load_land_mask(path):
    """Fractional land-sea mask (0-1) as a 2D DataArray with lat/lon dims."""
    ds = xr.open_dataset(path)
    rename = {k: v for k, v in [('latitude', 'lat'), ('longitude', 'lon')] if k in ds.dims}
    if rename:
        ds = ds.rename(rename)
    return ds['lsm'].squeeze()


def select_surface_level(hus):
    """Reduce 3D hus to its surface level; unchanged if already 2D."""
    level_dim = next((d for d in hus.dims if d not in SURFACE_DROP_DIMS), None)
    if level_dim is None:
        return hus
    if level_dim in hus.coords:
        return hus.sel({level_dim: float(hus[level_dim].max())})
    return hus.isel({level_dim: -1})


def add_humidity_datasets(datasets, file_lists, concat):
    """Add available humidity inputs to datasets with preference for huss then tdas+ps then 3D hus."""
    if 'huss' in file_lists:
        datasets['huss'] = concat([xr.open_dataset(f)['huss'] for f in file_lists['huss']])
    elif 'tdas' in file_lists and 'ps' in file_lists:
        for var in ['tdas', 'ps']:
            datasets[var] = concat([xr.open_dataset(f)[var] for f in file_lists[var]])
    elif 'hus' in file_lists:
        hus = concat([xr.open_dataset(f)['hus'] for f in file_lists['hus']])
        datasets['huss'] = select_surface_level(hus)
    return datasets


def _hpx3d_to_nest(data_3d):
    """DLESyM (face, h, w) cube to 1D NEST-ordered HEALPix array."""
    nfaces, nside, _ = data_3d.shape
    data_3d = np.flip(data_3d, axis=(1, 2))
    npix = nfaces * nside * nside
    y_all, x_all = np.mgrid[0:nside, 0:nside]
    y_flat, x_flat = y_all.ravel(), x_all.ravel()
    nbits = int(np.log2(nside))
    local_idx = np.zeros(nside * nside, dtype=np.int64)
    for bit in range(nbits):
        local_idx |= ((y_flat >> bit) & 1).astype(np.int64) << (2 * bit)
        local_idx |= ((x_flat >> bit) & 1).astype(np.int64) << (2 * bit + 1)
    hpx1d = np.empty(npix, dtype=data_3d.dtype)
    for f in range(nfaces):
        hpx1d[f * nside * nside + local_idx] = data_3d[f].ravel()
    return hpx1d


def regrid_to_target(da, target_lat, target_lon):
    """Interpolate onto a regular (target_lat, target_lon) grid. HEALPix sources (flat RING
    or a DLESyM face/h/w NEST cube) via healpy; regular grids are matched to the target
    longitude convention and padded with one wrapped column for periodic interpolation."""
    rename = {k: v for k, v in [('latitude', 'lat'), ('longitude', 'lon')] if k in da.dims}
    if rename:
        da = da.rename(rename)

    # HEALPix: no lat/lon dims
    if 'lat' not in da.dims and 'lon' not in da.dims:
        vals = da.values
        n_time = vals.shape[0]
        is_cube = 'face' in da.dims and da.sizes.get('face', 0) == 12
        if is_cube:
            nside = da.sizes.get('height', da.sizes.get('width'))
            flat = np.array([_hpx3d_to_nest(vals[t]) for t in range(n_time)])
            nest = True
        else:
            flat = vals.reshape(n_time, -1)
            nside = int(math.sqrt(flat.shape[1] / 12))
            nest = False
        n_pixels = flat.shape[1]
        if 12 * nside * nside != n_pixels or (nside & (nside - 1)) != 0:
            raise ValueError(f"Not a valid HEALPix grid.")
        lon2d, lat2d = np.meshgrid(target_lon, target_lat)
        theta = np.deg2rad(90.0 - lat2d)
        phi = np.deg2rad(lon2d)
        nlat, nlon = len(target_lat), len(target_lon)
        regridded = np.empty((n_time, nlat, nlon))
        for t in range(n_time):
            regridded[t] = hp.get_interp_val(flat[t], theta, phi, nest=nest)
        return xr.DataArray(regridded, dims=['time', 'lat', 'lon'], coords={'time': da.time.values, 'lat': target_lat, 'lon': target_lon}, attrs=da.attrs)

    # Match longitude convention to the target
    src_lon = da.lon.values
    src_is_neg180 = float(src_lon.min()) < -0.5
    tgt_is_neg180 = float(np.min(target_lon)) < -0.5
    if src_is_neg180 and not tgt_is_neg180:
        da = da.assign_coords(lon=src_lon % 360).sortby('lon')
    elif not src_is_neg180 and tgt_is_neg180:
        da = da.assign_coords(lon=((src_lon + 180) % 360) - 180).sortby('lon')

    # Wrapped pad column on each side
    lon = da.lon.values
    lo = da.isel(lon=-1).assign_coords(lon=lon[-1] - 360.0)
    hi = da.isel(lon=0).assign_coords(lon=lon[0] + 360.0)
    da = xr.concat([lo, da, hi], dim='lon')
    return da.interp(lat=target_lat, lon=target_lon, method='linear')
