"""
Shared analysis and plotting helpers for the figure pipeline.
"""

import colorsys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy import stats
from code import data
from code.data.loader_utils import load_land_mask
from code.cluster.cluster import Cluster
from code.figures.config import (ERA5_DIR, AIMIP_DIR, AMIP_DIR, LAND_MASK_PATH, ERA5_CACHE, BASELINE_START, BASELINE_END, RESOLUTION, KOPPEN_TABLE)
from code.figures.models import AIMIP_MODELS, AMIP_MODELS, amip_with_scenario

FONT_SIZE = 9
FIG_WIDTH = 8.0
mpl.rcParams.update({'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
                     'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
                     'figure.titlesize': FONT_SIZE, 'figure.dpi': 150,
                     'legend.frameon': True, 'legend.edgecolor': 'grey', 'legend.framealpha': 1.0, 'legend.fancybox': False,
                     'font.family': 'serif', 'font.serif': ['CMU Serif'], 'mathtext.fontset': 'cm'})

MAJOR_LABEL_MAP = {1: 'Tropical (A)', 2: 'Arid (B)', 3: 'Temperate (C)', 4: 'Cold (D)', 5: 'Polar (E)'}
SST_FORCING_MAP = {'0K': 0, '2K': 2, '4K': 4, '-4K': -4}
AIMIP_PALETTE = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
AIMIP_MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', 'h', '*', 'd', 'p', '>']
ORIGIN_HATCHES = {1: '///', 2: '\\\\\\', 3: '|||', 4: '---', 5: '...'}
PANEL_LABELS = 'abcdefghijklmnopqrstuvwxyz'

AIMIP_MMM_LABEL = 'AIMIP MMM'
NAMED_AMIP = {'NCAR_CESM2': 'CESM2', 'NOAA-GFDL_GFDL-CM4': 'GFDL-CM4'}
AMIP_MMM_LABEL = 'AMIP MMM'
NAMED_AMIP_STYLE = {'NCAR_CESM2': ('#000000', '*'), 'NOAA-GFDL_GFDL-CM4': ('#444444', 'P')}

SOUTH_CROP = -60.0

def set_map_aspect(ax, lat_span=180.0):
    """Box aspect for lat_span degrees over 360 of longitude."""
    ax.set_box_aspect(lat_span / 360.0)

def crop_south(ax, south=SOUTH_CROP):
    """Map aspect for a view without the uniform Antarctic strip."""
    set_map_aspect(ax, lat_span=90.0 - south)

def map_grid(n_maps, ncols, extra_ratios=(), hspace=0.25, wspace=0.05, left=0.01, right=0.99, top=0.92, bottom=0.08):
    """Figure, GridSpec, and axes for a grid of Antarctic-cropped maps, figure height derived from map aspect. Extra_ratios appends rows (in map-row heights) for legends/charts."""
    map_rows = -(-n_maps // ncols)
    cell_w = FIG_WIDTH * (right - left) / (ncols + (ncols - 1) * wspace)
    map_h = cell_w * (90.0 - SOUTH_CROP) / 360.0
    heights = [map_h] * map_rows + [r * map_h for r in extra_ratios]
    n_rows = len(heights)
    fig_h = sum(heights) * (1 + hspace * (n_rows - 1) / n_rows) / (top - bottom)
    fig = plt.figure(figsize=(FIG_WIDTH, fig_h))
    gs = GridSpec(n_rows, ncols, figure=fig, left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace, height_ratios=heights)
    axes = [fig.add_subplot(gs[divmod(i, ncols)]) for i in range(n_maps)]
    return fig, gs, axes

def draw_zones(ax, lon, lat, zones, palette):
    """Categorical zone map with one colour per zone id."""
    ids = sorted(palette)
    cmap = mcolors.ListedColormap([palette[i] for i in ids])
    norm = mcolors.BoundaryNorm(np.array(ids + [ids[-1] + 1], dtype=float) - 0.5, cmap.N)
    ax.pcolormesh(lon, lat, zones, cmap=cmap, norm=norm, shading='auto', rasterized=True)

def to_pm180(lon):
    """Convert a 0-360 longitude axis to -180-180, returning the new axis and its sort order."""
    lon = np.asarray(lon, dtype=float)
    new = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(new, kind='stable')
    return new[order], order

def roll_lon(obj, order):
    """Apply a longitude sort order to the last axis of every matching array recursing into dicts."""
    n = len(order)
    if isinstance(obj, np.ndarray):
        return obj[..., order] if obj.ndim and obj.shape[-1] == n else obj
    if isinstance(obj, dict):
        return {k: roll_lon(v, order) for k, v in obj.items()}
    return obj

def panel_label(ax, idx, x=0.015, y=0.985, fontsize=FONT_SIZE):
    """Bold panel letter inside the top-left of an axis."""
    letter = PANEL_LABELS[idx] if isinstance(idx, int) else idx
    ax.text(x, y, '(' + letter + ')', transform=ax.transAxes, fontweight='bold', fontsize=fontsize, va='top', ha='left', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75), zorder=10)

def panel_title(ax, idx, title, label_fs=FONT_SIZE, title_fs=FONT_SIZE, title_kw=None):
    """Left-aligned bold panel letter and a centred title."""
    letter = PANEL_LABELS[idx] if isinstance(idx, int) else idx
    ax.set_title('(' + letter + ')', loc='left', fontweight='bold', fontsize=label_fs)
    kw = {'fontweight': 'bold'}
    kw.update(title_kw or {})
    ax.set_title(title, loc='center', fontsize=title_fs, **kw)

def parse_key(key):
    """'{inst}_{model}_{scenario}' to (model_id, scenario), or None if the tail is not a known scenario."""
    mid, _, scn = key.rpartition('_')
    return (mid, scn) if mid and scn in SST_FORCING_MAP else None

def desaturate(hex_color, factor=0.35):
    """Lightened, desaturated version of a hex colour, as an RGB tuple."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(1.0, l + (1 - l) * (1 - factor)), s * factor)

def load_era5(start=BASELINE_START, end=BASELINE_END, land_only=False):
    """ERA5 baseline over the given year range."""
    return data.load_era5(ERA5_DIR, resolution=RESOLUTION, start_year=start, end_year=end, land_only=land_only)

def load_aimip(lat, lon, start=BASELINE_START, end=BASELINE_END, land_mask_path=None):
    """AIMIP ensemble means on the target grid, ocean-masked if a land mask is given."""
    raw = data.load_aimip(AIMIP_DIR, target_lat=lat, target_lon=lon, start_year=start, end_year=end, land_mask_path=land_mask_path)
    return data.ensemble_mean(raw)

def load_amip(lat, lon, start=BASELINE_START, end=BASELINE_END, land_mask_path=None):
    """AMIP ensemble means on the target grid; time sync and masking as in load_aimip."""
    raw = data.load_amip(AMIP_DIR, target_lat=lat, target_lon=lon, start_year=start, end_year=end, land_mask_path=land_mask_path)
    averaged = data.ensemble_mean(raw)
    # CESM2's baseline run is longer than its 4K run; truncate it so the pair spans the same years
    if 'NCAR_CESM2_0K' in averaged and 'NCAR_CESM2_4K' in averaged:
        n = averaged['NCAR_CESM2_4K']['T'].shape[0]
        b = averaged['NCAR_CESM2_0K']
        for v in ('T', 'P', 'Q'):
            if v in b:
                b[v] = b[v][:n]
        b['years'] = n
    return averaged

def load_era5_classification():
    """Cached ERA5 Koppen classification and confidence arrays."""
    cache = np.load(str(ERA5_CACHE), allow_pickle=True)
    return {
        'Class': cache['class_arr'], 'Major': cache['major_arr'],
        'Confidence': {'Class': cache['confidence_class'], 'Major': cache['confidence_major']},
    }

def land_fraction():
    """Fractional land-sea mask (0-1) on native grid."""
    return load_land_mask(LAND_MASK_PATH).values

def surface_mask(target_lat, target_lon):
    """Integer land/ocean mask on the target grid (thresholded at 0.5)."""
    lsm = load_land_mask(LAND_MASK_PATH).sortby('lat').interp(lat=target_lat, lon=target_lon, method='linear').values
    s = np.zeros_like(lsm, dtype=np.int8)
    s[lsm >= 0.5] = 1
    s[lsm < 0.5] = -1
    return s

def cos_lat_weights(lat, shape):
    """cos(lat) area weights broadcast to (lat, lon) shape."""
    cl = np.clip(np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64))), 0.0, None)
    return np.broadcast_to(cl[:, None], shape).astype(np.float64)

def mask_to_nan(ds, ocean):
    """Copies of T and P with NaN ocean cells."""
    out = {}
    for v in ('T', 'P'):
        a = ds[v].copy()
        a[:, :, ocean] = np.nan
        out[v] = a
    return out

def _zone_colors():
    """Mean subclass RGB from the Koppen table per major zone."""
    table = pd.read_csv(KOPPEN_TABLE)
    by_major = table.groupby('Major')[['Red', 'Green', 'Blue']].mean().astype(int)
    return {mid: '#%02X%02X%02X' % tuple(by_major.loc[mid]) for mid in sorted(MAJOR_LABEL_MAP)}

ZONE_COLORS = _zone_colors()

def zone_masks(classification):
    """Per-zone boolean masks, Major array, and the zone colours."""
    major = classification['Major']
    return {mid: (major == mid) for mid in sorted(MAJOR_LABEL_MAP)}, major, ZONE_COLORS

def group_by_model(data):
    """Flat {key: value} -> {model_id: {scenario: value}}."""
    out = {}
    for k, v in data.items():
        p = parse_key(k)
        if p is None:
            continue
        mid, scn = p
        out.setdefault(mid, {})[scn] = v
    return out

def require_scns(grouped, key, scenarios):
    """Scenario dict for one model, raising if the model or any required scenario is absent."""
    scns = grouped.get(key)
    if scns is None:
        raise ValueError(f'Required model {key} absent from loaded data')
    missing = [s for s in scenarios if s not in scns]
    if missing:
        raise ValueError(f'Model {key} missing required scenarios {missing}')
    return scns

def slice_baseline(full, n_years=36):
    """Truncate each entry's T/P/Q to the first n_years."""
    return {k: {v: d[v][:n_years] for v in ('T', 'P', 'Q') if v in d} for k, d in full.items()}

def dT_map(scns, ref='0K', tgt='4K'):
    """Annual-mean temperature change map between two scenarios."""
    if ref not in scns or tgt not in scns:
        return None
    return np.nanmean(scns[tgt]['T'], axis=(0, 1)) - np.nanmean(scns[ref]['T'], axis=(0, 1))

def dP_pct_map(scns, ref='0K', tgt='4K'):
    """Percent precipitation change map between two scenarios."""
    if ref not in scns or tgt not in scns:
        return None
    Pr_m, Pt_m = np.nanmean(scns[ref]['P'], axis=(0, 1)), np.nanmean(scns[tgt]['P'], axis=(0, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = 100.0 * (Pt_m - Pr_m) / Pr_m
    out[~np.isfinite(out)] = np.nan
    return out

def map_ensemble_mean(maps):
    """NaN-ignoring mean over a list of 2D maps."""
    valid = [m for m in maps if m is not None]
    return np.nanmean(np.stack(valid, axis=0), axis=0) if valid else None

def weighted_cell_mean(values, cell_weights):
    """Weighted mean over the last (cell) axis."""
    weights_full = np.broadcast_to(cell_weights, values.shape)
    weight_sum = (weights_full * np.isfinite(values)).sum()
    return np.nansum(values * weights_full) / weight_sum if weight_sum > 0 else np.nan

def zone_annual_diffs(scns, masks, weights, ref='0K'):
    """Per-zone cos-lat-weighted mean T and P differences of each scenario vs the reference."""
    if ref not in scns:
        return {}
    Tr = scns[ref]['T']
    Pr = scns[ref]['P']
    out = {}
    for scn, ds in scns.items():
        if scn == ref:
            continue
        Ts, Ps = ds['T'], ds['P']
        zone_diff = {}
        for zid, m in masks.items():
            if not m.any():
                zone_diff[zid] = {'T': np.nan, 'P': np.nan}
                continue
            cw = weights[m]
            dT = weighted_cell_mean(Ts[:, :, m], cw) - weighted_cell_mean(Tr[:, :, m], cw)
            dP = weighted_cell_mean(Ps[:, :, m], cw) - weighted_cell_mean(Pr[:, :, m], cw)
            zone_diff[zid] = {'T': float(dT), 'P': float(dP)}
        out[scn] = zone_diff
    return out

def per_year_classes(T, P, cluster=None):
    """Per-year Koppen Major classes (years, lat, lon)."""
    return (cluster or Cluster('koppen_geiger')).classify_per_year(T, P)

def mode_map(per_year_arr):
    """Per-cell modal zone across years."""
    return stats.mode(per_year_arr, axis=0, nan_policy='omit').mode.astype(np.float32)

def modal_fraction(per_year_arr):
    """Per-cell fraction of years matching the modal zone (temporal confidence)."""
    a = np.asarray(per_year_arr, dtype=float)
    n = np.isfinite(a).sum(axis=0).astype(float)
    md = mode_map(a)
    match = ((a == md[None]) & np.isfinite(a)).sum(axis=0).astype(float)
    return np.where(n > 0, match / n, np.nan)

def interannual_band(per_year_arr, lo=10, hi=90):
    """Per-cell (lo, hi) percentile bands of yearly Major zones."""
    a = np.asarray(per_year_arr, dtype=float)
    valid = np.isfinite(a).any(axis=0)
    p_lo = np.full(a.shape[1:], np.nan)
    p_hi = np.full(a.shape[1:], np.nan)
    if valid.any():
        col = a[:, valid]
        p_lo[valid] = np.nanpercentile(col, lo, axis=0)
        p_hi[valid] = np.nanpercentile(col, hi, axis=0)
    return p_lo, p_hi

def dest_visited_fraction(per_year_arr, dest_mode):
    """Per-cell fraction of baseline years already classified as destination zone."""
    a = np.asarray(per_year_arr, dtype=float)
    n = np.isfinite(a).sum(axis=0).astype(float)
    match = ((a == dest_mode[None]) & np.isfinite(a)).sum(axis=0).astype(float)
    return np.where(n > 0, match / n, np.nan)

def migration_destination(ref_mode, scn_mode, land_mask):
    """Scenario zone on land cells that changed zone."""
    return np.where((ref_mode != scn_mode) & land_mask, scn_mode, 0).astype(np.float32)

def zone_migration_fractions(ref_mode, scn_mode, surface, weights, want='land'):
    """Cos-weighted percent of each origin zone's area that migrates to a different zone."""
    surf_mask = (surface == 1) if want == 'land' else (surface == -1)
    migrated = (ref_mode != scn_mode) & np.isfinite(ref_mode) & np.isfinite(scn_mode)
    out = {}
    for zid in sorted({int(v) for v in np.unique(ref_mode) if np.isfinite(v)}):
        base = (ref_mode == zid) & surf_mask
        tot = float((base * weights).sum())
        mig = float(((base & migrated) * weights).sum())
        out[zid] = (100.0 * mig / tot) if tot > 0 else np.nan
    return out

def zone_arrival_fractions(ref_mode, scn_mode, surface, weights, want='land'):
    """Cos-weighted percent of total surface area that newly migrates into each zone."""
    surf_mask = (surface == 1) if want == 'land' else (surface == -1)
    migrated = (ref_mode != scn_mode) & np.isfinite(ref_mode) & np.isfinite(scn_mode)
    tot = float((surf_mask * weights).sum())
    out = {}
    for zid in sorted(MAJOR_LABEL_MAP):
        arrive = surf_mask & migrated & (scn_mode == zid)
        out[zid] = (100.0 * float((arrive * weights).sum()) / tot) if tot > 0 else np.nan
    return out
