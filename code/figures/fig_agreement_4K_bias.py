"""
Per-model +4K dT and fractional dP bias maps relative to AMIP ensemble mean.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from code.cluster.cluster import Cluster
from code.figures.config import FIGURE_DIR, BASELINE_START, BASELINE_END
from code.figures.utils import (
    load_era5, load_aimip, load_amip, group_by_model,
    slice_baseline, dT_map, dP_pct_map, map_ensemble_mean, require_scns,
    surface_mask, panel_title, to_pm180, roll_lon, amip_with_scenario, FIG_WIDTH,
    MAJOR_LABEL_MAP, AIMIP_MODELS, NAMED_AMIP, FONT_SIZE, per_year_classes, mode_map,
)

# 5 AIMIP + 2 named AMIP = 7 model panels, rendered two per figure row.
N_BIAS_ROWS = 7

ZONE_HATCHES = {1: '...', 2: '///', 3: '\\\\\\', 4: 'ooo', 5: '---'}

def symmetric_limit(fields, q=0.98):
    """Symmetric color limit from the qth quantile of the fields' magnitudes."""
    vals = np.concatenate([field[np.isfinite(field)].ravel() for field in fields if field is not None])
    if vals.size == 0:
        return 1.0
    return float(np.quantile(np.abs(vals), q))

def baseline_major(scns, cluster):
    """A model's own 0K modal major-zone map."""
    return mode_map(per_year_classes(scns['0K']['T'], scns['0K']['P'], cluster))

def draw_map(ax, lon, lat, field, cmap, norm, title, idx, zones, land, bold=False):
    """One global field map with zone hatching, coastlines, and a panel title."""
    ax.pcolormesh(lon, lat, field, cmap=cmap, norm=norm, shading='auto', rasterized=True, transform=ccrs.PlateCarree())
    if zones is not None:
        for zone_id, hatch in ZONE_HATCHES.items():
            zone_mask = ((zones == zone_id) & land).astype(float)
            if zone_mask.any():
                ax.contourf(lon, lat, zone_mask, levels=[0.5, 1.5], colors='none', hatches=[hatch], transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='black')
    ax.set_global()
    panel_title(ax, idx, title, title_kw={'fontweight': 'bold'} if bold else None)

def render(rows, cmap_T, nT, cmap_P, nP, label_T, label_P, suptitle, out, land):
    """Two-column (dT, fractional dP) bias figure; rows is a list of (name, Tfield, Pfield, bold, zones), two per figure row."""
    n = len(rows)
    proj = ccrs.PlateCarree(central_longitude=0)
    ncols, grid_cols, nrows, map_w, map_h, top_pad, bottom_pad = 4, 9, int(np.ceil(n / 2)), 3.0, 1.5, 1.0, 1.6
    hspace, wspace = 0.28, 0.015
    k = FIG_WIDTH / (ncols * map_w + 0.5)
    map_w, map_h, top_pad, bottom_pad = map_w * k, map_h * k, top_pad * k, bottom_pad * k
    fig_w = ncols * map_w + 0.5 * k
    fig_h = map_h * (nrows + (nrows - 1) * hspace) + top_pad + bottom_pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    gx_left, gx_right = 0.03, 0.99
    gs = GridSpec(nrows, grid_cols, figure=fig, hspace=hspace, wspace=wspace,
                  left=gx_left, right=gx_right, top=1 - top_pad / fig_h, bottom=bottom_pad / fig_h,
                  width_ratios=[1, 1, 1, 1, 0.12, 1, 1, 1, 1])
    for i, (name, Tfield, Pfield, bold, zones) in enumerate(rows):
        row, sub = divmod(i, 2)
        lone = i == n - 1 and n % 2 == 1
        tstart, pstart = (1, 6) if lone else (sub * 2, 5 + sub * 2)
        slT, slP, idxT, idxP = slice(tstart, tstart + 2), slice(pstart, pstart + 2), i, n + i
        ax_T = fig.add_subplot(gs[row, slT], projection=proj)
        draw_map(ax_T, lon, lat, Tfield, cmap_T, nT, name, idxT, zones, land, bold)
        ax_P = fig.add_subplot(gs[row, slP], projection=proj)
        draw_map(ax_P, lon, lat, Pfield, cmap_P, nP, name, idxP, zones, land, bold)

    # Subheader
    span = gx_right - gx_left
    yhead = 1 - 0.5 * top_pad / fig_h
    fig.text(gx_left + span * 0.25, yhead, r'$\Delta T$', ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE)
    fig.text(gx_left + span * 0.75, yhead, r'Fractional $\Delta P$', ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE)

    cy = (bottom_pad - 0.35 * k) / fig_h
    ax_cT = fig.add_axes([0.07, cy, 0.40, 0.02])
    sT = plt.cm.ScalarMappable(cmap=cmap_T, norm=nT); sT.set_array([])
    fig.colorbar(sT, cax=ax_cT, orientation='horizontal').set_label(label_T)
    ax_cP = fig.add_axes([0.55, cy, 0.40, 0.02])
    sP = plt.cm.ScalarMappable(cmap=cmap_P, norm=nP); sP.set_array([])
    fig.colorbar(sP, cax=ax_cP, orientation='horizontal').set_label(label_P)

    handles = [Patch(facecolor='white', edgecolor='black', hatch=ZONE_HATCHES[zone_id], label=MAJOR_LABEL_MAP[zone_id])
               for zone_id in sorted(ZONE_HATCHES)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.0),
               handlelength=2.4, handleheight=1.4)
    fig.suptitle(suptitle, y=0.997)
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('Saved ' + str(out))

# Load data
print('Loading data.')
era5 = load_era5(BASELINE_START, BASELINE_END)
lat, lon = era5['lat'], era5['lon']
land = surface_mask(lat, lon) == 1

aimip = load_aimip(lat, lon)
amip = load_amip(lat, lon)

# Shift maps onto -180 to 180 longitude axis
lon, order = to_pm180(lon)
land = roll_lon(land, order)
aimip = roll_lon(aimip, order)
amip = roll_lon(amip, order)

aimip_g = group_by_model(slice_baseline(aimip))
amip_g = group_by_model(amip)

# 5 AIMIP panels and 13 AMIP models with +4K
aimip_models = [p.key for p in AIMIP_MODELS]
amip_models = [m.key for m in amip_with_scenario('4K')]
named = list(NAMED_AMIP)
display = {p.key: p.display for p in AIMIP_MODELS}

# Per-model own-baseline zones for hatching.
# Each model is hatched by its own 0K classification, so the response bias can be read against where the model places its own zones.
print('Classifying per-model baseline zones.')
cluster = Cluster('koppen_geiger')
aimip_zones = {k: baseline_major(require_scns(aimip_g, k, ('0K',)), cluster) for k in aimip_models}
amip_zones = {k: baseline_major(require_scns(amip_g, k, ('0K',)), cluster) for k in named}

# +4K responses: per-model dT/dP with AMIP ensemble mean as bias reference.
aimip_dT = {model_key: dT_map(require_scns(aimip_g, model_key, ('0K', '4K'))) for model_key in aimip_models}
aimip_dP = {model_key: dP_pct_map(aimip_g[model_key]) for model_key in aimip_models}
amip_dT = {model_key: dT_map(require_scns(amip_g, model_key, ('0K', '4K'))) for model_key in amip_models}
amip_dP = {model_key: dP_pct_map(amip_g[model_key]) for model_key in amip_models}

ref_dT = map_ensemble_mean(list(amip_dT.values()))
ref_dP = map_ensemble_mean(list(amip_dP.values()))

# Bias wrt the AMIP ensemble mean hatched by each model's own baseline zones
bias_rows = [(display[model_key], aimip_dT[model_key] - ref_dT, aimip_dP[model_key] - ref_dP, False, aimip_zones[model_key]) for model_key in aimip_models]
bias_rows += [(NAMED_AMIP[model_key], amip_dT[model_key] - ref_dT, amip_dP[model_key] - ref_dP, True, amip_zones[model_key]) for model_key in named]
assert len(bias_rows) == N_BIAS_ROWS

# 95th percentile per variable; above it the fractional dP tail is dominated by near-zero baseline P cells
vT_bias = symmetric_limit([row[1] for row in bias_rows], q=0.95)
vP_bias = symmetric_limit([row[2] for row in bias_rows], q=0.95)
print('Color limits: T +/-%.2f K, P +/-%.1f%%' % (vT_bias, vP_bias))
render(bias_rows, plt.get_cmap('RdBu_r'), mcolors.Normalize(-vT_bias, vT_bias),
       plt.get_cmap('BrBG'), mcolors.Normalize(-vP_bias, vP_bias),
       r'$\Delta T$ bias vs. AMIP mean (K)', r'Fractional $\Delta P$ bias vs. AMIP mean (%)',
       'AIMIP Bias with respect to AMIP Ensemble Mean under +4K Forcing', FIGURE_DIR / 'agreement_4K_bias.pdf',
       land)
