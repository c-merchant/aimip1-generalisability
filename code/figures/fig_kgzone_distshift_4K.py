"""
Per-zone temperature and precipitation distribution shifts under +4K SST forcing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from code.cluster.cluster import Cluster
from code.figures.config import FIGURE_DIR, LAND_MASK_PATH, BASELINE_START, BASELINE_END
from code.figures.utils import (
    load_era5, load_aimip, load_amip, load_era5_classification, group_by_model,
    slice_baseline, zone_masks, require_scns, per_year_classes, mode_map, FIG_WIDTH,
    AIMIP_MODELS, amip_with_scenario, MAJOR_LABEL_MAP, PANEL_LABELS, cos_lat_weights,
)

SAMPLE_CAP = 50000
GRID_SIZE = 200
RNG_SEED = 0

def zone_samples(field, mask, w2d, rng):
    """Finite (values, cos-lat weights) sample of a field over a zone mask, capped at SAMPLE_CAP."""
    vals = field[:, :, mask]
    w = np.broadcast_to(w2d[mask], vals.shape).ravel()
    vals = vals.ravel()
    ok = np.isfinite(vals)
    vals, w = vals[ok], w[ok]
    if vals.size > SAMPLE_CAP:
        idx = rng.choice(vals.size, size=SAMPLE_CAP, replace=False)
        vals, w = vals[idx], w[idx]
    return vals, w

def kde(samples, x_grid):
    """Cos-lat-weighted Gaussian KDE evaluated on x_grid, or None if degenerate."""
    vals, w = samples
    if vals.size < 2 or np.ptp(vals) == 0:
        return None
    return gaussian_kde(vals, weights=w)(x_grid)

def common_grid(samples_list, pad=0.05):
    """Shared x-grid over the pooled 1-99 percentile range, padded."""
    pooled = np.concatenate([vals for vals, _ in samples_list if vals.size])
    lower, upper = np.percentile(pooled, [1, 99])
    span = upper - lower if upper > lower else 1.0
    return np.linspace(lower - pad * span, upper + pad * span, GRID_SIZE)

# Load data
print('Loading data.')
era5 = load_era5(BASELINE_START, BASELINE_END)
lat, lon = era5['lat'], era5['lon']
classification = load_era5_classification()

# Load land-masked so each model's own classification excludes ocean
aimip_all = slice_baseline(load_aimip(lat, lon, land_mask_path=LAND_MASK_PATH))
amip_all = load_amip(lat, lon, land_mask_path=LAND_MASK_PATH)

aimip_grouped = group_by_model(aimip_all)
amip_grouped = group_by_model(amip_all)

# Five AIMIP panels and available AMIP envelope members
for model in AIMIP_MODELS:
    require_scns(aimip_grouped, model.key, ('0K', '4K'))
amip_0k_models = amip_with_scenario('0K')
amip_4k_models = amip_with_scenario('4K')

print('Classifying per-model baseline zones.')
cluster = Cluster('koppen_geiger')

def baseline_zone_masks(scns):
    """Per-zone boolean masks from a model's own 0K modal classification."""
    major = mode_map(per_year_classes(scns['0K']['T'], scns['0K']['P'], cluster))
    return {mid: (major == mid) for mid in sorted(MAJOR_LABEL_MAP)}

era5_masks, _, _ = zone_masks(classification)
aimip_masks = {m.key: baseline_zone_masks(aimip_grouped[m.key]) for m in AIMIP_MODELS}
amip_masks = {m.key: baseline_zone_masks(amip_grouped[m.key]) for m in amip_0k_models}

# Compute per-zone densities
print('Computing per-zone KDEs.')
fig, axes = plt.subplots(2, 5, figsize=(1.75 * FIG_WIDTH, 5.6), sharex=False)
var_keys = ['T', 'P']
var_xlabels = {'T': 'Temperature (°C)', 'P': 'Precipitation (mm/month)'}
rng = np.random.default_rng(RNG_SEED)
weights2d = cos_lat_weights(lat, era5['T'].shape[-2:])

for col, zone_id in enumerate(sorted(MAJOR_LABEL_MAP)):
    for row, var in enumerate(var_keys):
        ax = axes[row, col]
        era5_samp = zone_samples(era5[var], era5_masks[zone_id], weights2d, rng)
        amip_0k = {model.key: zone_samples(amip_grouped[model.key]['0K'][var], amip_masks[model.key][zone_id], weights2d, rng) for model in amip_0k_models}
        amip_4k = {model.key: zone_samples(amip_grouped[model.key]['4K'][var], amip_masks[model.key][zone_id], weights2d, rng) for model in amip_4k_models}
        aimip_0k = {model.key: zone_samples(aimip_grouped[model.key]['0K'][var], aimip_masks[model.key][zone_id], weights2d, rng) for model in AIMIP_MODELS}
        aimip_4k = {model.key: zone_samples(aimip_grouped[model.key]['4K'][var], aimip_masks[model.key][zone_id], weights2d, rng) for model in AIMIP_MODELS}

        x = common_grid([era5_samp] + list(amip_0k.values()) + list(amip_4k.values()) + list(aimip_0k.values()) + list(aimip_4k.values()))

        amip_0k_kdes = [curve for curve in (kde(samp, x) for samp in amip_0k.values()) if curve is not None]
        if amip_0k_kdes:
            band = np.array(amip_0k_kdes)
            ax.fill_between(x, band.min(0), band.max(0), color='0.65', alpha=0.45, linewidth=0)
        amip_4k_kdes = [curve for curve in (kde(samp, x) for samp in amip_4k.values()) if curve is not None]
        if amip_4k_kdes:
            band = np.array(amip_4k_kdes)
            ax.fill_between(x, band.min(0), band.max(0), color='#377eb8', alpha=0.25, linewidth=0)

        era5_kde = kde(era5_samp, x)
        if era5_kde is not None:
            ax.plot(x, era5_kde, color='black', linewidth=1.4)

        for model in AIMIP_MODELS:
            kde_0k = kde(aimip_0k[model.key], x)
            if kde_0k is not None:
                ax.plot(x, kde_0k, color=model.color, linestyle='--', linewidth=0.9, alpha=0.85)
            kde_4k = kde(aimip_4k[model.key], x)
            if kde_4k is not None:
                ax.plot(x, kde_4k, color=model.color, linestyle='-', linewidth=1.1, alpha=0.95)

        if row == 0:
            ax.set_title(MAJOR_LABEL_MAP[zone_id], fontweight='bold')
            ax.text(0.03, 0.96, '(' + PANEL_LABELS[col] + ')', transform=ax.transAxes, ha='left', va='top', fontweight='bold')
        else:
            ax.text(0.97, 0.96, '(' + PANEL_LABELS[5 + col] + ')', transform=ax.transAxes, ha='right', va='top', fontweight='bold')
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(var_xlabels[var])

# Legend
handles = [
    Patch(facecolor='0.65', edgecolor='none', alpha=0.45, label='AMIP 0K range'),
    Patch(facecolor='#377eb8', edgecolor='none', alpha=0.25, label='AMIP +4K range'),
    Line2D([0], [0], color='black', linewidth=1.4, label='ERA5'),
]
for model in AIMIP_MODELS:
    handles.append(Line2D([0], [0], color=model.color, linestyle='--', linewidth=0.9, label=model.display + ' 0K'))
    handles.append(Line2D([0], [0], color=model.color, linestyle='-', linewidth=1.1, label=model.display + ' +4K'))

ncol = min(6, len(handles))
fig.legend(handles=handles, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, -0.10))
fig.suptitle('Per-Zone Distribution Shifts at +4K SST Forcing', y=1.0)
fig.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=1.0, w_pad=0.5)

out = FIGURE_DIR / 'kgzone_distshift_4K.pdf'
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close(fig)
print('Saved ' + str(out))
