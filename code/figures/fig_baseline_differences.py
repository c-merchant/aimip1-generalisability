"""
Map major Koppen-Geiger zone disagreement of each 0K model baseline against ERA5.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from code.cluster.cluster import Cluster
from code.figures.config import FIGURE_DIR, BASELINE_START, BASELINE_END
from code.figures.utils import (
    load_era5, load_aimip, load_amip, land_fraction, mask_to_nan,
    zone_masks, panel_title, cos_lat_weights, map_grid, draw_zones,
    crop_south, desaturate, to_pm180, roll_lon,
    MAJOR_LABEL_MAP, NAMED_AMIP, AIMIP_MMM_LABEL, AMIP_MMM_LABEL,
    AIMIP_MODELS, amip_with_scenario,
)

# 5 AIMIP + AIMIP MMM + 2 named AMIP + AMIP MMM = 9 maps on a 2-column grid.
N_MAPS = 9
N_COLS = 2
LEGEND_Y = -0.046
SUPTITLE_Y = 0.998

def agreement_pct(other_major, ref_major, weights):
    """Cos-lat-weighted percent of cells where two major-zone maps match."""
    valid = np.isfinite(ref_major) & np.isfinite(other_major)
    if not valid.any():
        return float('nan')
    valid_weights = weights[valid]
    return 100.0 * (valid_weights * (ref_major[valid] == other_major[valid])).sum() / valid_weights.sum()

# Load data
print('Loading ERA5 (land-only).')
era5 = load_era5(BASELINE_START, BASELINE_END, land_only=True)
lat, lon = era5['lat'], era5['lon']
ocean = land_fraction() < 0.5

print('Loading AIMIP and AMIP at baseline.')
aimip_all = load_aimip(lat, lon, BASELINE_START, BASELINE_END)
amip_all = load_amip(lat, lon, BASELINE_START, BASELINE_END)

# Shift maps onto -180 to 180 longitude axis
lon, order = to_pm180(lon)
era5 = roll_lon(era5, order)
ocean = roll_lon(ocean, order)
aimip_all = roll_lon(aimip_all, order)
amip_all = roll_lon(amip_all, order)

# Fixed 0K model sets
amip_0k_models = amip_with_scenario('0K')

def baseline_0k(key):
    """Ocean-masked 0K baseline arrays for a model key."""
    if key not in aimip_all and key not in amip_all:
        raise ValueError(f'Required baseline model {key} absent from loaded data')
    return mask_to_nan(aimip_all.get(key, amip_all.get(key)), ocean)


# Classify
print('Classifying.')
kg = Cluster('koppen_geiger')
era5_class = kg.classify(era5['T'], era5['P'])

aimip_classes = {}
for model in AIMIP_MODELS:
    masked = baseline_0k(model.scenario_key('0K'))
    aimip_classes[model.display] = kg.classify(masked['T'], masked['P'])

# CESM2 and GFDL-CM4 as named references
named_classes = {}
for model_key, display in NAMED_AMIP.items():
    masked = baseline_0k(model_key + '_0K')
    named_classes[display] = kg.classify(masked['T'], masked['P'])

def mmm_classification(models):
    """Modal classification of multi-model mean fields, keeping the year axis so the
    per-year-then-mode rule matches the individual model panels."""
    fields = [baseline_0k(model.scenario_key('0K')) for model in models]
    n_years = min(f['T'].shape[0] for f in fields)
    T = np.nanmean([f['T'][:n_years] for f in fields], axis=0)
    P = np.nanmean([f['P'][:n_years] for f in fields], axis=0)
    return kg.classify(T, P)

amip_class = mmm_classification(amip_0k_models)
aimip_class = mmm_classification(AIMIP_MODELS)

_, ref_major, colors = zone_masks(era5_class)
ref_major = ref_major.astype(float)
ref_valid = np.isfinite(ref_major)
weights = cos_lat_weights(lat, ref_major.shape)
zone_ids = sorted(MAJOR_LABEL_MAP)
muted = {zone_id: desaturate(colors[zone_id]) for zone_id in zone_ids}

# Comparison models, each tagged with its family for the panel title
compare = [('AIMIP: ' + name, cls['Major'].astype(float)) for name, cls in aimip_classes.items()]
compare.append((AIMIP_MMM_LABEL, aimip_class['Major'].astype(float)))
compare += [('AMIP: ' + name, cls['Major'].astype(float)) for name, cls in named_classes.items()]
compare.append((AMIP_MMM_LABEL, amip_class['Major'].astype(float)))
assert len(compare) == N_MAPS

fig, gs, axes = map_grid(N_MAPS, N_COLS, hspace=0.19, wspace=0.04, top=0.958, bottom=0.02)
# Centre a lone trailing panel by spanning it across the row
if N_MAPS % N_COLS:
    axes[-1].remove()
    axes[-1] = fig.add_subplot(gs[(N_MAPS - 1) // N_COLS, :])

seen_agree, seen_dest = set(), set()
for idx, (ax, (name, model_major)) in enumerate(zip(axes, compare)):
    valid = ref_valid & np.isfinite(model_major)
    agree = valid & (model_major == ref_major)
    disagree = valid & (model_major != ref_major)
    draw_zones(ax, lon, lat, np.where(agree, ref_major, np.nan), muted)
    draw_zones(ax, lon, lat, np.where(disagree, model_major, np.nan), colors)
    seen_agree.update(int(z) for z in np.unique(ref_major[agree]))
    seen_dest.update(int(z) for z in np.unique(model_major[disagree]))

    agreement = agreement_pct(model_major, ref_major, weights)
    is_mmm = name in (AIMIP_MMM_LABEL, AMIP_MMM_LABEL)
    panel_title(ax, idx, '%s (%.1f%%)' % (name, agreement), title_kw={'fontweight': 'bold'} if is_mmm else None)
    ax.set_xticks([]); ax.set_yticks([])
    crop_south(ax)

zones_shown = [zone_id for zone_id in zone_ids if zone_id in seen_agree or zone_id in seen_dest]
legend = []
for zone_id in zones_shown:
    legend.append(Patch(facecolor=muted[zone_id], edgecolor='k', label=MAJOR_LABEL_MAP[zone_id] + ' Agreement'))
    legend.append(Patch(facecolor=colors[zone_id], edgecolor='k', label=MAJOR_LABEL_MAP[zone_id] + ' in Model'))

fig.legend(handles=legend, loc='lower center', ncol=len(zones_shown), bbox_to_anchor=(0.5, LEGEND_Y),
           columnspacing=0.8, handlelength=1.1, handletextpad=0.4)
fig.suptitle('Disagreement in Major Köppen-Geiger Zones (0K Model Baseline vs. ERA5)', y=SUPTITLE_Y)
out = FIGURE_DIR / 'baseline_differences_2col.pdf'
fig.savefig(out, bbox_inches='tight', dpi=300)
plt.close(fig)
print('Saved ' + str(out))

# AMIP per-model agreement range
def _amip_agree(m):
    b = baseline_0k(m.scenario_key('0K'))
    return agreement_pct(kg.classify(b['T'], b['P'])['Major'].astype(float), ref_major, weights), m.display
_ag = sorted(_amip_agree(m) for m in amip_0k_models)
print('AMIP agreement: min %s %.1f%%, max %s %.1f%% (n=%d)' % (_ag[0][1], _ag[0][0], _ag[-1][1], _ag[-1][0], len(_ag)))
