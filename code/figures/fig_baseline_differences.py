"""
Map major Koppen-Geiger zone disagreement of each 0K model baseline against ERA5.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from code.cluster.cluster import Cluster
from code.figures.config import FIGURE_DIR, BASELINE_START, BASELINE_END
from code.figures.models import AIMIP_MODELS, amip_with_scenario
from code.figures.utils import (load_era5, load_aimip, iter_amip_members, group_by_member, mode_map, land_fraction, mask_to_nan, zone_masks, panel_title, cos_lat_weights, map_grid, draw_zones, crop_south, desaturate, to_pm180, roll_lon, MAJOR_LABEL_MAP, NAMED_AMIP, AIMIP_MMM_LABEL, AMIP_MMM_LABEL)

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
aimip_all = load_aimip(lat, lon, BASELINE_START, BASELINE_END, average_members=False)

# Shift maps onto -180 to 180 longitude axis
lon, order = to_pm180(lon)
era5 = roll_lon(era5, order)
ocean = roll_lon(ocean, order)
aimip_all = roll_lon(aimip_all, order)

# Fixed 0K model sets
amip_0k_models = amip_with_scenario('0K')

print('Classifying members.')
kg = Cluster('koppen_geiger')
era5_class = kg.classify(era5['T'], era5['P'])

_, ref_major, colors = zone_masks(era5_class)
ref_major = ref_major.astype(float)
ref_valid = np.isfinite(ref_major)
weights = cos_lat_weights(lat, ref_major.shape)
zone_ids = sorted(MAJOR_LABEL_MAP)
muted = {zone_id: desaturate(colors[zone_id]) for zone_id in zone_ids}

def member_agreement(members):
    majors = []
    for scns in members.values():
        masked = mask_to_nan(scns['0K'], ocean)
        majors.append(kg.classify(masked['T'], masked['P'])['Major'].astype(float))
    return mode_map(np.stack(majors)), np.nanmean([agreement_pct(m, ref_major, weights) for m in majors])

aimip_members = group_by_member(aimip_all, ('0K',))
aimip_results = {m.key: member_agreement(aimip_members[m.key]) for m in AIMIP_MODELS}
del aimip_all, aimip_members
amip_results = {}
for key, members in iter_amip_members(lat, lon, ('0K',)):
    amip_results[key] = member_agreement(members)
    del members

def model_consensus(results):
    # Equal model weight regardless of member count
    return (mode_map(np.stack([r[0] for r in results.values()])),
            np.nanmean([r[1] for r in results.values()]))

compare = [('AIMIP: ' + m.display, *aimip_results[m.key]) for m in AIMIP_MODELS]
compare.append((AIMIP_MMM_LABEL, *model_consensus(aimip_results)))
compare += [('AMIP: ' + name, *amip_results[key]) for key, name in NAMED_AMIP.items()]
compare.append((AMIP_MMM_LABEL, *model_consensus(amip_results)))
assert len(compare) == N_MAPS

fig, gs, axes = map_grid(N_MAPS, N_COLS, hspace=0.19, wspace=0.04, top=0.958, bottom=0.02)
if N_MAPS % N_COLS:
    axes[-1].remove()
    axes[-1] = fig.add_subplot(gs[(N_MAPS - 1) // N_COLS, :])

seen_agree, seen_dest = set(), set()
for idx, (ax, (name, model_major, agreement)) in enumerate(zip(axes, compare)):
    valid = ref_valid & np.isfinite(model_major)
    agree = valid & (model_major == ref_major)
    disagree = valid & (model_major != ref_major)
    draw_zones(ax, lon, lat, np.where(agree, ref_major, np.nan), muted)
    draw_zones(ax, lon, lat, np.where(disagree, model_major, np.nan), colors)
    seen_agree.update(int(z) for z in np.unique(ref_major[agree]))
    seen_dest.update(int(z) for z in np.unique(model_major[disagree]))

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
ag = sorted((amip_results[m.key][1], m.display) for m in amip_0k_models)
print('AMIP agreement: min %s %.1f%%, max %s %.1f%% (n=%d)' % (ag[0][1], ag[0][0], ag[-1][1], ag[-1][0], len(ag)))
