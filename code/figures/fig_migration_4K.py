"""
Per-model maps of Koppen-Geiger zone migration at +4K SST forcing with origin and destination summary charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from code.cluster.cluster import Cluster
from code.figures.config import FIGURE_DIR, BASELINE_START, BASELINE_END
from code.figures.models import AIMIP_MODELS, amip_with_scenario
from code.figures.utils import (load_era5, load_aimip, iter_amip_members, load_era5_classification, group_by_member, slice_baseline, zone_masks, require_scns, surface_mask, cos_lat_weights, per_year_classes, mode_map, migration_destination, zone_migration_fractions, zone_arrival_fractions, desaturate, panel_title, crop_south, to_pm180, roll_lon, map_grid, draw_zones, ORIGIN_HATCHES, MAJOR_LABEL_MAP, NAMED_AMIP, NAMED_AMIP_STYLE, AMIP_MMM_LABEL)

# 5 AIMIP + 2 named AMIP + AMIP MMM = 8 maps on a 2-column grid.
N_MAPS = 8
N_COLS = 2
N_MAP_ROWS = 4

def model_modes(grouped, keys, cluster):
    """Per-member (0K mode, 4K mode) classification pair from per-year classes."""
    modes = {}
    for key in keys:
        scns = require_scns(grouped, key, ('0K', '4K'))
        ref_mode = mode_map(per_year_classes(scns['0K']['T'], scns['0K']['P'], cluster))
        scn_mode = mode_map(per_year_classes(scns['4K']['T'], scns['4K']['P'], cluster))
        modes[key] = (ref_mode, scn_mode)
    return modes

def amip_mmm_migration(modes, keys, land_mask):
    """AMIP multi-model modal reference map and its migration destination."""
    ref_arr = np.stack([modes[key][0] for key in keys], axis=0)
    scn_arr = np.stack([modes[key][1] for key in keys], axis=0)
    ref_mmm = mode_map(ref_arr)
    scn_mmm = mode_map(scn_arr)
    return ref_mmm, migration_destination(ref_mmm, scn_mmm, land_mask)

# Load data
print('Loading data.')
era5 = load_era5(BASELINE_START, BASELINE_END)
lat, lon = era5['lat'], era5['lon']
classification = load_era5_classification()

aimip_raw = load_aimip(lat, lon, average_members=False)

surface = surface_mask(lat, lon)
# Shift maps onto -180 to 180 longitude axis.
lon, order = to_pm180(lon)
era5 = roll_lon(era5, order)
classification = roll_lon(classification, order)
surface = roll_lon(surface, order)
for key in aimip_raw:
    aimip_raw[key] = roll_lon(aimip_raw[key], order)


masks, major, colors = zone_masks(classification)
major_ids = sorted(MAJOR_LABEL_MAP)
muted = {zone_id: desaturate(colors[zone_id]) for zone_id in major_ids}
land_mask = np.isfinite(major)
weights = cos_lat_weights(lat, surface.shape)

# Per-year classification and migration
print('Per-year classification.')
cluster = Cluster('koppen_geiger')

# Five AIMIP panels, 13 AMIP models with +4K, and two named AMIP references
aimip_models = [p.key for p in AIMIP_MODELS]
amip_models = [m.key for m in amip_with_scenario('4K')]
named_models = list(NAMED_AMIP)
display = {p.key: p.display for p in AIMIP_MODELS}
style = {p.key: (p.color, p.marker) for p in AIMIP_MODELS}

def member_migration(members):
    modes = model_modes(members, members, cluster)
    consensus = tuple(mode_map(np.stack([pair[i] for pair in modes.values()])) for i in (0, 1))
    origin = np.array([[fracs.get(z, np.nan) for z in major_ids] for fracs in
        (zone_migration_fractions(ref, scn, surface, weights, 'land') for ref, scn in modes.values())])
    destination = np.array([[fracs.get(z, np.nan) for z in major_ids] for fracs in
        (zone_arrival_fractions(ref, scn, surface, weights, 'land') for ref, scn in modes.values())])
    return consensus, origin, destination

print('Per-member migration.')
aimip_members = group_by_member(slice_baseline(aimip_raw), ('0K', '4K'))
aimip_modes, member_orig_fracs, member_dest_fracs = {}, {}, {}
for key in aimip_models:
    aimip_modes[key], member_orig_fracs[key], member_dest_fracs[key] = member_migration(aimip_members[key])
del aimip_raw, aimip_members

print('Loading and classifying AMIP members.')
amip_modes, amip_orig_fracs, amip_dest_fracs = {}, {}, {}
for key, members in iter_amip_members(lat, lon, ('0K', '4K'), max_workers=2, models=amip_with_scenario('4K')):
    amip_modes[key], origin, destination = member_migration(members)
    amip_orig_fracs[key] = dict(zip(major_ids, np.nanmean(origin, axis=0)))
    amip_dest_fracs[key] = dict(zip(major_ids, np.nanmean(destination, axis=0)))
    del members

amip_ref_mmm, amip_mig_mmm = amip_mmm_migration(amip_modes, amip_models, land_mask)
aimip_dest = {key: migration_destination(ref, scn, land_mask) for key, (ref, scn) in aimip_modes.items()}
aimip_orig_fracs = {key: dict(zip(major_ids, np.nanmean(vals, axis=0))) for key, vals in member_orig_fracs.items()}
aimip_dest_fracs = {key: dict(zip(major_ids, np.nanmean(vals, axis=0))) for key, vals in member_dest_fracs.items()}

entries = [(display[model_key], aimip_dest[model_key], aimip_modes[model_key][0]) for model_key in aimip_models]
for model_key in named_models:
    ref_mode, scn_mode = amip_modes[model_key]
    entries.append((NAMED_AMIP[model_key], migration_destination(ref_mode, scn_mode, land_mask), ref_mode))
entries.append((AMIP_MMM_LABEL, amip_mig_mmm, amip_ref_mmm))
assert len(entries) == N_MAPS

# Map rows, legend row, and summary-chart row
fig, gs, map_axes = map_grid(N_MAPS, N_COLS, extra_ratios=(0.35, 1.1), hspace=0.18, wspace=0.05, left=0.012, right=0.992, top=0.96, bottom=0.04)
ax_legend = fig.add_subplot(gs[N_MAP_ROWS, :])
ax_legend.axis('off')

bottom_gs = gs[N_MAP_ROWS + 1, :].subgridspec(1, 4, wspace=0.3, width_ratios=[0.02, 1.0, 1.0, 0.40])
ax_origin = fig.add_subplot(bottom_gs[0, 1])
ax_dest = fig.add_subplot(bottom_gs[0, 2])
ax_models = fig.add_subplot(bottom_gs[0, 3])
ax_models.axis('off')

# Draw maps
plt.rcParams['hatch.linewidth'] = 0.25
all_dest, all_orig = set(), set()
for i, (label, dest_map, ref_mode) in enumerate(entries):
    ax = map_axes[i]
    changed = dest_map > 0

    draw_zones(ax, lon, lat, np.where(land_mask, ref_mode, np.nan), muted)
    draw_zones(ax, lon, lat, np.where(changed, dest_map, np.nan), colors)

    for zone_id in major_ids:
        orig = changed & (ref_mode == zone_id)
        if not orig.any():
            continue
        contour = ax.contourf(lon, lat, orig.astype(float), levels=[0.5, 1.5], hatches=[ORIGIN_HATCHES.get(zone_id, '///')], colors='none')
        contour.set_rasterized(True)

    if changed.any():
        all_dest.update(int(value) for value in np.unique(dest_map[changed]) if value > 0)
        all_orig.update(int(value) for value in np.unique(ref_mode[changed]))

    crop_south(ax)
    title_kw = {'fontweight': 'bold'} if label == AMIP_MMM_LABEL else {}
    panel_title(ax, i, label, title_kw=title_kw)
    ax.set_xticks([]); ax.set_yticks([])

# Summary panels
x_pos = np.arange(len(major_ids))

def draw_summary(ax, aimip_fracs, amip_fracs, member_fracs, ylabel, idx, title):
    """Per-zone scatter with AMIP model and AIMIP paired-member min-max ranges."""
    for xi, zone_id in enumerate(major_ids):
        vals = [fracs[zone_id] for fracs in amip_fracs.values() if zone_id in fracs and np.isfinite(fracs[zone_id])]
        if vals:
            ax.fill_between([xi - 0.4, xi + 0.4], [min(vals)] * 2, [max(vals)] * 2,
                            color='0.7', alpha=0.45, linewidth=0, zorder=1)
    for model_key in aimip_models:
        color, marker = style[model_key]
        ys = [aimip_fracs[model_key].get(zone_id, np.nan) for zone_id in major_ids]
        vals = member_fracs[model_key]
        for xi, lower, upper in zip(x_pos, np.nanmin(vals, axis=0), np.nanmax(vals, axis=0)):
            ax.fill_between([xi - 0.25, xi + 0.25], lower, upper,
                            color=color, alpha=0.35, linewidth=0, zorder=2)
        ax.scatter(x_pos, ys, color=color, marker=marker, s=50, edgecolor='k', linewidths=0.4, zorder=4)
    for model_key in named_models:
        color, marker = NAMED_AMIP_STYLE[model_key]
        ys = [amip_fracs[model_key].get(zone_id, np.nan) for zone_id in major_ids]
        ax.scatter(x_pos, ys, color=color, marker=marker, s=80, edgecolor='k', linewidths=0.5, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MAJOR_LABEL_MAP[zone_id] for zone_id in major_ids], rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    panel_title(ax, idx, title)

draw_summary(ax_origin, aimip_orig_fracs, amip_orig_fracs, member_orig_fracs, 'Land Leaving Zone (% of zone area)', N_MAPS, 'Origin Zones')
draw_summary(ax_dest, aimip_dest_fracs, amip_dest_fracs, member_dest_fracs, 'Land Entering Zone (% of total land)', N_MAPS + 1, 'Destination Zones')

# Legends
zones_changed = [zone_id for zone_id in major_ids if zone_id in all_dest or zone_id in all_orig]
map_patches = []
for zone_id in zones_changed:
    map_patches.append(Patch(facecolor=colors[zone_id], edgecolor='k', label='Migrated to ' + MAJOR_LABEL_MAP[zone_id]) if zone_id in all_dest else Patch(facecolor='none', edgecolor='none', label=' '))
    map_patches.append(Patch(facecolor='white', edgecolor='k', hatch=ORIGIN_HATCHES.get(zone_id, '///'), label='Was ' + MAJOR_LABEL_MAP[zone_id]) if zone_id in all_orig else Patch(facecolor='none', edgecolor='none', label=' '))
if map_patches:
    ax_legend.legend(handles=map_patches, loc='upper center', ncol=len(zones_changed), borderaxespad=0, columnspacing=0.8, handlelength=1.1, handletextpad=0.4)

summary_handles = [Patch(facecolor='0.7', edgecolor='none', alpha=0.45, label='AMIP range')]
summary_handles += [Patch(facecolor='0.4', edgecolor='none', alpha=0.35, label='AIMIP member\nmin–max')]
summary_handles += [Line2D([0], [0], color=style[model_key][0], marker=style[model_key][1], markersize=7, markeredgecolor='k', markeredgewidth=0.4, linestyle='None', label=display[model_key]) for model_key in aimip_models]
summary_handles += [Line2D([0], [0], color=NAMED_AMIP_STYLE[model_key][0], marker=NAMED_AMIP_STYLE[model_key][1], markersize=9, markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label=NAMED_AMIP[model_key]) for model_key in named_models]
leg_trans = mtransforms.blended_transform_factory(fig.transFigure, ax_models.transAxes)
ax_models.legend(handles=summary_handles, loc='center right', bbox_to_anchor=(0.992, 0, 0, 1), bbox_transform=leg_trans, borderaxespad=0, title='Models')

fig.suptitle('Zone Migration at +4K SST Forcing', y=1)
out_path = FIGURE_DIR / 'migration_4K.pdf'
fig.savefig(out_path, bbox_inches='tight', dpi=150)
plt.close(fig)
print('Saved ' + str(out_path))
