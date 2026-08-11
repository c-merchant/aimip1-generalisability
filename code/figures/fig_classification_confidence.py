"""
Baseline (0K) temporal classification confidence maps for ERA5, AIMIP models, and named AMIP references.
"""

import numpy as np
import matplotlib.pyplot as plt

from code.cluster.cluster import Cluster
from code.figures.config import FIGURE_DIR, BASELINE_START, BASELINE_END
from code.figures.utils import (
    load_era5, load_aimip, load_amip, load_era5_classification, group_by_model,
    slice_baseline, zone_masks, per_year_classes, require_scns,
    modal_fraction, panel_title, crop_south, to_pm180, roll_lon, map_grid,
    AIMIP_MODELS, NAMED_AMIP,
)

CMAP = 'RdYlGn'
VMIN, VMAX = 0.0, 1.0

# ERA5 + 5 AIMIP + 2 named AMIP = 8 maps on 3-column grid.
N_MAPS = 8
N_COLS = 3

# Load data
print('Loading data.')
era5 = load_era5(BASELINE_START, BASELINE_END)
lat, lon = era5['lat'], era5['lon']
classification = load_era5_classification()

aimip_all = slice_baseline(load_aimip(lat, lon))
amip_all = load_amip(lat, lon)

lon, order = to_pm180(lon)
era5 = roll_lon(era5, order)
classification = roll_lon(classification, order)
aimip_all = roll_lon(aimip_all, order)
amip_all = roll_lon(amip_all, order)

_, major, _ = zone_masks(classification)
land_mask = np.isfinite(major)

# Per-model temporal confidence
print('Per-year classification.')
cluster = Cluster('koppen_geiger')


def confidence(dataset):
    """Per-cell fraction of baseline years matching the modal major zone over land."""
    per_year = per_year_classes(dataset['T'], dataset['P'], cluster)
    confidence_map = modal_fraction(per_year)
    confidence_map[~land_mask] = np.nan
    return confidence_map


aimip_grouped = group_by_model(aimip_all)
amip_grouped = group_by_model(amip_all)

# ERA5, five AIMIP models, then two named AMIP references.
entries = [('ERA5', confidence(era5))]
for model in AIMIP_MODELS:
    entries.append((model.display, confidence(require_scns(aimip_grouped, model.key, ('0K',))['0K'])))
for key, display in NAMED_AMIP.items():
    entries.append((display, confidence(require_scns(amip_grouped, key, ('0K',))['0K'])))
assert len(entries) == N_MAPS

fig, _, axes = map_grid(N_MAPS, N_COLS, bottom=0.10)

mesh = None
for index, (label, confidence_map) in enumerate(entries):
    ax = axes[index]
    mesh = ax.pcolormesh(lon, lat, confidence_map, cmap=CMAP, vmin=VMIN, vmax=VMAX, shading='auto', rasterized=True)
    crop_south(ax)
    is_era5 = label == 'ERA5'
    panel_title(ax, index, label, title_kw={'fontweight': 'bold'} if is_era5 else None)
    ax.set_xticks([]); ax.set_yticks([])

colorbar_ax = fig.add_axes([0.25, 0.035, 0.5, 0.018])
colorbar = fig.colorbar(mesh, cax=colorbar_ax, orientation='horizontal')
colorbar.set_label('Temporal Classification Confidence (Modal Zone Year Fraction)')

fig.suptitle('Baseline Köppen-Geiger Zone Temporal Confidence', y=0.995)
output_path = FIGURE_DIR / 'classification_confidence.pdf'
fig.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close(fig)
print('Saved ' + str(output_path))
