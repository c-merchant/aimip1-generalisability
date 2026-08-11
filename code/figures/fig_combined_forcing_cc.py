"""
Per-zone forcing response (T, P) and Clausius-Clapeyron humidity scaling across SST forcings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from code.cluster.cluster import Cluster
from code.figures.config import FIGURE_DIR, LAND_MASK_PATH, BASELINE_START, BASELINE_END
from code.figures.utils import (
    load_era5, load_aimip, load_amip, group_by_model,
    slice_baseline, zone_annual_diffs, require_scns, per_year_classes, mode_map,
    cos_lat_weights, weighted_cell_mean, panel_label, FIG_WIDTH, SST_FORCING_MAP, MAJOR_LABEL_MAP,
    AIMIP_MODELS, AMIP_MODELS, NAMED_AMIP, NAMED_AMIP_STYLE,
)

# 3 variable rows (T, P, humidity) x 5 Koppen major-zone columns.
N_COLS = 5

def forcing_response_curve(scn_zone_diffs, var, zone_id):
    """(forcings, responses) curve anchored at (0, 0), or None if fewer than two points."""
    forcings, responses = [0.0], [0.0]
    for scn, zones in scn_zone_diffs.items():
        forcing = SST_FORCING_MAP.get(scn)
        if forcing is None or scn == '0K':
            continue
        response = zones.get(zone_id, {}).get(var)
        if response is not None and np.isfinite(response):
            forcings.append(forcing)
            responses.append(response)
    if len(forcings) < 2:
        return None
    order = np.argsort(forcings)
    return np.array(forcings)[order], np.array(responses)[order]

def humidity_curve(scns, mask, weights):
    """(forcings, pct change from 0K) humidity curve per zone."""
    cell_weights = weights[mask]
    q_base = weighted_cell_mean(scns['0K']['Q'][:, :, mask], cell_weights)
    if not np.isfinite(q_base) or q_base == 0:
        return None
    forcings, pct_changes = [0.0], [0.0]
    for scn, ds in scns.items():
        forcing = SST_FORCING_MAP.get(scn)
        if forcing is None or scn == '0K':
            continue
        q_scn = weighted_cell_mean(ds['Q'][:, :, mask], cell_weights)
        if np.isfinite(q_scn):
            forcings.append(forcing)
            pct_changes.append(100.0 * (q_scn - q_base) / q_base)
    if len(forcings) < 2:
        return None
    order = np.argsort(forcings)
    return np.array(forcings)[order], np.array(pct_changes)[order]

def draw_envelope(ax, curves, color='0.8', alpha=0.6, label=None):
    """Shade the min-max envelope across (forcings, responses) curves on shared x-grid."""
    if not curves:
        return
    grid_x = np.array(sorted(set(x for forcings, _ in curves for x in forcings)), dtype=float)
    interp_rows = []
    for forcings, responses in curves:
        # np.interp clamps outside the curve's own range, so null those points: a model lacking a forcing
        # would otherwise sit flat at its nearest value (e.g. the 0K anchor) and widen the envelope
        y_grid = np.interp(grid_x, forcings, responses)
        y_grid[(grid_x < forcings[0]) | (grid_x > forcings[-1])] = np.nan
        interp_rows.append(y_grid)
    interp_stack = np.array(interp_rows)
    lower, upper = np.nanmin(interp_stack, axis=0), np.nanmax(interp_stack, axis=0)
    ax.fill_between(grid_x, lower, upper, color=color, alpha=alpha, label=label, zorder=1)

def baseline_zone_masks(scns, cluster):
    """Per-zone boolean masks from a model's own 0K modal classification."""
    major = mode_map(per_year_classes(scns['0K']['T'], scns['0K']['P'], cluster))
    return {mid: (major == mid) for mid in sorted(MAJOR_LABEL_MAP)}

# Load data
print('Loading data.')
era5 = load_era5(BASELINE_START, BASELINE_END)
lat, lon = era5['lat'], era5['lon']
weights = cos_lat_weights(lat, (lat.size, lon.size))

aimip_all = load_aimip(lat, lon, land_mask_path=LAND_MASK_PATH)
amip_all = load_amip(lat, lon, land_mask_path=LAND_MASK_PATH)
aimip_resp = slice_baseline(aimip_all)

aimip_grouped = group_by_model(aimip_resp)
amip_grouped = group_by_model(amip_all)

# Five AIMIP curves per panel and the full AMIP set behind the envelope
aimip_models = [p.key for p in AIMIP_MODELS]
amip_models = [m.key for m in AMIP_MODELS]
display = {p.key: p.display for p in AIMIP_MODELS}
style = {p.key: (p.color, p.marker) for p in AIMIP_MODELS}

# Per-model own-baseline zones
print('Classifying per-model baseline zones.')
cluster = Cluster('koppen_geiger')
aimip_masks = {k: baseline_zone_masks(require_scns(aimip_grouped, k, ('0K',)), cluster) for k in aimip_models}
amip_masks = {k: baseline_zone_masks(amip_grouped[k], cluster) for k in amip_models}

# Per-zone annual diffs
print('Computing per-zone annual diffs.')
aimip_table = {k: zone_annual_diffs(aimip_grouped[k], aimip_masks[k], weights) for k in aimip_models}
amip_table = {k: zone_annual_diffs(amip_grouped[k], amip_masks[k], weights) for k in amip_models}

NAMED_GREY = {'NCAR_CESM2': '0.6', 'NOAA-GFDL_GFDL-CM4': '0.75'}
REF_LW = 2.2

active_zones = sorted(MAJOR_LABEL_MAP)
n_cols = N_COLS
assert len(active_zones) == n_cols

# Plot
print('Plotting.')
fig, axes = plt.subplots(3, n_cols, figsize=(1.25 * FIG_WIDTH, 7.5), sharex='all', squeeze=False)

last_col = n_cols - 1
for row in range(3):
    for col in range(1, n_cols):
        if row == 0 and col == last_col:
            continue
        axes[row, col].sharey(axes[row, 0])

row_specs = [('T', 'Temperature', 'K'), ('P', 'Precipitation', 'mm/month'), ('huss', 'Specific humidity', 'kg/kg')]

for row, (var, var_label, unit) in enumerate(row_specs):
    for col, zone_id in enumerate(active_zones):
        ax = axes[row, col]

        if row < 2:
            if row == 0:
                # Reference scalings for land warming (1:1 with SST forcing and +50% land amplification).
                x_ref = np.array([-4.0, 0.0, 2.0, 4.0])
                ax.plot(x_ref, x_ref, color='k', linestyle=(0, (5, 2)), linewidth=REF_LW, label='1:1 with SST' if col == 0 else None, zorder=2)
                ax.plot(x_ref, 1.5 * x_ref, color='k', linestyle=(0, (1, 1.2)), linewidth=REF_LW, label='+50% land warming' if col == 0 else None, zorder=2)

            amip_curves = [c for c in (forcing_response_curve(amip_table[k], var, zone_id) for k in amip_models) if c is not None]
            draw_envelope(ax, amip_curves, label='AMIP range' if (row == 0 and col == 0) else None)

            for model_key in aimip_models:
                curve = forcing_response_curve(aimip_table[model_key], var, zone_id)
                if curve is None:
                    continue
                forcings, responses = curve
                color, marker = style[model_key]
                label = display[model_key] if (row == 0 and col == 0) else None
                ax.plot(forcings, responses, color=color, linewidth=1.5, marker=marker, markersize=6, markeredgecolor='k', markeredgewidth=0.3, label=label, zorder=3)

            for model_key in NAMED_AMIP:
                curve = forcing_response_curve(amip_table[model_key], var, zone_id)
                if curve is None:
                    continue
                forcings, responses = curve
                color, marker = NAMED_GREY[model_key], NAMED_AMIP_STYLE[model_key][1]
                label = NAMED_AMIP[model_key] if (row == 0 and col == 0) else None
                ax.plot(forcings, responses, color=color, linewidth=2.0, linestyle='--', marker=marker, markersize=7, markeredgecolor='k', markeredgewidth=0.4, label=label, zorder=4)
        else:
            # Compounded CC scaling: dq/q = 1.07^dT - 1, not linear 7*dT
            x_cc = np.linspace(-4, 4, 100)
            ax.plot(x_cc, 100.0 * (1.07 ** x_cc - 1.0), 'k--', linewidth=REF_LW, label='Clausius-Clapeyron 7%/K' if col == 0 else None, zorder=2)

            amip_curves = [c for c in (humidity_curve(amip_grouped[k], amip_masks[k][zone_id], weights) for k in amip_models) if c is not None]
            draw_envelope(ax, amip_curves)

            for model_key in aimip_models:
                curve = humidity_curve(aimip_grouped[model_key], aimip_masks[model_key][zone_id], weights)
                if curve is None:
                    continue
                forcings, pct_changes = curve
                color, marker = style[model_key]
                ax.plot(forcings, pct_changes, color=color, linewidth=1.5, marker=marker, markersize=6, markeredgecolor='k', markeredgewidth=0.3, zorder=3)

            for model_key in NAMED_AMIP:
                curve = humidity_curve(amip_grouped[model_key], amip_masks[model_key][zone_id], weights)
                if curve is None:
                    continue
                forcings, pct_changes = curve
                color, marker = NAMED_GREY[model_key], NAMED_AMIP_STYLE[model_key][1]
                ax.plot(forcings, pct_changes, color=color, linewidth=2.0, linestyle='--', marker=marker, markersize=7, markeredgecolor='k', markeredgewidth=0.4, zorder=4)

        ax.axhline(0, color='k', linewidth=0.4, linestyle=':')
        ax.axvline(0, color='k', linewidth=0.4, linestyle=':')
        panel_label(ax, row * n_cols + col)

        if row == 0:
            ax.set_title(MAJOR_LABEL_MAP.get(zone_id, str(zone_id)), fontweight='bold')
        if row == 2:
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_xlabel('SST Forcing (K)')
        if col == 0:
            ax.set_ylabel('$\\Delta$ ' + var_label + ' (' + unit + ')' if row < 2 else '$\\Delta q$ (%)')
        elif row == 0 and col == last_col:
            # Individual polar temperature scale
            ax.yaxis.tick_right()
        else:
            ax.tick_params(labelleft=False)

handles = [Line2D([0], [0], color=style[model_key][0], marker=style[model_key][1], markersize=6, markeredgecolor='k', markeredgewidth=0.3, linewidth=1.5, label=display[model_key]) for model_key in aimip_models]
handles.insert(0, plt.Rectangle((0, 0), 1, 1, color='0.8', alpha=0.6, label='AMIP range'))
handles += [Line2D([0], [0], color=NAMED_GREY[model_key], marker=NAMED_AMIP_STYLE[model_key][1], markersize=7, markeredgecolor='k', markeredgewidth=0.4, linewidth=2.0, linestyle='--', label=NAMED_AMIP[model_key]) for model_key in NAMED_AMIP]
handles.append(Line2D([0], [0], color='k', linestyle=(0, (5, 2)), linewidth=REF_LW, label='1:1 with SST'))
handles.append(Line2D([0], [0], color='k', linestyle=(0, (1, 1.2)), linewidth=REF_LW, label='+50% land warming'))
handles.append(Line2D([0], [0], color='k', linestyle='--', linewidth=REF_LW, label='Clausius-Clapeyron 7%/K'))
ncol = int(np.ceil(len(handles) / 2))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=ncol)

fig.suptitle('Per-Zone Forcing Response and Clausius-Clapeyron Scaling', y=1.005)
fig.tight_layout(h_pad=2.0, rect=[0, 0.08, 1, 1])
out = FIGURE_DIR / 'combined_forcing_cc.pdf'
fig.savefig(out, bbox_inches='tight', dpi=300)
plt.close(fig)
print('Saved ' + str(out))
