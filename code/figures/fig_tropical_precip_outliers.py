"""Compare full, outlier-excluded, and interdecile AMIP tropical precipitation ranges."""

import numpy as np
import matplotlib.pyplot as plt
from code import data
from code.cluster.cluster import Cluster
from code.figures.config import AIMIP_DIR, BASELINE_START, BASELINE_END, FIGURE_DIR, LAND_MASK_PATH
from code.figures.models import AIMIP_MODELS, amip_with_scenario
from code.figures.utils import (FONT_SIZE, NAMED_AMIP, NAMED_AMIP_STYLE, SST_FORCING_MAP, cos_lat_weights, group_by_member, iter_amip_members, load_era5, mean_response_curve, mode_map, panel_label, per_year_classes, zone_annual_diffs)

def tropical_curves(members, weights):
    """Same member-first, own-baseline tropical land diagnostic as forcing_cc."""
    curves = []
    cluster = Cluster('koppen_geiger')
    for scns in members.values():
        baseline = scns['0K']
        mask = mode_map(per_year_classes(baseline['T'], baseline['P'], cluster)) == 1
        table = zone_annual_diffs(scns, {1: mask}, weights)
        points = [(0.0, 0.0)] + [(SST_FORCING_MAP[s], zones[1]['P']) for s, zones in table.items() if np.isfinite(zones[1]['P'])]
        assert len(points) > 1, 'No finite tropical precipitation response.'
        curves.append(tuple(np.asarray(sorted(points)).T))
    return curves


def envelope(ax, curves, percentiles=(0, 100), **kwargs):
    grid = np.unique(np.concatenate([x for x, _ in curves]))
    values = np.array([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in curves])
    lower, upper = np.nanpercentile(values, percentiles, axis=0)
    ax.fill_between(grid, lower, upper, edgecolor='none', linewidth=0, **kwargs)


def outlier_rows(amip):
    """Tukey fences across model means, separately at each observed nonzero forcing."""
    rows = []
    for forcing in (-4, 4):
        values = {key: float(y[x == forcing][0]) for key, (x, y) in amip.items() if forcing in x}
        q1, q3 = np.percentile(list(values.values()), [25, 75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.extend(dict(model=key, forcing_K=forcing, response_mm_month=value, lower_fence=lower, upper_fence=upper, outlier=value < lower or value > upper) for key, value in values.items())
    return rows


def plot(amip, aimip):
    rows = outlier_rows(amip)
    excluded = {r['model'] for r in rows if r['outlier']}
    retained = {key: curve for key, curve in amip.items() if key not in excluded}
    assert retained, 'Outlier screening removed every model'

    panels = [(amip, 'All AMIP Models', (0, 100)), (retained, f'Outliers Excluded (n={len(retained)})', (0, 100)), (amip, 'Interdecile Range', (10, 90))]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True)
    greys = {'NCAR_CESM2': '0.6', 'NOAA-GFDL_GFDL-CM4': '0.75'}
    for idx, (ax, (curves, title, percentiles)) in enumerate(zip(axes, panels)):
        panel_label(ax, idx, fontsize=FONT_SIZE + 3)
        envelope(ax, list(curves.values()), percentiles=percentiles, color='0.8', alpha=0.6,
                 label='AMIP range (as titled)', zorder=1)
        for key in NAMED_AMIP:
            if key in curves:
                ax.plot(*curves[key], color=greys[key], linestyle='--', linewidth=2, marker=NAMED_AMIP_STYLE[key][1], markersize=7, markeredgecolor='k', markeredgewidth=0.4, label=NAMED_AMIP[key], zorder=4)
        for model in AIMIP_MODELS:
            members = aimip[model.key]
            envelope(ax, members, color=model.color, alpha=0.4, zorder=2)
            ax.plot(*mean_response_curve(members), color=model.color, marker=model.marker, markersize=4, linewidth=1, markeredgecolor='k', markeredgewidth=0.3, label=model.display, zorder=5)
        ax.axhline(0, color='k', linewidth=0.4, linestyle=':')
        ax.axvline(0, color='k', linewidth=0.4, linestyle=':')
        ax.set_title(title, fontsize=FONT_SIZE + 3)
        ax.set(xlabel='SST Forcing (K)', xticks=[-4, -2, 0, 2, 4])
    axes[0].set_ylabel(r'$\Delta$ Precipitation (mm/month)')
    fig.suptitle('Tropical (A) Precipitation Forcing Response Outlier Sensitivity',
                 fontsize=FONT_SIZE + 5, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=4)
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])
    out = FIGURE_DIR / f'{'trop_p_outliers'}.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print(f'Saved {out}', flush=True)
    plt.close(fig)


def main():
    print('Loading grid.', flush=True)
    era5 = load_era5(BASELINE_START, BASELINE_END)
    lat, lon = era5['lat'], era5['lon']
    del era5
    weights = cos_lat_weights(lat, (lat.size, lon.size))
    amip = {}
    for key, members in iter_amip_members(lat, lon, models=amip_with_scenario('4K'), land_mask_path=LAND_MASK_PATH, max_workers=1):
        amip[key] = mean_response_curve(tropical_curves(members, weights))
        del members
    aimip = {}
    for model in AIMIP_MODELS:
        print(f'Loading AIMIP {model.display}.', flush=True)
        institution, name = model.key.split('_', 1)
        raw = data.load_aimip(AIMIP_DIR, institution=institution, model=name, target_lat=lat, target_lon=lon, start_year=BASELINE_START, end_year=BASELINE_END, land_mask_path=LAND_MASK_PATH, max_workers=1)
        members = group_by_member(raw, ('0K', '2K', '4K'))[model.key]
        aimip[model.key] = tropical_curves(members, weights)
        del raw, members
    plot(amip, aimip)


if __name__ == '__main__':
    main()
