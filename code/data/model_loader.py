"""
AIMIP and AMIP model-tree loaders and ensemble averaging.
"""

import os
import logging
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

from code.config import SKIP_MODELS
from .era5_loader import _to_arrays
from .loader_utils import add_humidity_datasets, concat_time, load_land_mask, regrid_to_target

# Scenario dir names and respective forcing labels (MD-1p5 spells warming scenario without 'p')
AIMIP_SCENARIOS = {'aimip': '0K', 'aimip-p2k': '2K', 'aimip-p4k': '4K', 'aimip-2k': '2K', 'aimip-4k': '4K'}
AMIP_SCENARIOS = {'amip': '0K', 'amip-p4K': '4K', 'amip-m4K': '-4K'}


def load_aimip(path, **kwargs):
    """Load AIMIP models, skipping SKIP_MODELS; options as in load_model_tree."""
    return load_model_tree(path, AIMIP_SCENARIOS, skip_models=SKIP_MODELS, label='AIMIP', **kwargs)


def load_amip(path, **kwargs):
    """Load AMIP models; unmapped scenario dirs (e.g. amip-future4K) are skipped."""
    return load_model_tree(path, AMIP_SCENARIOS, label='AMIP', **kwargs)


def load_model_tree(path, scenario_map, skip_models=frozenset(), label='model',
                    institution=None, model=None, scenario=None, ensemble=None,
                    start_year=None, end_year=None, land_mask_path=None,
                    target_lat=None, target_lon=None, max_workers=None):
    """Sweep {path}/inst/model/scenario/ensemble/var/*.nc and load every dataset with both
    tas and pr (plus any humidity inputs as Q), keyed {inst}_{model}_{scenario}_{ens}."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Land mask, regridded if necessary
    land_mask = None
    if land_mask_path is not None:
        lsm = load_land_mask(land_mask_path)
        land_mask = lsm.values
        if target_lat is not None and land_mask.shape != (len(target_lat), len(target_lon)):
            land_mask = regrid_to_target(lsm, target_lat, target_lon).values

    # Catalog files per dataset and variable
    catalog = defaultdict(lambda: defaultdict(list))
    for nc_file in sorted(data_path.rglob('*.nc')):
        if len(nc_file.parts) < 7:
            continue
        inst, mod, scen_dir, ens, var = nc_file.parts[-6:-1]
        scen = scenario_map.get(scen_dir)
        if scen is None or mod in skip_models:
            continue
        if (institution and inst != institution) or (model and mod != model) or (scenario and scen != scenario) or (ensemble and ens != ensemble):
            continue
        catalog[f'{inst}_{mod}_{scen}_{ens}'][var].append(nc_file)

    if not catalog:
        raise ValueError(f"No {label} data found under {path}")

    # Keeping one grid of a given var
    for key, file_lists in catalog.items():
        for var, files in file_lists.items():
            grids = {g for g in (_grid_of(f) for f in files) if g}
            if len(grids) > 1:
                keep = next((g for g in sorted(grids) if g.startswith('gr')), 'gn' if 'gn' in grids else sorted(grids)[0])
                file_lists[var] = [f for f in files if _grid_of(f) == keep]
                logging.debug("Multiple grids %s for %s/%s; keeping %s", sorted(grids), key, var, keep)

    tasks = [(key, dict(file_lists)) for key, file_lists in catalog.items() if 'tas' in file_lists and 'pr' in file_lists]
    args = (start_year, end_year, land_mask, target_lat, target_lon)
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)

    loaded = {}
    if max_workers <= 1 or len(tasks) <= 1:
        for key, file_lists in tasks:
            try:
                loaded[key] = _load_one(file_lists, *args)
            except (OSError, ValueError) as e:
                logging.warning("Skipping %s: %s", key, e)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load_one, file_lists, *args): key for key, file_lists in tasks}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    loaded[key] = fut.result()
                except (OSError, ValueError) as e:
                    logging.warning("Skipping %s: %s", key, e)

    # Catalog order
    return {k: loaded[k] for k, _ in tasks if k in loaded}


def ensemble_mean(loaded):
    """Average T/P/Q across members sharing {inst}_{model}_{scenario}. Members are truncated
    to the shortest one."""
    groups = defaultdict(list)
    for key, ds in loaded.items():
        groups[key.rsplit('_', 1)[0]].append(ds)

    result = {}
    for name, members in groups.items():
        ref = members[0]
        if any(m['T'].shape[1:] != ref['T'].shape[1:] for m in members):
            raise ValueError(f"{name} ensemble members are on different grids")
        n_years = min(m['T'].shape[0] for m in members)
        result[name] = {v: np.nanmean([m[v][:n_years] for m in members], axis=0) for v in ('T', 'P', 'Q') if v in ref}
        result[name].update(lat=ref['lat'], lon=ref['lon'], years=n_years)
    return result


def _grid_of(path):
    # CMIP6 filename: var_table_model_scenario_ensemble_grid[_timerange].nc
    for tok in Path(path).stem.split('_'):
        if tok[:1] == 'g' and tok[1:].rstrip('0123456789') in ('n', 'r', 'm'):
            return tok
    return None


def _load_one(file_lists, start_year, end_year, land_mask, target_lat, target_lon):
    """Open and concatenate one dataset then standardize."""
    datasets = {var: concat_time([xr.open_dataset(f)[var] for f in file_lists[var]]) for var in ('tas', 'pr')}
    add_humidity_datasets(datasets, file_lists, concat_time)
    return _to_arrays(datasets, start_year=start_year, end_year=end_year, land_mask=land_mask, target_lat=target_lat, target_lon=target_lon)
