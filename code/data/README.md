# Data
Module for preparing surface field data for clustering and subsequent analysis.

## Approach

We load ERA5, AIMIP, and AMIP tas (temperature) and pr (precipitation) data from NetCDF files and process them into standardized numpy arrays with shape `(years, 12, lat, lon)`. Where humidity inputs exist (huss, tdas+ps, or 3D hus), they are loaded alongside as `Q` (surface specific humidity).

```python
from code import data

# Load with optional filters (institution/model/scenario/ensemble, year range, target grid, land mask)
loaded = data.load_aimip('/path/to/aimip/data', scenario='0K')

# Average ensemble members sharing the same (institution, model, scenario)
means = data.ensemble_mean(loaded)
```

`load_aimip` and `load_amip` are wrappers around `load_model_tree`, which scans an `Org/Model/scenario/ensemble/var/file.nc` tree and only keeps datasets with both tas and pr. `load_era5` scans `{path}/{var}/mon/*.nc`.

Run anything that imports this module through `uv run` so it picks up the project environment (see the [root README](../../README.md) for `uv sync`):

```bash
uv run python -m pytest test/test_data    # unit tests for this module
```

The HEALPix regridding path needs the optional `healpix` extra (`uv sync --extra healpix`), which installs `earth2grid` from GitHub and therefore requires uv rather than pip.

## File structure

- era5_loader.py: `load_era5` and `_to_arrays` (shared standardization: units, whole calendar years, regridding, land mask).
- model_loader.py: `load_aimip` / `load_amip` / `load_model_tree` and `ensemble_mean`.
- loader_utils.py: time concatenation, land-mask loading, regridding (incl. HEALPix), humidity fallback.

Note: unit tests can be found in `../../test/test_data/`

## References

Data sources:
- ERA5: ECMWF Reanalysis v5
- AIMIP: Artificial Intelligence Atmospheric Model Intercomparison Project
- AMIP: Atmospheric Model Intercomparison Project
