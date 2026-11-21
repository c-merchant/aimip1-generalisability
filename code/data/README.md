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

Run anything that imports this module through `uv run`, so it picks up the project environment (see the [root README](../../README.md) for `uv sync`):

```bash
uv run python -m pytest test/test_data    # unit tests for this module
```

The HEALPix regridding path needs the optional `healpix` extra (`uv sync --extra healpix`), which installs `earth2grid` from GitHub which requires uv as opposed to pip.

## File structure

- era5_loader.py: `load_era5` and `_to_arrays` (shared standardization: units, whole calendar years, regridding, land mask).
- model_loader.py: `load_aimip` / `load_amip` / `load_model_tree` and `ensemble_mean`.
- loader_utils.py: time concatenation, land-mask loading, regridding (incl. HEALPix), humidity fallback.

Unit tests can be found in `../../test/test_data/`.

## Required data

Set the data paths in [`code/figures/config.py`](../figures/config.py).

**ERA5:** Monthly `tas`, `pr`, and a land-sea mask on a 1×1 grid, available from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

```text
<ERA5_DIR>/
├── tas/mon/*.nc
├── pr/mon/*.nc
└── lsm/mon/landseamask_1x1.nc
```

**AIMIP Phase 1:** AI-model output is available from DKRZ's S3 `ai-mip` endpoint (`https://s3.eu-dkrz-1.dkrz.cloud`). Replace the placeholders before running:

```bash
aws s3 sync --no-sign-request --endpoint-url https://s3.eu-dkrz-1.dkrz.cloud \
    's3://ai-mip/<Org>/<Model>/' '<AIMIP_DIR>/<Org>/<Model>/'
```

Remote keys follow `Org/Model/scenario/ensemble/Amon/var/grid[/version]/file.nc`. After downloading, remove the `Amon`, grid, and version directory levels so the loaders just see `Org/Model/scenario/ensemble/var/file.nc`.

We use `aimip` (+0K baseline), `aimip-p2k` (+2K), and `aimip-p4k` (+4K). MD-1p5 uses `aimip-2k` and `aimip-4k`; both conventions are accepted. Classification requires `tas` and `pr`, so models without `pr` (ArchesWeather/ArchesWeatherGen) are excluded. Humidity data are also needed for the humidity analysis.

**AMIP (CMIP6):** Physics-based model reference data are available from [ESGF](https://metagrid.esgf-west.org/search/cmip6/). We use monthly `tas` and `pr` from `amip` (+0K), `amip-m4k` (−4K), and `amip-p4k` (+4K), staged as `Org/Model/scenario/ensemble/var/file.nc`.

## References

Data sources:
- ERA5: ECMWF Reanalysis v5
- AIMIP: Artificial Intelligence Atmospheric Model Intercomparison Project
- AMIP: Atmospheric Model Intercomparison Project
