## How Do AI Climate Models Respond to Warming Across Climate Zones?

## Setup

We use [`uv`] for environment management (Python >= 3.11; dependencies are in `uv.lock`).

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Create `uv` environment

```bash
cd path/to/aimip1-generalisability
uv sync
uv sync --extra test
```

### Plotting figures

```bash
uv run python -m pytest
uv run python -m code.figures.fig_baseline_differences
```

### Repo Organization
```
code/           
├── data/                   Loader functions for AIMIP, ERA5, and AMIP
├── cluster/                Koppen-Geiger climate classification
└── figures/                Figure pipeline
test/                       Pytests
paper/                      Manuscript
era5_classification.npz     Cached ERA5 baseline classification
run_figs.sh                 Runs figure script sequentially
```

### Necessary Data

[`code/figures/config.py`](code/figures/config.py) configures associated paths. We require data from:

**ERA5** We need monthly `tas`, `pr` (and land-sea mask) on a 1x1 grid, which are available from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

```
<ERA5_DIR>/
├── tas/mon/*.nc
├── pr/mon/*.nc
└── lsm/mon/landseamask_1x1.nc
```

***AIMIP Phase 1*** This is the AI-model output, which can be downloaded from DKRZ's S3 `ai-mip` endpoint (`https://s3.eu-dkrz-1.dkrz.cloud`).

```bash
aws s3 sync --no-sign-request --endpoint-url https://s3.eu-dkrz-1.dkrz.cloud \
    s3://ai-mip/<Org>/<Model>/<AIMIP_DIR>/<Org>/<Model>/
```

The keys follow the structure of `Org/Model/scenario/ensemble/Amon/var/grid[/version]/file.nc`. We drop `Amon/grid/version` when staging for the loaders below.

We use data from the `aimip` (+0K baseline), `aimip-p2k` (+2K), and `aimip-p4k` (+4K) warming scenarios. MD-1p5 spells them `aimip-2k` and `aimip-4k`, but both conventions are accepted. We require `tas` and `pr` data, as well as humidity data where relevant, so models without `pr` (ArchesWeather/ArchesWeatherGen) are excluded from classification.

***AMIP (CMIP6)*** This is the physics-based model reference. The key structure is the same `Org/Model/scenario/ensemble/Amon/var`. We use monthly `tas` and `pr` data from the `amip` (+0K), `amip-m4k` (-4K), and `amip-p4k` (+4K) scenarios, which are available from [ESGF](https://metagrid.esgf-west.org/search/cmip6/) nodes based on location.

## Reproducing Figures

Each figure generation script in `code/figures/fig_*.py` reloads its data, reclassifies per year, and writes to `code/figures/pdfs/`. 

### Option 1: Generate a single figure.

```bash
uv run python -m code.figures.fig_baseline_differences
```

### Option 2: Generate all figures.

Run the full pipeline detached, and log progress to `run_figs.log`:

```bash
uv run python bash run_figs.sh > run_figs.log 2>&1 &
```

Scripts run sequentially.
