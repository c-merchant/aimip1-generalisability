"""
Paths, constants, and parameters for figure pipeline.
"""
import logging, warnings
from pathlib import Path

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", category=SmallSampleWarning)
warnings.filterwarnings('ignore', message='Mean of empty slice')

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path('/network/group/aopp/met_data/MET004_CMIP6/AIMIP-DATA')
AMIP_DIR = DATA_ROOT / 'AMIP-CMIP6'
AIMIP_DIR = DATA_ROOT / 'AIMIP-1'
ERA5_DIR = Path('/network/group/aopp/met_data/MET001_ERA5/data')
LAND_MASK_PATH = ERA5_DIR / 'lsm' / 'mon' / 'landseamask_1x1.nc'
ERA5_CACHE = PROJECT_ROOT / 'era5_classification.npz'
KOPPEN_TABLE = PROJECT_ROOT / 'code' / 'cluster' / 'cluster_assets' / 'koppen_table.csv'

FIGURE_DIR = PROJECT_ROOT / 'code' / 'figures' / 'pdfs'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_START = 1979
BASELINE_END = 2014
RESOLUTION = '1x1'
