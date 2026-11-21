"""
Unit testing for model tree cataloguing.
"""

from code.data.model_loader import load_model_tree, AIMIP_SCENARIOS, AMIP_SCENARIOS
import numpy as np
import pandas as pd
import xarray as xr
import pytest

TIMES = pd.date_range('1979-01-01', periods=24, freq='MS')  # 1979-1980
LAT, LON = np.array([-30.0, 0.0, 30.0]), np.array([0.0, 90.0, 180.0, 270.0])
UNITS = {'tas': 'K', 'pr': 'kg m-2 s-1', 'huss': 'kg kg-1'}
FILL = {'tas': 280.0, 'pr': 1e-5, 'huss': 5e-3}


def write_var(root, inst, model, scenario, ensemble, var, grid='gn', value=None):
    """Write one variable file into an Org/Model/scenario/ensemble/var tree."""
    var_dir = root / inst / model / scenario / ensemble / var
    var_dir.mkdir(parents=True, exist_ok=True)
    vals = np.full((len(TIMES), len(LAT), len(LON)), FILL[var] if value is None else value)
    da = xr.DataArray(vals, dims=['time', 'lat', 'lon'], coords={'time': TIMES, 'lat': LAT, 'lon': LON}, attrs={'units': UNITS[var]})
    path = var_dir / f'{var}_Amon_{model}_{scenario}_{ensemble}_{grid}_197901-198012.nc'
    xr.Dataset({var: da}).to_netcdf(path)
    return path


def write_dataset(root, inst, model, scenario, ensemble, variables=('tas', 'pr'), grid='gn'):
    for var in variables:
        write_var(root, inst, model, scenario, ensemble, var, grid=grid)


def test_load_model_tree_requires_tas_and_pr(tmp_path):
    """Datasets missing either tas or pr are dropped (as for the pr-less ArchesWeather models)."""
    write_dataset(tmp_path, 'Ai2', 'ACE2-ERA5', 'aimip', 'r1i1p1f1')
    write_dataset(tmp_path, 'ArchesWeather', 'ArchesWeatherGen', 'aimip', 'r1i1p1f1', variables=('tas',))
    write_dataset(tmp_path, 'DLESyM', 'DLESyM', 'aimip', 'r1i1p1f1', variables=('pr',))

    loaded = load_model_tree(tmp_path, AIMIP_SCENARIOS, max_workers=1)

    assert set(loaded) == {'Ai2_ACE2-ERA5_0K_r1i1p1f1'}, f"Only the tas+pr dataset should load, got {sorted(loaded)}."
    assert loaded['Ai2_ACE2-ERA5_0K_r1i1p1f1']['T'].shape == (2, 12, 3, 4)


def test_load_model_tree_scenario_dir_mapping(tmp_path):
    """Both warming spellings map to the same label and unmapped scenario dirs are skipped."""
    write_dataset(tmp_path, 'Google', 'NeuralGCM-HRD', 'aimip-p2k', 'r1i1p1f1')
    write_dataset(tmp_path, 'UMD-PARETO', 'MD-1p5', 'aimip-2k', 'r1i1p1f1')
    write_dataset(tmp_path, 'Google', 'NeuralGCM-HRD', 'aimip-piControl', 'r1i1p1f1')

    loaded = load_model_tree(tmp_path, AIMIP_SCENARIOS, max_workers=1)

    assert set(loaded) == {'Google_NeuralGCM-HRD_2K_r1i1p1f1', 'UMD-PARETO_MD-1p5_2K_r1i1p1f1'}, f"Got {sorted(loaded)}."


def test_load_model_tree_skips_unmapped_amip_scenario(tmp_path):
    """amip-future4K is not in AMIP_SCENARIOS and must not be passed through raw."""
    write_dataset(tmp_path, 'NCAR', 'CESM2', 'amip', 'r1i1p1f1')
    write_dataset(tmp_path, 'NCAR', 'CESM2', 'amip-future4K', 'r1i1p1f1')

    loaded = load_model_tree(tmp_path, AMIP_SCENARIOS, max_workers=1)

    assert set(loaded) == {'NCAR_CESM2_0K_r1i1p1f1'}, f"Got {sorted(loaded)}."


def test_load_model_tree_skip_models(tmp_path):
    write_dataset(tmp_path, 'Ai2', 'ACE2-ERA5', 'aimip', 'r1i1p1f1')
    write_dataset(tmp_path, 'Org', 'Excluded', 'aimip', 'r1i1p1f1')

    loaded = load_model_tree(tmp_path, AIMIP_SCENARIOS, skip_models={'Excluded'}, max_workers=1)

    assert set(loaded) == {'Ai2_ACE2-ERA5_0K_r1i1p1f1'}, f"Got {sorted(loaded)}."


def test_load_model_tree_filters(tmp_path):
    """institution/model/scenario/ensemble each narrow the catalog."""
    write_dataset(tmp_path, 'Ai2', 'ACE2-ERA5', 'aimip', 'r1i1p1f1')
    write_dataset(tmp_path, 'Ai2', 'ACE2-ERA5', 'aimip', 'r2i1p1f1')
    write_dataset(tmp_path, 'Ai2', 'ACE2-ERA5', 'aimip-p4k', 'r1i1p1f1')
    write_dataset(tmp_path, 'NVIDIA', 'cBottle-1-3', 'aimip', 'r1i1p1f1')

    assert set(load_model_tree(tmp_path, AIMIP_SCENARIOS, institution='NVIDIA', max_workers=1)) == {'NVIDIA_cBottle-1-3_0K_r1i1p1f1'}
    assert set(load_model_tree(tmp_path, AIMIP_SCENARIOS, model='cBottle-1-3', max_workers=1)) == {'NVIDIA_cBottle-1-3_0K_r1i1p1f1'}
    assert set(load_model_tree(tmp_path, AIMIP_SCENARIOS, scenario='4K', max_workers=1)) == {'Ai2_ACE2-ERA5_4K_r1i1p1f1'}
    assert set(load_model_tree(tmp_path, AIMIP_SCENARIOS, ensemble='r2i1p1f1', max_workers=1)) == {'Ai2_ACE2-ERA5_0K_r2i1p1f1'}


def test_load_model_tree_nonstandard_ensemble_dir(tmp_path):
    """ArchesWeatherGen warming uses a literal 'r*' ensemble dir, which must parse generically."""
    write_dataset(tmp_path, 'ArchesWeather', 'ArchesWeatherGen', 'aimip-p2k', 'r*')

    loaded = load_model_tree(tmp_path, AIMIP_SCENARIOS, max_workers=1)

    assert set(loaded) == {'ArchesWeather_ArchesWeatherGen_2K_r*'}, f"Got {sorted(loaded)}."


def test_load_model_tree_prefers_regridded_files(tmp_path):
    """With both gn and gr copies of a variable, only the gr files are kept."""
    write_var(tmp_path, 'Org', 'Model', 'aimip', 'r1i1p1f1', 'tas', grid='gn', value=280.0)
    write_var(tmp_path, 'Org', 'Model', 'aimip', 'r1i1p1f1', 'tas', grid='gr', value=300.0)
    write_var(tmp_path, 'Org', 'Model', 'aimip', 'r1i1p1f1', 'pr', grid='gn')

    loaded = load_model_tree(tmp_path, AIMIP_SCENARIOS, max_workers=1)

    T = loaded['Org_Model_0K_r1i1p1f1']['T']
    assert np.allclose(T, 300.0 - 273.15), f"Expected the gr copy (26.85 C), got {np.unique(T)}."


def test_load_model_tree_humidity_optional(tmp_path):
    """huss alongside tas/pr is loaded as Q; without it Q is all NaN."""
    write_dataset(tmp_path, 'Org', 'WithQ', 'aimip', 'r1i1p1f1', variables=('tas', 'pr', 'huss'))
    write_dataset(tmp_path, 'Org', 'NoQ', 'aimip', 'r1i1p1f1')

    loaded = load_model_tree(tmp_path, AIMIP_SCENARIOS, max_workers=1)

    assert np.allclose(loaded['Org_WithQ_0K_r1i1p1f1']['Q'], 5e-3), "huss should be loaded as Q."
    assert np.all(np.isnan(loaded['Org_NoQ_0K_r1i1p1f1']['Q'])), "Q should be NaN when no humidity input exists."


def test_load_model_tree_no_matching_data(tmp_path):
    write_dataset(tmp_path, 'NCAR', 'CESM2', 'amip', 'r1i1p1f1')
    with pytest.raises(ValueError, match="No AIMIP data found"):
        load_model_tree(tmp_path, AIMIP_SCENARIOS, label='AIMIP', max_workers=1)
