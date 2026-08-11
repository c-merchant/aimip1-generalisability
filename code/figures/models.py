"""
Model registry for figure pipelines.
"""
from dataclasses import dataclass
from typing import Literal

Scenario = Literal['0K', '2K', '4K', '-4K']

@dataclass(frozen=True)
class ModelPanel:
    """Per model loader key, display label, aesthetics, and available scenarios."""
    key: str
    display: str
    color: str=''
    marker: str=''
    scenarios: tuple[Scenario, ...]=('0K', '2K', '4K')

    def has_scenario(self, scenario):
        return scenario in self.scenarios

    def scenario_key(self, scenario):
        """Loader dict key for one scenario."""
        return f'{self.key}_{scenario}'

# AIMIP models
AIMIP_MODELS = [
    ModelPanel('Ai2_ACE2-ERA5', 'ACE2.1-ERA5', '#e41a1c', 'o'),
    ModelPanel('NVIDIA_cBottle-1-3', 'cBottle', '#377eb8', 's'),
    ModelPanel('DLESyM_DLESyM', 'DLESyM', '#4daf4a', 'D'),
    ModelPanel('UMD-PARETO_MD-1p5', 'MD-1p5', '#984ea3', '^'),
    ModelPanel('Google_NeuralGCM-HRD', 'NeuralGCM-HRD', '#ff7f00', 'v'),
]

# AMIP models
AMIP_MODELS = [
    ModelPanel('AS-RCEC_TaiESM1', 'TaiESM1', scenarios=('0K', '4K', '-4K')),
    ModelPanel('BCC_BCC-CSM2-MR', 'BCC-CSM2-MR', scenarios=('0K', '4K', '-4K')),
    ModelPanel('CCCma_CanESM5', 'CanESM5', scenarios=('0K', '4K', '-4K')),
    ModelPanel('CNRM-CERFACS_CNRM-CM6-1', 'CNRM-CM6-1', scenarios=('0K', '4K', '-4K')),
    ModelPanel('CSIRO-ARCCSS_ACCESS-CM2', 'ACCESS-CM2', scenarios=('0K',)),
    ModelPanel('CSIRO-ARCCSS_ACCESS-ESM1-5', 'ACCESS-ESM1-5', scenarios=('0K',)),
    ModelPanel('E3SM-project_E3SM-1-0', 'E3SM-1-0', scenarios=('0K', '4K')),
    ModelPanel('EC-Earth-Consortium_EC-Earth3-CC', 'EC-Earth3-CC', scenarios=('0K',)),
    ModelPanel('EC-Earth-Consortium_EC-Earth3', 'EC-Earth3', scenarios=('0K',)),
    ModelPanel('FIO_FIO-ESM-2-0', 'FIO-ESM-2-0', scenarios=('0K',)),
    ModelPanel('IPSL_IPSL-CM6A-LR', 'IPSL-CM6A-LR', scenarios=('0K', '4K', '-4K')),
    ModelPanel('IPSL_IPSL-CM6A-MR1', 'IPSL-CM6A-MR1', scenarios=('0K',)),
    ModelPanel('MIROC_MIROC6', 'MIROC6', scenarios=('0K', '4K')),
    ModelPanel('MOHC_HadGEM3-GC31-LL', 'HadGEM3-GC31-LL', scenarios=('0K', '4K', '-4K')),
    ModelPanel('MRI_MRI-ESM2-0', 'MRI-ESM2-0', scenarios=('0K', '4K', '-4K')),
    ModelPanel('NASA-GISS_GISS-E2-1-G', 'GISS-E2-1-G', scenarios=('0K', '4K')),
    ModelPanel('NASA-GISS_GISS-E2-2-G', 'GISS-E2-2-G', scenarios=('0K',)),
    ModelPanel('NCAR_CESM2', 'CESM2', scenarios=('0K', '4K', '-4K')),
    ModelPanel('NCC_NorCPM1', 'NorCPM1', scenarios=('0K',)),
    ModelPanel('NCC_NorESM2-LM', 'NorESM2-LM', scenarios=('0K', '4K')),
    ModelPanel('NOAA-GFDL_GFDL-CM4', 'GFDL-CM4', scenarios=('0K', '4K', '-4K')),
    ModelPanel('NUIST_NESM3', 'NESM3', scenarios=('0K',)),
]

def amip_with_scenario(scenario):
    """Fixed sub-list of AMIP models providing a given scenario (preserving order)."""
    return [m for m in AMIP_MODELS if m.has_scenario(scenario)]
