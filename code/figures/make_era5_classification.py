"""
Regenerate cached ERA5 Koppen classification (era5_classification.npz) from ERA5 baseline.
"""

import numpy as np
from code.cluster.cluster import Cluster
from code.figures.config import ERA5_CACHE, BASELINE_START, BASELINE_END
from code.figures.utils import load_era5

print('Loading ERA5 baseline.')
era5 = load_era5(BASELINE_START, BASELINE_END, land_only=True)

print('Classifying per year and taking the modal class.')
result = Cluster('koppen_geiger').classify(era5['T'], era5['P'])

np.savez(str(ERA5_CACHE),
         class_arr=result['Class'], major_arr=result['Major'],
         confidence_class=result['Confidence']['Class'], confidence_major=result['Confidence']['Major'],
         years=era5['years'])
print('Saved ' + str(ERA5_CACHE))
