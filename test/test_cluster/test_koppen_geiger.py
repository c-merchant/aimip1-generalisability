"""
Unit testing for the koppen_geiger functionality
"""

import cluster
import pandas as pd
import numpy as np



def test_koppen_geiger():
    # Load csv file containing the koppen geiger table
    koppen_table = pd.read_csv(cluster.__path__[0] + '/cluster_assets/koppen_table.csv')

    # Create dummy data for T and P
    np.random.seed(0)
    T = np.random.rand(12, 5, 5) * 30 - 10  # Random temperatures between -10 and 20 degrees Celsius
    P = np.random.rand(12, 5, 5) * 200  # Random precipitation between 0 and 200 mm/month

    # Call the koppen_geiger function
    climate_class, major_type = cluster.cluster_utils.koppen_geiger(T, P, koppen_table)

    