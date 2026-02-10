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
    output = cluster.cluster_utils.koppen_geiger(T, P, koppen_table)

    assert 'Class' in output, "Output should contain 'Class' key"
    assert 'Major' in output, "Output should contain 'Major' key"
    assert output['Class'].shape == (5, 5), "Class output should have shape (5, 5)"
    assert output['Major'].shape == (5, 5), "Major output should have shape (5, 5)"
    assert np.all(np.isfinite(output['Class'])), "Class output should not contain NaN values"
    assert np.all(np.isfinite(output['Major'])), "Major output should not contain NaN values"