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

def test_koppen_geiger_class():
    koppen_table_path = cluster.__path__[0] + '/cluster_assets/koppen_table.csv'
    gk_cluster = cluster.KoppenGeiger(koppen_table_path)

    # Create dummy data for T and P
    np.random.seed(0)
    T = np.random.rand(12, 5, 5) * 30 - 10  # Random temperatures between -10 and 20 degrees Celsius
    P = np.random.rand(12, 5, 5) * 200  # Random precipitation between 0 and 200 mm/month

    output = gk_cluster.classify(T, P)
    assert 'Class' in output, "Output should contain 'Class' key"
    assert 'Major' in output, "Output should contain 'Major' key"
    assert output['Class'].shape == (5, 5), "Class output should have shape (5, 5)"
    assert output['Major'].shape == (5, 5), "Major output should have shape (5, 5)"
    assert np.all(np.isfinite(output['Class'])), "Class output should not contain NaN values"
    assert np.all(np.isfinite(output['Major'])), "Major output should not contain NaN values"

def test_koppen_geiger_invalid_input():
    koppen_table_path = cluster.__path__[0] + '/cluster_assets/koppen_table.csv'
    gk_cluster = cluster.KoppenGeiger(koppen_table_path)

    # Create invalid dummy data for T and P
    T_invalid = np.random.rand(5, 5) * 30 - 10  # Invalid shape
    P_invalid = np.random.rand(12, 5) * 200  # Invalid shape

    try:
        gk_cluster.classify(T_invalid, P_invalid)
        assert False, "Expected ValueError for invalid input shapes"
    except ValueError as e:
        assert str(e) == "T and P should be 3D arrays with shape (12, spatial_dim_1, spatial_dim_2)", f"Unexpected error message: {str(e)}"


def test_koppen_geiger_multiyear():
    koppen_table_path = cluster.__path__[0] + '/cluster_assets/koppen_table.csv'
    gk_cluster = cluster.KoppenGeiger(koppen_table_path)

    # Create dummy data for T and P with multiple years
    np.random.seed(0)
    T_multiyear = np.random.rand(3, 12, 5, 5) * 30 - 10  # Random temperatures for 3 years
    P_multiyear = np.random.rand(3, 12, 5, 5) * 200  # Random precipitation for 3 years

    output = gk_cluster.classify(T_multiyear, P_multiyear)
    assert 'Class' in output, "Output should contain 'Class' key"
    assert 'Major' in output, "Output should contain 'Major' key"
    assert 'Confidence' in output, "Output should contain 'Confidence' key"
    assert output['Class'].shape == (5, 5), "Class output should have shape (5, 5)"
    assert output['Major'].shape == (5, 5), "Major output should have shape (5, 5)"
    assert output['Confidence']['Class'].shape == (5, 5), "Confidence Class output should have shape (5, 5)"
    assert output['Confidence']['Major'].shape == (5, 5), "Confidence Major output should have shape (5, 5)"
    assert np.all(np.isfinite(output['Class'])), "Class output should not contain NaN values"
    assert np.all(np.isfinite(output['Major'])), "Major output should not contain NaN values"
    assert np.all((output['Confidence']['Class'] >= 0) & (output['Confidence']['Class'] <= 1)), "Confidence Class output should be between 0 and 1"
    assert np.all((output['Confidence']['Major'] >= 0) & (output['Confidence']['Major'] <= 1)), "Confidence Major output should be between 0 and 1"