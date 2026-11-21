"""
Unit testing for the koppen_geiger functionality
"""

from code import cluster
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


def test_koppen_geiger_known_profiles():
    """Pin the classification arithmetic: known T/P profiles must map to their known major class.

    Major ids: 1=A tropical, 2=B arid, 3=C temperate, 4=D cold, 5=E polar.
    """
    koppen_table_path = cluster.__path__[0] + '/cluster_assets/koppen_table.csv'
    gk = cluster.KoppenGeiger(koppen_table_path)

    def major(t_months, p_months):
        T = np.array(t_months, dtype=float).reshape(12, 1, 1)
        P = np.array(p_months, dtype=float).reshape(12, 1, 1)
        return gk.classify(T, P)['Major'][0, 0]

    # A (tropical): every month warm (Tmin >= 18) and wet enough to escape arid.
    assert major([26] * 12, [200] * 12) == 1, "Warm wet all year should be Tropical (A)"

    # B (arid): warm but very dry (MAP well below the aridity threshold).
    assert major([25] * 12, [5] * 12) == 2, "Hot and very dry should be Arid (B)"

    # C (temperate): hottest month > 10, coldest month between 0 and 18, not arid.
    temperate_T = [4, 5, 8, 12, 16, 20, 22, 21, 18, 13, 8, 5]
    assert major(temperate_T, [80] * 12) == 3, "Mild winter, warm summer, wet should be Temperate (C)"

    # D (cold): hottest month > 10, coldest month <= 0, not arid.
    cold_T = [-10, -8, -2, 5, 12, 18, 20, 18, 12, 4, -3, -8]
    assert major(cold_T, [80] * 12) == 4, "Freezing winter, warm summer, wet should be Cold (D)"

    # E (polar): hottest month <= 10.
    assert major([-20, -18, -15, -8, -2, 3, 6, 5, 0, -8, -14, -18], [30] * 12) == 5, "No warm month should be Polar (E)"


def test_koppen_geiger_arid_precedence():
    """B (arid) must override C/D/E: a cold-winter dry profile that looks temperate/cold by temperature is still Arid."""
    koppen_table_path = cluster.__path__[0] + '/cluster_assets/koppen_table.csv'
    gk = cluster.KoppenGeiger(koppen_table_path)
    # Temperate-looking temperatures but almost no precipitation -> must be B, not C.
    T = np.array([4, 5, 8, 12, 16, 20, 22, 21, 18, 13, 8, 5], dtype=float).reshape(12, 1, 1)
    P = np.array([2] * 12, dtype=float).reshape(12, 1, 1)
    assert gk.classify(T, P)['Major'][0, 0] == 2, "Arid must take precedence over Temperate"


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