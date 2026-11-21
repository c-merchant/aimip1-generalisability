"""
Unit testing clustering class
"""

from code import cluster
import numpy as np


def test_cluster_initialization():
    cluster_method = 'koppen_geiger'
    cluster_instance = cluster.Cluster(cluster_method)
    assert cluster_instance.cluster_method_name == cluster_method, "Cluster method name should be set correctly"
    assert isinstance(cluster_instance.cluster_method, cluster.KoppenGeiger), "Cluster method should be an instance of KoppenGeiger"

def test_cluster_classify():
    cluster_method = 'koppen_geiger'
    cluster_instance = cluster.Cluster(cluster_method)

    # Create dummy data for T and P
    np.random.seed(0)
    T = np.random.rand(12, 5, 5) * 30 - 10  # Random temperatures between -10 and 20 degrees Celsius
    P = np.random.rand(12, 5, 5) * 200  # Random precipitation between 0 and 200 mm/month

    output = cluster_instance.classify(T, P)
    assert isinstance(output, dict), "Output should be a dictionary"
    assert 'Class' in output, "Output should contain 'Class' key"
    assert 'Major' in output, "Output should contain 'Major' key"
    assert output['Class'].shape == (5, 5), "Class output should have shape (5, 5)"
    assert output['Major'].shape == (5, 5), "Major output should have shape (5, 5)"
    assert np.all(np.isfinite(output['Class'])), "Class output should not contain NaN values"
    assert np.all(np.isfinite(output['Major'])), "Major output should not contain NaN values"