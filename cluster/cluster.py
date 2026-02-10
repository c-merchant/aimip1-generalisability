"""
Script containing a class for clustering climate data. 

Author: [BSC]

Version history:

    v1: 
        - Koppen-Geiger classification functionality implemented in the koppen_geiger function.


"""

import logging
import cluster
import numpy as np

class Cluster():
    def __init__(self, cluster_method: str):
        """
        Initializes the Cluster class. 

        Args:
            cluster_method (str): The clustering method to be used. 

        """

        logging.info(f"Initializing Cluster class with method: {cluster_method}")
        
        self.cluster_method_name = cluster_method

        self.cluster_method = self._get_cluster_method(cluster_method)

    
    def classify(self, T: np.ndarray, P: np.ndarray):
        """
        Classifies the input temperature and precipitation data using the specified clustering method. 

        Args:
            T (numpy.ndarray): monthly temperature climatology (degrees Celsius), shape (years, 12, spatial_dim_1, spatial_dim_2) or (12, spatial_dim_1, spatial_dim_2)
            P (numpy.ndarray): monthly precipitation climatology (mm/month), shape (years, 12, spatial_dim_1, spatial_dim_2) or (12, spatial_dim_1, spatial_dim_2)
    
        Returns:
            dict: A dictionary containing the classification results, which may include keys such as 'Class', 'Major', and 'Confidence' depending on the clustering method used.
            shapes of outputs are (spatial_dim_1, spatial_dim_2) for 'Class' and 'Major', and (spatial_dim_1, spatial_dim_2) for confidence values if applicable.
        
        """
        return self.cluster_method.classify(T, P)


    def _get_cluster_method(self, cluster_method_name: str):
        """
        Retrieves the clustering method based on the provided name. 

        Args:
            cluster_method_name (str): The name of the clustering method to retrieve.

        Returns:
            class: The clustering method class corresponding to the provided name.

        Raises:
            ValueError: If the provided cluster method name is not recognized.
        """

        if cluster_method_name == 'koppen_geiger':
            return cluster.KoppenGeiger()
        else:
            raise ValueError(f"Cluster method '{cluster_method_name}' not recognized. Please choose a valid cluster method.")