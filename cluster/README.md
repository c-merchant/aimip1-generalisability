# Cluster

Module for clustering surface fields into climate zones for tracking climate zone shifts caused by climate forcings. 

## Approach

We cluster based on ERA5 train data and use clusters to clasify data from AIMIP models and ERA5 test data. 
- We assume data is `xr.Dataset` and contains **normalised** surface fields
- We will implement several different clustering metrics, capture different approaches 
- We will output masks for each cluster as well as plots for analysis 

```
# Instantiate cluster class

cluster_class = Cluster(...)

# Get cluster 'rules'
Cluster.calculate_clusters(era5_train_xr_dataset)

# Apply cluster 'rules' to AIMIP (or other)
cluster_mask = Cluster.apply_clustering(AIMIP_xr_dataset)

...

```