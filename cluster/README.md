# Cluster

Module for clustering surface fields into climate zones for tracking climate zone shifts caused by climate forcings. 

## Approach

**V1:** We have integrated the Köppen-Geiger implementation from [1] that takes (**non-normalised**) monthly temperature and precipitation climatologies and returns major and subclass classifications for each spatial location. If multiple years of data are given, the mode is taken and a confidence value is given. This clustering method has pre-defined clusters as defined in `cluster_assets/kippen_table.csv` by [1]. It is expected future methods will be data driven and require normalisation. 

```
import cluster


# Instantiate cluster class
cluster_class = cluster.Cluster('koppen_geiger') 


# Apply cluster 'rules' to AIMIP (or other)
clusters = Cluster.classify(temperature, precipitation)

...

```

## File structure

- cluster.py: Public API for clustering and classification workflows.
- cluster_utils/: Utility implementations (e.g., Köppen-Geiger logic).
- cluster_assets/: Static reference data (e.g., Köppen-Geiger table CSV).
- README.md: Module overview and usage notes.

Note: unit tests can be found in `../test/test_cluster/`

## References

[1] Beck, H. E. et al. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. Sci Data 10, 724 (2023).
