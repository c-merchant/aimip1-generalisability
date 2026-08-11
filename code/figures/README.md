# Figures

Module for plotting manuscript figures from surface fields and their classifications.

## Approach

We have on script per figure. Each `fig_*.py` reloads its data, reclassifies, and writes to a PDF, so they can be individually reproduced and/or debugged. We put the shared analysis and plotting helpers in `utils.py`; paths and parameters in `config.py`; relevant models for each figure in `models.py`.


```bash
# Plot one figure
uv run python -m code.figures.fig_baseline_differences

# Plot all figures
uv run python bash run_figs.sh > /dev/null 2>&1 &
```

## File structure

- config.py: Paths and parameters.
- models.py: Registry of AIMIP and AMIP models, their display labels, aesthetics, and available scenarios.
- utils.py: Shared helpers.
- make_era5_classification.py: Regenerates the cached ERA5 classification.
- fig_baseline_differences.py: Major-zone disagreement of each 0K model baseline against ERA5 (`baseline_differences_2col.pdf`).
- fig_classification_confidence.py: Baseline temporal classification confidence for ERA5, AIMIP, and named AMIP references (`classification_confidence.pdf`).
- fig_combined_forcing_cc.py: Per-zone T and P response across SST forcings with Clausius-Clapeyron humidity scaling (`combined_forcing_cc.pdf`).
- fig_migration_4K.py: Per-model zone migration maps at +4K with origin and destination summary charts (`migration_4K.pdf`).
- fig_agreement_4K_bias.py: Per-model +4K dT and fractional dP bias against the AMIP ensemble mean (`agreement_4K_bias.pdf`).
- fig_kgzone_distshift_4K.py: Per-zone temperature and precipitation distribution shifts at +4K (`kgzone_distshift_4K.pdf`).
- pdfs/: Generated outputs.