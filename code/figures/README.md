# Figures

Module for plotting manuscript figures from surface fields and their classifications.

## Approach

We have one script per figure. Each `fig_*.py` reloads its data, reclassifies it, and writes it to a PDF, so they can be individually reproduced and/or debugged. We put the shared analysis and plotting helpers in `utils.py`; paths and parameters in `config.py`; relevant models for each figure in `models.py`.

We classify each AIMIP and AMIP member before averaging. Forcing responses use each member's own baseline zones. We calculate migration fractions, temporal confidence, agreement, and percentage changes per member. Categorical maps use modal member zones.

```bash
# Plot one figure
uv run python -m code.figures.fig_baseline_differences

# Plot all figures
uv run bash run_figs.sh > /dev/null 2>&1 &
```

## File structure

- config.py: Paths and parameters.
- models.py: Registry of AIMIP and AMIP models, their display labels, aesthetics, and available scenarios.
- utils.py: Shared helpers.
- make_era5_classification.py: Regenerates the cached ERA5 classification.
- fig_baseline_differences.py: Major-zone disagreement of each 0K model baseline against ERA5 (`baseline_differences_2col.pdf`).
- fig_classification_confidence.py: Baseline temporal classification confidence for ERA5, AIMIP, and named AMIP references (`classification_confidence.pdf`).
- fig_combined_forcing_cc.py: Per-zone T and P response across SST forcings with Clausius-Clapeyron humidity scaling (`combined_forcing_cc.pdf`).
- fig_tropical_precip_outliers.py: Full AMIP range, outliers excluded, and interdecile tropical precipitation comparison. (`trop_p_outliers.pdf`)
- fig_migration_4K.py: Per-model zone migration maps at +4K with origin and destination summary charts (`migration_4K.pdf`).
- fig_agreement_4K_bias.py: Per-model +4K dT and fractional dP bias against the AMIP ensemble mean (`agreement_4K_bias.pdf`).
- fig_kgzone_distshift_4K.py: Per-zone temperature and precipitation distribution shifts at +4K (`kgzone_distshift_4K.pdf`).
- pdfs/: Generated outputs.
