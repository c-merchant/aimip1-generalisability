#!/bin/bash
set -u
cd /home/m/merchantc/aimip1-generalisability
for f in fig_baseline_differences fig_combined_forcing_cc fig_migration_4K fig_agreement_4K_bias fig_kgzone_distshift_4K fig_classification_confidence; do
  echo "=== START $f $(date) ===" >> run_figs.log
  python -m code.figures.$f >> run_figs.log 2>&1
  echo "=== END $f rc=$? $(date) ===" >> run_figs.log
done
echo "ALL DONE $(date)" >> run_figs.log
