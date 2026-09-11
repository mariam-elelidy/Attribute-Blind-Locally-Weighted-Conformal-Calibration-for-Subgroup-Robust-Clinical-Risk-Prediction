#!/usr/bin/env bash
# Reproduces every result and figure in this repository end-to-end.
set -e
python analysis/00_make_splits.py
python analysis/01_conformal_baselines.py
python analysis/02_local_conformal_method.py
python analysis/03_shift_experiment.py
python analysis/04_abstention_sweep.py
python analysis/05_generate_figures.py
echo "Done. See results/tables/ and figures/."
