#!/bin/bash
#SBATCH --job-name=DSR_FIN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#
# Validate every surface of $DATASETS, write meta.json (provenance, QC, DSR
# normalisation) and the symlinked paper-protocol training view.
# Exit code 1 = at least one dataset incomplete (see the log).
# Required env: PREP_DIR RAW_ROOT PROC_ROOT DATASETS
set -euo pipefail
module purge
source "$PREP_DIR/env.sh"
python -u "$PREP_DIR/prep.py" finalize --datasets "$DATASETS" --workers "$SLURM_CPUS_PER_TASK"
python -u "$PREP_DIR/prep.py" status
