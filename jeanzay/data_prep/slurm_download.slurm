#!/bin/bash
#SBATCH --job-name=DSR_DL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#
# Download ONE raw source, then index its datasets (frame counts, selection,
# XTC offset cache). Needs internet -> submitted on the prepost partition by
# run_data_prep.sh, which also sets account/partition/output paths.
# Required env: PREP_DIR RAW_ROOT PROC_ROOT SOURCE DATASETS (+ ATLAS_* for atlas)
set -euo pipefail
module purge
source "$PREP_DIR/env.sh"
echo "node=$(hostname) job=${SLURM_JOB_ID} source=$SOURCE datasets=$DATASETS"
python -u "$PREP_DIR/prep.py" download --source "$SOURCE" --datasets "$DATASETS"
