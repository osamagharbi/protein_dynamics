#!/bin/bash
#SBATCH --job-name=DSR_CONV
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#
# Array body: task i converts row i of $TASKS (one frame chunk of one dataset)
# with $SLURM_CPUS_PER_TASK parallel PyMOL processes. Already-written frames
# are skipped, so failed/timed-out tasks can simply be resubmitted.
# Resources/account/partition are set by run_data_prep.sh (PROC_TARGET).
# Required env: PREP_DIR RAW_ROOT PROC_ROOT TASKS (+ ATLAS_* for atlas)
set -euo pipefail
module purge
source "$PREP_DIR/env.sh"
echo "node=$(hostname) array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
python -u "$PREP_DIR/prep.py" convert --tasks "$TASKS" \
    --task-id "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK"
