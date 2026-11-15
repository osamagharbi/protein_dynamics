#!/bin/bash
# ===========================================================================
#  Held-out evaluation (interpolation + future prediction + copy baselines) of
#  ONE trained MAMBA-DSR run. No training. Submitted by submit_heldout.sh.
#  Resumable: finished frames are cached, so a resubmit continues where it stopped.
# ===========================================================================
#SBATCH --job-name=dsr_heldout
#SBATCH --account=acu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
set -uo pipefail
: "${CASE:?CASE not set}"
TAG="${TAG:-}"; SEED="${SEED:-0}"; RESOLUTION="${RESOLUTION:-128}"
METHOD="${METHOD:-mamba}"; MODE="${MODE:-extrapolate}"; NAME="${NAME:-heldout_${CASE}${TAG}}"
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
STATE_DIR="$REPO/jeanzay/train/state/$NAME"; mkdir -p "$STATE_DIR"
[ -e "$STATE_DIR/DONE" ] && { echo "[$CASE] held-out evaluation already complete"; exit 0; }

source "$REPO/jeanzay/train_env/env_train.sh"
cd "$REPO/MAMBA-DSR"
echo "[$(date +%F' '%T)] job $SLURM_JOB_ID on $(hostname): held-out eval CASE=$CASE METHOD=$METHOD MODE=$MODE TAG=$TAG"
EXTRA=(); [ "$METHOD" = mamba ] && EXTRA=(--tag="$TAG" --future-mode="$MODE")
timeout --signal=TERM "$(( ${WALL_MIN:-600} - 5 ))m" \
  python eval_heldout.py --only "$CASE" --method "$METHOD" --seed "$SEED" --resolution "$RESOLUTION" ${EXTRA[@]+"${EXTRA[@]}"}
RC=$?
case $RC in
  0)   date +%F' '%T > "$STATE_DIR/DONE"; echo "[$CASE] held-out COMPLETE" ;;
  124) echo "[$CASE] walltime reached -- the next chained job continues from the frame cache" ;;
  *)   echo "[$CASE] FAILED rc=$RC" ;;
esac
exit 0
