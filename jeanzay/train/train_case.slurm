#!/bin/bash
# ===========================================================================
#  One training link for ONE dataset (CASE) and ONE method (METHOD).
#  Submitted by jeanzay/train/submit.sh as a chain of jobs: every link resumes
#  from the last checkpoint, skips finished stages, and exits at once when the
#  case is complete. Do not sbatch this directly unless you set the env vars.
#
#   METHOD=mamba : MAMBA-DSR  Stage A (decoder + per-frame latents, resumable)
#                             -> Stage B (bidirectional Mamba t->latent)
#                             -> evaluation (IoU / Chamfer / NC, res 128, normalized)
#   METHOD=dsr   : official DSR baseline (runner_igr_t.py --is_continue) + same evaluator
#   METHOD=flow  : MAMBA-Flow, ONE model for all datasets (CASE=all; FLOW_CASES = optional
#                  '+'-joined subset): cache -> Stage 1 shape VAE -> encode -> Stage 2 flow
#                  matching -> training-frame evaluation (run_flow.py, every stage resumable)
# ===========================================================================
#SBATCH --job-name=dsr_train
#SBATCH --account=acu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
set -uo pipefail

: "${CASE:?CASE not set}"
METHOD="${METHOD:-mamba}"
SEED="${SEED:-0}"
TAG="${TAG:-}"
STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-4000}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-5000}"
RESOLUTION="${RESOLUTION:-128}"
RESERVE_MIN="${RESERVE_MIN:-90}"      # walltime kept for data loading, Stage B and evaluation
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
STATE_DIR="$REPO/jeanzay/train/state/${METHOD}_${CASE}${TAG}"
mkdir -p "$STATE_DIR"

if [ -e "$STATE_DIR/DONE" ];   then echo "[$CASE] already complete -- nothing to do"; exit 0; fi
if [ -e "$STATE_DIR/FAILED" ]; then echo "[$CASE] an earlier link FAILED (see $STATE_DIR/FAILED) -- skipping"; exit 0; fi

source "$REPO/jeanzay/train_env/env_train.sh"
cd "$REPO/MAMBA-DSR"
echo "[$(date +%F' '%T)] job $SLURM_JOB_ID on $(hostname): METHOD=$METHOD CASE=$CASE SEED=$SEED TAG=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Minutes left in this job (squeue %L: [D-]HH:MM:SS or MM:SS).
left_min() {
  python - "$(squeue -h -j "$SLURM_JOB_ID" -o %L)" <<'PY'
import sys
s = sys.argv[1].strip(); d = 0
if "-" in s: d, s = s.split("-"); d = int(d)
p = [int(x) for x in s.split(":")]
while len(p) < 3: p.insert(0, 0)
print(d * 1440 + p[0] * 60 + p[1])
PY
}
LEFT=$(left_min 2>/dev/null || true)
[[ "$LEFT" =~ ^[0-9]+$ ]] || LEFT="${WALL_MIN:-1200}"   # fallback: full walltime from submit.sh
echo "[slurm] ${LEFT} min left in this job"

if [ "$METHOD" = mamba ]; then
  BUDGET=$(( LEFT - RESERVE_MIN )); [ "$BUDGET" -lt 10 ] && BUDGET=10
  # Hard stop 5 min before the walltime so a link never dies mid-checkpoint.
  timeout --signal=TERM "$(( LEFT - 5 ))m" \
    python run_multicase.py --gpu 0 --only "$CASE" --resume --seed "$SEED" --tag "$TAG" \
      --stage-a-epochs "$STAGE_A_EPOCHS" --stage-b-epochs "$STAGE_B_EPOCHS" \
      --resolution "$RESOLUTION" --stop-after-minutes "$BUDGET"
  RC=$?
elif [ "$METHOD" = dsr ]; then
  # DSR checkpoints every 200 epochs; a link stopped by the timeout resumes from there.
  timeout --signal=TERM "$(( LEFT - 5 ))m" \
    python run_dsr_paper.py --gpu 0 --only "$CASE" --resume --seed "$SEED" --tag "$TAG" \
      --epochs "$STAGE_A_EPOCHS" --resolution "$RESOLUTION"
  RC=$?
elif [ "$METHOD" = flow ]; then
  BUDGET=$(( LEFT - RESERVE_MIN )); [ "$BUDGET" -lt 10 ] && BUDGET=10
  FLOW_ARGS=(--seed "$SEED" --resolution "$RESOLUTION" --stop-after-minutes "$BUDGET")
  if [[ "$TAG" == *_smoke ]]; then FLOW_ARGS+=(--smoke --tag "${TAG%_smoke}"); else FLOW_ARGS+=(--tag "$TAG"); fi
  [ -n "${FLOW_CASES:-}" ] && FLOW_ARGS+=(--only ${FLOW_CASES//+/ })
  # shellcheck disable=SC2206
  [ -n "${FLOW_SET:-}" ] && for kv in ${FLOW_SET//+/ }; do FLOW_ARGS+=(--set "$kv"); done
  timeout --signal=TERM "$(( LEFT - 5 ))m" python run_flow.py "${FLOW_ARGS[@]}"
  RC=$?
else
  echo "unknown METHOD=$METHOD (mamba|dsr|flow)"; exit 1
fi

case $RC in
  0)       date +%F' '%T > "$STATE_DIR/DONE"; echo "[$CASE] COMPLETE" ;;
  3|124)   echo "[$CASE] not finished in this link (rc=$RC) -- the next chained job continues" ;;
  *)       echo "job $SLURM_JOB_ID rc=$RC $(date +%F' '%T)" > "$STATE_DIR/FAILED"
           echo "[$CASE] FAILED rc=$RC -- later links will skip; fix, then: rm $STATE_DIR/FAILED" ;;
esac
exit 0
