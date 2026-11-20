#!/bin/bash
# ===========================================================================
#  submit.sh -- queue MAMBA-DSR (and optionally DSR-baseline) training on Jean-Zay.
#  Run from anywhere on a LOGIN node, after jeanzay/train_env/setup_train_env.sh.
#
#  Each (method, case) gets a CHAIN of NCHAIN jobs linked with afterany: every
#  link resumes from the last checkpoint, so runs longer than the 20 h t3
#  walltime continue automatically, and finished cases cost ~1 min per spare link.
#
#     ./jeanzay/train/submit.sh                         # MAMBA-DSR on every ready dataset
#     CASES="adk_dims adk_froda" ./jeanzay/train/submit.sh
#     METHOD=dsr CASES=adk_equilibrium ./jeanzay/train/submit.sh   # official DSR baseline
#     SMOKE=1 CASES=adk_dims ./jeanzay/train/submit.sh  # 30-min qos_gpu-dev plumbing test
#     SEED=1 ./jeanzay/train/submit.sh                  # second seed (run names get _s1)
#     METHOD=dsr SEED=1 CASES="adk_dims" ./jeanzay/train/submit.sh  # DSR baseline, second seed
#     DRY=1 ./jeanzay/train/submit.sh                   # print the sbatch commands only
#     METHOD=flow ./jeanzay/train/submit.sh             # MAMBA-Flow: ONE chain, one model for all datasets
#     SMOKE=1 METHOD=flow ./jeanzay/train/submit.sh     # MAMBA-Flow 30-min plumbing test (tiny model)
#     METHOD=flow FLOW_SET="shape_steps=60000+dyn_steps=20000" ./jeanzay/train/submit.sh   # config overrides
#
#  Env: CASES, METHOD (mamba|dsr|flow), SEED, NCHAIN (default 4), CONSTRAINT (e.g. v100-32g),
#       STAGE_A_EPOCHS (4000), STAGE_B_EPOCHS (5000), RESOLUTION (128), QOS, TIME.
# ===========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
METHOD="${METHOD:-mamba}"
SEED="${SEED:-0}"
NCHAIN="${NCHAIN:-4}"
QOS="${QOS:-qos_gpu-t3}"
TIME="${TIME:-20:00:00}"
TAG=""; [ "$SEED" != 0 ] && TAG="_s$SEED"
wall_min() { IFS=: read -r h m _ <<<"$1"; echo $(( 10#$h * 60 + 10#$m )); }

if [ "${SMOKE:-0}" = 1 ]; then
  QOS=qos_gpu-dev; TIME=00:30:00; NCHAIN=1; TAG="${TAG}_smoke"
  export STAGE_A_EPOCHS=20 STAGE_B_EPOCHS=200 RESERVE_MIN=12
fi

[ -f "$REPO/jeanzay/train_env/env_train.sh" ] || {
  echo "missing jeanzay/train_env/env_train.sh -> run jeanzay/train_env/setup_train_env.sh first"; exit 1; }
source "$REPO/jeanzay/train_env/env_train.sh" >/dev/null 2>&1

if [ -z "${CASES:-}" ]; then   # every complete dataset prepared by jeanzay/data_prep
  CASES=$(cd "$REPO/MAMBA-DSR" && python run_multicase.py --list | awk '{print $1}' \
          | while read -r c; do [ -f "$REPO/datasets/DSR_processed/$c/meta.json" ] && echo "$c"; done)
fi
[ -n "$CASES" ] || { echo "no prepared datasets found under $REPO/datasets/DSR_processed"; exit 1; }

echo "[check] validating data + building train views (dry run)"
( cd "$REPO/MAMBA-DSR" && if [ "$METHOD" = dsr ]; then
    python run_dsr_paper.py --only $CASES --tag "$TAG" --dry-run
  elif [ "$METHOD" = flow ]; then
    python run_flow.py --check --only $CASES --tag "$TAG"
  else
    python run_multicase.py --only $CASES --dry-run
  fi )

if [ "$METHOD" = flow ]; then   # one model trained on all cases -> a single chain "flow_all<TAG>"
  FLOW_CASES=$(echo $CASES | tr ' ' '+'); CASES=all
  [ "${SMOKE:-0}" = 1 ] && export RESERVE_MIN=8
fi

for CASE in $CASES; do
  NAME="${METHOD}_${CASE}${TAG}"
  LOGDIR="$REPO/jeanzay/train/logs/$NAME"; mkdir -p "$LOGDIR"
  rm -f "$REPO/jeanzay/train/state/$NAME/FAILED"     # a fresh submit retries failed cases
  DEP=""; IDS=()
  for ((i = 1; i <= NCHAIN; i++)); do
    CMD=(sbatch --parsable --job-name="$NAME" --qos="$QOS" --time="$TIME"
         ${CONSTRAINT:+--constraint=$CONSTRAINT} $DEP
         --output="$LOGDIR/%j.out"
         --export=ALL,CASE="$CASE",METHOD="$METHOD",SEED="$SEED",TAG="$TAG",REPO="$REPO",WALL_MIN="$(wall_min "$TIME")",FLOW_CASES="${FLOW_CASES:-}",FLOW_SET="${FLOW_SET:-}"
         "$HERE/train_case.slurm")
    if [ "${DRY:-0}" = 1 ]; then echo "${CMD[*]}"; JID="DRY$i"; else JID=$("${CMD[@]}"); fi
    IDS+=("$JID"); DEP="--dependency=afterany:$JID"
  done
  echo "[submit] $NAME: jobs ${IDS[*]}  logs -> jeanzay/train/logs/$NAME/"
done
echo "monitor:  squeue -u \$USER   |   $HERE/status.sh"
