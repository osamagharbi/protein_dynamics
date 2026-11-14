#!/bin/bash
# ===========================================================================
#  submit_heldout.sh -- held-out evaluation of trained MAMBA-DSR runs (no training):
#    interpolation (midpoints between training frames), future prediction
#    (next 1-10 / 11-20 frames), and copy-nearest / copy-last reference baselines.
#
#     ./jeanzay/train/submit_heldout.sh                     # every finished seed-0 run
#     CASES="adk_dims ifabp_water" ./jeanzay/train/submit_heldout.sh
#     SEED=1 ./jeanzay/train/submit_heldout.sh              # runs trained with SEED=1 (_s1)
#     DRY=1 ./jeanzay/train/submit_heldout.sh               # print the sbatch commands only
#  Results:  cd MAMBA-DSR && python eval_heldout.py --report
# ===========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SEED="${SEED:-0}"; NCHAIN="${NCHAIN:-2}"; QOS="${QOS:-qos_gpu-t3}"; TIME="${TIME:-10:00:00}"
TAG=""; [ "$SEED" != 0 ] && TAG="_s$SEED"
wall_min() { IFS=: read -r h m _ <<<"$1"; echo $(( 10#$h * 60 + 10#$m )); }

if [ -z "${CASES:-}" ]; then   # every run whose training finished (DONE marker)
  CASES=$(for d in "$REPO"/jeanzay/train/state/mamba_*"$TAG"; do
            [ -e "$d/DONE" ] || continue
            n=$(basename "$d"); n=${n#mamba_}
            [[ "$n" == *_smoke ]] && continue
            if [ -n "$TAG" ]; then [[ "$n" == *"$TAG" ]] && echo "${n%$TAG}"
            else [[ "$n" =~ _s[0-9]+$ ]] || echo "$n"; fi
          done)
fi
[ -n "$CASES" ] || { echo "no finished training runs found"; exit 1; }

for CASE in $CASES; do
  NAME="heldout_${CASE}${TAG}"
  LOGDIR="$REPO/jeanzay/train/logs/$NAME"; mkdir -p "$LOGDIR"
  DEP=""; IDS=()
  for ((i = 1; i <= NCHAIN; i++)); do
    CMD=(sbatch --parsable --job-name="$NAME" --qos="$QOS" --time="$TIME"
         ${CONSTRAINT:+--constraint=$CONSTRAINT} $DEP --output="$LOGDIR/%j.out"
         --export=ALL,CASE="$CASE",SEED="$SEED",TAG="$TAG",REPO="$REPO",WALL_MIN="$(wall_min "$TIME")"
         "$HERE/heldout.slurm")
    if [ "${DRY:-0}" = 1 ]; then echo "${CMD[*]}"; JID="DRY$i"; else JID=$("${CMD[@]}"); fi
    IDS+=("$JID"); DEP="--dependency=afterany:$JID"
  done
  echo "[submit] $NAME: jobs ${IDS[*]}  logs -> jeanzay/train/logs/$NAME/"
done
echo "monitor: squeue -u \$USER   |   results: (cd $REPO/MAMBA-DSR && python eval_heldout.py --report)"
