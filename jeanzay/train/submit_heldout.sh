#!/bin/bash
# ===========================================================================
#  submit_heldout.sh -- held-out evaluation of trained MAMBA-DSR runs (no training):
#    interpolation (midpoints between training frames), future prediction
#    (next 1-10 / 11-20 frames), and copy-nearest / copy-last reference baselines.
#
#     ./jeanzay/train/submit_heldout.sh                     # every finished seed-0 run
#     CASES="adk_dims ifabp_water" ./jeanzay/train/submit_heldout.sh
#     SEED=1 ./jeanzay/train/submit_heldout.sh              # runs trained with SEED=1 (_s1)
#     METHOD=dsr ./jeanzay/train/submit_heldout.sh          # every finished DSR baseline run
#     METHOD=dsr SEED=1 ./jeanzay/train/submit_heldout.sh   # DSR runs trained with SEED=1
#     MODE=hold_t ./jeanzay/train/submit_heldout.sh         # diagnostic: decoder time held at t=1
#     MODE=hold_z ./jeanzay/train/submit_heldout.sh         # diagnostic: Mamba latent held at t=1
#     MODE=hold_all ./jeanzay/train/submit_heldout.sh       # diagnostic: decoder time + latent held
#     MODE=lerp ./jeanzay/train/submit_heldout.sh           # ablation: interpolate Stage-A codes, no Stage B
#     DRY=1 ./jeanzay/train/submit_heldout.sh               # print the sbatch commands only
#  Already-finished evaluations are skipped (DONE marker), so re-running is cheap.
#  Results:  cd MAMBA-DSR && python eval_heldout.py --report
# ===========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
METHOD="${METHOD:-mamba}"; MODE="${MODE:-extrapolate}"
SEED="${SEED:-0}"; NCHAIN="${NCHAIN:-2}"; QOS="${QOS:-qos_gpu-t3}"; TIME="${TIME:-10:00:00}"
TAG=""; [ "$SEED" != 0 ] && TAG="_s$SEED"
if [ "$METHOD" = dsr ]; then MODE=extrapolate; fi
SUFFIX=""; [ "$MODE" != extrapolate ] && SUFFIX="_$MODE"
wall_min() { IFS=: read -r h m _ <<<"$1"; echo $(( 10#$h * 60 + 10#$m )); }

if [ -z "${CASES:-}" ]; then   # every run whose training finished (DONE marker)
  CASES=$(for d in "$REPO"/jeanzay/train/state/${METHOD}_*"$TAG"; do
            [ -e "$d/DONE" ] || continue
            n=$(basename "$d"); n=${n#${METHOD}_}
            [[ "$n" == *_smoke ]] && continue
            if [ -n "$TAG" ]; then [[ "$n" == *"$TAG" ]] && echo "${n%$TAG}"
            else [[ "$n" =~ _s[0-9]+$ ]] || echo "$n"; fi
          done)
fi
[ -n "$CASES" ] || { echo "no finished training runs found"; exit 1; }

for CASE in $CASES; do
  if [ "$METHOD" = dsr ]; then NAME="heldout_dsr_${CASE}${TAG}"; else NAME="heldout_${CASE}${TAG}${SUFFIX}"; fi
  LOGDIR="$REPO/jeanzay/train/logs/$NAME"; mkdir -p "$LOGDIR"
  DEP=""; IDS=()
  for ((i = 1; i <= NCHAIN; i++)); do
    CMD=(sbatch --parsable --job-name="$NAME" --qos="$QOS" --time="$TIME"
         ${CONSTRAINT:+--constraint=$CONSTRAINT} $DEP --output="$LOGDIR/%j.out"
         --export=ALL,CASE="$CASE",SEED="$SEED",TAG="$TAG",METHOD="$METHOD",MODE="$MODE",NAME="$NAME",REPO="$REPO",WALL_MIN="$(wall_min "$TIME")"
         "$HERE/heldout.slurm")
    if [ "${DRY:-0}" = 1 ]; then echo "${CMD[*]}"; JID="DRY$i"; else JID=$("${CMD[@]}"); fi
    IDS+=("$JID"); DEP="--dependency=afterany:$JID"
  done
  echo "[submit] $NAME: jobs ${IDS[*]}  logs -> jeanzay/train/logs/$NAME/"
done
echo "monitor: squeue -u \$USER   |   results: (cd $REPO/MAMBA-DSR && python eval_heldout.py --report)"
