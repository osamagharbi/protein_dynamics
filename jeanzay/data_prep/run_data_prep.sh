#!/bin/bash
# ===========================================================================
#  run_data_prep.sh -- download + convert protein-surface-dynamics datasets on
#  Jean-Zay for DSR / MAMBA-DSR, fully in parallel, in ONE command.
#
#  Per raw source (independent chains, so one failure never blocks the rest):
#     download job (prepost: internet, not charged)          -> raw trajectories
#       └─afterok─> convert ARRAY (1 task = 1 frame chunk)     -> gt/traj_<i>.obj
#                     └─afterany─> finalize job               -> meta.json + train view
#
#  Usage (from a LOGIN node, after ./setup_env.sh):
#     DRY_RUN=1 ./run_data_prep.sh          # plan + sbatch --test-only, submits nothing
#     ./run_data_prep.sh                    # paper datasets (Aβ, AdK x3, I-FABP, NhaA, YiiP, 14 DRY_AD_MD)
#     DATASETS="adk_dims ifabp_water" ./run_data_prep.sh
#     DATASETS=atlas ATLAS_LIMIT=50 ./run_data_prep.sh
#     PHASE=convert ./run_data_prep.sh      # data already downloaded: convert + finalize only
#     DL_TARGET=login ./run_data_prep.sh    # if prepost is unavailable: download here (use tmux)
# ===========================================================================
set -euo pipefail
PREP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------- WHAT --------------------------------------
DATASETS="${DATASETS:-paper}"            # groups: paper dryad dryad_excluded gpcrmd atlas all | names
PHASE="${PHASE:-all}"                    # all | download | convert | finalize
CHUNK_SIZE="${CHUNK_SIZE:-200}"          # frames per convert task
ATLAS_LIMIT="${ATLAS_LIMIT:-0}"          # first N ATLAS entries (0 = none); or ATLAS_IDS=file|list
ATLAS_IDS="${ATLAS_IDS:-}"
ATLAS_REPLICA="${ATLAS_REPLICA:-1}"; ATLAS_STRIDE="${ATLAS_STRIDE:-100}"; ATLAS_PARALLEL="${ATLAS_PARALLEL:-4}"

# ------------------------------- WHERE -------------------------------------
RAW_ROOT="${RAW_ROOT:-$WORK/protein_dynamics/datasets/raw}"
PROC_ROOT="${PROC_ROOT:-$WORK/protein_dynamics/datasets/DSR_processed}"   # = MAMBA-DSR paper_protocol layout

# ------------------------------- CLUSTER -----------------------------------
ACCOUNT="${ACCOUNT:-acu@v100}"
DL_TARGET="${DL_TARGET:-prepost}"        # prepost | login
DL_ACCOUNT="${DL_ACCOUNT:-$ACCOUNT}"
DL_TIME="${DL_TIME:-03:00:00}"        # short walltime = backfills sooner
PROC_TARGET="${PROC_TARGET:-gpu}"        # gpu (acu@v100, known to work) | cpu (needs CPU_ACCOUNT) | prepost
PROC_TIME="${PROC_TIME:-01:00:00}"      # 200 frames / 10 procs ~ minutes; short = backfills
CONCURRENCY="${CONCURRENCY:-40}"         # running convert tasks per source array
DRY_RUN="${DRY_RUN:-0}"

case "$PROC_TARGET" in
  gpu)     PROC_FLAGS=(--account="$ACCOUNT" --qos="${PROC_QOS:-qos_gpu-t3}" --gres=gpu:1
                       ${CONSTRAINT:+--constraint=$CONSTRAINT} --cpus-per-task="${PROC_CPUS:-10}" --hint=nomultithread) ;;
  cpu)     PROC_FLAGS=(--account="${CPU_ACCOUNT:?set CPU_ACCOUNT, e.g. acu@cpu}" --partition=cpu_p1
                       --qos="${PROC_QOS:-qos_cpu-t3}" --cpus-per-task="${PROC_CPUS:-40}" --hint=nomultithread) ;;
  prepost) PROC_FLAGS=(--account="$DL_ACCOUNT" --partition=prepost --cpus-per-task="${PROC_CPUS:-8}") ;;
  *) echo "PROC_TARGET must be gpu|cpu|prepost"; exit 1 ;;
esac
DL_FLAGS=(--account="$DL_ACCOUNT" --partition=prepost --cpus-per-task=4)
# finalize is light: run it where downloads run (prepost, uncharged) unless downloads are on login
if [ "$DL_TARGET" = prepost ]; then FIN_FLAGS=("${DL_FLAGS[@]}"); else FIN_FLAGS=("${PROC_FLAGS[@]}"); fi
[ "$PROC_TARGET" = prepost ] && FIN_FLAGS=("${DL_FLAGS[@]}")

[ -f "$PREP_DIR/env.sh" ] || { echo "env.sh missing -> run ./setup_env.sh first"; exit 1; }
module purge 2>/dev/null || true
source "$PREP_DIR/env.sh"
export PREP_DIR RAW_ROOT PROC_ROOT ATLAS_LIMIT ATLAS_IDS ATLAS_REPLICA ATLAS_STRIDE ATLAS_PARALLEL
mkdir -p "$RAW_ROOT" "$PROC_ROOT"

PLAN_DIR="$PROC_ROOT/_plans/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PLAN_DIR/logs"; mkdir -p "$LOG_DIR"

# ------------------------------- PLAN --------------------------------------
PLAN_OUT=$(python "$PREP_DIR/prep.py" plan --datasets "$DATASETS" --chunk-size "$CHUNK_SIZE" \
                   --plan-dir "$PLAN_DIR" 2> "$LOG_DIR/plan.err") || { cat "$LOG_DIR/plan.err"; exit 1; }
NFRAMES=$(awk -F'\t' 'NR>1{s+=$6} END{print s+0}' "$PLAN_DIR/sources.tsv")
cat <<BANNER
===========================================================================
 DSR data prep   phase=$PHASE   dry_run=$DRY_RUN
 datasets   : $DATASETS
 raw root   : $RAW_ROOT
 output     : $PROC_ROOT/<dataset>/gt/traj_<i>.obj
 download   : $DL_TARGET  (${DL_FLAGS[*]})
 convert    : $PROC_TARGET  (${PROC_FLAGS[*]})  chunk=$CHUNK_SIZE  %$CONCURRENCY/source
 $PLAN_OUT
 frames     : <= $NFRAMES   (~$(( NFRAMES * 8 / 1000 )) GB at ~8 MB/frame; NhaA/YiiP ~23 MB/frame)
 plan + logs: $PLAN_DIR
---------------------------------------------------------------------------
BANNER
awk -F'\t' '{printf "  %-22s %-15s %8s %8s  %s\n", $1, $2, $5, $6, $3}' "$PLAN_DIR/sources.tsv"
echo "==========================================================================="

if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] sbatch --test-only checks (no submission):"
  test_spec() {  # $1=label, rest=sbatch flags
    local label="$1"; shift; local out
    if out=$(sbatch --test-only "$@" --time=01:00:00 --wrap=true 2>&1); then echo "  $label OK    $out"
    else echo "  $label FAIL  $out"; fi
  }
  [ "$DL_TARGET" = prepost ] && test_spec "download/finalize" "${DL_FLAGS[@]}"
  test_spec "convert          " "${PROC_FLAGS[@]}" --array=0-1
  [ "$DL_TARGET" = prepost ] && echo "  (prepost must also reach the internet; the download job checks this first)"
  echo "DRY_RUN=1 -> nothing submitted."; exit 0
fi

# ------------------------------- SUBMIT ------------------------------------
MANIFEST="$PLAN_DIR/MANIFEST.tsv"
echo -e "source\tdownload_jid\tconvert_jid\tfinalize_jid\tn_tasks" > "$MANIFEST"
LOGIN_PIDS=()
while IFS=$'\t' read -r SRC KIND DSETS TASKS NT NF; do
  [ "$SRC" = source ] && continue
  DL=""; CV=""; FIN=""
  if [[ "$PHASE" =~ ^(all|download)$ ]] && [ "$KIND" != manual ]; then
    if [ "$DL_TARGET" = prepost ]; then
      # values contain commas -> pass them through the inherited environment, not --export=VAR=
      DL=$(SOURCE="$SRC" DATASETS="$DSETS" sbatch --parsable "${DL_FLAGS[@]}" --time="$DL_TIME" \
             --job-name="DL_$SRC" --export=ALL \
             --output="$LOG_DIR/dl_${SRC}_%j.out" --error="$LOG_DIR/dl_${SRC}_%j.err" \
             "$PREP_DIR/slurm_download.slurm")
    else
      python -u "$PREP_DIR/prep.py" download --source "$SRC" --datasets "$DSETS" \
             > "$LOG_DIR/dl_${SRC}_login.log" 2>&1 &
      LOGIN_PIDS+=("$!:$SRC")
    fi
  fi
  if [[ "$PHASE" =~ ^(all|convert)$ ]]; then
    DEP=(); [ -n "$DL" ] && DEP=(--dependency=afterok:"$DL" --kill-on-invalid-dep=yes)
    if [ "$DL_TARGET" = login ] && [ "${#LOGIN_PIDS[@]}" -gt 0 ]; then
      : # convert is submitted after the login downloads finish (below)
    else
      CV=$(TASKS="$TASKS" sbatch --parsable "${PROC_FLAGS[@]}" "${DEP[@]}" --time="$PROC_TIME" \
             --array=0-$((NT - 1))%"$CONCURRENCY" --job-name="CV_$SRC" --export=ALL \
             --output="$LOG_DIR/cv_${SRC}_%A_%a.out" --error="$LOG_DIR/cv_${SRC}_%A_%a.err" \
             "$PREP_DIR/slurm_convert.slurm")
    fi
  fi
  if [[ "$PHASE" =~ ^(all|convert|finalize)$ ]] && { [ -n "$CV" ] || [ "$PHASE" = finalize ]; }; then
    DEP=(); [ -n "$CV" ] && DEP=(--dependency=afterany:"$CV")
    FIN=$(DATASETS="$DSETS" sbatch --parsable "${FIN_FLAGS[@]}" "${DEP[@]}" --time=00:30:00 \
            --job-name="FIN_$SRC" --export=ALL \
            --output="$LOG_DIR/fin_${SRC}_%j.out" --error="$LOG_DIR/fin_${SRC}_%j.err" \
            "$PREP_DIR/slurm_finalize.slurm")
  fi
  echo -e "$SRC\t${DL:--}\t${CV:--}\t${FIN:--}\t$NT" >> "$MANIFEST"
  printf "  %-22s download=%-9s convert=%-9s(%4s tasks) finalize=%s\n" "$SRC" "${DL:--}" "${CV:--}" "$NT" "${FIN:--}"
done < "$PLAN_DIR/sources.tsv"

# DL_TARGET=login: wait for downloads here, then submit convert+finalize for the ones that succeeded
if [ "${#LOGIN_PIDS[@]}" -gt 0 ]; then
  echo "[login] downloading ${#LOGIN_PIDS[@]} sources on $(hostname); logs in $LOG_DIR/dl_*_login.log"
  for entry in "${LOGIN_PIDS[@]}"; do
    pid="${entry%%:*}"; src="${entry#*:}"
    if wait "$pid"; then
      echo "  $src downloaded -> submitting convert+finalize"
      if [ "$PHASE" = all ]; then   # re-plan just this source (now indexed) and submit convert+finalize
        PHASE=convert DATASETS="$(awk -F'\t' -v s="$src" '$1==s{print $3}' "$PLAN_DIR/sources.tsv")" \
          bash "$0" | grep -E "^  $src|manifest" || true
      fi
    else
      echo "  $src download FAILED -> see $LOG_DIR/dl_${src}_login.log"
    fi
  done
fi

echo
echo "manifest : $MANIFEST"
echo "watch    : squeue -u \$USER -o '%.10i %.14j %.8T %.10M %R'"
echo "progress : python $PREP_DIR/prep.py status --proc-root $PROC_ROOT"
echo "retry    : re-run the same command (finished downloads/frames are skipped)"
