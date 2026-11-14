#!/usr/bin/env bash
# Stage B full protocol (configs/vgg_cifar10.yaml) on a Kaggle "GPU T4 x2" session.
#
# * Uses both GPUs: a shared task queue hands out whole seeds (sources -> comparison ->
#   reliability), then ablation C and A per pilot seed, to whichever GPU is free.
# * Stops cleanly before the time budget (default 11 h, below Kaggle's 12 h session limit) so
#   the session's /kaggle/working output is saved. Interrupted work resumes next session:
#   source training at epoch granularity, the comparison at row granularity.
# * Resumes from a previous session's output when RESTORE points at its vgg16 folder.
# * When every task is done it runs the benchmark (GPU) and the analysis.
#
# Usage, from the repository root:
#   bash scripts/research/run_vgg_kaggle.sh
#   RESTORE=/kaggle/input/<previous-output>/Self-Pruning-Network/artifacts/research/vgg16 \
#       bash scripts/research/run_vgg_kaggle.sh
# Environment: BUDGET_H (default 11), GPUS (default "0 1"), PY (default python),
#              DATA (default /tmp/cifar10, outside the saved output), WORKERS (default 2 per GPU),
#              DRY=1 prints the task order without running anything.
set -u
PY=${PY:-python}
CFG=configs/vgg_cifar10.yaml
OUT=artifacts/research/vgg16
DATA=${DATA:-/tmp/cifar10}
SET="data.data_dir=$DATA data.num_workers=${WORKERS:-2}"
GPUS=${GPUS:-"0 1"}
DEADLINE=$(( $(date +%s) + ${BUDGET_H:-11} * 3600 ))
STATE=$OUT/.kaggle_state
mkdir -p "$OUT" "$STATE"

if [ -n "${RESTORE:-}" ]; then
  echo "=== restoring previous output from $RESTORE"
  cp -r "$RESTORE"/. "$OUT"/ || { echo "restore failed"; exit 1; }
  chmod -R u+w "$OUT"                                  # /kaggle/input is read-only
fi
rm -rf "$STATE/claims"; mkdir -p "$STATE/claims"   # claims are per session; done markers persist

TASKS="seed42 seed123 seed2024 seed7 seed21 C42 C123 C2024 A42 A123 A2024"

if [ "${DRY:-0}" = 1 ]; then
  for t in $TASKS; do [ -e "$STATE/done_$t" ] && echo "done    $t" || echo "pending $t"; done
  exit 0
fi

"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() >= 1 else 'no CUDA GPU')" || exit 1
# download CIFAR-10 once, before the workers start (avoids two concurrent downloads)
"$PY" -c "from torchvision import datasets; [datasets.CIFAR10(root='$DATA', train=t, download=True) for t in (True, False)]" \
  > /dev/null || { echo "CIFAR-10 download failed (is Internet on in the notebook settings?)"; exit 1; }

left() { echo $(( DEADLINE - $(date +%s) )); }

# step GPU SCRIPT ARGS...: run one script on one GPU, killed with SIGINT when the budget ends
step() {
  local gpu=$1; shift
  local t; t=$(left)
  [ "$t" -gt 600 ] || return 124
  echo "=== $(date '+%m-%d %H:%M:%S') [gpu$gpu] $*"
  CUDA_VISIBLE_DEVICES=$gpu timeout --signal=INT --kill-after=120 "$t" \
    "$PY" "$@" --device cuda --set $SET >> "$OUT/pipeline_gpu$gpu.out" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || echo "=== $(date '+%m-%d %H:%M:%S') [gpu$gpu] rc=$rc: $*"
  return $rc
}

run_task() {
  local gpu=$1 task=$2
  case $task in
    seed*) local s=${task#seed}
           step "$gpu" scripts/research/train_sources.py           --config $CFG --seeds "$s" &&
           step "$gpu" scripts/research/run_comparison.py          --config $CFG --seeds "$s" &&
           step "$gpu" scripts/research/run_importance_analysis.py --config $CFG --seeds "$s" ;;
    C*)    step "$gpu" scripts/research/run_ablation.py --config $CFG --seeds "${task#C}" --which C ;;
    A*)    step "$gpu" scripts/research/run_ablation.py --config $CFG --seeds "${task#A}" --which A ;;
  esac
}

worker() {
  local gpu=$1 task
  for task in $TASKS; do
    [ -e "$STATE/done_$task" ] && continue
    mkdir "$STATE/claims/$task" 2>/dev/null || continue        # atomic claim
    # ablations need the main sources of their seed
    if [ "${task#seed}" = "$task" ]; then
      local s=${task#?}
      while [ ! -e "$STATE/done_seed$s" ]; do
        [ "$(left)" -gt 600 ] || return 0
        if [ ! -d "$STATE/claims/seed$s" ]; then break; fi          # unclaimed: run it below
        sleep 60
      done
      if [ ! -e "$STATE/done_seed$s" ]; then
        mkdir "$STATE/claims/seed$s" 2>/dev/null && { run_task "$gpu" "seed$s" && touch "$STATE/done_seed$s" || return 0; }
      fi
    fi
    if run_task "$gpu" "$task"; then
      touch "$STATE/done_$task"
      echo "=== $(date '+%m-%d %H:%M:%S') [gpu$gpu] DONE $task"
    else
      return 0                                                  # budget reached or failure
    fi
  done
}

for g in $GPUS; do worker "$g" & done
wait

pending=""
for t in $TASKS; do [ -e "$STATE/done_$t" ] || pending="$pending $t"; done
if [ -n "$pending" ]; then
  echo "=== $(date '+%m-%d %H:%M:%S') SESSION END, pending:$pending"
  echo "Save this version, then start the next session with RESTORE pointing at this output."
  exit 0
fi

g0=${GPUS%% *}
# A re-run with all pilot seeds only rebuilds ablation_gate_statistics.csv (every row is done)
step "$g0" scripts/research/run_ablation.py --config $CFG --pilot --which A &&
step "$g0" scripts/research/benchmark.py       --config $CFG &&
step "$g0" scripts/research/analyze_results.py --config $CFG &&
echo "=== $(date '+%m-%d %H:%M:%S') PIPELINE DONE"
