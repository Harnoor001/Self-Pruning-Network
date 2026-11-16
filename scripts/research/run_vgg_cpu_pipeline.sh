#!/usr/bin/env bash
# Stage B (VGG-16-BN) REDUCED-BUDGET CPU study: configs/vgg_cifar10_cpu.yaml.
# Not the full protocol (see the header of that config for every deviation).
# Runs one seed completely (sources -> comparison -> reliability -> analysis) before the next,
# so a usable single-seed result exists early. Sequential and resumable.
# Usage: bash scripts/research/run_vgg_cpu_pipeline.sh [python]
set -u
PY=${1:-python}
CFG=configs/vgg_cifar10_cpu.yaml
OUT=artifacts/research/vgg16_cpu
mkdir -p "$OUT"
step() { echo "=== $(date '+%m-%d %H:%M:%S') $*"; "$PY" "$@" --device cpu >> "$OUT/pipeline.out" 2>&1 || { echo "FAILED: $*"; exit 1; }; }
for SEED in 42 123 2024; do
  step scripts/research/train_sources.py           --config $CFG --seeds $SEED
  step scripts/research/run_comparison.py          --config $CFG --seeds $SEED
  step scripts/research/run_importance_analysis.py --config $CFG --seeds $SEED
  # interim analysis: allowed to fail (statistics need more than one seed); the final one below is not
  echo "=== $(date '+%m-%d %H:%M:%S') interim analysis after seed $SEED"
  "$PY" scripts/research/analyze_results.py --config $CFG >> "$OUT/pipeline.out" 2>&1 || echo "interim analysis failed (non-fatal)"
done
step scripts/research/benchmark.py                 --config $CFG
step scripts/research/analyze_results.py           --config $CFG
echo "=== $(date '+%m-%d %H:%M:%S') PIPELINE DONE"
