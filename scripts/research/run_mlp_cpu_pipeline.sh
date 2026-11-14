#!/usr/bin/env bash
# Full Stage A (MLP) research protocol, sequential and resumable (every step skips finished work).
# Order puts the main five-seed results first, then ablation C, then ablation A (most expensive).
# Usage: bash scripts/research/run_mlp_cpu_pipeline.sh [python]
set -u
PY=${1:-python}
CFG=configs/mlp.yaml
OUT=artifacts/research/mlp
mkdir -p "$OUT"
step() { echo "=== $(date +%H:%M:%S) $*"; "$PY" "$@" >> "$OUT/pipeline.out" 2>&1 || { echo "FAILED: $*"; exit 1; }; }
step scripts/research/train_sources.py           --config $CFG --pilot
step scripts/research/run_comparison.py          --config $CFG --pilot
step scripts/research/run_importance_analysis.py --config $CFG --pilot
step scripts/research/train_sources.py           --config $CFG
step scripts/research/run_comparison.py          --config $CFG
step scripts/research/run_importance_analysis.py --config $CFG
step scripts/research/benchmark.py               --config $CFG
step scripts/research/analyze_results.py         --config $CFG
step scripts/research/run_ablation.py            --config $CFG --pilot --which C
step scripts/research/run_ablation.py            --config $CFG --pilot --which A
step scripts/research/analyze_results.py         --config $CFG
echo "=== $(date +%H:%M:%S) PIPELINE DONE"
