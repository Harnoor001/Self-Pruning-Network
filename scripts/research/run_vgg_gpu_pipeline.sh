#!/usr/bin/env bash
# Stage B (VGG-16-BN, CIFAR-10) research protocol on one CUDA GPU, sequential and resumable.
# Every step skips finished work, so re-running after an interruption (e.g. a Colab timeout) continues.
# Estimated time on one modern GPU with AMP: ~4-5 h per seed (experimental_protocol.md section 13).
# Usage: bash scripts/research/run_vgg_gpu_pipeline.sh [python]
set -u
PY=${1:-python}
CFG=configs/vgg_cifar10.yaml
OUT=artifacts/research/vgg_cifar10
mkdir -p "$OUT"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'No CUDA GPU: Stage B is not CPU-feasible')" || exit 1
step() { echo "=== $(date +%H:%M:%S) $*"; "$PY" "$@" --device cuda >> "$OUT/pipeline.out" 2>&1 || { echo "FAILED: $*"; exit 1; }; }
# 1. sanity check: one dense source, inspect its validation accuracy before spending the full budget
step scripts/research/train_sources.py           --config $CFG --seeds 42 --sources dense
# 2. pilot (3 seeds), then the remaining seeds
step scripts/research/train_sources.py           --config $CFG --pilot
step scripts/research/run_comparison.py          --config $CFG --pilot
step scripts/research/run_importance_analysis.py --config $CFG --pilot
step scripts/research/run_comparison.py          --config $CFG
step scripts/research/run_importance_analysis.py --config $CFG
step scripts/research/benchmark.py               --config $CFG
step scripts/research/analyze_results.py         --config $CFG
# 3. ablations on the pilot seeds (A retrains six gated sources per seed: the most expensive part)
step scripts/research/run_ablation.py            --config $CFG --pilot --which C
step scripts/research/run_ablation.py            --config $CFG --pilot --which A
step scripts/research/analyze_results.py         --config $CFG
echo "=== $(date +%H:%M:%S) PIPELINE DONE"
