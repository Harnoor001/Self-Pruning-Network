# Running the research experiments

All commands run from the repository root. Every script accepts `--dry-run` (print the
plan, train nothing), `--pilot` (seeds 42 123 2024), `--seeds ...`, `--device cpu|cuda`
and `--set key=value` overrides. Every unit of work writes its own result file and is
skipped if that file exists, so **re-running a command resumes it**; source training
also resumes from `last.pt` at epoch granularity.

The legacy pipeline (`scripts/train_and_report.py`, `scripts/run_*.py`,
`artifacts/<legacy dirs>/`) is unchanged and still works; research outputs go to
`artifacts/research/<experiment>/`.

## 0. Environment

```bash
pip install -e .[dev]          # adds scipy for the statistics
pytest -q                      # 20 legacy + 44 research tests
```

Check the GPU (also recorded automatically in `<output_root>/environments/*.json`):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

## 1. Smoke tests (CPU, ~3 min each — results are not evidence)

```bash
python scripts/research/run_comparison.py --config configs/smoke_mlp.yaml --seeds 42
python scripts/research/run_comparison.py --config configs/smoke_vgg.yaml --seeds 42
python scripts/research/run_importance_analysis.py --config configs/smoke_vgg.yaml --seeds 42
python scripts/research/benchmark.py --config configs/smoke_vgg.yaml
python scripts/research/analyze_results.py --config configs/smoke_vgg.yaml
```

## 2. Stage A verification on the legacy checkpoints (CPU-feasible, ~2–3 h on CPU)

Loads the committed 5-epoch legacy dense/gated checkpoints for seeds 42/123/2024 and
reproduces each seed's legacy train/validation split exactly.

```bash
python scripts/research/run_comparison.py --config configs/mlp_legacy_sources.yaml
python scripts/research/run_importance_analysis.py --config configs/mlp_legacy_sources.yaml
python scripts/research/benchmark.py --config configs/mlp_legacy_sources.yaml
python scripts/research/analyze_results.py --config configs/mlp_legacy_sources.yaml
```

## 3. Stage A, research protocol (GPU recommended)

```bash
python scripts/research/run_comparison.py --config configs/mlp.yaml --pilot --dry-run
python scripts/research/train_sources.py   --config configs/mlp.yaml --pilot        # dense, gated, slimming
python scripts/research/run_comparison.py  --config configs/mlp.yaml --pilot
python scripts/research/run_importance_analysis.py --config configs/mlp.yaml --pilot
python scripts/research/run_ablation.py    --config configs/mlp.yaml --pilot --which A C
# final: extend to five seeds (already-finished seeds are skipped)
python scripts/research/run_comparison.py  --config configs/mlp.yaml
python scripts/research/run_importance_analysis.py --config configs/mlp.yaml
python scripts/research/benchmark.py       --config configs/mlp.yaml --device cuda
python scripts/research/analyze_results.py --config configs/mlp.yaml
```

## 4. Stage B, VGG-16-BN (GPU required in practice)

```bash
python scripts/research/run_comparison.py  --config configs/vgg_cifar10.yaml --pilot --dry-run
python scripts/research/train_sources.py   --config configs/vgg_cifar10.yaml --seeds 42 --sources dense   # ~30 min, check accuracy first
python scripts/research/train_sources.py   --config configs/vgg_cifar10.yaml --pilot
python scripts/research/run_comparison.py  --config configs/vgg_cifar10.yaml --pilot
python scripts/research/run_importance_analysis.py --config configs/vgg_cifar10.yaml --pilot
python scripts/research/run_ablation.py    --config configs/vgg_cifar10.yaml --pilot --which A C
python scripts/research/run_comparison.py  --config configs/vgg_cifar10.yaml           # five seeds
python scripts/research/run_importance_analysis.py --config configs/vgg_cifar10.yaml
python scripts/research/benchmark.py       --config configs/vgg_cifar10.yaml --device cuda
python scripts/research/analyze_results.py --config configs/vgg_cifar10.yaml
```

Do not launch Stage A and Stage B simultaneously on one GPU. Resource estimates are in
`experimental_protocol.md` §13. Useful overrides: `--set data.num_workers=8`,
`--set pruning.save_compact_checkpoints=true`.

## 5. Where results land

| File | Content |
|---|---|
| `<root>/results.csv`, `results.json` | one row per seed × method × ratio × FT budget (incl. config-derived fields) |
| `<root>/seed*/sources/<name>/info.json`, `training_history.csv` | source training metrics, time, config |
| `<root>/seed*/comparison/histories/*.csv` | fine-tuning curves |
| `<root>/seed*/comparison/masks/*.pt` | kept-unit indices (git-ignored) |
| `<root>/seed*/reliability/{correlations,damage,stability}.csv` | RQ5 / RQ3 |
| `<root>/benchmark/benchmark_results.csv` | measured latency, memory, size, MACs |
| `<root>/analysis/{summary,paired_tests,reliability_summary}.csv`, `findings.md`, `plots/` | analysis |
| `research/paper/{figures,tables}/<name>/` | copies used by the manuscript |
| `<root>/experiment_config.yaml`, `environments/*.json` | resolved config and environment per invocation |
