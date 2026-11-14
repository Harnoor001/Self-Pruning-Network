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

## 3. Stage A, research protocol (CPU-feasible: ~20 h on a 20-thread CPU; ~3–4 h on a GPU)

One resumable command runs the whole sequence below (pilot → five seeds → reliability →
benchmark → analysis → ablation C → ablation A):

```bash
bash scripts/research/run_mlp_cpu_pipeline.sh python
```

Or step by step:

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

One resumable command (refuses to start without CUDA; ~4–5 h per seed plus ablations):

```bash
bash scripts/research/run_vgg_gpu_pipeline.sh python
```

Or step by step:

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

### 4b. Reduced-budget Stage B on a CPU (not the full protocol)

`configs/vgg_cifar10_cpu.yaml` keeps the VGG-16-BN architecture, split, optimizer, ratios and
selection rule, and cuts only the compute budget: 20-epoch sources (instead of 160), 3
fine-tuning epochs (instead of 20), seeds 42/123/2024, 9 methods, 16 reliability units per
layer, no ablations. The header of that file lists every deviation; results from it must be
reported as reduced-budget results. About 10 min per source epoch on a 20-thread laptop CPU,
roughly 15–18 h per seed.

```bash
bash scripts/research/run_vgg_cpu_pipeline.sh python     # one seed at a time, resumable
```

Outputs go to `artifacts/research/vgg16_cpu/`, separate from the full-protocol folder
`artifacts/research/vgg16/`.

### 4d. Full Stage B on Kaggle (GPU T4 x2)

`scripts/research/run_vgg_kaggle.sh` runs the full protocol (`configs/vgg_cifar10.yaml`) on
both T4s: seeds and ablation tasks go to whichever GPU is free. It stops by itself after
`BUDGET_H` hours (default 11, under Kaggle's 12 h limit) so that the output is saved, and a
later session resumes from that output.

1. New notebook → Settings → **Accelerator: GPU T4 x2**, **Internet on** (needs a
   phone-verified Kaggle account).
2. One code cell (the whole cell is bash):

   ```bash
   %%bash
   pip install -q tabulate
   git clone -q -b research/comparative-pruning-study https://github.com/Harnoor001/Self-Pruning-Network.git
   cd Self-Pruning-Network
   export RESTORE=$(find /kaggle/input -type d -path '*artifacts/research/vgg16' 2>/dev/null | head -1)
   bash scripts/research/run_vgg_kaggle.sh
   ```

3. **Save Version → Save & Run All (Commit)**. It runs without the browser open and keeps
   `/kaggle/working` as the version's output. Progress: `artifacts/research/vgg16/pipeline_gpu{0,1}.out`.
4. Next session: **Add Input → Your Work →** the previous version's output, then Save & Run All
   again. The cell finds the previous results automatically, and finished seeds are skipped.
   Repeat until the log ends with `PIPELINE DONE`, then download the output's
   `artifacts/research/vgg16/` folder.

`DRY=1 bash scripts/research/run_vgg_kaggle.sh` lists finished and pending tasks without
running anything. Time on T4s has not been measured; check the per-epoch times in the logs
after the first few minutes. A rough guess is 15–20 h for the five main seeds on two GPUs plus
about 10 h for ablation A (it trains six extra gated sources per pilot seed), i.e. three or
more 11 h sessions.

## 4c. Compiling the manuscript

Any LaTeX distribution works (`pdflatex` + `bibtex`). Without one, Tectonic is a single
binary:

```bash
conda create -y -n texbuild --override-channels -c conda-forge tectonic
cd research/paper && conda run -n texbuild tectonic main.tex
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
