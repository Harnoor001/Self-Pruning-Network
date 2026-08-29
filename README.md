# Self-Pruning Neural Network

A PyTorch CIFAR-10 MLP that learns per-connection gate importance, converts those scores into explicit binary masks, hard-prunes connections, and fine-tunes with a fixed mask.

## Problem

Pruning can reduce the logical connectivity of a neural network. This project distinguishes logical sparsity from physical storage reduction and inference acceleration: setting entries to zero in an ordinary dense tensor does not automatically reduce checkpoint size, memory allocation, latency, or hardware work.

## Method

Each prunable weight has a learnable score S:

W_soft = W * sigmoid(S)

Training uses:

L = L_CE + lambda * L_sparsity

The original implementation initialized every score at 2.0, so each gate started at sigmoid(2) approximately 0.881. It measured sparsity only as gates below 0.01. Sigmoid saturation, gate-gradient scale, and the unnormalized sum over millions of gates made that threshold hard to reach. The old implementation also never created a binary mask or zeroed weights.

The corrected pipeline treats gates as importance scores. A gate does not need to naturally fall below 0.01:

gate learning -> importance ranking -> binary mask -> hard pruning -> fixed-mask fine-tuning -> evaluation

## Hard pruning

Threshold pruning uses:

M_i = 1 if g_i >= tau, else 0
W_hard = W * M

Target-sparsity pruning ranks all gate values globally across all prunable layers and removes the requested fraction. The model exposes soft, dense, and hard modes. During fine-tuning, the mask is re-applied after every optimizer step, preserving M_i = 0 => W_i = 0.

## Experimental audit

The old 51.74% result and the previous 31.84% result were not comparable:

| Configuration | Original reported run | Previous realistic run |
|---|---:|---:|
| Hidden dimensions | 2048 / 1024 / 512 | 128 / 64 |
| Weight connections | 8,918,016 | 402,048 |
| Training samples | 45,000 | 1,844 |
| Validation samples | 5,000 | 204 |
| Test samples | 10,000 | 1,024 |
| Epochs | 5 | 2 |
| Batch size | 128 | 128 |
| Learning rate | 0.0008 | 0.0008 |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| λ values | 0.0001 / 0.001 / 0.01 | 0.0001 / 0.001 |
| Hard pruning | No | Yes |

The architecture and dataset/training-budget changes explain the 51.74% to 31.84% difference. The old 51.74% checkpoint was a full-data soft-gated model with zero measured hard sparsity; the 31.84% result was a smaller subset experiment.

## Final full-data benchmark

This benchmark used the original architecture, full CIFAR-10 data, seed 42, deterministic 90/10 train-validation split, five training epochs, batch size 128, AdamW, learning rate 0.0008, weight decay 0.0001, CosineAnnealingLR, label smoothing 0.05, target-sparsity pruning, and one fine-tuning epoch at learning rate 0.0002.

Accuracy drops for hard models are relative to the corresponding unpruned selected soft model at λ = 0.0. Soft-model drops are relative to the separately trained dense baseline.

| Model | λ | Target sparsity | Actual sparsity | Density | Validation accuracy | Test accuracy | Accuracy drop | Total | Active | Pruned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | — | 0% | 0% | 100% | 51.66% | 52.21% | 0 pp | 8,918,016 | 8,918,016 | 0 |
| Soft | 0 | — | 0% | 100% | 51.74% | 52.28% | -0.07 pp | 8,918,016 | 8,918,016 | 0 |
| Soft | 0.001 | — | 0% | 100% | 51.50% | 52.27% | -0.06 pp | 8,918,016 | 8,918,016 | 0 |
| Hard | 0 | 0% | 0% | 100% | 51.74% | 52.10% | 0.18 pp | 8,918,016 | 8,918,016 | 0 |
| Hard | 0 | 20% | 20.0000% | 80.0000% | 52.52% | 52.08% | 0.20 pp | 8,918,016 | 7,134,413 | 1,783,603 |
| Hard | 0 | 40% | 40.0000% | 60.0000% | 51.76% | 51.87% | 0.41 pp | 8,918,016 | 5,350,810 | 3,567,206 |
| Hard | 0 | 60% | 60.0000% | 40.0000% | 51.50% | 51.84% | 0.44 pp | 8,918,016 | 3,567,206 | 5,350,810 |
| Hard | 0 | 80% | 80.0000% | 20.0000% | 48.54% | 49.11% | 3.17 pp | 8,918,016 | 1,783,603 | 7,134,413 |

The maximum measured sparsity with a test-accuracy drop below one percentage point was 60%: 51.84% test accuracy, 0.44 percentage-point drop, and 5,350,810 pruned connections.

Immediately after pruning and before fine-tuning, the 20%, 40%, 60%, and 80% models measured 50.61%, 50.65%, 49.80%, and 40.15% test accuracy. After one fine-tuning epoch they measured 52.08%, 51.87%, 51.84%, and 49.11%, respectively.

The generated source-of-truth artifacts are in [artifacts/final_benchmark/reports](artifacts/final_benchmark/reports), especially [results.csv](artifacts/final_benchmark/reports/results.csv), [results.md](artifacts/final_benchmark/reports/results.md), and [summary.json](artifacts/final_benchmark/reports/summary.json).

## Reproduce

Install dependencies:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Run the final benchmark:

    python scripts\train_and_report.py --epochs 5 --batch-size 128 --hidden-dims 2048 1024 512 --lambdas 0 0.001 --prune --pruning-strategy target --target-sparsities 20 40 60 80 --fine-tune-epochs 1

For a labeled smoke test, add --train-subset and --test-subset and write to a separate output directory. Smoke-test numbers must not be presented as full-CIFAR-10 benchmark results.

Run the reproduced key learned-vs-random ablation (the executed full-data command):

    python scripts\run_ablation.py --output-dir artifacts\ablation_key_benchmark --reuse-seed42 --seed42-checkpoint-dir artifacts\final_benchmark\checkpoints --seeds 42 123 2024 --targets 60 --multi-seed-targets 60 --epochs 5 --fine-tune-epochs 1 --batch-size 128 --hidden-dims 2048 1024 512 --device cpu

This command runs the dense/soft controls and learned/random 60% comparison for all three seeds. The 20/40/60/80 random sweep was not included in the official run because the full CPU training budget is expensive.

Threshold pruning is also available:

    python scripts\train_and_report.py --prune --pruning-strategy threshold --prune-threshold 0.1 --fine-tune-epochs 1

## Checkpoints and deployment accounting

Checkpoints include model configuration, experiment configuration, gates, binary masks, pruning metadata, training history, and metrics. Legacy soft checkpoints without masks remain loadable.

Gate parameters are training-time auxiliary parameters. The current deployment representation retains dense tensor shapes, so active/pruned counts represent logical connectivity reduction, not physical tensor-size reduction. No latency or hardware speedup is claimed.

## API

Set MODEL_CHECKPOINT to a dense, soft, or hard-pruned checkpoint:

    $env:MODEL_CHECKPOINT="artifacts\final_benchmark\checkpoints\hard_target_0.6000.pt"
    python -m uvicorn app.api:app --reload

GET /model/summary reports layer metrics plus total, active, pruned, density, sparsity, and threshold fields.

## Efficiency & Compression Analysis

The efficiency benchmark reloaded the validated final checkpoints without retraining:

    python scripts\benchmark.py --benchmark-dir artifacts\final_benchmark --output-dir artifacts\efficiency_benchmark --warmup 20 --iterations 50 --batch-sizes 1 32 --device cpu --threads 1

The run used CPU, PyTorch 2.11.0+cpu, Python 3.11.9, seed 42, 20 warm-up iterations, and 50 measured iterations. The complete source-of-truth files are in [artifacts/efficiency_benchmark/reports](artifacts/efficiency_benchmark/reports), including [efficiency_results.csv](artifacts/efficiency_benchmark/reports/efficiency_results.csv), [efficiency_results.md](artifacts/efficiency_benchmark/reports/efficiency_results.md), [layer_efficiency.csv](artifacts/efficiency_benchmark/reports/layer_efficiency.csv), and [summary.json](artifacts/efficiency_benchmark/reports/summary.json).

### Logical connectivity and theoretical computation

The model has 8,918,016 dense weight slots and 8,918,016 dense linear MACs under the estimator used here. At 60% logical sparsity, 5,350,810 connections are masked and 3,567,206 remain active. The ideal sparse-execution estimate is therefore 3,567,206 effective MACs, a 60.000004% theoretical MAC reduction. This is not a measured runtime reduction: ordinary dense PyTorch kernels still operate on dense tensor shapes.

The 60% hard-pruned per-layer accounting was:

| Layer | Input | Output | Total | Active | Pruned | Dense MACs | Estimated effective MACs |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer1 | 3,072 | 2,048 | 6,291,456 | 2,476,666 | 3,814,790 | 6,291,456 | 2,476,666 |
| layer2 | 2,048 | 1,024 | 2,097,152 | 863,508 | 1,233,644 | 2,097,152 | 863,508 |
| layer3 | 1,024 | 512 | 524,288 | 225,847 | 298,441 | 524,288 | 225,847 |
| layer4 | 512 | 10 | 5,120 | 1,185 | 3,935 | 5,120 | 1,185 |

### Measured accuracy, storage, and latency

Accuracy is from the final full-CIFAR-10 benchmark. Checkpoint sizes are measured file sizes; MB is decimal and MiB is binary. Latency is mean CPU inference latency from the same benchmark run; the CSV and Markdown report also contain p50 and p95.

| Model | Sparsity | Test accuracy | Accuracy drop | Active | Pruned | Effective MACs | MAC reduction | Checkpoint bytes | MB | Batch-1 mean ms | Batch-32 mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0% | 52.21% | 0.00 pp | 8,918,016 | 0 | 8,918,016 | 0.000000% | 107,104,166 | 107.104166 | 3.443226 | 21.431260 |
| Soft | 0% | 52.28% | -0.07 pp | 8,918,016 | 0 | 8,918,016 | 0.000000% | 107,104,125 | 107.104125 | 29.452186 | 54.027230 |
| Hard 20% | 20.0000% | 52.08% | 0.20 pp | 7,134,413 | 1,783,603 | 7,134,413 | 19.999998% | 107,103,805 | 107.103805 | 13.285496 | 30.394612 |
| Hard 40% | 40.0000% | 51.87% | 0.41 pp | 5,350,810 | 3,567,206 | 5,350,810 | 39.999996% | 107,103,805 | 107.103805 | 14.249796 | 35.967308 |
| Hard 60% | 60.0000% | 51.84% | 0.44 pp | 3,567,206 | 5,350,810 | 3,567,206 | 60.000004% | 107,103,805 | 107.103805 | 13.666054 | 30.459748 |
| Hard 80% | 80.0000% | 49.11% | 3.17 pp | 1,783,603 | 7,134,413 | 1,783,603 | 80.000002% | 107,103,805 | 107.103805 | 13.694068 | 30.654848 |

The hard-pruned checkpoints remain effectively the same size as the dense checkpoint because they retain the same dense tensor dimensions. The measured hard-60 file was 107,103,805 bytes versus 107,104,166 bytes for Dense; the 361-byte difference is checkpoint metadata, not physical sparse compression. No physical parameter or memory reduction is claimed.

On this CPU run, 60% logical sparsity did not produce a dense-kernel speedup: Hard 60% averaged 13.666054 ms at batch 1 and 30.459748 ms at batch 32, compared with 3.443226 ms and 21.431260 ms for Dense. The measured mean latency changes relative to Dense were -296.896805% and -42.127658% respectively, where a negative value means slower. The single-thread run also showed scheduling variability in p95 measurements, so these are environment-specific observations, not hardware-independent claims.

### Interpretation and required future optimization

Logical sparsity, theoretical MAC reduction, physical storage, and measured latency are separate properties. Actual storage reduction would require a physically sparse or compressed representation. Actual acceleration would require structured pruning, sparse tensor kernels, hardware-supported sparsity, or compiler/runtime support that skips zero-valued connections. Unstructured binary masks on dense `Linear` layers do not provide those benefits automatically.

## Ablation Study

The learned-vs-random ablation uses the same full-CIFAR-10 architecture, data split, seed-specific soft-gated checkpoint, global target sparsity, and one-epoch fixed-mask fine-tuning budget. The only selection difference is whether the lowest gate scores or a reproducible random permutation determines the pruned connections. The comparison uses `lambda = 0` for the soft-gated source checkpoint so that the ablation tests pruning selection rather than a different sparsity-training objective.

The full three-seed run was deliberately limited to the key 60% target because the full MLP is expensive on CPU. The implementation accepts arbitrary target levels; the existing final benchmark separately contains learned 20%, 40%, 60%, and 80% results, but this ablation run does not claim random results at those levels.

### Learned vs random pruning at 60%

| Seed | Dense test | Soft λ=0 test | Learned test | Random test | Learned advantage |
|---:|---:|---:|---:|---:|---:|
| 42 | 52.21% | 52.28% | 51.82% | 50.84% | +0.98 pp |
| 123 | 51.80% | 51.81% | 51.74% | 51.06% | +0.68 pp |
| 2024 | 52.01% | 51.69% | 51.46% | 50.50% | +0.96 pp |

All six pruned models have exactly 5,350,810 pruned and 3,567,206 active connections out of 8,918,016 (60.000004% measured logical sparsity). Across the three seeds, learned pruning reached 51.6733% ± 0.1890% test accuracy, versus 50.8000% ± 0.2821% for random pruning. The paired mean advantage was 0.8733 ± 0.1677 percentage points (sample standard deviation). This is descriptive evidence, not a statistical-significance claim.

### Ablation and fine-tuning recovery

The seed-42 positive-regularization soft ablation reached 52.27% test accuracy at `lambda = 0.001`, with no hard mask applied. For the multi-seed key comparison, the hard models were derived from the corresponding `lambda = 0` soft checkpoint.

| Seed | Learned before FT → after FT | Random before FT → after FT |
|---:|---:|---:|
| 42 | 49.80% → 51.82% (+2.02 pp) | 13.65% → 50.84% (+37.19 pp) |
| 123 | 46.90% → 51.74% (+4.84 pp) | 12.45% → 51.06% (+38.61 pp) |
| 2024 | 49.33% → 51.46% (+2.13 pp) | 11.80% → 50.50% (+38.70 pp) |

The corresponding accuracy drops from each seed's unpruned soft reference were 0.46, 0.07, and 0.23 pp for learned pruning, and 1.44, 0.75, and 1.19 pp for random pruning. Mean drops were 0.2533 ± 0.1960 pp and 1.1267 ± 0.3493 pp, respectively.

### Layer-wise and mask analysis

At seed 42 and 60% target sparsity, learned pruning removed 60.6345%, 58.8247%, 56.9231%, and 76.8555% of the connections in layers 1–4. Random pruning removed 60.0279%, 59.9349%, 59.9348%, and 59.1016%. This run therefore suggests that the learned ranking concentrated more pruning in the final layer; the observation is configuration-specific.

The overlap between learned-pruned and random-pruned connection sets was 59.9903% at seed 42, 59.9758% at seed 123, and 59.9881% at seed 2024. These overlap values are descriptive and should not be interpreted as an importance validation by themselves.

The source-of-truth ablation artifacts are in [artifacts/ablation_key_benchmark/reports](artifacts/ablation_key_benchmark/reports), including [ablation_results.csv](artifacts/ablation_key_benchmark/reports/ablation_results.csv), [multi_seed_results.csv](artifacts/ablation_key_benchmark/reports/multi_seed_results.csv), [learned_advantage.csv](artifacts/ablation_key_benchmark/reports/learned_advantage.csv), [layerwise_results.csv](artifacts/ablation_key_benchmark/reports/layerwise_results.csv), [mask_overlap.csv](artifacts/ablation_key_benchmark/reports/mask_overlap.csv), and [summary.json](artifacts/ablation_key_benchmark/reports/summary.json). The reproducible runner is [scripts/run_ablation.py](scripts/run_ablation.py).

## Testing

Run:

    pytest -q

The final repository test run passed 15 tests. Tests cover dense forward behavior, soft gate gradients, gate conversion, threshold masks, exact zeroing, fixed-mask enforcement, target sparsity, reproducible random and learned masks, connection/MAC accounting, checkpoint reload, and accuracy-drop calculation.


