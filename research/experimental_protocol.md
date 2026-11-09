# Experimental Protocol

Version 1.0 · fixed before running any research experiment. Changes after results are
seen must be logged in §11 with the reason.

## 1. Research questions → experiments

| RQ | Experiment | Script | Primary output |
|---|---|---|---|
| RQ1 gate vs conventional scoring | Structured comparison, all criteria on the **same** gated source | `scripts/research/run_comparison.py` | test acc. after FT per method × ratio × seed |
| RQ2 vs magnitude/activation/Taylor/Slimming | same + dense source + Slimming source | same | same |
| RQ3 seed stability | 5 seeds; mask overlap; calibration-resampling stability | `run_comparison.py`, `run_importance_analysis.py` | sd / CI, Jaccard, Spearman |
| RQ4 efficiency | compact model metrics | `run_comparison.py`, `benchmark.py` | params, MACs, deployable size, CPU/GPU latency, peak memory |
| RQ5 score reliability | single-unit removal damage vs score | `run_importance_analysis.py` | Spearman ρ per layer/method/seed |
| Ablations A–E | `run_ablation.py` | | |

## 2. Data

* CIFAR-10, 50 000 train / 10 000 test images.
* **Fixed partition:** 45 000 train / 5 000 validation from `torch.randperm(50000)` with
  `split_seed = 0`, identical for all seeds, methods and architectures. (The legacy loader
  used a seed-dependent split; see audit finding 8.)
* Normalisation: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616).
* Training augmentation — MLP: legacy pipeline (RandomCrop 32 pad 4, horizontal flip,
  ColorJitter 0.15/0.15/0.15) to stay comparable with the legacy artifacts. VGG:
  RandomCrop 32 pad 4 + horizontal flip (standard CIFAR CNN augmentation).
* Validation, test and calibration data use no augmentation.
* **Calibration set** (activation, Taylor): first 2 048 indices of the fixed *training*
  partition, evaluation transform, model in `eval()` mode.
* **Test set is used only for final reporting.** Checkpoint selection, λ selection and all
  importance-reliability measurements use validation data.

## 3. Architectures

* **Stage A — MLP** 3072→2048→1024→512→10, BN1d + GELU + Dropout(0.3) after each hidden
  layer (identical to the legacy `SelfPruningMLP`). Prunable units: 3 584 hidden neurons.
* **Stage B — VGG-16-BN for CIFAR**: cfg
  `[64,64,M,128,128,M,256,256,256,M,512,512,512,M,512,512,512,M]`, conv3×3 (no bias) +
  BN2d + ReLU, five max-pools → 512×1×1, classifier `Linear(512,10)`. Prunable units: all
  4 224 channels of the 13 conv layers. The classifier output is never pruned.
* Gated variants replace every `Linear`/`Conv2d` with a gated layer `W ⊙ σ(S)`, `S` init 2.0.

## 4. Source models (trained once per seed and cached)

| Source | Objective | Used by |
|---|---|---|
| `dense` | CE (label smoothing 0.05) | dense-source variants of post-training criteria; *dense + FT* control |
| `gated` | CE + λ·P(σ(S)) (existing method) | Method 6 and **all post-training criteria in the primary comparison** |
| `slimming` | CE + λ_s·Σ|γ| over BN weights | Method 7 |

Training hyper-parameters (`configs/mlp.yaml`, `configs/vgg_cifar10.yaml`):

| | MLP | VGG-16-BN |
|---|---|---|
| Optimiser | AdamW lr 8e-4, wd 1e-4 | SGD momentum 0.9, nesterov, lr 0.05, wd 5e-4 |
| Schedule | cosine, 30 epochs | cosine, 160 epochs |
| Batch | 128 | 128 |
| Gate wd | 1e-4 (legacy: AdamW over all params) | 0 (documented deviation: SGD wd 5e-4 would dominate gate dynamics) |
| Gate λ, penalty | 0 (primary, matches legacy structured study) and ablation values; **sum** (legacy) | same |
| Slimming λ_s | 1e-4 (Liu et al. 2017) | 1e-4 |
| AMP | on CUDA only (bf16/fp16 autocast + GradScaler) | same |
| Selection | best validation accuracy epoch | same |

The legacy 5-epoch MLP configuration is reproduced separately (`configs/mlp_legacy.yaml`)
only to check agreement with the legacy artifacts (Phase 3); it is not the main result.

**Training-overhead disclosure:** gated models have 2× the parameters during training and
Slimming has an extra penalty; the gated and Slimming sources are separate training runs.
Wall-clock training time is recorded for every source.

## 5. Methods (common interface `self_pruning_network.importance`)

Notation: hidden unit *j* of group *l*; producer weight `W_l` (rows/out-channels = units),
consumer weight `W_{l+1}` (columns/in-channels = units); for convolutions norms are over
all kernel elements; `a_j` = unit output after the non-linearity (the exact quantity
removed when the unit is pruned); all weights are **effective** weights (gates folded).

| # | Name | Score `I_j` |
|---|---|---|
| 0 | dense | no pruning |
| 1 | `random` | `U(0,1)` from a generator seeded with the run seed |
| 2 | `l1` | `‖W_l[j]‖₁ / mean_k ‖W_l[k]‖₁ + ‖W_{l+1}[:,j]‖₁ / mean_k ‖W_{l+1}[:,k]‖₁` |
| 3 | `l2` | same with ‖·‖₂ |
| 4 | `activation` | `E_x mean_spatial |a_j(x)|` over the calibration set |
| 5 | `taylor` | `E_x | Σ_spatial a_j(x) · ∂L(x)/∂a_j(x) |`, per-sample CE loss (sum reduction), eval mode |
| 6 | `gate_in` (**existing method**) | `mean_i σ(S_l[j,i])` (mean over incoming row / `in×k×k`) |
| 7 | `slimming` | `|γ_j|` of the BN after producer *l*, on the Slimming source |
| — | `bn_gamma` | `|γ_j|` on the dense/gated source (diagnostic) |
| V1 | `gate_out` (variant) | `mean_k σ(S_{l+1}[k,j])` (mean over outgoing column) |
| V2 | `gate_combined` (variant) | `gate_in_j/mean(gate_in) + gate_out_j/mean(gate_out)` |

Aggregation/normalisation decisions: incoming and outgoing terms are each divided by their
layer mean so both contribute equally and the score is scale-free within a layer; scores
are only compared *within* a layer under the uniform allocation, so no cross-layer
normalisation is needed.

## 6. Pruning procedure

* **Allocation (primary): uniform.** Each prunable group with `n` units keeps
  `k = max(min_keep, n − ⌊n·r⌋)` units, `r ∈ {0.2, 0.4, 0.6}`, `min_keep = 1` (MLP) / 8 (VGG).
  Every criterion therefore yields **identical compact architectures** at a given `r`.
* Selection: top-`k` by score, `torch.argsort(stable=True, descending=True)` (ties → lower
  index), surviving indices kept in ascending order.
* Secondary allocation `global` (Slimming's original rule, reported separately as
  `slimming_global`): single threshold across layers subject to `min_keep`, with the
  total number of removed units equal to the uniform budget.
* Compact reconstruction slices producer rows, bias, BN γ/β/running stats, and consumer
  columns; gates are folded first. Verified by tests to be function-preserving at `r=0`
  and exactly equivalent to zeroing the removed unit's activation (eval mode).

## 7. Fine-tuning (identical for every method)

* MLP: AdamW lr 2e-4, wd 1e-4; VGG: SGD m=0.9 lr 0.01, wd 5e-4; cosine over the FT epochs.
* Primary FT budget: **5 epochs** (MLP), **20 epochs** (VGG). Ablation C: 0/1/3/5 epochs.
* Same loader seed for all methods within a seed → identical data order.
* Checkpoint selection by validation accuracy (epoch 0 = before FT is a candidate).
* **Dense + FT control:** the unpruned source fine-tuned with the same policy, so pruning
  is not credited with gains from extra training (audit finding 6).

## 8. Seeds

Final: `42, 123, 2024, 7, 21`. Pilot: first three. The seed controls initialisation,
data order, dropout, random-pruning scores and unit sampling; **not** the data split.

## 9. Metrics

Classification: test acc. (final only), val acc., val loss (CE without smoothing),
degradation vs the source and vs *dense + FT*.
Compression: deployable parameters (weights + biases + BN affine; gate scores and masks
excluded), parameter reduction %, deployable state-dict size in bytes (serialised
in-memory; training-only tensors excluded), physical unit removal %, architecture widths.
Compute: MACs per image counted by forward hooks on `Linear`/`Conv2d` of the instantiated
model (BN/activation/pool ignored, documented), CPU latency (1 thread, batch 1 and 64,
20 warm-up, 100 timed, median and IQR), GPU latency (CUDA events, synchronised, batch 1
and 256), peak GPU memory (`max_memory_allocated`), training/pruning/fine-tuning wall time.
**No theoretical MAC reduction is reported as speedup; latency is reported only as measured.**

## 10. Statistical analysis (`scripts/research/analyze_results.py`)

* Per method × ratio: mean, sample SD, 95 % t-interval (n = number of seeds).
* Paired comparisons on matched seeds: each criterion vs `gate_in` and `gate_in` vs
  `random`; paired t-test **and** exact Wilcoxon signed-rank test; Holm correction over the
  family of comparisons within each architecture × ratio. Mean paired difference with
  95 % t-interval is always reported.
* **Power caveat:** with n = 5 the exact two-sided Wilcoxon test cannot reach p < 0.0625;
  the t-test assumes approximately normal paired differences (checked descriptively).
  "Statistically significant" is used only for Holm-adjusted p < 0.05 of the paired t-test,
  stating the assumption; otherwise results are described as descriptive.
* Reliability (RQ5): per layer Spearman ρ between score and measured damage on 64 randomly
  sampled units per layer (all units for layers with ≤ 64 units); summarised by the mean
  over layers; seeds summarised by mean ± SD.

## 11. Deviations log

| Date | Change | Reason |
|---|---|---|
| — | — | — |

## 12. Baseline comparison matrix

| Method | Needs data | Needs gradients | Extra training run | Uses gates | Source checkpoint | Notes |
|---|:-:|:-:|:-:|:-:|---|---|
| Dense | – | – | – | – | dense | reference |
| Dense + FT | yes | yes | – | – | dense | budget control |
| Random | – | – | – | – | gated (+dense) | lower bound |
| L1 in+out | – | – | – | – | gated (+dense) | folded weights |
| L2 in+out | – | – | – | – | gated (+dense) | folded weights |
| Activation | calib. | – | – | – | gated (+dense) | 2 048 train images |
| Taylor | calib. | yes | – | – | gated (+dense) | eval-mode BN |
| **Gate-in (existing)** | – | – | yes (gated) | yes | gated | principal method |
| Gate-out (V1) | – | – | yes (gated) | yes | gated | variant |
| Gate-combined (V2) | – | – | yes (gated) | yes | gated | variant |
| BN-γ | – | – | – | – | gated (+dense) | diagnostic |
| Network Slimming | – | – | yes (slimming) | – | slimming | per-layer & global |

## 13. Resource estimates (per seed, single modern GPU, AMP)

| Item | MLP | VGG-16-BN |
|---|---|---|
| Source training (3 sources) | 3 × 30 ep ≈ 10 min | 3 × 160 ep ≈ 1.5–2 h |
| Comparison (≈ 11 methods × 3 ratios × FT + dense+FT) | ≈ 34 × 5 ep ≈ 15 min | ≈ 34 × 20 ep ≈ 2.5 h |
| Reliability (≈ 10 scores, 64 units × groups val passes) | ≈ 3 min | ≈ 20 min |
| Checkpoints | ≈ 36 MB dense, ≤ 72 MB gated, compact ≤ 27 MB | ≈ 60 MB per source; compact ≤ 60 MB |

Totals for 5 seeds: MLP ≈ 2.5 h, VGG ≈ 20–25 h, storage ≈ 15 GB if every compact
checkpoint is kept (`save_compact_checkpoints: false` reduces this to sources only).
