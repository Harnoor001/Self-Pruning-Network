# Results analysis

This document interprets the experiments that were **actually executed**. Every number
below is copied from a generated file; the file is named next to each table. Nothing here
comes from the research-protocol (`configs/mlp.yaml`) or VGG (`configs/vgg_cifar10.yaml`)
studies, which are implemented but were **not run** (CPU-only machine, no GPU).

## 0. What was executed

| Item | Value |
|---|---|
| Experiment | `mlp_legacy_sources` (config `configs/mlp_legacy_sources.yaml`) |
| Sources | committed legacy 5-epoch checkpoints, seeds 42 / 123 / 2024 (dense and gated, λ = 0), each evaluated on its **own** legacy train/validation split (reproduced exactly; `research_data.legacy_split_indices`) |
| Criteria | random, L1 in+out, L2 in+out, activation, Taylor (on both sources); gate-in (existing method), gate-out (V1), gate-combined (V2), BN \|γ\| (on the gated source) |
| Ratios | 20 / 40 / 60 % of hidden units per layer (uniform allocation → identical architectures) |
| Fine-tuning | 1 epoch, AdamW, identical for every method; best-validation checkpoint (incl. epoch 0) |
| Control | unpruned source + the same fine-tuning budget |
| Hardware | CPU only (torch 2.11.0+cpu, 20 threads for training, 1 thread for latency) |
| Rows | 132 (`artifacts/research/mlp_legacy_sources/results.csv`) |
| Not run | Network Slimming *source* (needs training; `bn_gamma@gated` is only a diagnostic), global allocation, ablations A–E, 5-seed extension, VGG-16 |

Re-generate everything with the four commands in `research/RUNNING.md` §2.

## 1. RQ1/RQ2: does the existing gate score identify prunable neurons?

**No, not in this setting.** Source: `analysis/findings.md`, `analysis/summary.csv`.

Test accuracy (%, mean ± SD over 3 seeds), all criteria scoring the gated source:

| Method | 20 % | 40 % | 60 % |
|---|---|---|---|
| Gate-in (existing) | 50.75 ± 0.31 | 49.17 ± 0.72 | 47.05 ± 0.39 |
| Random | 51.62 ± 0.36 | 51.23 ± 0.35 | 50.14 ± 0.30 |
| Gate-out (V1) | 51.97 ± 0.43 | 51.38 ± 0.36 | 50.73 ± 0.43 |
| Gate-combined (V2) | 51.85 ± 0.28 | 51.35 ± 0.31 | 50.57 ± 0.19 |
| L1 in+out | 51.74 ± 0.47 | 51.51 ± 0.47 | 50.82 ± 0.19 |
| L2 in+out | 51.82 ± 0.46 | 51.62 ± 0.23 | 50.70 ± 0.55 |
| Activation | 51.80 ± 0.35 | 51.06 ± 0.49 | 49.96 ± 0.21 |
| Taylor | 51.91 ± 0.37 | 51.49 ± 0.51 | 51.03 ± 0.11 |
| Unpruned gated + FT (control) | 52.09 ± 0.39 | | |

Before fine-tuning (same file, `pre_ft_test_accuracy_mean`) the gap is much larger:
gate-in 36.27 / 23.57 / 16.66 % against random 50.24 / 45.06 / 27.37 %.

Paired comparison against gate-in (`analysis/paired_tests.csv`; paired t-test, Holm
correction across the 13 alternatives within each ratio):

* All 13 alternatives, random selection included, beat gate-in on **every seed at every ratio** (39/39 comparisons, 3 of 3 seeds each).
* Mean improvement over gate-in: +0.83 to +1.24 pp at 20 %, +1.90 to +2.46 pp at 40 %, +2.81 to +4.01 pp at 60 %.
* Holm-adjusted p < 0.05: **10/13 at 20 %, 0/13 at 40 %, 13/13 at 60 %.** At 40 % every unadjusted p is ≤ 0.022, but none survive Holm correction with n = 3.
* Random vs gate-in: +0.87 pp (Holm p = 0.018), +2.06 pp (Holm p = 0.10), +3.09 pp (Holm p = 0.014).
* The exact Wilcoxon test cannot go below p = 0.25 with three pairs, so it carries no evidential weight here. The Shapiro–Wilk checks have essentially no power at n = 3.

Interpretation. The mean-incoming-gate score is worse than choosing neurons at random. This
is consistent with the analysis in `novelty_audit.md` (O2): with BatchNorm after each gated
layer, the overall level of a neuron's incoming gates does not affect the function, so the
score is not driven by function-relevant learning. In layer 1 of the gated sources the
gate-in scores have a coefficient of variation of 1.8e-4 on every seed. Gate-out is 3.5e-4,
L1 0.11 and Taylor 0.63–0.65 (`seed*/reliability/scores.pt`; seed 123 is plotted in
`fig_score_distribution`). The ranking is therefore set by
tiny residual differences. That ranking is *anti*-correlated with removal damage (§3). A
hypothesis we did **not** test: the residual differences may track something like
input-weight scale, which BN cancels.

The two separately implemented gate variants recover to the level of the conventional
criteria. Gate-out (V1) uses outgoing gates, which BN does not cancel; gate-combined (V2)
is described in `experimental_protocol.md`. These are **separate variants**; the existing
method's formulation is unchanged.

## 2. Ranking among the non-gate-in criteria

The study compares everything against the existing method, so **no hypothesis test was run
between the other criteria**. Their means after fine-tuning lie within about 1 pp of each
other at 20–40 %, with seed SDs of 0.1–0.55 pp. Descriptively:

* At 60 %, Taylor (51.03 gated / 51.05 dense) and L1 (50.82 / 50.86) have the highest means. Activation (49.96 / 49.85) and random (50.14 / 50.07) have the lowest.
* Before fine-tuning at 60 %, magnitude criteria preserve the most accuracy (L1 36.1 %). Taylor gives 33.8 % and random 27.4 %. Activation is lowest at 25.3 %, below random.
* One fine-tuning epoch closes most of the gap. At 20 % every criterion except gate-in ends within 0.5 pp of the unpruned + FT control (52.09 %).

## 3. RQ5: does the score predict single-unit removal damage?

Source: `analysis/reliability_summary.csv`, `seed*/reliability/{damage,correlations}.csv`.
The setup samples 64 units per hidden layer, zeroes each unit's output in eval mode
(exactly equivalent to removing it), and measures the increase of validation loss.

Spearman ρ, mean over the 3 layers, then mean ± SD over seeds:

| Criterion | dense source | gated source |
|---|---|---|
| L1 in+out | 0.182 ± 0.023 | 0.169 ± 0.029 |
| L2 in+out | 0.170 ± 0.024 | 0.161 ± 0.030 |
| Gate-combined (V2) | – | 0.163 ± 0.080 |
| Gate-out (V1) | – | 0.158 ± 0.068 |
| BN \|γ\| | 0.153 ± 0.054 | 0.131 ± 0.046 |
| Taylor | 0.074 ± 0.062 | 0.032 ± 0.021 |
| L1 in only | 0.051 ± 0.092 | 0.023 ± 0.103 |
| Random | −0.037 ± 0.032 | −0.030 ± 0.023 |
| **Gate-in (existing)** | – | **−0.119 ± 0.039** |
| Activation | −0.142 ± 0.084 | −0.163 ± 0.049 |

Findings:

* **Every correlation is weak** (|ρ| ≤ 0.18). The main reason is visible in `damage.csv`: removing one unit from a 512–2048-wide layer changes validation loss by 1e-4 nats on average (SD 2.4e-4), and 27 % of the 1,152 measured removals *decrease* the loss. Single-unit damage is therefore a low-signal target in these wide, undertrained MLPs.
* Gate-in is **negatively** correlated with damage on every seed. Its per-layer means are 0.018, −0.156 and −0.221 for layers 1 to 3. So neurons with higher gate-in scores are, if anything, *less* important. This matches gate-in's below-random pruning accuracy.
* Activation magnitude is also negatively correlated, and it is the second-worst criterion before fine-tuning at 40–60 %.
* Taylor ranks poorly on single-unit damage (ρ ≈ 0.03–0.07) yet gives the best post-FT accuracy at 60 %. Single-unit damage is not additive over the large groups removed in practice, so RQ5 and RQ1 measure different things. We report both and do not claim either is the "true" importance.
* Calibration stability (`stability_summary.csv`): scores from two disjoint 1,024-image calibration subsets agree closely. Spearman is 0.987 ± 0.007 for Taylor and 0.971 ± 0.023 for activation (9 = 3 seeds × 3 layers per source). Their weak reliability is therefore not noise in the score estimate.

## 4. RQ3: stability across seeds

Seed-to-seed SDs of post-FT accuracy are 0.1–0.7 pp (table above; per-seed points in
`fig_multiseed_r{20,40,60}`). The gate-in deficit is consistent in sign across all seeds.
The dense-vs-gated source comparison is a descriptive check, not tested: the same criterion
gives nearly identical accuracy on either source (e.g. L1 at 60 %: 50.86 vs 50.82; Taylor
51.05 vs 51.03). The two unpruned controls are 52.04 and 52.09. At λ = 0 and 5 epochs,
training with gates therefore did not make the network more prunable.

## 5. RQ4: measured efficiency

Source: `benchmark/benchmark_results.csv`. CPU, 1 thread, 20 warm-up and 100 timed
iterations; median shown. **These are CPU measurements only; no GPU numbers exist.**

| Model | Deployable params | MACs | Size (bytes) | bs=1 (ms) | bs=64 (ms) |
|---|---|---|---|---|---|
| Dense (unpruned) | 8,928,778 | 8,918,016 | 35,750,686 | 1.174 | 11.540 |
| 20 % units removed | 6,727,905 (−24.6 %) | 6,719,288 | 26,941,406 | 1.419 | 9.293 |
| 40 % units removed | 4,730,289 (−47.0 %) | 4,723,823 | 18,945,118 | 0.640 | 6.417 |
| 60 % units removed | 2,945,655 (−67.0 %) | 2,941,340 | 11,800,862 | 0.563 | 3.582 |
| Gated training-time model (W·σ(S)) | 8,928,778 (+8.9 M gate scores) | 8,918,016 | 35,750,750 | 12.077 | 22.657 |

* Parameter, MAC and size reductions are **physical**: the compact models are smaller dense networks. They are identical for every criterion because uniform allocation fixes the architecture.
* Batch-64 latency falls monotonically: 11.5 → 9.3 → 6.4 → 3.6 ms. Batch-1 latency is **not monotone**: 20 % pruning measured slower than dense (1.42 vs 1.17 ms; IQRs 1.29–1.63 vs 1.15–1.20). At this size bs=1 timing is dominated by overhead and kernel selection, so we make no bs=1 speed-up claim at 20 %.
* The gated model *as trained* is about 10× slower at bs=1 than the equivalent dense model, because it recomputes W·σ(S) every forward pass. Gates must be folded before deployment.

## 6. Audit result reproduced

The legacy compact reconstruction drops the gate factor (INITIAL_AUDIT finding 1). Every
experiment here uses `to_plain`, which folds gates before slicing. The control rows confirm
that fold-then-compact at 0 % reproduces the source exactly: for every seed,
`control@gated` has `pre_ft_validation_accuracy == source_validation_accuracy` and
`pre_ft_test_accuracy == source_test_accuracy` (seed 42: 51.74 / 52.28 %; seed 123:
50.70 / 51.81 %; seed 2024: 50.40 / 51.69 %; `results.csv`).

## 7. Threats to validity

1. **Undertrained sources** (5 epochs, about 52 % test). The research protocol trains for 30 epochs; conclusions may change.
2. **λ = 0 gated sources.** The gates were trained without the sparsity penalty, which is what the committed checkpoints provide. With λ > 0, uniform drift (O3) makes gate-in rankings even less informative, but this was not measured here (ablation A not run).
3. **One fine-tuning epoch.** Longer fine-tuning may close the remaining gaps further (ablation C not run).
4. **n = 3 seeds.** Only the Holm-corrected t-test can reach significance, and it assumes normal paired differences, which cannot be checked meaningfully at n = 3.
5. **Single architecture.** MLP on raw pixels only. The VGG study is required before generalising to CNNs.
6. **Legacy splits differ per seed.** This is necessary to avoid evaluating on images the source was trained on. Seed therefore confounds initialisation and split, as in the original repository.

## 8. Reproduce

```bash
python scripts/research/run_comparison.py         --config configs/mlp_legacy_sources.yaml
python scripts/research/run_importance_analysis.py --config configs/mlp_legacy_sources.yaml
python scripts/research/benchmark.py               --config configs/mlp_legacy_sources.yaml
python scripts/research/analyze_results.py         --config configs/mlp_legacy_sources.yaml
```

Run time on this machine (CPU): about 85 min for the comparison (132 rows), 11 min for
reliability, under 1 min for the benchmark.
