# Results analysis

This document interprets the experiments that were **actually executed**. Every number
below is copied from a generated file; the file is named next to each table.

* **Part I** — the research-protocol MLP study (`configs/mlp.yaml`, `artifacts/research/mlp/`):
  30-epoch sources, five seeds, Network Slimming source, global allocation, ablations A and C.
* **Part II** — the earlier verification on the committed legacy 5-epoch checkpoints
  (`configs/mlp_legacy_sources.yaml`, `artifacts/research/mlp_legacy_sources/`).

* **Part III** — a **reduced-budget** VGG-16-BN / CIFAR-10 study on CPU
  (`configs/vgg_cifar10_cpu.yaml`, `artifacts/research/vgg16_cpu/`): 20-epoch sources, 3 FT
  epochs, 3 seeds, 9 methods, no ablations.

The full-protocol VGG-16-BN study (`configs/vgg_cifar10.yaml`: 160-epoch sources, 20 FT
epochs, 5 seeds, 16 methods, ablations) is implemented but was **not run**: the machine has no
GPU. Part III is not a substitute for it and is never reported as such.

# Part I — research-protocol MLP study

## I.0 What was executed

| Item | Value |
|---|---|
| Config | `configs/mlp.yaml` (resolved copy: `artifacts/research/mlp/experiment_config.yaml`) |
| Seeds | 42, 123, 2024, 7, 21 (pilot 42/123/2024 first, then extended); ablations A and C on the three pilot seeds |
| Data | CIFAR-10, fixed 45,000 / 5,000 train/validation split (`split_seed: 0`, same for every seed); test set used only for reporting |
| Sources (per seed, same initialization) | dense (CE), gated (existing objective, λ = 0, sum penalty), Slimming (CE + 1e-4·Σ\|γ\|); 30 epochs, AdamW lr 8e-4, cosine, label smoothing 0.05; best-validation epoch kept |
| Source test accuracy (mean ± SD, 5 seeds) | dense 60.06 ± 0.17, gated 60.14 ± 0.16, Slimming 59.51 ± 0.08 % (`results.csv`, `source_test_accuracy` of the control rows) |
| Criteria | random, L1/L2 in+out, activation, Taylor (dense and gated sources); gate-in (existing), gate-out (V1), gate-combined (V2), BN \|γ\| (gated source); Slimming \|γ\| on its own source, uniform and global allocation |
| Ratios | 20 / 40 / 60 % of hidden units |
| Fine-tuning | 5 epochs, AdamW lr 2e-4, cosine, identical for every method; best-validation checkpoint (incl. epoch 0) |
| Control | each unpruned source + the same 5-epoch fine-tuning |
| Rows | 525 in `results.csv` (435 main-study rows; the rest are ablation A/C rows) |
| Hardware / time | CPU only (torch 2.11.0+cpu, 20 threads); whole pipeline 23 h 08 min (01:40 → 00:48, `pipeline_steps.out`); seed 42 source training 13.4 min dense, 21.1 min gated (`seed42/sources/*/info.json`) |

Reproduce: `bash scripts/research/run_mlp_cpu_pipeline.sh python` (`research/RUNNING.md` §3).

## I.1 RQ1/RQ2: does the existing gate score identify prunable neurons?

**No — and with 30-epoch sources the deficit is larger than in the legacy check.**
Source: `analysis/findings.md`, `analysis/summary.csv`, `analysis/paired_tests.csv`.

Test accuracy after 5 fine-tuning epochs (%, mean ± SD over 5 seeds), criteria scoring the
gated source unless noted:

| Method | 20 % | 40 % | 60 % |
|---|---|---|---|
| **Gate-in (existing)** | **58.26 ± 0.24** | **56.29 ± 0.16** | **53.58 ± 0.28** |
| Random | 59.98 ± 0.24 | 59.22 ± 0.22 | 57.95 ± 0.07 |
| Gate-out (V1) | 60.16 ± 0.21 | 59.87 ± 0.21 | 58.72 ± 0.08 |
| Gate-combined (V2) | 60.26 ± 0.17 | 59.68 ± 0.28 | 58.66 ± 0.21 |
| L1 in+out | 60.39 ± 0.09 | 59.96 ± 0.26 | 58.82 ± 0.21 |
| L2 in+out | 60.26 ± 0.42 | 60.04 ± 0.14 | 58.63 ± 0.42 |
| Activation | 60.17 ± 0.40 | 59.60 ± 0.41 | 58.37 ± 0.18 |
| Taylor | 60.43 ± 0.24 | 59.98 ± 0.13 | 58.82 ± 0.33 |
| BN \|γ\| (diagnostic) | 60.39 ± 0.26 | 59.70 ± 0.39 | 58.83 ± 0.29 |
| Net. Slimming (own source, uniform) | 59.41 ± 0.23 | 59.27 ± 0.29 | 58.57 ± 0.19 |
| Net. Slimming (own source, global)¹ | 59.64 ± 0.27 | 59.57 ± 0.16 | 58.32 ± 0.16 |
| Unpruned gated + FT (control) | 60.46 ± 0.20 | | |

¹ Global allocation removes the same *number* of units but yields per-seed architectures
with larger parameter reductions (32.1 / 59.0 / 77.7 % instead of 24.6 / 47.0 / 67.0 %), so
it is not an equal-cost comparison.

Before fine-tuning (`pre_ft_test_accuracy_mean`) gate-in keeps 44.79 / 30.49 / 18.90 %
against random 57.43 / 46.89 / 24.03 % (gated source).

Paired comparison against gate-in (15 alternatives per ratio, Holm correction within ratio):

* **All 15 alternatives beat gate-in on 5/5 seeds at every ratio** (45/45 comparisons).
* Mean improvement: +1.16 to +2.18 pp at 20 %, +2.93 to +3.75 pp at 40 %, +4.37 to +5.34 pp at 60 %.
* Holm-adjusted paired t-test p < 0.05 for **15/15 at every ratio** (largest adjusted p per ratio: 0.0046, 0.00014, 0.000056).
* Random vs gate-in: +1.72, +2.93, +4.37 pp.
* The exact Wilcoxon test sits at its floor p = 0.0625 for every comparison (5/5 wins); with n = 5 it cannot go lower, so significance rests on the t-test. Shapiro–Wilk p < 0.05 for 4 of the 45 difference vectors (all at 20 %), where the t-test assumption is questionable; the 5/5 sign agreement holds regardless.

Interpretation is the same as in Part II: with BN after each gated layer the level of a
neuron's incoming gates does not affect the function (`novelty_audit.md`, O2). In layer 1
the gate-in score has a coefficient of variation of 9.2–9.6 × 10⁻⁴ on all five seeds
(gate-out 1.1 × 10⁻³, L1 0.11–0.12, Taylor 0.57–0.59; `seed*/reliability/scores.pt`), so
its ranking is set by tiny residual differences, which are anti-correlated with damage
(§I.3). Gate-out and gate-combined, the separately implemented variants, again reach the
level of the conventional criteria; the existing method's formulation is unchanged.

## I.2 Other comparisons (descriptive; not tested)

No hypothesis test was run between non-gate-in criteria. Descriptively:

* Taylor, L1, L2 and BN \|γ\| are within 0.35 pp of each other at every ratio; at 20 % Taylor (60.43) and L1 (60.39) are within 0.1 pp of the unpruned + FT control (60.46).
* Random is the lowest non-gate-in criterion at 40 % and 60 % (0.7–0.9 pp below the magnitude criteria); activation lies between random and the magnitude criteria.
* Dense vs gated source: the same criterion gives nearly the same accuracy (L1 at 60 %: 58.78 dense vs 58.82 gated; Taylor 58.92 vs 58.82). Gate training at λ = 0 did not make the network more prunable.
* **Network Slimming** preserves the most accuracy *before* fine-tuning at 40–60 % (55.43 / 44.75 % vs 53.26 / 39.34 % for Taylor on the gated source), but its source is 0.6 pp weaker (59.51 vs 60.14 %), and after 5 FT epochs it ends 0.25–1.0 pp below Taylor on the gated source. Slimming is evaluated on its *own* source, so source quality and criterion are confounded.

## I.3 RQ5: does the score predict single-unit removal damage?

Source: `analysis/reliability_summary.csv`, `seed*/reliability/{damage,correlations}.csv`
(64 units per hidden layer, 3 layers, 5 seeds; 960 removals per source).

Spearman ρ, mean over layers then mean ± SD over 5 seeds:

| Criterion | dense | gated | Slimming |
|---|---|---|---|
| L1 in+out | 0.068 ± 0.086 | 0.114 ± 0.068 | 0.269 ± 0.061 |
| L2 in+out | 0.044 ± 0.072 | 0.099 ± 0.077 | 0.257 ± 0.073 |
| BN \|γ\| | 0.035 ± 0.071 | 0.086 ± 0.064 | 0.321 ± 0.093 |
| Taylor | 0.106 ± 0.087 | 0.078 ± 0.037 | 0.263 ± 0.086 |
| Gate-out (V1) | – | 0.091 ± 0.080 | – |
| Gate-combined (V2) | – | 0.050 ± 0.064 | – |
| L1 in only | 0.054 ± 0.032 | 0.018 ± 0.067 | 0.219 ± 0.070 |
| Activation | −0.024 ± 0.032 | −0.066 ± 0.022 | 0.248 ± 0.091 |
| Random | −0.046 ± 0.026 | −0.043 ± 0.077 | −0.007 ± 0.095 |
| **Gate-in (existing)** | – | **−0.161 ± 0.088** | – |

* Gate-in is the lowest criterion, below random; its per-layer means are −0.084, −0.205 and −0.193 (layers 1–3).
* On the dense and gated sources every correlation is weak (|ρ| ≤ 0.17). Mean single-unit damage is 1.7 × 10⁻⁴ nats (SD 2.5 × 10⁻⁴); 23 % (dense) and 24 % (gated) of removals *decrease* the validation loss.
* On the **Slimming** source every non-random criterion correlates more strongly with damage (0.22–0.32), and fewer removals reduce the loss (20 %). A plausible reading is that the sparsity penalty concentrates importance in fewer units; this was not tested.
* Calibration stability (`stability_summary.csv`, 15 = 5 seeds × 3 layers): Spearman between scores from two disjoint calibration subsets is 0.977–0.987 for Taylor and 0.962–0.981 for activation. Weak reliability is not estimation noise.

## I.4 Ablation C: fine-tuning budget (FT 0/1/3 on 3 pilot seeds; FT 5 on 5 seeds)

Source: `analysis/findings.md` (section "Ablation C"), `research/paper/tables/mlp/ablation_c.tex`.

| Method | ratio | FT 0 | FT 1 | FT 3 | FT 5 |
|---|---|---|---|---|---|
| Gate-in (existing) | 20 % | 45.10 ± 0.40 | 56.38 ± 0.20 | 57.84 ± 0.47 | 58.26 ± 0.24 |
| | 40 % | 30.54 ± 0.44 | 53.05 ± 0.20 | 55.50 ± 0.26 | 56.29 ± 0.16 |
| | 60 % | 17.53 ± 1.41 | 48.55 ± 0.79 | 51.97 ± 0.49 | 53.58 ± 0.28 |
| Random | 20 % | 57.67 ± 0.43 | 59.17 ± 0.34 | 59.99 ± 0.34 | 59.98 ± 0.24 |
| | 40 % | 47.76 ± 2.02 | 57.83 ± 0.30 | 59.12 ± 0.26 | 59.22 ± 0.22 |
| | 60 % | 25.03 ± 1.36 | 56.23 ± 0.35 | 57.66 ± 0.27 | 57.95 ± 0.07 |
| L1 in+out | 60 % | 37.19 ± 1.41 | 56.92 ± 0.09 | 58.34 ± 0.23 | 58.82 ± 0.21 |
| Taylor | 60 % | 40.23 ± 2.91 | 57.27 ± 0.26 | 58.60 ± 0.19 | 58.82 ± 0.33 |
| Gate-combined (V2) | 60 % | 27.67 ± 0.73 | 56.77 ± 0.17 | 58.11 ± 0.08 | 58.66 ± 0.21 |
| Net. Slimming | 60 % | 44.62 ± 2.61 | 57.06 ± 0.17 | 58.46 ± 0.23 | 58.57 ± 0.19 |

(All ratios for all six methods are in the source file.)

* Fine-tuning shrinks but does **not** close the gate-in deficit. At 60 % the gap to random is 7.5 pp at 0 epochs, 7.7 at 1, 5.7 at 3 and 4.4 pp at 5 epochs.
* Every other method gains at most 0.55 pp from 3 to 5 epochs, so the 5-epoch budget is close to saturation for them.
* Without fine-tuning Slimming is the best method at 40 % and 60 % (tied with Taylor at 20 %: 58.67 vs 58.68), but not after 5 epochs. Method rankings depend on the fine-tuning budget, which is why the budget is identical for every method.

## I.5 Ablation A: gate penalty strength (3 pilot seeds, 40 % removed, FT 5 epochs)

Source: `analysis/ablation_a_summary.csv`, `analysis/ablation_a_gate_statistics.csv`,
`plots/fig_ablation_gate_lambda`. Each row is a separately trained gated source; the
existing method's formulation is unchanged, only its λ is varied.

| Source | source test acc | Gate-in | Gate-out | Gate-comb. | Random |
|---|---|---|---|---|---|
| λ = 0 (main gated source) | 60.25 ± 0.09 | **56.30 ± 0.14** | 59.84 ± 0.06 | 59.63 ± 0.38 | 59.27 ± 0.26 |
| λ = 1e-6 (sum) | 60.32 ± 0.11 | 59.85 ± 0.12 | 59.88 ± 0.41 | 59.76 ± 0.35 | 59.49 ± 0.30 |
| λ = 1e-5 (sum) | 60.09 ± 0.17 | 59.30 ± 0.19 | 59.35 ± 0.20 | 59.18 ± 0.09 | 58.68 ± 0.33 |
| λ = 1e-4 (sum) | 60.20 ± 0.20 | 58.31 ± 0.28 | 58.55 ± 0.40 | 58.33 ± 0.26 | 57.62 ± 0.09 |
| λ = 1e-3 (sum) | 60.05 ± 0.30 | 57.90 ± 0.25 | 58.04 ± 0.25 | 57.93 ± 0.14 | 57.52 ± 0.15 |
| λ = 0.1 (mean) | 60.23 ± 0.04 | 59.92 ± 0.20 | 59.91 ± 0.15 | 59.91 ± 0.17 | 58.94 ± 0.43 |
| λ = 1 (mean) | 60.28 ± 0.07 | 59.69 ± 0.04 | 59.81 ± 0.13 | 59.86 ± 0.18 | 59.09 ± 0.28 |

Findings (descriptive; n = 3, no tests run on ablation A):

* **The gate-in failure is specific to λ = 0.** With every positive penalty tested, gate-in is within 0.25 pp of gate-out and gate-combined and 0.36–0.98 pp above random.
* Source accuracy does not depend on λ (60.05–60.32 %), even at λ = 1e-3 (sum), where every connection gate has collapsed to σ(S) ≈ 0.087 (layer means 0.0873–0.0874, SD ≤ 2.5 × 10⁻⁴). BN absorbs the uniform scale change — a direct observation of the scale invariance (O2) and the uniform drift (O3).
* Larger *sum* penalties (1e-5 to 1e-3) lower post-pruning accuracy for **all** criteria, random included (random 59.49 → 57.52 %). The penalty makes the network harder to prune at equal FT budget rather than making the gates more informative. The mean-normalized penalties (0.1, 1) do not show this drop.
* At λ = 0 the gate-in neuron score has a within-layer SD of 1.6–8.2 × 10⁻⁴; at λ = 1e-6 (sum) it is 0.025–0.12. The penalty creates the spread the score needs, so λ = 0 — the setting of the committed checkpoints — is the worst case for the method's own score.

## I.6 RQ4: measured efficiency

Source: `benchmark/benchmark_results.csv` (CPU, 1 thread, median of 100 timed iterations).

| Model | Deployable params | MACs | bs=1 (ms) | bs=64 (ms) |
|---|---|---|---|---|
| Dense | 8,928,778 | 8,918,016 | 1.174 | 11.953 |
| 20 % (uniform) | 6,727,905 (−24.6 %) | 6,719,288 | 0.835 | 9.474 |
| 40 % (uniform) | 4,730,289 (−47.0 %) | 4,723,823 | 0.524 | 6.564 |
| 60 % (uniform) | 2,945,655 (−67.0 %) | 2,941,340 | 0.316 | 4.429 |
| Slimming global 60 % (5 per-seed architectures) | 1,921,055–2,049,129 | 1,916,740–2,044,814 | 0.252–0.369 | 2.509–2.705 |
| Gated training-time model | 8,928,778 (+8.9 M gate scores) | 8,918,016 | 12.975 | 23.935 |

* Reductions are physical (smaller dense layers). Latency falls monotonically at both batch sizes in this run; in Part II the 20 % model measured *slower* at bs = 1, so bs = 1 timings at this scale are fragile and we claim speed-ups only at bs = 64.
* The gated model as trained is about 11× slower at bs = 1 than the dense model; gates must be folded before deployment.
* No GPU measurements exist.

## I.7 Threats to validity (Part I)

1. **One architecture and dataset**: MLP on raw CIFAR-10 pixels (≈ 60 % test). The VGG study is required before generalizing to CNNs.
2. **Ablations A and C use three seeds** (except the FT 5 column) and are descriptive only.
3. **Slimming** is evaluated on its own, slightly weaker source; criterion and source are confounded. Global allocation changes the architecture and is not an equal-cost comparison.
4. **Statistics**: only the t-test can reach significance at n = 5, and a few difference vectors fail Shapiro–Wilk at 20 %.
5. **Single fixed split** for all seeds: seed varies initialization and data order, not the split.
6. **CPU latency** on one machine with one thread.

# Part II — legacy verification (5-epoch checkpoints)

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
| Not run in Part II | Slimming source, global allocation, ablations A and C, 5 seeds (all run in Part I); VGG-16 (not run anywhere) |

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

1. **Undertrained sources** (5 epochs, about 52 % test). Part I repeats the study with 30-epoch sources (≈ 60 %); the main conclusion holds.
2. **λ = 0 gated sources.** The gates were trained without the sparsity penalty, which is what the committed checkpoints provide. Ablation A (§I.5) shows the gate-in deficit disappears for every λ > 0 tested.
3. **One fine-tuning epoch.** Ablation C (§I.4) shows that up to 5 epochs shrink but do not close the gate-in gap.
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

# Part III — reduced-budget VGG-16-BN study on CPU

> **Reduced budget.** These are *not* results of the Stage B protocol. Every deviation from
> `configs/vgg_cifar10.yaml` is listed in §III.0 and in the header of
> `configs/vgg_cifar10_cpu.yaml`. Sources: `artifacts/research/vgg16_cpu/analysis/findings.md`,
> `summary.csv`, `paired_tests.csv`, `reliability_summary.csv`, `benchmark/benchmark_results.csv`.

## III.0 What was executed

| Item | Value |
|---|---|
| Config | `configs/vgg_cifar10_cpu.yaml` (extends `vgg_cifar10.yaml`; resolved copy `artifacts/research/vgg16_cpu/experiment_config.yaml`) |
| Model / data | VGG-16-BN, CIFAR-10, fixed 45k/5k/10k split (`split_seed 0`), standard augmentation — same as the full protocol |
| Sources (per seed) | dense (CE), gated (existing objective, **λ = 0**, sum penalty), Slimming (CE + 1e-4·Σ\|γ\|); SGD-Nesterov lr 0.05, wd 5e-4 (gate wd 0), cosine; best-validation epoch kept |
| **Deviations** | **20 source epochs (protocol 160); 3 FT epochs (protocol 20); seeds 42/123/2024 (protocol: five); 9 methods (protocol 16); 16 reliability units per layer on the gated and Slimming sources only (protocol 64, all sources); ablations A–E not run; no mixed precision** |
| Methods | random, L1, Taylor, gate-in (existing), gate-out (V1), gate-combined (V2), BN-\|γ\| (all on the gated source); L1 on the dense source; Network Slimming on its own source. Uniform allocation, so all methods produce identical compact architectures per ratio |
| Ratios | 20 / 40 / 60 % of channels removed in every conv layer |
| Fine-tuning | 3 epochs, SGD lr 0.01, cosine; best-validation epoch reported |
| Hardware | i7-12700H laptop CPU, 20 threads, torch 2.11.0+cpu; no GPU used |
| Rows | 90 in `results.csv` (3 seeds × (9 methods × 3 ratios + 3 unpruned controls)) |
| Wall-clock | per seed ≈ 9–10.5 h source training, 7.4–7.8 h comparison, 1.5–1.9 h reliability (`pipeline_steps.out`) |

Source test accuracy (`seed*/sources/*/info.json`, mean ± SD over 3 seeds): dense
89.55 ± 0.42, gated 90.41 ± 0.36, Slimming 89.76 ± 0.16 %. Eight of nine sources selected
their final epoch (one selected epoch 19), so the sources were still improving at 20 epochs.

## III.1 RQ1/RQ2: accuracy after pruning (3 FT epochs)

Test accuracy (%), mean ± SD over 3 seeds (`analysis/summary.csv`; paper table
`tables/vgg16_cpu/main_accuracy.tex`).

| Method | 20 % | 40 % | 60 % |
|---|---|---|---|
| **Gate-in (existing)** | **89.45 ± 0.42** | **87.32 ± 0.17** | **83.60 ± 0.84** |
| Taylor | 90.03 ± 0.35 | 88.94 ± 0.12 | 86.78 ± 0.13 |
| L1 in+out | 89.68 ± 0.22 | 88.86 ± 0.21 | 86.78 ± 0.45 |
| Gate-combined (V2) | 89.94 ± 0.28 | 88.84 ± 0.35 | 86.27 ± 0.10 |
| BN-\|γ\| | 89.67 ± 0.31 | 88.77 ± 0.24 | 86.40 ± 0.47 |
| Gate-out (V1) | 89.56 ± 0.52 | 88.42 ± 0.24 | 86.54 ± 0.21 |
| Network Slimming | 89.34 ± 0.20 | 88.54 ± 0.02 | 86.42 ± 0.23 |
| L1 on dense source | 89.15 ± 0.36 | 88.46 ± 0.24 | 86.63 ± 0.23 |
| Random | 89.31 ± 0.39 | 87.59 ± 0.43 | 85.02 ± 0.74 |
| Unpruned gated + FT | 90.41 ± 0.36 | | |

Parameter reduction 35.8 / 63.8 / 83.9 %, MAC reduction 35.4 / 63.5 / 83.6 % (the same for all
methods because allocation is uniform).

Paired comparisons against gate-in (`analysis/paired_tests.csv`; paired t-test, Holm within
ratio; the exact Wilcoxon test cannot go below p = 0.25 at n = 3):

* **20 %:** differences −0.30 … +0.58 pp; gate-in is statistically indistinguishable from every
  method, random included (all Holm p ≥ 0.39).
* **40 %:** every non-random method beats gate-in on 3/3 seeds, by +1.11 … +1.62 pp; only
  **Taylor** is supported after Holm (+1.62 pp, 95 % CI [1.28, 1.97], Holm p = 0.019).
  Random is +0.27 pp (2 wins, 1 loss, p = 0.49).
* **60 %:** every method beats gate-in on 3/3 seeds, by +2.67 … +3.18 pp for the non-random
  criteria and +1.42 pp for random. **None is supported after Holm** (smallest Holm p = 0.064,
  BN-\|γ\|; Taylor 0.112). Unadjusted t-test p < 0.05 for 7 of 8 comparisons.
* Before fine-tuning (`pre_ft_test_accuracy_mean`), gate-in drops the network to chance already
  at 20 % (10.33 %), while random keeps 25.35 %, gate-out 62.83 %, L1 64.88 % and Slimming
  67.46 %. At 40 % pre-FT accuracy ranges from 10.0 % (gate-in, random) to 32.6 % (L1 on the
  dense source); at 60 % all criteria are at 10–13 %.

**Reading.** The direction matches Part I — gate-in is last or second-to-last at every ratio
and below random at 40 % and 60 % — but with three seeds and three FT epochs the evidence is
much weaker: one of 24 comparisons survives Holm correction. The VGG numbers are consistent
with the MLP conclusion; they do not establish it independently for CNNs.

## III.2 The gate scores barely moved from initialisation

Gate statistics from `seed*/reliability/scores.pt` (all three seeds agree):

* Every gated layer's mean gate is σ(2.0) = 0.881 to three decimals. The within-layer
  coefficient of variation of the neuron-level gate-in score is **2.4 × 10⁻⁸ – 5.7 × 10⁻⁶
  (median ≈ 6 × 10⁻⁸)**; for gate-out it is 2.8 × 10⁻⁸ – 1.2 × 10⁻⁴ (median ≈ 2.4 × 10⁻⁷). For
  comparison, L1 has a median CV ≈ 0.07 and Taylor ≈ 0.4–0.5.
* With λ = 0, SGD and no gate weight decay, the gates moved only through the task gradient and
  stayed at their initial value to about 10⁻⁷ relative precision — much less movement than
  under AdamW in the MLP (layer-1 gate-in CV ≈ 10⁻³, §I.3). Gate-in rankings in VGG are
  therefore decided by displacements close to float32 resolution near 0.88 (≈ 6 × 10⁻⁸).
* These tiny displacements are nevertheless not neutral: gate-in pruning is worse than random
  (pre-FT 10.3 % vs 25.3 % at 20 %), while gate-out — equally small in spread — ranks channels
  about as well as L1. This repeats the MLP pattern (incoming-gate ranking adverse,
  outgoing-gate ranking informative); these runs do not identify the mechanism.
* Ablation A (gate penalty λ > 0) was **not** run on VGG. In the MLP every λ > 0 tested removed
  the gate-in deficit (§I.5); whether that holds for VGG is untested.

## III.3 RQ5: score vs single-unit removal damage

Spearman ρ between score and validation-loss increase after zeroing one channel
(16 random channels per layer, 13 conv layers; `analysis/reliability_summary.csv`; mean over
layers per seed, then mean ± SD over 3 seeds):

| Criterion | Gated source | Slimming source |
|---|---|---|
| Gate-combined | 0.195 ± 0.100 | — |
| Gate-out | 0.183 ± 0.121 | — |
| L1 in+out | 0.151 ± 0.093 | 0.167 ± 0.033 |
| L2 in+out | 0.149 ± 0.085 | 0.163 ± 0.028 |
| BN-\|γ\| | 0.103 ± 0.116 | 0.180 ± 0.052 |
| Taylor | 0.102 ± 0.039 | 0.190 ± 0.040 |
| Activation | 0.098 ± 0.079 | 0.168 ± 0.064 |
| **Gate-in** | **0.033 ± 0.053** | — |
| Random | −0.036 ± 0.031 | 0.083 ± 0.005 |

* All correlations are weak. Gate-in is the lowest non-random criterion and its mean is within
  one SD of zero.
* Per layer (mean over seeds, computed from `seed*/reliability/correlations.csv`), gate-in is
  strongly negative in the first conv layer (ρ = −0.67, where L1 is +0.83 and Taylor +0.91) and
  near zero elsewhere. In layers 8–12, L1 and Taylor are themselves negative (down to −0.75):
  the mean single-channel damage on the gated source is 2.2 × 10⁻³ nats (median 1.3 × 10⁻⁴),
  and **42 %** of single-channel removals (46 % on Slimming) *lower* the validation loss, so
  per-layer ρ in the deep layers is dominated by noise. With n = 16 per layer, |ρ| < 0.5 is not
  distinguishable from zero for an individual layer.
* Calibration-resampling stability (two disjoint calibration subsets): activation 0.977,
  Taylor 0.969 on the gated source; 0.989 / 0.981 on Slimming.

## III.4 RQ4: measured efficiency (CPU only)

`benchmark/benchmark_results.csv`; physically smaller compact models, CPU, median latency.

| Model | Params | MACs | Size | bs1 (ms) | bs64 (ms) |
|---|---|---|---|---|---|
| Dense | 14,724,042 | 313.2 M | 59.0 MB | 10.24 | 499.0 |
| 20 % removed | 9,447,740 | 202.4 M | 37.8 MB | 7.76 | 354.3 |
| 40 % removed | 5,332,682 | 114.2 M | 21.4 MB | 7.11 | 297.4 |
| 60 % removed | 2,369,066 | 51.3 M | 9.5 MB | 4.01 | 147.2 |
| Gated training model | 14,724,042 | 313.2 M | 59.0 MB | 34.18 | 579.0 |

MAC reductions of 35 / 64 / 84 % give measured CPU speed-ups of 1.32× / 1.44× / 2.56× at
batch 1 and 1.41× / 1.68× / 3.39× at batch 64 — well below the MAC ratio. The gated model,
which computes σ(S) on every forward pass, is 3.3× slower than dense at batch 1; it must be
folded (`to_plain`) before deployment. **No GPU latency or GPU memory was measured.**

## III.5 Threats to validity (Part III)

1. **Reduced budget.** 20-epoch sources (89.5–90.4 % test, still improving) and 3 FT epochs.
   Ablation C in the MLP showed that longer fine-tuning shrinks the gate-in gap; with the
   protocol's 20 FT epochs the VGG gap could be smaller.
2. **n = 3.** Exact Wilcoxon cannot reach p < 0.25; only the t-test can be significant, and its
   normality assumption cannot be checked at n = 3. One of 24 comparisons survives Holm.
3. **λ = 0 only.** The VGG gates barely moved from initialisation; a gated source trained with
   λ > 0 could behave differently (untested on VGG).
4. **9 of 16 methods**, uniform allocation only, no ablations, no GPU measurements.
5. **One dataset/architecture pair** (CIFAR-10 / VGG-16-BN).

## III.6 Reproduce

```bash
bash scripts/research/run_vgg_cpu_pipeline.sh python   # about 20 h per seed on a 20-thread CPU
```
