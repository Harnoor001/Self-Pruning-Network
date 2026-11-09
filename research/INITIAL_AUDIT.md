# Initial Repository Audit — Self-Pruning Network

Audit date: 2026-11-09 · Branch: `research/comparative-pruning-study` · Audited commit: `9a51231` (`main`)

This audit was written **before** any change to the existing implementation. Every
quantitative statement below is either read from a committed artifact (path given) or
was re-measured during the audit with the exact command recorded in
[Appendix A](#appendix-a-audit-measurements).

---

## 1. What the repository contains

| Area | Files | Notes |
|---|---|---|
| Core model | `src/self_pruning_network/model.py` (599 lines) | `PrunableLinear`, `SelfPruningMLP`, `StructuredPrunedMLP`, parameter counters |
| Training | `src/self_pruning_network/train.py` (404 lines) | dense / soft / hard pipeline, `fine_tune`, checkpoint I/O, CLI `main()` |
| Data | `src/self_pruning_network/data.py` | CIFAR-10 loaders, 90/10 train/val split, augmentation |
| Reporting | `src/self_pruning_network/reporting.py` | matplotlib plots, markdown tables |
| Scripts | `scripts/train_and_report.py`, `benchmark.py`, `run_ablation.py`, `run_structured_pruning.py`, `run_structured_recovery.py`, `run_structured_multiseed.py` | each writes into `artifacts/<experiment>/` |
| Serving | `app/api.py` | FastAPI inference UI; loads soft/hard/structured checkpoints |
| Tests | `tests/test_prunable_layer.py` | 20 collected tests, all passing (see §6) |
| Artifacts | `artifacts/{final_benchmark,efficiency_benchmark,ablation_key_benchmark,structured_pruning,structured_recovery,structured_multiseed,...}` | CSV/JSON/MD/PNG reports are versioned; checkpoints are git-ignored but present locally |
| Docs | `README.md`, `docs/reports/ASSESSMENT_REPORT.md`, root `ASSESSMENT_REPORT.md` (untracked, differs from the docs copy) | |

Untracked local directories at audit time (left untouched): `ASSESSMENT_REPORT.md`,
`artifacts/ablation_smoke/`, `artifacts/audit_smoke/`, `artifacts/pruning_realistic/`,
`artifacts/pruning_smoke/`.

## 2. Existing architecture

```
input 3×32×32 ──flatten──► 3072
   │
   ├─ PrunableLinear(3072→2048)  W1∈R^{2048×3072}, S1∈R^{2048×3072}, b1
   ├─ BatchNorm1d(2048) ─ GELU ─ Dropout(0.3)
   ├─ PrunableLinear(2048→1024)  W2∈R^{1024×2048}, S2 same shape, b2
   ├─ BatchNorm1d(1024) ─ GELU ─ Dropout(0.3)
   ├─ PrunableLinear(1024→512)   W3∈R^{512×1024},  S3 same shape, b3
   ├─ BatchNorm1d(512)  ─ GELU ─ Dropout(0.3)
   └─ PrunableLinear(512→10)     W4∈R^{10×512},    S4 same shape, b4
                                                     ▼ logits
```

* Connections (weight entries): 8,918,016. Gate scores: another 8,918,016 (training-only).
* Deployable parameters (weights + biases + BN affine): 8,928,778 (`deployable_parameter_count`).
* `nn.Sequential` indices: Linear 0/4/8/12, BN 1/5/9, GELU 2/6/10, Dropout 3/7/11.
  The state dict therefore contains every gated tensor twice (`network.N.*` and
  `prunable_layers.k.*`); storage is shared by `torch.save`, so the file is not doubled.

## 3. Existing methodology (as implemented, not as described)

**Gate parameterisation.** `PrunableLinear.forward` uses `W_eff = W ⊙ σ(S)` in *soft*
mode, `W ⊙ M` in *hard* mode, and `W` in *dense* mode. `S` is initialised to the
constant 2.0, so every gate starts at σ(2) ≈ 0.8808. Bias is **not** gated.

**Objective.** `L = CE(label_smoothing=0.05) + λ · Σ_l Σ_ij σ(S_l,ij)`. The penalty is an
**unnormalised sum over all 8.9 M gates** (not a mean). The optimiser is AdamW
(lr 8e-4, weight decay 1e-4) over *all* parameters, including the gate scores, i.e. the
gate scores also receive decoupled weight decay.

**Unstructured pruning.** `hard_prune_target_sparsity` ranks **all** gates of all four
layers globally (`torch.topk`) and keeps the top (1−p) fraction; `apply_mask` zeroes `W`
and switches to hard mode; `fine_tune` re-applies the mask after every optimiser step.

**Structured pruning (the method under evaluation in this study).**
`StructuredPrunedMLP.from_self_pruning` computes, for hidden layer *l* with weight
`W_l ∈ R^{n_out×n_in}` (PyTorch `Linear` layout, rows = output neurons),

```
importance_j = mean_i σ(S_l[j, i])      (mean over the INCOMING connection row of hidden neuron j)
```

i.e. `layer.gates().mean(dim=1)`. It then removes `floor(n_out · p)` neurons per hidden
layer (same ratio in every layer, at least one kept), keeps surviving indices in sorted
order, slices rows of `W_l`, `b_l`, BN parameters/buffers, and the matching **columns** of
`W_{l+1}`. The output layer is never pruned.

## 4. Existing results (read from artifacts)

| Source | Result |
|---|---|
| `artifacts/final_benchmark/reports/results.csv` | Dense 52.21 %, soft λ=0 52.28 %, soft λ=1e-3 52.27 % test acc. (seed 42, 5 epochs) |
| same | Unstructured hard 20/40/60/80 %: 52.08 / 51.87 / 51.84 / 49.11 % after 1 FT epoch |
| `artifacts/structured_pruning/reports/structured_results.csv` | Structured 20/40/60 %: 52.89 / 52.72 / 52.05 % after 1 FT epoch; pre-FT 37.89 / 28.02 / 21.75 % |
| `artifacts/structured_recovery/reports/recovery_results.csv` | Same targets, 1 FT epoch: **50.54 / 48.42 / 47.29 %** |
| `artifacts/structured_multiseed/reports/multi_seed_summary.csv` | Structured 60 %, 3 seeds, 5 FT epochs: 51.25 ± 0.73 % |
| `artifacts/ablation_key_benchmark/reports/learned_advantage_summary.csv` | Unstructured 60 %: learned − random = +0.87 ± 0.17 pp (3 seeds) |

## 5. Findings

Severity: **[C]** critical for the research question, **[M]** methodological, **[m]** minor.

### 5.1 Implementation bugs

1. **[C] Compact reconstruction drops the gate factor.** `from_self_pruning` copies the
   raw `W` instead of the effective soft-mode weight `W ⊙ σ(S)`. The compact model is
   therefore *not* the same function as the source even at 0 % pruning. Measured on the
   seed-42 soft λ=0 checkpoint: source validation accuracy **51.74 %**, compact-at-0 %
   **50.70 %**, compact-at-0 % with folded gates **51.74 %** (exact match). The BN
   running statistics were estimated for `W ⊙ σ(S)·x`, so the un-folded weights (≈ 1/0.88
   larger) shift every pre-activation. All legacy "pre-fine-tuning" structured
   accuracies (37.89 / 28.02 / 21.75 %) contain this error in addition to the effect of
   neuron removal. The existing test
   `test_structured_pruning_compacts_dimensions_and_transfers_weights` encodes the raw-copy
   behaviour, so the bug was locked in by the test.
   *Resolution:* the legacy function is kept bit-for-bit (to reproduce legacy artifacts)
   behind a new opt-in `fold_gates=False` keyword; the new research framework always
   folds gates. Both behaviours are tested.

2. **[m] Duplicate test names.** `tests/test_prunable_layer.py` defines
   `test_gate_conversion_and_threshold_mask` and
   `test_hard_pruning_zeroes_weights_and_survives_optimizer_step` twice each; Python keeps
   only the second definition, so two tests silently never run. (Kept as-is to avoid
   modifying legacy tests; noted for the author.)

3. **[m] Soft-model "sparsity" is always 0 %.** `gate_summary` counts the binary `mask`,
   which is all-ones in soft mode, so `lambda_vs_sparsity.png` and the
   `sparsity_percent` column of soft rows carry no information about gate behaviour.

### 5.2 Methodological problems relevant to RQ1–RQ5

4. **[C] The learned gates are almost uninformative at the neuron level.** Gate statistics
   of the committed checkpoints (Appendix A):

   | Checkpoint | gate mean | gate std | std of row-mean (neuron score), layer 1 | row-mean range, layer 1 |
   |---|---:|---:|---:|---:|
   | soft λ = 0 | 0.8805 | 0.0020 | 1.6 × 10⁻⁴ | 0.8798 – 0.8808 |
   | soft λ = 1e-3 | 0.7423 | < 10⁻⁴ | < 10⁻⁵ | 0.7423 – 0.7423 |
   | dense (gates frozen) | 0.8808 | 0 | 0 | constant |

   Neuron scores used for structured pruning therefore differ in the 4th decimal place.
   With λ = 1e-3 every gate moved by the *same* amount: the penalty gradient
   `λ σ'(S)` is identical for all gates and dominates the CE gradient, and Adam's
   per-coordinate normalisation converts a dominant constant gradient into an identical
   step of ≈ lr per iteration (2.0 − 1 760 steps × 8e-4 → S ≈ 1.06, σ ≈ 0.742, matching
   the observed 0.7423). **Under Adam the unnormalised L1 gate penalty acts as a uniform
   drift, not as a discriminative signal.** This is a mechanistic explanation that must
   be tested (ablation A), not assumed.

5. **[C] The mean-incoming-gate score is BatchNorm-invariant in this architecture.** Each
   gated hidden `Linear` is followed directly by `BatchNorm1d`. Scaling all incoming gates
   of neuron *j* by a constant `c > 0` scales `W_j·x` by `c`; BN subtracts the batch mean
   and divides by the batch standard deviation, so the normalised activation is unchanged
   (bias is not gated, but BN removes it with the mean). Consequently the *level* of the
   incoming row mean — exactly what `importance_j` measures — has no direct influence on
   the network function during training, and therefore receives no consistent gradient
   signal. Only the relative pattern of gates within a row matters. This is a structural
   reason to expect the existing neuron score to be weak, and it motivates the separately
   reported **outgoing-gate** variant (the outgoing gates scale neuron *j*'s contribution to
   the next pre-BN sum and are *not* cancelled by BN). The same argument applies to
   incoming-weight-norm criteria (L1/L2 of `W_l[j,:]`), which is a known weakness of
   magnitude criteria in BN networks [Liu et al., 2017].

6. **[M] Confounded training budget.** The dense reference is trained for only 5 epochs and
   is far from converged. Pruned models then receive 1–5 additional fine-tuning epochs, so
   "structured 20 % beats dense (52.89 % vs 52.21 %)" can be explained by extra training
   alone. A fair protocol needs a *dense + same fine-tuning budget* control.

7. **[M] Large run-to-run variation of single fine-tuning runs.** The same nominal
   configuration (seed 42, structured 20 %, 1 FT epoch, same lr) produced 52.89 %
   (`structured_pruning`) and 50.54 % (`structured_recovery`), a 2.35 pp spread, because the
   two scripts consume RNG differently (data order, dropout masks) and the first script
   used `fine_tune` with validation-best selection while the second used a hand-written
   loop. The README already documents spreads up to 0.76 pp at 5 epochs. Single-run
   differences below ~1–2 pp cannot support method rankings.

8. **[M] Data split depends on the seed.** `build_cifar10_loaders(seed=s)` uses `s` for
   the train/validation `random_split`, so each seed sees a different 45 k/5 k partition.
   The research protocol requires a fixed partition across all comparable runs; the
   new code uses a fixed split seed (0) independent of the training seed, and the
   legacy loader is left unchanged.

9. **[M] Only a random baseline exists.** No magnitude, activation, Taylor or Network
   Slimming baseline was implemented, so RQ1/RQ2 cannot currently be answered.

10. **[M] Learned-vs-random evidence is unstructured only.** The +0.87 pp learned
    advantage (3 seeds) refers to *connection-level global* pruning from a λ = 0 model. It
    says nothing about neuron-level scoring, which is what produces physical speedups.

11. **[M] No statistical testing.** Means ± sample std over 3 seeds are reported; no paired
    tests, intervals, or multiple-comparison control.

12. **[m] Test set used only for reporting — mostly.** Checkpoint selection and λ selection
    use validation accuracy (good). `fine_tune` evaluates the test set before fine-tuning
    for reporting only; it is never used for selection. No test-set leakage found.

13. **[m] Checkpoint-size comparisons.** Dense/soft checkpoints store gate scores and
    masks (107 MB) while compact checkpoints do not; the README already flags this. The new
    code reports *deployable* size by serialising only the compact deployable state dict.

14. **[m] Gate scores receive weight decay** (AdamW over all parameters). This is an
    undocumented implicit pull of `S` towards 0 (σ → 0.5). Its magnitude over 5 epochs is
    small (lr·wd ≈ 8e-8 per step) but it is part of the method and is now documented.

15. **[m] Latency methodology.** CPU, `torch.set_num_threads(1)`, 10 warm-up / 30 timed
    iterations, random inputs; no GPU timing. Soft/hard unstructured models multiply the
    mask on every forward pass, which explains why masked models are *slower* than dense.

### 5.3 Reproducibility assessment

| Item | Status |
|---|---|
| Seeds set (`random`, `numpy`, `torch`, CUDA) | yes |
| Deterministic algorithms (`torch.use_deterministic_algorithms`) | no |
| Fixed data split across seeds | **no** (seed-dependent) |
| Configs stored with results | yes (`summary.json`, checkpoint `experiment_config`) |
| Environment metadata | partial (torch/python version in structured benchmark only) |
| Checkpoints | present locally, git-ignored |
| Re-running reproduces numbers | not verified bit-exactly; documented 0.76–2.35 pp spread between nominal repeats |
| Hardware | CPU only (torch 2.11.0+cpu, Python 3.11.9, 20 logical cores); **no CUDA device available** |

## 6. Test status at audit time

`.venv/Scripts/python.exe -m pytest -q` → **20 passed** (5.8 s). 22 test functions are
defined; 2 are shadowed (finding 2).

## 7. Consequences for the research plan

* The existing gate method is kept unchanged as **Method 6** ("gate-in", mean incoming
  gate), evaluated faithfully — including its weaknesses (findings 4–5).
* Proposed changes are implemented only as *named variants* (`gate_out`,
  `gate_combined`, normalised penalty) and never replace Method 6.
* All post-training criteria are applied to the **same source checkpoint** (with gates
  folded into the weights) so that differences isolate the scoring criterion; the
  dense-trained checkpoint is used as a second source to measure the effect of the
  training objective.
* A *dense + fine-tuning* control and a fixed data partition are added to the protocol.
* Because no GPU is available in this environment, only CPU smoke tests are executed
  here; full-scale commands are provided for a GPU machine.

---

## Appendix A: audit measurements

Gate statistics (run from repository root):

```python
import torch
for p in ["artifacts/final_benchmark/checkpoints/soft_lambda_0.0000.pt",
          "artifacts/final_benchmark/checkpoints/soft_lambda_0.0010.pt"]:
    sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    for k, v in sd.items():
        if k.startswith("prunable_layers") and k.endswith("gate_scores"):
            g = torch.sigmoid(v); r = g.mean(1)
            print(p, k, g.mean(), g.std(), r.std(), r.min(), r.max())
```

Gate-folding check (validation split of seed 42, 5 000 images):

```python
src, _ = benchmark.load_model("artifacts/final_benchmark/checkpoints/soft_lambda_0.0000.pt", cpu)
evaluate(src, val)                                     # 0.5174
c = StructuredPrunedMLP.from_self_pruning(src, 0.0)
evaluate(c, val)                                       # 0.5070  (raw W copied)
for s, t in zip(src.prunable_layers, c.linear_layers): t.weight.copy_(s.weight * s.gates())
evaluate(c, val)                                       # 0.5174  (W ⊙ σ(S) copied)
```

The same check is now an automated test (`tests/research/test_reconstruction.py`).
