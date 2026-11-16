# Novelty Audit

Purpose: decide, before any experiment, which parts of the project are established
techniques and which questions are open. Claims of novelty are **not** made in the
manuscript unless they survive this audit. See `literature_review.md` for references.

## 1. Component-by-component

| Component of the existing method | Established? | Closest prior work | Verdict |
|---|---|---|---|
| Multiplicative per-connection gate `W ⊙ σ(S)` | Yes | SNIP indicator variables (Lee et al. 2019); L0 gates (Louizos et al. 2018) | Not novel |
| L1 penalty on gate values | Yes | Network Slimming (L1 on BN γ); L1 sparsity generally | Not novel |
| Ranking connections by learned gate, global top-k mask, fixed-mask fine-tuning | Yes | Han et al. 2015 (magnitude), SNIP (sensitivity) | Not novel |
| Neuron score = **mean incoming gate** | Rarely studied as such | Slimming uses one scale per unit; SNACS aggregates connection scores | Not a contribution in itself; its *reliability* is an open empirical question |
| Physically compact reconstruction (rows/cols/BN slicing) | Yes | Li et al. 2017; DepGraph 2023 | Engineering, not novel |
| Uniform per-layer pruning ratio | Yes | standard | Not novel |

## 2. Mathematical observations that matter for the study

**O1 — Gate gradients are accumulated Taylor saliencies.** For one connection with
`w_eff = w·g`, `g = σ(s)`:
`∂L/∂s = (∂L/∂w_eff) · w · σ'(s)`. At fixed `g`, `(∂L/∂w_eff)·w_eff = (∂L/∂g)·g` is the
first-order Taylor change of the loss when the connection is removed (SNIP's sensitivity).
Gate learning integrates this signal over training. Under Adam the per-coordinate
normalisation removes the magnitude and keeps (approximately) the sign consistency.

**O2 — BN scale invariance of incoming gates.** If `h_j = Σ_i W_ji g_ji x_i + b_j` is
followed by BatchNorm, then replacing `g_j· → c·g_j·` (`c>0`) leaves BN's output unchanged
up to ε (the bias is removed by mean subtraction). Hence `mean_i g_ji` — the existing
neuron score — is not identifiable from the network function and receives no consistent
gradient. Outgoing gates `g_{kj}` of the next layer are **not** invariant in this way.

**O3 — Uniform drift under Adam.** With `L_sparsity = Σ σ(S)` and λ large enough that
`λσ'(S)` dominates every gate's gradient, Adam produces almost identical updates for all
gates, so the penalty lowers all gates together and does not rank them. Observed in the
committed λ=1e-3 checkpoint (all 8.9 M gates within 0.7417–0.7431 in hidden layers).

O1–O3 are elementary consequences of the definitions; they are stated as
*explanations to be tested*, not as theorems we contribute.

## 3. Open questions this project can legitimately address

1. (RQ1/RQ2) Under matched budgets and identical architectures after pruning, do learned
   gate scores rank neurons/channels better than magnitude, activation, Taylor, random,
   or BN-γ (Slimming) criteria?
2. (RQ5) How well does each score's ranking correlate with *exact* single-unit removal
   damage? Is the mean-incoming-gate score systematically worse, as O2 predicts, and does
   an outgoing aggregation fix it?
3. (RQ3) How stable are scores and masks across seeds (rank correlation, Jaccard overlap)?
4. (Ablation A) Does a normalised penalty, or λ=0, change the discriminative power of
   gates — testing O3?

## 4. Permissible claims

* "We present a controlled comparison …" ✔
* "We identify a BN scale-invariance that makes the mean-incoming-gate score
  uninformative in BN networks, and verify it empirically" ✔ only if the experiments
  confirm it; the invariance itself must be credited as a known BN property
  (Ioffe & Szegedy 2015; cf. Ye et al. 2018 for norm criteria).
* "We propose an outgoing/combined gate aggregation" — permissible **only** as a
  *variant evaluated in an ablation*, not as a novel algorithm, unless a more complete
  literature search finds no equivalent and it clearly outperforms baselines.
* "Learnable gates are a new pruning method" ✘ not permissible.
