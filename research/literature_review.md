# Literature Review

Compiled 2026-11-09. **Verification policy:** an entry is marked ✅ when title, authors,
venue, year and DOI/official URL were checked against at least one official or
bibliographic source during this review (sources listed per entry). Entries marked ⚠️
have verified title/venue/URL but an incomplete author list; they are **not** in
`paper/references.bib` and must not be cited until completed. Entries marked 📘 are
standard references cited from the reviewer's knowledge of the canonical publication;
they should be spot-checked before submission. No reference, author, DOI or venue was
invented.

BibTeX keys refer to [`paper/references.bib`](paper/references.bib).

---

## 1. Directly compared methods

### 1.1 Network Slimming ✅ `liu2017slimming`
* **Title:** Learning Efficient Convolutional Networks through Network Slimming
* **Authors:** Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang
* **Venue / year:** IEEE International Conference on Computer Vision (ICCV), 2017
* **DOI:** 10.1109/ICCV.2017.298 (arXiv:1708.06519)
* **Contribution:** Imposes an L1 penalty on BatchNorm scaling factors γ during training and
  removes channels with small |γ| (global threshold), followed by fine-tuning.
* **Similarity to our implementation:** Learns a multiplicative per-unit scale with an L1
  sparsity penalty — conceptually the closest established method to the self-pruning
  gates. Our gate penalty is also L1 (gates are positive, so Σσ(S) = ‖σ(S)‖₁).
* **Difference:** Slimming learns **one scale per channel** placed *after* normalisation,
  so it is not cancelled by BN; the self-pruning network learns **one gate per connection**
  placed *before* BN, and its neuron score is a mean of connection gates (see
  `INITIAL_AUDIT.md` finding 5). In our protocol Slimming is used with a per-layer ratio
  (same architecture as other methods) rather than its original global threshold; the
  global variant is reported separately.

### 1.2 Taylor first-order pruning ✅ `molchanov2017pruning`, `molchanov2019importance`
* **(a)** *Pruning Convolutional Neural Networks for Resource Efficient Inference* —
  Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, Jan Kautz. ICLR 2017
  (arXiv:1611.06440; no DOI located).
* **(b)** *Importance Estimation for Neural Network Pruning* — Pavlo Molchanov, Arun Mallya,
  Stephen Tyree, Iuri Frosio, Jan Kautz. CVPR 2019, pp. 11264–11272,
  DOI 10.1109/CVPR.2019.01152.
* **Contribution:** (a) scores a feature map by |mean(a · ∂L/∂a)|, the first-order Taylor
  estimate of the loss change when the activation is zeroed. (b) scores via gates/weights
  and reports >93 % correlation of the estimate with "true" removal importance on ImageNet.
* **Similarity:** The gradient that trains a multiplicative gate `c` is exactly the
  first-order saliency `∂L/∂c = a·∂L/∂a` at `c=1`; gate learning is therefore an
  accumulated (and, under Adam, sign-normalised) Taylor signal.
* **Difference:** Taylor is computed post hoc from a fixed calibration set; gates are
  learned jointly with weights. (b)'s correlation analysis is the model for our RQ5
  reliability experiment; we use Spearman correlation against exact single-neuron removal
  damage on validation data.

### 1.3 ThiNet ✅ `luo2017thinet`
* **Title:** ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression
* **Authors:** Jian-Hao Luo, Jianxin Wu, Weiyao Lin · **Venue:** ICCV 2017 ·
  **DOI:** 10.1109/ICCV.2017.541 (arXiv:1707.06342)
* **Contribution:** Prunes filters of layer *i* based on how well the **next** layer's
  outputs can be reconstructed without the corresponding input channels (greedy selection).
* **Relevance:** Motivates "outgoing" criteria: a unit's importance is determined by its
  effect on downstream computation, not its incoming weights. Not implemented here
  (reconstruction-based; out of scope), discussed as related work.

### 1.4 DepGraph ✅ `fang2023depgraph`
* **Title:** DepGraph: Towards Any Structural Pruning
* **Authors:** Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, Xinchao Wang
* **Venue:** CVPR 2023, pp. 16091–16101 · **DOI:** 10.1109/CVPR52729.2023.01544 (arXiv:2301.12900)
* **Contribution:** Dependency graph that groups coupled parameters across layers so any
  architecture can be structurally pruned; reports that simple norm-based group criteria
  perform well.
* **Relevance:** Our MLP/VGG pruning groups (producer rows + BN + consumer columns) are a
  hand-written special case of DepGraph grouping; the L1/L2 "incoming+outgoing" criteria
  follow its group-norm idea. Required for the optional ResNet extension (Stage C).

### 1.5 Adaptive connectivity scores (SNACS) ✅ `ganesh2024snacs`
* **Title:** Slimming Neural Networks Using Adaptive Connectivity Scores
* **Authors:** Madan Ravi Ganesh, Dawsin Blanchard, Jason J. Corso, Salimeh Yasaei Sekeh
* **Venue:** IEEE Transactions on Neural Networks and Learning Systems, 35(3):3794–3808,
  2024 (published online Aug 2022) · **DOI:** 10.1109/TNNLS.2022.3198580 (arXiv:2006.12463)
* **Contribution:** Single-shot structured pruning scoring the connectivity between filters
  of adjacent layers with a hash-based estimate of adaptive conditional mutual information,
  with per-layer limits from weight-matrix constraints.
* **Similarity:** Scores *connections between units* and aggregates them into unit
  decisions — the same aggregation question we study (incoming vs outgoing).
* **Difference:** Information-theoretic, computed post hoc; not learned by gradient
  descent. Not reimplemented here.

## 2. Closely related foundations

| Key | Reference | Status | Relevance |
|---|---|---|---|
| `lee2019snip` | N. Lee, T. Ajanthan, P. H. S. Torr, "SNIP: Single-shot Network Pruning based on Connection Sensitivity", ICLR 2019 (arXiv:1810.02340) | ✅ (arXiv id verified) / 📘 | Introduces multiplicative per-connection indicator variables `c` and scores connections by `|∂L/∂c|` — mathematically the gradient that drives the self-pruning gates at initialisation. **Most important prior art for the novelty audit.** |
| `louizos2018l0` | C. Louizos, M. Welling, D. P. Kingma, "Learning Sparse Neural Networks through L0 Regularization", ICLR 2018 | 📘 | Learnable stochastic gates (hard-concrete) with an expected-L0 penalty; principled alternative to the sigmoid+L1 gates. |
| `han2015learning` | S. Han, J. Pool, J. Tran, W. J. Dally, "Learning both Weights and Connections for Efficient Neural Networks", NeurIPS 2015 | 📘 | Magnitude-based unstructured pruning + retraining; the legacy unstructured pipeline mirrors it. |
| `li2017filters` | H. Li, A. Kadav, I. Durdanovic, H. Samet, H. P. Graf, "Pruning Filters for Efficient ConvNets", ICLR 2017 | 📘 | L1-norm filter pruning: our Method 2 incoming term. |
| `hu2016trimming` | H. Hu, R. Peng, Y.-W. Tai, C.-K. Tang, "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures", arXiv:1607.03250, 2016 | 📘 | Activation-statistics (APoZ) neuron pruning; motivates Method 4. |
| `ye2018rethinking` | J. Ye, X. Lu, Z. Lin, J. Z. Wang, "Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers", ICLR 2018 | 📘 | Shows norm criteria are unreliable with BN (scale invariance) — supports audit finding 5. |
| `liu2019rethinking` | Z. Liu, M. Sun, T. Zhou, G. Huang, T. Darrell, "Rethinking the Value of Network Pruning", ICLR 2019 | 📘 | Training-budget confounds: pruned+fine-tuned vs trained-from-scratch; motivates our dense+FT control. |
| `blalock2020state` | D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, J. Guttag, "What is the State of Neural Network Pruning?", MLSys 2020 | 📘 | Reporting standards (same budgets, multiple seeds, report params *and* FLOPs). Our protocol follows it. |
| `he2024survey` | Y. He, L. Xiao, "Structured Pruning for Deep Convolutional Neural Networks: A Survey", IEEE TPAMI (IEEE Xplore doc. 10330640; arXiv:2303.00566) | ✅ title/venue/authors via arXiv & Xplore | Taxonomy: filter ranking, regularisation, dynamic execution. |
| `ioffe2015bn` | S. Ioffe, C. Szegedy, "Batch Normalization…", ICML 2015 | 📘 | Scale invariance argument. |
| `simonyan2015vgg` | K. Simonyan, A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR 2015 | 📘 | Stage B architecture. |
| `krizhevsky2009cifar` | A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", tech. report, Univ. Toronto, 2009 | 📘 | Dataset. |
| `kingma2015adam`, `loshchilov2019adamw` | Adam (ICLR 2015); Decoupled Weight Decay Regularization (ICLR 2019) | 📘 | Explains the uniform-drift effect of the L1 gate penalty under Adam (audit finding 4). |
| `holm1979` | S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure", Scand. J. Statistics 6(2):65–70, 1979 | 📘 | Multiple-comparison correction. |

## 3. Recent developments (2025–2026)

| Key | Reference | Status | Relation |
|---|---|---|---|
| `franssen2025connect` | C. Franssen, J. Jiang, Y. Peng, B. Heidergott, "CoNNect: Connectivity-Based Regularization for Structural Pruning of Neural Networks", TMLR 2025 (arXiv:2502.00744) | ✅ | Regulariser that favours preserving input–output connectivity; a structured, connectivity-aware alternative to per-connection L1 gates. |
| `le2025probe` | Q. Le, E. Diao, Z. Wang, X. Wang, J. Ding, L. Yang, A. Anwar, "Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing", ICLR 2025 (arXiv:2502.15618) | ✅ | Dynamic (batch-wise) structured pruning without fine-tuning. Different regime (LLMs, input-dependent masks); cited to delimit our static-pruning scope. |
| — | Sun et al., "MDP: Multidimensional Vision Model Pruning with Latency Constraint", CVPR 2025 (openaccess.thecvf.com) | ⚠️ author list unverified | Latency-constrained channel + block pruning; supports reporting *measured* latency rather than MACs alone. |
| — | Berisha et al., "Variance-Based Pruning for Accelerating and Compressing Trained Networks", ICCV 2025 (openaccess.thecvf.com) | ⚠️ author list unverified | Activation-variance criterion for trained networks; close to our Method 4. |
| — | Zhang et al., "Flexible Group Count Enables Hassle-Free Structured Pruning", CVPR 2025 | ⚠️ | Finer-than-filter structured granularity. |

A search for 2026 CNN channel-pruning papers on learnable connection gates did not surface
a verifiable venue publication at the time of writing; this gap should be re-checked
before submission.

## 4. Synthesis

1. **Per-connection multiplicative gates are established** (SNIP's indicator variables;
   L0 gates; magnitude×gate products). The self-pruning parameterisation `W⊙σ(S)` with an
   L1 penalty on σ(S) is not novel on its own.
2. **Per-unit learned scales are established** (Network Slimming).
3. **What is less studied**, and is the actual subject of this project, is *how
   connection-level learned scores should be aggregated into unit-level decisions*, and
   whether the aggregate is reliable when the gated layer is followed by normalisation.
   Our contribution is an empirical comparison and a reliability analysis, not a new
   algorithm.

Sources consulted (verification): arXiv abstract pages 1708.06519, 1611.06440, 1707.06342,
2301.12900, 2006.12463, 2502.00744, 2502.15618, 1810.02340, 2303.00566; CVF open access
(CVPR 2019, CVPR 2023, CVPR 2025, ICCV 2025); ICLR 2025 proceedings; mlanthology.org; IEEE
Xplore document 10330640; Stevens Institute research portal (SNACS); TMLR listing.
