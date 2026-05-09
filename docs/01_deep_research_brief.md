# Deep Research Brief

## Geometric & Resonant Inductive Biases vs Brute-Force Scaling: An Empirical Program

**Status:** Pre-lab framing document
**Date:** May 2026
**Purpose:** Establish the precise empirical question, survey the current state of evidence, identify the gap our lab fills.

---

## 1. The Question, Precisely Stated

For a target task $T$ and a target performance level $P^*$, we want to compare the **sample efficiency frontier** of two architecture classes:

- **Class A (Baseline):** Dense Transformers — the brute-force-scaling reference, treating intelligence as a statistical fitting problem over near-arbitrary representations.
- **Class B (Structured):** Architectures that explicitly encode geometric, spectral, or resonant priors — equivariant networks, Fourier operators, oscillator-coupled networks, liquid (continuous-time) systems, and hybrids of these.

Our central empirical question:

> **Holding parameter count and training compute constant, does any member of Class B reach $P^*$ on task $T$ using strictly fewer training samples than the matched Class A baseline — and if so, on which task families, by how much, and does the advantage persist as we scale?**

Sub-questions:

- **Where does the gap appear, vanish, or invert** across task families (geometric, periodic, compositional, unstructured)?
- **Is the advantage in sample efficiency, OOD generalization, calibration, or all three?**
- **Does the gap close with scale** (the "bitter lesson" hypothesis), persist, or widen?
- **Do structural priors compose** — i.e. does stacking equivariance + oscillation yield multiplicative gains, or do they interfere?

This question is fundable on its own terms regardless of any speculative metaphysics. The framing matters: we are not asking "is the universe a helix." We are asking whether enforcing certain geometric and resonant priors is empirically a better inductive bias than "let the data figure it out."

---

## 2. Why Now — Three Threads Converging

Until ~2023, the case for structural priors was largely theoretical (Bronstein et al.'s Geometric Deep Learning program) or restricted to narrow scientific domains (molecular property prediction, PDE solvers). Three developments in 2024–2026 changed the empirical picture:

**(a) Oscillator-based neurons solve reasoning tasks transformers fail.** Miyato et al.'s **Artificial Kuramoto Oscillatory Neurons (AKOrN)** was an ICLR 2025 Oral. The architecture replaces threshold units with phase-coupled oscillators governed by a generalized Kuramoto equation. AKOrN achieves competitive object discovery, sharply improved adversarial robustness, well-calibrated uncertainty, and — critically — **solves Sudoku puzzles**, a compositional reasoning task on which transformers stall. With test-time extension of Kuramoto steps and energy-based voting, OOD Sudoku accuracy rises from 18% to ~90%.

**(b) Synchronization-based encoding improves data efficiency on ARC-AGI.** **Kuramoto Oscillatory Phase Encoding (KoPE)**, posted in 2026, embeds Kuramoto dynamics into the positional/structural encoding layer of standard architectures. The reported result: improvements in training efficiency, parameter efficiency, **and data efficiency** across supervised and self-supervised learning, with measurable gains on ARC-AGI — the benchmark explicitly designed to defeat brute-force scaling.

**(c) ARC Prize 2025 made efficiency a first-class metric.** ARC-AGI-2 reporting now includes a cost/efficiency metric alongside accuracy. The top Kaggle entry achieved only **24% on ARC-AGI-2** (NVARC, test-time training + synthetic data). Frontier closed models do better but at extreme cost (Gemini 3 Deep Think: 84.6% at \$13.62/task). The ARC Prize Foundation explicitly states the **efficiency gap remains bottlenecked by science and ideas, not engineering**. This is precisely the gap structural priors plausibly address.

In parallel, two longer-running programs provide supporting evidence:

- **Equivariant networks** (Cohen, Welling, Bronstein) have demonstrated sample-complexity advantages on geometrically structured tasks for nearly a decade. A recent neural force field study (arxiv 2510.09768) shows **power-law scaling exponents increase with the order of equivariance** — direct evidence that structural priors compound with scale rather than being washed out by it.
- **Liquid neural networks** (Hasani, Rus et al., MIT CSAIL → Liquid AI) have produced 20,000-parameter controllers that beat much larger systems on drone navigation. **LrcSSM** (Farsang, Hasani, Rus, Grosu, 2025) outperforms quadratic-attention transformers at equal compute on long-range forecasting benchmarks. Liquid AI claims 10–1000× inference-energy efficiency over transformer baselines in production.

The convergence is not coincidence. Multiple independent groups, pursuing different theoretical motivations, are arriving at the same empirical pattern: **for tasks with structure, structured architectures beat unstructured ones at the same compute, often by large margins.**

---

## 3. State of the Art — Architecture Families

We survey five families that operationalize "geometric and resonant structure" in different ways. Each is a candidate for Class B in our experimental design.

### 3.1 Equivariant / Geometric Deep Learning

**Core principle:** Build the symmetry group of the data into the architecture, so the network's function commutes with that group. Translation equivariance is what makes CNNs work; the program generalizes to rotations (G-CNNs), continuous Lie groups (E(n)-equivariant networks), graphs (GNNs), and arbitrary differentiable manifolds.

**Evidence:** Strong sample-efficiency advantages on tasks with the relevant symmetry; cleanly degraded on tasks without it. Maurice Weiler's empirical summary: *equivariant models are more data-efficient, less prone to overfitting, and converge faster than non-equivariant counterparts on equivariant tasks.* The Northeastern "Surprising Effectiveness of Equivariant Models" study (ICLR 2023) confirms the advantage is most pronounced in the low-data regime — exactly what our research question targets.

**Caveat:** When the assumed symmetry doesn't hold ("invert-label" experiments), equivariance can hurt. Priors must be matched to task structure.

### 3.2 Spectral / Fourier Operators

**Core principle:** Operate in frequency space rather than the spatial/temporal domain. Convolution becomes pointwise multiplication; global mixing becomes O(n log n) instead of O(n²).

**Evidence:**
- **Fourier Neural Operator (Li et al.):** Solves PDEs by learning in Fourier basis; outperforms CNN/ResNet baselines on Burgers, Navier-Stokes, Darcy flow by orders of magnitude in some regimes. Matches the ground-truth solver at fraction of the cost.
- **FNet (Lee-Thorp et al., Google):** Replaces transformer attention with unparameterized Fourier transforms. 80% faster training on GPUs, 92% of BERT accuracy on GLUE.
- **AFNO (Adaptive Fourier Neural Operator):** Used as a transformer token mixer; outperforms self-attention on few-shot segmentation, handles 65k-token sequences.

**Caveat:** Best evidence is on continuous, periodic, or PDE-like data. Discrete reasoning tasks have less clear-cut spectral structure.

### 3.3 Oscillatory / Resonance Networks

**Core principle:** Replace scalar activations with phase-coupled oscillators. Information is carried by *synchronization patterns*, not just amplitudes. Theoretical roots in Kuramoto's coupled-oscillator model (1975) and in observed brain dynamics.

**Evidence:**
- **AKOrN (ICLR 2025 Oral):** Object discovery competitive with slot-based models; strong adversarial robustness; calibration that almost perfectly tracks accuracy; solves Sudoku via test-time Kuramoto iteration.
- **KoPE (2026):** Sample-efficient on ARC-AGI; improves attention concentration in vision-language alignment.
- **DcKONN (2025):** Biologically grounded EEG classification; 6,242 trainable parameters reaches 98%+ accuracy across frequency bands, outperforming classical pipelines.

**Caveat:** Most published results are below 100M parameters. Scaling behavior beyond this is largely unproven.

### 3.4 Liquid / Continuous-Time Networks

**Core principle:** Neurons governed by continuous differential equations rather than discrete activation functions. State-space and time-constant parameters are themselves dynamic.

**Evidence:**
- **Liquid Time-Constant Networks (LTCs)** and **Closed-form Continuous-time (CfC)** networks: 34-neuron drone controllers that outperform large RNN/transformer policies in unseen environments.
- **LrcSSM (2025):** Diagonalized liquid networks; matches Mamba scaling law (β ≈ 0.42), outperforms quadratic-attention transformers at equal compute, formal gradient-stability guarantees not held by Mamba or Liquid-S4.
- **Liquid Foundation Models:** Commercial language models, one 40B-parameter version reportedly beating Meta 70B on specific benchmarks.

**Caveat:** Commercial-source claims (Liquid AI) need independent verification. Strongest peer-reviewed results are still in narrow domains.

### 3.5 Hyperbolic / Non-Euclidean Embeddings

**Core principle:** Embed hierarchical data in negatively-curved space, where representational capacity grows exponentially with radius rather than polynomially.

**Evidence:** Strong on tree-like data (taxonomies, knowledge graphs, citation networks); weaker on flat or weakly-hierarchical data. Less direct relevance to our central question, but a candidate component for hybrid architectures.

---

## 4. The Evidence For — Concrete Empirical Hooks

The following are our strongest pre-existing data points. Each is a published, reproducible result that supports the hypothesis in a narrow domain. The lab's job is to test whether the pattern generalizes.

| Result | Source | Architecture Family | Magnitude |
|--------|--------|---------------------|-----------|
| FNO outperforms CNN/ResNet on Navier-Stokes by orders of magnitude | Li et al. 2020 | Spectral | 1–3 OOM speed/accuracy |
| AKOrN solves Sudoku at 90% OOD vs ~0% for transformers | Miyato et al., ICLR 2025 | Oscillatory | Categorical (works vs fails) |
| KoPE improves ARC-AGI sample efficiency | KoPE 2026 | Oscillatory + Attention | Reported gains on data-eff |
| Equivariant scaling-law exponents grow with equivariance order | arxiv 2510.09768 | Equivariant | Power-law exponent shift |
| 34-neuron liquid net outperforms LSTM/transformer drone control | Hasani et al. | Liquid | 100–1000× param reduction |
| FNet matches BERT at 80% training speedup | Lee-Thorp et al. | Spectral | 0.92 acc, 1.8× speed |

---

## 5. The Evidence Against — Honest Counter-Cases

A rigorous research program also catalogues what cuts against the hypothesis. The lab will design controls around each.

- **Wrong-prior penalty.** When the assumed symmetry doesn't hold, equivariant models underperform standard CNNs (the "invert-label" result). Priors aren't free.
- **Augmentation can substitute.** Gerken et al. (2202.03990) show that for invariant tasks (image classification), a sufficiently large CNN with enough rotation augmentation matches an S2CNN. Some structural priors may simply be a compute-efficient way of achieving what data augmentation does at scale.
- **Brute force still leads at the absolute frontier.** ARC-AGI-1: GPT-5.2 Pro reaches 90.5%, Opus 4.6 reaches 93.0%. No structured architecture comes close at that level — though both cost \$1.88–\$11.64/task vs human costs measured in cents.
- **Scaling track record is unproven.** AKOrN, KoPE, DcKONN are demonstrated below 100M params. Whether the advantage survives at 10B+ is the open question.
- **Implementation overhead.** Equivariant networks can be 3–5× slower per FLOP than dense baselines on current GPU hardware, partially eating sample-efficiency gains in wall-clock terms.

The honest version of the hypothesis is therefore: **structural priors yield large sample-efficiency gains on appropriately-structured tasks at small-to-medium scale, with scaling behavior unknown but plausibly favorable based on equivariant scaling-law data.**

---

## 6. The Gap This Lab Fills

Despite the convergent evidence, **no published study** does the following all at once:

1. Compares all major structural families against a strong dense-Transformer baseline
2. At **matched parameter count** AND **matched training compute**
3. Across a **deliberately heterogeneous benchmark** (structured, mixed, adversarially-unstructured)
4. With **sample-efficiency curves** as the primary reporting format (not just final accuracy)
5. **Pre-registered**, so the analysis plan cannot be retro-fit to results
6. With at least one **scaling sweep** to test whether advantages persist at 1B parameters

The closest analogues are within-family scaling studies (e.g. the 2510.09768 equivariant work) or single-architecture demonstrations (AKOrN, KoPE). A unified, cross-family, sample-efficiency-first benchmark is the missing artifact.

---

## 7. What a Positive Result Would Mean

If the hypothesis is confirmed strongly — i.e., one or more structural families consistently beats the dense baseline by 2× or more in sample efficiency on structured tasks while not catastrophically failing on unstructured ones — the implications are concrete:

- **Capital efficiency in frontier training.** A 2× sample-efficiency gain at 1B parameters compounds to large wall-clock and dollar savings; the same data buys more capability.
- **Edge & embodied AI.** Liquid-style and oscillatory networks are already preferred for edge inference; a confirmed sample-efficiency advantage strengthens the deployment case.
- **A genuine alternative path** to the data-and-GPU arms race. Not a replacement at the frontier yet, but a credible second axis.
- **A bridge to neuromorphic and photonic hardware.** If resonance and continuous dynamics are the right substrate, digital silicon is the wrong simulator. Photonic computing (Lightmatter, Lightelligence) and neuromorphic chips (Loihi 2, Akida) become natural targets.

A *negative* result is also valuable: it would settle a long-running theoretical debate and re-focus efforts on data and scale. We commit to publishing either way.

---

## 8. Selected References

- Miyato, T., Löwe, S., Geiger, A., Welling, M. *Artificial Kuramoto Oscillatory Neurons.* ICLR 2025 Oral. arXiv:2410.13821.
- *Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency.* arXiv:2604.07904 (2026).
- Bronstein, M.M., Bruna, J., Cohen, T., Veličković, P. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* 2021.
- Cohen, T., Welling, M. *Group Equivariant Convolutional Networks.* ICML 2016.
- Li, Z. et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
- Lee-Thorp, J. et al. *FNet: Mixing Tokens with Fourier Transforms.* 2021.
- Guibas, J. et al. *Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers.* ICLR 2022.
- Hasani, R., Lechner, M., Amini, A., Rus, D., Grosu, R. *Liquid Time-Constant Networks.* AAAI 2021.
- Farsang, M., Hasani, R., Rus, D., Grosu, R. *Scaling Up Liquid-Resistance Liquid-Capacitance Networks.* 2025. arXiv:2505.21717.
- *Scaling Laws and Symmetry: Evidence from Neural Force Fields.* 2025. arXiv:2510.09768.
- Gerken, J. et al. *Equivariance versus Augmentation for Spherical Images.* ICML 2022.
- Chollet, F. et al. *ARC-AGI-2 Technical Report.* ARC Prize Foundation, 2025.
- *ARC Prize 2025 Technical Report.* arXiv:2601.10904 (2026).

---

*Companion document: `02_lab_design_and_execution.md` — operationalizes this brief into a 9-month experimental program with hypotheses, benchmarks, compute budget, and pre-registration plan.*
