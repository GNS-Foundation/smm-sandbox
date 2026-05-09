# Lab Design & Execution

## Sample-Efficiency Showdown: Structured vs Dense Architectures

**Status:** Lab plan, draft v1
**Companion to:** `01_deep_research_brief.md`
**Duration:** 9 months
**Lab size:** 5 FTE

---

## 1. Operating Principle

We test a single, falsifiable claim under conditions designed to make falsification possible. The plan is **pre-registered**: hypotheses, architectures, benchmarks, metrics, and analysis are fixed before training begins. Surprises encountered mid-experiment become exploratory follow-ups, not retrofitted main results.

The core failure mode of this research area to date has been single-architecture demos on cherry-picked tasks. We avoid that by (a) running a fixed benchmark suite across all architectures, (b) including adversarial cases where structural priors should *fail*, and (c) reporting sample-efficiency *curves*, not single-point accuracies.

---

## 2. Hypotheses

**H1 (primary).** For structured tasks, at least one structural architecture family reaches the target performance threshold $P^*$ using ≥ 2× fewer training samples than a parameter-matched, compute-matched dense Transformer baseline.

**H2 (compounding).** Sample-efficiency advantages do not come at the cost of compute efficiency: the structural winner reaches $P^*$ in fewer FLOPs as well as fewer samples.

**H3 (composition).** Hybrid architectures combining two structural priors (e.g., equivariant + oscillatory) yield a sample-efficiency advantage greater than either prior alone — testing whether priors compose or interfere.

**H4 (no catastrophic failure).** On unstructured (Tier 3) tasks, structural architectures reach within 10% of the baseline's sample efficiency. We are not interested in priors that win on structure but tank on everything else.

**H5 (scaling).** The sample-efficiency advantage observed at 10M parameters is preserved (within a factor of 0.5×) at 1B parameters. This is the highest-risk hypothesis and the most important to test.

We will report results separately for each hypothesis; H1 alone constitutes a publishable positive result. H5 alone constitutes a major one.

---

## 3. Architecture Cohort

Six entries. All implemented in a shared codebase with shared training infrastructure, optimizer settings, and tokenization where applicable. Hyperparameter budgets are equalized (each architecture gets the same number of HP-search trials per benchmark).

| ID | Name | Family | Reference Implementation |
|----|------|--------|--------------------------|
| **A** | **Dense Transformer** (with RoPE) | Baseline | Standard Llama-style |
| **B** | **SE(n)-Equivariant Network** | Geometric | EGNN / E3NN / SEGNN |
| **C** | **Fourier Operator + AFNO** | Spectral | FNO + AFNO token mixer |
| **D** | **AKOrN-style Kuramoto** | Oscillatory | Miyato et al. 2025 |
| **E** | **LrcSSM** | Liquid / continuous | Farsang et al. 2025 |
| **F** | **Hybrid (Equivariant × Oscillatory)** | Compositional | Novel; built in Phase 3 |

**Three parameter scales:** 10M, 100M, 1B. Total: 6 × 3 = **18 model configurations**, each evaluated across the benchmark suite at multiple training-set sizes.

**Matching constraints (these matter):**

- **Parameter count match:** ±10% across architectures at each scale.
- **Compute match (training FLOPs):** All architectures get the same FLOP budget per (size × dataset-size) cell. This is not the same as wall-clock match — equivariant networks may be slower per FLOP on GPU. We report both.
- **Optimizer:** AdamW with cosine schedule, identical across runs unless an architecture has documented incompatibility (in which case we run both its native optimizer and AdamW for comparability).
- **Random seeds:** Three seeds per cell minimum; report mean ± std.

---

## 4. Benchmark Suite

Three tiers. All benchmarks evaluated with **sample-efficiency curves**: performance plotted against $N_{\text{train}}$ (log-spaced: 100, 1k, 10k, 100k, 1M, 10M training examples where applicable). The headline number for each is **N₅₀** — examples needed to reach 50% of the human or oracle baseline.

### Tier 1 — Structured (priors should help)

| Benchmark | Structure | Why |
|-----------|-----------|-----|
| **ARC-AGI-2** | Compositional, abstract | Designed to defeat brute-force scale; efficiency is a reported metric |
| **N-body simulation rollout** | SE(3) symmetry | Clean equivariance test; analytic ground truth |
| **Burgers + Navier-Stokes PDE** | Spectral structure | FNO's home turf; tests if FNO's edge is real or curated |
| **Sudoku (9×9 + 16×16 OOD)** | Compositional, periodic | AKOrN's headline result; tests reproducibility |

### Tier 2 — Mixed (priors *might* help)

| Benchmark | Structure | Why |
|-----------|-----------|-----|
| **Conway's Game of Life forecasting** | Local + temporal | Tests temporal-spatial inductive bias |
| **Drone navigation in unseen environments** | Continuous control, partial symmetry | Liquid networks' claimed strength |
| **Long-range sequence modeling (PathX, ListOps)** | Long-range dependency | Tests state-space + Fourier vs attention |

### Tier 3 — Unstructured (priors should *not* hurt)

| Benchmark | Why included |
|-----------|--------------|
| **Random-labeled MNIST** | Adversarial; no structure to exploit |
| **Tiny natural-language classification (50k SST-2 size)** | Mostly statistical, not geometric |
| **Adversarial permutation tasks** | Penalize wrong-prior overfitting |

**Total benchmark count:** 10. **Total runs:** 18 configs × 10 benchmarks × 5 sample sizes × 3 seeds = **2,700 training runs.** A non-trivial number; the compute budget below accommodates it.

---

## 5. Metrics

**Primary:**
- **Sample-efficiency curve** P(N) for every (architecture, benchmark, scale)
- **N₅₀** — samples to reach 50% of oracle/human performance
- **N₉₀** — samples to reach 90%

**Secondary:**
- **Compute to P\*** (total training FLOPs)
- **Energy per inference** (J), measured on a fixed reference GPU
- **OOD generalization gap** — performance on held-out distribution
- **Calibration (ECE)** — expected calibration error
- **Adversarial robustness** — accuracy under standard perturbations (FGSM/PGD where applicable)

**Reporting format:**
- All curves released as raw data (CSV) + plots
- Headline table: N₅₀ ratio (architecture / baseline) for each (architecture, benchmark, scale) cell
- One figure per benchmark showing all 6 curves at the largest scale tested

---

## 6. Phases

### Phase 0 — Infrastructure (Month 1)

- Shared codebase with reference implementations of all 6 architectures
- Unified training harness: dataset pipelines, eval loop, logging, seeding
- **Sanity-check requirement:** reproduce one published result from each architecture family within ±5% of the published number before that architecture enters the main runs

**Gate:** Phase 1 does not begin until all 6 architectures pass sanity-check on at least one published benchmark.

### Phase 1 — Tier 1 Sweeps (Months 2–4)

Train and evaluate all 18 configs on Tier 1 benchmarks. This is the largest single compute consumer; we use it to also produce within-family scaling curves.

**Outputs:** Sample-efficiency curves on Tier 1; preliminary headline numbers.

### Phase 2 — Tier 2 + Tier 3 (Months 5–6)

Test cross-task generalization (Tier 2) and adversarial unstructured cases (Tier 3). Tier 3 specifically tests H4 — whether priors fail gracefully when wrong.

### Phase 3 — Hybrid Architecture (Months 7–8)

Based on Phase 1 winners, build hybrid F (Equivariant × Oscillatory). Test on the same suite. This is the most exploratory phase; we accept higher risk.

### Phase 4 — Write-up & Release (Month 9)

Paper drafting (NeurIPS or ICLR target), open-source release of benchmark suite + reference implementations + raw run data, public leaderboard.

---

## 7. Compute Budget

Estimating from the FNO, AKOrN, and equivariant scaling-law papers as references for FLOP/sample, and using current frontier scaling-law data for the dense baseline:

| Phase | GPU-hours (H100-equivalent) |
|-------|-----------------------------|
| Phase 0 (sanity checks) | 2,000 |
| Phase 1 (Tier 1 sweeps) | 30,000 |
| Phase 2 (Tier 2 + 3) | 10,000 |
| Phase 3 (Hybrid) | 8,000 |
| Phase 4 (final eval, ablations, rebuttals) | 5,000 |
| **Buffer (15%)** | 8,000 |
| **Total** | **~63,000 H100-hours** |

At ~\$2/H100-hour cloud rate, this is ~\$125k in compute. Owning 8 H100s for 9 months covers it with margin. The 1B-parameter scale is the dominant cost; we pre-register a fallback to cap at 300M if we encounter overruns, with 1B as a stretch.

---

## 8. Team

- **1 PI** — research direction, paper writing, external coordination
- **2 ML research engineers** — architecture implementation, training-loop optimization, debugging
- **1 ML research scientist** — benchmark design, statistical analysis, theory
- **1 software/infrastructure engineer** — distributed training, eval harness, data pipelines, public release

A 5-person team is tight for 2,700 runs. Mitigations: heavy use of shared code, automated sweeps, and a strict no-bespoke-code rule (every architecture must conform to the shared training interface).

---

## 9. Success / Failure Criteria

We pre-commit to interpretation rules so we cannot move the goalposts.

**Strong success (publishable as headline result):**
- ≥ 1 structural family achieves N₅₀ ratio ≤ 0.5 (i.e. 2× sample efficiency) on ≥ 50% of Tier 1 benchmarks at ≥ 100M parameters, AND
- The same family does not exceed 1.1× baseline N₅₀ on Tier 3 (no catastrophic failure)

**Moderate success (publishable):**
- Advantages exist at small scale but close at 1B (interesting, supports the bitter lesson at frontier but motivates structural priors at edge / data-scarce settings)
- Or: advantage exists on a *subset* of Tier 1 cleanly mapped to specific structural assumptions

**Null result (still publish):**
- No architecture family beats baseline beyond noise. A high-quality null result on a unified benchmark is itself a contribution; we commit to publishing either way to avoid file-drawer bias.

**Execution failure:**
- < 70% of planned runs complete by Month 8. Triggers replan / scope cut.

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 1B-scale runs exceed compute budget | High | High | Pre-registered fallback to 300M cap; 1B as stretch |
| Implementation bugs in less-mature architectures | Medium | High | Phase 0 sanity-check gate; pair-program implementations |
| Benchmark contamination (e.g. ARC-AGI-1 memorized) | High for ARC-AGI-1 | Medium | Use ARC-AGI-2 + custom held-out tasks generated post-cutoff |
| Hyperparameter unfairness across architectures | Medium | High | Equal HP-search budget; document HP choices; release sweep configs |
| Wall-clock vs FLOP discrepancy obscures results | Medium | Medium | Report both metrics separately; do not collapse |
| Negative result discourages funding renewal | Medium | Medium | Pre-commit to publishing null; frame negative result as decision-relevant |
| Hybrid architecture (Phase 3) doesn't converge | Medium | Low | Treat as exploratory; main paper does not depend on it |
| Reviewer pushback on benchmark choice | High | Low | Document selection criteria; offer to add reviewer-suggested benchmarks in rebuttal |

---

## 11. Pre-Registration

Before Phase 1 begins, we file a pre-registration on OSF specifying:
- All hypotheses (verbatim from §2)
- All architectures and their reference implementations
- All benchmarks and their oracle/human baselines
- All metrics and their precise definitions
- Analysis plan (which statistical test for which comparison)
- Stopping rules and the fallback compute cap

Any deviation from the pre-registration is reported transparently in the final paper as exploratory rather than confirmatory.

---

## 12. Deliverables

By end of Month 9:

1. **Open benchmark suite** — code, data, eval harness, leaderboard infrastructure
2. **Six reference implementations** — clean, tested, at all three parameter scales
3. **Raw run data** — every loss curve, every metric, every seed, ~30 GB of CSV/JSON
4. **Paper** — submitted to NeurIPS or ICLR, posted to arXiv concurrent with submission
5. **Pre-registration record** — OSF link in the paper
6. **Blog post / explainer** — accessible writeup for non-specialists

The benchmark suite outlasts the paper. Even if our specific result is null or contested, the unified evaluation harness becomes a public good — others can extend it with their architectures and contribute to the leaderboard.

---

## 13. What Comes Next

Conditional on Phase 4 outcomes:

- **Strong success → Phase 5:** Push to 10B+ parameters (dependent on grant or partnership). Test scaling-law extrapolation directly.
- **Moderate success → Phase 5':** Focus on the specific task families where structural priors won. Develop production-grade implementations for those domains (edge, scientific computing, data-scarce settings).
- **Null result → Phase 5'':** Pivot to test whether the failure is fundamental or implementation-bound. Investigate whether structural priors can be *learned* from data rather than imposed (meta-learning the prior).

In all three scenarios, the lab continues. The benchmark suite continues to be maintained. The question — *does enforcing geometric and resonant structure beat brute-force scaling on data efficiency?* — is large enough to support multiple cycles of investigation regardless of any single result.
