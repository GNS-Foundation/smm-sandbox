# 03 — Findings & Future Work

**Internal research memo.** Not intended for publication; written to be readable in six months
when the implementation details have faded but the conclusions still need to be valid.

**Authors:** Camilo Ayerbe Posada, with Claude as research partner.
**Repo:** github.com/GNS-Foundation/smm-sandbox.
**Status:** End of the synthetic-Sudoku phase. Next-step decisions pending.
**Companion docs:** `01_deep_research_brief.md` (motivation), `02_lab_design_and_execution.md` (original plan).

---

## TL;DR

The original lab plan proposed: *structural priors give ≥2× sample efficiency that compounds with scale.* On synthetic Sudoku reasoning at scales up to ~14K parameters, this prediction is **partially supported but not in the form originally claimed**.

What's robust:

1. **Helical-distance-modulated Kuramoto coupling** (Module 2 of the SMM toolkit, wired into the
   Kuramoto layer's coupling matrix) provides a **20–50% sample-efficiency advantage** on
   in-distribution learning, reproducible across base puzzle pools (base seed 42 and base seed 137).

2. **Topology choice (helix vs torus) determines the bias-variance regime** the model occupies.
   Single-helix produces high raw accuracy with severe small-N overfitting (gap-ratio 3.4–3.6×
   baseline); torus produces lower raw accuracy with cleaner generalization gap (2.3×) and
   dramatically more reliable behavior at scale.

3. **toroidal_coupled_wide at N=1024 produces the best OOD-difficulty result in the entire sandbox**:
   0.687 ± 0.001 — the lowest seed variance any variant has shown anywhere.

What's refuted:

- The ≥2× sample-efficiency target. Best observed N₅₀ ratio: 1.42–1.50 (WEAK SIGNAL territory).
- Compounding-with-scale in raw accuracy. Advantage shrinks from +14pp to +1pp going N=64→1024.

What's interesting but more nuanced than expected:

- *Generalization stability* compounds with scale even when *raw accuracy* doesn't. The seed-σ
  drop from 0.022 (helical at N=1024) to 0.001 (toroidal_coupled_wide at N=1024) is a different
  kind of compounding — not in the lab plan's wording but a legitimate finding.

---

## Context

The hypothesis under test came from "Symphony of the Helix" / Vortex Geometrodynamic intuitions,
formalized as Spiral Manifold Mapping (SMM): four PyTorch modules implementing
`ζ_i = r_i · exp(j(θ_i + ω_i z_i))` (Module 1), the helical-deviation metric `ds² = dr² + r²(dφ - ω dz)² + dz²` (Module 2),
a complex-valued linear layer (Module 3), and a phase-coherence interference gate (Module 4).

Empirical question: *can geometric/resonant inductive biases beat brute-force scaling on data efficiency in oscillator-based reasoning networks?*

Reference architecture: a minimal AKOrN-style backbone with a learnable Kuramoto layer over an
81-cell Sudoku grid, trained at 8K–14K parameters on synthetic Sudoku puzzles.

---

## Methodology summary: eight probes, chronologically

Each probe was designed to falsify (or constrain) some claim derivable from the lab plan.
Successive probes informed by previous results.

| # | Probe | Variable tested | Result |
|---|---|---|---|
| 1 | **Encoder-only smm** | HelicalEmbedding at input only | +15% N₅₀ at small N (lr=3e-3); evaporates with proper LR (lr=1e-2) |
| 2 | **smm_amp** | + amplitude in readout | Worse everywhere. Dead end. |
| 3 | **Scale-up (8K→13K)** | Does encoder-only advantage compound with capacity? | Advantage shrinks; gap-ratio inverts |
| 4 | **OOD multi-base** | Does smm transfer better? | Apparent gap-ratio advantage at small capacity is a low-capacity artifact |
| 5 | **Path B (smm_full)** | Full helical body (RotationalHelicalLinear + PhaseCoherenceGate) | Underperforms baseline at default LR |
| 6 | **Path F (LR scout)** | Is Path B failure LR-driven? | Yes; at lr=1e-2 smm_full ties baseline (no longer fails) |
| 7 | **Path G (test-time compute)** | Do AKOrN-style techniques (more steps, energy voting) differentially help SMM? | No (0/36 comparisons support); softmax-averaging helps all variants uniformly |
| 8 | **Path J (HelicalCoupledKuramoto)** | Does Module 2 (the metric), wired into coupling, help? | **Yes**, +14/+7/+1 pp on in-dist at N=64/256/1024, robust to base-seed change. First positive. |
| 8a | **Path J reproducibility** | Does the win persist across base puzzle pools? | Yes, with quantitative shifts but identical direction |
| 8b | **Path J toroidal extension** | Does T² topology beat S¹? | Less raw accuracy, better gap-ratios, best OOD-difficulty at N=1024 |
| 8c | **Path J optim investigation** | Why is toroidal_coupled fragile at small N? | Hypothesis A (budget) confirmed, B (LR warmup) refuted; needs 60ep at N=64 |
| 8d | **Path J budget-corrected** | What does toroidal_coupled look like properly trained? | Helical wins absolute at N=64; toroidal still wins gap-ratio and N=1024 OOD |

A single metric to track across probes was useful: **N₅₀ ratio = baseline_N₅₀ / candidate_N₅₀**
(candidate's sample-efficiency advantage to reach baseline's max accuracy). Anything > 1.10 is
"WEAK SIGNAL"; > 2.0 is the lab plan's claim. We landed in WEAK SIGNAL.

---

## Robust findings, with effect sizes

### Finding 1 — Helical metric coupling provides sample efficiency

**Claim:** Replacing the free 81×81 K matrix in Kuramoto with a helical-distance-derived kernel
`K_ij = exp(-ds²_ij / σ²)` (where positions are learnable per-cell `(r_i, φ_i, z_i)`) improves
in-distribution sample efficiency at small N, with non-overlapping std bands.

**Evidence:**

| | base 42 (Path J) | base 137 (repro) |
|---|---|---|
| baseline at N=64 in_dist | 0.307 ± 0.033 | 0.311 ± 0.016 |
| helical_coupled at N=64 in_dist | 0.447 ± 0.034 | 0.405 ± 0.029 |
| Δ | **+14.0pp** | **+9.4pp** |

**Effect size, multi-N:** +14.0 / +7.4 / +1.1 pp going from N=64 to N=1024 on in-dist (base 42).
The advantage shrinks with data — consistent with "good inductive bias when data is scarce" rather
than "compounding prior."

**Robustness:** Direction (helical wins) is invariant to base-seed change; magnitude varies by
~5pp. N₅₀ ratios consistently in WEAK SIGNAL range (1.26–1.50).

**Mechanism (from diagnostic):** The learned coupling matrix `K[i,j]` shows clear hierarchy
matching Sudoku semantics. At 30 epochs at N=256:

| relation | mean K | vs non-peer |
|---|---|---|
| same row only | 0.453 | 1.52× |
| same col only | 0.450 | 1.51× |
| same box only | 0.354 | 1.19× |
| non-peer | 0.298 | 1.00× |

Peer/non-peer ratio = 1.50× ("MODERATE structural learning"). The model uses the helical metric
to recover row/col/box structure as a soft prior on coupling.

### Finding 2 — Topology determines bias-variance regime

**Claim:** Single-helix and toroidal positions produce systematically different trade-offs.
Single-helix accelerates training-distribution fitting at the cost of generalization gap; torus
constrains fitting and produces tighter generalization gap and seed variance.

**Evidence at N=64, 60 epochs (proper budget):**

| | in_dist | OOD-diff | gap (in_dist − ood_diff) | gap-ratio vs baseline |
|---|---|---|---|---|
| baseline | 0.319 ± 0.034 | 0.254 ± 0.008 | 0.065 | 1.00× |
| helical_coupled | **0.526 ± 0.010** | **0.292 ± 0.007** | 0.234 | 3.62× |
| toroidal_coupled | 0.394 ± 0.025 | 0.244 ± 0.011 | 0.150 | **2.32×** |
| toroidal_coupled_wide | 0.481 ± 0.041 | 0.285 ± 0.014 | 0.196 | 3.03× |

Helical is the absolute winner across in_dist, OOD-base, OOD-difficulty. But its overfitting
ratio is 3.62× baseline. Toroidal_coupled has the lowest gap-ratio (2.32×) at the cost of ~13pp
absolute accuracy.

**Mechanism:** Toroidal optimization landscape shows a long high-loss plateau (epochs ~5–24)
followed by a sudden phase transition; helical decreases monotonically from the start. The
toroidal architecture is *intrinsically harder* to fit, which both costs raw accuracy and limits
overfitting. This is consistent with the second angular axis adding constraint that the optimizer
must satisfy before useful training begins.

**Refuted hypothesis under this finding:** Initial expectation was that the torus would *naturally
factorize* the (row, col) structure. The diagnostic shows it doesn't — toroidal_coupled clusters
cells strongly by row (cluster score 0.620, "STRONG") but barely at all by column (0.979, "no
clustering"). The two angular axes are used redundantly for row, with column information encoded
along other dimensions. The reduced-overfitting effect comes from the geometric *constraint* of
the torus, not from any clean spatial factorization.

### Finding 3 — Variance compounds with scale even when accuracy doesn't

**Claim:** At large N, structural priors produce *more reliable* outputs across seeds, even when
they don't produce more accurate ones.

**Evidence at N=1024, 30 epochs, OOD-difficulty:**

| | mean | std |
|---|---|---|
| baseline | 0.661 | 0.016 |
| helical_coupled | 0.674 | 0.022 |
| toroidal_coupled | 0.676 | 0.014 |
| **toroidal_coupled_wide** | **0.687** | **0.001** |

The 0.687 ± 0.001 result is the cleanest single number anywhere in the sandbox. Three random
seeds, three near-identical answers (0.687, 0.686, 0.688). This is qualitatively different from
the typical std of 0.01–0.03 that all other variants produce.

**Interpretation, tentative:** The structural prior + sufficient data + sufficient capacity
(toroidal_coupled_wide is +12% wider than helical_coupled) appears to converge to a unique
attractor in the loss landscape. This is a candidate explanation that needs follow-up work to
confirm — likely via repeated training runs or learning-rate / init perturbations to test whether
the basin is robust to those changes.

---

## What's been refuted

1. **≥2× sample-efficiency target.** Best observed N₅₀ ratio after eight probes: 1.50.
   Real positive, but not the headline.

2. **Compounds with scale (in raw accuracy).** Effect direction is consistently *opposite*:
   advantage shrinks from +14pp at N=64 to +1pp at N=1024. The "structural priors compound" thesis
   from the lab plan does not survive at the scales we tested.

3. **The "compositional advantage at small N" reading from probe 4.** When measured with proper
   capacity and proper LR, the small-N gap-ratio advantage of encoder-only smm disappears or
   inverts. It was a low-capacity artifact.

4. **Path B's full helical body as a clear loss.** Path F's LR scout showed Path B's apparent
   underperformance was hyperparameter-driven; smm_full ties baseline at the right LR. This was
   a partial rescue: smm_full no longer *fails*, but it also doesn't *win*. The helical body
   contributes nothing measurable beyond what HelicalEmbedding provides at the input.

5. **AKOrN-style test-time techniques as a hidden lever for SMM** (Path G). 0/36 comparisons
   showed differential lift for SMM variants. Test-time compute helps everyone or no one;
   structural priors don't get a special multiplier from longer Kuramoto iteration.

---

## The bias-variance characterization

We've ended up with a substantive empirical picture: three regimes, three winners, principled
trade-off. Worth keeping as a reusable framing.

| Regime | Best variant | Why |
|---|---|---|
| Small N, raw accuracy | **helical_coupled** | Strongest prior, fits hardest. 0.526 at N=64, 60 ep. |
| Small N, generalization gap-ratio | **toroidal_coupled** | Topology constrains overfitting. 2.32× vs helical's 3.62× at N=64. |
| Large N, OOD-difficulty (mean + reliability) | **toroidal_coupled_wide** | 0.687 ± 0.001 at N=1024 is the cleanest number in the sandbox. |

This isn't the "SMM dominates everywhere" outcome the lab plan was set up for. It's a more
honest characterization of *what each variant is good for* — a useful contribution to the
oscillator-network literature even if it's modest.

---

## Limitations (be honest)

- **Synthetic task.** All puzzles generated from BASE_SOLVED via row/col/digit-permutation
  symmetries. Real-world Sudoku, ARC-AGI, or general reasoning may behave differently.
- **Small scale.** ~14K parameters. The compounding-with-scale question is genuinely *open at
  larger scale*; we tested at one capacity tier.
- **Single architecture family.** Minimal AKOrN-style backbone. The actual AKOrN paper achieves
  90% with multi-layer attention coupling and energy-aware readout that we don't have.
- **Sudoku-specific peer structure.** The coupling K matrix wants to learn (row, col, box) peer
  adjacency. That's a 1620-edge sparse pattern in an 81-node graph. Whether structural priors
  help on tasks with *less* clean peer structure is not tested.
- **No comparison against hand-crafted baseline.** We never asked: *would a fixed, hand-crafted
  Sudoku peer-adjacency K matrix reach 0.687 at N=1024 too?* If yes, the SMM win is "we
  successfully recovered the obvious answer," not "we discovered a new structure." This is a
  critical control we owe ourselves.

---

## Future work, prioritized by cost-effectiveness

| # | Direction | Cost | Information value | Notes |
|---|---|---|---|---|
| 1 | **Hand-crafted peer-adjacency baseline** | 2 hrs | very high | Decisive control. Distinguishes "structure recovered" from "structure discovered." Required before any further investment. |
| 2 | **Mix mode diagnostic** (HelicalCoupled + free K with learnable α) | 1 day | high | The optimizer's chosen α at small vs large N tells us *where* the prior helps. We have the implementation in `akorn_smm_sudoku.py`; just need to run sweeps. |
| 3 | **Conformal metric** (`ds² = λ(z) · [dr² + r²(dφ - ω dz)² + dz²]` with learnable scale function `λ(z)`) | 3–5 days | moderate | Lets the helix expand/contract. May address the small-N overfitting if the model can use less prior at large N. |
| 4 | **Multi-scale helical cascade** / polyatomic ω (each channel has multiple ω at different scales) | 1 week | moderate | Nested-frequency variant. Adds expressivity but adds parameters too. |
| 5 | **Input-conditioned coupling** (`cell_(r,α,β,z)` derived from puzzle, not just learned per-cell-fixed) | 3–5 days | high | Currently the coupling K is puzzle-independent. Making it puzzle-aware is a qualitative change in what the prior represents. |
| 6 | **Multi-step readout supervision** (train readout against trajectory states, not just final) | 1 week | high | Would unlock proper energy-based voting (Path G's failed approach), potentially closing gap to AKOrN's published 90% on Sudoku. |
| 7 | **Real AKOrN backbone integration** (multi-layer attention coupling J=attn, energy-trained readout) | 2–3 weeks | very high | The architecture comparison we never made. Decisive for whether SMM helps in production-scale oscillator networks or only in our minimal version. |
| 8 | **ARC-AGI-2 integration** | 1–2 months | very high (if signal) | The benchmark the lab plan was built around. Major architectural change (variable grids, few-shot meta-learning) but the test that matters most. |
| 9 | **GPU scale-up via Runpod** | $$ + weeks | low until upstream work done | Eight probes argue against scaling helping the *raw-accuracy* story. Reserve for after either ARC-AGI-2 shows signal or AKOrN-backbone shows compounding. |

The order matters. Items 1 and 2 are cheap and constrain the next steps significantly. Items 3–6
are architectural variations of the existing approach. Items 7–9 are larger commitments that
should only happen if the cheap diagnostics motivate them.

**Most under-invested item:** #1 (hand-crafted peer baseline). Costs almost nothing, resolves a
fundamental interpretive question. Should be the next experiment if any further work happens
on this sandbox.

---

## Open questions (meta-level)

These aren't experiments per se; they're things the data raises that I don't yet know how to
fully answer:

1. **Why is `toroidal_coupled_wide` deterministic at N=1024 OOD-difficulty?** σ=0.001 across
   three seeds is unusual. Hypothesis: a unique attractor in the optimization landscape. Test:
   five more seeds, see if the variance grows or stays at noise floor; perturb init scale and
   see if the basin survives.

2. **Why does helical_coupled overfit *more* with longer training (gap-ratio 3.4× → 5.5× going
   from 30 to 60 epochs) while baseline doesn't?** The helical coupling kernel's implicit
   regularization effect appears time-dependent. Possible mechanism: the learnable per-cell
   positions specialize too aggressively when given more updates, encoding training-distribution
   particulars rather than general Sudoku structure. Diagnostic: run the position-clustering
   analysis at 30 vs 60 epochs and see if the cluster scores tighten beyond what's useful.

3. **Why does adding the helical input encoder *to* helical-coupled coupling not help (Path J's
   smm_helical_coupled was no better than helical_coupled alone, sometimes worse)?** Two helical
   priors should compose, but they don't. Plausible mechanism: the encoder's `.angle()` extracts
   phases that the coupling wants to compute distances over directly, but the coupling uses
   *fixed* per-cell positions independent of those phases — the priors are operating in
   incompatible coordinate systems and effectively interfere.

4. **Was Path G's energy voting fundamentally broken or just architecturally mismatched to our
   readout?** The published AKOrN result (51% → 90% with energy voting) requires multi-step
   readout supervision we don't have. If Future Work #6 succeeds, we can revisit this.

5. **What is the "right" benchmark for this kind of structural prior?** Sudoku has an unusually
   clean peer-adjacency structure that any reasonable inductive bias might recover. ARC-AGI-2
   has variable grids and few-shot structure that may break our coupling-matrix paradigm
   entirely. Real reasoning tasks (math word problems, code completion, etc.) don't have a
   natural cell decomposition. The sandbox approach worked for diagnosis but the *positive*
   findings need a benchmark where they could plausibly matter.

---

## Decision points for next session

- [ ] Run hand-crafted peer-adjacency baseline (item #1) — 2 hours, decisive
- [ ] Run mix-mode diagnostic (item #2) — 1 day, high info value
- [ ] Decide based on (1)+(2) whether to push on architectural variants or pivot to ARC integration
- [ ] If pivoting: scope ARC-AGI-2 architecture (3–5 day exploration before commitment)
- [ ] If staying: pick one of conformal / multi-scale / input-conditioned and run it

Until those decision points: this is a coherent stopping place. The findings are documented,
the dead ends are marked, the future work is prioritized. We can pick this up cleanly later.

---

## Appendix: variant naming reference

For self-reference six months from now:

| Name | What it is |
|---|---|
| `baseline` | AKOrNBaseline: stock minimal AKOrN. embed → GELU(Linear) → Linear(phases) → KuramotoLayer → readout |
| `smm` | AKOrNWithSMM: HelicalEmbedding at input only, otherwise baseline |
| `smm_amp` | smm + amplitude in readout. Dead end. |
| `smm_full` | AKOrNWithSMMFull: HelicalEmbedding + 2× (RotationalHelicalLinear + PhaseCoherenceGate) blocks before phase extraction |
| `helical_coupled` | AKOrNHelicalCoupled: baseline encoder + HelicalCoupledKuramoto (K matrix derived from learnable single-helix per-cell positions via Module 2) |
| `smm_helical_coupled` | helical encoder + helical coupling. Doesn't compose well. |
| `toroidal_coupled` | AKOrNToroidalCoupled at n_helical_channels=12 (matched-param to helical_coupled) |
| `toroidal_coupled_wide` | AKOrNToroidalCoupled at n_helical_channels=16 (matched-channel to helical_coupled, +12% params) |
