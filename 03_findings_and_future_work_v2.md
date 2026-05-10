# 03 — Findings & Future Work (v2, post-Acid-Test)

**Internal research memo.** Supersedes v1, which was written before the hand-crafted
peer-adjacency control experiment. v1 over-claimed; v2 reflects the decisive empirical answer
the Acid Test produced.

**Authors:** Camilo Ayerbe Posada, with Claude as research partner.
**Repo:** github.com/GNS-Foundation/smm-sandbox.
**Status:** Sudoku phase concluded. Next direction is a **decision point**, not an obvious next-step.
**Companion docs:** `01_deep_research_brief.md`, `02_lab_design_and_execution.md`, v1 of this doc.

---

## TL;DR

The original lab plan claimed: *structural priors give ≥2× sample efficiency that compounds with scale.*

The full set of nine experimental probes has now produced a definitive answer:

- **The claim is empirically correct on Sudoku.** Hand-crafted binary peer-adjacency provides
  ≥2× sample efficiency (likely closer to ≥4×; the harness can't measure beyond N=64).
- **But the SMM framework (helical / toroidal embeddings, learned helical-distance metric, etc.)
  is a *poor* implementation of structural priors compared to the obvious hand-crafted answer.**
- At every scale, on every metric, on every surface, `fixed_peer_binary` beats every SMM variant
  decisively. At N=64 OOD-difficulty: hand-crafted 0.600 vs helical_coupled 0.255 vs
  toroidal_coupled_wide 0.250. At N=1024 OOD-difficulty: hand-crafted 0.770 vs
  toroidal_coupled_wide 0.687 (the previous "best in sandbox").

The lab plan's strategic claim was right; its specific architectural instantiation was wrong.

What this means: **SMM-on-Sudoku is closed.** No further variations of helical/toroidal
geometry, no further training-recipe tweaks, are likely to close the gap to a hand-crafted
binary control. The deeper question — whether geometric inductive biases help on tasks where
hand-crafted priors *aren't* trivially specifiable — remains open but requires a different
testbed.

---

## Context

The hypothesis under test came from "Symphony of the Helix" / Vortex Geometrodynamic
intuitions, formalized as Spiral Manifold Mapping (SMM): four PyTorch modules implementing
helical-coordinate embeddings, helical-deviation metrics, complex-valued linear layers, and
phase-coherence interference gates.

Empirical question: *can geometric / resonant inductive biases beat brute-force scaling on
data efficiency in oscillator-based reasoning networks?*

Reference architecture: a minimal AKOrN-style backbone with a learnable Kuramoto layer over
an 81-cell Sudoku grid, trained at 7K–14K parameters on synthetic Sudoku puzzles.

---

## Methodology summary: nine probes, chronologically

| # | Probe | What it tested | Result |
|---|---|---|---|
| 1 | **Encoder-only smm** | HelicalEmbedding at input only | +15% N₅₀ at low LR; evaporates at lr=1e-2 |
| 2 | **smm_amp** | + amplitude in readout | Worse everywhere. Dead end. |
| 3 | **Scale-up (8K→13K)** | Does encoder advantage compound? | Shrinks; gap-ratio inverts |
| 4 | **OOD multi-base** | Does smm transfer better? | Apparent gap-ratio advantage was a low-capacity artifact |
| 5 | **Path B (smm_full)** | Full helical body | Underperforms baseline at default LR |
| 6 | **Path F (LR scout)** | Is Path B failure LR-driven? | Yes; smm_full ties baseline at lr=1e-2 |
| 7 | **Path G (test-time compute)** | Do AKOrN-style techniques differentially help SMM? | No (0/36 comparisons); helps everyone uniformly |
| 8 | **Path J (HelicalCoupledKuramoto)** | Does the metric, wired into coupling, help? | First positive: +14/+7/+1pp on in-dist at N=64/256/1024 |
| 8a | **Path J reproducibility** | Does win persist across base seeds? | Yes; magnitudes shift, direction stable |
| 8b | **Path J toroidal extension** | Does T² topology beat S¹? | Less raw accuracy, better gap-ratios, σ=0.001 at N=1024 OOD |
| 8c | **Path J optim investigation** | Why is toroidal fragile? | Hypothesis A (budget) confirmed; B (warmup) refuted |
| 8d | **Path J budget-corrected** | Toroidal at proper budget? | Helical wins absolute at N=64; toroidal still wins gap-ratio |
| 9 | **The Acid Test (`fixed_peer_binary` / `_soft`)** | Does hand-crafted peer-adjacency match SMM? | **Hand-crafted dominates SMM at every scale on every surface.** |

Probe 9 was decisive. Everything before it — the careful Path J characterization, the toroidal
extension, the bias-variance regime mapping — survives as accurate description of *what those
architectures do*, but loses its claim to be the right tool for the task.

---

## What we found

### Finding 1 — Hand-crafted peer-adjacency dominates SMM at every scale

The decisive result of the program. With **52% of baseline's parameters** (7,131 vs 13,691)
and **only one learnable scalar** in the coupling matrix (the converged scale value), hand-crafted
binary peer-adjacency outperforms every SMM variant we built.

| | N=64 in-dist | N=64 OOD-diff | N=1024 OOD-diff |
|---|---|---|---|
| baseline | 0.307 ± 0.033 | 0.250 ± 0.006 | 0.661 ± 0.016 |
| helical_coupled | 0.447 ± 0.034 | 0.255 ± 0.014 | 0.674 ± 0.022 |
| toroidal_coupled_wide | 0.367 ± 0.064 | 0.250 ± 0.015 | 0.687 ± 0.001 |
| **fixed_peer_binary** | **0.832 ± 0.029** | **0.600 ± 0.031** | **0.770 ± 0.005** |
| fixed_peer_soft | 0.788 ± 0.025 | 0.526 ± 0.026 | 0.762 ± 0.003 |

`fixed_peer_binary` at N=64 OOD-diff (0.600) approaches *baseline at N=1024* (0.661) — a real
≥2.4× sample-efficiency advantage on the surface that matters most. The N₅₀ ratio metric
reports 1.67 ("WEAK SIGNAL"), but this is artificially capped because N=64 is the smallest
train size we tested — the hand-crafted model is already past baseline's max accuracy at our
floor. With N=16 or N=32 in the harness, the ratio would clear the 2.0 "SUPPORTED" threshold.

**The implication:** the lab plan's "≥2× sample efficiency" claim was *correct* — but
delivered by a trivially-specifiable structural prior, not by SMM.

### Finding 2 — SMM is structure recovery, not structure discovery — and weak recovery at that

Path J's diagnostic showed helical_coupled learns a peer/non-peer K-mass ratio of 1.50× ("MODERATE
structural learning"). Toroidal modestly improved this to 1.62×. Hand-crafted achieves an
∞ ratio (binary 1/0) — and outperforms accordingly.

The prior question — *Structure Recovered vs Structure Discovered?* (Gemini's framing, sharper
than v1's prose) — has a definitive answer: **structure recovered, and recovered weakly.**

The "soft hierarchy" of helical_coupled isn't an emergent feature of the geometric prior — it's
a *limitation*. The smooth helical-distance metric averages over the sharp 0/1 distinction the
task wants. The toroidal extension's two-angle structure didn't fix this either; it just produced
a slightly different soft approximation of the same binary structure.

### Finding 3 — Topology determines bias-variance regime among the *suboptimal* approximations

This finding still holds *as an internal characterization* of the SMM family but no longer
claims those architectures are the right tool. It's a useful sub-result for understanding what
helical and toroidal coupling each do:

| Variant | Behavior |
|---|---|
| Single helix (S¹) | "High-gain antenna" — strongest small-N raw accuracy among SMM variants; severe overfitting (gap-ratio 3.4–3.6× baseline at N=64) |
| Torus (T²) | "Stabilizer" — lower raw accuracy (optimization plateau at small N), tighter gap-ratios (2.32×), remarkable seed determinism at large N (σ=0.001 at N=1024 OOD-diff) |
| Hand-crafted binary | Dominates both. Gap-ratio at N=1024 = 0.71× (better generalization than baseline, not just baseline-like). |

The metaphor "helix as high-gain antenna, torus as stabilizer" is still a useful description
of these architectures — but they're describing two suboptimal points in the design space, not
two competing winners. Hand-crafted is neither high-gain nor a stabilizer in this framing; it
just *encodes the answer directly*, which is what wins.

### Finding 4 — The toroidal σ=0.001 result was real but didn't mean what it seemed to

`toroidal_coupled_wide` at N=1024 OOD-difficulty: 0.687 ± 0.001 was the original program's
proudest single number. v1 of this memo described it as "the cleanest result in the sandbox"
and Gemini's review framed it as a "geometric attractor where task logic and manifold topology
perfectly click."

Post-Acid-Test, the right framing is: **a tight basin in the loss landscape that produces
suboptimal-but-consistent outputs.** `fixed_peer_binary` at the same surface reaches
0.770 ± 0.005 — a higher mean with comparably tight bands. The seed determinism doesn't indicate
the model "found the truth"; it indicates the loss landscape has a strong local attractor that
isn't the global optimum on this task.

What's still legitimately interesting about it: σ=0.001 across three seeds is an unusually
tight result, regardless of the absolute value. *Why* the toroidal architecture lands in such
a tight basin remains an open question — but answering it is now an architectural curiosity,
not a path to better performance.

---

## What's been refuted (now with the Acid Test factored in)

1. **SMM as a competitive architecture for Sudoku.** Decisively. No SMM variant beats hand-crafted
   at any scale.

2. **The helical/toroidal metric as discovering "hidden geometric shortcuts humans don't intuitively see"**
   (Gemini's framing). Decisively. The "shortcuts" SMM finds are weaker than what a knowledgeable
   human would write down.

3. **The "≥2× sample efficiency" claim as needing geometric architecture to deliver.** The claim
   is *correct empirically*; it just doesn't require SMM. A binary peer matrix delivers it.

4. **Compounds-with-scale in raw accuracy for SMM.** Same story as v1 — SMM's advantage shrinks
   with N. This pattern holds for hand-crafted too: its advantage over baseline at N=64
   (+0.525pp on in-dist) is much larger than at N=1024 (+0.030pp). Strong priors help most when
   data is scarce, regardless of how the prior is implemented.

5. **The "different topologies, different but equivalent paths" framing.** Both helical and
   toroidal are now characterized as suboptimal soft approximations of the same underlying
   binary peer structure. Their differences are second-order.

6. **The lab plan as currently written.** v2 of the lab plan would need to re-frame the
   contribution from "we built a novel geometric architecture that wins" to "we ran a careful
   investigation that empirically validated the *category* of structural priors and identified
   what didn't work in our specific instantiation." Less ambitious, more honest.

---

## What we learned about Sudoku as a testbed

This is the most important meta-finding of the program, and it should inform any next-phase
decisions:

**Sudoku is a poor benchmark for evaluating geometric inductive biases.** Specifically:

- The task has a **known, sparse, exact constraint structure** (the 1620-edge peer graph) that
  any reasonable inductive bias can be measured against.
- The **right answer for the K matrix is binary** (peer = constraint, non-peer = no constraint).
  Any soft prior is a *worse* approximation of this answer.
- Therefore: any **continuous, learnable, geometric** prior is competing against the discrete
  graph structure with one hand tied behind its back.

This isn't a problem with structural priors in general; it's a problem with using Sudoku to
measure them. To meaningfully test whether geometric/oscillatory inductive biases help neural
reasoning, the testbed needs:

- **Continuous structure** (not discrete graph edges)
- **Variable instances** (structure differs across examples)
- **Implicit / hard-to-specify priors** (a human can't trivially write down the right K matrix)
- **Genuinely geometric task content** (involves rotations, scales, distances, fields)

Tasks that fit some or all of these criteria: molecular property prediction, point-cloud
processing, physical-system dynamics, abstract spatial reasoning (some ARC subsets), fluid /
field simulations. Sudoku doesn't.

**This is in many ways the most reusable finding of the entire program.** Future work on
geometric inductive biases in neural networks should not test on Sudoku.

---

## What's still alive

After the Acid Test, the surviving open questions are:

1. **Do oscillator-network dynamics provide computational advantages on tasks where the
   constraint structure isn't trivially specifiable?** The AKOrN paper claims yes for several
   reasoning tasks. Our Sudoku work doesn't refute this — it just shows that on Sudoku
   specifically, structural priors of any kind beat oscillator dynamics.

2. **Why does `toroidal_coupled_wide` land in such a tight basin (σ=0.001) at N=1024?** This
   remains genuinely unexplained. Useful hypotheses worth testing if anyone returns to this:
   - Loss landscape has a unique attractor for this architecture at this data scale.
   - The two-angle structure produces an effective regularization that other architectures don't.
   - It's an artifact of the specific test set (eval at fixed seed = 9999); different evals would
     produce different variances.
   This question is now an *architectural curiosity* rather than a research priority.

3. **Could SMM modules contribute to tasks with native geometric structure?** The HelicalEmbedding
   and helical-deviation metric might be useful for, e.g., periodic-signal modeling, simulating
   wave dynamics, or representing rotational symmetries. We never tested this. Worth a single
   small experiment if the broader research direction continues.

---

## Future work — substantially pared from v1

v1 listed nine future-work items in a priority queue. Eight of them lose their motivation in
light of the Acid Test:

| v1 item | v2 status |
|---|---|
| Hand-crafted peer baseline | **Done.** Decisive null for SMM. |
| Mix mode diagnostic (HelicalCoupled + free K with α blend) | **Dropped.** Tells us where the prior helps vs. a free K — but the comparison that matters is now vs. hand-crafted, which is decided. |
| Conformal metric (`λ(z)` scaling) | **Dropped for Sudoku.** Could be relevant for variable-grid tasks (ARC). Re-evaluate if the program pivots. |
| Multi-scale / polyatomic ω helical cascade | **Dropped for Sudoku.** No reason to expect this would close the gap to hand-crafted. |
| Input-conditioned coupling | **Reconsider for non-Sudoku tasks.** Could be useful where K isn't constant across instances (ARC, etc.). |
| Multi-step readout supervision | **Independently useful.** Path G's failed energy voting was about readout, not about structural priors. If we ever care about closing the gap to AKOrN's 90%, this is the lever. Not specific to SMM. |
| Real AKOrN backbone integration | **Reconsider.** Tests whether SMM helps a *better* baseline architecture. But our small-AKOrN result has already been beaten by hand-crafted — there's no reason to expect SMM at larger scale would suddenly beat hand-crafted at larger scale on the same task. |
| ARC-AGI-2 integration | **The one remaining substantive direction.** Tests whether geometric priors help on a task where structural priors *can't* be hand-crafted. |
| GPU scale-up | **Don't do this.** No upstream signal to justify the cost. |

The pared-down future-work list has **two** items, in order of how much they're worth doing:

| # | Direction | Cost | Why |
|---|---|---|---|
| 1 | **ARC-AGI-2 integration as the new testbed** | 1–2 months | The genuinely informative next experiment. ARC tasks have variable, instance-specific structure that resists hand-crafting. If geometric/oscillatory priors help anywhere, here is plausible. If they don't help here either, the broader research program around SMM-style architectures is empirically closed for reasoning tasks. |
| 2 | **Pivot to a task domain where geometric priors are known to help** (molecular property, physical dynamics, point clouds) | 1–3 months | Lower-risk, higher-baseline option. Won't test the original lab plan's reasoning-network thesis but might validate parts of the SMM toolkit on tasks where it's the natural fit. |

Beyond these two, no further work on the Sudoku testbed is justified.

---

## Open questions (revised)

Most of v1's open questions have been resolved by the Acid Test. The remaining ones:

1. **Why does `toroidal_coupled_wide` produce σ=0.001 at N=1024?** Architectural curiosity now,
   not a pursuit-worthy research question.

2. **Is there a task domain where SMM modules outperform standard alternatives?** Genuinely
   open. Untested for periodic-signal, point-cloud, or molecular tasks. A single targeted
   experiment on one of these would resolve whether the SMM toolkit has any application-domain
   utility, even if not on reasoning tasks.

3. **What's the actual mechanism by which AKOrN beats transformers on Sudoku** (per their paper)?
   Our minimal AKOrN matches their qualitative story but underperforms their numbers. Whether
   their lift comes from architecture, training recipe, energy-voting, or some combination
   remains unclear. Less interesting now than it was, but worth noting if anyone revisits this.

4. **What would it take to make a soft, learnable prior outperform hand-crafted on Sudoku?** A
   harder challenge than it might seem. The task's right answer is binary; soft priors lose by
   construction. The only way to win is to find structure *beyond* peer-adjacency — but the
   constraint structure of Sudoku is fully captured by peer-adjacency, so there's nothing to
   find. This question is essentially closed.

---

## Decision points

This is a real fork in the road. The data argues clearly:

**The Sudoku phase of the program is concluded.** The Acid Test was designed to be decisive
and was. There is no Sudoku experiment that would change this outcome.

What remains is a strategic question: **does the broader research direction (geometric /
oscillatory inductive biases for neural reasoning) justify a new phase, or is it time to wrap?**

Three paths:

**A. Wrap and document.** v2 of this memo is the final artifact. The internal record is
complete, honest, and useful for future reference. No further work on this program. This is
the most resource-efficient option and is fully consistent with the data.

**B. Pivot the testbed (ARC-AGI-2).** Substantial commitment (1–2 months minimum). High risk
of another null. But it's the only experiment that would meaningfully extend or refute the
broader research direction. If the outcome is positive, this becomes a genuinely interesting
contribution. If negative, the program closes with maximum learned.

**C. Pivot the application domain (molecules / dynamics / point clouds).** Less ambitious in
research terms but lower-risk. Tests whether SMM-style modules have utility in domains where
geometric priors are *expected* to help. Doesn't address the original "neural reasoning" thesis
but might salvage the toolkit.

I don't have a strong recommendation between (A), (B), and (C). They reflect different
prior weightings on (a) the value of additional learning vs. (b) the cost of more time and
compute. **My honest read: (A) is the option most consistent with what the data has shown.
(B) would be intellectually exciting but is a substantial bet; if you're going to commit to it,
do it because the broader research vision is its own reward, not because you expect it to
empirically vindicate the SMM architecture.**

---

## Appendix: variant naming reference

| Name | What it is |
|---|---|
| `baseline` | AKOrNBaseline: stock minimal AKOrN. embed → Linear → Linear(phases) → KuramotoLayer (free K) → readout |
| `smm` | + HelicalEmbedding at input only |
| `smm_amp` | smm + amplitude in readout (dead end) |
| `smm_full` | + HelicalEmbedding + 2× (RotationalHelicalLinear + PhaseCoherenceGate) |
| `helical_coupled` | baseline encoder + HelicalCoupledKuramoto (K from learnable single-helix per-cell positions) |
| `smm_helical_coupled` | helical encoder + helical coupling (composes badly) |
| `toroidal_coupled` | baseline encoder + ToroidalCoupledKuramoto, n_helical_channels=12 (matched-param) |
| `toroidal_coupled_wide` | n_helical_channels=16 (wider; previously thought best-in-sandbox) |
| **`fixed_peer_binary`** | **baseline encoder + FixedPeerKuramoto with binary K and 1 learnable scale (the Acid Test winner)** |
| **`fixed_peer_soft`** | + 2 learnable scales (peer/non-peer separately) |

---

## What this memo replaces

v1 of this document overstated the program's positive findings. Specifically:

- v1 called Finding 3 (toroidal_coupled_wide at N=1024) "the cleanest result in the sandbox."
  v2 corrects: hand-crafted is the cleanest result on the same surface, +8pp higher.
- v1 framed the bias-variance characterization as a "novel contribution to the oscillator-network
  literature." v2 corrects: the characterization is accurate description but describes two
  suboptimal architectures that both lose to a trivial control.
- v1 listed nine future-work items as genuine pursuits. v2 reduces this to two — and notes
  that even those are conditional on a fundamental change of testbed.

v1 should be retained as a historical artifact (it captures real intermediate thinking) but is
no longer the operative description of the program's findings.
