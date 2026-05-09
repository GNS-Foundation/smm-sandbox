# smm-sandbox

**Spiral Manifold Mapping for AKOrN-style oscillatory networks.**

A research sandbox for testing whether enforcing helical / oscillatory structure in latent space yields **parameter compression** on reasoning tasks (Sudoku, ARC-AGI-2) when injected into an AKOrN backbone.

## The empirical question

Holding parameter count and training compute constant, does an AKOrN backbone augmented with Spiral Manifold Mapping reach a target performance threshold $P^*$ using strictly fewer training samples than:

1. Plain AKOrN
2. A parameter-matched dense Transformer baseline

If yes, by how much, on which task families, and does the advantage persist with scale?

See [`docs/01_deep_research_brief.md`](docs/01_deep_research_brief.md) for the full framing and [`docs/02_lab_design_and_execution.md`](docs/02_lab_design_and_execution.md) for the experimental program.

## What's here

```
smm-sandbox/
├── src/smm/
│   ├── __init__.py
│   └── modules.py          # 4 PyTorch modules implementing SMM
├── tests/
│   └── test_modules.py     # pytest unit tests
├── experiments/            # (empty — training scripts go here)
├── notebooks/              # (empty — exploration notebooks)
└── docs/
    ├── 01_deep_research_brief.md
    └── 02_lab_design_and_execution.md
```

## The four modules

| Module | Replaces | Role |
|--------|----------|------|
| `HelicalEmbedding` | input embedding | Maps $v \in \mathbb{R}^n$ to $n/3$ complex helical channels $\zeta_i = r_i e^{j(\theta_i + \omega_i z_i)}$ |
| `HelicalDeviationMetric` | cosine similarity | Pairwise distance under $ds^2 = dr^2 + r^2(d\phi - \omega \, dz)^2 + dz^2$ |
| `RotationalHelicalLinear` | `nn.Linear` | Complex-valued linear layer; optional unitary constraint |
| `PhaseCoherenceGate` | `ReLU` | Interference-based gate: $g = \|\zeta + c\| / (\|\zeta\| + \|c\|)$ |

## Quickstart

```bash
# Clone and install (editable)
git clone https://github.com/GNS-Foundation/smm-sandbox.git
cd smm-sandbox
pip install -e .

# Run smoke test
python -m smm.modules

# Run unit tests
pytest tests/ -v
```

Expected smoke test output:
```
=== SMM smoke test ===
Input v        : shape=(2, 5, 12), dtype=torch.float32
After embed    : zeta (2, 5, 4) torch.complex64, z (2, 5, 4) torch.float32
After RH linear: (2, 5, 4) torch.complex64
After gate     : (2, 5, 4) torch.complex64
Distance ds    : (2, 5, 5, 4) torch.float32
Diagonal max   : 1.00e-06  (should be ~0)
[OK] Pipeline runs forward + backward end-to-end.
```

## Roadmap

- [x] Phase 0: SMM modules + unit tests
- [ ] AKOrN backbone integration (encoder-only injection first)
- [ ] Sudoku-9×9 sample-efficiency curves: AKOrN vs AKOrN+SMM vs Transformer
- [ ] ARC-AGI-2 small subset evaluation
- [ ] Parameter-compression ratio measurement at matched $P^*$
- [ ] Scale-up: 10M → 50M → 100M params
- [ ] Full replacement variant (SMM throughout the stack, not just encoder)

## Falsifiable claim

> AKOrN+SMM at parameter count $P$ matches AKOrN-baseline at parameter count $P/k$ on Sudoku-9×9 and ARC-AGI-2, for $k \geq 2$, at matched training compute and data.

Pre-registered failure condition: if AKOrN+SMM does not beat plain AKOrN by a margin larger than seed-to-seed variance on at least one Tier 1 benchmark, SMM is reformulated rather than scaled up.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (for full complex autograd support)
- pytest (for unit tests)

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this sandbox contributes to published work, please cite:

```bibtex
@software{smm_sandbox_2026,
  title  = {smm-sandbox: Spiral Manifold Mapping for oscillatory neural networks},
  year   = {2026},
  url    = {https://github.com/GNS-Foundation/smm-sandbox}
}
```

And cite the underlying AKOrN paper:

```bibtex
@inproceedings{miyato2025akorn,
  title     = {Artificial Kuramoto Oscillatory Neurons},
  author    = {Miyato, Takeru and L\"owe, Sindy and Geiger, Andreas and Welling, Max},
  booktitle = {ICLR},
  year      = {2025}
}
```
