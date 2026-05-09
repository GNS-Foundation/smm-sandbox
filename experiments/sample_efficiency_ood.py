"""
experiments/sample_efficiency_ood.py
=====================================
Path D: the discriminating task.

The existing sample-efficiency harness used a synthetic Sudoku dataset built
from ONE base solution + random symmetries. That makes train and test draw
from essentially the same distribution: a model that memorizes the base's
symmetry orbit "generalizes" without doing any real reasoning. Both SMM
variants tested negative on that task, but the result is suspect because the
task doesn't reward the kind of compositional structure SMM is supposed to
provide.

This harness fixes that by generating a POOL of distinct random Sudoku
solutions (via backtracking) and exposing three evaluation surfaces:

  1. in_dist    -- same base puzzles as training, same difficulty
                   (interpolation; sanity check)
  2. ood_base   -- held-out base puzzles, same difficulty
                   (structural generalization)
  3. ood_diff   -- training base puzzles, MORE blanks (harder)
                   (compositional reasoning — closest to AKOrN's headline test)

For each model variant, accuracy is reported on all three. The hypothesis we
care about: structural priors should help most on (3), then (2), perhaps
nowhere on (1). If SMM helps NOWHERE across the OOD surfaces, that's strong
evidence the inductive bias isn't paying off in this configuration.

Outputs:
    <out_dir>/results.csv             -- one row per (model, size, seed) run
                                          with three accuracy columns
    <out_dir>/sample_efficiency_ood.png -- 3-panel comparison plot

Usage:
    python experiments/sample_efficiency_ood.py
    python experiments/sample_efficiency_ood.py --seeds 5 --epochs 30

On an M-series Mac CPU, default config runs in ~3-4 minutes (slightly more
than the original harness due to the larger eval surface).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Reuse model definitions and utilities from the integration experiment
from akorn_smm_sudoku import (
    AKOrNBaseline,
    AKOrNWithSMM,
    AKOrNWithSMMAmp,
    loss_and_metrics,
    count_params,
)

# Backtracking on 9x9 is shallow (max depth 81) but be defensive
sys.setrecursionlimit(5000)


# =============================================================================
# Random Sudoku solver — generates DISTINCT base solutions
# =============================================================================

def _is_valid(grid: np.ndarray, i: int, j: int, d: int) -> bool:
    """Check if placing digit d at (i,j) violates Sudoku constraints."""
    if d in grid[i]:                           # row check
        return False
    if d in grid[:, j]:                        # col check
        return False
    bi, bj = (i // 3) * 3, (j // 3) * 3
    if d in grid[bi:bi + 3, bj:bj + 3]:        # 3x3 box check
        return False
    return True


def _solve_random(grid: np.ndarray, rng: np.random.Generator) -> bool:
    """Fill grid with a random valid Sudoku solution via backtracking.
    Mutates `grid` in-place. Returns True on success (always succeeds for
    a valid 9x9 starting state)."""
    for i in range(9):
        for j in range(9):
            if grid[i, j] == 0:
                digits = list(range(1, 10))
                rng.shuffle(digits)            # randomize digit order
                for d in digits:
                    if _is_valid(grid, i, j, d):
                        grid[i, j] = d
                        if _solve_random(grid, rng):
                            return True
                        grid[i, j] = 0
                return False
    return True


def random_solved_sudoku(rng: np.random.Generator) -> np.ndarray:
    """Generate a random valid solved Sudoku grid via backtracking."""
    grid = np.zeros((9, 9), dtype=np.int64)
    ok = _solve_random(grid, rng)
    if not ok:
        raise RuntimeError("Sudoku generation failed — should be impossible")
    return grid


def apply_random_symmetries(base: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """Apply random Sudoku-preserving symmetries to a solved grid.

    Symmetries:
      1. Random digit permutation (1..9 -> permutation of 1..9)
      2. Permute rows within bands + permute the bands themselves
      3. Permute cols within stacks + permute the stacks themselves
    """
    grid = base.copy()

    # 1. Digit permutation
    perm = rng.permutation(9) + 1
    digit_map = np.zeros(10, dtype=np.int64)
    for i in range(9):
        digit_map[i + 1] = perm[i]
    grid = digit_map[grid]

    # 2. Row permutation (within bands + bands themselves)
    band_order = rng.permutation(3)
    row_perm = []
    for b in band_order:
        rows_in_band = (rng.permutation(3) + b * 3).tolist()
        row_perm.extend(rows_in_band)
    grid = grid[row_perm]

    # 3. Column permutation (within stacks + stacks themselves)
    stack_order = rng.permutation(3)
    col_perm = []
    for s in stack_order:
        cols_in_stack = (rng.permutation(3) + s * 3).tolist()
        col_perm.extend(cols_in_stack)
    grid = grid[:, col_perm]

    return grid


def make_puzzle(solved: np.ndarray, n_blanks: int,
                rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Mask n_blanks cells of a solved grid (with 0) to create a puzzle."""
    puzzle = solved.copy()
    flat_idx = rng.permutation(81)[:n_blanks]
    for idx in flat_idx:
        puzzle[idx // 9, idx % 9] = 0
    return puzzle, solved


# =============================================================================
# OOD Sudoku dataset
# =============================================================================

class SudokuDatasetOOD(Dataset):
    """Sudoku dataset with explicit base-pool control for OOD splits.

    Each sample = (puzzle, solution) pair, where:
      - solution is a random symmetry of a randomly-chosen base from base_pool
      - puzzle is solution with n_blanks cells masked
    """

    def __init__(self, n_samples: int, base_pool: List[np.ndarray],
                 n_blanks: int = 20, seed: int = 0):
        assert len(base_pool) > 0, "base_pool must be non-empty"
        rng = np.random.default_rng(seed)
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_samples):
            base = base_pool[rng.integers(len(base_pool))]
            solved = apply_random_symmetries(base, rng)
            puzzle, solution = make_puzzle(solved, n_blanks, rng)
            self.samples.append((puzzle, solution))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        puzzle, solution = self.samples[idx]
        return torch.from_numpy(puzzle).long(), torch.from_numpy(solution).long()


# =============================================================================
# Run result
# =============================================================================

@dataclass
class RunResult:
    model_name: str
    train_size: int
    seed: int
    acc_in_dist:  float    # blank-cell acc on in-distribution val
    acc_ood_base: float    # blank-cell acc on held-out base puzzles
    acc_ood_diff: float    # blank-cell acc on harder (more blanks) puzzles
    n_params: int
    wall_time_s: float


# =============================================================================
# Train + multi-eval
# =============================================================================

def evaluate_blank_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Mean blank-cell accuracy on a single eval set."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for puzzle, solution in loader:
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            logits = model(puzzle)
            _, _, acc_blank = loss_and_metrics(logits, puzzle, solution)
            total += acc_blank
            n += 1
    return total / n


def train_and_eval_three_surfaces(model: nn.Module, train_loader: DataLoader,
                                   in_dist_loader: DataLoader,
                                   ood_base_loader: DataLoader,
                                   ood_diff_loader: DataLoader,
                                   epochs: int, lr: float, device: str
                                   ) -> Tuple[float, float, float]:
    """Train, then evaluate on all three eval surfaces."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for puzzle, solution in train_loader:
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            logits = model(puzzle)
            loss, _, _ = loss_and_metrics(logits, puzzle, solution)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return (evaluate_blank_acc(model, in_dist_loader,  device),
            evaluate_blank_acc(model, ood_base_loader, device),
            evaluate_blank_acc(model, ood_diff_loader, device))


# =============================================================================
# Model factory
# =============================================================================

def make_model(name: str, *, embed_dim: int, osc_per_cell: int,
               kuramoto_steps: int) -> nn.Module:
    intermediate_dim = 3 * osc_per_cell
    kwargs = dict(embed_dim=embed_dim, intermediate_dim=intermediate_dim,
                  osc_per_cell=osc_per_cell, kuramoto_steps=kuramoto_steps)
    if name == 'baseline':
        return AKOrNBaseline(**kwargs)
    if name == 'smm':
        return AKOrNWithSMM(**kwargs)
    if name == 'smm_amp':
        return AKOrNWithSMMAmp(**kwargs)
    raise ValueError(f"Unknown model: {name!r}")


# =============================================================================
# Sweep
# =============================================================================

def run_sweep(args, train_bases: List[np.ndarray],
              heldout_bases: List[np.ndarray]) -> List[RunResult]:
    train_sizes = sorted(int(x) for x in args.train_sizes.split(','))
    model_names = tuple(name.strip() for name in args.models.split(','))
    total_runs = len(model_names) * len(train_sizes) * args.seeds

    print(f"=== Path D: OOD sample-efficiency sweep ===")
    print(f"  models       : {list(model_names)}")
    print(f"  train_sizes  : {train_sizes}")
    print(f"  seeds        : {args.seeds}")
    print(f"  epochs       : {args.epochs}")
    print(f"  total runs   : {total_runs}")
    print(f"  bases        : {len(train_bases)} train, {len(heldout_bases)} held out")
    print(f"  blanks       : train={args.n_blanks}, ood_diff={args.ood_n_blanks}")
    print()

    # Eval sets are SHARED across all training runs (fixed seed for reproducibility).
    # This ensures variance reflects only training noise, not eval-set sampling.
    in_dist_set  = SudokuDatasetOOD(args.val_size, train_bases,
                                     n_blanks=args.n_blanks, seed=999)
    ood_base_set = SudokuDatasetOOD(args.val_size, heldout_bases,
                                     n_blanks=args.n_blanks, seed=998)
    ood_diff_set = SudokuDatasetOOD(args.val_size, train_bases,
                                     n_blanks=args.ood_n_blanks, seed=997)
    in_dist_loader  = DataLoader(in_dist_set,  batch_size=args.batch_size, shuffle=False)
    ood_base_loader = DataLoader(ood_base_set, batch_size=args.batch_size, shuffle=False)
    ood_diff_loader = DataLoader(ood_diff_set, batch_size=args.batch_size, shuffle=False)

    results: List[RunResult] = []
    sweep_start = time.time()
    run_idx = 0

    for seed in range(args.seeds):
        # Different training sample per seed (variance source #1)
        for train_size in train_sizes:
            train_set = SudokuDatasetOOD(train_size, train_bases,
                                         n_blanks=args.n_blanks, seed=seed)
            train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                      shuffle=True)

            for model_name in model_names:
                run_idx += 1
                # Different init per (seed, model) (variance source #2)
                torch.manual_seed(
                    seed * 10_000 + train_size + (0 if model_name == 'baseline'
                                                   else 1 if model_name == 'smm' else 2)
                )

                model = make_model(model_name,
                                   embed_dim=args.embed_dim,
                                   osc_per_cell=args.osc_per_cell,
                                   kuramoto_steps=args.kuramoto_steps)
                n_params = count_params(model)

                t0 = time.time()
                a_in, a_ood_b, a_ood_d = train_and_eval_three_surfaces(
                    model, train_loader,
                    in_dist_loader, ood_base_loader, ood_diff_loader,
                    epochs=args.epochs, lr=args.lr, device=args.device,
                )
                wall = time.time() - t0

                results.append(RunResult(
                    model_name=model_name, train_size=train_size, seed=seed,
                    acc_in_dist=a_in, acc_ood_base=a_ood_b, acc_ood_diff=a_ood_d,
                    n_params=n_params, wall_time_s=wall,
                ))

                elapsed = time.time() - sweep_start
                eta_min = (elapsed / run_idx) * (total_runs - run_idx) / 60
                print(f"  [{run_idx:3d}/{total_runs}] seed={seed} N={train_size:4d} "
                      f"{model_name:8s} | in_dist={a_in:.3f}  ood_base={a_ood_b:.3f}  "
                      f"ood_diff={a_ood_d:.3f}  {wall:4.1f}s  eta {eta_min:4.1f}m")

    print(f"\nSweep complete in {(time.time() - sweep_start)/60:.1f} min.")
    return results


# =============================================================================
# Aggregation, N_50, IO, plotting
# =============================================================================

EVAL_SURFACES = ('acc_in_dist', 'acc_ood_base', 'acc_ood_diff')
SURFACE_LABELS = {
    'acc_in_dist':  'in-distribution',
    'acc_ood_base': 'OOD-base',
    'acc_ood_diff': 'OOD-difficulty',
}


def aggregate(results: List[RunResult], surface: str
              ) -> Dict[Tuple[str, int], Dict[str, float]]:
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for r in results:
        grouped.setdefault((r.model_name, r.train_size), []).append(getattr(r, surface))
    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
            for k, v in grouped.items()}


def compute_n50(summary: Dict[Tuple[str, int], Dict[str, float]],
                model_name: str, target_acc: float) -> Optional[int]:
    """Smallest grid-point train_size where mean accuracy reaches target_acc.

    Discrete: returns one of the sizes in the sweep, or None.
    Use compute_n50_interp for sub-grid resolution.
    """
    sizes = sorted({s for n, s in summary if n == model_name})
    for size in sizes:
        if summary[(model_name, size)]['mean'] >= target_acc:
            return size
    return None


def compute_n50_interp(summary: Dict[Tuple[str, int], Dict[str, float]],
                       model_name: str, target_acc: float) -> Optional[float]:
    """Interpolated N_50 — linear interpolation in log-N space.

    Finds the train_size at which the linearly-interpolated mean accuracy
    equals target_acc. More sensitive than the grid-point version when
    crossings happen between adjacent grid points (which is most of the
    time at this grid resolution).

    Returns float (not int) since the result is a continuous estimate.
    None if target was never reached.
    """
    sizes = sorted({s for n, s in summary if n == model_name})
    accs = [summary[(model_name, s)]['mean'] for s in sizes]

    # If first point already at/above target
    if accs[0] >= target_acc:
        return float(sizes[0])

    # Find first crossing between consecutive grid points
    for i in range(len(sizes) - 1):
        if accs[i] < target_acc <= accs[i + 1]:
            log_lo = np.log(sizes[i])
            log_hi = np.log(sizes[i + 1])
            denom = accs[i + 1] - accs[i]
            if denom <= 0:  # non-monotone segment; fall back to upper grid point
                return float(sizes[i + 1])
            frac = (target_acc - accs[i]) / denom
            return float(np.exp(log_lo + frac * (log_hi - log_lo)))

    return None  # never reached


def save_csv(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def plot_three_panels(results: List[RunResult], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[note] matplotlib not installed; skipping plot.")
        return

    style = {
        'baseline': dict(color='#1f77b4', marker='o', label='AKOrN-baseline'),
        'smm':      dict(color='#d62728', marker='s', label='AKOrN+SMM'),
        'smm_amp':  dict(color='#2ca02c', marker='^', label='AKOrN+SMM (+amp)'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, surface in zip(axes, EVAL_SURFACES):
        summary = aggregate(results, surface)
        present = sorted({n for n, _ in summary})
        for name in present:
            st = style.get(name, dict(color='gray', marker='x', label=name))
            sizes = sorted({s for n, s in summary if n == name})
            means = np.array([summary[(name, s)]['mean'] for s in sizes])
            stds  = np.array([summary[(name, s)]['std']  for s in sizes])
            ax.plot(sizes, means, lw=2, **st)
            ax.fill_between(sizes, means - stds, means + stds,
                            color=st['color'], alpha=0.15)
        ax.set_xscale('log')
        ax.set_xlabel('Training set size')
        ax.set_title(SURFACE_LABELS[surface])
        ax.grid(True, which='both', alpha=0.3)

    axes[0].set_ylabel('val_acc on blanks')
    axes[-1].legend(loc='lower right')
    fig.suptitle('Path D: Sample-efficiency across three eval surfaces  (mean ± 1 std)',
                 y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot]   saved to {out_path}")


def print_summary(results: List[RunResult]) -> None:
    sizes = sorted({r.train_size for r in results})
    models = sorted({r.model_name for r in results})

    for surface in EVAL_SURFACES:
        summary = aggregate(results, surface)
        col_w = 18
        print()
        print(f"=== {SURFACE_LABELS[surface]} ===")
        header = f"{'train_size':<12}" + ''.join(f"{m:>{col_w}}" for m in models)
        print(header)
        print('-' * len(header))
        for size in sizes:
            row = f"{size:<12}"
            for m in models:
                cell = summary.get((m, size))
                row += (f"{cell['mean']:.3f} ± {cell['std']:.3f}".rjust(col_w)
                        if cell else 'N/A'.rjust(col_w))
            print(row)


def report_n50_per_surface(results: List[RunResult]) -> None:
    """Per-surface N_50 ratios using both grid-point and interpolated estimates."""
    models = sorted({r.model_name for r in results})
    sizes = sorted({r.train_size for r in results})
    if 'baseline' not in models:
        return
    max_size = max(sizes)

    for surface in EVAL_SURFACES:
        summary = aggregate(results, surface)
        base_max = summary[('baseline', max_size)]['mean']
        print()
        print(f"=== N_50 analysis on {SURFACE_LABELS[surface]} ===")
        print(f"  baseline max accuracy at N={max_size}: {base_max:.3f}")
        print(f"  {'model':<10} {'max':>8} {'N50_grid':>10} "
              f"{'N50_interp':>12} {'ratio_int':>10}  verdict")
        for name in models:
            if name == 'baseline':
                continue
            cand_max = summary[(name, max_size)]['mean']
            target = 0.5 * max(base_max, cand_max)

            n50_b_grid = compute_n50(summary, 'baseline', target)
            n50_c_grid = compute_n50(summary, name, target)
            n50_b_int = compute_n50_interp(summary, 'baseline', target)
            n50_c_int = compute_n50_interp(summary, name, target)

            if n50_b_int is None or n50_c_int is None:
                ratio_str = "  N/A"
                verdict = "target not reached"
            else:
                ratio_int = n50_b_int / n50_c_int
                if ratio_int >= 2.0:
                    verdict = "*** SUPPORTED ***"
                elif ratio_int >= 1.10:
                    verdict = "WEAK SIGNAL (smm ahead)"
                elif ratio_int >= 0.90:
                    verdict = "TIED (within 10%)"
                else:
                    verdict = "NOT SUPPORTED (baseline ahead)"
                ratio_str = f"{ratio_int:8.2f}"

            n50_b_str = str(n50_b_grid) if n50_b_grid else "—"
            n50_c_str = str(n50_c_grid) if n50_c_grid else "—"
            n50_b_int_str = f"{n50_b_int:.0f}" if n50_b_int else "—"
            n50_c_int_str = f"{n50_c_int:.0f}" if n50_c_int else "—"
            print(f"  {name:<10} {cand_max:>8.3f} "
                  f"{n50_b_str:>4}/{n50_c_str:<5} "
                  f"{n50_b_int_str:>5}/{n50_c_int_str:<6} "
                  f"{ratio_str}  {verdict}")


def report_generalization_gap(results: List[RunResult]) -> None:
    """Generalization gap (in_dist − ood_diff) per (model, train_size).

    Smaller gap = the model degrades less when puzzles get harder.
    A model with structural priors should have smaller gap if the prior
    captures task structure rather than just fitting easy examples.
    """
    in_dist = aggregate(results, 'acc_in_dist')
    ood_diff = aggregate(results, 'acc_ood_diff')
    sizes = sorted({r.train_size for r in results})
    models = sorted({r.model_name for r in results})

    print()
    print(f"=== Generalization gap (in_dist − ood_diff)  — smaller is better ===")
    print(f"   A smaller gap means the model degrades LESS on harder puzzles.")
    print(f"   This is the signature of an inductive bias that captures task structure.")
    print()
    col_w = 14
    header = f"{'train_size':<12}" + ''.join(f"{m:>{col_w}}" for m in models)
    print(header)
    print('-' * len(header))
    for size in sizes:
        row = f"{size:<12}"
        for m in models:
            id_ = in_dist.get((m, size))
            od = ood_diff.get((m, size))
            if id_ and od:
                gap = id_['mean'] - od['mean']
                row += f"{gap:>{col_w}.3f}"
            else:
                row += 'N/A'.rjust(col_w)
        print(row)

    # Compare each candidate's gap vs baseline's gap
    if 'baseline' in models:
        print()
        print(f"   Gap-ratio (candidate / baseline) — < 1.0 means SMM transfers BETTER:")
        for m in models:
            if m == 'baseline':
                continue
            ratios = []
            for size in sizes:
                bgap = (in_dist[('baseline', size)]['mean']
                        - ood_diff[('baseline', size)]['mean'])
                cgap = (in_dist[(m, size)]['mean']
                        - ood_diff[(m, size)]['mean'])
                if bgap > 0:
                    ratios.append((size, cgap / bgap))
            ratio_strs = ', '.join(f"N={s}: {r:.2f}" for s, r in ratios)
            print(f"     {m:<10}  {ratio_strs}")


def overall_verdict(results: List[RunResult]) -> None:
    """Summary across all three eval surfaces — the headline interpretation."""
    print()
    print(f"=== How to read this ===")
    print(f"Three signals worth checking, in order of how much they support SMM:")
    print()
    print(f"  1. INTERPOLATED N_50 ratios — does SMM reach a target with fewer")
    print(f"     samples than baseline? Look at ratio_int per surface:")
    print(f"        > 2.0   ->  SUPPORTED ")
    print(f"        > 1.10  ->  WEAK SIGNAL")
    print(f"        ≈ 1.0   ->  TIED (within 10%)")
    print(f"        < 0.90  ->  NOT SUPPORTED")
    print()
    print(f"  2. GENERALIZATION GAP (in_dist − ood_diff) — does SMM degrade LESS")
    print(f"     than baseline when puzzles get harder?")
    print(f"     Gap-ratio < 1.0 means SMM transfers better, even if its")
    print(f"     absolute accuracy is lower.")
    print()
    print(f"  3. CROSS-SURFACE PATTERN — does SMM's gap to baseline shrink")
    print(f"     as we move from in_dist → ood_base → ood_diff?")
    print(f"     If yes, the structural prior is doing real work even if it's")
    print(f"     not winning in absolute terms yet.")
    print()
    print(f"At fixed (small) capacity, signals 2 and 3 are what tell you")
    print(f"the inductive bias matters. Signal 1 typically only flips at scale.")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--train-sizes', type=str, default='32,64,128,256,512')
    p.add_argument('--models', type=str, default='baseline,smm,smm_amp')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--val-size', type=int, default=64)
    p.add_argument('--n-blanks', type=int, default=20,
                   help='Blanks per puzzle in train + in_dist + ood_base sets')
    p.add_argument('--ood-n-blanks', type=int, default=35,
                   help='Blanks per puzzle in ood_diff set (harder)')
    p.add_argument('--n-train-bases', type=int, default=40,
                   help='Number of distinct base solutions used for training')
    p.add_argument('--n-heldout-bases', type=int, default=10,
                   help='Number of base solutions held out for ood_base')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--embed-dim', type=int, default=24)
    p.add_argument('--osc-per-cell', type=int, default=4)
    p.add_argument('--kuramoto-steps', type=int, default=8)
    p.add_argument('--device', type=str, default='cpu',
                   choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', type=str, default='experiments/results_ood')
    p.add_argument('--base-pool-seed', type=int, default=42,
                   help='Seed for generating the base solution pool '
                        '(fixed across runs for reproducibility)')
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    # ---- Generate base pool ----
    print(f"Generating {args.n_train_bases + args.n_heldout_bases} distinct "
          f"random Sudoku solutions for the base pool...")
    t0 = time.time()
    base_rng = np.random.default_rng(args.base_pool_seed)
    all_bases = [random_solved_sudoku(base_rng)
                 for _ in range(args.n_train_bases + args.n_heldout_bases)]
    train_bases = all_bases[:args.n_train_bases]
    heldout_bases = all_bases[args.n_train_bases:]
    print(f"  base pool generated in {time.time() - t0:.1f}s")
    print()

    # ---- Run sweep ----
    results = run_sweep(args, train_bases, heldout_bases)

    # ---- Save and report ----
    save_csv(results, out_dir / 'results.csv')
    print(f"[csv]    {len(results)} rows saved to {out_dir / 'results.csv'}")

    print_summary(results)
    report_n50_per_surface(results)
    report_generalization_gap(results)
    overall_verdict(results)
    plot_three_panels(results, out_dir / 'sample_efficiency_ood.png')

    print(f"\nDone. Inspect {out_dir}/ for results.csv and the 3-panel plot.")


if __name__ == '__main__':
    main()
