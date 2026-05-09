"""
experiments/sample_efficiency.py
=================================
Sample-efficiency sweep — the scientific instrument behind the falsifiable claim.

For each (model_variant, train_size, seed) cell, train and record final
validation accuracy. Aggregate across seeds to get mean ± std per cell.
Compute the N_50 ratio between AKOrN-baseline and AKOrN+SMM:

    N_50(model) = smallest train_size at which mean validation accuracy
                  reaches 50% of the maximum accuracy achieved at the
                  largest training-set size in the sweep

The falsifiable claim is operationalized as:
    N_50(baseline) / N_50(smm) >= 2  (SMM is 2x more sample-efficient)

Outputs:
    <out_dir>/results.csv             -- one row per (model, size, seed) run
    <out_dir>/sample_efficiency.png   -- comparison plot, log-x, mean ± std bands

Usage:
    python experiments/sample_efficiency.py
    python experiments/sample_efficiency.py --seeds 5 --epochs 30
    python experiments/sample_efficiency.py --train-sizes 32,128,512,1024

Default config runs ~30 cells in 2-3 minutes on an M-series Mac CPU.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Reuse model definitions from the integration experiment.
# Both files live in experiments/, so the directory is on sys.path automatically
# when this script is run as `python experiments/sample_efficiency.py`.
from akorn_smm_sudoku import (
    AKOrNBaseline,
    AKOrNWithSMM,
    AKOrNWithSMMAmp,
    SudokuDataset,
    loss_and_metrics,
    count_params,
)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RunResult:
    model_name: str
    train_size: int
    seed: int
    final_val_loss: float
    final_val_acc_blank: float
    final_val_acc_full: float
    n_params: int
    wall_time_s: float


# =============================================================================
# Quiet training loop (no per-epoch logging — the sweep does its own progress)
# =============================================================================

def train_and_eval(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int, lr: float, device: str
                   ) -> Tuple[float, float, float]:
    """Train for `epochs` epochs, return (final_val_loss, val_acc_full, val_acc_blank)."""
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

    # Final evaluation
    model.eval()
    sums = {'loss': 0.0, 'acc_full': 0.0, 'acc_blank': 0.0, 'n': 0}
    with torch.no_grad():
        for puzzle, solution in val_loader:
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            logits = model(puzzle)
            loss, acc_full, acc_blank = loss_and_metrics(logits, puzzle, solution)
            sums['loss'] += loss.item()
            sums['acc_full'] += acc_full
            sums['acc_blank'] += acc_blank
            sums['n'] += 1

    return (sums['loss'] / sums['n'],
            sums['acc_full'] / sums['n'],
            sums['acc_blank'] / sums['n'])


# =============================================================================
# Model factory
# =============================================================================

def make_model(name: str, *, embed_dim: int, osc_per_cell: int,
               kuramoto_steps: int) -> nn.Module:
    intermediate_dim = 3 * osc_per_cell  # required by HelicalEmbedding
    kwargs = dict(embed_dim=embed_dim, intermediate_dim=intermediate_dim,
                  osc_per_cell=osc_per_cell, kuramoto_steps=kuramoto_steps)
    if name == 'baseline':
        return AKOrNBaseline(**kwargs)
    if name == 'smm':
        return AKOrNWithSMM(**kwargs)
    if name == 'smm_amp':
        return AKOrNWithSMMAmp(**kwargs)
    raise ValueError(f"Unknown model name: {name!r}")


# =============================================================================
# Sweep
# =============================================================================

def run_sweep(args) -> List[RunResult]:
    train_sizes = sorted(int(x) for x in args.train_sizes.split(','))
    model_names = tuple(name.strip() for name in args.models.split(','))
    largest_train_size = max(train_sizes)
    total_runs = len(model_names) * len(train_sizes) * args.seeds

    print(f"=== Sample-efficiency sweep ===")
    print(f"  models      : {list(model_names)}")
    print(f"  train_sizes : {train_sizes}")
    print(f"  seeds       : {args.seeds}")
    print(f"  epochs      : {args.epochs}")
    print(f"  total runs  : {total_runs}")
    print(f"  device      : {args.device}")
    print()

    results: List[RunResult] = []
    sweep_start = time.time()
    run_idx = 0

    for seed in range(args.seeds):
        # Different DATA per seed for proper variance estimation.
        # Generate one big dataset, subset it for each train_size.
        full_dataset = SudokuDataset(
            n_samples=largest_train_size + args.val_size,
            n_blanks=args.n_blanks,
            seed=seed,
        )
        val_set = Subset(full_dataset,
                         range(largest_train_size,
                               largest_train_size + args.val_size))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        for train_size in train_sizes:
            train_set = Subset(full_dataset, range(train_size))
            train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                      shuffle=True)

            for model_name in model_names:
                run_idx += 1
                # Different INIT per (seed, model) — independent variance
                torch.manual_seed(
                    seed * 10_000 + train_size + (0 if model_name == 'baseline' else 1)
                )

                model = make_model(
                    model_name,
                    embed_dim=args.embed_dim,
                    osc_per_cell=args.osc_per_cell,
                    kuramoto_steps=args.kuramoto_steps,
                )
                n_params = count_params(model)

                t0 = time.time()
                val_loss, val_acc_full, val_acc_blank = train_and_eval(
                    model, train_loader, val_loader,
                    epochs=args.epochs, lr=args.lr, device=args.device,
                )
                wall = time.time() - t0

                results.append(RunResult(
                    model_name=model_name,
                    train_size=train_size,
                    seed=seed,
                    final_val_loss=val_loss,
                    final_val_acc_blank=val_acc_blank,
                    final_val_acc_full=val_acc_full,
                    n_params=n_params,
                    wall_time_s=wall,
                ))

                elapsed = time.time() - sweep_start
                eta_min = (elapsed / run_idx) * (total_runs - run_idx) / 60
                print(f"  [{run_idx:3d}/{total_runs}] "
                      f"seed={seed} N={train_size:4d} {model_name:8s} | "
                      f"acc(blank)={val_acc_blank:.3f}  loss={val_loss:.3f}  "
                      f"{wall:4.1f}s  eta {eta_min:4.1f}m")

    print(f"\nSweep complete in {(time.time() - sweep_start)/60:.1f} min.")
    return results


# =============================================================================
# Aggregation, N_50, IO
# =============================================================================

def aggregate(results: List[RunResult],
              metric: str = 'final_val_acc_blank'
              ) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Mean and std of `metric` per (model_name, train_size) cell."""
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for r in results:
        key = (r.model_name, r.train_size)
        grouped.setdefault(key, []).append(getattr(r, metric))

    return {key: {'mean': float(np.mean(v)),
                  'std':  float(np.std(v)),
                  'n':    len(v)}
            for key, v in grouped.items()}


def compute_n50(summary: Dict[Tuple[str, int], Dict[str, float]],
                model_name: str, target_acc: float) -> Optional[int]:
    """Smallest train_size at which model's mean accuracy reaches target_acc."""
    sizes = sorted({s for n, s in summary if n == model_name})
    for size in sizes:
        if summary[(model_name, size)]['mean'] >= target_acc:
            return size
    return None


def save_csv(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def plot_curves(summary: Dict[Tuple[str, int], Dict[str, float]],
                out_path: Path,
                metric_label: str = 'val_acc on blanks') -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')   # headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[note] matplotlib not installed; skipping plot.")
        print("       install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    style = {
        'baseline': dict(color='#1f77b4', marker='o', label='AKOrN-baseline'),
        'smm':      dict(color='#d62728', marker='s', label='AKOrN+SMM (phase only)'),
        'smm_amp':  dict(color='#2ca02c', marker='^', label='AKOrN+SMM (phase + amplitude)'),
    }

    present_models = sorted({n for n, _ in summary})
    for name in present_models:
        st = style.get(name, dict(color='gray', marker='x', label=name))
        sizes = sorted({s for n, s in summary if n == name})
        means = np.array([summary[(name, s)]['mean'] for s in sizes])
        stds  = np.array([summary[(name, s)]['std']  for s in sizes])
        ax.plot(sizes, means, lw=2, **st)
        ax.fill_between(sizes, means - stds, means + stds,
                        color=st['color'], alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel('Training set size (puzzles, log scale)')
    ax.set_ylabel(metric_label)
    ax.set_title('Sample-efficiency comparison\n(mean ± 1 std across seeds)')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot]   saved to {out_path}")


def print_summary_table(summary: Dict[Tuple[str, int], Dict[str, float]]) -> None:
    sizes = sorted({s for _, s in summary})
    models = sorted({n for n, _ in summary})
    col_w = 18
    print()
    print(f"=== Sample-efficiency summary (val_acc on blanks) ===")
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
    print()


def report_n50(summary: Dict[Tuple[str, int], Dict[str, float]]) -> None:
    sizes = sorted({s for _, s in summary})
    models = sorted({n for n, _ in summary})
    max_size = max(sizes)

    if 'baseline' not in models:
        print("[note] baseline not in sweep; skipping N_50 ratio analysis")
        return

    base_max = summary[('baseline', max_size)]['mean']
    print(f"=== N_50 analysis ===")
    print(f"  baseline max accuracy at N={max_size}: {base_max:.3f}")
    print()

    for name in models:
        if name == 'baseline':
            continue
        cand_max = summary[(name, max_size)]['mean']
        target = 0.5 * max(base_max, cand_max)

        n50_b = compute_n50(summary, 'baseline', target)
        n50_c = compute_n50(summary, name, target)

        print(f"  --- {name} vs baseline ---")
        print(f"    {name} max accuracy at N={max_size}: {cand_max:.3f}")
        print(f"    target (50% of max)               : {target:.3f}")
        print(f"    N_50(baseline)                    : {n50_b}")
        print(f"    N_50({name}){' '*(max(0, 23-len(name)))}: {n50_c}")

        if n50_b is None or n50_c is None:
            print(f"    [N_50 not computable — target unreached at any size]")
            print(f"    Try increasing --epochs or extending --train-sizes upward.")
        else:
            ratio = n50_b / n50_c
            verdict = ("more sample-efficient" if ratio > 1
                       else "tied" if ratio == 1
                       else "less sample-efficient")
            print(f"    ratio (baseline / {name}): {ratio:.2f}    [{name} {verdict}]")
            if ratio >= 2.0:
                tag = "SUPPORTED at this scale"
                detail = f"({name} reaches target with at least 2x fewer puzzles)"
            elif ratio > 1.0:
                tag = "WEAK SIGNAL"
                detail = f"({name} ahead but below 2x threshold)"
            elif ratio == 1.0:
                tag = "TIED at this scale"
                detail = "(both reach target at the same train size)"
            else:
                tag = "NOT SUPPORTED at this scale"
                detail = f"(baseline more sample-efficient than {name})"
            print(f"    Falsifiable claim status: {tag}")
            print(f"      {detail}")
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--train-sizes', type=str, default='32,64,128,256,512',
                   help='Comma-separated training set sizes (log-spaced recommended)')
    p.add_argument('--models', type=str, default='baseline,smm,smm_amp',
                   help='Comma-separated model names to compare')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--val-size', type=int, default=64)
    p.add_argument('--n-blanks', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--embed-dim', type=int, default=24)
    p.add_argument('--osc-per-cell', type=int, default=4)
    p.add_argument('--kuramoto-steps', type=int, default=8)
    p.add_argument('--device', type=str, default='cpu',
                   choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', type=str, default='experiments/results')
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    results = run_sweep(args)

    save_csv(results, out_dir / 'results.csv')
    print(f"[csv]    {len(results)} rows saved to {out_dir / 'results.csv'}")

    summary = aggregate(results, metric='final_val_acc_blank')
    print_summary_table(summary)
    report_n50(summary)
    plot_curves(summary, out_dir / 'sample_efficiency.png')

    print(f"\nDone. Inspect {out_dir}/ for results.csv and sample_efficiency.png.")


if __name__ == '__main__':
    main()
