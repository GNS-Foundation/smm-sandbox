"""
experiments/lr_finder.py
=========================
Path F — LR sensitivity scout.

Purpose: determine whether smm_full's underperformance vs baseline at the
scale-up config is a fundamental architecture issue or a hyperparameter
artifact. Complex-valued networks are notoriously LR-sensitive; the default
lr=3e-3 was chosen for baseline and may be wrong for the deeper helical body.

Method: train each model variant across a range of learning rates at fixed
N=256 with a single seed (this is a scout, not a confirmation run).
Report final in-distribution accuracy. The model whose best LR differs from
the previous sweep's default is the one whose verdict needs revision.

Output:
  - <out_dir>/lr_finder.csv       table of (model, lr, final_acc, final_loss)
  - terminal: per-model best-LR recommendation

Usage:
    python experiments/lr_finder.py
    python experiments/lr_finder.py --lrs 1e-4,3e-4,1e-3,3e-3,1e-2
    python experiments/lr_finder.py --models baseline,smm_full

Wall time: ~15 min on M-series CPU at default config (3 models x 5 LRs x N=256
x 30 epochs). Single seed, so noisy — the goal is finding the *direction*
each model wants its LR to move, not the precise optimum.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from akorn_smm_sudoku import (
    AKOrNBaseline,
    AKOrNWithSMM,
    AKOrNWithSMMAmp,
    AKOrNWithSMMFull,
    loss_and_metrics,
    count_params,
)
from sample_efficiency_ood import (
    SudokuDatasetOOD,
    random_solved_sudoku,
    make_model,
)


@dataclass
class LRRunResult:
    model_name: str
    lr: float
    final_train_loss: float
    final_val_acc: float
    final_val_loss: float
    converged: bool        # False if NaN/Inf encountered
    n_params: int
    wall_time_s: float


def train_with_curve(model: nn.Module, train_loader: DataLoader,
                     val_loader: DataLoader, epochs: int, lr: float,
                     device: str) -> Tuple[float, float, float, bool]:
    """Train and return (final_train_loss, final_val_loss, final_val_acc, converged).

    `converged` is False if any NaN/Inf appeared during training — that's a
    real signal: too-high LR for that model.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    converged = True
    final_train_loss = float('nan')

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for puzzle, solution in train_loader:
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            logits = model(puzzle)
            loss, _, _ = loss_and_metrics(logits, puzzle, solution)
            if torch.isnan(loss) or torch.isinf(loss):
                converged = False
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        if not converged:
            break
        final_train_loss = float(np.mean(epoch_losses))

    # Final eval
    model.eval()
    val_loss_sum = 0.0
    val_acc_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for puzzle, solution in val_loader:
            puzzle = puzzle.to(device)
            solution = solution.to(device)
            logits = model(puzzle)
            loss, _, acc_blank = loss_and_metrics(logits, puzzle, solution)
            if torch.isnan(loss) or torch.isinf(loss):
                converged = False
                break
            val_loss_sum += loss.item()
            val_acc_sum += acc_blank
            n_batches += 1

    val_loss = val_loss_sum / max(n_batches, 1) if converged else float('nan')
    val_acc = val_acc_sum / max(n_batches, 1) if converged else 0.0
    return final_train_loss, val_loss, val_acc, converged


def run_lr_sweep(args) -> List[LRRunResult]:
    lrs = sorted(float(x) for x in args.lrs.split(','))
    model_names = [m.strip() for m in args.models.split(',')]
    total = len(lrs) * len(model_names)

    print(f"=== Path F: LR sensitivity scout ===")
    print(f"  models      : {model_names}")
    print(f"  learning_rates: {lrs}")
    print(f"  fixed N     : {args.train_size}")
    print(f"  epochs      : {args.epochs}")
    print(f"  total runs  : {total}")
    print(f"  seed        : {args.seed} (single seed — scout, not confirmation)")
    print()

    # Generate dataset (shared across all runs for apples-to-apples comparison)
    base_rng = np.random.default_rng(42)
    n_bases = max(20, args.train_size // 10)
    bases = [random_solved_sudoku(base_rng) for _ in range(n_bases)]
    train_set = SudokuDatasetOOD(args.train_size, bases,
                                  n_blanks=args.n_blanks, seed=args.seed)
    val_set = SudokuDatasetOOD(args.val_size, bases,
                                n_blanks=args.n_blanks, seed=args.seed + 999)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    results: List[LRRunResult] = []
    sweep_start = time.time()
    run_idx = 0

    for model_name in model_names:
        for lr in lrs:
            run_idx += 1
            torch.manual_seed(args.seed)
            model = make_model(model_name,
                               embed_dim=args.embed_dim,
                               osc_per_cell=args.osc_per_cell,
                               kuramoto_steps=args.kuramoto_steps)
            n_params = count_params(model)

            t0 = time.time()
            train_loss, val_loss, val_acc, conv = train_with_curve(
                model, train_loader, val_loader,
                epochs=args.epochs, lr=lr, device=args.device,
            )
            wall = time.time() - t0

            results.append(LRRunResult(
                model_name=model_name, lr=lr,
                final_train_loss=train_loss,
                final_val_loss=val_loss, final_val_acc=val_acc,
                converged=conv, n_params=n_params, wall_time_s=wall,
            ))

            elapsed = time.time() - sweep_start
            eta_min = (elapsed / run_idx) * (total - run_idx) / 60
            status = "OK" if conv else "DIVERGED"
            print(f"  [{run_idx:2d}/{total}] {model_name:9s} lr={lr:>7.0e} "
                  f"| acc={val_acc:.3f} loss={val_loss:.3f}  {status:>10}  "
                  f"{wall:5.1f}s  eta {eta_min:4.1f}m")

    print(f"\nSweep complete in {(time.time() - sweep_start)/60:.1f} min.")
    return results


def report(results: List[LRRunResult]) -> None:
    """Print per-model best LR and the full table."""
    models = sorted({r.model_name for r in results})
    lrs = sorted({r.lr for r in results})

    # Table
    print()
    print(f"=== LR sweep results (final val_acc on blanks) ===")
    col_w = 14
    header = f"{'lr':<10}" + ''.join(f"{m:>{col_w}}" for m in models)
    print(header)
    print('-' * len(header))
    by_key = {(r.model_name, r.lr): r for r in results}
    for lr in lrs:
        row = f"{lr:<10.0e}"
        for m in models:
            r = by_key.get((m, lr))
            if r is None or not r.converged:
                row += "DIVERGED".rjust(col_w)
            else:
                row += f"{r.final_val_acc:.3f}".rjust(col_w)
        print(row)

    # Best per model
    print()
    print(f"=== Best LR per model ===")
    best: Dict[str, LRRunResult] = {}
    for r in results:
        if not r.converged:
            continue
        if r.model_name not in best or r.final_val_acc > best[r.model_name].final_val_acc:
            best[r.model_name] = r
    for m in models:
        if m in best:
            r = best[m]
            print(f"  {m:9s} : lr={r.lr:.0e}  -> val_acc={r.final_val_acc:.3f}")
        else:
            print(f"  {m:9s} : ALL LRs DIVERGED")

    # Verdict
    print()
    print(f"=== Verdict ===")
    if 'smm_full' in best and 'baseline' in best:
        smm_full_best_lr = best['smm_full'].lr
        baseline_best_lr = best['baseline'].lr
        smm_full_at_baseline_lr = next(
            (r for r in results if r.model_name == 'smm_full'
             and abs(r.lr - baseline_best_lr) < 1e-9),
            None,
        )
        if smm_full_at_baseline_lr and smm_full_at_baseline_lr.converged:
            delta = (best['smm_full'].final_val_acc
                     - smm_full_at_baseline_lr.final_val_acc)
            if smm_full_best_lr == baseline_best_lr:
                print(f"  smm_full's optimal LR matches baseline ({baseline_best_lr:.0e}).")
                print(f"  -> Path B failure is NOT hyperparameter-driven.")
                print(f"     The architecture itself is the issue.")
            elif delta < 0.02:
                print(f"  smm_full prefers lr={smm_full_best_lr:.0e} (vs baseline's {baseline_best_lr:.0e})")
                print(f"  but the improvement is small (Δ={delta:+.3f} acc).")
                print(f"  -> Marginal LR effect. Path B verdict largely unchanged.")
            else:
                print(f"  smm_full prefers lr={smm_full_best_lr:.0e} (vs baseline's {baseline_best_lr:.0e})")
                print(f"  AND the improvement is meaningful (Δ={delta:+.3f} acc).")
                print(f"  -> Path B verdict needs revision. Re-run the full sweep with")
                print(f"     lr={smm_full_best_lr:.0e} for smm_full and see if it now beats baseline.")

    # Diverged runs (worth flagging — too-high LR signals)
    diverged = [r for r in results if not r.converged]
    if diverged:
        print()
        print(f"=== Diverged runs (NaN/Inf during training) ===")
        for r in diverged:
            print(f"  {r.model_name:9s} at lr={r.lr:.0e}")


def save_csv(results: List[LRRunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--models', type=str, default='baseline,smm,smm_full',
                   help='Comma-separated model names')
    p.add_argument('--lrs', type=str, default='1e-4,3e-4,1e-3,3e-3,1e-2',
                   help='Comma-separated learning rates to try')
    p.add_argument('--train-size', type=int, default=256)
    p.add_argument('--val-size', type=int, default=64)
    p.add_argument('--n-blanks', type=int, default=20)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--embed-dim', type=int, default=64)
    p.add_argument('--osc-per-cell', type=int, default=16)
    p.add_argument('--kuramoto-steps', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu',
                   choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', type=str, default='experiments/results_lr_finder')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    results = run_lr_sweep(args)

    save_csv(results, out_dir / 'lr_finder.csv')
    print(f"\n[csv]    {len(results)} rows saved to {out_dir / 'lr_finder.csv'}")

    report(results)

    print(f"\nDone. If smm_full has a clearly different optimal LR, re-run")
    print(f"  python experiments/sample_efficiency_ood.py --lr <new_lr> ...")
    print(f"with the recommended LR to confirm at full multi-seed scale.")


if __name__ == '__main__':
    main()
