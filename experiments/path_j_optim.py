"""
experiments/path_j_optim.py
============================
Investigate the optimization plateau in AKOrNToroidalCoupled at N=64.

Background
----------
Path J showed toroidal_coupled at N=64 in_dist = 0.263 +/- 0.067 across 3
seeds. Per-seed values: 0.188 / 0.350 / 0.252. Two of three seeds didn't
escape the high-loss plateau in 30 epochs. The diagnostic loss curve at
N=256 showed the same pattern: epochs 5-15 stuck at ~0.54, then sudden
drop to 0.16 at epoch 20, fully converged by 30. With only 4 batches per
epoch at N=64 (vs 16 at N=256), the optimizer gets ~75% fewer updates,
making the plateau harder to cross.

Three hypotheses about what's happening:

  A. BUDGET: the plateau breaks reliably with enough updates. The model
     just needs more iterations to escape. Fix: longer training.

  B. OPTIMIZATION: the plateau is a steep boundary that AdamW from a
     cold start can't cross. The high LR (1e-2) overshoots in the
     plateau region. Fix: linear LR warmup over first few epochs.

  C. COMBINED: both contribute; we need budget + warmup.

  D. INTRINSIC: the loss landscape has multiple basins separated by
     barriers that can't be reliably crossed. Fix: different init or
     architectural change.

This script tests A, B, and C. If A or C work (toroidal_coupled at N=64
becomes reliable), Path J + toroidal moves from "promising but fragile"
to "robust positive result." If only D is left, we know the next move
is curriculum init or a different optimizer.

Method
------
For each model in {baseline, helical_coupled, toroidal_coupled,
toroidal_coupled_wide} x each config in
{default(30ep), long(60ep), warmup(30ep+5), long_warmup(60ep+5)}
x 3 seeds:

  - Train at N=64
  - Record per-epoch loss
  - Find phase-transition epoch (first epoch with loss < 0.3)
  - Eval final accuracy on in_dist + ood_diff held-out sets

Output
------
experiments/results_path_j_optim/
    losses.csv     -- (model, seed, config, epoch, loss)
    summary.csv    -- (model, seed, config, transition_epoch, final_in_dist, final_ood_diff)
    loss_curves.png

Wall time on M-series CPU: ~15-20 min for the full 4 x 4 x 3 = 48 runs.

Usage:
    python experiments/path_j_optim.py
    python experiments/path_j_optim.py --models baseline,toroidal_coupled
    python experiments/path_j_optim.py --configs default,long
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from akorn_smm_sudoku import loss_and_metrics, count_params
from sample_efficiency_ood import (
    SudokuDatasetOOD, random_solved_sudoku, make_model,
)


CONFIGS: Dict[str, dict] = {
    'default':     {'epochs': 30, 'warmup_epochs': 0},
    'long':        {'epochs': 60, 'warmup_epochs': 0},
    'warmup':      {'epochs': 30, 'warmup_epochs': 5},
    'long_warmup': {'epochs': 60, 'warmup_epochs': 5},
}

PHASE_TRANSITION_THRESHOLD = 0.3   # loss below this = "escaped the plateau"


@dataclass
class RunResult:
    model_name: str
    seed: int
    config: str
    epochs: int
    warmup_epochs: int
    losses: List[float]                  # per-epoch
    transition_epoch: int                # 1-indexed; -1 if never crossed
    final_loss: float
    final_in_dist_acc: float
    final_ood_diff_acc: float
    n_params: int
    wall_time_s: float


def train_with_warmup_and_history(model: nn.Module, train_loader: DataLoader,
                                    epochs: int, peak_lr: float,
                                    warmup_epochs: int,
                                    device: str) -> List[float]:
    """Train and return per-epoch mean training loss.

    Linear LR warmup from 0 to `peak_lr` over `warmup_epochs`, then
    constant at `peak_lr`. Returns NaN-filled if training diverges.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    losses = []

    for epoch in range(epochs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            current_lr = peak_lr * (epoch + 1) / warmup_epochs
        else:
            current_lr = peak_lr
        for g in optimizer.param_groups:
            g['lr'] = current_lr

        model.train()
        epoch_losses = []
        for puzzle, solution in train_loader:
            puzzle = puzzle.to(device); solution = solution.to(device)
            logits = model(puzzle)
            loss, _, _ = loss_and_metrics(logits, puzzle, solution)
            if torch.isnan(loss) or torch.isinf(loss):
                # Pad remaining epochs with NaN to keep array shapes stable
                losses.extend([float('nan')] * (epochs - epoch))
                return losses
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
    return losses


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """Return (mean loss, mean blank-cell accuracy) on the loader."""
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for puzzle, solution in loader:
            puzzle = puzzle.to(device); solution = solution.to(device)
            logits = model(puzzle)
            loss, _, acc = loss_and_metrics(logits, puzzle, solution)
            losses.append(loss.item())
            accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


def find_phase_transition(losses: List[float],
                            threshold: float = PHASE_TRANSITION_THRESHOLD) -> int:
    """First epoch (1-indexed) where loss drops below threshold; -1 if never."""
    for i, l in enumerate(losses):
        if not np.isnan(l) and l < threshold:
            return i + 1
    return -1


def run_single(model_name: str, seed: int, config_name: str,
                 train_loader: DataLoader, eval_loaders: dict,
                 args) -> RunResult:
    cfg = CONFIGS[config_name]

    torch.manual_seed(seed)
    model = make_model(model_name,
                        embed_dim=args.embed_dim,
                        osc_per_cell=args.osc_per_cell,
                        kuramoto_steps=args.kuramoto_steps)
    n_params = count_params(model)

    t0 = time.time()
    losses = train_with_warmup_and_history(
        model, train_loader,
        epochs=cfg['epochs'], peak_lr=args.lr,
        warmup_epochs=cfg['warmup_epochs'], device=args.device,
    )
    transition = find_phase_transition(losses)
    _, in_dist_acc = evaluate_model(model, eval_loaders['in_dist'], args.device)
    _, ood_diff_acc = evaluate_model(model, eval_loaders['ood_diff'], args.device)
    wall_time = time.time() - t0

    return RunResult(
        model_name=model_name, seed=seed, config=config_name,
        epochs=cfg['epochs'], warmup_epochs=cfg['warmup_epochs'],
        losses=losses, transition_epoch=transition,
        final_loss=losses[-1] if losses else float('nan'),
        final_in_dist_acc=in_dist_acc,
        final_ood_diff_acc=ood_diff_acc,
        n_params=n_params, wall_time_s=wall_time,
    )


def run_sweep(args) -> List[RunResult]:
    model_names = [m.strip() for m in args.models.split(',')]
    config_names = [c.strip() for c in args.configs.split(',')]
    seeds = list(range(args.seeds))
    total = len(model_names) * len(config_names) * len(seeds)

    print(f"=== Path J optimization investigation ===")
    print(f"  models       : {model_names}")
    print(f"  configs      : {config_names}")
    print(f"  seeds        : {seeds}")
    print(f"  N (fixed)    : {args.train_size}")
    print(f"  peak lr      : {args.lr}")
    print(f"  total runs   : {total}")
    print(f"  phase-transition threshold (loss<X): {PHASE_TRANSITION_THRESHOLD}")
    print()

    # Build dataset (shared across all runs at this N)
    base_rng = np.random.default_rng(args.base_pool_seed)
    bases = [random_solved_sudoku(base_rng) for _ in range(40)]
    in_dist_eval = SudokuDatasetOOD(args.eval_size, bases,
                                      n_blanks=args.n_blanks_train,
                                      seed=args.eval_seed)
    ood_diff_eval = SudokuDatasetOOD(args.eval_size, bases,
                                       n_blanks=args.n_blanks_ood,
                                       seed=args.eval_seed + 1000)
    eval_loaders = {
        'in_dist':  DataLoader(in_dist_eval, batch_size=args.batch_size),
        'ood_diff': DataLoader(ood_diff_eval, batch_size=args.batch_size),
    }

    results: List[RunResult] = []
    sweep_start = time.time()
    run_idx = 0

    for seed in seeds:
        # Train set is per-seed, same across configs and models for fair comparison
        train_set = SudokuDatasetOOD(args.train_size, bases,
                                       n_blanks=args.n_blanks_train, seed=seed)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                    shuffle=True)
        for model_name in model_names:
            for config_name in config_names:
                run_idx += 1
                r = run_single(model_name, seed, config_name,
                                  train_loader, eval_loaders, args)
                results.append(r)

                elapsed = time.time() - sweep_start
                eta_min = (elapsed / run_idx) * (total - run_idx) / 60
                tag = (f"transition@{r.transition_epoch}"
                       if r.transition_epoch > 0 else "STUCK")
                print(f"  [{run_idx:2d}/{total}] seed={seed} {model_name:22s} "
                      f"{config_name:12s} | "
                      f"in_dist={r.final_in_dist_acc:.3f} "
                      f"final_loss={r.final_loss:.3f} "
                      f"{tag:>17}  "
                      f"{r.wall_time_s:5.1f}s  eta {eta_min:4.1f}m")

    print(f"\nSweep complete in {(time.time() - sweep_start)/60:.1f} min.")
    return results


def report(results: List[RunResult], args, out_dir: Path) -> None:
    print()
    print(f"=== Phase-transition reliability (transition_epoch < epochs) ===")
    print(f"   Reports the fraction of seeds that escaped the plateau "
          f"(loss < {PHASE_TRANSITION_THRESHOLD}).")
    model_names = [m.strip() for m in args.models.split(',')]
    config_names = [c.strip() for c in args.configs.split(',')]

    header = f"  {'model':<22} | " + ' | '.join(f"{c:>12}" for c in config_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in model_names:
        row_cells = []
        for c in config_names:
            rs = [r for r in results if r.model_name == m and r.config == c]
            if not rs:
                cell = "n/a"
            else:
                converged = sum(1 for r in rs if r.transition_epoch > 0)
                trans_epochs = [r.transition_epoch for r in rs if r.transition_epoch > 0]
                if trans_epochs:
                    avg = sum(trans_epochs) / len(trans_epochs)
                    cell = f"{converged}/{len(rs)} @ ~{avg:.0f}"
                else:
                    cell = f"{converged}/{len(rs)}"
            row_cells.append(f"{cell:>12}")
        print(f"  {m:<22} | " + ' | '.join(row_cells))

    print()
    print(f"=== Final in_dist accuracy at N={args.train_size} ===")
    print(f"  Mean +/- std across {args.seeds} seeds")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in model_names:
        row_cells = []
        for c in config_names:
            rs = [r for r in results if r.model_name == m and r.config == c]
            if not rs:
                cell = "n/a"
            else:
                accs = [r.final_in_dist_acc for r in rs]
                cell = f"{np.mean(accs):.3f}+-{np.std(accs):.3f}"
            row_cells.append(f"{cell:>12}")
        print(f"  {m:<22} | " + ' | '.join(row_cells))

    print()
    print(f"=== Final ood_diff accuracy at N={args.train_size} ===")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in model_names:
        row_cells = []
        for c in config_names:
            rs = [r for r in results if r.model_name == m and r.config == c]
            if not rs:
                cell = "n/a"
            else:
                accs = [r.final_ood_diff_acc for r in rs]
                cell = f"{np.mean(accs):.3f}+-{np.std(accs):.3f}"
            row_cells.append(f"{cell:>12}")
        print(f"  {m:<22} | " + ' | '.join(row_cells))

    # Verdict
    print()
    print(f"=== Verdict ===")
    if 'toroidal_coupled' in model_names:
        get = lambda c: [r for r in results
                          if r.model_name == 'toroidal_coupled' and r.config == c]
        rate = lambda c: (sum(1 for r in get(c) if r.transition_epoch > 0),
                          len(get(c)))
        for c in config_names:
            n, total = rate(c)
            tag = (
                "ALL ESCAPED" if n == total else
                "MOSTLY ESCAPED" if n >= total * 2 / 3 else
                "MOSTLY STUCK" if n <= total / 3 else
                "SPLIT"
            )
            print(f"  toroidal_coupled @ {c:12s} : {n}/{total} seeds escaped  ({tag})")
        print()
        d, d_total = rate('default')
        if 'long' in config_names:
            l, l_total = rate('long')
            if d < d_total and l == l_total:
                print(f"  -> HYPOTHESIS A (BUDGET) SUPPORTED: doubling epochs")
                print(f"     reliably breaks the plateau. Recommend 60 ep at small N.")
            elif d < d_total and l < l_total:
                print(f"  -> HYPOTHESIS A NOT SUPPORTED on its own: longer")
                print(f"     training alone doesn't fully fix it.")
        if 'warmup' in config_names:
            w, w_total = rate('warmup')
            if d < d_total and w == w_total:
                print(f"  -> HYPOTHESIS B (WARMUP) SUPPORTED: linear LR warmup")
                print(f"     over 5 epochs makes the plateau reliably crossable.")


def save_csv(results: List[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # losses.csv -- one row per (run, epoch)
    with open(out_dir / 'losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'seed', 'config', 'epoch', 'loss'])
        for r in results:
            for i, l in enumerate(r.losses):
                writer.writerow([r.model_name, r.seed, r.config, i + 1, l])

    # summary.csv -- one row per run
    with open(out_dir / 'summary.csv', 'w', newline='') as f:
        fieldnames = ['model_name', 'seed', 'config', 'epochs', 'warmup_epochs',
                       'transition_epoch', 'final_loss',
                       'final_in_dist_acc', 'final_ood_diff_acc',
                       'n_params', 'wall_time_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def plot_loss_curves(results: List[RunResult], args, out_dir: Path) -> None:
    """One panel per model, lines colored by config, light traces per seed."""
    model_names = [m.strip() for m in args.models.split(',')]
    config_names = [c.strip() for c in args.configs.split(',')]

    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5),
                                sharey=True, squeeze=False)
    colors = {'default': 'C0', 'long': 'C1',
              'warmup': 'C2', 'long_warmup': 'C3'}

    for ax_i, m in enumerate(model_names):
        ax = axes[0, ax_i]
        for c in config_names:
            rs = [r for r in results if r.model_name == m and r.config == c]
            for r in rs:
                ax.plot(range(1, len(r.losses) + 1), r.losses,
                          color=colors.get(c, 'gray'), alpha=0.4, linewidth=1)
            # Bold mean line if all seeds have same length
            if rs and all(len(r.losses) == len(rs[0].losses) for r in rs):
                arr = np.array([r.losses for r in rs])
                mean = np.nanmean(arr, axis=0)
                ax.plot(range(1, len(mean) + 1), mean,
                          color=colors.get(c, 'gray'), linewidth=2.5,
                          label=f"{c} (n={len(rs)})")
        ax.axhline(PHASE_TRANSITION_THRESHOLD, color='red', linestyle='--',
                     linewidth=0.7, label=f'transition (loss<{PHASE_TRANSITION_THRESHOLD})')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        if ax_i == 0:
            ax.set_ylabel('training loss (log)')
        ax.set_title(m)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    fig.suptitle(f"Path J optimization: per-epoch loss curves (N={args.train_size})")
    fig.tight_layout()
    fig.savefig(out_dir / 'loss_curves.png', dpi=120)
    plt.close(fig)
    print(f"  [saved] {out_dir / 'loss_curves.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--models', type=str,
                   default='baseline,helical_coupled,toroidal_coupled,toroidal_coupled_wide')
    p.add_argument('--configs', type=str,
                   default='default,long,warmup,long_warmup',
                   help='Comma-separated subset of: ' + ','.join(CONFIGS))
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--train-size', type=int, default=64,
                   help='Plateau effect is sharpest at small N')
    p.add_argument('--eval-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--n-blanks-train', type=int, default=20)
    p.add_argument('--n-blanks-ood', type=int, default=35)
    p.add_argument('--embed-dim', type=int, default=64)
    p.add_argument('--osc-per-cell', type=int, default=16)
    p.add_argument('--kuramoto-steps', type=int, default=16)
    p.add_argument('--base-pool-seed', type=int, default=42)
    p.add_argument('--eval-seed', type=int, default=9999)
    p.add_argument('--device', type=str, default='cpu',
                   choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--out-dir', type=str, default='experiments/results_path_j_optim')
    args = p.parse_args()

    # Validate configs
    for c in args.configs.split(','):
        if c.strip() not in CONFIGS:
            raise ValueError(f"Unknown config: {c!r}. "
                              f"Valid: {list(CONFIGS)}")

    out_dir = Path(args.out_dir)
    results = run_sweep(args)
    save_csv(results, out_dir)
    print(f"\n[csv] losses + summary saved to {out_dir}/")
    plot_loss_curves(results, args, out_dir)
    report(results, args, out_dir)
    print(f"\nDone. Results in {out_dir}/")


if __name__ == '__main__':
    main()
