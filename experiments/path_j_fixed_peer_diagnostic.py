"""
experiments/path_j_fixed_peer_diagnostic.py
============================================
Inspect the hand-crafted control's converged coupling.

After the Acid Test showed fixed_peer_binary dominates every SMM variant,
this script answers the residual question: what scalar value does the
single learnable parameter (log_scale) converge to, and what does the
final K matrix look like?

Train one fixed_peer_binary model at the standard scale-up config (N=256,
30 epochs, lr=1e-2). Report:
  - log_scale at init vs converged
  - K[peer] and K[non-peer] mean values
  - Compare to helical_coupled's converged peer/non-peer K-mass
    (the Path J diagnostic baseline)

Output:
    experiments/results_path_j_fixed_peer_diag/
        scale_summary.txt    -- converged scalar, K stats
        coupling_heatmap.png -- 81x81 K matrix visualization

Usage:
    python experiments/path_j_fixed_peer_diagnostic.py
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from akorn_smm_sudoku import (
    AKOrNFixedPeerCoupled, loss_and_metrics, count_params,
)
from sample_efficiency_ood import (
    SudokuDatasetOOD, random_solved_sudoku,
)


def train_diagnostic_model(args) -> AKOrNFixedPeerCoupled:
    print(f"Training fixed_peer_binary at standard scale-up config...")
    print(f"  N={args.train_size}, epochs={args.epochs}, lr={args.lr}, "
          f"seed={args.seed}")

    base_rng = np.random.default_rng(args.base_pool_seed)
    bases = [random_solved_sudoku(base_rng) for _ in range(40)]
    train_set = SudokuDatasetOOD(args.train_size, bases,
                                  n_blanks=args.n_blanks, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    torch.manual_seed(args.seed)
    intermediate_dim = 3 * args.osc_per_cell
    model = AKOrNFixedPeerCoupled(
        embed_dim=args.embed_dim, intermediate_dim=intermediate_dim,
        osc_per_cell=args.osc_per_cell, kuramoto_steps=args.kuramoto_steps,
        mode='binary',
    )
    print(f"  params: {count_params(model)}")

    init_scale = model.kuramoto.log_scale.detach().exp().item()
    print(f"  init log_scale = {model.kuramoto.log_scale.item():.4f} "
          f"(scale = {init_scale:.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for puzzle, solution in train_loader:
            logits = model(puzzle)
            loss, _, _ = loss_and_metrics(logits, puzzle, solution)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            cur_scale = model.kuramoto.log_scale.detach().exp().item()
            print(f"    epoch {epoch+1:2d}/{args.epochs}  "
                  f"loss={np.mean(epoch_losses):.4f}  scale={cur_scale:.4f}  "
                  f"({time.time()-t0:.1f}s)")
    print(f"  training done in {time.time()-t0:.1f}s")
    return model


def report_scale_and_coupling(model: AKOrNFixedPeerCoupled,
                                 out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    K = model.kuramoto.coupling().detach().cpu().numpy()
    peer = model.kuramoto.peer_matrix.cpu().numpy().astype(bool)
    non_peer = model.kuramoto.non_peer_matrix.cpu().numpy().astype(bool)

    log_scale = model.kuramoto.log_scale.item()
    scale = math.exp(log_scale)

    K_peer = K[peer]
    K_non_peer = K[non_peer]

    summary_lines = [
        "=== fixed_peer_binary diagnostic ===",
        "",
        f"Learnable scalar (log_scale -> scale via exp):",
        f"  log_scale  = {log_scale:.4f}",
        f"  scale      = {scale:.4f}",
        "",
        f"K[i,j] statistics:",
        f"  K[peer]_mean   = {K_peer.mean():.4f}  (= scale, by construction)",
        f"  K[non-peer]    = {K_non_peer.mean():.4f}  (= 0, by construction)",
        f"  peer/non-peer ratio: infinity (binary)",
        "",
        f"Comparison to helical_coupled's learned K (Path J diagnostic):",
        f"  helical K[peer]      ~ 0.42-0.45",
        f"  helical K[non-peer]  ~ 0.30",
        f"  helical ratio        ~ 1.50x",
        "",
        f"Interpretation:",
        f"  fixed_peer_binary uses a perfectly binary peer/non-peer prior.",
        f"  Single learnable scalar converged to {scale:.4f}.",
        f"  Sum of coupling per cell = 20 * {scale:.4f} = {20*scale:.4f}",
        f"  (vs. helical ~80 * 0.34 = 27 with much softer hierarchy)",
    ]
    summary = "\n".join(summary_lines)
    print()
    print(summary)
    (out_dir / "scale_summary.txt").write_text(summary + "\n")
    print(f"  [saved] {out_dir / 'scale_summary.txt'}")

    # Coupling heatmap with row/col/box separators
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = K.max() * 1.1 if K.max() > 0 else 1.0
    im = ax.imshow(K, cmap='magma', vmin=0, vmax=vmax)
    for i in range(1, 9):
        lw = 1.5 if i % 3 == 0 else 0.4
        ax.axhline(i * 9 - 0.5, color='cyan', linewidth=lw, alpha=0.6)
        ax.axvline(i * 9 - 0.5, color='cyan', linewidth=lw, alpha=0.6)
    ax.set_title(f"fixed_peer_binary coupling (scale={scale:.4f})")
    plt.colorbar(im, ax=ax, label='K[i,j]')
    fig.tight_layout()
    fig.savefig(out_dir / 'coupling_heatmap.png', dpi=120)
    plt.close(fig)
    print(f"  [saved] {out_dir / 'coupling_heatmap.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--train-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--base-pool-seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--n-blanks', type=int, default=20)
    p.add_argument('--embed-dim', type=int, default=64)
    p.add_argument('--osc-per-cell', type=int, default=16)
    p.add_argument('--kuramoto-steps', type=int, default=16)
    p.add_argument('--out-dir', type=str,
                   default='experiments/results_path_j_fixed_peer_diag')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    model = train_diagnostic_model(args)
    report_scale_and_coupling(model, out_dir)
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == '__main__':
    main()
