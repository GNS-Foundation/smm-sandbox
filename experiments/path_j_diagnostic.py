"""
experiments/path_j_diagnostic.py
=================================
What did the helical/toroidal metric actually learn?

Path J showed `helical_coupled` (baseline encoder + HelicalCoupledKuramoto)
beats baseline by +14/+7/+1pp at N=64/256/1024 on in-distribution. The
diagnostic on the helical version returned PARTIAL A: peer/non-peer K-mass
ratio of 1.50, with a clear hierarchy:
    same_row_only:  0.453   (1.52x non-peer)
    same_col_only:  0.450   (1.51x non-peer)
    same_box_only:  0.354   (1.19x non-peer)  <-- bottlenecked
    non_peer:       0.298

The single-helix metric handles 1D row/col fine but only partially captures
the 2D box pattern. Toroidal extension (T^2 = S^1 x S^1) gives the metric
two independent angular axes; the prediction is that box coupling will
rise toward row/col level, lifting the overall ratio.

This script trains ONE model (--model helical_coupled or --model
toroidal_coupled) and runs the same three tests, so you can directly
compare the outputs across topologies.

Tests:
  Test 1 (peer-mass): mean K[i,j] across pair categories
                      (same_row, same_col, same_box, non-peer).
  Test 2 (clustering): PCA of cell position features to 2D, then
                       intra/inter-group distance ratio per axis.
  Test 3 (heatmap):    81x81 K matrix visualization.

Usage:
    python experiments/path_j_diagnostic.py
    python experiments/path_j_diagnostic.py --model toroidal_coupled
    python experiments/path_j_diagnostic.py --model toroidal_coupled_wide \\
        --train-size 256 --epochs 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from akorn_smm_sudoku import (
    AKOrNHelicalCoupled, AKOrNToroidalCoupled,
    loss_and_metrics, count_params,
)
from sample_efficiency_ood import (
    SudokuDatasetOOD, random_solved_sudoku,
)


def make_diag_model(model_name: str, *, embed_dim: int, osc_per_cell: int,
                     kuramoto_steps: int) -> nn.Module:
    """Build the model variant under test. Returns a model whose `kuramoto`
    layer exposes both `coupling()` and `position_features()`."""
    intermediate_dim = 3 * osc_per_cell
    if model_name == 'helical_coupled':
        return AKOrNHelicalCoupled(
            embed_dim=embed_dim, intermediate_dim=intermediate_dim,
            osc_per_cell=osc_per_cell, kuramoto_steps=kuramoto_steps,
        )
    if model_name == 'toroidal_coupled':
        # Matched-param: n_helical_channels=12 vs helical's 16
        return AKOrNToroidalCoupled(
            embed_dim=embed_dim, intermediate_dim=intermediate_dim,
            osc_per_cell=osc_per_cell, kuramoto_steps=kuramoto_steps,
            n_helical_channels=12,
        )
    if model_name == 'toroidal_coupled_wide':
        return AKOrNToroidalCoupled(
            embed_dim=embed_dim, intermediate_dim=intermediate_dim,
            osc_per_cell=osc_per_cell, kuramoto_steps=kuramoto_steps,
            n_helical_channels=16,
        )
    raise ValueError(f"Unknown diagnostic model: {model_name!r}")


# Sudoku structure helpers -----------------------------------------------------

def cell_unit_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rows, cols, boxes) of length 81, the unit each cell belongs to."""
    rows = np.arange(81) // 9
    cols = np.arange(81) % 9
    boxes = (rows // 3) * 3 + (cols // 3)
    return rows, cols, boxes


def peer_masks() -> dict:
    """Return boolean (81, 81) masks for each Sudoku relationship type."""
    rows, cols, boxes = cell_unit_indices()
    same_row = rows[:, None] == rows[None, :]
    same_col = cols[:, None] == cols[None, :]
    same_box = boxes[:, None] == boxes[None, :]
    eye = np.eye(81, dtype=bool)
    same_row &= ~eye
    same_col &= ~eye
    same_box &= ~eye
    peer = same_row | same_col | same_box
    non_peer = ~peer & ~eye
    return {
        'same_row_only': same_row & ~same_col & ~same_box,
        'same_col_only': same_col & ~same_row & ~same_box,
        'same_box_only': same_box & ~same_row & ~same_col,
        'peer_any':      peer,
        'non_peer':      non_peer,
        'self':          eye,
    }


# Training a single model ------------------------------------------------------

def train_diagnostic_model(args) -> nn.Module:
    print(f"Training one {args.model} at standard scale-up config...")
    print(f"  N={args.train_size}, epochs={args.epochs}, lr={args.lr}, "
          f"seed={args.seed}")

    # Build dataset (same convention as sample_efficiency_ood.py)
    base_rng = np.random.default_rng(args.base_pool_seed)
    bases = [random_solved_sudoku(base_rng) for _ in range(40)]
    train_set = SudokuDatasetOOD(args.train_size, bases,
                                  n_blanks=args.n_blanks, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Build model
    torch.manual_seed(args.seed)
    model = make_diag_model(
        args.model,
        embed_dim=args.embed_dim,
        osc_per_cell=args.osc_per_cell,
        kuramoto_steps=args.kuramoto_steps,
    )
    n_params = count_params(model)
    print(f"  params: {n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for puzzle, solution in train_loader:
            logits = model(puzzle)
            loss, _, _ = loss_and_metrics(logits, puzzle, solution)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"diverged at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{args.epochs}  "
                  f"loss={np.mean(epoch_losses):.4f}  "
                  f"({time.time() - t0:.1f}s)")
    print(f"  training done in {time.time() - t0:.1f}s")
    return model


# Test 1: peer-mass analysis ---------------------------------------------------

def test_peer_mass(K: np.ndarray, out_dir: Path) -> dict:
    """Compare mean K over peer pairs vs non-peer pairs."""
    masks = peer_masks()
    stats = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        vals = K[mask]
        stats[name] = {
            'mean': float(vals.mean()),
            'std':  float(vals.std()),
            'n':    int(mask.sum()),
        }

    print(f"\n=== Test 1: peer-mass analysis ===")
    print(f"  K[i,j] statistics by Sudoku relationship:")
    header = f"    {'relation':<18}{'n':>8}{'mean K':>14}{'std K':>14}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name in ['self', 'same_row_only', 'same_col_only', 'same_box_only',
                 'peer_any', 'non_peer']:
        if name in stats:
            s = stats[name]
            print(f"    {name:<18}{s['n']:>8d}{s['mean']:>14.5f}{s['std']:>14.5f}")

    if 'peer_any' in stats and 'non_peer' in stats:
        ratio = stats['peer_any']['mean'] / max(stats['non_peer']['mean'], 1e-12)
        print(f"\n  peer/non-peer mean ratio: {ratio:.2f}x")
        if ratio > 2.0:
            verdict = "STRONG structural learning (peers >> non-peers)"
        elif ratio > 1.3:
            verdict = "MODERATE structural learning"
        elif ratio > 1.1:
            verdict = "WEAK structural learning"
        else:
            verdict = "NO structural learning (peer-mass ~ non-peer mass)"
        print(f"  verdict: {verdict}")
        stats['ratio_peer_nonpeer'] = ratio
        stats['verdict'] = verdict

    # Histogram plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(K.min(), K.max(), 40)
    ax.hist(K[masks['peer_any']], bins=bins, alpha=0.5,
            label=f"peers (n={masks['peer_any'].sum()})", color='C1')
    ax.hist(K[masks['non_peer']], bins=bins, alpha=0.5,
            label=f"non-peers (n={masks['non_peer'].sum()})", color='C0')
    ax.set_xlabel("K[i,j]")
    ax.set_ylabel("count")
    ax.set_title("Coupling-weight distribution: peers vs non-peers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'k_distribution_peers_vs_nonpeers.png', dpi=120)
    plt.close(fig)
    print(f"  [saved] {out_dir / 'k_distribution_peers_vs_nonpeers.png'}")

    return stats


# Test 2: position clustering --------------------------------------------------

def cell_features(model: nn.Module) -> np.ndarray:
    """Stack per-cell position features into (81, F) array.

    Uses the model.kuramoto.position_features() method, which is
    implemented by both HelicalCoupledKuramoto and ToroidalCoupledKuramoto
    so this function works with either topology. The feature dimension
    differs (4*H for helix, 6*H for torus) but the per-cell layout is
    consistent and PCA-compatible.
    """
    return model.kuramoto.position_features().cpu().numpy()


def pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center X and project to top-2 PCs. Returns (Y, explained_variance_ratio)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = U[:, :2] * S[:2]
    var_total = (S ** 2).sum()
    ev_ratio = (S[:2] ** 2) / var_total if var_total > 0 else np.zeros(2)
    return Y, ev_ratio


def cluster_score(Y: np.ndarray, group_labels: np.ndarray) -> float:
    """Mean intra-group pairwise distance / mean inter-group pairwise distance.

    Score < 0.7 means tight intra-group clustering relative to inter-group spread
    (= structural learning along this axis).
    Score ~ 1.0 means no preferred structure.
    """
    n = Y.shape[0]
    dists = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)
    same_group = group_labels[:, None] == group_labels[None, :]
    eye = np.eye(n, dtype=bool)
    intra = dists[same_group & ~eye]
    inter = dists[~same_group & ~eye]
    if len(intra) == 0 or len(inter) == 0:
        return float('nan')
    return float(intra.mean() / inter.mean())


def test_position_clustering(model: AKOrNHelicalCoupled, out_dir: Path) -> dict:
    """PCA the helical positions and test clustering against Sudoku structure."""
    print(f"\n=== Test 2: position clustering ===")
    X = cell_features(model)
    Y, ev_ratio = pca_2d(X)
    print(f"  PCA: top-2 components explain "
          f"{100*ev_ratio[0]:.1f}% + {100*ev_ratio[1]:.1f}% "
          f"= {100*ev_ratio.sum():.1f}% of variance")

    rows, cols, boxes = cell_unit_indices()
    scores = {
        'row': cluster_score(Y, rows),
        'col': cluster_score(Y, cols),
        'box': cluster_score(Y, boxes),
    }
    print(f"  Cluster scores (intra/inter mean distance, lower = tighter):")
    for axis, s in scores.items():
        if s < 0.7:
            tag = "STRONG clustering"
        elif s < 0.85:
            tag = "moderate clustering"
        elif s < 0.95:
            tag = "weak clustering"
        else:
            tag = "no clustering"
        print(f"    by {axis}: {s:.3f}  ({tag})")

    # 3-panel scatter: position colored by row, col, box
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (axis, labels) in zip(axes, [('row', rows), ('col', cols), ('box', boxes)]):
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=80,
                          edgecolor='black', linewidth=0.5)
        ax.set_title(f"colored by {axis}  (cluster_score = {scores[axis]:.3f})")
        ax.set_xlabel(f"PC1 ({100*ev_ratio[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100*ev_ratio[1]:.1f}%)")
        plt.colorbar(sc, ax=ax, label=axis, ticks=range(9))
    fig.suptitle("Cell helical positions (PCA), 3 colorings")
    fig.tight_layout()
    fig.savefig(out_dir / 'position_pca.png', dpi=120)
    plt.close(fig)
    print(f"  [saved] {out_dir / 'position_pca.png'}")

    return {'pca_explained_variance': ev_ratio.tolist(),
            'cluster_scores': scores}


# Test 3: K matrix heatmap -----------------------------------------------------

def test_coupling_heatmap(K: np.ndarray, out_dir: Path) -> None:
    """Plot the 81x81 K matrix as a heatmap, with row/col/box separators."""
    print(f"\n=== Test 3: coupling-matrix visualization ===")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(K, cmap='viridis', aspect='equal')
    # Major dividers at every 9 (rows) and every 27 (boxes)
    for k in range(9, 81, 9):
        ax.axhline(k - 0.5, color='white', linewidth=0.5, alpha=0.7)
        ax.axvline(k - 0.5, color='white', linewidth=0.5, alpha=0.7)
    for k in range(27, 81, 27):
        ax.axhline(k - 0.5, color='red', linewidth=1.0, alpha=0.7)
        ax.axvline(k - 0.5, color='red', linewidth=1.0, alpha=0.7)
    ax.set_title(f"Learned coupling matrix K  (range [{K.min():.4f}, {K.max():.4f}])")
    ax.set_xlabel("cell j (row-major)")
    ax.set_ylabel("cell i (row-major)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / 'coupling_heatmap.png', dpi=120)
    plt.close(fig)
    print(f"  [saved] {out_dir / 'coupling_heatmap.png'}")


# Main -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--model', type=str, default='helical_coupled',
                   choices=['helical_coupled', 'toroidal_coupled',
                            'toroidal_coupled_wide'],
                   help='Which coupled-Kuramoto variant to diagnose')
    p.add_argument('--train-size', type=int, default=256,
                   help='Path J showed clearest signal at N=256')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--base-pool-seed', type=int, default=42,
                   help='RNG seed for the base puzzle pool. Match to your '
                        'main sweep for consistent comparison.')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--n-blanks', type=int, default=20)
    p.add_argument('--embed-dim', type=int, default=64)
    p.add_argument('--osc-per-cell', type=int, default=16)
    p.add_argument('--kuramoto-steps', type=int, default=16)
    p.add_argument('--out-dir', type=str, default=None,
                   help='Default: experiments/results_path_j_diag_<model>')
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f'experiments/results_path_j_diag_{args.model}'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model = train_diagnostic_model(args)

    # Extract learned K (after training, deterministic since K is derived from
    # learned positions via the helical metric — no randomness at eval)
    model.eval()
    with torch.no_grad():
        K = model.kuramoto.coupling().cpu().numpy()

    # Run tests
    test_coupling_heatmap(K, out_dir)
    stats1 = test_peer_mass(K, out_dir)
    stats2 = test_position_clustering(model, out_dir)

    # Summary
    print(f"\n=== Diagnostic verdict ===")
    ratio = stats1.get('ratio_peer_nonpeer', float('nan'))
    cluster_min = min(stats2['cluster_scores'].values())
    print(f"  peer/non-peer K-mass ratio : {ratio:.2f}")
    print(f"  best cluster score (lower=tighter) : {cluster_min:.3f}")

    if ratio > 2.0 and cluster_min < 0.7:
        print(f"\n  --> INTERPRETATION (A): the metric LEARNED Sudoku peer structure.")
        print(f"      Coupling is much stronger between peers than non-peers, AND")
        print(f"      cell positions cluster by row/col/box. The structural prior")
        print(f"      is doing real geometric work. Toroidal/conformal variants are")
        print(f"      worth exploring; the compounding-with-scale question is open.")
    elif ratio > 1.3 or cluster_min < 0.85:
        print(f"\n  --> INTERPRETATION (PARTIAL A): some structural learning but")
        print(f"      not clean. The metric captures part of the prior; the rest")
        print(f"      is in-distribution fitting. Mixed verdict.")
    else:
        print(f"\n  --> INTERPRETATION (B): the metric did NOT learn Sudoku structure.")
        print(f"      Coupling is roughly uniform across peer/non-peer pairs and")
        print(f"      positions are unstructured. The Path J win is in-distribution")
        print(f"      fitting, NOT compositional understanding. The 3.4x worse")
        print(f"      generalization-gap at small N supports this read. Toroidal/")
        print(f"      conformal variants are unlikely to help; pivot needed.")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == '__main__':
    main()
