"""
experiments/akorn_smm_sudoku.py
================================
Encoder-only injection of SMM into a minimal AKOrN-style Kuramoto backbone,
evaluated on a tiny Sudoku-9x9 reconstruction task.

WHAT THIS IS:
  A sandbox integration test. Both model variants share an identical Kuramoto
  coupling layer and readout head; they differ only in how *initial phases*
  are produced from the input grid:

      AKOrNBaseline:  embed -> Linear -> Linear -> phases
      AKOrNWithSMM:   embed -> Linear -> HelicalEmbedding -> angle(zeta) -> phases

  Parameter counts are matched to within ~3% by construction.

WHAT THIS IS NOT:
  - A faithful reproduction of Miyato et al.'s full AKOrN architecture
  - Capable of solving Sudoku at SOTA accuracy (tiny model, tiny data)
  - A replacement for the proper sample-efficiency harness (next file)

GOALS:
  1. Verify SMM modules integrate cleanly into oscillator dynamics
  2. Confirm parameter parity between variants
  3. Establish that both models train (loss decreases, accuracy rises)
  4. Validate the harness pattern for full sample-efficiency runs later

USAGE:
    python experiments/akorn_smm_sudoku.py                    # default config
    python experiments/akorn_smm_sudoku.py --epochs 30        # longer
    python experiments/akorn_smm_sudoku.py --train-size 1024  # more data

On a 2024 M-series Mac CPU, the default config runs in ~3-5 minutes.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from smm import HelicalEmbedding


# =============================================================================
# Sudoku puzzle generator
# =============================================================================

# A known solved Sudoku grid (digits 1-9). All synthetic puzzles are derived
# from this one via valid Sudoku-preserving symmetries.
BASE_SOLVED = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
], dtype=np.int64)


def random_solved_sudoku(rng: np.random.Generator) -> np.ndarray:
    """Generate a valid solved Sudoku via random symmetries of BASE_SOLVED.

    Symmetries applied (each preserves Sudoku validity):
      1. Random digit permutation (1..9 -> permutation of 1..9)
      2. Permute rows within bands + permute the bands themselves
      3. Permute cols within stacks + permute the stacks themselves
    """
    grid = BASE_SOLVED.copy()

    # 1. Random digit permutation
    perm = rng.permutation(9) + 1
    digit_map = np.zeros(10, dtype=np.int64)
    for i in range(9):
        digit_map[i + 1] = perm[i]
    grid = digit_map[grid]

    # 2. Row permutation: shuffle rows within bands, shuffle bands
    band_order = rng.permutation(3)
    row_perm = []
    for b in band_order:
        rows_in_band = (rng.permutation(3) + b * 3).tolist()
        row_perm.extend(rows_in_band)
    grid = grid[row_perm]

    # 3. Column permutation: shuffle cols within stacks, shuffle stacks
    stack_order = rng.permutation(3)
    col_perm = []
    for s in stack_order:
        cols_in_stack = (rng.permutation(3) + s * 3).tolist()
        col_perm.extend(cols_in_stack)
    grid = grid[:, col_perm]

    return grid


def make_puzzle(solved: np.ndarray, n_blanks: int, rng: np.random.Generator):
    """Mask n_blanks cells of a solved grid (with 0) to create a puzzle."""
    puzzle = solved.copy()
    flat_idx = rng.permutation(81)[:n_blanks]
    for idx in flat_idx:
        puzzle[idx // 9, idx % 9] = 0
    return puzzle, solved


class SudokuDataset(Dataset):
    """Synthetic Sudoku-9x9 dataset.

    Each item: (puzzle, solution) as 9x9 long tensors.
    Empty cells in puzzle are 0. Solution always contains 1-9.
    """

    def __init__(self, n_samples: int, n_blanks: int = 20, seed: int = 0):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        rng = np.random.default_rng(seed)
        for _ in range(n_samples):
            solved = random_solved_sudoku(rng)
            puzzle, solution = make_puzzle(solved, n_blanks, rng)
            self.samples.append((puzzle, solution))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        puzzle, solution = self.samples[idx]
        return (
            torch.from_numpy(puzzle).long(),    # (9, 9), values 0-9 (0 = blank)
            torch.from_numpy(solution).long(),  # (9, 9), values 1-9
        )


# =============================================================================
# Minimal AKOrN-style Kuramoto coupling layer
# =============================================================================

class KuramotoLayer(nn.Module):
    """Iterated Kuramoto coupling on a grid of phase oscillators.

    Each cell holds `osc_per_cell` independent oscillator channels, all coupled
    cell-to-cell through a shared K matrix:

        theta_i^c <- theta_i^c + eta * (omega_i + sum_j K_ij sin(theta_j^c - theta_i^c))

    where i, j index cells (0..80) and c indexes the oscillator channel within
    a cell. This is a simplified version of AKOrN's generalized Kuramoto update;
    it captures the essential dynamics without per-channel coupling matrices.
    """
    def __init__(self, n_cells: int = 81, osc_per_cell: int = 4,
                 n_steps: int = 8, eta: float = 0.1):
        super().__init__()
        self.n_cells = n_cells
        self.osc_per_cell = osc_per_cell
        self.n_steps = n_steps
        self.eta = eta

        # Cell-to-cell coupling, shared across oscillator channels
        self.K = nn.Parameter(torch.randn(n_cells, n_cells) * 0.01)
        # Per-cell natural frequency, shared across channels within a cell
        self.omega = nn.Parameter(torch.zeros(n_cells, 1))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta: (B, n_cells, osc_per_cell)
        for _ in range(self.n_steps):
            # Pairwise phase differences across cells, per channel
            diff = theta.unsqueeze(2) - theta.unsqueeze(1)  # (B, N, N, C)
            # Apply same K to each channel: einsum mixes cells, preserves channels
            coupling = torch.einsum('ij,bijc->bic', self.K, diff.sin())
            theta = theta + self.eta * (self.omega + coupling)
        return theta


# =============================================================================
# Shared readout: phases -> per-cell digit logits
# =============================================================================

class CellReadout(nn.Module):
    """Maps final oscillator phases of each cell to 9 digit logits."""
    def __init__(self, osc_per_cell: int, hidden_dim: int = 32, n_digits: int = 9):
        super().__init__()
        # Use (sin, cos) of each phase as features — invariant to 2pi shifts
        self.head = nn.Sequential(
            nn.Linear(osc_per_cell * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_digits),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta: (B, n_cells, osc_per_cell)  ->  logits: (B, n_cells, n_digits)
        feat = torch.cat([theta.cos(), theta.sin()], dim=-1)
        return self.head(feat)


# =============================================================================
# Shared input embedding (used by both variants)
# =============================================================================

class GridEmbedding(nn.Module):
    """Embeds (digit, row, col) for each cell into a vector of `embed_dim`."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.digit_embed = nn.Embedding(10, embed_dim)   # 0=blank, 1-9=digits
        self.row_embed = nn.Embedding(9, embed_dim)
        self.col_embed = nn.Embedding(9, embed_dim)

    def forward(self, puzzle: torch.Tensor) -> torch.Tensor:
        # puzzle: (B, 9, 9) long
        B = puzzle.shape[0]
        device = puzzle.device
        rows = torch.arange(9, device=device).view(1, 9, 1).expand(B, 9, 9)
        cols = torch.arange(9, device=device).view(1, 1, 9).expand(B, 9, 9)
        e = self.digit_embed(puzzle) + self.row_embed(rows) + self.col_embed(cols)
        return e.view(B, 81, -1)  # (B, 81, embed_dim)


# =============================================================================
# Model variants
# =============================================================================

class AKOrNBaseline(nn.Module):
    """Baseline: GridEmbedding -> Linear -> Linear -> phases -> Kuramoto -> Readout."""
    def __init__(self, embed_dim: int = 24, intermediate_dim: int = 12,
                 osc_per_cell: int = 4, kuramoto_steps: int = 8):
        super().__init__()
        self.osc_per_cell = osc_per_cell
        self.embed = GridEmbedding(embed_dim)
        # Same intermediate projection as SMM variant (for fair comparison)
        self.proj_intermediate = nn.Linear(embed_dim, intermediate_dim)
        # Project intermediate features directly to phases
        self.proj_phases = nn.Linear(intermediate_dim, osc_per_cell)
        self.kuramoto = KuramotoLayer(n_cells=81, osc_per_cell=osc_per_cell,
                                       n_steps=kuramoto_steps)
        self.readout = CellReadout(osc_per_cell)

    def forward(self, puzzle: torch.Tensor) -> torch.Tensor:
        e = self.embed(puzzle)                       # (B, 81, embed_dim)
        h = F.gelu(self.proj_intermediate(e))        # (B, 81, intermediate_dim)
        phases = self.proj_phases(h)                 # (B, 81, osc_per_cell)
        phases = self.kuramoto(phases)
        logits = self.readout(phases)                # (B, 81, 9)
        return logits


class AKOrNWithSMM(nn.Module):
    """SMM variant: GridEmbedding -> Linear -> HelicalEmbedding -> angle -> Kuramoto -> Readout.

    The input projection produces a 3*osc_per_cell vector per cell, which is
    then split into (r, theta, z) chunks by HelicalEmbedding. The phase of the
    resulting complex zeta is used directly as the Kuramoto initial condition.
    """
    def __init__(self, embed_dim: int = 24, intermediate_dim: int = 12,
                 osc_per_cell: int = 4, kuramoto_steps: int = 8):
        super().__init__()
        self.osc_per_cell = osc_per_cell
        assert intermediate_dim == 3 * osc_per_cell, \
            f"intermediate_dim must equal 3 * osc_per_cell ({3 * osc_per_cell}); got {intermediate_dim}"
        self.embed = GridEmbedding(embed_dim)
        # Same intermediate projection as baseline (for fair comparison)
        self.proj_intermediate = nn.Linear(embed_dim, intermediate_dim)
        # SMM injection: helical embedding produces complex zeta; we extract phase
        self.helical = HelicalEmbedding(input_dim=intermediate_dim)
        self.kuramoto = KuramotoLayer(n_cells=81, osc_per_cell=osc_per_cell,
                                       n_steps=kuramoto_steps)
        self.readout = CellReadout(osc_per_cell)

    def forward(self, puzzle: torch.Tensor) -> torch.Tensor:
        e = self.embed(puzzle)                       # (B, 81, embed_dim)
        h = F.gelu(self.proj_intermediate(e))        # (B, 81, intermediate_dim)
        zeta, _ = self.helical(h)                    # zeta: (B, 81, osc_per_cell) complex
        phases = zeta.angle()                        # extract phase from complex repr
        phases = self.kuramoto(phases)
        logits = self.readout(phases)                # (B, 81, 9)
        return logits


# =============================================================================
# Training & evaluation
# =============================================================================

@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc_blank: float    # accuracy on cells that were blank in puzzle
    val_acc_full: float     # accuracy on all 81 cells


def loss_and_metrics(logits: torch.Tensor, puzzle: torch.Tensor, solution: torch.Tensor):
    """Cross-entropy on all 81 cells; track accuracy separately on blanks vs givens.

    logits:   (B, 81, 9)   — predicted distribution over digits 1..9
    puzzle:   (B, 9, 9)    — input (0 = blank, 1..9 = given)
    solution: (B, 9, 9)    — ground truth (1..9)
    """
    B = puzzle.shape[0]
    logits_flat = logits.reshape(B * 81, 9)
    target_flat = (solution.reshape(B * 81) - 1).long()  # 0..8 indexing
    loss = F.cross_entropy(logits_flat, target_flat)

    pred_flat = logits_flat.argmax(dim=-1)
    correct = (pred_flat == target_flat)
    blank_mask = (puzzle.reshape(B * 81) == 0)

    acc_full = correct.float().mean().item()
    acc_blank = correct[blank_mask].float().mean().item() if blank_mask.any() else 0.0
    return loss, acc_full, acc_blank


def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total_loss = 0.0
    total_acc_full = 0.0
    total_acc_blank = 0.0
    n_batches = 0
    with torch.no_grad():
        for puzzle, solution in loader:
            puzzle, solution = puzzle.to(device), solution.to(device)
            logits = model(puzzle)
            loss, acc_full, acc_blank = loss_and_metrics(logits, puzzle, solution)
            total_loss += loss.item()
            total_acc_full += acc_full
            total_acc_blank += acc_blank
            n_batches += 1
    return (total_loss / n_batches,
            total_acc_full / n_batches,
            total_acc_blank / n_batches)


def train_one_model(name: str, model: nn.Module, train_loader: DataLoader,
                    val_loader: DataLoader, epochs: int, lr: float, device: str,
                    log_every: int = 5) -> List[EpochStats]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: List[EpochStats] = []

    print(f"\n=== Training {name} ===")
    print(f"Device: {device}, params: {count_params(model):,}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for puzzle, solution in train_loader:
            puzzle, solution = puzzle.to(device), solution.to(device)
            logits = model(puzzle)
            loss, _, _ = loss_and_metrics(logits, puzzle, solution)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        val_loss, val_acc_full, val_acc_blank = evaluate(model, val_loader, device)
        history.append(EpochStats(epoch, train_loss, val_loss, val_acc_blank, val_acc_full))

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc(blank) {val_acc_blank:.3f} | "
                  f"val_acc(all) {val_acc_full:.3f} | {elapsed:5.1f}s")

    return history


# =============================================================================
# Utilities
# =============================================================================

def count_params(model: nn.Module) -> int:
    """Count trainable params, treating complex tensors as 2 reals."""
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        total += p.numel() * (2 if p.is_complex() else 1)
    return total


def assert_param_parity(m1: nn.Module, m2: nn.Module, tol: float = 0.05):
    p1, p2 = count_params(m1), count_params(m2)
    ratio = max(p1, p2) / min(p1, p2)
    print(f"\nParameter parity check:")
    print(f"  baseline: {p1:,}")
    print(f"  smm     : {p2:,}")
    print(f"  ratio   : {ratio:.4f}  (tolerance: {1 + tol:.2f})")
    if ratio > 1 + tol:
        raise ValueError(
            f"Parameter mismatch beyond tolerance: ratio {ratio:.3f} > {1 + tol:.3f}. "
            "Adjust embed_dim, intermediate_dim, or osc_per_cell to bring counts in line."
        )
    print(f"  [OK] within tolerance")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-size', type=int, default=256)
    parser.add_argument('--val-size', type=int, default=64)
    parser.add_argument('--n-blanks', type=int, default=20,
                        help='Cells masked per puzzle (out of 81)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--embed-dim', type=int, default=24)
    parser.add_argument('--osc-per-cell', type=int, default=4)
    parser.add_argument('--kuramoto-steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'mps', 'cuda'],
                        help='cpu is safest; mps complex-tensor support is incomplete in some torch builds')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== AKOrN+SMM Sudoku sandbox ===")
    print(f"epochs={args.epochs}  train_size={args.train_size}  val_size={args.val_size}  "
          f"n_blanks={args.n_blanks}  device={args.device}")

    # -- Data --
    print(f"\nGenerating {args.train_size + args.val_size} synthetic puzzles...")
    full = SudokuDataset(args.train_size + args.val_size,
                         n_blanks=args.n_blanks, seed=args.seed)
    train_set = torch.utils.data.Subset(full, range(args.train_size))
    val_set = torch.utils.data.Subset(full, range(args.train_size,
                                                   args.train_size + args.val_size))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # -- Models --
    intermediate_dim = 3 * args.osc_per_cell  # required by HelicalEmbedding

    baseline = AKOrNBaseline(
        embed_dim=args.embed_dim,
        intermediate_dim=intermediate_dim,
        osc_per_cell=args.osc_per_cell,
        kuramoto_steps=args.kuramoto_steps,
    )
    smm = AKOrNWithSMM(
        embed_dim=args.embed_dim,
        intermediate_dim=intermediate_dim,
        osc_per_cell=args.osc_per_cell,
        kuramoto_steps=args.kuramoto_steps,
    )

    assert_param_parity(baseline, smm, tol=0.05)

    # -- Train both --
    hist_baseline = train_one_model('AKOrN-Baseline', baseline, train_loader,
                                     val_loader, args.epochs, args.lr, args.device)
    hist_smm = train_one_model('AKOrN+SMM',     smm,      train_loader,
                                val_loader, args.epochs, args.lr, args.device)

    # -- Summary --
    print(f"\n=== Final results ===")
    print(f"{'metric':<28} {'baseline':>12} {'smm':>12}")
    print(f"{'-'*54}")
    final_b = hist_baseline[-1]
    final_s = hist_smm[-1]
    print(f"{'val_loss':<28} {final_b.val_loss:>12.4f} {final_s.val_loss:>12.4f}")
    print(f"{'val_acc (blanks only)':<28} {final_b.val_acc_blank:>12.3f} {final_s.val_acc_blank:>12.3f}")
    print(f"{'val_acc (all cells)':<28} {final_b.val_acc_full:>12.3f} {final_s.val_acc_full:>12.3f}")

    print(f"\n[note] This is an integration sanity-check, not a benchmark.")
    print(f"       Tiny model + tiny data => low absolute accuracy expected.")
    print(f"       The point is: both models train, parameter counts match,")
    print(f"       and the SMM modules plug into the oscillator backbone")
    print(f"       without errors. Use the sample-efficiency harness (next file)")
    print(f"       for proper N_50 comparisons across multiple training-set sizes.")


if __name__ == '__main__':
    main()
