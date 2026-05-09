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

from smm import HelicalEmbedding, RotationalHelicalLinear, PhaseCoherenceGate


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

    Path G additions
    ----------------
    forward() now accepts:
      * `n_steps` -- per-call override of self.n_steps. Trains at the default,
        but at eval can run more iterations (test-time compute extension,
        AKOrN's first reported improvement: 18%->51% on Sudoku OOD).
      * `return_trajectory` -- if True, also returns all intermediate phase
        snapshots, used for energy-based voting (AKOrN's second reported
        improvement: 51%->90% on Sudoku OOD).

    compute_energy(theta) returns the Kuramoto sync energy
        E(theta) = -sum_{i,j,c} K[i,j] * cos(theta[j,c] - theta[i,c])
    which is minimized when oscillators are aligned along the gradient of K.
    Lower energy = more coherent state, used to select the best timestep.
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

    def forward(self, theta: torch.Tensor,
                n_steps: int = None,
                return_trajectory: bool = False):
        # theta: (B, n_cells, osc_per_cell)
        n = self.n_steps if n_steps is None else n_steps
        if return_trajectory:
            trajectory = [theta]
        for _ in range(n):
            # Pairwise phase differences across cells, per channel
            diff = theta.unsqueeze(2) - theta.unsqueeze(1)  # (B, N, N, C)
            # Apply same K to each channel: einsum mixes cells, preserves channels
            coupling = torch.einsum('ij,bijc->bic', self.K, diff.sin())
            theta = theta + self.eta * (self.omega + coupling)
            if return_trajectory:
                trajectory.append(theta)
        if return_trajectory:
            return theta, torch.stack(trajectory, dim=1)  # (B, T+1, N, C)
        return theta

    def compute_energy(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-batch Kuramoto sync energy.

        E_b = -sum_{i,j,c} K[i,j] * cos(theta[b,j,c] - theta[b,i,c])

        Returns:
            energy: (B,) tensor. Lower is more synchronized.
        """
        # theta: (B, N, C)
        diff = theta.unsqueeze(2) - theta.unsqueeze(1)        # (B, N, N, C)
        cos_diff = diff.cos().sum(dim=-1)                      # (B, N, N)
        energy = -(self.K * cos_diff).sum(dim=(-1, -2))        # (B,)
        return energy


# =============================================================================
# Path J: Kuramoto with helical-distance-modulated coupling
# =============================================================================

class HelicalCoupledKuramoto(nn.Module):
    """Kuramoto layer where the cell-to-cell coupling matrix is *derived* from
    pairwise helical-deviation distances between learnable per-cell positions,
    rather than being a free 81x81 parameter.

    This is the experimental answer to: "what if we wire Module 2 (the
    HelicalDeviationMetric) into the dynamics?"

    Each cell i gets a learnable helical position (r_i, phi_i, z_i) per
    helical channel (independent of input). At forward time we compute
    pairwise helical-deviation distances ds_{ij} between these positions
    using the same metric as Module 2:

        ds^2_{ij} = dr^2 + r_avg^2 (dphi - omega dz)^2 + dz^2

    and turn distances into coupling weights via a Gaussian kernel:

        K_{ij} = exp( -ds^2_{ij} / sigma^2 ) * (1 - delta_{ij})

    The Kuramoto update is unchanged otherwise:

        theta_i^c <- theta_i^c + eta * (omega_i + sum_j K_{ij} sin(theta_j^c - theta_i^c))

    Modes:
      replace -- K is purely the helical-distance kernel above.
      mix     -- K = sigmoid(alpha) * K_helical + (1 - sigmoid(alpha)) * K_learned,
                 where K_learned is a free 81x81 parameter (alpha learnable).
                 Lets the optimizer interpolate between full structural prior
                 and full free coupling.

    Param accounting (n_cells=81, n_helical_channels=H):
      cell_r, cell_phi, cell_z : 3 * 81 * H
      omega                    : H
      omega_kuramoto           : 81
      sigma_log                : 1
      mix mode adds: K_learned (81*81 = 6561), alpha (1)

    Default: replace mode, n_helical_channels matched to osc_per_cell so the
    metric channels and oscillator channels align.
    """
    def __init__(self, n_cells: int = 81, osc_per_cell: int = 4,
                 n_steps: int = 8, eta: float = 0.1,
                 n_helical_channels: int = None,
                 mode: str = 'replace',
                 sigma_init: float = 1.0):
        super().__init__()
        if n_helical_channels is None:
            n_helical_channels = osc_per_cell
        self.n_cells = n_cells
        self.osc_per_cell = osc_per_cell
        self.n_steps = n_steps
        self.eta = eta
        self.n_helical_channels = n_helical_channels
        self.mode = mode

        # Learnable per-cell helical position. Init: small random spread so all
        # cells start near each other (high coupling), and gradients can push
        # them apart according to task structure.
        self.cell_r = nn.Parameter(torch.randn(n_cells, n_helical_channels) * 0.3)
        self.cell_phi = nn.Parameter(torch.randn(n_cells, n_helical_channels) * 0.3)
        self.cell_z = nn.Parameter(torch.randn(n_cells, n_helical_channels) * 0.3)

        # Per-channel winding frequency for the metric (matches HelicalEmbedding init)
        self.omega = nn.Parameter(
            torch.exp(torch.linspace(-2.0, 2.0, n_helical_channels))
        )

        # Per-cell natural frequency for Kuramoto dynamics (matches KuramotoLayer)
        self.omega_kuramoto = nn.Parameter(torch.zeros(n_cells, 1))

        # Learnable kernel bandwidth (parametrize log-sigma so sigma stays positive)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))

        # Optional residual learned coupling (mix mode)
        if mode == 'mix':
            self.K_learned = nn.Parameter(torch.randn(n_cells, n_cells) * 0.01)
            self.alpha = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5 init
        elif mode != 'replace':
            raise ValueError(f"mode must be 'replace' or 'mix'; got {mode!r}")

    def _helical_distance_squared(self) -> torch.Tensor:
        """Pairwise ds^2 between all 81 cells under the helical metric.

        Returns:
            ds_sq: (n_cells, n_cells) — sum over channels.
        """
        r = self.cell_r            # (N, C)
        phi = self.cell_phi        # (N, C)
        z = self.cell_z            # (N, C)

        # Pairwise differences (N, N, C)
        dr = r.unsqueeze(1) - r.unsqueeze(0)
        dz = z.unsqueeze(1) - z.unsqueeze(0)
        # Wrap phi differences to [-pi, pi]
        dphi = phi.unsqueeze(1) - phi.unsqueeze(0)
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi

        # Deviation from natural winding rate, also wrapped
        deviation = dphi - self.omega * dz
        deviation = (deviation + math.pi) % (2 * math.pi) - math.pi

        # Average radius (matches Module 2's discretization choice)
        r_avg = 0.5 * (r.unsqueeze(1) + r.unsqueeze(0))

        ds_sq_per_channel = dr.pow(2) + (r_avg * deviation).pow(2) + dz.pow(2)
        ds_sq = ds_sq_per_channel.sum(dim=-1)            # (N, N)
        return ds_sq

    def coupling(self) -> torch.Tensor:
        """Return the (n_cells, n_cells) coupling matrix used in dynamics."""
        ds_sq = self._helical_distance_squared()
        sigma_sq = (2.0 * self.log_sigma).exp() + 1e-8     # sigma^2 > 0
        K_helical = (-ds_sq / sigma_sq).exp()

        if self.mode == 'replace':
            K = K_helical
        else:
            # mix mode: blend with learned coupling
            a = torch.sigmoid(self.alpha)
            K = a * K_helical + (1.0 - a) * self.K_learned

        # Zero the diagonal AFTER mixing — applies to both modes uniformly
        eye = torch.eye(self.n_cells, device=K.device, dtype=K.dtype)
        return K * (1.0 - eye)

    def forward(self, theta: torch.Tensor,
                n_steps: int = None,
                return_trajectory: bool = False):
        # theta: (B, n_cells, osc_per_cell)
        n = self.n_steps if n_steps is None else n_steps
        K = self.coupling()
        if return_trajectory:
            trajectory = [theta]
        for _ in range(n):
            diff = theta.unsqueeze(2) - theta.unsqueeze(1)
            coupling = torch.einsum('ij,bijc->bic', K, diff.sin())
            theta = theta + self.eta * (self.omega_kuramoto + coupling)
            if return_trajectory:
                trajectory.append(theta)
        if return_trajectory:
            return theta, torch.stack(trajectory, dim=1)
        return theta

    def compute_energy(self, theta: torch.Tensor) -> torch.Tensor:
        """Same form as KuramotoLayer.compute_energy but uses the derived K."""
        K = self.coupling()
        diff = theta.unsqueeze(2) - theta.unsqueeze(1)
        cos_diff = diff.cos().sum(dim=-1)
        energy = -(K * cos_diff).sum(dim=(-1, -2))
        return energy


# =============================================================================
# Shared readout: phases -> per-cell digit logits
# =============================================================================

class CellReadout(nn.Module):
    """Maps final oscillator phases of each cell to 9 digit logits.

    Features per channel:
        use_amplitude=False : (cos(theta), sin(theta))           — 2 features
        use_amplitude=True  : (cos(theta), sin(theta), |zeta|)   — 3 features

    The amplitude-aware version is for SMM variants that compute zeta as a
    complex tensor; the magnitude carries information that would otherwise
    be discarded by `zeta.angle()`.
    """
    def __init__(self, osc_per_cell: int, hidden_dim: int = 32,
                 n_digits: int = 9, use_amplitude: bool = False):
        super().__init__()
        self.use_amplitude = use_amplitude
        n_features = osc_per_cell * (3 if use_amplitude else 2)
        self.head = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_digits),
        )

    def forward(self, theta: torch.Tensor,
                amplitude: torch.Tensor = None) -> torch.Tensor:
        # theta: (B, n_cells, osc_per_cell)  ->  logits: (B, n_cells, n_digits)
        feats = [theta.cos(), theta.sin()]
        if self.use_amplitude:
            assert amplitude is not None, \
                "amplitude required when use_amplitude=True"
            feats.append(amplitude)
        return self.head(torch.cat(feats, dim=-1))


# =============================================================================
# Path G: shared eval helper (test-time compute + softmax-averaged readout)
# =============================================================================

def kuramoto_evaluate(kuramoto: 'KuramotoLayer',
                      readout: 'CellReadout',
                      phases_initial: torch.Tensor,
                      *,
                      n_steps: int = None,
                      average_window: int = 0,
                      amplitude: torch.Tensor = None) -> torch.Tensor:
    """Run Kuramoto + readout with optional test-time compute extension and
    trajectory averaging.

    Modes:
      n_steps=None,  average_window=0  -- standard forward (training behavior)
      n_steps=K,     average_window=0  -- run K Kuramoto steps at eval time
                                          (test-time compute extension)
      n_steps=None,  average_window=W  -- average readout softmaxes over the
                                          last W trajectory steps
      n_steps=K,     average_window=W  -- both: K steps + softmax-averaging
                                          across the last W of them

    Why softmax averaging rather than energy-min voting:
        AKOrN's published "energy-based voting" picks the lowest-energy
        Kuramoto timestep and reads out from there. That requires the readout
        to have been trained against intermediate trajectory states. Our
        readout was trained against the FINAL post-Kuramoto state only, so
        feeding it earlier phases produces near-random predictions. Softmax
        averaging across the last W steps stays in-distribution (the late
        trajectory looks similar to the trained-on final step) and provides
        an ensembling effect that doesn't require retraining.

    Args:
        kuramoto: KuramotoLayer instance.
        readout: CellReadout instance.
        phases_initial: (B, N, C) starting phases (post-encoder, pre-Kuramoto).
        n_steps: override kuramoto.n_steps. None means use the trained value.
        average_window: number of late trajectory steps to average over.
                        0 disables averaging; only the final step's readout
                        is returned.
        amplitude: optional (B, N, C) amplitude tensor for use_amplitude readouts.

    Returns:
        logits: (B, N, n_digits)
    """
    if average_window <= 0:
        # Single-step readout from the final post-Kuramoto state
        phases_final = kuramoto(phases_initial, n_steps=n_steps)
        if amplitude is not None:
            return readout(phases_final, amplitude)
        return readout(phases_final)

    # Trajectory branch: collect (B, T+1, N, C), softmax-average over last W
    _, full_trajectory = kuramoto(phases_initial, n_steps=n_steps,
                                    return_trajectory=True)
    # full_trajectory[:, 0] is the pre-Kuramoto state — invalid for readout
    trajectory = full_trajectory[:, 1:]                       # (B, T, N, C)
    T = trajectory.shape[1]
    window = min(average_window, T)
    late = trajectory[:, T - window:]                         # (B, W, N, C)

    # Compute readout at each window step, average their softmaxes
    avg_probs = None
    for t in range(window):
        if amplitude is not None:
            logits_t = readout(late[:, t], amplitude)
        else:
            logits_t = readout(late[:, t])
        probs_t = logits_t.softmax(dim=-1)
        avg_probs = probs_t if avg_probs is None else avg_probs + probs_t
    avg_probs = avg_probs / window
    # Return as logits (log of average probabilities) so the existing
    # cross-entropy / argmax code path is unchanged
    return (avg_probs + 1e-12).log()


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

    def forward(self, puzzle: torch.Tensor,
                n_steps_eval: int = None,
                average_window: int = 0) -> torch.Tensor:
        e = self.embed(puzzle)                       # (B, 81, embed_dim)
        h = F.gelu(self.proj_intermediate(e))        # (B, 81, intermediate_dim)
        phases = self.proj_phases(h)                 # (B, 81, osc_per_cell)
        return kuramoto_evaluate(self.kuramoto, self.readout, phases,
                                  n_steps=n_steps_eval, average_window=average_window)


class AKOrNWithSMM(nn.Module):
    """SMM variant: GridEmbedding -> Linear -> HelicalEmbedding -> angle -> Kuramoto -> Readout.

    The input projection produces a 3*osc_per_cell vector per cell, which is
    then split into (r, theta, z) chunks by HelicalEmbedding. The phase of the
    resulting complex zeta is used directly as the Kuramoto initial condition.

    NOTE: this variant discards |zeta| (amplitude) when calling .angle().
    See AKOrNWithSMMAmp for the version that preserves amplitude info.
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

    def forward(self, puzzle: torch.Tensor,
                n_steps_eval: int = None,
                average_window: int = 0) -> torch.Tensor:
        e = self.embed(puzzle)                       # (B, 81, embed_dim)
        h = F.gelu(self.proj_intermediate(e))        # (B, 81, intermediate_dim)
        zeta, _ = self.helical(h)                    # zeta: (B, 81, osc_per_cell) complex
        phases = zeta.angle()                        # extract phase from complex repr
        return kuramoto_evaluate(self.kuramoto, self.readout, phases,
                                  n_steps=n_steps_eval, average_window=average_window)


class AKOrNWithSMMAmp(nn.Module):
    """SMM variant that preserves amplitude information (Path A diagnostic).

    Identical to AKOrNWithSMM except the readout receives BOTH phase
    (via cos/sin) AND amplitude (|zeta|) per channel. This tests the
    hypothesis that the lossy `zeta.angle()` extraction in AKOrNWithSMM
    was a major source of underperformance.

    Architectural choice: amplitude is taken from the *initial* zeta
    (before Kuramoto). Kuramoto only modifies phases; the amplitude
    carries direct information from the input embedding through to
    the readout, bypassing the oscillator dynamics. This is a deliberate
    decision to test whether the encoder's amplitude signal is what the
    baseline's flexible Linear projection was implicitly capturing.

    Param overhead vs AKOrNWithSMM:
        readout input grows from 2*osc_per_cell to 3*osc_per_cell features
        => +osc_per_cell * hidden_dim params (e.g., 4 * 32 = 128 at defaults)
    """
    def __init__(self, embed_dim: int = 24, intermediate_dim: int = 12,
                 osc_per_cell: int = 4, kuramoto_steps: int = 8):
        super().__init__()
        self.osc_per_cell = osc_per_cell
        assert intermediate_dim == 3 * osc_per_cell
        self.embed = GridEmbedding(embed_dim)
        self.proj_intermediate = nn.Linear(embed_dim, intermediate_dim)
        self.helical = HelicalEmbedding(input_dim=intermediate_dim)
        self.kuramoto = KuramotoLayer(n_cells=81, osc_per_cell=osc_per_cell,
                                       n_steps=kuramoto_steps)
        # Amplitude-aware readout: 3 features per channel (cos, sin, |zeta|)
        self.readout = CellReadout(osc_per_cell, use_amplitude=True)

    def forward(self, puzzle: torch.Tensor,
                n_steps_eval: int = None,
                average_window: int = 0) -> torch.Tensor:
        e = self.embed(puzzle)
        h = F.gelu(self.proj_intermediate(e))
        zeta, _ = self.helical(h)                    # complex
        phases = zeta.angle()
        amplitudes = zeta.abs()                      # *** preserved ***
        return kuramoto_evaluate(self.kuramoto, self.readout, phases,
                                  n_steps=n_steps_eval, average_window=average_window,
                                  amplitude=amplitudes)


class AKOrNWithSMMFull(nn.Module):
    """Full-stack SMM variant — Path B.

    Uses 3 of the 4 SMM modules in sequence as the encoder body:
      HelicalEmbedding  ->  the complex helical representation
      [ RotationalHelicalLinear -> PhaseCoherenceGate ] x N_blocks
                        ->  helical 'computation' done IN complex space
      angle(zeta)       ->  collapse to phases for Kuramoto

    The previous SMM variants (AKOrNWithSMM, AKOrNWithSMMAmp) only used
    HelicalEmbedding — a single helical projection at the input. The body
    of the network was identical to baseline. This variant gives the
    structural prior architectural depth: complex-valued processing through
    stacked rotational + interference blocks BEFORE collapsing to phases.

    Default config (n_smm_blocks=2, learnable_ref gate, non-unitary linear)
    matches baseline's effective depth (proj_intermediate + proj_phases is
    also 2 layers) so the comparison is depth-matched, not just param-matched.

    Tunable knobs:
      n_smm_blocks  -- depth of the helical body (default 2)
      gate_mode     -- 'learnable_ref' / 'mean_neighbors' / 'kuramoto_order'
      unitary       -- if True, RotationalHelicalLinear projects to nearest
                       unitary matrix per forward pass (more expensive,
                       preserves |zeta| spectrum; off by default)
    """
    def __init__(self, embed_dim: int = 24, intermediate_dim: int = 12,
                 osc_per_cell: int = 4, kuramoto_steps: int = 8,
                 n_smm_blocks: int = 2, gate_mode: str = 'learnable_ref',
                 unitary: bool = False):
        super().__init__()
        self.osc_per_cell = osc_per_cell
        assert intermediate_dim == 3 * osc_per_cell
        self.n_smm_blocks = n_smm_blocks

        self.embed = GridEmbedding(embed_dim)
        self.proj_intermediate = nn.Linear(embed_dim, intermediate_dim)
        self.helical = HelicalEmbedding(input_dim=intermediate_dim)

        # Stack of helical body blocks
        self.smm_body = nn.ModuleList([
            nn.ModuleDict({
                'linear': RotationalHelicalLinear(
                    in_channels=osc_per_cell,
                    out_channels=osc_per_cell,
                    unitary=unitary,
                ),
                'gate': PhaseCoherenceGate(
                    n_channels=osc_per_cell,
                    context_mode=gate_mode,
                ),
            })
            for _ in range(n_smm_blocks)
        ])

        self.kuramoto = KuramotoLayer(n_cells=81, osc_per_cell=osc_per_cell,
                                       n_steps=kuramoto_steps)
        self.readout = CellReadout(osc_per_cell)   # phase-only readout

    def forward(self, puzzle: torch.Tensor,
                n_steps_eval: int = None,
                average_window: int = 0) -> torch.Tensor:
        e = self.embed(puzzle)
        h = F.gelu(self.proj_intermediate(e))
        zeta, _ = self.helical(h)                  # complex (B, 81, osc_per_cell)

        # Process the helical representation through stacked SMM blocks
        for block in self.smm_body:
            zeta = block['linear'](zeta)           # complex linear mix
            zeta = block['gate'](zeta)             # interference-based gating

        # Collapse to real phases for Kuramoto, then run with optional eval extras
        phases = zeta.angle()
        return kuramoto_evaluate(self.kuramoto, self.readout, phases,
                                  n_steps=n_steps_eval, average_window=average_window)


class AKOrNHelicalCoupled(nn.Module):
    """Path J variant: baseline-style encoder + HelicalCoupledKuramoto.

    Identical to AKOrNBaseline except the standard KuramotoLayer is replaced
    with HelicalCoupledKuramoto, where the cell-to-cell coupling matrix is
    derived from learnable per-cell helical positions via Module 2's
    helical-deviation metric.

    The hypothesis: the helical metric provides a useful inductive bias on
    the *dynamics* (which cells should sync), independent of whether inputs
    are encoded helically. If this works, the metric carries genuine
    structural information; if not, helical geometry doesn't help even
    where it has the most direct opportunity to.
    """
    def __init__(self, embed_dim: int = 24, intermediate_dim: int = 12,
                 osc_per_cell: int = 4, kuramoto_steps: int = 8,
                 helical_mode: str = 'replace',
                 n_helical_channels: int = None):
        super().__init__()
        self.osc_per_cell = osc_per_cell
        self.embed = GridEmbedding(embed_dim)
        self.proj_intermediate = nn.Linear(embed_dim, intermediate_dim)
        self.proj_phases = nn.Linear(intermediate_dim, osc_per_cell)
        self.kuramoto = HelicalCoupledKuramoto(
            n_cells=81, osc_per_cell=osc_per_cell,
            n_steps=kuramoto_steps,
            n_helical_channels=n_helical_channels,
            mode=helical_mode,
        )
        self.readout = CellReadout(osc_per_cell)

    def forward(self, puzzle: torch.Tensor,
                n_steps_eval: int = None,
                average_window: int = 0) -> torch.Tensor:
        e = self.embed(puzzle)
        h = F.gelu(self.proj_intermediate(e))
        phases = self.proj_phases(h)
        return kuramoto_evaluate(self.kuramoto, self.readout, phases,
                                  n_steps=n_steps_eval,
                                  average_window=average_window)


class AKOrNWithSMMHelicalCoupled(nn.Module):
    """Path J's most aggressive variant: SMM encoder + HelicalCoupledKuramoto.

    Combines (a) helical input encoding (Module 1: HelicalEmbedding) with
    (b) helical-distance-modulated Kuramoto coupling (Module 2 wired in).

    If helical structure is genuinely useful for this task, this variant has
    the highest chance of showing it: helical priors operate on BOTH the
    representation and the dynamics. If even this combination ties baseline,
    the conclusion is firm.
    """
    def __init__(self, embed_dim: int = 24, intermediate_dim: int = 12,
                 osc_per_cell: int = 4, kuramoto_steps: int = 8,
                 helical_mode: str = 'replace',
                 n_helical_channels: int = None):
        super().__init__()
        self.osc_per_cell = osc_per_cell
        assert intermediate_dim == 3 * osc_per_cell
        self.embed = GridEmbedding(embed_dim)
        self.proj_intermediate = nn.Linear(embed_dim, intermediate_dim)
        self.helical = HelicalEmbedding(input_dim=intermediate_dim)
        self.kuramoto = HelicalCoupledKuramoto(
            n_cells=81, osc_per_cell=osc_per_cell,
            n_steps=kuramoto_steps,
            n_helical_channels=n_helical_channels,
            mode=helical_mode,
        )
        self.readout = CellReadout(osc_per_cell)

    def forward(self, puzzle: torch.Tensor,
                n_steps_eval: int = None,
                average_window: int = 0) -> torch.Tensor:
        e = self.embed(puzzle)
        h = F.gelu(self.proj_intermediate(e))
        zeta, _ = self.helical(h)
        phases = zeta.angle()
        return kuramoto_evaluate(self.kuramoto, self.readout, phases,
                                  n_steps=n_steps_eval,
                                  average_window=average_window)


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
