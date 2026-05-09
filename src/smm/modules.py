"""
SMM (Spiral Manifold Mapping) Modules — Sandbox Implementation
================================================================
Four drop-in modules for testing whether enforcing helical / oscillatory
structure in latent space yields parameter compression on reasoning tasks
(Sudoku, ARC-AGI-2, etc.) when injected into an AKOrN-style backbone.

Modules:
  1. HelicalEmbedding        — maps R^n -> n/3 complex helical channels
  2. HelicalDeviationMetric  — pairwise distance under
                               ds^2 = dr^2 + r^2 (dphi - omega dz)^2 + dz^2
  3. RotationalHelicalLinear — complex-valued linear layer (optional unitary)
  4. PhaseCoherenceGate      — interference-based replacement for ReLU

Usage:
    python smm_modules.py        # runs end-to-end smoke test

Requirements: PyTorch >= 2.0 (for full complex autograd support)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Module 1: HelicalEmbedding
# =============================================================================
class HelicalEmbedding(nn.Module):
    """
    Maps an input vector v in R^n to n/3 complex helical channels:
        zeta_i = r_i * exp( j * (theta_i + omega_i * z_i) )

    Convention:
      r     -- amplitude of the channel  (which voice)
      theta -- content phase             (which note)
      z     -- axial / sequence position (where in the symphony)
      omega -- per-channel learnable winding frequency (helix pitch)

    Returns (zeta, z) so the deviation metric can re-use the same z.
    """
    def __init__(self, input_dim, omega_init=None, force_positive_r=True):
        super().__init__()
        assert input_dim % 3 == 0, "input_dim must be divisible by 3 (r, theta, z)."
        self.n_channels = input_dim // 3
        self.force_positive_r = force_positive_r

        # Symmetry-breaking init: spread omegas geometrically across channels
        # (like sinusoidal positional encoding frequencies)
        if omega_init is None:
            omegas = torch.exp(torch.linspace(-2.0, 2.0, self.n_channels))
        else:
            omegas = torch.full((self.n_channels,), float(omega_init))
        self.omega = nn.Parameter(omegas)

    def forward(self, v):
        # v: (B, S, input_dim)
        r, theta, z = torch.chunk(v, 3, dim=-1)

        # Keep r >= 0 so the (r, phi) <-> (-r, phi+pi) ambiguity doesn't
        # waste capacity. softplus is smoother than abs.
        if self.force_positive_r:
            r = F.softplus(r)

        phase = theta + self.omega * z
        zeta = torch.complex(r * torch.cos(phase), r * torch.sin(phase))
        return zeta, z


# =============================================================================
# Module 2: HelicalDeviationMetric
# =============================================================================
class HelicalDeviationMetric(nn.Module):
    """
    Pairwise distance under the helical-deviation metric:

        ds^2 = dr^2 + r^2 (dphi - omega dz)^2 + dz^2

    The middle term penalizes deviation from the natural helical winding
    phi = omega * z + const. Two points sitting on the same helix are
    'close' regardless of how far apart they are along the spiral; two
    points off-helix pay the angular penalty.

    Pass the SAME omega tensor (nn.Parameter) used in HelicalEmbedding to
    keep metric and embedding aligned.
    """
    def __init__(self, omega):
        super().__init__()
        self.omega = omega   # tensor (n_channels,), shared with embedding

    @staticmethod
    def _wrap_pi(x):
        """Wrap angle to [-pi, pi]."""
        return (x + math.pi) % (2 * math.pi) - math.pi

    def forward(self, zeta_a, z_a, zeta_b=None, z_b=None):
        """
        Args:
          zeta_a: (B, Sa, C) complex
          z_a:    (B, Sa, C) real
          zeta_b: (B, Sb, C) complex   (defaults to zeta_a)
          z_b:    (B, Sb, C) real      (defaults to z_a)

        Returns:
          ds: (B, Sa, Sb, C) — per-channel helical distance
        """
        if zeta_b is None:
            zeta_b, z_b = zeta_a, z_a

        r_a, phi_a = zeta_a.abs(), zeta_a.angle()
        r_b, phi_b = zeta_b.abs(), zeta_b.angle()

        # Pairwise differences via broadcasting -> (B, Sa, Sb, C)
        dr   = r_a.unsqueeze(2)   - r_b.unsqueeze(1)
        dz   = z_a.unsqueeze(2)   - z_b.unsqueeze(1)
        dphi = self._wrap_pi(phi_a.unsqueeze(2) - phi_b.unsqueeze(1))

        # How much of dphi is NOT explained by the natural winding rate?
        deviation = self._wrap_pi(dphi - self.omega * dz)

        # Average radius for the angular metric coefficient
        r_avg = 0.5 * (r_a.unsqueeze(2) + r_b.unsqueeze(1))

        ds_sq = dr.pow(2) + (r_avg * deviation).pow(2) + dz.pow(2)
        return ds_sq.clamp_min(1e-12).sqrt()


# =============================================================================
# Module 3: RotationalHelicalLinear  (complex linear; optional unitary)
# =============================================================================
class RotationalHelicalLinear(nn.Module):
    """
    Complex-valued linear layer. Replaces (Wx + b) with (W zeta + b) where
    W in C^{out x in}. This is the 'Rotational Helical Tensor' operator:
      - arg(W)  -> per-channel phase shifts
      - |W|     -> per-channel amplitude modulation
      - off-diagonal entries -> cross-channel mixing

    If unitary=True, W is projected to its nearest unitary matrix via SVD.
    The unitary version is the strict 'pure phase shifts and frequency
    modulators' interpretation: it preserves |zeta| spectrum across layers
    and prevents helical structure from collapsing. It's also more
    expensive — start without it.
    """
    def __init__(self, in_channels, out_channels, unitary=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unitary = unitary
        self.use_bias = bias

        scale = 1.0 / math.sqrt(in_channels)
        W_real = torch.randn(out_channels, in_channels) * scale
        W_imag = torch.randn(out_channels, in_channels) * scale
        self.W = nn.Parameter(torch.complex(W_real, W_imag))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))
        else:
            self.register_parameter('b', None)

    def _effective_W(self):
        if not self.unitary:
            return self.W
        # Nearest unitary via thin SVD: W = U S V^H -> return U V^H
        U, _, Vh = torch.linalg.svd(self.W, full_matrices=False)
        return U @ Vh

    def forward(self, zeta):
        # zeta: (B, S, in_channels) complex
        W = self._effective_W()
        out = zeta @ W.T
        if self.b is not None:
            out = out + self.b
        return out


# =============================================================================
# Module 4: PhaseCoherenceGate  (replaces ReLU)
# =============================================================================
class PhaseCoherenceGate(nn.Module):
    """
    Phase-aware nonlinearity. Computes an interference ratio between zeta
    and a context signal c, then gates by that ratio:

        g = |zeta + c| / (|zeta| + |c| + eps)   in [0, 1]
        out = g * zeta

    g = 1 -> perfectly constructive (in-phase)  -> pass through
    g = 0 -> perfectly destructive (180 deg)    -> block

    Phase is preserved; amplitude is modulated. Differentiable everywhere.

    context_mode:
      'learnable_ref'  -- c is a learnable per-channel complex reference
      'mean_neighbors' -- c is the sequence-mean of zeta (cheap binding)
      'kuramoto_order' -- c uses unit-magnitude phase mean (synchrony)
                          — closest to the AKOrN dynamics
    """
    def __init__(self, n_channels, context_mode='learnable_ref'):
        super().__init__()
        self.n_channels = n_channels
        self.context_mode = context_mode

        if context_mode == 'learnable_ref':
            self.c_real = nn.Parameter(torch.randn(n_channels) * 0.1)
            self.c_imag = nn.Parameter(torch.randn(n_channels) * 0.1)

    def _context(self, zeta):
        if self.context_mode == 'learnable_ref':
            c = torch.complex(self.c_real, self.c_imag)
            return c.view(1, 1, -1).expand_as(zeta)

        if self.context_mode == 'mean_neighbors':
            return zeta.mean(dim=1, keepdim=True).expand_as(zeta)

        if self.context_mode == 'kuramoto_order':
            # Unit-magnitude mean: synchrony measure rather than amplitude mean
            phi = zeta.angle()
            unit = torch.complex(phi.cos(), phi.sin())
            return unit.mean(dim=1, keepdim=True).expand_as(zeta)

        raise ValueError(f"Unknown context_mode: {self.context_mode}")

    def forward(self, zeta):
        # zeta: (B, S, n_channels) complex
        eps = 1e-8
        c = self._context(zeta)
        g = (zeta + c).abs() / (zeta.abs() + c.abs() + eps)   # real, (B, S, C)
        return zeta * g                                        # complex * real broadcast


# =============================================================================
# Smoke test: end-to-end forward + backward pass
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    B, S, D = 2, 5, 12      # batch=2, seq=5, input_dim=12 -> n_channels=4
    v = torch.randn(B, S, D, requires_grad=True)

    print(f"=== SMM smoke test ===")
    print(f"Input v        : shape={tuple(v.shape)}, dtype={v.dtype}")

    # 1. Embed
    embed = HelicalEmbedding(D)
    zeta, z = embed(v)
    print(f"After embed    : zeta {tuple(zeta.shape)} {zeta.dtype}, "
          f"z {tuple(z.shape)} {z.dtype}")

    # 2. Rotational helical linear (4 -> 4)
    rh_linear = RotationalHelicalLinear(in_channels=4, out_channels=4, unitary=False)
    zeta = rh_linear(zeta)
    print(f"After RH linear: {tuple(zeta.shape)} {zeta.dtype}")

    # 3. Phase-coherence gate
    gate = PhaseCoherenceGate(n_channels=4, context_mode='mean_neighbors')
    zeta = gate(zeta)
    print(f"After gate     : {tuple(zeta.shape)} {zeta.dtype}")

    # 4. Pairwise helical distance using shared omega
    metric = HelicalDeviationMetric(omega=embed.omega)
    ds = metric(zeta, z)
    print(f"Distance ds    : {tuple(ds.shape)} {ds.dtype}")

    # Self-distance on the diagonal should be ~0 (each token vs itself)
    diag = ds.diagonal(dim1=1, dim2=2).abs().max().item()
    print(f"Diagonal max   : {diag:.2e}  (should be ~0)")

    # 5. Backward pass — proves the whole pipeline is differentiable
    loss = ds.mean()
    loss.backward()
    print()
    print(f"=== Gradients ===")
    print(f"  d/dv          : {v.grad.abs().mean().item():.4e}")
    print(f"  d/d omega     : {embed.omega.grad.abs().mean().item():.4e}")
    print(f"  d/d W (real)  : {rh_linear.W.grad.real.abs().mean().item():.4e}")
    print(f"  d/d W (imag)  : {rh_linear.W.grad.imag.abs().mean().item():.4e}")

    # Parameter accounting (complex params count as 2 reals)
    params = [embed.omega, rh_linear.W, rh_linear.b]
    n_real_eq = sum(p.numel() * (2 if p.is_complex() else 1) for p in params)
    print()
    print(f"=== Parameter count ===")
    print(f"  Real-equivalent: {n_real_eq}")

    print()
    print("[OK] Pipeline runs forward + backward end-to-end.")
