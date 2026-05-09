"""Unit tests for SMM modules — run with `pytest tests/`."""

import math
import pytest
import torch

from smm import (
    HelicalEmbedding,
    HelicalDeviationMetric,
    RotationalHelicalLinear,
    PhaseCoherenceGate,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def small_input():
    torch.manual_seed(0)
    B, S, D = 2, 5, 12
    return torch.randn(B, S, D, requires_grad=True)


# -----------------------------------------------------------------------------
# HelicalEmbedding
# -----------------------------------------------------------------------------
class TestHelicalEmbedding:
    def test_shape(self, small_input):
        embed = HelicalEmbedding(input_dim=12)
        zeta, z = embed(small_input)
        assert zeta.shape == (2, 5, 4)
        assert z.shape == (2, 5, 4)
        assert zeta.is_complex()
        assert not z.is_complex()

    def test_input_dim_must_be_divisible_by_3(self):
        with pytest.raises(AssertionError):
            HelicalEmbedding(input_dim=10)

    def test_omega_is_learnable(self, small_input):
        embed = HelicalEmbedding(input_dim=12)
        zeta, _ = embed(small_input)
        loss = zeta.abs().mean()
        loss.backward()
        assert embed.omega.grad is not None
        assert embed.omega.grad.abs().sum() > 0

    def test_force_positive_r(self, small_input):
        embed = HelicalEmbedding(input_dim=12, force_positive_r=True)
        zeta, _ = embed(small_input)
        # softplus is always > 0, so |zeta| should be > 0 everywhere
        assert (zeta.abs() > 0).all()


# -----------------------------------------------------------------------------
# HelicalDeviationMetric
# -----------------------------------------------------------------------------
class TestHelicalDeviationMetric:
    def test_self_distance_is_zero(self, small_input):
        embed = HelicalEmbedding(input_dim=12)
        metric = HelicalDeviationMetric(omega=embed.omega)
        zeta, z = embed(small_input)
        ds = metric(zeta, z)
        diag = ds.diagonal(dim1=1, dim2=2)
        assert diag.abs().max() < 1e-4

    def test_symmetric(self, small_input):
        embed = HelicalEmbedding(input_dim=12)
        metric = HelicalDeviationMetric(omega=embed.omega)
        zeta, z = embed(small_input)
        ds = metric(zeta, z)
        # ds[b, i, j, c] should equal ds[b, j, i, c]
        ds_T = ds.transpose(1, 2)
        assert torch.allclose(ds, ds_T, atol=1e-5)

    def test_non_negative(self, small_input):
        embed = HelicalEmbedding(input_dim=12)
        metric = HelicalDeviationMetric(omega=embed.omega)
        zeta, z = embed(small_input)
        ds = metric(zeta, z)
        assert (ds >= 0).all()

    def test_shared_omega(self, small_input):
        """Embedding and metric must share the same omega tensor."""
        embed = HelicalEmbedding(input_dim=12)
        metric = HelicalDeviationMetric(omega=embed.omega)
        assert metric.omega is embed.omega   # identity, not just equality


# -----------------------------------------------------------------------------
# RotationalHelicalLinear
# -----------------------------------------------------------------------------
class TestRotationalHelicalLinear:
    def test_forward_shape(self):
        layer = RotationalHelicalLinear(in_channels=4, out_channels=8)
        zeta = torch.randn(2, 5, 4, dtype=torch.complex64)
        out = layer(zeta)
        assert out.shape == (2, 5, 8)
        assert out.is_complex()

    def test_gradient_flow(self):
        layer = RotationalHelicalLinear(in_channels=4, out_channels=4)
        zeta = torch.randn(2, 5, 4, dtype=torch.complex64, requires_grad=True)
        out = layer(zeta)
        loss = out.abs().mean()
        loss.backward()
        assert layer.W.grad is not None
        assert layer.W.grad.abs().sum() > 0

    def test_unitary_preserves_norm_approximately(self):
        """Unitary W should approximately preserve |zeta| spectrum."""
        torch.manual_seed(42)
        layer = RotationalHelicalLinear(in_channels=8, out_channels=8, unitary=True)
        zeta = torch.randn(2, 5, 8, dtype=torch.complex64)
        out = layer(zeta)
        # Sum of |zeta|^2 should be approximately preserved (modulo bias)
        layer.b.data.zero_()
        out_no_bias = layer(zeta)
        in_norm  = zeta.abs().pow(2).sum(dim=-1)
        out_norm = out_no_bias.abs().pow(2).sum(dim=-1)
        assert torch.allclose(in_norm, out_norm, atol=1e-4)


# -----------------------------------------------------------------------------
# PhaseCoherenceGate
# -----------------------------------------------------------------------------
class TestPhaseCoherenceGate:
    @pytest.mark.parametrize(
        "mode", ["learnable_ref", "mean_neighbors", "kuramoto_order"]
    )
    def test_all_modes_run(self, mode):
        gate = PhaseCoherenceGate(n_channels=4, context_mode=mode)
        zeta = torch.randn(2, 5, 4, dtype=torch.complex64)
        out = gate(zeta)
        assert out.shape == zeta.shape
        assert out.is_complex()

    def test_constructive_passes_destructive_blocks(self):
        """When ζ and c are in-phase, gate ≈ 1; out-of-phase, gate ≈ 0."""
        gate = PhaseCoherenceGate(n_channels=1, context_mode='learnable_ref')

        # Pin reference c = 1+0j
        with torch.no_grad():
            gate.c_real.fill_(1.0)
            gate.c_imag.fill_(0.0)

        # In-phase input: zeta = 2+0j, expect g ≈ 1
        zeta_in = torch.tensor([[[2.0 + 0.0j]]], dtype=torch.complex64)
        out_in = gate(zeta_in)
        assert torch.allclose(out_in.abs(), torch.tensor([[[2.0]]]), atol=1e-3)

        # Out-of-phase: zeta = -1+0j, |zeta+c| = 0, expect g ≈ 0
        zeta_out = torch.tensor([[[-1.0 + 0.0j]]], dtype=torch.complex64)
        out_out = gate(zeta_out)
        assert out_out.abs().max() < 1e-3

    def test_invalid_mode_raises(self):
        gate = PhaseCoherenceGate(n_channels=4, context_mode='invalid_mode')
        zeta = torch.randn(2, 5, 4, dtype=torch.complex64)
        with pytest.raises(ValueError):
            gate(zeta)


# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------
class TestPipeline:
    def test_full_forward_backward(self, small_input):
        embed = HelicalEmbedding(input_dim=12)
        rh = RotationalHelicalLinear(in_channels=4, out_channels=4)
        gate = PhaseCoherenceGate(n_channels=4, context_mode='learnable_ref')
        metric = HelicalDeviationMetric(omega=embed.omega)

        zeta, z = embed(small_input)
        zeta = rh(zeta)
        zeta = gate(zeta)
        ds = metric(zeta, z)

        loss = ds.mean()
        loss.backward()

        assert small_input.grad is not None
        assert embed.omega.grad is not None
        assert rh.W.grad is not None
        assert gate.c_real.grad is not None
