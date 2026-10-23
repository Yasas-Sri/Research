import math

import pytest
import torch

from hybrid.attention import SlidingWindowAttention
from hybrid.config import MixerConfig
from hybrid.rope import UnifiedRoPE, apply_rope, rope_cos_sin
from hybrid.ssm import SSDBranch

D, BASE = 8, 10000.0


def test_rope_preserves_norm(seed, float64):
    positions = torch.tensor([0, 1, 5, 100, 1000, 12345])
    x = torch.randn(2, len(positions), 3, D)
    cos, sin = rope_cos_sin(positions, D)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), rtol=0, atol=1e-12)


def scores(q, k, positions):
    """<R_m q_m, R_n k_n> for all (m, n). q, k: (1, L, 1, D) -> (L, L)."""
    cos, sin = rope_cos_sin(positions, D)
    return torch.einsum("md,nd->mn", apply_rope(q, cos, sin)[0, :, 0], apply_rope(k, cos, sin)[0, :, 0])


@pytest.mark.parametrize("offset", [1, 1000])
def test_rope_relative_property(seed, float64, offset):
    positions = torch.tensor([0, 3, 4, 17, 50])
    q, k = torch.randn(1, len(positions), 1, D), torch.randn(1, len(positions), 1, D)
    s = scores(q, k, positions)
    torch.testing.assert_close(scores(q, k, positions + offset), s, rtol=0, atol=1e-10)
    # <R_m q, R_n k> == <q, R_{n-m} k>
    for m, n in [(0, 3), (4, 0), (17, 50)]:
        i, j = positions.tolist().index(m), positions.tolist().index(n)
        cos, sin = rope_cos_sin(torch.tensor([n - m]), D)
        k_rel = apply_rope(k[:, j : j + 1], cos, sin)  # (1, 1, 1, D)
        torch.testing.assert_close(s[i, j], q[0, i, 0] @ k_rel[0, 0, 0], rtol=0, atol=1e-10)


def test_rope_zero_position_is_identity(seed, float64):
    x = torch.randn(2, 5, 3, D)
    cos, sin = rope_cos_sin(torch.zeros(5, dtype=torch.long), D)
    assert torch.equal(apply_rope(x, cos, sin), x)


def test_rope_matches_explicit_rotation_matrix(seed, float64):
    positions = torch.tensor([0, 7, 3, 1000, 2])  # unsorted, non-contiguous: catches an internal arange
    cos, sin = UnifiedRoPE(D, BASE)(positions)
    assert cos.shape == sin.shape == (len(positions), D // 2)

    # §2.1 block rotation in rotate-half layout: pair (i, i + D/2), theta_i = base^(-2i/D)
    h = D // 2
    R = torch.zeros(len(positions), D, D)
    for l, m in enumerate(positions.tolist()):
        for i in range(h):
            a = m * BASE ** (-2 * i / D)
            R[l, i, i], R[l, i, i + h] = math.cos(a), -math.sin(a)
            R[l, i + h, i], R[l, i + h, i + h] = math.sin(a), math.cos(a)

    x = torch.randn(2, len(positions), 3, D)
    expected = torch.einsum("lde,blhe->blhd", R, x)
    torch.testing.assert_close(apply_rope(x, cos, sin), expected, rtol=0, atol=1e-12)


def test_unified_rope_shared_by_both_branches(seed, float64):
    """One UnifiedRoPE call feeds both branches (A2: D_h == N); both stay shift-invariant under it."""
    cfg = MixerConfig(d_model=32, attn_head_dim=8, ssm_d_state=8, ssm_head_dim=16, attn_window=4, ssm_chunk=8)
    rope = UnifiedRoPE(cfg.attn_head_dim, cfg.rope_base)
    swa, ssd = SlidingWindowAttention(cfg), SSDBranch(cfg)
    x, positions = torch.randn(2, 21, cfg.d_model), torch.arange(21)
    cos, sin = rope(positions)
    cos_s, sin_s = rope(positions + 1000)
    torch.testing.assert_close(swa(x, cos_s, sin_s), swa(x, cos, sin), rtol=0, atol=1e-10)
    torch.testing.assert_close(ssd(x, cos_s, sin_s), ssd(x, cos, sin), rtol=0, atol=1e-9)
