import pytest
import torch
import torch.nn.functional as F

from conftest import MaxNumel, assert_causal
from hybrid.config import MixerConfig
from hybrid.rope import apply_rope, rope_cos_sin
from hybrid.ssm import SSDBranch, ssd_efficient, ssd_reference

TOL = dict(rtol=0, atol=1e-9)


def ssd_inputs(L, Bsz=2, H=3, P=4, N=6):
    """Weak decay (|dt·A| ~ 0.1 per step) so inter-chunk state still matters at atol 1e-9."""
    x = torch.randn(Bsz, L, H, P)
    dt = F.softplus(torch.randn(Bsz, L, H) - 2)  # (B, L, H) > 0
    A = -torch.empty(H).uniform_(0.5, 2)  # (H,) < 0
    B, C = torch.randn(Bsz, L, N), torch.randn(Bsz, L, N)
    D = torch.randn(H)
    return x, dt, A, B, C, D


@pytest.mark.parametrize("L", [37, 100])  # not a multiple of 4, 8, 16, 64; L = 100 gives T = 64 two chunks
@pytest.mark.parametrize("T", [1, 4, 8, 16, 64])
def test_ssd_efficient_matches_reference(seed, float64, L, T):
    x, dt, A, B, C, D = ssd_inputs(L)
    torch.testing.assert_close(ssd_efficient(x, dt, A, B, C, D, T), ssd_reference(x, dt, A, B, C, D), **TOL)


def test_ssd_chunk_size_invariance(seed, float64):
    args = ssd_inputs(100)
    torch.testing.assert_close(ssd_efficient(*args, 8), ssd_efficient(*args, 32), **TOL)


@pytest.mark.parametrize("T", [4, 16])
@pytest.mark.parametrize("t0", [1, 9, 30])
def test_ssd_efficient_causal(seed, float64, T, t0):
    x, dt, A, B, C, D = ssd_inputs(37)
    assert_causal(lambda x, dt, B, C: ssd_efficient(x, dt, A, B, C, D, T), (x, dt, B, C), t0)


def test_ssd_efficient_float32_strong_decay_stable(seed):
    """Chunk sums of dt·A reach ~−1600: any cumprod of exp(dt·A) underflows in float32; exp of cumsums does not."""
    L, H = 200, 3
    x, _, _, B, C, D = ssd_inputs(L, H=H)
    dt = F.softplus(torch.randn(2, L, H) + 1)  # (B, L, H), strong
    A = -torch.tensor([1.0, 8.0, 16.0])  # (H,)
    y = ssd_efficient(x, dt, A, B, C, D, 64)
    assert torch.isfinite(y).all()
    ref = ssd_reference(*(t.double() for t in (x, dt, A, B, C, D)))
    torch.testing.assert_close(y.double(), ref, rtol=0, atol=1e-3)  # AGENTS §4.6 float32 stability tolerance


def max_numel(L, chunk):
    args = ssd_inputs(L, Bsz=1, H=1, P=1, N=1)
    with MaxNumel() as rec:
        ssd_efficient(*args, chunk)
    return rec.max


def test_ssd_efficient_no_LxL_allocation(seed):
    L, T = 1024, 16
    assert max_numel(L, L) >= L * L  # negative control: one chunk of length L builds an (L, L) decay matrix
    small, big = max_numel(L, T), max_numel(4 * L, T)
    assert small < L * L
    assert big < 8 * small  # ~4x for 4x L (linear), not ~16x (quadratic)


def small_cfg():
    return MixerConfig(d_model=32, attn_head_dim=8, ssm_d_state=8, ssm_head_dim=16, ssm_chunk=8)


def test_ssd_init_decay_in_unit_interval(seed):
    cfg = MixerConfig()
    m = SSDBranch(cfg)
    A = -torch.exp(m.A_log)
    assert ((A >= -16 * (1 + 1e-6)) & (A <= -1 * (1 - 1e-6))).all()
    dt0 = F.softplus(m.dt_bias)
    assert ((dt0 >= 1e-3 * (1 - 1e-4)) & (dt0 <= 1e-1 * (1 + 1e-4))).all()
    assert torch.equal(m.D, torch.ones_like(m.D))
    # per-token decay as forward computes it; dt is the last H columns of in_proj
    dt = F.softplus(m.in_proj(torch.randn(2, 50, cfg.d_model))[..., -m.A_log.shape[0] :] + m.dt_bias)  # (B, L, H)
    decay = torch.exp(dt * A)
    assert ((decay > 0) & (decay < 1)).all()


@pytest.mark.parametrize("t0", [1, 9, 20])
def test_ssd_module_causal(seed, float64, t0):
    cfg = small_cfg()
    m = SSDBranch(cfg)
    cos, sin = rope_cos_sin(torch.arange(21), cfg.ssm_d_state)
    assert_causal(lambda x: m(x, cos, sin), (torch.randn(2, 21, cfg.d_model),), t0)


def test_ssd_module_shift_invariant(seed, float64):
    cfg = small_cfg()
    m = SSDBranch(cfg)
    x, positions = torch.randn(2, 21, cfg.d_model), torch.arange(21)
    out = m(x, *rope_cos_sin(positions, cfg.ssm_d_state))
    torch.testing.assert_close(m(x, *rope_cos_sin(positions + 1000, cfg.ssm_d_state)), out, **TOL)


def test_ssd_rope_changes_output(seed, float64):
    cfg = small_cfg()
    m = SSDBranch(cfg)
    with torch.no_grad():
        m.D.zero_()  # at init D·x is ~99.6% of the output; RoPE only acts on the C·B term
    x = torch.randn(2, 21, cfg.d_model)
    positions = torch.tensor([5, 0, 9, 1000, 3, 2, 7, 1, 4, 6, 8, 11, 10, 500, 17, 250, 13, 999, 12, 40, 77])  # unsorted
    on = m(x, *rope_cos_sin(positions, cfg.ssm_d_state))
    off = m(x, *rope_cos_sin(torch.zeros(21, dtype=torch.long), cfg.ssm_d_state))  # identity rotation
    assert not torch.allclose(on, off, **TOL)  # different by the same tolerance that defines "equal"


def test_ssd_module_matches_reference_composition(seed, float64):
    cfg = small_cfg()
    m = SSDBranch(cfg)
    Bsz, L, k = 2, 21, cfg.ssm_conv_kernel
    d_in, N, P = cfg.ssm_expand * cfg.d_model, cfg.ssm_d_state, cfg.ssm_head_dim
    H = d_in // P
    x = torch.randn(Bsz, L, cfg.d_model)
    cos, sin = rope_cos_sin(torch.randperm(L) * 7, N)  # unsorted, non-contiguous positions

    z, xBC, dt = (x @ m.in_proj.weight.T).split([d_in, d_in + 2 * N, H], dim=-1)
    # explicit causal depthwise conv: out[t, c] = bias[c] + Σ_i w[c, i] · xBC[t − (k−1) + i, c]
    xpad = F.pad(xBC, (0, 0, k - 1, 0))  # left-pad time: (B, L, C) -> (B, k−1+L, C)
    w = m.conv.weight[:, 0, :]  # (C, 1, k) -> (C, k)
    xBC = F.silu(sum(xpad[:, i : i + L] * w[:, i] for i in range(k)) + m.conv.bias)
    xs, B, C = xBC.split([d_in, N, N], dim=-1)
    B = apply_rope(B[:, :, None], cos, sin)[:, :, 0]  # RoPE on B, C only, after conv
    C = apply_rope(C[:, :, None], cos, sin)[:, :, 0]
    y = ssd_reference(xs.reshape(Bsz, L, H, P), F.softplus(dt + m.dt_bias), -torch.exp(m.A_log), B, C, m.D)
    expected = m.norm(y.reshape(Bsz, L, d_in) * F.silu(z)) @ m.out_proj.weight.T  # gate, then norm
    torch.testing.assert_close(m(x, cos, sin), expected, **TOL)


def test_ssd_module_shape_finite_and_gradients(seed):
    cfg = MixerConfig()
    m = SSDBranch(cfg)
    L = 100
    y = m(torch.randn(2, L, cfg.d_model), *rope_cos_sin(torch.arange(L), cfg.ssm_d_state))
    assert y.shape == (2, L, cfg.d_model)
    assert torch.isfinite(y).all()
    (y * torch.randn_like(y)).sum().backward()
    for name in ["A_log", "dt_bias", "D", "conv.weight", "conv.bias", "in_proj.weight", "out_proj.weight", "norm.weight"]:
        g = m.get_parameter(name).grad
        assert g is not None and (g != 0).all(), name
