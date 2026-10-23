import pytest
import torch
import torch.nn.functional as F

from conftest import MaxNumel, assert_causal
from hybrid.attention import SlidingWindowAttention, swa_efficient, swa_reference
from hybrid.config import MixerConfig
from hybrid.rope import apply_rope, rope_cos_sin

B, H, D = 2, 3, 8
TOL = dict(rtol=0, atol=1e-10)


def qkv(L):
    return tuple(torch.randn(B, L, H, D) for _ in range(3))


def resolve(w, L):
    return {"L": L, "2L": 2 * L, "L+1": L + 1}.get(w, w)


@pytest.mark.parametrize("L", [13, 30])  # not a multiple of 4 or 8
@pytest.mark.parametrize("w", [1, 4, 8, "L", "2L"])
def test_swa_efficient_matches_reference(seed, float64, L, w):
    q, k, v = qkv(L)
    w = resolve(w, L)
    torch.testing.assert_close(swa_efficient(q, k, v, w), swa_reference(q, k, v, w), **TOL)


@pytest.mark.parametrize("w", ["L", "L+1", "2L"])
def test_swa_dense_baseline_matches_reference(seed, float64, w):
    L = 13
    q, k, v = qkv(L)
    w = resolve(w, L)
    out = swa_efficient(q, k, v, w)
    torch.testing.assert_close(out, swa_reference(q, k, v, w), **TOL)
    # independent dense causal oracle (test-only)
    dense = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)), is_causal=True)  # (B, H, L, D)
    torch.testing.assert_close(out, dense.transpose(1, 2), **TOL)


@pytest.mark.parametrize("w", [1, 4, 8, 13])
@pytest.mark.parametrize("t0", [1, 6, 12])
def test_swa_efficient_causal(seed, float64, w, t0):
    assert_causal(lambda q, k, v: swa_efficient(q, k, v, w), qkv(13), t0)


@pytest.mark.parametrize("w", [1, 4, 8, 13])
def test_swa_window_gradient_support(seed, float64, w):
    L = 13
    q, k, v = qkv(L)
    v.requires_grad_(True)
    out = swa_efficient(q, k, v, w)
    for t in range(L):
        (g,) = torch.autograd.grad(out[:, t].sum(), v, retain_graph=True)  # (B, L, H, D)
        support = g.abs().sum((0, 2, 3)).nonzero().flatten().tolist()
        assert support == list(range(max(0, t - w + 1), t + 1)), f"t={t}"


def max_numel(fn, L, w):
    q, k, v = (torch.randn(1, L, 1, 2) for _ in range(3))
    with MaxNumel() as rec:
        fn(q, k, v, w)
    return rec.max


def test_swa_efficient_no_LxL_allocation(seed):
    L, w = 1024, 16
    assert max_numel(swa_reference, L, w) >= L * L  # negative control: the detector sees a dense path
    small, big = max_numel(swa_efficient, L, w), max_numel(swa_efficient, 4 * L, w)
    assert small < L * L
    assert big < 8 * small  # ~4x for 4x L (linear), not ~16x (quadratic)


def small_cfg():
    return MixerConfig(d_model=32, attn_head_dim=8, ssm_d_state=8, ssm_head_dim=8, attn_window=4)


def test_swa_module_shift_invariant(seed, float64):
    cfg = small_cfg()
    m = SlidingWindowAttention(cfg)
    x, positions = torch.randn(2, 13, cfg.d_model), torch.arange(13)
    out = m(x, *rope_cos_sin(positions, cfg.attn_head_dim, cfg.rope_base))
    torch.testing.assert_close(m(x, *rope_cos_sin(positions + 1000, cfg.attn_head_dim, cfg.rope_base)), out, **TOL)


@pytest.mark.parametrize("t0", [1, 6, 12])
def test_swa_module_causal(seed, float64, t0):
    cfg = small_cfg()
    m = SlidingWindowAttention(cfg)
    cos, sin = rope_cos_sin(torch.arange(13), cfg.attn_head_dim)
    assert_causal(lambda x: m(x, cos, sin), (torch.randn(2, 13, cfg.d_model),), t0)


def test_swa_module_matches_reference_composition(seed, float64):
    cfg = small_cfg()
    m = SlidingWindowAttention(cfg)
    L, n_heads = 13, cfg.d_model // cfg.attn_head_dim
    x = torch.randn(2, L, cfg.d_model)
    cos, sin = rope_cos_sin(torch.tensor([5, 0, 9, 1000, 3, 2, 7, 1, 4, 6, 8, 11, 10]), cfg.attn_head_dim)
    heads = lambda t: t.view(2, L, n_heads, cfg.attn_head_dim)  # (B, L, d) -> (B, L, H, D_h)
    q = apply_rope(heads(x @ m.W_q.weight.T), cos, sin)
    k = apply_rope(heads(x @ m.W_k.weight.T), cos, sin)
    v = heads(x @ m.W_v.weight.T)
    expected = swa_reference(q, k, v, cfg.attn_window).reshape(2, L, cfg.d_model) @ m.W_o.weight.T
    torch.testing.assert_close(m(x, cos, sin), expected, **TOL)


def test_swa_module_shape_and_finite(seed):
    cfg = MixerConfig()
    L = 200  # > attn_window = 128, not a multiple of it
    y = SlidingWindowAttention(cfg)(torch.randn(2, L, cfg.d_model), *rope_cos_sin(torch.arange(L), cfg.attn_head_dim))
    assert y.shape == (2, L, cfg.d_model)
    assert torch.isfinite(y).all()
