import pytest
import torch

from conftest import assert_causal
from hybrid.config import MixerConfig


def test_default_config_is_valid():
    MixerConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(attn_head_dim=64, ssm_d_state=32),  # Unified RoPE needs D_h == N (A2)
        dict(d_model=250),  # d_model % attn_head_dim != 0
        dict(ssm_head_dim=48),  # (ssm_expand * d_model) % ssm_head_dim != 0
        dict(d_model=255, attn_head_dim=85, ssm_d_state=85, ssm_head_dim=85),  # odd RoPE dim
        dict(engram_orders=(1, 2)),  # orders must be >= 2
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        MixerConfig(**kwargs)


def test_assert_causal_accepts_causal_fn(seed):
    x = torch.randn(2, 10, 3)
    ids = torch.randint(0, 5, (2, 10))
    before = (x.clone(), ids.clone())
    assert_causal(lambda x, ids: x.cumsum(1) + ids.cumsum(1)[..., None], (x, ids), t0=4)
    assert torch.equal(x, before[0]) and torch.equal(ids, before[1])  # caller's tensors untouched


@pytest.mark.parametrize("t0", [1, 4, 9])
def test_assert_causal_rejects_acausal_fn(seed, t0):
    x = torch.randn(2, 10, 3)
    with pytest.raises(AssertionError):
        assert_causal(lambda x: x.flip(1).cumsum(1).flip(1), (x,), t0)
