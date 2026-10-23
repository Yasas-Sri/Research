import pytest
import torch
from torch.utils._pytree import tree_leaves
from torch.utils._python_dispatch import TorchDispatchMode


@pytest.fixture
def seed():
    torch.manual_seed(0)
    return 0


@pytest.fixture
def float64():
    """Opt-in: make float64 the default dtype for the test, restore afterwards."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield torch.float64
    torch.set_default_dtype(old)


def assert_causal(fn, inputs, t0):
    """Check outputs at t < t0 are unchanged when inputs at t >= t0 are perturbed.

    fn:     callable on *inputs, returning a tensor with time on dim 1.
    inputs: tuple of data tensors, each (B, L, ...) with time on dim 1. Bind non-data
            arguments (cos, sin, positions) with a closure; they are never perturbed.
    Float inputs get randn noise; integer inputs x become (x + 1) % (x.max() + 1),
    so every perturbed value changes and stays in range. Clones are perturbed,
    never the caller's tensors. Comparison is exact (torch.equal).
    """
    perturbed = []
    for x in inputs:
        x = x.clone()
        tail = x[:, t0:]
        if x.is_floating_point():
            tail += torch.randn_like(tail)
        else:
            tail.copy_((tail + 1) % (x.max() + 1))
        perturbed.append(x)
    out = fn(*inputs)
    out_p = fn(*perturbed)
    assert torch.equal(out[:, :t0], out_p[:, :t0]), f"output before t0={t0} changed"


class MaxNumel(TorchDispatchMode):
    """Records the largest tensor produced by any aten op (including einsum/softmax intermediates)."""

    def __init__(self):
        super().__init__()
        self.max = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.max = max([self.max] + [t.numel() for t in tree_leaves(out) if isinstance(t, torch.Tensor)])
        return out
