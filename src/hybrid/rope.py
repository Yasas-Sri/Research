import torch
import torch.nn as nn


# [TARGET]
def rope_cos_sin(positions, dim, base=10000.0):
    """RoPE tables (ARCHITECTURE §2.3). Angles always in float64; apply_rope casts to the activation dtype.

    positions: (L,) long
    returns:   cos, sin  each (L, dim/2) float64
    """
    inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device) / dim)  # (dim/2,)
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]  # (L, 1) * (1, dim/2) -> (L, dim/2)
    return ang.cos(), ang.sin()


# [TARGET]
def apply_rope(x, cos, sin):
    """Rotate-half RoPE: dim i is paired with i + D/2. One implementation for both branches.

    x:        (B, L, H, D)
    cos, sin: (L, D/2), cast to x.dtype here
    returns:  (B, L, H, D)
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]  # (B, L, H, D/2) each
    c = cos.to(x.dtype)[None, :, None, :]  # (L, D/2) -> (1, L, 1, D/2)
    s = sin.to(x.dtype)[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# [TARGET]
class UnifiedRoPE(nn.Module):
    """One rotation shared by attention Q/K and SSD C/B (ARCHITECTURE §2). No parameters or buffers."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim, self.base = dim, base

    def forward(self, positions):
        """positions: (L,) long -> (cos, sin) each (L, dim/2) float64."""
        return rope_cos_sin(positions, self.dim, self.base)
