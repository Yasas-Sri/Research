import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid.config import MixerConfig
from hybrid.rope import apply_rope


# [TARGET]
def swa_reference(q, k, v, w):
    """Sliding-window causal attention with a full (L, L) band mask (ARCHITECTURE §3.2). Test-only, O(L²) memory.

    q, k, v: (B, L, H, D)
    w:       keys per query including self: t - w < s <= t
    returns: (B, L, H, D)
    """
    L = q.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])  # (B, H, L, L)
    t = torch.arange(L, device=q.device)[:, None]  # (L, 1)
    s = torch.arange(L, device=q.device)[None, :]  # (1, L)
    band = (s <= t) & (t - s < w)  # (L, L)
    p = scores.masked_fill(~band, float("-inf")).softmax(-1)
    return torch.einsum("bhts,bshd->bthd", p, v)


# [TARGET]
def swa_efficient(q, k, v, w):
    """Blocked sliding-window attention (ARCHITECTURE §3.3): each block of w queries sees [prev ‖ current] keys.
    O(L·2w) memory; never builds an (L, L) tensor.

    q, k, v: (B, L, H, D)
    w:       keys per query including self; w >= L is dense causal attention
    returns: (B, L, H, D)
    """
    Bsz, L, H, D = q.shape
    w = min(w, L)  # exact: every causal key already has t - s < L
    nb = -(-L // w)  # ceil(L / w)
    pad = nb * w - L
    q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)).view(Bsz, nb, w, H, D) for x in (q, k, v))  # right-pad L -> (B, nb, w, H, D)
    k2 = torch.cat([F.pad(k, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1], k], dim=2)  # [block b-1 (zeros for b=0) ‖ block b] -> (B, nb, 2w, H, D)
    v2 = torch.cat([F.pad(v, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1], v], dim=2)  # (B, nb, 2w, H, D)
    i = torch.arange(w, device=q.device)[:, None]  # (w, 1) local query index
    j = torch.arange(2 * w, device=q.device)[None, :]  # (1, 2w) local key index
    b = torch.arange(nb, device=q.device)[:, None, None]  # (nb, 1, 1) block index
    # band i < j <= i + w  (0 <= t - s < w), and key position s = (b-1)w + j >= 0 (masks block 0's zero "previous")
    mask = (j > i) & (j <= i + w) & ((b - 1) * w + j >= 0)  # (nb, w, 2w)
    scores = torch.einsum("bnihd,bnjhd->bnhij", q, k2) / math.sqrt(D)  # (B, nb, H, w, 2w)
    p = scores.masked_fill(~mask[:, None], float("-inf")).softmax(-1)  # mask (nb, 1, w, 2w) broadcasts over H
    out = torch.einsum("bnhij,bnjhd->bnihd", p, v2)  # (B, nb, w, H, D)
    return out.reshape(Bsz, nb * w, H, D)[:, :L]  # crop right padding -> (B, L, H, D)


# [TARGET]
class SlidingWindowAttention(nn.Module):
    """SWA branch (ARCHITECTURE §3.1): bias-free projections, Unified RoPE on Q and K only."""

    def __init__(self, cfg: MixerConfig):
        super().__init__()
        d = cfg.d_model
        self.head_dim, self.window = cfg.attn_head_dim, cfg.attn_window
        self.n_heads = d // cfg.attn_head_dim
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        """x: (B, L, d); cos, sin: (L, D_h/2) -> (B, L, d)."""
        Bsz, L, d = x.shape
        heads = (Bsz, L, self.n_heads, self.head_dim)  # (B, L, d) -> (B, L, H_a, D_h)
        q = apply_rope(self.W_q(x).view(heads), cos, sin)
        k = apply_rope(self.W_k(x).view(heads), cos, sin)
        v = self.W_v(x).view(heads)
        o = swa_efficient(q, k, v, self.window)  # (B, L, H_a, D_h)
        return self.W_o(o.reshape(Bsz, L, d))  # (B, L, H_a, D_h) -> (B, L, d)
