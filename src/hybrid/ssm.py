import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid.config import MixerConfig
from hybrid.rope import apply_rope


# [TARGET]
def segsum(a):
    """Segment sums for SSD decays (ARCHITECTURE §4.3), as cumsum differences.

    a:       (..., T) log-decays
    returns: (..., T, T) with out[..., i, j] = Σ_{k=j+1..i} a_k for j <= i, -inf above the diagonal
    """
    T = a.shape[-1]
    a_cs = a.cumsum(-1)  # (..., T)
    out = a_cs[..., :, None] - a_cs[..., None, :]  # (..., T, 1) - (..., 1, T) -> (..., T, T)
    causal = torch.ones(T, T, dtype=torch.bool, device=a.device).tril()  # (T, T)
    return out.masked_fill(~causal, float("-inf"))


# [TARGET]
def ssd_reference(x, dt, A, B, C, D):
    """Literal SSD recurrence (ARCHITECTURE §4.2), sequential over t. Test-only.

    h_t = exp(dt_t·A)·h_{t-1} + dt_t·x_t b_tᵀ,   y_t = h_t c_t + D·x_t

    x:  (B, L, H, P);  dt: (B, L, H) > 0;  A: (H,) < 0;  B, C: (B, L, N);  D: (H,)
    returns: (B, L, H, P)
    """
    Bsz, L, H, P = x.shape
    h = x.new_zeros(Bsz, H, P, B.shape[-1])  # (B, H, P, N) state
    ys = []
    for t in range(L):
        decay = torch.exp(dt[:, t] * A)  # (B, H)
        inp = (dt[:, t, :, None] * x[:, t])[..., None] * B[:, t, None, None, :]  # (B, H, P, 1) * (B, 1, 1, N) -> (B, H, P, N)
        h = decay[:, :, None, None] * h + inp
        ys.append(torch.einsum("bhpn,bn->bhp", h, C[:, t]) + D[:, None] * x[:, t])  # (B, H, P)
    return torch.stack(ys, dim=1)  # (B, L, H, P)


# [TARGET]
def ssd_efficient(x, dt, A, B, C, D, chunk):
    """Chunked SSD (ARCHITECTURE §4.3): quadratic work only inside T x T chunks, sequential scan over L/T chunks.

    x:  (B, L, H, P);  dt: (B, L, H) > 0;  A: (H,) < 0;  B, C: (B, L, N);  D: (H,);  chunk: T
    returns: (B, L, H, P)
    """
    Bsz, L, H, P = x.shape
    N, T = B.shape[-1], chunk
    nc = -(-L // T)  # ceil(L / T)
    pad = nc * T - L
    # right-pad with a = 0, x̃ = 0, B = C = 0, then split into chunks
    a = F.pad(dt * A, (0, 0, 0, pad)).view(Bsz, nc, T, H).permute(0, 3, 1, 2)  # (B, L, H) -> (B, H, nc, T), a <= 0
    xt = F.pad(x * dt[..., None], (0, 0, 0, 0, 0, pad)).view(Bsz, nc, T, H, P)  # x̃ = dt·x -> (B, nc, T, H, P)
    Bc = F.pad(B, (0, 0, 0, pad)).view(Bsz, nc, T, N)  # (B, nc, T, N)
    Cc = F.pad(C, (0, 0, 0, pad)).view(Bsz, nc, T, N)
    a_cs = a.cumsum(-1)  # (B, H, nc, T)

    # 1. intra-chunk (diagonal blocks)
    Lmat = torch.exp(segsum(a))  # (B, H, nc, T, T)
    y_diag = torch.einsum("bcln,bcsn,bhcls,bcshp->bclhp", Cc, Bc, Lmat, xt)  # (B, nc, T, H, P)

    # 2. per-chunk states, decayed to the chunk end
    decay_to_end = torch.exp(a_cs[..., -1:] - a_cs)  # (B, H, nc, T)
    S = torch.einsum("bcln,bhcl,bclhp->bchpn", Bc, decay_to_end, xt)  # (B, nc, H, P, N)

    # 3. inter-chunk scan, sequential over chunks: H_0 = 0, H_{c+1} = exp(Σ a over chunk c)·H_c + S_c
    chunk_decay = torch.exp(a_cs[..., -1])  # (B, H, nc)
    h = x.new_zeros(Bsz, H, P, N)
    entering = []
    for c in range(nc):
        entering.append(h)
        h = chunk_decay[:, :, c, None, None] * h + S[:, c]
    H_in = torch.stack(entering, dim=1)  # (B, nc, H, P, N) state entering each chunk

    # 4. state -> output
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", Cc, H_in, torch.exp(a_cs))  # (B, nc, T, H, P)

    y = (y_diag + y_off).reshape(Bsz, nc * T, H, P)[:, :L]  # crop padding -> (B, L, H, P)
    return y + D[:, None] * x


# [TARGET]
class SSDBranch(nn.Module):
    """Mamba-2 / SSD branch, single B/C group (ARCHITECTURE §4.1). Unified RoPE on B and C only, after the conv."""

    def __init__(self, cfg: MixerConfig):
        super().__init__()
        d = cfg.d_model
        self.d_in, self.d_state = cfg.ssm_expand * d, cfg.ssm_d_state
        self.head_dim, self.chunk = cfg.ssm_head_dim, cfg.ssm_chunk
        self.n_heads = H = self.d_in // cfg.ssm_head_dim
        conv_dim = self.d_in + 2 * self.d_state
        self.in_proj = nn.Linear(d, 2 * self.d_in + 2 * self.d_state + H, bias=False)  # -> [z | x | B | C | dt]
        self.conv_pad = cfg.ssm_conv_kernel - 1  # causal left-pad
        self.conv = nn.Conv1d(conv_dim, conv_dim, cfg.ssm_conv_kernel, groups=conv_dim)
        self.A_log = nn.Parameter(torch.empty(H).uniform_(1, 16).log())
        dt0 = torch.empty(H).uniform_(math.log(1e-3), math.log(1e-1)).exp()
        self.dt_bias = nn.Parameter(dt0 + torch.log(-torch.expm1(-dt0)))  # softplus⁻¹(dt0)
        self.D = nn.Parameter(torch.ones(H))
        self.norm = nn.RMSNorm(self.d_in)
        self.out_proj = nn.Linear(self.d_in, d, bias=False)

    def forward(self, x, cos, sin):
        """x: (B, L, d); cos, sin: (L, N/2) -> (B, L, d)."""
        Bsz, L, _ = x.shape
        z, xBC, dt = self.in_proj(x).split([self.d_in, self.d_in + 2 * self.d_state, self.n_heads], dim=-1)
        xBC = F.pad(xBC.transpose(1, 2), (self.conv_pad, 0))  # (B, L, C) -> (B, C, k−1 + L)
        xBC = F.silu(self.conv(xBC)).transpose(1, 2)  # (B, C, L) -> (B, L, C)
        xs, B, C = xBC.split([self.d_in, self.d_state, self.d_state], dim=-1)
        B = apply_rope(B[:, :, None], cos, sin)[:, :, 0]  # (B, L, N) -> (B, L, 1, N) -> (B, L, N)
        C = apply_rope(C[:, :, None], cos, sin)[:, :, 0]
        dt = F.softplus(dt + self.dt_bias)  # (B, L, H) > 0
        A = -torch.exp(self.A_log)  # (H,) < 0
        y = ssd_efficient(xs.reshape(Bsz, L, self.n_heads, self.head_dim), dt, A, B, C, self.D, self.chunk)  # (B, L, H, P)
        y = self.norm(y.reshape(Bsz, L, self.d_in) * F.silu(z))  # gated RMSNorm: (B, L, H, P) -> (B, L, d_in)
        return self.out_proj(y)
