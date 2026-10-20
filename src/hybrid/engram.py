import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid.config import MixerConfig


def next_primes(start, count):
    """First `count` primes >= start, by trial division (start ~1e4, so this is instant)."""
    primes, p = [], max(start, 2)
    while len(primes) < count:
        if all(p % d for d in range(2, int(p**0.5) + 1)):
            primes.append(p)
        p += 1
    return primes


# [TARGET]
def hash_ngrams_reference(input_ids, orders, multipliers, primes, pad_id):
    """Multiplicative-XOR hash of suffix N-grams with Python ints (ARCHITECTURE §5.1 step 2). Test-only.

    input_ids:   (B, L) long
    orders:      tuple of N-gram orders, length |N|
    multipliers: (|N|, K, n_max) int64; entries j >= n of order n are unused
    primes:      (|N|, K) int64 table sizes M_{n,k}
    pad_id:      token used where t - j < 0
    returns:     z (B, L, |N|, K) int64, 0 <= z < M_{n,k}
    """
    ids, mult, M = input_ids.tolist(), multipliers.tolist(), primes.tolist()
    Bsz, L = input_ids.shape
    K = primes.shape[1]
    z = torch.zeros(Bsz, L, len(orders), K, dtype=torch.long, device=input_ids.device)
    for b in range(Bsz):
        for t in range(L):
            for i, n in enumerate(orders):
                for k in range(K):
                    h = 0
                    for j in range(n):
                        x = ids[b][t - j] if t - j >= 0 else pad_id
                        h ^= x * mult[i][k][j]
                    z[b, t, i, k] = h % M[i][k]
    return z


# [TARGET]
def ngram_xor_hash(input_ids, mult, pad_id):
    """Pre-mod hash  ⊕_j x_{t-j}·m_{k,j}  for one order n = mult.shape[1].

    input_ids: (B, L) long
    mult:      (K, n) int64
    returns:   (B, L, K) int64, >= 0 (no overflow: vocab·2³¹ < 2⁶³)
    """
    n, L = mult.shape[1], input_ids.shape[1]
    padded = F.pad(input_ids, (n - 1, 0), value=pad_id)  # (B, L+n-1)
    grams = torch.stack([padded[:, n - 1 - j : n - 1 - j + L] for j in range(n)], dim=-1)  # (B, L, n): x_{t-j}
    prod = grams[:, :, None, :] * mult  # (B, L, 1, n) * (K, n) -> (B, L, K, n)
    h = prod[..., 0]
    for j in range(1, n):
        h = h ^ prod[..., j]
    return h  # (B, L, K)


# [TARGET]
def hash_ngrams_efficient(input_ids, orders, multipliers, primes, pad_id):
    """Vectorised hashing; same arguments and output as hash_ngrams_reference."""
    z = [ngram_xor_hash(input_ids, multipliers[i, :, :n], pad_id) % primes[i] for i, n in enumerate(orders)]  # each (B, L, K)
    return torch.stack(z, dim=2)  # (B, L, |N|, K)


# [TARGET]
def lookup_reference(z, weight, offsets):
    """Per-table gather from the packed embedding weight. Test-only.

    z:       (B, L, |N|, K) long, un-offset indices
    weight:  (Σ M, d_e)
    offsets: (|N|, K) long, start row of each table
    returns: (B, L, |N|·K·d_e), tables concatenated in (n, k) order
    """
    out = []
    for i in range(z.shape[2]):
        for k in range(z.shape[3]):
            table = weight[int(offsets[i, k]) :]  # table (n, k) starts at its offset
            out.append(table[z[:, :, i, k]])  # (B, L, d_e)
    return torch.cat(out, dim=-1)


# [TARGET]
def lookup_efficient(z, weight, offsets):
    """One F.embedding call; same arguments and output as lookup_reference."""
    e = F.embedding(z + offsets, weight)  # (B, L, |N|, K) -> (B, L, |N|, K, d_e)
    return e.flatten(2)  # (B, L, |N|·K·d_e)


# [TARGET]
class Engram(nn.Module):
    """Engram N-gram memory (ARCHITECTURE §5). Returns Y; the caller does the residual add."""

    def __init__(self, cfg: MixerConfig):
        super().__init__()
        self.orders = tuple(cfg.engram_orders)
        self.pad_id = cfg.vocab_size
        n_ord, K, n_max = len(self.orders), cfg.engram_heads_per_order, max(self.orders)
        d, d_mem = cfg.d_model, n_ord * K * cfg.engram_head_dim

        g = torch.Generator().manual_seed(cfg.engram_seed)
        mult = torch.randint(0, 2**30, (n_ord, K, n_max), generator=g) * 2 + 1  # odd, in [1, 2³¹)
        for i, n in enumerate(self.orders):
            mult[i, :, n:] = 0  # unused slots of orders below n_max
        primes = torch.tensor(next_primes(cfg.engram_table_size, n_ord * K))  # (|N|·K,) ascending
        offsets = torch.cat([primes.new_zeros(1), primes.cumsum(0)[:-1]])  # (|N|·K,) exclusive cumsum
        self.register_buffer("multipliers", mult)
        self.register_buffer("primes", primes.view(n_ord, K))
        self.register_buffer("offsets", offsets.view(n_ord, K))

        self.embed = nn.Embedding(int(primes.sum()), cfg.engram_head_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.W_K = nn.Linear(d_mem, d, bias=False)
        self.W_V = nn.Linear(d_mem, d, bias=False)
        self.norm_q = nn.RMSNorm(d)
        self.norm_k = nn.RMSNorm(d)
        self.norm_v = nn.RMSNorm(d)
        self.conv_pad = (cfg.engram_conv_kernel - 1) * n_max  # causal left-pad, dilation = max order
        self.conv = nn.Conv1d(d, d, cfg.engram_conv_kernel, dilation=n_max, groups=d, bias=False)
        nn.init.zeros_(self.conv.weight)

    def gated_value(self, input_ids, query):
        """Context-aware gate (§5.1 steps 1-5).

        input_ids: (B, L) long; query: (B, L, d)
        returns:   alpha (B, L, 1) in (0, 1), v_tilde (B, L, d)
        """
        z = hash_ngrams_efficient(input_ids, self.orders, self.multipliers, self.primes, self.pad_id)  # (B, L, |N|, K)
        e = lookup_efficient(z, self.embed.weight, self.offsets)  # (B, L, d_mem)
        k, v = self.W_K(e), self.W_V(e)  # (B, L, d) each
        score = (self.norm_q(query) * self.norm_k(k)).sum(-1, keepdim=True)  # (B, L, d) -> (B, L, 1)
        alpha = torch.sigmoid(score / k.shape[-1] ** 0.5)
        return alpha, alpha * v

    def forward(self, input_ids, query):
        """input_ids: (B, L) long; query: (B, L, d) -> Y: (B, L, d) = SiLU(conv(RMSNorm(Ṽ))) + Ṽ."""
        _, v = self.gated_value(input_ids, query)
        u = F.pad(self.norm_v(v).transpose(1, 2), (self.conv_pad, 0))  # (B, d, L) -> (B, d, pad + L)
        return F.silu(self.conv(u)).transpose(1, 2) + v  # (B, d, L) -> (B, L, d)
