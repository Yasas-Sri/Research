import collections
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid.attention import SlidingWindowAttention
from hybrid.config import MixerConfig
from hybrid.engram import Engram
from hybrid.rope import UnifiedRoPE
from hybrid.ssm import SSDBranch

# AGENTS M5. vocab 16: the batch's exact no-context loss floor is >= 1.0 nats, so loss < 0.5 needs context.
VOCAB, B, L, STEPS, LR, TARGET = 16, 2, 64, 300, 3e-3, 0.5


def smoke_cfg():
    # window 16 < L (else SWA is dense); chunk 16 < L (else no inter-chunk scan)
    return MixerConfig(vocab_size=VOCAB, attn_window=16, ssm_chunk=16)


def fixed_batch():
    """Call right after the seed fixture: (B, L) ids -> inputs (B, L-1), next-token targets (B, L-1)."""
    ids = torch.randint(0, VOCAB, (B, L))
    return ids[:, :-1], ids[:, 1:]


def no_context_floor(inp, tgt):
    """Exact minimum mean CE of any model that sees only the current token: entropy of tgt given inp."""
    counts = collections.defaultdict(collections.Counter)
    for a, b in zip(inp.flatten().tolist(), tgt.flatten().tolist()):
        counts[a][b] += 1
    return -sum(k * math.log(k / sum(c.values())) for c in counts.values() for k in c.values()) / tgt.numel()


class Wrapper(nn.Module):
    """Throwaway LM for the smoke tests only — NOT the fusion layer.
    Embedding -> x + call(branch) -> RMSNorm -> LM head; `call(m, ids, x)` adapts each branch's signature."""

    def __init__(self, cfg, branch, call):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, cfg.d_model)
        self.norm_in, self.norm_out = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.branch, self.call = branch, call
        self.head = nn.Linear(cfg.d_model, VOCAB, bias=False)

    def forward(self, ids):
        """ids: (B, T) -> logits (B, T, VOCAB)."""
        x = self.embed(ids)
        x = x + self.call(self, ids, x)
        return self.head(self.norm_out(x))


def overfit(model, inp, tgt):
    """AdamW on one batch until loss < TARGET or STEPS updates. Returns (updates taken, last loss)."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(STEPS + 1):
        loss = F.cross_entropy(model(inp).flatten(0, 1), tgt.flatten())
        if loss.item() < TARGET or step == STEPS:
            return step, loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()


def rope_call(cfg):
    cos, sin = UnifiedRoPE(cfg.attn_head_dim, cfg.rope_base)(torch.arange(L - 1))
    return lambda m, ids, x: m.branch(m.norm_in(x), cos, sin)


def test_smoke_batch_needs_context(seed):
    inp, tgt = fixed_batch()
    assert no_context_floor(inp, tgt) >= 1.0  # measured 1.848 (seed 0)


@pytest.mark.slow
def test_smoke_swa_overfits(seed):
    inp, tgt = fixed_batch()
    cfg = smoke_cfg()
    steps, loss = overfit(Wrapper(cfg, SlidingWindowAttention(cfg), rope_call(cfg)), inp, tgt)
    assert loss < TARGET, f"loss {loss:.3f} after {steps} steps"


@pytest.mark.slow
def test_smoke_ssd_overfits(seed):
    inp, tgt = fixed_batch()
    cfg = smoke_cfg()
    steps, loss = overfit(Wrapper(cfg, SSDBranch(cfg), rope_call(cfg)), inp, tgt)
    assert loss < TARGET, f"loss {loss:.3f} after {steps} steps"


@pytest.mark.slow
def test_smoke_engram_overfits(seed):
    inp, tgt = fixed_batch()
    cfg = smoke_cfg()
    steps, loss = overfit(Wrapper(cfg, Engram(cfg), lambda m, ids, x: m.branch(ids, x)), inp, tgt)  # query = embedding
    assert loss < TARGET, f"loss {loss:.3f} after {steps} steps"
