import io

import pytest
import torch

from conftest import assert_causal
from hybrid.config import MixerConfig
from hybrid.engram import (
    Engram,
    hash_ngrams_efficient,
    hash_ngrams_reference,
    lookup_efficient,
    lookup_reference,
    next_primes,
    ngram_xor_hash,
)

VOCAB, D = 100, 64


def small_cfg(**kw):
    kw = dict(d_model=D, attn_head_dim=16, ssm_d_state=16, ssm_head_dim=16, vocab_size=VOCAB,
              engram_head_dim=8, engram_table_size=31) | kw
    return MixerConfig(**kw)


def hash_args(m):
    return m.orders, m.multipliers, m.primes, m.pad_id


def test_hash_deterministic_across_calls_and_state_dict(seed):
    m = Engram(small_cfg())
    ids = torch.randint(0, VOCAB, (2, 12))
    z = hash_ngrams_efficient(ids, *hash_args(m))
    assert torch.equal(z, hash_ngrams_efficient(ids, *hash_args(m)))
    assert torch.equal(z, hash_ngrams_efficient(ids, *hash_args(Engram(small_cfg()))))  # same seed, new module

    other = Engram(small_cfg(engram_seed=999))
    assert not torch.equal(other.multipliers, m.multipliers)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    other.load_state_dict(torch.load(buf))
    assert torch.equal(z, hash_ngrams_efficient(ids, *hash_args(other)))  # constants travel with the checkpoint

    consts = {"multipliers", "primes", "offsets"}
    assert consts <= dict(m.named_buffers()).keys()
    assert not consts & dict(m.named_parameters()).keys()


def test_hash_locality(seed):
    m = Engram(small_cfg())
    ids = torch.randint(0, VOCAB, (1, 12))
    z = hash_ngrams_efficient(ids, *hash_args(m))  # (1, L, |N|, K)
    for i, n in enumerate(m.orders):
        mult = m.multipliers[i, :, :n]  # (K, n)
        for t in range(n, ids.shape[1]):
            outside, inside = ids.clone(), ids.clone()
            outside[0, t - n] = (ids[0, t - n] + 1) % VOCAB
            inside[0, t - n + 1] = (ids[0, t - n + 1] + 1) % VOCAB
            assert torch.equal(hash_ngrams_efficient(outside, *hash_args(m))[0, t, i], z[0, t, i])
            # pre-mod hash always changes (odd multiplier, XOR); after % M a single head may collide
            pre, pre_in = ngram_xor_hash(ids, mult, m.pad_id), ngram_xor_hash(inside, mult, m.pad_id)
            assert (pre_in[0, t] != pre[0, t]).all()
            assert (hash_ngrams_efficient(inside, *hash_args(m))[0, t, i] != z[0, t, i]).any()


def test_index_range_and_tables_disjoint(seed):
    assert next_primes(24, 3) == [29, 31, 37]
    m = Engram(small_cfg(engram_table_size=30))  # non-prime start rounds up to 31
    primes = m.primes.flatten().tolist()
    assert primes == [31, 37, 41, 43, 47, 53, 59, 61]
    assert m.offsets.flatten().tolist() == [sum(primes[:i]) for i in range(len(primes))]
    assert m.embed.num_embeddings == sum(primes)

    ids = torch.randint(0, VOCAB, (3, 20))
    ids[0] = VOCAB - 1
    z = hash_ngrams_efficient(ids, *hash_args(m))  # (B, L, |N|, K); positions t < n hit pad_id
    assert (z >= 0).all() and (z < m.primes).all()


@pytest.mark.parametrize("cfg", [small_cfg(), MixerConfig()], ids=["small", "default"])
@pytest.mark.parametrize("L", [1, 2, 5, 17])
def test_hash_reference_equals_efficient(seed, cfg, L):
    m = Engram(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, L))
    ids[0, -1] = cfg.vocab_size - 1
    assert torch.equal(hash_ngrams_reference(ids, *hash_args(m)), hash_ngrams_efficient(ids, *hash_args(m)))


def test_lookup_reference_equals_efficient(seed, float64):
    m = Engram(small_cfg())
    z = hash_ngrams_efficient(torch.randint(0, VOCAB, (2, 9)), *hash_args(m))
    ref = lookup_reference(z, m.embed.weight, m.offsets)
    eff = lookup_efficient(z, m.embed.weight, m.offsets)
    assert eff.shape == (2, 9, 64)
    assert torch.allclose(ref, eff)


@pytest.mark.parametrize("t0", [1, 5, 11])
def test_engram_causal(seed, float64, t0):
    m = Engram(small_cfg())
    torch.nn.init.normal_(m.conv.weight)  # non-zero conv so its receptive field is exercised
    ids, q = torch.randint(0, VOCAB, (2, 16)), torch.randn(2, 16, D)
    assert_causal(m, (ids, q), t0)


def test_gate_in_open_unit_interval(seed):
    m = Engram(small_cfg())
    alpha, _ = m.gated_value(torch.randint(0, VOCAB, (2, 16)), 10 * torch.randn(2, 16, D))
    assert alpha.shape == (2, 16, 1)
    assert (alpha > 0).all() and (alpha < 1).all()


def test_zero_init_conv_output_equals_gated_value(seed):
    m = Engram(small_cfg())
    assert not m.conv.weight.any()
    ids, q = torch.randint(0, VOCAB, (2, 16)), torch.randn(2, 16, D)
    _, v_tilde = m.gated_value(ids, q)
    assert torch.equal(m(ids, q), v_tilde)


def test_grad_only_on_looked_up_rows(seed):
    m = Engram(small_cfg())
    ids, q = torch.randint(0, VOCAB, (2, 6)), torch.randn(2, 6, D)
    (m(ids, q) * torch.randn(2, 6, D)).sum().backward()
    looked_up = (hash_ngrams_efficient(ids, *hash_args(m)) + m.offsets).unique()  # sorted
    with_grad = m.embed.weight.grad.abs().sum(-1).nonzero().flatten()  # sorted
    assert torch.equal(with_grad, looked_up)


def test_engram_output_shape_and_finite(seed):
    cfg = MixerConfig()
    y = Engram(cfg)(torch.randint(0, cfg.vocab_size, (2, 16)), torch.randn(2, 16, cfg.d_model))
    assert y.shape == (2, 16, cfg.d_model)
    assert torch.isfinite(y).all()
