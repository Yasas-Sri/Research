# ARCHITECTURE.md — Hybrid Sub-Quadratic Sequence Mixer

> Specification for the sequence-mixing components of the FYP
> *"Efficient Long-Context Sequence Modelling with a Hybrid Sub-Quadratic Architecture:
> Intra-Layer Fusion of Selective State-Space Models and Sparse Attention."*
>
> **Scope of this revision (Reflective Journal 2 work package):**
> 1. Engram N-gram memory module — implemented from scratch, with reference + efficient paths and tests.
> 2. Sliding-window attention (SWA) branch and selective-SSM (Mamba-2 / SSD) branch, both using a single Unified RoPE.
>
> The intra-layer **fusion operator is NOT part of this revision** (Journal plan task 3). This file defines the
> branch interfaces so fusion can be added later without touching branch internals.

Stability tags used throughout: **[STABLE]** = settled, do not change without a recorded decision ·
**[TARGET]** = build in this revision · **[EXPERIMENTAL]** = design axis under investigation, do not build unless asked.

---

## 0. Notation and global conventions

| Symbol | Meaning | Tiny-config default |
|---|---|---|
| `B` | batch size | — |
| `L` | sequence length | — |
| `d` | model width (`d_model`) | 256 |
| `H_a`, `D_h` | attention heads, attention head dim | 4, 64 |
| `w` | attention window (keys per query, **including self**) | 128 |
| `E` | SSM expansion factor | 2 |
| `d_in = E·d` | SSM inner width | 512 |
| `P` | SSM head dim | 64 |
| `H_s = d_in / P` | SSM heads | 8 |
| `N` | SSM state dim (`d_state`) | 64 |
| `T` | SSD chunk length | 64 |
| `n ∈ 𝒩` | Engram N-gram orders | {2, 3} |
| `K` | Engram hash heads per order | 4 |
| `d_e` | Engram embedding dim per (order, head) | 32 |
| `d_mem = |𝒩|·K·d_e` | Engram memory vector width | 256 |

**Tensor layout [STABLE].** All branch inputs/outputs are `(B, L, d)`. Per-head tensors are `(B, L, H, D)`.
Every function docstring states input and output shapes.

**Positions [STABLE].** Positions are an explicit `LongTensor` of shape `(L,)`, default `torch.arange(L)`.
They are passed in, never recomputed inside a branch, so the relative-position tests (§2.4) can shift them.

**Reference vs efficient paths [STABLE].** Every non-trivial algorithm exists as two pure functions:
`<name>_reference` (obviously correct, slow, test-only) and `<name>_efficient` (the one `forward` calls).
Both take identical arguments and are tested for numerical equivalence in `float64`. No runtime flag selects
between them — tests call the reference function directly.

**Hardware [STABLE].** Pure PyTorch, CPU-first. No CUDA-only code paths, no Triton, no `mamba_ssm`,
no `flash_attn`, no `causal_conv1d`. Everything must run and pass tests on a laptop CPU.

---

## 1. Target layer (context only — fusion is a later milestone)

```
             x  (B, L, d)                     input_ids (B, L)
             │                                     │
        RMSNorm                                    │
             │                                     │
   ┌─────────┴───────────┐                         │
   │                     │       positions ──► UnifiedRoPE ──► (cos, sin)  shared by both branches
   ▼                     ▼                         │
 SWA branch          SSD branch                    │
 RoPE on Q,K         RoPE on C,B                   │
 O(L·w)              O(L·T)                        │
   │                     │                         │
   │                  y_ssm ──── query ──► Engram ◄┘   O(1) lookup / token
   │                     │                    │
   │                     └──── + ◄────────────┘   post-SSM additive gated residual [STABLE placement]
   │                     │
   └──── FUSION (Journal task 3, not this revision) ────► + x  → MLP (dense, no MoE) → ...
```

Branch interfaces this revision must expose:

```python
SlidingWindowAttention.forward(x: (B,L,d), cos: (L,D_h/2), sin: (L,D_h/2)) -> (B,L,d)
SSDBranch.forward(x: (B,L,d), cos: (L,N/2), sin: (L,N/2))                  -> (B,L,d)
Engram.forward(input_ids: (B,L) long, query: (B,L,d))                      -> (B,L,d)
UnifiedRoPE.forward(positions: (L,) long) -> (cos, sin)  each (L, dim/2)
```

The Engram module does **not** add its output to anything itself — the caller (fusion milestone) does the
residual add. This keeps the placement decision outside the module.

---

## 2. Unified RoPE  [TARGET]

### 2.1 Definition
Following TransXSSM (arXiv:2506.09507, §2.2), the **same** block-diagonal rotation is applied to the
"query-like" and "key-like" vectors of both mixers:

| Mixer | query-like (position m) | key-like (position n) |
|---|---|---|
| Attention | `q_m` | `k_n` |
| SSD | `c_m` (C projection) | `b_n` (B projection) |

$$R^{D}_{\Theta,m}=\bigoplus_{i=0}^{D/2-1}\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix},\qquad \theta_i = \text{base}^{-2i/D},\ \text{base}=10000$$

so that ⟨R_m q, R_n k⟩ = qᵀ R_{n−m} k and ⟨R_m c, R_n b⟩ = cᵀ R_{n−m} b: both score types depend only on relative position.

### 2.2 Implementation decisions
- **Rotate-half convention** (pairs dimension `i` with `i + D/2`). This is a fixed permutation of the
  block-diagonal form above and has identical relative-position properties. Use one `apply_rope` function
  for both branches — never two implementations.
- **One rotation, one dimension [STABLE, assumption A2].** To make *R* literally identical across branches,
  require `D_h == N` (both 64 in the tiny config). The config validator must assert this.
- cos/sin are computed once per layer forward from `positions` and passed to both branches.
- Compute angles in `float32` minimum (`float64` in tests); cast cos/sin to the activation dtype at the end.
- RoPE is **not** applied to Engram (N-gram hashing already encodes local order) [STABLE].

### 2.3 Reference
```python
def rope_cos_sin(positions, dim, base=10000.0):
    inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)  # (dim/2,)
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]            # (L, dim/2)
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):  # x (B, L, H, D); cos/sin (L, D/2)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
```
SSD's `B`, `C` are `(B, L, N)` (one group); unsqueeze a head axis to reuse `apply_rope`.

### 2.4 Invariants (must be unit-tested)
1. Norm preservation: `‖apply_rope(x)‖ == ‖x‖` per vector.
2. Relative property: `⟨R_m q, R_n k⟩` is unchanged when `m, n` are both shifted by the same offset.
3. **Branch-level shift invariance:** running either branch with `positions` and with `positions + 1000`
   gives the same output (verified for both blocked SWA and chunked SSD during design).
4. RoPE actually changes the SSD output vs. no RoPE (guards against a silently skipped rotation).

---

## 3. Sliding-Window Attention branch  [TARGET]

### 3.1 Definition
Causal multi-head attention where query `t` attends keys `s` with `t − w < s ≤ t` (exactly `w` keys incl. self).

```
q, k, v = x W_q, x W_k, x W_v          # (B,L,d) -> (B,L,H_a,D_h); no biases
q, k    = apply_rope(q), apply_rope(k)
o_t     = Σ_{s=t-w+1..t} softmax_s(q_t·k_s / √D_h) v_s
out     = concat_heads(o) W_o           # (B,L,d)
```

### 3.2 Reference path — `swa_reference(q, k, v, w)`
Full `(L, L)` score matrix with band mask `(s ≤ t) & (t − s < w)`, softmax, weighted sum.
O(L²) memory — tests only.

### 3.3 Efficient path — `swa_efficient(q, k, v, w)` (blocked, verified)
1. Right-pad `L` to a multiple of `w`; reshape into `nb = ⌈L/w⌉` blocks of `w` queries.
2. For each block, keys/values = `[previous block ‖ current block]` → `2w` keys. Block 0's "previous" is zero
   padding and fully masked.
3. Static mask over local indices: query `i ∈ [0,w)`, key `j ∈ [0,2w)` is allowed iff `i < j ≤ i + w`.
   (Derivation: `t = bw + i`, `s = (b−1)w + j`, need `0 ≤ t − s < w`.)
4. Batched einsum scores `(B, nb, H, w, 2w)`, masked softmax, einsum with values, reshape, crop to `L`.

Right-padding is safe: padded keys sit at positions `> L−1`, so no real query can see them (causality).

| | time | activation memory |
|---|---|---|
| reference | O(L²·D_h·H_a) | O(L²·H_a) |
| efficient | O(L·2w·D_h·H_a) | O(L·2w·H_a) |

`w ≥ L` must still be correct (it degenerates to dense causal attention) — this is the dense baseline used later.

### 3.4 Invariants (must be unit-tested)
- `swa_efficient == swa_reference` (`float64`, atol 1e-10) for `w ∈ {1, 4, 8, L, 2L}`; `L` is chosen so that `L` is not a multiple of `w` whenever `1 < w < L`.
- Causality: perturbing inputs at positions `≥ t0` leaves outputs `< t0` unchanged.
- Window: gradient of `out[:, t]` w.r.t. `v` is non-zero **only** on `[t−w+1, t]`.
- Shift invariance under `positions + offset` (§2.4.3).

---

## 4. Selective SSM branch — Mamba-2 / SSD  [TARGET]

> **Assumption A1:** this revision implements Mamba-2/SSD, the variant preferred in Journal 1 for cleaner
> Unified RoPE integration (its `C_tᵀ B_s` term is a bilinear form like `q_tᵀ k_s`). "Mamba-1 vs Mamba-2"
> remains formally open; Mamba-1 is **not** built in this revision.

### 4.1 Block (Dao & Gu, 2024), single B/C group

```
zxbcdt   = in_proj(x)                              # d -> [z: d_in | x: d_in | B: N | C: N | dt: H_s]
xBC      = SiLU(causal_depthwise_conv1d(xBC, k=4)) # over the concatenated x,B,C channels
x, B, C  = split(xBC)                              # x -> (B,L,H_s,P); B, C -> (B,L,N)
B, C     = apply_rope(B), apply_rope(C)            # Unified RoPE, applied AFTER the conv (our choice: rotation must be the last op before C·B)
dt       = softplus(dt + dt_bias)                  # (B,L,H_s) > 0
A        = -exp(A_log)                             # (H_s,) < 0, scalar per head
y        = SSD(x, dt, A, B, C) + D ⊙ x             # (B,L,H_s,P)
y        = RMSNorm(y_flat ⊙ SiLU(z))               # gated norm, d_in
out      = out_proj(y)                             # d_in -> d
```

Recurrence being computed, per head *h*:

$$h_t = e^{\Delta_t A}\,h_{t-1} + \Delta_t\,x_t\,b_t^{\top},\qquad y_t = h_t\,c_t + D\,x_t,\qquad h_t\in\mathbb R^{P\times N}$$

Initialisation (Mamba-2 defaults): `A_log = log(U[1,16])`; `dt_bias = softplus⁻¹(dt₀)` with
`dt₀ ~ exp(U[log 1e-3, log 1e-1])`; `D = 1`; `A_log`, `dt_bias`, `D` excluded from weight decay.

### 4.2 Reference path — `ssd_reference(x, dt, A, B, C, D)`
Literal Python loop over `t` implementing the recurrence above. O(L) sequential steps, tests only.

### 4.3 Efficient path — `ssd_efficient(x, dt, A, B, C, D, chunk=T)` (chunked SSD, verified)
Let `a_t = Δ_t A` (log-decay ≤ 0) and `x̃_t = Δ_t x_t`. Right-pad to a multiple of `T` with `x̃ = 0, a = 0, B = C = 0`.

1. **Intra-chunk (diagonal blocks):** `Lmat[i,j] = exp(Σ_{k=j+1..i} a_k)` for `j ≤ i`, else 0, built with a
   stable `segsum` (cumsum differences, masked to −inf above the diagonal).
   `y_diag = einsum("bcln,bcsn,bhcls,bcshp->bclhp", C, B, Lmat, x̃)`.
2. **Per-chunk states:** `S_c = Σ_l exp(a_cs[end] − a_cs[l]) · x̃_l b_lᵀ`.
3. **Inter-chunk scan (sequential over chunks, O(L/T)):** `H_0 = 0`; `H_{c+1} = exp(Σ a over chunk c)·H_c + S_c`.
   Use a plain Python loop over chunks — keeps total cost strictly linear in L and easy to read.
4. **State → output:** `y_off = einsum("bcln,bchpn,bhcl->bclhp", C, H_entering, exp(a_cs))`.
5. `y = y_diag + y_off`, crop to `L`, add `D ⊙ x`.

| | time | activation memory |
|---|---|---|
| reference | O(L·H_s·P·N), sequential | O(H_s·P·N) state |
| efficient | O(L·T·(N + P)·H_s + L·P·N·H_s) | O(L·T·H_s + (L/T)·H_s·P·N) |

Linear in `L` for fixed `T`. The quadratic work is confined to `T × T` chunks.

### 4.4 Invariants (must be unit-tested)
- `ssd_efficient == ssd_reference` (`float64`, atol 1e-9) for `T ∈ {1, 4, 8, 16, 64}` and `L` not a multiple of `T`.
- Chunk-size invariance (follows from the above, but test directly: `T=8` vs `T=32`).
- Causality: perturb `x` at `t ≥ t0`, outputs at `t < t0` unchanged.
- Decay sanity: `exp(Δ_t A) ∈ (0, 1)` for all heads at init.
- Shift invariance under `positions + offset`, and RoPE-on vs RoPE-off outputs differ (§2.4).
- Full block: output shape `(B,L,d)`, finite, gradients reach `A_log`, `dt_bias`, `D`, conv and projections.

---

## 5. Engram memory module  [TARGET]

Based on Cheng et al., *Conditional Memory via Scalable Lookup* (DeepSeek-AI, arXiv:2601.07372), adapted to the
FYP's post-SSM placement. Differences from the paper are listed in §5.6.

### 5.1 Forward

1. **Suffix N-grams.** For each order `n ∈ 𝒩` and position `t`: `g_{t,n} = (x_t, x_{t−1}, …, x_{t−n+1})`,
   left-padded with a reserved `pad_id = vocab_size` where `t − j < 0`.
2. **Multi-head hashing.** For each order `n` and head `k ∈ [0,K)`, an index into a table of prime size `M_{n,k}`:

   $$z_{t,n,k} = \Big(\bigoplus_{j=0}^{n-1} x_{t-j}\cdot m_{n,k,j}\Big) \bmod M_{n,k}$$

   `⊕` is bitwise XOR on `int64`; multipliers `m_{n,k,j}` are odd integers in `[1, 2³¹)` drawn once from a
   seeded generator and **registered as buffers** (saved in checkpoints, so lookups are reproducible).
   Overflow check: `vocab·2³¹ < 2⁶³` for any realistic vocab. The paper only says "multiplicative-XOR hash";
   this exact formula is **our** stated choice.
3. **Lookup.** `e_{t,n,k} = E_{n,k}[z_{t,n,k}] ∈ ℝ^{d_e}`. All tables are packed into **one** `nn.Embedding`
   of size `Σ M_{n,k}` using per-(n,k) offsets. Table sizes: distinct primes, ascending, starting from
   `table_size` (tiny config: 10 007, 10 009, …). Embedding init `N(0, 0.02)`.
4. **Concatenate.** `e_t = ‖_{n,k} e_{t,n,k} ∈ ℝ^{d_mem}`.
5. **Context-aware gate** (scalar per token):

   $$k_t = W_K e_t,\quad v_t = W_V e_t,\quad \alpha_t = \sigma\!\Big(\tfrac{\mathrm{RMSNorm}(h_t)^{\top}\,\mathrm{RMSNorm}(k_t)}{\sqrt d}\Big),\quad \tilde v_t = \alpha_t v_t$$

   `h_t` is the `query` argument (in the planned layer: the SSM branch output `y_ssm`). Query is **not** projected.
6. **Short causal conv.** `Y = SiLU(DWConv1d(RMSNorm(Ṽ))) + Ṽ`, depthwise over `d` channels, kernel 4,
   dilation `max(𝒩)` (= 3), causal left-pad `(4−1)·3 = 9`. **Conv weights zero-initialised**, so at init `Y = Ṽ`.
7. Return `Y` `(B, L, d)`. The caller adds it to the stream.

### 5.2 Reference vs efficient
| Step | `*_reference` | `*_efficient` |
|---|---|---|
| hashing | Python loops over `b, t, n, k, j` with Python ints | stack of shifted copies `(B,L,n)`, broadcast multiply with `(K,n)` multipliers, XOR-reduce, `% primes` |
| lookup | loop over `(n,k)`, slice the packed weight with its offset, gather | add offsets to all indices, one `F.embedding` call |

Equivalence must be **exact** (`torch.equal`) for indices and `allclose` for embeddings. The hashing pair was
verified exact during design.

### 5.3 Complexity
Per token: `|𝒩|·K` hash computations + gathers → **O(1) in L**; projections O(d_mem·d); conv O(4·d).
Parameters: `Σ M_{n,k}·d_e` (tables) + `2·d_mem·d` + norms + conv. Memory does not decay with distance — the
reason it sits outside the SSM recurrence [STABLE].

### 5.4 Invariants (must be unit-tested)
- Hash determinism: same ids + same seed → same indices across calls and across a save/load of `state_dict`.
- Hash locality: `z_{t,n,·}` depends only on `x_{t−n+1..t}` (change a token outside the n-gram → index unchanged).
- Index range: `0 ≤ z_{t,n,k} < M_{n,k}`; offsets never make two tables overlap.
- Reference == efficient (indices exact, embeddings allclose).
- Causality of the whole module: changing `input_ids` or `query` at `t ≥ t0` leaves outputs `< t0` unchanged
  (conv receptive field is backward only).
- Gate range: `0 < α_t < 1`.
- Zero-init conv: at init, `Y == Ṽ` exactly.
- Gradients reach the embedding rows that were looked up, and **only** those rows.

### 5.5 Optimizer notes (for the training milestone, not implemented now)
Paper: embedding tables trained with Adam at 5× the base LR, no weight decay. Record this in the training config later.

### 5.6 Deliberate deviations from the paper
| Paper | Here | Why |
|---|---|---|
| Tokenizer compression (NFKC/lowercase canonical IDs) | omitted | WikiText-2 scale; not part of the contribution. Possible ablation. |
| Inserted before attention, query = layer hidden state | post-SSM residual, query = caller's choice | FYP design: bypass SSM decay [STABLE] |
| Multi-branch (hyper-connection) variant | single branch | no hyper-connections in this model |
| Optional Δ-modulation pathway into the SSM | not built | **[EXPERIMENTAL]** |

---

## 6. Config  [TARGET]

One dataclass, validated in `__post_init__`. YAML experiment configs that populate it are deferred to the
training-pipeline milestone (pyyaml is not an allowed dependency in this work package).

```python
@dataclass
class MixerConfig:
    d_model: int = 256
    # attention
    attn_head_dim: int = 64          # must equal ssm_d_state (Unified RoPE, A2)
    attn_window: int = 128
    # ssm (Mamba-2 / SSD)
    ssm_expand: int = 2
    ssm_head_dim: int = 64
    ssm_d_state: int = 64
    ssm_conv_kernel: int = 4
    ssm_chunk: int = 64
    # rope
    rope_base: float = 10000.0
    # engram
    vocab_size: int = 50257
    engram_orders: tuple[int, ...] = (2, 3)
    engram_heads_per_order: int = 4
    engram_head_dim: int = 32
    engram_table_size: int = 10007   # first prime; subsequent tables use the next distinct primes
    engram_conv_kernel: int = 4
    engram_seed: int = 1234
```
Validation: `d_model % attn_head_dim == 0`, `(ssm_expand*d_model) % ssm_head_dim == 0`,
`attn_head_dim == ssm_d_state`, head dims even, all orders ≥ 2.

---

## 7. Proposed file layout
(Fresh repository — create this layout in M0. The discarded prototype remains only in the git history.)

```
src/hybrid/
  config.py        MixerConfig
  rope.py          rope_cos_sin, apply_rope, UnifiedRoPE
  attention.py     swa_reference, swa_efficient, SlidingWindowAttention
  ssm.py           segsum, ssd_reference, ssd_efficient, SSDBranch
  engram.py        hash_ngrams_reference/_efficient, lookup_reference/_efficient, Engram
tests/
  conftest.py      seeds, float64 helpers, causality helper
  test_config.py   MixerConfig validation, assert_causal self-check
  test_rope.py  test_attention.py  test_ssm.py  test_engram.py
  test_smoke.py    overfit-one-batch per branch (marked slow)
scripts/
  bench_scaling.py wall-clock & peak memory vs L (CPU) for dense vs SWA vs SSD
docs/
  DEVLOG.md        dated session log (feeds the reflective journal and Chapter 3)
```

---

## 8. Open decisions (do not resolve silently)

| ID | Question | Current default | Who decides |
|---|---|---|---|
| A1 | Mamba-1 vs Mamba-2/SSD | SSD | Student + supervisor, after experiments |
| A2 | `D_h == N` to share one rotation, or share only θ-schedule across different dims | `D_h == N = 64` | Student |
| A3 | Engram query: SSM output vs layer input | interface is agnostic; fusion milestone picks | Fusion milestone |
| A4 | Scalar vs vector Engram gate | scalar (paper) | [EXPERIMENTAL] |
| A5 | Engram tokenizer compression | off | possible ablation |
| A6 | Fusion operator form | not specified here | Journal task 3 |

## 9. References
- A. Gu, T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752.
- T. Dao, A. Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," ICML 2024.
- TransXSSM / Unified RoPE, arXiv:2506.09507 (§2.2, Eqs. 2–5; Appendix A justifies RoPE on the CB term).
- X. Cheng et al., "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models," arXiv:2601.07372.
- I. Beltagy et al., "Longformer," 2020 (sliding-window attention).
