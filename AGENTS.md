# AGENTS.md

Instructions for any coding agent (Claude Code, Codex, etc.) working in this repository.
**Read this file, then `ARCHITECTURE.md`, then `CLAUDE.md`, before writing anything.**
If this file and `ARCHITECTURE.md` disagree, `ARCHITECTURE.md` wins on *what* to build; this file wins on *how*.

---

## 1. Project in one paragraph

Undergraduate final-year research project (Faculty of Computing, University of Sri Jayewardenepura).
Goal: reduce the O(L²) cost of self-attention with an **intra-layer** hybrid that fuses sliding-window attention
(O(L·w)) and a selective state-space model (Mamba-2/SSD, O(L)), plus an Engram N-gram memory (O(1) lookup)
placed as a post-SSM gated residual. A shared **Unified RoPE** rotates attention Q/K and SSM C/B.
Language modelling (WikiText-2) and Long Range Arena are the *evaluation*, not the contribution.
The architecture is a research **output**: build components so they can be ablated, not so they look final.

## 2. Current work package (Reflective Journal 2)

| # | Task | Done when |
|---|---|---|
| 1 | Implement the Engram module from scratch (reference + efficient paths) with unit tests | every Engram invariant in ARCHITECTURE §5.4 has a passing test |
| 2 | Implement SWA and SSD branches with Unified RoPE | every invariant in ARCHITECTURE §2.4, §3.4, §4.4 has a passing test |

**Not in this work package — do not build:** the fusion operator, a full language model, the training pipeline,
data loading, Mamba-1, the Engram Δ-modulation pathway, tokenizer compression, KV-cache / recurrent decoding,
MoE, custom kernels. If a task seems to need one of these, stop and ask.

---

## 3. Behavioural rules

### 3.1 Think before coding
- State assumptions before implementing. If a spec line in `ARCHITECTURE.md` is ambiguous, name the ambiguity and ask.
- Anything listed in ARCHITECTURE §8 (Open decisions) is **not yours to decide**. Use the stated default and say so.
- If you see a simpler correct approach than the spec, propose it — don't silently substitute it.

### 3.2 Simplicity first
- Minimum code that satisfies the spec and its tests. No speculative options, registries, factories or plugin systems.
- No runtime flag to switch reference/efficient paths: two plain functions, `forward` calls the efficient one,
  tests call both.
- Dependencies: `torch` and `pytest` only. No `einops`, `mamba_ssm`, `flash_attn`, `triton`, `causal_conv1d`.
- No error handling for impossible states. Config validation (ARCHITECTURE §6) is the only input checking.

### 3.3 Surgical changes
- **Fresh start.** This work package starts from an empty `src/` and `tests/`. An earlier Engram prototype was
  discarded (it remains in the git history). Do not read, port or copy code from it; build from `ARCHITECTURE.md`.
- Never edit code tagged **[STABLE]** without an explicit instruction. Never delete **[EXPERIMENTAL]** code.
- Match existing style. Don't reformat or "tidy" files you weren't asked to touch.
- Remove only the imports/helpers that your own change orphaned. Mention other dead code; don't delete it.

### 3.4 Goal-driven execution
Every milestone below has a verification command. A milestone is complete only when that command passes.
Write the test first when the spec gives an invariant; then make it pass.

---

## 4. Hard technical rules

1. **CPU-first.** All code and all tests run on a laptop CPU with no GPU. `device` is a parameter, never hard-coded `"cuda"`.
2. **Shapes.** Branch I/O is `(B, L, d)`; per-head tensors `(B, L, H, D)`. Every function docstring lists input and output shapes.
3. **Positions are explicit.** Branches receive `cos, sin` built from a `positions` tensor; nothing inside a branch calls `torch.arange(L)` for RoPE.
4. **One RoPE implementation** (`rope.py`) used by both branches. Assert `attn_head_dim == ssm_d_state`.
5. **Determinism.** Engram hash multipliers are seeded and registered as buffers. Tests set `torch.manual_seed`.
6. **Numerics.** Equivalence tests run in `float64` (atol 1e-10 attention, 1e-9 SSD). Shape/causality tests may use `float32`.
   SSD decays are computed as `exp` of cumulative sums of `dt·A ≤ 0` — never as products of many small numbers.
7. **No silent broadcasting surprises.** When a tensor gains or loses a dimension, the line has a shape comment.
8. **Initialisation follows ARCHITECTURE** (Mamba-2 `A_log`/`dt_bias`/`D` init; Engram conv zero-init; embeddings N(0, 0.02)).
9. **Complexity honesty.** No hidden O(L²) tensor in an efficient path. If you need one, stop and ask.

---

## 5. Milestones

Run tests with `pytest -q` from the repo root. Mark the overfit tests `@pytest.mark.slow`; run them with `pytest -q -m slow`.

### M0 — Scaffold
1. Create the layout in ARCHITECTURE §7 (empty modules only), `pyproject.toml` with the pytest settings from
   `CLAUDE.md` §3, and `docs/DEVLOG.md` with a first entry recording the fresh start and why the prototype was archived.
2. Add `src/hybrid/config.py` (`MixerConfig` + validation) and `tests/conftest.py` with:
   a `seed` fixture, a `float64` default-dtype fixture, and a helper
   `assert_causal(fn, inputs, t0)` that perturbs inputs at `t ≥ t0` and checks outputs `< t0` are unchanged.
- **Verify:** `pytest -q` collects and passes; config rejects `attn_head_dim != ssm_d_state`.

### M1 — Engram  → ARCHITECTURE §5
1. `hash_ngrams_reference` / `hash_ngrams_efficient` (§5.1 step 2, §5.2).
2. Packed embedding with offsets; `lookup_reference` / `lookup_efficient`.
3. `Engram(nn.Module)`: concat → `W_K`, `W_V` → RMSNorm gate → `α·v` → RMSNorm → zero-init dilated causal depthwise conv → SiLU → `+ Ṽ`.
4. Tests (`tests/test_engram.py`), one per invariant in §5.4:
   determinism + state_dict round-trip · hash locality · index range / no table overlap ·
   ref == efficient (indices `torch.equal`) · module causality · `0<α<1` · zero-init ⇒ `Y == Ṽ` ·
   gradient only on looked-up embedding rows · output shape and finiteness.
- **Verify:** `pytest -q tests/test_engram.py` passes.

### M2 — Unified RoPE  → ARCHITECTURE §2
1. `rope_cos_sin`, `apply_rope`, `UnifiedRoPE(dim, base)` (module returns `(cos, sin)` for given positions).
2. Tests: norm preservation · relative property `⟨R_m q, R_n k⟩` invariant to a common shift · `positions=0` is identity.
- **Verify:** `pytest -q tests/test_rope.py` passes.

### M3 — Sliding-window attention  → ARCHITECTURE §3
1. `swa_reference`, `swa_efficient` (blocked, `[prev ‖ current]` keys, mask `i < j ≤ i+w`).
2. `SlidingWindowAttention(nn.Module)`: bias-free `W_q, W_k, W_v, W_o`; RoPE on Q, K.
3. Tests: ref == efficient for `w ∈ {1, 4, 8, L, 2L}`; `L` is chosen so that `L` is not a multiple of `w` whenever `1 < w < L` · causality · gradient support of
   `out[:, t]` w.r.t. `v` is exactly `[t−w+1, t]` · shift invariance with `positions + 1000` · module shape/finiteness.
- **Verify:** `pytest -q tests/test_attention.py` passes.

### M4 — SSD branch  → ARCHITECTURE §4
1. `segsum`, `ssd_reference` (loop over t), `ssd_efficient` (chunked; sequential loop over chunks).
2. `SSDBranch(nn.Module)`: `in_proj` → causal depthwise conv on `xBC` → SiLU → split → RoPE on B, C →
   `softplus(dt + dt_bias)` → SSD + `D·x` → gated RMSNorm with `SiLU(z)` → `out_proj`.
3. Tests: ref == efficient for `T ∈ {1, 4, 8, 16, 64}` with `L % T != 0` · `T=8` vs `T=32` equal · causality ·
   `exp(dt·A) ∈ (0,1)` at init · shift invariance with `positions + 1000` · RoPE-on ≠ RoPE-off ·
   gradients reach `A_log`, `dt_bias`, `D`, conv, projections.
- **Verify:** `pytest -q tests/test_ssm.py` passes.

### M5 — Smoke tests and scaling check  [TARGET, after M1–M4]
1. `tests/test_smoke.py` (slow): for each branch, a throwaway wrapper
   `Embedding → x + branch(RMSNorm(x)) → RMSNorm → LM head` overfits one fixed random batch
   (`B=2, L=64, vocab=128`, AdamW lr 3e-3, ≤300 steps) to loss < 0.5. For Engram, the wrapper uses the embedding as `query`.
   The wrapper lives in the test file only — it is **not** the fusion layer.
2. `scripts/bench_scaling.py`: CPU wall-clock and peak memory of one forward pass for
   dense causal attention (`swa_efficient` with `w = L`), SWA (`w = 128`) and SSD at `L ∈ {256, 512, 1024, 2048, 4096}`;
   print a table and save CSV to `results/scaling.csv`. Evidence for Chapter 3, not a benchmark claim.
- **Verify:** `pytest -q -m slow` passes; at the larger L values the CSV shows dense time trending toward ~4× per doubling of L and SWA/SSD toward ~2× (small-L numbers are dominated by overhead; report them honestly either way).

---

## 6. Definition of done (whole work package)

- `pytest -q` and `pytest -q -m slow` both pass on CPU.
- Every invariant in ARCHITECTURE §2.4, §3.4, §4.4, §5.4 maps to a named test (list them in `docs/DEVLOG.md`).
- No efficient path allocates an `(L, L)` tensor.
- `docs/DEVLOG.md` is up to date (see §8).
- Diff touches only files needed for M0–M5.

## 7. Stop and ask when

- A spec line is ambiguous or seems wrong (e.g., a shape that can't work).
- A test can only pass by loosening its tolerance beyond §4.6, or by changing an invariant.
- You need anything from the "Not in this work package" list, a new dependency, or a GPU.
- You'd need to touch code tagged [STABLE].

When asking, give: the question, the options, the trade-off of each, and your recommendation.

## 8. Session log (`docs/DEVLOG.md`)

Append one entry per session — this feeds Reflective Journal 2 and Chapter 3, so keep it factual:

```
## YYYY-MM-DD — <milestone>
- Built: <files / functions>
- Tests: <passed>/<total>; new tests: <names>
- Decisions: <what was decided, by whom, why>  (link to ARCHITECTURE §8 IDs where relevant)
- Problems & fixes: <bug → cause → fix>
- Deviations from ARCHITECTURE.md: <none | list>
- Next: <one line>
```
