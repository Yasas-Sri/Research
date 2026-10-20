# DEVLOG

## 2026-10-19 — M0 Scaffold
- Built: `pyproject.toml` (pytest config); layout from ARCHITECTURE §7 with empty `src/hybrid/{__init__,rope,attention,ssm,engram}.py`,
  `tests/test_{rope,attention,ssm,engram,smoke}.py`, `scripts/bench_scaling.py`; `src/hybrid/config.py` (`MixerConfig` + validation);
  `tests/conftest.py` (`seed`, opt-in `float64` fixture, `assert_causal`); `tests/test_config.py`; `.gitignore` += `__pycache__/`, `.pytest_cache/`.
- Fresh start: the earlier Engram prototype (`mamba3MEM.py`) was removed from the working tree; it remains in the git history.
  Reason: it did not match the written specification (ARCHITECTURE.md), so this work package rebuilds from the spec.
- Tests: 10/10 (`pytest -q`); new tests: `test_default_config_is_valid`, `test_config_rejects_invalid` ×5
  (D_h≠N, d_model%D_h, d_in%P, odd RoPE dim, order<2), `test_assert_causal_accepts_causal_fn`, `test_assert_causal_rejects_acausal_fn` ×3.
- Decisions (student):
  - `assert_causal(fn, inputs, t0)`: `inputs` are data tensors only, time on dim 1; non-data args (cos/sin, positions) are bound by a
    closure and never perturbed. Clones are perturbed (float: +randn; int: `(x+1) % (x.max()+1)`). Exact `torch.equal`; if it ever
    fails on a masked 0·x term, stop and report, do not loosen.
  - `float64` fixture is opt-in; `seed` = `torch.manual_seed(0)`; config errors raise `ValueError`.
  - Evenness checked on `attn_head_dim` only (covers `ssm_d_state` via A2 equality).
  - No YAML loader: pyyaml not allowed; YAML configs deferred to the training-pipeline milestone (ARCHITECTURE §6 updated).
  - No primality check on `engram_table_size`; M1 starts table sizes from the next prime ≥ it.
  - `tests/test_config.py` added to ARCHITECTURE §7 as a spec gap (M0 verify needs a config test).
- Problems & fixes: torch warns `Failed to initialize NumPy` on import — harmless, numpy not a dependency, left as is.
- Deviations from ARCHITECTURE.md: none.
- Next: M1 — Engram (plan mode first).

## 2026-10-19 — M1 Engram
- Built: `src/hybrid/engram.py` — `next_primes`, `hash_ngrams_reference`, `ngram_xor_hash` (pre-mod step), `hash_ngrams_efficient`,
  `lookup_reference`, `lookup_efficient`, `Engram` (`gated_value` exposes `(α, Ṽ)`; `forward` returns `Y`, no residual add). All tagged [TARGET].
- Tests: 29/29 (`pytest -q`); `tests/test_engram.py` 19/19. Invariant → test (ARCHITECTURE §5.4):
  - determinism across calls + `state_dict` save/load; constants are buffers → `test_hash_deterministic_across_calls_and_state_dict`
  - hash locality (pre-mod hash changes for all heads, `z` for ≥1 of K heads) → `test_hash_locality`
  - index range / tables disjoint / ascending distinct primes → `test_index_range_and_tables_disjoint`
  - ref == efficient: indices `torch.equal` → `test_hash_reference_equals_efficient` (L ∈ {1,2,5,17} × small/default cfg);
    embeddings allclose → `test_lookup_reference_equals_efficient`
  - module causality (ids + query, random conv weights) → `test_engram_causal` (t0 ∈ {1,5,11})
  - `0 < α < 1` → `test_gate_in_open_unit_interval`
  - zero-init conv ⇒ `Y == Ṽ` exactly → `test_zero_init_conv_output_equals_gated_value`
  - gradients only on looked-up rows → `test_grad_only_on_looked_up_rows`
  - output shape + finite (AGENTS M1) → `test_engram_output_shape_and_finite`
  - Mutation check (scratch copies, not committed): acausal conv pad, lookup without offsets, efficient hash dropping
    `x_{t−j}`, reference reading an extra token — each is caught by the intended test(s).
- Decisions (accepted by student in plan review):
  - No bias on the conv, `W_K`, `W_V` (keeps zero-init ⇒ `Y == Ṽ` exact).
  - `nn.RMSNorm` defaults (default eps, learnable weight) for query, key and pre-conv norms.
  - `d` in `√d` is `d_model`.
  - Single zero-padded multiplier buffer `(|𝒩|, K, n_max)`; slots `j ≥ n` are 0 and never read.
  - Conv kernel from `cfg.engram_conv_kernel`, dilation `max(cfg.engram_orders)`, left-pad `(kernel−1)·dilation`.
  - Table sizes start at the next prime ≥ `engram_table_size` (M0 decision); default cfg → 10007 … 10079, 2.70 M params.
  - Locality test asserts ≥1 of K heads changes after `% M` (not all — per-head collisions are ~1/M); table not enlarged.
  - §8 defaults used, none resolved: A3 (query taken as given, not projected), A4 (scalar gate), A5 (no tokenizer compression).
- Problems & fixes: `.venv` was not activated in this session; ran `.venv/bin/pytest` directly. No test failures beyond the intended first-run ImportError.
- Deviations from ARCHITECTURE.md: none.
- Next: M2 — Unified RoPE.

## 2026-10-19 — M2 Unified RoPE
- Built: `src/hybrid/rope.py` — `rope_cos_sin`, `apply_rope` (rotate-half), `UnifiedRoPE(dim, base)`. All tagged [TARGET].
- Tests: 34/34 (`pytest -q`); `tests/test_rope.py` 5/5. Invariant → test (ARCHITECTURE §2.4; 3 and 4 are tested in M3/M4):
  - 1 norm preservation (positions up to 12345) → `test_rope_preserves_norm`
  - 2 relative property, shifts 1 and 1000, and `⟨R_m q, R_n k⟩ == ⟨q, R_{n−m} k⟩` → `test_rope_relative_property`
  - `positions = 0` is identity (`torch.equal`) → `test_rope_zero_position_is_identity`
  - §2.1/§2.2 convention: equals an explicit block rotation built with `math.cos/sin` in rotate-half layout, at unsorted
    non-contiguous positions `[0, 7, 3, 1000, 2]`; `UnifiedRoPE` output shapes → `test_rope_matches_explicit_rotation_matrix`
  - Mutation check (scratch copies): sin sign flipped → explicit-matrix test only; θ exponent −i/D → explicit-matrix only;
    interleaved convention (valid rotation, wrong layout) → explicit-matrix only; input split interleaved but output
    rotate-half (scrambled) → relative, identity, explicit-matrix; internal `arange` ignoring `positions` → relative,
    identity, explicit-matrix. Norm test catches none of these by design (all are norm-preserving).
- Decisions (accepted by student in plan review):
  - R1: `rope_cos_sin` always computes angles in float64 (§2.3 reference); `apply_rope` casts cos/sin to `x.dtype`, since
    `UnifiedRoPE.forward(positions)` cannot know the activation dtype.
  - R2: `UnifiedRoPE` has no parameters or buffers; stores `dim`, `base`; `arange` built on `positions.device`.
  - R3: no extra input checks in `apply_rope`/`UnifiedRoPE`; evenness and `D_h == N` are enforced by `MixerConfig`.
  - Explicit-matrix test added because invariants 1–2 alone pass for a wrong sign, θ schedule or pairing convention.
  - §8: A2 default (`D_h == N`) used, not resolved.
- Problems & fixes: `python` resolved to `/usr/bin/python` (venv not active); used `.venv/bin/python` / `.venv/bin/pytest`.
- Deviations from ARCHITECTURE.md: none.
- Next: M3 — sliding-window attention (awaiting "go").

## 2026-10-20 — M3 Sliding-window attention
- Built: `src/hybrid/attention.py` — `swa_reference` (full band mask, test-only), `swa_efficient` (blocked `[prev ‖ current]`,
  mask `i < j ≤ i+w` plus key-position `s ≥ 0` to mask block 0's zero "previous"), `SlidingWindowAttention` (bias-free
  `W_q, W_k, W_v, W_o`; RoPE on Q, K only). All tagged [TARGET].
- Spec fix (student-approved): ARCHITECTURE §3.4 and AGENTS M3 now read "for `w ∈ {1, 4, 8, L, 2L}`; `L` is chosen so that `L` is
  not a multiple of `w` whenever `1 < w < L`" (the old "L not a multiple of w" is impossible for w = 1 and w = L).
- Tests: 70/70 (`pytest -q`); `tests/test_attention.py` 36/36. Invariant → test (ARCHITECTURE §3.4, plus §2.4.3):
  - efficient == reference, atol 1e-10, w ∈ {1, 4, 8, L, 2L}, L ∈ {13, 30} → `test_swa_efficient_matches_reference`
  - dense baseline w ∈ {L, L+1, 2L} == reference and == `F.scaled_dot_product_attention(is_causal=True)` (test-only oracle)
    → `test_swa_dense_baseline_matches_reference`
  - causality (function: q, k, v; module: x), exact `torch.equal` → `test_swa_efficient_causal`, `test_swa_module_causal`
  - window: grad of `out[:, t]` w.r.t. `v` non-zero exactly on `[max(0, t−w+1), t]`, every t → `test_swa_window_gradient_support`
  - shift invariance `positions + 1000` (§2.4.3) → `test_swa_module_shift_invariant`
  - module == `W_o(swa_reference(rope(xW_q), rope(xW_k), xW_v))` at unsorted positions → `test_swa_module_matches_reference_composition`
  - module shape `(B, L, d)` + finite (default cfg, L = 200) → `test_swa_module_shape_and_finite`
  - no (L, L) tensor → `test_swa_efficient_no_LxL_allocation`: a `TorchDispatchMode` records the largest output of every aten op
    (incl. einsum/softmax intermediates). B=H=1, D=2, w=16: reference L=1024 → 1 048 576 (= L², negative control fires);
    efficient L=1024 → 32 768 (= 2Lw), L=4096 → 131 072 (exactly 4×). Dense baseline w=L=1024 → 2 097 152 (= 2L², expected).
- Mutation check (scratch copies) — which tests fail:
  - window w+1 keys (`j ≥ i`) → matches_reference, window_gradient_support, composition
  - window w−1 keys (`j < i+w`) → all but one test (w=1 leaves rows with no key → NaN)
  - causal bound dropped (`j ≤ i+w` removed) → matches_reference, dense_baseline, both causal tests, window_gradient, composition
  - block-0 "previous" padding unmasked (shape-preserving) → matches_reference (all w > 1), dense_baseline, composition only;
    causality and gradient-support tests cannot see it (constant zero keys/values only dilute the softmax)
  - prev block not shifted for K → matches_reference, both causal tests, composition
  - missing `1/√D` → matches_reference, dense_baseline, composition
  - RoPE applied to V → shift_invariant, composition
  - RoPE skipped on K → shift_invariant, composition
  - efficient replaced by dense `(L, L)` path → no_LxL_allocation only
  - S1 clamp removed → all pass (confirms the clamp is exact; it only avoids waste)
  - First block-0 mutation attempt (deleting the mask term) also changed the mask shape → discarded as unclean, redone shape-preserving.
- Decisions (accepted by student in plan review):
  - S1: `swa_efficient` clamps `w = min(w, L)` (exact; avoids `(2L, 4L)` scores at w = 2L). Dense baseline w ≥ L keeps the §3.3
    layout: `(L, 2L)` scores, half masked.
  - S2: `1/√D_h` scaling inside both `swa_*` functions.
  - S3: `H_a = d_model // attn_head_dim`, window `cfg.attn_window`; default `nn.Linear` init (spec silent).
  - S4: the `arange` building the static mask is unrelated to RoPE positions (AGENTS §4.3 not affected).
  - No-(L, L) proof uses `torch.utils._python_dispatch.TorchDispatchMode` (internal module; acceptable with torch pinned 2.14.1).
  - §8: A2 default used via config; A6 (fusion) not built.
- Problems & fixes: block 0's "previous" is not excluded by the static `i < j ≤ i+w` mask alone (j ∈ (i, w) is allowed) →
  added the key-position term `(b−1)w + j ≥ 0`, giving a `(nb, w, 2w)` mask (O(L·2w)).
- Deviations from ARCHITECTURE.md: none (the §3.4 wording fix above is a spec correction).
- Open: AGENTS M4 test list says "`L % T != 0`" for `T ∈ {1, 4, 8, 16, 64}` — same impossibility for T = 1; not changed.
- Next: M4 — SSD branch (plan mode first).
