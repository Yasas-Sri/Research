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

## 2026-10-20 — M4 SSD branch
- Built: `src/hybrid/ssm.py` — `segsum` (cumsum differences, −inf above diagonal), `ssd_reference` (loop over t),
  `ssd_efficient` (chunked §4.3: intra-chunk `exp(segsum)`, per-chunk states, plain Python loop over chunks, state → output),
  `SSDBranch` (in_proj → causal depthwise conv + SiLU → split → RoPE on B, C only → softplus dt → SSD + D·x → RMSNorm(y·SiLU(z))
  → out_proj). All tagged [TARGET]. Moved `MaxNumel` recorder from `tests/test_attention.py` to `tests/conftest.py` (shared).
- Spec fix (student-approved, M4-h): ARCHITECTURE §4.4 and AGENTS M4 now read "for `T ∈ {1, 4, 8, 16, 64}`; `L` is chosen so
  that `L` is not a multiple of `T` whenever `1 < T < L`".
- Tests: 98/98 (`pytest -q`); `tests/test_ssm.py` 27/27, `tests/test_rope.py` 6/6 (after the post-review additions below). Invariant → test (ARCHITECTURE §4.4, plus §2.4.3–4):
  - efficient == reference, atol 1e-9, T ∈ {1, 4, 8, 16, 64}, L ∈ {37, 100} → `test_ssd_efficient_matches_reference`
    (measured max error 1.8e-15 … 8.9e-15 at L = 100; inputs use weak decay so inter-chunk state matters at 1e-9)
  - chunk-size invariance T=8 vs T=32 → `test_ssd_chunk_size_invariance`
  - causality (function: x, dt, B, C; module: x incl. conv), exact `torch.equal` → `test_ssd_efficient_causal`, `test_ssd_module_causal`
  - decay sanity at init: A ∈ [−16, −1], softplus(dt_bias) ∈ [1e-3, 1e-1], D == 1, per-token exp(dt·A) ∈ (0, 1) → `test_ssd_init_decay_in_unit_interval`
  - float32 stability under strong decay (chunk sums of dt·A down to ~−1600, L = 200, T = 64): finite and within atol 1e-3 of
    float64 `ssd_reference` → `test_ssd_efficient_float32_strong_decay_stable`
  - shift invariance `positions + 1000` (§2.4.3) → `test_ssd_module_shift_invariant`
  - one `UnifiedRoPE(cos, sin)` shared by SWA and SSD, both shift-invariant under it (A2) → `test_unified_rope_shared_by_both_branches`
  - RoPE-on ≠ RoPE-off (§2.4.4), unsorted positions up to 1000, D zeroed in the test, atol 1e-9 → `test_ssd_rope_changes_output`
  - shape `(B, L, d)`, finite, every element of grads non-zero on A_log, dt_bias, D, conv weight/bias, in_proj, out_proj, norm
    → `test_ssd_module_shape_finite_and_gradients`
  - module == explicit composition (hand-written causal conv, RoPE on B/C, `ssd_reference`, gate then norm) at unsorted
    positions → `test_ssd_module_matches_reference_composition`
  - no (L, L) tensor → `test_ssd_efficient_no_LxL_allocation` (B=H=P=N=1): chunk=L negative control 1 048 576 (= L²);
    T=16: L=1024 → 16 384, L=4096 → 65 536 (exactly 4×).
- Mutation check (scratch copies, full suite rerun after the post-review additions; all 16 caught) — which tests fail:
  - segsum off-by-one (includes a_j) → matches_reference, chunk_invariance, float32_strong_decay, composition
  - segsum upper triangle unmasked → matches_reference, chunk_invariance, both causal tests, float32_strong_decay, composition
  - H_entering one chunk late → matches_reference, chunk_invariance, both causal tests, float32_strong_decay, composition
  - chunk decay missing in scan → matches_reference, chunk_invariance, float32_strong_decay, composition
  - dt not applied to x → matches_reference, float32_strong_decay, composition
  - D·x skip missing → matches_reference, float32_strong_decay, composition, shape_finite_and_gradients (D gets no grad)
  - softplus missing on dt → module_causal, shift_invariant, unified_rope_shared, composition, shape_finite (negative dt → growth → non-finite)
  - conv not causal (right pad) → module_causal, composition
  - RoPE on x as well / RoPE skipped on C / RoPE before conv → shift_invariant, unified_rope_shared, composition (each)
  - RoPE skipped on both B and C (added) → rope_changes_output, composition
  - norm before gate → composition only
  - forced chunk = L (dense) → no_LxL_allocation only
  - decays via cumprod: `a_cs = log(cumprod(exp(a)))` / `Lmat` as a ratio of `cumprod(exp(a))` → float32_strong_decay only (each)
- Decisions (accepted by student in plan review):
  - M4-a: `segsum` as cumsum differences (§4.3 wording). The masked-cumsum form (Mamba-2 minimal) avoids float32 cancellation; not built.
  - M4-b: `in_proj`, `out_proj` bias-free, conv with bias (Mamba-2 defaults); default `nn.Linear` init.
  - M4-c: depthwise conv over `d_in + 2N` channels, kernel `cfg.ssm_conv_kernel`, causal left-pad k−1.
  - M4-d: gated norm `nn.RMSNorm(d_in)(y·SiLU(z))` (Mamba-2 `norm_before_gate=False`).
  - M4-e: "A_log, dt_bias, D excluded from weight decay" is an optimizer setting — deferred to the training milestone, no code.
  - M4-f: signatures per §4.2/§4.3: `ssd_reference` has no `chunk` arg; `ssd_efficient` does.
  - M4-g: `MaxNumel` shared via `conftest.py`.
  - §8: A1 default (SSD) used, not resolved; A2 (`D_h == N`) enforced by config.
  - Post-review (student): float32 stability tests compare against the float64 reference at atol 1e-3; added to AGENTS §4
    rule 6. Measured max error of the current code in `test_ssd_efficient_float32_strong_decay_stable` (seed 0): 2.8e-5.
  - Post-review finding: at init the D·x skip dominates the SSD branch output. Small init dt (1e-3 … 1e-1) keeps the C·B
    term small, so RoPE (which acts only on C·B) changed the output by ‖on−off‖/‖out‖ = 0.40% (8.37e-2 / 20.69; 0.65% with
    unsorted positions). With D = 0 the same measure is 28% (arange) / 54% (unsorted). `test_ssd_rope_changes_output` now
    zeroes D and uses unsorted positions up to 1000, keeping atol 1e-9.
- Problems & fixes:
  - Cumprod gap (found in post-review): building decays as products — `log(cumprod(exp(a)))` for all decays, or `Lmat` as a
    ratio of `cumprod(exp(a))` — passed all 96 tests, because every test used either float64 or weak init decay, where
    products don't underflow. Closed with `test_ssd_efficient_float32_strong_decay_stable` (float32, A ∈ {−1, −8, −16},
    dt = softplus(randn + 1)); both cumprod copies now give NaN and fail it. Also added the missing D == 1 init check and
    `test_unified_rope_shared_by_both_branches` (no test previously fed one UnifiedRoPE output to both branches).
  - `test_ssd_rope_changes_output` first used an arbitrary threshold (diff > 1e-3·max|out|); at init the
    measured diff was 4.2e-3 vs max|out| 1.86 — only ~2× margin, because small init dt makes the D·x skip dominate. Replaced
    (while passing) with "not allclose at the SSD equivalence tolerance atol 1e-9" — the same tolerance that defines equality.
- Deviations from ARCHITECTURE.md: none (the §4.4 wording fix above is a spec correction).
- Next: M5 — smoke tests and scaling check.

## 2026-10-20 — M5 Smoke tests and scaling check
- Built: `tests/test_smoke.py` — throwaway test-only wrapper `Embedding → x + branch(RMSNorm(x)) → RMSNorm → LM head`
  (Engram: `x + Engram(ids, query=x)`, query = raw embedding), not the fusion layer; `test_smoke_batch_needs_context` (fast)
  and `test_smoke_{swa,ssd,engram}_overfits` (`@pytest.mark.slow`). `scripts/bench_scaling.py` (torch + stdlib only) →
  `results/scaling.csv` (with footer notes) and `results/scaling_env.json` (environment + exponents).
- Tests: `pytest -q` 99/99 (3 slow deselected); `pytest -q -m slow` 3/3 (~2 s).
  Smoke (seed 0, B=2, L=64, vocab 16, AdamW lr 3e-3, ≤300 steps, stop at loss < 0.5): SWA 0.465 after 8 updates,
  SSD 0.223 after 4, Engram 0.411 after 19.
- Mutation check (scratch copies, `-m slow`):
  - branch output zeroed in the wrapper → all three smoke tests fail (loss stays > 1)
  - SWA window forced to 1 (self only) → `test_smoke_swa_overfits`
  - Engram returns zero → `test_smoke_engram_overfits`
  - SSD state dropped (y = D·x only) → **not caught**: the SSD conv (kernel 4) alone sees 3 tokens, and the batch's
    3-token-context floor is 0.011, so the conv can memorise it. The smoke test shows the SSD branch can use context, not
    that the recurrence is what does it; the recurrence is covered by the M4 equivalence/composition tests. No change made.
- Scaling (`results/scaling.csv`; i7-12700H, 4 threads, torch 2.14.1+cpu, Python 3.14.7, float32, B=1, no_grad,
  3 warm-up + median of 10, fresh subprocess per (method, L), 3 full sweeps). Median ms (median of per-sweep medians):

  | L | dense_swa (w=L) | dense_sdpa | swa (w=128) | ssd (T=64) |
  |---|---|---|---|---|
  | 256 | 2.06 | 0.65 | 1.63 | 2.15 |
  | 512 | 8.70 | 1.41 | 2.91 | 3.41 |
  | 1024 | 40.6 | 2.95 | 5.82 | 6.79 |
  | 2048 | 169 | 9.92 | 12.1 | 13.8 |
  | 4096 | 681 | 47.4 | 25.7 | 24.6 |

  Scaling exponent (least-squares slope of log time vs log L over L ∈ {1024, 2048, 4096}), median [min–max] over 3 sweeps:
  dense_swa 2.04 [2.03–2.04] · dense_sdpa 2.00 [1.98–2.01] · swa 1.11 [1.07–1.13] · ssd 0.93 [0.89–0.96].
  Dense ≈ 2 and SWA/SSD ≈ 1 as expected. SSD's < 1 means fixed per-call overhead (Python chunk loop, small einsums) is still
  a visible share at L = 1024; SWA's > 1 is unexplained noise/cache effects at this size. Small-L times are overhead-dominated.
  Memory at L = 4096 (sweep 0), largest tensor / peak RSS rise: dense_swa 512 MiB / 1620 MiB (the (1, 1, 4, 4096, 8192)
  score tensor, ~3 live copies); dense_sdpa 4 MiB / 8.8 MiB (fused CPU kernel, tiled — O(L) memory, O(L²) time);
  swa 16 MiB / 86 MiB; ssd 8 MiB / 101 MiB. Largest tensor grows ×4 per doubling for dense_swa, ×2 for the rest.
- Decisions:
  - Smoke vocab 128 → 16 (student): with vocab 128 the batch's exact no-context loss floor (entropy of next token given
    current token) was 0.576 / 0.517 / 0.518 for seeds 0/1/2 — too close to 0.5 to show the branch uses context. With
    vocab 16 it is 1.848 / 1.759 / 1.862 (2-token context 0.270 / 0.337 / 0.244; 3-token 0.011 / 0.033 / 0.011).
    Threshold 0.5 kept; AGENTS M5 updated to `vocab=16`. `test_smoke_batch_needs_context` asserts floor ≥ 1.0.
  - M5-a: task = next-token CE, inputs `ids[:, :-1]`, targets `ids[:, 1:]`.
  - M5-b: smoke config = defaults except `vocab_size=16`, `attn_window=16` (default 128 > L would make SWA dense),
    `ssm_chunk=16` (default 64 = L would skip the inter-chunk scan).
  - M5-c (student): benchmark the kernels only. Projections (W_q/W_k/W_v/W_o, in_proj/out_proj, conv) are excluded because
    they add the same O(L·d²) term to every method and would hide the dense O(L²) exponent at small L. Stated in the CSV
    footer. `dense_sdpa` (`F.scaled_dot_product_attention(is_causal=True)`) added as a second dense column because
    `swa_efficient(w=L)` computes (L, 2L) scores, half masked; absolute times compare implementations of different kinds,
    so the exponents are the comparable quantity.
  - M5-d: 4 threads (inside the 6 P-cores; no core pinning — E-core scheduling and turbo add noise), 3 warm-up, 10 repeats,
    3 sweeps, 2 GiB memory budget. Before each run the peak is estimated (measured RSS rise at L/2 × 4 for dense, × 2 for
    linear; dense also has an analytic floor of 3 float32 score tensors); over budget → `skipped (est. X GiB)`. At L = 4096
    dense_swa was estimated at 1.68 GiB and ran (measured 1.58 GiB). The skip path was exercised separately with a 0.05 GiB
    budget in a scratch root.
  - Memory columns: `max_numel` (dispatch recorder from `conftest.py`, largest single aten output — not total live
    memory, blind to fused-kernel internals) and `peak_rss_delta_mib` (rise of `ru_maxrss` over one forward in a fresh
    process — includes allocator caching, ~MiB noise at small L).
  - M5-e (student): `results/` created by the script; `.gitignore` untouched; committing is the student's call.
  - M5-f: Engram not benchmarked (AGENTS M5 lists dense, SWA, SSD only).
  - §8: A3 not resolved (the embedding-as-query choice is test-only, per AGENTS M5); A6 untouched.
- Problems & fixes: the first sweep never hit the budget, so the skip path was tested separately; that exposed (1) a
  missing guard when fewer than 2 L values ran (exponent now `n/a`), (2) after a skip, the next L's estimate extrapolated
  from the last *measured* run (underestimate) — now from the skipped estimate, (3) printout column overlap. Fixed and the
  full 3-sweep run repeated; the numbers above are from the final script.
- Deviations from ARCHITECTURE.md: none.
- Invariant → test map (AGENTS §6 definition of done):

  | Spec | Invariant | Test(s) |
  |---|---|---|
  | §2.4.1 | RoPE norm preservation | `test_rope_preserves_norm` |
  | §2.4.2 | ⟨R_m q, R_n k⟩ invariant to a common shift | `test_rope_relative_property` |
  | §2.4.3 | branch-level shift invariance (`positions + 1000`) | `test_swa_module_shift_invariant`, `test_ssd_module_shift_invariant`, `test_unified_rope_shared_by_both_branches` |
  | §2.4.4 | RoPE changes the SSD output | `test_ssd_rope_changes_output` |
  | §3.4 | `swa_efficient == swa_reference`, w ∈ {1, 4, 8, L, 2L} | `test_swa_efficient_matches_reference`, `test_swa_dense_baseline_matches_reference` |
  | §3.4 | causality | `test_swa_efficient_causal`, `test_swa_module_causal` |
  | §3.4 | gradient of `out[:, t]` w.r.t. v only on [t−w+1, t] | `test_swa_window_gradient_support` |
  | §3.4 | shift invariance | `test_swa_module_shift_invariant` |
  | §4.4 | `ssd_efficient == ssd_reference`, T ∈ {1, 4, 8, 16, 64} | `test_ssd_efficient_matches_reference` |
  | §4.4 | chunk-size invariance T=8 vs T=32 | `test_ssd_chunk_size_invariance` |
  | §4.4 | causality | `test_ssd_efficient_causal`, `test_ssd_module_causal` |
  | §4.4 | exp(Δ·A) ∈ (0, 1) at init (+ A_log, dt_bias, D init) | `test_ssd_init_decay_in_unit_interval` |
  | §4.4 | shift invariance; RoPE-on ≠ RoPE-off | `test_ssd_module_shift_invariant`; `test_ssd_rope_changes_output` |
  | §4.4 | shape, finite, gradients reach A_log, dt_bias, D, conv, projections | `test_ssd_module_shape_finite_and_gradients` |
  | §5.4 | hash determinism (calls + state_dict round-trip) | `test_hash_deterministic_across_calls_and_state_dict` |
  | §5.4 | hash locality | `test_hash_locality` |
  | §5.4 | index range; tables never overlap | `test_index_range_and_tables_disjoint` |
  | §5.4 | reference == efficient (indices exact, embeddings allclose) | `test_hash_reference_equals_efficient`, `test_lookup_reference_equals_efficient` |
  | §5.4 | module causality (input_ids and query) | `test_engram_causal` |
  | §5.4 | 0 < α < 1 | `test_gate_in_open_unit_interval` |
  | §5.4 | zero-init conv ⇒ Y == Ṽ | `test_zero_init_conv_output_equals_gated_value` |
  | §5.4 | gradients only on looked-up embedding rows | `test_grad_only_on_looked_up_rows` |

  Supporting (not spec invariants): no (L, L) allocation → `test_swa_efficient_no_LxL_allocation`,
  `test_ssd_efficient_no_LxL_allocation`; module == explicit composition → `test_swa_module_matches_reference_composition`,
  `test_ssd_module_matches_reference_composition`; float32 stability → `test_ssd_efficient_float32_strong_decay_stable`;
  RoPE identity/explicit matrix → `test_rope_zero_position_is_identity`, `test_rope_matches_explicit_rotation_matrix`;
  shape/finite → `test_swa_module_shape_and_finite`, `test_engram_output_shape_and_finite`; config → `test_config.py`.
- Next: work package (AGENTS §2) complete; the fusion operator (Journal task 3) is the next package — not started.
