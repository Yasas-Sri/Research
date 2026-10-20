# CLAUDE.md

Entry point for Claude Code. The project rules live in the two imported files; this file adds the
behavioural guidelines and the Claude Code session workflow.

@AGENTS.md
@ARCHITECTURE.md

Precedence: `ARCHITECTURE.md` decides **what** to build · `AGENTS.md` decides **how** · this file decides
**how a session runs**. If they conflict, stop and ask.

---

## 1. Behavioural guidelines

These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### Think before coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity first
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked. No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical changes
**Touch only what you must. Clean up only your own mess.**
- Don't "improve" adjacent code, comments or formatting. Don't refactor what isn't broken.
- Match existing style, even if you'd do it differently.
- Mention unrelated dead code; don't delete it.
- Remove imports/variables/functions that *your* change made unused — nothing else.
- Test: every changed line traces directly to the request.

### Goal-driven execution
**Define success criteria. Loop until verified.**
- "Add validation" → write tests for invalid inputs, then make them pass.
- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Refactor X" → tests pass before and after.
- For multi-step tasks, state a plan as `[step] → verify: [check]` before starting.

---

## 2. Session workflow

**One milestone (AGENTS.md §5) per session.** Do not start the next milestone in the same session unless asked.

**Start of session**
1. Read the last entry of `docs/DEVLOG.md` and run `pytest -q` to see the current state.
   First session only: neither exists yet — start with M0, which creates them.
2. Say which milestone you are on and list its steps as `[step] → verify: [check]`.
3. For M1 (Engram) and M4 (SSD), use plan mode and wait for approval before editing.
4. List any assumptions or open decisions (ARCHITECTURE §8) the milestone touches.

**During**
- Write the invariant test first, run it, see it fail, then implement.
- Run the milestone's verify command after every meaningful change, not only at the end.
- If a test fails twice for the same reason, stop and report the cause instead of patching around it.

**End of session**
1. Run `pytest -q` (and `pytest -q -m slow` from M5 on). Report pass/total.
2. Append the DEVLOG entry (AGENTS.md §8).
3. Summarise in ≤5 lines: what changed, what's verified, what's open.
4. Stop there. Git is handled by the user.

Git is the user's job. Never run git commands that change the repository or the remote: no add, commit,
push, tag, branch, checkout, merge, rebase, reset, stash, and no PRs. Read-only commands (status, diff,
log) are fine when useful. Don't ask to commit and don't write commit messages unless the user asks.

---

## 3. Commands

All commands run inside the project virtual environment `.venv` (activated before Claude Code starts).
Never `pip install` outside it, and never add a package without asking. `.venv/` is git-ignored.
Dependencies are pinned in `requirements.txt` (torch CPU build + pytest only).

```bash
pytest -q                          # fast suite (slow tests excluded)
pytest -q tests/test_engram.py     # one module
pytest -q -m slow                  # overfit smoke tests (M5)
python scripts/bench_scaling.py    # CPU scaling table → results/scaling.csv (M5)
```

`pyproject.toml` needs (add in M0 if missing):

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["slow: overfit smoke tests"]
addopts = "-m 'not slow'"
```

---

## 4. Stability tags in code

Mark the top of a function or class with a comment:

```python
# [STABLE] verified against ssd_reference; do not change without a recorded decision
def ssd_efficient(...):
```

- `[STABLE]` — never edit without an explicit instruction.
- `[TARGET]` — what you are building now.
- `[EXPERIMENTAL]` — never delete; don't extend unless asked.

A function becomes `[STABLE]` only after its equivalence and invariant tests pass **and** the user agrees.

---

## 5. Mistakes to avoid in this repo

- Importing `mamba_ssm`, `flash_attn`, `einops` or anything CUDA-only "just for reference".
- Building an `(L, L)` mask or score tensor in `swa_efficient` or `ssd_efficient`.
  `F.scaled_dot_product_attention(is_causal=True)` is dense — fine for a dense baseline, not for SWA.
- Computing SSD decays with `cumprod`; use `exp` of cumulative sums of `dt·A` (see `segsum`).
- Applying RoPE to SSM `x`, `z` or `dt`, or before the conv. RoPE goes on `B` and `C` only, after conv + SiLU.
- Applying RoPE inside Engram.
- Window off-by-one: `w` keys **including** self, i.e. `t − w < s ≤ t`.
- Casting Engram hash values to float, or using Python `hash()` (randomised per process). Use `int64` and seeded multipliers.
- Adding the Engram output to the residual inside the module — the caller does that.
- Creating multiplier/offset tensors as `nn.Parameter` instead of buffers.
- Loosening a test tolerance to make it pass.
- Building anything on the AGENTS.md "Not in this work package" list.