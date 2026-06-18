# Handoff: finish `feat/auction-budget-urgency`

**You are picking up work that was completed on an unreliable machine.** The dev box
has a degrading Raptor Lake CPU (i9-14900KF) that crashes/segfaults under heavy parallel
load (RMA pending — see `_crashdump/whea-cpu-evidence.txt` if present). The implementation
is **done and committed**; it crashed during the *final full-suite verification*. Your job
is to re-verify on reliable hardware and finish the branch. **Do not re-do the
implementation.**

---

## TL;DR — what to do

1. `git fetch && git checkout feat/auction-budget-urgency` (it's pushed to `origin`).
2. Run the four gates (commands below). Expect them to pass.
3. There are **two known full-suite failures that are NOT from this branch** — confirm they
   behave as described below, then disregard them for this branch.
4. Run the `superpowers:finishing-a-development-branch` skill and present the user the
   merge/PR options.
5. **Delete `INSTRUCTIONS.md` before any merge/PR to `main`** — it's a handoff note, not
   part of the feature. Do not let it reach `main`.

---

## What this branch is

The auction bid-model tournament's **budget-urgency** feature + a new contestant.
Data-gathering project — **no "winning" strategy is declared** (the decision is September
2026). Read these for full context, in order:

- `CLAUDE.md` (root) — project rules. Key ones: correctness over speed; tests/mypy/ruff are
  gates (fix code, not tests); reference enums never strings; `GsisId` canonical; specs/plans
  reach `main` only via PR.
- `docs/superpowers/specs/2026-06-18-auction-budget-urgency-design.md` — the approved spec
  (§A urgency formula, §B apply to the 7 models, §C `StudsAndDepthBid`, §D wire + Run F).
- `docs/superpowers/plans/2026-06-18-auction-budget-urgency.md` — the 5-task implementation
  plan (full TDD code per task).
- `reports/auction_tournament_validation_2026.md` — the experiment log; **Run F** (the new
  rows + narrative) is what this branch added.

## What was done (all committed)

Branch HEAD should be `7df66f3`. The 5 plan tasks, newest-last:

| Commit    | Task | What |
|-----------|------|------|
| `c013e06` | 1 | `_budget_urgency()` helper + `URGENCY_GAIN = 3.0` in `bid_strategy.py` |
| `dc67587` | 2 | All 7 existing contestants refactored to single-exit `round(base * _budget_urgency(...))` |
| `f22c7a8` | 3 | New `StudsAndDepthBid` contestant (stud premium + fair-value depth + $1 scrubs) |
| `b7a656a` | 4 | Wired `studsdepth` into the eight-model CLI + engine integration tests |
| `7df66f3` | 5 | **Run F** eight-model bake-off recorded in the report; no winner |

Files this branch touches (verify with `git diff --stat main...HEAD`): everything is under
`src/projections/draft/assistant/auction/`, its tests under `tests/test_draft/`, the spec/plan
docs, and `reports/auction_tournament_validation_2026.md`. **It touches no model, backtest,
ingest, or feature-engineering code.** That fact is what lets you dismiss the two failures below.

### Run F result (already in the report — for your awareness, don't re-run)

Run F was executed at **40 seeds** (reduced from the usual 60 because the 60-seed run crashed
on the CPU fault — a hardware event, explicitly *not* a code bug per the plan). Headline:
budget-urgency **compresses the field** by forcing the cash-hoarders to deploy — `inflation`
playoff 0.03→0.12, `marginal` 0.00→0.08, `vorpshare` 0.05→0.13 vs Run E, while the
already-spending `static`/`patient` barely move. `studsdepth` lands mid-pack. `anchors` stays
last. **No winner declared.** You do **not** need to re-run Run F.

---

## The gates (run these at repo root)

Tooling note: invoke via `python -m`. The repo's `pyproject.toml` configures `pytest-xdist`,
so **don't pass `-p no:xdist`** (it errors on the `-n` arg); use `-n0` if you want serial.

```
python -m pytest -q                       # full suite
python -m mypy src tests                  # strict; expect "Success: no issues found"
python -m ruff check src tests            # expect "All checks passed!"
python -m ruff format --check src tests   # expect "already formatted"
```

**On this branch's hardware, the auction subset + mypy + ruff + format all passed cleanly:**

```
python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py \
                 tests/test_draft/test_assistant_auction_tournament_cli.py \
                 tests/test_draft/test_assistant_auction_simulation.py -q
# -> 63 passed
```

mypy: `Success: no issues found in 332 source files`. ruff check: clean. ruff format: clean.

---

## The two full-suite failures — NEITHER is from this branch

When the full `pytest -q` ran here, it reported `4 failed, 1713 passed, 17 skipped`. The 4:

### 1. `tests/backtest/test_backtest_smoke.py::test_backtest_smoke_one_cell` — PRE-EXISTING

Fails with a pandera `SchemaError: column 'preseason_implied_team_total' not in dataframe`.
**This fails on `main` too** (confirmed: `git checkout main && python -m pytest
tests/backtest/test_backtest_smoke.py::test_backtest_smoke_one_cell` → 1 failed). It is a
data/feature-pipeline issue unrelated to auction work. **Out of scope for this branch — do
not try to fix it here.** (Worth a separate issue/note to the user, but it must not block
this branch.)

### 2. `tests/test_models/test_ensemble_model_smoke.py` (3 tests) — SUSPECTED CPU-FAULT ARTIFACT

`test_ensemble_save_load_round_trip`, `test_ensemble_predict_handles_empty_features`,
`test_fit_produces_non_static_weights`. These **PASS on `main`** and they **pass in isolation**
— they only failed inside the big parallel full-suite run on the failing CPU. Non-deterministic
numerical failures under heavy parallel load are the known signature of this box's CPU fault.

**Your verification (the whole reason for the handoff):** on reliable hardware, run
`python -m pytest tests/test_models/test_ensemble_model_smoke.py -q`. They should pass. Then
run the full `python -m pytest -q` and confirm the only remaining failure is the pre-existing
backtest-smoke one (#1). If the ensemble tests genuinely fail on good hardware, *then* there's
a real bug to investigate with `superpowers:systematic-debugging` — but given they're green on
`main` and this branch never touches model code, that's not expected.

---

## Finishing

Once the full suite is clean except the pre-existing backtest-smoke failure:

1. Invoke `superpowers:finishing-a-development-branch`.
2. It will present merge / PR / keep / discard options. The branch splits from `main`.
   This is a **normal repo** (not a worktree).
3. **Before merging or opening the PR, delete `INSTRUCTIONS.md`** so it doesn't land on `main`.
4. After finishing, update `project_management.md` / `TODO.md` per `CLAUDE.md`'s "Stay in sync"
   rule: record that budget-urgency + `studsdepth` shipped, Run F is logged, and the
   strategy-selection decision remains deferred to September 2026.

## Untracked artifacts (leave them alone)

`git status` shows untracked data/checkpoint dirs (`data/vorp_2026/`, `_h2h_ckpt_*`,
`_crashdump/`, `data/processed/espn_weekly_projections/`, several `reports/*.png`, etc.).
These are large local artifacts — **do not commit or push them.** Only the branch commits
above are the deliverable.
