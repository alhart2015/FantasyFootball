# FLEX position bounds — stop anchoring FLEX to RB

**Status:** design approved (2026-08-13). **Branch:** `fix/auction-flex-position-bounds`.
**Issue:** [#143](https://github.com/alhart2015/FantasyFootball/issues/143).

## Problem

`bot_position_bounds` adds every FLEX slot to **RB** unconditionally (and SUPER_FLEX to QB). In
Will's league (`QB1 RB2 WR2 TE1 FLEX2 BENCH5`) that yields:

| pos | MIN | MAX | VORP starter demand/team |
|---|---|---|---|
| QB | 1 | 2 | 1.03 |
| RB | **4** | **7** | 2.82 |
| WR | **2** | **4** | 3.14 |
| TE | 1 | 2 | 1.03 |

Two independent errors compound:

1. **RB's minimum is inflated from 2 to 4** by the anchor.
2. **The bench is then distributed proportionally to those minimums**, so RB collects 3 of the 5
   bench slots and WR only 2 — the anchor is applied a second time.

The result is a hard cap of **4 WR** per seat. Measured over 960 rosters, that cap **binds on
98–99%** of them, while the RB cap (7) binds on 3–5%. Every seat in the sim — the hero included, as
it is gated by the same rule (Run-O spec R2) — is structurally forbidden a 5th WR.

**The valuation layer disagrees.** `vorp._starter_demand` allocates FLEX by actually filling the
slots with whoever projects best (`_select_pool` on a bench-stripped league), and on the real Will
table that gives **WR 3.14 starters/team vs RB 2.82** — WR absorbs *more* of the FLEX than RB. So
the pricing layer and the roster-discipline layer hold opposite views of what a FLEX slot is, and
the discipline layer wins at draft time.

This is not a cosmetic issue: every result in `reports/auction_tournament_validation_2026.md` (Runs
A–X) was produced under it, including the bake-off that selected `overbid_noramp` and the entire
nomination line of work. It also biases the nomination heuristics specifically — `drain_off_position`
and `drain_value_gap_off_position` treat a position as poisonable once it reaches its *minimum*, so
WR qualifies at 2 while RB does not until 4, making them preferentially nominate WRs.

The docstring calls the anchor "unconditional" but no spec gives a rationale; it reads as a
simplification inherited from the snake `draft_field` rule.

## Goals

- Make the roster-discipline bounds treat FLEX as what it is: a slot **any** flex-eligible position
  can fill, not a guaranteed RB.
- Keep `bot_position_bounds` a pure function of `roster_slots` (no pool, no projections) so every
  existing caller works unchanged.
- Re-run the affected experiments and report what moved, including whether `overbid_noramp` is still
  the right hero for Will's league.

## Non-goals

- Re-tuning any strategy, or changing a default because the re-run says so. This slice restores
  correctness and **measures** the consequences; adoption decisions stay with the user (September).
- Touching the valuation layer. `vorp.py` is already correct.
- Reconciling the *snake* draft's use of the same rule beyond what falls out of the shared fix.

## Chosen approach

Split the two roles the current code conflates.

- **`minimums` = dedicated starter slots only.** What a seat must end up with. FLEX and SUPER_FLEX
  contribute **nothing**, because no single position is required to fill them.
- **`maximums` = dedicated + flex capacity + bench share.** A flex-eligible position could in
  principle fill *every* FLEX slot, so its cap gets the full FLEX count (and QB the full SUPER_FLEX
  count); the bench is then distributed proportionally to the **minimums** as today.

Will's league becomes:

| pos | MIN (was) | MAX (was) |
|---|---|---|
| QB | 1 (1) | 2 (2) |
| RB | **2** (4) | **6** (7) |
| WR | **2** (2) | **6** (4) |
| TE | 1 (1) | **4** (2) |

RB and WR are now symmetric, which is the point — the rule no longer expresses a preference the
valuation layer does not share. RB's cap *tightens* (7 → 6) and WR's *loosens* (4 → 6).

**Why not reuse `vorp._starter_demand`?** It is the more precise allocation, but it needs a
projections DataFrame; `bot_position_bounds` takes only `roster_slots` and is called from the engine
hot path and the live board. Threading a pool through every caller is a much larger change for a
second-order gain. The capacity-based rule above is pool-free and removes the actual defect (an
asymmetry between RB and WR that the data contradicts). Recorded as a deliberate trade-off.

## Requirements

- **R1 — no positional asymmetry.** For a league whose FLEX-eligible positions have equal dedicated
  starter counts, the resulting min and max must be equal across those positions.
- **R2 — starting lineup always fillable.** After minimums are met, the remaining picks must always
  be able to fill the FLEX slots. (With QB capped near its minimum, the residual picks are
  necessarily flex-eligible; asserted by test rather than argued.)
- **R3 — `Σmax ≥ roster_size`** preserved (the existing invariant: caps must permit a full roster).
- **R4 — SUPER_FLEX symmetric.** SUPER_FLEX raises the *cap* of every super-flex-eligible position,
  not just QB's minimum.
- **R5 — existing tests that pin the anchor are updated, not deleted,** and each edit says why in a
  comment. These tests are correct assertions about incorrect behaviour; the behaviour is what is
  changing. (`CLAUDE.md`: a test change needs an explicit stated reason — this is it, and the user
  authorised the fix.)
- **R6 — gates.** `pytest`, `mypy src tests` strict, `ruff check`, `ruff format --check`.

## Re-runs required (the point of the slice)

All on Will's settings, unchanged from the runs they supersede so the comparison is apples-to-apples
(`will_half12` + `will_half12_pass5`, `overbidder` field `overbid=0.2/pace=4.5/opening/jitter=0.35`,
ESPN market, ADP nomination jitter 12, 12 seats × 20 seeds × 300 sims):

1. **Cap-binding diagnostic** — confirm the WR cap no longer binds ~99% of rosters.
2. **Full-field bake-off** — does `overbid_noramp` survive as the hero for Will's league?
3. **Nomination probe** (`control`/`off_pos`/`gap`/`gap_off`) on the winning hero, since the
   "filled" test that drives two of those heuristics changes directly.

## Edge cases / failure modes

- **A position with no dedicated slots but flex eligibility** (e.g. TE in a `RB/WR/FLEX`-only
  league) gets minimum 0. It then has no reserved picks — correct, since nothing forces a TE — but
  it must still appear in the maps with a usable cap, or `bot_eligible` will exclude it entirely
  (the eligible set is drawn strictly from the bound keysets). Handled explicitly and tested.
- **Bench share is proportional to minimums**, which now sum lower, so caps loosen slightly overall.
  R3 still holds and looser caps cannot make a roster illegal.
- **Prior-run reproducibility is broken by design.** Every auction number in the repo moves. The
  report must say so rather than silently restating superseded figures.

## Rollout

`reports/auction_tournament_validation_2026.md` gains a prominent note that Runs A–X ran under the
anchored bounds, plus the new run recording what changed. `project_management.md` records the
decision. Nothing is adopted on the strength of the re-run alone.
