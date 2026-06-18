# Auction stars-and-scrubs bidders + position-aware hero — design

**Status:** approved (brainstormed 2026-06-17), pending spec review.
**Owner:** draft-hub / auction.
**Source context:** `reports/auction_tournament_validation_2026.md`, prior slices `docs/superpowers/specs/2026-06-17-auction-strategy-tournament-design.md` and `…-auction-sane-bots-design.md`.

## Problem

Roster-level inspection of the auction bake-off (one draft per model, half×16, seed 0) showed the hero builds **worse teams than a no-logic bot**, and the scorer is correct in saying so:

- The winning bot played textbook **stars-and-scrubs**: De'Von Achane $56 / Ja'Marr Chase $54 / Justin Jefferson $46 / TreVeyon Henderson $37 (four anchors for $193) then $1 fills (incl. Mahomes $1) → expPts 1158, playoff 0.63.
- `static` spent its full $200 but **spread it evenly** ($15–21 across 11 players, nobody over 175 pts) and, because the hero is **ungated**, built **7 WR / 0 TE** — leaving the TE starter empty (auto-scores 0) and stranding a 7th WR on the bench.
- `inflation` spent only **$69 of $200**; `marginal` only **$26** — they never buy anchors.

Two concrete, independent defects follow: (1) the hero has **no positional discipline**, and (2) we have **no strategy that plays a real auction** (concentrate budget on a few anchors, fill with $1 values). This slice fixes (1) for every bidder and adds three strategies that address (2).

## Goals

1. **Position-aware hero.** Every hero bid model obeys the *same* positional rules the bots already follow (`bot_eligible` + `bot_position_bounds`): never bid a position already at its max, always reserve enough picks to fill minimum starters. The hero stops being the ungated special case.
2. **Three new budget-spending bid models**, each position-aware and each concentrating the budget on high-VORP players in a distinct way: `AnchorBudgetBid`, `OverbidValueBid`, `VorpShareBid`.
3. **Six-contestant bake-off.** Race all six (existing `static`/`inflation`/`marginal`, now gated, + the three new) through the existing `project_draft` scorer; record the run in the tracking doc. **No winner is declared** — this is data-gathering; the adopt decision is the user's, in September.

## Non-goals

- **Not** fixing `inflation`/`marginal` underspend. Their bid *magnitudes* are unchanged (only the position gate is added) so the bake-off cleanly isolates position-awareness and contrasts the value-anchored trio against the budget-spenders. Their underspend is left intentionally for contrast.
- **Not** changing the bots, the market clearing (`resolve_bids`), or the scorer (`project_draft`). All three are verified correct.
- **Not** ingesting real published auction values (still its own future slice).
- **Not** strategic nomination (deferred axis).
- **No** new pandera schema; **no** new ID flavor; reference enums (`Position`, `RosterSlot`) not strings.

## Chosen approach

### A. Position-awareness — gate at the engine, not in each model

In `simulation.py::_simulate_to_state`, the hero is currently ungated:
- eligibility-build loop: `seat_eligible[hero0] = all_positions`;
- bid loop (line ~144): the hero bids unconditionally.

Generalize so **every open seat** (hero + bots) draws its eligible set from `bot_eligible(counts, picks_left, minimums=…, maximums=…)`, where `minimums, maximums = bot_position_bounds(config.roster_slots)` (already computed once per draft). The hero/bot branches then differ **only in bid magnitude**:

```
for seat in open seats:
    fmax = _feasible_max(state, seat, rs, min_bid)
    elig = all_positions if forced else seat_eligible[seat]      # same for hero and bots
    if pos_by_id[str(nominee_id)] not in elig:
        continue                                                 # abstain (hero too, now)
    if seat == hero0:
        desired = strategy.max_bid(_build_view(...), player, pool, config)
    else:
        desired = bot_max_bid(SeatView(open_slots=…, eligible_positions=elig), …)
        if desired <= 0:
            continue
    bids[seat] = max(min_bid, min(int(desired), fmax))
```

Consequences (all intended):
- The hero never exceeds a position max and never leaves a minimum starter unfilled — identical rule to the bots.
- The **nomination union** now includes the hero's *gated* set instead of `all_positions`; the existing forced-pick path (un-gates every open seat when the pool is thin) is unchanged and still guarantees ≥1 bid.
- Bid models keep their `max_bid(view, player, pool, config)` signature — the engine enforces legality, so models compute magnitude only. This makes all six models position-aware with no per-model gating code.

### B. Three new bid models (`bid_strategy.py`)

All three: receive `(view, player, pool, config)`; return a desired int bid (the engine clamps to `[min_bid, feasible_max]`); are deterministic (no RNG — only the bot market is stochastic); read VORP from the pool (`player["vorp"]`, `view.my_roster["vorp"]`, `pool["vorp"]`) and dollars from `view.baseline_dollars.loc[gsis, "auction_dollars"]`. Position legality is enforced upstream (§A), so a model is only ever consulted for an eligible nominee.

Shared module helpers (new, in `bid_strategy.py`):
- `_undrafted(pool, drafted) -> pd.DataFrame` — pool rows whose `gsis_id` ∉ `drafted`.
- `_vorp_threshold(pool, k) -> float` — the k-th highest `vorp` in `pool` (`pool["vorp"].nlargest(k).min()`; if `len(pool) < k`, the pool min). Used to define "anchor-grade".

**1. `AnchorBudgetBid(n_anchors: int = 4)`** — classic stars-and-scrubs.
- `league_anchor_count = n_anchors * config.n_teams`; `threshold = _vorp_threshold(pool, league_anchor_count)`.
- `is_anchor(p) = p["vorp"] >= threshold`.
- `anchors_held = (view.my_roster["vorp"] >= threshold).sum()` (0 if roster empty).
- `anchors_remaining = max(0, n_anchors - anchors_held)`.
- `open = view.my_open_slots`; `feasible_max = view.my_budget - min_bid * (open - 1)`.
- If `is_anchor(player)` **and** `anchors_remaining > 0`:
  - `reserve = min_bid * max(0, open - anchors_remaining)` ($1 for each non-anchor slot still to fill),
  - `cap = (view.my_budget - reserve) / anchors_remaining`,
  - `bid = round(min(cap, feasible_max))` (≥ market value for a real anchor → wins).
- Else `bid = min_bid` (the $1 scrubs).

**2. `OverbidValueBid(k: float = 1.3, stud_count: int | None = None)`** — pay up for studs, value for the rest.
- `stud_count` defaults to `3 * config.n_teams` (≈ top three rounds by VORP); `threshold = _vorp_threshold(pool, stud_count)`.
- `value = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])`.
- `bid = round(value * k) if player["vorp"] >= threshold else value`.
- (The `[min_bid, feasible_max]` engine clamp turns this into `min_bid` automatically once the budget is exhausted — no explicit "budget low" branch needed.)

**3. `VorpShareBid()`** — allocate the remaining budget proportionally to VORP across the players I still aim to roster.
- `cand = _undrafted(pool, view.drafted)`; `targets = cand.nlargest(view.my_open_slots, "vorp")` (the top-`open` undrafted by VORP — the slots I'd ideally fill).
- `pos_vorp(x) = max(0.0, x["vorp"])`; `denom = sum(pos_vorp over targets)`.
- If `player["gsis_id"]` ∈ `targets.gsis_id` **and** `denom > 0`:
  - `share = pos_vorp(player) / denom`; `bid = round(view.my_budget * share)`.
- Else `bid = min_bid`.
- Guards: `denom == 0` (all targets replacement-level) → `min_bid`; the engine clamp bounds the result.

Parameters (`n_anchors`, `k`, `stud_count`) are constructor args with the defaults above so a later sweep can vary them without code change.

### C. Tournament + CLI wiring

- `run_auction_tournament` already accepts `strategies: Mapping[str, AuctionBidStrategy]` and is model-agnostic — no change to its body.
- The CLI `compare` subcommand's default model set grows from three to **six**: `static`, `inflation`, `marginal`, `anchors`, `overbid`, `vorpshare`. (Keep an optional `--models a,b,c` subset flag if it already exists; otherwise default to all six — do not add new flags beyond what's needed.)
- `reports/auction_tournament_validation_2026.md` gains a new run table (Run C) with all six models. Framed as data; **no winner**.

## Requirements

R1. Every open seat — hero included — has its eligible position set computed by `bot_eligible(counts, picks_left, minimums=…, maximums=…)`; the hero is no longer assigned `all_positions` outside the forced path.
R2. In the bid loop both hero and bots abstain (no bid recorded) when `pos_by_id[nominee] ∉ elig`, where `elig = all_positions if forced else seat_eligible[seat]`.
R3. The nomination union is built from every open seat's (now gated) eligible set; the forced-pick path still un-gates all open seats and still guarantees `bids` is non-empty (the `assert bids` must remain unreachable on valid input).
R4. `AnchorBudgetBid`, `OverbidValueBid`, `VorpShareBid` exist in `bid_strategy.py`, satisfy the `AuctionBidStrategy` protocol, are frozen dataclasses, and behave exactly as specified in §B (including the division/zero/negative-VORP guards).
R5. Bid models keep the `max_bid(view, player, pool, config) -> int` signature; no model implements its own position gate (the engine owns it).
R6. The CLI `compare` races all six models by default and prints the existing per-model table + paired-diff output for the six.
R7. Snake-draft (`backtest/draft_field.py`) and bot (`market.py::bot_max_bid`, `resolve_bids`) behavior is byte-identical — untouched.
R8. Determinism: for a fixed `(strategy, seed)` the resulting rosters are identical run-to-run (hero bids are a pure function of state; only the bot market consumes RNG).
R9. Conventions: `GsisId` canonical; `Position`/`RosterSlot` enums not strings; `pd.StringDtype("pyarrow")` for gsis columns; `df = SCHEMA.validate(df)` at any boundary that emits a frame; no `df.to_parquet` outside the store.

## Edge cases / failure modes

- **All six models gated to empty mid-round** → some nominee fits no open seat's set → forced path un-gates everyone (warns "pool thin"), so the round still completes. (Same mechanism as the sane-bots slice; the hero being gated does not weaken it because the hero's gated set is part of the union.)
- **`AnchorBudgetBid` with `anchors_remaining == 0`** → every remaining bid is `min_bid` (the scrub phase).
- **`open - anchors_remaining < 0`** (more anchors wanted than slots left) → `reserve` clamped to ≥ 0; `cap` clamped by `feasible_max`.
- **`VorpShareBid` denom 0 / negative VORP** → `min_bid`; negative VORP clamped to 0 in the share numerator/denominator.
- **`OverbidValueBid` value lookup** — `player["gsis_id"]` is always present in `baseline_dollars` (built from the same pool); a missing key is a programming error, not a runtime branch.
- **Budget exhaustion** — any model returning more than `feasible_max` is clamped by the engine to keep `min_bid` per remaining slot; models never re-implement the reserve.
- **Empty `my_roster`** (first nomination) — `anchors_held = 0`, `my_open_slots = roster_size`; all formulas defined.

## Testing expectations

Per-model unit tests (hand-built small pool + `LeagueConfig`, no I/O):
- `AnchorBudgetBid`: bids **above market value** for a top-VORP nominee while anchors remain; bids `min_bid` once `anchors_remaining == 0`; respects the per-anchor `cap`; spends ~the full budget over a full draft.
- `OverbidValueBid`: bids `round(value*k)` for a stud, `value` for a non-stud; clamp turns it to `min_bid` when broke.
- `VorpShareBid`: concentrates on the top-`open` VORP targets; `min_bid` for an off-target nominee; `denom==0` → `min_bid`.

Engine-level position-gate tests (in the simulation/harness tests):
- A hero running `StaticDollarBid` no longer finishes with a position over its max or a minimum starter unfilled (e.g. not 7 WR / 0 TE) — the defect this slice fixes.
- `assert bids` never fires with a gated hero (forced-pick completeness preserved); the "pool thin" warning still triggers in the thin-pool fixture.
- Determinism: same `(strategy, seed)` → identical rosters (re-run equality).

Regression:
- Existing `static`/`inflation`/`marginal` tests still pass; any test that assumed an ungated hero is updated with a stated reason.
- The full backtest suite (snake field) is unchanged (no drift).

Gates: `pytest -v`, `mypy src tests` (strict), `ruff check src tests`, `ruff format --check src tests`; plus `pytest -v -k "ingest or store or schemas"` if any schema/store path is touched (none expected).

## Phasing

One cohesive slice; the plan should decompose into ~4 tasks, each ≤5 files:
1. **New bid models** — `AnchorBudgetBid`/`OverbidValueBid`/`VorpShareBid` + shared `_undrafted`/`_vorp_threshold` helpers in `bid_strategy.py`, with unit tests.
2. **Engine position-gate** — generalize `_simulate_to_state` so the hero is gated like a bot; engine tests (gate + completeness + determinism).
3. **Tournament/CLI wiring** — six-model default `compare`; CLI test.
4. **Bake-off + tracking doc** — run the six-model bake-off (chunked per the Raptor Lake CPU fault), record Run C; no winner.
