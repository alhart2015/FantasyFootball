# Big-Stack Overspend Hero (Auction) — Design

**Status:** design approved (brainstorming, 2026-07-16).
**Branch:** `feat/auction-bigstack-overspend` (stacked on `feat/auction-adp-nomination` / PR #99 — Phase 3 needs `market_adp_jitter` from #99). **Merge order:** #99 lands first, then this branch rebases onto `main`; its PR should not merge before #99.

## Problem

The shipped `balanced` p0.0 hero bids fair value capped at a LOW per-player pace cap (`pace × my_budget/my_open_slots` ≈ $23.5 opening). It never pursues studs and, when only low-value players remain, wins them cheap and **ends with unused budget** (observed: a miss roster left ~$69; even strong rosters left $7–25; the `BalancedBot`s leave $22–25). Unused budget at the end is pure waste — it could have been converted into marginal talent.

The hero cannot fix this by bidding more *late*: in a second-price auction, overbidding an uncontested player still clears at ~`min_bid`, so once opponents are broke the surplus is stranded. Cash only converts to talent when an opponent is still pushing the price. So the lever is: **while the hero is the "big stack" (holds more remaining budget than the field), pay up on contested players to deploy the lead — before the room dries up.**

## Goals

- Add a bid strategy that **overpays (bids above fair value, lifting the low pace cap) in proportion to the hero's budget advantage over the field**, and is byte-identical to `balanced` p0.0 when the hero is *not* the big stack.
- Build **two variants** of the "am I the big stack?" signal and A/B both:
  - **`max_opp`** — advantage vs the single richest opponent.
  - **`field_avg`** — advantage vs the league-average remaining budget per open slot (robust to a single hoarding opponent).
- Measure the lift in seat-averaged `reg_win_pct` (+ `make_playoffs_pct`, `champ_pct`) vs `balanced` p0.0, under the realistic **ADP nomination** market, across a small `overpay_gain` sweep. Data-gathering — no strategy adopted.

## Non-goals

- Adopting a new default hero (September strategy-decision policy stands).
- A late-game/endgame "dump" trigger — rejected: by the time opponents are fully broke, overbidding deploys nothing (second-price). The trigger is *relative stack size*, active throughout the draft.
- A `my_budget/my_open_slots`-vs-opening-share trigger — rejected: it is reactive (rises only *because* the hero underpaid in absolute terms) and does not capture "big stack relative to the table."
- Nomination changes, bot re-calibration, multi-season/size sweeps (separate axes).

## Chosen approach

### The bid (shared by both variants)

A new frozen dataclass `BigStackBid` in `bid_strategy.py`:

```python
@dataclass(frozen=True)
class BigStackBid:
    overpay_gain: float = 1.0
    reference: Literal["max_opp", "field_avg"] = "field_avg"
    premium: float = 0.0
    pace: float = 2.0
```

`max_bid(view, player, pool, config)`:

```
fair      = view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"]
per_slot  = view.my_budget / max(1, view.my_open_slots)
advantage = <reference-specific, see below>          # >1 ⇒ hero is the big stack
overpay   = overpay_gain * max(0.0, advantage - 1.0) # 0 when not dominant
target    = fair * (1.0 + premium + overpay)         # overpay lets the bid exceed fair
cap       = pace * per_slot * (1.0 + overpay)         # lift the low pace cap too
return round(min(target, cap))                        # engine then clamps to feasible_max
```

- At `advantage ≤ 1` → `overpay = 0` → `round(min(fair*(1+premium), pace*per_slot))`, which is **exactly `BalancedValueBid(premium=premium, pace=pace)`**. With the defaults (`premium=0, pace=2`) it is byte-identical to the shipped `balanced` p0.0.
- Both `target` and `cap` are lifted by the same `(1 + overpay)` factor so the hero can bid above fair value AND above its normal cap when dominant.
- The strategy returns a *desired* bid; the engine already clamps it to `feasible_max = budget − min_bid*(open_slots−1)` and to `min_bid`, so `BigStackBid` never needs `feasible_max` and can never overspend into insolvency.

### `advantage` per variant

Opponent budgets come from `view.budgets_by_seat` (all seats) minus the hero's own budget. The view exposes all seats' budgets but **not** the hero's seat index; remove one instance of `view.my_budget` to get the opponents' budgets (a duplicate-budget tie removes an arbitrary one — acceptable, it is a proxy).

- **`max_opp`:** `advantage = my_budget / max(max(opp_budgets), min_bid)`.
- **`field_avg`:** `advantage = (my_budget / max(1, my_open_slots)) / (Σ budgets_by_seat / max(1, total_open_slots))`, where `total_open_slots = n_teams*roster_size − len(view.drafted)` (reuse `_total_open_slots`). This is the hero's remaining budget-per-slot vs the league-average remaining budget-per-slot — a single hoarding opponent barely moves the league average, so it cannot neutralize the signal.

### Registration & measurement

- Phase 3 races a **dedicated contestant dict**, NOT `tournament_cli._MODELS` (the shipped default field must not be polluted with sweep variants). Exactly **7 contestants**: `balanced = BalancedValueBid(premium=0.0)` (the control, once) + `BigStackBid(reference=r, overpay_gain=g)` for `r ∈ {"max_opp", "field_avg"}` × `g ∈ {0.5, 1.0, 2.0}`. Registering a *chosen* variant into `_MODELS` is deferred to after the result (a separate decision, per the September policy).
- A/B via the seat-sweep runner's contestant/aggregation shape under ADP nomination (`--market-adp-jitter 12`), seat-averaged over 12 seats × both markets, `reg_win_pct` (+ playoff/champ), 20 seeds × 300 sims — the Run-P methodology. `auction_seat_sweep.py` races `_MODELS` specifically, so Phase 3 either parametrizes it to accept a custom contestant dict or uses a dedicated runner in the Run-N `premium_sweep` mould; the plan picks one.
- The three `overpay_gain` values {0.5, 1.0, 2.0} probe the tuning curve: too small won't deploy, too large overpays into −EV.

### Interpretation (data-gathering — no adopt bar)

There is **no pre-registered adopt/reject threshold** (the strategy decision is September). The Phase-3 deliverable is the *characterized* lift: for each `BigStackBid` variant×gain, report its seat-averaged `reg_win_pct` (+ playoff/champ) and the **delta vs `balanced` p0.0** in **both** markets, and flag whether each delta **exceeds the seed-noise band** (the per-seat spread at 20 seeds, ≈ ±0.03 per market cell; the seat-average is tighter — note if a delta sits inside it). "It works" = a delta clearly above noise in at least the ESPN market where the budget-deployment lever should bite; a wash/negative is an equally valid recorded result. No strategy is adopted from this run.

## Requirements

- **R1 — fallback identity.** With `overpay_gain` such that `advantage ≤ 1` on every bid (hero never the big stack), OR whenever `advantage ≤ 1` for a given bid, `BigStackBid(premium=p, pace=q).max_bid(...)` returns exactly `BalancedValueBid(premium=p, pace=q).max_bid(...)` (unit test on shared views).
- **R2 — overpay direction.** When `advantage > 1`, `BigStackBid` returns a bid **≥** the balanced bid on the same view, and strictly greater when the pace cap would otherwise bind (unit test).
- **R3 — variant correctness.** `max_opp` uses the richest opponent; `field_avg` uses the league-average per-slot. A test constructs a view with one hoarding opponent and asserts `field_avg` still signals big-stack while `max_opp` does not.
- **R4 — solvency.** `BigStackBid` never returns a bid that the engine cannot clamp to a feasible value; because it returns a finite desired bid and the engine clamps to `feasible_max`, the hero always affords `min_bid` for its remaining slots (covered by the existing engine invariant + a smoke that every seat fills a legal roster).
- **R5 — bid held-elsewhere-fixed.** The A/B changes only the bid strategy; nomination (`market_adp_jitter=12`), bot field, markets, seeds, sims match Run P so the lift is attributable to the bid.
- **R6 — gates.** `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check` clean.
- **R7 — validated `Literal`.** `reference` is a `Literal["max_opp", "field_avg"]`; an invalid value is a construction-time error (`__post_init__` guard), consistent with `BalancedValueBid`'s premium/pace validation.

## Edge cases / failure modes

- **Only the hero has budget (all opponents at min/0).** `max_opp` → `advantage = my_budget / min_bid` (huge) → large overpay, but second-price means it still clears cheap on uncontested players (no runaway spend); `feasible_max` clamps any single bid. Documented, not special-cased.
- **Hero is broke / behind.** `advantage ≤ 1` → `overpay = 0` → balanced behavior. No overpay when you're the short stack.
- **`my_open_slots == 0`** cannot occur (the engine only calls `max_bid` for a seat with an open slot), but `max(1, ...)` guards the division regardless.
- **Duplicate budgets (`opp_budgets` removal).** Removing one `my_budget` instance may drop an opponent who happens to share the hero's budget; acceptable (proxy), and `field_avg` — the recommended variant — does not depend on the per-opponent removal beyond the sum.
- **`overpay_gain = 0`** reduces `BigStackBid` to exactly `balanced` (a useful control / sanity anchor).
- **Second-price non-deployment when the room is fully broke** is expected and documented — the mechanism deploys *before* that, which is the point.

## Testing expectations

- `advantage ≤ 1` ⇒ output equals `BalancedValueBid` (both `premium/pace` defaults and a non-default set) — R1.
- big-stack view (hero rich, field poor) ⇒ bid > balanced bid, and above the pace cap — R2.
- one-hoarder view ⇒ `field_avg` signals big-stack, `max_opp` does not — R3.
- `reference` validation rejects a bad literal — R7.
- `overpay_gain=0` ⇒ identical to `balanced` on a full auction (smoke).
- A full `simulate_auction` with `BigStackBid` yields a legal, full roster (R4 smoke).

## Phasing

- **Phase 1 — the strategy.** `BigStackBid` + `advantage` variants + `__post_init__` validation, with unit tests R1/R2/R3/R7.
- **Phase 2 — wiring.** Register the two variants (or a gain grid) as tournament contestants; a smoke that they run in the sweep + `simulate_auction` (R4).
- **Phase 3 — the A/B.** Crash-safe seat sweep under ADP nomination across the `overpay_gain` grid; aggregate seat-avg both markets; write up as a Run in `reports/auction_tournament_validation_2026.md` + memory; report the lift (data, no adoption).
