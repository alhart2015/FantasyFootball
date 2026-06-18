# Auction budget-urgency + studs/depth hero — design

**Status:** approved (brainstormed 2026-06-18); **passed `superpowers-spec-review` 2026-06-18** (no Critical/High/Medium; Low residuals: integration threshold wording, the `open_slots==0` bound edge, stale dependency note). Proceeding to plan + execute via `superpowers-go`.
**Owner:** draft-hub / auction.
**Depends on:** the realistic-market slice (PR #79 / `feat/auction-realistic-market`) — needs `nomination_temp`, the mixed bot field, and the seven existing hero models incl. `PatientValueBid`. Branch will rebase onto `main` once #78+#79 land.
**Source context:** `reports/auction_tournament_validation_2026.md` (Run E eye-test), prior specs `2026-06-17-auction-stars-and-scrubs-design.md`, `2026-06-18-auction-realistic-market-design.md`.

## Problem

Run E's roster eye-test (realistic market, half/16) showed every hero strategy **underspends** and gets out-built by the sane mixed-bot field:
- `static` won 4 elite studs at fair value but **stopped at $165/$200** and filled the rest with $1 scrubs → #8/16.
- `patient` deliberately **passed on the studs**, spread across mid-tier, **overpaid $24 for a QB** (a deep position), and finished **last** with no elite anchor — total spend $145.
- The winning bots spent ~all $200 on a couple of studs *plus real mid-tier depth*; the cratering bots **overpaid** (e.g. $60 for a non-elite RB).

So the realistic market punishes three mistakes our models make: **underspending**, **passing studs**, and **overpaying**. None of the seven contestants do what a sharp manager does — secure a couple of studs near fair value, then deploy the *whole* budget on real depth. This slice adds that strategy and gives every existing model a mechanism to stop leaving money on the table.

**Framing (user):** if a balanced, spend-it-all bot is genuinely hard to beat, that *is* a valid strategy — so the new hero is deliberately "the good bot, as a hero."

## Goals

1. **`_budget_urgency` factor** — a shared, late-draft escalation each hero contestant applies so it doesn't end the draft with idle cash.
2. **Refine all seven existing contestants** to apply `_budget_urgency` (deploy the budget).
3. **`StudsAndDepthBid`** — a new (8th) contestant: secure a few studs near fair value, then bid fair value across mid-tier depth (no $1-dumping), deploying the full budget.
4. **Re-run (Run F)** the eight-model bake-off vs the realistic market; record. **No winner** — data-gathering; September.

## Non-goals

- **Not** changing the bots (`market.py` archetypes), the engine (`_simulate_to_state`), `resolve_bids`, the scorer, or the snake field. Urgency lives entirely in the hero-strategy layer (`bid_strategy.py`).
- **Not** anchoring bots on real published auction values (still its own future slice; the deeper realism lever).
- **Not** re-tuning the existing models' core bid logic beyond multiplying by urgency.
- **No** new pandera schema; `GsisId` canonical; `Position` enums not strings.

## Chosen approach

All changes are in `src/projections/draft/assistant/auction/bid_strategy.py` plus the CLI model registry. The bots and engine are untouched.

### A. `_budget_urgency(view, config) -> float` (shared helper)

Late-draft ramp — **exactly 1.0 at the draft start and when broke**, escalating only when a hero is overfunded *and* the draft is winding down:

```
surplus  = view.my_budget - config.min_bid * view.my_open_slots   # spare cash beyond $1/open slot
if surplus <= 0:
    return 1.0
progress = 1.0 - view.my_open_slots / config.roster_size           # 0 at the first pick, ->1 near the end
return 1.0 + URGENCY_GAIN * progress * (surplus / view.my_budget)
```

- `URGENCY_GAIN` (module constant, default **3.0**). Both `progress` and `surplus/my_budget` are in `[0, 1)`, so urgency is naturally bounded to `[1.0, 1.0 + URGENCY_GAIN)` — no separate cap needed (the engine's `[min_bid, feasible_max]` clamp bounds the resulting bid).
- `progress = 0` at an empty roster (`my_open_slots == roster_size`) ⇒ urgency `1.0` ⇒ existing empty-roster strategy tests are unchanged.
- `surplus <= 0` (the hero is at/below the $1-per-open-slot floor) ⇒ urgency `1.0` (don't escalate what you can't afford).

### B. Apply urgency to the seven existing contestants

Each of `StaticDollarBid`, `InflationBid`, `MarginalValueBid`, `AnchorBudgetBid`, `OverbidValueBid`, `VorpShareBid`, `PatientValueBid` computes its bid as today, then returns `round(base_bid * _budget_urgency(view, config))` at a single exit. Because urgency is `1.0` at progress 0, **empty-roster behavior (and its tests) is unchanged**; the change is purely a late-draft escalation that deploys hoarded budget (including bidding up the best remaining players when only scrubs are left and cash remains — the correct response to "I'll otherwise leave money").

### C. `StudsAndDepthBid` (8th contestant)

`StudsAndDepthBid(stud_premium=0.2, stud_frac=0.10, scrub_frac=0.20)` — the "good bot as a hero". Tier the nominee by VORP over the full pool (reuse `_vorp_threshold`); read dollars from `view.baseline_dollars`:
- `stud_cut = _vorp_threshold(pool, round(stud_frac * len(pool)))`, `scrub_cut = _vorp_threshold(pool, round((1 - scrub_frac) * len(pool)))`.
- **stud** (`v >= stud_cut`): `base = auction_dollars * (1 + stud_premium)` — a modest premium to actually *win* the anchor (unlike `static`, which bids exact value and loses ties).
- **scrub** (`v < scrub_cut`): `base = min_bid`.
- **mid-tier depth** (between): `base = auction_dollars` — fair value on real depth (no $1-dumping; note `scrub_frac=0.20` is deliberately small so *most* of the pool counts as depth, not scrubs).
- return `round(base * _budget_urgency(view, config))`.

The urgency factor turns the fair-value depth bids into a full-budget deployment as the draft winds down — securing a couple of studs, then spending the rest on a balanced, startable roster.

### D. Wire + re-run

- CLI `_MODELS` gains `"studsdepth": StudsAndDepthBid()` ⇒ **eight** contestants; the realistic-market default (random nomination + mixed field) is unchanged.
- `reports/auction_tournament_validation_2026.md` gains **Run F** (eight models, realistic market). No winner.

## Requirements

R1. `_budget_urgency(view, config) -> float` exists in `bid_strategy.py`, implements §A exactly, returns `1.0` at `progress == 0` and when `surplus <= 0`, and is bounded `[1.0, 1.0 + URGENCY_GAIN)`. `URGENCY_GAIN` is a module constant (default 3.0).
R2. All seven existing contestants multiply their final bid by `_budget_urgency(view, config)` at a single return point; empty-roster bids are unchanged (urgency 1.0).
R3. `StudsAndDepthBid` exists, satisfies `AuctionBidStrategy`, is a frozen dataclass, and behaves exactly as §C (stud premium / fair-value depth / min_bid scrub, all × urgency).
R4. CLI `compare` races eight models (adds `studsdepth`); realistic-market defaults unchanged; `AuctionBidStrategy` protocol satisfied by all.
R5. The bots (`market.py`), the engine (`_simulate_to_state`), `resolve_bids`, and the snake field are **untouched**.
R6. Determinism: every hero bid is a pure function of state (urgency reads only `view`/`config`); same `(seed, temp, mix, strategy)` ⇒ identical rosters.
R7. Conventions: `GsisId` canonical; `Position`/`RosterSlot` enums; `_PYARROW_STR`; no new schema.
R8. Run F recorded; no winner declared.

## Edge cases / failure modes

- **Empty roster / draft start** (`my_open_slots == roster_size`): `progress = 0` ⇒ urgency `1.0`. Preserves existing behavior + tests.
- **Broke** (`surplus <= 0`): urgency `1.0` — no escalation beyond what's affordable; the engine still clamps to `feasible_max`.
- **Urgency on `min_bid`/scrub returns**: late-draft, `round(min_bid * urgency)` rises above `min_bid` — intended (deploy idle cash on the best remaining players); bounded by `feasible_max`.
- **`StudsAndDepthBid` tiny pool**: `round(stud_frac * len(pool)) == 0` ⇒ `_vorp_threshold(pool, 0) == +inf` ⇒ no studs (degrades to depth/scrub) — safe.
- **`my_budget == 0`** late: `surplus < 0` ⇒ urgency 1.0; no division by zero (guard returns before the `surplus / my_budget` term).
- **Urgency never lowers a bid** (≥ 1.0), so it cannot make a strategy *underbid* a player it would otherwise win.

## Testing expectations

Unit (`tests/test_draft/test_assistant_auction_bid_strategy.py`, reusing `_vpool`/`_vbaseline`/`_aview`/`_vconfig`):
- `_budget_urgency`: `== 1.0` at an empty roster (progress 0); `== 1.0` when `surplus <= 0`; `> 1.0` for a partial roster with budget surplus; increases with `progress` and with `surplus`; bounded `< 1 + URGENCY_GAIN`.
- Each existing contestant: **empty-roster bid unchanged** vs its current value (urgency 1.0); a **partial-roster, overfunded** case bids strictly higher than the un-urgent base (the feature). Update the partial-roster tests that change, each with a one-line "urgency feature" reason.
- `StudsAndDepthBid`: stud → premium (`round(value*(1+stud_premium))` at progress 0); mid-tier → fair value; scrub → min_bid; all scale up under a partial overfunded view.

Integration (engine, realistic market):
- A `StudsAndDepthBid` hero **deploys ~the full budget** (spends materially more than `static` on the same seed) and ends with a startable roster.
- Determinism: same `(strategy, seed, temp, mix)` ⇒ identical rosters.

Gates: `pytest -v` (touched modules), `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`.

## Phasing

~5 tasks, each ≤5 files:
1. **`_budget_urgency` helper** + `URGENCY_GAIN` + unit tests (=1 early, ramps late, bounded).
2. **Apply urgency to the seven existing contestants** (single-exit refactor) + update affected partial-roster tests.
3. **`StudsAndDepthBid`** + unit tests.
4. **CLI** — eight-model `compare` + the integration "deploys full budget" test.
5. **Run F** bake-off (chunked per the Raptor Lake fault) + tracking doc; no winner.

## Open question for later

If Run F's `StudsAndDepthBid` (and the urgency-refined field) still trails the bots, that strengthens the case that the **mixed-bot field is mis-calibrated** (too strong) until anchored on real published auction values — the next realism slice.
