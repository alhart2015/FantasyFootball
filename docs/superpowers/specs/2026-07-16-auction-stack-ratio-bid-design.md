# Stack-Ratio Convex-Aggression Auction Hero — Design

**Status:** design approved (brainstorming, 2026-07-16).
**Branch:** `feat/auction-stack-ratio-bid` (off `main`, which already has `BigStackBid` + the ADP-nomination feature).

## Problem

The `BigStackBid` `field_avg` overspend hero (Run Q) deploys a budget lead by lifting the pace cap **linearly** with the per-slot budget advantage. A pick-by-pick trace (2026-07-16, `espn_overspend_trace.py`) showed why it degrades in the ESPN market: the linear multiplier fires **early** — the uncapped aggressive bots overpay for early studs, which drains the field's average budget fast, so the hero's advantage ratio spikes by Q1 (mean 1.30) and it **chases early studs**, collapsing the disciplined breadth hero into a fragile stars-and-scrubs shape ($126 of $200 on its top 5 picks vs `balanced`'s $66 on its top 3). That roster has more *static* optimal-lineup points but fewer *realized* `project_draft` points (ESPN: 1429 vs `balanced` 1460) once injuries/byes/weekly availability are simulated — a depth/robustness penalty — and it loses (reg_win 0.657 vs 0.688). The lever deploys budget into **concentration**, not depth.

The user's principle: aggression should scale **convexly** with the *raw* budget ratio to the field, so it stays disciplined at moderate leads (200/175 ≈ no aggression) and only unleashes at **dominant** leads (200/100, or a late 80/40 — both ratio 2.0, both aggressive). A dominant ratio arises naturally **late** in the draft, when only depth remains — so the surplus deploys into depth **via the draft's timing**, with no explicit value-tier gate (gating was rejected as result-chasing).

## Goals

- A bid strategy `StackRatioBid` whose aggression multiplier is a **convex function of the raw budget ratio** `my_budget / mean(opponent remaining budgets)`, reducing **exactly** to `balanced` (`BalancedValueBid(premium=0.0, pace)`) whenever the hero is not ahead (`ratio ≤ 1`).
- A `(gain, curve)` **parameter sweep** to characterize the ratio → multiplier relationship that best beats `balanced` under the realistic ADP-nomination market, both bot markets, seat-averaged.
- Measure not only `reg_win_pct` but the **roster shape** (spend by draft quartile — early-stud vs late-depth) so we can *see* whether convex curves defer the aggression to depth rather than studs.

## Non-goals

- **Value/VORP tier gating** of which players get the premium (rejected as result-chasing; the draft timing does the routing).
- **Per-slot** budget ratio (normalizing by open slots) — a noted alternative to explore later, not built now.
- **Pure power-law** multiplier family (`ratio^k`) — a noted alternative to explore later, not built now.
- Adopting a default hero (the September strategy decision stands).
- Registering the sweep variants into `tournament_cli._MODELS` (the default field stays clean).

## Chosen approach

### The bid

A new frozen dataclass `StackRatioBid` in `bid_strategy.py`:

```python
@dataclass(frozen=True)
class StackRatioBid:
    gain: float = 1.0
    curve: float = 2.0
    pace: float = 2.0
```

`max_bid(view, player, pool, config)`:

```
fair       = view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"]
per_slot   = my_budget / max(1, my_open_slots)
opp_mean   = (sum(budgets_by_seat) - my_budget) / max(1, n_teams - 1)   # mean opponent budget
ratio      = my_budget / max(opp_mean, min_bid)
mult       = 1.0 + gain * max(0.0, ratio - 1.0) ** curve                # >= 1.0; 1.0 when not ahead
target     = fair * mult
cap        = pace * per_slot * mult
return round(min(target, cap))                                          # engine clamps to feasible_max
```

- **Reduces to `balanced` exactly** when `ratio ≤ 1` (or `gain = 0`): `mult = 1` → `round(min(fair, pace·per_slot))`, byte-identical to `BalancedValueBid(premium=0.0, pace=pace)`.
- **Both** the target and the cap are lifted by `mult`. Lifting only the cap would help only where the cap binds (studs) → chase studs; lifting only the target would help only where the target binds (cheap players) but leaves the cap blocking; lifting both, combined with the convex-timing (below), deploys onto whatever is nominated *when the hero is dominant* — which is depth, late.
- `opp_mean` uses clean arithmetic `(Σ budgets − my_budget)/(n−1)` — no value-matching removal (this also avoids the `opp.remove(my_budget)` altitude smell noted on `BigStackBid`).

### Why `curve` (convexity) is the mechanism

`curve` controls how fast aggression ramps with the ratio. With `gain = 1`:

| ratio | `curve=1` (linear ≈ field_avg) | `curve=2` (convex) | `curve=3` |
|---|---|---|---|
| 1.14 (200/175) | mult 1.14 | mult **1.02** | mult **1.003** |
| 2.0 (200/100 or late 80/40) | mult 2.0 | mult **2.0** | mult **2.0** |

Convex curves (`curve>1`) keep the multiplier ≈ 1 at **moderate** leads (so the hero does **not** chase early studs when the field has merely spent a little — the exact failure that sank `field_avg`) and only unleash at a **dominant** ratio, which occurs late when depth is what remains. `curve=1` recovers the linear `field_avg`-style multiplier as the baseline the convex curves must beat.

### The sweep & measurement

- Phase 3 races a **dedicated contestant dict** (NOT `_MODELS`): `balanced = BalancedValueBid(premium=0.0)` (control) + `StackRatioBid(gain=g, curve=c)` for `g ∈ {0.5, 1.0, 2.0}` × `c ∈ {1, 2, 3}` (9 variants + control = 10 contestants).
- A/B under ADP nomination (`market_adp_jitter=12`), seat-averaged over 12 seats × both markets, 20 seeds × 300 sims — the Run-P/Q methodology, so the lift is attributable to the bid.
- **Roster-shape readout (separate trace run):** `run_auction_tournament` (the win-rate sweep) does NOT expose the `PickRecord` trace, so roster shape comes from a **separate** `_simulate_to_state`-with-`trace` analysis, on a **bounded** contestant set — `balanced` (control) + `StackRatioBid(curve=1)` (the linear baseline) + the **best convex variant** the sweep surfaces — at a representative hero seat across the sweep seeds, both markets. It reports each of those three heroes' **spend share by draft quartile** (early-stud vs late-depth) and **top-5 spend concentration** (share of budget on the 5 most-expensive buys), mirroring `espn_overspend_trace.py`. This answers the user's actual question — what ratio → multiplier relationship, and does convexity defer spend to depth — not just "which wins."

### Interpretation (data-gathering — no adopt bar)

No pre-registered adopt/reject threshold (the strategy decision is September). The deliverable is the *characterized* `(gain, curve)` surface: for each variant, the seat-averaged `reg_win_pct` delta vs `balanced` in **both** markets (flag whether it clears the ~±0.03 seed-noise band), plus the roster-shape evidence for whether convex curves genuinely defer spend from early studs to late depth. A convex curve that beats both `balanced` and the linear `curve=1` baseline in the less-circular ESPN market is the "it works" signal; a wash/negative is an equally valid recorded result.

## Requirements

- **R1 — fallback identity.** At `ratio ≤ 1` (hero not ahead) OR `gain = 0`, `StackRatioBid(gain, curve, pace).max_bid(...)` returns exactly `BalancedValueBid(premium=0.0, pace=pace).max_bid(...)` — compared against `BalancedValueBid`'s default `non_increasing_cap=False` (which `StackRatioBid`'s plain `per_slot = my_budget/max(1, open_slots)` mirrors) — on shared views (unit test).
- **R2 — convexity & monotonicity.** For `ratio > 1`, `mult` is strictly increasing in `ratio`; and for `curve > 1`, `mult` at a moderate ratio (e.g. 1.14) is strictly less than the `curve=1` linear multiplier, while at `ratio = 2` both equal `1 + gain` (unit test).
- **R3 — ratio definition.** The ratio uses the **mean opponent** remaining budget `(Σ budgets_by_seat − my_budget)/(n_teams−1)`, not a per-slot figure. A unit test constructs a view with known `budgets_by_seat` (e.g. hero 200, opponents averaging 100 → ratio 2.0) and asserts the resulting `mult`/bid.
- **R4 — solvency.** `StackRatioBid` returns a finite desired bid; the engine clamps to `feasible_max`, so the hero always fills a legal, full roster (engine-invariant + a `simulate_auction` smoke).
- **R5 — bid held-elsewhere-fixed.** The A/B changes only the bid strategy; nomination (`market_adp_jitter=12`), bot field, markets, seeds, sims match Run P/Q.
- **R6 — gates.** `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check` clean.
- **R7 — validated params.** `__post_init__` rejects a non-finite/negative `gain`, a non-finite/non-positive `curve`, and a non-finite/non-positive `pace` (construction-time `ValueError`), consistent with `BalancedValueBid`/`BigStackBid`.
- **R8 — mechanism visibility.** Phase 3 reports roster shape (spend by draft quartile / top-N concentration), not only win rates, so the ratio→multiplier→roster-shape story is observable.

## Edge cases / failure modes

- **`ratio ≤ 1` (short or even stack).** `max(0, ratio−1) = 0` → `mult = 1` → exactly `balanced`. The hero never bids *below* balanced when behind (no self-handicap); it just isn't aggressive.
- **`curve = 0`.** `0 ** 0 = 1` in Python would make `mult = 1 + gain` even at `ratio = 1` (a step, not a ramp) — nonsensical; `__post_init__` requires `curve > 0` (R7). The sweep uses `curve ∈ {1,2,3}`.
- **All opponents broke (`opp_mean → 0`).** `ratio = my_budget / max(opp_mean, min_bid)` is large → large `mult`, but second-price still clears uncontested players at `min_bid` and `feasible_max` clamps any single bid. Documented, not special-cased.
- **`n_teams` division.** `opp_mean` divides by `max(1, n_teams−1)`; `LeagueConfig.n_teams > 1` guarantees ≥ 1 opponent, so the guard is defensive only.
- **`my_open_slots == 0`** cannot occur (the engine only calls `max_bid` for a seat with an open slot); `max(1, ...)` guards the division regardless.

## Testing expectations

- `ratio ≤ 1` and `gain = 0` ⇒ output equals `BalancedValueBid(premium=0.0, pace)` — R1.
- convex vs linear: `mult(curve=2) < mult(curve=1)` at ratio 1.14, equal at ratio 2.0 — R2.
- known-budget view (hero 200, opps mean 100) ⇒ ratio 2.0 ⇒ expected `mult`/bid — R3.
- `__post_init__` rejects bad `gain`/`curve`/`pace` (incl. `curve=0`, NaN, inf) — R7.
- a full `simulate_auction` with `StackRatioBid` yields a legal, full roster — R4.

## Phasing

- **Phase 1 — the strategy.** `StackRatioBid` + `__post_init__` validation + unit tests R1/R2/R3/R7.
- **Phase 2 — wiring.** `simulate_auction` smoke that `StackRatioBid` fills a legal roster (R4).
- **Phase 3 — the sweep.** Crash-safe seat sweep of the `(gain, curve)` grid + `balanced` control under ADP nomination, both markets; aggregate seat-avg `reg_win_pct` (+ playoff/champ) deltas vs `balanced`; **plus** the roster-shape trace readout (spend by draft quartile / concentration). Write up as a Run in `reports/auction_tournament_validation_2026.md` + memory (data, no adoption).
