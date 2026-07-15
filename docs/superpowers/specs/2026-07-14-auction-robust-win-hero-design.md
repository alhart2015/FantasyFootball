# Auction — Robust Win-Maximizing Bid Hero

**Date:** 2026-07-14
**Status:** Design — awaiting user review
**Branch:** `feat/auction-robust-win-hero`
**Author:** Claude (brainstormed with Alden)

**Related:**
- `reports/auction_tournament_validation_2026.md` — Runs I/J (the hero-vs-bot gap), the standing experiment log
- `docs/superpowers/specs/2026-07-13-auction-balanced-value-design.md` — `BalancedValueBid`, which this extends
- TODO #49 (auction bid-model investigation); memory `auction-bid-model-investigation-status`

---

## Goal

Design a **single hero bid strategy** that maximizes **regular-season win% (`reg_win_pct`)** over a
Monte-Carlo bake-off in a **12-team half-PPR** field, and is **robust across both bot markets**
(`model`-priced and `espn`-anchored) *without knowing which market it faces*.

**Deliverable:** one shipped config — a single parameter set — judged by its **worst-case
`reg_win_pct` across the two markets**. Not two market-specific tunes (you can't observe the market
on draft day).

This is Slice 1 of a two-slice effort. Slice 2 (nomination poisoning) is a separate spec, written
after Slice 1's numbers are in.

---

## Background — the diagnosed bug

The best hero to date (`BalancedValueBid`, "balanced") tops out around `reg_win_pct` well below the
strongest bot archetype (`BalancedBot`, ~0.68 playoff). The gap is **not** currency and **not**
measurement noise — it is a **cap self-inflation effect**, diagnosed in the memory
`auction-bid-model-investigation-status` and visible directly in the code:

`BalancedValueBid.max_bid` (`src/projections/draft/assistant/auction/bid_strategy.py:289`):

```python
cap = self.pace * (view.my_budget / max(1, view.my_open_slots))
return round(min(fair * (1.0 + self.premium), cap))
```

When a breadth hero wins a player *cheaper than its current per-slot share*, `budget / open_slots`
**rises** (budget drops slower than the ratio's denominator), so the cap **ratchets up** over the
draft. The very strategy that wins by hoovering cheap mid-tier thereby inflates its own late-draft
cap → overpays late → builds a lopsided (RB-heavy) roster. The identical formula in `BalancedBot`
(`market.py:195`) scores ~0.60 as a *bot* but the same policy scores ~0.21 as the *measured hero* at
the nominate-first seat — the inflation only bites the seat we grade.

**Root fix:** stop the cap from ever rising above the opening per-slot pace.

---

## Non-goals

- **Not** the September strategy decision. This stays data-gathering; no default is *committed* to
  the live draft assistant. We ship the fix + record the data; the adopt call is the user's in
  September (project policy).
- **Not** nomination poisoning — that is Slice 2 (its own brainstorm → spec), built only after we
  measure how much Slice 1 alone closes the gap.
- **Not** 16-team. 12-team is this league's auction size (per the standing investigation).
- **Not** changing bot archetypes, the market model, `generate_auction_values`, or
  `espn_anchored_bot_prices`. The bots and both markets are held fixed so a measured hero delta is
  attributable to the hero.
- **Not** the #52 real-league-config re-validation. These numbers inherit the standing "provisional
  until the real config is locked" caveat; that is out of scope here.

---

## Approach

### 1. The fix — non-increasing cap (design decision C)

Add a **non-increasing cap** mode to `BalancedValueBid`. The cap may still *retreat* when the hero is
short on cash, but it can **never exceed the opening per-slot pace**:

```
cap = pace * min(my_budget / my_open_slots, budget0 / roster_size)
```

where `budget0 / roster_size` is the league's opening per-slot share (constant: `200 / 17 ≈ 11.76`
for 12-team half; at `pace = 2` that is the ~$24 opening cap Run J already found best). This matches
the diagnosis precisely — "self-inflates as the hero wins" — while preserving the safe downside
(bargain-hunt harder when broke). `budget0` and `roster_size` come from `LeagueConfig`
(`config.budget`, `config.roster_size`); no new inputs.

**Implementation shape:** a boolean field on `BalancedValueBid`, `non_increasing_cap: bool = False`.
The default is **`False` on purpose** — a bare `BalancedValueBid()` keeps the current inflating
behavior, so the existing `balanced` contestant is unchanged and stays byte-comparable to Runs I/J
(the control). The new fixed candidate passes `non_increasing_cap=True` explicitly. Flipping the
class default (or renaming the shipped default to the fixed behavior) is a **deferred, post-result**
change — separate from this slice and from the September strategy call. The engine's `[min_bid,
feasible_max]` clamp is unchanged and still bounds the final bid.

### 2. Contestants for the bake-off

Registered in `tournament_cli.py::_MODELS`:

- **`balanced_flat`** — `BalancedValueBid(non_increasing_cap=True)` — the primary Slice-1 candidate.
- **`balanced`** — the existing inflating-cap `BalancedValueBid` — kept as the control (the A/B).
- **`patient_deep`**, **`vorpshare`**, **`static`** — retained as the standing reference heroes
  (patient_deep is the multi-year breadth leader; the others anchor the field).

The other existing models stay registered (no removals); the report table already tracks them.

### 3. Re-tune `pace` + `premium` in both markets

Run J tuned `pace`/`premium` for the *inflating* cap; the optimum can shift once the cap can't
inflate. Run a **small** grid on `balanced_flat`:

- `pace ∈ {1.0, 1.5, 2.0, 2.5}` (opening caps ≈ $12 / $18 / $24 / $30)
- `premium ∈ {0.5, 1.0, 1.5}`

in **both** markets (`--bot-prices model` and `--bot-prices espn`), 12-team half, single-season 2026
(fast iteration). Pick the `(pace, premium)` with the best **worst-case `reg_win_pct`** across the
two markets. Keep the grid this small to avoid over-fitting to one season.

### 4. Measurement & success criterion

- **Primary metric:** `reg_win_pct` (the goal). Secondary: `make_playoffs_pct`, `champ_pct`,
  `mean_points` — reported, not optimized.
- **Field:** 12-team half-PPR (`data/vorp_2026/half_12team.parquet` +
  `configs/half_12team.league.json`), realistic mixed bot field (`_REALISTIC_FIELD` =
  Aggressive/Patient/Balanced), `nomination_temp = 1.0`.
- **Both markets, every run:** `--bot-prices model` and `--bot-prices espn`.
- **Tune fast, validate robust:** grid-search on single-season 2026; then **validate the finalist
  config multi-year (2021–2026)** on both markets using the per-season preset tables
  (`data/vorp_{2021..2026}/half_12team.parquet`), averaging per-model metrics — the reliability
  standard the project already uses.
- **Robust winner = best worst-case `reg_win_pct` across the two markets** at the multi-year level.
  Report per-market and combined; state the paired diff vs `balanced` (control) and vs `patient_deep`.
- **Reference line:** 12-team fair share `reg_win_pct ≈ 0.50`. Beating it = above-average team;
  we additionally note the distance to the `BalancedBot` tier as the "beat the best bot" bar.

### 5. Crash safety

The dev box has the confirmed Raptor Lake fault (memory `h2h-backtest-native-crash`). Runs go through
a **chunked / seed-ranged runner** (disjoint `--seed` base offsets across processes, aggregate the
per-seed arrays), not one long process. The tournament is already CRN-keyed on `base_seed + s`, so
chunking is exact and reproducible. A recent 73-min auction bake-off ran clean, but chunking stays
the default (the chip is unchanged; RMA pending).

---

## Testing

- **Unit tests** for the non-increasing cap on `BalancedValueBid`:
  - the cap is `≤ pace × budget0/roster_size` at every state (never inflates);
  - equals the old behavior on the *first* pick (opening state), diverges only after a
    below-share win;
  - retreats below the opening cap when the hero is broke (downside preserved);
  - `non_increasing_cap=False` reproduces the current inflating cap byte-for-byte (control unchanged);
  - `__post_init__` validation (`pace > 0`, `premium ≥ 0`, finite) still holds.
- **Gates** (CLAUDE.md end-of-effort checklist): `pytest -v` (relevant subset stated), `mypy src
  tests`, `ruff check`, `ruff format --check`. Touching a bid strategy is not a schema/ingest change,
  but the tournament path is exercised by the auction tests — run `-k "auction"` too.
- **No test edited to pass code.** If an existing auction test must change, state why and get
  confirmation (CLAUDE.md).

---

## Slice 2 — nomination poisoning (follow-on, NOT built here)

Documented so the sequence is legible; gets its own brainstorm → spec after Slice 1's numbers land.

Today the hero has **zero nomination control**: when `state.nominator == hero0`, the engine picks the
nominee via `_sample_nominee` (`simulation.py:197`); only *broke bots* steer their own nominations
(the snake-board path). Slice 2 adds a **hero-nomination hook**: an optional `NominationStrategy`
threaded through `simulate_auction` → `run_auction_tournament` → the CLI, consulted when the hero is
the (flush) nominator; bots unchanged; default preserves today's behavior byte-for-byte.

A **`poison` nominator** nominates high-`bot_dollars` players the hero does *not* want (positions it
has filled / deprioritized), early, to drain opponents' budgets so the hero's real targets clear
cheaper later. Slice 2 measures the **marginal `reg_win_pct` lift** of pairing this nominator with
Slice 1's best bid config, in both markets. Open design questions (for that spec): poison-target
selection (max `bot_dollars`, or max `bot_dollars − hero_value` overpay), poison-vs-nominate-my-own-
target timing, and whether poisoning helps at all against a value-rational bot field.

---

## Open questions

1. **Seat sensitivity.** The diagnosis is partly a *seat-role* effect (nominate-first seat 0). Tune
   at seat 1 (draft default) but spot-check one interior seat (e.g. 6) to confirm the fix isn't
   seat-specific. Not a full seat sweep (that's a separate axis).
2. **Tuning pool.** Single-season 2026 for the grid; is that representative enough before the
   multi-year validation? Mitigated by validating the finalist multi-year.
3. **Does the fix close the whole gap?** If `balanced_flat` still trails the `BalancedBot` tier after
   the fix + tune, that residual is what Slice 2 (poisoning) targets — this spec quantifies it, it
   does not promise to erase it.
