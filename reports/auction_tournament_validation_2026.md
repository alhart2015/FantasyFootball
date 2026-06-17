# Auction Draft — Bid-Model Experiment Tracking (2026)

**Status:** **Data-gathering.** We are *not* picking a winning bid model now. We run tests through the
summer and make the call in **September 2026**, close to the real draft. Every entry below records what a
single test *favors in isolation* — these are observations, not verdicts. Do not collapse the log into a
"the winner is X" claim before the September decision.

**Spec:** `docs/superpowers/specs/2026-06-17-auction-strategy-tournament-design.md`
**Companion (snake):** `reports/draft_tournament_validation_2026.md`, `reports/depth_aware_strategy_validation_2026.md`

## Purpose

Gather reproducible data on **which auction bid behavior produces the better hero roster** — bid to the
static Surplus-of-Surplus dollar, to an inflation-adjusted dollar, or to the player's marginal lineup
value. Each bid model is raced against a shared seeded bot market and the resulting **whole projected
league** is scored with the same projected-vs-projected simulation we use to evaluate snake drafts
(`project_draft`). The output of every run is a row (or set of rows) in the experiment log, not a decision.

## The contestants (bid models)

Same mechanism for all three (shared nominator, shared bot field, same `feasible_max` reserve clamp); the
**only** thing that varies is the valuation, so a measured difference is attributable to the bid model.

- **`static` — `StaticDollarBid` (control).** `max_bid = baseline_dollars[player]`. Bid straight to the
  static SOS auction value. It is exactly "trust `generate_auction_values`." No state.
- **`inflation` — `InflationBid`.** Re-price the static dollar by live **surplus inflation**
  (`remaining_surplus_money / remaining_surplus_value`, in the same surplus units `auction.py` uses):
  disciplined late when the room overspends early, premium on the remaining studs when the room
  underspends. One scalar re-pricing the whole board, recomputed each nomination.
- **`marginal` — `MarginalValueBid`.** Bid to the player's **marginal optimal-lineup lift**
  (`StartersValuer`-style starters surrogate) converted to dollars at the live market exchange rate. A
  player who doesn't crack the hero's starting lineup bids `min_bid`. **Known limitation:** the surrogate
  is single-week-optimal and availability-blind, so it under-values depth — the depth-aware
  (`SeasonValuer`-marginal) form is the deferred *live-assistant* model, not a tournament contestant.

## How we score (same as the projected draft eval)

- **Scoring = full-league projected-vs-projected sim**, not a single-roster scalar. The auction sim retains
  **every seat's full roster** (it already builds them) and feeds them to
  `project_draft(rosters, pool, availability, params, …)` →
  `SeatProjection` per seat. **`StartersValuer`/single-roster scalar scoring is not used** — dropped by
  decision (no added value over the league sim).
- **Metrics recorded for the hero seat** (the data we gather):
  - **Exp season pts** — expected regular-season points *(requires a small `mean_points` addition to
    `SeatProjection` / `project_draft`; today it returns the four rates + `mean_seed` only)*.
  - **Win %** — `reg_win_pct` (regular-season H2H win rate)
  - **Playoff %** — `make_playoffs_pct` (finish top-6)
  - **Bye %** — `bye_pct` (finish top-2)
  - **Champ %** — `champ_pct`
- **Paired design.** Same `base_seed + s` ⇒ same bot market across models; **common random numbers** on the
  season MC (a season-rng keyed off the auction seed, shared across models) so the season-sim noise largely
  cancels in paired differences. We report each model's **mean + 95% bootstrap CI** for every metric, and
  the **paired per-seed differences** with CIs. These are recorded as data — **no winner is declared.**
- **Bot-field caveat (load-bearing here).** H2H/playoff/champ % are **relative to the noisy-WTP bot
  field**. The paired design keeps the *ranking* robust to bot mediocrity, but the *absolute* numbers are
  only as meaningful as the bots. Bot realism remains the single biggest realism lever (shared with the
  snake-side follow-up, TODO #46). Note this when reading absolute champ % across runs.
- **Power note.** Champ % is the noisiest metric (≈`1/n` base rate); expected points and playoff % separate
  far more readily at a given seed/sim budget. Don't over-read a champ % gap that the CIs don't support.

## Fixed setup (defaults — fill once the harness lands)

| Knob | Default | Notes |
|------|---------|-------|
| Pool | `data/consensus_vorp_2026.parquet` (or preset table) | `VorpTableSchema`; `consensus_adp` not required for auctions |
| Baseline dollars | `generate_auction_values(pool, config)` | computed once per run, threaded in |
| League | preset `(scoring, n_teams)` from the board registry | budget / min_bid / roster shape all from `LeagueConfig` |
| `price_jitter` | _TBD_ | market-noise knob (auction analog of `adp_jitter`) |
| `my_seat` | _TBD_ | seat sweep is a planned axis |
| `base_seed` | 0 | reproducibility |
| `n_sims` (season MC) | _~500, with CRN_ | 2000 reserved for post-hoc deep-dives |
| Scoring | `project_draft` (league sim) | StartersValuer dropped |

## Experiment log

> One row per (run, bid model). Record the config so each run is reproducible from this table alone.
> "Favored" = what *this run* leaned toward, in isolation — not a standing conclusion.

| Date | Preset (scoring×size) | Seat | Seeds | n_sims | price_jitter | Model | Exp pts (95% CI) | Win % (CI) | Playoff % (CI) | Bye % (CI) | Champ % (CI) | Notes / what it favored |
|------|------------------------|------|-------|--------|--------------|-------|------------------|------------|----------------|------------|--------------|--------------------------|
| _—_  | _—_                    | _—_  | _—_   | _—_    | _—_          | _—_   | _(no runs yet — harness not built)_ | | | | | |

## Planned experiments / axes to sweep

- **Bid-model bake-off** (the core): `static` vs `inflation` vs `marginal`, all three metrics, at a fixed
  preset + seat + `price_jitter`.
- **`price_jitter` sweep** — how the ranking moves as the bot market gets noisier (analog of the snake
  `tune-sigma`).
- **Seat sweep** — repeat across seats; check whether any model is seat-sensitive (auction analog of the
  snake hero-slot sweep).
- **Scoring × size presets** — re-run under half-PPR / PPR / standard × {10, 12, 16}; auction values and
  inflation dynamics shift with budget/size, so the favored model may too.
- **Valuer cross-check** — score the same rosters under `season` vs the cheaper season-MC budgets to see
  how depth-blindness in v3's *bid* shows up in the *graded* season metric.

## Open questions / future slices

- **Strategic nomination** is a separate axis (deferred): dump-the-enemy's-budget, price-enforcement,
  nominate-your-target-late. Once the bid data matures, nominators become the contestants.
- **Better bot market** — a behavior-calibrated WTP model so absolute numbers reflect a real room.
- **Full-fidelity v3** — availability-aware `SeasonValuer`-marginal, reserved for the live auction
  assistant (evaluated once per real pick).

## Reproduce

_Recipe goes here once `scripts/auction_tournament.py` exists. Expected shape:_

```
python scripts/auction_tournament.py \
    --vorp-table <consensus_vorp.parquet> --league-config <league.json> \
    --my-seat N [--seeds K] [--price-jitter F] [--seed BASE] \
    [--n-sims M] [--season YYYY] \
    compare
```

`compare` runs the three bid models and prints each model's per-metric mean + CI table and the paired
differences. Paste the numbers into the log above with the exact flags used.
