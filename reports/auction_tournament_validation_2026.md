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
  - **Exp season pts** — `SeatProjection.mean_points` (per-seat mean over sims of regular-season
    points-for, weeks 1–13).
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

## Fixed setup (defaults)

| Knob | Default | Notes |
|------|---------|-------|
| Pool | `data/consensus_vorp_2026.parquet` (or preset table) | `VorpTableSchema`; `consensus_adp` not required for auctions |
| Baseline dollars | `generate_auction_values(pool, config)` | computed once per run, threaded in |
| League | preset `(scoring, n_teams)` from the board registry | budget / min_bid / roster shape all from `LeagueConfig` |
| `price_jitter` | 0.15 (`DEFAULT_PRICE_JITTER` in `src/projections/draft/assistant/auction/market.py`) | market-noise knob (auction analog of `adp_jitter`) |
| `my_seat` | 1 (CLI required; no default) | seat sweep is a planned axis |
| `base_seed` | 0 | reproducibility |
| `n_sims` (season MC) | 500 (CLI default `--n-sims`) | 2000 reserved for post-hoc deep-dives |
| Scoring | `project_draft` (league sim) | StartersValuer dropped |

## Experiment log

> One row per (run, bid model). Record the config so each run is reproducible from this table alone.
> "Favored" = what *this run* leaned toward, in isolation — not a standing conclusion.

| Date | Preset (scoring×size) | Seat | Seeds | n_sims | price_jitter | Model | Exp pts (95% CI) | Win % | Playoff % | Bye % | Champ % | Notes / what it favored |
|------|------------------------|------|-------|--------|--------------|-------|------------------|-------|-----------|-------|---------|--------------------------|
| 2026-06-17 | half × 16 | 1 | 150 | 500 | 0.15 | static    | 892.6 [881.3, 903.2] | 0.38 | 0.15 | 0.03 | 0.01 | Run A — see note ↓ |
| 2026-06-17 | half × 16 | 1 | 150 | 500 | 0.15 | marginal  | 876.3 [872.1, 880.2] | 0.37 | 0.11 | 0.02 | 0.01 | Run A |
| 2026-06-17 | half × 16 | 1 | 150 | 500 | 0.15 | inflation | 855.3 [847.2, 863.4] | 0.35 | 0.09 | 0.01 | 0.01 | Run A |
| 2026-06-17 | half × 16 | 1 | 150 | 500 | 0.15 | static    | 882.3 [869.9, 894.5] | 0.34 | 0.10 | 0.02 | 0.01 | Run B (sane bots) — see note ↓ |
| 2026-06-17 | half × 16 | 1 | 150 | 500 | 0.15 | marginal  | 864.5 [858.8, 870.3] | 0.32 | 0.06 | 0.01 | 0.00 | Run B (sane bots) |
| 2026-06-17 | half × 16 | 1 | 150 | 500 | 0.15 | inflation | 818.2 [809.3, 827.0] | 0.28 | 0.04 | 0.00 | 0.00 | Run B (sane bots) |

**Run A — 2026-06-17** (`half_16team`, seat 1, 150 seeds, n_sims=500, price_jitter=0.15, budget=200, seed=0; **byes OFF** — 2026 schedule not ingested, injury risk still applied).

Paired per-seed differences (point [95% CI]; all exclude 0 except where noted) — the trustworthy signal:
- **static − inflation:** +37.3 pts [+23.1, +52.3] · playoff +0.058 [+0.042, +0.074] · bye +0.015 [+0.011, +0.019] · champ +0.006 [+0.004, +0.009]
- **static − marginal:** +16.3 pts [+3.8, +28.4] · playoff +0.045 [+0.028, +0.062] · bye +0.013 [+0.008, +0.018] · champ +0.006 [+0.003, +0.008]
- **inflation − marginal:** −21.0 pts [−30.2, −11.3] · playoff −0.013 [−0.021, −0.004] · bye ≈ 0 · champ ≈ 0

*In isolation this run favors* **static ≥ marginal ≥ inflation** *on points & playoff%. Not a verdict (one seat, one preset, byes off; decide in September).*

⚠️ **Methodology flag (read before trusting levels):** the hero's **absolute** metrics sit *below* the uniform baseline (playoff 0.15 vs 0.375 expected at 6/16; champ 0.01 vs 0.0625). With 15 noisy-WTP bots, the *order-statistic max* of their bids on a stud exceeds `baseline_dollars`, so a value-bidding hero is outbid on the good players and ends with a below-average roster. This is the §3.6 bot-field caveat in action: **absolute numbers are bot-relative; only the paired diffs are interpretable.** Worth examining the bot model (and/or whether a competitive hero must bid above baseline) and a seat sweep before reading anything into levels.

---

**Run B — 2026-06-17** (`half_16team`, seat 1, 150 seeds, n_sims=500, price_jitter=0.15, budget=200, seed=0; **byes OFF** — 2026 schedule not ingested; **sane bots** — `bot_position_bounds` from `LeagueConfig` + shared `bot_eligible` iteration domain, branch `feat/auction-sane-bots`). Identical knobs to Run A; only the bot-field positional discipline changed.

Per-model metrics (mean [95% CI]):
- **static:** 882.3 pts [869.9, 894.5] · win 0.34 · playoff 0.10 · bye 0.02 · champ 0.01
- **marginal:** 864.5 pts [858.8, 870.3] · win 0.32 · playoff 0.06 · bye 0.01 · champ 0.00
- **inflation:** 818.2 pts [809.3, 827.0] · win 0.28 · playoff 0.04 · bye 0.00 · champ 0.00

Paired per-seed differences (point [95% CI]; all CIs exclude 0):
- **static − inflation:** +64.2 pts [+49.1, +78.8] · win +0.057 [+0.043, +0.071] · playoff +0.057 [+0.045, +0.071] · bye +0.012 [+0.009, +0.016] · champ +0.004 [+0.003, +0.006]
- **static − marginal:** +17.8 pts [+4.4, +31.1] · win +0.017 [+0.004, +0.030] · playoff +0.034 [+0.021, +0.049] · bye +0.009 [+0.005, +0.013] · champ +0.003 [+0.001, +0.004]
- **inflation − marginal:** −46.4 pts [−55.3, −37.5] · win −0.041 [−0.049, −0.032] · playoff −0.023 [−0.029, −0.017] · bye −0.003 [−0.004, −0.002] · champ −0.002 [−0.002, −0.001]

*Run A vs Run B comparison:* With positionally-disciplined bots (Run B), the hero's absolute metrics declined slightly vs Run A (static playoff 0.10 vs 0.15; champ 0.01 vs 0.01). The hero did **not** recover to the uniform baseline (0.375 playoff / 0.0625 champ) — the bot-field remains a competitive handicap even with positional discipline, because bots now spend their budgets more efficiently on positionally-appropriate targets. The paired-diff ordering is unchanged: **static ≥ marginal ≥ inflation** on both points and playoff% in both runs. The static−inflation advantage widened in Run B (+64.2 pts vs +37.3 pts), while static−marginal was similar (+17.8 vs +16.3 pts). In isolation, with sane bots, the hero's absolute numbers are lower because the disciplined bots are harder competition; the model ranking direction is the same; no winner declared (byes still off; September decision).

*In isolation this run also favors* **static ≥ marginal ≥ inflation** *on points & playoff%. Not a verdict.*

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

```
python scripts/auction_tournament.py \
    --vorp-table data/vorp_2026/half_16team.parquet \
    --league-config configs/league_espn_half_16team.json \
    --my-seat 1 --season 2026 \
    [--seeds K]          # default 200; use 20 for quick smoke, 100+ for tighter CIs
    [--price-jitter F]   # default 0.15
    [--seed BASE]        # default 0
    [--n-sims M]         # default 500; use 200 for speed, 2000 for deep-dives
    [--data-root PATH]   # default "data"; must have availability/byes partitions for 2026
    compare
```

`compare` races `static` / `inflation` / `marginal` against a shared seeded bot market and prints:
- each model's per-metric mean + 95% bootstrap CI table
- paired per-seed differences with CIs for every model pair

No winner is declared. Paste the per-model table into the experiment log above with the exact flags used.

**Swap the `--vorp-table` / `--league-config` pair together** — they must agree on scoring ruleset and
team count (the budget, roster shape, and replacement level are all derived from `LeagueConfig`).
Available preset pairs: `data/vorp_2026/{half,ppr,std}_{10,12,16}team.parquet` with
`configs/league_espn_{half_16team,ppr_12team,half_10team}.json` (or any custom `LeagueConfig` JSON).
