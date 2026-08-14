# Auction strategy tournament — validation log (2026)

> ## ⚠️ Runs A–X ran under a roster-bounds bug ([#143](https://github.com/alhart2015/FantasyFootball/issues/143))
>
> `bot_position_bounds` anchored every FLEX slot to **RB**, which capped every seat at **4 WR**
> (vs 7 RB). That cap bound on **98–99%** of all rosters, so no team in any run below could field a
> receiver-heavy roster. The **valuation layer was never affected** — `vorp.py` allocates FLEX by
> actually filling the slots, and had WR absorbing *more* of it than RB (3.14 vs 2.82 starters/team)
> — so auction dollars, the cheat sheet, and every price in this log are unchanged. What moved is
> every *simulated outcome*: win%, playoff%, champ%, and the roster shapes behind them.
>
> **Run Y (below) re-runs the load-bearing experiments on the fixed engine.** Headline: the hero
> choice is unchanged (`overbid_noramp` still wins Will's room), but levels shift by ~0.01–0.02 and
> one of Run V's conclusions is retracted. Treat Runs A–X as directionally informative and their
> absolute figures as superseded.

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
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | overbid   | 1205.1 [1195.5, 1215.7] | 0.65 | 0.72 | 0.37 | 0.22 | Run C (pos-aware hero) — see note ↓ |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | anchors   | 1154.4 [1145.7, 1162.6] | 0.61 | 0.62 | 0.28 | 0.14 | Run C |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | vorpshare | 977.8 [970.9, 984.3] | 0.43 | 0.22 | 0.05 | 0.02 | Run C |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | static    | 965.4 [956.3, 976.3] | 0.42 | 0.20 | 0.04 | 0.02 | Run C (now gated) |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | marginal  | 903.1 [897.5, 908.7] | 0.36 | 0.10 | 0.01 | 0.01 | Run C (now gated) |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | inflation | 867.8 [862.2, 873.3] | 0.33 | 0.06 | 0.01 | 0.00 | Run C (now gated) |
| 2026-06-18 | half × 12 | 1 | 60 | 200 | 0.15 | overbid   | 1561.3 [1553.0, 1569.1] | 0.68 | 0.87 | 0.48 | 0.27 | Run D (pos-aware hero) — see note ↓ |
| 2026-06-18 | half × 12 | 1 | 60 | 200 | 0.15 | anchors   | 1529.5 [1521.7, 1536.8] | 0.65 | 0.83 | 0.41 | 0.20 | Run D |
| 2026-06-18 | half × 12 | 1 | 60 | 200 | 0.15 | static    | 1461.4 [1454.1, 1469.1] | 0.60 | 0.72 | 0.27 | 0.14 | Run D |
| 2026-06-18 | half × 12 | 1 | 60 | 200 | 0.15 | vorpshare | 1460.9 [1455.3, 1467.0] | 0.59 | 0.72 | 0.25 | 0.14 | Run D |
| 2026-06-18 | half × 12 | 1 | 60 | 200 | 0.15 | inflation | 1424.5 [1419.9, 1428.9] | 0.56 | 0.65 | 0.20 | 0.11 | Run D |
| 2026-06-18 | half × 12 | 1 | 60 | 200 | 0.15 | marginal  | 1409.0 [1404.2, 1413.9] | 0.55 | 0.62 | 0.18 | 0.09 | Run D |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | static    | 806.6 [778.6, 835.1] | 0.35 | 0.14 | 0.03 | 0.02 | Run E (realistic mkt) — see note ↓ |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | patient   | 800.8 [776.0, 826.9] | 0.35 | 0.10 | 0.02 | 0.01 | Run E (realistic mkt) |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | overbid   | 760.6 [735.3, 784.6] | 0.31 | 0.08 | 0.02 | 0.01 | Run E (realistic mkt) |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | vorpshare | 655.2 [617.0, 691.3] | 0.23 | 0.05 | 0.01 | 0.00 | Run E (realistic mkt) |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | anchors   | 640.0 [615.4, 665.2] | 0.21 | 0.02 | 0.00 | 0.00 | Run E (realistic mkt) |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | inflation | 556.2 [514.0, 601.9] | 0.16 | 0.03 | 0.01 | 0.00 | Run E (realistic mkt) |
| 2026-06-18 | half × 16 | 1 | 60 | 200 | 0.15 | marginal  | 519.7 [499.9, 540.9] | 0.12 | 0.00 | 0.00 | 0.00 | Run E (realistic mkt) |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | inflation  | 810.4 [781.7, 837.3] | 0.36 | 0.12 | 0.02 | 0.01 | Run F (urgency + studsdepth) — see note ↓ |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | vorpshare  | 801.9 [767.4, 837.2] | 0.35 | 0.13 | 0.03 | 0.01 | Run F (urgency) |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | static     | 800.3 [774.3, 824.9] | 0.35 | 0.11 | 0.02 | 0.01 | Run F (urgency) |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | patient    | 800.3 [778.6, 819.5] | 0.35 | 0.09 | 0.01 | 0.00 | Run F (urgency) |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | marginal   | 787.7 [758.4, 814.9] | 0.34 | 0.08 | 0.01 | 0.01 | Run F (urgency) |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | overbid    | 769.0 [738.5, 799.4] | 0.32 | 0.09 | 0.02 | 0.01 | Run F (urgency) |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | studsdepth | 767.0 [741.7, 792.9] | 0.32 | 0.08 | 0.01 | 0.01 | Run F (new contestant) |
| 2026-06-18 | half × 16 | 1 | 40 | 200 | 0.15 | anchors    | 673.8 [646.1, 699.7] | 0.23 | 0.03 | 0.00 | 0.00 | Run F (urgency) |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | vorpshare  | 1213.2 [1171.1, 1252.8] | 0.48 | 0.44 | 0.12 | 0.05 | Run G (12-team urgency) — see note ↓ |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | patient    | 1170.1 [1147.5, 1194.0] | 0.45 | 0.39 | 0.04 | 0.01 | Run G (urgency) |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | inflation  | 1102.0 [1063.8, 1135.4] | 0.38 | 0.27 | 0.03 | 0.01 | Run G (urgency) |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | static     | 1095.9 [1055.8, 1129.6] | 0.37 | 0.26 | 0.04 | 0.01 | Run G (urgency) |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | studsdepth | 1079.1 [1043.1, 1112.0] | 0.37 | 0.25 | 0.03 | 0.01 | Run G (new contestant) |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | overbid    | 1063.6 [1030.9, 1092.1] | 0.35 | 0.21 | 0.02 | 0.01 | Run G (urgency) |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | marginal   | 1061.1 [1030.0, 1095.3] | 0.35 | 0.20 | 0.02 | 0.01 | Run G (urgency) |
| 2026-06-18 | half × 12 | 1 | 40 | 200 | 0.15 | anchors    | 971.0 [945.3, 998.0] | 0.27 | 0.10 | 0.01 | 0.00 | Run G (urgency) |

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

---

**Run C — 2026-06-18** (`half_16team`, seat 1, 60 seeds, n_sims=200, price_jitter=0.15, budget=200, seed=0; **byes OFF**; **position-aware hero** — every bidder, hero included, now obeys the shared `bot_eligible`/`bot_position_bounds` rule, branch `feat/auction-stars-and-scrubs`). Six contestants: the three value-anchored models (now position-gated) plus three budget-spending **stars-and-scrubs** models — `anchors` (AnchorBudgetBid, N=4), `overbid` (OverbidValueBid, k=1.3), `vorpshare` (VorpShareBid).

Seed/sim counts are lower than Runs A/B (60×200 vs 150×500): the i9-14900KF Raptor Lake fault segfaults the 6-model run at 150×300 in one process (memory `h2h-backtest-native-crash`); 60×200 completes cleanly. CIs are wider but the separations are large.

Per-model metrics (mean [95% CI on exp pts]; uniform baseline playoff 0.375 / champ 0.0625):
- **overbid:** 1205 [1195.5, 1215.7] · win 0.65 · **playoff 0.72** · bye 0.37 · **champ 0.22**
- **anchors:** 1154 [1145.7, 1162.6] · win 0.61 · **playoff 0.62** · bye 0.28 · champ 0.14
- **vorpshare:** 978 [970.9, 984.3] · win 0.43 · playoff 0.22 · bye 0.05 · champ 0.02
- **static:** 965 [956.3, 976.3] · win 0.42 · playoff 0.20 · bye 0.04 · champ 0.02
- **marginal:** 903 [897.5, 908.7] · win 0.36 · playoff 0.10 · bye 0.01 · champ 0.01
- **inflation:** 868 [862.2, 873.3] · win 0.33 · playoff 0.06 · bye 0.01 · champ 0.00

Paired per-seed playoff% differences (point [95% CI]; all exclude 0 except where noted):
- **overbid − anchors:** +0.092 [+0.068, +0.119] (overbid edges anchors)
- **overbid − static:** +0.513 [+0.484, +0.538] · **anchors − static:** +0.421 [+0.385, +0.452]
- **overbid − marginal:** +0.615 [+0.594, +0.635] · **overbid − inflation:** +0.652 [+0.633, +0.671]
- **anchors − vorpshare:** +0.405 [+0.377, +0.428] · **overbid − vorpshare:** +0.497 [+0.477, +0.517]
- **static − vorpshare:** −0.016 [−0.036, +0.004] (straddles 0 — indistinguishable)
- **static − marginal:** +0.102 [+0.080, +0.126] · **static − inflation:** +0.139 [+0.121, +0.160] · **inflation − marginal:** −0.037 [−0.047, −0.026]

*In isolation this run favors* **overbid > anchors ≫ vorpshare ≈ static > marginal > inflation** *on playoff% and points. Not a verdict (one seat, one preset, byes off, 60 seeds; decide in September).*

**What changed and what it means (data, not a decision):** the two stars-and-scrubs models that spend the full budget on a few high-VORP anchors (`overbid`, `anchors`) put the hero **well above** the uniform baseline (overbid playoff 0.72 / champ 0.22; anchors 0.62 / 0.14) — the first hero strategies to clear it. This is the converse of Runs A/B: there the value-bidding hero sat below a random team; here a budget-committing hero dominates the *same* bot field. The value-anchored trio stays below baseline even with the position gate (`static` 0.20, `marginal` 0.10, `inflation` 0.06); the gate's main visible effect on them is removing the empty-TE / over-stacked-WR rosters (`static` rose from Run B's 0.10 to 0.20). `vorpshare` (proportional VORP allocation) lands between — better than the timid trio, far below the two aggressive ones.

**Caveat:** absolute levels are **not** directly comparable to Runs A/B — gating the hero changes the nomination/award sequence and thus the points at which the auction RNG is consumed (the §3.6 bot-field caveat), and the bot WTP model is unchanged. The actionable read: a hero that commits its budget to anchors and bids *above* fair value to win them beats this bot field decisively. Which of `overbid`/`anchors` is best, how sensitive `overbid` is to `k` and `anchors` to `N`, and whether the edge holds across seats / presets / `price_jitter` are the next sweeps.

---

**Run D — 2026-06-18** (`half_12team` preset via `get_preset("half", 12)` → `configs/half_12team.league.json`; seat 1, 60 seeds, n_sims=200, price_jitter=0.15, budget=200, 17-man rosters QB1/RB2/WR3/TE1/FLEX1/BENCH9; **byes OFF**; position-aware hero). Same six contestants as Run C, in a shallower 12-team league. Fair share: champ ≈ 0.083, playoff ≈ 0.50.

Per-model metrics (mean [95% CI on exp pts]):
- **overbid:** 1561 [1553.0, 1569.1] · win 0.68 · playoff 0.87 · bye 0.48 · champ 0.27
- **anchors:** 1530 [1521.7, 1536.8] · win 0.65 · playoff 0.83 · bye 0.41 · champ 0.20
- **static:** 1461 [1454.1, 1469.1] · win 0.60 · playoff 0.72 · bye 0.27 · champ 0.14
- **vorpshare:** 1461 [1455.3, 1467.0] · win 0.59 · playoff 0.72 · bye 0.25 · champ 0.14
- **inflation:** 1424 [1419.9, 1428.9] · win 0.56 · playoff 0.65 · bye 0.20 · champ 0.11
- **marginal:** 1409 [1404.2, 1413.9] · win 0.55 · playoff 0.62 · bye 0.18 · champ 0.09

Paired playoff% (point [95% CI]; all exclude 0 except static≈vorpshare): overbid−anchors +0.040 [+0.026, +0.054]; overbid−static +0.143 [+0.126, +0.161]; anchors−static +0.103 [+0.082, +0.123]; static−vorpshare +0.006 [−0.014, +0.025] (ties); inflation−marginal +0.024 [+0.005, +0.043]; overbid−vorpshare +0.149 [+0.131, +0.168].

*In isolation favors* **overbid > anchors > static ≈ vorpshare > inflation > marginal.** Not a verdict.

**Cross-preset finding — the edge compresses in a shallower league.** Stars-and-scrubs still leads, but the gap over the value-anchored trio shrinks sharply vs Run C's 16-team field: `overbid − static` playoff is **+0.143 here vs +0.52 in 16-team**. In a 12-team league the pool is shallow enough that even `static`/`marginal`/`inflation` clear the ~0.50 playoff line (0.62–0.72); the deep 16-team field punishes mediocre drafting far harder. Read: **strategy choice matters most in deep leagues.** At the top, hero and the best bot are within MC noise on champ% (the 12-team top tier is tightly packed).

---

**Run E — 2026-06-18** (`half_16team`, seat 1, 60 seeds, n_sims=200, price_jitter=0.15, budget=200; **byes OFF**; **REALISTIC MARKET** — value-weighted-random nomination (`nomination_temp=1.0`) + a mixed bot field (⅓ aggressive / ⅓ patient value-hunter / ⅓ balanced); branch `feat/auction-realistic-market`). Seven contestants: the six prior models + the new `patient` (PatientValueBid). This run removes the $1-mid-tier artifact that Runs A–D's all-aggressive + value-descending market produced (Marvin Harrison no longer clears at $1). Fair share: playoff 0.375 / champ 0.0625.

Per-model metrics (mean [95% CI on exp pts]):
- **static:** 807 [778.6, 835.1] · win 0.35 · playoff 0.14 · champ 0.02
- **patient:** 801 [776.0, 826.9] · win 0.35 · playoff 0.10 · champ 0.01
- **overbid:** 761 [735.3, 784.6] · win 0.31 · playoff 0.08 · champ 0.01
- **vorpshare:** 655 [617.0, 691.3] · win 0.23 · playoff 0.05 · champ 0.00
- **anchors:** 640 [615.4, 665.2] · win 0.21 · playoff 0.02 · champ 0.00
- **inflation:** 556 [514.0, 601.9] · win 0.16 · playoff 0.03 · champ 0.00
- **marginal:** 520 [499.9, 540.9] · win 0.12 · playoff 0.00 · champ 0.00

Paired playoff% (point [95% CI]): static−patient +0.036 [−0.000, +0.071] (≈tied at top); static−overbid +0.055 [+0.025, +0.086]; overbid−patient −0.019 [−0.045, +0.006] (≈tied); static−anchors +0.115; anchors−overbid −0.060 (overbid > anchors); patient−marginal +0.100. In isolation: **static ≈ patient > overbid > vorpshare > inflation ≈ anchors > marginal.**

**Headline — the realistic market re-ranks everything (confirms the $1-stud artifact).** In Run C's rigged market `overbid`/`anchors` DOMINATED (playoff 0.72 / 0.62). Here, once mid-tier studs cost real money, they **collapse**: `overbid` 0.72→0.08 (#3), `anchors` 0.62→0.02 (near-last — its over-concentration on a few anchors is punished hardest when it can't backfill cheap). The "boring" strategies win: `static` (bid fair value) and the new `patient` (hold budget for mid-round value) are top, statistically tied. Stars-and-scrubs was an exploit of the broken market, not a durable edge.

**Caveat — absolute levels.** All seven sit *below* the uniform baseline (best `static` 0.14 vs 0.375). A realistic mixed field of 15 sane bots is hard for one seat, and absolute levels are **not** comparable to Runs A–D (different market model + nomination RNG). The interpretable signal is the within-run re-ranking. That all hero strategies trail a realistic field is itself a finding: beating a room of sane managers from one seat is genuinely hard — the open questions are whether a smarter hero (or a seat/preset sweep) closes the gap, and whether the mixed-bot field is calibrated against real prices. **No winner declared** (September).

**Run F — 2026-06-18** (`half_16team`, seat 1, **40 seeds** — reduced from 60 because the dev box's Raptor Lake fault crashed the 60-seed process mid-run; that is a hardware event, not a code bug (see memory `h2h-backtest-native-crash`) — n_sims=200, price_jitter=0.15, budget=200; **byes OFF**; **REALISTIC MARKET** identical to Run E (`nomination_temp=1.0` + mixed bot field ⅓ aggressive / ⅓ patient value-hunter / ⅓ balanced); branch `feat/auction-budget-urgency`). **Eight contestants** — the seven prior models, now all gated through the new `_budget_urgency` late-draft deployment factor (spec §A–B), plus the new `studsdepth` (StudsAndDepthBid, spec §C: a modest stud premium over fair value + fair-value mid-tier depth + $1 scrubs — the "good bot as hero"). Fair share: playoff 0.375 / champ 0.0625.

Per-model metrics (mean [95% CI on exp pts]), sorted by exp pts:
- **inflation:** 810 [781.7, 837.3] · win 0.36 · playoff 0.12 · champ 0.01
- **vorpshare:** 802 [767.4, 837.2] · win 0.35 · playoff 0.13 · champ 0.01
- **static:** 800 [774.3, 824.9] · win 0.35 · playoff 0.11 · champ 0.01
- **patient:** 800 [778.6, 819.5] · win 0.35 · playoff 0.09 · champ 0.00
- **marginal:** 788 [758.4, 814.9] · win 0.34 · playoff 0.08 · champ 0.01
- **overbid:** 769 [738.5, 799.4] · win 0.32 · playoff 0.09 · champ 0.01
- **studsdepth:** 767 [741.7, 792.9] · win 0.32 · playoff 0.08 · champ 0.01
- **anchors:** 674 [646.1, 699.7] · win 0.23 · playoff 0.03 · champ 0.00

Paired playoff% (point [95% CI]): the top three are statistically tied — static−inflation −0.015 [−0.042, +0.013], static−vorpshare −0.025 [−0.076, +0.019], static−patient +0.022 [−0.011, +0.055]. Significant gaps: inflation−marginal +0.039 [+0.006, +0.070], vorpshare−patient +0.047 [+0.011, +0.084], vorpshare−studsdepth +0.050 [+0.012, +0.095], and **everyone ≫ anchors** (static−anchors +0.076 [+0.048, +0.106]; inflation−anchors +0.091 [+0.057, +0.128]). In isolation: **inflation ≈ vorpshare ≈ static ≳ patient ≈ overbid ≈ marginal ≈ studsdepth ≫ anchors** — a tight, mostly-overlapping band with `anchors` the lone outlier.

**Headline — urgency compresses the field by rescuing the under-spenders.** The budget-urgency factor forces idle cash into bids as the roster fills, and the models it helps most are exactly the ones that hoarded budget in Run E. Comparing playoff% Run E→Run F: `inflation` 0.03→0.12, `marginal` 0.00→0.08 (both leapt from the bottom of Run E to mid-pack/top), `vorpshare` 0.05→0.13 (now top). The strategies that already deployed their budget barely moved (`static` 0.14→0.11, `patient` 0.10→0.09). The clear Run E ordering ("boring `static`/`patient` win, `marginal`/`inflation` lose") has **collapsed into a tight band** (top seven within playoff 0.08–0.13, all CIs overlapping). Only `anchors` is left behind — it already over-concentrates into a few anchors early, so urgency can't rescue a roster that's already all-in. This is the feature working as designed: deploying idle cash most helps the strategies that were sitting on it.

**studsdepth lands mid-pack — no winner.** The "good bot as hero" is statistically indistinguishable from `overbid` and `marginal` on both points and playoff%, trails `inflation` on both and `static`/`patient` on points significantly, and beats only `anchors`. It is competitive, not dominant — consistent with the standing finding that no single bid model separates from the pack under a realistic market.

**Caveats.** (1) 40 seeds (not 60) widen the CIs vs Run E, and absolute levels are **not** comparable to Runs A–E — urgency now changes every model's bids and the seed count differs, so the interpretable signal is the within-run re-ranking and the field compression, not the level. (2) The whole field still sits *below* the uniform baseline (best playoff 0.13 vs 0.375): beating a room of 15 sane bots from one seat remains hard, and urgency narrows the spread between hero strategies without lifting the seat above the field. **No winner declared** — data-gathering only; the strategy call is September 2026.

**Run G — 2026-06-18** (`half_12team`, seat 1, 40 seeds, n_sims=200, price_jitter=0.15, budget=200; **byes OFF**; **REALISTIC MARKET** identical to Runs E/F; **eight contestants** identical to Run F, incl. `_budget_urgency` + `studsdepth`; branch `feat/auction-budget-urgency`). **Run F's exact knobs, only the league size changes: 12 teams instead of 16.** Fair share at 12 teams: **playoff 0.50 / bye 0.167 / champ 0.083** (vs 16-team's 0.375 / 0.125 / 0.0625). VORP table `data/vorp_2026/half_12team.parquet` was rebuilt on this machine from the on-disk ESPN+Sleeper consensus snapshot (asof 2026-06-09), re-scored to ESPN_HALF via the scoring layer (`expected_points`; verified `half == ppr − 0.5·receptions` to machine precision) since `data/raw/external_projections` wasn't present to re-run `generate_preset_vorp_tables.py`.

Per-model metrics (mean [95% CI on exp pts]), sorted by exp pts:
- **vorpshare:** 1213 [1171.1, 1252.8] · win 0.48 · playoff 0.44 · bye 0.12 · champ 0.05
- **patient:** 1170 [1147.5, 1194.0] · win 0.45 · playoff 0.39 · bye 0.04 · champ 0.01
- **inflation:** 1102 [1063.8, 1135.4] · win 0.38 · playoff 0.27 · bye 0.03 · champ 0.01
- **static:** 1096 [1055.8, 1129.6] · win 0.37 · playoff 0.26 · bye 0.04 · champ 0.01
- **studsdepth:** 1079 [1043.1, 1112.0] · win 0.37 · playoff 0.25 · bye 0.03 · champ 0.01
- **overbid:** 1064 [1030.9, 1092.1] · win 0.35 · playoff 0.21 · bye 0.02 · champ 0.01
- **marginal:** 1061 [1030.0, 1095.3] · win 0.35 · playoff 0.20 · bye 0.02 · champ 0.01
- **anchors:** 971 [945.3, 998.0] · win 0.27 · playoff 0.10 · bye 0.01 · champ 0.00

**Headline — the shallower league SPREADS the field instead of compressing it, and `vorpshare`/`patient` break away.** Where Run F's 16-team field collapsed into a tight playoff band (0.08–0.13), at 12 teams two strategies separate cleanly: **`vorpshare` (playoff 0.44, champ 0.05) and `patient` (0.39) lead a value-bidding pack** (`inflation`/`static`/`studsdepth` ≈ 0.25–0.27), trailing down to `anchors` (0.10). `vorpshare` significantly beats every model on playoff% (paired CIs exclude 0) **except** `patient`, where it ties on playoff but wins on points (+43 [+0.4, +82.6]), bye (+0.079 [+0.041, +0.119]), and champ (+0.041 [+0.021, +0.061]). Two patterns persist from Run F: **`anchors` is dead last** (over-concentration, nothing for urgency to deploy), and **`studsdepth` lands mid-pack** with no separation.

**The hero is far more competitive in the shallower league — but the best strategy still sits below fair share.** Best playoff% is **0.44 vs the 0.50 baseline** (and best reg-win 0.48 vs ~0.50) — *nearly* at par, versus 16-team where the best hero managed only 0.13 against a 0.375 baseline. That the gap to baseline shrinks so much as the field thins (15 bots → 11) is itself evidence that the sub-baseline result is driven substantially by **deep-field bot contention** (the §3.6 / Run E–F caveat: 15 noisy-WTP bids order-statistic over `baseline_dollars`, so a value-bidding hero is outbid on studs), not purely by weak hero strategy. It also says **strategy choice matters more in a 12-team league** — the spread between best and worst hero (0.44 vs 0.10) dwarfs 16-team's (0.13 vs 0.03). **No winner declared** — one seat, byes off, 40 seeds; the call is September 2026. **Reproduce:** swap `--vorp-table data/vorp_2026/half_12team.parquet --league-config configs/half_12team.league.json` into the command below.

**Two directions this raises (user, 2026-06-18; tracked in `TODO.md`).** (1) **Average across all available seasons.** Year-to-year results swing wildly (Run A→B→C→D→E→F); a single 2026 snapshot is one noisy draw. Re-run every backtest across all seasons we have data for and average, so the strategy ranking reflects a multi-year mean rather than one year's projection set. (2) **Learn the strategy directly (RL).** That hand-authored bid models all cluster at or below the field's fair share suggests we're not searching the strategy space well by hand — a reinforcement-learning agent that learns a bid policy against the bot market (and ultimately self-play) may find an edge the fixed heuristics can't.

**Run H — 2026-06-19** (`half_12team` **and** `half_16team`, seat 6, 40 seeds, n_sims=500, price_jitter=0.15, budget=200; **byes OFF** — 2026 schedule not ingested; **REALISTIC MARKET** identical to Runs E–G: `nomination_temp=1.0` + mixed bot field (⅓ aggressive / ⅓ patient value-hunter / ⅓ balanced); **eight contestants** identical to Runs F/G incl. `_budget_urgency` + `studsdepth`; branch `feat/auction-bot-real-price-anchor`, TODO #49c Slice 2). **This is the first run with ESPN-anchored bots — the A/B that breaks the shared-value problem.** Both preset tables were regenerated from a **fresh `external_projections` re-ingest** (`asof 2026-06-19`, 199 crowd / 160 expert ESPN auction values), so `espn_auction_dollars` is populated (~201 priced of ~204–208 in-pool). The experiment is a clean one-variable A/B at each league size: `--bot-prices espn` (bots price off real ESPN auction values, SOS-allocated, $1 floor for unpriced) **vs** `--bot-prices model` (the prior shared-value bots — bots price off the hero's own SOS `auction_dollars` ± noise). Same table, same seat, same seeds; **only the bot pricing changes.** The hero always uses our SOS model. Fair share: 12-team playoff 0.50 / champ 0.083; 16-team playoff 0.375 / champ 0.0625.

**Playoff% by model — ESPN-anchored bots vs shared-value (model) bots** (Δ = espn − model):

| | 12-team (fair share 0.50) | | | 16-team (fair share 0.375) | | |
|---|---|---|---|---|---|---|
| model | espn | model | Δ | espn | model | Δ |
| static     | 0.43 | 0.43 | 0.00 | **0.48** | 0.35 | **+0.13** |
| inflation  | 0.45 | 0.54 | −0.09 | **0.50** | 0.55 | −0.05 |
| marginal   | 0.37 | 0.21 | **+0.16** | 0.29 | 0.12 | **+0.17** |
| anchors    | 0.15 | 0.18 | −0.03 | 0.11 | 0.12 | −0.01 |
| overbid    | 0.36 | 0.36 | 0.00 | **0.40** | 0.28 | **+0.12** |
| vorpshare  | 0.47 | 0.50 | −0.03 | 0.18 | 0.30 | **−0.12** |
| patient    | **0.62** | 0.53 | **+0.09** | 0.35 | 0.20 | **+0.15** |
| studsdepth | 0.35 | 0.31 | +0.04 | **0.41** | 0.31 | **+0.10** |

(Bold espn cells clear fair share.) Per-model mean points and full CIs for all five metrics are in `/tmp/run_h/h_{12,16}_{espn,model}.txt` from this run; the headline metric is playoff%.

**Headline — ESPN-anchored bots are *exploitable*, and the hero's edge appears for the first time in the realistic market.** Against the shared-value bots (Runs E–G's field), no hero cleared fair share in the realistic 16-team market (Run F best playoff 0.13 at seat 1). Here, switching *only* the bots' price anchor to real ESPN values, **most value-bidding heroes improve and several clear fair share**: 16-team `static` 0.35→**0.48**, `marginal` 0.12→0.29, `overbid` 0.28→**0.40**, `patient` 0.20→**0.35**, `studsdepth` 0.31→**0.41** (field-mean Δ ≈ **+0.06** playoff); 12-team `patient` 0.53→**0.62** (above the 0.50 baseline) and `marginal` 0.21→0.37. This is exactly the #49c thesis: a *predictably biased* market is beatable where a *randomly noisy* one is not. The shared-value bots gave the hero no informational edge (they priced off its own numbers); ESPN-anchored bots have a real, systematic bias the SOS hero exploits.

**Mechanism (the diagnostic).** The CLI's ESPN-vs-ours readout shows where the bias lives: ESPN underprices a cluster of players our model rates — e.g. several at ESPN $1–8 that our SOS values at $21–30 (value_delta +19 to +23) — while paying up ($36–53) for the studs its crowd loves. The largest disagreements are **rookies** (placeholder `99-` gsis): ESPN's crowd auction values are sparse/low for unproven rookies, but our projections rate them, so the SOS hero grabs them cheap while the ESPN-anchored bots leave them on the board. That deep-pool/rookie bargain is the hero's edge.

**Who it helps and who it doesn't.** `patient` (the value-hunter that holds budget for mid-tier value) benefits most — it is literally built to wait for the bargains ESPN-anchoring creates (best in 12-team at 0.62; +0.15 in 16-team). `marginal` (bids to marginal lineup lift) gains big in both sizes (+0.16 / +0.17). The two that *don't* improve are instructive: `inflation` does slightly **worse** vs ESPN bots (its live surplus-inflation repricing is calibrated to the SOS market it now diverges from), and `vorpshare` drops in 16-team (−0.12 — spreading budget proportionally doesn't concentrate on the ESPN bargains). The edge is also **larger in the deeper 16-team field** (field-mean Δ +0.06 vs ~+0.02 in 12-team): more contention + more deep pool = more ESPN mispricing to exploit. The **deep-league inflation property** (spec, accepted-not-guarded) showed up as expected — the ESPN-anchored bot vector tops out at $73 (12-team) / $100 (16-team) per player, above ESPN's nominal stud values, because the surplus concentrates on the priced players.

**Caveats (data, not a decision).** (1) **Not comparable to Runs F/G's levels** — this run uses **seat 6** (F/G used seat 1), a **fresh** consensus table (asof 2026-06-19 vs F/G's 2026-06-09), and **n_sims=500** (vs 200). The valid comparison is the *within-Run-H* espn-vs-model A/B (same seat/table/seeds/n_sims), not Run H vs F/G. (2) **Byes OFF** (2026 schedule not ingested) — injury availability is applied, byes are not. (3) **40 seeds, one seat** — the per-model playoff%s carry ~±0.05–0.10 sampling spread; treat the Δ pattern (most value heroes improve, esp. 16-team) as the signal, not any single cell. (4) The diagnostic's biggest deltas are all rookies because the 2026 `id_map` still carries them as placeholder gsis; veteran ESPN↔ours agreement is tighter. **No winner declared** — this confirms the bot *market* is now realistically exploitable (the #49c realism lever worked), not that any one hero strategy is the answer; the strategy call remains September 2026, and folding ESPN-anchored bots into the multi-year averaging (#49a) is the next reliability step.

**Run H follow-up — `PatientValueBid` is mis-tuned; mid-tier *breadth* (`scrub_frac`) is the biggest strategy lever found (2026-06-19, 12-team, ESPN bots).** Run H's 12-team leader `patient` only deploys ~$136 of $200 (it floors studs *and* the bottom 50% by VORP as scrubs at $1). Investigating whether the idle budget is a problem produced a clear, CI-separated result.

- **`scrub_frac` sweep (hold `stud_frac=0.10`; 30 seeds × 300 sims):** lowering `scrub_frac` (treat more of the pool as contested mid-tier instead of $1-scrubs) monotonically lifts playoff% **0.64 (sf=0.50, shipped default) → 0.83 (sf=0.0)** and raises spend ~$144 → ~$161. `midtier_premium` is a *wash-to-slightly-negative* (bidding harder per player overpays; it's breadth, not depth-of-bid, that wins).
- **Star-cap sweep (hold `scrub_frac=0`, `prem=0.35`; 40 seeds × 300 sims):** lowering the stud-floor count to contest the top players *does* deploy the budget (`floor_top_45` → $190, `floor_top_35` → $195), but playoff% stays **flat ~0.80–0.83** across `floor_top_35..60`, with the slight optimum at the *lower*-spend `floor_top_55–58`. So spending the last ~$30 on a contested stud is a **lateral trade, not an upgrade** — the idle cash was a symptom, not the problem.
- **Full test (40 seeds, n_sims=500, the 8 shipped contestants + the tuned `patient_deep` = `scrub_frac=0`, ESPN bots, seat 6):** `patient_deep` is the runaway field leader.

  | model | pts | reg-win | playoff | bye | champ |
  |---|---|---|---|---|---|
  | **patient_deep** | 1111 | 0.65 | **0.83** | 0.33 | 0.15 |
  | patient (shipped) | 991 | 0.55 | 0.62 | 0.16 | 0.07 |
  | vorpshare | 929 | 0.49 | 0.47 | 0.14 | 0.07 |
  | inflation | 905 | 0.48 | 0.45 | 0.13 | 0.06 |
  | static | 894 | 0.47 | 0.43 | 0.12 | 0.05 |
  | marginal / overbid / studsdepth | ~855 | ~0.44 | 0.35–0.37 | ≤0.08 | ≤0.04 |
  | anchors | 711 | 0.32 | 0.15 | 0.02 | 0.01 |

  Paired diff `patient_deep − patient` (CRN) is **CI-separated on every metric**: playoff **+0.205 [+0.172, +0.241]**, bye +0.168 [+0.145, +0.191], champ +0.078 [+0.066, +0.090], reg-win +0.098, points +120.

**Mechanism:** ESPN-anchored bots overpay for the studs ESPN prices high and leave a deep, *cheap* mid-tier; a breadth-maximizing hero hoovers up that mid-tier value for a far higher-floor roster than any stud-buyer. **Status (data, no default change):** the shipped `PatientValueBid(scrub_frac=0.50)` is the *worst* setting swept and a strong candidate for re-tuning to `scrub_frac≈0` (or adding a `patient_deep` contestant) — **deferred to the September strategy decision.** Scope caveat: 12-team only, ESPN-anchored field only, one 2026 snapshot, seat 6; 16-team auction not tested (the auction tuning question is 12-team-specific for this league).

**Multi-year bake-off (#49a) — 2021–2026, the reliability check (2026-06-19, branch `feat/patient-deep-contestant`).** With per-season preset tables now available for every season (branch `feat/multi-year-auction-ingest`, PR #84), the bake-off was run across all six seasons and the per-model metrics averaged — the multi-year mean the single-season Runs A–H couldn't give. Setup: 12-team half-PPR, ESPN-anchored bots, **20 seeds × 300 sims per season**, seat 6, all **9 standing contestants** (incl. the newly-added `patient_deep` = `PatientValueBid(scrub_frac=0)`).

**Methodology correction (mid-investigation):** the first pass priced every ESPN-*unranked* player at a flat `min_bid` for the bots, so the bot field rostered random $1 scrubs (a bot literally started a 3rd-string QB). `espn_anchored_bot_prices` now falls back to `_UNRANKED_MODEL_DISCOUNT` (0.4) × our VORP-based model value for unranked players — cheap but ordered, so bots field real depth. **This changed the rankings materially** (the flat-$1 field was inflating the stud-buying heroes). The table below is the **corrected** run; the buggy first run had `inflation`/`static` at 0.70/0.69 (apparent #2/#3) with a spurious "era-shift," now understood as an artifact.

| model | playoff% (mean) | champ% | sd | per-season playoff `[21 22 23 24 25 26]` |
|---|---|---|---|---|
| **patient_deep** | **0.825** | **0.164** | **0.020** | `0.81 0.81 0.81 0.86 0.85 0.82` |
| patient (shipped) | 0.577 | 0.071 | 0.090 | `0.44 0.49 0.61 0.57 0.70 0.65` |
| vorpshare | 0.529 | 0.079 | 0.075 | `0.53 0.48 0.58 0.64 0.55 0.40` |
| inflation | 0.506 | 0.076 | 0.056 | `0.56 0.50 0.51 0.57 0.50 0.40` |
| static | 0.497 | 0.073 | 0.054 | `0.54 0.49 0.50 0.56 0.50 0.39` |
| marginal | 0.488 | 0.049 | 0.078 | `0.46 0.46 0.63 0.54 0.40 0.44` |
| studsdepth | 0.417 | 0.051 | 0.045 | `0.45 0.44 0.41 0.42 0.46 0.32` |
| overbid | 0.404 | 0.050 | 0.046 | `0.40 0.42 0.40 0.41 0.47 0.32` |
| anchors | 0.121 | 0.006 | 0.026 | `0.09 0.12 0.10 0.10 0.14 0.17` |

**Findings (data; the strongest the harness has produced):**
- **`patient_deep` dominates by a wide, stable margin:** mean playoff **0.825 — +0.25 over #2** (`patient` 0.577), highest champ rate (0.164), **lowest cross-season variance (sd 0.020)**, and **#1 in all six seasons** (range 0.81–0.86). The single-season tuning result generalizes across six seasons and three auction-value regimes, and it *strengthened* once the bot field was made realistic.
- **The flat-$1 bug was inflating the stud-buyers.** `inflation`/`static` dropped from an apparent 0.70/0.69 to a true ~0.50 once the bots field real depth — buying studs only looks great against a field that punts half its roster to $1. The earlier "era-shift" (inflation/static winning the crowd-only early seasons) was largely the bug, not a real basis effect: corrected, they sit at a flat ~0.50 every year.
- Shipped `patient` (`scrub_frac=0.50`) is the clear #2 (0.577) — still well behind `patient_deep`. `anchors` is dead last every season.

**Caveats:** the ESPN auction-value *basis varies by season* (pre-2023 crowd-only, 2025 expert-only, 2023/24/26 both); 2026 has no schedule yet → empty byes (perturbs its bye/champ cells); 20×300 is modest (directional ranking, not tight CIs); the unranked-discount (0.4) is itself a tunable knob worth sweeping. Even so, the `patient_deep` separation is far outside the noise.

**Status (data, no default change):** `patient_deep` is now a standing contestant in every bake-off (`_MODELS`), and the corrected ESPN-anchored bot pricing is the new methodology baseline. The multi-year evidence makes `patient_deep` the leading **September** candidate — either re-tune the `patient` default to `scrub_frac≈0` or adopt `patient_deep` as the recommended hero. Decision still deferred to September per project policy.

**Multi-year bake-off RE-RUN — snake-draft broke bots (the new methodology baseline; 2026-06-19, branch `feat/auction-broke-bot-snake`, PR #86).** The discount-era run above priced ESPN-*unranked* players at `_UNRANKED_MODEL_DISCOUNT` (0.4) × our VORP model for *all* bots. That was a stopgap: it bent flush-bot pricing to patch a **broke-bot behavior** bug (a bot at the `feasible_max == min_bid` floor bid $1 on whatever was nominated — "started a 3rd-string QB"). The shipped fix targets the behavior — an out-of-money bot now drafts like a **snake drafter** (a per-bot fixed noisy-ADP `SnakeBoard`, `adp_jitter=8.0`, drawn once per draft from a dedicated CRN substream: nominate/snipe only its best-available-for-needs target; abstain otherwise; nominator-takes-$1 backstop). Flush bots / hero / flush-hero nomination unchanged. The bake-off was re-run with this field — **identical knobs to the discount-era run above** (12-team half-PPR, ESPN-anchored bots, **20 seeds × 300 sims**, seat 6, all 9 contestants), so the only variable is the broke-bot behavior.

| model | pts | win% | **playoff%** | bye% | champ% | sd(pl) | discount-era pl | Δ | per-season playoff `[21 22 23 24 25 26]` |
|---|---|---|---|---|---|---|---|---|---|
| **patient_deep** | 1145 | 0.57 | **0.660** | 0.229 | 0.112 | 0.057 | 0.825 | −0.165 | `0.72 0.68 0.55 0.72 0.67 0.62` |
| inflation | 1051 | 0.49 | 0.479 | 0.149 | 0.075 | 0.097 | 0.506 | −0.027 | `0.57 0.51 0.42 0.62 0.43 0.33` |
| vorpshare | 1051 | 0.49 | 0.477 | 0.157 | 0.083 | 0.040 | 0.529 | −0.052 | `0.56 0.46 0.48 0.45 0.47 0.45` |
| static | 1037 | 0.48 | 0.454 | 0.138 | 0.069 | 0.086 | 0.497 | −0.043 | `0.50 0.48 0.38 0.60 0.43 0.34` |
| patient (shipped) | 1009 | 0.46 | 0.408 | 0.094 | 0.044 | 0.029 | 0.577 | **−0.169** | `0.39 0.42 0.35 0.42 0.45 0.41` |
| overbid | 986 | 0.44 | 0.373 | 0.105 | 0.053 | 0.052 | 0.404 | −0.031 | `0.41 0.43 0.36 0.42 0.31 0.30` |
| studsdepth | 985 | 0.44 | 0.368 | 0.098 | 0.048 | 0.045 | 0.417 | −0.049 | `0.36 0.42 0.36 0.41 0.37 0.28` |
| marginal | 936 | 0.40 | 0.283 | 0.060 | 0.027 | 0.078 | 0.488 | **−0.205** | `0.32 0.27 0.38 0.36 0.18 0.19` |
| anchors | 764 | 0.27 | 0.116 | 0.021 | 0.010 | 0.033 | 0.121 | −0.005 | `0.10 0.12 0.06 0.12 0.13 0.16` |

**Findings (data; the snake-draft field is the new baseline):**
- **`patient_deep` still leads, and still #1 in all six seasons** (range 0.55–0.72), highest champ (0.112). Its margin over #2 **compressed from +0.25 to +0.18** and its level fell 0.825→0.660 — because the snake-draft bots are a **genuinely harder, more realistic field**: a broke bot now hoovers the best-available mid-tier by ADP instead of punting to $1 scrubs, so there is less free depth for any hero to collect. **Every** model's playoff% dropped (mean Δ ≈ −0.08); that uniform decline is the realism lever working, not a regression.
- **The mid-pack re-ranked.** Shipped `patient` (`scrub_frac=0.50`) **fell from #2 (0.577) to #5 (0.408), Δ −0.169**, and `marginal` cratered (Δ −0.205) — both relied on the scrub-dumping field to leave them cheap depth; against bots that now draft that depth themselves, they lose it. The new #2–#4 (`inflation`/`vorpshare`/`static`, ~0.45–0.48) are tightly packed. `anchors` is dead last every season (unchanged).
- **The `patient_deep` > `patient` story holds and slightly strengthens.** The paired gap is **+0.252 mean** (per-season `+0.33 +0.26 +0.20 +0.29 +0.22 +0.21`), vs +0.248 in the discount era — breadth-maximizing (`scrub_frac=0`) beats $1-dumping the bottom half *even harder* against a depth-drafting field. `patient_deep`'s cross-season variance rose (sd 0.020→0.057) but it never drops below #1.

**Caveats:** same as the discount-era run (ESPN basis varies by season; 2026 byes empty; 20×300 is directional, not tight CIs) — plus: the snake-draft field changes the auction RNG consumption (abstaining broke bots draw no `price_jitter` noise), so absolute levels are **not** byte-comparable to the discount-era run beyond the within-comparison Δ; the read is the *re-ranking* and the *uniform downward shift*, which are far outside 20×300 noise for the top/bottom. The `adp_jitter=8.0` broke-bot knob (reused from the snake-draft default) is itself unswept.

**Status (data, no default change):** the **snake-draft broke-bot field is the new methodology baseline** (supersedes the 0.4-discount field; the discount is retained only for flush-bot unranked pricing). `patient_deep` remains the leading **September** candidate — its dominance survived the realism upgrade. Decision still deferred to September per project policy.

**Run I — 2026-07-14** (`half_12team`, seat 1, 150 seeds, n_sims=300, price_jitter=0.15, budget=200; **byes OFF**; **REALISTIC MARKET** — `nomination_temp=1.0` + mixed bot field + snake-draft broke bots; branch `feat/auction-balanced-value`, Slice 1). **Ten contestants** — the nine standing models + the new `balanced` (`BalancedValueBid`: bid `round(min(fair×(1+0.15), 2×my_budget/my_open_slots))` — a small premium over fair value to win contested players, capped at 2× the even per-slot share so the budget spreads, and **deliberately NO `_budget_urgency`**). Fresh 2026 pool (`asof 2026-07-14`, 241 ESPN-priced of 578). Both bot markets A/B'd at identical knobs: `--bot-prices model` (shared-value) vs `--bot-prices espn` (ESPN-anchored).

Playoff% / champ% by model (model-priced vs ESPN-anchored bots):

| model | model playoff | model champ | espn playoff | espn champ |
|---|---|---|---|---|
| **balanced** | **0.46** | **0.06** | **0.24** | **0.02** |
| patient_deep | 0.19 | 0.01 | 0.21 | 0.01 |
| patient | 0.19 | 0.01 | 0.21 | 0.01 |
| vorpshare | 0.37 | 0.05 | 0.14 | 0.01 |
| marginal | 0.08 | 0.00 | 0.13 | 0.01 |
| inflation | 0.15 | 0.01 | 0.10 | 0.01 |
| static | 0.13 | 0.01 | 0.10 | 0.01 |
| studsdepth | 0.12 | 0.01 | 0.07 | 0.00 |
| overbid | 0.11 | 0.01 | 0.08 | 0.00 |
| anchors | 0.06 | 0.00 | 0.02 | 0.00 |

**Headline — `balanced` is the top hero in BOTH markets, and matches-or-beats `patient_deep` (the standing breadth leader) in the realistic ESPN market.** Model-priced: `balanced` playoff **0.46** / champ 0.06, CI-separated above `vorpshare` (0.37) and far above `patient_deep` (0.19). ESPN-anchored: `balanced` playoff **0.24** (top), and the paired diff `patient_deep − balanced` (CRN) is **CI-separated in `balanced`'s favor on champ (−0.008 [−0.014, −0.002]) and bye (−0.015 [−0.027, −0.003])**, leans balanced on playoff (+0.022, CI overlaps), ties on points/reg-win. So the **premium+cap** breadth mechanism is at least as strong as `patient_deep`'s **scrub_frac=0** breadth — two independent routes to the same "balanced breadth wins" result. Both spread the budget into a full, high-floor roster instead of concentrating on studs; `balanced` additionally bids a small premium so it actually wins contested mid-tier, and its pace cap makes the spread structural rather than tuning-dependent.

**Caveats.** (1) The ESPN absolute levels are far below Run H's (best 0.24 vs 0.83): the fresh 2026-07-14 pool is much larger (578 players, only ~42% ESPN-priced), so the unranked-discount fallback dilutes the ESPN signal — the interpretable read is the **within-run** `balanced ≥ patient_deep`, not the absolute level. (2) 12-team only, seat 1, byes off, one 2026 snapshot; 16-team untested. **No winner declared** — the strategy call is September 2026. `balanced` joins as a standing contestant alongside `patient_deep`.

**Run J — 2026-07-14 — `balanced` retune (cap-vs-premium sweep; shipped default `premium` 0.15 → 1.0).** Run I found `balanced` (premium 0.15) below-average in the ESPN-anchored market (playoff 0.24). A cap×premium sweep (12-team half, seat 1, 100 seeds × 300 sims, fresh 2026-07-14 pool) separated the two knobs:

| variant | model playoff% | ESPN playoff% |
|---|---|---|
| cap $24, prem 0.15 (old default) | 0.29 | 0.24 |
| cap $24, prem 0.5 | 0.28 | 0.38 |
| **cap $24, prem 1.0 (NEW default)** | **0.28** | **0.44** |
| cap $47, prem 0.15 | 0.22 | 0.20 |
| cap $71, prem 0.15 | 0.12 | 0.16 |

**Two robust findings.** (1) **A LOW cap wins both markets** — raising `pace` (the per-player cap) monotonically *hurts* (it lets the hero chase over-priced studs and starve depth). (2) **The premium only matters in an INFLATED market** — flat in the model market (the mid-tier clears near fair value, so even a timid premium wins it) but a ~2× swing in the ESPN market (the mid-tier clears *above* fair value, so a high premium is needed to reach the cap and win the contested mid-tier). So **`premium=1.0` is a Pareto improvement** — neutral in the model market, ~2× in ESPN — and is now `BalancedValueBid`'s default; `pace=2.0` (low cap) is unchanged. **Mechanism:** the winning play is to bid the *low* cap on *every* startable player (spread the whole budget into a full roster); the premium is what makes the timid fair-value bid actually reach that cap in an inflated room, and the low cap is what stops the hero over-paying for the studs (which the uncapped `Aggressive` bots win at $50–85 anyway). **Residual (OPEN):** even retuned, `balanced` (~0.44 ESPN) trails the elite `Balanced`-bot tier (~0.68) — a hero-vs-bot gap that is *not* currency (a hero bidding the bots' own `bot$`, capped, also stalls at ~0.24–0.44) and *not* a measurement artifact (a strong hero only drops the bots 0.71→0.68). Next diagnostic: a `price_jitter=0` run to test whether the bots' bid noise accounts for it, or whether it is structural (seat/tie-order). **No winner declared** — Sept decision.

**Run K — 2026-07-14 — cap-inflation fix (`balanced_flat`) + `pace×premium` re-tune, both markets, GOAL = reg-season win%** (`half_12team`, seats 1 & 6, 12-team half-PPR, n_sims=300, price_jitter=0.15, budget=200; **byes OFF**; **REALISTIC MARKET** — `nomination_temp=1.0` + mixed bot field + snake-draft broke bots; branch `feat/auction-robust-win-hero`, Slice 1). New `balanced_flat` = `BalancedValueBid(non_increasing_cap=True)`: the pace cap is clamped to the OPENING per-slot share (`pace × budget₀/roster_size`) so it can't self-inflate as the hero wins cheap players (the diagnosed bug). Raced a `pace ∈ {1.0,1.5,2.0,2.5} × premium ∈ {0.5,1.0,1.5}` grid of `balanced_flat` variants vs the inflating `balanced` control and `patient_deep`, on **both** bot markets (`--bot-prices model` and `espn`), scored by **`reg_win_pct`** (the user's goal metric). Seat 1 = 60 seeds/market (3×20 chunks); seat 6 = 20 seeds/market. Crash-safe chunked runner (`scripts/auction_cap_tuning.py`); all 8 chunks completed clean (no Raptor Lake fault). 12-team fair share `reg_win_pct ≈ 0.50`.

Best worst-case `reg_win_pct` across the two markets (top rows):

| seat | model | espn | worst | | seat | model | espn | worst |
|---|---|---|---|---|---|---|---|---|
| **seat 1** | | | | | **seat 6** | | | |
| `balanced` (inflating control) | 0.471 | 0.410 | **0.410** | | `flat_p2.0_prem0.5` | 0.604 | 0.559 | **0.559** |
| `flat_p2.5_prem1.0` | 0.465 | 0.399 | 0.399 | | `flat_p1.5_prem0.5` | 0.546 | 0.554 | 0.546 |
| `flat_p2.0_prem1.5` | 0.440 | 0.398 | 0.398 | | `balanced` (control) | 0.593 | 0.542 | 0.542 |
| `patient_deep` | 0.369 | 0.358 | 0.358 | | `patient_deep` | 0.576 | 0.521 | 0.521 |

**Finding 1 — the cap fix is a WASH, not an improvement.** At seat 1 the inflating `balanced` control (worst-case 0.410) *edges* the best `balanced_flat` variant (0.399); at seat 6 a low-premium `flat_p2.0_prem0.5` (0.559) edges `balanced` (0.542). Both gaps (~0.01–0.02) sit inside the seed noise (60-seed CI half-width ≈ 0.04). No single flat config robustly beats the inflating control across seats, and the optimal flat tuning even *flips* by seat (seat 1 wants high pace+premium; seat 6 wants low premium). **The cap self-inflation was a real mechanism but not a `reg_win_pct` mover** — `balanced_flat` does not beat `balanced`. (Note the reversal vs Run J's "low cap wins": with a *non-inflating* cap, a *higher* pace helps at the bad seat, because the cap can no longer balloon and a low pace just starves the roster.)

**Finding 2 — SEAT dominates, and it is NOT the cap-inflation effect.** The same `balanced` policy, on the **same CRN seeds (0–19)**, scores 0.434 (model) / 0.492 (espn) at **seat 1** but 0.542 / 0.593 at **seat 6** — a **~+0.10 `reg_win_pct` swing from seat alone**, an order of magnitude larger than any bid-strategy difference. At seat 6 the hero **clears the 0.50 fair share in both markets** (beats the field); at seat 1 no config does (~0.40–0.49). Crucially, **the flat cap did NOT close the seat gap** — so the prior memory diagnosis ("the seat-0 gap is a seat-role / *cap-inflation* effect") is **corrected**: fixing the cap leaves the seat effect intact, so it is a **seat-role effect not caused by cap inflation.** Candidate mechanisms (untested): seat-dependent bot-archetype assignment (`assign_bot_archetypes` round-robins the 3 archetypes over the 11 non-hero seats, so *which* neighbor is Aggressive/Balanced shifts with the hero seat) and the fixed gauntlet schedule in `project_draft` (seat determines opponents). A ~0.10 seat swing in an *auction* (where everyone can bid on everyone) is large enough to warrant its own diagnostic — is seat 1 structurally disadvantaged, or does seat 6 just draw an easy schedule?

**Implication for the goal (data, no decision).** Bid-strategy tuning has hit diminishing returns for `reg_win_pct`: the cap fix is a wash and the ceiling at a fixed seat is set by **seat/nomination position**, which a bid model can't touch. This directly motivates **Slice 2 (nomination warfare / poisoning)** — a nomination-side lever that could specifically address the disadvantaged nominate-first seat — and/or a **seat-effect diagnostic**. Multi-year validation of `balanced_flat` was **not** run (it is gated on regenerating the 2021–2025 pools via re-ingest, and validating a wash isn't worth that cost). `balanced_flat` stays a registered contestant (it is not *worse* on average, and is the cleaner mechanism), but is **not** adopted as a new default. **No winner declared** — Sept decision. Artifacts: `reports/_cap_tuning/2026{,_seat6}/*.json` (untracked).

**Run K seat diagnostic — 2026-07-14 — SEAT 1 IS A LONE STRUCTURAL OUTLIER; the hero already beats the field at 11/12 seats.** Swept the hero seat 1..12 for a fixed probe (`balanced` + `patient_deep`), 20 seeds/seat, n_sims=300, both markets, 2026 12-team half (scratchpad `seat_sweep.py`). `balanced` `reg_win_pct` by seat:

| seat | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | seat-avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model | **0.434** | 0.530 | 0.549 | 0.528 | 0.516 | 0.542 | 0.505 | 0.536 | 0.539 | 0.556 | 0.535 | 0.539 | **0.526** |
| espn | **0.492** | 0.599 | 0.613 | 0.590 | 0.572 | 0.593 | 0.572 | 0.573 | 0.573 | 0.584 | 0.570 | 0.573 | **0.575** |

**Seat 1 sits ~0.09–0.10 below the tight seats-2–12 cluster** (which all clear the 0.50 fair share) in both markets; `patient_deep` shows the identical shape (seat 1 lowest: 0.387 model / 0.396 espn, rest ~0.51 / ~0.55). So the effect is **structural to seat 1 specifically**, not a smooth curve or random schedule scatter.

**This reframes the entire auction investigation.** Every prior "hero is sub-baseline / can't beat the bots" result — Runs A–K and the memory's "0.21 as hero at seat 0" — was measured at **seat 1**, the one broken seat. Seat-averaged, the *current* `balanced` **already beats the field in both markets** (0.526 model / 0.575 espn); at 11 of 12 seats it clears fair share. The "bots overpay so we can't compete" premise was **seat-1 tunnel vision.** A ~0.10 `reg_win_pct` penalty for the *nominate-first* seat in an auction (anyone can bid on anyone) is almost certainly a **modeling artifact** (candidate causes: the hero replaces an rng-consuming bot at its seat, shifting the shared bid-noise stream vs a bots-only field; seat-dependent archetype placement; the fixed gauntlet schedule), and because seat 1 was the default measurement seat it has **biased every recorded auction number downward.** Highest-value next step is to diagnose/fix the seat-1 artifact (a correctness issue) before any further strategy work. Artifacts: scratchpad `seat_{model,espn}.json` (ephemeral).

**Run K seat FIX — 2026-07-14 — root-caused to a `resolve_bids` tie-break bug; FIXED and validated (commit on `feat/auction-robust-win-hero`).** A single-auction roster probe (seat 1 vs 6, same seeds) showed the hero at seat 0 acquiring **exactly 4 min-bid ($1) players every draft** (seat 6: zero), ~120 fewer points. Root cause: `resolve_bids` broke ties by **lowest seat index**, so when the hero and the min-bidding bots (e.g. `PatientValueBot` bids `min_bid` on scrubs) tied at $1, **seat 0 won every one** and hoarded junk that crowded out contested mid-tier value. Pure implementation artifact (lowest-index has no real-auction basis). **Fix:** ties now break **uniformly at random** among the top bidders via the engine rng (`resolve_bids(bids, min_bid, rng)`); the old `test_resolve_ties_break_on_seat_index` (which pinned the bug) is replaced by a randomized-tie-break test; all 153 auction tests pass, ruff/mypy clean.

**Validation (re-swept 12 seats, 20 seeds, both markets):**

| | seat 1 | seat-avg | spread across seats |
|---|---|---|---|
| model, before → after | 0.434 → **0.498** | 0.526 → **0.495** | 0.122 → **0.074** |
| espn, before → after | 0.492 → **0.566** | 0.575 → **0.554** | 0.121 → **0.061** |

Seat 1 moved into the pack (model +0.064, espn +0.074) and the seat curve flattened (spread ~halved; residual ±0.03 is 20-seed noise). **No lone outlier remains — the seat effect is resolved.**

**Corrected, seat-symmetric conclusion for the whole auction investigation:**
- The pre-fix "hero beats the field at 11/12 seats" was **partly artifact** — with the hero at seats 2–12, a *bot* at seat 0 ate the junk, giving the hero a free boost. Removing it drops the model-market seat-average from 0.526 to **~0.495**.
- **ESPN-anchored (realistic) market: current `balanced` genuinely beats the field, ~0.554 seat-averaged** (> 0.50) — a real edge from exploiting ESPN mispricing, and it survives the seat fix.
- **Model market (symmetric, bots price off our own numbers): the hero is ~even (~0.495)** — you cannot systematically beat a field using your own valuations; ~fair-share is the correct expectation.
- **Every prior auction number measured at seat 1 (Runs A–K, the memory's "0.21 at seat 0") was biased low by this bug** and should be read as pre-fix. `balanced` remains the win% leader among current strategies (≥ `patient_deep` at nearly every seat, both markets). The cap-fix "wash" (Run K) still holds — the tie-break bug affected `balanced` and `balanced_flat` equally.

**Run L — 2026-07-15 — FULL-FIELD seat-symmetric sweep (post-fix), GOAL = robust reg-season win%** (`half_12team`, all 12 seats × both bot markets, 20 seeds × 300 sims per (seat, market); **byes OFF**; **REALISTIC MARKET** — `nomination_temp=1.0` + mixed bot field + snake-draft broke bots; branch `feat/auction-fullfield-seat-sweep`). The first clean, apples-to-apples ranking of the **whole registered field** (`tournament_cli._MODELS`, all 11 heroes) with the seat-1 tie-break fix in place — every prior full-field ranking (Runs H/I) was pre-fix and seat-1-only, and the post-fix validation only re-ran a 2-hero probe (`balanced` + `patient_deep`). Crash-safe chunked runner `scripts/auction_seat_sweep.py`; all 24 chunks completed clean (~76 min, zero Raptor Lake faults). Ranked by **seat-averaged `reg_win_pct` worst-case across the two markets** (the robust-win goal metric). 12-team fair share = 0.50.

| rank | hero | espn | model | **worst** |
|---|---|---|---|---|
| **1** | **balanced** (shipped default) | **0.554** | **0.495** | **0.495** |
| 2 | balanced_flat | 0.522 | 0.487 | 0.487 |
| 3 | patient_deep | 0.516 | 0.474 | 0.474 |
| 4 | inflation | 0.408 | 0.490 | 0.408 |
| 5 | static | 0.396 | 0.455 | 0.396 |
| 6 | overbid | 0.388 | 0.427 | 0.388 |
| 7 | studsdepth | 0.378 | 0.417 | 0.378 |
| 8 | patient | 0.398 | 0.373 | 0.373 |
| 9 | vorpshare | 0.358 | 0.385 | 0.358 |
| 10 | marginal | 0.255 | 0.237 | 0.237 |
| 11 | anchors | 0.156 | 0.301 | 0.156 |

**Finding 1 — `balanced` (the current shipped default) is the outright robust win% leader: #1 in BOTH markets and #1 worst-case (0.495).** It tops the ESPN market at **0.554** (a real edge above the 0.50 fair share, from exploiting ESPN mispricing) and narrowly tops the symmetric model market too (0.495 vs `inflation` 0.490). No other strategy beats it in either market. **The robust-win-hero goal is met by the existing default** — the "hero can't beat the bots" premise was seat-1 tunnel vision (Run K), and seat-symmetric the answer is: beat the field in the realistic (ESPN) market, ~fair-share in the model market (you can't systematically beat a field pricing off your own numbers).

**Finding 2 — consistency check passed.** `balanced`'s seat-averages here (ESPN 0.554 / model 0.495) reproduce the Run-K 2-hero validation probe **exactly** — CRN + shared base seed makes each strategy's paired auctions identical whether raced alone or in the full field, so the full-field ranking is trustworthy, not a re-mix artifact.

**Finding 3 — `patient_deep` demoted to #3.** The former "multi-year era-robust leader" (Runs G/H) was crowned pre-fix at seat 1 under the older discount-era methodology; measured seat-symmetric with the fix, `balanced` beats it in both markets (worst-case 0.495 vs 0.474).

**Finding 4 — the stud-buyers are market-split, which is exactly why worst-case demotes them.** `inflation`/`static` are competitive in the *model* market (0.490 / 0.455) but **collapse in ESPN** (0.408 / 0.396): you cannot win overpriced studs cheaply when bots anchor on inflated ESPN values. The worst-case robustness gate punishes this market-fragility correctly — a hero that only wins when the room prices fairly is not robust.

**Finding 5 — the cap-fix "wash" holds at full-field scale.** `balanced` (inflating cap) slightly *edges* `balanced_flat` (non-inflating) in both markets (espn 0.554 vs 0.522, model 0.495 vs 0.487), so the inflating default stays; `balanced_flat` remains a registered contestant (#2, cleaner mechanism, not adopted).

**Finding 6 — no outlier seat remains.** `balanced`'s per-seat spread is tight (espn [0.518, 0.579], model [0.450, 0.523]) with seat 1 sitting right in the pack (espn 0.566 / model 0.498); `patient_deep`/`balanced_flat` show the same flat shape. The seat-1 fix holds across the whole field.

**Caveats.** 20 seeds/seat (12 seats pooled = 240 auction-seeds per market cell → the seat-*average* is fairly tight, but individual per-seat cells carry ±~0.03 noise); 2026 pool only, 12-team half-PPR only, byes off (2026 schedule not ingested → bye/champ cells perturbed, but reg_win_pct — the goal metric — is unaffected); the ESPN signal is diluted by the 42%-priced pool (only 241/578 ESPN-priced → unranked-discount fallback), so treat the **model market as the symmetric control and the ESPN market as the realistic-but-noisier read**. Directional ranking is robust; absolute levels are market-dependent. Artifacts: `reports/_seat_sweep/2026/*.json` (untracked). **No winner declared for the live draft** — the September strategy call stands — but for the stated goal, **`balanced` is the answer, and it is already shipped.** This is the clean baseline that Runs M–N below improve on (they retune the `balanced` default itself).

**Run M — 2026-07-15 — model-market microstructure diagnostic (is the model market actually unbeatable?).** Motivated by a challenge to the "beating a field that prices off your own numbers is impossible" reading (reg_win_pct is zero-sum → mean 0.50, but a *single* strategy can beat the field average). Added an opt-in `PickRecord` trace to `_simulate_to_state` (branch `feat/auction-market-inefficiency-diagnostic`); scratch harness raced the model market with hero=`balanced` (premium=1.0) at seat 1, 40 seeds × 200 sims, reading **all-seat** metrics from `project_draft` grouped by bot archetype.

Metrics by archetype (fair share 0.50):

| archetype | seats | win% | playoff% | champ% | end budget | roster value $ | spent $ | surplus $ |
|---|---|---|---|---|---|---|---|---|
| **BalancedBot** | 3 | **0.619** | **0.753** | **0.181** | 5.6 | **254** | 194 | **+60** |
| AggressiveBot | 4 | 0.506 | 0.512 | 0.075 | 0.0 | 213 | 200 | +13 |
| Hero (`balanced` prem1.0) | 1 | 0.493 | 0.474 | 0.045 | 3.8 | 132 | 196 | **−64** |
| PatientValueBot | 4 | 0.407 | 0.305 | 0.027 | 1.8 | 153 | 198 | −45 |

Clearing price vs fair value by draft quartile: **Q1 (studs) 1.34× (34% OVERPAY)**, Q2 1.03×, **Q3 (mid-tier) 0.76× (24% DISCOUNT)**, Q4 0.85×. Surplus captured per win: BalancedBot **+3.53**, Aggressive +0.75, Patient −2.66, Hero (`balanced`) **−3.75** (worst in the field).

**Finding — the model market is very inefficient, and the shipped hero exploits it BACKWARDS.** `BalancedBot` runs at **0.62 win** by *discipline*: it fades the early studs (which the uncapped aggressive bots bid to 1.34× fair) and banks the Q3 mid-tier discount (0.76× fair, once the room is tapped out), ending with **$254 of talent for $194**. Our hero `balanced` with **premium=1.0 bids fair×2**, so it chases the hot early tier, slams its pace cap early, **burns budget**, and misses the Q3 window — the **worst surplus in the field (−$64)** and only $132 of talent, stuck at 0.49. This **overturns Run J's "premium neutral in the model market"** (Run J was pre-seat-fix at seat 1, contaminated): post-fix, the premium is actively harmful in the model market.

**Run N — 2026-07-15 — premium sweep, both markets, seat-averaged → RETUNE default premium 1.0 → 0.0** (`half_12team`, all 12 seats × both markets, 20 seeds × 300 sims; **byes OFF**; **REALISTIC MARKET**; branch `feat/auction-market-inefficiency-diagnostic`). Raced `BalancedValueBid` at premium {0.0, 0.15, 0.5, 1.0} + `balbot_hero` (the noisy `BalancedBot` rule wrapped as a hero, reading `bot_dollars`) + `patient_deep`, scored by seat-averaged `reg_win_pct` worst-case across markets. Crash-safe chunked runner; 24 chunks clean (~27 min, zero faults).

| contestant | espn | model | **worst** | playoff worst | champ worst |
|---|---|---|---|---|---|
| **prem0.0** (disciplined) | 0.621 | 0.592 | **0.592** | **0.700** | **0.145** |
| prem0.15 | 0.640 | 0.561 | 0.561 | 0.638 | 0.104 |
| balbot_hero (noisy BalancedBot) | 0.560 | 0.583 | 0.560 | 0.630 | 0.120 |
| prem0.5 | 0.589 | 0.513 | 0.513 | 0.523 | 0.059 |
| **prem1.0 (old default)** | 0.554 | 0.495 | 0.495 | 0.484 | 0.043 |
| patient_deep | 0.516 | 0.474 | 0.474 | 0.436 | 0.037 |

**Finding — the shipped default (premium=1.0) was mis-tuned; premium=0.0 is the robust winner.** reg_win_pct is **monotone-decreasing in premium** on the worst-case metric: dropping 1.0 → 0.0 lifts worst-case win% from **0.495 to 0.592 (+0.097)**, roughly **triples model-market champ% (0.043 → 0.145)** and lifts playoff worst-case 0.484 → 0.700. Crucially, **low premium does NOT crater ESPN — it *improves* it (0.554 → 0.621)**, refuting Run J's "premium=1.0 needed for ESPN" (that tuning was the seat-1 bug). prem0.0 wins ESPN by pricing off *our model values* (exploiting ESPN mispricing) while staying disciplined; `balbot_hero`, which prices off the inflated ESPN values like the bots, only reaches 0.560 — so prem0.0 even beats the raw BalancedBot policy. The ~0.62 BalancedBot ceiling transfers to the hero seat post-fix (`balbot_hero` model 0.583), confirming the pre-fix "0.21 as hero" was entirely the tie-break bug. **Validation:** prem1.0 here reproduces Run L's `balanced` exactly (model 0.495 / espn 0.554) → the +0.097 is apples-to-apples, not a runner artifact. Per-seat vectors are flat (no outlier seat).

**Decision (this branch): retune `BalancedValueBid` default `premium` 1.0 → 0.0** (docstring + `test_balanced_default_is_disciplined_zero_premium` updated). This supersedes the Run J retune (PR #94) and updates Run L's `balanced` worst-case from 0.495 to **0.592**. `balanced` (now premium=0.0) genuinely beats the field in **both** markets (0.62 espn / 0.59 model, both > 0.50 fair share) — the robust-win-hero goal is met with margin. **Caveats:** 2026 pool only, 12-team half-PPR, byes off, ESPN signal diluted (241/578 priced); directional ranking robust, absolute levels market-dependent. **No live-draft winner declared** — the September call stands — but the shipped default is now the best robust hero we have found. This is the baseline Slice 2 (nomination poisoning) must beat (bar is now ~0.59, not ~0.49).

**Run O — 2026-07-15 — Slice 2 nomination-poisoning FEASIBILITY PROBE → NO-GO (but a directional near-miss)** (`half_12team`, all 12 seats × both markets, 20 seeds × 300 sims; **byes OFF**; **REALISTIC MARKET**; branch `feat/auction-nomination-poison-probe`; spec+plan `docs/superpowers/{specs,plans}/2026-07-15-auction-nomination-poisoning*`). Added an opt-in `hero_nominator` hook to the auction loop (hero-only, non-forced-only; `None` byte-identical) + two poison heuristics in `nomination.py`: `drain_max` (nominate the priciest player, forcing the room to spend on a stud the capped hero loses anyway) and `drain_off_position` (nominate the priciest player at a position the hero has already filled its starters for, so the drain lands on opponents who still need it). Bid held fixed at `balanced` p0.0 for all three contestants to isolate the *nomination* lift. Decision made on the **CRN-paired** per-`(seed, seat)` `poison − control` lift (cancels shared noise so a ~+0.02 effect is resolvable). Crash-safe chunked; 24 chunks clean (~30 min).

| contestant | espn level | espn Δ (paired) | model level | model Δ (paired) | seat-stable (espn/model) |
|---|---|---|---|---|---|
| `control` (no poison) | 0.621 | — | 0.592 | — | — |
| `drain_max` | 0.603 | **−0.018** | 0.593 | +0.001 | no / no |
| `drain_off_position` | 0.636 | **+0.015** | 0.608 | **+0.016** | yes / yes |

Context (seat-avg make_playoffs_pct / champ_pct): `control` espn 0.750/0.200, model 0.700/0.145 · `drain_max` espn 0.723/0.174, model 0.700/0.149 · `drain_off_position` espn **0.781/0.216**, model **0.733/0.164**.

**R8 sanity gate PASSES:** `control` (hook `None`) reproduces Run N `balanced` p0.0 **exactly** (espn 0.621 / model 0.592) — the harness is sound and the paired lift is apples-to-apples.

**Verdict — NO-GO (pre-registered bar: min market Δ ≥ +0.02 AND seat-stable in both markets):**
- **`drain_max` is actively HARMFUL** — −0.018 in espn (flat in model), not seat-stable. Nominating the single priciest player overall often surfaces a stud the hero itself wanted (or that the disciplined room clears near fair value), so it does not specifically drain opponents. Rejected.
- **`drain_off_position` is a real, seat-stable, both-market POSITIVE that just misses the bar** — reg_win_pct **+0.015 espn / +0.016 model** (positive at a majority of seats in *both* markets), and a *larger* lift on the deep-run metrics: **playoff ≈ +0.031 / +0.033**, **champ ≈ +0.016 / +0.019**. Targeting the drain at positions the hero is done with works in the intended direction — but the reg_win_pct lift (0.015/0.016) is below the +0.02 goal threshold.

**Conclusion (data; no default change):** nomination poisoning does **not** clear the pre-registered reg_win_pct bar, so the shipped **`balanced` p0.0 hero stands** as the robust-win hero. `drain_off_position` is a directionally-correct near-miss whose 20-seed espn CI just touched 0, so a **40-seed high-power re-run** (control + drain_off_position only; CI on the CRN-paired lift) was run to resolve "real edge vs noise."

**Run O high-power (40-seed) — RESOLVED, still NO-GO.** 24/24 chunks; R8 gate holds (control espn 0.620 / model 0.589 ≈ Run N). Paired lift with 95% CI over seats (adopt criterion pre-registered as **CI-separated from 0 in BOTH markets**):

| market | 20-seed Δ | 40-seed Δ | 95% CI | seats + | verdict |
|---|---|---|---|---|---|
| model | +0.0162 | **+0.0154** | **[+0.003, +0.028]** | 10/12 | **REAL (CI>0)** |
| espn | +0.0145 | **+0.0066** | [−0.003, +0.016] | 10/12 | **not sep. from 0** |

The espn lift appeared to **regress toward 0** with more data. **⚠️ This 40-seed run's CRN was compromised by a bug (see correction below); the firmed-up verdict lands back at NO-GO, but for the right reason — a model-market-only edge.**

**Run O CORRECTION (2026-07-15) — a CRN bug compromised the verdict; corrected + firmed-up = NO-GO (a real MODEL-market-only edge, not robust in ESPN).** A `/loop-review` of the Slice-2 diff caught a **CRN desync bug** in the `hero_nominator` hook: at `nomination_temp>0` the override path skipped the `_sample_nominee` `rng.choice` draw that the `control` path consumes, so the shared rng diverged after the hero's first nomination — breaking the CRN pairing for ~90% of every draft. That inflated the paired-lift CIs (the pairing that cancels shared bot-bid noise was mostly not happening), which is exactly why espn "spanned 0." Fixed (always draw the central nominee, then override; regression-tested — commit `f9ccb0e`), and the 40-seed probe was **re-run on the corrected engine** (24/24 chunks; control espn 0.620 / model 0.589, R8 holds):

| market | buggy-CRN Δ (95% CI) | **fixed-CRN Δ (95% CI)** | seats + | verdict |
|---|---|---|---|---|
| model | +0.0154 [+0.003, +0.028] | **+0.0135 [+0.0065, +0.0206]** | 10/12 | **REAL (CI>0)** |
| espn | +0.0066 [−0.003, +0.016] | **+0.0079 [+0.0002, +0.0156]** | 9/12 | **REAL (CI>0)** |

With correct pairing the espn CI initially tightened to **[+0.0002, +0.0156]** at 40 seeds — both markets *barely* CI-separated, a **marginal GO**. But espn's +0.0002 lower bound was knife-edge (~p=0.05), so an **80-seed firm-up** was run to test its robustness:

| market | 40-seed (fixed CRN) | 80-seed firm-up | seats + | verdict |
|---|---|---|---|---|
| model | +0.0135 [+0.0065, +0.0206] | **+0.0096 [+0.0020, +0.0173]** | 10/12 | **REAL (CI>0)** |
| espn | +0.0079 [+0.0002, +0.0156] | **+0.0052 [−0.0007, +0.0110]** | 9/12 | **not sep. from 0** |

espn **regressed back across 0** (lower bound +0.0002 → −0.0007) as its estimate shrank (+0.0079 → +0.0052) — the knife-edge did not hold. Model held (real, robust). **Final verdict: NO-GO** — `drain_off_position` is a genuine, robust edge in the **symmetric model market only** (+0.010, CI-separated across 40→80 seeds); in the realistic **ESPN market it is not distinguishable from zero** once firmed up. The pre-registered both-markets criterion fails at espn, so the bottom line matches the original Run O (model-specific, not adopted) — but now on **rigorous** footing (correct CRN, 80 seeds) instead of the buggy 40-seed run that reached NO-GO for the *wrong* reason. `balanced` p0.0 stays the hero; the `hero_nominator` hook + `nomination.py` remain a **validated opt-in probe** (real in model, not robust in espn), not wired into the default. **Methodological note:** the interim verdict flipped twice — buggy NO-GO → corrected marginal GO → firmed-up NO-GO — a case study in why CRN correctness *and* adequate power both matter (the loop-review CRN catch and the 80-seed firm-up each changed it). Artifacts: `reports/_nom_probe/2026_hp80/*.json` (definitive); `2026_hp/` (40-seed corrected), `2026_hp_precrn/` (pre-fix, invalid). Live-draft call still September.

**Run P — 2026-07-16 — ADP-driven nomination (market realism fix) → the strategy ranking is LARGELY a nomination-model artifact** (`half_12team`, all 11 `_MODELS` × 12 seats × both markets, 20 seeds × 300 sims; byes OFF; branch `feat/auction-adp-nomination`). Motivated by an eye-test failure: value-weighted-random nomination weights by OUR model's `auction_dollars`, so players our model under-rates fall implausibly late — Justin Jefferson (consensus ADP **12.4**, ESPN value $47) cleared **$24 at pick 85 of 204**. Added an opt-in `market_adp_jitter` engine option: flush seats nominate via a shared noisy-ADP "market board" (reuses `SnakeBoard`, noise drawn once/draft) — the room nominates roughly in ADP order with human randomness — instead of value-weighted-random. Under it (jitter 12) JJ goes **pick 8 / $61** (realistic). Re-ran the full-field bake-off vs the value-nomination baseline (Run L worst-cases, `balanced` from Run N p0.0):

| hero | value-nom worst | ADP-nom worst (espn/model) | Δ |
|---|---|---|---|
| balanced (p0.0) | 0.592 | **0.593** (0.684 / 0.593) | +0.001 |
| balanced_flat | 0.487 | 0.593 (0.684 / 0.593) | +0.106 |
| anchors | **0.156** | **0.555** (0.564 / 0.555) | **+0.399** |
| static | 0.396 | 0.549 | +0.153 |
| inflation | 0.408 | 0.548 | +0.140 |
| studsdepth | 0.378 | 0.542 | +0.164 |
| vorpshare | 0.358 | 0.540 | +0.182 |
| overbid | 0.388 | 0.539 | +0.151 |
| patient_deep | 0.474 | 0.530 | +0.056 |
| patient | 0.373 | 0.529 | +0.156 |
| marginal | 0.237 | 0.472 | +0.235 |

**Findings (data):** (1) EVERY hero improved; the field **compressed** to ~0.47–0.59. Heroes already good under value nomination barely moved (`balanced` +0.001, `patient_deep` +0.056); *collapsed* heroes recovered massively — **`anchors` (stars-and-scrubs) went LAST → #3 (0.156 → 0.555)**, `marginal` +0.24, all stud-buyers ~+0.15. (2) The ranking **scrambled** (`anchors` last→#3; `patient_deep` #3→#9). (3) **The prior "disciplined breadth dominates, stud-buyers collapse" conclusion (Runs E/L) was substantially a nomination-model artifact** — under a realistic ADP board, stud-concentration is viable (studs surface early, when a stars-and-scrubs hero can actually buy them). (4) `balanced` p0.0 remains the (bare) worst-case leader (0.593) — ESPN 0.554→0.684 (up, the aggressive bots overpay for early studs, softening the mid-market for the paced hero), model 0.593 (flat) — but its margin over #2 collapsed **+0.12 → +0.04** (inside 20-seed noise). **Caveats:** single jitter (12 — compression may be jitter-sensitive, unswept), 2026 only, byes off, the uncapped-`Aggressive`-bot overpay drives much of the ESPN effect (bot calibration is a standing open item), and the field-compression makes heroes near-indistinguishable at 20 seeds. **ADP nomination is a candidate realism-baseline for future runs — a methodology call, NOT made here.** No strategy adopted/rejected. Artifacts: `reports/_seat_sweep_adp/2026/*.json` (untracked). Live-draft call still September.

**Run Q — 2026-07-16 — big-stack overspend hero A/B → the overspend lever DOES lift reg-win, but its benefit is concentrated in the (circular) model market; the less-circular ESPN market disfavors the aggressive variant** (`half_12team`, all 12 seats × both markets, 20 seeds × 300 sims; byes OFF; **ADP nomination** `market_adp_jitter=12` — Run P's realism fix; branch `feat/auction-bigstack-overspend`; spec+plan `docs/superpowers/{specs,plans}/2026-07-16-auction-bigstack-overspend*`). Motivated by the Run-P eye-test: `balanced` p0.0 caps every bid at the LOW pace share (`2×budget/open_slots`), never buys studs, and ends with idle budget (miss rosters left ~$69). `BigStackBid` OVERPAYS above fair value (and lifts the pace cap) by `overpay = gain·max(0, advantage−1)`, where `advantage>1` iff the hero holds more remaining budget than the field — deploying a lead instead of stranding it. Two "am I the big stack?" signals × a gain grid, vs the `balanced` p0.0 control (7 contestants, bid-only A/B; nomination/field/seeds/sims held fixed):

- **`max_opp`** — `my_budget / richest opponent's budget` (one hoarding opponent flattens the signal).
- **`field_avg`** — `my per-slot budget / league-average per-slot budget` (robust to a single hoarder).

reg_win_pct (seat-avg over 12 seats; **worst** = min over markets, the robust-win goal metric):

| contestant | espn | model | worst | Δworst vs balanced |
|---|---|---|---|---|
| bigstack_field_avg_g0.5 | 0.651 | **0.666** | **0.651** | **+0.058** |
| bigstack_field_avg_g1.0 | 0.653 | 0.649 | 0.649 | +0.056 |
| bigstack_max_opp_g2.0 | 0.689 | 0.616 | 0.616 | +0.023 |
| bigstack_field_avg_g2.0 | 0.646 | 0.609 | 0.609 | +0.016 |
| bigstack_max_opp_g0.5 | 0.685 | 0.606 | 0.606 | +0.013 |
| bigstack_max_opp_g1.0 | **0.690** | 0.606 | 0.606 | +0.013 |
| balanced (p0.0, control) | 0.684 | 0.593 | 0.593 | — |

Δ vs balanced (espn / model), all three metrics:

| contestant | reg_win | make_playoffs | champ |
|---|---|---|---|
| field_avg g0.5 | −0.033 / **+0.073** | −0.054 / **+0.135** | −0.052 / **+0.105** |
| field_avg g1.0 | −0.030 / **+0.056** | −0.054 / **+0.107** | −0.051 / **+0.075** |
| field_avg g2.0 | −0.038 / +0.016 | −0.064 / +0.032 | −0.068 / +0.020 |
| max_opp g0.5 | +0.001 / +0.014 | +0.002 / +0.029 | +0.003 / +0.017 |
| max_opp g1.0 | +0.006 / +0.013 | +0.007 / +0.030 | +0.011 / +0.016 |
| max_opp g2.0 | +0.006 / +0.023 | +0.006 / +0.046 | +0.009 / +0.026 |

**Sanity gate PASSES:** `balanced` (control) reproduces Run P **exactly** (espn 0.684 / model 0.593) — the bid-only A/B is apples-to-apples.

**Findings (data — nothing adopted):**
1. **The overspend lever works on the goal metric:** *every* `BigStackBid` variant beats `balanced` on worst-case reg_win (all ≥ 0.606 vs 0.593). This is the first hero we have found that lifts the **model-market** floor above `balanced` — a mild counter to the "a field pricing off your own numbers is unbeatable by construction" reading (a single strategy can still beat the field *average*).
2. **Two distinct behaviors.** `field_avg` **trades ESPN for model**: it raises the model floor a lot (reg_win +0.06–0.07, playoff +0.11–0.14, champ +0.08–0.11 at gain 0.5–1.0) but pays a consistent **ESPN tax** (reg_win −0.03–0.04, playoff/champ −0.05–0.07). Its worst-case wins *because* it lifts its own weak market (model) above `balanced`'s weak market — a floor-raise, ceiling-cost. `max_opp` is a **smaller both-market-non-negative** lift (ESPN ≈ flat, model +0.01–0.02 reg_win), no market taxed.
3. **Gain tuning:** for `field_avg`, **lower gain is better** (g0.5 ≥ g1.0 ≫ g2.0) — g2.0 over-deploys and gives back most of the model gain while deepening the ESPN tax. For `max_opp` the gain barely matters (the trigger fires rarely). Best worst-case overall: **`field_avg` g0.5 (0.651)**.
4. **Circularity caveat — this is the important one.** The **model** market is highly self-referential: bots price off our SOS `auction_dollars`, the hero prices off the same numbers, AND `project_draft` scores the resulting rosters with the same projections. `field_avg`'s big *model* gain is therefore partly a shared-worldview artifact — the hero overspends on players our model loves, judged by a scorer that shares that love. The **ESPN** market is the less-circular test (bots price off ESPN values, hero+scorer off our model), and there the aggressive `field_avg` overspend **loses** on all three metrics; only the conservative `max_opp` stays neutral. So the apparent "overspend helps" is strongest exactly where the measurement is most self-confirming, and weakest/negative where it is most independent. Read `max_opp`'s tiny both-market lift as the more trustworthy signal, and `field_avg`'s model surge with heavy discount.
5. **Mechanism (hypothesis, consistent with the split):** `field_avg` keys off the league-average budget — in ESPN the uncapped `Aggressive` bots overpay early, dropping the average, so the hero reads itself as the big stack and overpays *into an already-inflated market* (−EV); in the model market bots leave budget on the table, so the hero genuinely is the big stack and converts idle cash to talent (+EV, the thesis). `max_opp` keys off the single richest opponent, who in ESPN is usually still flush early, suppressing the trigger — which is why it dodges the ESPN tax.

**Caveats:** single jitter (12), 2026 pool only, 12-team half-PPR, byes OFF, ESPN signal diluted (241/578 priced), and — unlike Run O — this is an **unpaired** seat-avg A/B (no CRN pairing), so the small `max_opp` deltas (≈ the ±0.03/20-seed per-cell noise band; the seat-avg is tighter but these are near it) are not resolved from zero; a CRN-paired re-run would tighten them. `field_avg`'s model gains (+0.05–0.07) and its ESPN tax (−0.03–0.04, consistent across all three gains) both clear the band. Unspent-budget was **not directly measured** (the summaries expose only win/playoff/champ/points), so "converts idle cash to talent" is inferred from the market split, not observed.

**No strategy adopted or rejected.** `balanced` p0.0 remains the shipped default; `BigStackBid` is a **validated opt-in** (unit-tested, engine-solvent, not registered in `tournament_cli._MODELS`). In isolation the data favors `field_avg` g0.5 on the worst-case goal metric and `max_opp` as the conservative both-market-safe deploy — but the circularity caveat means neither is a live-draft recommendation. Next candidate probes: a CRN-paired re-run to resolve `max_opp`; a jitter sweep; and hardening the ESPN market (bot calibration) so the less-circular signal carries more weight. Artifacts: `reports/_bigstack/2026/*.json` (untracked). Live-draft call still September.

**Run R — 2026-07-16 — stack-ratio CONVEX-aggression sweep → convexity FIXES the linear ESPN penalty and wins the goal metric, but the roster-shape trace REFUTES the "deploy into late depth" mechanism (it damps early over-aggression instead)** (`half_12team`, all 12 seats × both markets, 20 seeds × 300 sims; byes OFF; **ADP nomination** `market_adp_jitter=12`; branch `feat/auction-stack-ratio-bid`; spec+plan `docs/superpowers/{specs,plans}/2026-07-16-auction-stack-ratio-bid*`). Motivated by the Run-Q ESPN-degradation trace (`espn_overspend_trace.py`): `field_avg`'s linear cap-lift fires early (bots drain the field's budget early → advantage spikes by Q1) and chases early studs. **User's principle:** make aggression a CONVEX function of the raw budget ratio so it stays disciplined at a moderate lead and only unleashes at a dominant one. `StackRatioBid`: `mult = 1 + gain·max(0, ratio−1)^curve`, `ratio = my_budget / mean(opponent budgets)`, lifts BOTH target and cap; reduces to `balanced` at `ratio ≤ 1`. `curve=1` recovers the linear (field_avg-style) ramp; `curve>1` is convex. Swept `gain {0.5,1,2} × curve {1,2,3}` + `balanced` control (10 contestants, bid-only A/B). Crash-safe; 24 chunks clean (~104 min).

reg_win_pct (seat-avg; **worst** = min over markets):

| contestant | espn | model | worst | Δworst vs balanced |
|---|---|---|---|---|
| sr_g0.5_c2 | 0.679 | 0.670 | **0.670** | **+0.077** |
| sr_g0.5_c1 (linear) | 0.666 | 0.675 | 0.666 | +0.073 |
| sr_g2.0_c3 | 0.665 | 0.664 | 0.664 | +0.071 |
| sr_g0.5_c3 | **0.686** | 0.661 | 0.661 | +0.068 |
| sr_g1.0_c2 | 0.664 | 0.658 | 0.658 | +0.065 |
| sr_g1.0_c3 | 0.676 | 0.657 | 0.657 | +0.064 |
| sr_g1.0_c1 (linear) | 0.667 | 0.645 | 0.645 | +0.052 |
| sr_g2.0_c2 | 0.667 | 0.641 | 0.641 | +0.048 |
| sr_g2.0_c1 (linear) | 0.648 | 0.608 | 0.608 | +0.015 |
| balanced (control) | 0.684 | 0.593 | 0.593 | — |

**Sanity gate PASSES:** `balanced` reproduces Run Q **exactly** (espn 0.684 / model 0.593).

**Findings (data — nothing adopted):**
1. **Every stack-ratio variant beats `balanced` on the worst-case goal metric** (all ≥ 0.608 vs 0.593); **low gain (0.5) + convex is best** — `sr_g0.5_c2` = 0.670 worst-case (+0.077), the best hero found on this metric to date (beats `BigStackBid` `field_avg` from Run Q). High gain (2.0) is worst (over-aggressive), mirroring the Run-J/Q "low aggression wins."
2. **Convexity fixes the linear ESPN penalty.** ESPN Δ vs balanced by curve: the **linear** `curve=1` variants all LOSE ESPN (−0.017 to −0.036, reproducing `field_avg`); the **convex** ones claw it back to ~even (gain 0.5: c1 −0.018 → c2 −0.005 → c3 **+0.002**). So convexity does exactly what the user predicted for the *win metric* — it removes the early-stud-chasing penalty.
3. **BUT the roster-shape trace (`stackratio_shape.py`, seat 1, 10 seeds, spend share by draft quartile) REFUTES the intended "deploy surplus into late depth" mechanism.** The stack-ratio hero (every curve) is MORE front-loaded than `balanced` — ESPN Q1 share 0.64–0.67 vs balanced's 0.39, at higher top-5 concentration (0.58–0.60 vs 0.50), leftover $0 vs $10. Convexity only *marginally* softens the front-loading (Q1 0.67→0.64, top5 0.60→0.58). **Reason:** in ESPN the aggressive bots overpay early → the field's budget drains early → the ratio spikes by mid-Q1 → the multiplier fires *early* regardless of curve; convexity delays it a touch (enough to remove the ESPN penalty) but cannot push deployment to Q3/Q4, and second-price makes genuine late deployment moot. So convexity's `reg_win` gain comes from **damping early over-aggression**, NOT from depth-deployment. The surplus is deployed (leftover $0 vs $10) but into *earlier, more concentrated* buys — just slightly less so than linear.

**Caveats:** the worst-case improvement is concentrated in the (circular) model market (field_avg + scorer share our valuations); in the less-circular ESPN market the convex variants are ~EVEN with `balanced` (within the ±0.03 band), not clearly better — so this "deploys the surplus without the ESPN penalty" rather than "clearly beats balanced where it matters most." One-seat roster-shape trace; single jitter (12); 2026 pool only; byes off. **Deferred alternatives (noted for later): per-slot budget ratio, and a pure power-law multiplier family.**

**No strategy adopted or rejected.** `balanced` p0.0 remains the shipped default; `StackRatioBid` is a **validated opt-in** (unit-tested, engine-solvent, not in `_MODELS`). In isolation the data favors `sr_g0.5_c2` (low-gain convex) on the worst-case metric — the best hero found so far — but the ESPN wash + model circularity mean it is not a live-draft recommendation, and its mechanism is early-aggression-damping, not the depth-deployment the user was after. Artifacts: `reports/_stackratio/2026/*.json` (untracked). Live-draft call still September.

**Run S — 2026-07-17 — finer GAIN sweep of the convex StackRatioBid → low gain (~0.2) convex is the FIRST hero to WIN the less-circular ESPN market (not just draw even)** (`half_12team`, all 12 seats × both markets, 20 seeds × 300 sims; byes OFF; ADP nomination `market_adp_jitter=12`; branch `feat/auction-stack-ratio-bid`; scratch `stackratio_gain_sweep.py`). Motivated by Run R: convex curves clawed the linear ESPN penalty back to ~even — so does pushing gain BELOW Run R's best (0.5) actually WIN ESPN? Swept `gain {0.01, 0.2, 0.4, 0.6, 0.8, 0.99} × curve {2, 3}` (the convex curves) + `balanced` control (13 contestants). Crash-safe; 24 chunks clean (~144 min). Sanity: `balanced` reproduces (espn 0.684 / model 0.593).

reg_win_pct (seat-avg; **worst** = min over markets):

| contestant | espn | Δespn | model | Δmodel | worst |
|---|---|---|---|---|---|
| sr_g0.2_c2 | **0.693** | **+0.010** | 0.669 | +0.076 | 0.669 |
| sr_g0.2_c3 | 0.692 | +0.009 | 0.664 | +0.071 | 0.664 |
| sr_g0.8_c2 | 0.674 | −0.010 | 0.672 | +0.079 | **0.672** |
| sr_g0.4_c2 | 0.684 | +0.000 | 0.668 | +0.075 | 0.668 |
| sr_g0.6_c3 | 0.685 | +0.001 | 0.664 | +0.071 | 0.664 |
| sr_g0.01_c3 | **0.698** | **+0.014** | 0.610 | +0.017 | 0.610 |
| sr_g0.01_c2 | 0.693 | +0.009 | 0.597 | +0.004 | 0.597 |
| sr_g0.6_c2 | 0.662 | −0.021 | 0.665 | +0.072 | 0.662 |
| sr_g0.99_c2 | 0.666 | −0.018 | 0.656 | +0.063 | 0.656 |
| balanced | 0.684 | — | 0.593 | — | 0.593 |

(gain 0.8/0.99/0.6-c2 omitted-as-worse rows folded above; full data in artifacts.)

**Findings (data — nothing adopted):**
1. **A clean gain THRESHOLD for the ESPN sign:** `gain ≤ 0.2` **wins** ESPN (Δ +0.009 to +0.014), `gain = 0.4` is **even** (±0.000), `gain ≥ 0.6` **loses** (−0.007 to −0.021). Lower gain = gentler aggression = less early-stud-chasing = better ESPN. Within a gain, **`curve=3` protects ESPN better than `curve=2`** at higher gains (e.g. g0.6: c3 +0.001 vs c2 −0.021), consistent with more convexity deferring aggression.
2. **`sr_g0.2_c2` wins BOTH markets** — espn +0.010, model +0.076, worst-case 0.669 (near the top). This is the **first hero in the whole investigation to beat `balanced` in the less-circular ESPN market** while also gaining in model. The ESPN edge is clearer on **champ%** (+0.019) and playoff% (+0.008) than on reg_win, and it is **consistent across all low-gain convex variants** (g0.01/g0.2, both curves, all three metrics positive in ESPN) — which argues it is a real small edge, not a single noisy cell.
3. **Diminishing returns at very low gain:** `gain = 0.01` wins ESPN by the most (+0.014) but barely deploys in model (+0.004 to +0.017) — it is essentially `balanced` with a whisker of aggression. So **`gain ≈ 0.2` is the sweet spot** (wins ESPN AND banks the model gain), not the lowest gain.

**Caveats:** the ESPN reg_win win is small (~+0.010, near the seat-averaged noise floor) — the champ/playoff deltas and the cross-variant consistency are what make it credible, not any single cell. The model gains still carry the circularity caveat (StackRatioBid + the scorer share our valuations), but the ESPN win does NOT (bots price off ESPN there), so it is the more meaningful signal. Single jitter (12), 2026 pool, byes off.

**No strategy adopted or rejected.** `balanced` p0.0 remains the shipped default. But **`sr_g0.2_c2` (StackRatioBid gain=0.2, curve=2) is now the leading candidate** in the whole auction investigation: it is the only hero to beat `balanced` in BOTH markets, including the honest ESPN one — a genuine (if small) edge, unlike the model-only/circular gains of every prior overspend variant. Recommended next: a CRN-paired re-run at gain ∈ {0.1, 0.2, 0.3} × curve {2,3} to tighten the small ESPN deltas, and the deferred axes (per-slot ratio, power-law). Artifacts: `reports/_stackratio_gain/2026/*.json` (untracked). Live-draft call still September.

**Run T — 2026-07-17 — CRN-PAIRED re-run of the low-gain grid → confirms `sr_g0.2_c2`'s ESPN edge is REAL, not noise: the paired 95% CI on reg-win% AND champ% EXCLUDES ZERO in the less-circular market** (`half_12team`, all 12 seats × both markets, 20 seeds × 300 sims; byes OFF; ADP nomination `market_adp_jitter=12`; branch `feat/auction-stack-ratio-bid`; scratch `stackratio_paired.py`). Motivated by Run S: the ESPN reg-win edge (+0.010) sat near the seat-averaged noise floor, so Run S leaned on the champ delta + cross-variant consistency rather than a resolved reg-win CI. **This run resolves it.** Key realization: `run_auction_tournament` is *already* common-random-numbers paired — every contestant plays the identical auction draw (`default_rng(base_seed+s)`) and identical season draw (`default_rng(season_base_seed+s)`) per seed, and already returns `paired_diffs = bootstrap(per[a] − per[b])`. Run S **discarded** those and compared two overlapping *marginal* CIs; this run **records the paired diff** (contestant − `balanced`), where the shared-world variance cancels — a far tighter test. Tight grid `gain {0.1, 0.2, 0.3} × curve {2, 3}` + `balanced` control (7 contestants). Crash-safe; 24 chunks clean (~50 min).

**Reproducibility anchor PASSES exactly:** because seed(name, s) depends only on `base_seed+s` (not the contestant set), `balanced` (espn 0.684 / model 0.593) and `sr_g0.2_c2` (espn 0.693 / model 0.669) reproduce **Run S bit-for-bit** — the pipeline is verified consistent and the CRN pairing is genuinely shared.

**ESPN market (the meaningful, less-circular one) — paired diff `contestant − balanced`, seat-stratified 95% CI. `*` = excludes 0.**

| contestant | reg_win% | champ% | make_playoffs% | bye% |
|---|---|---|---|---|
| sr_g0.1_c2 | +0.0093 [+.0041,+.0145] `*` | +0.0178 [+.0085,+.0271] `*` | +0.0075 [+.0002,+.0149] `*` | +0.0268 `*` |
| sr_g0.1_c3 | +0.0084 [+.0029,+.0140] `*` | +0.0151 [+.0057,+.0244] `*` | +0.0072 [−.0007,+.0151] | +0.0268 `*` |
| **sr_g0.2_c2** | **+0.0096 [+.0037,+.0154] `*`** | **+0.0192 [+.0090,+.0293] `*`** | +0.0081 [−.0004,+.0166] | +0.0257 `*` |
| sr_g0.2_c3 | +0.0087 [+.0038,+.0135] `*` | +0.0161 [+.0067,+.0254] `*` | +0.0085 [+.0015,+.0156] `*` | +0.0216 `*` |
| sr_g0.3_c2 | +0.0070 [+.0018,+.0123] `*` | +0.0119 [+.0024,+.0213] `*` | +0.0059 [−.0016,+.0134] | +0.0179 `*` |
| sr_g0.3_c3 | +0.0048 [−.0007,+.0104] | +0.0095 [+.0003,+.0187] `*` | +0.0031 [−.0047,+.0109] | +0.0145 `*` |

**Model market (circular):** every variant × metric is strongly positive and excludes 0 — e.g. `sr_g0.2_c2` reg-win +0.0762 [+.0686,+.0838], champ +0.1138 [+.1038,+.1239]. Consistent with Runs Q–S; carries the circularity caveat, so not the load-bearing signal.

**Findings (data — nothing adopted):**
1. **The ESPN edge is real for `sr_g0.2_c2`.** Its paired reg-win% diff (+0.0096, CI [+0.0037, +0.0154]) and champ% diff (+0.0192, CI [+0.0090, +0.0293]) **both exclude zero** — what Run S's marginal comparison could not establish. The CRN pairing shrinks the reg-win CI to ±0.006 (vs the ≈±0.03/20-seed marginal band), which is what resolves a +0.010 effect. bye% is also clearly positive (+0.0257 `*`). make_playoffs% is directionally positive (+0.0081) but its CI just touches zero (lo −0.0004) — **not resolved**.
2. **It is a family effect, not a lucky cell.** Across the six low-gain convex variants: reg-win clears 0 in **5/6** (only the highest gain×curve, `sr_g0.3_c3`, straddles), champ in **6/6**, bye in **6/6**. A gentle monotone confirms Run S's tuning: the edge fades as gain rises (0.1–0.2 > 0.3) and `curve=2` ≥ `curve=3` on reg-win/champ. `sr_g0.2_c2` is the strongest reg-win + champ cell — the same winner Run S named.
3. **`sr_g0.2_c2` reproduces as the leading candidate under a stricter test.** It is the only hero in the investigation to beat `balanced` in the honest ESPN market, and that win now survives a paired-CI significance check on two of the three goal metrics (reg-win, champ), not just a point-estimate.

**Method note:** each (seat, market) chunk yields a 20-seed paired bootstrap CI per contestant; the 12 seats are combined as **fixed strata** (weight 1/12 — the goal metric's definition), per-seat SE recovered as CI-width/(2·1.96) and pooled as `SE = √(Σ SEₛ²)/12`. This is exact for independent seats under a normal approximation; the approximation is well-justified here — per-seat CIs are near-symmetric (median asymmetry 6%, max 27%). A full stratified *resampling* bootstrap would need the raw per-seed vectors, which the tournament does not currently expose (it returns only bootstrapped Intervals); exposing them is a possible small follow-up if a tighter method is ever wanted.

**Caveats:** the edge is **real but small** (~+0.010 reg-win, ~+0.019 champ) — statistically resolved, not large. make_playoffs% remains unresolved (CI touches 0). Everything is still single jitter (12), 2026 pool only, 12-team half-PPR, byes OFF, ESPN signal diluted (241/578 priced). The model-market surge keeps its circularity caveat; only the ESPN result is load-bearing, and that is the one now confirmed. Unspent-budget still not directly measured.

**No strategy adopted or rejected.** `balanced` p0.0 remains the shipped default. `StackRatioBid` stays a **validated opt-in** (unit-tested, engine-solvent, not in `_MODELS`). In isolation the data now favors **`sr_g0.2_c2`** more firmly than after Run S: its ESPN advantage on reg-win% and champ% is significant under CRN pairing, making it the strongest genuinely-non-circular hero found — but it is one pool, one jitter, one season, and the September call still stands. Remaining probes (unchanged): jitter sweep + multi-year re-run to test robustness of the small edge; the deferred per-slot-ratio and power-law axes; direct unspent-budget measurement. Artifacts: `reports/_stackratio_paired/2026/*.json` (untracked). Live-draft call still September.

**Run U — 2026-08-12 — value-gap nomination probe, WILL'S LEAGUE → NO-GO, and the sign is NEGATIVE: for a stud-buying hero, poison nomination COSTS more than it drains** (`will_half12`, all 12 seats, **ESPN market only**, 20 seeds × 300 sims; byes OFF; ADP nomination `market_adp_jitter=12`; field `overbidder` (`overbid=0.2`, `pace=4.5`, basis `opening`, `pace_jitter=0.35`); branch `feat/auction-value-gap-nomination`; spec `docs/superpowers/specs/2026-08-12-auction-value-gap-nomination-design.md`). Motivated by a challenge to Run O's reading: both of its heuristics ranked by **price**, so neither tested the actual poisoning premise — *nominate the players the room overvalues relative to us*. That also explains Run O's own null on `drain_max` (+0.001): the room already nominates in value/ADP order, so "nominate the priciest" mostly surfaces whoever was coming up next anyway, and the hero adds no information. Run O additionally predates the Run-P ADP realism fix and ran the generic field on `half_12team`, not Will's room.

New heuristics (`nomination.py`): **`gap`** = `argmax(bot_dollars − auction_dollars)` (the room's overpay vs our board, absolute dollars — a ratio would rank a $3 player priced at $9 above a $40 player priced at $55) and **`gap_off`** = the same argmax restricted to positions the hero has already filled, composing the disagreement signal with the one thing Run O found that worked. ESPN-only by construction: under model pricing the room prices off our own numbers, so every gap is identically zero (the runner refuses `--bot-prices model`). Contestants all bid **`overbid_noramp`** — the plan in the committed Will-league guide — so the probe isolates nomination. Raced via the new `run_auction_tournament(hero_nominators=...)` seam, which makes the four contestants CRN-paired for free (identical auction + season draws per seed; `paired_diffs` cancels the shared-world variance).

| contestant | seat-avg reg_win | paired Δ vs control (95% CI) | champ Δ | seats better |
|---|---|---|---|---|
| `control` (no hook) | **0.6389** | — | — | — |
| `gap` | 0.6345 | −0.0044 [−0.0115, +0.0026] | −0.0019 | 3/12 |
| `off_pos` (Run-O incumbent) | 0.6292 | **−0.0097 [−0.0171, −0.0023]** | −0.0092 | 2/12 |
| `gap_off` | 0.6248 | **−0.0141 [−0.0213, −0.0070]** | **−0.0155 [−0.0254, −0.0056]** | 1/12 |

**R7 sanity gate PASSES exactly:** `control` reproduces the committed `overbid_noramp` Will-league seat-average (`reports/_noramp_ab/espn.json`) to four decimals — 0.6389 vs 0.6389. The harness is sound and the paired lift is apples-to-apples.

**Verdict — NO-GO, and a stronger result than Run O's.** The pre-registered bar was a positive CI excluding zero; the measured effect is *negative* and CI-separated for both off-position variants. `gap_off` — the variant the hypothesis predicted should win, composing both signals — is the **worst** contestant, worse than control at **11 of 12 seats**, and the only one whose champ% penalty also excludes zero. The pure `gap` heuristic is merely flat (CI spans 0). The ordering is monotone in how aggressively the hero gives up its own nomination turn.

**Mechanism (the reconciliation with Run O).** Run O's hero was `balanced` p0.0 — pace-capped, never bought a stud, ended with ~$69 idle. For that hero a nomination turn was nearly worthless: it was going to lose the top lot regardless, so spending the turn on a decoy was close to free, and `drain_off_position` bought a small (+0.010, model-market-only) edge. `overbid_noramp` is the opposite hero: it pays up to 1.3× for studs and deliberately spends out early (~$180 on three RBs). For *it*, the engine's default value-first nomination (`candidates[0]`) is not a wasted turn — it surfaces the best player available at the moment the hero most wants to buy one. Poisoning trades that away. **The drain the poison buys is real but is money the room would have spent anyway** (every seat ends at $0 in this field); the buying opportunity it forfeits is not recoverable. So the lever's sign depends on the hero: near-free for a hero that cannot buy, actively costly for one whose whole plan is to buy early.

**What this does and does not settle.** It refutes the value-gap hypothesis **for this hero in this room** — the premise "drain the room and the RBs get cheaper" fails because budget is conserved and the drain is untargeted, while the nomination turn given up is a direct cost. It does not establish that nomination is worthless in general; a hero that genuinely cannot compete early (Run O's `balanced`) still showed a small positive. Single market by construction (the gap is undefined in the model market), so the pre-registered bar was a single-market bar and this verdict inherits that weaker footing — though the effect being *negative* at 11/12 seats makes it a robust null for the adopt question either way. Caveats otherwise as Run P onward: 2026 pool, single ADP jitter (12), byes OFF, ESPN signal diluted (241/578 priced), modeled field not Will's actual league-mates.

**Conclusion (data; no default change):** the committed guide's plan — `overbid_noramp` with the engine's default value-first nomination — stands unchanged, and is now positively supported on the nomination axis rather than merely untested. `nomination.py` keeps all four heuristics as tested opt-ins; none is wired into a default. Artifacts: `reports/_gap_nom_probe/will_2026/*.json` (untracked). Live-draft call still September.

> ⚠️ **Run U's generalization is RETRACTED by Run V (below).** The measured numbers above stand — for `overbid_noramp`, `gap` is flat and both off-position variants are harmful. But the *explanation* offered for them ("the lever's sign depends on whether the hero buys early"; "nomination poisoning is closed for stud-buying heroes") is **wrong**. Run V raced the same four nominators across three more heroes and found the value-gap signal **positive and CI-separated for three of the four**, including `static` — a hero that spends 80% of its budget early and ends at $0, i.e. a stud-buyer by any measure, and the cell with the **largest** gain in the whole study. The early-spend axis does not determine the sign. Read Run V for the corrected mechanism; treat the Run U conclusion as one cell of a grid, not a rule.

**Run V — 2026-08-12 — the same probe across FOUR heroes → the value-gap hypothesis is CONFIRMED (3 of 4 heroes, CI-separated); Run U was an `overbid_noramp`-specific null, and the real split is between the two SIGNALS, not between the heroes** (`will_half12`, all 12 seats, ESPN market, 20 seeds × 300 sims, byes OFF, ADP nomination `market_adp_jitter=12`, `overbidder` field at `overbid=0.2 / pace=4.5 / opening / pace_jitter=0.35` — Run U's settings exactly, only the hero varies; branch `feat/auction-value-gap-nomination`). Motivated by Run U's proposed mechanism, which made a falsifiable prediction: if poison nomination is costly *because* it forfeits a buying turn, then heroes that cannot buy early should gain from it.

**The "can it buy early?" axis was measured, not assumed** (engine `PickRecord` trace, 4 seats × 12 seeds per hero, same field):

| hero | mean top buy | early spend (first ¼ of picks) | $30+ buys | end cash |
|---|---|---|---|---|
| `overbid_noramp` | $77.1 | $181.1 (**91%** of budget) | 2.71 | $0.0 |
| `static` | $54.9 | $160.7 (**80%**) | 3.02 | $0.0 |
| `balanced` | $27.8 | $20.7 (**10%**) | 0.25 | $28.2 |
| `patient_deep` | $17.9 | $10.1 (**5%**) | 0.00 | $102.2 |

CRN-paired Δ `reg_win_pct` vs each hero's own no-hook control (95% CI; **bold** = excludes 0), with per-seat sign counts:

| hero (early spend) | `gap` | `gap_off` | `off_pos` | control level |
|---|---|---|---|---|
| `overbid_noramp` (91%) | −0.0044 [−.0115,+.0026] 3/12 | **−0.0141** [−.0213,−.0070] 1/12 | **−0.0097** [−.0171,−.0023] 2/12 | 0.6389 |
| `static` (80%) | **+0.0145** [+.0074,+.0216] 10/12 | +0.0013 [−.0065,+.0092] 7/12 | **−0.0077** [−.0152,−.0003] 4/12 | 0.6080 |
| `balanced` (10%) | **+0.0118** [+.0059,+.0176] 11/12 | **+0.0106** [+.0045,+.0167] 11/12 | **+0.0103** [+.0037,+.0170] 10/12 | 0.5971 |
| `patient_deep` (5%) | **+0.0057** [+.0004,+.0111] 9/12 | **+0.0064** [+.0010,+.0117] 11/12 | +0.0054 [−.0001,+.0109] 9/12 | 0.5388 |

**Findings (data):**

1. **The value-gap hypothesis is confirmed.** `gap` — nominate `argmax(bot_dollars − auction_dollars)`, i.e. whoever the room most overvalues relative to our board — is **positive and CI-separated for three of the four heroes**, at 9–11 of 12 seats each. This is the original poisoning premise, and it works. Run O never tested it (both its heuristics ranked by price) and Run U tested it on the one hero where it does not fire.
2. **Run U's mechanism is refuted.** `static` spends 80% of its budget in the first quarter, buys three $30+ players, and ends at $0 — and it posts the **largest gain in the study** (+0.0145). "Buys early" therefore does not predict the sign. Run U's `overbid_noramp` result was real but not generalizable.
3. **The real split is between the two SIGNALS, and it only shows up for heroes that buy.** For the two heroes that cannot buy (`balanced`, `patient_deep`), all three heuristics are statistically indistinguishable (+0.005 to +0.012) — a hero that is just waiting gains from *any* drain, and the choice of nominee barely matters. For the two heroes that do buy, the choice matters enormously: on `static`, the same hero in the same room on the same seeds swings from **+0.0145 (`gap`) to −0.0077 (`off_pos`) — a 0.022 spread**. **`gap` beats `off_pos` for all four heroes**, without exception.
4. **The off-position FILTER is the harmful part, not poisoning.** Restricting the nominee to positions the hero has already filled drags the pick toward categories it is done with; for a hero that has spent 80–91% of its budget, that is most of the board, so the filter surfaces cheap leftovers and wastes the turn. The disagreement signal alone keeps surfacing players the room will genuinely pay for. This reverses Run O's reading, where off-position targeting was the one thing that worked — for its pace-capped hero, which had no filled positions to speak of and so mostly hit the fallback path.
5. **No cell beats the shipped plan.** `overbid_noramp` + no hook (0.6389) is the best of the sixteen hero × nominator cells; the best poisoned cell is `static` + `gap` at 0.6225. Poisoning lifts *weak* heroes toward the strong one without reaching it — consistent with the drain being worth roughly one tier of bidding discipline, not more.

**Caveats:** one room (Will's `overbidder` field), one market (ESPN — the gap is undefined under model pricing), 2026 pool, single ADP jitter (12), byes OFF, ESPN signal diluted (241/578 priced), modeled field not real league-mates. The four heroes are not a random sample of strategy space; `overbid_noramp` remains an unexplained exception (its `gap` CI spans zero, so it is a null, not a demonstrated harm) and no diagnostic here isolates *why* it alone fails to profit — a worthwhile follow-up would trace which players `gap` actually nominates for it versus for `static`.

**Conclusion (data; no default change):** the guide's plan for Will's league — `overbid_noramp` with default value-first nomination — **stands**, now tested across a 4×4 hero × nominator grid rather than a single point, and confirmed as the one hero for which nomination control adds nothing. But the general question is now answered the other way from Run O and Run U: **value-gap nomination is a real, repeatable edge for most heroes**, worth ~+0.01 reg-win, and it is the *disagreement* signal that carries it — the price-ranked heuristics Run O tested were measuring the wrong thing. `nomination.py` keeps all four as tested opt-ins; none is wired into a default. Artifacts: `reports/_gap_nom_probe/will_2026{,_static,_balanced,_patient_deep}/*.json` (untracked). Live-draft call still September.

**Run W — 2026-08-12 — WHY `overbid_noramp` alone doesn't profit → poison nomination and bid aggression are SUBSTITUTES that reach the same ceiling; `k=1.3` is the one bid setting already at it** (`will_half12`, ESPN market, `overbidder` field, ADP nomination, byes OFF; diagnostics at 4 seats × 20 seeds, the confirmatory sweep at all 12 seats × 20 seeds × 150 season sims, CRN-paired; branch `feat/auction-value-gap-nomination`). Run V left one thing unexplained: `drain_value_gap` lifts three heroes but does nothing for `overbid_noramp`. Added `PickRecord.nominator_seat` (diagnostics-only; who put the lot up is otherwise unrecoverable — the engine advances the nominator pointer after every award) to measure it.

**Two hypotheses were formed and both were killed by measurement:**

1. **"The aggressive hero buys its own decoys."** *Refuted.* Under `gap` every hero buys its own nominations **less**, not more (`overbid_noramp` 1.54 → 1.25 lots, own-spend $21.96 → $1.35). The most aggressive bidder is not swallowing the poison.
2. **"The poison damages opponents' rosters."** *Refuted, and it explains a lot.* The drain is unmistakably real — the room pays **+$68** more on the hero's nominated lots ($23 → $92 of overpay vs our board). But mean opponent roster value moves by −0.04 / −0.02 / +0.39 dollars — i.e. **not at all**. In a fixed pool every player is bought by someone, so total roster value is conserved: **poison moves dollars, not players.** Any gain must come from the hero's own share, not from damage to the field.

**What actually happens.** The hero's roster *does* change substantially — ≈9 of 17 players differ between the control and poisoned drafts for both `static` and `overbid_noramp` — but only one of them converts it into points (paired, 4 seats × 20 seeds): `static` **+16.7 pts / +0.02 win**, `balanced` **+8.8 / +0.01**, `overbid_noramp` **+0.07 / +0.00**. The intervention is the same size for both; the conversion is not. The tell is the control level: `overbid_noramp` starts at **1368 pts**, above where `static` lands *even with* the poison (1349).

**Confirmatory sweep — the stud premium `k` is the aggression knob** (`OverbidValueBid(k, use_urgency=False)`; `k=1.0` pays plain value, `k=1.3` **is** `overbid_noramp`). All 12 seats, paired 95% CI on `reg_win_pct`:

| stud premium `k` | control win | with `gap` | paired Δ (95% CI) | seats + |
|---|---|---|---|---|
| 1.00 | 0.6256 | 0.6403 | **+0.0147 [+0.0077, +0.0218]** | 11/12 |
| 1.10 | 0.6216 | 0.6281 | +0.0065 [−0.0009, +0.0139] | 10/12 |
| 1.20 | 0.6215 | 0.6389 | **+0.0175 [+0.0093, +0.0256]** | 10/12 |
| **1.30** (`overbid_noramp`) | **0.6360** | 0.6342 | **−0.0019 [−0.0092, +0.0055]** | **6/12** |
| 1.50 | 0.6264 | 0.6320 | +0.0057 [−0.0021, +0.0134] | 10/12 |

`k=1.3` is the **only** cell that is not positive (6/12 seats — a coin flip) and simultaneously the **only** cell with a high control level (0.6360 vs ~0.622 for every neighbour). It reproduces the independent Run V probe figure (0.6389 seat-averaged) and the guide's finding that `overbid_noramp` is the strongest hero, so the peak is real, not a seed artifact.

**Findings (data):**
1. **The two levers are substitutes, not complements.** Every poisoned cell lands at 0.628–0.640; the best *unpoisoned* cell (`k=1.3`) is already at 0.6360. Poison nomination is an alternative route to the same ceiling that correct bid aggression reaches on its own, and at the aggression optimum it adds **nothing** (−0.002, CI spans 0).
2. **Nothing in the sweep beat the ceiling.** Best combined cell is `k=1.0` + `gap` at 0.6403 vs best control at 0.6360 — a +0.004 difference well inside its CI, i.e. **not a demonstrated improvement**. Consistent with Run V, where no hero × nominator cell beat `overbid_noramp` + no hook.
3. **This is why Run U looked like a null and Run O looked like a small win.** Both probed heroes sitting at very different points on this axis. Nomination poisoning pays exactly to the extent your bid is leaving value on the table.

**Caveats — the mechanism is supported but not nailed.** The `k` response is **not monotone** (`k=1.1` +0.0065 vs `k=1.2` +0.0175, and `k=1.5` +0.0057 with a CI touching zero), so there is real residual noise and no smooth "more aggression → less poison value" law is claimed; the robust fact is the single peculiar cell at `k=1.3`. The roster-composition and value-decomposition diagnostics ran at 4 seats × 20 seeds and their sub-1% deltas are at the noise floor — they are used here only for the two **refutations** (which are order-of-magnitude clear), not to support the positive claim. Season sims reduced to 150 for the sweep. One room, one market, one pool. **Not** established: why `k=1.3` specifically is the aggression optimum in this field.

**Conclusion (data; no default change):** the question "why isn't `overbid_noramp` better with this?" has an answer — **because it is already collecting the same edge by outbidding, and the two do not stack.** Will's plan is unchanged and now has a mechanism behind it rather than a bare measurement. Artifacts: scratch diagnostics (untracked); `PickRecord.nominator_seat` is committed and tested. Live-draft call still September.

**Run X — 2026-08-13 — the forced-negative-transaction argument is CORRECT; it is SATURATED in Will's room because the field is already broke by Q2, and it pays as predicted in a room that holds money** (`will_half12`, ESPN market, ADP nomination, byes OFF; quarter diagnostics at 4 seats × 20 seeds, the confirmatory probe at all 12 seats × 20 seeds × 300 sims, CRN-paired; branch `feat/auction-value-gap-nomination`). Prompted by a user challenge to Runs U–W: *budget is finite, so every dollar the room overpays on a player we don't want is a dollar it cannot spend on a player we do want; forcing negative-value transactions onto opponents must leave more positive value for us.* Runs V–W had measured the **drain** and the **outcome** but never the thing in between — **prices over the course of the draft**. This run measures it.

**The argument is right, and the mechanism is visible.** But so is its limit. Opponents' **total** cash (of 11 × $200 = $2,200) and the clearing price as a multiple of our value, by draft quarter:

| field | | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| `overbidder` (Will's modeled room) | opp cash | $367.6 | **$82.1** | $47.5 | $8.5 |
| | price / our value | 1.46× | 0.47× | **0.13×** | 0.92× |
| `balanced_field` (disciplined room) | opp cash | $1,296.7 | $731.6 | $542.9 | $493.8 |
| | price / our value | 0.79× | 0.89× | **0.65×** | 0.94× |

**In Will's modeled room the field has already spent 83% of its money by the end of Q1 and holds $82 across eleven teams by the end of Q2.** The late-draft bargain the argument predicts *already exists, at maximum magnitude*: Q3 lots clear at **13 cents on the dollar**. Every hero's surplus is earned there (`overbid_noramp` −$27.9 in Q1, +$62.6 in Q2, +$34.6 in Q3). There is no meaningful money left to force out — our ~8% of nominations moves ~$6–8 of opponent cash, against the ~$1,800 the bots spend out on their own. That is why the drain measured **+$68 of extra room overpay** in Run W and still produced no gain: it is a rounding error against a market that has already collapsed.

**Confirmatory test — the argument's own prediction, run in a room that holds money.** If the mechanism is real but saturated, it should pay in a field that does *not* bust early. Same hero (`overbid_noramp` — the one that showed nothing in Run V), same seeds, same pool, only the field changes to `balanced_field`:

| contestant | reg_win | paired Δ (95% CI) | champ Δ | mean_points Δ |
|---|---|---|---|---|
| `control` | 0.6396 | — | — | — |
| `off_pos` | **0.6494** | **+0.0098 [+0.0022, +0.0174]** | **+0.0184 [+0.0076, +0.0292]** | **+13.19 [+5.08, +21.29]** |
| `gap` | 0.6437 | +0.0041 [−0.0028, +0.0109] | **+0.0117 [+0.0019, +0.0215]** | **+9.36 [+1.98, +16.75]** |
| `gap_off` | 0.6389 | −0.0007 [−0.0078, +0.0063] | +0.0067 [−0.0032, +0.0167] | +3.26 [−4.52, +11.03] |

**The hero that gained nothing from poison in the overbidder room gains +0.0098 reg-win (and +0.018 champ, +13 points) in the disciplined one, CI-separated.** The lever is not weak; it is *contingent on there being money to take*.

**Findings (data):**
1. **The forced-negative-transaction argument is confirmed as a mechanism.** Poison nomination pays when opponents still hold cash. Runs U–W's nulls were a property of the *field*, not of the idea.
2. **Total roster value is genuinely conserved** — 2370.9 vs 2371.1 out of ~2371 across all 12 seats (0.01%), even though which 204 of 578 players get drafted is free to vary. So the gain cannot come from the field acquiring *less* talent in aggregate; it comes from **when** the money leaves and therefore what *we* can buy with ours. This is the correct version of Run W's "poison moves dollars, not players."
3. **Run W's "substitutes" reading needs narrowing.** Aggression and poison are substitutes *in a room that busts on its own*, because the busted room hands out the late discount for free. In a room that holds money they are **complements**: `overbid_noramp` is already the aggression optimum and still gains ~+0.010.
4. **The best heuristic is field-dependent.** In `balanced_field` the price-ranked `off_pos` (+0.0098) beats `gap` (+0.0041) — the reverse of Run V's ordering in the overbidder room. Not over-theorized here; recorded as a fact that any adoption decision must respect.

**Practical consequence (the decision-relevant part).** The value of nomination poisoning in Will's league depends entirely on **how disciplined his actual league-mates are**, which is a modeling assumption, not a measurement — bot calibration is a standing open item. If they spend out early like the `overbidder` model assumes, nomination control is worth ~0 and the printed sheet's plan is complete. If they hold money into the middle rounds, it is worth ~+0.01 and the right move is to nominate players we rate below the room. **The eye-test at the actual draft table settles which room he is in**, and it is observable in the first quarter: if lots are clearing well above sheet value and teams are near-broke by pick ~50, it is the overbidder room and nomination doesn't matter.

**Caveats:** quarter diagnostics at 4 seats × 20 seeds (the cash and price curves are order-of-magnitude effects, well clear of that noise floor; the per-quarter hero-surplus splits are not). `balanced_field` is a single alternative field, not a sweep of discipline levels — the natural follow-up is a `pace`/`overbid` sweep to map the lift as a continuous function of how fast the room spends out. One pool, one market, byes off.

**Conclusion (data; no default change):** the user's economic argument stands and is now supported end-to-end — drain → later cash → later prices → hero surplus. Nomination poisoning is a **real edge whose size is set by opponent discipline**, and it is worth ~0 only in the specific busted-market model of Will's room. `overbid_noramp` + default nomination remains the shipped plan for that modeled room; nothing is adopted, and the September call stands. Artifacts: `reports/_gap_nom_probe/will_2026_balancedfield/*.json` (untracked).

**Run Y — 2026-08-13 — POST-FIX re-run on corrected roster bounds ([#143](https://github.com/alhart2015/FantasyFootball/issues/143)) → the hero choice SURVIVES, levels shift ~0.01–0.02, and Run V's "gap beats off_pos for every hero" is RETRACTED as partly a bug artifact** (`will_half12`, ESPN market, ADP nomination `market_adp_jitter=12`, byes OFF, 12 seats × 20 seeds × 300 sims, CRN-paired; branch `fix/auction-flex-position-bounds`; spec `docs/superpowers/specs/2026-08-13-auction-flex-position-bounds-design.md`). Every knob is identical to the runs being superseded — only the engine changed.

**The fix.** `bot_position_bounds` now sets `min` from dedicated starting slots only and gives flex capacity to the **caps** of every flex-eligible position, instead of anchoring FLEX→RB and SUPER_FLEX→QB. Will's league: RB **4/7 → 2/6**, WR **2/4 → 2/6**. Measured effect on drafted rosters — mean WR **3.84 → 4.91**, mean RB **5.80 → 4.23**; the WR cap now binds on 40% of rosters instead of 99% (RB 20%). The roster shape now leans WR, matching what the valuation layer said all along.

**1. Full-field bake-off — the hero is unchanged.** Seat-averaged `reg_win_pct`, Will's `overbidder` room:

| hero | post-fix | pre-fix | rank change |
|---|---|---|---|
| **`overbid_noramp`** | **0.6210** | 0.6389 | **#1 → #1** |
| `overbid` | 0.6146 | 0.6272 | #2 → #2 |
| `studsdepth` | 0.6082 | 0.6168 | #3 → #3 |
| `sr_g0.2_c2` | 0.6007 | 0.6029 | #6 → #4 |
| `static` | 0.5933 | 0.6080 | #5 → #5 |
| `sr_g0.1_c2` | 0.5905 | 0.6144 | #4 → #6 |
| `balanced` | 0.5902 | 0.5971 | #8 → #7 |
| `anchors` | 0.5773 | 0.5975 | #7 → #9 |
| `patient` / `patient_deep` | 0.5162 | 0.5388 | #11 → #12 |
| `vorpshare` | 0.5034 | 0.5223 | last → last |

The top three are identical and `overbid_noramp` keeps a clear margin (+0.006 over #2). Middle-of-table cells shuffle by 1–2 places, all within seed noise. **Every** hero's win% drops slightly, which is expected rather than alarming: `reg_win_pct` is near-zero-sum across seats, and the bots benefit from the corrected caps too, so the hero's edge over a now-better-constructed field narrows.

**2. Nomination probe, Will's `overbidder` room, all four heroes** (paired Δ `reg_win_pct` vs each hero's own control; **bold** = 95% CI excludes 0; pre-fix in parentheses):

| hero | `gap` | `gap_off` | `off_pos` |
|---|---|---|---|
| `overbid_noramp` | **−0.0074** (−0.0044) | **−0.0219** (−0.0141) | −0.0008 (−0.0097) |
| `static` | **+0.0098** (+0.0145) | −0.0049 (+0.0013) | −0.0046 (−0.0077) |
| `balanced` | +0.0024 (+0.0118) | +0.0006 (+0.0106) | +0.0016 (+0.0103) |
| `patient_deep` | **+0.0062** (+0.0057) | +0.0054 (+0.0064) | +0.0017 (+0.0054) |

**3. Nomination probe, disciplined `balanced_field` room, `overbid_noramp`:** `off_pos` **+0.0114 [+0.0030, +0.0199]**, `gap_off` **+0.0093 [+0.0012, +0.0174]**, `gap` +0.0051 [−0.0029, +0.0131]. All three are now positive (pre-fix `gap_off` was −0.0007), and the two off-position variants are CI-separated.

**Findings (data):**
1. **The shipped plan survives the fix.** `overbid_noramp` is still the best hero for Will's modeled room, and still by a margin. No default changes.
2. **Run V's headline is retracted.** "`gap` beats the price-ranked `off_pos` for **every** hero, without exception" was **partly a bug artifact**: with WR "filled" at 2 and RB not until 4, `off_pos` was forced to dump WRs specifically, and WRs were the position the broken cap made worthless to hold. Post-fix `gap` wins for **3 of 4** heroes, not 4 of 4 — `overbid_noramp` now prefers `off_pos` (−0.0008) to `gap` (−0.0074). The *general* claim (the disagreement signal is the stronger one) still holds; the "without exception" does not.
3. **Run V's effect sizes were inflated.** `balanced`'s gain collapses from +0.0118 (CI-separated) to +0.0024 (not distinguishable from zero), so the count of heroes that measurably gain drops from **3 of 4 to 2 of 4** (`static`, `patient_deep`). The direction of the finding survives; its strength does not.
4. **Runs U/W/X survive qualitatively.** `overbid_noramp` still gains nothing from poison in the busted room (and `gap`/`gap_off` are now clearly *harmful* there, sharpening Run U); poison still pays in a room that holds money, now more strongly (Run X's +0.0098 → +0.0114). The busted-vs-disciplined contrast — the load-bearing practical conclusion — is unaffected.

**Caveats:** the disciplined-room probe was re-run for `overbid_noramp` only, not all four heroes; the `k`-sweep of Run W was **not** re-run, so its "substitutes" claim rests on pre-fix data and should be treated as unconfirmed on the fixed engine. Everything else as before: one pool, one market, single ADP jitter, byes off.

**Conclusion (data; no default change):** the bug was real and load-bearing on roster *shape*, but it did not change **which strategy to use**. Will's cheat sheet and its MAX BID numbers were never affected (valuation was always correct). `overbid_noramp` + default nomination stands, and the practical nomination advice is unchanged: poison is worth ~0 to slightly negative in a room that busts early, ~+0.01 in a room that holds money. Artifacts: `reports/_will_bakeoff/postfix_2026/`, `reports/_gap_nom_probe/postfix_*/` (untracked). Live-draft call still September.

**Run Z — 2026-08-13 — AGGRESSIVE/CONSERVATIVE field-mix sweep → `overbid_noramp` is the best or statistically tied-for-best across the whole range from 11/0 down to 5/6, and only clearly loses at 3/8** (`will_half12`, ESPN market, ADP nomination `market_adp_jitter=12`, byes OFF, `overbidder` field at `overbid=0.2/pace=4.5/opening/jitter=0.35`, full 15-hero field × 12 seats × 20 seeds × 300 sims per mix, CRN-paired; post-#143 engine; branch `exp/auction-field-mix-sweep`; design note `docs/superpowers/specs/2026-08-13-auction-field-mix-sweep-design.md`, written retroactively -- the branch did not follow the spec-first rule and that is recorded there). Motivated by the obvious unswept axis: every prior run modelled Will's room as **9 aggressive / 2 conservative** of 11 bots, and that ratio was a hard-coded assumption (`_PATIENT_EVERY`), never a measurement. `build_field` gains `n_patient` (exactly that many hoarder seats, spread evenly) so the mix can be swept.

Seat-averaged `reg_win_pct` by mix (aggressive/conservative of 11 bots), **all 15 contestants**, ordered by the 9/2 column:

| hero | 11/0 | **9/2** | 8/3 | 6/5 | 5/6 | 3/8 | 0/11 |
|---|---|---|---|---|---|---|---|
| **`overbid_noramp`** | 0.6080 | **0.6167** | **0.6107** | 0.6003 | 0.5992 | 0.5842 | **0.6744** |
| `overbid` | **0.6143** | 0.6108 | 0.6029 | 0.5927 | 0.5973 | 0.5864 | 0.6634 |
| `studsdepth` | 0.6007 | 0.6054 | 0.5987 | 0.5897 | 0.5774 | 0.5848 | 0.6634 |
| `static` | 0.6004 | 0.5965 | 0.5871 | 0.6002 | **0.6089** | 0.6091 | 0.6695 |
| `balanced` | 0.6010 | 0.5875 | 0.5803 | 0.5884 | 0.5524 | 0.5383 | 0.5391 |
| `balanced_flat` | 0.6010 | 0.5875 | 0.5803 | 0.5884 | 0.5524 | 0.5383 | 0.5391 |
| `anchors` | 0.6000 | 0.5846 | 0.5930 | **0.6045** | 0.6075 | 0.6068 | 0.6732 |
| `sr_g0.2_c2` | 0.5746 | 0.5833 | 0.5726 | 0.5711 | 0.5733 | 0.5628 | 0.5609 |
| `sr_g0.1_c2` | 0.5865 | 0.5794 | 0.5842 | 0.5743 | 0.5691 | 0.5651 | 0.5485 |
| `sr_g0.3_c2` | 0.5811 | 0.5781 | 0.5779 | 0.5872 | 0.5857 | 0.5734 | 0.5556 |
| `inflation` | 0.5421 | 0.5473 | 0.5300 | 0.5851 | 0.5846 | **0.6349** | 0.6722 |
| `patient` | 0.5196 | 0.5238 | 0.5088 | 0.4997 | 0.5017 | 0.4685 | 0.4754 |
| `patient_deep` | 0.5196 | 0.5238 | 0.5088 | 0.4997 | 0.5017 | 0.4685 | 0.4754 |
| `vorpshare` | 0.5111 | 0.5113 | 0.5127 | 0.5263 | 0.5354 | 0.5186 | 0.5012 |
| `marginal` | 0.5105 | 0.5076 | 0.5088 | 0.5216 | 0.5210 | 0.5016 | 0.4789 |

> **`balanced`/`balanced_flat` and `patient`/`patient_deep` post bit-identical figures — and they
> are NOT the same policy.** `registry.py` builds them with different constructor arguments
> (`non_increasing_cap=True`, `scrub_frac=0.0`) and the bid layer reads both. Instrumenting
> `BalancedValueBid.max_bid` over six full drafts of the **9/2 cell**: 354 bid calls, **2** where the two configs
> return different bids, and **6 of 6 identical hero rosters** — the clamp fires but never changes
> an outcome in those drafts, because the hero's bid is fair-value-limited rather than cap-limited on
> the players it actually wins. **That instrumentation covers the `balanced` pair only**; the
> `patient`/`scrub_frac` half is an unverified hypothesis (see #146), and neither explains why the
> agreement is exact to full float precision across every cell rather than merely close — a harness
> effect is not ruled out. Reproduce with `scripts/_diag_identical_contestants.py`. Run L measured them as clearly different (0.554/0.495 vs 0.522/0.487), but that is **not a clean
> contradiction**: `BalancedValueBid.premium` was retuned 1.0 → 0.0 between the two runs (Run N),
> and at premium 1.0 the bid is cap-bound where at 0.0 it is fair-value-bound — which is exactly the
> mechanism above. Run L also used a different league, nomination model and a pre-#143 engine. Filed as
> [#146](https://github.com/alhart2015/FantasyFootball/issues/146); **treat this field as 13
> distinct policies, not 15**, and do not read the duplicate rows as two independent measurements.
> (An earlier draft of this note asserted they were "identical policies under two registry names",
> which is false and closed the question instead of opening it.)

(An earlier draft of the table printed 11 of the 15 and described itself as the full field — the omitted `sr_g0.1_c2` outranks the printed `balanced` at 8/3, so the abridgement was not a top-N.)

Point estimates alone would read as four different winners. The CRN-paired test says otherwise — best challenger to `overbid_noramp` per mix (positive = beats it; `*` = 95% CI excludes 0):

| mix | best challenger | paired Δ | 95% CI | verdict |
|---|---|---|---|---|
| 11/0 | `overbid` | +0.0063 | [−0.0002, +0.0127] | **tied** (CI touches 0) |
| **9/2** | `overbid` | −0.0059 | [−0.0123, +0.0004] | **tied** |
| 8/3 | `overbid` | −0.0079 | [−0.0156, −0.0001]\*† | **tied** after correction (see †) |
| 6/5 | `anchors` | +0.0042 | [−0.0039, +0.0124] | **tied** |
| 5/6 | `static` | +0.0097 | [−0.0007, +0.0200] | **tied** (lower bound −0.0007) |
| **3/8** | **`inflation`** | **+0.0507** | **[+0.0401, +0.0614]\*** | **`overbid_noramp` LOSES** |
| 0/11 | `anchors` | −0.0012 | [−0.0096, +0.0073] | **tied** |

† **Multiplicity caveat.** Each row selects the best of 14 challenger *names* — only **12 of which are distinct policies**, see the note under the table — and then applies an uncorrected 95% CI, so the CIs are anti-conservative for exactly the comparison being made. For the
**tied** rows this cuts the safe way — a selected maximum that still fails to separate is stronger
evidence of a tie, not weaker. It bites only on 8/3, whose bound is **−0.0001**: a two-sided Bonferroni threshold is z≈2.91 at 14 challenger names (2.87 at the 12 distinct ones) and that row is z≈−1.99, so it does **not** survive correction. Its verdict cell therefore reads *tied* — as do 11/0, 9/2, 6/5, 5/6 and 0/11, whose intervals all include zero outright. **No row demonstrates separation in the hero's favour**, which is the finding. (The 3/8 row does separate, decisively, in the hero's *disfavour* — that is the other half of the finding.) Any correction for selection widens it across zero. The 3/8 loss (+0.0507, bound +0.0401) is far outside any plausible correction and is unaffected.

**Findings (data):**
1. **The answer to "when does it stop being best" is: 3 aggressive / 8 conservative.** From 11/0 through 5/6 **no** contestant — not merely the best challenger — beats `overbid_noramp` with a CI that excludes zero; it is the leader or statistically tied for it across that whole range. At **3/8** it is beaten decisively by `inflation` (**+0.051**), and also by `static` (+0.025\*) and `anchors` (+0.023\*). The crossover sits between **5/6 and 3/8**, i.e. once roughly three-quarters of the room is conservative.
2. **This is a strong robustness result for Will's plan.** The modelled room is 9/2. The hero survives being wrong about the mix by a wide margin — all the way to 5 aggressive of 11 — before the recommendation would even be in question, and it takes 3/8 before it is actually wrong.
3. **The response is non-monotone, and the worst cell is the middle-conservative one (3/8), not the all-conservative one (0/11).** At 3/8 `overbid_noramp` posts its lowest figure (0.5842) and at 0/11 its highest (0.6744). A plausible reading — *offered as a hypothesis, not a measurement*: 3/8 is the worst of both worlds, with enough aggressive bots to bid the studs up *and* a cash-rich rump that can still outbid you late; at 0/11 nobody contests the studs at all, so a stud-buyer simply takes them. Not verified here.
4. **`inflation` is the specialist to remember.** It is near-worst in aggressive rooms (0.5421 at 11/0, 0.5473 at 9/2) and the clear winner at 3/8 (0.6349). Nothing is adopted on that basis, but if the real room turns out to be cash-hoarding, the shipped hero is the wrong one and `inflation` is where to look.
5. **Moving the hoarders is worth ~0.004 even at a fixed 9/2 mix.** The 9/2 cell here reads 0.6167 against the post-fix bake-off's 0.6210 — same *count*, but `--n-patient 2` puts the hoarders at seats [2, 8] where the historical `_PATIENT_EVERY` rule put them at [4, 9]. **This is placement *and* pace reassignment, not placement alone:** `_spread_paces` hands its jittered caps to the non-hoarder seats in seat order, so moving a hoarder shifts which aggressive seat draws which cap for every seat after it. The two configurations differ in both respects at once and this run does not separate them. The operational conclusion is unaffected — they are not interchangeable, and the gap is the same order as several effects chased earlier in this log. **The Run-Y bake-off remains the reference for the 9/2 room**; this sweep is internally consistent across its own seven cells, which is what the comparison needs.

**Caveats:** one pool, one market, one seat-count, byes off; `overbid`/`pace`/`jitter` of the aggressive archetype held fixed, so this sweeps *how many* are aggressive and not *how* aggressive. The conservative archetype is the stock `PatientValueBot` throughout. **The grid is coarse and uneven:** the swept `n_patient` values are 0, 2, 3, 5, 6, 8, 11 — steps of 2, 1, 2, 1, 2 and 3 seats, i.e. 9 to 27 percentage points, not a uniform 9. **`n_patient=7` (4/7) was never run, and it sits inside the claimed 5/6→3/8 crossover bracket**, so the crossover is bracketed by two cells **eighteen** points apart (two seats) and is not located more precisely than that — nine points is the distance from each bound to the un-run midpoint, not the bracket width. Filling 4/7 (and 9/2-with-historical-placement) is the cheap follow-up if the crossover location ever matters.

**Conclusion (data; no default change):** `overbid_noramp` is robust to the field-mix assumption across the plausible range and only fails in a room that is ~75% cash-hoarders — which is not the room Will is modelled to be in, and is observable at the table (Run X: if lots clear above sheet value and teams are near-broke by ~pick 50, it is an aggressive room). Artifacts: `reports/_field_mix/p{0,2,3,5,6,8,11}/*.json` (untracked). Live-draft call still September.

## Planned experiments / axes to sweep

- **Seat sweep (NEW, Run K priority)** — sweep all 12 seats to quantify the ~0.10 seat effect and test whether seat 1 is structurally bad vs seat-6 easy-schedule luck; the goal metric (`reg_win_pct` in a random-seat league) should be the seat-averaged value.
- **Bid-model bake-off** (the core): `static` vs `inflation` vs `marginal`, all three metrics, at a fixed
  preset + seat + `price_jitter`.
- **`price_jitter` sweep** — how the ranking moves as the bot market gets noisier (analog of the snake
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
- **Multi-year averaging (user, 2026-06-18)** — results swing hard year-to-year (Runs A–G); re-run the
  bake-off across every season we have projection data for and average the per-model metrics, so the
  ranking reflects a multi-year mean rather than one noisy 2026 draw. The single biggest reliability lever.
- **Learn the bid policy (RL) (user, 2026-06-18)** — hand-authored bid models all cluster at/below the
  field's fair share; a reinforcement-learning agent that learns a bidding policy against the bot market
  (then self-play) may find an edge the fixed heuristics miss. Larger new slice; design separately.

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

`compare` races the eight contestants (`static` / `inflation` / `marginal` / `anchors` / `overbid` / `vorpshare` / `patient` / `studsdepth`) against a shared seeded bot market and prints:
- each model's per-metric mean + 95% bootstrap CI table
- paired per-seed differences with CIs for every model pair

No winner is declared. Paste the per-model table into the experiment log above with the exact flags used.

**Swap the `--vorp-table` / `--league-config` pair together** — they must agree on scoring ruleset and
team count (the budget, roster shape, and replacement level are all derived from `LeagueConfig`).
Available preset pairs: `data/vorp_2026/{half,ppr,std}_{10,12,16}team.parquet` with
`configs/league_espn_{half_16team,ppr_12team,half_10team}.json` (or any custom `LeagueConfig` JSON).
