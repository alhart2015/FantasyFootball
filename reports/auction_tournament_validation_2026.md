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
