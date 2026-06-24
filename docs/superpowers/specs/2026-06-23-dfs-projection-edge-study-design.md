# DFS Engine — Layer 1: Projection Edge Study (design)

- **Date:** 2026-06-23
- **Status:** Approved (brainstorm); spec-review iterations 1–3 applied
- **Branch:** `feat/dfs-projection-edge-study`
- **Sub-project:** DFS Engine (`CLAUDE.md` sub-project #4), first slice
- **Supersedes / closes:** TODO #39 (fair weekly start/sit benchmark), re-pointed at Sleeper + DraftKings scoring

## 1. Context & motivation

We want to know whether some combination of our projection models can **consistently outperform the market in daily fantasy (DFS)**. DFS is a sharp, *negative-sum* market: the operator takes a rake (~10%), so the player pool collectively gets back < $1 per $1 wagered. To profit you must beat enough of the field to clear both the median **and** the rake.

The critical, often-missed point: **our projections being "good" in isolation is not enough.** The field already prices in ESPN / Sleeper / industry consensus. If our blend merely *reproduces* consensus, our edge is zero *by definition*. The entire edge lives in **where our model disagrees with the market and is right.**

Building a full DFS engine (salary-cap optimizer, contest simulation, correlation/stacking, ownership leverage) is a large, multi-slice effort. It is only worth doing if an underlying projection edge exists. **This spec is Layer 1: the cheapest experiment that can kill the whole idea** — prove (or disprove) a projection edge over the market *before* building any DFS-specific machinery.

### Decision gate (the whole point of this slice)

- **Edge exists →** graduate to Layer 2 (DK/FD scoring-ruleset hardening, salary-cap ILP optimizer, contest/ROI backtest, possibly against a sharper market proxy).
- **No edge →** **stop.** Our model adds nothing over the free market; a DFS engine is not worth building. A cheap, valuable *negative* result.
- **Inconclusive (insufficient data/power) →** a distinct third outcome (§7.4), so we never mistake "too few cells to tell" for "no edge."

## 2. Goal

Determine, with statistical confidence and a single pre-registered primary test, whether our home-grown weekly model — **alone or blended with Sleeper** — produces weekly fantasy-point projections that **beat Sleeper's own weekly projections** under DraftKings base scoring, on historical NFL weeks. Per-position results are reported as **exploratory** (§7.3).

The deliverable is a **repeatable benchmark harness** plus a **committed verdict report** carrying an explicit ADOPT / STOP / INCONCLUSIVE recommendation.

## 3. Scope & non-goals

### In scope
- Sleeper **weekly** projection ingest (historical, retrospective — endpoint verified live, §4.3).
- A DraftKings **base** scoring ruleset (bonuses excluded from the comparison; see §6.2).
- A path that emits our home-grown model's **weekly projections** for arbitrary past `(season, week)` cells, walk-forward (§5.3).
- A small set of **blends** (home-grown + Sleeper) to test "some combination."
- A **metric harness** comparing each projection source against actuals (§7).
- A **verdict report** under `reports/`.
- **Skill positions only:** QB, RB, WR, TE.

### Explicit non-goals (deferred — do NOT build here)
- Salary-cap **optimizer** (ILP) — Layer 2.
- **Contest simulation / ROI** modeling — Layer 2.
- **Correlation / stacking** (QB+WR) — GPP-only, Layer 3 (TODO #1, option D).
- **Ownership** data / leverage — Layer 3.
- **ESPN historical** ingest — ESPN keeps only ~1 prior season of projections, deletes history, and gates historical access behind an authenticated `espn_s2` cookie (verified 2026-06-23). Sleeper is the retrospective proxy here; an ESPN **forward-collection** path is a separate future slice. (A `draft/backtest/espn_weekly.py` precedent exists and may inform it; not consumed here.)
- **DST / K** projections, and DST/K scoring in the DK ruleset (the `draftkings()` preset is **skill-scoring-only** in v1; §5.2, L1).
- **Modeling `E[bonus]` for projections** — deferred (Layer 2). v1 excludes the +3 yardage bonuses from the *projection* comparison as a conservative simplification (§6.2). It **does** add a small **deterministic, actuals-only** DK bonus helper (a fixed `≥300/≥100/≥100 → +3` add-on, not a `Ruleset` field) used solely for the §6.2 sensitivity check — that helper is in scope (§5.2).

## 4. Background facts (verified 2026-06-23)

### 4.1 DraftKings NFL Classic
- **Salary cap** $50,000; **9 roster slots**: QB, 2×RB, 3×WR, TE, FLEX (RB/WR/TE), DST. (Roster shape is irrelevant to this slice — it is a *per-player accuracy* study, not a lineup study — recorded for Layer 2.)
- **Scoring** (basis for `Ruleset.draftkings()`, skill positions): passing 0.04/yd, 4/pass TD, −1/INT; rushing & receiving 0.1/yd, 6/TD; **1.0 PPR**; −1/fumble lost; +3 bonus at 300+ pass yds; +3 at 100+ rush yds; +3 at 100+ rec yds. **Note vs. existing code:** `Ruleset.fumble_lost_pts` defaults to **−2.0** (ESPN); DK is **−1.0** — the DK preset must override it.

### 4.2 Contest economics (Layer 2 context, recorded not used here)
- **Cash games** (50/50, double-up, H2H): beat ~top-44% after rake (profitable ≈ 56%+ win rate). Projection accuracy pays off most directly.
- **GPP tournaments**: top-heavy payouts; goal is a top-~1% finish; needs correlation + ownership leverage.

### 4.3 Data feasibility — verified live (this is why Sleeper, not ESPN)
**Sleeper weekly endpoint — probed live 2026-06-23, confirmed usable:**
- `GET https://api.sleeper.com/projections/nfl/<season>/<week>?season_type=regular` (host `.com`; `.app` returns identical data — **use `.com` to match the existing ingest at `external_projections.py:54`**).
- Returns a list of ~9,400 rows; each carries a `stats` dict with a **full projected stat line**. Verified keys include: `pass_yd`, `pass_td`, `pass_int`, `pass_2pt`, `rush_att`, `rush_yd`, `rush_td` (when applicable), `rec`, `rec_yd`, `rec_td`, `fum_lost`, plus pre-scored `pts_ppr`/`pts_half_ppr`/`pts_std`. **There is NO DraftKings points column** — confirming we must re-score from the stat line under our own DK ruleset (§5.2).
- **History depth:** stat-line rows returned for **2019, 2020, 2021, 2022, 2023, 2024** (probed wk5 each). So ≥3 evaluation seasons is comfortably available; we will use the most recent N with both Sleeper data and home-grown coverage (default: 2021–2024, era-aligned to the 18-week regular season; 2019–2020 available as a robustness extension).
- **Distinct from the season endpoint.** The previously-verified finding (`2026-06-08-external-projection-benchmark-design.md`) that "Sleeper season projections are ADP-only, no stat line" applies to the **season** endpoint. The **weekly** endpoint *does* carry a stat line (verified above). The ingest must not assume the season-path's ADP-only shape.

**ESPN — not reliably retrievable** historically (§3 non-goals). **Our home-grown model** runs retrospectively on any past season (we have historical `weekly_stats` + features), subject to the rookie/cold-start coverage limits in §5.3.

**Honest caveat (recorded in the report):** Sleeper-alone is a *softer* proxy than the true DFS field (which aggregates many sharp sources). Beating Sleeper is **necessary but not sufficient** for real DFS profitability. The gating logic still holds: *if we cannot beat Sleeper retrospectively, we certainly cannot beat the field* — a valid, cheap kill-test. A sharper proxy is a Layer-2 concern.

## 5. Architecture & components

Six units, each with one clear purpose. Reuse existing machinery where noted; build new only where marked **NEW**.

### 5.1 Sleeper weekly ingest — **NEW** (extends the existing ingest pattern)
- Pull `api.sleeper.com/projections/nfl/<season>/<week>?season_type=regular` for the target `(season, week)` cells.
- Map Sleeper `stats` keys → our canonical stat line: `pass_yd→passing_yards`, `pass_td→passing_tds`, `pass_int→interceptions`, `rush_yd→rushing_yards`, `rush_td→rushing_tds`, `rec→receptions`, `rec_yd→receiving_yards`, `rec_td→receiving_tds`, `fum_lost→fumbles_lost` (confirm exact target field names against `Stat` enum / `ExternalProjectionSchema` at plan time). Ignore Sleeper's `pts_*` (we re-score).
- **ID join (known-weak seam — budgeted, M5):** map Sleeper `player_id → GsisId` via `id_map`. The raw `id_map.sleeper_id` is stored float-stringified like `'4374302.0'`, so a naive join yields **zero matches** (TODO #38). The working `.0`-stripping logic currently lives in `scripts/benchmark_projections.py:_normalize_join_id` (93-106) — but **`scripts/` is not an importable package** (`pyproject.toml` `packages.find where=["src"]`, no `scripts/__init__.py`), so `src/projections/ingest/` **cannot import from it**. **Lift `_normalize_join_id` into `src/`** (e.g. `ingest/identity.py` alongside `placeholder_name_key`) and have **both** the new ingest **and** the existing script import it from there (CLAUDE.md "reuse before writing," correct layering direction). Do **not** reuse `ingest/external_projections._attach_gsis_id` — it only casts dtype and does *not* strip the `.0` suffix, so it would reintroduce the zero-match bug. Rookies with no `id_map` entry get placeholder gsis (`placeholder_name_key`, `ingest/identity.py:18` — note: no leading underscore). **Placeholder-gsis rows cannot join real-gsis `weekly_stats` actuals and are therefore excluded from the comparison universe; they must be counted and reported as a coverage loss (§7.1), not silently dropped.**
- **Position is for filtering only, not bucketing.** Sleeper's `player.position` is primary-position only; the per-position metric (§7) buckets every cell by the **actuals' canonical position from `weekly_stats`** (the single source of truth for a cell's position), never by each source's self-reported position (M3).
- Follow the canonical ingest template (`ingest/external_projections.py` + `weekly_stats.py`); persist via `store.write_partition` only; cast numeric columns to uniform `Float64` to avoid the all-NA `pd.concat` FutureWarning (cf. the prior blend fix).
- **New schema** `ExternalProjectionWeeklySchema` (§8): `ExternalProjectionSchema` shape + a `week` column, `source = SLEEPER`. Validate-with-reassignment at the boundary.

### 5.2 DraftKings scoring ruleset — **NEW** (small extension to `scoring/`)
- Add `Ruleset.draftkings()` preset (the §4.1 skill values; **override `fumble_lost_pts` to −1.0**). The existing `Ruleset` already carries passing-yds/TD/reception/INT fields (`schemas.py:255-274`), so this is value-only — **no `Ruleset` shape change**.
- **Bonuses are NOT `Ruleset` fields.** `Ruleset` has no bonus/threshold fields and the scoring layer has no bonus logic. The +3 DK yardage bonuses live in a **separate, standalone deterministic helper** (`dk_actuals_bonus(stat_line) -> float`, a fixed `≥300 pass / ≥100 rush / ≥100 rec → +3` rule), applied **only when scoring actuals for the §6.2 sensitivity check**. The base `draftkings()` preset stays value-only; the projection comparison path runs **base scoring** (no bonuses) symmetrically across all sources and actuals. (This replaces the earlier, incorrect "the preset records bonus parameters" claim.)
- **Allowlist update (C3):** adding a `DRAFTKINGS` ruleset name requires extending `_RULESET_NAME_VALUES` (`schemas.py:302`) and the three `isin` sites that pin it — `ConsensusProjectionSchema` (`schemas.py:899`), `PreseasonProjectionSchema` (`1244`), `PreseasonBacktestSchema` (`1309`) — per CLAUDE.md convention #10. (`ProjectionWeeklySchema.ruleset` is a free `Series[str]`, no `isin`, so it needs no edit but is noted.) **Three further `DRAFTKINGS`-rejecting sites are deliberately NOT touched, justified here:** the hard-coded test literal `tests/test_schemas/test_dataframe_schemas.py:1001` (its golden row is `ESPN_PPR`, unaffected); and the closed preset registries `draft/league_config.py:16` (`_RULESET_PRESETS`) and `draft/assistant/presets.py:20` (`_RULESETS`) — the DFS path constructs `Ruleset.draftkings()` **directly**, bypassing those config-string lookups, so they need no DK entry. The plan must re-confirm no DFS code path routes a `Ruleset` through them.
- **Scoring path.** Sleeper gives a stat line → score directly via `scoring.score` / `expected_points` under `Ruleset.draftkings()`. Our home-grown model gives a **points mean already scored under its fit-time ruleset plus an opaque `params` blob** — see §5.3 for the required decode; do not assume our model's `mean` is already DK-base points.
- **v1 is skill-scoring-only** (no DST/K/defensive scoring rows); the preset is not a complete Layer-2 DK ruleset (L1).

### 5.3 Home-grown weekly projections — **NEW thin emitter over REUSED harness internals** (C2)
The existing model backtest (`src/projections/backtest/harness.py` — NOT the same-named `draft/backtest/harness.py`) returns **metric rows** (`BacktestRun.metrics`, `harness.py:81`); its per-cell projections live only in gitignored diagnostic `per_row_results`, and its `mean` column is **already scored under the ruleset passed at predict time** (`models/baseline.py:705,750`). It is therefore *not* a drop-in source of DK-base weekly projections. We will:
- Add a **thin projection-emitter** that reuses the harness's per-`(position, year)` fit→predict loop (train window `range(train_start, year)`, `harness.py:241`) but returns a `ProjectionWeeklySchema`-shaped frame for the requested cells instead of metrics. Reuse the harness's existing distribution-decode helper (`_per_stat_means_from_predictions`, `harness.py:107`) — promote it from private if needed (note: it takes a DataFrame and returns stat-string-named columns, not a `dict[Stat, float]`) — to recover **per-stat means from `params`**, then score those stat-line means under `Ruleset.draftkings()` (base) via the scoring layer. This keeps scoring in one place and yields a DK-base points projection comparable to Sleeper's.
- **Leakage proof is at the FEATURE level, not the harness loop, and has TWO distinct surfaces (H-1).** The harness fits once per `(position, year)` and predicts all weeks of `year` from one feature read, so "no leakage for week W" is a property of the **feature builders**, not the loop. But the builders use **two different** trailing mechanisms, and only one is currently guarded:
  1. **`prior_mask`-filtered stat inputs** (`features/_shared.py:20`, strict `<`): weekly_stats/snap_counts/ngs/depth_charts/schedules. Already leakage-tested (`test_{qb,rb,te,wr}_leakage.py` inject future rows and assert byte-equality). The plan confirms these still hold for the weekly grain.
  2. **Trajectory / vegas-team-context / weather features** are handed the **full, unfiltered** frames (e.g. `features/wr.py:~255-294`) and rely on internal `.shift(1)` / expanding semantics that `prior_mask` does **not** cover and the existing leakage tests do **not** exercise. This is an **unguarded** leakage surface in exactly the recent-form features. The plan must add a **separate, named** verification + new leakage test that injects future rows into the trajectory/vegas/weather inputs and asserts the week-W projection is unchanged. Do **not** fold this into the "trailing windows" claim — that would silently skip it.
- **Cold-start / rookie policy (M2):** our model cannot project rookies (no prior NFL data; cf. `benchmark_projections.py:233`) and has thin features in early-season weeks. State the policy explicitly: rookies and any cell our model cannot project are **excluded from the universe but counted**, and the report breaks coverage down **per-week-bucket** (e.g. wk 1–3 / 4–13 / 14–18) and by rookie/veteran, so a shrunken or late-season-only universe is visible, never silent.

### 5.4 Weekly actuals scorer — **REUSE with a deliberate fix** (`scoring/actuals.py`; do NOT reuse `build_weekly_actuals` as-is) (H1)
- We need **full-PPR DK-base** actual weekly points across the **era-aware** regular season. The existing `draft/backtest/weekly_actuals.py:build_weekly_actuals` **already takes a `ruleset` and scores via `scoring.score`** (its "half-PPR" is only a stale *docstring*; behavior is ruleset-driven). The genuine gaps are narrow: it **hard-caps `_MAX_WEEK = 17`** and `WeeklyActualSchema.week` is bounded `le=17` (`schemas.py:1347`) — so it would **silently drop Week 18** for 2021+ seasons (exactly the recent seasons we want).
- **Reuse the existing era helper.** An era-aware regular-season cutoff already exists: `draft/assistant/availability.py:24 _last_regular_week(season)` (17 pre-2021, 18 after) — **promote/reuse it** rather than inventing a new cutoff (CLAUDE.md "reuse before writing"; this is also the logic TODO #41 wants centralized).
- **Decision — prefer option (b) for blast radius.** (a) Extending `build_weekly_actuals` in place means widening `WeeklyActualSchema.week` (`le=17 → le=22` with a REG filter) **and consciously rewriting the test that pins the current behavior** (`tests/test_draft/test_backtest/test_weekly_actuals.py:42 test_excludes_week_18`) — a backtest-coupled change. (b) Adding a **sibling weekly-actuals scorer in `scoring/`** (ruleset-parameterized, era-aware via `_last_regular_week`, its own schema bound) avoids touching the backtest-pinned schema/test entirely. **Lean (b);** the plan confirms. Either way all fantasy-point math stays in the scoring layer. (Note: a second `le=17` bound exists at `WeeklyProjectionSchema.week` `schemas.py:1333` — the DFS path does not write through it, but it is flagged so no reader assumes `:1347` is the only week-17 cap.)
- The week universe (which weeks count) is stated explicitly in the plan and matched across all three sources (ours, Sleeper, actuals) so no source asymmetrically lacks a week (H1).

### 5.5 Blends — **REUSE** (`consensus/blend.py` pattern)
- Combine home-grown + Sleeper at a small fixed set of weights: at minimum home-grown-only, Sleeper-only (the baseline), and 50/50; optionally one or two more. **No learned weights in v1** (YAGNI; a tuned blend is a follow-up only if a fixed blend shows promise).
- **Blend space is pinned to stat-line space** (consistent with `build_consensus`, `blend.py:96-102`): blend the per-stat means (Sleeper's stat line and our decoded stat-line means from §5.3), then score once under `Ruleset.draftkings()`. This requires our model's decoded stat-line means (§5.3), not just its points mean (L3).

### 5.6 Metric harness & report — **NEW** (reuses `benchmark_projections.py` join + a clustered bootstrap)
- Join all projection sources + actuals on `(gsis_id, season, week)`, restrict to the comparable universe (§7.1), compute the metrics (§7.2) per the pre-registered gate (§7.3), and emit the report.
- Reuse the join/scoring scaffolding from `scripts/benchmark_projections.py` (`build_benchmark_frame`, lines 123-158 — note its join is `gsis_id`-level, so it is *adapted* to the `(gsis_id, season, week)` grain, not used verbatim). **The paired bootstrap must be clustered (§7.2.3); `draft/assistant/_compare.py:bootstrap_mean` is a flat i.i.d. resample and must be extended/replaced, not reused as-is.** `top_n_hit_rate` (`benchmark_projections.py:189`) ranks by **season totals** within position; the weekly version is **not a one-line `groupby` tweak** — it must drop the season-total aggregation (`actual_season_points`, lines 60-68) entirely and rank within `(position, week)` vs. actual weekly top-N (L2).

## 6. Key design decisions

### 6.1 Sleeper-alone as the market proxy (v1)
Accepted as a softer-but-valid kill-test (§4.3). Recorded as a known limitation in the report; a multi-source / sharper proxy is Layer 2.

### 6.2 Base DK scoring in the comparison; bonuses excluded — a *conservative* choice, not "unbiased" (M1)
The +3 yardage bonuses are **probabilistic** for a projection: `E[bonus] = 3 · P(yards ≥ threshold)`, which a *point* projection (all Sleeper offers) cannot express. Including them would either handicap Sleeper or demand a bonus model we are deferring. **Decision:** v1 scores **projections and actuals both under base DK scoring (no bonuses)**.

**Honest characterization of the bias (corrected from "unbiased"):** dropping bonuses is exactly neutral only for the symmetric **error** metric. For the **head-to-head** and **ranking** metrics it is **conservative, not neutral** — bonuses fire on right-tail (ceiling) games, so removing them compresses the top of the actual-points distribution where ranking is most contested. If our model is the ceiling-better source (plausible, since it has distributions), base scoring **understates** our edge. We therefore treat a positive base-scoring result as a *lower bound* on edge, and add a **sensitivity check**: re-run the headline metric scoring **actuals with full bonuses** — using the deterministic `dk_actuals_bonus` helper (§5.2), which is exact because actual yards are known (no `E[bonus]` modeling needed) — while keeping both **projections** on base scoring, and report whether the verdict flips. Both projections are still compared to the *same* bonus-inclusive actual, so the head-to-head stays fair. Estimating `E[bonus]` from our model's distribution (for projections) remains a candidate Layer-2 edge.

### 6.3 Skill positions only (QB/RB/WR/TE)
Matches the draft sub-project's skill-only choice. DST/K deferred. Per-position bucketing uses the actuals' canonical `weekly_stats` position (M3, §5.1).

### 6.4 Walk-forward, no leakage — proven at the feature level (C2, §5.3)
Week W's home-grown projection uses only data strictly before week W. This is a property of the feature builders' trailing windows, verified and tested in the plan — **not** assumed from the harness loop.

### 6.5 Comparable universe conditioned on ACTUALS, not on projections (H4)
Restrict to cells above a usage floor defined on **actual** usage (e.g. actual offensive snaps or actual touches/targets from `weekly_stats`), **never** on either source's projected points — conditioning the sample on a predictor is endogenous selection that tilts the head-to-head toward the other source. Additionally, cells only **one** source projects (inclusion disagreement — potentially where real edge lives) are **not silently dropped**: they are excluded from the paired head-to-head but reported as a separate **inclusion-disagreement diagnostic** (§7.1).

## 7. The metric (how we declare edge)

### 7.1 Comparable universe
- Cells where **both** Sleeper and our model produce a projection (paired comparison), bucketed by the actuals' `weekly_stats` position, QB/RB/WR/TE only.
- Above a usage floor defined on **actuals** (§6.5). Proposed default: actual offensive snap share ≥ a fixed threshold OR actual touches+targets ≥ a fixed count — the **exact threshold is pre-registered in the plan from a prior-year distribution, before looking at the metric outcome** (§7.2.1).
- **Evaluation seasons:** default 2021–2024 (18-week era; both Sleeper and home-grown coverage); 2019–2020 as a robustness extension.
- **Coverage accounting (M2, M5):** the report states, per season and per-week-bucket, how many cells were dropped for (a) placeholder-gsis (rookie) Sleeper rows that cannot join actuals, (b) cells our model cannot project (rookies/cold-start), (c) the usage floor — so the universe's shape is explicit.
- **Inclusion-disagreement diagnostic (H4):** count and characterize cells only one source projects (above the actual-usage floor); reported separately, not in the paired test.

### 7.2 Metrics
All three computed per the pre-registered gate (§7.3), with clustered bootstrap CIs (§7.2.3).

1. **Disagreement head-to-head (the headline).** Among cells where `|home_grown − sleeper|` (DK-base points) exceeds a **pre-registered** disagreement threshold `δ`, the fraction where our projection lands **closer to the actual** DK-base score than Sleeper's. **`δ` is fixed a priori in the plan (from a prior-year projection-difference distribution), not chosen to maximize the result** (H3). **Ties** (`|ours − actual| == |sleeper − actual|`, e.g. equidistant or coincident projections at the boundary) are **dropped from numerator and denominator** (documented; an alternative half-credit is a sensitivity check) (H3).
2. **Ranking skill.** Spearman(projection, actual) per source, and a top-N-at-position hit rate computed by ranking within `(position, week)` and comparing to actual weekly top-N (L2). Reported with a bootstrap CI on the *difference* vs. Sleeper (not a bare point comparison) so it has the same rigor as metric 1 (L4).
3. **Error (context only, not the verdict).** MAE / RMSE vs. actual base DK points. Low error that merely mirrors consensus is *not* edge.

### 7.2.3 Clustered bootstrap (H3)
Cells are **not** independent: the same player recurs across weeks, and same-week cells share a game environment. An i.i.d.-cell bootstrap understates CI width and over-declares significance. The bootstrap **resamples clusters — player-seasons (primary) — with replacement**, recomputing the paired statistic per resample. Two choices are **pre-registered** (not left as "alternative variants"), because the gate's reproducibility and the STOP/INCONCLUSIVE boundary (§7.4 keys on CI half-width in player-seasons) depend on them: (1) the disagreement subset (`|ours−sleeper| > δ`) is **re-derived inside each resample**, so δ-boundary uncertainty propagates into the CI; (2) the **primary** CI uses player-season clustering (captures same-player serial correlation); a **by-week block bootstrap** (capturing cross-player same-game correlation — an orthogonal source) is computed and **reported as a robustness check**, and if the two disagree on the verdict the result is treated as INCONCLUSIVE pending more data. This replaces the flat `bootstrap_mean` resample; the harness owns an explicit clustered implementation, tested on synthetic correlated data (§9).

### 7.3 The pre-registered gate (single primary test) (H2)
To avoid a multiple-comparisons machine (≥3 variants × 4 positions ≈ 12 simultaneous CIs → ~46% family-wise false-positive rate), the verdict rests on **one pre-registered primary test**:

- **Primary estimand:** the §7.2.1 disagreement head-to-head for **home-grown-only vs. Sleeper, pooled across QB/RB/WR/TE**, with a clustered-bootstrap 95% CI. The pooled statistic is a *fraction* (share of disagreement cells where we land closer), which is scale-free per position — so pooling avoids the QB-vs-TE base-points-scale problem. **Edge ⇔ this CI excludes 0.50 on the high side**, AND home-grown-only's pooled ranking-skill-difference CI (§7.2.2) is ≥ 0 (does not exclude 0 on the low side).
- **Anti-masking guardrail (M-4):** a pooled fraction is count-weighted across positions and could clear 0.50 while we actually *lose* at a position with few disagreement cells. So ADOPT additionally requires **no single position's point-estimate fraction below 0.50 by more than a pre-registered margin `m`** (and the report shows both the count-weighted pooled fraction and an equal-weight-across-positions average; a large gap between them flags composition-driven results). This keeps one primary test while preventing a single-position edge from masquerading as a general one.
- **Everything else is exploratory** (the blends, the per-position breakdowns): reported with CIs for hypothesis-generation, **explicitly labeled non-confirmatory**, and **not** sufficient on their own to declare edge. If a blend looks better than home-grown-only, that motivates a *follow-up* pre-registered test, not an ADOPT.
- Per-position exploratory results remain decision-relevant for *how* Layer 2 would be built (e.g. edge concentrated at WR), but do not gate the ADOPT/STOP decision.

### 7.4 Verdict tiers (M4)
- **ADOPT** — the primary test (§7.3) clears, and the §6.2 actuals-with-bonus sensitivity check does not flip it.
- **STOP** — the primary test is run with adequate power (below) and does not clear.
- **INCONCLUSIVE** — the universe has **too few clustered units** for the primary CI to be informative. A minimum is pre-registered (e.g. ≥ N_min player-seasons in the disagreement subset, and a target CI half-width); if unmet, the verdict is INCONCLUSIVE (collect more seasons / widen the universe), **never** STOP. This prevents a false STOP on thin data masquerading as absence of edge.

## 8. Schema additions

- `ExternalProjectionWeeklySchema` (pandera `DataFrameModel`), mirroring `ExternalProjectionSchema` (`schemas.py:820`): canonical `gsis_id` (pyarrow string), `source`, `season`, nullable-Float64 stat-line columns, **plus** a `week` column (`pd.Int64Dtype()`, era-aware regular-season bound — `le=22` with a REG filter, or an era-aware `le=18`, decided in the plan; cf. TODO #41). `strict="filter"`, validate-with-reassignment.
- **`_RULESET_NAME_VALUES` extended** to include `DRAFTKINGS`, plus the three pinned `isin` sites (`schemas.py:899,1244,1309`); the three deliberately-skipped sites (test literal `:1001`, `_RULESET_PRESETS`, `_RULESETS`) are justified in §5.2 (C3/M-1).
- **Weekly actuals week bound** reconciled (§5.4, prefer option b — a sibling `scoring/` scorer with its own era-aware bound, leaving `WeeklyActualSchema.week le=17` at `:1347` and `test_excludes_week_18` untouched). A separate `le=17` at `WeeklyProjectionSchema.week` (`:1333`) is noted but not on the DFS write path.
- **No new `Ruleset` fields for bonuses** — the `dk_actuals_bonus` helper (§5.2) is standalone scoring logic, not a schema/`Ruleset` change.
- No change to `ProjectionWeeklySchema`'s shape; `Ruleset` gains only the `draftkings()` preset (value-only).

## 9. Testing strategy

Per the project gates (`pytest`, `mypy --strict`, `ruff`, `ruff format`) and the ingest/store/schema integration tests:
- **Schema + ingest tests:** dtype correctness, `gsis_id` canonical, the float-stringified `sleeper_id` join (a row with `id_map.sleeper_id = '4374302.0'` must join), placeholder-rookie handling + coverage counting, `week` column + era-aware bound, skill-position filtering by actuals' position, no all-NA `pd.concat` FutureWarning.
- **DK ruleset unit tests:** known stat line → known DK base points; assert `fumble_lost_pts = −1.0`; assert `DRAFTKINGS` passes every updated allowlist.
- **`dk_actuals_bonus` test (§5.2/§6.2):** exactly +3 at each threshold boundary (299→0, 300→+3 pass; 99→0, 100→+3 rush/rec), stacking (e.g. 100 rush + 100 rec → +6), and that the sensitivity-check path adds it to actuals only (projections stay base).
- **Decode test (C2/§5.3):** a model prediction whose `params` encode known per-stat means decodes to those means and scores to the expected DK-base points (pins the params→stat-line→score path).
- **Leakage tests (§5.3) — both surfaces:** (a) `prior_mask` inputs: confirm the existing per-position leakage tests still hold at the weekly grain; (b) **NEW** — perturb/ inject future rows into the **trajectory/vegas/weather** inputs and assert the week-W projection is unchanged (the currently-unguarded `.shift(1)`/expanding surface, H-1). A failure here is a hard stop.
- **Metric-harness tests on synthetic inputs** where the answer is known: an always-closer source → head-to-head 1.0; identical sources → ties dropped (empty disagreement subset handled, not a divide-by-zero); monotone synthetic ranking → 1.0; **clustered bootstrap on correlated synthetic data yields wider CIs than an i.i.d. resample** (pins §7.2.3); INCONCLUSIVE fires below N_min.
- **Pre-registration test:** the disagreement threshold `δ` and the usage floor are read from a committed config/constant, not computed from the evaluation outcome (guards H3 threshold-snooping).
- **One-season end-to-end smoke** wiring ingest → projections → actuals → metric on a single season.
- Run `pytest -k "ingest or store or schemas"` per the dtype-regression rule.

## 10. Deliverables

1. Sleeper weekly ingest + `ExternalProjectionWeeklySchema` (`ingest/`).
2. `Ruleset.draftkings()` + the `_RULESET_NAME_VALUES` (and dependent-site) extension + base-DK scoring path.
3. The home-grown weekly **projection emitter** (thin path over the harness internals + params decode).
4. The era-aware, ruleset-parameterized weekly **actuals** scorer.
5. A benchmark module (a small `src/projections/dfs/` package — a home for Layer 2) computing §7 metrics with the clustered bootstrap + pre-registered gate, plus a thin CLI/script.
6. A committed verdict report `reports/dfs_projection_edge_<asof>.md`: the ADOPT/STOP/INCONCLUSIVE call from the **primary** test, the exploratory per-position/blend table (labeled non-confirmatory), the §6.2 sensitivity check, the coverage accounting, and the §4.3/§6.1 limitations stated.
7. TODO / `project_management.md` updates closing TODO #39 and recording the verdict.

## 11. Open questions (resolve during plan; none block feasibility)
- Exact Sleeper `stats`-key → `Stat`-enum field names (mapping sketched in §5.1; confirm against the enum).
- Concrete pre-registered values for the disagreement threshold `δ`, the actual-usage floor, the anti-masking per-position margin `m` (§7.3), and `N_min` + the target CI half-width (§7.4) — all derived from a prior-year distribution and committed before computing the verdict.
- Whether to extend `build_weekly_actuals` in place or add a sibling scorer in `scoring/` (§5.4).
- Whether the per-position feature builders already guarantee strictly-trailing windows at the weekly grain, or need a guard (§5.3 leakage).

## 12. References
- `CLAUDE.md` (conventions: `GsisId` canonical, `Ruleset` enums, schema-validate-with-reassignment, store I/O, ingest template, allowlist-update rule #10).
- TODO #39 (closed by this slice), TODO #1 (correlation, Layer 3), TODO #41 (era-aware playoff-week filtering), TODO #38 (the `id_map` `sleeper_id` dtype defect).
- Codebase seams: `schemas.py` (`Ruleset`:255-274, `_RULESET_NAME_VALUES`:302, pinned `isin` :899/:1244/:1309, test literal `test_dataframe_schemas.py:1001`, `ExternalProjectionSchema`:820, `ProjectionWeeklySchema`:906, `WeeklyActualSchema.week le=17`:1347, `WeeklyProjectionSchema.week le=17`:1333), `scoring/score.py`, `scoring/actuals.py`, `models/baseline.py:705,750` (`predict_distribution` scores `mean` under fit-time ruleset), `features/_shared.py:20` (`prior_mask`) + `features/wr.py:~255-294` (unguarded trajectory/vegas/weather), `ingest/external_projections.py` (Sleeper season path :54; `_attach_gsis_id` — dtype-only, does NOT strip `.0`), `scripts/benchmark_projections.py:_normalize_join_id`:93-106 (the `.0`-stripping logic — **lift into `src/`**, don't import from `scripts/`), `ingest/identity.py:18` (`placeholder_name_key`), `consensus/blend.py:94-102`, `backtest/harness.py` (return `BacktestRun.metrics`:81, loop :241, decode `_per_stat_means_from_predictions`:107), `draft/backtest/weekly_actuals.py` (`_MAX_WEEK`) + `test_weekly_actuals.py:42` (`test_excludes_week_18`), `draft/assistant/availability.py:24` (`_last_regular_week`, era-aware cutoff to reuse), `draft/league_config.py:16` (`_RULESET_PRESETS`) + `draft/assistant/presets.py:20` (`_RULESETS`) — DK skipped, justified §5.2, `draft/assistant/_compare.py:30` (`bootstrap_mean`, flat — replace with clustered), `scripts/benchmark_projections.py` (`build_benchmark_frame`:123-158, `top_n_hit_rate`:189, `actual_season_points`:60-68, `sleeper_id` dtype note :93), `store.write_partition/read_partition`.
