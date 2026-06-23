# DFS Engine — Layer 1: Projection Edge Study (design)

- **Date:** 2026-06-23
- **Status:** Approved (brainstorm), entering spec-review
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

## 2. Goal

Determine, with statistical confidence and per skill position, whether our home-grown weekly model — **alone or blended with Sleeper** — produces weekly fantasy-point projections that **beat Sleeper's own weekly projections** under DraftKings scoring, on historical NFL weeks.

The deliverable is a **repeatable benchmark harness** plus a **committed verdict report** carrying an explicit ADOPT (proceed to Layer 2) / STOP recommendation.

## 3. Scope & non-goals

### In scope
- Sleeper **weekly** projection ingest (historical, retrospective).
- A DraftKings scoring ruleset (base scoring; see §6.2 for the bonus decision).
- Running our home-grown weekly model **walk-forward** over past seasons to produce weekly projections.
- A small set of **blends** (home-grown + Sleeper) to test "some combination."
- A **metric harness** comparing each projection source against actuals (§7).
- A **verdict report** under `reports/`.
- **Skill positions only:** QB, RB, WR, TE.

### Explicit non-goals (deferred to later layers — do NOT build here)
- Salary-cap **optimizer** (ILP) — Layer 2.
- **Contest simulation / ROI** modeling — Layer 2.
- **Correlation / stacking** (QB+WR) — GPP-only, Layer 3 (see TODO #1, option D).
- **Ownership** data / leverage — Layer 3.
- **ESPN historical** ingest — ESPN keeps only ~1 prior season of projections, deletes history, and gates historical access behind an authenticated `espn_s2` cookie (verified 2026-06-23). Sleeper is the retrospective proxy here; an ESPN **forward-collection** path is a separate future slice. (Note: a `draft/backtest/espn_weekly.py` precedent already exists for ESPN-weekly handling and may inform that future slice; it is not consumed here.)
- **DST / K** projections.
- **Scoring bonuses** in the projection comparison — see §6.2.

## 4. Background facts (verified 2026-06-23)

### 4.1 DraftKings NFL Classic
- **Salary cap** $50,000; **9 roster slots**: QB, 2×RB, 3×WR, TE, FLEX (RB/WR/TE), DST. (Roster shape is irrelevant to this slice — it is a *per-player accuracy* study, not a lineup study — but recorded for Layer 2.)
- **Scoring** (the basis for `Ruleset.draftkings()`): passing 0.04/yd, 4/pass TD, −1/INT; rushing & receiving 0.1/yd, 6/TD; **1.0 PPR**; −1/fumble lost; +3 bonus at 300+ pass yds; +3 bonus at 100+ rush yds; +3 bonus at 100+ rec yds. **Note vs. existing code:** `Ruleset.fumble_lost_pts` defaults to **−2.0** (ESPN); DK is **−1.0** — the DK preset must override it.

### 4.2 Contest economics (Layer 2 context, recorded not used here)
- **Cash games** (50/50, double-up, H2H): beat ~top-44% after rake (profitable ≈ 56%+ win rate). Projection accuracy pays off most directly.
- **GPP tournaments**: top-heavy payouts; goal is a top-~1% finish; needs correlation + ownership leverage.

### 4.3 Data feasibility (this is why Sleeper, not ESPN)
- **Sleeper — historically retrievable.** `GET https://api.sleeper.app/projections/nfl/<season>/<week>?season_type=regular` returns weekly projections for any past season/week, no auth. Enables a **retrospective** study now.
- **ESPN — not reliably retrievable** historically (see §3 non-goals).
- **Our home-grown model** runs retrospectively on any past season (we have historical `weekly_stats` + features).

**Honest caveat:** Sleeper-alone is a *softer* proxy than the true DFS field (which aggregates many sharp sources). Beating Sleeper is therefore **necessary but not sufficient** for real DFS profitability. The gating logic still holds: *if we cannot beat Sleeper retrospectively, we certainly cannot beat the field* — so it is a valid, cheap kill-test. A sharper proxy is a Layer 2 concern.

## 5. Architecture & components

Six units, each with one clear purpose. Reuse existing machinery wherever noted; build new only where marked **NEW**.

### 5.1 Sleeper weekly ingest — **NEW** (extends the existing ingest pattern)
- Pull `api.sleeper.app/projections/nfl/<season>/<week>` for the target (season, week) cells.
- Normalize Sleeper's projection fields into our canonical stat line (passing/rushing/receiving yards & TDs, receptions, interceptions, fumbles lost). Map `sleeper_id → GsisId` via the existing id-map / crosswalk, reusing the placeholder-gsis handling for rookies (`ingest/identity.py`, `_placeholder_name_key`).
- Follow the canonical ingest template (`src/projections/ingest/external_projections.py` + `weekly_stats.py`); persist via `store.write_partition` only.
- **New schema** `ExternalProjectionWeeklySchema` (§8): `ExternalProjectionSchema` shape + a `week` column, `source` = Sleeper. Validate-with-reassignment at the module boundary.
- Confirm at plan time which Sleeper fields are populated for past seasons and how they map; restrict to skill positions.

### 5.2 DraftKings scoring ruleset — **NEW** (small extension to `scoring/`)
- Add `Ruleset.draftkings()` preset (the §4.1 values; **override `fumble_lost_pts` to −1.0**).
- Confirm the existing `Ruleset` already carries fields for passing-yards-per-point, TD points by type, reception points, interception points (read `schemas.py:245` at plan time). Add only what is missing.
- **Bonuses:** out of the *projection* comparison in v1 (§6.2). The `draftkings()` preset will still encode bonus *parameters* for later reuse, but the v1 scoring path used for the comparison runs **base scoring** (no bonuses) symmetrically across all sources and actuals.
- Scoring of projection point-estimates uses `scoring.score` / `expected_points`; scoring of actuals reuses `scoring.actuals.actual_season_total`'s weekly analogue (see 5.4).

### 5.3 Home-grown weekly projections — **REUSE** (`backtest/harness.py` walk-forward)
- Produce our model's weekly projection (mean — and distribution where available) for each `(gsis_id, season, week)` in the evaluation universe, **walk-forward** (train/feature window strictly prior to the projected week — no leakage).
- Reuse the existing model backtest harness (`src/projections/backtest/harness.py`) and the per-position model/feature builders; do not re-implement walk-forward.
- Output conforms to (or is adapted from) `ProjectionWeeklySchema`.

### 5.4 Actuals scorer — **REUSE** (`scoring/actuals.py` + `draft/backtest/weekly_actuals.py`)
- Compute each player's **actual** weekly DK (base) fantasy points from `weekly_stats`.
- Reuse `scoring.actuals.actual_season_total` semantics at weekly grain and/or the existing `draft/backtest/weekly_actuals.py`; confirm at plan time which is the right seam and extend minimally rather than duplicating scoring math (scoring layer is the single source of truth).

### 5.5 Blends — **REUSE** (`consensus/blend.py` pattern)
- Combine home-grown + Sleeper point projections at a small fixed set of weights (at minimum: home-grown-only, Sleeper-only [the baseline], 50/50; optionally one or two more).
- Reuse the consensus blend pattern; keep blending pure and testable. No learned weights in v1 (YAGNI — a tuned blend is a follow-up only if a fixed blend shows promise).

### 5.6 Metric harness & report — **NEW** (reuses `benchmark_projections.py` + paired bootstrap)
- Join all projection sources + actuals on `(gsis_id, season, week)`, restrict to the comparable universe (§7.1), compute the metrics (§7.2) with paired bootstrap CIs, per position, and emit the report.
- Reuse the join/scoring scaffolding from `scripts/benchmark_projections.py` and the paired-seed bootstrap pattern from the draft tournament (`draft/assistant` / `draft/backtest`).

## 6. Key design decisions

### 6.1 Sleeper-alone as the market proxy (v1)
Accepted as a softer-but-valid kill-test (§4.3). Recorded as a known limitation in the report; a multi-source / sharper proxy is Layer 2.

### 6.2 Base DK scoring in the projection comparison; bonuses excluded, symmetric (v1)
The +3 yardage bonuses are **probabilistic** for a projection: `E[bonus] = 3 · P(yards ≥ threshold)`, which a *point* projection cannot express. Our model has distributions and could estimate it; Sleeper provides only a point estimate. Including bonuses would therefore either (a) be unfair to Sleeper, or (b) require a bonus model we are deferring.

**Decision:** v1 scores **projections and actuals both under base DK scoring (no bonuses)**, symmetrically across all sources. This keeps the comparison apples-to-apples. Consequence: a small systematic understatement of absolute points, identical for every source, so the *relative* edge metric is unbiased.

**Noted upside for later:** estimating `E[bonus]` from our model's distribution (which Sleeper cannot) is a *candidate source of edge* — promoted to a Layer-2 experiment, not used to score v1.

### 6.3 Skill positions only (QB/RB/WR/TE)
Matches the draft sub-project's skill-only choice. DST/K deferred. The edge study is per-player accuracy and does not require a full DK lineup.

### 6.4 Walk-forward, no leakage
Home-grown projections for week *W* use only data strictly before week *W*. This is the same discipline as the existing model backtest; reuse it rather than re-deriving.

### 6.5 Comparable universe (anti-noise)
Restrict the comparison to cells **both** sources project, above a usage/snap floor, so deep-bench zeros do not dominate the metric (§7.1). The exact floor is a tunable spec parameter; default proposed in §7.1.

## 7. The metric (how we declare edge)

### 7.1 Comparable universe
- Players **both** Sleeper and our model project for that `(season, week)`.
- Above a **usage floor** to exclude noise from inactive/deep-bench players. Proposed default: actual snaps or a projected-points floor (e.g. projected ≥ ~5 DK base points by at least one source), **confirmed at plan time** against the data distribution.
- Restricted to QB/RB/WR/TE.
- Evaluation seasons: the recent past seasons for which both Sleeper weekly data and our model's walk-forward projections are available (target ≥ 3 seasons; exact set confirmed once Sleeper's historical depth is probed at implementation).

### 7.2 Metrics, per position, paired with bootstrap CIs
1. **Disagreement head-to-head (the headline).** Among cells where `|home_grown − sleeper|` exceeds a disagreement threshold, the fraction where our projection lands **closer to the actual** DK base score than Sleeper's. **Edge ⇔ this fraction's CI excludes 0.50 on the high side.** This isolates the only place edge can live.
2. **Ranking skill.** Spearman(projection, actual) per source, and a top-N-at-position hit rate (of each source's top-N at a position, how many actually finished top-N). Lineup-building consumes *ordering*; this measures it.
3. **Error (context only, not the verdict).** MAE / RMSE vs. actual base DK points. Reported for context; low error that merely mirrors consensus is *not* edge.

CIs via paired bootstrap over cells (reuse the existing paired-bootstrap pattern). The same three metrics are computed for each source/blend variant (home-grown-only, each blend) **against the Sleeper baseline**.

### 7.3 Verdict rule
**Edge declared** if, for at least one variant (home-grown-only or a blend) and at least one position, the **disagreement head-to-head** CI excludes 0.50 in our favor **and** that variant's ranking skill is **≥** Sleeper's at that position. The report states the verdict per (variant × position) and an overall ADOPT/STOP recommendation. A per-position result is decision-relevant (edge at WR but not QB still informs Layer-2 roster construction).

## 8. Schema additions

`ExternalProjectionWeeklySchema` (pandera `DataFrameModel`), mirroring `ExternalProjectionSchema` (`schemas.py:820`) with:
- A `week` column (`pd.Int64Dtype()`, regular-season bound — confirm era-aware max week at plan time, cf. TODO #41).
- Same `gsis_id` (canonical, pyarrow string), `source`, `season`, stat-line columns (nullable Float64 per the existing external schema).
- `strict="filter"`, validate-with-reassignment at the boundary.

No changes to `ProjectionWeeklySchema` or `Ruleset`'s shape beyond the new `draftkings()` preset (and any genuinely missing scoring field, justified at plan time).

## 9. Testing strategy

Per the project's verification gates (`pytest`, `mypy --strict`, `ruff`, `ruff format`), and the ingest/store/schema integration tests:
- **Schema + ingest tests:** dtype correctness, `gsis_id` canonical, placeholder-rookie handling, `week` column, skill-position filtering, no all-NA `pd.concat` FutureWarning (cf. the prior blend fix).
- **DK ruleset unit tests:** a known stat line → known DK base points; confirm `fumble_lost_pts = −1.0` for the DK preset.
- **Metric-harness tests on synthetic inputs** where the answer is known: a source that is always closer must score disagreement-head-to-head = 1.0; identical sources → 0.50; ranking-skill on a monotone synthetic = 1.0. These pin the metric math independent of real data.
- **One-season end-to-end smoke** wiring ingest → projections → actuals → metric on a single season/week.
- Run `pytest -k "ingest or store or schemas"` per the dtype-regression rule, since this touches an ingest/store path.

## 10. Deliverables

1. `src/projections/ingest/` Sleeper weekly ingest + `ExternalProjectionWeeklySchema`.
2. `Ruleset.draftkings()` + base-DK scoring path.
3. A weekly-actuals DK scorer (reused/extended).
4. A benchmark module (likely `src/projections/dfs/` or `scripts/`) computing §7 metrics, plus a thin CLI/script to run it.
5. A committed verdict report `reports/dfs_projection_edge_<asof>.md` with the ADOPT/STOP recommendation, per-position table, and the §6.1 limitation stated.
6. TODO / `project_management.md` updates closing TODO #39 and recording the verdict.

## 11. Open questions (resolve during spec-review / plan)
- Exact Sleeper field → stat-line mapping and historical depth (how many seasons back are usable).
- The usage floor and disagreement threshold defaults (§7.1, §7.2) — pick concrete values from the data distribution.
- Where the new benchmark code lives: a new `src/projections/dfs/` package vs. extending `scripts/` + `backtest/`. Lean toward a small `dfs/` package so Layer 2 has a home.
- Whether to reuse `draft/backtest/weekly_actuals.py` directly or factor a shared weekly-actuals helper into `scoring/`.

## 12. References
- `CLAUDE.md` (conventions: `GsisId` canonical, `Ruleset` enums, schema-validate-with-reassignment, store I/O, ingest template).
- TODO #39 (this slice closes it), TODO #1 (correlation, Layer 3), TODO #41 (playoff-week filtering at ingest).
- Codebase seams: `schemas.py` (`Ruleset`:245, `ExternalProjectionSchema`:820, `ProjectionWeeklySchema`:906), `scoring/score.py`, `scoring/actuals.py`, `ingest/external_projections.py`, `consensus/blend.py`, `backtest/harness.py`, `draft/backtest/weekly_actuals.py`, `draft/backtest/espn_weekly.py`, `scripts/benchmark_projections.py`, `store.write_partition/read_partition`.
