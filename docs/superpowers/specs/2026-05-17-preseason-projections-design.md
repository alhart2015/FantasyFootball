# Preseason Projections — v1 Framework + Naive Baseline — Design

**Status:** draft (brainstorming, 2026-05-17). Ready for user review.
**Date:** 2026-05-17
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Draft Hub (parallel to the existing in-season weekly pipeline)
**Branch:** `worktree-feat+preseason-projections` cut from `main` at `8ffa607` (post-PR #46 merge).

**Depends on:** the existing `data/raw/` ingest layer (weekly_stats, depth_charts, draft_picks, id_map, schedules) on `nflreadpy` for 2018-2025 (closed under TODO #32). No new ingest required. **Specifically requires that `data/raw/depth_charts/season=2026/` is materialized** before the v1 pipeline can produce a 2026 projection — `depth_charts_2026.parquet` is already published upstream by nflverse (HEAD-probed 200 on 2026-05-11; see TODO #32 footer); only the local refresh is missing.

**Related specs / docs:**

- `TODO.md` #31 — "Preseason full-season projections for the Draft Hub (required, not yet built)" — the original gap statement that motivates this spec.
- `draft_ready_checklist.md` §1a (data the prediction depends on) + §1b (code surface for `predict_season.py SEASON`) — this spec collectively delivers both.
- `docs/superpowers/specs/2026-05-16-vorp-design.md` — the immediate downstream consumer of v1 preseason output. VORP consumes a season-total-fpts table per `(gsis_id, position)`.
- `CLAUDE.md` — names "Draft Hub" as a planned sub-project; the `src/projections/preseason/` sub-package introduced here is parallel to the existing `src/projections/{models, features, scoring, distributions, store, ingest, backtest}` peers.

---

## 1. Overview

The current Projections Core produces *weekly in-season* projections — every per-position feature builder consumes trailing-N in-season stats (`*_per_game_l4`, `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`, NGS rolling windows). For Week 1 of a new season the builders fall back to the prior season's trailing data; for any week ≥ 2 they require games already played in the current season. None of that path can serve the Draft Hub's preseason use case.

This spec ships a parallel sub-package `src/projections/preseason/` whose v1.0 deliverable is:

- A **framework**: feature schema, model interface, projection driver, walk-forward backtest harness, and storage layout — all sized to the season-total target rather than per-week.
- A **naive baseline model** (`NaivePreseasonModel`) that returns `prior_season_per_game × projected_games` for veterans and a draft-capital-anchored Gamma GLM for rookies. v1.0 ships only this baseline as a degenerate (point-mass) distribution.
- A **walk-forward backtest** over target seasons `{2024, 2025}` that emits a `PreseasonBacktestSchema`-validated frame plus a markdown verdict report.

v1.5 is a separate spec that adds the first trained model class. That spec's job is to beat the v1.0 naive baseline on a published gate (§7). v1.0's job is to define the framework + characterize the floor those future models have to clear.

### 1.1 Goals (in scope)

- **New sub-package `src/projections/preseason/`** with four modules:
  - `features.py` — `build_preseason_features(...)` returning a `PreseasonFeaturesSchema`-validated frame.
  - `model.py` — `PreseasonModel` Protocol + `NaivePreseasonModel` implementation.
  - `project.py` — end-to-end driver: features → model → scoring → per-player season-total per-stat distribution.
  - `backtest.py` — walk-forward eval harness producing per-position RMSE + Spearman vs naive baseline.
- **Three new pandera schemas** added to `src/projections/schemas.py`:
  - `PreseasonFeaturesSchema` — one row per `(gsis_id, season)` rostered player. See §3.1.
  - `PreseasonProjectionSchema` — one row per `(gsis_id, season, ruleset)`. Per-stat season-total quantiles + scored fpts quantiles. See §3.4.
  - `PreseasonBacktestSchema` — one row per `(target_season, position, model_class)`. Per-position metrics + verdict. See §3.5.
- **Two new CLI scripts**:
  - `scripts/preseason_project_season.py` — `--season 2026 [--ruleset espn_ppr]`.
  - `scripts/backtest_preseason.py` — `--model naive-preseason --target-seasons 2024,2025 [--train-start 2018]`.
- **New storage paths** (extends the existing store layout):
  - `data/projections/preseason/season=<Y>/ruleset=<R>/part.parquet` — the v1 output.
  - `data/features/preseason/season=<Y>/part.parquet` — the feature frame, cached for backtest reuse.
  - `reports/backtest_preseason_<model>.md` + `reports/backtest_preseason_<model>.csv`.
  - `reports/preseason_<season>.csv` — human-readable top-N-per-position summary.
- **Tests** in `tests/test_preseason/` covering features (8 cases), model (6 cases), backtest (4 cases) + integration tests for both scripts. See §6.
- **No new ingest.** Reuses `data/raw/{weekly_stats, depth_charts, draft_picks, id_map, schedules}` as-is.
- **No new scoring.** Reuses `src/projections/scoring/` — the preseason model predicts per-stat season totals, and the project driver runs them through `scoring.score_season(ruleset)` to obtain fpts.

### 1.2 Non-goals (deferred)

- **No trained model.** v1.0 ships only the naive baseline. The first trained model class (GammaGLM / LightGBM-NB / similar) lands in a v1.5 spec that gates on §7.
- **No K / DST.** Out of scope for v1; TODO #10 still open. v1 output is QB / RB / WR / TE only. The draft-readiness checklist §1a flags K/DST as a parallel item.
- **No ADP integration.** No ADP ingest, no ADP-comparison gate. v1 gate is naive-prior only. ADP is a sibling project (`draft_ready_checklist.md` §2b.3).
- **No injury prior.** `projected_games_played` is hardcoded to **16** in v1 (the historical median across 2018-2025 rostered player-seasons; treated as a model constant, not a per-player feature). v2 ships per-player injury priors and makes this player-specific.
- **No explicit role-adjustment formula for free agents.** v1 surfaces `depth_chart_rank` and `team` as features but the naive baseline ignores them. They exist in the schema so the v1.5 trained model can learn from them. FAs in v1 are projected using prior-season per-game as-is — a documented limitation.
- **No college-stats ingest for rookies.** v1 rookie path is draft-capital only: `(draft_round, draft_pick_overall, position) → rookie-year stats` Gamma GLM.
- **No per-week breakdown in preseason output.** v1 outputs season totals only — one row per `(gsis_id, season, ruleset)`. Bye-week splits, opponent matchups, etc. are weekly-pipeline concerns.
- **No live update path.** v1 is run once preseason. In-season "what changed since last week" diffs are the weekly pipeline's job.
- **No multi-ruleset output in one run.** One `--ruleset` per script invocation, mirroring the existing weekly path.

---

## 2. Architecture

```
src/projections/preseason/                  (new sub-package)
├── __init__.py
├── features.py                              (new — build_preseason_features)
├── model.py                                 (new — PreseasonModel Protocol + NaivePreseasonModel)
├── project.py                               (new — end-to-end project_preseason)
└── backtest.py                              (new — walk_forward_backtest)

src/projections/schemas.py                   (edited — add 3 schemas)

scripts/preseason_project_season.py          (new — CLI)
scripts/backtest_preseason.py                (new — CLI)

tests/test_preseason/                        (new directory)
├── __init__.py
├── test_features.py
├── test_model.py
└── test_backtest.py

tests/test_scripts/
├── test_preseason_project_season_cli.py     (new)
└── test_backtest_preseason_cli.py           (new)
```

The sub-package sits peer to `src/projections/{models, features, scoring, distributions, store, ingest, backtest}`. It depends on `scoring` and `store` (and indirectly `schemas`); it does NOT depend on `models`, `features`, or `backtest` — those are weekly-pipeline concerns with different feature shapes and target shapes.

---

## 3. Schemas & data shapes

### 3.1 `PreseasonFeaturesSchema`

One row per `(gsis_id, season)` for every player on `depth_charts_<season>` with `position in {QB, RB, WR, TE}`.

**Identity columns:**

| Column | Dtype | Nullable | Source |
|---|---|---|---|
| `gsis_id` | `GsisId` | No | canonical key |
| `season` | int32 | No | target season Y |
| `position` | `Position` enum | No | `depth_charts_Y` |
| `team` | `Team` enum | No | `depth_charts_Y` |
| `depth_chart_rank` | `pd.Int64Dtype()` | No | `depth_charts_Y` |

**Player-profile columns:**

| Column | Dtype | Nullable | Source |
|---|---|---|---|
| `age` | float32 | Yes | `Y - birth_year` from `id_map` |
| `years_exp` | `pd.Int64Dtype()` | No | `Y - rookie_year`. 0 for rookies. |
| `is_rookie` | bool | No | `years_exp == 0` |
| `draft_round` | `pd.Int64Dtype()` | Yes | `draft_picks` (NaN for UDFAs) |
| `draft_pick_overall` | `pd.Int64Dtype()` | Yes | `draft_picks` (NaN for UDFAs) |

**Prior-season per-game aggregates** — for each of the last 3 seasons `Y-1`, `Y-2`, `Y-3`, one column per modeled stat. The set of modeled stats is position-dependent and mirrors what the weekly path predicts:

- QB: `passing_yards`, `passing_tds`, `passing_interceptions`, `rushing_yards`, `rushing_tds`.
- RB: `rushing_yards`, `rushing_tds`, `receptions`, `receiving_yards`, `receiving_tds`.
- WR: `receptions`, `receiving_yards`, `receiving_tds`, `rushing_yards`, `rushing_tds`.
- TE: `receptions`, `receiving_yards`, `receiving_tds`.

Column naming: `prior_{1,2,3}_season_per_game_<stat>`. Dtype `float32`, nullable. NaN if the player did not record that prior season (rookie, sophomore, missed-full-season, IR, retired-and-came-back).

Also include `prior_{1,2,3}_season_games_played` (`pd.Int64Dtype()`, nullable) — useful both as a feature (durability proxy) and as the denominator when computing per-game stats during the build.

**Rationale for prior-3 window:** matches the existing weekly trajectory features' 3-season lookback (TODO #24 / #25). Keeps feature count bounded; aligns with ingest window (a prior-5 window would reach back to 2013 for a 2018 target — outside the 2017+ weekly_stats coverage on nflreadpy).

### 3.2 No NGS / PBP / snap_count features in v1

By design. Those are weekly-path concerns. The preseason path's signal is dominated by season-aggregate volume + role, not per-snap efficiency. v1.5 can add them if the trained model demands them.

### 3.3 Feature builder contract

```python
# src/projections/preseason/features.py

def build_preseason_features(
    *,
    weekly_stats: pd.DataFrame,           # all seasons through target_season - 1
    depth_charts_target: pd.DataFrame,    # season=target_season only
    draft_picks: pd.DataFrame,            # all seasons through target_season
    id_map: pd.DataFrame,
    target_season: int,
) -> pd.DataFrame:
    """Returns a frame validated against PreseasonFeaturesSchema."""
```

Side effects:

- Players on `depth_charts_target` with no `id_map` entry are dropped and surfaced in a per-call return tuple — actually no, mirror the weekly-path convention: drop, log a `WARNING` via the project's standard logger, write a side-channel CSV at `reports/preseason_dropped_<season>.csv` recording `gsis_id, drop_reason`. Function return type stays a single DataFrame.
- Players with `position` outside `{QB, RB, WR, TE}` (K, DST, FB, LS, P, OL, DL, LB, DB, etc.) are filtered out silently. Row count logged at INFO.
- Duplicate `(gsis_id, season)` rows raise `ValueError` — indicates upstream depth_charts dedup bug; never silently swallow.

### 3.4 `PreseasonProjectionSchema`

One row per `(gsis_id, season, ruleset)`. Per-stat season-total quartet `<stat>_season_total_{mean, p10, p50, p90}` for each stat in the player's position's stat set, plus the scored `season_total_fpts_{mean, p10, p50, p90}` per ruleset.

**Identity columns:**

| Column | Dtype | Nullable |
|---|---|---|
| `gsis_id` | `GsisId` | No |
| `season` | int32 | No |
| `position` | `Position` enum | No |
| `team` | `Team` enum | No |
| `ruleset` | str (enum value: `ESPN_PPR` / `ESPN_HALF` / `STANDARD`) | No |
| `model_id` | `pd.StringDtype("pyarrow")` | No |

**Scored fpts:** `season_total_fpts_mean`, `_p10`, `_p50`, `_p90` (all `float32`, non-negative).

**Per-stat quartets:** for each modeled stat in the player's position's stat set, one quartet of `<stat>_season_total_{mean,p10,p50,p90}` (`float32`, non-negative). Columns not modeled for a position are absent (not NaN-filled). Pandera schema uses `strict="filter"` to preserve only the declared columns.

### 3.5 `PreseasonBacktestSchema`

One row per `(target_season, position, model_class)`.

| Column | Dtype | Notes |
|---|---|---|
| `target_season` | int32 | 2024 or 2025 for v1 |
| `position` | `Position` enum | QB / RB / WR / TE |
| `model_class` | str | e.g., `"naive-preseason-v1"` |
| `ruleset` | str | the eval ruleset (default ESPN_PPR) |
| `rmse` | float32 | RMSE of predicted vs actual `season_total_fpts` |
| `rmse_naive_baseline` | float32 | RMSE of `prior_1_season_per_game × 16` — the zero-skill floor |
| `rmse_delta_pct` | float32 | `(rmse - rmse_naive) / rmse_naive × 100`. Negative = beat baseline |
| `spearman_top50` | float32 | Spearman rank-correlation, restricted to top 50 actual finishers |
| `n_players` | `pd.Int64Dtype()` | Eval set size for this cell |
| `coverage_diff_projected_not_played` | `pd.Int64Dtype()` | Players we projected who didn't play |
| `coverage_diff_played_not_projected` | `pd.Int64Dtype()` | Players who played but we didn't project (rookies missing from `depth_charts`, mid-season call-ups) |
| `verdict` | str enum: `ADOPT` / `NULL` / `DO_NOT_ADOPT` | See §7 |

---

## 4. The `PreseasonModel` Protocol & `NaivePreseasonModel`

### 4.1 Protocol

```python
# src/projections/preseason/model.py

from typing import Protocol

class PreseasonModel(Protocol):
    """v1.0 returns degenerate point-mass distributions; v1.5+ returns real distributions."""

    model_id: str  # e.g., "naive-preseason-v1"

    def fit(
        self,
        *,
        weekly_stats: pd.DataFrame,    # full training window (e.g., 2018..target-1)
        draft_picks: pd.DataFrame,
        id_map: pd.DataFrame,
    ) -> None: ...

    def predict_season_distribution(
        self,
        features: pd.DataFrame,        # validated against PreseasonFeaturesSchema
        *,
        ruleset: Ruleset,
    ) -> pd.DataFrame:                 # validated against PreseasonProjectionSchema
        ...

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "PreseasonModel": ...
```

`predict_season_distribution` composes the per-stat prediction with scoring inside the model — but scoring math comes from `scoring.score_season(ruleset)`, never re-implemented. The model returns the full output frame; the project driver only handles I/O.

### 4.2 `NaivePreseasonModel` — three-branch logic

**Branch 1 — veterans with prior-year stats** (`is_rookie=False AND prior_1_season_games_played > 0`):

```
predicted_<stat>_season_total = prior_1_season_per_game_<stat> × 16
```

Degenerate distribution: `mean = p10 = p50 = p90 = predicted_<stat>_season_total`.

Why prior-1 only and not weighted prior-1/2/3? Keeps v1.0 maximally simple — gives the v1.5 trained model a clear lever ("use longer history") to beat the baseline on.

**Branch 2 — veterans missing prior-1 season** (`is_rookie=False AND (prior_1_season_games_played in {0, NaN})`):

Fall back to `prior_2_season_per_game_<stat>`; if also missing, fall back to `prior_3`. If all three missing, drop the player from output with a log warning + side-channel `reports/preseason_dropped_<season>.csv` entry (`drop_reason="veteran_no_prior_3_seasons"`).

**Branch 3 — rookies** (`is_rookie=True`):

Fit a per-position rookie-year Gamma GLM at `NaivePreseasonModel.fit()` time:

- Training data: rookies whose rookie-year fell in `[train_start, target_season - 1]` (default 2018-2024 for a target of 2025, or 2018-2025 for a target of 2026), joined to their rookie-year `weekly_stats` aggregate. ~110 modeled-position rookies/year (≈10 QB, ≈30 RB, ≈50 WR, ≈20 TE) over 7-8 years → ~800-1,000 observations total; the smallest per-position cell (QB) has ~70-90 rows, enough for a 1-regressor GLM.
- One GLM per `(position, stat)`. Family = Gamma; link = log; single regressor = `log(draft_pick_overall + 1)`.
- Prediction: `predicted_<stat>_season_total = exp(β₀ + β₁ · log(pick + 1)) × 16` for the rookie's draft pick.
- UDFAs: impute `draft_pick_overall = 300` before prediction. This is a step past the last drafted player (32×7 = 224 picks; 300 is a clear post-draft constant) and produces a "very late-round equivalent" projection.

Degenerate distribution: `mean = p10 = p50 = p90 = GLM prediction × 16`.

**Persistence.** `NaivePreseasonModel.save(path)` writes the rookie GLM coefficients per `(position, stat)`. `.load(path)` rehydrates. Artifact name pattern: `naive-preseason-<YYYY-MM-DD>-<train_start>-<train_end>.joblib`, stored under `models/artifacts/`. Mirrors the existing weekly-model pattern.

### 4.3 Free agents and team changes

v1 naive baseline does nothing special. `prior_1_season_per_game` gets used as-is regardless of `team` change between `season-1` and `season`. The `depth_chart_rank` feature is materialized in the schema but `NaivePreseasonModel` ignores it.

**This is the explicit slot for the v1.5 trained model.** That model learns role-adjustment by training on the `(prior_*_season_per_game, depth_chart_rank, team) → season_total` relationship across 2018-2025. The naive baseline's "ignore team changes" is the bug v1.5 is built to fix; the §7 gate explicitly measures the win.

The spec documents this limitation in the v1 ship notes.

### 4.4 Distribution shape contract

v1.0 naive returns degenerate point-mass. Downstream consumers (VORP, auction $, snake cheat sheet, future "confidence bands per ranking" per `draft_ready_checklist.md` §2a.3) see `p10 == p50 == p90 == mean`. They degrade gracefully — they order by mean, and "confidence band" rendering is a no-op for zero-width intervals.

v1.5+ returns real distributions (e.g., GammaGLM with dispersion, LightGBM-quantile, etc.); the framework supports that without schema changes.

---

## 5. End-to-end data flow

For a target season `Y`:

1. Read raw inputs (all via `store.read_partition`):
   - `weekly_stats` for all seasons `train_start..Y-1` — used for both training (rookie GLM) and feature aggregation.
   - `depth_charts` for `Y` — the rostered-player gate.
   - `draft_picks` for all seasons `1980..Y` — historical draft pick info.
   - `id_map` — name + birth_date + cross-platform IDs.
   - `schedules` for `Y` — currently unused in v1 features (no opponent-strength features) but read for future use and ingest verification.
2. Compute per-season per-game aggregates from `weekly_stats` (one row per `(gsis_id, season)` × stat columns / `games_played`).
3. Call `features.build_preseason_features(...)` — returns a `PreseasonFeaturesSchema`-validated frame.
4. Cache the feature frame to `data/features/preseason/season=Y/part.parquet` for backtest reuse (idempotent overwrite).
5. Fit `NaivePreseasonModel.fit(weekly_stats=..., draft_picks=..., id_map=...)` — fits the per-`(position, stat)` rookie GLMs.
6. Call `model.predict_season_distribution(features, ruleset=ruleset)` — returns a `PreseasonProjectionSchema`-validated frame.
7. Write to `data/projections/preseason/season=Y/ruleset=<R>/part.parquet`.
8. Optionally write a human-readable top-N CSV summary to `reports/preseason_<Y>.csv` (gated behind `--summary` flag, default on).

The backtest harness inverts steps 1-7 over `Y ∈ {2024, 2025}` with the training window restricted to `train_start..Y-1`, then joins predictions to actuals (aggregated from `weekly_stats[season=Y]`), computes per-cell metrics, and writes the `PreseasonBacktestSchema` rows plus a markdown report.

---

## 6. Error handling

| Condition | Behavior |
|---|---|
| `depth_charts_Y` partition missing | Error: `"depth_charts season=Y not found at <path>. Run refresh_depth_charts({Y}) first."` Don't silently produce empty output. |
| `weekly_stats season=Y-1` missing | Falls back to Y-2 then Y-3 per Branch 2 logic. If all three missing for a player, drop with warning. |
| `draft_picks season=Y` missing | Error if any 2026-roster player is flagged rookie. Otherwise pass through. |
| Player in `depth_charts_Y` with no `id_map` entry | Drop, log WARNING, append to `reports/preseason_dropped_<Y>.csv`. |
| Player marked rookie in `depth_charts_Y` but missing from `draft_picks` (UDFA) | Impute `draft_pick_overall = 300`. |
| Negative or zero `prior_1_season_games_played` | Treated as missing; trips Branch 2 fallback. |
| Feature frame violates `PreseasonFeaturesSchema` | Pandera raises at builder boundary. Fail fast. |
| Projection frame violates `PreseasonProjectionSchema` | Pandera raises before write. Partition does not land on disk. |
| K, DST, or any non-skill position on `depth_charts_Y` | Filtered out at builder entry. Row count logged at INFO. |
| Duplicate `(gsis_id, season)` rows in features | `ValueError` — upstream dedup bug; never silently swallow. |
| Player's prior season has games_played > 17 (rare playoff/regular conflation) | Cap at 17 before computing per-game; log WARNING. |

---

## 7. Backtest gate & verdict logic

### 7.1 Walk-forward algorithm

For each `target_season ∈ {2024, 2025}`:

1. Fit `NaivePreseasonModel` (i.e., fit the rookie-year GLM) on data covering `[train_start, target_season - 1]`. Default `train_start = 2018`.
2. Build features for `target_season` using only data available "at preseason time" — weekly_stats through `target_season - 1`, depth_charts for `target_season`, draft_picks through `target_season`.
3. Predict per-player season totals.
4. Aggregate `weekly_stats[season=target_season]` to actual per-player season totals (sum per-week fpts, where fpts is computed via `scoring.score_week(ruleset)`).
5. Inner-join predicted vs actual on `gsis_id`. Coverage diff (projected_not_played + played_not_projected) is reported separately in the markdown but excluded from the metric set — including no-play players inflates RMSE in ways unrelated to model quality.

### 7.2 Per-cell metrics

Per `(target_season, position)` cell:

- `rmse` — RMSE of predicted vs actual `season_total_fpts.mean` for the eval set.
- `rmse_naive_baseline` — RMSE of `prior_1_season_per_game × 16` for the same set. For v1.0 naive this equals `rmse` by construction; for v1.5+ trained models this is the floor to beat.
- `rmse_delta_pct` — `(rmse - rmse_naive) / rmse_naive × 100`. Negative = beat baseline. v1.0 = 0 by construction.
- `spearman_top50` — Spearman rank-correlation between predicted-rank and actual-rank, restricted to the **top 50 actual finishers** in that position. This is the rank-quality metric that drives draft-day usefulness — we don't care about the WR80 → WR85 ordering, we care about whether our top 24 WRs match reality's top 24.
- `n_players` — eval-set size after inner join.

### 7.3 Verdict logic

Per cell:

- `ADOPT` if `rmse_delta_pct < 0` AND `spearman_top50 ≥ 0.70`.
- `DO_NOT_ADOPT` if `rmse_delta_pct ≥ 0` (model is worse than baseline) OR `spearman_top50 < 0.50` (rankings worse than what a coin-flip-with-prior-year-weighting would give).
- `NULL` otherwise (e.g., a small RMSE win with Spearman 0.50-0.70). Surfaced for human review; doesn't auto-ship.

### 7.4 Aggregate verdict for the v1.5+ ship gate

Cells = 2 target_seasons × 4 positions = 8.

- **v1.0 ship gate (this spec):** none — v1.0 ships the framework + naive baseline regardless of verdict numbers. The v1.0 backtest is purely characterization, producing a floor for v1.5 specs to compare against.
- **v1.5+ ship gate (future spec):** **≥ 6 of 8 cells `ADOPT` AND zero cells `DO_NOT_ADOPT`.** Matches the project's existing backtest culture (any single-cell regression blocks ship). This spec defines the gate; v1.5 inherits it.

### 7.5 Calibration — informational only

For models that emit non-degenerate distributions (v1.5+), the backtest also computes `coverage_80` per cell: fraction of actuals falling inside `[p10, p90]`. Expected ~80%. Reported in the markdown; does NOT affect verdict. Matches the existing project pattern (`draft_ready_checklist.md` §1c) where calibration is informational while draft tooling consumes the mean.

### 7.6 Report format

`reports/backtest_preseason_<model>.md` contains:

- Header (model_id, train_start, target_seasons, ruleset, run timestamp).
- One `PreseasonBacktestSchema` table per `(target_season, position)` cell — 8 cells.
- Aggregate verdict count: `n_adopt / n_null / n_do_not_adopt`.
- For each position: top-20 predicted vs actual table for spot-checking. Format: `predicted_rank, actual_rank, player, predicted_fpts, actual_fpts, delta`.
- Coverage-diff sidebars: lists of `projected_not_played` and `played_not_projected` players per cell.

`reports/backtest_preseason_<model>.csv` is the `PreseasonBacktestSchema`-validated frame for programmatic consumption.

---

## 8. Testing

### 8.1 `tests/test_preseason/test_features.py`

- `test_veteran_with_three_prior_seasons` — player active 2021/22/23, projecting 2024; all three prior columns populated correctly with per-game stats.
- `test_veteran_missing_prior_1_season` — on 2024 roster, no 2023 stats but 2022 stats; `prior_1_*` NaN, `prior_2_*` populated.
- `test_rookie_in_draft_picks` — 2024 rookie with draft_round + draft_pick_overall; `is_rookie=True`, `years_exp=0`, prior_* all NaN.
- `test_rookie_udfa` — on roster, `is_rookie=True`, missing from `draft_picks` for that year; verifies upstream gap is preserved (imputation happens in model layer, not features).
- `test_team_change_player_keeps_prior_stats` — player on Team A in 2023, Team B in 2024; `depth_chart_rank` and `team` reflect B, `prior_1_*_per_game` reflects A.
- `test_missing_id_map_entry_drops_with_warning` — assert dropped_players CSV surfacing + WARNING log captured.
- `test_position_filter_excludes_k_dst` — depth_chart has K + DST rows; features frame doesn't.
- `test_schema_validates` — `PreseasonFeaturesSchema.validate(frame)` passes on the golden-path fixture.

### 8.2 `tests/test_preseason/test_model.py`

- `test_naive_predict_veteran_branch` — single veteran, predicted stat = `prior_1_per_game × 16`.
- `test_naive_predict_rookie_branch` — single rookie, predicted stat = `exp(β₀ + β₁ · log(pick+1)) × 16`.
- `test_naive_predict_fallback_branch` — veteran missing prior-1, falls through to prior-2.
- `test_naive_predict_degenerate_distribution` — all quantiles (`mean`, `p10`, `p50`, `p90`) equal for v1.0.
- `test_rookie_glm_fit_persists_and_roundtrips` — `.fit()` → `.save()` → `.load()` → `.predict_season_distribution()` matches exactly.
- `test_predict_output_schema_validates` — output passes `PreseasonProjectionSchema.validate`.

### 8.3 `tests/test_preseason/test_backtest.py`

- `test_walk_forward_split` — train_start=2018, target=2024 → fit uses 2018-2023 data only; train data for 2024+ excluded.
- `test_metric_computation_rmse_and_spearman` — synthetic predicted + actual frames, asserted exact RMSE + Spearman values.
- `test_verdict_logic_thresholds` — parametrized over `(rmse_delta_pct, spearman_top50) → expected_verdict`; covers ADOPT / NULL / DO_NOT_ADOPT bands including boundary values.
- `test_inner_join_coverage_diff_reported` — projected_not_played + played_not_projected counts surface in the backtest frame.

### 8.4 Integration tests

`tests/test_scripts/test_preseason_project_season_cli.py` — happy-path integration on a 3-position, 6-player fixture; asserts parquet partition written + schema valid + CSV summary emitted.

`tests/test_scripts/test_backtest_preseason_cli.py` — same fixture; asserts markdown report contains expected sections (verdict table, per-position metrics, coverage diff).

### 8.5 Coverage targets

- `features.py`, `model.py`, `project.py`, `backtest.py` — ≥ 95% line coverage each.
- Scripts — ≥ 80% (entry-point glue is harder to fully exercise).
- No new fixture files in `tests/fixtures/`. Synthesize fixture frames inline using the existing `_make_*` helpers pattern from `tests/test_draft/test_vorp.py`. Keeps the fixture surface bounded.

---

## 9. Risks & open items

- **Naive baseline rookie GLM choice — Gamma on `log(pick+1)`.** Simplest defensible option. Alternatives: empirical-Bayes "median of rookies within pick window", or a Poisson GLM on counts. Picked Gamma because it cleanly handles the continuous-non-negative stats (yards, fpts); count stats (TDs, receptions) get the same treatment for simplicity, knowing it's slightly mis-specified — this is the floor model, not the production model.
- **UDFA imputation at pick=300 is arbitrary.** Defensible (post-draft constant) but the v1.5 trained model should treat UDFA as a separate categorical rather than a synthetic pick number. Flagged for the v1.5 spec.
- **`projected_games_played = 16` is a constant.** Median over 2018-2025. v2 injury-prior work replaces this.
- **2026 depth charts refresh required before v1 can produce output.** Spec assumes `depth_charts_2026.parquet` ingest happens out-of-band as a one-liner: `from projections.ingest.depth_charts import refresh_depth_charts; refresh_depth_charts(seasons=[2026])`. Pre-ingest the 2026 partition before running the v1 driver. The v2025+ derivation path landed in PR #37 / TODO #34 — confirm it produces sensible 2026 output before relying on it for v1 ship.
- **Free-agent / team-change accuracy is bad in v1.** Documented limitation. The v1.5 trained model is the fix; the §7.4 gate measures the actual win.
- **No backtest verdict gates v1.0 itself.** The framework + naive baseline ships as the floor. Easy to misread the §7 gate as applying to v1.0; the spec is explicit that it applies to v1.5+ only.

---

## 10. Workflow

- Spec → plan → execute, all on `worktree-feat+preseason-projections`. Reaches `main` only via PR.
- Project entries in `project_management.md` updated when the v1 framework ships.
- `draft_ready_checklist.md` §1b first row (`predict_season.py SEASON` end-to-end) flips to `[x]` when this spec's `scripts/preseason_project_season.py` lands AND a 2026 partition has been successfully written via it.
- TODO.md #31 closed-out with a pointer to this spec.
