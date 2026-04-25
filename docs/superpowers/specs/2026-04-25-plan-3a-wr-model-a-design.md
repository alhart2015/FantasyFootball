# Plan 3a — WR Model A baseline (end-to-end pipeline) — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-25
**Author:** alden + claude
**Builds on:** `2026-04-24-projections-core-design.md` (foundations); `2026-04-24-plan-2a-ingest-and-wr-features-design.md` (WR features); `2026-04-24-plan-2b-qb-rb-te-features-design.md` (other-position features)
**Plan-3 series context:** Plan 3 is split into three sub-plans following the validated 2a→2b precedent.
- **Plan 3a (this design):** Model A on WR only, end-to-end. Pins the `Model` interface, the per-stat-regression pipeline, and persistence / registry mechanics. Includes the first real-data ingest pull.
- **Plan 3b:** generalize Model A to QB / RB / TE. Mostly mechanical once 3a's interface is stable.
- **Plan 3c:** weekly→season aggregation (Monte Carlo with bye + availability) + walk-forward backtest harness with CI threshold gating.

---

## 1. Overview

Plan 3a delivers the first trained projection model in the codebase. It is deliberately scoped to a single position so that the `Model` interface, the per-stat-regression-to-fantasy-points pipeline, and joblib persistence are validated end-to-end on real data before the pattern is replicated to QB / RB / TE in 3b.

This is also the first time `ingest.refresh(...)` runs against real `nfl_data_py` data. Every test in the repo today runs against synthetic in-memory fixtures (per the 2a/2b decision log entry). 3a forces us to confront any `nfl_data_py` API drift on a real pull.

### 1.1 Goals

- First real-data ingest run: `data/raw/` populated for seasons 2018–2025.
- New `src/projections/models/` package: `Model` Protocol + `BaselineModel` implementation + `wr_baseline()` factory.
- Trained `WrBaselineModel` artifact persisted via joblib, keyed by a `model_id` derived from the source files that affect the artifact.
- WR weekly projections for every week of 2025 written to `data/projections/weekly/season=2025/week=WW/ruleset=ESPN_PPR/part.parquet`, validated against the existing `ProjectionWeeklySchema`.
- Sanity-check evaluation on the 2025 held-out season — informational stdout report only, not a CI gate.

### 1.2 Non-goals (deferred to other plans)

- QB / RB / TE models — Plan 3b.
- Season aggregation, availability model, bye-week handling — Plan 3c.
- Walk-forward backtest harness with CI threshold gating — Plan 3c.
- Public Python API + CLI verbs — Plan 4.
- K / DST models — TODO #10 (data-dependent).
- Joint correlations between players' outcomes — TODO #1.
- Feature parquet caching (`data/features/wr/...`) — deferred unless WR feature rebuild from raw is slow enough during 3a development to motivate it. If kept pure-function, the foundations spec's storage layout adds the directory in 3c when backtest performance forces our hand.

---

## 2. Architecture

### 2.1 New package layout

```
src/projections/models/
├── __init__.py
├── base.py        # Model Protocol + model_id construction helper
└── baseline.py    # BaselineModel implementation + wr_baseline() factory

models/artifacts/  # gitignored; joblib files live here at runtime
```

Plan 3b extends `baseline.py` with `qb_baseline()`, `rb_baseline()`, `te_baseline()` factories. Plan 3c adds sibling packages `aggregate/` and `backtest/`.

### 2.2 `Model` Protocol (pinned in this plan)

```python
from typing import Protocol
from pathlib import Path
import pandas as pd
from projections.schemas import Position, Ruleset

class Model(Protocol):
    """Position-specific projection model. Plugs in at the fit/predict seam."""

    @property
    def position(self) -> Position: ...

    @property
    def model_id(self) -> str: ...
    # Format: "baseline:wr:<8-char-hash>:<train-start>-<train-end>"

    def fit(
        self,
        features: pd.DataFrame,         # validated against {Position}FeaturesSchema
        weekly_stats: pd.DataFrame,     # validated against WeeklyStatsSchema
    ) -> None: ...
    # Inner-joins (gsis_id, season, week) to align features with truth.

    def predict_distribution(
        self,
        features: pd.DataFrame,         # validated against {Position}FeaturesSchema
        ruleset: Ruleset,
    ) -> pd.DataFrame: ...
    # Returns ProjectionWeeklySchema-validated rows.

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "Model": ...
```

`Model` is a `Protocol`, not an `ABC`. Structural typing is sufficient — there are no `isinstance(x, Model)` checks anywhere; only mypy enforces the contract at use sites. Same posture as the existing `Distribution` Protocol next door.

Re-scoring under a different ruleset is a second `predict_distribution` call with a different `ruleset` argument. No retraining; no separate per-ruleset artifacts.

### 2.3 `BaselineModel` (the implementation)

One parameterizable class. Per-position factory functions construct correctly-configured instances.

**Configuration (frozen at construction):**
- `position: Position`
- `target_stats: tuple[Stat, ...]` — the underlying stats to fit per-stat regressions for. For WR: `(receptions, receiving_yards, receiving_tds, rushing_yards, rushing_tds, fumbles_lost)`.
- `feature_columns: tuple[str, ...]` — feature column names from the position's features schema, minus identity columns.
- `dist_families: dict[Stat, DistributionFamily]` — see §3.3.

**Persisted state (populated by `fit`):**
- `feature_means: pd.Series` — per-column medians used for predict-time NaN imputation.
- `ridges: dict[Stat, RidgeCV]` — one fitted regression per target stat.
- `variance_params: dict[Stat, dict[str, float]]` — per-stat residual-variance parameters. Concrete keys per family in §3.4.
- `train_seasons: tuple[int, int]` — `(min_season, max_season)` of the data used.
- `code_hash: str` — see §5.

### 2.4 `wr_baseline()` factory

```python
def wr_baseline() -> BaselineModel:
    """WR-specific BaselineModel construction; configures target stats,
    feature columns, and distribution families. Plan 3b adds qb_/rb_/te_
    siblings using the same BaselineModel class."""
```

Returns an unfitted `BaselineModel`. Caller calls `.fit(features, weekly_stats)` and `.save(path)`.

---

## 3. Data → projection pipeline

### 3.1 Training-time (`fit`)

1. Validate `features` against `WrFeaturesSchema`; validate `weekly_stats` against `WeeklyStatsSchema`.
2. Inner-join on `(gsis_id, season, week)` so every training row has both feature inputs and ground-truth stats. Players in the depth chart who didn't actually play that week are dropped (no truth available).
3. Filter to `position == WR` on the weekly_stats side as a defense-in-depth guard.
4. Build feature matrix `X` from the configured `feature_columns`; coerce booleans to 0/1; persist column order.
5. Compute per-column medians on `X`; persist them in `feature_means` for predict-time imputation. (Training-time NaN handling: drop rows with NaN in any feature column — they are mostly week-1-of-2018 rows where rolling features have no prior history; we're throwing away ~1 season's worth of week-1 data, acceptable for a baseline.)
6. For each target stat in `target_stats`: fit `RidgeCV(alphas=np.logspace(-3, 3, 13))` on `(X, y_stat)` with built-in generalized cross-validation. Store the fitted regressor.
7. For each target stat: compute residuals `y_stat - μ̂_stat` on the training set; derive variance parameters per the family (§3.4).
8. Compute `code_hash` (§5); record `train_seasons`.

### 3.2 Predict-time (`predict_distribution(features, ruleset)`)

1. Validate `features` against `WrFeaturesSchema`.
2. Build `X` with the persisted column order; impute NaNs with `feature_means`; coerce booleans to 0/1.
3. For each target stat: `μ̂_stat = ridges[stat].predict(X)`. For gamma stats, clamp to `≥ 1e-3` so the rate parameter is well-defined.
4. Per stat, build a parametric `Distribution`:
   - **normal** stats: `Distribution(family=NORMAL, params={"mean": μ̂, "std": σ_stat})`
   - **gamma** stats: `Distribution(family=GAMMA, params={"shape": α_stat, "rate": α_stat / μ̂})`
5. Pass the per-row `dict[Stat, Distribution]` to `scoring.score_distribution(stat_dists, ruleset)`. Output is a fantasy-points `Distribution` per player-week.
6. Extract `mean`, `p10`, `p50`, `p90` from each points `Distribution`. Serialize `(family, params)` to msgpack bytes.
7. Assemble identity columns (`gsis_id`, `season`, `week`, `position=WR`, `team`, `opponent`) from `features`, plus `ruleset=ruleset.name`, `model_id`, `generated_at=now(UTC)`.
8. Validate against `ProjectionWeeklySchema`; return.

### 3.3 Distribution family per WR-scored stat

| Stat | Family | Rationale |
|---|---|---|
| `receptions` | gamma | non-negative count |
| `receiving_yards` | normal | net yardage (rare negatives on TFLs) |
| `receiving_tds` | gamma | non-negative count, rare event |
| `rushing_yards` | normal | net yardage |
| `rushing_tds` | gamma | non-negative count, very rare |
| `fumbles_lost` | gamma | non-negative count, very rare |

Targets (the stat) are *not* in `target_stats` — they are an input to scoring only via the receiving stats they translate into, and PPR doesn't directly score them.

### 3.4 Variance estimation

- **Normal stats:** `σ_stat = std(y - μ̂)` over the entire training set. One global standard deviation per stat. Homoscedastic — same `σ` at every prediction. Stored as `variance_params[stat] = {"std": σ_stat}`.
- **Gamma stats:** shape parameter `α_stat` fit globally via method of moments — `α̂ = mean(μ̂)² / var(y - μ̂)`. Clipped to `[0.01, 100.0]` for numerical safety. Stored as `variance_params[stat] = {"shape": α_stat}`. At predict time, the rate `β = α_stat / μ̂` makes the variance `μ̂² / α_stat` — heteroscedastic by construction at no extra parameter cost.

For very rare events (TDs, fumbles) where method of moments may produce degenerate `α`, the clip range is the safety net. If 3a's sanity check shows TD calibration is bad, MLE via `scipy.optimize.minimize` is a one-line follow-up; not in scope here.

### 3.5 Dependency on `scoring.score_distribution`

`src/projections/scoring/score_distribution.py` exists from Plan 1 (foundations) but its current signature has not been re-read in this brainstorm. The implementation plan's first task verifies it accepts `(stat_dists: dict[Stat, Distribution], ruleset: Ruleset) -> Distribution` and extends it in-place if not. Per `CLAUDE.md`'s rule "scoring is the only place that knows what counts as a fantasy point," any extension lands inside `scoring/`, never in the model.

---

## 4. Data scope

### 4.1 Training and held-out seasons

- **Training:** 2018–2024 (7 seasons).
- **Held-out (sanity check only in 3a):** 2025 (full season — 2025 NFL season ran Sept 2025 – Feb 2026, so it is complete as of 2026-04-25).

NGS receiving has full coverage from 2018 onward; older seasons would force NGS-NaN imputation that adds modeling complexity disproportionate to a baseline. Drawing the line at 2018 keeps the feature stack clean.

The choice of 2025 as 3a's held-out year doesn't lock anything in: Plan 3c's walk-forward backtest harness will iterate over the full 2018+ window, holding out one season at a time.

### 4.2 First real-data ingest

`ingest.refresh(seasons=range(2018, 2026))` writes raw partitions for every ingest source (`weekly_stats`, `schedules`, `snap_counts`, `depth_charts`, `ngs_passing`, `ngs_rushing`, `ngs_receiving`) plus the `id_map.parquet`. The implementation plan stages this as:

1. Run `ingest.refresh(seasons=[2018])` first as a smoke check. Any `nfl_data_py` API drift since the Plan 2 fixtures were authored surfaces here, in isolation, on a single season's worth of data.
2. After 2018 succeeds, run the full 2018–2025 refresh.

Expected size: small — full 2018–2025 raw parquet should be under ~100 MB. Manifest at `data/manifests/ingest_manifest.parquet` records what was fetched.

### 4.3 Training data shape

After inner-joining WR features with weekly_stats and dropping NaN-feature rows: ≈ 9 000 WR-week rows over the 7-season training window. Comfortably within memory; sklearn `RidgeCV` fits in milliseconds at this scale.

---

## 5. Persistence and `model_id`

### 5.1 Artifact path

```
models/artifacts/wr-baseline-{train_start}-{train_end}-{code_hash}.joblib
```

Example: `models/artifacts/wr-baseline-2018-2024-a1b2c3d4.joblib`.

`models/artifacts/` is gitignored (large binary blobs; reproducible from source + ingested data).

### 5.2 `model_id` construction

```
baseline:<position>:<8-char-code-hash>:<train-start>-<train-end>
```

Example: `baseline:wr:a1b2c3d4:2018-2024`.

The `code_hash` is the first 8 characters of the SHA-256 hash of the concatenated source content of (in deterministic, sorted order):
- `src/projections/models/base.py`
- `src/projections/models/baseline.py`
- `src/projections/features/wr.py`
- `src/projections/features/_shared.py`
- `src/projections/features/_rolling.py`
- `src/projections/features/_opponent.py`
- `src/projections/scoring/score.py`
- `src/projections/scoring/score_distribution.py`

Anything whose change ought to invalidate the artifact is in this list — model code, the WR feature builder and its shared helpers, and both scoring modules (because a change to `score.py`'s rule application would propagate through `score_distribution`'s composition into the points distribution). The hash is computed inside `BaselineModel.fit()` and stored on the instance; it changes deterministically when any of these files change. The implementation plan finalizes the exact file list with a unit test that asserts the hash changes when any tracked file is mutated.

This `model_id` is written into every `ProjectionWeeklySchema` row produced by `predict_distribution`, so we can always trace which model generated which projection. Plan 3c's backtest harness keys results by `model_id` so reruns are idempotent.

### 5.3 What `joblib` actually serializes

The whole `BaselineModel` instance (including all `RidgeCV` regressors, the `feature_means`, the `variance_params`, the `target_stats`/`feature_columns`/`dist_families` config, `train_seasons`, and `code_hash`). `joblib` is the de-facto sklearn serialization mechanism — handles numpy arrays efficiently and works out of the box with sklearn estimators.

---

## 6. Sanity-check evaluation on 2025

Run after `fit` completes; **not a CI gate** in 3a (Plan 3c formalizes thresholds). Output is printed to stdout.

### 6.1 Procedure

For each week W of 2025:
1. Build WR features with `as_of_week=W` from data through W−1 (`features.build_wr_features(...)`).
2. Predict for W: `model.predict_distribution(features=..., ruleset=Ruleset.espn_ppr())`.
3. Join predictions with the actual 2025 `weekly_stats` for that week on `(gsis_id, season=2025, week=W)`.

Concatenate across all 18 weeks → one DataFrame keyed by `(gsis_id, week)` with `(predicted_mean, predicted_p10, predicted_p50, predicted_p90, actual_points)` plus per-stat columns.

### 6.2 Metrics reported

Three families of numbers, one print block each:

**Per-stat fit** — for each scored stat, compute and print:
- RMSE, MAE
- Mean predicted vs. mean actual (catches systematic bias)

**Composite (PPR fantasy points)**:
- RMSE / MAE on realized PPR points vs. predicted mean
- Spearman top-30 rank correlation (predicted mean ranking vs. realized ranking, restricted to WRs who actually played in the held-out year)

**Calibration spot-check** on the points distribution:
- Fraction of actual realizations falling inside `[predicted_p10, predicted_p90]` — target ≈ 80%
- Fraction `≤ predicted_p90` — target ≈ 90%

### 6.3 Soft thresholds (informational)

These are sanity guideposts, not gates. If 3a's run misses them, that's a debug signal logged to project_management.md, not a blocked merge:

- Spearman top-30 correlation ≥ 0.4
- Calibration `[p10, p90]` coverage in 70–90% range
- Per-stat RMSE within 2× of the naive-baseline RMSE (predict the rolling-4-week mean)

Plan 3a still ships if it misses these — the deliverable is the pipeline, not a great model. Bad numbers feed into Plan 3c's threshold-setting and Model C planning.

---

## 7. Testing strategy

Following the 2a/2b precedent: synthetic fixtures + leakage tests + smoke-test extension + schema validation at every boundary.

### 7.1 Unit tests

`tests/test_models/test_baseline.py`:
- **fit-then-predict round trip** on a tiny synthetic fixture (extends the 2a fixtures), asserts `ProjectionWeeklySchema.validate(out)` passes.
- **save / load round trip** — predictions must be byte-identical pre and post.
- **`model_id` determinism** — hashing the same source files twice yields the same hash; modifying a tracked file changes the hash.
- **gamma α from method-of-moments matches a hand-computed value** on a synthetic residual array.
- **gamma μ̂ clamping** — when a regression predicts μ̂ ≤ 0, the gamma rate doesn't blow up.
- **NaN imputation** at predict time uses `feature_means` from fit.
- **empty input** — empty features → empty schema-valid output (mirrors 2b's empty-depth-chart fix; `coerce=True` already on `ProjectionWeeklySchema`? — verify in implementation plan, add if missing).

### 7.2 Leakage test

`tests/test_models/test_baseline_leakage.py`:
- Build features with `as_of_week=W`, fit Model A.
- Mutate week-W+1 rows in the source data, re-run feature build with same `as_of_week=W`, fit again.
- Assert: every fitted regressor's coefficients are identical.

This catches any accidental future-leakage either in the model's training-time feature handling or in the feature builder being asked for inputs at the wrong week.

### 7.3 Smoke-test extension

Extend `tests/test_smoke.py` (or add a sibling file) so the existing ingest→features chain feeds into a `wr_baseline().fit(...)` call → `predict_distribution(...)` → `store.write_partition(...)` round trip on synthetic data. Asserts a `ProjectionWeeklySchema`-valid parquet is written.

### 7.4 Network test (opt-in)

A `@pytest.mark.network` test under `tests/test_models/test_baseline_real_data.py` that runs `wr_baseline().fit(...)` against actual ingested 2018 data (one season, smallest viable) and asserts it produces non-degenerate predictions. Skipped by default; useful for `nfl_data_py` API-drift catches alongside TODO #8.

---

## 8. Edge cases handled in 3a

- **Rookies / no-prior-history players.** Their feature row has all-zero rolling means by construction (the 2a feature builder zero-fills `_ROLLING_ZERO_FILL_COLS`). The model just predicts a low-mean distribution. No special-case code needed.
- **Gamma μ̂ ≤ 0** at predict time. Clamp to `1e-3` so the rate parameter `β = α / μ̂` is finite. Logged at debug level when triggered.
- **NaN in NGS-derived features** for early-career or never-qualified WRs. Median-imputed at predict time using `feature_means` persisted from training.
- **Predicted negative `receiving_yards`.** Legal under the normal family; `score_distribution` consumes the negative tail correctly (a rare WR has negative-yards games).
- **Empty input frame.** Caller may pass an empty WR feature frame (e.g., a quirky week with no rostered WRs). Model returns an empty `ProjectionWeeklySchema`-valid frame, mirroring the 2b empty-input fix.

---

## 9. Risks / open questions for the implementation plan

1. **`scoring.score_distribution` API verification.** First task of the implementation plan: read the current signature, decide if it already supports `(dict[Stat, Distribution], Ruleset) -> Distribution`. Extend if not, in `scoring/`.
2. **`nfl_data_py` API drift on first real-data pull.** Mitigated by staging 2018 alone before 2018–2025, but still a risk. The implementation plan reserves a debug task in case ingest fails on real data.
3. **Gamma `α` instability for very rare events.** Method-of-moments is closed-form but can produce degenerate shape parameters when the mean is near zero (rushing TDs, fumbles). The clip range `[0.01, 100]` is the v1 safety net. If sanity-check calibration is bad on TDs/fumbles, MLE via `scipy.optimize` is a backlog item, not 3a scope.
4. **Held-out 2025 has no precedent in the repo.** Every prior test is on synthetic data. Plan 3a is also where we discover that the first real-data run is, in fact, what it claims to be. The implementation plan calls out a checkpoint between "ingest succeeds" and "training proceeds" so we can eyeball one real WR feature row before depending on it.
5. **Boolean features in Ridge.** `is_home`, `roof_dome`, `designed_rusher` cast to 0/1. Ridge handles them but loses any nonlinear interaction (e.g., dome × implied_team_total). That's exactly what Model C (LightGBM) is for; not a 3a concern.

---

## 10. Decisions captured during this brainstorm

| Decision | Rationale |
|---|---|
| Split Plan 3 into 3a (WR end-to-end) / 3b (QB/RB/TE generalization) / 3c (aggregation + backtest) | Same validation-on-one-position-first pattern that paid off in 2a→2b |
| Per-stat independent Ridge sub-models (architecture A) | Closest match to the foundations spec wording; per-stat residuals are debuggable; assumes per-stat independence within a game, which is "option D" / TODO #1 territory |
| 2018–2024 train, 2025 held-out for sanity check | NGS receiving full coverage from 2018; 2025 is complete by 2026-04-25; 3c's walk-forward backtest will revisit the held-out choice |
| `Model` as `Protocol` not `ABC` | Structural typing is sufficient; no runtime `isinstance` checks; consistent with existing `Distribution` Protocol |
| One `BaselineModel` class with per-position factories (`wr_baseline()`, future `qb_baseline()` etc.) | Minimizes 3a→3b copy; per-position quirks expressible as config; refactor toward subclasses only if a position needs custom training logic |
| Gamma for non-negative volume / count stats; normal for net yardage | Foundations spec wording; `DistributionFamily` enum already has both |
| Heteroscedastic-by-construction gamma (one shape `α` from MoM) + global `σ` for normal stats | Simplest baseline that matches the spec's "fit positional residual variance from training residuals"; one parameter per stat to tune later |
| Method of moments for gamma `α`, with clip to `[0.01, 100]` | Closed-form; MLE via `scipy.optimize` is a follow-up if calibration is bad |
| Model artifact via joblib at `models/artifacts/wr-baseline-{train_start}-{train_end}-{code_hash}.joblib` | Standard sklearn pattern; gitignored; reproducible from source + data |
| `model_id = "baseline:<pos>:<8-char-code-hash>:<train-start>-<train-end>"` | Unambiguous, stable, written into every projection row for traceability |
| `code_hash` covers `models/`, `features/wr.py`, `_shared.py`, `_rolling.py`, `_opponent.py`, `score_distribution.py` | Anything whose change ought to invalidate the artifact |
| Sanity-check eval is informational only in 3a; thresholds in 3c | Plan 3a's deliverable is the pipeline; bad numbers feed into 3c, not a blocker |
| Stage `ingest.refresh(seasons=[2018])` before the full 2018–2025 pull | Isolate any `nfl_data_py` API drift on a single season |
| Feature parquet caching deferred unless training is slow | Foundations spec already gates this on backtest performance (TODO #4); 3a stays pure-function unless forced |

---

## 11. What an MVP delivers (steps in order)

1. Verify `scoring.score_distribution` signature handles `(dict[Stat, Distribution], Ruleset) -> Distribution`. Extend in-place if needed (in `scoring/`, with unit tests).
2. Add `models/` package skeleton: `base.py` with the `Model` Protocol; `__init__.py` re-exporting it; gitignore `models/artifacts/`.
3. Implement `BaselineModel` in `baseline.py` (config + persisted state + `fit` + `predict_distribution` + `save` + `load` + `model_id` derivation).
4. Add `wr_baseline()` factory configuring WR's `target_stats`, `feature_columns`, `dist_families`.
5. Run `ingest.refresh(seasons=[2018])` as the API-drift smoke check; debug any column shape issues against the existing schemas.
6. Run the full `ingest.refresh(seasons=range(2018, 2026))`.
7. Write a small training script (kept in `scripts/` or as a `pytest -m manual` test) that builds WR features for 2018–2024, calls `wr_baseline().fit(...)`, and saves the artifact.
8. Write the sanity-check eval routine (`scripts/sanity_check_wr_baseline.py` or similar). Walks 2025 week-by-week, prints the §6.2 metric block, optionally writes a diagnostic CSV to `data/sanity_checks/`.
9. Write 2025 weekly projections to `data/projections/weekly/season=2025/week=WW/ruleset=ESPN_PPR/part.parquet` via `store.write_partition`.
10. Unit tests + leakage test + smoke-test extension per §7.
11. Update `project_management.md` (close 3a, queue 3b) and `TODO.md` (any items discovered along the way).

Anything beyond this — QB/RB/TE models, season aggregation, backtest harness — is Plan 3b/3c.
