# Plan 3b — QB / RB / TE Model A baselines (generalize 3a) — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-25
**Author:** alden + claude
**Builds on:** `2026-04-25-plan-3a-wr-model-a-design.md` (the architecture this plan generalizes); `2026-04-24-plan-2b-qb-rb-te-features-design.md` (the per-position feature builders this plan trains against).
**Plan-3 series context:**
- **Plan 3a (merged):** Model A on WR only, end-to-end. Pinned the `Model` Protocol, the per-stat-regression-to-fantasy-points pipeline, joblib persistence, and the first real-data ingest.
- **Plan 3b (this design):** Generalize Model A to QB / RB / TE. Extend the existing `BaselineModel` with two constructor args, add three factories, add a position-dispatch registry, generalize the three CLI scripts. No new packages.
- **Plan 3c (next):** Weekly→season aggregation (Monte Carlo with bye + availability) + walk-forward backtest harness with CI threshold gating.

---

## 1. Overview

3a delivered the first trained projection model on a single position so that the architecture could be hardened end-to-end on real data. 3b applies the same pattern to QB / RB / TE. The work is mostly mechanical because 3a anticipated this generalization in the dataclass shape — `BaselineModel` already accepts `(position, target_stats, feature_columns, dist_families)`. 3b removes the two remaining position-specific hardcodes inside `BaselineModel` (the schema validation reference, and the code-hash file list), adds three factory functions, generalizes the three CLI scripts to dispatch by `--position`, and trains + evaluates each position on real 2018-2023 data.

### 1.1 Goals

- Three new factory functions in `src/projections/models/baseline.py`: `qb_baseline()`, `rb_baseline()`, `te_baseline()`.
- `BaselineModel` extended with two **required** constructor args: `feature_schema: type[pa.DataFrameModel]` and `code_hash_files: tuple[Path, ...]`. The hardcoded `WrFeaturesSchema.validate(...)` and the hardcoded code-hash file list inside `fit()` are removed; both come from the constructor.
- New `POSITION_DISPATCH: Mapping[Position, _PositionDispatch]` registry in `src/projections/models/__init__.py`. Tells callers (the CLI scripts, future Plan 3c backtest harness) which factory + feature builder + feature schema + NGS source to use per position.
- Three CLI scripts generalized to take `--position {qb|rb|te|wr}`:
  - `scripts/train_baseline.py` (replaces `train_wr_baseline.py`).
  - `scripts/predict_2024.py` (replaces `predict_2024_wr.py`).
  - `scripts/sanity_check_baseline.py` (replaces `sanity_check_wr_baseline.py`).
  The three WR-specific scripts are deleted; `--position wr` produces identical behavior.
- `TeFeaturesSchema` extended with `rushing_attempts_per_game_l4` + `rushing_yards_per_game_l4`; `build_te_features` populates them. Required because Q3 set TE rushing as a baseline target stat (Taysom Hill rationale).
- Six new test files under `tests/test_models/`: `test_baseline_qb.py` / `_rb.py` / `_te.py` and `test_baseline_qb_leakage.py` / `_rb_leakage.py` / `_te_leakage.py`. The smoke test (`tests/test_smoke.py`) is parametrized across all four positions.
- Per-position 2024 sanity-check eval recorded in `project_management.md`. Same template as 3a's (per-stat RMSE/MAE, composite, top-N rank corr, calibration coverage). **Stdout-only — no CI gate added in 3b.**
- Per-position trained artifacts at `models/artifacts/baseline-{pos}-2018-2023-{hash}.joblib` (gitignored).
- 2024 weekly projections written to `data/projections/weekly/ruleset=ESPN_PPR/season=2024/week=WW/part.parquet` for every position.

### 1.2 Non-goals (deferred)

- Walk-forward backtest harness with CI threshold gating → **Plan 3c**.
- Season aggregation, availability model, bye-week handling → **Plan 3c**.
- POISSON distribution family for low-mean integer stats (interceptions, fumbles_lost) → **Plan 3c**, contingent on calibration evidence.
- K / DST models → **TODO #10** (data-dependent).
- Joint correlations between players' outcomes → **TODO #1**.
- Public Python API + CLI verbs (`python -m projections refresh|project|...`) → **Plan 4**.

---

## 2. Architecture

### 2.1 No new packages

All work lands in three places. The package layout from 3a is unchanged:

```
src/projections/models/
├── __init__.py            # Q5 adds POSITION_DISPATCH registry here
├── base.py                # unchanged (Model Protocol + compute_code_hash helper)
└── baseline.py            # extended with three new factories + dataclass changes
```

### 2.2 `BaselineModel` constructor changes

The existing dataclass:

```python
@dataclass
class BaselineModel:
    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    dist_families: Mapping[Stat, DistributionFamily]
    # ... fit-populated state
```

3b adds two required fields immediately after `dist_families`:

```python
    feature_schema: type[pa.DataFrameModel]   # Q2 — replaces hardcoded WrFeaturesSchema in fit/predict
    code_hash_files: tuple[Path, ...]          # 5b — replaces hardcoded file list in fit
```

Both are required (no defaults). The existing 3a-trained WR artifact (`models/artifacts/wr-baseline-2018-2023-925f492b.joblib`, gitignored) becomes unloadable after this change. Mitigation: the load path raises a clear `TypeError` from joblib's pickle reconstruction; the user retrains via `scripts/train_baseline.py wr`. Tracked as a TODO entry written to `TODO.md` as part of this plan: **"retrain WR 2018-2023 artifact after 3b lands."** Phase 6 closes this TODO.

Inside `fit()`:

- Replace `features = WrFeaturesSchema.validate(features)` with `features = self.feature_schema.validate(features)`.
- Replace the hardcoded `tracked = [...]` file list with `self.code_hash = compute_code_hash(self.code_hash_files)`.

Inside `predict_distribution()`:

- Replace `features = WrFeaturesSchema.validate(features)` with `features = self.feature_schema.validate(features)`.

No other behavior changes. The `_GAMMA_MU_FLOOR`, `_GAMMA_ALPHA_CLIP`, residual-variance estimators, model_id format, save/load round-trip — all preserved exactly.

### 2.3 `POSITION_DISPATCH` registry

```python
# src/projections/models/__init__.py

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa

from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.te import build_te_features
from projections.features.wr import build_wr_features
from projections.ingest.ngs import NgsStatType
from projections.models.baseline import (
    BaselineModel,
    qb_baseline,
    rb_baseline,
    te_baseline,
    wr_baseline,
)
from projections.schemas import (
    Position,
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)


@dataclass(frozen=True)
class _PositionDispatch:
    factory: Callable[[], BaselineModel]
    feature_builder: Callable[..., pd.DataFrame]
    feature_schema: type[pa.DataFrameModel]
    ngs_stat_type: NgsStatType


POSITION_DISPATCH: Mapping[Position, _PositionDispatch] = {
    Position.QB: _PositionDispatch(qb_baseline, build_qb_features, QbFeaturesSchema, "passing"),
    Position.RB: _PositionDispatch(rb_baseline, build_rb_features, RbFeaturesSchema, "rushing"),
    Position.TE: _PositionDispatch(te_baseline, build_te_features, TeFeaturesSchema, "receiving"),
    Position.WR: _PositionDispatch(wr_baseline, build_wr_features, WrFeaturesSchema, "receiving"),
}
```

Adding a fifth position becomes one new line in this registry plus the corresponding factory + feature builder. The registry is the canonical "what positions does the system know about" answer; future code (Plan 3c's backtest harness, Plan 4's CLI verbs) reads from it.

### 2.4 Refactor sequencing (phases)

Each phase below is one commit on the feature branch. Phases are sequenced so each leaves the test suite green before the next starts. The plan doc (next step) breaks each phase into TDD-shaped tasks.

1. **Phase 1 — TE feature schema extension.** Extend `TeFeaturesSchema` and `build_te_features` with `rushing_attempts_per_game_l4` + `rushing_yards_per_game_l4`. Update `tests/test_features/test_te.py` (new fixture rows; new schema assertions). `test_te_leakage.py` should pass without changes (verify). Tests pass.
2. **Phase 2 — `BaselineModel` constructor change.** Add `feature_schema` + `code_hash_files` as required fields. Update `wr_baseline()` to pass them. Replace the two hardcoded sites inside `fit()` and one inside `predict_distribution()`. Existing `test_baseline.py` and `test_baseline_leakage.py` pass; existing 3a artifact on disk is now unloadable (TODO entry written to `TODO.md`).
3. **Phase 3 — three new factories + registry.** Add `qb_baseline()` / `rb_baseline()` / `te_baseline()` to `baseline.py` (each defines its own `_TARGET_STATS`, `_DIST_FAMILIES`, `_FEATURE_COLUMNS`, `_CODE_HASH_FILES`). Add `_PositionDispatch` + `POSITION_DISPATCH` to `models/__init__.py`. No new tests in this phase — Phase 4 covers them.
4. **Phase 4 — per-position model tests + smoke parametrization.** Six new test files under `tests/test_models/`. Per-position fixtures live in `tests/test_models/conftest.py`. Smoke test parametrized across all four positions; the existing root-conftest fixtures get renamed `baseline_features` → `baseline_features_wr` (and `baseline_weekly_stats` → `baseline_weekly_stats_wr`).
5. **Phase 5 — script generalization.** Three new scripts (`train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`); three old scripts deleted. Test by running each on the 2018-2023 / 2024 data already in `data/raw/`. Manual smoke; no automated test (these are operator scripts, not library code).
6. **Phase 6 — real-data train + sanity-check.** For each of QB / RB / TE / WR (the WR retrain closes the artifact-break TODO from phase 2): run `train_baseline.py {pos}` then `sanity_check_baseline.py {pos}`; record the eval output in `project_management.md`. Run `predict_2024.py {pos}` to write 2024 partitions.
7. **Phase 7 — `project_management.md` + `TODO.md` finalization.** Decision-log entries; "next action: Plan 3c"; the artifact-retrain TODO written in phase 2 is now closed (just delete it since the retrain happened in phase 6).

---

## 3. Per-position configuration

### 3.1 Target stats

Locked in brainstorming Q3.

| Position | target_stats |
|---|---|
| QB | `passing_yards`, `passing_tds`, `interceptions`, `rushing_yards`, `rushing_tds`, `fumbles_lost` |
| RB | `rushing_yards`, `rushing_tds`, `receptions`, `receiving_yards`, `receiving_tds`, `fumbles_lost` |
| TE | `receptions`, `receiving_yards`, `receiving_tds`, `rushing_yards`, `rushing_tds`, `fumbles_lost` |
| WR (existing) | `receptions`, `receiving_yards`, `receiving_tds`, `rushing_yards`, `rushing_tds`, `fumbles_lost` |

Deliberately omitted across all positions: `passing_2pt_conversions`, `rushing_2pt_conversions`, `receiving_2pt_conversions`, `return_tds`. Same omission as 3a's WR set: extreme-low-base-rate noise — Ridge would shrink to ~0 and the residual-variance estimator would be dominated by sampling noise rather than signal.

### 3.2 Distribution families

Locked in brainstorming Q4. **NORMAL for yards, GAMMA for counts**, mechanically applied. No position-specific deviations.

| Stat | Family |
|---|---|
| passing_yards | NORMAL |
| passing_tds | GAMMA |
| interceptions | GAMMA |
| rushing_yards | NORMAL |
| rushing_tds | GAMMA |
| receptions | GAMMA |
| receiving_yards | NORMAL |
| receiving_tds | GAMMA |
| fumbles_lost | GAMMA |

Caveats accepted by design:

- **3a's WR calibration was 70.8% in `[p10, p90]`** (target ~80%). Carrying the same family choice forward likely repeats this in 3b. Plan 3c's backtest harness owns calibration tuning (MLE-fit gamma α, residual-variance bucketing, or a SAMPLED_SUMMARY family) — **not 3b**.
- **Interceptions are very low-mean integer counts** (~0.7/game). GAMMA's continuous-positive support is a poor model for `{0, 1, 2, 3}` outcomes; a Poisson would be more honest. Adding `DistributionFamily.POISSON` + `ParametricPoisson` is real code (new distribution class, scoring composition update). **Defer to Plan 3c**, contingent on calibration evidence.

### 3.3 Feature columns per position

Each position's feature columns are exactly the schema columns minus identity columns (`gsis_id`, `season`, `week`, `team`, `opponent`). The factory functions encode the exact tuple in source.

| Position | Feature columns (count is approximate; exact set comes from the schema) |
|---|---|
| QB (~22) | `depth_rank`, `passing_yards_per_game_l4`, `passing_tds_per_game_l4`, `interceptions_per_game_l4`, `attempts_per_game_l4`, `completions_per_game_l4`, `sacks_per_game_l4`, `rushing_attempts_per_game_l4`, `rushing_yards_per_game_l4`, `rushing_qb`, `snap_pct_l4`, `avg_time_to_throw_std`, `avg_completed_air_yards_std`, `avg_intended_air_yards_std`, `completion_percentage_above_expectation_std`, `implied_team_total`, `spread`, `is_home`, `roof_dome`, `opp_allowed_qb_fppg_l4` |
| RB (~18) | `depth_rank`, `carries_per_game_l4`, `rushing_yards_per_game_l4`, `rushing_tds_per_game_l4`, `rush_share_l4`, `targets_per_game_l4`, `receptions_per_game_l4`, `receiving_yards_per_game_l4`, `target_share_l4`, `targets_per_game_std`, `passing_down_back`, `snap_pct_l4`, `efficiency_std`, `rush_yards_over_expected_per_att_std`, `percent_attempts_gte_eight_defenders_std`, `implied_team_total`, `spread`, `is_home`, `roof_dome`, `opp_allowed_rb_fppg_l4` |
| TE (~17, after Phase 1 schema extension) | `depth_rank`, `targets_per_game_l4`, `targets_per_game_std`, `target_share_l4`, `receptions_per_game_l4`, `receiving_yards_per_game_l4`, `receiving_tds_per_game_l4`, `rushing_attempts_per_game_l4`, `rushing_yards_per_game_l4`, `snap_pct_l4`, `avg_separation_std`, `avg_intended_air_yards_std`, `avg_yac_above_expectation_std`, `implied_team_total`, `spread`, `is_home`, `roof_dome`, `opp_allowed_te_fppg_l4` |

Source of truth: each position's `*FeaturesSchema.to_schema().columns.keys()` minus the identity tuple. The factory function encodes the column list literally; if a future schema column is added, the factory must be updated.

### 3.4 Code-hash file list per factory

3a's hardcoded list inside `fit()` becomes per-factory. Each factory's `_CODE_HASH_FILES` includes the position-specific feature module **plus** the shared modules whose change should invalidate any artifact:

```
models/base.py
models/baseline.py
features/_shared.py
features/_rolling.py
features/_opponent.py
features/{position}.py
scoring/score.py
scoring/score_distribution.py
```

Eight files. Same shape as 3a's hardcoded list, just with the position-specific feature file swapped in. The list is **identical to 3a's** for the WR factory (verified against `baseline.py` in the merged 3a code). The WR retrain in Phase 6 still produces a new `model_id` regardless, because `baseline.py` itself is part of the hashed list and 3b modifies it (new fields, new factories). Acceptable: any consumer that joins on `model_id` is regenerating projections anyway.

---

## 4. Testing pattern

### 4.1 Six new files

Per brainstorming Q6 (per-position files mirror the precedent in `tests/test_features/`):

- `tests/test_models/test_baseline_qb.py` — fixture-based unit tests on `qb_baseline()`. Mirrors `test_baseline.py`.
- `tests/test_models/test_baseline_rb.py`, `test_baseline_te.py` — same shape, position-specific fixtures.
- `tests/test_models/test_baseline_qb_leakage.py`, `_rb_leakage.py`, `_te_leakage.py` — assert `fit` doesn't peek at future weeks.

About 70 lines per leakage file × 3 = ~210 lines of necessary duplication. Per-position fixtures make abstracting harder than copy-paste; the cost is acceptable.

### 4.2 Fixture organization

The existing root-conftest `baseline_features` / `baseline_weekly_stats` fixtures (3a-promoted because the smoke test consumed them) get renamed to `baseline_features_wr` / `baseline_weekly_stats_wr`. Three sibling fixtures per non-WR position (`baseline_features_qb` / `_rb` / `_te` and the corresponding `baseline_weekly_stats_*`) are also added at root scope, because 4.3's parametrized smoke test consumes them — pytest fixtures only inherit downward, so anything reached by the top-level smoke must live at root.

A small `_make_baseline_fixtures(position: Position) -> tuple[pd.DataFrame, pd.DataFrame]` helper in root `conftest.py` keeps the duplication shallow. Per-position rows differ only in `position`, `target_stats` columns, and which feature columns are populated; the rest of the structure is shared.

### 4.3 Smoke test (`tests/test_smoke.py`) — parametrize across positions

Per brainstorming Q6 (option B). The smoke test in 3a does fit→predict→write_partition for WR only. 3b parametrizes it across all four positions:

```python
@pytest.mark.parametrize("position", [Position.QB, Position.RB, Position.TE, Position.WR])
def test_smoke_round_trip(position: Position, ...) -> None:
    dispatch = POSITION_DISPATCH[position]
    model = dispatch.factory()
    # ... fit, predict, write_partition, read_partition, assert
```

Catches "I extended `BaselineModel` and broke RB silently" regressions earlier than the per-position test files would. ~5s per position × 4 = ~20s total smoke runtime.

Fixtures consumed by the smoke test must reach root scope (pytest fixtures only inherit downward). The cleanest path: leave the renamed `baseline_features_wr` / `baseline_weekly_stats_wr` in root conftest, and **also** promote `baseline_features_qb` / `_rb` / `_te` (and the corresponding `baseline_weekly_stats_*`) to root. That's six fixtures at root; alternative is a single `baseline_fixtures_for(position)` factory helper, but per-position named fixtures are clearer to readers.

### 4.4 TE feature builder tests (Phase 1 schema change)

`tests/test_features/test_te.py` gets two new assertions (rushing columns present in output) and a new fixture row exercising non-zero TE rushing (a Taysom-Hill-shape row with carries > 0). `tests/test_features/test_te_leakage.py` should pass without changes — the leakage assertions don't depend on which columns exist.

---

## 5. Real-data hardening + sanity-check eval

### 5.1 Per-position 2024 sanity-check eval

After Phase 6 trains each artifact, `scripts/sanity_check_baseline.py {pos}` runs against 2024 (the same held-out window as 3a) and reports:

- Per-stat RMSE / MAE / mean_pred / mean_actual.
- Composite (PPR points) RMSE / MAE.
- Top-N season-total rank correlation (Spearman, all rostered players at the position).
- Calibration: fraction in `[p10, p90]`, fraction `<= p90`.

**Stdout-only, no CI gate added in 3b.** Plan 3c owns CI gating.

The 3a spec's soft thresholds (Spearman top-N corr ≥ 0.4, calibration coverage in 70-90%, per-stat RMSE within 2× naive baseline) get reported in the PR description per position but explicitly **not enforced**. If a position's number is concerning, that's a flag for Plan 3c's backtest harness to formalize, not something we hand-tune in 3b.

Each position's eval output gets recorded as a section in `project_management.md`, mirroring the existing "Plan 3a — 2024 WR sanity check" entry. Four sections total at end of plan (one per position; the WR retrain produces a new `model_id` even though the WR factory's `code_hash_files` list is identical to 3a's — `baseline.py` itself is part of the hashed list and 3b modifies it).

### 5.2 Real-data drift expectations

3a's first real-data pull required eight ingest/feature drift fixes (TODO #16). 3b's expectations:

- **Column-rename drift:** front-run by TODO #8's smoke tests (`pytest -m network --run-network`), which now exist. Run before Phase 6.
- **Position-specific edge cases in feature builders:** unknowns. Plausible candidates: bye-week QB rosters; defensive-team rosters in QB/RB depth charts; players traded mid-window (TODO #15's fix should propagate automatically since the helper now joins on `(gsis_id, team)`).
- **Calibration / fit quality per position:** WR was 70.8% in `[p10, p90]`. QB/RB/TE will likely be similar or worse for low-mean stats. Acceptable; Plan 3c owns this.

If Phase 6 surfaces a drift that requires a non-trivial fix, it gets its own commit and a corresponding entry in TODO #16's drift list.

---

## 6. Decisions made during brainstorming

| Q | Choice | Rationale |
|---|---|---|
| Q1 (scope) | C — mirror 3a end-to-end + generalize CLI scripts from the start | Avoids producing four near-duplicate scripts after 3b ships; the registry pattern pays off in 3c's backtest harness. |
| Q2 (schema dispatch) | A — add `feature_schema` as required `BaselineModel` constructor arg | Mirrors how the dataclass already carries `target_stats` / `feature_columns` / `dist_families`; per-position config stays per-factory. |
| Q3 (target stats) | B — TE includes rushing (Taysom Hill rationale) | Per-position natural sets, 6 stats each except WR (also 6); 2PT and return_tds omitted as low-base-rate noise. |
| Q4 (dist families) | A — NORMAL for yards, GAMMA for counts; no per-position deviation | Carrying 3a's convention. POISSON for low-mean integer stats deferred to 3c. |
| Q5a (script dispatch) | A — centralized `POSITION_DISPATCH` registry in `models/__init__.py` | One canonical "what positions does the system know about" answer; reused by 3c's backtest harness and 4's CLI verbs. |
| Q5b (code-hash file list) | Constructor arg per factory (mirrors Q2's pattern) | Per-position config stays per-factory. |
| Q6 (testing pattern) | A — per-position test files | Mirrors precedent under `tests/test_features/`; clearer failure isolation than parametrized files. |
| (Section 1) | Required (no-default) constructor args; break the 3a artifact | Cleaner code than a None-default branch; retrain takes ~3 minutes. TODO entry to retrain WR. |
| (Section 2.1) | TE rushing features extension | Q3 set TE rushing as a target; the schema currently omits it. ~10-line schema/builder change. |
| (Section 4.3) | Smoke test parametrizes across all four positions | Catches cross-position regressions (one factory broken silently) earlier. ~20s runtime. |

---

## 7. Required follow-ups (during this plan)

These get written to `TODO.md` as part of the plan execution; they are **not** deferred — they're tracked here so the spec → plan crosswalk catches them.

1. **Retrain WR 2018-2023 artifact** after Phase 2 breaks the dataclass shape. Closed by Phase 6's WR run.
2. **Update `project_management.md`** with per-position sanity-check sections + Plan 3c next-action.
3. **Verify TODO #15's fix propagates correctly** to RB/TE — they inherited the `(gsis_id, team)` merge automatically; Phase 6's training is the first real-data exercise. If a traded-player issue still surfaces, fix and amend TODO #15's closing note.

## 8. Out of scope / explicitly deferred

- Walk-forward backtest harness with CI gating → Plan 3c.
- Season aggregation, availability model, bye-week handling → Plan 3c.
- POISSON distribution family → Plan 3c.
- K / DST baselines → TODO #10.
- Joint correlations between players → TODO #1.
- Public Python API + CLI verbs (`python -m projections refresh|...`) → Plan 4.
- Calibration tuning (MLE-fit gamma α, residual-variance bucketing) → Plan 3c.
- Feature parquet caching (`data/features/{pos}/...`) → deferred unless training is slow enough during Phase 6 to motivate it.
