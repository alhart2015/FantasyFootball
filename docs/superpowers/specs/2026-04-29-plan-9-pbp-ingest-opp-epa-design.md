# Plan 9 — PBP ingest + opponent-adjusted EPA features — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-29
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Parent spec:** `docs/superpowers/specs/2026-04-24-projections-core-design.md`
**Predecessor:** Plan 8 (adoption gate redesign) merged at commit `6675359` (PR #16)

---

## 1. Overview

Plan 9 is the first feature-class plan after the post-Plan-8 pivot. It adds play-by-play ingest from `nfl_data_py.import_pbp_data` and replaces the v1 opponent-strength proxy (`features/_opponent.py:opp_allowed_fppg`) with a schedule-of-strength-adjusted EPA-per-play residual feature on every per-position feature builder.

### 1.1 Why this scope

The post-Plan-8 PM diagnosis identified a series of recent model-class / calibration plans (3e / 5 / 5b / 5c / 6 / 7) that all failed the prior §1.3 gate while extracting the same information from identical features. The next real RMSE lift (estimated 5–15% per TODO #3) lives in features, not model class. Within feature work, opponent-adjusted EPA from PBP is the single most directly motivated upgrade: the v1 helper at `features/_opponent.py` carries an explicit "deferred to PBP" comment, and the post-Plan-3e brainstorm flagged opponent-strength refinement as one of three model-improvement tracks.

Per Plan 2a's "ingest + first feature builder" precedent, Plan 9 bundles ingest with one feature consumer so the column subset is validated by a real downstream user — pure-plumbing ingest plans risk over- or under-curating the column set.

Subsequent PBP-derived features (pace, PROE, air-yards distributions, pressure rate, redzone usage shares) land as separate plans on top of validated PBP plumbing. TODO #3 splits accordingly into 3a (this plan) and 3b (the remaining feature slices).

### 1.2 Goals

- Add `src/projections/ingest/pbp.py` covering `nfl_data_py.import_pbp_data` for 2018+ seasons, written as per-season parquet partitions following the `weekly_stats.py` template.
- Add `PbpSchema` to `schemas.py` with a curated ~25-column subset.
- Replace `opp_allowed_fppg` with `opp_epa_allowed_residual` in `features/_opponent.py`, computing schedule-of-strength-adjusted EPA-per-play residuals over a trailing 4-game window per (defteam, season, week).
- Swap one column on each per-position FeaturesSchema: `opp_allowed_<pos>_fppg_l4` → `opp_pass_epa_allowed_l4` (QB / WR / TE) or `opp_run_epa_allowed_l4` (RB).
- Run the new adoption gate (Plan 8) on per-position pre-Plan-9 vs post-Plan-9 baseline predictions; ship per-position changes per the verdict.
- Opt-in `--run-network` smoke for the new ingest source.

### 1.3 Adoption gate

Adoption decisions are **per position**. For each `Position P ∈ {QB, RB, TE, WR}`, the adoption gate compares the **post-Plan-9 baseline model class** against the **pre-Plan-9 baseline model class** (i.e., same model class, different feature set — the v1 `opp_allowed_<pos>_fppg_l4` column is replaced by `opp_pass_epa_allowed_l4` or `opp_run_epa_allowed_l4`).

This deviates from Plan 8's canonical use case (which compared *different* model classes against the incumbent default within a single backtest run's `results.parquet`). The bootstrap mechanism is unchanged — paired bootstrap on per-row composite scores — but the comparison's semantic interpretation is "did this feature swap improve the model" rather than "is this a better model class." Per-row pairing on `(gsis_id, season, week)` is identical row coverage on both sides because the model class and held-out years are unchanged; only the feature input differs.

**CLI extension required.** The current `scripts/adoption_gate.py` accepts a single `--run` directory and pairs rows between two `model_class` values within that one run. Plan 9 needs to compare two backtest runs (pre-Plan-9 features vs post-Plan-9 features), with the same `model_class="baseline"` value on both sides. Plan 9 extends the CLI to additionally accept `--baseline-run` and `--candidate-run` as separate paths; when both are passed, rows from `baseline-run` are treated as incumbent regardless of `model_class` column, and rows from `candidate-run` are treated as candidate. The single-run `--run` path remains as Plan 8's canonical mode. This extension amortizes across every future feature-class plan in the TODO #3b split.

**Inputs.** Per-row predictions from both feature sets for the same `(gsis_id, season, week)` rows across all held-out years (currently 2021–2024), pulled from two backtest runs' `results.parquet`. After pairing, position P contributes ~3,000–8,000 paired rows.

**Statistical machinery.** Paired bootstrap with `n_bootstrap=1000`, deterministic seed `42`. Resampling unit is the paired player-week — both candidate (post-Plan-9 features) and incumbent (pre-Plan-9 features) are scored on the same draw.

**Per-position metrics.**
- **RMSE delta** (`candidate - incumbent`): pooled across all held-out years. Negative = candidate wins.
- **Spearman delta**: per-year Spearman computed within each held-out year, then averaged unweighted across years.

Per-cell breakdowns (one row per held-out year) are emitted for inspection but do **not** gate adoption.

**Verdict rule.**
```
PASS_RMSE      := rmse_delta.hi_95     <  0.0
PASS_SPEARMAN  := spearman_delta.lo_95 > -0.02

if  PASS_RMSE and  PASS_SPEARMAN:  ADOPT
if  PASS_RMSE and !PASS_SPEARMAN:  MARGINAL — investigate before adopting
if !PASS_RMSE and  PASS_SPEARMAN:  DO_NOT_ADOPT
if !PASS_RMSE and !PASS_SPEARMAN:  DO_NOT_ADOPT
```

**What this gate does not check:**
- No per-cell pass/fail — per-year deltas are informational; only the position-pooled CI gates.
- No Spearman-improvement requirement — only the catastrophic-regression floor.
- No calibration check at all. `weekly_calibration_*` and `season_calibration_*` continue to be emitted into the snapshot for monitoring; the adoption decision ignores them.
- No "max worse cell" floor — sampling variation on a single year is not adoption-blocking.

**Adoption is per position, not all-or-nothing.** A mixed verdict (e.g., QB ADOPT, RB DO_NOT_ADOPT) ships the feature swap for the adopting positions and reverts the swap for non-adopting positions. Per-position routing is mediated by the per-position FeaturesSchema and `BaselineModel` factories — each position's feature list is independent. The ingest layer (PBP partition + `PbpSchema`) ships unconditionally regardless of per-position outcomes; only the FeaturesSchema columns and `_opponent.py` wiring are reverted on a non-adopting position.

If zero positions adopt, the entire feature change is reverted in a single commit; PBP ingest still ships as plumbing for future feature plans.

**Tooling.** Run:
```
python -m scripts.adoption_gate \
  --baseline-run data/backtest/run_pre_plan9_baseline \
  --candidate-run data/backtest/run_post_plan9 \
  --csv-out reports/adoption_gate_plan9.csv
```

Capture the per-position verdict + CI table in §6 of this spec at gate-run time.

### 1.4 Non-goals

- Other PBP-derived features (pace, PROE, air-yards / aDOT distributions, pressure rate allowed, redzone usage shares). Each is a separate plan per the TODO #3 split.
- LightGBM / Tuned / NB / Ensemble model-class re-evaluation under the new feature set. The new feature column flows to those classes automatically through the per-position FeaturesSchema, but the gate is run only against `BaselineModel` in this plan; future model-class plans rerun the gate as needed.
- RB-side `opp_pass_epa_allowed_l4` for pass-catching backs. Defer until per-stat residual analysis suggests RB pass-catching mismatches matter.
- QB-side `opp_run_epa_allowed_l4` for designed-run / scrambling QBs. Same defer rationale — QB fpts are dominated by passing, and run-EPA contribution is a Plan-N+1 refinement.
- Network-fetched smoke as default-on. The opt-in `pytest -m network --run-network` smoke is the canonical drift catch per CONTRIBUTING.md.
- Refactor of the snapshot regression gate. The per-cell tolerance audit in Plan 8 §5 confirmed `rmse_relative=5%` is above per-cell noise floor; expected snapshot drift on a feature change is informative, not noise.

---

## 2. Scope & deliverables

### 2.1 In scope

1. **New ingest module** at `src/projections/ingest/pbp.py`:
   - `_fetch_raw_pbp(seasons)` thin wrapper around `nfl.import_pbp_data(seasons)`.
   - `_normalize_one_season(raw)` — column subset filter, dtype coercion, team-code normalization, schema validation at boundary.
   - `refresh_pbp(data_root, *, seasons)` — per-season idempotent partition write + manifest entry.
   - Mirrors `weekly_stats.py` shape exactly per CONTRIBUTING.md template.

2. **Schema addition to `schemas.py`:** `PbpSchema` (~25 columns; specifics in §4.1).

3. **Schema modifications in `schemas.py`:** four per-position FeaturesSchema columns swapped — `opp_allowed_<pos>_fppg_l4` removed, `opp_pass_epa_allowed_l4` added (QB/WR/TE) or `opp_run_epa_allowed_l4` added (RB).

4. **`features/_opponent.py` rewrite:**
   - Delete `opp_allowed_fppg` and `_row_to_statline` helper.
   - Add `opp_epa_allowed_residual(pbp, *, play_type, n_weeks)` returning `(season, week, opp_team, opp_epa_allowed_residual)` shape, mirroring v1's join interface.
   - One `Literal["pass", "run"]` parameter rather than two functions — the play-type filter is the only delta in algorithm.

5. **Per-position builder updates** (`features/qb.py`, `features/rb.py`, `features/wr.py`, `features/te.py`):
   - New `pbp: pd.DataFrame` arg in each `build_<pos>_features` signature.
   - Replace v1 fppg join with `opp_epa_allowed_residual` call + merge.
   - Output column rename to `opp_pass_epa_allowed_l4` (QB/WR/TE) or `opp_run_epa_allowed_l4` (RB).

6. **Caller plumbing** — four scripts that directly invoke per-position builders:
   - `scripts/refresh_features.py` — load PBP partitions for requested seasons via `read_partition`, pass into each per-position builder. (This is the only writer of the feature cache; the backtest harness reads from cache transparently after the first regenerate.)
   - `scripts/train_baseline.py` — same load step.
   - `scripts/predict_2024.py` — same.
   - `scripts/sanity_check_baseline.py` — same.

   `scripts/backtest.py` reads cached features via `projections.features.cache.read_features` and does not invoke the builders directly — no PBP-load change needed there.

7. **`scripts/adoption_gate.py` extension** (per §1.3 above):
   - Add `--baseline-run` and `--candidate-run` mutually-exclusive-with-`--run` arguments.
   - When both new args are passed, load each run's `results.parquet` separately, label all rows in `baseline-run` as incumbent and all rows in `candidate-run` as candidate (overriding the `model_class` column for pairing purposes), then proceed through the existing pair / bootstrap / verdict pipeline unchanged.
   - Existing single-run path stays the canonical Plan 8 mode; the new dual-run path is the canonical Plan-9-and-beyond feature-comparison mode.
   - 5–8 new tests covering the dual-run code path, mirror the existing single-run test pattern.

8. **Tests** (detail in §7):
   - New `tests/test_ingest/test_pbp.py` (synthetic-fixture coverage).
   - New `test_pbp_api_columns_and_schema` opt-in network smoke in `tests/test_ingest/test_api_drift.py`.
   - New `fake_pbp_df` fixture in `tests/test_ingest/conftest.py`.
   - Replaced `tests/test_features/test_opponent.py` covering the residual algorithm and leakage discipline.
   - Per-position test fixtures gain a PBP fixture; existing leakage / shape tests assert the new column.
   - Backtest snapshot `tests/backtest/model_metrics.json` regenerated; expected drift on every cell from the feature change.

9. **Documentation updates** on merge:
   - `project_management.md`: append decision-log rows for the major design calls (residual formulation, replace-not-augment, per-position adoption); update "Current status" / "Next action" sections with Plan 9 verdict and the next-up plan in the TODO #3 split.
   - `TODO.md`: split TODO #3 into 3a (closed by this plan) + 3b (remaining feature slices on top of PBP plumbing).
   - `CONTRIBUTING.md`: no changes — `pbp` is a standard ingest source under the existing template; the smoke pattern is unchanged.

### 2.2 Explicitly deferred

- All other TODO #3 features. They land as separate plans referencing this one as predecessor.
- RB pass-catcher pass-EPA column. Add only if a residual analysis post-Plan-9 shows RB pass catchers' fpts variance is materially driven by their opponents' pass defense — currently a hypothesis, not evidence.
- PBP per-week partitioning. Per-season partitions handle ~50k rows comfortably; future per-week partitions can land additively when storage or memory matters.
- Ingest of pre-2018 PBP. `nfl_data_py.import_pbp_data` covers 1999+ but column shape varies pre-2016; align to the existing 2018+ ingest window.
- A unified `python -m projections.ingest.refresh` CLI verb (TODO #18). Plan 9 invokes `refresh_pbp` from existing scripts, not a new CLI.
- Feature cache code-hash auto-invalidation (TODO #21). `BaselineModel.code_hash_files` already tracks `_opponent.py` and per-position builder files, so manual `refresh_features.py` re-run after the feature swap is the expected workflow.

---

## 3. Ingest source shape

The new module conforms to the `src/projections/ingest/weekly_stats.py` template per CONTRIBUTING.md.

| Property | Value |
|---|---|
| `nfl_data_py` call | `import_pbp_data(years=[season])` per season |
| Partition path | `data/raw/pbp/season=YYYY/part.parquet` |
| Manifest table name | `"pbp"` |
| Smoke test | `tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema` (opt-in `--run-network`) |
| Storage size | ~50k rows × ~25 cols × 7 seasons ≈ 30–50 MB total |
| Per-season memory | ~10 MB peak during normalize (same order as `weekly_stats`) |

**Curated `_KEEP` column subset (~25 columns):**

| Column | Source dtype | Schema dtype | Notes |
|---|---|---|---|
| `play_id` | int | `Series[int]` | PK with `game_id` |
| `game_id` | str | `Series[str]` | join key for game-level metadata |
| `season` | int32 | `Series[int]` | partition key, coerced to int64 |
| `week` | int32 | `Series[int]` | coerced to int64 |
| `posteam` | str | `Series[str]`, nullable | normalized via `normalize_team_code` |
| `defteam` | str | `Series[str]`, nullable | normalized via `normalize_team_code` |
| `play_type` | str | `Series[str]`, nullable | raw values (`pass` / `run` / `kickoff` / `punt` / `field_goal` / `extra_point` / `qb_kneel` / `qb_spike` / `no_play`) |
| `qb_dropback` | float | `Series[float]`, nullable | reserved for pass-play classification |
| `qb_scramble` | float | `Series[float]`, nullable | counts toward pass plays (sacks-on-scramble + designed scramble runs are split per-stats) |
| `sack` | float | `Series[float]`, nullable | counts toward pass plays |
| `rush_attempt` | float | `Series[float]`, nullable | run-play classification |
| `pass_attempt` | float | `Series[float]`, nullable | pass-play classification (does not include sacks) |
| `epa` | float | `Series[float]`, nullable | per-play EPA (load-bearing for Plan 9) |
| `wpa` | float | `Series[float]`, nullable | reserved for future plans |
| `success` | float | `Series[float]`, nullable | reserved |
| `air_yards` | float | `Series[float]`, nullable | reserved for air-yards-distribution plan |
| `yards_after_catch` | float | `Series[float]`, nullable | reserved |
| `complete_pass` | float | `Series[float]`, nullable | reserved |
| `xpass` | float | `Series[float]`, nullable | reserved for PROE plan |
| `pass_oe` | float | `Series[float]`, nullable | reserved for PROE plan |
| `down` | float | `Series[float]`, nullable | game-state context |
| `ydstogo` | int | `Series[int]`, nullable | game-state context |
| `yardline_100` | float | `Series[float]`, nullable | game-state context |
| `half_seconds_remaining` | float | `Series[float]`, nullable | game-state context |
| `passer_player_id` | str | `Series[str]`, nullable, gsis-id-format-checked | reserved for player-level plans |
| `rusher_player_id` | str | `Series[str]`, nullable, gsis-id-format-checked | reserved |
| `receiver_player_id` | str | `Series[str]`, nullable, gsis-id-format-checked | reserved |

Columns kept-but-reserved (`wpa`, `success`, `air_yards`, `yards_after_catch`, `complete_pass`, `xpass`, `pass_oe`, `down`, `ydstogo`, `yardline_100`, `half_seconds_remaining`, `passer_player_id`, `rusher_player_id`, `receiver_player_id`) are not consumed by Plan 9's feature but are essential to the next 3 plans in TODO #3's split. Pulling them now means future feature plans don't trigger an ingest-schema migration.

Columns explicitly **not** kept include kicker stats, defensive-player IDs, special-teams play detail, advanced WP / EP variants, weather (already in `schedules`), and ~340 of the upstream's ~370 columns. Any future feature plan needing one adds it via additive `_KEEP` extension and a schema update.

---

## 4. Schema additions

### 4.1 New `PbpSchema`

```python
class PbpSchema(pa.DataFrameModel):
    """Per-play data — what `ingest.pbp` produces. Curated subset of
    `nfl_data_py.import_pbp_data`'s ~370-column output."""

    play_id: Series[int] = pa.Field(ge=1)
    game_id: Series[str]
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    posteam: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)
    defteam: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)
    play_type: Series[str] = pa.Field(nullable=True)
    qb_dropback: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    qb_scramble: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    sack: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    rush_attempt: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    pass_attempt: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    epa: Series[float] = pa.Field(nullable=True)
    wpa: Series[float] = pa.Field(nullable=True)
    success: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    air_yards: Series[float] = pa.Field(nullable=True)
    yards_after_catch: Series[float] = pa.Field(nullable=True)
    complete_pass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    xpass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    pass_oe: Series[float] = pa.Field(nullable=True)
    down: Series[float] = pa.Field(ge=1, le=4, nullable=True)
    ydstogo: Series[int] = pa.Field(ge=0, le=99, nullable=True)
    yardline_100: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    half_seconds_remaining: Series[float] = pa.Field(ge=0, le=1800, nullable=True)
    passer_player_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True)
    rusher_player_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True)
    receiver_player_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True)

    class Config:
        strict = "filter"
```

**Dtype rationale:**
- `posteam` / `defteam` are nullable: special-teams plays (kickoffs, punts) lack a clean possession team. We keep these rows because `play_type` filtering happens at feature time, not ingest time.
- Boolean-like indicators (`qb_dropback`, `qb_scramble`, `sack`, `rush_attempt`, `pass_attempt`, `success`, `complete_pass`) are nullable `Series[float]` not `Series[int]`. `nfl_data_py` returns them as float64 with NaN for `play_type=no_play` rows. Coercing to nullable `Int64` would cost a 2× normalize-time hit and risk silent dtype regression in pandera.
- `down` is nullable float for the same reason — `no_play` and pre-snap penalties carry NaN.
- `*_player_id` columns are gsis-id-format-checked even though nullable, matching the codebase's ID-hygiene rule.

### 4.2 FeaturesSchema modifications

Four per-position schemas in `src/projections/schemas.py` swap one column each:

```python
# QbFeaturesSchema, WrFeaturesSchema, TeFeaturesSchema
- opp_allowed_<pos>_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)
+ opp_pass_epa_allowed_l4: Series[float] = pa.Field(nullable=True)

# RbFeaturesSchema
- opp_allowed_rb_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)
+ opp_run_epa_allowed_l4: Series[float] = pa.Field(nullable=True)
```

The new column has no `ge=` / `le=` bound — schedule-of-strength residuals can be negative (stout defenses) or positive (soft defenses), typically in the range ±0.3 EPA/play. A nominal `ge=-2.0, le=2.0` bound would catch unit-mismatches but isn't load-bearing; left out for v1.

`nullable=True` matches the v1 column's nullability — opponents in the first 4 weeks of any season have no trailing-window data, and bye weeks produce sparse (defteam, season, week) rows.

---

## 5. Feature builder shape

### 5.1 `features/_opponent.py` rewrite

The existing module is replaced wholesale. New surface:

```python
from typing import Literal

import pandas as pd


def opp_epa_allowed_residual(
    pbp: pd.DataFrame,
    *,
    play_type: Literal["pass", "run"],
    n_weeks: int,
) -> pd.DataFrame:
    """Schedule-of-strength-adjusted EPA-allowed per play type, computed as
    a trailing-window mean of per-play residuals.

    Per-play residual = EPA(p) - mean_EPA_for(posteam, play_type, in_window),
    where mean_EPA_for is that offense's overall pass/run EPA-per-play in the
    same trailing window. The residual answers: "given who they faced, how
    much better/worse than expected did this defense play?"

    Returns one row per (season, target_week, opp_team) with target_week
    shifted +1 from the trailing window's last week, mirroring the v1
    `opp_allowed_fppg` join interface. The opp_team column carries the
    defense; join onto offense-side feature rows on (season, week, opponent).
    """
```

**Algorithm.** Concrete steps:

1. Filter `pbp` to plays with non-null `epa`, `posteam`, `defteam`. Drop `play_type=no_play`.
2. Classify per play:
   - `is_pass = (play_type == "pass") | (qb_scramble == 1) | (sack == 1)`
   - `is_run  = (play_type == "run") & (qb_scramble != 1)`
3. Per the `play_type` argument, keep only matching plays.
4. Per (offense_team `posteam`, season), compute trailing-window mean EPA per play. This is `E[EPA | offense_team]` for the relevant play type.
5. Per-play residual = `EPA(p) - E[EPA | offense_team_of_p]`, joined back per play.
6. Group by (defteam, season, week), take mean of per-play residuals.
7. For each (defteam, season), apply trailing-N-week mean over the per-week mean residuals.
8. Shift target week +1 (so a week-W trailing-mean joins onto opponent's feature row at week W+1, matching v1 fppg shape).

**Trailing-window edge cases:**
- Weeks 1–`n_weeks` of any season have an underfilled window. v1 `opp_allowed_fppg` uses an expanding window for early weeks (mean over 1, 2, … n_weeks games). Plan 9 uses the same expanding-window discipline — leave the first `n_weeks-1` weeks underfilled rather than requiring `n_weeks` complete prior weeks before emitting.
- A defense with zero plays of the relevant type in a week (extreme — would require an injury-shortened or bizarre game) emits no row for that week; the trailing window skips it.
- A defense's first appearance of the season has no prior rows → emits no row at all for week 1; the offense-side join produces NaN, which the FeaturesSchema's `nullable=True` tolerates.

**Determinism.** Pure-function over the input PBP DataFrame; no random state; bit-identical output for bit-identical input.

### 5.2 Per-position builder integration

Each builder gains a `pbp: pd.DataFrame` arg and one extra merge step:

```python
# features/qb.py / wr.py / te.py — pass-EPA column
pass_epa = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)
features = features.merge(
    pass_epa.rename(
        columns={"opp_team": "opponent",
                 "opp_epa_allowed_residual": "opp_pass_epa_allowed_l4"}
    ),
    on=["season", "week", "opponent"],
    how="left",
)

# features/rb.py — run-EPA column
run_epa = opp_epa_allowed_residual(pbp, play_type="run", n_weeks=4)
features = features.merge(
    run_epa.rename(
        columns={"opp_team": "opponent",
                 "opp_epa_allowed_residual": "opp_run_epa_allowed_l4"}
    ),
    on=["season", "week", "opponent"],
    how="left",
)
```

The v1 fppg merge step is deleted in the same edit. The schema validation at the end of each builder (already a `<Pos>FeaturesSchema.validate(df)` call) catches any column-rename or dtype regression at the boundary.

### 5.3 Caller plumbing

Per-position feature builders are invoked directly (not via cache) from four scripts: `refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`. Each updates to load PBP partitions and pass them in.

`scripts/backtest.py` and the `run_backtest` harness consume *cached* features via `projections.features.cache.read_features`, not the live builder. The cache is written once by `refresh_features.py`, so the backtest harness picks up the new feature column transparently after the first cache regenerate. No backtest-side code change.

```python
# refresh_features.py / train_baseline.py / predict_2024.py / sanity_check_baseline.py — added load step
pbp = pd.concat(
    [read_partition(data_root / "raw", "pbp", season=s) for s in seasons],
    ignore_index=True,
)
features = builder(..., pbp=pbp)
```

`BaselineModel.code_hash_files` already includes `_opponent.py` (verified: `src/projections/models/baseline.py:434`) and per-position feature builder files, so the model artifact's `code_hash` updates automatically when these files change. Existing pre-Plan-9 artifacts won't load against the new feature schema (new column name); retrain via `train_baseline.py` per usual. The same `_opponent.py` path is also tracked by `lightgbm.py:111`, `lightgbm_nb.py:124`, `lightgbm_tuned.py:124` — so any subsequent peer-model adoption gate against the new feature would also pick up the right code-hash invalidation.

---

## 6. Adoption-gate validation procedure

This section will be filled in with concrete verdicts at gate-run time. The procedure is fixed:

1. **Capture pre-Plan-9 baseline predictions.** Either:
   - Locate the most recent `data/backtest/run_<ts>/` from `main` HEAD and copy/rename the per-row `results.parquet` to `data/backtest/run_pre_plan9_baseline/results.parquet`, or
   - Re-run `scripts/backtest.py --model baseline --update-snapshot=false` against the pre-Plan-9 commit if no clean run exists in the local cache.

2. **Run post-Plan-9 backtest (gate input only — no snapshot update yet).** After the feature swap is implemented and tested, run `scripts/backtest.py --model baseline --report` for held-out years 2021–2024 across all four positions. Output written to `data/backtest/run_post_plan9/results.parquet`. The snapshot is **not** updated yet because per-position revert decisions in step 5 may roll back changes; the final snapshot update happens after all reverts are applied (step 7).

3. **Run the adoption-gate CLI.** Invoke:
   ```
   python -m scripts.adoption_gate \
     --baseline-run data/backtest/run_pre_plan9_baseline \
     --candidate-run data/backtest/run_post_plan9 \
     --csv-out reports/adoption_gate_plan9.csv
   ```
   This uses the dual-run mode added in §2.1 deliverable 7.

4. **Capture verdicts in this spec.** Append a verdicts table to §6 (one row per position × year + one pooled row per position) and a "Per-position routing changes shipped" table mirroring Plan 8 §4. Format reuses Plan 8's adoption-gate report shape.

5. **Per-position outcome handling:**
   - **ADOPT** for position P → ship the feature swap for P. P's FeaturesSchema column rename + `features/p.py` builder change stays.
   - **DO_NOT_ADOPT** for position P → revert the FeaturesSchema column for P (restore `opp_allowed_<pos>_fppg_l4`) and the v1 fppg merge in `features/p.py`. Add a follow-up TODO entry capturing the negative result.
   - **MARGINAL** for position P (RMSE clears, Spearman doesn't) → human reviews the per-year breakdown; default is to investigate before adopting.

6. **Zero-position-adopt branch.** If no position adopts, revert all four per-position feature changes and the `features/_opponent.py` rewrite in a single revert commit. PBP ingest, `PbpSchema`, the network smoke, and the `pbp` arg threading still ship — they are pure plumbing for future feature plans.

7. **Final snapshot update.** After all per-position revert decisions are applied (step 5/6), re-run `scripts/refresh_features.py` to regenerate the cache for the surviving feature change, then run `scripts/backtest.py --model baseline --update-snapshot` once. The committed `tests/backtest/model_metrics.json` reflects the final (possibly mixed-adoption) state, not the all-positions-changed gate-input run from step 2.

### §6 verdicts (run dispatched 2026-04-29 21:04)

Paired bootstrap, n_bootstrap=1000, seed=42. Baseline-run = pre-Plan-9 BaselineModel predictions; candidate-run = post-Plan-9 BaselineModel predictions (same model class, only `opp_allowed_<pos>_fppg_l4` → `opp_pass_epa_allowed_l4` / `opp_run_epa_allowed_l4` feature column swapped). Pairing key `(gsis_id, season, week, position)`.

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | +0.0005 ([-0.0125, +0.0144]) | -0.0001 ([-0.0029, +0.0024]) | 2676 |
| RB | DO_NOT_ADOPT | +0.0001 ([-0.0110, +0.0111]) | +0.0006 ([-0.0014, +0.0027]) | 5273 |
| TE | DO_NOT_ADOPT | -0.0037 ([-0.0121, +0.0050]) | +0.0000 ([-0.0028, +0.0027]) | 4257 |
| WR | DO_NOT_ADOPT | +0.0083 ([+0.0043, +0.0124]) | -0.0013 ([-0.0021, -0.0004]) | 8460 |

Per-year breakdowns: `reports/adoption_gate_plan9.csv`. Markdown report: `reports/adoption_gate_plan9.md`.

**All four positions DO_NOT_ADOPT.** QB / RB / TE are null results — RMSE and Spearman CIs both bracket zero. WR is a small but statistically significant **regression**: RMSE +0.0083 fpts (CI strictly above 0); Spearman -0.0013 (CI strictly below 0). Triggers the §6 step 6 "zero-position-adopt branch": revert the per-position feature changes and the `_opponent.py` rewrite; ship the plumbing (PBP ingest, `PbpSchema`, network smoke, `pbp` arg threading, adoption-gate CLI dual-run extension) for future feature plans.

### §6 routing changes shipped

| Position | Pre-Plan-9 feature column | Post-Plan-9 feature column | Reason |
|----------|---------------------------|----------------------------|--------|
| QB | `opp_allowed_qb_fppg_l4` | `opp_allowed_qb_fppg_l4` (reverted) | DO_NOT_ADOPT — null RMSE delta. |
| RB | `opp_allowed_rb_fppg_l4` | `opp_allowed_rb_fppg_l4` (reverted) | DO_NOT_ADOPT — null RMSE delta. |
| TE | `opp_allowed_te_fppg_l4` | `opp_allowed_te_fppg_l4` (reverted) | DO_NOT_ADOPT — point estimate negative but CI brackets 0. |
| WR | `opp_allowed_wr_fppg_l4` | `opp_allowed_wr_fppg_l4` (reverted) | DO_NOT_ADOPT — small but significant RMSE regression (+0.0083 fpts) AND Spearman regression (-0.0013, CI strictly below 0). |

### §6 mechanism interpretation

The post-Plan-3e brainstorm hypothesized 5–15% RMSE improvement from PBP-derived features. The empirical result is essentially flat at the BaselineModel level. Three plausible explanations:

1. **Ridge model is feature-saturated.** Adding/swapping one column doesn't move the needle when the baseline already integrates `implied_team_total`, `spread`, `opp_allowed_<pos>_fppg_l4`, NGS metrics, and snap-rate history. The opponent-strength signal is partially captured by the existing features.
2. **Schedule-of-strength residual is a mild upgrade in expectation.** The v1 `opp_allowed_fppg_l4` is a fpts-weighted summary that implicitly encodes some opponent-strength signal even without explicit schedule adjustment. The marginal gain from explicit residual computation is small.
3. **WR-specific noise.** The new feature regresses WR by ~0.5% RMSE. Possible mechanisms: WR opponent-strength is more variable game-to-game than other positions; the schedule-of-strength baseline's offense-mean has higher variance for low-volume offenses; or v1 fppg's PPR weighting captures something the EPA residual doesn't (e.g., target volume opportunity).

The plan's §1.3 gate correctly identifies this as not-an-improvement, regardless of which mechanism dominates. Per Plan 8's lesson, don't chase noise floor: revert and pivot to either a fundamentally different feature class (PBP-derived but volume-oriented: pace, PROE, air-yards distributions) or a different model class that benefits from the residual feature (LightGBM, ensemble) — both deferred to follow-up plans on top of Plan 9's plumbing.

---

## 7. Tests

| File | Coverage |
|---|---|
| `tests/test_ingest/conftest.py` | Add `fake_pbp_df` fixture mimicking `nfl_data_py.import_pbp_data` shape. ~30 rows covering pass plays, run plays, sacks, scrambles, no-plays, ST plays, with realistic EPA values and team codes (including `JAX` / `LA` aliases for normalization coverage). |
| `tests/test_ingest/test_pbp.py` | (a) `refresh_pbp` writes per-season parquet; (b) idempotent (run twice → row count unchanged); (c) team-code normalization (`JAX` → `JAC`, `LA` → `LAR`); (d) schema validates; (e) curated columns kept (no extras leak through `strict="filter"`); (f) `play_type=no_play` rows survive ingest and are filtered only at feature time. |
| `tests/test_ingest/test_api_drift.py` | New `test_pbp_api_columns_and_schema` opt-in `--run-network` smoke. Pulls one season (2023), asserts every `_KEEP` column is present in the upstream return, runs `_normalize_one_season` end-to-end. |
| `tests/test_features/test_opponent.py` | Wholesale replaced. New cases: (a) residual is zero when defense plays only league-mean offenses; (b) residual is positive when defense allows average EPA against weak offenses (= performed worse than expected); (c) residual is negative when defense allows below-mean EPA against strong offenses; (d) trailing window only uses prior weeks (leakage check); (e) target_week shift correct (residual for trailing window ending at week W joins onto opponent's feature row at week W+1); (f) `play_type="pass"` filter includes sacks and scrambles, `play_type="run"` excludes them; (g) expanding-window discipline for season-start weeks; (h) defteam with zero relevant plays in a week emits no row for that week. |
| `tests/test_features/test_{qb,rb,wr,te}.py` | Each per-position test file gains a `fake_pbp_df` fixture. Existing leakage / shape tests assert the new column appears in the output and is finite for known defenses. References to `opp_allowed_<pos>_fppg_l4` swap to the new column name. |
| `tests/test_models/test_baseline.py` | If any test references `opp_allowed_*_fppg_l4` by string literal, swap the column name. (Likely none — verify during implementation.) |
| `tests/backtest/model_metrics.json` | Snapshot regenerates on the feature swap. Per-cell drift on every cell is expected (a feature change is a real model change); the Plan 8 §5 audit confirmed snapshot tolerances are above per-cell noise floor, so genuine drift surfaces correctly. Update as part of the gate run. |
| `tests/test_scripts/test_adoption_gate.py` | Add 5–8 tests for the new dual-run mode (per §2.1 deliverable 7): (a) `--baseline-run` + `--candidate-run` produce verdicts; (b) mutually exclusive with `--run`; (c) row coverage mismatch between runs raises a clear error; (d) verdict matches the equivalent single-run case when both runs hold the same model_class but the dual-run path is taken; (e) CSV output shape unchanged. Mirror existing single-run test pattern. |

**Test-data prep.** `fake_pbp_df` should include at least:
- 2 offenses with distinct EPA-per-play means (one strong, one weak).
- 2 defenses with distinct schedule-of-strength (one faces only strong offenses, the other only weak).
- 4+ weeks per defense to exercise trailing-window logic.
- One row of each `play_type` (`pass`, `run`, `kickoff`, `punt`, `field_goal`, `extra_point`, `qb_kneel`, `qb_spike`, `no_play`).
- One sack play and one scramble play to exercise the pass-classification.
- One row with `posteam=NaN` to exercise the filter.

---

## 8. Cleanups

Bundled in this plan:

- Delete `features/_opponent.py:opp_allowed_fppg` and `_row_to_statline` helper. The new `opp_epa_allowed_residual` is the replacement; nothing else imports the old function.
- Delete `opp_allowed_<pos>_fppg_l4` columns from each FeaturesSchema (4 schemas).
- Delete tests covering the v1 helper (`tests/test_features/test_opponent.py` is replaced wholesale).

No drive-by cleanups outside this scope. Ingest module additions and per-position builder edits are tightly coupled and ship as one PR.

---

## 9. Risks

- **Per-row PBP runtime.** Loading 7 seasons × computing residuals × per-position is single-threaded pandas. Back-of-envelope ~5–10s per backtest fold. Acceptable; if profiling shows >30s/fold, cache the team-week aggregate as a derived parquet at `data/features/_intermediate/opp_epa_allowed/` and the per-fold compute drops to a single read + a per-position merge.
- **`epa` nullability.** Some plays (`no_play`, kneels, two-point conversions in older seasons) carry `epa=NaN`. Filter step (1) drops them. After filtering, per-team-week play counts are sensibly ~120–150; verify on the network smoke output to catch upstream regressions.
- **Adoption-gate verdict mixed across positions.** Per-position routing handles this naturally — each position's FeaturesSchema and builder are independent. Commit message + decision-log entry must clearly call out which positions adopted and which reverted.
- **Snapshot drift on non-adopted positions.** If RB doesn't adopt and we revert RB's feature change, RB snapshot rows return to pre-Plan-9 baseline (no drift after revert). Adopted positions' snapshot rows update. Snapshot regression gate stays consistent.
- **`nfl_data_py.import_pbp_data` upstream variability.** Column names have shifted historically (e.g., `pass_oe` was renamed at some point). The opt-in network smoke catches column-rename / removal drift; per CONTRIBUTING.md, run after any `nfl_data_py` version bump. Plan 9 pins to the existing `nfl_data_py>=0.3.2` constraint.
- **Schema strictness on player-id columns.** `passer_player_id` / `rusher_player_id` / `receiver_player_id` are gsis-id-format-checked; if upstream emits any malformed legacy IDs (per TODO #16's id_map drift list), the schema validation may fail loudly. Plan 9's normalize step should filter rows with malformed player IDs analogously to id_map's existing handling — verify on the network smoke.

---

## 10. Documentation updates on merge

- **`project_management.md`:**
  - Append a Plan 9 entry at the top (status: complete after gate verdicts).
  - Append decision-log rows for: (a) PBP storage shape (raw per-play parquet, per-season partition, ~25-column curated subset); (b) schedule-of-strength residual formulation as the chosen "opp-adjusted" interpretation; (c) replace-not-augment v1 fppg; (d) per-position adoption verdicts.
  - Update "Current status" / "Next action" sections to reflect Plan 9 verdict and queue the next-up plan in the TODO #3b split (likely PROE or pace per spec §6 of TODO #3).
- **`TODO.md`:**
  - Split TODO #3 into 3a (this plan, closed) and 3b (remaining feature slices on top of PBP plumbing).
  - Add follow-up entries for any non-adopting position (capturing the negative result for future reference).
- **No changes to `CONTRIBUTING.md`.** The PBP ingest follows the existing template; the test pattern is unchanged.
