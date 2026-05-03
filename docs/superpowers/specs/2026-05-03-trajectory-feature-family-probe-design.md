# Trajectory Feature Family Probe — Design

**Date:** 2026-05-03
**Branch:** `feat/probe-trajectory`
**Status:** spec
**Predecessor:** PR #24 (PBP pressure family probe — verdict NULL durable). Track 2A complete; pivoting to model-improvement tracks per project_management.md "Next action" (2026-05-03).
**Sibling probe (queued, separate spec on this branch):** weather-feature family. This spec covers trajectory only.

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether the **trajectory feature family** carries orthogonal signal beyond v1 + the already-shipped PR #21 RB PBP cols, on the production model classes (BaselineModel + lgb-nb composite).

The trajectory family targets a **mechanism axis the existing model does not see**: career arc / age / role transitions. Today's per-position feature builders use only trailing-N levels (`*_per_game_l4`); they do not encode whether a player is rising or declining within their career.

### 1.2 Family-level prior framework

Per project_management.md (2026-04-30 "PBP Feature Family Probe"), the canonical workflow is:

1. Bundle 3–4 mechanism-distinct candidates per family probe.
2. Run the probe at BaselineModel + lgb-nb composite × augment + swap modes.
3. SIGNAL → greenlight a per-position integration plan (analogous to PR #20 → PR #21).
4. NULL → close the family at this unit; refined-unit candidates are unlikely to clear absent independent evidence.

Track 2A's verdict pattern: 1 of 4 family probes returned SIGNAL (PR #20 RB PBP team features). Three returned NULL durable (PR #22 receiver air-yards/aDOT, PR #23 red-zone, PR #24 pressure). The mean prior for any single family probe returning SIGNAL is therefore ~25% — not high — but the cost of probing is bounded and the upside is concrete (PR #21 shipped -0.0124 fpts adoption gate on RB).

### 1.3 Success criteria

The probe **ships** when all four pass. The verdict (ADOPT/NULL) is informational, not a ship gate.

1. **Coverage:** per-(position, season) override coverage ≥95% over the 2021–2024 eval window. UDFA fallback in place so coverage isn't draft-pick-restricted. The 2018 cold-start may drag pooled coverage below 95% (trend + snap-gradient features need 8 prior active games); per-season 2019–2024 ≥95% per (position, season) pair satisfies criterion 1 even if pooled requires a relaxed `--coverage-threshold 0.90` invocation, per PR #23 precedent.
2. **Probe completeness:** 4 reports per family — BaselineModel × augment, BaselineModel × swap, lgb-nb composite × augment, lgb-nb composite × swap. All complete without error. Phase 1 (per-stat pooled) and Phase 2 (composite via `--force-composite`) verdicts rendered to the per-(model, mode) reports + summary report following the PR #24 template.
3. **Both model classes tested at composite** — Phase 1 is RidgeCV-only regardless of `--model`, so bare lgb-nb runs are tautological with baseline. Re-run lgb-nb with `--force-composite` to actually test lgb-nb at composite (same fix as PR #22 §3.2 spec gap).
4. **Verification gates green:** mypy strict + ruff + ruff format clean across `src/`, `scripts/`, `tests/`. Relevant pytest subset clean.

### 1.4 Out of scope (deferred)

- **Production integration.** A SIGNAL verdict greenlights a separate per-position integration plan (extending each `*FeaturesSchema` + builder, à la PR #21). This branch is probe-only.
- **Weather family probe.** Sibling spec on the same branch *after* the trajectory verdict lands. Not bundled into this probe per Question 2's design decision (separate mechanism axes — career arc vs game environment — so attribution requires separate probes).
- **Refined trajectory candidates** — `is_2nd_year`, `is_3rd_year`, breakout flags, depth-chart-rank trends. Heavily collinear with `age` for non-veteran players; revisit only if this family returns SIGNAL and we want to refine.
- **Position-specific aging-curve features** (e.g., `is_past_rb_peak = age >= 28`). Tree-based models capture non-linearities natively; Ridge baseline gets only the linear term. Acceptable v1 limitation.
- **id_map cleanup** (TODO #9b's `unique=True` on `pfr_id`). Independent change.

---

## 2. Ingest extension: draft picks

### 2.1 Source

`nfl_data_py.import_draft_picks(years)` returns one row per draft pick. Cover years `1980-2024` (safe horizon for any active 2018+ player; UDFAs and pre-1980 entrants both go through the inference fallback).

Relevant columns (verified empirically against `import_draft_picks([2022])`):

| Source column | Canonical name | Type | Notes |
|---|---|---|---|
| `gsis_id` | `gsis_id` | `GsisId` | Validated via `validate_gsis_id`; primary key. Source returns `str` but may contain empty / null values for older drafts; rows without a valid gsis_id filtered with a logging warning during ingest. |
| `season` | `draft_year` | `pd.Int64Dtype()` | Year of the draft |
| `round` | `draft_round` | `pd.Int64Dtype()` (nullable) | Informational; not used by trajectory features |
| `pick` | `draft_overall_pick` | `pd.Int64Dtype()` (nullable) | Informational; not used by trajectory features |
| `pfr_player_id` | `pfr_id` | `PfrId` (nullable) | Cross-reference against id_map |
| `age` | `draft_age` | `pd.Float64Dtype()` (nullable) | **Age at the time of the draft** (April of `draft_year`). Load-bearing for the `age` feature in §3.1 — gives us biological age, not just pro-experience age. |

### 2.2 Schema

`DraftPicksSchema` in `schemas.py`:
- Primary key: `gsis_id`, `unique=True`.
- Strict pandera schema with `strict="filter"` and reassignment per project convention.
- All non-key columns nullable except `draft_year` (load-bearing for trajectory features).

### 2.3 Ingest module

`src/projections/ingest/draft_picks.py`:
- `refresh_draft_picks(data_root: Path, seasons: range | Iterable[int]) -> None`.
- Snapshot semantics: a season's draft never changes after the draft completes, so re-runs are idempotent overwrites.
- Persists one parquet partition per `season=YYYY` under `data/draft_picks/season=YYYY/part.parquet`.
- Rows with malformed `gsis_id` filtered with a `logging.warning` (mirrors `id_map.py` malformed-row handling).
- Uses `store.write_partition` (the only sanctioned parquet write path).

Mirrors `pbp.py`'s shape directly. Follows the canonical ingest template per `CONTRIBUTING.md` "Adding a new ingest source".

### 2.4 Threading through to features

The trajectory compute module reads the full draft_picks partition set on demand (via `store.read_partition` per-season + `pd.concat`) and builds an in-memory `DraftLookup = dict[GsisId, tuple[int, float]]` keyed by gsis_id, carrying `(draft_year, draft_age)` per player. `draft_age` may be NaN for rows where the source returned a missing age. No schema-level wiring through builder kwargs — this is probe-only and the override script reads draft_picks directly.

If the family returns SIGNAL and we proceed to integration, the follow-up plan will thread a `draft_picks: DraftPicksDataFrame` kwarg through each builder (analogous to `pbp` in Plan 9).

### 2.5 UDFA / pre-coverage fallback

Players absent from the draft_picks parquet are either undrafted (UDFAs) or signed before the API's coverage horizon. Both go through the same fallback:

- Inferred `draft_year` = the player's earliest `season` value across all `weekly_stats` partitions.
- `is_rookie` uses the inferred value.
- `age` uses `season - inferred_draft_year + offset` — represents pro-experience age rather than biological age. Acceptable v1.

Override row gets a `draft_year_inferred: bool` informational column (not a probe feature) so we can audit fallback frequency post-hoc. Expected fallback rate is moderate — UDFAs are ~30% of NFL roster spots — so this is a load-bearing path.

### 2.6 Opt-in network smoke

One new test in `tests/test_ingest/test_api_drift.py::test_draft_picks_api_columns_and_schema`:
- Marker: `@pytest.mark.network`.
- Calls `import_draft_picks([2022])`, asserts the 5 source columns exist with expected dtypes.
- Runs `refresh_draft_picks(tmp_path, [2022])` end-to-end; pandera surfaces dtype drift after a `nfl_data_py` version bump.

Same pattern as the existing `test_pbp_api_columns_and_schema`.

### 2.7 What this does not do

- **Does not extend `IdMapSchema`.** The draft_picks parquet is a separate partitioned table keyed by `gsis_id`. Joining draft info into id_map would muddy id_map's role as the cross-platform identity table (and would break the `pfr_id`/`espn_id`/`sleeper_id` `NewType` separation).

---

## 3. Trajectory feature definitions

Four mechanism-distinct features, all `pa.Column(pa.Float64, nullable=True)` in the override schema.

### 3.1 `age`

- Continuous, position-agnostic. Biological age in the target season.
- Definition: `draft_age + (season - draft_year)`. The `import_draft_picks` source returns `age` (age at draft, April of `draft_year`) for every drafted player — we use it directly.
- For UDFAs / pre-1980 / drafted-but-missing-`age`: fall back to `season - inferred_draft_year + age_offset`, where `age_offset = 22` (a reasonable mean entry age) and `inferred_draft_year` per §2.5. The `age_offset` lives as a module-level constant in `trajectory_features.py`. The fallback gives pro-experience age + a constant; the model's per-position coefficient handles the rest. Note that the `inferred_draft_year` value is *less* reliable than the explicit fallback path (i.e., a 2018-debut player's inferred year is only correct if that was their actual rookie year, not a return from a long absence), so most coverage of this feature comes from the primary `draft_age` path.
- The `draft_year_inferred` informational column on the override (§2.5) doubles as an audit for which rows used the fallback `age`.
- Compute fn: `compute_age(weekly_stats, draft_lookup) -> DataFrame[gsis_id, season, age, draft_year_inferred]` — one row per (player, season) where the player has at least one weekly_stats row that season. The lookup carries `(draft_year, draft_age)` per gsis_id; missing key → inferred fallback path.

### 3.2 `is_rookie`

- Binary 0/1 encoded as `Float64` (not `bool`) for ML-compatible schema dtype.
- Definition: `1.0` if `season == draft_year` (or `season == inferred_draft_year`), else `0.0`.
- Compute fn: `compute_is_rookie(weekly_stats, draft_lookup) -> DataFrame[gsis_id, season, is_rookie]` — one row per (player, season). Uses the same fallback as `compute_age`.

### 3.3 `volume_trend_l4_minus_prior_l4`

- Continuous, **position-tailored** source stat.
- Position-specific source stat:
  - QB: `attempts` per game
  - RB: `carries` per game
  - WR / TE: `targets` per game
- Definition: `mean(stat_per_game over [w-4, w-1]) - mean(stat_per_game over [w-8, w-5])`. **Non-overlapping windows.**
- Active-game denominator: per-game means use only games where the player has a `weekly_stats` row that week (mirrors PR #22's "active games" framing). A bye/IR week is excluded from the denominator, not counted as 0.
- Cold-start: weeks where the player has fewer than 8 prior active games yield `NaN`. This drives the 2018 cold-start coverage pattern.
- Per-position compute fns: `compute_qb_volume_trend(weekly_stats)`, `compute_rb_volume_trend(weekly_stats)`, `compute_wr_te_volume_trend(weekly_stats)` — each returns `(gsis_id, season, week, volume_trend_l4_minus_prior_l4)`. Position dispatch happens at the assembler level (`attach_trajectory_features` selects the correct fn per position).

### 3.4 `snap_pct_change_l4_vs_prior_l4`

- Continuous, position-agnostic.
- `snap_pct` = `SnapCountsSchema.offense_pct` directly. Already a per-player per-game offensive-snap share in [0, 1]; no ratio computation needed in this module.
- Definition: `mean(offense_pct over [w-4, w-1]) - mean(offense_pct over [w-8, w-5])`. Non-overlapping windows; same active-game denominator pattern as §3.3.
- A player without a `snap_counts` row in a given week (bench / inactive / practice-squad) is excluded from the active-game denominator (matches §3.3 framing) — *not* counted as `offense_pct = 0`.
- Compute fn: `compute_snap_pct_change(snap_counts) -> DataFrame[gsis_id, season, week, snap_pct_change_l4_vs_prior_l4]` — one row per (gsis_id, season, week) where the player has a snap_counts row. Simplified signature: no `weekly_stats` parameter needed since `snap_counts` is the sole source.

### 3.5 Schema integration deferred

Like Track 2A's probe-only specs, the four columns live in the *override parquet*, not the position schemas yet. A SIGNAL verdict triggers a follow-up plan to:
- Add the four cols to each `{Qb,Rb,Wr,Te}FeaturesSchema`.
- Wire `attach_trajectory_features` into each `build_*_features`.
- Update `BaselineModel._{POS}_FEATURE_COLUMNS` (the spec-gap fixed in PR #21 — hardcoded tuples per position; lightgbm derives features from the schema dynamically and would auto-pick-up).
- Refresh feature caches under `data/features/{position}/`.
- Run the dual-run adoption gate (`scripts/adoption_gate.py --baseline-run ... --candidate-run ...`).

---

## 4. Module structure

### 4.1 New files

```
src/projections/ingest/draft_picks.py     # refresh_draft_picks + helpers
src/projections/features/trajectory_features.py  # 6 compute fns + attach helper + assembler
scripts/build_trajectory_override.py      # CLI override generator
tests/test_ingest/test_draft_picks.py     # synthetic-fixture tests
tests/test_features/test_trajectory_features.py  # 24+ feature unit tests
tests/test_scripts/test_build_trajectory_override_cli.py  # 4 CLI tests
```

### 4.2 Modified files

```
src/projections/schemas.py                # +DraftPicksSchema
src/projections/ingest/__init__.py        # +refresh_draft_picks export
tests/test_ingest/test_api_drift.py       # +test_draft_picks_api_columns_and_schema (network smoke)
CONTRIBUTING.md                           # +"Regenerating the trajectory override" subsection
```

### 4.3 `trajectory_features.py` interface

Pattern matches PR #24's `pbp_pressure_features.py` — each `compute_*` returns every (gsis_id, season[, week]) combo with the feature value (NaN where prerequisite history is insufficient). The assembler merges all per-week feature frames onto the player-team-week index.

```python
# DraftLookup carries (draft_year, draft_age) per gsis_id. draft_age may be NaN
# (drafted-but-missing-age, rare); a missing key means UDFA / pre-coverage —
# both edge cases route to the inferred-draft-year fallback in compute_age.
DraftLookup = dict[GsisId, tuple[int, float]]

# Per-(gsis_id, season) features — age and rookie status do not change within a season.
def compute_age(
    weekly_stats: pd.DataFrame,
    draft_lookup: DraftLookup,
) -> pd.DataFrame:
    """Returns (gsis_id, season, age, draft_year_inferred) — one row per (player, season)
    where the player has at least one weekly_stats row in that season."""
    ...

def compute_is_rookie(
    weekly_stats: pd.DataFrame,
    draft_lookup: DraftLookup,
) -> pd.DataFrame:
    """Returns (gsis_id, season, is_rookie) — one row per (player, season).
    Uses the same inferred-draft-year fallback as compute_age."""
    ...

# Per-(gsis_id, season, week) features — volume trends and snap-share change. The
# trailing-4 / prior-4 windows mean (season, week) granularity is load-bearing.
def compute_qb_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame: ...
def compute_rb_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame: ...
def compute_wr_te_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame: ...
def compute_snap_pct_change(snap_counts: pd.DataFrame) -> pd.DataFrame: ...

# Attach helper — merges all per-week feature frames onto the player-team-week index.
# Position controls which volume_trend variant is used; ageis_rookie / snap_pct_change
# are uniform.
def attach_trajectory_features(
    index: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    draft_lookup: DraftLookup,
    position: Position,
) -> pd.DataFrame:
    """Returns a copy of index with 4 nullable-Float64 cols appended:
    age, is_rookie, volume_trend_l4_minus_prior_l4, snap_pct_change_l4_vs_prior_l4.
    + draft_year_inferred informational column."""
    ...

# Public assembler — dispatches per position across the index.
def build_trajectory_overrides(
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    draft_lookup: DraftLookup,
    player_team_week_index: pd.DataFrame,  # gsis_id, season, week, team, opp, position
) -> pd.DataFrame:
    """Returns the override frame: (gsis_id, season, week, age, is_rookie,
    volume_trend_l4_minus_prior_l4, snap_pct_change_l4_vs_prior_l4,
    draft_year_inferred). One row per input index row.

    Raises:
        ValueError: malformed gsis_id format or duplicate (gsis_id, season, week) keys.
    """
    ...
```

### 4.4 `build_trajectory_override.py` CLI

Mirrors `scripts/build_pbp_pressure_override.py`:
- Args: `--seasons RANGE`, `--data-root PATH`, `--output PATH`.
- Reads weekly_stats, snap_counts, draft_picks via `store.read_partition`.
- Builds the player-team-week index for each season's eval window.
- Calls `build_trajectory_overrides`, writes the output parquet.

Acknowledged carry-over from PR #24: `_read_concat`, `_FANTASY_POSITIONS`, `_build_player_team_week_index`, `_parse_season_range` are duplicated across PR #20 / #23 / #24 override scripts. We will *not* fold the extraction into this branch (would scope-creep the probe spec); the trajectory script will copy the same patterns. Extraction to `scripts/_pbp_override_common.py` (and a sibling `scripts/_override_common.py` for the trajectory case) is captured in the existing project_management.md housekeeping list.

---

## 5. Probe protocol

### 5.1 Modes & model classes

Four reports per family — same matrix as PR #24:

| Model | Mode | Phase 1 | Phase 2 | Output |
|---|---|---|---|---|
| BaselineModel | augment | yes (RidgeCV per stat) | composite (default; via summary verdict) | `feature_probe_trajectory_augment.{md,csv}` |
| BaselineModel | swap | yes (RidgeCV per stat) | composite | `feature_probe_trajectory_swap.{md,csv}` |
| lgb-nb composite | augment | tautological (RidgeCV; skipped) | composite via `--force-composite` | `feature_probe_trajectory_lgbnb_augment.{md,csv}` |
| lgb-nb composite | swap | tautological | composite via `--force-composite` | `feature_probe_trajectory_lgbnb_swap.{md,csv}` |

### 5.2 Probe invocation

For each (model, mode) cell:
```
python scripts/probe_feature_signal.py \
    --override data/feature_overrides/trajectory.parquet \
    --override-cols age is_rookie volume_trend_l4_minus_prior_l4 snap_pct_change_l4_vs_prior_l4 \
    --mode {augment,swap} \
    --model {baseline,lgbnb} \
    --positions QB RB WR TE \
    --seasons 2021 2022 2023 2024 \
    [--coverage-threshold 0.95] \
    [--force-composite]  # only on lgbnb runs
```

In swap mode, `--drop` lists the same 4 columns; the candidate side substitutes the override values for the dropped existing columns. Trajectory features have no existing-column counterparts, so swap mode effectively *adds* them only when present in the override (matches PR #24 swap-mode semantics on a brand-new feature class).

### 5.3 Eval window

- Train: 2018–2020.
- Holdout: 2021–2024.
- Matches all prior probes for cross-comparability.

### 5.4 Coverage threshold

- Default: `--coverage-threshold 0.95`.
- Fallback: relax to 0.90 if pooled coverage drags below 0.95 due to 2018 cold-start (precedent: PR #23). Document the relaxation explicitly in the summary report.

### 5.5 Summary report template

`reports/feature_probe_trajectory_summary.md` follows the PR #24 template:
- Decision log (commits + reasoning).
- Per-mode verdict table (Phase 1 SIGNAL count + Phase 2 ADOPT/MARGINAL/DO_NOT_ADOPT verdicts).
- Mechanism annotation (which feature, if any, drove signal; or which `target_stat` cells regressed).
- Coverage note + threshold relaxation if applied.
- Refined-unit candidates left unexplored.

---

## 6. Testing strategy

### 6.1 Per-feature unit tests (`tests/test_features/test_trajectory_features.py`, ~24 tests)

For each of the 4 features (age, is_rookie, volume_trend, snap_pct_change), 6 tests:
1. Correct computation on a representative synthetic fixture.
2. Cold-start NaN: weeks where the player has < 8 prior active games yield `NaN` (n/a for age + is_rookie).
3. Mid-season trade: player switches teams within the trailing-8 window — values computed across both teams' games (player-level metric).
4. Bye-week / IR / inactive: excluded from the denominator, not counted as 0.
5. UDFA fallback (age + is_rookie only): uses inferred_draft_year correctly.
6. Cross-season window: week 1 of season N+1 uses tail-end of season N.

### 6.2 Override assembler tests (~3 tests)

- Coverage check on a representative fixture.
- `draft_year_inferred` column populated correctly for UDFAs.
- Schema validation holds (override parquet round-trips through pandera).

### 6.3 CLI tests (`tests/test_scripts/test_build_trajectory_override_cli.py`, 4 tests)

- argparse smoke (parse_args with valid args).
- Default invocation writes expected file structure.
- Custom `--seasons` range respected.
- Error on missing `data_root` (clean error, not a stack trace).

### 6.4 Ingest tests (`tests/test_ingest/test_draft_picks.py`, ~5 tests)

- Synthetic-fixture: `refresh_draft_picks` writes one partition per season.
- Schema validation on the persisted parquet.
- Malformed `gsis_id` rows filtered with a logging warning (mirrors `id_map.py` test pattern).
- Idempotent re-run produces bit-identical output.
- Empty seasons range is a no-op (no error).

### 6.5 Network smoke (opt-in)

`tests/test_ingest/test_api_drift.py::test_draft_picks_api_columns_and_schema` (one test):
- `@pytest.mark.network`, opt-in via `--run-network`.
- Asserts `import_draft_picks([2022])` returns the 5 source columns with expected dtypes.
- Runs `refresh_draft_picks` end-to-end on `tmp_path`.

### 6.6 Verification gate

Per CLAUDE.md "End-of-effort checklist":

```
pytest -v -k "trajectory or draft_picks or schemas"
mypy src tests
ruff check src tests
ruff format --check src tests
```

Plus, before final report: full `pytest -v` to confirm no cross-module regressions.

---

## 7. Decision log (in-spec)

| Decision | Rationale |
|---|---|
| Probe-first (not direct integration) | Track 2A's 1-of-4 SIGNAL hit rate makes probe-first the disciplined default. PR #21's per-position integration is the SIGNAL path. |
| Two separate probes back-to-back (trajectory then weather) on one branch, not one combined probe | Different mechanism axes (career arc vs game environment) make combined attribution ambiguous. Track 2A's pattern is one mechanism per family probe. |
| Add `import_draft_picks` ingest in this branch (not a separate prior branch) | Mirrors PR #20's "ingest + features in one branch" pattern (Plan 9 added `pbp.py` + features together). Mechanical addition; no controversial design surface. |
| Position-tailored volume trend (Approach 2 from brainstorm), not uniform fantasy-points trend | Mechanism-aligned per position: QB attempts, RB carries, WR/TE targets. Modest extra code (3 compute fns vs 1) in exchange for interpretable per-position signal. |
| Non-overlapping `prior_l4` window, not `l8` | `l4 - l8` overlaps (l8 includes the trailing l4); the resulting delta is `l4 - average(l4 + earlier_l4)` which compresses the signal. Non-overlapping is a true "this 4 vs prior 4" trend. |
| Active-game denominator (PR #22 framing) | Bye / IR / inactive weeks excluded from denominator; not counted as 0. Otherwise an injured-but-recovered RB looks like a trend-down. |
| Use `import_draft_picks`'s `age` column (age at draft) directly when available; fall back to `season - inferred_draft_year + age_offset = 22` only when missing | The source data already contains biological age at draft for every drafted player back to ~1980. Computing `season - draft_year + 21` would throw away that information and approximate age within a year, when the source gives us the exact value. The fallback applies only to UDFAs / pre-coverage-horizon players. Empirical 2022 sample: 0/262 missing on `gsis_id`, 0/262 missing on `age` — fallback rate is small. |
| `compute_*` functions return every-week-at-once frames keyed by (gsis_id, season[, week]); no asof-week pivoting | Matches PR #24's established pattern (`pbp_pressure_features.py`). The assembler merges all per-week frames onto the player-team-week index in one pass — simpler than asof-pivoting and reuses existing pandas rolling/groupby. Per-(player, season) features (age, is_rookie) emit one row per (gsis_id, season); per-(player, season, week) features (volume_trend, snap_pct_change) emit one row per (gsis_id, season, week). |
| `is_rookie` encoded as `Float64`, not `bool` | ML-compatible schema dtype across pandera + LightGBM + RidgeCV without coercion gymnastics. |
| Schema integration deferred (probe-only) | Track 2A pattern; SIGNAL verdict greenlights integration plan with adoption gate. Avoids feature-cache invalidation churn for a NULL-likely family. |
| Draft picks stored as separate parquet partitioned table, not folded into id_map | id_map's role is cross-platform identity translation; draft_year is metadata, not an identity column. Keeps `NewType` separation clean. |
| UDFA fallback infers `draft_year` from earliest weekly_stats appearance | Cleanest fallback within existing data. Conflates "true UDFA" with "drafted before 1980" but the latter is rare in our 2018+ window. |
| `draft_year_inferred` informational column on override | Audits fallback frequency post-hoc. Not a probe feature. |
| Eval window 2021–2024 holdout, train 2018–2020 | Matches all prior probes for cross-comparability (PR #20–#24). |
| `--coverage-threshold 0.95` default; relax to 0.90 only if pooled drags below | PR #23 precedent. Document any relaxation in the summary report. |
| Override script duplicates `_read_concat` / `_FANTASY_POSITIONS` / etc. from PR #20 / #23 / #24 | Triplicate-extraction is on the housekeeping list; folding it into a probe-only spec is scope creep. Replicate the pattern. |

---

## 8. Open questions / known limitations

- **Position-specific aging curves not encoded.** Tree-based models capture non-linearities natively, but Ridge baseline only sees the linear `age` term. A `is_past_position_peak` flag would help Ridge but is collinear with `age` for trees. Acceptable v1 limitation; revisit if SIGNAL.
- **`age` accuracy depends on the primary draft-picks path.** When `draft_age` is present in the source (the common case for 1980+ drafted players), the feature is biologically accurate to within ~1 year (the source records age at the April draft date; we treat it as constant for the season). When the fallback path fires (UDFAs / pre-coverage / drafted-but-missing-age), the feature is `season - inferred_draft_year + 22` — pro-experience age + a constant. The model fits a per-position coefficient regardless; the offset is a constant shift that doesn't bias relative ordering, but the within-position dispersion is wider on the fallback path. The `draft_year_inferred` audit column tracks fallback frequency.
- **Snap-counts ingest filters bench/practice players.** A `snap_pct = 0` game is genuinely "0% snap share" only if the player was on the active roster. Players inactive for the week have no `snap_counts` row → excluded from the active-game denominator. Same shape as PR #22's receiver-active-games framing.
- **The `draft_picks` parquet covers 1980–2024.** Players signed before 1980 (none active in our 2018+ window) and UDFAs both go through inference. The override's `draft_year_inferred` audit column tracks this rate.
- **No explicit handling of the COVID-shortened 2020 season.** 2020 is in the training window; trailing-N windows that span 2020↔2021 may have noisier means. Same limitation as all prior probes.

---

## 9. Predecessor pointers

- `docs/superpowers/specs/2026-05-02-pbp-pressure-feature-family-probe-design.md` — PR #24 spec (canonical probe-only template).
- `docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md` — PR #23 spec.
- `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md` — PR #20 spec (the SIGNAL-returning predecessor).
- `docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md` — feature signal probe infrastructure spec.
- `src/projections/features/pbp_pressure_features.py` — implementation template for the compute-fn / attach / assembler structure.
- `scripts/build_pbp_pressure_override.py` — CLI script template.
- `reports/feature_probe_pbp_pressure_summary.md` — summary-report template.
