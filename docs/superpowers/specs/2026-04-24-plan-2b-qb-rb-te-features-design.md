# Plan 2b — QB/RB/TE feature builders — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-24
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Parent spec:** `docs/superpowers/specs/2026-04-24-projections-core-design.md`
**Predecessor:** Plan 2a merged at `7926090`.

---

## 1. Overview

Plan 2b adds per-position feature builders for QB, RB, and TE — the three remaining offensive skill positions — following the validated WR pattern from Plan 2a. K and DST are split out into a future plan because they need data sources we haven't ingested yet (FG-attempt detail for K; play-by-play and a team-level builder shape for DST).

### 1.1 Goals

- Stand up `build_qb_features`, `build_rb_features`, `build_te_features` — pure-function builders mirroring `build_wr_features`'s shape.
- Add the three corresponding pandera schemas (`QbFeaturesSchema`, `RbFeaturesSchema`, `TeFeaturesSchema`).
- Extend `WeeklyStatsSchema` with QB-needed columns (`attempts`, `completions`, `sacks`) — same incremental pattern as 2a's extension for `targets`/`carries`/`receiving_air_yards`.
- Generalize `wr.py`'s position-share helper into a reusable function in `_rolling.py` so RB/TE can use it for analogous share computations.
- Per-position non-leakage tests + 5 leakage tests each (one per input source) + smoke-test extension covering all 4 builders.

### 1.2 Non-goals (deferred)

- **K and DST** — both need data we don't currently ingest. Tracked as a new TODO #10 (see §10.3) for a future plan after the dependencies land.
- **Play-by-play ingest** (TODO #3) — would unlock proper opponent-adjusted EPA features for all positions, but the FPPG-allowed proxy from 2a is sufficient for v1.
- **Feature parquet storage** (TODO #4) — pure-function only, like 2a.
- **Model training** (Plan 3+) — feature builders only, not consumers.

### 1.3 Why split K/DST out

- **K** intended features (recent FG distance distribution, opp redzone TD allowed %) require data not in `WeeklyStatsSchema`: FG attempts by distance, accuracy by range, redzone trip volume. Either ingest a new source or wait for play-by-play.
- **DST** is fundamentally different in shape — *team-level*, not player-level. Schema's primary key is `Team` not `GsisId`. Intended features (opp pass-block win rate, turnover-worthy throw rate, sack rate allowed) all need play-by-play. Bundling DST into a plan structured around per-player builders would force an awkward shoehorn.

Splitting lets 2b stay mechanical (copy the WR pattern × 3 positions) and lets K/DST get their own brainstorm once dependencies land.

---

## 2. Scope & deliverables

### 2.1 In scope

1. **Three new per-position builders** in `src/projections/features/`:
   - `qb.py` — `build_qb_features(...)` consuming `weekly_stats`, `snap_counts`, `depth_charts`, `ngs_passing`, `schedules`.
   - `rb.py` — `build_rb_features(...)` consuming `weekly_stats`, `snap_counts`, `depth_charts`, `ngs_rushing`, `schedules`.
   - `te.py` — `build_te_features(...)` consuming `weekly_stats`, `snap_counts`, `depth_charts`, `ngs_receiving`, `schedules`.

2. **Three new feature schemas** in `src/projections/schemas.py`: `QbFeaturesSchema`, `RbFeaturesSchema`, `TeFeaturesSchema`.

3. **`WeeklyStatsSchema` extension** with QB-needed columns:
   - `attempts: Series[int] = pa.Field(ge=0, le=70)` — pass attempts.
   - `completions: Series[int] = pa.Field(ge=0, le=60)` — completions.
   - `sacks: Series[int] = pa.Field(ge=0, le=15)` — sacks taken (for sack rate).

   Plus matching `Stat` enum entries (`PASSING_ATTEMPTS`, `COMPLETIONS`, `SACKS`). Same pattern as 2a's incremental extension.

4. **Shared helper extraction in `_rolling.py`**:
   - `trailing_n_share_in_group(weekly_stats, *, value_col, n=4, denom_filter=None)` — generalizes `wr.py`'s `_trailing_4_share_per_team`. Default behavior unchanged (share among same-position teammates). Optional `denom_filter` callable lets RB/TE compute share against a different group (e.g., RB target_share against all team pass-catchers).
   - `wr.py`'s `_trailing_4_share_per_team` becomes a one-line wrapper around the helper.

5. **Per-position tests** mirroring `test_wr.py` and `test_wr_leakage.py`:
   - Non-leakage: `test_qb.py`, `test_rb.py`, `test_te.py` (~6-8 tests each).
   - Leakage: `test_qb_leakage.py`, `test_rb_leakage.py`, `test_te_leakage.py` (5 tests each, one per input source).
   - Synthetic frames added to `tests/test_features/conftest.py` for each position (or split into per-position conftests if the file grows past ~600 lines — judgement call during implementation).

6. **End-to-end smoke test extension** — extend `tests/test_smoke_2a.py` to also build QB/RB/TE features (rename to `tests/test_smoke.py` since it now covers more than just 2a deliverables).

7. **Documentation updates** (per spec §10 pattern from 2a) — see §10.

### 2.2 Out of scope

- K and TST builders — TODO #10.
- Play-by-play ingest — TODO #3.
- Feature parquet storage — TODO #4.
- Model training, backtest harness — Plan 3+.

---

## 3. Per-position feature lists

Each schema is keyed by `(gsis_id, season, week, team, opponent)` and includes `depth_rank`, `is_home`, `roof_dome`, `implied_team_total`, `spread`, plus position-specific feature blocks below.

### 3.1 QB

**Receiving usage:** N/A.

**Passing usage (rolling):**
- `pass_attempts_per_game_l4` (from extended `attempts` column)
- `passing_yards_per_game_l4`
- `passing_tds_per_game_l4`
- `interceptions_per_game_l4`
- `sacks_per_game_l4`

**Season-to-date:**
- `passing_yards_per_game_std`

**Rushing usage (for rushing QBs):**
- `rushing_attempts_per_game_l4`
- `rushing_yards_per_game_l4`
- `rushing_qb` (boolean, `rushing_attempts_per_game_l4 ≥ 5.0`)

**Snap / role:**
- `snap_pct_l4` (always near 1.0 for the active starter; flags backup-QB scenarios)

**NGS passing (latest snapshot from prior week):**
- `aggressiveness_std`
- `completion_percentage_above_expectation_std`
- `avg_intended_air_yards_std`
- `avg_time_to_throw_std`

**Game environment:**
- `implied_team_total`, `spread`, `is_home`, `roof_dome` (same as WR)

**Opponent strength:**
- `opp_allowed_qb_fppg_l4`

### 3.2 RB

**Rushing usage (rolling):**
- `carries_per_game_l4`
- `rushing_yards_per_game_l4`
- `rushing_tds_per_game_l4`
- `rush_share_l4` (player's trailing-4 carries / team's RBs' trailing-4 carries)

**Receiving usage (for pass-catching RBs):**
- `targets_per_game_l4`
- `receptions_per_game_l4`
- `receiving_yards_per_game_l4`
- `target_share_l4` (player's trailing-4 targets / team's full pass-catching group's trailing-4 targets)

**Season-to-date:**
- `targets_per_game_std`

**Snap / role:**
- `snap_pct_l4` (key for committee-RB detection; near 0.7+ for workhorses, much lower for shared backfields)

**NGS rushing (latest snapshot from prior week):**
- `efficiency_std`
- `rush_yards_over_expected_per_att_std`
- `percent_attempts_gte_eight_defenders_std` (workload indicator: stacked boxes signal designed runner)

**Derived flags:**
- `passing_down_back` (boolean, `targets_per_game_l4 ≥ 4.0`)

**Game environment + opponent strength** as for WR: `implied_team_total`, `spread`, `is_home`, `roof_dome`, `opp_allowed_rb_fppg_l4`.

### 3.3 TE

**Receiving usage (rolling, like WR):**
- `targets_per_game_l4`
- `receptions_per_game_l4`
- `receiving_yards_per_game_l4`
- `receiving_tds_per_game_l4`
- `target_share_l4` (player's trailing-4 targets / team's full pass-catching group's trailing-4 targets — meaningful for TEs since usually only 1 fantasy-relevant TE per team, so same-position-share would always be ~1.0)

**Season-to-date:**
- `targets_per_game_std`

**Snap / role:**
- `snap_pct_l4`

**NGS receiving (latest snapshot from prior week):**
- `avg_separation_std`
- `avg_intended_air_yards_std`
- `avg_yac_above_expectation_std`

**Game environment + opponent strength** as for WR: `implied_team_total`, `spread`, `is_home`, `roof_dome`, `opp_allowed_te_fppg_l4`.

No derived boolean flag in v1 — defer "move TE" classification (uses NGS positional / route data) to a later refinement.

---

## 4. Schema additions to `schemas.py`

### 4.1 Extended `WeeklyStatsSchema`

Three new fields, placed alongside their conceptual neighbors:

```python
class WeeklyStatsSchema(pa.DataFrameModel):
    """Canonical weekly stats — what `ingest.weekly_stats` produces."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)
    passing_yards: Series[float] = pa.Field(ge=-100, le=800)
    passing_tds: Series[int] = pa.Field(ge=0, le=15)
    interceptions: Series[int] = pa.Field(ge=0, le=15)
    attempts: Series[int] = pa.Field(ge=0, le=70)            # NEW
    completions: Series[int] = pa.Field(ge=0, le=60)         # NEW
    sacks: Series[int] = pa.Field(ge=0, le=15)               # NEW
    rushing_yards: Series[float] = pa.Field(ge=-50, le=400)
    rushing_tds: Series[int] = pa.Field(ge=0, le=10)
    carries: Series[int] = pa.Field(ge=0, le=50)
    receptions: Series[int] = pa.Field(ge=0, le=30)
    receiving_yards: Series[float] = pa.Field(ge=-50, le=400)
    receiving_tds: Series[int] = pa.Field(ge=0, le=10)
    receiving_air_yards: Series[float] = pa.Field(ge=-50, le=400)
    targets: Series[int] = pa.Field(ge=0, le=30)
    fumbles_lost: Series[int] = pa.Field(ge=0, le=10)

    class Config:
        strict = "filter"
```

### 4.2 `Stat` enum additions

```python
PASSING_ATTEMPTS = "attempts"        # nfl_data_py uses bare `attempts`
COMPLETIONS = "completions"
SACKS = "sacks"
```

### 4.3 New feature schemas

#### `QbFeaturesSchema`

```python
class QbFeaturesSchema(pa.DataFrameModel):
    """QB feature DataFrame produced by `features.qb.build_qb_features`."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Passing usage (rolling)
    pass_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    passing_yards_per_game_l4: Series[float] = pa.Field(ge=0)
    passing_tds_per_game_l4: Series[float] = pa.Field(ge=0)
    interceptions_per_game_l4: Series[float] = pa.Field(ge=0)
    sacks_per_game_l4: Series[float] = pa.Field(ge=0)
    passing_yards_per_game_std: Series[float] = pa.Field(ge=0)

    # Rushing usage
    rushing_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_yards_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_qb: Series[bool]

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)

    # NGS passing (season-to-date snapshot from prior week)
    aggressiveness_std: Series[float] = pa.Field(nullable=True)
    completion_percentage_above_expectation_std: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards_std: Series[float] = pa.Field(nullable=True)
    avg_time_to_throw_std: Series[float] = pa.Field(nullable=True)

    # Game environment
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength proxy
    opp_allowed_qb_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
```

#### `RbFeaturesSchema`

```python
class RbFeaturesSchema(pa.DataFrameModel):
    """RB feature DataFrame produced by `features.rb.build_rb_features`."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Rushing usage (rolling)
    carries_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_yards_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_tds_per_game_l4: Series[float] = pa.Field(ge=0)
    rush_share_l4: Series[float] = pa.Field(ge=0, le=1)

    # Receiving usage
    targets_per_game_l4: Series[float] = pa.Field(ge=0)
    receptions_per_game_l4: Series[float] = pa.Field(ge=0)
    receiving_yards_per_game_l4: Series[float] = pa.Field(ge=0)
    target_share_l4: Series[float] = pa.Field(ge=0, le=1)
    targets_per_game_std: Series[float] = pa.Field(ge=0)

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)
    passing_down_back: Series[bool]

    # NGS rushing (season-to-date snapshot from prior week)
    efficiency_std: Series[float] = pa.Field(nullable=True)
    rush_yards_over_expected_per_att_std: Series[float] = pa.Field(nullable=True)
    percent_attempts_gte_eight_defenders_std: Series[float] = pa.Field(nullable=True)

    # Game environment
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength proxy
    opp_allowed_rb_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
```

#### `TeFeaturesSchema`

```python
class TeFeaturesSchema(pa.DataFrameModel):
    """TE feature DataFrame produced by `features.te.build_te_features`."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Receiving usage (rolling)
    targets_per_game_l4: Series[float] = pa.Field(ge=0)
    targets_per_game_std: Series[float] = pa.Field(ge=0)
    target_share_l4: Series[float] = pa.Field(ge=0, le=1)
    receptions_per_game_l4: Series[float] = pa.Field(ge=0)
    receiving_yards_per_game_l4: Series[float] = pa.Field(ge=0)
    receiving_tds_per_game_l4: Series[float] = pa.Field(ge=0)

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)

    # NGS receiving (season-to-date snapshot from prior week)
    avg_separation_std: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards_std: Series[float] = pa.Field(nullable=True)
    avg_yac_above_expectation_std: Series[float] = pa.Field(nullable=True)

    # Game environment
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength proxy
    opp_allowed_te_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
```

---

## 5. Helper additions in `_rolling.py`

### 5.1 Generalized share helper

Move and generalize `wr.py`'s `_trailing_4_share_per_team` into `_rolling.py` as a public helper. The denominator is "sum across all players in the input" — caller controls which group that is by pre-filtering. No internal position filtering, which keeps the helper simple and the call sites explicit:

```python
def trailing_n_share_in_group(
    weekly_stats: pd.DataFrame,
    *,
    value_col: str,
    n: int = 4,
) -> pd.DataFrame:
    """Per-player share of `value_col` within their team over the trailing N games.

    Numerator: each player's trailing-N sum of `value_col`.
    Denominator: sum across all players in `weekly_stats` on the same team
    (over the same trailing-N windows).

    Returns a frame keyed by `gsis_id` with column `share_l<n>` (`share_l4` when
    n=4, the default).

    The caller controls the share-group by pre-filtering `weekly_stats`:
    - WR target_share among the team's WRs:   filter input to `position == WR`.
    - RB target_share among the team's pass-catchers: filter input to
      `position in {WR, RB, TE}`, then keep only the RB rows from the output.
    - RB rush_share among the team's RBs:     filter input to `position == RB`.
    """
```

`wr.py` becomes a one-line consumer:

```python
target_share = trailing_n_share_in_group(ws_wr, value_col=Stat.TARGETS.value)
```

For RB target_share against full pass-catching group:

```python
ws_pass_catchers = ws[ws["position"].isin(["WR", "RB", "TE"])]
all_shares = trailing_n_share_in_group(ws_pass_catchers, value_col=Stat.TARGETS.value)
rb_target_share = all_shares.merge(rb_dc[["gsis_id"]], on="gsis_id")
```

TE target_share against full pass-catching group: identical shape, with `te_dc` instead of `rb_dc` for the post-filter.

RB rush_share within the team's RB group: input pre-filtered to `position == RB`.

### 5.2 Migration impact on `wr.py`

Replace the local `_trailing_4_share_per_team` with the import from `_rolling.py`. Behavior unchanged. Existing WR tests pass without modification.

### 5.3 Other helper additions

None. The 2a helpers (`last_n_per_group`, `per_game_rate`, `season_to_date_mean`, `opp_allowed_fppg`) cover everything QB/RB/TE need. `season_to_date_mean` and `per_game_rate` finally get used (they were unused in 2a — the reviewer flagged this as expected per the "pin helper API for 2b" intent).

---

## 6. Testing strategy

Same shape as 2a:

### 6.1 Per-position non-leakage tests (`tests/test_features/test_{qb,rb,te}.py`)

Each module has ~6-8 tests covering:
- Schema validation of output.
- One row per rostered player at the position.
- Trailing-4 rolling correctness on a synthetic player with known weekly numbers.
- Share math correctness (rush_share for RB, target_share for RB/TE).
- Position-specific flag correctness (rushing_qb threshold, passing_down_back threshold).
- Game environment join (`implied_team_total`, `is_home` correctness).
- Rookie zero-fill (player with no prior games gets zeros, no crash).

### 6.2 Per-position leakage tests (`tests/test_features/test_{qb,rb,te}_leakage.py`)

5 tests each, one per input source. Same strategy as `test_wr_leakage.py`:

1. Build features for `(season=2024, as_of_week=5)` against synthetic frames.
2. For each input frame independently, fabricate implausible rows for `week ≥ 5`.
3. Rebuild and assert byte-equal (`pd.testing.assert_frame_equal`) to baseline.

15 leakage tests total across the 3 positions.

### 6.3 Synthetic frames in `tests/test_features/conftest.py`

Add per-position fixtures mirroring 2a's (`wr_weekly_stats`, `wr_snap_counts`, etc.):

- `qb_weekly_stats`, `qb_snap_counts`, `qb_depth_charts`, `qb_ngs_passing`, `qb_schedules`
- `rb_weekly_stats`, `rb_snap_counts`, `rb_depth_charts`, `rb_ngs_rushing`, `rb_schedules`
- `te_weekly_stats`, `te_snap_counts`, `te_depth_charts`, `te_ngs_receiving`, `te_schedules`

Each fixture is small (~3 players × 8 weeks of synthetic data) with values designed for round-number rolling expectations (e.g., 12/10/8/6 attempts week 1-4 → mean 9.0).

If `conftest.py` grows past ~600 lines, split per position into `tests/test_features/conftest_qb.py` etc. via pytest-style fixture modules — judgement call during implementation.

### 6.4 Smoke-test extension

Rename `tests/test_smoke_2a.py` to `tests/test_smoke.py` (since it now covers more than just 2a). Extend to also build QB/RB/TE features for each rostered player from the same ingest output. Verifies all 4 builders round-trip cleanly through the partition layer.

### 6.5 Test budget

- ~7 (QB) + ~7 (RB) + ~7 (TE) = ~21 non-leakage tests.
- 5 × 3 = 15 leakage tests.
- Plus 2 schema-additions tests (one per new schema, validate-passes + reject-bad-value pair) = 6 schema tests.
- Plus 1-2 tests for the extended `WeeklyStatsSchema` (new columns persist, new Stat enum entries exist).
- Smoke-test extension: existing 1 test stays, just covers more.

Total: **~45 new tests** on top of 158 baseline → **~200 total**.

### 6.6 End-of-effort checklist (per CLAUDE.md)

Every PR run of:

- `pytest -v`
- `mypy src tests`
- `ruff check src tests`
- `ruff format --check src tests`
- `pytest -v -k "ingest or store or schemas"` (CLAUDE.md required for any schema/ingest touch)

---

## 7. Open questions deliberately deferred

- **`pass_attempts` vs `attempts`**: `nfl_data_py.import_weekly_data` uses bare `attempts` for pass attempts. We adopt that name to avoid a `_RENAME` hop — the schema field is `attempts`, the Stat enum entry is `PASSING_ATTEMPTS = "attempts"`. The verbose enum name disambiguates from rushing attempts (which is `Stat.CARRIES`).
- **`rushing_qb` and `passing_down_back` thresholds (5.0 and 4.0)**: hand-picked from rough heuristic. May want to tune later from data; not load-bearing for v1.
- **TE move/inline classification**: needs NGS positional data we don't currently extract. Defer to a later refinement.
- **Sack rate calculation (sacks / dropbacks)**: would need dropbacks = attempts + sacks. v1 just exposes `sacks_per_game_l4` raw; let the model layer compute rate if useful.
- **RB carries-per-snap / target-per-snap rate**: more sensitive than per-game means; defer until we see if the simple per-game means are insufficient at backtest time.

---

## 8. Risks

- **Pattern divergence across the 4 builders.** With WR already merged and 3 new builders landing simultaneously, there's risk of slight inconsistencies in helper usage / naming / dtype handling. Mitigation: review all 3 new files together in one PR; one task per position so reviewers see commits side-by-side.
- **Synthetic-fixture maintenance burden.** Each builder has its own ~5 fixtures. If a shared schema field is added later (similar to 2a's `targets`/`carries`/`receiving_air_yards`), every position fixture needs updating. Mitigation: factor out shared row-builder helpers within `conftest.py` if duplication exceeds two positions.
- **`trailing_n_share_in_group` API drift.** The new helper signature must match what wr.py needs (no behavior change there) AND what RB/TE need (new `denom_position_filter` param). Risk of subtle breakage in WR if migration is sloppy. Mitigation: WR tests run unchanged after migration; if they break, the migration is wrong.

---

## 9. What an MVP for 2b delivers

In order:

1. Fill in `<TBD-after-merge>` in `project_management.md` with `7926090` (2a merge commit).
2. Extend `WeeklyStatsSchema` with `attempts`, `completions`, `sacks` + matching `Stat` enum entries. Update `weekly_stats.py` `_KEEP` and dtype coercion. Update `fake_weekly_df` and `_good_weekly_stats` helpers.
3. Migrate `_trailing_4_share_per_team` from `wr.py` to `_rolling.py` as `trailing_n_share_in_group`. Update `wr.py` to consume the migrated helper.
4. Add `QbFeaturesSchema` to `schemas.py`.
5. Add `RbFeaturesSchema` to `schemas.py`.
6. Add `TeFeaturesSchema` to `schemas.py`.
7. Add synthetic QB/RB/TE fixtures to `tests/test_features/conftest.py`.
8. Implement `qb.py` + `test_qb.py`.
9. Implement `qb` leakage tests (`test_qb_leakage.py`).
10. Implement `rb.py` + `test_rb.py`.
11. Implement `rb` leakage tests.
12. Implement `te.py` + `test_te.py`.
13. Implement `te` leakage tests.
14. Re-export `build_qb_features`/`build_rb_features`/`build_te_features` from `src/projections/features/__init__.py`.
15. Extend smoke test (`tests/test_smoke_2a.py` → `tests/test_smoke.py`) to cover all 4 builders.
16. End-of-effort verification gate green.
17. `project_management.md` and `TODO.md` updated per §10.
18. PR opened against `main`.

Anything beyond this — K and DST builders, model training, feature parquet storage — is out of scope.

---

## 10. Documentation updates on merge

Lands the PM/TODO edits on the same PR as the implementation, per the 2a pattern.

### 10.1 `project_management.md` — decision-log additions

Append rows to the decision log (newest at top):

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-24 | K and DST split out into a future plan; 2b covers QB/RB/TE only | K needs FG-attempt data not in `WeeklyStatsSchema`; DST is team-level not player-level and needs play-by-play. Both should wait for the data they need rather than ship degraded v0 features |
| 2026-04-24 | All 4 position builders use parallel files (no WR/TE shared base) | Each position's feature list will diverge over time as we add play-by-play-derived features. Premature DRY hurts later. Shared logic lives in `_rolling.py` / `_opponent.py` |
| 2026-04-24 | One bundled PR for QB/RB/TE (not three per-position PRs) | Repetitive, interlinked work; reviewing all three together catches drift. Each position lands as its own commit inside the bundle for easy retrospection |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `attempts`, `completions`, `sacks` | QB feature builder needs these source columns. All three are present in raw `nfl_data_py.import_weekly_data` output. Same incremental pattern as 2a's extension for `targets`/`carries`/`receiving_air_yards` |
| 2026-04-24 | Migrate `_trailing_4_share_per_team` from `wr.py` to `_rolling.py` as `trailing_n_share_in_group` | RB needs target_share against the full pass-catching group (not just RBs); TE needs the same. Generalize once, in the shared helper module, rather than duplicate in three builders |
| 2026-04-24 | RB target_share denominator includes WR + RB + TE (full pass-catching group) | A workhorse RB getting 5 targets/game on a 30-target offense is meaningfully different from one getting 5 on a 20-target offense. Full-group denominator captures team passing volume, not just RB-on-RB share |
| 2026-04-24 | TE target_share denominator includes WR + RB + TE (full pass-catching group) | TEs usually have only one fantasy-relevant player per team, so same-position-share would be ~1.0 and useless. Full-group share captures meaningful gradient |
| 2026-04-24 | `rushing_qb` boolean threshold = 5.0 carries/game over trailing 4; `passing_down_back` = 4.0 targets/game | Rough heuristics from feel. Not load-bearing; revisit at backtest time if categorization matters |

### 10.2 `project_management.md` — status section update

Update Current Status:

```markdown
## Current status (as of 2026-04-24)

**Projections Core — Plan 2b (QB/RB/TE feature builders) merged to `main` at commit `<TBD-after-merge>`.**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Dev tooling merged via `feat/dev-tooling`.
- Plan 2a (Ingest expansion + WR feature builder) merged at `7926090`.

**Plan 2b delivered:**
- `build_qb_features`, `build_rb_features`, `build_te_features` — pure-function builders mirroring `build_wr_features`'s shape.
- Three new feature schemas (`QbFeaturesSchema`, `RbFeaturesSchema`, `TeFeaturesSchema`).
- `WeeklyStatsSchema` extended with `attempts`, `completions`, `sacks`.
- Generalized `trailing_n_share_in_group` helper in `_rolling.py` (migrated from `wr.py`'s local helper).
- ~45 new tests (~200 total). 5 leakage tests per position (15 new).
```

Update Next Action:

```markdown
## Next action

**Recommended: Plan 3 — Model A baseline + season aggregation + first-class backtest harness.**

All 4 offensive skill positions (QB/RB/WR/TE) now have feature builders. Plan 3 trains the v1 model per position, aggregates weekly outputs to season distributions (Monte Carlo with bye + availability), and stands up the backtest harness that gates future model changes.

K and DST builders (TODO #10) can land in parallel with Plan 3 — they're independent.
```

### 10.3 `TODO.md` — updates

- **Resolve TODO #2** (Plan 2b): mark partially complete (QB/RB/TE done in 2b); the K/DST portion split out into TODO #10.
- **Add TODO #10**:

```markdown
### 10. Plan 2c — K and DST feature builders

Both positions need data we don't currently ingest:

- **K**: spec calls for "recent FG distance distribution" and "opp redzone TD allowed %." Neither is in `WeeklyStatsSchema`. Need to ingest a new source covering FG attempts by distance and accuracy by range. `nfl_data_py.import_weekly_pfr_data` may have this — verify before designing.
- **DST**: team-level not player-level. Schema's primary key is `Team`, not `GsisId` — fundamentally different from the per-player pattern Plan 2a/2b established. Intended features (opp pass-block win rate, sack rate allowed, turnover-worthy throw rate) all need play-by-play (TODO #3).

Decision before brainstorming Plan 2c: do we ingest the missing data first (extending the ingest layer), or build degraded v0 K/DST features from `implied_team_total` alone? The latter is fast but creates a future rewrite; the former takes longer but yields the right shape.

Plan 3 (Model A baseline) doesn't depend on K/DST, so this can run in parallel.
```

### 10.4 Decision-log accuracy

If any of §10.1's design calls change during implementation (e.g., `trailing_n_share_in_group`'s API ends up needing a different shape, or QB ends up not needing `sacks` after all), the decision-log entries get edited to match what *actually* happened, not what we planned.
