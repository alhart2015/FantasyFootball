# Plan 2a — Ingest expansion + WR feature builder — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-24
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Parent spec:** `docs/superpowers/specs/2026-04-24-projections-core-design.md`
**Predecessor:** Plan 1 (Foundations) merged at commit `8f02a6c`

---

## 1. Overview

Plan 2a is the first half of the backlogged "Plan 2" from `project_management.md`. It expands the ingest layer to cover the four `nfl_data_py` sources the parent spec called out (schedules, snap_counts, depth_charts, NGS) and stands up the `src/projections/features/` module with one fully-built per-position feature builder — **WR** — to shake out the builder pattern (signatures, leakage prevention, schemas, tests) before copy-pasting it across the remaining positions in Plan 2b.

### 1.1 Why split Plan 2 into 2a + 2b

Plan 2 as originally backlogged combined four new ingest sources with six per-position feature builders. The ingest half is mechanical (a replay of `weekly_stats.py`'s pattern); the features half is greenfield design. Doing one position end-to-end first lets us validate the builder API on real data — pure-function vs. cached, schema strictness, leakage-test rigor, helper extraction — before propagating any wrong call across five more files.

**Plan 2a** = all four new ingest sources + WR feature builder + supporting `_rolling.py` / `_opponent.py` shared helpers.
**Plan 2b** (separate spec) = QB / RB / TE / K / DST feature builders, reusing helpers and the validated pattern from 2a.

### 1.2 Goals

- Ingest the remaining `nfl_data_py` sources we need for projections: **schedules** (with Vegas lines), **snap_counts**, **depth_charts**, **NGS** (passing / rushing / receiving).
- Stand up `src/projections/features/` with leakage-safe, pure-function builders.
- Deliver a fully-tested `build_wr_features` that consumes all five raw tables and produces a strictly-validated WR feature DataFrame.
- Establish reusable helpers (`_rolling.py`, `_opponent.py`) that 2b's other position builders will share.

### 1.3 Non-goals

- Other position feature builders — Plan 2b.
- Feature parquet storage / `refresh_features` CLI / feature manifest — deferred until backtest performance demands it (Plan 3+).
- A unified `refresh` CLI verb wrapping all ingest sources — Plan 4.
- Play-by-play ingest (`nfl_data_py.import_pbp_data`) — needed eventually for true opponent-adjusted EPA. v1 uses an `opp_allowed_fppg_l4` proxy.
- Model training, model interface, distribution layer changes — Plan 3+.
- Joint correlations across players — TODO #1.

---

## 2. Scope & deliverables

### 2.1 In scope

1. **Four new ingest modules** (one per source, plus the parameterized NGS module covering three stat types):
   - `src/projections/ingest/schedules.py`
   - `src/projections/ingest/snap_counts.py`
   - `src/projections/ingest/depth_charts.py`
   - `src/projections/ingest/ngs.py` (parameterized by `stat_type`)

   Each follows the `weekly_stats.py` template exactly: `_fetch_raw_*`, `_normalize_one_season`, `refresh_*`, schema validation at the boundary, `record_manifest`, `write_partition`. Idempotent: re-running a season overwrites that partition only.

2. **Schema additions to `src/projections/schemas.py`** (6 ingest + 1 feature schemas, plus `Stat` enum entries). Detail in §4.

3. **`src/projections/features/` module** (greenfield):
   - `features/__init__.py` — re-exports `build_wr_features`.
   - `features/_rolling.py` — shared trailing-window helpers.
   - `features/_opponent.py` — shared opponent-strength proxy.
   - `features/wr.py` — `build_wr_features(...)`.

   Pure-function, no parquet storage, no CLI verb, no manifest entry.

4. **Tests** — per-source ingest tests against fixture parquet, leakage / correctness tests for the WR builder, and one end-to-end smoke test. Detail in §6.

5. **Drive-by cleanups** carried over from `project_management.md`:
   - Move `_PYARROW_STR` from `weekly_stats.py` into `schemas.py` as a module constant.
   - Replace hard-coded `_INTEGER_STATS` list in scoring with a programmatic derivation from `StatLine` annotations.
   - Drop ingest helpers from `__all__` (only public refresh entry points remain exported).

6. **Documentation updates** (per CLAUDE.md "stay in sync" rule). On merge of 2a:
   - Append decision-log rows to `project_management.md` for the major design calls made here.
   - Update the "Current status" / "Next action" sections to reflect 2a complete and 2b queued.
   - Add follow-up entries to `TODO.md` for work explicitly punted from this plan.

   Concrete proposed entries are listed in §10.

### 2.2 Explicitly deferred to Plan 2b

Per-position feature builders for QB, RB, TE, K, DST. Helpers in `_rolling.py` and `_opponent.py` are designed in 2a so that 2b can consume them without revision.

### 2.3 Explicitly deferred to later plans

- Feature parquet storage (Plan 3+ if needed).
- `refresh` CLI verb (Plan 4).
- Play-by-play ingest, opponent-adjusted EPA (Plan 3+).
- Model training, backtest harness (Plan 3+).

---

## 3. Ingest source shape

The four new ingest modules conform to the existing `src/projections/ingest/weekly_stats.py` template. The `ngs` module is parameterized by `stat_type` and produces three distinct partition tables (one per stat type), so the table below has six rows even though there are only four modules:

| Source | nfl_data_py call | Partition path | Key columns | Notes |
|---|---|---|---|---|
| schedules | `import_schedules(years)` | `data/raw/schedules/season=YYYY/part.parquet` | `season, week, game_id, home_team, away_team, kickoff, spread_line, total_line, home_moneyline, away_moneyline, surface, roof, temp, wind` | Vegas lines populated. Team codes normalized via `normalize_team_code`. Future-week games may have NaN for kickoff / weather / line — schema marks those nullable. |
| snap_counts | `import_snap_counts(years)` | `data/raw/snap_counts/season=YYYY/part.parquet` | `gsis_id, season, week, team, opponent, position, offense_snaps, offense_pct, defense_snaps, defense_pct, st_snaps, st_pct` | Position-filtered to fantasy positions. Team codes normalized. |
| depth_charts | `import_depth_charts(years)` | `data/raw/depth_charts/season=YYYY/part.parquet` | `gsis_id, season, week, team, position, depth_team, depth_rank` | `depth_team` is the raw slot label (e.g., `WR1`, `LWR`); `depth_rank` is parsed numeric rank within the position group. Position-filtered. |
| ngs (passing) | `import_ngs_data("passing", years)` | `data/raw/ngs_passing/season=YYYY/part.parquet` | `gsis_id, season, week, team, position` + NGS passing columns (`avg_time_to_throw`, `avg_intended_air_yards`, `aggressiveness`, `completion_percentage_above_expectation`, …) | Returns season-to-date snapshots per week (naturally leakage-safe). |
| ngs (rushing) | `import_ngs_data("rushing", years)` | `data/raw/ngs_rushing/season=YYYY/part.parquet` | `gsis_id, season, week, team, position` + NGS rushing columns (`efficiency`, `rush_yards_over_expected`, `avg_time_to_los`, …) | Same shape; written by the same parameterized `refresh_ngs(stat_type=…)` function. |
| ngs (receiving) | `import_ngs_data("receiving", years)` | `data/raw/ngs_receiving/season=YYYY/part.parquet` | `gsis_id, season, week, team, position` + NGS receiving columns (`avg_separation`, `avg_intended_air_yards`, `percent_share_of_intended_air_yards`, `avg_yac_above_expectation`, …) | Consumed by the WR feature builder. |

All six new manifest tables (`schedules`, `snap_counts`, `depth_charts`, `ngs_passing`, `ngs_rushing`, `ngs_receiving`) are tracked alongside `weekly_stats` and `id_map` in `data/manifests/ingest_manifest.parquet`.

---

## 4. Schema additions to `schemas.py`

### 4.1 New pandera schemas

```python
class SchedulesSchema(pa.DataFrameModel):
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    game_id: Series[str]
    home_team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    away_team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    kickoff: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": "UTC", "unit": "us"}, nullable=True
    )
    spread_line: Series[float] = pa.Field(nullable=True)
    total_line: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    home_moneyline: Series[int] = pa.Field(nullable=True)
    away_moneyline: Series[int] = pa.Field(nullable=True)
    surface: Series[str] = pa.Field(nullable=True)
    roof: Series[str] = pa.Field(nullable=True)
    temp: Series[int] = pa.Field(nullable=True)
    wind: Series[int] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class SnapCountsSchema(pa.DataFrameModel):
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    offense_snaps: Series[int] = pa.Field(ge=0, le=200)
    offense_pct: Series[float] = pa.Field(ge=0, le=1)
    defense_snaps: Series[int] = pa.Field(ge=0, le=200)
    defense_pct: Series[float] = pa.Field(ge=0, le=1)
    st_snaps: Series[int] = pa.Field(ge=0, le=100)
    st_pct: Series[float] = pa.Field(ge=0, le=1)

    class Config:
        strict = "filter"


class DepthChartsSchema(pa.DataFrameModel):
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    depth_team: Series[str]
    depth_rank: Series[int] = pa.Field(ge=1, le=10)

    class Config:
        strict = "filter"


class NgsPassingSchema(pa.DataFrameModel):
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2016, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    avg_time_to_throw: Series[float] = pa.Field(nullable=True)
    avg_completed_air_yards: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards: Series[float] = pa.Field(nullable=True)
    avg_air_yards_differential: Series[float] = pa.Field(nullable=True)
    aggressiveness: Series[float] = pa.Field(nullable=True)
    max_completed_air_distance: Series[float] = pa.Field(nullable=True)
    avg_air_yards_to_sticks: Series[float] = pa.Field(nullable=True)
    completion_percentage: Series[float] = pa.Field(nullable=True)
    expected_completion_percentage: Series[float] = pa.Field(nullable=True)
    completion_percentage_above_expectation: Series[float] = pa.Field(nullable=True)
    avg_air_distance: Series[float] = pa.Field(nullable=True)
    max_air_distance: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class NgsRushingSchema(pa.DataFrameModel):
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2016, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    efficiency: Series[float] = pa.Field(nullable=True)
    percent_attempts_gte_eight_defenders: Series[float] = pa.Field(nullable=True)
    avg_time_to_los: Series[float] = pa.Field(nullable=True)
    rush_attempts: Series[int] = pa.Field(ge=0, nullable=True)
    rush_yards: Series[int] = pa.Field(nullable=True)
    expected_rush_yards: Series[float] = pa.Field(nullable=True)
    rush_yards_over_expected: Series[float] = pa.Field(nullable=True)
    avg_rush_yards: Series[float] = pa.Field(nullable=True)
    rush_yards_over_expected_per_att: Series[float] = pa.Field(nullable=True)
    rush_pct_over_expected: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class NgsReceivingSchema(pa.DataFrameModel):
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=2016, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    avg_cushion: Series[float] = pa.Field(nullable=True)
    avg_separation: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards: Series[float] = pa.Field(nullable=True)
    percent_share_of_intended_air_yards: Series[float] = pa.Field(nullable=True)
    receptions: Series[int] = pa.Field(ge=0, nullable=True)
    targets: Series[int] = pa.Field(ge=0, nullable=True)
    catch_percentage: Series[float] = pa.Field(nullable=True)
    yards: Series[int] = pa.Field(nullable=True)
    rec_touchdowns: Series[int] = pa.Field(ge=0, nullable=True)
    avg_yac: Series[float] = pa.Field(nullable=True)
    avg_expected_yac: Series[float] = pa.Field(nullable=True)
    avg_yac_above_expectation: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"


class WrFeaturesSchema(pa.DataFrameModel):
    """WR feature DataFrame produced by features.wr.build_wr_features.
    Schema enforced at the module boundary (CLAUDE.md typing posture)."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)

    # Receiving usage (rolling)
    targets_per_game_l4: Series[float] = pa.Field(ge=0)
    targets_per_game_std: Series[float] = pa.Field(ge=0)
    target_share_l4: Series[float] = pa.Field(ge=0, le=1)
    air_yards_share_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    receptions_per_game_l4: Series[float] = pa.Field(ge=0)
    receiving_yards_per_game_l4: Series[float] = pa.Field(ge=0)
    receiving_tds_per_game_l4: Series[float] = pa.Field(ge=0)

    # Rushing usage (Deebo / jet-sweep WRs)
    rushing_attempts_per_game_l4: Series[float] = pa.Field(ge=0)
    rushing_yards_per_game_l4: Series[float] = pa.Field(ge=0)
    designed_rusher: Series[bool]

    # Snap / role
    snap_pct_l4: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    depth_rank: Series[int] = pa.Field(ge=1, le=10, nullable=True)

    # NGS receiving (season-to-date snapshot from prior week)
    avg_separation_std: Series[float] = pa.Field(nullable=True)
    avg_intended_air_yards_std: Series[float] = pa.Field(nullable=True)
    percent_share_intended_air_yards_std: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    avg_yac_above_expectation_std: Series[float] = pa.Field(nullable=True)

    # Game environment (from schedules)
    implied_team_total: Series[float] = pa.Field(ge=0, le=60, nullable=True)
    spread: Series[float] = pa.Field(nullable=True)
    is_home: Series[bool]
    roof_dome: Series[bool]

    # Opponent strength (proxy)
    opp_allowed_wr_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
```

### 4.2 `Stat` enum additions

Add canonical column names referenced by the new feature/scoring code, so misspellings fail at type-check time:

```python
class Stat(StrEnum):
    # ... existing entries ...
    TARGETS = "targets"
    CARRIES = "carries"
    RECEIVING_AIR_YARDS = "receiving_air_yards"
    OFFENSE_PCT = "offense_pct"
```

### 4.3 `WeeklyStatsSchema` extension

The current `WeeklyStatsSchema` (foundations) ingests a minimal subset of `nfl_data_py.import_weekly_data` columns. The WR feature builder requires three additional columns that are present in the raw `nfl_data_py` output but not currently retained:

- `targets` (int) — for `targets_per_game_l4`, `targets_per_game_std`, `target_share_l4`.
- `receiving_air_yards` (float) — for properly-weighted `air_yards_share_l4` (sum across the window divided by team-total air yards, not naive average of per-game shares).
- `carries` (int) — for `rushing_attempts_per_game_l4` and the `designed_rusher` flag.

`WeeklyStatsSchema` adds three new fields:

```python
targets: Series[int] = pa.Field(ge=0, le=30)
receiving_air_yards: Series[float] = pa.Field(ge=-50, le=400)
carries: Series[int] = pa.Field(ge=0, le=50)
```

The existing `_KEEP` list in `src/projections/ingest/weekly_stats.py` is extended to include the three new column names (`carries`, `targets`, `receiving_air_yards` are the raw-side names — `nfl_data_py` returns them under those exact names, no rename needed). The existing `_normalize_one_season` int/float coercion lists are extended.

The existing `fake_weekly_df` fixture in `tests/test_ingest/conftest.py` is extended with the three new columns so existing tests continue to pass.

### 4.4 Module-constant promotion

`_PYARROW_STR = pd.StringDtype("pyarrow")` moves from `weekly_stats.py` to `schemas.py` as a module-level constant. All ingest modules import it from `schemas.py`. (Drive-by cleanup deferred from foundations review.)

---

## 5. `src/projections/features/` module

### 5.1 File layout

```
src/projections/features/
├── __init__.py          # re-exports build_wr_features
├── _rolling.py          # trailing-window helpers (private; shared with 2b)
├── _opponent.py         # opponent-allowed-fantasy-points proxy (private; shared with 2b)
└── wr.py                # build_wr_features
```

`_rolling.py` and `_opponent.py` are designed and tested in 2a even though only `wr.py` consumes them, because Plan 2b's other position builders will reuse them. Pinning the helper API here avoids retro-fitting it across five files later.

### 5.2 Public surface

```python
def build_wr_features(
    *,
    weekly_stats: pd.DataFrame,        # validated against WeeklyStatsSchema
    snap_counts: pd.DataFrame,         # validated against SnapCountsSchema
    depth_charts: pd.DataFrame,        # validated against DepthChartsSchema
    ngs_receiving: pd.DataFrame,       # validated against NgsReceivingSchema
    schedules: pd.DataFrame,           # validated against SchedulesSchema
    season: int,
    as_of_week: int,
) -> pd.DataFrame:                     # validated against WrFeaturesSchema
    ...
```

Keyword-only arguments — positional confusion across five same-shape DataFrames would be a high-cost bug.

The output has one row per `(gsis_id, season, week)` for every WR on a roster in week `as_of_week` of `season`. The row's `week` column equals `as_of_week`; the row represents "what we knew as of week `as_of_week`, scoring this WR's prospects for that week."

### 5.3 Leakage prevention contract

The first action of the function filters every input to rows the builder is permitted to see:

```python
prior_mask = lambda df: (df.season < season) | ((df.season == season) & (df.week < as_of_week))
ws  = weekly_stats[prior_mask(weekly_stats)]
sc  = snap_counts[prior_mask(snap_counts)]
ngs = ngs_receiving[prior_mask(ngs_receiving)]
dc  = depth_charts[(depth_charts.season == season) & (depth_charts.week == as_of_week)]
sch = schedules[(schedules.season == season) & (schedules.week == as_of_week)]
```

Rationale: weekly stats / snaps / NGS are *outcomes* — only prior weeks may contribute. Depth charts and schedules are *pre-game knowledge* — the target-week snapshot is allowed (kickoff time, opponent, betting line, depth designation are all pre-game).

Everything downstream of these filters operates on truncated frames. The leakage tests (§6) inject deliberately-implausible rows for `week ≥ as_of_week` and assert the output is byte-equal — separately for each input source so a leak surfaces with a precise failure.

### 5.4 Shared helpers

**`_rolling.py`:**

- `last_n_games(df, *, group_cols, sort_cols, n)` — rolling N-game window per group, counting *games played* (handles missed games correctly).
- `season_to_date(df, *, group_cols, sort_cols)` — running totals/means per group within a season.
- `per_game_rate(df, num_col, denom_col)` — safe division with explicit zero handling.

**`_opponent.py`:**

- `opp_allowed_fppg(weekly_stats, *, position, ruleset, n_weeks)` — per `(team, season, week)`, the opponent's average fantasy points allowed to `position` over the trailing `n_weeks`. Calls `scoring.score()` so we don't reimplement scoring math. Returns a frame keyed by `(season, week, team)` with column `opp_allowed_{pos}_fppg_l{n}`.

### 5.5 Why pure-function (no parquet store)

WR feature output for a season is ~100 WRs × 18 weeks ≈ 1.8K rows from ≤50K rows of input. Builds in milliseconds. Caching is premature until backtest scale (Plan 3+) demands it. If it ever does, adding storage is a localized change behind a `refresh_features` wrapper without touching the builder API.

---

## 6. Testing strategy

Following the existing pattern under `tests/projections/`. New test packages: extensions to `tests/projections/ingest/` and a new `tests/projections/features/`.

### 6.1 Per-ingest-source tests

One test module per ingest module (4 modules total; the `ngs` test is parameterized across the three stat types). Same shape for each:

- **Synthetic in-memory fixtures, not network or parquet snapshots.** Each source gets a `pytest.fixture` in `tests/test_ingest/conftest.py` returning a small synthetic `pd.DataFrame` matching `nfl_data_py.import_<source>`'s expected raw column shape. Tests monkeypatch `_fetch_raw_<source>` (per the existing `weekly_stats.py` pattern) to return the fixture. Matches the existing convention from foundations (`fake_id_map_df`, `fake_weekly_df`). CI never hits the network. Trade-off: synthetic fixtures don't catch real-world `nfl_data_py` API drift — that's tracked as a separate concern (see §7).
- **Round-trip:** `refresh_<source>(tmp_path, seasons=[2023])` writes a partition; reading it back and re-validating the schema returns the same DataFrame.
- **Schema enforcement:** mutate the fixture to violate the schema (wrong dtype, out-of-range value, unknown team code) and assert `_normalize_one_season` raises.
- **Idempotency:** call `refresh_<source>` twice for the same season; assert manifest has one row per `(table, season)` (last-write-wins on `fetched_at` + checksum) and the partition is overwritten cleanly.
- **Team-code normalization (schedules, snap_counts):** inject historical aliases (`STL`, `SD`, `OAK`, `WSH`, `JAX`, `LA`) and assert canonicalization to `Team` enum values.
- **Position filtering (snap_counts, depth_charts):** inject IDP/OL positions and assert they're dropped.
- **NGS specifically:** parameterize over `stat_type ∈ {passing, rushing, receiving}`; assert distinct partition paths (`raw/ngs_passing/...`, etc.) and three independent manifest entries.

### 6.2 Feature-builder tests (`tests/projections/features/test_wr.py`)

- **Schema validation:** output passes `WrFeaturesSchema.validate(...)`. Boundaries are real (column dtypes, ranges).
- **Shape:** for season 2023, week 6, with N WRs on rosters → output has exactly N rows.
- **Leakage test (load-bearing):** build features for `(season=2023, as_of_week=6)`. Then for each input frame independently, fabricate implausible rows for `week ≥ 6` (e.g., Tyreek Hill records 1000 receiving yards in week 7), rebuild, assert output byte-equal to original. Five separate assertions — leak in any one source surfaces with a precise failure.
- **Rolling correctness:** synthetic WR with known weekly target counts (8 / 6 / 10 / 4 / 12 over weeks 1-5); assert `targets_per_game_l4` for `as_of_week=6` equals `(6+10+4+12)/4 = 8.0`. One assertion per rolling helper.
- **Designed-rusher flag:** synthetic WR with 2.0 rush attempts/game over trailing 4 weeks → `designed_rusher == True`. Threshold edge cases at 1.5 ± epsilon.
- **Opponent-allowed proxy:** synthetic 2-team season; team A allows 25 / 20 / 15 / 10 WR fantasy points/game over weeks 1-4. For team B's WRs in week 5, `opp_allowed_wr_fppg_l4 == 17.5`.
- **Missing data graceful handling:** rookie WR with no prior games → all `*_l4` columns are 0 (or NaN where schema permits), no crash, schema passes.
- **Game environment join:** WR's row carries the correct `implied_team_total` (from `total_line` and `spread_line`), `is_home` matches the schedule, `roof_dome` is `True` for indoor venues.

### 6.3 Integration smoke test (`tests/projections/test_smoke_2a.py`)

End-to-end against fixtures only:

1. `refresh_weekly_stats`, `refresh_schedules`, `refresh_snap_counts`, `refresh_depth_charts`, `refresh_ngs(stat_type='receiving')` against `tmp_path`.
2. `read_partition` each table back.
3. `build_wr_features(...)` for `season=2023, as_of_week=6`.
4. Assert output has ≥1 row, schema validates, no NaNs in non-nullable columns.

Catches integration gaps that per-module tests miss (partition path mismatches between `write_partition` and `read_partition`, manifest update bugs, schema dtype drift between ingest output and feature input).

### 6.4 Drive-by cleanup verification

- `_PYARROW_STR` in `schemas.py`: existing `weekly_stats` tests pass unchanged.
- Programmatic `_INTEGER_STATS` from `StatLine` annotations: existing scoring tests pass unchanged.
- Trimmed ingest `__all__`: `mypy` and `ruff` clean.

### 6.5 Test budget

Roughly 4 ingest test modules (~6-8 tests each, with the parameterized `ngs` test covering three stat types ≈ 30-35 total ingest tests) + 1 feature test module (~10-12 tests) + 1 smoke test ≈ **~45-50 new tests**, putting total project tests at ~135-140.

### 6.6 End-of-effort checklist (per CLAUDE.md)

Every PR run of:

- `pytest -v`
- `mypy src tests`
- `ruff check src tests`
- `ruff format --check src tests`
- `pytest -v -k "ingest or store or schemas"` (this plan touches every ingest seam)

---

## 7. Open questions deliberately deferred

These are documented decisions to revisit, not blockers for 2a:

- **NGS missing-data policy.** When a WR doesn't meet NGS qualifying thresholds in a given week, columns are NaN. v1: leave NaN; schema permits it; downstream model handles via imputation in Plan 3. Alternative (forward-fill from last qualifying week) deferred until we see how often it matters in real data.
- **Depth chart slot-label parsing.** `nfl_data_py` mixes conventions across seasons (alignment-based `LWR`/`RWR`/`SWR` vs. rank-based `WR1`/`WR2`/`WR3`). v1 parser extracts the trailing digit when present; falls back to `1` for unrankable labels with a warning. If this bites Plan 3, we revisit.
- **Mid-season position changes.** A player listed as TE in `weekly_stats` may appear as FB in `depth_charts`. Joins use `gsis_id` only (not `position`); WR builder filters `weekly_stats.position == "WR"`, so a player who switches to RB mid-season correctly drops out. Behavior asserted in tests.
- **Vegas line provenance.** `import_schedules` returns *closing* lines. For a backtest harness this is good enough; if Plan 5 ever projects pre-week selections, we revisit (would need opening or week-of lines from a separate source).
- **Schedule pre-population for future weeks.** `import_schedules` returns the full season schedule including unplayed weeks (NaN for game-day weather, possibly TBD kickoff). Schemas mark those columns nullable; we ingest the full schedule.
- **`nfl_data_py` API drift detection.** Synthetic fixtures (§6.1) don't catch column renames or type changes in real `nfl_data_py` output. Mitigation strategy: an opt-in `@pytest.mark.network` smoke test (one per ingest source) that hits the live API, fetches a tiny slice (e.g., 1 week of 2023), and asserts the column set matches the schema's expectation. Marked `skip` by default; run manually after `nfl_data_py` version bumps. **Building these network smoke tests is deferred to a future task; not in 2a's scope.** Captured as TODO #8.

---

## 8. Risks

- **`nfl_data_py` API surface drift.** The library is community-maintained and occasionally renames columns between releases. Version is pinned in `pyproject.toml`; bumps are explicit, gated by green tests.
- **Synthetic-fixture blindness to API drift.** Synthetic in-memory fixtures match what we *expect* `nfl_data_py` to return, not what it *actually* returns. If a column is renamed, dropped, or retyped between versions, our tests stay green but production ingest breaks. Mitigation: opt-in network smoke tests on version bump (see §7, TODO #8).
- **NGS coverage starts in 2016.** Anything earlier has no NGS. Schemas reflect this (`ge=2016` on NGS season fields). Backtest in Plan 3 will need to handle the discontinuity (restrict training to ≥2016, or train two model variants).

---

## 9. What an MVP for 2a delivers

In order:

1. Schema additions to `schemas.py` (6 new ingest schemas + extension of `WeeklyStatsSchema` with `targets`/`receiving_air_yards`/`carries` + `WrFeaturesSchema` + `Stat` enum entries + `_PYARROW_STR` constant).
2. Four new ingest modules (`schedules.py`, `snap_counts.py`, `depth_charts.py`, `ngs.py`), each with the `weekly_stats.py`-template structure, schema validation at the boundary, and idempotent partition writes. The `ngs.py` module is parameterized by `stat_type` and produces three partition tables.
3. Per-source ingest tests against checked-in fixture parquet (no network).
4. `src/projections/features/` package with `_rolling.py`, `_opponent.py`, `wr.py`, and `__init__.py`.
5. `build_wr_features(...)` consuming all five raw tables, returning a `WrFeaturesSchema`-validated DataFrame.
6. Feature tests: schema, shape, leakage (one per input source), rolling correctness, opponent proxy, edge cases.
7. End-to-end smoke test (`test_smoke_2a.py`) wiring ingest → features.
8. Drive-by cleanups landed (`_PYARROW_STR`, programmatic `_INTEGER_STATS`, ingest `__all__`).
9. End-of-effort checklist green: `pytest`, `mypy --strict`, `ruff check`, `ruff format --check`.
10. `project_management.md` and `TODO.md` updated per §10 of this spec.
11. PR opened against `main`.

Anything beyond this — other position builders, feature parquet storage, CLI verbs, model training — is out of scope.

---

## 10. Documentation updates on merge

The implementation plan's final task lands these edits to `project_management.md` and `TODO.md` on the same PR. Listing them here so the design captures *why* each entry exists, not just *that* it exists.

### 10.1 `project_management.md` — decision-log additions

Append these rows to the decision log (newest at top of the table):

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-24 | Split Plan 2 into 2a (ingest expansion + WR feature builder) and 2b (QB / RB / TE / K / DST feature builders) | Validate the feature-builder pattern end-to-end on one position before copy-pasting across five files; isolate ingest (mechanical) from features (greenfield design) |
| 2026-04-24 | WR is the first end-to-end position | Exercises every new ingest source (snap_counts, depth_charts, NGS receiving) in one builder; surfaces design issues before propagating to other positions |
| 2026-04-24 | Feature builders are pure functions in 2a — no parquet storage | Output is small (~1.8K rows/season for WR) and computes in milliseconds; defer caching until backtest performance demands it (Plan 3+) |
| 2026-04-24 | Ingest all three NGS stat types (passing, rushing, receiving) in 2a, even though only NGS receiving is consumed by WR | The hard part of NGS ingest is the snapshot/partition decision; make it once across all three rather than three times |
| 2026-04-24 | Opponent strength via `opp_allowed_fppg_l4` proxy in 2a, not play-by-play EPA | True EPA needs play-by-play ingest (separate concern, deferred); the FPPG-allowed proxy is sufficient for v1 baseline |
| 2026-04-24 | Shared `_rolling.py` and `_opponent.py` helpers built and tested in 2a | Pin helper API on the first builder so 2b's five other builders consume a stable contract |
| 2026-04-24 | Schedule ingest captures Vegas lines (spread, total, moneyline) | "Implied team total" is a load-bearing feature for every offensive position |
| 2026-04-24 | Drive-by cleanups (`_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, ingest `__all__`) folded into 2a | We're touching every ingest module anyway; cheaper to clean up once than across two PRs |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `targets`, `receiving_air_yards`, `carries` | Discovered during plan-writing: WR feature builder needs these source columns and the foundations-era schema didn't include them. All three are present in raw `nfl_data_py.import_weekly_data` output |
| 2026-04-24 | Test fixtures are synthetic in-memory `pd.DataFrame`s, not real-data parquet snapshots | Matches existing convention from foundations (`fake_weekly_df` etc.); simpler maintenance; `nfl_data_py` API drift is handled separately by opt-in network smoke tests (TODO #8) |

### 10.2 `project_management.md` — status section update

Replace the "Current status" / "Next action" sections to reflect:

- Plan 1 (Foundations) and dev tooling: merged (unchanged).
- Plan 2a (this plan): complete; commit hash + summary of what landed.
- Next action: Plan 2b — per-position feature builders for QB / RB / TE / K / DST, reusing 2a's helpers and validated WR pattern.

### 10.3 `TODO.md` — follow-up additions

Add as new numbered items:

- **#2: Plan 2b — remaining position feature builders.** QB, RB, TE, K, DST, each consuming the validated `wr.py` pattern, `_rolling.py`, and `_opponent.py` helpers. Each gets its own pandera schema (`QbFeaturesSchema`, etc.). One PR per position or one bundled — TBD when 2b is brainstormed.
- **#3: Play-by-play ingest (`nfl_data_py.import_pbp_data`).** Required for true opponent-adjusted EPA features. Defer until Plan 3 backtest reveals whether the FPPG-allowed proxy is good enough; if not, ingest PBP and add EPA-derived opponent features in a focused plan.
- **#4: Decide feature parquet storage during Plan 3.** Gated on backtest performance: if a single training pass takes >~30s recomputing features, add `data/features/{position}/...` storage and a `refresh_features` CLI verb; otherwise stay pure-function.
- **#5: NGS missing-data forward-fill policy.** v1 leaves NaN. Revisit after a notebook investigation against a recent season quantifying how often qualifying-threshold misses happen and whether forward-fill changes feature distributions materially.
- **#6: Opening / week-of Vegas line source.** `import_schedules` returns *closing* lines. Closing is fine for backtest. Only worth pursuing if Plan 5 ever projects pre-week selections (e.g., DFS workflow uses lines that change through the week).
- **#7: Depth chart slot-label parser refinement.** v1 extracts the trailing digit from labels like `WR1`, falling back to `1` for unrankable labels (`LWR`/`RWR`/`SWR`) with a warning. If Plan 3 model fitting shows depth_rank is noisy or wrong, build a richer parser using alignment + rank.
- **#8: Build opt-in `nfl_data_py` API-drift smoke tests.** One per ingest source, marked `@pytest.mark.network`, skipped by default. Hits the live API, fetches a tiny slice (e.g., 1 week of 2023), asserts the column set matches the schema. Run manually after `nfl_data_py` version bumps. Document the run-after-bump step in `CONTRIBUTING.md`. The synthetic in-memory fixtures used by 2a's CI tests don't catch API drift on their own.

### 10.4 Decision-log accuracy

If during execution we revise any of the design calls in §10.1 (e.g., we find feature builders need a thin cache after all, or NGS ingest can't be parameterized cleanly), the decision-log entries get updated to reflect what *actually* happened, not what we planned. The spec is fixed once approved; the decision log is living.
