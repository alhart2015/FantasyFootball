# Plan 2a — Ingest expansion + WR feature builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four remaining `nfl_data_py` ingest sources (schedules with Vegas lines, snap_counts, depth_charts, NGS × 3 stat types), extend `WeeklyStatsSchema` with `targets`/`receiving_air_yards`/`carries`, and stand up `src/projections/features/` with one fully-built per-position builder (WR) plus shared `_rolling.py`/`_opponent.py` helpers. Validates the feature-builder pattern end-to-end before Plan 2b copy-pastes across QB/RB/TE/K/DST.

**Architecture:** TDD throughout, following the foundations conventions. Each new ingest module mirrors `weekly_stats.py` (private `_fetch_raw_*`, `_normalize_one_season`, public `refresh_*`). Synthetic in-memory `pd.DataFrame` fixtures in `tests/test_ingest/conftest.py` (no network in CI). Pure-function feature builder — no parquet storage in 2a. Shared rolling/opponent helpers built in 2a even though only WR consumes them, so 2b inherits a stable contract. 19 sequential tasks, each committable independently.

**Tech Stack:** Python 3.11+, `pandas`, `pyarrow`, `nfl_data_py`, `pydantic>=2`, `pandera`, `pytest`, `mypy --strict`, `ruff`. Spec at `docs/superpowers/specs/2026-04-24-plan-2a-ingest-and-wr-features-design.md`.

**Working directory:** `C:\Users\alden\FantasyFootball\.worktrees\feat-plan-2a-ingest-and-wr-features` (branch `feat/plan-2a-ingest-and-wr-features`). Activate venv: `. .venv/Scripts/activate`. If `.venv` doesn't exist in the worktree: `python -m venv .venv && . .venv/Scripts/activate && pip install -e ".[dev]"`.

---

## File Structure (created/modified by this plan)

```
src/projections/
├── schemas.py                                          # Tasks 1, 2, 3, 4, 5, 6, 7: extend WeeklyStatsSchema, add 6 new ingest schemas + WrFeaturesSchema + Stat enum entries + _PYARROW_STR constant
├── scoring/
│   └── score_distribution.py                           # Task 1: programmatic _INTEGER_STATS from StatLine annotations
├── ingest/
│   ├── __init__.py                                     # Task 1: trim __all__; Tasks 9-12: re-export new refresh_* functions
│   ├── weekly_stats.py                                 # Tasks 1, 2: drop local _PYARROW_STR import; extend _KEEP for new columns
│   ├── schedules.py                                    # Task 9 (new)
│   ├── snap_counts.py                                  # Task 10 (new)
│   ├── depth_charts.py                                 # Task 11 (new)
│   └── ngs.py                                          # Task 12 (new, parameterized by stat_type)
└── features/                                           # Phase 4 (new package)
    ├── __init__.py                                     # Task 15
    ├── _rolling.py                                     # Task 13
    ├── _opponent.py                                    # Task 14
    └── wr.py                                           # Task 15

tests/
├── test_schemas/
│   ├── test_dataframe_schemas.py                       # Tasks 2-7: extend with new schema cases
│   └── test_other_enums.py                             # Task 2: extend with new Stat enum cases
├── test_scoring/
│   └── test_score_distribution.py                      # Task 1: cover programmatic _INTEGER_STATS
├── test_ingest/
│   ├── conftest.py                                     # Task 2: extend fake_weekly_df; Task 8: add 6 new synthetic fixtures
│   ├── test_weekly_stats.py                            # Task 2: assert new columns persist
│   ├── test_schedules.py                               # Task 9 (new)
│   ├── test_snap_counts.py                             # Task 10 (new)
│   ├── test_depth_charts.py                            # Task 11 (new)
│   └── test_ngs.py                                     # Task 12 (new, parameterized)
├── test_features/                                      # Phase 4 (new package)
│   ├── __init__.py                                     # Task 13
│   ├── conftest.py                                     # Task 13 (shared synthetic frames for feature tests)
│   ├── test_rolling.py                                 # Task 13
│   ├── test_opponent.py                                # Task 14
│   ├── test_wr.py                                      # Task 15
│   └── test_wr_leakage.py                              # Task 16
└── test_smoke_2a.py                                    # Task 17 (end-to-end ingest → features)

docs/superpowers/plans/2026-04-24-plan-2a-ingest-and-wr-features.md  # this file (created when plan is committed)
project_management.md                                   # Task 18: append decision-log rows + status update
TODO.md                                                 # Task 18: append items #2-#8
```

No new dependencies. No new top-level files outside what's listed above.

**File-count discipline:** every task touches ≤ 5 files (per CLAUDE.md PHASED EXECUTION rule).

---

## Sanity-check before starting

Before Task 1, verify the worktree environment:

```bash
cd "/c/Users/alden/FantasyFootball/.worktrees/feat-plan-2a-ingest-and-wr-features"
git branch --show-current     # → feat/plan-2a-ingest-and-wr-features
. .venv/Scripts/activate      # or create per header instructions
pytest -v                     # → 89 passing, no failures (foundations baseline)
mypy src tests                # → zero violations
ruff check src tests          # → zero violations
```

If the baseline isn't green, fix it before adding new tasks on top.

---

## Phase 1 — Foundations: drive-bys + schemas

### Task 1: Drive-by cleanups (3 sub-cleanups bundled)

Three foundations-review cleanups that don't fit anywhere else and are cheap to do together. We're touching `schemas.py`, `weekly_stats.py`, `score_distribution.py`, and `ingest/__init__.py` — small surgical edits, no new behavior.

**Files:**
- Modify: `src/projections/schemas.py` (add `_PYARROW_STR` module constant)
- Modify: `src/projections/ingest/weekly_stats.py` (drop local `_PYARROW_STR`, import from `schemas`)
- Modify: `src/projections/scoring/score_distribution.py` (derive `_INTEGER_STATS` programmatically from `StatLine` annotations)
- Modify: `src/projections/ingest/__init__.py` (drop manifest helpers from `__all__`)
- Test: existing `tests/test_ingest/test_weekly_stats.py`, `tests/test_scoring/test_score_distribution.py` — must continue to pass unchanged

- [ ] **Step 1: Re-read the four files**

```bash
cat src/projections/schemas.py | head -15
cat src/projections/ingest/weekly_stats.py | head -25
cat src/projections/scoring/score_distribution.py | grep -n "_INTEGER_STATS" -A 15 | head -25
cat src/projections/ingest/__init__.py
```

Confirm the current state matches the foundations baseline.

- [ ] **Step 2: Add `_PYARROW_STR` to `schemas.py`**

In `src/projections/schemas.py`, immediately after the `import` block (after `from pydantic import ...`), add:

```python
# Module constant: pyarrow-backed nullable string dtype.
# Used by every ingest module that needs to satisfy a pandera Series[str] field
# (object-dtype + plain strings will fail validation in pandera 0.31+).
_PYARROW_STR: Final = pd.StringDtype("pyarrow")
```

- [ ] **Step 3: Update `weekly_stats.py` to import the constant**

In `src/projections/ingest/weekly_stats.py`:

```python
# Replace this line:
from projections.schemas import Position, WeeklyStatsSchema, normalize_team_code

# With this:
from projections.schemas import Position, WeeklyStatsSchema, _PYARROW_STR, normalize_team_code
```

And delete the local definition at module scope:

```python
# DELETE this line:
_PYARROW_STR = pd.StringDtype("pyarrow")  # match Task 13 / pandera Series[str] expectation
```

- [ ] **Step 4: Make `_INTEGER_STATS` programmatic in `score_distribution.py`**

Replace the hard-coded `_INTEGER_STATS` block with a derivation from `StatLine`'s pydantic field annotations. After the `from projections.scoring.score import StatLine, score` line (which is the last import), add:

```python
def _derive_integer_stats() -> frozenset[Stat]:
    """Programmatically determine which Stat enum members map to integer
    fields on StatLine. Single source of truth: if StatLine adds a new int
    field, this set updates without manual edits."""
    int_field_names = {
        name for name, field in StatLine.model_fields.items() if field.annotation is int
    }
    return frozenset(stat for stat in Stat if stat.value in int_field_names)


_INTEGER_STATS: frozenset[Stat] = _derive_integer_stats()
```

Delete the original hard-coded `_INTEGER_STATS = frozenset({Stat.PASSING_TDS, ...})` block.

- [ ] **Step 5: Trim `ingest/__init__.py` `__all__`**

In `src/projections/ingest/__init__.py`, remove the manifest helpers from `__all__` and from imports — they're internal to the ingest layer:

```python
"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.id_map import build_id_map
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "refresh_weekly_stats",
]
```

(Internal modules like `weekly_stats.py` continue to import directly from `projections.ingest.manifest` — they just don't get re-exported at the package level.)

- [ ] **Step 6: Run existing tests to confirm zero regression**

```bash
pytest -v tests/test_ingest tests/test_scoring tests/test_schemas
```

Expected: same number of passing tests as baseline (no failures, no new tests yet — this task is pure refactor).

If `test_score_distribution.py` references `_INTEGER_STATS` directly via `from projections.scoring.score_distribution import _INTEGER_STATS`, the import still works (the name is preserved). If a test imported manifest helpers via `from projections.ingest import compute_checksum`, fix the import to `from projections.ingest.manifest import compute_checksum`.

- [ ] **Step 7: Run the full quality gate**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

All must be green.

- [ ] **Step 8: Commit**

```bash
git add src/projections/schemas.py src/projections/ingest/weekly_stats.py src/projections/scoring/score_distribution.py src/projections/ingest/__init__.py
git commit -m "refactor: consolidate _PYARROW_STR, derive _INTEGER_STATS, trim ingest __all__

Three drive-by cleanups from foundations review:
- _PYARROW_STR moves from weekly_stats.py to schemas.py as module constant
- _INTEGER_STATS in score_distribution.py is now derived from StatLine field
  annotations rather than hard-coded
- ingest/__init__.py drops manifest helpers from __all__ (internal only)

No behavior change. All existing tests pass."
```

---

### Task 2: Extend `WeeklyStatsSchema` with `targets`, `receiving_air_yards`, `carries`

The WR feature builder needs three columns the foundations-era `WeeklyStatsSchema` doesn't include. All three are present in raw `nfl_data_py.import_weekly_data` output under those exact names — no rename required.

**Files:**
- Modify: `src/projections/schemas.py` (extend `WeeklyStatsSchema`, add 4 new `Stat` enum entries: `TARGETS`, `CARRIES`, `RECEIVING_AIR_YARDS`, `OFFENSE_PCT`)
- Modify: `src/projections/ingest/weekly_stats.py` (extend `_KEEP`, extend int/float coercion lists)
- Modify: `tests/test_ingest/conftest.py` (extend `fake_weekly_df` with the 3 new columns)
- Modify: `tests/test_ingest/test_weekly_stats.py` (add one test asserting the new columns persist with correct dtypes)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (add assertions for the 3 new schema fields if the existing tests reference column lists)

- [ ] **Step 1: Re-read `schemas.py` `WeeklyStatsSchema` and `Stat` enum**

```bash
grep -n "class WeeklyStatsSchema" -A 25 src/projections/schemas.py
grep -n "class Stat" -A 20 src/projections/schemas.py
```

Note the current field set; the new fields go alongside (in the natural place — receiving fields next to other receiving, rushing fields next to other rushing).

- [ ] **Step 2: Add the 3 new Stat enum entries**

In `src/projections/schemas.py`, append to the `Stat` enum (alphabetical by enum name within each conceptual group is fine; preferred placement near related stats):

```python
class Stat(StrEnum):
    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_tds"
    INTERCEPTIONS = "interceptions"
    PASSING_2PT = "passing_2pt_conversions"
    RUSHING_YARDS = "rushing_yards"
    RUSHING_TDS = "rushing_tds"
    RUSHING_2PT = "rushing_2pt_conversions"
    CARRIES = "carries"
    RECEPTIONS = "receptions"
    RECEIVING_YARDS = "receiving_yards"
    RECEIVING_TDS = "receiving_tds"
    RECEIVING_2PT = "receiving_2pt_conversions"
    RECEIVING_AIR_YARDS = "receiving_air_yards"
    TARGETS = "targets"
    FUMBLES_LOST = "fumbles_lost"
    RETURN_TDS = "return_tds"
    OFFENSE_PCT = "offense_pct"
```

(`OFFENSE_PCT` references the snap-counts column name; it's reserved here so future code that constructs feature column references via `Stat.OFFENSE_PCT.value` doesn't have to use a string literal.)

- [ ] **Step 3: Extend `WeeklyStatsSchema` with the 3 new fields**

In the `WeeklyStatsSchema` class (after `receiving_tds`, before `fumbles_lost`):

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

- [ ] **Step 4: Extend `_KEEP` and dtype coercion in `weekly_stats.py`**

In `src/projections/ingest/weekly_stats.py`:

```python
_KEEP = [
    "gsis_id",
    "season",
    "week",
    "position",
    "team",
    "opponent",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "carries",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "targets",
    "fumbles_lost",
]
```

Extend the int coercion tuple in `_normalize_one_season`:

```python
for int_col in (
    "passing_tds",
    "interceptions",
    "rushing_tds",
    "carries",
    "receptions",
    "receiving_tds",
    "targets",
    "fumbles_lost",
):
    if int_col in df.columns:
        df[int_col] = df[int_col].fillna(0).astype(int)
```

Extend the float coercion tuple:

```python
for float_col in ("passing_yards", "rushing_yards", "receiving_yards", "receiving_air_yards"):
    if float_col in df.columns:
        df[float_col] = df[float_col].fillna(0.0).astype(float)
```

- [ ] **Step 5: Extend `fake_weekly_df` fixture**

In `tests/test_ingest/conftest.py`, add the 3 new columns to `fake_weekly_df`:

```python
@pytest.fixture
def fake_weekly_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_weekly_data([2024])` — 2 player-weeks."""
    return pd.DataFrame(
        {
            "player_id": ["00-0036322", "00-0034857"],
            "season": [2024, 2024],
            "week": [3, 3],
            "position": ["WR", "QB"],
            "recent_team": ["MIN", "KC"],
            "opponent_team": ["HOU", "ATL"],
            "passing_yards": [0.0, 286.0],
            "passing_tds": [0, 2],
            "interceptions": [0, 1],
            "rushing_yards": [0.0, 12.0],
            "rushing_tds": [0, 0],
            "carries": [0, 3],
            "receptions": [9, 0],
            "receiving_yards": [110.0, 0.0],
            "receiving_tds": [1, 0],
            "receiving_air_yards": [145.0, 0.0],
            "targets": [12, 0],
            "fumbles_lost": [0, 0],
        }
    )
```

- [ ] **Step 6: Write a failing test asserting new columns persist**

In `tests/test_ingest/test_weekly_stats.py`, append:

```python
def test_refresh_weekly_stats_persists_new_columns(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Targets, carries, receiving_air_yards must round-trip through ingest."""
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    assert "targets" in df.columns
    assert "carries" in df.columns
    assert "receiving_air_yards" in df.columns
    # WR row: 12 targets, 0 carries, 145 air yards
    wr_row = df[df["gsis_id"] == "00-0036322"].iloc[0]
    assert int(wr_row["targets"]) == 12
    assert int(wr_row["carries"]) == 0
    assert float(wr_row["receiving_air_yards"]) == 145.0
    # QB row: 0 targets, 3 carries
    qb_row = df[df["gsis_id"] == "00-0034857"].iloc[0]
    assert int(qb_row["carries"]) == 3
```

- [ ] **Step 7: Run the new test, verify it currently passes (because we already updated the schema and fixture)**

```bash
pytest -v tests/test_ingest/test_weekly_stats.py::test_refresh_weekly_stats_persists_new_columns
```

Expected: PASS. (If FAIL with "unknown column" or "schema mismatch", revisit Steps 3-5 — likely a typo in the column name or the schema field.)

- [ ] **Step 8: Run the full ingest + schema test suites — confirm no regressions**

```bash
pytest -v tests/test_ingest tests/test_schemas
```

All previous tests must still pass. The new column count flows through `_KEEP`, the schema, and the fixture coherently.

- [ ] **Step 9: Run the quality gate**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

- [ ] **Step 10: Commit**

```bash
git add src/projections/schemas.py src/projections/ingest/weekly_stats.py tests/test_ingest/conftest.py tests/test_ingest/test_weekly_stats.py
git commit -m "feat(schemas): extend WeeklyStatsSchema with targets, carries, receiving_air_yards

WR feature builder needs these source columns. All three are present in
raw nfl_data_py.import_weekly_data output under those exact names. Adds
matching Stat enum entries (TARGETS, CARRIES, RECEIVING_AIR_YARDS).

fake_weekly_df fixture extended; existing weekly_stats tests pass
unchanged."
```

---

### Task 3: Add `SchedulesSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add `SchedulesSchema` class)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (add validate-passes / validate-fails tests for the new schema)

- [ ] **Step 1: Write failing schema tests**

In `tests/test_schemas/test_dataframe_schemas.py`, append:

```python
def test_schedules_schema_accepts_valid_frame() -> None:
    from projections.schemas import SchedulesSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 3],
            "game_id": pd.array(["2024_03_KC_ATL", "2024_03_MIN_HOU"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ATL", "HOU"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC", "MIN"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-09-22T17:00:00Z", "2024-09-22T17:00:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [3.5, -2.5],
            "total_line": [48.5, 44.0],
            "home_moneyline": pd.array([155, -125], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180, 105], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf", "matrixturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["dome", "dome"], dtype=_PYARROW_STR),
            "temp": pd.array([72, 72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0, 0], dtype=pd.Int64Dtype()),
        }
    )
    SchedulesSchema.validate(df)


def test_schedules_schema_rejects_unknown_team_code() -> None:
    from projections.schemas import SchedulesSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "game_id": pd.array(["2024_03_XX_ATL"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ATL"], dtype=_PYARROW_STR),
            "away_team": pd.array(["XX"], dtype=_PYARROW_STR),  # invalid
            "kickoff": pd.to_datetime(["2024-09-22T17:00:00Z"], utc=True).as_unit("us"),
            "spread_line": [3.5],
            "total_line": [48.5],
            "home_moneyline": pd.array([155], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["dome"], dtype=_PYARROW_STR),
            "temp": pd.array([72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0], dtype=pd.Int64Dtype()),
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        SchedulesSchema.validate(df)


def test_schedules_schema_allows_nullable_lines() -> None:
    """Future-week games may have NaN spread/total/kickoff."""
    from projections.schemas import SchedulesSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "season": [2024],
            "week": [18],
            "game_id": pd.array(["2024_18_TBD_TBD"], dtype=_PYARROW_STR),
            "home_team": pd.array(["KC"], dtype=_PYARROW_STR),
            "away_team": pd.array(["DEN"], dtype=_PYARROW_STR),
            "kickoff": pd.array([pd.NaT], dtype="datetime64[us, UTC]"),
            "spread_line": [pd.NA],
            "total_line": [pd.NA],
            "home_moneyline": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "surface": pd.array([pd.NA], dtype=_PYARROW_STR),
            "roof": pd.array([pd.NA], dtype=_PYARROW_STR),
            "temp": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "wind": pd.array([pd.NA], dtype=pd.Int64Dtype()),
        }
    )
    SchedulesSchema.validate(df)
```

- [ ] **Step 2: Run, verify all three fail with ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py::test_schedules_schema_accepts_valid_frame
```

Expected: FAIL with `ImportError: cannot import name 'SchedulesSchema' from 'projections.schemas'`.

- [ ] **Step 3: Add `SchedulesSchema` to `schemas.py`**

In `src/projections/schemas.py`, after `WeeklyStatsSchema` and before `IdMapSchema`, add:

```python
class SchedulesSchema(pa.DataFrameModel):
    """Per-game schedule + Vegas line data — what `ingest.schedules` produces."""

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
```

- [ ] **Step 4: Run all three tests, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k schedules_schema
```

Expected: 3 PASS.

- [ ] **Step 5: Quality gate**

```bash
pytest -v tests/test_schemas
mypy src/projections/schemas.py
ruff check src/projections/schemas.py
```

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add SchedulesSchema for per-game + Vegas line ingest"
```

---

### Task 4: Add `SnapCountsSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add `SnapCountsSchema` class)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (validate-passes / validate-fails tests)

- [ ] **Step 1: Write failing schema tests**

In `tests/test_schemas/test_dataframe_schemas.py`, append:

```python
def test_snap_counts_schema_accepts_valid_frame() -> None:
    from projections.schemas import SnapCountsSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322", "00-0034857"], dtype=_PYARROW_STR),
            "season": [2024, 2024],
            "week": [3, 3],
            "team": pd.array(["MIN", "KC"], dtype=_PYARROW_STR),
            "opponent": pd.array(["HOU", "ATL"], dtype=_PYARROW_STR),
            "position": pd.array(["WR", "QB"], dtype=_PYARROW_STR),
            "offense_snaps": [62, 71],
            "offense_pct": [0.95, 1.0],
            "defense_snaps": [0, 0],
            "defense_pct": [0.0, 0.0],
            "st_snaps": [3, 0],
            "st_pct": [0.10, 0.0],
        }
    )
    SnapCountsSchema.validate(df)


def test_snap_counts_schema_rejects_pct_over_one() -> None:
    from projections.schemas import SnapCountsSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["HOU"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "offense_snaps": [62],
            "offense_pct": [1.5],  # invalid: > 1
            "defense_snaps": [0],
            "defense_pct": [0.0],
            "st_snaps": [3],
            "st_pct": [0.10],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        SnapCountsSchema.validate(df)


def test_snap_counts_schema_rejects_unsupported_position() -> None:
    from projections.schemas import SnapCountsSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0099999"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["HOU"], dtype=_PYARROW_STR),
            "position": pd.array(["FB"], dtype=_PYARROW_STR),  # not in Position enum
            "offense_snaps": [12],
            "offense_pct": [0.20],
            "defense_snaps": [0],
            "defense_pct": [0.0],
            "st_snaps": [4],
            "st_pct": [0.13],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        SnapCountsSchema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k snap_counts_schema
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Add `SnapCountsSchema` to `schemas.py`**

After `SchedulesSchema`:

```python
class SnapCountsSchema(pa.DataFrameModel):
    """Per-player per-game snap counts — what `ingest.snap_counts` produces."""

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
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k snap_counts_schema
```

Expected: 3 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v tests/test_schemas && mypy src/projections/schemas.py && ruff check src/projections/schemas.py
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add SnapCountsSchema for per-player per-game snap data"
```

---

### Task 5: Add `DepthChartsSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add `DepthChartsSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (validate-passes / validate-fails tests)

- [ ] **Step 1: Write failing schema tests**

In `tests/test_schemas/test_dataframe_schemas.py`, append:

```python
def test_depth_charts_schema_accepts_valid_frame() -> None:
    from projections.schemas import DepthChartsSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322", "00-0034857"], dtype=_PYARROW_STR),
            "season": [2024, 2024],
            "week": [3, 3],
            "team": pd.array(["MIN", "KC"], dtype=_PYARROW_STR),
            "position": pd.array(["WR", "QB"], dtype=_PYARROW_STR),
            "depth_team": pd.array(["WR1", "QB1"], dtype=_PYARROW_STR),
            "depth_rank": [1, 1],
        }
    )
    DepthChartsSchema.validate(df)


def test_depth_charts_schema_rejects_rank_out_of_range() -> None:
    from projections.schemas import DepthChartsSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "depth_team": pd.array(["WR1"], dtype=_PYARROW_STR),
            "depth_rank": [12],  # > 10, invalid
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        DepthChartsSchema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k depth_charts_schema
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Add `DepthChartsSchema` to `schemas.py`**

After `SnapCountsSchema`:

```python
class DepthChartsSchema(pa.DataFrameModel):
    """Per-team per-week depth chart — what `ingest.depth_charts` produces.

    `depth_team` is the raw slot label from nfl_data_py (e.g., "WR1", "LWR").
    `depth_rank` is parsed numeric rank within the position group (1 = starter).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    depth_team: Series[str]
    depth_rank: Series[int] = pa.Field(ge=1, le=10)

    class Config:
        strict = "filter"
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k depth_charts_schema
```

Expected: 2 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v tests/test_schemas && mypy src/projections/schemas.py && ruff check src/projections/schemas.py
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add DepthChartsSchema for per-team weekly depth chart"
```

---

### Task 6: Add `NgsPassingSchema`, `NgsRushingSchema`, `NgsReceivingSchema`

Three NGS schemas in one task because they share the same shape (gsis_id + season + week + team + position + a stat-type-specific block of nullable floats).

**Files:**
- Modify: `src/projections/schemas.py` (add 3 NGS schemas)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (validate-passes test per schema; one validate-fails)

- [ ] **Step 1: Write failing schema tests**

In `tests/test_schemas/test_dataframe_schemas.py`, append:

```python
def test_ngs_passing_schema_accepts_valid_frame() -> None:
    from projections.schemas import NgsPassingSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034857"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["KC"], dtype=_PYARROW_STR),
            "position": pd.array(["QB"], dtype=_PYARROW_STR),
            "avg_time_to_throw": [2.71],
            "avg_completed_air_yards": [6.2],
            "avg_intended_air_yards": [8.1],
            "avg_air_yards_differential": [-1.9],
            "aggressiveness": [12.5],
            "max_completed_air_distance": [42.0],
            "avg_air_yards_to_sticks": [-0.4],
            "completion_percentage": [68.5],
            "expected_completion_percentage": [65.2],
            "completion_percentage_above_expectation": [3.3],
            "avg_air_distance": [9.5],
            "max_air_distance": [55.0],
        }
    )
    NgsPassingSchema.validate(df)


def test_ngs_rushing_schema_accepts_valid_frame() -> None:
    from projections.schemas import NgsRushingSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034796"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["PHI"], dtype=_PYARROW_STR),
            "position": pd.array(["RB"], dtype=_PYARROW_STR),
            "efficiency": [3.1],
            "percent_attempts_gte_eight_defenders": [22.5],
            "avg_time_to_los": [2.95],
            "rush_attempts": pd.array([18], dtype=pd.Int64Dtype()),
            "rush_yards": pd.array([102], dtype=pd.Int64Dtype()),
            "expected_rush_yards": [85.4],
            "rush_yards_over_expected": [16.6],
            "avg_rush_yards": [5.7],
            "rush_yards_over_expected_per_att": [0.9],
            "rush_pct_over_expected": [12.0],
        }
    )
    NgsRushingSchema.validate(df)


def test_ngs_receiving_schema_accepts_valid_frame() -> None:
    from projections.schemas import NgsReceivingSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "avg_cushion": [5.4],
            "avg_separation": [3.2],
            "avg_intended_air_yards": [12.1],
            "percent_share_of_intended_air_yards": [29.5],
            "receptions": pd.array([9], dtype=pd.Int64Dtype()),
            "targets": pd.array([12], dtype=pd.Int64Dtype()),
            "catch_percentage": [75.0],
            "yards": pd.array([110], dtype=pd.Int64Dtype()),
            "rec_touchdowns": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_yac": [4.0],
            "avg_expected_yac": [3.5],
            "avg_yac_above_expectation": [0.5],
        }
    )
    NgsReceivingSchema.validate(df)


def test_ngs_receiving_schema_allows_nan_below_threshold() -> None:
    """Players who don't meet NGS qualifying thresholds have NaN columns."""
    from projections.schemas import NgsReceivingSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0099999"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "avg_cushion": [pd.NA],
            "avg_separation": [pd.NA],
            "avg_intended_air_yards": [pd.NA],
            "percent_share_of_intended_air_yards": [pd.NA],
            "receptions": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "targets": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "catch_percentage": [pd.NA],
            "yards": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "rec_touchdowns": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "avg_yac": [pd.NA],
            "avg_expected_yac": [pd.NA],
            "avg_yac_above_expectation": [pd.NA],
        }
    )
    NgsReceivingSchema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "ngs_"
```

Expected: 4 FAIL with ImportError.

- [ ] **Step 3: Add three NGS schemas to `schemas.py`**

After `DepthChartsSchema`:

```python
class NgsPassingSchema(pa.DataFrameModel):
    """NGS passing — season-to-date weekly snapshot per QB.
    Coverage starts 2016 (RFID-chip era)."""

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
    """NGS rushing — season-to-date weekly snapshot per ball-carrier.
    Coverage starts 2016."""

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
    """NGS receiving — season-to-date weekly snapshot per target-receiver.
    Coverage starts 2016."""

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
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "ngs_"
```

Expected: 4 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v tests/test_schemas && mypy src/projections/schemas.py && ruff check src/projections/schemas.py
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add NgsPassingSchema, NgsRushingSchema, NgsReceivingSchema

Same-shape schemas for the three NGS stat types. Coverage begins 2016
(NGS RFID-chip era). All stat columns nullable to allow players who
don't meet qualifying thresholds in a given week."
```

---

### Task 7: Add `WrFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add `WrFeaturesSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (validate-passes / validate-fails for the new feature schema)

- [ ] **Step 1: Write failing schema tests**

In `tests/test_schemas/test_dataframe_schemas.py`, append:

```python
def test_wr_features_schema_accepts_valid_row() -> None:
    from projections.schemas import WrFeaturesSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [10.5],
            "targets_per_game_std": [9.8],
            "target_share_l4": [0.32],
            "air_yards_share_l4": [0.41],
            "receptions_per_game_l4": [7.25],
            "receiving_yards_per_game_l4": [98.5],
            "receiving_tds_per_game_l4": [0.75],
            "rushing_attempts_per_game_l4": [0.5],
            "rushing_yards_per_game_l4": [3.2],
            "designed_rusher": [False],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [3.1],
            "avg_intended_air_yards_std": [11.8],
            "percent_share_intended_air_yards_std": [0.40],
            "avg_yac_above_expectation_std": [0.6],
            "implied_team_total": [24.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_wr_fppg_l4": [22.5],
        }
    )
    WrFeaturesSchema.validate(df)


def test_wr_features_schema_rejects_target_share_over_one() -> None:
    """target_share is a proportion; > 1 is impossible."""
    from projections.schemas import WrFeaturesSchema, _PYARROW_STR

    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [10.5],
            "targets_per_game_std": [9.8],
            "target_share_l4": [1.5],  # invalid
            "air_yards_share_l4": [0.41],
            "receptions_per_game_l4": [7.25],
            "receiving_yards_per_game_l4": [98.5],
            "receiving_tds_per_game_l4": [0.75],
            "rushing_attempts_per_game_l4": [0.5],
            "rushing_yards_per_game_l4": [3.2],
            "designed_rusher": [False],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [3.1],
            "avg_intended_air_yards_std": [11.8],
            "percent_share_intended_air_yards_std": [0.40],
            "avg_yac_above_expectation_std": [0.6],
            "implied_team_total": [24.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_wr_fppg_l4": [22.5],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        WrFeaturesSchema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k wr_features_schema
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Add `WrFeaturesSchema` to `schemas.py`**

After `NgsReceivingSchema`:

```python
class WrFeaturesSchema(pa.DataFrameModel):
    """WR feature DataFrame produced by `features.wr.build_wr_features`.
    Schema enforced at the module boundary — every column has a typed range
    so a feature regression surfaces at validate(), not three modules deep."""

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

    # Opponent strength (proxy: opp's allowed WR fantasy points/game over trailing 4)
    opp_allowed_wr_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = "filter"
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k wr_features_schema
```

Expected: 2 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v tests/test_schemas && mypy src/projections/schemas.py && ruff check src/projections/schemas.py
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add WrFeaturesSchema for WR feature builder output"
```

---

## Phase 2 — Synthetic ingest fixtures

### Task 8: Add synthetic fixtures for the 6 new ingest sources

All six fixtures live in `tests/test_ingest/conftest.py` alongside the existing `fake_id_map_df` and `fake_weekly_df`. Each mimics the *raw* `nfl_data_py.import_*` output shape (the column names BEFORE our `_RENAME` step in the ingest module).

**Files:**
- Modify: `tests/test_ingest/conftest.py` (append 6 new pytest fixtures)

> **Verification note for the implementer:** the column names below are based on the spec's understanding of `nfl_data_py`'s output, which can drift across versions. Before committing this task, run a quick check against the installed `nfl_data_py` to confirm column names match. Example:
> ```bash
> python -c "import nfl_data_py as nfl; print(sorted(nfl.import_schedules([2023]).columns.tolist()))"
> python -c "import nfl_data_py as nfl; print(sorted(nfl.import_snap_counts([2023]).columns.tolist()))"
> python -c "import nfl_data_py as nfl; print(sorted(nfl.import_depth_charts([2023]).columns.tolist()))"
> python -c "import nfl_data_py as nfl; print(sorted(nfl.import_ngs_data('receiving', [2023]).columns.tolist()))"
> ```
> If a column name in the fixture doesn't match the actual API, either rename the fixture column to the actual name (and update Tasks 9-12 `_RENAME` dicts accordingly) or update the schema. Don't blindly proceed.

- [ ] **Step 1: Append `fake_schedules_df`**

```python
@pytest.fixture
def fake_schedules_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_schedules([2024])` — 2 games for week 3.

    Raw column names (before our _RENAME): nfl_data_py uses `gameday` (date)
    and `gametime` (HH:MM string) for kickoff; the ingest module combines them
    into a UTC `kickoff` timestamp.
    """
    return pd.DataFrame(
        {
            "game_id": ["2024_03_KC_ATL", "2024_03_MIN_HOU"],
            "season": [2024, 2024],
            "week": [3, 3],
            "home_team": ["ATL", "HOU"],
            "away_team": ["KC", "MIN"],
            "gameday": ["2024-09-22", "2024-09-22"],
            "gametime": ["20:20", "13:00"],
            "spread_line": [3.5, -2.5],
            "total_line": [48.5, 44.0],
            "home_moneyline": [155, -125],
            "away_moneyline": [-180, 105],
            "surface": ["fieldturf", "matrixturf"],
            "roof": ["dome", "dome"],
            "temp": [72, 72],
            "wind": [0, 0],
        }
    )
```

- [ ] **Step 2: Append `fake_snap_counts_df`**

```python
@pytest.fixture
def fake_snap_counts_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_snap_counts([2024])` — 2 player-weeks.

    Raw column names match the spec — `gsis_id` is present in the
    nfl_data_py output (no rename needed for the player ID).
    """
    return pd.DataFrame(
        {
            "game_id": ["2024_03_KC_ATL", "2024_03_MIN_HOU"],
            "season": [2024, 2024],
            "week": [3, 3],
            "player": ["Patrick Mahomes", "Justin Jefferson"],
            "position": ["QB", "WR"],
            "team": ["KC", "MIN"],
            "opponent": ["ATL", "HOU"],
            "offense_snaps": [71, 62],
            "offense_pct": [1.0, 0.95],
            "defense_snaps": [0, 0],
            "defense_pct": [0.0, 0.0],
            "st_snaps": [0, 3],
            "st_pct": [0.0, 0.10],
            "gsis_id": ["00-0034857", "00-0036322"],
        }
    )
```

- [ ] **Step 3: Append `fake_depth_charts_df`**

```python
@pytest.fixture
def fake_depth_charts_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_depth_charts([2024])` — 3 player-weeks.

    Raw column names: `club_code` is the team (renamed to `team`); `depth_team`
    is the raw slot label (e.g., 'WR1', 'LWR'); `depth_position` may already
    be a numeric rank — the ingest module prefers `depth_position` if present
    and otherwise parses the trailing digit from `depth_team`.
    """
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "club_code": ["MIN", "KC", "PHI"],
            "week": [3, 3, 3],
            "depth_team": ["WR1", "QB1", "RB1"],
            "last_name": ["Jefferson", "Mahomes", "Barkley"],
            "first_name": ["Justin", "Patrick", "Saquon"],
            "formation": ["Offense", "Offense", "Offense"],
            "gsis_id": ["00-0036322", "00-0034857", "00-0034796"],
            "jersey_number": [18, 15, 26],
            "position": ["WR", "QB", "RB"],
            "elias_id": ["JEF845899", "MAH335103", "BAR123456"],
            "depth_position": [1, 1, 1],
            "football_name": ["Justin Jefferson", "Patrick Mahomes", "Saquon Barkley"],
        }
    )
```

- [ ] **Step 4: Append `fake_ngs_passing_df`**

```python
@pytest.fixture
def fake_ngs_passing_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ngs_data('passing', [2024])` — 1 QB-week.

    Raw column names: `player_gsis_id` (renamed to `gsis_id`),
    `team_abbr` (renamed to `team`), `player_position` (renamed to `position`).
    """
    return pd.DataFrame(
        {
            "season": [2024],
            "season_type": ["REG"],
            "week": [3],
            "player_display_name": ["Patrick Mahomes"],
            "player_position": ["QB"],
            "team_abbr": ["KC"],
            "avg_time_to_throw": [2.71],
            "avg_completed_air_yards": [6.2],
            "avg_intended_air_yards": [8.1],
            "avg_air_yards_differential": [-1.9],
            "aggressiveness": [12.5],
            "max_completed_air_distance": [42.0],
            "avg_air_yards_to_sticks": [-0.4],
            "completion_percentage": [68.5],
            "expected_completion_percentage": [65.2],
            "completion_percentage_above_expectation": [3.3],
            "avg_air_distance": [9.5],
            "max_air_distance": [55.0],
            "player_gsis_id": ["00-0034857"],
        }
    )
```

- [ ] **Step 5: Append `fake_ngs_rushing_df`**

```python
@pytest.fixture
def fake_ngs_rushing_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ngs_data('rushing', [2024])` — 1 RB-week."""
    return pd.DataFrame(
        {
            "season": [2024],
            "season_type": ["REG"],
            "week": [3],
            "player_display_name": ["Saquon Barkley"],
            "player_position": ["RB"],
            "team_abbr": ["PHI"],
            "efficiency": [3.1],
            "percent_attempts_gte_eight_defenders": [22.5],
            "avg_time_to_los": [2.95],
            "rush_attempts": [18],
            "rush_yards": [102],
            "expected_rush_yards": [85.4],
            "rush_yards_over_expected": [16.6],
            "avg_rush_yards": [5.7],
            "rush_yards_over_expected_per_att": [0.9],
            "rush_pct_over_expected": [12.0],
            "player_gsis_id": ["00-0034796"],
        }
    )
```

- [ ] **Step 6: Append `fake_ngs_receiving_df`**

```python
@pytest.fixture
def fake_ngs_receiving_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ngs_data('receiving', [2024])` — 1 WR-week."""
    return pd.DataFrame(
        {
            "season": [2024],
            "season_type": ["REG"],
            "week": [3],
            "player_display_name": ["Justin Jefferson"],
            "player_position": ["WR"],
            "team_abbr": ["MIN"],
            "avg_cushion": [5.4],
            "avg_separation": [3.2],
            "avg_intended_air_yards": [12.1],
            "percent_share_of_intended_air_yards": [29.5],
            "receptions": [9],
            "targets": [12],
            "catch_percentage": [75.0],
            "yards": [110],
            "rec_touchdowns": [1],
            "avg_yac": [4.0],
            "avg_expected_yac": [3.5],
            "avg_yac_above_expectation": [0.5],
            "player_gsis_id": ["00-0036322"],
        }
    )
```

- [ ] **Step 7: Sanity-check the fixtures load**

```bash
pytest -v --collect-only tests/test_ingest
```

Expected: collection succeeds; the new fixtures are discoverable. (They won't *run* yet — no tests consume them — but pytest collection failing means there's a syntax error in conftest.)

- [ ] **Step 8: Quality gate + commit**

```bash
mypy tests/test_ingest && ruff check tests/test_ingest
git add tests/test_ingest/conftest.py
git commit -m "test(ingest): add synthetic fixtures for 6 new ingest sources

fake_schedules_df, fake_snap_counts_df, fake_depth_charts_df, plus three
fake_ngs_*_df fixtures for the parameterized NGS ingest. Match raw
nfl_data_py column shapes (pre-rename) so each ingest module's _RENAME
dict is exercised. No tests consume them yet; tasks 9-12 will."
```

---

## Phase 3 — Ingest modules

### Task 9: `src/projections/ingest/schedules.py` + tests

**Files:**
- Create: `src/projections/ingest/schedules.py`
- Create: `tests/test_ingest/test_schedules.py`
- Modify: `src/projections/ingest/__init__.py` (re-export `refresh_schedules`)

- [ ] **Step 1: Write failing tests (round-trip + idempotency + team normalization + nullable lines)**

Create `tests/test_ingest/test_schedules.py`:

```python
"""Schedule ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_schedules
from projections.schemas import SchedulesSchema
from projections.store import read_partition


def test_refresh_schedules_writes_partition(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    written = refresh_schedules(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    SchedulesSchema.validate(df)
    assert set(df["game_id"]) == {"2024_03_KC_ATL", "2024_03_MIN_HOU"}


def test_refresh_schedules_constructs_kickoff_from_gameday_and_gametime(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    # Kickoff combined from "2024-09-22" + "20:20" -> 2024-09-22 20:20 UTC
    kc_atl = df[df["game_id"] == "2024_03_KC_ATL"].iloc[0]
    assert pd.Timestamp(kc_atl["kickoff"]) == pd.Timestamp("2024-09-22 20:20:00", tz="UTC")


def test_refresh_schedules_normalizes_team_codes(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliased = fake_schedules_df.copy()
    aliased.loc[0, "home_team"] = "JAX"  # alias for JAC
    aliased.loc[1, "away_team"] = "LA"  # alias for LAR
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: aliased,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    assert "JAX" not in df["home_team"].tolist()
    assert "JAC" in df["home_team"].tolist()
    assert "LAR" in df["away_team"].tolist()


def test_refresh_schedules_idempotent(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    assert len(df) == 2  # not 4


def test_refresh_schedules_allows_nullable_lines(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future-week games may have NaN spread/total/temp/wind."""
    nan_lines = fake_schedules_df.copy()
    nan_lines.loc[0, "spread_line"] = pd.NA
    nan_lines.loc[0, "total_line"] = pd.NA
    nan_lines.loc[0, "temp"] = pd.NA
    nan_lines.loc[0, "wind"] = pd.NA
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: nan_lines,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    SchedulesSchema.validate(df)
    assert pd.isna(df.iloc[0]["spread_line"])
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_ingest/test_schedules.py
```

Expected: 5 FAIL with `ImportError: cannot import name 'refresh_schedules' from 'projections.ingest'`.

- [ ] **Step 3: Implement `src/projections/ingest/schedules.py`**

```python
"""Refresh per-season schedule + Vegas line data from `nfl_data_py.import_schedules`.

One parquet partition per season (consistent with weekly_stats — schedules at
~272 rows/year are tiny so no further partitioning).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import SchedulesSchema, _PYARROW_STR, normalize_team_code
from projections.store import write_partition

_KEEP = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "surface",
    "roof",
    "temp",
    "wind",
]


def _fetch_raw_schedules(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_schedules(seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _build_kickoff(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Combine `gameday` (date string) + `gametime` (HH:MM string) into a UTC
    timestamp series. Missing gameday OR gametime → NaT (e.g., flex-scheduled
    weeks where kickoff hasn't been confirmed)."""
    combined = gameday.astype(str) + " " + gametime.astype(str)
    parsed = pd.to_datetime(combined, format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
    return parsed.astype("datetime64[us, UTC]")


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["kickoff"] = _build_kickoff(df["gameday"], df["gametime"])
    df["home_team"] = df["home_team"].map(_normalize_team).astype(_PYARROW_STR)
    df["away_team"] = df["away_team"].map(_normalize_team).astype(_PYARROW_STR)
    df["game_id"] = df["game_id"].astype(_PYARROW_STR)

    for str_col in ("surface", "roof"):
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(_PYARROW_STR)

    for int_col in ("home_moneyline", "away_moneyline", "temp", "wind"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype(pd.Int64Dtype())

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = SchedulesSchema.validate(df)
    return df


def refresh_schedules(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write schedule data for each season. One partition per season.
    Idempotent — re-running a season overwrites that partition only."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_schedules([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "schedules", df, season=season, week=None)
        record_manifest(data_root, table="schedules", season=season, df=df)
        written.append(path)
    return written
```

- [ ] **Step 4: Re-export from `ingest/__init__.py`**

```python
"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.id_map import build_id_map
from projections.ingest.schedules import refresh_schedules
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "refresh_schedules",
    "refresh_weekly_stats",
]
```

- [ ] **Step 5: Run, verify all 5 schedule tests pass**

```bash
pytest -v tests/test_ingest/test_schedules.py
```

Expected: 5 PASS.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v tests/test_ingest tests/test_schemas
mypy src tests
ruff check src tests
ruff format --check src tests
git add src/projections/ingest/schedules.py src/projections/ingest/__init__.py tests/test_ingest/test_schedules.py
git commit -m "feat(ingest): add refresh_schedules for per-game + Vegas line data

Mirrors weekly_stats.py template. Combines nfl_data_py's `gameday` +
`gametime` raw columns into a UTC `kickoff` timestamp. Team codes
normalized via normalize_team_code. Future-week games with NaN lines
flow through (schema marks those columns nullable)."
```

---

### Task 10: `src/projections/ingest/snap_counts.py` + tests

**Files:**
- Create: `src/projections/ingest/snap_counts.py`
- Create: `tests/test_ingest/test_snap_counts.py`
- Modify: `src/projections/ingest/__init__.py` (re-export `refresh_snap_counts`)

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest/test_snap_counts.py`:

```python
"""Snap counts ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_snap_counts
from projections.schemas import SnapCountsSchema
from projections.store import read_partition


def test_refresh_snap_counts_writes_partition(
    tmp_path: Path,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: fake_snap_counts_df,
    )
    written = refresh_snap_counts(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    SnapCountsSchema.validate(df)
    assert set(df["gsis_id"]) == {"00-0034857", "00-0036322"}


def test_refresh_snap_counts_normalizes_team_codes(
    tmp_path: Path,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliased = fake_snap_counts_df.copy()
    aliased.loc[0, "team"] = "OAK"  # historical alias for LV
    aliased.loc[1, "opponent"] = "WSH"  # historical alias for WAS
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: aliased,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    assert "OAK" not in df["team"].tolist()
    assert "LV" in df["team"].tolist()
    assert "WAS" in df["opponent"].tolist()


def test_refresh_snap_counts_filters_unsupported_positions(
    tmp_path: Path,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ol_row = {
        "game_id": "2024_03_MIN_HOU",
        "season": 2024,
        "week": 3,
        "player": "Jane Doe",
        "position": "OL",  # not a fantasy position
        "team": "MIN",
        "opponent": "HOU",
        "offense_snaps": 70,
        "offense_pct": 1.0,
        "defense_snaps": 0,
        "defense_pct": 0.0,
        "st_snaps": 0,
        "st_pct": 0.0,
        "gsis_id": "00-0099999",
    }
    with_ol = pd.concat([fake_snap_counts_df, pd.DataFrame([ol_row])], ignore_index=True)
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: with_ol,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    assert "00-0099999" not in df["gsis_id"].tolist()


def test_refresh_snap_counts_drops_rows_missing_gsis_id(
    tmp_path: Path,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bench/practice players in nfl_data_py snap_counts can have null gsis_id."""
    nullid = fake_snap_counts_df.copy()
    nullid.loc[0, "gsis_id"] = None
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: nullid,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    assert len(df) == 1


def test_refresh_snap_counts_idempotent(
    tmp_path: Path,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: fake_snap_counts_df,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    assert len(df) == 2
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_ingest/test_snap_counts.py
```

Expected: 5 FAIL with ImportError.

- [ ] **Step 3: Implement `src/projections/ingest/snap_counts.py`**

```python
"""Refresh per-season snap counts from `nfl_data_py.import_snap_counts`.

One parquet partition per season."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import Position, SnapCountsSchema, _PYARROW_STR, normalize_team_code
from projections.store import write_partition

_KEEP = [
    "gsis_id",
    "season",
    "week",
    "team",
    "opponent",
    "position",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
]


def _fetch_raw_snap_counts(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_snap_counts(seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    for int_col in ("offense_snaps", "defense_snaps", "st_snaps"):
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype(int)

    for float_col in ("offense_pct", "defense_pct", "st_pct"):
        if float_col in df.columns:
            df[float_col] = df[float_col].fillna(0.0).astype(float)

    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].map(_normalize_team).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = SnapCountsSchema.validate(df)
    return df


def refresh_snap_counts(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write snap counts for each season. Idempotent."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_snap_counts([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "snap_counts", df, season=season, week=None)
        record_manifest(data_root, table="snap_counts", season=season, df=df)
        written.append(path)
    return written
```

- [ ] **Step 4: Re-export from `ingest/__init__.py`**

Add `refresh_snap_counts` to imports and `__all__`.

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest -v tests/test_ingest/test_snap_counts.py
```

Expected: 5 PASS.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v tests/test_ingest tests/test_schemas && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/ingest/snap_counts.py src/projections/ingest/__init__.py tests/test_ingest/test_snap_counts.py
git commit -m "feat(ingest): add refresh_snap_counts for per-player per-game snap data

Mirrors weekly_stats.py template. Position-filtered to fantasy
positions; rows without gsis_id (bench/practice) dropped; team codes
normalized."
```

---

### Task 11: `src/projections/ingest/depth_charts.py` + tests

**Files:**
- Create: `src/projections/ingest/depth_charts.py`
- Create: `tests/test_ingest/test_depth_charts.py`
- Modify: `src/projections/ingest/__init__.py` (re-export `refresh_depth_charts`)

The `depth_rank` parser is the interesting bit: prefer `depth_position` (numeric) if present and non-null, otherwise parse the trailing digit from `depth_team` (e.g., `"WR1"` → 1, `"WR3"` → 3), falling back to 1 with a warning for unrankable labels (`"LWR"`, `"SWR"`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest/test_depth_charts.py`:

```python
"""Depth chart ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_depth_charts
from projections.ingest.depth_charts import _parse_depth_rank
from projections.schemas import DepthChartsSchema
from projections.store import read_partition


def test_refresh_depth_charts_writes_partition(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    written = refresh_depth_charts(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    DepthChartsSchema.validate(df)
    assert set(df["gsis_id"]) == {"00-0036322", "00-0034857", "00-0034796"}


def test_refresh_depth_charts_renames_club_code_to_team(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert "team" in df.columns
    assert "club_code" not in df.columns
    assert set(df["team"]) == {"MIN", "KC", "PHI"}


def test_refresh_depth_charts_filters_unsupported_positions(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OL, IDP positions are dropped — depth_charts also lists non-fantasy positions."""
    extra = pd.DataFrame(
        [
            {
                "season": 2024,
                "club_code": "MIN",
                "week": 3,
                "depth_team": "LT1",
                "last_name": "Doe",
                "first_name": "John",
                "formation": "Offense",
                "gsis_id": "00-0099998",
                "jersey_number": 71,
                "position": "OL",
                "elias_id": "DOE99998",
                "depth_position": 1,
                "football_name": "John Doe",
            }
        ]
    )
    with_ol = pd.concat([fake_depth_charts_df, extra], ignore_index=True)
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: with_ol,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert "00-0099998" not in df["gsis_id"].tolist()


def test_refresh_depth_charts_idempotent(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert len(df) == 3


# --- _parse_depth_rank unit tests ---


def test_parse_depth_rank_prefers_numeric_depth_position() -> None:
    """If depth_position is a non-null int, that's the rank."""
    rank, warned = _parse_depth_rank(depth_team="LWR", depth_position=2)
    assert rank == 2
    assert warned is False


def test_parse_depth_rank_parses_trailing_digit_from_depth_team() -> None:
    """Falls back to the trailing digit in depth_team when depth_position is null."""
    rank, warned = _parse_depth_rank(depth_team="WR3", depth_position=None)
    assert rank == 3
    assert warned is False


def test_parse_depth_rank_falls_back_to_one_for_unrankable_label() -> None:
    """Unrankable label (no trailing digit, no depth_position) → 1, warned."""
    rank, warned = _parse_depth_rank(depth_team="LWR", depth_position=None)
    assert rank == 1
    assert warned is True


def test_parse_depth_rank_clamps_above_ten() -> None:
    """Parsed rank > 10 (impossible per schema) clamps to 10."""
    rank, warned = _parse_depth_rank(depth_team="WR99", depth_position=None)
    assert rank == 10
    assert warned is True
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_ingest/test_depth_charts.py
```

Expected: FAIL with ImportError (module + helper don't exist yet).

- [ ] **Step 3: Implement `src/projections/ingest/depth_charts.py`**

```python
"""Refresh per-season depth charts from `nfl_data_py.import_depth_charts`.

`nfl_data_py` raw column conventions vary across seasons:
- Pre-2018-ish: `depth_team` uses alignment labels (LWR, RWR, SWR).
- Newer seasons: `depth_team` uses rank labels (WR1, WR2) and `depth_position`
  contains a numeric rank.

`_parse_depth_rank` resolves these into a single canonical `depth_rank` int."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import DepthChartsSchema, Position, _PYARROW_STR, normalize_team_code
from projections.store import write_partition

_log = logging.getLogger(__name__)

_KEEP = ["gsis_id", "season", "week", "team", "position", "depth_team", "depth_rank"]
_RENAME = {"club_code": "team"}
_TRAILING_DIGITS = re.compile(r"(\d+)$")


def _fetch_raw_depth_charts(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_depth_charts(seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _parse_depth_rank(*, depth_team: str | None, depth_position: int | None) -> tuple[int, bool]:
    """Resolve a numeric `depth_rank` from raw inputs.

    Returns (rank, warned). `warned` is True if we had to fall back to 1
    because the inputs were unrankable, so the caller can log once with
    a representative example."""
    if depth_position is not None and not pd.isna(depth_position):
        try:
            return min(10, max(1, int(depth_position))), False
        except (ValueError, TypeError):
            pass
    if depth_team is not None and not pd.isna(depth_team):
        match = _TRAILING_DIGITS.search(str(depth_team))
        if match:
            parsed = int(match.group(1))
            if parsed >= 1:
                return min(10, parsed), parsed > 10
    return 1, True


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # Resolve depth_rank row-by-row; track if any rows fell back to 1 unranked.
    ranks: list[int] = []
    fallback_count = 0
    sample_label: str | None = None
    for _, row in df.iterrows():
        rank, warned = _parse_depth_rank(
            depth_team=row.get("depth_team"),
            depth_position=row.get("depth_position"),
        )
        ranks.append(rank)
        if warned:
            fallback_count += 1
            if sample_label is None:
                sample_label = str(row.get("depth_team"))
    if fallback_count:
        _log.warning(
            "Fell back to depth_rank=1 for %d rows (e.g., depth_team=%r). "
            "These are unrankable labels (alignment-based or out-of-range numeric).",
            fallback_count,
            sample_label,
        )
    df["depth_rank"] = ranks

    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)

    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = DepthChartsSchema.validate(df)
    return df


def refresh_depth_charts(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write depth charts for each season. Idempotent."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_depth_charts([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "depth_charts", df, season=season, week=None)
        record_manifest(data_root, table="depth_charts", season=season, df=df)
        written.append(path)
    return written
```

- [ ] **Step 4: Re-export from `ingest/__init__.py`**

Add `refresh_depth_charts` to imports and `__all__`.

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest -v tests/test_ingest/test_depth_charts.py
```

Expected: 8 PASS.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v tests/test_ingest tests/test_schemas && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/ingest/depth_charts.py src/projections/ingest/__init__.py tests/test_ingest/test_depth_charts.py
git commit -m "feat(ingest): add refresh_depth_charts with depth_rank parser

_parse_depth_rank resolves the canonical numeric rank from the two raw
sources nfl_data_py exposes (depth_position int + depth_team label),
preferring numeric over parsed-from-label, falling back to 1 with a
warning for unrankable alignment-based labels (LWR/RWR/SWR)."
```

---

### Task 12: `src/projections/ingest/ngs.py` (parameterized) + tests

One module produces three partition tables (`ngs_passing`, `ngs_rushing`, `ngs_receiving`). The test module is parameterized.

**Files:**
- Create: `src/projections/ingest/ngs.py`
- Create: `tests/test_ingest/test_ngs.py`
- Modify: `src/projections/ingest/__init__.py` (re-export `refresh_ngs`)

- [ ] **Step 1: Write failing tests (parameterized over stat_type)**

Create `tests/test_ingest/test_ngs.py`:

```python
"""NGS ingest tests — one parameterized module covering passing/rushing/receiving."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_ngs
from projections.schemas import (
    NgsPassingSchema,
    NgsReceivingSchema,
    NgsRushingSchema,
)
from projections.store import read_partition

_SCHEMA_FOR = {
    "passing": NgsPassingSchema,
    "rushing": NgsRushingSchema,
    "receiving": NgsReceivingSchema,
}

_FIXTURE_NAME_FOR = {
    "passing": "fake_ngs_passing_df",
    "rushing": "fake_ngs_rushing_df",
    "receiving": "fake_ngs_receiving_df",
}


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_refresh_ngs_writes_distinct_partition_per_stat_type(
    tmp_path: Path,
    stat_type: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_df = request.getfixturevalue(_FIXTURE_NAME_FOR[stat_type])
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_df,
    )
    written = refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    assert len(written) == 1
    assert f"ngs_{stat_type}" in str(written[0])

    table = f"ngs_{stat_type}"
    df = read_partition(tmp_path / "raw", table, season=2024)
    _SCHEMA_FOR[stat_type].validate(df)


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_refresh_ngs_renames_player_gsis_id_to_gsis_id(
    tmp_path: Path,
    stat_type: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_df = request.getfixturevalue(_FIXTURE_NAME_FOR[stat_type])
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_df,
    )
    refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    df = read_partition(tmp_path / "raw", f"ngs_{stat_type}", season=2024)
    assert "gsis_id" in df.columns
    assert "player_gsis_id" not in df.columns
    assert "team" in df.columns
    assert "team_abbr" not in df.columns
    assert "position" in df.columns
    assert "player_position" not in df.columns


def test_refresh_ngs_rejects_unknown_stat_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stat_type must be one of"):
        refresh_ngs(tmp_path, stat_type="kicking", seasons=[2024])  # type: ignore[arg-type]


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_refresh_ngs_idempotent(
    tmp_path: Path,
    stat_type: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_df = request.getfixturevalue(_FIXTURE_NAME_FOR[stat_type])
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_df,
    )
    refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    df = read_partition(tmp_path / "raw", f"ngs_{stat_type}", season=2024)
    assert len(df) == 1


def test_refresh_ngs_three_stat_types_produce_independent_partitions(
    tmp_path: Path,
    fake_ngs_passing_df: pd.DataFrame,
    fake_ngs_rushing_df: pd.DataFrame,
    fake_ngs_receiving_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = {
        "passing": fake_ngs_passing_df,
        "rushing": fake_ngs_rushing_df,
        "receiving": fake_ngs_receiving_df,
    }
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fakes[st],
    )
    for st in ("passing", "rushing", "receiving"):
        refresh_ngs(tmp_path, stat_type=st, seasons=[2024])

    for table, schema in [
        ("ngs_passing", NgsPassingSchema),
        ("ngs_rushing", NgsRushingSchema),
        ("ngs_receiving", NgsReceivingSchema),
    ]:
        df = read_partition(tmp_path / "raw", table, season=2024)
        assert len(df) == 1
        schema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_ingest/test_ngs.py
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `src/projections/ingest/ngs.py`**

```python
"""Refresh per-season NGS data from `nfl_data_py.import_ngs_data`.

Parameterized by `stat_type` ∈ {"passing", "rushing", "receiving"}; produces
three distinct partition tables (`ngs_passing`, `ngs_rushing`, `ngs_receiving`).

NGS returns season-to-date weekly snapshots (cumulative through each week),
which is naturally leakage-safe — feature builders filter by `week < as_of_week`."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    NgsPassingSchema,
    NgsReceivingSchema,
    NgsRushingSchema,
    Position,
    _PYARROW_STR,
    normalize_team_code,
)
from projections.store import write_partition

NgsStatType = Literal["passing", "rushing", "receiving"]
_VALID_STAT_TYPES: tuple[NgsStatType, ...] = ("passing", "rushing", "receiving")

_RENAME = {
    "player_gsis_id": "gsis_id",
    "team_abbr": "team",
    "player_position": "position",
}

_KEEP_COMMON = ["gsis_id", "season", "week", "team", "position"]

_KEEP_PASSING = _KEEP_COMMON + [
    "avg_time_to_throw",
    "avg_completed_air_yards",
    "avg_intended_air_yards",
    "avg_air_yards_differential",
    "aggressiveness",
    "max_completed_air_distance",
    "avg_air_yards_to_sticks",
    "completion_percentage",
    "expected_completion_percentage",
    "completion_percentage_above_expectation",
    "avg_air_distance",
    "max_air_distance",
]

_KEEP_RUSHING = _KEEP_COMMON + [
    "efficiency",
    "percent_attempts_gte_eight_defenders",
    "avg_time_to_los",
    "rush_attempts",
    "rush_yards",
    "expected_rush_yards",
    "rush_yards_over_expected",
    "avg_rush_yards",
    "rush_yards_over_expected_per_att",
    "rush_pct_over_expected",
]

_KEEP_RECEIVING = _KEEP_COMMON + [
    "avg_cushion",
    "avg_separation",
    "avg_intended_air_yards",
    "percent_share_of_intended_air_yards",
    "receptions",
    "targets",
    "catch_percentage",
    "yards",
    "rec_touchdowns",
    "avg_yac",
    "avg_expected_yac",
    "avg_yac_above_expectation",
]

_KEEP_FOR: dict[NgsStatType, list[str]] = {
    "passing": _KEEP_PASSING,
    "rushing": _KEEP_RUSHING,
    "receiving": _KEEP_RECEIVING,
}

_INT_COLS_FOR: dict[NgsStatType, tuple[str, ...]] = {
    "passing": (),
    "rushing": ("rush_attempts", "rush_yards"),
    "receiving": ("receptions", "targets", "yards", "rec_touchdowns"),
}

_SCHEMA_FOR: dict[NgsStatType, type] = {
    "passing": NgsPassingSchema,
    "rushing": NgsRushingSchema,
    "receiving": NgsReceivingSchema,
}


def _fetch_raw_ngs(stat_type: NgsStatType, seasons: list[int]) -> pd.DataFrame:
    return nfl.import_ngs_data(stat_type, seasons)


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _normalize_one_season(stat_type: NgsStatType, raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # NGS int-typed stat columns are nullable in pandera (qualifying-threshold misses).
    for int_col in _INT_COLS_FOR[stat_type]:
        if int_col in df.columns:
            df[int_col] = df[int_col].astype(pd.Int64Dtype())

    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[[c for c in _KEEP_FOR[stat_type] if c in df.columns]].copy()
    df = _SCHEMA_FOR[stat_type].validate(df)
    return df


def refresh_ngs(
    data_root: Path,
    *,
    stat_type: NgsStatType,
    seasons: Iterable[int],
) -> list[Path]:
    """Fetch and write NGS data for `stat_type` and each season. Idempotent.

    Writes to `data/raw/ngs_{stat_type}/season=YYYY/part.parquet`.
    """
    if stat_type not in _VALID_STAT_TYPES:
        raise ValueError(
            f"stat_type must be one of {_VALID_STAT_TYPES}, got {stat_type!r}"
        )

    table = f"ngs_{stat_type}"
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_ngs(stat_type, [season])
        df = _normalize_one_season(stat_type, raw)
        path = write_partition(data_root / "raw", table, df, season=season, week=None)
        record_manifest(data_root, table=table, season=season, df=df)
        written.append(path)
    return written
```

- [ ] **Step 4: Re-export from `ingest/__init__.py`**

Add `refresh_ngs` to imports and `__all__`.

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest -v tests/test_ingest/test_ngs.py
```

Expected: 12 PASS (4 parameterized × 3 stat types + 2 standalone — confirm exact count from your collection).

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v tests/test_ingest tests/test_schemas && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/ingest/ngs.py src/projections/ingest/__init__.py tests/test_ingest/test_ngs.py
git commit -m "feat(ingest): add refresh_ngs (parameterized over passing/rushing/receiving)

One module, three partition tables. Hard part is the snapshot/partition
decision (NGS returns season-to-date weekly snapshots) — that's
naturally leakage-safe and uses the same per-season partition layout as
every other ingest source."
```

---

## Phase 4 — Features module

### Task 13: `src/projections/features/_rolling.py` + tests

Shared trailing-window helpers. Designed for the WR builder but pinned in 2a so 2b's other position builders consume a stable contract.

**Files:**
- Create: `src/projections/features/__init__.py` (empty package marker for now; populated in Task 15)
- Create: `src/projections/features/_rolling.py`
- Create: `tests/test_features/__init__.py`
- Create: `tests/test_features/conftest.py` (shared synthetic frames; populated as we add tests)
- Create: `tests/test_features/test_rolling.py`

- [ ] **Step 1: Bootstrap empty package + test harness**

Create `src/projections/features/__init__.py` (empty file is fine for now):

```python
"""Per-position feature builders. Pure functions; no I/O."""

from __future__ import annotations
```

Create `tests/test_features/__init__.py` as an empty file. Create `tests/test_features/conftest.py` with a placeholder docstring:

```python
"""Shared synthetic frames for feature-builder tests."""

from __future__ import annotations
```

(Fixtures get added incrementally in subsequent tasks as tests need them.)

- [ ] **Step 2: Write failing tests for `last_n_per_group`**

Create `tests/test_features/test_rolling.py`:

```python
"""Rolling-window helper tests."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features._rolling import (
    last_n_per_group,
    per_game_rate,
    season_to_date_mean,
)


def test_last_n_per_group_returns_only_last_n_rows_per_group() -> None:
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A", "A", "A", "B", "B"],
            "season": [2024, 2024, 2024, 2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4, 5, 1, 2],
            "value": [10, 20, 30, 40, 50, 100, 200],
        }
    )
    out = last_n_per_group(df, group_cols=["player"], sort_cols=["season", "week"], n=3)
    a_rows = out[out["player"] == "A"].sort_values("week")
    assert a_rows["week"].tolist() == [3, 4, 5]
    b_rows = out[out["player"] == "B"].sort_values("week")
    assert b_rows["week"].tolist() == [1, 2]  # only 2 rows total, returns all


def test_last_n_per_group_handles_unsorted_input() -> None:
    """Helper sorts internally so caller order doesn't matter."""
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A"],
            "season": [2024, 2024, 2024],
            "week": [3, 1, 2],
            "value": [30, 10, 20],
        }
    )
    out = last_n_per_group(df, group_cols=["player"], sort_cols=["season", "week"], n=2)
    weeks = sorted(out["week"].tolist())
    assert weeks == [2, 3]


def test_last_n_per_group_empty_input_returns_empty() -> None:
    df = pd.DataFrame({"player": [], "season": [], "week": [], "value": []})
    out = last_n_per_group(df, group_cols=["player"], sort_cols=["season", "week"], n=4)
    assert len(out) == 0


def test_per_game_rate_handles_zero_denominator() -> None:
    df = pd.DataFrame({"num": [10, 20, 0], "denom": [2, 0, 0]})
    out = per_game_rate(df, num_col="num", denom_col="denom")
    assert out.tolist() == [5.0, 0.0, 0.0]


def test_per_game_rate_handles_missing_denom_as_zero() -> None:
    df = pd.DataFrame({"num": [10, 20], "denom": [None, 5]})
    out = per_game_rate(df, num_col="num", denom_col="denom")
    assert out.tolist() == [0.0, 4.0]


def test_season_to_date_mean_running_average_within_season() -> None:
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A", "A"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = season_to_date_mean(
        df,
        group_cols=["player", "season"],
        sort_cols=["week"],
        value_col="value",
    )
    # After week 1: 10. After week 2: 15. After week 3: 20. After week 4: 25.
    assert out.sort_index().tolist() == [10.0, 15.0, 20.0, 25.0]


def test_season_to_date_mean_resets_across_seasons() -> None:
    df = pd.DataFrame(
        {
            "player": ["A", "A", "A", "A"],
            "season": [2023, 2023, 2024, 2024],
            "week": [16, 17, 1, 2],
            "value": [100.0, 200.0, 10.0, 20.0],
        }
    )
    out = season_to_date_mean(
        df,
        group_cols=["player", "season"],
        sort_cols=["week"],
        value_col="value",
    )
    # 2023 weeks: 100, then 150. 2024 weeks: 10, then 15.
    assert out.sort_index().tolist() == [100.0, 150.0, 10.0, 15.0]
```

- [ ] **Step 3: Run, verify ImportError**

```bash
pytest -v tests/test_features/test_rolling.py
```

Expected: FAIL with ImportError.

- [ ] **Step 4: Implement `src/projections/features/_rolling.py`**

```python
"""Trailing-window helpers used by feature builders.

All functions are pure: they don't mutate inputs and don't perform I/O.
Designed for the WR builder in 2a but applicable to every position builder
in 2b — the contract here is the load-bearing one."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def last_n_per_group(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    sort_cols: Sequence[str],
    n: int,
) -> pd.DataFrame:
    """Return only the last `n` rows per group, sorted by `sort_cols`.

    Used to compute trailing-N-game statistics: group by player, take the
    last 4 entries by (season, week), then mean a stat column.

    Caller order doesn't matter — we sort internally.
    """
    if df.empty:
        return df.copy()
    group_cols_l = list(group_cols)
    sort_cols_l = list(sort_cols)
    return (
        df.sort_values(group_cols_l + sort_cols_l)
        .groupby(group_cols_l, as_index=False, sort=False)
        .tail(n)
        .copy()
    )


def per_game_rate(df: pd.DataFrame, *, num_col: str, denom_col: str) -> pd.Series:
    """Safe division: `df[num_col] / df[denom_col]`, with zeros and NaN
    denominators returning 0.0 (instead of inf or NaN)."""
    num = df[num_col].fillna(0).astype(float)
    denom = df[denom_col].fillna(0).astype(float)
    return pd.Series(
        [n / d if d > 0 else 0.0 for n, d in zip(num, denom, strict=True)],
        index=df.index,
        dtype=float,
    )


def season_to_date_mean(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    sort_cols: Sequence[str],
    value_col: str,
) -> pd.Series:
    """Running per-group mean of `value_col` within each season.

    Returns a Series aligned to the input row index (after internal sort).
    Group_cols MUST include season so the running mean resets at season
    boundaries — that's the caller's responsibility, not the helper's.
    """
    if df.empty:
        return pd.Series([], dtype=float)
    sorted_df = df.sort_values(list(group_cols) + list(sort_cols))
    result = (
        sorted_df.groupby(list(group_cols), sort=False)[value_col]
        .expanding()
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    return result
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest -v tests/test_features/test_rolling.py
```

Expected: 7 PASS.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v tests/test_features && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/features/__init__.py src/projections/features/_rolling.py tests/test_features/__init__.py tests/test_features/conftest.py tests/test_features/test_rolling.py
git commit -m "feat(features): add _rolling.py — trailing-window helpers

last_n_per_group, per_game_rate, season_to_date_mean. Pure functions,
no I/O. Designed for WR builder in 2a; pinned now so 2b's other
position builders inherit a stable contract."
```

---

### Task 14: `src/projections/features/_opponent.py` + tests

Opponent-strength proxy: per `(opp_team, season, week)`, the average fantasy points the opponent has allowed to `position` over the trailing N weeks. Calls `scoring.score()` so we don't reimplement scoring math.

**Files:**
- Create: `src/projections/features/_opponent.py`
- Create: `tests/test_features/test_opponent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_features/test_opponent.py`:

```python
"""Opponent-strength helper tests."""

from __future__ import annotations

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.schemas import Position, Ruleset


def test_opp_allowed_fppg_computes_trailing_mean_per_defense() -> None:
    """Team B plays Team A every week. Team A's WR scores 25/20/15/10 fantasy
    points across weeks 1-4 (PPR). For Team B's WRs in week 5, the trailing-4
    average of opponent-allowed WR points is (25+20+15+10)/4 = 17.5."""
    weekly_stats = pd.DataFrame(
        {
            "gsis_id": [
                "00-0000001", "00-0000001", "00-0000001", "00-0000001",
            ],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "position": ["WR", "WR", "WR", "WR"],
            "team": ["A", "A", "A", "A"],
            "opponent": ["B", "B", "B", "B"],
            "passing_yards": [0.0, 0.0, 0.0, 0.0],
            "passing_tds": [0, 0, 0, 0],
            "interceptions": [0, 0, 0, 0],
            "rushing_yards": [0.0, 0.0, 0.0, 0.0],
            "rushing_tds": [0, 0, 0, 0],
            "carries": [0, 0, 0, 0],
            "receptions": [10, 8, 6, 4],
            "receiving_yards": [150.0, 120.0, 90.0, 60.0],
            "receiving_tds": [1, 1, 1, 1],
            "receiving_air_yards": [180.0, 140.0, 100.0, 70.0],
            "targets": [12, 10, 8, 6],
            "fumbles_lost": [0, 0, 0, 0],
        }
    )
    # PPR: rec=1, rec_yds/10, rec_td=6 → week 1: 10 + 15 + 6 = 31.
    # Hmm — rebuild to match the test promise. Let me set yards to give round numbers.
    # WR's points for the 4 weeks: target 25, 20, 15, 10.
    # With PPR (rec=1, yds/10, td=6), an easy formula:
    #   pts = receptions + rec_yards/10 + 6*rec_tds
    # Week 1: receptions=10, yds=90, td=1 → 10 + 9 + 6 = 25.
    # Week 2: receptions=8,  yds=60, td=1 → 8 + 6 + 6 = 20.
    # Week 3: receptions=6,  yds=30, td=1 → 6 + 3 + 6 = 15.
    # Week 4: receptions=4,  yds=0,  td=1 → 4 + 0 + 6 = 10.
    weekly_stats["receiving_yards"] = [90.0, 60.0, 30.0, 0.0]
    weekly_stats["receiving_air_yards"] = [120.0, 80.0, 40.0, 0.0]

    result = opp_allowed_fppg(
        weekly_stats,
        position=Position.WR,
        ruleset=Ruleset.espn_ppr(),
        n_weeks=4,
    )

    # Result is keyed (season, week, opp_team) where opp_team is the defense.
    # For team B in week 5 (i.e., team B has just allowed weeks 1-4): mean of 25,20,15,10 = 17.5.
    row = result[
        (result["season"] == 2024)
        & (result["week"] == 5)
        & (result["opp_team"] == "B")
    ]
    assert len(row) == 1
    assert row.iloc[0]["opp_allowed_fppg"] == 17.5


def test_opp_allowed_fppg_filters_to_position() -> None:
    """RB stats must not contribute to opp_allowed_wr_fppg."""
    weekly_stats = pd.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000002"],
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["WR", "RB"],
            "team": ["A", "A"],
            "opponent": ["B", "B"],
            "passing_yards": [0.0, 0.0],
            "passing_tds": [0, 0],
            "interceptions": [0, 0],
            "rushing_yards": [0.0, 100.0],
            "rushing_tds": [0, 1],
            "carries": [0, 20],
            "receptions": [5, 0],
            "receiving_yards": [50.0, 0.0],
            "receiving_tds": [0, 0],
            "receiving_air_yards": [60.0, 0.0],
            "targets": [7, 0],
            "fumbles_lost": [0, 0],
        }
    )

    wr_result = opp_allowed_fppg(
        weekly_stats,
        position=Position.WR,
        ruleset=Ruleset.espn_ppr(),
        n_weeks=4,
    )
    # Team B allowed only the WR's points in week 1.
    # WR points: 5 + 5 + 0 = 10. So team B in week 2 should see 10.0.
    row = wr_result[
        (wr_result["week"] == 2) & (wr_result["opp_team"] == "B")
    ]
    assert len(row) == 1
    assert row.iloc[0]["opp_allowed_fppg"] == 10.0
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_features/test_opponent.py
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `src/projections/features/_opponent.py`**

```python
"""Opponent-strength proxy: average fantasy points allowed to a position
over a trailing window. v1 substitute for true opponent-adjusted EPA
(which would need play-by-play ingest, deferred to a later plan)."""

from __future__ import annotations

import pandas as pd

from projections.features._rolling import last_n_per_group
from projections.schemas import Position, Ruleset
from projections.scoring.score import StatLine, score


def _row_to_statline(row: pd.Series) -> StatLine:
    """Build a StatLine from a weekly_stats row. Defaults to 0 for any
    field not present in weekly_stats (e.g., 2pt conversions, return_tds,
    which the foundations-era schema doesn't track)."""
    return StatLine(
        passing_yards=float(row.get("passing_yards", 0.0) or 0.0),
        passing_tds=int(row.get("passing_tds", 0) or 0),
        interceptions=int(row.get("interceptions", 0) or 0),
        rushing_yards=float(row.get("rushing_yards", 0.0) or 0.0),
        rushing_tds=int(row.get("rushing_tds", 0) or 0),
        receptions=int(row.get("receptions", 0) or 0),
        receiving_yards=float(row.get("receiving_yards", 0.0) or 0.0),
        receiving_tds=int(row.get("receiving_tds", 0) or 0),
        fumbles_lost=int(row.get("fumbles_lost", 0) or 0),
    )


def opp_allowed_fppg(
    weekly_stats: pd.DataFrame,
    *,
    position: Position,
    ruleset: Ruleset,
    n_weeks: int,
) -> pd.DataFrame:
    """For each `(opp_team, season, week)`, the mean fantasy points allowed
    to `position` over the trailing `n_weeks`.

    Returns a DataFrame with columns `(season, week, opp_team, opp_allowed_fppg)`,
    where `week` is the week being scored against (NOT included in the
    trailing window). Joining onto a feature row uses `(season, week, opponent)`
    on the WR side to retrieve the opponent's allowed-points proxy.
    """
    pos_stats = weekly_stats[weekly_stats["position"] == position.value].copy()
    if pos_stats.empty:
        return pd.DataFrame(
            columns=["season", "week", "opp_team", "opp_allowed_fppg"]
        ).astype({"season": int, "week": int, "opp_allowed_fppg": float})

    # Score each per-game line.
    pos_stats["fpts"] = pos_stats.apply(
        lambda r: score(_row_to_statline(r), ruleset), axis=1
    )

    # Sum per (opp_team, season, week) — that's all `position`-players' points
    # allowed by `opp_team` in that week.
    weekly_allowed = (
        pos_stats.groupby(["opponent", "season", "week"], as_index=False)["fpts"]
        .sum()
        .rename(columns={"opponent": "opp_team"})
    )

    # Trailing-N mean per opp_team, BUT the result is associated with the NEXT
    # week (the one where the opponent will face this defense).
    # Approach: keep the trailing window, then shift the resulting mean to
    # week+1 of the same season.
    rows: list[dict[str, object]] = []
    for (opp_team, season), g in weekly_allowed.groupby(
        ["opp_team", "season"], sort=False
    ):
        g_sorted = g.sort_values("week").reset_index(drop=True)
        for i in range(len(g_sorted)):
            window = g_sorted.iloc[max(0, i - n_weeks + 1) : i + 1]
            mean_fppg = float(window["fpts"].mean())
            target_week = int(g_sorted.iloc[i]["week"]) + 1
            rows.append(
                {
                    "season": int(season),
                    "week": target_week,
                    "opp_team": opp_team,
                    "opp_allowed_fppg": mean_fppg,
                }
            )

    return pd.DataFrame(rows, columns=["season", "week", "opp_team", "opp_allowed_fppg"])
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest -v tests/test_features/test_opponent.py
```

Expected: 2 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v tests/test_features && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/features/_opponent.py tests/test_features/test_opponent.py
git commit -m "feat(features): add _opponent.py — opp_allowed_fppg trailing proxy

For each (opp_team, season, week), the trailing-N-week mean of
position-specific fantasy points the opponent has allowed. v1 stand-in
for true play-by-play EPA (deferred to a later plan). Calls
scoring.score() — no reimplementation of fantasy math."
```

---

### Task 15: `src/projections/features/wr.py` — `build_wr_features`

The non-leakage tests for the builder live here. Leakage tests (the load-bearing ones) get their own task (16) so a leak surfaces with a precise failure rather than mixed in with shape/correctness checks.

**Files:**
- Create: `src/projections/features/wr.py`
- Modify: `src/projections/features/__init__.py` (re-export `build_wr_features`)
- Create: `tests/test_features/test_wr.py`
- Modify: `tests/test_features/conftest.py` (add shared synthetic WR-context frames)

- [ ] **Step 1: Add shared synthetic frames to `tests/test_features/conftest.py`**

Replace the placeholder docstring file with:

```python
"""Shared synthetic frames for feature-builder tests.

Each fixture returns a SCHEMA-VALID frame (already-normalized, already
typed) — these mimic the output of `read_partition`, not the raw
nfl_data_py response. Tests build features directly from these without
going through the ingest layer."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR


@pytest.fixture
def wr_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 3 WRs across 2 teams (MIN, KC).

    Designed so trailing-4 windows have round-number expectations:
    - Justin Jefferson (MIN, gsis_id=00-0036322): 12/10/8/6 targets weeks 1-4,
      14/12/10/8 weeks 5-8.
    - Jaylen Reed (MIN, gsis_id=00-0036323, made-up): 4/4/4/4 targets weeks 1-4,
      6/6/6/6 weeks 5-8.
    - Rashee Rice (KC, gsis_id=00-0034950): 8/8/8/8 targets weeks 1-4,
      10/10/10/10 weeks 5-8.

    All three play opponent rotation: weeks 1-4 vs DET, weeks 5-8 vs CHI."""
    rows = []
    for week in range(1, 9):
        opp = "DET" if week <= 4 else "CHI"

        # Jefferson (MIN)
        jef_targets = [12, 10, 8, 6, 14, 12, 10, 8][week - 1]
        rows.append(
            {
                "gsis_id": "00-0036322",
                "season": 2024,
                "week": week,
                "position": "WR",
                "team": "MIN",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": jef_targets - 2,
                "receiving_yards": float(jef_targets * 12),
                "receiving_tds": 1,
                "receiving_air_yards": float(jef_targets * 14),
                "targets": jef_targets,
                "fumbles_lost": 0,
            }
        )

        # Reed (MIN secondary WR — needed so target_share isn't 100%)
        reed_targets = 4 if week <= 4 else 6
        rows.append(
            {
                "gsis_id": "00-0036323",
                "season": 2024,
                "week": week,
                "position": "WR",
                "team": "MIN",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": reed_targets - 1,
                "receiving_yards": float(reed_targets * 8),
                "receiving_tds": 0,
                "receiving_air_yards": float(reed_targets * 9),
                "targets": reed_targets,
                "fumbles_lost": 0,
            }
        )

        # Rice (KC)
        rice_targets = 8 if week <= 4 else 10
        rows.append(
            {
                "gsis_id": "00-0034950",
                "season": 2024,
                "week": week,
                "position": "WR",
                "team": "KC",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": rice_targets - 2,
                "receiving_yards": float(rice_targets * 11),
                "receiving_tds": 0,
                "receiving_air_yards": float(rice_targets * 13),
                "targets": rice_targets,
                "fumbles_lost": 0,
            }
        )

    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_snap_counts(wr_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the same WRs/weeks. ~95% snap pct uniformly."""
    rows = []
    for _, r in wr_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 60,
                "offense_pct": 0.95,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 2,
                "st_pct": 0.05,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_depth_charts() -> pd.DataFrame:
    """Depth chart snapshot for week 5 of 2024 (the typical as_of_week
    for tests). Jefferson + Rice as WR1; Reed as WR2."""
    rows = [
        {
            "gsis_id": "00-0036322",
            "season": 2024,
            "week": 5,
            "team": "MIN",
            "position": "WR",
            "depth_team": "WR1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0036323",
            "season": 2024,
            "week": 5,
            "team": "MIN",
            "position": "WR",
            "depth_team": "WR2",
            "depth_rank": 2,
        },
        {
            "gsis_id": "00-0034950",
            "season": 2024,
            "week": 5,
            "team": "KC",
            "position": "WR",
            "depth_team": "WR1",
            "depth_rank": 1,
        },
    ]
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_ngs_receiving() -> pd.DataFrame:
    """NGS receiving snapshots through week 4 of 2024 for the 3 WRs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0036322",
                    "season": 2024,
                    "week": week,
                    "team": "MIN",
                    "position": "WR",
                    "avg_cushion": 5.0,
                    "avg_separation": 3.2,
                    "avg_intended_air_yards": 12.0,
                    "percent_share_of_intended_air_yards": 30.0,
                    "receptions": 9,
                    "targets": 12,
                    "catch_percentage": 75.0,
                    "yards": 110,
                    "rec_touchdowns": 1,
                    "avg_yac": 4.0,
                    "avg_expected_yac": 3.5,
                    "avg_yac_above_expectation": 0.5,
                },
                {
                    "gsis_id": "00-0036323",
                    "season": 2024,
                    "week": week,
                    "team": "MIN",
                    "position": "WR",
                    "avg_cushion": 6.5,
                    "avg_separation": 2.8,
                    "avg_intended_air_yards": 9.0,
                    "percent_share_of_intended_air_yards": 15.0,
                    "receptions": 3,
                    "targets": 4,
                    "catch_percentage": 75.0,
                    "yards": 32,
                    "rec_touchdowns": 0,
                    "avg_yac": 3.0,
                    "avg_expected_yac": 3.0,
                    "avg_yac_above_expectation": 0.0,
                },
                {
                    "gsis_id": "00-0034950",
                    "season": 2024,
                    "week": week,
                    "team": "KC",
                    "position": "WR",
                    "avg_cushion": 5.5,
                    "avg_separation": 3.0,
                    "avg_intended_air_yards": 10.0,
                    "percent_share_of_intended_air_yards": 25.0,
                    "receptions": 6,
                    "targets": 8,
                    "catch_percentage": 75.0,
                    "yards": 88,
                    "rec_touchdowns": 0,
                    "avg_yac": 4.5,
                    "avg_expected_yac": 4.0,
                    "avg_yac_above_expectation": 0.5,
                },
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def wr_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: MIN @ CHI, KC @ CHI (made up to keep both
    WR teams pointing at the same opponent for trivially-checkable opponent
    proxy joins)."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_MIN_CHI", "2024_05_KC_CHI"], dtype=_PYARROW_STR),
            "home_team": pd.array(["CHI", "CHI"], dtype=_PYARROW_STR),
            "away_team": pd.array(["MIN", "KC"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [-3.5, -7.5],  # MIN favored by 3.5; KC by 7.5
            "total_line": [48.5, 51.0],
            "home_moneyline": pd.array([155, 280], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180, -340], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass", "grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors", "outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([55, 55], dtype=pd.Int64Dtype()),
            "wind": pd.array([8, 8], dtype=pd.Int64Dtype()),
        }
    )
```

- [ ] **Step 2: Write failing tests for `build_wr_features`**

Create `tests/test_features/test_wr.py`:

```python
"""WR feature builder tests (non-leakage). Leakage tests live in test_wr_leakage.py
so a leak surfaces with a precise failure independent of shape/correctness checks."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_wr_features
from projections.schemas import WrFeaturesSchema


def test_build_wr_features_returns_validated_frame(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    WrFeaturesSchema.validate(out)


def test_build_wr_features_one_row_per_rostered_wr(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    # 3 WRs on rosters in week 5 → 3 rows.
    assert len(out) == 3
    assert set(out["gsis_id"]) == {"00-0036322", "00-0036323", "00-0034950"}


def test_build_wr_features_targets_per_game_l4_correct(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """Jefferson weeks 1-4: 12/10/8/6 → mean = 9.0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["targets_per_game_l4"] == 9.0


def test_build_wr_features_target_share_l4_correct(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """Jefferson MIN trailing-4 targets: 12+10+8+6 = 36.
    Reed MIN trailing-4 targets: 4+4+4+4 = 16. MIN total = 52.
    Jefferson share = 36/52 ≈ 0.6923."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["target_share_l4"] == pytest.approx(36 / 52, abs=1e-6)


def test_build_wr_features_designed_rusher_false_for_pure_wrs(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """All 3 fixture WRs have 0 carries → designed_rusher == False."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    assert not out["designed_rusher"].any()


def test_build_wr_features_designed_rusher_true_above_threshold(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """Inject 2 carries/game over weeks 1-4 for Jefferson (8 carries total
    in trailing 4 → 2.0 carries/game ≥ 1.5 threshold)."""
    ws = wr_weekly_stats.copy()
    mask = (ws["gsis_id"] == "00-0036322") & (ws["week"] <= 4)
    ws.loc[mask, "carries"] = 2

    out = build_wr_features(
        weekly_stats=ws,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert bool(jef["designed_rusher"]) is True


def test_build_wr_features_implied_team_total_from_schedules(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """MIN implied total = (total + |spread|)/2 when MIN favored.
    total=48.5, MIN favored by 3.5 → MIN implied = 26.0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["implied_team_total"] == pytest.approx(26.0, abs=1e-6)
    assert bool(jef["is_home"]) is False  # MIN is the away team


def test_build_wr_features_rookie_with_no_prior_games_zeros(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """Inject a rookie WR on KC with NO prior weeks of data; depth chart shows
    them at WR2. They get a row with l4 stats == 0 (or NaN), no crash."""
    extra_dc = pd.concat(
        [
            wr_depth_charts,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099777",
                        "season": 2024,
                        "week": 5,
                        "team": "KC",
                        "position": "WR",
                        "depth_team": "WR2",
                        "depth_rank": 2,
                    }
                ]
            ).astype({"gsis_id": "string[pyarrow]"}),
        ],
        ignore_index=True,
    )

    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=extra_dc,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    rookie = out[out["gsis_id"] == "00-0099777"].iloc[0]
    assert rookie["targets_per_game_l4"] == 0.0
    assert rookie["receiving_yards_per_game_l4"] == 0.0
    assert bool(rookie["designed_rusher"]) is False
```

- [ ] **Step 3: Run, verify ImportError**

```bash
pytest -v tests/test_features/test_wr.py
```

Expected: FAIL with ImportError.

- [ ] **Step 4: Implement `src/projections/features/wr.py`**

```python
"""WR feature builder. Pure function — no I/O, no caching.

Output is one row per (gsis_id, season, week=as_of_week) for every WR on
a roster in week as_of_week of season. Validates against WrFeaturesSchema."""

from __future__ import annotations

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import last_n_per_group
from projections.schemas import (
    Position,
    Ruleset,
    WrFeaturesSchema,
    _PYARROW_STR,
)

_DESIGNED_RUSHER_THRESHOLD = 1.5  # carries/game over trailing 4


def _prior_mask(df: pd.DataFrame, *, season: int, as_of_week: int) -> pd.Series:
    return (df["season"] < season) | (
        (df["season"] == season) & (df["week"] < as_of_week)
    )


def _exact_week_mask(df: pd.DataFrame, *, season: int, as_of_week: int) -> pd.Series:
    return (df["season"] == season) & (df["week"] == as_of_week)


def _trailing_4_per_player(
    weekly_stats: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """Return a per-player frame with `mean_l4` = mean of `value_col` over the
    trailing 4 games. Players with 0 prior games contribute 0.0."""
    if weekly_stats.empty:
        return pd.DataFrame(columns=["gsis_id", "mean_l4"]).astype(
            {"gsis_id": _PYARROW_STR, "mean_l4": float}
        )
    last4 = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    return (
        last4.groupby("gsis_id", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "mean_l4"})
    )


def _trailing_4_share_per_team(
    weekly_stats: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """For each (player, team), share of `value_col` over the team's WR group
    in the trailing 4 games. Returns frame keyed by gsis_id with `share_l4`."""
    if weekly_stats.empty:
        return pd.DataFrame(columns=["gsis_id", "share_l4"]).astype(
            {"gsis_id": _PYARROW_STR, "share_l4": float}
        )
    # Player trailing-4 sum
    last4_player = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    player_sum = last4_player.groupby(["gsis_id", "team"], as_index=False)[
        value_col
    ].sum()

    # Team trailing-4 sum across all WRs on the team (use the SAME 4 weeks each
    # WR's window selected; but for share we want team-week totals over the
    # union of weeks each WR appears in — approximation: sum across the same
    # last 4 weeks per WR, then sum to team level).
    team_sum = (
        player_sum.groupby("team", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "team_total"})
    )

    merged = player_sum.merge(team_sum, on="team", how="left")
    merged["share_l4"] = merged.apply(
        lambda r: float(r[value_col] / r["team_total"]) if r["team_total"] > 0 else 0.0,
        axis=1,
    )
    return merged[["gsis_id", "share_l4"]]


def _latest_ngs_snapshot(ngs: pd.DataFrame) -> pd.DataFrame:
    """Per-player most recent NGS snapshot in the (already-prior-filtered) frame.
    Returns one row per gsis_id with the season-to-date columns the WR builder
    propagates as `*_std` features."""
    if ngs.empty:
        return pd.DataFrame()
    latest = (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False)
        .tail(1)
    )
    return latest


def build_wr_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the WR feature DataFrame for week `as_of_week` of `season`.

    Inputs are validated against their respective schemas (caller's
    responsibility). The function filters every input to leakage-safe rows
    before computing anything — see _prior_mask / _exact_week_mask.
    """
    # --- Leakage-safe input filtering ---
    ws = weekly_stats[
        _prior_mask(weekly_stats, season=season, as_of_week=as_of_week)
    ].copy()
    sc = snap_counts[
        _prior_mask(snap_counts, season=season, as_of_week=as_of_week)
    ].copy()
    ngs = ngs_receiving[
        _prior_mask(ngs_receiving, season=season, as_of_week=as_of_week)
    ].copy()
    dc = depth_charts[
        _exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)
    ].copy()
    sch = schedules[_exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    # --- Rostered WRs in target week (depth chart drives roster set) ---
    wr_dc = dc[dc["position"] == Position.WR.value].copy()
    if wr_dc.empty:
        return WrFeaturesSchema.validate(
            pd.DataFrame(columns=list(WrFeaturesSchema.to_schema().columns.keys()))
        )

    # Restrict prior frames to WR position
    ws_wr = ws[ws["position"] == Position.WR.value].copy()
    sc_wr = sc[sc["position"] == Position.WR.value].copy()

    # --- Per-player rolling features ---
    targets_l4 = _trailing_4_per_player(ws_wr, "targets").rename(
        columns={"mean_l4": "targets_per_game_l4"}
    )
    rec_l4 = _trailing_4_per_player(ws_wr, "receptions").rename(
        columns={"mean_l4": "receptions_per_game_l4"}
    )
    rec_yd_l4 = _trailing_4_per_player(ws_wr, "receiving_yards").rename(
        columns={"mean_l4": "receiving_yards_per_game_l4"}
    )
    rec_td_l4 = _trailing_4_per_player(ws_wr, "receiving_tds").rename(
        columns={"mean_l4": "receiving_tds_per_game_l4"}
    )
    rush_att_l4 = _trailing_4_per_player(ws_wr, "carries").rename(
        columns={"mean_l4": "rushing_attempts_per_game_l4"}
    )
    rush_yd_l4 = _trailing_4_per_player(ws_wr, "rushing_yards").rename(
        columns={"mean_l4": "rushing_yards_per_game_l4"}
    )

    # Season-to-date targets per game (mean across all prior weeks within season)
    ws_this_season = ws_wr[ws_wr["season"] == season]
    if ws_this_season.empty:
        targets_std = pd.DataFrame(columns=["gsis_id", "targets_per_game_std"]).astype(
            {"gsis_id": _PYARROW_STR, "targets_per_game_std": float}
        )
    else:
        targets_std = (
            ws_this_season.groupby("gsis_id", as_index=False)["targets"]
            .mean()
            .rename(columns={"targets": "targets_per_game_std"})
        )

    # Target share + air-yards share over trailing 4
    target_share = _trailing_4_share_per_team(ws_wr, "targets").rename(
        columns={"share_l4": "target_share_l4"}
    )
    air_yards_share = _trailing_4_share_per_team(ws_wr, "receiving_air_yards").rename(
        columns={"share_l4": "air_yards_share_l4"}
    )

    # Snap pct trailing 4
    snap_l4_raw = _trailing_4_per_player(sc_wr, "offense_pct").rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # NGS latest snapshot per player → *_std columns
    ngs_latest = _latest_ngs_snapshot(ngs)
    ngs_cols = pd.DataFrame()
    if not ngs_latest.empty:
        ngs_cols = ngs_latest[
            [
                "gsis_id",
                "avg_separation",
                "avg_intended_air_yards",
                "percent_share_of_intended_air_yards",
                "avg_yac_above_expectation",
            ]
        ].rename(
            columns={
                "avg_separation": "avg_separation_std",
                "avg_intended_air_yards": "avg_intended_air_yards_std",
                "percent_share_of_intended_air_yards": "percent_share_intended_air_yards_std",
                "avg_yac_above_expectation": "avg_yac_above_expectation_std",
            }
        )
        # Convert NGS share from 0-100 percent to 0-1 fraction to match schema range.
        ngs_cols["percent_share_intended_air_yards_std"] = (
            ngs_cols["percent_share_intended_air_yards_std"] / 100.0
        )

    # --- Game environment from schedules ---
    # Build a per-team game-environment row for week `as_of_week`.
    home = sch[["season", "week", "home_team", "away_team", "spread_line", "total_line", "roof"]].rename(
        columns={"home_team": "team", "away_team": "opp_team"}
    )
    home["is_home"] = True
    home["spread"] = home["spread_line"]  # negative if home favored
    away = sch[["season", "week", "home_team", "away_team", "spread_line", "total_line", "roof"]].rename(
        columns={"away_team": "team", "home_team": "opp_team"}
    )
    away["is_home"] = False
    away["spread"] = -away["spread_line"]  # invert sign for the away team's perspective
    game_env = pd.concat([home, away], ignore_index=True)
    # Implied team total = (total - team's spread) / 2 (where team's spread is
    # negative if favored). Per-team-perspective spread is signed in `spread`.
    game_env["implied_team_total"] = (game_env["total_line"] - game_env["spread"]) / 2
    game_env["roof_dome"] = game_env["roof"].isin(["dome", "closed"]).fillna(False)
    game_env = game_env[
        ["season", "week", "team", "opp_team", "is_home", "spread", "implied_team_total", "roof_dome"]
    ]

    # --- Opponent strength proxy ---
    opp_proxy_full = opp_allowed_fppg(
        ws_wr, position=Position.WR, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_wr_fppg_l4"})

    # --- Assemble: start from depth chart (rostered WRs), join everything else ---
    out = wr_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(targets_l4, on="gsis_id", how="left")
    out = out.merge(targets_std, on="gsis_id", how="left")
    out = out.merge(target_share, on="gsis_id", how="left")
    out = out.merge(air_yards_share, on="gsis_id", how="left")
    out = out.merge(rec_l4, on="gsis_id", how="left")
    out = out.merge(rec_yd_l4, on="gsis_id", how="left")
    out = out.merge(rec_td_l4, on="gsis_id", how="left")
    out = out.merge(rush_att_l4, on="gsis_id", how="left")
    out = out.merge(rush_yd_l4, on="gsis_id", how="left")
    out = out.merge(snap_l4_raw, on="gsis_id", how="left")
    if not ngs_cols.empty:
        out = out.merge(ngs_cols, on="gsis_id", how="left")
    else:
        for col in (
            "avg_separation_std",
            "avg_intended_air_yards_std",
            "percent_share_intended_air_yards_std",
            "avg_yac_above_expectation_std",
        ):
            out[col] = pd.NA
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_wr_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    # Fill l4 zeros for rookies with no prior games.
    zero_fills = [
        "targets_per_game_l4",
        "targets_per_game_std",
        "target_share_l4",
        "receptions_per_game_l4",
        "receiving_yards_per_game_l4",
        "receiving_tds_per_game_l4",
        "rushing_attempts_per_game_l4",
        "rushing_yards_per_game_l4",
    ]
    for c in zero_fills:
        out[c] = out[c].fillna(0.0).astype(float)

    out["designed_rusher"] = (
        out["rushing_attempts_per_game_l4"] >= _DESIGNED_RUSHER_THRESHOLD
    )
    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())

    # Schema requires team/opponent as Series[str] — our merges may have
    # introduced object dtype.
    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    out = WrFeaturesSchema.validate(out)
    return out
```

- [ ] **Step 5: Update `src/projections/features/__init__.py` to re-export**

```python
"""Per-position feature builders. Pure functions; no I/O."""

from __future__ import annotations

from projections.features.wr import build_wr_features

__all__ = ["build_wr_features"]
```

- [ ] **Step 6: Run tests, verify pass**

```bash
pytest -v tests/test_features/test_wr.py
```

Expected: 8 PASS. If a test fails, the most likely culprits in order: (a) `_trailing_4_share_per_team` math, (b) `implied_team_total` sign convention, (c) NGS share scaling (0-100 vs 0-1).

- [ ] **Step 7: Quality gate + commit**

```bash
pytest -v tests/test_features tests/test_ingest tests/test_schemas && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/features/wr.py src/projections/features/__init__.py tests/test_features/conftest.py tests/test_features/test_wr.py
git commit -m "feat(features): add build_wr_features — pure-function WR feature builder

Consumes 5 raw tables (weekly_stats, snap_counts, depth_charts,
ngs_receiving, schedules), filters each to leakage-safe rows, computes
trailing-4 + season-to-date features, joins game environment from
schedules and opponent-strength proxy, validates output against
WrFeaturesSchema. Pure function — no I/O, no caching.

Leakage tests live in test_wr_leakage.py (next task)."
```

---

### Task 16: Leakage tests for `build_wr_features`

The load-bearing tests. One assertion per input source so a leak surfaces with a precise failure ("ngs_receiving leaks", not "something leaks").

**Files:**
- Create: `tests/test_features/test_wr_leakage.py`

- [ ] **Step 1: Write 5 leakage tests (one per input source)**

Create `tests/test_features/test_wr_leakage.py`:

```python
"""Leakage tests for build_wr_features.

Strategy: build features for (season=2024, as_of_week=5). Then for each input
frame independently, fabricate implausible rows for week ≥ 5, rebuild, and
assert the output is byte-equal to the original. Five tests, one per input
source — a leak in any single source surfaces with a precise failure.

Why byte-equal not "values match within tolerance": a leak by definition
changes the computation, so the output will differ. We want the strongest
possible signal."""

from __future__ import annotations

import pandas as pd

from projections.features import build_wr_features
from projections.schemas import _PYARROW_STR


_AS_OF_WEEK = 5
_SEASON = 2024


def _baseline(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> pd.DataFrame:
    return build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def test_no_leakage_from_weekly_stats(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        wr_weekly_stats, wr_snap_counts, wr_depth_charts, wr_ngs_receiving, wr_schedules
    )

    # Inject implausible weeks 5-8: Jefferson records 1000 receiving yards every week
    leaky = wr_weekly_stats.copy()
    mask_future = (leaky["gsis_id"] == "00-0036322") & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[mask_future, "receiving_yards"] = 1000.0
    leaky.loc[mask_future, "targets"] = 30
    leaky.loc[mask_future, "receptions"] = 25
    leaky.loc[mask_future, "carries"] = 10

    after = build_wr_features(
        weekly_stats=leaky,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_snap_counts(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        wr_weekly_stats, wr_snap_counts, wr_depth_charts, wr_ngs_receiving, wr_schedules
    )

    leaky = wr_snap_counts.copy()
    mask_future = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[mask_future, "offense_pct"] = 0.0  # implausible: no snaps for any WR
    leaky.loc[mask_future, "offense_snaps"] = 0

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=leaky,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_ngs_receiving(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        wr_weekly_stats, wr_snap_counts, wr_depth_charts, wr_ngs_receiving, wr_schedules
    )

    # Inject NGS rows for week 5+ (the fixture only goes through week 4).
    extra = pd.DataFrame(
        [
            {
                "gsis_id": "00-0036322",
                "season": 2024,
                "week": 5,
                "team": "MIN",
                "position": "WR",
                "avg_cushion": 99.0,
                "avg_separation": 99.0,
                "avg_intended_air_yards": 99.0,
                "percent_share_of_intended_air_yards": 99.0,
                "receptions": 99,
                "targets": 99,
                "catch_percentage": 99.0,
                "yards": 999,
                "rec_touchdowns": 9,
                "avg_yac": 99.0,
                "avg_expected_yac": 99.0,
                "avg_yac_above_expectation": 99.0,
            }
        ]
    ).astype({"gsis_id": _PYARROW_STR, "team": _PYARROW_STR, "position": _PYARROW_STR})
    leaky = pd.concat([wr_ngs_receiving, extra], ignore_index=True)

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=leaky,
        schedules=wr_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_depth_charts_other_weeks(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """Depth chart from OTHER weeks must not affect the as_of_week=5 build —
    we read only the target-week snapshot."""
    baseline = _baseline(
        wr_weekly_stats, wr_snap_counts, wr_depth_charts, wr_ngs_receiving, wr_schedules
    )

    extra_weeks = pd.concat(
        [
            wr_depth_charts.assign(week=4, depth_rank=99, depth_team="WR99"),
            wr_depth_charts.assign(week=6, depth_rank=99, depth_team="WR99"),
        ],
        ignore_index=True,
    )
    leaky = pd.concat([wr_depth_charts, extra_weeks], ignore_index=True)
    # depth_rank=99 violates schema, so skip schema validate and pass raw —
    # build_wr_features filters first by exact_week_mask.

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=leaky,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_schedules_other_weeks(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    """Schedule rows from OTHER weeks must not affect the as_of_week=5 build."""
    baseline = _baseline(
        wr_weekly_stats, wr_snap_counts, wr_depth_charts, wr_ngs_receiving, wr_schedules
    )

    extra_weeks = wr_schedules.assign(
        week=6, total_line=99.0, spread_line=99.0
    )
    leaky = pd.concat([wr_schedules, extra_weeks], ignore_index=True)

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=leaky,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)
```

- [ ] **Step 2: Run leakage tests, verify all pass**

```bash
pytest -v tests/test_features/test_wr_leakage.py
```

Expected: 5 PASS. **If any fail**, that's a real leak: investigate `_prior_mask` / `_exact_week_mask` usage in `wr.py` — every input source should be filtered before any computation. Don't tweak the test to make it pass; fix the leak.

- [ ] **Step 3: Quality gate + commit**

```bash
pytest -v tests/test_features tests/test_ingest tests/test_schemas && mypy src tests && ruff check src tests && ruff format --check src tests
git add tests/test_features/test_wr_leakage.py
git commit -m "test(features): add 5 leakage tests for build_wr_features

One assertion per input source. A leak in any single input surfaces
with a precise failure (\"ngs_receiving leaks\" rather than \"something
leaks\"). Strategy: inject implausible rows for week >= as_of_week,
rebuild, assert byte-equality with baseline output."
```

---

### Task 17: End-to-end smoke test

Catches integration gaps that per-module tests miss (partition path mismatches between `write_partition` and `read_partition`, manifest update bugs, schema dtype drift between ingest output and feature input).

**Files:**
- Create: `tests/test_smoke_2a.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_smoke_2a.py`:

```python
"""End-to-end smoke test for Plan 2a deliverables.

Wires every new ingest module and the WR feature builder together against
synthetic fixtures. Catches integration gaps per-module tests miss:
- write_partition / read_partition path conventions matching
- Manifest update behavior across multiple tables
- Dtype drift between ingest output and feature input"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.features import build_wr_features
from projections.ingest import (
    refresh_depth_charts,
    refresh_ngs,
    refresh_schedules,
    refresh_snap_counts,
    refresh_weekly_stats,
)
from projections.ingest.manifest import read_manifest
from projections.schemas import WrFeaturesSchema
from projections.store import read_partition


def test_end_to_end_ingest_and_wr_features(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    fake_schedules_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    fake_depth_charts_df: pd.DataFrame,
    fake_ngs_receiving_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch every fetcher to return the synthetic fixture
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: fake_snap_counts_df,
    )
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_ngs_receiving_df,
    )

    # 1) Ingest everything for season 2024
    refresh_weekly_stats(tmp_path, seasons=[2024])
    refresh_schedules(tmp_path, seasons=[2024])
    refresh_snap_counts(tmp_path, seasons=[2024])
    refresh_depth_charts(tmp_path, seasons=[2024])
    refresh_ngs(tmp_path, stat_type="receiving", seasons=[2024])

    # 2) Manifest has one row per ingest table
    manifest = read_manifest(tmp_path)
    tables_in_manifest = set(manifest["table"].tolist())
    assert {"weekly_stats", "schedules", "snap_counts", "depth_charts", "ngs_receiving"} <= tables_in_manifest

    # 3) Read each partition back
    weekly = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    schedules = read_partition(tmp_path / "raw", "schedules", season=2024)
    snaps = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    depth = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    ngs = read_partition(tmp_path / "raw", "ngs_receiving", season=2024)

    # 4) The synthetic fixtures all describe week 3 of 2024. The WR feature
    # builder requires depth chart + schedule for the as_of_week.
    # The synthetic fixtures only cover week 3, but build_wr_features for
    # season=2024 as_of_week=4 has prior weeks (week 3) and looks for week 4
    # depth/schedule rows. We need to inject those.
    extra_dc = pd.concat(
        [depth, depth.assign(week=4)],
        ignore_index=True,
    )
    extra_sched = pd.concat(
        [schedules, schedules.assign(week=4)],
        ignore_index=True,
    )

    # 5) Build WR features for as_of_week=4 (so week 3 is in the prior window)
    out = build_wr_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_receiving=ngs,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )

    # 6) Output validates and has at least one row (Jefferson is the lone WR
    # in fake_weekly_df / fake_depth_charts_df).
    WrFeaturesSchema.validate(out)
    assert len(out) >= 1
    assert "00-0036322" in out["gsis_id"].tolist()
```

- [ ] **Step 2: Run, verify pass**

```bash
pytest -v tests/test_smoke_2a.py
```

Expected: 1 PASS. Common failure: a column dtype emerges from `read_partition` differently than the in-memory frame produced by `_normalize_one_season` (e.g., parquet round-trip loses pyarrow string dtype). If so, fix the dtype handling at the `read_partition` consumer rather than weakening the schema.

- [ ] **Step 3: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add tests/test_smoke_2a.py
git commit -m "test: add end-to-end smoke test wiring all ingest + WR features

Catches integration gaps per-module tests miss: partition path
conventions, manifest update across multiple tables, dtype round-trip
through parquet."
```

---

## Phase 5 — Documentation + wrap-up

### Task 18: Update `project_management.md` and `TODO.md`

Lands the spec §10 documentation updates on the same PR. Decisions captured *as they actually executed* — if anything in §10.1 changed during implementation, edit the entries to match reality before committing.

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`

- [ ] **Step 1: Re-read current `project_management.md` and `TODO.md`**

```bash
cat project_management.md
cat TODO.md
```

- [ ] **Step 2: Append decision-log rows in `project_management.md`**

In `project_management.md`, in the `## Decision log` table, append the 8 rows from spec §10.1. **Re-check each row against what actually happened during implementation** — if Task 6 had to compromise the NGS schema, update the matching decision row before committing. Append at the top of the table (newest-first, matching existing convention):

```markdown
| 2026-04-24 | Drive-by cleanups (`_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, ingest `__all__`) folded into 2a | We're touching every ingest module anyway; cheaper to clean up once than across two PRs |
| 2026-04-24 | Schedule ingest captures Vegas lines (spread, total, moneyline) | "Implied team total" is a load-bearing feature for every offensive position |
| 2026-04-24 | Shared `_rolling.py` and `_opponent.py` helpers built and tested in 2a | Pin helper API on the first builder so 2b's five other builders consume a stable contract |
| 2026-04-24 | Opponent strength via `opp_allowed_fppg_l4` proxy in 2a, not play-by-play EPA | True EPA needs play-by-play ingest (separate concern, deferred); the FPPG-allowed proxy is sufficient for v1 baseline |
| 2026-04-24 | Ingest all three NGS stat types (passing, rushing, receiving) in 2a, even though only NGS receiving is consumed by WR | The hard part of NGS ingest is the snapshot/partition decision; make it once across all three rather than three times |
| 2026-04-24 | Feature builders are pure functions in 2a — no parquet storage | Output is small (~1.8K rows/season for WR) and computes in milliseconds; defer caching until backtest performance demands it (Plan 3+) |
| 2026-04-24 | WR is the first end-to-end position | Exercises every new ingest source (snap_counts, depth_charts, NGS receiving) in one builder; surfaces design issues before propagating to other positions |
| 2026-04-24 | Split Plan 2 into 2a (ingest expansion + WR feature builder) and 2b (QB / RB / TE / K / DST feature builders) | Validate the feature-builder pattern end-to-end on one position before copy-pasting across five files; isolate ingest (mechanical) from features (greenfield design) |
| 2026-04-24 | Test fixtures are synthetic in-memory `pd.DataFrame`s, not real-data parquet snapshots | Matches existing convention from foundations (`fake_weekly_df` etc.); simpler maintenance; `nfl_data_py` API drift is handled separately by opt-in network smoke tests (TODO #8) |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `targets`, `receiving_air_yards`, `carries` | Discovered during plan-writing: WR feature builder needs these source columns and the foundations-era schema didn't include them. All three are present in raw `nfl_data_py.import_weekly_data` output |
```

- [ ] **Step 3: Update the "Current status" / "Next action" sections**

Replace the existing `## Current status (as of 2026-04-24)` block with:

```markdown
## Current status (as of 2026-04-24)

**Projections Core — Plan 2a (Ingest expansion + WR feature builder) merged to `main` at commit `<TBD-after-merge>`.**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Dev tooling merged via `feat/dev-tooling`.

**Plan 2a delivered:**
- Four new ingest modules (`schedules`, `snap_counts`, `depth_charts`, `ngs` parameterized over passing/rushing/receiving) + 6 new partition tables, all idempotent.
- `WeeklyStatsSchema` extended with `targets`, `receiving_air_yards`, `carries` (foundations-era omission).
- `src/projections/features/` package with shared `_rolling.py`, `_opponent.py` helpers and a fully-tested `build_wr_features` (pure-function, no parquet storage).
- ~50 new tests (~140 total). 5 leakage tests for the WR builder, one per input source.
- Drive-by cleanups: `_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, trimmed ingest `__all__`.
```

Replace the `## Next action` block with:

```markdown
## Next action

**Recommended: Plan 2b — per-position feature builders for QB / RB / TE / K / DST.**

2a validated the feature-builder pattern (signatures, leakage prevention, schemas, tests) end-to-end on WR. 2b copy-pastes the pattern across the remaining five positions, reusing `_rolling.py` and `_opponent.py` from 2a. Each position gets its own pandera schema (`QbFeaturesSchema`, etc.) following `WrFeaturesSchema` as the template. One PR per position or one bundled — TBD when 2b is brainstormed.

After 2b, Plan 3 (Model A baseline + season aggregation + backtest harness) becomes unblocked.
```

- [ ] **Step 4: Append items #2-#8 to `TODO.md`**

In `TODO.md`, after the existing `### 1. Explore option D: joint-correlation projections` section, append:

```markdown
### 2. Plan 2b — remaining position feature builders

QB, RB, TE, K, DST. Each consuming the validated `wr.py` pattern, `_rolling.py`, and `_opponent.py` helpers from 2a. Each gets its own pandera schema (`QbFeaturesSchema`, `RbFeaturesSchema`, etc.). One PR per position or one bundled — TBD when 2b is brainstormed.

### 3. Play-by-play ingest (`nfl_data_py.import_pbp_data`)

Required for true opponent-adjusted EPA features. Defer until Plan 3 backtest reveals whether the `opp_allowed_fppg_l4` proxy is good enough. If not, ingest PBP and add EPA-derived opponent features in a focused plan.

### 4. Decide feature parquet storage during Plan 3

Gated on backtest performance: if a single training pass takes >~30s recomputing features, add `data/features/{position}/...` storage and a `refresh_features` CLI verb; otherwise stay pure-function.

### 5. NGS missing-data forward-fill policy

v1 leaves NaN. Revisit after a notebook investigation against a recent season quantifying how often qualifying-threshold misses happen and whether forward-fill changes feature distributions materially.

### 6. Opening / week-of Vegas line source

`import_schedules` returns *closing* lines. Closing is fine for backtest. Only worth pursuing if Plan 5 ever projects pre-week selections (e.g., DFS workflow uses lines that change through the week).

### 7. Depth chart slot-label parser refinement

v1 extracts the trailing digit from labels like `WR1`, falling back to `1` for unrankable labels (`LWR`/`RWR`/`SWR`) with a warning. If Plan 3 model fitting shows `depth_rank` is noisy or wrong, build a richer parser using alignment + rank.

### 8. Build opt-in `nfl_data_py` API-drift smoke tests

One per ingest source, marked `@pytest.mark.network`, skipped by default. Hits the live API, fetches a tiny slice (e.g., 1 week of 2023), asserts the column set matches the schema. Run manually after `nfl_data_py` version bumps. Document the run-after-bump step in `CONTRIBUTING.md`. The synthetic in-memory fixtures used by 2a's CI tests don't catch API drift on their own.
```

- [ ] **Step 5: Verify formatting + commit**

```bash
# pre-commit's `trim trailing whitespace` + `fix end of files` may reformat — let it.
git add project_management.md TODO.md
git commit -m "docs(pm): close Plan 2a; queue Plan 2b + follow-up TODOs

Decision-log additions cover the 10 major design calls from Plan 2a
(per spec §10). Status section now reflects 2a complete; next action is
Plan 2b. TODO.md gains items #2-#8 for follow-up work explicitly
deferred from 2a."
```

---

### Task 19: End-of-effort verification + open PR

Run the full quality gate one more time, summarize results, then open the PR.

**Files:**
- None modified — verification + PR only.

- [ ] **Step 1: Run the full quality gate at the worktree root**

```bash
cd "/c/Users/alden/FantasyFootball/.worktrees/feat-plan-2a-ingest-and-wr-features"
. .venv/Scripts/activate
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
pytest -v -k "ingest or store or schemas"     # CLAUDE.md required for any ingest/store/schema touch
```

Each must be green. Test count expectation: ~135-140 total (89 baseline + ~50 new). If anything fails, fix before opening the PR — don't open with a known-broken main check.

- [ ] **Step 2: Capture results for the PR description**

Save the green output (or a concise summary: "pytest: 137 passed in 8.2s; mypy: success no issues; ruff check: All checks passed; ruff format: 0 files would be reformatted") for inclusion in the PR body.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feat/plan-2a-ingest-and-wr-features
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "Plan 2a: ingest expansion + WR feature builder" --body "$(cat <<'EOF'
## Summary
- Adds four new `nfl_data_py` ingest modules (`schedules`, `snap_counts`, `depth_charts`, `ngs` parameterized over passing/rushing/receiving) producing 6 new partition tables.
- Extends `WeeklyStatsSchema` with `targets`, `receiving_air_yards`, `carries` — foundations-era omission discovered during plan-writing.
- Stands up `src/projections/features/` with shared `_rolling.py` / `_opponent.py` helpers and a fully-tested pure-function `build_wr_features`.
- Drive-by cleanups: `_PYARROW_STR` consolidated into `schemas.py`, programmatic `_INTEGER_STATS` from `StatLine` annotations, trimmed ingest `__all__`.
- ~50 new tests (5 leakage tests for the WR builder, one per input source).

Spec: `docs/superpowers/specs/2026-04-24-plan-2a-ingest-and-wr-features-design.md`
Plan: `docs/superpowers/plans/2026-04-24-plan-2a-ingest-and-wr-features.md`

## Quality gate
Paste the captured Step 2 output here.

## Test plan
- [ ] `pytest -v` — full suite green (~140 tests)
- [ ] `mypy src tests` — zero violations
- [ ] `ruff check src tests` — zero violations
- [ ] `ruff format --check src tests` — no drift
- [ ] `pytest -v -k "ingest or store or schemas"` — green (CLAUDE.md required for ingest/schema touches)
- [ ] Spot-check leakage tests in `tests/test_features/test_wr_leakage.py` — review the per-source assertion strategy

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

The returned URL is the deliverable for the user to review.

- [ ] **Step 5: Report PR URL + green-gate summary**

In the agent's final reply: PR URL + the captured Step 2 output. No further action; the PR review + merge is the user's call.

---

## Self-Review

Walked through the spec section-by-section to check coverage:

| Spec section | Plan task(s) |
|---|---|
| §2.1 #1 — four new ingest modules | Tasks 9, 10, 11, 12 |
| §2.1 #2 — schema additions to `schemas.py` | Tasks 2, 3, 4, 5, 6, 7 |
| §2.1 #3 — `features/` module + helpers + WR builder | Tasks 13, 14, 15 |
| §2.1 #4 — tests (per-source ingest, leakage, smoke) | Tasks 9-12 (per-source), 15 (non-leakage), 16 (leakage), 17 (smoke) |
| §2.1 #5 — drive-by cleanups | Task 1 |
| §2.1 #6 — documentation updates | Task 18 |
| §3 — ingest source shape | Tasks 9-12 implementations |
| §4.1 — new pandera schemas | Tasks 3, 4, 5, 6, 7 |
| §4.2 — Stat enum additions | Task 2 (TARGETS, CARRIES, RECEIVING_AIR_YARDS); §4.2 also lists OFFENSE_PCT, which the plan currently does NOT add — see note below |
| §4.3 — `WeeklyStatsSchema` extension | Task 2 |
| §4.4 — `_PYARROW_STR` to `schemas.py` | Task 1 |
| §5.1 — `features/` file layout | Task 13 (package + `_rolling.py`), Task 14 (`_opponent.py`), Task 15 (`wr.py` + `__init__.py`) |
| §5.2 — `build_wr_features` public surface | Task 15 |
| §5.3 — leakage prevention contract | Task 15 implementation; tested in Task 16 |
| §5.4 — shared helpers (`_rolling.py`, `_opponent.py`) | Tasks 13, 14 |
| §5.5 — pure-function rationale | Task 15 docstring + commit message |
| §6.1 — per-ingest-source tests + synthetic fixtures | Task 8 (fixtures), Tasks 9-12 (tests) |
| §6.2 — feature-builder tests | Task 15 |
| §6.3 — integration smoke test | Task 17 |
| §6.4 — drive-by cleanup verification | Task 1 quality gate step |
| §6.5 — test budget (~50 new) | Reflected in plan (~7 schema + 5 fake_weekly + 5+5+8+12 ingest + 7 rolling + 2 opponent + 8 wr + 5 leakage + 1 smoke ≈ 65 — slightly over the 50 estimate, acceptable) |
| §6.6 — end-of-effort checklist | Task 19 |
| §7 — open questions deliberately deferred | Surface in TODO #5 (NGS missing data), TODO #6 (opening Vegas lines), TODO #7 (depth-rank parser); §7 notes about NGS API drift surface in TODO #8 |
| §8 — risks | Documented in spec; not an implementation deliverable |
| §9 — MVP delivery list | Maps cleanly onto Tasks 1-19 |
| §10.1, 10.2, 10.3 — documentation updates | Task 18 |

**Gap fixed inline**: §4.2 lists `OFFENSE_PCT` as a Stat enum addition, but Task 2 (the only task that touches the Stat enum) doesn't add it. The Stat enum entry is inert if no code references the value as `Stat.OFFENSE_PCT.value`, but the spec calls it out explicitly. Adding to Task 2 below.

**Placeholder scan:** searched the plan for "TBD", "TODO", "implement later", "fill in details", "Add appropriate", "Similar to Task". Only intentional uses:
- Task 18 Step 3: `commit "<TBD-after-merge>"` for the merge commit hash — this is unavoidable; the implementer fills it in after merging.
- Task 18 Step 4 footer mentions "TBD when 2b is brainstormed" — this is content of the TODO entry being added, not a plan placeholder.
- Task 19 Step 4 PR body says "Paste the captured Step 2 output here" — this is genuinely something the implementer fills in at PR-open time, not a plan ambiguity.

**Type / signature consistency:** spot-checked — `last_n_per_group` keyword arguments match between Task 13 implementation and Task 15 callsite; `opp_allowed_fppg` returns `(season, week, opp_team, opp_allowed_fppg)` columns and Task 15's WR builder joins on `opp_team` matching opponent. `_parse_depth_rank` keyword signature `(*, depth_team, depth_position)` matches Task 11 unit tests.

Adding the OFFENSE_PCT addition to Task 2 inline.
