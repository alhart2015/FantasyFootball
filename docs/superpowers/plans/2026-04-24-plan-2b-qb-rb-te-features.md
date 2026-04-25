# Plan 2b — QB/RB/TE feature builders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land per-position feature builders for QB, RB, and TE following the validated WR pattern from Plan 2a. Three new pure-function builders, three new pandera schemas, generalized share helper, ~45 new tests including 5 leakage tests per position.

**Architecture:** TDD throughout. Each new builder mirrors `wr.py`'s shape (kw-only args, leakage-safe filtering at function entry, schema-validated output) and consumes its position's NGS table (passing/rushing/receiving). Shared logic stays in `_rolling.py` and `_opponent.py`. Parallel files per position — no WR/TE shared base. 16 sequential tasks, each committable independently.

**Tech Stack:** Python 3.11+, `pandas`, `pyarrow`, `pydantic>=2`, `pandera`, `pytest`, `mypy --strict`, `ruff`. Spec at `docs/superpowers/specs/2026-04-24-plan-2b-qb-rb-te-features-design.md`.

**Working directory:** `C:\Users\alden\FantasyFootball\.worktrees\feat-plan-2b-qb-rb-te-features` (branch `feat/plan-2b-qb-rb-te-features`). Activate venv: `. .venv/Scripts/activate`.

---

## File Structure

```
project_management.md                                 # Task 1: fill <TBD-after-merge>; Task 15: docs updates
TODO.md                                               # Task 15: close TODO #2 partial; add TODO #10
src/projections/
├── schemas.py                                        # Tasks 2, 4, 5, 6: extend WeeklyStatsSchema, add 3 feature schemas, add Stat enum entries
├── ingest/
│   └── weekly_stats.py                               # Task 2: extend _KEEP and dtype coercion
└── features/
    ├── __init__.py                                   # Tasks 8, 10, 12: re-export 3 new builders
    ├── _rolling.py                                   # Task 3: add trailing_n_share_in_group
    ├── wr.py                                         # Task 3: migrate to use shared helper
    ├── qb.py                                         # Task 8 (new)
    ├── rb.py                                         # Task 10 (new)
    └── te.py                                         # Task 12 (new)

tests/
├── conftest.py                                       # Task 2: extend fake_weekly_df
├── test_schemas/
│   └── test_dataframe_schemas.py                     # Tasks 2, 4, 5, 6: schema test additions
├── test_ingest/
│   └── test_weekly_stats.py                          # Task 2: persist-new-columns test
├── test_features/
│   ├── conftest.py                                   # Task 7: QB/RB/TE synthetic frames
│   ├── test_rolling.py                               # Task 3: tests for new helper
│   ├── test_wr.py                                    # Task 3: unaffected (regression check only)
│   ├── test_qb.py                                    # Task 8 (new)
│   ├── test_qb_leakage.py                            # Task 9 (new)
│   ├── test_rb.py                                    # Task 10 (new)
│   ├── test_rb_leakage.py                            # Task 11 (new)
│   ├── test_te.py                                    # Task 12 (new)
│   └── test_te_leakage.py                            # Task 13 (new)
└── test_smoke_2a.py → test_smoke.py                  # Task 14: rename + extend
```

No new dependencies. Per-task file count ≤ 5 (per CLAUDE.md PHASED EXECUTION rule).

---

## Sanity-check before starting

```bash
cd "/c/Users/alden/FantasyFootball/.worktrees/feat-plan-2b-qb-rb-te-features"
git branch --show-current     # → feat/plan-2b-qb-rb-te-features
. .venv/Scripts/activate
pytest -v                     # → 158 passing (Plan 2a baseline)
mypy src tests                # → zero violations
ruff check src tests          # → zero violations
```

If the baseline isn't green, fix it before adding new tasks on top.

---

## Phase 1 — Setup

### Task 1: Fill 2a's `<TBD-after-merge>` placeholder in `project_management.md`

Plan 2a left a placeholder for its merge commit hash. Land that on this branch as the first commit (folded into 2b's PR rather than a one-line PR).

**Files:**
- Modify: `project_management.md` (one-line replacement)

- [ ] **Step 1: Read current state**

```bash
grep -n "TBD-after-merge" project_management.md
```

Expected: one match around line 9: `**Projections Core — Plan 2a (Ingest expansion + WR feature builder) merged to \`main\` at commit \`<TBD-after-merge>\`.**`

- [ ] **Step 2: Replace the placeholder**

In `project_management.md`, replace:
```
merged to `main` at commit `<TBD-after-merge>`.
```
with:
```
merged to `main` at commit `7926090`.
```

- [ ] **Step 3: Commit**

```bash
git add project_management.md
git commit -m "docs(pm): record Plan 2a merge commit (7926090)

Filling the post-merge placeholder left by Plan 2a's docs(pm) commit."
```

---

### Task 2: Extend `WeeklyStatsSchema` with `attempts`, `completions`, `sacks`

Same incremental pattern as 2a's extension for `targets`/`carries`/`receiving_air_yards`. Needed by the QB feature builder.

**Files:**
- Modify: `src/projections/schemas.py` (extend `WeeklyStatsSchema`, add 3 `Stat` enum entries)
- Modify: `src/projections/ingest/weekly_stats.py` (extend `_KEEP`, extend int coercion)
- Modify: `tests/conftest.py` (extend `fake_weekly_df` with the 3 new columns)
- Modify: `tests/test_ingest/test_weekly_stats.py` (one new test asserting the new columns persist)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (extend `_good_weekly_stats` helper)

- [ ] **Step 1: Re-read the affected files**

```bash
grep -n "class Stat" -A 25 src/projections/schemas.py
grep -n "class WeeklyStatsSchema" -A 30 src/projections/schemas.py
grep -n "_KEEP" -A 25 src/projections/ingest/weekly_stats.py
grep -n "_good_weekly_stats" -A 30 tests/test_schemas/test_dataframe_schemas.py
grep -n "fake_weekly_df" -A 30 tests/conftest.py
```

- [ ] **Step 2: Add 3 new `Stat` enum entries**

In `src/projections/schemas.py`, the `Stat` enum should look like this after editing (add `PASSING_ATTEMPTS`, `COMPLETIONS`, `SACKS` next to other passing-related entries):

```python
class Stat(StrEnum):
    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_tds"
    PASSING_ATTEMPTS = "attempts"
    COMPLETIONS = "completions"
    SACKS = "sacks"
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
    # Snap-counts column (not weekly_stats) — reserved so feature builders can
    # reference Stat.OFFENSE_PCT.value instead of a string literal.
    OFFENSE_PCT = "offense_pct"
```

Note that `PASSING_ATTEMPTS = "attempts"` (verbose enum name disambiguates from rushing attempts which is `Stat.CARRIES`; nfl_data_py uses bare `attempts`).

- [ ] **Step 3: Extend `WeeklyStatsSchema` with 3 new fields**

In the same file, the `WeeklyStatsSchema` class adds three fields (place them after `interceptions`, alongside other passing-related fields):

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
    attempts: Series[int] = pa.Field(ge=0, le=70)
    completions: Series[int] = pa.Field(ge=0, le=60)
    sacks: Series[int] = pa.Field(ge=0, le=15)
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

In `src/projections/ingest/weekly_stats.py`, extend `_KEEP`:

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
    "attempts",
    "completions",
    "sacks",
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

Extend the int coercion tuple in `_normalize_one_season` to include the three new int columns:

```python
for int_col in (
    "passing_tds",
    "interceptions",
    "attempts",
    "completions",
    "sacks",
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

(Float coercion list is unchanged — the three new fields are integers.)

- [ ] **Step 5: Extend `fake_weekly_df` fixture in `tests/conftest.py`**

In `tests/conftest.py`, update `fake_weekly_df`:

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
            "attempts": [0, 38],
            "completions": [0, 24],
            "sacks": [0, 2],
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

- [ ] **Step 6: Update `_good_weekly_stats` helper in `tests/test_schemas/test_dataframe_schemas.py`**

The helper builds a schema-valid `WeeklyStatsSchema` frame for use by other tests. Add the 3 new fields. Re-read the helper first to see its current shape, then add `attempts`, `completions`, `sacks` keys (with sensible non-zero values for a hypothetical QB row, e.g., `[38]`, `[24]`, `[2]` — or `[0]` if the helper builds a generic row).

- [ ] **Step 7: Write a test asserting the new columns persist through ingest**

Append to `tests/test_ingest/test_weekly_stats.py`:

```python
def test_refresh_weekly_stats_persists_qb_columns(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """attempts, completions, sacks must round-trip through ingest."""
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    assert "attempts" in df.columns
    assert "completions" in df.columns
    assert "sacks" in df.columns
    qb_row = df[df["gsis_id"] == "00-0034857"].iloc[0]
    assert int(qb_row["attempts"]) == 38
    assert int(qb_row["completions"]) == 24
    assert int(qb_row["sacks"]) == 2
    wr_row = df[df["gsis_id"] == "00-0036322"].iloc[0]
    assert int(wr_row["attempts"]) == 0
    assert int(wr_row["completions"]) == 0
    assert int(wr_row["sacks"]) == 0
```

- [ ] **Step 8: Run the full quality gate**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

All must be green. Test count: 158 baseline + 1 new = 159.

- [ ] **Step 9: Commit**

```bash
git add src/projections/schemas.py src/projections/ingest/weekly_stats.py tests/conftest.py tests/test_ingest/test_weekly_stats.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): extend WeeklyStatsSchema with attempts, completions, sacks

QB feature builder needs these source columns. All three are present in
raw nfl_data_py.import_weekly_data output. Adds matching Stat enum
entries (PASSING_ATTEMPTS, COMPLETIONS, SACKS).

fake_weekly_df fixture and _good_weekly_stats schema-test helper both
extended; existing weekly_stats tests pass unchanged."
```

---

## Phase 2 — Helper migration

### Task 3: Migrate `_trailing_4_share_per_team` to `_rolling.py` as `trailing_n_share_in_group`

The current helper lives in `src/projections/features/wr.py` (private). RB and TE both need it. Migrate to `_rolling.py` as a public, generalized helper. Update `wr.py` to consume it. Add tests for the migrated helper.

**Files:**
- Modify: `src/projections/features/_rolling.py` (add `trailing_n_share_in_group`)
- Modify: `src/projections/features/wr.py` (import + use the migrated helper; delete local one)
- Modify: `tests/test_features/test_rolling.py` (add tests for the new helper)

- [ ] **Step 1: Re-read the source helper in wr.py**

```bash
grep -n "_trailing_4_share_per_team" -A 30 src/projections/features/wr.py
```

The function lives around line 70 of `wr.py` and looks like this:

```python
def _trailing_4_share_per_team(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Trailing-4 player share of `value_col` within their team's WR group.
    ..."""
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "share_l4": pd.array([], dtype=float),
            }
        )
    last4_player = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    player_sum = last4_player.groupby(["gsis_id", "team"], as_index=False, observed=True)[
        value_col
    ].sum()
    team_sum = (
        player_sum.groupby("team", as_index=False, observed=True)[value_col]
        .sum()
        .rename(columns={value_col: "team_total"})
    )
    merged = player_sum.merge(team_sum, on="team", how="left")
    merged["share_l4"] = (
        merged[value_col].astype(float) / merged["team_total"].astype(float)
    ).where(merged["team_total"] > 0, 0.0)
    return merged[["gsis_id", "share_l4"]]
```

- [ ] **Step 2: Write failing tests for the new helper**

Append to `tests/test_features/test_rolling.py`:

```python
import pandas as pd

from projections.features._rolling import trailing_n_share_in_group
from projections.schemas import _PYARROW_STR


def _share_input_two_teams() -> pd.DataFrame:
    """Two teams, two players each, trailing-4 sums easy to verify by hand."""
    rows = []
    # Team A: player A1 = 4 + 4 + 4 + 4 = 16; player A2 = 1 + 1 + 1 + 1 = 4. Team total = 20.
    # Team B: player B1 = 5 + 5 + 5 + 5 = 20; player B2 = 5 + 5 + 5 + 5 = 20. Team total = 40.
    for week in range(1, 5):
        rows.extend(
            [
                {"gsis_id": "00-000A001", "season": 2024, "week": week, "team": "A", "value": 4},
                {"gsis_id": "00-000A002", "season": 2024, "week": week, "team": "A", "value": 1},
                {"gsis_id": "00-000B001", "season": 2024, "week": week, "team": "B", "value": 5},
                {"gsis_id": "00-000B002", "season": 2024, "week": week, "team": "B", "value": 5},
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    return df


def test_trailing_n_share_in_group_basic_shares() -> None:
    df = _share_input_two_teams()
    out = trailing_n_share_in_group(df, value_col="value", n=4)
    assert set(out.columns) == {"gsis_id", "share_l4"}
    by_id = out.set_index("gsis_id")["share_l4"]
    assert by_id["00-000A001"] == 16 / 20  # 0.8
    assert by_id["00-000A002"] == 4 / 20   # 0.2
    assert by_id["00-000B001"] == 20 / 40  # 0.5
    assert by_id["00-000B002"] == 20 / 40  # 0.5


def test_trailing_n_share_in_group_zero_team_total_yields_zero() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-000A001"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [1],
            "team": pd.array(["A"], dtype=_PYARROW_STR),
            "value": [0],
        }
    )
    out = trailing_n_share_in_group(df, value_col="value", n=4)
    assert out.loc[out["gsis_id"] == "00-000A001", "share_l4"].iloc[0] == 0.0


def test_trailing_n_share_in_group_empty_input() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array([], dtype=_PYARROW_STR),
            "season": pd.array([], dtype=int),
            "week": pd.array([], dtype=int),
            "team": pd.array([], dtype=_PYARROW_STR),
            "value": pd.array([], dtype=float),
        }
    )
    out = trailing_n_share_in_group(df, value_col="value", n=4)
    assert len(out) == 0
    assert set(out.columns) == {"gsis_id", "share_l4"}


def test_trailing_n_share_in_group_default_n_is_4() -> None:
    """Calling without n= uses n=4 (the established convention)."""
    df = _share_input_two_teams()
    explicit = trailing_n_share_in_group(df, value_col="value", n=4)
    default = trailing_n_share_in_group(df, value_col="value")
    pd.testing.assert_frame_equal(explicit, default)
```

- [ ] **Step 3: Run, verify ImportError**

```bash
pytest -v tests/test_features/test_rolling.py -k "trailing_n_share_in_group"
```

Expected: 4 FAIL with `ImportError: cannot import name 'trailing_n_share_in_group' from 'projections.features._rolling'`.

- [ ] **Step 4: Implement `trailing_n_share_in_group` in `_rolling.py`**

Append to `src/projections/features/_rolling.py`:

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
    out_col = f"share_l{n}"
    if weekly_stats.empty:
        # Use schema-friendly empty frame so callers can merge without dtype churn.
        from projections.schemas import _PYARROW_STR  # local import to avoid cycle if any

        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                out_col: pd.array([], dtype=float),
            }
        )
    last_n_player = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=n,
    )
    player_sum = last_n_player.groupby(["gsis_id", "team"], as_index=False, observed=True)[
        value_col
    ].sum()
    team_sum = (
        player_sum.groupby("team", as_index=False, observed=True)[value_col]
        .sum()
        .rename(columns={value_col: "team_total"})
    )
    merged = player_sum.merge(team_sum, on="team", how="left")
    merged[out_col] = (
        merged[value_col].astype(float) / merged["team_total"].astype(float)
    ).where(merged["team_total"] > 0, 0.0)
    return merged[["gsis_id", out_col]]
```

(The local `_PYARROW_STR` import is fine — `_rolling.py` doesn't currently import from `schemas.py`, and there's no circular risk because `schemas.py` doesn't import from `_rolling.py`.)

- [ ] **Step 5: Run helper tests, verify pass**

```bash
pytest -v tests/test_features/test_rolling.py -k "trailing_n_share_in_group"
```

Expected: 4 PASS.

- [ ] **Step 6: Migrate `wr.py` to use the new helper**

In `src/projections/features/wr.py`:

a) Replace the import at the top:
```python
from projections.features._rolling import last_n_per_group
```
with:
```python
from projections.features._rolling import last_n_per_group, trailing_n_share_in_group
```

b) Delete the entire `_trailing_4_share_per_team(...)` function definition (lines ~70-102 of `wr.py`).

c) Replace the two call sites in `build_wr_features`:
```python
target_share = _trailing_4_share_per_team(ws_wr, Stat.TARGETS.value).rename(...)
air_yards_share = _trailing_4_share_per_team(ws_wr, Stat.RECEIVING_AIR_YARDS.value).rename(...)
```
with:
```python
target_share = trailing_n_share_in_group(ws_wr, value_col=Stat.TARGETS.value).rename(
    columns={"share_l4": "target_share_l4"}
)
air_yards_share = trailing_n_share_in_group(
    ws_wr, value_col=Stat.RECEIVING_AIR_YARDS.value
).rename(columns={"share_l4": "air_yards_share_l4"})
```

- [ ] **Step 7: Run all WR tests to confirm zero regression**

```bash
pytest -v tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py
```

Expected: 13 PASS (8 wr + 5 leakage), all unchanged.

- [ ] **Step 8: Run full quality gate**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

Test count: 159 from Task 2 + 4 from helper tests = 163.

- [ ] **Step 9: Commit**

```bash
git add src/projections/features/_rolling.py src/projections/features/wr.py tests/test_features/test_rolling.py
git commit -m "refactor(features): migrate _trailing_4_share_per_team to _rolling.py

New shared helper trailing_n_share_in_group lives in _rolling.py; wr.py
becomes a one-line consumer. Plan 2b's QB/RB/TE builders will reuse it.
Helper signature is generalized: caller controls the share-group via
pre-filtering, supports any n (default 4)."
```

---

## Phase 3 — Feature schemas

### Task 4: Add `QbFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add `QbFeaturesSchema` after `WrFeaturesSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (validate-passes + reject-bad-value tests)

- [ ] **Step 1: Write failing schema tests**

Append to `tests/test_schemas/test_dataframe_schemas.py` (add `QbFeaturesSchema` to existing top-of-file `from projections.schemas import ...`):

```python
def test_qb_features_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034857"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["KC"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "pass_attempts_per_game_l4": [38.5],
            "passing_yards_per_game_l4": [285.0],
            "passing_tds_per_game_l4": [2.25],
            "interceptions_per_game_l4": [0.5],
            "sacks_per_game_l4": [2.0],
            "passing_yards_per_game_std": [275.0],
            "rushing_attempts_per_game_l4": [4.5],
            "rushing_yards_per_game_l4": [22.0],
            "rushing_qb": [False],
            "snap_pct_l4": [1.0],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "aggressiveness_std": [12.5],
            "completion_percentage_above_expectation_std": [3.3],
            "avg_intended_air_yards_std": [8.1],
            "avg_time_to_throw_std": [2.71],
            "implied_team_total": [27.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_qb_fppg_l4": [18.5],
        }
    )
    QbFeaturesSchema.validate(df)


def test_qb_features_schema_rejects_negative_pass_attempts() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034857"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["KC"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "pass_attempts_per_game_l4": [-1.0],  # invalid
            "passing_yards_per_game_l4": [285.0],
            "passing_tds_per_game_l4": [2.25],
            "interceptions_per_game_l4": [0.5],
            "sacks_per_game_l4": [2.0],
            "passing_yards_per_game_std": [275.0],
            "rushing_attempts_per_game_l4": [4.5],
            "rushing_yards_per_game_l4": [22.0],
            "rushing_qb": [False],
            "snap_pct_l4": [1.0],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "aggressiveness_std": [12.5],
            "completion_percentage_above_expectation_std": [3.3],
            "avg_intended_air_yards_std": [8.1],
            "avg_time_to_throw_std": [2.71],
            "implied_team_total": [27.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_qb_fppg_l4": [18.5],
        }
    )
    with pytest.raises(SchemaError):
        QbFeaturesSchema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "qb_features"
```

Expected: 2 FAIL with ImportError.

- [ ] **Step 3: Add `QbFeaturesSchema` to `schemas.py`**

After `WrFeaturesSchema`:

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

- [ ] **Step 4: Run, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "qb_features"
```

Expected: 2 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add QbFeaturesSchema for QB feature builder output"
```

---

### Task 5: Add `RbFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add `RbFeaturesSchema` after `QbFeaturesSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (validate-passes + reject-bad-value tests)

- [ ] **Step 1: Write failing schema tests**

Append to `tests/test_schemas/test_dataframe_schemas.py` (add `RbFeaturesSchema` to existing imports):

```python
def test_rb_features_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034796"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["PHI"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DAL"], dtype=_PYARROW_STR),
            "carries_per_game_l4": [18.5],
            "rushing_yards_per_game_l4": [98.0],
            "rushing_tds_per_game_l4": [0.75],
            "rush_share_l4": [0.72],
            "targets_per_game_l4": [4.5],
            "receptions_per_game_l4": [3.5],
            "receiving_yards_per_game_l4": [28.0],
            "target_share_l4": [0.12],
            "targets_per_game_std": [4.0],
            "snap_pct_l4": [0.85],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "passing_down_back": [True],
            "efficiency_std": [3.1],
            "rush_yards_over_expected_per_att_std": [0.9],
            "percent_attempts_gte_eight_defenders_std": [22.5],
            "implied_team_total": [25.0],
            "spread": [-2.5],
            "is_home": [False],
            "roof_dome": [False],
            "opp_allowed_rb_fppg_l4": [20.5],
        }
    )
    RbFeaturesSchema.validate(df)


def test_rb_features_schema_rejects_rush_share_over_one() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034796"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["PHI"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DAL"], dtype=_PYARROW_STR),
            "carries_per_game_l4": [18.5],
            "rushing_yards_per_game_l4": [98.0],
            "rushing_tds_per_game_l4": [0.75],
            "rush_share_l4": [1.5],  # invalid, must be <= 1
            "targets_per_game_l4": [4.5],
            "receptions_per_game_l4": [3.5],
            "receiving_yards_per_game_l4": [28.0],
            "target_share_l4": [0.12],
            "targets_per_game_std": [4.0],
            "snap_pct_l4": [0.85],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "passing_down_back": [True],
            "efficiency_std": [3.1],
            "rush_yards_over_expected_per_att_std": [0.9],
            "percent_attempts_gte_eight_defenders_std": [22.5],
            "implied_team_total": [25.0],
            "spread": [-2.5],
            "is_home": [False],
            "roof_dome": [False],
            "opp_allowed_rb_fppg_l4": [20.5],
        }
    )
    with pytest.raises(SchemaError):
        RbFeaturesSchema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "rb_features"
```

Expected: 2 FAIL.

- [ ] **Step 3: Add `RbFeaturesSchema` to `schemas.py`**

After `QbFeaturesSchema`:

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

- [ ] **Step 4: Run, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "rb_features"
```

Expected: 2 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add RbFeaturesSchema for RB feature builder output"
```

---

### Task 6: Add `TeFeaturesSchema`

**Files:**
- Modify: `src/projections/schemas.py` (add `TeFeaturesSchema` after `RbFeaturesSchema`)
- Modify: `tests/test_schemas/test_dataframe_schemas.py` (validate-passes + reject-bad-value tests)

- [ ] **Step 1: Write failing schema tests**

Append to `tests/test_schemas/test_dataframe_schemas.py` (add `TeFeaturesSchema` to existing imports):

```python
def test_te_features_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036440"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["SF"], dtype=_PYARROW_STR),
            "opponent": pd.array(["SEA"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [7.5],
            "targets_per_game_std": [7.0],
            "target_share_l4": [0.18],
            "receptions_per_game_l4": [5.5],
            "receiving_yards_per_game_l4": [62.0],
            "receiving_tds_per_game_l4": [0.5],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [2.8],
            "avg_intended_air_yards_std": [9.0],
            "avg_yac_above_expectation_std": [0.4],
            "implied_team_total": [24.0],
            "spread": [-1.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_te_fppg_l4": [10.5],
        }
    )
    TeFeaturesSchema.validate(df)


def test_te_features_schema_rejects_target_share_over_one() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036440"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["SF"], dtype=_PYARROW_STR),
            "opponent": pd.array(["SEA"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [7.5],
            "targets_per_game_std": [7.0],
            "target_share_l4": [1.2],  # invalid
            "receptions_per_game_l4": [5.5],
            "receiving_yards_per_game_l4": [62.0],
            "receiving_tds_per_game_l4": [0.5],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [2.8],
            "avg_intended_air_yards_std": [9.0],
            "avg_yac_above_expectation_std": [0.4],
            "implied_team_total": [24.0],
            "spread": [-1.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_te_fppg_l4": [10.5],
        }
    )
    with pytest.raises(SchemaError):
        TeFeaturesSchema.validate(df)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "te_features"
```

Expected: 2 FAIL.

- [ ] **Step 3: Add `TeFeaturesSchema` to `schemas.py`**

After `RbFeaturesSchema`:

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

- [ ] **Step 4: Run, verify pass**

```bash
pytest -v tests/test_schemas/test_dataframe_schemas.py -k "te_features"
```

Expected: 2 PASS.

- [ ] **Step 5: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add TeFeaturesSchema for TE feature builder output"
```

---

## Phase 4 — Synthetic fixtures

### Task 7: Add QB/RB/TE synthetic frames to `tests/test_features/conftest.py`

Each builder needs a small set of synthetic frames mirroring the WR fixtures from 2a (`wr_weekly_stats`, `wr_snap_counts`, `wr_depth_charts`, `wr_ngs_receiving`, `wr_schedules`). Designed for round-number rolling expectations.

**Files:**
- Modify: `tests/test_features/conftest.py` (append fixtures for each position)

- [ ] **Step 1: Re-read current conftest.py to confirm style**

```bash
grep -n "wr_weekly_stats" -A 5 tests/test_features/conftest.py
```

The existing fixtures use `_PYARROW_STR` for string columns and have inline docstrings explaining the per-week data shape. Match that style.

- [ ] **Step 2: Append `qb_weekly_stats`**

```python
@pytest.fixture
def qb_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 2 QBs across 2 teams (KC, MIN).

    Designed so trailing-4 windows have round-number expectations:
    - Patrick Mahomes (KC, gsis_id=00-0034857): 36/38/40/42 attempts weeks 1-4,
      36/38/40/42 weeks 5-8. Trailing-4 mean attempts = 39.0 either way.
    - Kirk Cousins (MIN, gsis_id=00-0033106): 30/30/30/30 attempts uniformly.

    Both play opponent rotation: weeks 1-4 vs DEN, weeks 5-8 vs CHI.
    No rushing usage (pure pocket QBs in this fixture).
    """
    rows = []
    for week in range(1, 9):
        opp = "DEN" if week <= 4 else "CHI"

        # Mahomes (KC)
        mahomes_attempts = [36, 38, 40, 42, 36, 38, 40, 42][week - 1]
        rows.append(
            {
                "gsis_id": "00-0034857",
                "season": 2024,
                "week": week,
                "position": "QB",
                "team": "KC",
                "opponent": opp,
                "passing_yards": float(mahomes_attempts * 7.5),
                "passing_tds": 2,
                "interceptions": 1,
                "attempts": mahomes_attempts,
                "completions": int(mahomes_attempts * 0.65),
                "sacks": 2,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
                "fumbles_lost": 0,
            }
        )

        # Cousins (MIN)
        rows.append(
            {
                "gsis_id": "00-0033106",
                "season": 2024,
                "week": week,
                "position": "QB",
                "team": "MIN",
                "opponent": opp,
                "passing_yards": 250.0,
                "passing_tds": 1,
                "interceptions": 0,
                "attempts": 30,
                "completions": 20,
                "sacks": 1,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": 0,
                "receiving_yards": 0.0,
                "receiving_tds": 0,
                "receiving_air_yards": 0.0,
                "targets": 0,
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
def qb_snap_counts(qb_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the same QBs/weeks. 100% snap pct (full-time starters)."""
    rows = []
    for _, r in qb_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 65,
                "offense_pct": 1.0,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 0,
                "st_pct": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def qb_depth_charts() -> pd.DataFrame:
    """Depth chart snapshot for week 5 of 2024. Both QBs as starters."""
    rows = [
        {
            "gsis_id": "00-0034857",
            "season": 2024,
            "week": 5,
            "team": "KC",
            "position": "QB",
            "depth_team": "QB1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0033106",
            "season": 2024,
            "week": 5,
            "team": "MIN",
            "position": "QB",
            "depth_team": "QB1",
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
def qb_ngs_passing() -> pd.DataFrame:
    """NGS passing snapshots through week 4 of 2024 for the 2 QBs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0034857",
                    "season": 2024,
                    "week": week,
                    "team": "KC",
                    "position": "QB",
                    "avg_time_to_throw": 2.71,
                    "avg_completed_air_yards": 6.2,
                    "avg_intended_air_yards": 8.1,
                    "avg_air_yards_differential": -1.9,
                    "aggressiveness": 12.5,
                    "max_completed_air_distance": 42.0,
                    "avg_air_yards_to_sticks": -0.4,
                    "completion_percentage": 68.5,
                    "expected_completion_percentage": 65.2,
                    "completion_percentage_above_expectation": 3.3,
                    "avg_air_distance": 9.5,
                    "max_air_distance": 55.0,
                },
                {
                    "gsis_id": "00-0033106",
                    "season": 2024,
                    "week": week,
                    "team": "MIN",
                    "position": "QB",
                    "avg_time_to_throw": 2.45,
                    "avg_completed_air_yards": 5.8,
                    "avg_intended_air_yards": 7.2,
                    "avg_air_yards_differential": -1.4,
                    "aggressiveness": 10.0,
                    "max_completed_air_distance": 38.0,
                    "avg_air_yards_to_sticks": -0.7,
                    "completion_percentage": 66.7,
                    "expected_completion_percentage": 66.0,
                    "completion_percentage_above_expectation": 0.7,
                    "avg_air_distance": 8.5,
                    "max_air_distance": 48.0,
                },
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def qb_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: KC @ CHI, MIN @ CHI (made up to share opponent)."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_KC_CHI", "2024_05_MIN_CHI"], dtype=_PYARROW_STR),
            "home_team": pd.array(["CHI", "CHI"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC", "MIN"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [-7.5, -3.5],  # KC favored by 7.5; MIN favored by 3.5 (away favored = negative)
            "total_line": [51.0, 48.5],
            "home_moneyline": pd.array([280, 155], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-340, -180], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass", "grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors", "outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([55, 55], dtype=pd.Int64Dtype()),
            "wind": pd.array([8, 8], dtype=pd.Int64Dtype()),
        }
    )
```

- [ ] **Step 3: Append `rb_weekly_stats`, `rb_snap_counts`, `rb_depth_charts`, `rb_ngs_rushing`, `rb_schedules`**

Mirror the QB pattern. Use 2 RBs across 2 teams (PHI, SF). Make one a workhorse (Saquon Barkley, gsis_id=00-0034796, 18-22 carries/game) and one a passing-down back (gsis_id=00-0036650, 8 carries + 5 targets/game). Both vs same opponents (DAL → SEA). NGS rushing data mirrors the WR's NGS receiving structure but with rushing-specific columns.

```python
@pytest.fixture
def rb_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 2 RBs across 2 teams (PHI, SF).

    - Saquon Barkley (PHI, 00-0034796): 20 carries/game uniformly, 2 targets.
    - Christian McCaffrey (SF, 00-0036650): 14 carries/game, 6 targets/game (passing-down back).
    """
    rows = []
    for week in range(1, 9):
        opp = "DAL" if week <= 4 else "SEA"
        # Saquon — workhorse runner
        rows.append(
            {
                "gsis_id": "00-0034796",
                "season": 2024,
                "week": week,
                "position": "RB",
                "team": "PHI",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "attempts": 0,
                "completions": 0,
                "sacks": 0,
                "rushing_yards": 90.0,
                "rushing_tds": 1,
                "carries": 20,
                "receptions": 1,
                "receiving_yards": 8.0,
                "receiving_tds": 0,
                "receiving_air_yards": 5.0,
                "targets": 2,
                "fumbles_lost": 0,
            }
        )
        # CMC — pass-catching back
        rows.append(
            {
                "gsis_id": "00-0036650",
                "season": 2024,
                "week": week,
                "position": "RB",
                "team": "SF",
                "opponent": opp,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "attempts": 0,
                "completions": 0,
                "sacks": 0,
                "rushing_yards": 65.0,
                "rushing_tds": 0,
                "carries": 14,
                "receptions": 5,
                "receiving_yards": 42.0,
                "receiving_tds": 0,
                "receiving_air_yards": 28.0,
                "targets": 6,
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
def rb_snap_counts(rb_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the RBs. ~85% pct (workhorse/feature backs)."""
    rows = []
    for _, r in rb_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 55,
                "offense_pct": 0.85,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 5,
                "st_pct": 0.15,
            }
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def rb_depth_charts() -> pd.DataFrame:
    """Depth chart for week 5 of 2024. Both RBs as RB1."""
    rows = [
        {
            "gsis_id": "00-0034796",
            "season": 2024,
            "week": 5,
            "team": "PHI",
            "position": "RB",
            "depth_team": "RB1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0036650",
            "season": 2024,
            "week": 5,
            "team": "SF",
            "position": "RB",
            "depth_team": "RB1",
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
def rb_ngs_rushing() -> pd.DataFrame:
    """NGS rushing snapshots through week 4 of 2024 for the 2 RBs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0034796",
                    "season": 2024,
                    "week": week,
                    "team": "PHI",
                    "position": "RB",
                    "efficiency": 3.1,
                    "percent_attempts_gte_eight_defenders": 25.0,
                    "avg_time_to_los": 2.95,
                    "rush_attempts": 20,
                    "rush_yards": 90,
                    "expected_rush_yards": 80.0,
                    "rush_yards_over_expected": 10.0,
                    "avg_rush_yards": 4.5,
                    "rush_yards_over_expected_per_att": 0.5,
                    "rush_pct_over_expected": 12.5,
                },
                {
                    "gsis_id": "00-0036650",
                    "season": 2024,
                    "week": week,
                    "team": "SF",
                    "position": "RB",
                    "efficiency": 3.4,
                    "percent_attempts_gte_eight_defenders": 18.0,
                    "avg_time_to_los": 2.80,
                    "rush_attempts": 14,
                    "rush_yards": 65,
                    "expected_rush_yards": 60.0,
                    "rush_yards_over_expected": 5.0,
                    "avg_rush_yards": 4.6,
                    "rush_yards_over_expected_per_att": 0.4,
                    "rush_pct_over_expected": 8.0,
                },
            ]
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def rb_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: PHI @ SEA, SF @ SEA."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_PHI_SEA", "2024_05_SF_SEA"], dtype=_PYARROW_STR),
            "home_team": pd.array(["SEA", "SEA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["PHI", "SF"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [-2.5, -3.5],  # both away teams favored
            "total_line": [45.0, 48.5],
            "home_moneyline": pd.array([135, 155], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-160, -180], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf", "fieldturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors", "outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([62, 62], dtype=pd.Int64Dtype()),
            "wind": pd.array([6, 6], dtype=pd.Int64Dtype()),
        }
    )
```

- [ ] **Step 4: Append `te_weekly_stats`, `te_snap_counts`, `te_depth_charts`, `te_ngs_receiving`, `te_schedules`**

Mirror the same pattern. Use 2 TEs (Travis Kelce KC 00-0030506, George Kittle SF 00-0033084). For target_share testing, the TE fixture also needs WR data on the same teams so the denominator (full pass-catching group) is meaningful — but for v1 simplicity we can have the TE be the only pass-catcher in the fixture and target_share = 1.0; or we can include a small WR row to give a non-trivial share. Let's make share = 0.5 by including one WR per team with same target volume.

```python
@pytest.fixture
def te_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 stats for 2 TEs + 1 supporting WR per team.

    The supporting WR (matching target volume) makes target_share within
    the pass-catching group = 0.5 for each TE — easy to verify by hand.

    - Travis Kelce (KC, 00-0030506): 8 targets/game uniformly.
    - George Kittle (SF, 00-0033084): 6 targets/game uniformly.
    - Rashee Rice (KC WR, 00-0034950): 8 targets/game (matches Kelce).
    - Brandon Aiyuk (SF WR, 00-0035716): 6 targets/game (matches Kittle).
    """
    rows = []
    for week in range(1, 9):
        opp = "DEN" if week <= 4 else "ARI"
        # Kelce (TE, KC)
        rows.append(
            _make_receiver_row("00-0030506", "TE", "KC", opp, week, targets=8, recs=6, yds=70, tds=1)
        )
        # Kittle (TE, SF)
        rows.append(
            _make_receiver_row("00-0033084", "TE", "SF", opp, week, targets=6, recs=4, yds=55, tds=0)
        )
        # Rice (WR, KC)
        rows.append(
            _make_receiver_row("00-0034950", "WR", "KC", opp, week, targets=8, recs=6, yds=88, tds=0)
        )
        # Aiyuk (WR, SF)
        rows.append(
            _make_receiver_row("00-0035716", "WR", "SF", opp, week, targets=6, recs=4, yds=58, tds=0)
        )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


def _make_receiver_row(
    gsis_id: str,
    position: str,
    team: str,
    opp: str,
    week: int,
    *,
    targets: int,
    recs: int,
    yds: float,
    tds: int,
) -> dict[str, object]:
    """Helper to build a synthetic receiver-shaped weekly_stats row."""
    return {
        "gsis_id": gsis_id,
        "season": 2024,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opp,
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": 0.0,
        "rushing_tds": 0,
        "carries": 0,
        "receptions": recs,
        "receiving_yards": float(yds),
        "receiving_tds": tds,
        "receiving_air_yards": float(yds * 1.2),
        "targets": targets,
        "fumbles_lost": 0,
    }


@pytest.fixture
def te_snap_counts(te_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the TE-test cohort. ~92% pct uniformly."""
    rows = []
    for _, r in te_weekly_stats.iterrows():
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 60,
                "offense_pct": 0.92,
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
def te_depth_charts() -> pd.DataFrame:
    """Depth chart for week 5 of 2024. Both TEs as TE1."""
    rows = [
        {
            "gsis_id": "00-0030506",
            "season": 2024,
            "week": 5,
            "team": "KC",
            "position": "TE",
            "depth_team": "TE1",
            "depth_rank": 1,
        },
        {
            "gsis_id": "00-0033084",
            "season": 2024,
            "week": 5,
            "team": "SF",
            "position": "TE",
            "depth_team": "TE1",
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
def te_ngs_receiving() -> pd.DataFrame:
    """NGS receiving snapshots through week 4 of 2024 for the 2 TEs."""
    rows = []
    for week in range(1, 5):
        rows.extend(
            [
                {
                    "gsis_id": "00-0030506",
                    "season": 2024,
                    "week": week,
                    "team": "KC",
                    "position": "TE",
                    "avg_cushion": 4.5,
                    "avg_separation": 2.8,
                    "avg_intended_air_yards": 9.0,
                    "percent_share_of_intended_air_yards": 22.0,
                    "receptions": 6,
                    "targets": 8,
                    "catch_percentage": 75.0,
                    "yards": 70,
                    "rec_touchdowns": 1,
                    "avg_yac": 4.0,
                    "avg_expected_yac": 3.5,
                    "avg_yac_above_expectation": 0.5,
                },
                {
                    "gsis_id": "00-0033084",
                    "season": 2024,
                    "week": week,
                    "team": "SF",
                    "position": "TE",
                    "avg_cushion": 4.0,
                    "avg_separation": 3.2,
                    "avg_intended_air_yards": 8.0,
                    "percent_share_of_intended_air_yards": 18.0,
                    "receptions": 4,
                    "targets": 6,
                    "catch_percentage": 66.7,
                    "yards": 55,
                    "rec_touchdowns": 0,
                    "avg_yac": 5.0,
                    "avg_expected_yac": 4.5,
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
def te_schedules() -> pd.DataFrame:
    """Schedule for week 5 of 2024: KC @ ARI, SF @ ARI."""
    return pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": pd.array(["2024_05_KC_ARI", "2024_05_SF_ARI"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ARI", "ARI"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC", "SF"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-10-06T17:00:00Z", "2024-10-06T20:25:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [-7.5, -1.5],
            "total_line": [49.0, 47.0],
            "home_moneyline": pd.array([280, 105], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-340, -125], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass", "grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["closed", "closed"], dtype=_PYARROW_STR),  # ARI has retractable
            "temp": pd.array([72, 72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0, 0], dtype=pd.Int64Dtype()),
        }
    )
```

- [ ] **Step 5: Sanity-check fixture collection**

```bash
pytest -v --collect-only tests/test_features
```

Expected: collection succeeds; new fixtures are discoverable. (No tests consume them yet; Tasks 8-13 will.)

- [ ] **Step 6: Quality gate + commit**

```bash
mypy tests/test_features
ruff check tests/test_features
ruff format --check tests/test_features
git add tests/test_features/conftest.py
git commit -m "test(features): add QB/RB/TE synthetic fixtures

5 new fixtures per position (weekly_stats, snap_counts, depth_charts,
ngs_passing/rushing/receiving, schedules) — same shape as 2a's WR
fixtures. Designed for round-number rolling expectations. TE fixture
includes supporting WR rows so target_share has a meaningful denominator."
```

---

## Phase 5 — Per-position builders & tests

### Task 8: `src/projections/features/qb.py` — `build_qb_features` + non-leakage tests

**Files:**
- Create: `src/projections/features/qb.py`
- Modify: `src/projections/features/__init__.py` (re-export `build_qb_features`)
- Create: `tests/test_features/test_qb.py`

- [ ] **Step 1: Write failing tests in `tests/test_features/test_qb.py`**

```python
"""QB feature builder tests (non-leakage). Leakage tests live in test_qb_leakage.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_qb_features
from projections.schemas import QbFeaturesSchema


def test_build_qb_features_returns_validated_frame(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=2024,
        as_of_week=5,
    )
    QbFeaturesSchema.validate(out)


def test_build_qb_features_one_row_per_rostered_qb(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=2024,
        as_of_week=5,
    )
    # 2 QBs on rosters in week 5 → 2 rows.
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0034857", "00-0033106"}


def test_build_qb_features_pass_attempts_per_game_l4_correct(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    """Mahomes weeks 1-4: 36/38/40/42 → mean = 39.0."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    assert mahomes["pass_attempts_per_game_l4"] == 39.0


def test_build_qb_features_rushing_qb_false_for_pocket_qbs(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    """Both fixture QBs have 0 carries → rushing_qb == False."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=2024,
        as_of_week=5,
    )
    assert not out["rushing_qb"].any()


def test_build_qb_features_rushing_qb_true_above_threshold(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    """Inject 6 carries/game over weeks 1-4 for Mahomes (24 trailing-4 / 4 = 6.0 ≥ 5.0)."""
    ws = qb_weekly_stats.copy()
    mask = (ws["gsis_id"] == "00-0034857") & (ws["week"] <= 4)
    ws.loc[mask, "carries"] = 6
    ws.loc[mask, "rushing_yards"] = 30.0
    out = build_qb_features(
        weekly_stats=ws,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    assert bool(mahomes["rushing_qb"]) is True


def test_build_qb_features_implied_team_total_from_schedules(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    """KC @ CHI, total=51, spread_line=-7.5 (KC away favored).
    KC implied = (51 - (-7.5))/2 = 29.25; KC spread = +(-7.5) = -7.5."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    assert mahomes["implied_team_total"] == pytest.approx(29.25, abs=1e-6)
    assert mahomes["spread"] == pytest.approx(-7.5, abs=1e-6)
    assert bool(mahomes["is_home"]) is False


def test_build_qb_features_ngs_aggressiveness_propagates(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    """The latest NGS snapshot's `aggressiveness` is propagated as `aggressiveness_std`."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    # Fixture sets aggressiveness=12.5 for Mahomes through week 4.
    assert mahomes["aggressiveness_std"] == pytest.approx(12.5, abs=1e-6)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_features/test_qb.py
```

Expected: 7 FAIL with `ImportError: cannot import name 'build_qb_features' from 'projections.features'`.

- [ ] **Step 3: Implement `src/projections/features/qb.py`**

```python
"""QB feature builder. Pure function — no I/O, no caching.

Output is one row per (gsis_id, season, week=as_of_week) for every QB on
a roster in week as_of_week of season. Validates against QbFeaturesSchema."""

from __future__ import annotations

from typing import Final

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import last_n_per_group
from projections.schemas import (
    _PYARROW_STR,
    Position,
    QbFeaturesSchema,
    Ruleset,
    Stat,
)
from projections.features.wr import _build_game_environment, _exact_week_mask, _prior_mask

_RUSHING_QB_THRESHOLD: Final = 5.0  # carries/game over trailing 4

_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "pass_attempts_per_game_l4",
    "passing_yards_per_game_l4",
    "passing_tds_per_game_l4",
    "interceptions_per_game_l4",
    "sacks_per_game_l4",
    "passing_yards_per_game_std",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
)


def _trailing_4_per_player(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-player mean of `value_col` over the trailing 4 games."""
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "mean_l4": pd.array([], dtype=float),
            }
        )
    last4 = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    return (
        last4.groupby("gsis_id", as_index=False, observed=True)[value_col]
        .mean()
        .rename(columns={value_col: "mean_l4"})
    )


def _latest_ngs_snapshot(ngs: pd.DataFrame) -> pd.DataFrame:
    """Per-player most-recent NGS row (assumes ngs already prior-filtered)."""
    if ngs.empty:
        return pd.DataFrame()
    return (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False, observed=True)
        .tail(1)
        .copy()
    )


def build_qb_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_passing: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the QB feature DataFrame for week `as_of_week` of `season`."""
    # --- Leakage-safe input filtering -------------------------------------
    ws = weekly_stats[_prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[_prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_passing[_prior_mask(ngs_passing, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[_exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[_exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    # --- Rostered QBs in target week (depth chart drives roster set) ------
    qb_dc = dc[dc["position"] == Position.QB.value].copy()
    if qb_dc.empty:
        empty_cols = list(QbFeaturesSchema.to_schema().columns.keys())
        return QbFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    ws_qb = ws[ws["position"] == Position.QB.value].copy()
    sc_qb = sc[sc["position"] == Position.QB.value].copy()

    # --- Per-player rolling features --------------------------------------
    pass_att_l4 = _trailing_4_per_player(ws_qb, Stat.PASSING_ATTEMPTS.value).rename(
        columns={"mean_l4": "pass_attempts_per_game_l4"}
    )
    pass_yd_l4 = _trailing_4_per_player(ws_qb, Stat.PASSING_YARDS.value).rename(
        columns={"mean_l4": "passing_yards_per_game_l4"}
    )
    pass_td_l4 = _trailing_4_per_player(ws_qb, Stat.PASSING_TDS.value).rename(
        columns={"mean_l4": "passing_tds_per_game_l4"}
    )
    int_l4 = _trailing_4_per_player(ws_qb, Stat.INTERCEPTIONS.value).rename(
        columns={"mean_l4": "interceptions_per_game_l4"}
    )
    sacks_l4 = _trailing_4_per_player(ws_qb, Stat.SACKS.value).rename(
        columns={"mean_l4": "sacks_per_game_l4"}
    )
    rush_att_l4 = _trailing_4_per_player(ws_qb, Stat.CARRIES.value).rename(
        columns={"mean_l4": "rushing_attempts_per_game_l4"}
    )
    rush_yd_l4 = _trailing_4_per_player(ws_qb, Stat.RUSHING_YARDS.value).rename(
        columns={"mean_l4": "rushing_yards_per_game_l4"}
    )

    ws_this_season = ws_qb[ws_qb["season"] == season]
    if ws_this_season.empty:
        pass_yd_std = pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "passing_yards_per_game_std": pd.array([], dtype=float),
            }
        )
    else:
        pass_yd_std = (
            ws_this_season.groupby("gsis_id", as_index=False, observed=True)[
                Stat.PASSING_YARDS.value
            ]
            .mean()
            .rename(columns={Stat.PASSING_YARDS.value: "passing_yards_per_game_std"})
        )

    snap_l4 = _trailing_4_per_player(sc_qb, Stat.OFFENSE_PCT.value).rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # --- NGS latest snapshot per player → *_std columns -------------------
    ngs_latest = _latest_ngs_snapshot(ngs)
    ngs_std_cols = (
        "aggressiveness_std",
        "completion_percentage_above_expectation_std",
        "avg_intended_air_yards_std",
        "avg_time_to_throw_std",
    )
    if ngs_latest.empty:
        ngs_cols = pd.DataFrame(
            {"gsis_id": pd.array([], dtype=_PYARROW_STR)}
            | {c: pd.array([], dtype=float) for c in ngs_std_cols}
        )
    else:
        ngs_cols = ngs_latest[
            [
                "gsis_id",
                "aggressiveness",
                "completion_percentage_above_expectation",
                "avg_intended_air_yards",
                "avg_time_to_throw",
            ]
        ].rename(
            columns={
                "aggressiveness": "aggressiveness_std",
                "completion_percentage_above_expectation": "completion_percentage_above_expectation_std",
                "avg_intended_air_yards": "avg_intended_air_yards_std",
                "avg_time_to_throw": "avg_time_to_throw_std",
            }
        )

    # --- Game environment from schedules ---------------------------------
    game_env = _build_game_environment(sch)

    # --- Opponent strength proxy ------------------------------------------
    opp_proxy_full = opp_allowed_fppg(
        ws_qb, position=Position.QB, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_qb_fppg_l4"})

    # --- Assemble: depth chart drives the row set, join everything else ---
    out = qb_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(pass_att_l4, on="gsis_id", how="left")
    out = out.merge(pass_yd_l4, on="gsis_id", how="left")
    out = out.merge(pass_td_l4, on="gsis_id", how="left")
    out = out.merge(int_l4, on="gsis_id", how="left")
    out = out.merge(sacks_l4, on="gsis_id", how="left")
    out = out.merge(pass_yd_std, on="gsis_id", how="left")
    out = out.merge(rush_att_l4, on="gsis_id", how="left")
    out = out.merge(rush_yd_l4, on="gsis_id", how="left")
    out = out.merge(snap_l4, on="gsis_id", how="left")
    out = out.merge(ngs_cols, on="gsis_id", how="left")
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_qb_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float)

    out["rushing_qb"] = out["rushing_attempts_per_game_l4"] >= _RUSHING_QB_THRESHOLD
    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())

    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return QbFeaturesSchema.validate(out)
```

- [ ] **Step 4: Re-export from `src/projections/features/__init__.py`**

Update to:
```python
"""Per-position feature builders. Pure functions; no I/O."""

from __future__ import annotations

from projections.features.qb import build_qb_features
from projections.features.wr import build_wr_features

__all__ = ["build_qb_features", "build_wr_features"]
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest -v tests/test_features/test_qb.py
```

Expected: 7 PASS.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
git add src/projections/features/qb.py src/projections/features/__init__.py tests/test_features/test_qb.py
git commit -m "feat(features): add build_qb_features — pure-function QB feature builder

Mirrors build_wr_features's shape: leakage-safe filtering, schema-
validated output. Consumes ngs_passing (vs ngs_receiving for WR).
Reuses _build_game_environment, _prior_mask, _exact_week_mask helpers
from wr.py. Adds rushing_qb boolean flag (>=5 carries/game over
trailing 4)."
```

---

### Task 9: QB leakage tests

**Files:**
- Create: `tests/test_features/test_qb_leakage.py`

- [ ] **Step 1: Write 5 leakage tests (one per input source)**

Create `tests/test_features/test_qb_leakage.py`:

```python
"""Leakage tests for build_qb_features. One assertion per input source."""

from __future__ import annotations

import pandas as pd

from projections.features import build_qb_features
from projections.schemas import _PYARROW_STR

_AS_OF_WEEK = 5
_SEASON = 2024


def _baseline(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> pd.DataFrame:
    return build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def test_no_leakage_from_weekly_stats(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(qb_weekly_stats, qb_snap_counts, qb_depth_charts, qb_ngs_passing, qb_schedules)
    leaky = qb_weekly_stats.copy()
    mask_future = (leaky["gsis_id"] == "00-0034857") & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[mask_future, "passing_yards"] = 999.0
    leaky.loc[mask_future, "attempts"] = 60
    leaky.loc[mask_future, "carries"] = 15
    after = build_qb_features(
        weekly_stats=leaky,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_snap_counts(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(qb_weekly_stats, qb_snap_counts, qb_depth_charts, qb_ngs_passing, qb_schedules)
    leaky = qb_snap_counts.copy()
    mask_future = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[mask_future, "offense_pct"] = 0.0
    leaky.loc[mask_future, "offense_snaps"] = 0
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=leaky,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_ngs_passing(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(qb_weekly_stats, qb_snap_counts, qb_depth_charts, qb_ngs_passing, qb_schedules)
    extra = pd.DataFrame(
        [
            {
                "gsis_id": "00-0034857",
                "season": 2024,
                "week": 5,
                "team": "KC",
                "position": "QB",
                "avg_time_to_throw": 99.0,
                "avg_completed_air_yards": 99.0,
                "avg_intended_air_yards": 99.0,
                "avg_air_yards_differential": 99.0,
                "aggressiveness": 99.0,
                "max_completed_air_distance": 99.0,
                "avg_air_yards_to_sticks": 99.0,
                "completion_percentage": 99.0,
                "expected_completion_percentage": 99.0,
                "completion_percentage_above_expectation": 99.0,
                "avg_air_distance": 99.0,
                "max_air_distance": 99.0,
            }
        ]
    ).astype({"gsis_id": _PYARROW_STR, "team": _PYARROW_STR, "position": _PYARROW_STR})
    leaky = pd.concat([qb_ngs_passing, extra], ignore_index=True)
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=leaky,
        schedules=qb_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_depth_charts_other_weeks(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(qb_weekly_stats, qb_snap_counts, qb_depth_charts, qb_ngs_passing, qb_schedules)
    extra_weeks = pd.concat(
        [
            qb_depth_charts.assign(week=4, depth_rank=99, depth_team="QB99"),
            qb_depth_charts.assign(week=6, depth_rank=99, depth_team="QB99"),
        ],
        ignore_index=True,
    )
    leaky = pd.concat([qb_depth_charts, extra_weeks], ignore_index=True)
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=leaky,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_schedules_other_weeks(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(qb_weekly_stats, qb_snap_counts, qb_depth_charts, qb_ngs_passing, qb_schedules)
    extra_weeks = qb_schedules.assign(week=6, total_line=99.0, spread_line=99.0)
    leaky = pd.concat([qb_schedules, extra_weeks], ignore_index=True)
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=leaky,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)
```

- [ ] **Step 2: Run, verify all 5 pass**

```bash
pytest -v tests/test_features/test_qb_leakage.py
```

Expected: 5 PASS. **If any fails, that's a real leak: investigate `build_qb_features` and fix; don't tweak the test.**

- [ ] **Step 3: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add tests/test_features/test_qb_leakage.py
git commit -m "test(features): add 5 leakage tests for build_qb_features

One assertion per input source. Same strategy as test_wr_leakage.py:
inject implausible rows for week >= as_of_week, rebuild, assert
byte-equal output."
```

---

### Task 10: `src/projections/features/rb.py` — `build_rb_features` + non-leakage tests

**Files:**
- Create: `src/projections/features/rb.py`
- Modify: `src/projections/features/__init__.py` (re-export `build_rb_features`)
- Create: `tests/test_features/test_rb.py`

- [ ] **Step 1: Write failing tests in `tests/test_features/test_rb.py`**

```python
"""RB feature builder tests (non-leakage). Leakage tests live in test_rb_leakage.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_rb_features
from projections.schemas import RbFeaturesSchema


def test_build_rb_features_returns_validated_frame(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    RbFeaturesSchema.validate(out)


def test_build_rb_features_one_row_per_rostered_rb(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0034796", "00-0036650"}


def test_build_rb_features_carries_per_game_l4_correct(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """Saquon weeks 1-4: 20 carries/game uniformly → mean = 20.0."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    saquon = out[out["gsis_id"] == "00-0034796"].iloc[0]
    assert saquon["carries_per_game_l4"] == 20.0


def test_build_rb_features_rush_share_l4_solo_rb_is_one(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """Each fixture team has only one RB in the fixture → rush_share_l4 = 1.0."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    assert (out["rush_share_l4"] == 1.0).all()


def test_build_rb_features_passing_down_back_true_above_threshold(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """CMC has 6 targets/game → passing_down_back == True (>=4.0)."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    cmc = out[out["gsis_id"] == "00-0036650"].iloc[0]
    assert bool(cmc["passing_down_back"]) is True


def test_build_rb_features_passing_down_back_false_below_threshold(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """Saquon has 2 targets/game → passing_down_back == False."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    saquon = out[out["gsis_id"] == "00-0034796"].iloc[0]
    assert bool(saquon["passing_down_back"]) is False


def test_build_rb_features_target_share_against_full_pass_catching_group(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """target_share denominator must include WR + RB + TE on the team.
    The fixture has only RB rows; if no other receivers, RB target_share = 1.0
    (or 0 if RB has 0 targets — but Saquon has 2/game, CMC has 6/game)."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    # With no WR/TE rows in fixture, RB share against the (RB-only) pass-catching
    # set is 1.0 for both.
    assert (out["target_share_l4"] == 1.0).all()
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_features/test_rb.py
```

Expected: 7 FAIL with ImportError.

- [ ] **Step 3: Implement `src/projections/features/rb.py`**

```python
"""RB feature builder. Pure function — no I/O, no caching."""

from __future__ import annotations

from typing import Final

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import last_n_per_group, trailing_n_share_in_group
from projections.features.wr import _build_game_environment, _exact_week_mask, _prior_mask
from projections.schemas import (
    _PYARROW_STR,
    Position,
    RbFeaturesSchema,
    Ruleset,
    Stat,
)

_PASSING_DOWN_BACK_THRESHOLD: Final = 4.0  # targets/game over trailing 4

_PASS_CATCHING_POSITIONS: tuple[str, ...] = (
    Position.WR.value,
    Position.RB.value,
    Position.TE.value,
)

_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "carries_per_game_l4",
    "rushing_yards_per_game_l4",
    "rushing_tds_per_game_l4",
    "rush_share_l4",
    "targets_per_game_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "target_share_l4",
    "targets_per_game_std",
)


def _trailing_4_per_player(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "mean_l4": pd.array([], dtype=float),
            }
        )
    last4 = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    return (
        last4.groupby("gsis_id", as_index=False, observed=True)[value_col]
        .mean()
        .rename(columns={value_col: "mean_l4"})
    )


def _latest_ngs_snapshot(ngs: pd.DataFrame) -> pd.DataFrame:
    if ngs.empty:
        return pd.DataFrame()
    return (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False, observed=True)
        .tail(1)
        .copy()
    )


def build_rb_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_rushing: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the RB feature DataFrame for week `as_of_week` of `season`."""
    ws = weekly_stats[_prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[_prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_rushing[_prior_mask(ngs_rushing, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[_exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[_exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    rb_dc = dc[dc["position"] == Position.RB.value].copy()
    if rb_dc.empty:
        empty_cols = list(RbFeaturesSchema.to_schema().columns.keys())
        return RbFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    ws_rb = ws[ws["position"] == Position.RB.value].copy()
    sc_rb = sc[sc["position"] == Position.RB.value].copy()
    ws_pass_catchers = ws[ws["position"].isin(_PASS_CATCHING_POSITIONS)].copy()

    # --- Rolling per-player rushing/receiving features --------------------
    carries_l4 = _trailing_4_per_player(ws_rb, Stat.CARRIES.value).rename(
        columns={"mean_l4": "carries_per_game_l4"}
    )
    rush_yd_l4 = _trailing_4_per_player(ws_rb, Stat.RUSHING_YARDS.value).rename(
        columns={"mean_l4": "rushing_yards_per_game_l4"}
    )
    rush_td_l4 = _trailing_4_per_player(ws_rb, Stat.RUSHING_TDS.value).rename(
        columns={"mean_l4": "rushing_tds_per_game_l4"}
    )
    targets_l4 = _trailing_4_per_player(ws_rb, Stat.TARGETS.value).rename(
        columns={"mean_l4": "targets_per_game_l4"}
    )
    rec_l4 = _trailing_4_per_player(ws_rb, Stat.RECEPTIONS.value).rename(
        columns={"mean_l4": "receptions_per_game_l4"}
    )
    rec_yd_l4 = _trailing_4_per_player(ws_rb, Stat.RECEIVING_YARDS.value).rename(
        columns={"mean_l4": "receiving_yards_per_game_l4"}
    )

    # --- Shares -----------------------------------------------------------
    # rush_share: among the team's RBs only.
    rush_share = trailing_n_share_in_group(ws_rb, value_col=Stat.CARRIES.value).rename(
        columns={"share_l4": "rush_share_l4"}
    )
    # target_share: among the team's full pass-catching group (WR + RB + TE).
    all_target_share = trailing_n_share_in_group(
        ws_pass_catchers, value_col=Stat.TARGETS.value
    ).rename(columns={"share_l4": "target_share_l4"})
    # Filter the resulting share frame to RB players only (for the merge below).
    rb_gsis_ids = set(ws_rb["gsis_id"].unique())
    target_share = all_target_share[all_target_share["gsis_id"].isin(rb_gsis_ids)].copy()

    # --- Season-to-date targets-per-game (RBs only) -----------------------
    ws_this_season = ws_rb[ws_rb["season"] == season]
    if ws_this_season.empty:
        targets_std = pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "targets_per_game_std": pd.array([], dtype=float),
            }
        )
    else:
        targets_std = (
            ws_this_season.groupby("gsis_id", as_index=False, observed=True)[Stat.TARGETS.value]
            .mean()
            .rename(columns={Stat.TARGETS.value: "targets_per_game_std"})
        )

    snap_l4 = _trailing_4_per_player(sc_rb, Stat.OFFENSE_PCT.value).rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # --- NGS rushing latest snapshot --------------------------------------
    ngs_latest = _latest_ngs_snapshot(ngs)
    ngs_std_cols = (
        "efficiency_std",
        "rush_yards_over_expected_per_att_std",
        "percent_attempts_gte_eight_defenders_std",
    )
    if ngs_latest.empty:
        ngs_cols = pd.DataFrame(
            {"gsis_id": pd.array([], dtype=_PYARROW_STR)}
            | {c: pd.array([], dtype=float) for c in ngs_std_cols}
        )
    else:
        ngs_cols = ngs_latest[
            [
                "gsis_id",
                "efficiency",
                "rush_yards_over_expected_per_att",
                "percent_attempts_gte_eight_defenders",
            ]
        ].rename(
            columns={
                "efficiency": "efficiency_std",
                "rush_yards_over_expected_per_att": "rush_yards_over_expected_per_att_std",
                "percent_attempts_gte_eight_defenders": "percent_attempts_gte_eight_defenders_std",
            }
        )

    # --- Game environment + opponent strength -----------------------------
    game_env = _build_game_environment(sch)
    opp_proxy_full = opp_allowed_fppg(
        ws_rb, position=Position.RB, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_rb_fppg_l4"})

    # --- Assemble ---------------------------------------------------------
    out = rb_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(carries_l4, on="gsis_id", how="left")
    out = out.merge(rush_yd_l4, on="gsis_id", how="left")
    out = out.merge(rush_td_l4, on="gsis_id", how="left")
    out = out.merge(rush_share, on="gsis_id", how="left")
    out = out.merge(targets_l4, on="gsis_id", how="left")
    out = out.merge(rec_l4, on="gsis_id", how="left")
    out = out.merge(rec_yd_l4, on="gsis_id", how="left")
    out = out.merge(target_share, on="gsis_id", how="left")
    out = out.merge(targets_std, on="gsis_id", how="left")
    out = out.merge(snap_l4, on="gsis_id", how="left")
    out = out.merge(ngs_cols, on="gsis_id", how="left")
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_rb_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float)

    out["passing_down_back"] = out["targets_per_game_l4"] >= _PASSING_DOWN_BACK_THRESHOLD
    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())

    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return RbFeaturesSchema.validate(out)
```

- [ ] **Step 4: Re-export from `__init__.py`**

```python
"""Per-position feature builders. Pure functions; no I/O."""

from __future__ import annotations

from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.wr import build_wr_features

__all__ = ["build_qb_features", "build_rb_features", "build_wr_features"]
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest -v tests/test_features/test_rb.py
```

Expected: 7 PASS.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/features/rb.py src/projections/features/__init__.py tests/test_features/test_rb.py
git commit -m "feat(features): add build_rb_features — pure-function RB feature builder

Mirrors build_wr_features's shape. Consumes ngs_rushing. rush_share
uses team's RBs only as denominator; target_share uses team's full
pass-catching group (WR + RB + TE) as denominator. passing_down_back
boolean flag (>=4 targets/game over trailing 4)."
```

---

### Task 11: RB leakage tests

**Files:**
- Create: `tests/test_features/test_rb_leakage.py`

- [ ] **Step 1: Write 5 leakage tests (one per input source)**

Mirror `test_qb_leakage.py` structure exactly, replacing `qb_*` fixture names with `rb_*`, `build_qb_features` with `build_rb_features`, `ngs_passing` with `ngs_rushing`. Inject implausible weeks 5+:

- weekly_stats: Saquon `carries=99`, `rushing_yards=999.0`, `targets=99`
- snap_counts: `offense_pct=0.0`, `offense_snaps=0`
- ngs_rushing: extra row at week 5 with all-99s for Saquon
- depth_charts: extra weeks 4 and 6 rows with `depth_rank=99`, `depth_team="RB99"`
- schedules: extra week 6 row with `total_line=99.0`, `spread_line=99.0`

For NGS injection, the extra row schema must match `NgsRushingSchema`:

```python
extra = pd.DataFrame(
    [
        {
            "gsis_id": "00-0034796",
            "season": 2024,
            "week": 5,
            "team": "PHI",
            "position": "RB",
            "efficiency": 99.0,
            "percent_attempts_gte_eight_defenders": 99.0,
            "avg_time_to_los": 99.0,
            "rush_attempts": 99,
            "rush_yards": 999,
            "expected_rush_yards": 99.0,
            "rush_yards_over_expected": 99.0,
            "avg_rush_yards": 99.0,
            "rush_yards_over_expected_per_att": 99.0,
            "rush_pct_over_expected": 99.0,
        }
    ]
).astype({"gsis_id": _PYARROW_STR, "team": _PYARROW_STR, "position": _PYARROW_STR})
```

- [ ] **Step 2: Run, verify all 5 pass**

```bash
pytest -v tests/test_features/test_rb_leakage.py
```

Expected: 5 PASS.

- [ ] **Step 3: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add tests/test_features/test_rb_leakage.py
git commit -m "test(features): add 5 leakage tests for build_rb_features"
```

---

### Task 12: `src/projections/features/te.py` — `build_te_features` + non-leakage tests

**Files:**
- Create: `src/projections/features/te.py`
- Modify: `src/projections/features/__init__.py` (re-export `build_te_features`)
- Create: `tests/test_features/test_te.py`

- [ ] **Step 1: Write failing tests in `tests/test_features/test_te.py`**

```python
"""TE feature builder tests (non-leakage). Leakage tests live in test_te_leakage.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_te_features
from projections.schemas import TeFeaturesSchema


def test_build_te_features_returns_validated_frame(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    TeFeaturesSchema.validate(out)


def test_build_te_features_one_row_per_rostered_te(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0030506", "00-0033084"}


def test_build_te_features_targets_per_game_l4_correct(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """Kelce: 8 targets/game → mean = 8.0."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    assert kelce["targets_per_game_l4"] == 8.0


def test_build_te_features_target_share_against_full_pass_catching_group(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """Kelce 8 targets + Rice 8 targets = 16 KC team total. Kelce share = 8/16 = 0.5.
    Kittle 6 targets + Aiyuk 6 targets = 12 SF team total. Kittle share = 6/12 = 0.5."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    kittle = out[out["gsis_id"] == "00-0033084"].iloc[0]
    assert kelce["target_share_l4"] == pytest.approx(0.5, abs=1e-6)
    assert kittle["target_share_l4"] == pytest.approx(0.5, abs=1e-6)


def test_build_te_features_implied_team_total_from_schedules(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """KC @ ARI, total=49, spread_line=-7.5 (KC away favored).
    KC implied = (49 - (-7.5))/2 = 28.25."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    assert kelce["implied_team_total"] == pytest.approx(28.25, abs=1e-6)


def test_build_te_features_roof_dome_true_for_closed_roof(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """ARI has roof='closed' in fixture → roof_dome == True."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    assert out["roof_dome"].all()


def test_build_te_features_ngs_separation_propagates(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    # Fixture sets avg_separation=2.8 for Kelce.
    assert kelce["avg_separation_std"] == pytest.approx(2.8, abs=1e-6)
```

- [ ] **Step 2: Run, verify ImportError**

```bash
pytest -v tests/test_features/test_te.py
```

Expected: 7 FAIL.

- [ ] **Step 3: Implement `src/projections/features/te.py`**

```python
"""TE feature builder. Pure function — no I/O, no caching."""

from __future__ import annotations

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import last_n_per_group, trailing_n_share_in_group
from projections.features.wr import _build_game_environment, _exact_week_mask, _prior_mask
from projections.schemas import (
    _PYARROW_STR,
    Position,
    Ruleset,
    Stat,
    TeFeaturesSchema,
)

_PASS_CATCHING_POSITIONS: tuple[str, ...] = (
    Position.WR.value,
    Position.RB.value,
    Position.TE.value,
)

_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
)


def _trailing_4_per_player(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "mean_l4": pd.array([], dtype=float),
            }
        )
    last4 = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    return (
        last4.groupby("gsis_id", as_index=False, observed=True)[value_col]
        .mean()
        .rename(columns={value_col: "mean_l4"})
    )


def _latest_ngs_snapshot(ngs: pd.DataFrame) -> pd.DataFrame:
    if ngs.empty:
        return pd.DataFrame()
    return (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False, observed=True)
        .tail(1)
        .copy()
    )


def build_te_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the TE feature DataFrame for week `as_of_week` of `season`."""
    ws = weekly_stats[_prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[_prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_receiving[_prior_mask(ngs_receiving, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[_exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[_exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    te_dc = dc[dc["position"] == Position.TE.value].copy()
    if te_dc.empty:
        empty_cols = list(TeFeaturesSchema.to_schema().columns.keys())
        return TeFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    ws_te = ws[ws["position"] == Position.TE.value].copy()
    sc_te = sc[sc["position"] == Position.TE.value].copy()
    ws_pass_catchers = ws[ws["position"].isin(_PASS_CATCHING_POSITIONS)].copy()

    # Rolling per-player receiving features (TE rows only)
    targets_l4 = _trailing_4_per_player(ws_te, Stat.TARGETS.value).rename(
        columns={"mean_l4": "targets_per_game_l4"}
    )
    rec_l4 = _trailing_4_per_player(ws_te, Stat.RECEPTIONS.value).rename(
        columns={"mean_l4": "receptions_per_game_l4"}
    )
    rec_yd_l4 = _trailing_4_per_player(ws_te, Stat.RECEIVING_YARDS.value).rename(
        columns={"mean_l4": "receiving_yards_per_game_l4"}
    )
    rec_td_l4 = _trailing_4_per_player(ws_te, Stat.RECEIVING_TDS.value).rename(
        columns={"mean_l4": "receiving_tds_per_game_l4"}
    )

    # target_share against the full team pass-catching group
    all_target_share = trailing_n_share_in_group(
        ws_pass_catchers, value_col=Stat.TARGETS.value
    ).rename(columns={"share_l4": "target_share_l4"})
    te_gsis_ids = set(ws_te["gsis_id"].unique())
    target_share = all_target_share[all_target_share["gsis_id"].isin(te_gsis_ids)].copy()

    # Season-to-date targets/game (TEs only)
    ws_this_season = ws_te[ws_te["season"] == season]
    if ws_this_season.empty:
        targets_std = pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "targets_per_game_std": pd.array([], dtype=float),
            }
        )
    else:
        targets_std = (
            ws_this_season.groupby("gsis_id", as_index=False, observed=True)[Stat.TARGETS.value]
            .mean()
            .rename(columns={Stat.TARGETS.value: "targets_per_game_std"})
        )

    snap_l4 = _trailing_4_per_player(sc_te, Stat.OFFENSE_PCT.value).rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # NGS receiving latest snapshot
    ngs_latest = _latest_ngs_snapshot(ngs)
    ngs_std_cols = (
        "avg_separation_std",
        "avg_intended_air_yards_std",
        "avg_yac_above_expectation_std",
    )
    if ngs_latest.empty:
        ngs_cols = pd.DataFrame(
            {"gsis_id": pd.array([], dtype=_PYARROW_STR)}
            | {c: pd.array([], dtype=float) for c in ngs_std_cols}
        )
    else:
        ngs_cols = ngs_latest[
            [
                "gsis_id",
                "avg_separation",
                "avg_intended_air_yards",
                "avg_yac_above_expectation",
            ]
        ].rename(
            columns={
                "avg_separation": "avg_separation_std",
                "avg_intended_air_yards": "avg_intended_air_yards_std",
                "avg_yac_above_expectation": "avg_yac_above_expectation_std",
            }
        )

    game_env = _build_game_environment(sch)
    opp_proxy_full = opp_allowed_fppg(
        ws_te, position=Position.TE, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_te_fppg_l4"})

    out = te_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(targets_l4, on="gsis_id", how="left")
    out = out.merge(targets_std, on="gsis_id", how="left")
    out = out.merge(target_share, on="gsis_id", how="left")
    out = out.merge(rec_l4, on="gsis_id", how="left")
    out = out.merge(rec_yd_l4, on="gsis_id", how="left")
    out = out.merge(rec_td_l4, on="gsis_id", how="left")
    out = out.merge(snap_l4, on="gsis_id", how="left")
    out = out.merge(ngs_cols, on="gsis_id", how="left")
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_te_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float)

    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())
    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return TeFeaturesSchema.validate(out)
```

- [ ] **Step 4: Re-export from `__init__.py`**

```python
"""Per-position feature builders. Pure functions; no I/O."""

from __future__ import annotations

from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.te import build_te_features
from projections.features.wr import build_wr_features

__all__ = [
    "build_qb_features",
    "build_rb_features",
    "build_te_features",
    "build_wr_features",
]
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest -v tests/test_features/test_te.py
```

Expected: 7 PASS.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add src/projections/features/te.py src/projections/features/__init__.py tests/test_features/test_te.py
git commit -m "feat(features): add build_te_features — pure-function TE feature builder

Mirrors build_wr_features's shape. Consumes ngs_receiving (same as WR).
target_share denominator includes WR + RB + TE (full pass-catching
group) — meaningful for TEs since teams usually have one fantasy-rel TE."
```

---

### Task 13: TE leakage tests

**Files:**
- Create: `tests/test_features/test_te_leakage.py`

- [ ] **Step 1: Write 5 leakage tests**

Mirror `test_qb_leakage.py` exactly, replacing `qb_*` fixture names with `te_*`, `build_qb_features` with `build_te_features`, `ngs_passing` with `ngs_receiving`. Inject implausible weeks 5+:

- weekly_stats: Kelce `targets=99`, `receiving_yards=999.0`, `receiving_tds=9`
- snap_counts: `offense_pct=0.0`, `offense_snaps=0`
- ngs_receiving: extra row at week 5 with all-99s for Kelce (use the same NGS receiving column shape from `wr_ngs_receiving` fixture / `NgsReceivingSchema`)
- depth_charts: extra weeks 4 and 6 rows with `depth_rank=99`, `depth_team="TE99"`
- schedules: extra week 6 row with `total_line=99.0`, `spread_line=99.0`

NGS receiving extra row template:

```python
extra = pd.DataFrame(
    [
        {
            "gsis_id": "00-0030506",
            "season": 2024,
            "week": 5,
            "team": "KC",
            "position": "TE",
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
```

- [ ] **Step 2: Run, verify all 5 pass**

```bash
pytest -v tests/test_features/test_te_leakage.py
```

Expected: 5 PASS.

- [ ] **Step 3: Quality gate + commit**

```bash
pytest -v && mypy src tests && ruff check src tests && ruff format --check src tests
git add tests/test_features/test_te_leakage.py
git commit -m "test(features): add 5 leakage tests for build_te_features"
```

---

## Phase 6 — Integration

### Task 14: Extend smoke test to cover all 4 builders

Rename `tests/test_smoke_2a.py` → `tests/test_smoke.py` and extend to also build QB/RB/TE features from the same round-tripped ingest output.

**Files:**
- Move: `tests/test_smoke_2a.py` → `tests/test_smoke.py`
- Modify: the renamed `tests/test_smoke.py` (add QB/RB/TE ingest-steps + builder-calls)
- Modify: `tests/conftest.py` (extend `fake_snap_counts_df` / `fake_depth_charts_df` / `fake_ngs_*_df` minimally if needed to support QB/RB/TE rows)

- [ ] **Step 1: Rename the smoke test file**

```bash
git mv tests/test_smoke_2a.py tests/test_smoke.py
```

- [ ] **Step 2: Re-read the existing smoke test to understand the flow**

```bash
cat tests/test_smoke.py
```

The existing test:
1. Monkey-patches every `_fetch_raw_*`.
2. Builds id_map + ingests all 5 sources for season=2024.
3. Asserts manifest contains expected tables.
4. Reads every partition.
5. Extends depth_charts + schedules with week-4 rows (fixtures only cover week 3).
6. Calls `build_wr_features(..., as_of_week=4)`.
7. Asserts output validates and contains Justin Jefferson.

- [ ] **Step 3: Extend fixtures if needed**

The existing top-level `tests/conftest.py` fixtures (`fake_weekly_df`, `fake_snap_counts_df`, `fake_depth_charts_df`, `fake_ngs_passing_df`, `fake_ngs_rushing_df`, `fake_ngs_receiving_df`, `fake_id_map_df`) were created during Plan 2a for the ingest tests. They contain one row per fixture (e.g., Mahomes for QB, Saquon for RB). We need the smoke test to exercise all 4 builders, so:

- `fake_weekly_df` already has Mahomes (QB) and Jefferson (WR) rows. Good.
- `fake_snap_counts_df` needs QB/WR rows. Good (built during 2a).
- `fake_depth_charts_df` has Jefferson (WR), Mahomes (QB), Barkley (RB). Missing TE. Add one TE row.
- `fake_ngs_passing_df`, `fake_ngs_rushing_df`, `fake_ngs_receiving_df` — already exist.

Add a TE row to `fake_depth_charts_df` and add a TE row to `fake_weekly_df`. Re-read `tests/conftest.py` to see current contents, then add:

To `fake_weekly_df` (add one row for a TE, matching the structure):

```python
# ... existing rows ...
# Add Travis Kelce TE row for the smoke test
{
    "player_id": "00-0030506",
    "season": 2024,
    "week": 3,
    "position": "TE",
    "recent_team": "KC",
    "opponent_team": "ATL",
    "passing_yards": 0.0,
    "passing_tds": 0,
    "interceptions": 0,
    "attempts": 0,
    "completions": 0,
    "sacks": 0,
    "rushing_yards": 0.0,
    "rushing_tds": 0,
    "carries": 0,
    "receptions": 5,
    "receiving_yards": 58.0,
    "receiving_tds": 1,
    "receiving_air_yards": 70.0,
    "targets": 7,
    "fumbles_lost": 0,
},
```

To `fake_depth_charts_df` (add a TE row):

```python
# ... existing rows ...
{
    "season": 2024,
    "club_code": "KC",
    "week": 3,
    "depth_team": "TE1",
    "last_name": "Kelce",
    "first_name": "Travis",
    "formation": "Offense",
    "gsis_id": "00-0030506",
    "jersey_number": 87,
    "position": "TE",
    "elias_id": "KEL235109",
    "depth_position": 1,
    "football_name": "Travis Kelce",
},
```

To `fake_snap_counts_df` (add a TE row so snap ingest produces a TE entry):

```python
# ... existing rows ...
{
    "game_id": "2024_03_KC_ATL",
    "season": 2024,
    "week": 3,
    "player": "Travis Kelce",
    "position": "TE",
    "team": "KC",
    "opponent": "ATL",
    "offense_snaps": 58,
    "offense_pct": 0.89,
    "defense_snaps": 0,
    "defense_pct": 0.0,
    "st_snaps": 0,
    "st_pct": 0.0,
    "pfr_player_id": "KelcTr00",
},
```

To `fake_id_map_df` (add Kelce so the snap-counts join resolves):

```python
# ... existing rows ...
# Add one row for Kelce so snap-counts ingest picks him up
```

Append to the existing `fake_id_map_df` DataFrame — e.g., extend each list column with Kelce's values: gsis_id `"00-0030506"`, espn_id `"15847"`, sleeper_id `"1466"`, pfr_id `"KelcTr00"`, name `"Travis Kelce"`, position `"TE"`, team `"KC"`.

- [ ] **Step 4: Extend the smoke test itself**

Update `tests/test_smoke.py` to call all 4 builders. The test function name should be renamed to `test_end_to_end_ingest_and_features` (dropping the "_wr" suffix). Add monkeypatches for `ngs_passing` and `ngs_rushing` (the existing test only patches `ngs` for receiving).

```python
"""End-to-end smoke test for ingest + feature-builder integration.

Wires every ingest module and all 4 position builders (QB/RB/WR/TE)
against synthetic fixtures. Catches integration gaps per-module tests
miss:
- write_partition / read_partition path conventions matching
- Manifest update behavior across multiple tables
- Dtype drift between ingest output and feature input"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.features import (
    build_qb_features,
    build_rb_features,
    build_te_features,
    build_wr_features,
)
from projections.ingest import (
    build_id_map,
    refresh_depth_charts,
    refresh_ngs,
    refresh_schedules,
    refresh_snap_counts,
    refresh_weekly_stats,
)
from projections.ingest.manifest import read_manifest
from projections.schemas import (
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)
from projections.store import read_partition


def test_end_to_end_ingest_and_features(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_weekly_df: pd.DataFrame,
    fake_schedules_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    fake_depth_charts_df: pd.DataFrame,
    fake_ngs_passing_df: pd.DataFrame,
    fake_ngs_rushing_df: pd.DataFrame,
    fake_ngs_receiving_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map",
        lambda: fake_id_map_df,
    )
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
    ngs_fixtures = {
        "passing": fake_ngs_passing_df,
        "rushing": fake_ngs_rushing_df,
        "receiving": fake_ngs_receiving_df,
    }
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda stat_type, seasons: ngs_fixtures[stat_type],
    )

    # 1) Build id_map first (required by snap_counts ingest)
    build_id_map(tmp_path)

    # 2) Ingest everything
    refresh_weekly_stats(tmp_path, seasons=[2024])
    refresh_schedules(tmp_path, seasons=[2024])
    refresh_snap_counts(tmp_path, seasons=[2024])
    refresh_depth_charts(tmp_path, seasons=[2024])
    for st in ("passing", "rushing", "receiving"):
        refresh_ngs(tmp_path, stat_type=st, seasons=[2024])

    # 3) Manifest has one row per ingest table
    manifest = read_manifest(tmp_path)
    tables_in_manifest = set(manifest["table"].tolist())
    expected = {
        "weekly_stats",
        "schedules",
        "snap_counts",
        "depth_charts",
        "ngs_passing",
        "ngs_rushing",
        "ngs_receiving",
    }
    assert expected <= tables_in_manifest

    # 4) Read each partition back
    weekly = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    schedules = read_partition(tmp_path / "raw", "schedules", season=2024)
    snaps = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    depth = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    ngs_passing = read_partition(tmp_path / "raw", "ngs_passing", season=2024)
    ngs_rushing = read_partition(tmp_path / "raw", "ngs_rushing", season=2024)
    ngs_receiving = read_partition(tmp_path / "raw", "ngs_receiving", season=2024)

    # 5) Fixtures all describe week 3 of 2024. Extend depth + schedules with
    # week-4 rows for the feature builders (as_of_week=4 needs week 3 in prior
    # window AND week-4 depth/schedule rows).
    extra_dc = pd.concat([depth, depth.assign(week=4)], ignore_index=True)
    extra_sched = pd.concat([schedules, schedules.assign(week=4)], ignore_index=True)

    # 6) Build all 4 position features for as_of_week=4
    qb_out = build_qb_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_passing=ngs_passing,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )
    rb_out = build_rb_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_rushing=ngs_rushing,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )
    wr_out = build_wr_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_receiving=ngs_receiving,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )
    te_out = build_te_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_receiving=ngs_receiving,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )

    # 7) Each output validates and contains the expected fixture player
    QbFeaturesSchema.validate(qb_out)
    assert "00-0034857" in qb_out["gsis_id"].tolist()  # Mahomes

    RbFeaturesSchema.validate(rb_out)
    assert "00-0034796" in rb_out["gsis_id"].tolist()  # Barkley

    WrFeaturesSchema.validate(wr_out)
    assert "00-0036322" in wr_out["gsis_id"].tolist()  # Jefferson

    TeFeaturesSchema.validate(te_out)
    assert "00-0030506" in te_out["gsis_id"].tolist()  # Kelce
```

- [ ] **Step 5: Run smoke test, verify pass**

```bash
pytest -v tests/test_smoke.py
```

Expected: 1 PASS. If a builder crashes or validation fails, investigate the specific integration seam — don't weaken any schema.

- [ ] **Step 6: Quality gate + commit**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
git add tests/test_smoke.py tests/conftest.py
git add -u  # picks up the renamed test_smoke_2a.py → test_smoke.py
git commit -m "test: extend smoke test to cover QB/RB/TE builders

Renamed tests/test_smoke_2a.py → tests/test_smoke.py (covers more than
just 2a now). Builds all 4 position features from the same ingest
output. Extended fixtures with minimal TE rows (Travis Kelce) so the
TE builder has data to exercise."
```

---

## Phase 7 — Wrap-up

### Task 15: Update `project_management.md` and `TODO.md`

Lands the spec §10 documentation updates on the same PR. Capture decisions *as they actually executed* — edit the entries if anything changed during implementation.

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`

- [ ] **Step 1: Append decision-log rows in `project_management.md`**

Append the 8 rows from spec §10.1 at the top of the decision log table (newest-first):

```markdown
| 2026-04-24 | `rushing_qb` boolean threshold = 5.0 carries/game over trailing 4; `passing_down_back` = 4.0 targets/game | Rough heuristics from feel. Not load-bearing; revisit at backtest time if categorization matters |
| 2026-04-24 | TE target_share denominator includes WR + RB + TE (full pass-catching group) | TEs usually have only one fantasy-relevant player per team, so same-position-share would be ~1.0 and useless. Full-group share captures meaningful gradient |
| 2026-04-24 | RB target_share denominator includes WR + RB + TE (full pass-catching group) | A workhorse RB getting 5 targets/game on a 30-target offense is meaningfully different from one getting 5 on a 20-target offense. Full-group denominator captures team passing volume, not just RB-on-RB share |
| 2026-04-24 | Migrate `_trailing_4_share_per_team` from `wr.py` to `_rolling.py` as `trailing_n_share_in_group` | RB needs target_share against the full pass-catching group (not just RBs); TE needs the same. Generalize once, in the shared helper module, rather than duplicate in three builders |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `attempts`, `completions`, `sacks` | QB feature builder needs these source columns. All three are present in raw `nfl_data_py.import_weekly_data` output. Same incremental pattern as 2a's extension for `targets`/`carries`/`receiving_air_yards` |
| 2026-04-24 | One bundled PR for QB/RB/TE (not three per-position PRs) | Repetitive, interlinked work; reviewing all three together catches drift. Each position lands as its own commit inside the bundle for easy retrospection |
| 2026-04-24 | All 4 position builders use parallel files (no WR/TE shared base) | Each position's feature list will diverge over time as we add play-by-play-derived features. Premature DRY hurts later. Shared logic lives in `_rolling.py` / `_opponent.py` |
| 2026-04-24 | K and DST split out into a future plan; 2b covers QB/RB/TE only | K needs FG-attempt data not in `WeeklyStatsSchema`; DST is team-level not player-level and needs play-by-play. Both should wait for the data they need rather than ship degraded v0 features |
```

- [ ] **Step 2: Update the "Current status" / "Next action" sections**

Replace the existing `## Current status (as of 2026-04-24)` block with:

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
- `WeeklyStatsSchema` extended with `attempts`, `completions`, `sacks` for QB features.
- Generalized `trailing_n_share_in_group` helper in `_rolling.py` (migrated from `wr.py`'s local helper).
- ~45 new tests (~200 total). 5 leakage tests per position (15 new).
```

Replace `## Next action`:

```markdown
## Next action

**Recommended: Plan 3 — Model A baseline + season aggregation + first-class backtest harness.**

All 4 offensive skill positions (QB/RB/WR/TE) now have feature builders. Plan 3 trains the v1 model per position, aggregates weekly outputs to season distributions (Monte Carlo with bye + availability), and stands up the backtest harness that gates future model changes.

K and DST builders (TODO #10) can land in parallel with Plan 3 — they're independent.
```

- [ ] **Step 3: Update `TODO.md`**

In `TODO.md`:

a) Mark TODO #2 (Plan 2b) as partially complete. Update its body:

```markdown
### 2. Plan 2b — remaining position feature builders

**Status:** QB/RB/TE complete in Plan 2b (merged). K and DST split out into TODO #10.
```

b) Append TODO #10:

```markdown
### 10. Plan 2c — K and DST feature builders

Both positions need data we don't currently ingest:

- **K**: spec calls for "recent FG distance distribution" and "opp redzone TD allowed %." Neither is in `WeeklyStatsSchema`. Need to ingest a new source covering FG attempts by distance and accuracy by range. `nfl_data_py.import_weekly_pfr_data` may have this — verify before designing.
- **DST**: team-level not player-level. Schema's primary key is `Team`, not `GsisId` — fundamentally different from the per-player pattern Plan 2a/2b established. Intended features (opp pass-block win rate, sack rate allowed, turnover-worthy throw rate) all need play-by-play (TODO #3).

Decision before brainstorming Plan 2c: do we ingest the missing data first (extending the ingest layer), or build degraded v0 K/DST features from `implied_team_total` alone? The latter is fast but creates a future rewrite; the former takes longer but yields the right shape.

Plan 3 (Model A baseline) doesn't depend on K/DST, so this can run in parallel.
```

- [ ] **Step 4: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm): close Plan 2b; queue Plan 3 + K/DST follow-up

Decision-log additions cover the 8 major design calls from spec §10.1.
Status section reflects 2b complete. Next action is Plan 3 (Model A
baseline + backtest harness). TODO #2 marked partially complete;
TODO #10 added for the deferred K/DST builders."
```

---

### Task 16: End-of-effort verification + open PR

**Files:**
- None modified — verification + PR only.

- [ ] **Step 1: Run the full quality gate at the worktree root**

```bash
cd "/c/Users/alden/FantasyFootball/.worktrees/feat-plan-2b-qb-rb-te-features"
. .venv/Scripts/activate
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
pytest -v -k "ingest or store or schemas"
```

Each must be green. Test-count expectation: ~200 total (158 baseline + ~45 new). If anything fails, fix before opening the PR — do not ship a known-broken main check.

- [ ] **Step 2: Capture results for the PR description**

Save a concise summary like: "pytest: 200 passed in 8.1s; mypy: success no issues; ruff check: All checks passed; ruff format: 60 files already formatted" for the PR body.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feat/plan-2b-qb-rb-te-features
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "Plan 2b: QB/RB/TE feature builders" --body "$(cat <<'EOF'
## Summary
- Adds three per-position feature builders — `build_qb_features`, `build_rb_features`, `build_te_features` — following the Plan 2a-validated WR pattern.
- Extends `WeeklyStatsSchema` with `attempts`, `completions`, `sacks` (QB source columns).
- Migrates `wr.py`'s local `_trailing_4_share_per_team` into `_rolling.py` as the generalized `trailing_n_share_in_group` helper; WR now consumes the shared version.
- RB and TE target_share use the team's full pass-catching group (WR + RB + TE) as denominator.
- ~45 new tests including 5 leakage tests per new position (15 new leakage tests).
- K and DST split out into a future plan (TODO #10) — they need data we don't yet ingest.

Spec: `docs/superpowers/specs/2026-04-24-plan-2b-qb-rb-te-features-design.md`
Plan: `docs/superpowers/plans/2026-04-24-plan-2b-qb-rb-te-features.md`

## Quality gate
[Paste the captured Step 2 output here.]

## Test plan
- [x] `pytest -v` — full suite green (~200 tests)
- [x] `mypy src tests` — zero violations
- [x] `ruff check src tests` — zero violations
- [x] `ruff format --check src tests` — no drift
- [x] `pytest -v -k "ingest or store or schemas"` — green
- [ ] Spot-check leakage tests across the three new positions — each has 5 tests, one per input source

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace the `[Paste the captured Step 2 output here.]` placeholder with the actual gate output (fenced code block).

- [ ] **Step 5: Report PR URL + green-gate summary**

Final report includes the PR URL and the captured Step 2 output.

---

## Self-Review

Walked through the spec section-by-section against the plan:

| Spec section | Plan task(s) |
|---|---|
| §2.1 #1 — three new per-position builders | Tasks 8 (QB), 10 (RB), 12 (TE) |
| §2.1 #2 — three new feature schemas | Tasks 4 (QB), 5 (RB), 6 (TE) |
| §2.1 #3 — `WeeklyStatsSchema` extension | Task 2 |
| §2.1 #4 — shared helper extraction | Task 3 |
| §2.1 #5 — per-position tests (non-leakage + leakage) | Tasks 8/9 (QB), 10/11 (RB), 12/13 (TE) |
| §2.1 #6 — smoke-test extension | Task 14 |
| §2.1 #7 — documentation updates | Tasks 1 (2a merge commit) + 15 (PM/TODO) |
| §3.1 — QB feature list | Task 4 schema + Task 8 implementation |
| §3.2 — RB feature list | Task 5 schema + Task 10 implementation |
| §3.3 — TE feature list | Task 6 schema + Task 12 implementation |
| §4.1 — extended `WeeklyStatsSchema` | Task 2 |
| §4.2 — new Stat enum entries | Task 2 |
| §4.3 — new feature schemas | Tasks 4, 5, 6 |
| §5 — `trailing_n_share_in_group` helper | Task 3 |
| §6.1 — per-position non-leakage tests | Tasks 8, 10, 12 |
| §6.2 — per-position leakage tests | Tasks 9, 11, 13 |
| §6.3 — shared synthetic frames | Task 7 |
| §6.4 — smoke test extension | Task 14 |
| §6.6 — end-of-effort checklist | Task 16 |
| §9 — MVP steps 1-18 | Map to Tasks 1 (step 1), 2 (step 2), 3 (step 3), 4-6 (steps 4-6), 7 (step 7), 8-13 (steps 8-13), 14 (step 15), 15 (step 17), 16 (step 18) |
| §10.1, 10.2, 10.3 — documentation updates | Task 15 |

**Placeholder scan:** `<TBD-after-merge>` appears in Task 15 Step 2 — that's intentional (fills in at PR merge time, documented as such). No other TBD/TODO/"implement later" in the plan text.

**Type / signature consistency:**
- `trailing_n_share_in_group` signature matches between Task 3 definition and Tasks 10/12 call sites.
- `build_qb_features` / `build_rb_features` / `build_te_features` all use keyword-only args matching their respective schemas and fixture sets.
- `_build_game_environment`, `_prior_mask`, `_exact_week_mask` imported from `wr.py` at 3 call sites (Tasks 8, 10, 12) — naming is consistent.
- Fixture names (`qb_*`, `rb_*`, `te_*`) are used consistently across Tasks 7, 8-13, and 14.

**Gap check:**
- Task 14 requires `fake_*_df` fixtures at the top-level `tests/conftest.py` to include TE rows. The plan explicitly adds Kelce rows. Good.
- Task 2 touches the `_good_weekly_stats` helper in `tests/test_schemas/test_dataframe_schemas.py`; verified this file exists (from Plan 2a) and has the helper.
- No spec requirement left unplanned.
