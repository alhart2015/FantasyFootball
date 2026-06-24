# DFS Layer 1 — Projection Edge Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a retrospective benchmark that decides, with one pre-registered statistical test, whether our home-grown weekly model (alone or blended with Sleeper) beats Sleeper's own weekly projections under DraftKings base scoring — an ADOPT/STOP/INCONCLUSIVE kill-test gating the larger DFS Engine.

**Architecture:** Pull Sleeper weekly projections (historical, retrospective) and run our model walk-forward over the same past weeks; score both — plus actuals — under a DraftKings *base* `Ruleset`; join on `(gsis_id, season, week)`, restrict to an actuals-conditioned universe, and compute a disagreement head-to-head + ranking skill with a player-season clustered bootstrap and a single pre-registered primary gate. New code lives in a `src/projections/dfs/` package (the home for Layer 2) plus one new ingest module and small scoring/schema additions; the model walk-forward, scoring layer, store, and id-crosswalk are reused.

**Tech Stack:** Python 3.11, pandas (pyarrow dtypes), pandera schemas, numpy, pydantic `Ruleset`, the existing `projections` package (models/backtest/scoring/ingest/store).

## Global Constraints

(Copied from the spec + `CLAUDE.md`; every task implicitly includes these.)

- **`GsisId` is canonical**; all joins/storage use it. Never join on names. Use `validate_gsis_id` only at untrusted boundaries.
- **Reference enums, never the strings they wrap:** `Position.QB`, `Stat.RECEIVING_YARDS`, `ProjectionSource.SLEEPER`, etc.
- **`df = SCHEMA.validate(df)` with reassignment** at every module boundary producing a DataFrame (pandera `strict="filter"` returns a new frame).
- **`pd.StringDtype("pyarrow")` for nullable strings, `pd.Int64Dtype()` for nullable ints.** The repo alias is `_PYARROW_STR` in `schemas.py`.
- **Scoring layer is the only place that knows fantasy-point math.** All scoring routes through `scoring.score` / `scoring.expected_points`. The one exception this slice adds is the deterministic `dk_actuals_bonus` (Task 4), which is *additive* DK-specific logic, still inside `scoring/`.
- **Store I/O only via `projections.store.write_partition` / `read_partition`** — never `df.to_parquet` directly.
- **Skill positions only:** QB, RB, WR, TE.
- **Verification gates (run before any task is "done"):** `pytest -v` (relevant subset OK, state which), `mypy src tests` (strict, zero errors), `ruff check src tests`, `ruff format --check src tests`. For ingest/store/schema tasks also run `pytest -k "ingest or store or schemas"`.
- **Use the worktree venv:** `./.venv/Scripts/python -m pytest ...` (the worktree's editable install points at this worktree's `src`).
- **Pre-registration discipline:** `δ` (disagreement threshold), the actual-usage floor, the anti-masking margin `m`, `N_min`, and the target CI half-width are **constants in `dfs/config.py`**, committed before the verdict is computed — never tuned to the outcome.

**Data dependencies (must exist before Task 8 + Task 10 run on real data):** feature partitions `data/features/{qb,rb,wr,te}/season=YYYY/` and raw `data/raw/weekly_stats/season=YYYY/`, `data/raw/id_map/` for eval seasons 2021–2024 (the model-backtest default `held_out_years`). If absent, build via `scripts/refresh_features.py` / the ingest scripts. Unit tests use synthetic fixtures and do **not** need these.

---

### Task 1: Promote the era-aware regular-season-week helper to a shared module

Centralizes the "17 weeks pre-2021, 18 after" cutoff (TODO #41) so the DK weekly-actuals scorer (Task 7) can reuse it instead of duplicating or importing a private.

**Files:**
- Create: `src/projections/season_calendar.py`
- Modify: `src/projections/draft/assistant/availability.py` (replace private `_sched_games`/`_last_regular_week` with imports)
- Test: `tests/test_season_calendar.py`

**Interfaces:**
- Produces: `regular_season_games(season: int) -> int` (16 if `season <= 2020` else 17); `last_regular_week(season: int) -> int` (`regular_season_games(season) + 1`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_season_calendar.py
from projections.season_calendar import last_regular_week, regular_season_games


def test_regular_season_games_era_split():
    assert regular_season_games(2020) == 16
    assert regular_season_games(2021) == 17


def test_last_regular_week_era_split():
    assert last_regular_week(2019) == 17
    assert last_regular_week(2020) == 17
    assert last_regular_week(2021) == 18
    assert last_regular_week(2024) == 18
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_season_calendar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'projections.season_calendar'`.

- [ ] **Step 3: Write the module**

```python
# src/projections/season_calendar.py
"""Shared NFL regular-season calendar helpers.

The regular season is 16 games (17 calendar weeks incl. the bye) through 2020,
and 17 games (18 weeks) from 2021 on. Ingested `weekly_stats`/`schedules`
number playoff weeks above this (up to 22), so `last_regular_week` is the
"regular season only" cutoff on either. Centralized here per TODO #41.
"""

from __future__ import annotations


def regular_season_games(season: int) -> int:
    """Number of regular-season games: 16 through 2020, 17 from 2021 on."""
    return 16 if season <= 2020 else 17


def last_regular_week(season: int) -> int:
    """Last regular-season calendar week (games + the one bye)."""
    return regular_season_games(season) + 1
```

- [ ] **Step 4: Refactor `availability.py` to reuse it**

In `src/projections/draft/assistant/availability.py`, delete the private `_sched_games` (lines ~19-21) and `_last_regular_week` (lines ~24-30), add `from projections.season_calendar import last_regular_week, regular_season_games` (import **both** — `_sched_games` is called directly at `availability.py:107`, so its replacement `regular_season_games` must be imported too), and replace every internal `_last_regular_week(` call with `last_regular_week(` and every `_sched_games(` call with `regular_season_games(`. Verify with `rg "_last_regular_week|_sched_games" src tests`: there are **no test references** and only a stale *comment* mention in `draft/assistant/season_value.py:~29` (no code change needed there). Expect the grep to come back clean (only the comment) after the edit.

- [ ] **Step 5: Run tests + gates**

Run: `./.venv/Scripts/python -m pytest tests/test_season_calendar.py tests/test_draft -k "availability" -v`
Expected: PASS. Then `mypy src tests` / `ruff check src tests` / `ruff format --check src tests` clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/season_calendar.py src/projections/draft/assistant/availability.py tests/test_season_calendar.py
git commit -m "refactor(calendar): hoist era-aware last_regular_week to season_calendar (TODO #41)"
```

---

### Task 2: Lift `_normalize_join_id` from the script into `src/`

The `.0`-stripping id normalizer currently lives in `scripts/benchmark_projections.py`, which is not an importable package. Lift it into `src/` so the new Sleeper ingest (Task 6) can reuse it (correct layering: `scripts/` → `src/`, never the reverse).

**Files:**
- Modify: `src/projections/ingest/identity.py` (add `normalize_join_id`)
- Modify: `scripts/benchmark_projections.py` (import it; drop the local copy)
- Test: `tests/test_ingest/test_identity.py` (add a case)

**Interfaces:**
- Produces: `normalize_join_id(s: pd.Series) -> pd.Series` — casts to `string`, strips whitespace and a trailing `.0+`, so `'4374302.0'` and `'4374302'` join.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest/test_identity.py  (add)
import pandas as pd
from projections.ingest.identity import normalize_join_id


def test_normalize_join_id_strips_float_suffix_and_whitespace():
    s = pd.Series(["4374302.0", " 4374302 ", "4374302.00", "00-0036900"])
    out = normalize_join_id(s)
    assert out.tolist() == ["4374302", "4374302", "4374302", "00-0036900"]
    assert out.dtype == "string"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_ingest/test_identity.py::test_normalize_join_id_strips_float_suffix_and_whitespace -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_join_id'`.

- [ ] **Step 3: Add the function to `identity.py`**

```python
# src/projections/ingest/identity.py  (add; needs `import pandas as pd`)
def normalize_join_id(s: pd.Series) -> pd.Series:
    """Canonicalize a platform-id column for joining against `id_map`.

    `id_map` stores espn_id/sleeper_id float-stringified ('4374302.0'); external
    pulls write clean int-strings ('4374302'). Cast both sides to a plain string
    with surrounding whitespace and any trailing '.0'(/'.00'...) stripped, so the
    merge matches and dtypes line up. Without this the join silently yields ZERO
    matches (TODO #38 — the deeper fix is casting id_map's id columns to Int64
    at ingest).
    """
    return s.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
```

- [ ] **Step 4: Update the script to import it**

In `scripts/benchmark_projections.py`, delete the local `_normalize_join_id` (lines ~93-106) and add `from projections.ingest.identity import normalize_join_id`. Replace internal `_normalize_join_id(` calls with `normalize_join_id(`.

- [ ] **Step 5: Run tests + gates**

Run: `./.venv/Scripts/python -m pytest tests/test_ingest/test_identity.py tests/test_scripts -k "benchmark or identity" -v`
Expected: PASS (run whichever script tests exist; if none for the script, just the identity test). Then mypy/ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/ingest/identity.py scripts/benchmark_projections.py tests/test_ingest/test_identity.py
git commit -m "refactor(ingest): lift normalize_join_id into src so ingest can reuse it"
```

---

### Task 3: Add the DraftKings base `Ruleset` preset + extend the name allowlist

**Files:**
- Modify: `src/projections/schemas.py` (`Ruleset.draftkings()` classmethod; `_RULESET_NAME_VALUES`)
- Test: `tests/test_schemas/test_ruleset.py` (or the existing ruleset test module)

**Interfaces:**
- Produces: `Ruleset.draftkings() -> Ruleset` — `name="DRAFTKINGS"`, `interception_pts=-1.0`, `fumble_lost_pts=-1.0`; all other coefficients identical to the ESPN-PPR defaults (0.04/pass yd ⇒ `passing_yds_per_pt=25`, 0.1/yd ⇒ `*_yds_per_pt=10`, `reception_pts=1.0`, TDs 4/6/6). `_RULESET_NAME_VALUES` includes `"DRAFTKINGS"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas/test_ruleset.py  (add)
from projections.schemas import Ruleset, _RULESET_NAME_VALUES
from projections.scoring.score import StatLine, score


def test_draftkings_preset_values():
    dk = Ruleset.draftkings()
    assert dk.name == "DRAFTKINGS"
    assert dk.interception_pts == -1.0  # DK is -1, ESPN is -2
    assert dk.fumble_lost_pts == -1.0
    assert dk.reception_pts == 1.0      # full PPR
    assert dk.passing_yds_per_pt == 25.0
    assert dk.rushing_yds_per_pt == 10.0


def test_draftkings_scores_known_line():
    dk = Ruleset.draftkings()
    # 300 pass yd, 2 pass TD, 1 INT, 50 rush yd, 5 rec, 80 rec yd, 1 fum
    line = StatLine(
        passing_yards=300, passing_tds=2, interceptions=1,
        rushing_yards=50, receptions=5, receiving_yards=80, fumbles_lost=1,
    )
    # 300/25 + 2*4 + 1*-1 + 50/10 + 5*1 + 80/10 + 1*-1 = 12 + 8 -1 +5 +5 +8 -1
    assert score(line, dk) == 36.0


def test_draftkings_in_allowlist():
    assert "DRAFTKINGS" in _RULESET_NAME_VALUES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_schemas/test_ruleset.py -k draftkings -v`
Expected: FAIL — `AttributeError: type object 'Ruleset' has no attribute 'draftkings'`.

- [ ] **Step 3: Implement the preset + allowlist**

In `src/projections/schemas.py`, add to `Ruleset` (after `standard`):

```python
    @classmethod
    def draftkings(cls) -> Ruleset:
        """DraftKings NFL Classic *base* scoring (skill positions, no yardage
        bonuses — those are a separate deterministic helper, see
        scoring.draftkings_bonus). Differs from ESPN PPR only in turnovers:
        INT and fumble lost are -1 (ESPN: -2)."""
        return cls(name="DRAFTKINGS", interception_pts=-1.0, fumble_lost_pts=-1.0)
```

And extend the allowlist (line ~302):

```python
_RULESET_NAME_VALUES = ["ESPN_PPR", "ESPN_HALF", "STANDARD", "DRAFTKINGS"]
```

The three pinned `isin` sites (`schemas.py:899/1244/1309`) reference this constant, so they update automatically. **Do not** touch `tests/test_schemas/test_dataframe_schemas.py:1001` (its golden row is `ESPN_PPR`, unaffected) or the closed preset registries `draft/league_config.py:16` / `draft/assistant/presets.py:20` (the DFS path builds `Ruleset.draftkings()` directly — confirm by grep that no DFS code routes a ruleset *name* through those registries).

- [ ] **Step 4: Run tests + gates**

Run: `./.venv/Scripts/python -m pytest tests/test_schemas -v` then `pytest -k "ingest or store or schemas"`.
Expected: PASS (no existing schema test should break — the allowlist only widens).

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_ruleset.py
git commit -m "feat(scoring): add Ruleset.draftkings() base preset + allowlist entry"
```

---

### Task 4: Deterministic DraftKings actuals-bonus helper

A standalone `≥300 pass / ≥100 rush / ≥100 rec → +3` add-on, used only to score **actuals** for the §6.2 sensitivity check. Exact (actual yards are known); not a `Ruleset` field and not used in the base comparison.

**Files:**
- Create: `src/projections/scoring/draftkings_bonus.py`
- Modify: `src/projections/scoring/__init__.py` (re-export `dk_actuals_bonus`)
- Test: `tests/test_scoring/test_draftkings_bonus.py`

**Interfaces:**
- Produces: `dk_actuals_bonus(*, passing_yards: float, rushing_yards: float, receiving_yards: float) -> float` — returns `3.0` per threshold met, stacking (0.0, 3.0, 6.0, or 9.0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring/test_draftkings_bonus.py
import pytest
from projections.scoring import dk_actuals_bonus


@pytest.mark.parametrize(
    "pass_yd, rush_yd, rec_yd, expected",
    [
        (299, 99, 99, 0.0),
        (300, 0, 0, 3.0),
        (0, 100, 0, 3.0),
        (0, 0, 100, 3.0),
        (300, 100, 0, 6.0),
        (350, 120, 110, 9.0),
    ],
)
def test_dk_actuals_bonus_thresholds(pass_yd, rush_yd, rec_yd, expected):
    assert dk_actuals_bonus(
        passing_yards=pass_yd, rushing_yards=rush_yd, receiving_yards=rec_yd
    ) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_scoring/test_draftkings_bonus.py -v`
Expected: FAIL — `ImportError: cannot import name 'dk_actuals_bonus'`.

- [ ] **Step 3: Implement**

```python
# src/projections/scoring/draftkings_bonus.py
"""DraftKings yardage-bonus scoring (skill positions).

DraftKings awards +3 for a 300+ yard passing game, +3 for 100+ rushing, +3 for
100+ receiving. These stack. This is *deterministic actuals* logic: it takes
realized yards, so no probability model is involved. It is intentionally NOT a
`Ruleset` field — the base projection comparison excludes bonuses (a point
projection cannot express E[bonus]); this helper is used only to score actuals
for the edge study's bonus sensitivity check.
"""

from __future__ import annotations

_BONUS = 3.0
_PASS_THRESHOLD = 300.0
_RUSH_THRESHOLD = 100.0
_REC_THRESHOLD = 100.0


def dk_actuals_bonus(
    *, passing_yards: float, rushing_yards: float, receiving_yards: float
) -> float:
    """Total DK yardage bonus for a realized stat line (0/3/6/9)."""
    bonus = 0.0
    if passing_yards >= _PASS_THRESHOLD:
        bonus += _BONUS
    if rushing_yards >= _RUSH_THRESHOLD:
        bonus += _BONUS
    if receiving_yards >= _REC_THRESHOLD:
        bonus += _BONUS
    return bonus
```

Add to `src/projections/scoring/__init__.py`: `from projections.scoring.draftkings_bonus import dk_actuals_bonus` and include `"dk_actuals_bonus"` in `__all__` if present.

- [ ] **Step 4: Run tests + gates**

Run: `./.venv/Scripts/python -m pytest tests/test_scoring/test_draftkings_bonus.py -v` then mypy/ruff.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/scoring/draftkings_bonus.py src/projections/scoring/__init__.py tests/test_scoring/test_draftkings_bonus.py
git commit -m "feat(scoring): deterministic DK actuals yardage-bonus helper"
```

---

### Task 5: `ExternalProjectionWeeklySchema`

A weekly sibling of `ExternalProjectionSchema` (one row per source/player/season/**week**).

**Files:**
- Modify: `src/projections/schemas.py` (new schema)
- Test: `tests/test_schemas/test_external_weekly_schema.py`

**Interfaces:**
- Produces: `ExternalProjectionWeeklySchema` — columns: `source` (`isin=_SOURCE_VALUES`), `source_player_id: str`, `gsis_id: str` (GSIS pattern), `is_placeholder_gsis: bool`, `full_name: str`, `position` (`isin=_POSITION_VALUES`), `season: int` (ge=1999, le=2100), `week: Int64` (ge=1, le=22), and the 9 nullable-float stat fields (`passing_yards`, `passing_tds`, `interceptions`, `rushing_yards`, `rushing_tds`, `receptions`, `receiving_yards`, `receiving_tds`, `fumbles_lost`). `Config: strict="filter", coerce=True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas/test_external_weekly_schema.py
import pandas as pd
import pytest
from projections.schemas import ExternalProjectionWeeklySchema, _PYARROW_STR


def _valid_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": pd.array(["SLEEPER"], dtype=_PYARROW_STR),
            "source_player_id": pd.array(["4046"], dtype=_PYARROW_STR),
            "gsis_id": pd.array(["00-0036900"], dtype=_PYARROW_STR),
            "is_placeholder_gsis": [False],
            "full_name": pd.array(["Ja'Marr Chase"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "season": [2023],
            "week": pd.array([5], dtype="Int64"),
            "passing_yards": pd.array([0.0], dtype="Float64"),
            "passing_tds": pd.array([0.0], dtype="Float64"),
            "interceptions": pd.array([0.0], dtype="Float64"),
            "rushing_yards": pd.array([0.0], dtype="Float64"),
            "rushing_tds": pd.array([0.0], dtype="Float64"),
            "receptions": pd.array([6.0], dtype="Float64"),
            "receiving_yards": pd.array([78.0], dtype="Float64"),
            "receiving_tds": pd.array([0.5], dtype="Float64"),
            "fumbles_lost": pd.array([0.0], dtype="Float64"),
        }
    )


def test_valid_weekly_row_passes():
    out = ExternalProjectionWeeklySchema.validate(_valid_row())
    assert out["week"].iloc[0] == 5


def test_week_out_of_range_rejected():
    bad = _valid_row()
    bad["week"] = pd.array([23], dtype="Int64")
    with pytest.raises(Exception):
        ExternalProjectionWeeklySchema.validate(bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_schemas/test_external_weekly_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'ExternalProjectionWeeklySchema'`.

- [ ] **Step 3: Add the schema** (place it directly after `ExternalProjectionSchema`, reusing the same module constants `_SOURCE_VALUES`, `_POSITION_VALUES`, `GSIS_ID_PATTERN`)

```python
class ExternalProjectionWeeklySchema(pa.DataFrameModel):
    """Per-(source, player, season, week) external weekly projection stat line.

    Weekly sibling of ExternalProjectionSchema. Sleeper's weekly endpoint
    carries a real stat line (unlike its season endpoint). Skill positions only.
    """

    source: Series[str] = pa.Field(isin=_SOURCE_VALUES)
    source_player_id: Series[str]
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    is_placeholder_gsis: Series[bool]
    full_name: Series[str]
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[pd.Int64Dtype] = pa.Field(ge=1, le=22)
    passing_yards: Series[float] = pa.Field(nullable=True)
    passing_tds: Series[float] = pa.Field(nullable=True)
    interceptions: Series[float] = pa.Field(nullable=True)
    rushing_yards: Series[float] = pa.Field(nullable=True)
    rushing_tds: Series[float] = pa.Field(nullable=True)
    receptions: Series[float] = pa.Field(nullable=True)
    receiving_yards: Series[float] = pa.Field(nullable=True)
    receiving_tds: Series[float] = pa.Field(nullable=True)
    fumbles_lost: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True
```

- [ ] **Step 4: Run tests + gates**

Run: `./.venv/Scripts/python -m pytest tests/test_schemas/test_external_weekly_schema.py -v` then `pytest -k "schemas"`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_external_weekly_schema.py
git commit -m "feat(schemas): ExternalProjectionWeeklySchema (weekly external stat line)"
```

---

### Task 6: Sleeper weekly projection ingest

Fetch `api.sleeper.com/projections/nfl/<season>/<week>`, parse the stat line, attach `gsis_id`, persist a weekly partition.

**Files:**
- Create: `src/projections/ingest/sleeper_weekly_projections.py`
- Test: `tests/test_ingest/test_sleeper_weekly_projections.py`

**Interfaces:**
- Consumes: `SLEEPER_STAT_FIELDS` (`ingest/external_projections.py:112`), `normalize_join_id` (Task 2), `placeholder_name_key` + `_make_placeholder_gsis` pattern (`ingest/external_projections.py:242`), `ExternalProjectionWeeklySchema` (Task 5), `write_partition`/`read_partition` (`projections.store`), `_PYARROW_STR`.
- Produces:
  - `parse_sleeper_weekly(payload: list[dict], *, season: int, week: int) -> pd.DataFrame` — pure parser → columns `["sleeper_id","full_name","position","season","week", *9 stat fields]`, skill positions only, rows with an empty `stats` dict dropped.
  - `refresh_sleeper_weekly(data_root: Path, *, season: int, week: int) -> Path` — fetch + parse + attach gsis + validate + `write_partition(... "sleeper_weekly_projections", season=season, week=week)`.

- [ ] **Step 1: Write the failing test (pure parser)**

```python
# tests/test_ingest/test_sleeper_weekly_projections.py
from projections.ingest.sleeper_weekly_projections import parse_sleeper_weekly

_PAYLOAD = [
    {  # WR with a stat line
        "player_id": "4046",
        "player": {"first_name": "Ja'Marr", "last_name": "Chase", "position": "WR"},
        "stats": {"rec": 6.0, "rec_yd": 78.0, "rec_td": 0.5, "fum_lost": 0.1},
    },
    {  # K — non-skill, must be dropped
        "player_id": "999",
        "player": {"first_name": "Foot", "last_name": "Ball", "position": "K"},
        "stats": {"pts_ppr": 8.0},
    },
    {  # empty stats — dropped
        "player_id": "111",
        "player": {"first_name": "No", "last_name": "Proj", "position": "RB"},
        "stats": {},
    },
]


def test_parse_keeps_skill_with_stats_maps_fields():
    df = parse_sleeper_weekly(_PAYLOAD, season=2023, week=5)
    assert df["sleeper_id"].tolist() == ["4046"]
    row = df.iloc[0]
    assert row["position"] == "WR"
    assert row["receptions"] == 6.0
    assert row["receiving_yards"] == 78.0
    assert row["season"] == 2023 and row["week"] == 5
    # unmapped stat keys ignored; absent mapped keys -> NA (not 0)
    assert "passing_yards" in df.columns
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_ingest/test_sleeper_weekly_projections.py::test_parse_keeps_skill_with_stats_maps_fields -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the parser**

```python
# src/projections/ingest/sleeper_weekly_projections.py
"""Ingest Sleeper *weekly* projections (historical, retrospective).

Unlike the season endpoint (ADP-only), the weekly endpoint
`api.sleeper.com/projections/nfl/<season>/<week>` returns a per-player stat
line. We map it to canonical stat fields, attach gsis_id, and store a weekly
partition. Skill positions only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date  # noqa: F401  (kept for symmetry; remove if unused)
from pathlib import Path
from typing import Any

import pandas as pd

from projections.ingest.external_projections import (
    SLEEPER_STAT_FIELDS,
    _make_placeholder_gsis,  # reuse the same placeholder scheme
)
from projections.ingest.identity import normalize_join_id
from projections.schemas import (
    ExternalProjectionWeeklySchema,
    Position,
    ProjectionSource,
    _PYARROW_STR,
)
from projections.store import read_partition, write_partition

_SLEEPER_WEEKLY_URL = (
    "https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular"
)
_SKILL_POSITIONS = {Position.QB.value, Position.RB.value, Position.WR.value, Position.TE.value}
_STAT_FIELDS = list(SLEEPER_STAT_FIELDS.values())


class SleeperWeeklyError(RuntimeError):
    """Raised when the Sleeper weekly endpoint fetch/parse fails."""


def parse_sleeper_weekly(
    payload: list[dict[str, Any]], *, season: int, week: int
) -> pd.DataFrame:
    """Parse the weekly payload into canonical columns. Pure (no I/O)."""
    rows: list[dict[str, Any]] = []
    for entry in payload:
        player = entry.get("player") or {}
        position = player.get("position")
        if position not in _SKILL_POSITIONS:
            continue
        stats = entry.get("stats") or {}
        mapped = {
            canonical: float(stats[key])
            for key, canonical in SLEEPER_STAT_FIELDS.items()
            if key in stats and stats[key] is not None
        }
        if not mapped:  # ADP-only / empty stat line
            continue
        full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        row: dict[str, Any] = {
            "sleeper_id": str(entry["player_id"]),
            "full_name": full_name,
            "position": position,
            "season": season,
            "week": week,
        }
        for field in _STAT_FIELDS:
            row[field] = mapped.get(field, pd.NA)
        rows.append(row)

    columns = ["sleeper_id", "full_name", "position", "season", "week", *_STAT_FIELDS]
    return pd.DataFrame(rows, columns=columns)
```

- [ ] **Step 4: Run parser test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_ingest/test_sleeper_weekly_projections.py::test_parse_keeps_skill_with_stats_maps_fields -v`
Expected: PASS.

- [ ] **Step 5: Write the gsis-attach + store test (with a fake fetch + tmp store)**

```python
# tests/test_ingest/test_sleeper_weekly_projections.py  (add)
import pandas as pd
from projections.ingest import sleeper_weekly_projections as swp
from projections.schemas import ExternalProjectionWeeklySchema
from projections.store import read_partition, write_partition


def test_refresh_attaches_gsis_and_stores(tmp_path, monkeypatch):
    # id_map: a real gsis for sleeper_id 4046; nothing for 7777 (-> placeholder)
    id_map = pd.DataFrame(
        {"gsis_id": ["00-0036900"], "sleeper_id": ["4046.0"]}  # float-stringified on purpose
    )
    write_partition(tmp_path / "raw", "id_map", id_map, season=None)

    payload = [
        {"player_id": "4046", "player": {"first_name": "JaMarr", "last_name": "Chase", "position": "WR"},
         "stats": {"rec": 6.0, "rec_yd": 78.0}},
        {"player_id": "7777", "player": {"first_name": "Rook", "last_name": "Ie", "position": "RB"},
         "stats": {"rush_yd": 40.0, "rush_td": 0.3}},
    ]
    monkeypatch.setattr(swp, "_fetch_sleeper_weekly", lambda season, week: payload)

    out_path = swp.refresh_sleeper_weekly(tmp_path / "raw", season=2023, week=5)
    assert out_path.exists()

    stored = read_partition(tmp_path / "raw", "sleeper_weekly_projections", season=2023, week=5)
    stored = ExternalProjectionWeeklySchema.validate(stored)
    by_name = stored.set_index("full_name")
    # the float-stringified id_map join must still match -> real gsis
    assert by_name.loc["JaMarr Chase", "gsis_id"] == "00-0036900"
    assert bool(by_name.loc["JaMarr Chase", "is_placeholder_gsis"]) is False
    # rookie with no id_map entry -> placeholder gsis (99-...)
    assert by_name.loc["Rook Ie", "gsis_id"].startswith("99-")
    assert bool(by_name.loc["Rook Ie", "is_placeholder_gsis"]) is True
```

- [ ] **Step 6: Implement fetch + refresh**

```python
# src/projections/ingest/sleeper_weekly_projections.py  (append)
def _fetch_sleeper_weekly(season: int, week: int) -> list[dict[str, Any]]:
    url = _SLEEPER_WEEKLY_URL.format(season=season, week=week)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted host)
            data = json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SleeperWeeklyError(f"Sleeper weekly fetch failed for {season} wk{week}: {exc}") from exc
    if not isinstance(data, list):
        raise SleeperWeeklyError(f"Unexpected Sleeper weekly payload type: {type(data).__name__}")
    return data


def _attach_gsis(df: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    crosswalk = (
        id_map[["gsis_id", "sleeper_id"]].dropna(subset=["sleeper_id"]).drop_duplicates("sleeper_id").copy()
    )
    crosswalk["sleeper_id"] = normalize_join_id(crosswalk["sleeper_id"])
    df = df.copy()
    df["sleeper_id"] = normalize_join_id(df["sleeper_id"])
    merged = df.merge(crosswalk, on="sleeper_id", how="left")
    mask = merged["gsis_id"].isna()
    merged["is_placeholder_gsis"] = mask
    merged["gsis_id"] = merged["gsis_id"].astype("object")
    merged.loc[mask, "gsis_id"] = [
        _make_placeholder_gsis(name, pos)
        for name, pos in zip(
            merged.loc[mask, "full_name"], merged.loc[mask, "position"], strict=True
        )
    ]
    return merged


def refresh_sleeper_weekly(data_root: Path, *, season: int, week: int) -> Path:
    """Fetch, parse, attach gsis, validate, and store one weekly partition."""
    payload = _fetch_sleeper_weekly(season, week)
    parsed = parse_sleeper_weekly(payload, season=season, week=week)
    id_map = read_partition(data_root, "id_map", season=None)
    attached = _attach_gsis(parsed, id_map)
    attached = attached.rename(columns={"sleeper_id": "source_player_id"})
    attached["source"] = ProjectionSource.SLEEPER.value
    # dtype hygiene so concat/validate are stable (no all-NA object inference)
    for field in _STAT_FIELDS:
        attached[field] = attached[field].astype("Float64")
    frame = ExternalProjectionWeeklySchema.validate(attached)
    return write_partition(
        data_root, "sleeper_weekly_projections", frame, season=season, week=week
    )
```

- [ ] **Step 7: Run tests + gates**

Run: `./.venv/Scripts/python -m pytest tests/test_ingest/test_sleeper_weekly_projections.py -v` then `pytest -k "ingest or store or schemas"`.
Expected: PASS. mypy/ruff clean (resolve the placeholder `date` import — drop it if unused).

- [ ] **Step 8: Commit**

```bash
git add src/projections/ingest/sleeper_weekly_projections.py tests/test_ingest/test_sleeper_weekly_projections.py
git commit -m "feat(ingest): Sleeper weekly projection ingest -> ExternalProjectionWeeklySchema"
```

---

### Task 7: Era-aware DraftKings weekly actuals scorer

Produce realized weekly DK-base points per `(gsis_id, season, week, position)`, across the full era-aware regular season (no week-17 cap), keeping position (needed for per-position bucketing).

**Files:**
- Create: `src/projections/dfs/__init__.py` (empty package marker)
- Create: `tests/test_dfs/__init__.py` (empty — package marker so the new test dir matches repo convention, e.g. `tests/test_scripts/__init__.py`)
- Create: `src/projections/dfs/actuals.py`
- Test: `tests/test_dfs/test_actuals.py`

**Interfaces:**
- Consumes: `StatLine`, `score` (`scoring.score`), `dk_actuals_bonus` (Task 4), `last_regular_week` (Task 1), `_PYARROW_STR`.
- Produces: `dk_weekly_actuals(weekly_stats: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame` → columns `["gsis_id","season","week","position","actual_points","actual_points_with_bonus"]`, regular-season weeks only (per-row `week <= last_regular_week(season)`), skill positions only. `actual_points` is DK **base** (no bonus); `actual_points_with_bonus = actual_points + dk_actuals_bonus(yards)` — the target for the §6.2 sensitivity check. (Distinct from `draft/backtest/weekly_actuals.build_weekly_actuals`, which caps at week 17, is bonus-free, and drops `position`.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dfs/test_actuals.py
import pandas as pd
from projections.dfs.actuals import dk_weekly_actuals
from projections.schemas import Ruleset

_COLS = [
    "gsis_id", "season", "week", "position",
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds", "receptions",
    "receiving_yards", "receiving_tds", "fumbles_lost",
]


def _ws(rows):
    return pd.DataFrame(rows, columns=_COLS)


def test_scores_dk_base_and_keeps_position():
    ws = _ws([
        ["00-0036900", 2023, 5, "WR", 0, 0, 0, 0, 0, 6, 78, 1, 0],
    ])
    out = dk_weekly_actuals(ws, ruleset=Ruleset.draftkings())
    row = out.iloc[0]
    # 6*1 + 78/10 + 1*6 = 6 + 7.8 + 6 = 19.8
    assert round(float(row["actual_points"]), 2) == 19.8
    assert row["position"] == "WR"


def test_drops_playoff_weeks_era_aware():
    # 2023 (18-week era): week 18 kept, week 19 dropped
    ws = _ws([
        ["00-0000001", 2023, 18, "RB", 0, 0, 0, 50, 1, 0, 0, 0, 0],
        ["00-0000001", 2023, 19, "RB", 0, 0, 0, 99, 9, 0, 0, 0, 0],
    ])
    out = dk_weekly_actuals(ws, ruleset=Ruleset.draftkings())
    assert out["week"].tolist() == [18]


def test_drops_non_skill_positions():
    ws = _ws([["00-0000002", 2023, 5, "K", 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert dk_weekly_actuals(ws, ruleset=Ruleset.draftkings()).empty


def test_bonus_column_adds_three_at_100_rec_yards():
    ws = _ws([["00-0000003", 2023, 5, "WR", 0, 0, 0, 0, 0, 8, 110, 0, 0]])
    out = dk_weekly_actuals(ws, ruleset=Ruleset.draftkings()).iloc[0]
    # base: 8*1 + 110/10 = 19.0 ; +3 bonus -> 22.0
    assert round(float(out["actual_points"]), 2) == 19.0
    assert round(float(out["actual_points_with_bonus"]), 2) == 22.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_actuals.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/projections/dfs/actuals.py
"""DraftKings-base weekly actual points (era-aware, skill positions).

A sibling of draft/backtest/weekly_actuals.build_weekly_actuals that (a) is
ruleset-parameterized for DK base, (b) uses the era-aware regular-season cutoff
(18 weeks for 2021+, not a hard week-17 cap), and (c) retains `position` for the
edge study's per-position bucketing. All point math routes through scoring.score.
"""

from __future__ import annotations

import pandas as pd

from projections.scoring.score import StatLine, score
from projections.scoring import dk_actuals_bonus
from projections.schemas import Ruleset, _PYARROW_STR
from projections.season_calendar import last_regular_week

_SKILL = {"QB", "RB", "WR", "TE"}
_EMPTY_COLS = ["gsis_id", "season", "week", "position", "actual_points", "actual_points_with_bonus"]


def dk_weekly_actuals(weekly_stats: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame:
    """One row per (gsis_id, season, week, position) of realized DK-base points
    (+ a bonus-inclusive column for the sensitivity check), regular-season weeks
    only, skill positions only."""
    ws = weekly_stats[weekly_stats["position"].isin(_SKILL)].copy()
    if not ws.empty:
        cutoff = ws["season"].map(last_regular_week)
        ws = ws[ws["week"] <= cutoff].copy()

    if ws.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "season": pd.array([], dtype="Int64"),
                "week": pd.array([], dtype="Int64"),
                "position": pd.array([], dtype=_PYARROW_STR),
                "actual_points": pd.array([], dtype="Float64"),
                "actual_points_with_bonus": pd.array([], dtype="Float64"),
            }
        )

    points: list[float] = []
    with_bonus: list[float] = []
    for _, row in ws.iterrows():
        line = StatLine(
            passing_yards=float(row["passing_yards"]),
            passing_tds=int(row["passing_tds"]),
            interceptions=int(row["interceptions"]),
            rushing_yards=float(row["rushing_yards"]),
            rushing_tds=int(row["rushing_tds"]),
            receptions=int(row["receptions"]),
            receiving_yards=float(row["receiving_yards"]),
            receiving_tds=int(row["receiving_tds"]),
            fumbles_lost=int(row["fumbles_lost"]),
        )
        base = score(line, ruleset)
        bonus = dk_actuals_bonus(
            passing_yards=float(row["passing_yards"]),
            rushing_yards=float(row["rushing_yards"]),
            receiving_yards=float(row["receiving_yards"]),
        )
        points.append(base)
        with_bonus.append(base + bonus)

    return pd.DataFrame(
        {
            "gsis_id": ws["gsis_id"].astype(_PYARROW_STR).to_numpy(),
            "season": ws["season"].astype("Int64").to_numpy(),
            "week": ws["week"].astype("Int64").to_numpy(),
            "position": ws["position"].astype(_PYARROW_STR).to_numpy(),
            "actual_points": pd.array(points, dtype="Float64"),
            "actual_points_with_bonus": pd.array(with_bonus, dtype="Float64"),
        }
    )
```

- [ ] **Step 4: Run tests + gates**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_actuals.py -v`. Expected: PASS. mypy/ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/projections/dfs/__init__.py src/projections/dfs/actuals.py tests/test_dfs/test_actuals.py
git commit -m "feat(dfs): era-aware DK-base weekly actuals scorer"
```

---

### Task 8: Home-grown weekly projection emitter (walk-forward) + leakage guard

Emit our model's per-stat means (and DK-base points) for the eval cells, walk-forward, reusing the model-backtest fit/predict path. Add the second-surface leakage test.

**Files:**
- Create: `src/projections/dfs/projections.py`
- Test: `tests/test_dfs/test_projections.py`
- Test: `tests/test_features/test_wr_trajectory_vegas_leakage.py` (the §5.3(b) guard)

**Interfaces:**
- Consumes: `POSITION_DISPATCH`, `read_features`, `read_partition`, `_per_stat_means_from_predictions` (from `projections.backtest.harness`), `Ruleset.draftkings()`, `expected_points`, `Stat`, `Position`.
- Produces: `emit_weekly_projections(*, seasons, positions, train_start=2018, model_class=None, features_root, raw_root, ruleset) -> pd.DataFrame` → columns `["gsis_id","season","week","position", <stat.value per emitted stat>, "our_pts"]` where `our_pts = expected_points(stat_means, ruleset)`. `model_class=None` resolves **per position** to the production model (`dispatch.default_model_class`: WR `ensemble-decomposed`, QB `lightgbm-nb`, RB/TE `baseline`) — the edge study must give our model its real (production) form, not a weakened baseline (spec §5.3 "our home-grown weekly model").

**Note on reuse vs. leakage:** the loop mirrors `harness.run_backtest` (`harness.py:239-272`) but collects per-stat means instead of metrics. `predict_distribution(features, ruleset=DK)` already returns DK-scored `mean`, but because DK base scoring is *linear* in stats, `expected_points(per_stat_means, DK)` equals it exactly and is what the blend (Task 9) also consumes — so we score from the decoded means for consistency. `read_features` lives in `projections.features.cache` (NOT `projections.features` — that exports only the `build_*_features`); `_per_stat_means_from_predictions` is module-level in `projections.backtest.harness`. `.target_stats` is not on the `Model` protocol, so read it via a local structural `Protocol` cast (avoids importing/maintaining the concrete 4-class union the harness casts to).

- [ ] **Step 1: Write the failing test (small, real model on synthetic features is heavy — test the assembly with a stubbed harness path)**

```python
# tests/test_dfs/test_projections.py
import pandas as pd
from projections.dfs import projections as proj
from projections.schemas import Position, Ruleset, Stat


def test_emit_assembles_points_from_stat_means(monkeypatch):
    # Stub the per-(position, year) predict step: return canned per-stat means.
    def fake_one_cell(position, year, *, train_start, model_class, features_root, raw_root, ruleset):
        return pd.DataFrame(
            {
                "gsis_id": ["00-0036900"],
                "season": [year],
                "week": [5],
                "position": [position.value],
                Stat.RECEPTIONS.value: [6.0],
                Stat.RECEIVING_YARDS.value: [78.0],
                Stat.RECEIVING_TDS.value: [0.5],
            }
        )

    monkeypatch.setattr(proj, "_emit_one_cell", fake_one_cell)

    out = proj.emit_weekly_projections(
        seasons=[2023],
        positions=[Position.WR],
        features_root="unused",
        raw_root="unused",
        ruleset=Ruleset.draftkings(),
    )
    row = out.iloc[0]
    # 6*1 + 78/10 + 0.5*6 = 6 + 7.8 + 3 = 16.8
    assert round(float(row["our_pts"]), 2) == 16.8
    assert row["position"] == "WR"
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_projections.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** (the `_emit_one_cell` does the fit/predict/decode; `emit_weekly_projections` loops + scores)

```python
# src/projections/dfs/projections.py
"""Walk-forward home-grown weekly projections for the DFS edge study.

Reuses the model-backtest fit/predict path (backtest.harness) but collects
per-stat predicted means rather than metrics. Scores DK-base points from those
means via the scoring layer (exact: DK base scoring is linear in stats, and the
blend in dfs.blend consumes the same per-stat means).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from projections.backtest.harness import _per_stat_means_from_predictions
from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.schemas import Position, Ruleset, Stat
from projections.scoring import expected_points
from projections.store import read_partition

_STAT_COLS = [s.value for s in Stat]


class _HasTargetStats(Protocol):
    """Structural view of the per-position production models, which all expose
    `target_stats` (not on the base `Model` protocol). Casting to this avoids
    importing/maintaining the concrete 4-class union the harness uses."""

    target_stats: tuple[Stat, ...]


def _emit_one_cell(
    position: Position,
    year: int,
    *,
    train_start: int,
    model_class: str | None,
    features_root: Path,
    raw_root: Path,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Fit on seasons < year, predict all weeks of `year`, return per-stat means.
    `model_class=None` -> the position's production model (`default_model_class`)."""
    dispatch = POSITION_DISPATCH[position]
    resolved_class = model_class or dispatch.default_model_class

    train_seasons = list(range(train_start, year))
    train_features = pd.concat(
        [read_features(position, s, features_root=features_root) for s in train_seasons],
        ignore_index=True,
    )
    train_actuals = pd.concat(
        [read_partition(raw_root, "weekly_stats", season=s) for s in train_seasons],
        ignore_index=True,
    )
    predict_features = read_features(position, year, features_root=features_root)

    model = dispatch.factories[resolved_class]()
    model.fit(train_features, train_actuals)
    predictions = model.predict_distribution(predict_features, ruleset=ruleset)
    target_stats = tuple(cast(_HasTargetStats, model).target_stats)
    means = _per_stat_means_from_predictions(predictions, target_stats=target_stats)
    means["position"] = position.value
    return means


def emit_weekly_projections(
    *,
    seasons: list[int],
    positions: list[Position],
    train_start: int = 2018,
    model_class: str | None = None,
    features_root: Path | str,
    raw_root: Path | str,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Per-(gsis_id, season, week, position) per-stat means + DK-base `our_pts`.
    `model_class=None` uses each position's production model."""
    frames: list[pd.DataFrame] = []
    for position in positions:
        for year in seasons:
            frames.append(
                _emit_one_cell(
                    position,
                    year,
                    train_start=train_start,
                    model_class=model_class,
                    features_root=Path(features_root),
                    raw_root=Path(raw_root),
                    ruleset=ruleset,
                )
            )
    out = pd.concat(frames, ignore_index=True)
    stat_cols = [c for c in out.columns if c in _STAT_COLS]
    out["our_pts"] = out[stat_cols].apply(
        lambda r: expected_points({k: float(r[k]) for k in stat_cols if pd.notna(r[k])}, ruleset),
        axis=1,
    )
    return out
```

- [ ] **Step 4: Run the assembly test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_projections.py -v`. Expected: PASS.

- [ ] **Step 5: Write the second-surface leakage guard (trajectory/vegas/weather)**

Mirror `tests/test_features/test_wr_leakage.py` structure (same fixtures + `_baseline` helper), but inject implausible **future-week** rows into the FULL frames `attach_trajectory_features` / `attach_vegas_team_context_features` consume (`weekly_stats`, `snap_counts`, `schedules`) and assert the produced feature frame is byte-identical, focusing the assertion on the trajectory/vegas/weather columns.

```python
# tests/test_features/test_wr_trajectory_vegas_leakage.py
import pandas as pd
from projections.features.wr import build_wr_features

_SEASON, _AS_OF_WEEK = 2024, 5
_TRAJ_VEGAS_COLS = [
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
    "season_avg_implied_team_total",
    "season_avg_spread",
]


def _build(weekly_stats, snap_counts, schedules, depth_charts, ngs_receiving, fake_pbp_df):
    return build_wr_features(
        weekly_stats=weekly_stats, snap_counts=snap_counts, depth_charts=depth_charts,
        ngs_receiving=ngs_receiving, schedules=schedules, pbp=fake_pbp_df,
        season=_SEASON, as_of_week=_AS_OF_WEEK,
    )


def test_future_weekly_stats_do_not_leak_into_trajectory(
    wr_weekly_stats, wr_snap_counts, wr_schedules, wr_depth_charts, wr_ngs_receiving, fake_pbp_df
):
    base = _build(wr_weekly_stats, wr_snap_counts, wr_schedules, wr_depth_charts, wr_ngs_receiving, fake_pbp_df)
    leaky = wr_weekly_stats.copy()
    fut = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[fut, ["receiving_yards", "targets", "receptions"]] = 999.0
    after = _build(leaky, wr_snap_counts, wr_schedules, wr_depth_charts, wr_ngs_receiving, fake_pbp_df)
    pd.testing.assert_frame_equal(base[_TRAJ_VEGAS_COLS], after[_TRAJ_VEGAS_COLS], check_like=True)


def test_future_schedules_do_not_leak_into_vegas(
    wr_weekly_stats, wr_snap_counts, wr_schedules, wr_depth_charts, wr_ngs_receiving, fake_pbp_df
):
    base = _build(wr_weekly_stats, wr_snap_counts, wr_schedules, wr_depth_charts, wr_ngs_receiving, fake_pbp_df)
    leaky = pd.concat(
        [wr_schedules, wr_schedules.assign(week=6, total_line=999.0, spread_line=999.0)],
        ignore_index=True,
    )
    after = _build(wr_weekly_stats, wr_snap_counts, leaky, wr_depth_charts, wr_ngs_receiving, fake_pbp_df)
    pd.testing.assert_frame_equal(base[_TRAJ_VEGAS_COLS], after[_TRAJ_VEGAS_COLS], check_like=True)
```

> If the fixture column names differ (`total_line`/`spread_line` vs `implied_team_total`), adapt to the actual `wr_schedules` fixture columns — read `tests/test_features/conftest.py` first. If a test FAILS, that is a real leak (§5.3 hard stop) — stop and report, do not weaken the assertion.

- [ ] **Step 6: Run the leakage tests**

Run: `./.venv/Scripts/python -m pytest tests/test_features/test_wr_trajectory_vegas_leakage.py tests/test_features/test_wr_leakage.py -v`
Expected: PASS (confirming no leak). If FAIL → STOP and report.

- [ ] **Step 7: Gates + commit**

Run mypy/ruff. Then:
```bash
git add src/projections/dfs/projections.py tests/test_dfs/test_projections.py tests/test_features/test_wr_trajectory_vegas_leakage.py
git commit -m "feat(dfs): walk-forward weekly projection emitter + trajectory/vegas leakage guard"
```

---

### Task 9: Stat-line-space blend (home-grown + Sleeper)

**Files:**
- Create: `src/projections/dfs/blend.py`
- Test: `tests/test_dfs/test_blend.py`

**Interfaces:**
- Consumes: `expected_points`, the emitter output (per-stat means; Task 8), the Sleeper weekly stat-line frame (`ExternalProjectionWeeklySchema`; Task 6), `Stat`.
- Produces:
  - `sleeper_weekly_points(sleeper: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame` → per `(gsis_id, season, week)` a `sleeper_pts` column = `expected_points(stat line, ruleset)`. **This is the frame `build_universe` (Task 10) consumes as `sleeper_pts`** — it is the source of the Sleeper-only baseline column the headline metric needs.
  - `blend_statlines(ours: pd.DataFrame, sleeper: pd.DataFrame, *, weight_ours: float, ruleset: Ruleset) -> pd.DataFrame` → per `(gsis_id, season, week)` a `blended_pts` column = `expected_points(weighted-mean stat line, ruleset)`. Blend is in **stat-line space** (matches `consensus.blend`). **Role:** the blend at `weight_ours=0.5` is an **exploratory** candidate — Task 11 renames its `blended_pts → our_pts` and runs it through the *same* `build_universe`/metrics as a non-confirmatory variant (per spec §7.3). `weight_ours=1.0` reproduces the emitter's home-grown-only line (a cross-check); `0.0` reproduces `sleeper_weekly_points`. So `blended_pts` is **not** dead — it is the exploratory-variant input.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dfs/test_blend.py
import pandas as pd
from projections.dfs.blend import blend_statlines
from projections.schemas import Ruleset, Stat

_KEY = ["gsis_id", "season", "week"]


def _ours():
    return pd.DataFrame({
        "gsis_id": ["g1"], "season": [2023], "week": [5],
        Stat.RECEPTIONS.value: [4.0], Stat.RECEIVING_YARDS.value: [40.0],
    })


def _sleeper():
    return pd.DataFrame({
        "gsis_id": ["g1"], "season": [2023], "week": [5],
        "receptions": [6.0], "receiving_yards": [80.0],
    })


def test_blend_50_50_in_statline_space():
    out = blend_statlines(_ours(), _sleeper(), weight_ours=0.5, ruleset=Ruleset.draftkings())
    # blended line: rec=5, rec_yd=60 -> 5*1 + 60/10 = 11.0
    assert round(float(out.set_index(_KEY).loc[("g1", 2023, 5), "blended_pts"]), 2) == 11.0


def test_weight_one_is_home_grown_only():
    out = blend_statlines(_ours(), _sleeper(), weight_ours=1.0, ruleset=Ruleset.draftkings())
    # ours: rec=4, rec_yd=40 -> 4 + 4 = 8.0
    assert round(float(out.set_index(_KEY).loc[("g1", 2023, 5), "blended_pts"]), 2) == 8.0


def test_sleeper_weekly_points_scores_statline():
    from projections.dfs.blend import sleeper_weekly_points
    out = sleeper_weekly_points(_sleeper(), ruleset=Ruleset.draftkings())
    # sleeper: rec=6, rec_yd=80 -> 6 + 8 = 14.0
    assert round(float(out.set_index(_KEY).loc[("g1", 2023, 5), "sleeper_pts"]), 2) == 14.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_blend.py -v`. Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/projections/dfs/blend.py
"""Blend home-grown + Sleeper weekly projections in stat-line space.

Matches consensus.blend: average per-stat means (here a weighted average),
assemble one stat line, score once. weight_ours=1.0 -> home-grown-only,
0.0 -> Sleeper-only.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset, Stat
from projections.scoring import expected_points

# canonical stat fields shared by both sources (Sleeper uses these names; our
# emitter uses Stat.value, which are the same strings).
_BLEND_FIELDS = [
    Stat.PASSING_YARDS.value, Stat.PASSING_TDS.value, Stat.INTERCEPTIONS.value,
    Stat.RUSHING_YARDS.value, Stat.RUSHING_TDS.value, Stat.RECEPTIONS.value,
    Stat.RECEIVING_YARDS.value, Stat.RECEIVING_TDS.value, Stat.FUMBLES_LOST.value,
]
_KEY = ["gsis_id", "season", "week"]


def blend_statlines(
    ours: pd.DataFrame, sleeper: pd.DataFrame, *, weight_ours: float, ruleset: Ruleset
) -> pd.DataFrame:
    """Weighted stat-line blend -> one `blended_pts` per (gsis_id, season, week)."""
    o = ours[_KEY + [c for c in _BLEND_FIELDS if c in ours.columns]]
    s = sleeper[_KEY + [c for c in _BLEND_FIELDS if c in sleeper.columns]]
    merged = o.merge(s, on=_KEY, how="inner", suffixes=("_ours", "_slp"))

    pts: list[float] = []
    for _, row in merged.iterrows():
        line: dict[str, float] = {}
        for field in _BLEND_FIELDS:
            ov, sv = row.get(f"{field}_ours"), row.get(f"{field}_slp")
            vals = [(weight_ours, ov), (1.0 - weight_ours, sv)]
            num = sum(w * float(v) for w, v in vals if pd.notna(v))
            wsum = sum(w for w, v in vals if pd.notna(v))
            if wsum > 0:
                line[field] = num / wsum
        pts.append(expected_points(line, ruleset))

    out = merged[_KEY].copy()
    out["blended_pts"] = pts
    return out


def sleeper_weekly_points(sleeper: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame:
    """Score the Sleeper weekly stat-line frame to DK-base points per cell.
    This is the `sleeper_pts` frame Task 10's build_universe consumes."""
    present = [c for c in _BLEND_FIELDS if c in sleeper.columns]
    pts = [
        expected_points({c: float(row[c]) for c in present if pd.notna(row[c])}, ruleset)
        for _, row in sleeper.iterrows()
    ]
    out = sleeper[_KEY].copy()
    out["sleeper_pts"] = pts
    return out
```

> Note: `how="inner"` enforces the "both sources project" paired universe (Task 10 handles inclusion-disagreement separately). When one side is missing a field but not the row, the weighted mean renormalizes over present sources — acceptable for v1; document in the report.

- [ ] **Step 4: Run + gates + commit**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_blend.py -v`; mypy/ruff.
```bash
git add src/projections/dfs/blend.py tests/test_dfs/test_blend.py
git commit -m "feat(dfs): stat-line-space blend of home-grown + Sleeper"
```

---

### Task 10: Metric harness — universe, metrics, clustered bootstrap, pre-registered gate, verdict

The core. Pre-registered constants, the comparable universe (actuals-conditioned), the disagreement head-to-head + ranking skill, the player-season clustered bootstrap (subset re-derived per resample), and the ADOPT/STOP/INCONCLUSIVE verdict with the anti-masking guardrail.

**Files:**
- Create: `src/projections/dfs/config.py` (pre-registered constants)
- Create: `src/projections/dfs/edge_study.py`
- Test: `tests/test_dfs/test_edge_study.py`

**Interfaces:**
- Produces (config): `DELTA: float`, `USAGE_FLOOR_SNAPS: float` (or touches+targets count), `MARGIN_M: float`, `N_MIN_CLUSTERS: int`, `TARGET_CI_HALFWIDTH: float`, `N_BOOTSTRAP: int`, `BOOTSTRAP_SEED: int`.
- Produces (edge_study):
  - `build_universe(ours, sleeper_pts, actuals, *, usage) -> pd.DataFrame` — inner-join on `(gsis_id, season, week)`, position + `actual_points` + `actual_points_with_bonus` from actuals, filtered by the actual-usage floor; columns include `our_pts`, `sleeper_pts`, `actual_points`, `actual_points_with_bonus`, `position`, `player_season` (= `gsis_id+"-"+season`).
  - `head_to_head_fraction(df, *, target_col="actual_points") -> float` — on the `|our_pts - sleeper_pts| > DELTA` subset, share where `|our_pts-target| < |sleeper_pts-target|` (ties dropped). `target_col="actual_points_with_bonus"` drives the §6.2 sensitivity.
  - `clustered_bootstrap_fraction(df, *, seed, target_col="actual_points") -> Interval` — resample distinct `player_season` clusters; **re-derive the disagreement subset inside each resample**; return the `Interval`.
  - `block_bootstrap_by_week(df, *, seed, target_col="actual_points") -> Interval` — resample `(season, week)` blocks (robustness, §7.2.3).
  - `ranking_skill_diff(df) -> float` — pooled Spearman(our,actual) − Spearman(sleeper,actual).
  - `inclusion_disagreement(ours, sleeper_pts, *, usage) -> dict[str,int]` — above-floor one-source-only cell counts (§6.5/§7.1).
  - `coverage_report(universe) -> dict[str,int]` — per-week-bucket cell counts (CLI augments with drop-reason counts).
  - `run_edge_study_from_universe(universe) -> EdgeStudyResult` — primary gate (home-grown-only pooled) + anti-masking + by-week robustness + bonus sensitivity → verdict tier. `EdgeStudyResult` fields: `verdict, primary, byweek, sensitivity, ranking_diff, n_clusters, per_position_fraction, equal_weight_fraction`.

- [ ] **Step 1: Write `config.py`** (committed *before* the verdict; values are placeholders to be finalized from a prior-year distribution in Task 11's calibration step, but fixed in code here)

```python
# src/projections/dfs/config.py
"""Pre-registered constants for the DFS edge study. Fixed BEFORE computing the
verdict; never tuned to the outcome. Calibrated from a prior-year (e.g. 2020)
projection-difference + usage distribution — see the plan's Task 11 calibration.
"""

DELTA: float = 3.0                 # DK-base-point disagreement threshold
USAGE_FLOOR_TOUCHES_TARGETS: int = 3  # actual (carries + targets) floor per cell
MARGIN_M: float = 0.05             # anti-masking: no position below 0.50 - m
N_MIN_CLUSTERS: int = 100          # min player-seasons in the disagreement subset
TARGET_CI_HALFWIDTH: float = 0.05  # else INCONCLUSIVE
N_BOOTSTRAP: int = 2000
BOOTSTRAP_SEED: int = 20260623
```

- [ ] **Step 2: Write the failing tests (synthetic oracles — the metric math must be provable without real data)**

```python
# tests/test_dfs/test_edge_study.py
import numpy as np
import pandas as pd
from projections.dfs import config, edge_study as es


def _frame(our, slp, actual, player_seasons=None, positions=None):
    n = len(our)
    return pd.DataFrame({
        "gsis_id": [f"g{i}" for i in range(n)],
        "season": [2023] * n,
        "week": list(range(1, n + 1)),
        "player_season": player_seasons or [f"g{i}-2023" for i in range(n)],
        "position": positions or ["WR"] * n,
        "our_pts": our, "sleeper_pts": slp,
        "actual_points": actual, "actual_points_with_bonus": actual,
    })


def test_head_to_head_always_closer_is_one():
    # ours always closer to actual; disagreement large
    df = _frame(our=[10, 20, 30], slp=[0, 0, 0], actual=[11, 21, 31])
    assert es.head_to_head_fraction(df) == 1.0


def test_head_to_head_identical_sources_drops_all_ties():
    df = _frame(our=[10, 20], slp=[10, 20], actual=[5, 5])
    # no disagreement cells -> empty subset -> NaN (handled, not div-by-zero)
    assert np.isnan(es.head_to_head_fraction(df))


def test_clustered_bootstrap_wider_than_iid_on_correlated_data():
    # 10 player-seasons x 10 weeks. WITHIN a cluster the same source is closer
    # every week (perfect within-cluster correlation); ACROSS clusters it is 50/50.
    # The i.i.d. cell bootstrap sees ~100 "independent" cells (narrow CI); the
    # clustered bootstrap sees only 10 effective units (wide CI). Pins spec §7.2.3.
    rows = []
    for p in range(10):
        favor_ours = p % 2 == 0
        our, slp = (51.0, 58.0) if favor_ours else (58.0, 51.0)  # |err|=1 vs 8; disagreement=7>delta
        for w in range(10):
            rows.append(dict(player_season=f"p{p}", gsis_id=f"p{p}", season=2023,
                             week=w + 1, position="WR",
                             our_pts=our, sleeper_pts=slp, actual_points=50.0))
    df = pd.DataFrame(rows)

    clustered = es.clustered_bootstrap_fraction(df, seed=1)
    assert 0.0 <= clustered.lo_95 <= clustered.point <= clustered.hi_95 <= 1.0

    # i.i.d. cell bootstrap of the same head-to-head statistic, for comparison.
    closer = ((df["our_pts"] - df["actual_points"]).abs()
              < (df["sleeper_pts"] - df["actual_points"]).abs()).astype(float).to_numpy()
    rngb = np.random.default_rng(1)
    boot = np.array([closer[rngb.integers(0, len(closer), len(closer))].mean()
                     for _ in range(config.N_BOOTSTRAP)])
    iid_halfwidth = float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2
    clustered_halfwidth = (clustered.hi_95 - clustered.lo_95) / 2
    assert clustered_halfwidth > iid_halfwidth


def test_verdict_inconclusive_when_too_few_clusters():
    df = _frame(our=[11], slp=[0], actual=[10])  # 1 cluster << N_MIN
    res = es.run_edge_study_from_universe(df)
    assert res.verdict == "INCONCLUSIVE"


def test_inclusion_disagreement_counts_one_source_cells():
    key = ["gsis_id", "season", "week"]
    usage = pd.DataFrame({
        "gsis_id": ["a", "b", "c"], "season": [2023] * 3, "week": [1, 1, 1],
        "touches_targets": [10, 10, 10],  # all above floor
    })
    ours = pd.DataFrame({"gsis_id": ["a", "b"], "season": [2023, 2023], "week": [1, 1]})
    sleeper = pd.DataFrame({"gsis_id": ["b", "c"], "season": [2023, 2023], "week": [1, 1]})
    out = es.inclusion_disagreement(ours[key], sleeper[key], usage=usage)
    assert out == {"ours_only": 1, "sleeper_only": 1, "both": 1}  # a-only, c-only, b-both
```

- [ ] **Step 3: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_edge_study.py -v`. Expected: FAIL — module not found.

- [ ] **Step 4: Implement `edge_study.py`**

```python
# src/projections/dfs/edge_study.py
"""DFS edge study: comparable universe, disagreement head-to-head + ranking
skill, player-season clustered bootstrap, pre-registered single-primary gate,
ADOPT/STOP/INCONCLUSIVE verdict. See the design spec §6-§7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projections.dfs import config
from projections.draft.assistant._compare import Interval

_KEY = ["gsis_id", "season", "week"]


def _disagreement(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["our_pts"] - df["sleeper_pts"]).abs() > config.DELTA]


def head_to_head_fraction(df: pd.DataFrame, *, target_col: str = "actual_points") -> float:
    """Share of disagreement cells where ours is strictly closer to `target_col`.
    Ties (equidistant) dropped from numerator and denominator. `target_col` is
    `actual_points` (base) for the primary metric, `actual_points_with_bonus`
    for the §6.2 sensitivity check."""
    sub = _disagreement(df)
    our_err = (sub["our_pts"] - sub[target_col]).abs()
    slp_err = (sub["sleeper_pts"] - sub[target_col]).abs()
    decisive = our_err != slp_err
    n = int(decisive.sum())
    if n == 0:
        return float("nan")
    return float((our_err[decisive] < slp_err[decisive]).sum()) / n


def _bootstrap_over(
    df: pd.DataFrame, group_cols: list[str], *, seed: int, target_col: str
) -> Interval:
    """Percentile bootstrap of head_to_head_fraction, resampling whole groups
    (clusters) with replacement; the disagreement subset is re-derived inside
    each resample so threshold-boundary uncertainty propagates."""
    groups = [g for _, g in df.groupby(group_cols)]
    rng = np.random.default_rng(seed)
    n = len(groups)
    boot = np.empty(config.N_BOOTSTRAP, dtype=np.float64)
    for b in range(config.N_BOOTSTRAP):
        pick = rng.integers(0, n, size=n)
        resampled = pd.concat([groups[i] for i in pick], ignore_index=True)
        boot[b] = head_to_head_fraction(resampled, target_col=target_col)
    boot = boot[~np.isnan(boot)]
    point = head_to_head_fraction(df, target_col=target_col)
    if boot.size == 0:
        return Interval(point=point, lo_95=float("nan"), hi_95=float("nan"))
    lo, hi = np.percentile(boot, (2.5, 97.5))
    return Interval(point=float(point), lo_95=float(lo), hi_95=float(hi))


def clustered_bootstrap_fraction(
    df: pd.DataFrame, *, seed: int, target_col: str = "actual_points"
) -> Interval:
    """Primary CI: resample player-season clusters (same-player serial corr)."""
    return _bootstrap_over(df, ["player_season"], seed=seed, target_col=target_col)


def block_bootstrap_by_week(
    df: pd.DataFrame, *, seed: int, target_col: str = "actual_points"
) -> Interval:
    """Robustness CI: resample (season, week) blocks (cross-player same-game
    corr — an orthogonal source to player-season clustering, spec §7.2.3)."""
    return _bootstrap_over(df, ["season", "week"], seed=seed, target_col=target_col)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def ranking_skill_diff(df: pd.DataFrame) -> float:
    """Pooled Spearman(our, actual) - Spearman(sleeper, actual)."""
    return _spearman(df["our_pts"], df["actual_points"]) - _spearman(
        df["sleeper_pts"], df["actual_points"]
    )


def inclusion_disagreement(
    ours: pd.DataFrame, sleeper_pts: pd.DataFrame, *, usage: pd.DataFrame
) -> dict[str, int]:
    """Above the actual-usage floor, count cells only ONE source projects
    (the inclusion-disagreement diagnostic, spec §6.5/§7.1 — reported, not in
    the paired test)."""
    floor = usage[usage["touches_targets"].fillna(0) >= config.USAGE_FLOOR_TOUCHES_TARGETS]
    floor = floor[_KEY].drop_duplicates()
    o = set(map(tuple, floor.merge(ours[_KEY].drop_duplicates(), on=_KEY).to_numpy()))
    s = set(map(tuple, floor.merge(sleeper_pts[_KEY].drop_duplicates(), on=_KEY).to_numpy()))
    return {"ours_only": len(o - s), "sleeper_only": len(s - o), "both": len(o & s)}


def coverage_report(universe: pd.DataFrame) -> dict[str, int]:
    """Per-week-bucket cell counts of the final universe (spec §7.1/§5.3). The
    CLI augments this with drop-reason counts (placeholder-gsis, cold-start,
    usage floor) it computes from the raw frames."""
    def bucket(w: int) -> str:
        return "wk1_3" if w <= 3 else ("wk4_13" if w <= 13 else "wk14_18")

    counts = {"universe_cells": int(len(universe))}
    tagged = universe.assign(_b=universe["week"].map(bucket))
    for b, g in tagged.groupby("_b"):
        counts[f"universe_{b}"] = int(len(g))
    return counts


@dataclass(frozen=True)
class EdgeStudyResult:
    verdict: str  # "ADOPT" | "STOP" | "INCONCLUSIVE"
    primary: Interval
    byweek: Interval        # robustness (block-by-week) bootstrap
    sensitivity: Interval   # primary metric vs bonus-inclusive actuals
    ranking_diff: float
    n_clusters: int
    per_position_fraction: dict[str, float]
    equal_weight_fraction: float


def run_edge_study_from_universe(universe: pd.DataFrame) -> EdgeStudyResult:
    """Compute the pre-registered primary gate + robustness/sensitivity + verdict
    (home-grown-only vs Sleeper, pooled)."""
    sub = _disagreement(universe)
    n_clusters = int(sub["player_season"].nunique())
    primary = clustered_bootstrap_fraction(universe, seed=config.BOOTSTRAP_SEED)
    byweek = block_bootstrap_by_week(universe, seed=config.BOOTSTRAP_SEED)
    sensitivity = clustered_bootstrap_fraction(
        universe, seed=config.BOOTSTRAP_SEED, target_col="actual_points_with_bonus"
    )
    ranking_diff = ranking_skill_diff(universe)

    per_pos = {pos: head_to_head_fraction(g) for pos, g in universe.groupby("position")}
    finite = [v for v in per_pos.values() if not np.isnan(v)]
    equal_weight = float(np.mean(finite)) if finite else float("nan")

    half_width = (primary.hi_95 - primary.lo_95) / 2 if not np.isnan(primary.lo_95) else float("inf")
    underpowered = n_clusters < config.N_MIN_CLUSTERS or half_width > config.TARGET_CI_HALFWIDTH

    edge_primary = (
        primary.lo_95 > 0.50
        and not (np.isnan(ranking_diff) or ranking_diff < 0)
        and all((np.isnan(v) or v >= 0.50 - config.MARGIN_M) for v in per_pos.values())
    )
    robust = byweek.lo_95 > 0.50          # by-week agrees on the edge
    sens_holds = sensitivity.lo_95 > 0.50  # bonus sensitivity does not flip

    if underpowered:
        verdict = "INCONCLUSIVE"
    elif edge_primary and robust and sens_holds:
        verdict = "ADOPT"
    elif edge_primary:
        verdict = "INCONCLUSIVE"  # primary edge but robustness/sensitivity disagree
    else:
        verdict = "STOP"

    return EdgeStudyResult(
        verdict=verdict, primary=primary, byweek=byweek, sensitivity=sensitivity,
        ranking_diff=ranking_diff, n_clusters=n_clusters,
        per_position_fraction=per_pos, equal_weight_fraction=equal_weight,
    )


def build_universe(
    ours: pd.DataFrame, sleeper_pts: pd.DataFrame, actuals: pd.DataFrame, *, usage: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join ours+sleeper+actuals on (gsis_id, season, week); position +
    actual_points (+ bonus-inclusive) from `actuals`; filter by the actual-usage
    floor in `usage` (columns gsis_id, season, week, touches_targets)."""
    df = (
        ours[_KEY + ["our_pts"]]
        .merge(sleeper_pts[_KEY + ["sleeper_pts"]], on=_KEY, how="inner")
        .merge(
            actuals[_KEY + ["position", "actual_points", "actual_points_with_bonus"]],
            on=_KEY,
            how="inner",
        )
        .merge(usage[_KEY + ["touches_targets"]], on=_KEY, how="left")
    )
    df = df[df["touches_targets"].fillna(0) >= config.USAGE_FLOOR_TOUCHES_TARGETS].copy()
    df["player_season"] = df["gsis_id"].astype(str) + "-" + df["season"].astype(str)
    return df
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_edge_study.py -v`. Expected: PASS (add `scipy` is already a dependency).

- [ ] **Step 6: Gates + commit**

```bash
git add src/projections/dfs/config.py src/projections/dfs/edge_study.py tests/test_dfs/test_edge_study.py
git commit -m "feat(dfs): edge-study metrics, clustered bootstrap, pre-registered gate + verdict"
```

---

### Task 11: Usage frame, `run_study` orchestrator, CLI, smoke

Wire ingest → projections → actuals → universe → verdict into a tested orchestrator + thin CLI.

**Files:**
- Create: `src/projections/dfs/usage.py`
- Create: `src/projections/dfs/run.py` (the orchestrator + report writer — keeps the script thin)
- Create: `scripts/dfs_edge_study.py` (argparse only)
- Test: `tests/test_dfs/test_usage.py`
- Test: `tests/test_dfs/test_run_smoke.py`

**Interfaces:**
- Produces: `build_usage(weekly_stats) -> pd.DataFrame` → `(gsis_id, season, week, touches_targets)`, `touches_targets = carries + targets` (both columns exist in `weekly_stats` — verified).
- Produces: `run_study(*, seasons, positions, data_root, features_root, ruleset) -> StudyOutput` (dataclass: `primary: EdgeStudyResult`, `exploratory_blend: EdgeStudyResult`, `inclusion: dict`, `coverage: dict`). Wires every Task 6–10 function.
- Produces: `write_report(path, out: StudyOutput, *, seasons) -> None`.
- Produces (CLI): subcommands `ingest-sleeper --seasons 2021-2024`, `calibrate --prior-season 2020` (prints the empirical δ / usage-floor / cluster-count distributions — does NOT auto-write `config.py`), `study --seasons 2021-2024 --out reports/...`.

- [ ] **Step 1: `build_usage` — failing test**

```python
# tests/test_dfs/test_usage.py
import pandas as pd
from projections.dfs.usage import build_usage


def test_touches_targets_is_carries_plus_targets():
    ws = pd.DataFrame({
        "gsis_id": ["a", "b"], "season": [2023, 2023], "week": [1, 1],
        "carries": [10, 0], "targets": [2, 7],
    })
    out = build_usage(ws).set_index("gsis_id")
    assert float(out.loc["a", "touches_targets"]) == 12.0
    assert float(out.loc["b", "touches_targets"]) == 7.0
```

- [ ] **Step 2: Run to verify it fails, then implement `usage.py`**

```python
# src/projections/dfs/usage.py
"""Actual-usage frame for the edge-study universe floor (spec §6.5)."""

from __future__ import annotations

import pandas as pd


def build_usage(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """(gsis_id, season, week, touches_targets) where touches_targets = carries
    + targets (actual usage — never a projection, to avoid endogenous selection)."""
    out = weekly_stats[["gsis_id", "season", "week"]].copy()
    out["touches_targets"] = (
        weekly_stats["carries"].fillna(0) + weekly_stats["targets"].fillna(0)
    ).astype("Float64")
    return out
```

Run: `./.venv/Scripts/python -m pytest tests/test_dfs/test_usage.py -v` → PASS.

- [ ] **Step 3: Implement `run.py`** (the orchestrator + report writer)

```python
# src/projections/dfs/run.py
"""Edge-study orchestrator + report writer. The CLI is a thin wrapper over this."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from projections.dfs.actuals import dk_weekly_actuals
from projections.dfs.blend import blend_statlines, sleeper_weekly_points
from projections.dfs.edge_study import (
    EdgeStudyResult,
    build_universe,
    coverage_report,
    inclusion_disagreement,
    run_edge_study_from_universe,
)
from projections.dfs.projections import emit_weekly_projections
from projections.dfs.usage import build_usage
from projections.schemas import Position, Ruleset
from projections.season_calendar import last_regular_week
from projections.store import read_partition


@dataclass(frozen=True)
class StudyOutput:
    primary: EdgeStudyResult
    exploratory_blend: EdgeStudyResult
    inclusion: dict[str, int]
    coverage: dict[str, int]


def _load_sleeper(data_root: Path, seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        for week in range(1, last_regular_week(season) + 1):
            try:
                frames.append(
                    read_partition(
                        data_root / "raw", "sleeper_weekly_projections",
                        season=season, week=week,
                    )
                )
            except FileNotFoundError:
                continue
    if not frames:
        raise FileNotFoundError("no Sleeper weekly partitions found; run ingest-sleeper first")
    return pd.concat(frames, ignore_index=True)


def run_study(
    *, seasons: list[int], positions: list[Position], data_root: Path,
    features_root: Path, ruleset: Ruleset,
) -> StudyOutput:
    ours = emit_weekly_projections(
        seasons=seasons, positions=positions,
        features_root=features_root, raw_root=data_root / "raw", ruleset=ruleset,
    )
    sleeper_raw = _load_sleeper(data_root, seasons)
    sleeper_pts = sleeper_weekly_points(sleeper_raw, ruleset=ruleset)

    raw_actuals = pd.concat(
        [read_partition(data_root / "raw", "weekly_stats", season=s) for s in seasons],
        ignore_index=True,
    )
    actuals = dk_weekly_actuals(raw_actuals, ruleset=ruleset)
    usage = build_usage(raw_actuals)

    universe = build_universe(ours, sleeper_pts, actuals, usage=usage)
    primary = run_edge_study_from_universe(universe)

    blend = blend_statlines(ours, sleeper_raw, weight_ours=0.5, ruleset=ruleset)
    blend_universe = build_universe(
        blend.rename(columns={"blended_pts": "our_pts"}), sleeper_pts, actuals, usage=usage
    )
    exploratory = run_edge_study_from_universe(blend_universe)

    return StudyOutput(
        primary=primary, exploratory_blend=exploratory,
        inclusion=inclusion_disagreement(ours, sleeper_pts, usage=usage),
        coverage=coverage_report(universe),
    )


def write_report(path: Path, out: StudyOutput, *, seasons: list[int]) -> None:
    p = out.primary
    lines = [
        f"# DFS Projection Edge Study — verdict ({'-'.join(map(str, (min(seasons), max(seasons))))})",
        "",
        f"**VERDICT: {p.verdict}**",
        "",
        "## Primary test (home-grown-only vs Sleeper, pooled, DK base)",
        f"- head-to-head fraction: {p.primary.point:.3f} "
        f"(95% CI {p.primary.lo_95:.3f}–{p.primary.hi_95:.3f}), clustered by player-season",
        f"- by-week robustness CI: {p.byweek.lo_95:.3f}–{p.byweek.hi_95:.3f}",
        f"- bonus-sensitivity CI (actuals+bonus): {p.sensitivity.lo_95:.3f}–{p.sensitivity.hi_95:.3f}",
        f"- ranking-skill diff (Spearman, ours−Sleeper): {p.ranking_diff:.3f}",
        f"- disagreement clusters (player-seasons): {p.n_clusters}",
        f"- pooled (count-weighted) {p.primary.point:.3f} vs equal-weight {p.equal_weight_fraction:.3f}",
        "",
        "## Per-position (EXPLORATORY — non-confirmatory)",
        *[f"- {pos}: {frac:.3f}" for pos, frac in sorted(p.per_position_fraction.items())],
        "",
        "## Exploratory 50/50 blend (non-confirmatory)",
        f"- verdict {out.exploratory_blend.verdict}; "
        f"fraction {out.exploratory_blend.primary.point:.3f} "
        f"({out.exploratory_blend.primary.lo_95:.3f}–{out.exploratory_blend.primary.hi_95:.3f})",
        "",
        "## Coverage & inclusion disagreement",
        f"- inclusion: {out.inclusion}",
        f"- coverage: {out.coverage}",
        "",
        "## Limitations",
        "- Sleeper-alone is a softer proxy than the true DFS field (necessary, not "
        "sufficient — spec §4.3/§6.1). Bonuses excluded from the projection comparison "
        "(conservative; spec §6.2).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Implement `scripts/dfs_edge_study.py`** (argparse only — three subcommands delegating to `dfs/`)

```python
# scripts/dfs_edge_study.py
"""CLI for the DFS projection edge study. All logic lives in projections.dfs."""

from __future__ import annotations

import argparse
from pathlib import Path

from projections.dfs.run import run_study, write_report
from projections.ingest.sleeper_weekly_projections import refresh_sleeper_weekly
from projections.schemas import Position, Ruleset
from projections.season_calendar import last_regular_week


def _seasons(arg: str) -> list[int]:
    lo, hi = (int(x) for x in arg.split("-")) if "-" in arg else (int(arg), int(arg))
    return list(range(lo, hi + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="DFS projection edge study")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest-sleeper")
    ing.add_argument("--seasons", type=_seasons, required=True)
    ing.add_argument("--data-root", type=Path, default=Path("data"))

    cal = sub.add_parser("calibrate")
    cal.add_argument("--prior-season", type=int, default=2020)
    cal.add_argument("--data-root", type=Path, default=Path("data"))

    stu = sub.add_parser("study")
    stu.add_argument("--seasons", type=_seasons, required=True)
    stu.add_argument("--out", type=Path, required=True)
    stu.add_argument("--data-root", type=Path, default=Path("data"))
    stu.add_argument("--features-root", type=Path, default=Path("data/features"))

    args = parser.parse_args()
    positions = [Position.QB, Position.RB, Position.WR, Position.TE]

    if args.cmd == "ingest-sleeper":
        for season in args.seasons:
            for week in range(1, last_regular_week(season) + 1):
                refresh_sleeper_weekly(args.data_root / "raw", season=season, week=week)
    elif args.cmd == "calibrate":
        # Print empirical |our-sleeper| / usage / cluster-count distributions so the
        # committed config.py constants are justified; do NOT auto-write them.
        from projections.dfs.run import run_study  # local import to keep startup light
        out = run_study(
            seasons=[args.prior_season], positions=positions,
            data_root=args.data_root, features_root=args.features_root,
            ruleset=Ruleset.draftkings(),
        )
        print("prior-season diagnostics:", out.coverage, out.inclusion,
              "n_clusters=", out.primary.n_clusters)
    elif args.cmd == "study":
        out = run_study(
            seasons=args.seasons, positions=positions,
            data_root=args.data_root, features_root=args.features_root,
            ruleset=Ruleset.draftkings(),
        )
        write_report(args.out, out, seasons=args.seasons)
        print(f"verdict: {out.primary.verdict} -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke test** (skips cleanly when real partitions are absent, so data-less CI passes)

```python
# tests/test_dfs/test_run_smoke.py
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("data/features/wr/season=2023").exists()
    or not Path("data/raw/sleeper_weekly_projections/season=2023").exists(),
    reason="requires built feature/raw + ingested Sleeper partitions",
)


def test_one_season_end_to_end():
    from projections.dfs.run import run_study
    from projections.schemas import Position, Ruleset

    out = run_study(
        seasons=[2023], positions=[Position.WR], data_root=Path("data"),
        features_root=Path("data/features"), ruleset=Ruleset.draftkings(),
    )
    assert out.primary.verdict in {"ADOPT", "STOP", "INCONCLUSIVE"}
```

- [ ] **Step 6: Run available tests + gates; commit**

Run: `./.venv/Scripts/python -m pytest tests/test_dfs -v` (smoke skips without data); mypy/ruff clean.
```bash
git add src/projections/dfs/usage.py src/projections/dfs/run.py scripts/dfs_edge_study.py tests/test_dfs/test_usage.py tests/test_dfs/test_run_smoke.py
git commit -m "feat(dfs): run_study orchestrator + edge-study CLI + usage frame + smoke"
```

---

### Task 12: Run the study, write the verdict report, update docs

**Files:**
- Create: `reports/dfs_projection_edge_2026-06-23.md` (the verdict)
- Modify: `TODO.md` (close #39, record the verdict), `project_management.md` (top entry)

- [ ] **Step 1: Calibrate** — run `python scripts/dfs_edge_study.py calibrate --prior-season 2020`, record the empirical δ / usage-floor / cluster-count, and confirm (or adjust, with justification) the `config.py` constants. Commit any change to `config.py` as a `chore(dfs): pre-register study constants` commit *before* running the study.

- [ ] **Step 2: Ingest** — `python scripts/dfs_edge_study.py ingest-sleeper --seasons 2021-2024` (loops every regular-season week per season internally; requires network, ~9k rows/week).

- [ ] **Step 3: Run** — `python scripts/dfs_edge_study.py study --seasons 2021-2024 --out reports/dfs_projection_edge_2026-06-23.md`. The report must include: the primary verdict (ADOPT/STOP/INCONCLUSIVE) with the CI; the per-position exploratory table (labeled non-confirmatory); the equal-weight vs count-weighted pooled fractions; the §6.2 actuals-with-bonus sensitivity result; the coverage accounting (placeholder-gsis drops, rookie/cold-start drops, usage-floor drops, per-week-bucket counts); and the §4.3/§6.1 limitations (Sleeper-as-soft-proxy).

- [ ] **Step 4: Update docs** — In `TODO.md`, mark #39 closed and add a short DFS Layer 1 result line. In `project_management.md`, add a top entry summarizing the verdict and the next-layer recommendation (proceed to optimizer/contest backtest only on ADOPT).

- [ ] **Step 5: Commit**

```bash
git add reports/dfs_projection_edge_2026-06-23.md TODO.md project_management.md
git commit -m "report(dfs): Layer 1 projection edge study verdict + close TODO #39"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** §5.1 ingest → T5/T6; §5.2 DK ruleset + allowlist + bonus helper → T3/T4; §5.3 projection emitter + two-surface leakage → T8; §5.4 era-aware actuals → T1/T7; §5.5 blend → T9; §5.6 metric harness → T10; §6.2 sensitivity → T4+T12; §7.1 universe/coverage → T10/T11; §7.2 metrics → T10; §7.2.3 clustered bootstrap → T10; §7.3 pre-registered gate + anti-masking → T10; §7.4 verdict tiers → T10; §8 schema/allowlist/week-bound → T3/T5/T7; §9 tests → every task; §10 deliverables → T6–T12; H-2 id-join lift → T2. No spec section is unmapped.

**Placeholder scan:** Task 11 is now fully coded (`build_usage`, `run_study`, `write_report`, the argparse CLI, the smoke test all have complete bodies). Task 12 steps 1-3 are operational *runs* (calibrate/ingest/study) — inherently runtime actions invoking the specified CLI, not code placeholders. The §7 diagnostics the spec requires (inclusion-disagreement, coverage, by-week robustness bootstrap, §6.2 bonus sensitivity) are tested functions in `edge_study.py` and wired through `run_study`/`write_report`. No "TODO/similar-to/add-error-handling" placeholders remain.

**Type consistency:** stat columns use `Stat.value` strings consistently across emitter (T8), blend (T9), and universe (T10); `Interval` reused from `_compare.py`; `our_pts`/`sleeper_pts`/`blended_pts`/`actual_points`/`player_season` names consistent across T8–T10.
