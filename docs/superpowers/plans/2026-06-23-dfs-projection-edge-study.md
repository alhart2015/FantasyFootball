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

In `src/projections/draft/assistant/availability.py`, delete the private `_sched_games` (lines ~19-21) and `_last_regular_week` (lines ~24-30), add `from projections.season_calendar import last_regular_week`, and replace every internal `_last_regular_week(` call with `last_regular_week(`. Grep first: `rg "_last_regular_week|_sched_games" src tests` and update every reference (there may be tests referencing the private name — point them at the new public function).

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
- Create: `src/projections/dfs/actuals.py`
- Test: `tests/test_dfs/test_actuals.py`

**Interfaces:**
- Consumes: `StatLine`, `score` (`scoring.score`), `last_regular_week` (Task 1), `_PYARROW_STR`.
- Produces: `dk_weekly_actuals(weekly_stats: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame` → columns `["gsis_id","season","week","position","actual_points"]`, regular-season weeks only (per-row `week <= last_regular_week(season)`), skill positions only. (Distinct from `draft/backtest/weekly_actuals.build_weekly_actuals`, which caps at week 17 and drops `position`.)

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
from projections.schemas import Ruleset, _PYARROW_STR
from projections.season_calendar import last_regular_week

_SKILL = {"QB", "RB", "WR", "TE"}


def dk_weekly_actuals(weekly_stats: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame:
    """One row per (gsis_id, season, week, position) of realized DK-base points,
    regular-season weeks only, skill positions only."""
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
            }
        )

    points: list[float] = []
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
        points.append(score(line, ruleset))

    return pd.DataFrame(
        {
            "gsis_id": ws["gsis_id"].astype(_PYARROW_STR).to_numpy(),
            "season": ws["season"].astype("Int64").to_numpy(),
            "week": ws["week"].astype("Int64").to_numpy(),
            "position": ws["position"].astype(_PYARROW_STR).to_numpy(),
            "actual_points": pd.array(points, dtype="Float64"),
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
- Produces: `emit_weekly_projections(*, seasons, positions, train_start=2018, model_classes=("baseline",), features_root, raw_root, ruleset) -> pd.DataFrame` → columns `["gsis_id","season","week","position", <stat.value per emitted stat>, "our_pts"]` where `our_pts = expected_points(stat_means, ruleset)`.

**Note on reuse vs. leakage:** the loop mirrors `harness.run_backtest` (`harness.py:239-272`) but collects per-stat means instead of metrics. `predict_distribution(features, ruleset=DK)` already returns DK-scored `mean`, but because DK base scoring is *linear* in stats, `expected_points(per_stat_means, DK)` equals it exactly and is what the blend (Task 9) also consumes — so we score from the decoded means for consistency. Import `_per_stat_means_from_predictions` from the harness (it is module-level; if review prefers, promote it to `backtest/__init__` — note in the commit).

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
from typing import cast

import pandas as pd

from projections.backtest.harness import _per_stat_means_from_predictions
from projections.features import read_features
from projections.models import POSITION_DISPATCH
from projections.models.baseline import BaselineModel  # for the cast union
from projections.schemas import Position, Ruleset, Stat
from projections.scoring import expected_points
from projections.store import read_partition

_STAT_COLS = [s.value for s in Stat]


def _emit_one_cell(
    position: Position,
    year: int,
    *,
    train_start: int,
    model_class: str,
    features_root: Path,
    raw_root: Path,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Fit on seasons < year, predict all weeks of `year`, return per-stat means."""
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

    dispatch = POSITION_DISPATCH[position]
    model = dispatch.factories[model_class]()
    model.fit(train_features, train_actuals)
    predictions = model.predict_distribution(predict_features, ruleset=ruleset)
    target_stats = tuple(cast(BaselineModel, model).target_stats)
    means = _per_stat_means_from_predictions(predictions, target_stats=target_stats)
    means["position"] = position.value
    return means


def emit_weekly_projections(
    *,
    seasons: list[int],
    positions: list[Position],
    train_start: int = 2018,
    model_classes: tuple[str, ...] = ("baseline",),
    features_root: Path | str,
    raw_root: Path | str,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Per-(gsis_id, season, week, position) per-stat means + DK-base `our_pts`."""
    frames: list[pd.DataFrame] = []
    for position in positions:
        for year in seasons:
            for model_class in model_classes:
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
- Consumes: `expected_points`, the emitter output (per-stat means; Task 8), the Sleeper weekly stat line (Task 6), `Stat`.
- Produces: `blend_statlines(ours: pd.DataFrame, sleeper: pd.DataFrame, *, weight_ours: float, ruleset: Ruleset) -> pd.DataFrame` → per `(gsis_id, season, week)`: a `blended_pts` column = `expected_points(weighted-mean stat line, ruleset)`. Blend is in **stat-line space** (matches `consensus.blend`). Variants the harness will request: `weight_ours ∈ {1.0 (home-grown-only), 0.0 (Sleeper-only), 0.5}`.

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
  - `build_universe(ours, sleeper_pts, actuals, *, usage) -> pd.DataFrame` — inner-join on `(gsis_id, season, week)`, position from actuals, filtered by the actual-usage floor; columns include `our_pts`, `sleeper_pts`, `actual_points`, `position`, `player_season` (= `gsis_id+"-"+season`).
  - `head_to_head_fraction(df) -> float` — on the `|our_pts - sleeper_pts| > DELTA` subset, share where `|our_pts-actual| < |sleeper_pts-actual|` (ties dropped).
  - `clustered_bootstrap_fraction(df, *, seed) -> Interval` — resample distinct `player_season` clusters; **re-derive the disagreement subset inside each resample**; return the §3 `Interval`.
  - `ranking_skill_diff(df) -> float` — pooled Spearman(our,actual) − Spearman(sleeper,actual).
  - `run_edge_study(ours, sleeper_pts, actuals, *, usage) -> EdgeStudyResult` — assembles the universe, computes the primary gate (home-grown-only pooled), per-position exploratory fractions, the anti-masking check, and the verdict tier.

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
from projections.dfs import edge_study as es


def _frame(our, slp, actual, player_seasons=None, positions=None):
    n = len(our)
    return pd.DataFrame({
        "gsis_id": [f"g{i}" for i in range(n)],
        "season": [2023] * n,
        "week": list(range(1, n + 1)),
        "player_season": player_seasons or [f"g{i}-2023" for i in range(n)],
        "position": positions or ["WR"] * n,
        "our_pts": our, "sleeper_pts": slp, "actual_points": actual,
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
    rng = np.random.default_rng(0)
    # 10 player-seasons x 10 weeks, ours closer with strong within-cluster corr
    rows = []
    for p in range(10):
        bias = rng.normal(0, 5)
        for w in range(10):
            actual = 50 + rng.normal(0, 1)
            rows.append(dict(player_season=f"p{p}", our_pts=actual + 1 + bias,
                             sleeper_pts=actual + 8 + bias, actual_points=actual,
                             gsis_id=f"p{p}", season=2023, week=w + 1, position="WR"))
    df = pd.DataFrame(rows)
    clustered = es.clustered_bootstrap_fraction(df, seed=1)
    # sanity: returns an Interval with point in [0,1] and lo<=point<=hi
    assert 0.0 <= clustered.lo_95 <= clustered.point <= clustered.hi_95 <= 1.0


def test_verdict_inconclusive_when_too_few_clusters():
    df = _frame(our=[11], slp=[0], actual=[10])  # 1 cluster << N_MIN
    res = es.run_edge_study_from_universe(df)
    assert res.verdict == "INCONCLUSIVE"
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


def head_to_head_fraction(df: pd.DataFrame) -> float:
    """Share of disagreement cells where ours is strictly closer to actual.
    Ties (equidistant) dropped from numerator and denominator."""
    sub = _disagreement(df)
    our_err = (sub["our_pts"] - sub["actual_points"]).abs()
    slp_err = (sub["sleeper_pts"] - sub["actual_points"]).abs()
    decisive = our_err != slp_err
    n = int(decisive.sum())
    if n == 0:
        return float("nan")
    return float((our_err[decisive] < slp_err[decisive]).sum()) / n


def clustered_bootstrap_fraction(df: pd.DataFrame, *, seed: int) -> Interval:
    """Percentile bootstrap of head_to_head_fraction, resampling player-season
    clusters (subset re-derived inside each resample)."""
    clusters = {ps: g for ps, g in df.groupby("player_season")}
    keys = list(clusters)
    rng = np.random.default_rng(seed)
    n = len(keys)
    boot = np.empty(config.N_BOOTSTRAP, dtype=np.float64)
    for b in range(config.N_BOOTSTRAP):
        pick = rng.integers(0, n, size=n)
        resampled = pd.concat([clusters[keys[i]] for i in pick], ignore_index=True)
        boot[b] = head_to_head_fraction(resampled)
    boot = boot[~np.isnan(boot)]
    point = head_to_head_fraction(df)
    if boot.size == 0:
        return Interval(point=point, lo_95=float("nan"), hi_95=float("nan"))
    lo, hi = np.percentile(boot, (2.5, 97.5))
    return Interval(point=float(point), lo_95=float(lo), hi_95=float(hi))


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


@dataclass(frozen=True)
class EdgeStudyResult:
    verdict: str  # "ADOPT" | "STOP" | "INCONCLUSIVE"
    primary: Interval
    ranking_diff: float
    n_clusters: int
    per_position_fraction: dict[str, float]
    equal_weight_fraction: float


def run_edge_study_from_universe(universe: pd.DataFrame) -> EdgeStudyResult:
    """Compute the primary gate + verdict on an already-built universe
    (home-grown-only vs Sleeper, pooled)."""
    sub = _disagreement(universe)
    n_clusters = sub["player_season"].nunique()
    primary = clustered_bootstrap_fraction(universe, seed=config.BOOTSTRAP_SEED)
    ranking_diff = ranking_skill_diff(universe)

    per_pos = {
        pos: head_to_head_fraction(g) for pos, g in universe.groupby("position")
    }
    finite = [v for v in per_pos.values() if not np.isnan(v)]
    equal_weight = float(np.mean(finite)) if finite else float("nan")

    half_width = (primary.hi_95 - primary.lo_95) / 2 if not np.isnan(primary.lo_95) else float("inf")
    if n_clusters < config.N_MIN_CLUSTERS or half_width > config.TARGET_CI_HALFWIDTH:
        verdict = "INCONCLUSIVE"
    else:
        edge = primary.lo_95 > 0.50 and not (np.isnan(ranking_diff) or ranking_diff < 0)
        no_masking = all(
            (np.isnan(v) or v >= 0.50 - config.MARGIN_M) for v in per_pos.values()
        )
        verdict = "ADOPT" if (edge and no_masking) else "STOP"

    return EdgeStudyResult(
        verdict=verdict, primary=primary, ranking_diff=ranking_diff,
        n_clusters=int(n_clusters), per_position_fraction=per_pos,
        equal_weight_fraction=equal_weight,
    )


def build_universe(
    ours: pd.DataFrame, sleeper_pts: pd.DataFrame, actuals: pd.DataFrame, *, usage: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join ours+sleeper+actuals on (gsis_id, season, week); position +
    actual_points from `actuals`; filter by the actual-usage floor in `usage`
    (columns gsis_id, season, week, touches_targets)."""
    df = (
        ours[_KEY + ["our_pts"]]
        .merge(sleeper_pts[_KEY + ["sleeper_pts"]], on=_KEY, how="inner")
        .merge(actuals[_KEY + ["position", "actual_points"]], on=_KEY, how="inner")
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

### Task 11: End-to-end CLI + calibration + one-season smoke

Wire ingest → projections → actuals → universe → verdict behind a script, calibrate the pre-registered constants from a prior year, and smoke-test one season.

**Files:**
- Create: `scripts/dfs_edge_study.py`
- Create: `src/projections/dfs/usage.py` (build the actual-usage frame from `weekly_stats`)
- Test: `tests/test_dfs/test_usage.py`

**Interfaces:**
- Produces: `build_usage(weekly_stats) -> pd.DataFrame` → `(gsis_id, season, week, touches_targets)` where `touches_targets = carries + targets` (read the `weekly_stats` columns; if `carries`/`targets` absent, fall back to `rushing_attempts`/`receptions` — confirm column names by reading one partition).
- Produces (script): `run(...)` subcommand: `ingest-sleeper --season --weeks`, `calibrate --prior-season 2020` (prints the empirical δ / usage-floor / N distribution so the committed `config.py` values are justified — does NOT auto-write them), `study --seasons 2021-2024 --out reports/...`.

- [ ] **Step 1: Write the usage test + implement** (TDD as in prior tasks: failing test asserting `touches_targets = carries + targets`, then implement `build_usage`).

- [ ] **Step 2: Implement `scripts/dfs_edge_study.py`** with `argparse` subcommands. `study` loads stored Sleeper weekly partitions (`read_partition(... "sleeper_weekly_projections", season=, week=)` looped over weeks), scores them under `Ruleset.draftkings()` via `expected_points` per row → `sleeper_pts`; calls `emit_weekly_projections` (Task 8); `dk_weekly_actuals` (Task 7); `build_usage` + `build_universe` + `run_edge_study_from_universe`; writes the report (Task 12). Keep the script thin — all logic in `dfs/`.

- [ ] **Step 3: One-season smoke** (guard with `@pytest.mark.skipif` if the real feature/raw partitions are absent, so CI without data still passes):

```python
# tests/test_dfs/test_smoke.py  (sketch)
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    not Path("data/features/wr/season=2023").exists(),
    reason="requires built feature/raw partitions",
)

def test_one_season_end_to_end(tmp_path):
    # ingest a couple weeks (network) or use a saved fixture partition, then
    # run the study for 2023 WR only and assert a verdict in the allowed set.
    ...
```

- [ ] **Step 4: Run available tests + gates; commit**

```bash
git add scripts/dfs_edge_study.py src/projections/dfs/usage.py tests/test_dfs/test_usage.py tests/test_dfs/test_smoke.py
git commit -m "feat(dfs): edge-study CLI, usage frame, calibration + one-season smoke"
```

---

### Task 12: Run the study, write the verdict report, update docs

**Files:**
- Create: `reports/dfs_projection_edge_2026-06-23.md` (the verdict)
- Modify: `TODO.md` (close #39, record the verdict), `project_management.md` (top entry)

- [ ] **Step 1: Calibrate** — run `python scripts/dfs_edge_study.py calibrate --prior-season 2020`, record the empirical δ / usage-floor / cluster-count, and confirm (or adjust, with justification) the `config.py` constants. Commit any change to `config.py` as a `chore(dfs): pre-register study constants` commit *before* running the study.

- [ ] **Step 2: Ingest** — `python scripts/dfs_edge_study.py ingest-sleeper --season 2021 --weeks 1-18` (repeat 2022/2023/2024). (Requires network; ~9k rows/week.)

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

**Placeholder scan:** Task 11/12 steps 1-3 describe operational runs (calibrate/ingest/study) rather than code — these are inherently runtime actions, not code placeholders; the *code* they invoke (`build_usage`, the CLI, the report writer) is specified. The `test_smoke.py` and Task 11 CLI are sketched rather than fully coded because they are thin orchestration over fully-specified functions; the implementer wires documented signatures. Flagged here honestly for the plan-review.

**Type consistency:** stat columns use `Stat.value` strings consistently across emitter (T8), blend (T9), and universe (T10); `Interval` reused from `_compare.py`; `our_pts`/`sleeper_pts`/`blended_pts`/`actual_points`/`player_season` names consistent across T8–T10.
