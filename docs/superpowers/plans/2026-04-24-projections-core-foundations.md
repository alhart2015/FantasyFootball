# Projections Core — Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the typed foundations of Projections Core: project bootstrap, all canonical schemas/enums/typed IDs, scoring engine, distribution layer, parquet+DuckDB store, and a first ingest path (id_map + weekly stats). After this plan, we can ingest NFL weekly data, validate it against schemas, and score arbitrary stat lines under arbitrary rulesets.

**Architecture:** Single Python package `projections` (src layout). Strong typing throughout: `pandera` schemas at every module boundary, `pydantic` v2 for configs/records, `NewType` per ID flavor, enums for every reused string-keyed concept. TDD, with `pytest` and per-module fixtures (no network in tests). See spec at `docs/superpowers/specs/2026-04-24-projections-core-design.md` for context.

**Tech Stack:** Python 3.11+, `pandas`, `pyarrow`, `nfl_data_py`, `pydantic>=2`, `pandera`, `duckdb`, `numpy`, `scipy`, `joblib`, `pytest`, `mypy`, `ruff`.

**Scope of THIS plan (15 tasks):** project bootstrap → enums → NewType IDs → Ruleset → pandera schemas → Distribution layer → scoring (point + distribution) → parquet store → DuckDB views → ingest of `id_map` and `weekly_stats` with manifest + idempotency.

**Explicitly NOT in this plan (future plans):** schedules / snap_counts / depth_charts / NGS ingest, feature builders, Model A, aggregation, backtest harness, full Python API, CLI verbs beyond `refresh`, web UI.

---

## File Structure (created by this plan)

```
pyproject.toml                                          # Task 1: project metadata + deps + tool config
.gitignore                                              # Task 1 (extended)
src/
└── projections/
    ├── __init__.py                                     # Task 1
    ├── schemas.py                                      # Tasks 2-7: ALL canonical types live here
    ├── distributions/
    │   ├── __init__.py                                 # Task 8
    │   ├── base.py                                     # Task 8: Distribution Protocol
    │   └── parametric.py                               # Task 8: ParametricNormal, ParametricGamma
    ├── scoring/
    │   ├── __init__.py                                 # Task 9
    │   ├── score.py                                    # Task 9: score(stat_line, ruleset)
    │   └── score_distribution.py                       # Task 10: score_distribution(...)
    ├── store/
    │   ├── __init__.py                                 # Task 11
    │   ├── parquet.py                                  # Task 11: read/write helpers
    │   └── duckdb_views.py                             # Task 12: view layer
    └── ingest/
        ├── __init__.py                                 # Task 13
        ├── id_map.py                                   # Task 13: build_id_map()
        ├── weekly_stats.py                             # Task 14: refresh_weekly_stats()
        └── manifest.py                                 # Task 15: manifest writer + idempotency
tests/
├── conftest.py                                         # Task 1: shared fixtures
├── test_schemas/                                       # Tasks 2-7
├── test_distributions/                                 # Task 8
├── test_scoring/                                       # Tasks 9-10
├── test_store/                                         # Tasks 11-12
└── test_ingest/                                        # Tasks 13-15
```

`schemas.py` is intentionally one file: it's the single source of truth for all module-boundary types. If it grows past ~600 lines, split by concept (`schemas_enums.py`, `schemas_dataframes.py`, etc.) — but not yet.

---

### Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore` (extend existing)
- Create: `src/projections/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "projections"
version = "0.0.1"
description = "FantasyFootball Projections Core"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15",
    "numpy>=1.26",
    "scipy>=1.12",
    "pydantic>=2.6",
    "pandera>=0.18",
    "duckdb>=0.10",
    "joblib>=1.3",
    "nfl_data_py>=0.3.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=4",
    "mypy>=1.9",
    "ruff>=0.3",
    "types-setuptools",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.mypy]
strict = true
python_version = "3.11"
mypy_path = "src"

[[tool.mypy.overrides]]
module = ["nfl_data_py.*", "pandera.*", "scipy.*", "joblib.*"]
ignore_missing_imports = true

[tool.ruff]
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N"]
```

- [ ] **Step 2: Extend `.gitignore`**

Append to existing `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Local virtualenv
.venv/

# Generated data (do not commit raw or projection parquet)
data/raw/
data/features/
data/projections/
data/backtests/
data/manifests/
*.duckdb
```

- [ ] **Step 3: Create empty package files**

`src/projections/__init__.py`:
```python
"""FantasyFootball Projections Core."""

from __future__ import annotations

__version__ = "0.0.1"
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Write `tests/conftest.py` (skeleton — fixtures added in later tasks)**

```python
"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
```

- [ ] **Step 5: Write smoke test `tests/test_smoke.py`**

```python
"""Smoke test: package imports and version is set."""

from __future__ import annotations

import projections


def test_package_imports() -> None:
    assert projections.__version__ == "0.0.1"
```

- [ ] **Step 6: Install in editable mode and run smoke test**

Run:
```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows bash; PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```
Expected: 1 passed.

- [ ] **Step 7: Run mypy and ruff on the (tiny) codebase**

Run:
```bash
mypy src tests
ruff check src tests
```
Expected: both clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: project bootstrap with src layout, mypy strict, ruff"
```

---

### Task 2: `Position` enum

**Files:**
- Create: `src/projections/schemas.py`
- Create: `tests/test_schemas/__init__.py`
- Create: `tests/test_schemas/test_position.py`

- [ ] **Step 1: Write the failing test `tests/test_schemas/test_position.py`**

```python
"""Position enum tests."""

from __future__ import annotations

import pytest

from projections.schemas import Position


def test_all_skill_positions_present() -> None:
    assert {p.value for p in Position} >= {"QB", "RB", "WR", "TE", "K", "DST"}


def test_position_is_string_enum() -> None:
    # str(Position.QB) should be usable in pandas filters.
    assert Position.QB.value == "QB"
    assert Position("QB") is Position.QB


def test_unknown_position_raises() -> None:
    with pytest.raises(ValueError):
        Position("FB")
```

`tests/test_schemas/__init__.py`: empty file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_position.py -v`
Expected: FAIL with `ImportError: cannot import name 'Position'`.

- [ ] **Step 3: Implement `Position` in `src/projections/schemas.py`**

```python
"""Single source of truth for canonical types: enums, NewTypes, pydantic models, pandera schemas."""

from __future__ import annotations

from enum import StrEnum


class Position(StrEnum):
    """NFL fantasy-relevant positions. Use Position.QB, never the string \"QB\"."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"
    # Reserved for future IDP support; kept here so RosterSlot can refer to them
    # without a circular import. Not currently produced by ingest.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_position.py -v`
Expected: 3 passed.

- [ ] **Step 5: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas
git commit -m "feat(schemas): add Position enum"
```

---

### Task 3: `Team` enum + alias map

**Files:**
- Modify: `src/projections/schemas.py` (append `Team`)
- Create: `tests/test_schemas/test_team.py`

- [ ] **Step 1: Write failing test `tests/test_schemas/test_team.py`**

```python
"""Team enum tests — including alias normalization."""

from __future__ import annotations

import pytest

from projections.schemas import Team, normalize_team_code


def test_thirty_two_teams() -> None:
    assert len(list(Team)) == 32


def test_canonical_codes_are_uppercase_short() -> None:
    for t in Team:
        assert t.value.isupper()
        assert 2 <= len(t.value) <= 3


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("JAX", Team.JAC),
        ("JAC", Team.JAC),
        ("LA", Team.LAR),
        ("LAR", Team.LAR),
        ("STL", Team.LAR),  # Rams pre-2016
        ("SD", Team.LAC),   # Chargers pre-2017
        ("OAK", Team.LV),   # Raiders pre-2020
        ("WAS", Team.WAS),
        ("WSH", Team.WAS),
    ],
)
def test_normalize_known_aliases(alias: str, expected: Team) -> None:
    assert normalize_team_code(alias) is expected


def test_normalize_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown team code"):
        normalize_team_code("XXX")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_team.py -v`
Expected: FAIL — `Team` and `normalize_team_code` don't exist.

- [ ] **Step 3: Add `Team` and `normalize_team_code` to `src/projections/schemas.py`**

Append:

```python
class Team(StrEnum):
    """Canonical NFL team codes. 32 teams.

    `nfl_data_py` historically uses some non-canonical aliases (JAX vs JAC,
    LA vs LAR). Use `normalize_team_code()` to coerce input before storing.
    """

    ARI = "ARI"
    ATL = "ATL"
    BAL = "BAL"
    BUF = "BUF"
    CAR = "CAR"
    CHI = "CHI"
    CIN = "CIN"
    CLE = "CLE"
    DAL = "DAL"
    DEN = "DEN"
    DET = "DET"
    GB = "GB"
    HOU = "HOU"
    IND = "IND"
    JAC = "JAC"
    KC = "KC"
    LAC = "LAC"
    LAR = "LAR"
    LV = "LV"
    MIA = "MIA"
    MIN = "MIN"
    NE = "NE"
    NO = "NO"
    NYG = "NYG"
    NYJ = "NYJ"
    PHI = "PHI"
    PIT = "PIT"
    SEA = "SEA"
    SF = "SF"
    TB = "TB"
    TEN = "TEN"
    WAS = "WAS"


# Aliases keyed lowercase for case-insensitive lookup.
_TEAM_ALIASES: dict[str, Team] = {
    "jax": Team.JAC,
    "la": Team.LAR,
    "stl": Team.LAR,   # Rams pre-2016
    "sd": Team.LAC,    # Chargers pre-2017
    "oak": Team.LV,    # Raiders pre-2020
    "wsh": Team.WAS,
    # Self-aliases for fast normalize_team_code passthrough:
    **{t.value.lower(): t for t in Team},
}


def normalize_team_code(code: str) -> Team:
    """Coerce a possibly-aliased team code to the canonical `Team`.

    Why: `nfl_data_py` and other sources use inconsistent codes across seasons.
    """
    try:
        return _TEAM_ALIASES[code.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown team code: {code!r}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_team.py -v`
Expected: all parametrized cases pass.

- [ ] **Step 5: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_team.py
git commit -m "feat(schemas): add Team enum with alias normalization"
```

---

### Task 4: `RosterSlot`, `DistributionFamily`, `Stat` enums

**Files:**
- Modify: `src/projections/schemas.py`
- Create: `tests/test_schemas/test_other_enums.py`

- [ ] **Step 1: Write failing test `tests/test_schemas/test_other_enums.py`**

```python
"""Tests for RosterSlot, DistributionFamily, Stat."""

from __future__ import annotations

import pytest

from projections.schemas import DistributionFamily, RosterSlot, Stat


def test_roster_slot_includes_super_flex() -> None:
    # Spec calls out superflex-readiness from day 1.
    assert RosterSlot.SUPER_FLEX.value == "SUPER_FLEX"
    assert RosterSlot.FLEX.value == "FLEX"


def test_roster_slot_has_bench_and_ir() -> None:
    assert RosterSlot.BENCH.value == "BENCH"
    assert RosterSlot.IR.value == "IR"


def test_distribution_family_options() -> None:
    assert {f.value for f in DistributionFamily} == {
        "NORMAL",
        "GAMMA",
        "EMPIRICAL_QUANTILE",
        "SAMPLED",
    }


@pytest.mark.parametrize(
    "stat",
    [
        Stat.PASSING_YARDS,
        Stat.PASSING_TDS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.INTERCEPTIONS,
        Stat.FUMBLES_LOST,
    ],
)
def test_core_stats_exist(stat: Stat) -> None:
    assert isinstance(stat.value, str)
    assert stat.value.islower()  # column-name style
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_other_enums.py -v`
Expected: FAIL — enums don't exist.

- [ ] **Step 3: Append to `src/projections/schemas.py`**

```python
class RosterSlot(StrEnum):
    """Roster slot identifiers used by downstream draft / lineup tools.

    SUPER_FLEX is included from day 1 even though the current league uses 1QB,
    so adding a superflex league later is a config flip rather than a rewrite.
    """

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    FLEX = "FLEX"          # RB / WR / TE
    SUPER_FLEX = "SUPER_FLEX"  # QB / RB / WR / TE
    K = "K"
    DST = "DST"
    BENCH = "BENCH"
    IR = "IR"


class DistributionFamily(StrEnum):
    """Backing representation of a `Distribution`."""

    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    EMPIRICAL_QUANTILE = "EMPIRICAL_QUANTILE"  # quantile-regression output
    SAMPLED = "SAMPLED"                        # explicit sample array


class Stat(StrEnum):
    """Canonical column names for player stats. Reference these instead of literals
    in scoring rules and feature builders so typos fail at type-check time."""

    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_tds"
    INTERCEPTIONS = "interceptions"
    PASSING_2PT = "passing_2pt_conversions"
    RUSHING_YARDS = "rushing_yards"
    RUSHING_TDS = "rushing_tds"
    RUSHING_2PT = "rushing_2pt_conversions"
    RECEPTIONS = "receptions"
    RECEIVING_YARDS = "receiving_yards"
    RECEIVING_TDS = "receiving_tds"
    RECEIVING_2PT = "receiving_2pt_conversions"
    FUMBLES_LOST = "fumbles_lost"
    RETURN_TDS = "return_tds"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_other_enums.py -v`
Expected: all pass.

- [ ] **Step 5: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_other_enums.py
git commit -m "feat(schemas): add RosterSlot, DistributionFamily, Stat enums"
```

---

### Task 5: `NewType` IDs

**Files:**
- Modify: `src/projections/schemas.py`
- Create: `tests/test_schemas/test_id_types.py`

- [ ] **Step 1: Write failing test `tests/test_schemas/test_id_types.py`**

```python
"""ID NewType tests — runtime they're str; mypy treats them as distinct."""

from __future__ import annotations

import re

from projections.schemas import EspnId, GsisId, PfrId, SleeperId, validate_gsis_id


def test_gsis_id_is_str_at_runtime() -> None:
    pid = GsisId("00-0036322")
    assert isinstance(pid, str)


def test_validate_gsis_id_format() -> None:
    pid = validate_gsis_id("00-0036322")
    assert pid == GsisId("00-0036322")


def test_validate_gsis_id_rejects_bad_format() -> None:
    import pytest

    for bad in ["", "not-an-id", "0036322", "00-12345", "00-0036322a"]:
        with pytest.raises(ValueError):
            validate_gsis_id(bad)


def test_id_types_are_string_at_runtime_only() -> None:
    # NewType wrappers are noops at runtime — used only by mypy.
    assert EspnId("12345") == "12345"
    assert SleeperId("4046") == "4046"
    assert PfrId("JeffJu00") == "JeffJu00"


def test_gsis_id_pattern_matches_pattern_constant() -> None:
    from projections.schemas import GSIS_ID_PATTERN

    assert re.fullmatch(GSIS_ID_PATTERN, "00-0036322")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_id_types.py -v`
Expected: FAIL.

- [ ] **Step 3: Append to `src/projections/schemas.py`**

```python
import re
from typing import Final, NewType

# Each ID flavor is a distinct mypy type so passing one where another is expected
# is a type error. At runtime they are bare strings.
GsisId = NewType("GsisId", str)
EspnId = NewType("EspnId", str)
SleeperId = NewType("SleeperId", str)
PfrId = NewType("PfrId", str)

GSIS_ID_PATTERN: Final[str] = r"\d{2}-\d{7}"
_GSIS_ID_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")


def validate_gsis_id(raw: str) -> GsisId:
    """Validate that `raw` matches the canonical gsis_id format and return it
    as a `GsisId`. The only sanctioned way to construct a `GsisId` from
    untrusted input."""
    if not _GSIS_ID_RE.fullmatch(raw):
        raise ValueError(f"Invalid gsis_id format: {raw!r}")
    return GsisId(raw)
```

(Place these imports at the top of `schemas.py` if not already; the `from __future__ import annotations` already there should remain at the top.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_id_types.py -v`
Expected: all pass.

- [ ] **Step 5: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_id_types.py
git commit -m "feat(schemas): add NewType IDs and gsis_id validator"
```

---

### Task 6: `Ruleset` pydantic model + named presets

**Files:**
- Modify: `src/projections/schemas.py`
- Create: `tests/test_schemas/test_ruleset.py`

- [ ] **Step 1: Write failing test `tests/test_schemas/test_ruleset.py`**

```python
"""Ruleset tests — point values for each scoring component, and named presets."""

from __future__ import annotations

import pytest

from projections.schemas import Ruleset


def test_default_ruleset_is_espn_ppr() -> None:
    r = Ruleset()
    # ESPN standard PPR defaults.
    assert r.passing_yds_per_pt == 25.0
    assert r.passing_td_pts == 4.0
    assert r.interception_pts == -2.0
    assert r.rushing_yds_per_pt == 10.0
    assert r.rushing_td_pts == 6.0
    assert r.receiving_yds_per_pt == 10.0
    assert r.receiving_td_pts == 6.0
    assert r.reception_pts == 1.0  # full PPR
    assert r.fumble_lost_pts == -2.0
    assert r.two_pt_pts == 2.0
    assert r.return_td_pts == 6.0


def test_espn_half_preset() -> None:
    r = Ruleset.espn_half()
    assert r.reception_pts == 0.5
    assert r.passing_td_pts == 4.0


def test_standard_preset() -> None:
    r = Ruleset.standard()
    assert r.reception_pts == 0.0
    assert r.passing_td_pts == 4.0


def test_ruleset_is_immutable() -> None:
    r = Ruleset()
    with pytest.raises(Exception):  # pydantic raises ValidationError on frozen
        r.reception_pts = 0.5  # type: ignore[misc]


def test_ruleset_has_name() -> None:
    assert Ruleset().name == "ESPN_PPR"
    assert Ruleset.espn_half().name == "ESPN_HALF"
    assert Ruleset.standard().name == "STANDARD"


def test_ruleset_custom_name_allowed() -> None:
    r = Ruleset(name="MY_LEAGUE", reception_pts=0.5, passing_td_pts=6.0)
    assert r.name == "MY_LEAGUE"
    assert r.passing_td_pts == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_ruleset.py -v`
Expected: FAIL.

- [ ] **Step 3: Append to `src/projections/schemas.py`**

```python
from pydantic import BaseModel, ConfigDict, Field


class Ruleset(BaseModel):
    """Scoring ruleset. Defaults match ESPN standard PPR.

    Rulesets are immutable so we can hash/cache them and pass them around
    confidently. Use the named class methods for common presets, or pass field
    overrides for custom leagues.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "ESPN_PPR"

    # Passing
    passing_yds_per_pt: float = Field(default=25.0, gt=0)
    passing_td_pts: float = 4.0
    interception_pts: float = -2.0

    # Rushing
    rushing_yds_per_pt: float = Field(default=10.0, gt=0)
    rushing_td_pts: float = 6.0

    # Receiving
    receiving_yds_per_pt: float = Field(default=10.0, gt=0)
    receiving_td_pts: float = 6.0
    reception_pts: float = 1.0  # PPR

    # Misc
    fumble_lost_pts: float = -2.0
    two_pt_pts: float = 2.0
    return_td_pts: float = 6.0

    @classmethod
    def espn_ppr(cls) -> "Ruleset":
        return cls()

    @classmethod
    def espn_half(cls) -> "Ruleset":
        return cls(name="ESPN_HALF", reception_pts=0.5)

    @classmethod
    def standard(cls) -> "Ruleset":
        return cls(name="STANDARD", reception_pts=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_ruleset.py -v`
Expected: all pass.

- [ ] **Step 5: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_ruleset.py
git commit -m "feat(schemas): add Ruleset model with ESPN_PPR, ESPN_HALF, STANDARD presets"
```

---

### Task 7: `pandera` DataFrame schemas (raw + projection contracts)

**Files:**
- Modify: `src/projections/schemas.py`
- Create: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write failing test `tests/test_schemas/test_dataframe_schemas.py`**

```python
"""Pandera schema tests — verify schemas accept good DataFrames and reject bad ones."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from projections.schemas import (
    IdMapSchema,
    ProjectionWeeklySchema,
    WeeklyStatsSchema,
)


def _good_weekly_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "season": [2024],
            "week": [3],
            "position": ["WR"],
            "team": ["MIN"],
            "opponent": ["HOU"],
            "passing_yards": [0.0],
            "passing_tds": [0],
            "interceptions": [0],
            "rushing_yards": [0.0],
            "rushing_tds": [0],
            "receptions": [9],
            "receiving_yards": [110.0],
            "receiving_tds": [1],
            "fumbles_lost": [0],
        }
    )


def test_weekly_stats_accepts_good_frame() -> None:
    WeeklyStatsSchema.validate(_good_weekly_stats())


def test_weekly_stats_rejects_bad_position() -> None:
    bad = _good_weekly_stats()
    bad["position"] = "FB"
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_weekly_stats_rejects_week_out_of_range() -> None:
    bad = _good_weekly_stats()
    bad["week"] = 99
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_weekly_stats_rejects_bad_gsis_id() -> None:
    bad = _good_weekly_stats()
    bad["gsis_id"] = "not-an-id"
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_id_map_schema() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "espn_id": ["4262921"],
            "sleeper_id": ["6794"],
            "pfr_id": ["JeffJu00"],
            "full_name": ["Justin Jefferson"],
            "position": ["WR"],
            "team": ["MIN"],
        }
    )
    IdMapSchema.validate(df)


def test_projection_weekly_schema() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "season": [2026],
            "week": [3],
            "position": ["WR"],
            "team": ["MIN"],
            "opponent": ["HOU"],
            "ruleset": ["ESPN_PPR"],
            "family": ["GAMMA"],
            "params": [b"\x00"],
            "mean": [18.4],
            "p10": [6.1],
            "p50": [17.2],
            "p90": [33.5],
            "model_id": ["baseline:abc123:2014-2025"],
            "generated_at": [pd.Timestamp("2026-09-01", tz="UTC")],
        }
    )
    ProjectionWeeklySchema.validate(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v`
Expected: FAIL — schemas don't exist.

- [ ] **Step 3: Append to `src/projections/schemas.py`**

```python
import pandera as pa
from pandera.typing import Series


_POSITION_VALUES = [p.value for p in Position]
_TEAM_VALUES = [t.value for t in Team]
_DIST_FAMILY_VALUES = [f.value for f in DistributionFamily]


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
    receptions: Series[int] = pa.Field(ge=0, le=30)
    receiving_yards: Series[float] = pa.Field(ge=-50, le=400)
    receiving_tds: Series[int] = pa.Field(ge=0, le=10)
    fumbles_lost: Series[int] = pa.Field(ge=0, le=10)

    class Config:
        strict = "filter"  # extra columns are dropped, not errored


class IdMapSchema(pa.DataFrameModel):
    """Cross-platform player id translation table."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    espn_id: Series[str] = pa.Field(nullable=True)
    sleeper_id: Series[str] = pa.Field(nullable=True)
    pfr_id: Series[str] = pa.Field(nullable=True)
    full_name: Series[str]
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)

    class Config:
        strict = "filter"


class ProjectionWeeklySchema(pa.DataFrameModel):
    """Published per-week projection (the consumer-facing contract)."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    team: Series[str] = pa.Field(isin=_TEAM_VALUES)
    opponent: Series[str] = pa.Field(isin=_TEAM_VALUES)
    ruleset: Series[str]
    family: Series[str] = pa.Field(isin=_DIST_FAMILY_VALUES)
    params: Series[bytes]
    mean: Series[float]
    p10: Series[float]
    p50: Series[float]
    p90: Series[float]
    model_id: Series[str]
    generated_at: Series[pd.Timestamp] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})

    class Config:
        strict = "filter"
```

(Add `import pandas as pd` at the top of `schemas.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v`
Expected: all pass.

- [ ] **Step 5: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add pandera schemas (WeeklyStats, IdMap, ProjectionWeekly)"
```

---

### Task 8: `Distribution` interface + parametric backings

**Files:**
- Create: `src/projections/distributions/__init__.py`
- Create: `src/projections/distributions/base.py`
- Create: `src/projections/distributions/parametric.py`
- Create: `tests/test_distributions/__init__.py`
- Create: `tests/test_distributions/test_parametric.py`

- [ ] **Step 1: Write failing test `tests/test_distributions/test_parametric.py`**

```python
"""Parametric Distribution tests — math correctness and sampling determinism."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projections.distributions import ParametricGamma, ParametricNormal


def test_normal_mean_std_quantile() -> None:
    d = ParametricNormal(mean=10.0, std=2.0)
    assert d.mean() == pytest.approx(10.0)
    assert d.std() == pytest.approx(2.0)
    assert d.quantile(0.5) == pytest.approx(10.0)
    # ~68% within 1 std => p84.13 ~ 12.0
    assert d.quantile(0.8413) == pytest.approx(12.0, abs=0.01)


def test_normal_sample_shape_and_determinism() -> None:
    d = ParametricNormal(mean=10.0, std=2.0)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1 = d.sample(1000, rng=rng1)
    s2 = d.sample(1000, rng=rng2)
    assert s1.shape == (1000,)
    assert np.array_equal(s1, s2)


def test_normal_sample_summary() -> None:
    d = ParametricNormal(mean=10.0, std=2.0)
    rng = np.random.default_rng(0)
    s = d.sample(100_000, rng=rng)
    assert math.isclose(float(s.mean()), 10.0, abs_tol=0.05)
    assert math.isclose(float(s.std()), 2.0, abs_tol=0.05)


def test_gamma_positive_support() -> None:
    d = ParametricGamma(shape=4.0, scale=3.0)
    assert d.mean() == pytest.approx(12.0)         # shape * scale
    assert d.std() == pytest.approx(math.sqrt(36)) # sqrt(shape) * scale
    rng = np.random.default_rng(1)
    s = d.sample(10_000, rng=rng)
    assert (s >= 0).all()


def test_gamma_quantile_monotonic() -> None:
    d = ParametricGamma(shape=2.0, scale=5.0)
    q10 = d.quantile(0.1)
    q50 = d.quantile(0.5)
    q90 = d.quantile(0.9)
    assert q10 < q50 < q90
```

`tests/test_distributions/__init__.py`: empty.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributions/test_parametric.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `src/projections/distributions/base.py`**

```python
"""Distribution interface — value object exposing mean/quantile/sample."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Distribution(Protocol):
    """A probability distribution over a single player's fantasy points (or
    underlying stat). Backings: parametric (Normal/Gamma), empirical-quantile,
    or sampled. Same surface regardless."""

    def mean(self) -> float: ...
    def std(self) -> float: ...
    def quantile(self, q: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]: ...
```

- [ ] **Step 4: Implement `src/projections/distributions/parametric.py`**

```python
"""Parametric distribution backings: Normal and Gamma."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(slots=True, frozen=True)
class ParametricNormal:
    mean_: float
    std_: float

    def __init__(self, mean: float, std: float) -> None:
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        object.__setattr__(self, "mean_", float(mean))
        object.__setattr__(self, "std_", float(std))

    def mean(self) -> float:
        return self.mean_

    def std(self) -> float:
        return self.std_

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(stats.norm.ppf(q, loc=self.mean_, scale=self.std_))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.normal(loc=self.mean_, scale=self.std_, size=n).astype(np.float64)


@dataclass(slots=True, frozen=True)
class ParametricGamma:
    """Shape (k) / scale (theta) parameterization. mean = k*theta, var = k*theta^2."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError(f"shape and scale must be positive; got {self.shape}, {self.scale}")

    def mean(self) -> float:
        return self.shape * self.scale

    def std(self) -> float:
        return float(np.sqrt(self.shape) * self.scale)

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(stats.gamma.ppf(q, a=self.shape, scale=self.scale))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.gamma(shape=self.shape, scale=self.scale, size=n).astype(np.float64)
```

- [ ] **Step 5: Implement `src/projections/distributions/__init__.py`**

```python
"""Distribution layer — interface + parametric implementations."""

from __future__ import annotations

from projections.distributions.base import Distribution
from projections.distributions.parametric import ParametricGamma, ParametricNormal

__all__ = ["Distribution", "ParametricGamma", "ParametricNormal"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_distributions/ -v`
Expected: all pass.

- [ ] **Step 7: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/projections/distributions tests/test_distributions
git commit -m "feat(distributions): add Distribution Protocol + ParametricNormal + ParametricGamma"
```

---

### Task 9: Scoring engine — `score(stat_line, ruleset)`

**Files:**
- Create: `src/projections/scoring/__init__.py`
- Create: `src/projections/scoring/score.py`
- Create: `tests/test_scoring/__init__.py`
- Create: `tests/test_scoring/test_score.py`

- [ ] **Step 1: Write failing test `tests/test_scoring/test_score.py`**

```python
"""Scoring tests — every rule × every preset."""

from __future__ import annotations

import pytest

from projections.schemas import Ruleset
from projections.scoring import StatLine, score


def _zero_line(**overrides: float) -> StatLine:
    base = dict(
        passing_yards=0.0,
        passing_tds=0,
        interceptions=0,
        passing_2pt_conversions=0,
        rushing_yards=0.0,
        rushing_tds=0,
        rushing_2pt_conversions=0,
        receptions=0,
        receiving_yards=0.0,
        receiving_tds=0,
        receiving_2pt_conversions=0,
        fumbles_lost=0,
        return_tds=0,
    )
    base.update(overrides)
    return StatLine(**base)  # type: ignore[arg-type]


def test_passing_yards() -> None:
    line = _zero_line(passing_yards=300.0)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(12.0)


def test_passing_yards_partial() -> None:
    line = _zero_line(passing_yards=275.0)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(11.0)


def test_passing_td_and_int() -> None:
    line = _zero_line(passing_tds=2, interceptions=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(2 * 4 + 1 * -2)


def test_rushing() -> None:
    line = _zero_line(rushing_yards=120.0, rushing_tds=2)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(12.0 + 12.0)


def test_receiving_full_ppr() -> None:
    line = _zero_line(receptions=8, receiving_yards=100.0, receiving_tds=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(8 + 10 + 6)


def test_receiving_half_ppr() -> None:
    line = _zero_line(receptions=8, receiving_yards=100.0, receiving_tds=1)
    assert score(line, Ruleset.espn_half()) == pytest.approx(4 + 10 + 6)


def test_receiving_standard() -> None:
    line = _zero_line(receptions=8, receiving_yards=100.0, receiving_tds=1)
    assert score(line, Ruleset.standard()) == pytest.approx(0 + 10 + 6)


def test_fumble_and_two_pt_and_return_td() -> None:
    line = _zero_line(fumbles_lost=1, rushing_2pt_conversions=1, return_tds=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(-2 + 2 + 6)


def test_jefferson_real_line() -> None:
    # 9 rec, 110 rec yds, 1 rec TD => 9 + 11 + 6 = 26 in PPR.
    line = _zero_line(receptions=9, receiving_yards=110.0, receiving_tds=1)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(26.0)


def test_negative_yards_dock_below_zero() -> None:
    line = _zero_line(rushing_yards=-5.0)
    assert score(line, Ruleset.espn_ppr()) == pytest.approx(-0.5)
```

`tests/test_scoring/__init__.py`: empty.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring/test_score.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/projections/scoring/score.py`**

```python
"""Pure stat-line → fantasy-points scoring. Table-driven for testability."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from projections.schemas import Ruleset


class StatLine(BaseModel):
    """A single player-week stat line. Values are *realized* counts/yards.

    Used by `score()` and as the natural per-sample shape produced when
    `Distribution.sample()` is paired with the underlying stat dimensions.
    """

    model_config = ConfigDict(frozen=True)

    passing_yards: float = 0.0
    passing_tds: int = 0
    interceptions: int = 0
    passing_2pt_conversions: int = 0

    rushing_yards: float = 0.0
    rushing_tds: int = 0
    rushing_2pt_conversions: int = 0

    receptions: int = 0
    receiving_yards: float = 0.0
    receiving_tds: int = 0
    receiving_2pt_conversions: int = 0

    fumbles_lost: int = 0
    return_tds: int = 0


def score(line: StatLine, ruleset: Ruleset) -> float:
    """Convert a `StatLine` to fantasy points under `ruleset`. Pure function."""
    pts = 0.0
    pts += line.passing_yards / ruleset.passing_yds_per_pt
    pts += line.passing_tds * ruleset.passing_td_pts
    pts += line.interceptions * ruleset.interception_pts
    pts += line.passing_2pt_conversions * ruleset.two_pt_pts

    pts += line.rushing_yards / ruleset.rushing_yds_per_pt
    pts += line.rushing_tds * ruleset.rushing_td_pts
    pts += line.rushing_2pt_conversions * ruleset.two_pt_pts

    pts += line.receptions * ruleset.reception_pts
    pts += line.receiving_yards / ruleset.receiving_yds_per_pt
    pts += line.receiving_tds * ruleset.receiving_td_pts
    pts += line.receiving_2pt_conversions * ruleset.two_pt_pts

    pts += line.fumbles_lost * ruleset.fumble_lost_pts
    pts += line.return_tds * ruleset.return_td_pts
    return pts
```

- [ ] **Step 4: Implement `src/projections/scoring/__init__.py`**

```python
"""Scoring engine — pure stat → points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.score import StatLine, score

__all__ = ["StatLine", "score"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_scoring/test_score.py -v`
Expected: all pass.

- [ ] **Step 6: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/projections/scoring tests/test_scoring/__init__.py tests/test_scoring/test_score.py
git commit -m "feat(scoring): add StatLine and score() with full PPR/half/standard coverage"
```

---

### Task 10: Scoring engine — `score_distribution`

**Files:**
- Create: `src/projections/scoring/score_distribution.py`
- Modify: `src/projections/scoring/__init__.py`
- Create: `tests/test_scoring/test_score_distribution.py`

- [ ] **Step 1: Write failing test `tests/test_scoring/test_score_distribution.py`**

```python
"""score_distribution: convert per-stat distributions into a fantasy-points distribution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projections.distributions import ParametricGamma, ParametricNormal
from projections.schemas import Ruleset, Stat
from projections.scoring import score_distribution


def test_passing_yards_only_distribution() -> None:
    # If only passing_yards is uncertain (mean=300, std=50), the points
    # distribution should have mean ~ 12 and std ~ 50/25 = 2.
    stat_dists = {Stat.PASSING_YARDS: ParametricNormal(mean=300.0, std=50.0)}
    out = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=20_000, seed=42)
    assert math.isclose(out.mean(), 12.0, abs_tol=0.05)
    assert math.isclose(out.std(), 2.0, abs_tol=0.05)


def test_combined_receiving_distribution() -> None:
    # Receptions ~ Gamma(shape=8, scale=1) => mean 8, var 8
    # Rec yards ~ Normal(mean=100, std=20)
    # Rec TDs constant 0
    # In PPR: pts = 1*rec + rec_yds/10 => mean = 8 + 10 = 18, std ≈ sqrt(8 + 4) = sqrt(12)
    stat_dists = {
        Stat.RECEPTIONS: ParametricGamma(shape=8.0, scale=1.0),
        Stat.RECEIVING_YARDS: ParametricNormal(mean=100.0, std=20.0),
    }
    out = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=50_000, seed=0)
    assert math.isclose(out.mean(), 18.0, abs_tol=0.1)
    assert math.isclose(out.std(), math.sqrt(12), abs_tol=0.1)


def test_score_distribution_is_deterministic_with_seed() -> None:
    stat_dists = {Stat.PASSING_YARDS: ParametricNormal(mean=300.0, std=50.0)}
    a = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=1000, seed=7)
    b = score_distribution(stat_dists, Ruleset.espn_ppr(), n_samples=1000, seed=7)
    assert a.mean() == pytest.approx(b.mean())
    assert a.std() == pytest.approx(b.std())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring/test_score_distribution.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/projections/scoring/score_distribution.py`**

```python
"""Convert a dict of per-stat distributions into a single fantasy-points distribution.

Strategy: Monte Carlo. Sample n times from each underlying-stat distribution,
score each row through the scoring function, and return an empirical (sampled)
distribution backed by the resulting array.

This is intentionally NOT analytic: real per-stat distributions are not Gaussian
and don't combine cleanly. Sampling lets us re-score under any ruleset for free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from projections.distributions import Distribution
from projections.schemas import Ruleset, Stat
from projections.scoring.score import StatLine, score


@dataclass(slots=True, frozen=True)
class SampledDistribution:
    """Empirical distribution backed by a samples array."""

    samples: NDArray[np.float64]

    def mean(self) -> float:
        return float(self.samples.mean())

    def std(self) -> float:
        return float(self.samples.std())

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(np.quantile(self.samples, q))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.choice(self.samples, size=n, replace=True)


_INTEGER_STATS: frozenset[Stat] = frozenset(
    {
        Stat.PASSING_TDS,
        Stat.INTERCEPTIONS,
        Stat.PASSING_2PT,
        Stat.RUSHING_TDS,
        Stat.RUSHING_2PT,
        Stat.RECEPTIONS,
        Stat.RECEIVING_TDS,
        Stat.RECEIVING_2PT,
        Stat.FUMBLES_LOST,
        Stat.RETURN_TDS,
    }
)


def score_distribution(
    stat_dists: Mapping[Stat, Distribution],
    ruleset: Ruleset,
    *,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> SampledDistribution:
    """Build a points distribution by sampling each stat distribution and scoring."""
    rng = np.random.default_rng(seed)

    # Sample each stat n_samples times. Missing stats default to 0.
    samples_per_stat: dict[Stat, NDArray[np.float64]] = {}
    for stat, dist in stat_dists.items():
        s = dist.sample(n_samples, rng=rng)
        if stat in _INTEGER_STATS:
            # Round to non-negative integers; floor at 0 since count stats can't be negative.
            s = np.maximum(np.rint(s), 0.0)
        samples_per_stat[stat] = s

    # Score each row.
    points = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        kwargs: dict[str, float | int] = {}
        for stat, arr in samples_per_stat.items():
            kwargs[stat.value] = arr[i] if stat not in _INTEGER_STATS else int(arr[i])
        points[i] = score(StatLine(**kwargs), ruleset)  # type: ignore[arg-type]

    return SampledDistribution(samples=points)
```

- [ ] **Step 4: Update `src/projections/scoring/__init__.py`**

```python
"""Scoring engine — pure stat → points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.score import StatLine, score
from projections.scoring.score_distribution import SampledDistribution, score_distribution

__all__ = ["SampledDistribution", "StatLine", "score", "score_distribution"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_scoring/test_score_distribution.py -v`
Expected: all pass.

- [ ] **Step 6: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/projections/scoring tests/test_scoring/test_score_distribution.py
git commit -m "feat(scoring): add score_distribution + SampledDistribution backing"
```

---

### Task 11: Store — parquet read/write

**Files:**
- Create: `src/projections/store/__init__.py`
- Create: `src/projections/store/parquet.py`
- Create: `tests/test_store/__init__.py`
- Create: `tests/test_store/test_parquet.py`

- [ ] **Step 1: Write failing test `tests/test_store/test_parquet.py`**

```python
"""Parquet store tests — partitioned writes/reads, idempotency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.store import read_partition, write_partition


def _frame(season: int, week: int, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": [f"00-00{i:05d}" for i in range(n)],
            "season": [season] * n,
            "week": [week] * n,
            "value": [float(i) for i in range(n)],
        }
    )


def test_write_and_read_single_partition(tmp_path: Path) -> None:
    df = _frame(2024, 3)
    write_partition(tmp_path, "weekly_stats", df, season=2024, week=3)
    out = read_partition(tmp_path, "weekly_stats", season=2024, week=3)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True), check_like=True
    )


def test_write_idempotent_overwrites_partition(tmp_path: Path) -> None:
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3, n=5), season=2024, week=3)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3, n=2), season=2024, week=3)
    out = read_partition(tmp_path, "weekly_stats", season=2024, week=3)
    assert len(out) == 2  # second write replaces first


def test_read_across_weeks(tmp_path: Path) -> None:
    write_partition(tmp_path, "weekly_stats", _frame(2024, 1), season=2024, week=1)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 2), season=2024, week=2)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3), season=2024, week=3)
    all_2024 = read_partition(tmp_path, "weekly_stats", season=2024)
    assert sorted(all_2024["week"].unique().tolist()) == [1, 2, 3]


def test_season_only_partition(tmp_path: Path) -> None:
    df = pd.DataFrame({"gsis_id": ["00-0036322"], "espn_id": ["4262921"]})
    write_partition(tmp_path, "id_map", df, season=None, week=None)
    out = read_partition(tmp_path, "id_map", season=None, week=None)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True), check_like=True
    )
```

`tests/test_store/__init__.py`: empty.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store/test_parquet.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/projections/store/parquet.py`**

```python
"""Parquet partitioned read/write helpers. Layout is `{table}/season=YYYY/week=WW/part.parquet`.
Tables without season (e.g., id_map) are written to `{table}.parquet`."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def _partition_dir(root: Path, table: str, season: int | None, week: int | None) -> Path:
    if season is None:
        if week is not None:
            raise ValueError("week cannot be set when season is None")
        return root / table
    p = root / table / f"season={season}"
    if week is not None:
        p = p / f"week={week:02d}"
    return p


def _partition_file(root: Path, table: str, season: int | None, week: int | None) -> Path:
    if season is None:
        return root / f"{table}.parquet"
    return _partition_dir(root, table, season, week) / "part.parquet"


def write_partition(
    root: Path,
    table: str,
    df: pd.DataFrame,
    *,
    season: int | None,
    week: int | None,
) -> Path:
    """Write `df` to the parquet partition for `(table, season, week)`. Idempotent:
    removes the existing partition file first if present."""
    target = _partition_file(root, table, season, week)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    df.to_parquet(target, index=False)
    return target


def read_partition(
    root: Path,
    table: str,
    *,
    season: int | None = None,
    week: int | None = None,
) -> pd.DataFrame:
    """Read parquet partition(s). If `week` is None and `season` is set, reads all
    weeks under that season. If `season` is None, reads the unpartitioned table file."""
    if season is None:
        return pd.read_parquet(_partition_file(root, table, None, None))

    if week is not None:
        return pd.read_parquet(_partition_file(root, table, season, week))

    season_dir = _partition_dir(root, table, season, None)
    files = sorted(season_dir.rglob("part.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet partitions under {season_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def delete_partition(
    root: Path, table: str, *, season: int | None, week: int | None = None
) -> None:
    """Remove a partition directory or unpartitioned file. Used by tests and re-ingests."""
    if season is None:
        f = _partition_file(root, table, None, None)
        if f.exists():
            f.unlink()
        return
    target = _partition_dir(root, table, season, week)
    if target.exists():
        shutil.rmtree(target)
```

- [ ] **Step 4: Implement `src/projections/store/__init__.py`**

```python
"""Store — parquet partitioned reads/writes plus DuckDB views."""

from __future__ import annotations

from projections.store.parquet import delete_partition, read_partition, write_partition

__all__ = ["delete_partition", "read_partition", "write_partition"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_store/test_parquet.py -v`
Expected: all pass.

- [ ] **Step 6: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/projections/store tests/test_store
git commit -m "feat(store): partitioned parquet read/write with idempotency"
```

---

### Task 12: Store — DuckDB view layer

**Files:**
- Create: `src/projections/store/duckdb_views.py`
- Modify: `src/projections/store/__init__.py`
- Create: `tests/test_store/test_duckdb_views.py`

- [ ] **Step 1: Write failing test `tests/test_store/test_duckdb_views.py`**

```python
"""DuckDB view layer — query parquet partitions as SQL tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.store import query, write_partition


def _seed(root: Path) -> None:
    df1 = pd.DataFrame(
        {"gsis_id": ["00-0036322"], "season": [2024], "week": [1], "mean": [18.0]}
    )
    df2 = pd.DataFrame(
        {"gsis_id": ["00-0036322"], "season": [2024], "week": [2], "mean": [22.5]}
    )
    write_partition(root, "projections_weekly", df1, season=2024, week=1)
    write_partition(root, "projections_weekly", df2, season=2024, week=2)


def test_query_combines_partitions(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = query(
        tmp_path,
        "SELECT week, mean FROM projections_weekly ORDER BY week",
    )
    assert out["week"].tolist() == [1, 2]
    assert out["mean"].tolist() == [18.0, 22.5]


def test_query_handles_missing_table(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(Exception):
        query(tmp_path, "SELECT * FROM no_such_table")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store/test_duckdb_views.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/projections/store/duckdb_views.py`**

```python
"""DuckDB view layer — register parquet directories as queryable tables."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _connect_with_views(root: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and register every directory under
    `root` as a view over its parquet files. Tables that don't exist yet
    simply don't get a view."""
    con = duckdb.connect(database=":memory:")
    if not root.exists():
        return con

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            # Partitioned table: glob over season=*/week=*/part.parquet
            glob = str(entry / "**" / "part.parquet")
            con.execute(
                f"CREATE OR REPLACE VIEW {entry.name} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
            )
        elif entry.is_file() and entry.suffix == ".parquet":
            # Unpartitioned table.
            con.execute(
                f"CREATE OR REPLACE VIEW {entry.stem} AS "
                f"SELECT * FROM read_parquet('{entry}')"
            )
    return con


def query(root: Path, sql: str) -> pd.DataFrame:
    """Run a SQL query against the parquet views under `root`. Returns a pandas
    DataFrame. Connection is opened/closed per call (cheap; in-memory)."""
    con = _connect_with_views(root)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()
```

- [ ] **Step 4: Update `src/projections/store/__init__.py`**

```python
"""Store — parquet partitioned reads/writes plus DuckDB views."""

from __future__ import annotations

from projections.store.duckdb_views import query
from projections.store.parquet import delete_partition, read_partition, write_partition

__all__ = ["delete_partition", "query", "read_partition", "write_partition"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_store/test_duckdb_views.py -v`
Expected: all pass.

- [ ] **Step 6: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/projections/store tests/test_store/test_duckdb_views.py
git commit -m "feat(store): DuckDB view layer over parquet partitions"
```

---

### Task 13: Ingest — `id_map` (with mocked `nfl_data_py`)

**Files:**
- Create: `src/projections/ingest/__init__.py`
- Create: `src/projections/ingest/id_map.py`
- Create: `tests/test_ingest/__init__.py`
- Create: `tests/test_ingest/conftest.py`
- Create: `tests/test_ingest/test_id_map.py`

- [ ] **Step 1: Write fixture `tests/test_ingest/conftest.py`**

```python
"""Shared ingest test fixtures — fake `nfl_data_py` responses."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def fake_id_map_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ids()` — a row per player with cross-platform IDs."""
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036322", "00-0034857", "00-0034796"],
            "espn_id": ["4262921", "3915511", "4035687"],
            "sleeper_id": ["6794", "5849", "5045"],
            "pfr_id": ["JeffJu00", "MahoPa00", "BarkSa00"],
            "name": ["Justin Jefferson", "Patrick Mahomes", "Saquon Barkley"],
            "position": ["WR", "QB", "RB"],
            "team": ["MIN", "KC", "PHI"],
        }
    )


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
            "receptions": [9, 0],
            "receiving_yards": [110.0, 0.0],
            "receiving_tds": [1, 0],
            "fumbles_lost": [0, 0],
        }
    )
```

`tests/test_ingest/__init__.py`: empty.

- [ ] **Step 2: Write failing test `tests/test_ingest/test_id_map.py`**

```python
"""id_map ingest tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from projections.ingest import build_id_map
from projections.schemas import IdMapSchema
from projections.store import read_partition


def test_build_id_map_writes_validated_parquet(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map",
        lambda: fake_id_map_df,
    )
    out_path = build_id_map(tmp_path)
    assert out_path.exists()

    df = read_partition(tmp_path, "id_map", season=None, week=None)
    IdMapSchema.validate(df)  # raises if anything is off
    assert set(df["gsis_id"]) == {"00-0036322", "00-0034857", "00-0034796"}


def test_build_id_map_renames_name_column(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map",
        lambda: fake_id_map_df,
    )
    build_id_map(tmp_path)
    df = read_partition(tmp_path, "id_map", season=None, week=None)
    assert "full_name" in df.columns
    assert "name" not in df.columns


def test_build_id_map_drops_rows_without_gsis_id(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    bad = fake_id_map_df.copy()
    bad.loc[len(bad)] = {
        "gsis_id": None,
        "espn_id": "1",
        "sleeper_id": "1",
        "pfr_id": "x",
        "name": "no gsis",
        "position": "WR",
        "team": "MIN",
    }
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: bad)
    build_id_map(tmp_path)
    df = read_partition(tmp_path, "id_map", season=None, week=None)
    assert df["gsis_id"].notna().all()
    assert len(df) == 3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ingest/test_id_map.py -v`
Expected: FAIL — module/functions don't exist.

- [ ] **Step 4: Implement `src/projections/ingest/id_map.py`**

```python
"""Build the canonical id_map.parquet from `nfl_data_py.import_ids()`.

`_fetch_raw_id_map` is split out so tests can monkeypatch it instead of
hitting the network.
"""

from __future__ import annotations

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.schemas import IdMapSchema, normalize_team_code
from projections.store import write_partition


def _fetch_raw_id_map() -> pd.DataFrame:
    return nfl.import_ids()


def build_id_map(data_root: Path) -> Path:
    """Fetch player IDs across platforms, normalize, and write to id_map.parquet.
    Idempotent — re-running overwrites the existing file."""
    raw = _fetch_raw_id_map()

    cols = ["gsis_id", "espn_id", "sleeper_id", "pfr_id", "name", "position", "team"]
    df = raw[[c for c in cols if c in raw.columns]].copy()
    df = df.rename(columns={"name": "full_name"})

    # Drop rows without canonical id; downstream joins are unusable without it.
    df = df[df["gsis_id"].notna()].copy()

    # Coerce IDs to nullable object strings (some sources return numeric espn_id /
    # sleeper_id). Using object dtype + None for missing keeps pandera happy with
    # Series[str] + nullable=True.
    for col in ("espn_id", "sleeper_id", "pfr_id"):
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), other=None).astype(object)

    df["gsis_id"] = df["gsis_id"].astype(str)
    df["full_name"] = df["full_name"].astype(str)
    df["position"] = df["position"].astype(str)
    df["team"] = (
        df["team"]
        .where(df["team"].notna(), other=None)
        .map(lambda v: normalize_team_code(v).value if v is not None else None)
        .astype(object)
    )

    df = df.drop_duplicates(subset=["gsis_id"], keep="first").reset_index(drop=True)

    IdMapSchema.validate(df)
    return write_partition(data_root / "raw", "id_map", df, season=None, week=None)
```

- [ ] **Step 5: Implement `src/projections/ingest/__init__.py`**

```python
"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.id_map import build_id_map

__all__ = ["build_id_map"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_ingest/test_id_map.py -v`
Expected: all pass.

- [ ] **Step 7: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/projections/ingest tests/test_ingest
git commit -m "feat(ingest): build_id_map() with team normalization and schema validation"
```

---

### Task 14: Ingest — weekly stats

**Files:**
- Create: `src/projections/ingest/weekly_stats.py`
- Modify: `src/projections/ingest/__init__.py`
- Create: `tests/test_ingest/test_weekly_stats.py`

- [ ] **Step 1: Write failing test `tests/test_ingest/test_weekly_stats.py`**

```python
"""Weekly-stats ingest tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from projections.ingest import refresh_weekly_stats
from projections.schemas import WeeklyStatsSchema
from projections.store import read_partition


def test_refresh_weekly_stats_writes_partitioned_parquet(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    written = refresh_weekly_stats(tmp_path, seasons=[2024])
    assert len(written) == 1  # one season => one parquet file (week-level not split here)

    df = read_partition(tmp_path, "weekly_stats", season=2024)
    WeeklyStatsSchema.validate(df)
    assert set(df["gsis_id"]) == {"00-0036322", "00-0034857"}


def test_refresh_weekly_stats_renames_columns(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path, "weekly_stats", season=2024)
    assert "gsis_id" in df.columns
    assert "team" in df.columns
    assert "opponent" in df.columns
    assert "player_id" not in df.columns
    assert "recent_team" not in df.columns
    assert "opponent_team" not in df.columns


def test_refresh_weekly_stats_normalizes_team_codes(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    aliased = fake_weekly_df.copy()
    aliased.loc[0, "recent_team"] = "JAX"   # alias for JAC
    aliased.loc[1, "opponent_team"] = "LA"  # alias for LAR
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: aliased,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path, "weekly_stats", season=2024)
    assert "JAX" not in df["team"].tolist()
    assert "JAC" in df["team"].tolist()
    assert "LAR" in df["opponent"].tolist()


def test_refresh_weekly_stats_idempotent(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path, "weekly_stats", season=2024)
    assert len(df) == 2  # not 4 — second run replaced the partition
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest/test_weekly_stats.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/projections/ingest/weekly_stats.py`**

```python
"""Refresh per-season weekly stats from `nfl_data_py.import_weekly_data`.

Writes one parquet partition per season (further per-week splitting is
unnecessary at this scale — a season is small).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.schemas import WeeklyStatsSchema, normalize_team_code
from projections.store import write_partition


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
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
]

_RENAME = {
    "player_id": "gsis_id",
    "recent_team": "team",
    "opponent_team": "opponent",
}


def _fetch_raw_weekly(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_weekly_data(seasons)


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # Coerce dtypes that nfl_data_py sometimes returns as floats.
    for int_col in ("passing_tds", "interceptions", "rushing_tds", "receptions",
                    "receiving_tds", "fumbles_lost"):
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype(int)

    for float_col in ("passing_yards", "rushing_yards", "receiving_yards"):
        if float_col in df.columns:
            df[float_col] = df[float_col].fillna(0.0).astype(float)

    df["team"] = df["team"].map(lambda v: normalize_team_code(v).value)
    df["opponent"] = df["opponent"].map(lambda v: normalize_team_code(v).value)

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(str)

    WeeklyStatsSchema.validate(df)
    return df


def refresh_weekly_stats(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write weekly stats for each season. One partition per season.
    Idempotent — re-running a season overwrites that partition only."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_weekly([season])
        df = _normalize_one_season(raw)
        path = write_partition(
            data_root / "raw", "weekly_stats", df, season=season, week=None
        )
        written.append(path)
    return written
```

- [ ] **Step 4: Update `src/projections/ingest/__init__.py`**

```python
"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.id_map import build_id_map
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = ["build_id_map", "refresh_weekly_stats"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_ingest/test_weekly_stats.py -v`
Expected: all pass.

- [ ] **Step 6: Type-check and lint**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/projections/ingest tests/test_ingest/test_weekly_stats.py
git commit -m "feat(ingest): refresh_weekly_stats() with column rename, team normalization, schema validation"
```

---

### Task 15: Ingest — manifest + idempotency record

**Files:**
- Create: `src/projections/ingest/manifest.py`
- Modify: `src/projections/ingest/id_map.py` (call manifest after write)
- Modify: `src/projections/ingest/weekly_stats.py` (call manifest after write)
- Modify: `src/projections/ingest/__init__.py`
- Create: `tests/test_ingest/test_manifest.py`

- [ ] **Step 1: Write failing test `tests/test_ingest/test_manifest.py`**

```python
"""Ingest manifest tests — every refresh records (table, season, fetched_at, rowcount, checksum)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from projections.ingest import build_id_map, read_manifest, refresh_weekly_stats


def test_id_map_refresh_records_manifest_entry(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map", lambda: fake_id_map_df
    )
    build_id_map(tmp_path)
    manifest = read_manifest(tmp_path)
    rows = manifest[manifest["table"] == "id_map"]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["rowcount"] == 3
    assert pd.isna(row["season"])
    assert isinstance(row["checksum"], str) and len(row["checksum"]) == 64


def test_weekly_refresh_records_one_entry_per_season(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2023, 2024])
    manifest = read_manifest(tmp_path)
    rows = manifest[manifest["table"] == "weekly_stats"]
    assert sorted(rows["season"].tolist()) == [2023, 2024]
    assert (rows["rowcount"] == 2).all()


def test_re_refresh_replaces_manifest_entry(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    refresh_weekly_stats(tmp_path, seasons=[2024])
    manifest = read_manifest(tmp_path)
    rows = manifest[(manifest["table"] == "weekly_stats") & (manifest["season"] == 2024)]
    assert len(rows) == 1  # second run replaced, not appended
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest/test_manifest.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/projections/ingest/manifest.py`**

```python
"""Track every ingest write in `data/manifests/ingest_manifest.parquet`.

Schema: (table, season, fetched_at, rowcount, checksum). One row per
(table, season) pair — re-runs replace the row in place so the manifest
always reflects the *current* on-disk state."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_MANIFEST_FILE = Path("manifests") / "ingest_manifest.parquet"


def _manifest_path(data_root: Path) -> Path:
    return data_root / _MANIFEST_FILE


def compute_checksum(df: pd.DataFrame) -> str:
    """SHA-256 over the parquet bytes of `df` — stable identifier for content."""
    blob = df.to_parquet(index=False)
    return hashlib.sha256(blob).hexdigest()


def record(
    data_root: Path,
    *,
    table: str,
    season: int | None,
    df: pd.DataFrame,
) -> None:
    """Upsert a manifest row for `(table, season)`. Replaces any existing row."""
    path = _manifest_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame(
        [
            {
                "table": table,
                "season": pd.NA if season is None else int(season),
                "fetched_at": datetime.now(timezone.utc),
                "rowcount": int(len(df)),
                "checksum": compute_checksum(df),
            }
        ]
    )

    if path.exists():
        existing = pd.read_parquet(path)
        if season is None:
            mask = (existing["table"] == table) & existing["season"].isna()
        else:
            mask = (existing["table"] == table) & (existing["season"] == season)
        existing = existing[~mask]
        out = pd.concat([existing, new_row], ignore_index=True)
    else:
        out = new_row

    out.to_parquet(path, index=False)


def read_manifest(data_root: Path) -> pd.DataFrame:
    path = _manifest_path(data_root)
    if not path.exists():
        return pd.DataFrame(columns=["table", "season", "fetched_at", "rowcount", "checksum"])
    return pd.read_parquet(path)
```

- [ ] **Step 4: Wire manifest into `id_map.build_id_map`**

In `src/projections/ingest/id_map.py`, add a direct module-level import (NOT `from projections.ingest import manifest` — that would re-trigger the package init while it's loading and circular-import) and call after writing:

```python
from projections.ingest.manifest import record as record_manifest

def build_id_map(data_root: Path) -> Path:
    # ... existing fetch + normalize logic, ending with:

    IdMapSchema.validate(df)
    out = write_partition(data_root / "raw", "id_map", df, season=None, week=None)
    record_manifest(data_root, table="id_map", season=None, df=df)
    return out
```

- [ ] **Step 5: Wire manifest into `weekly_stats.refresh_weekly_stats`**

In `src/projections/ingest/weekly_stats.py`, add the direct import and record after each season:

```python
from projections.ingest.manifest import record as record_manifest

def refresh_weekly_stats(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_weekly([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "weekly_stats", df, season=season, week=None)
        record_manifest(data_root, table="weekly_stats", season=season, df=df)
        written.append(path)
    return written
```

- [ ] **Step 6: Update `src/projections/ingest/__init__.py`**

```python
"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.id_map import build_id_map
from projections.ingest.manifest import compute_checksum, read_manifest, record
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "compute_checksum",
    "read_manifest",
    "record",
    "refresh_weekly_stats",
]
```

- [ ] **Step 7: Run all tests to verify everything still passes**

Run: `pytest -v`
Expected: every test in the suite passes.

- [ ] **Step 8: Type-check and lint the full codebase**

Run: `mypy src tests && ruff check src tests`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/projections/ingest tests/test_ingest/test_manifest.py
git commit -m "feat(ingest): manifest writer with checksum + idempotent upsert; wire into id_map and weekly_stats"
```

---

## Self-review checklist (run after the last task)

- [ ] All 15 tasks committed; `git log --oneline` shows the bite-sized history.
- [ ] `pytest -v` is green.
- [ ] `mypy src tests` is clean.
- [ ] `ruff check src tests` is clean.
- [ ] Re-run `pytest -v --collect-only | wc -l` and confirm at least ~30 tests across schemas, distributions, scoring, store, and ingest.
- [ ] Confirm no `Any` leaked into source without justification (`grep -r ": Any" src/` should return only fixture-related uses if any).

## What you can do after this plan

- Construct any `Ruleset`, `StatLine`, and call `score()` / `score_distribution()`.
- Build `ParametricNormal` / `ParametricGamma` distributions and sample/quantile them.
- `build_id_map(Path("data"))` writes a validated `id_map.parquet`.
- `refresh_weekly_stats(Path("data"), seasons=[2014, ..., 2025])` writes per-season weekly stats to `data/raw/weekly_stats/season=YYYY/part.parquet`.
- `query(Path("data"), "SELECT * FROM weekly_stats WHERE position='WR' LIMIT 10")` runs SQL over the parquet partitions via DuckDB.
- The manifest at `data/manifests/ingest_manifest.parquet` reflects exactly what's on disk.

## What's still missing (next plan: "Projections Core — Ingest expansion + features")

- Schedules, snap_counts, depth_charts, NGS ingest paths.
- Per-position feature builders (`features/`).
- A few deferred pandera schemas (`SchedulesSchema`, `SnapCountsSchema`, etc.).
