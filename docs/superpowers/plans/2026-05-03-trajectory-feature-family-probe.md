# Trajectory Feature Family Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe whether the trajectory feature family (age, is_rookie, position-tailored volume trend, snap-pct change) carries orthogonal signal beyond v1 + PR #21 RB PBP cols at BaselineModel + lgb-nb composite.

**Architecture:** Mirrors PR #24 (`pbp_pressure_features.py` + override-builder script). Adds one ingest source (`import_draft_picks`) + one feature module (`trajectory_features.py`) + one CLI script. Probe-only — schema integration deferred to a SIGNAL-greenlit follow-up. Spec: `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md`.

**Tech Stack:** Python 3.12, pandas, pandera, pyarrow, nfl_data_py; pytest + mypy strict + ruff.

---

## File structure

**New files:**

| Path | Responsibility |
|---|---|
| `src/projections/ingest/draft_picks.py` | `refresh_draft_picks` — fetch + normalize + persist one parquet partition per season |
| `src/projections/features/trajectory_features.py` | 6 pure compute fns + attach helper + public assembler `build_trajectory_overrides` |
| `scripts/build_trajectory_override.py` | CLI: load weekly_stats / snap_counts / depth_charts / schedules / draft_picks, build override parquet |
| `tests/test_ingest/test_draft_picks.py` | Ingest unit tests (synthetic-fixture; no network) |
| `tests/test_features/test_trajectory_features.py` | Feature compute + attach + assembler tests |
| `tests/test_scripts/test_build_trajectory_override_cli.py` | CLI tests |

**Modified files:**

| Path | Change |
|---|---|
| `src/projections/schemas.py` | `+DraftPicksSchema` (sibling of `PbpSchema` etc.) |
| `src/projections/ingest/__init__.py` | `+refresh_draft_picks` export + `__all__` entry |
| `tests/test_ingest/test_api_drift.py` | `+test_draft_picks_api_columns_and_schema` (network smoke) |
| `CONTRIBUTING.md` | "Regenerating the trajectory override" subsection |

---

## Task ordering rationale

Each task closes one self-contained concern, ordered so later tasks build on tested earlier work:

1. **Schema first** (Task 1): `DraftPicksSchema` lands before any code that produces or validates against it.
2. **Ingest** (Tasks 2-4): module + wiring + network smoke. Once green, draft_picks parquets can be regenerated locally.
3. **Feature module skeleton + per-feature tasks** (Tasks 5-10): each compute fn lands with its own tests. Modules grow in isolation; no cross-task code needed.
4. **Composition** (Tasks 11-12): attach helper then public assembler — both depend on the per-feature compute fns.
5. **CLI + docs** (Tasks 13-14): runnable end-to-end.
6. **Probe execution + report** (Task 15): real-data validation + summary write-up.

---

### Task 1: Add `DraftPicksSchema` to `src/projections/schemas.py`

**Files:**
- Modify: `src/projections/schemas.py` (insert after `PbpSchema`, around line 670)
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Add to `tests/test_schemas.py` (or create a new test class if file is large):

```python
def test_draft_picks_schema_accepts_valid_row():
    from projections.schemas import DraftPicksSchema
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0033873"], dtype=pd.StringDtype("pyarrow")),
            "draft_year": pd.array([2017], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([1], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([10], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array(["MahoPa00"], dtype=pd.StringDtype("pyarrow")),
            "draft_age": pd.array([21.5], dtype=pd.Float64Dtype()),
        }
    )
    validated = DraftPicksSchema.validate(df)
    assert len(validated) == 1


def test_draft_picks_schema_rejects_duplicate_gsis_id():
    from projections.schemas import DraftPicksSchema
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0033873", "00-0033873"], dtype=pd.StringDtype("pyarrow")),
            "draft_year": pd.array([2017, 2018], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([1, 2], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([10, 50], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array([pd.NA, pd.NA], dtype=pd.StringDtype("pyarrow")),
            "draft_age": pd.array([21.5, 22.0], dtype=pd.Float64Dtype()),
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        DraftPicksSchema.validate(df)


def test_draft_picks_schema_rejects_malformed_gsis_id():
    from projections.schemas import DraftPicksSchema
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["malformed-id"], dtype=pd.StringDtype("pyarrow")),
            "draft_year": pd.array([2017], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([1], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([10], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array([pd.NA], dtype=pd.StringDtype("pyarrow")),
            "draft_age": pd.array([21.5], dtype=pd.Float64Dtype()),
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        DraftPicksSchema.validate(df)


def test_draft_picks_schema_allows_nullable_optional_columns():
    from projections.schemas import DraftPicksSchema
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0033873"], dtype=pd.StringDtype("pyarrow")),
            "draft_year": pd.array([2017], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array([pd.NA], dtype=pd.StringDtype("pyarrow")),
            "draft_age": pd.array([np.nan], dtype=pd.Float64Dtype()),
        }
    )
    DraftPicksSchema.validate(df)
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_schemas.py -v -k draft_picks
```

Expected: ImportError on `DraftPicksSchema` — schema not yet defined.

- [ ] **Step 3: Add `DraftPicksSchema` to `schemas.py`**

Insert after `PbpSchema` (search for `class PbpSchema(pa.DataFrameModel):` and add the new class after its closing `class Config: strict = "filter"`):

```python
class DraftPicksSchema(pa.DataFrameModel):
    """Per-player NFL draft pick metadata — what `ingest.draft_picks` produces.

    Snapshot semantics: a season's draft never changes after the draft completes.
    Source: `nfl_data_py.import_draft_picks`. UDFAs and pre-coverage players
    (drafts before 1980) are not present; downstream feature compute handles
    that with an inferred-draft-year fallback (see trajectory_features.py).
    """

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$", unique=True)
    draft_year: Series[int] = pa.Field(ge=1936, le=2100)
    draft_round: Series[int] = pa.Field(ge=1, le=15, nullable=True)
    draft_overall_pick: Series[int] = pa.Field(ge=1, le=500, nullable=True)
    pfr_id: Series[str] = pa.Field(nullable=True)
    draft_age: Series[float] = pa.Field(ge=18.0, le=40.0, nullable=True)

    class Config:
        strict = "filter"
```

- [ ] **Step 4: Run tests to verify pass**

```
pytest tests/test_schemas.py -v -k draft_picks
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/schemas.py tests/test_schemas.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): add DraftPicksSchema"
```

---

### Task 2: Create `src/projections/ingest/draft_picks.py`

**Files:**
- Create: `src/projections/ingest/draft_picks.py`
- Create: `tests/test_ingest/test_draft_picks.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest/test_draft_picks.py`:

```python
"""Synthetic-fixture tests for the draft_picks ingest module.

No network calls — _fetch_raw_draft_picks is monkey-patched to return a
hand-crafted DataFrame mirroring nfl_data_py.import_draft_picks's output
shape (verified empirically: 36 columns, str gsis_id / pfr_player_id,
float64 age, int32 season/round/pick).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from projections.ingest.draft_picks import _normalize_one_season, refresh_draft_picks
from projections.schemas import DraftPicksSchema
from projections.store import read_partition


def _fake_raw(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        rows.append(
            {
                "season": s,
                "round": 1,
                "pick": 10,
                "team": "KC",
                "gsis_id": f"00-003{3000 + s}",
                "pfr_player_id": f"Pl{s}00",
                "pfr_player_name": "Test Player",
                "position": "QB",
                "age": 22.0,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_keeps_canonical_columns():
    raw = _fake_raw([2022])
    df = _normalize_one_season(raw)
    assert list(df.columns) == [
        "gsis_id",
        "draft_year",
        "draft_round",
        "draft_overall_pick",
        "pfr_id",
        "draft_age",
    ]


def test_normalize_drops_malformed_gsis_id_rows():
    raw = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "round": [1, 1, 1],
            "pick": [10, 11, 12],
            "team": ["KC", "BUF", "MIA"],
            "gsis_id": ["00-0033000", "malformed", None],
            "pfr_player_id": ["A", "B", "C"],
            "pfr_player_name": ["x", "y", "z"],
            "position": ["QB", "RB", "WR"],
            "age": [22.0, 23.0, 21.0],
        }
    )
    df = _normalize_one_season(raw)
    # Only the well-formed row survives.
    assert len(df) == 1
    assert df.iloc[0]["gsis_id"] == "00-0033000"


def test_normalize_handles_missing_pfr_id():
    raw = _fake_raw([2022])
    raw["pfr_player_id"] = None
    df = _normalize_one_season(raw)
    assert pd.isna(df.iloc[0]["pfr_id"])


def test_normalize_handles_missing_age():
    raw = _fake_raw([2022])
    raw["age"] = None
    df = _normalize_one_season(raw)
    assert pd.isna(df.iloc[0]["draft_age"])


def test_normalize_validates_against_schema():
    raw = _fake_raw([2018, 2019, 2020])
    df = _normalize_one_season(raw)
    DraftPicksSchema.validate(df)


def test_refresh_draft_picks_writes_one_partition_per_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(seasons: list[int]) -> pd.DataFrame:
        return _fake_raw(seasons)

    monkeypatch.setattr(
        "projections.ingest.draft_picks._fetch_raw_draft_picks", fake_fetch
    )
    written = refresh_draft_picks(tmp_path, seasons=[2018, 2019, 2020])
    assert len(written) == 3
    for s in (2018, 2019, 2020):
        df = read_partition(tmp_path / "raw", "draft_picks", season=s)
        assert len(df) == 1
        DraftPicksSchema.validate(df)


def test_refresh_draft_picks_idempotent_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(seasons: list[int]) -> pd.DataFrame:
        return _fake_raw(seasons)

    monkeypatch.setattr(
        "projections.ingest.draft_picks._fetch_raw_draft_picks", fake_fetch
    )
    refresh_draft_picks(tmp_path, seasons=[2018])
    refresh_draft_picks(tmp_path, seasons=[2018])
    df = read_partition(tmp_path / "raw", "draft_picks", season=2018)
    assert len(df) == 1


def test_refresh_draft_picks_empty_seasons_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(seasons: list[int]) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr(
        "projections.ingest.draft_picks._fetch_raw_draft_picks", fake_fetch
    )
    written = refresh_draft_picks(tmp_path, seasons=[])
    assert written == []
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_ingest/test_draft_picks.py -v
```

Expected: ImportError on `projections.ingest.draft_picks`.

- [ ] **Step 3: Create `src/projections/ingest/draft_picks.py`**

```python
"""Refresh per-season draft picks from `nfl_data_py.import_draft_picks`.

Writes one parquet partition per season (curated subset). Snapshot
semantics — a season's draft never changes after the draft completes,
so re-running a season overwrites that partition only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _PYARROW_STR,
    DraftPicksSchema,
    GSIS_ID_PATTERN,
)
from projections.store import write_partition

_GSIS_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")


def _fetch_raw_draft_picks(seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around nfl_data_py; tests monkey-patch this."""
    if not seasons:
        return pd.DataFrame()
    return nfl.import_draft_picks(seasons)


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "draft_year": pd.array([], dtype=pd.Int64Dtype()),
                "draft_round": pd.array([], dtype=pd.Int64Dtype()),
                "draft_overall_pick": pd.array([], dtype=pd.Int64Dtype()),
                "pfr_id": pd.array([], dtype=_PYARROW_STR),
                "draft_age": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    df = raw.rename(
        columns={
            "season": "draft_year",
            "round": "draft_round",
            "pick": "draft_overall_pick",
            "pfr_player_id": "pfr_id",
            "age": "draft_age",
        }
    )

    # Filter rows without a valid gsis_id (older drafts may have nulls).
    df = df[df["gsis_id"].notna()].copy()
    df = df[df["gsis_id"].astype(str).str.match(_GSIS_RE)].copy()

    # Coerce dtypes: source returns int32 for season/round/pick and
    # float64 for age; pandera schema expects Int64/Float64 nullable types.
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["draft_year"] = df["draft_year"].astype(pd.Int64Dtype())
    df["draft_round"] = df["draft_round"].astype(pd.Int64Dtype())
    df["draft_overall_pick"] = df["draft_overall_pick"].astype(pd.Int64Dtype())
    df["pfr_id"] = df["pfr_id"].where(df["pfr_id"].notna(), other=pd.NA).astype(_PYARROW_STR)
    df["draft_age"] = df["draft_age"].astype(pd.Float64Dtype())

    df = df[
        [
            "gsis_id",
            "draft_year",
            "draft_round",
            "draft_overall_pick",
            "pfr_id",
            "draft_age",
        ]
    ].drop_duplicates(subset=["gsis_id"], keep="first").reset_index(drop=True)

    df = DraftPicksSchema.validate(df)
    return df


def refresh_draft_picks(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write draft pick data for each season.

    One partition per season. Idempotent — re-running a season overwrites
    that partition only.
    """
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_draft_picks([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "draft_picks", df, season=season, week=None)
        record_manifest(data_root, table="draft_picks", season=season, df=df)
        written.append(path)
    return written
```

- [ ] **Step 4: Run tests to verify pass**

```
pytest tests/test_ingest/test_draft_picks.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/ingest/draft_picks.py tests/test_ingest/test_draft_picks.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): refresh_draft_picks ingest module"
```

---

### Task 3: Wire `refresh_draft_picks` into the ingest layer + add network smoke

**Files:**
- Modify: `src/projections/ingest/__init__.py`
- Modify: `tests/test_ingest/test_api_drift.py`

- [ ] **Step 1: Update `src/projections/ingest/__init__.py`**

Read the current file first to confirm sort order, then add the new import + `__all__` entry alphabetically:

```python
"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.draft_picks import refresh_draft_picks
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import refresh_ngs
from projections.ingest.pbp import refresh_pbp
from projections.ingest.refresh import refresh
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "refresh",
    "refresh_depth_charts",
    "refresh_draft_picks",
    "refresh_ngs",
    "refresh_pbp",
    "refresh_schedules",
    "refresh_snap_counts",
    "refresh_weekly_stats",
]
```

- [ ] **Step 2: Add network smoke test**

Read the existing `tests/test_ingest/test_api_drift.py` to find the import block structure and the test pattern (e.g., `test_pbp_api_columns_and_schema`). Add the new smoke test at the bottom, mirroring that pattern:

```python
# Add to the existing import block (top of file):
from projections.ingest.draft_picks import (
    _fetch_raw_draft_picks,
)
from projections.ingest.draft_picks import (
    _normalize_one_season as _normalize_draft_picks,
)

# Add at the bottom of the file:
@pytest.mark.network
def test_draft_picks_api_columns_and_schema(tmp_path: Path) -> None:
    """Live network smoke for nfl_data_py.import_draft_picks.

    Asserts the source returns the columns we depend on with reasonable
    dtypes, then runs the normalize step end-to-end. Pandera surfaces any
    dtype drift after a nfl_data_py version bump.
    """
    raw = _fetch_raw_draft_picks([2023])
    expected_source_cols = {"season", "round", "pick", "gsis_id", "pfr_player_id", "age"}
    missing = expected_source_cols - set(raw.columns)
    assert not missing, f"missing source columns: {missing}"

    # Normalize end-to-end — pandera will throw on dtype/value drift.
    normalized = _normalize_draft_picks(raw)
    assert len(normalized) > 0
    assert set(normalized.columns) == {
        "gsis_id",
        "draft_year",
        "draft_round",
        "draft_overall_pick",
        "pfr_id",
        "draft_age",
    }
```

- [ ] **Step 3: Run smoke offline (network skipped) to ensure it imports**

```
pytest tests/test_ingest/test_api_drift.py -v
```

Expected: existing tests pass; the new test is **deselected** (no `--run-network` flag). No import errors.

- [ ] **Step 4: Optional — run with network if connectivity available**

```
pytest tests/test_ingest/test_api_drift.py -m network --run-network -k draft_picks -v
```

Expected: 1 passed (or skipped if no network). Optional — main verification gate doesn't depend on it.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/ingest/__init__.py tests/test_ingest/test_api_drift.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): wire refresh_draft_picks + network smoke"
```

---

### Task 4: Module skeleton for `trajectory_features.py` + `DraftLookup` type + active-game test fixture

**Files:**
- Create: `src/projections/features/trajectory_features.py` (skeleton only — populated incrementally)
- Create: `tests/test_features/test_trajectory_features.py` (test fixtures only)

- [ ] **Step 1: Create the test fixture file**

```python
"""Synthetic-fixture tests for trajectory_features.

Each compute fn is exercised against hand-rolled DataFrames; no real
weekly_stats / snap_counts / draft_picks parquets are read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.features.trajectory_features import (
    DraftLookup,
)


def _ws_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    position: str = "QB",
    team: str = "KC",
    opponent: str = "BUF",
    attempts: int = 30,
    completions: int = 20,
    sacks: int = 2,
    passing_yards: float = 250.0,
    passing_tds: int = 2,
    interceptions: int = 0,
    rushing_yards: float = 10.0,
    rushing_tds: int = 0,
    carries: int = 3,
    receptions: int = 0,
    receiving_yards: float = 0.0,
    receiving_tds: int = 0,
    receiving_air_yards: float = 0.0,
    targets: int = 0,
    fumbles_lost: int = 0,
) -> dict:
    """Helper: one weekly_stats row with sensible defaults."""
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "attempts": attempts,
        "completions": completions,
        "sacks": sacks,
        "passing_yards": passing_yards,
        "passing_tds": passing_tds,
        "interceptions": interceptions,
        "rushing_yards": rushing_yards,
        "rushing_tds": rushing_tds,
        "carries": carries,
        "receptions": receptions,
        "receiving_yards": receiving_yards,
        "receiving_tds": receiving_tds,
        "receiving_air_yards": receiving_air_yards,
        "targets": targets,
        "fumbles_lost": fumbles_lost,
    }


def _draft_lookup(*entries: tuple[str, int, float]) -> DraftLookup:
    return {gsis_id: (year, age) for gsis_id, year, age in entries}


def test_module_imports():
    """Smoke: confirm the module loads cleanly."""
    from projections.features import trajectory_features  # noqa: F401
```

- [ ] **Step 2: Run to confirm the test fails**

```
pytest tests/test_features/test_trajectory_features.py -v
```

Expected: ImportError on `projections.features.trajectory_features`.

- [ ] **Step 3: Create the module skeleton**

```python
"""Trajectory feature family — career-arc / role / volume-trend signals.

Probe-only at this stage: the override produced by build_trajectory_overrides
is consumed by scripts/probe_feature_signal.py via the standard --override
mechanism. Schema integration into per-position FeaturesSchemas is deferred
to a SIGNAL-greenlit follow-up.

Each compute_* function returns every (gsis_id, season[, week]) combo with
the feature value. The assembler merges all per-week feature frames onto
the player-team-week index in one pass.

Spec: docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.schemas import GSIS_ID_PATTERN, Position

# DraftLookup maps gsis_id -> (draft_year, draft_age). draft_age may be NaN
# (drafted-but-missing-age, rare). Missing key: UDFA / pre-coverage; falls
# back to inferred draft year from earliest weekly_stats appearance.
DraftLookup = dict[str, tuple[int, float]]

_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_AGE_OFFSET_FALLBACK: Final[float] = 22.0  # mean entry age for inferred path
```

- [ ] **Step 4: Run to confirm the smoke test passes**

```
pytest tests/test_features/test_trajectory_features.py -v
```

Expected: `test_module_imports` passes.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): module skeleton + DraftLookup type"
```

---

### Task 5: `compute_age` + tests

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_features/test_trajectory_features.py`:

```python
def test_compute_age_uses_draft_age_when_available():
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
            _ws_row(gsis_id="00-0033873", season=2018, week=2),
            _ws_row(gsis_id="00-0033873", season=2024, week=1),
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_age(weekly_stats, lookup)
    # One row per (gsis_id, season).
    assert len(out) == 2
    assert set(out.columns) == {"gsis_id", "season", "age", "draft_year_inferred"}
    age_2018 = out[out["season"] == 2018]["age"].iloc[0]
    age_2024 = out[out["season"] == 2024]["age"].iloc[0]
    assert age_2018 == pytest.approx(22.5)  # 21.5 + (2018 - 2017)
    assert age_2024 == pytest.approx(28.5)  # 21.5 + (2024 - 2017)
    assert (~out["draft_year_inferred"]).all()


def test_compute_age_falls_back_for_udfa():
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0099999", season=2020, week=1),
            _ws_row(gsis_id="00-0099999", season=2024, week=1),
        ]
    )
    # No entry in the lookup → UDFA path.
    lookup: DraftLookup = {}
    out = compute_age(weekly_stats, lookup)
    assert len(out) == 2
    age_2020 = out[out["season"] == 2020]["age"].iloc[0]
    age_2024 = out[out["season"] == 2024]["age"].iloc[0]
    # inferred_draft_year = 2020 (earliest); age = season - 2020 + 22.0
    assert age_2020 == pytest.approx(22.0)
    assert age_2024 == pytest.approx(26.0)
    assert out["draft_year_inferred"].all()


def test_compute_age_falls_back_when_draft_age_is_nan():
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
        ]
    )
    # Drafted but no draft_age — fall back to inferred path.
    lookup = _draft_lookup(("00-0033873", 2017, float("nan")))
    out = compute_age(weekly_stats, lookup)
    age_2018 = out[out["season"] == 2018]["age"].iloc[0]
    # inferred_draft_year = 2018 (earliest); 2018 - 2018 + 22 = 22.0
    assert age_2018 == pytest.approx(22.0)
    assert out["draft_year_inferred"].all()


def test_compute_age_one_row_per_player_season():
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=w) for w in range(1, 18)
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_age(weekly_stats, lookup)
    assert len(out) == 1


def test_compute_age_empty_input():
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        columns=["gsis_id", "season", "week", "position", "team", "opponent"]
    )
    lookup: DraftLookup = {}
    out = compute_age(weekly_stats, lookup)
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "age", "draft_year_inferred"}
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_age
```

Expected: ImportError on `compute_age`.

- [ ] **Step 3: Implement `compute_age`**

Append to `src/projections/features/trajectory_features.py`:

```python
def compute_age(
    weekly_stats: pd.DataFrame,
    draft_lookup: DraftLookup,
) -> pd.DataFrame:
    """Per-(player, season) biological age in the target season.

    Primary path: if gsis_id is in draft_lookup AND draft_age is finite,
    age = draft_age + (season - draft_year).

    Fallback path: missing key OR NaN draft_age → uses inferred_draft_year
    (earliest weekly_stats season for the player); age = season -
    inferred_draft_year + _AGE_OFFSET_FALLBACK. The draft_year_inferred
    column is True for these rows so the override audit can track fallback
    frequency.

    Output: (gsis_id, season, age, draft_year_inferred).
    One row per (player, season) where the player has at least one
    weekly_stats row that season.
    """
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "age": pd.array([], dtype=pd.Float64Dtype()),
                "draft_year_inferred": pd.array([], dtype=bool),
            }
        )

    # Earliest-season-played per player (for the fallback path).
    earliest = (
        weekly_stats.groupby("gsis_id", as_index=False, observed=True)["season"]
        .min()
        .rename(columns={"season": "inferred_draft_year"})
    )

    distinct = weekly_stats[["gsis_id", "season"]].drop_duplicates()
    merged = distinct.merge(earliest, on="gsis_id", how="left")

    def _age_row(row: pd.Series) -> tuple[float, bool]:
        entry = draft_lookup.get(row["gsis_id"])
        if entry is not None:
            draft_year, draft_age = entry
            if pd.notna(draft_age):
                return float(draft_age) + (int(row["season"]) - int(draft_year)), False
        # Fallback path.
        inferred = int(row["inferred_draft_year"])
        return float(row["season"]) - inferred + _AGE_OFFSET_FALLBACK, True

    age_inferred = merged.apply(_age_row, axis=1, result_type="expand")
    age_inferred.columns = ["age", "draft_year_inferred"]
    out = pd.concat([merged[["gsis_id", "season"]].reset_index(drop=True), age_inferred], axis=1)
    return out[["gsis_id", "season", "age", "draft_year_inferred"]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify pass**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_age
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): compute_age + tests"
```

---

### Task 6: `compute_is_rookie` + tests

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

```python
def test_compute_is_rookie_marks_drafted_player_in_draft_year():
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2017, week=1),
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_is_rookie(weekly_stats, lookup)
    assert len(out) == 2
    assert set(out.columns) == {"gsis_id", "season", "is_rookie"}
    rookie_2017 = out[out["season"] == 2017]["is_rookie"].iloc[0]
    rookie_2018 = out[out["season"] == 2018]["is_rookie"].iloc[0]
    assert rookie_2017 == 1.0
    assert rookie_2018 == 0.0


def test_compute_is_rookie_uses_inferred_year_for_udfa():
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0099999", season=2020, week=1),
            _ws_row(gsis_id="00-0099999", season=2021, week=1),
        ]
    )
    lookup: DraftLookup = {}
    out = compute_is_rookie(weekly_stats, lookup)
    rookie_2020 = out[out["season"] == 2020]["is_rookie"].iloc[0]
    rookie_2021 = out[out["season"] == 2021]["is_rookie"].iloc[0]
    assert rookie_2020 == 1.0
    assert rookie_2021 == 0.0


def test_compute_is_rookie_dtype_is_float():
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2017, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_is_rookie(weekly_stats, lookup)
    # Float (not bool) for ML-compat; matches schema dtype.
    assert out["is_rookie"].dtype == np.float64


def test_compute_is_rookie_empty_input():
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame(
        columns=["gsis_id", "season", "week", "position", "team", "opponent"]
    )
    out = compute_is_rookie(weekly_stats, {})
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "is_rookie"}
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_is_rookie
```

Expected: ImportError on `compute_is_rookie`.

- [ ] **Step 3: Implement `compute_is_rookie`**

```python
def compute_is_rookie(
    weekly_stats: pd.DataFrame,
    draft_lookup: DraftLookup,
) -> pd.DataFrame:
    """Per-(player, season) rookie flag (1.0 if season == draft_year, else 0.0).

    For UDFAs / missing-from-lookup, uses the same inferred_draft_year
    fallback as compute_age (earliest weekly_stats season).

    Output: (gsis_id, season, is_rookie) — one row per (player, season)
    where the player has at least one weekly_stats row. is_rookie is float64
    for ML-compat.
    """
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "is_rookie": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    earliest = (
        weekly_stats.groupby("gsis_id", as_index=False, observed=True)["season"]
        .min()
        .rename(columns={"season": "inferred_draft_year"})
    )
    distinct = weekly_stats[["gsis_id", "season"]].drop_duplicates()
    merged = distinct.merge(earliest, on="gsis_id", how="left")

    def _rookie_year(row: pd.Series) -> int:
        entry = draft_lookup.get(row["gsis_id"])
        if entry is not None:
            return int(entry[0])
        return int(row["inferred_draft_year"])

    rookie_years = merged.apply(_rookie_year, axis=1)
    out = pd.DataFrame(
        {
            "gsis_id": merged["gsis_id"].values,
            "season": merged["season"].values,
            "is_rookie": (merged["season"].values == rookie_years.values).astype(float),
        }
    )
    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_is_rookie
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): compute_is_rookie + tests"
```

---

### Task 7: `compute_qb_volume_trend` + tests (also defines the shared helper)

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

This task introduces `_volume_trend` — the shared private helper that all three position-specific volume trend fns delegate to (DRY). The QB compute fn is just `_volume_trend(weekly_stats, position="QB", value_col="attempts")`.

- [ ] **Step 1: Write failing tests**

```python
def test_compute_qb_volume_trend_basic_arithmetic():
    """Player has 8 prior games of QB attempts; week 9 reflects l4 - prior_l4."""
    from projections.features.trajectory_features import compute_qb_volume_trend

    rows = []
    # Weeks 1-4: attempts = 20, 22, 24, 26 (mean = 23)
    # Weeks 5-8: attempts = 30, 32, 34, 36 (mean = 33)
    # Week 9 row will get l4 = mean(weeks 5-8) = 33; prior_l4 = mean(weeks 1-4) = 23.
    # trend = 33 - 23 = 10.
    attempts_by_week = {1: 20, 2: 22, 3: 24, 4: 26, 5: 30, 6: 32, 7: 34, 8: 36, 9: 40}
    for w, att in attempts_by_week.items():
        rows.append(_ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", attempts=att))

    weekly_stats = pd.DataFrame(rows)
    out = compute_qb_volume_trend(weekly_stats)

    assert set(out.columns) == {"gsis_id", "season", "week", "volume_trend_l4_minus_prior_l4"}
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    assert len(week_9) == 1
    assert week_9["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(10.0)


def test_compute_qb_volume_trend_nan_before_8_prior_games():
    """Weeks 1-8 have fewer than 8 prior active games → NaN."""
    from projections.features.trajectory_features import compute_qb_volume_trend

    rows = [_ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", attempts=20 + w) for w in range(1, 10)]
    weekly_stats = pd.DataFrame(rows)
    out = compute_qb_volume_trend(weekly_stats)
    early = out[(out["season"] == 2018) & (out["week"] <= 8)]
    assert early["volume_trend_l4_minus_prior_l4"].isna().all()
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    assert week_9["volume_trend_l4_minus_prior_l4"].notna().all()


def test_compute_qb_volume_trend_filters_position():
    """Non-QB rows in weekly_stats must NOT contaminate QB rolling windows."""
    from projections.features.trajectory_features import compute_qb_volume_trend

    rows = []
    rows += [_ws_row(gsis_id="00-0099001", season=2018, week=w, position="QB", attempts=30) for w in range(1, 10)]
    rows += [_ws_row(gsis_id="00-0099002", season=2018, week=w, position="WR", attempts=0) for w in range(1, 10)]
    weekly_stats = pd.DataFrame(rows)
    out = compute_qb_volume_trend(weekly_stats)
    # Only the QB row appears.
    assert set(out["gsis_id"].unique()) == {"00-0099001"}


def test_compute_qb_volume_trend_crosses_season_boundary():
    """Week 1 of 2019 uses tail-end of 2018 for the trailing-N window."""
    from projections.features.trajectory_features import compute_qb_volume_trend

    rows = []
    # 8 games in 2018 (attempts 20, 22, 24, 26, 30, 32, 34, 36).
    rows += [
        _ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", attempts=a)
        for w, a in [(1, 20), (2, 22), (3, 24), (4, 26), (5, 30), (6, 32), (7, 34), (8, 36)]
    ]
    # Week 1 of 2019.
    rows.append(_ws_row(gsis_id="00-0033873", season=2019, week=1, position="QB", attempts=40))

    weekly_stats = pd.DataFrame(rows)
    out = compute_qb_volume_trend(weekly_stats)
    week1_2019 = out[(out["season"] == 2019) & (out["week"] == 1)]
    # l4 = mean(weeks 5-8 of 2018) = 33; prior_l4 = mean(weeks 1-4 of 2018) = 23.
    assert week1_2019["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(10.0)


def test_compute_qb_volume_trend_traded_player_unbroken_window():
    """A QB traded mid-season has rows on both teams; the rolling window
    groups by gsis_id and treats the career as unbroken."""
    from projections.features.trajectory_features import compute_qb_volume_trend

    rows = []
    # 4 games on team A (weeks 1-4, attempts 20, 22, 24, 26).
    rows += [
        _ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", team="KC", attempts=a)
        for w, a in [(1, 20), (2, 22), (3, 24), (4, 26)]
    ]
    # 4 games on team B (weeks 5-8, attempts 30, 32, 34, 36).
    rows += [
        _ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", team="BUF", attempts=a)
        for w, a in [(5, 30), (6, 32), (7, 34), (8, 36)]
    ]
    rows.append(_ws_row(gsis_id="00-0033873", season=2018, week=9, position="QB", team="BUF", attempts=40))
    weekly_stats = pd.DataFrame(rows)
    out = compute_qb_volume_trend(weekly_stats)
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    # Same arithmetic as the basic test — group is by gsis_id, not by team.
    assert week_9["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(10.0)


def test_compute_qb_volume_trend_empty_input():
    from projections.features.trajectory_features import compute_qb_volume_trend

    weekly_stats = pd.DataFrame(
        columns=["gsis_id", "season", "week", "position", "team", "opponent", "attempts"]
    )
    out = compute_qb_volume_trend(weekly_stats)
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "week", "volume_trend_l4_minus_prior_l4"}
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_qb_volume_trend
```

Expected: ImportError on `compute_qb_volume_trend`.

- [ ] **Step 3: Implement the shared helper + the QB variant**

Append to `src/projections/features/trajectory_features.py`:

```python
def _volume_trend(
    weekly_stats: pd.DataFrame,
    *,
    position: str | tuple[str, ...],
    value_col: str,
) -> pd.DataFrame:
    """Per-(player, season, week) volume trend on `value_col`, defined as
    mean over trailing-4 active games minus mean over prior-4 active games.

    Active game = game with a weekly_stats row for this player. Bye / IR /
    inactive weeks are not in weekly_stats and therefore excluded from the
    rolling denominator (treated as gaps, NOT as 0-value games).

    Within-player rolling: groups by gsis_id, sorts by (season, week). The
    trailing-4 window uses .rolling(4).mean().shift(1) — the row at week W
    reflects the mean over W-4..W-1 (NOT W). The prior-4 window uses
    .shift(5) — mean over W-8..W-5. Fewer than 8 prior active games yields
    NaN for prior_l4 (and therefore NaN for the trend).

    Position is a string or tuple of strings; rows whose position is not in
    that set are excluded before the rolling computation.
    """
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "week": pd.array([], dtype=pd.Int64Dtype()),
                "volume_trend_l4_minus_prior_l4": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    positions = (position,) if isinstance(position, str) else tuple(position)
    filtered = weekly_stats[weekly_stats["position"].isin(positions)].copy()
    if filtered.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "week": pd.array([], dtype=pd.Int64Dtype()),
                "volume_trend_l4_minus_prior_l4": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    sorted_df = filtered.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    grouped = sorted_df.groupby("gsis_id", sort=False)[value_col]
    l4 = grouped.transform(lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(1))
    prior_l4 = grouped.transform(
        lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(5)
    )
    sorted_df["volume_trend_l4_minus_prior_l4"] = l4 - prior_l4
    return sorted_df[["gsis_id", "season", "week", "volume_trend_l4_minus_prior_l4"]].reset_index(
        drop=True
    )


def compute_qb_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """QB volume trend on `attempts`, trailing-4 minus prior-4 (active games)."""
    return _volume_trend(weekly_stats, position="QB", value_col="attempts")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_qb_volume_trend
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): _volume_trend helper + compute_qb_volume_trend"
```

---

### Task 8: `compute_rb_volume_trend` + tests

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

```python
def test_compute_rb_volume_trend_uses_carries():
    from projections.features.trajectory_features import compute_rb_volume_trend

    rows = []
    carries_by_week = {1: 5, 2: 7, 3: 9, 4: 11, 5: 15, 6: 17, 7: 19, 8: 21, 9: 25}
    for w, c in carries_by_week.items():
        rows.append(
            _ws_row(gsis_id="00-0033873", season=2018, week=w, position="RB", carries=c)
        )
    weekly_stats = pd.DataFrame(rows)
    out = compute_rb_volume_trend(weekly_stats)
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    # l4 = mean(15,17,19,21) = 18; prior_l4 = mean(5,7,9,11) = 8; trend = 10.
    assert week_9["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(10.0)


def test_compute_rb_volume_trend_filters_to_rb_only():
    from projections.features.trajectory_features import compute_rb_volume_trend

    rows = []
    rows += [
        _ws_row(gsis_id="00-0099001", season=2018, week=w, position="RB", carries=10)
        for w in range(1, 10)
    ]
    rows += [
        _ws_row(gsis_id="00-0099002", season=2018, week=w, position="QB", carries=2)
        for w in range(1, 10)
    ]
    out = compute_rb_volume_trend(pd.DataFrame(rows))
    assert set(out["gsis_id"].unique()) == {"00-0099001"}
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_rb_volume_trend
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
def compute_rb_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """RB volume trend on `carries`, trailing-4 minus prior-4 (active games)."""
    return _volume_trend(weekly_stats, position="RB", value_col="carries")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_rb_volume_trend
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): compute_rb_volume_trend"
```

---

### Task 9: `compute_wr_te_volume_trend` + tests

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

```python
def test_compute_wr_te_volume_trend_uses_targets_for_wr():
    from projections.features.trajectory_features import compute_wr_te_volume_trend

    rows = []
    targets_by_week = {1: 4, 2: 6, 3: 8, 4: 10, 5: 12, 6: 14, 7: 16, 8: 18, 9: 20}
    for w, t in targets_by_week.items():
        rows.append(
            _ws_row(gsis_id="00-0033873", season=2018, week=w, position="WR", targets=t)
        )
    weekly_stats = pd.DataFrame(rows)
    out = compute_wr_te_volume_trend(weekly_stats)
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    # l4 = mean(12,14,16,18) = 15; prior_l4 = mean(4,6,8,10) = 7; trend = 8.
    assert week_9["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(8.0)


def test_compute_wr_te_volume_trend_includes_te():
    from projections.features.trajectory_features import compute_wr_te_volume_trend

    rows = []
    rows += [
        _ws_row(gsis_id="00-0099001", season=2018, week=w, position="WR", targets=10)
        for w in range(1, 10)
    ]
    rows += [
        _ws_row(gsis_id="00-0099002", season=2018, week=w, position="TE", targets=8)
        for w in range(1, 10)
    ]
    out = compute_wr_te_volume_trend(pd.DataFrame(rows))
    assert set(out["gsis_id"].unique()) == {"00-0099001", "00-0099002"}


def test_compute_wr_te_volume_trend_excludes_rb():
    from projections.features.trajectory_features import compute_wr_te_volume_trend

    rows = []
    rows += [
        _ws_row(gsis_id="00-0099001", season=2018, week=w, position="WR", targets=10)
        for w in range(1, 10)
    ]
    rows += [
        _ws_row(gsis_id="00-0099002", season=2018, week=w, position="RB", targets=4)
        for w in range(1, 10)
    ]
    out = compute_wr_te_volume_trend(pd.DataFrame(rows))
    assert set(out["gsis_id"].unique()) == {"00-0099001"}
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_wr_te_volume_trend
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
def compute_wr_te_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """WR/TE volume trend on `targets`, trailing-4 minus prior-4 (active games)."""
    return _volume_trend(weekly_stats, position=("WR", "TE"), value_col="targets")
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_wr_te_volume_trend
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): compute_wr_te_volume_trend"
```

---

### Task 10: `compute_snap_pct_change` + tests

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

```python
def _snap_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    position: str = "WR",
    team: str = "KC",
    opponent: str = "BUF",
    offense_snaps: int = 50,
    offense_pct: float = 0.7,
    defense_snaps: int = 0,
    defense_pct: float = 0.0,
    st_snaps: int = 0,
    st_pct: float = 0.0,
) -> dict:
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "offense_snaps": offense_snaps,
        "offense_pct": offense_pct,
        "defense_snaps": defense_snaps,
        "defense_pct": defense_pct,
        "st_snaps": st_snaps,
        "st_pct": st_pct,
    }


def test_compute_snap_pct_change_basic():
    from projections.features.trajectory_features import compute_snap_pct_change

    rows = []
    pct_by_week = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.7, 6: 0.7, 7: 0.7, 8: 0.7, 9: 0.7}
    for w, p in pct_by_week.items():
        rows.append(_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=p))
    snap_counts = pd.DataFrame(rows)
    out = compute_snap_pct_change(snap_counts)
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    # l4 mean = 0.7; prior_l4 mean = 0.5; change = 0.2.
    assert week_9["snap_pct_change_l4_vs_prior_l4"].iloc[0] == pytest.approx(0.2)


def test_compute_snap_pct_change_nan_before_8_prior_games():
    from projections.features.trajectory_features import compute_snap_pct_change

    rows = [_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.6) for w in range(1, 10)]
    snap_counts = pd.DataFrame(rows)
    out = compute_snap_pct_change(snap_counts)
    early = out[(out["season"] == 2018) & (out["week"] <= 8)]
    assert early["snap_pct_change_l4_vs_prior_l4"].isna().all()
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    assert week_9["snap_pct_change_l4_vs_prior_l4"].notna().all()


def test_compute_snap_pct_change_inactive_week_excluded_from_window():
    """A player who skips week 5 (no snap_counts row) has the window
    'shifted' — week 9's prior-4 includes weeks 1-4 and l4 includes 6-9."""
    from projections.features.trajectory_features import compute_snap_pct_change

    rows = []
    pct_by_week = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 6: 0.8, 7: 0.8, 8: 0.8, 9: 0.8, 10: 0.8}
    for w, p in pct_by_week.items():
        rows.append(_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=p))
    snap_counts = pd.DataFrame(rows)
    out = compute_snap_pct_change(snap_counts)
    # Row at week 10: l4 = mean(weeks 6,7,8,9) = 0.8; prior_l4 = mean(weeks 1-4) = 0.5; change = 0.3.
    week_10 = out[(out["season"] == 2018) & (out["week"] == 10)]
    assert week_10["snap_pct_change_l4_vs_prior_l4"].iloc[0] == pytest.approx(0.3)


def test_compute_snap_pct_change_crosses_season_boundary():
    from projections.features.trajectory_features import compute_snap_pct_change

    rows = []
    rows += [_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.5) for w in range(1, 5)]
    rows += [_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.7) for w in range(5, 9)]
    rows.append(_snap_row(gsis_id="00-0033873", season=2019, week=1, offense_pct=0.8))
    snap_counts = pd.DataFrame(rows)
    out = compute_snap_pct_change(snap_counts)
    week1_2019 = out[(out["season"] == 2019) & (out["week"] == 1)]
    # l4 = mean(weeks 5-8 of 2018) = 0.7; prior_l4 = mean(weeks 1-4 of 2018) = 0.5; change = 0.2.
    assert week1_2019["snap_pct_change_l4_vs_prior_l4"].iloc[0] == pytest.approx(0.2)


def test_compute_snap_pct_change_empty_input():
    from projections.features.trajectory_features import compute_snap_pct_change

    snap_counts = pd.DataFrame(
        columns=[
            "gsis_id", "season", "week", "team", "opponent", "position",
            "offense_snaps", "offense_pct", "defense_snaps", "defense_pct",
            "st_snaps", "st_pct",
        ]
    )
    out = compute_snap_pct_change(snap_counts)
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "week", "snap_pct_change_l4_vs_prior_l4"}
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_snap_pct_change
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
def compute_snap_pct_change(snap_counts: pd.DataFrame) -> pd.DataFrame:
    """Per-(player, season, week) change in offensive snap share, trailing-4
    minus prior-4 (active games — where active = has a snap_counts row).

    Output: (gsis_id, season, week, snap_pct_change_l4_vs_prior_l4).
    Players inactive that week (no snap_counts row) are skipped (not 0).
    """
    if snap_counts.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "week": pd.array([], dtype=pd.Int64Dtype()),
                "snap_pct_change_l4_vs_prior_l4": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    sorted_df = snap_counts.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    grouped = sorted_df.groupby("gsis_id", sort=False)["offense_pct"]
    l4 = grouped.transform(lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(1))
    prior_l4 = grouped.transform(
        lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(5)
    )
    sorted_df["snap_pct_change_l4_vs_prior_l4"] = l4 - prior_l4
    return sorted_df[
        ["gsis_id", "season", "week", "snap_pct_change_l4_vs_prior_l4"]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_features/test_trajectory_features.py -v -k compute_snap_pct_change
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): compute_snap_pct_change + cross-season test"
```

---

### Task 11: `attach_trajectory_features` joiner + tests

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

```python
def test_attach_trajectory_features_appends_4_cols_qb():
    from projections.features.trajectory_features import attach_trajectory_features

    weekly_stats = pd.DataFrame(
        [_ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", attempts=20 + w) for w in range(1, 10)]
    )
    snap_counts = pd.DataFrame(
        [_snap_row(gsis_id="00-0033873", season=2018, week=w, position="QB", offense_pct=0.5 + w * 0.01) for w in range(1, 10)]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    index = pd.DataFrame(
        [
            {"gsis_id": "00-0033873", "season": 2018, "week": 9, "team": "KC", "opp": "BUF"},
        ]
    )
    out = attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.QB)

    assert len(out) == 1
    expected_added = {
        "age",
        "is_rookie",
        "volume_trend_l4_minus_prior_l4",
        "snap_pct_change_l4_vs_prior_l4",
        "draft_year_inferred",
    }
    assert expected_added <= set(out.columns)


def test_attach_trajectory_features_uses_correct_volume_trend_per_position():
    from projections.features.trajectory_features import attach_trajectory_features

    # RB row at week 9 with carries trend.
    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=w, position="RB",
                    carries=c, attempts=0)
            for w, c in [(1, 5), (2, 7), (3, 9), (4, 11), (5, 15), (6, 17), (7, 19), (8, 21), (9, 25)]
        ]
    )
    snap_counts = pd.DataFrame(
        [_snap_row(gsis_id="00-0033873", season=2018, week=w, position="RB", offense_pct=0.6) for w in range(1, 10)]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 22.0))
    index = pd.DataFrame(
        [{"gsis_id": "00-0033873", "season": 2018, "week": 9, "team": "KC", "opp": "BUF"}]
    )
    out = attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.RB)
    # Carries trend at week 9 = 18 - 8 = 10.
    assert out["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(10.0)


def test_attach_trajectory_features_preserves_index_columns():
    from projections.features.trajectory_features import attach_trajectory_features

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1)])
    snap_counts = pd.DataFrame([_snap_row(gsis_id="00-0033873", season=2018, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    index = pd.DataFrame(
        [{"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF"}]
    )
    out = attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.QB)
    for col in ("gsis_id", "season", "week", "team", "opp"):
        assert col in out.columns


def test_attach_trajectory_features_rejects_invalid_position():
    from projections.features.trajectory_features import attach_trajectory_features

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1)])
    snap_counts = pd.DataFrame([_snap_row(gsis_id="00-0033873", season=2018, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    index = pd.DataFrame(
        [{"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF"}]
    )
    with pytest.raises(ValueError, match="position"):
        attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.K)
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k attach_trajectory_features
```

Expected: ImportError.

- [ ] **Step 3: Implement**

Add to `src/projections/features/trajectory_features.py` (also update the imports at the top to include `Position`):

```python
def attach_trajectory_features(
    index: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    draft_lookup: DraftLookup,
    position: Position,
) -> pd.DataFrame:
    """Append the 4 trajectory features (+ informational draft_year_inferred)
    to a player-team-week index for one position.

    Args:
        index: (gsis_id, season, week, team, opp) — one row per player-week.
        weekly_stats: full weekly_stats frame (multiple positions OK; the
            position-specific volume_trend filters internally).
        snap_counts: full snap_counts frame.
        draft_lookup: gsis_id -> (draft_year, draft_age) lookup.
        position: which position is being processed (selects volume_trend variant).

    Returns:
        A copy of `index` with 5 columns appended:
            age, is_rookie, volume_trend_l4_minus_prior_l4,
            snap_pct_change_l4_vs_prior_l4, draft_year_inferred.
        Row count equals len(index).

    Raises:
        ValueError: position not in {QB, RB, WR, TE}.
    """
    if position == Position.QB:
        trend = compute_qb_volume_trend(weekly_stats)
    elif position == Position.RB:
        trend = compute_rb_volume_trend(weekly_stats)
    elif position in (Position.WR, Position.TE):
        trend = compute_wr_te_volume_trend(weekly_stats)
    else:
        raise ValueError(f"unsupported position for trajectory features: {position!r}")

    age = compute_age(weekly_stats, draft_lookup)
    is_rookie = compute_is_rookie(weekly_stats, draft_lookup)
    snap_change = compute_snap_pct_change(snap_counts)

    out = index.merge(age, on=["gsis_id", "season"], how="left")
    out = out.merge(is_rookie, on=["gsis_id", "season"], how="left")
    out = out.merge(trend, on=["gsis_id", "season", "week"], how="left")
    out = out.merge(snap_change, on=["gsis_id", "season", "week"], how="left")

    if len(out) != len(index):
        raise AssertionError(
            f"row count mismatch in attach_trajectory_features: input {len(index)}, "
            f"output {len(out)}; suggests a many-to-many merge regression"
        )

    return out
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_features/test_trajectory_features.py -v -k attach_trajectory_features
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): attach_trajectory_features joiner + tests"
```

---

### Task 12: `build_trajectory_overrides` public assembler + tests

**Files:**
- Modify: `src/projections/features/trajectory_features.py`
- Modify: `tests/test_features/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

```python
def test_build_trajectory_overrides_dispatches_per_position():
    from projections.features.trajectory_features import build_trajectory_overrides

    weekly_stats = pd.DataFrame(
        [_ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", attempts=30) for w in range(1, 10)]
        + [_ws_row(gsis_id="00-0099001", season=2018, week=w, position="WR", targets=8) for w in range(1, 10)]
    )
    snap_counts = pd.DataFrame(
        [_snap_row(gsis_id=g, season=2018, week=w, position=pos, offense_pct=0.6)
         for g, pos in [("00-0033873", "QB"), ("00-0099001", "WR")]
         for w in range(1, 10)]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5), ("00-0099001", 2016, 22.0))
    index = pd.DataFrame(
        [
            {"gsis_id": "00-0033873", "season": 2018, "week": 9, "team": "KC", "opp": "BUF", "position": "QB"},
            {"gsis_id": "00-0099001", "season": 2018, "week": 9, "team": "BUF", "opp": "KC", "position": "WR"},
        ]
    )
    out = build_trajectory_overrides(weekly_stats, snap_counts, lookup, index)

    assert set(out.columns) == {
        "gsis_id",
        "season",
        "week",
        "age",
        "is_rookie",
        "volume_trend_l4_minus_prior_l4",
        "snap_pct_change_l4_vs_prior_l4",
        "draft_year_inferred",
    }
    assert len(out) == 2


def test_build_trajectory_overrides_rejects_malformed_gsis_id():
    from projections.features.trajectory_features import build_trajectory_overrides

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1)])
    snap_counts = pd.DataFrame([_snap_row(gsis_id="00-0033873", season=2018, week=1)])
    lookup: DraftLookup = {}
    bad_index = pd.DataFrame(
        [{"gsis_id": "malformed", "season": 2018, "week": 1, "team": "KC", "opp": "BUF", "position": "WR"}]
    )
    with pytest.raises(ValueError, match="gsis_id"):
        build_trajectory_overrides(weekly_stats, snap_counts, lookup, bad_index)


def test_build_trajectory_overrides_rejects_duplicate_index_keys():
    from projections.features.trajectory_features import build_trajectory_overrides

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1)])
    snap_counts = pd.DataFrame([_snap_row(gsis_id="00-0033873", season=2018, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    dup_index = pd.DataFrame(
        [
            {"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF", "position": "QB"},
            {"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF", "position": "QB"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_trajectory_overrides(weekly_stats, snap_counts, lookup, dup_index)


def test_build_trajectory_overrides_handles_missing_position_in_index():
    """An index row whose position is not in {QB,RB,WR,TE} is dropped silently
    (mirrors the behavior of the per-position feature builders)."""
    from projections.features.trajectory_features import build_trajectory_overrides

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1)])
    snap_counts = pd.DataFrame([_snap_row(gsis_id="00-0033873", season=2018, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    index = pd.DataFrame(
        [{"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF", "position": "K"}]
    )
    out = build_trajectory_overrides(weekly_stats, snap_counts, lookup, index)
    # K not in fantasy positions for trajectory probe — dropped.
    assert out.empty
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_features/test_trajectory_features.py -v -k build_trajectory_overrides
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
_FANTASY_POSITIONS_ENUM: Final[tuple[Position, ...]] = (
    Position.QB,
    Position.RB,
    Position.WR,
    Position.TE,
)


def build_trajectory_overrides(
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    draft_lookup: DraftLookup,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Public assembler. Returns the trajectory override frame ready to write.

    Args:
        weekly_stats: full multi-season weekly_stats frame matching
            WeeklyStatsSchema. Must include the season(s) covered by the
            index plus enough prior history for trailing-8-game windows.
        snap_counts: full multi-season snap_counts frame matching
            SnapCountsSchema. Same coverage requirement.
        draft_lookup: gsis_id -> (draft_year, draft_age) lookup. Missing
            keys route to inferred-draft-year fallback.
        player_team_week_index: (gsis_id, season, week, team, opp, position)
            — one row per player-week. Position values must be canonical
            Position enum string values.

    Returns:
        (gsis_id, season, week, age, is_rookie,
         volume_trend_l4_minus_prior_l4, snap_pct_change_l4_vs_prior_l4,
         draft_year_inferred) — one row per fantasy-position index row.
        Index rows with non-fantasy positions (K, DST, etc.) are dropped.

    Raises:
        ValueError: malformed gsis_id format or duplicate
            (gsis_id, season, week) keys in the index.
    """
    bad_ids = [
        g
        for g in player_team_week_index["gsis_id"].dropna()
        if not _GSIS_RE.match(str(g))
    ]
    if bad_ids:
        raise ValueError(
            f"invalid gsis_id format(s): {bad_ids[:3]} (and {max(0, len(bad_ids) - 3)} more)"
        )

    dup_mask = player_team_week_index.duplicated(
        subset=["gsis_id", "season", "week"], keep=False
    )
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(f"duplicate (gsis_id, season, week) keys in index: {n_dup} rows")

    chunks: list[pd.DataFrame] = []
    for pos in _FANTASY_POSITIONS_ENUM:
        idx_pos = player_team_week_index[player_team_week_index["position"] == pos.value]
        if idx_pos.empty:
            continue
        chunk = attach_trajectory_features(idx_pos, weekly_stats, snap_counts, draft_lookup, pos)
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "gsis_id",
                "season",
                "week",
                "age",
                "is_rookie",
                "volume_trend_l4_minus_prior_l4",
                "snap_pct_change_l4_vs_prior_l4",
                "draft_year_inferred",
            ]
        )

    out = pd.concat(chunks, ignore_index=True)
    return out[
        [
            "gsis_id",
            "season",
            "week",
            "age",
            "is_rookie",
            "volume_trend_l4_minus_prior_l4",
            "snap_pct_change_l4_vs_prior_l4",
            "draft_year_inferred",
        ]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_features/test_trajectory_features.py -v -k build_trajectory_overrides
```

Expected: 4 passed.

- [ ] **Step 5: Run full test suite, mypy, ruff**

```
pytest tests/test_features/test_trajectory_features.py tests/test_ingest/test_draft_picks.py tests/test_schemas.py -v
mypy src/projections/features/trajectory_features.py src/projections/ingest/draft_picks.py src/projections/schemas.py
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add src/projections/features/trajectory_features.py tests/test_features/test_trajectory_features.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): build_trajectory_overrides public assembler"
```

---

### Task 13: Override-builder script + CLI tests

**Files:**
- Create: `scripts/build_trajectory_override.py`
- Create: `tests/test_scripts/test_build_trajectory_override_cli.py`

- [ ] **Step 1: Write failing tests**

```python
"""CLI tests for scripts.build_trajectory_override.

Mirrors tests/test_scripts/test_build_pbp_pressure_override_cli.py.
Real data is monkey-patched out — tests assert argparse + main()'s
file-write contract, not the feature math (which has its own tests).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.build_trajectory_override import _parse_season_range, main


def test_parse_season_range_dash():
    assert _parse_season_range("2018-2024") == range(2018, 2025)


def test_parse_season_range_single():
    assert _parse_season_range("2024") == range(2024, 2025)


def test_main_rejects_existing_output_without_force(tmp_path: Path):
    output = tmp_path / "trajectory.parquet"
    output.write_bytes(b"placeholder")
    with pytest.raises(SystemExit) as exc:
        main(["--seasons", "2024", "--data-root", str(tmp_path), "--output", str(output)])
    assert exc.value.code != 0


def test_main_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: monkey-patch _read_concat to feed synthetic frames; assert
    main() writes a parquet with the expected schema."""
    output = tmp_path / "trajectory.parquet"

    def fake_read_concat(raw_root: Path, table: str, seasons: Sequence[int]) -> pd.DataFrame:
        if table == "weekly_stats":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "season": 2024,
                        "week": 1,
                        "position": "QB",
                        "team": "KC",
                        "opponent": "BUF",
                        "attempts": 30,
                        "completions": 20,
                        "sacks": 2,
                        "passing_yards": 250.0,
                        "passing_tds": 2,
                        "interceptions": 0,
                        "rushing_yards": 10.0,
                        "rushing_tds": 0,
                        "carries": 3,
                        "receptions": 0,
                        "receiving_yards": 0.0,
                        "receiving_tds": 0,
                        "receiving_air_yards": 0.0,
                        "targets": 0,
                        "fumbles_lost": 0,
                    }
                ]
            )
        if table == "snap_counts":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "season": 2024,
                        "week": 1,
                        "team": "KC",
                        "opponent": "BUF",
                        "position": "QB",
                        "offense_snaps": 50,
                        "offense_pct": 0.7,
                        "defense_snaps": 0,
                        "defense_pct": 0.0,
                        "st_snaps": 0,
                        "st_pct": 0.0,
                    }
                ]
            )
        if table == "depth_charts":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "season": 2024,
                        "week": 1,
                        "team": "KC",
                        "position": "QB",
                        "depth_team": "QB1",
                        "depth_rank": 1,
                    }
                ]
            )
        if table == "schedules":
            return pd.DataFrame(
                [
                    {
                        "season": 2024,
                        "week": 1,
                        "game_id": "2024_01_BUF_KC",
                        "home_team": "KC",
                        "away_team": "BUF",
                        "kickoff": pd.Timestamp("2024-09-05", tz="UTC"),
                        "spread_line": -3.5,
                        "total_line": 47.5,
                        "home_moneyline": -180,
                        "away_moneyline": 160,
                        "surface": "grass",
                        "roof": "outdoors",
                        "temp": 70,
                        "wind": 5,
                    }
                ]
            )
        if table == "draft_picks":
            return pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0033873",
                        "draft_year": 2017,
                        "draft_round": 1,
                        "draft_overall_pick": 10,
                        "pfr_id": "MahoPa00",
                        "draft_age": 21.5,
                    }
                ]
            )
        raise FileNotFoundError(table)

    monkeypatch.setattr("scripts.build_trajectory_override._read_concat", fake_read_concat)

    rc = main(["--seasons", "2024", "--data-root", str(tmp_path), "--output", str(output)])
    assert rc == 0
    assert output.exists()
    written = pd.read_parquet(output)
    assert set(written.columns) >= {
        "gsis_id",
        "season",
        "week",
        "age",
        "is_rookie",
        "volume_trend_l4_minus_prior_l4",
        "snap_pct_change_l4_vs_prior_l4",
        "draft_year_inferred",
    }
    assert len(written) == 1
```

- [ ] **Step 2: Run tests to verify failure**

```
pytest tests/test_scripts/test_build_trajectory_override_cli.py -v
```

Expected: ImportError on `scripts.build_trajectory_override`.

- [ ] **Step 3: Create the CLI script**

```python
"""Build the trajectory override parquet for the trajectory family probe.

One-shot CLI. Loads weekly_stats / snap_counts / depth_charts / schedules /
draft_picks across the requested season range (plus the prior season for
weekly_stats / snap_counts at week 1 trailing-8-game backfill), calls
build_trajectory_overrides, writes the resulting frame to a parquet.

Output is NOT committed — it's regenerable from the live raw partitions.

Usage:
    python -m scripts.build_trajectory_override --seasons 2018-2024
    python -m scripts.build_trajectory_override --seasons 2018-2024 --force
    python -m scripts.build_trajectory_override --output data/features_probe/x.parquet

Spec: docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.features.trajectory_features import (
    DraftLookup,
    build_trajectory_overrides,
)
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_OUTPUT = Path("data/features_probe/trajectory.parquet")


def _parse_season_range(s: str) -> range:
    """`'2018-2024'` -> `range(2018, 2025)`; `'2024'` -> `range(2024, 2025)`."""
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return range(lo, hi + 1)
    n = int(s)
    return range(n, n + 1)


def _read_concat(raw_root: Path, table: str, seasons: Sequence[int]) -> pd.DataFrame:
    """Read one partition per season and concat. Skip seasons without a partition."""
    frames: list[pd.DataFrame] = []
    for s in seasons:
        try:
            frames.append(read_partition(raw_root, table, season=s))
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError(
            f"no partitions found for table={table!r} in seasons={list(seasons)}"
        )
    return pd.concat(frames, ignore_index=True)


_FANTASY_POSITIONS: tuple[str, ...] = tuple(
    p.value for p in (Position.QB, Position.RB, Position.WR, Position.TE)
)


def _build_player_team_week_index(
    depth_charts: pd.DataFrame, schedules: pd.DataFrame, seasons: range
) -> pd.DataFrame:
    """Inner-join depth_charts (filtered to fantasy positions) with schedules
    to produce (gsis_id, season, week, team, opp, position).

    Mirrors PR #24's helper, with `position` preserved (the trajectory
    assembler dispatches per-position volume_trend so it needs position on
    every row).
    """
    dc = depth_charts[
        depth_charts["season"].isin(seasons) & depth_charts["position"].isin(_FANTASY_POSITIONS)
    ][["gsis_id", "season", "week", "team", "position"]].drop_duplicates(
        subset=["gsis_id", "season", "week"]
    )
    sch = schedules[schedules["season"].isin(seasons)][["season", "week", "home_team", "away_team"]]
    home = sch.rename(columns={"home_team": "team", "away_team": "opp"})
    away = sch.rename(columns={"away_team": "team", "home_team": "opp"})
    team_opp = pd.concat([home, away], ignore_index=True)[["season", "week", "team", "opp"]]
    return dc.merge(team_opp, on=["season", "week", "team"], how="inner")


def _build_draft_lookup(draft_picks: pd.DataFrame) -> DraftLookup:
    """draft_picks DataFrame -> {gsis_id: (draft_year, draft_age)}."""
    return {
        str(row["gsis_id"]): (int(row["draft_year"]), float(row["draft_age"]) if pd.notna(row["draft_age"]) else float("nan"))
        for _, row in draft_picks.iterrows()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument(
        "--seasons",
        type=_parse_season_range,
        default=range(2018, 2025),
        help="Season range, e.g. '2018-2024' or '2024'. Default: 2018-2024.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root for raw and features partitions. Default: data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Override output parquet path. Default: {_DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output if it already exists.",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite.")

    seasons: range = args.seasons
    raw_root = args.data_root / "raw"
    history_seasons = range(seasons.start - 1, seasons.stop)  # +1 prior for trailing-8 backfill

    weekly_stats = _read_concat(raw_root, "weekly_stats", list(history_seasons))
    snap_counts = _read_concat(raw_root, "snap_counts", list(history_seasons))
    depth_charts = _read_concat(raw_root, "depth_charts", list(seasons))
    schedules = _read_concat(raw_root, "schedules", list(seasons))
    # draft_picks: we need rows for every drafted-from-coverage-horizon player
    # that might appear in the index. Read the full available range.
    try:
        draft_picks = _read_concat(raw_root, "draft_picks", list(range(1980, seasons.stop)))
    except FileNotFoundError:
        draft_picks = pd.DataFrame(
            columns=["gsis_id", "draft_year", "draft_round", "draft_overall_pick", "pfr_id", "draft_age"]
        )

    idx = _build_player_team_week_index(depth_charts, schedules, seasons)
    draft_lookup = _build_draft_lookup(draft_picks)
    overrides = build_trajectory_overrides(weekly_stats, snap_counts, draft_lookup, idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.output, index=False)
    print(f"wrote {len(overrides)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_scripts/test_build_trajectory_override_cli.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run mypy + ruff on the new files**

```
mypy src/projections/features/trajectory_features.py scripts/build_trajectory_override.py
ruff check src/projections/features/trajectory_features.py scripts/build_trajectory_override.py tests/test_features/test_trajectory_features.py tests/test_scripts/test_build_trajectory_override_cli.py
ruff format --check src tests scripts
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add scripts/build_trajectory_override.py tests/test_scripts/test_build_trajectory_override_cli.py
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "feat(trajectory-probe): override-builder script + CLI tests"
```

---

### Task 14: Add CONTRIBUTING.md "Regenerating the trajectory override" subsection

**Files:**
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: Find the existing "Regenerating the PBP pressure override" subsection**

```
grep -n "Regenerating the PBP pressure override" CONTRIBUTING.md
```

Expected: returns one line number; that's the section to mirror.

- [ ] **Step 2: Add the new subsection immediately after it**

Insert (use the line number from step 1 to find the end of that subsection):

```markdown
### Regenerating the trajectory override

The trajectory family probe (added 2026-05-03) bundles 4 player-trajectory
features under one override parquet. To regenerate:

```bash
python -m scripts.build_trajectory_override --seasons 2018-2024
```

Output: `data/features_probe/trajectory.parquet`. Add `--force` to overwrite
an existing file. Requires `weekly_stats`, `snap_counts`, `depth_charts`,
`schedules`, and `draft_picks` raw partitions to be present in
`data/raw/`. The override script reads `draft_picks` for the full
1980+ range so the script can resolve UDFA / pre-coverage fallbacks.

Spec: `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md`.
```
```

- [ ] **Step 3: Verify the file lints**

```
ruff format --check CONTRIBUTING.md  # no-op for non-Python; sanity only
```

(CONTRIBUTING.md is not formatted by ruff; this step is a sanity check that nothing broke.)

- [ ] **Step 4: Commit**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add CONTRIBUTING.md
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "docs(contributing): add 'Regenerating the trajectory override' subsection"
```

---

### Task 15: Run probes + write summary report

**Files:**
- Create: `reports/feature_probe_trajectory_summary.md`
- Create: 4× `reports/feature_probe_trajectory_{,lgbnb_}{augment,swap}.{md,csv}`

This task is the executable validation phase. It assumes `data/raw/draft_picks/` partitions exist; if not, run the ingest first.

- [ ] **Step 1: Refresh draft_picks partitions (if not already present)**

```
python -c "from projections.ingest.draft_picks import refresh_draft_picks; from pathlib import Path; refresh_draft_picks(Path('data'), seasons=range(1980, 2025))"
```

Expected: writes `data/raw/draft_picks/season=YYYY/part.parquet` for each year. ~10-30 seconds for the full range.

- [ ] **Step 2: Generate the trajectory override**

```
python -m scripts.build_trajectory_override --seasons 2018-2024
```

Expected: prints "wrote N rows to data/features_probe/trajectory.parquet" where N is on the order of 100k-200k. ~30-60 seconds.

- [ ] **Step 3: Run the 4 probes**

```
python scripts/probe_feature_signal.py \
    --override data/features_probe/trajectory.parquet \
    --override-cols age is_rookie volume_trend_l4_minus_prior_l4 snap_pct_change_l4_vs_prior_l4 \
    --mode augment --model baseline \
    --positions QB RB WR TE --seasons 2021 2022 2023 2024 \
    --output-md reports/feature_probe_trajectory_augment.md \
    --output-csv reports/feature_probe_trajectory_augment.csv

python scripts/probe_feature_signal.py \
    --override data/features_probe/trajectory.parquet \
    --override-cols age is_rookie volume_trend_l4_minus_prior_l4 snap_pct_change_l4_vs_prior_l4 \
    --mode swap --model baseline \
    --positions QB RB WR TE --seasons 2021 2022 2023 2024 \
    --output-md reports/feature_probe_trajectory_swap.md \
    --output-csv reports/feature_probe_trajectory_swap.csv

python scripts/probe_feature_signal.py \
    --override data/features_probe/trajectory.parquet \
    --override-cols age is_rookie volume_trend_l4_minus_prior_l4 snap_pct_change_l4_vs_prior_l4 \
    --mode augment --model lgbnb --force-composite \
    --positions QB RB WR TE --seasons 2021 2022 2023 2024 \
    --output-md reports/feature_probe_trajectory_lgbnb_augment.md \
    --output-csv reports/feature_probe_trajectory_lgbnb_augment.csv

python scripts/probe_feature_signal.py \
    --override data/features_probe/trajectory.parquet \
    --override-cols age is_rookie volume_trend_l4_minus_prior_l4 snap_pct_change_l4_vs_prior_l4 \
    --mode swap --model lgbnb --force-composite \
    --positions QB RB WR TE --seasons 2021 2022 2023 2024 \
    --output-md reports/feature_probe_trajectory_lgbnb_swap.md \
    --output-csv reports/feature_probe_trajectory_lgbnb_swap.csv
```

Expected: each baseline run completes in ~1-2 minutes; each lgb-nb run ~10-15 minutes. If a run fails on coverage (`coverage<0.95`), retry with `--coverage-threshold 0.90` per spec §1.3 fallback.

Verify the actual `--override-cols` / `--output-md` / `--output-csv` flag names by running `python scripts/probe_feature_signal.py --help` first — flag names may differ slightly from the above (mirror PR #24's invocation; commands here are illustrative).

- [ ] **Step 4: Write the summary report**

Create `reports/feature_probe_trajectory_summary.md` mirroring `reports/feature_probe_pbp_pressure_summary.md`:
- Decision log (commits + reasoning).
- Per-mode verdict table (Phase 1 SIGNAL count + Phase 2 ADOPT/MARGINAL/DO_NOT_ADOPT).
- Mechanism annotation (which feature drove signal, or which `target_stat` cells regressed).
- Coverage note + threshold relaxation if applied.
- Refined-unit candidates left unexplored.

Read `reports/feature_probe_pbp_pressure_summary.md` first to use it as a template.

- [ ] **Step 5: Commit reports**

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add reports/feature_probe_trajectory*.md reports/feature_probe_trajectory*.csv
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "report(trajectory-probe): family summary — verdict <verdict>"
```

(Replace `<verdict>` with the actual outcome — `SIGNAL`, `NULL (durable)`, etc.)

- [ ] **Step 6: Final verification gate**

Per CLAUDE.md "End-of-effort checklist":

```
pytest -v -k "trajectory or draft_picks or schemas"
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green. If any fail, fix before declaring the task complete and creating the PR.

- [ ] **Step 7: Update project_management.md + TODO.md**

- Add a new top-of-file entry to `project_management.md` mirroring the "PBP Pressure Family Probe" entry's structure: status, verdict, what this closes, refined-unit candidates left open, coverage note, reports.
- Update `TODO.md` #3c (or wherever trajectory/weather are tracked) to reflect that trajectory is closed at this unit; flag the sibling weather probe as the next slot.

```
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory add project_management.md TODO.md
git -C C:/Users/alden/FantasyFootball/.worktrees/feat-probe-trajectory commit -m "docs(pm): record trajectory family probe verdict"
```

---

## Self-review checklist (post-plan, before handoff)

**Spec coverage:**
- §1.3 success criteria 1 (coverage ≥95%): Task 15 step 3 (run probe; relax to 0.90 if needed).
- §1.3 success criteria 2 (4 reports): Task 15 step 3.
- §1.3 success criteria 3 (lgb-nb composite via --force-composite): Task 15 step 3 lgb-nb runs.
- §1.3 success criteria 4 (mypy / ruff / pytest clean): Task 15 step 6.
- §1.4 out-of-scope: weather sibling probe, schema integration — both are deferred per the spec, no plan task for them.
- §2 ingest extension: Tasks 1, 2, 3.
- §2.5 UDFA fallback: covered in Task 5 (compute_age fallback path), Task 6 (compute_is_rookie fallback path).
- §3.1-3.4 feature definitions: Tasks 5, 6, 7, 8, 9, 10.
- §4.1 new files: Tasks 1-13.
- §4.2 modified files: Tasks 1, 3, 14.
- §4.3 interface: matches Task 11 (attach), Task 12 (build_trajectory_overrides).
- §5 probe protocol: Task 15.
- §6.1-6.5 testing strategy: Tasks 5-13 each include their tests.
- §6.6 verification gate: Task 12 step 5, Task 15 step 6.

**Type consistency:**
- `DraftLookup` defined in Task 4, used in Tasks 5-13.
- `Position` enum used consistently in Tasks 11-13.
- Compute fn signatures: per-(player, season) for age/is_rookie; per-(player, season, week) for volume_trend / snap_pct_change. Consistent across spec §4.3 and Tasks 5-10.
- Feature column names (`age`, `is_rookie`, `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`, `draft_year_inferred`): identical across Tasks 5, 6, 7, 10, 11, 12, 13.

**Placeholder scan:** none.
