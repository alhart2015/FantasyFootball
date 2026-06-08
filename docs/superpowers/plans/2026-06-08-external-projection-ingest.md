# External Projection Ingest Mechanism (v1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repeatable, dated-snapshot ingest of ESPN + Sleeper preseason projections into the store, canonically `gsis_id`-keyed (placeholder ids for pre-camp rookies), re-runnable weekly up to draft day.

**Architecture:** A new ingest source `src/projections/ingest/external_projections.py` (following the `refresh_<source>` template) writes one `ExternalProjectionSchema`-validated snapshot per pull, partitioned `season=YYYY/asof=YYYY-MM-DD/`. The store gains an optional `asof` partition dimension. Veterans get their real `gsis_id` via the `id_map` crosswalk; rookies (no `gsis_id` until ~late July) get a deterministic placeholder `99-XXXXXXX` flagged `is_placeholder_gsis`, auto-reconciling on later refreshes.

**Tech Stack:** Python 3, pandas, pandera, `urllib` (stdlib), pytest, mypy strict, ruff.

---

## Background facts the implementer needs (verified; do not re-derive)

- **Store** (`src/projections/store/parquet.py`): `write_partition(root, table, df, *, season, week)` and `read_partition(root, table, *, season=None, week=None)`; path layout `{table}/season=YYYY/week=WW/part.parquet`. `read_partition` with only `season` set already does `season_dir.rglob("part.parquet")` (recursive), so it will transparently read across `asof=*` subdirs once we nest them. Sanctioned I/O — no direct `df.to_parquet` from ingest.
- **Schemas** (`src/projections/schemas.py`): `_PYARROW_STR = pd.StringDtype("pyarrow")` (line 17); enums are `StrEnum` subclasses (`Position`, `Stat`, `DistributionFamily`); `GSIS_ID_PATTERN = r"\d{2}-\d{7}"` (line 194); `_POSITION_VALUES = [p.value for p in Position]` (line 255). Pandera schemas are `pa.DataFrameModel` with `Series[...] = pa.Field(...)` and `class Config: strict = "filter"`. Import `import pandera.pandas as pa` and `from pandera.typing import Series` the same way existing schemas do (copy the import block from the top of `schemas.py`).
- **id_map** (`data/raw/id_map.parquet`, produced by `src/projections/ingest/id_map.py`): columns `gsis_id, espn_id, sleeper_id, pfr_id, full_name, position, team`. **Defect:** `espn_id`/`sleeper_id` are persisted float-stringified (`'4374302.0'`) because the upstream column is float64; the persistence block is at `id_map.py` lines ~80-82 (`df[col] = df[col].where(df[col].notna(), other=pd.NA).astype(_PYARROW_STR)`).
- **Spike parsers** (`scripts/pull_external_projections.py`, on `main`): `ESPN_STAT_IDS`, `STAT_FIELDS` (the 9 scoring fields), `COUNT_FIELDS`, `round_count`, `espn_stats_to_statline_dict`, `parse_espn_players(payload, season)`, `fetch_espn`, `fetch_sleeper_season`, `parse_sleeper_adp`. Verified end-to-end against real 2024 + 2026 data. We **copy/adapt** these into the new `src/` module (the spike script stays as a historical artifact; retiring it is a future cleanup, out of scope).
- **The 9 stat fields:** `passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fumbles_lost`.
- **ESPN endpoint:** `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3?view=kona_player_info`, header `X-Fantasy-Filter: {"players":{"limit":800,"sortPercOwned":{"sortPriority":1,"sortAsc":false}}}`. ESPN preseason proj = `stats[]` entry with `seasonId==season, statSplitTypeId==0, statSourceId==1`. `defaultPositionId` 1=QB/2=RB/3=WR/4=TE. ADP = `ownership.averageDraftPosition`. Draft rank = `draftRanksByRankType.PPR.rank` (overall PPR rank — verify positional-vs-overall against data, store as-is).
- **Sleeper endpoint:** `https://api.sleeper.com/projections/nfl/{season}?season_type=regular` → list of items, each with `player_id`, `stats.adp_ppr`, and a `player` object carrying `first_name`, `last_name`, `position` (verified: present even for rookies). No stat line at the season level (ADP only).
- **datetime:** use `from datetime import UTC, date, datetime`; today's date (UTC) = `datetime.now(UTC).date()`.
- **No manifest:** unlike `schedules.refresh_schedules`, this ingest does NOT call `record_manifest` — the dated `asof` snapshots are themselves the provenance record. (Decision; keep v1 simple.)
- **Verification commands** (run from the worktree root): `pytest -v <subset>`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`, and `pytest -v -k "ingest or store or schemas"` for any schema/store/ingest change.

---

## Phase 1 — Store: optional `asof` partition

### Task 1: Add `asof` to the store helpers + `read_latest_partition`

**Files:**
- Modify: `src/projections/store/parquet.py`
- Test: `tests/test_store/test_parquet.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_store/test_parquet.py`; add `from datetime import date` to its imports)

```python
def test_write_read_asof_partition_roundtrip(tmp_path):
    from projections.store.parquet import read_partition, write_partition

    df = pd.DataFrame({"gsis_id": ["00-0000001"], "adp": [3.5]})
    p = write_partition(tmp_path, "external_projections", df, season=2026, asof=date(2026, 7, 15))
    assert p == tmp_path / "external_projections" / "season=2026" / "asof=2026-07-15" / "part.parquet"
    back = read_partition(tmp_path, "external_projections", season=2026, asof=date(2026, 7, 15))
    assert back["adp"].tolist() == [3.5]


def test_read_all_asof_snapshots_concatenates(tmp_path):
    from projections.store.parquet import read_partition, write_partition

    write_partition(tmp_path, "external_projections",
                    pd.DataFrame({"gsis_id": ["00-0000001"], "asof": ["2026-07-01"]}),
                    season=2026, asof=date(2026, 7, 1))
    write_partition(tmp_path, "external_projections",
                    pd.DataFrame({"gsis_id": ["00-0000001"], "asof": ["2026-07-15"]}),
                    season=2026, asof=date(2026, 7, 15))
    allrows = read_partition(tmp_path, "external_projections", season=2026)
    assert sorted(allrows["asof"].tolist()) == ["2026-07-01", "2026-07-15"]


def test_read_latest_partition_returns_newest_asof(tmp_path):
    from projections.store.parquet import read_latest_partition, write_partition

    write_partition(tmp_path, "external_projections",
                    pd.DataFrame({"gsis_id": ["00-0000001"], "adp": [9.0]}),
                    season=2026, asof=date(2026, 7, 1))
    write_partition(tmp_path, "external_projections",
                    pd.DataFrame({"gsis_id": ["00-0000001"], "adp": [4.0]}),
                    season=2026, asof=date(2026, 7, 15))
    latest = read_latest_partition(tmp_path, "external_projections", season=2026)
    assert latest["adp"].tolist() == [4.0]


def test_write_partition_season_week_unchanged(tmp_path):
    from projections.store.parquet import read_partition, write_partition

    df = pd.DataFrame({"x": [1]})
    write_partition(tmp_path, "weekly_stats", df, season=2024, week=3)
    assert read_partition(tmp_path, "weekly_stats", season=2024, week=3)["x"].tolist() == [1]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_store/test_parquet.py -k "asof or latest or season_week_unchanged" -v`
Expected: FAIL (`write_partition() got an unexpected keyword argument 'asof'` / `read_latest_partition` undefined).

- [ ] **Step 3: Implement** — rewrite `src/projections/store/parquet.py` to thread an optional `asof`:

```python
"""Parquet partitioned read/write helpers. Layout is `{table}/season=YYYY/week=WW/part.parquet`,
or `{table}/season=YYYY/asof=YYYY-MM-DD/part.parquet` for date-snapshotted tables.
Tables without season (e.g., id_map) are written to `{table}.parquet`."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd


def _partition_dir(
    root: Path, table: str, season: int | None, week: int | None, asof: date | None
) -> Path:
    if season is None:
        if week is not None or asof is not None:
            raise ValueError("week/asof cannot be set when season is None")
        return root / table
    p = root / table / f"season={season}"
    if week is not None:
        p = p / f"week={week:02d}"
    if asof is not None:
        p = p / f"asof={asof.isoformat()}"
    return p


def _partition_file(
    root: Path, table: str, season: int | None, week: int | None, asof: date | None
) -> Path:
    if season is None:
        return root / f"{table}.parquet"
    return _partition_dir(root, table, season, week, asof) / "part.parquet"


def write_partition(
    root: Path,
    table: str,
    df: pd.DataFrame,
    *,
    season: int | None,
    week: int | None = None,
    asof: date | None = None,
) -> Path:
    """Write `df` to the parquet partition for `(table, season, week, asof)`. Idempotent:
    removes the existing partition file first if present."""
    target = _partition_file(root, table, season, week, asof)
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
    asof: date | None = None,
) -> pd.DataFrame:
    """Read parquet partition(s). `season=None` reads the unpartitioned table file. With a
    season set: a specific `asof` (or `week`) reads that one partition; otherwise all
    `part.parquet` under the season (across week/asof subdirs) are concatenated."""
    if season is None:
        return pd.read_parquet(_partition_file(root, table, None, None, None))
    if asof is not None or week is not None:
        return pd.read_parquet(_partition_file(root, table, season, week, asof))
    season_dir = _partition_dir(root, table, season, None, None)
    files = sorted(season_dir.rglob("part.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet partitions under {season_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def read_latest_partition(root: Path, table: str, *, season: int) -> pd.DataFrame:
    """Read only the newest `asof` snapshot under a season (ISO dates sort chronologically)."""
    season_dir = _partition_dir(root, table, season, None, None)
    asof_dirs = sorted(d for d in season_dir.glob("asof=*") if d.is_dir())
    if not asof_dirs:
        raise FileNotFoundError(f"No asof snapshots under {season_dir}")
    return pd.read_parquet(asof_dirs[-1] / "part.parquet")


def delete_partition(
    root: Path, table: str, *, season: int | None, week: int | None = None, asof: date | None = None
) -> None:
    """Remove a partition directory or unpartitioned file. Used by tests and re-ingests."""
    if season is None:
        f = _partition_file(root, table, None, None, None)
        if f.exists():
            f.unlink()
        return
    target = _partition_dir(root, table, season, week, asof)
    if target.exists():
        shutil.rmtree(target)
```

- [ ] **Step 4: Export `read_latest_partition`** — in `src/projections/store/__init__.py`, add `read_latest_partition` to the imports and `__all__` alongside `read_partition`/`write_partition` (open the file, mirror the existing line).

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_store/test_parquet.py -v`
Expected: PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add src/projections/store/parquet.py src/projections/store/__init__.py tests/test_store/test_parquet.py
git commit -m "feat(store): optional asof date partition + read_latest_partition"
```

---

## Phase 2 — Schema + source enum

### Task 2: `ProjectionSource` enum + `ExternalProjectionSchema`

**Files:**
- Modify: `src/projections/schemas.py`
- Test: `tests/test_schemas/test_dataframe_schemas.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_schemas/test_dataframe_schemas.py`)

```python
def test_external_projection_schema_accepts_espn_and_sleeper_rows():
    import pandas as pd

    from projections.schemas import ExternalProjectionSchema, ProjectionSource

    df = pd.DataFrame(
        {
            "source": [ProjectionSource.ESPN.value, ProjectionSource.SLEEPER.value],
            "source_player_id": pd.array(["4374302", "6794"], dtype="string[pyarrow]"),
            "gsis_id": pd.array(["00-0036900", "99-0001234"], dtype="string[pyarrow]"),
            "is_placeholder_gsis": [False, True],
            "full_name": pd.array(["Ja'Marr Chase", "Some Rookie"], dtype="string[pyarrow]"),
            "position": pd.array(["WR", "RB"], dtype="string[pyarrow]"),
            "season": [2026, 2026],
            "asof": pd.array(["2026-07-15", "2026-07-15"], dtype="string[pyarrow]"),
            "adp": [4.8, 120.0],
            "espn_draft_rank": [20.0, None],  # nullable
            "passing_yards": [0.0, None],
            "passing_tds": [0.0, None],
            "interceptions": [0.0, None],
            "rushing_yards": [18.0, None],
            "rushing_tds": [0.0, None],
            "receptions": [105.0, None],
            "receiving_yards": [1335.0, None],
            "receiving_tds": [8.0, None],
            "fumbles_lost": [1.0, None],
        }
    )
    validated = ExternalProjectionSchema.validate(df)
    assert len(validated) == 2


def test_external_projection_schema_rejects_bad_gsis():
    import pandas as pd
    import pytest

    from projections.schemas import ExternalProjectionSchema

    bad = pd.DataFrame(
        {
            "source": ["ESPN"],
            "source_player_id": pd.array(["4374302"], dtype="string[pyarrow]"),
            "gsis_id": pd.array(["not-a-gsis"], dtype="string[pyarrow]"),
            "is_placeholder_gsis": [False],
            "full_name": pd.array(["X"], dtype="string[pyarrow]"),
            "position": pd.array(["WR"], dtype="string[pyarrow]"),
            "season": [2026],
            "asof": pd.array(["2026-07-15"], dtype="string[pyarrow]"),
            "adp": [4.8],
            "espn_draft_rank": [20.0],
            "passing_yards": [0.0], "passing_tds": [0.0], "interceptions": [0.0],
            "rushing_yards": [0.0], "rushing_tds": [0.0], "receptions": [0.0],
            "receiving_yards": [0.0], "receiving_tds": [0.0], "fumbles_lost": [0.0],
        }
    )
    with pytest.raises(Exception):
        ExternalProjectionSchema.validate(bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -k external_projection -v`
Expected: FAIL (`cannot import name 'ExternalProjectionSchema'`).

- [ ] **Step 3: Implement** — add to `src/projections/schemas.py`. Place the enum next to the other `StrEnum`s (after `Stat`), and the schema next to the other `pa.DataFrameModel`s (e.g., after `IdMapSchema`):

```python
class ProjectionSource(StrEnum):
    """External preseason projection sources. Use ProjectionSource.ESPN, never "ESPN"."""

    ESPN = "ESPN"
    SLEEPER = "SLEEPER"


_SOURCE_VALUES = [s.value for s in ProjectionSource]
```

```python
class ExternalProjectionSchema(pa.DataFrameModel):
    """One row per (source, player, season, asof) of external preseason projection data.

    Stat line is nullable: ESPN provides it; Sleeper provides ADP only (null stat line).
    gsis_id is the real id for crosswalked veterans, else a synthetic 99-XXXXXXX placeholder
    (flagged is_placeholder_gsis) for pre-camp rookies; source_player_id is the stable
    cross-snapshot join key.
    """

    source: Series[str] = pa.Field(isin=_SOURCE_VALUES)
    source_player_id: Series[str]
    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    is_placeholder_gsis: Series[bool]
    full_name: Series[str]
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    season: Series[int] = pa.Field(ge=1999, le=2100)
    asof: Series[str]  # ISO YYYY-MM-DD; also encoded in the partition path
    adp: Series[float] = pa.Field(nullable=True)
    espn_draft_rank: Series[float] = pa.Field(nullable=True)
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

(`espn_draft_rank` and the stat counts are stored as nullable float to avoid pandera nullable-`Int64` friction — ranks/counts like `20.0`/`105.0` are exact in float. `coerce = True` lets the empty-frame / mixed-construction cases validate; mirror whatever the nearest existing schema does.)

- [ ] **Step 4: Run test**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -k external_projection -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): ProjectionSource enum + ExternalProjectionSchema"
```

---

## Phase 3 — id_map float-id fix

### Task 3: Persist `espn_id`/`sleeper_id` as clean integer-strings

**Files:**
- Modify: `src/projections/ingest/id_map.py`
- Test: `tests/test_ingest/test_id_map.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_ingest/test_id_map.py`; check the file's existing helper for building a raw id_map frame and reuse it — if there's a `_normalize_*`/builder under test, call it; otherwise test the persisted-dtype behavior via the smallest public entry point the file already exercises)

```python
def test_float_valued_external_id_persists_without_trailing_dot_zero():
    # Upstream load_ff_playerids() returns espn_id/sleeper_id as float64 (NaNs force float),
    # so an integer id arrives as 4374302.0. It must persist as "4374302", not "4374302.0",
    # or the external-projection crosswalk join silently misses.
    import pandas as pd

    from projections.ingest.id_map import _coerce_external_id  # added in Step 3

    s = pd.Series([4374302.0, float("nan"), 6794.0])
    out = _coerce_external_id(s)
    assert out.tolist()[0] == "4374302"
    assert out.tolist()[2] == "6794"
    assert pd.isna(out.tolist()[1])
    assert str(out.dtype) == "string"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ingest/test_id_map.py -k trailing_dot_zero -v`
Expected: FAIL (`cannot import name '_coerce_external_id'`).

- [ ] **Step 3: Implement** — in `src/projections/ingest/id_map.py`, add a helper and use it in the persistence block. Add `_PYARROW_STR` is already imported there. Add the helper:

```python
def _coerce_external_id(s: pd.Series) -> pd.Series:
    """Persist an external id column (espn_id/sleeper_id) as a clean integer-string.
    Upstream returns these as float64 (NaNs force float dtype), so a plain .astype(str)
    yields '4374302.0'. Round-trip through nullable Int64 to drop the spurious '.0',
    keeping NaN as <NA>."""
    if pd.api.types.is_float_dtype(s):
        s = s.astype("Int64")
    return s.where(s.notna(), other=pd.NA).astype(_PYARROW_STR)
```

Then replace the existing loop:

```python
    for col in ("espn_id", "sleeper_id", "pfr_id"):
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), other=pd.NA).astype(_PYARROW_STR)
```

with:

```python
    for col in ("espn_id", "sleeper_id", "pfr_id"):
        if col in df.columns:
            df[col] = _coerce_external_id(df[col])
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ingest/test_id_map.py -v`
Expected: PASS (new + existing id_map tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/ingest/id_map.py tests/test_ingest/test_id_map.py
git commit -m "fix(ingest): persist espn_id/sleeper_id as clean integer-strings (no trailing .0)"
```

---

## Phase 4 — The external-projection ingest module

All tasks create/extend `src/projections/ingest/external_projections.py` and `tests/test_ingest/test_external_projections.py`.

### Task 4: Source parsers (ESPN + Sleeper, with name/position)

- [ ] **Step 1: Write the failing tests** (`tests/test_ingest/test_external_projections.py`, new file)

```python
import pandas as pd

from projections.ingest import external_projections as ext


def test_parse_espn_players_extracts_statline_adp_rank():
    payload = {
        "players": [
            {
                "player": {
                    "id": 4374302,
                    "fullName": "Ja'Marr Chase",
                    "defaultPositionId": 3,
                    "ownership": {"averageDraftPosition": 4.8},
                    "draftRanksByRankType": {"PPR": {"rank": 20}},
                    "stats": [
                        {
                            "seasonId": 2026,
                            "statSourceId": 1,
                            "statSplitTypeId": 0,
                            "stats": {"53": 105.0, "42": 1335.0, "43": 8.0},
                        }
                    ],
                }
            },
            {"player": {"id": 1, "defaultPositionId": 16, "stats": []}},  # DST -> dropped
        ]
    }
    df = ext.parse_espn_players(payload, season=2026)
    assert df["espn_id"].tolist() == ["4374302"]
    r = df.iloc[0]
    assert r["position"] == "WR" and r["full_name"] == "Ja'Marr Chase"
    assert r["espn_adp"] == 4.8 and r["espn_pos_rank"] == 20
    assert r["receptions"] == 105 and r["receiving_yards"] == 1335.0 and r["receiving_tds"] == 8


def test_parse_sleeper_projections_keeps_name_position_adp_filters_to_skill():
    payload = [
        {"player_id": "6794", "stats": {"adp_ppr": 14.5},
         "player": {"first_name": "A", "last_name": "B", "position": "WR"}},
        {"player_id": "99", "stats": {"adp_ppr": 1.0},
         "player": {"first_name": "K", "last_name": "K", "position": "K"}},  # kicker -> dropped
        {"player_id": None, "stats": {"adp_ppr": 9.0},
         "player": {"first_name": "x", "last_name": "y", "position": "RB"}},  # no id -> dropped
    ]
    df = ext.parse_sleeper_projections(payload)
    assert df["sleeper_id"].tolist() == ["6794"]
    r = df.iloc[0]
    assert r["full_name"] == "A B" and r["position"] == "WR" and r["sleeper_adp"] == 14.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_ingest/test_external_projections.py -v`
Expected: FAIL (`No module named 'projections.ingest.external_projections'`).

- [ ] **Step 3: Implement** — create `src/projections/ingest/external_projections.py`:

```python
"""Ingest source for external preseason projections (ESPN + Sleeper).

Repeatable, dated-snapshot ingest: each `refresh_external_projections(...)` writes one
`ExternalProjectionSchema` snapshot under data/raw/external_projections/season=YYYY/
asof=YYYY-MM-DD/. Veterans get their real gsis_id via the id_map crosswalk; pre-camp
rookies get a deterministic placeholder (99-XXXXXXX, flagged is_placeholder_gsis) that
auto-reconciles on later refreshes once id_map propagates the real id. Stat lines are
stored, not fantasy points (the scoring layer converts downstream).

Network-dependent; the pure parsers/normalizers are unit-tested with synthetic payloads.

Usage:
    python -m projections.ingest.external_projections --season 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from projections.schemas import (
    ExternalProjectionSchema,
    ProjectionSource,
    _PYARROW_STR,
)
from projections.store import read_partition, write_partition

_ESPN_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
    "{season}/segments/0/leaguedefaults/3?view=kona_player_info"
)
_SLEEPER_URL = "https://api.sleeper.com/projections/nfl/{season}?season_type=regular"
_UA = "Mozilla/5.0"

# ESPN numeric stat-id -> StatLine field (common scoring set). Verified against real data.
ESPN_STAT_IDS: dict[str, str] = {
    "3": "passing_yards",
    "4": "passing_tds",
    "20": "interceptions",
    "24": "rushing_yards",
    "25": "rushing_tds",
    "53": "receptions",
    "42": "receiving_yards",
    "43": "receiving_tds",
    "72": "fumbles_lost",
}
STAT_FIELDS: tuple[str, ...] = (
    "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds", "fumbles_lost",
)
_COUNT_FIELDS = frozenset(
    {"passing_tds", "interceptions", "rushing_tds", "receptions", "receiving_tds", "fumbles_lost"}
)
_ESPN_POSITIONS: dict[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
_SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


def _round_count(value: float) -> int:
    """Half-up rounding for non-negative projected count stats (Python's round() is banker's)."""
    return int(value + 0.5)


def _espn_stats_to_statline(stats: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {f: 0.0 for f in STAT_FIELDS}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in stats:
            val = float(stats[sid])
            out[field] = float(_round_count(val)) if field in _COUNT_FIELDS else val
    return out


def parse_espn_players(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Tidy one ESPN kona_player_info payload -> one row per QB/RB/WR/TE with a preseason
    projected stat line + espn_id + ADP + PPR draft rank."""
    rows: list[dict[str, object]] = []
    for entry in payload.get("players", []):
        pl = entry.get("player", {})
        position = _ESPN_POSITIONS.get(pl.get("defaultPositionId"))
        if position is None:
            continue
        espn_id = pl.get("id")
        if espn_id is None:
            continue
        proj_stats: dict[str, float] | None = None
        for s in pl.get("stats", []):
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
                continue
            if s.get("statSourceId") == 1:
                proj_stats = s.get("stats", {})
        if not proj_stats:
            continue
        ownership = pl.get("ownership") or {}
        ppr_rank = ((pl.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank")
        row: dict[str, object] = {
            "espn_id": str(espn_id),
            "full_name": pl.get("fullName"),
            "position": position,
            "espn_adp": ownership.get("averageDraftPosition"),
            "espn_pos_rank": ppr_rank,
        }
        row.update(_espn_stats_to_statline(proj_stats))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_sleeper_projections(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Tidy Sleeper season projections -> one row per QB/RB/WR/TE with sleeper_id + name +
    position + PPR ADP (Sleeper has no stat line at the season level)."""
    rows: list[dict[str, object]] = []
    for item in payload:
        pid = item.get("player_id")
        if pid is None:
            continue
        pl = item.get("player") or {}
        position = pl.get("position")
        if position not in _SKILL_POSITIONS:
            continue
        first = pl.get("first_name") or ""
        last = pl.get("last_name") or ""
        stats = item.get("stats") or {}
        rows.append(
            {
                "sleeper_id": str(pid),
                "full_name": f"{first} {last}".strip(),
                "position": position,
                "sleeper_adp": stats.get("adp_ppr"),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ingest/test_external_projections.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/ingest/external_projections.py tests/test_ingest/test_external_projections.py
git commit -m "feat(ingest): external_projections ESPN+Sleeper parsers (with name/position)"
```

---

### Task 5: Placeholder gsis + crosswalk

**Files:** Modify `src/projections/ingest/external_projections.py`; extend the test file.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_make_placeholder_gsis_is_deterministic_and_pattern_valid():
    import re

    a = ext._make_placeholder_gsis("ESPN", "5555")
    b = ext._make_placeholder_gsis("ESPN", "5555")
    assert a == b  # deterministic
    assert a.startswith("99-") and re.fullmatch(r"\d{2}-\d{7}", a)
    assert ext._make_placeholder_gsis("SLEEPER", "5555") != a  # source-scoped


def test_attach_gsis_id_real_for_matched_placeholder_for_rookie():
    df = pd.DataFrame({"espn_id": ["4374302", "9999999"], "x": [1, 2]})
    id_map = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
        }
    )
    out = ext._attach_gsis_id(df, id_map, source="ESPN", id_col="espn_id")
    veteran = out[out["espn_id"] == "4374302"].iloc[0]
    rookie = out[out["espn_id"] == "9999999"].iloc[0]
    assert veteran["gsis_id"] == "00-0036900"
    assert not bool(veteran["is_placeholder_gsis"])
    assert bool(rookie["is_placeholder_gsis"])
    assert rookie["gsis_id"] == ext._make_placeholder_gsis("ESPN", "9999999")
    assert len(out) == 2  # no row multiplication
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_ingest/test_external_projections.py -k "placeholder or attach_gsis" -v`
Expected: FAIL (`module ... has no attribute '_make_placeholder_gsis'`).

- [ ] **Step 3: Implement** (append to the module)

```python
def _make_placeholder_gsis(source: str, source_player_id: str) -> str:
    """Deterministic synthetic gsis_id for a player not in id_map (e.g., a pre-camp rookie).
    Matches GSIS_ID_PATTERN with a reserved 99- prefix. Source-scoped so an ESPN and a
    Sleeper id never collide into the same placeholder."""
    digest = hashlib.sha1(f"{source}:{source_player_id}".encode()).hexdigest()
    return f"99-{int(digest, 16) % 10_000_000:07d}"


def _attach_gsis_id(
    df: pd.DataFrame, id_map: pd.DataFrame, *, source: str, id_col: str
) -> pd.DataFrame:
    """Left-join df to id_map on `id_col` (espn_id/sleeper_id) to attach a real gsis_id;
    unmatched rows get a deterministic placeholder. Adds `gsis_id` + `is_placeholder_gsis`.
    Dedupes the crosswalk on `id_col` so a duplicate id_map mapping can't multiply rows."""
    crosswalk = (
        id_map[["gsis_id", id_col]].dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    )
    merged = df.merge(crosswalk, on=id_col, how="left")
    merged["is_placeholder_gsis"] = merged["gsis_id"].isna()
    placeholder = merged[id_col].map(lambda pid: _make_placeholder_gsis(source, pid))
    merged["gsis_id"] = merged["gsis_id"].where(merged["gsis_id"].notna(), placeholder)
    return merged
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ingest/test_external_projections.py -k "placeholder or attach_gsis" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/ingest/external_projections.py tests/test_ingest/test_external_projections.py
git commit -m "feat(ingest): deterministic placeholder gsis + id_map crosswalk"
```

---

### Task 6: Normalize to `ExternalProjectionSchema`

**Files:** Modify the module + test file.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _tiny_id_map():
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["6794"], dtype="string[pyarrow]"),
        }
    )


def test_espn_to_canonical_is_schema_valid_with_stat_line():
    from datetime import date

    from projections.schemas import ExternalProjectionSchema

    espn = ext.parse_espn_players(
        {
            "players": [
                {
                    "player": {
                        "id": 4374302, "fullName": "Ja'Marr Chase", "defaultPositionId": 3,
                        "ownership": {"averageDraftPosition": 4.8},
                        "draftRanksByRankType": {"PPR": {"rank": 20}},
                        "stats": [{"seasonId": 2026, "statSourceId": 1, "statSplitTypeId": 0,
                                   "stats": {"53": 105.0, "42": 1335.0, "43": 8.0}}],
                    }
                }
            ]
        },
        season=2026,
    )
    out = ext._espn_to_canonical(espn, season=2026, asof=date(2026, 7, 15), id_map=_tiny_id_map())
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "ESPN" and r["source_player_id"] == "4374302"
    assert r["gsis_id"] == "00-0036900" and r["asof"] == "2026-07-15"
    assert r["adp"] == 4.8 and r["receptions"] == 105.0


def test_sleeper_to_canonical_has_null_stat_line():
    from datetime import date

    from projections.schemas import ExternalProjectionSchema

    sl = ext.parse_sleeper_projections(
        [{"player_id": "6794", "stats": {"adp_ppr": 14.5},
          "player": {"first_name": "A", "last_name": "B", "position": "WR"}}]
    )
    out = ext._sleeper_to_canonical(sl, season=2026, asof=date(2026, 7, 15), id_map=_tiny_id_map())
    ExternalProjectionSchema.validate(out)
    r = out.iloc[0]
    assert r["source"] == "SLEEPER" and r["adp"] == 14.5
    assert pd.isna(r["receptions"]) and pd.isna(r["espn_draft_rank"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_ingest/test_external_projections.py -k "to_canonical" -v`
Expected: FAIL (`has no attribute '_espn_to_canonical'`).

- [ ] **Step 3: Implement** (append). The helper builds the canonical frame in the exact `ExternalProjectionSchema` shape (string dtypes per convention), then `strict="filter"` drops anything extra at the `refresh` boundary.

```python
_CANONICAL_STR_COLS = ("source", "source_player_id", "gsis_id", "full_name", "position", "asof")


def _finish_canonical(df: pd.DataFrame, *, season: int, asof: date) -> pd.DataFrame:
    df = df.copy()
    df["season"] = season
    df["asof"] = asof.isoformat()
    for c in _CANONICAL_STR_COLS:
        df[c] = df[c].astype(_PYARROW_STR)
    return df


def _espn_to_canonical(
    espn: pd.DataFrame, *, season: int, asof: date, id_map: pd.DataFrame
) -> pd.DataFrame:
    keyed = _attach_gsis_id(espn, id_map, source=ProjectionSource.ESPN.value, id_col="espn_id")
    out = pd.DataFrame(
        {
            "source": ProjectionSource.ESPN.value,
            "source_player_id": keyed["espn_id"],
            "gsis_id": keyed["gsis_id"],
            "is_placeholder_gsis": keyed["is_placeholder_gsis"],
            "full_name": keyed["full_name"],
            "position": keyed["position"],
            "adp": keyed["espn_adp"].astype(float),
            "espn_draft_rank": keyed["espn_pos_rank"].astype(float),
        }
    )
    for f in STAT_FIELDS:
        out[f] = keyed[f].astype(float)
    return _finish_canonical(out, season=season, asof=asof)


def _sleeper_to_canonical(
    sleeper: pd.DataFrame, *, season: int, asof: date, id_map: pd.DataFrame
) -> pd.DataFrame:
    keyed = _attach_gsis_id(
        sleeper, id_map, source=ProjectionSource.SLEEPER.value, id_col="sleeper_id"
    )
    out = pd.DataFrame(
        {
            "source": ProjectionSource.SLEEPER.value,
            "source_player_id": keyed["sleeper_id"],
            "gsis_id": keyed["gsis_id"],
            "is_placeholder_gsis": keyed["is_placeholder_gsis"],
            "full_name": keyed["full_name"],
            "position": keyed["position"],
            "adp": keyed["sleeper_adp"].astype(float),
            "espn_draft_rank": pd.NA,
        }
    )
    for f in STAT_FIELDS:
        out[f] = pd.NA
    return _finish_canonical(out, season=season, asof=asof)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ingest/test_external_projections.py -k "to_canonical" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/ingest/external_projections.py tests/test_ingest/test_external_projections.py
git commit -m "feat(ingest): normalize ESPN/Sleeper to ExternalProjectionSchema"
```

---

### Task 7: Fetchers, `refresh_external_projections`, CLI

**Files:** Modify the module + test file.

- [ ] **Step 1: Write the failing test** (append — pure-path test of the orchestrator with the network fetchers monkeypatched, writing to a tmp store)

```python
def test_refresh_writes_validated_asof_snapshot(tmp_path, monkeypatch):
    from datetime import date

    from projections.schemas import ExternalProjectionSchema
    from projections.store import read_latest_partition

    espn_payload = {
        "players": [
            {"player": {"id": 4374302, "fullName": "Ja'Marr Chase", "defaultPositionId": 3,
                        "ownership": {"averageDraftPosition": 4.8},
                        "draftRanksByRankType": {"PPR": {"rank": 20}},
                        "stats": [{"seasonId": 2026, "statSourceId": 1, "statSplitTypeId": 0,
                                   "stats": {"53": 105.0, "42": 1335.0, "43": 8.0}}]}}
        ]
    }
    sleeper_payload = [
        {"player_id": "6794", "stats": {"adp_ppr": 14.5},
         "player": {"first_name": "A", "last_name": "B", "position": "WR"}}
    ]
    monkeypatch.setattr(ext, "fetch_espn", lambda season: espn_payload)
    monkeypatch.setattr(ext, "fetch_sleeper_season", lambda season: sleeper_payload)
    # id_map lives at <data_root>/raw/id_map.parquet
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
            "espn_id": pd.array(["4374302"], dtype="string[pyarrow]"),
            "sleeper_id": pd.array(["6794"], dtype="string[pyarrow]"),
        }
    ).to_parquet(tmp_path / "raw" / "id_map.parquet", index=False)

    ext.refresh_external_projections(tmp_path, season=2026, asof=date(2026, 7, 15))
    latest = read_latest_partition(tmp_path / "raw", "external_projections", season=2026)
    ExternalProjectionSchema.validate(latest)
    assert set(latest["source"]) == {"ESPN", "SLEEPER"}
    assert (latest["gsis_id"] == "00-0036900").sum() == 2  # both sources crosswalked the veteran


def test_refresh_refuses_empty_pull(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(ext, "fetch_espn", lambda season: {"players": []})
    monkeypatch.setattr(ext, "fetch_sleeper_season", lambda season: [])
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame({"gsis_id": pd.array(["00-0036900"], dtype="string[pyarrow]"),
                  "espn_id": pd.array(["x"], dtype="string[pyarrow]"),
                  "sleeper_id": pd.array(["y"], dtype="string[pyarrow]")}
                 ).to_parquet(tmp_path / "raw" / "id_map.parquet", index=False)
    with pytest.raises(SystemExit):
        ext.refresh_external_projections(tmp_path, season=2026)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_ingest/test_external_projections.py -k "refresh" -v`
Expected: FAIL (`has no attribute 'refresh_external_projections'`).

- [ ] **Step 3: Implement** (append)

```python
def fetch_espn(season: int, limit: int = 800) -> dict[str, Any]:
    flt = {"players": {"limit": limit, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    req = urllib.request.Request(
        _ESPN_URL.format(season=season),
        headers={"User-Agent": _UA, "X-Fantasy-Filter": json.dumps(flt)},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
        return json.load(resp)  # type: ignore[no-any-return]


def fetch_sleeper_season(season: int) -> list[dict[str, Any]]:
    req = urllib.request.Request(_SLEEPER_URL.format(season=season), headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted host)
        return json.load(resp)  # type: ignore[no-any-return]


def refresh_external_projections(
    data_root: Path, *, season: int, asof: date | None = None
) -> Path:
    """Fetch ESPN + Sleeper preseason projections, crosswalk to gsis_id (placeholder for
    rookies), validate, and write one dated snapshot. `asof` defaults to today (UTC)."""
    asof = asof or datetime.now(UTC).date()
    try:
        espn_payload = fetch_espn(season)
        sleeper_payload = fetch_sleeper_season(season)
    except urllib.error.URLError as exc:
        detail = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc.reason)
        raise SystemExit(f"External API error for season {season}: {detail}") from exc

    espn = parse_espn_players(espn_payload, season)
    sleeper = parse_sleeper_projections(sleeper_payload)
    if espn.empty or sleeper.empty:
        raise SystemExit(
            f"Empty pull for {season} (espn={len(espn)} rows, sleeper={len(sleeper)} rows) — "
            f"refusing to write an empty asof snapshot."
        )

    id_map = read_partition(data_root / "raw", "id_map")
    frame = pd.concat(
        [
            _espn_to_canonical(espn, season=season, asof=asof, id_map=id_map),
            _sleeper_to_canonical(sleeper, season=season, asof=asof, id_map=id_map),
        ],
        ignore_index=True,
    )
    frame = ExternalProjectionSchema.validate(frame)
    return write_partition(
        data_root / "raw", "external_projections", frame, season=season, asof=asof
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest external preseason projections (one snapshot).")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument(
        "--asof", type=lambda s: date.fromisoformat(s), default=None,
        help="Pull-date partition (ISO YYYY-MM-DD); defaults to today (UTC).",
    )
    args = ap.parse_args()
    path = refresh_external_projections(args.data_root, season=args.season, asof=args.asof)
    print(f"Wrote external-projection snapshot: {path}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + full gate**

Run: `pytest tests/test_ingest/test_external_projections.py -v`
Expected: all PASS.
Run: `pytest -k "ingest or store or schemas" -q && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: green / no violations. (If mypy flags the `_PYARROW_STR` private import from `schemas`, mirror how existing ingest modules import it — `schedules.py`/`id_map.py` already `from projections.schemas import _PYARROW_STR, ...`.)

- [ ] **Step 5: Commit**

```bash
git add src/projections/ingest/external_projections.py tests/test_ingest/test_external_projections.py
git commit -m "feat(ingest): refresh_external_projections orchestrator + CLI"
```

---

## Phase 5 — Real run + docs

### Task 8: Live 2026 pull + verification + PM/TODO

**Files:** writes `data/raw/external_projections/season=2026/asof=<today>/`; modifies `project_management.md`, `TODO.md`.

- [ ] **Step 1: Run the real ingest for 2026**

Run: `python -m projections.ingest.external_projections --season 2026`
Expected: `Wrote external-projection snapshot: data/raw/external_projections/season=2026/asof=<today>/part.parquet`.

- [ ] **Step 2: Verify a veteran (real gsis) and a rookie (placeholder)**

Run:
```bash
python -c "import pandas as pd, glob; \
f=sorted(glob.glob('data/raw/external_projections/season=2026/asof=*/part.parquet'))[-1]; \
d=pd.read_parquet(f); print('rows', len(d), 'sources', set(d['source'])); \
print('placeholder rookies:', int(d['is_placeholder_gsis'].sum()), '/', len(d)); \
v=d[d.full_name=='Ja\\'Marr Chase'].head(1); print(v[['source','gsis_id','is_placeholder_gsis','adp','receptions']].to_string(index=False)); \
print(d[d.is_placeholder_gsis][['source','full_name','position','gsis_id','adp']].head(3).to_string(index=False))"
```
Expected: a veteran (Chase) with a real `00-00…` gsis_id from at least ESPN; some placeholder `99-…` rows (2026 rookies); both ESPN + SLEEPER sources present.

- [ ] **Step 3: Verify the snapshot is gitignored / not committed**

Run: `git status --short data/`
Expected: nothing staged under `data/raw/external_projections/` (it's gitignored — confirm `data/raw/` is in `.gitignore`; if `data/external_projections/` was separately ignored from the spike, that's fine, but the canonical home is now `data/raw/external_projections/` which `data/raw/` already covers).

- [ ] **Step 4: Update project_management.md + TODO.md**

Add a `project_management.md` top entry (house format: date, branch, status, what shipped, what's next) summarizing the v1 ingest mechanism. Update `TODO.md` #38: mark the "promote the spike puller into a proper ingest source" + "fix the id_map float-id defect" sub-items done; the remaining #38 scope (scraped sources, consensus blend, distribution-wrapping, Draft Hub) stays open. Note the dated-snapshot mechanism is ready to refresh weekly up to the August draft.

```bash
git add project_management.md TODO.md
git commit -m "docs(pm,todo): external-projection ingest mechanism v1 shipped (TODO #38)"
```

---

## Final verification (before opening the PR)

- [ ] `pytest -v -k "ingest or store or schemas or external"` — all pass.
- [ ] `mypy src tests` — zero violations.
- [ ] `ruff check src tests` — zero violations.
- [ ] `ruff format --check src tests` — no drift.
- [ ] Paste outputs into the PR description (per CLAUDE.md forced-verification).

---

## Spec coverage self-check

- Repeatable ingest source following the template → Tasks 4–7 (`refresh_external_projections`, CLI).
- ESPN (stat line + ADP + rank) + Sleeper (ADP + name/position) → Task 4 parsers.
- Dated-snapshot storage (`asof` partition) → Task 1 + Task 7.
- Store extension in the sanctioned layer → Task 1.
- `ExternalProjectionSchema` (stat line nullable, source id, asof) → Task 2.
- Rookie placeholder gsis (deterministic, auto-reconciling, narrow scope) → Task 5.
- In-scope id_map float-id fix → Task 3.
- Empty-pull = hard error → Task 7 (`refresh` guard).
- Store stat lines not points → Task 2/6 (stat fields stored; no scoring here).
- Testing (pure transforms unit-tested; network verified by running) → all phases + Task 8.
- Out of scope (consensus, scraping, distribution-wrap, Draft Hub) → not implemented; TODO #38 remainder.
