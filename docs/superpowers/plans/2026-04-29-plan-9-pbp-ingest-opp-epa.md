# Plan 9 — PBP ingest + opponent-adjusted EPA features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add play-by-play ingest from `nfl_data_py.import_pbp_data` and replace the v1 opponent-strength proxy (`features/_opponent.py:opp_allowed_fppg`) with a schedule-of-strength-adjusted EPA-per-play residual feature on every per-position feature builder. Run the new Plan 8 adoption gate on the feature swap; ship per-position changes per the verdict.

**Architecture:** Mirrors `weekly_stats.py` ingest template for the new PBP source (per-season parquet partition, manifest entry, opt-in `--run-network` smoke). Replaces the v1 `opp_allowed_fppg` helper with `opp_epa_allowed_residual` in `features/_opponent.py` — same join interface (per-(season, week, opp_team) shape, target_week +1 shift) so per-position builders only swap the function call and the output column name. Extends `scripts/adoption_gate.py` with a `--baseline-run`/`--candidate-run` dual-run mode for cross-run pairing (load-bearing for every future feature-class plan in the TODO #3b split).

**Tech Stack:** Python 3.13+, pandas, pandera, nfl_data_py, pytest. No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-04-29-plan-9-pbp-ingest-opp-epa-design.md`.
**Predecessor:** Plan 8 (adoption gate redesign) merged at `6675359` (PR #16).

---

## Phase structure

| Phase | Tasks | Files touched | Description |
|---|---|---|---|
| 1 | 1–4 | ≤5 per task | PBP ingest plumbing (schema, fixture, ingest module, network smoke) |
| 2 | 5 | 2 | `_opponent.py` rewrite + tests |
| 3 | 6–9 | 3 per task | Per-position feature swap (schemas.py + features/{pos}.py + tests) |
| 4 | 10 | 4 | Thread `pbp` through 4 direct-builder scripts |
| 5 | 11 | 2 | Adoption-gate CLI dual-run mode + tests |
| 6 | 12–14 | data + spec | Real-data ingest, pre/post backtest runs, gate run, verdict capture |
| 7 | 15–16 | per-position + docs | Per-position revert, snapshot finalize, PM/TODO updates |

The per-task **5-files-or-fewer rule** (CLAUDE.md "phased execution") is honored throughout. Each task ends with a commit. Each phase ends with a green `pytest` / `mypy src tests` / `ruff check src tests scripts` run.

---

## Task 1: Add `PbpSchema` to `schemas.py`

**Files:**
- Modify: `src/projections/schemas.py` (insert new schema + register `_dist_family_values` style if needed — single-class addition)
- Test: `tests/test_schemas/test_dataframe_schemas.py` (add a test for `PbpSchema`)

- [ ] **Step 1: Re-read `src/projections/schemas.py` to confirm current contents and locate insertion point**

Run: `cat src/projections/schemas.py | tail -100`
Expected: confirms file currently ends with `ProjectionSeasonSchema`. New schema goes after `NgsReceivingSchema` (alphabetical/topical ordering with the other ingest schemas) and before `WrFeaturesSchema`. Read once before editing per CLAUDE.md context-decay rule.

- [ ] **Step 2: Write a failing schema test**

Edit `tests/test_schemas/test_dataframe_schemas.py`. Add:

```python
def test_pbp_schema_validates_minimal_row() -> None:
    """A row with required fields and nullable extras passes."""
    from projections.schemas import PbpSchema

    df = pd.DataFrame(
        {
            "play_id": [1],
            "game_id": ["2024_03_KC_ATL"],
            "season": [2024],
            "week": [3],
            "posteam": pd.array(["KC"], dtype=pd.StringDtype("pyarrow")),
            "defteam": pd.array(["ATL"], dtype=pd.StringDtype("pyarrow")),
            "play_type": pd.array(["pass"], dtype=pd.StringDtype("pyarrow")),
            "qb_dropback": [1.0],
            "qb_scramble": [0.0],
            "sack": [0.0],
            "rush_attempt": [0.0],
            "pass_attempt": [1.0],
            "epa": [0.42],
            "wpa": [0.05],
            "success": [1.0],
            "air_yards": [12.0],
            "yards_after_catch": [3.0],
            "complete_pass": [1.0],
            "xpass": [0.65],
            "pass_oe": [0.10],
            "down": [1.0],
            "ydstogo": [10],
            "yardline_100": [75.0],
            "half_seconds_remaining": [1200.0],
            "passer_player_id": pd.array(
                ["00-0034857"], dtype=pd.StringDtype("pyarrow")
            ),
            "rusher_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
            "receiver_player_id": pd.array(
                ["00-0036322"], dtype=pd.StringDtype("pyarrow")
            ),
        }
    )
    PbpSchema.validate(df)


def test_pbp_schema_rejects_invalid_team_code() -> None:
    """defteam must be in canonical Team enum."""
    import pandera.pandas as pa
    from projections.schemas import PbpSchema

    df = pd.DataFrame(
        {
            "play_id": [1],
            "game_id": ["x"],
            "season": [2024],
            "week": [3],
            "posteam": pd.array(["KC"], dtype=pd.StringDtype("pyarrow")),
            "defteam": pd.array(["JAX"], dtype=pd.StringDtype("pyarrow")),  # alias
            "play_type": pd.array(["pass"], dtype=pd.StringDtype("pyarrow")),
            "qb_dropback": [1.0],
            "qb_scramble": [0.0],
            "sack": [0.0],
            "rush_attempt": [0.0],
            "pass_attempt": [1.0],
            "epa": [0.0],
            "wpa": [0.0],
            "success": [0.0],
            "air_yards": [0.0],
            "yards_after_catch": [0.0],
            "complete_pass": [0.0],
            "xpass": [0.5],
            "pass_oe": [0.0],
            "down": [1.0],
            "ydstogo": [10],
            "yardline_100": [75.0],
            "half_seconds_remaining": [1200.0],
            "passer_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
            "rusher_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
            "receiver_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        PbpSchema.validate(df)
```

- [ ] **Step 3: Run tests, expect ImportError**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py::test_pbp_schema_validates_minimal_row -v`
Expected: FAIL with `ImportError: cannot import name 'PbpSchema'`.

- [ ] **Step 4: Add `PbpSchema` to `src/projections/schemas.py`**

Insert after `NgsReceivingSchema` (around line 426, before `WrFeaturesSchema`). The schema text matches spec §4.1:

```python
class PbpSchema(pa.DataFrameModel):
    """Per-play data — what `ingest.pbp` produces. Curated subset of
    `nfl_data_py.import_pbp_data`'s ~370-column output."""

    play_id: Series[int] = pa.Field(ge=1)
    game_id: Series[str]
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    posteam: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)
    defteam: Series[str] = pa.Field(isin=_TEAM_VALUES, nullable=True)
    play_type: Series[str] = pa.Field(nullable=True)
    qb_dropback: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    qb_scramble: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    sack: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    rush_attempt: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    pass_attempt: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    epa: Series[float] = pa.Field(nullable=True)
    wpa: Series[float] = pa.Field(nullable=True)
    success: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    air_yards: Series[float] = pa.Field(nullable=True)
    yards_after_catch: Series[float] = pa.Field(nullable=True)
    complete_pass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    xpass: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    pass_oe: Series[float] = pa.Field(nullable=True)
    down: Series[float] = pa.Field(ge=1, le=4, nullable=True)
    ydstogo: Series[int] = pa.Field(ge=0, le=99, nullable=True)
    yardline_100: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    half_seconds_remaining: Series[float] = pa.Field(ge=0, le=1800, nullable=True)
    passer_player_id: Series[str] = pa.Field(
        str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True
    )
    rusher_player_id: Series[str] = pa.Field(
        str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True
    )
    receiver_player_id: Series[str] = pa.Field(
        str_matches=rf"^{GSIS_ID_PATTERN}$", nullable=True
    )

    class Config:
        strict = "filter"
```

- [ ] **Step 5: Run tests, expect pass**

Run: `pytest tests/test_schemas/test_dataframe_schemas.py -v -k pbp`
Expected: 2 passed.

- [ ] **Step 6: Run full schemas test directory and mypy**

Run:
```
pytest tests/test_schemas/ -v
mypy src/projections/schemas.py
ruff check src/projections/schemas.py tests/test_schemas/
ruff format --check src/projections/schemas.py tests/test_schemas/
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_dataframe_schemas.py
git commit -m "feat(schemas): add PbpSchema for play-by-play ingest — Plan 9 Phase 1"
```

---

## Task 2: Add `fake_pbp_df` fixture to top-level conftest.py

**Files:**
- Modify: `tests/conftest.py` (append `fake_pbp_df` fixture matching spec §7 test-data prep)

The fixture must include enough variety to exercise the residual algorithm later: 2 offenses with distinct EPA means, 2 defenses with distinct schedule-of-strength, 4+ weeks per defense, one of each `play_type`, sacks/scrambles, one row with `posteam=NaN`.

- [ ] **Step 1: Re-read `tests/conftest.py` to find the right insertion point**

Run: `tail -30 tests/conftest.py`
Expected: confirms file currently ends after `fake_ngs_receiving_df`. Append `fake_pbp_df` after the last fixture.

- [ ] **Step 2: Append the fixture**

Add to the end of `tests/conftest.py`:

```python
@pytest.fixture
def fake_pbp_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_pbp_data([2024])` — handcrafted plays.

    Two offenses (KC strong, NYG weak) and two defenses (BUF faces strong
    schedule, MIA faces weak schedule) across 5 weeks, with deliberate EPA
    variation to exercise schedule-of-strength residual logic. Includes one
    of each play_type, plus one ST play and one no_play row to exercise the
    ingest filter.
    """
    rows: list[dict[str, object]] = []

    # KC (strong offense) plays BUF (good defense) every week 1-5; KC's EPA
    # against BUF is below KC's overall mean (BUF defense lowers it).
    # NYG (weak offense) plays MIA every week 1-5; NYG's EPA against MIA is
    # below NYG's overall mean (MIA also a competent defense), but MIA's
    # schedule-of-strength is much weaker than BUF's.
    week_epa_kc_vs_buf = [0.05, -0.10, 0.02, -0.08, 0.00]  # near zero
    week_epa_nyg_vs_mia = [-0.20, -0.30, -0.15, -0.25, -0.10]  # negative

    play_id = 1
    for w_idx, week in enumerate([1, 2, 3, 4, 5]):
        # KC offense vs BUF defense — 4 plays per week (2 pass, 2 run).
        for play_kind in ("pass", "pass", "run", "run"):
            rows.append(
                {
                    "play_id": play_id,
                    "game_id": f"2024_{week:02d}_KC_BUF",
                    "season": 2024,
                    "week": week,
                    "posteam": "KC",
                    "defteam": "BUF",
                    "play_type": play_kind,
                    "qb_dropback": 1.0 if play_kind == "pass" else 0.0,
                    "qb_scramble": 0.0,
                    "sack": 0.0,
                    "rush_attempt": 1.0 if play_kind == "run" else 0.0,
                    "pass_attempt": 1.0 if play_kind == "pass" else 0.0,
                    "epa": week_epa_kc_vs_buf[w_idx],
                    "wpa": 0.0,
                    "success": 1.0,
                    "air_yards": 8.0 if play_kind == "pass" else None,
                    "yards_after_catch": 3.0 if play_kind == "pass" else None,
                    "complete_pass": 1.0 if play_kind == "pass" else 0.0,
                    "xpass": 0.55 if play_kind == "pass" else 0.45,
                    "pass_oe": 0.0,
                    "down": 1.0,
                    "ydstogo": 10,
                    "yardline_100": 75.0 - 5.0 * play_id,
                    "half_seconds_remaining": 1200.0,
                    "passer_player_id": "00-0034857" if play_kind == "pass" else None,
                    "rusher_player_id": "00-0030506" if play_kind == "run" else None,
                    "receiver_player_id": "00-0036322" if play_kind == "pass" else None,
                }
            )
            play_id += 1

        # NYG offense vs MIA defense — 4 plays per week (2 pass, 2 run).
        for play_kind in ("pass", "pass", "run", "run"):
            rows.append(
                {
                    "play_id": play_id,
                    "game_id": f"2024_{week:02d}_NYG_MIA",
                    "season": 2024,
                    "week": week,
                    "posteam": "NYG",
                    "defteam": "MIA",
                    "play_type": play_kind,
                    "qb_dropback": 1.0 if play_kind == "pass" else 0.0,
                    "qb_scramble": 0.0,
                    "sack": 0.0,
                    "rush_attempt": 1.0 if play_kind == "run" else 0.0,
                    "pass_attempt": 1.0 if play_kind == "pass" else 0.0,
                    "epa": week_epa_nyg_vs_mia[w_idx],
                    "wpa": 0.0,
                    "success": 0.0,
                    "air_yards": 6.0 if play_kind == "pass" else None,
                    "yards_after_catch": 1.0 if play_kind == "pass" else None,
                    "complete_pass": 1.0 if play_kind == "pass" else 0.0,
                    "xpass": 0.55 if play_kind == "pass" else 0.45,
                    "pass_oe": 0.0,
                    "down": 1.0,
                    "ydstogo": 10,
                    "yardline_100": 75.0,
                    "half_seconds_remaining": 1200.0,
                    "passer_player_id": None,
                    "rusher_player_id": None,
                    "receiver_player_id": None,
                }
            )
            play_id += 1

    # Edge-case rows (week 1 only) — sack, scramble, kickoff, punt, no_play,
    # field_goal, extra_point, qb_kneel, qb_spike, posteam=NaN.
    edge_rows = [
        # Sack — pass-classified.
        {
            "play_id": play_id,
            "game_id": "2024_01_KC_BUF",
            "season": 2024,
            "week": 1,
            "posteam": "KC",
            "defteam": "BUF",
            "play_type": "pass",
            "qb_dropback": 1.0,
            "qb_scramble": 0.0,
            "sack": 1.0,
            "rush_attempt": 0.0,
            "pass_attempt": 0.0,
            "epa": -1.5,
            "wpa": 0.0,
            "success": 0.0,
            "air_yards": None,
            "yards_after_catch": None,
            "complete_pass": 0.0,
            "xpass": 0.55,
            "pass_oe": 0.0,
            "down": 3.0,
            "ydstogo": 10,
            "yardline_100": 30.0,
            "half_seconds_remaining": 600.0,
            "passer_player_id": "00-0034857",
            "rusher_player_id": None,
            "receiver_player_id": None,
        },
        # Scramble — pass-classified.
        {
            "play_id": play_id + 1,
            "game_id": "2024_01_NYG_MIA",
            "season": 2024,
            "week": 1,
            "posteam": "NYG",
            "defteam": "MIA",
            "play_type": "run",  # nfl_data_py marks scrambles play_type=run + qb_scramble=1
            "qb_dropback": 1.0,
            "qb_scramble": 1.0,
            "sack": 0.0,
            "rush_attempt": 1.0,
            "pass_attempt": 0.0,
            "epa": 0.30,
            "wpa": 0.0,
            "success": 1.0,
            "air_yards": None,
            "yards_after_catch": None,
            "complete_pass": 0.0,
            "xpass": 0.55,
            "pass_oe": 0.0,
            "down": 2.0,
            "ydstogo": 5,
            "yardline_100": 50.0,
            "half_seconds_remaining": 900.0,
            "passer_player_id": None,
            "rusher_player_id": "00-0030506",
            "receiver_player_id": None,
        },
        # Kickoff — has posteam=NaN per nfl_data_py.
        {
            "play_id": play_id + 2,
            "game_id": "2024_01_KC_BUF",
            "season": 2024,
            "week": 1,
            "posteam": None,
            "defteam": None,
            "play_type": "kickoff",
            "qb_dropback": 0.0,
            "qb_scramble": 0.0,
            "sack": 0.0,
            "rush_attempt": 0.0,
            "pass_attempt": 0.0,
            "epa": None,
            "wpa": None,
            "success": None,
            "air_yards": None,
            "yards_after_catch": None,
            "complete_pass": None,
            "xpass": None,
            "pass_oe": None,
            "down": None,
            "ydstogo": None,
            "yardline_100": None,
            "half_seconds_remaining": None,
            "passer_player_id": None,
            "rusher_player_id": None,
            "receiver_player_id": None,
        },
        # no_play — epa=NaN, filtered at feature time.
        {
            "play_id": play_id + 3,
            "game_id": "2024_01_KC_BUF",
            "season": 2024,
            "week": 1,
            "posteam": "KC",
            "defteam": "BUF",
            "play_type": "no_play",
            "qb_dropback": None,
            "qb_scramble": None,
            "sack": None,
            "rush_attempt": None,
            "pass_attempt": None,
            "epa": None,
            "wpa": None,
            "success": None,
            "air_yards": None,
            "yards_after_catch": None,
            "complete_pass": None,
            "xpass": None,
            "pass_oe": None,
            "down": None,
            "ydstogo": None,
            "yardline_100": None,
            "half_seconds_remaining": None,
            "passer_player_id": None,
            "rusher_player_id": None,
            "receiver_player_id": None,
        },
    ]
    rows.extend(edge_rows)

    return pd.DataFrame(rows)
```

- [ ] **Step 3: Verify the fixture is valid for the schema**

This sanity-check is a one-off; not a committed test. Run:

```
python -c "
import pandas as pd
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
from projections.schemas import PbpSchema
import importlib.util
spec = importlib.util.spec_from_file_location('conftest', 'tests/conftest.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Manually call the fixture function (it's just a function under @pytest.fixture).
df = mod.fake_pbp_df.__wrapped__()
print(df.shape)
print(df['play_type'].value_counts(dropna=False))
"
```

Expected: prints the shape and the play_type counts. (This is informational — the actual assertion happens via pandera in Task 3 after we wire ingest normalization.)

- [ ] **Step 4: Run the full conftest-affecting test suite to confirm no regression**

Run: `pytest tests/test_ingest/ -v --no-header`
Expected: same passing count as before this task; the new fixture has no consumers yet.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "test(conftest): add fake_pbp_df fixture for PBP ingest tests — Plan 9 Phase 1"
```

---

## Task 3: Create `src/projections/ingest/pbp.py` + tests

**Files:**
- Create: `src/projections/ingest/pbp.py`
- Modify: `src/projections/ingest/__init__.py` (re-export `refresh_pbp`)
- Test: `tests/test_ingest/test_pbp.py` (new file)
- Modify: `tests/conftest.py` (already done in Task 2 — read-only here)

- [ ] **Step 1: Re-read template `weekly_stats.py` and `__init__.py`**

Run: `cat src/projections/ingest/weekly_stats.py | head -30 && echo --- && cat src/projections/ingest/__init__.py`
Expected: confirm `_KEEP`/`_RENAME`/`_fetch_raw_*`/`_normalize_one_season`/`refresh_*` shape; confirm what gets re-exported.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_ingest/test_pbp.py`:

```python
"""PBP ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_pbp
from projections.schemas import PbpSchema
from projections.store import read_partition


def test_refresh_pbp_writes_partitioned_parquet(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    written = refresh_pbp(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    PbpSchema.validate(df)


def test_refresh_pbp_idempotent(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    n_first = len(read_partition(tmp_path / "raw", "pbp", season=2024))
    refresh_pbp(tmp_path, seasons=[2024])
    n_second = len(read_partition(tmp_path / "raw", "pbp", season=2024))
    assert n_first == n_second


def test_refresh_pbp_normalizes_team_codes(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliased = fake_pbp_df.copy()
    # First non-null posteam row gets a JAX alias.
    first_idx = aliased[aliased["posteam"].notna()].index[0]
    aliased.loc[first_idx, "posteam"] = "JAX"
    aliased.loc[first_idx, "defteam"] = "LA"
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: aliased,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    assert "JAX" not in set(df["posteam"].dropna())
    assert "JAC" in set(df["posteam"].dropna())
    assert "LAR" in set(df["defteam"].dropna())


def test_refresh_pbp_curates_columns(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict='filter' drops upstream columns we didn't keep."""
    polluted = fake_pbp_df.copy()
    polluted["unwanted_extra_column"] = 0.0
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: polluted,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    assert "unwanted_extra_column" not in df.columns


def test_refresh_pbp_preserves_no_play_rows(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """no_play rows are preserved at ingest; feature-time filters remove them
    where appropriate. This guards against accidentally narrowing the ingest."""
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    assert (df["play_type"] == "no_play").any()


def test_refresh_pbp_writes_manifest(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    manifest = pd.read_parquet(tmp_path / "manifests" / "ingest_manifest.parquet")
    assert (
        (manifest["table"] == "pbp")
        & (manifest["season"] == 2024)
    ).any()
```

- [ ] **Step 3: Run tests, expect ImportError**

Run: `pytest tests/test_ingest/test_pbp.py -v`
Expected: collection error (`cannot import name 'refresh_pbp' from 'projections.ingest'`).

- [ ] **Step 4: Create `src/projections/ingest/pbp.py`**

```python
"""Refresh per-season play-by-play from `nfl_data_py.import_pbp_data`.

Writes one parquet partition per season (curated subset of upstream's ~370
columns; see PbpSchema). Idempotent — re-running a season overwrites that
partition only.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import _PYARROW_STR, PbpSchema, normalize_team_code
from projections.store import write_partition

_KEEP: tuple[str, ...] = (
    "play_id",
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "qb_dropback",
    "qb_scramble",
    "sack",
    "rush_attempt",
    "pass_attempt",
    "epa",
    "wpa",
    "success",
    "air_yards",
    "yards_after_catch",
    "complete_pass",
    "xpass",
    "pass_oe",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
)


def _fetch_raw_pbp(seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around nfl_data_py; tests monkey-patch this."""
    return nfl.import_pbp_data(seasons)


def _normalize_team_or_none(v: object) -> str | None:
    """Apply normalize_team_code, but pass None through for special-teams plays."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return normalize_team_code(str(v)).value


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Coerce season/week to int64 — nfl_data_py returns int32 here too.
    for int_col in ("season", "week"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype("int64")

    # ydstogo: nullable Int64 — pandera Series[int] with nullable=True needs
    # pandas Int64Dtype, not float64. Fillna(-1) would corrupt; preserve NaN.
    if "ydstogo" in df.columns:
        # nfl_data_py returns ydstogo as float64 with NaN; convert to Int64.
        df["ydstogo"] = df["ydstogo"].astype("Int64")

    # play_id: int (PK). Coerce to int64.
    if "play_id" in df.columns:
        df["play_id"] = df["play_id"].astype("int64")

    # Team codes — nullable string (kickoffs/punts have NaN posteam/defteam).
    for team_col in ("posteam", "defteam"):
        if team_col in df.columns:
            df[team_col] = (
                df[team_col].map(_normalize_team_or_none).astype(_PYARROW_STR)
            )

    # String columns — pyarrow-backed.
    for str_col in (
        "game_id",
        "play_type",
        "passer_player_id",
        "rusher_player_id",
        "receiver_player_id",
    ):
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(_PYARROW_STR)

    # Filter rows with malformed gsis_ids on the player columns. Per
    # CLAUDE.md ID hygiene: passer_player_id is a GsisId-format string.
    # Upstream sometimes emits PFR-style legacy IDs (e.g., "MahPa00") on
    # very old plays. Coerce malformed values to None so the schema's
    # nullable=True path accepts them.
    import re

    gsis_re = re.compile(r"^\d{2}-\d{7}$")

    def _coerce_player_id(v: object) -> str | None:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v)
        return s if gsis_re.fullmatch(s) else None

    for pid_col in ("passer_player_id", "rusher_player_id", "receiver_player_id"):
        if pid_col in df.columns:
            df[pid_col] = df[pid_col].map(_coerce_player_id).astype(_PYARROW_STR)

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = PbpSchema.validate(df)
    return df


def refresh_pbp(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write play-by-play data for each season.

    One partition per season. Idempotent — re-running a season overwrites
    that partition only.
    """
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_pbp([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "pbp", df, season=season, week=None)
        record_manifest(data_root, table="pbp", season=season, df=df)
        written.append(path)
    return written
```

- [ ] **Step 5: Re-export from `src/projections/ingest/__init__.py`**

Add to `__init__.py` `__all__` and imports. Example diff:

```python
from projections.ingest.pbp import refresh_pbp

__all__ = [
    # ...existing exports...
    "refresh_pbp",
]
```

(Read the current file first; insert in alphabetical order with the existing `refresh_*` exports.)

- [ ] **Step 6: Run tests, expect pass**

Run: `pytest tests/test_ingest/test_pbp.py -v`
Expected: 6 passed.

- [ ] **Step 7: Run full ingest test suite, mypy, ruff**

Run:
```
pytest tests/test_ingest/ -v
mypy src/projections/ingest/pbp.py src/projections/ingest/__init__.py
ruff check src/projections/ingest/ tests/test_ingest/
ruff format --check src/projections/ingest/ tests/test_ingest/
```
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/projections/ingest/pbp.py src/projections/ingest/__init__.py tests/test_ingest/test_pbp.py
git commit -m "feat(ingest): refresh_pbp module with curated 27-column subset — Plan 9 Phase 1"
```

---

## Task 4: Add opt-in network smoke for PBP

**Files:**
- Modify: `tests/test_ingest/test_api_drift.py` (append `test_pbp_api_columns_and_schema`)

This test is `@pytest.mark.network`-decorated and skipped by default; only runs with `pytest -m network --run-network`.

- [ ] **Step 1: Re-read existing api-drift smokes for pattern**

Run: `cat tests/test_ingest/test_api_drift.py | head -80`
Expected: confirms one `test_<source>_api_columns_and_schema` per source pattern.

- [ ] **Step 2: Append the smoke**

Add to end of `tests/test_ingest/test_api_drift.py`:

```python
@pytest.mark.network
def test_pbp_api_columns_and_schema() -> None:
    """Opt-in network smoke: pulls a small live PBP slice and asserts every
    column we keep is present, then runs the normalize end-to-end so pandera
    surfaces dtype / value drift."""
    import nfl_data_py as nfl

    from projections.ingest.pbp import _KEEP, _normalize_one_season

    # Single-season fetch, but pbp is large — limit to a known good year.
    raw = nfl.import_pbp_data([2023])
    missing = set(_KEEP) - set(raw.columns)
    assert not missing, (
        f"PBP upstream missing columns we depend on: {sorted(missing)}. "
        "If this fails after a nfl_data_py bump, patch _KEEP / _normalize_one_season "
        "in src/projections/ingest/pbp.py and re-run."
    )

    normalized = _normalize_one_season(raw)
    # Sanity: lots of plays, sensibly distributed.
    assert len(normalized) > 30000  # ~50k expected
    assert normalized["play_type"].notna().any()
    assert normalized["epa"].notna().mean() > 0.6  # ~80% in practice
```

- [ ] **Step 3: Verify the smoke is skipped by default**

Run: `pytest tests/test_ingest/test_api_drift.py -v`
Expected: existing smokes show as `SKIPPED`; new smoke collected and skipped.

- [ ] **Step 4: (Optional) verify with network on**

Run: `pytest tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema -v -m network --run-network`
Expected: passes (takes ~10–30s for the actual fetch). This is opt-in and may be skipped during local plan execution; a later phase task (Task 12) covers the real-data ingest.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingest/test_api_drift.py
git commit -m "test(ingest): network smoke for refresh_pbp — Plan 9 Phase 1"
```

**End of Phase 1 — verification:**

```
pytest tests/test_ingest/ tests/test_schemas/ -v
mypy src tests
ruff check src tests scripts
ruff format --check src tests scripts
```
All four must be green before starting Phase 2.

---

## Task 5: Replace `_opponent.py` with `opp_epa_allowed_residual`

**Files:**
- Modify: `src/projections/features/_opponent.py` (full replacement)
- Modify: `tests/test_features/test_opponent.py` (full replacement)

This is the critical algorithmic task. Tests come first per TDD.

- [ ] **Step 1: Re-read current `_opponent.py` to confirm what's being deleted**

Run: `cat src/projections/features/_opponent.py`
Expected: confirms the v1 `opp_allowed_fppg` and `_row_to_statline`. Both go away.

- [ ] **Step 2: Replace the test file with the new algorithm's tests**

Overwrite `tests/test_features/test_opponent.py`:

```python
"""Opponent-strength helper tests — Plan 9.

Schedule-of-strength residual EPA-allowed by play type. The new helper
replaces the v1 opp_allowed_fppg from Plan 2a.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from projections.features._opponent import opp_epa_allowed_residual


def _make_pbp_row(
    *,
    play_id: int,
    week: int,
    posteam: str,
    defteam: str,
    play_type: str,
    epa: float,
    qb_scramble: float = 0.0,
    sack: float = 0.0,
) -> dict[str, object]:
    """Helper to keep test data construction terse."""
    return {
        "play_id": play_id,
        "game_id": f"2024_{week:02d}_{posteam}_{defteam}",
        "season": 2024,
        "week": week,
        "posteam": posteam,
        "defteam": defteam,
        "play_type": play_type,
        "qb_dropback": 1.0 if play_type == "pass" else 0.0,
        "qb_scramble": qb_scramble,
        "sack": sack,
        "rush_attempt": 1.0 if play_type == "run" else 0.0,
        "pass_attempt": 1.0 if play_type == "pass" else 0.0,
        "epa": epa,
        "wpa": 0.0,
        "success": 1.0 if epa > 0 else 0.0,
    }


def test_opp_epa_allowed_residual_zero_when_offenses_are_league_mean() -> None:
    """If every offense has the same overall mean EPA and the defense allows
    that same mean, the residual is zero."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        # Two offenses, one play each, both at mean EPA = 0.10 vs DEF.
        for posteam in ("OFF1", "OFF2"):
            rows.append(
                _make_pbp_row(
                    play_id=play_id,
                    week=week,
                    posteam=posteam,
                    defteam="DEF",
                    play_type="pass",
                    epa=0.10,
                )
            )
            play_id += 1

    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # DEF target_week=5 (after weeks 1-4 trailing window): residual ≈ 0.
    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    assert math.isclose(
        float(row.iloc[0]["opp_epa_allowed_residual"]), 0.0, abs_tol=1e-9
    )


def test_opp_epa_allowed_residual_positive_against_weak_offenses() -> None:
    """Defense faces only weak offenses but still allows mean EPA at the
    league level — implies they performed worse than expected (residual > 0)."""
    rows: list[dict[str, object]] = []
    play_id = 1
    # OFF_WEAK has overall mean EPA = -0.20 across all opponents.
    # DEF allows OFF_WEAK 0.0 EPA per play (above their average → DEF is weak).
    # Other offense OFF_STRONG plays NO_DEF (different defense) — their plays
    # set OFF_STRONG's overall mean to +0.20 to make the league mean balanced
    # at 0.0, but they don't affect DEF.
    for week in [1, 2, 3, 4]:
        # OFF_WEAK vs DEF — DEF allows 0.0 EPA (vs OFF_WEAK overall mean -0.20).
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_WEAK",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
        # OFF_WEAK vs NO_DEF — sets OFF_WEAK overall mean.
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_WEAK",
                defteam="NO_DEF",
                play_type="pass",
                epa=-0.40,  # so OFF_WEAK overall mean = (-0.40 + 0.0)/2 = -0.20
            )
        )
        play_id += 1

    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    # DEF allowed 0.0 vs OFF_WEAK whose overall mean is -0.20 → residual = +0.20.
    assert math.isclose(
        float(row.iloc[0]["opp_epa_allowed_residual"]), 0.20, abs_tol=1e-9
    )


def test_opp_epa_allowed_residual_negative_against_strong_offenses() -> None:
    """Defense holds strong offenses below their average → residual < 0."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        # OFF_STRONG vs DEF — DEF allows 0.0 EPA.
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_STRONG",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
        # OFF_STRONG vs NO_DEF — sets OFF_STRONG overall mean.
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF_STRONG",
                defteam="NO_DEF",
                play_type="pass",
                epa=0.40,  # OFF_STRONG overall = (0.40 + 0.0)/2 = +0.20
            )
        )
        play_id += 1

    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    # DEF allowed 0.0 vs OFF_STRONG whose mean is +0.20 → residual = -0.20.
    assert math.isclose(
        float(row.iloc[0]["opp_epa_allowed_residual"]), -0.20, abs_tol=1e-9
    )


def test_opp_epa_allowed_residual_pass_filter_includes_sacks_and_scrambles() -> None:
    """play_type='pass' must include sacks and scrambles."""
    rows = [
        _make_pbp_row(
            play_id=1, week=1, posteam="OFF", defteam="DEF",
            play_type="pass", epa=0.1,
        ),
        _make_pbp_row(
            play_id=2, week=2, posteam="OFF", defteam="DEF",
            play_type="pass", epa=-1.5, sack=1.0,
        ),
        _make_pbp_row(
            play_id=3, week=3, posteam="OFF", defteam="DEF",
            play_type="run", epa=0.3, qb_scramble=1.0,  # scramble = pass
        ),
        _make_pbp_row(
            play_id=4, week=4, posteam="OFF", defteam="DEF",
            play_type="run", epa=2.0,  # designed run = NOT pass
        ),
    ]
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # Pass plays counted: weeks 1, 2, 3 (regular pass + sack + scramble = 3 plays).
    # Designed run in week 4 is excluded.
    # Mean across pass plays: (0.1 + -1.5 + 0.3) / 3 ≈ -0.367.
    # OFF's overall pass mean (same plays): same -0.367.
    # Residual: 0.0 (all plays come from one offense; mean of residuals is zero).
    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    assert len(row) == 1
    assert math.isclose(
        float(row.iloc[0]["opp_epa_allowed_residual"]), 0.0, abs_tol=1e-9
    )


def test_opp_epa_allowed_residual_run_filter_excludes_scrambles() -> None:
    """play_type='run' must exclude qb_scramble and sack rows."""
    rows = [
        _make_pbp_row(
            play_id=1, week=1, posteam="OFF", defteam="DEF",
            play_type="run", epa=0.5,
        ),
        _make_pbp_row(
            play_id=2, week=2, posteam="OFF", defteam="DEF",
            play_type="run", epa=0.5, qb_scramble=1.0,  # excluded
        ),
        _make_pbp_row(
            play_id=3, week=3, posteam="OFF", defteam="DEF",
            play_type="run", epa=0.5,
        ),
    ]
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="run", n_weeks=4)

    row = result[(result["week"] == 5) & (result["opp_team"] == "DEF")]
    # 2 designed-run plays included; OFF mean = 0.5; residual = 0.0.
    assert len(row) == 1
    assert math.isclose(
        float(row.iloc[0]["opp_epa_allowed_residual"]), 0.0, abs_tol=1e-9
    )


def test_opp_epa_allowed_residual_target_week_shifted_plus_one() -> None:
    """Residual computed from weeks 1-4 joins onto opponent's week-5 row."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # The week=5 row is what offense-side feature builders join onto.
    # No week=4 row should exist (that would be leakage — using week 4 plays
    # to predict week 4).
    weeks_emitted = set(result["week"].unique())
    assert 5 in weeks_emitted
    # Earlier weeks may be emitted via expanding window (see next test).


def test_opp_epa_allowed_residual_expanding_window_for_early_weeks() -> None:
    """Weeks 2-4 emit rows with underfilled trailing windows (expanding)."""
    rows: list[dict[str, object]] = []
    play_id = 1
    for week in [1, 2, 3, 4]:
        rows.append(
            _make_pbp_row(
                play_id=play_id,
                week=week,
                posteam="OFF",
                defteam="DEF",
                play_type="pass",
                epa=0.0,
            )
        )
        play_id += 1
    pbp = pd.DataFrame(rows)
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)

    # Weeks 2-5 get rows. Week 1 cannot — no prior-week data exists.
    weeks_emitted = sorted(result["week"].unique())
    assert weeks_emitted == [2, 3, 4, 5]


def test_opp_epa_allowed_residual_skips_no_play_and_nan_epa() -> None:
    """Rows with epa=NaN are dropped (no_play / pre-snap penalties)."""
    rows: list[dict[str, object]] = [
        _make_pbp_row(
            play_id=1, week=1, posteam="OFF", defteam="DEF",
            play_type="pass", epa=0.1,
        ),
        # epa=NaN → drop.
        {
            "play_id": 2,
            "game_id": "x",
            "season": 2024,
            "week": 1,
            "posteam": "OFF",
            "defteam": "DEF",
            "play_type": "no_play",
            "qb_dropback": None,
            "qb_scramble": None,
            "sack": None,
            "rush_attempt": None,
            "pass_attempt": None,
            "epa": None,
            "wpa": None,
            "success": None,
        },
    ]
    pbp = pd.DataFrame(rows)
    # Should not raise; the NaN-epa row is filtered before averaging.
    result = opp_epa_allowed_residual(pbp, play_type="pass", n_weeks=4)
    assert "opp_epa_allowed_residual" in result.columns
```

- [ ] **Step 3: Run tests, expect ImportError**

Run: `pytest tests/test_features/test_opponent.py -v`
Expected: `ImportError: cannot import name 'opp_epa_allowed_residual' from 'projections.features._opponent'`.

- [ ] **Step 4: Replace `src/projections/features/_opponent.py` content**

Overwrite `src/projections/features/_opponent.py`:

```python
"""Opponent-strength helper: schedule-of-strength-adjusted EPA-per-play
residual, computed from play-by-play data.

Replaces the v1 `opp_allowed_fppg` (Plan 2a) which used team-week fppg
trailing means without schedule-of-strength adjustment.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def _is_pass_play(df: pd.DataFrame) -> "pd.Series[bool]":
    """A play is pass-classified if play_type=pass OR a sack OR a scramble."""
    return (
        (df["play_type"] == "pass")
        | (df["sack"].fillna(0).astype(int) == 1)
        | (df["qb_scramble"].fillna(0).astype(int) == 1)
    )


def _is_run_play(df: pd.DataFrame) -> "pd.Series[bool]":
    """A play is run-classified if play_type=run AND not a scramble."""
    return (
        (df["play_type"] == "run")
        & (df["qb_scramble"].fillna(0).astype(int) != 1)
    )


def opp_epa_allowed_residual(
    pbp: pd.DataFrame,
    *,
    play_type: Literal["pass", "run"],
    n_weeks: int,
) -> pd.DataFrame:
    """Schedule-of-strength-adjusted EPA-allowed per play type.

    Per-play residual = EPA(p) - mean_EPA_for(posteam, play_type, in_window),
    where mean_EPA_for is that offense's overall pass/run EPA-per-play in the
    same trailing window. The residual answers: "given who they faced, how
    much better/worse than expected did this defense play?"

    Returns one row per (season, target_week, opp_team) with target_week
    shifted +1 from the trailing window's last week, mirroring the v1
    `opp_allowed_fppg` join interface. The opp_team column carries the
    defense; join onto offense-side feature rows on (season, week, opponent).
    """
    if pbp.empty:
        return pd.DataFrame(
            columns=["season", "week", "opp_team", "opp_epa_allowed_residual"]
        ).astype(
            {"season": "int64", "week": "int64", "opp_epa_allowed_residual": float}
        )

    df = pbp.copy()

    # Step 1-3: filter to plays of interest.
    df = df[
        df["epa"].notna()
        & df["posteam"].notna()
        & df["defteam"].notna()
        & (df["play_type"] != "no_play")
    ].copy()
    if play_type == "pass":
        df = df[_is_pass_play(df)].copy()
    else:
        df = df[_is_run_play(df)].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["season", "week", "opp_team", "opp_epa_allowed_residual"]
        ).astype(
            {"season": "int64", "week": "int64", "opp_epa_allowed_residual": float}
        )

    # Step 4: per (offense, season), trailing-window mean EPA per play.
    # We compute this per offense across the entire window of weeks 1..(target_week - 1)
    # for each defense's target week. Implementation: per offense, compute the
    # full-season mean (within season), then we'll restrict per-window when
    # iterating defenses below. For the n_weeks=4 trailing window we compute
    # offense-mean over the same trailing window each defense uses.

    # Per-week offense EPA totals — used to compute trailing-window mean per
    # offense-week.
    off_weekly = (
        df.groupby(["posteam", "season", "week"], as_index=False)
        .agg(off_epa_sum=("epa", "sum"), off_epa_n=("epa", "size"))
    )

    # Per-week defense plays — needed for defense-trailing aggregation.
    def_weekly = (
        df.groupby(["defteam", "season", "week"], as_index=False)
        .agg(def_epa_count=("epa", "size"))
    )

    # Step 5-6: compute the trailing-window mean of per-play residuals for
    # each (defteam, season, target_week). target_week = last_window_week + 1.
    rows: list[dict[str, object]] = []
    for (defteam, season), g_def in def_weekly.groupby(
        ["defteam", "season"], sort=False
    ):
        weeks_def = sorted(g_def["week"].unique())
        for last_week in weeks_def:
            target_week = int(last_week) + 1
            window_min = max(1, int(last_week) - n_weeks + 1)
            window_weeks = list(range(window_min, int(last_week) + 1))

            # Plays this defense allowed in window_weeks.
            mask_def_window = (
                (df["defteam"] == defteam)
                & (df["season"] == season)
                & (df["week"].isin(window_weeks))
            )
            window_plays = df[mask_def_window]
            if window_plays.empty:
                continue

            # For each row, we need the offense's overall mean EPA in the
            # same trailing window (across ALL defenses, not just this one).
            # Compute per-offense window mean once.
            mask_off_window = (
                (df["season"] == season) & (df["week"].isin(window_weeks))
            )
            off_window = df[mask_off_window]
            off_means = (
                off_window.groupby("posteam")["epa"].mean().rename("off_window_mean")
            )

            joined = window_plays.merge(
                off_means.to_frame(),
                left_on="posteam",
                right_index=True,
                how="left",
            )
            joined["residual"] = joined["epa"] - joined["off_window_mean"]
            mean_residual = float(joined["residual"].mean())

            rows.append(
                {
                    "season": int(season),
                    "week": target_week,
                    "opp_team": defteam,
                    "opp_epa_allowed_residual": mean_residual,
                }
            )

    out = pd.DataFrame(
        rows, columns=["season", "week", "opp_team", "opp_epa_allowed_residual"]
    )
    if out.empty:
        out = out.astype(
            {"season": "int64", "week": "int64", "opp_epa_allowed_residual": float}
        )
    else:
        out["season"] = out["season"].astype("int64")
        out["week"] = out["week"].astype("int64")
    return out
```

- [ ] **Step 5: Run tests, expect pass**

Run: `pytest tests/test_features/test_opponent.py -v`
Expected: 8 passed.

- [ ] **Step 6: Run feature-test suite (will fail — per-position builders still call old helper)**

Run: `pytest tests/test_features/ -v -k "opponent"`
Expected: opponent tests pass; other feature tests will fail in Phase 3 once we touch builders. Don't run the full suite yet.

- [ ] **Step 7: Verify mypy + ruff on the new file**

Run:
```
mypy src/projections/features/_opponent.py
ruff check src/projections/features/_opponent.py tests/test_features/test_opponent.py
ruff format --check src/projections/features/_opponent.py tests/test_features/test_opponent.py
```
Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/projections/features/_opponent.py tests/test_features/test_opponent.py
git commit -m "refactor(features): replace opp_allowed_fppg with opp_epa_allowed_residual — Plan 9 Phase 2"
```

---

## Task 6: QB schema swap + builder update

**Files:**
- Modify: `src/projections/schemas.py` (swap one column on `QbFeaturesSchema`)
- Modify: `src/projections/features/qb.py` (swap helper call + output column)
- Modify: `tests/test_features/test_qb.py` (rename column references)
- Modify: `tests/test_features/test_qb_leakage.py` (rename column references if any)
- Modify: `src/projections/models/__init__.py` (if QB feature_columns list references the column)

- [ ] **Step 1: Re-read each file before editing (CLAUDE.md context-decay rule)**

Run:
```
grep -n "opp_allowed_qb_fppg" src/projections/schemas.py src/projections/features/qb.py src/projections/models/__init__.py tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py
```
Expected: locates every reference. Read each file's relevant lines.

- [ ] **Step 2: Update test references (failing-test mode)**

In `tests/test_features/test_qb.py` and `tests/test_features/test_qb_leakage.py`, replace every occurrence of:
- `"opp_allowed_qb_fppg_l4"` → `"opp_pass_epa_allowed_l4"`

Per-file Edit calls; never `replace_all` blindly across files. After edits:

Run: `pytest tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py -v`
Expected: FAIL — KeyError or AssertionError on missing column (the builder hasn't been updated yet).

- [ ] **Step 3: Update `QbFeaturesSchema` in `src/projections/schemas.py`**

Replace:
```python
    opp_allowed_qb_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)
```
With:
```python
    opp_pass_epa_allowed_l4: Series[float] = pa.Field(nullable=True)
```

- [ ] **Step 4: Update `src/projections/features/qb.py` builder**

Add `pbp: pd.DataFrame` to `build_qb_features` signature. Replace the v1 opp-strength block (line 12 import + lines 159-166 + lines 182-188) with the new helper call and merge:

In imports, change:
```python
from projections.features._opponent import opp_allowed_fppg
```
to:
```python
from projections.features._opponent import opp_epa_allowed_residual
```

In `build_qb_features` signature, add `pbp: pd.DataFrame,` to the keyword-only args list.

In the opp-strength block (around line 159), replace:
```python
    # --- Opponent strength proxy ------------------------------------------
    opp_proxy_full = opp_allowed_fppg(
        ws_qb, position=Position.QB, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_qb_fppg_l4"})
```
with:
```python
    # --- Opponent strength: opp-adjusted pass-EPA residual (Plan 9) -------
    pbp_window = pbp[prior_mask(pbp, season=season, as_of_week=as_of_week)].copy()
    opp_proxy_full = opp_epa_allowed_residual(pbp_window, play_type="pass", n_weeks=4)
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_epa_allowed_residual": "opp_pass_epa_allowed_l4"})
```

In the assemble block (around line 182), replace:
```python
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_qb_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )
```
with:
```python
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_pass_epa_allowed_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )
```

Also drop the unused `Ruleset` import if nothing else in the file uses it (search; if only the `Ruleset.espn_ppr()` call we just removed referenced it, drop from imports).

- [ ] **Step 5: Update `src/projections/models/__init__.py` if QB feature_columns reference the old name**

Run: `grep -n "opp_allowed_qb_fppg\|opp_pass_epa_allowed_l4" src/projections/models/__init__.py`

If there's a `feature_columns` list that names `opp_allowed_qb_fppg_l4` for QB, replace with `opp_pass_epa_allowed_l4`. (Likely there is — Model A's BaselineModel needs an explicit column list.)

- [ ] **Step 6: Pass `pbp` from per-position test fixtures**

Each QB-feature test file calls `build_qb_features(...)` directly. Inspect:

Run: `grep -n "build_qb_features(" tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py`

For each call site, add `pbp=fake_pbp_df` (or an empty PBP for tests that don't care about the new feature). Use the `fake_pbp_df` fixture from `tests/conftest.py` — it's globally available.

For tests where `fake_pbp_df` is overkill (e.g., a test only exercising rolling stats), pass an empty DataFrame matching `PbpSchema`'s columns:
```python
empty_pbp = pd.DataFrame(columns=list(PbpSchema.to_schema().columns.keys()))
build_qb_features(..., pbp=empty_pbp)
```

(The helper handles empty PBP gracefully — see Task 5 step 4.)

- [ ] **Step 7: Run tests, expect pass**

Run:
```
pytest tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py -v
```
Expected: all green.

- [ ] **Step 8: Run mypy + ruff on touched files**

Run:
```
mypy src/projections/features/qb.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py
ruff check src/projections/features/qb.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py
ruff format --check src/projections/features/qb.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py
```
Expected: green.

- [ ] **Step 9: Commit**

```bash
git add src/projections/features/qb.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_qb.py tests/test_features/test_qb_leakage.py
git commit -m "feat(features): swap QB opp-fppg → opp_pass_epa_allowed_l4 — Plan 9 Phase 3"
```

---

## Task 7: RB schema swap + builder update

**Files:**
- Modify: `src/projections/schemas.py` (swap `RbFeaturesSchema` column)
- Modify: `src/projections/features/rb.py` (swap helper call; output column `opp_run_epa_allowed_l4`)
- Modify: `tests/test_features/test_rb.py` (rename column references)
- Modify: `tests/test_features/test_rb_leakage.py` (rename column references)
- Modify: `src/projections/models/__init__.py` (RB feature_columns)

Same pattern as Task 6. Key difference: `play_type="run"` and output column name is `opp_run_epa_allowed_l4`.

- [ ] **Step 1: Re-read each file**

Run:
```
grep -n "opp_allowed_rb_fppg" src/projections/schemas.py src/projections/features/rb.py src/projections/models/__init__.py tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py
```

- [ ] **Step 2: Update test references**

Replace `"opp_allowed_rb_fppg_l4"` → `"opp_run_epa_allowed_l4"` in both RB test files.

Run: `pytest tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py -v`
Expected: FAIL.

- [ ] **Step 3: Update `RbFeaturesSchema`**

```python
# Before
    opp_allowed_rb_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)
# After
    opp_run_epa_allowed_l4: Series[float] = pa.Field(nullable=True)
```

- [ ] **Step 4: Update `src/projections/features/rb.py`**

Same pattern as Task 6 step 4. Find the `opp_allowed_fppg` block and replace with:

```python
    pbp_window = pbp[prior_mask(pbp, season=season, as_of_week=as_of_week)].copy()
    opp_proxy_full = opp_epa_allowed_residual(pbp_window, play_type="run", n_weeks=4)
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_epa_allowed_residual": "opp_run_epa_allowed_l4"})
```

Update the merge to reference `opp_run_epa_allowed_l4`.

Add `pbp: pd.DataFrame` to `build_rb_features` keyword-only args.

Update import:
```python
from projections.features._opponent import opp_epa_allowed_residual
```

- [ ] **Step 5: Update RB feature_columns in `src/projections/models/__init__.py`**

Replace `opp_allowed_rb_fppg_l4` with `opp_run_epa_allowed_l4`.

- [ ] **Step 6: Add `pbp` arg to test calls**

Same as Task 6 step 6.

- [ ] **Step 7: Run tests + mypy + ruff**

Run:
```
pytest tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py -v
mypy src/projections/features/rb.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py
ruff check src/projections/features/rb.py tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py
ruff format --check src/projections/features/rb.py tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py
```
Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/projections/features/rb.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_rb.py tests/test_features/test_rb_leakage.py
git commit -m "feat(features): swap RB opp-fppg → opp_run_epa_allowed_l4 — Plan 9 Phase 3"
```

---

## Task 8: WR schema swap + builder update

**Files:**
- Modify: `src/projections/schemas.py` (swap `WrFeaturesSchema` column)
- Modify: `src/projections/features/wr.py`
- Modify: `tests/test_features/test_wr.py`
- Modify: `tests/test_features/test_wr_leakage.py`
- Modify: `src/projections/models/__init__.py`

Same pattern as Task 7 with `play_type="pass"` and output column `opp_pass_epa_allowed_l4`. The WR builder's existing import + opp-strength block + output-merge are structurally identical to QB's. WR also has the `test_empty_depth_chart.py` test that validates schema columns for an empty DataFrame — verify column rename there too.

- [ ] **Step 1: Re-read each file**

Run:
```
grep -n "opp_allowed_wr_fppg" src/projections/schemas.py src/projections/features/wr.py src/projections/models/__init__.py tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py tests/test_features/test_empty_depth_chart.py
```

- [ ] **Step 2: Update test references**

Replace `"opp_allowed_wr_fppg_l4"` → `"opp_pass_epa_allowed_l4"` in `test_wr.py`, `test_wr_leakage.py`, and `test_empty_depth_chart.py` (if it references the column).

Run: `pytest tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py tests/test_features/test_empty_depth_chart.py -v`
Expected: FAIL.

- [ ] **Step 3: Update `WrFeaturesSchema`**

```python
# Before
    opp_allowed_wr_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)
# After
    opp_pass_epa_allowed_l4: Series[float] = pa.Field(nullable=True)
```

- [ ] **Step 4: Update `src/projections/features/wr.py`**

Same pattern as Task 6. Replace `opp_allowed_fppg` block with `opp_epa_allowed_residual(pbp_window, play_type="pass", n_weeks=4)`. Add `pbp: pd.DataFrame` to signature.

- [ ] **Step 5: Update WR feature_columns in models/__init__.py**

Replace `opp_allowed_wr_fppg_l4` with `opp_pass_epa_allowed_l4`.

- [ ] **Step 6: Add `pbp` arg to test calls**

Same as Task 6 step 6.

- [ ] **Step 7: Run tests + mypy + ruff**

Run:
```
pytest tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py tests/test_features/test_empty_depth_chart.py -v
mypy src/projections/features/wr.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/
ruff check src/projections/features/wr.py tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py
ruff format --check src/projections/features/wr.py tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py
```
Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/projections/features/wr.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_wr.py tests/test_features/test_wr_leakage.py tests/test_features/test_empty_depth_chart.py
git commit -m "feat(features): swap WR opp-fppg → opp_pass_epa_allowed_l4 — Plan 9 Phase 3"
```

---

## Task 9: TE schema swap + builder update

**Files:**
- Modify: `src/projections/schemas.py` (swap `TeFeaturesSchema` column)
- Modify: `src/projections/features/te.py`
- Modify: `tests/test_features/test_te.py`
- Modify: `tests/test_features/test_te_leakage.py`
- Modify: `src/projections/models/__init__.py`

Identical pattern to WR (TE also uses `play_type="pass"` and `opp_pass_epa_allowed_l4`).

- [ ] **Step 1: Re-read each file**

Run:
```
grep -n "opp_allowed_te_fppg" src/projections/schemas.py src/projections/features/te.py src/projections/models/__init__.py tests/test_features/test_te.py tests/test_features/test_te_leakage.py
```

- [ ] **Step 2: Update test references**

Replace `"opp_allowed_te_fppg_l4"` → `"opp_pass_epa_allowed_l4"` in both TE test files.

Run: `pytest tests/test_features/test_te.py tests/test_features/test_te_leakage.py -v`
Expected: FAIL.

- [ ] **Step 3: Update `TeFeaturesSchema`**

```python
# Before
    opp_allowed_te_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)
# After
    opp_pass_epa_allowed_l4: Series[float] = pa.Field(nullable=True)
```

- [ ] **Step 4: Update `src/projections/features/te.py`**

Same pattern as Task 8 (TE uses pass-EPA, just like WR). Add `pbp: pd.DataFrame` to signature.

- [ ] **Step 5: Update TE feature_columns in models/__init__.py**

Replace `opp_allowed_te_fppg_l4` with `opp_pass_epa_allowed_l4`.

- [ ] **Step 6: Add `pbp` arg to test calls**

Same as Task 6 step 6.

- [ ] **Step 7: Run tests + mypy + ruff**

Run:
```
pytest tests/test_features/test_te.py tests/test_features/test_te_leakage.py -v
mypy src/projections/features/te.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_te.py tests/test_features/test_te_leakage.py
ruff check src/projections/features/te.py tests/test_features/test_te.py tests/test_features/test_te_leakage.py
ruff format --check src/projections/features/te.py tests/test_features/test_te.py tests/test_features/test_te_leakage.py
```
Expected: green.

- [ ] **Step 8: Run full feature test suite to confirm Phase 3 done**

Run: `pytest tests/test_features/ -v`
Expected: all green. WR + QB + RB + TE + opponent helper all aligned.

- [ ] **Step 9: Commit**

```bash
git add src/projections/features/te.py src/projections/schemas.py src/projections/models/__init__.py tests/test_features/test_te.py tests/test_features/test_te_leakage.py
git commit -m "feat(features): swap TE opp-fppg → opp_pass_epa_allowed_l4 — Plan 9 Phase 3"
```

**End of Phase 3 — verification:**

```
pytest tests/test_features/ tests/test_models/ tests/test_schemas/ -v
mypy src tests
ruff check src tests scripts
ruff format --check src tests scripts
```
All four must be green before starting Phase 4.

---

## Task 10: Caller plumbing — thread `pbp` through 4 scripts

**Files:**
- Modify: `scripts/refresh_features.py`
- Modify: `scripts/train_baseline.py`
- Modify: `scripts/predict_2024.py`
- Modify: `scripts/sanity_check_baseline.py`
- Test: `tests/test_scripts/test_refresh_features.py` (likely exists; if not, add a minimal one)

Each script invokes `dispatch.feature_builder(...)` directly. The new signature requires `pbp: pd.DataFrame`. Each script must load the PBP partition for the requested seasons before calling.

- [ ] **Step 1: Re-read each script and find the builder call**

Run:
```
grep -n "feature_builder\|read_partition" scripts/refresh_features.py scripts/train_baseline.py scripts/predict_2024.py scripts/sanity_check_baseline.py
```

- [ ] **Step 2: Pattern for each script — load PBP partitions**

For each script, before the `builder(...)` call, add:

```python
from projections.store import read_partition

pbp_frames: list[pd.DataFrame] = []
for s in seasons:
    try:
        pbp_frames.append(read_partition(data_root / "raw", "pbp", season=s))
    except FileNotFoundError:
        # Pre-Plan-9 ingest hasn't run for this season yet — degrade
        # gracefully with an empty frame; the helper handles it.
        pass
pbp = pd.concat(pbp_frames, ignore_index=True) if pbp_frames else pd.DataFrame()
```

Then add `pbp=pbp` to the `builder(...)` call.

(Where `seasons` and `data_root` are already in scope; if not, use the script's existing variable names.)

- [ ] **Step 3: Apply to `scripts/refresh_features.py`**

Re-read the file once more, then edit. Add the load step before the existing builder invocation. The exact line numbers will depend on what's there; preserve the surrounding code.

- [ ] **Step 4: Apply to `scripts/train_baseline.py`**

Same pattern.

- [ ] **Step 5: Apply to `scripts/predict_2024.py`**

Same pattern.

- [ ] **Step 6: Apply to `scripts/sanity_check_baseline.py`**

Same pattern.

- [ ] **Step 7: Run any existing script tests**

Run: `pytest tests/test_scripts/ -v`
Expected: green (most script tests are smoke / argparse / IO; the new PBP load step degrades gracefully when no partition exists).

- [ ] **Step 8: Smoke each script with --help**

Run:
```
python -m scripts.refresh_features --help
python -m scripts.train_baseline --help
python -m scripts.predict_2024 --help
python -m scripts.sanity_check_baseline --help
```
Expected: each prints usage without import errors. (Don't actually run the scripts in CI — they need real data and downstream parquet partitions.)

- [ ] **Step 9: Run mypy + ruff on touched scripts**

Run:
```
mypy scripts/refresh_features.py scripts/train_baseline.py scripts/predict_2024.py scripts/sanity_check_baseline.py
ruff check scripts/refresh_features.py scripts/train_baseline.py scripts/predict_2024.py scripts/sanity_check_baseline.py
ruff format --check scripts/refresh_features.py scripts/train_baseline.py scripts/predict_2024.py scripts/sanity_check_baseline.py
```
Expected: green.

- [ ] **Step 10: Commit**

```bash
git add scripts/refresh_features.py scripts/train_baseline.py scripts/predict_2024.py scripts/sanity_check_baseline.py
git commit -m "feat(scripts): thread pbp through direct-builder scripts — Plan 9 Phase 4"
```

---

## Task 11: Adoption-gate CLI dual-run mode

**Files:**
- Modify: `scripts/adoption_gate.py` (add `--baseline-run` / `--candidate-run` mode)
- Modify: `tests/test_scripts/test_adoption_gate.py` (add 5–8 tests for dual-run mode)

The new mode loads two `results.parquet` files separately, treats one as incumbent and the other as candidate (overriding the `model_class` column for pairing purposes), then proceeds through the existing pair / bootstrap / verdict pipeline unchanged.

- [ ] **Step 1: Re-read `scripts/adoption_gate.py` and the existing test file**

Run:
```
cat scripts/adoption_gate.py | head -150
ls tests/test_scripts/
grep -n "adoption_gate\|--run\|--candidate\|--baseline" tests/test_scripts/test_adoption_gate.py | head -20
```
Expected: confirm current arg layout (`--run` + `--candidate` + `--incumbent`), and where pair_rows is invoked.

- [ ] **Step 2: Write failing tests for dual-run mode**

Append to `tests/test_scripts/test_adoption_gate.py`:

```python
def test_dual_run_mode_loads_two_runs_and_emits_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--baseline-run + --candidate-run produces verdicts equivalent to the
    single-run path when both runs hold the same model_class but different
    feature sets."""
    # Build two runs with identical (gsis_id, season, week) coverage.
    rows_baseline = pd.DataFrame(
        {
            "gsis_id": ["00-0034857"] * 4,
            "season": [2024] * 4,
            "week": [1, 2, 3, 4],
            "position": ["QB"] * 4,
            "model_class": ["baseline"] * 4,
            "predicted": [10.0, 12.0, 14.0, 16.0],
            "actual": [11.0, 11.5, 13.0, 17.0],
        }
    )
    rows_candidate = rows_baseline.copy()
    rows_candidate["predicted"] = [10.5, 11.8, 13.5, 16.5]  # closer to actuals

    baseline_dir = tmp_path / "run_baseline"
    candidate_dir = tmp_path / "run_candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    rows_baseline.to_parquet(baseline_dir / "results.parquet")
    rows_candidate.to_parquet(candidate_dir / "results.parquet")

    csv_path = tmp_path / "out.csv"

    # Invoke main with the dual-run args.
    monkeypatch.setattr(
        "sys.argv",
        [
            "adoption_gate",
            "--baseline-run", str(baseline_dir),
            "--candidate-run", str(candidate_dir),
            "--csv-out", str(csv_path),
            "--n-bootstrap", "100",
            "--seed", "42",
        ],
    )
    from scripts.adoption_gate import main
    main()
    assert csv_path.is_file()
    df = pd.read_csv(csv_path)
    assert "QB" in df["position"].tolist()


def test_dual_run_mutually_exclusive_with_single_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing both --run and --baseline-run must fail loudly."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "adoption_gate",
            "--run", str(tmp_path / "x"),
            "--baseline-run", str(tmp_path / "y"),
            "--candidate-run", str(tmp_path / "z"),
        ],
    )
    from scripts.adoption_gate import main
    with pytest.raises(SystemExit):
        main()


def test_dual_run_requires_both_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--baseline-run without --candidate-run must fail."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "adoption_gate",
            "--baseline-run", str(tmp_path / "x"),
        ],
    )
    from scripts.adoption_gate import main
    with pytest.raises(SystemExit):
        main()


def test_dual_run_pair_rows_treats_runs_as_incumbent_candidate(
    tmp_path: Path,
) -> None:
    """The dual-run loader assigns model_class=incumbent / candidate per file,
    regardless of the underlying values."""
    from scripts.adoption_gate import load_dual_run_paired

    rows_baseline = pd.DataFrame(
        {
            "gsis_id": ["00-0034857", "00-0036322"],
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["QB", "WR"],
            "model_class": ["something", "anything"],  # ignored
            "predicted": [10.0, 8.0],
            "actual": [11.0, 9.0],
        }
    )
    rows_candidate = rows_baseline.copy()
    rows_candidate["predicted"] = [10.5, 8.5]

    baseline_dir = tmp_path / "b"
    candidate_dir = tmp_path / "c"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    rows_baseline.to_parquet(baseline_dir / "results.parquet")
    rows_candidate.to_parquet(candidate_dir / "results.parquet")

    df = load_dual_run_paired(baseline_dir, candidate_dir)
    assert set(df["model_class"].unique()) == {"_baseline_run", "_candidate_run"}
    assert len(df) == 4  # 2 rows × 2 runs


def test_dual_run_row_coverage_mismatch_raises(
    tmp_path: Path,
) -> None:
    """If the two runs have different row coverage, the loader raises clearly."""
    from scripts.adoption_gate import load_dual_run_paired

    rows_baseline = pd.DataFrame(
        {
            "gsis_id": ["00-0034857", "00-0036322"],
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["QB", "WR"],
            "model_class": ["baseline", "baseline"],
            "predicted": [10.0, 8.0],
            "actual": [11.0, 9.0],
        }
    )
    rows_candidate = rows_baseline.iloc[:1].copy()  # missing 1 row

    baseline_dir = tmp_path / "b"
    candidate_dir = tmp_path / "c"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    rows_baseline.to_parquet(baseline_dir / "results.parquet")
    rows_candidate.to_parquet(candidate_dir / "results.parquet")

    with pytest.raises(ValueError, match="row coverage"):
        load_dual_run_paired(baseline_dir, candidate_dir)
```

- [ ] **Step 3: Run tests, expect ImportError / argparse failure**

Run: `pytest tests/test_scripts/test_adoption_gate.py -v -k dual`
Expected: FAIL — `load_dual_run_paired` doesn't exist; `--baseline-run` not recognized.

- [ ] **Step 4: Extend `scripts/adoption_gate.py`**

Add a new helper near `load_run_parquet`:

```python
def load_dual_run_paired(baseline_run: Path, candidate_run: Path) -> pd.DataFrame:
    """Load two backtest runs and return a single combined frame with
    model_class column synthesized to '_baseline_run' / '_candidate_run' so
    the existing pair_rows logic can pair them.

    Both runs must hold identical (gsis_id, season, week, position) coverage.
    """
    base = load_run_parquet(baseline_run)
    cand = load_run_parquet(candidate_run)

    keys = ["gsis_id", "season", "week", "position"]
    base_keys = base[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    cand_keys = cand[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    if not base_keys.equals(cand_keys):
        only_in_base = (
            base_keys.merge(cand_keys, on=keys, how="left", indicator=True)
            .query('_merge == "left_only"')
        )
        only_in_cand = (
            cand_keys.merge(base_keys, on=keys, how="left", indicator=True)
            .query('_merge == "left_only"')
        )
        raise ValueError(
            f"row coverage mismatch between runs: "
            f"{len(only_in_base)} rows only in baseline-run, "
            f"{len(only_in_cand)} rows only in candidate-run. "
            "Both runs must hold identical (gsis_id, season, week, position) coverage."
        )

    base = base.assign(model_class="_baseline_run")
    cand = cand.assign(model_class="_candidate_run")
    return pd.concat([base, cand], ignore_index=True)
```

Update `main()` (or the argparse parser construction) to accept the new args. The existing `--run` is now mutually exclusive with `(--baseline-run, --candidate-run)`. Implementation pattern:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(...)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path)
    mode.add_argument("--baseline-run", type=Path)
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--candidate", default=None)  # required only in single-run mode
    parser.add_argument("--incumbent", default="baseline")
    # ... existing args
    args = parser.parse_args(argv)

    if args.baseline_run is not None and args.candidate_run is None:
        parser.error("--baseline-run requires --candidate-run")
    if args.run is not None and args.candidate is None:
        parser.error("--run requires --candidate")
    return args
```

In `main()`, branch on which mode was selected:

```python
if args.baseline_run is not None:
    df = load_dual_run_paired(args.baseline_run, args.candidate_run)
    incumbent_class = "_baseline_run"
    candidate_class = "_candidate_run"
else:
    df = load_run_parquet(args.run)
    incumbent_class = args.incumbent
    candidate_class = args.candidate

validate_model_classes_present(df, incumbent=incumbent_class, candidate=candidate_class)
# ... rest of main proceeds unchanged
```

- [ ] **Step 5: Run tests, expect pass**

Run: `pytest tests/test_scripts/test_adoption_gate.py -v`
Expected: all tests pass (existing single-run tests still work; new dual-run tests pass).

- [ ] **Step 6: Run mypy + ruff**

Run:
```
mypy scripts/adoption_gate.py tests/test_scripts/test_adoption_gate.py
ruff check scripts/adoption_gate.py tests/test_scripts/test_adoption_gate.py
ruff format --check scripts/adoption_gate.py tests/test_scripts/test_adoption_gate.py
```
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add scripts/adoption_gate.py tests/test_scripts/test_adoption_gate.py
git commit -m "feat(adoption-gate-cli): add --baseline-run/--candidate-run dual-run mode — Plan 9 Phase 5"
```

**End of Phase 5 — full verification before real-data work:**

```
pytest -v
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
pytest -v -k "ingest or store or schemas"
```
All five must be green before moving to Phase 6.

---

## Task 12: Real-data PBP ingest

**Files:**
- Run-only: `scripts/refresh_features.py` invocations + new `data/raw/pbp/season=YYYY/` partitions

This task is operational — it pulls live data and writes partitions to disk. No code changes.

- [ ] **Step 1: Activate the worktree's venv**

Per CONTRIBUTING.md, each worktree needs its own .venv activation per shell:

```
cd .worktrees/feat-plan-9-pbp
python -m venv .venv
source .venv/Scripts/activate  # Windows bash
pip install -e ".[dev]"
pre-commit install
```

(If a .venv already exists, just activate.)

- [ ] **Step 2: Run the network smoke once to verify upstream is healthy**

Run: `pytest tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema -v -m network --run-network`
Expected: pass. If it fails on column drift, patch `_KEEP` / `_normalize_one_season` per CONTRIBUTING.md before proceeding.

- [ ] **Step 3: Run ingest for 2018-2024**

```
python -c "
from pathlib import Path
from projections.ingest.pbp import refresh_pbp
written = refresh_pbp(Path('data'), seasons=range(2018, 2025))
for p in written:
    print(p)
"
```

Expected: 7 partitions written, ~4-7 MB each. Total ~30-50 MB under `data/raw/pbp/`.

- [ ] **Step 4: Verify each partition reads cleanly**

```
python -c "
from pathlib import Path
from projections.schemas import PbpSchema
from projections.store import read_partition
for s in range(2018, 2025):
    df = read_partition(Path('data/raw'), 'pbp', season=s)
    PbpSchema.validate(df)
    print(f'{s}: {len(df)} rows, {df[\"epa\"].notna().sum()} epa-non-null')
"
```

Expected: each season prints ~50k rows total, ~40k epa-non-null. Pandera validation passes.

- [ ] **Step 5: Verify manifest**

```
python -c "
import pandas as pd
m = pd.read_parquet('data/manifests/ingest_manifest.parquet')
print(m[m['table'] == 'pbp'])
"
```

Expected: 7 rows, one per season 2018-2024.

- [ ] **Step 6: No commit needed (data is gitignored)**

`data/` is in `.gitignore`. Skip git add. The partitions exist locally for downstream tasks.

---

## Task 13: Pre-Plan-9 baseline backtest run

**Files:**
- Run-only: produces `data/backtest/run_pre_plan9_baseline/results.parquet`

This captures the v1 baseline predictions as the gate's incumbent. Runs against the pre-Plan-9 commit so feature builders use `opp_allowed_*_fppg_l4`.

- [ ] **Step 1: Save current state, checkout pre-Plan-9 commit**

The pre-Plan-9 commit is `origin/main`'s HEAD (i.e., `6675359` — Plan 8 merged). Plan 9's branch was forked from it, so the pre-Plan-9 state is just `main`.

In a separate worktree (NOT this Plan 9 worktree), check out main:

```
cd /c/Users/alden/FantasyFootball  # main worktree
git checkout main
git pull origin main
```

(The Plan 9 worktree at `.worktrees/feat-plan-9-pbp` stays on its branch; we use the main worktree for the pre-Plan-9 run.)

- [ ] **Step 2: Refresh features under the pre-Plan-9 model**

In the main worktree (with its own .venv activated):

```
python -m scripts.refresh_features all --seasons 2018-2024
```

Expected: feature partitions regenerate under `data/features/{position}/season=YYYY/week=WW/part.parquet` using v1 `opp_allowed_*_fppg_l4` features.

- [ ] **Step 3: Run pre-Plan-9 backtest in --report mode**

```
python -m scripts.backtest --model baseline --report
```

Expected: writes `data/backtest/run_<ts>/results.parquet` for the per-row predictions.

- [ ] **Step 4: Rename the run dir for stable reference**

```
mv data/backtest/run_<latest_ts> data/backtest/run_pre_plan9_baseline
```

(Replace `<latest_ts>` with the actual timestamp directory name.)

- [ ] **Step 5: Verify the file exists**

```
ls -la data/backtest/run_pre_plan9_baseline/results.parquet
python -c "
import pandas as pd
df = pd.read_parquet('data/backtest/run_pre_plan9_baseline/results.parquet')
print(df.shape, df['position'].value_counts())
"
```

Expected: ~20k+ rows; per-position counts match the held-out years' player coverage.

- [ ] **Step 6: Switch the main worktree back to a non-active branch (optional)**

```
git checkout main
```

(Stays on main; no commit needed. The Plan 9 work continues in `.worktrees/feat-plan-9-pbp`.)

---

## Task 14: Post-Plan-9 backtest run + adoption-gate verdicts

**Files:**
- Run-only: produces `data/backtest/run_post_plan9/results.parquet` and `reports/adoption_gate_plan9.csv`
- Modify: `docs/superpowers/specs/2026-04-29-plan-9-pbp-ingest-opp-epa-design.md` (append §6 verdicts table)

- [ ] **Step 1: In the Plan 9 worktree, refresh features under the new feature set**

```
cd .worktrees/feat-plan-9-pbp
source .venv/Scripts/activate
python -m scripts.refresh_features all --seasons 2018-2024
```

Expected: regenerates feature partitions with `opp_pass_epa_allowed_l4` / `opp_run_epa_allowed_l4` columns.

- [ ] **Step 2: Run post-Plan-9 backtest**

```
python -m scripts.backtest --model baseline --report
```

Expected: writes `data/backtest/run_<new_ts>/results.parquet`.

- [ ] **Step 3: Rename the run dir**

```
mv data/backtest/run_<new_ts> data/backtest/run_post_plan9
```

- [ ] **Step 4: Run the adoption gate in dual-run mode**

```
python -m scripts.adoption_gate \
  --baseline-run data/backtest/run_pre_plan9_baseline \
  --candidate-run data/backtest/run_post_plan9 \
  --csv-out reports/adoption_gate_plan9.csv
```

Expected: emits a per-position verdict table to stdout and writes the CSV. Capture the stdout to a file:

```
python -m scripts.adoption_gate \
  --baseline-run data/backtest/run_pre_plan9_baseline \
  --candidate-run data/backtest/run_post_plan9 \
  --csv-out reports/adoption_gate_plan9.csv \
  > reports/adoption_gate_plan9.md \
  2> reports/adoption_gate_plan9.stderr
```

- [ ] **Step 5: Append verdict table to spec §6**

Edit `docs/superpowers/specs/2026-04-29-plan-9-pbp-ingest-opp-epa-design.md`. After §6 step 7, append:

```markdown
### §6 verdicts (run_<post_ts>)

Paired bootstrap, n_bootstrap=1000, seed=42. Pairing key (gsis_id, season, week, position).

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---|
| QB | <verdict> | <ci> | <ci> | <n> |
| RB | <verdict> | <ci> | <ci> | <n> |
| TE | <verdict> | <ci> | <ci> | <n> |
| WR | <verdict> | <ci> | <ci> | <n> |

Per-year breakdown is emitted in `reports/adoption_gate_plan9.csv`.

### §6 routing changes shipped

| Position | Pre-Plan-9 feature column | Post-Plan-9 feature column | Reason |
|---|---|---|---|
| QB | opp_allowed_qb_fppg_l4 | <kept old / replaced with opp_pass_epa_allowed_l4> | <verdict-driven> |
| RB | opp_allowed_rb_fppg_l4 | <kept old / replaced with opp_run_epa_allowed_l4> | <verdict-driven> |
| TE | opp_allowed_te_fppg_l4 | <kept old / replaced with opp_pass_epa_allowed_l4> | <verdict-driven> |
| WR | opp_allowed_wr_fppg_l4 | <kept old / replaced with opp_pass_epa_allowed_l4> | <verdict-driven> |
```

Replace each `<verdict>` / `<ci>` / `<n>` / `<verdict-driven>` placeholder with the actual values from the gate's CSV/stdout output.

- [ ] **Step 6: Commit verdicts**

```bash
git add docs/superpowers/specs/2026-04-29-plan-9-pbp-ingest-opp-epa-design.md reports/adoption_gate_plan9.md reports/adoption_gate_plan9.csv reports/adoption_gate_plan9.stderr
git commit -m "docs(plan-9): adoption gate verdicts captured in spec §6 — Plan 9 Phase 6"
```

---

## Task 15: Per-position revert + final snapshot update

**Files:** depends on per-position verdicts.

If verdict for any position is **DO_NOT_ADOPT**, revert that position's feature change. If verdict is **MARGINAL**, pause and consult the per-year breakdown + user before deciding.

- [ ] **Step 1: Determine action per position**

For each position P:
- ADOPT → no revert; continue.
- DO_NOT_ADOPT → revert P's feature change in the next steps.
- MARGINAL → STOP. Surface the per-year breakdown to the user. Continue only after the user resolves.

- [ ] **Step 2 (per DO_NOT_ADOPT position): Revert P's feature change**

For each non-adopting P, undo the changes from Tasks 6–9:
- In `src/projections/schemas.py`: restore `opp_allowed_<P>_fppg_l4: Series[float] = pa.Field(ge=0, nullable=True)` and remove the new `opp_pass_epa_allowed_l4` (or `opp_run_epa_allowed_l4`).
- In `src/projections/features/<P>.py`: restore the v1 `opp_allowed_fppg` import + invocation. (You can use `git show <plan-8-merge-commit>:src/projections/features/<P>.py` to retrieve the original.)
- In `src/projections/models/__init__.py`: restore the old column name in the P feature_columns list.
- In `tests/test_features/test_<P>.py` and `test_<P>_leakage.py`: revert column-name updates.

Also: keep the new `pbp` arg in the builder signature, but make it optional with a default of `pd.DataFrame()`, so callers don't need to differentiate between adopting and non-adopting positions:

```python
def build_<P>_features(
    ...,
    pbp: pd.DataFrame = pd.DataFrame(),  # ignored on non-adopting positions
) -> pd.DataFrame:
```

This keeps the caller plumbing in scripts/* unchanged.

If ZERO positions adopt: revert all four positions' changes AND the v1 helper revert in `_opponent.py`. PBP ingest, PbpSchema, network smoke, and the `pbp` arg threading still ship as plumbing for future feature plans.

- [ ] **Step 3: Run tests after each per-position revert**

Run after each position revert:
```
pytest tests/test_features/test_<P>.py tests/test_features/test_<P>_leakage.py -v
```
Expected: green.

- [ ] **Step 4: Refresh the feature cache to reflect the final state**

```
python -m scripts.refresh_features all --seasons 2018-2024
```

Expected: regenerates all per-position feature partitions with the post-revert column set.

- [ ] **Step 5: Run the snapshot regression gate with --update-snapshot**

```
python -m scripts.backtest --model baseline --update-snapshot
```

Expected: writes a new `tests/backtest/model_metrics.json`. Drift is expected on adopted-position cells; non-adopted-position cells should match pre-Plan-9 baseline.

- [ ] **Step 6: Run the snapshot regression gate in --check mode to confirm clean diff**

```
python -m scripts.backtest --model baseline --check
```

Expected: PASS — the snapshot we just wrote matches the run we just produced.

- [ ] **Step 7: Run all gates**

```
pytest -v
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

Expected: green across the board.

- [ ] **Step 8: Commit**

If any reverts happened:
```bash
git add src tests
git commit -m "refactor(features): revert <positions> per Plan 9 adoption-gate verdict — Plan 9 Phase 7"
```

Always commit the final snapshot:
```bash
git add tests/backtest/model_metrics.json
git commit -m "test(backtest): refresh snapshot after Plan 9 feature swap — Plan 9 Phase 7"
```

---

## Task 16: PM doc + TODO updates

**Files:**
- Modify: `project_management.md` (append Plan 9 entry at top + decision-log rows + Current status / Next action)
- Modify: `TODO.md` (split TODO #3 into 3a closed / 3b open; add per-non-adopting-position follow-up entries if any)

- [ ] **Step 1: Append a new Plan 9 entry at the top of `project_management.md`**

Append (with concrete verdicts replacing placeholders):

```markdown
## Plan 9 — PBP ingest + opponent-adjusted EPA features — complete; <N> positions adopted (2026-04-29 + ..., on branch `feat/plan-9-pbp-ingest-opp-epa`)

**Status:** all 7 phases shipped. Verification gates green: pytest <count> passed, mypy clean, ruff check + format clean. Per-position adoption verdicts: QB <V>, RB <V>, TE <V>, WR <V>.

### What shipped

- `src/projections/ingest/pbp.py` — refresh_pbp module covering `nfl_data_py.import_pbp_data` for 2018-2024 with a curated 27-column subset.
- `PbpSchema` in `src/projections/schemas.py`.
- `opp_epa_allowed_residual` in `src/projections/features/_opponent.py` replacing the v1 `opp_allowed_fppg` proxy.
- Per-position FeaturesSchema column swaps for adopted positions (`opp_allowed_<pos>_fppg_l4` → `opp_pass_epa_allowed_l4` / `opp_run_epa_allowed_l4`).
- `pbp` arg threaded through 4 direct-builder scripts (refresh_features, train_baseline, predict_2024, sanity_check_baseline).
- `scripts/adoption_gate.py` extended with `--baseline-run` / `--candidate-run` dual-run mode for cross-run pairing.
- New tests: <count> ingest + <count> opponent + <count> dual-run-CLI.
- Opt-in `--run-network` smoke for PBP.
- Adoption-gate verdicts captured in spec §6.

### Per-position adoption verdicts (run_<post_ts>)

[copy from spec §6 verdicts table]

### Per-position routing changes shipped

[copy from spec §6 routing changes table]

### Next track after Plan 9

TODO #3 split into 3a (closed by this plan) and 3b (remaining feature slices). Pick one of pace, PROE, air-yards distributions, pressure rate, redzone usage shares as the next plan.
```

- [ ] **Step 2: Append decision-log rows in `project_management.md`**

After the existing decision log table, append:

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-29 | PBP storage shape: raw per-play parquet, per-season partition, ~25-column curated subset | Mirrors weekly_stats.py template; future feature plans extend `_KEEP` additively without forcing a re-ingest. |
| 2026-04-29 | "Opp-adjusted" interpretation = schedule-of-strength residual (per-play residual = EPA - offense's overall mean EPA in the same trailing window) | Standard nfl-stats formulation; the v1 fppg's flaw was lack of schedule adjustment, so any non-residual formulation would have shipped a renamed v1. |
| 2026-04-29 | Replace v1 fppg directly rather than running side-by-side | Side-by-side muddles the experiment (Ridge sees both, can't attribute the lift). Adoption gate decides per-position; revert is a one-commit undo. |
| 2026-04-29 | Per-position adoption: <verdict summary> | Adoption gate ran with paired bootstrap (n=1000, seed=42). Routing changes shipped per the verdicts above. |
| 2026-04-29 | Adoption-gate CLI extended with --baseline-run/--candidate-run dual-run mode | Plan 8's CLI assumed model-class-vs-model-class within ONE run; Plan 9 needs feature-set-vs-feature-set across two runs. Same paired-bootstrap math, different inputs. Amortizes across every future feature-class plan. |

- [ ] **Step 3: Update "Current status" / "Next action" sections**

Update the existing "Current status" / "Next action" blocks to reflect Plan 9 done and queue the next-up feature plan in TODO #3b.

- [ ] **Step 4: Update `TODO.md`**

Rename TODO #3 to `### 3a. Play-by-play ingest + opponent-adjusted EPA features — closed in Plan 9 (2026-04-29)` and reduce its body to a brief closure note + pointer to the Plan 9 entry.

Add `### 3b. PBP-derived feature plans on top of Plan 9 plumbing` capturing the remaining feature slices (pace, PROE, air-yards distributions, pressure rate, redzone usage shares) as a backlog.

If any position's verdict was DO_NOT_ADOPT, add an entry capturing the negative result for that position (e.g., "TODO #N: RB pass-EPA defer — Plan 9 verdict was DO_NOT_ADOPT for RB; investigate whether RB pass-catching role split would change the call").

- [ ] **Step 5: Verify markdown renders**

Run: `python -m mdformat --check project_management.md TODO.md` (if mdformat is in deps; else just visually scan).

- [ ] **Step 6: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(plan-9): mark Plan 9 complete in PM doc; split TODO #3 into 3a/3b — Plan 9 Phase 7"
```

---

## Final verification — branch ready for PR

After Task 16:

- [ ] **Step 1: Confirm clean tip**

```
git status
git log --oneline -16
```

Expected: working tree clean; ~12-16 commits since fork point covering Phases 1-7.

- [ ] **Step 2: Run the full gate**

```
pytest -v
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
pytest -v -k "ingest or store or schemas"
```

Expected: 100% green.

- [ ] **Step 3: Push the branch + open PR**

```
git push -u origin feat/plan-9-pbp-ingest-opp-epa
gh pr create --title "Plan 9 — PBP ingest + opp-adjusted EPA features" --body "$(cat <<'EOF'
## Summary
- New ingest source: nfl_data_py PBP (curated 27-column subset, 2018-2024).
- New opp_epa_allowed_residual helper replacing v1 opp_allowed_fppg.
- Per-position FeaturesSchema swaps per adoption-gate verdicts.
- Adoption-gate CLI extended for dual-run feature-comparison mode.

Spec: docs/superpowers/specs/2026-04-29-plan-9-pbp-ingest-opp-epa-design.md
Plan: docs/superpowers/plans/2026-04-29-plan-9-pbp-ingest-opp-epa.md

## Test plan
- [x] pytest (full suite green)
- [x] mypy / ruff / format checks clean
- [x] adoption gate verdicts captured in spec §6
- [x] snapshot regression gate green after final updates

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR opened. Return the URL to the user.

---

## Self-review notes (for the implementer)

- **Spec coverage:** Every section of `2026-04-29-plan-9-pbp-ingest-opp-epa-design.md` has at least one task implementing it. Tasks 1–4 cover §3 + §4.1 + §7 (ingest tests). Task 5 covers §5.1 + §7 (opponent helper). Tasks 6–9 cover §4.2 + §5.2 + §7 (per-position changes). Task 10 covers §5.3. Task 11 covers §1.3 + §2.1 deliverable 7 + §7. Tasks 12–14 cover §6. Task 15 covers §6 step 5–7. Task 16 covers §10.
- **No placeholders:** Every step contains either runnable code, an exact command, or a clearly-marked gate-run-time placeholder (Task 14 step 5's verdict-table fields, which depend on the actual gate output).
- **Type consistency:** `opp_epa_allowed_residual` signature matches between spec §5.1, Task 5 step 4, and per-position builder calls in Tasks 6–9. Output column names (`opp_pass_epa_allowed_l4` for QB/WR/TE; `opp_run_epa_allowed_l4` for RB) match across schema, builder, model feature_columns, and tests.
- **Edit-integrity:** Per CLAUDE.md rule 9, every multi-edit task starts with a re-read step. Tasks 6–9 each touch `schemas.py` — re-read at the start of each task.
