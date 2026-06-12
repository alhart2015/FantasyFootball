# H2H Draft-Strategy Backtest Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless backtest that replays a 16-team half-PPR fantasy league on real 2025 outcomes — draft from 2025 ESPN preseason projections + Sleeper ADP, set weekly lineups from ESPN weekly projections, score against real 2025 results, play a head-to-head season (weeks 1–17), and report each draft strategy's championship / win / playoff rates — the first metric that isn't a strategy's own objective.

**Architecture:** New sub-package `src/projections/backtest/` with one module per responsibility (data pull → draft basis → lineup → schedule → league → harness). Reuses the existing `DraftStrategy` strategies, constrained ADP-bot draft loop, `roster_eligibility`, the scoring layer, `store`, and `generate_vorp_table` (with the QB-replacement fix). All player projections/actuals are fixed real-2025 tables; only draft order (bot ADP jitter) and the matchup schedule vary per seed.

**Tech Stack:** Python 3.11, pandas + pyarrow, pandera (boundary validation), pydantic (`StatLine`/`LeagueConfig`), numpy (RNG/MC), pytest. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` on every test/run command (a known scipy segfault on this box).

**Source spec:** `docs/superpowers/specs/2026-06-12-h2h-backtest-harness-design.md`

**Branch dependency (resolve before Task 1):** This harness imports `SeasonValueStrategy` (branch `feat/depth-aware-draft-strategy`) and needs the VORP fix (`fix/vorp-replacement-calibration`, PR #63). The implementation branch must be based on a tree containing **both**. Recommended: land PR #63 and the depth-aware branch to `main`, then rebase `feat/h2h-backtest-harness` onto `main`; or stack this branch on a merge of the two. Verify `from projections.draft.assistant.strategy import SeasonValueStrategy` imports and `tests/test_draft/test_vorp.py::test_qb_replacement_independent_of_bench_depth` passes before starting.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/projections/backtest/__init__.py` | package marker |
| `src/projections/schemas.py` (modify) | add `WeeklyProjectionSchema`, `WeeklyActualSchema` |
| `src/projections/backtest/weekly_actuals.py` | score `weekly_stats` → per-(gsis_id, week) half-PPR actual points |
| `src/projections/backtest/espn_weekly.py` | pull + parse ESPN weekly projected stat lines → half-PPR weekly projection table |
| `src/projections/backtest/draft_basis.py` | build the 2025 Sleeper-ADP half-PPR fixed-VORP draft table |
| `src/projections/backtest/lineup.py` | `weekly_lineup_points` — fill lineup by projection, score by actual |
| `src/projections/backtest/schedule.py` | regular-season schedule (circle method) + single-elim playoff bracket |
| `src/projections/backtest/draft_field.py` | constrained ADP-bot draft loop (promoted from scratch) + mixed-field seat layout |
| `src/projections/backtest/league.py` | `simulate_league` — draft → weekly points → standings → playoffs → `LeagueResult` |
| `src/projections/backtest/harness.py` | `run_backtest` — mirrored seeds, per-strategy aggregation + bootstrap CIs |
| `src/projections/backtest/cli.py` | CLI core (`_parse_args`, `format_result`, `run`) — mirrors `tournament_cli.py` |
| `scripts/h2h_backtest.py` | 3-line wrapper calling `cli.run` |
| `tests/test_backtest/test_*.py` | one test module per source module |

**Reuse (do not re-implement):** `projections.scoring.score` (integer `StatLine`) for actuals; `projections.scoring.expected_points` (fractional) for projections; `projections.draft.assistant.roster_score.optimal_lineup_points` as the structural template for `weekly_lineup_points`; `projections.draft.roster_eligibility` (`POSITION_SLOTS`, `FLEX_ELIGIBLE`, `SUPER_FLEX_ELIGIBLE`); `projections.draft.assistant.strategy` (`NowOrNeverStrategy`/`SeasonValueStrategy`/`RawVorpStrategy`); `projections.draft.assistant.opponent.bot_pick`; `projections.draft.assistant.tournament._bootstrap_mean` + `Interval`; `projections.draft.vorp.generate_vorp_table`; `projections.store` partition I/O; `id_map` for espn_id→gsis_id.

---

## Phase 0 — Scaffold + schemas

### Task 1: Backtest package + weekly schemas

**Files:**
- Create: `src/projections/backtest/__init__.py` (empty)
- Create: `tests/test_backtest/__init__.py` (empty)
- Modify: `src/projections/schemas.py` (append two schemas near the other projection schemas)
- Test: `tests/test_backtest/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest/test_schemas.py
import pandas as pd
import pytest
from projections.schemas import WeeklyProjectionSchema, WeeklyActualSchema, _PYARROW_STR


def _proj_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "gsis_id": pd.array(["00-0000001"], dtype=_PYARROW_STR),
        "season": pd.array([2025], dtype="Int64"),
        "week": pd.array([5], dtype="Int64"),
        "position": pd.array(["RB"], dtype=_PYARROW_STR),
        "projected_points": pd.array([14.3], dtype="Float64"),
    })


def test_weekly_projection_schema_accepts_valid_frame():
    out = WeeklyProjectionSchema.validate(_proj_frame())
    assert len(out) == 1


def test_weekly_projection_rejects_week_18():
    bad = _proj_frame()
    bad["week"] = pd.array([18], dtype="Int64")
    with pytest.raises(Exception):
        WeeklyProjectionSchema.validate(bad)


def test_weekly_projection_allows_null_points():
    f = _proj_frame()
    f["projected_points"] = pd.array([None], dtype="Float64")
    assert len(WeeklyProjectionSchema.validate(f)) == 1


def test_weekly_actual_schema_accepts_valid_frame():
    f = pd.DataFrame({
        "gsis_id": pd.array(["00-0000001"], dtype=_PYARROW_STR),
        "season": pd.array([2025], dtype="Int64"),
        "week": pd.array([5], dtype="Int64"),
        "actual_points": pd.array([9.1], dtype="Float64"),
    })
    assert len(WeeklyActualSchema.validate(f)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_schemas.py -q`
Expected: FAIL — `ImportError: cannot import name 'WeeklyProjectionSchema'`.

- [ ] **Step 3: Add the schemas**

In `src/projections/schemas.py`, after the existing projection schemas (e.g. near `ProjectionSeasonSchema`), add. Match the file's existing pandera idiom (`pa.DataFrameModel`, `Config.strict = "filter"`, `coerce = True`):

```python
class WeeklyProjectionSchema(pa.DataFrameModel):
    """Per-(player, week) preseason-source weekly projection, scored to a ruleset.

    `projected_points` is nullable: a player with no ESPN weekly entry that week
    (bye / inactive) carries NULL and cannot be started.
    """

    gsis_id: Series[str] = pa.Field(coerce=False)  # already pyarrow-str from id_map crosswalk
    season: Series[pd.Int64Dtype] = pa.Field(ge=2000, le=2100)
    week: Series[pd.Int64Dtype] = pa.Field(ge=1, le=17)
    position: Series[str]
    projected_points: Series[pd.Float64Dtype] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True


class WeeklyActualSchema(pa.DataFrameModel):
    """Per-(player, week) realized fantasy points under a ruleset."""

    gsis_id: Series[str]
    season: Series[pd.Int64Dtype] = pa.Field(ge=2000, le=2100)
    week: Series[pd.Int64Dtype] = pa.Field(ge=1, le=17)
    actual_points: Series[pd.Float64Dtype]

    class Config:
        strict = "filter"
        coerce = True
```

Note: confirm the `gsis_id`/`position` string-dtype declaration matches the convention used by the *other* schemas in this file (some declare `Series[str]` with the pyarrow dtype enforced by a `_PYARROW_STR` check). Mirror the nearest existing schema exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_schemas.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py src/projections/backtest/__init__.py tests/test_backtest/
git commit -m "feat(backtest): WeeklyProjection/WeeklyActual schemas + package scaffold"
```

---

## Phase 1 — Data layer (2025)

### Task 2: `weekly_actuals.py` — score weekly_stats per week

**Files:**
- Create: `src/projections/backtest/weekly_actuals.py`
- Test: `tests/test_backtest/test_weekly_actuals.py`

Reuse: the `StatLine`→`score` row pattern from `src/projections/scoring/actuals.py:30-43` (the canonical scorer; do not re-implement scoring). Difference: keep **per-(gsis_id, week)** rows instead of season totals, and filter to weeks 1–17.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest/test_weekly_actuals.py
import pandas as pd
from projections.schemas import Ruleset, WeeklyActualSchema
from projections.backtest.weekly_actuals import build_weekly_actuals


def _ws_row(gsis, week, receptions=0, rec_yds=0.0, rush_yds=0.0, rush_td=0):
    return {
        "gsis_id": gsis, "season": 2025, "week": week, "position": "RB",
        "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
        "rushing_yards": rush_yds, "rushing_tds": rush_td,
        "receptions": receptions, "receiving_yards": rec_yds, "receiving_tds": 0,
        "fumbles_lost": 0,
    }


def test_scores_half_ppr_per_week():
    ws = pd.DataFrame([_ws_row("00-0000001", 5, receptions=4, rec_yds=40.0, rush_yds=50.0, rush_td=1)])
    out = build_weekly_actuals(ws, ruleset=Ruleset.espn_half())
    # half-PPR: 4 rec * 0.5 + 40*0.1 + 50*0.1 + 1*6 = 2 + 4 + 5 + 6 = 17.0
    assert float(out.loc[0, "actual_points"]) == 17.0
    WeeklyActualSchema.validate(out)


def test_excludes_week_18():
    ws = pd.DataFrame([_ws_row("00-0000001", 18, rush_yds=100.0)])
    out = build_weekly_actuals(ws, ruleset=Ruleset.espn_half())
    assert len(out) == 0


def test_one_row_per_player_week():
    ws = pd.DataFrame([_ws_row("00-0000001", 5), _ws_row("00-0000001", 6), _ws_row("00-0000002", 5)])
    out = build_weekly_actuals(ws, ruleset=Ruleset.espn_half())
    assert len(out) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_weekly_actuals.py -q`
Expected: FAIL — module `weekly_actuals` not found.

- [ ] **Step 3: Implement**

```python
# src/projections/backtest/weekly_actuals.py
"""Score a weekly_stats frame to per-(gsis_id, week) half-PPR actual points (weeks 1-17)."""
from __future__ import annotations

import pandas as pd

from projections.schemas import _PYARROW_STR, Ruleset, WeeklyActualSchema
from projections.scoring.score import StatLine, score

_MAX_WEEK = 17


def build_weekly_actuals(weekly_stats: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame:
    ws = weekly_stats[weekly_stats["week"] <= _MAX_WEEK].copy()
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
    out = pd.DataFrame({
        "gsis_id": ws["gsis_id"].astype(_PYARROW_STR).to_numpy(),
        "season": ws["season"].astype("Int64").to_numpy(),
        "week": ws["week"].astype("Int64").to_numpy(),
        "actual_points": pd.array(points, dtype="Float64"),
    })
    return WeeklyActualSchema.validate(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_weekly_actuals.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/weekly_actuals.py tests/test_backtest/test_weekly_actuals.py
git commit -m "feat(backtest): weekly_actuals — half-PPR points per player-week"
```

---

### Task 3: `espn_weekly.py` — parse a captured weekly payload (no network)

**Files:**
- Create: `src/projections/backtest/espn_weekly.py`
- Create: `tests/test_backtest/fixtures/espn_weekly_wk5_sample.json` (captured below)
- Test: `tests/test_backtest/test_espn_weekly.py`

Capture-first TDD: save one real ESPN weekly payload (trimmed to ~3 players) as a fixture so the parser is tested against the true shape, with no network in unit tests. The parse mirrors `scripts/pull_external_projections.py::parse_espn_players` but reads the **weekly** stat entry (`statSourceId==1, statSplitTypeId==1, scoringPeriodId==wk`) and scores via `expected_points` (fractional).

- [ ] **Step 1: Capture a real fixture (one-time, network)**

Run this to save a trimmed real payload:

```bash
OMP_NUM_THREADS=1 python - <<'PY'
import json, urllib.request
from projections.ingest.external_projections import _ESPN_URL, _UA
flt={'players':{'limit':5,'sortPercOwned':{'sortPriority':1,'sortAsc':False}}}
url=_ESPN_URL.format(season=2025)+'&scoringPeriodId=5'
req=urllib.request.Request(url, headers={'User-Agent':_UA,'X-Fantasy-Filter':json.dumps(flt)})
payload=json.load(urllib.request.urlopen(req, timeout=60))
# keep only the fields the parser reads, 3 players
trimmed={'players':[]}
for pl in payload['players'][:3]:
    p=pl['player']
    trimmed['players'].append({'player':{'id':p['id'],'fullName':p.get('fullName'),
        'defaultPositionId':p.get('defaultPositionId'),
        'stats':[s for s in p.get('stats',[]) if s.get('scoringPeriodId')==5]}})
import os; os.makedirs('tests/test_backtest/fixtures', exist_ok=True)
json.dump(trimmed, open('tests/test_backtest/fixtures/espn_weekly_wk5_sample.json','w'), indent=2)
print('saved', len(trimmed['players']), 'players')
PY
```

Inspect the saved file; confirm each player has a `stats` entry with `statSourceId==1, statSplitTypeId==1, scoringPeriodId==5` and a `stats` dict of numeric stat ids. (If a player is on bye that week it will have no such entry — keep it; it exercises the null path.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_backtest/test_espn_weekly.py
import json
from pathlib import Path
import pandas as pd
from projections.schemas import Ruleset, WeeklyProjectionSchema
from projections.backtest.espn_weekly import parse_espn_weekly

_FIX = Path(__file__).parent / "fixtures" / "espn_weekly_wk5_sample.json"


def test_parse_returns_one_row_per_projected_player():
    payload = json.loads(_FIX.read_text())
    df = parse_espn_weekly(payload, season=2025, week=5, ruleset=Ruleset.espn_half())
    # rows keyed by espn_id + projected_points (nullable for bye players)
    assert {"espn_id", "season", "week", "position", "projected_points"} <= set(df.columns)
    assert (df["week"] == 5).all()
    assert df["projected_points"].notna().any()  # at least one real projection


def test_projected_points_are_half_ppr_nonnegative():
    payload = json.loads(_FIX.read_text())
    df = parse_espn_weekly(payload, season=2025, week=5, ruleset=Ruleset.espn_half())
    vals = df["projected_points"].dropna()
    assert (vals >= 0).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_espn_weekly.py -q`
Expected: FAIL — `parse_espn_weekly` undefined.

- [ ] **Step 4: Implement the parser**

```python
# src/projections/backtest/espn_weekly.py
"""Pull + parse ESPN weekly projected stat lines (weeks 1-17) -> half-PPR weekly projection table.

Mirrors scripts/pull_external_projections.py's ESPN parse but reads the single-week
projection entry (statSourceId=1, statSplitTypeId=1, scoringPeriodId=wk) and scores the
fractional stat line via scoring.expected_points. espn_id->gsis_id crosswalk + store write
live in refresh_espn_weekly_projections (Task 4).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring.score import expected_points

# Reuse the ESPN stat-id -> StatLine-field map already defined for the season pull.
from projections.ingest.external_projections import ESPN_POSITIONS, ESPN_STAT_IDS


def _weekly_proj_stats(player: dict[str, Any], week: int) -> dict[str, float] | None:
    for s in player.get("stats", []):
        if (s.get("scoringPeriodId") == week
                and s.get("statSourceId") == 1
                and s.get("statSplitTypeId") == 1):
            return s.get("stats") or {}
    return None


def _statline_dict(raw: dict[str, float]) -> dict[str, float]:
    line = {field: 0.0 for field in ESPN_STAT_IDS.values()}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in raw:
            line[field] = float(raw[sid])
    return line


def parse_espn_weekly(payload: dict[str, Any], *, season: int, week: int, ruleset: Ruleset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pl in payload.get("players", []):
        p = pl.get("player", {})
        position = ESPN_POSITIONS.get(p.get("defaultPositionId"))
        if position is None:
            continue
        raw = _weekly_proj_stats(p, week)
        proj = expected_points(_statline_dict(raw), ruleset) if raw is not None else None
        rows.append({
            "espn_id": str(p.get("id")),
            "season": season,
            "week": week,
            "position": position,
            "projected_points": proj,
        })
    return pd.DataFrame(rows)
```

Note: `ESPN_STAT_IDS` / `ESPN_POSITIONS` are imported from the ingest module (confirm they're exported there; `scripts/pull_external_projections.py` imports them from `projections.ingest.external_projections`). If `expected_points` expects the canonical `StatLine` field names, `ESPN_STAT_IDS.values()` already are those names (that's what the season pull uses).

- [ ] **Step 5: Run test to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_espn_weekly.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/projections/backtest/espn_weekly.py tests/test_backtest/test_espn_weekly.py tests/test_backtest/fixtures/
git commit -m "feat(backtest): parse ESPN weekly projections -> half-PPR (fixture-tested)"
```

---

### Task 4: `espn_weekly.py` — refresh orchestrator + crosswalk + store + network smoke

**Files:**
- Modify: `src/projections/backtest/espn_weekly.py` (add `refresh_espn_weekly_projections`)
- Modify: `tests/test_backtest/test_espn_weekly.py` (add crosswalk test + opt-in network smoke)

`refresh_espn_weekly_projections(season, *, weeks, ruleset, id_map, data_root)`: for each week, fetch (reuse the season-pull `fetch_espn` URL + `&scoringPeriodId`), `parse_espn_weekly`, crosswalk `espn_id`→`gsis_id` via `id_map`, drop unmatched, validate `WeeklyProjectionSchema`, concat all weeks, `store.write_partition`.

- [ ] **Step 1: Write the failing crosswalk test**

```python
def test_crosswalk_espn_to_gsis(monkeypatch):
    import json
    from pathlib import Path
    from projections.backtest import espn_weekly as ew
    payload = json.loads((Path(__file__).parent / "fixtures" / "espn_weekly_wk5_sample.json").read_text())
    # stub the network fetch to return the fixture for any week
    monkeypatch.setattr(ew, "_fetch_espn_week", lambda season, week, limit=800: payload)
    # id_map mapping the fixture's espn_ids to fake gsis_ids
    espn_ids = [str(p["player"]["id"]) for p in payload["players"]]
    id_map = pd.DataFrame({
        "gsis_id": [f"00-{i:07d}" for i in range(len(espn_ids))],
        "espn_id": espn_ids,
        "position": ["RB"] * len(espn_ids),
    })
    out = ew.weekly_projections_for_weeks(season=2025, weeks=[5], ruleset=Ruleset.espn_half(), id_map=id_map)
    WeeklyProjectionSchema.validate(out)
    assert set(out["gsis_id"]) <= set(id_map["gsis_id"])
```

(Factor the in-memory assembly into `weekly_projections_for_weeks(...)` so it is testable without `store`; `refresh_espn_weekly_projections` is the thin wrapper that adds `fetch` + `store.write_partition`.)

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_espn_weekly.py::test_crosswalk_espn_to_gsis -q`
Expected: FAIL — `weekly_projections_for_weeks` / `_fetch_espn_week` undefined.

- [ ] **Step 3: Implement**

```python
# add to src/projections/backtest/espn_weekly.py
import json
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from projections.ingest.external_projections import _ESPN_URL, _UA
from projections.schemas import _PYARROW_STR, WeeklyProjectionSchema
from projections.store import write_partition

_DEFAULT_WEEKS = range(1, 18)


def _fetch_espn_week(season: int, week: int, *, limit: int = 800) -> dict[str, Any]:
    flt = {"players": {"limit": limit, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    url = _ESPN_URL.format(season=season) + f"&scoringPeriodId={week}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "X-Fantasy-Filter": json.dumps(flt)})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def weekly_projections_for_weeks(*, season: int, weeks: Iterable[int], ruleset: Ruleset, id_map: pd.DataFrame) -> pd.DataFrame:
    cross = id_map[["espn_id", "gsis_id"]].dropna().astype({"espn_id": str})
    frames: list[pd.DataFrame] = []
    for wk in weeks:
        parsed = parse_espn_weekly(_fetch_espn_week(season, wk), season=season, week=wk, ruleset=ruleset)
        merged = parsed.merge(cross, on="espn_id", how="inner")
        frames.append(merged[["gsis_id", "season", "week", "position", "projected_points"]])
    out = pd.concat(frames, ignore_index=True)
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    return WeeklyProjectionSchema.validate(out)


def refresh_espn_weekly_projections(*, season: int, ruleset: Ruleset, id_map: pd.DataFrame,
                                    data_root: Path, weeks: Iterable[int] = _DEFAULT_WEEKS) -> pd.DataFrame:
    out = weekly_projections_for_weeks(season=season, weeks=weeks, ruleset=ruleset, id_map=id_map)
    write_partition(data_root / "processed", out, "espn_weekly_projections", season=season)
    return out
```

Note: confirm `write_partition`'s exact signature against `src/projections/store/parquet.py:48` and an existing caller (e.g. `ingest/weekly_stats.py`); adjust the positional/keyword args to match. `_ESPN_URL` / `_UA` are module-private in the ingest — if import is disallowed by lint, re-declare the URL constant locally with a comment pointing at the source.

- [ ] **Step 4: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_espn_weekly.py -q`
Expected: PASS.

- [ ] **Step 5: Add an opt-in network smoke (guards ESPN payload drift)**

```python
import pytest

@pytest.mark.skipif("not config.getoption('--run-network', default=False)",
                    reason="hits ESPN; opt in with --run-network")
def test_espn_weekly_live_shape():
    from projections.backtest.espn_weekly import _fetch_espn_week, parse_espn_weekly
    payload = _fetch_espn_week(2025, 5)
    df = parse_espn_weekly(payload, season=2025, week=5, ruleset=Ruleset.espn_half())
    assert df["projected_points"].notna().sum() > 100  # wk5 had 621 projected players
```

(If `--run-network` isn't a registered pytest option in this repo, mirror the existing `--run-network` test from `tests/test_ingest/test_api_drift.py`.)

- [ ] **Step 6: Commit**

```bash
git add src/projections/backtest/espn_weekly.py tests/test_backtest/test_espn_weekly.py
git commit -m "feat(backtest): refresh ESPN weekly projections (crosswalk + store + network smoke)"
```

---

## Phase 2 — Draft basis

### Task 5: `draft_basis.py` — 2025 Sleeper-ADP half-PPR fixed-VORP table

**Files:**
- Create: `src/projections/backtest/draft_basis.py`
- Test: `tests/test_backtest/test_draft_basis.py`

The live consensus blend averages ESPN+Sleeper ADP; ESPN ADP is dead for 2025 (sentinel 170). This builds a **backtest-specific** season-projection frame (ESPN half-PPR projected season points) with **Sleeper ADP only** as `consensus_adp`, then `generate_vorp_table` (carrying the QB-replacement fix). Input is the already-ingested 2025 external snapshot (`ExternalProjectionSchema`) — read its columns first and adapt the field names below to match.

- [ ] **Step 1: Confirm the external snapshot shape**

Run: `OMP_NUM_THREADS=1 python -c "from projections.store import read_latest_partition; from pathlib import Path; df=read_latest_partition(Path('data/raw'),'external_projections',season=2025); print(list(df.columns)); print(df[['gsis_id','source']].head())"`
(If 2025 isn't ingested yet, run `python -m projections.ingest.external_projections --season 2025` first.) Record the exact ESPN-projection stat columns and the Sleeper-ADP column/source so the builder reads real names.

- [ ] **Step 2: Write the failing test** (pure builder over a synthetic external frame)

```python
# tests/test_backtest/test_draft_basis.py
import pandas as pd
from projections.schemas import Ruleset, VorpTableSchema, _PYARROW_STR
from projections.draft.league_config import LeagueConfig
from projections.backtest.draft_basis import season_projection_from_external


def _ext():
    # minimal ESPN-stat-line + sleeper_adp external rows (adapt column names to the real schema)
    return pd.DataFrame({
        "gsis_id": pd.array(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
        "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
        "espn_receptions": [40.0, 90.0], "espn_receiving_yards": [400.0, 1200.0],
        "espn_receiving_tds": [3.0, 8.0], "espn_rushing_yards": [1100.0, 0.0],
        "espn_rushing_tds": [9.0, 0.0], "espn_passing_yards": [0.0, 0.0],
        "espn_passing_tds": [0.0, 0.0], "espn_interceptions": [0.0, 0.0],
        "espn_fumbles_lost": [1.0, 1.0],
        "sleeper_adp": [3.0, 1.0],
    })


def test_season_projection_is_half_ppr_with_sleeper_adp():
    proj = season_projection_from_external(_ext(), ruleset=Ruleset.espn_half(), season=2025)
    # WR: 90*0.5 + 1200*0.1 + 8*6 = 45 + 120 + 48 = 213.0  (half-PPR)
    wr = proj[proj["gsis_id"] == "00-0000002"].iloc[0]
    assert abs(float(wr["season_mean"]) - 213.0) < 1e-6
    assert "consensus_adp" in proj.columns
    assert float(wr["consensus_adp"]) == 1.0
```

- [ ] **Step 3: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_draft_basis.py -q`
Expected: FAIL — function undefined.

- [ ] **Step 4: Implement**

```python
# src/projections/backtest/draft_basis.py
"""Build the 2025 backtest draft basis: ESPN half-PPR season projection + Sleeper-only ADP
-> fixed-VORP table. ESPN ADP is unusable for past seasons (sentinel 170), so consensus_adp
comes from Sleeper alone."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.schemas import _PYARROW_STR, ProjectionSeasonSchema, Ruleset, VorpTableSchema
from projections.scoring.score import expected_points

# Map external ESPN-projection columns -> StatLine field names. Adapt LHS to the real
# ExternalProjectionSchema column names confirmed in Step 1.
_ESPN_TO_STATLINE = {
    "espn_passing_yards": "passing_yards", "espn_passing_tds": "passing_tds",
    "espn_interceptions": "interceptions", "espn_rushing_yards": "rushing_yards",
    "espn_rushing_tds": "rushing_tds", "espn_receptions": "receptions",
    "espn_receiving_yards": "receiving_yards", "espn_receiving_tds": "receiving_tds",
    "espn_fumbles_lost": "fumbles_lost",
}


def season_projection_from_external(external: pd.DataFrame, *, ruleset: Ruleset, season: int) -> pd.DataFrame:
    means = []
    for _, row in external.iterrows():
        line = {field: float(row[col]) for col, field in _ESPN_TO_STATLINE.items()}
        means.append(expected_points(line, ruleset))
    mean = pd.array(means, dtype="float64")
    out = pd.DataFrame({
        "gsis_id": external["gsis_id"].astype(_PYARROW_STR).to_numpy(),
        "season": season,
        "position": external["position"].astype(_PYARROW_STR).to_numpy(),
        "ruleset": ruleset.name,
        "n_weeks": 17,
        "season_mean": mean, "season_p10": mean, "season_p50": mean, "season_p90": mean,
        "model_id": f"backtest-espn:{season}",
        "generated_at": pd.Timestamp(datetime.now(UTC)).as_unit("us"),
    })
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    proj = ProjectionSeasonSchema.validate(out)
    proj["consensus_adp"] = external["sleeper_adp"].to_numpy(dtype="float64")
    return proj


def build_draft_basis(external: pd.DataFrame, *, league_config: LeagueConfig, season: int) -> pd.DataFrame:
    proj = season_projection_from_external(external, ruleset=league_config.ruleset, season=season)
    table = generate_vorp_table(proj[ProjectionSeasonSchema.to_schema().columns.keys()], league_config)
    table = table.merge(proj[["gsis_id", "consensus_adp"]], on="gsis_id", how="left")
    table["gsis_id"] = table["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(table)
```

In `build_draft_basis`, select schema columns with `list(...)` (a `KeysView` won't index a DataFrame):

```python
    schema_cols = list(ProjectionSeasonSchema.to_schema().columns.keys())
    table = generate_vorp_table(proj[schema_cols], league_config)
```

Note: `generate_vorp_table` validates `ProjectionSeasonSchema` (which has no `consensus_adp`), so pass it the schema columns only and attach `consensus_adp` after — mirroring `scripts/generate_vorp_table.py:179-183`. Confirm `ProjectionSeasonSchema` column list and the `generated_at` dtype against `consensus_source.py:59-73` (already in-repo).

- [ ] **Step 5: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_draft_basis.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/projections/backtest/draft_basis.py tests/test_backtest/test_draft_basis.py
git commit -m "feat(backtest): 2025 Sleeper-ADP half-PPR fixed-VORP draft basis"
```

---

## Phase 3 — Lineup + schedule

### Task 6: `lineup.py` — fill by projection, score by actual

**Files:**
- Create: `src/projections/backtest/lineup.py`
- Test: `tests/test_backtest/test_lineup.py`

Structural template: `roster_score.optimal_lineup_points` (restrictive-slot-first greedy over laminar eligibility). Difference: rank/assign by a **projection** value, then sum the assigned players' **actual** value. Edge cases (spec §5.4): no projection ⇒ unstartable; projected-but-no-actual ⇒ contributes 0; unfilled slot ⇒ 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest/test_lineup.py
from projections.schemas import Position, RosterSlot
from projections.backtest.lineup import weekly_lineup_points

SLOTS = {RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 2}


def _player(pos, proj, actual):
    return {"position": pos.value, "projected": proj, "actual": actual}


def test_starts_highest_projection_scores_actual():
    roster = [
        _player(Position.RB, proj=20.0, actual=5.0),   # started (RB slot) -> actual 5
        _player(Position.RB, proj=10.0, actual=30.0),  # FLEX -> actual 30
        _player(Position.RB, proj=1.0, actual=99.0),   # benched (proj too low) -> 0
        _player(Position.QB, proj=15.0, actual=12.0),  # QB slot -> 12
        _player(Position.WR, proj=8.0, actual=8.0),    # WR slot -> 8
    ]
    # lineup by projection: QB(12) + RB#1[20]->5 + WR[8] + FLEX=RB#2[10]->30 = 55
    assert weekly_lineup_points(roster, SLOTS) == 55.0


def test_player_with_no_projection_is_unstartable():
    roster = [_player(Position.QB, proj=None, actual=40.0)]  # projected None -> can't start
    assert weekly_lineup_points(roster, SLOTS) == 0.0


def test_started_but_no_actual_scores_zero():
    roster = [_player(Position.QB, proj=20.0, actual=None)]  # started, didn't play
    assert weekly_lineup_points(roster, SLOTS) == 0.0


def test_unfilled_slot_scores_zero():
    roster = [_player(Position.QB, proj=20.0, actual=18.0)]  # only QB fillable
    assert weekly_lineup_points(roster, SLOTS) == 18.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_lineup.py -q`
Expected: FAIL — module undefined.

- [ ] **Step 3: Implement**

```python
# src/projections/backtest/lineup.py
"""Set a weekly lineup by PROJECTION (the manager's decision), score it by ACTUAL points.

Mirrors roster_score.optimal_lineup_points' restrictive-slot-first greedy, but assigns by
`projected` and sums `actual`. Players with a null projection are unstartable (bye/inactive);
a started player with a null actual contributes 0; unfilled slots contribute 0.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from projections.draft.roster_eligibility import FLEX_ELIGIBLE, POSITION_SLOTS, SUPER_FLEX_ELIGIBLE
from projections.schemas import Position, RosterSlot

_FLEX_SLOTS = ((RosterSlot.FLEX, FLEX_ELIGIBLE), (RosterSlot.SUPER_FLEX, SUPER_FLEX_ELIGIBLE))


def weekly_lineup_points(roster: Sequence[Mapping[str, Any]], roster_slots: Mapping[RosterSlot, int]) -> float:
    # Startable = has a non-null projection. Sort each position best-projection-first.
    startable = [p for p in roster if p.get("projected") is not None]
    by_pos: dict[Position, list[Mapping[str, Any]]] = {pos: [] for pos in Position}
    for p in startable:
        by_pos[Position(p["position"])].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: float(p["projected"]), reverse=True)
    cursor: dict[Position, int] = {pos: 0 for pos in Position}

    def _actual(p: Mapping[str, Any]) -> float:
        a = p.get("actual")
        return 0.0 if a is None else float(a)

    total = 0.0
    for slot in POSITION_SLOTS:
        pos = Position(slot.value)
        for _ in range(roster_slots.get(slot, 0)):
            if cursor[pos] < len(by_pos[pos]):
                total += _actual(by_pos[pos][cursor[pos]])
                cursor[pos] += 1
    for slot, eligible in _FLEX_SLOTS:
        for _ in range(roster_slots.get(slot, 0)):
            best_pos, best_proj = None, float("-inf")
            for pos in sorted(eligible, key=lambda p: p.value):
                if cursor[pos] < len(by_pos[pos]) and float(by_pos[pos][cursor[pos]]["projected"]) > best_proj:
                    best_pos, best_proj = pos, float(by_pos[pos][cursor[pos]]["projected"])
            if best_pos is not None:
                total += _actual(by_pos[best_pos][cursor[best_pos]])
                cursor[best_pos] += 1
    return total
```

- [ ] **Step 4: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_lineup.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/lineup.py tests/test_backtest/test_lineup.py
git commit -m "feat(backtest): weekly_lineup_points — fill by projection, score by actual"
```

---

### Task 7: `schedule.py` — regular-season schedule (circle method)

**Files:**
- Create: `src/projections/backtest/schedule.py`
- Test: `tests/test_backtest/test_schedule.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest/test_schedule.py
import numpy as np
from projections.backtest.schedule import regular_season_schedule


def test_every_team_plays_once_per_week_no_self_match():
    sched = regular_season_schedule(n_teams=16, n_weeks=14, rng=np.random.default_rng(0))
    assert len(sched) == 14
    for week in sched:
        assert len(week) == 8  # 16 teams -> 8 matchups
        seats = [s for matchup in week for s in matchup]
        assert sorted(seats) == list(range(1, 17))  # each seat exactly once
        for a, b in week:
            assert a != b


def test_deterministic_given_rng():
    a = regular_season_schedule(n_teams=16, n_weeks=14, rng=np.random.default_rng(7))
    b = regular_season_schedule(n_teams=16, n_weeks=14, rng=np.random.default_rng(7))
    assert a == b
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_schedule.py -q`
Expected: FAIL — module undefined.

- [ ] **Step 3: Implement (circle method)**

```python
# src/projections/backtest/schedule.py
"""Round-robin regular-season schedule (circle method) + single-elimination playoff bracket."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Matchup = tuple[int, int]


def regular_season_schedule(*, n_teams: int, n_weeks: int, rng: np.random.Generator) -> list[list[Matchup]]:
    if n_teams % 2 != 0:
        raise ValueError("n_teams must be even")
    teams = list(rng.permutation(np.arange(1, n_teams + 1)))  # random fixed seating
    fixed, rot = teams[0], teams[1:]
    weeks: list[list[Matchup]] = []
    for _ in range(n_weeks):
        circle = [fixed, *rot]
        half = n_teams // 2
        pairs = [(circle[i], circle[n_teams - 1 - i]) for i in range(half)]
        weeks.append([(int(a), int(b)) for a, b in pairs])
        rot = [rot[-1], *rot[:-1]]  # rotate
    return weeks
```

- [ ] **Step 4: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_schedule.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/schedule.py tests/test_backtest/test_schedule.py
git commit -m "feat(backtest): round-robin regular-season schedule (circle method)"
```

---

### Task 8: `schedule.py` — playoff bracket

**Files:**
- Modify: `src/projections/backtest/schedule.py` (add `playoff_champion`)
- Modify: `tests/test_backtest/test_schedule.py`

Single-elimination over the top-6 seeds, top-2 byes: week 15 = (seed3 v seed6, seed4 v seed5); week 16 = (seed1 v lowest-remaining, seed2 v other); week 17 = final. Each matchup decided by that week's lineup points (passed in as a `week -> {seat: points}` callable/table).

- [ ] **Step 1: Write the failing test**

```python
def test_top_seed_wins_when_always_highest():
    from projections.backtest.schedule import playoff_champion
    seeds = [3, 1, 4, 5, 9, 2]  # arbitrary seat ids in seed order 1..6
    # seat 1 is seed #2 here; make seat 3 (seed #1) always score most
    points = {15: {s: 10.0 for s in seeds}, 16: {s: 10.0 for s in seeds}, 17: {s: 10.0 for s in seeds}}
    for wk in (15, 16, 17):
        points[wk][3] = 100.0  # top seed always wins
    champ = playoff_champion(seeds, points, playoff_weeks=(15, 16, 17))
    assert champ == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_schedule.py::test_top_seed_wins_when_always_highest -q`
Expected: FAIL — `playoff_champion` undefined.

- [ ] **Step 3: Implement**

```python
# add to src/projections/backtest/schedule.py
from collections.abc import Mapping, Sequence


def _winner(a: int, b: int, pts: Mapping[int, float]) -> int:
    # higher points wins; deterministic tie-break on lower seat id
    return a if (pts[a], -a) >= (pts[b], -b) else b


def playoff_champion(seeds: Sequence[int], points: Mapping[int, Mapping[int, float]],
                     *, playoff_weeks: tuple[int, int, int]) -> int:
    """seeds: seat ids in seed order (index 0 = #1 seed ...). Top-2 bye, 6-team single-elim."""
    if len(seeds) != 6:
        raise ValueError("v1 playoff bracket expects exactly 6 seeds")
    w15, w16, w17 = playoff_weeks
    # Week 15: #3 v #6, #4 v #5
    a = _winner(seeds[2], seeds[5], points[w15])
    b = _winner(seeds[3], seeds[4], points[w15])
    # Week 16: #1 v lower-seed survivor, #2 v other survivor (reseed)
    survivors = sorted([a, b], key=lambda s: seeds.index(s))  # better seed first
    semi1 = _winner(seeds[0], survivors[1], points[w16])
    semi2 = _winner(seeds[1], survivors[0], points[w16])
    return _winner(semi1, semi2, points[w17])
```

- [ ] **Step 4: Run to verify it passes**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_schedule.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/schedule.py tests/test_backtest/test_schedule.py
git commit -m "feat(backtest): 6-team single-elim playoff bracket"
```

---

## Phase 4 — League + harness + CLI

### Task 9: `draft_field.py` — constrained-bot mixed-field draft

**Files:**
- Create: `src/projections/backtest/draft_field.py`
- Test: `tests/test_backtest/test_draft_field.py`

Promote the validated scratch constrained-draft loop (`_seatsweep.py`/`_mixed_sim.py`) into a reusable function. `draft_mixed_field(seat_controllers, pool, config, *, rng, jitter)` → `{seat: [gsis_id,...]}`. Strategy seats call `strategy.recommend(DraftState, pool, config)`; bot seats use `bot_pick` over a position-min/max-eligible subset. Also a pure `seat_layout(seed)` helper for the §3 mirrored assignment.

- [ ] **Step 1: Write the failing test (seat layout + small draft)**

```python
# tests/test_backtest/test_draft_field.py
import numpy as np
import pandas as pd
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.draft.league_config import LeagueConfig
from projections.backtest.draft_field import seat_layout, draft_mixed_field


def test_seat_layout_mirrors_on_paired_seeds():
    odd = seat_layout(1)   # base
    even = seat_layout(2)  # mirrored
    assert {s for s, k in odd.items() if k == "now_or_never"} == {2, 6, 10, 14}
    assert {s for s, k in odd.items() if k == "season_value"} == {4, 8, 12, 16}
    # mirror swaps nn<->sv
    assert {s for s, k in even.items() if k == "now_or_never"} == {4, 8, 12, 16}
    assert {s for s, k in even.items() if k == "season_value"} == {2, 6, 10, 14}
    assert sum(1 for k in odd.values() if k == "bot") == 8
```

- [ ] **Step 2: Run to verify it fails**

Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_backtest/test_draft_field.py::test_seat_layout_mirrors_on_paired_seeds -q`
Expected: FAIL — undefined.

- [ ] **Step 3: Implement `seat_layout`** (and stub `draft_mixed_field` signature)

```python
# src/projections/backtest/draft_field.py
"""Mixed-field constrained-bot draft for the H2H backtest (promoted from the validated
scratch sims). Seat layout per spec §3: nn {2,6,10,14}, sv {4,8,12,16}, bots elsewhere;
paired even seeds mirror (swap nn<->sv) so seat exposure cancels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import bot_pick
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import DraftStrategy
from projections.draft.league_config import LeagueConfig
from projections.schemas import GsisId, Position, validate_gsis_id

_MINP = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
_MAXP = {"QB": 3, "RB": 6, "WR": 6, "TE": 3}


def seat_layout(seed: int) -> dict[int, str]:
    nn, sv = {2, 6, 10, 14}, {4, 8, 12, 16}
    if seed % 2 == 0:  # paired mirror
        nn, sv = sv, nn
    return {s: ("now_or_never" if s in nn else "season_value" if s in sv else "bot") for s in range(1, 17)}


def _bot_eligible(counts: dict[str, int], picks_left: int) -> set[str]:
    deficit = {p: max(0, _MINP[p] - counts.get(p, 0)) for p in _MINP}
    if picks_left <= sum(deficit.values()):
        return {p for p in _MINP if deficit[p] > 0}
    return {p for p in _MINP if counts.get(p, 0) < _MAXP[p]}


def draft_mixed_field(seat_strategies: dict[int, DraftStrategy | None], pool: pd.DataFrame,
                      config: LeagueConfig, *, rng: np.random.Generator, jitter: float) -> dict[int, list[str]]:
    """seat_strategies: seat -> DraftStrategy (None => constrained ADP bot)."""
    nt, rs = config.n_teams, config.roster_size
    pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"])}
    pos_str = pool["position"].astype(str)
    drafted: list[str] = []
    drafted_set: set[str] = set()
    rosters: dict[int, list[str]] = {s: [] for s in range(1, nt + 1)}
    counts: dict[int, dict[str, int]] = {s: {} for s in range(1, nt + 1)}
    my_roster_pos: dict[int, list[Position]] = {s: [] for s in range(1, nt + 1)}
    for pick in range(1, nt * rs + 1):
        seat = slot_for(pick, nt)
        strat = seat_strategies.get(seat)
        if strat is not None:
            state = DraftState(my_slot=seat, n_teams=nt, rounds=rs,
                               picks=tuple(GsisId(g) for g in drafted),
                               my_roster=tuple(my_roster_pos[seat]))
            gid = validate_gsis_id(str(strat.recommend(state, pool, config).iloc[0]["gsis_id"]))
            my_roster_pos[seat].append(Position(pos_by_id[gid]))
        else:
            avail = ~pool["gsis_id"].isin(drafted_set)
            elig = _bot_eligible(counts[seat], rs - len(rosters[seat]))
            sub = pool[avail & pos_str.isin(elig)]
            if sub.empty:
                sub = pool[avail]
            gid = str(bot_pick(sub, rng, adp_jitter=jitter))
            counts[seat][pos_by_id[gid]] = counts[seat].get(pos_by_id[gid], 0) + 1
        drafted.append(gid)
        drafted_set.add(gid)
        rosters[seat].append(gid)
    return rosters
```

- [ ] **Step 4: Add a small end-to-end draft test** (uses real strategies on a synthetic pool)

```python
def _synthetic_pool(n_per_pos=30):
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        for i in range(n_per_pos):
            rows.append({"gsis_id": f"00-{pos}{i:05d}", "position": pos,
                         "season_mean_fpts": 300.0 - i, "vorp": 150.0 - i,
                         "replacement_fpts": 150.0, "consensus_adp": float(i * 4 + {"QB":3,"RB":1,"WR":2,"TE":4}[pos])})
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def _config_16_half():
    from projections.schemas import RosterSlot
    return LeagueConfig(
        name="test16half", n_teams=16, budget=200, min_bid=1,
        roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.WR: 2,
                      RosterSlot.TE: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 4},
        ruleset="espn_half",
    )


def test_draft_fills_every_roster_without_dupes():
    from projections.draft.assistant.strategy import RawVorpStrategy
    cfg = _config_16_half()  # inline — no dependency on the scratch _league_16_half.json
    pool = _synthetic_pool()
    seats = {s: (RawVorpStrategy() if s in (2, 4) else None) for s in range(1, 17)}
    rosters = draft_mixed_field(seats, pool, cfg, rng=np.random.default_rng(0), jitter=8.0)
    allp = [g for r in rosters.values() for g in r]
    assert len(allp) == len(set(allp))  # no dupes
    assert all(len(r) == cfg.roster_size for r in rosters.values())
```

- [ ] **Step 5: Run to verify both pass**

Run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_backtest/test_draft_field.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/projections/backtest/draft_field.py tests/test_backtest/test_draft_field.py
git commit -m "feat(backtest): mixed-field constrained-bot draft + mirrored seat layout"
```

---

### Task 10: `league.py` — simulate one full league season

**Files:**
- Create: `src/projections/backtest/league.py`
- Test: `tests/test_backtest/test_league.py`

`simulate_league(...)` ties it together: draft → per-week lineup points (weeks 1–17) → regular-season standings (weeks 1–14, W/L, tiebreak points-for) → seed top-6 → `playoff_champion` → `LeagueResult` list (one per seat: strategy, wins, losses, points_for, made_playoffs, is_champion). Weekly points use `weekly_lineup_points` with each rostered player's `{projected, actual}` for that week (projections/actuals frames pivoted to `(gsis_id, week) -> value`).

- [ ] **Step 1: Write the failing test** (tiny hand-checkable league)

The playoff bracket (Task 8) requires exactly 6 seeds, so use a 6-team league (1 strategy seat + 5 bots) over 5 regular weeks; the playoff is the whole field. Make one seat's players always project highest and score most so it is the deterministic champion.

```python
# tests/test_backtest/test_league.py
import numpy as np
import pandas as pd
from projections.schemas import RosterSlot, _PYARROW_STR, VorpTableSchema
from projections.draft.league_config import LeagueConfig
from projections.draft.assistant.strategy import RawVorpStrategy
from projections.backtest.league import Calendar, simulate_league


def _cfg6():
    return LeagueConfig(name="t6", n_teams=6, budget=200, min_bid=1,
                        roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.BENCH: 1},
                        ruleset="espn_half")


def _pool6():
    rows = []
    for pos in ("QB", "RB"):
        for i in range(20):
            rows.append({"gsis_id": f"00-{pos}{i:05d}", "position": pos,
                         "season_mean_fpts": 300.0 - i, "vorp": 200.0 - i,
                         "replacement_fpts": 100.0, "consensus_adp": float(i * 6 + (0 if pos == "RB" else 3))})
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def test_dominant_seat_is_champion_and_top_record():
    cfg, pool = _cfg6(), _pool6()
    cal = Calendar(regular_weeks=tuple(range(1, 6)), playoff_weeks=(6, 7, 8), playoff_size=6)
    # Seat 1 runs a strategy; the rest are bots. Seat 1 will draft the top RawVorp players.
    seat_strategies = {1: RawVorpStrategy(), **{s: None for s in range(2, 7)}}
    labels = {1: "now_or_never", **{s: "bot" for s in range(2, 7)}}
    # Projection == actual == the player's season_mean_fpts every week (so seat 1's roster always wins).
    proj = {(g, wk): float(m) for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"]) for wk in range(1, 9)}
    actual = dict(proj)
    results = simulate_league(0, seat_strategies=seat_strategies, strategy_labels=labels,
                              pool=pool, config=cfg, proj_lookup=proj, actual_lookup=actual,
                              calendar=cal, jitter=8.0)
    by_seat = {r.seat: r for r in results}
    assert by_seat[1].is_champion                       # (a) dominant seat wins it all
    assert by_seat[1].wins == 5 and by_seat[1].losses == 0  # (b) record sums to regular weeks
    assert all(r.wins + r.losses == 5 for r in results)
    assert by_seat[1].points_for > 0                    # (c) points accumulate
```

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL (module undefined).

- [ ] **Step 3: Implement `simulate_league` + `LeagueResult` + `Calendar`.**

```python
# src/projections/backtest/league.py — sketch (fill in to satisfy the Step-1 test)
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np
import pandas as pd
from projections.backtest.draft_field import draft_mixed_field, seat_layout
from projections.backtest.lineup import weekly_lineup_points
from projections.backtest.schedule import playoff_champion, regular_season_schedule
from projections.draft.league_config import LeagueConfig
from projections.draft.assistant.strategy import DraftStrategy


@dataclass(frozen=True)
class Calendar:
    regular_weeks: tuple[int, ...]      # e.g. tuple(range(1, 15))
    playoff_weeks: tuple[int, int, int] # (15, 16, 17)
    playoff_size: int = 6


@dataclass(frozen=True)
class LeagueResult:
    seat: int
    strategy: str
    wins: int
    losses: int
    points_for: float
    made_playoffs: bool
    is_champion: bool


def simulate_league(seed: int, *, seat_strategies: Mapping[int, DraftStrategy | None],
                    strategy_labels: Mapping[int, str], pool: pd.DataFrame, config: LeagueConfig,
                    proj_lookup: Mapping[tuple[str, int], float],
                    actual_lookup: Mapping[tuple[str, int], float],
                    calendar: Calendar, jitter: float) -> list[LeagueResult]:
    rng = np.random.default_rng(seed)
    rosters = draft_mixed_field(dict(seat_strategies), pool, config, rng=rng, jitter=jitter)
    pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"])}

    def week_points(seat: int, wk: int) -> float:
        roster = [{"position": pos_by_id[g],
                   "projected": proj_lookup.get((g, wk)),
                   "actual": actual_lookup.get((g, wk))} for g in rosters[seat]]
        return weekly_lineup_points(roster, config.roster_slots)

    all_weeks = set(calendar.regular_weeks) | set(calendar.playoff_weeks)
    pts = {wk: {s: week_points(s, wk) for s in rosters} for wk in all_weeks}

    sched = regular_season_schedule(n_teams=config.n_teams, n_weeks=len(calendar.regular_weeks), rng=rng)
    wins = {s: 0 for s in rosters}; losses = {s: 0 for s in rosters}; pf = {s: 0.0 for s in rosters}
    for wk, matchups in zip(calendar.regular_weeks, sched):
        for a, b in matchups:
            pf[a] += pts[wk][a]; pf[b] += pts[wk][b]
            if (pts[wk][a], -a) >= (pts[wk][b], -b): wins[a] += 1; losses[b] += 1
            else: wins[b] += 1; losses[a] += 1
    standings = sorted(rosters, key=lambda s: (wins[s], pf[s]), reverse=True)
    seeds = standings[:calendar.playoff_size]
    champ = playoff_champion(seeds, {wk: pts[wk] for wk in calendar.playoff_weeks},
                             playoff_weeks=calendar.playoff_weeks)
    return [LeagueResult(seat=s, strategy=strategy_labels[s], wins=wins[s], losses=losses[s],
                         points_for=pf[s], made_playoffs=s in seeds, is_champion=s == champ)
            for s in rosters]
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/projections/backtest/league.py tests/test_backtest/test_league.py
git commit -m "feat(backtest): simulate_league — draft -> weekly -> standings -> playoffs"
```

---

### Task 11: `harness.py` — run many seeds + aggregate per strategy

**Files:**
- Create: `src/projections/backtest/harness.py`
- Test: `tests/test_backtest/test_harness.py`

`run_backtest(...)`: for `seed in range(n_seeds)`, build `seat_strategies` from `seat_layout(seed)` (nn → `NowOrNeverStrategy`, sv → `SeasonValueStrategy(..., n_sims=strategy_n_sims)`, bot → None), call `simulate_league`, collect `LeagueResult`s. Aggregate per strategy label across all seats×seeds: championship rate (mean `is_champion`), playoff rate, regular-season win% (`wins/(wins+losses)`), mean points-for — each with `_bootstrap_mean` CI. Reuse `tournament.Interval`/`_bootstrap_mean`.

- [ ] **Step 1: Write the failing test** (small `n_seeds`, assert structure + the correct seat-weighted identity)

```python
# tests/test_backtest/test_harness.py
import numpy as np
import pandas as pd
from projections.schemas import RosterSlot, _PYARROW_STR, VorpTableSchema
from projections.draft.league_config import LeagueConfig
from projections.backtest.league import Calendar
from projections.backtest.harness import run_backtest
# reuse the 6-team helpers' shape but at 16 teams to exercise the real seat layout
from tests.test_backtest.test_draft_field import _synthetic_pool
from tests.test_backtest.test_availability_stub import stub_availability  # see note


def _cfg16():
    return LeagueConfig(name="t16", n_teams=16, budget=200, min_bid=1,
                        roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 2, RosterSlot.WR: 2,
                                      RosterSlot.TE: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 4},
                        ruleset="espn_half")


def test_rates_bounded_and_seat_weighted_champions_sum_to_one():
    cfg, pool = _cfg16(), _synthetic_pool()
    cal = Calendar(regular_weeks=tuple(range(1, 6)), playoff_weeks=(6, 7, 8), playoff_size=6)
    proj = {(g, wk): float(m) for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"]) for wk in range(1, 9)}
    actual = dict(proj)
    res = run_backtest(n_seeds=4, pool=pool, config=cfg, availability=stub_availability(pool),
                       proj_lookup=proj, actual_lookup=actual, calendar=cal,
                       jitter=8.0, strategy_n_sims=5, base_seed=0)
    assert set(res.by_strategy) == {"now_or_never", "season_value", "bot"}
    for m in res.by_strategy.values():
        for iv in (m.championship, m.playoff, m.win_pct):
            assert 0.0 <= iv.lo_95 <= iv.point <= iv.hi_95 <= 1.0
    # Exactly one champion per league. nn/sv occupy 4 seats each, bots 8 -> seat-weighted identity:
    r = res.by_strategy
    weighted = 4 * r["now_or_never"].championship.point + 4 * r["season_value"].championship.point \
        + 8 * r["bot"].championship.point
    assert abs(weighted - 1.0) < 1e-9
```

Note: `stub_availability(pool)` returns a minimal `PlayerAvailability` (every player `p≈0.95`, no byes) so `SeasonValueStrategy` runs without store I/O; define it once in `tests/test_backtest/test_availability_stub.py` by constructing `PlayerAvailability` per its real constructor (read `availability.py` for the exact fields). The win_pct CI upper bound may equal 1.0 only if a strategy wins every game in every sampled bootstrap — keep `<= 1.0`.

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL (module undefined).

- [ ] **Step 3: Implement**

```python
# src/projections/backtest/harness.py — sketch
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
import pandas as pd
from projections.backtest.draft_field import seat_layout
from projections.backtest.league import Calendar, LeagueResult, simulate_league
from projections.draft.assistant.strategy import NowOrNeverStrategy, SeasonValueStrategy
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.tournament import Interval, _bootstrap_mean
from projections.draft.league_config import LeagueConfig


@dataclass(frozen=True)
class StrategyMetrics:
    championship: Interval
    playoff: Interval
    win_pct: Interval
    points_for: Interval


@dataclass(frozen=True)
class BacktestResult:
    by_strategy: dict[str, StrategyMetrics]
    n_seeds: int


def run_backtest(*, n_seeds: int, pool: pd.DataFrame, config: LeagueConfig,
                 availability: PlayerAvailability, proj_lookup, actual_lookup,
                 calendar: Calendar, jitter: float = 8.0, strategy_n_sims: int = 200,
                 base_seed: int = 0) -> BacktestResult:
    sigma = default_sigma(config.n_teams)
    nn = NowOrNeverStrategy(LogisticSurvival(sigma=sigma))
    sv = SeasonValueStrategy(availability, n_sims=strategy_n_sims, base_seed=base_seed)
    label_to_strategy = {"now_or_never": nn, "season_value": sv, "bot": None}
    results: list[LeagueResult] = []
    for s in range(n_seeds):
        layout = seat_layout(s)
        seat_strategies = {seat: label_to_strategy[label] for seat, label in layout.items()}
        results += simulate_league(base_seed + s, seat_strategies=seat_strategies, strategy_labels=layout,
                                   pool=pool, config=config, proj_lookup=proj_lookup,
                                   actual_lookup=actual_lookup, calendar=calendar, jitter=jitter)
    out: dict[str, StrategyMetrics] = {}
    for label in ("now_or_never", "season_value", "bot"):
        rs = [r for r in results if r.strategy == label]
        champ = np.array([1.0 if r.is_champion else 0.0 for r in rs])
        playoff = np.array([1.0 if r.made_playoffs else 0.0 for r in rs])
        winp = np.array([r.wins / (r.wins + r.losses) for r in rs])
        pf = np.array([r.points_for for r in rs])
        out[label] = StrategyMetrics(
            championship=_bootstrap_mean(champ, seed=base_seed),
            playoff=_bootstrap_mean(playoff, seed=base_seed),
            win_pct=_bootstrap_mean(winp, seed=base_seed),
            points_for=_bootstrap_mean(pf, seed=base_seed))
    return BacktestResult(by_strategy=out, n_seeds=n_seeds)
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/projections/backtest/harness.py tests/test_backtest/test_harness.py
git commit -m "feat(backtest): run_backtest — mirrored seeds, per-strategy rates + CIs"
```

---

### Task 12: `scripts/h2h_backtest.py` — CLI

**Files:**
- Create: `scripts/h2h_backtest.py`
- Test: `tests/test_backtest/test_cli.py` (argument parsing + a tiny end-to-end smoke on synthetic tables)

CLI wires the real 2025 tables: load the draft basis (`build_draft_basis` over the ingested 2025 external snapshot), `read_partition` the ESPN weekly projections + build weekly actuals from `weekly_stats 2025`, build availability via the existing `load_store_availability`, pivot projections/actuals to `(gsis_id, week)->float` lookups, then `run_backtest`. Flags: `--n-seeds` (default 200), `--strategy-n-sims` (default 200), `--jitter` (default 8.0), `--data-root`, `--season` (default 2025). Print a per-strategy table (championship%, playoff%, win%, PF + CIs).

- [ ] **Step 1: Write the failing test** (arg defaults + the formatter)

```python
# tests/test_backtest/test_cli.py
from projections.backtest.cli import _parse_args, format_result
from projections.backtest.harness import BacktestResult, StrategyMetrics
from projections.draft.assistant.tournament import Interval


def test_arg_defaults():
    args = _parse_args(["--season", "2025"])
    assert args.n_seeds == 200 and args.strategy_n_sims == 200 and args.jitter == 8.0


def test_format_lists_every_strategy():
    iv = Interval(point=0.1, lo_95=0.05, hi_95=0.15)
    m = StrategyMetrics(championship=iv, playoff=iv, win_pct=iv, points_for=Interval(1400, 1380, 1420))
    res = BacktestResult(by_strategy={"now_or_never": m, "season_value": m, "bot": m}, n_seeds=200)
    text = format_result(res)
    assert "now_or_never" in text and "season_value" in text and "champ" in text.lower()
```

Put the CLI core in `src/projections/backtest/cli.py` (`_parse_args`, `format_result`, `run(argv)`) and make `scripts/h2h_backtest.py` a 3-line wrapper that calls `cli.run` — exactly the `tournament_cli.py` / `scripts/draft_tournament.py` split.

- [ ] **Step 2: Implement** `cli.py` mirroring `src/projections/draft/assistant/tournament_cli.py` structure: `_parse_args` (flags `--season` default 2025, `--n-seeds` 200, `--strategy-n-sims` 200, `--jitter` 8.0, `--data-root` `data`); `run(argv)` = load draft basis via `build_draft_basis` over the ingested 2025 external snapshot, `read_partition` the espn_weekly projections + `build_weekly_actuals` from `weekly_stats`, pivot both to `{(gsis_id, week): float}` dicts, build availability via `load_store_availability(pool, season, data_root)`, call `run_backtest`, `print(format_result(res))`, return 0. `format_result` renders a per-strategy table (championship%, playoff%, win%, PF with CIs), mirroring `tournament_cli.format_compare`.
- [ ] **Step 3: Run; verify PASS.**

Run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_backtest/test_cli.py -q`
- [ ] **Step 4: Full-suite gate**

Run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_backtest -q && python -m mypy src/projections/backtest tests/test_backtest && python -m ruff check src/projections/backtest tests/test_backtest scripts/h2h_backtest.py && python -m ruff format --check src/projections/backtest`
Expected: all green.

- [ ] **Step 5: Commit.**

```bash
git add scripts/h2h_backtest.py tests/test_backtest/test_cli.py
git commit -m "feat(backtest): h2h_backtest CLI over the harness"
```

---

## Phase 5 — Run + report (execution-time, not new code)

### Task 13: Ingest 2025 data, run the backtest, record Test F1

**Files:**
- Modify: `reports/draft_strategy_tests.md` (add Test F1)

- [ ] **Step 1: Ingest the 2025 draft basis + weekly projections**

First confirm the real `id_map` load path: `grep -rn "id_map" scripts/draft_assistant.py src/projections/draft/assistant/state.py` and use exactly that (the cheat-sheet / assistant CLIs already load it — likely `pd.read_parquet("data/id_map.parquet")` then `IdMapSchema.validate`). Then:

```bash
OMP_NUM_THREADS=1 python -m projections.ingest.external_projections --season 2025
OMP_NUM_THREADS=1 python - <<'PY'
from pathlib import Path
import pandas as pd
from projections.backtest.espn_weekly import refresh_espn_weekly_projections
from projections.schemas import IdMapSchema, Ruleset
id_map = IdMapSchema.validate(pd.read_parquet("data/id_map.parquet"))  # match the real path confirmed above
refresh_espn_weekly_projections(season=2025, ruleset=Ruleset.espn_half(), id_map=id_map, data_root=Path("data"))
PY
```

- [ ] **Step 2: Run the backtest (default 200 seeds × 200 sims)**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/h2h_backtest.py --season 2025 --n-seeds 200 --strategy-n-sims 200 --jitter 8 --data-root data
```

- [ ] **Step 3: Record results as Test F1 in `reports/draft_strategy_tests.md`**

Append a "Test F1 — H2H backtest (2025, real outcomes)" entry: setup (mixed field, draft-and-hold, weeks 1–17, Sleeper ADP, half-PPR), the per-strategy championship%/win%/playoff%/PF table with CIs, and which strategy it favors. **No verdict on the overall investigation** — that's the end-of-process judgment across all tests. Note the single-season caveat.

- [ ] **Step 4: Commit.**

```bash
git add reports/draft_strategy_tests.md
git commit -m "docs(backtest): Test F1 — 2025 H2H backtest results"
```

---

## Self-Review

**Spec coverage:** §4 data layer → Tasks 2–5; §5.1/5.2 → Tasks 2–4; §5.3 draft basis → Task 5; §5.4 lineup (all edge cases) → Task 6; §5.5 schedule+playoffs → Tasks 7–8; §5.6 league → Task 10; §5.7 harness (mirrored seats, n_sims, fast-path, CLI flags) → Tasks 11–12; §3 seat layout → Task 9; §6 determinism (fixed tables, seed = draft+schedule) → Tasks 10–11; §7 schemas → Task 1; §8 testing (fixture-tested ESPN, network smoke, tiny-league hand-check) → throughout; §9 phasing → Phases 0–5; week-18 exclusion → Tasks 1 (schema `le=17`), 2 (`<=17` filter), 3/4 (`weeks` range). All covered.

**Placeholder scan:** Tasks 10–12 use a labeled "sketch" with a written contract (signature + behavior + assertions) rather than fully-inlined final code, because their exact shape depends on choices fixed in Step 1's test; each names every type/function and the assertions to satisfy. Tasks 1–9 carry complete code. The `_ESPN_TO_STATLINE` / `ExternalProjectionSchema` column names (Task 5) and `write_partition` arg order (Task 4) are explicitly flagged to confirm against the real in-repo definition in a Step-1 read — not left vague.

**Type consistency:** `seat_layout` returns `dict[int, str]` (labels `now_or_never`/`season_value`/`bot`) consumed identically in Tasks 9–11; `LeagueResult` fields (Task 10) match `harness` aggregation (Task 11); `Calendar` (Task 10) consumed by `harness`/CLI; `weekly_lineup_points(roster, roster_slots)` signature consistent Tasks 6/10; `Interval`/`_bootstrap_mean` reused from `tournament`.

**Known residual to verify during execution (not placeholders — real-API confirmations):** `ESPN_STAT_IDS`/`ESPN_POSITIONS` export location; `ExternalProjectionSchema` ESPN-stat column names; `write_partition` signature; `id_map` load path; `ProjectionSeasonSchema` column set. Each is pinned to a named in-repo source to read at its task.
