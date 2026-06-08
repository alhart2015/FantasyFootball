# External Projection Benchmark Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark our `BaselineModel` preseason projection against ESPN's preseason projection at predicting actual 2024 fantasy outcomes, and emit a go/no-go verdict on building an external-consensus layer.

**Architecture:** Two throwaway-tolerant scripts in `scripts/` (no `src/` core changes). `pull_external_projections.py` fetches ESPN preseason stat lines + ADP and Sleeper ADP, writing intermediate parquet under `data/external_projections/2024/`. `benchmark_projections.py` joins ESPN + our model's CSV + in-house actuals on `GsisId`, scores every stat line through the existing PPR ruleset, computes RMSE/MAE/Spearman/top-20-hit-rate for a full-population and a top-20-per-position cohort, and renders `reports/external_projection_benchmark_2024.md`. Pure transforms are TDD'd with synthetic fixtures; network pulls are verified by running them.

**Tech Stack:** Python 3, pandas, pydantic (`StatLine`/`Ruleset` from `projections.scoring`/`projections.schemas`), `urllib` (stdlib, no new deps), pytest, mypy strict, ruff.

---

## Background facts the implementer needs (verified, do not re-derive)

- **Scoring:** `from projections.scoring.score import StatLine, score` and `from projections.schemas import Ruleset`. `Ruleset.espn_ppr()` = PPR (1.0/rec, 0.1/rec-yd, 0.04/pass-yd i.e. 25 yds/pt, 4/pass-TD, -2/INT, 0.1/rush-yd, 6/rush-&-rec-TD, -2/fumble-lost). `score(StatLine(...), ruleset)` returns fantasy points. `StatLine` fields used here: `passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fumbles_lost` (counts are ints, yards floats).
- **Common scoring set only.** `weekly_stats` has **no** 2-pt or return-TD columns, so to stay apples-to-apples we score every side (ESPN, ours, actual) on the nine fields above and nothing else. Drop ESPN's projected 2-pt ids; they contribute fractions of a point.
- **ESPN stat-id → field map** (decoded empirically against 2024 Josh Allen / Saquon / Chase; reconstructing through PPR matches ESPN `appliedTotal` within ~1 pt):
  `{"3":"passing_yards","4":"passing_tds","20":"interceptions","24":"rushing_yards","25":"rushing_tds","53":"receptions","42":"receiving_yards","43":"receiving_tds","72":"fumbles_lost"}`.
- **ESPN endpoint (no auth):** `GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2024/segments/0/leaguedefaults/3?view=kona_player_info` with header `X-Fantasy-Filter: {"players":{"limit":800,"sortPercOwned":{"sortPriority":1,"sortAsc":false}}}`. One `limit=800` pull covers all draft-relevant players; no pagination needed. Per player: `player.id` (ESPN id), `player.fullName`, `player.defaultPositionId` (1=QB,2=RB,3=WR,4=TE), `player.stats[]` entries (each with `seasonId`, `statSourceId` 1=proj/0=actual, `statSplitTypeId` 0=season, `stats` raw dict, `appliedTotal`), `player.ownership.averageDraftPosition` (ADP), `player.draftRanksByRankType.PPR.rank` (positional draft rank).
- **Sleeper endpoint (no auth, rank reference only):** `GET https://api.sleeper.com/projections/nfl/2024?season_type=regular` → list of `{player_id, stats:{adp_ppr, ...}}`. No season stat line; ADP only.
- **Our model output:** `scripts/project_season.py --season 2024 --out reports/season_projection_2024.csv` writes a CSV with columns `rank, gsis_id, position, season_total_mean, n_weeks, full_name, team`. `season_total_mean` is **already PPR fantasy points** — use it directly as our projected points; do **not** rebuild a StatLine for our side.
- **Actuals:** `read_partition(Path("data/raw"), "weekly_stats", season=2024)` → rows with `gsis_id, position` + the nine stat columns. Sum per `gsis_id`, build `StatLine`, score.
- **ID crosswalk:** `read_partition(Path("data/raw"), "id_map")` → columns `gsis_id, espn_id, sleeper_id, pfr_id, full_name, position, team` (`espn_id`/`sleeper_id` nullable strings). Join ESPN on `espn_id`, Sleeper on `sleeper_id`.
- **Store import:** `from projections.store import read_partition`.
- **Test location:** put tests in `tests/test_scripts/` — its conftest puts `scripts/` on `sys.path`, so `import pull_external_projections` / `import benchmark_projections` work by bare name (mirror the existing `tests/test_scripts/test_tune_lightgbm.py`).
- **Spearman:** use `pred.corr(actual, method="spearman")` on aligned non-NaN pandas Series.

---

## Phase 1 — External pull (`scripts/pull_external_projections.py`)

### Task 1: ESPN stat-id decoder

**Files:**
- Create: `scripts/pull_external_projections.py`
- Test: `tests/test_scripts/test_pull_external_projections.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scripts/test_pull_external_projections.py
import pull_external_projections as pull


def test_espn_stats_to_statline_dict_maps_ids_and_rounds_counts():
    # Chase-like projected dict (subset): rec yds 1335.25, receptions 104.87,
    # rec tds 8.24, rush yds 18.16, fumbles lost 0.99.
    raw = {"42": 1335.25, "53": 104.87, "43": 8.24, "24": 18.16, "72": 0.99,
           "99": 123.0}  # 99 is an unmapped id and must be ignored
    out = pull.espn_stats_to_statline_dict(raw)
    assert out["receiving_yards"] == 1335.25      # float kept
    assert out["receptions"] == 105               # count rounded to int
    assert out["receiving_tds"] == 8              # rounds 8.24 -> 8
    assert out["rushing_yards"] == 18.16
    assert out["fumbles_lost"] == 1
    assert out["passing_yards"] == 0.0            # missing id -> 0
    assert "99" not in out.values() and 123.0 not in out.values()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_pull_external_projections.py::test_espn_stats_to_statline_dict_maps_ids_and_rounds_counts -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pull_external_projections'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/pull_external_projections.py
"""Spike: pull preseason projections + ADP from free public sources for one season.

Writes intermediate parquet under data/external_projections/{season}/:
  - espn.parquet       : per-player preseason projected stat line + ADP/rank + ESPN actual total
  - sleeper_adp.parquet : per-player preseason ADP (rank reference only)

ESPN's season projection (statSourceId=1, statSplitTypeId=0) is the genuine
preseason forecast (verified against 2024: rookie/breakout misses, not
contaminated end-of-season values). Sleeper exposes only ADP at the season
level, so it is a rank reference, not a stat-line source.

Network-dependent; the pure parse helpers are unit-tested with synthetic payloads.

Usage:
    python scripts/pull_external_projections.py --season 2024
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

# ESPN numeric stat-id -> StatLine field (common scoring set only). Decoded
# empirically against known 2024 players; reconstructing through ESPN PPR
# matches appliedTotal within ~1 pt.
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
_COUNT_FIELDS = frozenset(
    {"passing_tds", "interceptions", "rushing_tds", "receptions", "receiving_tds", "fumbles_lost"}
)
_ALL_FIELDS = (
    "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds", "fumbles_lost",
)


def espn_stats_to_statline_dict(stats: dict[str, float]) -> dict[str, float]:
    """Map ESPN's numeric stat dict to our StatLine field names. Missing ids -> 0.
    Count fields are rounded to the nearest integer; yards stay float."""
    out: dict[str, float] = {f: 0.0 for f in _ALL_FIELDS}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in stats:
            val = float(stats[sid])
            out[field] = float(round(val)) if field in _COUNT_FIELDS else val
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_pull_external_projections.py::test_espn_stats_to_statline_dict_maps_ids_and_rounds_counts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_external_projections.py tests/test_scripts/test_pull_external_projections.py
git commit -m "feat(spike): ESPN stat-id decoder for external projection pull"
```

---

### Task 2: ESPN payload parser

**Files:**
- Modify: `scripts/pull_external_projections.py`
- Test: `tests/test_scripts/test_pull_external_projections.py`

- [ ] **Step 1: Write the failing test**

```python
def _fake_espn_payload():
    # One QB with a 2024 projected season entry, a 2024 actual entry, ADP + rank,
    # plus a defense (defaultPositionId 16) that must be filtered out.
    return {"players": [
        {"player": {
            "id": 3918298, "fullName": "Josh Allen", "defaultPositionId": 1,
            "ownership": {"averageDraftPosition": 25.3},
            "draftRanksByRankType": {"PPR": {"rank": 3}},
            "stats": [
                {"seasonId": 2024, "statSourceId": 1, "statSplitTypeId": 0,
                 "appliedTotal": 313.6, "stats": {"3": 3752.62, "4": 23.02, "20": 12.44,
                                                  "24": 495.76, "25": 8.57, "72": 4.08}},
                {"seasonId": 2024, "statSourceId": 0, "statSplitTypeId": 0,
                 "appliedTotal": 379.04, "stats": {}},
                {"seasonId": 2023, "statSourceId": 1, "statSplitTypeId": 0,
                 "appliedTotal": 999.0, "stats": {}},  # wrong season, ignore
            ],
        }},
        {"player": {"id": 9999, "fullName": "Some DST", "defaultPositionId": 16, "stats": []}},
    ]}


def test_parse_espn_players_extracts_proj_actual_adp_rank_and_filters_positions():
    df = pull.parse_espn_players(_fake_espn_payload(), season=2024)
    assert list(df["espn_id"]) == ["3918298"]          # DST filtered out
    row = df.iloc[0]
    assert row["position"] == "QB"
    assert row["full_name"] == "Josh Allen"
    assert row["espn_adp"] == 25.3
    assert row["espn_pos_rank"] == 3
    assert row["espn_actual_applied_total"] == 379.04
    assert row["passing_yards"] == 3752.62             # from the 2024 PROJ entry
    assert row["passing_tds"] == 23                     # rounded
    assert row["interceptions"] == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_pull_external_projections.py::test_parse_espn_players_extracts_proj_actual_adp_rank_and_filters_positions -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'parse_espn_players'`.

- [ ] **Step 3: Write minimal implementation** (append to `scripts/pull_external_projections.py`)

```python
_ESPN_POSITIONS: dict[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}


def parse_espn_players(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Tidy one ESPN kona_player_info payload into one row per QB/RB/WR/TE with a
    preseason projected season stat line. Players without a season-proj entry, or
    not in QB/RB/WR/TE, are dropped."""
    rows: list[dict[str, object]] = []
    for entry in payload.get("players", []):
        pl = entry.get("player", {})
        position = _ESPN_POSITIONS.get(pl.get("defaultPositionId"))
        if position is None:
            continue
        proj_stats: dict[str, float] | None = None
        actual_total: float | None = None
        for s in pl.get("stats", []):
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
                continue
            if s.get("statSourceId") == 1:
                proj_stats = s.get("stats", {})
            elif s.get("statSourceId") == 0:
                actual_total = s.get("appliedTotal")
        if proj_stats is None:
            continue
        ownership = pl.get("ownership") or {}
        ppr_rank = ((pl.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank")
        row: dict[str, object] = {
            "espn_id": str(pl.get("id")),
            "full_name": pl.get("fullName"),
            "position": position,
            "espn_adp": ownership.get("averageDraftPosition"),
            "espn_pos_rank": ppr_rank,
            "espn_actual_applied_total": actual_total,
        }
        row.update(espn_stats_to_statline_dict(proj_stats))
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_pull_external_projections.py::test_parse_espn_players_extracts_proj_actual_adp_rank_and_filters_positions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_external_projections.py tests/test_scripts/test_pull_external_projections.py
git commit -m "feat(spike): parse ESPN payload into tidy preseason projection rows"
```

---

### Task 3: Sleeper ADP parser + fetchers + CLI main

**Files:**
- Modify: `scripts/pull_external_projections.py`
- Test: `tests/test_scripts/test_pull_external_projections.py`

- [ ] **Step 1: Write the failing test**

```python
def test_parse_sleeper_adp_keeps_id_and_ppr_adp():
    payload = [
        {"player_id": "4046", "stats": {"adp_ppr": 1.2, "gp": 17.0}},
        {"player_id": "6794", "stats": {"adp_ppr": 14.5}},
        {"player_id": None, "stats": {"adp_ppr": 9.0}},  # no id -> dropped
    ]
    df = pull.parse_sleeper_adp(payload)
    assert list(df["sleeper_id"]) == ["4046", "6794"]
    assert list(df["sleeper_adp"]) == [1.2, 14.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_pull_external_projections.py::test_parse_sleeper_adp_keeps_id_and_ppr_adp -v`
Expected: FAIL with `AttributeError: ... 'parse_sleeper_adp'`.

- [ ] **Step 3: Write minimal implementation** (append to `scripts/pull_external_projections.py`)

```python
_ESPN_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
    "{season}/segments/0/leaguedefaults/3?view=kona_player_info"
)
_SLEEPER_URL = "https://api.sleeper.com/projections/nfl/{season}?season_type=regular"
_UA = "Mozilla/5.0"


def parse_sleeper_adp(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep sleeper player id + PPR ADP. Rows without a player id are dropped."""
    rows: list[dict[str, object]] = []
    for item in payload:
        pid = item.get("player_id")
        if pid is None:
            continue
        stats = item.get("stats") or {}
        rows.append({"sleeper_id": str(pid), "sleeper_adp": stats.get("adp_ppr")})
    return pd.DataFrame(rows)


def fetch_espn(season: int, limit: int = 800) -> dict[str, Any]:
    flt = {"players": {"limit": limit, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    req = urllib.request.Request(
        _ESPN_URL.format(season=season),
        headers={"User-Agent": _UA, "X-Fantasy-Filter": json.dumps(flt)},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted ESPN host)
        return json.load(resp)


def fetch_sleeper_season(season: int) -> list[dict[str, Any]]:
    req = urllib.request.Request(_SLEEPER_URL.format(season=season), headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted Sleeper host)
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull preseason external projections for one season.")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--out-root", type=Path, default=Path("data/external_projections"))
    args = ap.parse_args()

    out_dir = args.out_root / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)

    espn = parse_espn_players(fetch_espn(args.season), args.season)
    espn.to_parquet(out_dir / "espn.parquet", index=False)
    print(f"ESPN: {len(espn)} players -> {out_dir / 'espn.parquet'}", flush=True)

    sleeper = parse_sleeper_adp(fetch_sleeper_season(args.season))
    sleeper.to_parquet(out_dir / "sleeper_adp.parquet", index=False)
    print(f"Sleeper ADP: {len(sleeper)} players -> {out_dir / 'sleeper_adp.parquet'}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes; lint/type the module**

Run: `pytest tests/test_scripts/test_pull_external_projections.py -v`
Expected: all 3 tests PASS.
Run: `ruff check scripts/pull_external_projections.py tests/test_scripts/test_pull_external_projections.py && ruff format --check scripts/pull_external_projections.py tests/test_scripts/test_pull_external_projections.py && mypy src tests`
Expected: no violations. (If `mypy` flags the untyped `payload: dict`/`list[dict]` returns from `json.load`, they are acceptable here — `json.load` returns `Any`; keep the annotations as written.)

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_external_projections.py tests/test_scripts/test_pull_external_projections.py
git commit -m "feat(spike): Sleeper ADP parser + ESPN/Sleeper fetchers + pull CLI"
```

---

### Task 4: Run the real ESPN + Sleeper pull (network)

**Files:**
- Creates data: `data/external_projections/2024/espn.parquet`, `data/external_projections/2024/sleeper_adp.parquet`

- [ ] **Step 1: Run the pull**

Run: `python scripts/pull_external_projections.py --season 2024`
Expected: prints `ESPN: <N> players ...` with N in the hundreds, and `Sleeper ADP: <M> players ...`.

- [ ] **Step 2: Sanity-check the ESPN parquet against a known player**

Run:
```bash
python -c "import pandas as pd; df=pd.read_parquet('data/external_projections/2024/espn.parquet'); \
r=df[df.full_name=='Ja\'Marr Chase'].iloc[0]; \
print(r[['position','espn_pos_rank','receptions','receiving_yards','receiving_tds','espn_actual_applied_total']])"
```
Expected: WR, receptions ~105, receiving_yards ~1335, receiving_tds ~8, actual ~403 (verifies the decode end-to-end on real data). If the values are wildly off, the ESPN stat-id map drifted — stop and re-decode before continuing.

- [ ] **Step 3: Commit the pulled data**

```bash
git add data/external_projections/2024/espn.parquet data/external_projections/2024/sleeper_adp.parquet
git commit -m "data(spike): 2024 ESPN preseason projections + Sleeper ADP pull"
```

---

## Phase 2 — Benchmark compute + report (`scripts/benchmark_projections.py`)

### Task 5: Score actuals + load our model's points

**Files:**
- Create: `scripts/benchmark_projections.py`
- Test: `tests/test_scripts/test_benchmark_projections.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scripts/test_benchmark_projections.py
import pandas as pd

import benchmark_projections as bench
from projections.schemas import Ruleset


def test_actual_season_points_sums_weeks_and_scores_ppr():
    # One WR, two weeks. Wk1: 5 rec, 70 yds, 1 TD. Wk2: 3 rec, 30 yds, 0 TD.
    # PPR points = receptions*1 + yards*0.1 + tds*6 = (8) + (100*0.1=10) + (1*6=6) = 24.0
    ws = pd.DataFrame({
        "gsis_id": ["00-0000001", "00-0000001"],
        "position": ["WR", "WR"],
        "passing_yards": [0.0, 0.0], "passing_tds": [0, 0], "interceptions": [0, 0],
        "rushing_yards": [0.0, 0.0], "rushing_tds": [0, 0],
        "receptions": [5, 3], "receiving_yards": [70.0, 30.0], "receiving_tds": [1, 0],
        "fumbles_lost": [0, 0],
    })
    out = bench.actual_season_points(ws, Ruleset.espn_ppr())
    assert list(out["gsis_id"]) == ["00-0000001"]
    assert out.iloc[0]["position"] == "WR"
    assert out.iloc[0]["actual_pts"] == 24.0


def test_our_season_points_reads_csv_mean_as_points():
    df = pd.DataFrame({
        "gsis_id": ["00-0000001"], "position": ["WR"],
        "season_total_mean": [180.5], "full_name": ["A B"],
    })
    out = bench.our_season_points(df)
    assert out.iloc[0]["our_pts"] == 180.5
    assert out.iloc[0]["gsis_id"] == "00-0000001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_benchmark_projections.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark_projections'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/benchmark_projections.py
"""Spike: benchmark our BaselineModel preseason projection vs ESPN's preseason
projection at predicting actual 2024 fantasy outcomes. Emits a verdict report.

Inputs:
  - data/external_projections/{season}/espn.parquet   (from pull_external_projections.py)
  - data/external_projections/{season}/sleeper_adp.parquet
  - reports/season_projection_{season}.csv            (from project_season.py --out)
  - data/raw weekly_stats + id_map                    (in-house)

Output:
  - reports/external_projection_benchmark_{season}.md

Preseason-vs-preseason only. Every stat line is scored through OUR PPR ruleset so
the comparison is under one scoring rule. Pure transforms are unit-tested; the
end-to-end run is a manual phase.

Usage:
    python scripts/benchmark_projections.py --season 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, score
from projections.store import read_partition

_STAT_FIELDS = (
    "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds", "fumbles_lost",
)
_COUNT_FIELDS = frozenset(
    {"passing_tds", "interceptions", "rushing_tds", "receptions", "receiving_tds", "fumbles_lost"}
)


def _score_row(row: pd.Series, ruleset: Ruleset) -> float:
    kwargs = {
        f: int(round(float(row[f]))) if f in _COUNT_FIELDS else float(row[f])
        for f in _STAT_FIELDS
    }
    return score(StatLine(**kwargs), ruleset)


def actual_season_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Sum each player's weekly stat lines to a season total and score under `ruleset`.
    Position is the modal value across the player's weeks."""
    agg = {f: "sum" for f in _STAT_FIELDS}
    summed = weekly_stats.groupby("gsis_id", as_index=False).agg(agg)
    pos = (
        weekly_stats.groupby("gsis_id")["position"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    out = summed.merge(pos, on="gsis_id", how="left")
    out["actual_pts"] = out.apply(lambda r: _score_row(r, ruleset), axis=1)
    return out[["gsis_id", "position", "actual_pts"]]


def our_season_points(csv_df: pd.DataFrame) -> pd.DataFrame:
    """Our model's CSV: season_total_mean is already PPR fantasy points."""
    out = csv_df[["gsis_id", "position", "season_total_mean"]].copy()
    out = out.rename(columns={"season_total_mean": "our_pts"})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_benchmark_projections.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_projections.py tests/test_scripts/test_benchmark_projections.py
git commit -m "feat(spike): actuals + our-model season points for benchmark"
```

---

### Task 6: ESPN season points + join frame

**Files:**
- Modify: `scripts/benchmark_projections.py`
- Test: `tests/test_scripts/test_benchmark_projections.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_benchmark_frame_joins_on_gsis_and_scores_espn():
    # ESPN row keyed by espn_id; our + actual keyed by gsis_id; id_map crosswalks.
    espn = pd.DataFrame([{
        "espn_id": "E1", "full_name": "A B", "position": "WR",
        "espn_adp": 4.0, "espn_pos_rank": 2, "espn_actual_applied_total": 200.0,
        "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
        "rushing_yards": 0.0, "rushing_tds": 0,
        "receptions": 80, "receiving_yards": 1000.0, "receiving_tds": 5, "fumbles_lost": 0,
    }])  # ESPN proj PPR = 80 + 100 + 30 = 210.0
    ours = pd.DataFrame({"gsis_id": ["00-0000001"], "position": ["WR"], "our_pts": [195.0]})
    actuals = pd.DataFrame({"gsis_id": ["00-0000001"], "position": ["WR"], "actual_pts": [188.0]})
    id_map = pd.DataFrame({
        "gsis_id": ["00-0000001"], "espn_id": ["E1"], "sleeper_id": ["S1"],
        "full_name": ["A B"], "position": ["WR"], "team": ["KC"],
    })
    sleeper = pd.DataFrame({"sleeper_id": ["S1"], "sleeper_adp": [3.5]})
    frame = bench.build_benchmark_frame(espn, ours, actuals, id_map, sleeper, Ruleset.espn_ppr())
    row = frame.iloc[0]
    assert row["gsis_id"] == "00-0000001"
    assert row["espn_pts"] == 210.0
    assert row["our_pts"] == 195.0
    assert row["actual_pts"] == 188.0
    assert row["espn_adp"] == 4.0 and row["sleeper_adp"] == 3.5
    assert row["espn_pos_rank"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_benchmark_projections.py::test_build_benchmark_frame_joins_on_gsis_and_scores_espn -v`
Expected: FAIL with `AttributeError: ... 'build_benchmark_frame'`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
def espn_season_points(espn: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Score ESPN's preseason stat line under `ruleset`, keyed by espn_id."""
    out = espn.copy()
    out["espn_pts"] = out.apply(lambda r: _score_row(r, ruleset), axis=1)
    return out[
        ["espn_id", "full_name", "position", "espn_pts",
         "espn_adp", "espn_pos_rank", "espn_actual_applied_total"]
    ]


def build_benchmark_frame(
    espn: pd.DataFrame,
    ours: pd.DataFrame,
    actuals: pd.DataFrame,
    id_map: pd.DataFrame,
    sleeper: pd.DataFrame,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Join ESPN + our model + actuals on gsis_id (ESPN via id_map.espn_id,
    Sleeper ADP via id_map.sleeper_id). Base universe = actuals (ground truth).
    Position is taken from actuals."""
    espn_scored = espn_season_points(espn, ruleset)
    espn_keyed = espn_scored.merge(
        id_map[["gsis_id", "espn_id"]].dropna(subset=["espn_id"]), on="espn_id", how="left"
    )

    frame = actuals.copy()
    frame = frame.merge(ours[["gsis_id", "our_pts"]], on="gsis_id", how="left")
    # Drop ESPN's own position/full_name/espn_id before the merge: position comes
    # from actuals, full_name is re-attached from id_map below, and keeping any of
    # them here would create _x/_y collisions that break render_report.
    frame = frame.merge(
        espn_keyed.drop(columns=["position", "full_name", "espn_id"]).dropna(subset=["gsis_id"]),
        on="gsis_id", how="left",
    )

    sleeper_keyed = sleeper.merge(
        id_map[["gsis_id", "sleeper_id"]].dropna(subset=["sleeper_id"]),
        on="sleeper_id", how="left",
    ).dropna(subset=["gsis_id"])
    frame = frame.merge(sleeper_keyed[["gsis_id", "sleeper_adp"]], on="gsis_id", how="left")

    # full_name for readability (from id_map).
    frame = frame.merge(id_map[["gsis_id", "full_name"]], on="gsis_id", how="left")
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_benchmark_projections.py::test_build_benchmark_frame_joins_on_gsis_and_scores_espn -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_projections.py tests/test_scripts/test_benchmark_projections.py
git commit -m "feat(spike): score ESPN + join benchmark frame on gsis_id"
```

---

### Task 7: Metrics + cohort selection

**Files:**
- Modify: `scripts/benchmark_projections.py`
- Test: `tests/test_scripts/test_benchmark_projections.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np


def test_source_metrics_drops_nan_pairs_and_computes_error():
    frame = pd.DataFrame({
        "espn_pts": [100.0, 200.0, np.nan],   # 3rd row unmatched -> dropped
        "actual_pts": [110.0, 180.0, 50.0],
    })
    m = bench.source_metrics(frame, "espn_pts")
    assert m["n"] == 2
    # residuals: -10, +20 -> RMSE = sqrt((100+400)/2)=sqrt(250)=15.811..., MAE=15.0
    assert round(m["rmse"], 3) == 15.811
    assert m["mae"] == 15.0


def test_top_n_by_rank_picks_smallest_rank_per_position():
    frame = pd.DataFrame({
        "position": ["WR", "WR", "WR", "RB"],
        "espn_pos_rank": [1.0, 2.0, 3.0, 1.0],
        "actual_pts": [1, 2, 3, 4],
    })
    top2 = bench.top_n_by_rank(frame, "espn_pos_rank", n=2)
    assert set(zip(top2["position"], top2["espn_pos_rank"])) == {("WR", 1.0), ("WR", 2.0), ("RB", 1.0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_benchmark_projections.py -k "source_metrics or top_n_by_rank" -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
def source_metrics(frame: pd.DataFrame, pred_col: str, actual_col: str = "actual_pts") -> dict[str, float]:
    """RMSE / MAE / Spearman of pred vs actual over rows where both are present."""
    sub = frame[[pred_col, actual_col]].dropna()
    n = len(sub)
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "spearman": float("nan")}
    resid = sub[pred_col] - sub[actual_col]
    rmse = float((resid**2).mean() ** 0.5)
    mae = float(resid.abs().mean())
    spearman = float(sub[pred_col].corr(sub[actual_col], method="spearman")) if n > 1 else float("nan")
    return {"n": n, "rmse": rmse, "mae": mae, "spearman": spearman}


def top_n_by_rank(frame: pd.DataFrame, rank_col: str, n: int = 20) -> pd.DataFrame:
    """Top-n rows per position by smallest rank (best). Rows with NaN rank dropped."""
    ranked = frame.dropna(subset=[rank_col])
    return (
        ranked.sort_values(rank_col)
        .groupby("position", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_n_hit_rate(frame: pd.DataFrame, rank_col: str, n: int = 20) -> float:
    """Of each position's top-n by preseason rank, the share that finished top-n in actuals."""
    pre = top_n_by_rank(frame, rank_col, n)
    if pre.empty:
        return float("nan")
    actual_top = top_n_by_rank(
        frame.assign(_actual_rank=frame.groupby("position")["actual_pts"].rank(ascending=False)),
        "_actual_rank", n,
    )
    hit_keys = set(zip(actual_top["position"], actual_top["gsis_id"]))
    pre_keys = list(zip(pre["position"], pre["gsis_id"]))
    return sum(k in hit_keys for k in pre_keys) / len(pre_keys)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_benchmark_projections.py -k "source_metrics or top_n_by_rank" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_projections.py tests/test_scripts/test_benchmark_projections.py
git commit -m "feat(spike): RMSE/MAE/Spearman metrics + top-N cohort selection"
```

---

### Task 8: Report renderer + CLI main

**Files:**
- Modify: `scripts/benchmark_projections.py`
- Test: `tests/test_scripts/test_benchmark_projections.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_report_contains_verdict_and_per_source_rows():
    frame = pd.DataFrame({
        "gsis_id": ["00-0000001", "00-0000002"],
        "full_name": ["A B", "C D"],
        "position": ["WR", "RB"],
        "espn_pts": [210.0, 150.0], "our_pts": [195.0, None],
        "actual_pts": [188.0, 160.0],
        "espn_pos_rank": [2.0, 1.0], "espn_adp": [4.0, 1.0], "sleeper_adp": [3.5, 1.2],
    })
    md = bench.render_report(frame, season=2024)
    assert "# External Projection Benchmark" in md
    assert "ESPN" in md and "Ours" in md
    assert "Verdict" in md
    # our model missing a row (C D) must be surfaced as coverage
    assert "match" in md.lower() or "coverage" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_benchmark_projections.py::test_render_report_contains_verdict_and_per_source_rows -v`
Expected: FAIL with `AttributeError: ... 'render_report'`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
_POSITIONS = ("QB", "RB", "WR", "TE")


def _metric_block(frame: pd.DataFrame, label: str) -> list[str]:
    lines = [f"### {label}", "", "| Source | n | RMSE | MAE | Spearman |", "|---|---:|---:|---:|---:|"]
    for src, col in (("ESPN", "espn_pts"), ("Ours", "our_pts")):
        m = source_metrics(frame, col)
        lines.append(
            f"| {src} | {m['n']} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['spearman']:.3f} |"
        )
    lines.append("")
    return lines


def render_report(frame: pd.DataFrame, season: int) -> str:
    out: list[str] = [f"# External Projection Benchmark — {season}", ""]
    out += [
        f"Preseason-vs-preseason. Every stat line scored through our ESPN PPR ruleset. "
        f"Base universe = {len(frame)} players with a {season} actual season.",
        "",
        "## Coverage / match rate",
        "",
        f"- ESPN matched: {int(frame['espn_pts'].notna().sum())} / {len(frame)}",
        f"- Ours matched: {int(frame['our_pts'].notna().sum())} / {len(frame)} "
        f"(unmatched are mostly rookies our model cannot project — a real, reportable weakness)",
        "",
        "## Full population",
        "",
    ]
    out += _metric_block(frame, "All QB/RB/WR/TE")
    for pos in _POSITIONS:
        out += _metric_block(frame[frame["position"] == pos], f"{pos} only")

    out += ["## Top-20 per position (by ESPN preseason rank)", ""]
    top = top_n_by_rank(frame, "espn_pos_rank", 20)
    out += _metric_block(top, "Top-20/pos — all")

    out += ["## Veterans-only (rows our model projects)", ""]
    out += _metric_block(frame[frame["our_pts"].notna()], "Veterans — all")

    out += [
        "## ADP rank lens (vs actual finish)",
        "",
        "| Ranking | Spearman vs actual | Top-20 hit rate |",
        "|---|---:|---:|",
    ]
    for label, col in (("ESPN ADP", "espn_adp"), ("Sleeper ADP", "sleeper_adp")):
        sub = frame[[col, "actual_pts"]].dropna()
        # ADP: smaller = better, so correlate negative ADP with actual points.
        sp = float((-sub[col]).corr(sub["actual_pts"], method="spearman")) if len(sub) > 1 else float("nan")
        hit = top_n_hit_rate(frame.assign(_adp_rank=frame[col]), "_adp_rank", 20)
        out.append(f"| {label} | {sp:.3f} | {hit:.2f} |")
    out += [""]

    espn_all = source_metrics(frame, "espn_pts")
    our_all = source_metrics(frame[frame["our_pts"].notna()], "our_pts")
    out += [
        "## Verdict",
        "",
        f"- ESPN full-population RMSE: {espn_all['rmse']:.2f} (n={espn_all['n']})",
        f"- Our model veterans-only RMSE: {our_all['rmse']:.2f} (n={our_all['n']})",
        "",
        "_Reading notes:_ one season (2024) — rerun on 2023 if close. ESPN is one strong "
        "source, not a consensus: losing to ESPN alone strongly implies losing to a consensus; "
        "beating ESPN alone does not yet prove beating a consensus. Our model cannot project "
        "pure rookies (no prior-NFL features); the veterans-only cut is the fairest model-vs-model "
        "comparison.",
        "",
        "**Go/no-go for sub-project #2 (external consensus layer):** _fill in after reading the "
        "numbers above — see plan Task 9._",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark external vs our projections for one season.")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--ext-root", type=Path, default=Path("data/external_projections"))
    ap.add_argument("--our-csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    our_csv = args.our_csv or Path("reports") / f"season_projection_{args.season}.csv"
    out_path = args.out or Path("reports") / f"external_projection_benchmark_{args.season}.md"
    ext_dir = args.ext_root / str(args.season)
    ruleset = Ruleset.espn_ppr()

    espn = pd.read_parquet(ext_dir / "espn.parquet")
    sleeper = pd.read_parquet(ext_dir / "sleeper_adp.parquet")
    ours = our_season_points(pd.read_csv(our_csv))
    weekly = read_partition(args.raw_root, "weekly_stats", season=args.season)
    actuals = actual_season_points(weekly, ruleset)
    id_map = read_partition(args.raw_root, "id_map")

    frame = build_benchmark_frame(espn, ours, actuals, id_map, sleeper, ruleset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(frame, args.season), encoding="utf-8")
    print(f"Wrote {out_path} ({len(frame)} players)", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + lint + type**

Run: `pytest tests/test_scripts/test_benchmark_projections.py -v`
Expected: all PASS.
Run: `ruff check scripts/benchmark_projections.py tests/test_scripts/test_benchmark_projections.py && ruff format --check scripts/benchmark_projections.py tests/test_scripts/test_benchmark_projections.py && mypy src tests`
Expected: no violations.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_projections.py tests/test_scripts/test_benchmark_projections.py
git commit -m "feat(spike): benchmark report renderer + CLI"
```

---

## Phase 3 — Real run + verdict

### Task 9: End-to-end run, fill the verdict, update PM/TODO

**Files:**
- Creates: `reports/season_projection_2024.csv`, `reports/external_projection_benchmark_2024.md`
- Modify: `reports/external_projection_benchmark_2024.md` (verdict paragraph), `project_management.md`, `TODO.md`

- [ ] **Step 1: Generate our model's 2024 projection**

Run: `python scripts/project_season.py --season 2024 --out reports/season_projection_2024.csv`
Expected: trains QB/RB/WR/TE on 2018–2023, projects 2024, writes the CSV. (Long-running — minutes. If the production EnsembleModel/lightgbm path is too slow, this is acceptable; let it finish.)

- [ ] **Step 2: Run the benchmark**

Run: `python scripts/benchmark_projections.py --season 2024`
Expected: `Wrote reports/external_projection_benchmark_2024.md (<N> players)`.

- [ ] **Step 3: Read the report and write the verdict**

Open `reports/external_projection_benchmark_2024.md`. Replace the placeholder go/no-go line with a concrete recommendation grounded in the numbers, using this rubric:
- If ESPN full-population RMSE is meaningfully **lower** than our veterans-only RMSE (ours worse even on its favorable cut) → **GO**: retire home-grown modeling, build the consensus layer.
- If ours is **lower** (we beat ESPN even though ESPN projects rookies) → **HOLD**: widen to a multi-source consensus before deciding; our model may be worth keeping.
- If within ~1–2 fantasy points/season → **lean GO**: the model isn't earning its complexity; prefer consensus + tools, state it explicitly.
Also confirm the rookie-coverage gap (Ours matched << ESPN matched) is reflected honestly in the verdict.

- [ ] **Step 4: Sanity-check the verdict for plausibility**

Cross-check two or three top-2024 players in the frame by hand (e.g. confirm a known boom like Saquon shows a large positive actual-minus-projection on both sources, and that neither source's "projection" suspiciously equals the actual — which would signal a contaminated pull). State in the report that this check passed.

- [ ] **Step 5: Record the verdict in project_management.md + TODO.md and commit**

Add a `project_management.md` top entry (mirror the house format: date, branch, status, verdict, what-it-closes, recommended-next-direction) summarizing the benchmark result and the go/no-go on sub-project #2. Add/update a `TODO.md` item for "External consensus projection layer (sub-project #2)" reflecting the verdict.

```bash
git add reports/season_projection_2024.csv reports/external_projection_benchmark_2024.md \
        project_management.md TODO.md
git commit -m "report(spike): 2024 external projection benchmark verdict"
```

---

## Final verification (run before opening the PR)

- [ ] `pytest -v -k "external or benchmark or pull"` — spike tests pass.
- [ ] `pytest -v -k "ingest or store or schemas"` — no dtype/schema regressions (spec/CLAUDE.md gate for any store-touching path; this spike reads partitions).
- [ ] `mypy src tests` — zero violations.
- [ ] `ruff check src tests scripts` — zero violations.
- [ ] `ruff format --check src tests scripts` — no drift.
- [ ] Paste the command outputs into the PR description as evidence (per CLAUDE.md "forced verification").

---

## Spec coverage self-check

- Goal / go-no-go verdict → Task 9.
- Preseason-vs-preseason, ESPN clean (verified) → Background facts + Task 4 Step 2.
- Score through our ruleset, common stat set → Tasks 5–6, `_STAT_FIELDS`.
- Two cohorts (full + top-20/pos by preseason rank) → Task 8 `render_report`.
- Veterans-only sub-cut (rookie coverage) → Task 8 `render_report`.
- Match-rate / coverage reported → Task 8 `render_report`.
- ADP rank lens (ESPN + Sleeper) → Tasks 3, 6, 8.
- Metrics RMSE/MAE/Spearman/top-20 hit-rate → Task 7.
- No core changes; spike lives in `scripts/` + `data/external_projections/` → all tasks.
- Synthetic-fixture tests for pure transforms; network verified by running → Phases 1–3.
- Honest-reading notes (one season, one source, rookie gap) → Task 8 verdict block.
