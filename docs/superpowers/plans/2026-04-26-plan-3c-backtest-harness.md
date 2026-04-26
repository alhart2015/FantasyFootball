# Plan 3c — Walk-forward backtest harness with snapshot-diff gating — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a walk-forward backtest harness over 4 held-out years × 4 positions with snapshot-diff gating so future model changes have to clear a regression bar in pytest. Closes TODO #4 (feature parquet caching).

**Architecture:** New `src/projections/backtest/` package (harness + metrics + naive baseline + snapshot diff). New feature-cache layer at `data/features/{position}/season=YYYY/week=WW/part.parquet` with reader at `src/projections/features/cache.py` and writer at `scripts/refresh_features.py`. Snapshot file at `tests/backtest/baseline_metrics.json` (~352 rows, committed) plus tolerance config at `tests/backtest/tolerances.json`. Opt-in `pytest -m backtest --run-backtest` mirrors the existing `--run-network` plumbing. Plan 3c uses summed weekly means as the season aggregator; real Monte Carlo aggregation is Plan 3d.

**Tech Stack:** Python 3.11+, pandas, pandera (`pandera.pandas`), scikit-learn (`RidgeCV`), pytest, mypy strict, ruff, joblib (for parallelism if needed).

**Spec:** `docs/superpowers/specs/2026-04-26-plan-3c-backtest-harness-design.md` (commit `3b53058` on `feat/plan-3c-backtest-harness`).

**Branch:** `feat/plan-3c-backtest-harness`. All commands below run from the repo root on this branch.

**Venv note (Windows / bash):** prepend `/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH` (or activate the venv directly) when running `pytest` / `ruff` / `mypy` / `git commit` (pre-commit hooks need the venv on PATH). Examples below assume venv-on-PATH; the explicit `PATH=...` prefix is shown only on the first commit so the pattern is clear.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `.gitignore` | modify | Add `data/features/` and `data/backtest/` to the existing `data/raw/` / `data/projections/` ignores. |
| `pyproject.toml` | modify | Register the `backtest` pytest marker alongside `network`. |
| `tests/conftest.py` | modify | Register `--run-backtest` CLI flag and skip logic for `@pytest.mark.backtest`, mirroring the existing `--run-network` block. |
| `src/projections/features/cache.py` | create | `read_features(position, season, weeks=None, features_root=...) -> pd.DataFrame`, validates against the appropriate `*FeaturesSchema`. |
| `src/projections/backtest/__init__.py` | create | Public surface: re-export `run_backtest`, `BacktestRun`, `GateResult`, `read_snapshot`, `write_snapshot`, `diff_snapshot`. |
| `src/projections/backtest/harness.py` | create | `BacktestRun` dataclass + `run_backtest(...)` walk-forward driver. |
| `src/projections/backtest/metrics.py` | create | Per-stat RMSE / MAE / mean_pred, composite RMSE / MAE, Spearman top-N, calibration coverage. |
| `src/projections/backtest/naive.py` | create | Per-player trailing-4-game stat-mean baseline + per-position cold-start fallback. |
| `src/projections/backtest/snapshot.py` | create | `read_snapshot` / `write_snapshot` / `diff_snapshot` + tolerance application + direction-aware comparison. |
| `scripts/refresh_features.py` | create | CLI: `python scripts/refresh_features.py {position\|all} [--seasons RANGE]`. Iterates (position × season × week), calls each builder, writes to `data/features/`. |
| `scripts/backtest.py` | create | CLI: `--check` (default), `--update-snapshot`, `--report`. |
| `tests/test_features/test_cache.py` | create | Unit tests for `read_features`. |
| `tests/test_backtest/conftest.py` | create | Synthetic eval-DataFrame fixtures shared across metrics / naive / harness tests. |
| `tests/test_backtest/test_metrics.py` | create | Unit tests for the metric primitives. |
| `tests/test_backtest/test_naive.py` | create | Unit tests for `compute_naive_predictions`. |
| `tests/test_backtest/test_snapshot.py` | create | Unit tests for `read_snapshot` / `write_snapshot` / `diff_snapshot`. |
| `tests/test_backtest/test_harness.py` | create | Integration test for `run_backtest` against synthetic feature cache + raw data. |
| `tests/backtest/__init__.py` | create | Empty marker so pytest treats this dir as a test package distinct from `tests/test_backtest/`. |
| `tests/backtest/baseline_metrics.json` | create (Phase 6) | The v1 gated snapshot; emitted by `scripts/backtest.py --update-snapshot` against the populated feature cache. |
| `tests/backtest/tolerances.json` | create | Per-metric-type defaults + empty `overrides` list. |
| `tests/backtest/test_backtest_smoke.py` | create | Default-on smoke covering one (position, year) cell. |
| `tests/backtest/test_backtest_gate.py` | create | Opt-in `@pytest.mark.backtest` test that runs the full harness + diffs the snapshot. |
| `CONTRIBUTING.md` | modify | Add "After touching `src/projections/features/`" instruction; add backtest-gate workflow notes. |
| `TODO.md` | modify | Phase 6: close #4. Phase 6: add three new TODOs (#19, #20, #21 numbering chosen by editor at write time). |
| `project_management.md` | modify | Phase 6: prepend post-3c status section with per-(position, year) metric table; bump "Next action" to Plan 3d. |

---

## Phase 1 — Feature cache layer (closes TODO #4)

### Task 1: gitignore + `cache.py read_features`

**Files:**
- Modify: `.gitignore`
- Create: `src/projections/features/cache.py`
- Create: `tests/test_features/test_cache.py`

**Why TDD-shaped:** the read helper has a specific contract (validates against the position's schema, supports week filtering, raises FileNotFoundError on missing cache). Test the contract first.

- [ ] **Step 1.1: Update `.gitignore`.**

Open `.gitignore`. Find the existing `data/raw/` / `data/projections/` lines (search for `data/raw`). Add two lines in the same block:

```
data/features/
data/backtest/
```

- [ ] **Step 1.2: Write the failing test for cache reading.**

Create `tests/test_features/test_cache.py` with this content:

```python
"""Unit tests for src/projections/features/cache.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.features.cache import read_features
from projections.schemas import Position
from projections.store import write_partition


def _minimal_wr_features_row(week: int) -> dict[str, object]:
    """Construct one fully-populated WrFeaturesSchema row.
    Mirrors the column set used by `tests/test_features/conftest.py` fixtures.
    """
    return {
        "gsis_id": "00-0036322",
        "season": 2024,
        "week": week,
        "position": Position.WR.value,
        "team": "MIN",
        "depth_rank": 1,
        "targets_per_game_l4": 8.0,
        "targets_per_game_std": 1.0,
        "target_share_l4": 0.30,
        "receptions_per_game_l4": 6.0,
        "receiving_yards_per_game_l4": 80.0,
        "receiving_tds_per_game_l4": 0.5,
        "rushing_attempts_per_game_l4": 0.0,
        "rushing_yards_per_game_l4": 0.0,
        "snap_pct_l4": 0.95,
        "avg_separation_std": 2.5,
        "avg_intended_air_yards_std": 12.0,
        "avg_yac_above_expectation_std": 0.3,
        "implied_team_total": 24.5,
        "spread": -3.0,
        "is_home": True,
        "roof_dome": False,
        "opp_allowed_wr_fppg_l4": 35.0,
        "opponent": "GB",
    }


def test_read_features_validates_and_returns_concatenated_weeks(
    tmp_path: Path,
) -> None:
    """read_features returns one DataFrame across all available weeks for a
    (position, season) and validates against WrFeaturesSchema."""
    features_root = tmp_path
    for week in (1, 2, 3):
        df = pd.DataFrame([_minimal_wr_features_row(week)])
        write_partition(features_root, "wr", df, season=2024, week=week)

    out = read_features(Position.WR, 2024, features_root=features_root)
    assert len(out) == 3
    assert sorted(out["week"].tolist()) == [1, 2, 3]
    assert out["gsis_id"].iloc[0] == "00-0036322"


def test_read_features_filters_to_requested_weeks(tmp_path: Path) -> None:
    """The optional `weeks` kwarg returns only those week partitions."""
    features_root = tmp_path
    for week in (1, 2, 3):
        df = pd.DataFrame([_minimal_wr_features_row(week)])
        write_partition(features_root, "wr", df, season=2024, week=week)

    out = read_features(Position.WR, 2024, weeks=[2, 3], features_root=features_root)
    assert sorted(out["week"].tolist()) == [2, 3]


def test_read_features_raises_when_cache_missing(tmp_path: Path) -> None:
    """If the (position, season) directory has no parquet partitions, raise
    FileNotFoundError with a clear message."""
    with pytest.raises(FileNotFoundError, match="No feature cache"):
        read_features(Position.WR, 2024, features_root=tmp_path)
```

- [ ] **Step 1.3: Run the test — expect failure.**

Run: `pytest tests/test_features/test_cache.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projections.features.cache'`.

- [ ] **Step 1.4: Implement `read_features`.**

Create `src/projections/features/cache.py`:

```python
"""Feature-cache reader. Pairs with scripts/refresh_features.py (writer).

Plan 3c — closes TODO #4. The cache layout
``data/features/{position}/season=YYYY/week=WW/part.parquet`` mirrors the
existing ``data/raw/{table}/...`` and ``data/projections/...`` conventions.
``{position}`` is lowercase (qb / rb / te / wr).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from projections.schemas import Position
from projections.store import read_partition


def read_features(
    position: Position,
    season: int,
    *,
    weeks: Iterable[int] | None = None,
    features_root: Path = Path("data/features"),
) -> pd.DataFrame:
    """Load cached features for a (position, season).

    Returns a DataFrame concatenated across the requested weeks (or all
    available weeks for the season if ``weeks`` is None), re-validated
    against the appropriate FeaturesSchema looked up via POSITION_DISPATCH.

    Raises:
        FileNotFoundError: the (position, season) cache directory is missing
            or empty. The error message names the path so the caller can
            run ``scripts/refresh_features.py`` against it.
    """
    # Local import to avoid a top-level circular: __init__.py imports baseline,
    # baseline imports schemas, schemas is imported by features/cache.py.
    from projections.models import POSITION_DISPATCH

    table = position.value.lower()
    season_dir = features_root / table / f"season={season}"
    if not season_dir.exists() or not any(season_dir.rglob("part.parquet")):
        raise FileNotFoundError(
            f"No feature cache for ({position.value}, {season}) at {season_dir}. "
            f"Run: python scripts/refresh_features.py {table} --seasons {season}"
        )

    if weeks is None:
        df = read_partition(features_root, table, season=season)
    else:
        frames = [
            read_partition(features_root, table, season=season, week=int(w))
            for w in weeks
        ]
        df = pd.concat(frames, ignore_index=True)

    schema = POSITION_DISPATCH[position].feature_schema
    return schema.validate(df)
```

- [ ] **Step 1.5: Run the test — expect pass.**

Run: `pytest tests/test_features/test_cache.py -v`

Expected: 3 tests pass.

- [ ] **Step 1.6: Commit.**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add .gitignore src/projections/features/cache.py tests/test_features/test_cache.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(features): cache.read_features helper + gitignore data/features

Plan 3c Phase 1, part 1. The backtest harness needs cached feature
partitions so the 16 walk-forward fits don't recompute features from
raw on every run. read_features is the canonical reader; the writer
(scripts/refresh_features.py) lands in the next commit.

Layout: data/features/{position}/season=YYYY/week=WW/part.parquet
matching the existing data/raw/ and data/projections/ partition
conventions. {position} is lowercase (qb / rb / te / wr) so the
table name maps cleanly to POSITION_DISPATCH lookups.

The read path validates against the position's *FeaturesSchema (looked
up via POSITION_DISPATCH) so a stale cache that survived a schema
change fails loudly instead of silently corrupting metrics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(All subsequent commits in this plan assume the venv is on PATH; the prefix is omitted for brevity.)

---

### Task 2: `scripts/refresh_features.py` (writer)

**Files:**
- Create: `scripts/refresh_features.py`

This task has no automated test file: it's a CLI that drives feature builders against real raw data. The smoke is a manual run in Phase 6 (Task 13). Unit tests for the builders themselves already exist under `tests/test_features/`.

- [ ] **Step 2.1: Implement `scripts/refresh_features.py`.**

Create `scripts/refresh_features.py`:

```python
"""Plan 3c Phase 1 — refresh the feature cache from raw data.

For each (position, season) pair, iterate every week present in the season's
depth_charts partition, call the position's feature builder via
POSITION_DISPATCH, validate against the position's FeaturesSchema, and write
to data/features/{position}/season=YYYY/week=WW/part.parquet.

Per-week feature builds need raw data for the prior season too (rolling
windows of trailing 4 games can cross a season boundary at week 1-4 of a
season). The script always concatenates the prior season's raw data when
present.

Usage:
    python scripts/refresh_features.py wr --seasons 2018-2024
    python scripts/refresh_features.py all --seasons 2018-2024
    python scripts/refresh_features.py qb              # default seasons 2018-2024
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from projections.models import POSITION_DISPATCH
from projections.schemas import Position
from projections.store import read_partition, write_partition

_DEFAULT_SEASONS = range(2018, 2025)


def _parse_season_range(s: str) -> range:
    """`"2018-2024"` -> `range(2018, 2025)`; `"2024"` -> `range(2024, 2025)`."""
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return range(lo, hi + 1)
    n = int(s)
    return range(n, n + 1)


def _read_raw_with_prior(
    raw_root: Path, table: str, season: int, *, include_prior: bool = True
) -> pd.DataFrame:
    """Read raw partition for (table, season), optionally concatenated with
    (table, season-1) so trailing-4 rolling windows have history at week 1-4."""
    cur = read_partition(raw_root, table, season=season)
    if not include_prior:
        return cur
    try:
        prior = read_partition(raw_root, table, season=season - 1)
    except FileNotFoundError:
        return cur
    return pd.concat([prior, cur], ignore_index=True)


def _refresh_one(
    position: Position,
    season: int,
    *,
    raw_root: Path,
    features_root: Path,
) -> int:
    """Build + write every available week of features for (position, season).
    Returns the number of week partitions written."""
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_kwarg = {
        "passing": "ngs_passing",
        "rushing": "ngs_rushing",
        "receiving": "ngs_receiving",
    }[dispatch.ngs_stat_type]
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"

    ws_full = _read_raw_with_prior(raw_root, "weekly_stats", season)
    sc_full = _read_raw_with_prior(raw_root, "snap_counts", season)
    ngs_full = _read_raw_with_prior(raw_root, ngs_table, season)
    dc = read_partition(raw_root, "depth_charts", season=season)
    sch = read_partition(raw_root, "schedules", season=season)

    weeks = sorted(int(w) for w in dc["week"].unique())
    written = 0
    table = position.value.lower()
    for week in weeks:
        kwargs: dict[str, Any] = {
            "weekly_stats": ws_full,
            "snap_counts": sc_full,
            "depth_charts": dc,
            "schedules": sch,
            "season": season,
            "as_of_week": week,
            ngs_kwarg: ngs_full,
        }
        feats = builder(**kwargs)
        if feats.empty:
            continue
        feats = dispatch.feature_schema.validate(feats)
        write_partition(features_root, table, feats, season=season, week=week)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the feature cache for a position (or all)."
    )
    parser.add_argument(
        "position",
        choices=["qb", "rb", "te", "wr", "all"],
        help="Target position, or 'all' for QB/RB/TE/WR.",
    )
    parser.add_argument(
        "--seasons",
        default="2018-2024",
        help="Inclusive season range, e.g. '2018-2024' or '2024'. Default 2018-2024.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    seasons = _parse_season_range(args.seasons)
    raw_root = args.data_root / "raw"
    features_root = args.data_root / "features"

    positions: tuple[Position, ...] = (
        (Position.QB, Position.RB, Position.TE, Position.WR)
        if args.position == "all"
        else (Position(args.position.upper()),)
    )

    total = 0
    for position in positions:
        for season in seasons:
            n = _refresh_one(
                position,
                season,
                raw_root=raw_root,
                features_root=features_root,
            )
            print(f"  {position.value} {season}: wrote {n} week partition(s)")
            total += n
    print(f"\nTotal partitions written: {total}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Smoke-run on a single (position, season) where raw data already exists.**

Run: `python scripts/refresh_features.py wr --seasons 2024`

Expected: prints `  WR 2024: wrote N week partition(s)` for some N (likely 17–22) and `Total partitions written: N`. Creates files under `data/features/wr/season=2024/week=*/part.parquet`.

If raw data for 2024 isn't present locally (`data/raw/` is gitignored), this step may fail with `FileNotFoundError: No parquet partitions under data/raw/...`. That's expected on a fresh checkout. In that case, run `python -c "from projections.ingest.refresh import refresh; from pathlib import Path; refresh(seasons=range(2018, 2025), data_root=Path('data'))"` first (TODO #18 captures the missing CLI entrypoint).

- [ ] **Step 2.3: Verify schema validation by reading back.**

Run: `python -c "from projections.features.cache import read_features; from projections.schemas import Position; df = read_features(Position.WR, 2024); print(df.shape, sorted(df['week'].unique()))"`

Expected: prints something like `(N, M) [1, 2, ..., 22]`.

- [ ] **Step 2.4: Commit.**

```bash
git add scripts/refresh_features.py
git commit -m "$(cat <<'EOF'
feat(scripts): refresh_features.py — populate data/features/ cache

Plan 3c Phase 1, part 2. CLI driver that iterates POSITION_DISPATCH
to build each position's per-week features for a season range and
write them to the cache via store.write_partition. Idempotent —
write_partition overwrites by design.

Per-week builds always concatenate the prior season's raw data when
present so trailing-4 rolling features at week 1-4 of a season have
the history they need. This mirrors the existing pattern in
scripts/sanity_check_baseline.py.

Closes the writer side of TODO #4 (closed for real in Phase 6 once the
cache is populated and the gate runs).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Backtest package skeleton

### Task 3: `backtest/` package + `BacktestRun` dataclass + stub `run_backtest`

**Files:**
- Create: `src/projections/backtest/__init__.py`
- Create: `src/projections/backtest/harness.py`
- Create: `tests/test_backtest/__init__.py` (empty)
- Create: `tests/test_backtest/conftest.py`
- Create: `tests/test_backtest/test_harness.py`

This task creates the package shell and a stub `run_backtest` that returns an empty `BacktestRun`. Real metrics get wired in Phase 3. The structural test in this task locks in the public surface so Phase 3's unit tests against `metrics.py` and `naive.py` can run independently.

- [ ] **Step 3.1: Write the failing structural test.**

Create `tests/test_backtest/__init__.py` as an empty file (so pytest treats the directory as a test package; mirrors `tests/test_models/__init__.py`).

Create `tests/test_backtest/conftest.py` with:

```python
"""Synthetic fixtures for the backtest unit tests.

Plan 3c Phase 2 onward. Fixtures live here (not in the per-file test
modules) so test_metrics.py / test_naive.py / test_snapshot.py /
test_harness.py can share a coherent set of synthetic inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def fake_eval_df() -> pd.DataFrame:
    """A tiny synthetic eval DataFrame matching the shape produced by
    harness.run_backtest's inner-join of predictions and actuals.

    Three player-weeks for two players. receptions/receiving_yards
    columns are suffixed _pred / _actual. Composite columns: mean,
    p10, p90, actual_ppr. (Other p-quantiles and per-stat columns
    are omitted; tests only consume what they assert against.)
    """
    return pd.DataFrame(
        {
            "gsis_id": ["00-A", "00-A", "00-B"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 1],
            "mean": [12.0, 14.0, 6.0],
            "p10": [4.0, 6.0, 1.0],
            "p90": [22.0, 24.0, 14.0],
            "actual_ppr": [10.0, 18.0, 4.0],
            "receptions_pred": [4.0, 5.0, 2.0],
            "receptions_actual": [3.0, 6.0, 1.0],
            "receiving_yards_pred": [55.0, 70.0, 25.0],
            "receiving_yards_actual": [40.0, 95.0, 12.0],
        }
    )
```

Create `tests/test_backtest/test_harness.py` with:

```python
"""Structural tests for src/projections/backtest/harness.py."""

from __future__ import annotations

import pandas as pd

from projections.backtest import BacktestRun, run_backtest


def test_backtest_run_dataclass_shape() -> None:
    """BacktestRun is a frozen slots dataclass with the four documented
    attributes (timestamp, metrics, naive_metrics, per_row_results)."""
    run = BacktestRun(
        timestamp=pd.Timestamp("2026-04-26", tz="UTC"),
        metrics=pd.DataFrame(columns=["position", "year", "metric", "value"]),
        naive_metrics=pd.DataFrame(columns=["position", "year", "metric", "value"]),
        per_row_results=pd.DataFrame(),
    )
    assert isinstance(run.metrics, pd.DataFrame)
    assert isinstance(run.naive_metrics, pd.DataFrame)
    assert run.timestamp.tzname() == "UTC"


def test_run_backtest_skeleton_returns_empty_metrics_when_no_positions() -> None:
    """Calling run_backtest with positions=[] returns an empty BacktestRun
    with the expected schema. This is the structural smoke before Phase 3
    wires real metrics."""
    out = run_backtest(positions=[], held_out_years=[])
    assert isinstance(out, BacktestRun)
    assert list(out.metrics.columns) == ["position", "year", "metric", "value"]
    assert list(out.naive_metrics.columns) == ["position", "year", "metric", "value"]
    assert out.metrics.empty
```

- [ ] **Step 3.2: Run the test — expect failure.**

Run: `pytest tests/test_backtest/test_harness.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projections.backtest'`.

- [ ] **Step 3.3: Implement the package skeleton.**

Create `src/projections/backtest/__init__.py`:

```python
"""Walk-forward backtest harness + snapshot-diff gating.

Plan 3c. Public surface for the harness, the metric primitives, the naive
baseline, and the snapshot-diff machinery.
"""

from __future__ import annotations

from projections.backtest.harness import BacktestRun, run_backtest

__all__ = [
    "BacktestRun",
    "run_backtest",
]
```

Create `src/projections/backtest/harness.py`:

```python
"""Walk-forward backtest driver.

For each (position, year) in the cartesian product, train Model A on
cached features for [train_start, year-1], predict every week of `year`
from cached features, score against actuals from data/raw/weekly_stats,
and return a BacktestRun with model + naive metrics.

Plan 3c Phase 2 lands the dataclass + driver shell with empty metrics.
Phase 3 wires metrics.py + naive.py into the per-(position, year) loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from projections.schemas import Position, Ruleset

_METRICS_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """Result of a single walk-forward backtest invocation.

    Attributes:
        timestamp: UTC time the run started; used to name diagnostic
            output directories under data/backtest/run_<ts>/.
        metrics: long-form DataFrame with columns
            (position, year, metric, value) — the model's metrics across
            (position, year, metric) cells. Becomes the snapshot input.
        naive_metrics: same shape; computed alongside model metrics for
            informational reporting. Not gated.
        per_row_results: per-(position, year, week, gsis_id) row of
            actuals + model predictions for diagnosis. Plan 3c writes
            this to data/backtest/run_<ts>/results.parquet (gitignored).
    """

    timestamp: pd.Timestamp
    metrics: pd.DataFrame
    naive_metrics: pd.DataFrame
    per_row_results: pd.DataFrame


def run_backtest(
    *,
    held_out_years: Iterable[int] = (2021, 2022, 2023, 2024),
    positions: Iterable[Position] | None = None,
    train_start: int = 2018,
    features_root: Path = Path("data/features"),
    raw_root: Path = Path("data/raw"),
    ruleset: Ruleset | None = None,
) -> BacktestRun:
    """Walk-forward backtest. Plan 3c Phase 2 returns an empty BacktestRun;
    Phase 3 fills in metrics + naive_metrics + per_row_results."""
    if ruleset is None:
        ruleset = Ruleset.espn_ppr()
    if positions is None:
        positions = (Position.QB, Position.RB, Position.TE, Position.WR)

    timestamp = pd.Timestamp(datetime.now(UTC))

    # Phase 2 stub: just enumerate the cartesian product to validate args.
    # Phase 3 replaces this with real per-cell training + scoring.
    _ = list(positions)
    _ = list(held_out_years)
    _ = train_start
    _ = features_root
    _ = raw_root
    _ = ruleset

    empty_metrics = pd.DataFrame(columns=list(_METRICS_COLUMNS))
    return BacktestRun(
        timestamp=timestamp,
        metrics=empty_metrics,
        naive_metrics=empty_metrics.copy(),
        per_row_results=pd.DataFrame(),
    )
```

- [ ] **Step 3.4: Run the test — expect pass.**

Run: `pytest tests/test_backtest/test_harness.py -v`

Expected: 2 tests pass.

- [ ] **Step 3.5: Quality gate (incremental).**

Run:
```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations across all three. Fix any drift before committing.

- [ ] **Step 3.6: Commit.**

```bash
git add src/projections/backtest/__init__.py src/projections/backtest/harness.py tests/test_backtest/__init__.py tests/test_backtest/conftest.py tests/test_backtest/test_harness.py
git commit -m "$(cat <<'EOF'
feat(backtest): package skeleton + BacktestRun dataclass

Plan 3c Phase 2. Lands the public surface for the backtest package and
a stub run_backtest that returns an empty BacktestRun with the documented
metrics columns. Phase 3 fills in real metrics + naive baselines.

The stub is wired so downstream callers (scripts/backtest.py, the
pytest gate test) can be drafted against the final API even before the
internals are fully populated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Metrics + naive baseline

### Task 4: `backtest/metrics.py` — metric primitives

**Files:**
- Create: `src/projections/backtest/metrics.py`
- Create: `tests/test_backtest/test_metrics.py`

The metric primitives consume an `eval_df` (the inner-join of predictions and actuals) plus the `target_stats` tuple, and return a `dict[str, float]` per metric category. The harness composes these per (position, year). `eval_df`'s shape is documented in `tests/test_backtest/conftest.py:fake_eval_df`.

- [ ] **Step 4.1: Write the failing metrics tests.**

Create `tests/test_backtest/test_metrics.py`:

```python
"""Unit tests for src/projections/backtest/metrics.py."""

from __future__ import annotations

import math

import pandas as pd

from projections.backtest.metrics import (
    compute_calibration_metrics,
    compute_composite_metrics,
    compute_per_stat_metrics,
    compute_spearman_topN,
)
from projections.schemas import Stat


def test_compute_per_stat_metrics_returns_rmse_mae_mean_pred(
    fake_eval_df: pd.DataFrame,
) -> None:
    """Per-stat RMSE, MAE, and mean_pred computed against suffixed columns."""
    out = compute_per_stat_metrics(fake_eval_df, target_stats=(Stat.RECEPTIONS,))

    # receptions_pred = [4, 5, 2], receptions_actual = [3, 6, 1]
    # diffs = [1, -1, 1] -> abs mean = 1.0; rmse = sqrt(mean([1, 1, 1])) = 1.0
    assert math.isclose(out["receptions_rmse"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["receptions_mae"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["receptions_mean_pred"], (4 + 5 + 2) / 3, rel_tol=1e-9)


def test_compute_per_stat_metrics_handles_multiple_stats(
    fake_eval_df: pd.DataFrame,
) -> None:
    """Two stats produce 6 keys (3 per stat)."""
    out = compute_per_stat_metrics(
        fake_eval_df, target_stats=(Stat.RECEPTIONS, Stat.RECEIVING_YARDS)
    )
    assert set(out.keys()) == {
        "receptions_rmse",
        "receptions_mae",
        "receptions_mean_pred",
        "receiving_yards_rmse",
        "receiving_yards_mae",
        "receiving_yards_mean_pred",
    }


def test_compute_composite_metrics(fake_eval_df: pd.DataFrame) -> None:
    """Composite RMSE/MAE against the `mean` and `actual_ppr` columns."""
    out = compute_composite_metrics(fake_eval_df)
    # diffs = [12-10, 14-18, 6-4] = [2, -4, 2] -> abs mean = 8/3; rmse = sqrt((4+16+4)/3)
    assert math.isclose(out["composite_mae"], 8 / 3, rel_tol=1e-9)
    assert math.isclose(out["composite_rmse"], math.sqrt(24 / 3), rel_tol=1e-9)


def test_compute_spearman_topN_groups_by_player(fake_eval_df: pd.DataFrame) -> None:
    """Spearman is computed on summed-mean season totals per gsis_id."""
    # Player A: pred sum=12+14=26, actual sum=10+18=28
    # Player B: pred sum=6,        actual sum=4
    # Two players: ranks are perfectly aligned -> Spearman = 1.0
    out = compute_spearman_topN(fake_eval_df)
    assert math.isclose(out, 1.0, rel_tol=1e-9)


def test_compute_calibration_metrics(fake_eval_df: pd.DataFrame) -> None:
    """Calibration is fraction of player-weeks with actual in [p10, p90] / <= p90."""
    # Row 0: actual=10, p10=4, p90=22 -> in [4,22], <= 22
    # Row 1: actual=18, p10=6, p90=24 -> in [6,24], <= 24
    # Row 2: actual=4,  p10=1, p90=14 -> in [1,14], <= 14
    out = compute_calibration_metrics(fake_eval_df)
    assert math.isclose(out["calibration_p10p90"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["calibration_le_p90"], 1.0, rel_tol=1e-9)
```

- [ ] **Step 4.2: Run the test — expect failure.**

Run: `pytest tests/test_backtest/test_metrics.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projections.backtest.metrics'`.

- [ ] **Step 4.3: Implement `metrics.py`.**

Create `src/projections/backtest/metrics.py`:

```python
"""Metric primitives for the walk-forward backtest harness.

Each function consumes an ``eval_df`` (the inner-join of predictions and
actuals; shape documented in tests/test_backtest/conftest.py) plus the
``target_stats`` for the position, and returns a dict[str, float] keyed
by metric name.

The harness composes these per (position, year) and folds the
dicts into the long-form metrics DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.schemas import Stat


def compute_per_stat_metrics(
    eval_df: pd.DataFrame,
    *,
    target_stats: tuple[Stat, ...],
) -> dict[str, float]:
    """Per-stat RMSE / MAE / mean_pred against suffixed columns.

    Expects columns ``{stat}_pred`` and ``{stat}_actual`` for each stat.
    Returns three keys per stat: ``{stat}_rmse``, ``{stat}_mae``,
    ``{stat}_mean_pred``.
    """
    out: dict[str, float] = {}
    for stat in target_stats:
        pred_col = f"{stat.value}_pred"
        actual_col = f"{stat.value}_actual"
        diffs = eval_df[pred_col] - eval_df[actual_col]
        out[f"{stat.value}_rmse"] = float(np.sqrt((diffs**2).mean()))
        out[f"{stat.value}_mae"] = float(diffs.abs().mean())
        out[f"{stat.value}_mean_pred"] = float(eval_df[pred_col].mean())
    return out


def compute_composite_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """RMSE / MAE on the composite mean (PPR points) prediction.

    Expects columns ``mean`` (model's composite mean) and ``actual_ppr``
    (realized PPR points). Returns ``composite_rmse`` and ``composite_mae``.
    """
    diffs = eval_df["mean"] - eval_df["actual_ppr"]
    return {
        "composite_rmse": float(np.sqrt((diffs**2).mean())),
        "composite_mae": float(diffs.abs().mean()),
    }


def compute_spearman_topN(eval_df: pd.DataFrame) -> float:
    """Spearman correlation on summed-mean season totals across players.

    Expects columns ``gsis_id``, ``mean``, ``actual_ppr``. Returns the
    Spearman rho across all players in the (position, year). NaN if
    fewer than two distinct players are present (rare but possible on
    a synthetic fixture).
    """
    pred_rank = eval_df.groupby("gsis_id")["mean"].sum().rank()
    actual_rank = eval_df.groupby("gsis_id")["actual_ppr"].sum().rank()
    common = pred_rank.index.intersection(actual_rank.index)
    if len(common) < 2:
        return float("nan")
    return float(np.corrcoef(pred_rank.loc[common], actual_rank.loc[common])[0, 1])


def compute_calibration_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """Calibration coverage at the weekly level.

    Expects columns ``p10``, ``p90``, ``actual_ppr``. Returns
    ``calibration_p10p90`` (fraction of player-weeks where actual in
    [p10, p90]) and ``calibration_le_p90`` (fraction where actual <= p90).
    """
    in_p10p90 = (
        (eval_df["actual_ppr"] >= eval_df["p10"]) & (eval_df["actual_ppr"] <= eval_df["p90"])
    ).mean()
    le_p90 = (eval_df["actual_ppr"] <= eval_df["p90"]).mean()
    return {
        "calibration_p10p90": float(in_p10p90),
        "calibration_le_p90": float(le_p90),
    }


def compute_all_metrics(
    eval_df: pd.DataFrame,
    *,
    target_stats: tuple[Stat, ...],
) -> dict[str, float]:
    """Convenience wrapper: returns the union of all metric dicts."""
    out: dict[str, float] = {}
    out.update(compute_per_stat_metrics(eval_df, target_stats=target_stats))
    out.update(compute_composite_metrics(eval_df))
    out["spearman_topN"] = compute_spearman_topN(eval_df)
    out.update(compute_calibration_metrics(eval_df))
    return out
```

- [ ] **Step 4.4: Run the test — expect pass.**

Run: `pytest tests/test_backtest/test_metrics.py -v`

Expected: 5 tests pass.

- [ ] **Step 4.5: Commit.**

```bash
git add src/projections/backtest/metrics.py tests/test_backtest/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(backtest): metrics primitives

Plan 3c Phase 3, part 1. Per-stat RMSE/MAE/mean_pred, composite
RMSE/MAE, Spearman top-N on summed season totals, calibration coverage.
Each primitive consumes the eval_df shape produced by harness's inner-
join of predictions and actuals (cf. tests/test_backtest/conftest.py).

compute_all_metrics composes the four primitives into a single
dict[str, float] that the harness folds into long-form rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `backtest/naive.py` — trailing-4 baseline

**Files:**
- Create: `src/projections/backtest/naive.py`
- Create: `tests/test_backtest/test_naive.py`

The naive baseline produces, for each held-out (position, year, week, gsis_id, stat), a point prediction. The composite naive prediction is `score(StatLine(**naive_per_stat), ruleset)`.

Naive logic: trailing-4-game player mean across all games strictly prior to (year, week). Earlier weeks of the held-out year are allowed (no leakage — they're observed). Cold start (fewer than 4 prior games) → fall back to per-position mean across the train window.

- [ ] **Step 5.1: Write the failing naive test.**

Create `tests/test_backtest/test_naive.py`:

```python
"""Unit tests for src/projections/backtest/naive.py."""

from __future__ import annotations

import math

import pandas as pd

from projections.backtest.naive import compute_naive_predictions
from projections.schemas import Position, Stat


def _ws_row(*, gsis_id: str, season: int, week: int, position: str = "WR", **stats: float) -> dict[str, object]:
    base = {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": "MIN",
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": 0.0,
        "rushing_tds": 0,
        "carries": 0,
        "receptions": 0,
        "receiving_yards": 0.0,
        "receiving_tds": 0,
        "targets": 0,
        "receiving_air_yards": 0.0,
        "fumbles_lost": 0,
    }
    base.update(stats)
    return base


def test_naive_trailing_4_uses_player_history_within_holdout_year() -> None:
    """A player with 4 prior weeks of receptions in the held-out year gets
    naive_pred = mean of those 4 weeks (no need to fall back)."""
    train = pd.DataFrame([_ws_row(gsis_id="00-A", season=2023, week=w, receptions=2) for w in range(1, 18)])
    holdout = pd.DataFrame(
        [
            _ws_row(gsis_id="00-A", season=2024, week=1, receptions=4),
            _ws_row(gsis_id="00-A", season=2024, week=2, receptions=6),
            _ws_row(gsis_id="00-A", season=2024, week=3, receptions=8),
            _ws_row(gsis_id="00-A", season=2024, week=4, receptions=10),
            _ws_row(gsis_id="00-A", season=2024, week=5, receptions=99),
        ]
    )

    out = compute_naive_predictions(
        train_actuals=train,
        holdout_actuals=holdout,
        position=Position.WR,
        target_stats=(Stat.RECEPTIONS,),
        held_out_year=2024,
    )

    # Week 5 prediction uses weeks 1-4 of 2024 -> mean(4, 6, 8, 10) = 7.0
    week5 = out[(out["gsis_id"] == "00-A") & (out["week"] == 5)]
    assert math.isclose(float(week5["receptions"].iloc[0]), 7.0, rel_tol=1e-9)


def test_naive_cold_start_falls_back_to_position_mean() -> None:
    """A player with no prior games gets the per-position mean from the
    train window (NEVER the held-out year)."""
    train = pd.DataFrame(
        [_ws_row(gsis_id="00-X", season=2023, week=w, receptions=5) for w in range(1, 5)]
    )
    holdout = pd.DataFrame([_ws_row(gsis_id="00-NEW", season=2024, week=1, receptions=99)])

    out = compute_naive_predictions(
        train_actuals=train,
        holdout_actuals=holdout,
        position=Position.WR,
        target_stats=(Stat.RECEPTIONS,),
        held_out_year=2024,
    )
    new_player = out[out["gsis_id"] == "00-NEW"]
    assert math.isclose(float(new_player["receptions"].iloc[0]), 5.0, rel_tol=1e-9)


def test_naive_uses_pre_holdout_history_when_available() -> None:
    """Week 1 of the held-out year falls back to the player's prior-season
    games (not all the way to per-position mean) if 4+ prior games exist."""
    train = pd.DataFrame(
        [_ws_row(gsis_id="00-A", season=2023, week=w, receptions=3) for w in (14, 15, 16, 17)]
    )
    holdout = pd.DataFrame([_ws_row(gsis_id="00-A", season=2024, week=1, receptions=99)])

    out = compute_naive_predictions(
        train_actuals=train,
        holdout_actuals=holdout,
        position=Position.WR,
        target_stats=(Stat.RECEPTIONS,),
        held_out_year=2024,
    )
    assert math.isclose(float(out["receptions"].iloc[0]), 3.0, rel_tol=1e-9)
```

- [ ] **Step 5.2: Run the test — expect failure.**

Run: `pytest tests/test_backtest/test_naive.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projections.backtest.naive'`.

- [ ] **Step 5.3: Implement `naive.py`.**

Create `src/projections/backtest/naive.py`:

```python
"""Naive baseline for the walk-forward backtest harness.

Per-player trailing-4-game stat mean, with cold-start fallback to per-
position mean across the train window. Used for informational comparison;
never gated.

Plan 3c spec section 3 (naive baseline definition).
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Position, Stat

_TRAILING_N: int = 4


def _per_position_means(
    train_actuals: pd.DataFrame,
    *,
    position: Position,
    target_stats: tuple[Stat, ...],
) -> dict[str, float]:
    """Per-stat mean across the train window for the given position.
    Cold-start fallback when a player has fewer than _TRAILING_N prior games.
    """
    pos_rows = train_actuals[train_actuals["position"] == position.value]
    return {stat.value: float(pos_rows[stat.value].mean()) for stat in target_stats}


def compute_naive_predictions(
    *,
    train_actuals: pd.DataFrame,
    holdout_actuals: pd.DataFrame,
    position: Position,
    target_stats: tuple[Stat, ...],
    held_out_year: int,
) -> pd.DataFrame:
    """For each (gsis_id, week) in holdout_actuals, produce a per-stat
    naive prediction equal to the player's trailing-4-game mean of that
    stat across all games strictly prior to (held_out_year, week).

    Earlier weeks of held_out_year are allowed in the trailing window
    (no leakage — they're already observed at the simulated time of
    prediction). Cold start (< 4 prior games) falls back to per-position
    mean across train_actuals (held_out_year excluded).

    Returns:
        DataFrame with columns ``gsis_id``, ``season``, ``week``, plus
        one float column per target stat. Same row count as holdout_actuals
        (filtered to the position).
    """
    holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
    cold_means = _per_position_means(
        train_actuals, position=position, target_stats=target_stats
    )

    # Combine train + holdout-prior for the trailing window. We re-filter
    # per (gsis_id, week) below.
    combined = pd.concat(
        [
            train_actuals[train_actuals["position"] == position.value],
            holdout_pos,
        ],
        ignore_index=True,
    )
    combined = combined.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _idx, hold_row in holdout_pos.iterrows():
        gsis_id = hold_row["gsis_id"]
        season = int(hold_row["season"])
        week = int(hold_row["week"])

        # Strictly-prior mask: same player, and (season < held_out_year)
        # OR (season == held_out_year AND week < target week).
        prior = combined[
            (combined["gsis_id"] == gsis_id)
            & (
                (combined["season"] < held_out_year)
                | ((combined["season"] == held_out_year) & (combined["week"] < week))
            )
        ]
        prior = prior.tail(_TRAILING_N)

        out_row: dict[str, object] = {
            "gsis_id": gsis_id,
            "season": season,
            "week": week,
        }
        if len(prior) >= _TRAILING_N:
            for stat in target_stats:
                out_row[stat.value] = float(prior[stat.value].mean())
        else:
            for stat in target_stats:
                out_row[stat.value] = cold_means[stat.value]
        rows.append(out_row)

    return pd.DataFrame(rows)
```

- [ ] **Step 5.4: Run the test — expect pass.**

Run: `pytest tests/test_backtest/test_naive.py -v`

Expected: 3 tests pass.

- [ ] **Step 5.5: Commit.**

```bash
git add src/projections/backtest/naive.py tests/test_backtest/test_naive.py
git commit -m "$(cat <<'EOF'
feat(backtest): naive baseline — trailing-4 with per-position cold start

Plan 3c Phase 3, part 2. Spec section 3. Per-(gsis_id, week) naive
prediction is the player's trailing-4-game mean across games strictly
prior in time. Earlier weeks of the held-out year are allowed in the
trailing window (already observed; no leakage). Cold start (< 4 prior
games) falls back to per-position mean from the train window only.

Used for informational comparison alongside model metrics. Not gated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Wire metrics + naive into `run_backtest`

**Files:**
- Modify: `src/projections/backtest/harness.py`
- Modify: `tests/test_backtest/test_harness.py` (add integration test)
- Modify: `tests/test_backtest/conftest.py` (add a fixture that builds a tiny on-disk feature cache + raw data root)

This task replaces the Phase 2 stub with real per-cell training + scoring + metric computation. The integration test exercises a one-position, one-year run end-to-end against synthetic data on disk.

- [ ] **Step 6.1: Add the on-disk fixture.**

Edit `tests/test_backtest/conftest.py`. Append to the file (after `fake_eval_df`):

```python
from pathlib import Path

from projections.schemas import Position
from projections.store import write_partition


@pytest.fixture
def synthetic_backtest_layout(
    tmp_path: Path,
    baseline_features_wr: pd.DataFrame,
    baseline_weekly_stats_wr: pd.DataFrame,
) -> dict[str, Path]:
    """Stand up a tiny data/raw + data/features layout under tmp_path so
    run_backtest can be exercised against synthetic data with no network.

    The fixtures from tests/conftest.py cover multiple seasons. If the
    integration test fails because they don't cover enough seasons for
    a (train_start=2018, held_out_year=2024) call, the executor should
    either narrow the test's call to a smaller range or extend the
    fixtures with additional rows (acceptable refactor).

    Returns paths suitable for run_backtest:
        {"raw_root": ..., "features_root": ...}
    """
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    features_root = data_root / "features"

    # Reuse the per-position fixtures from tests/conftest.py — they already
    # cover multiple seasons. Group by season + write one parquet partition
    # per (table, season[, week]).
    feats_by_season = baseline_features_wr.groupby("season")
    for season, sf in feats_by_season:
        for week, wf in sf.groupby("week"):
            write_partition(
                features_root, "wr", wf.reset_index(drop=True),
                season=int(season), week=int(week),
            )

    ws_by_season = baseline_weekly_stats_wr.groupby("season")
    for season, sf in ws_by_season:
        write_partition(
            raw_root, "weekly_stats", sf.reset_index(drop=True),
            season=int(season), week=None,
        )

    return {"raw_root": raw_root, "features_root": features_root}
```

(Note: the `baseline_features_wr` / `baseline_weekly_stats_wr` fixtures live in `tests/conftest.py` — pytest's hierarchical fixture resolution makes them available here.)

- [ ] **Step 6.2: Write the failing integration test.**

Edit `tests/test_backtest/test_harness.py`. Append:

```python
from projections.schemas import Position


def test_run_backtest_populates_metrics_for_one_cell(
    synthetic_backtest_layout: dict,
) -> None:
    """End-to-end: train on prior seasons of synthetic features, predict the
    held-out year, score, and emit long-form metric rows."""
    out = run_backtest(
        held_out_years=[2024],
        positions=[Position.WR],
        train_start=2018,
        features_root=synthetic_backtest_layout["features_root"],
        raw_root=synthetic_backtest_layout["raw_root"],
    )
    assert not out.metrics.empty
    assert set(out.metrics.columns) == {"position", "year", "metric", "value"}
    # Should produce at least one composite_rmse + one spearman_topN row.
    metric_names = set(out.metrics["metric"].unique())
    assert "composite_rmse" in metric_names
    assert "spearman_topN" in metric_names
    # naive_metrics has the same shape and is non-empty.
    assert not out.naive_metrics.empty
```

- [ ] **Step 6.3: Run the test — expect failure.**

Run: `pytest tests/test_backtest/test_harness.py -v`

Expected: FAIL — the test in step 3.1 still passes; the new test fails because `run_backtest` is still a stub returning empty metrics.

- [ ] **Step 6.4: Implement the per-cell loop in `run_backtest`.**

Edit `src/projections/backtest/harness.py`. Replace the body of `run_backtest` with the real driver:

```python
"""Walk-forward backtest driver.

For each (position, year) in the cartesian product, train Model A on
cached features for [train_start, year-1], predict every week of `year`
from cached features, score against actuals from data/raw/weekly_stats,
and return a BacktestRun with model + naive metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from projections.backtest.metrics import compute_all_metrics
from projections.backtest.naive import compute_naive_predictions
from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.scoring import score
from projections.scoring.score import StatLine
from projections.schemas import Position, Ruleset, Stat
from projections.store import read_partition

_METRICS_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """(docstring unchanged from Phase 2)"""

    timestamp: pd.Timestamp
    metrics: pd.DataFrame
    naive_metrics: pd.DataFrame
    per_row_results: pd.DataFrame


def _realized_ppr_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.Series:
    """Compute realized PPR points per row of weekly_stats. Mirrors
    scripts/sanity_check_baseline.py's helper of the same name."""
    points: list[float] = []
    for _idx, row in weekly_stats.iterrows():
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
    return pd.Series(points, index=weekly_stats.index, name="actual_ppr")


def _build_eval_df(
    *,
    predictions: pd.DataFrame,
    per_stat_pred_means: pd.DataFrame,
    held_out_pos: pd.DataFrame,
    target_stats: tuple[Stat, ...],
) -> pd.DataFrame:
    """Inner-join model preds + per-stat predicted means + actuals on
    (gsis_id, season, week). Result has {stat}_pred / {stat}_actual
    columns, mean / p10 / p90, and actual_ppr."""
    keep = ["gsis_id", "season", "week", "actual_ppr"] + [s.value for s in target_stats]
    pred_with_per_stat = predictions.merge(
        per_stat_pred_means, on=["gsis_id", "season", "week"], how="left"
    )
    return pred_with_per_stat.merge(
        held_out_pos[keep],
        on=["gsis_id", "season", "week"],
        how="inner",
        suffixes=("_pred", "_actual"),
    )


def _naive_metrics_for_cell(
    *,
    train_actuals: pd.DataFrame,
    holdout_actuals: pd.DataFrame,
    position: Position,
    target_stats: tuple[Stat, ...],
    held_out_year: int,
    ruleset: Ruleset,
) -> dict[str, float]:
    """Compute the naive per-stat predictions, build an eval_df-shaped
    frame, and run compute_all_metrics on it."""
    naive_per_stat = compute_naive_predictions(
        train_actuals=train_actuals,
        holdout_actuals=holdout_actuals,
        position=position,
        target_stats=target_stats,
        held_out_year=held_out_year,
    )
    holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
    holdout_pos["actual_ppr"] = _realized_ppr_points(holdout_pos, ruleset)

    # Naive composite point prediction: feed naive per-stat into score().
    naive_composite: list[float] = []
    for _idx, row in naive_per_stat.iterrows():
        kwargs = {stat.value: row[stat.value] for stat in target_stats}
        # Coerce ints for integer stats; score expects them.
        kwargs_clean: dict[str, float | int] = {}
        for k, v in kwargs.items():
            if k in {"passing_tds", "interceptions", "rushing_tds", "receiving_tds",
                     "receptions", "fumbles_lost"}:
                kwargs_clean[k] = int(round(float(v)))
            else:
                kwargs_clean[k] = float(v)
        line = StatLine(**kwargs_clean)  # type: ignore[arg-type]
        naive_composite.append(score(line, ruleset))
    naive_per_stat = naive_per_stat.copy()
    naive_per_stat["mean"] = naive_composite
    # p10/p90 are not meaningful for a point baseline; populate with mean
    # so calibration metrics return 1.0 (informational only — never gated).
    naive_per_stat["p10"] = naive_per_stat["mean"]
    naive_per_stat["p90"] = naive_per_stat["mean"]

    eval_df = _build_eval_df(
        predictions=naive_per_stat[["gsis_id", "season", "week", "mean", "p10", "p90"]],
        per_stat_pred_means=naive_per_stat.drop(columns=["mean", "p10", "p90"]),
        held_out_pos=holdout_pos,
        target_stats=target_stats,
    )
    return compute_all_metrics(eval_df, target_stats=target_stats)


def _model_metrics_for_cell(
    *,
    train_features: pd.DataFrame,
    train_actuals: pd.DataFrame,
    predict_features: pd.DataFrame,
    holdout_actuals: pd.DataFrame,
    position: Position,
    ruleset: Ruleset,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train BaselineModel on train_features+train_actuals, predict each
    week of predict_features, score against holdout_actuals. Returns
    (metrics_dict, eval_df) — eval_df is captured for per_row_results."""
    dispatch = POSITION_DISPATCH[position]
    model = dispatch.factory()
    model.fit(train_features, train_actuals)

    predictions = model.predict_distribution(predict_features, ruleset=ruleset)
    stat_dists_per_row = model._build_stat_distributions(predict_features)
    per_stat_pred_means = pd.DataFrame(
        {
            stat.value: [d[stat].mean() for d in stat_dists_per_row]
            for stat in model.target_stats
        }
    )
    per_stat_pred_means["gsis_id"] = predict_features["gsis_id"].values
    per_stat_pred_means["season"] = predict_features["season"].astype(int).values
    per_stat_pred_means["week"] = predict_features["week"].astype(int).values

    holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
    holdout_pos["actual_ppr"] = _realized_ppr_points(holdout_pos, ruleset)

    eval_df = _build_eval_df(
        predictions=predictions,
        per_stat_pred_means=per_stat_pred_means,
        held_out_pos=holdout_pos,
        target_stats=tuple(model.target_stats),
    )
    metrics = compute_all_metrics(eval_df, target_stats=tuple(model.target_stats))
    return metrics, eval_df


def run_backtest(
    *,
    held_out_years: Iterable[int] = (2021, 2022, 2023, 2024),
    positions: Iterable[Position] | None = None,
    train_start: int = 2018,
    features_root: Path = Path("data/features"),
    raw_root: Path = Path("data/raw"),
    ruleset: Ruleset | None = None,
) -> BacktestRun:
    """Walk-forward backtest. Spec section 2.3."""
    if ruleset is None:
        ruleset = Ruleset.espn_ppr()
    if positions is None:
        positions = (Position.QB, Position.RB, Position.TE, Position.WR)

    timestamp = pd.Timestamp(datetime.now(UTC))
    positions_list = list(positions)
    years_list = list(held_out_years)

    metrics_rows: list[dict[str, object]] = []
    naive_rows: list[dict[str, object]] = []
    per_row_frames: list[pd.DataFrame] = []

    for position in positions_list:
        for year in years_list:
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
            holdout_actuals = read_partition(raw_root, "weekly_stats", season=year)

            model_metrics, eval_df = _model_metrics_for_cell(
                train_features=train_features,
                train_actuals=train_actuals,
                predict_features=predict_features,
                holdout_actuals=holdout_actuals,
                position=position,
                ruleset=ruleset,
            )
            for metric_name, value in model_metrics.items():
                metrics_rows.append(
                    {"position": position.value, "year": year,
                     "metric": metric_name, "value": float(value)}
                )

            naive_metrics = _naive_metrics_for_cell(
                train_actuals=train_actuals,
                holdout_actuals=holdout_actuals,
                position=position,
                target_stats=tuple(POSITION_DISPATCH[position].factory().target_stats),
                held_out_year=year,
                ruleset=ruleset,
            )
            for metric_name, value in naive_metrics.items():
                naive_rows.append(
                    {"position": position.value, "year": year,
                     "metric": metric_name, "value": float(value)}
                )

            eval_df = eval_df.assign(position=position.value)
            per_row_frames.append(eval_df)

    metrics_df = pd.DataFrame(metrics_rows, columns=list(_METRICS_COLUMNS))
    naive_metrics_df = pd.DataFrame(naive_rows, columns=list(_METRICS_COLUMNS))
    per_row_results = (
        pd.concat(per_row_frames, ignore_index=True) if per_row_frames else pd.DataFrame()
    )
    return BacktestRun(
        timestamp=timestamp,
        metrics=metrics_df,
        naive_metrics=naive_metrics_df,
        per_row_results=per_row_results,
    )
```

Note: `POSITION_DISPATCH` is imported once at module scope above — there's no package cycle since `backtest/` does not import from `models/__init__.py` transitively.

- [ ] **Step 6.5: Run all backtest tests — expect pass.**

Run: `pytest tests/test_backtest/ -v`

Expected: all tests pass (the structural test from Task 3 + the metric/naive tests from Tasks 4-5 + the integration test added in this task).

- [ ] **Step 6.6: Quality gate (incremental).**

Run:
```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations. Resolve any drift before committing. Common gotchas: ensure the local `from projections.models import POSITION_DISPATCH` is hoisted to module scope; ruff will flag the unused `Mapping` import in `metrics.py` if you left it in.

- [ ] **Step 6.7: Commit.**

```bash
git add src/projections/backtest/harness.py tests/test_backtest/conftest.py tests/test_backtest/test_harness.py
git commit -m "$(cat <<'EOF'
feat(backtest): wire metrics + naive baseline into run_backtest

Plan 3c Phase 3, part 3. Replaces the Phase 2 stub with the real per-
(position, year) loop:

  1. Read cached training features for [train_start, year-1].
  2. Read training actuals (weekly_stats partitions) for those seasons.
  3. Read prediction features for `year`.
  4. Read held-out actuals for `year`.
  5. Construct + fit the position's BaselineModel via POSITION_DISPATCH.
  6. Predict; build per-stat predicted means via _build_stat_distributions.
  7. Inner-join predictions + actuals into eval_df with _pred/_actual
     suffixed columns + actual_ppr.
  8. Run compute_all_metrics; emit long-form rows.
  9. Compute naive baseline metrics on the same cell; emit long-form rows.

Integration test exercises a single (WR, 2024) cell against the
existing tests/conftest.py synthetic fixtures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Snapshot read/write/diff + tolerance config

### Task 7: `backtest/snapshot.py` read/write

**Files:**
- Create: `src/projections/backtest/snapshot.py`
- Create: `tests/test_backtest/test_snapshot.py`
- Modify: `src/projections/backtest/__init__.py` (re-export the snapshot helpers)

This task lands the file IO + sort logic. The diff/tolerance logic comes in Task 8.

- [ ] **Step 7.1: Write the failing IO test.**

Create `tests/test_backtest/test_snapshot.py` with the IO-only tests (diff tests come in Task 8):

```python
"""Unit tests for src/projections/backtest/snapshot.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projections.backtest.snapshot import read_snapshot, write_snapshot


def test_write_then_read_roundtrips_a_metrics_df(tmp_path: Path) -> None:
    """write_snapshot serializes a long-form metrics DataFrame; read_snapshot
    returns the same columns + values, sorted by (metric, position, year)."""
    df = pd.DataFrame(
        [
            {"position": "WR", "year": 2024, "metric": "composite_rmse", "value": 6.78},
            {"position": "QB", "year": 2021, "metric": "spearman_topN", "value": 0.928},
        ]
    )
    path = tmp_path / "snap.json"
    write_snapshot(df, path)

    out = read_snapshot(path)
    assert set(out.columns) == {"position", "year", "metric", "value"}
    assert len(out) == 2
    # Sorted by (metric, position, year): composite_rmse-WR-2024 then spearman_topN-QB-2021
    assert out["metric"].tolist() == ["composite_rmse", "spearman_topN"]


def test_write_snapshot_emits_human_readable_json(tmp_path: Path) -> None:
    """The on-disk JSON is a list of objects (not pandas-serialized) with
    a 2-space indent so PR diffs stay clean."""
    df = pd.DataFrame(
        [{"position": "WR", "year": 2024, "metric": "composite_rmse", "value": 6.78}]
    )
    path = tmp_path / "snap.json"
    write_snapshot(df, path)

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert parsed[0] == {
        "position": "WR",
        "year": 2024,
        "metric": "composite_rmse",
        "value": 6.78,
    }
    # Indented for readability.
    assert "\n  " in raw
```

- [ ] **Step 7.2: Run — expect failure.**

Run: `pytest tests/test_backtest/test_snapshot.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projections.backtest.snapshot'`.

- [ ] **Step 7.3: Implement IO functions.**

Create `src/projections/backtest/snapshot.py`:

```python
"""Snapshot file IO + diff for the walk-forward gate.

Plan 3c Phase 4. Snapshot is a JSON list of
{"position", "year", "metric", "value"} entries, sorted lexicographically
by (metric, position, year) so PR diffs stay clean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


_SCHEMA_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


def write_snapshot(metrics: pd.DataFrame, path: Path) -> None:
    """Serialize a long-form metrics DataFrame to JSON, sorted by
    (metric, position, year)."""
    if set(metrics.columns) != set(_SCHEMA_COLUMNS):
        raise ValueError(
            f"metrics must have columns {_SCHEMA_COLUMNS}, got {tuple(metrics.columns)}"
        )
    sorted_df = metrics.sort_values(["metric", "position", "year"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _idx, row in sorted_df.iterrows():
        rows.append(
            {
                "position": str(row["position"]),
                "year": int(row["year"]),
                "metric": str(row["metric"]),
                "value": float(row["value"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def read_snapshot(path: Path) -> pd.DataFrame:
    """Load a snapshot JSON file into a long-form metrics DataFrame."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(raw, columns=list(_SCHEMA_COLUMNS))
```

Edit `src/projections/backtest/__init__.py` to re-export:

```python
"""Walk-forward backtest harness + snapshot-diff gating.

Plan 3c. Public surface for the harness, the metric primitives, the naive
baseline, and the snapshot-diff machinery.
"""

from __future__ import annotations

from projections.backtest.harness import BacktestRun, run_backtest
from projections.backtest.snapshot import read_snapshot, write_snapshot

__all__ = [
    "BacktestRun",
    "read_snapshot",
    "run_backtest",
    "write_snapshot",
]
```

- [ ] **Step 7.4: Run — expect pass.**

Run: `pytest tests/test_backtest/test_snapshot.py -v`

Expected: 2 tests pass.

- [ ] **Step 7.5: Commit.**

```bash
git add src/projections/backtest/snapshot.py src/projections/backtest/__init__.py tests/test_backtest/test_snapshot.py
git commit -m "$(cat <<'EOF'
feat(backtest): snapshot IO — read/write JSON snapshot files

Plan 3c Phase 4, part 1. write_snapshot serializes a long-form metrics
DataFrame to a JSON list sorted by (metric, position, year). read_snapshot
reverses it. The sort order is the spec's "PR diffs stay clean" guarantee.

Diff + tolerance application lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `snapshot.diff_snapshot` + tolerance config

**Files:**
- Modify: `src/projections/backtest/snapshot.py` (add `GateResult`, `diff_snapshot`, tolerance kinds)
- Modify: `src/projections/backtest/__init__.py` (re-export `GateResult`, `diff_snapshot`)
- Create: `tests/backtest/__init__.py` (empty marker — distinct from `tests/test_backtest/`)
- Create: `tests/backtest/tolerances.json` (committed config)
- Modify: `tests/test_backtest/test_snapshot.py` (diff tests)

The diff function consumes a current run's metrics + a baseline snapshot + a tolerance config, and returns a `GateResult` with pass/fail + per-row regression details. Tolerance application is direction-aware per the spec's §2.4 table.

- [ ] **Step 8.1: Write the failing diff tests.**

Edit `tests/test_backtest/test_snapshot.py`. Append:

```python
import pytest

from projections.backtest.snapshot import GateResult, diff_snapshot


def _baseline_row(metric: str, value: float) -> dict:
    return {"position": "WR", "year": 2024, "metric": metric, "value": value}


_DEFAULT_TOLS = {
    "rmse_relative": 0.05,
    "mae_relative": 0.05,
    "spearman_absolute": 0.02,
    "calibration_absolute": 0.03,
    "mean_pred_relative": 0.10,
}


def test_diff_passes_when_metrics_within_tolerance() -> None:
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.2)])  # +3.3%, under 5%

    out = diff_snapshot(
        current=current,
        baseline=baseline,
        defaults=_DEFAULT_TOLS,
        overrides=[],
    )
    assert isinstance(out, GateResult)
    assert out.passed is True
    assert out.regressions == []


def test_diff_fails_on_rmse_regression_above_tolerance() -> None:
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.5)])  # +8.3% > 5%

    out = diff_snapshot(
        current=current, baseline=baseline,
        defaults=_DEFAULT_TOLS, overrides=[],
    )
    assert out.passed is False
    assert len(out.regressions) == 1
    assert out.regressions[0].metric == "composite_rmse"
    assert out.regressions[0].direction == "worse"


def test_diff_passes_on_rmse_improvement() -> None:
    """Direction-aware: RMSE going DOWN is improvement, never a regression."""
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 4.0)])  # 33% better

    out = diff_snapshot(
        current=current, baseline=baseline,
        defaults=_DEFAULT_TOLS, overrides=[],
    )
    assert out.passed is True


def test_diff_fails_on_spearman_drop_below_absolute_tolerance() -> None:
    baseline = pd.DataFrame([_baseline_row("spearman_topN", 0.97)])
    current = pd.DataFrame([_baseline_row("spearman_topN", 0.94)])  # -0.03 > 0.02

    out = diff_snapshot(
        current=current, baseline=baseline,
        defaults=_DEFAULT_TOLS, overrides=[],
    )
    assert out.passed is False


def test_diff_passes_on_spearman_improvement() -> None:
    """Spearman going UP is improvement."""
    baseline = pd.DataFrame([_baseline_row("spearman_topN", 0.94)])
    current = pd.DataFrame([_baseline_row("spearman_topN", 0.99)])

    out = diff_snapshot(
        current=current, baseline=baseline,
        defaults=_DEFAULT_TOLS, overrides=[],
    )
    assert out.passed is True


def test_diff_calibration_fails_on_drift_in_either_direction() -> None:
    """calibration_p10p90's target is 0.80; both 0.75 and 0.85 would be valid
    if baseline were at 0.80, but since baseline is the snapshot value we
    treat any move of >tolerance from snapshot as regression."""
    baseline = pd.DataFrame([_baseline_row("calibration_p10p90", 0.80)])
    current = pd.DataFrame([_baseline_row("calibration_p10p90", 0.85)])  # +0.05 > 0.03

    out = diff_snapshot(
        current=current, baseline=baseline,
        defaults=_DEFAULT_TOLS, overrides=[],
    )
    assert out.passed is False


def test_diff_overrides_loosen_per_row_tolerance() -> None:
    """An override for the (position, year, metric) cell uses its tolerance
    instead of the default."""
    baseline = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.5)])  # +8.3%

    out = diff_snapshot(
        current=current, baseline=baseline, defaults=_DEFAULT_TOLS,
        overrides=[
            {
                "position": "WR", "year": 2024, "metric": "composite_rmse",
                "tolerance_kind": "rmse_relative", "tolerance_value": 0.10,
                "rationale": "fixture noise",
            }
        ],
    )
    assert out.passed is True


def test_diff_unknown_metric_kind_fails_closed() -> None:
    """A metric in the snapshot whose name doesn't match any known suffix
    raises so we never silently let a metric through ungated."""
    baseline = pd.DataFrame([_baseline_row("totally_made_up_metric", 1.0)])
    current = pd.DataFrame([_baseline_row("totally_made_up_metric", 1.5)])

    with pytest.raises(ValueError, match="unknown tolerance kind"):
        diff_snapshot(
            current=current, baseline=baseline,
            defaults=_DEFAULT_TOLS, overrides=[],
        )


def test_diff_missing_baseline_row_fails() -> None:
    """A current-run row with no baseline row to compare against fails the
    gate (the snapshot must be re-generated to add new metrics
    intentionally)."""
    baseline = pd.DataFrame(columns=["position", "year", "metric", "value"])
    current = pd.DataFrame([_baseline_row("composite_rmse", 6.0)])

    out = diff_snapshot(
        current=current, baseline=baseline,
        defaults=_DEFAULT_TOLS, overrides=[],
    )
    assert out.passed is False
    assert any("missing from baseline" in r.message for r in out.regressions)
```

- [ ] **Step 8.2: Run — expect failure.**

Run: `pytest tests/test_backtest/test_snapshot.py -v`

Expected: the IO tests still pass; the new diff tests fail with `ImportError: cannot import name 'GateResult'`.

- [ ] **Step 8.3: Implement diff + tolerance kinds.**

Edit `src/projections/backtest/snapshot.py`. Replace the file's body with the full version (IO + diff together):

```python
"""Snapshot file IO + diff for the walk-forward gate.

Plan 3c Phase 4. Snapshot is a JSON list of
{"position", "year", "metric", "value"} entries, sorted lexicographically
by (metric, position, year) so PR diffs stay clean.

Tolerance application is direction-aware. Spec section 2.4 lists the
mapping; this module owns the suffix-based metric -> tolerance-kind
registry and the comparison logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


_SCHEMA_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


# Suffix-based metric -> tolerance-kind mapping. Order matters: longer
# suffixes are tried first so "_mean_pred" matches before a hypothetical
# "_pred". The keys are the tolerance kind names; the values are tuples of
# substrings that, if present in the metric name, route the metric to
# this kind.
_METRIC_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mean_pred_relative", ("_mean_pred",)),
    ("rmse_relative", ("_rmse",)),
    ("mae_relative", ("_mae",)),
    ("spearman_absolute", ("spearman_",)),
    ("calibration_absolute", ("calibration_",)),
)


def _classify_metric(metric: str) -> str:
    """Return the tolerance-kind name for a metric. Raises if no rule
    matches — fail-closed for unknown metric names."""
    for kind, needles in _METRIC_KIND_RULES:
        for n in needles:
            if n in metric:
                return kind
    raise ValueError(
        f"unknown tolerance kind for metric {metric!r}; add a rule to "
        f"_METRIC_KIND_RULES in projections/backtest/snapshot.py"
    )


@dataclass(frozen=True, slots=True)
class Regression:
    """A single (position, year, metric) cell that failed the gate."""

    position: str
    year: int
    metric: str
    baseline_value: float
    current_value: float
    direction: str  # "worse", "better", "missing", "unknown"
    tolerance_kind: str
    tolerance_value: float
    message: str


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of comparing a current run's metrics against a baseline snapshot."""

    passed: bool
    regressions: list[Regression] = field(default_factory=list)


def write_snapshot(metrics: pd.DataFrame, path: Path) -> None:
    """Serialize a long-form metrics DataFrame to JSON, sorted by
    (metric, position, year)."""
    if set(metrics.columns) != set(_SCHEMA_COLUMNS):
        raise ValueError(
            f"metrics must have columns {_SCHEMA_COLUMNS}, got {tuple(metrics.columns)}"
        )
    sorted_df = metrics.sort_values(["metric", "position", "year"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _idx, row in sorted_df.iterrows():
        rows.append(
            {
                "position": str(row["position"]),
                "year": int(row["year"]),
                "metric": str(row["metric"]),
                "value": float(row["value"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def read_snapshot(path: Path) -> pd.DataFrame:
    """Load a snapshot JSON file into a long-form metrics DataFrame."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(raw, columns=list(_SCHEMA_COLUMNS))


def _check_one(
    *,
    position: str,
    year: int,
    metric: str,
    baseline_value: float,
    current_value: float,
    tolerance_kind: str,
    tolerance_value: float,
) -> Regression | None:
    """Apply direction-aware tolerance to a single cell. Returns None on
    pass; a Regression on fail."""
    if tolerance_kind in {"rmse_relative", "mae_relative"}:
        # RMSE/MAE worse = larger.
        if current_value <= baseline_value:
            return None
        rel = (current_value - baseline_value) / max(baseline_value, 1e-12)
        if rel <= tolerance_value:
            return None
        return Regression(
            position=position, year=year, metric=metric,
            baseline_value=baseline_value, current_value=current_value,
            direction="worse", tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{position}/{year}/{metric}: {baseline_value:.4f} -> {current_value:.4f} "
                f"({rel:+.2%} > {tolerance_value:+.2%})"
            ),
        )

    if tolerance_kind == "spearman_absolute":
        # Spearman worse = smaller.
        if current_value >= baseline_value:
            return None
        delta = baseline_value - current_value
        if delta <= tolerance_value:
            return None
        return Regression(
            position=position, year=year, metric=metric,
            baseline_value=baseline_value, current_value=current_value,
            direction="worse", tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{position}/{year}/{metric}: {baseline_value:.4f} -> {current_value:.4f} "
                f"(drop {delta:.4f} > {tolerance_value:.4f})"
            ),
        )

    if tolerance_kind in {"calibration_absolute", "mean_pred_relative"}:
        # Drift in either direction past tolerance is a regression.
        if tolerance_kind == "calibration_absolute":
            delta = abs(current_value - baseline_value)
            if delta <= tolerance_value:
                return None
        else:  # mean_pred_relative
            rel = abs(current_value - baseline_value) / max(abs(baseline_value), 1e-12)
            if rel <= tolerance_value:
                return None
            delta = rel
        return Regression(
            position=position, year=year, metric=metric,
            baseline_value=baseline_value, current_value=current_value,
            direction="worse", tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{position}/{year}/{metric}: {baseline_value:.4f} -> {current_value:.4f} "
                f"(drift {delta:.4f} > {tolerance_value:.4f})"
            ),
        )

    raise ValueError(f"unknown tolerance_kind {tolerance_kind!r}")


def diff_snapshot(
    *,
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    defaults: dict[str, float],
    overrides: list[dict[str, Any]],
) -> GateResult:
    """Compare a current run's metrics against a baseline snapshot.

    For each row in ``current``, look up the matching ``baseline`` row by
    (position, year, metric); apply the override row's tolerance if
    present, otherwise apply the default tolerance for the metric kind.
    Returns a GateResult; ``passed`` is True iff no regressions found.

    A current-run row missing from baseline is itself a regression — the
    snapshot must be regenerated to include it intentionally.
    """
    overrides_index = {
        (o["position"], int(o["year"]), o["metric"]): o for o in overrides
    }
    baseline_index = {
        (str(r["position"]), int(r["year"]), str(r["metric"])): float(r["value"])
        for _, r in baseline.iterrows()
    }

    regressions: list[Regression] = []
    for _, row in current.iterrows():
        position = str(row["position"])
        year = int(row["year"])
        metric = str(row["metric"])
        current_value = float(row["value"])

        key = (position, year, metric)
        if key not in baseline_index:
            regressions.append(
                Regression(
                    position=position, year=year, metric=metric,
                    baseline_value=float("nan"), current_value=current_value,
                    direction="missing", tolerance_kind="<n/a>",
                    tolerance_value=float("nan"),
                    message=f"{position}/{year}/{metric}: missing from baseline",
                )
            )
            continue

        baseline_value = baseline_index[key]
        if key in overrides_index:
            ov = overrides_index[key]
            tol_kind = str(ov["tolerance_kind"])
            tol_value = float(ov["tolerance_value"])
        else:
            tol_kind = _classify_metric(metric)
            tol_value = defaults[tol_kind]

        reg = _check_one(
            position=position, year=year, metric=metric,
            baseline_value=baseline_value, current_value=current_value,
            tolerance_kind=tol_kind, tolerance_value=tol_value,
        )
        if reg is not None:
            regressions.append(reg)

    return GateResult(passed=not regressions, regressions=regressions)
```

Edit `src/projections/backtest/__init__.py` to re-export the new symbols:

```python
"""Walk-forward backtest harness + snapshot-diff gating.

Plan 3c. Public surface for the harness, the metric primitives, the naive
baseline, and the snapshot-diff machinery.
"""

from __future__ import annotations

from projections.backtest.harness import BacktestRun, run_backtest
from projections.backtest.snapshot import (
    GateResult,
    Regression,
    diff_snapshot,
    read_snapshot,
    write_snapshot,
)

__all__ = [
    "BacktestRun",
    "GateResult",
    "Regression",
    "diff_snapshot",
    "read_snapshot",
    "run_backtest",
    "write_snapshot",
]
```

Create the committed tolerance config at `tests/backtest/tolerances.json`:

```json
{
  "defaults": {
    "rmse_relative": 0.05,
    "mae_relative": 0.05,
    "spearman_absolute": 0.02,
    "calibration_absolute": 0.03,
    "mean_pred_relative": 0.10
  },
  "overrides": []
}
```

Create `tests/backtest/__init__.py` as an empty file.

- [ ] **Step 8.4: Run — expect pass.**

Run: `pytest tests/test_backtest/test_snapshot.py -v`

Expected: all 11 tests pass (2 IO + 9 diff).

- [ ] **Step 8.5: Quality gate (incremental).**

Run:
```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 8.6: Commit.**

```bash
git add src/projections/backtest/snapshot.py src/projections/backtest/__init__.py tests/backtest/__init__.py tests/backtest/tolerances.json tests/test_backtest/test_snapshot.py
git commit -m "$(cat <<'EOF'
feat(backtest): diff_snapshot + direction-aware tolerance application

Plan 3c Phase 4, part 2. Adds:

- GateResult / Regression dataclasses.
- diff_snapshot(current, baseline, defaults, overrides) returning a
  GateResult with per-cell regression details.
- Suffix-based metric -> tolerance-kind registry (_METRIC_KIND_RULES).
  Fails closed: a metric with an unknown suffix raises rather than
  silently routing to a default.
- Direction-aware comparison: RMSE/MAE worse = larger, Spearman worse
  = smaller, calibration drift in either direction beyond tolerance.
- Per-row overrides loosen the default for known-noisy cells.

Also commits the v1 tests/backtest/tolerances.json (empty overrides) and
the tests/backtest/__init__.py marker file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — pytest gate wiring + CLI

### Task 9: Register `backtest` marker + `--run-backtest` flag

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 9.1: Register the marker.**

Edit `pyproject.toml`. Find the `[tool.pytest.ini_options]` block. Update `markers` to include `backtest`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "network: tests that hit the live nfl_data_py API; skipped by default",
    "backtest: tests that run the full walk-forward backtest gate; skipped without --run-backtest",
]
```

- [ ] **Step 9.2: Extend `tests/conftest.py` with the new flag.**

Edit `tests/conftest.py`. Find `pytest_addoption` and add a second `parser.addoption` call:

```python
def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--run-network` and `--run-backtest` so the slow opt-in
    suites only run when explicitly requested."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run @pytest.mark.network tests that hit the live nfl_data_py API.",
    )
    parser.addoption(
        "--run-backtest",
        action="store_true",
        default=False,
        help="Run @pytest.mark.backtest tests (the full walk-forward gate).",
    )
```

Find `pytest_collection_modifyitems` and extend it to skip backtest-marked tests too:

```python
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip opt-in marked tests unless their gate flag is passed."""
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="needs --run-network to hit the live API")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
    if not config.getoption("--run-backtest"):
        skip_backtest = pytest.mark.skip(
            reason="needs --run-backtest to run the full walk-forward gate"
        )
        for item in items:
            if "backtest" in item.keywords:
                item.add_marker(skip_backtest)
```

- [ ] **Step 9.3: Verify the marker is recognized.**

Run: `pytest --markers | grep backtest`

Expected: prints the `backtest` marker description.

- [ ] **Step 9.4: Commit.**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "$(cat <<'EOF'
feat(test): register backtest marker + --run-backtest opt-in flag

Plan 3c Phase 5, part 1. Mirrors the existing network/--run-network
plumbing. Tests with @pytest.mark.backtest are skipped on default
pytest invocations and only run when --run-backtest is passed.

The default-on backtest smoke test (Phase 5 Task 10) is unaffected;
only the full-gate test (Task 11) carries the marker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Default-on smoke test

**Files:**
- Create: `tests/backtest/test_backtest_smoke.py`

The smoke test runs the harness with one (position, year) cell against the populated feature cache. It does NOT diff against the snapshot — it only asserts the harness produced a non-empty result with the right schema. Its purpose is to catch "I broke the harness import path" without paying full runtime.

The smoke test depends on `data/features/wr/season=2024/...` existing, which is true after Phase 6. Until then, the smoke test is skipped via `pytest.importorskip` on the cache directory.

- [ ] **Step 10.1: Create the smoke test.**

Create `tests/backtest/test_backtest_smoke.py`:

```python
"""Default-on smoke for the backtest harness.

Catches "I broke the harness wiring" without paying the full 16-fit
runtime of the gated test. Runs in default `pytest -v`.

Skipped automatically if the feature cache for (WR, 2024) doesn't exist
locally (fresh checkout before Phase 6 of Plan 3c, or before
`scripts/refresh_features.py` has been run).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.backtest import BacktestRun, run_backtest
from projections.schemas import Position

_FEATURES_ROOT = Path("data/features")
_RAW_ROOT = Path("data/raw")


def _cache_present() -> bool:
    return (_FEATURES_ROOT / "wr" / "season=2024").exists() and (
        _RAW_ROOT / "weekly_stats" / "season=2024"
    ).exists()


@pytest.mark.skipif(
    not _cache_present(),
    reason="data/features/wr/season=2024 missing — run scripts/refresh_features.py wr",
)
def test_backtest_smoke_one_cell() -> None:
    """One (position, year) cell. Asserts the harness produces a non-empty
    result with the expected long-form schema."""
    out = run_backtest(
        held_out_years=[2024],
        positions=[Position.WR],
        train_start=2018,
        features_root=_FEATURES_ROOT,
        raw_root=_RAW_ROOT,
    )
    assert isinstance(out, BacktestRun)
    assert not out.metrics.empty
    assert set(out.metrics.columns) == {"position", "year", "metric", "value"}
    # Exactly one (position, year) cell.
    assert sorted(out.metrics["position"].unique().tolist()) == ["WR"]
    assert sorted(out.metrics["year"].unique().tolist()) == [2024]
    # Composite + Spearman + at least one per-stat row are present.
    metric_names = set(out.metrics["metric"].unique())
    assert "composite_rmse" in metric_names
    assert "spearman_topN" in metric_names
```

- [ ] **Step 10.2: Run the smoke (skipped on a fresh checkout — that's expected).**

Run: `pytest tests/backtest/test_backtest_smoke.py -v`

Expected: SKIPPED (with the reason about the missing cache) on a fresh checkout. Once Phase 6 populates the cache, this test will run and pass.

- [ ] **Step 10.3: Commit.**

```bash
git add tests/backtest/test_backtest_smoke.py
git commit -m "$(cat <<'EOF'
feat(test): default-on backtest smoke covering one (WR, 2024) cell

Plan 3c Phase 5, part 2. Runs in default `pytest -v`. Auto-skips if
the local feature cache for (WR, 2024) is missing — true on fresh
checkouts before Phase 6 populates data/features/. Catches "I broke
the harness import path" without paying the full 16-fit runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Opt-in gate test

**Files:**
- Create: `tests/backtest/test_backtest_gate.py`

The gate test runs the full harness, loads the committed snapshot + tolerances, calls `diff_snapshot`, and asserts `passed`. Carries `@pytest.mark.backtest` so it runs only with `--run-backtest`.

The committed snapshot at `tests/backtest/baseline_metrics.json` is created in Phase 6 (Task 13) — until then the test is also gated by file existence to avoid a hard failure during development.

- [ ] **Step 11.1: Create the gate test.**

Create `tests/backtest/test_backtest_gate.py`:

```python
"""Opt-in walk-forward backtest gate.

Plan 3c Phase 5, part 3. Runs only with `pytest -m backtest --run-backtest`.
Loads the committed snapshot + tolerances, runs the full harness, calls
diff_snapshot, and asserts the gate passed.

The committed snapshot file is produced by Phase 6 of Plan 3c. Until
then this test additionally guards on file existence and skips with a
clear message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from projections.backtest import diff_snapshot, read_snapshot, run_backtest

_SNAPSHOT_PATH = Path("tests/backtest/baseline_metrics.json")
_TOLERANCES_PATH = Path("tests/backtest/tolerances.json")


@pytest.mark.backtest
def test_backtest_gate_does_not_regress() -> None:
    """Run the full harness, diff against the committed snapshot, fail on
    any regression beyond per-metric-type tolerance."""
    if not _SNAPSHOT_PATH.exists():
        pytest.skip(
            f"{_SNAPSHOT_PATH} missing — Phase 6 of Plan 3c hasn't produced "
            f"the v1 snapshot yet. Run: python scripts/backtest.py --update-snapshot"
        )

    tolerances = json.loads(_TOLERANCES_PATH.read_text(encoding="utf-8"))
    baseline = read_snapshot(_SNAPSHOT_PATH)

    run = run_backtest()
    result = diff_snapshot(
        current=run.metrics,
        baseline=baseline,
        defaults=tolerances["defaults"],
        overrides=tolerances["overrides"],
    )

    if not result.passed:
        msg_lines = [
            f"Backtest gate failed with {len(result.regressions)} regression(s):",
            *[f"  - {r.message}" for r in result.regressions],
        ]
        pytest.fail("\n".join(msg_lines))
```

- [ ] **Step 11.2: Verify it's gated correctly.**

Run: `pytest tests/backtest/test_backtest_gate.py -v`

Expected: SKIPPED with reason "needs --run-backtest" (because no `--run-backtest` flag was passed).

Run: `pytest tests/backtest/test_backtest_gate.py -v --run-backtest`

Expected: SKIPPED with reason about `tests/backtest/baseline_metrics.json` missing (Phase 6 hasn't run yet).

- [ ] **Step 11.3: Commit.**

```bash
git add tests/backtest/test_backtest_gate.py
git commit -m "$(cat <<'EOF'
feat(test): opt-in backtest gate test (--run-backtest)

Plan 3c Phase 5, part 3. Loads the committed snapshot + tolerances,
runs the full walk-forward harness across (4 positions x 4 years = 16
cells), diffs against the snapshot, fails on any regression beyond
per-metric-type tolerance.

Skipped without `pytest --run-backtest`. Also auto-skipped if the
v1 snapshot file is missing (Phase 6 of Plan 3c produces it).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: `scripts/backtest.py` CLI

**Files:**
- Create: `scripts/backtest.py`

The CLI exposes the harness + snapshot machinery as a script. Three modes:

- `--check` (default): run + diff + exit 0/1 with diagnostic output.
- `--update-snapshot`: run + write `tests/backtest/baseline_metrics.json` + print diff vs prior snapshot if one existed.
- `--report`: run + print model + naive metrics side by side; no gate.

- [ ] **Step 12.1: Create the CLI.**

Create `scripts/backtest.py`:

```python
"""Plan 3c — walk-forward backtest CLI.

Three modes:
  --check (default):   run harness, diff snapshot, exit 0/1.
  --update-snapshot:   run harness, overwrite tests/backtest/baseline_metrics.json.
  --report:            run harness, print model + naive metrics; no gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from projections.backtest import diff_snapshot, read_snapshot, run_backtest, write_snapshot

_SNAPSHOT_PATH = Path("tests/backtest/baseline_metrics.json")
_TOLERANCES_PATH = Path("tests/backtest/tolerances.json")


def _print_metrics_table(label: str, metrics: pd.DataFrame, naive: pd.DataFrame) -> None:
    """Print a per-(position, year) table merging model + naive metrics
    side by side."""
    if metrics.empty:
        print(f"({label}: no metrics)")
        return
    pivot = metrics.pivot_table(
        index=["position", "year"], columns="metric", values="value"
    )
    print(f"\n=== {label} ===")
    print(pivot.to_string(float_format=lambda x: f"{x:8.3f}"))

    if not naive.empty:
        naive_pivot = naive.pivot_table(
            index=["position", "year"], columns="metric", values="value"
        )
        print(f"\n=== {label} — naive baseline (informational) ===")
        # Print only composite + Spearman to keep the report compact.
        compact_cols = [c for c in naive_pivot.columns if c in {
            "composite_rmse", "composite_mae", "spearman_topN",
        }]
        if compact_cols:
            print(naive_pivot[compact_cols].to_string(float_format=lambda x: f"{x:8.3f}"))


def _check(run: object, tolerances: dict[str, object]) -> int:
    """Run the diff against the committed snapshot. Returns POSIX exit code."""
    if not _SNAPSHOT_PATH.exists():
        print(
            f"ERROR: {_SNAPSHOT_PATH} missing. Run with --update-snapshot first.",
            file=sys.stderr,
        )
        return 2

    baseline = read_snapshot(_SNAPSHOT_PATH)
    result = diff_snapshot(
        current=run.metrics,  # type: ignore[attr-defined]
        baseline=baseline,
        defaults=tolerances["defaults"],  # type: ignore[arg-type]
        overrides=tolerances["overrides"],  # type: ignore[arg-type]
    )
    if result.passed:
        print(f"PASS — {len(run.metrics)} metrics within tolerance.")  # type: ignore[attr-defined]
        return 0
    print(f"FAIL — {len(result.regressions)} regression(s):")
    for r in result.regressions:
        print(f"  - {r.message}")
    return 1


def _update(run: object) -> int:
    """Write the current run's metrics as the new snapshot. Print a diff
    against the prior snapshot for human review."""
    if _SNAPSHOT_PATH.exists():
        prior = read_snapshot(_SNAPSHOT_PATH)
        # Quick diff summary.
        print(f"Previous snapshot: {len(prior)} rows.")
        print(f"New snapshot:      {len(run.metrics)} rows.")  # type: ignore[attr-defined]
    write_snapshot(run.metrics, _SNAPSHOT_PATH)  # type: ignore[attr-defined]
    print(f"Wrote {_SNAPSHOT_PATH}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest CLI.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Run + diff + exit 0/1 (default).")
    mode.add_argument("--update-snapshot", action="store_true",
                      help="Run + overwrite the committed snapshot.")
    mode.add_argument("--report", action="store_true",
                      help="Run + print model + naive metrics; no gate.")
    args = parser.parse_args()

    tolerances = json.loads(_TOLERANCES_PATH.read_text(encoding="utf-8"))
    run = run_backtest()

    if args.update_snapshot:
        sys.exit(_update(run))
    if args.report:
        _print_metrics_table("Backtest", run.metrics, run.naive_metrics)
        sys.exit(0)
    # Default: check.
    sys.exit(_check(run, tolerances))


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.2: Verify the CLI parses and reaches the runtime gate.**

Run: `python scripts/backtest.py --report`

Expected: either runs successfully (if the cache is populated) or fails with a clear `FileNotFoundError` from `read_features` (cache missing). Either way, the CLI parses correctly.

- [ ] **Step 12.3: Commit.**

```bash
git add scripts/backtest.py
git commit -m "$(cat <<'EOF'
feat(scripts): backtest.py CLI — --check / --update-snapshot / --report

Plan 3c Phase 5, part 4. Three-mode CLI on top of the harness +
snapshot machinery:

  --check (default): runs harness, diffs against committed snapshot,
    prints PASS/FAIL with regression list, exits 0/1.
  --update-snapshot: runs harness, overwrites
    tests/backtest/baseline_metrics.json; prints row-count summary
    against the prior snapshot for human review.
  --report: runs harness, prints model + naive metrics side by side;
    no gate.

This is the human-facing entry point. The pytest gate test
(test_backtest_gate.py) calls run_backtest + diff_snapshot directly,
not through this CLI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — First end-to-end run + commit snapshot

### Task 13: Populate cache + produce v1 snapshot

This is the load-bearing manual step that closes TODO #4 for real and produces the v1 `baseline_metrics.json`.

- [ ] **Step 13.1: Refresh raw data if needed.**

If `data/raw/` is missing on this machine, run:

```bash
python -c "from projections.ingest.refresh import refresh; from pathlib import Path; refresh(seasons=range(2018, 2025), data_root=Path('data'))"
```

Expected: pulls 7 seasons of weekly_stats / depth_charts / NGS / schedules / id_map / snap_counts. Takes ~5–10 minutes the first time. (This is the same incantation Plan 3a/3b used; TODO #18 captures the missing argparse `main()` entry point.)

- [ ] **Step 13.2: Refresh the feature cache.**

Run: `python scripts/refresh_features.py all --seasons 2018-2024`

Expected: prints per-(position, season) write counts; total ~480–528 partitions (4 positions × 7 seasons × ~17–22 weeks).

Spot-check: `ls data/features/wr/season=2024/` should list `week=01` through `week=22` (or however many weeks the 2024 season has).

- [ ] **Step 13.3: Run the harness in `--report` mode for sanity.**

Run: `python scripts/backtest.py --report`

Expected: prints two tables (model metrics + naive baseline). Sanity check the numbers:
- Composite RMSE per (position, year) should be roughly 5–9 (PPR points).
- Spearman top-N should be 0.85–0.98.
- Calibration `[p10, p90]` should be 0.65–0.80.
- Naive composite RMSE should be **larger** than model composite RMSE by some non-trivial margin (otherwise Model A isn't adding value).

If anything is wildly off (e.g., negative Spearman, calibration ~0.30, composite RMSE > 15), STOP and investigate before producing the snapshot — the issue is in the model/feature/harness, not the snapshot.

Total runtime expected: 30–90 seconds.

- [ ] **Step 13.4: Produce the v1 snapshot.**

Run: `python scripts/backtest.py --update-snapshot`

Expected: writes `tests/backtest/baseline_metrics.json` with ~352 rows. Prints row-count summary.

Open the file. Verify:
- It's sorted lexicographically by `(metric, position, year)`.
- All 4 positions × 4 years × ~22 metrics = ~352 rows are present.

- [ ] **Step 13.5: Run the opt-in gate against the fresh snapshot.**

Run: `pytest tests/backtest/test_backtest_gate.py -v --run-backtest`

Expected: PASS (the snapshot matches itself within tolerance — modulo any non-determinism, which spec section 7.4 says shouldn't exist).

- [ ] **Step 13.6: Verify the gate fails when expected.**

Manually edit `tests/backtest/baseline_metrics.json`. Find a `spearman_topN` row and change its value to `0.999`. Save.

Run: `pytest tests/backtest/test_backtest_gate.py -v --run-backtest`

Expected: FAIL with a regression message like `WR/2024/spearman_topN: 0.999 -> 0.971 (drop 0.028 > 0.020)`.

Revert the snapshot:

```bash
git checkout tests/backtest/baseline_metrics.json
```

- [ ] **Step 13.7: Run the default-on smoke against the populated cache.**

Run: `pytest tests/backtest/test_backtest_smoke.py -v`

Expected: PASS (no longer skipped — cache is populated).

- [ ] **Step 13.8: Commit the snapshot.**

```bash
git add tests/backtest/baseline_metrics.json
git commit -m "$(cat <<'EOF'
feat(test): commit v1 baseline_metrics.json — Plan 3c gate snapshot

Plan 3c Phase 6. First walk-forward run produces ~352 metric rows
across (QB / RB / TE / WR) x (2021 / 2022 / 2023 / 2024) x ~22
metrics each. From this point forward any PR that regresses Model A
beyond the per-metric-type tolerances in tolerances.json will fail
`pytest -m backtest --run-backtest`.

Sanity-check ranges (from the inspection in Step 13.3):
  composite_rmse:        5.X - 8.X depending on (position, year)
  spearman_topN:         0.92 - 0.97
  calibration_p10p90:    0.67 - 0.78

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Update project_management.md, close TODO #4, file new TODOs, update CONTRIBUTING.md

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`
- Modify: `CONTRIBUTING.md`

- [ ] **Step 14.1: Update CONTRIBUTING.md.**

Open `CONTRIBUTING.md`. Find the section that describes the daily workflow / pattern recipes. Add a new subsection (or extend an existing one) titled "After touching `src/projections/features/`":

```markdown
### After touching `src/projections/features/`

The feature cache at `data/features/{position}/season=YYYY/week=WW/part.parquet`
is **not** auto-invalidated when feature builders change. After modifying
any file under `src/projections/features/` you must rebuild the cache
before running the backtest gate:

```bash
python scripts/refresh_features.py all --seasons 2018-2024
```

Then re-snapshot if your change intentionally alters Model A's metrics:

```bash
python scripts/backtest.py --update-snapshot
git diff tests/backtest/baseline_metrics.json    # review the metric deltas
git add tests/backtest/baseline_metrics.json
```

Auto-invalidation is TODO #21 (see TODO.md).
```

Also add (or extend) a "Backtest gate" section near the existing "After bumping `nfl_data_py`" section:

```markdown
### Running the backtest gate before opening a PR

The gate is opt-in:

```bash
pytest -m backtest --run-backtest
```

If the gate fails, the failure message lists each regressing
(position, year, metric) cell with baseline vs current values.

If the regression is intentional (e.g., a feature change that
legitimately improves the model on some cells but slightly worsens
others within tolerance overrides), update the snapshot and commit:

```bash
python scripts/backtest.py --update-snapshot
git add tests/backtest/baseline_metrics.json
```

For genuinely-noisy cells (rare-event RMSE on small samples, etc.),
add a per-row override to `tests/backtest/tolerances.json` instead of
loosening the default. Each override row needs a `rationale` field
describing why the cell is noisy.
```

- [ ] **Step 14.2: Close TODO #4 and add new TODO entries.**

Edit `TODO.md`. Find the "### 4. Decide feature parquet storage during Plan 3" section. Replace it with a "Closed" note:

```markdown
### 4. Feature parquet storage — closed in Plan 3c

Closed 2026-04-26. `data/features/{position}/season=YYYY/week=WW/part.parquet`
populated by `scripts/refresh_features.py`; read by `src/projections/features/cache.py`
and consumed by the backtest harness. Manual invalidation only — see TODO #(next)
below for code-hash auto-invalidation.
```

Append three new entries at the bottom of the open list (use the next available numbers — currently #18 is the last, so add #19, #20, #21):

```markdown
### 19. Walk-forward gate non-determinism check

Phase 6 of Plan 3c may surface tiny RMSE jitter on the 2021 cells
where RidgeCV trains on only 3 prior seasons. If empirically observed
on a re-run with unchanged data, add explicit `random_state` propagation
through `BaselineModel.fit` and re-snapshot. Otherwise close.

### 20. Naive-baseline parquet output for trend tracking

Plan 3c writes naive metrics into the in-memory `BacktestRun` and prints
them in `--report` mode but does not persist them. If we ever want to
track "how much value is Model A adding over naive *over time*", persist
naive metrics to a parquet table at `data/backtest/naive_history/...`
keyed by run timestamp. Not load-bearing for v1.

### 21. Feature cache code-hash auto-invalidation

Plan 3c's feature cache is invalidated manually — the user must re-run
`scripts/refresh_features.py` after touching any feature builder.
Auto-invalidation reads the source files for the feature builder (the
same set `BaselineModel.code_hash_files` already tracks) and refuses
to read stale cache. Deferred until manual invalidation produces a
real-world bug.
```

- [ ] **Step 14.3: Update `project_management.md`.**

Open `project_management.md`. Prepend a new "Plan 3c — backtest harness" section above the existing "Plan 3b — 2024 sanity check" section.

The new section template (replace the placeholder values with the actual numbers from Step 13.4's snapshot):

```markdown
## Plan 3c — Walk-forward backtest gate (run 2026-04-26, on branch `feat/plan-3c-backtest-harness`)

Held-out years: 2021, 2022, 2023, 2024 (4 years × 4 positions = 16 fits per gate run).
Train window: expanding from 2018 → year-1.
Snapshot file: `tests/backtest/baseline_metrics.json` (~352 rows committed).
Gate: `pytest -m backtest --run-backtest` — opt-in, pre-PR.

### Composite metrics by (position, year)

| Position | Year | composite_rmse | composite_mae | spearman_topN | calibration_p10p90 |
|---|---|---|---|---|---|
| QB | 2021 | <fill> | <fill> | <fill> | <fill> |
| QB | 2022 | <fill> | <fill> | <fill> | <fill> |
| QB | 2023 | <fill> | <fill> | <fill> | <fill> |
| QB | 2024 | <fill> | <fill> | <fill> | <fill> |
| RB | 2021 | ... |
| ...

### Naive baseline comparison (informational)

Per-position median composite RMSE across years:
- QB: model <fill> vs naive <fill>  (Model A beats naive by <fill>%)
- RB: model <fill> vs naive <fill>
- TE: model <fill> vs naive <fill>
- WR: model <fill> vs naive <fill>

If any position's model RMSE is within 5% of naive RMSE, that's a signal to revisit feature
engineering — see TODO #(next) for a more formal trend-tracking layer.

### Phase 6 spot-check decisions

- [Any per-row tolerance overrides added during Phase 6 inspection]
- [Any unexpected metric ranges that prompted re-investigation]

```

Then update the "Current status" and "Next action" sections:

```markdown
## Current status (as of 2026-04-26)

**Projections Core — Plan 3c (walk-forward backtest harness + snapshot-diff gate) merged to `main` at commit `<TBD-after-merge>` (PR #<TBD>).**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Plan 2a merged at `7926090`; Plan 2b merged at `af325ea`.
- Plan 3a (WR Model A baseline) merged at `598ab9c`.
- Plan 3b (QB / RB / TE Model A baselines) merged at `c4a0401`.

**Plan 3c delivered:**
- New `src/projections/backtest/` package: walk-forward harness, metric primitives, naive baseline, snapshot diff with direction-aware tolerances.
- New `src/projections/features/cache.py` reader + `scripts/refresh_features.py` writer; closes TODO #4.
- `tests/backtest/baseline_metrics.json` committed (~352 rows). Per-metric-type tolerances in `tests/backtest/tolerances.json` (defaults: 5% RMSE/MAE relative; 0.02 Spearman absolute; 0.03 calibration absolute; 0.10 mean_pred relative).
- Opt-in `pytest -m backtest --run-backtest` gate; default-on smoke covering one (WR, 2024) cell.
- `scripts/backtest.py` CLI (`--check / --update-snapshot / --report`).
- ~30–40 new tests (~325 total).
- TODO #4 closed.
- TODOs #19–#21 filed (gate non-determinism check, naive-baseline trend persistence, feature-cache code-hash auto-invalidation).

## Next action

**Recommended: Plan 3d — real Monte Carlo season-distribution aggregation.**

Plan 3a/3b/3c land the per-week distribution + the regression gate.
Plan 3d closes TODO #13 (per-row seeds), TODO #14 (SAMPLED_SUMMARY family), and the `score_distribution` vectorization perf TODO together — they all converge in a real season-aggregation pipeline. Adds season-total calibration to the gated metrics.

After 3d: Plan 4 (public Python API + CLI verbs + free-tier web hosting).
```

Add the corresponding decision-log row:

```markdown
| 2026-04-26 | Plan 3c gate is opt-in `pytest -m backtest --run-backtest`, not default-on | A full gate run is 30-90s; default-on adds material drag to every dev iteration. Default-on smoke covering one (WR, 2024) cell catches harness wiring bugs cheaply. |
| 2026-04-26 | Snapshot at (position, year, metric) granularity (~352 rows); per-metric-type tolerances | Per-year visibility is the whole point of multi-year backtest; aggregating loses the "regressed only on 2022" signal. Tolerances grouped by metric type keeps maintenance low; per-row overrides added empirically as we observe noise. |
| 2026-04-26 | Held-out years 2021-2024 (skip 2019 / 2020) | 2019's 1-season train window is too small; 2020 is COVID-shortened structural outlier. Each held-out year has at least 3 seasons of training history. |
| 2026-04-26 | Plan 3c uses summed weekly means as season totals (degenerate aggregation); real Monte Carlo aggregation deferred to Plan 3d | Decouples gating infrastructure from season-distribution design. Plan 3d converges three open TODOs (#13 per-row seeds, #14 SAMPLED_SUMMARY, score_distribution perf). |
| 2026-04-26 | Feature cache invalidation is manual via `scripts/refresh_features.py`; auto-invalidation deferred (TODO #21) | Manual is documented in CONTRIBUTING.md and produces a clear FileNotFoundError pointing at the refresh command. Auto-invalidation via code-hash is straightforward but adds surface area; defer until manual produces a real-world bug. |
```

(Update `<TBD-after-merge>` and `<TBD>` after the actual PR lands.)

- [ ] **Step 14.4: Run the full default-on test suite (sanity).**

Run: `pytest -v`

Expected: ~325 tests pass; backtest gate skipped; network smokes skipped. No mypy/ruff/format drift.

- [ ] **Step 14.5: Commit the docs + TODO updates.**

```bash
git add CONTRIBUTING.md TODO.md project_management.md
git commit -m "$(cat <<'EOF'
docs(pm): close Plan 3c — gate live, TODO #4 closed, TODO #19-21 filed

Plan 3c Phase 6 final commit. Records:

- The v1 (position, year) metric table from the first walk-forward run.
- TODO #4 closed (feature parquet caching is live).
- TODO #19 (gate non-determinism check), TODO #20 (naive-baseline
  trend persistence), TODO #21 (feature-cache code-hash auto-
  invalidation) filed for follow-on work.
- CONTRIBUTING.md procedures: refresh_features.py after touching
  features/; backtest gate workflow before opening a PR.
- "Next action" bumped to Plan 3d (real Monte Carlo season
  aggregation; converges TODOs #13 / #14 / score_distribution perf).

Decision-log entries cover the snapshot granularity, held-out year
selection, opt-in pytest mechanism, and the deliberate degenerate-
aggregation choice.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7 — Quality gate

### Task 15: Final verification

Per `CLAUDE.md` § "Forced verification — end-of-effort checklist". Run all five and paste the summary into the PR body. Fix any failure before opening the PR.

- [ ] **Step 15.1: Full default test suite.**

Run: `pytest -v`

Expected: ~325 tests pass; backtest gate skipped (no `--run-backtest`); network smokes skipped (no `--run-network`). No errors, no mypy/ruff complaints.

- [ ] **Step 15.2: Backtest gate.**

Run: `pytest -m backtest --run-backtest -v`

Expected: PASS (snapshot matches itself within tolerance).

- [ ] **Step 15.3: mypy.**

Run: `mypy src tests`

Expected: `Success: no issues found in N source files`. (N grows by ~10 vs 3b — new backtest package + tests.)

- [ ] **Step 15.4: ruff lint.**

Run: `ruff check src tests`

Expected: `All checks passed!`.

- [ ] **Step 15.5: ruff format.**

Run: `ruff format --check src tests`

Expected: `N files already formatted`.

- [ ] **Step 15.6: Network smokes (verify the existing opt-in path didn't regress).**

Run: `pytest -m network --run-network -v`

Expected: 8 passed (the existing TODO #8 smokes).

- [ ] **Step 15.7: Open PR.**

Use `gh pr create` with the standard summary template. Paste the verification output above into the PR body. Reference the spec commit (`3b53058`) and link to it.

Suggested PR title: `Plan 3c: walk-forward backtest harness with snapshot-diff gating`

Suggested PR body sections:
- Summary (3 bullets)
- Plan-3 series context (3a/3b merged, 3c this PR, 3d next)
- Quality gate output (the 5 commands above + their results)
- Phase-6 first-run snapshot inspection (per-position composite metrics, naive comparison)
- TODO items touched (closed: #4; filed: #19–#21)
- Notes for review (the snapshot file size, the deliberate degenerate-aggregation, the manual cache-invalidation)
