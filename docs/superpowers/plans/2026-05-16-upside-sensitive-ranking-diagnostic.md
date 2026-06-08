# Upside-Sensitive Ranking Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Phase 1 diagnostic for TODO #33d: extend `aggregate_to_season` with a `return_samples=True` flag, plumb proper Monte-Carlo aggregation into `scripts/project_season.py`, write a new `scripts/diagnose_upside_ranking.py` CLI that compares ranking under `mean / p90 / blend_70_30 / p_elite`, run it on 2024 + 2025, and emit a markdown report whose final line is a greenlight / marginal / no-greenlight verdict for Phase 2.

**Architecture:** Three additive plumbing changes — (1) `aggregate_to_season(return_samples=True)` so callers can compute `P(season ≥ threshold)` directly from MC sample arrays; (2) `project_season.py` writes a weekly `ProjectionWeeklySchema`-validated parquet + a distributions CSV alongside its existing naïve season-totals CSV (existing CSV byte-identical for back-compat with the VORP / cheat-sheet CLIs); (3) `diagnose_upside_ranking.py` reads both new artifacts + actuals, computes four ranking metrics per (position, season), and renders a markdown verdict. No model retraining; no new feature builders; no production-routing changes; no schema changes for downstream production surfaces.

**Tech Stack:** Python 3.12 strict mode, pandas, pandera, numpy, scipy.stats (Kendall's tau — already a dep), pytest. No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md`

**Branch:** `feat/upside-ranking-diagnostic` (cut from `main` at `8ffa607`; spec committed at `f11e19c`).

**Worktree recommendation:** Tasks 13–14 run `project_season.py` end-to-end for 2 seasons (~10–20 min wall time each, model fits included). If you want the main checkout usable during that time, create a worktree at execution time per `superpowers:using-git-worktrees`. Otherwise execute in-place on the current branch.

**Verification at end of every task:** the CLAUDE.md §4 checklist relevant to the changes — at minimum the pytest subset for the files touched. Full sweep runs in Task 16.

**Pre-commit / venv quirk:** mypy's pre-commit hook resolves to system Python (pydantic v1). Per memory `project_pre_commit_venv_quirk.md`, prepend `.venv/Scripts` to `PATH` before `git commit` whenever a Python file is staged. In PowerShell:

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git commit -m "..."
```

In bash (POSIX `git commit` works too):

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git commit -m "..."
```

**Dirty working-tree note before starting:** main currently has `M reports/season_projection.csv` plus three untracked files (`compare_prediction_to_actuals_2025_results.txt`, `reports/demo_cheat_sheet.csv`, `reports/demo_vorp.parquet`). These are user artifacts from prior work and **should NOT be deleted / reverted by this plan**. Tasks 13–14 will overwrite `reports/season_projection.csv` legitimately as part of generating fresh 2024/2025 projections; that overwrite is correct, just be aware the original `M` state is replaced rather than reverted.

---

## File Structure

**Created in this plan:**
- `scripts/_actuals_helper.py` — extracted `actual_ppr_total(weekly_stats, ruleset)` from `compare_predictions_to_actuals.py`.
- `scripts/diagnose_upside_ranking.py` — new CLI emitting the markdown verdict.
- `reports/season_projection_weekly_<season>.parquet` — per-season `ProjectionWeeklySchema` parquet (Task 13 output, not in git).
- `reports/season_projection_distributions_<season>.csv` — per-season quantile-summary CSV (Task 13 output, not in git).
- `reports/upside_ranking_diagnostic.md` — diagnostic markdown report (Task 14 output, **committed** in Task 15).
- `reports/upside_ranking_diagnostic_table.csv` — per-player per-metric ranks (Task 14 output, **committed** in Task 15).
- `tests/test_aggregation/test_season_return_samples.py` — `aggregate_to_season(return_samples=True)` flag test.
- `tests/test_scripts/test_actuals_helper.py` — extraction parity test.
- `tests/test_scripts/test_project_season_artifacts.py` — `_write_season_artifacts` helper test.
- `tests/test_scripts/test_upside_ranking_metrics.py` — `top_k_overlap`, `top5_rank_err`, Kendall's tau, cell-verdict unit tests.
- `tests/test_scripts/test_diagnose_upside_ranking_cli.py` — end-to-end CLI smoke.

**Modified:**
- `src/projections/aggregation/season.py` — add `return_samples: bool = False` overload; return tuple when True.
- `src/projections/aggregation/__init__.py` — no change (function name stable).
- `scripts/compare_predictions_to_actuals.py` — import `actual_ppr_total` from `_actuals_helper`; drop the inlined helper.
- `scripts/project_season.py` — extract `_write_season_artifacts(weekly, ruleset, out_dir, season)` helper that writes naïve CSV + weekly parquet + distributions CSV; wire into `main()`.
- `TODO.md` — Task 15 appends a verdict subsection to entry #33d.
- `project_management.md` — Task 15 prepends a top-of-file entry summarizing the run + verdict.

---

## Phase 1 — `aggregate_to_season(return_samples=True)` flag

### Task 1: Add `return_samples=True` overload

Pure additive change to `aggregate_to_season`. Existing callers unaffected (default `False` returns the same `pd.DataFrame` as today).

**Files:**
- Create: `tests/test_aggregation/test_season_return_samples.py`
- Modify: `src/projections/aggregation/season.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregation/test_season_return_samples.py`. Reuse the fixture helpers `_build_weekly_row()` and `_to_weekly_frame()` from `tests/test_aggregation/test_season.py` by importing them.

```python
"""Tests for aggregate_to_season(return_samples=True)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.aggregation import aggregate_to_season
from projections.schemas import Ruleset

from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

_RULESET = Ruleset.espn_ppr()


def test_return_samples_false_default_returns_dataframe() -> None:
    """Backward-compat: default behavior is unchanged."""
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    out = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1


def test_return_samples_true_returns_tuple() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    out = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200, return_samples=True)
    assert isinstance(out, tuple)
    assert len(out) == 2
    summary, samples = out
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(samples, dict)


def test_return_samples_summary_matches_no_samples_call() -> None:
    """Same input -> same summary frame whether samples are returned or not (determinism)."""
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    summary_only = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200)
    summary_pair, _ = aggregate_to_season(
        weekly, ruleset=_RULESET, n_samples=200, return_samples=True
    )
    pd.testing.assert_frame_equal(summary_only, summary_pair)


def test_return_samples_dict_keys_match_summary_rows() -> None:
    weekly = _to_weekly_frame(
        [_build_weekly_row(gsis_id="00-0000001", week=w) for w in range(1, 3)]
        + [_build_weekly_row(gsis_id="00-0000002", week=w) for w in range(1, 3)]
    )
    summary, samples = aggregate_to_season(
        weekly, ruleset=_RULESET, n_samples=200, return_samples=True
    )
    expected_keys = {(row["gsis_id"], int(row["season"])) for _, row in summary.iterrows()}
    assert set(samples.keys()) == expected_keys


def test_return_samples_arrays_have_expected_shape_and_mean() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    summary, samples = aggregate_to_season(
        weekly, ruleset=_RULESET, n_samples=500, return_samples=True
    )
    assert len(samples) == 1
    key = next(iter(samples.keys()))
    arr = samples[key]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (500,)
    row = summary.iloc[0]
    assert arr.mean() == pytest.approx(float(row["season_mean"]), rel=1e-6)
    assert float(np.quantile(arr, 0.9)) == pytest.approx(float(row["season_p90"]), rel=1e-6)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_aggregation/test_season_return_samples.py -v
```

Expected: `TypeError: aggregate_to_season() got an unexpected keyword argument 'return_samples'` on the four tests that pass `return_samples=True`; first test (`return_samples_false_default_returns_dataframe`) PASSES.

- [ ] **Step 3: Add the `return_samples` parameter to `aggregate_to_season`**

In `src/projections/aggregation/season.py`, change the signature and inner loop:

```python
from typing import overload

import numpy as np
import pandas as pd

from projections.distributions import unpack_per_stat_params
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    ProjectionSeasonSchema,
    ProjectionWeeklySchema,
    Ruleset,
)
from projections.scoring import derive_row_seed, score_distribution


@overload
def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
    return_samples: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
    return_samples: Literal[True],
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]: ...


def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
    return_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    # ... existing docstring ...
    weekly = ProjectionWeeklySchema.validate(weekly)
    if weekly.empty:
        empty_cols = list(ProjectionSeasonSchema.to_schema().columns.keys())
        empty_frame = ProjectionSeasonSchema.validate(pd.DataFrame(columns=empty_cols))
        if return_samples:
            return empty_frame, {}
        return empty_frame

    # ... existing family / ruleset validation ...

    rows: list[dict[str, object]] = []
    samples_out: dict[tuple[str, int], np.ndarray] = {}
    generated_at = datetime.now(UTC)
    for (gsis_id, season), group in weekly.groupby(["gsis_id", "season"], sort=False):
        season_samples = np.zeros(n_samples, dtype=np.float64)
        for _idx, week_row in group.iterrows():
            # ... existing per-week loop body unchanged ...
            season_samples += week_dist.samples

        # ... existing row dict append ...

        if return_samples:
            samples_out[(str(gsis_id), int(season))] = season_samples.copy()

    out = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    validated = ProjectionSeasonSchema.validate(out)
    if return_samples:
        return validated, samples_out
    return validated
```

Add `from typing import Literal, overload` to the imports.

**Important:** the `season_samples.copy()` is load-bearing — `season_samples` is mutated by the next group's `+=` loop without a copy, but actually since `season_samples` is re-bound to `np.zeros(...)` at the top of each group iteration it's a NEW array each time, so the `.copy()` is technically unnecessary. Keeping it is defensive; remove if mypy complains about return type.

Actually re-reading the existing code: `season_samples = np.zeros(...)` is created INSIDE the group loop, so each group gets a fresh array. The `.copy()` is unnecessary. Drop it:

```python
        if return_samples:
            samples_out[(str(gsis_id), int(season))] = season_samples
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_aggregation/test_season_return_samples.py -v
```

Expected: all 5 tests PASS. Also run the existing aggregation tests to confirm no regression:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_aggregation/ -v
```

Expected: all existing tests still PASS.

- [ ] **Step 5: mypy + ruff + format**

```bash
.venv/Scripts/python.exe -m mypy src/projections/aggregation tests/test_aggregation
.venv/Scripts/python.exe -m ruff check src/projections/aggregation tests/test_aggregation
.venv/Scripts/python.exe -m ruff format --check src/projections/aggregation tests/test_aggregation
```

Expected: zero violations on all three.

- [ ] **Step 6: Commit**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add src/projections/aggregation/season.py tests/test_aggregation/test_season_return_samples.py
git commit -m "feat(aggregation): return_samples=True flag on aggregate_to_season"
```

---

## Phase 2 — Actuals helper extraction

### Task 2: Extract `actual_ppr_total` to `scripts/_actuals_helper.py`

Refactor lifting the inline `_actual_ppr_total` helper out of `compare_predictions_to_actuals.py` into a sibling module. Both `compare_predictions_to_actuals.py` (post-refactor) and the new `diagnose_upside_ranking.py` (Task 11) import it. Public name drops the leading underscore (`actual_ppr_total`) since it's now cross-script.

**Files:**
- Create: `scripts/_actuals_helper.py`
- Modify: `scripts/compare_predictions_to_actuals.py`
- Create: `tests/test_scripts/test_actuals_helper.py`

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_scripts/test_actuals_helper.py`:

```python
"""Parity test: actual_ppr_total in scripts/_actuals_helper.py reproduces the
inline helper that previously lived in scripts/compare_predictions_to_actuals.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from projections.schemas import Ruleset

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_actuals_helper() -> object:
    spec = importlib.util.spec_from_file_location(
        "_actuals_helper", _REPO_ROOT / "scripts" / "_actuals_helper.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_actuals_helper"] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_weekly_stats() -> pd.DataFrame:
    """Two players × two weeks, exercising both passing + rushing + receiving stat sums."""
    return pd.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000001", "00-0000002", "00-0000002"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "position": ["QB", "QB", "WR", "WR"],
            "passing_yards": [300.0, 250.0, 0.0, 0.0],
            "passing_tds": [2, 1, 0, 0],
            "interceptions": [1, 0, 0, 0],
            "rushing_yards": [30.0, 15.0, 5.0, 0.0],
            "rushing_tds": [0, 0, 0, 0],
            "receptions": [0, 0, 7, 5],
            "receiving_yards": [0.0, 0.0, 90.0, 60.0],
            "receiving_tds": [0, 0, 1, 0],
            "fumbles_lost": [0, 0, 0, 0],
        }
    )


def test_actual_ppr_total_groups_by_gsis_id_position() -> None:
    helper = _import_actuals_helper()
    out = helper.actual_ppr_total(_synthetic_weekly_stats(), Ruleset.espn_ppr())
    assert set(out.columns) >= {"gsis_id", "position", "actual_total", "actual_n_weeks"}
    assert len(out) == 2
    qb_row = out[out["gsis_id"] == "00-0000001"].iloc[0]
    wr_row = out[out["gsis_id"] == "00-0000002"].iloc[0]
    assert qb_row["actual_n_weeks"] == 2
    assert wr_row["actual_n_weeks"] == 2
    # QB: 550 pass yd / 25 = 22, 3 pass td * 4 = 12, -2 int, 45 rush yd / 10 = 4.5 -> 36.5
    assert qb_row["actual_total"] == pytest.approx(36.5)
    # WR: 12 rec * 1 PPR = 12, 150 rec yd / 10 = 15, 1 rec td * 6 = 6, 5 rush yd / 10 = 0.5 -> 33.5
    assert wr_row["actual_total"] == pytest.approx(33.5)
```

Add `import pytest` to the imports.

- [ ] **Step 2: Run the test to confirm it fails**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_actuals_helper.py -v
```

Expected: `FileNotFoundError` on the `spec_from_file_location` call (the helper file doesn't exist yet).

- [ ] **Step 3: Create `scripts/_actuals_helper.py`**

```python
"""Shared helper: convert a weekly_stats frame to per-(gsis_id, position) actual
fantasy-point totals under a Ruleset. Lifted from scripts/compare_predictions_to_actuals.py
so both that script and scripts/diagnose_upside_ranking.py can call it.

Public name: actual_ppr_total (dropped the leading underscore since the function
is now cross-script).
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring import score
from projections.scoring.score import StatLine


def actual_ppr_total(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Per-row fantasy points, summed per (gsis_id, position) -> actual season total.

    Returns one row per (gsis_id, position) with columns:
      - gsis_id: str
      - position: str (raw from input)
      - actual_total: float (sum of weekly scored points under `ruleset`)
      - actual_n_weeks: int (distinct weeks present for that player-position)
    """
    points: list[float] = []
    for _, row in weekly_stats.iterrows():
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
    ws = weekly_stats.copy()
    ws["actual_ppr"] = points
    return ws.groupby(["gsis_id", "position"], as_index=False).agg(
        actual_total=("actual_ppr", "sum"),
        actual_n_weeks=("week", "nunique"),
    )
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_actuals_helper.py -v
```

Expected: 1 PASS.

- [ ] **Step 5: Refactor `compare_predictions_to_actuals.py` to import the helper**

In `scripts/compare_predictions_to_actuals.py`:

(a) Drop the entire local `_actual_ppr_total` function definition.

(b) Add at the top of the imports (after the stdlib block, before the `projections` imports):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _actuals_helper import actual_ppr_total
```

Add `import sys` if not already imported.

(c) Replace any internal call to `_actual_ppr_total(ws, ruleset)` with `actual_ppr_total(ws, ruleset)`.

(d) Replace `merged["season_total_mean"] - merged["actual_total"]` if the column name changed (it didn't — `actual_total` is preserved).

- [ ] **Step 6: Sanity-check `compare_predictions_to_actuals.py` still imports**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -c "import sys; sys.path.insert(0, 'scripts'); import compare_predictions_to_actuals"
```

Expected: no output, exit 0.

- [ ] **Step 7: mypy + ruff + format**

```bash
.venv/Scripts/python.exe -m mypy scripts/_actuals_helper.py scripts/compare_predictions_to_actuals.py tests/test_scripts/test_actuals_helper.py
.venv/Scripts/python.exe -m ruff check scripts/_actuals_helper.py scripts/compare_predictions_to_actuals.py tests/test_scripts/test_actuals_helper.py
.venv/Scripts/python.exe -m ruff format --check scripts/_actuals_helper.py scripts/compare_predictions_to_actuals.py tests/test_scripts/test_actuals_helper.py
```

Expected: zero violations.

- [ ] **Step 8: Commit**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/_actuals_helper.py scripts/compare_predictions_to_actuals.py tests/test_scripts/test_actuals_helper.py
git commit -m "refactor(scripts): extract actual_ppr_total helper"
```

---

## Phase 3 — `project_season.py` extension

### Task 3: Extract `_write_season_artifacts` helper + test it

Refactor the trailing block of `project_season.py:main()` (the section that builds `season_totals` and writes the CSV) into a private helper `_write_season_artifacts(weekly, ruleset, out_dir, season, id_map)` so the artifact-writing contract can be unit-tested without retraining models. Helper writes all three artifacts: naïve CSV (byte-identical to today), weekly parquet (new), distributions CSV (new).

**Files:**
- Modify: `scripts/project_season.py`
- Create: `tests/test_scripts/test_project_season_artifacts.py`

- [ ] **Step 1: Write the failing helper test**

Create `tests/test_scripts/test_project_season_artifacts.py`:

```python
"""Tests for project_season._write_season_artifacts: writes three artifacts given
an in-memory weekly ProjectionWeeklySchema-validated frame."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from projections.schemas import Ruleset, _PYARROW_STR
from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULESET = Ruleset.espn_ppr()


def _import_project_season() -> object:
    spec = importlib.util.spec_from_file_location(
        "project_season", _REPO_ROOT / "scripts" / "project_season.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_season"] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_id_map() -> pd.DataFrame:
    """Minimal id_map with just the columns _write_season_artifacts needs."""
    return pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-0033873", "00-0000001"], dtype=_PYARROW_STR),
            "full_name": pd.Series(["Patrick Mahomes", "Synthetic Player"], dtype=_PYARROW_STR),
            "team": pd.Series(["KC", "FA"], dtype=_PYARROW_STR),
        }
    )


def test_write_season_artifacts_emits_three_files(tmp_path: Path) -> None:
    module = _import_project_season()
    weekly = _to_weekly_frame(
        [_build_weekly_row(gsis_id="00-0033873", week=w) for w in range(1, 5)]
        + [_build_weekly_row(gsis_id="00-0000001", week=w) for w in range(1, 5)]
    )
    module._write_season_artifacts(  # type: ignore[attr-defined]
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    assert (tmp_path / "season_projection.csv").exists()
    assert (tmp_path / "season_projection_weekly_2024.parquet").exists()
    assert (tmp_path / "season_projection_distributions_2024.csv").exists()


def test_write_season_artifacts_naive_csv_columns_preserved(tmp_path: Path) -> None:
    """Back-compat: existing downstream surfaces consume this CSV by column name."""
    module = _import_project_season()
    weekly = _to_weekly_frame([_build_weekly_row(week=1)])
    module._write_season_artifacts(  # type: ignore[attr-defined]
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    naive = pd.read_csv(tmp_path / "season_projection.csv")
    assert set(naive.columns) >= {
        "rank",
        "gsis_id",
        "position",
        "season_total_mean",
        "n_weeks",
        "full_name",
        "team",
    }


def test_write_season_artifacts_distributions_csv_has_quantile_cols(tmp_path: Path) -> None:
    module = _import_project_season()
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    module._write_season_artifacts(  # type: ignore[attr-defined]
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    dist = pd.read_csv(tmp_path / "season_projection_distributions_2024.csv")
    assert set(dist.columns) >= {
        "gsis_id",
        "position",
        "season_mean",
        "season_p10",
        "season_p50",
        "season_p90",
        "n_weeks",
        "full_name",
        "team",
    }


def test_write_season_artifacts_weekly_parquet_validates(tmp_path: Path) -> None:
    """Weekly parquet round-trips through ProjectionWeeklySchema."""
    from projections.schemas import ProjectionWeeklySchema

    module = _import_project_season()
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    module._write_season_artifacts(  # type: ignore[attr-defined]
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    weekly_parquet = pd.read_parquet(tmp_path / "season_projection_weekly_2024.parquet")
    ProjectionWeeklySchema.validate(weekly_parquet)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_project_season_artifacts.py -v
```

Expected: `AttributeError: module 'project_season' has no attribute '_write_season_artifacts'`.

- [ ] **Step 3: Add the `_write_season_artifacts` helper to `project_season.py`**

In `scripts/project_season.py`, add the helper above `def main()`:

```python
from projections.aggregation import aggregate_to_season


def _write_season_artifacts(
    *,
    weekly: pd.DataFrame,
    ruleset: Ruleset,
    out_dir: Path,
    season: int,
    id_map: pd.DataFrame,
) -> None:
    """Write three artifacts:
      1. <out_dir>/season_projection.csv         (naive sum-of-means, back-compat)
      2. <out_dir>/season_projection_weekly_<season>.parquet  (ProjectionWeeklySchema)
      3. <out_dir>/season_projection_distributions_<season>.csv  (MC quantile summary)

    weekly: ProjectionWeeklySchema-validated weekly predictions for one season.
    id_map: must have at least [gsis_id, full_name, team] columns.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Artifact 2: weekly parquet (write first; useful as a raw-data drop even if
    # later steps fail).
    weekly_parquet = out_dir / f"season_projection_weekly_{season}.parquet"
    weekly.to_parquet(weekly_parquet, index=False)

    # Artifact 1: naive sum-of-weekly-means CSV (unchanged contract).
    season_totals = weekly.groupby(["gsis_id", "position"], as_index=False).agg(
        season_total_mean=("mean", "sum"), n_weeks=("week", "nunique")
    )
    season_totals = season_totals.merge(
        id_map[["gsis_id", "full_name", "team"]], on="gsis_id", how="left"
    )
    season_totals = season_totals.sort_values(
        "season_total_mean", ascending=False
    ).reset_index(drop=True)
    season_totals.insert(0, "rank", range(1, len(season_totals) + 1))
    season_totals.to_csv(out_dir / "season_projection.csv", index=False)

    # Artifact 3: MC-aggregated distributions CSV (new in this spec).
    season_dist = aggregate_to_season(weekly, ruleset=ruleset, n_samples=10_000)
    season_dist = season_dist.merge(
        id_map[["gsis_id", "full_name", "team"]], on="gsis_id", how="left"
    )
    season_dist = season_dist.sort_values("season_mean", ascending=False).reset_index(drop=True)
    season_dist.to_csv(
        out_dir / f"season_projection_distributions_{season}.csv", index=False
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_project_season_artifacts.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: mypy + ruff + format**

```bash
.venv/Scripts/python.exe -m mypy scripts/project_season.py tests/test_scripts/test_project_season_artifacts.py
.venv/Scripts/python.exe -m ruff check scripts/project_season.py tests/test_scripts/test_project_season_artifacts.py
.venv/Scripts/python.exe -m ruff format --check scripts/project_season.py tests/test_scripts/test_project_season_artifacts.py
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/project_season.py tests/test_scripts/test_project_season_artifacts.py
git commit -m "refactor(project_season): extract _write_season_artifacts helper"
```

### Task 4: Wire `_write_season_artifacts` into `main()`

Replace the existing inline CSV-writing block in `project_season.py:main()` with a single call to the new helper. No new tests — the helper test in Task 3 covers the new contract; this task just wires the helper into the real entry point.

**Files:**
- Modify: `scripts/project_season.py`

- [ ] **Step 1: Read the current `main()` body to confirm the exact lines to replace**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -c "
import pathlib, re
src = pathlib.Path('scripts/project_season.py').read_text()
m = re.search(r'    # Aggregate weekly mean.*?to_string\\(index=False\\)\\s*\\)', src, re.DOTALL)
print(m.group(0) if m else 'NOT FOUND')
"
```

Expected: prints the block from `# Aggregate weekly mean -> season total per gsis_id.` down through the per-position `to_string` print loop. Reference it for the next step.

- [ ] **Step 2: Replace the inline block with a `_write_season_artifacts` call + minimal stdout summary**

Locate the block from the comment `# Aggregate weekly mean -> season total per gsis_id.` down to (but NOT including) the final `if __name__ == "__main__":`. Replace with:

```python
    # Load id_map once for name lookups in the artifact writer.
    id_map = read_partition(args.raw_root, "id_map")

    _write_season_artifacts(
        weekly=weekly,
        ruleset=ruleset,
        out_dir=args.out.parent,
        season=target_season,
        id_map=id_map,
    )
    print(f"\nWrote naïve season totals CSV: {args.out}", flush=True)
    print(
        f"Wrote weekly distributions parquet: "
        f"{args.out.parent / f'season_projection_weekly_{target_season}.parquet'}",
        flush=True,
    )
    print(
        f"Wrote MC distributions CSV: "
        f"{args.out.parent / f'season_projection_distributions_{target_season}.csv'}",
        flush=True,
    )

    # Preserve the existing top-100 / top-10-per-position console summary.
    season_totals = pd.read_csv(args.out)
    print(f"\n=== TOP 100 overall ({target_season} ESPN PPR projection) ===")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    print(
        season_totals.head(100)[
            ["rank", "full_name", "position", "team", "season_total_mean", "n_weeks"]
        ].to_string(index=False)
    )

    for pos_str in ("QB", "RB", "WR", "TE"):
        pos_df = season_totals[season_totals["position"] == pos_str].head(10).copy()
        pos_df.insert(0, "pos_rank", range(1, len(pos_df) + 1))
        print(f"\n=== TOP 10 {pos_str} ===")
        print(
            pos_df[
                ["pos_rank", "rank", "full_name", "team", "season_total_mean", "n_weeks"]
            ].to_string(index=False)
        )
```

The `season_totals = pd.read_csv(args.out)` rereads the just-written CSV to keep the stdout summary code identical to today — avoids a second sort path.

- [ ] **Step 3: Verify the existing `_args.out.parent.mkdir(...)` line is no longer present**

The helper now owns the mkdir. Confirm no duplicate `args.out.parent.mkdir(parents=True, exist_ok=True)` survives in `main()`.

- [ ] **Step 4: mypy + ruff + format**

```bash
.venv/Scripts/python.exe -m mypy scripts/project_season.py
.venv/Scripts/python.exe -m ruff check scripts/project_season.py
.venv/Scripts/python.exe -m ruff format --check scripts/project_season.py
```

Expected: zero violations.

- [ ] **Step 5: Smoke-test `project_season.py --help` parses**

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/project_season.py --help
```

Expected: usage banner prints; exit 0.

- [ ] **Step 6: Commit**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/project_season.py
git commit -m "feat(project_season): wire _write_season_artifacts (weekly parquet + distributions CSV)"
```

---

## Phase 4 — Diagnostic script foundation

### Task 5: `_compute_elite_thresholds` — per-position threshold from historical actuals

Helper inside `scripts/diagnose_upside_ranking.py` that reads `data/raw/weekly_stats/season=<s>` for `s in {2019, ..., 2023}` and returns a `dict[Position, float]` whose values are the mean over those 5 seasons of the 5th-highest actual season ESPN-PPR fantasy points among players with `actual_n_weeks ≥ 8`.

**Files:**
- Create: `scripts/diagnose_upside_ranking.py` (initial skeleton)
- Create: `tests/test_scripts/test_upside_ranking_metrics.py` (will accrete tests over Tasks 5-9)

- [ ] **Step 1: Write the failing test for `_compute_elite_thresholds`**

Create `tests/test_scripts/test_upside_ranking_metrics.py` with:

```python
"""Unit tests for diagnose_upside_ranking helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projections.schemas import Position, Ruleset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULESET = Ruleset.espn_ppr()


def _import_diagnose() -> object:
    spec = importlib.util.spec_from_file_location(
        "diagnose_upside_ranking", _REPO_ROOT / "scripts" / "diagnose_upside_ranking.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_upside_ranking"] = module
    spec.loader.exec_module(module)
    return module


def test_compute_elite_thresholds_returns_one_per_position(tmp_path: Path) -> None:
    """Synthetic 2 seasons × 4 positions × 10 players: threshold = mean over
    seasons of the 5th-highest actual total at each position (≥ 8 games filter)."""
    module = _import_diagnose()
    raw_root = tmp_path / "data" / "raw"
    weekly_stats_root = raw_root / "weekly_stats"
    for season in (2019, 2020):
        partition = weekly_stats_root / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        rows = []
        for pos_idx, pos in enumerate(("QB", "RB", "WR", "TE")):
            for player_idx in range(10):
                # Give each player 10 weeks with a deterministic per-player point total.
                target_ppr = 100.0 + pos_idx * 50 + player_idx * 20  # 100..390 per pos
                for week in range(1, 11):
                    rows.append(
                        {
                            "gsis_id": f"00-{pos}-{player_idx:04d}",
                            "season": season,
                            "week": week,
                            "position": pos,
                            "passing_yards": target_ppr * 0 if pos != "QB" else target_ppr / 10 * 25,
                            "passing_tds": 0,
                            "interceptions": 0,
                            "rushing_yards": 0.0 if pos in ("QB", "WR", "TE") else target_ppr / 10 * 10,
                            "rushing_tds": 0,
                            "receptions": 0 if pos in ("QB", "RB") else int(target_ppr / 10),
                            "receiving_yards": 0.0 if pos in ("QB", "RB") else 0.0,
                            "receiving_tds": 0,
                            "fumbles_lost": 0,
                        }
                    )
        pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)

    thresholds = module._compute_elite_thresholds(
        raw_root=raw_root,
        seasons=(2019, 2020),
        ruleset=_RULESET,
        min_games=8,
    )
    assert set(thresholds.keys()) == {Position.QB, Position.RB, Position.WR, Position.TE}
    for pos, v in thresholds.items():
        assert isinstance(v, float)
        assert v > 0
    # The synthetic data has 10 players per position; the 5th-best by index should
    # be player_idx=5 (target_ppr depends on pos), so thresholds should differ
    # across positions.
    assert len(set(thresholds.values())) == 4
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: `FileNotFoundError` on `spec_from_file_location` (the diagnose script doesn't exist yet).

- [ ] **Step 3: Create `scripts/diagnose_upside_ranking.py` skeleton with `_compute_elite_thresholds`**

```python
"""Phase 1 diagnostic for TODO #33d. Reads weekly-distribution parquet +
distributions CSV from project_season.py output + actuals from data/raw/weekly_stats,
computes ranking under four metrics (mean / p90 / blend_70_30 / p_elite), and
writes a markdown report with a Phase-2-decision verdict.

See docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from projections.schemas import Position, Ruleset
from projections.store import read_partition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _actuals_helper import actual_ppr_total  # noqa: E402


def _compute_elite_thresholds(
    *,
    raw_root: Path,
    seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
    ruleset: Ruleset,
    min_games: int = 8,
) -> dict[Position, float]:
    """Per-position elite threshold = mean over `seasons` of the 5th-highest
    actual season fantasy points at that position, computed over players with
    >= min_games games played that season."""
    per_season_top5: dict[Position, list[float]] = {p: [] for p in (Position.QB, Position.RB, Position.WR, Position.TE)}
    for season in seasons:
        ws = read_partition(raw_root, "weekly_stats", season=season)
        actuals = actual_ppr_total(ws, ruleset)
        actuals = actuals[actuals["actual_n_weeks"] >= min_games]
        for pos in per_season_top5:
            pos_rows = actuals[actuals["position"] == pos.value].sort_values(
                "actual_total", ascending=False
            )
            if len(pos_rows) >= 5:
                per_season_top5[pos].append(float(pos_rows["actual_total"].iloc[4]))
    out: dict[Position, float] = {}
    for pos, vals in per_season_top5.items():
        if not vals:
            raise ValueError(
                f"No seasons in {seasons} produced ≥ 5 players with ≥ {min_games} games at {pos.value}"
            )
        out[pos] = sum(vals) / len(vals)
    return out
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py::test_compute_elite_thresholds_returns_one_per_position -v
```

Expected: 1 PASS.

- [ ] **Step 5: mypy + ruff + format**

```bash
.venv/Scripts/python.exe -m mypy scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff format --check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
git commit -m "feat(diagnose): _compute_elite_thresholds helper for upside-ranking diagnostic"
```

### Task 6: Rank-recovery metric helpers (`top_k_overlap`, `top5_rank_err`, `kendall_tau_filtered`)

Three small numpy/pandas/scipy helpers, fully unit-testable on synthetic Series.

**Files:**
- Modify: `scripts/diagnose_upside_ranking.py` (add three helpers)
- Modify: `tests/test_scripts/test_upside_ranking_metrics.py` (add three test functions)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scripts/test_upside_ranking_metrics.py`:

```python
def test_top_k_overlap_perfect_match() -> None:
    module = _import_diagnose()
    pred_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    actual_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    assert module.top_k_overlap(pred_rank, actual_rank, k=3) == pytest.approx(1.0)
    assert module.top_k_overlap(pred_rank, actual_rank, k=5) == pytest.approx(1.0)


def test_top_k_overlap_partial_match() -> None:
    module = _import_diagnose()
    pred_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    actual_rank = pd.Series([3, 1, 2, 4, 5], index=["a", "b", "c", "d", "e"])
    # pred top-3 = {a, b, c}, actual top-3 = {b, c, a} -> overlap 3/3 = 1.0
    assert module.top_k_overlap(pred_rank, actual_rank, k=3) == pytest.approx(1.0)
    # pred top-1 = {a}, actual top-1 = {b} -> overlap 0/1 = 0
    assert module.top_k_overlap(pred_rank, actual_rank, k=1) == pytest.approx(0.0)


def test_top5_rank_err_median_abs() -> None:
    module = _import_diagnose()
    # actual top-5 = {a, b, c, d, e} with ranks 1..5.
    # pred ranks for {a, b, c, d, e} = [1, 4, 3, 2, 5] -> errors [0, 2, 0, 2, 0]
    # median = 0.0
    pred_rank = pd.Series([1, 4, 3, 2, 5, 6], index=["a", "b", "c", "d", "e", "f"])
    actual_rank = pd.Series([1, 2, 3, 4, 5, 6], index=["a", "b", "c", "d", "e", "f"])
    assert module.top5_rank_err(pred_rank, actual_rank) == pytest.approx(0.0)


def test_kendall_tau_filtered_excludes_low_nweeks() -> None:
    module = _import_diagnose()
    pred = pd.Series([100.0, 90.0, 80.0, 70.0], index=["a", "b", "c", "d"])
    actual = pd.Series([100.0, 90.0, 80.0, 70.0], index=["a", "b", "c", "d"])
    n_weeks = pd.Series([10, 10, 10, 3], index=["a", "b", "c", "d"])
    tau, n = module.kendall_tau_filtered(pred, actual, n_weeks, min_n_weeks=6)
    # 'd' is excluded; perfect rank agreement on the remaining 3 -> tau = 1.0, n = 3
    assert tau == pytest.approx(1.0)
    assert n == 3
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: 4 new tests fail with `AttributeError: module 'diagnose_upside_ranking' has no attribute 'top_k_overlap'` (and similar). `_compute_elite_thresholds` test still passes.

- [ ] **Step 3: Add the three helpers to `diagnose_upside_ranking.py`**

Append to the module:

```python
import numpy as np
from scipy.stats import kendalltau


def top_k_overlap(pred_rank: pd.Series, actual_rank: pd.Series, *, k: int) -> float:
    """|predicted_top_k ∩ actual_top_k| / k. Ranks are 1-based, smallest = best."""
    pred_top = set(pred_rank.nsmallest(k).index)
    actual_top = set(actual_rank.nsmallest(k).index)
    return len(pred_top & actual_top) / k


def top5_rank_err(pred_rank: pd.Series, actual_rank: pd.Series) -> float:
    """For each player in actual top-5: median(|predicted_rank - actual_rank|)."""
    top5 = actual_rank.nsmallest(5).index
    return float((pred_rank.loc[top5] - actual_rank.loc[top5]).abs().median())


def kendall_tau_filtered(
    pred_score: pd.Series,
    actual_score: pd.Series,
    n_weeks: pd.Series,
    *,
    min_n_weeks: int,
) -> tuple[float, int]:
    """Kendall's tau over players with n_weeks >= min_n_weeks. Returns (tau, n)."""
    eligible = n_weeks[n_weeks >= min_n_weeks].index
    pred_e = pred_score.loc[eligible]
    actual_e = actual_score.loc[eligible]
    result = kendalltau(pred_e.to_numpy(), actual_e.to_numpy())
    return float(result.statistic), int(len(eligible))
```

Note: `kendalltau` returns a namedtuple-like object with `.statistic` and `.pvalue` (scipy ≥1.9). If the test fails with `AttributeError: 'KendalltauResult' object has no attribute 'statistic'`, use `result[0]` instead.

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: all 5 tests PASS (1 from Task 5 + 4 from this task).

- [ ] **Step 5: mypy + ruff + format**

```bash
.venv/Scripts/python.exe -m mypy scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff format --check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
git commit -m "feat(diagnose): top_k_overlap + top5_rank_err + kendall_tau_filtered helpers"
```

### Task 7: Per-cell verdict logic (§3.5 of spec)

A pure function that takes (per-metric overlap, per-metric rank-err) vs the mean-baseline values and returns one of {SIGNAL, MARGINAL, NULL, REGRESSION}.

**Files:**
- Modify: `scripts/diagnose_upside_ranking.py`
- Modify: `tests/test_scripts/test_upside_ranking_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scripts/test_upside_ranking_metrics.py`:

```python
def test_cell_verdict_signal() -> None:
    module = _import_diagnose()
    # Metric beats mean by ≥ 1/12 on top-12 overlap AND beats mean on top-5 rank-err.
    verdict = module.cell_verdict(
        metric_top12=0.92,
        mean_top12=0.83,  # delta = 0.09 ≥ 1/12 ≈ 0.083
        metric_rank_err=0.5,
        mean_rank_err=1.5,
    )
    assert verdict == "SIGNAL"


def test_cell_verdict_marginal_one_dim() -> None:
    module = _import_diagnose()
    verdict = module.cell_verdict(
        metric_top12=0.92,
        mean_top12=0.83,
        metric_rank_err=1.5,
        mean_rank_err=1.5,  # tie -> not "better" on this dim
    )
    assert verdict == "MARGINAL"


def test_cell_verdict_null() -> None:
    module = _import_diagnose()
    verdict = module.cell_verdict(
        metric_top12=0.83,
        mean_top12=0.83,
        metric_rank_err=1.5,
        mean_rank_err=1.5,
    )
    assert verdict == "NULL"


def test_cell_verdict_regression() -> None:
    module = _import_diagnose()
    verdict = module.cell_verdict(
        metric_top12=0.50,
        mean_top12=0.83,
        metric_rank_err=3.0,
        mean_rank_err=1.5,
    )
    assert verdict == "REGRESSION"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: 4 new tests fail with `AttributeError`.

- [ ] **Step 3: Add the `cell_verdict` function**

Append to `scripts/diagnose_upside_ranking.py`:

```python
def cell_verdict(
    *,
    metric_top12: float,
    mean_top12: float,
    metric_rank_err: float,
    mean_rank_err: float,
    top12_delta_threshold: float = 1.0 / 12.0,
) -> str:
    """Per-cell verdict per spec §3.5. Returns one of: SIGNAL, MARGINAL, NULL, REGRESSION."""
    top12_better = metric_top12 - mean_top12 >= top12_delta_threshold
    rankerr_better = metric_rank_err < mean_rank_err
    top12_worse = metric_top12 < mean_top12
    rankerr_worse = metric_rank_err > mean_rank_err
    if top12_better and rankerr_better:
        return "SIGNAL"
    if top12_worse and rankerr_worse:
        return "REGRESSION"
    if top12_better or rankerr_better:
        return "MARGINAL"
    return "NULL"
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: mypy + ruff + format + Commit**

```bash
.venv/Scripts/python.exe -m mypy scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff format --check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
```

Then:

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
git commit -m "feat(diagnose): cell_verdict logic per spec §3.5"
```

### Task 8: Cross-season decision-gate logic

Pure function over the per-cell verdict table that returns "Greenlight" / "Marginal" / "No greenlight" per spec §1.3 #3.

**Files:**
- Modify: `scripts/diagnose_upside_ranking.py`
- Modify: `tests/test_scripts/test_upside_ranking_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scripts/test_upside_ranking_metrics.py`:

```python
def _verdict_frame(rows: list[tuple[int, str, str, str]]) -> pd.DataFrame:
    """Helper: rows are (season, position, metric, cell_verdict)."""
    return pd.DataFrame(rows, columns=["season", "position", "metric", "cell_verdict"])


def test_decision_greenlight_when_metric_signal_3plus_positions_both_years() -> None:
    module = _import_diagnose()
    verdicts = _verdict_frame(
        [
            (2024, "QB", "p90", "SIGNAL"),
            (2024, "RB", "p90", "SIGNAL"),
            (2024, "WR", "p90", "SIGNAL"),
            (2024, "TE", "p90", "NULL"),
            (2025, "QB", "p90", "SIGNAL"),
            (2025, "RB", "p90", "SIGNAL"),
            (2025, "WR", "p90", "SIGNAL"),
            (2025, "TE", "p90", "NULL"),
        ]
    )
    assert module.decision_gate(verdicts) == "Greenlight"


def test_decision_marginal_when_signal_only_one_year() -> None:
    module = _import_diagnose()
    verdicts = _verdict_frame(
        [
            (2024, "QB", "p90", "SIGNAL"),
            (2024, "RB", "p90", "SIGNAL"),
            (2024, "WR", "p90", "SIGNAL"),
            (2024, "TE", "p90", "NULL"),
            (2025, "QB", "p90", "MARGINAL"),
            (2025, "RB", "p90", "NULL"),
            (2025, "WR", "p90", "MARGINAL"),
            (2025, "TE", "p90", "NULL"),
        ]
    )
    assert module.decision_gate(verdicts) == "Marginal"


def test_decision_marginal_when_signal_or_marginal_3plus_both_years() -> None:
    module = _import_diagnose()
    verdicts = _verdict_frame(
        [
            (2024, "QB", "blend_70_30", "SIGNAL"),
            (2024, "RB", "blend_70_30", "MARGINAL"),
            (2024, "WR", "blend_70_30", "MARGINAL"),
            (2024, "TE", "blend_70_30", "NULL"),
            (2025, "QB", "blend_70_30", "MARGINAL"),
            (2025, "RB", "blend_70_30", "MARGINAL"),
            (2025, "WR", "blend_70_30", "SIGNAL"),
            (2025, "TE", "blend_70_30", "NULL"),
        ]
    )
    assert module.decision_gate(verdicts) == "Marginal"


def test_decision_no_greenlight_when_all_null() -> None:
    module = _import_diagnose()
    verdicts = _verdict_frame(
        [
            (yr, pos, metric, "NULL")
            for yr in (2024, 2025)
            for pos in ("QB", "RB", "WR", "TE")
            for metric in ("p90", "blend_70_30", "p_elite")
        ]
    )
    assert module.decision_gate(verdicts) == "No greenlight"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: 4 new tests fail with `AttributeError: ... 'decision_gate'`.

- [ ] **Step 3: Add the `decision_gate` function**

Append to `scripts/diagnose_upside_ranking.py`:

```python
def decision_gate(verdicts: pd.DataFrame) -> str:
    """Roll up per-(season, position, metric) cell verdicts to a Phase 2 decision.

    Per spec §1.3 #3:
      - Greenlight iff some single non-mean metric M is SIGNAL at >= 3 of 4 positions
        in both 2024 AND 2025.
      - Marginal iff (a) some M is SIGNAL at >= 3 of 4 positions in exactly one year,
        OR (b) some M is SIGNAL-or-MARGINAL at >= 3 of 4 positions in both years.
      - No greenlight otherwise.
    """
    metrics = [m for m in verdicts["metric"].unique() if m != "mean"]
    years = sorted(verdicts["season"].unique())
    if len(years) != 2:
        raise ValueError(f"decision_gate expects exactly 2 seasons; got {years}")
    y1, y2 = int(years[0]), int(years[1])

    def signal_positions(metric: str, year: int) -> int:
        sub = verdicts[
            (verdicts["metric"] == metric)
            & (verdicts["season"] == year)
            & (verdicts["cell_verdict"] == "SIGNAL")
        ]
        return len(sub)

    def signal_or_marginal_positions(metric: str, year: int) -> int:
        sub = verdicts[
            (verdicts["metric"] == metric)
            & (verdicts["season"] == year)
            & (verdicts["cell_verdict"].isin(["SIGNAL", "MARGINAL"]))
        ]
        return len(sub)

    for metric in metrics:
        if signal_positions(metric, y1) >= 3 and signal_positions(metric, y2) >= 3:
            return "Greenlight"

    for metric in metrics:
        if signal_positions(metric, y1) >= 3 or signal_positions(metric, y2) >= 3:
            return "Marginal"
        if (
            signal_or_marginal_positions(metric, y1) >= 3
            and signal_or_marginal_positions(metric, y2) >= 3
        ):
            return "Marginal"

    return "No greenlight"
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: mypy + ruff + format + Commit**

```bash
.venv/Scripts/python.exe -m mypy scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff format --check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
```

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
git commit -m "feat(diagnose): decision_gate rollup per spec §1.3 #3"
```

---

## Phase 5 — Diagnostic CLI assembly + end-to-end smoke

### Task 9: Per-season per-position metric assembly

The non-CLI core: given a weekly parquet, a distributions CSV, an actuals frame for one season, an elite-threshold dict, and a ruleset, produce (a) a per-player per-metric ranks DataFrame and (b) a per-(position, metric) summary DataFrame with overlap / rank-err / cell-verdict columns.

**Files:**
- Modify: `scripts/diagnose_upside_ranking.py`
- Modify: `tests/test_scripts/test_upside_ranking_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scripts/test_upside_ranking_metrics.py`:

```python
def _synthetic_weekly_with_three_players(season: int) -> pd.DataFrame:
    """Three QBs × 17 weeks, real ProjectionWeeklySchema-valid frame."""
    from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

    rows = []
    for gsis_id, base_yards in [
        ("00-0033873", 280.0),
        ("00-0033874", 250.0),
        ("00-0033875", 220.0),
    ]:
        for week in range(1, 18):
            rows.append(
                _build_weekly_row(
                    gsis_id=gsis_id,
                    season=season,
                    week=week,
                    position="QB",
                    rec_yards_mean=base_yards,
                    rec_yards_std=60.0,
                )
            )
    return _to_weekly_frame(rows)


def _synthetic_distributions_csv(weekly: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    from projections.aggregation import aggregate_to_season

    summary = aggregate_to_season(weekly, ruleset=ruleset, n_samples=1000)
    summary["full_name"] = summary["gsis_id"].map(
        {"00-0033873": "Alpha", "00-0033874": "Beta", "00-0033875": "Gamma"}
    )
    summary["team"] = "TST"
    return summary


def test_assemble_season_diagnostic_returns_per_player_and_summary(tmp_path: Path) -> None:
    module = _import_diagnose()
    weekly = _synthetic_weekly_with_three_players(season=2024)
    dist = _synthetic_distributions_csv(weekly, _RULESET)

    actuals = pd.DataFrame(
        {
            "gsis_id": ["00-0033873", "00-0033874", "00-0033875"],
            "position": ["QB", "QB", "QB"],
            "actual_total": [300.0, 250.0, 200.0],  # matches predicted order
            "actual_n_weeks": [17, 17, 17],
        }
    )
    thresholds = {Position.QB: 290.0, Position.RB: 290.0, Position.WR: 290.0, Position.TE: 290.0}

    per_player, summary = module.assemble_season_diagnostic(
        weekly=weekly,
        distributions=dist,
        actuals=actuals,
        elite_thresholds=thresholds,
        ruleset=_RULESET,
        n_samples=1000,
    )
    assert set(per_player.columns) >= {
        "gsis_id",
        "position",
        "full_name",
        "actual_total",
        "actual_rank",
        "mean",
        "p90",
        "blend_70_30",
        "p_elite",
        "rank_mean",
        "rank_p90",
        "rank_blend_70_30",
        "rank_p_elite",
    }
    assert set(summary.columns) >= {
        "position",
        "metric",
        "top5_overlap",
        "top12_overlap",
        "top24_overlap",
        "top5_rank_err",
        "kendall_tau",
        "cell_verdict",
    }
    # The mean metric should perfectly recover the order in this synthetic setup.
    qb_mean = summary[(summary["position"] == "QB") & (summary["metric"] == "mean")].iloc[0]
    assert qb_mean["top5_overlap"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: 1 new test fails with `AttributeError: ... 'assemble_season_diagnostic'`.

- [ ] **Step 3: Add `assemble_season_diagnostic` to the script**

Append to `scripts/diagnose_upside_ranking.py`:

```python
from projections.aggregation import aggregate_to_season

_METRIC_NAMES = ("mean", "p90", "blend_70_30", "p_elite")


def assemble_season_diagnostic(
    *,
    weekly: pd.DataFrame,
    distributions: pd.DataFrame,
    actuals: pd.DataFrame,
    elite_thresholds: dict[Position, float],
    ruleset: Ruleset,
    n_samples: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_player_df, summary_df) for one season.

    per_player_df: one row per (gsis_id), with metric scores + per-metric ranks
        (descending within position) + actual_total + actual_rank.
    summary_df: one row per (position, metric), with rank-recovery measurements +
        cell_verdict per spec §3.5.
    """
    # 1. Compute per-player p_elite via aggregate_to_season(return_samples=True).
    _, samples = aggregate_to_season(
        weekly, ruleset=ruleset, n_samples=n_samples, return_samples=True
    )

    # 2. Build per-player frame off distributions CSV.
    df = distributions[["gsis_id", "position", "full_name", "season_mean", "season_p90", "n_weeks"]].copy()
    df = df.rename(columns={"season_mean": "mean", "season_p90": "p90"})
    df["blend_70_30"] = 0.7 * df["mean"] + 0.3 * df["p90"]

    # 3. p_elite per row: P(season_samples >= elite_threshold[position]).
    def _p_elite_for(row: pd.Series) -> float:
        pos = Position(row["position"])
        threshold = elite_thresholds[pos]
        # samples keyed by (gsis_id, season); season inferred from the distributions CSV
        # (single-season input here, all rows same season). Look up by gsis_id only,
        # taking the first matching key.
        matching = [arr for (gid, _ssn), arr in samples.items() if gid == row["gsis_id"]]
        if not matching:
            return float("nan")
        return float((matching[0] >= threshold).mean())

    df["p_elite"] = df.apply(_p_elite_for, axis=1)

    # 4. Join actuals.
    df = df.merge(
        actuals[["gsis_id", "actual_total", "actual_n_weeks"]],
        on="gsis_id",
        how="left",
    )
    df["actual_total"] = df["actual_total"].fillna(0.0)
    df["actual_n_weeks"] = df["actual_n_weeks"].fillna(0).astype(int)

    # 5. Per-position ranks under each metric (1 = best).
    for metric in _METRIC_NAMES:
        df[f"rank_{metric}"] = df.groupby("position")[metric].rank(ascending=False, method="min")
    df["actual_rank"] = df.groupby("position")["actual_total"].rank(ascending=False, method="min")

    # 6. Per-(position, metric) summary.
    summary_rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        pos_df = df[df["position"] == pos.value].set_index("gsis_id")
        if pos_df.empty:
            continue
        # Mean baseline (used to compare every other metric to).
        mean_top12 = top_k_overlap(pos_df["rank_mean"], pos_df["actual_rank"], k=12)
        mean_rank_err = top5_rank_err(pos_df["rank_mean"], pos_df["actual_rank"])

        for metric in _METRIC_NAMES:
            m_top5 = top_k_overlap(pos_df[f"rank_{metric}"], pos_df["actual_rank"], k=5)
            m_top12 = top_k_overlap(pos_df[f"rank_{metric}"], pos_df["actual_rank"], k=12)
            m_top24 = top_k_overlap(pos_df[f"rank_{metric}"], pos_df["actual_rank"], k=24)
            m_rank_err = top5_rank_err(pos_df[f"rank_{metric}"], pos_df["actual_rank"])
            tau, n_tau = kendall_tau_filtered(
                pos_df[metric], pos_df["actual_total"], pos_df["actual_n_weeks"], min_n_weeks=6
            )
            if metric == "mean":
                verdict = "BASELINE"
            else:
                verdict = cell_verdict(
                    metric_top12=m_top12,
                    mean_top12=mean_top12,
                    metric_rank_err=m_rank_err,
                    mean_rank_err=mean_rank_err,
                )
            summary_rows.append(
                {
                    "position": pos.value,
                    "metric": metric,
                    "top5_overlap": m_top5,
                    "top12_overlap": m_top12,
                    "top24_overlap": m_top24,
                    "top5_rank_err": m_rank_err,
                    "kendall_tau": tau,
                    "kendall_n": n_tau,
                    "cell_verdict": verdict,
                }
            )
    summary = pd.DataFrame(summary_rows)
    return df, summary
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: mypy + ruff + format + Commit**

```bash
.venv/Scripts/python.exe -m mypy scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
.venv/Scripts/python.exe -m ruff format --check scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
```

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/diagnose_upside_ranking.py tests/test_scripts/test_upside_ranking_metrics.py
git commit -m "feat(diagnose): assemble_season_diagnostic per-player + summary"
```

### Task 10: Markdown rendering + CLI main()

The final layer: render markdown report sections from the per-player + summary frames, run the decision gate, and write the report + the per-player CSV. Wraps in `argparse`.

**Files:**
- Modify: `scripts/diagnose_upside_ranking.py`
- Create: `tests/test_scripts/test_diagnose_upside_ranking_cli.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/test_scripts/test_diagnose_upside_ranking_cli.py`:

```python
"""End-to-end CLI smoke for diagnose_upside_ranking.py.

Builds synthetic weekly parquet + distributions CSV + raw weekly_stats for 2 seasons
× 4 positions, runs the CLI, asserts the markdown report exists and has the
Phase-2-decision line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from projections.schemas import Ruleset, _PYARROW_STR
from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULESET = Ruleset.espn_ppr()


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Returns (raw_root, reports_dir, out_path)."""
    raw_root = tmp_path / "data" / "raw"
    reports_dir = tmp_path / "reports"
    out_path = reports_dir / "upside_ranking_diagnostic.md"

    # Threshold seasons: write minimal weekly_stats for 2019..2023.
    for season in range(2019, 2024):
        partition = raw_root / "weekly_stats" / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        rows = []
        for pos_idx, pos in enumerate(("QB", "RB", "WR", "TE")):
            for player_idx in range(10):
                ppr = 100.0 + pos_idx * 50 + player_idx * 25
                for week in range(1, 11):
                    rows.append(
                        {
                            "gsis_id": f"00-{pos}-{player_idx:04d}",
                            "season": season,
                            "week": week,
                            "position": pos,
                            "passing_yards": (ppr * 25 / 10) if pos == "QB" else 0.0,
                            "passing_tds": 0,
                            "interceptions": 0,
                            "rushing_yards": (ppr * 10 / 10) if pos == "RB" else 0.0,
                            "rushing_tds": 0,
                            "receptions": int(ppr / 10) if pos in ("WR", "TE") else 0,
                            "receiving_yards": 0.0,
                            "receiving_tds": 0,
                            "fumbles_lost": 0,
                        }
                    )
        pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)

    # Eval seasons: 2024 + 2025 weekly_stats actuals + weekly parquets + distributions CSVs.
    reports_dir.mkdir(parents=True, exist_ok=True)
    for season in (2024, 2025):
        # Actuals.
        partition = raw_root / "weekly_stats" / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        rows = []
        for pos_idx, pos in enumerate(("QB", "RB", "WR", "TE")):
            for player_idx in range(8):
                ppr = 100.0 + pos_idx * 50 + player_idx * 30
                for week in range(1, 18):
                    rows.append(
                        {
                            "gsis_id": f"00-{pos}-{player_idx:04d}",
                            "season": season,
                            "week": week,
                            "position": pos,
                            "passing_yards": (ppr * 25 / 17) if pos == "QB" else 0.0,
                            "passing_tds": 0,
                            "interceptions": 0,
                            "rushing_yards": (ppr * 10 / 17) if pos == "RB" else 0.0,
                            "rushing_tds": 0,
                            "receptions": int(ppr / 17) if pos in ("WR", "TE") else 0,
                            "receiving_yards": 0.0,
                            "receiving_tds": 0,
                            "fumbles_lost": 0,
                        }
                    )
        pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)

        # Weekly parquet: 2 players per position × 17 weeks via _build_weekly_row.
        weekly_rows = []
        for pos in ("QB", "RB", "WR", "TE"):
            for player_idx in range(2):
                for week in range(1, 18):
                    weekly_rows.append(
                        _build_weekly_row(
                            gsis_id=f"00-{pos}-{player_idx:04d}",
                            season=season,
                            week=week,
                            position=pos,
                            rec_yards_mean=50.0 + player_idx * 30,
                        )
                    )
        weekly_df = _to_weekly_frame(weekly_rows)
        weekly_df.to_parquet(
            reports_dir / f"season_projection_weekly_{season}.parquet", index=False
        )

        # Distributions CSV (mirror of what project_season would write).
        from projections.aggregation import aggregate_to_season

        summary = aggregate_to_season(weekly_df, ruleset=_RULESET, n_samples=1000)
        summary["full_name"] = "Synthetic " + summary["gsis_id"]
        summary["team"] = "TST"
        summary.to_csv(reports_dir / f"season_projection_distributions_{season}.csv", index=False)

    return raw_root, reports_dir, out_path


def test_diagnose_cli_writes_report_with_decision_line(tmp_path: Path) -> None:
    raw_root, reports_dir, out_path = _write_synthetic_inputs(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "diagnose_upside_ranking.py"),
            "--seasons",
            "2024",
            "2025",
            "--raw-root",
            str(raw_root),
            "--weekly-parquet-template",
            str(reports_dir / "season_projection_weekly_{season}.parquet"),
            "--distributions-csv-template",
            str(reports_dir / "season_projection_distributions_{season}.csv"),
            "--out",
            str(out_path),
            "--n-samples",
            "1000",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert out_path.exists()
    body = out_path.read_text()
    assert "Phase 2 decision" in body
    assert any(verdict in body for verdict in ("Greenlight", "Marginal", "No greenlight"))
    table_csv = out_path.parent / "upside_ranking_diagnostic_table.csv"
    assert table_csv.exists()
    table = pd.read_csv(table_csv)
    assert set(table.columns) >= {
        "season",
        "position",
        "gsis_id",
        "actual_total",
        "actual_rank",
        "mean",
        "p90",
        "blend_70_30",
        "p_elite",
    }
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_diagnose_upside_ranking_cli.py -v
```

Expected: subprocess fails (no `main()` yet — script has no `if __name__ == "__main__":` guard).

- [ ] **Step 3: Add markdown rendering + `main()` to `diagnose_upside_ranking.py`**

Append to `scripts/diagnose_upside_ranking.py`:

```python
import argparse
from io import StringIO


def _render_position_section(
    *,
    season: int,
    position: str,
    per_player: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    out = StringIO()
    out.write(f"\n### {position}\n\n")

    # Per-player top-12 by actual.
    pos_pp = per_player[per_player["position"] == position].sort_values("actual_rank").head(12)
    cols = [
        "actual_rank",
        "full_name",
        "actual_total",
        "mean",
        "p90",
        "blend_70_30",
        "p_elite",
        "rank_mean",
        "rank_p90",
        "rank_blend_70_30",
        "rank_p_elite",
    ]
    out.write(pos_pp[cols].to_markdown(index=False, floatfmt=".2f") + "\n\n")

    # Per-metric summary.
    pos_sum = summary[summary["position"] == position]
    out.write(pos_sum.to_markdown(index=False, floatfmt=".3f") + "\n")
    return out.getvalue()


def _render_report(
    *,
    seasons: tuple[int, ...],
    thresholds: dict[Position, float],
    n_samples: int,
    per_season_per_player: dict[int, pd.DataFrame],
    per_season_summary: dict[int, pd.DataFrame],
    decision: str,
) -> str:
    out = StringIO()
    out.write(f"# Upside-Sensitive Ranking Diagnostic — {', '.join(str(s) for s in seasons)}\n\n")
    out.write("## Setup\n\n")
    out.write(f"- Ruleset: ESPN PPR\n")
    out.write(f"- MC samples: {n_samples} per player per season\n")
    out.write(f"- Elite thresholds (computed from 2019-2023 actuals, >=8 games):\n")
    for pos, v in thresholds.items():
        out.write(f"  - {pos.value} = {v:.1f}\n")
    out.write("\n")

    for season in seasons:
        out.write(f"\n## {season}: per-position diagnostic\n")
        for pos_str in ("QB", "RB", "WR", "TE"):
            out.write(
                _render_position_section(
                    season=season,
                    position=pos_str,
                    per_player=per_season_per_player[season],
                    summary=per_season_summary[season],
                )
            )

    # Cross-season summary.
    out.write("\n## Cross-season summary\n\n")
    cross_rows = []
    for season in seasons:
        for _, row in per_season_summary[season].iterrows():
            cross_rows.append(
                {
                    "season": season,
                    "position": row["position"],
                    "metric": row["metric"],
                    "cell_verdict": row["cell_verdict"],
                }
            )
    cross = pd.DataFrame(cross_rows)
    pivoted = cross.pivot_table(
        index=["position", "metric"],
        columns="season",
        values="cell_verdict",
        aggfunc="first",
    )
    out.write(pivoted.to_markdown() + "\n\n")

    out.write(f"\n## Phase 2 decision\n\n**{decision}**\n")
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="TODO #33d Phase 1 diagnostic.")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--weekly-parquet-template",
        type=str,
        default="reports/season_projection_weekly_{season}.parquet",
    )
    parser.add_argument(
        "--distributions-csv-template",
        type=str,
        default="reports/season_projection_distributions_{season}.csv",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/upside_ranking_diagnostic.md"))
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument(
        "--threshold-seasons",
        type=int,
        nargs="+",
        default=(2019, 2020, 2021, 2022, 2023),
    )
    args = parser.parse_args()

    ruleset = Ruleset.espn_ppr()
    print(
        f"Computing elite thresholds from {args.threshold_seasons[0]}-{args.threshold_seasons[-1]} actuals...",
        flush=True,
    )
    thresholds = _compute_elite_thresholds(
        raw_root=args.raw_root,
        seasons=tuple(args.threshold_seasons),
        ruleset=ruleset,
        min_games=8,
    )
    for pos, v in thresholds.items():
        print(f"  {pos.value} elite_threshold = {v:.1f}", flush=True)

    per_season_per_player: dict[int, pd.DataFrame] = {}
    per_season_summary: dict[int, pd.DataFrame] = {}
    for season in args.seasons:
        weekly_path = Path(args.weekly_parquet_template.format(season=season))
        dist_path = Path(args.distributions_csv_template.format(season=season))
        print(f"\n[{season}] loading {weekly_path}", flush=True)
        weekly = pd.read_parquet(weekly_path)
        dist = pd.read_csv(dist_path)
        ws = read_partition(args.raw_root, "weekly_stats", season=season)
        actuals = actual_ppr_total(ws, ruleset)
        per_player, summary = assemble_season_diagnostic(
            weekly=weekly,
            distributions=dist,
            actuals=actuals,
            elite_thresholds=thresholds,
            ruleset=ruleset,
            n_samples=args.n_samples,
        )
        per_player["season"] = season
        per_season_per_player[season] = per_player
        per_season_summary[season] = summary

    # Cross-season decision-gate.
    cross_rows = []
    for season in args.seasons:
        for _, row in per_season_summary[season].iterrows():
            cross_rows.append(
                {
                    "season": season,
                    "position": row["position"],
                    "metric": row["metric"],
                    "cell_verdict": row["cell_verdict"],
                }
            )
    decision = decision_gate(pd.DataFrame(cross_rows))

    report = _render_report(
        seasons=tuple(args.seasons),
        thresholds=thresholds,
        n_samples=args.n_samples,
        per_season_per_player=per_season_per_player,
        per_season_summary=per_season_summary,
        decision=decision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"\nWrote diagnostic report: {args.out}", flush=True)

    table = pd.concat(
        [df.assign(season=season) for season, df in per_season_per_player.items()],
        ignore_index=True,
    )
    table_path = args.out.parent / "upside_ranking_diagnostic_table.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote per-player CSV: {table_path}", flush=True)

    print(f"\n=== Phase 2 decision ===\n{decision}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to confirm it passes**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_diagnose_upside_ranking_cli.py -v
```

Expected: PASS. If `to_markdown` fails with `ImportError: tabulate`, run `pip install tabulate` (it's a transitive pandas dep, usually present).

- [ ] **Step 5: Run all upside-ranking tests together**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_scripts/test_upside_ranking_metrics.py tests/test_scripts/test_diagnose_upside_ranking_cli.py tests/test_scripts/test_actuals_helper.py tests/test_scripts/test_project_season_artifacts.py tests/test_aggregation/test_season_return_samples.py -v
```

Expected: all PASS.

- [ ] **Step 6: mypy + ruff + format**

```bash
.venv/Scripts/python.exe -m mypy scripts/diagnose_upside_ranking.py tests/test_scripts/test_diagnose_upside_ranking_cli.py
.venv/Scripts/python.exe -m ruff check scripts/diagnose_upside_ranking.py tests/test_scripts/test_diagnose_upside_ranking_cli.py
.venv/Scripts/python.exe -m ruff format --check scripts/diagnose_upside_ranking.py tests/test_scripts/test_diagnose_upside_ranking_cli.py
```

Expected: zero violations.

- [ ] **Step 7: Commit**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add scripts/diagnose_upside_ranking.py tests/test_scripts/test_diagnose_upside_ranking_cli.py
git commit -m "feat(diagnose): CLI assembly + markdown rendering + decision-gate output"
```

---

## Phase 6 — Generate artifacts + run diagnostic

### Task 11: Regenerate 2024 + 2025 projections with the extended `project_season.py`

Run `project_season.py` for 2024 and 2025 to produce the weekly parquet + distributions CSV for each season. Multi-minute model fits per season; expect 20–40 min total wall time.

**Files (artifact generation — no source changes):**
- Create: `reports/season_projection.csv` (overwritten — currently `M` in working tree)
- Create: `reports/season_projection_weekly_2024.parquet`
- Create: `reports/season_projection_distributions_2024.csv`
- Create: `reports/season_projection_weekly_2025.parquet`
- Create: `reports/season_projection_distributions_2025.csv`

- [ ] **Step 1: Confirm `data/raw/weekly_stats/season=2024` and `season=2025` are populated**

```bash
ls data/raw/weekly_stats/
```

Expected: includes `season=2024` and `season=2025` directories. If `season=2025` is missing, the nflreadpy migration (TODO #32) was the most recent thing to populate it — re-ingest if needed:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -c "
from pathlib import Path
from projections.ingest.refresh import refresh
refresh(data_root=Path('data'), seasons=range(2018, 2026))
"
```

- [ ] **Step 2: Run `project_season.py` for 2024**

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/project_season.py --season 2024 2>&1 | tee reports/project_season_2024.log
```

Expected output trailer:

```
Wrote naïve season totals CSV: reports/season_projection.csv
Wrote weekly distributions parquet: reports/season_projection_weekly_2024.parquet
Wrote MC distributions CSV: reports/season_projection_distributions_2024.csv
```

Plus the existing top-100 / top-10-per-position summary.

Wall time: 5–15 minutes (depends on training-set size + model class).

- [ ] **Step 3: Backup the 2024 naïve CSV before the 2025 run overwrites it**

```bash
cp reports/season_projection.csv reports/season_projection_2024.csv
```

- [ ] **Step 4: Run `project_season.py` for 2025**

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/project_season.py --season 2025 2>&1 | tee reports/project_season_2025.log
```

Expected: same output trailer with `_2025` in the new artifact names.

- [ ] **Step 5: Verify all five new artifacts exist + are non-empty**

```bash
ls -la reports/season_projection*.{csv,parquet}
```

Expected:
- `reports/season_projection.csv` (latest run = 2025)
- `reports/season_projection_2024.csv` (backed up in step 3)
- `reports/season_projection_weekly_2024.parquet` + `..._2025.parquet`
- `reports/season_projection_distributions_2024.csv` + `..._2025.csv`

- [ ] **Step 6: Quick sanity check on the distributions CSVs**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -c "
import pandas as pd
for season in (2024, 2025):
    df = pd.read_csv(f'reports/season_projection_distributions_{season}.csv')
    print(f'{season}: {len(df)} rows, cols = {list(df.columns)}')
    print(df.sort_values(\"season_mean\", ascending=False).head(5).to_string(index=False))
"
```

Expected: ~400-600 players per season; columns include `season_mean / season_p10 / season_p50 / season_p90 / n_weeks / full_name / team`. Top-5 should look like the existing top-5 by mean (Lamar / Allen / Maye for QB-heavy slates, etc.).

- [ ] **Step 7: Do NOT commit the parquet/CSV artifacts yet** (Task 14 commits the diagnostic outputs; raw projections are not normally checked in but the spec keeps `reports/upside_ranking_diagnostic.md` + `..._table.csv` for the record).

### Task 12: Run the diagnostic on 2024 + 2025

Execute the CLI; capture the verdict.

**Files:**
- Create: `reports/upside_ranking_diagnostic.md`
- Create: `reports/upside_ranking_diagnostic_table.csv`

- [ ] **Step 1: Run the diagnostic**

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/diagnose_upside_ranking.py --seasons 2024 2025 2>&1 | tee reports/upside_diagnostic_run.log
```

Expected:
- Stdout: elite-threshold values per position (QB ~370, RB ~330, WR ~320, TE ~210); per-season `[season] loading …` lines; final `=== Phase 2 decision === GREENLIGHT|MARGINAL|NO GREENLIGHT` line.
- Files: `reports/upside_ranking_diagnostic.md` + `reports/upside_ranking_diagnostic_table.csv`.

Wall time: < 1 minute (no model fitting; just MC aggregation × 2 seasons + table assembly).

- [ ] **Step 2: Read the verdict + skim the report**

```bash
tail -50 reports/upside_ranking_diagnostic.md
```

Expected: the final `## Phase 2 decision` section + 1-line verdict.

- [ ] **Step 3: Skim a few per-position tables for sanity**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -c "
import pandas as pd
tbl = pd.read_csv('reports/upside_ranking_diagnostic_table.csv')
for season in (2024, 2025):
    print(f'\n=== {season} WR top-5 actual ===')
    wr = tbl[(tbl.season==season) & (tbl.position=='WR')].sort_values('actual_rank').head(5)
    print(wr[['full_name', 'actual_total', 'mean', 'p90', 'blend_70_30', 'p_elite', 'rank_mean', 'rank_p90']].to_string(index=False))
"
```

Expected: Chase 2024 row visible, with `mean` vs `p90` vs `actual_total` all side-by-side. This is the literal load-bearing comparison TODO #33d was framed around.

- [ ] **Step 4: Commit the diagnostic outputs**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add reports/upside_ranking_diagnostic.md reports/upside_ranking_diagnostic_table.csv
git commit -m "report(33d): upside-ranking diagnostic verdict for 2024 + 2025"
```

---

## Phase 7 — Documentation + verification + PR

### Task 13: Update TODO + project_management based on verdict

The exact wording of the TODO #33d update depends on the verdict observed in Task 12 (which can't be known in advance). Two sub-templates below — pick the one matching the verdict and adapt the numbers.

**Files:**
- Modify: `TODO.md` (extend entry #33d with the verdict)
- Modify: `project_management.md` (prepend a new top-of-file entry)

- [ ] **Step 1: Locate the existing TODO #33d entry**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m grep "### 33\\. Elite-season under-projection" -A 1 TODO.md | head -3
```

Or use `rg -n "33\\. Elite-season"` if installed. Note the line number range of the entry; the new subsection goes at the end of the existing entry, before the next top-level `### NN.` heading.

- [ ] **Step 2: Append a 33d-Phase-1 subsection to the TODO entry**

Add at the end of the `### 33.` entry in `TODO.md` (the next sibling section is the end-of-file or whatever follows):

**Greenlight template:**

```markdown
**33d — Phase 1 diagnostic verdict, 2026-05-16: GREENLIGHT.** Diagnostic shipped on
branch `feat/upside-ranking-diagnostic`. Best non-mean metric: <metric> (top-12 overlap
+<X> vs mean at <N>/4 positions in both 2024 + 2025). For 2024 elite WR (Chase, actual
403): mean=<m>, p90=<p>, blend=<b>, p_elite=<pe>. Phase 2 spec greenlit — production
ranking surface (new column on VorpTableSchema + SnakeCheatSheetSchema) is the next
natural step. See `reports/upside_ranking_diagnostic.md` for the full report.
```

**Marginal template:**

```markdown
**33d — Phase 1 diagnostic verdict, 2026-05-16: MARGINAL.** Diagnostic shipped on
branch `feat/upside-ranking-diagnostic`. <Metric M> showed SIGNAL at <N>/4 positions in
<year>; at <M>/4 positions in <other year>; OR mixed SIGNAL/MARGINAL across years.
Phase 2 spec written but flagged as low-confidence; should sweep blend weights and
threshold definitions as part of Phase 2 implementation. See report for details.
```

**No-greenlight template:**

```markdown
**33d — Phase 1 diagnostic verdict, 2026-05-16: NO GREENLIGHT.** Diagnostic shipped on
branch `feat/upside-ranking-diagnostic`. No non-mean metric beats season_mean on top-12
positional overlap at ≥3/4 positions in both 2024 + 2025. The model's per-week
distribution does NOT contain elite-season upside that mean-ranking discards — the
distribution is uniformly compressed for elites, not just its mean. Close 33d; the
elite-season signal lives in feature coverage (TODOs #33b decomposed factor-appropriate
sub-models on WR yards/TDs, or #33c forward-looking Vegas team-context features), not
in distribution-tail mining. See `reports/upside_ranking_diagnostic.md` for the full
report.
```

- [ ] **Step 3: Prepend a new top-of-file entry to `project_management.md`**

Add at the top, after the `# Project Management` H1 and the explanatory paragraph, before the first `## ...` H2 already present:

```markdown
## Upside-Sensitive Ranking Diagnostic — Phase 1 verdict <VERDICT> (2026-05-16, on branch `feat/upside-ranking-diagnostic`)

**Status:** Spec + plan + diagnostic CLI on `feat/upside-ranking-diagnostic`. Phase 1 of TODO #33d. New module `scripts/diagnose_upside_ranking.py` consumes the new `season_projection_weekly_<season>.parquet` + `season_projection_distributions_<season>.csv` artifacts (now emitted by extended `scripts/project_season.py` alongside the existing naïve CSV), computes per-player ranks under `mean / p90 / blend_70_30 / p_elite`, and produces a markdown verdict. Spec at `docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md`; plan at `docs/superpowers/plans/2026-05-16-upside-sensitive-ranking-diagnostic.md`.

**Verdict:** <Greenlight | Marginal | No greenlight>. <One-paragraph summary of which metric won where, with the headline elite-player numbers (e.g. "Chase 2024: mean=<m> p90=<p> actual=403"). For greenlight: name the binding metric + positions + per-year overlap deltas. For no-greenlight: name the strongest cell across all metrics + state that even that doesn't clear the bar.>

**Shipped surface:**
- `src/projections/aggregation/season.py` — `aggregate_to_season(return_samples=True)` overload returning `(summary_df, dict[(gsis_id, season), np.ndarray])`. Backward-compatible default.
- `scripts/_actuals_helper.py` — `actual_ppr_total(weekly_stats, ruleset)` shared helper extracted from `compare_predictions_to_actuals.py`.
- `scripts/project_season.py` — `_write_season_artifacts(...)` helper; emits 3 artifacts per run (naïve CSV unchanged, weekly parquet + distributions CSV new).
- `scripts/diagnose_upside_ranking.py` — new CLI; metric helpers + decision-gate + markdown renderer.
- 14 new tests across `tests/test_aggregation/test_season_return_samples.py`, `tests/test_scripts/test_actuals_helper.py`, `tests/test_scripts/test_project_season_artifacts.py`, `tests/test_scripts/test_upside_ranking_metrics.py`, `tests/test_scripts/test_diagnose_upside_ranking_cli.py`.
- 2 report artifacts: `reports/upside_ranking_diagnostic.md`, `reports/upside_ranking_diagnostic_table.csv`.

**Decision log:**
- **Diagnostic-first scope** (per spec §1.1). Phase 2 production ranking surface (new columns on VorpTableSchema / SnakeCheatSheetSchema) is greenlit / not greenlit based on this verdict; no Phase 2 work in this PR.
- **4 metrics committed before run:** `mean` (baseline), `season_p90`, `blend_70_30 = 0.7·mean + 0.3·p90`, `p_elite = P(season ≥ elite_threshold)`. Blend coefficient committed in spec to avoid data-snooping; sweep deferred to Phase 2.
- **Elite threshold:** 2019–2023 mean of the 5th-highest season fpts at position, ≥8 games. Computed at run time + printed in the report header. Observed thresholds: <fill in actuals from the run>.
- **Decision gate (spec §1.3 #3):** Greenlight requires a single metric SIGNAL at ≥3/4 positions in BOTH 2024 AND 2025. Strict-on-greenlight, lenient-on-marginal; the bar for committing weeks of Phase 2 work is the strong signal.
- **`project_season.py` extension is additive.** Existing `reports/season_projection.csv` contract preserved byte-identically; VORP / cheat-sheet / `compare_predictions_to_actuals.py` consumers untouched.

**Risks logged (spec §6):**
- Independent weekly draws understate true season variance; `season_p90` is biased toward the center vs the true season distribution (composite [p10, p90] coverage shortfall, Plan 5c / Plan 6). Same bias as production aggregator; OK for relative ranking.
- Multiple-comparison effect on 32 verdict cells. Decision-gate's "≥3/4 positions in BOTH years" is the correction.
- Elite-threshold choice is opinionated; ≥8-game filter and top-5 cutoff committed before run.

**Recommended next direction (depends on verdict — pick one):**
1. **If Greenlight:** Write Phase 2 spec for the production ranking surface. New column on `VorpTableSchema` + `SnakeCheatSheetSchema` for the winning metric; CLI flag to switch ranking mode; estimated scope similar to PR #46 (snake cheat sheet).
2. **If Marginal:** Phase 2 spec is written but explicitly flags low-confidence; consider sweeping blend weights as part of Phase 2 itself.
3. **If No greenlight:** Close 33d. Pivot to TODO #33b (decomposed targets with factor-appropriate sub-models on WR yards/TDs) or TODO #33c (forward-looking Vegas team-context features family probe). 33c was flagged in the dry-well-summary as the highest-leverage unexplored direction.

**Plan-vs-execution deviations:** <fill in as they happen — e.g., test fixture sizing, helper extraction-import-path quirks, mypy narrow-vs-widen calls; this section should mirror the spec/plan-deviation sections of other PRs>.

See `reports/upside_ranking_diagnostic.md` for the full verdict report and `reports/upside_ranking_diagnostic_table.csv` for the per-player drill-down.

---
```

- [ ] **Step 4: Commit the docs**

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add TODO.md project_management.md
git commit -m "docs(33d): Phase 1 diagnostic verdict + PM entry"
```

### Task 14: Full verification gates

End-to-end check that the entire suite + lint + type sweep is green before opening the PR.

- [ ] **Step 1: Full pytest run (or scoped if main suite is slow)**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_aggregation tests/test_scripts -v
```

Expected: all PASS, no failures. If you want the full sweep:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest -v
```

Expected: all PASS (modulo the known pre-existing failure noted in `project_management.md`: `test_decomposed_baseline.py::test_dispatch_default_model_class_for_wr_is_unchanged` — stale pin from PR #40/#41 reviews, fails on `main` too, not introduced by this branch).

- [ ] **Step 2: mypy strict on the full project**

```bash
.venv/Scripts/python.exe -m mypy src tests scripts
```

Expected: zero violations across `src/`, `tests/`, `scripts/`. If a violation surfaces in a file not touched by this branch, check whether it exists on `main` too (`git stash; mypy ...; git stash pop`); if yes, note in the PR description and don't fix.

- [ ] **Step 3: ruff check + format on the full project**

```bash
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src tests scripts
```

Expected: zero violations.

- [ ] **Step 4: Commit any final tweaks (if needed)**

If the verification surfaced new fixes:

```powershell
$env:PATH = ".venv\Scripts;$env:PATH"
git add <files>
git commit -m "fix(33d): verification-gate cleanup"
```

### Task 15: Push branch + open PR

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/upside-ranking-diagnostic
```

Expected: branch tracking set up; remote URL printed.

- [ ] **Step 2: Open PR via gh CLI**

```bash
gh pr create --title "feat(33d): upside-sensitive ranking diagnostic (Phase 1)" --body "$(cat <<'EOF'
## Summary

- Phase 1 of TODO #33d (upside-sensitive ranking). Diagnostic-only spec; Phase 2 (production ranking surface) is conditional on verdict.
- New `aggregate_to_season(return_samples=True)` flag; new `_write_season_artifacts` helper in `project_season.py` emits weekly parquet + distributions CSV alongside the unchanged naïve CSV.
- New `scripts/diagnose_upside_ranking.py` CLI computes ranking under mean / p90 / blend_70_30 / p_elite for 2024 + 2025, emits markdown verdict.
- Verdict: see `reports/upside_ranking_diagnostic.md`.

## Test plan

- [x] Unit tests for `aggregate_to_season(return_samples=True)`, `actual_ppr_total` extraction, `_write_season_artifacts`, `top_k_overlap` / `top5_rank_err` / `kendall_tau_filtered`, `cell_verdict`, `decision_gate`, `assemble_season_diagnostic`.
- [x] End-to-end CLI smoke test for `diagnose_upside_ranking.py`.
- [x] mypy strict + ruff check + ruff format clean across `src/ tests/ scripts/`.
- [x] Diagnostic actually run on 2024 + 2025; report committed under `reports/`.
- [x] TODO.md #33d + project_management.md updated with the verdict.

Spec: `docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md`
Plan: `docs/superpowers/plans/2026-05-16-upside-sensitive-ranking-diagnostic.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Confirm PR URL printed; copy to the user**

The terminal output will print the PR URL. Hand it to the user.

---

## Self-review checklist (run after every task in §6/§7 if you skip ahead)

- `mypy src tests scripts` strict — zero violations.
- `ruff check src tests scripts` — zero violations.
- `ruff format --check src tests scripts` — zero violations.
- Relevant `pytest` subset green.
- Commit messages follow the project convention (`type(scope): subject`).
- No accidental deletions of the user's working-tree files (`reports/season_projection.csv` is legitimately overwritten in Task 11; the three untracked files at branch start should be untouched).
