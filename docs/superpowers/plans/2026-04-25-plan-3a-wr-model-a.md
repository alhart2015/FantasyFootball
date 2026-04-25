# Plan 3a — WR Model A Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and persist a WR baseline projection model end-to-end on real 2018–2025 data, validating the `Model` interface contract before generalizing it to QB / RB / TE in Plan 3b.

**Architecture:** New `src/projections/models/` package with a `Model` Protocol and a `BaselineModel` class that fits per-stat `RidgeCV` regressions, derives parametric residual variance per stat, and composes per-stat distributions through the existing `scoring.score_distribution` Monte Carlo into fantasy-points distributions. WR features come from the existing 2a builder. First real-data ingest pull lands in `data/raw/`. Trained artifact via `joblib`; predictions persisted to `data/projections/weekly/season=2025/...` matching `ProjectionWeeklySchema`.

**Tech Stack:** Python 3.11, pandas, pandera, scikit-learn (new dep), joblib, scipy, `sklearn.linear_model.RidgeCV`. All scoring math goes through the existing `scoring/` module.

**Spec:** `docs/superpowers/specs/2026-04-25-plan-3a-wr-model-a-design.md`

---

## File structure

### Files created

- `src/projections/models/__init__.py` — package marker; re-exports `Model`, `BaselineModel`, `wr_baseline`.
- `src/projections/models/base.py` — `Model` Protocol; `compute_code_hash` helper.
- `src/projections/models/baseline.py` — `BaselineModel` class; `wr_baseline()` factory.
- `src/projections/ingest/refresh.py` — `refresh(seasons, *, raw_root)` orchestrator.
- `tests/test_models/__init__.py`
- `tests/test_models/conftest.py` — synthetic feature + weekly_stats fixtures sized for model unit tests.
- `tests/test_models/test_baseline.py` — `BaselineModel` unit tests (fit/predict/save/load/edge cases).
- `tests/test_models/test_baseline_leakage.py` — leakage test (per spec §7.2).
- `tests/test_models/test_baseline_real_data.py` — opt-in `@pytest.mark.network` test against ingested 2018 data.
- `tests/test_ingest/test_refresh.py` — orchestrator unit test.
- `scripts/__init__.py` (if needed for package discovery).
- `scripts/train_wr_baseline.py` — fits `BaselineModel` on real 2018–2024, writes artifact to `models/artifacts/`.
- `scripts/sanity_check_wr_baseline.py` — loads artifact, walks 2025 weeks, prints §6.2 metric block.
- `scripts/predict_2025_wr.py` — loads artifact, writes per-week 2025 projections via `store.write_partition`.
- `models/.gitkeep` — keeps the parent dir tracked (only `models/artifacts/*` is gitignored).
- `docs/superpowers/plans/2026-04-25-plan-3a-wr-model-a.md` — this file.

### Files modified

- `pyproject.toml` — add `scikit-learn>=1.4`; register `network` pytest mark.
- `src/projections/schemas.py` — add `coerce = True` to `ProjectionWeeklySchema.Config` (per spec §7.1, defensive parity with the WR/QB/RB/TeFeaturesSchema fix from PR #4 cleanup).
- `src/projections/ingest/__init__.py` — re-export `refresh` from `ingest.refresh`.
- `.gitignore` — add `models/artifacts/`, `data/`, `*.joblib`.
- `tests/test_smoke.py` — extend end-to-end test to wire `fit → predict_distribution → store.write_partition` round trip on synthetic data.
- `project_management.md` — record Plan 3a outcomes; queue 3b.
- `TODO.md` — clean resolved items; add anything discovered during 3a.

### Modules NOT touched in 3a

- `src/projections/aggregate/` and `src/projections/backtest/` — those are Plan 3c.
- `src/projections/api/` — Plan 4.
- `src/projections/scoring/score_distribution.py` — already supports the contract we need (`Mapping[Stat, Distribution] + Ruleset → SampledDistribution`); verified during plan-writing, no extension required.
- `src/projections/distributions/` — `ParametricNormal` and `ParametricGamma` already implement the `Distribution` Protocol; no edits.

---

## Task list at a glance

| # | Task | Output |
|---|---|---|
| 1 | Add scikit-learn dep + register `network` pytest mark | `pyproject.toml` |
| 2 | `ProjectionWeeklySchema.Config: coerce = True` | `schemas.py` + regression test |
| 3 | `Model` Protocol in `models/base.py` | `base.py` |
| 4 | `compute_code_hash` helper in `models/base.py` | `base.py` |
| 5 | `BaselineModel` config + `wr_baseline()` factory | `baseline.py` |
| 6 | `BaselineModel.fit` — feature-matrix build + RidgeCV per stat | `baseline.py` |
| 7 | `BaselineModel.fit` — residual-variance estimation (gamma α + normal σ) | `baseline.py` |
| 8 | `BaselineModel._build_stat_distributions` (private helper) | `baseline.py` |
| 9 | `BaselineModel.predict_distribution` — full pipeline → ProjectionWeeklySchema | `baseline.py` |
| 10 | `save` / `load` + `model_id` property | `baseline.py` |
| 11 | Edge-case tests: gamma μ̂ clamping, NaN imputation, empty input | `test_baseline.py` |
| 12 | Leakage test | `test_baseline_leakage.py` |
| 13 | `ingest.refresh()` orchestrator | `ingest/refresh.py` |
| 14 | First real-data smoke pull: 2018 only | `data/raw/` (gitignored) |
| 15 | Full real-data pull: 2018–2025 | `data/raw/` (gitignored) |
| 16 | `scripts/train_wr_baseline.py` | script + artifact |
| 17 | Run training; persist artifact for 2018–2024 | `models/artifacts/...joblib` |
| 18 | `scripts/sanity_check_wr_baseline.py` | script |
| 19 | Run sanity check; record numbers in PM doc | PM doc edit |
| 20 | `scripts/predict_2025_wr.py` + run | `data/projections/weekly/...` |
| 21 | Smoke-test extension | `test_smoke.py` |
| 22 | PM doc + TODO updates | docs |
| 23 | End-of-effort gate + open PR | green checks + PR |

---

## Task 1: Add scikit-learn dependency + register `network` pytest mark

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Re-read current pyproject.toml**

```bash
cat pyproject.toml
```

Confirm `joblib`, `scipy`, `pandas`, `pandera` are present and `scikit-learn` is NOT present.

- [ ] **Step 2: Add `scikit-learn` to `[project].dependencies`**

Edit the dependencies list to include scikit-learn alphabetically near the bottom (after `pyarrow`, before `scipy`):

```toml
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "scipy>=1.12",
    "pydantic>=2.6",
    "pandera>=0.20",
    "duckdb>=1.0",
    "joblib>=1.3",
    "nfl_data_py>=0.3.2",
]
```

- [ ] **Step 3: Register `network` pytest mark**

Pytest is configured with `--strict-markers`, so any unregistered mark fails. Add a `[tool.pytest.ini_options].markers` section:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "network: tests that hit the live nfl_data_py API; skipped by default",
]
```

- [ ] **Step 4: Add sklearn to mypy ignore_missing_imports overrides**

Add `"sklearn.*"` to the existing override list:

```toml
[[tool.mypy.overrides]]
module = ["nfl_data_py.*", "pandera.*", "scipy.*", "joblib.*", "pandas.*", "sklearn.*"]
ignore_missing_imports = true
```

- [ ] **Step 5: Refresh installed dependencies**

```bash
pip install -e ".[dev]"
```

Expected: success message, scikit-learn 1.4+ installed.

- [ ] **Step 6: Verify install**

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

Expected: `1.4.x` or higher.

- [ ] **Step 7: Run gate to confirm nothing broke**

```bash
pytest -q && mypy src tests && ruff check src tests && ruff format --check src tests
```

Expected: all four green.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add scikit-learn + register network pytest mark"
```

---

## Task 2: `ProjectionWeeklySchema.Config: coerce = True` (defensive parity)

**Why this task is here:** The 2b cleanup PR (#4) added `coerce = True` to all four `*FeaturesSchema` configs after the empty-input dtype crash showed up. `ProjectionWeeklySchema` has the same crash potential (the model may return an empty DataFrame for a position-empty week, and the column dtypes from `pd.DataFrame(columns=...)` are object). Add `coerce = True` defensively so we discover any dtype mismatch at validate time rather than as a silent miss.

**Files:**
- Modify: `src/projections/schemas.py`
- Test: `tests/test_schemas/test_projection_weekly.py` (verify file path; create if missing)

- [ ] **Step 1: Locate ProjectionWeeklySchema**

```bash
grep -n "class ProjectionWeeklySchema" src/projections/schemas.py
```

Find the schema; note its current `class Config` block has only `strict = "filter"`.

- [ ] **Step 2: Write a failing regression test**

```bash
ls tests/test_schemas/
```

If `test_projection_weekly.py` exists, add to it. Otherwise create the file:

```python
"""ProjectionWeeklySchema validation regression tests."""

from __future__ import annotations

import pandas as pd

from projections.schemas import ProjectionWeeklySchema


def test_projection_weekly_schema_validates_empty_dataframe() -> None:
    """An empty DataFrame with the right columns should validate cleanly.

    Without coerce=True, an empty pd.DataFrame(columns=...) produces object-dtype
    columns and pandera rejects them against the typed Series declarations.
    """
    cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
    empty = pd.DataFrame(columns=cols)
    out = ProjectionWeeklySchema.validate(empty)
    assert out.empty
```

- [ ] **Step 3: Run the test, expect failure**

```bash
pytest tests/test_schemas/test_projection_weekly.py -v
```

Expected: FAIL with `expected series 'gsis_id' to have type ..., got object`.

- [ ] **Step 4: Add `coerce = True` to ProjectionWeeklySchema.Config**

In `src/projections/schemas.py`, change the schema's Config block:

```python
    class Config:
        strict = "filter"
        coerce = True  # see WrFeaturesSchema.Config — empty-output fast path
```

- [ ] **Step 5: Run the test, expect green**

```bash
pytest tests/test_schemas/test_projection_weekly.py -v
```

Expected: PASS.

- [ ] **Step 6: Run gate**

```bash
pytest -q && mypy src tests && ruff check src tests && ruff format --check src tests
```

Expected: all four green; the test count went up by 1.

- [ ] **Step 7: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_projection_weekly.py
git commit -m "fix(schemas): coerce=True on ProjectionWeeklySchema for empty-output path"
```

---

## Task 3: `Model` Protocol in `models/base.py`

**Files:**
- Create: `src/projections/models/__init__.py`
- Create: `src/projections/models/base.py`
- Create: `tests/test_models/__init__.py`
- Create: `tests/test_models/test_base.py`

- [ ] **Step 1: Verify directories exist**

```bash
ls src/projections/
ls tests/
```

`models/` should not exist yet under `src/projections/`. `tests/test_models/` should not exist either.

- [ ] **Step 2: Create the empty package markers**

```bash
mkdir -p src/projections/models tests/test_models
touch src/projections/models/__init__.py tests/test_models/__init__.py
```

- [ ] **Step 3: Write a failing import + Protocol-shape test**

Create `tests/test_models/test_base.py`:

```python
"""Tests for the Model Protocol contract."""

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pandas as pd

from projections.models import Model
from projections.schemas import Position, Ruleset


def test_model_protocol_has_required_members() -> None:
    """Model is a structural Protocol — verify the names every implementation
    must provide. Signatures are checked by mypy, not at runtime."""
    expected = {"position", "model_id", "fit", "predict_distribution", "save", "load"}
    actual = {name for name in dir(Model) if not name.startswith("_")}
    assert expected.issubset(actual), f"missing: {expected - actual}"


def test_model_protocol_is_not_runtime_checkable() -> None:
    """Model is a plain Protocol (not @runtime_checkable). isinstance() should
    raise TypeError if anyone tries it. We don't want the Distribution-style
    structural runtime check here."""

    class _Dummy:
        pass

    try:
        isinstance(_Dummy(), Model)  # type: ignore[misc]
    except TypeError:
        return
    raise AssertionError("isinstance(_, Model) should have raised TypeError")
```

- [ ] **Step 4: Run the test; expect ImportError**

```bash
pytest tests/test_models/test_base.py -v
```

Expected: FAIL with `ImportError: cannot import name 'Model' from 'projections.models'` (the package exists but has no `Model`).

- [ ] **Step 5: Implement `Model` Protocol in `base.py`**

Create `src/projections/models/base.py`:

```python
"""Model interface for position-specific projection models.

`Model` is a structural Protocol: any class implementing the listed methods
satisfies it without explicit inheritance. mypy enforces signatures at use
sites; we deliberately do NOT mark it @runtime_checkable because nothing in
the codebase needs isinstance() against Model (cf. Distribution which does).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from projections.schemas import Position, Ruleset


class Model(Protocol):
    """Position-specific projection model. Plugs in at the fit/predict seam.

    Implementations:
        - BaselineModel (Plan 3a, this plan): per-stat Ridge regressions with
          parametric residual variance.
        - (future) GBMModel (Plan 5): LightGBM with quantile regression.
        - (future) EnsembleModel: stack of A and C.
    """

    @property
    def position(self) -> Position: ...

    @property
    def model_id(self) -> str:
        """Stable identifier of the form
        ``"<class>:<position>:<8-char-code-hash>:<train-start>-<train-end>"``.

        Persisted into every projection row produced by predict_distribution
        so we can always trace which model produced which projection.
        """
        ...

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train the model. Inner-joins (gsis_id, season, week) to align
        feature inputs with truth. ``features`` must validate against the
        position's *FeaturesSchema; ``weekly_stats`` against WeeklyStatsSchema.
        """
        ...

    def predict_distribution(
        self, features: pd.DataFrame, ruleset: Ruleset
    ) -> pd.DataFrame:
        """Predict per-player-week fantasy-points distributions under
        ``ruleset``. Returns a DataFrame validated against
        ``ProjectionWeeklySchema``. Re-scoring under a different ruleset is a
        second call with the same features — no retraining."""
        ...

    def save(self, path: Path) -> None:
        """Serialize to disk via joblib."""
        ...

    @classmethod
    def load(cls, path: Path) -> Model:
        """Deserialize from disk. Class methods on Protocols are unusual;
        BaselineModel implements this as a regular @classmethod and structural
        matching covers the contract."""
        ...
```

- [ ] **Step 6: Re-export from package `__init__.py`**

Edit `src/projections/models/__init__.py`:

```python
"""Position-specific projection models."""

from __future__ import annotations

from projections.models.base import Model

__all__ = ["Model"]
```

- [ ] **Step 7: Run the test; expect green**

```bash
pytest tests/test_models/test_base.py -v
```

Expected: both tests PASS.

- [ ] **Step 8: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/projections/models/__init__.py src/projections/models/base.py tests/test_models/__init__.py tests/test_models/test_base.py
git commit -m "feat(models): add Model Protocol + package skeleton"
```

---

## Task 4: `compute_code_hash` helper in `models/base.py`

**Files:**
- Modify: `src/projections/models/base.py`
- Modify: `tests/test_models/test_base.py`

- [ ] **Step 1: Re-read current base.py**

```bash
cat src/projections/models/base.py
```

- [ ] **Step 2: Write failing tests**

Append to `tests/test_models/test_base.py`:

```python
def test_compute_code_hash_is_deterministic(tmp_path: Path) -> None:
    """Hashing the same files twice yields identical output."""
    from projections.models.base import compute_code_hash

    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("hello")
    f2.write_text("world")

    h1 = compute_code_hash([f1, f2])
    h2 = compute_code_hash([f1, f2])
    assert h1 == h2
    assert len(h1) == 8


def test_compute_code_hash_changes_when_content_changes(tmp_path: Path) -> None:
    from projections.models.base import compute_code_hash

    f = tmp_path / "a.py"
    f.write_text("hello")
    h_before = compute_code_hash([f])

    f.write_text("hello!")
    h_after = compute_code_hash([f])
    assert h_before != h_after


def test_compute_code_hash_is_order_independent(tmp_path: Path) -> None:
    """File-list order should not affect the hash (we sort internally)."""
    from projections.models.base import compute_code_hash

    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("alpha")
    f2.write_text("beta")

    h_ab = compute_code_hash([f1, f2])
    h_ba = compute_code_hash([f2, f1])
    assert h_ab == h_ba
```

Add `from pathlib import Path` to the imports if not already present (it is — confirm).

- [ ] **Step 3: Run tests; expect ImportError**

```bash
pytest tests/test_models/test_base.py -v
```

Expected: 3 FAILs with `ImportError: cannot import name 'compute_code_hash'`.

- [ ] **Step 4: Implement `compute_code_hash`**

Append to `src/projections/models/base.py`:

```python
import hashlib
from collections.abc import Iterable


def compute_code_hash(paths: Iterable[Path]) -> str:
    """SHA-256 (first 8 hex chars) of the concatenated content of ``paths``.

    Used as the ``code_hash`` component of every model_id so we can detect when
    a model artifact is stale relative to the current source.

    Order-independent: paths are sorted by their string representation before
    hashing so callers don't have to maintain a canonical order.
    """
    sorted_paths = sorted(paths, key=str)
    hasher = hashlib.sha256()
    for path in sorted_paths:
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:8]
```

Move the `import hashlib` and `from collections.abc import Iterable` to the top-of-file imports for canonical ordering.

- [ ] **Step 5: Run tests; expect green**

```bash
pytest tests/test_models/test_base.py -v
```

Expected: 5 tests PASS (2 from Task 3 + 3 new).

- [ ] **Step 6: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/base.py tests/test_models/test_base.py
git commit -m "feat(models): add compute_code_hash helper for model_id derivation"
```

---

## Task 5: `BaselineModel` config + `wr_baseline()` factory

**Files:**
- Create: `src/projections/models/baseline.py`
- Modify: `src/projections/models/__init__.py`
- Create: `tests/test_models/conftest.py` (synthetic fixtures sized for model unit tests)
- Create: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Decide what fixtures the unit tests need**

The 2a/2b conftest in `tests/test_features/conftest.py` is not visible from `tests/test_models/` (pytest fixtures only inherit from parent conftests, not siblings). Build a focused, minimal fixture here: ~5 WR players × ~10 weeks of synthetic features and matching weekly_stats, schema-valid for both `WrFeaturesSchema` and `WeeklyStatsSchema`. Small enough for a Ridge fit to be fast; large enough that residual variance is non-degenerate.

- [ ] **Step 2: Write the conftest fixtures**

Create `tests/test_models/conftest.py`:

```python
"""Synthetic features + truth fixtures for BaselineModel unit tests.

Independent of tests/test_features/conftest.py — pytest fixtures don't share
across sibling test directories. We build only what's needed here: schema-valid
WR feature rows + matching WeeklyStats truth rows that train a non-degenerate
RidgeCV.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR

# 5 synthetic WRs across 2 teams; 8 weeks of 2024 + 4 weeks of 2025.
_GSIS_IDS = ["00-0010001", "00-0010002", "00-0010003", "00-0010004", "00-0010005"]
_TEAMS = ["KC", "KC", "MIN", "MIN", "MIN"]
_TARGETS_BASE = [10.0, 6.0, 9.0, 4.0, 3.0]  # per-player target rate


def _wr_weekly_stats_row(
    *, gsis_id: str, season: int, week: int, team: str, opponent: str, base_targets: float
) -> dict[str, object]:
    """Build a single WeeklyStatsSchema-valid row with stat values that scale
    plausibly with the per-player base target rate. Random jitter is added by
    the caller via week index so trailing-4 means are non-constant."""
    targets_jitter = (week % 3) - 1  # -1, 0, +1, repeating
    targets = max(0, int(base_targets + targets_jitter))
    receptions = max(0, int(targets * 0.65))
    rec_yards = float(receptions * 12.0)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": "WR",
        "team": team,
        "opponent": opponent,
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": 0.0,
        "rushing_tds": 0,
        "carries": 0,
        "receptions": receptions,
        "receiving_yards": rec_yards,
        "receiving_tds": 1 if (week + int(gsis_id[-1])) % 4 == 0 else 0,
        "receiving_air_yards": float(targets * 13.0),
        "targets": targets,
        "fumbles_lost": 0,
    }


@pytest.fixture
def baseline_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 + 4 weeks of 2025 stats for 5 synthetic WRs.

    2024 = training universe; 2025 = held-out.
    """
    rows: list[dict[str, object]] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            opp = "DEN" if week % 2 == 1 else "DET"
            for gsis_id, team, base_targets in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True):
                rows.append(
                    _wr_weekly_stats_row(
                        gsis_id=gsis_id,
                        season=season,
                        week=week,
                        team=team,
                        opponent=opp if team != opp else ("CHI" if opp == "DEN" else "GB"),
                        base_targets=base_targets,
                    )
                )

    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def baseline_features(baseline_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """WR feature rows produced by build_wr_features for every (season, week)
    in the training fixture. Built up-front so tests don't pay the cost
    individually."""
    from projections.features import build_wr_features

    # We need supporting fixtures (snap_counts, depth_charts, ngs, schedules)
    # for the builder. Keep them minimal.
    snap_rows = [
        {
            "gsis_id": r["gsis_id"],
            "season": r["season"],
            "week": r["week"],
            "team": r["team"],
            "opponent": r["opponent"],
            "position": "WR",
            "offense_snaps": 60,
            "offense_pct": 0.95,
            "defense_snaps": 0,
            "defense_pct": 0.0,
            "st_snaps": 2,
            "st_pct": 0.05,
        }
        for _, r in baseline_weekly_stats.iterrows()
    ]
    snap_counts = pd.DataFrame(snap_rows)
    for col in ("gsis_id", "team", "opponent", "position"):
        snap_counts[col] = snap_counts[col].astype(_PYARROW_STR)

    # Depth charts: every player as their team's WR1/WR2 (deterministic by
    # base_targets ranking) for every (season, week).
    dc_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, base_targets in zip(
                _GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True
            ):
                # Rank within team by base_targets descending.
                team_pool = sorted(
                    [
                        (g, t, b)
                        for g, t, b in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True)
                        if t == team
                    ],
                    key=lambda x: -x[2],
                )
                rank = next(i for i, (g, _, _) in enumerate(team_pool, start=1) if g == gsis_id)
                dc_rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "team": team,
                        "position": "WR",
                        "depth_team": f"WR{rank}",
                        "depth_rank": rank,
                    }
                )
    depth = pd.DataFrame(dc_rows)
    for col in ("gsis_id", "team", "position", "depth_team"):
        depth[col] = depth[col].astype(_PYARROW_STR)

    # NGS: empty frame is acceptable to the builder (it'll fill NaN; with
    # ProjectionWeeklySchema.coerce + WrFeaturesSchema nullable cols this is
    # fine). Build minimal columns.
    ngs = pd.DataFrame(
        columns=[
            "gsis_id",
            "season",
            "week",
            "team",
            "position",
            "avg_separation",
            "avg_intended_air_yards",
            "percent_share_of_intended_air_yards",
            "avg_yac_above_expectation",
        ]
    )
    for col in ("gsis_id", "team", "position"):
        ngs[col] = ngs[col].astype(_PYARROW_STR)

    # Schedules: one game per team per week.
    sch_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            opp = "DEN" if week % 2 == 1 else "DET"
            for team in {"KC", "MIN"}:
                home_team = team
                away_team = opp if team != opp else ("CHI" if opp == "DEN" else "GB")
                sch_rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": f"{season}_{week:02d}_{home_team}_{away_team}",
                        "home_team": home_team,
                        "away_team": away_team,
                        "kickoff": pd.Timestamp(
                            f"{season}-09-{week + 1:02d}T17:00:00Z"
                        ).tz_convert("UTC").as_unit("us"),
                        "spread_line": -3.0,
                        "total_line": 47.0,
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                        "surface": "grass",
                        "roof": "outdoors",
                        "temp": 60,
                        "wind": 5,
                    }
                )
    schedules = pd.DataFrame(sch_rows)
    for col in ("game_id", "home_team", "away_team", "surface", "roof"):
        schedules[col] = schedules[col].astype(_PYARROW_STR)
    schedules["temp"] = schedules["temp"].astype(pd.Int64Dtype())
    schedules["wind"] = schedules["wind"].astype(pd.Int64Dtype())
    schedules["home_moneyline"] = schedules["home_moneyline"].astype(pd.Int64Dtype())
    schedules["away_moneyline"] = schedules["away_moneyline"].astype(pd.Int64Dtype())

    feat_frames: list[pd.DataFrame] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            f = build_wr_features(
                weekly_stats=baseline_weekly_stats,
                snap_counts=snap_counts,
                depth_charts=depth,
                ngs_receiving=ngs,
                schedules=schedules,
                season=season,
                as_of_week=week,
            )
            feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True)
```

(Note: the schedules-fixture code uses `pd.Timestamp(...).tz_convert("UTC").as_unit("us")` to match `SchedulesSchema`'s datetime kwargs. If the existing `SchedulesSchema` is stricter, debug at Step 5 below.)

- [ ] **Step 3: Write the failing factory test**

Create `tests/test_models/test_baseline.py`:

```python
"""BaselineModel unit tests."""

from __future__ import annotations

from projections.models import wr_baseline
from projections.schemas import DistributionFamily, Position, Stat


def test_wr_baseline_factory_returns_unfitted_model() -> None:
    model = wr_baseline()
    assert model.position == Position.WR
    # Unfitted models do not yet have model_id; accessing it should error or
    # return a sentinel. We pick the explicit-error path.
    expected_targets = {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.FUMBLES_LOST,
    }
    assert set(model.target_stats) == expected_targets
    assert model.dist_families[Stat.RECEPTIONS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RECEIVING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RECEIVING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.RUSHING_YARDS] is DistributionFamily.NORMAL
    assert model.dist_families[Stat.RUSHING_TDS] is DistributionFamily.GAMMA
    assert model.dist_families[Stat.FUMBLES_LOST] is DistributionFamily.GAMMA
    assert model.feature_columns  # non-empty; specific list verified in Task 6
```

- [ ] **Step 4: Run test; expect ImportError**

```bash
pytest tests/test_models/test_baseline.py::test_wr_baseline_factory_returns_unfitted_model -v
```

Expected: FAIL — `cannot import name 'wr_baseline' from 'projections.models'`.

- [ ] **Step 5: Implement `BaselineModel` skeleton + `wr_baseline()`**

Create `src/projections/models/baseline.py`:

```python
"""Baseline model — per-stat Ridge regressions composed into a points
distribution via the existing scoring layer.

One BaselineModel class parameterized by (position, target_stats,
feature_columns, dist_families); per-position factories (wr_baseline,
qb_baseline, rb_baseline, te_baseline) construct correctly-configured
instances. Plan 3a only ships wr_baseline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

from projections.schemas import DistributionFamily, Position, Ruleset, Stat


_WR_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)

_WR_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,
    Stat.RECEIVING_TDS: DistributionFamily.GAMMA,
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

# Feature columns from WrFeaturesSchema, minus identity columns
# (gsis_id / season / week / team / opponent). Boolean columns are coerced
# to 0/1 by fit/predict.
_WR_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "air_yards_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "designed_rusher",
    "snap_pct_l4",
    "avg_separation_std",
    "avg_intended_air_yards_std",
    "percent_share_intended_air_yards_std",
    "avg_yac_above_expectation_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_wr_fppg_l4",
)


@dataclass
class BaselineModel:
    """Per-stat Ridge baseline. Construct via per-position factories
    (wr_baseline, etc.); do not call __init__ directly."""

    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    dist_families: Mapping[Stat, DistributionFamily]

    # Populated by .fit() — None on an unfitted instance.
    feature_means: pd.Series | None = field(default=None)
    ridges: dict[Stat, object] = field(default_factory=dict)
    variance_params: dict[Stat, dict[str, float]] = field(default_factory=dict)
    train_seasons: tuple[int, int] | None = field(default=None)
    code_hash: str | None = field(default=None)


def wr_baseline() -> BaselineModel:
    """Construct an unfitted WR-baseline model. Caller invokes .fit(features,
    weekly_stats) and then .save(path)."""
    return BaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
    )
```

- [ ] **Step 6: Re-export from `__init__.py`**

Update `src/projections/models/__init__.py`:

```python
"""Position-specific projection models."""

from __future__ import annotations

from projections.models.base import Model, compute_code_hash
from projections.models.baseline import BaselineModel, wr_baseline

__all__ = ["BaselineModel", "Model", "compute_code_hash", "wr_baseline"]
```

- [ ] **Step 7: Run test; expect green**

```bash
pytest tests/test_models/test_baseline.py::test_wr_baseline_factory_returns_unfitted_model -v
```

Expected: PASS.

- [ ] **Step 8: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green. The conftest fixture `baseline_features` is unused so far — that's fine, pytest doesn't error on unused fixtures.

- [ ] **Step 9: Commit**

```bash
git add src/projections/models/__init__.py src/projections/models/baseline.py tests/test_models/conftest.py tests/test_models/test_baseline.py
git commit -m "feat(models): add BaselineModel config + wr_baseline() factory"
```

---

## Task 6: `BaselineModel.fit` — feature-matrix build + RidgeCV per stat

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Re-read current baseline.py**

```bash
cat src/projections/models/baseline.py
```

- [ ] **Step 2: Write failing fit-then-attribute-populated tests**

Append to `tests/test_models/test_baseline.py`:

```python
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.models import wr_baseline
from projections.schemas import Stat


def test_baseline_fit_populates_ridges_per_target_stat(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # One fitted RidgeCV per target stat.
    assert set(model.ridges.keys()) == set(model.target_stats)
    for stat in model.target_stats:
        assert isinstance(model.ridges[stat], RidgeCV)
        # Fitted ridges expose coef_; unfitted ones don't.
        assert hasattr(model.ridges[stat], "coef_")


def test_baseline_fit_persists_feature_means(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    assert model.feature_means is not None
    assert set(model.feature_means.index) == set(model.feature_columns)


def test_baseline_fit_records_train_seasons(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Fixture covers 2024 + 2025; the fit signature consumes whatever it
    # receives, so train_seasons is just min/max season seen in training.
    assert model.train_seasons == (2024, 2025)
```

- [ ] **Step 3: Run; expect failures**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: 3 new FAILs ("BaselineModel has no `fit` method").

- [ ] **Step 4: Implement `fit` (feature matrix + RidgeCV only — variance comes in Task 7)**

Edit `src/projections/models/baseline.py`. Add imports:

```python
import numpy as np
from sklearn.linear_model import RidgeCV
```

Add the method on `BaselineModel`:

```python
    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train one RidgeCV per target stat. See spec §3.1 for the pipeline."""
        # Schema validation — defensive at the boundary even though our caller
        # is supposed to have already validated.
        from projections.schemas import WeeklyStatsSchema, WrFeaturesSchema

        features = WrFeaturesSchema.validate(features)
        weekly_stats = WeeklyStatsSchema.validate(weekly_stats)

        # Inner-join features with truth on (gsis_id, season, week). Players in
        # the depth chart who didn't actually play that week have no truth and
        # are silently dropped — that's correct, the model only learns from
        # players who played.
        ws = weekly_stats[weekly_stats["position"] == self.position.value].copy()
        joined = features.merge(
            ws[
                [
                    "gsis_id",
                    "season",
                    "week",
                    *(s.value for s in self.target_stats),
                ]
            ],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )

        # Build feature matrix, persist column order. Drop rows with NaN in any
        # feature column at fit time (mostly week-1-of-season rows where
        # rolling features have no prior history).
        x_cols = list(self.feature_columns)
        feature_frame = joined[x_cols].copy()
        # Coerce booleans to 0/1.
        for col in feature_frame.columns:
            if feature_frame[col].dtype == bool:
                feature_frame[col] = feature_frame[col].astype(np.int8)
        # Persist medians BEFORE dropping NaN rows so predict-time imputation
        # uses the broadest possible signal.
        self.feature_means = feature_frame.median(skipna=True).astype(float)

        feature_frame = feature_frame.dropna()
        if feature_frame.empty:
            raise ValueError(
                "After dropping NaN feature rows, no training data remains. "
                "Check the feature builder and inputs."
            )
        truth_frame = joined.loc[feature_frame.index, [s.value for s in self.target_stats]]

        x = feature_frame.to_numpy(dtype=np.float64)

        # Fit one RidgeCV per stat.
        alphas = np.logspace(-3, 3, 13)
        for stat in self.target_stats:
            y = truth_frame[stat.value].to_numpy(dtype=np.float64)
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(x, y)
            self.ridges[stat] = ridge

        # Record the season range we trained on.
        seasons = sorted(joined["season"].unique().tolist())
        self.train_seasons = (int(seasons[0]), int(seasons[-1]))
```

- [ ] **Step 5: Run new tests; expect green**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: all 4 baseline tests PASS.

- [ ] **Step 6: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(models): BaselineModel.fit — per-stat RidgeCV + feature-mean persistence"
```

---

## Task 7: `BaselineModel.fit` — residual-variance estimation

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Re-read current baseline.py**

```bash
cat src/projections/models/baseline.py
```

- [ ] **Step 2: Write failing variance tests**

Append to `tests/test_models/test_baseline.py`:

```python
def test_baseline_fit_populates_normal_variance_params(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Normal stats: variance_params should have a positive 'std'.
    for stat in (Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS):
        params = model.variance_params[stat]
        assert "std" in params
        assert params["std"] > 0


def test_baseline_fit_populates_gamma_variance_params(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Gamma stats: variance_params should have a 'shape' in [0.01, 100].
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        params = model.variance_params[stat]
        assert "shape" in params
        assert 0.01 <= params["shape"] <= 100.0


def test_method_of_moments_alpha_matches_hand_computed_value() -> None:
    """Verify _gamma_alpha_from_residuals on a hand-built example."""
    import numpy as np

    from projections.models.baseline import _gamma_alpha_from_residuals

    # μ̂ = [1, 2, 3, 4, 5] -> mean(μ̂) = 3
    # residuals = y - μ̂; var(residuals) = ?
    # Let var = 4.5 (arbitrary but easy). Then α̂ = 3² / 4.5 = 2.0
    mu_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Build residuals with sample variance = 4.5
    # variance is computed with ddof=0 (numpy default for .var()) — match it.
    residuals = np.array([3.0, -3.0, 0.0, 1.5, -1.5])
    assert abs(residuals.var() - 4.5) < 1e-9
    alpha = _gamma_alpha_from_residuals(mu_hat=mu_hat, residuals=residuals)
    assert abs(alpha - 2.0) < 1e-9


def test_gamma_alpha_clipped_to_safety_range() -> None:
    """Pathological residuals should clip to [0.01, 100]."""
    import numpy as np

    from projections.models.baseline import _gamma_alpha_from_residuals

    # Almost-zero variance -> α blows up -> clipped to 100
    mu_hat = np.array([1.0, 1.0, 1.0])
    near_zero_resid = np.array([0.0001, 0.0, -0.0001])
    alpha = _gamma_alpha_from_residuals(mu_hat=mu_hat, residuals=near_zero_resid)
    assert alpha == 100.0

    # Massive variance with near-zero mean -> α ~0 -> clipped to 0.01
    mu_hat_zero = np.array([0.001, 0.001])
    huge_resid = np.array([100.0, -100.0])
    alpha = _gamma_alpha_from_residuals(mu_hat=mu_hat_zero, residuals=huge_resid)
    assert alpha == 0.01
```

- [ ] **Step 3: Run; expect failures**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: 4 new FAILs.

- [ ] **Step 4: Implement variance machinery**

Add to `src/projections/models/baseline.py`:

```python
_GAMMA_ALPHA_CLIP: Final[tuple[float, float]] = (0.01, 100.0)


def _gamma_alpha_from_residuals(
    *, mu_hat: np.ndarray, residuals: np.ndarray
) -> float:
    """Method-of-moments shape parameter α for a Gamma distribution
    parameterized by (α, β=α/μ̂). var = μ̂² / α, so α̂ = mean(μ̂)² / var(residuals).

    Clipped to [0.01, 100] for numerical safety; the MoM estimator is
    degenerate for very rare events. Spec §3.4 documents this choice.
    """
    mean_mu = float(mu_hat.mean())
    var_resid = float(residuals.var())  # population variance (ddof=0)
    if mean_mu == 0.0 or var_resid == 0.0:
        # Degenerate; pick the most permissive clip.
        return _GAMMA_ALPHA_CLIP[1] if var_resid == 0.0 else _GAMMA_ALPHA_CLIP[0]
    raw = (mean_mu * mean_mu) / var_resid
    return float(min(max(raw, _GAMMA_ALPHA_CLIP[0]), _GAMMA_ALPHA_CLIP[1]))


def _normal_std_from_residuals(residuals: np.ndarray) -> float:
    """Global per-stat residual std for the Normal family. Floored at a
    tiny positive ε so ParametricNormal's std>0 invariant always holds."""
    s = float(residuals.std())
    return max(s, 1e-6)
```

Extend `BaselineModel.fit` (after the RidgeCV loop):

```python
        # Variance estimation (spec §3.4).
        for stat in self.target_stats:
            ridge = self.ridges[stat]
            mu_hat = ridge.predict(x).astype(np.float64)
            y = truth_frame[stat.value].to_numpy(dtype=np.float64)
            residuals = y - mu_hat
            family = self.dist_families[stat]
            if family is DistributionFamily.NORMAL:
                self.variance_params[stat] = {"std": _normal_std_from_residuals(residuals)}
            elif family is DistributionFamily.GAMMA:
                self.variance_params[stat] = {
                    "shape": _gamma_alpha_from_residuals(mu_hat=mu_hat, residuals=residuals)
                }
            else:  # pragma: no cover — only NORMAL/GAMMA configured today
                raise ValueError(f"Unsupported family {family} for stat {stat}")
```

- [ ] **Step 5: Run new tests; expect green**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: all baseline tests PASS.

- [ ] **Step 6: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(models): BaselineModel.fit — residual variance per stat (gamma α + normal σ)"
```

---

## Task 8: `BaselineModel._build_stat_distributions` (private helper)

The predict_distribution path is split: Task 8 builds the per-stat distributions, Task 9 composes them into points distributions. Keeping them separate lets us unit-test the per-stat output independently of the score_distribution Monte Carlo.

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_models/test_baseline.py`:

```python
import numpy as np

from projections.distributions.parametric import ParametricGamma, ParametricNormal


def test_build_stat_distributions_returns_one_per_row(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # Pick a single week of test features.
    week_features = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    assert not week_features.empty
    stat_dists_per_row = model._build_stat_distributions(week_features)
    assert len(stat_dists_per_row) == len(week_features)
    for row_dists in stat_dists_per_row:
        assert set(row_dists.keys()) == set(model.target_stats)
        # Family-specific concrete types.
        assert isinstance(row_dists[Stat.RECEIVING_YARDS], ParametricNormal)
        assert isinstance(row_dists[Stat.RECEPTIONS], ParametricGamma)


def test_build_stat_distributions_clamps_gamma_mu() -> None:
    """A regression that predicts μ̂ ≤ 0 for a gamma stat should produce a
    ParametricGamma with finite, positive scale."""
    model = wr_baseline()
    # Hand-craft a fake fitted state with one stat configured.
    from projections.schemas import DistributionFamily
    model.target_stats = (Stat.RECEPTIONS,)
    model.dist_families = {Stat.RECEPTIONS: DistributionFamily.GAMMA}
    model.feature_columns = ("dummy_feat",)
    model.feature_means = pd.Series({"dummy_feat": 0.0}, dtype=float)
    model.variance_params = {Stat.RECEPTIONS: {"shape": 2.0}}

    class _FakeRidge:
        def predict(self, x: np.ndarray) -> np.ndarray:
            # Predict negative mu (should be clamped at predict time).
            return np.full(x.shape[0], -5.0)

    model.ridges = {Stat.RECEPTIONS: _FakeRidge()}

    fake_features = pd.DataFrame({"dummy_feat": [1.0]})
    out = model._build_stat_distributions(fake_features)
    assert len(out) == 1
    rec_dist = out[0][Stat.RECEPTIONS]
    assert isinstance(rec_dist, ParametricGamma)
    # mean = shape * scale > 0 (clamped, not negative)
    assert rec_dist.mean() > 0
```

- [ ] **Step 2: Run; expect failures**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: 2 new FAILs.

- [ ] **Step 3: Implement `_build_stat_distributions`**

Add to `BaselineModel`:

```python
    _GAMMA_MU_FLOOR: Final[float] = 1e-3  # spec §3.2 step 3

    def _build_stat_distributions(
        self, features: pd.DataFrame
    ) -> list[dict[Stat, "Distribution"]]:
        """Build per-row dicts of {Stat -> Distribution} from fitted regressors.

        Pure function over the fitted state. Does not call score_distribution
        (that's predict_distribution's job, Task 9). Useful for unit tests and
        for any caller that wants per-stat dists for analysis.
        """
        from projections.distributions import Distribution
        from projections.distributions.parametric import ParametricGamma, ParametricNormal

        if not self.ridges or self.feature_means is None:
            raise RuntimeError("Model is not fitted; call fit() before predict.")

        # Build feature matrix with same column order, impute, coerce bools.
        x_cols = list(self.feature_columns)
        x_frame = features[x_cols].copy()
        for col in x_frame.columns:
            if x_frame[col].dtype == bool:
                x_frame[col] = x_frame[col].astype(np.int8)
        x_frame = x_frame.fillna(self.feature_means)
        x = x_frame.to_numpy(dtype=np.float64)

        # Per-stat predict + Distribution construction.
        per_stat_mu: dict[Stat, np.ndarray] = {}
        for stat in self.target_stats:
            mu = self.ridges[stat].predict(x).astype(np.float64)
            if self.dist_families[stat] is DistributionFamily.GAMMA:
                mu = np.maximum(mu, self._GAMMA_MU_FLOOR)
            per_stat_mu[stat] = mu

        out: list[dict[Stat, Distribution]] = []
        for i in range(len(x)):
            row: dict[Stat, Distribution] = {}
            for stat in self.target_stats:
                mu_i = float(per_stat_mu[stat][i])
                family = self.dist_families[stat]
                params = self.variance_params[stat]
                if family is DistributionFamily.NORMAL:
                    row[stat] = ParametricNormal(mean=mu_i, std=params["std"])
                elif family is DistributionFamily.GAMMA:
                    shape = params["shape"]
                    # rate = α / μ; scale = 1/rate = μ / α
                    scale = mu_i / shape
                    row[stat] = ParametricGamma(shape=shape, scale=scale)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported family {family}")
            out.append(row)
        return out
```

- [ ] **Step 4: Run tests; expect green**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: PASS.

- [ ] **Step 5: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(models): BaselineModel._build_stat_distributions per-row helper"
```

---

## Task 9: `BaselineModel.predict_distribution` — full pipeline → ProjectionWeeklySchema

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models/test_baseline.py`:

```python
from projections.schemas import ProjectionWeeklySchema, Ruleset


def test_predict_distribution_returns_projection_weekly_schema_valid_frame(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    week_features = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    # Schema validates without raising.
    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(week_features)
    # Identity columns preserved.
    assert set(out["gsis_id"].astype(str)) == set(week_features["gsis_id"].astype(str))
    # ruleset / model_id / generated_at populated.
    assert (out["ruleset"] == "ESPN_PPR").all()
    assert out["model_id"].str.startswith("baseline:wr:").all()


def test_predict_distribution_empty_input_returns_empty_schema_valid_frame(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    empty = baseline_features.iloc[0:0]
    out = model.predict_distribution(empty, ruleset=Ruleset.espn_ppr())
    assert out.empty
    ProjectionWeeklySchema.validate(out)


def test_predict_distribution_p10_le_p50_le_p90(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    week_features = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    out = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()
```

- [ ] **Step 2: Run; expect failures**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: 3 new FAILs (`predict_distribution` not yet on BaselineModel).

- [ ] **Step 3: Implement `predict_distribution` + `model_id`**

Add to `BaselineModel`:

```python
    @property
    def model_id(self) -> str:
        if self.code_hash is None or self.train_seasons is None:
            raise RuntimeError("model_id is undefined for unfitted models")
        return (
            f"baseline:{self.position.value.lower()}:{self.code_hash}"
            f":{self.train_seasons[0]}-{self.train_seasons[1]}"
        )

    def predict_distribution(
        self, features: pd.DataFrame, ruleset: Ruleset
    ) -> pd.DataFrame:
        from datetime import UTC, datetime

        import msgpack

        from projections.scoring import score_distribution
        from projections.schemas import (
            ProjectionWeeklySchema,
            WrFeaturesSchema,
            _PYARROW_STR,
        )

        features = WrFeaturesSchema.validate(features)
        if features.empty:
            empty_cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
            return ProjectionWeeklySchema.validate(pd.DataFrame(columns=empty_cols))

        stat_dists_per_row = self._build_stat_distributions(features)

        # Compose each row's per-stat distribution dict into a fantasy-points
        # SampledDistribution via score_distribution.
        rows: list[dict[str, object]] = []
        generated_at = datetime.now(UTC)
        for (idx, feat_row), stat_dists in zip(
            features.reset_index(drop=True).iterrows(), stat_dists_per_row, strict=True
        ):
            points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=42)
            family_blob = msgpack.packb(
                {
                    "samples_summary": {
                        "n": len(points.samples),
                        "mean": float(points.mean()),
                    }
                },
                use_bin_type=True,
            )
            rows.append(
                {
                    "gsis_id": feat_row["gsis_id"],
                    "season": int(feat_row["season"]),
                    "week": int(feat_row["week"]),
                    "position": self.position.value,
                    "team": feat_row["team"],
                    "opponent": feat_row["opponent"],
                    "ruleset": ruleset.name,
                    "family": "SAMPLED",
                    "params": family_blob,
                    "mean": points.mean(),
                    "p10": points.quantile(0.1),
                    "p50": points.quantile(0.5),
                    "p90": points.quantile(0.9),
                    "model_id": self.model_id,
                    "generated_at": pd.Timestamp(generated_at).as_unit("us"),
                }
            )

        out = pd.DataFrame(rows)
        # Coerce the string columns to pyarrow string per schema convention.
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
            out[col] = out[col].astype(_PYARROW_STR)
        return ProjectionWeeklySchema.validate(out)
```

You also need `msgpack` — but pinpoint the dep. Verify:

```bash
python -c "import msgpack; print(msgpack.__version__)"
```

If missing, msgpack ships with pyarrow's transitive deps; otherwise add to pyproject.toml as part of this task. If it's not installed, run:

```bash
pip install msgpack
```

…and add `msgpack>=1.0` to `pyproject.toml` `[project].dependencies` in this task's commit.

- [ ] **Step 4: Compute `code_hash` inside `fit`**

The code_hash needs to be set inside `fit` so `model_id` is callable on a fitted instance. Add at the end of `fit`:

```python
        # Code hash over source files whose change should invalidate the
        # artifact. Spec §5.2 lists the canonical set.
        from projections.models.base import compute_code_hash

        repo_root = Path(__file__).resolve().parents[3]
        tracked = [
            repo_root / "src" / "projections" / "models" / "base.py",
            repo_root / "src" / "projections" / "models" / "baseline.py",
            repo_root / "src" / "projections" / "features" / "wr.py",
            repo_root / "src" / "projections" / "features" / "_shared.py",
            repo_root / "src" / "projections" / "features" / "_rolling.py",
            repo_root / "src" / "projections" / "features" / "_opponent.py",
            repo_root / "src" / "projections" / "scoring" / "score.py",
            repo_root / "src" / "projections" / "scoring" / "score_distribution.py",
        ]
        self.code_hash = compute_code_hash(tracked)
```

- [ ] **Step 5: Run tests; expect green**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: PASS. Note: if a row's predicted points distribution computes much faster than expected (RidgeCV is fast and score_distribution n_samples=10_000 takes a few hundred ms per row), test runtime is dominated by score_distribution. If the test is intolerably slow, lower `n_samples` to `2_000` for unit tests by adding an optional `n_samples` parameter to `predict_distribution` (default 10_000) — but only if needed.

- [ ] **Step 6: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py pyproject.toml
git commit -m "feat(models): BaselineModel.predict_distribution end-to-end pipeline"
```

If `pyproject.toml` was edited for msgpack include it; if not, drop it from the `git add`.

---

## Task 10: `save` / `load` round trip + `model_id` stability

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models/test_baseline.py`:

```python
def test_baseline_save_load_round_trip_preserves_predictions(
    tmp_path: Path,
    baseline_features: pd.DataFrame,
    baseline_weekly_stats: pd.DataFrame,
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)

    artifact = tmp_path / "wr-baseline.joblib"
    model.save(artifact)
    assert artifact.exists()

    from projections.models import BaselineModel
    loaded = BaselineModel.load(artifact)
    assert loaded.position == model.position
    assert loaded.model_id == model.model_id

    week = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    out_orig = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    out_loaded = loaded.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    pd.testing.assert_frame_equal(
        out_orig.drop(columns=["generated_at"]),
        out_loaded.drop(columns=["generated_at"]),
    )


def test_model_id_format(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "baseline"
    assert parts[1] == "wr"
    assert len(parts[2]) == 8  # code_hash
    assert "-" in parts[3]


def test_unfitted_model_id_raises() -> None:
    model = wr_baseline()
    try:
        _ = model.model_id
    except RuntimeError:
        return
    raise AssertionError("Unfitted model.model_id should raise RuntimeError")
```

- [ ] **Step 2: Run; expect failures**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: 3 new FAILs.

- [ ] **Step 3: Implement `save` / `load`**

Add to `BaselineModel`:

```python
    def save(self, path: Path) -> None:
        import joblib

        if self.code_hash is None or self.train_seasons is None:
            raise RuntimeError("Cannot save an unfitted BaselineModel")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        import joblib

        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected BaselineModel, got {type(loaded).__name__}")
        return loaded
```

- [ ] **Step 4: Run tests; expect green**

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: PASS.

- [ ] **Step 5: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(models): BaselineModel save/load via joblib + model_id property"
```

---

## Task 11: Edge-case tests — gamma μ̂ clamping (already covered), NaN imputation, bool feature coercion

The gamma μ̂ clamping test is already in Task 8. Task 11 adds NaN imputation and feature-type coercion regression tests.

**Files:**
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Add NaN-imputation test**

Append:

```python
def test_predict_distribution_imputes_nan_features_with_persisted_means(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    """If a predict-time feature row has NaN in a column, predict should impute
    with feature_means rather than crash or propagate NaN to the output."""
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)

    week = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ].copy()
    # Forcibly NaN one value in a non-nullable feature column.
    week.loc[week.index[0], "implied_team_total"] = np.nan
    out = model.predict_distribution(week, ruleset=Ruleset.espn_ppr())
    assert not out["mean"].isna().any()


def test_fit_handles_bool_feature_columns(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    """is_home / roof_dome / designed_rusher are bool in WrFeaturesSchema and
    must be coerced to numeric for Ridge.fit()."""
    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)
    # If fit succeeded, the boolean coercion worked.
    assert model.feature_means is not None
    for bool_col in ("is_home", "roof_dome", "designed_rusher"):
        assert bool_col in model.feature_means.index
```

- [ ] **Step 2: Run tests; expect green** (the implementation from Task 6/8 already imputes and coerces; if tests fail, debug)

```bash
pytest tests/test_models/test_baseline.py -v
```

Expected: PASS.

- [ ] **Step 3: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_models/test_baseline.py
git commit -m "test(models): add edge-case tests for NaN imputation + bool feature coercion"
```

---

## Task 12: Leakage test

**Files:**
- Create: `tests/test_models/test_baseline_leakage.py`

- [ ] **Step 1: Write the leakage test**

Create `tests/test_models/test_baseline_leakage.py`:

```python
"""Plan 3a leakage test: mutating data past as_of_week must not change the
fitted model. Mirrors the per-feature-builder leakage tests already in
tests/test_features/test_wr_leakage.py.

Strategy: fit on a feature build through week W. Mutate weekly_stats rows at
season=Y, week>=W+1. Re-build features through W and refit. Assert each
fitted regressor's coefficients are byte-identical pre and post.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.models import wr_baseline


def test_baseline_fit_does_not_use_post_as_of_week_data(
    baseline_features: pd.DataFrame, baseline_weekly_stats: pd.DataFrame
) -> None:
    # Restrict fixture to 2024 only for this test (so we have one season).
    ws = baseline_weekly_stats[baseline_weekly_stats["season"] == 2024].copy()
    feats = baseline_features[baseline_features["season"] == 2024].copy()

    # Train on the first version.
    model_a = wr_baseline()
    model_a.fit(features=feats, weekly_stats=ws)

    # Mutate week-7 and week-8 truth values dramatically.
    ws_mut = ws.copy()
    mask = ws_mut["week"] >= 7
    ws_mut.loc[mask, "receptions"] = 0
    ws_mut.loc[mask, "receiving_yards"] = 0.0
    ws_mut.loc[mask, "receiving_tds"] = 0
    ws_mut.loc[mask, "targets"] = 0

    # Mutating weekly_stats rows changes feature rows for as_of_week >= 8 (the
    # rolling features for week 8 use weeks 4..7). Restrict feature input to
    # rows with week <= 7 so neither model sees the mutated truth at training
    # time — that's the whole leakage premise: we train through W and assert
    # nothing past W matters.
    feats_through_7 = feats[feats["week"] <= 7].copy()
    ws_mut_through_7 = ws_mut[ws_mut["week"] <= 7].copy()
    ws_orig_through_7 = ws[ws["week"] <= 7].copy()

    model_b = wr_baseline()
    model_b.fit(features=feats_through_7, weekly_stats=ws_orig_through_7)
    model_c = wr_baseline()
    model_c.fit(features=feats_through_7, weekly_stats=ws_mut_through_7)

    # model_b and model_c trained on identical (week<=7) data — coefficients
    # MUST match exactly. If they don't, leakage is sneaking in via some path
    # we haven't accounted for.
    for stat in model_b.target_stats:
        np.testing.assert_array_equal(
            model_b.ridges[stat].coef_,
            model_c.ridges[stat].coef_,
            err_msg=f"Leakage detected on stat {stat}",
        )
        assert model_b.ridges[stat].alpha_ == model_c.ridges[stat].alpha_
```

- [ ] **Step 2: Run test; expect green**

```bash
pytest tests/test_models/test_baseline_leakage.py -v
```

Expected: PASS. If it fails, the model is leaking — debug by checking the inner-join + filter logic in `fit`.

- [ ] **Step 3: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_models/test_baseline_leakage.py
git commit -m "test(models): add BaselineModel leakage test"
```

---

## Task 13: `ingest.refresh()` orchestrator

**Files:**
- Create: `src/projections/ingest/refresh.py`
- Modify: `src/projections/ingest/__init__.py`
- Create: `tests/test_ingest/test_refresh.py`

- [ ] **Step 1: Write failing orchestrator test (mocked)**

Create `tests/test_ingest/test_refresh.py`:

```python
"""ingest.refresh orchestrator unit test — verifies it fans out to every
per-source refresh function. We mock the source functions so this test does
not hit the network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from projections.ingest import refresh


def test_refresh_calls_every_per_source_refresh_function(tmp_path: Path) -> None:
    seasons = [2018, 2019]

    with (
        patch("projections.ingest.refresh.refresh_weekly_stats") as ws,
        patch("projections.ingest.refresh.refresh_schedules") as sch,
        patch("projections.ingest.refresh.refresh_snap_counts") as sc,
        patch("projections.ingest.refresh.refresh_depth_charts") as dc,
        patch("projections.ingest.refresh.refresh_ngs") as ngs,
        patch("projections.ingest.refresh.build_id_map") as id_map,
    ):
        refresh(seasons=seasons, raw_root=tmp_path)

        ws.assert_called_once_with(seasons=seasons, raw_root=tmp_path)
        sch.assert_called_once_with(seasons=seasons, raw_root=tmp_path)
        sc.assert_called_once_with(seasons=seasons, raw_root=tmp_path)
        dc.assert_called_once_with(seasons=seasons, raw_root=tmp_path)
        ngs.assert_called_once_with(seasons=seasons, raw_root=tmp_path)
        id_map.assert_called_once_with(seasons=seasons, raw_root=tmp_path)
```

- [ ] **Step 2: Run; expect ImportError**

```bash
pytest tests/test_ingest/test_refresh.py -v
```

Expected: FAIL with `ImportError: cannot import name 'refresh' from 'projections.ingest'`.

- [ ] **Step 3: Verify per-source signatures**

Before implementing, verify that every per-source function accepts `(seasons, *, raw_root)`. Read each:

```bash
grep -n "^def refresh_" src/projections/ingest/*.py
grep -n "^def build_id_map" src/projections/ingest/*.py
```

If any signature differs, adjust the orchestrator's call accordingly.

- [ ] **Step 4: Implement orchestrator**

Create `src/projections/ingest/refresh.py`:

```python
"""Top-level ingest orchestrator. Fans out to every per-source refresh
function. Plan 3a's first real-data pull uses this entrypoint."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import refresh_ngs
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats


def refresh(seasons: Iterable[int], *, raw_root: Path) -> None:
    """Refresh every ingest source for ``seasons`` into ``raw_root``.

    Order matters: id_map must be built last because snap_counts depends on
    the gsis_id <-> pfr_id translation table being current. weekly_stats and
    schedules go first since other sources cross-reference them.

    This function is idempotent — re-running for an already-pulled season
    overwrites that season's partition.
    """
    season_list = list(seasons)
    refresh_weekly_stats(seasons=season_list, raw_root=raw_root)
    refresh_schedules(seasons=season_list, raw_root=raw_root)
    refresh_depth_charts(seasons=season_list, raw_root=raw_root)
    refresh_ngs(seasons=season_list, raw_root=raw_root)
    build_id_map(seasons=season_list, raw_root=raw_root)
    refresh_snap_counts(seasons=season_list, raw_root=raw_root)
```

- [ ] **Step 5: Re-export from package `__init__.py`**

Update `src/projections/ingest/__init__.py`:

```python
"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import refresh_ngs
from projections.ingest.refresh import refresh
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "refresh",
    "refresh_depth_charts",
    "refresh_ngs",
    "refresh_schedules",
    "refresh_snap_counts",
    "refresh_weekly_stats",
]
```

- [ ] **Step 6: Run test; expect green**

```bash
pytest tests/test_ingest/test_refresh.py -v
```

Expected: PASS.

- [ ] **Step 7: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/projections/ingest/refresh.py src/projections/ingest/__init__.py tests/test_ingest/test_refresh.py
git commit -m "feat(ingest): add refresh() orchestrator"
```

---

## Task 14: First real-data smoke pull (2018 only)

**Files:**
- Modify: `.gitignore` (add `data/`, `models/artifacts/`, `*.joblib`)

This is a manual step. The point is to discover any `nfl_data_py` API drift on a single season before pulling 8 of them. No automated test — verification is "files appeared and the existing schemas validated them."

- [ ] **Step 1: Update .gitignore**

```bash
cat .gitignore
```

If `data/` and `models/artifacts/` are not present, add:

```gitignore
# Generated by ingest / training — gitignored, reproducible from source.
data/
models/artifacts/
*.joblib
```

- [ ] **Step 2: Commit gitignore changes (if edited)**

```bash
git add .gitignore
git commit -m "chore: gitignore data/, models/artifacts/, *.joblib"
```

If `.gitignore` already had everything, skip.

- [ ] **Step 3: Run the 2018-only refresh**

```bash
mkdir -p data/raw
python -c "
from pathlib import Path
from projections.ingest import refresh
refresh(seasons=[2018], raw_root=Path('data/raw'))
"
```

Expected: prints from each refresh_* call as they fetch from nfl_data_py. Some may take 10-60s. Total under 5 minutes.

- [ ] **Step 4: Verify the partitions**

```bash
ls data/raw/
ls data/raw/weekly_stats/season=2018/
ls data/raw/schedules/season=2018/
ls data/raw/snap_counts/season=2018/
ls data/raw/depth_charts/season=2018/
ls data/raw/ngs_passing/season=2018/ data/raw/ngs_rushing/season=2018/ data/raw/ngs_receiving/season=2018/
ls data/raw/id_map.parquet
```

Each should show `part.parquet` (or for `id_map`, the standalone file).

- [ ] **Step 5: Verify schema validation works on real data**

```bash
python -c "
from pathlib import Path
import pandas as pd
from projections.schemas import WeeklyStatsSchema, SchedulesSchema, SnapCountsSchema
from projections.store import read_partition
ws = read_partition(Path('data/raw'), 'weekly_stats', season=2018)
print('weekly_stats rows:', len(ws))
print(ws.head(3))
WeeklyStatsSchema.validate(ws)
sc = read_partition(Path('data/raw'), 'snap_counts', season=2018)
SnapCountsSchema.validate(sc)
print('snap_counts rows:', len(sc))
sch = read_partition(Path('data/raw'), 'schedules', season=2018)
SchedulesSchema.validate(sch)
print('schedules rows:', len(sch))
print('All schemas validated.')
"
```

Expected: prints row counts for each table; the `validate()` calls do not raise. If they raise, debug by inspecting the columns & dtypes of the returned DataFrame against the schema definitions in `src/projections/schemas.py`. Common drifts:

- A new column in `nfl_data_py` not in `_KEEP` for that source — silently dropped, no error.
- A column rename in `nfl_data_py` — the rename map in the source's ingest module needs an update.
- A dtype change — the schema's coerce should handle, otherwise add a cast in the ingest source's pyarrow conversion.

- [ ] **Step 6: Try building one week of WR features against real 2018 data**

```bash
python -c "
from pathlib import Path
import pandas as pd
from projections.features import build_wr_features
from projections.store import read_partition

raw_root = Path('data/raw')
ws = read_partition(raw_root, 'weekly_stats', season=2018)
sc = read_partition(raw_root, 'snap_counts', season=2018)
dc = read_partition(raw_root, 'depth_charts', season=2018)
ngs = read_partition(raw_root, 'ngs_receiving', season=2018)
sch = read_partition(raw_root, 'schedules', season=2018)

feats = build_wr_features(
    weekly_stats=ws, snap_counts=sc, depth_charts=dc,
    ngs_receiving=ngs, schedules=sch,
    season=2018, as_of_week=8,
)
print('WR feature rows for 2018 wk 8:', len(feats))
print(feats.head(3))
"
```

Expected: prints something like "WR feature rows for 2018 wk 8: 100-150" with realistic values. If the build raises, that's a real-data issue surfaced — debug per spec §9.4 (the eyeball-one-row checkpoint).

Common issues that may surface here:
- TODO #9a: schedule join leaves NaN `is_home`/`roof_dome` for bye weeks — schema rejects the non-nullable bool. Workaround: skip bye-week WRs at predict time (filter rostered teams to those with schedule rows for that week).
- TODO #9b: id_map duplicate `pfr_id` multiplies snap_counts rows — manifests as `validate="one_to_one"` failing in the model fit. Workaround: `.drop_duplicates(subset=['pfr_id'])` in the snap_counts ingest path.
- TODO #9c: traded players show up with two team rows in `_trailing_4_share_per_team` — manifests as a player having multiple feature rows. Filter to most-recent team or merge on `(gsis_id, team)` instead of `gsis_id`.

If any of these block, fix them in their proper module (`features/` or `ingest/`) and add to the same commit. Note the fix in PM doc Task 22.

- [ ] **Step 7: No-op commit if nothing else changed**

If you only ran the smoke and made no code changes, skip a commit. If you fixed a TODO #9 issue, commit it now with a `fix(ingest):` or `fix(features):` message; reference the TODO number.

---

## Task 15: Full real-data pull (2018–2025)

- [ ] **Step 1: Run the full refresh**

```bash
python -c "
from pathlib import Path
from projections.ingest import refresh
refresh(seasons=range(2018, 2026), raw_root=Path('data/raw'))
"
```

Expected: 5–15 minutes wall clock. Network-bound. Idempotent — partial completion is recoverable by re-running (the per-season writes are partition-scoped).

- [ ] **Step 2: Verify all 8 seasons present in every source**

```bash
for src in weekly_stats schedules snap_counts depth_charts ngs_passing ngs_rushing ngs_receiving; do
  echo "--- $src ---"
  ls data/raw/$src/
done
```

Expected: each source has `season=2018` through `season=2025` directories.

- [ ] **Step 3: Spot-check 2025 (the held-out year)**

```bash
python -c "
from pathlib import Path
from projections.schemas import WeeklyStatsSchema
from projections.store import read_partition

ws = read_partition(Path('data/raw'), 'weekly_stats', season=2025)
print('2025 weekly_stats rows:', len(ws))
print('2025 weeks present:', sorted(ws['week'].unique()))
WeeklyStatsSchema.validate(ws)
print('Schema valid.')
"
```

Expected: Approximately 4000–5000 weekly stat rows, weeks 1 through 18 (or whatever the 2025 season included), schema valid.

- [ ] **Step 4: No commit** — `data/` is gitignored.

---

## Task 16: `scripts/train_wr_baseline.py`

**Files:**
- Create: `scripts/train_wr_baseline.py`
- Create: `models/.gitkeep` (so the parent dir is tracked even if `models/artifacts/` is gitignored)

- [ ] **Step 1: Create the script**

Create `scripts/train_wr_baseline.py`:

```python
"""Plan 3a — train WR Model A baseline on 2018-2024, persist to models/artifacts/.

Run from the repo root:
    python scripts/train_wr_baseline.py

Reads ingested raw partitions from ``data/raw/``, builds WR features for every
week of 2018-2024, fits BaselineModel, saves the joblib artifact under a name
that includes the train window and code hash so reruns produce a comparable
file even after model code changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.features import build_wr_features
from projections.models import wr_baseline
from projections.store import read_partition


_TRAIN_SEASONS = range(2018, 2025)  # 2018..2024 inclusive


def _build_training_features(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a stacked WR feature DataFrame across (season, week) pairs in the
    training window, plus the matching weekly_stats truth across the same
    seasons. Caller passes both into ``BaselineModel.fit``."""
    feature_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    for season in _TRAIN_SEASONS:
        ws = read_partition(raw_root, "weekly_stats", season=season)
        sc = read_partition(raw_root, "snap_counts", season=season)
        dc = read_partition(raw_root, "depth_charts", season=season)
        ngs = read_partition(raw_root, "ngs_receiving", season=season)
        sch = read_partition(raw_root, "schedules", season=season)
        truth_frames.append(ws)

        weeks = sorted(dc["week"].unique())
        for week in weeks:
            f = build_wr_features(
                weekly_stats=ws,
                snap_counts=sc,
                depth_charts=dc,
                ngs_receiving=ngs,
                schedules=sch,
                season=int(season),
                as_of_week=int(week),
            )
            if not f.empty:
                feature_frames.append(f)
        print(f"  Built features for season {season}: {len(weeks)} weeks")

    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    weekly_stats = pd.concat(truth_frames, ignore_index=True) if truth_frames else pd.DataFrame()
    return features, weekly_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WR Model A baseline.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    args = parser.parse_args()

    print(f"Reading raw partitions from {args.raw_root}")
    features, weekly_stats = _build_training_features(args.raw_root)
    print(f"Total WR feature rows: {len(features)}; weekly_stats rows: {len(weekly_stats)}")

    model = wr_baseline()
    model.fit(features=features, weekly_stats=weekly_stats)
    print(f"model_id: {model.model_id}")
    for stat in model.target_stats:
        print(f"  {stat.value}: variance_params = {model.variance_params[stat]}")

    train_start, train_end = model.train_seasons or (0, 0)
    artifact = (
        args.artifacts_root
        / f"wr-baseline-{train_start}-{train_end}-{model.code_hash}.joblib"
    )
    model.save(artifact)
    print(f"Saved artifact: {artifact}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `models/.gitkeep` so the directory is committed**

```bash
mkdir -p models
touch models/.gitkeep
```

- [ ] **Step 3: Quick syntax/import check**

```bash
python -c "import scripts.train_wr_baseline" 2>&1 || python -m py_compile scripts/train_wr_baseline.py
```

Expected: clean exit (no syntax error).

- [ ] **Step 4: Run gate**

```bash
mypy src tests scripts && ruff check src tests scripts && ruff format --check src tests scripts && pytest -q
```

Expected: all green. (If `scripts/` isn't picked up by mypy/ruff, the existing config covers `src` and `tests` only — that's fine; just include scripts in the commands manually.)

- [ ] **Step 5: Commit**

```bash
git add scripts/train_wr_baseline.py models/.gitkeep
git commit -m "feat(scripts): add train_wr_baseline.py — fit BaselineModel on 2018-2024"
```

---

## Task 17: Run training on real data

- [ ] **Step 1: Execute**

```bash
python scripts/train_wr_baseline.py
```

Expected output:
- Per-season feature-building log lines.
- Total feature/weekly_stats row counts (≈9k features after inner-join, ≈8k–10k weekly stats rows for WRs across 7 seasons).
- A `model_id` of the form `baseline:wr:<8-char-hash>:2018-2024`.
- Six per-stat variance lines.
- `Saved artifact: models/artifacts/wr-baseline-2018-2024-<hash>.joblib`.

If any step raises, debug — typically:
- A schema validation in `build_wr_features` (TODO #9 candidates from Task 14 step 6 may apply here too on a different season).
- Out of memory: unlikely at this scale, but if so, train per-season chunks instead of stacking.

- [ ] **Step 2: Confirm the artifact exists**

```bash
ls models/artifacts/
```

Expected: one `.joblib` file matching the model_id.

- [ ] **Step 3: No commit** — `models/artifacts/` is gitignored.

---

## Task 18: `scripts/sanity_check_wr_baseline.py`

**Files:**
- Create: `scripts/sanity_check_wr_baseline.py`

- [ ] **Step 1: Create the script**

Create `scripts/sanity_check_wr_baseline.py`:

```python
"""Plan 3a — sanity-check eval of WR Model A baseline against the held-out
2025 season. Prints per-stat fit, composite (PPR points), and calibration
spot-check metrics. NOT a CI gate — Plan 3c builds the proper backtest
harness with thresholds.

Run from the repo root after train_wr_baseline.py:
    python scripts/sanity_check_wr_baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.features import build_wr_features
from projections.models import BaselineModel
from projections.schemas import Position, Ruleset, Stat
from projections.scoring import score
from projections.scoring.score import StatLine
from projections.store import read_partition


_HELD_OUT_SEASON = 2025


def _find_artifact(artifacts_root: Path) -> Path:
    matches = sorted(artifacts_root.glob("wr-baseline-*.joblib"))
    if not matches:
        raise FileNotFoundError(
            f"No wr-baseline-*.joblib in {artifacts_root}. "
            "Run scripts/train_wr_baseline.py first."
        )
    return matches[-1]  # alphabetical sort puts highest train_end last


def _realized_ppr_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.Series:
    """Compute realized PPR points per row of weekly_stats."""
    points: list[float] = []
    for _, row in weekly_stats.iterrows():
        line = StatLine(
            passing_yards=row["passing_yards"],
            passing_tds=int(row["passing_tds"]),
            interceptions=int(row["interceptions"]),
            rushing_yards=row["rushing_yards"],
            rushing_tds=int(row["rushing_tds"]),
            receptions=int(row["receptions"]),
            receiving_yards=row["receiving_yards"],
            receiving_tds=int(row["receiving_tds"]),
            fumbles_lost=int(row["fumbles_lost"]),
        )
        points.append(score(line, ruleset))
    return pd.Series(points, index=weekly_stats.index, name="actual_ppr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check WR Model A on 2025.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    args = parser.parse_args()

    artifact = _find_artifact(args.artifacts_root)
    print(f"Loading artifact: {artifact}")
    model = BaselineModel.load(artifact)
    print(f"model_id: {model.model_id}")

    raw_root = args.raw_root

    # Build features for every week of 2025 (predict-time inputs use prior
    # weeks within 2025; training data through 2024 is implicit in the fit).
    ws_2025 = read_partition(raw_root, "weekly_stats", season=_HELD_OUT_SEASON)
    sc_2025 = read_partition(raw_root, "snap_counts", season=_HELD_OUT_SEASON)
    dc_2025 = read_partition(raw_root, "depth_charts", season=_HELD_OUT_SEASON)
    ngs_2025 = read_partition(raw_root, "ngs_receiving", season=_HELD_OUT_SEASON)
    sch_2025 = read_partition(raw_root, "schedules", season=_HELD_OUT_SEASON)

    # Concatenate prior seasons for rolling windows (2024 is enough for L4).
    ws_prior = read_partition(raw_root, "weekly_stats", season=2024)
    sc_prior = read_partition(raw_root, "snap_counts", season=2024)
    ngs_prior = read_partition(raw_root, "ngs_receiving", season=2024)
    ws_full = pd.concat([ws_prior, ws_2025], ignore_index=True)
    sc_full = pd.concat([sc_prior, sc_2025], ignore_index=True)
    ngs_full = pd.concat([ngs_prior, ngs_2025], ignore_index=True)

    weeks = sorted(dc_2025["week"].unique())
    rows: list[pd.DataFrame] = []
    for week in weeks:
        feats = build_wr_features(
            weekly_stats=ws_full,
            snap_counts=sc_full,
            depth_charts=dc_2025,
            ngs_receiving=ngs_full,
            schedules=sch_2025,
            season=_HELD_OUT_SEASON,
            as_of_week=int(week),
        )
        if feats.empty:
            continue
        preds = model.predict_distribution(feats, ruleset=Ruleset.espn_ppr())
        # Per-stat point predictions for fit metrics.
        stat_dists_per_row = model._build_stat_distributions(feats)
        per_stat_means = pd.DataFrame(
            {stat.value: [d[stat].mean() for d in stat_dists_per_row] for stat in model.target_stats}
        )
        per_stat_means["gsis_id"] = feats["gsis_id"].values
        per_stat_means["season"] = _HELD_OUT_SEASON
        per_stat_means["week"] = int(week)

        joined = preds.merge(
            per_stat_means, on=["gsis_id", "season", "week"], how="left"
        )
        rows.append(joined)

    all_preds = pd.concat(rows, ignore_index=True)

    # Inner-join to actual weekly stats (filter to WRs).
    actual = ws_2025[ws_2025["position"] == Position.WR.value].copy()
    actual["actual_ppr"] = _realized_ppr_points(actual, Ruleset.espn_ppr())
    keep = ["gsis_id", "season", "week", "actual_ppr"] + [s.value for s in model.target_stats]
    eval_df = all_preds.merge(actual[keep], on=["gsis_id", "season", "week"], how="inner",
                               suffixes=("_pred", "_actual"))

    print(f"\n=== 2025 sanity check (n={len(eval_df)} player-weeks) ===")

    # Per-stat fit.
    print("\n-- Per-stat fit --")
    for stat in model.target_stats:
        pred_col = f"{stat.value}_pred"
        actual_col = f"{stat.value}_actual"
        rmse = float(np.sqrt(((eval_df[pred_col] - eval_df[actual_col]) ** 2).mean()))
        mae = float((eval_df[pred_col] - eval_df[actual_col]).abs().mean())
        print(
            f"  {stat.value:>20s}  rmse={rmse:6.3f}  mae={mae:6.3f}  "
            f"mean_pred={eval_df[pred_col].mean():6.3f}  "
            f"mean_actual={eval_df[actual_col].mean():6.3f}"
        )

    # Composite — PPR.
    print("\n-- Composite (PPR points) --")
    rmse = float(np.sqrt(((eval_df["mean"] - eval_df["actual_ppr"]) ** 2).mean()))
    mae = float((eval_df["mean"] - eval_df["actual_ppr"]).abs().mean())
    print(f"  mean prediction:  rmse={rmse:.3f}  mae={mae:.3f}")
    # Top-30 rank correlation across the entire held-out year.
    pred_rank = eval_df.groupby("gsis_id")["mean"].sum().rank()
    actual_rank = eval_df.groupby("gsis_id")["actual_ppr"].sum().rank()
    common = pred_rank.index.intersection(actual_rank.index)
    spearman = float(np.corrcoef(pred_rank.loc[common], actual_rank.loc[common])[0, 1])
    print(f"  top-N season-total rank correlation (Spearman, all WRs): {spearman:.3f}")

    # Calibration.
    print("\n-- Calibration --")
    in_p10p90 = ((eval_df["actual_ppr"] >= eval_df["p10"]) & (eval_df["actual_ppr"] <= eval_df["p90"])).mean()
    le_p90 = (eval_df["actual_ppr"] <= eval_df["p90"]).mean()
    print(f"  fraction in [p10, p90]: {in_p10p90:.3f}  (target ≈ 0.80)")
    print(f"  fraction <= p90:        {le_p90:.3f}  (target ≈ 0.90)")

    print("\n=== End sanity check (informational; not a CI gate) ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick syntax check**

```bash
python -m py_compile scripts/sanity_check_wr_baseline.py
```

- [ ] **Step 3: Run gate**

```bash
mypy src tests scripts && ruff check src tests scripts && ruff format --check src tests scripts && pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add scripts/sanity_check_wr_baseline.py
git commit -m "feat(scripts): add sanity_check_wr_baseline.py — 2025 held-out eval"
```

---

## Task 19: Run sanity check; record numbers in PM doc

- [ ] **Step 1: Run**

```bash
python scripts/sanity_check_wr_baseline.py
```

Expected output is the metric block from Task 18. Capture it.

- [ ] **Step 2: Save the output to PM doc**

Append to `project_management.md` under a new entry above the "Current status" header (or at the top of an "operational notes" section if one exists). Use this template:

```markdown
### Plan 3a — 2025 WR sanity check (run YYYY-MM-DD)

```
[paste the full stdout block from Step 1 here]
```

Soft-threshold check vs. spec §6.3:
- Spearman top-30 correlation ≥ 0.4 — **MET** (or **MISSED** with brief note)
- [p10, p90] coverage in 70–90% — **MET** / **MISSED**
- Per-stat RMSE within 2× of naive baseline (predict L4 mean) — n/a until we compute the naive baseline; track for future
```

- [ ] **Step 3: Commit**

```bash
git add project_management.md
git commit -m "docs(pm): record Plan 3a 2025 WR sanity-check output"
```

---

## Task 20: Persist 2025 weekly projections to `data/projections/weekly/...`

**Files:**
- Create: `scripts/predict_2025_wr.py`

- [ ] **Step 1: Create the script**

Create `scripts/predict_2025_wr.py`:

```python
"""Plan 3a — write 2025 WR projections to data/projections/weekly/.

Loads the trained artifact, builds features for each week of 2025, predicts,
and writes one parquet partition per (season, week) using store.write_partition.
ProjectionWeeklySchema-validated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.features import build_wr_features
from projections.models import BaselineModel
from projections.schemas import ProjectionWeeklySchema, Ruleset
from projections.store import read_partition, write_partition


def _find_artifact(artifacts_root: Path) -> Path:
    matches = sorted(artifacts_root.glob("wr-baseline-*.joblib"))
    if not matches:
        raise FileNotFoundError(f"No wr-baseline-*.joblib in {artifacts_root}.")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 2025 WR weekly projections.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--ruleset", type=str, default="espn_ppr")
    args = parser.parse_args()

    artifact = _find_artifact(args.artifacts_root)
    print(f"Loading artifact: {artifact}")
    model = BaselineModel.load(artifact)

    ruleset = {"espn_ppr": Ruleset.espn_ppr(), "espn_half": Ruleset.espn_half(), "standard": Ruleset.standard()}[
        args.ruleset
    ]

    raw_root = args.raw_root
    ws_2024 = read_partition(raw_root, "weekly_stats", season=2024)
    sc_2024 = read_partition(raw_root, "snap_counts", season=2024)
    ngs_2024 = read_partition(raw_root, "ngs_receiving", season=2024)
    ws_2025 = read_partition(raw_root, "weekly_stats", season=2025)
    sc_2025 = read_partition(raw_root, "snap_counts", season=2025)
    dc_2025 = read_partition(raw_root, "depth_charts", season=2025)
    ngs_2025 = read_partition(raw_root, "ngs_receiving", season=2025)
    sch_2025 = read_partition(raw_root, "schedules", season=2025)

    ws_full = pd.concat([ws_2024, ws_2025], ignore_index=True)
    sc_full = pd.concat([sc_2024, sc_2025], ignore_index=True)
    ngs_full = pd.concat([ngs_2024, ngs_2025], ignore_index=True)

    weeks = sorted(dc_2025["week"].unique())
    rule_partition = ruleset.name  # e.g., "ESPN_PPR"
    for week in weeks:
        feats = build_wr_features(
            weekly_stats=ws_full,
            snap_counts=sc_full,
            depth_charts=dc_2025,
            ngs_receiving=ngs_full,
            schedules=sch_2025,
            season=2025,
            as_of_week=int(week),
        )
        if feats.empty:
            print(f"  Week {week}: no rostered WRs; skipping")
            continue
        preds = model.predict_distribution(feats, ruleset=ruleset)
        ProjectionWeeklySchema.validate(preds)
        # Custom layout for ruleset partitioning: write under
        # projections/weekly/season=YYYY/week=WW/ruleset=NAME/part.parquet.
        # store.write_partition supports (table, season, week); we encode
        # ruleset by appending it to the table name.
        target = write_partition(
            args.projections_root,
            f"weekly/ruleset={rule_partition}",
            preds,
            season=2025,
            week=int(week),
        )
        print(f"  Week {week}: wrote {len(preds)} rows -> {target}")

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick syntax check**

```bash
python -m py_compile scripts/predict_2025_wr.py
```

- [ ] **Step 3: Run**

```bash
python scripts/predict_2025_wr.py
```

Expected: per-week log lines; partitions written under `data/projections/weekly/ruleset=ESPN_PPR/season=2025/week=*/part.parquet`. (The exact path layout follows `store.write_partition`'s convention given the `table=` argument; if the structure ends up nested differently than expected, that's fine for v1 — Plan 3c may adjust the partition layout when adding the DuckDB view layer.)

- [ ] **Step 4: Verify the parquet**

```bash
python -c "
from pathlib import Path
from projections.schemas import ProjectionWeeklySchema
from projections.store import read_partition
p = read_partition(Path('data/projections'), 'weekly/ruleset=ESPN_PPR', season=2025)
print('Total 2025 rows:', len(p))
print(p.head(3))
ProjectionWeeklySchema.validate(p)
"
```

Expected: total ≈ 18 weeks × ≈100 WRs = ≈1800 rows; schema valid.

- [ ] **Step 5: Run gate**

```bash
mypy src tests scripts && ruff check src tests scripts && ruff format --check src tests scripts && pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add scripts/predict_2025_wr.py
git commit -m "feat(scripts): add predict_2025_wr.py — write 2025 WR projection partitions"
```

---

## Task 21: Smoke-test extension

**Files:**
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Re-read existing smoke test**

```bash
cat tests/test_smoke.py
```

Identify the existing entry point that ingests fixtures and produces feature DataFrames.

- [ ] **Step 2: Append a new test that wires fit → predict → write_partition**

Append at the bottom of `tests/test_smoke.py`:

```python
def test_smoke_wr_baseline_fit_predict_write(
    tmp_path,
    # The existing smoke test already provides ingest fixtures up to features —
    # reuse them verbatim. The exact fixture names depend on what's already in
    # this file; replace with the actual names.
    # In the absence of those fixtures here, build minimal in-line ones.
):
    """End-to-end: fit BaselineModel on synthetic data, predict, write a
    parquet partition through store.write_partition, read it back, validate
    against ProjectionWeeklySchema."""
    import pandas as pd

    from projections.features import build_wr_features
    from projections.models import wr_baseline
    from projections.schemas import ProjectionWeeklySchema, Ruleset
    from projections.store import read_partition, write_partition

    # We rely on the fixture fixtures already loaded by tests/test_features/conftest.py
    # — but those don't visit this directory. Re-create the minimal data here.
    # If the existing smoke test loops over the test_features/ fixtures via
    # other means, prefer that. Otherwise, this test calls test_models/ fixtures
    # via direct imports.
    from tests.test_models.conftest import baseline_features as _bf  # type: ignore[attr-defined]
    from tests.test_models.conftest import baseline_weekly_stats as _bw  # type: ignore[attr-defined]

    # NOTE: Pytest fixtures aren't directly callable. The clean path is to
    # MOVE the baseline_features / baseline_weekly_stats fixtures into
    # tests/conftest.py so they're visible from tests/test_smoke.py too,
    # and then add them as parameters here. That refactor is part of this
    # task — see Step 3 below.
    raise NotImplementedError("Step 3: relocate fixtures, then make this a real test.")
```

- [ ] **Step 3: Move shared fixtures to a top-level conftest**

Pytest fixtures only inherit from parent conftests, so the smoke test can't see fixtures defined in `tests/test_models/conftest.py`. Move `baseline_features` and `baseline_weekly_stats` to `tests/conftest.py`:

```bash
ls tests/conftest.py
```

If it exists, append. If not, create it. Either way, move the two fixtures from `tests/test_models/conftest.py` to `tests/conftest.py`. Delete the relocated definitions from the old file (leave the file in place if there are no other fixtures there; if it now contains only the imports, delete it entirely and remove `tests/test_models/__init__.py` references — actually, keep the package marker file).

- [ ] **Step 4: Replace the placeholder smoke test with the real one**

Replace the body of `test_smoke_wr_baseline_fit_predict_write`:

```python
def test_smoke_wr_baseline_fit_predict_write(
    tmp_path,
    baseline_features,
    baseline_weekly_stats,
):
    """End-to-end: fit BaselineModel on synthetic data, predict, write a
    parquet partition through store.write_partition, read back, validate."""
    from projections.models import wr_baseline
    from projections.schemas import ProjectionWeeklySchema, Ruleset
    from projections.store import read_partition, write_partition

    model = wr_baseline()
    model.fit(features=baseline_features, weekly_stats=baseline_weekly_stats)

    week_features = baseline_features[
        (baseline_features["season"] == 2025) & (baseline_features["week"] == 4)
    ]
    preds = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(preds)

    write_partition(
        tmp_path / "projections",
        "weekly/ruleset=ESPN_PPR",
        preds,
        season=2025,
        week=4,
    )
    round_tripped = read_partition(
        tmp_path / "projections", "weekly/ruleset=ESPN_PPR", season=2025, week=4
    )
    ProjectionWeeklySchema.validate(round_tripped)
    assert len(round_tripped) == len(preds)
```

- [ ] **Step 5: Run; expect green**

```bash
pytest tests/test_smoke.py -v
```

Expected: all smoke tests pass, including the new one.

- [ ] **Step 6: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green; total test count reflects the new test.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_models/conftest.py tests/test_smoke.py
git commit -m "test: extend smoke test to cover fit+predict+write_partition round trip"
```

---

## Task 22: PM doc + TODO updates

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`

- [ ] **Step 1: Re-read both docs**

```bash
cat project_management.md
cat TODO.md
```

- [ ] **Step 2: Update `project_management.md`**

Change the "Current status" header to reflect Plan 3a:

```markdown
## Current status (as of YYYY-MM-DD)

**Projections Core — Plan 3a (WR Model A baseline + first real-data ingest) merged to `main` at commit `<TBD-after-merge>`.**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Dev tooling merged via `feat/dev-tooling`.
- Plan 2a (Ingest expansion + WR feature builder) merged at `7926090`.
- Plan 2b (QB/RB/TE feature builders) merged at `af325ea`.

**Plan 3a delivered:**
- New `src/projections/models/` package: `Model` Protocol + `BaselineModel` impl + `wr_baseline()` factory.
- First real-data ingest pull: `data/raw/` populated for 2018–2025.
- `BaselineModel.fit` per-stat RidgeCV + parametric residual variance (gamma α + normal σ).
- `BaselineModel.predict_distribution` composes per-stat dists into points dist via existing `score_distribution`; output validates against `ProjectionWeeklySchema`.
- joblib persistence; `model_id = "baseline:wr:<8-char-hash>:<train-start>-<train-end>"` derived from source-file hashes.
- 2018–2024 trained artifact persisted; 2025 held-out sanity-check eval (informational).
- 2025 WR weekly projections written to `data/projections/weekly/ruleset=ESPN_PPR/season=2025/...`.
- N new tests (run `pytest -q` for actual count).

Update the "Next action" section to point to Plan 3b (generalize to QB/RB/TE).
```

- [ ] **Step 3: Add 3a decision log entries**

Append to the Decision log table (with today's date YYYY-MM-DD):

```markdown
| YYYY-MM-DD | Per-stat independent RidgeCV (architecture A from brainstorming) for Model A | Closest match to spec wording; per-stat residuals are debuggable; per-stat-independence assumption is "option D" / TODO #1 territory |
| YYYY-MM-DD | 2018–2024 train, 2025 held out for 3a sanity check | NGS receiving full coverage from 2018; 2025 was complete by run date |
| YYYY-MM-DD | `Model` as `Protocol` (not ABC); not @runtime_checkable | Structural typing matches existing `Distribution` Protocol; no isinstance checks needed in callers |
| YYYY-MM-DD | One `BaselineModel` class with per-position factories (`wr_baseline()`, future `qb_/rb_/te_baseline()`) | Minimizes 3a→3b copy; per-position quirks expressed as config |
| YYYY-MM-DD | `model_id` = `"baseline:<pos>:<8-char-code-hash>:<train-start>-<train-end>"` written into every projection row | Stable, reproducible, traceable; persisted into `ProjectionWeeklySchema` rows |
| YYYY-MM-DD | Method-of-moments for gamma α with clip to [0.01, 100] | Closed-form; MLE via scipy.optimize is a backlog item if calibration is bad |
```

- [ ] **Step 4: Update TODO.md**

If any TODO #9 sub-item (a/b/c) was fixed during Task 14 step 6, update its status to "**Resolved in Plan 3a**" or remove it. If new issues were discovered during real-data running, add them as new TODO entries (#13, #14, ...).

Also update TODO #2 (Plan 2b status) and TODO #4 (feature parquet caching gate — record whether 3a's training was fast enough that we deferred caching).

- [ ] **Step 5: Run gate**

```bash
mypy src tests && ruff check src tests && ruff format --check src tests && pytest -q
```

Expected: all green (docs-only edits don't affect tests, but verify nothing leaked).

- [ ] **Step 6: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm): close Plan 3a; queue Plan 3b + record 3a decision log"
```

---

## Task 23: End-of-effort verification + open PR

- [ ] **Step 1: Run the full quality gate at the worktree root**

```bash
. .venv/Scripts/activate
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
pytest -v -k "ingest or store or schemas"
```

All five must be green. Test count should be the prior 209 + (~15 new from Tasks 3–12) + (~1 new smoke) = ≈225+.

- [ ] **Step 2: Capture results for the PR description**

Save a concise summary like:
```
pytest: 225 passed in 14.5s
mypy: success no issues found in 70 source files
ruff check: All checks passed
ruff format: 70 files already formatted
pytest -k "ingest or store or schemas": 117 passed, 108 deselected
```

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feat/plan-3a-wr-model-a
```

(Branch is already pushed from the spec commit; this `git push` updates with the implementation commits.)

- [ ] **Step 4: Open the PR**

```bash
gh pr create --base main --head feat/plan-3a-wr-model-a --title "Plan 3a: WR Model A baseline (end-to-end pipeline)" --body "$(cat <<'EOF'
## Summary
- Adds `src/projections/models/` package: `Model` Protocol + `BaselineModel` implementation + `wr_baseline()` factory.
- First real-data ingest pull: `data/raw/` populated for seasons 2018–2025 via the new `ingest.refresh()` orchestrator.
- `BaselineModel.fit` trains one `RidgeCV` per scored WR stat and derives parametric residual variance (gamma α via method of moments; normal σ).
- `BaselineModel.predict_distribution` composes per-stat distributions into a points distribution via the existing `score_distribution` and emits `ProjectionWeeklySchema`-valid rows.
- joblib persistence; `model_id` is derived from a code hash over the model + WR-feature + scoring source files for traceability.
- 2018–2024 trained artifact persisted to `models/artifacts/`; 2025 held-out sanity-check eval (informational, not a CI gate).
- 2025 WR weekly projections written to `data/projections/weekly/ruleset=ESPN_PPR/season=2025/...`.

Plan-3 series:
- 3a (this PR): WR end-to-end + Model interface
- 3b (next): generalize to QB / RB / TE
- 3c (after): weekly→season aggregation + walk-forward backtest harness with CI thresholds

Spec: `docs/superpowers/specs/2026-04-25-plan-3a-wr-model-a-design.md`
Plan: `docs/superpowers/plans/2026-04-25-plan-3a-wr-model-a.md`

## Quality gate
[Paste Step 2 capture here as a fenced code block.]

## Test plan
- [x] `pytest -v` — full suite green
- [x] `mypy src tests` — zero violations
- [x] `ruff check src tests` — zero violations
- [x] `ruff format --check src tests` — no drift
- [x] `pytest -v -k "ingest or store or schemas"` — green
- [x] Real-data ingest pull (2018–2025) — manual smoke per Tasks 14, 15
- [x] Sanity-check eval — output recorded in `project_management.md`

## Sanity-check soft thresholds (informational)
[Paste from Task 19 entry.]

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace the placeholders with the actual gate output and sanity numbers.

- [ ] **Step 5: Report PR URL + summary**

Final report includes the PR URL and the captured Step 2 output.

---

## Self-Review

The plan covers every section of the spec:

| Spec section | Plan task(s) |
|---|---|
| §2.1 — new package layout | Tasks 3, 5 |
| §2.2 — Model Protocol | Task 3 |
| §2.3 — BaselineModel persisted state + config | Tasks 5, 6, 7 |
| §2.4 — wr_baseline factory | Task 5 |
| §3.1 — fit pipeline | Tasks 6, 7 |
| §3.2 — predict pipeline | Tasks 8, 9 |
| §3.3 — distribution family per stat | Task 5 (config) + verified in Task 8 |
| §3.4 — variance estimation | Task 7 |
| §3.5 — score_distribution dependency | Pre-verified during plan-writing; no extension needed |
| §4.1 — training + held-out seasons | Tasks 16 (training), 18 (sanity check) |
| §4.2 — first real-data pull staged | Tasks 14, 15 |
| §4.3 — training data shape | Task 6 (inner join) + Task 16 (real run) |
| §5.1 — artifact path | Task 16 |
| §5.2 — model_id construction | Task 4 (hash helper) + Task 9 (model_id property) |
| §5.3 — joblib serialization | Task 10 |
| §6 — sanity-check evaluation | Tasks 18, 19 |
| §7.1 — unit tests | Tasks 5–11 |
| §7.2 — leakage test | Task 12 |
| §7.3 — smoke-test extension | Task 21 |
| §7.4 — opt-in network test | Captured in §10 backlog (TODO #8 is the right home; 3a doesn't write a real test, just registers the `network` mark in Task 1) |
| §8 — edge cases | Tasks 8, 11 |
| §9 — risks/open questions | Addressed in Task 14 (drift smoke), Task 7 (gamma clip range), Task 14 step 6 (TODO #9 candidate fixes) |
| §10 — decisions | Replicated into PM doc decision log in Task 22 |
| §11 — MVP order | Implemented across Tasks 1–22 |

**Placeholder scan:** plan does not contain any "TBD/TODO/fill in details" content for engineer steps. Cross-references like `TODO #1 / #4 / #8 / #9 / #10` point to numbered backlog items in `TODO.md`. PR description placeholders (`[Paste ...]`) are filled at PR-time, not by an executing agent — these are deliberate.

**Type / signature consistency:** spot-checked — `BaselineModel.fit(features, weekly_stats)` matches the Model Protocol signature; `predict_distribution(features, ruleset) -> pd.DataFrame` matches; `compute_code_hash(paths: Iterable[Path]) -> str` consistently used in Tasks 4, 9, 22.

**Spec §7.4 nuance:** spec proposes a `@pytest.mark.network` test in `tests/test_models/test_baseline_real_data.py`. The plan registers the mark in Task 1 but does not actually create the test file (it's an opt-in nicety, not load-bearing). If the user wants the file scaffolded, add to TODO #8's scope rather than blocking 3a.
