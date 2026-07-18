# Waiver-Wire / Undrafted-Pool Assessment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-17-waiver-pool-assessment-design.md`

**Goal:** Characterize the undrafted (waiver) pool by position after a hero+15-bots 16-team draft, averaged over N seeds, producing a per-position readout of how good the best still-available players are and how deep the wire is.

**Architecture:** A pure, tested core function (`undrafted_pool_by_position`) computes per-position metrics for one draft; a thin `scripts/` driver runs `draft_mixed_field` over N seeds and aggregates to mean + bootstrap 95% CI; a `reports/` writeup interprets the result. Everything reuses existing sim machinery (`draft_mixed_field`, `hero_seat_layout`, the VORP pool, `bootstrap_mean`).

**Tech Stack:** Python 3.12, pandas, numpy, pandera (schemas), pytest, mypy (strict), ruff.

## Global Constraints

- All gates clean: `pytest`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`.
- The core function's output DataFrame is validated with `WaiverPoolSchema.validate(df)` (with reassignment) at its boundary.
- Reference enums, never raw strings: `Position.QB`, etc. Positions iterated as `(Position.QB, Position.RB, Position.WR, Position.TE)`.
- Nullable string columns use `_PYARROW_STR` (= `pd.StringDtype("pyarrow")`); the schema uses `Series[...]` + `pa.Field(...)` per the existing `schemas.py` style; `class Config: strict = "filter"; coerce = True`.
- `vorp` may be negative (sub-replacement players) — do not add a lower bound on the `top*_vorp` columns.
- The driver reads the preset VORP table with `pd.read_parquet` (matching existing analysis scripts, e.g. the snake bake-off); the `store.*` partition rule is for ingest/feature code, not preset-table analysis.
- Hero is analytic only (no availability): keys `now_or_never`, `now_or_never_floored`, `raw_vorp`. The floored hero uses shipped default floor (`_DEFAULT_FLOOR=40` / `_DEFAULT_FLOOR_WEIGHT=1`).

---

### Task 1: `WaiverPoolSchema` in `schemas.py`

**Files:**
- Modify: `src/projections/schemas.py` (add class after `HeroResultSchema`, ~line 1153)
- Test: `tests/test_draft/test_backtest/test_waiver_pool.py` (new file; the schema test lives here alongside Task 2's function tests)

**Interfaces:**
- Produces: `WaiverPoolSchema` (pandera `DataFrameModel`) with columns `position` (str, isin skill positions, unique), `top1_vorp`/`top2_vorp`/`top3_vorp` (float, nullable), `best_avail_proj_pts` (float, ge=0, nullable), `n_above_replacement` (int, ge=0), `drain_rate` (float, ge=0, le=1, nullable).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_draft/test_backtest/test_waiver_pool.py` (new file):

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.schemas import WaiverPoolSchema, _PYARROW_STR


def _valid_waiver_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": pd.array(["QB", "RB", "WR", "TE"], dtype=_PYARROW_STR),
            "top1_vorp": [80.0, 120.0, 100.0, np.nan],
            "top2_vorp": [70.0, 110.0, 95.0, np.nan],
            "top3_vorp": [60.0, 100.0, 90.0, np.nan],
            "best_avail_proj_pts": [280.0, 250.0, 240.0, np.nan],
            "n_above_replacement": [5, 12, 20, 0],
            "drain_rate": [0.5, 0.8, 0.3, np.nan],
        }
    )


def test_waiver_pool_schema_accepts_valid_frame() -> None:
    out = WaiverPoolSchema.validate(_valid_waiver_frame())
    assert list(out["position"]) == ["QB", "RB", "WR", "TE"]


def test_waiver_pool_schema_rejects_drain_rate_above_one() -> None:
    bad = _valid_waiver_frame()
    bad.loc[0, "drain_rate"] = 1.5
    with pytest.raises(Exception):
        WaiverPoolSchema.validate(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_backtest/test_waiver_pool.py -v`
Expected: FAIL with `ImportError: cannot import name 'WaiverPoolSchema'`.

- [ ] **Step 3: Write minimal implementation**

In `src/projections/schemas.py`, after the `HeroResultSchema` class (before `PreseasonFeaturesSchema`), add:

```python
class WaiverPoolSchema(pa.DataFrameModel):
    """Per-position undrafted-pool ("waiver wire") metrics for ONE simulated draft.

    Output of `undrafted_pool_by_position`. Exactly one row per skill position
    (QB/RB/WR/TE), always all four even when a position is fully drafted (its
    `top*_vorp` / `best_avail_proj_pts` are then NaN). `top{1,2,3}_vorp` are the
    three highest undrafted `vorp` at the position (NaN when fewer remain; `vorp`
    may be negative). `n_above_replacement` counts undrafted players with vorp > 0.
    `drain_rate` = drafted-above-replacement / total-above-replacement in [0, 1],
    NaN when the position has no above-replacement players in the pool (0/0).
    """

    position: Series[str] = pa.Field(isin=_SKILL_POSITION_VALUES, unique=True)
    top1_vorp: Series[float] = pa.Field(nullable=True)
    top2_vorp: Series[float] = pa.Field(nullable=True)
    top3_vorp: Series[float] = pa.Field(nullable=True)
    best_avail_proj_pts: Series[float] = pa.Field(ge=0, nullable=True)
    n_above_replacement: Series[int] = pa.Field(ge=0)
    drain_rate: Series[float] = pa.Field(ge=0, le=1, nullable=True)

    class Config:
        strict = "filter"
        coerce = True
```

Confirm `_SKILL_POSITION_VALUES` and `_PYARROW_STR` are already defined in `schemas.py` (they are — `_SKILL_POSITION_VALUES` ~line 302, `_PYARROW_STR` in the string-dtype block) and that `WaiverPoolSchema` and `_PYARROW_STR` are exported if the module uses `__all__` (check the bottom of `schemas.py`; add `WaiverPoolSchema` to `__all__` if one exists).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_backtest/test_waiver_pool.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_draft/test_backtest/test_waiver_pool.py
git commit -m "feat(schemas): WaiverPoolSchema — per-position undrafted-pool metrics"
```

---

### Task 2: `undrafted_pool_by_position` core function

**Files:**
- Create: `src/projections/draft/backtest/waiver_pool.py`
- Test: `tests/test_draft/test_backtest/test_waiver_pool.py` (append to Task 1's file)

**Interfaces:**
- Consumes: `WaiverPoolSchema` (Task 1); `VorpTableSchema`, `Position`, `_PYARROW_STR` from `schemas`; `LeagueConfig`.
- Produces: `undrafted_pool_by_position(rosters: Mapping[int, list[str]], pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame` — a `WaiverPoolSchema`-valid 4-row frame (one per skill position, in QB, RB, WR, TE order). `config` is accepted for interface stability / future use (which positions are in scope) but v1 always reports the four skill positions.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft/test_backtest/test_waiver_pool.py`:

```python
from collections.abc import Mapping

from projections.draft.backtest.waiver_pool import undrafted_pool_by_position
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, VorpTableSchema


def _config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=200,
        min_bid=1,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 2,
        },
        ruleset="espn_half",  # type: ignore[arg-type]
    )


def _pool() -> pd.DataFrame:
    # replacement_fpts is per-position constant; vorp = season_mean_fpts - replacement.
    # QB: 3 players, vorps 30/10/-5 (repl 100). RB: 4 players vorps 50/40/30/20 (repl 60).
    # WR: 2 players vorps 25/15 (repl 80). TE: 1 player vorp -3 (repl 90) -> none above repl.
    rows = [
        # (pos, id_num, season_mean, vorp, repl)
        ("QB", 0, 130.0, 30.0, 100.0),
        ("QB", 1, 110.0, 10.0, 100.0),
        ("QB", 2, 95.0, -5.0, 100.0),
        ("RB", 10, 110.0, 50.0, 60.0),
        ("RB", 11, 100.0, 40.0, 60.0),
        ("RB", 12, 90.0, 30.0, 60.0),
        ("RB", 13, 80.0, 20.0, 60.0),
        ("WR", 20, 105.0, 25.0, 80.0),
        ("WR", 21, 95.0, 15.0, 80.0),
        ("TE", 30, 87.0, -3.0, 90.0),
    ]
    df = pd.DataFrame(
        [
            {
                "gsis_id": f"00-{n:07d}",
                "position": pos,
                "season_mean_fpts": sm,
                "vorp": v,
                "replacement_fpts": r,
                "consensus_adp": float(n + 1),
            }
            for pos, n, sm, v, r in rows
        ]
    )
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def test_metrics_exact() -> None:
    pool = _pool()
    # Draft: seat1 takes QB0 (best QB) + RB10 (best RB); seat2 takes WR20 (best WR) + RB11.
    rosters: Mapping[int, list[str]] = {1: ["00-0000000", "00-0000010"], 2: ["00-0000020", "00-0000011"]}
    out = undrafted_pool_by_position(rosters, pool, _config())
    out = out.set_index("position")

    # QB: undrafted vorps {10, -5}. top1=10, top2=-5, top3=NaN. best_proj=110 (the vorp=10 guy).
    assert out.loc["QB", "top1_vorp"] == 10.0
    assert out.loc["QB", "top2_vorp"] == -5.0
    assert np.isnan(out.loc["QB", "top3_vorp"])
    assert out.loc["QB", "best_avail_proj_pts"] == 110.0
    assert out.loc["QB", "n_above_replacement"] == 1  # only vorp=10 > 0
    # QB total-above = {30,10} = 2 drafted-above = {30} = 1 -> drain 0.5
    assert out.loc["QB", "drain_rate"] == pytest.approx(0.5)

    # RB: undrafted vorps {30, 20} (10,11 drafted). top1=30, top2=20, top3=NaN.
    assert out.loc["RB", "top1_vorp"] == 30.0
    assert out.loc["RB", "top2_vorp"] == 20.0
    assert np.isnan(out.loc["RB", "top3_vorp"])
    assert out.loc["RB", "n_above_replacement"] == 2
    # RB total-above = 4 (50,40,30,20 all > 0), drafted-above = {50,40} = 2 -> drain 0.5
    assert out.loc["RB", "drain_rate"] == pytest.approx(0.5)

    # WR: undrafted vorps {15} (20 drafted). top1=15, top2/3 NaN.
    assert out.loc["WR", "top1_vorp"] == 15.0
    assert np.isnan(out.loc["WR", "top2_vorp"])
    assert out.loc["WR", "n_above_replacement"] == 1
    # WR total-above = 2, drafted-above = 1 -> drain 0.5
    assert out.loc["WR", "drain_rate"] == pytest.approx(0.5)

    # TE: 1 undrafted, vorp -3 (below repl). top1=-3, top2/3 NaN. n_above=0.
    # TE total-above = 0 -> drain_rate NaN (0/0).
    assert out.loc["TE", "top1_vorp"] == -3.0
    assert out.loc["TE", "n_above_replacement"] == 0
    assert np.isnan(out.loc["TE", "drain_rate"])


def test_fully_drafted_position_top_is_nan() -> None:
    pool = _pool()
    # Draft the only TE -> TE fully drafted. drain=1.0 would need total-above>0; TE total-above=0
    # so drain stays NaN. Use RB: draft ALL four RBs -> top* NaN, n_above 0, drain 1.0.
    rosters: Mapping[int, list[str]] = {
        1: ["00-0000010", "00-0000011"],
        2: ["00-0000012", "00-0000013"],
    }
    out = undrafted_pool_by_position(rosters, pool, _config()).set_index("position")
    assert np.isnan(out.loc["RB", "top1_vorp"])
    assert np.isnan(out.loc["RB", "best_avail_proj_pts"])
    assert out.loc["RB", "n_above_replacement"] == 0
    assert out.loc["RB", "drain_rate"] == pytest.approx(1.0)  # 4 of 4 above-repl drafted


def test_output_validates_and_has_four_positions() -> None:
    out = undrafted_pool_by_position({1: [], 2: []}, _pool(), _config())
    assert set(out["position"]) == {"QB", "RB", "WR", "TE"}
    WaiverPoolSchema.validate(out)  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_backtest/test_waiver_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: projections.draft.backtest.waiver_pool`.

- [ ] **Step 3: Write minimal implementation**

Create `src/projections/draft/backtest/waiver_pool.py`:

```python
"""Characterize the undrafted ("waiver wire") pool by position after a draft.

Pure, per-draft core for the waiver-pool assessment (spec 2026-07-17): given the
drafted rosters and the VORP pool, report per skill position how good the best
still-available players are (top-3 VORP + the leader's projected points), how much
startable-quality depth remains (count above replacement), and how hard the field
drained the position (drain_rate). The driver (scripts/waiver_pool_assessment.py)
runs this over many simulated drafts and aggregates.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import Position, VorpTableSchema, WaiverPoolSchema

_SKILL_POSITIONS = (Position.QB, Position.RB, Position.WR, Position.TE)


def undrafted_pool_by_position(
    rosters: Mapping[int, list[str]],
    pool: pd.DataFrame,
    config: LeagueConfig,
) -> pd.DataFrame:
    """Per-position undrafted-pool metrics for one draft (WaiverPoolSchema-valid).

    `rosters` is {seat: [gsis_id, ...]} from `draft_mixed_field`; `pool` is a
    VorpTableSchema-valid consensus VORP table. `config` is accepted for interface
    stability; v1 always reports the four skill positions (QB/RB/WR/TE). See the
    module and WaiverPoolSchema docstrings for the metric definitions and edge cases.
    """
    pool = VorpTableSchema.validate(pool)
    drafted = {str(g) for roster in rosters.values() for g in roster}
    pos = pool["position"].astype(str)
    vorp = pool["vorp"]
    undrafted = ~pool["gsis_id"].astype(str).isin(drafted)

    rows: list[dict[str, object]] = []
    for p in _SKILL_POSITIONS:
        at_pos = pos == p.value
        above = at_pos & (vorp > 0)
        total_above = int(above.sum())
        drafted_above = int((above & ~undrafted).sum())

        undf = pool[at_pos & undrafted]
        top = np.sort(undf["vorp"].to_numpy())[::-1]  # descending
        top1 = float(top[0]) if len(top) > 0 else float("nan")
        top2 = float(top[1]) if len(top) > 1 else float("nan")
        top3 = float(top[2]) if len(top) > 2 else float("nan")

        if len(undf):
            best_proj = float(undf.loc[undf["vorp"].idxmax(), "season_mean_fpts"])
        else:
            best_proj = float("nan")

        rows.append(
            {
                "position": p.value,
                "top1_vorp": top1,
                "top2_vorp": top2,
                "top3_vorp": top3,
                "best_avail_proj_pts": best_proj,
                "n_above_replacement": int((undf["vorp"] > 0).sum()),
                "drain_rate": (drafted_above / total_above) if total_above else float("nan"),
            }
        )

    return WaiverPoolSchema.validate(pd.DataFrame(rows))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_backtest/test_waiver_pool.py -v`
Expected: PASS (all tests). Then `mypy src/projections/draft/backtest/waiver_pool.py` and `ruff check src/projections/draft/backtest/waiver_pool.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/backtest/waiver_pool.py tests/test_draft/test_backtest/test_waiver_pool.py
git commit -m "feat(draft): undrafted_pool_by_position — per-position waiver metrics"
```

---

### Task 3: Driver `scripts/waiver_pool_assessment.py`

**Files:**
- Create: `scripts/waiver_pool_assessment.py`
- Test: `tests/test_scripts/test_waiver_pool_assessment.py`

**Interfaces:**
- Consumes: `undrafted_pool_by_position` (Task 2); `draft_mixed_field`, `hero_seat_layout` (`draft_field`); `LogisticSurvival`, `default_sigma` (`survival`); `NowOrNeverStrategy`, `NowOrNeverFlooredStrategy`, `RawVorpStrategy`, `DraftStrategy` (`strategy`); `bootstrap_mean` (`_compare`); `VorpTableSchema`, `LeagueConfig`.
- Produces: `run_assessment(pool, config, *, hero, hero_seat, seeds, jitter, base_seed) -> pd.DataFrame` (long-format: columns `position`, `metric`, `mean`, `lo95`, `hi95`; 4 positions × 6 metrics = 24 rows); `format_assessment(agg) -> str`; `_build_hero(key, n_teams) -> DraftStrategy`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scripts/test_waiver_pool_assessment.py`:

```python
from __future__ import annotations

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, VorpTableSchema, _PYARROW_STR
from waiver_pool_assessment import _build_hero, format_assessment, run_assessment

_METRICS = {
    "top1_vorp",
    "top2_vorp",
    "top3_vorp",
    "best_avail_proj_pts",
    "n_above_replacement",
    "drain_rate",
}


def _config() -> LeagueConfig:
    return LeagueConfig(
        name="t16",
        n_teams=16,
        budget=200,
        min_bid=1,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 4,
        },
        ruleset="espn_half",  # type: ignore[arg-type]
    )


def _pool(n_per_pos: int = 60) -> pd.DataFrame:
    pos_offset = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    rows = []
    for pos, off in pos_offset.items():
        for i in range(n_per_pos):
            n = off * 1000 + i
            rows.append(
                {
                    "gsis_id": f"00-{n:07d}",
                    "position": pos,
                    "season_mean_fpts": 300.0 - i,
                    "vorp": 150.0 - i,
                    "replacement_fpts": 150.0,
                    "consensus_adp": float(i * 4 + {"QB": 3, "RB": 1, "WR": 2, "TE": 4}[pos]),
                }
            )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def test_run_assessment_smoke() -> None:
    pool, config = _pool(), _config()
    hero = _build_hero("raw_vorp", config.n_teams)
    agg = run_assessment(pool, config, hero=hero, hero_seat=1, seeds=2, jitter=8.0, base_seed=0)
    assert set(agg["position"]) == {"QB", "RB", "WR", "TE"}
    assert set(agg["metric"]) == _METRICS
    assert len(agg) == 24  # 4 positions x 6 metrics
    assert agg["mean"].notna().all()  # deep synthetic pool -> no NaN metrics
    # format renders without error and mentions each position
    text = format_assessment(agg)
    for pos in ("QB", "RB", "WR", "TE"):
        assert pos in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_waiver_pool_assessment.py -v`
Expected: FAIL with `ModuleNotFoundError: waiver_pool_assessment` (scripts on path via `tests/test_scripts/conftest.py`).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/waiver_pool_assessment.py`:

```python
"""Waiver-wire / undrafted-pool assessment (spec 2026-07-17, issue #112).

Runs a hero + 15 constrained-ADP-bot 16-team draft over many seeds, then reports,
per skill position, how good the best still-available players are and how deep the
wire is (mean + 95% bootstrap CI). Analytic hero only (now_or_never[_floored] /
raw_vorp).

    python scripts/waiver_pool_assessment.py \
        --vorp-table data/vorp_2026/half_16team.parquet \
        --league-config data/vorp_2026/half_16team.league.json \
        --hero-strategy now_or_never_floored --seeds 200 --out reports/waiver_pool_2026.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverFlooredStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.assistant._compare import bootstrap_mean
from projections.draft.backtest.draft_field import draft_mixed_field, hero_seat_layout
from projections.draft.backtest.waiver_pool import undrafted_pool_by_position
from projections.draft.league_config import LeagueConfig
from projections.schemas import VorpTableSchema

_METRIC_COLS = (
    "top1_vorp",
    "top2_vorp",
    "top3_vorp",
    "best_avail_proj_pts",
    "n_above_replacement",
    "drain_rate",
)
_ANALYTIC = ("now_or_never", "now_or_never_floored", "raw_vorp")


def _build_hero(key: str, n_teams: int) -> DraftStrategy:
    """Build an analytic hero strategy (no availability load) from its key."""
    surv = LogisticSurvival(sigma=default_sigma(n_teams))
    if key == "now_or_never_floored":
        return NowOrNeverFlooredStrategy(surv)
    if key == "now_or_never":
        return NowOrNeverStrategy(surv)
    if key == "raw_vorp":
        return RawVorpStrategy()
    raise ValueError(
        f"hero strategy {key!r} not supported in v1 (analytic keys only: {_ANALYTIC}; "
        f"MC strategies need availability wiring)"
    )


def run_assessment(
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    hero: DraftStrategy,
    hero_seat: int,
    seeds: int,
    jitter: float,
    base_seed: int,
) -> pd.DataFrame:
    """Run `seeds` hero+bots drafts and aggregate per-(position, metric) to mean+95% CI.

    Returns a long-format frame: columns position, metric, mean, lo95, hi95 (one row
    per position x metric).
    """
    layout = hero_seat_layout(hero_seat=hero_seat, hero_label="hero", n_teams=config.n_teams)
    seat_strategies: dict[int, DraftStrategy | None] = {
        s: (hero if lbl == "hero" else None) for s, lbl in layout.items()
    }
    per_seed: list[pd.DataFrame] = []
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        rosters = draft_mixed_field(seat_strategies, pool, config, rng=rng, jitter=jitter)
        per_seed.append(undrafted_pool_by_position(rosters, pool, config))

    stacked = pd.concat(per_seed, ignore_index=True)
    out_rows: list[dict[str, object]] = []
    for position, grp in stacked.groupby("position", sort=False):
        for metric in _METRIC_COLS:
            iv = bootstrap_mean(grp[metric].to_numpy(dtype=float), seed=base_seed)
            out_rows.append(
                {
                    "position": position,
                    "metric": metric,
                    "mean": iv.point,
                    "lo95": iv.lo_95,
                    "hi95": iv.hi_95,
                }
            )
    return pd.DataFrame(out_rows)


def format_assessment(agg: pd.DataFrame) -> str:
    """Render the per-position table, positions sorted by mean top1_vorp descending."""
    wide = agg.pivot(index="position", columns="metric", values="mean")
    wide = wide.sort_values("top1_vorp", ascending=False)
    lines = ["POSITION  TOP1_VORP  TOP2  TOP3  BEST_PROJ  #>REPL  DRAIN%"]
    for position, r in wide.iterrows():
        lines.append(
            f"{position:<8}  {r['top1_vorp']:8.1f}  {r['top2_vorp']:5.1f}  {r['top3_vorp']:5.1f}  "
            f"{r['best_avail_proj_pts']:8.1f}  {r['n_above_replacement']:5.1f}  "
            f"{r['drain_rate'] * 100:5.1f}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Waiver-wire / undrafted-pool assessment.")
    p.add_argument("--vorp-table", type=Path, default=Path("data/vorp_2026/half_16team.parquet"))
    p.add_argument(
        "--league-config", type=Path, default=Path("data/vorp_2026/half_16team.league.json")
    )
    p.add_argument("--hero-strategy", choices=list(_ANALYTIC), default="now_or_never_floored")
    p.add_argument("--hero-seat", type=int, default=1)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--jitter", type=float, default=8.0)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Optional path to write the table.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    pool = VorpTableSchema.validate(pd.read_parquet(args.vorp_table))
    hero = _build_hero(args.hero_strategy, config.n_teams)
    agg = run_assessment(
        pool,
        config,
        hero=hero,
        hero_seat=args.hero_seat,
        seeds=args.seeds,
        jitter=args.jitter,
        base_seed=args.base_seed,
    )
    text = format_assessment(agg)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_waiver_pool_assessment.py -v`
Expected: PASS. Then `mypy scripts/waiver_pool_assessment.py` and `ruff check scripts/waiver_pool_assessment.py` clean. (If mypy flags the `groupby(...).iterrows()` row access dtypes, coerce with `float(r["..."])` in `format_assessment`.)

- [ ] **Step 5: Commit**

```bash
git add scripts/waiver_pool_assessment.py tests/test_scripts/test_waiver_pool_assessment.py
git commit -m "feat(draft): waiver_pool_assessment driver — hero+bots sim, per-position CI table"
```

---

### Task 4: Generate the 2026 report

**Files:**
- Create: `reports/waiver_pool_2026.md`

**Interfaces:**
- Consumes: the driver (Task 3), the real `data/vorp_2026/half_16team.parquet` pool.

- [ ] **Step 1: Run the driver on the real 2026 pool**

Run (PowerShell, `KMP_DUPLICATE_LIB_OK=TRUE`):

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python.exe scripts/waiver_pool_assessment.py \
  --vorp-table data/vorp_2026/half_16team.parquet \
  --league-config data/vorp_2026/half_16team.league.json \
  --hero-strategy now_or_never_floored --seeds 200
```

Expected: a 4-row per-position table printed, positions sorted by best-available VORP. Capture the output.

- [ ] **Step 2: Write the report**

Create `reports/waiver_pool_2026.md` with: (1) the captured per-position table; (2) the readout — which positions are **depleted** (best-available VORP near/below replacement, high drain%) vs **relatively strong / streamable** (good VORP still on the board, deep count above replacement); (3) the tie-back to the scarcity thread (does TE drain to nothing, i.e. is elite-TE scarcity real, or does it stay streamable?); (4) caveats (single 2026 pool + format; bots = noisy-ADP proxy; hero second-order; analytic hero) and the reproduce command. Record **in isolation** — data-gathering, **no strategy adopt/reject verdict** (the standing rule).

- [ ] **Step 3: Commit**

```bash
git add reports/waiver_pool_2026.md
git commit -m "docs(draft): waiver-pool 2026 readout (issue #112)"
```

---

## Self-Review

**1. Spec coverage:**
- §3 core function + all metrics (top1/2/3_vorp, best_avail_proj_pts, n_above_replacement, drain_rate incl. 0/0 → NaN) → Task 2 + tests. ✓
- §3 WaiverPoolSchema (columns/dtypes/index-as-column, always 4 rows) → Task 1. ✓
- §4 driver (hero_seat_layout + draft_mixed_field over N seeds, bootstrap CI aggregation, flags/defaults, `_build_hero` analytic-only with clear error, shipped-default floor) → Task 3. ✓
- §5 output (per-position table sorted by top1_vorp desc; report md) → Task 3 `format_assessment` + Task 4. ✓
- §6 tests (exact-value unit, schema, edge cases NaN-pad / drain=1 / drain=NaN, driver smoke on synthetic pool) → Tasks 1–3. ✓
- §2 non-goals (no per-season, no MC hero, no live tool, no new sim machinery) — respected; `_build_hero` rejects MC keys. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases" — every code step shows complete code; edge cases are named with expected values. Task 4 Step 2 is an analysis write-up (inherently prose) but enumerates exactly what the report must contain. ✓

**3. Type consistency:** `undrafted_pool_by_position(rosters, pool, config)` signature identical in Task 2 def and Task 3 call. `run_assessment(...)` / `_build_hero(...)` / `format_assessment(...)` signatures identical between Task 3 def and its test. Metric column names identical across schema (Task 1), function (Task 2), and driver `_METRIC_COLS` (Task 3). `WaiverPoolSchema` fields match the function's output dict keys. ✓

**Note (spec refinement, non-behavioral):** the spec says the output is "indexed by position"; the plan realizes this as a validated `position` **column** (pandera-idiomatic here — every existing schema is column-based), preserving the "one row per position" contract. Tests use `.set_index("position")` for lookups.
