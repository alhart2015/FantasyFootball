"""Smoke test for scripts/waiver_pool_assessment.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from waiver_pool_assessment import (
    _bootstrap_or_nan,
    _build_hero,
    format_assessment,
    run_assessment,
)

from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, VorpTableSchema

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
    text = format_assessment(agg)
    for pos in ("QB", "RB", "WR", "TE"):
        assert pos in text


def test_format_assessment_renders_nan_cells() -> None:
    # A fully-drained position aggregates to NaN; the table must render it, not crash.
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        v = float("nan") if pos == "TE" else 1.0
        for metric in _METRICS:
            rows.append({"position": pos, "metric": metric, "mean": v, "lo95": v, "hi95": v})
    text = format_assessment(pd.DataFrame(rows))
    assert "nan" in text.lower()  # TE row's NaN cells render without error


def test_bootstrap_or_nan_drops_nan_and_handles_all_nan() -> None:
    # NaN dropped -> mean over the defined values.
    point, lo, hi = _bootstrap_or_nan(np.array([5.0, np.nan, 5.0]), seed=0)
    assert point == 5.0 and lo == 5.0 and hi == 5.0
    # All-NaN -> (nan, nan, nan) else-branch (bootstrap_mean would crash on an empty array).
    p2, lo2, hi2 = _bootstrap_or_nan(np.array([np.nan, np.nan]), seed=0)
    assert np.isnan(p2) and np.isnan(lo2) and np.isnan(hi2)
