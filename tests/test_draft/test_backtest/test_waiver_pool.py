"""Tests for WaiverPoolSchema and undrafted_pool_by_position (waiver-pool assessment)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from projections.draft.backtest.waiver_pool import undrafted_pool_by_position
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, VorpTableSchema, WaiverPoolSchema


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
    with pytest.raises(pa.errors.SchemaError):
        WaiverPoolSchema.validate(bad)


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
    rosters: Mapping[int, list[str]] = {
        1: ["00-0000000", "00-0000010"],
        2: ["00-0000020", "00-0000011"],
    }
    out = undrafted_pool_by_position(rosters, pool, _config()).set_index("position")

    # QB: undrafted vorps {10, -5}. top1=10, top2=-5, top3=NaN. best_proj=110 (the vorp=10 guy).
    assert out.loc["QB", "top1_vorp"] == 10.0
    assert out.loc["QB", "top2_vorp"] == -5.0
    assert np.isnan(out.loc["QB", "top3_vorp"])
    assert out.loc["QB", "best_avail_proj_pts"] == 110.0
    assert out.loc["QB", "n_above_replacement"] == 1  # only vorp=10 > 0
    # QB total-above = {30,10} = 2; drafted-above = {30} = 1 -> drain 0.5
    assert out.loc["QB", "drain_rate"] == pytest.approx(0.5)

    # RB: undrafted vorps {30, 20} (10, 11 drafted). top1=30, top2=20, top3=NaN.
    assert out.loc["RB", "top1_vorp"] == 30.0
    assert out.loc["RB", "top2_vorp"] == 20.0
    assert np.isnan(out.loc["RB", "top3_vorp"])
    assert out.loc["RB", "n_above_replacement"] == 2
    # RB total-above = 4 (50,40,30,20 all > 0); drafted-above = {50,40} = 2 -> drain 0.5
    assert out.loc["RB", "drain_rate"] == pytest.approx(0.5)

    # WR: undrafted vorps {15} (20 drafted). top1=15, top2/3 NaN.
    assert out.loc["WR", "top1_vorp"] == 15.0
    assert np.isnan(out.loc["WR", "top2_vorp"])
    assert out.loc["WR", "n_above_replacement"] == 1
    # WR total-above = 2; drafted-above = 1 -> drain 0.5
    assert out.loc["WR", "drain_rate"] == pytest.approx(0.5)

    # TE: 1 undrafted, vorp -3 (below repl). top1=-3, top2/3 NaN. n_above=0.
    # TE total-above = 0 -> drain_rate NaN (0/0).
    assert out.loc["TE", "top1_vorp"] == -3.0
    assert out.loc["TE", "n_above_replacement"] == 0
    assert np.isnan(out.loc["TE", "drain_rate"])


def test_fully_drafted_position_top_is_nan() -> None:
    pool = _pool()
    # Draft ALL four RBs -> RB fully drafted: top* NaN, n_above 0, drain 1.0 (4 of 4 above-repl).
    rosters: Mapping[int, list[str]] = {
        1: ["00-0000010", "00-0000011"],
        2: ["00-0000012", "00-0000013"],
    }
    out = undrafted_pool_by_position(rosters, pool, _config()).set_index("position")
    assert np.isnan(out.loc["RB", "top1_vorp"])
    assert np.isnan(out.loc["RB", "best_avail_proj_pts"])
    assert out.loc["RB", "n_above_replacement"] == 0
    assert out.loc["RB", "drain_rate"] == pytest.approx(1.0)


def test_output_validates_and_has_four_positions() -> None:
    out = undrafted_pool_by_position({1: [], 2: []}, _pool(), _config())
    assert set(out["position"]) == {"QB", "RB", "WR", "TE"}
    WaiverPoolSchema.validate(out)  # no raise
