"""Smoke test for scripts/waiver_pool_assessment.py."""

from __future__ import annotations

import pandas as pd
from waiver_pool_assessment import _build_hero, format_assessment, run_assessment

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
