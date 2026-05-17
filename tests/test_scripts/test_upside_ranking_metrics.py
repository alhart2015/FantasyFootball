"""Unit tests for diagnose_upside_ranking helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from diagnose_upside_ranking import _compute_elite_thresholds

from projections.schemas import Position, Ruleset

_RULESET = Ruleset.espn_ppr()


def test_compute_elite_thresholds_returns_one_per_position(tmp_path: Path) -> None:
    """Synthetic 2 seasons x 4 positions x 10 players: threshold = mean over
    seasons of the 5th-highest actual total at each position (>= 8 games filter)."""
    raw_root = tmp_path / "data" / "raw"
    weekly_stats_root = raw_root / "weekly_stats"
    for season in (2019, 2020):
        partition = weekly_stats_root / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        rows = []
        for pos_idx, pos in enumerate(("QB", "RB", "WR", "TE")):
            for player_idx in range(10):
                target_ppr = 100.0 + pos_idx * 50 + player_idx * 20  # 100..390 per pos
                for week in range(1, 11):
                    rows.append(
                        {
                            "gsis_id": f"00-{pos}-{player_idx:04d}",
                            "season": season,
                            "week": week,
                            "position": pos,
                            "passing_yards": 0.0 if pos != "QB" else target_ppr / 10 * 25,
                            "passing_tds": 0,
                            "interceptions": 0,
                            "rushing_yards": (
                                0.0 if pos in ("QB", "WR", "TE") else target_ppr / 10 * 10
                            ),
                            "rushing_tds": 0,
                            "receptions": 0 if pos in ("QB", "RB") else int(target_ppr / 10),
                            "receiving_yards": 0.0,
                            "receiving_tds": 0,
                            "fumbles_lost": 0,
                        }
                    )
        pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)

    thresholds = _compute_elite_thresholds(
        raw_root=raw_root,
        seasons=(2019, 2020),
        ruleset=_RULESET,
        min_games=8,
    )
    assert set(thresholds.keys()) == {Position.QB, Position.RB, Position.WR, Position.TE}
    for _pos, v in thresholds.items():
        assert isinstance(v, float)
        assert v > 0
    # 10 players per position; thresholds should differ across positions because
    # per-position scoring math produces different per-player totals.
    assert len(set(thresholds.values())) == 4
