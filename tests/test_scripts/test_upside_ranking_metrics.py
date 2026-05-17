"""Unit tests for diagnose_upside_ranking helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
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


def test_top_k_overlap_perfect_match() -> None:
    from diagnose_upside_ranking import top_k_overlap

    pred_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    actual_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    assert top_k_overlap(pred_rank, actual_rank, k=3) == pytest.approx(1.0)
    assert top_k_overlap(pred_rank, actual_rank, k=5) == pytest.approx(1.0)


def test_top_k_overlap_partial_match() -> None:
    from diagnose_upside_ranking import top_k_overlap

    pred_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    actual_rank = pd.Series([3, 1, 2, 4, 5], index=["a", "b", "c", "d", "e"])
    # pred top-3 = {a, b, c}, actual top-3 = {b, c, a} -> overlap 3/3 = 1.0
    assert top_k_overlap(pred_rank, actual_rank, k=3) == pytest.approx(1.0)
    # pred top-1 = {a}, actual top-1 = {b} -> overlap 0/1 = 0
    assert top_k_overlap(pred_rank, actual_rank, k=1) == pytest.approx(0.0)


def test_top5_rank_err_median_abs() -> None:
    from diagnose_upside_ranking import top5_rank_err

    # actual top-5 = {a, b, c, d, e} with ranks 1..5.
    # pred ranks for {a, b, c, d, e} = [1, 4, 3, 2, 5] -> errors [0, 2, 0, 2, 0]
    # median = 0.0
    pred_rank = pd.Series([1, 4, 3, 2, 5, 6], index=["a", "b", "c", "d", "e", "f"])
    actual_rank = pd.Series([1, 2, 3, 4, 5, 6], index=["a", "b", "c", "d", "e", "f"])
    assert top5_rank_err(pred_rank, actual_rank) == pytest.approx(0.0)


def test_kendall_tau_filtered_excludes_low_nweeks() -> None:
    from diagnose_upside_ranking import kendall_tau_filtered

    pred = pd.Series([100.0, 90.0, 80.0, 70.0], index=["a", "b", "c", "d"])
    actual = pd.Series([100.0, 90.0, 80.0, 70.0], index=["a", "b", "c", "d"])
    n_weeks = pd.Series([10, 10, 10, 3], index=["a", "b", "c", "d"])
    tau, n = kendall_tau_filtered(pred, actual, n_weeks, min_n_weeks=6)
    # 'd' is excluded; perfect rank agreement on the remaining 3 -> tau = 1.0, n = 3
    assert tau == pytest.approx(1.0)
    assert n == 3
