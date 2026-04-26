"""Unit tests for src/projections/backtest/naive.py."""

from __future__ import annotations

import math

import pandas as pd

from projections.backtest.naive import compute_naive_predictions
from projections.schemas import Position, Stat


def _ws_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    position: str = "WR",
    **stats: float,
) -> dict[str, object]:
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
    train = pd.DataFrame(
        [_ws_row(gsis_id="00-A", season=2023, week=w, receptions=2) for w in range(1, 18)]
    )
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
