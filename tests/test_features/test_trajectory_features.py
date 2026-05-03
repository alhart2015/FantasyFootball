"""Synthetic-fixture tests for trajectory_features.

Each compute fn is exercised against hand-rolled DataFrames; no real
weekly_stats / snap_counts / draft_picks parquets are read.
"""

from __future__ import annotations

# numpy / pandas / pytest are scaffolding for fixtures + per-compute tests
# added in subsequent tasks (Tasks 5-12); kept imported now so each task is
# a pure addition without re-importing.
import numpy as np  # noqa: F401  # used in subsequent tasks
import pandas as pd
import pytest

from projections.features.trajectory_features import (
    DraftLookup,
)


def _ws_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    position: str = "QB",
    team: str = "KC",
    opponent: str = "BUF",
    attempts: int = 30,
    completions: int = 20,
    sacks: int = 2,
    passing_yards: float = 250.0,
    passing_tds: int = 2,
    interceptions: int = 0,
    rushing_yards: float = 10.0,
    rushing_tds: int = 0,
    carries: int = 3,
    receptions: int = 0,
    receiving_yards: float = 0.0,
    receiving_tds: int = 0,
    receiving_air_yards: float = 0.0,
    targets: int = 0,
    fumbles_lost: int = 0,
) -> dict[str, object]:
    """Helper: one weekly_stats row with sensible defaults."""
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "attempts": attempts,
        "completions": completions,
        "sacks": sacks,
        "passing_yards": passing_yards,
        "passing_tds": passing_tds,
        "interceptions": interceptions,
        "rushing_yards": rushing_yards,
        "rushing_tds": rushing_tds,
        "carries": carries,
        "receptions": receptions,
        "receiving_yards": receiving_yards,
        "receiving_tds": receiving_tds,
        "receiving_air_yards": receiving_air_yards,
        "targets": targets,
        "fumbles_lost": fumbles_lost,
    }


def _draft_lookup(*entries: tuple[str, int, float]) -> DraftLookup:
    return {gsis_id: (year, age) for gsis_id, year, age in entries}


def test_module_imports() -> None:
    """Smoke: confirm the module loads cleanly."""
    from projections.features import trajectory_features  # noqa: F401


def test_compute_age_uses_draft_age_when_available() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
            _ws_row(gsis_id="00-0033873", season=2018, week=2),
            _ws_row(gsis_id="00-0033873", season=2024, week=1),
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_age(weekly_stats, lookup)
    # One row per (gsis_id, season).
    assert len(out) == 2
    assert set(out.columns) == {"gsis_id", "season", "age", "draft_year_inferred"}
    age_2018 = out[out["season"] == 2018]["age"].iloc[0]
    age_2024 = out[out["season"] == 2024]["age"].iloc[0]
    assert age_2018 == pytest.approx(22.5)  # 21.5 + (2018 - 2017)
    assert age_2024 == pytest.approx(28.5)  # 21.5 + (2024 - 2017)
    assert (~out["draft_year_inferred"]).all()


def test_compute_age_falls_back_for_udfa() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0099999", season=2020, week=1),
            _ws_row(gsis_id="00-0099999", season=2024, week=1),
        ]
    )
    # No entry in the lookup → UDFA path.
    lookup: DraftLookup = {}
    out = compute_age(weekly_stats, lookup)
    assert len(out) == 2
    age_2020 = out[out["season"] == 2020]["age"].iloc[0]
    age_2024 = out[out["season"] == 2024]["age"].iloc[0]
    # inferred_draft_year = 2020 (earliest); age = season - 2020 + 22.0
    assert age_2020 == pytest.approx(22.0)
    assert age_2024 == pytest.approx(26.0)
    assert out["draft_year_inferred"].all()


def test_compute_age_falls_back_when_draft_age_is_nan() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
        ]
    )
    # Drafted but no draft_age — fall back to inferred path.
    lookup = _draft_lookup(("00-0033873", 2017, float("nan")))
    out = compute_age(weekly_stats, lookup)
    age_2018 = out[out["season"] == 2018]["age"].iloc[0]
    # inferred_draft_year = 2018 (earliest); 2018 - 2018 + 22 = 22.0
    assert age_2018 == pytest.approx(22.0)
    assert out["draft_year_inferred"].all()


def test_compute_age_one_row_per_player_season() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [_ws_row(gsis_id="00-0033873", season=2018, week=w) for w in range(1, 18)]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_age(weekly_stats, lookup)
    assert len(out) == 1


def test_compute_age_empty_input() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        columns=["gsis_id", "season", "week", "position", "team", "opponent"]
    )
    lookup: DraftLookup = {}
    out = compute_age(weekly_stats, lookup)
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "age", "draft_year_inferred"}
