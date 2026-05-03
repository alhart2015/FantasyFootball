"""Synthetic-fixture tests for trajectory_features.

Each compute fn is exercised against hand-rolled DataFrames; no real
weekly_stats / snap_counts / draft_picks parquets are read.
"""

from __future__ import annotations

# numpy / pandas / pytest are scaffolding for fixtures + per-compute tests
# added in subsequent tasks (Tasks 5-12); kept imported now so each task is
# a pure addition without re-importing.
import numpy as np  # noqa: F401  # used in subsequent tasks
import pandas as pd  # noqa: F401  # used in subsequent tasks
import pytest  # noqa: F401  # used in subsequent tasks

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
