"""Tests for `projections.draft.consensus_source.consensus_to_season_projections`."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.consensus_source import consensus_to_season_projections
from projections.schemas import (
    _PYARROW_STR,
    ConsensusProjectionSchema,
    Position,
    ProjectionSeasonSchema,
)


def _consensus_row(
    *,
    gsis_id: str,
    position: Position,
    has_points: bool,
    projected_points_ppr: float | None,
    consensus_adp: float | None = 10.0,
    consensus_rank: int | None = 1,
    asof: str = "2026-06-09",
    season: int = 2026,
) -> dict[str, object]:
    """One ConsensusProjectionSchema-shaped row. Stat-line cols are irrelevant to
    the adapter (it reads projected_points_ppr), so they are left null."""
    return {
        "gsis_id": gsis_id,
        "season": season,
        "asof": asof,
        "full_name": "Test Player",
        "position": position.value,
        "consensus_adp": consensus_adp,
        "consensus_rank": consensus_rank,
        "n_adp_sources": 2,
        "has_points": has_points,
        "projected_points_ppr": projected_points_ppr,
        "passing_yards": None,
        "passing_tds": None,
        "interceptions": None,
        "rushing_yards": None,
        "rushing_tds": None,
        "receptions": None,
        "receiving_yards": None,
        "receiving_tds": None,
        "fumbles_lost": None,
        "is_placeholder_gsis": False,
        "ruleset": "ESPN_PPR",
    }


def _consensus_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["asof"] = df["asof"].astype(_PYARROW_STR)
    df["full_name"] = df["full_name"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["ruleset"] = df["ruleset"].astype(_PYARROW_STR)
    return ConsensusProjectionSchema.validate(df)


def test_filters_to_has_points_and_maps_points_to_season_mean() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(
                gsis_id="00-3000001",
                position=Position.WR,
                has_points=True,
                projected_points_ppr=250.5,
            ),
            _consensus_row(
                gsis_id="00-3000002",
                position=Position.WR,
                has_points=False,
                projected_points_ppr=None,
            ),
        ]
    )
    out = consensus_to_season_projections(frame)
    assert list(out["gsis_id"]) == ["00-3000001"]  # ADP-only row dropped
    assert out["season_mean"].iloc[0] == pytest.approx(250.5)


def test_degenerate_distribution_and_metadata() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(
                gsis_id="00-1000001",
                position=Position.QB,
                has_points=True,
                projected_points_ppr=300.0,
                asof="2026-06-09",
                season=2026,
            ),
        ]
    )
    out = consensus_to_season_projections(frame)
    row = out.iloc[0]
    assert row["season_p10"] == row["season_p50"] == row["season_p90"] == row["season_mean"]
    assert row["n_weeks"] == 17
    assert row["model_id"] == "consensus:2026-06-09"
    assert row["ruleset"] == "ESPN_PPR"
    assert int(row["season"]) == 2026
    assert out["generated_at"].dt.tz is not None  # tz-aware, ProjectionSeasonSchema requires it


def test_output_validates_against_projection_season_schema() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(
                gsis_id="00-2000001",
                position=Position.RB,
                has_points=True,
                projected_points_ppr=220.0,
            ),
        ]
    )
    out = consensus_to_season_projections(frame)
    # Idempotent re-validation = it is a conforming frame.
    pd.testing.assert_frame_equal(out, ProjectionSeasonSchema.validate(out))


def test_empty_and_all_adp_only_inputs_return_valid_empty_frame() -> None:
    # No has_points rows -> valid empty ProjectionSeasonSchema frame (not an error).
    frame = _consensus_frame(
        [
            _consensus_row(
                gsis_id="00-4000001",
                position=Position.TE,
                has_points=False,
                projected_points_ppr=None,
            ),
        ]
    )
    out = consensus_to_season_projections(frame)
    assert out.empty
    ProjectionSeasonSchema.validate(out)


def test_raises_on_mixed_asof() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(
                gsis_id="00-3000001",
                position=Position.WR,
                has_points=True,
                projected_points_ppr=250.0,
                asof="2026-06-09",
            ),
            _consensus_row(
                gsis_id="00-3000002",
                position=Position.WR,
                has_points=True,
                projected_points_ppr=240.0,
                asof="2026-06-08",
            ),
        ]
    )
    with pytest.raises(ValueError, match="asof"):
        consensus_to_season_projections(frame)


def test_raises_on_mixed_season() -> None:
    frame = _consensus_frame(
        [
            _consensus_row(
                gsis_id="00-3000001",
                position=Position.WR,
                has_points=True,
                projected_points_ppr=250.0,
                asof="2026-06-09",
                season=2026,
            ),
            _consensus_row(
                gsis_id="00-3000002",
                position=Position.WR,
                has_points=True,
                projected_points_ppr=240.0,
                asof="2026-06-09",
                season=2025,
            ),
        ]
    )
    with pytest.raises(ValueError, match="season"):
        consensus_to_season_projections(frame)
