"""ESPN D/ST ingest — long-format projection rows.

Spec: docs/superpowers/specs/2026-09-06-dst-projections-design.md §5, and DstProjectionSchema
for why the shape is long rather than wide.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from projections.ingest.external_projections import (
    ExternalProjectionError,
    parse_espn_dst,
    refresh_dst_projections,
)
from projections.schemas import DST_GSIS_IDS, DstProjectionSchema, ProjectionSource, Team
from projections.scoring.dst import score_dst
from projections.store import read_partition
from tests.test_scoring.test_dst import CRITTS_DST_POINTS

#: ESPN's proTeamId for Houston, and the negative player id it gives that defense. Both are
#: real values observed in the live 2026 payload — ESPN identifies a D/ST by a NEGATIVE id.
HOU_PRO_TEAM_ID = 34
HOU_DST_ESPN_ID = -16034


def _dst_player(
    pro_team_id: int,
    espn_id: int,
    stats: dict[str, float],
    *,
    season: int = 2026,
    split: int = 0,
    source: int = 1,
) -> dict[str, Any]:
    return {
        "player": {
            "id": espn_id,
            "fullName": "Test D/ST",
            "defaultPositionId": 16,
            "proTeamId": pro_team_id,
            "stats": [
                {
                    "seasonId": season,
                    "statSplitTypeId": split,
                    "statSourceId": source,
                    "stats": stats,
                }
            ],
        }
    }


def _payload(players: list[dict[str, Any]]) -> dict[str, Any]:
    return {"players": players}


def test_parses_one_row_per_stat() -> None:
    df = parse_espn_dst(
        _payload([_dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4, "95": 0.7})]), 2026
    )
    assert len(df) == 2
    assert set(df["stat_id"]) == {"99", "95"}
    assert df["value"].tolist() == [2.4, 0.7]


def test_uses_the_synthetic_team_gsis_id() -> None:
    df = parse_espn_dst(
        _payload([_dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4})]), 2026
    )
    assert df["gsis_id"].iloc[0] == DST_GSIS_IDS[Team.HOU]
    assert df["team"].iloc[0] == Team.HOU.value
    assert df["source"].iloc[0] == ProjectionSource.ESPN.value


def test_carries_espns_negative_player_id_verbatim() -> None:
    """ESPN's D/ST ids are negative. Storing it lets a later id_map join work; mangling it
    into a positive number would silently point at a real player."""
    df = parse_espn_dst(
        _payload([_dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4})]), 2026
    )
    assert df["source_player_id"].iloc[0] == str(HOU_DST_ESPN_ID)


def test_skips_skill_players() -> None:
    payload = _payload(
        [
            {"player": {"id": 3139477, "defaultPositionId": 1, "proTeamId": 12, "stats": []}},
            _dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4}),
        ]
    )
    assert len(parse_espn_dst(payload, 2026)) == 1


def test_skips_a_defense_with_no_season_projection() -> None:
    """statSplitTypeId 1 is the weekly split; only the season total (0) belongs in this
    preseason table."""
    payload = _payload([_dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4}, split=1)])
    assert parse_espn_dst(payload, 2026).empty


def test_skips_actuals_rather_than_projections() -> None:
    """statSourceId 0 is what actually happened; 1 is the projection."""
    payload = _payload([_dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4}, source=0)])
    assert parse_espn_dst(payload, 2026).empty


def test_skips_a_different_season() -> None:
    payload = _payload([_dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4}, season=2025)])
    assert parse_espn_dst(payload, 2026).empty


def test_skips_an_unknown_pro_team() -> None:
    """proTeamId 0 is 'free agent', which a real defense never is. Guessing from the name
    would put a malformed row in the store."""
    assert parse_espn_dst(_payload([_dst_player(0, -1, {"99": 2.4})]), 2026).empty


def test_empty_payload_yields_typed_empty_frame() -> None:
    df = parse_espn_dst(_payload([]), 2026)
    assert df.empty
    assert list(df.columns) == [
        "source",
        "source_player_id",
        "gsis_id",
        "team",
        "season",
        "stat_id",
        "value",
    ]


# --- refresh ---------------------------------------------------------------


def test_refresh_writes_a_validated_snapshot(tmp_path: Path) -> None:
    payload = _payload(
        [
            _dst_player(HOU_PRO_TEAM_ID, HOU_DST_ESPN_ID, {"99": 2.4, "95": 0.7}),
            _dst_player(25, -16025, {"99": 3.1, "89": 0.05}),  # SF
        ]
    )
    out = refresh_dst_projections(tmp_path, season=2026, asof=date(2026, 9, 6), payload=payload)
    assert out.exists()

    stored = read_partition(tmp_path / "raw", "dst_projections", season=2026, asof=date(2026, 9, 6))
    stored = DstProjectionSchema.validate(stored)
    assert len(stored) == 4
    assert set(stored["asof"]) == {"2026-09-06"}
    assert stored["gsis_id"].nunique() == 2


def test_refresh_refuses_an_empty_snapshot(tmp_path: Path) -> None:
    """Writing zero defenses would read back later as 'ESPN has no defenses this year'."""
    with pytest.raises(ExternalProjectionError, match="no D/ST projections"):
        refresh_dst_projections(tmp_path, season=2026, asof=date(2026, 9, 6), payload=_payload([]))


# --- the round trip that matters -------------------------------------------


def test_stored_rows_score_through_the_scoring_layer(tmp_path: Path) -> None:
    """The point of the long format: stored rows feed score_dst directly, with no column
    interpreted along the way. 3 sacks + 1 INT + shutout + 180 yds under Critts = 13."""
    from projections.schemas import Ruleset

    payload = _payload(
        [
            _dst_player(
                HOU_PRO_TEAM_ID,
                HOU_DST_ESPN_ID,
                {"99": 3.0, "95": 1.0, "89": 1.0, "129": 1.0},
            )
        ]
    )
    refresh_dst_projections(tmp_path, season=2026, asof=date(2026, 9, 6), payload=payload)
    stored = read_partition(tmp_path / "raw", "dst_projections", season=2026, asof=date(2026, 9, 6))

    ruleset = Ruleset(
        name="ESPN_HALF",
        dst_stat_points=tuple(sorted(CRITTS_DST_POINTS.items(), key=lambda kv: int(kv[0]))),
    )
    vector = dict(zip(stored["stat_id"], stored["value"], strict=True))
    assert score_dst(vector, ruleset) == pytest.approx(13.0)


def test_duplicate_stat_rows_are_rejected() -> None:
    """A duplicate would silently double that category when the dot product sums the rows."""
    row = {
        "source": ProjectionSource.ESPN.value,
        "source_player_id": "-16034",
        "gsis_id": DST_GSIS_IDS[Team.HOU],
        "team": Team.HOU.value,
        "season": 2026,
        "asof": "2026-09-06",
        "stat_id": "99",
        "value": 2.4,
    }
    with pytest.raises(Exception, match=r"(?i)unique|duplicate"):
        DstProjectionSchema.validate(pd.DataFrame([row, row]))
