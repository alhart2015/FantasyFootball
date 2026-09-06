"""Stored D/ST stat vectors -> pool rows the VORP generator can use.

Spec §5. Without this bridge a league with a D/ST slot has no defenses in its pool and every
mid-season tool skips each rostered one (issue #166).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.dst_pool import (
    DstPoolError,
    dst_display_name,
    load_dst_season_projections,
)
from projections.ingest.external_projections import refresh_dst_projections
from projections.schemas import DST_GSIS_IDS, Position, Ruleset, Team
from tests.test_scoring.test_dst import CRITTS_DST_POINTS

GEN_AT = pd.Timestamp("2026-09-06", tz="UTC")


def _ruleset() -> Ruleset:
    return Ruleset(
        name="ESPN_HALF",
        dst_stat_points=tuple(sorted(CRITTS_DST_POINTS.items(), key=lambda kv: int(kv[0]))),
    )


def _dst_player(pro_team_id: int, espn_id: int, stats: dict[str, float]) -> dict[str, object]:
    return {
        "player": {
            "id": espn_id,
            "fullName": "Test D/ST",
            "defaultPositionId": 16,
            "proTeamId": pro_team_id,
            "stats": [{"seasonId": 2026, "statSplitTypeId": 0, "statSourceId": 1, "stats": stats}],
        }
    }


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    payload = {
        "players": [
            # HOU: 3 sacks + 1 INT + shutout + 180 yds allowed = 13 under Critts.
            _dst_player(34, -16034, {"99": 3.0, "95": 1.0, "89": 1.0, "129": 1.0}),
            _dst_player(25, -16025, {"99": 1.0}),  # SF: 1
        ]
    }
    refresh_dst_projections(
        tmp_path,
        season=2026,
        asof=date(2026, 9, 6),
        payload=payload,
        # Two synthetic defenses on purpose; the 32-team gate is an ingest guard, not a
        # statement about what a fixture may contain.
        expect_all_teams=False,
    )
    return tmp_path


def test_scores_each_defense_under_the_caller_ruleset(data_root: Path) -> None:
    """The reason the stored table holds stats, not points: two leagues score the same defense
    differently, so scoring happens here at read time."""
    rows = load_dst_season_projections(
        data_root, season=2026, ruleset=_ruleset(), generated_at=GEN_AT
    )
    by_id = dict(zip(rows["gsis_id"], rows["season_mean"], strict=True))
    assert by_id[DST_GSIS_IDS[Team.HOU]] == pytest.approx(13.0)
    assert by_id[DST_GSIS_IDS[Team.SF]] == pytest.approx(1.0)


def test_a_different_ruleset_gives_different_points(data_root: Path) -> None:
    doubled = Ruleset(
        name="ESPN_HALF",
        dst_stat_points=tuple(
            sorted(((k, v * 2) for k, v in CRITTS_DST_POINTS.items()), key=lambda kv: int(kv[0]))
        ),
    )
    rows = load_dst_season_projections(data_root, season=2026, ruleset=doubled, generated_at=GEN_AT)
    by_id = dict(zip(rows["gsis_id"], rows["season_mean"], strict=True))
    assert by_id[DST_GSIS_IDS[Team.HOU]] == pytest.approx(26.0)


def test_rows_are_projection_season_shaped(data_root: Path) -> None:
    rows = load_dst_season_projections(
        data_root, season=2026, ruleset=_ruleset(), generated_at=GEN_AT
    )
    assert set(rows["position"]) == {Position.DST.value}
    assert set(rows["ruleset"]) == {"ESPN_HALF"}
    for column in ("gsis_id", "season", "n_weeks", "season_mean", "model_id", "generated_at"):
        assert column in rows.columns


def test_carries_a_display_name(data_root: Path) -> None:
    """Without it every tool renders a defense as a bare 98- id."""
    rows = load_dst_season_projections(
        data_root, season=2026, ruleset=_ruleset(), generated_at=GEN_AT
    )
    names = dict(zip(rows["gsis_id"], rows["full_name"], strict=True))
    assert names[DST_GSIS_IDS[Team.HOU]] == "HOU D/ST"


def test_display_name_matches_the_id_map_spelling() -> None:
    """The id_map and the pool must spell a defense identically or a name-based fallback
    lookup resolves on one side and not the other."""
    from projections.ingest.id_map import dst_id_map_rows

    rows = dst_id_map_rows()
    id_map_names = dict(zip(rows["gsis_id"], rows["full_name"], strict=True))
    for team in Team:
        assert dst_display_name(DST_GSIS_IDS[team]) == id_map_names[DST_GSIS_IDS[team]]


def test_a_missing_snapshot_raises_with_the_command_to_run(tmp_path: Path) -> None:
    """Silently returning nothing would look like a league that has no defenses."""
    with pytest.raises(DstPoolError, match="No dst_projections snapshot"):
        load_dst_season_projections(tmp_path, season=2026, ruleset=_ruleset(), generated_at=GEN_AT)


# --- review follow-ups (PR #168) -------------------------------------------


def test_a_ruleset_without_dst_scoring_raises_the_documented_error(data_root: Path) -> None:
    """DstPoolError, not the raw DstScoringError from deep inside the loop. Callers catch
    DstPoolError (generate_league_vorp_table.py), so the wrong type is an unhandled traceback
    where a clean message was intended."""
    with pytest.raises(DstPoolError, match="scores no D/ST categories"):
        load_dst_season_projections(
            data_root, season=2026, ruleset=Ruleset.espn_half(), generated_at=GEN_AT
        )


def test_an_explicit_asof_reads_that_snapshot(data_root: Path) -> None:
    rows = load_dst_season_projections(
        data_root,
        season=2026,
        ruleset=_ruleset(),
        generated_at=GEN_AT,
        asof=date(2026, 9, 6),
    )
    assert len(rows) == 2


def test_a_missing_asof_names_the_asof_it_looked_for(data_root: Path) -> None:
    """Previously this filtered the LATEST frame by the requested asof, silently produced an
    empty frame, and reported 'is empty' — never having read the snapshot asked for."""
    with pytest.raises(DstPoolError, match="asof=2026-08-01"):
        load_dst_season_projections(
            data_root,
            season=2026,
            ruleset=_ruleset(),
            generated_at=GEN_AT,
            asof=date(2026, 8, 1),
        )
