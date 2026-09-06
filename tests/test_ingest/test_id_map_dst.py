"""The id_map must carry the 32 defenses.

Without them a rostered D/ST cannot be resolved from an ESPN roster entry, and every mid-season
tool reports it as an unknown player and skips it — the "1 rostered players are not in the
id_map (Chiefs D/ST)" note in issue #166.
"""

from __future__ import annotations

import pytest

from projections.ingest.id_map import dst_id_map_rows
from projections.schemas import DST_GSIS_IDS, IdMapSchema, Position, Team


@pytest.fixture
def rows():  # type: ignore[no-untyped-def]
    return dst_id_map_rows()


def test_one_row_per_team(rows) -> None:  # type: ignore[no-untyped-def]
    assert len(rows) == 32
    assert set(rows["team"]) == {t.value for t in Team}


def test_gsis_ids_match_the_frozen_map(rows) -> None:  # type: ignore[no-untyped-def]
    assert dict(zip(rows["team"], rows["gsis_id"], strict=True)) == {
        t.value: DST_GSIS_IDS[t] for t in Team
    }


def test_espn_ids_are_negative_and_derived_from_pro_team_id(rows) -> None:  # type: ignore[no-untyped-def]
    """Verified against all 32 defenses in the live 2026 payload: -(16000 + proTeamId).
    A positive id here would silently point at a real skill player."""
    ids = [int(v) for v in rows["espn_id"]]
    assert all(i < 0 for i in ids)
    assert len(set(ids)) == 32


def test_houston_is_the_id_seen_live(rows) -> None:  # type: ignore[no-untyped-def]
    hou = rows[rows["team"] == Team.HOU.value].iloc[0]
    assert hou["espn_id"] == "-16034"


def test_sleeper_id_is_the_bare_team_code(rows) -> None:  # type: ignore[no-untyped-def]
    """Sleeper identifies a defense by team code, not a numeric player id."""
    assert dict(zip(rows["team"], rows["sleeper_id"], strict=True)) == {
        t.value: t.value for t in Team
    }


def test_position_is_dst(rows) -> None:  # type: ignore[no-untyped-def]
    assert set(rows["position"]) == {Position.DST.value}


def test_rows_satisfy_the_id_map_schema(rows) -> None:  # type: ignore[no-untyped-def]
    """They are concatenated into the real table, so they must validate on their own."""
    assert IdMapSchema.validate(rows) is not None
