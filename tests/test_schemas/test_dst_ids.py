"""The 32 synthetic D/ST gsis_ids are persisted in parquet — they must never move.

See docs/superpowers/specs/2026-09-06-dst-projections-design.md §3.
"""

from __future__ import annotations

import pytest

from projections.schemas import (
    DST_GSIS_IDS,
    DST_TEAM_BY_GSIS,
    GSIS_ID_PATTERN,
    Team,
    validate_gsis_id,
)

#: Every id, pinned literally. This is the point of the test: the ids live in stored
#: partitions, so a reorder or a renumber must fail here rather than silently orphan
#: history. Adding a team (expansion) is the only legitimate reason to touch this, and
#: it appends — it never renumbers.
EXPECTED: dict[str, str] = {
    "ARI": "98-0000001",
    "ATL": "98-0000002",
    "BAL": "98-0000003",
    "BUF": "98-0000004",
    "CAR": "98-0000005",
    "CHI": "98-0000006",
    "CIN": "98-0000007",
    "CLE": "98-0000008",
    "DAL": "98-0000009",
    "DEN": "98-0000010",
    "DET": "98-0000011",
    "GB": "98-0000012",
    "HOU": "98-0000013",
    "IND": "98-0000014",
    "JAC": "98-0000015",
    "KC": "98-0000016",
    "LAC": "98-0000017",
    "LAR": "98-0000018",
    "LV": "98-0000019",
    "MIA": "98-0000020",
    "MIN": "98-0000021",
    "NE": "98-0000022",
    "NO": "98-0000023",
    "NYG": "98-0000024",
    "NYJ": "98-0000025",
    "PHI": "98-0000026",
    "PIT": "98-0000027",
    "SEA": "98-0000028",
    "SF": "98-0000029",
    "TB": "98-0000030",
    "TEN": "98-0000031",
    "WAS": "98-0000032",
}


def test_ids_are_frozen() -> None:
    assert {team.value: gsis for team, gsis in DST_GSIS_IDS.items()} == EXPECTED


def test_covers_every_team_exactly_once() -> None:
    assert set(DST_GSIS_IDS) == set(Team)
    assert len(DST_GSIS_IDS) == 32


def test_ids_are_injective() -> None:
    assert len(set(DST_GSIS_IDS.values())) == len(DST_GSIS_IDS)


@pytest.mark.parametrize("team", list(Team))
def test_every_id_is_a_valid_gsis_id(team: Team) -> None:
    raw = DST_GSIS_IDS[team]
    assert validate_gsis_id(raw) == raw


def test_ids_avoid_the_rookie_placeholder_block() -> None:
    """`ingest.external_projections` mints `99-XXXXXXX` for pre-camp rookies. A collision
    would make a defense look like an unreconciled rookie placeholder."""
    assert not any(gsis.startswith("99-") for gsis in DST_GSIS_IDS.values())


def test_ids_match_the_canonical_pattern() -> None:
    import re

    pattern = re.compile(rf"^{GSIS_ID_PATTERN}$")
    assert all(pattern.fullmatch(gsis) for gsis in DST_GSIS_IDS.values())


def test_inverse_round_trips() -> None:
    assert DST_TEAM_BY_GSIS == {gsis: team for team, gsis in DST_GSIS_IDS.items()}
    for team in Team:
        assert DST_TEAM_BY_GSIS[DST_GSIS_IDS[team]] is team
