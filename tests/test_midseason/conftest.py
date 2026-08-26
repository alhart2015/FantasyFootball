"""Shared synthetic-league builders for the mid-season tests.

Pass 1 of the review flagged that three test modules each built the same synthetic roster;
the fix for it added a fourth. The league shape now lives here once, so changing it is one
edit rather than four coordinated ones with nothing failing if you miss the last.

The shape is deliberately awkward in the two ways real ESPN data is: the team ids are
non-contiguous and unsorted (like a league with deleted franchises), and the pairings give
every team exactly one game per week over a short season.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from projections.draft.league_calendar import LeagueCalendar
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset, VorpTableSchema

#: Non-contiguous and unsorted on purpose: `SlotMap` has to map these onto 1..n.
TEAM_IDS: list[int] = [17, 3, 11, 5, 9, 1]
REG_WEEKS = 4
POSITIONS: tuple[str, ...] = ("QB", "RB", "RB", "WR", "WR", "TE")
#: Every team plays exactly once a week.
PAIRINGS: list[list[tuple[int, int]]] = [
    [(17, 3), (11, 5), (9, 1)],
    [(17, 11), (3, 9), (5, 1)],
    [(17, 5), (11, 9), (3, 1)],
    [(17, 9), (5, 3), (11, 1)],
]
CALENDAR = LeagueCalendar(reg_weeks=REG_WEEKS, playoff_size=2, n_byes=0, final_weeks=1)

#: ESPN lineup slot ids: 0 QB, 2 RB, 4 WR, 6 TE, 20 BENCH.
ESPN_SLOT_COUNTS: dict[str, int] = {"0": 1, "2": 2, "4": 2, "6": 1, "20": 1}
#: ESPN defaultPositionId.
ESPN_POSITION_ID: dict[str, int] = {"QB": 1, "RB": 2, "WR": 3, "TE": 4}


def espn_player_id(team_id: int, index: int) -> int:
    return 100_000 + team_id * 100 + index


def gsis_id(team_id: int, index: int) -> str:
    return f"00-{team_id:04d}{index:03d}"


def league_config(n_teams: int = len(TEAM_IDS), *, name: str = "midseason_test") -> LeagueConfig:
    return LeagueConfig(
        name=name,
        n_teams=n_teams,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
        },
        ruleset=Ruleset.espn_half(),
    )


def vorp_pool(team_ids: list[int] | None = None) -> pd.DataFrame:
    """A schema-validated `VorpTableSchema` frame, strongest roster first in `team_ids` order.

    Validated rather than merely shaped like one: an unvalidated fixture keeps every test
    passing after the schema gains a required column, against a frame the production loader
    would reject.
    """
    ids = TEAM_IDS if team_ids is None else team_ids
    rows: list[dict[str, object]] = []
    for rank, team_id in enumerate(ids):
        for i, pos in enumerate(POSITIONS):
            mean = 240.0 - 20.0 * rank - 2.0 * i
            rows.append(
                {
                    "gsis_id": gsis_id(team_id, i),
                    "full_name": f"Player {team_id}-{i}",
                    "position": pos,
                    "season_mean_fpts": mean,
                    "vorp": mean - 80.0,
                    "replacement_fpts": 80.0,
                    "is_rookie": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    frame["position"] = frame["position"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(frame)


def id_map(team_ids: list[int] | None = None) -> pd.DataFrame:
    """The (espn_id, gsis_id) crosswalk `rosters_to_slots` joins on."""
    ids = TEAM_IDS if team_ids is None else team_ids
    pairs = [(str(espn_player_id(t, i)), gsis_id(t, i)) for t in ids for i in range(len(POSITIONS))]
    return pd.DataFrame(
        {
            "espn_id": pd.Series([e for e, _ in pairs], dtype=_PYARROW_STR),
            "gsis_id": pd.Series([g for _, g in pairs], dtype=_PYARROW_STR),
        }
    )


def schedule_frame(played_weeks: int = 0) -> pd.DataFrame:
    """`parse_schedule` output shape, with the first `played_weeks` weeks decided.

    Every home side wins 120-90, so a played week is a clean 1-0 for the home team.
    """
    rows: list[dict[str, object]] = []
    for week, games in enumerate(PAIRINGS, start=1):
        for home, away in games:
            played = week <= played_weeks
            rows.append(
                {
                    "week": week,
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_team": f"T{home}",
                    "away_team": f"T{away}",
                    "home_points": 120.0 if played else 0.0,
                    "away_points": 90.0 if played else 0.0,
                    "winner": "HOME" if played else "UNDECIDED",
                    "is_played": played,
                }
            )
    frame = pd.DataFrame(rows)
    for column in ("home_team", "away_team", "winner"):
        frame[column] = frame[column].astype(_PYARROW_STR)
    return frame


def espn_payload(*, played_weeks: int = 2, with_schedule: bool = True) -> dict[str, Any]:
    """A realistically shaped mid-season ESPN payload over the same league.

    Built from `PAIRINGS` so it cannot drift from `schedule_frame`.
    """
    schedule: list[dict[str, Any]] = []
    for week, games in enumerate(PAIRINGS, start=1):
        for home, away in games:
            played = week <= played_weeks
            schedule.append(
                {
                    "matchupPeriodId": week,
                    "winner": "HOME" if played else "UNDECIDED",
                    "home": {"teamId": home, "totalPoints": 120.0 if played else 0.0},
                    "away": {"teamId": away, "totalPoints": 90.0 if played else 0.0},
                }
            )

    teams: list[dict[str, Any]] = []
    for team_id in TEAM_IDS:
        entries = [
            {
                "lineupSlotId": 20,
                "playerId": espn_player_id(team_id, i),
                "playerPoolEntry": {
                    "player": {
                        "id": espn_player_id(team_id, i),
                        "fullName": f"Player {team_id}-{i}",
                        "defaultPositionId": ESPN_POSITION_ID[pos],
                        "proTeamId": 1,
                    }
                },
            }
            for i, pos in enumerate(POSITIONS)
        ]
        teams.append(
            {
                "id": team_id,
                "abbrev": f"T{team_id}",
                "name": f"Team {team_id}",
                "owners": [],
                "roster": {"entries": entries},
            }
        )

    return {
        "id": 999,
        "settings": {
            "name": "Pipeline Test League",
            "size": len(TEAM_IDS),
            "draftSettings": {"type": "SNAKE", "auctionBudget": 0, "keeperCount": 0},
            "rosterSettings": {"lineupSlotCounts": ESPN_SLOT_COUNTS},
            "scoringSettings": {"scoringItems": [{"statId": 53, "points": 0.5}]},
            "scheduleSettings": {
                "matchupPeriodCount": REG_WEEKS,
                "playoffTeamCount": CALENDAR.playoff_size,
                "playoffMatchupPeriodLength": CALENDAR.final_weeks,
            },
        },
        "teams": teams,
        "schedule": schedule if with_schedule else [],
        "members": [],
    }
