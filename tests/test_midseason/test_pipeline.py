"""End-to-end cover for `project_league_standings`.

This is the coverage the branch was missing. The whole in-season pipeline used to live in
`scripts/projected_standings.py`, where nothing could reach it, and every silent wiring defect
the review found lived exactly there: rosters resolving to nothing, the rest-of-season module
never being called, an empty schedule read as a finished season. A synthetic ESPN payload
exercises the real path instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.midseason.standings import project_league_standings
from projections.schemas import _PYARROW_STR

_N_TEAMS = 6
_TEAM_IDS = [17, 3, 11, 5, 9, 1]
_REG_WEEKS = 4
#: ESPN lineup slot ids: 0 QB, 2 RB, 4 WR, 6 TE, 20 BENCH.
_SLOT_COUNTS = {"0": 1, "2": 2, "4": 2, "6": 1, "20": 1}
_POSITIONS = ("QB", "RB", "RB", "WR", "WR", "TE")
#: ESPN defaultPositionId: 1 QB, 2 RB, 3 WR, 4 TE.
_ESPN_POS_ID = {"QB": 1, "RB": 2, "WR": 3, "TE": 4}


def _espn_player_id(team_id: int, index: int) -> int:
    return 100_000 + team_id * 100 + index


def _gsis(team_id: int, index: int) -> str:
    return f"00-{team_id:04d}{index:03d}"


def _payload(*, played_weeks: int = 2, with_schedule: bool = True) -> dict[str, Any]:
    """A realistically shaped mid-season ESPN payload for a `_N_TEAMS`-team league."""
    pairings = [
        [(17, 3), (11, 5), (9, 1)],
        [(17, 11), (3, 9), (5, 1)],
        [(17, 5), (11, 9), (3, 1)],
        [(17, 9), (5, 3), (11, 1)],
    ]
    schedule: list[dict[str, Any]] = []
    for week, games in enumerate(pairings, start=1):
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
    for team_id in _TEAM_IDS:
        entries = [
            {
                "lineupSlotId": 20,
                "playerId": _espn_player_id(team_id, i),
                "playerPoolEntry": {
                    "player": {
                        "id": _espn_player_id(team_id, i),
                        "fullName": f"Player {team_id}-{i}",
                        "defaultPositionId": _ESPN_POS_ID[pos],
                        "proTeamId": 1,
                    }
                },
            }
            for i, pos in enumerate(_POSITIONS)
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
            "size": _N_TEAMS,
            "draftSettings": {"type": "SNAKE", "auctionBudget": 0, "keeperCount": 0},
            "rosterSettings": {"lineupSlotCounts": _SLOT_COUNTS},
            "scoringSettings": {"scoringItems": [{"statId": 53, "points": 0.5}]},
            "scheduleSettings": {
                "matchupPeriodCount": _REG_WEEKS,
                "playoffTeamCount": 2,
                "playoffMatchupPeriodLength": 1,
            },
        },
        "teams": teams,
        "schedule": schedule if with_schedule else [],
        "members": [],
    }


def _pool() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank, team_id in enumerate(_TEAM_IDS):
        for i, pos in enumerate(_POSITIONS):
            mean = 240.0 - 20.0 * rank - 2.0 * i
            rows.append(
                {
                    "gsis_id": _gsis(team_id, i),
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
    return frame


def _id_map() -> pd.DataFrame:
    pairs = [
        (str(_espn_player_id(t, i)), _gsis(t, i)) for t in _TEAM_IDS for i in range(len(_POSITIONS))
    ]
    return pd.DataFrame(
        {
            "espn_id": pd.Series([e for e, _ in pairs], dtype=_PYARROW_STR),
            "gsis_id": pd.Series([g for _, g in pairs], dtype=_PYARROW_STR),
        }
    )


def _run(*, played_weeks: int = 2, with_schedule: bool = True, n_sims: int = 80):
    pool = _pool()
    return project_league_standings(
        _payload(played_weeks=played_weeks, with_schedule=with_schedule),
        pool,
        _id_map(),
        PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
        VarianceParams.load(),
        season=2026,
        n_sims=n_sims,
        rng=np.random.default_rng(11),
    )


def test_the_pipeline_produces_standings_for_every_team() -> None:
    run = _run()
    assert len(run.standings) == _N_TEAMS
    assert set(run.standings["team_id"]) == set(_TEAM_IDS)
    assert run.league_name == "Pipeline Test League"


def test_the_pipeline_resolves_real_rosters_rather_than_simulating_zeros() -> None:
    """The defect that made this test necessary: rosters read through a `gsis_id` column ESPN
    never provides came back empty, every team scored zero, `0 >= 0` handed every matchup to
    the home side, and the output was a full table decided entirely by home-fixture counts.

    Two things falsify that here. Nothing is dropped, and the teams do not all look alike --
    the pool gives team 17 the strongest roster and team 1 the weakest.
    """
    run = _run()
    assert run.n_players_dropped == 0
    projected = run.standings.set_index("team_id")["projected_wins"]
    assert projected.loc[17] > projected.loc[1]
    assert run.standings["projected_points_for"].nunique() > 1


def test_the_calendar_comes_from_the_payload_not_the_defaults() -> None:
    run = _run()
    assert run.calendar.reg_weeks == _REG_WEEKS
    assert run.calendar.playoff_size == 2
    assert run.calendar.final_weeks == 1


def test_played_weeks_are_banked_and_not_replayed() -> None:
    """Two of four weeks played, every home side winning. The banked record must be real, and
    the projection cannot fall below it."""
    run = _run(played_weeks=2)
    assert run.snapshot_week == 3
    assert run.weeks_remaining == 2
    assert run.n_matchups_played == 6
    assert (run.standings["games_played"] == 2).all()
    assert (run.standings["projected_wins"] >= run.standings["wins"]).all()
    # 6 teams x 2 played weeks = 6 wins and 6 losses in total, counted once each.
    assert run.standings["wins"].sum() == 6
    assert run.standings["losses"].sum() == 6


def test_playoff_odds_are_an_identity_over_the_field() -> None:
    run = _run()
    assert run.standings["make_playoffs_pct"].sum() == pytest.approx(
        run.calendar.playoff_size
    )
    assert run.standings["champ_pct"].sum() == pytest.approx(1.0)


def test_matchup_odds_cover_the_remaining_fixtures_and_carry_their_snapshot() -> None:
    run = _run(played_weeks=2)
    assert (run.odds["snapshot_week"] == run.snapshot_week).all()
    assert set(run.odds["week"]) == {3, 4}
    assert len(run.odds) == 6  # 2 remaining weeks x 3 matchups


def test_a_preseason_payload_projects_a_full_season() -> None:
    run = _run(played_weeks=0)
    assert run.snapshot_week == 1
    assert run.weeks_remaining == _REG_WEEKS
    assert (run.standings["games_played"] == 0).all()
    assert run.standings["projected_wins"].sum() == pytest.approx(
        _REG_WEEKS * _N_TEAMS / 2, abs=1e-6
    )


def test_a_completed_season_leaves_nothing_to_simulate() -> None:
    run = _run(played_weeks=_REG_WEEKS)
    assert run.snapshot_week == _REG_WEEKS + 1
    assert run.weeks_remaining == 0
    assert run.odds.empty
    # Every record is known, so the standings are facts rather than a distribution.
    assert (run.standings["projected_wins"] == run.standings["wins"]).all()


def test_a_missing_schedule_raises_rather_than_reading_as_a_finished_season() -> None:
    """Without the guard, `first_unplayed_week` returns `reg_weeks + 1` (no unplayed week
    exists), the pool is silently zeroed, and the run dies later on an unrelated error."""
    with pytest.raises(ValueError, match="no schedule"):
        _run(with_schedule=False)


def test_rosters_that_resolve_to_nothing_raise() -> None:
    """An id_map that covers none of the league is an ingest failure, and simulating it would
    hand every matchup to the home team behind a plausible-looking table."""
    pool = _pool()
    empty_map = pd.DataFrame(
        {
            "espn_id": pd.Series([], dtype=_PYARROW_STR),
            "gsis_id": pd.Series([], dtype=_PYARROW_STR),
        }
    )
    with pytest.raises(ValueError, match="no rostered player"):
        project_league_standings(
            _payload(),
            pool,
            empty_map,
            PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
            VarianceParams.load(),
            season=2026,
            n_sims=10,
            rng=np.random.default_rng(0),
        )
