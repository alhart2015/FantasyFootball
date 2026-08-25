"""End-to-end cover for `project_league_standings`.

This is the coverage the branch was missing. The whole in-season pipeline used to live in
`scripts/projected_standings.py`, where nothing could reach it, and every silent wiring defect
the review found lived exactly there: rosters resolving to nothing, the rest-of-season module
never being called, an empty schedule read as a finished season. A synthetic ESPN payload
exercises the real path instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.midseason.standings import (
    ProjectionInputError,
    StandingsRun,
    project_league_standings,
)
from projections.schemas import _PYARROW_STR
from tests.test_midseason.conftest import (
    REG_WEEKS,
    TEAM_IDS,
    espn_payload,
    id_map,
    vorp_pool,
)

_N_TEAMS = len(TEAM_IDS)
_TEAM_IDS = TEAM_IDS
_REG_WEEKS = REG_WEEKS


def _run(*, played_weeks: int = 2, with_schedule: bool = True, n_sims: int = 80) -> StandingsRun:
    pool = vorp_pool()
    return project_league_standings(
        espn_payload(played_weeks=played_weeks, with_schedule=with_schedule),
        pool,
        id_map(),
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
    assert run.standings["make_playoffs_pct"].sum() == pytest.approx(run.calendar.playoff_size)
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
    with pytest.raises(ProjectionInputError, match="no schedule"):
        _run(with_schedule=False)


def test_rosters_that_resolve_to_nothing_raise() -> None:
    """An id_map that covers none of the league is an ingest failure, and simulating it would
    hand every matchup to the home team behind a plausible-looking table."""
    pool = vorp_pool()
    empty_map = pd.DataFrame(
        {
            "espn_id": pd.Series([], dtype=_PYARROW_STR),
            "gsis_id": pd.Series([], dtype=_PYARROW_STR),
        }
    )
    with pytest.raises(ProjectionInputError, match="no projectable players"):
        project_league_standings(
            espn_payload(),
            pool,
            empty_map,
            PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
            VarianceParams.load(),
            season=2026,
            n_sims=10,
            rng=np.random.default_rng(0),
        )
