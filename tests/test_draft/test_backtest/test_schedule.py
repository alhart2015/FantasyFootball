import numpy as np

from projections.draft.backtest.schedule import playoff_champion, regular_season_schedule


def test_every_team_plays_once_per_week_no_self_match() -> None:
    sched = regular_season_schedule(n_teams=16, n_weeks=14, rng=np.random.default_rng(0))
    assert len(sched) == 14
    for week in sched:
        assert len(week) == 8  # 16 teams -> 8 matchups
        seats = [s for matchup in week for s in matchup]
        assert sorted(seats) == list(range(1, 17))  # each seat exactly once
        for a, b in week:
            assert a != b


def test_deterministic_given_rng() -> None:
    a = regular_season_schedule(n_teams=16, n_weeks=14, rng=np.random.default_rng(7))
    b = regular_season_schedule(n_teams=16, n_weeks=14, rng=np.random.default_rng(7))
    assert a == b


def test_top_seed_wins_when_always_highest() -> None:
    seeds = [3, 1, 4, 5, 9, 2]  # seat ids in seed order (#1..#6); seat 3 is the #1 seed
    points = {
        15: {s: 10.0 for s in seeds},
        16: {s: 10.0 for s in seeds},
        17: {s: 10.0 for s in seeds},
    }
    for wk in (15, 16, 17):
        points[wk][3] = 100.0  # top seed always scores most
    champ = playoff_champion(seeds, points, playoff_weeks=(15, 16, 17))
    assert champ == 3


def test_odd_n_teams_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="even"):
        regular_season_schedule(n_teams=15, n_weeks=14, rng=np.random.default_rng(0))
