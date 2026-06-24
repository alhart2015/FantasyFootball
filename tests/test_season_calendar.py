from projections.season_calendar import last_regular_week, regular_season_games


def test_regular_season_games_era_split() -> None:
    assert regular_season_games(2020) == 16
    assert regular_season_games(2021) == 17


def test_last_regular_week_era_split() -> None:
    assert last_regular_week(2019) == 17
    assert last_regular_week(2020) == 17
    assert last_regular_week(2021) == 18
    assert last_regular_week(2024) == 18
