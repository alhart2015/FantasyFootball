"""Vegas team-context feature computes — tests."""

from __future__ import annotations

import pandas as pd


def _make_schedule_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic schedules frame with sane defaults for unspecified columns.

    Mirrors `SchedulesSchema`'s shape. Tests fill in only the columns the function
    under test reads, defaults the rest.
    """
    defaults: dict[str, object] = {
        "season": 2024,
        "week": 1,
        "game_id": "2024_01_HOME_AWAY",
        "home_team": "HOME",
        "away_team": "AWAY",
        "kickoff": pd.Timestamp("2024-09-08 17:00:00", tz="UTC"),
        "spread_line": 0.0,
        "total_line": 50.0,
        "home_moneyline": -110,
        "away_moneyline": -110,
        "surface": "grass",
        "roof": "outdoors",
        "temp": 70,
        "wind": 5,
    }
    out = []
    for r in rows:
        out.append({**defaults, **r})
    df = pd.DataFrame(out)
    df["temp"] = df["temp"].astype(pd.Int64Dtype())
    df["wind"] = df["wind"].astype(pd.Int64Dtype())
    df["surface"] = df["surface"].astype(pd.StringDtype("pyarrow"))
    df["roof"] = df["roof"].astype(pd.StringDtype("pyarrow"))
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True).astype("datetime64[us, UTC]")
    return df


def test_compute_returns_two_rows_per_game_and_four_feature_cols() -> None:
    """One schedule row → two output rows (home + away). Both carry
    preseason_implied_team_total / preseason_spread; both have NaN
    season_avg_* at week 1 (cold-start)."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": 3.0,  # KC favored by 3 (home favored → +spread_line)
                "total_line": 48.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)

    assert len(out) == 2
    assert set(out["team"]) == {"KC", "BAL"}
    for team, expected_spread, expected_itt in (
        ("KC", -3.0, 25.5),  # favorite: spread = -3, ITT = (48+3)/2
        ("BAL", 3.0, 22.5),  # dog: spread = +3, ITT = (48-3)/2
    ):
        row = out.loc[out["team"] == team].iloc[0]
        assert row["preseason_spread"] == expected_spread
        assert row["preseason_implied_team_total"] == expected_itt
        # season_avg_* NaN at week 1 (no prior games)
        assert pd.isna(row["season_avg_spread"])
        assert pd.isna(row["season_avg_implied_team_total"])


def test_compute_preseason_broadcasts_across_all_weeks() -> None:
    """A team's week-1 spread + ITT values are broadcast across all weeks of
    that team-season — same values appear in week 4, week 10, etc."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # Week 1: KC home vs BAL, KC favored by 3
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": 3.0,
                "total_line": 48.0,
            },
            # Week 2: KC away vs PHI, KC favored by 1 (away favored → -spread_line)
            {
                "season": 2024,
                "week": 2,
                "home_team": "PHI",
                "away_team": "KC",
                "spread_line": -1.0,
                "total_line": 46.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)

    kc_rows = out.loc[out["team"] == "KC"].sort_values("week")
    # preseason_* broadcasts week-1 values to all weeks of KC's season
    assert (kc_rows["preseason_spread"] == -3.0).all()
    assert (kc_rows["preseason_implied_team_total"] == 25.5).all()


def test_compute_season_avg_is_expanding_mean_shifted_by_one() -> None:
    """At week N, season_avg_* = mean of (spread, implied_team_total) over
    weeks 1..N-1. Leakage-safe: week N does NOT include its own value."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # KC vs BAL, week 1: KC favored 3, total 48 -> KC ITT=25.5, BAL ITT=22.5
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": 3.0,
                "total_line": 48.0,
            },
            # KC vs PHI, week 2: KC favored 1 (away favored), total 46 -> KC ITT=23.5
            {
                "season": 2024,
                "week": 2,
                "home_team": "PHI",
                "away_team": "KC",
                "spread_line": -1.0,
                "total_line": 46.0,
            },
            # KC vs NYJ, week 3: KC home favored 5, total 50 -> KC ITT=27.5
            {
                "season": 2024,
                "week": 3,
                "home_team": "KC",
                "away_team": "NYJ",
                "spread_line": 5.0,
                "total_line": 50.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].sort_values("week")
    # Week 1: NaN (no prior games)
    assert pd.isna(kc.iloc[0]["season_avg_spread"])
    assert pd.isna(kc.iloc[0]["season_avg_implied_team_total"])
    # Week 2: mean of week 1 only -> KC spread = -3, ITT = 25.5
    assert kc.iloc[1]["season_avg_spread"] == -3.0
    assert kc.iloc[1]["season_avg_implied_team_total"] == 25.5
    # Week 3: mean of weeks 1+2 -> KC spread = (-3 + -1) / 2 = -2, ITT = (25.5 + 23.5) / 2 = 24.5
    assert kc.iloc[2]["season_avg_spread"] == -2.0
    assert kc.iloc[2]["season_avg_implied_team_total"] == 24.5


def test_compute_season_avg_skips_bye_weeks_correctly() -> None:
    """A team's bye week produces no schedule row; the expanding mean updates
    only on weeks with actual games."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # KC week 1: home favored 3, total 48 -> KC ITT=25.5
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": 3.0,
                "total_line": 48.0,
            },
            # KC bye in week 2 (no row)
            # KC week 3: away favored 1, total 46 -> KC ITT=23.5
            {
                "season": 2024,
                "week": 3,
                "home_team": "PHI",
                "away_team": "KC",
                "spread_line": -1.0,
                "total_line": 46.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].sort_values("week")
    # Two rows: week 1 + week 3 (bye week 2 produces no row)
    assert list(kc["week"]) == [1, 3]
    # Week 3 sees week 1's value (only prior game)
    assert kc.iloc[1]["season_avg_spread"] == -3.0
    assert kc.iloc[1]["season_avg_implied_team_total"] == 25.5


def test_compute_independent_seasons_do_not_leak() -> None:
    """season_avg_* resets at the start of each season. Week 1 of 2024 is
    NaN regardless of 2023's values."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # 2023 KC games
            {
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": 3.0,
                "total_line": 48.0,
            },
            {
                "season": 2023,
                "week": 2,
                "home_team": "KC",
                "away_team": "PHI",
                "spread_line": 7.0,
                "total_line": 52.0,
            },
            # 2024 KC week 1
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "NYJ",
                "spread_line": 5.0,
                "total_line": 50.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)
    mask = (out["team"] == "KC") & (out["season"] == 2024) & (out["week"] == 1)
    kc_2024_wk1 = out.loc[mask].iloc[0]
    assert pd.isna(kc_2024_wk1["season_avg_spread"])
    assert pd.isna(kc_2024_wk1["season_avg_implied_team_total"])
    # And 2024 KC preseason_* uses 2024-week-1 line, not 2023's
    assert kc_2024_wk1["preseason_spread"] == -5.0
    assert kc_2024_wk1["preseason_implied_team_total"] == 27.5


def test_compute_sign_convention_favorite_negative_dog_positive() -> None:
    """Sanity-check: a team that was the favorite in its week-1 game has
    negative preseason_spread; a dog has positive. Matches
    _shared.build_game_environment's team-perspective convention."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            # 2023 KC vs DET week 1: KC home favored ~6 (positive spread_line = home favored)
            {
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "DET",
                "spread_line": 6.5,
                "total_line": 53.0,
            },
            # 2023 ARI vs WAS week 1: ARI home dog ~6.5 (negative spread_line = away favored)
            {
                "season": 2023,
                "week": 1,
                "home_team": "ARI",
                "away_team": "WAS",
                "spread_line": -6.5,
                "total_line": 41.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].iloc[0]
    ari = out.loc[out["team"] == "ARI"].iloc[0]
    assert kc["preseason_spread"] < 0  # favored
    assert ari["preseason_spread"] > 0  # dog


def test_compute_handles_nan_spread_line_propagates_nan() -> None:
    """A schedule row with NaN spread_line or total_line produces NaN in
    derived spread / implied_team_total / season_avg_* (NaN propagation
    via build_game_environment)."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": float("nan"),
                "total_line": 48.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)
    kc = out.loc[out["team"] == "KC"].iloc[0]
    assert pd.isna(kc["preseason_spread"])
    assert pd.isna(kc["preseason_implied_team_total"])


def test_compute_returns_sorted_by_season_week_team() -> None:
    """Output row order is (season, week, team) ascending — caller convenience."""
    from projections.features.vegas_team_context_features import (
        compute_vegas_team_context_features,
    )

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 3,
                "home_team": "BAL",
                "away_team": "NYJ",
                "spread_line": 4.0,
                "total_line": 45.0,
            },
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "spread_line": 3.0,
                "total_line": 48.0,
            },
        ]
    )
    out = compute_vegas_team_context_features(sch)
    # Lexicographic on (season, week, team)
    assert list(out["week"]) == [1, 1, 3, 3]
