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
