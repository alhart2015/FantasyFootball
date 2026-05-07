"""Weather feature computes — tests."""

from __future__ import annotations

import pandas as pd


def _make_schedule_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic schedules frame with sane defaults for unspecified columns.

    Mirrors `SchedulesSchema`'s shape: nullable Int64 for wind/temp, pyarrow
    string for surface/roof, etc. Tests fill in only the columns the function
    under test reads, defaults the rest.
    """
    defaults: dict[str, object] = {
        "season": 2024,
        "week": 1,
        "game_id": "2024_01_KC_BAL",
        "home_team": "KC",
        "away_team": "BAL",
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
        merged = {**defaults, **r}
        out.append(merged)
    df = pd.DataFrame(out)
    # Match SchedulesSchema dtypes.
    df["temp"] = df["temp"].astype(pd.Int64Dtype())
    df["wind"] = df["wind"].astype(pd.Int64Dtype())
    df["surface"] = df["surface"].astype(pd.StringDtype("pyarrow"))
    df["roof"] = df["roof"].astype(pd.StringDtype("pyarrow"))
    return df


def test_outdoor_basic_returns_two_rows_per_game() -> None:
    """One schedule row → two output rows (home + away), both carry the same
    weather features (game-level attribute applies to both teams)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 10,
                "temp": 50,
                "roof": "outdoors",
                "surface": "grass",
            },
        ]
    )
    out = compute_weather_features(sch)

    assert len(out) == 2
    assert set(out["team"]) == {"KC", "BAL"}
    for team in ("KC", "BAL"):
        row = out.loc[out["team"] == team].iloc[0]
        assert row["wind_speed_mph"] == 10.0
        assert row["is_high_wind"] == 0.0
        assert row["temperature_f"] == 50.0
        assert row["is_grass_surface"] == 1.0


def test_high_wind_threshold_at_20_inclusive() -> None:
    """is_high_wind = 1.0 if wind_speed_mph >= 20.0 — boundary is inclusive."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 20,
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_high_wind"].tolist() == [1.0, 1.0]


def test_high_wind_threshold_below_20() -> None:
    """wind=19 → is_high_wind = 0.0 (strict < threshold)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 19,
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_high_wind"].tolist() == [0.0, 0.0]


def test_high_wind_threshold_well_above_20() -> None:
    """wind=35 → is_high_wind = 1.0."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 35,
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_high_wind"].tolist() == [1.0, 1.0]


def test_dome_roof_fills_wind_zero_temp_70() -> None:
    """roof=dome → wind_speed_mph=0.0, temperature_f=70.0, is_high_wind=0.0
    even when source wind/temp are NaN. Spec §3.5."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "MIN",
                "away_team": "DET",
                "wind": pd.NA,
                "temp": pd.NA,
                "roof": "dome",
                "surface": "fieldturf",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].tolist() == [0.0, 0.0]
    assert out["temperature_f"].tolist() == [70.0, 70.0]
    assert out["is_high_wind"].tolist() == [0.0, 0.0]
    assert out["is_grass_surface"].tolist() == [0.0, 0.0]


def test_closed_roof_treated_as_dome() -> None:
    """roof=closed → same fill as dome (spec §3.5: closed retracted = dome
    for game-time conditions)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 5,
                "home_team": "DAL",
                "away_team": "NYG",
                "wind": pd.NA,
                "temp": pd.NA,
                "roof": "closed",
                "surface": "matrixturf",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].tolist() == [0.0, 0.0]
    assert out["temperature_f"].tolist() == [70.0, 70.0]
    assert out["is_high_wind"].tolist() == [0.0, 0.0]


def test_outdoor_nan_wind_propagates_nan() -> None:
    """Outdoor game with NaN wind → wind_speed_mph=NaN, is_high_wind=NaN.
    The naive `(wind >= 20).astype(float)` would map NaN→False→0.0, masking
    the data-quality issue. Spec §3.2 requires NaN-preserving."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "BUF",
                "away_team": "NYJ",
                "wind": pd.NA,
                "temp": 45,
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].isna().all()
    assert out["is_high_wind"].isna().all()
    # temp is fine, only wind is NaN
    assert out["temperature_f"].tolist() == [45.0, 45.0]


def test_outdoor_nan_temp_propagates_nan() -> None:
    """Outdoor game with NaN temp → temperature_f=NaN; wind features unaffected."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "GB",
                "away_team": "DET",
                "wind": 10,
                "temp": pd.NA,
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["temperature_f"].isna().all()
    assert out["wind_speed_mph"].tolist() == [10.0, 10.0]
    assert out["is_high_wind"].tolist() == [0.0, 0.0]


def test_surface_grass_returns_one() -> None:
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "GB",
                "away_team": "DET",
                "surface": "grass",
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_grass_surface"].tolist() == [1.0, 1.0]


def test_surface_fieldturf_returns_zero() -> None:
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "DET",
                "away_team": "GB",
                "surface": "fieldturf",
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_grass_surface"].tolist() == [0.0, 0.0]


def test_surface_nan_returns_zero() -> None:
    """Null surface → is_grass_surface = 0.0 (informational fallback per spec §3.4)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "GB",
                "away_team": "DET",
                "surface": pd.NA,
                "roof": "outdoors",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["is_grass_surface"].tolist() == [0.0, 0.0]


def test_unknown_roof_treated_as_outdoor() -> None:
    """roof='open' (not in {dome, closed}) → no fill, NaN wind/temp propagate."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "GB",
                "away_team": "DET",
                "wind": pd.NA,
                "temp": pd.NA,
                "roof": "open",
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].isna().all()
    assert out["temperature_f"].isna().all()
    assert out["is_high_wind"].isna().all()


def test_null_roof_treated_as_outdoor() -> None:
    """roof=NaN → no fill (matches build_game_environment's
    `.isin([dome, closed]).fillna(False)` predicate)."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "GB",
                "away_team": "DET",
                "wind": pd.NA,
                "temp": pd.NA,
                "roof": pd.NA,
            },
        ]
    )
    out = compute_weather_features(sch)
    assert out["wind_speed_mph"].isna().all()
    assert out["temperature_f"].isna().all()


def test_output_columns_and_dtypes() -> None:
    """Output schema: (season, week, team, wind_speed_mph, is_high_wind,
    temperature_f, is_grass_surface). All four feature cols are Float64."""
    from projections.features.weather_features import compute_weather_features

    sch = _make_schedule_rows(
        [
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL"},
        ]
    )
    out = compute_weather_features(sch)

    expected_cols = {
        "season",
        "week",
        "team",
        "wind_speed_mph",
        "is_high_wind",
        "temperature_f",
        "is_grass_surface",
    }
    assert set(out.columns) == expected_cols
    for col in ("wind_speed_mph", "is_high_wind", "temperature_f", "is_grass_surface"):
        assert str(out[col].dtype) == "Float64", f"{col} dtype: {out[col].dtype}"
