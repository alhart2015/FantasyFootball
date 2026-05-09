"""Weather feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest


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


def _make_index_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic player-team-week index frame matching
    `_build_player_team_week_index`'s output shape."""
    defaults: dict[str, object] = {
        "gsis_id": "00-0011111",
        "season": 2024,
        "week": 1,
        "team": "KC",
        "opp": "BAL",
        "position": "WR",
    }
    out = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(out)
    return df.astype(
        {
            "gsis_id": pd.StringDtype("pyarrow"),
            "season": pd.Int64Dtype(),
            "week": pd.Int64Dtype(),
            "team": pd.StringDtype("pyarrow"),
            "opp": pd.StringDtype("pyarrow"),
            "position": pd.StringDtype("pyarrow"),
        }
    )


def test_attach_weather_features_basic() -> None:
    """Joins compute output onto index on (season, week, team); preserves
    every input row."""
    from projections.features.weather_features import attach_weather_features

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
            {"gsis_id": "00-0022222", "season": 2024, "week": 1, "team": "BAL", "opp": "KC"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 12,
                "temp": 65,
                "roof": "outdoors",
                "surface": "grass",
            },
        ]
    )

    out = attach_weather_features(idx, sch)

    assert len(out) == 2
    # Both teams in the matchup pick up the same weather (game-level)
    for team in ("KC", "BAL"):
        row = out.loc[out["team"] == team].iloc[0]
        assert row["wind_speed_mph"] == 12.0
        assert row["temperature_f"] == 65.0
        assert row["is_high_wind"] == 0.0
        assert row["is_grass_surface"] == 1.0
    # Original index columns preserved
    assert set(out.columns) == {
        "gsis_id",
        "season",
        "week",
        "team",
        "opp",
        "position",
        "wind_speed_mph",
        "is_high_wind",
        "temperature_f",
        "is_grass_surface",
    }


def test_attach_weather_features_unmatched_index_row_propagates_nan() -> None:
    """Index row with no matching schedule entry → NaN in all four weather cols."""
    from projections.features.weather_features import attach_weather_features

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0011111", "season": 2024, "week": 99, "team": "KC", "opp": "BAL"},
        ]
    )
    sch = _make_schedule_rows(
        [
            # Different week — won't match.
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 12,
                "temp": 65,
                "roof": "outdoors",
                "surface": "grass",
            },
        ]
    )

    out = attach_weather_features(idx, sch)

    assert len(out) == 1
    row = out.iloc[0]
    assert pd.isna(row["wind_speed_mph"])
    assert pd.isna(row["is_high_wind"])
    assert pd.isna(row["temperature_f"])
    assert pd.isna(row["is_grass_surface"])


def test_attach_weather_features_preserves_index_row_count() -> None:
    """Multiple players on the same team-game produce multiple output rows
    (one per index row), all carrying identical weather."""
    from projections.features.weather_features import attach_weather_features

    idx = _make_index_rows(
        [
            {
                "gsis_id": "00-0011111",
                "season": 2024,
                "week": 1,
                "team": "KC",
                "opp": "BAL",
                "position": "QB",
            },
            {
                "gsis_id": "00-0022222",
                "season": 2024,
                "week": 1,
                "team": "KC",
                "opp": "BAL",
                "position": "WR",
            },
            {
                "gsis_id": "00-0033333",
                "season": 2024,
                "week": 1,
                "team": "KC",
                "opp": "BAL",
                "position": "TE",
            },
        ]
    )
    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 25,
                "temp": 30,
                "roof": "outdoors",
                "surface": "grass",
            },
        ]
    )

    out = attach_weather_features(idx, sch)

    assert len(out) == 3
    assert (out["wind_speed_mph"] == 25.0).all()
    assert (out["is_high_wind"] == 1.0).all()
    assert (out["temperature_f"] == 30.0).all()


def test_build_weather_overrides_returns_one_row_per_index_row() -> None:
    from projections.features.weather_features import build_weather_overrides

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
            {"gsis_id": "00-0022222", "season": 2024, "week": 2, "team": "KC", "opp": "CIN"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 10,
                "temp": 55,
                "roof": "outdoors",
            },
            {
                "season": 2024,
                "week": 2,
                "home_team": "KC",
                "away_team": "CIN",
                "wind": 5,
                "temp": 75,
                "roof": "outdoors",
            },
        ]
    )

    out = build_weather_overrides(sch, idx)
    assert len(out) == 2
    assert out.loc[out["week"] == 1, "wind_speed_mph"].iloc[0] == 10.0
    assert out.loc[out["week"] == 2, "wind_speed_mph"].iloc[0] == 5.0


def test_build_weather_overrides_raises_on_duplicate_index_keys() -> None:
    """Probe override must have unique (gsis_id, season, week) keys to merge
    cleanly into the probe runner. Duplicates raise immediately."""
    from projections.features.weather_features import build_weather_overrides

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
            {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 10,
                "roof": "outdoors",
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_weather_overrides(sch, idx)


def test_build_weather_overrides_raises_on_missing_required_index_columns() -> None:
    from projections.features.weather_features import build_weather_overrides

    bad = pd.DataFrame({"gsis_id": ["00-0011111"], "season": [2024]})
    sch = _make_schedule_rows([{}])

    with pytest.raises(ValueError, match="required column"):
        build_weather_overrides(sch, bad)


def test_build_weather_overrides_raises_on_invalid_gsis_id() -> None:
    """Index carrying a malformed gsis_id raises ValueError. Mirrors the
    sibling probe modules' boundary validation (CLAUDE.md #11)."""
    from projections.features.weather_features import build_weather_overrides

    idx = _make_index_rows(
        [
            {"gsis_id": "not-a-gsis-id", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 10,
                "roof": "outdoors",
            },
        ]
    )

    with pytest.raises(ValueError, match="invalid gsis_id"):
        build_weather_overrides(sch, idx)


def test_build_weather_overrides_raises_on_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `attach_weather_features` ever produces more rows than the input
    index (e.g., a future schedules-data quirk producing duplicate
    (season, week, team) rows from `compute_weather_features`), the assembler
    must fail loudly. Real schedule data won't produce duplicates from valid
    input, so we monkey-patch to simulate the regression."""
    from projections.features import weather_features
    from projections.features.weather_features import build_weather_overrides

    idx = _make_index_rows(
        [
            {"gsis_id": "00-0011111", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
        ]
    )
    sch = _make_schedule_rows(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "wind": 10,
                "roof": "outdoors",
            },
        ]
    )

    def _bad_attach(index: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
        # Simulate a many-to-many regression: returned frame has more rows
        # than the input index.
        doubled = pd.concat([index, index], ignore_index=True)
        doubled["wind_speed_mph"] = 0.0
        doubled["is_high_wind"] = 0.0
        doubled["temperature_f"] = 70.0
        doubled["is_grass_surface"] = 0.0
        return doubled

    monkeypatch.setattr(weather_features, "attach_weather_features", _bad_attach)

    with pytest.raises(AssertionError, match="row count mismatch"):
        build_weather_overrides(sch, idx)


def test_surface_codes_tuple_well_formed() -> None:
    """Pinned _SURFACE_CODES tuple is non-empty, contains 'grass', and every
    code is a clean snake-case-able string. Protects against silent drift if a
    future refactor mangles the constant."""
    from projections.features.weather_features import _SURFACE_CODES, _SURFACE_COL_NAMES

    assert len(_SURFACE_CODES) >= 4, "should have at least grass + 3 turf variants"
    assert "grass" in _SURFACE_CODES
    for code in _SURFACE_CODES:
        assert isinstance(code, str)
        assert code == code.lower(), f"{code!r} should be lowercase"
        assert " " not in code, f"{code!r} should not contain spaces"

    # Column names mirror the codes via lower + replace('-', '_').
    assert len(_SURFACE_COL_NAMES) == len(_SURFACE_CODES)
    assert "is_grass" in _SURFACE_COL_NAMES
    for name in _SURFACE_COL_NAMES:
        assert name.startswith("is_")
        assert "-" not in name
