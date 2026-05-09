"""build_weather_override CLI smokes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_parse_season_range_single_year() -> None:
    from scripts.build_weather_override import _parse_season_range

    r = _parse_season_range("2024")
    assert list(r) == [2024]


def test_parse_season_range_inclusive_range() -> None:
    from scripts.build_weather_override import _parse_season_range

    r = _parse_season_range("2018-2024")
    assert list(r) == [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def test_parse_args_defaults(tmp_path: Path) -> None:
    from scripts.build_weather_override import parse_args

    out = tmp_path / "weather.parquet"
    args = parse_args(["--output", str(out)])
    assert args.output == out
    assert list(args.seasons) == list(range(2018, 2025))
    assert args.data_root == Path("data")


def test_parse_args_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """If output exists and --force not set, parser exits."""
    from scripts.build_weather_override import parse_args

    out = tmp_path / "weather.parquet"
    out.write_text("")  # touch
    with pytest.raises(SystemExit):
        parse_args(["--output", str(out)])


def test_print_audit_includes_refined_unit_rates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`_print_audit` prints rates for the refined-unit columns
    (`is_cold_weather`, per-surface bools, `is_primetime`) in addition
    to the v1 rates. Direct unit test on a small synthetic frame —
    avoids the wall-time of a full main() integration test."""
    from scripts.build_weather_override import _print_audit

    from projections.features.weather_features import _SURFACE_COL_NAMES

    overrides = pd.DataFrame(
        {
            "wind_speed_mph": [0.0, 5.0, 25.0, 0.0],
            "is_high_wind": [0.0, 0.0, 1.0, 0.0],
            "temperature_f": [70.0, 30.0, 50.0, 70.0],
            "is_cold_weather": [0.0, 1.0, 0.0, 0.0],
            "is_grass_surface": [0.0, 1.0, 1.0, 0.0],
            "is_primetime": [0.0, 0.0, 1.0, 0.0],
            **{col: [0.0, 0.0, 0.0, 0.0] for col in _SURFACE_COL_NAMES},
        }
    )
    overrides["is_grass"] = [0.0, 1.0, 1.0, 0.0]

    schedules = pd.DataFrame(
        {
            "roof": pd.array(
                ["dome", "outdoors", "outdoors", "closed"],
                dtype=pd.StringDtype("pyarrow"),
            ),
        }
    )

    _print_audit(overrides, schedules)
    captured = capsys.readouterr()

    assert "weather override audit (4 rows)" in captured.out
    assert "is_cold_weather" in captured.out
    assert "is_primetime" in captured.out
    assert "is_grass" in captured.out
    # v1 lines still printed.
    assert "is_high_wind=1.0 rate (v1)" in captured.out
    assert "is_grass_surface=1.0 rate (v1)" in captured.out
    # Refined lines printed.
    assert "is_cold_weather=1.0 rate (refined)" in captured.out
    assert "is_primetime=1.0 rate (refined)" in captured.out
