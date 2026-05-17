"""build_vegas_team_context_override CLI smokes."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_season_range_single_year() -> None:
    from scripts.build_vegas_team_context_override import _parse_season_range

    r = _parse_season_range("2024")
    assert list(r) == [2024]


def test_parse_season_range_inclusive_range() -> None:
    from scripts.build_vegas_team_context_override import _parse_season_range

    r = _parse_season_range("2018-2024")
    assert list(r) == [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def test_parse_args_defaults(tmp_path: Path) -> None:
    from scripts.build_vegas_team_context_override import parse_args

    out = tmp_path / "vegas_team_context.parquet"
    args = parse_args(["--output", str(out)])
    assert args.output == out
    assert list(args.seasons) == list(range(2018, 2025))
    assert args.data_root == Path("data")


def test_parse_args_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """If output exists and --force not set, parser exits."""
    from scripts.build_vegas_team_context_override import parse_args

    out = tmp_path / "vegas_team_context.parquet"
    out.write_text("")  # touch
    with pytest.raises(SystemExit):
        parse_args(["--output", str(out)])


def test_print_audit_includes_coverage_and_week1_rates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`_print_audit` prints per-column coverage rates, week-1 NaN rates,
    unique team-season counts, and histogram bounds. Direct unit test on a
    small synthetic frame — avoids the wall-time of a main() integration test."""
    import pandas as pd
    from scripts.build_vegas_team_context_override import _print_audit

    overrides = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "preseason_implied_team_total": [25.5, 25.5, 22.5, 22.5],
            "preseason_spread": [-3.0, -3.0, 3.0, 3.0],
            "season_avg_implied_team_total": [
                float("nan"),
                25.5,
                float("nan"),
                22.5,
            ],
            "season_avg_spread": [float("nan"), -3.0, float("nan"), 3.0],
        }
    )

    _print_audit(overrides)
    captured = capsys.readouterr()
    # Coverage rates printed for each feature col
    assert "preseason_spread coverage:" in captured.out
    assert "season_avg_spread coverage:" in captured.out
    # Week-1 NaN rate printed
    assert "season_avg_spread week-1 NaN rate:" in captured.out
    # Unique-team-season count per season
    assert "season 2024:" in captured.out
    # Histogram bounds
    assert "preseason_spread: min=" in captured.out
