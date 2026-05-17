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
