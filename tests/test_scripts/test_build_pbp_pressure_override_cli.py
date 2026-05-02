"""scripts.build_pbp_pressure_override CLI tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_parse_args_defaults() -> None:
    """Default args produce documented defaults."""
    from scripts.build_pbp_pressure_override import _parse_season_range

    parsed = _parse_season_range("2018-2024")
    assert parsed == range(2018, 2025)


def test_parse_args_seasons_single_year() -> None:
    """`--seasons 2024` parses to range(2024, 2025)."""
    from scripts.build_pbp_pressure_override import _parse_season_range

    parsed = _parse_season_range("2024")
    assert parsed == range(2024, 2025)


def test_main_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: monkeypatched read_partition returning small synthetic
    frames; verify the parquet is written with the expected row count."""
    import scripts.build_pbp_pressure_override as mod

    pbp_2023 = pd.DataFrame(
        {
            "season": [2023] * 5,
            "week": [14, 15, 16, 17, 18],
            "posteam": ["KC"] * 5,
            "defteam": ["BAL"] * 5,
            "qb_dropback": [1.0] * 5,
            "qb_scramble": [0.0] * 5,
            "sack": [0.0] * 5,
        }
    )
    pbp_2024 = pd.DataFrame(
        {
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "posteam": ["KC"] * 5,
            "defteam": ["BAL"] * 5,
            "qb_dropback": [1.0] * 5,
            "qb_scramble": [0.0] * 5,
            "sack": [0.0] * 5,
        }
    )
    depth_charts_2024 = pd.DataFrame(
        {
            "gsis_id": ["00-0011111"],
            "season": [2024],
            "week": [5],
            "team": ["KC"],
            "position": ["QB"],
        }
    )
    schedules_2024 = pd.DataFrame(
        {
            "season": [2024],
            "week": [5],
            "home_team": ["KC"],
            "away_team": ["BAL"],
        }
    )

    def fake_read_partition(raw_root: Path, table: str, *, season: int) -> pd.DataFrame:
        if table == "pbp" and season == 2023:
            return pbp_2023
        if table == "pbp" and season == 2024:
            return pbp_2024
        if table == "depth_charts" and season == 2024:
            return depth_charts_2024
        if table == "schedules" and season == 2024:
            return schedules_2024
        raise FileNotFoundError(f"no fixture for ({table}, {season})")

    monkeypatch.setattr(mod, "read_partition", fake_read_partition)

    output = tmp_path / "pbp_pressure.parquet"
    rc = mod.main(["--seasons", "2024", "--data-root", str(tmp_path), "--output", str(output)])

    assert rc == 0
    assert output.exists()
    df = pd.read_parquet(output)
    assert len(df) == 1
    assert list(df.columns) == [
        "gsis_id",
        "season",
        "week",
        "team_sack_rate_allowed_l4",
        "team_qb_scramble_rate_l4",
        "team_def_sack_rate_l4",
        "team_def_scramble_rate_l4",
    ]


def test_main_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-existing output → parser.error exit (SystemExit, code != 0)."""
    import scripts.build_pbp_pressure_override as mod

    output = tmp_path / "pbp_pressure.parquet"
    output.write_bytes(b"placeholder")  # pre-existing

    with pytest.raises(SystemExit):
        mod.main(["--seasons", "2024", "--data-root", str(tmp_path), "--output", str(output)])
