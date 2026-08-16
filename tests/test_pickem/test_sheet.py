"""Organizer-sheet reading and template writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.pickem.sheet import read_sheet, write_template
from projections.schemas import _PYARROW_STR, PickemSheetSchema


def _write_csv(path: Path, body: str) -> Path:
    path.write_text("away_team,home_team,home_spread\n" + body, encoding="utf-8")
    return path


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2025],
            "week": [1, 1, 1, 1],
            "game_id": pd.array(
                ["2026_01_NE_SEA", "2026_01_CHI_CAR", "2026_01_SF_LA", "2025_01_AA_BB"],
                dtype=_PYARROW_STR,
            ),
            "home_team": pd.array(["SEA", "CAR", "LA", "KC"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE", "CHI", "SF", "BUF"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                [
                    "2026-09-13T17:00:00Z",
                    "2026-09-13T13:00:00Z",
                    "2026-09-10T00:20:00Z",
                    "2025-09-07T17:00:00Z",
                ],
                utc=True,
            ).as_unit("us"),
        }
    )


# --- read_sheet ------------------------------------------------------------


def test_read_sheet_produces_a_valid_frame(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "sheet.csv", "NE,SEA,-3.5\nCHI,CAR,2.5\n")
    df = read_sheet(path, season=2026, week=1)
    PickemSheetSchema.validate(df)
    assert list(df["home_team"]) == ["SEA", "CAR"]
    assert list(df["home_spread"]) == [-3.5, 2.5]
    assert set(df["season"]) == {2026}
    assert set(df["week"]) == {1}


def test_read_sheet_normalizes_aliased_team_codes(tmp_path: Path) -> None:
    """The organizer may write JAX or WSH; canonical here is JAC / WAS."""
    path = _write_csv(tmp_path / "sheet.csv", "JAX,WSH,-1.5\n")
    df = read_sheet(path, season=2026, week=1)
    assert df.loc[0, "away_team"] == "JAC"
    assert df.loc[0, "home_team"] == "WAS"


def test_read_sheet_tolerates_whitespace_and_case(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "sheet.csv", " ne , sea ,-3.5\n")
    df = read_sheet(path, season=2026, week=1)
    assert df.loc[0, "away_team"] == "NE"
    assert df.loc[0, "home_team"] == "SEA"


def test_read_sheet_rejects_unknown_team_naming_the_row(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "sheet.csv", "NE,SEA,-3.5\nZZZ,CAR,2.5\n")
    with pytest.raises(ValueError, match="row 3"):
        read_sheet(path, season=2026, week=1)


def test_read_sheet_rejects_blank_spread_naming_the_row(tmp_path: Path) -> None:
    """A blank spread is the likeliest hand-entry slip, and it silently removes
    a game's underdog if allowed through."""
    path = _write_csv(tmp_path / "sheet.csv", "NE,SEA,-3.5\nCHI,CAR,\n")
    with pytest.raises(ValueError, match=r"blank on row\(s\) \[3\]"):
        read_sheet(path, season=2026, week=1)


def test_read_sheet_rejects_non_numeric_spread(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "sheet.csv", "NE,SEA,pick\n")
    with pytest.raises(ValueError, match="not a number"):
        read_sheet(path, season=2026, week=1)


def test_read_sheet_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "sheet.csv"
    path.write_text("away_team,home_team\nNE,SEA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="home_spread"):
        read_sheet(path, season=2026, week=1)


def test_read_sheet_rejects_duplicate_matchup(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "sheet.csv", "NE,SEA,-3.5\nNE,SEA,-4.0\n")
    with pytest.raises(ValueError, match="duplicate matchup"):
        read_sheet(path, season=2026, week=1)


def test_read_sheet_rejects_a_team_playing_itself(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "sheet.csv", "SEA,SEA,-3.5\n")
    with pytest.raises(ValueError, match="same team"):
        read_sheet(path, season=2026, week=1)


def test_read_sheet_accepts_a_zero_spread(tmp_path: Path) -> None:
    """0.0 is a legitimate true pick'em, distinct from a blank cell."""
    path = _write_csv(tmp_path / "sheet.csv", "NE,SEA,0\n")
    df = read_sheet(path, season=2026, week=1)
    assert df.loc[0, "home_spread"] == 0.0


# --- write_template --------------------------------------------------------


def test_write_template_has_one_row_per_game_that_week(tmp_path: Path) -> None:
    path = write_template(tmp_path / "t.csv", _schedules(), season=2026, week=1)
    written = pd.read_csv(path)
    assert len(written) == 3  # the 2025 row is a different season
    assert list(written.columns) == ["away_team", "home_team", "home_spread"]


def test_write_template_orders_by_kickoff(tmp_path: Path) -> None:
    path = write_template(tmp_path / "t.csv", _schedules(), season=2026, week=1)
    written = pd.read_csv(path)
    assert list(written["home_team"]) == ["LA", "CAR", "SEA"]


def test_write_template_leaves_the_spread_column_blank(tmp_path: Path) -> None:
    path = write_template(tmp_path / "t.csv", _schedules(), season=2026, week=1)
    written = pd.read_csv(path)
    assert written["home_spread"].isna().all()


def test_write_template_output_is_readable_once_spreads_are_filled(tmp_path: Path) -> None:
    """The round trip is the point: template out, numbers typed in, sheet back."""
    path = write_template(tmp_path / "t.csv", _schedules(), season=2026, week=1)
    filled = pd.read_csv(path)
    filled["home_spread"] = [-3.0, 2.5, -3.5]
    filled.to_csv(path, index=False)

    df = read_sheet(path, season=2026, week=1)
    PickemSheetSchema.validate(df)
    assert len(df) == 3


def test_write_template_raises_for_a_week_with_no_games(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no scheduled games"):
        write_template(tmp_path / "t.csv", _schedules(), season=2026, week=18)
