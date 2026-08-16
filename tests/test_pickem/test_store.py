"""Pick'em store round-trips."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.pickem.store import (
    read_picks,
    read_picks_season,
    read_sheet_partition,
    write_picks,
    write_sheet,
)
from projections.schemas import _PYARROW_STR


def _sheet(week: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [week, week],
            "home_team": pd.array(["SEA", "CAR"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE", "CHI"], dtype=_PYARROW_STR),
            "home_spread": [-3.5, 2.5],
        }
    )


def _picks(week: int = 1, *, pick: str = "SEA", graded: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026],
            "week": [week],
            "game_id": pd.array([f"2026_{week:02d}_NE_SEA"], dtype=_PYARROW_STR),
            "home_team": pd.array(["SEA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["NE"], dtype=_PYARROW_STR),
            "pick": pd.array([pick], dtype=_PYARROW_STR),
            "pick_win_prob": [0.62],
            "is_dog_pick": [False],
            "forced": [False],
            "switch_cost": [0.0],
            "winner": pd.array(["SEA" if graded else None], dtype=_PYARROW_STR),
            "correct": pd.array([True if graded else None], dtype=pd.BooleanDtype()),
        }
    )


def test_sheet_round_trip(tmp_path: Path) -> None:
    write_sheet(tmp_path, _sheet())
    back = read_sheet_partition(tmp_path, season=2026, week=1)
    assert list(back["home_team"]) == ["SEA", "CAR"]
    assert list(back["home_spread"]) == [-3.5, 2.5]


def test_picks_round_trip_preserves_nullable_dtypes(tmp_path: Path) -> None:
    """The NA in `correct` must survive parquet as NA, not become False."""
    write_picks(tmp_path, _picks())
    back = read_picks(tmp_path, season=2026, week=1)
    assert back["correct"].dtype == pd.BooleanDtype()
    assert back["correct"].isna().all()
    assert back["winner"].isna().all()


def test_grading_overwrites_the_same_partition(tmp_path: Path) -> None:
    """Picks are written Thursday and re-written Monday once graded — the week
    must end up with one row, not two."""
    write_picks(tmp_path, _picks())
    write_picks(tmp_path, _picks(graded=True))
    back = read_picks(tmp_path, season=2026, week=1)
    assert len(back) == 1
    assert bool(back.loc[0, "correct"])
    assert back.loc[0, "winner"] == "SEA"


def test_weeks_are_stored_separately(tmp_path: Path) -> None:
    write_picks(tmp_path, _picks(week=1))
    write_picks(tmp_path, _picks(week=2, pick="NE"))
    assert read_picks(tmp_path, season=2026, week=1).loc[0, "pick"] == "SEA"
    assert read_picks(tmp_path, season=2026, week=2).loc[0, "pick"] == "NE"


def test_read_picks_season_concatenates_every_week(tmp_path: Path) -> None:
    write_picks(tmp_path, _picks(week=1))
    write_picks(tmp_path, _picks(week=2, pick="NE"))
    season = read_picks_season(tmp_path, season=2026)
    assert len(season) == 2
    assert set(season["week"]) == {1, 2}


def test_writing_a_multi_week_frame_raises(tmp_path: Path) -> None:
    """A partition holds one week; a frame spanning two would silently write
    only the last one's coordinates."""
    both = pd.concat([_picks(week=1), _picks(week=2)], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        write_picks(tmp_path, both)


def test_writing_a_multi_week_sheet_raises(tmp_path: Path) -> None:
    both = pd.concat([_sheet(week=1), _sheet(week=2)], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        write_sheet(tmp_path, both)


def test_partition_path_layout(tmp_path: Path) -> None:
    path = write_picks(tmp_path, _picks(week=3))
    assert path.is_relative_to(tmp_path / "pickem" / "picks")
    assert "season=2026" in str(path)
    assert "week=03" in str(path)  # the store zero-pads week numbers
