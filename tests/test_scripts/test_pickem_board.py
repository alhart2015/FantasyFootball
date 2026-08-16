"""Pick'em board CLI tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scripts.pickem_board import main, render_board

from projections.pickem.optimize import choose_picks
from projections.pickem.slate import build_slate
from projections.schemas import _PYARROW_STR
from projections.store import write_partition

_HOME = ["SEA", "CAR", "LAR", "CIN", "DET", "HOU"]
_AWAY = ["NE", "CHI", "SF", "TB", "NO", "BUF"]


def _schedules(
    *,
    spread_lines: list[float] | None = None,
    scores: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Six week-1 games. `spread_line` positive means the home team is favored."""
    n = len(_HOME)
    spreads = spread_lines if spread_lines is not None else [7.0] * n
    # Moneylines consistent with the spread sign: home favored -> shorter price.
    home_ml = [-300 if s > 0 else 250 for s in spreads]
    away_ml = [250 if s > 0 else -300 for s in spreads]
    home_score = [s[0] for s in scores] if scores else [float("nan")] * n
    away_score = [s[1] for s in scores] if scores else [float("nan")] * n
    return pd.DataFrame(
        {
            "season": [2026] * n,
            "week": [1] * n,
            "game_id": pd.array(
                [f"2026_01_{_AWAY[i]}_{_HOME[i]}" for i in range(n)], dtype=_PYARROW_STR
            ),
            "home_team": pd.array(_HOME, dtype=_PYARROW_STR),
            "away_team": pd.array(_AWAY, dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(["2026-09-13T17:00:00Z"] * n, utc=True).as_unit("us"),
            "spread_line": spreads,
            "home_moneyline": pd.array(home_ml, dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array(away_ml, dtype=pd.Int64Dtype()),
            "home_score": pd.array(home_score, dtype=pd.Int64Dtype()),
            "away_score": pd.array(away_score, dtype=pd.Int64Dtype()),
        }
    )


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    write_partition(tmp_path / "raw", "schedules", _schedules(), season=2026, week=None)
    return tmp_path


def _sheet_csv(path: Path, spreads: list[float]) -> Path:
    rows = "\n".join(f"{_AWAY[i]},{_HOME[i]},{spreads[i]}" for i in range(len(spreads)))
    path.write_text("away_team,home_team,home_spread\n" + rows + "\n", encoding="utf-8")
    return path


# --- template mode ----------------------------------------------------------


def test_template_mode_writes_the_weeks_matchups(data_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "sheet.csv"
    assert (
        main(
            [
                "--season",
                "2026",
                "--week",
                "1",
                "--template",
                str(out),
                "--data-root",
                str(data_root),
            ]
        )
        == 0
    )
    written = pd.read_csv(out)
    assert len(written) == 6
    assert list(written.columns) == ["away_team", "home_team", "home_spread"]
    assert written["home_spread"].isna().all()


def test_template_output_feeds_straight_back_in(data_root: Path, tmp_path: Path) -> None:
    """The round trip is the workflow: template out, spreads typed in, picks back."""
    out = tmp_path / "sheet.csv"
    main(["--season", "2026", "--week", "1", "--template", str(out), "--data-root", str(data_root)])
    filled = pd.read_csv(out)
    filled["home_spread"] = [-7.0] * 6
    filled.to_csv(out, index=False)

    assert (
        main(
            ["--season", "2026", "--week", "1", "--sheet", str(out), "--data-root", str(data_root)]
        )
        == 0
    )


# --- picks mode -------------------------------------------------------------


def test_picks_mode_saves_sheet_and_picks(data_root: Path, tmp_path: Path) -> None:
    sheet = _sheet_csv(tmp_path / "sheet.csv", [-7.0] * 6)
    assert (
        main(
            [
                "--season",
                "2026",
                "--week",
                "1",
                "--sheet",
                str(sheet),
                "--data-root",
                str(data_root),
            ]
        )
        == 0
    )
    assert (data_root / "pickem" / "picks" / "season=2026" / "week=01" / "part.parquet").exists()
    assert (data_root / "pickem" / "sheet" / "season=2026" / "week=01" / "part.parquet").exists()


def test_no_save_leaves_the_store_untouched(data_root: Path, tmp_path: Path) -> None:
    sheet = _sheet_csv(tmp_path / "sheet.csv", [-7.0] * 6)
    main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--sheet",
            str(sheet),
            "--data-root",
            str(data_root),
            "--no-save",
        ]
    )
    assert not (data_root / "pickem").exists()


def test_picks_mode_reports_exactly_three_dogs(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = _sheet_csv(tmp_path / "sheet.csv", [-7.0] * 6)
    main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--sheet",
            str(sheet),
            "--data-root",
            str(data_root),
            "--no-save",
        ]
    )
    out = capsys.readouterr().out
    assert "Underdog picks (3)" in out
    assert "Expected correct:" in out


def test_min_dogs_flag_is_honoured(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = _sheet_csv(tmp_path / "sheet.csv", [-7.0] * 6)
    main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--sheet",
            str(sheet),
            "--min-dogs",
            "5",
            "--data-root",
            str(data_root),
            "--no-save",
        ]
    )
    assert "Underdog picks (5)" in capsys.readouterr().out


def test_partial_sheet_is_flagged(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The organizer may leave games off; the board should say so rather than
    quietly picking a shorter slate."""
    sheet = _sheet_csv(tmp_path / "sheet.csv", [-7.0] * 4)
    main(
        [
            "--season",
            "2026",
            "--week",
            "1",
            "--sheet",
            str(sheet),
            "--data-root",
            str(data_root),
            "--no-save",
        ]
    )
    assert "sheet covers 4 of 6 scheduled games" in capsys.readouterr().out


def test_requires_a_mode() -> None:
    with pytest.raises(SystemExit, match="required"):
        main(["--season", "2026", "--week", "1"])


# --- the board itself -------------------------------------------------------


def test_board_highlights_a_free_dog(data_root: Path, tmp_path: Path) -> None:
    """Sheet has SEA as a +3 dog; the market has them favored by 7. That is the
    case the whole tool exists to surface."""
    schedules = _schedules()
    sheet_path = _sheet_csv(tmp_path / "sheet.csv", [3.0] + [-7.0] * 5)
    from projections.pickem.sheet import read_sheet

    sheet = read_sheet(sheet_path, season=2026, week=1)
    slate = build_slate(sheet, schedules)
    picks = choose_picks(slate)
    board = render_board(slate, picks, min_dogs=3)

    assert "FREE DOGS" in board
    assert "market flipped it" in board
    assert "LINE MOVES" in board


def test_board_marks_forced_dogs_when_there_is_no_free_one(data_root: Path, tmp_path: Path) -> None:
    from projections.pickem.sheet import read_sheet

    sheet = read_sheet(_sheet_csv(tmp_path / "s.csv", [-7.0] * 6), season=2026, week=1)
    slate = build_slate(sheet, _schedules())
    board = render_board(slate, choose_picks(slate), min_dogs=3)
    assert "forced dog" in board
    assert "FREE DOGS" not in board


# --- grade mode -------------------------------------------------------------


def test_grade_mode_scores_stored_picks(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = _sheet_csv(tmp_path / "sheet.csv", [-7.0] * 6)
    main(["--season", "2026", "--week", "1", "--sheet", str(sheet), "--data-root", str(data_root)])

    # Home wins every game; picks were the three biggest home favorites plus
    # three forced away dogs, so exactly 3 of 6 land.
    write_partition(
        data_root / "raw",
        "schedules",
        _schedules(scores=[(24, 17)] * 6),
        season=2026,
        week=None,
    )
    capsys.readouterr()
    assert main(["--season", "2026", "--week", "1", "--grade", "--data-root", str(data_root)]) == 0
    out = capsys.readouterr().out
    assert "3 of 6 correct" in out
    assert "underdog picks: 0 of 3 correct" in out


def test_grade_mode_persists_the_result(data_root: Path, tmp_path: Path) -> None:
    from projections.pickem.store import read_picks

    sheet = _sheet_csv(tmp_path / "sheet.csv", [-7.0] * 6)
    main(["--season", "2026", "--week", "1", "--sheet", str(sheet), "--data-root", str(data_root)])
    write_partition(
        data_root / "raw", "schedules", _schedules(scores=[(24, 17)] * 6), season=2026, week=None
    )
    main(["--season", "2026", "--week", "1", "--grade", "--data-root", str(data_root)])

    stored = read_picks(data_root, season=2026, week=1)
    assert stored["correct"].notna().all()
    assert int(stored["correct"].sum()) == 3
