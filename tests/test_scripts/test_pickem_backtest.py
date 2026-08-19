"""Pick'em backtest CLI tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scripts.pickem_backtest import (
    _parse_seasons,
    main,
    render_baseline,
    render_calibration,
)

from projections.pickem.backtest import baseline_week_scores, calibration_table
from projections.schemas import _PYARROW_STR
from projections.store import write_partition

_HOME = ["SEA", "CAR", "LAR", "CIN", "DET", "HOU"]
_AWAY = ["NE", "CHI", "SF", "TB", "NO", "BUF"]


def _games(season: int, week: int) -> pd.DataFrame:
    n = len(_HOME)
    spreads = [7.0, -3.0, 5.0, -1.0, 2.0, -6.0]
    return pd.DataFrame(
        {
            "season": [season] * n,
            "week": [week] * n,
            "game_id": pd.array(
                [f"{season}_{week:02d}_{_AWAY[i]}_{_HOME[i]}" for i in range(n)],
                dtype=_PYARROW_STR,
            ),
            "home_team": pd.array(_HOME, dtype=_PYARROW_STR),
            "away_team": pd.array(_AWAY, dtype=_PYARROW_STR),
            "game_type": pd.array(["REG"] * n, dtype=_PYARROW_STR),
            "spread_line": spreads,
            "home_moneyline": pd.array(
                [-300 if s > 0 else 250 for s in spreads], dtype=pd.Int64Dtype()
            ),
            "away_moneyline": pd.array(
                [250 if s > 0 else -300 for s in spreads], dtype=pd.Int64Dtype()
            ),
            "home_score": pd.array([24.0, 10.0, 31.0, 17.0, 20.0, 13.0], dtype=pd.Int64Dtype()),
            "away_score": pd.array([17.0, 24.0, 14.0, 20.0, 13.0, 27.0], dtype=pd.Int64Dtype()),
        }
    )


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    for season in (2023, 2024):
        frames = pd.concat([_games(season, w) for w in range(1, 5)], ignore_index=True)
        write_partition(tmp_path / "raw", "schedules", frames, season=season, week=None)
    return tmp_path


def test_parse_seasons_accepts_a_range() -> None:
    assert _parse_seasons("2015-2018") == [2015, 2016, 2017, 2018]


def test_parse_seasons_accepts_a_list() -> None:
    assert _parse_seasons("2023,2025") == [2023, 2025]


def test_main_runs_end_to_end(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--seasons", "2023-2024", "--data-root", str(data_root)]) == 0
    out = capsys.readouterr().out
    assert "MARKET CALIBRATION" in out
    assert "BASELINE" in out
    assert "BY SEASON" in out


def test_main_skips_missing_seasons(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--seasons", "2023-2025", "--data-root", str(data_root)]) == 0
    assert "no schedules partition for 2025" in capsys.readouterr().out


def test_main_exits_when_no_seasons_are_present(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no schedules found"):
        main(["--seasons", "2023", "--data-root", str(tmp_path)])


def test_render_calibration_reports_every_bin(data_root: Path) -> None:
    schedules = pd.concat([_games(2024, w) for w in range(1, 5)], ignore_index=True)
    rendered = render_calibration(calibration_table(schedules, n_bins=4))
    assert "PROBABILITY BIN" in rendered
    assert "Mean absolute error" in rendered


def test_render_baseline_states_that_it_is_a_floor(data_root: Path) -> None:
    """The number is easy to misread as a forecast; the report must say plainly
    that it models the staleness edge away."""
    schedules = pd.concat([_games(2024, w) for w in range(1, 5)], ignore_index=True)
    rendered = render_baseline(baseline_week_scores(schedules), min_dogs=3)
    assert "FLOOR" in rendered
    assert "Cost of the 3-dog rule" in rendered
