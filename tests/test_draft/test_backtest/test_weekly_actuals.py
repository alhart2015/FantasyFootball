import pandas as pd

from projections.draft.backtest.weekly_actuals import build_weekly_actuals
from projections.schemas import Ruleset, WeeklyActualSchema


def _ws_row(
    gsis: str,
    week: int,
    receptions: int = 0,
    rec_yds: float = 0.0,
    rush_yds: float = 0.0,
    rush_td: int = 0,
) -> dict[str, object]:
    return {
        "gsis_id": gsis,
        "season": 2025,
        "week": week,
        "position": "RB",
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "rushing_yards": rush_yds,
        "rushing_tds": rush_td,
        "receptions": receptions,
        "receiving_yards": rec_yds,
        "receiving_tds": 0,
        "fumbles_lost": 0,
    }


def test_scores_half_ppr_per_week() -> None:
    ws = pd.DataFrame(
        [_ws_row("00-0000001", 5, receptions=4, rec_yds=40.0, rush_yds=50.0, rush_td=1)]
    )
    out = build_weekly_actuals(ws, ruleset=Ruleset.espn_half())
    # half-PPR: 4 rec * 0.5 + 40*0.1 + 50*0.1 + 1*6 = 2 + 4 + 5 + 6 = 17.0
    assert float(out.loc[0, "actual_points"]) == 17.0
    WeeklyActualSchema.validate(out)


def test_excludes_week_18() -> None:
    ws = pd.DataFrame([_ws_row("00-0000001", 18, rush_yds=100.0)])
    out = build_weekly_actuals(ws, ruleset=Ruleset.espn_half())
    assert len(out) == 0


def test_one_row_per_player_week() -> None:
    ws = pd.DataFrame(
        [_ws_row("00-0000001", 5), _ws_row("00-0000001", 6), _ws_row("00-0000002", 5)]
    )
    out = build_weekly_actuals(ws, ruleset=Ruleset.espn_half())
    assert len(out) == 3
