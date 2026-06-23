import pandas as pd

from projections.dfs.actuals import dk_weekly_actuals
from projections.schemas import Ruleset

_COLS = [
    "gsis_id",
    "season",
    "week",
    "position",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
]


def _ws(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLS)


def test_scores_dk_base_and_keeps_position() -> None:
    ws = _ws(
        [
            ["00-0036900", 2023, 5, "WR", 0, 0, 0, 0, 0, 6, 78, 1, 0],
        ]
    )
    out = dk_weekly_actuals(ws, ruleset=Ruleset.draftkings())
    row = out.iloc[0]
    # 6*1 + 78/10 + 1*6 = 6 + 7.8 + 6 = 19.8
    assert round(float(row["actual_points"]), 2) == 19.8
    assert row["position"] == "WR"


def test_drops_playoff_weeks_era_aware() -> None:
    # 2023 (18-week era): week 18 kept, week 19 dropped
    ws = _ws(
        [
            ["00-0000001", 2023, 18, "RB", 0, 0, 0, 50, 1, 0, 0, 0, 0],
            ["00-0000001", 2023, 19, "RB", 0, 0, 0, 99, 9, 0, 0, 0, 0],
        ]
    )
    out = dk_weekly_actuals(ws, ruleset=Ruleset.draftkings())
    assert out["week"].tolist() == [18]


def test_drops_non_skill_positions() -> None:
    ws = _ws([["00-0000002", 2023, 5, "K", 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert dk_weekly_actuals(ws, ruleset=Ruleset.draftkings()).empty


def test_bonus_column_adds_three_at_100_rec_yards() -> None:
    ws = _ws([["00-0000003", 2023, 5, "WR", 0, 0, 0, 0, 0, 8, 110, 0, 0]])
    out = dk_weekly_actuals(ws, ruleset=Ruleset.draftkings()).iloc[0]
    # base: 8*1 + 110/10 = 19.0 ; +3 bonus -> 22.0
    assert round(float(out["actual_points"]), 2) == 19.0
    assert round(float(out["actual_points_with_bonus"]), 2) == 22.0
