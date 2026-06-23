import pandas as pd

from projections.dfs.blend import blend_statlines
from projections.schemas import Ruleset, Stat

_KEY = ["gsis_id", "season", "week"]


def _ours() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["g1"],
            "season": [2023],
            "week": [5],
            Stat.RECEPTIONS.value: [4.0],
            Stat.RECEIVING_YARDS.value: [40.0],
        }
    )


def _sleeper() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["g1"],
            "season": [2023],
            "week": [5],
            "receptions": [6.0],
            "receiving_yards": [80.0],
        }
    )


def test_blend_50_50_in_statline_space() -> None:
    out = blend_statlines(_ours(), _sleeper(), weight_ours=0.5, ruleset=Ruleset.draftkings())
    # blended line: rec=5, rec_yd=60 -> 5*1 + 60/10 = 11.0
    assert round(float(out.set_index(_KEY).loc[("g1", 2023, 5), "blended_pts"]), 2) == 11.0


def test_weight_one_is_home_grown_only() -> None:
    out = blend_statlines(_ours(), _sleeper(), weight_ours=1.0, ruleset=Ruleset.draftkings())
    # ours: rec=4, rec_yd=40 -> 4 + 4 = 8.0
    assert round(float(out.set_index(_KEY).loc[("g1", 2023, 5), "blended_pts"]), 2) == 8.0


def test_sleeper_weekly_points_scores_statline() -> None:
    from projections.dfs.blend import sleeper_weekly_points

    out = sleeper_weekly_points(_sleeper(), ruleset=Ruleset.draftkings())
    # sleeper: rec=6, rec_yd=80 -> 6 + 8 = 14.0
    assert round(float(out.set_index(_KEY).loc[("g1", 2023, 5), "sleeper_pts"]), 2) == 14.0
