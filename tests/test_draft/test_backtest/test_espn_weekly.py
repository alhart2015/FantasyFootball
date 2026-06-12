import json
from pathlib import Path

from projections.draft.backtest.espn_weekly import parse_espn_weekly
from projections.schemas import Ruleset

_FIX = Path(__file__).parent / "fixtures" / "espn_weekly_wk5_sample.json"


def test_parse_returns_one_row_per_projected_player() -> None:
    payload = json.loads(_FIX.read_text())
    df = parse_espn_weekly(payload, season=2025, week=5, ruleset=Ruleset.espn_half())
    assert {"espn_id", "season", "week", "position", "projected_points"} <= set(df.columns)
    assert (df["week"] == 5).all()
    assert df["projected_points"].notna().any()  # at least one real projection


def test_projected_points_are_half_ppr_nonnegative() -> None:
    payload = json.loads(_FIX.read_text())
    df = parse_espn_weekly(payload, season=2025, week=5, ruleset=Ruleset.espn_half())
    vals = df["projected_points"].dropna()
    assert (vals >= 0).all()


def test_player_with_no_weekly_entry_gets_null_projection() -> None:
    # synthetic player with no statSourceId=1/statSplitTypeId=1 entry -> projected_points is None
    payload = {
        "players": [
            {"player": {"id": 999999, "fullName": "Bye Guy", "defaultPositionId": 2, "stats": []}}
        ]
    }
    df = parse_espn_weekly(payload, season=2025, week=5, ruleset=Ruleset.espn_half())
    assert len(df) == 1
    assert df["projected_points"].isna().all()
