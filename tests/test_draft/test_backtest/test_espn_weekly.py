import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_crosswalk_espn_to_gsis(monkeypatch: pytest.MonkeyPatch) -> None:
    from projections.draft.backtest import espn_weekly as ew
    from projections.schemas import WeeklyProjectionSchema

    payload = json.loads(_FIX.read_text())
    monkeypatch.setattr(ew, "_fetch_espn_week", lambda season, week, limit=800: payload)
    espn_ids = [str(p["player"]["id"]) for p in payload["players"]]
    id_map = pd.DataFrame(
        {
            "gsis_id": [f"00-{i:07d}" for i in range(len(espn_ids))],
            "espn_id": espn_ids,
            "position": ["RB"] * len(espn_ids),
        }
    )
    out = ew.weekly_projections_for_weeks(
        season=2025, weeks=[5], ruleset=Ruleset.espn_half(), id_map=id_map
    )
    WeeklyProjectionSchema.validate(out)
    assert set(out["gsis_id"]) <= set(id_map["gsis_id"])
    assert (out["week"] == 5).all()


@pytest.mark.network
def test_espn_weekly_live_shape() -> None:
    from projections.draft.backtest.espn_weekly import _fetch_espn_week

    payload = _fetch_espn_week(2025, 5)
    df = parse_espn_weekly(payload, season=2025, week=5, ruleset=Ruleset.espn_half())
    assert df["projected_points"].notna().sum() > 100  # wk5 had 621 projected players
