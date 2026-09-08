"""The dashboard's standings page must price injuries exactly as the CLI does.

`scripts/projected_standings.py` shipped simulating on the raw pool, and the first fix for
that missed this page — a fourth consumer of `project_league_standings`, and the more
authoritative-looking of the two surfaces. A CLI and a web page disagreeing about one team's
playoff odds is worse than either being wrong alone.

Every other test of this route stubs `_run_projection` wholesale, which is why none of them
could have caught it. These drive `_run_projection` itself with the network stubbed and assert
on the pool the simulator receives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from projections.draft.assistant.performance_variance import VarianceParams
from projections.ingest.espn_league import EspnCredentials
from projections.midseason.swap_impact import injury_adjusted_pool_at_current_week
from projections.schemas import RosterSlot
from projections.web import DashboardConfig
from projections.web.routes import standings as standings_route


def _payload(*, injured_espn_id: int) -> dict[str, Any]:
    entries = [
        {
            # `injuryStatus` hangs off the PLAYER; `parse_rosters` reads
            # `entry.playerPoolEntry.player.injuryStatus`.
            "playerPoolEntry": {
                "player": {
                    "id": pid,
                    "fullName": f"P{pid}",
                    "defaultPositionId": 2,
                    "injuryStatus": "INJURY_RESERVE" if pid == injured_espn_id else "ACTIVE",
                }
            },
            "lineupSlotId": 2,
        }
        for pid in (101, 102)
    ]
    return {
        "settings": {
            "name": "Test League",
            "scheduleSettings": {"matchupPeriodCount": 2, "playoffTeamCount": 2},
            "rosterSettings": {"lineupSlotCounts": {"2": 1, "20": 3}},
            "scoringSettings": {"scoringItems": []},
            "draftSettings": {"auctionBudget": 200},
        },
        "teams": [
            {"id": 1, "name": "Alpha", "roster": {"entries": entries[:1]}},
            {"id": 2, "name": "Beta", "roster": {"entries": entries[1:]}},
        ],
        "schedule": [
            {
                "matchupPeriodId": 1,
                "home": {"teamId": 1},
                "away": {"teamId": 2},
                "winner": "UNDECIDED",
            },
            {
                "matchupPeriodId": 2,
                "home": {"teamId": 2},
                "away": {"teamId": 1},
                "winner": "UNDECIDED",
            },
        ],
    }


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0000101", "00-0000102"],
            "position": [RosterSlot.RB.value, RosterSlot.RB.value],
            "season_mean_fpts": [200.0, 200.0],
            "vorp": [50.0, 50.0],
            "replacement_fpts": [150.0, 150.0],
            "is_rookie": [False, False],
        }
    )


def _id_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0000101", "00-0000102"],
            "espn_id": ["101", "102"],
            "sleeper_id": ["s101", "s102"],
        }
    )


@pytest.fixture
def _captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    payload = _payload(injured_espn_id=101)

    monkeypatch.setattr(EspnCredentials, "resolve", classmethod(lambda cls, path: object()))
    monkeypatch.setattr(standings_route, "fetch_league_payload", lambda *a, **k: payload)
    monkeypatch.setattr(standings_route, "load_pool", lambda config: _pool())
    monkeypatch.setattr(standings_route, "load_id_map", lambda config: _id_map())
    monkeypatch.setattr(standings_route, "attach_is_rookie", lambda pool, **k: pool)
    monkeypatch.setattr(standings_route, "load_store_availability", lambda *a, **k: object())
    monkeypatch.setattr(VarianceParams, "load", classmethod(lambda cls, *a, **k: object()))

    def _capture(payload_arg: Any, pool_arg: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        captured["pool"] = pool_arg.copy()
        raise RuntimeError("stop here — the pool is what we came for")

    monkeypatch.setattr(standings_route, "project_league_standings", _capture)

    config = DashboardConfig(
        data_root=tmp_path,
        league_dir=tmp_path,
        pool_path=tmp_path / "pool.parquet",
        season=2026,
        league_id=1,
        my_team_id=1,
        credentials_path=tmp_path / "creds.json",
        n_sims=10,
    )
    with pytest.raises(RuntimeError):
        standings_route._run_projection(config, [])
    return {"captured": captured, "payload": payload}


def test_the_dashboard_simulator_receives_an_injury_adjusted_pool(
    _captured: dict[str, Any],
) -> None:
    """The regression. Before the fix both players reached the simulator at 200.0."""
    pool = _captured["captured"]["pool"]
    points = dict(zip(pool["gsis_id"].astype(str), pool["season_mean_fpts"], strict=True))

    assert points["00-0000101"] < 200.0, "an IR player must not simulate at full strength"
    assert points["00-0000102"] == pytest.approx(200.0), "a healthy player must be untouched"


def test_the_dashboard_and_the_cli_apply_the_identical_discount(
    _captured: dict[str, Any],
) -> None:
    """One league, two surfaces, one number. Both go through the shared helper, so a divergence
    here means one of them started deriving its own horizon again."""
    expected = injury_adjusted_pool_at_current_week(_pool(), _captured["payload"], _id_map())
    pd.testing.assert_series_equal(
        _captured["captured"]["pool"].set_index("gsis_id")["season_mean_fpts"].sort_index(),
        expected.set_index("gsis_id")["season_mean_fpts"].sort_index(),
        check_dtype=False,
    )
