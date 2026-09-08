"""The projected-standings run must reach the simulator with injuries priced in.

It shipped calling `project_league_standings` on the RAW pool, so every projected finish in
the league simulated an injured player at full strength -- a team carrying an IR back read as
though he would play all seventeen games. The waiver and trade tools already adjusted before
simulating; the one tool whose entire output IS the simulation did not, which made its numbers
the least trustworthy of the three and the most authoritative-looking.

These drive `main` with the network stubbed, and assert on the pool the simulator actually
receives -- the bug was a missing call at the seam, so testing the helper alone would not have
caught it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import projected_standings
import pytest

from projections.draft.assistant.performance_variance import VarianceParams
from projections.ingest.espn_league import EspnCredentials
from projections.midseason import context as ctx_mod
from projections.midseason.standings import ProjectionInputError
from projections.midseason.swap_impact import injury_adjusted_pool
from projections.schemas import RosterSlot


def _payload(*, injured_espn_id: int) -> dict[str, Any]:
    """Two teams, a two-week schedule, and one player carrying an IR designation."""
    entries = [
        {
            # `injuryStatus` hangs off the PLAYER, not the entry -- `parse_rosters` reads
            # `entry.playerPoolEntry.player.injuryStatus`. Putting it on the entry makes the
            # fixture silently healthy, and the first cut of this test did exactly that.
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


def _pool(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "gsis_id": ["00-0000101", "00-0000102"],
            "position": [RosterSlot.RB.value, RosterSlot.RB.value],
            "season_mean_fpts": [200.0, 200.0],
            "vorp": [50.0, 50.0],
            "replacement_fpts": [150.0, 150.0],
        }
    )
    path = tmp_path / "pool.parquet"
    frame.to_parquet(path)
    return path


def _id_map(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "gsis_id": ["00-0000101", "00-0000102"],
            "espn_id": ["101", "102"],
            "sleeper_id": ["s101", "s102"],
        }
    )
    root = tmp_path / "raw"
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "id_map.parquet")
    return root / "id_map.parquet"


@pytest.fixture
def _stubbed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub every network and history read; capture the pool the simulator is handed."""
    captured: dict[str, Any] = {}
    payload = _payload(injured_espn_id=101)
    _id_map(tmp_path)

    monkeypatch.setattr(EspnCredentials, "resolve", classmethod(lambda cls, path: object()))
    # The fetch and the history scans now live in the shared context, so they are patched
    # there. The assertion is unchanged: what pool does the simulator receive?
    monkeypatch.setattr(ctx_mod, "fetch_league_payload", lambda *a, **k: payload)
    monkeypatch.setattr(ctx_mod, "attach_is_rookie", lambda pool, **k: pool.assign(is_rookie=False))
    monkeypatch.setattr(ctx_mod, "load_store_availability", lambda *a, **k: object())
    monkeypatch.setattr(VarianceParams, "load", classmethod(lambda cls, *a, **k: object()))

    def _capture(payload_arg: Any, pool_arg: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        captured["pool"] = pool_arg.copy()
        raise ProjectionInputError("stop here — the pool is what we came for")

    monkeypatch.setattr(projected_standings, "project_league_standings", _capture)
    return {"captured": captured, "payload": payload, "pool_path": _pool(tmp_path)}


def test_the_simulator_receives_an_injury_adjusted_pool(
    _stubbed: dict[str, Any], tmp_path: Path
) -> None:
    """The regression. Before the fix this pool was the raw one and both players read 200.0."""
    projected_standings.main(
        [
            "--league-id",
            "1",
            "--season",
            "2026",
            "--team-id",
            "1",
            "--pool",
            str(_stubbed["pool_path"]),
            "--data-root",
            str(tmp_path),
            "--n-sims",
            "10",
        ]
    )
    pool = _stubbed["captured"]["pool"]
    points = dict(zip(pool["gsis_id"].astype(str), pool["season_mean_fpts"], strict=True))

    # 00-0000101 is on IR: four games missed out of seventeen remaining in week 1.
    assert points["00-0000101"] < 200.0, "an IR player must not simulate at full strength"
    assert points["00-0000102"] == pytest.approx(200.0), "a healthy player must be untouched"


def test_the_adjustment_matches_the_shared_helper_exactly(
    _stubbed: dict[str, Any], tmp_path: Path
) -> None:
    """Not a re-implementation. The waiver and trade tools must price the same injury the same
    way, or a combined report shows two different numbers for one player."""
    projected_standings.main(
        [
            "--league-id",
            "1",
            "--season",
            "2026",
            "--team-id",
            "1",
            "--pool",
            str(_stubbed["pool_path"]),
            "--data-root",
            str(tmp_path),
            "--n-sims",
            "10",
        ]
    )
    raw = pd.read_parquet(_stubbed["pool_path"]).assign(is_rookie=False)
    expected = injury_adjusted_pool(
        raw, _stubbed["payload"], pd.read_parquet(tmp_path / "raw" / "id_map.parquet"), week=1
    )
    got = _stubbed["captured"]["pool"]
    pd.testing.assert_series_equal(
        got.set_index("gsis_id")["season_mean_fpts"].sort_index(),
        expected.set_index("gsis_id")["season_mean_fpts"].sort_index(),
        check_dtype=False,
    )
