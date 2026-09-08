"""The shared in-season context.

Every assertion here is about something that was previously duplicated across the four CLIs
and could therefore drift: the week, the config source, the pool normalisation, the size of
the rostered-player request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import EspnCredentials
from projections.midseason import context as ctx_mod
from projections.midseason.context import InSeasonContext, build_context
from projections.schemas import RosterSlot


def _payload() -> dict[str, Any]:
    entries = [
        {
            "playerPoolEntry": {
                "player": {
                    "id": pid,
                    "fullName": f"P{pid}",
                    "defaultPositionId": 2,
                    "injuryStatus": "ACTIVE",
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


@pytest.fixture
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A whole league on disk, with the network and the history scans stubbed."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "gsis_id": ["00-0000101", "00-0000102"],
            "espn_id": ["101", "102"],
            "sleeper_id": ["s101", "s102"],
        }
    ).to_parquet(raw / "id_map.parquet")

    pool_path = tmp_path / "pool.parquet"
    pd.DataFrame(
        {
            # object dtype on purpose: the context must normalise it to _PYARROW_STR, which
            # the trade analyzer used to skip.
            "gsis_id": pd.Series(["00-0000101", "00-0000102"], dtype=object),
            "position": [RosterSlot.RB.value, RosterSlot.RB.value],
            "season_mean_fpts": [200.0, 180.0],
            "vorp": [50.0, 30.0],
            "replacement_fpts": [150.0, 150.0],
        }
    ).to_parquet(pool_path)

    league_dir = tmp_path / "league"
    league_dir.mkdir()
    config = {
        "name": "Test League",
        "n_teams": 2,
        "roster_slots": {"RB": 1, "BENCH": 3},
        "ruleset": "espn_half",
    }
    (league_dir / "league_config.json").write_text(json.dumps(config), encoding="utf-8")

    payload = _payload()
    calls: dict[str, int] = {"payload": 0, "availability": 0, "onteam": 0}

    def _fetch_payload(*a: Any, **k: Any) -> dict[str, Any]:
        calls["payload"] += 1
        return payload

    def _availability(*a: Any, **k: Any) -> object:
        calls["availability"] += 1
        return object()

    def _onteam(*a: Any, **k: Any) -> dict[str, Any]:
        calls["onteam"] += 1
        return {"players": [], "_args": k}

    monkeypatch.setattr(ctx_mod, "fetch_league_payload", _fetch_payload)
    monkeypatch.setattr(ctx_mod, "load_store_availability", _availability)
    monkeypatch.setattr(ctx_mod, "fetch_free_agents", _onteam)
    monkeypatch.setattr(ctx_mod, "attach_is_rookie", lambda pool, **k: pool.assign(is_rookie=False))
    monkeypatch.setattr(EspnCredentials, "resolve", classmethod(lambda cls, path: object()))

    args = argparse.Namespace(
        league_id=1,
        season=2026,
        team_id=1,
        pool=pool_path,
        league_dir=league_dir,
        data_root=tmp_path,
        credentials=tmp_path / "creds.json",
        week=None,
    )
    return {"args": args, "calls": calls, "payload": payload, "tmp": tmp_path}


def _build(env: dict[str, Any], **overrides: Any) -> InSeasonContext:
    for key, value in overrides.items():
        setattr(env["args"], key, value)
    return build_context(env["args"])


def test_the_pool_is_normalised_once_and_for_everyone(_env: dict[str, Any]) -> None:
    """Schema validation + is_rookie, from one code path. The trade analyzer skipped both.

    Deliberately does NOT assert a pyarrow `gsis_id`: `VorpTableSchema` declares
    `Series[str]`, which is object dtype in pandera, so validation coerces the `astype` in
    `load_pool` straight back. That line has been a no-op in every caller that had it; the
    dtype is pinned below so the day the schema changes is a visible one.
    """
    ctx = _build(_env)
    assert "is_rookie" in ctx.pool.columns
    assert ctx.pool["gsis_id"].is_unique  # the schema's uniqueness rule really ran
    assert ctx.pool["gsis_id"].dtype == object, (
        "VorpTableSchema coerces gsis_id to object; if this fails the schema was fixed and "
        "load_pool's astype is no longer dead"
    )


def test_missing_weekly_stats_is_an_empty_frame_not_an_error(_env: dict[str, Any]) -> None:
    """No refresh yet means no points scored yet, which is the truth, not a failure."""
    ctx = _build(_env)
    assert ctx.weekly_stats.empty


def test_the_week_comes_from_the_schedule(_env: dict[str, Any]) -> None:
    ctx = _build(_env)
    assert ctx.week == ctx.my_team.week == 1


def test_an_explicit_week_moves_every_consumer(_env: dict[str, Any]) -> None:
    """A real behaviour change, and the reason it is worth making.

    Today `--week` shifts the waiver and start/sit horizons and leaves standings and trades on
    the real week, so one flag means two things depending on which tool reads it.
    """
    ctx = _build(_env, week=7)
    assert ctx.week == 7
    assert ctx.my_team.week == 1  # the schedule still says what it says


def test_the_config_comes_from_the_file(_env: dict[str, Any]) -> None:
    ctx = _build(_env)
    assert ctx.config.roster_slots == {RosterSlot.RB: 1, RosterSlot.BENCH: 3}


def test_a_config_that_disagrees_with_espn_is_reported_not_silently_preferred(
    _env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured identical on the live league; this exists for the day that stops being true.

    `roster_slots` sizes the rostered-player request, so a silent drift would thin the
    projections behind every number in every section.
    """
    espn_says = LeagueConfig.model_validate(
        {
            "name": "Test League",
            "n_teams": 3,
            "roster_slots": {"RB": 2, "BENCH": 3},
            "ruleset": "espn_ppr",
        }
    )
    monkeypatch.setattr(ctx_mod, "build_league_config", lambda payload, name: espn_says)
    ctx = _build(_env)

    assert any("roster slots" in note for note in ctx.notes), ctx.notes
    assert any("teams" in note for note in ctx.notes), ctx.notes
    assert any("scoring" in note for note in ctx.notes), ctx.notes
    # and the FILE still wins, including for the request size
    assert ctx.config.roster_slots == {RosterSlot.RB: 1, RosterSlot.BENCH: 3}
    assert ctx.rostered_limit() == 2 * (1 + 3 + 2)


def test_a_config_that_cannot_be_derived_is_a_note_not_a_crash(_env: dict[str, Any]) -> None:
    """The live path for this fixture, and the right shape generally: a cross-check that
    cannot run must not take the whole report down with it."""
    ctx = _build(_env)
    assert any("cross-check" in note for note in ctx.notes), ctx.notes
    assert ctx.config.roster_slots == {RosterSlot.RB: 1, RosterSlot.BENCH: 3}


def test_the_payload_is_fetched_exactly_once(_env: dict[str, Any]) -> None:
    """The point of the whole exercise: four sections, one league fetch."""
    ctx = _build(_env)
    ctx.roster()
    ctx.teams()
    assert _env["calls"]["payload"] == 1


def test_availability_is_computed_once_and_reused(_env: dict[str, Any]) -> None:
    """It rescans 2018..season-1. Four sections asking separately is eight full scans."""
    ctx = _build(_env)
    first, second = ctx.availability(), ctx.availability()
    assert first is second
    assert _env["calls"]["availability"] == 1


def test_the_onteam_payload_is_fetched_once_at_the_context_week(_env: dict[str, Any]) -> None:
    """Shared between the waiver tool and start/sit, and only sound because they agree on the
    week — which they now do by construction rather than by luck."""
    ctx = _build(_env, week=5)
    first, second = ctx.onteam_payload(), ctx.onteam_payload()
    assert first is second
    assert _env["calls"]["onteam"] == 1
    assert first["_args"]["scoring_period"] == 5
    assert first["_args"]["statuses"] == ("ONTEAM",)


def test_the_rostered_limit_is_league_sized(_env: dict[str, Any]) -> None:
    """Sharing a free-agent-sized limit once dropped the waiver tool's own starters from the
    projections and inflated every candidate's gain."""
    ctx = _build(_env)
    assert ctx.rostered_limit() == 2 * (1 + 3 + 2)
    assert ctx.onteam_payload()["_args"]["limit"] == ctx.rostered_limit()


def test_the_roster_is_mine_and_is_a_copy(_env: dict[str, Any]) -> None:
    """Four sections share one context. A frame handed out by reference is a `df[...] = ...`
    away from one section corrupting another's inputs."""
    ctx = _build(_env)
    roster = ctx.roster()
    assert set(roster["team_id"]) == {1}

    roster["player"] = "MUTATED"
    assert (ctx.roster()["player"] != "MUTATED").all()


def test_a_config_matching_espn_produces_no_config_note(
    _env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quiet path. A note on every run trains the reader to ignore notes."""
    file_config = LeagueConfig.model_validate(
        {
            "name": "Test League",
            "n_teams": 2,
            "roster_slots": {"RB": 1, "BENCH": 3},
            "ruleset": "espn_half",
        }
    )
    monkeypatch.setattr(ctx_mod, "build_league_config", lambda payload, name: file_config)
    ctx = _build(_env)
    assert not [note for note in ctx.notes if "league_config.json" in note]
