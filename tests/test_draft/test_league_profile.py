"""Unit tests for `board_profile.json` loading and discovery.

The profile exists so the live board comes up already configured for the user's league. Its
whole value is that nobody re-reads it on draft night — so every field it can get wrong is
worth pinning, and every way it can fail must be *loud* rather than a silent fallback to a
generic preset that looks perfectly fine and is not the user's league.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from projections.draft.assistant.league_profile import (
    DEFAULT_ID_MAP,
    DEFAULT_STRATEGY,
    LEAGUE_ARGUMENTS,
    discover_profiles,
    load_profile,
    resolve_league_target,
    resolve_profile,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _write_league(tmp_path: Path, *, n_teams: int = 16) -> Path:
    league = LeagueConfig(
        name="Critts-shaped",
        n_teams=n_teams,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
            RosterSlot.IR: 2,
        },
        ruleset=Ruleset.espn_half(),
    )
    path = tmp_path / "league_config.json"
    path.write_text(league.model_dump_json(indent=2))
    return path


def _write_profile(dir_: Path, body: dict[str, object]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / "board_profile.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_load_profile_reads_every_field(tmp_path: Path) -> None:
    league_path = _write_league(tmp_path)
    profile_path = _write_profile(
        tmp_path / "critts_2025_2026",
        {
            "name": "Critts 2026",
            "league_config": str(league_path),
            "vorp_table": "data/vorp_2026/critts_half16_snake.parquet",
            "id_map": "data/raw/id_map.parquet",
            "my_slot": 8,
            "season": 2026,
            "strategy": "raw_vorp",
        },
    )

    p = load_profile(profile_path)

    assert p.key == "critts_2025_2026"  # the directory, not the display name
    assert p.name == "Critts 2026"
    assert p.my_slot == 8
    assert p.season == 2026
    assert p.strategy == "raw_vorp"
    assert p.vorp_path == Path("data/vorp_2026/critts_half16_snake.parquet")
    # The config is loaded, not merely pointed at: the board draws roster eligibility from it.
    assert p.league.roster_slots[RosterSlot.WR] == 2
    assert p.league.roster_size == 12


def test_optional_fields_fall_back_to_board_defaults(tmp_path: Path) -> None:
    """`raw_vorp` is the tournament-winning strategy and the right default; the board's own
    dropdown order starts at `now_or_never`, which the tournament measured as worse."""
    league_path = _write_league(tmp_path)
    profile_path = _write_profile(
        tmp_path / "minimal",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 1},
    )

    p = load_profile(profile_path)

    assert p.strategy == DEFAULT_STRATEGY == "raw_vorp"
    assert p.id_map_path == DEFAULT_ID_MAP
    assert p.name == "minimal"  # falls back to the directory name


def test_a_slot_outside_the_league_is_rejected(tmp_path: Path) -> None:
    """This is the quiet one: a board on slot 17 of a 16-team league runs fine and puts
    every single one of the user's picks in the wrong place."""
    league_path = _write_league(tmp_path, n_teams=16)
    profile_path = _write_profile(
        tmp_path / "bad_slot",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 17},
    )

    with pytest.raises(ValueError, match=r"outside 1\.\.16"):
        load_profile(profile_path)


def test_slot_zero_is_rejected(tmp_path: Path) -> None:
    league_path = _write_league(tmp_path)
    profile_path = _write_profile(
        tmp_path / "zero_slot",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 0},
    )

    with pytest.raises(ValueError, match=r"outside 1\.\.16"):
        load_profile(profile_path)


def test_an_unknown_strategy_is_rejected(tmp_path: Path) -> None:
    league_path = _write_league(tmp_path)
    profile_path = _write_profile(
        tmp_path / "bad_strategy",
        {
            "league_config": str(league_path),
            "vorp_table": "pool.parquet",
            "my_slot": 1,
            "strategy": "raw_vorpp",
        },
    )

    with pytest.raises(ValueError, match="raw_vorpp"):
        load_profile(profile_path)


def test_a_missing_league_config_is_rejected(tmp_path: Path) -> None:
    profile_path = _write_profile(
        tmp_path / "dangling",
        {
            "league_config": str(tmp_path / "nope.json"),
            "vorp_table": "pool.parquet",
            "my_slot": 1,
        },
    )

    with pytest.raises(ValueError, match="does not exist"):
        load_profile(profile_path)


@pytest.mark.parametrize("missing", ["league_config", "vorp_table", "my_slot"])
def test_each_required_key_is_required(tmp_path: Path, missing: str) -> None:
    league_path = _write_league(tmp_path)
    body: dict[str, object] = {
        "league_config": str(league_path),
        "vorp_table": "pool.parquet",
        "my_slot": 1,
    }
    del body[missing]
    profile_path = _write_profile(tmp_path / "incomplete", body)

    with pytest.raises(ValueError, match=f"missing required key '{missing}'"):
        load_profile(profile_path)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    d = tmp_path / "broken"
    d.mkdir()
    (d / "board_profile.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_profile(d / "board_profile.json")


def test_discover_returns_profiles_sorted_and_skips_dirs_without_one(tmp_path: Path) -> None:
    league_path = _write_league(tmp_path)
    body: dict[str, object] = {
        "league_config": str(league_path),
        "vorp_table": "pool.parquet",
        "my_slot": 1,
    }
    _write_profile(tmp_path / "zeta", body)
    _write_profile(tmp_path / "alpha", body)
    (tmp_path / "no_profile_here").mkdir()

    profiles, errors = discover_profiles(tmp_path)

    assert [p.key for p in profiles] == ["alpha", "zeta"]
    assert errors == []


def test_discover_reports_a_broken_profile_instead_of_dropping_it(tmp_path: Path) -> None:
    """A skipped profile would silently fall the board back to a generic preset — the exact
    failure this module exists to prevent. The good one must still load."""
    league_path = _write_league(tmp_path)
    _write_profile(
        tmp_path / "good",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 1},
    )
    _write_profile(
        tmp_path / "bad",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 99},
    )

    profiles, errors = discover_profiles(tmp_path)

    assert [p.key for p in profiles] == ["good"]
    assert [e.path.parent.name for e in errors] == ["bad"]
    assert "outside 1..16" in errors[0].message


def test_discover_on_a_missing_root_is_empty_not_an_error(tmp_path: Path) -> None:
    """A fresh clone has no `data/leagues/`; the board must still start on presets."""
    profiles, errors = discover_profiles(tmp_path / "does_not_exist")

    assert profiles == [] and errors == []


def test_the_espn_ids_are_read_and_are_not_my_slot(tmp_path: Path) -> None:
    """`my_slot` is a draft seat in 1..n_teams; `team_id` is ESPN's franchise id and is
    unconstrained — 17 in a 16-team league is normal. The in-season tools key on the second,
    so conflating them would analyse the wrong roster with no error anywhere."""
    league_path = _write_league(tmp_path, n_teams=16)
    profile_path = _write_profile(
        tmp_path / "critts",
        {
            "league_config": str(league_path),
            "vorp_table": "pool.parquet",
            "my_slot": 5,
            "league_id": 856974,
            "team_id": 17,
        },
    )

    p = load_profile(profile_path)

    assert (p.league_id, p.team_id, p.my_slot) == (856974, 17, 5)


def test_the_espn_ids_are_optional(tmp_path: Path) -> None:
    """Every profile written before the in-season tools existed lacks them, and must still
    load — the draft board does not need either one."""
    league_path = _write_league(tmp_path)
    profile_path = _write_profile(
        tmp_path / "board_only",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 1},
    )

    p = load_profile(profile_path)

    assert p.league_id is None and p.team_id is None


@pytest.mark.parametrize("bad", [0, -1, "seventeen"])
def test_a_nonsense_espn_id_is_recorded_without_failing_the_load(
    tmp_path: Path, bad: object
) -> None:
    """Caught at load, but carried rather than raised. `league_id` / `team_id` are read by the
    in-season CLIs and by NOTHING the draft board touches, so raising here would make
    `discover_profiles` report a `ProfileError`, and the board drops every errored profile back
    to a generic preset — the wrong league on draft night, over a key it never reads."""
    league_path = _write_league(tmp_path)
    profile_path = _write_profile(
        tmp_path / "bad_team",
        {
            "league_config": str(league_path),
            "vorp_table": "pool.parquet",
            "my_slot": 1,
            "team_id": bad,
        },
    )

    p = load_profile(profile_path)

    assert p.team_id is None
    assert p.id_error is not None and "team_id" in p.id_error
    # Still a usable board profile: everything the board reads survived.
    assert p.my_slot == 1 and p.league.n_teams == 16


@pytest.mark.parametrize("bad", [0, -1, "seventeen"])
def test_a_nonsense_espn_id_still_fails_the_consumer_that_reads_it(
    tmp_path: Path, bad: object
) -> None:
    """The other half of the bargain: the CLIs that DO read these keys must refuse to run
    rather than fall back to a default franchise."""
    league_path = _write_league(tmp_path)
    _write_profile(
        tmp_path / "bad_team",
        {
            "league_config": str(league_path),
            "vorp_table": "pool.parquet",
            "my_slot": 1,
            "team_id": bad,
        },
    )
    args = argparse.Namespace(**dict.fromkeys(LEAGUE_ARGUMENTS))

    with pytest.raises(ValueError, match="team_id"):
        resolve_league_target(args, root=tmp_path)


def test_a_directory_with_no_profile_is_a_value_error_not_an_oserror(tmp_path: Path) -> None:
    """`--league-dir some/folder` reaches `load_profile` with a path that does not exist. Every
    CLI catches `ValueError` and none catches `OSError`, so a raw read here surfaced as a
    traceback where a one-line message was intended."""
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="cannot read"):
        load_profile(tmp_path / "empty" / "board_profile.json")


def test_resolve_picks_the_only_profile(tmp_path: Path) -> None:
    league_path = _write_league(tmp_path)
    _write_profile(
        tmp_path / "critts",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 1},
    )

    assert resolve_profile(root=tmp_path).key == "critts"


def test_resolve_names_an_explicit_directory(tmp_path: Path) -> None:
    league_path = _write_league(tmp_path)
    body: dict[str, object] = {
        "league_config": str(league_path),
        "vorp_table": "pool.parquet",
        "my_slot": 1,
    }
    _write_profile(tmp_path / "one", body)
    _write_profile(tmp_path / "two", body)

    assert resolve_profile(tmp_path / "two").key == "two"


def test_resolve_refuses_to_guess_between_two_leagues(tmp_path: Path) -> None:
    league_path = _write_league(tmp_path)
    body: dict[str, object] = {
        "league_config": str(league_path),
        "vorp_table": "pool.parquet",
        "my_slot": 1,
    }
    _write_profile(tmp_path / "one", body)
    _write_profile(tmp_path / "two", body)

    with pytest.raises(ValueError, match="--league-dir"):
        resolve_profile(root=tmp_path)


def test_resolve_refuses_when_a_sibling_profile_is_broken(tmp_path: Path) -> None:
    """Falling back to "the one that happened to parse" is exactly the silent wrong-league
    failure this module exists to prevent — the broken file may be the one meant."""
    league_path = _write_league(tmp_path)
    _write_profile(
        tmp_path / "good",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 1},
    )
    _write_profile(
        tmp_path / "bad",
        {"league_config": str(league_path), "vorp_table": "pool.parquet", "my_slot": 99},
    )

    with pytest.raises(ValueError, match="unreadable league profile"):
        resolve_profile(root=tmp_path)


def test_resolve_on_an_empty_root_says_to_pass_the_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pass the league arguments explicitly"):
        resolve_profile(root=tmp_path / "nothing_here")
