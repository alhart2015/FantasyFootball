"""What the in-season CLIs run against when the user types nothing.

Four tools — projected standings, waiver recommender, trade analyzer, dashboard — used to
demand `--league-id --season --team-id --pool` on every invocation. They now share
`add_league_arguments` / `resolve_league_target`, so the everyday run is bare.

That convenience is only safe if the fallback is unambiguous and loud: a tool that quietly
picks the wrong franchise still prints a complete, confident report about somebody else's
roster, and nothing downstream would flag it. These tests pin the resolution rule once, and
then pin that each of the four scripts is actually wired to it — a script that forgot the
call would still parse, still run, and still be wrong.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import projected_standings
import pytest
import run_season_dashboard
import trade_analyzer
import waiver_recommender

from projections.draft.assistant.league_profile import (
    LEAGUE_ARGUMENTS,
    resolve_league_target,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset

#: Every CLI that takes the shared flags, by the parser the test can drive. Adding a fifth
#: tool without adding it here is the gap this list exists to close.
PARSERS: dict[str, Callable[[list[str]], argparse.Namespace]] = {
    "projected_standings": projected_standings._parse_args,
    "waiver_recommender": waiver_recommender._parse_args,
    "trade_analyzer": trade_analyzer._parse_args,
    "run_season_dashboard": run_season_dashboard._parse_args,
}


def _write_league(tmp_path: Path) -> Path:
    league = LeagueConfig(
        name="Critts-shaped",
        n_teams=16,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 5,
        },
        ruleset=Ruleset.espn_half(),
    )
    path = tmp_path / "league_config.json"
    path.write_text(league.model_dump_json(indent=2))
    return path


def _write_profile(dir_: Path, body: dict[str, object]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "board_profile.json").write_text(json.dumps(body), encoding="utf-8")
    return dir_


def _full_profile(tmp_path: Path, dir_name: str = "critts_2025_2026") -> Path:
    """A league folder configured the way the user's real one is."""
    return _write_profile(
        tmp_path / dir_name,
        {
            "name": "Critts 2026",
            "league_config": str(_write_league(tmp_path)),
            "vorp_table": "data/vorp_2026/critts_half16_snake.parquet",
            "my_slot": 5,
            "season": 2026,
            "league_id": 856974,
            "team_id": 17,
        },
    )


def _bare(**overrides: object) -> argparse.Namespace:
    """The five shared flags, all unset unless a test sets one — what every parser produces
    for an empty command line."""
    base: dict[str, object] = dict.fromkeys(LEAGUE_ARGUMENTS)
    return argparse.Namespace(**{**base, **overrides})


# ---------------------------------------------------------------------------------------
# every CLI is wired to the shared flags
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PARSERS))
def test_every_cli_parses_an_empty_command_line(name: str) -> None:
    """The whole point: none of the five is `required=True` any more. A script that still
    demanded one would `SystemExit(2)` here."""
    args = PARSERS[name]([])

    assert all(getattr(args, flag) is None for flag in LEAGUE_ARGUMENTS)


@pytest.mark.parametrize("name", sorted(PARSERS))
def test_every_cli_resolves_the_same_league_from_the_same_profile(
    name: str, tmp_path: Path
) -> None:
    """One profile, four tools, one answer. Drift here would mean the standings and the
    trade analyzer disagreeing about which team is mine."""
    league_dir = _full_profile(tmp_path)

    args = PARSERS[name](["--league-dir", str(league_dir)])
    target = resolve_league_target(args)

    assert (target.league_id, target.season, target.team_id) == (856974, 2026, 17)
    assert target.pool == Path("data/vorp_2026/critts_half16_snake.parquet")
    # the folder the profile sits in — where rosters.tsv / schedule.tsv live — and NOT
    # the folder league_config.json happens to be in.
    assert target.league_dir == league_dir
    assert target.source == "Critts 2026"


# ---------------------------------------------------------------------------------------
# the resolution rule itself
# ---------------------------------------------------------------------------------------


def test_a_bare_run_finds_the_only_profile(tmp_path: Path) -> None:
    _full_profile(tmp_path)

    target = resolve_league_target(_bare(), root=tmp_path)

    assert (target.league_id, target.team_id) == (856974, 17)


def test_my_slot_is_not_borrowed_as_a_team_id(tmp_path: Path) -> None:
    """The two live side by side in one file and mean different things — a draft seat in
    1..16, and ESPN's arbitrary franchise id (5 and 17 here). Confusing them silently
    analyses the wrong roster."""
    league_dir = _full_profile(tmp_path)

    assert resolve_league_target(_bare(league_dir=league_dir)).team_id == 17


def test_typed_arguments_win_over_the_profile(tmp_path: Path) -> None:
    league_dir = _full_profile(tmp_path)

    target = resolve_league_target(
        _bare(league_dir=league_dir, team_id=8, pool=Path("other.parquet"), season=2025)
    )

    assert (target.team_id, target.season, target.pool) == (8, 2025, Path("other.parquet"))
    assert target.league_id == 856974  # the one field still coming from the file


def test_a_fully_typed_run_never_opens_a_profile(tmp_path: Path) -> None:
    """The explicit invocation must keep working in a checkout with no `data/leagues/` at
    all — otherwise the convenience would have made the old form conditional on it."""
    target = resolve_league_target(
        _bare(league_id=1, season=2026, team_id=2, pool=Path("p.parquet")),
        root=tmp_path / "does_not_exist",
    )

    assert target.source is None
    assert (target.league_id, target.season, target.team_id) == (1, 2026, 2)


def test_an_explicit_league_dir_without_a_profile_still_works_when_fully_typed(
    tmp_path: Path,
) -> None:
    """`waiver_recommender` accepted a bare league folder before this change. Requiring a
    `board_profile.json` there would have broken that invocation."""
    league_dir = tmp_path / "plain_folder"
    league_dir.mkdir()

    target = resolve_league_target(
        _bare(league_id=1, season=2026, pool=Path("p.parquet"), league_dir=league_dir)
    )

    assert target.require_league_config() == league_dir / "league_config.json"
    assert target.source is None


def test_team_id_stays_optional_for_the_tools_that_have_an_answer_without_one(
    tmp_path: Path,
) -> None:
    """Standings covers the whole league; the waiver tool lists the teams to pick from.
    Neither should fail just because the profile omits `team_id`."""
    league_dir = _write_profile(
        tmp_path / "board_only",
        {
            "league_config": str(_write_league(tmp_path)),
            "vorp_table": "pool.parquet",
            "my_slot": 5,
            "league_id": 856974,
        },
    )

    target = resolve_league_target(_bare(league_dir=league_dir))

    assert target.team_id is None
    with pytest.raises(ValueError, match="--team-id"):
        target.require_team()


def test_the_trade_analyzer_demands_a_team_and_names_the_file(tmp_path: Path) -> None:
    """It has no meaning without one, so it asks at resolve time — where the error can name
    the profile that should have carried it."""
    league_dir = _write_profile(
        tmp_path / "board_only",
        {
            "league_config": str(_write_league(tmp_path)),
            "vorp_table": "pool.parquet",
            "my_slot": 5,
            "league_id": 856974,
        },
    )

    with pytest.raises(ValueError, match=r"board_profile\.json has no team_id.*--team-id"):
        resolve_league_target(_bare(league_dir=league_dir), require_team_id=True)


def test_a_profile_without_a_league_id_names_the_missing_key(tmp_path: Path) -> None:
    """Profiles written for the draft board predate `league_id`; there is no safe guess."""
    league_dir = _write_profile(
        tmp_path / "board_only",
        {
            "league_config": str(_write_league(tmp_path)),
            "vorp_table": "pool.parquet",
            "my_slot": 5,
        },
    )

    with pytest.raises(ValueError, match=r"has no league_id.*--league-id"):
        resolve_league_target(_bare(league_dir=league_dir))


def test_a_typed_id_covers_the_key_the_profile_is_missing(tmp_path: Path) -> None:
    league_dir = _write_profile(
        tmp_path / "half_configured",
        {
            "league_config": str(_write_league(tmp_path)),
            "vorp_table": "pool.parquet",
            "my_slot": 5,
            "team_id": 17,
        },
    )

    target = resolve_league_target(_bare(league_dir=league_dir, league_id=856974))

    assert (target.league_id, target.team_id) == (856974, 17)


def test_two_leagues_are_not_guessed_between(tmp_path: Path) -> None:
    _full_profile(tmp_path, "one")
    _full_profile(tmp_path, "two")

    with pytest.raises(ValueError, match="--league-dir"):
        resolve_league_target(_bare(), root=tmp_path)


def test_describe_names_the_league_the_run_chose(tmp_path: Path) -> None:
    """The banner is the only thing standing between a wrong default and a report the user
    reads as their own."""
    target = resolve_league_target(_bare(league_dir=_full_profile(tmp_path)))

    line = target.describe()

    assert "Critts 2026" in line and "856974" in line and "team 17" in line


def test_the_five_flags_are_consumed_off_the_namespace(tmp_path: Path) -> None:
    """A script that resolved a target and then still read `args.season` would pass `None`
    into a schema and fail a long way from the mistake — that is how a real bug in this change
    surfaced, as a column of nulls ninety seconds into a simulation. Reading one afterwards
    must be an immediate AttributeError instead."""
    args = _bare(league_dir=_full_profile(tmp_path))

    resolve_league_target(args)

    for flag in LEAGUE_ARGUMENTS:
        assert not hasattr(args, flag), f"{flag} still readable off args"


# ---------------------------------------------------------------------------------------
# regressions found reviewing this change
# ---------------------------------------------------------------------------------------


def test_require_team_id_is_not_skipped_by_a_fully_typed_run(tmp_path: Path) -> None:
    """The first draft returned early once league_id/season/pool were all typed, BEFORE the
    require_team_id check — so the trade analyzer got `team_id=None` from a caller that had
    demanded one, and died on a traceback where argparse used to exit cleanly."""
    args = _bare(league_id=1, season=2026, pool=Path("p.parquet"))

    with pytest.raises(ValueError, match="team_id"):
        resolve_league_target(args, require_team_id=True, root=tmp_path)


def test_a_fully_typed_run_still_finds_its_league_directory_by_id(tmp_path: Path) -> None:
    """`run_season_dashboard` needs `league_dir` for rosters.tsv / schedule.tsv, and its own
    documented example does not pass one. Matching the typed `league_id` against configured
    profiles finds it — a keyed lookup, so it cannot land on another league's folder."""
    league_dir = _full_profile(tmp_path)

    target = resolve_league_target(
        _bare(league_id=856974, season=2026, team_id=17, pool=Path("p.parquet")), root=tmp_path
    )

    assert target.require_league_dir() == league_dir
    assert target.source is None  # nothing substantive came from the file


def test_a_typed_league_id_matching_no_profile_leaves_the_directory_unset(
    tmp_path: Path,
) -> None:
    """The safety half: an unrecognised league must NOT inherit the only profile's folder, or
    the dashboard would read a different league's rosters and say nothing."""
    _full_profile(tmp_path)

    target = resolve_league_target(
        _bare(league_id=999999, season=2026, pool=Path("p.parquet")), root=tmp_path
    )

    assert target.league_dir is None
    with pytest.raises(ValueError, match="--league-dir"):
        target.require_league_dir()


def test_a_league_dir_with_no_profile_is_a_clean_message_not_a_traceback(
    tmp_path: Path,
) -> None:
    """Every CLI catches ValueError and none catches OSError, so an unguarded read here
    surfaced as a traceback."""
    empty = tmp_path / "no_profile"
    empty.mkdir()

    with pytest.raises(ValueError, match=r"cannot read|no board_profile"):
        resolve_league_target(_bare(league_dir=empty))


def test_one_league_directory_resolves_to_one_league_config(tmp_path: Path) -> None:
    """`league_config_path` was derived two ways — `league_dir/league_config.json` on the
    explicit path, the profile's own `league_config` key on the other. A profile pointing at a
    relocated config made the same folder resolve to two different rulesets, silently."""
    league_dir = tmp_path / "critts"
    _write_profile(
        league_dir,
        {
            "league_config": str(_write_league(tmp_path)),  # NOT inside league_dir
            "vorp_table": "pool.parquet",
            "my_slot": 5,
            "league_id": 856974,
            "team_id": 17,
        },
    )

    bare = resolve_league_target(_bare(league_dir=league_dir))
    typed = resolve_league_target(
        _bare(league_dir=league_dir, league_id=856974, season=2026, pool=Path("pool.parquet"))
    )

    assert bare.require_league_config() == typed.require_league_config()
    assert bare.require_league_config() == tmp_path / "league_config.json"
