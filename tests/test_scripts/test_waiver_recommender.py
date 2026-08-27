"""The waiver recommender CLI.

Every other script in the repo has one of these, and this one holds the printing rules — which
are where the tool's honesty lives. Three of pass 1's findings were the CLI claiming something
it had not done: an adjustment on a healthy player, "roster spot open" on a full roster, and a
blank line where "we could not simulate him" belonged.

No network. The printers take objects; `test_a_missing_team_id_is_not_guessed_at` drives `run`
itself with the league fetch patched, and stops at the team-id branch -- so the fetch surface is
proven patchable, but the parquet reads, the free-agent calls and the injury-note calls are NOT
exercised here. `tests/test_midseason/` covers the logic behind them.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from projections.ingest.espn_league import EspnCredentials
from projections.ingest.injury_news import InjuryNote
from projections.midseason.swap_impact import SwapImpact
from projections.midseason.waivers import Candidate
from projections.schemas import InjuryStatus

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "waiver_recommender.py"


def _module() -> Any:
    """Import the script by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("waiver_recommender", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(**overrides: Any) -> Candidate:
    fields: dict[str, Any] = {
        "player_id": 1,
        "player": "Wire Stud",
        "position": "WR",
        "nfl_team": "SF",
        "lineup_gain": 4.2,
        "projected": 14.0,
        "injury_status": InjuryStatus.ACTIVE,
        "on_waivers": False,
        "percent_owned": 12.0,
        "needs_no_drop": False,
        "drop_player_id": 7,
        "drop_player": "Bench RB",
        "drop_cost": 21.0,
    }
    fields.update(overrides)
    return Candidate(**fields)


def _impact(candidate: Candidate, **overrides: Any) -> SwapImpact:
    fields: dict[str, Any] = {
        "candidate": candidate,
        "delta_wins": 0.21,
        "delta_playoff_pct": 0.04,
        "delta_title_pct": 0.01,
        "simulated": True,
    }
    fields.update(overrides)
    return SwapImpact(**fields)


# --- the three states of "who do I drop" --------------------------------------------------------


def test_a_needed_drop_is_named_with_its_cost(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    candidate = _candidate()
    module._print_candidate(candidate, None, _impact(candidate))
    out = capsys.readouterr().out
    assert "drop Bench RB" in out
    assert "costs 21 rest-of-season points" in out


def test_no_drop_needed_and_no_drop_found_do_not_print_the_same(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Opposite facts. `is_free` reported both as "roster spot open", which is a lie in one of
    them -- and it is the one the module calls the first thing worth telling a reader."""
    module = _module()
    free = _candidate(needs_no_drop=True, drop_player_id=None, drop_player="", drop_cost=0.0)
    module._print_candidate(free, None, _impact(free))
    free_out = capsys.readouterr().out

    stuck = _candidate(needs_no_drop=False, drop_player_id=None, drop_player="", drop_cost=0.0)
    module._print_candidate(stuck, None, _impact(stuck))
    stuck_out = capsys.readouterr().out

    assert "no drop needed" in free_out
    assert "NO DROP FOUND" in stuck_out
    assert "costs" not in stuck_out, "there is no drop, so there is no cost to quote"


# --- saying only what was actually done -----------------------------------------------------------


def test_a_healthy_free_agent_is_not_claimed_to_have_been_adjusted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FREE_AGENT is the expected status for much of the wire and carries a multiplier of 1.0.
    Keying the notice off `is not ACTIVE` announced a discount that was never applied."""
    module = _module()
    for status in (
        InjuryStatus.ACTIVE,
        InjuryStatus.NORMAL,
        InjuryStatus.DAY_TO_DAY,
        InjuryStatus.FREE_AGENT,
        InjuryStatus.UNKNOWN,
    ):
        candidate = _candidate(injury_status=status)
        module._print_candidate(candidate, None, _impact(candidate))
        assert "adjusted for this" not in capsys.readouterr().out, status


def test_an_actually_adjusted_player_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    candidate = _candidate(injury_status=InjuryStatus.QUESTIONABLE)
    module._print_candidate(candidate, None, _impact(candidate))
    assert "QUESTIONABLE — adjusted for this" in capsys.readouterr().out


def test_an_unsimulated_candidate_is_not_printed_as_a_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Silence read as "simulated, and it came to nothing" on the row where the answer is most
    uncertain."""
    module = _module()
    candidate = _candidate()
    module._print_candidate(
        candidate,
        None,
        _impact(
            candidate,
            simulated=False,
            delta_wins=0.0,
            not_simulated_because="no season projection for him",
        ),
    )
    out = capsys.readouterr().out
    assert "NOT SIMULATED" in out
    assert "no season projection for him" in out, "it says WHY, which is the actionable part"
    assert "+0.00 wins" not in out


def test_an_unsimulated_impact_must_carry_a_reason() -> None:
    """The CLI interpolates it into a sentence, so an empty one renders "NOT SIMULATED — ."
    The dataclass refuses rather than letting a caller construct that."""
    with pytest.raises(ValueError, match="must say why"):
        SwapImpact(
            candidate=_candidate(),
            delta_wins=0.0,
            delta_playoff_pct=0.0,
            delta_title_pct=0.0,
            simulated=False,
        )


def test_a_delta_inside_the_noise_is_marked(capsys: pytest.CaptureFixture[str]) -> None:
    """A tool printing 0.03 wins to two decimals when its own measured spread is 0.062 is
    inventing precision."""
    module = _module()
    candidate = _candidate()
    module._print_candidate(candidate, None, _impact(candidate, delta_wins=0.03))
    out = capsys.readouterr().out
    assert "inside noise" in out
    # The row names the floor it was judged against, because a swap and a free add are held to
    # different ones -- a bare "inside noise" on a 0.09 free add contradicts a footer quoting
    # the paired floor. Three decimals, because two rounds a delta of -0.0615 and its floor of
    # 0.062 to the same "0.06" and the row then reads as though 0.06 were inside 0.06.
    assert "floor 0.062" in out

    module._print_candidate(candidate, None, _impact(candidate, delta_wins=0.30))
    assert "inside noise" not in capsys.readouterr().out


def test_a_free_add_is_judged_against_the_wider_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """0.09 wins is signal for a swap and noise for a free add, and the row says which."""
    module = _module()
    free = _candidate(needs_no_drop=True, drop_player_id=None, drop_player="", drop_cost=0.0)
    module._print_candidate(free, None, _impact(free, delta_wins=0.09))
    out = capsys.readouterr().out
    assert "inside noise" in out and "floor 0.127" in out

    swap = _candidate()
    module._print_candidate(swap, None, _impact(swap, delta_wins=0.09))
    assert "inside noise" not in capsys.readouterr().out


# --- the write-up ---------------------------------------------------------------------------------


def test_the_long_write_up_prints_only_for_a_multi_week_absence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The short line is always useful; the long one is where a timeline appears, and a timeline
    only matters when the numeric guess is the multi-week one."""
    module = _module()
    note = InjuryNote(
        player_id=1,
        status=InjuryStatus.INJURY_RESERVE,
        injury_type="Hamstring",
        detail="Strain",
        reported="2026-10-14T00:00:00Z",
        short_comment="Hall (hamstring) was placed on IR Tuesday.",
        long_comment="The team is hopeful for a Week 12 return.",
    )
    module._print_note(note)
    out = capsys.readouterr().out
    assert "Hamstring · Strain · 2026-10-14" in out
    assert "placed on IR Tuesday" in out
    assert "Week 12 return" in out

    module._print_note(InjuryNote(**{**note.__dict__, "status": InjuryStatus.QUESTIONABLE}))
    short_out = capsys.readouterr().out
    assert "placed on IR Tuesday" in short_out
    assert "Week 12 return" not in short_out, "one week's absence needs no timeline"


def test_no_note_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    _module()._print_note(None)
    assert capsys.readouterr().out == ""


# --- the argument surface ---------------------------------------------------------------------


#: The flags every invocation needs, so a test can vary only the one it is about.
_MINIMAL = ["--league-id", "1", "--season", "2026", "--league-dir", ".", "--pool", "p"]


def _parse(module: Any, argv: list[str]) -> argparse.Namespace:
    """The parsed arguments, without running the tool."""
    captured: dict[str, argparse.Namespace] = {}

    def fake_run(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    original = module.run
    module.run = fake_run
    try:
        module.main(argv)
    finally:
        module.run = original
    return captured["args"]


def test_simulation_is_on_by_default() -> None:
    """Δ wins is the objective, so the default run has to contain one. An opt-in flag made the
    default a points ranking with no wins number in it at all."""
    assert _parse(_module(), _MINIMAL).fast is False


def test_fast_is_available_for_when_seconds_matter() -> None:
    assert _parse(_module(), [*_MINIMAL, "--fast"]).fast is True


def test_a_missing_team_id_is_not_guessed_at(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run` lists the league's teams and exits 2 rather than picking one. Choosing a team for
    the user is the one thing this tool must never do.

    Drives `run` itself -- asserting on the PARSED ARGUMENTS, as an earlier version did, cannot
    fail if `run` starts guessing.
    """
    module = _module()
    monkeypatch.setattr(
        module.EspnCredentials, "resolve", classmethod(lambda cls, path=None: _CREDS)
    )
    calls: list[str] = []

    def spy(*args: object, **kwargs: object) -> dict[str, Any]:
        calls.append("league")
        return _LEAGUE_PAYLOAD

    monkeypatch.setattr(module, "fetch_league_payload", spy)
    args = argparse.Namespace(**{**vars(_parse(module, _MINIMAL)), "team_id": None})
    assert module.run(args) == 2
    out = capsys.readouterr().out
    assert "--team-id is required" in out
    assert "Silence of the Lamb" in out, "it lists what you could pass"
    assert calls == ["league"], "and it got there through the patched name, not the network"


#: A league payload with two teams and nothing else -- enough for `parse_teams`.
_LEAGUE_PAYLOAD: dict[str, Any] = {
    "teams": [
        {"id": 17, "name": "Silence of the Lamb", "owners": []},
        {"id": 3, "name": "HTTRedhogs", "owners": []},
    ],
    "members": [],
}

_CREDS = EspnCredentials(swid="{X}", espn_s2="s2")
