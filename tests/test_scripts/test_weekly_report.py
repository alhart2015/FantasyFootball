"""The combined weekly report.

The properties worth pinning are about composition, not about any section's numbers — those
are each tool's own tests. What can only break here: is the league fetched once, does one
section's failure take the document down, and do `--only`/`--skip` mean what they say.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest
import weekly_report

from projections.midseason.context import InSeasonContext


class _Ctx:
    """A stand-in. The sections are stubbed, so only what `main` itself reads is here."""

    notes: tuple[str, ...] = ()


def _ctx() -> Any:
    return _Ctx()


def _args(**overrides: Any) -> argparse.Namespace:
    base = {"only": None, "skip": None}
    return argparse.Namespace(**{**base, **overrides})


def test_every_section_runs_in_deadline_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lineup first: it locks at kickoff. Trades last: they keep."""
    assert [name for name, _, _ in weekly_report.SECTIONS] == [
        "start-sit",
        "waivers",
        "standings",
        "trades",
    ]


def test_the_league_is_assembled_once_for_the_whole_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entire point. Four sections used to mean four identical league fetches."""
    calls: list[str] = []
    ctx = _ctx()

    monkeypatch.setattr(weekly_report, "resolve_league_target", lambda args, **k: _Target())

    def _assemble(target: Any, args: Any, **k: Any) -> Any:
        calls.append("assemble")
        return ctx

    monkeypatch.setattr(weekly_report, "assemble_context", _assemble)
    monkeypatch.setattr(
        weekly_report,
        "SECTIONS",
        tuple((name, name.upper(), _spy(name, calls)) for name in ("a", "b", "c")),
    )
    monkeypatch.setattr(weekly_report, "_NAMES", ("a", "b", "c"))
    monkeypatch.setattr(weekly_report, "_parse_args", lambda argv: _args())

    assert weekly_report.main([]) == 0
    assert calls == ["assemble", "a", "b", "c"]


def test_a_failing_section_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale `external_projections` breaks trades and says nothing about your lineup. A
    report that dies at section three has told you nothing at all."""
    calls: list[str] = []

    def _boom(ctx: Any, args: Any) -> int:
        calls.append("boom")
        raise ValueError("external projections are stale")

    monkeypatch.setattr(weekly_report, "resolve_league_target", lambda args, **k: _Target())
    monkeypatch.setattr(weekly_report, "assemble_context", lambda target, args, **k: _ctx())
    monkeypatch.setattr(
        weekly_report,
        "SECTIONS",
        (
            ("a", "A", _spy("a", calls)),
            ("b", "B", _boom),
            ("c", "C", _spy("c", calls)),
        ),
    )
    monkeypatch.setattr(weekly_report, "_NAMES", ("a", "b", "c"))
    monkeypatch.setattr(weekly_report, "_parse_args", lambda argv: _args())

    # non-zero, because the document is incomplete -- but every other section still ran
    assert weekly_report.main([]) == 1
    assert calls == ["a", "boom", "c"]
    err = capsys.readouterr().err
    assert "could not run" in err and "external projections are stale" in err


def test_only_and_skip_select_sections() -> None:
    assert weekly_report._selected(_args(only="start-sit,trades")) == ["start-sit", "trades"]
    assert weekly_report._selected(_args(skip="trades")) == [
        "start-sit",
        "waivers",
        "standings",
    ]


def test_an_unknown_section_name_is_an_error_not_a_silent_no_op(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--skip trade` (singular) must not quietly produce the slow report it was meant to
    avoid. Silently ignoring the flag is the worst possible reading of it."""
    assert weekly_report._selected(_args(skip="trade")) is None
    err = capsys.readouterr().err
    assert "unknown section" in err
    assert "trades" in err, "it names the valid options"


def test_section_defaults_are_taken_from_each_tool_rather_than_listed_by_hand() -> None:
    """The first cut hand-wrote the union and missed `--max-players`, so the trades section
    crashed on the first real run. Each tool's parser is the only thing that knows what that
    tool reads."""
    args = weekly_report._parse_args([])
    for flag in ("max_players", "espn_tolerance", "min_gain", "top", "weight_espn"):
        assert hasattr(args, flag), f"{flag} is read by a section and must be present"


def test_the_report_declares_its_own_flags_over_a_section_default() -> None:
    """`--n-sims` and `--seed` are the report's to set; a section default must not win."""
    args = weekly_report._parse_args(["--n-sims", "17", "--seed", "9"])
    assert args.n_sims == 17
    assert args.seed == 9


def _spy(name: str, calls: list[str]) -> Any:
    def _run(ctx: InSeasonContext, args: argparse.Namespace) -> int:
        calls.append(name)
        return 0

    return _run


class _Target:
    source = None
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        return "test league"
