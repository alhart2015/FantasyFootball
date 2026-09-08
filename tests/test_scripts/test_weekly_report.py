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
        tuple((name, name.upper(), _FakeModule(name, calls)) for name in ("a", "b", "c")),
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

    monkeypatch.setattr(weekly_report, "resolve_league_target", lambda args, **k: _Target())
    monkeypatch.setattr(weekly_report, "assemble_context", lambda target, args, **k: _ctx())
    monkeypatch.setattr(
        weekly_report,
        "SECTIONS",
        (
            ("a", "A", _FakeModule("a", calls)),
            ("b", "B", _FakeModule("b", calls, raises="external projections are stale")),
            ("c", "C", _FakeModule("c", calls)),
        ),
    )
    monkeypatch.setattr(weekly_report, "_NAMES", ("a", "b", "c"))
    monkeypatch.setattr(weekly_report, "_parse_args", lambda argv: _args())

    # non-zero, because the document is incomplete -- but every other section still ran
    assert weekly_report.main([]) == 1
    assert calls == ["a", "b", "c"]
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


def test_each_section_gets_its_own_defaults_not_a_merged_namespace() -> None:
    """The collisions a single merged Namespace got wrong, resolved by module import order.

    `--top` is 5 for waivers and 8 for trades; `--n-sims` is 20,000 for start/sit's P(right)
    and 2,000 everywhere else. Merging silently ran trades at 5 proposals and computed
    P(right) from a tenth of the draws the standalone tool uses -- both contradicting this
    report's claim that a section behaves here exactly as it does alone.
    """
    import start_sit
    import trade_analyzer
    import waiver_recommender

    args = weekly_report._parse_args([])
    assert weekly_report.section_args(start_sit, args).n_sims == 20_000
    assert weekly_report.section_args(trade_analyzer, args).n_sims == 2_000
    assert weekly_report.section_args(trade_analyzer, args).top == 8
    assert weekly_report.section_args(waiver_recommender, args).top == 5


def test_a_flag_the_report_owns_overrides_the_section_default() -> None:
    """`--week`, `--fast` and `--seed` are the reader's instruction to the whole report."""
    import start_sit

    args = weekly_report._parse_args(["--seed", "9", "--week", "5", "--fast"])
    section = weekly_report.section_args(start_sit, args)
    assert section.seed == 9
    assert section.week == 5
    assert section.fast is True
    # ...and a flag it merely shares a name with is untouched
    assert section.n_sims == 20_000


def test_every_section_carries_its_own_module() -> None:
    """One tuple, so a section cannot be half-registered: the module supplies both the
    `report` to call and the `_parse_args` that owns its defaults."""
    for name, heading, module in weekly_report.SECTIONS:
        assert callable(module.report), name
        assert callable(module._parse_args), name
        assert heading


class _FakeModule:
    """Stands in for a section module: a `report` to call and a parser owning its defaults."""

    def __init__(self, name: str, calls: list[str], raises: str | None = None) -> None:
        self._name, self._calls, self._raises = name, calls, raises

    def report(self, ctx: InSeasonContext, args: argparse.Namespace) -> int:
        self._calls.append(self._name)
        if self._raises:
            raise ValueError(self._raises)
        return 0

    def _parse_args(self, argv: list[str]) -> argparse.Namespace:
        return argparse.Namespace()


class _Target:
    source = None
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        return "test league"
