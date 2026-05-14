"""CLI branching tests for scripts/backtest.py.

These are unit tests of the dispatch/argument-resolution logic only — they
monkey-patch run_backtest so no real data or model training occurs.
"""

from __future__ import annotations

import sys
import unittest.mock as mock

import pytest
import scripts.backtest as backtest

from projections.schemas import Position


def test_backtest_cli_decomposed_baseline_restricts_to_wr_only() -> None:
    """When --model decomposed-baseline is selected, the CLI restricts
    positions to (Position.WR,) to avoid KeyError on other positions'
    factory dicts which lack the decomposed-baseline registration.
    """
    captured: dict[str, object] = {}

    def fake_run_backtest(**kwargs: object) -> object:
        captured.update(kwargs)
        raise SystemExit(0)  # short-circuit before _check/_update logic

    with mock.patch.object(backtest, "run_backtest", fake_run_backtest):
        with mock.patch.object(
            sys, "argv", ["backtest", "--report", "--model", "decomposed-baseline"]
        ):
            with pytest.raises(SystemExit) as ex_info:
                backtest.main()
            assert ex_info.value.code == 0

    assert captured.get("model_classes") == ("decomposed-baseline",)
    assert captured.get("positions") == (Position.WR,)


def test_backtest_cli_baseline_has_no_positions_restriction() -> None:
    """--model baseline must NOT pass a positions kwarg (positions=None path)
    so that all four positions are exercised by run_backtest's default."""
    captured: dict[str, object] = {}

    def fake_run_backtest(**kwargs: object) -> object:
        captured.update(kwargs)
        raise SystemExit(0)

    with mock.patch.object(backtest, "run_backtest", fake_run_backtest):
        with mock.patch.object(sys, "argv", ["backtest", "--report", "--model", "baseline"]):
            with pytest.raises(SystemExit) as ex_info:
                backtest.main()
            assert ex_info.value.code == 0

    assert captured.get("model_classes") == ("baseline",)
    assert "positions" not in captured  # positions kwarg must be absent


def test_backtest_cli_both_has_no_positions_restriction() -> None:
    """--model both (the default) must not pass a positions kwarg."""
    captured: dict[str, object] = {}

    def fake_run_backtest(**kwargs: object) -> object:
        captured.update(kwargs)
        raise SystemExit(0)

    with mock.patch.object(backtest, "run_backtest", fake_run_backtest):
        with mock.patch.object(sys, "argv", ["backtest", "--report", "--model", "both"]):
            with pytest.raises(SystemExit) as ex_info:
                backtest.main()
            assert ex_info.value.code == 0

    assert captured.get("model_classes") == ("baseline", "lightgbm")
    assert "positions" not in captured
