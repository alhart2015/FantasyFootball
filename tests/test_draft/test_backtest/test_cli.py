"""Unit tests for the H2H backtest CLI (pure surfaces only — no real data required)."""

from __future__ import annotations

from projections.draft.assistant.tournament import Interval
from projections.draft.backtest.cli import _parse_args, format_result
from projections.draft.backtest.harness import BacktestResult, StrategyMetrics


def test_arg_defaults() -> None:
    args = _parse_args(["--season", "2025", "--league-config", "x.json"])
    assert args.n_seeds == 200
    assert args.strategy_n_sims == 200
    assert args.jitter == 8.0
    assert args.season == 2025


def test_format_lists_every_strategy() -> None:
    iv = Interval(point=0.1, lo_95=0.05, hi_95=0.15)
    m = StrategyMetrics(
        championship=iv,
        playoff=iv,
        win_pct=iv,
        points_for=Interval(point=1400.0, lo_95=1380.0, hi_95=1420.0),
    )
    table = {"now_or_never": m, "season_value": m, "bot": m}
    res = BacktestResult(by_strategy_actual=table, by_strategy_projected=table, n_seeds=200)
    text = format_result(res)
    assert "now_or_never" in text and "season_value" in text and "bot" in text
    assert "champ" in text.lower()
    # Both scorings are rendered.
    assert "PROJECTED" in text and "ACTUAL" in text
