"""CLI core for the H2H backtest harness. scripts/h2h_backtest.py wraps this.

Runs a full season backtest over three fixed strategies (now_or_never, season_value, bot)
and prints a per-strategy championship/playoff/win-pct/points-for table with 95% CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from projections.draft.assistant.tournament import Interval
from projections.draft.backtest.harness import BacktestResult, StrategyMetrics, run_backtest
from projections.draft.backtest.inputs import load_inputs
from projections.draft.league_config import LeagueConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H2H draft-strategy backtest.")
    p.add_argument("--season", type=int, default=2025, help="Target season (default 2025).")
    p.add_argument(
        "--league-config",
        type=Path,
        required=True,
        help="LeagueConfig JSON path.",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=200,
        help="Number of paired league seeds (default 200).",
    )
    p.add_argument(
        "--strategy-n-sims",
        type=int,
        default=200,
        help="Monte-Carlo seasons per roster for SeasonValueStrategy (default 200).",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=8.0,
        help="Bot ADP noise SD in picks (default 8.0).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Store root for partitioned parquet tables (default data/).",
    )
    return p.parse_args(argv)


def _fmt_interval(iv: Interval, *, pct: bool = False) -> str:
    """Format a point estimate + CI. `pct=True` multiplies by 100 and appends %."""
    if pct:
        return f"{iv.point * 100:6.1f}%  [{iv.lo_95 * 100:.1f}%, {iv.hi_95 * 100:.1f}%]"
    return f"{iv.point:8.1f}  [{iv.lo_95:.1f}, {iv.hi_95:.1f}]"


def _format_table(title: str, by_strategy: dict[str, StrategyMetrics]) -> str:
    """Render one metric's per-strategy table (champ%, playoff%, win%, points_for + CIs)."""
    rows: list[str] = [
        title,
        f"{'STRATEGY':<16} {'CHAMP%':>14} {'PLAYOFF%':>20} {'WIN%':>16} {'PTS FOR':>24}",
    ]
    for name, m in sorted(by_strategy.items()):
        champ = _fmt_interval(m.championship, pct=True)
        playoff = _fmt_interval(m.playoff, pct=True)
        winp = _fmt_interval(m.win_pct, pct=True)
        pts = _fmt_interval(m.points_for, pct=False)
        rows.append(f"{name:<16} {champ:>20} {playoff:>26} {winp:>22} {pts:>28}")
    return "\n".join(rows)


def format_result(result: BacktestResult) -> str:
    """Render both scorings: PROJECTED (who drafted better) and ACTUAL (real outcomes)."""
    header = f"H2H Backtest -- {result.n_seeds} seeds"
    projected = _format_table(
        "[PROJECTED points -- draft quality under shared projections]",
        result.by_strategy_projected,
    )
    actual = _format_table(
        "[ACTUAL points -- realized outcomes (also reflects projection error/luck)]",
        result.by_strategy_actual,
    )
    return f"{header}\n\n{projected}\n\n{actual}"


def run(argv: list[str] | None = None) -> int:
    """Entry point for the H2H backtest CLI. Not unit-tested (requires 2025 data partitions)."""
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    inputs = load_inputs(season=args.season, config=config, data_root=args.data_root)
    result = run_backtest(
        n_seeds=args.n_seeds,
        pool=inputs.pool,
        config=config,
        availability=inputs.availability,
        proj_lookup=inputs.proj_lookup,
        actual_lookup=inputs.actual_lookup,
        calendar=inputs.calendar,
        jitter=args.jitter,
        strategy_n_sims=args.strategy_n_sims,
    )
    print(format_result(result))
    return 0
