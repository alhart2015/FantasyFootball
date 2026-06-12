"""CLI core for the H2H backtest harness. scripts/h2h_backtest.py wraps this.

Runs a full season backtest over three fixed strategies (now_or_never, season_value, bot)
and prints a per-strategy championship/playoff/win-pct/points-for table with 95% CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.tournament import Interval
from projections.draft.backtest.draft_basis import build_draft_basis
from projections.draft.backtest.harness import BacktestResult, run_backtest
from projections.draft.backtest.league import Calendar
from projections.draft.backtest.weekly_actuals import build_weekly_actuals
from projections.draft.league_config import LeagueConfig
from projections.schemas import ExternalProjectionSchema, WeeklyProjectionSchema
from projections.store import read_latest_partition, read_partition


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


def format_result(result: BacktestResult) -> str:
    """Render a per-strategy table showing champ%, playoff%, win%, points_for with CIs."""
    header = (
        f"H2H Backtest -- {result.n_seeds} seeds\n"
        f"{'STRATEGY':<16} {'CHAMP%':>14} {'PLAYOFF%':>20} {'WIN%':>16} {'PTS FOR':>24}"
    )
    rows: list[str] = [header]
    for name, m in sorted(result.by_strategy.items()):
        champ = _fmt_interval(m.championship, pct=True)
        playoff = _fmt_interval(m.playoff, pct=True)
        winp = _fmt_interval(m.win_pct, pct=True)
        pts = _fmt_interval(m.points_for, pct=False)
        rows.append(f"{name:<16} {champ:>20} {playoff:>26} {winp:>22} {pts:>28}")
    return "\n".join(rows)


def run(argv: list[str] | None = None) -> int:
    """Entry point for the H2H backtest CLI. Not unit-tested (requires 2025 data partitions)."""
    args = _parse_args(argv)
    data_root: Path = args.data_root
    config = LeagueConfig.model_validate_json(args.league_config.read_text())

    # Draft basis from the ingested external snapshot (Sleeper-ADP half-PPR fixed VORP).
    external = ExternalProjectionSchema.validate(
        read_latest_partition(data_root / "raw", "external_projections", season=args.season)
    )
    pool = build_draft_basis(external, league_config=config)

    # Weekly projections (start/sit) + actuals, pivoted to {(gsis_id, week): float}.
    proj_df = WeeklyProjectionSchema.validate(
        read_partition(data_root / "processed", "espn_weekly_projections", season=args.season)
    )
    weekly_stats = read_partition(data_root / "processed", "weekly_stats", season=args.season)
    actual_df = build_weekly_actuals(weekly_stats, ruleset=config.ruleset)

    proj_lookup = {
        (str(r.gsis_id), int(r.week)): float(r.projected_points)
        for r in proj_df.itertuples(index=False)
        if pd.notna(r.projected_points)
    }
    actual_lookup = {
        (str(r.gsis_id), int(r.week)): float(r.actual_points)
        for r in actual_df.itertuples(index=False)
    }

    availability = load_store_availability(pool, season=args.season, data_root=data_root)
    calendar = Calendar(
        regular_weeks=tuple(range(1, 15)),
        playoff_weeks=(15, 16, 17),
        playoff_size=6,
    )
    result = run_backtest(
        n_seeds=args.n_seeds,
        pool=pool,
        config=config,
        availability=availability,
        proj_lookup=proj_lookup,
        actual_lookup=actual_lookup,
        calendar=calendar,
        jitter=args.jitter,
        strategy_n_sims=args.strategy_n_sims,
    )
    print(format_result(result))
    return 0
