"""CLI engine for the auction bid-model tournament (spec §3.7). Mirrors tournament_cli.py.

`run([...])` loads the VORP pool + LeagueConfig, attaches is_rookie, loads availability +
variance params (store-backed), then races static/inflation/marginal and prints per-metric
means + CIs and paired diffs. No winner is printed (data-gathering, spec §5.1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.auction.bid_strategy import (
    AuctionBidStrategy,
    InflationBid,
    MarginalValueBid,
    StaticDollarBid,
)
from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.tournament import (
    METRICS,
    AuctionTournamentResult,
    run_auction_tournament,
)
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema

_MODELS: dict[str, AuctionBidStrategy] = {
    "static": StaticDollarBid(),
    "inflation": InflationBid(),
    "marginal": MarginalValueBid(),
}


def _load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def _load_config(path: Path) -> LeagueConfig:
    return LeagueConfig.model_validate_json(path.read_text())


def format_compare(result: AuctionTournamentResult) -> str:
    lines: list[str] = []
    lines.append(
        f"Auction bid-model data (seat {result.my_seat}, {result.n_seeds} seeds, "
        f"n_sims={result.n_sims}, price_jitter={result.price_jitter}, "
        f"budget={result.budget}, min_bid={result.min_bid})"
        " — data-gathering only; no decision declared."
    )
    header = f"{'model':<12}" + "".join(f"{m:>22}" for m in METRICS)
    lines.append(header)
    for name, metrics in result.summaries.items():
        cells = "".join(
            f"{iv.point:>10.2f} [{iv.lo_95:.1f},{iv.hi_95:.1f}]".rjust(22)
            for iv in (metrics[m] for m in METRICS)
        )
        lines.append(f"{name:<12}{cells}")
    lines.append("")
    lines.append("paired per-seed differences (point [95% CI]):")
    for pair, metrics in result.paired_diffs.items():
        lines.append(f"  {pair}")
        for m in METRICS:
            iv = metrics[m]
            lines.append(f"    {m:<20} {iv.point:+.3f} [{iv.lo_95:+.3f}, {iv.hi_95:+.3f}]")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auction bid-model data-gathering harness.")
    p.add_argument("--vorp-table", type=Path, required=True, help="Consensus VORP parquet.")
    p.add_argument(
        "--league-config", type=Path, required=True, help="LeagueConfig JSON (matches the table)."
    )
    p.add_argument("--my-seat", type=int, required=True, help="Hero seat (1-based).")
    p.add_argument(
        "--season", type=int, required=True, help="Season for availability/byes + is_rookie."
    )
    p.add_argument("--seeds", type=int, default=200, help="Paired auction sims per model.")
    p.add_argument(
        "--price-jitter",
        type=float,
        default=DEFAULT_PRICE_JITTER,
        help="Bot WTP noise (fractional).",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument("--n-sims", type=int, default=500, help="Monte-Carlo seasons per league (CRN).")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Store root for availability/rookies.",
    )
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("compare", help="Race static/inflation/marginal; record per-metric data.")
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pool = _load_pool(args.vorp_table)
    config = _load_config(args.league_config)
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)
    availability = load_store_availability(pool, season=args.season, data_root=args.data_root)
    params = VarianceParams.load()
    result = run_auction_tournament(
        _MODELS,
        pool,
        config,
        my_seat=args.my_seat,
        n_seeds=args.seeds,
        price_jitter=args.price_jitter,
        base_seed=args.seed,
        n_sims=args.n_sims,
        availability=availability,
        params=params,
    )
    print(format_compare(result))
    return 0
