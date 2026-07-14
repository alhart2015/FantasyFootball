"""Cap-fix tuning: race BalancedValueBid(non_increasing_cap=True) across a pace x premium grid
against both bot markets on the 12-team half-PPR preset, and pick the best worst-case reg_win_pct.

Crash-safe: one (market, seed-chunk) per process (the dev box's Raptor Lake fault wants bounded
processes — memory h2h-backtest-native-crash). `run` writes a per-chunk JSON; `aggregate` combines
the chunk JSONs and prints the reg_win_pct table + finalist. Data-gathering; no default changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from projections.draft.assistant.auction.bid_strategy import (
    AuctionBidStrategy,
    BalancedValueBid,
    PatientValueBid,
)
from projections.draft.assistant.auction.tournament import run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_config,
    _load_pool,
)
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie

PACES: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
PREMIUMS: tuple[float, ...] = (0.5, 1.0, 1.5)


def grid() -> dict[str, AuctionBidStrategy]:
    """The pace x premium flat-cap grid + the inflating control + the standing breadth leader."""
    models: dict[str, AuctionBidStrategy] = {}
    for pace in PACES:
        for prem in PREMIUMS:
            models[f"flat_p{pace}_prem{prem}"] = BalancedValueBid(
                premium=prem, pace=pace, non_increasing_cap=True
            )
    models["balanced"] = BalancedValueBid()  # inflating-cap control (default False)
    models["patient_deep"] = PatientValueBid(scrub_frac=0.0)  # standing multi-year leader reference
    return models


def aggregate_chunks(
    chunks: list[dict[str, object]],
) -> tuple[list[str], list[tuple[str, list[float], float]], str]:
    """Combine per-chunk reg_win_pct into (markets, rows, best). Equal chunk sizes -> mean of
    chunk means == overall mean. Row = (name, per-market means, worst-case); sorted worst desc."""
    by_market: dict[str, dict[str, list[float]]] = {}
    for c in chunks:
        m = str(c["market"])
        rwp = c["reg_win_pct"]
        if not isinstance(rwp, dict):
            raise ValueError(f"chunk reg_win_pct must be a dict; got {type(rwp)}")
        for name, val in rwp.items():
            by_market.setdefault(m, {}).setdefault(str(name), []).append(float(val))
    markets = sorted(by_market)
    names = sorted({n for m in by_market for n in by_market[m]})
    rows: list[tuple[str, list[float], float]] = []
    for name in names:
        cells = [float(np.mean(by_market[m][name])) for m in markets if name in by_market[m]]
        rows.append((name, cells, min(cells)))
    rows.sort(key=lambda r: r[2], reverse=True)
    best = rows[0][0] if rows else ""
    return markets, rows, best


def _run_chunk(args: argparse.Namespace) -> int:
    pool = _load_pool(args.vorp_table)
    config = _load_config(args.league_config)
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)
    availability = load_store_availability(pool, season=args.season, data_root=args.data_root)
    params = VarianceParams.load()
    market: Literal["espn", "model"] = "espn" if args.bot_prices == "espn" else "model"
    result = run_auction_tournament(
        grid(),
        pool,
        config,
        my_seat=args.my_seat,
        n_seeds=args.seeds,
        price_jitter=0.15,
        base_seed=args.seed,
        n_sims=args.n_sims,
        availability=availability,
        params=params,
        nomination_temp=1.0,
        bot_archetypes=_REALISTIC_FIELD,
        bot_prices=market,
    )
    payload = {
        "market": market,
        "base_seed": args.seed,
        "n_seeds": args.seeds,
        "n_sims": args.n_sims,
        "my_seat": args.my_seat,
        "season": args.season,
        "reg_win_pct": {n: result.summaries[n]["reg_win_pct"].point for n in result.summaries},
        "all_metrics": {
            n: {m: result.summaries[n][m].point for m in result.summaries[n]}
            for n in result.summaries
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} (market={market}, base_seed={args.seed}, {args.seeds} seeds)")
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    chunks = [json.loads(p.read_text()) for p in sorted(args.chunk_dir.glob("*.json"))]
    if not chunks:
        raise SystemExit(f"no chunk JSONs in {args.chunk_dir}")
    markets, rows, best = aggregate_chunks(chunks)
    print(f"{'model':<22}" + "".join(f"{m:>12}" for m in markets) + f"{'worst':>12}")
    for name, cells, worst in rows:
        print(f"{name:<22}" + "".join(f"{c:>12.3f}" for c in cells) + f"{worst:>12.3f}")
    print(f"\nbest worst-case reg_win_pct across markets: {best} ({rows[0][2]:.3f})")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cap-fix both-market tuning (crash-safe chunks).")
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run", help="Race the grid for one market + one seed chunk -> JSON.")
    r.add_argument("--vorp-table", type=Path, required=True)
    r.add_argument("--league-config", type=Path, required=True)
    r.add_argument("--my-seat", type=int, required=True)
    r.add_argument("--season", type=int, required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--n-sims", type=int, default=300)
    r.add_argument("--seed", type=int, default=0, help="Base seed for this chunk.")
    r.add_argument("--bot-prices", choices=("espn", "model"), required=True)
    r.add_argument("--data-root", type=Path, default=Path("data"))
    r.add_argument("--out", type=Path, required=True, help="Chunk JSON output path.")
    r.set_defaults(func=_run_chunk)
    a = sub.add_parser("aggregate", help="Combine chunk JSONs -> reg_win_pct table + finalist.")
    a.add_argument("--chunk-dir", type=Path, required=True)
    a.set_defaults(func=_aggregate)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
