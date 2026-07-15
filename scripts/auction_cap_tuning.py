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
from typing import Literal, NamedTuple

from projections.draft.assistant.auction.bid_strategy import (
    AuctionBidStrategy,
    BalancedValueBid,
    PatientValueBid,
)
from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.tournament import run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.auction import has_usable_espn_prices

PACES: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
PREMIUMS: tuple[float, ...] = (0.5, 1.0, 1.5)


class MarketRow(NamedTuple):
    """One model's aggregated result. `cells` are per-market seed-weighted reg_win_pct means aligned
    to `markets` order (None where the model was not scored in that market); `worst` is the min over
    present markets; `complete` is True iff the model was scored in every market present."""

    name: str
    cells: list[float | None]
    worst: float
    complete: bool


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
) -> tuple[list[str], list[MarketRow], str]:
    """Combine per-chunk reg_win_pct into (markets, rows, best). Each (market, model) mean is
    seed-weighted by the chunk's `n_seeds`, and duplicate (market, model, base_seed) chunks are
    counted once (a re-run chunk with the same base_seed is not double-weighted). Only
    coverage-complete models are eligible to be `best` — a "worst-case across markets" from partial
    coverage is not a real worst-case. `rows` is sorted worst-case desc for display; `best` (which
    skips partials) is the authoritative winner and may differ from rows[0]."""
    # (market, model) -> base_seed -> (value, weight); the base_seed key dedups re-run chunks.
    acc: dict[tuple[str, str], dict[object, tuple[float, float]]] = {}
    for i, c in enumerate(chunks):
        m = str(c["market"])
        rwp = c["reg_win_pct"]
        if not isinstance(rwp, dict):
            raise ValueError(f"chunk reg_win_pct must be a dict; got {type(rwp)}")
        raw_w = c.get("n_seeds", 1)
        weight = float(raw_w) if isinstance(raw_w, (int, float)) else 1.0  # seed-weighted mean
        # dedup key; a str fallback for a missing base_seed can't collide with real int base_seeds.
        key = c.get("base_seed", f"__chunk_{i}__")
        for name, val in rwp.items():
            acc.setdefault((m, str(name)), {}).setdefault(key, (float(val), weight))
    markets = sorted({mk for mk, _ in acc})
    names = sorted({nm for _, nm in acc})
    rows: list[MarketRow] = []
    for name in names:
        cells: list[float | None] = []
        for m in markets:
            entries = acc.get((m, name))
            if not entries:
                cells.append(None)  # not scored in this market — keep the column aligned
                continue
            total_w = sum(w for _v, w in entries.values())
            cells.append(sum(v * w for v, w in entries.values()) / total_w)
        present = [c for c in cells if c is not None]
        if not present:
            continue
        rows.append(MarketRow(name, cells, min(present), len(present) == len(markets)))
    rows.sort(key=lambda r: r.worst, reverse=True)
    complete_rows = [r for r in rows if r.complete]
    best = complete_rows[0].name if complete_rows else ""
    return markets, rows, best


def _run_chunk(args: argparse.Namespace) -> int:
    pool, config, availability, params = _load_tournament_inputs(
        args.vorp_table, args.league_config, season=args.season, data_root=args.data_root
    )
    market: Literal["espn", "model"] = "espn" if args.bot_prices == "espn" else "model"
    if market == "espn" and not has_usable_espn_prices(pool):
        # run_auction_tournament would silently fall back to model pricing (only a stderr warning),
        # and this chunk would still be written as market="espn" — a mislabeled result. (A rarer
        # fallback — espn_anchored_bot_prices raising on degenerate drift — is not caught here.)
        raise SystemExit(
            "bot_prices='espn' but the pool has no usable espn_auction_dollars; the chunk would be "
            "mislabeled model-priced. Use --bot-prices model or a pool with ESPN values."
        )
    result = run_auction_tournament(
        grid(),
        pool,
        config,
        my_seat=args.my_seat,
        n_seeds=args.seeds,
        price_jitter=DEFAULT_PRICE_JITTER,
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


def _load_chunks(chunk_dir: Path) -> tuple[list[dict[str, object]], int]:
    """Read every *.json under `chunk_dir`, skipping any that is unreadable, non-JSON, or not a
    chunk dict (missing 'market'/'reg_win_pct'). Returns (valid_chunks, skipped_count). Robust by
    design — a byte-corrupted or foreign file must not abort aggregation (crash-safety)."""
    chunks: list[dict[str, object]] = []
    skipped = 0
    for p in sorted(chunk_dir.glob("*.json")):
        try:
            # ValueError covers JSONDecodeError + UnicodeDecodeError (both ValueError subclasses).
            data = json.loads(p.read_text())
        except (ValueError, OSError) as exc:
            skipped += 1
            print(f"skipping unreadable chunk {p.name}: {exc}")
            continue
        if not (isinstance(data, dict) and "market" in data and "reg_win_pct" in data):
            skipped += 1
            print(f"skipping non-chunk json {p.name} (missing market/reg_win_pct)")
            continue
        chunks.append(data)
    return chunks, skipped


def _aggregate(args: argparse.Namespace) -> int:
    chunks, skipped = _load_chunks(args.chunk_dir)
    if not chunks:
        raise SystemExit(f"no readable chunk JSONs in {args.chunk_dir}")
    markets, rows, best = aggregate_chunks(chunks)
    if not rows:
        print("no reg_win_pct data in chunks")
    else:
        print(f"{'model':<22}" + "".join(f"{m:>12}" for m in markets) + f"{'worst':>12}")
        for row in rows:
            flag = "" if row.complete else "  (partial coverage)"
            cells = "".join(f"{c:>12.3f}" if c is not None else f"{'—':>12}" for c in row.cells)
            print(f"{row.name:<22}{cells}{row.worst:>12.3f}{flag}")
        if best:
            best_worst = next(r.worst for r in rows if r.name == best)
            print(f"\nbest worst-case reg_win_pct across all markets: {best} ({best_worst:.3f})")
        else:
            print("\nno model was scored in every market — no cross-market worst-case winner")
    if skipped:  # always report lost chunks, even when the readable ones yielded no rows
        print(f"WARNING: {skipped} chunk(s) excluded; results are a reduced sample.")
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
    r.add_argument(
        "--seed", type=int, default=0, help="Base seed for this chunk (distinct per chunk)."
    )
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
