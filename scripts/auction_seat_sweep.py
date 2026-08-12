"""Full-field post-seat-fix sweep: race ALL registered bid heroes (`auction.registry.BID_MODELS`)
across every hero seat (1..N) in BOTH bot markets, scored by reg_win_pct, and rank by the
seat-averaged worst-case across markets (the robust-win-hero goal metric).

Why this exists: the seat-1 `resolve_bids` tie-break bug (fixed in PR #95) biased every prior
full-field ranking downward — and it was only ever measured at seat 1, the one broken seat. The
post-fix validation only re-ran a 2-hero probe (`balanced` + `patient_deep`). This runner produces
the first apples-to-apples, seat-symmetric ranking of the whole field with the fix in place, so the
"balanced is the win% leader" claim rests on a clean comparison and Slice 2 (nomination poisoning)
has an honest baseline to beat.

Crash-safe: one (seat, market) per process (the dev box's Raptor Lake fault wants bounded
processes — memory h2h-backtest-native-crash). `run` writes a per-chunk JSON; `aggregate` combines
the chunk JSONs into a seat-averaged reg_win_pct table + robust finalist. Data-gathering; no default
changes. Mirrors `auction_cap_tuning.py`'s methodology exactly (same inputs, same field, same
nomination/bot config) but sweeps the seat axis instead of a pace x premium grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, NamedTuple

from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.registry import BID_MODELS
from projections.draft.assistant.auction.tournament import run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.auction import has_usable_espn_prices


class SweepRow(NamedTuple):
    """One hero's seat-averaged result. `seat_avg` is the per-market mean reg_win_pct over the
    seats it was scored in, aligned to `markets` order (None where the hero has no cell in that
    market). `worst` is the min seat-average over present markets — the robust-win metric.
    `complete` is True iff the hero has a reg_win_pct in every (market, seat) cell in the sweep, the
    gate for eligibility as the cross-market winner. `per_seat[market]` is the per-seat vector
    (aligned to `seats`) kept so a new outlier seat is visible, not averaged away."""

    name: str
    seat_avg: list[float | None]
    worst: float
    complete: bool
    per_seat: dict[str, list[float | None]]


def aggregate_seat_sweep(
    chunks: list[dict[str, object]],
) -> tuple[list[str], list[int], list[SweepRow], str]:
    """Combine per-(seat, market) chunks into (markets, seats, rows, best). Each hero's per-market
    figure is the simple mean of its reg_win_pct across the seats present (every seat carries equal
    weight — the goal metric is reg_win_pct in a uniformly-random-seat league). A re-run of the same
    (market, seat) counts once (last value wins — a corrected re-run supersedes). Only
    coverage-complete heroes (scored in every market x seat cell) are eligible to be `best`; a
    worst-case built from missing cells is not a real worst-case. `rows` is sorted worst-case desc
    for display; `best` (skipping partials) is the authoritative robust winner."""
    # Guard the nomination axis: value-nom (market_adp_jitter absent/None) and ADP-nom chunks price
    # different markets, so pooling them into one (market, seat) average silently blends regimes and
    # yields a winner that matches no real configuration. Refuse rather than mislead.
    jitters = {c.get("market_adp_jitter") for c in chunks}
    if len(jitters) > 1:
        raise ValueError(
            f"chunks mix market_adp_jitter values {sorted(map(str, jitters))}; value-nom and "
            "ADP-nom drafts must not be aggregated together — separate them by chunk directory."
        )
    # (market, model, seat) -> reg_win_pct; the seat key dedups re-run chunks of the same cell.
    cell: dict[tuple[str, str, int], float] = {}
    for c in chunks:
        m = str(c["market"])
        seat = int(c["seat"])  # type: ignore[call-overload]
        rwp = c["reg_win_pct"]
        if not isinstance(rwp, dict):
            raise ValueError(f"chunk reg_win_pct must be a dict; got {type(rwp)}")
        for name, val in rwp.items():
            cell[(m, str(name), seat)] = float(val)
    markets = sorted({m for m, _n, _s in cell})
    seats = sorted({s for _m, _n, s in cell})
    names = sorted({n for _m, n, _s in cell})
    rows: list[SweepRow] = []
    for name in names:
        seat_avg: list[float | None] = []
        per_seat: dict[str, list[float | None]] = {}
        for m in markets:
            vec: list[float | None] = [cell.get((m, name, s)) for s in seats]
            per_seat[m] = vec
            present = [v for v in vec if v is not None]
            seat_avg.append(sum(present) / len(present) if present else None)
        present_avg = [a for a in seat_avg if a is not None]
        if not present_avg:
            continue
        complete = all(v is not None for vec in per_seat.values() for v in vec)
        rows.append(SweepRow(name, seat_avg, min(present_avg), complete, per_seat))
    rows.sort(key=lambda r: r.worst, reverse=True)
    complete_rows = [r for r in rows if r.complete]
    best = complete_rows[0].name if complete_rows else ""
    return markets, seats, rows, best


def _run_chunk(args: argparse.Namespace) -> int:
    pool, config, availability, params = _load_tournament_inputs(
        args.vorp_table, args.league_config, season=args.season, data_root=args.data_root
    )
    market: Literal["espn", "model"] = "espn" if args.bot_prices == "espn" else "model"
    if market == "espn" and not has_usable_espn_prices(pool):
        # run_auction_tournament would silently fall back to model pricing (only a stderr warning)
        # and still write the chunk as market="espn" — a mislabeled result. Fail loudly instead.
        raise SystemExit(
            "bot_prices='espn' but the pool has no usable espn_auction_dollars; the chunk would be "
            "mislabeled model-priced. Use --bot-prices model or a pool with ESPN values."
        )
    result = run_auction_tournament(
        BID_MODELS,
        pool,
        config,
        my_seat=args.seat,
        n_seeds=args.seeds,
        price_jitter=DEFAULT_PRICE_JITTER,
        base_seed=args.seed,
        n_sims=args.n_sims,
        availability=availability,
        params=params,
        nomination_temp=1.0,
        bot_archetypes=_REALISTIC_FIELD,
        bot_prices=market,
        market_adp_jitter=args.market_adp_jitter,
    )
    payload = {
        "market": market,
        "seat": args.seat,
        "base_seed": args.seed,
        "n_seeds": args.seeds,
        "n_sims": args.n_sims,
        "season": args.season,
        "market_adp_jitter": args.market_adp_jitter,
        "reg_win_pct": {n: result.summaries[n]["reg_win_pct"].point for n in result.summaries},
        "all_metrics": {
            n: {m: result.summaries[n][m].point for m in result.summaries[n]}
            for n in result.summaries
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} (market={market}, seat={args.seat}, {args.seeds} seeds)")
    return 0


def _load_chunks(chunk_dir: Path) -> tuple[list[dict[str, object]], int]:
    """Read every *.json under `chunk_dir`, skipping any that is unreadable, non-JSON, or not a
    sweep chunk (missing 'market'/'seat'/'reg_win_pct'). Returns (valid_chunks, skipped_count).
    Robust by design — a byte-corrupt or foreign file must not abort aggregation (crash-safety)."""
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
        if not (
            isinstance(data, dict) and "market" in data and "seat" in data and "reg_win_pct" in data
        ):
            skipped += 1
            print(f"skipping non-chunk json {p.name} (missing market/seat/reg_win_pct)")
            continue
        chunks.append(data)
    return chunks, skipped


def _aggregate(args: argparse.Namespace) -> int:
    chunks, skipped = _load_chunks(args.chunk_dir)
    if not chunks:
        raise SystemExit(f"no readable chunk JSONs in {args.chunk_dir}")
    markets, seats, rows, best = aggregate_seat_sweep(chunks)
    if not rows:
        print("no reg_win_pct data in chunks")
    else:
        print(f"seats swept: {seats}")
        print(f"{'model':<14}" + "".join(f"{m:>10}" for m in markets) + f"{'worst':>10}")
        for row in rows:
            flag = "" if row.complete else "  (partial coverage)"
            cells = "".join(f"{a:>10.3f}" if a is not None else f"{'—':>10}" for a in row.seat_avg)
            print(f"{row.name:<14}{cells}{row.worst:>10.3f}{flag}")
        if best:
            best_worst = next(r.worst for r in rows if r.name == best)
            print(f"\nbest worst-case seat-avg reg_win_pct: {best} ({best_worst:.3f})")
        else:
            print("\nno model was scored in every market x seat — no cross-market winner")
        # per-seat vectors for the top few, so a new outlier seat is visible (not averaged away).
        print("\nper-seat reg_win_pct (top 4 by worst-case):")
        for row in rows[:4]:
            print(f"  {row.name}")
            for m in markets:
                vec = "".join(
                    f"{v:>7.3f}" if v is not None else f"{'—':>7}" for v in row.per_seat[m]
                )
                print(f"    {m:<6}{vec}")
    if skipped:  # always report lost chunks, even when the readable ones yielded no rows
        print(f"WARNING: {skipped} chunk(s) excluded; results are a reduced sample.")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-field seat sweep (crash-safe chunks).")
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run", help="Race the full field at one seat + one market -> JSON.")
    r.add_argument("--vorp-table", type=Path, required=True)
    r.add_argument("--league-config", type=Path, required=True)
    r.add_argument("--seat", type=int, required=True, help="Hero seat (1-based).")
    r.add_argument("--season", type=int, required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--n-sims", type=int, default=300)
    r.add_argument("--seed", type=int, default=0, help="Base RNG seed (shared across seats = CRN).")
    r.add_argument("--bot-prices", choices=("espn", "model"), required=True)
    r.add_argument(
        "--market-adp-jitter",
        type=float,
        default=None,
        help="If set, flush seats nominate by a shared noisy-ADP market board with this jitter "
        "(realistic ADP-ordered nomination) instead of value-weighted-random. Omit = value nom.",
    )
    r.add_argument("--data-root", type=Path, default=Path("data"))
    r.add_argument("--out", type=Path, required=True, help="Chunk JSON output path.")
    r.set_defaults(func=_run_chunk)
    a = sub.add_parser("aggregate", help="Combine chunk JSONs -> seat-avg table + robust finalist.")
    a.add_argument("--chunk-dir", type=Path, required=True)
    a.set_defaults(func=_aggregate)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
