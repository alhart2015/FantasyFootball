"""Seat-symmetric auction bake-off against a CONFIGURABLE opponent field.

`auction_seat_sweep.py` races the hero field against one hard-coded `_REALISTIC_FIELD`
(aggressive / patient / balanced, round-robin). That is the right generic baseline, but a real
league is not generic: the answer to "what should I do in MY auction?" depends on who else is in
the room. This runner takes the field as a named archetype mix, so a room described as
"mostly aggressive over-bidders plus a few under-bidders who hoard cash and buy depth" can be
modelled directly and the hero ranking re-derived under it.

Two differences from `auction_seat_sweep.py` beyond the field knob:

1. **Paired diffs are recorded, not discarded.** `run_auction_tournament` is already
   common-random-numbers paired (every contestant plays the identical auction and season draw per
   seed), so the paired diff against a control has the shared-world variance cancelled out — a far
   tighter test than comparing two overlapping marginal CIs. This is the Run-T methodology; Runs
   S and earlier threw the pairing away.
2. **Every metric is aggregated**, not just `reg_win_pct`, because a bid model that trades
   regular-season wins for championship equity is a real and interesting trade-off.

Crash-safe by construction: one (seat, market) per process, each writing its own chunk JSON (the
dev box's Raptor Lake fault wants bounded processes — memory `h2h-backtest-native-crash`).

Run:
    python scripts/auction_field_bakeoff.py run --vorp-table ... --league-config ... \
        --seat 1 --season 2026 --bot-prices espn --field overbidder --out chunk.json
    python scripts/auction_field_bakeoff.py aggregate --chunk-dir DIR [--control balanced]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy, StackRatioBid
from projections.draft.assistant.auction.market import (
    DEFAULT_PRICE_JITTER,
    AggressiveBot,
    BalancedBot,
    BotArchetype,
    PatientValueBot,
)
from projections.draft.assistant.auction.tournament import METRICS, run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _MODELS,
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.auction import has_usable_espn_prices

_Z95 = 1.959963984540054

# The hero contestants: every registered bid model, plus the low-gain convex StackRatio variants
# that Run T resolved as the only heroes to beat `balanced` in the less-circular ESPN market. They
# are library-tested opt-ins deliberately kept out of `_MODELS`, so name them explicitly here.
CONTESTANTS: dict[str, AuctionBidStrategy] = {
    **_MODELS,
    "sr_g0.1_c2": StackRatioBid(gain=0.1, curve=2.0),
    "sr_g0.2_c2": StackRatioBid(gain=0.2, curve=2.0),
    "sr_g0.3_c2": StackRatioBid(gain=0.3, curve=2.0),
}

# An under-bidder who buys depth and strands cash: halves its bid on studs (never wins one), pays
# exactly fair value — no premium — on the mid tier, and $1s the bottom third. Against a room that
# pays over the board it loses most contested lots, so it reaches the endgame with budget left.
# Contrast the stock `PatientValueBot`, whose 0.35 mid-tier premium makes it a depth *spender*.
_HOARDER = PatientValueBot(understud=0.5, midtier_premium=0.0, stud_frac=0.10, scrub_frac=0.35)


def build_field(name: str, overbid: float) -> list[BotArchetype]:
    """Named opponent-field mix, round-robined across the bot seats by the engine.

    `overbid` is the fraction over the field's own board that the aggressive archetypes pay; it
    applies only to fields that contain them.
    """
    if name == "realistic":  # the standing cross-run baseline; `overbid` does not apply
        return list(_REALISTIC_FIELD)
    if name == "overbidder":
        # 5-cycle -> with 11 bot seats: 9 over-bidders, 2 hoarders (~18% of the room).
        ob = AggressiveBot(overbid=overbid)
        return [ob, ob, ob, ob, _HOARDER]
    if name == "overbidder_only":  # sensitivity: no hoarders at all
        return [AggressiveBot(overbid=overbid)]
    if name == "balanced_field":  # sensitivity: a disciplined room
        return [BalancedBot()]
    raise ValueError(f"unknown field {name!r}")


FIELDS: tuple[str, ...] = ("realistic", "overbidder", "overbidder_only", "balanced_field")


# ---------------------------------------------------------------------------
# run: one (seat, market) chunk
# ---------------------------------------------------------------------------


def _run_chunk(args: argparse.Namespace) -> int:
    pool, config, availability, params = _load_tournament_inputs(
        args.vorp_table, args.league_config, season=args.season, data_root=args.data_root
    )
    market: Literal["espn", "model"] = "espn" if args.bot_prices == "espn" else "model"
    if market == "espn" and not has_usable_espn_prices(pool):
        # run_auction_tournament would silently fall back to model pricing (stderr warning only)
        # and still write the chunk as market="espn" — a mislabeled result. Fail loudly.
        raise SystemExit(
            "bot_prices='espn' but the pool has no usable espn_auction_dollars; the chunk would be "
            "mislabeled model-priced. Use --bot-prices model or a pool with ESPN values."
        )
    result = run_auction_tournament(
        CONTESTANTS,
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
        bot_archetypes=build_field(args.field, args.overbid),
        bot_prices=market,
        market_adp_jitter=args.market_adp_jitter,
    )
    payload = {
        "market": market,
        "seat": args.seat,
        "field": args.field,
        "overbid": args.overbid,
        "base_seed": args.seed,
        "n_seeds": args.seeds,
        "n_sims": args.n_sims,
        "season": args.season,
        "market_adp_jitter": args.market_adp_jitter,
        "reg_win_pct": {n: result.summaries[n]["reg_win_pct"].point for n in result.summaries},
        "all_metrics": {
            n: {m: result.summaries[n][m].point for m in METRICS} for n in result.summaries
        },
        # pair key -> metric -> [point, lo95, hi95]; the CRN-paired signal (Run T).
        "paired": {
            pair: {m: [iv.point, iv.lo_95, iv.hi_95] for m, iv in metrics.items()}
            for pair, metrics in result.paired_diffs.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.out} (field={args.field} overbid={args.overbid} market={market} "
        f"seat={args.seat}, {args.seeds} seeds)"
    )
    return 0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class PairedRow(NamedTuple):
    """One contestant's seat-stratified paired diff vs the control, per metric."""

    name: str
    point: dict[str, float]
    se: dict[str, float]
    n_seats: int

    def significant(self, metric: str) -> bool:
        """95% CI excludes zero."""
        lo = self.point[metric] - _Z95 * self.se[metric]
        hi = self.point[metric] + _Z95 * self.se[metric]
        return lo > 0.0 or hi < 0.0


def _oriented(
    paired: dict[str, dict[str, list[float]]], name: str, control: str
) -> dict[str, list[float]] | None:
    """The `name - control` diff from a chunk's paired block, flipping sign if stored the other way.

    `run_auction_tournament` keys pairs as `f"{a}_vs_{b}"` over `combinations(strategies, 2)`, so
    only one of the two orientations exists for any pair.
    """
    if (fwd := paired.get(f"{name}_vs_{control}")) is not None:
        return fwd
    if (rev := paired.get(f"{control}_vs_{name}")) is not None:
        return {m: [-v[0], -v[2], -v[1]] for m, v in rev.items()}
    return None


def stratified_paired(
    chunks: Sequence[dict[str, object]], control: str
) -> dict[str, list[PairedRow]]:
    """Combine per-seat paired CIs into one seat-stratified diff per (market, contestant).

    Seats are FIXED STRATA with weight 1/n_seats — that is the goal metric's own definition
    (`reg_win_pct` in a league whose seat is uniformly random), not a random sample to be pooled by
    precision. Per-seat SE is recovered from the bootstrap CI half-width and the strata combined as
    `SE = sqrt(sum SE_s^2) / n_seats`, exact for independent seats under a normal approximation.
    Seats ARE independent here: each is its own process with its own draws.
    """
    # (market, name) -> metric -> list of (point, se) across seats
    acc: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {}
    for c in chunks:
        paired = c.get("paired")
        if not isinstance(paired, dict):
            continue
        market = str(c["market"])
        contestants = c["all_metrics"]
        if not isinstance(contestants, dict):
            continue
        for name in sorted(set(map(str, contestants)) - {control}):
            block = _oriented(paired, name, control)
            if block is None:
                continue
            per_metric = acc.setdefault((market, name), {})
            for metric, (point, lo, hi) in block.items():
                per_metric.setdefault(metric, []).append((point, (hi - lo) / (2.0 * _Z95)))
    out: dict[str, list[PairedRow]] = {}
    for (market, name), per_metric in acc.items():
        points = {m: sum(p for p, _ in v) / len(v) for m, v in per_metric.items()}
        ses = {m: math.sqrt(sum(s * s for _, s in v)) / len(v) for m, v in per_metric.items()}
        n_seats = max(len(v) for v in per_metric.values())
        out.setdefault(market, []).append(PairedRow(name, points, ses, n_seats))
    for rows in out.values():
        rows.sort(key=lambda r: r.point.get("reg_win_pct", 0.0), reverse=True)
    return out


def seat_averages(chunks: Sequence[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    """(market -> contestant -> metric -> seat-averaged mean). Every seat weighted equally; a
    re-run of the same (market, seat) counts once (last value wins — a corrected re-run
    supersedes)."""
    cell: dict[tuple[str, str, int], dict[str, float]] = {}
    for c in chunks:
        allm = c["all_metrics"]
        if not isinstance(allm, dict):
            raise ValueError(f"chunk all_metrics must be a dict; got {type(allm)}")
        for name, metrics in allm.items():
            cell[(str(c["market"]), str(name), int(c["seat"]))] = {  # type: ignore[call-overload]
                str(m): float(v) for m, v in metrics.items()
            }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (market, name, _seat), metrics in cell.items():
        bucket = out.setdefault(market, {}).setdefault(name, {})
        bucket.setdefault("_n", 0.0)
        bucket["_n"] += 1
        for m, v in metrics.items():
            bucket[m] = bucket.get(m, 0.0) + v
    for per_name in out.values():
        for bucket in per_name.values():
            n = bucket.pop("_n")
            for m in list(bucket):
                bucket[m] /= n
    return out


def _load_chunks(chunk_dir: Path) -> tuple[list[dict[str, object]], int]:
    """Read every *.json under `chunk_dir`, skipping anything unreadable or not a bake-off chunk.
    Robust by design — a byte-corrupt or foreign file must not abort aggregation."""
    chunks: list[dict[str, object]] = []
    skipped = 0
    for p in sorted(chunk_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())  # ValueError covers JSON + Unicode decode errors
        except (ValueError, OSError) as exc:
            skipped += 1
            print(f"skipping unreadable chunk {p.name}: {exc}")
            continue
        if not (isinstance(data, dict) and {"market", "seat", "all_metrics"} <= set(data)):
            skipped += 1
            print(f"skipping non-chunk json {p.name}")
            continue
        chunks.append(data)
    return chunks, skipped


def _guard_homogeneous(chunks: Sequence[dict[str, object]]) -> None:
    """Refuse to pool chunks that priced different markets. Mixing nomination models or opponent
    fields into one average yields a winner that matches no real configuration."""
    for key in ("market_adp_jitter", "field", "overbid", "n_seeds", "n_sims"):
        seen = {json.dumps(c.get(key)) for c in chunks}
        if len(seen) > 1:
            raise ValueError(
                f"chunks disagree on {key}: {sorted(seen)}; separate them by chunk directory "
                "— pooling different market configurations silently blends regimes."
            )


def _aggregate(args: argparse.Namespace) -> int:
    chunks, skipped = _load_chunks(args.chunk_dir)
    if not chunks:
        raise SystemExit(f"no readable chunk JSONs in {args.chunk_dir}")
    _guard_homogeneous(chunks)
    head = chunks[0]
    print(
        f"field={head.get('field')} overbid={head.get('overbid')} "
        f"adp_jitter={head.get('market_adp_jitter')} seeds={head.get('n_seeds')} "
        f"sims={head.get('n_sims')} chunks={len(chunks)}"
    )
    avg = seat_averages(chunks)
    markets = sorted(avg)
    for market in markets:
        seats = sorted({int(c["seat"]) for c in chunks if c["market"] == market})  # type: ignore[call-overload]
        print(f"\n=== market={market} — seat-averaged over {len(seats)} seats {seats} ===")
        print(f"{'model':<14}" + "".join(f"{m:>18}" for m in METRICS))
        for name, metrics in sorted(
            avg[market].items(), key=lambda kv: kv[1].get("reg_win_pct", 0.0), reverse=True
        ):
            print(
                f"{name:<14}" + "".join(f"{metrics.get(m, float('nan')):>18.4f}" for m in METRICS)
            )

    # Worst-case across markets on reg_win_pct: the robust-win goal metric.
    if len(markets) > 1:
        names = set.intersection(*(set(avg[m]) for m in markets))
        print("\n=== worst-case seat-avg reg_win_pct across markets (robust-win metric) ===")
        worst = sorted(
            ((n, min(avg[m][n]["reg_win_pct"] for m in markets)) for n in names),
            key=lambda kv: kv[1],
            reverse=True,
        )
        for name, w in worst:
            cells = "".join(f"{avg[m][name]['reg_win_pct']:>10.4f}" for m in markets)
            print(f"{name:<14}{cells}{w:>10.4f}")

    paired = stratified_paired(chunks, args.control)
    for market in sorted(paired):
        print(
            f"\n=== market={market} — CRN-paired diff vs `{args.control}` "
            "(95% CI, * excludes 0) ==="
        )
        print(f"{'model':<14}" + "".join(f"{m:>28}" for m in METRICS))
        for row in paired[market]:
            cells = ""
            for m in METRICS:
                if m not in row.point:
                    cells += f"{'—':>28}"
                    continue
                half = _Z95 * row.se[m]
                star = "*" if row.significant(m) else " "
                cells += (
                    f"{row.point[m]:+.4f} [{row.point[m] - half:+.4f},"
                    f"{row.point[m] + half:+.4f}]{star}"
                ).rjust(28)
            print(f"{row.name:<14}{cells}")
    if skipped:
        print(f"\nWARNING: {skipped} chunk(s) excluded; results are a reduced sample.")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seat-symmetric bake-off vs a configurable field.")
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run", help="Race every contestant at one seat + one market -> JSON.")
    r.add_argument("--vorp-table", type=Path, required=True)
    r.add_argument("--league-config", type=Path, required=True)
    r.add_argument("--seat", type=int, required=True, help="Hero seat (1-based).")
    r.add_argument("--season", type=int, required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--n-sims", type=int, default=300)
    r.add_argument("--seed", type=int, default=0, help="Base RNG seed (shared across seats = CRN).")
    r.add_argument("--bot-prices", choices=("espn", "model"), required=True)
    r.add_argument("--field", choices=FIELDS, default="overbidder")
    r.add_argument(
        "--overbid",
        type=float,
        default=0.20,
        help="Fraction over its own board the aggressive archetypes pay (0 = value-rational).",
    )
    r.add_argument(
        "--market-adp-jitter",
        type=float,
        default=12.0,
        help="Flush seats nominate off a shared noisy-ADP board with this jitter (the realistic "
        "nomination model, Run P). Pass 0 or omit --market-adp-jitter for value-weighted nom.",
    )
    r.add_argument("--data-root", type=Path, default=Path("data"))
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=_run_chunk)
    a = sub.add_parser("aggregate", help="Combine chunk JSONs -> seat-avg + paired-CI tables.")
    a.add_argument("--chunk-dir", type=Path, required=True)
    a.add_argument("--control", default="balanced", help="Paired-diff reference contestant.")
    a.set_defaults(func=_aggregate)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
