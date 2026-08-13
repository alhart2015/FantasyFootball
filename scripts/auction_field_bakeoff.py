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

`--nominator-probe BID_MODEL` flips what is being raced: instead of many bid models under one
nomination policy, it races many NOMINATION policies under one bid model. Same field, same seeds,
same pairing — see `NOMINATOR_PROBE`.

Run:
    python scripts/auction_field_bakeoff.py run --vorp-table ... --league-config ... \
        --seat 1 --season 2026 --bot-prices espn --field overbidder --out chunk.json
    python scripts/auction_field_bakeoff.py aggregate --chunk-dir DIR [--control balanced]

    # nomination probe (contestants: control / off_pos / gap / gap_off)
    python scripts/auction_field_bakeoff.py run ... --nominator-probe overbid_noramp --out c.json
    python scripts/auction_field_bakeoff.py aggregate --chunk-dir DIR --control control
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy
from projections.draft.assistant.auction.market import (
    DEFAULT_PRICE_JITTER,
    AggressiveBot,
    BalancedBot,
    BotArchetype,
    PatientValueBot,
)
from projections.draft.assistant.auction.nomination import (
    HeroNominator,
    drain_off_position,
    drain_value_gap,
    drain_value_gap_off_position,
)
from projections.draft.assistant.auction.registry import ALL_BID_MODELS
from projections.draft.assistant.auction.tournament import METRICS, run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.auction import has_usable_espn_prices

_Z95 = 1.959963984540054

# The hero contestants: every registered bid model, plus the library-tested opt-ins (the no-ramp
# overbid variant and the low-gain convex StackRatio variants that Run T resolved as the only
# heroes to beat `balanced` in the less-circular ESPN market), which are deliberately kept out of
# the tournament roster. `registry.ALL_BID_MODELS` is exactly that union.
CONTESTANTS: Mapping[str, AuctionBidStrategy] = ALL_BID_MODELS

# The under-bidder: the library's stock `PatientValueBot` at its defaults — under-bids studs at
# 0.5x (never wins one), pays a mid-tier premium out of the reserve it saved, $1s the bottom half.
# Deliberately NOT re-tuned to force leftover cash: how much a patient bidder actually strands is
# an OUTPUT of the market (how hard the room bids it out of lots), not an input to be assumed.
_HOARDER = PatientValueBot()


# Pace cap for the over-bidding archetype, calibrated against the realized price curve rather than
# assumed: at 4.5x the even per-slot share the top player of a 12-team $200 draft clears ~$69 (~34%
# of a budget), ~13 players beat $50, and 19% of the room's money is still live at pick 48 of 156.
# The UNPACED AggressiveBot instead clears its top player at ~$105 and is down to 7% by pick 48,
# which forces stars-and-scrubs on every seat and makes the late pool free — a caricature that
# flatters any hero whose plan is to buy the tail. See `overbidder_unpaced` to reproduce it.
_OVERBID_PACE = 4.5
# "opening" holds the cap at a CONSTANT `pace x budget/roster_size`, so a bot can chase two or
# three expensive players and then be forced to $1 filler. Under the "running" basis the cap
# shrinks with every purchase, so a seat that lands one stud can never afford a second: only ~5 of
# 12 seats ended with two $50+ players, versus ~10 of 12 here, and almost no seat went star-heavy
# early then bargain-hunting late. Real aggressive managers bust that way; the running basis
# structurally forbids it.
_OVERBID_BASIS: Literal["running", "opening"] = "opening"
# Per-seat spread of the pace cap, as a fraction of `pace`. A field of IDENTICAL over-bidders puts
# the same hard ceiling on all of them -- at pace 4.5 every bot stops dead at $69 -- and the hero,
# which has no cap of its own, wins any player it wants by bidding exactly $70. That is a wall to
# step over, not a market, and it hands a free win to whichever hero is willing to pay over sheet
# value. Spreading the cap across seats (here 4.5 +/- 35% -> ceilings from ~$45 to ~$93) removes the
# single cliff: there is always some seat still bidding above the hero's next dollar.
_OVERBID_PACE_JITTER = 0.35
_PATIENT_EVERY = 5  # every 5th bot seat is a patient bidder (~2 of 11 in a 12-team league)

# --nominator-probe: race NOMINATION policies at ONE fixed bid, instead of racing bid models.
# `run_auction_tournament` is already CRN-paired per contestant, so passing the identical bid model
# under four names and varying only `hero_nominators` reuses the whole pairing + aggregation stack.
# `control` maps to None = the engine's own nomination. See
# docs/superpowers/specs/2026-08-12-auction-value-gap-nomination-design.md.
NOMINATOR_PROBE: dict[str, HeroNominator | None] = {
    "control": None,
    "off_pos": drain_off_position,  # the Run-O incumbent (price-ranked, off-position)
    "gap": drain_value_gap,  # rank by the room's overpay vs our board
    "gap_off": drain_value_gap_off_position,  # both signals composed
}


def _spread_paces(pace: float, jitter: float, n: int) -> list[float]:
    """`n` pace values evenly spanning `pace*(1±jitter)`, deterministic (no RNG -> CRN-safe).

    Ordered low-to-high; which seat draws which is set by `assign_bot_archetypes` and shifts as the
    hero seat moves, so the seat sweep already averages over the assignment.
    """
    if n <= 1:
        return [pace]
    lo, hi = pace * (1.0 - jitter), pace * (1.0 + jitter)
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def build_field(
    name: str,
    overbid: float,
    pace: float = _OVERBID_PACE,
    basis: Literal["running", "opening"] = _OVERBID_BASIS,
    *,
    n_bots: int | None = None,
    pace_jitter: float = _OVERBID_PACE_JITTER,
    n_patient: int | None = None,
) -> list[BotArchetype]:
    """Named opponent-field mix, round-robined across the bot seats by the engine.

    `overbid` is the fraction over the field's own board that the aggressive archetypes pay;
    `pace` caps any one buy at that multiple of a per-slot share, and `basis` picks whether that
    share is the shrinking running one or the constant opening one.

    When `n_bots` is given and `pace_jitter > 0`, the returned list is exactly `n_bots` long and
    every aggressive seat gets its OWN cap, so the room has no single ceiling. Without `n_bots` the
    5-entry cycle is returned unchanged (identical caps), which is what earlier runs used.

    `n_patient` sets exactly how many bot seats are conservative hoarders, spread as evenly as
    possible so they are never clustered next to the hero. `None` keeps the historical rule (every
    `_PATIENT_EVERY`-th seat -> 2 of 11 in a 12-team league), so every prior run reproduces
    byte-for-byte; pass an int only to sweep the aggressive/conservative mix.
    """
    if name == "realistic":  # the standing cross-run baseline; overbid/pace do not apply
        return list(_REALISTIC_FIELD)
    if name == "overbidder_unpaced":  # the no-pace-cap caricature, kept reproducible
        ob_u = AggressiveBot(overbid=overbid)
        return [ob_u, ob_u, ob_u, ob_u, _HOARDER]
    if name == "balanced_field":  # sensitivity: a disciplined room that pays fair value
        return [BalancedBot()]
    if name not in ("overbidder", "overbidder_only"):
        raise ValueError(f"unknown field {name!r}")

    with_patient = name == "overbidder"
    if n_bots is None or pace_jitter <= 0.0:  # uniform-cap cycle (pre-jitter behaviour)
        ob = BalancedBot(pace=pace, overbid=overbid, pace_basis=basis)
        return [ob, ob, ob, ob, _HOARDER] if with_patient else [ob]
    seats = list(range(n_bots))
    if not with_patient:
        patient: set[int] = set()
    elif n_patient is None:
        patient = {i for i in seats if i % _PATIENT_EVERY == _PATIENT_EVERY - 1}
    else:
        if not 0 <= n_patient <= n_bots:
            raise ValueError(f"n_patient must be in 0..{n_bots}; got {n_patient}")
        # Evenly spaced rather than the first/last k: clustering the hoarders would change who the
        # hero sits next to as well as how many there are, confounding the mix with seat adjacency.
        # Half-step offset ((2i+1)n / 2k) so the set never starts at seat 0. Consecutive values
        # differ by n/k >= 1, so this yields exactly `n_patient` distinct seats -- a plain
        # round(i*n/k) does NOT (it collides once k approaches n, silently returning too few).
        patient = {(2 * i + 1) * n_bots // (2 * n_patient) for i in range(n_patient)}
    paces = _spread_paces(pace, pace_jitter, n_bots - len(patient))
    out: list[BotArchetype] = []
    agg = 0
    for i in seats:
        if i in patient:
            out.append(_HOARDER)
            continue
        out.append(BalancedBot(pace=paces[agg], overbid=overbid, pace_basis=basis))
        agg += 1
    return out


FIELDS: tuple[str, ...] = (
    "realistic",
    "overbidder",
    "overbidder_unpaced",
    "overbidder_only",
    "balanced_field",
)


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
    contestants, nominators = CONTESTANTS, None
    if args.nominator_probe is not None:
        if args.nominator_probe not in ALL_BID_MODELS:
            raise SystemExit(
                f"unknown bid model {args.nominator_probe!r}; choose from {sorted(ALL_BID_MODELS)}"
            )
        if market != "espn":
            # The gap heuristics rank by (room's price - our value), and under model pricing the
            # room prices off our own numbers, so every gap is 0 and they degenerate to an
            # arbitrary tie-break. A model-market probe would report a meaningless null.
            raise SystemExit(
                "--nominator-probe requires --bot-prices espn: under model pricing the room shares "
                "our board, so the value-gap signal is identically zero."
            )
        nominators = NOMINATOR_PROBE
        contestants = {name: ALL_BID_MODELS[args.nominator_probe] for name in NOMINATOR_PROBE}
    result = run_auction_tournament(
        contestants,
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
        bot_archetypes=build_field(
            args.field,
            args.overbid,
            args.overbid_pace,
            args.overbid_basis,
            n_bots=config.n_teams - 1,
            pace_jitter=args.overbid_pace_jitter,
            n_patient=args.n_patient,
        ),
        bot_prices=market,
        market_adp_jitter=args.market_adp_jitter,
        hero_nominators=nominators,
    )
    payload = {
        "market": market,
        "seat": args.seat,
        # None for a bid-model bake-off; the fixed hero bid when racing nominators. Guarded as a
        # homogeneity key so probe chunks can never be pooled into a bake-off average.
        "nominator_probe": args.nominator_probe,
        "field": args.field,
        "overbid": args.overbid,
        "overbid_pace": args.overbid_pace,
        "overbid_basis": args.overbid_basis,
        "overbid_pace_jitter": args.overbid_pace_jitter,
        # None = the historical every-5th rule (2 of 11); an int is a swept mix.
        "n_patient": args.n_patient,
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
    for key in (
        "nominator_probe",
        "market_adp_jitter",
        "field",
        "overbid",
        "overbid_pace",
        "overbid_basis",
        "overbid_pace_jitter",
        "n_patient",
        "n_seeds",
        "n_sims",
    ):
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
        f"nominator_probe={head.get('nominator_probe')} "
        f"field={head.get('field')} overbid={head.get('overbid')} "
        f"pace={head.get('overbid_pace')}/{head.get('overbid_basis')} "
        f"pace_jitter={head.get('overbid_pace_jitter')} "
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
        "--overbid-pace",
        type=float,
        default=_OVERBID_PACE,
        help="Cap on any one buy by the aggressive archetypes, as a multiple of their even "
        "per-slot share. Calibrated to 4.5 (top stud ~$69 in a 12-team $200 draft).",
    )
    r.add_argument(
        "--overbid-basis",
        choices=("running", "opening"),
        default=_OVERBID_BASIS,
        help="Whether the pace cap shrinks as a bot spends ('running') or stays at the constant "
        "opening per-slot share ('opening', the default — lets a bot buy two studs then bust).",
    )
    r.add_argument(
        "--overbid-pace-jitter",
        type=float,
        default=_OVERBID_PACE_JITTER,
        help="Per-seat spread of the pace cap as a fraction of --overbid-pace. 0 gives every bot "
        "the same ceiling, which the hero can simply outbid by $1.",
    )
    r.add_argument(
        "--market-adp-jitter",
        type=float,
        default=12.0,
        help="Flush seats nominate off a shared noisy-ADP board with this jitter (the realistic "
        "nomination model, Run P). Pass 0 or omit --market-adp-jitter for value-weighted nom.",
    )
    r.add_argument(
        "--nominator-probe",
        default=None,
        metavar="BID_MODEL",
        help="Race NOMINATION policies instead of bid models, with every contestant bidding "
        "BID_MODEL (e.g. overbid_noramp). Contestants become control/off_pos/gap/gap_off. "
        "Requires --bot-prices espn.",
    )
    r.add_argument(
        "--n-patient",
        type=int,
        default=None,
        help="How many bot seats are conservative hoarders, spread evenly. Omit for the historical "
        "every-5th rule (2 of 11 in a 12-team league). Use to sweep the aggressive/conservative "
        "mix.",
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
