"""Pick-by-pick trace of one auction: who bought whom, for how much, and where the money went.

The roster views show WHAT a strategy ended up with. They cannot show WHY a player was cheap —
that lives in the order lots came up and how much money was still in the room when they did. This
prints the full draft in pick order (price vs our value vs ESPN's price, the winning seat and its
archetype, and the room's remaining budget after each award), then a per-seat and per-archetype
spend summary.

Use it when a price fails the eye test: a stud clearing under value is either a genuine market
inefficiency or an artifact of the room being broke, and only the trace distinguishes them.

Run:
    python scripts/auction_draft_trace.py --vorp-table ... --league-config ... \
        --strategy static --seat 6 --draft 2 --highlight "Jahmyr Gibbs,Bijan Robinson"
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER, assign_bot_archetypes
from projections.draft.assistant.auction.simulation import PickRecord, _simulate_to_state
from projections.draft.assistant.auction.tournament_cli import _load_pool
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)
from projections.draft.league_config import LeagueConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

from auction_field_bakeoff import CONTESTANTS, FIELDS, N_PATIENT_HELP, build_field


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick-by-pick trace of one simulated auction.")
    p.add_argument("--vorp-table", type=Path, required=True)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--strategy", default="static", choices=sorted(CONTESTANTS))
    p.add_argument("--seat", type=int, default=6, help="hero seat (1-based).")
    p.add_argument("--draft", type=int, default=0, help="draft index == RNG seed offset.")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--picks", type=int, default=60, help="picks to print (default 60).")
    p.add_argument("--highlight", default="", help="comma-separated names to flag with '<<'.")
    p.add_argument("--bot-prices", choices=("espn", "model"), default="espn")
    p.add_argument("--field", choices=FIELDS, default="overbidder")
    p.add_argument(
        "--n-patient",
        type=int,
        default=None,
        help=N_PATIENT_HELP,
    )
    p.add_argument("--overbid", type=float, default=0.20)
    p.add_argument("--overbid-pace", type=float, default=4.5)
    p.add_argument("--market-adp-jitter", type=float, default=12.0)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    pool = attach_is_rookie(
        _load_pool(args.vorp_table), season=args.season, data_root=args.data_root
    )
    baseline = generate_auction_values(pool, config)
    bot_dollars: pd.Series | None = None
    if args.bot_prices == "espn" and has_usable_espn_prices(pool):
        bot_dollars = espn_anchored_bot_prices(pool, config, model_values=baseline)
    field = build_field(
        args.field,
        args.overbid,
        args.overbid_pace,
        n_bots=config.n_teams - 1,
        n_patient=args.n_patient,
    )
    name_by_id = dict(zip(pool["gsis_id"].astype(str), pool["full_name"], strict=True))
    espn_by_id = dict(zip(pool["gsis_id"].astype(str), pool["espn_auction_dollars"], strict=True))

    hero0 = args.seat - 1
    bot_seats = [s for s in range(config.n_teams) if s != hero0]
    assigned = assign_bot_archetypes(len(bot_seats), field)
    label: dict[int, str] = {hero0: f"HERO({args.strategy})"}
    for i, s in enumerate(bot_seats):
        kind = "OVERBID" if type(assigned[i]).__name__ == "BalancedBot" else "PATIENT"
        label[s] = f"{kind}{s + 1}"

    trace: list[PickRecord] = []
    _simulate_to_state(
        CONTESTANTS[args.strategy],
        args.seat,
        pool,
        config,
        baseline_dollars=baseline,
        price_jitter=DEFAULT_PRICE_JITTER,
        rng=np.random.default_rng(args.base_seed + args.draft),
        nomination_temp=1.0,
        bot_archetypes=field,
        bot_dollars=bot_dollars,
        market_adp_jitter=args.market_adp_jitter,
        trace=trace,
    )
    flags = {n.strip().lower() for n in args.highlight.split(",") if n.strip()}
    total = config.total_budget

    print(
        f"hero={args.strategy} seat {args.seat}/{config.n_teams} draft #{args.draft} | "
        f"field={args.field} n_patient={args.n_patient} "
        f"overbid={args.overbid} pace={args.overbid_pace} "
        f"market={args.bot_prices}"
    )
    print(
        f"\n{'#':>3} {'player':<22}{'pos':<4}{'paid':>5}{'val':>5}{'espn':>6}  "
        f"{'winner':<16}{'room $':>8}"
    )
    for r in trace[: args.picks]:
        nm = str(name_by_id.get(r.gsis_id, r.gsis_id))
        e = espn_by_id.get(r.gsis_id)
        estr = "—" if e is None or pd.isna(e) else f"${int(e)}"
        mark = " <<" if nm.lower() in flags else ""
        print(
            f"{r.pick:>3} {nm[:21]:<22}{r.position:<4}{'$' + str(r.price):>5}"
            f"{'$' + str(int(r.value)):>5}{estr:>6}  {label[r.winner_seat]:<16}"
            f"{r.room_budget:>8}{mark}"
        )

    spent_by_seat: defaultdict[int, int] = defaultdict(int)
    bought_by_seat: defaultdict[int, int] = defaultdict(int)
    for r in trace:
        spent_by_seat[r.winner_seat] += r.price
        bought_by_seat[r.winner_seat] += 1
    print(f"\n{'seat':<18}{'spent':>8}{'players':>9}{'$/player':>10}")
    for s in sorted(spent_by_seat, key=lambda s: -spent_by_seat[s]):
        n = bought_by_seat[s]
        print(f"{label[s]:<18}{'$' + str(spent_by_seat[s]):>8}{n:>9}{spent_by_seat[s] / n:>10.1f}")

    print(f"\nroom budget remaining after pick N (of ${total}):")
    for cut in (12, 24, 36, 48, 60, 90, 120, len(trace)):
        if cut <= len(trace):
            rec = trace[cut - 1]
            print(f"  after {cut:>3}: ${rec.room_budget:>5} ({rec.room_budget / total:>4.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
