"""What a bid model actually DOES at the table: the average spend profile over many drafts.

The bake-off (`auction_field_bakeoff.py`) says which bid model wins under a given opponent field.
It does not say what the winner is *doing* — and a strategy you cannot execute by hand at a live
auction is worthless. This script runs the winner (and any comparison models) through N drafts on
the real engine and averages the shape of the resulting roster:

  * dollars by position, and by price rank (the priciest buy, the 2nd, ...) — the budget curve
  * how many players clear each price band, and how much budget is left unspent
  * price paid vs the model's own fair value — is the model buying at a premium or a discount
  * the positional count of the roster

No season Monte-Carlo runs here (the bake-off already graded the outcomes), so it is fast.

Run:
    python scripts/auction_spend_profile.py --vorp-table ... --league-config ... \
        --strategies sr_g0.2_c2,balanced --seat 6 --drafts 60 --field overbidder --overbid 0.2
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.simulation import _simulate_to_state
from projections.draft.assistant.auction.tournament_cli import _load_pool
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import Position

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

from auction_field_bakeoff import CONTESTANTS, FIELDS, build_field

_BANDS: tuple[tuple[str, int, int], ...] = (
    ("$40+", 40, 10**9),
    ("$25-39", 25, 39),
    ("$15-24", 15, 24),
    ("$5-14", 5, 14),
    ("$1-4", 1, 4),
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Average spend profile for auction bid models.")
    p.add_argument("--vorp-table", type=Path, required=True)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument(
        "--strategies",
        default="sr_g0.2_c2,balanced",
        help=f"comma-separated; any of: {','.join(sorted(CONTESTANTS))}",
    )
    p.add_argument("--seat", type=int, default=6, help="hero seat (1-based).")
    p.add_argument("--drafts", type=int, default=60)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--bot-prices", choices=("espn", "model"), default="espn")
    p.add_argument("--field", choices=FIELDS, default="overbidder")
    p.add_argument("--overbid", type=float, default=0.20)
    p.add_argument("--market-adp-jitter", type=float, default=12.0)
    p.add_argument("--top-names", type=int, default=12, help="most-bought players to list.")
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
    field = build_field(args.field, args.overbid, n_bots=config.n_teams - 1)
    name_by_id = dict(zip(pool["gsis_id"].astype(str), pool["full_name"], strict=True))
    # generate_auction_values returns a reset-index frame; key fair value by gsis_id the way the
    # engine does (simulation.py sets the same index) or every lookup silently misses and reads $0.
    fair_by_id = dict(
        zip(baseline["gsis_id"].astype(str), baseline["auction_dollars"], strict=True)
    )

    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [n for n in names if n not in CONTESTANTS]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {sorted(CONTESTANTS)}")

    print(
        f"league={config.name} seat={args.seat}/{config.n_teams} budget=${config.budget} "
        f"roster={config.roster_size} | field={args.field} overbid={args.overbid} "
        f"market={args.bot_prices} | {args.drafts} drafts"
    )
    for name in names:
        rows: list[list[tuple[str, str, int]]] = []
        for s in range(args.drafts):
            state = _simulate_to_state(
                CONTESTANTS[name],
                args.seat,
                pool,
                config,
                baseline_dollars=baseline,
                price_jitter=DEFAULT_PRICE_JITTER,
                rng=np.random.default_rng(args.base_seed + s),
                nomination_temp=1.0,
                bot_archetypes=field,
                bot_dollars=bot_dollars,
                market_adp_jitter=args.market_adp_jitter,
            )
            rows.append(list(state.rosters[args.seat - 1]))

        n = float(len(rows))
        by_pos_dollars: defaultdict[str, float] = defaultdict(float)
        by_pos_count: defaultdict[str, float] = defaultdict(float)
        band_count: defaultdict[str, float] = defaultdict(float)
        rank_dollars: defaultdict[int, float] = defaultdict(float)
        bought: Counter[str] = Counter()
        spend_by_name: defaultdict[str, float] = defaultdict(float)
        total_spent = paid_on_priced = fair_on_priced = 0.0
        for roster in rows:
            for gid, pos, price in roster:
                by_pos_dollars[pos] += price
                by_pos_count[pos] += 1
                total_spent += price
                bought[str(gid)] += 1
                spend_by_name[str(gid)] += price
                for label, lo, hi in _BANDS:
                    if lo <= price <= hi:
                        band_count[label] += 1
                        break
                fair = float(fair_by_id.get(str(gid), 0.0))
                if fair > 1.0:  # $1 filler carries no informative fair value
                    paid_on_priced += price
                    fair_on_priced += fair
            for i, (_g, _p, price) in enumerate(sorted(roster, key=lambda t: -t[2])):
                rank_dollars[i] += price

        left = config.budget - total_spent / n
        print(f"\n=== {name} ===")
        print(f"  spent ${total_spent / n:.0f} of ${config.budget}, left ${left:.0f}")
        if fair_on_priced > 0:
            print(
                f"  paid/fair on non-$1 buys: {paid_on_priced / fair_on_priced:.2f}x "
                f"(>1 = buying at a premium)"
            )
        order = [p.value for p in Position if p.value in by_pos_dollars]
        print("  by position:  " + "  ".join(f"{p} ${by_pos_dollars[p] / n:.0f}" for p in order))
        print("  roster shape: " + "  ".join(f"{p} {by_pos_count[p] / n:.1f}" for p in order))
        print(
            "  price bands (players/draft): "
            + "  ".join(f"{lab} {band_count[lab] / n:.1f}" for lab, _lo, _hi in _BANDS)
        )
        curve = "  ".join(
            f"#{i + 1} ${rank_dollars[i] / n:.0f}" for i in range(min(8, config.roster_size))
        )
        print(f"  budget curve (priciest first): {curve}")
        top = bought.most_common(args.top_names)
        print(f"  most-bought ({args.top_names}):")
        for gid, cnt in top:
            print(
                f"     {name_by_id.get(gid, gid)[:24]:<24} {cnt / n * 100:>5.0f}% of drafts  "
                f"avg ${spend_by_name[gid] / max(cnt, 1):>5.1f}  "
                f"(fair ${fair_by_id.get(gid, 0):.0f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
