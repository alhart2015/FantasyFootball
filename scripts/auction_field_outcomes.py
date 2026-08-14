"""Per-archetype outcomes for the OPPONENTS, in the same drafts the bake-off scored the hero in.

`run_auction_tournament` keeps only `proj[my_seat]` — every bot seat's projection is computed and
then discarded. So "how did the over-bidders actually do while `static` was winning?" cannot be
answered from the stored chunks; it has to be recomputed. This does that, reproducing the
bake-off's conditions exactly (same seed derivation, field, nomination model, and market) so the
hero row here can be checked against the chunk it is meant to match.

Reports each seat's metrics grouped by the archetype seated there, which the seat-averaged bake-off
tables cannot show: a room of over-bidders is not uniform, and the patient seats are a different
population from the aggressive ones.

Run:
    python scripts/auction_field_outcomes.py --vorp-table ... --league-config ... \
        --strategy static --seats 1-12 --seeds 20 --n-sims 300
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
from projections.draft.assistant.auction.simulation import _simulate_to_state
from projections.draft.assistant.auction.tournament import METRICS
from projections.draft.assistant.auction.tournament_cli import _load_tournament_inputs
from projections.draft.assistant.league_projection import project_draft
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

from auction_field_bakeoff import CONTESTANTS, FIELDS, build_field

_SNAKE_SUBSTREAM = 20260619  # must match tournament.py, or the bot field is not the same draw


def _parse_seats(spec: str, n_teams: int) -> list[int]:
    if spec == "all":
        return list(range(1, n_teams + 1))
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",")]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Opponent-archetype outcomes for a given hero.")
    p.add_argument("--vorp-table", type=Path, required=True)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--strategy", default="static", choices=sorted(CONTESTANTS))
    p.add_argument("--seats", default="all", help="'all', '1-12', or '1,6,12'.")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-sims", type=int, default=300)
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (matches the bake-off).")
    p.add_argument("--bot-prices", choices=("espn", "model"), default="espn")
    p.add_argument("--field", choices=FIELDS, default="overbidder")
    p.add_argument(
        "--n-patient",
        type=int,
        default=None,
        help="Conservative hoarder seats, spread evenly (matches auction_field_bakeoff). "
        "Omit for the historical every-5th rule. Set it to make this diagnostic describe the "
        "same room as a swept field-mix cell.",
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
    pool, config, availability, params = _load_tournament_inputs(
        args.vorp_table, args.league_config, season=args.season, data_root=args.data_root
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
    hero_strategy = CONTESTANTS[args.strategy]
    seats = _parse_seats(args.seats, config.n_teams)
    season_base_seed = args.seed + 1_000_000  # tournament.py's default derivation

    # label -> metric -> list of per-(seat, seed) values
    acc: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    hero_label = f"HERO ({args.strategy})"
    for my_seat in seats:
        bot_seats = [s for s in range(config.n_teams) if s != my_seat - 1]  # 0-based
        assigned = assign_bot_archetypes(len(bot_seats), field)
        label_by_seat0 = {
            s: type(assigned[i]).__name__.replace("Bot", "") for i, s in enumerate(bot_seats)
        }
        for s in range(args.seeds):
            state = _simulate_to_state(
                hero_strategy,
                my_seat,
                pool,
                config,
                baseline_dollars=baseline,
                price_jitter=DEFAULT_PRICE_JITTER,
                rng=np.random.default_rng(args.seed + s),
                snake_rng=np.random.default_rng([args.seed + s, _SNAKE_SUBSTREAM]),
                nomination_temp=1.0,
                bot_archetypes=field,
                bot_dollars=bot_dollars,
                market_adp_jitter=args.market_adp_jitter,
            )
            league = {
                seat + 1: [g for (g, _p, _pr) in state.rosters[seat]]
                for seat in range(config.n_teams)
            }
            proj = project_draft(
                league,
                pool,
                availability,
                params,
                league_config=config,
                n_sims=args.n_sims,
                rng=np.random.default_rng(season_base_seed + s),
            )
            for seat0 in range(config.n_teams):
                label = hero_label if seat0 == my_seat - 1 else label_by_seat0[seat0]
                for m in METRICS:
                    acc[label][m].append(float(getattr(proj[seat0 + 1], m)))

    print(
        f"hero={args.strategy} field={args.field} overbid={args.overbid} pace={args.overbid_pace} "
        f"market={args.bot_prices} | seats {seats[0]}-{seats[-1]}, {args.seeds} seeds x "
        f"{args.n_sims} sims"
    )
    print(f"\n{'who':<22}{'n(seat,seed)':>13}" + "".join(f"{m:>18}" for m in METRICS))
    order = [hero_label, *sorted(k for k in acc if k != hero_label)]
    for label in order:
        vals = acc[label]
        n = len(vals[METRICS[0]])
        print(f"{label:<22}{n:>13}" + "".join(f"{np.mean(vals[m]):>18.4f}" for m in METRICS))

    # Zero-sum identity: across a 12-seat league every game has one winner and every season one
    # champion, so the seat-weighted means must be 0.5 and 1/n_teams. A drift here means the
    # per-seat weighting is wrong, not that a strategy found free wins.
    tot_n = sum(len(acc[k][METRICS[0]]) for k in acc)
    for metric, expected in (("reg_win_pct", 0.5), ("champ_pct", 1.0 / config.n_teams)):
        got = sum(float(np.sum(acc[k][metric])) for k in acc) / tot_n
        print(f"\ncheck: league-wide mean {metric} = {got:.4f} (must be {expected:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
