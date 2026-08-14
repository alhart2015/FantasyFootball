"""Eye-test view: real rosters a bid model drafted, shown with the season each one actually had.

The bake-off reports rates ("23% title equity"). Rates hide whether a strategy is buying teams a
human would recognise as good. This runs the model through N drafts on the real engine, plays each
resulting league out as a full projected season (`simulate_seasons`), and prints one roster per
realized OUTCOME bucket — won the title, lost the final, missed the playoffs — as a starting lineup
with the price paid for every player.

Each sample also carries its roster's own distribution over MANY seasons, which is what separates
"a bad roster" from "a good roster that ran cold": a missed-playoffs sample with 20% title equity
is variance, one with 2% is a genuinely bad draft.

Run:
    python scripts/auction_sample_rosters.py --vorp-table ... --league-config ... \
        --strategy balanced --seat 6 --drafts 40 --field overbidder --overbid 0.2
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.simulation import _simulate_to_state
from projections.draft.assistant.auction.tournament_cli import _load_pool
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_projection import (
    REG_WEEKS,
    SeasonOutcomes,
    project_draft,
    simulate_seasons,
)
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)
from projections.draft.league_config import LeagueConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script imports

from auction_field_bakeoff import (
    CONTESTANTS,
    FIELDS,
    N_PATIENT_HELP,
    build_field,
    format_n_patient,
)
from inspect_auction_drafts import _lineup, _RosterItem, _slot_plan

# One sample per bucket, in the order they are printed.
_BUCKETS: tuple[str, ...] = ("won championship", "lost championship", "missed playoffs")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample rosters by realized season outcome.")
    p.add_argument("--vorp-table", type=Path, required=True)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--strategy", default="balanced", choices=sorted(CONTESTANTS))
    p.add_argument("--seat", type=int, default=6, help="hero seat (1-based).")
    p.add_argument("--drafts", type=int, default=40)
    p.add_argument(
        "--seasons-per-draft",
        type=int,
        default=12,
        help="realized seasons played per draft (each is a candidate sample).",
    )
    p.add_argument(
        "--equity-sims",
        type=int,
        default=300,
        help="sims used for the chosen roster's own outcome distribution.",
    )
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--bot-prices", choices=("espn", "model"), default="espn")
    p.add_argument("--field", choices=FIELDS, default="overbidder")
    p.add_argument(
        "--n-patient",
        type=int,
        default=None,
        help=N_PATIENT_HELP,
    )
    p.add_argument("--overbid", type=float, default=0.20)
    p.add_argument("--market-adp-jitter", type=float, default=12.0)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    return p.parse_args(argv)


def _print_roster(
    roster: list[_RosterItem],
    plan: list[tuple[str, frozenset[str]]],
    n_bench: int,
    name_by_id: dict[str, str],
    pts_by_id: dict[str, float],
    fair_by_id: dict[str, float],
    budget: int,
) -> None:
    rows = _lineup(roster, pts_by_id, plan, n_bench)
    spent = sum(pr for _g, _p, pr in roster)
    print(f"    {'slot':<6}{'player':<24}{'pos':<5}{'paid':>6}{'fair':>7}{'proj pts':>10}")
    for label, item in rows:
        if item is None:
            print(f"    {label:<6}{'(empty)':<24}")
            continue
        gid, pos, price = item
        print(
            f"    {label:<6}{name_by_id.get(str(gid), str(gid))[:23]:<24}{pos:<5}"
            f"{'$' + str(price):>6}{'$' + str(int(fair_by_id.get(str(gid), 0))):>7}"
            f"{pts_by_id.get(str(gid), 0.0):>10.0f}"
        )
    print(f"    spent ${spent} of ${budget}  ->  ${budget - spent} unspent")


def main(argv: list[str] | None = None) -> int:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    pool = attach_is_rookie(
        _load_pool(args.vorp_table), season=args.season, data_root=args.data_root
    )
    availability = load_store_availability(pool, season=args.season, data_root=args.data_root)
    params = VarianceParams.load()
    baseline = generate_auction_values(pool, config)
    bot_dollars: pd.Series | None = None
    if args.bot_prices == "espn" and has_usable_espn_prices(pool):
        bot_dollars = espn_anchored_bot_prices(pool, config, model_values=baseline)
    field = build_field(
        args.field, args.overbid, n_bots=config.n_teams - 1, n_patient=args.n_patient
    )
    name_by_id = dict(zip(pool["gsis_id"].astype(str), pool["full_name"], strict=True))
    pts_by_id = dict(zip(pool["gsis_id"].astype(str), pool["season_mean_fpts"], strict=True))
    fair_by_id = dict(
        zip(baseline["gsis_id"].astype(str), baseline["auction_dollars"], strict=True)
    )
    plan, n_bench = _slot_plan(config.roster_slots)
    hero = args.seat

    # draft index -> (roster, league, outcomes); one candidate sample per (draft, sim).
    found: dict[str, tuple[int, int, list[_RosterItem], dict[int, list[str]], SeasonOutcomes]] = {}
    for d in range(args.drafts):
        state = _simulate_to_state(
            CONTESTANTS[args.strategy],
            hero,
            pool,
            config,
            baseline_dollars=baseline,
            price_jitter=DEFAULT_PRICE_JITTER,
            rng=np.random.default_rng(args.base_seed + d),
            nomination_temp=1.0,
            bot_archetypes=field,
            bot_dollars=bot_dollars,
            market_adp_jitter=args.market_adp_jitter,
        )
        league = {s + 1: [g for (g, _p, _pr) in state.rosters[s]] for s in range(config.n_teams)}
        outcomes = simulate_seasons(
            league,
            pool,
            availability,
            params,
            league_config=config,
            n_sims=args.seasons_per_draft,
            rng=np.random.default_rng(500_000 + d),
        )
        # At most ONE bucket per draft. A single roster routinely produces all three outcomes
        # across its own season draws (variance, not construction), so filling every bucket from
        # one draft would print the same 13 players three times and show nothing about how the
        # rosters differ. Take the rarest still-missing outcome this draft can supply.
        labels = outcomes.outcome_labels(hero)
        missing = [b for b in _BUCKETS if b not in found]
        for bucket in missing:
            sims = [i for i, label in enumerate(labels) if label == bucket]
            if sims:
                found[bucket] = (d, sims[0], list(state.rosters[hero - 1]), league, outcomes)
                break
        if len(found) == len(_BUCKETS):
            break

    print(
        f"strategy={args.strategy} league={config.name} seat={hero}/{config.n_teams} "
        f"budget=${config.budget} | field={args.field} "
        f"n_patient={format_n_patient(args.n_patient)} "
        f"overbid={args.overbid} "
        f"market={args.bot_prices}"
    )
    col = None
    for bucket in _BUCKETS:
        if bucket not in found:
            print(f"\n### {bucket.upper()}: not seen in {args.drafts} drafts — none to show.")
            continue
        d, sim, roster, league, outcomes = found[bucket]
        col = outcomes.slots.index(hero)
        wins = int(outcomes.wins[sim, col])
        pf = float(outcomes.points_for[sim, col])
        seed = int(outcomes.seed[sim, col])
        champ = int(outcomes.champion[sim])
        print(f"\n{'=' * 74}\n### {bucket.upper()}  (draft #{d}, season #{sim})")
        print(
            f"  realized: {wins}-{len(REG_WEEKS) - wins}, {pf:.0f} pts, seed {seed}"
            f"{'' if champ == hero else f' (title went to seat {champ})'}"
        )
        equity = project_draft(
            league,
            pool,
            availability,
            params,
            league_config=config,
            n_sims=args.equity_sims,
            rng=np.random.default_rng(900_000 + d),
        )[hero]
        print(
            f"  this roster over {args.equity_sims} seasons: "
            f"{equity.reg_win_pct:.0%} reg-win, {equity.make_playoffs_pct:.0%} playoffs, "
            f"{equity.champ_pct:.0%} title, {equity.mean_points:.0f} mean pts"
        )
        _print_roster(roster, plan, n_bench, name_by_id, pts_by_id, fair_by_id, config.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
