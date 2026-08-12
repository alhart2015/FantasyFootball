"""How wide is a single season? Distribution of realized 13-game records for a bid strategy.

The bake-off reports a mean win rate. A mean hides the thing that actually decides a fantasy
season: one 13-game sample is extremely noisy, so a better roster loses to a worse one constantly.
This draws many drafts, plays each out over many seasons, and histograms the hero's realized win
totals — the spread a single real season is drawn from.

Separates the two sources of variance:
  * ACROSS DRAFTS + SEASONS — the full spread you face going in (which players you land AND how
    the season breaks).
  * WITHIN ONE ROSTER — the same 13 players replayed, isolating pure season luck.

Racing two strategies shows how much their record distributions OVERLAP, which is the honest
picture of how much a strategy edge is worth over one year.

Run:
    python scripts/auction_record_spread.py --vorp-table ... --league-config ... \
        --strategies overbid_noramp,patient --drafts 30 --seasons-per-draft 200
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
from projections.draft.assistant.auction.tournament_cli import _load_tournament_inputs
from projections.draft.assistant.league_projection import REG_WEEKS, simulate_seasons
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

from auction_field_bakeoff import CONTESTANTS, FIELDS, build_field

_N_GAMES = len(REG_WEEKS)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realized-record distribution for bid strategies.")
    p.add_argument("--vorp-table", type=Path, required=True)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--strategies", default="overbid_noramp,patient")
    p.add_argument("--seat", type=int, default=6)
    p.add_argument("--drafts", type=int, default=30)
    p.add_argument("--seasons-per-draft", type=int, default=200)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--bot-prices", choices=("espn", "model"), default="espn")
    p.add_argument("--field", choices=FIELDS, default="overbidder")
    p.add_argument("--overbid", type=float, default=0.20)
    p.add_argument("--market-adp-jitter", type=float, default=12.0)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    return p.parse_args(argv)


def _histogram(wins: np.ndarray, width: int = 46) -> list[str]:
    """Text histogram of win totals 0..13 with per-bucket share."""
    counts = np.bincount(wins.astype(int), minlength=_N_GAMES + 1)[: _N_GAMES + 1]
    share = counts / max(1, counts.sum())
    peak = share.max() if share.size else 1.0
    lines = []
    for w, (c, sh) in enumerate(zip(counts, share, strict=True)):
        if c == 0 and (w < 2 or w > _N_GAMES - 1):
            continue
        bar = "#" * round(width * (sh / peak)) if peak > 0 else ""
        lines.append(f"  {w:>2}-{_N_GAMES - w:<2} {sh * 100:>5.1f}%  {bar}")
    return lines


def _describe(wins: np.ndarray) -> str:
    qs = np.percentile(wins, [5, 25, 50, 75, 95])
    return (
        f"mean {wins.mean():.1f}-{_N_GAMES - wins.mean():.1f} "
        f"({wins.mean() / _N_GAMES:.1%})  sd {wins.std():.2f} wins  "
        f"p5 {qs[0]:.0f}  p25 {qs[1]:.0f}  median {qs[2]:.0f}  p75 {qs[3]:.0f}  p95 {qs[4]:.0f}"
    )


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
    field = build_field(args.field, args.overbid, n_bots=config.n_teams - 1)
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [n for n in names if n not in CONTESTANTS]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {sorted(CONTESTANTS)}")

    print(
        f"{config.n_teams}-team, seat {args.seat}, {_N_GAMES}-game regular season | "
        f"{args.drafts} drafts x {args.seasons_per_draft} seasons = "
        f"{args.drafts * args.seasons_per_draft} seasons per strategy"
    )
    all_wins: dict[str, np.ndarray] = {}
    per_draft_means: dict[str, np.ndarray] = {}
    single_roster: dict[str, np.ndarray] = {}
    for name in names:
        buckets: list[np.ndarray] = []
        for d in range(args.drafts):
            state = _simulate_to_state(
                CONTESTANTS[name],
                args.seat,
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
            league = {
                s + 1: [g for (g, _p, _pr) in state.rosters[s]] for s in range(config.n_teams)
            }
            out = simulate_seasons(
                league,
                pool,
                availability,
                params,
                league_config=config,
                n_sims=args.seasons_per_draft,
                rng=np.random.default_rng(700_000 + d),
            )
            buckets.append(out.wins[:, out.slots.index(args.seat)])
        all_wins[name] = np.concatenate(buckets)
        per_draft_means[name] = np.array([b.mean() for b in buckets])
        # the draft whose expected win rate is closest to this strategy's own average
        med = int(np.argmin(np.abs(per_draft_means[name] - per_draft_means[name].mean())))
        single_roster[name] = buckets[med]

    for name in names:
        print(f"\n=== {name} — ALL drafts x ALL seasons ===")
        print(f"  {_describe(all_wins[name])}")
        print("\n".join(_histogram(all_wins[name])))
        print(f"\n  --- ONE typical roster replayed {args.seasons_per_draft} seasons ---")
        print(f"  {_describe(single_roster[name])}")
        print(
            f"  roster-to-roster spread (expected win% by draft): "
            f"{per_draft_means[name].min() / _N_GAMES:.1%} to "
            f"{per_draft_means[name].max() / _N_GAMES:.1%}"
        )

    if len(names) == 2:
        a, b = names
        wa, wb = all_wins[a], all_wins[b]
        # P(a's season beats b's season) with ties split — how often the better plan actually shows
        # up as the better record in ONE year.
        pa = float((wa[:, None] > wb[None, : min(len(wb), 4000)]).mean())
        pt = float((wa[:, None] == wb[None, : min(len(wb), 4000)]).mean())
        print(
            f"\nOne season, head to head: {a} finishes with a better record than {b} "
            f"{pa + pt / 2:.0%} of the time (ties split)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
