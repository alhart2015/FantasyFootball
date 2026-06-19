"""Inspect individual auction drafts for one hero bid-model against the ESPN-anchored bot field.

Runs N drafts (the real `_simulate_to_state` engine + the same ESPN bot pricing the bake-off uses),
scores each league with `project_draft`, and prints the hero's BEST and WORST drafts (ranked by
projected roster points) — full rosters with dollars paid + remaining budget + projected finish —
each alongside the strongest and weakest opposing bot in that same draft.

A reusable "what does this strategy's draft actually look like, and how does the roster stack up
against the field" view. Defaults to patient_deep, 2026, 12-team half-PPR, seat 6.

Run (from the repo root):
    python scripts/inspect_auction_drafts.py [--strategy patient_deep] [--season 2026]
        [--scoring half] [--size 12] [--seat 6] [--drafts 20] [--n-sims 200] [--bot-prices espn]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.auction.market import assign_bot_archetypes
from projections.draft.assistant.auction.simulation import AuctionState, _simulate_to_state
from projections.draft.assistant.auction.tournament_cli import _MODELS, _REALISTIC_FIELD
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_projection import SeatProjection, project_draft
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.presets import get_preset
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)
from projections.schemas import _PYARROW_STR, VorpTableSchema


def _load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def main(argv: list[str] | None = None) -> int:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description="Inspect a bid-model's best/worst auction drafts.")
    p.add_argument("--strategy", default="patient_deep", choices=sorted(_MODELS))
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--scoring", default="half", choices=("half", "ppr", "std"))
    p.add_argument("--size", type=int, default=12, choices=(10, 12, 16))
    p.add_argument("--seat", type=int, default=6, help="hero seat (1-based).")
    p.add_argument("--drafts", type=int, default=20, help="number of drafts to run.")
    p.add_argument("--n-sims", type=int, default=200, help="season MC sims per draft (scoring).")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--bot-prices", choices=("espn", "model"), default="espn")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    preset = get_preset(args.scoring, args.size, season=args.season)
    pool = _load_pool(preset.table_path)
    config = preset.league_config
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)
    availability = load_store_availability(pool, season=args.season, data_root=args.data_root)
    params = VarianceParams.load()
    baseline = generate_auction_values(pool, config)
    bot_dollars: pd.Series | None = None
    if args.bot_prices == "espn" and has_usable_espn_prices(pool):
        bot_dollars = espn_anchored_bot_prices(pool, config)

    hero = _MODELS[args.strategy]
    seat0, n, budget = args.seat - 1, config.n_teams, config.budget
    bot_seats = [s for s in range(n) if s != seat0]
    assigned = assign_bot_archetypes(len(bot_seats), _REALISTIC_FIELD)
    arch = {s: type(assigned[i]).__name__.replace("Bot", "") for i, s in enumerate(bot_seats)}
    arch[seat0] = args.strategy.upper()
    name_by_id = dict(zip(pool["gsis_id"].astype(str), pool["full_name"], strict=True))

    sims: list[tuple[AuctionState, dict[int, SeatProjection]]] = []
    for s in range(args.drafts):
        state = _simulate_to_state(
            hero,
            args.seat,
            pool,
            config,
            baseline_dollars=baseline,
            price_jitter=0.0,
            rng=np.random.default_rng(args.base_seed + s),
            nomination_temp=1.0,
            bot_archetypes=_REALISTIC_FIELD,
            bot_dollars=bot_dollars,
        )
        league = {seat + 1: [g for (g, _, _) in state.rosters[seat]] for seat in range(n)}
        proj = project_draft(
            league,
            pool,
            availability,
            params,
            league_config=config,
            n_sims=args.n_sims,
            rng=np.random.default_rng(10_000 + s),
        )
        sims.append((state, proj))

    hero_pts = [proj[args.seat].mean_points for _, proj in sims]
    best_i = max(range(args.drafts), key=lambda i: hero_pts[i])
    worst_i = min(range(args.drafts), key=lambda i: hero_pts[i])

    def show(state: AuctionState, proj: dict[int, SeatProjection], seat: int, tag: str) -> None:
        roster = sorted(state.rosters[seat], key=lambda t: -t[2])  # priciest first
        spent, rem, sp = sum(pr for _, _, pr in roster), state.budgets[seat], proj[seat + 1]
        print(
            f"\n  {tag} - {arch[seat]} (seat {seat + 1}): proj pts {sp.mean_points:.0f}, "
            f"playoff {sp.make_playoffs_pct:.2f} | spent ${spent}/{budget}, left ${rem}"
        )
        for g, pos, pr in roster:
            print(f"     ${pr:>3}  {pos:<4}{name_by_id.get(str(g), g)}")

    def table(i: int, header: str) -> None:
        state, proj = sims[i]
        order = sorted(range(n), key=lambda s: -proj[s + 1].mean_points)
        rank = {s: r + 1 for r, s in enumerate(order)}
        others = [s for s in order if s != seat0]
        print(f"\n{'=' * 74}\n{header}  (draft seed {args.base_seed + i})")
        print(
            "finish by proj pts: "
            + " > ".join(f"{arch[s]}#{rank[s]}({proj[s + 1].mean_points:.0f})" for s in order)
        )
        show(state, proj, seat0, f"{args.strategy} finished #{rank[seat0]}")
        show(state, proj, others[0], f"top bot, finished #{rank[others[0]]}")
        show(state, proj, others[-1], f"bottom bot, finished #{rank[others[-1]]}")

    table(best_i, f"BEST {args.strategy} draft")
    table(worst_i, f"WORST {args.strategy} draft")
    print(
        f"\n{args.strategy} proj-pts across {args.drafts} drafts: best {max(hero_pts):.0f}, "
        f"worst {min(hero_pts):.0f}, mean {sum(hero_pts) / len(hero_pts):.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
