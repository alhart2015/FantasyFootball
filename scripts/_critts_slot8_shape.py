"""What the winning strategy actually drafts from one snake slot, round by round.

`draft_tournament.py compare` answers "which strategy scores more"; it does not say what
that strategy *does*. This runs the winner over many simulated drafts and reports, for each
of the hero's own picks, how often each position is taken and who the modal player is.

That distinction matters at the table. A VORP board reads as an unbroken wall of RB in this
league, because RB replacement level sits far below WR's. But every strategy here takes the
best *roster-eligible* player, so once RB1/RB2/FLEX are filled a fourth RB stops being
takeable and the board's apparent advice inverts. The per-round frequencies below are the
board *after* roster construction is applied -- which is the thing you can actually follow.

`--strategy` defaults to the board's own default. A shape read off one strategy while
drafting on another is worse than no table at all, so the header prints which one ran.

Usage:
    python scripts/_critts_slot8_shape.py --sims 300
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.live import MC_STRATEGIES, build_session_strategy
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.simulation import _draft_picks
from projections.draft.assistant.survival import default_sigma
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema

_LEAGUE = Path("data/leagues/critts_2025_2026")
_POOL = Path("data/vorp_2026/critts_half16_snake.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--my-slot", type=int, default=8)
    p.add_argument("--sims", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    # The shape is a property of the STRATEGY, not of the league -- reading a raw_vorp shape
    # while drafting on another strategy is worse than having no table at all.
    p.add_argument("--strategy", default="now_or_never_targeted")
    p.add_argument("--strategy-n-sims", type=int, default=100)
    p.add_argument("--season", type=int, default=2026)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = LeagueConfig.model_validate_json((_LEAGUE / "league_config.json").read_text())
    pool = pd.read_parquet(_POOL)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    names = pool.set_index("gsis_id")["full_name"].to_dict()
    validated = VorpTableSchema.validate(pool)

    availability = None
    if args.strategy in MC_STRATEGIES:
        availability = load_store_availability(
            validated, season=args.season, data_root=Path("data")
        )
    strategy = build_session_strategy(
        args.strategy,
        league=config,
        sigma=None,
        availability=availability,
        n_sims=args.strategy_n_sims,
        base_seed=0,
    )
    jitter = default_sigma(config.n_teams)

    # round index -> Counter(position), and -> Counter(player)
    pos_counts: dict[int, Counter[str]] = {}
    who_counts: dict[int, Counter[str]] = {}

    for s in range(args.sims):
        rng = np.random.default_rng(args.seed + s)
        picks = _draft_picks(strategy, args.my_slot, validated, config, adp_jitter=jitter, rng=rng)
        mine = [
            (i + 1, gid)
            for i, gid in enumerate(picks)
            if slot_for(i + 1, config.n_teams) == args.my_slot
        ]
        by_pos = validated.set_index("gsis_id")["position"]
        for rnd, (_, gid) in enumerate(mine, start=1):
            pos_counts.setdefault(rnd, Counter())[str(by_pos.get(gid, "?"))] += 1
            who_counts.setdefault(rnd, Counter())[names.get(gid, gid)] += 1

    print(f"{args.strategy} from slot {args.my_slot} of {config.n_teams} -- {args.sims} drafts")
    print(f"(strategy_n_sims={args.strategy_n_sims})\n")
    print(f"{'Rd':<4}{'Pick':<6}{'position mix':<44}modal player")
    for rnd in sorted(pos_counts):
        pc, wc = pos_counts[rnd], who_counts[rnd]
        total = sum(pc.values())
        mix = "  ".join(
            f"{pos} {100 * n / total:.0f}%" for pos, n in pc.most_common() if n / total >= 0.05
        )
        player, hits = wc.most_common(1)[0]
        pick_no = None
        # Recover the absolute pick number for this round from the snake formula.
        r0 = rnd - 1
        pick_no = (
            r0 * config.n_teams + args.my_slot
            if r0 % 2 == 0
            else r0 * config.n_teams + (config.n_teams - args.my_slot + 1)
        )
        print(f"{rnd:<4}{pick_no:<6}{mix:<44}{player} ({100 * hits / total:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
