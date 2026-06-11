"""Simulate one full snake draft (spec §3.3).

The hero seat runs a `DraftStrategy` (re-asked from the current `DraftState` at
each of its picks); every other seat runs the noisy-ADP `bot_pick`. One seeded
RNG drives all bot noise, so same seed + same strategy => identical hero roster.
Assumes a validated pool (size + ADP signal checked at the tournament entry).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import bot_pick
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import DraftStrategy
from projections.draft.league_config import LeagueConfig
from projections.schemas import GsisId, Position, validate_gsis_id


def _draft_picks(
    strategy: DraftStrategy,
    my_slot: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    adp_jitter: float,
    rng: np.random.Generator,
) -> list[GsisId]:
    """Run the draft; return every pick's gsis_id in absolute pick order."""
    pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"], strict=True)}
    n_teams = config.n_teams
    total_picks = n_teams * config.roster_size

    drafted: list[GsisId] = []
    drafted_set: set[GsisId] = set()
    my_roster: list[Position] = []

    for pick_number in range(1, total_picks + 1):
        if slot_for(pick_number, n_teams) == my_slot:
            state = DraftState(
                my_slot=my_slot,
                n_teams=n_teams,
                rounds=config.roster_size,
                picks=tuple(drafted),
                my_roster=tuple(my_roster),
            )
            rec = strategy.recommend(state, pool, config)
            if rec.empty:
                raise ValueError(
                    f"strategy returned no eligible pick at pick {pick_number}; "
                    "pool too small or fully ineligible (should be caught upstream)"
                )
            gid = validate_gsis_id(str(rec.iloc[0]["gsis_id"]))
            if gid in drafted_set:
                raise ValueError(
                    f"strategy returned already-drafted player {gid!r} at pick {pick_number}"
                )
            my_roster.append(Position(pos_by_id[gid]))
        else:
            available = pool[~pool["gsis_id"].isin(drafted_set)]
            gid = bot_pick(available, rng, adp_jitter=adp_jitter)
        drafted.append(gid)
        drafted_set.add(gid)

    return drafted


def simulate_draft(
    strategy: DraftStrategy,
    my_slot: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    adp_jitter: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run one full draft; return the hero's drafted rows (a sub-frame of `pool`)."""
    all_picks = _draft_picks(strategy, my_slot, pool, config, adp_jitter=adp_jitter, rng=rng)
    mine = {pick for i, pick in enumerate(all_picks) if slot_for(i + 1, config.n_teams) == my_slot}
    return pool[pool["gsis_id"].isin(mine)].copy()
