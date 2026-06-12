"""Mixed-field constrained-bot draft for the H2H backtest (promoted from validated scratch sims).

Seat layout per spec: nn {2,6,10,14}, sv {4,8,12,16}, bots elsewhere; paired even seeds mirror
(swap nn<->sv) so seat exposure cancels when pooled over paired seeds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import bot_pick
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.state import DraftState
from projections.draft.assistant.strategy import DraftStrategy
from projections.draft.league_config import LeagueConfig
from projections.schemas import GsisId, Position, validate_gsis_id

_MINP = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
_MAXP = {"QB": 3, "RB": 6, "WR": 6, "TE": 3}


def seat_layout(seed: int) -> dict[int, str]:
    """Return a {seat: strategy_name} map for a 16-team snake draft.

    Odd seeds: now_or_never at {2,6,10,14}, season_value at {4,8,12,16}.
    Even seeds: mirrored — exposures cancel when summed over paired seeds.
    Remaining 8 seats are bots.
    """
    nn, sv = {2, 6, 10, 14}, {4, 8, 12, 16}
    if seed % 2 == 0:  # paired mirror
        nn, sv = sv, nn
    return {
        s: ("now_or_never" if s in nn else "season_value" if s in sv else "bot")
        for s in range(1, 17)
    }


def _bot_eligible(counts: dict[str, int], picks_left: int) -> set[str]:
    """Return the set of positions the ADP bot may select given roster constraints."""
    deficit = {p: max(0, _MINP[p] - counts.get(p, 0)) for p in _MINP}
    if picks_left <= sum(deficit.values()):
        return {p for p in _MINP if deficit[p] > 0}
    return {p for p in _MINP if counts.get(p, 0) < _MAXP[p]}


def draft_mixed_field(
    seat_strategies: dict[int, DraftStrategy | None],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    rng: np.random.Generator,
    jitter: float,
) -> dict[int, list[str]]:
    """Run a full snake draft and return {seat: [gsis_id, ...]} rosters.

    seat_strategies: seat -> DraftStrategy (None => constrained ADP bot).
    Pool must be a VorpTableSchema-valid DataFrame with `consensus_adp`.
    """
    nt, rs = config.n_teams, config.roster_size
    pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"], strict=False)}
    pos_str = pool["position"].astype(str)
    drafted: list[str] = []
    drafted_set: set[str] = set()
    rosters: dict[int, list[str]] = {s: [] for s in range(1, nt + 1)}
    counts: dict[int, dict[str, int]] = {s: {} for s in range(1, nt + 1)}
    my_roster_pos: dict[int, list[Position]] = {s: [] for s in range(1, nt + 1)}

    for pick in range(1, nt * rs + 1):
        seat = slot_for(pick, nt)
        strat = seat_strategies.get(seat)
        if strat is not None:
            state = DraftState(
                my_slot=seat,
                n_teams=nt,
                rounds=rs,
                picks=tuple(GsisId(g) for g in drafted),
                my_roster=tuple(my_roster_pos[seat]),
            )
            rec = strat.recommend(state, pool, config)
            gid = validate_gsis_id(str(rec.iloc[0]["gsis_id"]))
            my_roster_pos[seat].append(Position(pos_by_id[gid]))
        else:
            avail = ~pool["gsis_id"].isin(drafted_set)
            elig = _bot_eligible(counts[seat], rs - len(rosters[seat]))
            sub = pool[avail & pos_str.isin(elig)]
            if sub.empty:
                warnings.warn(
                    f"draft_mixed_field: bot at seat {seat}, pick {pick}: "
                    f"no available player at required positions {sorted(elig)}; "
                    f"picking best available — this bot roster will miss a positional minimum "
                    f"(pool is too thin at that position).",
                    stacklevel=2,
                )
                sub = pool[avail]
            gid = validate_gsis_id(str(bot_pick(sub, rng, adp_jitter=jitter)))
            counts[seat][pos_by_id[gid]] = counts[seat].get(pos_by_id[gid], 0) + 1

        drafted.append(gid)
        drafted_set.add(gid)
        rosters[seat].append(gid)

    return rosters
