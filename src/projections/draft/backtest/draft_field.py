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
from projections.draft.roster_eligibility import bot_eligible
from projections.schemas import GsisId, Position, validate_gsis_id

_MINP = {Position.QB: 1, Position.RB: 3, Position.WR: 3, Position.TE: 1}
_MAXP = {Position.QB: 3, Position.RB: 6, Position.WR: 6, Position.TE: 3}


def seat_layout(
    seed: int, label_a: str = "now_or_never", label_b: str = "season_value"
) -> dict[int, str]:
    """Return a {seat: strategy_label} map for a 16-team snake draft.

    Odd seeds: role A at {2,6,10,14}, role B at {4,8,12,16}. Even seeds mirror
    (A<->B swap) so exposures cancel when summed over paired seeds. The other 8
    seats are bots. Defaults reproduce the historical now_or_never / season_value
    field byte-identically.
    """
    a, b = {2, 6, 10, 14}, {4, 8, 12, 16}
    if seed % 2 == 0:  # paired mirror
        a, b = b, a
    return {s: (label_a if s in a else label_b if s in b else "bot") for s in range(1, 17)}


def hero_seat_layout(*, hero_seat: int, hero_label: str, n_teams: int) -> dict[int, str]:
    """Seat map for the hero-vs-bots eval: `hero_label` at `hero_seat`, bots elsewhere.

    Works for any team count (the mixed `seat_layout` is hardcoded 16-team). The hero is
    the single non-bot seat; the rest are constrained-ADP bots (label "bot" => None
    strategy downstream).
    """
    if not 1 <= hero_seat <= n_teams:
        raise ValueError(f"hero_seat must be in [1, {n_teams}]; got {hero_seat}")
    return {s: (hero_label if s == hero_seat else "bot") for s in range(1, n_teams + 1)}


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
            counts_pos = {Position(p): c for p, c in counts[seat].items()}
            elig = bot_eligible(counts_pos, rs - len(rosters[seat]), minimums=_MINP, maximums=_MAXP)
            elig_values = {p.value for p in elig}
            sub = pool[avail & pos_str.isin(elig_values)]
            if sub.empty:
                warnings.warn(
                    f"draft_mixed_field: bot at seat {seat}, pick {pick}: "
                    f"no available player at required positions {sorted(p.value for p in elig)}; "
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
