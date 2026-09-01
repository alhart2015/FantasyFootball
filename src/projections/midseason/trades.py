"""Find trades that are fair on the market's numbers and good on ours.

Two stages, the same shape the waiver recommender settled on and for the same reasons.

**Stage 1 — lineup arithmetic.** Δ optimal starting-lineup points for both sides, over every
candidate that clears the surplus/need screen. Cheap enough to run exhaustively, and verifiable
by hand against a roster in a way a simulated win total is not. It decides *what gets simulated*;
it is not the answer.

**Stage 2 — Δ expected wins.** A paired Monte-Carlo over the real remaining schedule, for the
shortlist only. Unlike a waiver add, a trade perturbs the *league*: the opponent's roster changes
too, so their record moves, and if they are a playoff rival that matters on top of the points.

**What makes a trade proposable.** Two conditions, and the asymmetry between them is the whole
feature:

- **Acceptable to them, on their numbers** — their optimal lineup improves under ESPN's
  projections, and the ESPN-point balance does not favour me.
- **Good for me, on ours** — my lineup, and then my expected wins, improve under the consensus.

This is a stated, checkable proxy, not a model of a person. Issue #154 warns against pretending
to know what an opponent will accept, and that warning stands: nothing here knows who never
trades in-division. **The output says "this looks fair on ESPN's numbers", never "he will say
yes".**

**Two different reasons a trade can be good, kept apart.** *Fit* is positional — I have surplus
RB, you have surplus WR, we each start more of what we lack — and it is real even if both
managers value every player identically. *Edge* is disagreement: we think a player is worth more
than the market does. `TradeProposal` attributes the gain to each, because a fit trade is one to
send with a straight face and an edge trade is a bet on our model.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.midseason.roster_shape import TeamShape, lineup_points
from projections.midseason.standings import ProjectionInputError, project_league_standings
from projections.midseason.swap_impact import _injury_adjusted_pool
from projections.midseason.valuation import PlayerValue
from projections.schemas import RosterSlot

#: Below this, a Δ-wins figure is inside the paired simulation noise measured by
#: `scripts/measure_swap_noise.py` (sd ~0.062 wins at 2,000 sims) and is not a difference.
WINS_NOISE_FLOOR = 0.15


@dataclass(frozen=True)
class TradeProposal:
    """One size-neutral swap, scored from both sides."""

    partner_id: int
    partner_name: str
    send: tuple[PlayerValue, ...]
    receive: tuple[PlayerValue, ...]

    #: Stage 1, my optimal lineup, our projections.
    my_lineup_gain: float
    #: Their optimal lineup, ESPN's projections — the acceptability proxy.
    their_lineup_gain_market: float
    #: Sum of ESPN points in, minus ESPN points out. Negative or near zero is the sellable trade.
    espn_balance: float

    #: Stage 2, filled in only for the shortlist.
    delta_wins: float | None = None
    delta_playoff: float | None = None
    delta_title: float | None = None

    @property
    def edge_gain(self) -> float:
        """The part of the gain that is disagreement with the market, not positional fit."""
        return sum(p.edge for p in self.receive) - sum(p.edge for p in self.send)

    @property
    def fit_gain(self) -> float:
        """The part that would hold even if we valued every player exactly as ESPN does."""
        return self.my_lineup_gain - self.edge_gain

    @property
    def is_lowball(self) -> bool:
        """True when I take more ESPN value than I give. Printed, never silently ranked first."""
        return self.espn_balance > 0.0

    @property
    def above_noise(self) -> bool:
        return self.delta_wins is not None and abs(self.delta_wins) >= WINS_NOISE_FLOOR


def _swap(
    players: Sequence[PlayerValue],
    out: Sequence[PlayerValue],
    incoming: Sequence[PlayerValue],
) -> list[PlayerValue]:
    """Replace `out` with `incoming` **in place, in slot order**.

    Order is load-bearing and this is not cosmetic. The waiver work measured that appending an
    arriving player instead of putting him in the departing player's slot shifts every later
    player onto a different random draw, which silently unpairs the comparison -- a no-op swap
    read as 0.05 wins until it was fixed, and only an exact-zero assertion catches it.
    """
    leaving = {p.gsis_id for p in out}
    arriving = list(incoming)
    result: list[PlayerValue] = []
    for player in players:
        if player.gsis_id in leaving:
            if arriving:
                result.append(arriving.pop(0))
        else:
            result.append(player)
    result.extend(arriving)
    return result


def generate(
    mine: TeamShape,
    theirs: TeamShape,
    roster_slots: Mapping[RosterSlot, int],
    *,
    max_players: int = 2,
    min_lineup_gain: float = 1.0,
    espn_tolerance: float = 5.0,
) -> list[TradeProposal]:
    """Every size-neutral swap with this one partner that clears both §5 conditions.

    `espn_tolerance` is how far the ESPN balance may tip in my favour before the trade reads as
    a lowball rather than a fair offer. Trades beyond it are dropped here rather than ranked and
    flagged, because a proposal nobody would accept is not a recommendation.
    """
    proposals: list[TradeProposal] = []
    my_base = lineup_points(mine.players, roster_slots)
    their_base_market = lineup_points(theirs.players, roster_slots, use_market=True)

    for size in range(1, max_players + 1):
        for send in combinations(mine.players, size):
            for receive in combinations(theirs.players, size):
                my_after = lineup_points(_swap(mine.players, send, receive), roster_slots)
                my_gain = my_after - my_base
                if my_gain < min_lineup_gain:
                    continue

                their_after = lineup_points(
                    _swap(theirs.players, receive, send), roster_slots, use_market=True
                )
                their_gain = their_after - their_base_market
                if their_gain <= 0.0:
                    # They evaluate on the market view. A trade that does not improve their own
                    # lineup THERE is not one they have a reason to accept.
                    continue

                balance = sum(p.market for p in receive) - sum(p.market for p in send)
                if balance > espn_tolerance:
                    continue

                proposals.append(
                    TradeProposal(
                        partner_id=theirs.team_id,
                        partner_name=theirs.team_name,
                        send=tuple(send),
                        receive=tuple(receive),
                        my_lineup_gain=my_gain,
                        their_lineup_gain_market=their_gain,
                        espn_balance=balance,
                    )
                )
    return proposals


def generate_all(
    shapes: Mapping[int, TeamShape],
    my_team_id: int,
    roster_slots: Mapping[RosterSlot, int],
    *,
    max_players: int = 2,
    min_lineup_gain: float = 1.0,
    espn_tolerance: float = 5.0,
    top: int = 20,
) -> list[TradeProposal]:
    """Stage 1 over the whole league, best lineup gain first."""
    mine = shapes[my_team_id]
    found: list[TradeProposal] = []
    for team_id, theirs in shapes.items():
        if team_id == my_team_id:
            continue
        found.extend(
            generate(
                mine,
                theirs,
                roster_slots,
                max_players=max_players,
                min_lineup_gain=min_lineup_gain,
                espn_tolerance=espn_tolerance,
            )
        )
    found.sort(key=lambda t: t.my_lineup_gain, reverse=True)
    return found[:top]


def payload_with_trade(
    payload: Mapping[str, Any], proposal: TradeProposal, *, my_team_id: int
) -> dict[str, Any]:
    """A deep copy of the league payload with the two rosters swapped, **entry for entry**.

    Both players already exist in the payload, so nothing is synthesised: the entries are
    exchanged at their existing list positions and keep their lineup slots. That matters for the
    same reason it does in `swap_impact._payload_with_swap` -- roster order reaches the
    simulator, which draws per player in order, so moving a player to the end of a list shifts
    every later player onto a different draw and silently unpairs the comparison. A no-op trade
    must read exactly 0.0, and only that assertion catches this.

    Raises if either side's player is not where the proposal says he is, rather than absorbing
    it: a mis-sized roster produces a plausible-looking delta and no error.
    """
    swapped = copy.deepcopy(dict(payload))
    wanted = {
        my_team_id: ({p.espn_id for p in proposal.send}, list(proposal.receive)),
        proposal.partner_id: ({p.espn_id for p in proposal.receive}, list(proposal.send)),
    }
    seen: dict[int, int] = {my_team_id: 0, proposal.partner_id: 0}

    for team in swapped.get("teams", []) or []:
        team_id = int(team.get("id", 0) or 0)
        if team_id not in wanted:
            continue
        leaving, arriving = wanted[team_id]
        entries = team.setdefault("roster", {}).setdefault("entries", [])
        incoming = list(arriving)
        for index, entry in enumerate(entries):
            player = (entry.get("playerPoolEntry", {}) or {}).get("player", {}) or {}
            pid = int(player.get("id", entry.get("playerId", 0)) or 0)
            if pid not in leaving:
                continue
            new = incoming.pop(0)
            replacement = copy.deepcopy(entry)
            replacement["playerId"] = new.espn_id
            pool_entry = replacement.setdefault("playerPoolEntry", {})
            pool_entry["id"] = new.espn_id
            new_player = pool_entry.setdefault("player", {})
            new_player["id"] = new.espn_id
            new_player["fullName"] = new.full_name
            new_player["injuryStatus"] = new.injury_raw or "ACTIVE"
            entries[index] = replacement
            seen[team_id] += 1

    for team_id, (leaving, _) in wanted.items():
        if seen[team_id] != len(leaving):
            raise ProjectionInputError(
                f"team {team_id} was expected to send {len(leaving)} player(s) but "
                f"{seen[team_id]} matched its roster; the swap has no slot to occupy and the "
                "comparison would run against a different-sized roster."
            )
    return swapped


def simulate_trades(
    payload: Mapping[str, Any],
    pool: pd.DataFrame,
    id_map: pd.DataFrame,
    availability: PlayerAvailability,
    params: VarianceParams,
    proposals: Sequence[TradeProposal],
    *,
    season: int,
    my_team_id: int,
    week: int,
    n_sims: int = 2000,
    seed: int = 0,
) -> list[TradeProposal]:
    """Stage 2: re-simulate the season with each trade applied. Expensive; shortlist only.

    **One baseline, one seed, differences only** -- the same discipline `simulate_swaps`
    documents. Every run starts from `np.random.default_rng(seed)` so the two sides share their
    draws and most of the noise cancels; the residual paired sd is ~0.062 wins at 2,000 sims,
    which is why `WINS_NOISE_FLOOR` exists and is printed rather than left to the reader.

    **The pool is injury-adjusted first**, reusing `swap_impact._injury_adjusted_pool`.
    `project_league_standings` knows nothing about `injury_status`, so the raw pool would
    simulate a suspended player at full strength -- which on this league is the single case the
    tool was built for.
    """
    adjusted = _injury_adjusted_pool(pool, payload, id_map, week=week)

    def run(pay: Mapping[str, Any]) -> tuple[float, float, float]:
        standings = project_league_standings(
            pay,
            adjusted,
            id_map,
            availability,
            params,
            season=season,
            n_sims=n_sims,
            rng=np.random.default_rng(seed),
        ).standings
        row = standings[standings["team_id"] == my_team_id].iloc[0]
        return (
            float(row["projected_wins"]),
            float(row["make_playoffs_pct"]),
            float(row["champ_pct"]),
        )

    base_wins, base_playoff, base_title = run(payload)
    scored: list[TradeProposal] = []
    for proposal in proposals:
        wins, playoff, title = run(payload_with_trade(payload, proposal, my_team_id=my_team_id))
        scored.append(
            replace(
                proposal,
                delta_wins=wins - base_wins,
                delta_playoff=playoff - base_playoff,
                delta_title=title - base_title,
            )
        )
    scored.sort(key=lambda t: t.delta_wins or 0.0, reverse=True)
    return scored


__all__ = [
    "WINS_NOISE_FLOOR",
    "TradeProposal",
    "generate",
    "generate_all",
    "payload_with_trade",
    "simulate_trades",
]
