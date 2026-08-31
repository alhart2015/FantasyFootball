"""Where each roster is strong and where it is thin — measured, not eyeballed.

Raw season points cannot answer "am I strong at RB". In a league that starts two backs and one
flex, a fourth running back is worth **nothing** to the lineup and a great deal on the trade
market. Both quantities here are therefore **marginal and roster-aware**, and both fall out of
`choose_starters`, which the waiver work extracted for exactly this reason.

**Surplus** — what a player is worth to his *own* team:

    surplus(p) = best_lineup(roster) - best_lineup(roster - p)

The points the lineup loses once the next man steps up and the whole thing re-optimises. Low
surplus means tradeable. This is measured through the cascade, not against a direct backup: drop
an RB2 and the flex RB slides up and a bench player enters the flex, so the naive comparison
understates every move.

**Need** — what an upgrade at a position is worth:

    need(q) = best_lineup(roster + ref_q) - best_lineup(roster)

`ref_q` is a fixed reference player, the **league-median starter** at `q`. Fixed so the number
compares across teams; median rather than a star (which makes every team look needy) or
replacement level (which makes none of them).

The worked case this was built against: Lamar Jackson outprojects Josh Jacobs by 106 points and
his surplus on that roster is **half** Jacobs'. One QB starts and a 229-point quarterback sits on
waivers, so losing Lamar costs the gap to a free replacement while losing Jacobs costs the gap to
a bench back plus the bye cover. Surplus finds that; points do not.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from projections.draft.roster_eligibility import choose_starters
from projections.midseason.valuation import PlayerValue
from projections.schemas import InjuryStatus, Position, RosterSlot

#: The positions a trade can involve. Kickers and defenses are not valued (§2 of the spec) and
#: are never traded by this tool.
TRADEABLE: tuple[str, ...] = tuple(p.value for p in Position)


def lineup_points(
    players: Sequence[PlayerValue],
    roster_slots: Mapping[RosterSlot, int],
    *,
    use_market: bool = False,
) -> float:
    """Optimal starting-lineup points for `players`.

    `use_market` scores the same lineup under ESPN's numbers instead of ours — which is how the
    opponent's side of a trade gets judged on the view they actually hold.
    """
    chosen = choose_starters(
        players,
        roster_slots,
        value=lambda p: p.market if use_market else p.ours,
        position=lambda p: p.position,
    )
    return sum((players[i].market if use_market else players[i].ours) for i in chosen)


def surplus(
    players: Sequence[PlayerValue],
    player: PlayerValue,
    roster_slots: Mapping[RosterSlot, int],
    *,
    use_market: bool = False,
) -> float:
    """Lineup points lost if `player` leaves this roster. Zero for a true bench body."""
    without = [p for p in players if p.gsis_id != player.gsis_id]
    return lineup_points(players, roster_slots, use_market=use_market) - lineup_points(
        without, roster_slots, use_market=use_market
    )


def need(
    players: Sequence[PlayerValue],
    position: str,
    reference: float,
    roster_slots: Mapping[RosterSlot, int],
    *,
    use_market: bool = False,
) -> float:
    """Lineup points gained by adding a `reference`-quality player at `position`.

    Never negative: adding a player cannot make the optimal lineup worse, since the optimiser is
    free to leave him on the bench.
    """
    # Healthy by construction: `reference` is already a median over players whose own haircuts
    # were applied, so a second one here would discount the same fact twice.
    probe = PlayerValue(
        gsis_id="__reference__",
        espn_id=-1,
        full_name=f"reference {position}",
        position=position,
        market=reference,
        ours=reference,
        injury_status=InjuryStatus.ACTIVE,
        injury_raw="",
    )
    return lineup_points([*players, probe], roster_slots, use_market=use_market) - lineup_points(
        players, roster_slots, use_market=use_market
    )


def median_starters(
    rosters: Mapping[int, Sequence[PlayerValue]],
    roster_slots: Mapping[RosterSlot, int],
) -> dict[str, float]:
    """`position -> median points of a STARTER at that position`, across every team.

    Starters only. Taken over all rostered players it would sit near replacement level, because
    a 16-team league rosters far more players than it starts, and every team would then look
    equally needy everywhere.
    """
    by_pos: dict[str, list[float]] = {pos: [] for pos in TRADEABLE}
    for players in rosters.values():
        chosen = choose_starters(
            players, roster_slots, value=lambda p: p.ours, position=lambda p: p.position
        )
        for i in chosen:
            if players[i].position in by_pos:
                by_pos[players[i].position].append(players[i].ours)
    return {pos: (statistics.median(vals) if vals else 0.0) for pos, vals in by_pos.items()}


@dataclass(frozen=True)
class TeamShape:
    """One team's tradeable surplus and unmet need, per position."""

    team_id: int
    team_name: str
    #: gsis -> lineup points lost if he goes. Low = a chip.
    surplus: dict[str, float]
    #: position -> lineup points a median starter would add. High = they will pay.
    need: dict[str, float]
    players: tuple[PlayerValue, ...]

    def chips(self, position: str, *, max_surplus: float) -> list[PlayerValue]:
        """Players at `position` this team can afford to give up, cheapest first."""
        out = [
            p
            for p in self.players
            if p.position == position and self.surplus.get(p.gsis_id, 0.0) <= max_surplus
        ]
        return sorted(out, key=lambda p: self.surplus.get(p.gsis_id, 0.0))


def team_shapes(
    rosters: Mapping[int, Sequence[PlayerValue]],
    names: Mapping[int, str],
    roster_slots: Mapping[RosterSlot, int],
) -> dict[int, TeamShape]:
    """Surplus and need for every team in the league.

    **Computed identically for all 16 teams — "mine" is a display flag applied later, never a
    separate code path.** Two sides of a trade scored by subtly different rules is how a tool
    like this produces a confident wrong answer.
    """
    reference = median_starters(rosters, roster_slots)
    shapes: dict[int, TeamShape] = {}
    for team_id, players in rosters.items():
        seq = tuple(players)
        shapes[team_id] = TeamShape(
            team_id=team_id,
            team_name=str(names.get(team_id, team_id)),
            surplus={p.gsis_id: surplus(seq, p, roster_slots) for p in seq},
            need={pos: need(seq, pos, reference.get(pos, 0.0), roster_slots) for pos in TRADEABLE},
            players=seq,
        )
    return shapes


__all__ = [
    "TRADEABLE",
    "TeamShape",
    "lineup_points",
    "median_starters",
    "need",
    "surplus",
    "team_shapes",
]
