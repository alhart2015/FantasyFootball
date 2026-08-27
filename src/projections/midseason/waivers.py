"""Is anyone on waivers better than someone on my team?

Stage 1 of the recommender: **does this free agent crack my starting lineup this week?**

In a 16-team league with seven starters and five bench spots, 192 players are rostered and the
wire is thin. "Is he better than my worst player" is therefore nearly always yes and nearly
always irrelevant — the move that matters is a stream, and a stream is worth making only if it
changes who I *start*. So the measure is the lineup, not the roster:

    lineup_gain = best_lineup(roster + him) - best_lineup(roster)

**A genuinely good free agent who still would not start scores exactly 0.0**, which is the
truthful answer most weeks and the reason this is a filter rather than a ranking. It also
handles for free the three things that actually drive streaming, none of which needed a rule:

- **Byes.** A player on bye has no weekly projection, so he is unstartable and the hole in the
  lineup appears by itself.
- **FLEX.** A receiver who beats my RB2 into the flex counts; one who does not, does not.
- **Positional scarcity**, without anyone hand-writing what scarce means.

Stage 2 — what the swap is worth in expected wins — is a separate, much more expensive
calculation. See §5 of `docs/superpowers/specs/2026-08-26-waiver-recommender-design.md`. This
module exists to make sure it only ever runs on candidates that could possibly matter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import choose_starters
from projections.midseason.injuries import weekly_multiplier
from projections.schemas import InjuryStatus, RosterSlot, display_str, parse_injury_status


@dataclass(frozen=True)
class Candidate:
    """One free agent, and what adding him would do to this week's starting lineup."""

    player_id: int
    player: str
    position: str
    nfl_team: str
    #: Points this week's optimal lineup gains by adding him. Zero when he would not start.
    lineup_gain: float
    #: His own projection for the week, after the injury adjustment.
    projected: float | None
    injury_status: InjuryStatus
    #: On a waiver claim rather than addable now. Different action, same value.
    on_waivers: bool
    percent_owned: float
    #: Who to drop for him, and what that costs. `None` when a roster spot is already free.
    drop_player_id: int | None = None
    drop_player: str = ""
    #: The dropped player's remaining-season projection — the cost side of the trade.
    drop_cost: float = 0.0

    @property
    def is_free(self) -> bool:
        """No one has to be dropped. The first thing worth telling a reader."""
        return self.drop_player_id is None


def adjusted_weekly_points(
    projected: float | None,
    status: InjuryStatus,
    *,
    source_is_injury_aware: bool,
) -> float | None:
    """One week's projection after the injury adjustment, or None if he cannot play.

    `None` propagates: a player with no projection for the week (a bye, or nobody projected
    him) is unstartable, which is a different fact from projecting zero. `choose_starters`
    depends on that distinction and so does every count downstream of it.
    """
    if projected is None:
        return None
    return float(projected) * weekly_multiplier(
        status, source_is_injury_aware=source_is_injury_aware
    )


@dataclass(frozen=True)
class _LineupRow:
    """A player as the lineup chooser sees him.

    `projected is None` means unstartable -- a bye, or nobody projected him -- which is a
    different fact from a projection of 0.0, and `choose_starters` depends on the difference.
    """

    player_id: int
    player: str
    position: str
    projected: float | None


def _row(player: Mapping[str, object], projected: float | None) -> _LineupRow:
    # `player_id` arrives from pandas as a numpy int, an object column, or missing. `str()`
    # first so every one of those reaches `int()` as something it accepts -- a numpy value
    # passed straight through type-checks as `object`, which it is.
    return _LineupRow(
        player_id=int(str(player.get("player_id", 0) or 0)),
        player=display_str(player.get("player")),
        position=display_str(player.get("pos")),
        projected=projected,
    )


def _startable_slots(config: LeagueConfig) -> dict[RosterSlot, int]:
    """Starting slots only. BENCH and IR hold players; they do not score."""
    return {
        slot: count
        for slot, count in config.roster_slots.items()
        if slot not in (RosterSlot.BENCH, RosterSlot.IR)
    }


def _open_spots(roster: pd.DataFrame, config: LeagueConfig) -> int:
    """Roster spots not currently filled.

    An add into an open spot costs nothing, which makes it categorically different from every
    other recommendation this tool makes — so it is counted rather than inferred.

    **IR slots do not count.** They hold a player who is already hurt; you cannot park a healthy
    add there. Counting them made a full 12-man roster look like it had two spaces going spare,
    so the tool reported every recommendation as free and never named a drop -- which is the
    failure mode where the tool is most confidently wrong, because a free add needs no
    justification and a costly one does.
    """
    capacity = sum(count for slot, count in config.roster_slots.items() if slot != RosterSlot.IR)
    return max(int(capacity) - len(roster), 0)


def lineup_points(rows: Sequence[_LineupRow], config: LeagueConfig) -> tuple[float, list[int]]:
    """This week's best startable total, and the indices of the players who start."""
    chosen = choose_starters(
        list(rows),
        _startable_slots(config),
        value=lambda row: row.projected,
        position=lambda row: row.position,
    )
    total = sum(float(rows[i].projected or 0.0) for i in chosen)
    return total, chosen


def rank_free_agents(
    roster: pd.DataFrame,
    free_agents: pd.DataFrame,
    weekly_projections: Mapping[str, float],
    remaining_points: Mapping[str, float],
    config: LeagueConfig,
    *,
    source_is_injury_aware: bool = True,
    min_gain: float = 0.5,
) -> list[Candidate]:
    """Free agents who would improve this week's starting lineup, best first.

    `weekly_projections` and `remaining_points` are keyed by ESPN player id as a string —
    rosters and free agents both arrive from ESPN, and going through `gsis_id` here would drop
    exactly the just-signed players a waiver tool is about.

    `source_is_injury_aware` says whether `weekly_projections` came from ESPN's own weekly feed,
    which already zeroes players it lists as Out. Defaults to True because that feed is where
    they come from; see `injuries.weekly_multiplier` for why applying a second discount produces
    a plausible wrong number.

    `min_gain` filters the noise. Half a point of projected lineup gain is not a roster move,
    and a list that includes it trains the reader to skip the list.
    """
    roster_rows = [
        _row(
            player,
            adjusted_weekly_points(
                weekly_projections.get(str(int(str(player.get("player_id", 0) or 0)))),
                _status(player),
                source_is_injury_aware=source_is_injury_aware,
            ),
        )
        for _, player in roster.iterrows()
    ]
    baseline, _ = lineup_points(roster_rows, config)
    open_spots = _open_spots(roster, config)

    candidates: list[Candidate] = []
    for _, agent in free_agents.iterrows():
        espn_id = int(str(agent.get("player_id", 0) or 0))
        status = _status(agent)
        projected = adjusted_weekly_points(
            weekly_projections.get(str(espn_id)),
            status,
            source_is_injury_aware=source_is_injury_aware,
        )
        with_him = [*roster_rows, _row(agent, projected)]
        after, chosen = lineup_points(with_him, config)
        gain = after - baseline
        if gain < min_gain:
            continue

        # An open bench or IR spot means nobody has to go, which is categorically different
        # from every other recommendation here -- so it is checked before a drop is chosen at
        # all, rather than by pricing a drop nobody has to make.
        drop = (
            None
            if open_spots > 0
            else _drop_candidate(with_him, chosen, remaining_points, skip_index=len(with_him) - 1)
        )
        candidates.append(
            Candidate(
                player_id=espn_id,
                player=display_str(agent.get("player")),
                position=display_str(agent.get("pos")),
                nfl_team=display_str(agent.get("nfl_team")),
                lineup_gain=gain,
                projected=projected,
                injury_status=status,
                on_waivers=bool(agent.get("on_waivers", False)),
                percent_owned=float(agent.get("percent_owned", 0.0) or 0.0),
                drop_player_id=None if drop is None else drop.player_id,
                drop_player="" if drop is None else drop.player,
                drop_cost=0.0 if drop is None else drop.cost,
            )
        )

    candidates.sort(key=lambda c: (-c.lineup_gain, c.player))
    return candidates


def _status(player: Mapping[str, object]) -> InjuryStatus:
    status, _ = parse_injury_status(player.get("injury_status"))
    return status


@dataclass(frozen=True)
class _Drop:
    player_id: int
    player: str
    cost: float


def _drop_candidate(
    rows: Sequence[_LineupRow],
    chosen: Sequence[int],
    remaining_points: Mapping[str, float],
    *,
    skip_index: int,
) -> _Drop | None:
    """The cheapest player NOT in the optimal lineup, by remaining-season points.

    Two rules, both of which exist because the obvious version gets them wrong:

    **Never someone the lineup uses.** The drop is picked from the leftovers *after* the free
    agent is in the lineup, so a player the add just displaced is droppable and a player the add
    did not displace is not. Picking "my worst player" instead would happily suggest dropping
    someone who is still starting.

    **Never the free agent himself**, which is why `skip_index` exists — he is the last row, and
    a candidate who does not crack the lineup would otherwise be his own cheapest leftover and
    the tool would recommend adding and immediately dropping him.
    """
    starters = set(chosen)
    cheapest: _Drop | None = None
    for index, row in enumerate(rows):
        if index in starters or index == skip_index:
            continue
        cost = float(remaining_points.get(str(row.player_id), 0.0))
        if cheapest is None or cost < cheapest.cost:
            cheapest = _Drop(player_id=row.player_id, player=row.player, cost=cost)
    return cheapest
