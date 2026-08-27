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

Stage 2 — what the swap is worth in expected wins, which is the OBJECTIVE — lives in
`midseason.swap_impact`. It is a full Monte-Carlo season per candidate, and this module exists
to make sure it only ever runs on candidates that could possibly matter. See §5 of
`docs/superpowers/specs/2026-08-26-waiver-recommender-design.md`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from projections.draft.assistant.performance_variance import SEASON_GAMES
from projections.draft.backtest.espn_weekly import parse_espn_weekly
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import choose_starters
from projections.ingest.espn_league import espn_gsis_crosswalk
from projections.midseason.injuries import season_multiplier, weekly_multiplier
from projections.midseason.my_team import MyTeamRun
from projections.schemas import InjuryStatus, RosterSlot, Ruleset, display_str, parse_injury_status


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
    #: An active roster spot is already free, so nobody has to go. Stated as its own field
    #: rather than inferred from an absent drop: "we found nobody to drop" and "you do not need
    #: to drop anyone" are opposite facts, and `is_free` used to report both as the latter.
    needs_no_drop: bool = False
    #: Who to drop for him, and what that costs. `None` when no drop was named -- which is
    #: either because none is needed (`needs_no_drop`) or because nobody droppable could be
    #: priced.
    drop_player_id: int | None = None
    drop_player: str = ""
    #: The dropped player's remaining-season projection — the cost side of the trade.
    drop_cost: float = 0.0

    @property
    def is_free(self) -> bool:
        """No one has to be dropped. The first thing worth telling a reader."""
        return self.needs_no_drop


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
    #: Parked in an IR slot. He cannot start, and dropping him frees no ACTIVE roster spot, so
    #: he is not a drop candidate either -- which is the third place `is_on_ir` says needs it
    #: and the one an earlier version left out.
    on_ir: bool = False


def player_id(player: Mapping[str, object]) -> int:
    """An ESPN player id out of a pandas row.

    `float()` first, then `int()`. The column arrives as int64 normally, but any frame that has
    been through a merge introducing an NA becomes float64, and `int("12345.0")` raises -- which
    is precisely the shape an earlier comment here claimed to be handling.
    """
    raw = player.get("player_id", 0)
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return 0
    if isinstance(raw, str):
        return int(float(raw)) if raw.strip() else 0
    if isinstance(raw, int | float):
        return int(raw)
    # A numpy scalar, which types as `object` but converts fine.
    return int(float(str(raw)))


def is_on_ir(player: Mapping[str, object]) -> bool:
    """Whether this roster row is parked in an IR slot.

    **One definition, and all three places read it.** An IR player does not occupy an active
    roster spot, cannot be started, and is not a drop candidate — dropping him frees an IR slot,
    not the active one an add needs. Each was got wrong separately: the headcount charged him
    against active capacity, the lineup let him hold a starting slot, and `_drop_candidate`
    named him (his forced-`None` projection makes him a permanent leftover and his discounted
    cost makes him the cheapest). The first two together meant that the morning after an injury
    the tool reported "nothing on the wire would change your lineup" — the one case it exists
    for.
    """
    return display_str(player.get("lineup_slot")) == RosterSlot.IR


def _row(player: Mapping[str, object], projected: float | None) -> _LineupRow:
    return _LineupRow(
        player_id=player_id(player),
        player=display_str(player.get("player")),
        position=display_str(player.get("pos")),
        # An IR player cannot legally start, whatever ESPN projects for him. `None` rather than
        # 0.0 because that is the value `choose_starters` reads as unstartable, and 0.0 can
        # still fill a slot nobody else is eligible for.
        projected=None if is_on_ir(player) else projected,
        on_ir=is_on_ir(player),
    )


def _open_spots(roster: pd.DataFrame, config: LeagueConfig) -> int:
    """ACTIVE roster spots not currently filled.

    An add into an open spot costs nothing, which makes it categorically different from every
    other recommendation this tool makes — so it is counted rather than inferred.

    **IR is excluded from both sides of the subtraction**, and getting only one side right is
    how this went wrong twice in opposite directions. Counting IR *slots* as capacity made a
    full roster look like it had spares, so every recommendation came back free and no drop was
    named. Then counting IR *players* against active capacity made a roster with someone on IR
    look full, so the tool named a drop nobody had to make. A spot is active, and so is the
    player who fills it.
    """
    capacity = sum(count for slot, count in config.roster_slots.items() if slot != RosterSlot.IR)
    active = sum(1 for _, player in roster.iterrows() if not is_on_ir(player))
    return max(int(capacity) - active, 0)


def lineup_points(rows: Sequence[_LineupRow], config: LeagueConfig) -> tuple[float, list[int]]:
    """This week's best startable total, and the indices of the players who start."""
    # `config.roster_slots` unfiltered: `choose_starters` only ever reads POSITION_SLOTS and
    # FLEX_SLOTS, so removing BENCH and IR first changed nothing. `backtest.lineup` passes it
    # through unfiltered too, and gets the same answer.
    chosen = choose_starters(
        list(rows),
        config.roster_slots,
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
) -> tuple[list[Candidate], int]:
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

    Returns the candidates and the number of ACTIVE roster spots currently open. Every
    `needs_no_drop` candidate is claiming the same spots, so a caller showing three of them
    without saying there is only one free spot is inviting a roster overfill.
    """
    roster_rows = [
        _row(
            player,
            adjusted_weekly_points(
                weekly_projections.get(str(player_id(player))),
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
        espn_id = player_id(agent)
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
        # `open_spots` is the same for every candidate, so each one is reported against the
        # SAME free spot. Acting on two of them overfills the roster, which is why the caller
        # is told how many there are rather than just that there is one.
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
                needs_no_drop=open_spots > 0,
                drop_player_id=None if drop is None else drop.player_id,
                drop_player="" if drop is None else drop.player,
                drop_cost=0.0 if drop is None else drop.cost,
            )
        )

    candidates.sort(key=lambda c: (-c.lineup_gain, c.player))
    return candidates, open_spots


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
        if row.on_ir:
            # Dropping him frees an IR slot, not the ACTIVE spot the add needs -- so the move
            # does not fit and the recommendation is unactionable. He is also the likeliest
            # player to be named: his projection is forced to None so he is always a leftover,
            # and his cost is injury-discounted so he is often the cheapest one.
            continue
        cost = remaining_points.get(str(row.player_id))
        if cost is None:
            # **Not 0.0.** A player the pool cannot price -- a kicker, a defense, a just-signed
            # back nobody projects yet -- would otherwise be the cheapest leftover by
            # construction, and the tool would recommend dropping him "at no cost". Those are
            # exactly the players a waiver tool is supposed to be careful with, and "we have no
            # number for him" is not "he is worth nothing".
            continue
        if cheapest is None or cost < cheapest.cost:
            cheapest = _Drop(player_id=row.player_id, player=row.player, cost=float(cost))
    return cheapest


# ---------------------------------------------------------------------------------------------
# The two inputs `rank_free_agents` needs, built from what ESPN and the pool give us.
# ---------------------------------------------------------------------------------------------


def weekly_projections_by_espn_id(
    payload: Mapping[str, Any], week: int, ruleset: Ruleset
) -> dict[str, float]:
    """ESPN's weekly projections out of a `kona_player_info` payload, keyed by ESPN id.

    **Keyed by ESPN id, not gsis.** `refresh_espn_weekly_projections` crosswalks through the
    id_map and drops whoever it cannot resolve, which is exactly the just-signed player a waiver
    tool exists to find. Rosters and free agents both arrive from ESPN, so the ESPN id is the
    key both sides already share.

    Scored under the league's own ruleset rather than read from ESPN's `appliedTotal`, so a free
    agent and a rostered player are valued the same way — and the same way the rest of this repo
    values anybody.

    A player with no projection for the week is ABSENT from the mapping rather than present at
    zero. That is what makes him unstartable downstream, which is how bye weeks work without a
    rule about bye weeks.

    Kickers and defenses are kept (`skill_positions_only=False`) because this also prices MY
    roster, and a starter the tool cannot price is a starter it silently treats as unstartable.
    """
    parsed = parse_espn_weekly(
        dict(payload), season=0, week=week, ruleset=ruleset, skill_positions_only=False
    )
    if parsed.empty:
        # `parse_espn_weekly` returns a COLUMN-LESS frame for an empty payload, so the lookup
        # below would raise `KeyError: 'projected_points'` -- out through the CLI's caught
        # tuple and into a traceback. An empty response is an ordinary answer (a filter that
        # matched nobody, a pre-publication week), and an empty mapping is what it means.
        return {}
    projected = parsed[parsed["projected_points"].notna()]
    return {
        str(espn_id): float(points)
        for espn_id, points in zip(projected["espn_id"], projected["projected_points"], strict=True)
    }


def remaining_points_by_espn_id(run: MyTeamRun, id_map: pd.DataFrame) -> dict[str, float]:
    """Rest-of-season points per ESPN id — the cost side of a drop.

    Injury-adjusted, because the point of this number is deciding who to let go: a player on IR
    is worth less for the rest of the season than his projection says, and that is exactly the
    situation in which you are looking for a drop candidate.

    The horizon is `run.week`, NOT a caller-supplied week, because `run.ros` already holds
    remaining points *as of* `run.week` — scaling that total by a multiplier derived from a
    different week applies the discount over a denominator the numerator was never built for.
    An earlier version took the caller's `week` to make a `--week` override move "everything",
    which moved this one thing out of step with the frame it operates on.

    A player the pool cannot price is ABSENT, not zero. `_drop_candidate` reads that absence as
    "cannot price him" rather than "he is worthless", which is what stops a kicker being
    recommended as a free drop.
    """
    crosswalk = espn_gsis_crosswalk(id_map)
    by_gsis = dict(
        zip(
            run.ros["gsis_id"].astype(str),
            run.ros["season_mean_fpts"].astype(float),
            strict=True,
        )
    )
    status_by_gsis = {
        display_str(player.get("gsis_id")): parse_injury_status(player.get("injury_status"))[0]
        for _, player in run.roster.iterrows()
        if display_str(player.get("gsis_id"))
    }
    games_left = max(SEASON_GAMES - (run.week - 1), 0)
    out: dict[str, float] = {}
    for espn_id, gsis in crosswalk.items():
        points = by_gsis.get(gsis)
        if points is None:
            continue
        status = status_by_gsis.get(gsis, InjuryStatus.ACTIVE)
        out[espn_id] = points * season_multiplier(status, games_remaining=games_left)
    return out
