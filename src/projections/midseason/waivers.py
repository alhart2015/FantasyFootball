"""Is anyone on waivers better than someone on my team?

**Better than anyone on it — starter or bench — on two horizons, reported separately.**

- **Season** (`season_upgrades`): his rest-of-season points against the weakest player on my
  roster by the same measure. This is the keep-or-cut question, and the list that feeds the
  expected-wins simulation in `midseason.swap_impact`.
- **This week** (`weekly_upgrades`): his projection for the week against the weakest player on
  my roster who is playing. This is the "who is going off this week" question, and it is
  deliberately blind to the rest of his season — a one-week stream is still worth seeing.

An earlier version asked only whether a free agent would crack this week's starting lineup.
That scored a free agent who was better than my whole bench as exactly zero, because he still
would not start — which hid the most common useful move there is: swap out the worst guy on the
bench for a better one. The lineup change is still computed and shown on every row, because it
is the part a reader can check against their own roster, but it no longer decides who is listed.

**The two horizons are never summed or mixed.** Weekly points and season points are different
currencies (see §5 of `docs/superpowers/specs/2026-08-26-waiver-recommender-design.md`); a
table that ranked on both would make the reader do the conversion by eye.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from projections.draft.assistant.performance_variance import SEASON_GAMES
from projections.draft.backtest.espn_weekly import parse_espn_weekly
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import bench_eligible_positions, choose_starters
from projections.ingest.espn_league import espn_gsis_crosswalk
from projections.midseason.injuries import season_multiplier, weekly_multiplier
from projections.midseason.my_team import MyTeamRun
from projections.schemas import InjuryStatus, RosterSlot, Ruleset, display_str, parse_injury_status


@dataclass(frozen=True)
class Candidate:
    """One free agent, compared against the weakest player on my roster on one horizon."""

    player_id: int
    player: str
    position: str
    nfl_team: str
    #: What this week's optimal lineup does if the move is made -- him in, the drop (if any)
    #: out. Shown, not filtered on: a bench upgrade legitimately reads 0.0 here, and a season
    #: upgrade that drops a starter can read negative.
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
    #: Who to drop for him, and what that costs. Set on the season list only -- the weekly list
    #: names who he out-projects, which is not a drop recommendation.
    drop_player_id: int | None = None
    drop_player: str = ""
    #: The dropped player's remaining-season projection — the cost side of the trade.
    drop_cost: float = 0.0
    #: His rest-of-season points, injury-adjusted. `None` when the pool cannot price him.
    season_points: float | None = None
    #: The weakest player on my roster on this list's horizon, and his points on it.
    beats_player: str = ""
    beats_points: float = 0.0
    #: His points minus `beats_points`, on this list's horizon. What the list is sorted by.
    margin: float = 0.0

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
    against active capacity, the lineup let him hold a starting slot, and the drop picker
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


def _status(player: Mapping[str, object]) -> InjuryStatus:
    status, _ = parse_injury_status(player.get("injury_status"))
    return status


def _startable_positions(config: LeagueConfig) -> frozenset[str]:
    """Positions this league can start. A kicker in a kicker-less league beats nobody."""
    return frozenset(position.value for position in bench_eligible_positions(config.roster_slots))


def _roster_rows(
    roster: pd.DataFrame,
    weekly_projections: Mapping[str, float],
    *,
    source_is_injury_aware: bool,
) -> list[_LineupRow]:
    return [
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


@dataclass(frozen=True)
class _Weakest:
    index: int
    player_id: int
    player: str
    points: float


def _weakest_by_position(
    rows: Sequence[_LineupRow], points: Mapping[int, float]
) -> dict[str, _Weakest]:
    """Per position, the roster row with the fewest `points` -- starter or bench, ignoring IR
    and the unpriced.

    **Per position, not across the roster.** Raw points do not compare across positions: run
    against the whole roster on a real league, backup quarterbacks filled both lists because a
    190-point QB "beats" a 55-point bench back -- while being useless to a team that starts one
    QB and already has him. A free agent is better than someone on my team when he is better than
    someone who plays his position.

    **IR is skipped** because dropping him frees an IR slot, not the ACTIVE spot an add needs,
    and because his forced-`None` projection says nothing about how good he is.

    **A player with no number is skipped, not scored 0.0.** On the season horizon that is a
    kicker, a defense, or a just-signed back the pool does not project yet; on the weekly one it
    is a bye. "We have no number for him" is not "he is worth nothing", and treating it as such
    made every free agent on the wire "better" than him -- and, on the season list, named him as
    a drop "at no cost".
    """
    weakest: dict[str, _Weakest] = {}
    for index, row in enumerate(rows):
        if row.on_ir or index not in points:
            continue
        value = points[index]
        current = weakest.get(row.position)
        if current is None or value < current.points:
            weakest[row.position] = _Weakest(index, row.player_id, row.player, value)
    return weakest


def _percent_owned(agent: Mapping[str, object]) -> float:
    """0.0 when absent or NA. numpy floats subclass `float`, so parser output passes through."""
    raw = agent.get("percent_owned")
    if isinstance(raw, int | float) and not pd.isna(raw):
        return float(raw)
    return 0.0


def _candidate(
    agent: Mapping[str, object],
    *,
    projected: float | None,
    lineup_gain: float,
    open_spots: int,
    beats: _Weakest | None,
    margin: float,
    season_points: float | None,
    drop: _Weakest | None = None,
) -> Candidate:
    return Candidate(
        player_id=player_id(agent),
        player=display_str(agent.get("player")),
        position=display_str(agent.get("pos")),
        nfl_team=display_str(agent.get("nfl_team")),
        lineup_gain=lineup_gain,
        projected=projected,
        injury_status=_status(agent),
        on_waivers=bool(agent.get("on_waivers", False)),
        percent_owned=_percent_owned(agent),
        needs_no_drop=open_spots > 0,
        drop_player_id=None if drop is None else drop.player_id,
        drop_player="" if drop is None else drop.player,
        drop_cost=0.0 if drop is None else drop.points,
        season_points=season_points,
        beats_player="" if beats is None else beats.player,
        beats_points=0.0 if beats is None else beats.points,
        margin=margin,
    )


def season_upgrades(
    roster: pd.DataFrame,
    free_agents: pd.DataFrame,
    weekly_projections: Mapping[str, float],
    remaining_points: Mapping[str, float],
    config: LeagueConfig,
    *,
    source_is_injury_aware: bool = True,
    min_margin: float = 5.0,
) -> tuple[list[Candidate], int]:
    """Free agents with more rest-of-season points than my weakest player at their position.

    **Starter or bench.** The comparison is against whoever at his position on my active roster
    has the fewest remaining points, wherever he sits this week -- and he is the drop. Dropping
    within the position keeps the roster's shape, so the swap never empties a starting slot
    nobody else can fill. If he is a starter, the
    row's `lineup_gain` says what that costs this week, which can be negative; the season
    simulation in `swap_impact` is what weighs the two.

    `weekly_projections` and `remaining_points` are keyed by ESPN player id as a string —
    rosters and free agents both arrive from ESPN, and going through `gsis_id` here would drop
    exactly the just-signed players a waiver tool is about. `remaining_points` must cover the
    free agents as well as my roster; `remaining_points_by_espn_id(..., free_agents=...)` does.

    `min_margin` is in SEASON points. Roughly 140 of them make a win, so the default of 5 is a
    few hundredths of a win: small, and still more than a rounding error.

    Returns the candidates and the number of ACTIVE roster spots currently open. With a spot
    open nobody has to go, and every such row is claiming the SAME spot.
    """
    rows = _roster_rows(roster, weekly_projections, source_is_injury_aware=source_is_injury_aware)
    open_spots = _open_spots(roster, config)
    season = {
        index: float(remaining_points[str(row.player_id)])
        for index, row in enumerate(rows)
        if str(row.player_id) in remaining_points
    }
    weakest_at = _weakest_by_position(rows, season)
    baseline, _ = lineup_points(rows, config)
    startable = _startable_positions(config)

    candidates: list[Candidate] = []
    for _, agent in free_agents.iterrows():
        if display_str(agent.get("pos")) not in startable:
            continue
        espn_id = str(player_id(agent))
        if espn_id not in remaining_points:
            continue
        weakest = weakest_at.get(display_str(agent.get("pos")))
        if weakest is None:
            # Nobody priced at his position: nobody to beat, and no like-for-like drop. A
            # position I roster nobody at is a lineup hole, which the weekly list catches.
            continue
        his_season = float(remaining_points[espn_id])
        margin = his_season - weakest.points
        if margin < min_margin:
            continue
        projected = adjusted_weekly_points(
            weekly_projections.get(espn_id),
            _status(agent),
            source_is_injury_aware=source_is_injury_aware,
        )
        # An open spot means nobody has to go, which is categorically different from every
        # other row -- so it is checked before any drop is named.
        drop = None if open_spots > 0 else weakest
        after_rows = [r for i, r in enumerate(rows) if drop is None or i != drop.index]
        after, _ = lineup_points([*after_rows, _row(agent, projected)], config)
        candidates.append(
            _candidate(
                agent,
                projected=projected,
                lineup_gain=after - baseline,
                open_spots=open_spots,
                beats=weakest,
                margin=margin,
                season_points=his_season,
                drop=drop,
            )
        )

    candidates.sort(key=lambda c: (-c.margin, c.player))
    return candidates, open_spots


def weekly_upgrades(
    roster: pd.DataFrame,
    free_agents: pd.DataFrame,
    weekly_projections: Mapping[str, float],
    remaining_points: Mapping[str, float],
    config: LeagueConfig,
    *,
    source_is_injury_aware: bool = True,
    min_margin: float = 0.5,
) -> list[Candidate]:
    """Free agents projected to outscore my weakest player at their position THIS WEEK.

    **Blind to the rest of the season on purpose.** This is the list for "who is going off this
    week", and a one-week wonder belongs on it. `season_points` is carried for display only.

    The weakest player is the lowest-projected one at his position on my active roster who is
    PLAYING (see `_weakest_by_position` for why per position). A bye has no projection, and
    counting it as 0.0 would make every free agent on the wire "better" than him. The hole a
    bye leaves is caught instead by `lineup_gain` -- a free agent who would start for me this
    week is listed even when he out-projects nobody who is playing.

    No drop is named. Who he out-projects this week is not who you should cut; the season list
    answers that.
    """
    rows = _roster_rows(roster, weekly_projections, source_is_injury_aware=source_is_injury_aware)
    open_spots = _open_spots(roster, config)
    weekly = {index: row.projected for index, row in enumerate(rows) if row.projected is not None}
    weakest_at = _weakest_by_position(rows, weekly)
    baseline, _ = lineup_points(rows, config)
    startable = _startable_positions(config)

    candidates: list[Candidate] = []
    for _, agent in free_agents.iterrows():
        if display_str(agent.get("pos")) not in startable:
            continue
        espn_id = str(player_id(agent))
        projected = adjusted_weekly_points(
            weekly_projections.get(espn_id),
            _status(agent),
            source_is_injury_aware=source_is_injury_aware,
        )
        if projected is None:
            continue
        weakest = weakest_at.get(display_str(agent.get("pos")))
        # Nobody playing at his position this week: he is compared against an empty spot.
        margin = projected - (0.0 if weakest is None else weakest.points)
        after, _ = lineup_points([*rows, _row(agent, projected)], config)
        gain = after - baseline
        if margin < min_margin and gain < min_margin:
            continue
        season = remaining_points.get(espn_id)
        candidates.append(
            _candidate(
                agent,
                projected=projected,
                lineup_gain=gain,
                open_spots=open_spots,
                beats=weakest,
                margin=margin,
                season_points=None if season is None else float(season),
            )
        )

    candidates.sort(key=lambda c: (-c.margin, c.player))
    return candidates


# ---------------------------------------------------------------------------------------------
# What `season_upgrades` and `weekly_upgrades` need, built from ESPN and the pool.
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


def remaining_points_by_espn_id(
    run: MyTeamRun, id_map: pd.DataFrame, free_agents: pd.DataFrame | None = None
) -> dict[str, float]:
    """Rest-of-season points per ESPN id — both sides of a season swap.

    Injury-adjusted, because the point of this number is deciding who to let go: a player on IR
    is worth less for the rest of the season than his projection says, and that is exactly the
    situation in which you are looking for a drop candidate.

    The horizon is `run.week`, NOT a caller-supplied week, because `run.ros` already holds
    remaining points *as of* `run.week` — scaling that total by a multiplier derived from a
    different week applies the discount over a denominator the numerator was never built for.
    An earlier version took the caller's `week` to make a `--week` override move "everything",
    which moved this one thing out of step with the frame it operates on.

    **Pass `free_agents`** so the add side is discounted too. Without it a free agent's status is
    unknown here, and an injured one would be compared at full health against a roster that is
    not -- flattering exactly the players a waiver list should be careful with.

    A player the pool cannot price is ABSENT, not zero. `_weakest_by_position` reads that
    absence as "cannot price him" rather than "he is worthless", which is what stops a kicker
    being recommended as a free drop.
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
    status_by_espn = (
        {}
        if free_agents is None
        else {str(player_id(agent)): _status(agent) for _, agent in free_agents.iterrows()}
    )
    games_left = max(SEASON_GAMES - (run.week - 1), 0)
    out: dict[str, float] = {}
    for espn_id, gsis in crosswalk.items():
        points = by_gsis.get(gsis)
        if points is None:
            continue
        status = status_by_gsis.get(gsis) or status_by_espn.get(espn_id, InjuryStatus.ACTIVE)
        out[espn_id] = points * season_multiplier(status, games_remaining=games_left)
    return out
