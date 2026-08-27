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

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import choose_starters
from projections.midseason.injuries import weekly_multiplier
from projections.midseason.standings import (
    ProjectionInputError,
    StandingsRun,
    project_league_standings,
)
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


def _player_id(player: Mapping[str, object]) -> int:
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

    **One definition, because three places need it and they disagreed.** An IR player does not
    occupy an active roster spot, cannot be started, and is not a drop candidate that frees an
    active spot. Each of those was got wrong separately: the headcount charged him against
    active capacity, the lineup let him hold a starting slot, and the two facts together meant
    the morning after an injury the tool reported "nothing on the wire would change your
    lineup" -- the one case it exists for.
    """
    return display_str(player.get("lineup_slot")) == RosterSlot.IR


def _row(player: Mapping[str, object], projected: float | None) -> _LineupRow:
    return _LineupRow(
        player_id=_player_id(player),
        player=display_str(player.get("player")),
        position=display_str(player.get("pos")),
        # An IR player cannot legally start, whatever ESPN projects for him. `None` rather than
        # 0.0 because that is the value `choose_starters` reads as unstartable, and 0.0 can
        # still fill a slot nobody else is eligible for.
        projected=None if is_on_ir(player) else projected,
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
                weekly_projections.get(str(_player_id(player))),
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
        espn_id = _player_id(agent)
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
# Stage 2: what is the swap worth in expected wins?
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapImpact:
    """A candidate, re-simulated with the swap applied.

    The deltas are against a baseline computed once at the same seed — see `simulate_swaps`.
    """

    candidate: Candidate
    delta_wins: float
    delta_playoff_pct: float
    delta_title_pct: float

    @property
    def helps(self) -> bool:
        """Whether the move is worth making at all.

        Expected wins is the whole point of stage 2: it is the only currency in which "better
        this week" and "worse for the rest of the season" are the same unit, so a candidate
        with a big lineup gain and a negative delta is a candidate the drop ruins.
        """
        return self.delta_wins > 0.0


#: Standard error of a paired delta at 2,000 sims, measured by `scripts/measure_swap_noise.py`
#: on a synthetic 16-team league. Deltas smaller than this are not distinguishable from
#: simulation noise, and a caller printing them to three decimals is inventing precision.
PAIRED_DELTA_NOISE = 0.062


def simulate_swaps(
    payload: Mapping[str, Any],
    pool: pd.DataFrame,
    id_map: pd.DataFrame,
    availability: PlayerAvailability,
    params: VarianceParams,
    candidates: Sequence[Candidate],
    *,
    season: int,
    my_team_id: int,
    n_sims: int = 2000,
    seed: int = 0,
) -> list[SwapImpact]:
    """Re-simulate the season with each swap applied. Expensive; run it on a shortlist.

    **One baseline, one seed, differences only.** Two independent runs of the same league
    differ by simulation noise alone, and at 2,000 sims that noise (sd 0.090 wins) is larger
    than a realistic swap is worth. Every run here — the baseline and every candidate — starts
    from `np.random.default_rng(seed)`, so the two sides share their draws and most of the
    noise cancels. Measured: the paired difference carries sd 0.062 against 0.127 unpaired.

    That is a 2x reduction rather than the order of magnitude common random numbers can give,
    because `project_league_standings` reseeds internally and a roster change perturbs the draw
    sequence. It is enough: a 10-point season swap reads at 3.7x its own noise. See
    `scripts/measure_swap_noise.py`, which is the gate that established this and exits non-zero
    if it stops holding.

    Candidates whose player is not in the pool are skipped rather than simulated as a zero —
    a free agent nobody projects would otherwise look like a roster downgrade, which is a
    confident wrong answer rather than an absence of one.
    """
    baseline = _standings_row(
        project_league_standings(
            payload,
            pool,
            id_map,
            availability,
            params,
            season=season,
            n_sims=n_sims,
            rng=np.random.default_rng(seed),
        ),
        my_team_id,
    )

    projectable = set(pool["gsis_id"].astype(str))
    espn_to_gsis = _espn_to_gsis(id_map)

    impacts: list[SwapImpact] = []
    for candidate in candidates:
        gsis = espn_to_gsis.get(str(candidate.player_id))
        if gsis is None or gsis not in projectable:
            continue
        swapped = _payload_with_swap(payload, candidate, my_team_id=my_team_id)
        after = _standings_row(
            project_league_standings(
                swapped,
                pool,
                id_map,
                availability,
                params,
                season=season,
                n_sims=n_sims,
                # A FRESH generator at the SAME seed, not the one the baseline consumed.
                # Reusing a spent generator makes every candidate share a different draw
                # sequence from the baseline, which is the unpaired case wearing paired
                # clothing.
                rng=np.random.default_rng(seed),
            ),
            my_team_id,
        )
        impacts.append(
            SwapImpact(
                candidate=candidate,
                delta_wins=after["projected_wins"] - baseline["projected_wins"],
                delta_playoff_pct=after["make_playoffs_pct"] - baseline["make_playoffs_pct"],
                delta_title_pct=after["champ_pct"] - baseline["champ_pct"],
            )
        )

    impacts.sort(key=lambda impact: -impact.delta_wins)
    return impacts


def _standings_row(run: StandingsRun, my_team_id: int) -> dict[str, float]:
    """My row of a `StandingsRun`, as plain floats."""
    mine = run.standings[run.standings["team_id"] == my_team_id]
    if mine.empty:
        raise ProjectionInputError(
            f"team {my_team_id} is not in the projected standings; it cannot be compared "
            "against itself."
        )
    row = mine.iloc[0]
    return {
        "projected_wins": float(row["projected_wins"]),
        "make_playoffs_pct": float(row["make_playoffs_pct"]),
        "champ_pct": float(row["champ_pct"]),
    }


def _espn_to_gsis(id_map: pd.DataFrame) -> dict[str, str]:
    """ESPN id -> gsis, deduplicated on the ESPN side.

    `IdMapSchema` marks only `gsis_id` unique, and the live id_map holds ESPN ids that map to
    two different players. Same dedup rule as `espn_league.espn_to_gsis`, which is where the
    reasoning lives.
    """
    cross = id_map[["espn_id", "gsis_id"]].dropna().astype({"espn_id": str})
    cross = cross.drop_duplicates("espn_id")
    return dict(zip(cross["espn_id"].astype(str), cross["gsis_id"].astype(str), strict=True))


def _payload_with_swap(
    payload: Mapping[str, Any], candidate: Candidate, *, my_team_id: int
) -> dict[str, Any]:
    """A copy of the payload with the candidate added to my roster and the drop removed.

    **The add takes the dropped player's place in the list, rather than being appended.**
    Roster order reaches the simulator, which draws per player in that order, so appending
    shifts every subsequent player onto a different draw and the paired comparison silently
    becomes an unpaired one. Caught by `test_a_no_op_swap_reports_exactly_zero`, which read
    0.05 wins for dropping a player and adding the same player back.

    **Deep-copied.** Mutating the caller's payload would make the baseline and every subsequent
    candidate compare against a roster that has been silently accumulating adds, and the
    resulting deltas would look plausible.
    """
    swapped = copy.deepcopy(dict(payload))
    for team in swapped.get("teams", []) or []:
        if int(team.get("id", 0) or 0) != my_team_id:
            continue
        roster = team.setdefault("roster", {}).setdefault("entries", [])
        entry = _roster_entry(candidate)
        replaced = False
        if candidate.drop_player_id is not None:
            for index, existing in enumerate(roster):
                if _entry_player_id(existing) == candidate.drop_player_id:
                    # In place, and inheriting the slot: the simulator sets its own lineup, but
                    # keeping the slot means a bench-for-bench swap leaves the payload
                    # otherwise identical.
                    entry["lineupSlotId"] = existing.get("lineupSlotId", 20)
                    roster[index] = entry
                    replaced = True
                    break
        if not replaced:
            roster.append(entry)
    return swapped


def _roster_entry(candidate: Candidate) -> dict[str, Any]:
    """A `teams[].roster.entries` entry for a player who was not on the roster."""
    return {
        #: BENCH by default. The simulator fills its own lineup from projections, so the slot
        #: only has to be one that counts toward the roster rather than the right one.
        "lineupSlotId": 20,
        "playerId": candidate.player_id,
        "playerPoolEntry": {
            "player": {
                "id": candidate.player_id,
                "fullName": candidate.player,
                "defaultPositionId": _ESPN_POSITION_IDS.get(candidate.position, 3),
                "proTeamId": 0,
            }
        },
    }


#: Position name -> ESPN `defaultPositionId`, for writing a synthetic roster entry back.
_ESPN_POSITION_IDS: dict[str, int] = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DST": 16}


def _entry_player_id(entry: Mapping[str, Any]) -> int:
    player = (entry.get("playerPoolEntry", {}) or {}).get("player", {}) or {}
    return int(player.get("id", entry.get("playerId", 0)) or 0)
