"""What a roster swap is worth in expected season wins.

**This is the objective, not a decoration on one.** Points on two horizons are not a common
currency: "20% better next week" and "2.5 expected wins worse for the rest of the year" cannot
be compared by eye. Expected wins is the conversion — 20% for one week IS 0.2 wins — so the
drop's future cost and the add's near-term gain land on the same scale and the sign of their
sum is the recommendation.

Split out of `waivers.py`, whose docstring had been calling this "a separate, much more
expensive calculation" while it sat in the same file. Stage 1 there needs a `LeagueConfig` and
a lineup chooser; this needs numpy, the availability model, the variance model, the season
simulator and ESPN payload surgery. They share only `Candidate`.

**Expensive on purpose.** Each candidate is a full Monte-Carlo season. `waivers.rank_free_agents`
exists to make sure this only ever runs on candidates that could possibly matter.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import SEASON_GAMES, VarianceParams
from projections.ingest.espn_league import ESPN_POSITION_IDS, espn_gsis_crosswalk
from projections.midseason.injuries import season_multiplier
from projections.midseason.standings import (
    ProjectionInputError,
    StandingsRun,
    project_league_standings,
)
from projections.midseason.waivers import Candidate
from projections.schemas import InjuryStatus, parse_injury_status


@dataclass(frozen=True)
class SwapImpact:
    """A candidate, re-simulated with the swap applied.

    The deltas are against a baseline computed once at the same seed — see `simulate_swaps`.
    """

    candidate: Candidate
    delta_wins: float
    delta_playoff_pct: float
    delta_title_pct: float
    #: False when the swap could not be simulated -- the added player is not in the projection
    #: pool, so there is nothing to put on the roster. The deltas are then zero and MEAN
    #: nothing, which is a different fact from a simulated zero and must not print the same.
    simulated: bool = True

    @property
    def helps(self) -> bool:
        """Whether the move is worth making at all.

        Expected wins is the whole point: it is the only currency in which "better this week"
        and "worse for the rest of the season" are the same unit, so a candidate with a big
        lineup gain and a negative delta is a candidate the drop ruins. That is the case the
        design was written around, and reporting it requires this property to be read rather
        than merely defined.
        """
        return self.simulated and self.delta_wins > 0.0

    @property
    def beats_noise(self) -> bool:
        """Whether the delta is bigger than the simulation noise on it.

        A tool that prints 0.03 wins to two decimals when its own measured spread is 0.062 is
        inventing precision. `scripts/measure_swap_noise.py` is the gate that established the
        number and re-establishes it if the simulator changes.
        """
        return self.simulated and abs(self.delta_wins) > PAIRED_DELTA_NOISE


#: Standard DEVIATION of a paired delta at 2,000 sims, measured by
#: `scripts/measure_swap_noise.py` on a synthetic 16-team league. (The standard error over six
#: seeds is 0.025; the sd is the conservative choice for "can I tell this from noise on one
#: run", which is the question a single invocation is asking. An earlier comment called this a
#: standard error, which is a different quantity.)
#:
#: Deltas smaller than this are not distinguishable from simulation noise, and a caller
#: printing them to two decimals is inventing precision.
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
    week: int | None = None,
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

    **The pool is injury-adjusted first.** `project_league_standings` knows nothing about
    `injury_status`, so handing it the raw pool simulates a player on IR at full season
    strength — which made the headline number health-blind on the exact case this tool was
    built for. `week` sets the horizon the adjustment is applied over; omitted, no adjustment
    is made and the docstring of the result says so rather than implying one.

    Candidates whose player is not in the pool come back with `simulated=False` rather than
    being dropped from the list. Silently omitting them printed them identically to a candidate
    that simulated at exactly 0.00, which is the row where the answer is most uncertain.
    """
    adjusted = pool if week is None else _injury_adjusted_pool(pool, payload, id_map, week=week)
    baseline = _standings_row(
        project_league_standings(
            payload,
            adjusted,
            id_map,
            availability,
            params,
            season=season,
            n_sims=n_sims,
            rng=np.random.default_rng(seed),
        ),
        my_team_id,
    )

    projectable = set(adjusted["gsis_id"].astype(str))
    espn_to_gsis = espn_gsis_crosswalk(id_map)
    padded = _payload_with_filler(payload, my_team_id=my_team_id)

    impacts: list[SwapImpact] = []
    for candidate in candidates:
        gsis = espn_to_gsis.get(str(candidate.player_id))
        if gsis is None or gsis not in projectable:
            impacts.append(
                SwapImpact(
                    candidate=candidate,
                    delta_wins=0.0,
                    delta_playoff_pct=0.0,
                    delta_title_pct=0.0,
                    simulated=False,
                )
            )
            continue
        # A free add GROWS the roster, and roster size reaches the simulator -- so the honest
        # baseline for one is a roster of the same size with an inert filler in the spot. Using
        # the unpadded baseline made every free add an unpaired comparison, carrying roughly
        # twice the noise while being printed against the paired threshold.
        base = (
            baseline
            if candidate.drop_player_id is not None
            else _standings_row(
                project_league_standings(
                    padded,
                    adjusted,
                    id_map,
                    availability,
                    params,
                    season=season,
                    n_sims=n_sims,
                    rng=np.random.default_rng(seed),
                ),
                my_team_id,
            )
        )
        swapped = _payload_with_swap(
            payload if candidate.drop_player_id is not None else padded,
            candidate,
            my_team_id=my_team_id,
        )
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
                delta_wins=after["projected_wins"] - base["projected_wins"],
                delta_playoff_pct=after["make_playoffs_pct"] - base["make_playoffs_pct"],
                delta_title_pct=after["champ_pct"] - base["champ_pct"],
            )
        )

    # Simulated candidates first, best delta first. An unsimulated one has no delta to rank on,
    # so it sorts last on its lineup gain rather than being interleaved at a fictitious zero.
    impacts.sort(key=lambda i: (not i.simulated, -i.delta_wins, -i.candidate.lineup_gain))
    return impacts


def _injury_adjusted_pool(
    pool: pd.DataFrame, payload: Mapping[str, Any], id_map: pd.DataFrame, *, week: int
) -> pd.DataFrame:
    """The pool with each rostered player's remaining points scaled by his injury status.

    The simulator has no concept of an injury, so this is the only place a designation can
    reach it. Without it, dropping a player on IR and dropping the same player healthy produce
    identical expected wins -- and "my starter just got hurt, who do I add" is the question the
    whole tool was built to answer.

    Only ROSTERED players carry a status (it comes off `teams[].roster.entries`); everyone else
    is left alone, which is correct — a free agent's designation is applied to his WEEKLY
    projection in stage 1, over a horizon of one week, and applying it again here would
    double-count.
    """
    from projections.ingest.espn_league import parse_rosters

    rosters = parse_rosters(dict(payload))
    if rosters.empty:
        return pool
    crosswalk = espn_gsis_crosswalk(id_map)
    games_left = max(SEASON_GAMES - (week - 1), 0)
    factors: dict[str, float] = {}
    for _, player in rosters.iterrows():
        gsis = crosswalk.get(str(_entry_id(player)))
        if gsis is None:
            continue
        status, _ = parse_injury_status(player.get("injury_status"))
        if status is InjuryStatus.ACTIVE or status.is_healthy:
            continue
        factors[gsis] = season_multiplier(status, games_remaining=games_left)
    if not factors:
        return pool
    out = pool.copy()
    scale = out["gsis_id"].astype(str).map(factors).fillna(1.0)
    out["season_mean_fpts"] = out["season_mean_fpts"] * scale
    return out


def _entry_id(player: Mapping[str, Any]) -> int:
    raw = player.get("player_id", 0)
    return 0 if raw is None or pd.isna(raw) else int(float(raw))


def _payload_with_filler(payload: Mapping[str, Any], *, my_team_id: int) -> dict[str, Any]:
    """The payload with one inert extra player on my roster.

    The baseline for a FREE add, so both sides of that comparison have the same roster size.
    The filler resolves to no gsis, so `rosters_to_slots` drops him and he contributes nothing
    -- he exists only to keep the draw sequence aligned.
    """
    padded = copy.deepcopy(dict(payload))
    for team in padded.get("teams", []) or []:
        if int(team.get("id", 0) or 0) != my_team_id:
            continue
        roster = team.setdefault("roster", {}).setdefault("entries", [])
        roster.append(
            {
                "lineupSlotId": 20,
                "playerId": _FILLER_PLAYER_ID,
                "playerPoolEntry": {
                    "player": {
                        "id": _FILLER_PLAYER_ID,
                        "fullName": "(open roster spot)",
                        "defaultPositionId": 3,
                        "proTeamId": 0,
                    }
                },
            }
        )
    return padded


#: An ESPN id no real player has, so the filler cannot resolve through the id_map.
_FILLER_PLAYER_ID = -1


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
        # A free add replaces the FILLER that `_payload_with_filler` put in the open spot, so
        # the roster is the same size on both sides of the comparison and the draw sequence
        # stays aligned. Appending is what made every free add an unpaired comparison.
        target = (
            candidate.drop_player_id if candidate.drop_player_id is not None else _FILLER_PLAYER_ID
        )
        for index, existing in enumerate(roster):
            if _entry_player_id(existing) == target:
                # In place, and inheriting the slot: the simulator sets its own lineup, but
                # keeping the slot means a bench-for-bench swap leaves the payload otherwise
                # identical.
                entry["lineupSlotId"] = existing.get("lineupSlotId", 20)
                roster[index] = entry
                replaced = True
                break
        if not replaced:
            # Nothing to replace means the caller passed a drop that is not on the roster, or a
            # free add without the padded baseline. Appending would change the roster size and
            # silently unpair the comparison, which is the failure a no-op swap reading 0.05
            # wins exposed -- so it is refused rather than absorbed.
            raise ProjectionInputError(
                f"cannot place {candidate.player}: player {target} is not on team "
                f"{my_team_id}'s roster, so the swap has no slot to occupy and the paired "
                "comparison would be against a different-sized roster."
            )
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
                "defaultPositionId": ESPN_POSITION_IDS.get(candidate.position, 3),
                "proTeamId": 0,
            }
        },
    }


def _entry_player_id(entry: Mapping[str, Any]) -> int:
    player = (entry.get("playerPoolEntry", {}) or {}).get("player", {}) or {}
    return int(player.get("id", entry.get("playerId", 0)) or 0)
