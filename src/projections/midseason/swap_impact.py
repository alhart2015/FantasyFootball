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
from projections.ingest.espn_league import (
    ESPN_POSITION_IDS,
    espn_gsis_crosswalk,
    parse_rosters,
)
from projections.midseason.injuries import season_multiplier
from projections.midseason.standings import (
    ProjectionInputError,
    StandingsRun,
    project_league_standings,
)
from projections.midseason.waivers import Candidate, player_id
from projections.schemas import parse_injury_status


@dataclass(frozen=True)
class SwapImpact:
    """A candidate, re-simulated with the swap applied.

    The deltas are against a baseline computed once at the same seed — see `simulate_swaps`.
    """

    candidate: Candidate
    delta_wins: float
    delta_playoff_pct: float
    delta_title_pct: float
    #: False when the swap could not be simulated at all. The deltas are then zero and MEAN
    #: nothing, which is a different fact from a simulated zero and must not print the same.
    simulated: bool = True
    #: Why, when it could not. Two causes -- one about the data (nobody projects him), one
    #: about the roster (full, and nothing on it can be dropped) -- and they are different
    #: facts, so a caller that prints only the first asserts something false whenever it was
    #: the second. **Required whenever `simulated` is False**, checked below: the CLI
    #: interpolates it into a sentence, and an empty one renders "NOT SIMULATED — ."
    not_simulated_because: str = ""

    def __post_init__(self) -> None:
        if not self.simulated and not self.not_simulated_because:
            raise ValueError(
                "an unsimulated SwapImpact must say why: the reason is printed to the reader, "
                "and the two causes (no projection / no legal drop) are different facts."
            )

    @property
    def paired(self) -> bool:
        """Whether the two runs simulated same-sized rosters.

        False for a free add, which grows the roster: the draws stop lining up and the delta
        carries the unpaired error rather than the paired one. Still the best estimate
        available, just less certain, and `beats_noise` holds it to the wider bar.

        **Derived, not stored.** It has to agree with what `_payload_with_swap` actually did,
        and a stored field can be constructed disagreeing with the candidate it describes --
        which the default on the unsimulated branch already did.
        """
        return self.candidate.drop_player_id is not None

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
    def noise_floor(self) -> float:
        """The spread this delta has to clear to count as signal.

        **One owner.** The verdict and the number a caller prints beside it were computed
        independently from the same inputs, so a change to one would silently leave the other
        quoting a floor the verdict was never taken against — the same disagreement `paired`
        was made a property to prevent.
        """
        return PAIRED_DELTA_NOISE if self.paired else UNPAIRED_DELTA_NOISE

    @property
    def beats_noise(self) -> bool:
        """Whether the delta is bigger than the simulation noise on it.

        A tool that prints 0.03 wins to two decimals when its own measured spread is 0.062 is
        inventing precision. `scripts/measure_swap_noise.py` established `PAIRED_DELTA_NOISE`
        and re-checks that pairing still helps and that runs stay deterministic. It does NOT
        assert either constant directly, so a simulator change that doubled the noise would
        leave both stale while the script still exited 0. `UNPAIRED_DELTA_NOISE` is derived
        rather than measured; see its own comment.

        An UNPAIRED delta -- a free add, which grows the roster so the draws stop lining up --
        is held to the wider unpaired bar. Printing it against the paired one would mark noise
        as signal on exactly the recommendations that need no justification to act on.
        """
        return self.simulated and abs(self.delta_wins) > self.noise_floor


#: Standard DEVIATION of a paired delta at 2,000 sims, measured by
#: `scripts/measure_swap_noise.py` on a synthetic 16-team league. (The standard error over six
#: seeds is 0.025; the sd is the conservative choice for "can I tell this from noise on one
#: run", which is the question a single invocation is asking. An earlier comment called this a
#: standard error, which is a different quantity.)
#:
#: Deltas smaller than this are not distinguishable from simulation noise, and a caller
#: printing them to two decimals is inventing precision.
PAIRED_DELTA_NOISE = 0.062

#: The same spread when the two runs do NOT share their draws.
#:
#: **Derived, not measured.** `measure_swap_noise.py` reports it as `sd * sqrt(2)` from
#: same-size runs, which is what two fully independent errors compose to; it never simulates a
#: roster that gained a player, which is the regime a free add is actually in. The true figure
#: is somewhere in `[PAIRED_DELTA_NOISE, 0.127]` and this takes the conservative end, so a
#: free-add delta has to be larger before the tool will call it signal. Measuring it properly
#: means teaching that script to change a roster's size.
UNPAIRED_DELTA_NOISE = 0.127


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
    noise cancels. Measured: the paired difference carries sd 0.062; the ~0.127 unpaired
    figure is DERIVED as `sd * sqrt(2)`, not measured.

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

    impacts: list[SwapImpact] = []
    for candidate in candidates:
        gsis = espn_to_gsis.get(str(candidate.player_id))
        # An impossible move: the roster is full and nothing droppable could be priced, so
        # there is no legal way to make this add. Simulating it as a free one -- which the
        # earlier `drop_player_id is None` branch did -- reported the value of an overfilled
        # roster as the objective, directly under a line saying NO DROP FOUND.
        impossible = candidate.drop_player_id is None and not candidate.needs_no_drop
        unprojected = gsis is None or gsis not in projectable
        if impossible or unprojected:
            impacts.append(
                SwapImpact(
                    candidate=candidate,
                    delta_wins=0.0,
                    delta_playoff_pct=0.0,
                    delta_title_pct=0.0,
                    simulated=False,
                    # `impossible` first. When a move is BOTH illegal and unprojectable,
                    # the illegal half is the actionable one -- no amount of better data makes
                    # a full roster with nothing droppable into a legal add.
                    not_simulated_because=(
                        "your active roster is full and nobody on it can be dropped"
                        if impossible
                        else "no season projection for him"
                    ),
                )
            )
            continue
        swapped = _payload_with_swap(payload, candidate, my_team_id=my_team_id)
        after = _standings_row(
            project_league_standings(
                swapped,
                # `adjusted`, NOT `pool`. Reading the raw pool here while the baseline read the
                # adjusted one put the whole league's injury discount into every delta as an
                # offset instead of cancelling -- which is worse than the health-blindness it
                # was meant to fix, because health-blind is at least unbiased.
                adjusted,
                id_map,
                availability,
                params,
                season=season,
                n_sims=n_sims,
                # A FRESH generator at the SAME seed, not the one the baseline consumed.
                # Reusing a spent generator gives every candidate a different draw sequence
                # from the baseline, which is the unpaired case wearing paired clothing.
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

    # Simulated candidates first, best delta first. An unsimulated one has no delta to rank on,
    # so it sorts last on its lineup gain rather than being interleaved at a fictitious zero.
    #
    # Paired and unpaired deltas ARE ranked against each other, which mixes two precisions: a
    # free add worth nothing can land at +0.10 from noise alone and outrank a swap genuinely
    # worth +0.08. Accepted rather than hidden -- each row prints which floor it was judged
    # against, so a reader can see it. Ranking them apart would need a common scale the two
    # estimates do not have.
    impacts.sort(key=lambda i: (not i.simulated, -i.delta_wins, -i.candidate.lineup_gain))
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
    rosters = parse_rosters(dict(payload))
    crosswalk = espn_gsis_crosswalk(id_map)
    games_left = max(SEASON_GAMES - (week - 1), 0)
    factors: dict[str, float] = {}
    for _, player in rosters.iterrows():
        gsis = crosswalk.get(str(player_id(player)))
        if gsis is None:
            continue
        status, _ = parse_injury_status(player.get("injury_status"))
        if status.is_healthy:
            continue
        factors[gsis] = season_multiplier(status, games_remaining=games_left)
    if not factors:
        return pool
    out = pool.copy()
    scale = out["gsis_id"].astype(str).map(factors).fillna(1.0)
    out["season_mean_fpts"] = out["season_mean_fpts"] * scale
    return out


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
        # A free add legitimately grows the roster; anything else replaces in place, so the
        # draw sequence stays aligned for the case where pairing is achievable.
        if candidate.drop_player_id is None:
            roster.append(entry)
            continue
        for index, existing in enumerate(roster):
            if _entry_player_id(existing) == candidate.drop_player_id:
                # In place, and inheriting the slot: the simulator sets its own lineup, but
                # keeping the slot means a bench-for-bench swap leaves the payload otherwise
                # identical.
                entry["lineupSlotId"] = existing.get("lineupSlotId", 20)
                roster[index] = entry
                replaced = True
                break
        if not replaced:
            # A named drop who is not on the roster is a caller error. Appending anyway would
            # change the roster size and silently unpair the comparison -- the failure a no-op
            # swap reading 0.05 wins exposed -- so it is refused rather than absorbed.
            raise ProjectionInputError(
                f"cannot place {candidate.player}: player {candidate.drop_player_id} is not "
                f"on team {my_team_id}'s roster, so the swap has no slot to occupy and the "
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
