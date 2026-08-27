"""Stage 2: what a swap is worth in expected wins.

Split out with the module it tests. The pairing tests are the ones that matter -- a paired
simulation that is quietly unpaired reports noise to two decimal places, and nothing about the
output looks wrong.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.midseason.standings import ProjectionInputError
from projections.midseason.swap_impact import SwapImpact, simulate_swaps
from projections.midseason.waivers import Candidate
from projections.schemas import _PYARROW_STR, InjuryStatus, VorpTableSchema
from tests.test_midseason.conftest import (
    MY_TEAM_ID,
    POSITIONS,
    TEAM_IDS,
    espn_payload,
    espn_player_id,
    id_map,
    vorp_pool,
)

#: A free agent nobody rosters, projected above everyone. `vorp_pool` contains only rostered
#: players and gives MY team the strongest roster, so no rostered player is an upgrade -- which
#: is realistic for a league leader and useless for testing that an upgrade registers.
FREE_AGENT_ESPN_ID = 900_001
FREE_AGENT_GSIS = "00-9000001"


def _pool_with_free_agent() -> pd.DataFrame:
    extra = pd.DataFrame(
        [
            {
                "gsis_id": FREE_AGENT_GSIS,
                "full_name": "Waiver Stud",
                "position": "WR",
                "season_mean_fpts": 400.0,
                "vorp": 320.0,
                "replacement_fpts": 80.0,
                "is_rookie": False,
            }
        ]
    )
    frame = pd.concat([vorp_pool(), extra], ignore_index=True)
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    frame["position"] = frame["position"].astype(_PYARROW_STR)
    frame["full_name"] = frame["full_name"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(frame)


def _id_map_with_free_agent() -> pd.DataFrame:
    extra = pd.DataFrame(
        {
            "espn_id": pd.Series([str(FREE_AGENT_ESPN_ID)], dtype=_PYARROW_STR),
            "gsis_id": pd.Series([FREE_AGENT_GSIS], dtype=_PYARROW_STR),
        }
    )
    return pd.concat([id_map(), extra], ignore_index=True)


def _sim_inputs() -> dict[str, object]:
    """The shared synthetic league, plus the pieces `simulate_swaps` needs."""
    pool = _pool_with_free_agent()
    return {
        "payload": espn_payload(played_weeks=0),
        "pool": pool,
        "id_map": _id_map_with_free_agent(),
        "availability": PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
        "params": VarianceParams.load(),
        "season": 2026,
        "my_team_id": MY_TEAM_ID,
        "n_sims": 60,
    }


def _candidate(
    player_id: int, name: str, *, drop_id: int | None = None, position: str = "RB"
) -> Candidate:
    """`drop_id=None` means a FREE add -- a spot is open -- which is what `needs_no_drop` says.
    The third state (roster full, nothing priceable to drop) is built explicitly in the test
    that is about it."""
    return Candidate(
        player_id=player_id,
        player=name,
        position=position,
        nfl_team="KC",
        lineup_gain=5.0,
        projected=15.0,
        injury_status=InjuryStatus.ACTIVE,
        on_waivers=False,
        percent_owned=40.0,
        needs_no_drop=drop_id is None,
        drop_player_id=drop_id,
        drop_player="" if drop_id is None else f"Player {drop_id}",
        drop_cost=10.0,
    )


def test_a_no_op_swap_reports_exactly_zero() -> None:
    """The load-bearing property. Dropping a player and adding the same player back is a
    change of nothing, so a paired simulation must report 0.0 -- not "close to zero".

    If this ever returns a small non-zero number, the two runs are not sharing their draws and
    every delta the tool prints is simulation noise wearing a decimal point.
    """
    inputs = _sim_inputs()
    mine = espn_player_id(MY_TEAM_ID, 0)
    [impact] = simulate_swaps(
        # Same id, same name, same position: literally the roster it already has.
        candidates=[_candidate(mine, "Player 17-0", drop_id=mine, position=POSITIONS[0])],
        **inputs,  # type: ignore[arg-type]
    )
    assert impact.delta_wins == 0.0
    assert impact.delta_playoff_pct == 0.0
    assert impact.delta_title_pct == 0.0


def test_the_same_candidate_twice_reports_the_same_number() -> None:
    """Determinism, which pairing is worthless without. Not approximately the same -- the
    same, because both runs start from a generator seeded identically."""
    inputs = _sim_inputs()
    candidate = _candidate(
        espn_player_id(TEAM_IDS[1], 0), "Somebody", drop_id=espn_player_id(MY_TEAM_ID, 5)
    )
    first = simulate_swaps(candidates=[candidate], **inputs)  # type: ignore[arg-type]
    second = simulate_swaps(candidates=[candidate], **inputs)  # type: ignore[arg-type]
    assert first[0].delta_wins == second[0].delta_wins


def test_adding_a_better_player_helps_and_the_tool_says_so() -> None:
    """The direction check: a 400-point free agent replacing a bench player must raise my
    projected wins, and `helps` must agree with the sign."""
    inputs = _sim_inputs()
    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    [impact] = simulate_swaps(
        candidates=[
            _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR")
        ],
        **inputs,  # type: ignore[arg-type]
    )
    assert impact.delta_wins > 0.0
    assert impact.helps


def test_a_player_the_pool_cannot_project_comes_back_marked_not_simulated() -> None:
    """Two things must both be true, and the first version only managed one.

    He must not be SIMULATED -- a free agent nobody projects would come out as a roster
    downgrade, a confident wrong answer rather than an absence of one. And he must not be
    silently DROPPED from the list either, because then he prints identically to a candidate
    that simulated at exactly 0.00, which is the row where the answer is most uncertain.
    """
    inputs = _sim_inputs()
    [impact] = simulate_swaps(
        candidates=[_candidate(999_999, "Unknown Rookie", drop_id=espn_player_id(MY_TEAM_ID, 5))],
        **inputs,  # type: ignore[arg-type]
    )
    assert not impact.simulated
    assert impact.delta_wins == 0.0, "a placeholder, not a measurement"
    assert not impact.helps, "an unsimulated candidate cannot be said to help"
    assert not impact.beats_noise


def test_unsimulated_candidates_sort_last() -> None:
    """They have no delta to rank on, so interleaving them at a fictitious zero would put an
    unknown above a measured negative."""
    inputs = _sim_inputs()
    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    impacts = simulate_swaps(
        candidates=[
            _candidate(999_999, "Unknown Rookie", drop_id=my_bench_wr),
            _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR"),
        ],
        **inputs,  # type: ignore[arg-type]
    )
    assert [i.candidate.player for i in impacts] == ["Waiver Stud", "Unknown Rookie"]
    assert impacts[0].simulated and not impacts[1].simulated


def test_the_caller_payload_is_not_mutated() -> None:
    """Every candidate must be compared against the ORIGINAL roster. Mutating in place would
    have each swap build on the last, and the deltas would still look plausible."""
    inputs = _sim_inputs()
    payload = cast("dict[str, Any]", inputs["payload"])
    before = len(next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)["roster"]["entries"])
    simulate_swaps(
        candidates=[
            _candidate(espn_player_id(TEAM_IDS[1], i), f"Add {i}", drop_id=None) for i in range(3)
        ],
        **inputs,  # type: ignore[arg-type]
    )
    after = len(next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)["roster"]["entries"])
    assert before == after


def test_impacts_come_back_best_first() -> None:
    inputs = _sim_inputs()
    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    impacts = simulate_swaps(
        candidates=[
            # The weakest team's tight end -- a real downgrade for a league leader.
            _candidate(
                espn_player_id(TEAM_IDS[-1], 5), "Scrub", drop_id=my_bench_wr, position="TE"
            ),
            _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR"),
        ],
        **inputs,  # type: ignore[arg-type]
    )
    assert len(impacts) == 2
    assert impacts[0].delta_wins >= impacts[1].delta_wins
    assert impacts[0].candidate.player == "Waiver Stud", "best first, whatever order they arrive"


# --- the objective, and the pairing it rests on ------------------------------------------------


def test_a_free_add_is_reported_as_unpaired() -> None:
    """A free add GROWS the roster, so the two runs simulate different-sized teams and the draws
    stop lining up -- it is genuinely a noisier estimate than a swap.

    An earlier version padded the baseline with an inert filler to fix that. The filler
    resolved to no gsis and `rosters_to_slots` dropped it before the draw, so the padded
    baseline was bit-identical to the plain one: the fix changed nothing and cost one extra
    full simulation per candidate. Saying the delta is noisier is honest; pretending it is
    paired was not, and it marked noise as signal on exactly the recommendations that need no
    justification to act on.
    """
    inputs = _sim_inputs()
    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    [swap] = simulate_swaps(
        candidates=[
            _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR")
        ],
        **inputs,  # type: ignore[arg-type]
    )
    [free] = simulate_swaps(
        candidates=[_candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=None, position="WR")],
        **inputs,  # type: ignore[arg-type]
    )
    assert swap.paired, "a swap replaces in place, so the rosters stay the same size"
    assert not free.paired, "a free add grows the roster, so the draws stop lining up"
    # And the wider bar is actually applied: a delta between the two floors counts for a swap
    # and does not for a free add.
    between = SwapImpact(
        candidate=free.candidate, delta_wins=0.09, delta_playoff_pct=0.0, delta_title_pct=0.0
    )
    assert SwapImpact(**{**between.__dict__, "paired": True}).beats_noise
    assert not SwapImpact(**{**between.__dict__, "paired": False}).beats_noise


def test_an_injury_does_not_leak_into_every_candidates_delta() -> None:
    """The bug three reviewers found independently, and the one an inequality assertion cannot.

    The baseline read the injury-adjusted pool and the swapped run read the raw one, so every
    delta carried "+ the value of un-injuring my whole league" as a constant offset. Relative
    ORDER survived; the sign, the magnitude, `helps` and `beats_noise` did not -- and on the
    motivating case (my starter got hurt) every candidate came back large and positive.

    The test that catches it compares the SAME swap on a healthy payload and an injured one and
    asserts the difference is small, not merely non-zero: an injury to a player neither side of
    the swap touches should barely move what the swap is worth.
    """
    healthy = _sim_inputs()
    hurt = _sim_inputs()
    payload = cast("dict[str, Any]", hurt["payload"])
    my_team = next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)
    # A player who is neither added nor dropped by the swap under test.
    my_team["roster"]["entries"][0]["playerPoolEntry"]["player"]["injuryStatus"] = "INJURY_RESERVE"

    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    swap = _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR")
    [fit] = simulate_swaps(candidates=[swap], week=5, **healthy)  # type: ignore[arg-type]
    [injured] = simulate_swaps(candidates=[swap], week=5, **hurt)  # type: ignore[arg-type]
    assert abs(fit.delta_wins - injured.delta_wins) < 0.15, (
        f"an unrelated injury moved this swap's value from {fit.delta_wins:.3f} to "
        f"{injured.delta_wins:.3f} -- the discount is leaking into the delta instead of "
        "cancelling between the two runs"
    )


def test_an_injured_player_is_worth_less_to_the_simulator() -> None:
    """The gap that made the headline number health-blind on the branch's motivating case.

    `project_league_standings` knows nothing about `injury_status`, so handing it the raw pool
    simulated a player on IR at full season strength. Dropping him then looked exactly as
    costly as dropping the same player healthy -- and "my starter is hurt, who do I add" is the
    question the whole tool exists for.
    """
    healthy = _sim_inputs()
    hurt = _sim_inputs()
    payload = cast("dict[str, Any]", hurt["payload"])
    my_team = next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)
    my_team["roster"]["entries"][0]["playerPoolEntry"]["player"]["injuryStatus"] = "INJURY_RESERVE"

    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    swap = _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR")
    [fit] = simulate_swaps(candidates=[swap], week=5, **healthy)  # type: ignore[arg-type]
    [injured] = simulate_swaps(candidates=[swap], week=5, **hurt)  # type: ignore[arg-type]
    assert fit.delta_wins != injured.delta_wins, (
        "an IR'd player on my roster must change what the simulator thinks my team is worth"
    )


def test_no_injury_adjustment_without_a_week_to_apply_it_over() -> None:
    """`season_multiplier` needs a horizon. Omitting `week` means NO adjustment rather than an
    adjustment over a guessed number of games -- asserted by showing that an injured payload
    then behaves exactly like a healthy one."""
    healthy = _sim_inputs()
    hurt = _sim_inputs()
    payload = cast("dict[str, Any]", hurt["payload"])
    my_team = next(t for t in payload["teams"] if t["id"] == MY_TEAM_ID)
    my_team["roster"]["entries"][0]["playerPoolEntry"]["player"]["injuryStatus"] = "INJURY_RESERVE"
    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    swap = _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR")
    [fit] = simulate_swaps(candidates=[swap], **healthy)  # type: ignore[arg-type]
    [injured] = simulate_swaps(candidates=[swap], **hurt)  # type: ignore[arg-type]
    assert fit.delta_wins == injured.delta_wins, "no week, no adjustment, no difference"


def test_a_swap_that_cannot_be_placed_is_refused_not_absorbed() -> None:
    """Appending would change the roster size and silently unpair the comparison -- the failure
    a no-op swap reading 0.05 wins exposed. A drop that is not on the roster is a caller error,
    so it raises rather than producing a plausible number."""
    inputs = _sim_inputs()
    swap = _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=777_777, position="WR")
    with pytest.raises(ProjectionInputError, match="not on team"):
        simulate_swaps(candidates=[swap], **inputs)  # type: ignore[arg-type]


def test_helps_and_beats_noise_are_honest_about_what_is_known() -> None:
    """Both are read by the CLI now. An unsimulated candidate cannot be said to help, and a
    delta smaller than the measured noise is not a delta the tool should assert."""
    inputs = _sim_inputs()
    my_bench_wr = espn_player_id(MY_TEAM_ID, 4)
    [good] = simulate_swaps(
        candidates=[
            _candidate(FREE_AGENT_ESPN_ID, "Waiver Stud", drop_id=my_bench_wr, position="WR")
        ],
        **inputs,  # type: ignore[arg-type]
    )
    assert good.simulated and good.helps and good.beats_noise


def test_an_impossible_move_is_never_simulated() -> None:
    """Roster full, nothing droppable could be priced. There is no legal way to make this add,
    so reporting a delta for it -- which the old `drop_player_id is None` branch did, as though
    it were free -- put the value of an overfilled roster under a line saying NO DROP FOUND."""
    inputs = _sim_inputs()
    stuck = Candidate(
        player_id=FREE_AGENT_ESPN_ID,
        player="Waiver Stud",
        position="WR",
        nfl_team="KC",
        lineup_gain=5.0,
        projected=15.0,
        injury_status=InjuryStatus.ACTIVE,
        on_waivers=False,
        percent_owned=40.0,
        needs_no_drop=False,
        drop_player_id=None,
    )
    [impact] = simulate_swaps(candidates=[stuck], **inputs)  # type: ignore[arg-type]
    assert not impact.simulated
    assert impact.delta_wins == 0.0
