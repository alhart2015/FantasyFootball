"""Rest-of-season projections: the subtraction, the scaling, and the guard.

Step 3 of the projected-standings spec. Three things can go wrong here and only one of them
is loud:

- **The scaling.** `sample_weekly_points` divides `season_mean_fpts` by a fixed `SEASON_GAMES`
  to get a per-game mean. Writing a remaining-points total straight into that column spreads
  it over a whole season instead of the weeks left, shrinking every projection by roughly
  `weeks_remaining / SEASON_GAMES`. The numbers stay plausible, which is what makes it
  dangerous, so it is pinned arithmetically here.
- **The assumption.** The subtraction takes a provider's in-season "season total" to include
  games already played. If it is already rest-of-season, subtracting double-counts. That
  cannot be observed before Week 1, so the detector for it is tested instead.
- **Missing players.** Rookies with synthetic ids and mid-season pickups have no fresh
  projection and must fall back rather than vanish or zero out.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.assistant.performance_variance import SEASON_GAMES
from projections.midseason.rest_of_season import (
    RosDiagnostics,
    rest_of_season_points,
    rest_of_season_pool,
)
from projections.schemas import _PYARROW_STR


def _pool(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """A real `VorpTableSchema` frame, not a subset of one.

    `rest_of_season_pool` validates what it returns, so its input has to satisfy the same
    contract the CLI actually hands it. The earlier fixture carried only the four columns the
    function reads, which quietly under-specified the boundary being tested.
    """
    replacement = 80.0
    frame = pd.DataFrame(
        [
            {
                "gsis_id": gsis,
                "full_name": name,
                "position": "RB",
                "season_mean_fpts": preseason,
                "vorp": preseason - replacement,
                "replacement_fpts": replacement,
                "is_rookie": False,
            }
            for gsis, name, preseason in rows
        ]
    )
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    frame["position"] = frame["position"].astype(_PYARROW_STR)
    return frame


# --- the subtraction -----------------------------------------------------------------------


def test_half_a_season_scored_leaves_half_projected() -> None:
    points, clamped, fallback = rest_of_season_points(
        200.0, 100.0, preseason_points=180.0, weeks_remaining=7
    )
    assert (points, clamped, fallback) == (100.0, False, False)


def test_outscoring_the_projection_clamps_to_zero_and_is_reported() -> None:
    """Real and ordinary late in a season. Clamped rather than allowed to subtract from a
    lineup, and counted so it can be seen."""
    points, clamped, fallback = rest_of_season_points(
        150.0, 175.0, preseason_points=180.0, weeks_remaining=3
    )
    assert (points, clamped, fallback) == (0.0, True, False)


def test_a_player_with_no_fresh_projection_prorates_the_preseason_number() -> None:
    """Rookies with synthetic `99-` ids and mid-season pickups. Prorating is the honest
    fallback — there is nothing newer to use — but it must be counted, not silent.

    The denominator is SEASON_GAMES, not the fantasy regular season: `preseason_points` is a
    SEASON_GAMES-game total and `rest_of_season_pool` re-expands over SEASON_GAMES, so any
    other denominator makes the round trip a scale change instead of a no-op.
    """
    points, clamped, fallback = rest_of_season_points(
        None, 0.0, preseason_points=float(SEASON_GAMES) * 10.0, weeks_remaining=7
    )
    assert points == pytest.approx(70.0)  # 10/game pace x 7 weeks left
    assert (clamped, fallback) == (False, True)


# --- the scaling ---------------------------------------------------------------------------


def test_the_column_is_a_full_season_equivalent_pace_not_the_remaining_total() -> None:
    """The subtle one.

    A player with 100 points left over 10 weeks is on a 10/game pace. The variance model
    recovers a per-game mean by dividing `season_mean_fpts` by SEASON_GAMES, so the column has
    to carry `10 * SEASON_GAMES`, not `100`. Writing the raw remaining total would give the
    model `100 / 17 = 5.9`/game — a 41% understatement that still looks like a real number.
    """
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh_season_points={"00-0000001": 200.0},
        points_to_date={"00-0000001": 100.0},
        weeks_remaining=10,
    )
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(10.0 * SEASON_GAMES)
    # And the round trip the model actually performs recovers the pace we meant.
    assert pool.loc[0, "season_mean_fpts"] / SEASON_GAMES == pytest.approx(10.0)


def test_fewer_weeks_remaining_raises_the_pace_for_the_same_remaining_points() -> None:
    """Same 100 points left, half the weeks to score them in, so twice the per-week pace."""
    fresh = {"00-0000001": 200.0}
    scored = {"00-0000001": 100.0}
    ten, _ = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh,
        scored,
        weeks_remaining=10,
    )
    five, _ = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh,
        scored,
        weeks_remaining=5,
    )
    assert five.loc[0, "season_mean_fpts"] == pytest.approx(2 * ten.loc[0, "season_mean_fpts"])


def test_a_finished_regular_season_zeroes_the_pool() -> None:
    """Nothing left to project; the caller's locked records already carry the whole answer."""
    pool, diag = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh_season_points={"00-0000001": 200.0},
        points_to_date={"00-0000001": 200.0},
        weeks_remaining=0,
    )
    assert (pool["season_mean_fpts"] == 0.0).all()
    assert diag.n_players == 1


# --- the guard -----------------------------------------------------------------------------


def test_diagnostics_are_quiet_when_nothing_needed_papering_over() -> None:
    _, diag = rest_of_season_pool(
        _pool([("00-0000001", "A", 180.0), ("00-0000002", "B", 160.0)]),
        fresh_season_points={"00-0000001": 200.0, "00-0000002": 150.0},
        points_to_date={"00-0000001": 50.0, "00-0000002": 40.0},
        weeks_remaining=7,
    )
    assert (diag.n_clamped, diag.n_fallback) == (0, 0)
    assert diag.warning() is None


def test_a_clamped_player_is_named_in_the_warning() -> None:
    """A quietly-zeroed roster looks like a bad team rather than a bad ingest."""
    _, diag = rest_of_season_pool(
        _pool([("00-0000001", "Overachiever", 180.0), ("00-0000002", "B", 160.0)]),
        fresh_season_points={"00-0000001": 100.0, "00-0000002": 150.0},
        points_to_date={"00-0000001": 150.0, "00-0000002": 40.0},
        weeks_remaining=7,
    )
    assert diag.n_clamped == 1
    warning = diag.warning()
    assert warning is not None
    assert "Overachiever" in warning


def test_wholesale_clamping_is_flagged_as_probable_double_counting() -> None:
    """The detector for the assumption that cannot be verified before Week 1.

    If a provider already reports rest-of-season points, subtracting actuals drives a large
    share of the pool to zero at once. One clamped player is ordinary; a third of the league
    is a bug, and the warning has to say so rather than reading as routine.
    """
    rows = [(f"00-000000{i}", f"P{i}", 180.0) for i in range(1, 7)]
    # Four of six already "outscored" their projection — the double-counting signature.
    fresh = {f"00-000000{i}": (50.0 if i <= 4 else 200.0) for i in range(1, 7)}
    to_date = {f"00-000000{i}": 120.0 for i in range(1, 7)}
    _, diag = rest_of_season_pool(
        _pool(rows),
        fresh_season_points=fresh,
        points_to_date=to_date,
        weeks_remaining=7,
    )
    assert diag.n_clamped == 4
    assert diag.looks_like_double_counting
    warning = diag.warning()
    assert warning is not None
    assert "double-count" in warning.lower()


def test_one_clamped_player_is_not_flagged_as_double_counting() -> None:
    """The complement: the detector must not cry wolf on an ordinary overachiever."""
    rows = [(f"00-000000{i}", f"P{i}", 180.0) for i in range(1, 7)]
    fresh = {f"00-000000{i}": (50.0 if i == 1 else 200.0) for i in range(1, 7)}
    to_date = {f"00-000000{i}": 120.0 for i in range(1, 7)}
    _, diag = rest_of_season_pool(
        _pool(rows),
        fresh_season_points=fresh,
        points_to_date=to_date,
        weeks_remaining=7,
    )
    assert diag.n_clamped == 1
    assert not diag.looks_like_double_counting
    warning = diag.warning()
    assert warning is not None
    assert "double-count" not in warning.lower()


def test_missing_players_are_counted_as_fallbacks() -> None:
    _, diag = rest_of_season_pool(
        _pool([("00-0000001", "Known", 180.0), ("99-1234567", "Rookie", 120.0)]),
        fresh_season_points={"00-0000001": 200.0},
        points_to_date={"00-0000001": 50.0},
        weeks_remaining=7,
    )
    assert diag.n_fallback == 1
    warning = diag.warning()
    assert warning is not None
    assert "prorated preseason" in warning


def test_empty_diagnostics_do_not_claim_double_counting() -> None:
    assert not RosDiagnostics(n_players=0).looks_like_double_counting


# --- the horizon invariant ------------------------------------------------------------------


def test_the_fallback_is_a_pace_no_op_not_a_21_percent_raise() -> None:
    """The invariant both ROS branches must satisfy: a player with no fresh projection and no
    points scored is on exactly his preseason pace, so his `season_mean_fpts` must come out
    unchanged.

    It did not. The fallback prorated over `reg_weeks` and `rest_of_season_pool` then
    re-expressed over `SEASON_GAMES`, composing to `preseason * SEASON_GAMES / reg_weeks` --
    17/14 = 1.21x for this league, 17/13 = 1.31x on the default calendar. Two horizons inside
    one function. And because the CLI passed an empty `fresh_season_points`, *every* player
    took this branch, so every team scored ~21% more than its roster projects while the
    standings looked completely reasonable.
    """
    preseason = 238.0
    pool, diag = rest_of_season_pool(
        _pool([("00-0000001", "Untouched", preseason)]),
        fresh_season_points={},  # nobody has a fresh projection -> fallback for all
        points_to_date={},
        weeks_remaining=10,
    )
    assert diag.n_fallback == 1
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(preseason)


@pytest.mark.parametrize("reg_weeks", [13, 14, 17])
@pytest.mark.parametrize("weeks_remaining", [1, 5, 10, 14])
def test_the_fallback_no_op_holds_for_every_calendar(reg_weeks: int, weeks_remaining: int) -> None:
    """A pace no-op cannot depend on how many weeks are left or how long the season is. The
    old arithmetic made it depend on both."""
    if weeks_remaining > reg_weeks:
        pytest.skip("weeks_remaining cannot exceed the regular season")
    preseason = 200.0
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Untouched", preseason)]),
        fresh_season_points={},
        points_to_date={},
        weeks_remaining=weeks_remaining,
    )
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(preseason)


def test_a_fresh_projection_with_nothing_scored_is_also_a_pace_no_op() -> None:
    """The other branch, same invariant: a fresh 17-game total with zero points scored means
    the player is on exactly that pace, so the column must come back as that number."""
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Fresh", 180.0)]),
        fresh_season_points={"00-0000001": 204.0},
        points_to_date={"00-0000001": 0.0},
        weeks_remaining=SEASON_GAMES,
    )
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(204.0)


# --- guards that actually guard ---------------------------------------------------------


def test_a_legitimately_zero_projection_is_not_counted_as_a_clamp() -> None:
    """The detector must not cry wolf.

    A full VORP pool routinely holds hundreds of deep-bench players a provider projects at
    0.0. Treating `0.0 - 0.0 <= 0` as a clamp put them all in the count, so a healthy ingest
    tripped the "MORE THAN A THIRD OF THE POOL CLAMPED" alarm -- and once that fires
    spuriously, the real signal it exists to carry is worthless.

    A clamp means "this player has ALREADY OUTSCORED the projection", which requires points
    on the board.
    """
    rows = [(f"00-000000{i}", f"P{i}", 180.0) for i in range(1, 7)]
    _, diag = rest_of_season_pool(
        _pool(rows),
        fresh_season_points=dict.fromkeys((f"00-000000{i}" for i in range(1, 7)), 0.0),
        points_to_date={},
        weeks_remaining=7,
    )
    assert diag.n_clamped == 0
    assert not diag.looks_like_double_counting


def test_outscoring_the_projection_is_still_counted_as_a_clamp() -> None:
    """The complement: a real overachiever must still register."""
    _, diag = rest_of_season_pool(
        _pool([("00-0000001", "Overachiever", 180.0)]),
        fresh_season_points={"00-0000001": 100.0},
        points_to_date={"00-0000001": 150.0},
        weeks_remaining=7,
    )
    assert diag.n_clamped == 1


def test_a_pool_projected_implausibly_near_zero_is_reported() -> None:
    """The second guard spec 3.1 required and the code never implemented.

    The inverted-assumption failure has a near-miss form: a provider reporting a tiny but
    POSITIVE rest-of-season figure. Nothing clamps, so `n_clamped` stays 0,
    `looks_like_double_counting` stays False, `warning()` returns None -- and the entire pool
    is silently near-zeroed. `_IMPLAUSIBLY_SMALL_ROS` was defined for exactly this and never
    referenced.
    """
    rows = [(f"00-000000{i}", f"P{i}", 200.0) for i in range(1, 7)]
    _, diag = rest_of_season_pool(
        _pool(rows),
        fresh_season_points=dict.fromkeys((f"00-000000{i}" for i in range(1, 7)), 0.4),
        points_to_date={},
        weeks_remaining=7,
    )
    assert diag.n_near_zero == 6
    warning = diag.warning()
    assert warning is not None
    assert "implausibly" in warning.lower()


def test_a_healthy_pool_reports_no_near_zero_players() -> None:
    rows = [(f"00-000000{i}", f"P{i}", 200.0) for i in range(1, 7)]
    _, diag = rest_of_season_pool(
        _pool(rows),
        fresh_season_points=dict.fromkeys((f"00-000000{i}" for i in range(1, 7)), 200.0),
        points_to_date={},
        weeks_remaining=7,
    )
    assert diag.n_near_zero == 0
    assert diag.warning() is None


def test_a_missing_full_name_does_not_crash_the_diagnostics() -> None:
    """`getattr(row, "full_name", gsis) or gsis` evaluates `bool(pd.NA)`, which raises
    TypeError. Name columns here are pyarrow-backed nullable strings, so pd.NA is admissible
    -- and the crash would land in the one code path whose entire purpose is making a failure
    visible."""
    frame = _pool([("00-0000001", "Named", 180.0), ("00-0000002", "x", 180.0)])
    frame.loc[1, "full_name"] = pd.NA
    _, diag = rest_of_season_pool(
        frame,
        fresh_season_points={"00-0000001": 10.0, "00-0000002": 10.0},
        points_to_date={"00-0000001": 500.0, "00-0000002": 500.0},
        weeks_remaining=7,
    )
    assert diag.n_clamped == 2
    assert diag.warning() is not None
