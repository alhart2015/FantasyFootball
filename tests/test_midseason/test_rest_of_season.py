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
    frame = pd.DataFrame(
        [
            {
                "gsis_id": gsis,
                "full_name": name,
                "position": "RB",
                "season_mean_fpts": preseason,
                "is_rookie": False,
            }
            for gsis, name, preseason in rows
        ]
    )
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    return frame


# --- the subtraction -----------------------------------------------------------------------


def test_half_a_season_scored_leaves_half_projected() -> None:
    points, clamped, fallback = rest_of_season_points(
        200.0, 100.0, preseason_points=180.0, weeks_remaining=7, reg_weeks=14
    )
    assert (points, clamped, fallback) == (100.0, False, False)


def test_outscoring_the_projection_clamps_to_zero_and_is_reported() -> None:
    """Real and ordinary late in a season. Clamped rather than allowed to subtract from a
    lineup, and counted so it can be seen."""
    points, clamped, fallback = rest_of_season_points(
        150.0, 175.0, preseason_points=180.0, weeks_remaining=3, reg_weeks=14
    )
    assert (points, clamped, fallback) == (0.0, True, False)


def test_a_player_with_no_fresh_projection_prorates_the_preseason_number() -> None:
    """Rookies with synthetic `99-` ids and mid-season pickups. Prorating is the honest
    fallback — there is nothing newer to use — but it must be counted, not silent."""
    points, clamped, fallback = rest_of_season_points(
        None, 0.0, preseason_points=140.0, weeks_remaining=7, reg_weeks=14
    )
    assert points == pytest.approx(70.0)
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
        reg_weeks=14,
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
        reg_weeks=14,
    )
    five, _ = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh,
        scored,
        weeks_remaining=5,
        reg_weeks=14,
    )
    assert five.loc[0, "season_mean_fpts"] == pytest.approx(2 * ten.loc[0, "season_mean_fpts"])


def test_a_finished_regular_season_zeroes_the_pool() -> None:
    """Nothing left to project; the caller's locked records already carry the whole answer."""
    pool, diag = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh_season_points={"00-0000001": 200.0},
        points_to_date={"00-0000001": 200.0},
        weeks_remaining=0,
        reg_weeks=14,
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
        reg_weeks=14,
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
        reg_weeks=14,
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
        reg_weeks=14,
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
        reg_weeks=14,
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
        reg_weeks=14,
    )
    assert diag.n_fallback == 1
    warning = diag.warning()
    assert warning is not None
    assert "prorated preseason" in warning


def test_empty_diagnostics_do_not_claim_double_counting() -> None:
    assert not RosDiagnostics(n_players=0).looks_like_double_counting
