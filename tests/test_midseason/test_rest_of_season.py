"""Rest-of-season projections: the subtraction, the scaling, and the guard.

Step 3 of the projected-standings spec. Three things can go wrong here and only one of them
is loud:

- **The scaling.** `sample_weekly_points` divides `season_mean_fpts` by a fixed `SEASON_GAMES`
  to get a per-game mean. Writing a remaining-points total straight into that column spreads
  it over a whole season instead of the weeks left, shrinking every projection by roughly
  `games_remaining / SEASON_GAMES`. The numbers stay plausible, which is what makes it
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
    remaining_totals,
    rest_of_season_points,
    rest_of_season_pool,
)
from projections.schemas import _PYARROW_STR, VorpTableSchema


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
        200.0, 100.0, preseason_points=180.0, games_remaining=7
    )
    assert (points, clamped, fallback) == (100.0, False, False)


def test_outscoring_the_projection_clamps_to_zero_and_is_reported() -> None:
    """Real and ordinary late in a season. Clamped rather than allowed to subtract from a
    lineup, and counted so it can be seen."""
    points, clamped, fallback = rest_of_season_points(
        150.0, 175.0, preseason_points=180.0, games_remaining=3
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
        None, 0.0, preseason_points=float(SEASON_GAMES) * 10.0, games_remaining=7
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
        games_remaining=10,
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
        games_remaining=10,
    )
    five, _ = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh,
        scored,
        games_remaining=5,
    )
    assert five.loc[0, "season_mean_fpts"] == pytest.approx(2 * ten.loc[0, "season_mean_fpts"])


def test_a_finished_regular_season_zeroes_the_pool() -> None:
    """Nothing left to project; the caller's locked records already carry the whole answer."""
    pool, diag = rest_of_season_pool(
        _pool([("00-0000001", "Test Back", 180.0)]),
        fresh_season_points={"00-0000001": 200.0},
        points_to_date={"00-0000001": 200.0},
        games_remaining=0,
    )
    assert (pool["season_mean_fpts"] == 0.0).all()
    assert diag.n_players == 1


# --- the guard -----------------------------------------------------------------------------


def test_diagnostics_are_quiet_when_nothing_needed_papering_over() -> None:
    _, diag = rest_of_season_pool(
        _pool([("00-0000001", "A", 180.0), ("00-0000002", "B", 160.0)]),
        fresh_season_points={"00-0000001": 200.0, "00-0000002": 150.0},
        points_to_date={"00-0000001": 50.0, "00-0000002": 40.0},
        games_remaining=7,
    )
    assert (diag.n_clamped, diag.n_fallback) == (0, 0)
    assert diag.warning() is None


def test_a_clamped_player_is_named_in_the_warning() -> None:
    """A quietly-zeroed roster looks like a bad team rather than a bad ingest."""
    _, diag = rest_of_season_pool(
        _pool([("00-0000001", "Overachiever", 180.0), ("00-0000002", "B", 160.0)]),
        fresh_season_points={"00-0000001": 100.0, "00-0000002": 150.0},
        points_to_date={"00-0000001": 150.0, "00-0000002": 40.0},
        games_remaining=7,
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
        games_remaining=7,
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
        games_remaining=7,
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
        games_remaining=7,
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
        games_remaining=10,
    )
    assert diag.n_fallback == 1
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(preseason)


@pytest.mark.parametrize("games_remaining", [1, 5, 10, 14, 17])
def test_the_fallback_no_op_holds_at_every_point_in_the_season(games_remaining: int) -> None:
    """A pace no-op cannot depend on how many games are left. The old arithmetic made it
    depend on that AND on the fantasy season's length, neither of which it should."""
    preseason = 200.0
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Untouched", preseason)]),
        fresh_season_points={},
        points_to_date={},
        games_remaining=games_remaining,
    )
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(preseason)


def test_a_fresh_projection_with_nothing_scored_is_also_a_pace_no_op() -> None:
    """The other branch, same invariant: a fresh 17-game total with zero points scored means
    the player is on exactly that pace, so the column must come back as that number."""
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Fresh", 180.0)]),
        fresh_season_points={"00-0000001": 204.0},
        points_to_date={"00-0000001": 0.0},
        games_remaining=SEASON_GAMES,
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
        games_remaining=7,
    )
    assert diag.n_clamped == 0
    assert not diag.looks_like_double_counting


def test_outscoring_the_projection_is_still_counted_as_a_clamp() -> None:
    """The complement: a real overachiever must still register."""
    _, diag = rest_of_season_pool(
        _pool([("00-0000001", "Overachiever", 180.0)]),
        fresh_season_points={"00-0000001": 100.0},
        points_to_date={"00-0000001": 150.0},
        games_remaining=7,
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
        games_remaining=7,
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
        games_remaining=7,
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
        games_remaining=7,
    )
    assert diag.n_clamped == 2
    assert diag.warning() is not None


# --- the denominator is NFL games remaining, not fantasy weeks remaining --------------------


@pytest.mark.parametrize("snapshot_week", [1, 5, 10, 14, 15])
def test_a_player_exactly_on_pace_stays_on_pace_at_any_point_in_the_season(
    snapshot_week: int,
) -> None:
    """The invariant that caught a regression the first fix batch wrote.

    `fresh - to_date` is points over the remaining **NFL games**; the earlier code divided it
    by the remaining **fantasy regular-season weeks**. Those are different quantities, and the
    mismatch inflated everyone -- measured at 1.21x preseason, 3.40x at week 10 of 14, and 17x
    at week 14.

    A player who has scored exactly his share so far, against an unrevised season total, is on
    exactly that pace for what remains -- whenever you ask.
    """
    mean = 238.0
    games_elapsed = snapshot_week - 1
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "OnPace", mean)]),
        fresh_season_points={"00-0000001": mean},
        points_to_date={"00-0000001": mean * games_elapsed / SEASON_GAMES},
        games_remaining=SEASON_GAMES - games_elapsed,
    )
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(mean)


def test_a_player_scoring_to_pace_keeps_that_pace() -> None:
    """Nine games in, exactly on a 14/game pace: the remaining projection must still be 14 a
    game, i.e. `14 * SEASON_GAMES` in the column."""
    per_game = 14.0
    played, remaining_games = 9, SEASON_GAMES - 9
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "OnPace", per_game * SEASON_GAMES)]),
        fresh_season_points={"00-0000001": per_game * SEASON_GAMES},
        points_to_date={"00-0000001": per_game * played},
        games_remaining=remaining_games,
    )
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(per_game * SEASON_GAMES)


def test_a_revised_down_season_total_lowers_the_remaining_pace() -> None:
    """The realistic shape of a slump: the provider revises the season total down.

    Note what does NOT happen -- underperforming against an *unrevised* total raises the
    remaining pace, because the arithmetic says the player has to make it up. That is correct
    and is the reason the fresh pull matters: a stale season total makes a slumping player
    look stronger, not weaker.
    """
    per_game, played = 14.0, 9
    scored = per_game * played * 0.5  # half pace so far
    revised = scored + per_game * 0.5 * (SEASON_GAMES - played)  # provider marks him down
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Slump", per_game * SEASON_GAMES)]),
        fresh_season_points={"00-0000001": revised},
        points_to_date={"00-0000001": scored},
        games_remaining=SEASON_GAMES - played,
    )
    assert pool.loc[0, "season_mean_fpts"] == pytest.approx(per_game * 0.5 * SEASON_GAMES)


def test_underperforming_against_a_stale_total_raises_the_remaining_pace() -> None:
    """Stated explicitly because it is surprising and it is correct: if the season total is
    not revised, the points a player still 'owes' are squeezed into fewer games."""
    total = 238.0
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Stale", total)]),
        fresh_season_points={"00-0000001": total},
        points_to_date={"00-0000001": total * 9 / SEASON_GAMES * 0.5},
        games_remaining=SEASON_GAMES - 9,
    )
    assert pool.loc[0, "season_mean_fpts"] > total


def test_the_playoff_weeks_are_not_projected_at_zero() -> None:
    """A finished fantasy regular season still has NFL games left, and the bracket needs
    points for them. Keying off remaining fantasy weeks zeroed the whole pool at exactly the
    moment the playoffs are simulated, which handed every playoff matchup to the home side."""
    pool, _ = rest_of_season_pool(
        _pool([("00-0000001", "Playoffs", 238.0)]),
        fresh_season_points={"00-0000001": 238.0},
        points_to_date={"00-0000001": 200.0},
        games_remaining=SEASON_GAMES - 14,
    )
    assert pool.loc[0, "season_mean_fpts"] > 0.0


# --- remaining_totals: the sibling that writes a total rather than a pace -----------------------


def _small_pool() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
            "full_name": pd.Series(["Star RB", "Spent WR"], dtype=_PYARROW_STR),
            "position": pd.Series(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [240.0, 100.0],
            "vorp": [160.0, 20.0],
            "replacement_fpts": [80.0, 80.0],
        }
    )
    return VorpTableSchema.validate(frame)


def test_remaining_totals_writes_a_total_not_a_pace() -> None:
    """The whole reason it exists beside `rest_of_season_pool`. A reader under the header
    "projected points for the rest of the season" wants the points still coming; the simulator
    wants a full-season-equivalent pace. Printing the pace roughly doubles the number by week
    10, which is the bug this pair of functions was split to make impossible."""
    pool, _ = remaining_totals(_small_pool(), {"00-0000001": 100.0}, games_remaining=8)
    by_id = pool.set_index("gsis_id")["season_mean_fpts"]
    assert by_id["00-0000001"] == pytest.approx(140.0), "240 projected minus 100 scored"
    assert by_id["00-0000002"] == pytest.approx(100.0), "nothing scored, nothing subtracted"


def test_remaining_totals_clamps_a_player_who_beat_his_projection_and_says_so() -> None:
    pool, diagnostics = remaining_totals(_small_pool(), {"00-0000002": 150.0}, games_remaining=8)
    assert pool.set_index("gsis_id")["season_mean_fpts"]["00-0000002"] == 0.0
    assert diagnostics.n_clamped == 1
    assert "Spent WR" in (diagnostics.warning() or "")


def test_remaining_totals_quotes_the_cutoff_it_actually_applied() -> None:
    """The near-zero cutoff is scaled by the games left, so at week 10 it is 0.47, not 1. The
    message quoted the unscaled constant -- telling the reader a check ran at a threshold it
    did not use, on the one page a human reads."""
    pool = _small_pool()
    pool["season_mean_fpts"] = [0.3, 0.3]
    _, diagnostics = remaining_totals(pool, {}, games_remaining=8)
    assert diagnostics.n_near_zero == 2
    warning = diagnostics.warning() or ""
    assert "0.47" in warning, warning


def test_remaining_totals_says_so_when_there_is_no_season_left() -> None:
    """`games_remaining <= 0` zeroes every cell, which looks like a bad team rather than an
    exhausted season. Reachable in a league whose regular season runs to 17 weeks or more,
    where `first_unplayed_week` returns `reg_weeks + 1`."""
    pool, diagnostics = remaining_totals(_small_pool(), {}, games_remaining=0)
    assert (pool["season_mean_fpts"] == 0.0).all()
    assert diagnostics.warning() is not None, "a zeroed roster must explain itself"


def test_remaining_totals_refuses_a_pool_with_a_duplicated_player() -> None:
    """Two rows sharing a gsis_id were both scored against whichever came last -- a plausible
    wrong number rather than a failure. `VorpTableSchema` forbids it, but this function is
    public and validates on the way OUT."""
    pool = pd.concat([_small_pool(), _small_pool().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate gsis_id"):
        remaining_totals(pool, {}, games_remaining=8)
