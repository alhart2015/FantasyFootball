"""Rest-of-season player projections for the in-season Monte Carlo.

The draft pool carries `season_mean_fpts`, a **preseason full-season** projection. By week 6
it is stale in the way that matters most: it does not know who tore an ACL and who broke out.
This module rewrites it into a rest-of-season figure.

    ros_points = fresh_season_projection - points_scored_to_date

Chosen over prorating the preseason number (blind to everything that has happened) and over
blending actual pace with the preseason projection (more accurate in principle, but its
weighting is a free parameter that deserves its own backtest — a follow-up worth doing once
this exists to compare against). Re-pulling `external_projections` mid-season is a path that
already works, and providers revise season totals weekly, so a fresh pull reflects injuries,
benchings and depth-chart moves without any new modelling.

**The assumption, stated rather than buried.** This takes ESPN's / Sleeper's in-season
"season total" to mean *the full season including games already played*, so subtracting
actuals leaves what is still to come. If a provider instead reports a figure that is already
rest-of-season, the subtraction double-counts and every projection comes out low. **That
cannot be verified until Week 1 has been played** — before kickoff, points-to-date are zero
for everyone and both readings agree. Hence `RosDiagnostics` and the loud clamp below:
the failure is designed to be visible rather than silent, and `rest_of_season_points` is the
single place to change if the assumption turns out to be wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from projections.draft.assistant.performance_variance import SEASON_GAMES
from projections.schemas import VorpTableSchema, display_str

#: Points-over-a-full-season below which a projection is far more likely to be an ingest
#: problem than a real number. Scaled by the games actually remaining at the point of use --
#: see `rest_of_season_pool` -- because the figure it is compared against is a remaining total.
_IMPLAUSIBLY_SMALL_ROS = 1.0


@dataclass(frozen=True)
class RosDiagnostics:
    """What the rest-of-season build had to paper over. Never discard this silently.

    A quietly-zeroed roster looks like a bad team rather than a bad ingest, which is the exact
    failure this exists to make visible.
    """

    n_players: int
    n_clamped: int = 0
    n_fallback: int = 0
    #: Players left with a positive but implausibly small remaining projection. The near-miss
    #: form of the inverted-assumption failure: nothing clamps, so the clamp counter stays at
    #: zero while the whole pool is quietly near-zeroed.
    n_near_zero: int = 0
    clamped_examples: tuple[str, ...] = field(default_factory=tuple)
    #: The cutoff `n_near_zero` was counted against. Carried rather than re-derived in the
    #: message, which quoted the unscaled constant while the code applied
    #: `_IMPLAUSIBLY_SMALL_ROS * games_remaining / SEASON_GAMES` -- so at week 10 the note said
    #: "under 1 points" about a check that used 0.47.
    near_zero_cutoff: float = _IMPLAUSIBLY_SMALL_ROS
    #: The projection that was subtracted from is a PRESEASON one rather than a fresh
    #: mid-season pull. It changes what a wholesale clamp means -- see `warning()`.
    projection_is_preseason: bool = False

    @property
    def looks_like_double_counting(self) -> bool:
        """Heuristic that the season-total assumption is inverted.

        If a provider already reports rest-of-season, subtracting actuals drives a large share
        of the pool to zero at once. One or two clamped players is ordinary (someone genuinely
        outran their projection); a third of the league is a bug.
        """
        return self.n_players > 0 and self.n_clamped / self.n_players > 0.33

    def warning(self) -> str | None:
        """A human-readable problem report, or None when nothing needed papering over."""
        if not self.n_clamped and not self.n_fallback and not self.n_near_zero:
            return None
        parts: list[str] = []
        if self.n_clamped:
            shown = ", ".join(self.clamped_examples)
            parts.append(
                f"{self.n_clamped} of {self.n_players} players had no projected points left "
                f"and were clamped to zero ({shown})"
            )
        if self.n_near_zero:
            parts.append(
                f"{self.n_near_zero} of {self.n_players} players project implausibly near "
                f"zero for the rest of the season (under {self.near_zero_cutoff:.2g} points) "
                "without clamping"
            )
        if self.n_fallback:
            parts.append(
                f"{self.n_fallback} had no fresh projection and fell back to a prorated "
                "preseason number (rookies with synthetic ids, mid-season pickups)"
            )
        message = "; ".join(parts) + "."
        if self.n_near_zero and self.n_near_zero / max(self.n_players, 1) > 0.33:
            message += (
                " A THIRD OF THE POOL PROJECTS TO ALMOST NOTHING. Like a wholesale clamp this "
                "is the shape of a provider already reporting rest-of-season points, but the "
                "near-miss form that clamping alone cannot see. Check one player by hand."
            )
        if self.looks_like_double_counting and self.projection_is_preseason:
            # Two causes, and from a preseason pool the innocent one gets likelier every week:
            # by December a good share of any August projection has simply been beaten. Naming
            # only the provider bug here would be confidently wrong on the page a human reads.
            message += (
                " MORE THAN A THIRD OF THE POOL CLAMPED. Either the projections being "
                "subtracted from are stale — they are preseason numbers, and late in a season "
                "many players have beaten them outright — or the provider is already "
                "reporting rest-of-season points, in which case every projection here is too "
                "low. Re-pull projections; if it persists, check one player by hand."
            )
        elif self.looks_like_double_counting:
            message += (
                " MORE THAN A THIRD OF THE POOL CLAMPED — this is the signature of the "
                "provider already reporting rest-of-season points, in which case subtracting "
                "actuals double-counts and every projection is too low. Check one player by "
                "hand before trusting these standings."
            )
        return message


def _validated(pool: pd.DataFrame) -> pd.DataFrame:
    """`df = SCHEMA.validate(df)` at the module boundary, per the repo convention.

    Reassignment matters: `strict="filter"` returns a NEW frame with extras dropped, so
    validating without taking the result back leaves them in place. `is_rookie` is declared
    optional on the schema rather than stripped and restored here -- a per-column allowlist
    would silently drop the next extra column a caller needs.

    **What this does NOT assert.** Only `season_mean_fpts` is rewritten to a rest-of-season
    figure; `vorp` and `replacement_fpts` are carried through as their preseason values, so a
    consumer reading those off the returned frame gets preseason valuations. The validate is
    here to catch a frame that has lost `gsis_id` or `season_mean_fpts`, not to certify that
    every column means what its name suggests after the rewrite.
    """
    return VorpTableSchema.validate(pool)


def rest_of_season_points(
    fresh_season_points: float | None,
    points_to_date: float,
    *,
    preseason_points: float,
    games_remaining: int,
) -> tuple[float, bool, bool]:
    """One player's remaining points -> (points, was_clamped, used_fallback).

    `fresh_season_points` is a full-season projection from a mid-season re-pull, taken to
    include games already played (see the module docstring). None means no fresh projection
    exists for this player, in which case the preseason number is prorated across the weeks
    that remain — the honest fallback, since there is nothing newer to use.

    A negative result means the player has already outscored the projection. That is real and
    ordinary late in a season; it is clamped to zero rather than allowed to subtract from a
    lineup, and reported.
    """
    if fresh_season_points is None:
        # Prorated over the GAMES that remain, which is the same horizon
        # `rest_of_season_pool` divides by. Any other denominator makes the round trip a
        # silent scale change rather than a pace no-op -- see that function's docstring for
        # the two ways this was got wrong.
        prorated = preseason_points * (games_remaining / SEASON_GAMES)
        return max(prorated, 0.0), False, True
    remaining = fresh_season_points - points_to_date
    if remaining <= 0.0:
        # A clamp means "this player has ALREADY OUTSCORED the projection", which requires
        # points on the board. A provider projecting a deep-bench player at 0.0 with nothing
        # scored is not that -- and a full VORP pool holds hundreds of them, enough to trip
        # the wholesale-clamp alarm on a perfectly healthy ingest. Once that fires spuriously
        # the signal it carries is worthless, so it is counted as a clamp only when actuals
        # are what pushed it under.
        return 0.0, points_to_date > 0.0, False
    return remaining, False, False


def rest_of_season_pool(
    pool: pd.DataFrame,
    fresh_season_points: Mapping[str, float],
    points_to_date: Mapping[str, float],
    *,
    games_remaining: int,
) -> tuple[pd.DataFrame, RosDiagnostics]:
    """Rewrite a pool's `season_mean_fpts` to a rest-of-season pace. Returns (pool, diagnostics).

    **The column stays a full-season-equivalent figure, not the raw remaining total**, because
    that is what the variance model consumes: `sample_weekly_points` divides
    `season_mean_fpts` by a fixed `SEASON_GAMES` to get a per-game mean. Writing the remaining
    total straight into the column would therefore spread it across a whole season instead of
    the weeks that are left, and quietly shrink every projection by roughly
    `games_remaining / SEASON_GAMES` — a bug that produces plausible-looking numbers.

    So the conversion is: per-game pace = `ros_points / games_remaining`, written back as
    `pace * SEASON_GAMES`.

    **`games_remaining` is NFL games left, not fantasy weeks left,** and the distinction has
    now bitten twice. `fresh - points_to_date` is the points a player will score over the
    games he has left to play; dividing that by the remaining *fantasy regular-season* weeks
    mixes two horizons and inflates everyone -- measured at 1.21x preseason, 3.40x at week 10
    of a 14-week league, and 17x at week 14. Callers should pass
    `SEASON_GAMES - (snapshot_week - 1)`.

    Using games rather than weeks also keeps the playoff bracket honest: a finished fantasy
    regular season still has NFL games left, and keying off fantasy weeks zeroed the entire
    pool at exactly the point the bracket is simulated, handing every playoff matchup to the
    home side.

    `games_remaining <= 0` returns a zeroed pool -- genuinely nothing left to play.
    """
    out = pool.copy()
    if games_remaining <= 0:
        out["season_mean_fpts"] = 0.0
        return _validated(out), RosDiagnostics(n_players=len(out))

    totals, diagnostics = _remaining(
        out, fresh_season_points, points_to_date, games_remaining=games_remaining
    )
    # Per-game pace, re-expressed over a full season so the variance model's fixed
    # SEASON_GAMES divisor recovers the pace we meant. See the docstring.
    out["season_mean_fpts"] = [points / games_remaining * SEASON_GAMES for points in totals]
    return _validated(out), diagnostics


def _remaining(
    pool: pd.DataFrame,
    fresh_season_points: Mapping[str, float],
    points_to_date: Mapping[str, float],
    *,
    games_remaining: int,
    projection_is_preseason: bool = False,
) -> tuple[list[float], RosDiagnostics]:
    """Per-player remaining totals, plus what had to be papered over.

    The shared half of `rest_of_season_pool` and `remaining_totals`. Those two differ ONLY in
    what they write back -- a full-season-equivalent pace for the simulator, a raw remaining
    total for a table cell -- so everything before that point lives here. The alternative was
    two copies of the clamp accounting, the near-zero rule and the diagnostics assembly, which
    is what the first version of `remaining_totals` was.
    """
    # Scaled by the games that remain. `points` here is a REMAINING total, so an absolute
    # threshold means "under a point for the rest of the season" early on and "under a point
    # for one game" at the end -- and a full pool's deep bench clears the latter routinely,
    # which would fire the wholesale alarm on a healthy run. That is the same cry-wolf failure
    # the clamp semantics above were narrowed to remove.
    near_zero_cutoff = _IMPLAUSIBLY_SMALL_ROS * games_remaining / SEASON_GAMES

    totals: list[float] = []
    clamped: list[str] = []
    n_fallback = 0
    n_near_zero = 0
    for row in pool.itertuples():
        gsis = str(row.gsis_id)
        points, was_clamped, used_fallback = rest_of_season_points(
            fresh_season_points.get(gsis),
            float(points_to_date.get(gsis, 0.0)),
            preseason_points=float(row.season_mean_fpts),
            games_remaining=games_remaining,
        )
        if was_clamped:
            # `display_str` rather than `name or gsis`: `bool(pd.NA)` raises, and this is the
            # one path whose whole purpose is making a failure visible, so it must not be the
            # path that crashes.
            clamped.append(display_str(getattr(row, "full_name", None)) or gsis)
        elif not used_fallback and 0.0 < points < near_zero_cutoff:
            n_near_zero += 1
        n_fallback += used_fallback
        totals.append(points)

    return totals, RosDiagnostics(
        n_players=len(pool),
        n_clamped=len(clamped),
        n_fallback=n_fallback,
        n_near_zero=n_near_zero,
        near_zero_cutoff=near_zero_cutoff,
        clamped_examples=tuple(clamped[:5]),
        projection_is_preseason=projection_is_preseason,
    )


def remaining_totals(
    pool: pd.DataFrame,
    points_to_date: Mapping[str, float],
    *,
    games_remaining: int,
) -> tuple[pd.DataFrame, RosDiagnostics]:
    """Rewrite `season_mean_fpts` to the points a player still has to score. Returns
    (pool, diagnostics).

    **Sibling of `rest_of_season_pool`, and deliberately not the same thing.** Both subtract
    points-to-date from a season projection; they differ in what they write back:

    - `rest_of_season_pool` writes a full-season-equivalent PACE, because its consumer is the
      variance model, which divides `season_mean_fpts` by a fixed `SEASON_GAMES`.
    - this one writes the raw remaining TOTAL, because its consumer is a table cell under the
      header "projected points for the rest of the season", where a reader wants the number of
      points still coming.

    Printing a pace under that header roughly doubles it by week 10; feeding a total to the
    simulator shrinks every roster by `games_remaining / SEASON_GAMES`. Both mistakes have been
    made in this repo, which is why the two functions sit next to each other with this
    paragraph between them. Everything before the write-back is `_remaining`, shared.

    **The pool is a PRESEASON projection here, not a fresh mid-season pull**, and that changes
    what a clamp means. `rest_of_season_pool`'s caller supplies re-pulled season totals, so a
    wholesale clamp really is the signature of a provider reporting rest-of-season. Here it is
    at least as likely to mean "the pool is from August and the season is nearly over" -- by
    week 15 a good share of any preseason projection has been beaten. `projection_is_preseason`
    makes the warning say both, rather than confidently naming the wrong cause on the one page
    a human reads.

    `games_remaining` is still needed even though nothing is prorated: it scales the near-zero
    cutoff, which is compared against a remaining total and would otherwise mean "under a point
    for the whole rest of the season" in September and "under a point for one game" in
    December.
    """
    out = pool.copy()
    if games_remaining <= 0:
        # Every cell renders 0.0 and the roster total with it, which looks like a bad team
        # rather than an exhausted season -- so it is counted as a wholesale clamp, which is
        # what makes `warning()` say something. Reachable in a league whose regular season runs
        # to 17 weeks or more, where `first_unplayed_week` returns `reg_weeks + 1`.
        out["season_mean_fpts"] = 0.0
        return _validated(out), RosDiagnostics(
            n_players=len(out),
            n_clamped=len(out),
            clamped_examples=tuple(str(g) for g in out["gsis_id"].head(5)),
            projection_is_preseason=True,
        )
    # Each player is measured against his own whole-season projection, so `fresh` IS the pool
    # and nothing is ever prorated -- `used_fallback` cannot be set from here.
    #
    # `VorpTableSchema` marks `gsis_id` unique, but this function is public and `_remaining`
    # runs BEFORE the validate on the way out, so a caller handing over an unvalidated pool
    # would have had both of a duplicated player's rows scored against whichever came last.
    # Checked rather than assumed, because the symptom is a plausible wrong number.
    duplicated = out["gsis_id"].astype(str).duplicated()
    if duplicated.any():
        raise ValueError(
            f"pool holds duplicate gsis_id(s) {sorted(set(out.loc[duplicated, 'gsis_id']))}; "
            "each player must appear once or his remaining points are ambiguous"
        )
    fresh = {str(row.gsis_id): float(row.season_mean_fpts) for row in out.itertuples()}
    totals, diagnostics = _remaining(
        out,
        fresh,
        points_to_date,
        games_remaining=games_remaining,
        projection_is_preseason=True,
    )
    out["season_mean_fpts"] = totals
    return _validated(out), diagnostics
