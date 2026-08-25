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
from projections.schemas import VorpTableSchema

#: A healthy starter with this many points or fewer left, this early, is far more likely to be
#: an ingest problem than a real projection. Used only to decide whether to warn.
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
                f"zero for the rest of the season (under {_IMPLAUSIBLY_SMALL_ROS:g} points) "
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
        if self.looks_like_double_counting:
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
    validating without taking the result back leaves them in place. `is_rookie` is not on
    `VorpTableSchema` but `team_weekly_points` needs it, so it is restored afterwards -- the
    validate is here to catch a frame that has lost `season_mean_fpts` or `gsis_id`, not to
    strip the columns the simulator runs on.
    """
    extras = {col: pool[col] for col in ("is_rookie",) if col in pool.columns}
    validated = VorpTableSchema.validate(pool)
    for col, values in extras.items():
        validated[col] = values
    return validated


def rest_of_season_points(
    fresh_season_points: float | None,
    points_to_date: float,
    *,
    preseason_points: float,
    weeks_remaining: int,
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
        # Prorated over SEASON_GAMES, NOT reg_weeks. `preseason_points` is already a
        # SEASON_GAMES-game total, and `rest_of_season_pool` re-expresses whatever comes back
        # here as `points / weeks_remaining * SEASON_GAMES`. Dividing by reg_weeks instead
        # composed to `preseason * SEASON_GAMES / reg_weeks` -- a silent 1.21x on a 14-week
        # league and 1.31x on a 13-week one, for a player whose pace should be unchanged.
        # Two horizons inside one function is exactly what this module exists to prevent.
        prorated = preseason_points * (weeks_remaining / SEASON_GAMES)
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
    weeks_remaining: int,
) -> tuple[pd.DataFrame, RosDiagnostics]:
    """Rewrite a pool's `season_mean_fpts` to a rest-of-season pace. Returns (pool, diagnostics).

    **The column stays a full-season-equivalent figure, not the raw remaining total**, because
    that is what the variance model consumes: `sample_weekly_points` divides
    `season_mean_fpts` by a fixed `SEASON_GAMES` to get a per-game mean. Writing the remaining
    total straight into the column would therefore spread it across a whole season instead of
    the weeks that are left, and quietly shrink every projection by roughly
    `weeks_remaining / SEASON_GAMES` — a bug that produces plausible-looking numbers.

    So the conversion is: per-game pace = `ros_points / weeks_remaining`, written back as
    `pace * SEASON_GAMES`.

    `weeks_remaining == 0` (a finished regular season) returns a zeroed pool: there is nothing
    left to project, and the caller's locked records already carry the whole answer.
    """
    out = pool.copy()
    if weeks_remaining <= 0:
        out["season_mean_fpts"] = 0.0
        return _validated(out), RosDiagnostics(n_players=len(out))

    ros: list[float] = []
    clamped: list[str] = []
    n_fallback = 0
    n_near_zero = 0
    for row in out.itertuples():
        gsis = str(row.gsis_id)
        points, was_clamped, used_fallback = rest_of_season_points(
            fresh_season_points.get(gsis),
            float(points_to_date.get(gsis, 0.0)),
            preseason_points=float(row.season_mean_fpts),
            weeks_remaining=weeks_remaining,
        )
        if was_clamped:
            name = getattr(row, "full_name", None)
            # `x or y` evaluates bool(x), and bool(pd.NA) raises. Name columns here are
            # pyarrow-backed nullable strings, so pd.NA is admissible -- and this is the one
            # path whose whole purpose is making a failure visible, so it must not be the path
            # that crashes.
            clamped.append(gsis if name is None or pd.isna(name) else str(name))
        elif not used_fallback and 0.0 < points < _IMPLAUSIBLY_SMALL_ROS:
            n_near_zero += 1
        n_fallback += used_fallback
        # Per-game pace, re-expressed over a full season so the variance model's fixed
        # SEASON_GAMES divisor recovers the pace we meant. See the docstring.
        ros.append(points / weeks_remaining * SEASON_GAMES)

    out["season_mean_fpts"] = ros
    return _validated(out), RosDiagnostics(
        n_players=len(out),
        n_clamped=len(clamped),
        n_fallback=n_fallback,
        n_near_zero=n_near_zero,
        clamped_examples=tuple(str(name) for name in clamped[:5]),
    )
