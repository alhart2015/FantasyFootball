"""What an injury designation costs, and where to apply it.

The constants here were measured, not guessed — `scripts/measure_injury_impact.py` produces
them from five seasons of weekly injury reports joined to ESPN weekly projections and to
`weekly_stats` scored under the league ruleset. Re-run it when a season is added. §4 of
`docs/superpowers/specs/2026-08-26-waiver-recommender-design.md` records the reasoning.

**Two adjustments, because a designation costs different things on different horizons.** A
Questionable tag covers one week:

- On **this week's** projection it is a 14% cut, which is the size that decides a close
  start/sit. That is `weekly_multiplier`.
- On a **rest-of-season** total it is one week out of however many remain — a ~1.4% haircut at
  ten weeks, which will never flip a drop/add decision. That is `expected_games_missed`.

Collapsing them into one number was the original design mistake. What moves a season-long
figure is a *multi-week* absence, and that is `INJURY_RESERVE`, which is also the one number
here still guessed.
"""

from __future__ import annotations

from projections.schemas import InjuryStatus

#: Share of his projection a player delivers, given a designation, for ONE week.
#:
#: `QUESTIONABLE` is measured directly: 83.4% of projection delivered against a healthy
#: baseline of 96.5% (n=846, players projected 5+ points). `DOUBTFUL` and `OUT` are measured
#: from play rate instead — ESPN gives them almost no real projection, so the delivered-share
#: sample is n=1 for each and worthless, while the play-rate sample is 269 and 1,661.
WEEKLY_MULTIPLIER: dict[InjuryStatus, float] = {
    InjuryStatus.QUESTIONABLE: 0.86,
    InjuryStatus.DOUBTFUL: 0.04,
    InjuryStatus.OUT: 0.0,
    InjuryStatus.SUSPENSION: 0.0,
    InjuryStatus.INJURY_RESERVE: 0.0,
}

#: NFL games a player is expected to miss, given a designation.
#:
#: Everything except `INJURY_RESERVE` follows from `WEEKLY_MULTIPLIER` and the fact that a game
#: status covers exactly one week — ESPN re-reports it weekly.
EXPECTED_GAMES_MISSED: dict[InjuryStatus, float] = {
    InjuryStatus.QUESTIONABLE: 1.0 - WEEKLY_MULTIPLIER[InjuryStatus.QUESTIONABLE],
    InjuryStatus.DOUBTFUL: 1.0 - WEEKLY_MULTIPLIER[InjuryStatus.DOUBTFUL],
    InjuryStatus.OUT: 1.0,
    InjuryStatus.SUSPENSION: 1.0,
    # THE ONE GUESS, and the one that matters most. IR is a roster designation rather than a
    # game status, so it does not appear in the injury report at all and cannot be measured
    # the way the others were. Four is the NFL minimum stay, i.e. the OPTIMISTIC end: it
    # biases toward keeping the player, because dropping someone who returns in Week 12 is a
    # worse mistake than holding him a week too long. `injury_note` is how a reader overrules
    # it — the beat-reporter write-up usually says more than four-or-not.
    InjuryStatus.INJURY_RESERVE: 4.0,
}

#: Statuses ESPN has already priced into its own weekly projections. See `weekly_multiplier`.
_ALREADY_PRICED_BY_ESPN: frozenset[InjuryStatus] = frozenset(
    {InjuryStatus.OUT, InjuryStatus.DOUBTFUL, InjuryStatus.INJURY_RESERVE}
)


def weekly_multiplier(status: InjuryStatus, *, source_is_injury_aware: bool = False) -> float:
    """What share of ONE week's projection this player is expected to deliver.

    `source_is_injury_aware` is the double-discount guard, and it is not optional politeness.
    ESPN's weekly feed already zeroes players it lists as `Out` — of 1,475 such designations
    reaching the feed, exactly one carried a projection above five points. Multiplying an
    already-zeroed projection by zero again is arithmetically harmless; multiplying an already
    *reduced* one by 0.04 is not, and either way the resulting number looks perfectly
    plausible, which is what makes it dangerous.

    So: pass `True` when the projection came from ESPN's weekly feed, and the statuses ESPN
    prices are left alone. `QUESTIONABLE` is applied either way, because ESPN verifiably does
    NOT discount it — a Questionable player's projection is 100.4% of his own healthy-week
    median (n=843).
    """
    if status.is_healthy:
        return 1.0
    if source_is_injury_aware and status in _ALREADY_PRICED_BY_ESPN:
        return 1.0
    return WEEKLY_MULTIPLIER.get(status, 1.0)


def expected_games_missed(status: InjuryStatus) -> float:
    """NFL games this designation implies the player will miss."""
    return 0.0 if status.is_healthy else EXPECTED_GAMES_MISSED.get(status, 0.0)


def season_multiplier(status: InjuryStatus, *, games_remaining: int) -> float:
    """What share of a REST-OF-SEASON total this player is expected to deliver.

    A rest-of-season projection assumes a healthy player over the games that remain; an injured
    one plays fewer of them. Clamped to `[0, 1]`: a designation cannot add points, and a player
    cannot miss more games than are left.

    `games_remaining <= 0` returns 0.0 — there is nothing left to deliver, which is a different
    statement from "he is fine".
    """
    if games_remaining <= 0:
        return 0.0
    if status.is_healthy:
        return 1.0
    missed = min(expected_games_missed(status), float(games_remaining))
    return max(0.0, (games_remaining - missed) / games_remaining)


def is_multi_week(status: InjuryStatus) -> bool:
    """Whether this designation implies an absence longer than the coming week.

    The line between "the number is measured and small" and "the number is a guess and large".
    A caller showing the injury write-up (§5.3) should show it for these, because the text is
    the only real information about how long the absence runs.
    """
    return expected_games_missed(status) > 1.0
