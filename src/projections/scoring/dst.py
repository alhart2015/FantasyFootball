"""Team defense / special teams scoring.

The only place a D/ST stat vector becomes fantasy points, mirroring the rule the rest of the
scoring layer follows.

**D/ST scoring is exactly a dot product.** ESPN's own number satisfies

    appliedTotal = sum over statIds of  raw_stat[id] * points_for_dst[id]

where `points_for_dst[id]` is the league's `pointsOverrides["16"][id]`. This was verified against
every D/ST projection ESPN publishes for 2026 -- 1215 (team, scoring period) rows, worst absolute
error 5e-9, i.e. float noise. See docs/superpowers/specs/2026-09-06-dst-projections-design.md
§1.3.

A D/ST stat vector carries no skill stat ids (measured: the 47 ids ESPN populates are all in the
defensive/special-teams/team ranges), so scoring items that have a base value but no position-16
override cannot reach a defense. They are excluded from the map rather than multiplied by zero.

The consequence shapes this module: **nothing here needs to know what a statId means.** There is
no `{99: "sacks"}` table in the scoring path, so there is no way for a mis-transcribed id to
produce a wrong projection that still looks reasonable. `DST_STAT_LABELS` exists for display and
diagnostics and is never consulted by `score_dst`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from projections.schemas import Ruleset

#: ESPN D/ST statId -> the label ESPN's own League Info screen shows.
#:
#: **Display and diagnostics only. Never used to score.** Verified 2026-09-06 by diffing the
#: Critts league payload against the league's League Info screens: 26/26 exact, no unmatched
#: labels either way (spec §1.4).
DST_STAT_LABELS: Final[Mapping[str, str]] = {
    "89": "0 points allowed",
    "90": "1-6 points allowed",
    "91": "7-13 points allowed",
    "92": "14-17 points allowed",
    "93": "blocked punt or FG returned for TD",
    "95": "interception",
    "96": "fumble recovered",
    "97": "blocked punt, PAT or FG",
    "98": "safety",
    "99": "sack",
    "101": "kickoff return TD",
    "102": "punt return TD",
    "103": "fumble return TD",
    "104": "interception return TD",
    "123": "28-34 points allowed",
    "124": "35-45 points allowed",
    "125": "46+ points allowed",
    "128": "under 100 total yards allowed",
    "129": "100-199 total yards allowed",
    "130": "200-299 total yards allowed",
    "132": "350-399 total yards allowed",
    "133": "400-449 total yards allowed",
    "134": "450-499 total yards allowed",
    "135": "500-549 total yards allowed",
    "136": "550+ total yards allowed",
    "206": "2-point return",
}


class DstScoringError(RuntimeError):
    """Raised when a D/ST score is asked for under a ruleset that does not score defenses.

    Deliberately an error rather than a 0.0: a league with a D/ST roster slot and no D/ST
    scoring is a contradiction, and silently returning zero would rank every defense equally
    and identically worthless -- a wrong answer that looks like a real one.
    """


def score_dst(stat_vector: Mapping[str, float], ruleset: Ruleset) -> float:
    """Fantasy points for one defense-week, from its raw stat vector.

    `stat_vector` maps ESPN statId (as a string) to the projected or actual value for that
    stat. Ids the ruleset does not score contribute nothing, which is how a projection
    carrying ESPN's full ~40-id vector scores correctly under a league that uses 26 of them.

    Raises:
        DstScoringError: If `ruleset` scores no D/ST categories at all.
    """
    if not ruleset.dst_stat_points:
        raise DstScoringError(
            f"Ruleset {ruleset.name!r} has no D/ST scoring categories, so a defense cannot be "
            "scored. If this league rosters a D/ST, its scoring failed to parse; if it does "
            "not, no defense should have reached this call."
        )
    return sum(
        points * float(stat_vector.get(stat_id, 0.0)) for stat_id, points in ruleset.dst_stat_points
    )


__all__ = ["DST_STAT_LABELS", "DstScoringError", "score_dst"]
