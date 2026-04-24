"""Pure stat-line → fantasy-points scoring. Table-driven for testability."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from projections.schemas import Ruleset


class StatLine(BaseModel):
    """A single player-week stat line. Values are *realized* counts/yards.

    Used by `score()` and as the natural per-sample shape produced when
    `Distribution.sample()` is paired with the underlying stat dimensions.
    """

    model_config = ConfigDict(frozen=True)

    passing_yards: float = 0.0
    passing_tds: int = 0
    interceptions: int = 0
    passing_2pt_conversions: int = 0

    rushing_yards: float = 0.0
    rushing_tds: int = 0
    rushing_2pt_conversions: int = 0

    receptions: int = 0
    receiving_yards: float = 0.0
    receiving_tds: int = 0
    receiving_2pt_conversions: int = 0

    fumbles_lost: int = 0
    return_tds: int = 0


def score(line: StatLine, ruleset: Ruleset) -> float:
    """Convert a `StatLine` to fantasy points under `ruleset`. Pure function."""
    pts = 0.0
    pts += line.passing_yards / ruleset.passing_yds_per_pt
    pts += line.passing_tds * ruleset.passing_td_pts
    pts += line.interceptions * ruleset.interception_pts
    pts += line.passing_2pt_conversions * ruleset.two_pt_pts

    pts += line.rushing_yards / ruleset.rushing_yds_per_pt
    pts += line.rushing_tds * ruleset.rushing_td_pts
    pts += line.rushing_2pt_conversions * ruleset.two_pt_pts

    pts += line.receptions * ruleset.reception_pts
    pts += line.receiving_yards / ruleset.receiving_yds_per_pt
    pts += line.receiving_tds * ruleset.receiving_td_pts
    pts += line.receiving_2pt_conversions * ruleset.two_pt_pts

    pts += line.fumbles_lost * ruleset.fumble_lost_pts
    pts += line.return_tds * ruleset.return_td_pts
    return pts
