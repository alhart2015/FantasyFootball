"""Pure stat-line → fantasy-points scoring. Table-driven for testability."""

from __future__ import annotations

from collections.abc import Mapping

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


def _score_fields(f: Mapping[str, float], ruleset: Ruleset) -> float:
    """Shared scoring arithmetic over a stat-field -> value mapping. Absent keys count as 0.0.

    The single source of fantasy-points truth; both score() (realized int lines) and
    expected_points() (fractional projections) delegate here.
    """
    pts = 0.0
    pts += f.get("passing_yards", 0.0) / ruleset.passing_yds_per_pt
    pts += f.get("passing_tds", 0.0) * ruleset.passing_td_pts
    pts += f.get("interceptions", 0.0) * ruleset.interception_pts
    pts += f.get("passing_2pt_conversions", 0.0) * ruleset.two_pt_pts

    pts += f.get("rushing_yards", 0.0) / ruleset.rushing_yds_per_pt
    pts += f.get("rushing_tds", 0.0) * ruleset.rushing_td_pts
    pts += f.get("rushing_2pt_conversions", 0.0) * ruleset.two_pt_pts

    pts += f.get("receptions", 0.0) * ruleset.reception_pts
    pts += f.get("receiving_yards", 0.0) / ruleset.receiving_yds_per_pt
    pts += f.get("receiving_tds", 0.0) * ruleset.receiving_td_pts
    pts += f.get("receiving_2pt_conversions", 0.0) * ruleset.two_pt_pts

    pts += f.get("fumbles_lost", 0.0) * ruleset.fumble_lost_pts
    pts += f.get("return_tds", 0.0) * ruleset.return_td_pts
    return pts


def score(line: StatLine, ruleset: Ruleset) -> float:
    """Convert a `StatLine` to fantasy points under `ruleset`. Pure function."""
    return _score_fields(line.model_dump(), ruleset)


def expected_points(line: Mapping[str, float], ruleset: Ruleset) -> float:
    """Score a fractional *expected* stat line (e.g. a preseason projection's 8.4 receiving TDs)
    under `ruleset`, using the same coefficients as score(). Absent fields count as 0.0."""
    return _score_fields(line, ruleset)
