"""Synthetic features + truth fixtures for BaselineModel unit tests.

Independent of tests/test_features/conftest.py — pytest fixtures don't share
across sibling test directories. We build only what's needed here: schema-valid
WR feature rows + matching WeeklyStats truth rows that train a non-degenerate
RidgeCV.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR

# 5 synthetic WRs across 2 teams; 8 weeks of 2024 + 4 weeks of 2025.
_GSIS_IDS = ["00-0010001", "00-0010002", "00-0010003", "00-0010004", "00-0010005"]
_TEAMS = ["KC", "KC", "MIN", "MIN", "MIN"]
_TARGETS_BASE = [10.0, 6.0, 9.0, 4.0, 3.0]  # per-player target rate


def _wr_weekly_stats_row(
    *, gsis_id: str, season: int, week: int, team: str, opponent: str, base_targets: float
) -> dict[str, object]:
    """Build a single WeeklyStatsSchema-valid row with stat values that scale
    plausibly with the per-player base target rate. Random jitter is added by
    the caller via week index so trailing-4 means are non-constant."""
    targets_jitter = (week % 3) - 1  # -1, 0, +1, repeating
    targets = max(0, int(base_targets + targets_jitter))
    receptions = max(0, int(targets * 0.65))
    rec_yards = float(receptions * 12.0)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": "WR",
        "team": team,
        "opponent": opponent,
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": 0.0,
        "rushing_tds": 0,
        "carries": 0,
        "receptions": receptions,
        "receiving_yards": rec_yards,
        "receiving_tds": 1 if (week + int(gsis_id[-1])) % 4 == 0 else 0,
        "receiving_air_yards": float(targets * 13.0),
        "targets": targets,
        "fumbles_lost": 0,
    }


@pytest.fixture
def baseline_weekly_stats() -> pd.DataFrame:
    """8 weeks of 2024 + 4 weeks of 2025 stats for 5 synthetic WRs.

    2024 = training universe; 2025 = held-out. KC and MIN play each other
    every week so the opponent-allowed-fppg proxy resolves (MIN's allowed WR
    FPPG comes from KC's WRs and vice versa).
    """
    rows: list[dict[str, object]] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            for gsis_id, team, base_targets in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True):
                opponent = "MIN" if team == "KC" else "KC"
                rows.append(
                    _wr_weekly_stats_row(
                        gsis_id=gsis_id,
                        season=season,
                        week=week,
                        team=team,
                        opponent=opponent,
                        base_targets=base_targets,
                    )
                )

    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def baseline_features(baseline_weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """WR feature rows produced by build_wr_features for every (season, week)
    in the training fixture. Built up-front so tests don't pay the cost
    individually."""
    from projections.features import build_wr_features

    # We need supporting fixtures (snap_counts, depth_charts, ngs, schedules)
    # for the builder. Keep them minimal.
    snap_rows = [
        {
            "gsis_id": r["gsis_id"],
            "season": r["season"],
            "week": r["week"],
            "team": r["team"],
            "opponent": r["opponent"],
            "position": "WR",
            "offense_snaps": 60,
            "offense_pct": 0.95,
            "defense_snaps": 0,
            "defense_pct": 0.0,
            "st_snaps": 2,
            "st_pct": 0.05,
        }
        for _, r in baseline_weekly_stats.iterrows()
    ]
    snap_counts = pd.DataFrame(snap_rows)
    for col in ("gsis_id", "team", "opponent", "position"):
        snap_counts[col] = snap_counts[col].astype(_PYARROW_STR)

    # Depth charts: every player as their team's WR1/WR2 (deterministic by
    # base_targets ranking) for every (season, week).
    dc_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, _base_targets in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True):
                # Rank within team by base_targets descending.
                team_pool = sorted(
                    [
                        (g, t, b)
                        for g, t, b in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True)
                        if t == team
                    ],
                    key=lambda x: -x[2],
                )
                rank = next(i for i, (g, _, _) in enumerate(team_pool, start=1) if g == gsis_id)
                dc_rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "team": team,
                        "position": "WR",
                        "depth_team": f"WR{rank}",
                        "depth_rank": rank,
                    }
                )
    depth = pd.DataFrame(dc_rows)
    for col in ("gsis_id", "team", "position", "depth_team"):
        depth[col] = depth[col].astype(_PYARROW_STR)

    # NGS: per-(season, week) snapshot per WR with stable, per-player synthetic
    # values keyed off base_targets so the rows are non-NaN at fit time.
    # The wr.py builder applies prior_mask and then takes the latest snapshot
    # per gsis_id; values vary slightly week-to-week so the snapshot isn't
    # week-1-only.
    ngs_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, base_targets in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True):
                ngs_rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "team": team,
                        "position": "WR",
                        "avg_separation": 2.5 + base_targets * 0.05,
                        "avg_intended_air_yards": 9.0 + base_targets * 0.2,
                        # 0-100 scale (wr.py divides by 100 for the schema's 0-1 range).
                        "percent_share_of_intended_air_yards": min(95.0, 5.0 + base_targets * 4.0),
                        "avg_yac_above_expectation": -0.2 + base_targets * 0.05,
                    }
                )
    ngs = pd.DataFrame(ngs_rows)
    for col in ("gsis_id", "team", "position"):
        ngs[col] = ngs[col].astype(_PYARROW_STR)

    # Schedules: KC hosts MIN every week. Single game per (season, week)
    # covers both teams' WRs.
    sch_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            sch_rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week:02d}_KC_MIN",
                    "home_team": "KC",
                    "away_team": "MIN",
                    "kickoff": pd.Timestamp(f"{season}-09-{week + 1:02d}T17:00:00Z")
                    .tz_convert("UTC")
                    .as_unit("us"),
                    "spread_line": -3.0,
                    "total_line": 47.0,
                    "home_moneyline": -150,
                    "away_moneyline": 130,
                    "surface": "grass",
                    "roof": "outdoors",
                    "temp": 60,
                    "wind": 5,
                }
            )
    schedules = pd.DataFrame(sch_rows)
    for col in ("game_id", "home_team", "away_team", "surface", "roof"):
        schedules[col] = schedules[col].astype(_PYARROW_STR)
    schedules["temp"] = schedules["temp"].astype(pd.Int64Dtype())
    schedules["wind"] = schedules["wind"].astype(pd.Int64Dtype())
    schedules["home_moneyline"] = schedules["home_moneyline"].astype(pd.Int64Dtype())
    schedules["away_moneyline"] = schedules["away_moneyline"].astype(pd.Int64Dtype())

    feat_frames: list[pd.DataFrame] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            f = build_wr_features(
                weekly_stats=baseline_weekly_stats,
                snap_counts=snap_counts,
                depth_charts=depth,
                ngs_receiving=ngs,
                schedules=schedules,
                season=season,
                as_of_week=week,
            )
            feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True)
