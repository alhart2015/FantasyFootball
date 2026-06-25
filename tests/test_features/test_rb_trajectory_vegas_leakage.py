"""Second-surface leakage guard for build_rb_features (#55 trajectory lift).

The first-surface guard (test_rb_leakage.py) asserts no leak into the
volume/snap/ngs/schedule rollups at as_of_week=5. This guard targets the
trajectory-trend feature surface specifically: inject implausible
CURRENT/FUTURE-week rows into the full frames that
attach_trajectory_features consumes (weekly_stats, snap_counts) and assert
those feature columns are byte-identical. A leak by definition changes the
computation, so byte-equality is the strongest possible signal.

Why this file builds its OWN richer 9-week fixtures instead of reusing the
4-week rb_* conftest frames (or the as_of_week=5 harness in
test_rb_leakage.py): the two asserted columns are structurally NaN with
only weeks 1-4 of history —

  - volume_trend_l4_minus_prior_l4: trailing-4 minus prior-4, where prior_l4
    is rolling(4).mean().shift(5), so it needs >=8 prior active games.
  - snap_pct_change_l4_vs_prior_l4: same trailing-4-vs-prior-4 shape on
    offense_pct.

With an all-NaN baseline, assert_frame_equal compares NaN-vs-NaN and
catches nothing (the WR analog proved the prior version was a vacuous
pass). We therefore build 9 weeks of weekly_stats + snap_counts, set
as_of_week=9, and assert byte-equality of the two columns under leak
injection into the windows the leak-safe shifts actually consume.

For RB the volume-trend driver is `carries` (not WR's `targets`): the
fixture varies per-week carries and snap pct so both trend columns are
genuinely non-NaN at week 9 (l4 sees weeks 5-8, prior_l4 sees weeks 1-4).

If a test here FAILS, that is a real leak — report it; do NOT weaken the
assertion.
"""

from __future__ import annotations

import pandas as pd

from projections.features import build_rb_features
from projections.schemas import _PYARROW_STR

_SEASON = 2024
_AS_OF_WEEK = 9  # late enough that trailing-8-game windows are full.

# The two trajectory-trend columns on RbFeaturesSchema (verified against
# src/projections/features/rb.py). RB keeps no age/is_rookie/Vegas trend cols.
_TRAJ_COLS = [
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
]

# One RB plus a same-team RB2 (so rush_share isn't a degenerate 100%),
# both on PHI. The leak probes target the RB1.
_RB1 = "00-0034796"  # Saquon Barkley
_RB2 = "00-0034797"  # synthetic PHI RB2
_TEAM = "PHI"
_OPP = "DAL"  # PHI's opponent every week in this synthetic universe.


def _weekly_stats() -> pd.DataFrame:
    """9 weeks of 2024 weekly_stats for two PHI RBs.

    RB1 carries a varied carry trajectory so the trailing-4 vs prior-4
    trend is a non-trivial, non-NaN number at week 9 (l4 sees weeks 5-8,
    prior_l4 sees weeks 1-4). RB2 keeps rush_share sane.
    """
    rows: list[dict[str, object]] = []
    # Distinct per-week carries so l4 (wk5-8) != prior_l4 (wk1-4).
    rb1_carries = [14, 15, 16, 17, 22, 23, 24, 25, 21]
    for week in range(1, 10):
        c1 = rb1_carries[week - 1]
        rows.append(
            {
                "gsis_id": _RB1,
                "season": _SEASON,
                "week": week,
                "position": "RB",
                "team": _TEAM,
                "opponent": _OPP,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "attempts": 0,
                "completions": 0,
                "sacks": 0,
                "rushing_yards": float(c1 * 4),
                "rushing_tds": 1,
                "carries": c1,
                "receptions": 1,
                "receiving_yards": 8.0,
                "receiving_tds": 0,
                "receiving_air_yards": 5.0,
                "targets": 2,
                "fumbles_lost": 0,
            }
        )
        rows.append(
            {
                "gsis_id": _RB2,
                "season": _SEASON,
                "week": week,
                "position": "RB",
                "team": _TEAM,
                "opponent": _OPP,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "attempts": 0,
                "completions": 0,
                "sacks": 0,
                "rushing_yards": 24.0,
                "rushing_tds": 0,
                "carries": 6,
                "receptions": 2,
                "receiving_yards": 15.0,
                "receiving_tds": 0,
                "receiving_air_yards": 10.0,
                "targets": 3,
                "fumbles_lost": 0,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "team", "opponent"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _snap_counts(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the same RBs/weeks. RB1's offense_pct ramps across
    weeks so the trailing-4 vs prior-4 snap-pct change is non-NaN and
    non-zero at week 9."""
    rb1_pct = [0.60, 0.62, 0.64, 0.66, 0.76, 0.78, 0.80, 0.82, 0.74]
    rows: list[dict[str, object]] = []
    for _, r in weekly_stats.iterrows():
        if r["gsis_id"] == _RB1:
            pct = rb1_pct[int(r["week"]) - 1]
        else:
            pct = 0.40
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 55,
                "offense_pct": pct,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 5,
                "st_pct": 0.15,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "opponent", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _depth_charts() -> pd.DataFrame:
    """Week-9 depth chart snapshot. RB1 as RB1, RB2 as RB2."""
    rows = [
        {
            "gsis_id": _RB1,
            "season": _SEASON,
            "week": _AS_OF_WEEK,
            "team": _TEAM,
            "position": "RB",
            "depth_team": "RB1",
            "depth_rank": 1,
        },
        {
            "gsis_id": _RB2,
            "season": _SEASON,
            "week": _AS_OF_WEEK,
            "team": _TEAM,
            "position": "RB",
            "depth_team": "RB2",
            "depth_rank": 2,
        },
    ]
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "position", "depth_team"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _ngs_rushing() -> pd.DataFrame:
    """NGS rushing snapshots weeks 1-8 for RB1 (latest-snapshot feed).

    These columns are not under test here, but the builder needs a
    non-empty feed so the merge doesn't NaN-out unrelated rows. Values
    are constant; the leakage probes never touch this frame.
    """
    rows: list[dict[str, object]] = []
    for week in range(1, 9):
        rows.append(
            {
                "gsis_id": _RB1,
                "season": _SEASON,
                "week": week,
                "team": _TEAM,
                "position": "RB",
                "efficiency": 3.1,
                "percent_attempts_gte_eight_defenders": 25.0,
                "avg_time_to_los": 2.95,
                "rush_attempts": 20,
                "rush_yards": 90,
                "expected_rush_yards": 80.0,
                "rush_yards_over_expected": 10.0,
                "avg_rush_yards": 4.5,
                "rush_yards_over_expected_per_att": 0.5,
                "rush_pct_over_expected": 12.5,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _schedules() -> pd.DataFrame:
    """9 schedule weeks: PHI @ DAL every week. Only the week-9 row is the
    exact-week row the builder consumes; the rest give the trajectory frames
    a full per-team history. PHI is the away team."""
    rows: list[dict[str, object]] = []
    for week in range(1, 10):
        rows.append(
            {
                "season": _SEASON,
                "week": week,
                "game_id": f"2024_{week:02d}_PHI_DAL",
                "home_team": "DAL",
                "away_team": _TEAM,
                "kickoff": pd.Timestamp(f"2024-09-{week + 1:02d}T17:00:00Z")
                .tz_convert("UTC")
                .as_unit("us"),
                "spread_line": -2.5,
                "total_line": 45.0,
                "home_moneyline": 135,
                "away_moneyline": -160,
                "surface": "grass",
                "roof": "outdoors",
                "temp": 62,
                "wind": 6,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("game_id", "home_team", "away_team", "surface", "roof"):
        df[col] = df[col].astype(_PYARROW_STR)
    for col in ("temp", "wind", "home_moneyline", "away_moneyline"):
        df[col] = df[col].astype(pd.Int64Dtype())
    return df


def _build(
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
) -> pd.DataFrame:
    return build_rb_features(
        weekly_stats=weekly_stats,
        snap_counts=snap_counts,
        depth_charts=_depth_charts(),
        ngs_rushing=_ngs_rushing(),
        schedules=_schedules(),
        pbp=pd.DataFrame(),
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def _baseline() -> pd.DataFrame:
    ws = _weekly_stats()
    return _build(ws, _snap_counts(ws))


def _assert_columns_non_nan(frame: pd.DataFrame) -> None:
    """Guard against a future fixture regression silently re-vacuuming the
    test: both trend columns must be non-NaN on the RB1 row."""
    rb1 = frame[frame["gsis_id"] == _RB1]
    assert not rb1.empty, "RB1 row missing from feature output"
    for col in _TRAJ_COLS:
        assert rb1[col].notna().all(), (
            f"{col} is NaN at as_of_week={_AS_OF_WEEK}; fixture history is too "
            f"short to populate it, which would make this leakage guard vacuous"
        )


def test_traj_baseline_columns_are_non_nan() -> None:
    """Sanity floor: the two leakage-probed columns must be populated in
    the baseline, else the equality assertions below are vacuous."""
    _assert_columns_non_nan(_baseline())


def test_future_weekly_stats_do_not_leak_into_trajectory() -> None:
    """Injecting implausible week-9 (>= as_of_week) carries must not move the
    trailing-4-vs-prior-4 volume trend, which is computed from weeks 1-8 only
    via the rolling .shift(1)/.shift(5)."""
    base = _baseline()
    _assert_columns_non_nan(base)

    ws = _weekly_stats()
    leaky = ws.copy()
    fut = (leaky["gsis_id"] == _RB1) & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[fut, ["carries", "rushing_yards", "targets"]] = 999.0
    after = _build(leaky, _snap_counts(ws))

    pd.testing.assert_frame_equal(base[_TRAJ_COLS], after[_TRAJ_COLS], check_like=True)


def test_future_snap_counts_do_not_leak_into_trajectory() -> None:
    """Injecting an implausible week-9 snap_counts row must not move the
    trailing-4-vs-prior-4 snap-pct change (computed from weeks 1-8 only)."""
    base = _baseline()
    _assert_columns_non_nan(base)

    ws = _weekly_stats()
    sc = _snap_counts(ws)
    leaky = sc.copy()
    fut = (leaky["gsis_id"] == _RB1) & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[fut, "offense_pct"] = 0.0  # implausible: RB1 vanishes in week 9
    after = _build(ws, leaky)

    pd.testing.assert_frame_equal(base[_TRAJ_COLS], after[_TRAJ_COLS], check_like=True)
