"""Second-surface leakage guard for build_wr_features (spec §5.3(b)).

The first-surface guard (test_wr_leakage.py) asserts no leak into the
volume/snap/ngs/depth/schedule rollups. This guard targets the
trajectory + Vegas team-context feature surface specifically: inject
implausible CURRENT/PRIOR-week rows into the full frames that
attach_trajectory_features / the Vegas team-context join consume
(weekly_stats, snap_counts, schedules) and assert those feature columns
are byte-identical. A leak by definition changes the computation, so
byte-equality is the strongest possible signal.

Why this file builds its OWN richer fixtures instead of reusing the
4-week wr_* conftest frames: the four asserted columns are structurally
NaN with only weeks 1-4 of history —

  - volume_trend_l4_minus_prior_l4 / snap_pct_change_l4_vs_prior_l4: the
    prior_l4 term is rolling(4).mean().shift(5), so it needs >=8 prior
    active games before it is non-NaN.
  - season_avg_implied_team_total / season_avg_spread: expanding mean
    over weeks 1..N-1, so it needs >=2 schedule weeks for the team.

With an all-NaN baseline, assert_frame_equal compares NaN-vs-NaN and
catches nothing (the reviewer proved the prior version was a vacuous
pass). We therefore build 9 weeks of weekly_stats + snap_counts and 9
schedule weeks, set as_of_week=9, and assert byte-equality of the four
columns under leak injection into the windows the leak-safe shifts
actually consume.

If a test here FAILS, that is a real leak and a §5.3 hard stop — report
it; do NOT weaken the assertion.
"""

from __future__ import annotations

import pandas as pd

from projections.features import build_wr_features
from projections.schemas import _PYARROW_STR

_SEASON = 2024
_AS_OF_WEEK = 9  # late enough that trailing-8-game + expanding windows are full.

# Trajectory + Vegas team-context columns on WrFeaturesSchema (verified
# against src/projections/schemas.py).
_TRAJ_VEGAS_COLS = [
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
    "season_avg_implied_team_total",
    "season_avg_spread",
]

# One WR plus a same-team WR2 (so target_share isn't a degenerate 100%),
# both on MIN. The leak probes target the WR1.
_WR1 = "00-0036322"  # Justin Jefferson
_WR2 = "00-0036323"  # synthetic MIN WR2
_TEAM = "MIN"
_OPP = "CHI"  # MIN's opponent every week in this synthetic universe.


def _weekly_stats() -> pd.DataFrame:
    """9 weeks of 2024 weekly_stats for two MIN WRs.

    WR1 carries a varied target/yard trajectory so the trailing-4 vs
    prior-4 trend is a non-trivial, non-NaN number at week 9 (l4 sees
    weeks 5-8, prior_l4 sees weeks 1-4). WR2 keeps target_share sane.
    """
    rows: list[dict[str, object]] = []
    # Distinct per-week targets so l4 (wk5-8) != prior_l4 (wk1-4).
    wr1_targets = [6, 7, 8, 9, 12, 13, 14, 15, 11]
    for week in range(1, 10):
        t1 = wr1_targets[week - 1]
        rows.append(
            {
                "gsis_id": _WR1,
                "season": _SEASON,
                "week": week,
                "position": "WR",
                "team": _TEAM,
                "opponent": _OPP,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": t1 - 2,
                "receiving_yards": float(t1 * 12),
                "receiving_tds": 1,
                "receiving_air_yards": float(t1 * 14),
                "targets": t1,
                "fumbles_lost": 0,
            }
        )
        rows.append(
            {
                "gsis_id": _WR2,
                "season": _SEASON,
                "week": week,
                "position": "WR",
                "team": _TEAM,
                "opponent": _OPP,
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "carries": 0,
                "receptions": 3,
                "receiving_yards": 40.0,
                "receiving_tds": 0,
                "receiving_air_yards": 45.0,
                "targets": 5,
                "fumbles_lost": 0,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "team", "opponent"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _snap_counts(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Snap counts for the same WRs/weeks. WR1's offense_pct ramps across
    weeks so the trailing-4 vs prior-4 snap-pct change is non-NaN and
    non-zero at week 9."""
    wr1_pct = [0.70, 0.72, 0.74, 0.76, 0.86, 0.88, 0.90, 0.92, 0.84]
    rows: list[dict[str, object]] = []
    for _, r in weekly_stats.iterrows():
        if r["gsis_id"] == _WR1:
            pct = wr1_pct[int(r["week"]) - 1]
        else:
            pct = 0.80
        rows.append(
            {
                "gsis_id": r["gsis_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["team"],
                "opponent": r["opponent"],
                "position": r["position"],
                "offense_snaps": 60,
                "offense_pct": pct,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 2,
                "st_pct": 0.05,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "opponent", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _depth_charts() -> pd.DataFrame:
    """Week-9 depth chart snapshot. WR1 as WR1, WR2 as WR2."""
    rows = [
        {
            "gsis_id": _WR1,
            "season": _SEASON,
            "week": _AS_OF_WEEK,
            "team": _TEAM,
            "position": "WR",
            "depth_team": "WR1",
            "depth_rank": 1,
        },
        {
            "gsis_id": _WR2,
            "season": _SEASON,
            "week": _AS_OF_WEEK,
            "team": _TEAM,
            "position": "WR",
            "depth_team": "WR2",
            "depth_rank": 2,
        },
    ]
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "position", "depth_team"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _ngs_receiving() -> pd.DataFrame:
    """NGS receiving snapshots weeks 1-8 for WR1 (latest-snapshot feed).

    These columns are not under test here, but the builder needs a
    non-empty feed so the merge doesn't NaN-out unrelated rows. Values
    are constant; the leakage probes never touch this frame.
    """
    rows: list[dict[str, object]] = []
    for week in range(1, 9):
        rows.append(
            {
                "gsis_id": _WR1,
                "season": _SEASON,
                "week": week,
                "team": _TEAM,
                "position": "WR",
                "avg_cushion": 5.0,
                "avg_separation": 3.2,
                "avg_intended_air_yards": 12.0,
                "percent_share_of_intended_air_yards": 30.0,
                "receptions": 9,
                "targets": 12,
                "catch_percentage": 75.0,
                "yards": 110,
                "rec_touchdowns": 1,
                "avg_yac": 4.0,
                "avg_expected_yac": 3.5,
                "avg_yac_above_expectation": 0.5,
            }
        )
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def _schedules() -> pd.DataFrame:
    """9 schedule weeks: MIN @ CHI every week, total/spread varying so the
    expanding-mean season_avg_* is non-NaN and non-constant at week 9.

    MIN is the away team. build_game_environment: away spread = spread_line,
    away implied = (total - spread_line) / 2. With total ~47 and spread_line
    ~ -3, MIN's implied total ~25 — comfortably inside the schema's [0, 60].
    """
    # Distinct lines per week so season_avg (mean of wk1..N-1) is non-trivial.
    total_lines = [45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 47.0, 48.0, 46.0]
    spread_lines = [-2.0, -3.0, -1.0, -4.0, -2.5, -3.5, -1.5, -2.5, -3.0]
    rows: list[dict[str, object]] = []
    for week in range(1, 10):
        rows.append(
            {
                "season": _SEASON,
                "week": week,
                "game_id": f"2024_{week:02d}_MIN_CHI",
                "home_team": "CHI",
                "away_team": _TEAM,
                "kickoff": pd.Timestamp(f"2024-09-{week + 1:02d}T17:00:00Z")
                .tz_convert("UTC")
                .as_unit("us"),
                "spread_line": spread_lines[week - 1],
                "total_line": total_lines[week - 1],
                "home_moneyline": 130,
                "away_moneyline": -150,
                "surface": "grass",
                "roof": "outdoors",
                "temp": 55,
                "wind": 8,
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
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    return build_wr_features(
        weekly_stats=weekly_stats,
        snap_counts=snap_counts,
        depth_charts=_depth_charts(),
        ngs_receiving=_ngs_receiving(),
        schedules=schedules,
        pbp=pd.DataFrame(),
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def _baseline() -> pd.DataFrame:
    return _build(_weekly_stats(), _snap_counts(_weekly_stats()), _schedules())


def _assert_columns_non_nan(frame: pd.DataFrame) -> None:
    """Guard against a future fixture regression silently re-vacuuming the
    test: every asserted column must be non-NaN on the WR1 row."""
    wr1 = frame[frame["gsis_id"] == _WR1]
    assert not wr1.empty, "WR1 row missing from feature output"
    for col in _TRAJ_VEGAS_COLS:
        assert wr1[col].notna().all(), (
            f"{col} is NaN at as_of_week={_AS_OF_WEEK}; fixture history is too "
            f"short to populate it, which would make this leakage guard vacuous"
        )


def test_traj_vegas_baseline_columns_are_non_nan() -> None:
    """Sanity floor: the four leakage-probed columns must be populated in
    the baseline, else the equality assertions below are vacuous."""
    _assert_columns_non_nan(_baseline())


def test_future_weekly_stats_do_not_leak_into_trajectory() -> None:
    """Injecting an implausible week-9 (= as_of_week) weekly_stats row must
    not move the trailing-4-vs-prior-4 volume trend, which is computed from
    weeks 1-8 only via the rolling .shift(1)/.shift(5)."""
    base = _baseline()
    _assert_columns_non_nan(base)

    ws = _weekly_stats()
    leaky = ws.copy()
    fut = (leaky["gsis_id"] == _WR1) & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[fut, ["receiving_yards", "targets", "receptions"]] = 999.0
    after = _build(leaky, _snap_counts(ws), _schedules())

    pd.testing.assert_frame_equal(base[_TRAJ_VEGAS_COLS], after[_TRAJ_VEGAS_COLS], check_like=True)


def test_future_snap_counts_do_not_leak_into_trajectory() -> None:
    """Injecting an implausible week-9 snap_counts row must not move the
    trailing-4-vs-prior-4 snap-pct change (computed from weeks 1-8 only)."""
    base = _baseline()
    _assert_columns_non_nan(base)

    ws = _weekly_stats()
    sc = _snap_counts(ws)
    leaky = sc.copy()
    fut = (leaky["gsis_id"] == _WR1) & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[fut, "offense_pct"] = 0.0  # implausible: WR1 vanishes in week 9
    after = _build(ws, leaky, _schedules())

    pd.testing.assert_frame_equal(base[_TRAJ_VEGAS_COLS], after[_TRAJ_VEGAS_COLS], check_like=True)


def test_current_week_schedule_does_not_leak_into_vegas() -> None:
    """season_avg_* at week 9 is the expanding mean of weeks 1..8 (leak-safe
    .shift(1)). Mutating the week-9 (= as_of_week) line — the row whose own
    value the leak-safe shift must exclude — must not move season_avg_*.

    This is the corrected probe direction: the prior version appended a
    FUTURE week-6 row that a week-5 computation could never consume, so it
    proved nothing."""
    base = _baseline()
    _assert_columns_non_nan(base)

    sch = _schedules()
    leaky = sch.copy()
    cur = leaky["week"] == _AS_OF_WEEK
    leaky.loc[cur, ["total_line", "spread_line"]] = 999.0
    after = _build(_weekly_stats(), _snap_counts(_weekly_stats()), leaky)

    pd.testing.assert_frame_equal(base[_TRAJ_VEGAS_COLS], after[_TRAJ_VEGAS_COLS], check_like=True)
