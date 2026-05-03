"""Synthetic-fixture tests for trajectory_features.

Each compute fn is exercised against hand-rolled DataFrames; no real
weekly_stats / snap_counts / draft_picks parquets are read.
"""

from __future__ import annotations

# numpy / pandas / pytest are scaffolding for fixtures + per-compute tests
# added in subsequent tasks (Tasks 5-12); kept imported now so each task is
# a pure addition without re-importing.
import numpy as np  # noqa: F401  # used in subsequent tasks
import pandas as pd
import pytest

from projections.features.trajectory_features import (
    DraftLookup,
)
from projections.schemas import Position


def _ws_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    position: str = "QB",
    team: str = "KC",
    opponent: str = "BUF",
    attempts: int = 30,
    completions: int = 20,
    sacks: int = 2,
    passing_yards: float = 250.0,
    passing_tds: int = 2,
    interceptions: int = 0,
    rushing_yards: float = 10.0,
    rushing_tds: int = 0,
    carries: int = 3,
    receptions: int = 0,
    receiving_yards: float = 0.0,
    receiving_tds: int = 0,
    receiving_air_yards: float = 0.0,
    targets: int = 0,
    fumbles_lost: int = 0,
) -> dict[str, object]:
    """Helper: one weekly_stats row with sensible defaults."""
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "attempts": attempts,
        "completions": completions,
        "sacks": sacks,
        "passing_yards": passing_yards,
        "passing_tds": passing_tds,
        "interceptions": interceptions,
        "rushing_yards": rushing_yards,
        "rushing_tds": rushing_tds,
        "carries": carries,
        "receptions": receptions,
        "receiving_yards": receiving_yards,
        "receiving_tds": receiving_tds,
        "receiving_air_yards": receiving_air_yards,
        "targets": targets,
        "fumbles_lost": fumbles_lost,
    }


def _draft_lookup(*entries: tuple[str, int, float]) -> DraftLookup:
    return {gsis_id: (year, age) for gsis_id, year, age in entries}


def _snap_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    position: str = "WR",
    team: str = "KC",
    opponent: str = "BUF",
    offense_snaps: int = 50,
    offense_pct: float = 0.7,
    defense_snaps: int = 0,
    defense_pct: float = 0.0,
    st_snaps: int = 0,
    st_pct: float = 0.0,
) -> dict[str, object]:
    """Helper: one snap_counts row with sensible defaults."""
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "offense_snaps": offense_snaps,
        "offense_pct": offense_pct,
        "defense_snaps": defense_snaps,
        "defense_pct": defense_pct,
        "st_snaps": st_snaps,
        "st_pct": st_pct,
    }


def test_module_imports() -> None:
    """Smoke: confirm the module loads cleanly."""
    from projections.features import trajectory_features  # noqa: F401


def test_compute_age_uses_draft_age_when_available() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
            _ws_row(gsis_id="00-0033873", season=2018, week=2),
            _ws_row(gsis_id="00-0033873", season=2024, week=1),
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_age(weekly_stats, lookup)
    # One row per (gsis_id, season).
    assert len(out) == 2
    assert set(out.columns) == {"gsis_id", "season", "age", "draft_year_inferred"}
    age_2018 = out[out["season"] == 2018]["age"].iloc[0]
    age_2024 = out[out["season"] == 2024]["age"].iloc[0]
    assert age_2018 == pytest.approx(22.5)  # 21.5 + (2018 - 2017)
    assert age_2024 == pytest.approx(28.5)  # 21.5 + (2024 - 2017)
    assert (~out["draft_year_inferred"]).all()
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["age"].dtype == pd.Float64Dtype()
    assert out["draft_year_inferred"].dtype == pd.BooleanDtype()


def test_compute_age_falls_back_for_udfa() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0099999", season=2020, week=1),
            _ws_row(gsis_id="00-0099999", season=2024, week=1),
        ]
    )
    # No entry in the lookup → UDFA path.
    lookup: DraftLookup = {}
    out = compute_age(weekly_stats, lookup)
    assert len(out) == 2
    age_2020 = out[out["season"] == 2020]["age"].iloc[0]
    age_2024 = out[out["season"] == 2024]["age"].iloc[0]
    # inferred_draft_year = 2020 (earliest); age = season - 2020 + 22.0
    assert age_2020 == pytest.approx(22.0)
    assert age_2024 == pytest.approx(26.0)
    assert out["draft_year_inferred"].all()


def test_compute_age_falls_back_when_draft_age_is_nan() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
        ]
    )
    # Drafted but no draft_age — fall back to inferred path.
    lookup = _draft_lookup(("00-0033873", 2017, float("nan")))
    out = compute_age(weekly_stats, lookup)
    age_2018 = out[out["season"] == 2018]["age"].iloc[0]
    # inferred_draft_year = 2018 (earliest); 2018 - 2018 + 22 = 22.0
    assert age_2018 == pytest.approx(22.0)
    assert out["draft_year_inferred"].all()


def test_compute_age_one_row_per_player_season() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        [_ws_row(gsis_id="00-0033873", season=2018, week=w) for w in range(1, 18)]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_age(weekly_stats, lookup)
    assert len(out) == 1


def test_compute_age_empty_input() -> None:
    from projections.features.trajectory_features import compute_age

    weekly_stats = pd.DataFrame(
        columns=["gsis_id", "season", "week", "position", "team", "opponent"]
    )
    lookup: DraftLookup = {}
    out = compute_age(weekly_stats, lookup)
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "age", "draft_year_inferred"}
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["age"].dtype == pd.Float64Dtype()
    assert out["draft_year_inferred"].dtype == pd.BooleanDtype()


def test_compute_is_rookie_marks_drafted_player_in_draft_year() -> None:
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2017, week=1),
            _ws_row(gsis_id="00-0033873", season=2018, week=1),
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_is_rookie(weekly_stats, lookup)
    assert len(out) == 2
    assert set(out.columns) == {"gsis_id", "season", "is_rookie"}
    rookie_2017 = out[out["season"] == 2017]["is_rookie"].iloc[0]
    rookie_2018 = out[out["season"] == 2018]["is_rookie"].iloc[0]
    assert rookie_2017 == 1.0
    assert rookie_2018 == 0.0
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["is_rookie"].dtype == pd.Float64Dtype()


def test_compute_is_rookie_uses_inferred_year_for_udfa() -> None:
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0099999", season=2020, week=1),
            _ws_row(gsis_id="00-0099999", season=2021, week=1),
        ]
    )
    lookup: DraftLookup = {}
    out = compute_is_rookie(weekly_stats, lookup)
    rookie_2020 = out[out["season"] == 2020]["is_rookie"].iloc[0]
    rookie_2021 = out[out["season"] == 2021]["is_rookie"].iloc[0]
    assert rookie_2020 == 1.0
    assert rookie_2021 == 0.0


def test_compute_is_rookie_dtype_is_float() -> None:
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2017, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    out = compute_is_rookie(weekly_stats, lookup)
    # Float (not bool) for ML-compat; matches schema dtype.
    assert out["is_rookie"].dtype == pd.Float64Dtype()


def test_compute_is_rookie_empty_input() -> None:
    from projections.features.trajectory_features import compute_is_rookie

    weekly_stats = pd.DataFrame(
        columns=["gsis_id", "season", "week", "position", "team", "opponent"]
    )
    out = compute_is_rookie(weekly_stats, {})
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "is_rookie"}
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["is_rookie"].dtype == pd.Float64Dtype()


def test_compute_qb_volume_trend_basic_arithmetic() -> None:
    from projections.features.trajectory_features import compute_qb_volume_trend

    # 9 weeks of attempts: weeks 1-4 = 20,22,24,26 (mean 23 = prior_l4 at week 9),
    # weeks 5-8 = 30,32,34,36 (mean 33 = l4 at week 9). Trend = 33 - 23 = 10.
    attempts_by_week = {1: 20, 2: 22, 3: 24, 4: 26, 5: 30, 6: 32, 7: 34, 8: 36, 9: 40}
    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=w, attempts=a)
            for w, a in attempts_by_week.items()
        ]
    )
    out = compute_qb_volume_trend(weekly_stats)
    week9 = out[(out["season"] == 2018) & (out["week"] == 9)][
        "volume_trend_l4_minus_prior_l4"
    ].iloc[0]
    assert week9 == pytest.approx(10.0)
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["week"].dtype == pd.Int64Dtype()
    assert out["volume_trend_l4_minus_prior_l4"].dtype == pd.Float64Dtype()


def test_compute_qb_volume_trend_nan_before_8_prior_games() -> None:
    from projections.features.trajectory_features import compute_qb_volume_trend

    weekly_stats = pd.DataFrame(
        [_ws_row(gsis_id="00-0033873", season=2018, week=w, attempts=20 + w) for w in range(1, 10)]
    )
    out = compute_qb_volume_trend(weekly_stats).sort_values("week").reset_index(drop=True)
    # Weeks 1-8 lack 8 prior active games → NaN.
    for w in range(1, 9):
        val = out[out["week"] == w]["volume_trend_l4_minus_prior_l4"].iloc[0]
        assert pd.isna(val)
    # Week 9 should be a finite number.
    val_w9 = out[out["week"] == 9]["volume_trend_l4_minus_prior_l4"].iloc[0]
    assert pd.notna(val_w9)


def test_compute_qb_volume_trend_filters_position() -> None:
    from projections.features.trajectory_features import compute_qb_volume_trend

    qb_rows = [
        _ws_row(gsis_id="00-0033873", season=2018, week=w, attempts=30, position="QB")
        for w in range(1, 10)
    ]
    rb_rows = [
        _ws_row(gsis_id="00-0099999", season=2018, week=w, attempts=15, position="RB")
        for w in range(1, 10)
    ]
    weekly_stats = pd.DataFrame(qb_rows + rb_rows)
    out = compute_qb_volume_trend(weekly_stats)
    assert set(out["gsis_id"].unique()) == {"00-0033873"}


def test_compute_qb_volume_trend_crosses_season_boundary() -> None:
    from projections.features.trajectory_features import compute_qb_volume_trend

    # 2018 weeks 1-8: prior-4 will end up being weeks 1-4 (20,22,24,26 → mean 23),
    # 2018 weeks 5-8 (30,32,34,36 → mean 33), so 2019 week 1 trend = 33 - 23 = 10.
    rows = []
    attempts_2018 = [20, 22, 24, 26, 30, 32, 34, 36]
    for i, a in enumerate(attempts_2018, start=1):
        rows.append(_ws_row(gsis_id="00-0033873", season=2018, week=i, attempts=a))
    rows.append(_ws_row(gsis_id="00-0033873", season=2019, week=1, attempts=40))
    weekly_stats = pd.DataFrame(rows)
    out = compute_qb_volume_trend(weekly_stats)
    val = out[(out["season"] == 2019) & (out["week"] == 1)]["volume_trend_l4_minus_prior_l4"].iloc[
        0
    ]
    assert val == pytest.approx(10.0)


def test_compute_qb_volume_trend_traded_player_unbroken_window() -> None:
    from projections.features.trajectory_features import compute_qb_volume_trend

    # Same player, team change at week 5. Trend computes by gsis_id, so window unbroken.
    attempts_by_week = {1: 20, 2: 22, 3: 24, 4: 26, 5: 30, 6: 32, 7: 34, 8: 36, 9: 40}
    rows = []
    for w, a in attempts_by_week.items():
        team = "KC" if w <= 4 else "DEN"
        rows.append(_ws_row(gsis_id="00-0033873", season=2018, week=w, attempts=a, team=team))
    weekly_stats = pd.DataFrame(rows)
    out = compute_qb_volume_trend(weekly_stats)
    week9 = out[(out["season"] == 2018) & (out["week"] == 9)][
        "volume_trend_l4_minus_prior_l4"
    ].iloc[0]
    # Same arithmetic as the basic case — window is per-gsis_id, not per-team.
    assert week9 == pytest.approx(10.0)


def test_compute_qb_volume_trend_empty_input() -> None:
    from projections.features.trajectory_features import compute_qb_volume_trend

    weekly_stats = pd.DataFrame(
        columns=["gsis_id", "season", "week", "position", "team", "opponent", "attempts"]
    )
    out = compute_qb_volume_trend(weekly_stats)
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "week", "volume_trend_l4_minus_prior_l4"}
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["week"].dtype == pd.Int64Dtype()
    assert out["volume_trend_l4_minus_prior_l4"].dtype == pd.Float64Dtype()


def test_compute_rb_volume_trend_uses_carries() -> None:
    from projections.features.trajectory_features import compute_rb_volume_trend

    rows = []
    carries_by_week = {1: 5, 2: 7, 3: 9, 4: 11, 5: 15, 6: 17, 7: 19, 8: 21, 9: 25}
    for w, c in carries_by_week.items():
        rows.append(_ws_row(gsis_id="00-0033873", season=2018, week=w, position="RB", carries=c))
    weekly_stats = pd.DataFrame(rows)
    out = compute_rb_volume_trend(weekly_stats)
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    # l4 = mean(15,17,19,21) = 18; prior_l4 = mean(5,7,9,11) = 8; trend = 10.
    assert week_9["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(10.0)


def test_compute_rb_volume_trend_filters_to_rb_only() -> None:
    from projections.features.trajectory_features import compute_rb_volume_trend

    rows = []
    rows += [
        _ws_row(gsis_id="00-0099001", season=2018, week=w, position="RB", carries=10)
        for w in range(1, 10)
    ]
    rows += [
        _ws_row(gsis_id="00-0099002", season=2018, week=w, position="QB", carries=2)
        for w in range(1, 10)
    ]
    out = compute_rb_volume_trend(pd.DataFrame(rows))
    assert set(out["gsis_id"].unique()) == {"00-0099001"}


def test_compute_wr_te_volume_trend_uses_targets_for_wr() -> None:
    from projections.features.trajectory_features import compute_wr_te_volume_trend

    rows = []
    targets_by_week = {1: 4, 2: 6, 3: 8, 4: 10, 5: 12, 6: 14, 7: 16, 8: 18, 9: 20}
    for w, t in targets_by_week.items():
        rows.append(_ws_row(gsis_id="00-0033873", season=2018, week=w, position="WR", targets=t))
    weekly_stats = pd.DataFrame(rows)
    out = compute_wr_te_volume_trend(weekly_stats)
    week_9 = out[(out["season"] == 2018) & (out["week"] == 9)]
    # l4 = mean(12,14,16,18) = 15; prior_l4 = mean(4,6,8,10) = 7; trend = 8.
    assert week_9["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(8.0)


def test_compute_wr_te_volume_trend_includes_te() -> None:
    from projections.features.trajectory_features import compute_wr_te_volume_trend

    rows = []
    rows += [
        _ws_row(gsis_id="00-0099001", season=2018, week=w, position="WR", targets=10)
        for w in range(1, 10)
    ]
    rows += [
        _ws_row(gsis_id="00-0099002", season=2018, week=w, position="TE", targets=8)
        for w in range(1, 10)
    ]
    out = compute_wr_te_volume_trend(pd.DataFrame(rows))
    assert set(out["gsis_id"].unique()) == {"00-0099001", "00-0099002"}


def test_compute_wr_te_volume_trend_excludes_rb() -> None:
    from projections.features.trajectory_features import compute_wr_te_volume_trend

    rows = []
    rows += [
        _ws_row(gsis_id="00-0099001", season=2018, week=w, position="WR", targets=10)
        for w in range(1, 10)
    ]
    rows += [
        _ws_row(gsis_id="00-0099002", season=2018, week=w, position="RB", targets=4)
        for w in range(1, 10)
    ]
    out = compute_wr_te_volume_trend(pd.DataFrame(rows))
    assert set(out["gsis_id"].unique()) == {"00-0099001"}


def test_compute_snap_pct_change_basic() -> None:
    from projections.features.trajectory_features import compute_snap_pct_change

    # 9 weeks of offense_pct: weeks 1-4 = 0.5 (prior_l4 at week 9),
    # weeks 5-8 = 0.7 (l4 at week 9). Change = 0.7 - 0.5 = 0.2.
    pct_by_week = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.7, 6: 0.7, 7: 0.7, 8: 0.7, 9: 0.8}
    snap_counts = pd.DataFrame(
        [
            _snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=p)
            for w, p in pct_by_week.items()
        ]
    )
    out = compute_snap_pct_change(snap_counts)
    week9 = out[(out["season"] == 2018) & (out["week"] == 9)][
        "snap_pct_change_l4_vs_prior_l4"
    ].iloc[0]
    assert week9 == pytest.approx(0.2)
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["week"].dtype == pd.Int64Dtype()
    assert out["snap_pct_change_l4_vs_prior_l4"].dtype == pd.Float64Dtype()


def test_compute_snap_pct_change_nan_before_8_prior_games() -> None:
    from projections.features.trajectory_features import compute_snap_pct_change

    snap_counts = pd.DataFrame(
        [
            _snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.5 + 0.01 * w)
            for w in range(1, 10)
        ]
    )
    out = compute_snap_pct_change(snap_counts).sort_values("week").reset_index(drop=True)
    for w in range(1, 9):
        val = out[out["week"] == w]["snap_pct_change_l4_vs_prior_l4"].iloc[0]
        assert pd.isna(val)
    val_w9 = out[out["week"] == 9]["snap_pct_change_l4_vs_prior_l4"].iloc[0]
    assert pd.notna(val_w9)


def test_compute_snap_pct_change_inactive_week_excluded_from_window() -> None:
    from projections.features.trajectory_features import compute_snap_pct_change

    # Player has rows for weeks 1-4 (0.5) and weeks 6-10 (0.8) but NOT week 5.
    # Window operates over the 9 active games. Row at week 10:
    #   l4 = mean(weeks 6,7,8,9) = 0.8; prior_l4 = mean(weeks 1-4) = 0.5; change = 0.3.
    rows = []
    for w in range(1, 5):
        rows.append(_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.5))
    for w in range(6, 11):
        rows.append(_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.8))
    snap_counts = pd.DataFrame(rows)
    out = compute_snap_pct_change(snap_counts)
    week10 = out[(out["season"] == 2018) & (out["week"] == 10)][
        "snap_pct_change_l4_vs_prior_l4"
    ].iloc[0]
    assert week10 == pytest.approx(0.3)


def test_compute_snap_pct_change_crosses_season_boundary() -> None:
    from projections.features.trajectory_features import compute_snap_pct_change

    # 4 games in 2018 (0.5), 4 games later in 2018 (0.7), 1 game in 2019 week 1 (0.8).
    # Week 1 of 2019: l4 = 0.7, prior_l4 = 0.5, change = 0.2.
    rows = []
    for w in range(1, 5):
        rows.append(_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.5))
    for w in range(5, 9):
        rows.append(_snap_row(gsis_id="00-0033873", season=2018, week=w, offense_pct=0.7))
    rows.append(_snap_row(gsis_id="00-0033873", season=2019, week=1, offense_pct=0.8))
    snap_counts = pd.DataFrame(rows)
    out = compute_snap_pct_change(snap_counts)
    val = out[(out["season"] == 2019) & (out["week"] == 1)]["snap_pct_change_l4_vs_prior_l4"].iloc[
        0
    ]
    assert val == pytest.approx(0.2)


def test_compute_snap_pct_change_empty_input() -> None:
    from projections.features.trajectory_features import compute_snap_pct_change

    snap_counts = pd.DataFrame(
        columns=[
            "gsis_id",
            "season",
            "week",
            "position",
            "team",
            "opponent",
            "offense_snaps",
            "offense_pct",
            "defense_snaps",
            "defense_pct",
            "st_snaps",
            "st_pct",
        ]
    )
    out = compute_snap_pct_change(snap_counts)
    assert out.empty
    assert set(out.columns) == {"gsis_id", "season", "week", "snap_pct_change_l4_vs_prior_l4"}
    assert out["gsis_id"].dtype == pd.StringDtype("pyarrow")
    assert out["season"].dtype == pd.Int64Dtype()
    assert out["week"].dtype == pd.Int64Dtype()
    assert out["snap_pct_change_l4_vs_prior_l4"].dtype == pd.Float64Dtype()


def test_attach_trajectory_features_appends_4_cols_qb() -> None:
    from projections.features.trajectory_features import attach_trajectory_features

    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=w, position="QB", attempts=20 + w)
            for w in range(1, 10)
        ]
    )
    snap_counts = pd.DataFrame(
        [
            _snap_row(
                gsis_id="00-0033873", season=2018, week=w, position="QB", offense_pct=0.5 + w * 0.01
            )
            for w in range(1, 10)
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    index = pd.DataFrame(
        [
            {"gsis_id": "00-0033873", "season": 2018, "week": 9, "team": "KC", "opp": "BUF"},
        ]
    )
    out = attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.QB)

    assert len(out) == 1
    expected_added = {
        "age",
        "is_rookie",
        "volume_trend_l4_minus_prior_l4",
        "snap_pct_change_l4_vs_prior_l4",
        "draft_year_inferred",
    }
    assert expected_added <= set(out.columns)


def test_attach_trajectory_features_uses_correct_volume_trend_per_position() -> None:
    from projections.features.trajectory_features import attach_trajectory_features

    # RB row at week 9 with carries trend.
    weekly_stats = pd.DataFrame(
        [
            _ws_row(gsis_id="00-0033873", season=2018, week=w, position="RB", carries=c, attempts=0)
            for w, c in [
                (1, 5),
                (2, 7),
                (3, 9),
                (4, 11),
                (5, 15),
                (6, 17),
                (7, 19),
                (8, 21),
                (9, 25),
            ]
        ]
    )
    snap_counts = pd.DataFrame(
        [
            _snap_row(gsis_id="00-0033873", season=2018, week=w, position="RB", offense_pct=0.6)
            for w in range(1, 10)
        ]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 22.0))
    index = pd.DataFrame(
        [{"gsis_id": "00-0033873", "season": 2018, "week": 9, "team": "KC", "opp": "BUF"}]
    )
    out = attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.RB)
    # Carries trend at week 9 = 18 - 8 = 10.
    assert out["volume_trend_l4_minus_prior_l4"].iloc[0] == pytest.approx(10.0)


def test_attach_trajectory_features_preserves_index_columns() -> None:
    from projections.features.trajectory_features import attach_trajectory_features

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1)])
    snap_counts = pd.DataFrame([_snap_row(gsis_id="00-0033873", season=2018, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    index = pd.DataFrame(
        [{"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF"}]
    )
    out = attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.QB)
    for col in ("gsis_id", "season", "week", "team", "opp"):
        assert col in out.columns


def test_attach_trajectory_features_rejects_invalid_position() -> None:
    from projections.features.trajectory_features import attach_trajectory_features

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1)])
    snap_counts = pd.DataFrame([_snap_row(gsis_id="00-0033873", season=2018, week=1)])
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    index = pd.DataFrame(
        [{"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF"}]
    )
    with pytest.raises(ValueError, match="position"):
        attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.K)


def test_attach_trajectory_features_preserves_dtypes_on_merge_miss() -> None:
    """An index row with no matching weekly_stats / draft_lookup entry must
    not regress dtypes. Specifically, draft_year_inferred must remain
    BooleanDtype (not object) when the left-merge introduces NaN.
    Regression test for a bug surfaced in Task 11's code review.
    """
    from projections.features.trajectory_features import attach_trajectory_features

    weekly_stats = pd.DataFrame([_ws_row(gsis_id="00-0033873", season=2018, week=1, position="QB")])
    snap_counts = pd.DataFrame(
        [_snap_row(gsis_id="00-0033873", season=2018, week=1, position="QB")]
    )
    lookup = _draft_lookup(("00-0033873", 2017, 21.5))
    # Index has TWO rows: one with a player who has data, one with a phantom
    # player who doesn't appear in weekly_stats / snap_counts / lookup.
    index = pd.DataFrame(
        [
            {"gsis_id": "00-0033873", "season": 2018, "week": 1, "team": "KC", "opp": "BUF"},
            {"gsis_id": "00-0099999", "season": 2018, "week": 1, "team": "BUF", "opp": "KC"},
        ]
    )
    out = attach_trajectory_features(index, weekly_stats, snap_counts, lookup, Position.QB)
    assert len(out) == 2
    # Critical: the phantom row's columns are NaN/NA, but the dtype stays
    # nullable rather than object.
    assert out["draft_year_inferred"].dtype == pd.BooleanDtype()
    assert out["age"].dtype == pd.Float64Dtype()
    assert out["is_rookie"].dtype == pd.Float64Dtype()
