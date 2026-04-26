"""End-to-end smoke test for ingest + feature-builder integration.

Wires every ingest module and all 4 position builders (QB/RB/WR/TE)
against synthetic fixtures. Catches integration gaps per-module tests
miss:
- write_partition / read_partition path conventions matching
- Manifest update behavior across multiple tables
- Dtype drift between ingest output and feature input
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import projections
from projections.features import (
    build_qb_features,
    build_rb_features,
    build_te_features,
    build_wr_features,
)
from projections.ingest import (
    build_id_map,
    refresh_depth_charts,
    refresh_ngs,
    refresh_schedules,
    refresh_snap_counts,
    refresh_weekly_stats,
)
from projections.ingest.manifest import read_manifest
from projections.schemas import (
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)
from projections.store import read_partition


def test_package_imports() -> None:
    """Sanity check that the top-level package is importable and versioned."""
    assert projections.__version__ == "0.0.1"


def test_end_to_end_ingest_and_features(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_weekly_df: pd.DataFrame,
    fake_schedules_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    fake_depth_charts_df: pd.DataFrame,
    fake_ngs_passing_df: pd.DataFrame,
    fake_ngs_rushing_df: pd.DataFrame,
    fake_ngs_receiving_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest every Plan 2a/2b table from synthetic fixtures, read the
    partitions back through `read_partition`, and feed them into all 4
    position feature builders.

    Asserts:
    1. Every ingest table writes a manifest entry.
    2. Every partition can be read back through `read_partition`.
    3. Each of the QB/RB/WR/TE feature builders consumes the round-tripped
       frames without schema-validation failures (catches dtype drift
       through parquet).
    4. Each output contains the expected fixture player by gsis_id.
    """
    # Patch every fetcher to return the synthetic fixture instead of hitting
    # the network.
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map",
        lambda: fake_id_map_df,
    )
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: fake_snap_counts_df,
    )
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    ngs_fixtures = {
        "passing": fake_ngs_passing_df,
        "rushing": fake_ngs_rushing_df,
        "receiving": fake_ngs_receiving_df,
    }
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda stat_type, seasons: ngs_fixtures[stat_type],
    )

    # 1) Build id_map first — required by snap_counts ingest for the
    # pfr_id -> gsis_id resolution.
    build_id_map(tmp_path)

    # 2) Ingest every table for season 2024.
    refresh_weekly_stats(tmp_path, seasons=[2024])
    refresh_schedules(tmp_path, seasons=[2024])
    refresh_snap_counts(tmp_path, seasons=[2024])
    refresh_depth_charts(tmp_path, seasons=[2024])
    for stat_type in ("passing", "rushing", "receiving"):
        refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])

    # 3) Manifest has one row per ingest table.
    manifest = read_manifest(tmp_path)
    tables_in_manifest = set(manifest["table"].tolist())
    expected = {
        "weekly_stats",
        "schedules",
        "snap_counts",
        "depth_charts",
        "ngs_passing",
        "ngs_rushing",
        "ngs_receiving",
    }
    assert expected <= tables_in_manifest

    # 4) Read each partition back through the sanctioned reader.
    weekly = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    schedules = read_partition(tmp_path / "raw", "schedules", season=2024)
    snaps = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    depth = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    ngs_passing = read_partition(tmp_path / "raw", "ngs_passing", season=2024)
    ngs_rushing = read_partition(tmp_path / "raw", "ngs_rushing", season=2024)
    ngs_receiving = read_partition(tmp_path / "raw", "ngs_receiving", season=2024)

    # 5) Fixtures all describe week 3 of 2024. The feature builders require
    # depth chart + schedule rows for the *as_of_week* itself (filtered with
    # `exact_week_mask`). Compute features for as_of_week=4 so week 3 lands
    # in the prior window; inject week-4 depth/schedule rows from the week-3
    # ones.
    extra_dc = pd.concat([depth, depth.assign(week=4)], ignore_index=True)
    extra_sched = pd.concat([schedules, schedules.assign(week=4)], ignore_index=True)

    # 6) Build all 4 position features for as_of_week=4. Round-tripped frames
    # go in; any parquet dtype regression surfaces here at *.validate().
    qb_out = build_qb_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_passing=ngs_passing,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )
    rb_out = build_rb_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_rushing=ngs_rushing,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )
    wr_out = build_wr_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_receiving=ngs_receiving,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )
    te_out = build_te_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_receiving=ngs_receiving,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )

    # 7) Each output validates and contains the expected fixture player.
    QbFeaturesSchema.validate(qb_out)
    assert "00-0034857" in qb_out["gsis_id"].tolist()  # Mahomes

    RbFeaturesSchema.validate(rb_out)
    assert "00-0034796" in rb_out["gsis_id"].tolist()  # Barkley

    WrFeaturesSchema.validate(wr_out)
    assert "00-0036322" in wr_out["gsis_id"].tolist()  # Jefferson

    TeFeaturesSchema.validate(te_out)
    assert "00-0030506" in te_out["gsis_id"].tolist()  # Kelce


def test_smoke_wr_baseline_fit_predict_write(
    tmp_path: Path,
    baseline_features_wr: pd.DataFrame,
    baseline_weekly_stats_wr: pd.DataFrame,
) -> None:
    """End-to-end: fit BaselineModel on synthetic data, predict, write a
    parquet partition through store.write_partition, read back, validate."""
    from projections.models import wr_baseline
    from projections.schemas import ProjectionWeeklySchema, Ruleset
    from projections.store import read_partition, write_partition

    model = wr_baseline()
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)

    week_features = baseline_features_wr[
        (baseline_features_wr["season"] == 2025) & (baseline_features_wr["week"] == 4)
    ]
    preds = model.predict_distribution(week_features, ruleset=Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(preds)

    write_partition(
        tmp_path / "projections",
        "weekly/ruleset=ESPN_PPR",
        preds,
        season=2025,
        week=4,
    )
    round_tripped = read_partition(
        tmp_path / "projections", "weekly/ruleset=ESPN_PPR", season=2025, week=4
    )
    ProjectionWeeklySchema.validate(round_tripped)
    assert len(round_tripped) == len(preds)
