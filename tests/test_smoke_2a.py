"""End-to-end smoke test for Plan 2a deliverables.

Wires every new ingest module and the WR feature builder together against
synthetic fixtures. Catches integration gaps per-module tests miss:
- write_partition / read_partition path conventions matching
- Manifest update behavior across multiple tables
- Dtype drift between ingest output and feature input
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.features import build_wr_features
from projections.ingest import (
    build_id_map,
    refresh_depth_charts,
    refresh_ngs,
    refresh_schedules,
    refresh_snap_counts,
    refresh_weekly_stats,
)
from projections.ingest.manifest import read_manifest
from projections.schemas import WrFeaturesSchema
from projections.store import read_partition


def test_end_to_end_ingest_and_wr_features(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_weekly_df: pd.DataFrame,
    fake_schedules_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    fake_depth_charts_df: pd.DataFrame,
    fake_ngs_receiving_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest every Plan 2a table from synthetic fixtures, read the partitions
    back through `read_partition`, and feed them into `build_wr_features`.

    Asserts:
    1. Every ingest table writes a manifest entry.
    2. Every partition can be read back through `read_partition`.
    3. The WR feature builder consumes the round-tripped frames without
       schema-validation failures (catches dtype drift through parquet).
    4. The output contains the lone WR (Jefferson) from the fixtures.
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
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_ngs_receiving_df,
    )

    # 1) Build id_map first — required by snap_counts ingest for the
    # pfr_id -> gsis_id resolution.
    build_id_map(tmp_path)

    # 2) Ingest every table for season 2024.
    refresh_weekly_stats(tmp_path, seasons=[2024])
    refresh_schedules(tmp_path, seasons=[2024])
    refresh_snap_counts(tmp_path, seasons=[2024])
    refresh_depth_charts(tmp_path, seasons=[2024])
    refresh_ngs(tmp_path, stat_type="receiving", seasons=[2024])

    # 3) Manifest has one row per ingest table (id_map's row has season=NA).
    manifest = read_manifest(tmp_path)
    tables_in_manifest = set(manifest["table"].tolist())
    assert {
        "id_map",
        "weekly_stats",
        "schedules",
        "snap_counts",
        "depth_charts",
        "ngs_receiving",
    } <= tables_in_manifest

    # 4) Read each partition back through the sanctioned reader.
    weekly = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    schedules = read_partition(tmp_path / "raw", "schedules", season=2024)
    snaps = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    depth = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    ngs = read_partition(tmp_path / "raw", "ngs_receiving", season=2024)

    # 5) The synthetic fixtures all describe week 3 of 2024. The WR feature
    # builder requires depth chart + schedule rows for the *as_of_week*
    # itself (those are filtered with `_exact_week_mask`). Compute features
    # for as_of_week=4 so week 3 lands in the prior window; inject week-4
    # depth/schedule rows from the week-3 ones.
    extra_dc = pd.concat([depth, depth.assign(week=4)], ignore_index=True)
    extra_sched = pd.concat([schedules, schedules.assign(week=4)], ignore_index=True)

    # 6) Build WR features for as_of_week=4. Round-tripped frames go in;
    # any parquet dtype regression surfaces here at WrFeaturesSchema.validate().
    out = build_wr_features(
        weekly_stats=weekly,
        snap_counts=snaps,
        depth_charts=extra_dc,
        ngs_receiving=ngs,
        schedules=extra_sched,
        season=2024,
        as_of_week=4,
    )

    # 7) Output validates and has at least one row (Jefferson is the lone WR
    # in the fixtures).
    WrFeaturesSchema.validate(out)
    assert len(out) >= 1
    assert "00-0036322" in out["gsis_id"].tolist()
