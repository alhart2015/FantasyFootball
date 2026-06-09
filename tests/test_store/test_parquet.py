"""Parquet store tests — partitioned writes/reads, idempotency."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.store import read_partition, write_partition


def _frame(season: int, week: int, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": [f"00-00{i:05d}" for i in range(n)],
            "season": [season] * n,
            "week": [week] * n,
            "value": [float(i) for i in range(n)],
        }
    )


def test_write_and_read_single_partition(tmp_path: Path) -> None:
    df = _frame(2024, 3)
    write_partition(tmp_path, "weekly_stats", df, season=2024, week=3)
    out = read_partition(tmp_path, "weekly_stats", season=2024, week=3)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True), check_like=True
    )


def test_write_idempotent_overwrites_partition(tmp_path: Path) -> None:
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3, n=5), season=2024, week=3)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3, n=2), season=2024, week=3)
    out = read_partition(tmp_path, "weekly_stats", season=2024, week=3)
    assert len(out) == 2  # second write replaces first


def test_read_across_weeks(tmp_path: Path) -> None:
    write_partition(tmp_path, "weekly_stats", _frame(2024, 1), season=2024, week=1)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 2), season=2024, week=2)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3), season=2024, week=3)
    all_2024 = read_partition(tmp_path, "weekly_stats", season=2024)
    assert sorted(all_2024["week"].unique().tolist()) == [1, 2, 3]


def test_season_only_partition(tmp_path: Path) -> None:
    df = pd.DataFrame({"gsis_id": ["00-0036322"], "espn_id": ["4262921"]})
    write_partition(tmp_path, "id_map", df, season=None, week=None)
    out = read_partition(tmp_path, "id_map", season=None, week=None)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True), check_like=True
    )


def test_write_read_asof_partition_roundtrip(tmp_path: Path) -> None:
    from projections.store.parquet import read_partition, write_partition

    df = pd.DataFrame({"gsis_id": ["00-0000001"], "adp": [3.5]})
    p = write_partition(tmp_path, "external_projections", df, season=2026, asof=date(2026, 7, 15))
    expected = (
        tmp_path / "external_projections" / "season=2026" / "asof=2026-07-15" / "part.parquet"
    )
    assert p == expected
    back = read_partition(tmp_path, "external_projections", season=2026, asof=date(2026, 7, 15))
    assert back["adp"].tolist() == [3.5]


def test_season_only_read_of_asof_table_raises(tmp_path: Path) -> None:
    import pytest

    from projections.store.parquet import read_partition, write_partition

    # A season-only read of an asof-snapshotted table must raise, not silently concatenate
    # every dated snapshot (which would duplicate each player once per snapshot).
    write_partition(
        tmp_path,
        "external_projections",
        pd.DataFrame({"gsis_id": ["00-0000001"], "asof": ["2026-07-01"]}),
        season=2026,
        asof=date(2026, 7, 1),
    )
    write_partition(
        tmp_path,
        "external_projections",
        pd.DataFrame({"gsis_id": ["00-0000001"], "asof": ["2026-07-15"]}),
        season=2026,
        asof=date(2026, 7, 15),
    )
    with pytest.raises(ValueError, match="asof-snapshotted"):
        read_partition(tmp_path, "external_projections", season=2026)
    # explicit single-snapshot reads still work
    one = read_partition(tmp_path, "external_projections", season=2026, asof=date(2026, 7, 15))
    assert one["asof"].tolist() == ["2026-07-15"]


def test_read_latest_partition_returns_newest_asof(tmp_path: Path) -> None:
    from projections.store.parquet import read_latest_partition, write_partition

    write_partition(
        tmp_path,
        "external_projections",
        pd.DataFrame({"gsis_id": ["00-0000001"], "adp": [9.0]}),
        season=2026,
        asof=date(2026, 7, 1),
    )
    write_partition(
        tmp_path,
        "external_projections",
        pd.DataFrame({"gsis_id": ["00-0000001"], "adp": [4.0]}),
        season=2026,
        asof=date(2026, 7, 15),
    )
    latest = read_latest_partition(tmp_path, "external_projections", season=2026)
    assert latest["adp"].tolist() == [4.0]


def test_write_partition_season_week_unchanged(tmp_path: Path) -> None:
    from projections.store.parquet import read_partition, write_partition

    df = pd.DataFrame({"x": [1]})
    write_partition(tmp_path, "weekly_stats", df, season=2024, week=3)
    assert read_partition(tmp_path, "weekly_stats", season=2024, week=3)["x"].tolist() == [1]


def test_read_latest_ignores_stray_non_date_asof_dir(tmp_path: Path) -> None:
    from datetime import date

    from projections.store.parquet import read_latest_partition, write_partition

    write_partition(
        tmp_path,
        "external_projections",
        pd.DataFrame({"gsis_id": ["00-0000001"], "adp": [4.0]}),
        season=2026,
        asof=date(2026, 7, 15),
    )
    stray = tmp_path / "external_projections" / "season=2026" / "asof=backup"
    stray.mkdir(parents=True)
    (stray / "part.parquet").write_bytes(b"")  # would crash if selected
    latest = read_latest_partition(tmp_path, "external_projections", season=2026)
    assert latest["adp"].tolist() == [4.0]


def test_partition_dir_raises_on_week_and_asof_together(tmp_path: Path) -> None:
    import pytest

    from projections.store.parquet import write_partition

    with pytest.raises(ValueError, match="mutually exclusive"):
        write_partition(
            tmp_path,
            "some_table",
            pd.DataFrame({"x": [1]}),
            season=2026,
            week=3,
            asof=date(2026, 7, 15),
        )
