"""Tests for project_season._write_season_artifacts: writes three artifacts given
an in-memory weekly ProjectionWeeklySchema-validated frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from project_season import _write_season_artifacts

from projections.schemas import _PYARROW_STR, ProjectionWeeklySchema, Ruleset
from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

_RULESET = Ruleset.espn_ppr()


def _synthetic_id_map() -> pd.DataFrame:
    """Minimal id_map with just the columns _write_season_artifacts needs."""
    return pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-0033873", "00-0000001"], dtype=_PYARROW_STR),
            "full_name": pd.Series(["Patrick Mahomes", "Synthetic Player"], dtype=_PYARROW_STR),
            "team": pd.Series(["KC", "FA"], dtype=_PYARROW_STR),
        }
    )


def test_write_season_artifacts_emits_three_files(tmp_path: Path) -> None:
    weekly = _to_weekly_frame(
        [_build_weekly_row(gsis_id="00-0033873", week=w) for w in range(1, 5)]
        + [_build_weekly_row(gsis_id="00-0000001", week=w) for w in range(1, 5)]
    )
    _write_season_artifacts(
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    assert (tmp_path / "season_projection.csv").exists()
    assert (tmp_path / "season_projection_weekly_2024.parquet").exists()
    assert (tmp_path / "season_projection_distributions_2024.csv").exists()


def test_write_season_artifacts_naive_csv_columns_preserved(tmp_path: Path) -> None:
    """Back-compat: existing downstream surfaces consume this CSV by column name."""
    weekly = _to_weekly_frame([_build_weekly_row(week=1)])
    _write_season_artifacts(
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    naive = pd.read_csv(tmp_path / "season_projection.csv")
    assert set(naive.columns) >= {
        "rank",
        "gsis_id",
        "position",
        "season_total_mean",
        "n_weeks",
        "full_name",
        "team",
    }


def test_write_season_artifacts_distributions_csv_has_quantile_cols(tmp_path: Path) -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    _write_season_artifacts(
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    dist = pd.read_csv(tmp_path / "season_projection_distributions_2024.csv")
    assert set(dist.columns) >= {
        "gsis_id",
        "position",
        "season_mean",
        "season_p10",
        "season_p50",
        "season_p90",
        "n_weeks",
        "full_name",
        "team",
    }


def test_write_season_artifacts_weekly_parquet_validates(tmp_path: Path) -> None:
    """Weekly parquet round-trips through ProjectionWeeklySchema."""
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    _write_season_artifacts(
        weekly=weekly,
        ruleset=_RULESET,
        out_dir=tmp_path,
        season=2024,
        id_map=_synthetic_id_map(),
    )
    weekly_parquet = pd.read_parquet(tmp_path / "season_projection_weekly_2024.parquet")
    ProjectionWeeklySchema.validate(weekly_parquet)
