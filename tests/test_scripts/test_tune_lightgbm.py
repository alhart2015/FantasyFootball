"""End-to-end tests for scripts/tune_lightgbm.py — Plan 5b Phase 1.

The Optuna driver loads features + weekly_stats from the cache and the raw
parquet store. To exercise it without those caches, the tests monkeypatch
`_load_join_for_position` to return a synthetic in-memory join shaped
exactly like the production output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def tune_module() -> Any:
    """Import the script directly. tests/test_scripts/conftest.py adds
    scripts/ to sys.path; the same pattern is used by test_diagnose_calibration.py.
    """
    import tune_lightgbm

    return tune_lightgbm


# Real WR feature columns the LightGBMModel reads from its config — required
# so that run_studies(), which derives feat_cols from the model config, can
# index into the synthetic frame.
_WR_FEAT_COLUMNS: tuple[str, ...] = (
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "air_yards_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "designed_rusher",
    "snap_pct_l4",
    "depth_rank",
    "avg_separation_std",
    "avg_intended_air_yards_std",
    "percent_share_intended_air_yards_std",
    "avg_yac_above_expectation_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_wr_fppg_l4",
    # Trajectory features (Plan: WR trajectory). Bounded ranges enforced by
    # WrFeaturesSchema; values populated below in _build_synthetic_joined.
    "age",
    "is_rookie",
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
    # Weather features (Plan: 2026-05-08 RB+WR weather integration). All
    # nullable in WrFeaturesSchema; values populated below with random noise
    # like the other unbounded feature columns.
    "wind_speed_mph",
    "is_high_wind",
    "temperature_f",
    "is_grass_surface",
    # Vegas team-context features (PR #51 WR+QB Vegas integration). Bounded
    # ranges: implied_team_total in [0, 60]; spread unbounded but sane NFL range.
    "preseason_implied_team_total",
    "preseason_spread",
    "season_avg_implied_team_total",
    "season_avg_spread",
)

# Real WR target stats — needed in the joined frame because _run_one_study
# selects y by stat.value.
_WR_TARGET_COLUMNS: tuple[str, ...] = (
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "rushing_yards",
    "rushing_tds",
    "fumbles_lost",
)


def _build_synthetic_joined(seed: int = 42, n_per_season: int = 80) -> pd.DataFrame:
    """Synthetic feature + target frame covering 2018-2023 for one position.

    Shape: (gsis_id, season, week, team, opponent) + the real WR feature
    columns (so run_studies' feat_cols lookup succeeds) + the WR target
    stats. Sufficient for the Optuna trial loop.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for season in range(2018, 2024):  # 2018..2023 inclusive
        for week in range(1, 18):
            for p in range(n_per_season):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": np.int64(season),
                        "week": np.int64(week),
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)

    # Throwaway feature columns kept for the explicit feat_cols tests below;
    # not used by run_studies (which derives feat_cols from the model config).
    df["feat_a"] = rng.normal(0.0, 1.0, size=len(df))
    df["feat_b"] = rng.normal(0.0, 1.0, size=len(df))
    df["feat_c"] = rng.normal(0.0, 1.0, size=len(df))

    # Real WR feature columns — populated with random noise. The trial loop
    # only cares about column presence and dtype; the schema is bypassed
    # because tests monkeypatch _load_join_for_position. Trajectory cols
    # need bounded ranges (age in [15,50]; is_rookie in [0,1];
    # snap_pct_change in [-1,1]) so any downstream sanity check that does
    # peek at values still sees realistic data.
    bounded_trajectory_cols = {
        "age",
        "is_rookie",
        "volume_trend_l4_minus_prior_l4",
        "snap_pct_change_l4_vs_prior_l4",
    }
    # Weather cols need bounded / categorical values (is_high_wind,
    # is_grass_surface are 0/1 indicators; wind_speed_mph is non-negative;
    # temperature_f is realistic NFL range). The schema is bypassed here, but
    # downstream LightGBM still benefits from sane numeric ranges.
    bounded_weather_cols = {
        "wind_speed_mph",
        "is_high_wind",
        "temperature_f",
        "is_grass_surface",
    }
    # Vegas cols: implied totals must stay in [0, 60] (WrFeaturesSchema ge=0
    # le=60); spread is unbounded but realistic NFL range is [-28, 28].
    bounded_vegas_cols = {
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    }
    bounded_cols = bounded_trajectory_cols | bounded_weather_cols | bounded_vegas_cols
    for col in _WR_FEAT_COLUMNS:
        if col in bounded_cols:
            continue
        df[col] = rng.normal(0.0, 1.0, size=len(df))
    df["age"] = rng.uniform(21.0, 35.0, size=len(df))
    df["is_rookie"] = rng.integers(0, 2, size=len(df)).astype(np.float64)
    df["volume_trend_l4_minus_prior_l4"] = rng.normal(0.0, 1.0, size=len(df))
    df["snap_pct_change_l4_vs_prior_l4"] = rng.uniform(-1.0, 1.0, size=len(df))
    df["wind_speed_mph"] = rng.uniform(0.0, 30.0, size=len(df))
    df["is_high_wind"] = rng.integers(0, 2, size=len(df)).astype(np.float64)
    df["temperature_f"] = rng.uniform(20.0, 90.0, size=len(df))
    df["is_grass_surface"] = rng.integers(0, 2, size=len(df)).astype(np.float64)
    df["preseason_implied_team_total"] = rng.uniform(15.0, 35.0, size=len(df))
    df["preseason_spread"] = rng.uniform(-14.0, 14.0, size=len(df))
    df["season_avg_implied_team_total"] = rng.uniform(15.0, 35.0, size=len(df))
    df["season_avg_spread"] = rng.uniform(-14.0, 14.0, size=len(df))

    # Synthetic targets with mild signal so trials produce non-degenerate
    # pinball losses (not all-zero predictions). receiving_yards uses
    # feat_a/feat_b so the explicit-feat_cols tests have signal too.
    df["receiving_yards"] = (
        20.0 + 5.0 * df["feat_a"] + 3.0 * df["feat_b"] + rng.normal(0, 10, size=len(df))
    )
    for col in _WR_TARGET_COLUMNS:
        if col == "receiving_yards":
            continue
        df[col] = rng.normal(0.0, 1.0, size=len(df))
    return df


def test_sample_params_covers_all_axes(tune_module: Any) -> None:
    """_sample_params must call suggest_* for every search-space axis."""
    captured: list[tuple[str, str]] = []

    class RecordingTrial:
        def suggest_float(self, name: str, lo: float, hi: float, *, log: bool = False) -> float:
            captured.append((name, "float-log" if log else "float"))
            return (lo + hi) / 2

        def suggest_int(self, name: str, lo: int, hi: int) -> int:
            captured.append((name, "int"))
            return (lo + hi) // 2

    sampled = tune_module._sample_params(RecordingTrial())
    assert set(sampled.keys()) == {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
    captured_names = {n for n, _ in captured}
    assert captured_names == set(sampled.keys())
    log_axes = {n for n, kind in captured if kind == "float-log"}
    assert log_axes == {"learning_rate", "reg_alpha", "reg_lambda"}


def test_pinball_loss_zero_when_perfect(tune_module: Any) -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert tune_module._pinball_loss(y, y, 0.5) == pytest.approx(0.0)


def test_run_one_study_returns_best_params(tune_module: Any) -> None:
    """Run one tiny study end-to-end on synthetic data."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    feat_cols = ["feat_a", "feat_b", "feat_c"]

    best = tune_module._run_one_study(
        Position.WR,
        Stat.RECEIVING_YARDS,
        joined=joined,
        feat_cols=feat_cols,
        n_trials=3,
        seed=42,
        studies_db=None,  # in-memory storage
    )
    assert set(best.keys()) == {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }


def test_run_one_study_determinism_in_memory(tune_module: Any) -> None:
    """Two runs with the same seed against the same synthetic data and a
    fresh in-memory study yield identical best_params."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    feat_cols = ["feat_a", "feat_b", "feat_c"]

    best_a = tune_module._run_one_study(
        Position.WR,
        Stat.RECEIVING_YARDS,
        joined=joined,
        feat_cols=feat_cols,
        n_trials=3,
        seed=42,
        studies_db=None,
    )
    best_b = tune_module._run_one_study(
        Position.WR,
        Stat.RECEIVING_YARDS,
        joined=joined,
        feat_cols=feat_cols,
        n_trials=3,
        seed=42,
        studies_db=None,
    )
    assert best_a == best_b


def test_run_studies_writes_dense_tuned_dict(
    tune_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_studies returns a dict keyed by (position, stat) with full axes."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    monkeypatch.setattr(tune_module, "_load_join_for_position", lambda position, **kwargs: joined)

    out = tune_module.run_studies(
        positions=[Position.WR],
        stats_per_position={Position.WR: [Stat.RECEIVING_YARDS]},
        n_trials=3,
        seed=42,
        data_root=Path("data"),
        features_root=Path("data/features"),
        studies_db=None,
    )
    assert "wr" in out
    assert "receiving_yards" in out["wr"]
    assert set(out["wr"]["receiving_yards"].keys()) == {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }


def test_run_studies_resume_from_sqlite(
    tune_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming with the same study DB does not duplicate trials."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    monkeypatch.setattr(tune_module, "_load_join_for_position", lambda position, **kwargs: joined)
    db = tmp_path / "studies.db"

    tune_module.run_studies(
        positions=[Position.WR],
        stats_per_position={Position.WR: [Stat.RECEIVING_YARDS]},
        n_trials=2,
        seed=42,
        data_root=Path("data"),
        features_root=Path("data/features"),
        studies_db=db,
    )
    tune_module.run_studies(
        positions=[Position.WR],
        stats_per_position={Position.WR: [Stat.RECEIVING_YARDS]},
        n_trials=2,
        seed=42,
        data_root=Path("data"),
        features_root=Path("data/features"),
        studies_db=db,
    )

    import optuna

    storage_url = f"sqlite:///{db}"
    study = optuna.load_study(study_name="lightgbm:wr:receiving_yards:v1", storage=storage_url)
    # Resume preserves prior trials and adds new ones rather than restarting:
    # 2 from first call + 2 from second call = 4 total.
    assert len(study.trials) == 4


def test_main_dry_run(
    tune_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main(['--dry-run', ...])` prints the tuned dict and does not write --out."""
    joined = _build_synthetic_joined()
    monkeypatch.setattr(tune_module, "_load_join_for_position", lambda position, **kwargs: joined)
    out_path = tmp_path / "lightgbm.json"
    rc = tune_module.main(
        [
            "--position",
            "wr",
            "--stat",
            "receiving_yards",
            "--trials",
            "3",
            "--seed",
            "42",
            "--out",
            str(out_path),
            "--in-memory-storage",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not out_path.exists()
    captured = capsys.readouterr().out
    assert "--dry-run" in captured
