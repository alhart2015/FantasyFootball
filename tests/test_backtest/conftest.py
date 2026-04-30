"""Synthetic fixtures for the backtest unit tests.

Plan 3c Phase 2 onward. Fixtures live here (not in the per-file test
modules) so test_metrics.py / test_naive.py / test_snapshot.py /
test_harness.py can share a coherent set of synthetic inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR
from projections.store import write_partition


@pytest.fixture
def fake_eval_df() -> pd.DataFrame:
    """A tiny synthetic eval DataFrame matching the shape produced by
    harness.run_backtest's inner-join of predictions and actuals.

    Three player-weeks for two players. receptions/receiving_yards
    columns are suffixed _pred / _actual. Composite columns: mean,
    p10, p90, actual_ppr. (Other p-quantiles and per-stat columns
    are omitted; tests only consume what they assert against.)
    """
    return pd.DataFrame(
        {
            "gsis_id": ["00-A", "00-A", "00-B"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 1],
            "mean": [12.0, 14.0, 6.0],
            "p10": [4.0, 6.0, 1.0],
            "p90": [22.0, 24.0, 14.0],
            "actual_ppr": [10.0, 18.0, 4.0],
            "receptions_pred": [4.0, 5.0, 2.0],
            "receptions_actual": [3.0, 6.0, 1.0],
            "receiving_yards_pred": [55.0, 70.0, 25.0],
            "receiving_yards_actual": [40.0, 95.0, 12.0],
        }
    )


@pytest.fixture
def synthetic_backtest_layout(
    tmp_path: Path,
    baseline_features_wr: pd.DataFrame,
    baseline_weekly_stats_wr: pd.DataFrame,
) -> dict[str, Path]:
    """Stand up a tiny data/raw + data/features layout under tmp_path so
    run_backtest can be exercised against synthetic data with no network.

    The root-scope WR fixtures cover 2024 (weeks 1-8) + 2025 (weeks 1-4),
    so the integration test trains on 2024 and holds out 2025.

    Returns paths suitable for run_backtest:
        {"raw_root": ..., "features_root": ...}
    """
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    features_root = data_root / "features"

    feats_by_season = baseline_features_wr.groupby("season")
    for season, sf in feats_by_season:
        for week, wf in sf.groupby("week"):
            write_partition(
                features_root,
                "wr",
                wf.reset_index(drop=True),
                season=int(season),
                week=int(week),
            )

    ws_by_season = baseline_weekly_stats_wr.groupby("season")
    for season, sf in ws_by_season:
        write_partition(
            raw_root,
            "weekly_stats",
            sf.reset_index(drop=True),
            season=int(season),
            week=None,
        )

    return {"raw_root": raw_root, "features_root": features_root}


@pytest.fixture
def probe_synthetic_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """500-row deterministic features + weekly_stats for probe testing.

    Construction:
      - 25 synthetic players x 5 seasons x 4 weeks = 500 rows.
      - Baseline features: ``base_x1``, ``base_x2``, ``base_x3`` -- i.i.d. normal.
      - Candidate features:
          - ``cand_signal``  -- orthogonal noise that contributes ``+1.0 * cand_signal`` to
            the synthetic target. Adding this column should produce SIGNAL.
          - ``cand_null``    -- orthogonal noise that does NOT enter the target. Adding
            this column should produce NULL.
          - ``cand_redundant`` -- a copy of ``base_x1`` + small noise. Adding this column
            should produce NULL (Ridge shrinks one of the two correlated columns).
      - Target stat: ``passing_yards`` = ``base_x1 + 0.5*base_x2 + 1.0*cand_signal + eps``
        (eps ~ N(0, 0.5)). Other target stats (``passing_tds``, etc.) are zero -- the
        probe code path doesn't care.
    """
    rng = np.random.default_rng(42)
    n_players, seasons, weeks_per_season = 25, range(2018, 2023), range(1, 5)

    rows: list[dict[str, object]] = []
    for player_idx in range(n_players):
        gsis_id = f"00-003{player_idx:04d}"
        for season in seasons:
            for week in weeks_per_season:
                rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "position": "QB",
                        "team": "KC",
                        "opponent": "BUF",
                    }
                )
    base = pd.DataFrame(rows)
    n = len(base)
    base["base_x1"] = rng.normal(size=n)
    base["base_x2"] = rng.normal(size=n)
    base["base_x3"] = rng.normal(size=n)
    base["cand_signal"] = rng.normal(size=n)
    base["cand_null"] = rng.normal(size=n)
    base["cand_redundant"] = base["base_x1"] + rng.normal(scale=0.05, size=n)

    weekly_stats = base[["gsis_id", "season", "week", "position"]].copy()
    weekly_stats["passing_yards"] = (
        base["base_x1"]
        + 0.5 * base["base_x2"]
        + 1.0 * base["cand_signal"]
        + rng.normal(scale=0.5, size=n)
    )
    weekly_stats["passing_tds"] = 0.0
    weekly_stats["interceptions"] = 0.0
    weekly_stats["rushing_yards"] = 0.0
    weekly_stats["rushing_tds"] = 0.0
    weekly_stats["fumbles_lost"] = 0.0

    for col in ("gsis_id", "team", "opponent", "position"):
        if col in base.columns:
            base[col] = base[col].astype(_PYARROW_STR)
    weekly_stats["gsis_id"] = weekly_stats["gsis_id"].astype(_PYARROW_STR)
    weekly_stats["position"] = weekly_stats["position"].astype(_PYARROW_STR)

    return base, weekly_stats
