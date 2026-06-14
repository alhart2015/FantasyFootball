from __future__ import annotations

import numpy as np
from fit_performance_variance import fit_params


def test_fit_recovers_affine_and_logsd() -> None:
    rng = np.random.default_rng(0)
    # Synthetic WR veteran player-seasons with weekly std = 0.3*pg + 3.
    rows = []
    for i in range(60):
        pg = rng.uniform(6, 18)
        weekly = rng.normal(pg, 0.3 * pg + 3.0, size=17).clip(min=0)
        rows.append(
            {
                "gsis_id": f"00-{i:07d}",
                "position": "WR",
                "season": 2022,
                "weekly": weekly,
                "projected_pg": pg,
                "is_rookie": False,
            }
        )
    params = fit_params(rows)
    wr = params["weekly_std_affine"]["WR"]
    assert abs(wr["a"] - 0.3) < 0.12 and abs(wr["b"] - 3.0) < 1.5
    assert "default|veteran" in params["mean_mult_log_sd"]
    assert "default|rookie" in params["mean_mult_log_sd"]
    assert "default" in params["weekly_std_affine"]


def test_fit_affine_only_rows_without_projection() -> None:
    # Rows with projected_pg=0 (pre-2021, no snapshot) contribute to the affine, not the log-SD.
    rng = np.random.default_rng(1)
    rows = [
        {
            "gsis_id": f"00-{i:07d}",
            "position": "RB",
            "season": 2019,
            "weekly": rng.normal(10.0, 5.0, size=17).clip(min=0),
            "projected_pg": 0.0,
            "is_rookie": False,
        }
        for i in range(30)
    ]
    params = fit_params(rows)
    assert "RB" in params["weekly_std_affine"]
    # no projection rows -> default log-SD cells are 0.0 (empty), still present
    assert params["mean_mult_log_sd"]["default|veteran"] == 0.0
