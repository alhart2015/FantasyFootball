"""Parity test: actual_ppr_total in scripts/_actuals_helper.py reproduces the
inline helper that previously lived in scripts/compare_predictions_to_actuals.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from projections.schemas import Ruleset

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_actuals_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_actuals_helper", _REPO_ROOT / "scripts" / "_actuals_helper.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_actuals_helper"] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_weekly_stats() -> pd.DataFrame:
    """Two players x two weeks, exercising both passing + rushing + receiving stat sums."""
    return pd.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000001", "00-0000002", "00-0000002"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "position": ["QB", "QB", "WR", "WR"],
            "passing_yards": [300.0, 250.0, 0.0, 0.0],
            "passing_tds": [2, 1, 0, 0],
            "interceptions": [1, 0, 0, 0],
            "rushing_yards": [30.0, 15.0, 5.0, 0.0],
            "rushing_tds": [0, 0, 0, 0],
            "receptions": [0, 0, 7, 5],
            "receiving_yards": [0.0, 0.0, 90.0, 60.0],
            "receiving_tds": [0, 0, 1, 0],
            "fumbles_lost": [0, 0, 0, 0],
        }
    )


def test_actual_ppr_total_groups_by_gsis_id_position() -> None:
    helper = _import_actuals_helper()
    actual_ppr_total = helper.actual_ppr_total
    out = actual_ppr_total(_synthetic_weekly_stats(), Ruleset.espn_ppr())
    assert set(out.columns) >= {"gsis_id", "position", "actual_total", "actual_n_weeks"}
    assert len(out) == 2
    qb_row = out[out["gsis_id"] == "00-0000001"].iloc[0]
    wr_row = out[out["gsis_id"] == "00-0000002"].iloc[0]
    assert qb_row["actual_n_weeks"] == 2
    assert wr_row["actual_n_weeks"] == 2
    # QB: 550 pass yd / 25 = 22, 3 pass td * 4 = 12, -2 int, 45 rush yd / 10 = 4.5 -> 36.5
    assert qb_row["actual_total"] == pytest.approx(36.5)
    # WR: 12 rec * 1 PPR = 12, 150 rec yd / 10 = 15, 1 rec td * 6 = 6, 5 rush yd / 10 = 0.5 -> 33.5
    assert wr_row["actual_total"] == pytest.approx(33.5)
