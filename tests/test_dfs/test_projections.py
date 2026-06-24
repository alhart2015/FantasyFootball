"""Assembly test for the walk-forward weekly projection emitter.

The real model fit/predict path is heavy on synthetic features, so we stub
`_emit_one_cell` and assert that `emit_weekly_projections` scores DK-base
points correctly from canned per-stat means."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.dfs import projections as proj
from projections.schemas import Position, Ruleset, Stat


def test_emit_assembles_points_from_stat_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the per-(position, year) predict step: return canned per-stat means.
    def fake_one_cell(
        position: Position,
        year: int,
        *,
        train_start: int,
        model_class: str | None,
        features_root: Path,
        raw_root: Path,
        ruleset: Ruleset,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gsis_id": ["00-0036900"],
                "season": [year],
                "week": [5],
                "position": [position.value],
                Stat.RECEPTIONS.value: [6.0],
                Stat.RECEIVING_YARDS.value: [78.0],
                Stat.RECEIVING_TDS.value: [0.5],
            }
        )

    monkeypatch.setattr(proj, "_emit_one_cell", fake_one_cell)

    out = proj.emit_weekly_projections(
        seasons=[2023],
        positions=[Position.WR],
        features_root="unused",
        raw_root="unused",
        ruleset=Ruleset.draftkings(),
    )
    row = out.iloc[0]
    # 6*1 + 78/10 + 0.5*6 = 6 + 7.8 + 3 = 16.8
    assert round(float(row["our_pts"]), 2) == 16.8
    assert row["position"] == "WR"
