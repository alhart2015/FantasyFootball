from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.draft.assistant.rookies import attach_is_rookie
from projections.schemas import _PYARROW_STR
from projections.store import write_partition


def test_attach_is_rookie(tmp_path: Path) -> None:
    # 00-0000001 appeared in 2024 -> veteran; 00-0000002 never appeared -> rookie.
    ws = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [1],
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
        }
    )
    write_partition(tmp_path / "raw", "weekly_stats", ws, season=2024)
    pool = pd.DataFrame({"gsis_id": ["00-0000001", "00-0000002"], "position": ["WR", "RB"]})
    out = attach_is_rookie(pool, season=2026, data_root=tmp_path)
    by = dict(zip(out["gsis_id"].astype(str), out["is_rookie"], strict=True))
    assert bool(by["00-0000001"]) is False  # appeared before -> veteran
    assert bool(by["00-0000002"]) is True  # never appeared -> rookie


def test_attach_is_rookie_does_not_mutate_input(tmp_path: Path) -> None:
    pool = pd.DataFrame({"gsis_id": ["00-0000003"], "position": ["TE"]})
    attach_is_rookie(pool, season=2026, data_root=tmp_path)
    assert "is_rookie" not in pool.columns  # returns a new frame
