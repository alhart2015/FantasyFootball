from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.draft.assistant.rookies import attach_is_rookie


def test_attach_is_rookie(tmp_path: Path) -> None:
    # No prior-season partitions exist -> every player is treated as a rookie.
    pool = pd.DataFrame({"gsis_id": ["00-0000001", "00-0000002"], "position": ["WR", "RB"]})
    out = attach_is_rookie(pool, season=2026, data_root=tmp_path)
    assert out["is_rookie"].all()


def test_attach_is_rookie_does_not_mutate_input(tmp_path: Path) -> None:
    pool = pd.DataFrame({"gsis_id": ["00-0000003"], "position": ["TE"]})
    attach_is_rookie(pool, season=2026, data_root=tmp_path)
    assert "is_rookie" not in pool.columns  # returns a new frame
