"""Tests for the shared is_rookie attachment helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.draft.assistant.rookies import attach_is_rookie
from projections.schemas import _PYARROW_STR
from projections.store import write_partition


def test_attach_is_rookie_flags_players_without_prior_appearance(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    ws = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [1],
            "position": pd.array(["RB"], dtype=_PYARROW_STR),
        }
    )
    write_partition(raw, "weekly_stats", ws, season=2024)
    pool = pd.DataFrame({"gsis_id": pd.array(["00-0000001", "99-8467088"], dtype=_PYARROW_STR)})
    out = attach_is_rookie(pool, season=2026, data_root=tmp_path)
    rookie = dict(zip(out["gsis_id"].astype(str), out["is_rookie"], strict=True))
    assert not rookie["00-0000001"]  # seen in 2024 -> veteran
    assert rookie["99-8467088"]  # never seen -> rookie
