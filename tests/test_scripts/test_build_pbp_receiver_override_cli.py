"""scripts/build_pbp_receiver_override.py — argparse + main smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scripts.build_pbp_receiver_override import (
    _build_receiver_index,
    _parse_season_range,
    main,
)


def test_parse_season_range_dash() -> None:
    assert _parse_season_range("2018-2024") == range(2018, 2025)


def test_parse_season_range_single() -> None:
    assert _parse_season_range("2024") == range(2024, 2025)


def test_build_receiver_index_filters_to_wr_te() -> None:
    """Index includes WR + TE rows only; QB / RB filtered out; deduped on
    (gsis_id, season, week)."""
    depth_charts = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": 1, "position": "WR"},
            {"gsis_id": "00-0000001", "season": 2024, "week": 1, "position": "WR"},  # dup
            {"gsis_id": "00-0000002", "season": 2024, "week": 1, "position": "TE"},
            {"gsis_id": "00-0000003", "season": 2024, "week": 1, "position": "QB"},  # filtered
            {"gsis_id": "00-0000004", "season": 2024, "week": 1, "position": "RB"},  # filtered
        ]
    )
    idx = _build_receiver_index(depth_charts, range(2024, 2025))
    assert set(idx["gsis_id"]) == {"00-0000001", "00-0000002"}
    assert len(idx) == 2  # dedupe of player A's duplicate row
    assert list(idx.columns) == ["gsis_id", "season", "week"]


def test_main_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    """If output exists and --force is not passed, main exits with code 2."""
    output = tmp_path / "pbp_receiver.parquet"
    output.write_text("placeholder")  # exists
    with pytest.raises(SystemExit) as excinfo:
        main(["--output", str(output), "--data-root", str(tmp_path)])
    assert excinfo.value.code == 2  # argparse.error -> exit 2
