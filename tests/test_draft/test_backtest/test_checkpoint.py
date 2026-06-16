"""Tests for the resumable chunk-runner's pure helpers."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from projections.draft.backtest.checkpoint import dump_results, load_results, plan_chunks
from projections.draft.backtest.league import LeagueResult


def test_plan_chunks_covers_range_without_gaps_or_overlap() -> None:
    chunks = plan_chunks(n_seeds=200, chunk_size=20)
    assert chunks[0] == (0, 20)
    assert chunks[-1] == (180, 200)
    # contiguous, non-overlapping, exactly covering [0, 200)
    assert chunks[0][0] == 0 and chunks[-1][1] == 200
    for (_, hi), (lo2, _) in pairwise(chunks):
        assert hi == lo2


def test_plan_chunks_handles_ragged_last_chunk() -> None:
    chunks = plan_chunks(n_seeds=50, chunk_size=20)
    assert chunks == [(0, 20), (20, 40), (40, 50)]


def test_results_round_trip_preserves_values() -> None:
    actual = [LeagueResult(1, "now_or_never", 9, 5, 1400.5, True, False)]
    projected = [LeagueResult(2, "season_value", 10, 4, 1502.25, True, True)]
    restored_a, restored_p = load_results(dump_results(actual, projected))
    assert restored_a == actual
    assert restored_p == projected


def test_verify_or_write_manifest_writes_then_matches(tmp_path: Path) -> None:
    from projections.draft.backtest.checkpoint import verify_or_write_manifest

    key = {"season": 2025, "strategy_a": "season_value_timing", "strategy_b": "season_value"}
    verify_or_write_manifest(tmp_path, key)  # first call writes the manifest
    verify_or_write_manifest(tmp_path, key)  # identical run_key -> no raise


def test_verify_or_write_manifest_rejects_mismatch(tmp_path: Path) -> None:
    import pytest

    from projections.draft.backtest.checkpoint import verify_or_write_manifest

    verify_or_write_manifest(tmp_path, {"season": 2025, "strategy_a": "now_or_never"})
    with pytest.raises(ValueError, match="fresh"):
        verify_or_write_manifest(tmp_path, {"season": 2024, "strategy_a": "now_or_never"})


def test_manifest_rejects_changed_floor(tmp_path: Path) -> None:
    import pytest

    from projections.draft.backtest.checkpoint import verify_or_write_manifest

    key = {
        "season": 2025,
        "strategy_a": "now_or_never_floored",
        "strategy_b": "now_or_never",
        "strategy_n_sims": 200,
        "jitter": 8.0,
        "floor": 40.0,
        "floor_weight": 1.0,
    }
    verify_or_write_manifest(tmp_path, key)  # first write OK
    verify_or_write_manifest(tmp_path, key)  # identical run_key -> no raise
    with pytest.raises(ValueError, match="was built with params"):
        verify_or_write_manifest(tmp_path, {**key, "floor": 60.0})  # changed floor -> reject
