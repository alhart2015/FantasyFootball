"""Tests for reconcile_pool_gsis (placeholder -> real gsis relabeling)."""

from __future__ import annotations

import pandas as pd

from projections.draft.assistant.pool_identity import reconcile_pool_gsis
from projections.schemas import _PYARROW_STR


def _idmap(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["gsis_id", "full_name", "position"]).astype(
        {"gsis_id": _PYARROW_STR, "full_name": _PYARROW_STR, "position": _PYARROW_STR}
    )


def _pool(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["gsis_id", "full_name", "position"]).astype(
        {"gsis_id": _PYARROW_STR, "full_name": _PYARROW_STR, "position": _PYARROW_STR}
    )


def test_unique_match_is_reconciled() -> None:
    id_map = _idmap([("00-0000001", "Christian McCaffrey", "RB")])
    pool = _pool([("99-1234567", "Christian McCaffrey", "RB")])
    out = reconcile_pool_gsis(pool, id_map)
    assert out["gsis_id"].tolist() == ["00-0000001"]
    assert out["gsis_id"].dtype == _PYARROW_STR


def test_already_real_is_unchanged() -> None:
    id_map = _idmap([("00-0000001", "Christian McCaffrey", "RB")])
    pool = _pool([("00-0000009", "Christian McCaffrey", "RB")])
    out = reconcile_pool_gsis(pool, id_map)
    assert out["gsis_id"].tolist() == ["00-0000009"]


def test_ambiguous_key_left_as_placeholder() -> None:
    # Two distinct real gsis share the same name+position key -> ambiguous, skip.
    id_map = _idmap([("00-0000001", "John Smith", "WR"), ("00-0000002", "John Smith", "WR")])
    pool = _pool([("99-1234567", "John Smith", "WR")])
    out = reconcile_pool_gsis(pool, id_map)
    assert out["gsis_id"].tolist() == ["99-1234567"]


def test_collision_with_existing_real_id_is_skipped() -> None:
    # The real id is already present in the pool -> reconciling would duplicate it.
    id_map = _idmap([("00-0000001", "Christian McCaffrey", "RB")])
    pool = _pool(
        [
            ("00-0000001", "Christian McCaffrey", "RB"),
            ("99-1234567", "Christian McCaffrey", "RB"),
        ]
    )
    out = reconcile_pool_gsis(pool, id_map)
    assert out["gsis_id"].tolist() == ["00-0000001", "99-1234567"]
    assert out["gsis_id"].is_unique


def test_no_match_left_as_placeholder() -> None:
    id_map = _idmap([("00-0000001", "Christian McCaffrey", "RB")])
    pool = _pool([("99-7654321", "Unknown Rookie", "WR")])
    out = reconcile_pool_gsis(pool, id_map)
    assert out["gsis_id"].tolist() == ["99-7654321"]


def test_position_mismatch_is_not_reconciled() -> None:
    # Same name, different position -> different key -> no match.
    id_map = _idmap([("00-0000001", "Cordarrelle Patterson", "RB")])
    pool = _pool([("99-1234567", "Cordarrelle Patterson", "WR")])
    out = reconcile_pool_gsis(pool, id_map)
    assert out["gsis_id"].tolist() == ["99-1234567"]


def test_suffix_and_accent_folding_match() -> None:
    # placeholder_name_key folds accents and drops Jr/Sr suffixes.
    id_map = _idmap([("00-0000050", "Michael Pittman Jr.", "WR")])
    pool = _pool([("99-1234567", "Michael Pittman", "WR")])
    out = reconcile_pool_gsis(pool, id_map)
    assert out["gsis_id"].tolist() == ["00-0000050"]


def test_missing_full_name_column_raises() -> None:
    id_map = _idmap([("00-0000001", "Christian McCaffrey", "RB")])
    pool = pd.DataFrame({"gsis_id": ["99-1"], "position": ["RB"]}).astype(
        {"gsis_id": _PYARROW_STR, "position": _PYARROW_STR}
    )
    try:
        reconcile_pool_gsis(pool, id_map)
    except ValueError as exc:
        assert "full_name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing full_name")
