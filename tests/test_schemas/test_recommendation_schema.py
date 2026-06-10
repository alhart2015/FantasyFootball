"""Tests for RecommendationSchema."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from projections.schemas import _PYARROW_STR, RecommendationSchema


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "vorp": [50.0, 40.0],
            "consensus_adp": pd.array([3.0, pd.NA], dtype=pd.Float64Dtype()),
            "p_available_next": pd.array([0.1, pd.NA], dtype=pd.Float64Dtype()),
            "fills_starting_slot": [True, True],
            "score": [12.6, 2.6],
            "rank": pd.array([1, 2], dtype=pd.Int64Dtype()),
        }
    )


def test_valid_frame_passes() -> None:
    out = RecommendationSchema.validate(_valid_frame())
    assert list(out["rank"]) == [1, 2]


def test_missing_column_fails() -> None:
    bad = _valid_frame().drop(columns=["score"])
    with pytest.raises(SchemaError):
        RecommendationSchema.validate(bad)


def test_rank_must_be_unique() -> None:
    bad = _valid_frame()
    bad["rank"] = pd.array([1, 1], dtype=pd.Int64Dtype())
    with pytest.raises(SchemaError):
        RecommendationSchema.validate(bad)
