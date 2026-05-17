"""Tests for src/projections/preseason/model.py."""

from __future__ import annotations

from projections.preseason.model import NaivePreseasonModel, PreseasonModel


def test_naive_preseason_model_implements_protocol() -> None:
    """NaivePreseasonModel should satisfy the PreseasonModel Protocol at runtime."""
    m = NaivePreseasonModel()
    assert isinstance(m, PreseasonModel)
    assert m.model_id == "naive-preseason-v1"
