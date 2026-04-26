"""Aggregate weekly per-player projections into season-total distributions."""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset


def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
) -> pd.DataFrame:
    """Stub — see Task 4.2 for the real implementation."""
    raise NotImplementedError("Implemented in Task 4.2")
