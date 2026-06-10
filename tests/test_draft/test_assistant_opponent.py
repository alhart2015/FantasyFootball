"""Tests for the ADP-bot pick policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import bot_pick
from projections.schemas import _PYARROW_STR


def _available(rows: list[tuple[str, float | None]]) -> pd.DataFrame:
    """rows = [(gsis_id, consensus_adp_or_None), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "consensus_adp": pd.array(
                [r[1] if r[1] is not None else pd.NA for r in rows], dtype=pd.Float64Dtype()
            ),
        }
    )


def test_zero_jitter_picks_lowest_adp() -> None:
    avail = _available([("00-0000001", 10.0), ("00-0000002", 3.0), ("00-0000003", 7.0)])
    rng = np.random.default_rng(0)
    assert bot_pick(avail, rng, adp_jitter=0.0) == "00-0000002"


def test_null_adp_left_for_hero_until_nothing_else() -> None:
    avail = _available([("00-0000001", None), ("00-0000002", 50.0)])
    # Even with big jitter, a finite ADP always beats +inf.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert bot_pick(avail, rng, adp_jitter=10.0) == "00-0000002"


def test_all_null_falls_back_to_gsis_order() -> None:
    avail = _available([("00-0000002", None), ("00-0000001", None)])
    rng = np.random.default_rng(0)
    # All +inf -> deterministic gsis_id tie-break (ascending).
    assert bot_pick(avail, rng, adp_jitter=5.0) == "00-0000001"


def test_deterministic_given_seed() -> None:
    avail = _available([("00-0000001", 5.0), ("00-0000002", 5.5), ("00-0000003", 6.0)])
    a = bot_pick(_available_copy(avail), np.random.default_rng(42), adp_jitter=3.0)
    b = bot_pick(_available_copy(avail), np.random.default_rng(42), adp_jitter=3.0)
    assert a == b


def _available_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()
