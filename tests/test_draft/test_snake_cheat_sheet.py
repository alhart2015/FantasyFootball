"""Tests for `projections.draft.snake_cheat_sheet`."""

from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: F401  # used in later tasks per plan
import pytest  # noqa: F401  # used in later tasks per plan

from projections.draft.snake_cheat_sheet import _assign_tiers


def test_assign_tiers_gap_based_correctness() -> None:
    """§5.1 #8 — synthetic gaps produce the documented tier partition."""
    vorps = np.array([100.0, 99.0, 98.0, 50.0, 49.0, 48.0, 10.0, 9.0, 8.0])
    tiers = _assign_tiers(vorps, n_tiers=3)
    assert list(tiers) == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_assign_tiers_fallback_when_n_in_pool_less_than_n() -> None:
    """§5.1 #9 — fewer in-pool players than tiers: each gets own tier."""
    vorps = np.array([50.0, 30.0, 20.0, 10.0, 5.0])  # 5 players
    tiers = _assign_tiers(vorps, n_tiers=8)
    assert list(tiers) == [1, 2, 3, 4, 5]


def test_assign_tiers_exact_when_n_in_pool_equals_n() -> None:
    """§5.1 #10 — exactly N in-pool players: 1-per-tier."""
    vorps = np.array([100.0, 80.0, 60.0, 40.0])
    tiers = _assign_tiers(vorps, n_tiers=4)
    assert list(tiers) == [1, 2, 3, 4]


def test_assign_tiers_with_n_equal_one_all_tier_one() -> None:
    """§5.1 #19 — tiers_per_position=1 collapses everyone into tier 1."""
    vorps = np.array([100.0, 50.0, 25.0, 10.0, 1.0])
    tiers = _assign_tiers(vorps, n_tiers=1)
    assert list(tiers) == [1, 1, 1, 1, 1]


def test_assign_tiers_tie_break_prefers_earlier_gap() -> None:
    """§5.1 #21 — when gaps are tied, the earlier (higher-rank) gap wins."""
    # gaps = [1, 4, 1, 4]: two gaps of 4 competing for the single allowed cut
    # under n_tiers=2. Earlier gap (index 1) wins; later gap (index 3) loses.
    vorps = np.array([10.0, 9.0, 5.0, 4.0, 0.0])
    tiers = _assign_tiers(vorps, n_tiers=2)
    assert list(tiers) == [1, 1, 2, 2, 2]


def test_assign_tiers_empty_input() -> None:
    vorps = np.array([], dtype=np.float64)
    tiers = _assign_tiers(vorps, n_tiers=8)
    assert tiers.shape == (0,)
    assert tiers.dtype == np.int64
