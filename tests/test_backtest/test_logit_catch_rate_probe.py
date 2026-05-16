"""Tests for the logit catch_rate probe.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md.
"""

from __future__ import annotations

import numpy as np

from projections.backtest.logit_catch_rate_probe import _expand_to_trials


def test_expand_to_trials_basic_shape_and_labels() -> None:
    """3 rows with (T, S) = (4, 3), (2, 0), (5, 5). Expansion yields 11 trial
    rows: 3+1+0+2+5+0 = 7 successes and 1+2+0 = 3 failures, sharing X.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    successes = np.array([3, 0, 5], dtype=np.int64)
    trials = np.array([4, 2, 5], dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (11, 2)
    assert y_trials.shape == (11,)
    assert np.allclose(x_trials[:4], np.tile([1.0, 2.0], (4, 1)))
    assert np.array_equal(y_trials[:4], np.array([1, 1, 1, 0]))
    assert np.allclose(x_trials[4:6], np.tile([3.0, 4.0], (2, 1)))
    assert np.array_equal(y_trials[4:6], np.array([0, 0]))
    assert np.allclose(x_trials[6:11], np.tile([5.0, 6.0], (5, 1)))
    assert np.array_equal(y_trials[6:11], np.array([1, 1, 1, 1, 1]))


def test_expand_to_trials_zero_trials_dropped() -> None:
    """A row with T=0 must be dropped from the expansion entirely
    (rather than panicking on shape mismatch in the per-row alloc).
    """
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    successes = np.array([2, 0, 1], dtype=np.int64)
    trials = np.array([2, 0, 3], dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (5, 1)
    assert y_trials.shape == (5,)
    assert np.allclose(x_trials[:2], np.array([[1.0], [1.0]]))
    assert np.array_equal(y_trials[:2], np.array([1, 1]))
    assert np.allclose(x_trials[2:], np.array([[3.0], [3.0], [3.0]]))
    assert np.array_equal(y_trials[2:], np.array([1, 0, 0]))


def test_expand_to_trials_validates_successes_le_trials() -> None:
    """successes[i] > trials[i] is a bug in the caller. Raise ValueError
    rather than producing a corrupt expansion.
    """
    import pytest

    x = np.array([[1.0]], dtype=np.float64)
    successes = np.array([5], dtype=np.int64)
    trials = np.array([3], dtype=np.int64)

    with pytest.raises(ValueError, match=r"successes\[0\]=5 > trials\[0\]=3"):
        _expand_to_trials(x, successes, trials)


def test_expand_to_trials_handles_empty_input() -> None:
    """Empty (X, successes, trials) returns empty arrays of the right shape."""
    x = np.empty((0, 3), dtype=np.float64)
    successes = np.empty((0,), dtype=np.int64)
    trials = np.empty((0,), dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (0, 3)
    assert y_trials.shape == (0,)
