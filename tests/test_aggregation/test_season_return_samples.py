"""Tests for aggregate_to_season(return_samples=True)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.aggregation import aggregate_to_season
from projections.schemas import Ruleset
from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

_RULESET = Ruleset.espn_ppr()


def test_return_samples_false_default_returns_dataframe() -> None:
    """Backward-compat: default behavior is unchanged."""
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    out = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1


def test_return_samples_true_returns_tuple() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    out = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200, return_samples=True)
    assert isinstance(out, tuple)
    assert len(out) == 2
    summary, samples = out
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(samples, dict)


def test_return_samples_summary_matches_no_samples_call() -> None:
    """Same input -> same summary frame whether samples are returned or not (determinism).

    ``generated_at`` is a wall-clock timestamp captured at call time and so
    differs between the two calls; every other column is a deterministic
    function of the input (sample seed derives from ids + ruleset).
    """
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    summary_only = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200)
    summary_pair, _ = aggregate_to_season(
        weekly, ruleset=_RULESET, n_samples=200, return_samples=True
    )
    compare_cols = [c for c in summary_only.columns if c != "generated_at"]
    pd.testing.assert_frame_equal(summary_only[compare_cols], summary_pair[compare_cols])


def test_return_samples_dict_keys_match_summary_rows() -> None:
    weekly = _to_weekly_frame(
        [_build_weekly_row(gsis_id="00-0000001", week=w) for w in range(1, 3)]
        + [_build_weekly_row(gsis_id="00-0000002", week=w) for w in range(1, 3)]
    )
    summary, samples = aggregate_to_season(
        weekly, ruleset=_RULESET, n_samples=200, return_samples=True
    )
    expected_keys = {(row["gsis_id"], int(row["season"])) for _, row in summary.iterrows()}
    assert set(samples.keys()) == expected_keys


def test_return_samples_arrays_have_expected_shape_and_mean() -> None:
    weekly = _to_weekly_frame([_build_weekly_row(week=w) for w in range(1, 4)])
    summary, samples = aggregate_to_season(
        weekly, ruleset=_RULESET, n_samples=500, return_samples=True
    )
    assert len(samples) == 1
    key = next(iter(samples.keys()))
    arr = samples[key]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (500,)
    row = summary.iloc[0]
    assert arr.mean() == pytest.approx(float(row["season_mean"]), rel=1e-6)
    assert float(np.quantile(arr, 0.9)) == pytest.approx(float(row["season_p90"]), rel=1e-6)
