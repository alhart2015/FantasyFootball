from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projections.draft.assistant.performance_variance import VarianceParams, sample_weekly_points


def test_params_round_trip_and_lookup(tmp_path: Path) -> None:
    blob = {
        "weekly_std_affine": {
            "QB": {"a": 0.20, "b": 5.0},
            "WR": {"a": 0.30, "b": 3.0},
            "default": {"a": 0.25, "b": 4.4},
        },
        "mean_mult_log_sd": {
            "QB|veteran": 0.30,
            "QB|rookie": 0.45,
            "default|veteran": 0.39,
            "default|rookie": 0.58,
        },
    }
    p = tmp_path / "params.json"
    p.write_text(json.dumps(blob))
    vp = VarianceParams.load(p)
    assert vp.weekly_std("QB", 10.0) == 0.20 * 10.0 + 5.0
    assert vp.log_sd("QB", is_rookie=True) == 0.45
    # unknown position falls back to the 'default' cell
    assert vp.log_sd("RB", is_rookie=False) == 0.39
    assert (
        vp.weekly_std("RB", 10.0) == 0.25 * 10.0 + 4.4
    )  # default affine used for unknown position


def _params() -> VarianceParams:
    return VarianceParams(
        weekly_std_affine={"default": {"a": 0.25, "b": 4.4}, "WR": {"a": 0.30, "b": 3.0}},
        mean_mult_log_sd={
            "default|veteran": 0.39,
            "default|rookie": 0.58,
            "WR|veteran": 0.35,
            "WR|rookie": 0.55,
        },
    )


def test_sampler_shape_nonneg_deterministic() -> None:
    vp = _params()
    pos = np.array(["WR", "RB"])
    means = np.array([200.0, 150.0])
    rook = np.array([False, True])
    a = sample_weekly_points(
        vp, pos, means, rook, n_sims=50, n_weeks=14, rng=np.random.default_rng(0)
    )
    b = sample_weekly_points(
        vp, pos, means, rook, n_sims=50, n_weeks=14, rng=np.random.default_rng(0)
    )
    assert a.shape == (50, 14, 2)
    assert (a >= 0).all()
    assert np.array_equal(a, b)


def test_sampler_recovers_mean_and_zero_floor() -> None:
    vp = _params()
    pos = np.array(["WR", "QB"])
    means = np.array([170.0, 0.0])
    rook = np.array([False, False])
    s = sample_weekly_points(
        vp, pos, means, rook, n_sims=4000, n_weeks=17, rng=np.random.default_rng(1)
    )
    assert abs(s[:, :, 0].mean() - 170.0 / 17) < 0.3
    assert (s[:, :, 1] == 0).all()


def test_rookie_wider_than_veteran() -> None:
    vp = _params()
    pos = np.array(["WR", "WR"])
    means = np.array([170.0, 170.0])
    rook = np.array([False, True])
    s = sample_weekly_points(
        vp, pos, means, rook, n_sims=6000, n_weeks=17, rng=np.random.default_rng(2)
    )
    season = s.sum(axis=1)
    assert season[:, 1].std() > season[:, 0].std()
