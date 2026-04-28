"""Tests for the public NB-2 dispersion helper relocated to parametric.py
(Plan 5c Phase 0). The function body is unchanged from baseline.py's prior
private implementation; these tests pin the behavior at the new location.
"""

from __future__ import annotations

import numpy as np

from projections.distributions.parametric import (
    _NB_DISPERSION_CLIP,
    _NB_MU_FLOOR,
    nb_dispersion_from_residuals,
)


def test_returns_clip_endpoint_on_empty_input() -> None:
    """Fewer than 2 rows -> degenerate; helper returns clip top to keep downstream NB defined."""
    out = nb_dispersion_from_residuals(mu_hat=np.array([0.5]), actual=np.array([0]))
    assert out == _NB_DISPERSION_CLIP[1]


def test_returns_clip_lower_bound_on_all_zero_actuals() -> None:
    """All-zero actuals drive the likelihood toward the lower clip endpoint."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.1, 1.0, size=200)
    out = nb_dispersion_from_residuals(mu_hat=mu, actual=np.zeros(200, dtype=np.int64))
    assert out == _NB_DISPERSION_CLIP[0]


def test_recovers_dispersion_within_30pct_on_known_nb_data() -> None:
    """On samples drawn from a known NB-2 with mean=mu and dispersion=5,
    the MLE should land within ~30% of the truth on a reasonable-size sample."""
    rng = np.random.default_rng(42)
    n = 5000
    mu = rng.uniform(0.5, 2.0, size=n)
    true_dispersion = 5.0
    # Sample NB-2 row-wise: var = mu + mu^2 / dispersion.
    p = true_dispersion / (true_dispersion + mu)
    actual = rng.negative_binomial(n=true_dispersion, p=p, size=n)

    fitted = nb_dispersion_from_residuals(mu_hat=mu, actual=actual.astype(np.float64))

    assert _NB_DISPERSION_CLIP[0] < fitted < _NB_DISPERSION_CLIP[1]
    assert abs(fitted - true_dispersion) / true_dispersion < 0.3


def test_clips_negative_actuals_to_zero() -> None:
    """The helper rounds + clips actuals to non-negative integers before
    fitting NB. Pass a negative value and confirm the fit still returns
    a value in the clip range (i.e., the negative was treated as 0)."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 1.0, size=100)
    actual = np.array([-1.0, -0.5, 0.0, 1.0, 2.0] * 20)
    out = nb_dispersion_from_residuals(mu_hat=mu, actual=actual)
    assert _NB_DISPERSION_CLIP[0] <= out <= _NB_DISPERSION_CLIP[1]


def test_constants_exposed_for_baseline_import() -> None:
    """baseline.py imports both constants; pin their existence + types."""
    assert isinstance(_NB_DISPERSION_CLIP, tuple)
    assert len(_NB_DISPERSION_CLIP) == 2
    assert _NB_DISPERSION_CLIP[0] < _NB_DISPERSION_CLIP[1]
    assert isinstance(_NB_MU_FLOOR, float)
    assert _NB_MU_FLOOR > 0
