"""Tests for the wr_ensemble_decomposed factory.

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md.

Task 1 covers factory wiring + registry. Fit/predict tests (Task 2) live in
the same file and share the synthetic-data helper below.
"""

from __future__ import annotations

from projections.models import (
    POSITION_DISPATCH,
    DecomposedBaselineModel,
    EnsembleModel,
    LightGBMNbModel,
    wr_ensemble_decomposed,
)
from projections.models.ensemble import _EnsembleConfig
from projections.schemas import Position


def test_wr_ensemble_decomposed_returns_ensemble_with_decomposed_child_a() -> None:
    """Factory returns an EnsembleModel whose child A factory yields a
    DecomposedBaselineModel and child B factory yields an LightGBMNbModel.
    """
    model = wr_ensemble_decomposed()
    assert isinstance(model, EnsembleModel)
    assert model.position == Position.WR

    # Children are constructed by the factories on demand; instantiate to
    # verify type (not the same as the lazily-fit children inside fit()).
    config: _EnsembleConfig = model._config
    child_a = config.child_a_factory()
    child_b = config.child_b_factory()
    assert isinstance(child_a, DecomposedBaselineModel), (
        f"child A should be DecomposedBaselineModel, got {type(child_a).__name__}"
    )
    assert isinstance(child_b, LightGBMNbModel), (
        f"child B should be LightGBMNbModel, got {type(child_b).__name__}"
    )


def test_wr_ensemble_decomposed_registered_in_factories() -> None:
    """_WR_FACTORIES has an 'ensemble-decomposed' entry resolving to a freshly
    instantiated EnsembleModel with the decomposed child A wiring.
    """
    wr_dispatch = POSITION_DISPATCH[Position.WR]
    assert "ensemble-decomposed" in wr_dispatch.factories, (
        f"_WR_FACTORIES missing 'ensemble-decomposed'; available: {sorted(wr_dispatch.factories)}"
    )
    model = wr_dispatch.factories["ensemble-decomposed"]()
    assert isinstance(model, EnsembleModel)
    child_a = model._config.child_a_factory()
    assert isinstance(child_a, DecomposedBaselineModel)


def test_default_model_class_unchanged_after_registration() -> None:
    """Registering 'ensemble-decomposed' does NOT flip production routing.
    The flip is the §1.3.5 ADOPT outcome (Task 5).
    """
    assert POSITION_DISPATCH[Position.WR].default_model_class == "ensemble"


def test_wr_ensemble_decomposed_in_models_all() -> None:
    """wr_ensemble_decomposed is exported via projections.models.__all__."""
    import projections.models as models_pkg

    assert "wr_ensemble_decomposed" in models_pkg.__all__, (
        f"wr_ensemble_decomposed missing from __all__; got {sorted(models_pkg.__all__)}"
    )
