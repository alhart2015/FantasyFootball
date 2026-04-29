"""Plan 8 — per-position routing field on _PositionDispatch."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from projections.models import (
    POSITION_DISPATCH,
    BaselineModel,
    EnsembleModel,
    LightGBMNbModel,
    production_model_for,
)
from projections.models.base import Model
from projections.schemas import Position


def test_every_dispatch_entry_has_default_model_class() -> None:
    for pos, dispatch in POSITION_DISPATCH.items():
        assert hasattr(dispatch, "default_model_class"), (
            f"{pos} dispatch missing default_model_class"
        )
        assert dispatch.default_model_class in dispatch.factories, (
            f"{pos} default_model_class={dispatch.default_model_class!r} "
            f"not present in factories keys {sorted(dispatch.factories)}"
        )


def test_default_model_class_per_position() -> None:
    """Defaults reflect Plan 8 Phase 4 re-evaluation verdicts.

    See reports/adoption_gate_summary.md for the full report. Per-position
    routing per spec §6 mechanical tie-break (most-negative rmse_delta.point):

    - QB: lightgbm-nb (3 ADOPTers; NB wins tie-break)
    - RB: baseline (no ADOPT verdict)
    - TE: baseline (no ADOPT verdict)
    - WR: ensemble (sole ADOPTer)
    """
    expected = {
        Position.QB: "lightgbm-nb",
        Position.RB: "baseline",
        Position.TE: "baseline",
        Position.WR: "ensemble",
    }
    for pos, want in expected.items():
        got = POSITION_DISPATCH[pos].default_model_class
        assert got == want, f"{pos} default_model_class={got!r} expected {want!r}"


def test_post_init_raises_when_default_not_in_factories() -> None:
    """The factory dict's value is irrelevant for post-init validation
    (only the keys are inspected). Use the real qb_baseline factory so
    the dict typechecks; the factory itself is never invoked here.
    """
    from projections.models import _PositionDispatch, qb_baseline

    factories: dict[str, Callable[[], Model]] = {"baseline": qb_baseline}
    with pytest.raises(ValueError, match=r"default_model_class.*not in factories"):
        _PositionDispatch(
            factories=factories,
            feature_builder=lambda: None,
            feature_schema=None,  # type: ignore[arg-type]
            ngs_stat_type="passing",
            default_model_class="lightgbm",  # not in factories
        )


def test_production_model_for_returns_expected_class_per_position() -> None:
    """`production_model_for` instantiates the per-position default per Plan 8
    Phase 4 verdicts (see reports/adoption_gate_summary.md)."""
    expected: dict[Position, type[Model]] = {
        Position.QB: LightGBMNbModel,
        Position.RB: BaselineModel,
        Position.TE: BaselineModel,
        Position.WR: EnsembleModel,
    }
    for pos, want_cls in expected.items():
        model = production_model_for(pos)
        assert isinstance(model, want_cls), (
            f"{pos} production model should be a {want_cls.__name__}, got {type(model).__name__}"
        )


def test_production_model_for_respects_default_model_class_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If we patch QB's default to 'ensemble', production_model_for(QB) returns
    an EnsembleModel instance."""
    from projections.models import _PositionDispatch
    from projections.models.ensemble import EnsembleModel

    qb_dispatch = POSITION_DISPATCH[Position.QB]
    patched = _PositionDispatch(
        factories=qb_dispatch.factories,
        feature_builder=qb_dispatch.feature_builder,
        feature_schema=qb_dispatch.feature_schema,
        ngs_stat_type=qb_dispatch.ngs_stat_type,
        default_model_class="ensemble",
    )
    monkeypatch.setitem(POSITION_DISPATCH, Position.QB, patched)
    model = production_model_for(Position.QB)
    assert isinstance(model, EnsembleModel)
