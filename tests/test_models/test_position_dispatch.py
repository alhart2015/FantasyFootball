"""Plan 8 — per-position routing field on _PositionDispatch."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from projections.models import POSITION_DISPATCH
from projections.models.base import Model


def test_every_dispatch_entry_has_default_model_class() -> None:
    for pos, dispatch in POSITION_DISPATCH.items():
        assert hasattr(dispatch, "default_model_class"), (
            f"{pos} dispatch missing default_model_class"
        )
        assert dispatch.default_model_class in dispatch.factories, (
            f"{pos} default_model_class={dispatch.default_model_class!r} "
            f"not present in factories keys {sorted(dispatch.factories)}"
        )


def test_initial_default_is_baseline_for_every_position() -> None:
    for pos, dispatch in POSITION_DISPATCH.items():
        assert dispatch.default_model_class == "baseline", (
            f"{pos} default_model_class is {dispatch.default_model_class!r}, "
            f"expected 'baseline' (initial Plan 8 state — Phase 4 will update)"
        )


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
