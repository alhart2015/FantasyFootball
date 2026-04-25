"""Tests for the Model Protocol contract."""

from __future__ import annotations

from projections.models import Model


def test_model_protocol_has_required_members() -> None:
    """Model is a structural Protocol — verify the names every implementation
    must provide. Signatures are checked by mypy, not at runtime."""
    expected = {"position", "model_id", "fit", "predict_distribution", "save", "load"}
    actual = {name for name in dir(Model) if not name.startswith("_")}
    assert expected.issubset(actual), f"missing: {expected - actual}"


def test_model_protocol_is_not_runtime_checkable() -> None:
    """Model is a plain Protocol (not @runtime_checkable). isinstance() should
    raise TypeError if anyone tries it. We don't want the Distribution-style
    structural runtime check here."""

    class _Dummy:
        pass

    try:
        isinstance(_Dummy(), Model)  # type: ignore[misc]
    except TypeError:
        return
    raise AssertionError("isinstance(_, Model) should have raised TypeError")
