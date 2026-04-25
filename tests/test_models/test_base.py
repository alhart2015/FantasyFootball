"""Tests for the Model Protocol contract."""

from __future__ import annotations

from pathlib import Path

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


def test_compute_code_hash_is_deterministic(tmp_path: Path) -> None:
    """Hashing the same files twice yields identical output."""
    from projections.models.base import compute_code_hash

    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("hello")
    f2.write_text("world")

    h1 = compute_code_hash([f1, f2])
    h2 = compute_code_hash([f1, f2])
    assert h1 == h2
    assert len(h1) == 8


def test_compute_code_hash_changes_when_content_changes(tmp_path: Path) -> None:
    from projections.models.base import compute_code_hash

    f = tmp_path / "a.py"
    f.write_text("hello")
    h_before = compute_code_hash([f])

    f.write_text("hello!")
    h_after = compute_code_hash([f])
    assert h_before != h_after


def test_compute_code_hash_is_order_independent(tmp_path: Path) -> None:
    """File-list order should not affect the hash (we sort internally)."""
    from projections.models.base import compute_code_hash

    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("alpha")
    f2.write_text("beta")

    h_ab = compute_code_hash([f1, f2])
    h_ba = compute_code_hash([f2, f1])
    assert h_ab == h_ba
