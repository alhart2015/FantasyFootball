"""Smoke test: package imports and version is set."""

from __future__ import annotations

import projections


def test_package_imports() -> None:
    assert projections.__version__ == "0.0.1"
