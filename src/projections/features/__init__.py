"""Per-position feature builders. Pure functions; no I/O.

Lazy re-exports via ``__getattr__`` — submodules are only loaded on first
access. This preserves the ``from projections.features import build_X``
public API while letting tests for individual helpers (e.g.
``_opponent.opp_epa_allowed_residual``) import without eagerly pulling in
every per-position builder. Required during the Plan 9 phased rollout
where ``_opponent`` is rewritten before each per-position builder is
updated to call the new helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "build_qb_features",
    "build_rb_features",
    "build_te_features",
    "build_wr_features",
]

if TYPE_CHECKING:
    from projections.features.qb import build_qb_features
    from projections.features.rb import build_rb_features
    from projections.features.te import build_te_features
    from projections.features.wr import build_wr_features


_LAZY_BUILDERS = {
    "build_qb_features": ("projections.features.qb", "build_qb_features"),
    "build_rb_features": ("projections.features.rb", "build_rb_features"),
    "build_te_features": ("projections.features.te", "build_te_features"),
    "build_wr_features": ("projections.features.wr", "build_wr_features"),
}


def __getattr__(name: str) -> Any:
    """Resolve per-position builders on first access.

    Avoids eager submodule imports so unrelated helpers (e.g.
    ``projections.features._opponent.opp_epa_allowed_residual``) can be
    imported even while a sibling submodule is mid-refactor.
    """
    if name in _LAZY_BUILDERS:
        module_name, attr_name = _LAZY_BUILDERS[name]
        from importlib import import_module

        mod = import_module(module_name)
        attr = getattr(mod, attr_name)
        globals()[name] = attr  # cache for subsequent accesses
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
