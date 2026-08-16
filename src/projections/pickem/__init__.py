"""Pick'em Hub — straight-up NFL picks under a minimum-underdogs constraint.

Design: `docs/superpowers/specs/2026-08-16-pickem-hub-design.md`.

The invariant that governs this whole sub-package: the **organizer's sheet**
decides who counts as the underdog, and the **consensus market** decides who is
likely to win. Two sources, two jobs. Conflating them is the most likely way to
produce picks that look plausible and are quietly wrong.
"""

from __future__ import annotations

from projections.pickem._validate import require_schedule_columns
from projections.pickem.probability import (
    add_win_probs,
    american_to_implied,
    devig_pair,
)

__all__ = [
    "add_win_probs",
    "american_to_implied",
    "devig_pair",
    "require_schedule_columns",
]
