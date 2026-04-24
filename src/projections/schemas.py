"""Single source of truth for canonical types: enums, NewTypes, pydantic models, pandera schemas."""

from __future__ import annotations

from enum import StrEnum


class Position(StrEnum):
    """NFL fantasy-relevant positions. Use Position.QB, never the string "QB"."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"
    # Reserved for future IDP support; kept here so RosterSlot can refer to them
    # without a circular import. Not currently produced by ingest.
