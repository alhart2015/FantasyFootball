"""Pluggable roster valuation (spec §3.5).

`StartersValuer` is the cheap starters-only metric (today's default). `SeasonValuer`
is the risk-aware expected-season-points metric. Both satisfy `RosterValuer`, so the
tournament can A/B them. `SeasonValuer` derives a deterministic per-roster seed (a
sha256 of the sorted gsis_ids, xored with base_seed) so identical rosters score
identically and the tournament stays reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.season_value import expected_season_points
from projections.schemas import RosterSlot


@runtime_checkable
class RosterValuer(Protocol):
    def value(self, roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> float:
        """Score a completed roster."""
        ...


@dataclass(frozen=True)
class StartersValuer:
    """Optimal single-week starting lineup (the cheap, deterministic default)."""

    def value(self, roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> float:
        return optimal_lineup_points(roster, roster_slots)


def _roster_seed(base_seed: int, roster: pd.DataFrame) -> int:
    """Stable 32-bit seed from base_seed + the roster's sorted gsis_ids."""
    key = ",".join(sorted(str(g) for g in roster["gsis_id"]))
    digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    return (base_seed ^ digest) & 0xFFFFFFFF


@dataclass(frozen=True)
class SeasonValuer:
    """Expected season points under availability risk (spec §3.4)."""

    availability: PlayerAvailability
    n_sims: int
    base_seed: int
    weeks: Iterable[int] = field(default_factory=lambda: range(1, 18))

    def value(self, roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> float:
        rng = np.random.default_rng(_roster_seed(self.base_seed, roster))
        return expected_season_points(
            roster,
            roster_slots,
            self.availability,
            n_sims=self.n_sims,
            rng=rng,
            weeks=self.weeks,
        )
