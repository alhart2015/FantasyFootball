"""Out-of-money auction bots draft like snake drafters: act on the best-available-by-noisy-ADP
player at a needed position. `SnakeBoard` holds one bot's fixed (drawn-once-per-draft) noisy-ADP
ranking; `best_available` answers the per-nomination target query. See the design doc
`docs/superpowers/specs/2026-06-19-auction-broke-bot-snake-design.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import _best_by_noisy_adp
from projections.schemas import GsisId, Position

DEFAULT_BROKE_ADP_JITTER: float = 8.0  # the snake-draft ADP jitter (hero_harness default)


def adp_usable(pool: pd.DataFrame) -> bool:
    """True iff the pool carries a usable consensus_adp signal (present and not all-null).

    `consensus_adp` is OPTIONAL on VorpTableSchema (weekly-path tables omit it). When unusable the
    auction engine disables the snake regime and runs exactly as before.
    """
    return "consensus_adp" in pool.columns and bool(pool["consensus_adp"].notna().any())


class SnakeBoard:
    """One bot's fixed noisy-ADP board for a single draft.

    The noise is drawn ONCE at construction (a real manager's board is set on draft day and does not
    reshuffle every nomination). `best_available` consumes no RNG.
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        rng: np.random.Generator,
        *,
        adp_jitter: float = DEFAULT_BROKE_ADP_JITTER,
    ) -> None:
        ordered = pool.sort_values("gsis_id", ignore_index=True)
        self._gsis = ordered["gsis_id"].to_numpy(dtype=str)
        self._pos = ordered["position"].astype(str).to_numpy(dtype=str)
        adp = ordered["consensus_adp"].to_numpy(dtype=float, na_value=np.inf)
        self._noisy = adp + rng.normal(0.0, adp_jitter, size=len(ordered))

    def best_available(
        self, drafted: frozenset[str], eligible: frozenset[Position]
    ) -> GsisId | None:
        """Lowest fixed-noisy-ADP undrafted gsis whose position is in `eligible`; None if none."""
        if not eligible:
            return None
        elig_str = np.array([p.value for p in eligible], dtype=str)
        mask = np.isin(self._pos, elig_str) & ~np.isin(
            self._gsis, np.array(list(drafted), dtype=str)
        )
        if not mask.any():
            return None
        return _best_by_noisy_adp(self._gsis[mask], self._noisy[mask])
