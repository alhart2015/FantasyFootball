"""The ADP-bot: a non-hero seat's pick policy (spec §3.2).

Pure noisy-ADP. Realism comes from ADP itself (consensus ADP already spaces
positions like a real room), so the bot takes no roster argument -- a
roster-eligibility filter would be a no-op under a shared bench anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.schemas import GsisId, validate_gsis_id


def bot_pick(available: pd.DataFrame, rng: np.random.Generator, *, adp_jitter: float) -> GsisId:
    """Return the lowest noisy-ADP player among `available`.

    `available` needs columns `gsis_id` and `consensus_adp` (nullable Float64).
    Null ADP -> treated as `+inf` (no market signal). Ties (incl. all-null) break
    on `gsis_id` ascending. `available` must be non-empty (caller guarantees it).
    """
    adp = available["consensus_adp"].to_numpy(dtype=float, na_value=np.inf)
    noisy = adp + rng.normal(0.0, adp_jitter, size=len(available))
    gsis = available["gsis_id"].to_numpy(dtype=str)
    # lexsort sorts by the LAST key first -> primary noisy asc, secondary gsis asc.
    winner = int(np.lexsort((gsis, noisy))[0])
    return validate_gsis_id(str(gsis[winner]))
