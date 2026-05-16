"""Snake-draft cheat sheet — per-position rankings with gap-based tiers.

Pure transform over the VORP table (`VorpTableSchema`). Reuses `_select_pool`
for in-pool identification. Emits `SnakeCheatSheetSchema`-validated output.

Spec: docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _assign_tiers(
    vorp_desc: NDArray[np.float64],
    n_tiers: int,
) -> NDArray[np.int64]:
    """Gap-based tier assignment over a VORP-descending array.

    Returns a 1-indexed int64 array of the same length as `vorp_desc` giving
    the tier (1..N) for each in-pool player. See spec §3.2 for the algorithm.

    Tie-break: when multiple gaps share the value that is competing for the
    `(N-1)`th-largest slot, the earlier (higher-rank) gap-index wins.
    Deterministic.
    """
    n = len(vorp_desc)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= n_tiers:
        return np.arange(1, n + 1, dtype=np.int64)

    gaps = vorp_desc[:-1] - vorp_desc[1:]
    # lexsort: primary key (last arg) sorts ascending; -gaps ascending = gaps
    # descending. Ties broken by np.arange (gap-index) ascending.
    order = np.lexsort((np.arange(n - 1), -gaps))
    cut_indices = np.sort(order[: n_tiers - 1])

    tier = np.empty(n, dtype=np.int64)
    start = 0
    for t, cut in enumerate(cut_indices, start=1):
        tier[start : cut + 1] = t
        start = int(cut) + 1
    tier[start:] = n_tiers
    return tier
