"""Positional ranking, shared by the draft cheat sheet and the season dashboard.

Promoted out of `draft/snake_cheat_sheet.py`, where it was a private helper. Two things now
need it -- the cheat sheet ranks a pool by VORP and ADP, the My Team page ranks players by
year-to-date and rest-of-season points -- and reaching across a package boundary for a
`_private` name is how a module's internals become someone else's contract by accident.
"""

from __future__ import annotations

import pandas as pd


def rank_within_position(frame: pd.DataFrame, by: str, *, ascending: bool) -> pd.Series:
    """Gap-free 1-based integer rank within each position, index-aligned to `frame`.

    Sorts by `by`, tie-broken on `gsis_id` so the order is deterministic across runs, then
    counts within each position group.

    **Deliberately not `Series.rank()`.** Its default `method="average"` yields fractional
    ranks for ties, and "RB 4.5" reads as a bug rather than as a tie -- especially in a table
    where every other rank is a whole number. A shared tie here resolves to consecutive
    integers instead, which is what a reader expects a rank to be.

    `frame` needs `position`, `gsis_id`, and `by`. A null in `by` sorts last under pandas'
    default `na_position`, so an unprojected player ranks behind every projected one rather
    than jumping to the top.
    """
    missing = {"position", "gsis_id", by} - set(frame.columns)
    if missing:
        raise KeyError(f"rank_within_position needs column(s) {sorted(missing)}")
    return (
        frame.sort_values(["position", by, "gsis_id"], ascending=[True, ascending, True])
        .groupby("position", sort=False)
        .cumcount()
        + 1
    )
