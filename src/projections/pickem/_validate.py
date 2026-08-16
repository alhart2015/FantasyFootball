"""Shared column guards for the pick'em pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

_REFRESH_HINT = (
    "Schedules partitions written before those columns existed do not carry them. "
    "Re-ingest the season(s) with:\n"
    '    python -c "from pathlib import Path; '
    "from projections.ingest import refresh_schedules; "
    'refresh_schedules(Path("data"), seasons=[2026])"'
)


def require_schedule_columns(df: pd.DataFrame, columns: Sequence[str], *, needed_for: str) -> None:
    """Raise if `df` is missing any of `columns`, naming the remedy.

    `SchedulesSchema` declares `game_type` / `home_score` / `away_score` as
    not-required so that partitions predating them still validate. That makes
    presence a caller's responsibility: anything that actually reads them must
    check first, or it will fail later with an unhelpful KeyError.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"schedules frame is missing {missing}, required for {needed_for}.\n{_REFRESH_HINT}"
        )
