"""Feature-cache reader. Pairs with scripts/refresh_features.py (writer).

Plan 3c — closes TODO #4. The cache layout
``data/features/{position}/season=YYYY/week=WW/part.parquet`` mirrors the
existing ``data/raw/{table}/...`` and ``data/projections/...`` conventions.
``{position}`` is lowercase (qb / rb / te / wr).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from projections.schemas import Position
from projections.store import read_partition


def read_features(
    position: Position,
    season: int,
    *,
    weeks: Iterable[int] | None = None,
    features_root: Path = Path("data/features"),
) -> pd.DataFrame:
    """Load cached features for a (position, season).

    Returns a DataFrame concatenated across the requested weeks (or all
    available weeks for the season if ``weeks`` is None), re-validated
    against the appropriate FeaturesSchema looked up via POSITION_DISPATCH.

    Raises:
        FileNotFoundError: the (position, season) cache directory is missing
            or empty. The error message names the path so the caller can
            run ``scripts/refresh_features.py`` against it.
    """
    # Local import to avoid a top-level circular: __init__.py imports baseline,
    # baseline imports schemas, schemas is imported by features/cache.py.
    from projections.models import POSITION_DISPATCH

    table = position.value.lower()
    season_dir = features_root / table / f"season={season}"
    if not season_dir.exists() or not any(season_dir.rglob("part.parquet")):
        raise FileNotFoundError(
            f"No feature cache for ({position.value}, {season}) at {season_dir}. "
            f"Run: python scripts/refresh_features.py {table} --seasons {season}"
        )

    if weeks is None:
        df = read_partition(features_root, table, season=season)
    else:
        frames = [read_partition(features_root, table, season=season, week=int(w)) for w in weeks]
        df = pd.concat(frames, ignore_index=True)

    schema = POSITION_DISPATCH[position].feature_schema
    return schema.validate(df)
