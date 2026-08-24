"""Skip guard for tests that read the local `data/features/` cache.

`data/features/` is gitignored — it is a per-machine artifact built by
`scripts/refresh_features.py`, not fixture data in the repo. So every test that reads it has
to decide whether the local cache is usable and skip when it is not.

The obvious guard, `(features_root / position / f"season={season}").exists()`, is not enough,
and four test modules each carried a copy of it. `read_features` re-validates the cached frame
against the *current* `FeaturesSchema`, so a cache written before a schema gained a column
exists, passes an existence check, and then fails deep inside pandera with

    SchemaError: column 'preseason_implied_team_total' not in dataframe

That is exactly what happened: caches built 2026-05-11 predate the four Vegas team-context
columns added in fe10645 / 1f432eb, so `test_backtest_smoke_one_cell` failed on every fresh
checkout with a stale cache instead of skipping with a message that says what to run.

`feature_cache_skip_reason` closes that gap by actually attempting the read the test will
perform. It is deliberately strict about *why* it swallows an exception: a schema mismatch or
a missing partition means "your cache is stale or absent, go rebuild it", which is a skip;
anything else is a real bug and propagates.
"""

from __future__ import annotations

from pathlib import Path

import pandera.errors

from projections.features.cache import read_features
from projections.schemas import Position

DEFAULT_FEATURES_ROOT = Path("data/features")
DEFAULT_RAW_ROOT = Path("data/raw")


def feature_cache_skip_reason(
    position: Position,
    season: int,
    *,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> str | None:
    """Why this machine cannot run a test over the (position, season) feature cache, or None.

    Checks the raw weekly_stats partition the harness joins against, then tries the cached
    read itself — including the schema re-validation, which is the step an existence check
    misses. Returns a message naming the exact `refresh_features.py` invocation to fix it.
    """
    table = position.value.lower()
    weekly = raw_root / "weekly_stats" / f"season={season}"
    if not weekly.exists():
        return f"{weekly} missing — run the raw refresh for {season}"

    rebuild = f"run: python scripts/refresh_features.py {table} --seasons {season}"
    try:
        read_features(position, season, features_root=features_root)
    except FileNotFoundError:
        return f"no {table} feature cache for {season} — {rebuild}"
    except pandera.errors.SchemaError as exc:
        # A cache that predates a schema change. Naming the mismatch matters: without it the
        # skip reads as "no cache" and you rebuild nothing, because the directory is right
        # there. Keep only the part before pandera's full column dump — a 34-name list in a
        # skip reason buries the one column that actually differs.
        detail = str(exc).split(". Columns in dataframe:")[0].strip()
        return f"{table} feature cache for {season} is stale ({detail}) — {rebuild}"
    return None
