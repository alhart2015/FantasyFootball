"""The availability model must read every weekly_stats season that has been ingested.

The failure this guards against is silent and was live in the repo: `weekly_stats`
for 2025 was ingested and sitting on disk, but `_HISTORY_SEASONS` still ended at
2024, so every availability probability was built from a history one full season
stale. Nothing errored — `load_store_availability` skips missing partitions by
design, so a range that is too NARROW looks identical to a range that is correct.

The bias is not neutral either. Adding 2025 moved De'Von Achane from 0.844 to
0.876 and Jahmyr Gibbs from 0.941 to 0.961, i.e. the stale range made the model
systematically pessimistic about availability.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from projections.draft.assistant.availability_loader import _HISTORY_SEASONS

_WEEKLY_STATS = Path("data/raw/weekly_stats")


def _ingested_seasons() -> list[int]:
    return sorted(
        int(m.group(1))
        for d in _WEEKLY_STATS.iterdir()
        if d.is_dir() and (m := re.fullmatch(r"season=(\d{4})", d.name))
    )


def test_history_range_is_contiguous_from_2018() -> None:
    """A `range` with a step or a moved start would silently drop seasons."""
    assert _HISTORY_SEASONS.start == 2018
    assert _HISTORY_SEASONS.step == 1


@pytest.mark.skipif(not _WEEKLY_STATS.exists(), reason="requires ingested weekly_stats partitions")
def test_history_range_covers_every_ingested_season() -> None:
    """Bump `_HISTORY_SEASONS` when a new weekly_stats season is ingested.

    `load_store_availability` skips partitions it cannot find, so an under-wide
    range never raises — it just quietly computes availability from less data.
    """
    ingested = _ingested_seasons()
    assert ingested, "weekly_stats directory exists but holds no season= partitions"

    covered = set(_HISTORY_SEASONS)
    missed = sorted(s for s in ingested if s not in covered)
    assert not missed, (
        f"weekly_stats seasons {missed} are ingested but excluded from the availability "
        f"model (_HISTORY_SEASONS covers {_HISTORY_SEASONS.start}-{_HISTORY_SEASONS.stop - 1}). "
        "Bump the upper bound in availability_loader.py."
    )


@pytest.mark.skipif(not _WEEKLY_STATS.exists(), reason="requires ingested weekly_stats partitions")
def test_history_range_does_not_claim_seasons_that_are_not_ingested() -> None:
    """The mirror check. Over-claiming is harmless at runtime (missing partitions are
    skipped) but it makes the constant lie about what the model actually reads."""
    ingested = set(_ingested_seasons())
    over = sorted(s for s in _HISTORY_SEASONS if s not in ingested)
    assert not over, (
        f"_HISTORY_SEASONS claims seasons {over} that have no weekly_stats partition; "
        "ingest them or lower the upper bound."
    )
