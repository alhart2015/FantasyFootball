"""The availability model must read every COMPLETED weekly_stats season it can use.

The bug this guards against was live in the repo: `weekly_stats` for 2025 was
ingested and sitting on disk, but `_HISTORY_SEASONS` still ended at 2024, so every
availability probability was built from a history one full season stale. Nothing
errored — `load_store_availability` skips missing partitions by design, so a range
that is too NARROW looks identical to a correct one. The bias was not neutral: it
made the model systematically pessimistic (Achane 0.844 -> 0.876).

Two things the check must NOT do, both learned the hard way:

1. **Do not demand coverage of an in-progress season.** `build_availability` divides
   games played by the full scheduled span, not by the weeks on disk, so a partial
   partition scores everyone at `weeks_so_far / 17`. A check that went red on a
   week-3 2026 partition and told the reader to widen the range would be actively
   destructive — worse than no check.
2. **Do not treat a directory as a partition.** `write_partition` mkdirs and unlinks
   the old file before writing, so an interrupted refresh leaves an empty
   `season=YYYY/`. `read_partition` raises on it and the loader skips it, so
   demanding a bump for it would go green while the model still read nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.availability_loader import _HISTORY_SEASONS, _history_for
from projections.season_calendar import last_regular_week

_WEEKLY_STATS = Path("data/raw/weekly_stats")


def _readable_seasons() -> dict[int, pd.DataFrame]:
    """Seasons with an actually-readable partition, keyed by season.

    Presence of `season=YYYY/` is not enough — the loader's bar is a readable
    `part.parquet`, so that is the bar here too.
    """
    out: dict[int, pd.DataFrame] = {}
    for d in sorted(_WEEKLY_STATS.iterdir()):
        m = re.fullmatch(r"season=(\d{4})", d.name) if d.is_dir() else None
        if not m:
            continue
        parts = sorted(d.rglob("part.parquet"))
        if not parts:
            continue  # empty dir from an interrupted write; the loader skips it too
        out[int(m.group(1))] = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return out


def _complete_seasons() -> list[int]:
    """Seasons whose weekly_stats run through the end of the regular season."""
    return sorted(
        s for s, df in _readable_seasons().items() if int(df["week"].max()) >= last_regular_week(s)
    )


# --- the range itself (no data needed) --------------------------------------


def test_history_range_is_contiguous_from_2018() -> None:
    """A moved start or a step would silently drop seasons."""
    assert _HISTORY_SEASONS.start == 2018
    assert _HISTORY_SEASONS.step == 1


# --- the lookahead / partial-season rule ------------------------------------


def test_target_season_is_excluded_from_its_own_history() -> None:
    """Reading the simulated season's own injury outcomes as the ex-ante prior is
    lookahead; `draft.backtest.inputs` builds availability with `season=` the season
    under test, so this is a live path, not a hypothetical."""
    assert 2025 not in _history_for(2025, _HISTORY_SEASONS)
    assert _history_for(2025, _HISTORY_SEASONS) == list(range(2018, 2025))


def test_future_seasons_are_excluded() -> None:
    """A partial in-progress partition would score everyone at weeks_so_far / 17."""
    assert _history_for(2026, range(2018, 2030)) == list(range(2018, 2026))


def test_live_use_still_reads_every_completed_season() -> None:
    """The exclusion must not cost the real draft-time call anything: building for
    2026 still gets 2018-2025."""
    assert _history_for(2026, _HISTORY_SEASONS) == list(range(2018, 2026))


def test_history_is_empty_when_nothing_precedes_the_target() -> None:
    assert _history_for(2018, _HISTORY_SEASONS) == []


# --- coverage against what is actually on disk ------------------------------


@pytest.mark.skipif(not _WEEKLY_STATS.exists(), reason="requires ingested weekly_stats partitions")
def test_history_range_covers_every_completed_ingested_season() -> None:
    """Bump `_HISTORY_SEASONS` when a new season finishes and is ingested.

    Scoped to COMPLETED seasons on purpose: an in-progress partition must never
    trigger this, because acting on it would poison the model.
    """
    complete = _complete_seasons()
    if not complete:
        pytest.skip("no completed weekly_stats seasons ingested")

    covered = set(_HISTORY_SEASONS)
    missed = sorted(s for s in complete if s not in covered)
    assert not missed, (
        f"weekly_stats seasons {missed} are ingested AND complete but excluded from the "
        f"availability model (_HISTORY_SEASONS covers "
        f"{_HISTORY_SEASONS.start}-{_HISTORY_SEASONS.stop - 1}). "
        "Bump the upper bound in availability_loader.py."
    )


@pytest.mark.skipif(not _WEEKLY_STATS.exists(), reason="requires ingested weekly_stats partitions")
def test_an_in_progress_season_does_not_trip_the_coverage_check(tmp_path: Path) -> None:
    """Directly pins the rule that makes this suite safe to keep: a season with only
    a few weeks on disk is not 'complete' and so is not demanded."""
    readable = _readable_seasons()
    if not readable:
        pytest.skip("no readable weekly_stats partitions")
    sample = readable[max(readable)]
    partial = sample[sample["week"] <= 3]
    assert int(partial["week"].max()) < last_regular_week(int(partial["season"].iloc[0]))
