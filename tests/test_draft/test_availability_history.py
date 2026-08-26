"""Which weekly_stats seasons the availability model is allowed to read.

Two rules, both enforced in `_usable_history` rather than by a constant someone has
to remember to bump:

1. **Strictly before the target season.** Reading the target season's own outcomes is
   lookahead — `draft.backtest.inputs` builds availability with `season=` the season
   being simulated.
2. **Completed seasons only.** `build_availability` divides games played by the full
   scheduled span, not by the weeks on disk, so a partial partition scores everyone at
   `weeks_so_far / full_season`.

The bug that started this: weekly_stats 2025 was ingested but a hardcoded ceiling still
said 2024, and `read_partition`'s skip-if-missing made the too-narrow range
indistinguishable from a correct one at runtime. There is no ceiling to drift now.

These tests build their own store under `tmp_path`. They deliberately do not read
`data/raw`, so they cannot silently skip on a machine without the partitions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.availability import build_availability
from projections.draft.assistant.availability_loader import (
    _HISTORY_FLOOR,
    _usable_history,
    load_store_availability,
)
from projections.schemas import _PYARROW_STR
from projections.season_calendar import last_regular_week, regular_season_games
from projections.store import write_partition

_PLAYERS = [("00-0000001", "RB"), ("00-0000002", "WR")]


def _complete(season: int) -> int:
    return last_regular_week(season)


def _weekly_stats(season: int, through_week: int, *, reach_week: int | None = None) -> pd.DataFrame:
    """Measured players appear in weeks 1..`through_week`.

    `reach_week` adds a filler player in that week alone. It lets a partition satisfy
    the completeness check (max week >= last_regular_week) WITHOUT giving the measured
    players a game in every calendar week — which would push `frac` to 1.0 and land `p`
    on the `hi=0.97` clamp, so a `p > 0.9` assertion would pass even if the span math
    were wrong.
    """
    rows = [(g, season, w, pos) for g, pos in _PLAYERS for w in range(1, through_week + 1)]
    if reach_week is not None:
        rows.append(("00-0000099", season, reach_week, "WR"))
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "season": [r[1] for r in rows],
            "week": [r[2] for r in rows],
            "position": pd.array([r[3] for r in rows], dtype=_PYARROW_STR),
        }
    )


def _full_season(season: int) -> pd.DataFrame:
    """A complete partition in which the measured players miss exactly one game.

    frac = (games - 1) / games, so p is a real number below the clamp.
    """
    games = regular_season_games(season)
    return _weekly_stats(season, games - 1, reach_week=last_regular_week(season))


def _expected_p(season: int) -> float:
    games = regular_season_games(season)
    return (games - 1) / games


def _id_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array([g for g, _ in _PLAYERS], dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * len(_PLAYERS), dtype=_PYARROW_STR),
        }
    )


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array([g for g, _ in _PLAYERS], dtype=_PYARROW_STR),
            "position": pd.array([p for _, p in _PLAYERS], dtype=_PYARROW_STR),
        }
    )


def _store(tmp_path: Path, seasons: dict[int, int | pd.DataFrame]) -> Path:
    """Write one weekly_stats partition per season, plus id_map.

    A value may be a `through_week` int or a prebuilt frame (see `_full_season`).
    """
    raw = tmp_path / "raw"
    for season, spec in seasons.items():
        frame = spec if isinstance(spec, pd.DataFrame) else _weekly_stats(season, spec)
        write_partition(raw, "weekly_stats", frame, season=season)
    raw.mkdir(parents=True, exist_ok=True)
    _id_map().to_parquet(raw / "id_map.parquet", index=False)
    return tmp_path


def _seasons_read(frames: list[pd.DataFrame]) -> list[int]:
    return sorted(int(f["season"].iloc[0]) for f in frames)


# --- the target season is never its own history -----------------------------


def test_target_season_is_excluded(tmp_path: Path) -> None:
    root = _store(tmp_path, {2022: _complete(2022), 2023: _complete(2023)})
    found = _usable_history(root / "raw", 2023)
    assert _seasons_read(found.frames) == [2022]
    assert found.incomplete == []


def test_seasons_after_the_target_are_excluded(tmp_path: Path) -> None:
    root = _store(tmp_path, {2021: _complete(2021), 2022: _complete(2022), 2023: _complete(2023)})
    found = _usable_history(root / "raw", 2022)
    assert _seasons_read(found.frames) == [2021]


def test_explicit_candidates_cannot_widen_past_the_target(tmp_path: Path) -> None:
    """The override narrows the candidate span; it must not reintroduce lookahead."""
    root = _store(tmp_path, {2021: _complete(2021), 2022: _complete(2022)})
    found = _usable_history(root / "raw", 2022, range(2018, 2030))
    assert _seasons_read(found.frames) == [2021]


# --- incomplete seasons are never read --------------------------------------


def test_in_progress_season_is_skipped_and_reported(tmp_path: Path) -> None:
    """A week-10 partition is what any in-season refresh writes."""
    root = _store(tmp_path, {2022: _complete(2022), 2023: 10})
    found = _usable_history(root / "raw", 2026)
    assert _seasons_read(found.frames) == [2022]
    assert found.incomplete == [2023]


def test_incomplete_season_is_skipped_even_when_it_is_not_the_target(tmp_path: Path) -> None:
    """Target-season exclusion alone would not catch a stale partial partition left
    behind from an earlier year."""
    root = _store(tmp_path, {2021: 4, 2022: _complete(2022)})
    found = _usable_history(root / "raw", 2026)
    assert _seasons_read(found.frames) == [2022]
    assert found.incomplete == [2021]


def test_one_week_short_still_counts_as_incomplete(tmp_path: Path) -> None:
    root = _store(tmp_path, {2022: _complete(2022) - 1})
    found = _usable_history(root / "raw", 2026)
    assert found.frames == []
    assert found.incomplete == [2022]


def test_empty_partition_is_skipped_not_crashed_on(tmp_path: Path) -> None:
    """`write_partition` unlinks before writing, so a failed refresh can leave a
    readable but empty file. `df['week'].max()` on it is NaN, which would raise."""
    raw = _store(tmp_path, {2022: _complete(2022)}) / "raw"
    write_partition(raw, "weekly_stats", _weekly_stats(2023, 1).iloc[:0], season=2023)
    found = _usable_history(raw, 2026)
    assert _seasons_read(found.frames) == [2022]
    assert found.incomplete == [2023]


def test_never_ingested_is_not_reported_as_incomplete(tmp_path: Path) -> None:
    root = _store(tmp_path, {2022: _complete(2022)})
    found = _usable_history(root / "raw", 2026)
    assert _seasons_read(found.frames) == [2022]
    assert found.incomplete == []


# --- why the completeness rule exists, priced -------------------------------


def test_reading_a_partial_season_would_have_poisoned_p() -> None:
    """The harm the rule prevents, measured rather than asserted in prose.

    Four weeks of a 17-game season is scored 4/17, and `p_raw` averages that against
    the good season unweighted.
    """
    schedules = pd.DataFrame(columns=["season", "week", "home_team", "away_team"])
    complete = _weekly_stats(2022, _complete(2022))
    partial = _weekly_stats(2023, 4)

    with pytest.warns(UserWarning):  # empty schedules -> no byes
        good = build_availability(complete, schedules, _id_map(), _pool(), season=2026)
    with pytest.warns(UserWarning):
        poisoned = build_availability(
            pd.concat([complete, partial], ignore_index=True),
            schedules,
            _id_map(),
            _pool(),
            season=2026,
        )

    assert good.p_week("00-0000001") > 0.9
    assert poisoned.p_week("00-0000001") < 0.75
    assert poisoned.p_week("00-0000001") < good.p_week("00-0000001") - 0.2


# --- loader wiring ----------------------------------------------------------


def test_loader_uses_only_completed_prior_seasons(tmp_path: Path) -> None:
    """End-to-end through `load_store_availability`, not just the helper."""
    root = _store(tmp_path, {2022: _full_season(2022), 2023: _full_season(2023), 2024: 6})
    with pytest.warns(UserWarning, match="incomplete"):
        avail = load_store_availability(_pool(), season=2026, data_root=root)

    # Both usable seasons have the player missing exactly one game, so p is a real
    # 16/17 rather than the hi=0.97 clamp a play-every-week fixture would produce.
    assert avail.p_week("00-0000001") == pytest.approx(_expected_p(2022), abs=1e-9)
    assert avail.p_week("00-0000001") < 0.97


def test_loader_excludes_the_target_season_end_to_end(tmp_path: Path) -> None:
    """2023 is complete on disk but is the target, so it must not be read. If it were,
    the run would succeed either way — so assert on the selection, not just success."""
    root = _store(tmp_path, {2022: _complete(2022), 2023: _complete(2023)})
    found = _usable_history(root / "raw", 2023)
    assert _seasons_read(found.frames) == [2022]
    load_store_availability(_pool(), season=2023, data_root=root)


def test_error_names_the_season_range_when_nothing_precedes_the_target(tmp_path: Path) -> None:
    """`--data-root` is the wrong thing to send someone to check when the real problem
    is that no season exists before the target."""
    root = _store(tmp_path, {2022: _complete(2022)})
    with pytest.raises(FileNotFoundError, match="not a data-root problem"):
        load_store_availability(_pool(), season=_HISTORY_FLOOR, data_root=root)


def test_error_points_at_the_data_root_when_the_store_is_empty(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    _id_map().to_parquet(raw / "id_map.parquet", index=False)
    with pytest.raises(FileNotFoundError, match="check --data-root"):
        load_store_availability(_pool(), season=2026, data_root=tmp_path)


# --- a season that vanished --------------------------------------------------


def test_missing_season_between_present_ones_is_reported(tmp_path: Path) -> None:
    """`write_partition` unlinks before writing, so an interrupted refresh leaves no
    file. Silently narrowing the history is the same failure this module exists to
    prevent, just from a different cause than a stale constant."""
    root = _store(tmp_path, {2021: _complete(2021), 2023: _complete(2023)})
    found = _usable_history(root / "raw", 2026)
    assert _seasons_read(found.frames) == [2021, 2023]
    assert found.gaps == [2022]


def test_a_store_that_simply_starts_late_is_not_a_gap(tmp_path: Path) -> None:
    """Only holes BETWEEN present seasons signal loss. A partial backfill is normal
    and must not cry wolf on every year before it."""
    root = _store(tmp_path, {2024: _complete(2024), 2025: _complete(2025)})
    found = _usable_history(root / "raw", 2026)
    assert found.gaps == []


def test_loader_warns_about_a_vanished_season(tmp_path: Path) -> None:
    root = _store(tmp_path, {2021: _full_season(2021), 2023: _full_season(2023)})
    with pytest.warns(UserWarning, match="no partition at all"):
        load_store_availability(_pool(), season=2026, data_root=root)


# --- the explicit span override ---------------------------------------------


def test_loader_honours_an_explicit_history_span(tmp_path: Path) -> None:
    """The public keyword, exercised through the loader rather than the helper."""
    root = _store(tmp_path, {2021: _full_season(2021), 2022: _full_season(2022)})
    avail = load_store_availability(
        _pool(), season=2026, data_root=root, history_seasons=range(2022, 2023)
    )
    found = _usable_history(root / "raw", 2026, range(2022, 2023))
    assert _seasons_read(found.frames) == [2022]
    assert avail.p_week("00-0000001") == pytest.approx(_expected_p(2022), abs=1e-9)


def test_error_names_only_the_span_that_was_examined(tmp_path: Path) -> None:
    """With an explicit span, naming the full floor-to-target range would point the
    reader at seasons that were never candidates."""
    root = _store(tmp_path, {2021: _full_season(2021)})
    with pytest.raises(FileNotFoundError, match="2022-2022") as exc:
        load_store_availability(
            _pool(), season=2026, data_root=root, history_seasons=range(2022, 2023)
        )
    assert "2018" not in str(exc.value)


# --- the same messages, in a form a UI can render ----------------------------------------------


def test_the_loader_hands_back_its_warnings_when_asked(tmp_path: Path) -> None:
    """`warnings.warn` alone reaches stderr, which is fine for a CLI and useless for a web page.

    The season dashboard has a notes slot for exactly this, and without it renders a projection
    built on a thinner injury history than the reader believes, saying nothing. An explicit list
    rather than `warnings.catch_warnings` at the call site: that mutates process-global filter
    state, which is not thread-safe, and the caller is a request handler.
    """
    root = _store(tmp_path, {2022: _full_season(2022), 2023: _full_season(2023), 2024: 6})
    notes: list[str] = []
    with pytest.warns(UserWarning, match="incomplete"):
        load_store_availability(_pool(), season=2026, data_root=root, notes=notes)

    assert len(notes) == 1
    assert "incomplete" in notes[0] and "2024" in notes[0]


def test_a_vanished_season_reaches_the_notes_too(tmp_path: Path) -> None:
    root = _store(tmp_path, {2022: _full_season(2022), 2024: _full_season(2024)})
    notes: list[str] = []
    with pytest.warns(UserWarning, match="no partition at all"):
        load_store_availability(_pool(), season=2026, data_root=root, notes=notes)

    assert any("no partition at all" in note for note in notes), notes


def test_a_clean_history_adds_no_notes(tmp_path: Path) -> None:
    """Otherwise every page carries a warning, which trains the reader to ignore all of them."""
    root = _store(tmp_path, {2022: _full_season(2022), 2023: _full_season(2023)})
    notes: list[str] = []
    load_store_availability(_pool(), season=2026, data_root=root, notes=notes)
    assert notes == []
