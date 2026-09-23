"""Injury report ingest tests.

The traps this source has, one test each: postseason week collision, the two enum mappings
(one forgiving, one strict), and null semantics that are the opposite of ESPN's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_injury_report
from projections.ingest.injury_report import _normalize_one_season
from projections.schemas import InjuryReportSchema, InjuryStatus, PracticeStatus
from projections.store import read_partition

_FULL = "Full Participation in Practice"
_LIMITED = "Limited Participation in Practice"
_DNP = "Did Not Participate In Practice"


def _raw(**overrides: list[object]) -> pd.DataFrame:
    """A three-row `load_injuries` payload, shaped like the real one (Int32 season/week)."""
    base: dict[str, list[object]] = {
        "season": [2024, 2024, 2024],
        "season_type": ["REG", "REG", "REG"],
        "game_type": ["REG", "REG", "REG"],
        "team": ["KC", "MIN", "PHI"],
        "week": [1, 1, 1],
        "gsis_id": ["00-0034857", "00-0036322", "00-0034796"],
        "position": ["QB", "WR", "RB"],
        "full_name": ["Patrick Mahomes", "Justin Jefferson", "Saquon Barkley"],
        "first_name": ["Patrick", "Justin", "Saquon"],
        "last_name": ["Mahomes", "Jefferson", "Barkley"],
        "report_primary_injury": ["Ankle", None, "Hamstring"],
        "report_secondary_injury": [None, None, None],
        "report_status": ["Questionable", None, "Out"],
        "practice_primary_injury": ["Ankle", None, "Hamstring"],
        "practice_secondary_injury": [None, None, None],
        "practice_status": [_LIMITED, _FULL, _DNP],
    }
    base.update(overrides)
    frame = pd.DataFrame(base)
    return frame.astype({"season": "int32", "week": "int32"})


def test_refresh_writes_a_validated_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "projections.ingest.injury_report._fetch_raw_injuries", lambda seasons: _raw()
    )

    written = refresh_injury_report(tmp_path, seasons=[2024])

    assert len(written) == 1
    df = read_partition(tmp_path / "raw", "injury_report", season=2024)
    InjuryReportSchema.validate(df)
    assert len(df) == 3


def test_postseason_rows_are_dropped() -> None:
    """POST week numbering restarts, so a Week 1 POST row would collide with a Week 1 REG row
    on any (gsis_id, season, week) join -- the reason this filter exists at all."""
    raw = _raw(season_type=["REG", "POST", "POST"])

    out = _normalize_one_season(raw)

    assert list(out["gsis_id"]) == ["00-0034857"]


def test_a_null_designation_stays_null_rather_than_becoming_active() -> None:
    """The opposite of `parse_injury_status`, which maps empty to ACTIVE because ESPN omits the
    field for healthy players. Here a null means "on the report, no Sunday designation yet",
    and calling that healthy would move every number downstream."""
    out = _normalize_one_season(_raw())

    by_id = out.set_index("gsis_id")["report_status"]
    assert pd.isna(by_id["00-0036322"])
    assert by_id["00-0034857"] == InjuryStatus.QUESTIONABLE.value
    assert by_id["00-0034796"] == InjuryStatus.OUT.value


def test_verbose_practice_sentences_become_enum_values() -> None:
    out = _normalize_one_season(_raw())

    assert list(out["practice_status"]) == [
        PracticeStatus.LIMITED.value,
        PracticeStatus.FULL.value,
        PracticeStatus.DNP.value,
    ]


def test_a_null_practice_line_is_preserved() -> None:
    out = _normalize_one_season(_raw(practice_status=[_FULL, None, _DNP]))

    assert pd.isna(out.set_index("gsis_id")["practice_status"]["00-0036322"])


def test_an_unknown_practice_label_raises() -> None:
    """Strict where the designation mapping is forgiving: this column has exactly three upstream
    spellings, so a fourth is a source change. Nulling it silently would widen a null that
    already means something specific."""
    raw = _raw(practice_status=[_FULL, "Rested", _DNP])

    with pytest.raises(ValueError, match="unrecognised practice_status"):
        _normalize_one_season(raw)


def test_an_unknown_designation_maps_to_unknown_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Forgiving where the practice mapping is strict: an unmapped designation is a gap in our
    table, not evidence about the player, and must not take a season's ingest down."""
    raw = _raw(report_status=["Questionable", "Probable", "Out"])

    with caplog.at_level(logging.WARNING):
        out = _normalize_one_season(raw)

    assert out.set_index("gsis_id")["report_status"]["00-0036322"] == InjuryStatus.UNKNOWN.value
    assert "Probable" in caplog.text


def test_unmodelled_positions_are_filtered() -> None:
    """The report covers the whole roster -- linemen, defenders, punters -- and we model six."""
    raw = _raw(position=["QB", "OL", "DB"])

    out = _normalize_one_season(raw)

    assert list(out["position"]) == ["QB"]


def test_placeholder_gsis_rows_are_dropped() -> None:
    raw = _raw(gsis_id=["00-0034857", "WAS569019", "00-0034796"])

    out = _normalize_one_season(raw)

    assert list(out["gsis_id"]) == ["00-0034857", "00-0034796"]


def test_team_codes_are_normalized() -> None:
    """nflverse is inconsistent about JAX/JAC and LA/LAR; the schema only admits canonical."""
    raw = _raw(team=["JAX", "LA", "PHI"])

    out = _normalize_one_season(raw)

    assert list(out["team"]) == ["JAC", "LAR", "PHI"]


def test_season_and_week_are_int64_not_int32() -> None:
    """Upstream returns int32; pandera's Series[int] requires int64."""
    out = _normalize_one_season(_raw())

    assert out["season"].dtype == "int64"
    assert out["week"].dtype == "int64"
