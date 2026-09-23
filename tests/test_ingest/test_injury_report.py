"""Injury report ingest tests.

The traps this source has, one test each: two upstream payload shapes that differ by season,
float-typed season/week before 2022, the two enum mappings (one forgiving, one strict), and
null semantics that are the opposite of ESPN's.
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
    """A three-row `load_injuries` payload in the **2020-2024** shape.

    That shape carries `game_type` and `date_modified` and has no `season_type`; 2025 carries
    `season_type` and drops `date_modified`. The original fixture here assumed the 2025 shape
    and a real pull against 2024 failed on the missing column, so the default is now the older
    one and `test_both_upstream_payload_shapes_normalize` covers the newer.
    """
    base: dict[str, list[object]] = {
        "season": [2024, 2024, 2024],
        "game_type": ["REG", "REG", "REG"],
        "date_modified": ["2024-09-04", "2024-09-04", "2024-09-04"],
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


def test_postseason_rows_are_kept() -> None:
    """Upstream's postseason weeks continue rather than restart (WC=19, DIV=20, CON=21, SB=22),
    so there is no (gsis_id, season, week) collision and nothing to filter. Keeping them matches
    every other per-game source here."""
    raw = _raw(week=[19, 20, 22], game_type=["WC", "DIV", "SB"])

    out = _normalize_one_season(raw)

    assert sorted(out["week"]) == [19, 20, 22]


def test_both_upstream_payload_shapes_normalize() -> None:
    """2020-2024 ship `game_type` + `date_modified`; 2025 ships `season_type` and no
    `date_modified`. Neither column is needed, but a filter keyed on one of them dies on the
    seasons that lack it -- which is exactly what a real 2024 pull did."""
    older = _raw()
    newer = _raw().drop(columns=["date_modified"]).assign(season_type=["REG", "REG", "REG"])

    assert len(_normalize_one_season(older)) == 3
    assert len(_normalize_one_season(newer)) == 3


def test_float_typed_season_and_week_are_coerced() -> None:
    """2020 and earlier return both as Float64 where 2022+ return int32."""
    raw = _raw().astype({"season": "float64", "week": "float64"})

    out = _normalize_one_season(raw)

    assert out["season"].dtype == "int64"
    assert out["week"].dtype == "int64"


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


def test_an_unknown_practice_label_maps_to_unknown_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Upstream really does emit a fourth value -- `"Note"`, 7 rows across 2016-2025, marking a
    free-text clarification rather than a practice line. One unmapped label in a decade must not
    take a season's ingest down (issue #169), so it warns and lands as UNKNOWN."""
    raw = _raw(practice_status=[_FULL, "Note", _DNP])

    with caplog.at_level(logging.WARNING):
        out = _normalize_one_season(raw)

    assert out.set_index("gsis_id")["practice_status"]["00-0036322"] == PracticeStatus.UNKNOWN.value
    assert "Note" in caplog.text


def test_a_whitespace_only_practice_line_is_null_not_unknown() -> None:
    """Upstream emits 212 whitespace-only values across 2016-2025. Those mean "no practice line"
    -- the existing null -- not "a label we could not map"."""
    raw = _raw(practice_status=[_FULL, "\n    ", _DNP])

    out = _normalize_one_season(raw)

    assert pd.isna(out.set_index("gsis_id")["practice_status"]["00-0036322"])


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
