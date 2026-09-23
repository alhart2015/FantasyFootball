from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import build_id_map
from projections.ingest.identity import (
    drop_placeholder_gsis_rows,
    normalize_join_id,
    placeholder_name_key,
    resolve_gsis_via_id_map,
)


def test_folds_accents_and_lowercases() -> None:
    # 'José'/'Jose' must agree across sources
    assert placeholder_name_key("José Hernández", "RB") == placeholder_name_key(
        "Jose Hernandez", "RB"
    )


def test_strips_generational_suffixes_and_punctuation() -> None:
    assert placeholder_name_key("Marvin Harrison Jr.", "WR") == placeholder_name_key(
        "Marvin Harrison", "WR"
    )
    # hyphen/punctuation removed
    assert placeholder_name_key("Amon-Ra St. Brown", "WR") == "amonrastbrown|wr"


def test_position_is_part_of_key() -> None:
    assert placeholder_name_key("Taysom Hill", "QB") != placeholder_name_key("Taysom Hill", "TE")


def test_degenerate_name_falls_back_to_raw_lower() -> None:
    # all-suffix/punctuation name keys on the raw name, not the empty '|pos'
    assert placeholder_name_key("Jr.", "WR") == "jr.|wr"


def test_normalize_join_id_strips_float_suffix_and_whitespace() -> None:
    s = pd.Series(["4374302.0", " 4374302 ", "4374302.00", "00-0036900"])
    out = normalize_join_id(s)
    assert out.tolist() == ["4374302", "4374302", "4374302", "00-0036900"]
    assert out.dtype == "string"


# --- drop_placeholder_gsis_rows (issue #169) --------------------------------------------------


def test_drops_pfr_style_placeholder_ids() -> None:
    """The exact ids from the 2026 payload that aborted `depth_charts` ingest.

    `WAS569019` is Mike Washington Jr. (LV); nflverse holds a PFR-style placeholder until NFL.com
    assigns a real gsis_id. Every schema keyed on gsis_id enforces GSIS_ID_PATTERN, so these
    cannot be persisted -- the only question is whether a source drops twelve players or dies.
    """
    frame = pd.DataFrame(
        {
            "gsis_id": ["00-0041562", "WAS569019", "BAI173035", "00-0040906", "MEN516487"],
            "full_name": ["real a", "Mike Washington Jr.", "David Bailey", "real b", "rookie"],
        }
    )
    kept = drop_placeholder_gsis_rows(frame, source="test")
    assert list(kept["gsis_id"]) == ["00-0041562", "00-0040906"]


def test_drops_nulls_without_claiming_they_were_placeholders() -> None:
    """A null id is an ordinary older-season gap, not the placeholder story. Counting it into the
    placeholder warning would overstate how many players nflverse is holding back."""
    frame = pd.DataFrame({"gsis_id": ["00-0041562", None, pd.NA]})
    kept = drop_placeholder_gsis_rows(frame, source="test")
    assert list(kept["gsis_id"]) == ["00-0041562"]


def test_warns_once_naming_the_source_and_the_count(caplog: pytest.LogCaptureFixture) -> None:
    """A thin partition for a fresh draft class must not be a silent diagnostic chase -- the log
    line is the only signal that rows were dropped rather than never published."""
    frame = pd.DataFrame({"gsis_id": ["00-0041562", "WAS569019", "BAI173035"]})
    with caplog.at_level(logging.WARNING):
        drop_placeholder_gsis_rows(frame, source="depth_charts (snapshot format)")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "depth_charts (snapshot format)" in warnings[0].getMessage()
    assert "2 row(s)" in warnings[0].getMessage()


def test_stays_quiet_when_every_id_is_canonical(caplog: pytest.LogCaptureFixture) -> None:
    """Most refreshes drop nothing; a warning every run would train the reader to ignore it."""
    frame = pd.DataFrame({"gsis_id": ["00-0041562", "00-0040906"]})
    with caplog.at_level(logging.WARNING):
        kept = drop_placeholder_gsis_rows(frame, source="test")
    assert len(kept) == 2
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_a_null_alone_does_not_trigger_the_placeholder_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame({"gsis_id": ["00-0041562", None]})
    with caplog.at_level(logging.WARNING):
        drop_placeholder_gsis_rows(frame, source="test")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_does_not_mutate_the_input() -> None:
    """Ingest paths reuse the frame they pass in; an in-place filter would surprise the caller."""
    frame = pd.DataFrame({"gsis_id": ["00-0041562", "WAS569019"]})
    drop_placeholder_gsis_rows(frame, source="test")
    assert len(frame) == 2


def test_honours_a_custom_id_column() -> None:
    frame = pd.DataFrame({"player_id": ["00-0041562", "WAS569019"]})
    kept = drop_placeholder_gsis_rows(frame, source="test", gsis_col="player_id")
    assert list(kept["player_id"]) == ["00-0041562"]


def test_dropping_every_row_raises_rather_than_wiping_a_partition() -> None:
    """The guard that keeps this helper from being worse than the crash it replaced.

    `store.write_partition` unlinks the existing file before writing, so an empty result
    overwrites a good season's partition with zero rows -- and `scripts/refresh_data.py` reports
    OK and exits 0. Losing a handful of players is a roster quirk; losing all of them is upstream
    changing its id format.
    """
    frame = pd.DataFrame({"gsis_id": ["WAS569019", "BAI173035", "SMI283040"]})
    with pytest.raises(ValueError, match="every one of the 3 row"):
        drop_placeholder_gsis_rows(frame, source="depth_charts (snapshot format)")


def test_the_raise_names_the_source_and_what_to_check() -> None:
    """The message has to be actionable from a summary line alone -- this fires on an unattended
    refresh, where nobody is holding the payload."""
    frame = pd.DataFrame({"gsis_id": ["WAS569019"]})
    with pytest.raises(ValueError) as excinfo:
        drop_placeholder_gsis_rows(frame, source="refresh_ngs (passing)")
    message = str(excinfo.value)
    assert "refresh_ngs (passing)" in message
    assert "GSIS_ID_PATTERN" in message


def test_one_surviving_row_is_enough_to_not_raise() -> None:
    """The line is at *all* dropped, not at a percentage. Any threshold would be unvalidated and
    would eventually abort a legitimately thin week; a format change is all-or-nothing anyway."""
    frame = pd.DataFrame({"gsis_id": ["WAS569019", "BAI173035", "00-0041562"]})
    kept = drop_placeholder_gsis_rows(frame, source="test")
    assert list(kept["gsis_id"]) == ["00-0041562"]


def test_an_all_null_frame_does_not_raise() -> None:
    """Nulls are an ordinary older-season gap, not an id-format change, so they must not trip the
    guard -- `refresh_draft_picks` legitimately sees null-heavy frames for old drafts."""
    frame = pd.DataFrame({"gsis_id": [None, None]})
    assert drop_placeholder_gsis_rows(frame, source="test").empty


def test_an_empty_frame_survives() -> None:
    """A season with nothing published yet must return empty, not raise."""
    frame = pd.DataFrame({"gsis_id": pd.Series([], dtype="object")})
    assert drop_placeholder_gsis_rows(frame, source="test").empty


@pytest.mark.parametrize(
    "bad",
    [
        "00-004156",  # too few digits
        "00-00415622",  # too many
        "0-0041562",  # too few leading digits
        " 00-0041562",  # leading space
        "00-0041562x",  # trailing junk
        "98-0000001-extra",
    ],
)
def test_near_miss_ids_are_dropped(bad: str) -> None:
    """The pattern is anchored on both ends. A near-miss that survived here would reach a schema
    and abort the ingest anyway -- which is the failure this whole helper exists to prevent."""
    frame = pd.DataFrame({"gsis_id": ["00-0041562", bad]})
    assert list(drop_placeholder_gsis_rows(frame, source="test")["gsis_id"]) == ["00-0041562"]


def test_reserved_placeholder_gsis_ids_are_kept() -> None:
    """`external_projections` mints deterministic `99-`/`98-` ids that DO match the pattern (see
    `_make_placeholder_gsis`, and the D/ST rows in id_map). Those are canonical-shaped by design
    and must survive -- dropping them would silently empty the defense pool."""
    frame = pd.DataFrame({"gsis_id": ["99-0001234", "98-0000001"]})
    assert len(drop_placeholder_gsis_rows(frame, source="test")) == 2


# --- resolve_gsis_via_id_map ------------------------------------------------------------------
#
# The pfr_id -> gsis_id crosswalk, shared by `snap_counts` and every `pfr_advstats` stat type.
# `tests/test_ingest/test_snap_counts.py` covers it through one real caller; these cover the
# helper's own contract, including the two edges no caller currently exercises.


def _build_id_map(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: fake_id_map_df)
    build_id_map(tmp_path)


def test_resolve_attaches_gsis_and_drops_both_id_columns(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_id_map(tmp_path, fake_id_map_df, monkeypatch)
    frame = pd.DataFrame({"pfr_player_id": ["MahoPa00", "KelcTr00"], "snaps": [60, 55]})

    out = resolve_gsis_via_id_map(frame, tmp_path)

    assert set(out["gsis_id"]) == {"00-0034857", "00-0030506"}
    # The frame is gsis-keyed now; neither pfr column may survive into a schema.
    assert "pfr_player_id" not in out.columns
    assert "pfr_id" not in out.columns
    assert list(out["snaps"]) == [60, 55]


def test_resolve_drops_rows_with_no_id_map_entry(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pfr id we don't carry is a player we don't track -- deep bench, practice squad, a
    lineman on a fumble recovery. There is no GsisId to give them, so the row goes."""
    _build_id_map(tmp_path, fake_id_map_df, monkeypatch)
    frame = pd.DataFrame({"pfr_player_id": ["MahoPa00", "NobdyWh00"], "snaps": [60, 2]})

    out = resolve_gsis_via_id_map(frame, tmp_path)

    assert list(out["gsis_id"]) == ["00-0034857"]


def test_resolve_honours_a_custom_pfr_column(
    tmp_path: Path, fake_id_map_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pfr_col` is parameterized so a caller need not assume upstream's spelling."""
    _build_id_map(tmp_path, fake_id_map_df, monkeypatch)
    frame = pd.DataFrame({"player_pfr": ["BarkSa00"], "carries": [22]})

    out = resolve_gsis_via_id_map(frame, tmp_path, pfr_col="player_pfr")

    assert list(out["gsis_id"]) == ["00-0034796"]
    assert "player_pfr" not in out.columns


def test_resolve_without_an_id_map_raises(tmp_path: Path) -> None:
    """`INGEST_SOURCES` orders id_map first so this cannot happen in a normal refresh; when it
    does, a loud FileNotFoundError beats an empty partition that reports success."""
    frame = pd.DataFrame({"pfr_player_id": ["MahoPa00"]})

    with pytest.raises(FileNotFoundError):
        resolve_gsis_via_id_map(frame, tmp_path)
