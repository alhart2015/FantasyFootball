from __future__ import annotations

import logging

import pandas as pd
import pytest

from projections.ingest.identity import (
    drop_placeholder_gsis_rows,
    normalize_join_id,
    placeholder_name_key,
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
