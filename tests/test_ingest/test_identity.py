from __future__ import annotations

import pandas as pd

from projections.ingest.identity import normalize_join_id, placeholder_name_key


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
