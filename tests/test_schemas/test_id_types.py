"""ID NewType tests — runtime they're str; mypy treats them as distinct."""

from __future__ import annotations

import re

from projections.schemas import EspnId, GsisId, PfrId, SleeperId, validate_gsis_id


def test_gsis_id_is_str_at_runtime() -> None:
    pid = GsisId("00-0036322")
    assert isinstance(pid, str)


def test_validate_gsis_id_format() -> None:
    pid = validate_gsis_id("00-0036322")
    assert pid == GsisId("00-0036322")


def test_validate_gsis_id_rejects_bad_format() -> None:
    import pytest

    for bad in ["", "not-an-id", "0036322", "00-12345", "00-0036322a"]:
        with pytest.raises(ValueError):
            validate_gsis_id(bad)


def test_id_types_are_string_at_runtime_only() -> None:
    # NewType wrappers are noops at runtime — used only by mypy.
    assert EspnId("12345") == "12345"
    assert SleeperId("4046") == "4046"
    assert PfrId("JeffJu00") == "JeffJu00"


def test_gsis_id_pattern_matches_pattern_constant() -> None:
    from projections.schemas import GSIS_ID_PATTERN

    assert re.fullmatch(GSIS_ID_PATTERN, "00-0036322")
