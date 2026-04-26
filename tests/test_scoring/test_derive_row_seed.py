"""derive_row_seed -- determinism, independence, range, cross-process stability."""

from __future__ import annotations

import os
import subprocess
import sys

from projections.scoring import derive_row_seed


def test_seed_is_deterministic_within_process() -> None:
    s1 = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    s2 = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert s1 == s2


def test_seed_fits_uint32() -> None:
    s = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert 0 <= s < 2**32


def test_seed_differs_when_gsis_id_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0035640", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert a != b


def test_seed_differs_when_season_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2023, week=4, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert a != b


def test_seed_differs_when_week_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2024, week=3, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    assert a != b


def test_seed_differs_when_ruleset_differs() -> None:
    a = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_PPR")
    b = derive_row_seed(gsis_id="00-0033873", season=2024, week=4, ruleset_name="ESPN_HALF")
    assert a != b


def _child_env(seed: str) -> dict[str, str]:
    """Build a minimal child env that still lets python launch on Windows."""
    e = {"PYTHONHASHSEED": seed}
    for var in ("PATH", "SYSTEMROOT", "PYTHONPATH"):
        if var in os.environ:
            e[var] = os.environ[var]
    return e


def test_seed_stable_across_processes_with_different_pythonhashseed() -> None:
    """Python's built-in hash() is salt-randomized via PYTHONHASHSEED;
    derive_row_seed uses sha256 instead and must be invariant."""
    code = (
        "from projections.scoring import derive_row_seed; "
        "print(derive_row_seed("
        "gsis_id='00-0033873', season=2024, week=4, ruleset_name='ESPN_PPR'))"
    )
    out_a = subprocess.run(
        [sys.executable, "-c", code],
        env=_child_env("0"),
        capture_output=True,
        text=True,
        check=True,
    )
    out_b = subprocess.run(
        [sys.executable, "-c", code],
        env=_child_env("12345"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_a.stdout.strip() == out_b.stdout.strip()
