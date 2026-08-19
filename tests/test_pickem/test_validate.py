"""Column-guard tests, including that the remedy it prints actually runs."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from projections.pickem._validate import _REFRESH_HINT, require_schedule_columns


def test_passes_when_every_column_is_present() -> None:
    require_schedule_columns(pd.DataFrame({"a": [1], "b": [2]}), ["a", "b"], needed_for="x")


def test_raises_naming_the_missing_columns_and_the_remedy() -> None:
    with pytest.raises(ValueError, match=r"missing \['b'\].*required for grading") as exc:
        require_schedule_columns(pd.DataFrame({"a": [1]}), ["a", "b"], needed_for="grading")
    assert "refresh_schedules" in str(exc.value)


def test_hint_command_is_shell_safe() -> None:
    """The hint is the whole remediation path, so it has to survive a paste.

    It wraps the program in double quotes; a nested `Path("data")` would close
    that string early and the shell would hand Python `Path(data)` -> NameError.
    """
    command = _REFRESH_HINT.split("python -c ", 1)[1].strip()
    assert command.startswith('"') and command.endswith('"')
    assert '"' not in command[1:-1], f"nested double quote breaks the paste: {command}"


def test_hint_command_actually_executes() -> None:
    """Run the exact program the hint prints, with the import swapped for a no-op
    so the test does no network I/O. A quoting regression fails here."""
    program = _REFRESH_HINT.split("python -c ", 1)[1].strip()[1:-1]
    program = program.replace(
        "from projections.ingest import refresh_schedules",
        "refresh_schedules = lambda *a, **k: None",
    )
    r = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert r.returncode == 0, f"hint command failed:\n{r.stderr}"
