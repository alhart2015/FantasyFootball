"""Unit tests for the chunked H2H backtest runner's crash/hang retry logic.

The runner orchestrates worker subprocesses; the dev box both *crashes* workers (non-zero
exit) and *hangs* them (worker stalls forever). `_run_chunk_with_retries` must recover from
BOTH by killing + retrying, bounded by `--chunk-timeout`. These tests mock `subprocess.run`
so no real worker is spawned. See docs/dev-box-stability.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import scripts.h2h_backtest_chunked as mod


def test_chunk_timeout_arg_parsed() -> None:
    assert mod._parse_args(["--league-config", "x.json"]).chunk_timeout is None
    parsed = mod._parse_args(["--league-config", "x.json", "--chunk-timeout", "600"])
    assert parsed.chunk_timeout == 600.0


def test_hung_chunk_is_killed_and_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # First attempt hangs (TimeoutExpired), second completes cleanly.
    seen_timeouts: list[float | None] = []

    def fake_run(
        cmd: list[str], env: dict[str, str] | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        seen_timeouts.append(timeout)
        if len(seen_timeouts) == 1:
            raise subprocess.TimeoutExpired(cmd, timeout or 0.0)
        return subprocess.CompletedProcess[bytes](cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(mod, "_valid_chunk_file", lambda out, expected: True)
    ok = mod._run_chunk_with_retries(
        ["worker"], {}, tmp_path / "c.json", 16, lo=0, hi=5, max_retries=5, timeout=600.0
    )
    assert ok is True
    assert seen_timeouts == [600.0, 600.0]  # bound passed through on every attempt


def test_persistent_hang_exhausts_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    attempts = 0

    def fake_run(
        cmd: list[str], env: dict[str, str] | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal attempts
        attempts += 1
        raise subprocess.TimeoutExpired(cmd, timeout if timeout is not None else 0.0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(mod, "_valid_chunk_file", lambda out, expected: True)
    ok = mod._run_chunk_with_retries(
        ["worker"], {}, tmp_path / "c.json", 16, lo=0, hi=5, max_retries=3, timeout=1.0
    )
    assert ok is False
    assert attempts == 3


def test_crash_then_success_still_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Existing behavior preserved: a non-zero exit (e.g. 0xC0000005) retries, no timeout set.
    rcs = iter([3221225477, 0])

    def fake_run(
        cmd: list[str], env: dict[str, str] | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess[bytes](cmd, next(rcs))

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(mod, "_valid_chunk_file", lambda out, expected: True)
    ok = mod._run_chunk_with_retries(
        ["worker"], {}, tmp_path / "c.json", 16, lo=0, hi=5, max_retries=5, timeout=None
    )
    assert ok is True
