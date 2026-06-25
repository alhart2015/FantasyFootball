"""CLI smoke tests — the backtest + refresh scripts expose the passthrough
flags that #55 (RB feature-signal probe) needs to A/B two feature caches."""

from __future__ import annotations

import subprocess
import sys


def test_backtest_help_exposes_features_root_and_position() -> None:
    out = subprocess.run(
        [sys.executable, "scripts/backtest.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--features-root" in out
    assert "--position" in out


def test_refresh_features_help_exposes_features_root() -> None:
    out = subprocess.run(
        [sys.executable, "scripts/refresh_features.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--features-root" in out
