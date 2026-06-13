"""CLI: H2H draft-strategy backtest. See src/projections/draft/backtest/cli.py."""

from __future__ import annotations

import sys

from projections.draft.backtest.cli import run

if __name__ == "__main__":
    sys.exit(run())
