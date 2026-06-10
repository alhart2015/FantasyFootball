"""CLI wrapper for the live draft assistant. See projections.draft.assistant.cli."""

from __future__ import annotations

import sys

from projections.draft.assistant.cli import run

if __name__ == "__main__":
    sys.exit(run())
