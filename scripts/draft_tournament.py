"""CLI wrapper for the draft strategy tournament. See projections.draft.assistant.tournament_cli."""

from __future__ import annotations

import sys

from projections.draft.assistant.tournament_cli import run

if __name__ == "__main__":
    sys.exit(run())
