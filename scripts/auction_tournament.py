"""CLI wrapper for the auction bid-model tournament. See auction.tournament_cli."""

from __future__ import annotations

import sys

from projections.draft.assistant.auction.tournament_cli import run

if __name__ == "__main__":
    sys.exit(run())
