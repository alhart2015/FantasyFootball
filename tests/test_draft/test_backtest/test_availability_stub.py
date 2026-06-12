"""Minimal PlayerAvailability for tests: every pool player ~95% available, no byes."""

from __future__ import annotations

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability


def stub_availability(pool: pd.DataFrame) -> PlayerAvailability:
    return PlayerAvailability(p={str(g): 0.95 for g in pool["gsis_id"]}, bye={})
