"""Pick'em Hub — straight-up NFL picks under a minimum-underdogs constraint.

Design: `docs/superpowers/specs/2026-08-16-pickem-hub-design.md`.

The invariant that governs this whole sub-package: the **organizer's sheet**
decides who counts as the underdog, and the **consensus market** decides who is
likely to win. Two sources, two jobs. Conflating them is the most likely way to
produce picks that look plausible and are quietly wrong.

Weekly flow:

    sheet.write_template   ->  (organizer's spreads typed in)  ->  sheet.read_sheet
    slate.build_slate      ->  optimize.choose_picks           ->  grade.grade_picks

Driven from the command line by `scripts/pickem_board.py`.
"""

from __future__ import annotations

from projections.pickem._validate import require_schedule_columns
from projections.pickem.backtest import (
    baseline_week_scores,
    calibration_table,
    playable_games,
    summarize_baseline,
)
from projections.pickem.grade import grade_picks, record
from projections.pickem.optimize import DEFAULT_MIN_DOGS, choose_picks, expected_correct
from projections.pickem.probability import (
    add_win_probs,
    american_to_implied,
    devig_pair,
)
from projections.pickem.sheet import read_sheet, write_template
from projections.pickem.slate import build_slate
from projections.pickem.store import (
    read_picks,
    read_picks_season,
    read_sheet_partition,
    write_picks,
    write_sheet,
)

__all__ = [
    "DEFAULT_MIN_DOGS",
    "add_win_probs",
    "american_to_implied",
    "baseline_week_scores",
    "build_slate",
    "calibration_table",
    "choose_picks",
    "devig_pair",
    "expected_correct",
    "grade_picks",
    "playable_games",
    "read_picks",
    "read_picks_season",
    "read_sheet",
    "read_sheet_partition",
    "record",
    "require_schedule_columns",
    "summarize_baseline",
    "write_picks",
    "write_sheet",
    "write_template",
]
