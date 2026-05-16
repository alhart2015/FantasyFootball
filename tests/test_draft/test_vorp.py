"""Tests for `projections.draft.vorp.generate_vorp_table`."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest  # noqa: F401  # reused by later tasks (5-9) for raises/parametrize.

from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.schemas import (
    _PYARROW_STR,
    Position,
    ProjectionSeasonSchema,
    RosterSlot,
    Ruleset,
    VorpTableSchema,
)

_POSITION_ID_PREFIX: dict[Position, int] = {
    Position.QB: 1,
    Position.RB: 2,
    Position.WR: 3,
    Position.TE: 4,
    Position.K: 5,
    Position.DST: 6,
}


def _make_config(
    n_teams: int = 4,
    roster_slots: dict[RosterSlot, int] | None = None,
    ruleset: Ruleset | None = None,
) -> LeagueConfig:
    return LeagueConfig(
        name="test",
        n_teams=n_teams,
        budget=100,
        min_bid=1,
        roster_slots=roster_slots
        or {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=ruleset or Ruleset.espn_ppr(),
    )


def _bulk_rows(
    position: Position,
    count: int,
    base_fpts: float = 300.0,
    season: int = 2026,
    ruleset_name: str = "ESPN_PPR",
) -> list[dict[str, object]]:
    """Build `count` ProjectionSeasonSchema-shaped rows for `position`.

    `ruleset_name` defaults to `"ESPN_PPR"` so the synthetic input matches
    `Ruleset.espn_ppr().name` — every ingest path stores `ruleset.name`
    (uppercase) and `generate_vorp_table` compares the two strings exactly.
    Tests that exercise mismatch / mixed-ruleset paths pass other values
    (e.g. `"STANDARD"`) explicitly.
    """
    prefix = _POSITION_ID_PREFIX[position]
    return [
        {
            "gsis_id": f"00-{prefix}{i:06d}",
            "season": season,
            "position": position.value,
            "ruleset": ruleset_name,
            "n_weeks": 17,
            "season_mean": base_fpts - i,
            "season_p10": (base_fpts - i) * 0.7,
            "season_p50": (base_fpts - i) * 1.0,
            "season_p90": (base_fpts - i) * 1.3,
            "model_id": "test-model-v0",
            "generated_at": pd.Timestamp(datetime.now(UTC)).as_unit("us"),
        }
        for i in range(count)
    ]


def _make_season_projections(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["ruleset"] = df["ruleset"].astype(_PYARROW_STR)
    df["model_id"] = df["model_id"].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(df)


def _bulk_input(positions: dict[Position, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pos, count in positions.items():
        rows.extend(_bulk_rows(pos, count=count))
    return _make_season_projections(rows)


def test_output_validates_against_schema() -> None:
    cfg = _make_config()
    inputs = _bulk_input({Position.QB: 20, Position.RB: 20, Position.WR: 20, Position.TE: 20})
    out = generate_vorp_table(inputs, cfg)
    VorpTableSchema.validate(out)


def test_row_count_preserved_for_in_scope_positions() -> None:
    cfg = _make_config()
    inputs = _bulk_input({Position.QB: 20, Position.RB: 20, Position.WR: 20, Position.TE: 20})
    out = generate_vorp_table(inputs, cfg)
    assert len(out) == 80  # all positions are in cfg.roster_slots
    assert out["gsis_id"].nunique() == 80


def test_rename_invariant_season_mean_to_season_mean_fpts() -> None:
    cfg = _make_config()
    inputs = _bulk_input({Position.QB: 20, Position.RB: 20, Position.WR: 20, Position.TE: 20})
    out = generate_vorp_table(inputs, cfg)
    merged = out.merge(inputs[["gsis_id", "season_mean"]], on="gsis_id", suffixes=("", "_input"))
    pd.testing.assert_series_equal(
        merged["season_mean_fpts"].astype("float64"),
        merged["season_mean"].astype("float64"),
        check_names=False,
    )


def test_vorp_equation() -> None:
    cfg = _make_config()
    inputs = _bulk_input({Position.QB: 20, Position.RB: 20, Position.WR: 20, Position.TE: 20})
    out = generate_vorp_table(inputs, cfg)
    # vorp == season_mean_fpts - replacement_fpts exactly (no rounding).
    delta = out["season_mean_fpts"] - out["replacement_fpts"] - out["vorp"]
    assert (delta.abs() < 1e-9).all()
