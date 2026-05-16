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


def test_replacement_level_pinned_per_position() -> None:
    """For a known input, replacement_fpts(pos) is the projection of the boundary player."""
    # Oversized per-position inputs (10-15 each) make every position contain a
    # non-pool tail. Pin: replacement_fpts at each position must equal an actual
    # input projection (set membership), and every player projected above
    # replacement must have strictly positive VORP.
    cfg = _make_config()  # 4-team default
    rows: list[dict[str, object]] = []
    rows.extend(_bulk_rows(Position.QB, count=10, base_fpts=300.0))  # QBs 0..9 (300..291 fpts)
    rows.extend(_bulk_rows(Position.RB, count=15, base_fpts=280.0))  # RBs 0..14 (280..266 fpts)
    rows.extend(_bulk_rows(Position.WR, count=15, base_fpts=260.0))  # WRs 0..14 (260..246 fpts)
    rows.extend(_bulk_rows(Position.TE, count=10, base_fpts=200.0))  # TEs 0..9 (200..191 fpts)
    inputs = _make_season_projections(rows)
    out = generate_vorp_table(inputs, cfg)
    # Verify replacement is the best non-pool player at each position.
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        pos_rows = out[out["position"] == pos.value].sort_values(
            "season_mean_fpts", ascending=False
        )
        replacement_value = float(pos_rows["replacement_fpts"].iloc[0])
        # Replacement must equal one of the input projections at this position.
        pos_input_fpts = set(
            float(v) for v in inputs[inputs["position"] == pos.value]["season_mean"].tolist()
        )
        assert replacement_value in pos_input_fpts
        # Every player ranked above replacement has VORP > 0.
        above_replacement = pos_rows[pos_rows["season_mean_fpts"] > replacement_value]
        assert (above_replacement["vorp"] > 0).all()


def test_top_of_position_non_negative_vorp() -> None:
    cfg = _make_config()
    inputs = _bulk_input({Position.QB: 20, Position.RB: 20, Position.WR: 20, Position.TE: 20})
    out = generate_vorp_table(inputs, cfg)
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        wanted = cfg.roster_slots.get(getattr(RosterSlot, pos.name), 0)
        if wanted == 0:
            continue
        top_n = out[out["position"] == pos.value].nlargest(cfg.n_teams * wanted, "season_mean_fpts")
        assert (top_n["vorp"] >= 0).all()


def test_replacement_player_has_zero_vorp() -> None:
    """At least one player at each in-pool position has vorp == 0 (the replacement player)."""
    cfg = _make_config()
    inputs = _bulk_input({Position.QB: 20, Position.RB: 20, Position.WR: 20, Position.TE: 20})
    out = generate_vorp_table(inputs, cfg)
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        pos_rows = out[out["position"] == pos.value]
        assert (pos_rows["vorp"].abs() < 1e-9).any(), f"no zero-VORP row at {pos.value}"


def test_sub_replacement_players_have_negative_vorp() -> None:
    cfg = _make_config()
    inputs = _bulk_input({Position.QB: 20, Position.RB: 20, Position.WR: 20, Position.TE: 20})
    out = generate_vorp_table(inputs, cfg)
    # The very last player at each position (smallest season_mean_fpts) must have vorp <= 0.
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        pos_rows = out[out["position"] == pos.value].sort_values("season_mean_fpts")
        worst_vorp = float(pos_rows["vorp"].iloc[0])
        assert worst_vorp <= 0


def _replacement_by_position(out: pd.DataFrame) -> dict[str, float]:
    return {
        str(pos): float(out[out["position"] == pos]["replacement_fpts"].iloc[0])
        for pos in out["position"].unique()
    }


def test_flex_deepens_rb_wr_te_replacement() -> None:
    """A LeagueConfig with FLEX=1 should produce ≤ replacement_fpts at RB/WR/TE vs FLEX=0."""
    base_slots = {
        RosterSlot.QB: 1,
        RosterSlot.RB: 2,
        RosterSlot.WR: 2,
        RosterSlot.TE: 1,
        RosterSlot.BENCH: 1,
    }
    cfg_no_flex = _make_config(roster_slots=base_slots)
    cfg_with_flex = _make_config(roster_slots={**base_slots, RosterSlot.FLEX: 1})
    inputs = _bulk_input({Position.QB: 20, Position.RB: 30, Position.WR: 30, Position.TE: 20})

    out_no_flex = generate_vorp_table(inputs, cfg_no_flex)
    out_with_flex = generate_vorp_table(inputs, cfg_with_flex)

    repl_no_flex = _replacement_by_position(out_no_flex)
    repl_with_flex = _replacement_by_position(out_with_flex)

    for pos in (Position.RB.value, Position.WR.value, Position.TE.value):
        assert repl_with_flex[pos] <= repl_no_flex[pos], (
            f"{pos} replacement should deepen (lower) when FLEX is added: "
            f"no_flex={repl_no_flex[pos]} with_flex={repl_with_flex[pos]}"
        )


def test_super_flex_deepens_qb_replacement() -> None:
    base_slots = {
        RosterSlot.QB: 1,
        RosterSlot.RB: 2,
        RosterSlot.WR: 2,
        RosterSlot.TE: 1,
        RosterSlot.BENCH: 1,
    }
    cfg_no_sf = _make_config(roster_slots=base_slots)
    cfg_with_sf = _make_config(roster_slots={**base_slots, RosterSlot.SUPER_FLEX: 1})
    inputs = _bulk_input({Position.QB: 20, Position.RB: 30, Position.WR: 30, Position.TE: 20})

    out_no_sf = generate_vorp_table(inputs, cfg_no_sf)
    out_with_sf = generate_vorp_table(inputs, cfg_with_sf)

    repl_no_sf = _replacement_by_position(out_no_sf)
    repl_with_sf = _replacement_by_position(out_with_sf)

    assert repl_with_sf[Position.QB.value] <= repl_no_sf[Position.QB.value]
