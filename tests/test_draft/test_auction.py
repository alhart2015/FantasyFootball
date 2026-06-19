"""Unit tests for `projections.draft.auction.generate_auction_values` and helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft._pool import _select_pool
from projections.draft.auction import espn_anchored_bot_prices, generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import FLEX_ELIGIBLE as _FLEX_ELIGIBLE
from projections.schemas import (
    _PYARROW_STR,
    AuctionValuesSchema,
    Position,
    RosterSlot,
    Ruleset,
)


def _make_config(
    n_teams: int = 4,
    roster_slots: dict[RosterSlot, int] | None = None,
    budget: int = 100,
    min_bid: int = 1,
) -> LeagueConfig:
    return LeagueConfig(
        name="test",
        n_teams=n_teams,
        budget=budget,
        min_bid=min_bid,
        roster_slots=roster_slots
        or {
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 1,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _make_vorp_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a VORP-table-shaped DataFrame from a list of dicts.

    Required keys: gsis_id, position, season_mean_fpts, vorp.
    """
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["season_mean_fpts"] = df["season_mean_fpts"].astype("float64")
    df["vorp"] = df["vorp"].astype("float64")
    return df


# Map positions to a per-position digit so synthetic gsis_ids stay within the
# canonical `\d{2}-\d{7}` regex enforced by AuctionValuesSchema.
_POSITION_ID_PREFIX: dict[Position, int] = {
    Position.QB: 1,
    Position.RB: 2,
    Position.WR: 3,
    Position.TE: 4,
    Position.K: 5,
    Position.DST: 6,
}


def _bulk_position_rows(
    position: Position, count: int, base_fpts: float = 200.0
) -> list[dict[str, object]]:
    """Generate `count` rows for `position` with descending season_mean_fpts and matching VORP."""
    prefix = _POSITION_ID_PREFIX[position]
    out: list[dict[str, object]] = []
    for i in range(count):
        out.append(
            {
                "gsis_id": f"00-{prefix}{i:06d}",
                "position": position.value,
                "season_mean_fpts": base_fpts - i,
                "vorp": (base_fpts - i) - 100.0,
            }
        )
    return out


def test_select_pool_size_matches_total_pool_size() -> None:
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=20))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    assert len(pool_ids) == cfg.total_pool_size


def test_select_pool_respects_position_quotas() -> None:
    cfg = _make_config()
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=20))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    counts = pool_df["position"].value_counts().to_dict()
    # 4 teams x 1 QB = 4 QBs guaranteed; 4 x 2 = 8 RBs; 4 x 2 = 8 WRs; 4 x 1 = 4 TEs.
    # FLEX (4 slots) + BENCH (4 slots) fill from the best remaining.
    assert counts[Position.QB.value] >= 4
    assert counts[Position.RB.value] >= 8
    assert counts[Position.WR.value] >= 8
    assert counts[Position.TE.value] >= 4
    assert sum(counts.values()) == cfg.total_pool_size


def test_select_pool_omits_low_projection_players() -> None:
    """With only 12 RBs and a league that wants 12 in-pool RBs (8 strict + 4 FLEX-eligible),
    RB13+ should be out of pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.BENCH: 0,
        }
    )
    rows: list[dict[str, object]] = []
    rows.extend(_bulk_position_rows(Position.QB, count=10))
    rows.extend(_bulk_position_rows(Position.RB, count=20))
    rows.extend(_bulk_position_rows(Position.WR, count=20))
    rows.extend(_bulk_position_rows(Position.TE, count=10))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    rb_pool = pool_df[pool_df["position"] == Position.RB.value]
    # 4 teams * 2 RB starters = 8 strict RBs. FLEX may add more.
    assert len(rb_pool) >= 8
    # The omitted RBs are the lowest-projection ones
    all_rb_ids = set(df[df["position"] == Position.RB.value]["gsis_id"])
    omitted_rb_ids = all_rb_ids - set(rb_pool["gsis_id"])
    if omitted_rb_ids:
        min_in_pool_fpts = rb_pool["season_mean_fpts"].min()
        omitted_max_fpts = df[df["gsis_id"].isin(omitted_rb_ids)]["season_mean_fpts"].max()
        assert omitted_max_fpts <= min_in_pool_fpts


def test_select_pool_omits_position_not_in_roster_slots() -> None:
    """A config without K or DST should not consume K or DST players into the pool."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.BENCH: 2,
        }
    )
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST):
        rows.extend(_bulk_position_rows(pos, count=15))
    df = _make_vorp_table(rows)
    pool_ids = _select_pool(df, cfg)
    pool_df = df[df["gsis_id"].isin(pool_ids)]
    assert (pool_df["position"] != Position.K.value).all()
    assert (pool_df["position"] != Position.DST.value).all()


def test_select_pool_errors_on_missing_required_position() -> None:
    """If the config requires a position the VORP table doesn't cover, raise clearly."""
    cfg = _make_config(
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.K: 1,
            RosterSlot.BENCH: 0,
        }
    )
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        rows.extend(_bulk_position_rows(pos, count=10))
    df = _make_vorp_table(rows)
    with pytest.raises(ValueError, match=r"\bK\b"):
        _select_pool(df, cfg)


def _full_pool_vorp_table(cfg: LeagueConfig, extra_per_position: int = 5) -> pd.DataFrame:
    """Build a VORP table large enough to fill `cfg`'s pool plus a buffer of out-of-pool rows."""
    rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST):
        rostered = cfg.roster_slots.get(RosterSlot(pos.value), 0) > 0
        if rostered or pos in _FLEX_ELIGIBLE:
            rows.extend(_bulk_position_rows(pos, count=cfg.n_teams * 4 + extra_per_position))
    return _make_vorp_table(rows)


def test_sum_invariant_matches_total_budget() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    assert int(out["auction_dollars"].sum()) == cfg.total_budget


def test_min_bid_floor_for_in_pool_players() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]]
    assert (in_pool["auction_dollars"] >= cfg.min_bid).all()


def test_out_of_pool_players_get_zero_dollars() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    out_of_pool = out[~out["in_pool"]]
    assert (out_of_pool["auction_dollars"] == 0).all()
    assert out_of_pool["pool_rank"].isna().all()


def test_pool_size_exactly_total_pool_size() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    assert int(out["in_pool"].sum()) == cfg.total_pool_size


def test_negative_vorp_in_pool_gets_min_bid() -> None:
    """In-pool players with vorp <= 0 should get exactly min_bid (modulo drift adjustments
    that never reach this part of the curve in realistic test sizes)."""
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]]
    neg_vorp = in_pool[in_pool["vorp"] <= 0]
    if len(neg_vorp) > 0:
        # Drift adjustments only land on players with the largest fractional parts,
        # which are mid-pack high-VORP players, never the min-bid floor.
        assert (neg_vorp["auction_dollars"] == cfg.min_bid).all()


def test_vorp_scale_invariance() -> None:
    """Doubling all positive VORPs leaves auction_dollars unchanged (proportional allocation)."""
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out_a = generate_auction_values(df, cfg)
    df_scaled = df.copy()
    df_scaled["vorp"] = df_scaled["vorp"] * 2.0
    out_b = generate_auction_values(df_scaled, cfg)
    merged = out_a.merge(
        out_b[["gsis_id", "auction_dollars"]],
        on="gsis_id",
        suffixes=("_a", "_b"),
    )
    assert (merged["auction_dollars_a"] == merged["auction_dollars_b"]).all()


def test_higher_budget_scales_surplus() -> None:
    """With identical VORPs but budget=200 vs budget=100, in-pool players get
    approximately 2x the dollars (exactly 2x after subtracting min_bid)."""
    cfg_a = _make_config(budget=100)
    cfg_b = _make_config(budget=200)
    df = _full_pool_vorp_table(cfg_a)
    out_a = generate_auction_values(df, cfg_a)
    out_b = generate_auction_values(df, cfg_b)
    merged = out_a[out_a["in_pool"]].merge(
        out_b[["gsis_id", "auction_dollars"]],
        on="gsis_id",
        suffixes=("_a", "_b"),
    )
    # Compare expected and actual extras above min_bid
    expected_b = 2 * (merged["auction_dollars_a"] - cfg_a.min_bid) + cfg_b.min_bid
    # Allow +/- 2 for rounding-drift redistribution: each run rounds independently
    # (up to +/- 1 from the float allocation) and the drift-correction step adds
    # up to another +/- 1, so the two runs can diverge by 2 units even though the
    # underlying float allocations are exactly proportional.
    diff = (merged["auction_dollars_b"] - expected_b).abs()
    assert (diff <= 2).all()


def test_pool_rank_is_dense_and_ordered() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]].sort_values("pool_rank")
    assert in_pool["pool_rank"].tolist() == list(range(1, cfg.total_pool_size + 1))
    # auction_dollars is non-increasing with pool_rank
    dollars = in_pool["auction_dollars"].tolist()
    assert dollars == sorted(dollars, reverse=True)


def test_output_validates_against_schema() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg)
    AuctionValuesSchema.validate(out)


def test_degenerate_zero_positive_vorp_distributes_uniformly() -> None:
    """If every in-pool player has vorp <= 0, distribute surplus uniformly."""
    cfg = _make_config(
        n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.BENCH: 0}
    )
    # 4 players total; all VORP = 0
    rows = [
        {"gsis_id": "00-1000001", "position": "QB", "season_mean_fpts": 200.0, "vorp": 0.0},
        {"gsis_id": "00-1000002", "position": "QB", "season_mean_fpts": 190.0, "vorp": 0.0},
        {"gsis_id": "00-2000001", "position": "RB", "season_mean_fpts": 180.0, "vorp": 0.0},
        {"gsis_id": "00-2000002", "position": "RB", "season_mean_fpts": 170.0, "vorp": 0.0},
    ]
    df = _make_vorp_table(rows)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]]
    # total_budget = 200, total_pool_size = 4 -> $50 each (or +/-1 if drift redistributes).
    assert sorted(in_pool["auction_dollars"].tolist()) in (
        [50, 50, 50, 50],
        [49, 50, 50, 51],
    )
    assert int(in_pool["auction_dollars"].sum()) == cfg.total_budget


def test_duplicate_gsis_id_rejected() -> None:
    cfg = _make_config(n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 0})
    rows = [
        {"gsis_id": "00-1000001", "position": "QB", "season_mean_fpts": 200.0, "vorp": 50.0},
        {"gsis_id": "00-1000001", "position": "QB", "season_mean_fpts": 190.0, "vorp": 40.0},
    ]
    df = _make_vorp_table(rows)
    with pytest.raises(ValueError, match="duplicate"):
        generate_auction_values(df, cfg)


def test_min_bid_floor_preserved_under_negative_drift() -> None:
    """Regression: drift-correction step must NOT push in-pool players below min_bid.

    Setup: tiny pool where naive rounding produces negative drift AND the
    smallest-fractional candidates are already at the min_bid floor. Without
    the floor-protection in step 4, one player ends up at $0.
    """
    cfg = LeagueConfig(
        name="tiny_drift_case",
        n_teams=5,
        budget=2,
        min_bid=1,
        roster_slots={RosterSlot.QB: 1},
        ruleset=Ruleset.standard(),
    )
    # 5 QBs, vorps [1, 1, 1, 0, 0] → extras [1.667, 1.667, 1.667, 0, 0]
    # → _dollars_float [2.667, 2.667, 2.667, 1.0, 1.0]
    # → round [3, 3, 3, 1, 1], sum=11, drift=-1
    # Without the fix, one of the floor players gets adjusted to 0.
    rows = [
        {"gsis_id": "00-1000001", "position": "QB", "season_mean_fpts": 200.0, "vorp": 1.0},
        {"gsis_id": "00-1000002", "position": "QB", "season_mean_fpts": 190.0, "vorp": 1.0},
        {"gsis_id": "00-1000003", "position": "QB", "season_mean_fpts": 180.0, "vorp": 1.0},
        {"gsis_id": "00-1000004", "position": "QB", "season_mean_fpts": 170.0, "vorp": 0.0},
        {"gsis_id": "00-1000005", "position": "QB", "season_mean_fpts": 160.0, "vorp": 0.0},
    ]
    df = _make_vorp_table(rows)
    out = generate_auction_values(df, cfg)
    in_pool = out[out["in_pool"]]
    # Every in-pool player must be at least min_bid.
    assert (in_pool["auction_dollars"] >= cfg.min_bid).all(), in_pool[
        ["gsis_id", "auction_dollars"]
    ].to_string()
    # And the sum invariant still holds.
    assert int(in_pool["auction_dollars"].sum()) == cfg.total_budget


def test_reference_prices_pass_through_matched_rows() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    # Build a partial reference table covering only the first two players.
    first_two = df["gsis_id"].iloc[:2].tolist()
    ref = pd.DataFrame(
        {
            "gsis_id": pd.array(first_two, dtype=_PYARROW_STR),
            "reference_dollars": pd.array([45, 30], dtype=pd.Int64Dtype()),
        }
    )
    out = generate_auction_values(df, cfg, reference_prices=ref)
    matched = out[out["gsis_id"].isin(first_two)].sort_values("gsis_id")
    expected_ref = ref.sort_values("gsis_id")
    assert matched["reference_dollars"].tolist() == expected_ref["reference_dollars"].tolist()
    # value_delta = auction_dollars - reference_dollars on matched rows
    deltas = (matched["auction_dollars"] - expected_ref["reference_dollars"].values).tolist()
    assert matched["value_delta"].tolist() == deltas


def test_reference_prices_unmatched_rows_get_na() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    ref = pd.DataFrame(
        {
            "gsis_id": pd.array([df["gsis_id"].iloc[0]], dtype=_PYARROW_STR),
            "reference_dollars": pd.array([45], dtype=pd.Int64Dtype()),
        }
    )
    out = generate_auction_values(df, cfg, reference_prices=ref)
    unmatched = out[out["gsis_id"] != df["gsis_id"].iloc[0]]
    assert unmatched["reference_dollars"].isna().all()
    assert unmatched["value_delta"].isna().all()


def test_no_reference_prices_columns_all_na() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    out = generate_auction_values(df, cfg, reference_prices=None)
    assert "reference_dollars" in out.columns
    assert "value_delta" in out.columns
    assert out["reference_dollars"].isna().all()
    assert out["value_delta"].isna().all()


def test_reference_prices_duplicate_gsis_id_rejected() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    ref = pd.DataFrame(
        {
            "gsis_id": pd.array(
                [df["gsis_id"].iloc[0], df["gsis_id"].iloc[0]],
                dtype=_PYARROW_STR,
            ),
            "reference_dollars": pd.array([45, 50], dtype=pd.Int64Dtype()),
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        generate_auction_values(df, cfg, reference_prices=ref)


def _hand_pool_with_espn() -> tuple[LeagueConfig, pd.DataFrame]:
    """4-player pool (2 QB + 2 RB). total_budget = n_teams*budget = 2*100 = 200, min_bid 1 ->
    surplus 196. Two priced players (espn 60, 36) absorb the surplus; two unpriced (NA) park at
    min_bid. Drift is 0:
      value_signal=[60,0,36,0]; sum=96; extra=[60,0,36,0]/96*196=[122.5,0,73.5,0];
      dollars=round([123.5,1,74.5,1])=[124,1,74,1] (round-half-even); sum=200.
    """
    cfg = _make_config(
        n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.BENCH: 0}
    )
    df = _make_vorp_table(
        [
            {"gsis_id": "00-1000001", "position": "QB", "season_mean_fpts": 200.0, "vorp": 50.0},
            {"gsis_id": "00-1000002", "position": "QB", "season_mean_fpts": 190.0, "vorp": 40.0},
            {"gsis_id": "00-2000001", "position": "RB", "season_mean_fpts": 180.0, "vorp": 30.0},
            {"gsis_id": "00-2000002", "position": "RB", "season_mean_fpts": 170.0, "vorp": 20.0},
        ]
    )
    df["espn_auction_dollars"] = pd.array([60, pd.NA, 36, pd.NA], dtype=pd.Int64Dtype())
    return cfg, df


def test_espn_bot_prices_sum_to_total_budget() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert int(out.sum()) == cfg.total_budget


def test_espn_bot_prices_unpriced_park_at_min_bid() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert out["00-1000002"] == cfg.min_bid
    assert out["00-2000002"] == cfg.min_bid


def test_espn_bot_prices_priced_split_surplus_and_are_monotonic() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert out["00-1000001"] == 124  # round(min_bid + (60/96)*196)
    assert out["00-2000001"] == 74  # round(min_bid + (36/96)*196)
    assert out["00-1000001"] > out["00-2000001"]  # higher ESPN $ -> higher bot $


def test_espn_bot_prices_dtype_is_int64() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert out.dtype == pd.Int64Dtype()


def test_espn_bot_prices_out_of_pool_get_zero() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)  # has extra out-of-pool rows per position
    df["espn_auction_dollars"] = pd.array([10] * len(df), dtype=pd.Int64Dtype())
    pool_ids = set(_select_pool(df, cfg))
    out = espn_anchored_bot_prices(df, cfg)
    out_of_pool = [g for g in df["gsis_id"] if g not in pool_ids]
    assert all(out[g] == 0 for g in out_of_pool)


def test_espn_bot_prices_every_in_pool_at_least_min_bid() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    df["espn_auction_dollars"] = pd.array(
        [50 if i < 5 else pd.NA for i in range(len(df))], dtype=pd.Int64Dtype()
    )
    pool_ids = set(_select_pool(df, cfg))
    out = espn_anchored_bot_prices(df, cfg)
    assert all(out[g] >= cfg.min_bid for g in pool_ids)


def test_espn_bot_prices_absent_column_uniform_fallback() -> None:
    cfg, df = _hand_pool_with_espn()
    df = df.drop(columns=["espn_auction_dollars"])
    out = espn_anchored_bot_prices(df, cfg)
    # all-zero weight -> uniform split of total_budget (200) over 4 in-pool players:
    # surplus 196 / 4 = 49 + min_bid 1 = 50 each; drift 0
    assert int(out.sum()) == cfg.total_budget
    in_pool = out[out > 0] if (out > 0).any() else out
    assert sorted(in_pool.tolist()) in ([50, 50, 50, 50], [49, 50, 50, 51])


def test_espn_bot_prices_deep_league_inflation() -> None:
    """One priced player among many unpriced absorbs nearly the whole surplus -> bot $ >> ESPN $."""
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    espn = [pd.NA] * len(df)
    espn[0] = 5
    df["espn_auction_dollars"] = pd.array(espn, dtype=pd.Int64Dtype())
    priced_gsis = df["gsis_id"].iloc[0]
    out = espn_anchored_bot_prices(df, cfg)
    assert out[priced_gsis] > 5
