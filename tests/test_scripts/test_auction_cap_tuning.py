from scripts.auction_cap_tuning import aggregate_chunks, grid

from projections.draft.assistant.auction.bid_strategy import BalancedValueBid


def test_grid_has_flat_variants_plus_controls() -> None:
    g = grid()
    # 4 paces x 3 premiums flat variants + balanced control + patient_deep reference = 14
    assert len(g) == 14
    flat = [k for k in g if k.startswith("flat_")]
    assert len(flat) == 12
    for k in flat:
        strat = g[k]
        assert isinstance(strat, BalancedValueBid) and strat.non_increasing_cap is True
    control = g["balanced"]
    assert isinstance(control, BalancedValueBid) and control.non_increasing_cap is False
    assert "patient_deep" in g


def test_aggregate_picks_best_worst_case_across_markets() -> None:
    # Two markets, two chunks each. 'robust' wins the worst-case; 'specialist' wins one market only.
    chunks: list[dict[str, object]] = [
        {"market": "model", "reg_win_pct": {"robust": 0.50, "specialist": 0.60}},
        {"market": "model", "reg_win_pct": {"robust": 0.52, "specialist": 0.62}},
        {"market": "espn", "reg_win_pct": {"robust": 0.49, "specialist": 0.20}},
        {"market": "espn", "reg_win_pct": {"robust": 0.51, "specialist": 0.22}},
    ]
    markets, rows, best = aggregate_chunks(chunks)
    assert markets == ["espn", "model"]
    assert best == "robust"  # worst-case: robust ~0.50 vs specialist ~0.21
    assert rows[0][0] == "robust"  # rows sorted by worst-case descending
