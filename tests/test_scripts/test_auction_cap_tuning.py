from pathlib import Path

from scripts.auction_cap_tuning import _load_chunks, aggregate_chunks, grid

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
    assert rows[0].name == "robust"  # rows sorted by worst-case descending


def test_aggregate_seed_weights_by_n_seeds() -> None:
    # A 99-seed chunk dominates a 1-seed chunk for the same (market, model) — not an equal mean.
    chunks: list[dict[str, object]] = [
        {"market": "model", "n_seeds": 1, "base_seed": 0, "reg_win_pct": {"a": 0.0}},
        {"market": "model", "n_seeds": 99, "base_seed": 20, "reg_win_pct": {"a": 1.0}},
    ]
    _, rows, _ = aggregate_chunks(chunks)
    val = rows[0].cells[0]
    assert val is not None and abs(val - 0.99) < 1e-9  # (0*1 + 1*99)/100, weighted not 0.5


def test_aggregate_partial_coverage_not_chosen_best() -> None:
    # 'partial' scores high but only in one market; 'full' is scored in both -> 'full' wins best.
    chunks: list[dict[str, object]] = [
        {"market": "model", "base_seed": 0, "reg_win_pct": {"partial": 0.9, "full": 0.55}},
        {"market": "espn", "base_seed": 0, "reg_win_pct": {"full": 0.52}},
    ]
    _markets, rows, best = aggregate_chunks(chunks)
    assert best == "full"  # 'partial' has no espn cell -> ineligible despite the higher lone score
    partial_row = next(r for r in rows if r.name == "partial")
    assert partial_row.complete is False  # flagged incomplete coverage
    assert partial_row.cells == [None, 0.9]  # aligned to ['espn','model']: espn absent -> None


def test_aggregate_dedups_repeated_base_seed() -> None:
    # Two chunks with the same (market, base_seed) are one sample, not two.
    chunks: list[dict[str, object]] = [
        {"market": "model", "base_seed": 0, "n_seeds": 20, "reg_win_pct": {"a": 0.4}},
        {"market": "model", "base_seed": 0, "n_seeds": 20, "reg_win_pct": {"a": 0.4}},
        {"market": "model", "base_seed": 20, "n_seeds": 20, "reg_win_pct": {"a": 0.6}},
    ]
    _, rows, _ = aggregate_chunks(chunks)
    val = rows[0].cells[0]
    assert val is not None and abs(val - 0.5) < 1e-9  # dedup base_seed 0 -> mean(0.4, 0.6)


def test_aggregate_missing_base_seed_does_not_collide_with_index() -> None:
    # A chunk lacking base_seed must not dedup against a real base_seed equal to its list index:
    # the field-less chunk is at index 1 and another chunk has base_seed == 1; both must count.
    chunks: list[dict[str, object]] = [
        {"market": "model", "base_seed": 1, "n_seeds": 1, "reg_win_pct": {"a": 0.0}},
        {"market": "model", "n_seeds": 1, "reg_win_pct": {"a": 1.0}},  # no base_seed, list index 1
    ]
    _, rows, _ = aggregate_chunks(chunks)
    val = rows[0].cells[0]
    assert val is not None and abs(val - 0.5) < 1e-9  # both counted: mean(0.0, 1.0), no collision


def test_load_chunks_skips_unreadable_and_foreign(tmp_path: Path) -> None:
    # Unreadable JSON (same ValueError path as a byte-corrupt UnicodeDecodeError), a foreign JSON
    # array, and a dict missing 'market' are all skipped; only the real chunk loads.
    (tmp_path / "good.json").write_text('{"market": "model", "reg_win_pct": {"a": 0.5}}')
    (tmp_path / "bad.json").write_text("{not valid json")
    (tmp_path / "foreign.json").write_text("[1, 2, 3]")
    (tmp_path / "nomarket.json").write_text('{"reg_win_pct": {}}')
    chunks, skipped = _load_chunks(tmp_path)
    assert len(chunks) == 1 and chunks[0]["market"] == "model"
    assert skipped == 3
