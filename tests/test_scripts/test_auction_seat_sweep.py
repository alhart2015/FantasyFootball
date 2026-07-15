from pathlib import Path

from scripts.auction_seat_sweep import _load_chunks, aggregate_seat_sweep


def test_aggregate_seat_averages_and_picks_robust_worst_case() -> None:
    # Two markets, two seats each. 'robust' has the better worst-case seat-average across markets;
    # 'specialist' wins the model market but collapses in espn -> loses on worst-case.
    chunks: list[dict[str, object]] = [
        {"market": "model", "seat": 1, "reg_win_pct": {"robust": 0.50, "specialist": 0.60}},
        {"market": "model", "seat": 2, "reg_win_pct": {"robust": 0.52, "specialist": 0.62}},
        {"market": "espn", "seat": 1, "reg_win_pct": {"robust": 0.49, "specialist": 0.20}},
        {"market": "espn", "seat": 2, "reg_win_pct": {"robust": 0.51, "specialist": 0.22}},
    ]
    markets, seats, rows, best = aggregate_seat_sweep(chunks)
    assert markets == ["espn", "model"]
    assert seats == [1, 2]
    assert best == "robust"  # worst-case: robust ~0.50 vs specialist ~0.21
    assert rows[0].name == "robust"  # sorted by worst-case descending
    robust = rows[0]
    # seat_avg aligned to ['espn','model']: espn mean(0.49,0.51)=0.50, model mean(0.50,0.52)=0.51
    assert robust.seat_avg[0] is not None and abs(robust.seat_avg[0] - 0.50) < 1e-9
    assert robust.seat_avg[1] is not None and abs(robust.seat_avg[1] - 0.51) < 1e-9
    assert abs(robust.worst - 0.50) < 1e-9


def test_aggregate_seat_average_is_equal_weight_mean() -> None:
    # Three seats in one market: the seat-average is the plain mean, each seat weighted equally.
    chunks: list[dict[str, object]] = [
        {"market": "model", "seat": 1, "reg_win_pct": {"a": 0.30}},
        {"market": "model", "seat": 2, "reg_win_pct": {"a": 0.60}},
        {"market": "model", "seat": 3, "reg_win_pct": {"a": 0.90}},
    ]
    _markets, _seats, rows, _best = aggregate_seat_sweep(chunks)
    val = rows[0].seat_avg[0]
    assert val is not None and abs(val - 0.60) < 1e-9  # mean(0.3, 0.6, 0.9)


def test_aggregate_partial_coverage_not_chosen_best() -> None:
    # 'partial' is missing an espn cell -> ineligible for best despite a higher lone seat-average.
    chunks: list[dict[str, object]] = [
        {"market": "model", "seat": 1, "reg_win_pct": {"partial": 0.9, "full": 0.55}},
        {"market": "espn", "seat": 1, "reg_win_pct": {"full": 0.52}},
    ]
    markets, _seats, rows, best = aggregate_seat_sweep(chunks)
    assert markets == ["espn", "model"]
    assert best == "full"  # 'partial' has no espn cell -> not coverage-complete
    partial = next(r for r in rows if r.name == "partial")
    assert partial.complete is False
    assert partial.seat_avg == [None, 0.9]  # aligned to ['espn','model']: espn absent -> None


def test_aggregate_incomplete_when_a_seat_is_missing_in_one_market() -> None:
    # 'x' is scored at seats 1 and 2 in model but only seat 1 in espn -> not complete (a hole).
    chunks: list[dict[str, object]] = [
        {"market": "model", "seat": 1, "reg_win_pct": {"x": 0.5}},
        {"market": "model", "seat": 2, "reg_win_pct": {"x": 0.5}},
        {"market": "espn", "seat": 1, "reg_win_pct": {"x": 0.5}},
    ]
    _markets, seats, rows, best = aggregate_seat_sweep(chunks)
    assert seats == [1, 2]
    row = rows[0]
    assert row.complete is False  # espn seat 2 cell is missing
    assert row.per_seat["espn"] == [0.5, None]  # the hole is visible, not averaged away
    assert best == ""  # no coverage-complete model


def test_aggregate_dedups_repeated_seat_last_wins() -> None:
    # A re-run of the same (market, seat) supersedes the earlier value; it is not double-counted.
    chunks: list[dict[str, object]] = [
        {"market": "model", "seat": 1, "reg_win_pct": {"a": 0.10}},
        {"market": "model", "seat": 1, "reg_win_pct": {"a": 0.90}},  # corrected re-run of seat 1
        {"market": "model", "seat": 2, "reg_win_pct": {"a": 0.50}},
    ]
    _markets, _seats, rows, _best = aggregate_seat_sweep(chunks)
    val = rows[0].seat_avg[0]
    assert val is not None and abs(val - 0.70) < 1e-9  # mean(0.90, 0.50), the 0.10 superseded


def test_load_chunks_skips_unreadable_and_foreign(tmp_path: Path) -> None:
    # Corrupt JSON, a foreign array, and a dict missing 'seat' are all skipped; only the real chunk
    # loads. (Missing 'seat' would silently collapse the seat axis, so it must be rejected.)
    (tmp_path / "good.json").write_text('{"market": "model", "seat": 1, "reg_win_pct": {"a": 0.5}}')
    (tmp_path / "bad.json").write_text("{not valid json")
    (tmp_path / "foreign.json").write_text("[1, 2, 3]")
    (tmp_path / "noseat.json").write_text('{"market": "model", "reg_win_pct": {}}')
    chunks, skipped = _load_chunks(tmp_path)
    assert len(chunks) == 1 and chunks[0]["market"] == "model"
    assert skipped == 3
