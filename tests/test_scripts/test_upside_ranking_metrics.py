"""Unit tests for diagnose_upside_ranking helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from diagnose_upside_ranking import _compute_elite_thresholds

from projections.schemas import Position, Ruleset

_RULESET = Ruleset.espn_ppr()


def test_compute_elite_thresholds_returns_one_per_position(tmp_path: Path) -> None:
    """Synthetic 2 seasons x 4 positions x 10 players: threshold = mean over
    seasons of the 5th-highest actual total at each position (>= 8 games filter)."""
    raw_root = tmp_path / "data" / "raw"
    weekly_stats_root = raw_root / "weekly_stats"
    for season in (2019, 2020):
        partition = weekly_stats_root / f"season={season}"
        partition.mkdir(parents=True, exist_ok=True)
        rows = []
        for pos_idx, pos in enumerate(("QB", "RB", "WR", "TE")):
            for player_idx in range(10):
                target_ppr = 100.0 + pos_idx * 50 + player_idx * 20  # 100..390 per pos
                for week in range(1, 11):
                    rows.append(
                        {
                            "gsis_id": f"00-{pos}-{player_idx:04d}",
                            "season": season,
                            "week": week,
                            "position": pos,
                            "passing_yards": 0.0 if pos != "QB" else target_ppr / 10 * 25,
                            "passing_tds": 0,
                            "interceptions": 0,
                            "rushing_yards": (
                                0.0 if pos in ("QB", "WR", "TE") else target_ppr / 10 * 10
                            ),
                            "rushing_tds": 0,
                            "receptions": 0 if pos in ("QB", "RB") else int(target_ppr / 10),
                            "receiving_yards": 0.0,
                            "receiving_tds": 0,
                            "fumbles_lost": 0,
                        }
                    )
        pd.DataFrame(rows).to_parquet(partition / "part.parquet", index=False)

    thresholds = _compute_elite_thresholds(
        raw_root=raw_root,
        seasons=(2019, 2020),
        ruleset=_RULESET,
        min_games=8,
    )
    assert set(thresholds.keys()) == {Position.QB, Position.RB, Position.WR, Position.TE}
    for _pos, v in thresholds.items():
        assert isinstance(v, float)
        assert v > 0
    # 10 players per position; thresholds should differ across positions because
    # per-position scoring math produces different per-player totals.
    assert len(set(thresholds.values())) == 4


def test_top_k_overlap_perfect_match() -> None:
    from diagnose_upside_ranking import top_k_overlap

    pred_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    actual_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    assert top_k_overlap(pred_rank, actual_rank, k=3) == pytest.approx(1.0)
    assert top_k_overlap(pred_rank, actual_rank, k=5) == pytest.approx(1.0)


def test_top_k_overlap_partial_match() -> None:
    from diagnose_upside_ranking import top_k_overlap

    pred_rank = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    actual_rank = pd.Series([3, 1, 2, 4, 5], index=["a", "b", "c", "d", "e"])
    # pred top-3 = {a, b, c}, actual top-3 = {b, c, a} -> overlap 3/3 = 1.0
    assert top_k_overlap(pred_rank, actual_rank, k=3) == pytest.approx(1.0)
    # pred top-1 = {a}, actual top-1 = {b} -> overlap 0/1 = 0
    assert top_k_overlap(pred_rank, actual_rank, k=1) == pytest.approx(0.0)


def test_top5_rank_err_median_abs() -> None:
    from diagnose_upside_ranking import top5_rank_err

    # actual top-5 = {a, b, c, d, e} with ranks 1..5.
    # pred ranks for {a, b, c, d, e} = [1, 4, 3, 2, 5] -> errors [0, 2, 0, 2, 0]
    # median = 0.0
    pred_rank = pd.Series([1, 4, 3, 2, 5, 6], index=["a", "b", "c", "d", "e", "f"])
    actual_rank = pd.Series([1, 2, 3, 4, 5, 6], index=["a", "b", "c", "d", "e", "f"])
    assert top5_rank_err(pred_rank, actual_rank) == pytest.approx(0.0)


def test_kendall_tau_filtered_excludes_low_nweeks() -> None:
    from diagnose_upside_ranking import kendall_tau_filtered

    pred = pd.Series([100.0, 90.0, 80.0, 70.0], index=["a", "b", "c", "d"])
    actual = pd.Series([100.0, 90.0, 80.0, 70.0], index=["a", "b", "c", "d"])
    n_weeks = pd.Series([10, 10, 10, 3], index=["a", "b", "c", "d"])
    tau, n = kendall_tau_filtered(pred, actual, n_weeks, min_n_weeks=6)
    # 'd' is excluded; perfect rank agreement on the remaining 3 -> tau = 1.0, n = 3
    assert tau == pytest.approx(1.0)
    assert n == 3


def test_cell_verdict_signal() -> None:
    from diagnose_upside_ranking import cell_verdict

    # Metric beats mean by >= 1/12 on top-12 overlap AND beats mean on top-5 rank-err.
    verdict = cell_verdict(
        metric_top12=0.92,
        mean_top12=0.83,  # delta = 0.09 >= 1/12 ~ 0.083
        metric_rank_err=0.5,
        mean_rank_err=1.5,
    )
    assert verdict == "SIGNAL"


def test_cell_verdict_marginal_one_dim() -> None:
    from diagnose_upside_ranking import cell_verdict

    verdict = cell_verdict(
        metric_top12=0.92,
        mean_top12=0.83,
        metric_rank_err=1.5,
        mean_rank_err=1.5,  # tie -> not "better" on this dim
    )
    assert verdict == "MARGINAL"


def test_cell_verdict_null() -> None:
    from diagnose_upside_ranking import cell_verdict

    verdict = cell_verdict(
        metric_top12=0.83,
        mean_top12=0.83,
        metric_rank_err=1.5,
        mean_rank_err=1.5,
    )
    assert verdict == "NULL"


def test_cell_verdict_regression() -> None:
    from diagnose_upside_ranking import cell_verdict

    verdict = cell_verdict(
        metric_top12=0.50,
        mean_top12=0.83,
        metric_rank_err=3.0,
        mean_rank_err=1.5,
    )
    assert verdict == "REGRESSION"


def _verdict_frame(rows: list[tuple[int, str, str, str]]) -> pd.DataFrame:
    """Helper: rows are (season, position, metric, cell_verdict)."""
    return pd.DataFrame(rows, columns=["season", "position", "metric", "cell_verdict"])


def test_decision_greenlight_when_metric_signal_3plus_positions_both_years() -> None:
    from diagnose_upside_ranking import decision_gate

    verdicts = _verdict_frame(
        [
            (2024, "QB", "p90", "SIGNAL"),
            (2024, "RB", "p90", "SIGNAL"),
            (2024, "WR", "p90", "SIGNAL"),
            (2024, "TE", "p90", "NULL"),
            (2025, "QB", "p90", "SIGNAL"),
            (2025, "RB", "p90", "SIGNAL"),
            (2025, "WR", "p90", "SIGNAL"),
            (2025, "TE", "p90", "NULL"),
        ]
    )
    assert decision_gate(verdicts) == "Greenlight"


def test_decision_marginal_when_signal_only_one_year() -> None:
    from diagnose_upside_ranking import decision_gate

    verdicts = _verdict_frame(
        [
            (2024, "QB", "p90", "SIGNAL"),
            (2024, "RB", "p90", "SIGNAL"),
            (2024, "WR", "p90", "SIGNAL"),
            (2024, "TE", "p90", "NULL"),
            (2025, "QB", "p90", "MARGINAL"),
            (2025, "RB", "p90", "NULL"),
            (2025, "WR", "p90", "MARGINAL"),
            (2025, "TE", "p90", "NULL"),
        ]
    )
    assert decision_gate(verdicts) == "Marginal"


def test_decision_marginal_when_signal_or_marginal_3plus_both_years() -> None:
    from diagnose_upside_ranking import decision_gate

    verdicts = _verdict_frame(
        [
            (2024, "QB", "blend_70_30", "SIGNAL"),
            (2024, "RB", "blend_70_30", "MARGINAL"),
            (2024, "WR", "blend_70_30", "MARGINAL"),
            (2024, "TE", "blend_70_30", "NULL"),
            (2025, "QB", "blend_70_30", "MARGINAL"),
            (2025, "RB", "blend_70_30", "MARGINAL"),
            (2025, "WR", "blend_70_30", "SIGNAL"),
            (2025, "TE", "blend_70_30", "NULL"),
        ]
    )
    assert decision_gate(verdicts) == "Marginal"


def test_decision_no_greenlight_when_all_null() -> None:
    from diagnose_upside_ranking import decision_gate

    verdicts = _verdict_frame(
        [
            (yr, pos, metric, "NULL")
            for yr in (2024, 2025)
            for pos in ("QB", "RB", "WR", "TE")
            for metric in ("p90", "blend_70_30", "p_elite")
        ]
    )
    assert decision_gate(verdicts) == "No greenlight"


def _synthetic_weekly_with_three_players(season: int) -> pd.DataFrame:
    """Three QBs x 17 weeks, real ProjectionWeeklySchema-valid frame."""
    from tests.test_aggregation.test_season import _build_weekly_row, _to_weekly_frame

    rows = []
    for gsis_id, base_yards in [
        ("00-0033873", 280.0),
        ("00-0033874", 250.0),
        ("00-0033875", 220.0),
    ]:
        for week in range(1, 18):
            rows.append(
                _build_weekly_row(
                    gsis_id=gsis_id,
                    season=season,
                    week=week,
                    position="QB",
                    rec_yards_mean=base_yards,
                    rec_yards_std=60.0,
                )
            )
    return _to_weekly_frame(rows)


def _synthetic_distributions_csv(weekly: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    from projections.aggregation import aggregate_to_season

    summary = aggregate_to_season(weekly, ruleset=ruleset, n_samples=1000)
    summary["full_name"] = summary["gsis_id"].map(
        {"00-0033873": "Alpha", "00-0033874": "Beta", "00-0033875": "Gamma"}
    )
    summary["team"] = "TST"
    return summary


def test_assemble_season_diagnostic_returns_per_player_and_summary(tmp_path: Path) -> None:
    from diagnose_upside_ranking import assemble_season_diagnostic

    weekly = _synthetic_weekly_with_three_players(season=2024)
    dist = _synthetic_distributions_csv(weekly, _RULESET)

    actuals = pd.DataFrame(
        {
            "gsis_id": ["00-0033873", "00-0033874", "00-0033875"],
            "position": ["QB", "QB", "QB"],
            "actual_total": [300.0, 250.0, 200.0],  # matches predicted order
            "actual_n_weeks": [17, 17, 17],
        }
    )
    thresholds = {Position.QB: 290.0, Position.RB: 290.0, Position.WR: 290.0, Position.TE: 290.0}

    per_player, summary = assemble_season_diagnostic(
        weekly=weekly,
        distributions=dist,
        actuals=actuals,
        elite_thresholds=thresholds,
        ruleset=_RULESET,
        n_samples=1000,
    )
    assert set(per_player.columns) >= {
        "gsis_id",
        "position",
        "full_name",
        "actual_total",
        "actual_rank",
        "mean",
        "p90",
        "blend_70_30",
        "p_elite",
        "rank_mean",
        "rank_p90",
        "rank_blend_70_30",
        "rank_p_elite",
    }
    assert set(summary.columns) >= {
        "position",
        "metric",
        "top5_overlap",
        "top12_overlap",
        "top24_overlap",
        "top5_rank_err",
        "kendall_tau",
        "cell_verdict",
    }
    # The mean metric should perfectly recover the order in this synthetic setup.
    qb_mean = summary[(summary["position"] == "QB") & (summary["metric"] == "mean")].iloc[0]
    assert qb_mean["top5_overlap"] == pytest.approx(1.0)
