"""Phase 1 diagnostic for TODO #33d. Reads weekly-distribution parquet +
distributions CSV from project_season.py output + actuals from data/raw/weekly_stats,
computes ranking under four metrics (mean / p90 / blend_70_30 / p_elite), and
writes a markdown report with a Phase-2-decision verdict.

See docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from projections.aggregation import aggregate_to_season
from projections.schemas import Position, Ruleset
from projections.scoring import actual_season_total
from projections.store import read_partition

_METRIC_NAMES = ("mean", "p90", "blend_70_30", "p_elite")


def _compute_elite_thresholds(
    *,
    raw_root: Path,
    seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
    ruleset: Ruleset,
    min_games: int = 8,
) -> dict[Position, float]:
    """Per-position elite threshold = mean over `seasons` of the 5th-highest
    actual season fantasy points at that position, computed over players with
    >= min_games games played that season."""
    per_season_top5: dict[Position, list[float]] = {
        p: [] for p in (Position.QB, Position.RB, Position.WR, Position.TE)
    }
    for season in seasons:
        ws = read_partition(raw_root, "weekly_stats", season=season)
        actuals = actual_season_total(ws, ruleset)
        actuals = actuals[actuals["actual_n_weeks"] >= min_games]
        for pos in per_season_top5:
            pos_rows = actuals[actuals["position"] == pos.value].sort_values(
                "actual_total", ascending=False
            )
            if len(pos_rows) >= 5:
                per_season_top5[pos].append(float(pos_rows["actual_total"].iloc[4]))
    out: dict[Position, float] = {}
    for pos, vals in per_season_top5.items():
        if not vals:
            raise ValueError(
                f"No seasons in {seasons} produced >= 5 players with "
                f">= {min_games} games at {pos.value}"
            )
        out[pos] = sum(vals) / len(vals)
    return out


def top_k_overlap(pred_rank: pd.Series, actual_rank: pd.Series, *, k: int) -> float:
    """|predicted_top_k ∩ actual_top_k| / min(k, n). Ranks are 1-based, smallest = best.

    The denominator is bounded by the number of available rows so that small
    populations (e.g. only 3 players at a position in a synthetic test) don't
    cap the metric below 1.0 even under perfect rank agreement.
    """
    effective_k = min(k, len(actual_rank))
    if effective_k == 0:
        return 0.0
    pred_top = set(pred_rank.nsmallest(k).index)
    actual_top = set(actual_rank.nsmallest(k).index)
    return len(pred_top & actual_top) / effective_k


def top5_rank_err(pred_rank: pd.Series, actual_rank: pd.Series) -> float:
    """For each player in actual top-5: median(|predicted_rank - actual_rank|)."""
    top5 = actual_rank.nsmallest(5).index
    return float((pred_rank.loc[top5] - actual_rank.loc[top5]).abs().median())


def kendall_tau_filtered(
    pred_score: pd.Series,
    actual_score: pd.Series,
    n_weeks: pd.Series,
    *,
    min_n_weeks: int,
) -> tuple[float, int]:
    """Kendall's tau over players with n_weeks >= min_n_weeks. Returns (tau, n)."""
    eligible = n_weeks[n_weeks >= min_n_weeks].index
    pred_e = pred_score.loc[eligible]
    actual_e = actual_score.loc[eligible]
    result = kendalltau(pred_e.to_numpy(), actual_e.to_numpy())
    return float(result.statistic), len(eligible)


def cell_verdict(
    *,
    metric_top12: float,
    mean_top12: float,
    metric_rank_err: float,
    mean_rank_err: float,
    top12_delta_threshold: float = 1.0 / 12.0,
) -> str:
    """Per-cell verdict per spec §3.5. Returns one of: SIGNAL, MARGINAL, NULL, REGRESSION."""
    top12_better = metric_top12 - mean_top12 >= top12_delta_threshold
    rankerr_better = metric_rank_err < mean_rank_err
    top12_worse = metric_top12 < mean_top12
    rankerr_worse = metric_rank_err > mean_rank_err
    if top12_better and rankerr_better:
        return "SIGNAL"
    if top12_worse and rankerr_worse:
        return "REGRESSION"
    if top12_better or rankerr_better:
        return "MARGINAL"
    return "NULL"


def decision_gate(verdicts: pd.DataFrame) -> str:
    """Roll up per-(season, position, metric) cell verdicts to a Phase 2 decision.

    Per spec §1.3 #3:
      - Greenlight iff some single non-mean metric M is SIGNAL at >= 3 of 4 positions
        in both years.
      - Marginal iff (a) some M is SIGNAL at >= 3 of 4 positions in exactly one year,
        OR (b) some M is SIGNAL-or-MARGINAL at >= 3 of 4 positions in both years.
      - No greenlight otherwise.

    Input: long DataFrame with columns [season, position, metric, cell_verdict].
    Expects exactly 2 distinct seasons; raises ValueError if not.
    """
    metrics = [m for m in verdicts["metric"].unique() if m != "mean"]
    years = sorted(verdicts["season"].unique())
    if len(years) != 2:
        raise ValueError(f"decision_gate expects exactly 2 seasons; got {years}")
    y1, y2 = int(years[0]), int(years[1])

    def signal_positions(metric: str, year: int) -> int:
        sub = verdicts[
            (verdicts["metric"] == metric)
            & (verdicts["season"] == year)
            & (verdicts["cell_verdict"] == "SIGNAL")
        ]
        return len(sub)

    def signal_or_marginal_positions(metric: str, year: int) -> int:
        sub = verdicts[
            (verdicts["metric"] == metric)
            & (verdicts["season"] == year)
            & (verdicts["cell_verdict"].isin(["SIGNAL", "MARGINAL"]))
        ]
        return len(sub)

    for metric in metrics:
        if signal_positions(metric, y1) >= 3 and signal_positions(metric, y2) >= 3:
            return "Greenlight"

    for metric in metrics:
        if signal_positions(metric, y1) >= 3 or signal_positions(metric, y2) >= 3:
            return "Marginal"
        if (
            signal_or_marginal_positions(metric, y1) >= 3
            and signal_or_marginal_positions(metric, y2) >= 3
        ):
            return "Marginal"

    return "No greenlight"


def assemble_season_diagnostic(
    *,
    weekly: pd.DataFrame,
    distributions: pd.DataFrame,
    actuals: pd.DataFrame,
    elite_thresholds: dict[Position, float],
    ruleset: Ruleset,
    n_samples: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_player_df, summary_df) for one season.

    per_player_df: one row per gsis_id, with metric scores + per-metric ranks
        (descending within position) + actual_total + actual_rank.
    summary_df: one row per (position, metric), with rank-recovery measurements +
        cell_verdict per spec §3.5.
    """
    # 1. Compute per-player p_elite via aggregate_to_season(return_samples=True).
    _, samples = aggregate_to_season(
        weekly, ruleset=ruleset, n_samples=n_samples, return_samples=True
    )

    # 2. Build per-player frame from distributions CSV.
    df = distributions[
        ["gsis_id", "position", "full_name", "season_mean", "season_p90", "n_weeks"]
    ].copy()
    df = df.rename(columns={"season_mean": "mean", "season_p90": "p90"})
    df["blend_70_30"] = 0.7 * df["mean"] + 0.3 * df["p90"]

    # 3. p_elite per row: P(season_samples >= elite_threshold[position]).
    # Pre-flatten samples dict for O(1) per-row lookup (single-season input).
    samples_by_gsis: dict[str, np.ndarray] = {gid: arr for (gid, _ssn), arr in samples.items()}

    def _p_elite_for(row: pd.Series) -> float:
        pos = Position(row["position"])
        threshold = elite_thresholds[pos]
        arr = samples_by_gsis.get(row["gsis_id"])
        if arr is None:
            return float("nan")
        return float((arr >= threshold).mean())

    df["p_elite"] = df.apply(_p_elite_for, axis=1)

    # 4. Join actuals.
    df = df.merge(
        actuals[["gsis_id", "actual_total", "actual_n_weeks"]],
        on="gsis_id",
        how="left",
    )
    df["actual_total"] = df["actual_total"].fillna(0.0)
    df["actual_n_weeks"] = df["actual_n_weeks"].fillna(0).astype(int)

    # 5. Per-position ranks under each metric (1 = best).
    for metric in _METRIC_NAMES:
        df[f"rank_{metric}"] = df.groupby("position")[metric].rank(ascending=False, method="min")
    df["actual_rank"] = df.groupby("position")["actual_total"].rank(ascending=False, method="min")

    # 6. Per-(position, metric) summary.
    summary_rows: list[dict[str, object]] = []
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        pos_df = df[df["position"] == pos.value].set_index("gsis_id")
        if pos_df.empty:
            continue
        # Mean baseline (used to compare every other metric against).
        mean_top12 = top_k_overlap(pos_df["rank_mean"], pos_df["actual_rank"], k=12)
        mean_rank_err = top5_rank_err(pos_df["rank_mean"], pos_df["actual_rank"])

        for metric in _METRIC_NAMES:
            m_top5 = top_k_overlap(pos_df[f"rank_{metric}"], pos_df["actual_rank"], k=5)
            m_top12 = top_k_overlap(pos_df[f"rank_{metric}"], pos_df["actual_rank"], k=12)
            m_top24 = top_k_overlap(pos_df[f"rank_{metric}"], pos_df["actual_rank"], k=24)
            m_rank_err = top5_rank_err(pos_df[f"rank_{metric}"], pos_df["actual_rank"])
            tau, n_tau = kendall_tau_filtered(
                pos_df[metric],
                pos_df["actual_total"],
                pos_df["actual_n_weeks"],
                min_n_weeks=6,
            )
            if metric == "mean":
                verdict = "BASELINE"
            else:
                verdict = cell_verdict(
                    metric_top12=m_top12,
                    mean_top12=mean_top12,
                    metric_rank_err=m_rank_err,
                    mean_rank_err=mean_rank_err,
                )
            summary_rows.append(
                {
                    "position": pos.value,
                    "metric": metric,
                    "top5_overlap": m_top5,
                    "top12_overlap": m_top12,
                    "top24_overlap": m_top24,
                    "top5_rank_err": m_rank_err,
                    "kendall_tau": tau,
                    "kendall_n": n_tau,
                    "cell_verdict": verdict,
                }
            )
    summary = pd.DataFrame(summary_rows)
    return df, summary
