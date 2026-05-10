"""WR receiving-stats target decomposition probe - CLI.

Loads WR feature cache (data/features/wr/season=YYYY/week=WW/part.parquet)
and weekly stats (data/raw/weekly_stats/season=YYYY/part.parquet); runs the
walk-forward harness in `projections.backtest.target_decomposition_probe`;
renders per-stat markdown + CSV reports into --output-dir.

Spec: docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md
Plan: docs/superpowers/plans/2026-05-10-target-decomposition-probe.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from projections.backtest.target_decomposition_probe import (
    WalkForwardOutput,
    render_probe_report,
    walk_forward_residuals,
    write_per_stat_csv,
    write_per_stat_markdown,
)
from projections.features.cache import read_features
from projections.models.baseline import _WR_FEATURE_COLUMNS
from projections.schemas import Position, Stat, WeeklyStatsSchema
from projections.store import read_partition

_LOG = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the rendered reports (created if missing).",
    )
    p.add_argument(
        "--eval-years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024],
        help="Walk-forward eval years (default: 2021 2022 2023 2024).",
    )
    p.add_argument(
        "--train-start",
        type=int,
        default=2018,
        help="Inclusive lower bound for any training window (default: 2018).",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=5000,
        help="Paired-bootstrap resample count (default: 5000).",
    )
    p.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="Per-eval-year and per-train-window targets > 0 rate floor "
        "(default: 0.95). Relaxation must be documented in the report.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0xD3C0,
        help="Bootstrap seed (default: 0xD3C0). Pin to reproduce.",
    )
    p.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features"),
        help="Feature cache root (default: data/features).",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Raw partition root (default: data/raw). Weekly stats live at "
        "<raw-root>/weekly_stats/season=YYYY/part.parquet - read via "
        "projections.store.read_partition(raw_root, 'weekly_stats', season=Y).",
    )
    return p


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, list[str]]:
    """Load WR features + weekly stats from canonical paths.

    Returns (features_by_year, weekly_stats, feature_columns).
    """
    needed_seasons = list(range(args.train_start, max(args.eval_years) + 1))
    features_by_year: dict[int, pd.DataFrame] = {}
    for season in needed_seasons:
        features_by_year[season] = read_features(
            position=Position.WR,
            season=season,
            features_root=args.features_root,
        )

    # Weekly stats: load all needed seasons via store.read_partition.
    # Pattern matches scripts/predict_2024.py - read_partition(raw_root,
    # "weekly_stats", season=Y) returns the canonical WeeklyStatsSchema frame.
    weekly_frames = [
        read_partition(args.raw_root, "weekly_stats", season=s) for s in needed_seasons
    ]
    weekly_stats = pd.concat(weekly_frames, ignore_index=True)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)

    return features_by_year, weekly_stats, list(_WR_FEATURE_COLUMNS)


def _enforce_coverage(walk_forward_output: WalkForwardOutput, threshold: float) -> list[str]:
    """Return a list of human-readable warnings for any per-eval-year
    coverage below `threshold`. Empty list if all clear."""
    warnings: list[str] = []
    for c in walk_forward_output.coverage_by_year:
        if c.targets_positive_rate < threshold:
            warnings.append(
                f"Eval year {c.eval_year}: targets > 0 rate "
                f"{c.targets_positive_rate:.3f} below threshold {threshold:.2f}"
            )
    for c in walk_forward_output.train_coverage_by_year:
        if c.targets_positive_rate < threshold:
            warnings.append(
                f"Train window for eval {c.eval_year}: targets > 0 rate "
                f"{c.targets_positive_rate:.3f} below threshold {threshold:.2f}"
            )
    return warnings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    features_by_year, weekly_stats, feature_columns = _load_inputs(args)

    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=feature_columns,
        eval_years=tuple(args.eval_years),
        train_start=args.train_start,
    )

    coverage_warnings = _enforce_coverage(out, args.coverage_threshold)
    if coverage_warnings:
        _LOG.warning(
            "Coverage threshold %.2f not met:\n  %s",
            args.coverage_threshold,
            "\n  ".join(coverage_warnings),
        )
        _LOG.warning(
            "(Continuing per the PR #31 retrospective rule - relaxation "
            "must be documented in the summary report.)"
        )

    report = render_probe_report(out, bootstrap_n=args.bootstrap_n, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "feature_probe_target_decomposition_per_stat.csv"
    write_per_stat_csv(report, csv_path)
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RECEIVING_TDS):
        md_path = args.output_dir / f"feature_probe_target_decomposition_{stat.value}.md"
        write_per_stat_markdown(report, stat, md_path)

    _LOG.info("Wrote outputs to %s", args.output_dir)
    for stat, v in report.verdicts.items():
        _LOG.info(
            "  %s: %s (Delta-RMSE %+.4f, CI [%+.4f, %+.4f], n=%d)",
            stat.value,
            v.verdict,
            v.rmse_delta.point,
            v.rmse_delta.lo_95,
            v.rmse_delta.hi_95,
            v.n_paired,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
