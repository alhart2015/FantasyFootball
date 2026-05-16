"""CLI driver for the Tweedie yards_per_target sub-model probe.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md.

Reads WR features + weekly_stats from disk, runs walk_forward_residuals on
2021-2024 by default, computes the verdict, writes a summary markdown +
per-year CSV.

Usage:
    python scripts/probe_tweedie_yards_per_target.py \\
        --summary-out reports/feature_probe_tweedie_yards_per_target_summary.md \\
        --csv-out reports/feature_probe_tweedie_yards_per_target.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from projections.backtest.adoption_gate import paired_bootstrap_rmse_delta
from projections.backtest.tweedie_yards_per_target_probe import (
    PerStatVerdict,
    ProbeResults,
    compute_verdict,
    walk_forward_residuals,
)
from projections.features.cache import read_features
from projections.schemas import Position, Ruleset
from projections.store import read_partition

_DEFAULT_EVAL_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024)
_VALID_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
_COVERAGE_THRESHOLD: float = 0.95

# Marginal-zone threshold per PR #31's retrospective rule (|delta_fpts| < 0.005
# composite-fpts). The probe works in yards space (the per-stat unit), so the
# yards threshold is derived from the canonical PPR scoring constant rather
# than hardcoded. ASCII text only in stdout/file output to avoid Windows cp1252
# encoding crashes (spec section 5 risk #8).
_YARDS_PER_FPT: float = Ruleset.espn_ppr().receiving_yds_per_pt
_MARGINAL_ZONE_FPTS: float = 0.005
_MARGINAL_ZONE_THRESHOLD: float = _MARGINAL_ZONE_FPTS * _YARDS_PER_FPT

# Minimum paired-row count for a per-year bootstrap CI to be meaningful. Below
# this, _per_year_breakdown emits NaN deltas for the year.
_MIN_ROWS_PER_YEAR_BOOTSTRAP: int = 100


def _load_inputs(
    *,
    eval_years: Sequence[int],
    features_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WR features + weekly_stats for the seasons needed by the walk-forward.

    Train span starts at 2018 (matches BaselineModel.fit's lower bound).
    """
    seasons_needed = [y for y in _VALID_YEARS if y <= max(eval_years)]
    feat_parts = [
        read_features(Position.WR, s, features_root=features_root) for s in seasons_needed
    ]
    features = pd.concat(feat_parts, ignore_index=True)

    ws_parts = [read_partition(raw_root, "weekly_stats", season=s) for s in seasons_needed]
    weekly_stats = pd.concat(ws_parts, ignore_index=True)
    return features, weekly_stats


def _per_year_breakdown(results: ProbeResults, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """One-row-per-year breakdown of Delta-RMSE point + CI."""
    rows: list[dict[str, object]] = []
    for year in np.unique(results.year):
        mask = results.year == year
        n_paired = int(mask.sum())
        point: float = float("nan")
        lo: float = float("nan")
        hi: float = float("nan")
        if n_paired >= _MIN_ROWS_PER_YEAR_BOOTSTRAP:
            inc_residuals = results.actual_yards[mask] - results.pred_ridge[mask]
            cand_residuals = results.actual_yards[mask] - results.pred_tweedie[mask]
            delta = paired_bootstrap_rmse_delta(
                inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
            )
            point, lo, hi = delta.point, delta.lo_95, delta.hi_95
        rows.append(
            {
                "year": int(year),
                "n_paired": n_paired,
                "rmse_delta_point": point,
                "rmse_delta_lo": lo,
                "rmse_delta_hi": hi,
                "coverage": results.coverage_per_year.get(int(year), float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _write_summary(
    path: Path,
    *,
    verdict: PerStatVerdict,
    results: ProbeResults,
    per_year: pd.DataFrame,
    coverage_threshold: float,
    args: argparse.Namespace,
) -> None:
    """Markdown summary report. ASCII-only stdout for Windows cp1252 safety."""
    lines: list[str] = [
        "# Tweedie yards_per_target Probe -- Summary",
        "",
        "**Spec:** `docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md`",
        f"**Eval years:** {sorted({int(y) for y in results.year})}",
        f"**n_bootstrap:** {args.n_bootstrap}, seed: {args.seed}",
        "",
        f"## Verdict: **{verdict.verdict}**",
        "",
        f"- n_paired: {verdict.n_paired}",
        (
            f"- RMSE delta (tweedie - ridge): {verdict.rmse_delta.point:+.4f} yards "
            f"(95% CI [{verdict.rmse_delta.lo_95:+.4f}, "
            f"{verdict.rmse_delta.hi_95:+.4f}])"
        ),
        (
            f"- Composite-fpts equivalent (yards / {_YARDS_PER_FPT:g}): "
            f"{verdict.rmse_delta.point / _YARDS_PER_FPT:+.4f} fpts"
        ),
        "",
    ]
    if abs(verdict.rmse_delta.point) < _MARGINAL_ZONE_THRESHOLD:
        lines.append(
            f"**Magnitude flag:** |delta| {abs(verdict.rmse_delta.point):.4f} < "
            f"{_MARGINAL_ZONE_THRESHOLD:.3f} yards "
            f"(|delta_fpts| < {_MARGINAL_ZONE_FPTS}) -- in the marginal zone per PR #31's "
            "retrospective rule. Integration go/no-go must weight CI strength "
            "against magnitude."
        )
        lines.append("")

    lines.append("## Mechanism caveat")
    lines.append("")
    lines.append(
        "Incumbent arm is Ridge-decomp (a probe-internal construction), NOT "
        "current production. Current production for receiving_yards is direct "
        "RidgeCV (via `ensemble-decomposed`, which decomposes Stat.RECEPTIONS "
        "only per PR #36/#38). A SIGNAL verdict here does NOT imply "
        "Tweedie-decomp beats current production; that comparison is the "
        "integration adoption-gate's question on a separate cycle."
    )
    lines.append("")

    lines.append("## Per-year breakdown")
    lines.append("")
    lines.append(per_year.to_string(index=False))
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append(
        f"Coverage threshold: {coverage_threshold:.2f} (`targets > 0` rate per eval year)."
    )
    lines.append("")
    for year in sorted(results.coverage_per_year):
        rate = results.coverage_per_year[year]
        flag = "" if rate >= coverage_threshold else " -- BELOW THRESHOLD"
        lines.append(f"- {year}: {rate:.4f}{flag}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, per_year: pd.DataFrame, verdict: PerStatVerdict) -> None:
    """Long-form CSV: per-year rows + one pooled row."""
    pooled = pd.DataFrame(
        [
            {
                "year": "pooled",
                "n_paired": verdict.n_paired,
                "rmse_delta_point": verdict.rmse_delta.point,
                "rmse_delta_lo": verdict.rmse_delta.lo_95,
                "rmse_delta_hi": verdict.rmse_delta.hi_95,
                "coverage": float("nan"),
            }
        ]
    )
    out = pd.concat([per_year, pooled], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tweedie yards_per_target sub-model probe.")
    parser.add_argument(
        "--eval-years",
        type=int,
        nargs="+",
        choices=_VALID_YEARS,
        default=list(_DEFAULT_EVAL_YEARS),
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features"),
        help="root dir for the feature cache",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="root dir for the weekly_stats parquet store",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("reports/feature_probe_tweedie_yards_per_target_summary.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/feature_probe_tweedie_yards_per_target.csv"),
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=_COVERAGE_THRESHOLD,
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    features, weekly_stats = _load_inputs(
        eval_years=args.eval_years,
        features_root=args.features_root,
        raw_root=args.raw_root,
    )
    results = walk_forward_residuals(features, weekly_stats, eval_years=args.eval_years)
    verdict = compute_verdict(results, n_bootstrap=args.n_bootstrap, seed=args.seed)

    per_year = _per_year_breakdown(results, n_bootstrap=args.n_bootstrap, seed=args.seed)
    _write_summary(
        args.summary_out,
        verdict=verdict,
        results=results,
        per_year=per_year,
        coverage_threshold=args.coverage_threshold,
        args=args,
    )
    _write_csv(args.csv_out, per_year, verdict)

    # ASCII-only stdout; Windows cp1252 crashed on the catch_rate probe's
    # Delta symbol per PR #39's follow-up flag.
    print(f"Verdict: {verdict.verdict}")
    print(
        f"  RMSE delta {verdict.rmse_delta.point:+.4f} yards "
        f"(CI [{verdict.rmse_delta.lo_95:+.4f}, {verdict.rmse_delta.hi_95:+.4f}])"
    )
    print(f"  Summary: {args.summary_out}")
    print(f"  CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
