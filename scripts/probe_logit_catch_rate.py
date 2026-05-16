"""CLI driver for the logit catch_rate sub-model probe.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md.

Reads WR features + weekly_stats from disk, runs walk_forward_residuals on
2021-2024 by default, computes the verdict, writes a summary markdown +
per-year CSV.

Usage:
    python scripts/probe_logit_catch_rate.py \\
        --summary-out reports/feature_probe_logit_catch_rate_summary.md \\
        --csv-out reports/feature_probe_logit_catch_rate.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from projections.backtest.logit_catch_rate_probe import (
    PerStatVerdict,
    ProbeResults,
    compute_verdict,
    walk_forward_residuals,
)
from projections.features.cache import read_features
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_EVAL_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024)
_VALID_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
_COVERAGE_THRESHOLD: float = 0.95
_MARGINAL_ZONE_THRESHOLD: float = 0.005  # receptions; per PR #31 retrospective


def _load_inputs(
    *,
    eval_years: Sequence[int],
    features_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WR features (for all seasons needed for the eval window) +
    weekly_stats for the union of train and eval years.

    Train span starts at 2018 (matches BaselineModel.fit's lower bound).
    """
    seasons_needed = sorted({*_VALID_YEARS[: _VALID_YEARS.index(max(eval_years)) + 1]})
    feat_parts = [
        read_features(Position.WR, s, features_root=features_root) for s in seasons_needed
    ]
    features = pd.concat(feat_parts, ignore_index=True)

    ws_parts = [read_partition(raw_root, "weekly_stats", season=s) for s in seasons_needed]
    weekly_stats = pd.concat(ws_parts, ignore_index=True)
    return features, weekly_stats


def _per_year_breakdown(results: ProbeResults, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """One-row-per-year breakdown of Δ-RMSE point + CI."""
    from projections.backtest.adoption_gate import paired_bootstrap_rmse_delta

    rows: list[dict[str, object]] = []
    for year in np.unique(results.year):
        mask = results.year == year
        if mask.sum() < 100:
            rows.append(
                {
                    "year": int(year),
                    "n_paired": int(mask.sum()),
                    "rmse_delta_point": float("nan"),
                    "rmse_delta_lo": float("nan"),
                    "rmse_delta_hi": float("nan"),
                    "coverage": results.coverage_per_year.get(int(year), float("nan")),
                }
            )
            continue
        inc_residuals = results.actual_receptions[mask] - results.pred_ridge[mask]
        cand_residuals = results.actual_receptions[mask] - results.pred_logit[mask]
        delta = paired_bootstrap_rmse_delta(
            inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
        )
        rows.append(
            {
                "year": int(year),
                "n_paired": int(mask.sum()),
                "rmse_delta_point": delta.point,
                "rmse_delta_lo": delta.lo_95,
                "rmse_delta_hi": delta.hi_95,
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
    """Markdown summary report."""
    lines: list[str] = [
        "# Logit catch_rate Probe — Summary",
        "",
        "**Spec:** `docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md`",
        f"**Eval years:** {sorted({int(y) for y in results.year})}",
        f"**n_bootstrap:** {args.n_bootstrap}, seed: {args.seed}",
        "",
        f"## Verdict: **{verdict.verdict}**",
        "",
        f"- n_paired: {verdict.n_paired}",
        (
            f"- RMSE Δ (logit - ridge): {verdict.rmse_delta.point:+.4f} "
            f"(95% CI [{verdict.rmse_delta.lo_95:+.4f}, {verdict.rmse_delta.hi_95:+.4f}])"
        ),
        "",
    ]
    if abs(verdict.rmse_delta.point) < _MARGINAL_ZONE_THRESHOLD:
        magnitude = abs(verdict.rmse_delta.point)
        lines.append(
            f"**Magnitude flag:** |Δ| {magnitude:.4f} < "
            f"{_MARGINAL_ZONE_THRESHOLD:.3f} receptions — in the marginal zone "
            "per PR #31's retrospective rule. Integration go/no-go must weight "
            "CI strength against magnitude."
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
        flag = "" if rate >= coverage_threshold else " — BELOW THRESHOLD"
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
    parser = argparse.ArgumentParser(description="Logit catch_rate sub-model probe.")
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
        default=Path("reports/feature_probe_logit_catch_rate_summary.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/feature_probe_logit_catch_rate.csv"),
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

    print(f"Verdict: {verdict.verdict}")
    print(
        f"  RMSE Δ {verdict.rmse_delta.point:+.4f} "
        f"(CI [{verdict.rmse_delta.lo_95:+.4f}, {verdict.rmse_delta.hi_95:+.4f}])"
    )
    print(f"  Summary: {args.summary_out}")
    print(f"  CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
