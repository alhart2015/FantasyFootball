"""CLI driver for the RB rushing + receiving decomposition probe.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md.

Reads RB features + weekly_stats from disk, runs walk_forward_residuals on
2021-2024 by default, computes per-stat verdicts, writes a summary markdown
+ per-stat CSV.

Usage:
    python scripts/probe_rb_decomposition.py \\
        --summary-out reports/feature_probe_rb_decomposition_summary.md \\
        --csv-out reports/feature_probe_rb_decomposition_per_stat.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.backtest.rb_decomposition_probe import (
    PerStatVerdict,
    WalkForwardOutput,
    compute_verdicts,
    walk_forward_residuals,
)
from projections.features.cache import read_features
from projections.schemas import Position, Ruleset
from projections.store import read_partition

_DEFAULT_EVAL_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024)
_VALID_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
_COVERAGE_THRESHOLD: float = 0.95

# Marginal-zone threshold per PR #31's retrospective rule: |delta_fpts| < 0.005.
# ASCII text only in stdout/file output to avoid Windows cp1252 encoding crashes
# (spec section 5 risk #6 / PR #39 follow-up).
_MARGINAL_ZONE_FPTS: float = 0.005
_RULESET: Ruleset = Ruleset.espn_ppr()


def _load_inputs(
    *,
    eval_years: Sequence[int],
    features_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RB features + weekly_stats for the seasons needed by the walk-forward."""
    seasons_needed = [y for y in _VALID_YEARS if y <= max(eval_years)]
    feat_parts = [
        read_features(Position.RB, s, features_root=features_root) for s in seasons_needed
    ]
    features = pd.concat(feat_parts, ignore_index=True)
    ws_parts = [read_partition(raw_root, "weekly_stats", season=s) for s in seasons_needed]
    weekly_stats = pd.concat(ws_parts, ignore_index=True)
    return features, weekly_stats


def _stat_to_fpts(stat_value: str, delta_yards_or_count: float) -> float:
    """Translate per-stat delta to composite-fpts delta via Ruleset.espn_ppr()."""
    if stat_value == "rushing_yards":
        return delta_yards_or_count / _RULESET.rushing_yds_per_pt
    if stat_value == "receiving_yards":
        return delta_yards_or_count / _RULESET.receiving_yds_per_pt
    if stat_value == "rushing_tds":
        return delta_yards_or_count * _RULESET.rushing_td_pts
    if stat_value == "receiving_tds":
        return delta_yards_or_count * _RULESET.receiving_td_pts
    if stat_value == "receptions":
        return delta_yards_or_count * _RULESET.reception_pts
    raise ValueError(f"unknown stat value for fpts translation: {stat_value}")


def _write_summary(
    path: Path,
    *,
    verdicts: list[PerStatVerdict],
    output: WalkForwardOutput,
    coverage_threshold: float,
    args: argparse.Namespace,
) -> None:
    """Markdown summary report. ASCII-only stdout for Windows cp1252 safety."""
    lines: list[str] = [
        "# RB Rushing + Receiving Decomposition Probe -- Summary",
        "",
        "**Spec:** `docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md`",
        f"**Eval years:** {list(output.eval_years)}",
        f"**n_bootstrap:** {args.n_bootstrap}, seed: {args.seed}",
        "",
        "## Per-stat verdicts",
        "",
        (
            "| Stat | n_paired | RMSE delta (decomp - direct) | 95% CI | "
            "Composite-fpts equiv | Magnitude flag | Verdict |"
        ),
        "|---|---:|---:|---|---:|---|:---:|",
    ]
    for v in verdicts:
        fpts_delta = _stat_to_fpts(v.stat.value, v.rmse_delta.point)
        mag_flag = "MARGINAL" if abs(fpts_delta) < _MARGINAL_ZONE_FPTS else ""
        lines.append(
            f"| {v.stat.value} | {v.n_paired} | "
            f"{v.rmse_delta.point:+.4f} | "
            f"[{v.rmse_delta.lo_95:+.4f}, {v.rmse_delta.hi_95:+.4f}] | "
            f"{fpts_delta:+.4f} | {mag_flag} | **{v.verdict}** |"
        )
    lines.append("")

    lines.append("## Coverage (eval rows)")
    lines.append("")
    lines.append(f"Coverage threshold: {coverage_threshold:.2f} per volume axis per eval year.")
    lines.append("")
    lines.append("### Carries > 0 rate (rushing axis)")
    for year in sorted(output.coverage_carries_by_year):
        rate = output.coverage_carries_by_year[year]
        flag = "" if rate >= coverage_threshold else " -- BELOW THRESHOLD"
        lines.append(f"- {year}: {rate:.4f}{flag}")
    lines.append("")
    lines.append("### Targets > 0 rate (receiving axis)")
    for year in sorted(output.coverage_targets_by_year):
        rate = output.coverage_targets_by_year[year]
        flag = "" if rate >= coverage_threshold else " -- BELOW THRESHOLD"
        lines.append(f"- {year}: {rate:.4f}{flag}")
    lines.append("")

    lines.append("## Mechanism caveat")
    lines.append("")
    lines.append(
        "This probe tests decomposition with RidgeCV everywhere (the same model "
        "class on both arms). Factor-appropriate sub-model classes (Poisson, "
        "Gamma / Tweedie, logistic) are separate probe + integration cycles per "
        "spec section 1.4 #3. PR #39 / PR #44 closed two of these on WR with NULL "
        "verdicts; RB-side factor-class probes remain independent tests if any "
        "stat here SIGNALs."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, verdicts: list[PerStatVerdict]) -> None:
    """Long-form CSV: one row per stat with delta + CI + composite-fpts equiv."""
    rows: list[dict[str, object]] = []
    for v in verdicts:
        rows.append(
            {
                "stat": v.stat.value,
                "n_paired": v.n_paired,
                "rmse_delta_point": v.rmse_delta.point,
                "rmse_delta_lo": v.rmse_delta.lo_95,
                "rmse_delta_hi": v.rmse_delta.hi_95,
                "composite_fpts_equivalent": _stat_to_fpts(v.stat.value, v.rmse_delta.point),
                "verdict": v.verdict,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RB rushing + receiving decomposition probe.")
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
        default=Path("reports/feature_probe_rb_decomposition_summary.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/feature_probe_rb_decomposition_per_stat.csv"),
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
    output = walk_forward_residuals(features, weekly_stats, eval_years=args.eval_years)
    verdicts = compute_verdicts(output, n_bootstrap=args.n_bootstrap, seed=args.seed)

    _write_summary(
        args.summary_out,
        verdicts=verdicts,
        output=output,
        coverage_threshold=args.coverage_threshold,
        args=args,
    )
    _write_csv(args.csv_out, verdicts)

    # ASCII-only stdout (Windows cp1252 guard per PR #39 follow-up).
    print("Per-stat verdicts:")
    for v in verdicts:
        print(
            f"  {v.stat.value:<18s} -> {v.verdict:<10s} "
            f"(delta {v.rmse_delta.point:+.4f}, "
            f"CI [{v.rmse_delta.lo_95:+.4f}, {v.rmse_delta.hi_95:+.4f}], "
            f"n_paired {v.n_paired})"
        )
    print(f"  Summary: {args.summary_out}")
    print(f"  CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
