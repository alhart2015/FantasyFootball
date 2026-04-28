"""Plan 7 Phase 0 — calibration breakdown diagnostic.

Reads the per-row backtest output written by scripts/backtest.py. Computes
per-stat empirical [p10, p90] coverage for Model C-NB rows on each
held-out (position, year) cell, weights each stat's contribution by its
share of total fantasy-point variance, and emits a CSV breakdown
attributing the cell's coverage gap vs Model A to ``count_share`` and
``yards_share`` in [0, 1].

Decision rule encoded in the CSV ``decision`` column:
  - count_share >= 0.5  -> "proceed_phase_1"
  - yards_share >= 0.5  -> "stop_file_yards_plan"
  - else                -> "proceed_with_followup"

Spec: docs/superpowers/specs/2026-04-28-plan-7-calibration-aware-nb-design.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from projections.distributions import Distribution, unpack_per_stat_params
from projections.models import POSITION_DISPATCH, BaselineModel
from projections.schemas import Stat

# Stats Plan 5c routes through NB-2 (count) vs QuantileDistribution (yards).
# Match COUNT_STATS_FOR_NB in models/lightgbm_nb.py.
_COUNT_STATS: Final[frozenset[str]] = frozenset(
    {"passing_tds", "rushing_tds", "receiving_tds", "interceptions", "fumbles_lost"}
)


def _stat_class(stat_value: str) -> str:
    return "count" if stat_value in _COUNT_STATS else "yards"


def _resolve_target_stats() -> dict[str, tuple[str, ...]]:
    """{position_value: (stat_value, ...)} from the baseline factory.

    target_stats are identical across model classes by construction
    (each Model implementation declares the same per-position target
    list). Using the baseline factory avoids loading lightgbm just to
    read attribute metadata.
    """
    out: dict[str, tuple[str, ...]] = {}
    for position, dispatch in POSITION_DISPATCH.items():
        # BaselineModel exposes ``target_stats`` as an attribute; the
        # generic Model Protocol does not, so we narrow the type here.
        model = cast(BaselineModel, dispatch.factories["baseline"]())
        out[position.value] = tuple(s.value for s in model.target_stats)
    return out


def compute_per_stat_coverage(per_row: pd.DataFrame, *, position: str, year: int) -> pd.DataFrame:
    """Compute empirical [p10, p90] coverage per stat for a single
    (position, year) cell.

    Per-stat distributions are unpacked from the row's ``params`` blob
    (msgpack bytes -> dict[Stat, Distribution]). Per-stat actuals come
    from ``<stat>_actual`` columns. Composite p10/p90 columns at the row
    level are NOT per-stat — those are composite fantasy-point quantiles
    and unrelated to this decomposition.

    Returns a frame with columns:
      stat, stat_class, n_rows, coverage_p10p90, variance_contribution.

    ``variance_contribution`` is empirical Var(actual) per stat. Only
    ratios matter for the share decomposition, so the scoring-weight^2
    multiplier is folded in implicitly: count stats (low-mean integer)
    have small Var(actual); yards stats (continuous, large variance)
    have much larger Var(actual). The ratio is the right proxy for
    share-of-composite-fantasy-point-variance.
    """
    cell = per_row[(per_row["position"] == position) & (per_row["season"] == year)]
    if cell.empty:
        return pd.DataFrame(
            columns=[
                "stat",
                "stat_class",
                "n_rows",
                "coverage_p10p90",
                "variance_contribution",
            ]
        )

    target_stats = _resolve_target_stats().get(position, ())
    if not target_stats:
        raise ValueError(f"No target stats resolved for position={position}")

    unpacked: list[dict[Stat, Distribution]] = [
        unpack_per_stat_params(b) for b in cell["params"].tolist()
    ]

    rows: list[dict[str, object]] = []
    for stat_value in target_stats:
        actual_col = f"{stat_value}_actual"
        if actual_col not in cell.columns:
            continue
        actual = cell[actual_col].to_numpy(dtype=np.float64)

        p10_list: list[float] = []
        p90_list: list[float] = []
        kept_actual: list[float] = []
        for i, per_stat_dists in enumerate(unpacked):
            dist = next((d for s, d in per_stat_dists.items() if s.value == stat_value), None)
            if dist is None:
                continue
            p10_list.append(float(dist.quantile(0.10)))
            p90_list.append(float(dist.quantile(0.90)))
            kept_actual.append(float(actual[i]))

        if not kept_actual:
            continue

        actual_arr = np.asarray(kept_actual, dtype=np.float64)
        p10 = np.asarray(p10_list, dtype=np.float64)
        p90 = np.asarray(p90_list, dtype=np.float64)
        in_band = (actual_arr >= p10) & (actual_arr <= p90)
        coverage = float(in_band.mean()) if in_band.size else 0.0
        variance_contribution = float(np.var(actual_arr)) if actual_arr.size else 0.0
        rows.append(
            {
                "stat": stat_value,
                "stat_class": _stat_class(stat_value),
                "n_rows": int(actual_arr.size),
                "coverage_p10p90": coverage,
                "variance_contribution": variance_contribution,
            }
        )
    return pd.DataFrame(rows)


def attribute_coverage_gap(per_stat: pd.DataFrame) -> dict[str, float]:
    """Reduce a per-stat frame to a single (position, year) cell summary.

    Computes:
      count_coverage_gap, yards_coverage_gap: variance-weighted average of
        per-stat (target_coverage - empirical_coverage) within each class.
        Target is 0.80 (the nominal [p10, p90] interval).
      count_share, yards_share: each class's share of total fantasy-point
        variance contribution. Sum to 1.0 (or 0.0 if total variance is 0).
    """
    if per_stat.empty:
        return {
            "count_share": 0.0,
            "yards_share": 0.0,
            "count_coverage_gap": 0.0,
            "yards_coverage_gap": 0.0,
        }

    total_var = float(per_stat["variance_contribution"].sum())
    if total_var <= 0:
        return {
            "count_share": 0.0,
            "yards_share": 0.0,
            "count_coverage_gap": 0.0,
            "yards_coverage_gap": 0.0,
        }

    has_coverage = "coverage_p10p90" in per_stat.columns
    out: dict[str, float] = {}
    for class_name in ("count", "yards"):
        sub = per_stat[per_stat["stat_class"] == class_name]
        share = float(sub["variance_contribution"].sum()) / total_var
        if sub.empty or share == 0.0 or not has_coverage:
            gap = 0.0
        else:
            w = sub["variance_contribution"].to_numpy()
            cov = sub["coverage_p10p90"].to_numpy()
            gap = float(np.sum(w * (0.80 - cov)) / w.sum())
        out[f"{class_name}_share"] = share
        out[f"{class_name}_coverage_gap"] = gap
    return out


def _decision(count_share: float, yards_share: float) -> str:
    if count_share >= 0.5:
        return "proceed_phase_1"
    if yards_share >= 0.5:
        return "stop_file_yards_plan"
    return "proceed_with_followup"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-row-parquet",
        type=Path,
        required=True,
        help="Path to a per-row results.parquet from scripts/backtest.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the breakdown CSV.",
    )
    args = parser.parse_args(argv)

    per_row = pd.read_parquet(args.per_row_parquet)
    nb_mask = per_row["model_id"].str.startswith("lightgbm-nb:")
    per_row = per_row[nb_mask].copy()
    if per_row.empty:
        print(
            "No lightgbm-nb rows found in per-row frame; nothing to attribute.",
            file=sys.stderr,
        )
        sys.exit(2)

    summary_rows: list[dict[str, object]] = []
    for (position, year), _ in per_row.groupby(["position", "season"]):
        per_stat = compute_per_stat_coverage(per_row, position=position, year=year)
        attribution = attribute_coverage_gap(per_stat)
        summary_rows.append(
            {
                "position": position,
                "year": year,
                "count_share": attribution["count_share"],
                "yards_share": attribution["yards_share"],
                "count_coverage_gap": attribution["count_coverage_gap"],
                "yards_coverage_gap": attribution["yards_coverage_gap"],
                "decision": _decision(attribution["count_share"], attribution["yards_share"]),
            }
        )

    out = pd.DataFrame(summary_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
