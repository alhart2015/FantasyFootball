"""Plan 8 — adoption gate CLI.

Reads a backtest run's per-row results.parquet, pairs rows on
(gsis_id, season, week) between two model classes, and emits per-position
adoption verdicts via paired-bootstrap CIs.

Usage:
    python -m scripts.adoption_gate \\
        --run data/backtest/run_<ts> \\
        --candidate <model_class> \\
        [--incumbent baseline] \\
        [--position QB|RB|TE|WR|all] \\
        [--csv-out reports/adoption_gate_<cand>_<ts>.csv] \\
        [--n-bootstrap 1000] \\
        [--seed 42]

Spec: docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from projections.backtest.adoption_gate import (
    PositionVerdict,
    paired_bootstrap_rmse_delta,
    paired_bootstrap_spearman_delta,
    verdict_for_position,
)
from projections.schemas import Position


def load_run_parquet(run_dir: Path) -> pd.DataFrame:
    """Load <run_dir>/results.parquet.

    Raises:
        FileNotFoundError: results.parquet missing under run_dir.
    """
    results_path = run_dir / "results.parquet"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"results.parquet missing under {run_dir}; this CLI expects per-row "
            f"backtest output produced by scripts/backtest.py."
        )
    return pd.read_parquet(results_path)


def validate_model_classes_present(df: pd.DataFrame, *, incumbent: str, candidate: str) -> None:
    """Raise ValueError if either incumbent or candidate is not in df['model_class']."""
    present = set(df["model_class"].unique())
    if candidate not in present:
        raise ValueError(
            f"candidate model_class '{candidate}' not present in run; "
            f"present classes: {sorted(present)}"
        )
    if incumbent not in present:
        raise ValueError(
            f"incumbent model_class '{incumbent}' not present in run; "
            f"present classes: {sorted(present)}"
        )


def pair_rows(
    position_df: pd.DataFrame,
    *,
    incumbent: str,
    candidate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pair rows on (gsis_id, season, week) between incumbent and candidate.

    Args:
        position_df: rows for a single position (caller filters).
        incumbent: incumbent model_class.
        candidate: candidate model_class.

    Returns:
        (predicted_incumbent, predicted_candidate, actual, grouping, n_dropped)
        as 1-d numpy arrays plus the count of unpaired rows that were dropped.
        grouping is the held-out year per row.
    """
    inc_rows = position_df[position_df["model_class"] == incumbent]
    cand_rows = position_df[position_df["model_class"] == candidate]
    keys = ["gsis_id", "season", "week"]
    paired = inc_rows.merge(
        cand_rows,
        on=keys,
        how="inner",
        suffixes=("_inc", "_cand"),
        validate="one_to_one",
    )
    n_inc = len(inc_rows)
    n_cand = len(cand_rows)
    n_paired = len(paired)
    n_dropped = (n_inc - n_paired) + (n_cand - n_paired)
    if n_dropped > 0:
        warnings.warn(
            f"pair_rows dropped {n_dropped} unpaired rows (incumbent={n_inc}, "
            f"candidate={n_cand}, paired={n_paired}); both model classes should "
            f"be scored on identical (gsis_id, season, week) inputs.",
            stacklevel=2,
        )
    return (
        paired["mean_inc"].to_numpy(dtype=np.float64),
        paired["mean_cand"].to_numpy(dtype=np.float64),
        paired["actual_ppr_inc"].to_numpy(dtype=np.float64),
        paired["season"].to_numpy(),
        n_dropped,
    )


def _per_year_breakdown(
    inc_pred: np.ndarray,
    cand_pred: np.ndarray,
    actual: np.ndarray,
    grouping: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """One row per year with per-year-only bootstrap CIs (informational)."""
    years = np.unique(grouping)
    rows = []
    for y in years:
        mask = grouping == y
        inc_y = actual[mask] - inc_pred[mask]
        cand_y = actual[mask] - cand_pred[mask]
        if mask.sum() < 100:
            rows.append(
                {
                    "year": int(y),
                    "n_paired": int(mask.sum()),
                    "rmse_delta_point": float("nan"),
                    "rmse_delta_lo": float("nan"),
                    "rmse_delta_hi": float("nan"),
                    "spearman_delta_point": float("nan"),
                    "spearman_delta_lo": float("nan"),
                    "spearman_delta_hi": float("nan"),
                }
            )
            continue
        rmse = paired_bootstrap_rmse_delta(inc_y, cand_y, n_bootstrap=n_bootstrap, seed=seed)
        spear = paired_bootstrap_spearman_delta(
            inc_pred[mask],
            cand_pred[mask],
            actual[mask],
            np.full(mask.sum(), y),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append(
            {
                "year": int(y),
                "n_paired": int(mask.sum()),
                "rmse_delta_point": rmse.point,
                "rmse_delta_lo": rmse.lo_95,
                "rmse_delta_hi": rmse.hi_95,
                "spearman_delta_point": spear.point,
                "spearman_delta_lo": spear.lo_95,
                "spearman_delta_hi": spear.hi_95,
            }
        )
    return pd.DataFrame(rows)


def evaluate_position(
    df: pd.DataFrame,
    *,
    position: Position,
    incumbent: str,
    candidate: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PositionVerdict:
    """Build a PositionVerdict for a single position from the run frame."""
    pos_df = df[df["position"] == position.value]
    inc_pred, cand_pred, actual, grouping, _ = pair_rows(
        pos_df, incumbent=incumbent, candidate=candidate
    )
    inc_residuals = actual - inc_pred
    cand_residuals = actual - cand_pred
    rmse = paired_bootstrap_rmse_delta(
        inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
    )
    spear = paired_bootstrap_spearman_delta(
        inc_pred, cand_pred, actual, grouping, n_bootstrap=n_bootstrap, seed=seed
    )
    label, reason = verdict_for_position(rmse, spear)
    breakdown = _per_year_breakdown(
        inc_pred, cand_pred, actual, grouping, n_bootstrap=n_bootstrap, seed=seed
    )
    return PositionVerdict(
        position=position,
        incumbent_class=incumbent,
        candidate_class=candidate,
        rmse_delta=rmse,
        spearman_delta=spear,
        verdict=label,
        reason=reason,
        per_year_breakdown=breakdown,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 8 adoption gate.")
    parser.add_argument("--run", type=Path, required=True, help="run_<ts> directory")
    parser.add_argument("--candidate", type=str, required=True, help="candidate model_class")
    parser.add_argument(
        "--incumbent", type=str, default="baseline", help="incumbent model_class (default baseline)"
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["QB", "RB", "TE", "WR", "all"],
        default="all",
        help="position to evaluate (default all)",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="optional CSV output path")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_run_parquet(args.run)
    validate_model_classes_present(df, incumbent=args.incumbent, candidate=args.candidate)
    print(f"Loaded {len(df)} rows from {args.run / 'results.parquet'}.")
    print(f"Model classes present: {sorted(df['model_class'].unique())}")


if __name__ == "__main__":
    main()
