"""Plan 8 / Plan 9 — adoption gate CLI.

Reads per-row backtest results, pairs rows on (gsis_id, season, week), and
emits per-position adoption verdicts via paired-bootstrap CIs.

Two pairing modes:

1. **Single-run (Plan 8):** one ``results.parquet`` containing two
   ``model_class`` values (e.g. ``baseline`` vs ``ensemble``). Rows are paired
   on the model_class column.

2. **Dual-run (Plan 9+):** two ``results.parquet`` files representing the
   same backtest configuration with different feature sets (both files share
   the same ``model_class``). The dual-run loader synthesizes per-file
   ``_baseline_run`` / ``_candidate_run`` labels so the existing pair logic
   applies unchanged.

Usage (single-run):
    python -m scripts.adoption_gate \\
        --run data/backtest/run_<ts> \\
        --candidate <model_class> \\
        [--incumbent baseline] \\
        [--position QB|RB|TE|WR|all] \\
        [--csv-out reports/adoption_gate_<cand>_<ts>.csv] \\
        [--n-bootstrap 1000] \\
        [--seed 42]

Usage (dual-run):
    python -m scripts.adoption_gate \\
        --baseline-run data/backtest/run_pre_<feature> \\
        --candidate-run data/backtest/run_post_<feature> \\
        [--position QB|RB|TE|WR|all] \\
        [--csv-out reports/adoption_gate_<feature>.csv] \\
        [--n-bootstrap 1000] \\
        [--seed 42]

Spec: docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md (single)
      docs/superpowers/specs/2026-04-29-plan-9-pbp-ingest-opp-epa-design.md (dual)
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


# Synthetic model_class labels used by the dual-run loader. They are
# intentionally underscore-prefixed so a real model_class (e.g. "baseline",
# "ensemble", "lightgbm") can never collide with them.
DUAL_RUN_INCUMBENT_CLASS = "_baseline_run"
DUAL_RUN_CANDIDATE_CLASS = "_candidate_run"


def load_dual_run_paired(baseline_run: Path, candidate_run: Path) -> pd.DataFrame:
    """Load two backtest runs and return a single combined frame with
    ``model_class`` synthesized to ``_baseline_run`` / ``_candidate_run`` so
    the existing :func:`pair_rows` logic can pair them.

    Both runs must hold identical (gsis_id, season, week, position) coverage —
    they're meant to be the same backtest configuration with only the
    feature set differing. Any row in one but not the other indicates a
    backtest mismatch (e.g. a feature builder dropped rows on one side) and
    must surface as an error rather than silently shrink the paired sample.

    Used for feature-set vs feature-set comparisons (Plan 9 onward) where
    both runs share the same ``model_class``. The single-run ``--run`` path
    remains the canonical Plan 8 mode for model-class-vs-model-class
    comparisons.

    Args:
        baseline_run: directory containing ``results.parquet`` for the
            pre-feature run (treated as incumbent for the verdict).
        candidate_run: directory containing ``results.parquet`` for the
            post-feature run (treated as candidate for the verdict).

    Returns:
        Concatenated frame with the original ``model_class`` column overwritten
        by the synthesized labels above.

    Raises:
        FileNotFoundError: either run is missing its ``results.parquet``.
        ValueError: row coverage between the two runs differs.
    """
    base = load_run_parquet(baseline_run)
    cand = load_run_parquet(candidate_run)

    keys = ["gsis_id", "season", "week", "position"]
    base_keys = base[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    cand_keys = cand[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    if not base_keys.equals(cand_keys):
        only_in_base = base_keys.merge(cand_keys, on=keys, how="left", indicator=True).query(
            '_merge == "left_only"'
        )
        only_in_cand = cand_keys.merge(base_keys, on=keys, how="left", indicator=True).query(
            '_merge == "left_only"'
        )
        raise ValueError(
            f"row coverage mismatch between runs: "
            f"{len(only_in_base)} rows only in baseline-run, "
            f"{len(only_in_cand)} rows only in candidate-run. "
            "Both runs must hold identical (gsis_id, season, week, position) coverage."
        )

    base = base.assign(model_class=DUAL_RUN_INCUMBENT_CLASS)
    cand = cand.assign(model_class=DUAL_RUN_CANDIDATE_CLASS)
    return pd.concat([base, cand], ignore_index=True)


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


def format_position_report(pv: PositionVerdict) -> str:
    """One-position markdown report: verdict line + per-year breakdown table."""
    lines: list[str] = []
    lines.append(
        f"### {pv.position.value} — {pv.candidate_class} vs {pv.incumbent_class}: **{pv.verdict}**"
    )
    lines.append("")
    lines.append(f"_{pv.reason}_")
    lines.append("")
    lines.append(
        f"- n_paired: {pv.rmse_delta.n_paired_rows}; n_bootstrap: {pv.rmse_delta.n_bootstrap}"
    )
    lines.append(
        f"- RMSE delta: {pv.rmse_delta.point:+.4f} "
        f"(95% CI [{pv.rmse_delta.lo_95:+.4f}, {pv.rmse_delta.hi_95:+.4f}])"
    )
    lines.append(
        f"- Spearman delta: {pv.spearman_delta.point:+.4f} "
        f"(95% CI [{pv.spearman_delta.lo_95:+.4f}, {pv.spearman_delta.hi_95:+.4f}])"
    )
    lines.append("")
    lines.append("Per-year breakdown (informational):")
    lines.append("")
    lines.append(pv.per_year_breakdown.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def _write_csv(verdicts: list[PositionVerdict], path: Path) -> None:
    """Long-form CSV: one row per (position, metric, year-or-pooled)."""
    rows: list[dict[str, object]] = []
    for pv in verdicts:
        rows.append(
            {
                "position": pv.position.value,
                "incumbent": pv.incumbent_class,
                "candidate": pv.candidate_class,
                "year": "pooled",
                "metric": "rmse_delta",
                "point": pv.rmse_delta.point,
                "lo_95": pv.rmse_delta.lo_95,
                "hi_95": pv.rmse_delta.hi_95,
                "n_paired": pv.rmse_delta.n_paired_rows,
                "verdict": pv.verdict,
                "reason": pv.reason,
            }
        )
        rows.append(
            {
                "position": pv.position.value,
                "incumbent": pv.incumbent_class,
                "candidate": pv.candidate_class,
                "year": "pooled",
                "metric": "spearman_delta",
                "point": pv.spearman_delta.point,
                "lo_95": pv.spearman_delta.lo_95,
                "hi_95": pv.spearman_delta.hi_95,
                "n_paired": pv.spearman_delta.n_paired_rows,
                "verdict": pv.verdict,
                "reason": pv.reason,
            }
        )
        for _, by in pv.per_year_breakdown.iterrows():
            for metric in ("rmse_delta", "spearman_delta"):
                rows.append(
                    {
                        "position": pv.position.value,
                        "incumbent": pv.incumbent_class,
                        "candidate": pv.candidate_class,
                        "year": int(by["year"]),
                        "metric": metric,
                        "point": by[f"{metric}_point"],
                        "lo_95": by[f"{metric}_lo"],
                        "hi_95": by[f"{metric}_hi"],
                        "n_paired": int(by["n_paired"]),
                        "verdict": "",
                        "reason": "",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the adoption-gate CLI arg namespace.

    Two mutually-exclusive modes:

    1. **Single-run** (``--run`` + ``--candidate``): pairs rows between two
       ``model_class`` values within one backtest run. Plan 8 mode.
    2. **Dual-run** (``--baseline-run`` + ``--candidate-run``): pairs rows
       across two backtest runs that share the same ``model_class`` but
       differ in feature set. Plan 9+ mode.
    """
    parser = argparse.ArgumentParser(description="Plan 8/9 adoption gate.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run",
        type=Path,
        default=None,
        help="single backtest run directory (Plan 8 mode); pair on model_class within this run",
    )
    mode.add_argument(
        "--baseline-run",
        type=Path,
        default=None,
        help=(
            "baseline (pre-feature) run directory; requires --candidate-run "
            "(Plan 9+ feature-comparison mode)"
        ),
    )
    parser.add_argument(
        "--candidate-run",
        type=Path,
        default=None,
        help="candidate (post-feature) run directory; required when --baseline-run is given",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="candidate model_class; required when --run is given",
    )
    parser.add_argument(
        "--incumbent",
        type=str,
        default="baseline",
        help="incumbent model_class (single-run mode only; default baseline)",
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
    args = parser.parse_args(argv)

    # argparse's mutually_exclusive_group only enforces "at most one of"; the
    # second-arg-of-each-pair invariant is enforced manually so the error
    # messages name the missing flag explicitly.
    if args.baseline_run is not None and args.candidate_run is None:
        parser.error("--baseline-run requires --candidate-run")
    if args.candidate_run is not None and args.baseline_run is None:
        parser.error("--candidate-run requires --baseline-run")
    if args.run is not None and args.candidate is None:
        parser.error("--run requires --candidate")
    return args


def main() -> None:
    args = parse_args()

    if args.baseline_run is not None:
        df = load_dual_run_paired(args.baseline_run, args.candidate_run)
        incumbent_class = DUAL_RUN_INCUMBENT_CLASS
        candidate_class = DUAL_RUN_CANDIDATE_CLASS
        run_label = f"baseline=`{args.baseline_run}` candidate=`{args.candidate_run}`"
    else:
        df = load_run_parquet(args.run)
        incumbent_class = args.incumbent
        candidate_class = args.candidate
        run_label = f"`{args.run}`"

    validate_model_classes_present(df, incumbent=incumbent_class, candidate=candidate_class)

    positions = (
        [Position.QB, Position.RB, Position.TE, Position.WR]
        if args.position == "all"
        else [Position(args.position)]
    )

    print(f"# Adoption gate report — {candidate_class} vs {incumbent_class}")
    print()
    print(f"Run: {run_label}")
    print(f"n_bootstrap: {args.n_bootstrap}, seed: {args.seed}")
    print()

    verdicts: list[PositionVerdict] = []
    for pos in positions:
        pv = evaluate_position(
            df,
            position=pos,
            incumbent=incumbent_class,
            candidate=candidate_class,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(format_position_report(pv))
        verdicts.append(pv)

    if args.csv_out is not None:
        _write_csv(verdicts, args.csv_out)
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
