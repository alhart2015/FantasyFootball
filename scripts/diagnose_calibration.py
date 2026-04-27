"""Plan 3e Phase 0 — calibration diagnostic CLI.

Reads the most recent data/backtest/run_<ts>/results.parquet, computes
per-(position, stat) residual diagnostics for the held-out years,
fits 2-3 alternative families per cell with AIC ranking, and writes
structured artifacts to data/diagnostics/calibration_<ts>/.

Spec: docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    unpack_per_stat_params,
)
from projections.models import POSITION_DISPATCH
from projections.schemas import DistributionFamily, Stat


def find_latest_run_dir(backtest_root: Path) -> Path:
    """Return the most recent data/backtest/run_<ts>/ directory.

    Sorts lexicographically by directory name, which is correct because
    the timestamp format (YYYYMMDDTHHMMSSZ) sorts the same way as time.

    Raises:
        FileNotFoundError: backtest_root doesn't exist or contains no run_* subdirs.
    """
    if not backtest_root.is_dir():
        raise FileNotFoundError(f"Backtest root not found: {backtest_root}")
    candidates = sorted(
        p for p in backtest_root.iterdir() if p.is_dir() and p.name.startswith("run_")
    )
    if not candidates:
        raise FileNotFoundError(f"No run_<ts>/ subdirectories under {backtest_root}")
    return candidates[-1]


def load_per_row_results(run_dir: Path) -> pd.DataFrame:
    """Load `<run_dir>/results.parquet` produced by scripts/backtest.py.

    Returns the frame as-is (columns include identifiers, family, params,
    plus per-stat <stat>_pred / <stat>_actual columns whose names depend on
    which positions ran).

    Raises:
        FileNotFoundError: results.parquet missing under run_dir.
    """
    results_path = run_dir / "results.parquet"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"results.parquet missing under {run_dir}; run "
            f"`python scripts/backtest.py --report` to generate one."
        )
    return pd.read_parquet(results_path)


def _resolve_target_stats() -> dict[str, tuple[str, ...]]:
    """Return {position_value: (stat_value, ...)} for every position in
    POSITION_DISPATCH. The factory call is cheap (constructs a dataclass
    of constants); the resulting dict is built once per script invocation."""
    out: dict[str, tuple[str, ...]] = {}
    for position, dispatch in POSITION_DISPATCH.items():
        model = dispatch.factory()
        out[position.value] = tuple(s.value for s in model.target_stats)
    return out


def _classify_family(dist: Distribution) -> str:
    """Return the Plan 3d DistributionFamily.value matching the concrete class."""
    if isinstance(dist, ParametricNormal):
        return DistributionFamily.NORMAL.value
    if isinstance(dist, ParametricGamma):
        return DistributionFamily.GAMMA.value
    raise ValueError(f"Unrecognized distribution type: {type(dist).__name__}")


def _extract_assumed_params(dist: Distribution, family_name: str) -> tuple[float, float]:
    """Return (param_a, param_b) for the family. NORMAL packs std into a; GAMMA
    packs (shape, scale) into (a, b). Convention: param_b is NaN for one-param
    families."""
    if family_name == DistributionFamily.NORMAL.value:
        assert isinstance(dist, ParametricNormal)
        return float(dist.std()), float("nan")
    if family_name == DistributionFamily.GAMMA.value:
        assert isinstance(dist, ParametricGamma)
        return float(dist.shape), float(dist.scale)
    raise ValueError(f"Unhandled family: {family_name}")


def extract_per_stat_residuals(per_row: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide per-row results frame into a long-form residuals frame.

    For each row, look up the position's target stats via POSITION_DISPATCH
    and emit one output row per (position, stat) tuple where both
    `<stat>_pred` and `<stat>_actual` are present in the input. Pull the
    assumed-family parameters from the row's `params` blob via the existing
    Plan 3d codec.

    Output columns:
        position (str), stat (str), gsis_id (str), season (int), week (int),
        pred (float), actual (float), residual (= actual - pred),
        assumed_family (str: NORMAL or GAMMA),
        assumed_param_a (float: std for NORMAL, shape for GAMMA),
        assumed_param_b (float: NaN for NORMAL, scale for GAMMA)
    """
    target_stats = _resolve_target_stats()
    rows: list[dict[str, object]] = []
    for _idx, row in per_row.iterrows():
        position = str(row["position"])
        if position not in target_stats:
            continue  # K / DST / unknown — skip silently, not in scope
        per_stat_dists = unpack_per_stat_params(bytes(row["params"]))
        for stat_value in target_stats[position]:
            pred_col = f"{stat_value}_pred"
            actual_col = f"{stat_value}_actual"
            if pred_col not in per_row.columns or actual_col not in per_row.columns:
                continue
            pred = row[pred_col]
            actual = row[actual_col]
            if pd.isna(pred) or pd.isna(actual):
                continue
            stat_enum = Stat(stat_value)
            dist = per_stat_dists.get(stat_enum)
            if dist is None:
                continue
            family_name = _classify_family(dist)
            param_a, param_b = _extract_assumed_params(dist, family_name)
            rows.append(
                {
                    "position": position,
                    "stat": stat_value,
                    "gsis_id": str(row["gsis_id"]),
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "pred": float(pred),
                    "actual": float(actual),
                    "residual": float(actual) - float(pred),
                    "assumed_family": family_name,
                    "assumed_param_a": param_a,
                    "assumed_param_b": param_b,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Path to a data/backtest/run_<ts>/ directory. "
            "Defaults to the lexicographically-latest run_<ts>/ directory."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Path to write diagnostic artifacts. Defaults to data/diagnostics/calibration_<ts>/.",
    )
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
