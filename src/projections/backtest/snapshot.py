"""Snapshot file IO + diff for the walk-forward gate.

Plan 3c Phase 4. Snapshot is a JSON list of
{"position", "year", "metric", "value"} entries, sorted lexicographically
by (metric, position, year) so PR diffs stay clean.

Tolerance application is direction-aware. Spec section 2.4 lists the
mapping; this module owns the suffix-based metric -> tolerance-kind
registry and the comparison logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

_SCHEMA_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


# Suffix-based metric -> tolerance-kind mapping. Order matters: longer
# suffixes are tried first so "_mean_pred" matches before a hypothetical
# "_pred". The keys are the tolerance kind names; the values are tuples of
# substrings that, if present in the metric name, route the metric to
# this kind.
_METRIC_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mean_pred_relative", ("_mean_pred",)),
    ("rmse_relative", ("_rmse",)),
    ("mae_relative", ("_mae",)),
    ("spearman_absolute", ("spearman_",)),
    ("calibration_absolute", ("calibration_",)),
)


def _classify_metric(metric: str) -> str:
    """Return the tolerance-kind name for a metric. Raises if no rule
    matches — fail-closed for unknown metric names."""
    for kind, needles in _METRIC_KIND_RULES:
        for n in needles:
            if n in metric:
                return kind
    raise ValueError(
        f"unknown tolerance kind for metric {metric!r}; add a rule to "
        f"_METRIC_KIND_RULES in projections/backtest/snapshot.py"
    )


@dataclass(frozen=True, slots=True)
class Regression:
    """A single (position, year, metric) cell that failed the gate."""

    position: str
    year: int
    metric: str
    baseline_value: float
    current_value: float
    direction: str  # "worse", "better", "missing", "unknown"
    tolerance_kind: str
    tolerance_value: float
    message: str


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of comparing a current run's metrics against a baseline snapshot."""

    passed: bool
    regressions: list[Regression] = field(default_factory=list)


def write_snapshot(metrics: pd.DataFrame, path: Path) -> None:
    """Serialize a long-form metrics DataFrame to JSON, sorted by
    (metric, position, year)."""
    if set(metrics.columns) != set(_SCHEMA_COLUMNS):
        raise ValueError(
            f"metrics must have columns {_SCHEMA_COLUMNS}, got {tuple(metrics.columns)}"
        )
    sorted_df = metrics.sort_values(["metric", "position", "year"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _idx, row in sorted_df.iterrows():
        rows.append(
            {
                "position": str(row["position"]),
                "year": int(row["year"]),
                "metric": str(row["metric"]),
                "value": float(row["value"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def read_snapshot(path: Path) -> pd.DataFrame:
    """Load a snapshot JSON file into a long-form metrics DataFrame."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(raw, columns=list(_SCHEMA_COLUMNS))


def _check_one(
    *,
    position: str,
    year: int,
    metric: str,
    baseline_value: float,
    current_value: float,
    tolerance_kind: str,
    tolerance_value: float,
) -> Regression | None:
    """Apply direction-aware tolerance to a single cell. Returns None on
    pass; a Regression on fail."""
    if tolerance_kind in {"rmse_relative", "mae_relative"}:
        # RMSE/MAE worse = larger.
        if current_value <= baseline_value:
            return None
        rel = (current_value - baseline_value) / max(baseline_value, 1e-12)
        if rel <= tolerance_value:
            return None
        return Regression(
            position=position,
            year=year,
            metric=metric,
            baseline_value=baseline_value,
            current_value=current_value,
            direction="worse",
            tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{position}/{year}/{metric}: {baseline_value:.4f} -> {current_value:.4f} "
                f"({rel:+.2%} > {tolerance_value:+.2%})"
            ),
        )

    if tolerance_kind == "spearman_absolute":
        # Spearman worse = smaller.
        if current_value >= baseline_value:
            return None
        delta = baseline_value - current_value
        if delta <= tolerance_value:
            return None
        return Regression(
            position=position,
            year=year,
            metric=metric,
            baseline_value=baseline_value,
            current_value=current_value,
            direction="worse",
            tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{position}/{year}/{metric}: {baseline_value:.4f} -> {current_value:.4f} "
                f"(drop {delta:.4f} > {tolerance_value:.4f})"
            ),
        )

    if tolerance_kind in {"calibration_absolute", "mean_pred_relative"}:
        # Drift in either direction past tolerance is a regression.
        if tolerance_kind == "calibration_absolute":
            delta = abs(current_value - baseline_value)
            if delta <= tolerance_value:
                return None
        else:  # mean_pred_relative
            rel = abs(current_value - baseline_value) / max(abs(baseline_value), 1e-12)
            if rel <= tolerance_value:
                return None
            delta = rel
        return Regression(
            position=position,
            year=year,
            metric=metric,
            baseline_value=baseline_value,
            current_value=current_value,
            direction="worse",
            tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{position}/{year}/{metric}: {baseline_value:.4f} -> {current_value:.4f} "
                f"(drift {delta:.4f} > {tolerance_value:.4f})"
            ),
        )

    raise ValueError(f"unknown tolerance_kind {tolerance_kind!r}")


def diff_snapshot(
    *,
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    defaults: dict[str, float],
    overrides: list[dict[str, Any]],
) -> GateResult:
    """Compare a current run's metrics against a baseline snapshot.

    For each row in ``current``, look up the matching ``baseline`` row by
    (position, year, metric); apply the override row's tolerance if
    present, otherwise apply the default tolerance for the metric kind.
    Returns a GateResult; ``passed`` is True iff no regressions found.

    A current-run row missing from baseline is itself a regression — the
    snapshot must be regenerated to include it intentionally.
    """
    overrides_index = {(o["position"], int(o["year"]), o["metric"]): o for o in overrides}
    baseline_index = {
        (str(r["position"]), int(r["year"]), str(r["metric"])): float(r["value"])
        for _, r in baseline.iterrows()
    }

    regressions: list[Regression] = []
    for _, row in current.iterrows():
        position = str(row["position"])
        year = int(row["year"])
        metric = str(row["metric"])
        current_value = float(row["value"])

        key = (position, year, metric)
        if key not in baseline_index:
            regressions.append(
                Regression(
                    position=position,
                    year=year,
                    metric=metric,
                    baseline_value=float("nan"),
                    current_value=current_value,
                    direction="missing",
                    tolerance_kind="<n/a>",
                    tolerance_value=float("nan"),
                    message=f"{position}/{year}/{metric}: missing from baseline",
                )
            )
            continue

        baseline_value = baseline_index[key]
        if key in overrides_index:
            ov = overrides_index[key]
            tol_kind = str(ov["tolerance_kind"])
            tol_value = float(ov["tolerance_value"])
        else:
            tol_kind = _classify_metric(metric)
            tol_value = defaults[tol_kind]

        reg = _check_one(
            position=position,
            year=year,
            metric=metric,
            baseline_value=baseline_value,
            current_value=current_value,
            tolerance_kind=tol_kind,
            tolerance_value=tol_value,
        )
        if reg is not None:
            regressions.append(reg)

    return GateResult(passed=not regressions, regressions=regressions)
