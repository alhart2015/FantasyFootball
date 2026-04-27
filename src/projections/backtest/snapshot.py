"""Snapshot file IO + diff for the walk-forward gate.

Plan 3c Phase 4. Snapshot is a JSON list of
{"position", "year", "metric", "model_class", "value"} entries, sorted
lexicographically by (metric, position, year, model_class) so PR diffs
stay clean.

Plan 5 Task 13: rows acquire a ``model_class`` column (e.g. ``"baseline"``
or ``"lightgbm"``); the snapshot file is renamed
``baseline_metrics.json`` -> ``model_metrics.json``; row identity is the
4-tuple ``(position, year, metric, model_class)``. Cells may be present
for one model class but not another (e.g. ``season_calibration_*``
currently only emitted by the SAMPLED_SUMMARY-based baseline) — the diff
logic must tolerate that asymmetry.

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

# Plan 5 Task 13: snapshot rows now carry ``model_class`` so a single
# snapshot can hold metrics from multiple model classes side by side.
_SCHEMA_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "model_class", "value")


# Suffix-based metric -> tolerance-kind mapping. Rules are tried in
# declaration order; declare more-specific suffixes before more-general
# ones (e.g. "_mean_pred" before any future "_pred" rule). The keys are
# the tolerance-kind names; the values are tuples of substrings that, if
# present in the metric name, route the metric to this kind.
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
    """A single (position, year, metric, model_class) cell that failed the gate."""

    position: str
    year: int
    metric: str
    model_class: str
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
    (metric, position, year, model_class)."""
    if set(metrics.columns) != set(_SCHEMA_COLUMNS):
        raise ValueError(
            f"metrics must have columns {_SCHEMA_COLUMNS}, got {tuple(metrics.columns)}"
        )
    sorted_df = metrics.sort_values(["metric", "position", "year", "model_class"]).reset_index(
        drop=True
    )
    rows: list[dict[str, Any]] = []
    for _idx, row in sorted_df.iterrows():
        rows.append(
            {
                "position": str(row["position"]),
                "year": int(row["year"]),
                "metric": str(row["metric"]),
                "model_class": str(row["model_class"]),
                "value": float(row["value"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def read_snapshot(path: Path) -> pd.DataFrame:
    """Load a snapshot JSON file into a long-form metrics DataFrame.

    Defensive: if a row is missing the ``model_class`` field (e.g. a
    pre-Plan-5 snapshot snuck through), fill it with ``"baseline"`` so
    legacy snapshots load cleanly. Post-migration there should be no such
    rows in the committed file.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    for row in raw:
        row.setdefault("model_class", "baseline")
    return pd.DataFrame(raw, columns=list(_SCHEMA_COLUMNS))


def _check_one(
    *,
    position: str,
    year: int,
    metric: str,
    model_class: str,
    baseline_value: float,
    current_value: float,
    tolerance_kind: str,
    tolerance_value: float,
) -> Regression | None:
    """Apply direction-aware tolerance to a single cell. Returns None on
    pass; a Regression on fail."""
    cell_label = f"{position}/{year}/{metric}/{model_class}"
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
            model_class=model_class,
            baseline_value=baseline_value,
            current_value=current_value,
            direction="worse",
            tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{cell_label}: {baseline_value:.4f} -> {current_value:.4f} "
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
            model_class=model_class,
            baseline_value=baseline_value,
            current_value=current_value,
            direction="worse",
            tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{cell_label}: {baseline_value:.4f} -> {current_value:.4f} "
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
            model_class=model_class,
            baseline_value=baseline_value,
            current_value=current_value,
            direction="worse",
            tolerance_kind=tolerance_kind,
            tolerance_value=tolerance_value,
            message=(
                f"{cell_label}: {baseline_value:.4f} -> {current_value:.4f} "
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
    (position, year, metric, model_class); apply the override row's
    tolerance if present, otherwise apply the default tolerance for the
    metric kind. Returns a GateResult; ``passed`` is True iff no
    regressions found.

    A current-run row missing from baseline is itself a regression — the
    snapshot must be regenerated to include it intentionally. The reverse
    asymmetry (a baseline row with no current-run row) is intentionally
    permitted: e.g. ``season_calibration_*`` rows exist for the
    SAMPLED_SUMMARY-based baseline but not for LightGBM, so a baseline-only
    run will not produce LightGBM rows and vice versa. We only compare
    rows the current run actually emitted.

    Override rows may omit ``model_class`` for backward compatibility; an
    override without ``model_class`` applies to every model class for that
    (position, year, metric) cell.
    """
    overrides_with_class: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    overrides_any_class: dict[tuple[str, int, str], dict[str, Any]] = {}
    for o in overrides:
        if "model_class" in o:
            overrides_with_class[
                (str(o["position"]), int(o["year"]), str(o["metric"]), str(o["model_class"]))
            ] = o
        else:
            overrides_any_class[(str(o["position"]), int(o["year"]), str(o["metric"]))] = o
    baseline_index = {
        (
            str(r["position"]),
            int(r["year"]),
            str(r["metric"]),
            str(r["model_class"]),
        ): float(r["value"])
        for _, r in baseline.iterrows()
    }

    regressions: list[Regression] = []
    for _, row in current.iterrows():
        position = str(row["position"])
        year = int(row["year"])
        metric = str(row["metric"])
        model_class = str(row["model_class"])
        current_value = float(row["value"])

        key = (position, year, metric, model_class)
        if key not in baseline_index:
            regressions.append(
                Regression(
                    position=position,
                    year=year,
                    metric=metric,
                    model_class=model_class,
                    baseline_value=float("nan"),
                    current_value=current_value,
                    direction="missing",
                    tolerance_kind="<n/a>",
                    tolerance_value=float("nan"),
                    message=(f"{position}/{year}/{metric}/{model_class}: missing from baseline"),
                )
            )
            continue

        baseline_value = baseline_index[key]
        if key in overrides_with_class:
            ov = overrides_with_class[key]
            tol_kind = str(ov["tolerance_kind"])
            tol_value = float(ov["tolerance_value"])
        elif (position, year, metric) in overrides_any_class:
            ov = overrides_any_class[(position, year, metric)]
            tol_kind = str(ov["tolerance_kind"])
            tol_value = float(ov["tolerance_value"])
        else:
            tol_kind = _classify_metric(metric)
            tol_value = defaults[tol_kind]

        reg = _check_one(
            position=position,
            year=year,
            metric=metric,
            model_class=model_class,
            baseline_value=baseline_value,
            current_value=current_value,
            tolerance_kind=tol_kind,
            tolerance_value=tol_value,
        )
        if reg is not None:
            regressions.append(reg)

    return GateResult(passed=not regressions, regressions=regressions)
