# scripts/probe_feature_signal.py
"""Feature signal probe — CLI.

Pre-spec screening tool that takes a baseline feature set, applies a
candidate-column override, and emits per-stat Δ-CV-RMSE bootstrap CIs
(Phase 1) plus, conditionally on any Phase-1 SIGNAL, composite fpts ΔRMSE
(Phase 2) shaped identically to scripts/adoption_gate.py's output.

The probe answers "is there enough signal here to be worth scoping a full
feature plan around?" — it is NOT a substitute for the adoption gate, which
remains the final word on whether a feature change ships. A SIGNAL verdict
is necessary but not sufficient for shipping.

Usage (typical — augment-not-swap mode):

    python -m scripts.probe_feature_signal \\
        --candidate-name "augment_opp_epa_residual" \\
        --override data/features_probe/opp_epa_residual.parquet \\
        --csv-out reports/feature_probe_opp_epa_augment.csv

Usage (swap mode — adds an override AND drops an existing column):

    python -m scripts.probe_feature_signal \\
        --candidate-name "swap_opp_epa_residual" \\
        --override data/features_probe/opp_epa_residual.parquet \\
        --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4 \\
        --drop opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \\
        --csv-out reports/feature_probe_opp_epa_swap.csv

Usage (ablation mode — drops a column with no override):

    python -m scripts.probe_feature_signal \\
        --candidate-name "ablate_implied_team_total" \\
        --drop implied_team_total

Spec: docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md
"""

from __future__ import annotations

import argparse
import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from projections.backtest.feature_probe import PerStatVerdict, ProbeReport
from projections.store import read_partition

_VALID_POSITIONS = ("QB", "RB", "WR", "TE")
_VALID_MODELS = ("baseline", "lightgbm-nb")


class OverrideCoverageError(ValueError):
    """Raised when an override parquet's row coverage falls below the threshold."""


class OverrideCollisionError(ValueError):
    """Raised when an override column shadows a baseline column without --drop."""


@dataclass
class ProbeArgs:
    """Parsed CLI args. Extracted as a dataclass so tests can construct
    one directly without going through argparse."""

    candidate_name: str
    baseline_features: Path
    override: list[Path]
    drop: list[str]
    model: str
    position: list[str]
    seasons: tuple[int, int]
    holdout_years: tuple[int, int]
    n_bootstrap: int
    seed: int
    csv_out: Path | None
    composite: bool


def _parse_year_range(raw: str) -> tuple[int, int]:
    parts = raw.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"year range must be 'START-END' (e.g., '2018-2024'), got {raw!r}"
        )
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise argparse.ArgumentTypeError(f"start year {start} > end year {end}")
    return (start, end)


def _parse_drop_csv(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def parse_args(argv: list[str] | None = None) -> ProbeArgs:
    """Parse CLI argv. Extracted for testability — same pattern as
    scripts/adoption_gate.py's parse_args."""
    p = argparse.ArgumentParser(
        prog="probe_feature_signal",
        description="Feature signal probe — pre-spec screening for candidate feature columns.",
    )
    p.add_argument("--candidate-name", required=True, help="Label for the report header.")
    p.add_argument(
        "--baseline-features",
        type=Path,
        default=Path("data/features"),
        help="Root of the existing feature cache. Default: data/features.",
    )
    p.add_argument(
        "--override",
        type=Path,
        action="append",
        default=[],
        help="Override parquet (gsis_id, season, week, <candidate cols>). Repeatable.",
    )
    p.add_argument(
        "--drop",
        type=_parse_drop_csv,
        action="append",
        default=[],
        help="Comma-separated list of baseline feature columns to drop. Repeatable.",
    )
    p.add_argument(
        "--model",
        choices=_VALID_MODELS,
        default="baseline",
        help="Production model class to probe. Default: baseline.",
    )
    p.add_argument(
        "--position",
        choices=_VALID_POSITIONS,
        action="append",
        default=[],
        help="Position to probe. Repeatable. Default: all 4.",
    )
    p.add_argument(
        "--seasons",
        type=_parse_year_range,
        default=(2018, 2024),
        help="Season range as 'START-END'. Default: 2018-2024.",
    )
    p.add_argument(
        "--holdout-years",
        type=_parse_year_range,
        default=(2021, 2024),
        help="Held-out year range as 'START-END'. Default: 2021-2024.",
    )
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv-out", type=Path, default=None)
    composite_grp = p.add_mutually_exclusive_group()
    composite_grp.add_argument(
        "--composite",
        dest="composite",
        action="store_true",
        default=True,
        help="Run Phase 2 if Phase 1 fires it. Default.",
    )
    composite_grp.add_argument(
        "--no-composite",
        dest="composite",
        action="store_false",
        help="Skip Phase 2 even on a Phase-1 SIGNAL.",
    )

    ns = p.parse_args(argv)

    # Flatten action="append" lists for --drop (each invocation is a sub-list).
    drop_flat: list[str] = []
    for sub in ns.drop:
        drop_flat.extend(sub if isinstance(sub, list) else [sub])

    if not ns.override and not drop_flat:
        p.error(
            "must pass at least one of --override or --drop; a probe with no "
            "candidate transform is a no-op."
        )

    positions = ns.position if ns.position else list(_VALID_POSITIONS)

    return ProbeArgs(
        candidate_name=ns.candidate_name,
        baseline_features=ns.baseline_features,
        override=list(ns.override),
        drop=drop_flat,
        model=ns.model,
        position=positions,
        seasons=ns.seasons,
        holdout_years=ns.holdout_years,
        n_bootstrap=ns.n_bootstrap,
        seed=ns.seed,
        csv_out=ns.csv_out,
        composite=ns.composite,
    )


def validate_override_coverage(
    *,
    baseline: pd.DataFrame,
    joined: pd.DataFrame,
    candidate_columns: tuple[str, ...],
    threshold: float,
) -> None:
    """Raise OverrideCoverageError if the fraction of joined rows where ANY
    candidate column is non-null falls below ``threshold``.

    Coverage is computed against the baseline row count, not the joined count
    (so a missed join is a failure, not a silent drop).
    """
    n_baseline = len(baseline)
    if n_baseline == 0:
        raise OverrideCoverageError("baseline frame is empty — nothing to probe")
    for col in candidate_columns:
        if col not in joined.columns:
            raise OverrideCollisionError(
                f"override candidate column {col!r} not present after join — "
                f"likely a bad join key or empty override"
            )
        n_covered = int(joined[col].notna().sum())
        coverage = n_covered / n_baseline
        if coverage < threshold:
            raise OverrideCoverageError(
                f"candidate column {col!r}: only {coverage:.0%} of {n_baseline} "
                f"baseline rows have a non-null override value (threshold "
                f"{threshold:.0%}). Probe results would be biased by silent NaN "
                f"imputation; fix the override generator and retry."
            )


def load_features_with_overrides(
    *,
    position: str,
    features_root: Path,
    override_paths: Sequence[Path],
    drop_columns: Sequence[str],
    seasons: Iterable[int],
    baseline_columns: tuple[str, ...],
    coverage_threshold: float = 0.95,
) -> pd.DataFrame:
    """Load cached features for ``(position, seasons)``, left-join overrides
    on ``(gsis_id, season, week)``, validate coverage, and return the joined
    frame.

    The frame is NOT pandera-validated against the production FeaturesSchema
    here — the probe uses a loosened schema downstream so undeclared candidate
    columns survive. Schema validation happens inside ``probe_per_stat`` /
    ``probe_composite`` if at all.
    """
    table = position.lower()
    season_frames: list[pd.DataFrame] = []
    for season in seasons:
        season_frames.append(read_partition(features_root, table, season=season))
    if not season_frames:
        raise FileNotFoundError(
            f"No baseline features under {features_root}/{table}/ for the requested seasons."
        )
    baseline = pd.concat(season_frames, ignore_index=True)

    # Detect collisions before joining: every override column that is NOT
    # gsis_id/season/week must be either (a) absent from baseline, or
    # (b) listed in drop_columns.
    drop_set = set(drop_columns)
    candidate_columns: list[str] = []
    overrides_combined: pd.DataFrame | None = None
    for path in override_paths:
        ov = pd.read_parquet(path)
        for required in ("gsis_id", "season", "week"):
            if required not in ov.columns:
                raise ValueError(f"override {path}: missing required column {required!r}")
        ov_candidate_cols = [c for c in ov.columns if c not in ("gsis_id", "season", "week")]
        if not ov_candidate_cols:
            raise ValueError(f"override {path}: no candidate columns (only key columns present)")
        for col in ov_candidate_cols:
            if col in baseline.columns and col not in drop_set:
                raise OverrideCollisionError(
                    f"override column {col!r} collides with an existing baseline "
                    f"feature column. Either rename the override column, or "
                    f"add {col!r} to --drop to make the swap explicit."
                )
            if col in baseline_columns and col not in drop_set:
                # Same condition; double-check against the explicit baseline_columns
                # list passed by the caller (covers test cases where baseline.columns
                # contains identity cols beyond features).
                raise OverrideCollisionError(
                    f"override column {col!r} collides with baseline feature column. "
                    f"Add {col!r} to --drop to make the swap explicit."
                )
            candidate_columns.append(col)
        if overrides_combined is None:
            overrides_combined = ov
        else:
            overrides_combined = overrides_combined.merge(
                ov, on=["gsis_id", "season", "week"], how="outer"
            )

    # Drop dropped columns from the baseline before joining (a column listed
    # in drop is removed from baseline regardless of whether an override
    # replaces it).
    baseline_post_drop = baseline.drop(columns=[c for c in drop_columns if c in baseline.columns])

    if overrides_combined is None:
        # Drop-only mode: no overrides, just return baseline minus drops.
        return baseline_post_drop

    joined = baseline_post_drop.merge(
        overrides_combined, on=["gsis_id", "season", "week"], how="left"
    )

    validate_override_coverage(
        baseline=baseline,
        joined=joined,
        candidate_columns=tuple(candidate_columns),
        threshold=coverage_threshold,
    )
    return joined


def _format_paths_or_none(paths: tuple[str, ...]) -> str:
    return ", ".join(paths) if paths else "(none)"


def render_markdown(report: ProbeReport) -> str:
    """Render the probe report as markdown to a string."""
    lines: list[str] = []
    lines.append(f"# Feature signal probe — {report.candidate_name}")
    lines.append("")
    lines.append(f"Baseline features: {report.baseline_features_path}")
    lines.append(f"Overrides:        {_format_paths_or_none(report.override_paths)}")
    lines.append(f"Drops:            {_format_paths_or_none(report.drop_columns)}")
    lines.append(f"Model class:      {report.model_class}")
    lines.append("")
    lines.append("## Phase 1 — per-stat screening")
    lines.append("")

    by_pos: dict[str, list[PerStatVerdict]] = {}
    for v in report.phase1:
        by_pos.setdefault(v.position.value, []).append(v)

    for pos in ("QB", "RB", "WR", "TE"):
        if pos not in by_pos:
            continue
        lines.append(f"### {pos}")
        lines.append("")
        lines.append(
            "| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for v in by_pos[pos]:
            lines.append(
                f"| {v.stat.value} | {v.year_or_pooled} | {v.n_paired} | "
                f"{v.rmse_delta.point:+.4f} | {v.rmse_delta.lo_95:+.4f} | "
                f"{v.rmse_delta.hi_95:+.4f} | {v.r_squared_delta:+.4f} | {v.verdict} |"
            )
        lines.append("")

    n_signal = sum(1 for v in report.phase1 if v.verdict == "SIGNAL")
    n_null = sum(1 for v in report.phase1 if v.verdict == "NULL")
    n_regression = sum(1 for v in report.phase1 if v.verdict == "REGRESSION")
    n_total = len(report.phase1)
    lines.append("## Phase 1 verdict")
    lines.append("")
    lines.append(
        f"{n_signal}/{n_total} cells SIGNAL, {n_null}/{n_total} NULL, "
        f"{n_regression}/{n_total} REGRESSION."
    )

    if report.phase2 is None:
        if n_signal == 0:
            lines.append(
                "No SIGNAL cells — Phase 2 skipped. Probe predicts the adoption gate "
                "would return DO_NOT_ADOPT."
            )
        else:
            lines.append("Phase 2 disabled by --no-composite — composite verdict not computed.")
        lines.append("")
    else:
        lines.append("Phase 2 fired.")
        lines.append("")
        lines.append("## Phase 2 — composite ΔRMSE")
        lines.append("")
        lines.append(
            "| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |"
        )
        lines.append("|---|---|---|---|---:|")
        for pv in report.phase2:
            lines.append(
                f"| {pv.position.value} | {pv.verdict} | "
                f"{pv.rmse_delta.point:+.4f} ([{pv.rmse_delta.lo_95:+.4f}, "
                f"{pv.rmse_delta.hi_95:+.4f}]) | "
                f"{pv.spearman_delta.point:+.4f} ([{pv.spearman_delta.lo_95:+.4f}, "
                f"{pv.spearman_delta.hi_95:+.4f}]) | {pv.rmse_delta.n_paired_rows} |"
            )
        lines.append("")

    n_adopt = sum(1 for pv in (report.phase2 or []) if pv.verdict == "ADOPT")
    n_positions_run = len(report.phase2) if report.phase2 else 0
    lines.append("## Probe verdict")
    lines.append("")
    lines.append(f"Phase 1: {n_signal}/{n_total} cells SIGNAL.")
    if report.phase2 is None:
        lines.append("Phase 2: not run.")
    else:
        lines.append(f"Phase 2: {n_adopt}/{n_positions_run} positions ADOPT.")
    lines.append("")
    lines.append(
        "**This probe is a screen, not the gate.** The adoption gate is the final "
        "word on whether a feature change ships; SIGNAL is necessary but not "
        "sufficient. Run the full backtest + adoption gate before any production "
        "routing change."
    )
    lines.append("")
    return "\n".join(lines)


def render_csv(report: ProbeReport) -> str:
    """Render the probe report as a long-format CSV string.

    Columns: phase, position, stat_or_composite, year_or_pooled,
    metric_name, point, lo_95, hi_95, n_paired, verdict, reason.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "phase",
            "position",
            "stat_or_composite",
            "year_or_pooled",
            "metric_name",
            "point",
            "lo_95",
            "hi_95",
            "n_paired",
            "verdict",
            "reason",
        ]
    )
    for v in report.phase1:
        writer.writerow(
            [
                "phase1",
                v.position.value,
                v.stat.value,
                v.year_or_pooled,
                "rmse_delta",
                v.rmse_delta.point,
                v.rmse_delta.lo_95,
                v.rmse_delta.hi_95,
                v.n_paired,
                v.verdict,
                "",
            ]
        )
        writer.writerow(
            [
                "phase1",
                v.position.value,
                v.stat.value,
                v.year_or_pooled,
                "r_squared_delta",
                v.r_squared_delta,
                "",
                "",
                v.n_paired,
                v.verdict,
                "",
            ]
        )
    if report.phase2 is not None:
        for pv in report.phase2:
            writer.writerow(
                [
                    "phase2",
                    pv.position.value,
                    "composite",
                    "pooled",
                    "rmse_delta",
                    pv.rmse_delta.point,
                    pv.rmse_delta.lo_95,
                    pv.rmse_delta.hi_95,
                    pv.rmse_delta.n_paired_rows,
                    pv.verdict,
                    pv.reason,
                ]
            )
            writer.writerow(
                [
                    "phase2",
                    pv.position.value,
                    "composite",
                    "pooled",
                    "spearman_delta",
                    pv.spearman_delta.point,
                    pv.spearman_delta.lo_95,
                    pv.spearman_delta.hi_95,
                    pv.spearman_delta.n_paired_rows,
                    pv.verdict,
                    pv.reason,
                ]
            )
    return buf.getvalue()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover — wired up in Task 3.1
    """Entry point. Implemented in Task 3.1."""
    raise NotImplementedError("main() implemented in Task 3.1")


if __name__ == "__main__":  # pragma: no cover
    main()
