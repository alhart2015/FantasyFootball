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
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from projections.backtest.adoption_gate import PositionVerdict
from projections.backtest.feature_probe import (
    PerStatVerdict,
    ProbeReport,
    _build_factory_with_columns,
    _loosened_features_schema,
    phase1_should_fire_phase2,
    probe_composite,
    probe_per_stat,
)
from projections.models import POSITION_DISPATCH
from projections.models.base import Model
from projections.models.baseline import BaselineModel
from projections.models.lightgbm import LightGBMModel
from projections.schemas import Position, Ruleset, Stat
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
    coverage_threshold: float


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
    p.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="Minimum non-null coverage of override candidate columns vs baseline rows. "
        "Default 0.95. Lower (e.g., 0.80) for overrides with structural NaN patterns "
        "like bye-week trailing-window features.",
    )
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
        coverage_threshold=ns.coverage_threshold,
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


def features_baseline_columns_set(
    joined: pd.DataFrame, production_columns: tuple[str, ...]
) -> set[str]:
    """Helper: which columns of the joined frame came from baseline (vs override)?

    ``production_columns`` is the canonical list of feature columns from
    ``POSITION_DISPATCH[position].factories[model].feature_columns``. Anything in
    ``joined`` that is also in ``production_columns`` is baseline; anything else
    (modulo identity columns) is override-supplied or schema-extra.
    """
    return set(joined.columns) & set(production_columns)


def _add_composite_fpts_column(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Vectorized composite fpts under ``ruleset`` — formula matches
    ``src/projections/scoring/score.py:score()`` exactly.

    Returns a copy of ``weekly_stats`` with an added ``fpts`` column. Missing
    stat columns are treated as zero (e.g., a QB-only frame has no
    ``receptions`` column — that contributes nothing to fpts).
    """
    out = weekly_stats.copy()

    def _col(name: str) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(0.0, index=out.index)

    fpts = (
        _col("passing_yards") / ruleset.passing_yds_per_pt
        + _col("passing_tds") * ruleset.passing_td_pts
        + _col("interceptions") * ruleset.interception_pts
        + _col("passing_2pt_conversions") * ruleset.two_pt_pts
        + _col("rushing_yards") / ruleset.rushing_yds_per_pt
        + _col("rushing_tds") * ruleset.rushing_td_pts
        + _col("rushing_2pt_conversions") * ruleset.two_pt_pts
        + _col("receptions") * ruleset.reception_pts
        + _col("receiving_yards") / ruleset.receiving_yds_per_pt
        + _col("receiving_tds") * ruleset.receiving_td_pts
        + _col("receiving_2pt_conversions") * ruleset.two_pt_pts
        + _col("fumbles_lost") * ruleset.fumble_lost_pts
        + _col("return_tds") * ruleset.return_td_pts
    )
    out["fpts"] = fpts.astype(np.float64)
    return out


def _get_production_columns_and_stats(
    model: Model,
) -> tuple[tuple[str, ...], tuple[Stat, ...]]:
    """Return ``(feature_columns, target_stats)`` from a production model
    instance, branching on the concrete model class.

    ``BaselineModel`` exposes both as direct dataclass attributes;
    ``LightGBMModel`` (and subclasses, including ``LightGBMNbModel``) stores
    ``feature_columns`` inside ``self._config`` and exposes ``target_stats``
    as a property — we read both via ``self._config`` for symmetry.

    Raises ``NotImplementedError`` for unknown model classes so the probe
    fails loud rather than silently misreading an attribute that doesn't exist.
    """
    if isinstance(model, BaselineModel):
        return model.feature_columns, model.target_stats
    if isinstance(model, LightGBMModel):
        return model._config.feature_columns, model._config.target_stats
    raise NotImplementedError(
        f"feature-column / target-stat extraction is not implemented for "
        f"{type(model).__name__}; extend _get_production_columns_and_stats "
        "to handle this model class."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seasons_range = range(args.seasons[0], args.seasons[1] + 1)
    holdout_years = tuple(range(args.holdout_years[0], args.holdout_years[1] + 1))

    # weekly_stats is shared across positions.
    weekly_stats_frames = [
        read_partition(Path("data/raw"), "weekly_stats", season=s) for s in seasons_range
    ]
    weekly_stats = pd.concat(weekly_stats_frames, ignore_index=True)

    phase1_all: list[PerStatVerdict] = []
    phase2_all: list[PositionVerdict] = []
    candidate_columns_per_position: dict[Position, tuple[str, ...]] = {}
    baseline_columns_per_position: dict[Position, tuple[str, ...]] = {}
    features_per_position: dict[Position, pd.DataFrame] = {}
    features_baseline_per_position: dict[Position, pd.DataFrame] = {}

    for pos_value in args.position:
        position = Position(pos_value)
        production_factory = POSITION_DISPATCH[position].factories[args.model]
        production_columns, target_stats = _get_production_columns_and_stats(production_factory())
        baseline_cols = tuple(c for c in production_columns if c not in args.drop)

        features_joined = load_features_with_overrides(
            position=pos_value,
            features_root=args.baseline_features,
            override_paths=args.override,
            drop_columns=args.drop,
            seasons=seasons_range,
            baseline_columns=production_columns,
            coverage_threshold=args.coverage_threshold,
        )

        # Determine added candidate columns (those that came from the overrides
        # and are not present in the baseline column list).
        if args.override:
            override_added = sorted(
                {
                    c
                    for c in features_joined.columns
                    if c not in features_baseline_columns_set(features_joined, production_columns)
                }
                - {"gsis_id", "season", "week", "team", "opponent", "position"}
                - set(args.drop)
            )
        else:
            override_added = []
        candidate_cols = baseline_cols + tuple(c for c in override_added if c not in baseline_cols)

        candidate_columns_per_position[position] = candidate_cols
        baseline_columns_per_position[position] = baseline_cols

        # For probe_per_stat, baseline frame uses baseline_cols (which is post-drop);
        # candidate frame uses candidate_cols.
        features_per_position[position] = features_joined
        # For Phase 2, both factories use the same joined frame; the column
        # selection is via feature_columns on each model instance.
        features_baseline_per_position[position] = features_joined

        phase1_all.extend(
            probe_per_stat(
                position=position,
                features_baseline_cols=baseline_cols,
                features_candidate_cols=candidate_cols,
                features=features_joined,
                weekly_stats=weekly_stats,
                target_stats=target_stats,
                holdout_years=holdout_years,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
        )

    fire_phase2 = phase1_should_fire_phase2(phase1_all) and args.composite

    if fire_phase2:
        # Derive composite fpts inline: production scoring/score.py is row-by-row;
        # here we vectorize it (matching score()'s formula) and add a `fpts` column
        # to weekly_stats. The default Ruleset() matches scoring/score.py defaults.
        ruleset = Ruleset()
        weekly_stats = _add_composite_fpts_column(weekly_stats, ruleset)
        composite_truth_col = "fpts"

        for pos_value in args.position:
            position = Position(pos_value)
            base_schema = POSITION_DISPATCH[position].feature_schema
            loose_schema = _loosened_features_schema(base_schema)
            factory_baseline = _build_factory_with_columns(
                position=position,
                model_class=args.model,
                columns=baseline_columns_per_position[position],
                feature_schema=loose_schema,
            )
            factory_candidate = _build_factory_with_columns(
                position=position,
                model_class=args.model,
                columns=candidate_columns_per_position[position],
                feature_schema=loose_schema,
            )
            verdict = probe_composite(
                position=position,
                factory_baseline=factory_baseline,
                factory_candidate=factory_candidate,
                features_baseline=features_baseline_per_position[position],
                features_candidate=features_per_position[position],
                weekly_stats=weekly_stats,
                composite_truth_column=composite_truth_col,
                holdout_years=holdout_years,
                ruleset=ruleset,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            phase2_all.append(verdict)

    report = ProbeReport(
        candidate_name=args.candidate_name,
        model_class=args.model,
        baseline_features_path=str(args.baseline_features),
        override_paths=tuple(str(p) for p in args.override),
        drop_columns=tuple(args.drop),
        phase1=phase1_all,
        phase2=phase2_all if fire_phase2 else None,
    )

    sys.stdout.write(render_markdown(report))
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        args.csv_out.write_text(render_csv(report), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
