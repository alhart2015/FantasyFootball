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
from dataclasses import dataclass
from pathlib import Path

_VALID_POSITIONS = ("QB", "RB", "WR", "TE")
_VALID_MODELS = ("baseline", "lightgbm-nb")


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


def main(argv: list[str] | None = None) -> None:  # pragma: no cover — wired up in Task 3.1
    """Entry point. Implemented in Task 3.1."""
    raise NotImplementedError("main() implemented in Task 3.1")


if __name__ == "__main__":  # pragma: no cover
    main()
