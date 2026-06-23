"""Edge-study orchestrator + report writer. The CLI is a thin wrapper over this."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from projections.dfs.actuals import dk_weekly_actuals
from projections.dfs.blend import blend_statlines, sleeper_weekly_points
from projections.dfs.edge_study import (
    EdgeStudyResult,
    build_universe,
    coverage_report,
    inclusion_disagreement,
    run_edge_study_from_universe,
)
from projections.dfs.projections import emit_weekly_projections
from projections.dfs.usage import build_usage
from projections.schemas import Position, Ruleset, WeeklyStatsSchema
from projections.store import read_partition


@dataclass(frozen=True)
class StudyOutput:
    primary: EdgeStudyResult
    exploratory_blend: EdgeStudyResult
    inclusion: dict[str, int]
    coverage: dict[str, int]


def _load_sleeper(data_root: Path, seasons: list[int]) -> pd.DataFrame:
    """Concatenate the stored Sleeper weekly partitions across seasons. A
    season-only `read_partition` recurses into that season's `week=` partitions,
    so we don't enumerate weeks by hand; a fully-missing season is skipped."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frames.append(
                read_partition(data_root / "raw", "sleeper_weekly_projections", season=season)
            )
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError("no Sleeper weekly partitions found; run ingest-sleeper first")
    return pd.concat(frames, ignore_index=True)


def run_study(
    *,
    seasons: list[int],
    positions: list[Position],
    data_root: Path,
    features_root: Path,
    ruleset: Ruleset,
) -> StudyOutput:
    ours = emit_weekly_projections(
        seasons=seasons,
        positions=positions,
        features_root=features_root,
        raw_root=data_root / "raw",
        ruleset=ruleset,
    )
    sleeper_raw = _load_sleeper(data_root, seasons)
    sleeper_pts = sleeper_weekly_points(sleeper_raw, ruleset=ruleset)

    raw_actuals = WeeklyStatsSchema.validate(
        pd.concat(
            [read_partition(data_root / "raw", "weekly_stats", season=s) for s in seasons],
            ignore_index=True,
        )
    )
    actuals = dk_weekly_actuals(raw_actuals, ruleset=ruleset)
    usage = build_usage(raw_actuals)

    universe = build_universe(ours, sleeper_pts, actuals, usage=usage)
    primary = run_edge_study_from_universe(universe)

    blend = blend_statlines(ours, sleeper_raw, weight_ours=0.5, ruleset=ruleset)
    blend_universe = build_universe(
        blend.rename(columns={"blended_pts": "our_pts"}), sleeper_pts, actuals, usage=usage
    )
    exploratory = run_edge_study_from_universe(blend_universe)

    return StudyOutput(
        primary=primary,
        exploratory_blend=exploratory,
        inclusion=inclusion_disagreement(ours, sleeper_pts, usage=usage),
        coverage=coverage_report(universe),
    )


def write_report(path: Path, out: StudyOutput, *, seasons: list[int]) -> None:
    p = out.primary
    lines = [
        f"# DFS Projection Edge Study — verdict "
        f"({'-'.join(map(str, (min(seasons), max(seasons))))})",
        "",
        f"**VERDICT: {p.verdict}**",
        "",
        "## Primary test (home-grown-only vs Sleeper, pooled, DK base)",
        f"- head-to-head fraction: {p.primary.point:.3f} "
        f"(95% CI {p.primary.lo_95:.3f} to {p.primary.hi_95:.3f}), clustered by player-season",
        f"- by-week robustness CI: {p.byweek.lo_95:.3f} to {p.byweek.hi_95:.3f}",
        f"- bonus-sensitivity CI (actuals+bonus): "
        f"{p.sensitivity.lo_95:.3f} to {p.sensitivity.hi_95:.3f}",
        f"- ranking-skill diff (Spearman, ours-Sleeper): {p.ranking_diff:.3f} "
        f"(95% CI {p.ranking_diff_ci.lo_95:.3f} to {p.ranking_diff_ci.hi_95:.3f}), "
        f"clustered by player-season",
        f"- disagreement clusters (player-seasons): {p.n_clusters}",
        f"- pooled (count-weighted) {p.primary.point:.3f} vs equal-weight "
        f"{p.equal_weight_fraction:.3f}",
        "",
        "## Per-position (EXPLORATORY — non-confirmatory)",
        *[f"- {pos}: {frac:.3f}" for pos, frac in sorted(p.per_position_fraction.items())],
        "",
        "## Exploratory 50/50 blend (non-confirmatory)",
        f"- verdict {out.exploratory_blend.verdict}; "
        f"fraction {out.exploratory_blend.primary.point:.3f} "
        f"({out.exploratory_blend.primary.lo_95:.3f} to "
        f"{out.exploratory_blend.primary.hi_95:.3f})",
        "",
        "## Coverage & inclusion disagreement",
        f"- inclusion: {out.inclusion}",
        f"- coverage: {out.coverage}",
        "",
        "## Limitations",
        "- Sleeper-alone is a softer proxy than the true DFS field (necessary, not "
        "sufficient — spec §4.3/§6.1). Bonuses excluded from the projection comparison "
        "(conservative; spec §6.2).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
