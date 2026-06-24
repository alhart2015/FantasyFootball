"""DFS edge study: comparable universe, disagreement head-to-head + ranking
skill, player-season clustered bootstrap, pre-registered single-primary gate,
ADOPT/STOP/INCONCLUSIVE verdict. See the design spec §6-§7.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projections.dfs import config
from projections.draft.assistant._compare import Interval

_KEY = ["gsis_id", "season", "week"]


def _disagreement(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["our_pts"] - df["sleeper_pts"]).abs() > config.DELTA]


def head_to_head_fraction(df: pd.DataFrame, *, target_col: str = "actual_points") -> float:
    """Share of disagreement cells where ours is strictly closer to `target_col`.
    Ties (equidistant) dropped from numerator and denominator. `target_col` is
    `actual_points` (base) for the primary metric, `actual_points_with_bonus`
    for the §6.2 sensitivity check."""
    sub = _disagreement(df)
    our_err = (sub["our_pts"] - sub[target_col]).abs()
    slp_err = (sub["sleeper_pts"] - sub[target_col]).abs()
    decisive = our_err != slp_err
    n = int(decisive.sum())
    if n == 0:
        return float("nan")
    return float((our_err[decisive] < slp_err[decisive]).sum()) / n


def _cluster_bootstrap(
    df: pd.DataFrame,
    group_cols: list[str],
    statistic: Callable[[pd.DataFrame], float],
    *,
    seed: int,
) -> Interval:
    """Percentile bootstrap of `statistic`, resampling whole groups (clusters)
    with replacement. A statistic that re-derives its own subset (e.g. the
    disagreement head-to-head) propagates threshold-boundary uncertainty inside
    each resample. Degenerate resamples (`statistic` → NaN) are dropped; if none
    survive, returns `Interval(point, NaN, NaN)`, which conservatively fails the
    gate."""
    groups = [g for _, g in df.groupby(group_cols)]
    rng = np.random.default_rng(seed)
    n = len(groups)
    boot = np.empty(config.N_BOOTSTRAP, dtype=np.float64)
    for b in range(config.N_BOOTSTRAP):
        pick = rng.integers(0, n, size=n)
        resampled = pd.concat([groups[i] for i in pick], ignore_index=True)
        boot[b] = statistic(resampled)
    boot = boot[~np.isnan(boot)]
    point = statistic(df)
    if boot.size == 0:
        return Interval(point=float(point), lo_95=float("nan"), hi_95=float("nan"))
    lo, hi = np.percentile(boot, (2.5, 97.5))
    return Interval(point=float(point), lo_95=float(lo), hi_95=float(hi))


def clustered_bootstrap_fraction(
    df: pd.DataFrame, *, seed: int, target_col: str = "actual_points"
) -> Interval:
    """Primary CI: resample player-season clusters (same-player serial corr)."""
    return _cluster_bootstrap(
        df, ["player_season"], lambda d: head_to_head_fraction(d, target_col=target_col), seed=seed
    )


def block_bootstrap_by_week(
    df: pd.DataFrame, *, seed: int, target_col: str = "actual_points"
) -> Interval:
    """Robustness CI: resample (season, week) blocks (cross-player same-game
    corr — an orthogonal source to player-season clustering, spec §7.2.3)."""
    return _cluster_bootstrap(
        df, ["season", "week"], lambda d: head_to_head_fraction(d, target_col=target_col), seed=seed
    )


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def ranking_skill_diff(df: pd.DataFrame) -> float:
    """Pooled Spearman(our, actual) - Spearman(sleeper, actual)."""
    return _spearman(df["our_pts"], df["actual_points"]) - _spearman(
        df["sleeper_pts"], df["actual_points"]
    )


def ranking_skill_diff_ci(df: pd.DataFrame, *, seed: int) -> Interval:
    """Pre-registered ranking-skill CI (spec §7.2.2/§7.3): resample player-season
    clusters and recompute the pooled Spearman difference per resample. The
    ranking-skill edge is gated on this CI's low side, not the point estimate."""
    return _cluster_bootstrap(df, ["player_season"], ranking_skill_diff, seed=seed)


def inclusion_disagreement(
    ours: pd.DataFrame, sleeper_pts: pd.DataFrame, *, usage: pd.DataFrame
) -> dict[str, int]:
    """Above the actual-usage floor, count cells only ONE source projects
    (the inclusion-disagreement diagnostic, spec §6.5/§7.1 — reported, not in
    the paired test)."""
    floor = usage[usage["touches_targets"].fillna(0) >= config.USAGE_FLOOR_TOUCHES_TARGETS]
    floor = floor[_KEY].drop_duplicates()
    o = set(map(tuple, floor.merge(ours[_KEY].drop_duplicates(), on=_KEY).to_numpy()))
    s = set(map(tuple, floor.merge(sleeper_pts[_KEY].drop_duplicates(), on=_KEY).to_numpy()))
    return {"ours_only": len(o - s), "sleeper_only": len(s - o), "both": len(o & s)}


def coverage_report(universe: pd.DataFrame) -> dict[str, int]:
    """Per-week-bucket cell counts of the final universe (spec §7.1/§5.3)."""

    def bucket(w: int) -> str:
        return "wk1_3" if w <= 3 else ("wk4_13" if w <= 13 else "wk14_18")

    counts = {"universe_cells": len(universe)}
    tagged = universe.assign(_b=universe["week"].map(bucket))
    for b, g in tagged.groupby("_b"):
        counts[f"universe_{b}"] = len(g)
    return counts


@dataclass(frozen=True)
class EdgeStudyResult:
    verdict: str  # "ADOPT" | "STOP" | "INCONCLUSIVE"
    primary: Interval
    byweek: Interval  # robustness (block-by-week) bootstrap
    sensitivity: Interval  # primary metric vs bonus-inclusive actuals
    ranking_diff: float
    ranking_diff_ci: Interval  # clustered-bootstrap CI of the ranking-skill diff
    n_clusters: int
    per_position_fraction: dict[str, float]
    equal_weight_fraction: float


def run_edge_study_from_universe(universe: pd.DataFrame) -> EdgeStudyResult:
    """Compute the pre-registered primary gate + robustness/sensitivity + verdict
    (home-grown-only vs Sleeper, pooled)."""
    sub = _disagreement(universe)
    n_clusters = int(sub["player_season"].nunique())
    primary = clustered_bootstrap_fraction(universe, seed=config.BOOTSTRAP_SEED)
    byweek = block_bootstrap_by_week(universe, seed=config.BOOTSTRAP_SEED)
    sensitivity = clustered_bootstrap_fraction(
        universe, seed=config.BOOTSTRAP_SEED, target_col="actual_points_with_bonus"
    )
    ranking_diff = ranking_skill_diff(universe)
    ranking_diff_ci = ranking_skill_diff_ci(universe, seed=config.BOOTSTRAP_SEED)

    per_pos = {pos: head_to_head_fraction(g) for pos, g in universe.groupby("position")}
    finite = [v for v in per_pos.values() if not np.isnan(v)]
    equal_weight = float(np.mean(finite)) if finite else float("nan")

    half_width = (
        (primary.hi_95 - primary.lo_95) / 2 if not np.isnan(primary.lo_95) else float("inf")
    )
    underpowered = n_clusters < config.N_MIN_CLUSTERS or half_width > config.TARGET_CI_HALFWIDTH

    edge_primary = (
        primary.lo_95 > 0.50
        and ranking_diff_ci.lo_95 >= 0  # NaN -> False (conservative; spec §7.2.2/§7.3)
        and all((np.isnan(v) or v >= 0.50 - config.MARGIN_M) for v in per_pos.values())
    )
    robust = byweek.lo_95 > 0.50  # by-week agrees on the edge
    sens_holds = sensitivity.lo_95 > 0.50  # bonus sensitivity does not flip

    if underpowered:
        verdict = "INCONCLUSIVE"
    elif edge_primary and robust and sens_holds:
        verdict = "ADOPT"
    elif edge_primary:
        verdict = "INCONCLUSIVE"  # primary edge but robustness/sensitivity disagree
    else:
        verdict = "STOP"

    return EdgeStudyResult(
        verdict=verdict,
        primary=primary,
        byweek=byweek,
        sensitivity=sensitivity,
        ranking_diff=ranking_diff,
        ranking_diff_ci=ranking_diff_ci,
        n_clusters=n_clusters,
        per_position_fraction=per_pos,
        equal_weight_fraction=equal_weight,
    )


def build_universe(
    ours: pd.DataFrame, sleeper_pts: pd.DataFrame, actuals: pd.DataFrame, *, usage: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join ours+sleeper+actuals on (gsis_id, season, week); position +
    actual_points (+ bonus-inclusive) from `actuals`; filter by the actual-usage
    floor in `usage` (columns gsis_id, season, week, touches_targets)."""
    df = (
        ours[[*_KEY, "our_pts"]]
        .merge(sleeper_pts[[*_KEY, "sleeper_pts"]], on=_KEY, how="inner")
        .merge(
            actuals[[*_KEY, "position", "actual_points", "actual_points_with_bonus"]],
            on=_KEY,
            how="inner",
        )
        .merge(usage[[*_KEY, "touches_targets"]], on=_KEY, how="left")
    )
    df = df[df["touches_targets"].fillna(0) >= config.USAGE_FLOOR_TOUCHES_TARGETS].copy()
    df["player_season"] = df["gsis_id"].astype(str) + "-" + df["season"].astype(str)
    return df
