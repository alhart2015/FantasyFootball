"""Waiver-wire / undrafted-pool assessment (spec 2026-07-17, issue #112).

Runs a hero + 15 constrained-ADP-bot 16-team draft over many seeds, then reports,
per skill position, how good the best still-available players are and how deep the
wire is (mean + 95% bootstrap CI). Analytic hero only (now_or_never[_floored] /
raw_vorp).

    python scripts/waiver_pool_assessment.py \
        --vorp-table data/vorp_2026/half_16team.parquet \
        --league-config data/vorp_2026/half_16team.league.json \
        --hero-strategy now_or_never_floored --seeds 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import bootstrap_mean
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverFlooredStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.backtest.draft_field import draft_mixed_field, hero_seat_layout
from projections.draft.backtest.waiver_pool import undrafted_pool_by_position
from projections.draft.league_config import LeagueConfig
from projections.schemas import VorpTableSchema

# (metric key, display header): single source of truth for the six metrics. Tuple order is
# the report's column order; run_assessment aggregates the same set (order-agnostic there).
_TABLE_COLS = (
    ("top1_vorp", "TOP1_VORP"),
    ("top2_vorp", "TOP2_VORP"),
    ("top3_vorp", "TOP3_VORP"),
    ("best_avail_proj_pts", "BEST_PROJ"),
    ("n_above_replacement", "#>REPL"),
    ("drain_rate", "DRAIN%"),
)
_METRIC_COLS = tuple(m for m, _ in _TABLE_COLS)
_ANALYTIC = ("now_or_never", "now_or_never_floored", "raw_vorp")


def _build_hero(key: str, n_teams: int) -> DraftStrategy:
    """Build an analytic hero strategy (no availability load) from its key."""
    surv = LogisticSurvival(sigma=default_sigma(n_teams))
    if key == "now_or_never_floored":
        return NowOrNeverFlooredStrategy(surv)
    if key == "now_or_never":
        return NowOrNeverStrategy(surv)
    if key == "raw_vorp":
        return RawVorpStrategy()
    raise ValueError(
        f"hero strategy {key!r} not supported in v1 (analytic keys only: {_ANALYTIC}; "
        f"MC strategies need availability wiring)"
    )


def _bootstrap_or_nan(vals: np.ndarray, *, seed: int) -> tuple[float, float, float]:
    """Bootstrap mean + 95% CI over the non-NaN values; all-NaN -> (nan, nan, nan).

    Dropping per-seed NaN keeps one fully-drained-position seed from collapsing the whole
    cell to NaN; the all-NaN guard avoids bootstrap_mean crashing on an empty array.
    """
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return float("nan"), float("nan"), float("nan")
    iv = bootstrap_mean(vals, seed=seed)
    return iv.point, iv.lo_95, iv.hi_95


def run_assessment(
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    hero: DraftStrategy,
    hero_seat: int,
    seeds: int,
    jitter: float,
    base_seed: int,
) -> pd.DataFrame:
    """Run `seeds` hero+bots drafts and aggregate per-(position, metric) to mean+95% CI.

    Returns a long-format frame: columns position, metric, mean, lo95, hi95 (one row per
    position x metric). Per-seed NaN is dropped before bootstrapping, so each metric is
    averaged over the seeds where it is defined; for a position that fully drains on some
    seeds (thin pools only -- never at 16-team), top*_vorp is then conditional on
    availability, so read it alongside drain% / # above-repl.
    """
    if seeds < 1:
        raise ValueError(f"seeds must be >= 1, got {seeds}")
    if "consensus_adp" not in pool.columns or pool["consensus_adp"].isna().all():
        raise ValueError(
            "pool has no populated consensus_adp -- the ADP bots in the draft sim need it. "
            "Pass a consensus-path VORP table (not a weekly-path one)."
        )
    layout = hero_seat_layout(hero_seat=hero_seat, hero_label="hero", n_teams=config.n_teams)
    seat_strategies: dict[int, DraftStrategy | None] = {
        s: (hero if lbl == "hero" else None) for s, lbl in layout.items()
    }
    per_seed: list[pd.DataFrame] = []
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        rosters = draft_mixed_field(seat_strategies, pool, config, rng=rng, jitter=jitter)
        per_seed.append(undrafted_pool_by_position(rosters, pool, config))

    stacked = pd.concat(per_seed, ignore_index=True)
    out_rows: list[dict[str, object]] = []
    for position, grp in stacked.groupby("position", sort=False):
        for metric in _METRIC_COLS:
            point, lo95, hi95 = _bootstrap_or_nan(grp[metric].to_numpy(dtype=float), seed=base_seed)
            out_rows.append(
                {"position": position, "metric": metric, "mean": point, "lo95": lo95, "hi95": hi95}
            )
    return pd.DataFrame(out_rows)


def format_assessment(agg: pd.DataFrame) -> str:
    """Per-position table sorted by mean top1_vorp desc; each metric as `mean [lo95, hi95]`.

    `drain_rate` is shown as a percentage. A NaN cell (a position drained in every seed,
    or a 0/0 drain_rate) renders as `nan [nan, nan]` — a real "undefined here" signal.
    """
    piv = {
        stat: agg.pivot(index="position", columns="metric", values=stat)
        for stat in ("mean", "lo95", "hi95")
    }
    order = piv["mean"].sort_values("top1_vorp", ascending=False).index

    def cell(pos: object, metric: str) -> str:
        scale = 100.0 if metric == "drain_rate" else 1.0
        m = float(piv["mean"].loc[pos, metric]) * scale
        lo = float(piv["lo95"].loc[pos, metric]) * scale
        hi = float(piv["hi95"].loc[pos, metric]) * scale
        return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"

    w = 24
    lines = ["POS   " + "".join(f"{h:<{w}}" for _, h in _TABLE_COLS)]
    for pos in order:
        cells = "".join(f"{cell(pos, m):<{w}}" for m, _ in _TABLE_COLS)
        lines.append(f"{pos!s:<6}{cells}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Waiver-wire / undrafted-pool assessment.")
    p.add_argument("--vorp-table", type=Path, default=Path("data/vorp_2026/half_16team.parquet"))
    p.add_argument(
        "--league-config", type=Path, default=Path("data/vorp_2026/half_16team.league.json")
    )
    p.add_argument("--hero-strategy", choices=list(_ANALYTIC), default="now_or_never_floored")
    p.add_argument("--hero-seat", type=int, default=1)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--jitter", type=float, default=8.0)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Optional path to write the table.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    pool = VorpTableSchema.validate(pd.read_parquet(args.vorp_table))
    hero = _build_hero(args.hero_strategy, config.n_teams)
    agg = run_assessment(
        pool,
        config,
        hero=hero,
        hero_seat=args.hero_seat,
        seeds=args.seeds,
        jitter=args.jitter,
        base_seed=args.base_seed,
    )
    text = format_assessment(agg)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
