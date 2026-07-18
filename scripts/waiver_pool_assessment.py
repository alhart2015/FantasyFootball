"""Waiver-wire / undrafted-pool assessment (spec 2026-07-17, issue #112).

Runs a hero + 15 constrained-ADP-bot 16-team draft over many seeds, then reports,
per skill position, how good the best still-available players are and how deep the
wire is (mean + 95% bootstrap CI). Analytic hero only (now_or_never[_floored] /
raw_vorp).

    python scripts/waiver_pool_assessment.py \
        --vorp-table data/vorp_2026/half_16team.parquet \
        --league-config data/vorp_2026/half_16team.league.json \
        --hero-strategy now_or_never_floored --seeds 200 --out reports/waiver_pool_2026.md
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

_METRIC_COLS = (
    "top1_vorp",
    "top2_vorp",
    "top3_vorp",
    "best_avail_proj_pts",
    "n_above_replacement",
    "drain_rate",
)
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

    Returns a long-format frame: columns position, metric, mean, lo95, hi95 (one row
    per position x metric).
    """
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
            iv = bootstrap_mean(grp[metric].to_numpy(dtype=float), seed=base_seed)
            out_rows.append(
                {
                    "position": position,
                    "metric": metric,
                    "mean": iv.point,
                    "lo95": iv.lo_95,
                    "hi95": iv.hi_95,
                }
            )
    return pd.DataFrame(out_rows)


def format_assessment(agg: pd.DataFrame) -> str:
    """Render the per-position table, positions sorted by mean top1_vorp descending."""
    wide = agg.pivot(index="position", columns="metric", values="mean")
    wide = wide.sort_values("top1_vorp", ascending=False)
    lines = ["POSITION  TOP1_VORP  TOP2   TOP3   BEST_PROJ  #>REPL  DRAIN%"]
    for position, r in wide.iterrows():
        lines.append(
            f"{position!s:<8}  {float(r['top1_vorp']):8.1f}  {float(r['top2_vorp']):5.1f}  "
            f"{float(r['top3_vorp']):5.1f}  {float(r['best_avail_proj_pts']):8.1f}  "
            f"{float(r['n_above_replacement']):5.1f}  {float(r['drain_rate']) * 100:5.1f}"
        )
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
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
