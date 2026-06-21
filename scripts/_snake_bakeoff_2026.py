"""Scratch: projected-H2H snake-draft bake-off on 2026 data (NOT committed).

Mirrors the auction multi-model bake-off (run_auction_tournament) but for the snake
draft: each strategy drafts as the sole hero vs a constrained noisy-ADP bot field, the
full league is reconstructed, and project_draft scores projected-vs-projected H2H
(reg_win%/playoff%/bye%/champ%) over n_sims MC seasons. CRN: the season RNG is shared
across strategies per seed, so the paired diffs are apples-to-apples.

No real actuals are used (2026 hasn't happened) -- this evaluates roster quality under
OUR projections + the variance model, exactly like the auction eval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import Interval, bootstrap_mean
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_projection import project_draft
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.assistant.simulation import _draft_picks
from projections.draft.assistant.strategy import STRATEGY_KEYS
from projections.draft.backtest.harness import _build_strategy
from projections.draft.league_config import LeagueConfig

METRICS = ("reg_win_pct", "make_playoffs_pct", "bye_pct", "champ_pct")
_SEASON_OFFSET = 1_000_000  # season RNG stream, disjoint from the draft stream


def _full_league(picks: list[str], n_teams: int) -> dict[int, list[str]]:
    """Group an absolute-order snake pick list into seat -> roster gsis_ids."""
    rosters: dict[int, list[str]] = {s: [] for s in range(1, n_teams + 1)}
    for i, gid in enumerate(picks):
        rosters[slot_for(i + 1, n_teams)].append(str(gid))
    return rosters


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vorp-table", type=Path, required=True)
    ap.add_argument("--league-config", type=Path, required=True)
    ap.add_argument("--my-slot", type=int, default=6)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument("--strategy-n-sims", type=int, default=50)
    ap.add_argument("--adp-jitter", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args(argv)

    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    pool = pd.read_parquet(args.vorp_table)
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)
    availability = load_store_availability(pool, season=args.season, data_root=args.data_root)
    params = VarianceParams.load()
    n_teams = config.n_teams

    keys = [k for k in STRATEGY_KEYS]
    per: dict[str, dict[str, np.ndarray]] = {
        k: {m: np.empty(args.seeds, dtype=np.float64) for m in METRICS} for k in keys
    }

    for key in keys:
        strat = _build_strategy(
            key,
            availability=availability,
            n_teams=n_teams,
            strategy_n_sims=args.strategy_n_sims,
            base_seed=args.seed,
        )
        assert strat is not None
        for s in range(args.seeds):
            picks = _draft_picks(
                strat,
                args.my_slot,
                pool,
                config,
                adp_jitter=args.adp_jitter,
                rng=np.random.default_rng(args.seed + s),
            )
            rosters = _full_league([str(p) for p in picks], n_teams)
            proj = project_draft(
                rosters,
                pool,
                availability,
                params,
                league_config=config,
                n_sims=args.n_sims,
                rng=np.random.default_rng(args.seed + _SEASON_OFFSET + s),  # CRN across strategies
            )
            sp = proj[args.my_slot]
            for m in METRICS:
                per[key][m][s] = float(getattr(sp, m))
        done = {m: float(per[key][m].mean()) for m in METRICS}
        print(f"[done] {key:<22} " + "  ".join(f"{m}={done[m]:.3f}" for m in METRICS), flush=True)

    summaries = {k: {m: bootstrap_mean(per[k][m], seed=args.seed) for m in METRICS} for k in keys}

    print(
        f"\n=== Snake bake-off (projected H2H) | {args.league_config.name} | "
        f"seat {args.my_slot} | {args.seeds} seeds x {args.n_sims} sims | "
        f"strat_n_sims={args.strategy_n_sims} | jitter={args.adp_jitter} ==="
    )
    hdr = f"{'STRATEGY':<22}" + "".join(f"{m:>20}" for m in METRICS)
    print(hdr)
    # rank by champ_pct desc
    order = sorted(keys, key=lambda k: summaries[k]["champ_pct"].point, reverse=True)
    for k in order:
        row = f"{k:<22}"
        for m in METRICS:
            iv: Interval = summaries[k][m]
            row += f"{iv.point:>9.3f} [{iv.lo_95:.3f},{iv.hi_95:.3f}]".rjust(20)
        print(row)

    print("\n--- paired diffs vs now_or_never (champ_pct, playoff_pct) ---")
    ref = "now_or_never"
    for k in keys:
        if k == ref:
            continue
        for m in ("make_playoffs_pct", "champ_pct"):
            iv = bootstrap_mean(per[k][m] - per[ref][m], seed=args.seed)
            star = "*" if (iv.lo_95 > 0 or iv.hi_95 < 0) else " "
            print(f"{k:<22} {m:<18} {iv.point:+.3f} [{iv.lo_95:+.3f},{iv.hi_95:+.3f}] {star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
