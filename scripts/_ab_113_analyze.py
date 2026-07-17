"""One-off analysis for issue #113: now_or_never_floored vs season_value H2H A/B.

Reads the chunked-runner checkpoints for a season, prints the harness marginal-CI
table (matching Test 10's convention), and additionally computes the CRN-paired
per-seed difference (mean of the 4 floored seats - mean of the 4 season_value
seats, per seed) with a percentile bootstrap CI. The paired diff is the tighter,
direct answer to "does the floor close now_or_never's gap to season_value?" because
both strategies share the same board/bot field within each seed (mirrored seats).

Not a shipped tool -- a throwaway analysis script kept alongside the run for
reproducibility. Usage:
    python scripts/_ab_113_analyze.py --checkpoint-dir _h2h_ckpt_113_2025 --season 2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projections.draft.assistant._compare import Interval, bootstrap_mean
from projections.draft.backtest.checkpoint import load_results
from projections.draft.backtest.cli import format_result
from projections.draft.backtest.harness import aggregate
from projections.draft.backtest.league import LeagueResult

FLOORED = "now_or_never_floored"
SV = "season_value"
N_TEAMS = 16


def _pool(checkpoint_dir: Path) -> tuple[list[LeagueResult], list[LeagueResult]]:
    """Pool all chunk files in seed order (chunk_LO_HI.json sorted by LO)."""
    chunks = sorted(checkpoint_dir.glob("chunk_*.json"))
    actual: list[LeagueResult] = []
    projected: list[LeagueResult] = []
    for c in chunks:
        a, p = load_results(json.loads(c.read_text()))
        actual += a
        projected += p
    return actual, projected


def _seat_metric(rs: list[LeagueResult], metric: str) -> float:
    """Mean of `metric` over a set of same-strategy seat rows within one seed."""
    if metric == "win_pct":
        vals = [r.wins / (r.wins + r.losses) for r in rs]
    elif metric == "playoff":
        vals = [1.0 if r.made_playoffs else 0.0 for r in rs]
    elif metric == "champ":
        vals = [1.0 if r.is_champion else 0.0 for r in rs]
    elif metric == "points_for":
        vals = [r.points_for for r in rs]
    else:
        raise ValueError(metric)
    return float(np.mean(vals))


def _paired_diffs(results: list[LeagueResult]) -> dict[str, np.ndarray]:
    """Per-seed paired diff arrays: floored - season_value, one value per seed.

    Consecutive N_TEAMS-row blocks are one seed (collect_results appends exactly
    n_teams rows per seed, chunks pool in seed order). Within a block, filter by
    strategy label -> 4 floored + 4 season_value + 8 bot.
    """
    if len(results) % N_TEAMS != 0:
        raise ValueError(f"pooled rows {len(results)} not a multiple of {N_TEAMS}")
    n_seeds = len(results) // N_TEAMS
    metrics = ("win_pct", "playoff", "champ", "points_for")
    out: dict[str, list[float]] = {m: [] for m in metrics}
    for s in range(n_seeds):
        block = results[s * N_TEAMS : (s + 1) * N_TEAMS]
        floored = [r for r in block if r.strategy == FLOORED]
        sv = [r for r in block if r.strategy == SV]
        if len(floored) != 4 or len(sv) != 4:
            raise ValueError(f"seed {s}: expected 4 floored + 4 sv, got {len(floored)} + {len(sv)}")
        for m in metrics:
            out[m].append(_seat_metric(floored, m) - _seat_metric(sv, m))
    return {m: np.array(v) for m, v in out.items()}


def _fmt(iv: Interval, *, pct: bool) -> str:
    if pct:
        return f"{iv.point * 100:+6.2f}pp  [{iv.lo_95 * 100:+.2f}, {iv.hi_95 * 100:+.2f}]"
    return f"{iv.point:+7.2f}    [{iv.lo_95:+.1f}, {iv.hi_95:+.1f}]"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--n-seeds", type=int, default=200)
    args = p.parse_args()

    actual, projected = _pool(args.checkpoint_dir)
    result = aggregate(actual, projected, n_seeds=args.n_seeds, base_seed=0)

    print(f"\n########## SEASON {args.season}  ({args.checkpoint_dir}) ##########\n")
    print(format_result(result))

    for axis_name, res in (("ACTUAL", actual), ("PROJECTED", projected)):
        diffs = _paired_diffs(res)
        print(f"\n[PAIRED per-seed diff  (floored - season_value)  -- {axis_name}]")
        print(f"  (n_seeds={len(res) // N_TEAMS}; >0 favors floored, CI excludes 0 = significant)")
        for m, label, pct in (
            ("win_pct", "win%   ", True),
            ("playoff", "playoff", True),
            ("champ", "champ% ", True),
            ("points_for", "pts_for", False),
        ):
            iv = bootstrap_mean(diffs[m], seed=0)
            print(f"    {label}: {_fmt(iv, pct=pct)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
