"""Pool the hero-eval checkpoints across seasons and bootstrap paired diffs (scratch).

Reads the per-season hero checkpoints (_hero_{season}/), pairs each strategy against a
reference per (season, seat, seed) cell (CRN -> a true paired comparison), and reports
pooled WIN% / CHAMP% with bootstrap CIs plus the paired ΔWIN% / ΔCHAMP% vs the reference.
This is the cross-season aggregation the per-season `report` command does not do.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.backtest.hero_harness import consolidate_cells, load_hero_cells

_STRATS = (
    "raw_vorp",
    "now_or_never",
    "now_or_never_floored",
    "season_value",
    "season_value_var",
    "season_value_timing",
    "seat_aware",
)


def _ci(paired: np.ndarray, *, b: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(paired)
    means = paired[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(paired.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--seed-hi", type=int, default=25)
    ap.add_argument("--n-teams", type=int, default=16)
    ap.add_argument("--reference", default="now_or_never_floored")
    args = ap.parse_args(argv)

    frames = []
    for season in args.seasons:
        cells = load_hero_cells(
            seed_hi=args.seed_hi,
            strategies=_STRATS,
            season=season,
            n_teams=args.n_teams,
            checkpoint_dir=Path(f"_hero_{season}"),
        )
        frames.append(consolidate_cells(cells))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["scoring"] == "actual"].copy()
    df["win"] = df["wins"] / (df["wins"] + df["losses"])
    df["champ"] = df["is_champion"].astype(float)
    df["playoff"] = df["made_playoffs"].astype(float)

    keys = ["season", "seat", "seed"]
    win = df.pivot_table(index=keys, columns="strategy", values="win")
    champ = df.pivot_table(index=keys, columns="strategy", values="champ")
    playoff = df.pivot_table(index=keys, columns="strategy", values="playoff")

    n_cells = len(win)
    print(f"pooled over {args.seasons} | {n_cells} cells/strategy | ref={args.reference}\n")
    print(f"{'strategy':<22}{'WIN%':>7}{'PLAYOFF%':>10}{'CHAMP%':>9}   paired dWIN% [95% CI]")
    order = sorted(_STRATS, key=lambda s: -win[s].mean())
    for s in order:
        dwin = (win[s] - win[args.reference]).to_numpy()
        pt, lo, hi = _ci(dwin)
        sep = "*" if (lo > 0 or hi < 0) else " "
        tag = "  (ref)" if s == args.reference else ""
        body = (
            f"{100 * win[s].mean():>6.1f} {100 * playoff[s].mean():>9.1f} "
            f"{100 * champ[s].mean():>8.1f}"
        )
        diff = f"{100 * pt:+.2f} [{100 * lo:+.2f},{100 * hi:+.2f}] {sep}{tag}"
        print(f"{s:<22}{body}   {diff}")

    print(f"\n--- paired dCHAMP% vs {args.reference} (CI excludes 0 => *) ---")
    for s in sorted(_STRATS, key=lambda s: -champ[s].mean()):
        if s == args.reference:
            continue
        dch = (champ[s] - champ[args.reference]).to_numpy()
        pt, lo, hi = _ci(dch)
        sep = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {s:<22}{100 * pt:+.2f} [{100 * lo:+.2f},{100 * hi:+.2f}] {sep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
