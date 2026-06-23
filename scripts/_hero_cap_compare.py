"""Scratch: paired capped-vs-uncapped comparison for the QB<=2 season_value family.

Uncapped cells live in _hero_{season}/ (the existing N=25 run); capped cells in
_hero_cap_{season}/. Same seeds -> same league seed -> identical bots+schedule (CRN), so a
cell-level (capped - uncapped) diff is a true paired comparison. Reports actual-scoring
WIN% / PLAYOFF% / CHAMP% means + paired bootstrap diffs per base strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projections.draft.assistant._compare import bootstrap_mean
from projections.draft.backtest.checkpoint import load_results

_BASES = ("season_value", "season_value_var", "season_value_timing")
_SEASONS = (2021, 2022, 2023, 2024, 2025)
_N_SEEDS = 25
_N_TEAMS = 16


def _metrics(ckpt: Path, strategy: str, seat: int, seed: int) -> tuple[float, float, float]:
    """(win_pct, made_playoffs, is_champion) from a cell's ACTUAL result."""
    actual, _ = load_results(
        json.loads((ckpt / f"cell_{strategy}_{seat:02d}_{seed:05d}.json").read_text())
    )
    r = actual[0]
    return r.wins / (r.wins + r.losses), float(r.made_playoffs), float(r.is_champion)


def main() -> int:
    print(f"capped (QB<=2) vs uncapped, paired: {_SEASONS} x {_N_TEAMS} seats x {_N_SEEDS} seeds\n")
    for base in _BASES:
        unc = {"win": [], "po": [], "champ": []}
        cap = {"win": [], "po": [], "champ": []}
        for season in _SEASONS:
            u_dir, c_dir = Path(f"_hero_{season}"), Path(f"_hero_cap_{season}")
            for seat in range(1, _N_TEAMS + 1):
                for seed in range(_N_SEEDS):
                    uw, upo, uch = _metrics(u_dir, base, seat, seed)
                    cw, cpo, cch = _metrics(c_dir, f"{base}_qbcap2", seat, seed)
                    unc["win"].append(uw)
                    unc["po"].append(upo)
                    unc["champ"].append(uch)
                    cap["win"].append(cw)
                    cap["po"].append(cpo)
                    cap["champ"].append(cch)
        print(f"--- {base} ---")
        for m, label in (("win", "WIN%"), ("po", "PLAYOFF%"), ("champ", "CHAMP%")):
            u = np.array(unc[m])
            c = np.array(cap[m])
            diff = bootstrap_mean(c - u, seed=0)
            sep = "*" if (diff.lo_95 > 0 or diff.hi_95 < 0) else " "
            ci = f"[{100 * diff.lo_95:+.2f},{100 * diff.hi_95:+.2f}]"
            print(
                f"  {label:<9} uncapped={100 * u.mean():5.1f}  capped={100 * c.mean():5.1f}  "
                f"cap-unc={100 * diff.point:+5.2f} {ci} {sep}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
