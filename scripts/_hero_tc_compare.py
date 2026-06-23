"""Scratch: paired target-cap vs uncapped comparison (Test 18).

Uncapped cells in _hero_{season}/; target-capped (QB<=2,TE<=2,RB<=6,WR<=5) in
_hero_tc_{season}/. Same seeds -> CRN paired. Reports actual-scoring WIN%/PLAYOFF%/CHAMP%
means + paired bootstrap diffs per base strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projections.draft.assistant._compare import bootstrap_mean
from projections.draft.backtest.checkpoint import load_results

_BASES = ("now_or_never_floored", "season_value_timing")
_SEASONS = (2021, 2022, 2023, 2024, 2025)
_N_SEEDS = 25
_N_TEAMS = 16


def _metrics(ckpt: Path, strategy: str, seat: int, seed: int) -> tuple[float, float, float]:
    actual, _ = load_results(
        json.loads((ckpt / f"cell_{strategy}_{seat:02d}_{seed:05d}.json").read_text())
    )
    r = actual[0]
    return r.wins / (r.wins + r.losses), float(r.made_playoffs), float(r.is_champion)


def main() -> int:
    print(
        f"target-cap (QB2/TE2/RB6/WR5) vs uncapped, paired: {_SEASONS} x {_N_TEAMS} x {_N_SEEDS}\n"
    )
    for base in _BASES:
        unc = {"win": [], "po": [], "champ": []}
        cap = {"win": [], "po": [], "champ": []}
        for season in _SEASONS:
            u, c = Path(f"_hero_{season}"), Path(f"_hero_tc_{season}")
            for seat in range(1, _N_TEAMS + 1):
                for seed in range(_N_SEEDS):
                    for store, ck, strat in ((unc, u, base), (cap, c, f"{base}_targetcap")):
                        w, po, ch = _metrics(ck, strat, seat, seed)
                        store["win"].append(w)
                        store["po"].append(po)
                        store["champ"].append(ch)
        print(f"--- {base} ---")
        for m, label in (("win", "WIN%"), ("po", "PLAYOFF%"), ("champ", "CHAMP%")):
            uu, cc = np.array(unc[m]), np.array(cap[m])
            d = bootstrap_mean(cc - uu, seed=0)
            sep = "*" if (d.lo_95 > 0 or d.hi_95 < 0) else " "
            ci = f"[{100 * d.lo_95:+.2f},{100 * d.hi_95:+.2f}]"
            print(
                f"  {label:<9} uncapped={100 * uu.mean():5.1f}  capped={100 * cc.mean():5.1f}  "
                f"cap-unc={100 * d.point:+5.2f} {ci} {sep}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
