"""Post-draft assessment: projected points-for vs actual H2H outcomes from backtest checkpoints.

Reads the per-(seed, seat) team rows from one or more H2H backtest checkpoint directories (each a
set of ``chunk_*.json`` written by ``scripts/h2h_backtest_chunked.py``) and reports, per season and
combined:

  1. corr(projected season points-for, actual H2H win%) with bootstrap CIs, for season_value
     drafters and for all teams,
  2. the post-draft assessment for a season_value draft (champ% / playoff% / expected wins ± CI),
  3. outcomes binned by projected-PF quintile (the "given my team's projected PF, what are my
     odds" lookup), combined across seasons.

The default checkpoint root is the committed 2021-2025 blended-basis run that backs
``reports/post_draft_assessment_2021_2025.md``. Point ``--checkpoint-root`` / ``--seasons`` at any
other run to reproduce the analysis on fresh checkpoints.

Usage:
    python scripts/post_draft_assessment.py
    python scripts/post_draft_assessment.py --checkpoint-root <dir> --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import numpy.typing as npt

_DEFAULT_ROOT = Path("reports/data/post_draft_2021_2025")
_DEFAULT_SEASONS = ("2021", "2022", "2023", "2024", "2025")
_SEATS_PER_LEAGUE = 16  # rows are grouped 16-per-seed; only used to walk chunk payloads in order


def load(dirs: list[Path]) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float64]]:
    """Load (strategy, projPF, win%, wins, champ, playoff) per team across the given dirs."""
    rows: list[tuple[str, float, float, float, float, float]] = []
    for d in dirs:
        for f in sorted(glob(f"{d}/chunk_*.json")):
            payload = json.loads(Path(f).read_text())
            actual, projected = payload["actual"], payload["projected"]
            for i in range(0, len(actual), _SEATS_PER_LEAGUE):
                window = zip(
                    actual[i : i + _SEATS_PER_LEAGUE],
                    projected[i : i + _SEATS_PER_LEAGUE],
                    strict=True,
                )
                for a, p in window:
                    games = a["wins"] + a["losses"]
                    rows.append(
                        (
                            a["strategy"],
                            float(p["points_for"]),
                            a["wins"] / games,
                            float(a["wins"]),
                            1.0 if a["is_champion"] else 0.0,
                            1.0 if a["made_playoffs"] else 0.0,
                        )
                    )
    strat = np.array([r[0] for r in rows])
    metrics = np.array([r[1:] for r in rows], dtype=float)  # projPF, winpct, wins, champ, playoff
    return strat, metrics


def boot_ci(
    v: npt.NDArray[np.float64], b: int = 20000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of v with a percentile bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), (b, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def corr_ci(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], b: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Pearson r(x, y) with a paired percentile bootstrap 95% CI."""
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (b, len(x)))
    bs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint-root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--seasons", nargs="+", default=list(_DEFAULT_SEASONS))
    args = ap.parse_args()

    seasons: dict[str, Path] = {s: args.checkpoint_root / s for s in args.seasons}
    grouped: list[tuple[str, list[Path]]] = [(s, [d]) for s, d in seasons.items()]
    everything: list[tuple[str, list[Path]]] = [*grouped, ("combined", list(seasons.values()))]

    print("=== corr(projected PF, actual win%) ===")
    for name, dirs in everything:
        strat, m = load(dirs)
        if len(m) == 0:
            print(f"  {name:<9} (no chunks found in {dirs})")
            continue
        for who, mask in (
            ("season_value", strat == "season_value"),
            ("all teams", np.ones(len(strat), bool)),
        ):
            r, lo, hi = corr_ci(m[mask, 0], m[mask, 1])
            print(
                f"  {name:<9} {who:<13} r={r:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                f"r^2={r * r:.2f}  (n={int(mask.sum())})"
            )

    print("\n=== If you draft season_value (post-draft assessment) ===")
    for name, dirs in everything:
        strat, m = load(dirs)
        if len(m) == 0:
            continue
        mask = strat == "season_value"
        c, cl, ch = boot_ci(m[mask, 3])
        p, pl, ph = boot_ci(m[mask, 4])
        w, wl, wh = boot_ci(m[mask, 2])
        print(
            f"  {name:<9} champ {c * 100:4.1f}% [{cl * 100:.1f},{ch * 100:.1f}]   "
            f"playoff {p * 100:4.1f}% [{pl * 100:.1f},{ph * 100:.1f}]   "
            f"wins {w:.1f} [{wl:.1f},{wh:.1f}]"
        )

    strat, m = load(list(seasons.values()))
    if len(m) == 0:
        return
    proj_pf, wins, champ, playoff = m[:, 0], m[:, 2], m[:, 3], m[:, 4]
    print(f"\n=== Outcomes by projected-PF quintile (combined, n={len(m)}) ===")
    print(f"{'projPF range':<15}{'champ%':>20}{'playoff%':>20}{'exp wins/14':>18}")
    qs = np.percentile(proj_pf, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        # top bin is closed on the right so the max value lands in a bin
        upper = proj_pf <= qs[i + 1] if i == 4 else proj_pf < qs[i + 1]
        sel = (proj_pf >= qs[i]) & upper
        c, cl, ch = boot_ci(champ[sel])
        p, pl, ph = boot_ci(playoff[sel])
        w, wl, wh = boot_ci(wins[sel])
        print(
            f"{qs[i]:>6.0f}-{qs[i + 1]:<6.0f}{c * 100:>8.1f}% [{cl * 100:.1f},{ch * 100:.1f}]"
            f"{p * 100:>8.1f}% [{pl * 100:.1f},{ph * 100:.1f}]{w:>8.1f} [{wl:.1f},{wh:.1f}]"
        )


if __name__ == "__main__":
    main()
