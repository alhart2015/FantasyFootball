"""One-off: per-position significance breakdown of the DFS edge study.

Rebuilds the 2021-2024 comparable universe exactly as `run_study` does (persists
it to data/ so re-analysis is instant), reproduces the pooled primary fraction as
a sanity check, then for each position reports the disagreement head-to-head
fraction with a player-season clustered-bootstrap 95% CI + cell/cluster counts.

Reuses the shipped harness (same DELTA, same clustered bootstrap) so the numbers
are methodologically identical to the verdict. Throwaway — delete after use.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.dfs import config
from projections.dfs.actuals import dk_weekly_actuals
from projections.dfs.blend import sleeper_weekly_points
from projections.dfs.edge_study import (
    _disagreement,
    build_universe,
    clustered_bootstrap_fraction,
)
from projections.dfs.projections import emit_weekly_projections
from projections.dfs.usage import build_usage
from projections.draft.assistant._compare import Interval
from projections.schemas import Position, Ruleset, WeeklyStatsSchema
from projections.store import read_partition

DATA_ROOT = Path("data")
FEATURES_ROOT = DATA_ROOT / "features"
SEASONS = [2021, 2022, 2023, 2024]
POSITIONS = [Position.QB, Position.RB, Position.WR, Position.TE]
UNIVERSE_CACHE = DATA_ROOT / "dfs_universe_2021-2024.parquet"


def build_or_load_universe() -> pd.DataFrame:
    if UNIVERSE_CACHE.exists():
        print(f"[load] cached universe {UNIVERSE_CACHE}", flush=True)
        return pd.read_parquet(UNIVERSE_CACHE)
    ruleset = Ruleset.draftkings()
    print("[build] emit_weekly_projections (this is the slow step)...", flush=True)
    ours = emit_weekly_projections(
        seasons=SEASONS,
        positions=POSITIONS,
        features_root=FEATURES_ROOT,
        raw_root=DATA_ROOT / "raw",
        ruleset=ruleset,
    )
    sleeper_raw = pd.concat(
        [
            read_partition(DATA_ROOT / "raw", "sleeper_weekly_projections", season=s)
            for s in SEASONS
        ],
        ignore_index=True,
    )
    sleeper_pts = sleeper_weekly_points(sleeper_raw, ruleset=ruleset)
    raw_actuals = WeeklyStatsSchema.validate(
        pd.concat(
            [read_partition(DATA_ROOT / "raw", "weekly_stats", season=s) for s in SEASONS],
            ignore_index=True,
        )
    )
    actuals = dk_weekly_actuals(raw_actuals, ruleset=ruleset)
    usage = build_usage(raw_actuals)
    universe = build_universe(ours, sleeper_pts, actuals, usage=usage)
    universe.to_parquet(UNIVERSE_CACHE, index=False)
    print(f"[save] universe -> {UNIVERSE_CACHE} ({len(universe)} cells)", flush=True)
    return universe


def main() -> None:
    u = build_or_load_universe()
    print(f"\nDELTA (disagreement threshold) = {config.DELTA} DK-base pts")
    print(f"N_BOOTSTRAP = {config.N_BOOTSTRAP}, seed = {config.BOOTSTRAP_SEED}\n")

    # Pooled sanity check (should reproduce ~0.476).
    pooled = clustered_bootstrap_fraction(u, seed=config.BOOTSTRAP_SEED)
    pooled_dis = _disagreement(u)
    print(
        f"POOLED: frac {pooled.point:.3f} (95% CI {pooled.lo_95:.3f}-{pooled.hi_95:.3f}); "
        f"comparable={len(u)}, disagreement_cells={len(pooled_dis)}, "
        f"disagreement_clusters={pooled_dis['player_season'].nunique()}\n"
    )

    model_class = {
        "QB": "lightgbm-nb",
        "RB": "baseline",
        "TE": "baseline",
        "WR": "ensemble-decomposed",
    }
    rows: list[tuple[str, Interval, int, int, int, str]] = []
    print(
        f"{'pos':<5}{'frac':>7}{'lo95':>8}{'hi95':>8}{'sig>0.5':>9}"
        f"{'cmp':>7}{'dis_cell':>9}{'dis_clus':>9}"
    )
    for pos in POSITIONS:
        sub = u[u["position"] == pos.value]
        if sub.empty:
            print(f"{pos.value:<5}  (no cells)")
            continue
        ci = clustered_bootstrap_fraction(sub, seed=config.BOOTSTRAP_SEED)
        dis = _disagreement(sub)
        sig = "YES" if ci.lo_95 > 0.50 else ("low" if ci.hi_95 < 0.50 else "ns")
        rows.append((pos.value, ci, len(sub), len(dis), dis["player_season"].nunique(), sig))
        print(
            f"{pos.value:<5}{ci.point:>7.3f}{ci.lo_95:>8.3f}{ci.hi_95:>8.3f}{sig:>9}"
            f"{len(sub):>7}{len(dis):>9}{dis['player_season'].nunique():>9}"
        )
    print(
        "\nsig>0.5: YES = CI entirely above 0.50 (real edge); "
        "low = CI entirely below (real deficit); ns = straddles 0.50 (not significant)."
    )
    _write_report(pooled, pooled_dis, len(u), rows, model_class)


def _write_report(
    pooled: Interval,
    pooled_dis: pd.DataFrame,
    n_comparable: int,
    rows: list[tuple[str, Interval, int, int, int, str]],
    model_class: dict[str, str],
) -> None:
    """Persist per-position CIs to a committed markdown so this never needs a rerun."""
    sig_label = {
        "YES": "**edge** (CI > 0.50)",
        "low": "deficit (CI < 0.50)",
        "ns": "not significant (straddles 0.50)",
    }
    lines = [
        "# DFS Edge Study — per-position significance (2021-2024)",
        "",
        "Companion to the pooled verdict in `dfs_projection_edge_2026-06-24.md` (**STOP**). "
        "Per-position head-to-head fractions with player-season **clustered-bootstrap 95% CIs** "
        f"(N_BOOTSTRAP={config.N_BOOTSTRAP}, seed={config.BOOTSTRAP_SEED}, DELTA={config.DELTA} "
        "DK-base pts) — the per-position breakdown the verdict report omitted. Universe persisted "
        f"to `data/dfs_universe_2021-2024.parquet` (gitignored) so re-cuts don't rebuild the "
        "16 model cells.",
        "",
        f"**Pooled:** {pooled.point:.3f} (95% CI {pooled.lo_95:.3f}-{pooled.hi_95:.3f}); "
        f"comparable cells {n_comparable}, disagreement cells {len(pooled_dis)}, "
        f"disagreement clusters {pooled_dis['player_season'].nunique()}.",
        "",
        "| Position | Model | Fraction | 95% CI | Verdict | "
        "Cmp cells | Disagree cells | Clusters |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for posv, ci, ncmp, ndis, nclus, sig in rows:
        lines.append(
            f"| {posv} | `{model_class.get(posv, '?')}` | {ci.point:.3f} | "
            f"{ci.lo_95:.3f}-{ci.hi_95:.3f} | {sig_label[sig]} | {ncmp} | {ndis} | {nclus} |"
        )
    lines += [
        "",
        "**Fraction** = share of disagreement cells (|ours - Sleeper| >= DELTA) where our "
        "projection is strictly closer to the DK-base actual. > 0.50 = we beat Sleeper. "
        "Per-position tests are exploratory/non-confirmatory (no multiple-comparison "
        "correction); the pre-registered gate is the pooled test only.",
    ]
    out = Path("reports/dfs_per_position_significance_2026-06-25.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[report] -> {out}")


if __name__ == "__main__":
    main()
