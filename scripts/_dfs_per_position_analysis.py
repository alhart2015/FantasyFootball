"""One-off: per-position significance breakdown of the DFS edge study.

Loads (or builds + persists) the 2021-2024 comparable universe via the shared
`build_study_inputs` path — the same construction `run_study` uses — reproduces
the pooled primary fraction as a sanity check, then for each position reports the
disagreement head-to-head fraction with a player-season clustered-bootstrap 95%
CI + cell/cluster counts.

Reuses the shipped harness (same DELTA, same clustered bootstrap) so the numbers
are methodologically identical to the verdict. The universe cache is keyed only
by filename: delete `data/dfs_universe_2021-2024.parquet` to rebuild after any
config (DELTA / usage floor) or upstream data change. Throwaway — delete after use.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from projections.dfs import config
from projections.dfs.edge_study import _disagreement, clustered_bootstrap_fraction
from projections.dfs.run import build_study_inputs
from projections.draft.assistant._compare import Interval
from projections.models import POSITION_DISPATCH
from projections.schemas import Position, Ruleset

DATA_ROOT = Path("data")
FEATURES_ROOT = DATA_ROOT / "features"
SEASONS = [2021, 2022, 2023, 2024]
POSITIONS = [Position.QB, Position.RB, Position.WR, Position.TE]
UNIVERSE_CACHE = DATA_ROOT / "dfs_universe_2021-2024.parquet"

# Production model class per position (for the report's "Model" column), read from
# the single source of truth so the label can't drift when a position's model is
# swapped (e.g. RB graduating off `baseline` — TODO #55).
MODEL_CLASS = {pos: POSITION_DISPATCH[pos].default_model_class for pos in POSITIONS}
# Maps the significance tag to its report-table label; reused (asterisks stripped)
# for the console legend so the classification lives in exactly one place.
SIG_LABEL = {
    "YES": "**edge** (CI > 0.50)",
    "low": "deficit (CI < 0.50)",
    "ns": "not significant (straddles 0.50)",
    "n/a": "unmeasurable (CI is NaN — too few clusters)",
}
# Columns the per-position analysis reads off the (possibly cached) universe.
_REQUIRED_COLS = {"position", "season", "our_pts", "sleeper_pts", "actual_points", "player_season"}


class PositionRow(NamedTuple):
    position: str
    model: str
    ci: Interval
    n_comparable: int
    n_disagreement: int
    n_clusters: int
    sig: str


def _require_complete_universe(universe: pd.DataFrame) -> None:
    """Refuse to report off a stale or quietly-partial cached universe. Checks the
    expected columns are present (a schema-drifted cache otherwise dies with an
    opaque KeyError mid-run) and that every (position, season) cell exists — a
    missing Sleeper/feature season inner-joins away silently, and `_load_sleeper`
    skips missing seasons, so a partial universe could be reported as the full
    2021-2024 result."""
    missing_cols = sorted(_REQUIRED_COLS - set(universe.columns))
    if missing_cols:
        raise SystemExit(
            f"cached universe is missing columns {missing_cols} — it predates a schema "
            f"change. Delete {UNIVERSE_CACHE} and re-run to rebuild."
        )
    present = {
        (str(p), int(s))
        for p, s in universe[["position", "season"]].drop_duplicates().itertuples(index=False)
    }
    missing = sorted({(pos.value, season) for pos in POSITIONS for season in SEASONS} - present)
    if missing:
        raise SystemExit(
            f"universe is missing (position, season) cells {missing} (expected every "
            f"{[p.value for p in POSITIONS]} x {SEASONS}); refusing to report a partial "
            f"result. Delete {UNIVERSE_CACHE} and re-run after ingesting all partitions."
        )


def build_or_load_universe() -> pd.DataFrame:
    if UNIVERSE_CACHE.exists():
        print(f"[load] cached universe {UNIVERSE_CACHE} (delete to rebuild)", flush=True)
        universe = pd.read_parquet(UNIVERSE_CACHE)
    else:
        print("[build] build_study_inputs (this is the slow step)...", flush=True)
        universe = build_study_inputs(
            seasons=SEASONS,
            positions=POSITIONS,
            data_root=DATA_ROOT,
            features_root=FEATURES_ROOT,
            ruleset=Ruleset.draftkings(),
        ).universe
        universe.to_parquet(UNIVERSE_CACHE, index=False)
        print(f"[save] universe -> {UNIVERSE_CACHE} ({len(universe)} cells)", flush=True)
    _require_complete_universe(universe)
    return universe


def _position_row(pos: Position, universe: pd.DataFrame) -> PositionRow | None:
    sub = universe[universe["position"] == pos.value]
    if sub.empty:
        return None
    ci = clustered_bootstrap_fraction(sub, seed=config.BOOTSTRAP_SEED)
    dis = _disagreement(sub)
    if pd.isna(ci.lo_95) or pd.isna(ci.hi_95):
        sig = "n/a"  # degenerate bootstrap — unmeasurable, NOT "not significant"
    elif ci.lo_95 > 0.50:
        sig = "YES"
    elif ci.hi_95 < 0.50:
        sig = "low"
    else:
        sig = "ns"
    return PositionRow(
        pos.value, MODEL_CLASS[pos], ci, len(sub), len(dis), dis["player_season"].nunique(), sig
    )


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

    rows: list[PositionRow] = []
    print(
        f"{'pos':<5}{'frac':>7}{'lo95':>8}{'hi95':>8}{'sig>0.5':>9}"
        f"{'cmp':>7}{'dis_cell':>9}{'dis_clus':>9}"
    )
    for pos in POSITIONS:
        row = _position_row(pos, u)
        if row is None:
            print(f"{pos.value:<5}  (no comparable cells)")
            continue
        rows.append(row)
        print(
            f"{row.position:<5}{row.ci.point:>7.3f}{row.ci.lo_95:>8.3f}{row.ci.hi_95:>8.3f}"
            f"{row.sig:>9}{row.n_comparable:>7}{row.n_disagreement:>9}{row.n_clusters:>9}"
        )
    print("\nsig>0.5: " + "; ".join(f"{k} = {v.replace('*', '')}" for k, v in SIG_LABEL.items()))
    _write_report(pooled, pooled_dis, len(u), rows)


def _write_report(
    pooled: Interval, pooled_dis: pd.DataFrame, n_comparable: int, rows: list[PositionRow]
) -> None:
    """Persist per-position CIs to a committed markdown so this never needs a rerun."""
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
    for r in rows:
        lines.append(
            f"| {r.position} | `{r.model}` | {r.ci.point:.3f} | "
            f"{r.ci.lo_95:.3f}-{r.ci.hi_95:.3f} | {SIG_LABEL[r.sig]} | "
            f"{r.n_comparable} | {r.n_disagreement} | {r.n_clusters} |"
        )
    lines += [
        "",
        "**Fraction** = share of disagreement cells (|ours - Sleeper| > DELTA) where our "
        "projection is strictly closer to the DK-base actual. > 0.50 = we beat Sleeper. "
        "Per-position tests are exploratory/non-confirmatory (no multiple-comparison "
        "correction); the pre-registered gate is the pooled test only.",
    ]
    out = Path("reports/dfs_per_position_significance_2026-06-25.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[report] -> {out}")


if __name__ == "__main__":
    main()
