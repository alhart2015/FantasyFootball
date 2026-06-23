"""Scratch: reconstruct hero rosters from the hero-eval cells and compare their structure
by outcome (champion / made-playoffs-no-title / missed-playoffs), per strategy.

The hero checkpoints store only the hero's W/L/playoff/champion result, not the roster.
But the draft is deterministic given (strategy, seat, seed), so we replay it with the
identical setup (hero_seat_layout + draft_mixed_field + rng=default_rng(seed)) to recover
the roster, then join to the cell's ACTUAL-scoring outcome.

For each roster we compute DRAFTED-TIME (projected) structural features + variance-model
weekly volatility, then average them within each (strategy, outcome) group. Question this
answers: do champion rosters look structurally different at draft time from
playoff-but-not-champion rosters (structural), or the same (-> actual-performance/bracket luck)?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.performance_variance import _GAMES, VarianceParams
from projections.draft.backtest.checkpoint import load_results
from projections.draft.backtest.draft_field import draft_mixed_field, hero_seat_layout
from projections.draft.backtest.harness import _build_strategy
from projections.draft.backtest.inputs import load_inputs
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import allocate_roster_slots
from projections.schemas import Position, RosterSlot

_STRATEGIES = ("now_or_never_floored", "season_value_timing")
_SEASONS = (2021, 2022, 2023, 2024, 2025)
_JITTER = 8.0
_STRAT_N_SIMS = 40  # must match the hero run so the replayed draft is identical


def _cell_outcome(checkpoint_dir: Path, strategy: str, seat: int, seed: int) -> tuple[bool, bool]:
    """(made_playoffs, is_champion) from the cell's ACTUAL-scoring result."""
    path = checkpoint_dir / f"cell_{strategy}_{seat:02d}_{seed:05d}.json"
    actual, _ = load_results(json.loads(path.read_text()))
    return bool(actual[0].made_playoffs), bool(actual[0].is_champion)


def _optimal_starters(sub: pd.DataFrame, slots: dict[RosterSlot, int]) -> pd.DataFrame:
    """Rows in the optimal legal starting lineup (greedy by season_mean_fpts, restrictive-first)."""
    s = sub.sort_values("season_mean_fpts", ascending=False)
    players = [
        (gid, Position(p))
        for gid, p in zip(s["gsis_id"].astype(str), s["position"].astype(str), strict=True)
    ]
    placements, _, _ = allocate_roster_slots(players, slots)
    starter_ids = {key for key, _pos, slot in placements if slot != RosterSlot.BENCH}
    return s[s["gsis_id"].astype(str).isin(starter_ids)]


def _features(
    roster: list[str],
    pool: pd.DataFrame,
    slots: dict[RosterSlot, int],
    params: VarianceParams,
    pweek: dict[str, float],
) -> dict[str, float]:
    sub = pool[pool["gsis_id"].astype(str).isin(roster)]
    pos = sub["position"].astype(str)
    mean = sub["season_mean_fpts"].to_numpy(dtype=float)
    total = float(mean.sum())
    top3 = float(np.sort(mean)[::-1][:3].sum())

    starters = _optimal_starters(sub, slots)
    starters_proj = float(starters["season_mean_fpts"].sum())
    # Per-week boom magnitude of the starting lineup: independent-sum std of weekly noise.
    wk_var = sum(
        params.weekly_std(str(p), m / _GAMES) ** 2
        for p, m in zip(starters["position"].astype(str), starters["season_mean_fpts"], strict=True)
    )
    wk_std = math.sqrt(wk_var)
    starter_pg = starters_proj / _GAMES

    rookie = (
        sub["is_rookie"].to_numpy(dtype=bool)
        if "is_rookie" in sub.columns
        else np.zeros(len(sub), bool)
    )
    pw = np.array([pweek.get(str(g), float("nan")) for g in sub["gsis_id"].astype(str)])
    return {
        "nQB": int((pos == "QB").sum()),
        "nRB": int((pos == "RB").sum()),
        "nWR": int((pos == "WR").sum()),
        "nTE": int((pos == "TE").sum()),
        "total_proj": total,
        "starters_proj": starters_proj,
        "top3_share": top3 / total if total else float("nan"),
        "wk_std": wk_std,  # per-week boom magnitude of the starting lineup
        "wk_cv": wk_std / starter_pg if starter_pg else float("nan"),  # boom relative to level
        "mean_pweek": float(np.nanmean(pw)) if len(pw) else float("nan"),
        "n_rookie": int(rookie.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--league-config", type=Path, default=Path("configs/league_espn_half_16team.json")
    )
    ap.add_argument("--n-seeds", type=int, default=25)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args(argv)

    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    params = VarianceParams.load()
    slots = config.roster_slots

    rows: list[dict[str, object]] = []
    for season in _SEASONS:
        inputs = load_inputs(season=season, config=config, data_root=args.data_root)
        pool = inputs.pool
        pweek = {g: inputs.availability.p_week(g) for g in pool["gsis_id"].astype(str)}
        ckpt = Path(f"_hero_{season}")
        for strategy in _STRATEGIES:
            strat = _build_strategy(
                strategy,
                availability=inputs.availability,
                n_teams=config.n_teams,
                strategy_n_sims=_STRAT_N_SIMS,
                base_seed=0,
            )
            for seat in range(1, config.n_teams + 1):
                layout = hero_seat_layout(
                    hero_seat=seat, hero_label=strategy, n_teams=config.n_teams
                )
                seat_strats = {s: (strat if lbl != "bot" else None) for s, lbl in layout.items()}
                for seed in range(args.n_seeds):
                    rosters = draft_mixed_field(
                        dict(seat_strats),
                        pool,
                        config,
                        rng=np.random.default_rng(seed),
                        jitter=_JITTER,
                    )
                    made_po, champ = _cell_outcome(ckpt, strategy, seat, seed)
                    outcome = "champion" if champ else ("playoff_no_title" if made_po else "missed")
                    feat = _features([str(g) for g in rosters[seat]], pool, slots, params, pweek)
                    rows.append(
                        {
                            "strategy": strategy,
                            "season": season,
                            "seat": seat,
                            "seed": seed,
                            "outcome": outcome,
                            **feat,
                        }
                    )
            print(f"[done] {season} {strategy}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet("_roster_anatomy.parquet", index=False)
    feat_cols = [
        "total_proj",
        "starters_proj",
        "top3_share",
        "wk_std",
        "wk_cv",
        "mean_pweek",
        "n_rookie",
        "nQB",
        "nRB",
        "nWR",
        "nTE",
    ]
    print("\n=== mean roster features by (strategy, outcome) ===")
    for strategy in _STRATEGIES:
        d = df[df["strategy"] == strategy]
        print(f"\n--- {strategy} ---")
        grp = d.groupby("outcome")[feat_cols].mean()
        cnt = d.groupby("outcome").size().rename("n")
        out = grp.join(cnt)
        for oc in ["champion", "playoff_no_title", "missed"]:
            if oc in out.index:
                r = out.loc[oc]
                print(
                    f"{oc:<18} n={int(r['n']):<5} " + " ".join(f"{c}={r[c]:.3f}" for c in feat_cols)
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
