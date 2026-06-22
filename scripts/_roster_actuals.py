"""Scratch: ACTUAL-performance cut of the hero rosters (iteration 2 of the champ-vs-win%
investigation), plus a sample championship roster dump per strategy.

Replays each hero cell's deterministic draft, then scores the roster week-by-week the way
the harness does (lineup set by ESPN weekly projection, summed by ACTUAL), splitting
regular weeks (1-14) from playoff weeks (15-17). The luck-vs-structural test: within a
strategy, is a champion's edge over a playoff-no-title team in its *regular-season* ppg
(roster genuinely better -> structural) or only in its *playoff-week* ppg (ran hot in the
3 single-elim weeks -> variance/luck)?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.backtest.checkpoint import load_results
from projections.draft.backtest.draft_field import draft_mixed_field, hero_seat_layout
from projections.draft.backtest.harness import _build_strategy
from projections.draft.backtest.inputs import load_inputs
from projections.draft.backtest.lineup import weekly_lineup_points
from projections.draft.league_config import LeagueConfig

_STRATEGIES = ("now_or_never_floored", "season_value_timing")
_SEASONS = (2021, 2022, 2023, 2024, 2025)
_JITTER = 8.0
_STRAT_N_SIMS = 40
_DUMP_SEASON = 2023  # pick the sample champion rosters from a mid, "normal" season


def _cell_outcome(ckpt: Path, strategy: str, seat: int, seed: int) -> tuple[bool, bool]:
    actual, _ = load_results(
        json.loads((ckpt / f"cell_{strategy}_{seat:02d}_{seed:05d}.json").read_text())
    )
    return bool(actual[0].made_playoffs), bool(actual[0].is_champion)


def _weekly_actuals(roster, pos_by_id, slots, reg_weeks, po_weeks, proj_lookup, actual_lookup):
    """Per-week actual points (lineup set by weekly projection), for reg + playoff weeks."""
    reg, po = [], []
    for bucket, weeks in ((reg, reg_weeks), (po, po_weeks)):
        for wk in weeks:
            rows = [
                {
                    "position": pos_by_id[g],
                    "projected": proj_lookup.get((g, wk)),
                    "actual": actual_lookup.get((g, wk)),
                }
                for g in roster
            ]
            bucket.append(weekly_lineup_points(rows, slots, score_by="actual"))
    return np.array(reg, dtype=float), np.array(po, dtype=float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--league-config", type=Path, default=Path("configs/league_espn_half_16team.json")
    )
    ap.add_argument("--n-seeds", type=int, default=25)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args(argv)

    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    slots = config.roster_slots

    rows: list[dict[str, object]] = []
    dumps: dict[str, dict] = {}  # strategy -> a representative champion roster payload
    for season in _SEASONS:
        inputs = load_inputs(season=season, config=config, data_root=args.data_root)
        pool = inputs.pool
        pos_by_id = {str(g): str(p) for g, p in zip(pool["gsis_id"], pool["position"], strict=True)}
        reg_weeks = list(inputs.calendar.regular_weeks)
        po_weeks = list(inputs.calendar.playoff_weeks)
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
                    league = draft_mixed_field(
                        dict(seat_strats),
                        pool,
                        config,
                        rng=np.random.default_rng(seed),
                        jitter=_JITTER,
                    )
                    roster = [str(g) for g in league[seat]]
                    made_po, champ = _cell_outcome(ckpt, strategy, seat, seed)
                    reg, po = _weekly_actuals(
                        roster,
                        pos_by_id,
                        slots,
                        reg_weeks,
                        po_weeks,
                        inputs.proj_lookup,
                        inputs.actual_lookup,
                    )
                    outcome = "champion" if champ else ("playoff_no_title" if made_po else "missed")
                    allw = np.concatenate([reg, po])
                    rows.append(
                        {
                            "strategy": strategy,
                            "season": season,
                            "seat": seat,
                            "seed": seed,
                            "outcome": outcome,
                            "reg_ppg": float(reg.mean()),
                            "playoff_ppg": float(po.mean()),
                            "gap": float(po.mean() - reg.mean()),
                            "wk_std_actual": float(allw.std()),
                            "actual_total": float(allw.sum()),
                        }
                    )
                    # capture a representative champion roster for the dump
                    if champ and season == _DUMP_SEASON and strategy not in dumps:
                        dumps[strategy] = _roster_payload(roster, pool, inputs, reg, po, seat, seed)
            print(f"[done] {season} {strategy}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet("_roster_actuals.parquet", index=False)
    cols = ["reg_ppg", "playoff_ppg", "gap", "wk_std_actual", "actual_total"]
    print("\n=== mean ACTUAL performance by (strategy, outcome) ===")
    for strategy in _STRATEGIES:
        d = df[df["strategy"] == strategy]
        print(f"\n--- {strategy} ---")
        g = d.groupby("outcome")[cols].mean().join(d.groupby("outcome").size().rename("n"))
        for oc in ["champion", "playoff_no_title", "missed"]:
            if oc in g.index:
                r = g.loc[oc]
                print(f"{oc:<18} n={int(r['n']):<5} " + " ".join(f"{c}={r[c]:.2f}" for c in cols))

    for strategy, payload in dumps.items():
        loc = f"seat {payload['seat']}, seed {payload['seed']}"
        print(f"\n=== SAMPLE CHAMPION ROSTER — {strategy} ({_DUMP_SEASON}, {loc}) ===")
        print(f"reg_ppg={payload['reg_ppg']:.1f}  playoff_ppg={payload['playoff_ppg']:.1f}")
        print(f"{'pos':<4}{'player':<24}{'proj':>7}{'actual':>8}{'p_week':>8}")
        for row in payload["players"]:
            print(
                f"{row['pos']:<4}{row['name'][:23]:<24}{row['proj']:>7.1f}{row['actual']:>8.1f}{row['pweek']:>8.2f}"
            )
    return 0


def _roster_payload(roster, pool, inputs, reg, po, seat, seed) -> dict:
    sub = pool[pool["gsis_id"].astype(str).isin(roster)].copy()
    name = dict(zip(pool["gsis_id"].astype(str), pool["full_name"].astype(str), strict=True))
    players = []
    for g in roster:
        actual_total = sum(
            inputs.actual_lookup.get((g, wk), 0.0)
            for wk in list(inputs.calendar.regular_weeks) + list(inputs.calendar.playoff_weeks)
        )
        prow = sub[sub["gsis_id"].astype(str) == g].iloc[0]
        players.append(
            {
                "pos": str(prow["position"]),
                "name": name.get(g, g),
                "proj": float(prow["season_mean_fpts"]),
                "actual": actual_total,
                "pweek": inputs.availability.p_week(g),
            }
        )
    players.sort(key=lambda r: -r["proj"])
    return {
        "seat": seat,
        "seed": seed,
        "reg_ppg": float(reg.mean()),
        "playoff_ppg": float(po.mean()),
        "players": players,
    }


if __name__ == "__main__":
    raise SystemExit(main())
