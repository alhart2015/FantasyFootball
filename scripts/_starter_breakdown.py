"""Scratch: per-position depth-slot scoring breakdown (QB1/QB2/QB3, RB1..RBn, ...).

For each strategy, replay every hero cell's deterministic draft, label each rostered player
by DRAFT ORDER within its position (RB1 = first RB drafted, ...), then attribute season
points to each depth slot counting ONLY the weeks that player was in the starting lineup.

Two engines (the user asked for both):
  realized -- lineup set by ESPN weekly projection, credited by ACTUAL points (weeks 1-17),
              over ALL cells. The "what actually happened" depth curve.
  mc       -- the season-value availability model: per (sim, week) sample availability
              (injury + bye) + weekly points, fill the optimal lineup of available players,
              credit sampled points. A modeled cross-check, over a seed subset.

A slot's value = total points credited to it / number of rosters (rosters that never draft a
k-th RB contribute 0 to RBk), i.e. the expected season points your k-th player at a position
adds to the starting lineup. Shows e.g. QB3 ~ 0 (the wasted pick the cap removes) vs RB3-6
still positive (real injury/bye coverage).
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from projections.draft.assistant.performance_variance import VarianceParams, sample_weekly_points
from projections.draft.backtest.draft_field import draft_mixed_field, hero_seat_layout
from projections.draft.backtest.harness import _build_strategy
from projections.draft.backtest.inputs import load_inputs
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import FLEX_SLOTS, POSITION_SLOTS
from projections.schemas import Position, RosterSlot

_STRATEGIES = ("now_or_never_floored", "season_value_timing", "season_value_qb_cap")
_SEASONS = (2021, 2022, 2023, 2024, 2025)
_JITTER = 8.0
_STRAT_N_SIMS = 40
_WEEKS = list(range(1, 18))  # 1..17, reg 1-14 + playoff 15-17
_MC_SEEDS = 5  # MC breakdown uses seeds [0, _MC_SEEDS); realized uses all
_MC_NSIMS = 60


def _fill_starters(
    positions: list[str], values: list[float], available: list[bool], slots: dict[RosterSlot, int]
) -> list[int]:
    """Indices of the optimal legal starting lineup (mirrors backtest.lineup fill order)."""
    by_pos: dict[Position, list[int]] = defaultdict(list)
    for i, (p, a) in enumerate(zip(positions, available, strict=True)):
        if a:
            by_pos[Position(p)].append(i)
    for p in by_pos:
        by_pos[p].sort(key=lambda i: values[i], reverse=True)
    cursor: dict[Position, int] = defaultdict(int)
    chosen: list[int] = []
    for slot in POSITION_SLOTS:
        pos = Position(slot.value)
        for _ in range(slots.get(slot, 0)):
            if cursor[pos] < len(by_pos[pos]):
                chosen.append(by_pos[pos][cursor[pos]])
                cursor[pos] += 1
    for slot, eligible in FLEX_SLOTS:
        for _ in range(slots.get(slot, 0)):
            best_pos, best_v = None, -math.inf
            for pos in sorted(eligible, key=lambda p: p.value):
                if cursor[pos] < len(by_pos[pos]):
                    v = values[by_pos[pos][cursor[pos]]]
                    if v > best_v:
                        best_pos, best_v = pos, v
            if best_pos is not None:
                chosen.append(by_pos[best_pos][cursor[best_pos]])
                cursor[best_pos] += 1
    return chosen


def _depth_labels(positions: list[str]) -> list[str]:
    """Draft-order depth label per roster index: RB1, RB2, ... (roster is in draft order)."""
    seen: dict[str, int] = defaultdict(int)
    labels = []
    for p in positions:
        seen[p] += 1
        labels.append(f"{p}{seen[p]}")
    return labels


def _realized(roster, positions, labels, slots, proj_lookup, actual_lookup, acc):
    for wk in _WEEKS:
        proj = [proj_lookup.get((g, wk)) for g in roster]
        avail = [p is not None for p in proj]
        vals = [float(p) if p is not None else 0.0 for p in proj]
        for i in _fill_starters(positions, vals, avail, slots):
            acc[labels[i]] += float(actual_lookup.get((roster[i], wk), 0.0))


def _mc(roster, positions, labels, slots, means, rookie, byes, pweek, params, rng, acc):
    n = len(roster)
    pts = sample_weekly_points(
        params, np.array(positions), means, rookie, n_sims=_MC_NSIMS, n_weeks=len(_WEEKS), rng=rng
    )  # (n_sims, n_weeks, n)
    draws = rng.random((_MC_NSIMS, len(_WEEKS), n))
    for s in range(_MC_NSIMS):
        for wi, wk in enumerate(_WEEKS):
            avail = [(byes[i] != wk) and (draws[s, wi, i] < pweek[i]) for i in range(n)]
            vals = [float(pts[s, wi, i]) for i in range(n)]
            for i in _fill_starters(positions, vals, avail, slots):
                acc[labels[i]] += vals[i] / _MC_NSIMS


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
    params = VarianceParams.load()

    real_acc = {s: defaultdict(float) for s in _STRATEGIES}
    mc_acc = {s: defaultdict(float) for s in _STRATEGIES}
    n_rosters = dict.fromkeys(_STRATEGIES, 0)
    n_mc_rosters = dict.fromkeys(_STRATEGIES, 0)

    for season in _SEASONS:
        inputs = load_inputs(season=season, config=config, data_root=args.data_root)
        pool = inputs.pool
        prow = pool.set_index(pool["gsis_id"].astype(str))
        rookie_map = dict(
            zip(pool["gsis_id"].astype(str), pool["is_rookie"].astype(bool), strict=True)
        )
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
                ss = {s: (strat if lbl != "bot" else None) for s, lbl in layout.items()}
                for seed in range(args.n_seeds):
                    rosters = draft_mixed_field(
                        dict(ss), pool, config, rng=np.random.default_rng(seed), jitter=_JITTER
                    )
                    roster = [str(g) for g in rosters[seat]]
                    positions = [str(prow.loc[g, "position"]) for g in roster]
                    labels = _depth_labels(positions)
                    _realized(
                        roster,
                        positions,
                        labels,
                        slots,
                        inputs.proj_lookup,
                        inputs.actual_lookup,
                        real_acc[strategy],
                    )
                    n_rosters[strategy] += 1
                    if seed < _MC_SEEDS:
                        means = np.array([float(prow.loc[g, "season_mean_fpts"]) for g in roster])
                        rookie = np.array([rookie_map[g] for g in roster])
                        byes = [inputs.availability.bye_week(g) or -1 for g in roster]
                        pweek = [inputs.availability.p_week(g) for g in roster]
                        _mc(
                            roster,
                            positions,
                            labels,
                            slots,
                            means,
                            rookie,
                            byes,
                            pweek,
                            params,
                            np.random.default_rng(10_000 + seed),
                            mc_acc[strategy],
                        )
                        n_mc_rosters[strategy] += 1
            print(f"[done] {season} {strategy}", flush=True)

    for strategy in _STRATEGIES:
        nr, nm = n_rosters[strategy], n_mc_rosters[strategy]
        print(f"\n=== {strategy} — expected season pts per depth slot (realized / MC) ===")
        slots_seen = set(real_acc[strategy]) | set(mc_acc[strategy])
        for pos in ("QB", "RB", "WR", "TE"):
            ranks = sorted(
                int(s[len(pos) :])
                for s in slots_seen
                if s.startswith(pos) and s[len(pos) :].isdigit()
            )
            for r in ranks:
                lab = f"{pos}{r}"
                rv = real_acc[strategy].get(lab, 0.0) / nr
                mv = mc_acc[strategy].get(lab, 0.0) / nm if nm else float("nan")
                if rv >= 0.05 or mv >= 0.05:
                    print(f"  {lab:<5} realized={rv:6.1f}   mc={mv:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
