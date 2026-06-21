"""Scratch: (sigma, floor, floor_weight) search + holdout validation for a snake
strategy that beats every baseline in 6-yr-mean win% at all seats (NOT committed).

Generalizes the top baseline `now_or_never_floored` (which already takes all three
knobs) and searches them; no new strategy code needed until a winner is found.

Two modes:
  search   -- run a grid of (sigma,F,lambda) configs over the TRAINING split
              (default years 2021/2023/2025, seeds 0-19, seats 1/8/16), rank by
              mean win%.
  validate -- run ONE config + all 6 baselines over the HOLDOUT split (default
              years 2022/2024/2026, seeds 100-179, seats 1/8/16) at high N, with
              per-seat means + bootstrap CIs + paired diffs vs each baseline.

Metric: project_draft reg_win_pct (projected-vs-projected H2H), CRN season RNG
shared across configs per (year, seed). Repeatable by seed.

Search-trail note: the `TiltedFlooredNN` tilts (durability `mu`, bye `nu`) both fell out
as DEAD levers (the projection already prices durability; bye collisions are too rare to
matter) and the `sigma/F/lambda` knobs are a plateau. The shipped winner is the routing-only
`SeatAwareStrategy` (`build_seat_aware`, in src) -- `TiltedFlooredNN` and its tilt plumbing are
retained here only as the recorded negative result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import bootstrap_mean
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_projection import project_draft
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.pick_timing import my_next_pick, slot_for
from projections.draft.assistant.pool_identity import reconcile_pool_gsis
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.assistant.simulation import _draft_picks
from projections.draft.assistant.strategy import (
    NowOrNeverFlooredStrategy,
    SeasonValueTimingStrategy,
    _eligible_subset,
    _finalize,
    _raw_vorp_result,
    build_seat_aware,
)
from projections.draft.assistant.survival import (
    LogisticSurvival,
    default_sigma,
    expected_best_by_position,
)
from projections.draft.backtest.harness import _build_strategy
from projections.draft.league_config import LeagueConfig

_FLEX_GROUP = frozenset({"RB", "WR", "TE"})


@dataclass(frozen=True)
class TiltedFlooredNN:
    """now_or_never_floored + two orthogonal tilts (scratch experiment).

    score = vorp - E[best survivor] - lambda*max(0, F - vorp)
            - mu*(1 - p_week)*vorp                 # durability tilt
            - nu*(my roster-mates sharing this bye)  # bye-diversification tilt

    Durability (mu): penalize value by injury risk -- DEAD lever (projections already
    price durability), kept for completeness. Bye (nu): penalize a candidate whose
    team bye collides with my already-rostered FLEX-group players (RB/WR/TE share the
    flex; QB counts only QB), the one signal absent from a season-total projection but
    present in the metric (weekly bye masking). mu == nu == 0 reproduces nn_floored.
    """

    survival: LogisticSurvival
    availability: PlayerAvailability
    floor: float
    floor_weight: float
    mu: float
    nu: float

    def recommend(self, state, pool, config):
        df, elig = _eligible_subset(state, pool, config)
        next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
        if next_pick is None:
            return _raw_vorp_result(df, elig)
        adp = df["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(
                float(a) if pd.notna(a) else float("nan"), next_pick
            )
        ).astype(float)
        display_p = internal_p.where(adp.notna(), other=pd.NA)
        pos = df["position"].to_numpy()
        vorp = df["vorp"].to_numpy(dtype=float)
        p = internal_p.to_numpy(dtype=float)
        gsis = df["gsis_id"].to_numpy()
        e_best = expected_best_by_position(pos, vorp, p, gsis)
        p_week = np.array([self.availability.p_week(str(g)) for g in gsis], dtype=float)
        floor_pen = self.floor_weight * np.maximum(0.0, self.floor - vorp)
        durab_pen = self.mu * (1.0 - p_week) * vorp
        bye_pen = self._bye_penalty(state, pool, pos, gsis) if self.nu else np.zeros(len(df))
        df["score"] = (
            vorp
            - np.array([e_best[pos_i] for pos_i in pos], dtype=float)
            - floor_pen
            - durab_pen
            - bye_pen
        )
        return _finalize(df, elig, display_p)

    def _bye_penalty(self, state, pool, pos, gsis) -> np.ndarray:
        """nu * (count of my roster-mates in the same flex-group sharing each bye)."""
        my_ids = {str(g) for g in state.my_pick_ids}
        sub = pool[pool["gsis_id"].astype(str).isin(my_ids)]
        # my drafted (bye_week, flex-group-key) tallies
        roster: list[tuple[int, str]] = []
        for g, pp in zip(sub["gsis_id"].astype(str), sub["position"].astype(str), strict=True):
            bw = self.availability.bye_week(g)
            if bw is not None:
                roster.append((bw, "FLEX" if pp in _FLEX_GROUP else pp))
        out = np.zeros(len(gsis))
        for i, (g, pp) in enumerate(zip(gsis, pos, strict=True)):
            bw = self.availability.bye_week(str(g))
            if bw is None:
                continue
            key = "FLEX" if str(pp) in _FLEX_GROUP else str(pp)
            out[i] = self.nu * sum(1 for rb, rk in roster if rb == bw and rk == key)
        return out


_SEASON_OFFSET = 1_000_000  # season RNG stream, disjoint from the draft stream
_DATA = Path("data")
_ID_MAP: pd.DataFrame | None = None


def _load_id_map() -> pd.DataFrame:
    global _ID_MAP
    if _ID_MAP is None:
        _ID_MAP = pd.read_parquet(_DATA / "raw" / "id_map.parquet")
    return _ID_MAP


BASELINES = (
    "now_or_never",
    "now_or_never_floored",
    "season_value",
    "season_value_var",
    "season_value_timing",
    "raw_vorp",
)


@dataclass(frozen=True)
class YearCtx:
    pool: pd.DataFrame
    config: LeagueConfig
    availability: PlayerAvailability
    n_teams: int


_PARAMS = VarianceParams.load()
_YEAR_CACHE: dict[tuple[int, bool], YearCtx] = {}
_RECONCILE = True  # set from CLI; gsis reconciliation on per-season pools


def _year_ctx(year: int, size: str = "half_16team", *, reconcile: bool = True) -> YearCtx:
    cache_key = (year, reconcile)
    if cache_key not in _YEAR_CACHE:
        cfg = LeagueConfig.model_validate_json(
            (_DATA / f"vorp_{year}" / f"{size}.league.json").read_text()
        )
        pool = pd.read_parquet(_DATA / f"vorp_{year}" / f"{size}.parquet")
        if reconcile:
            pool = reconcile_pool_gsis(pool, _load_id_map())
        pool = attach_is_rookie(pool, season=year, data_root=_DATA)
        avail = load_store_availability(pool, season=year, data_root=_DATA)
        _YEAR_CACHE[cache_key] = YearCtx(pool, cfg, avail, cfg.n_teams)
    return _YEAR_CACHE[cache_key]


def _full_league(picks: list[str], n_teams: int) -> dict[int, list[str]]:
    rosters: dict[int, list[str]] = {s: [] for s in range(1, n_teams + 1)}
    for i, gid in enumerate(picks):
        rosters[slot_for(i + 1, n_teams)].append(str(gid))
    return rosters


def _win_for_cell(strat, ctx: YearCtx, seat: int, seed: int, n_sims: int, jitter: float) -> float:
    """One (strategy, year, seat, seed) cell -> the hero seat's projected reg_win_pct."""
    picks = _draft_picks(
        strat, seat, ctx.pool, ctx.config, adp_jitter=jitter, rng=np.random.default_rng(seed)
    )
    rosters = _full_league([str(p) for p in picks], ctx.n_teams)
    proj = project_draft(
        rosters,
        ctx.pool,
        ctx.availability,
        _PARAMS,
        league_config=ctx.config,
        n_sims=n_sims,
        rng=np.random.default_rng(seed + _SEASON_OFFSET),  # CRN across configs
    )
    return float(proj[seat].reg_win_pct)


def _build(key_or_cfg, ctx: YearCtx, strat_n_sims: int):
    """Build a strategy: a baseline key (str) or a (sigma,F,lambda) tuple."""
    if isinstance(key_or_cfg, tuple):
        if len(key_or_cfg) >= 4:
            sigma, floor, lam, mu = key_or_cfg[:4]
            nu = key_or_cfg[4] if len(key_or_cfg) == 5 else 0.0
            return TiltedFlooredNN(
                LogisticSurvival(sigma=sigma),
                ctx.availability,
                floor=floor,
                floor_weight=lam,
                mu=mu,
                nu=nu,
            )
        sigma, floor, lam = key_or_cfg
        return NowOrNeverFlooredStrategy(
            LogisticSurvival(sigma=sigma), floor=floor, floor_weight=lam
        )
    if key_or_cfg == "sv_var_timing":
        # variance-aware MC marginals UNDER the timing layer -- the "best of both"
        # not wired as a baseline (harness builds season_value_timing risk_aware=False).
        return SeasonValueTimingStrategy(
            ctx.availability,
            n_sims=strat_n_sims,
            base_seed=0,
            survival=LogisticSurvival(sigma=default_sigma(ctx.n_teams)),
            risk_aware=True,
        )
    if key_or_cfg == "seat_aware":
        # Per-slot router (the shipped winner): season_value_timing off the turn,
        # season_value_var at the last two seats -- the per-seat Pareto frontier (Test 14).
        return build_seat_aware(
            ctx.availability,
            n_sims=strat_n_sims,
            base_seed=0,
            survival=LogisticSurvival(sigma=default_sigma(ctx.n_teams)),
        )
    return _build_strategy(
        key_or_cfg,
        availability=ctx.availability,
        n_teams=ctx.n_teams,
        strategy_n_sims=strat_n_sims,
        base_seed=0,
    )


def _run(key_or_cfg, years, seats, seeds, n_sims, strat_n_sims, jitter):
    """Return (per_seat: dict seat->win array, all: win array) over the cell grid."""
    per_seat: dict[int, list[float]] = {s: [] for s in seats}
    for year in years:
        ctx = _year_ctx(year, reconcile=_RECONCILE)
        strat = _build(key_or_cfg, ctx, strat_n_sims)
        for seat in seats:
            for seed in seeds:
                per_seat[seat].append(_win_for_cell(strat, ctx, seat, seed, n_sims, jitter))
    per_seat_arr = {s: np.array(v) for s, v in per_seat.items()}
    allv = np.concatenate([per_seat_arr[s] for s in seats])
    return per_seat_arr, allv


def _fmt_cfg(cfg: tuple[float, ...]) -> str:
    extra = ""
    if len(cfg) >= 4:
        extra += f" mu={cfg[3]:>4}"
    if len(cfg) == 5:
        extra += f" nu={cfg[4]:>4}"
    return f"sig={cfg[0]:>5} F={cfg[1]:>4} lam={cfg[2]:>4}{extra}"


def _parse_grid(spec: str) -> list[tuple[float, ...]]:
    """'sig,F,lam[,mu];...' -> list of configs (4-tuple => durability variant)."""
    out: list[tuple[float, ...]] = []
    for chunk in spec.split(";"):
        if not chunk.strip():
            continue
        out.append(tuple(float(x) for x in chunk.split(",")))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "validate"])
    ap.add_argument("--grid", default="", help="search: 'sig,F,lam;sig,F,lam;...'")
    ap.add_argument("--config", default="", help="validate: 'sig,F,lam'")
    ap.add_argument("--years", default="", help="comma ints; default per-mode split")
    ap.add_argument("--seats", default="1,8,16")
    ap.add_argument("--seed-lo", type=int, default=0)
    ap.add_argument("--seed-hi", type=int, default=20)
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--strat-n-sims", type=int, default=50)
    ap.add_argument("--jitter", type=float, default=8.0)
    ap.add_argument("--no-reconcile", action="store_true", help="skip gsis reconciliation (debug)")
    ap.add_argument("--baselines", default="", help="validate: comma keys to override BASELINES")
    args = ap.parse_args(argv)
    global _RECONCILE, BASELINES
    _RECONCILE = not args.no_reconcile
    if args.baselines:
        BASELINES = tuple(args.baselines.split(","))

    seats = [int(x) for x in args.seats.split(",")]
    seeds = list(range(args.seed_lo, args.seed_hi))
    if args.years:
        years = [int(x) for x in args.years.split(",")]
    else:
        years = [2021, 2023, 2025] if args.mode == "search" else [2022, 2024, 2026]

    print(
        f"[{args.mode}] years={years} seats={seats} seeds={seeds[0]}-{seeds[-1]} "
        f"n_sims={args.n_sims} jitter={args.jitter}",
        flush=True,
    )

    if args.mode == "search":
        grid = _parse_grid(args.grid)
        results = []
        for cfg in grid:
            per_seat, allv = _run(
                cfg, years, seats, seeds, args.n_sims, args.strat_n_sims, args.jitter
            )
            mean = float(allv.mean())
            per = {s: float(per_seat[s].mean()) for s in seats}
            results.append((cfg, mean, per))
            print(
                f"  {_fmt_cfg(cfg)} | mean={mean:.4f} | "
                + " ".join(f"s{s}={per[s]:.4f}" for s in seats),
                flush=True,
            )
        results.sort(key=lambda r: -r[1])
        print("\n=== RANKED (by mean win%) ===")
        for cfg, mean, per in results:
            print(
                f"  {_fmt_cfg(cfg)} | mean={mean:.4f} | "
                + " ".join(f"s{s}={per[s]:.4f}" for s in seats)
            )
        return 0

    # validate -- candidate: numeric tuple (TiltedFlooredNN) or a string key (e.g. sv_var_timing)
    try:
        cand: object = tuple(float(x) for x in args.config.split(","))
    except ValueError:
        cand = args.config
    contenders: list = [("candidate", cand), *[(k, k) for k in BASELINES]]
    per_seat_all: dict[str, dict[int, np.ndarray]] = {}
    for label, koc in contenders:
        per_seat, allv = _run(koc, years, seats, seeds, args.n_sims, args.strat_n_sims, args.jitter)
        per_seat_all[label] = per_seat
        iv = bootstrap_mean(allv, seed=0)
        print(
            f"[done] {label:<22} pooled win%={iv.point:.4f} [{iv.lo_95:.4f},{iv.hi_95:.4f}]",
            flush=True,
        )

    print(
        f"\n=== VALIDATE candidate {_fmt_cfg(cand) if isinstance(cand, tuple) else cand} | "
        f"years={years} seeds={seeds[0]}-{seeds[-1]} ==="
    )
    print(f"{'strategy':<22}" + "".join(f"{'s' + str(s):>22}" for s in seats) + f"{'POOLED':>22}")
    labels = ["candidate", *BASELINES]
    for label in labels:
        row = f"{label:<22}"
        for s in seats:
            iv = bootstrap_mean(per_seat_all[label][s], seed=0)
            row += f"{iv.point:.4f} [{iv.lo_95:.3f},{iv.hi_95:.3f}]".rjust(22)
        allv = np.concatenate([per_seat_all[label][s] for s in seats])
        iv = bootstrap_mean(allv, seed=0)
        row += f"{iv.point:.4f} [{iv.lo_95:.3f},{iv.hi_95:.3f}]".rjust(22)
        print(row)

    print("\n--- per-seat paired diff: candidate - baseline (CI excludes 0 => *) ---")
    cand_seat = per_seat_all["candidate"]
    win_all = True
    for s in seats:
        print(f" seat {s}:")
        for base in BASELINES:
            iv = bootstrap_mean(cand_seat[s] - per_seat_all[base][s], seed=0)
            sep = iv.lo_95 > 0
            if not (iv.point > 0 and sep):
                win_all = False
            star = "*" if (iv.lo_95 > 0 or iv.hi_95 < 0) else " "
            print(f"   vs {base:<22} {iv.point:+.4f} [{iv.lo_95:+.4f},{iv.hi_95:+.4f}] {star}")
    print(f"\nGOAL MET (candidate CI-beats every baseline at every seat): {win_all}")

    print("\n--- POOLED (all seats) paired diff: candidate - baseline (CI excludes 0 => *) ---")
    cand_all = np.concatenate([cand_seat[s] for s in seats])
    pooled_win = True
    for base in BASELINES:
        base_all = np.concatenate([per_seat_all[base][s] for s in seats])
        iv = bootstrap_mean(cand_all - base_all, seed=0)
        if not (iv.point > 0 and iv.lo_95 > 0):
            pooled_win = False
        star = "*" if (iv.lo_95 > 0 or iv.hi_95 < 0) else " "
        print(f"   vs {base:<22} {iv.point:+.4f} [{iv.lo_95:+.4f},{iv.hi_95:+.4f}] {star}")
    print(f"\nGOAL MET (pooled win% CI-beats every baseline): {pooled_win}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
