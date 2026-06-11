"""Sanity checks for the draft tournament: survival-model calibration + sample rosters.

Two diagnostics against the real consensus VORP pool:

1. CALIBRATION — the survival model claims P(player available at pick N). Simulate many
   ADP-bot draft prefixes and compare the model's predicted availability to the empirical
   availability at the hero's next pick. A well-calibrated model tracks the diagonal.
2. SAMPLE ROSTERS — run one draft (fixed seed) with NowOrNeverStrategy and with
   RawVorpStrategy from the same slot, and print the two hero rosters side by side, each
   scored by BOTH the starters metric (best single-week lineup) and the season metric
   (expected points under injuries + byes). The gap between the two exposes bench quality:
   a QB-hoarding roster barely moves on starters but bleeds heavily on the season metric.

Usage:
    python scripts/survival_sanity_check.py --vorp-table <parquet> --league-config <json> \
        --my-slot 6 [--sigma N] [--adp-jitter F] [--seeds K] \
        [--season 2026] [--n-sims 4000] [--data-root data]
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import bot_pick
from projections.draft.assistant.pick_timing import my_next_pick
from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.simulation import simulate_draft
from projections.draft.assistant.strategy import NowOrNeverStrategy, RawVorpStrategy
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.assistant.tournament_cli import _build_season_valuer
from projections.draft.assistant.valuer import SeasonValuer
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema


def _load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def _names(id_map_path: Path) -> dict[str, str]:
    if not id_map_path.exists():
        return {}
    m = pd.read_parquet(id_map_path)
    return {str(g): str(n) for g, n in zip(m["gsis_id"], m["full_name"], strict=False)}


def _empirical_availability(
    pool: pd.DataFrame, config: LeagueConfig, *, at_pick: int, adp_jitter: float, seeds: int
) -> dict[str, float]:
    """Fraction of ADP-bot drafts (all seats bots) in which each player is still
    available immediately before `at_pick`."""
    gsis_all = pool["gsis_id"].astype(str).tolist()
    avail_count: Counter[str] = Counter()
    for s in range(seeds):
        rng = np.random.default_rng(s)
        drafted: set[str] = set()
        for _ in range(at_pick - 1):  # picks before at_pick
            available = pool[~pool["gsis_id"].isin(drafted)]
            drafted.add(str(bot_pick(available, rng, adp_jitter=adp_jitter)))
        for g in gsis_all:
            if g not in drafted:
                avail_count[g] += 1
    return {g: avail_count[g] / seeds for g in gsis_all}


def _calibration(
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    sigma: float,
    adp_jitter: float,
    seeds: int,
) -> None:
    next_pick = my_next_pick(my_slot, my_slot, config.n_teams, config.roster_size)
    assert next_pick is not None
    survival = LogisticSurvival(sigma=sigma)
    empirical = _empirical_availability(
        pool, config, at_pick=next_pick, adp_jitter=adp_jitter, seeds=seeds
    )

    rows = []
    for gid, adp in zip(pool["gsis_id"].astype(str), pool["consensus_adp"], strict=True):
        if pd.isna(adp):
            continue
        rows.append((gid, float(adp), survival.p_available(float(adp), next_pick), empirical[gid]))
    rows.sort(key=lambda r: r[2])  # by predicted p

    print(
        f"\n=== SURVIVAL CALIBRATION (slot {my_slot}: decide at pick {my_slot}, survive to "
        f"pick {next_pick}; sigma={sigma}, adp_jitter={adp_jitter}, {seeds} bot drafts) ==="
    )
    print(f"{'predicted-p bin':>16} {'n':>4} {'mean pred':>10} {'mean empirical':>15}")
    preds = np.array([r[2] for r in rows])
    emps = np.array([r[3] for r in rows])
    edges = np.linspace(0, 1, 11)
    for lo, hi in pairwise(edges):
        mask = (preds >= lo) & (preds < hi) if hi < 1 else (preds >= lo) & (preds <= hi)
        if mask.sum() == 0:
            continue
        print(
            f"  [{lo:.1f}, {hi:.1f})   {int(mask.sum()):>4} "
            f"{preds[mask].mean():>10.3f} {emps[mask].mean():>15.3f}"
        )
    err = float(np.mean(np.abs(preds - emps)))
    print(f"  mean |predicted - empirical| over {len(rows)} ADP'd players: {err:.3f}")

    # Players nearest the survival boundary (~50/50) — the most informative cases.
    boundary = sorted(rows, key=lambda r: abs(r[2] - 0.5))[:5]
    print("  near-50/50 players (adp | predicted | empirical):")
    for gid, adp, pred, emp in boundary:
        print(f"    {gid}  adp={adp:>6.1f}  pred={pred:.2f}  empirical={emp:.2f}")


def _print_roster(
    label: str,
    roster: pd.DataFrame,
    names: dict[str, str],
    config: LeagueConfig,
    season_valuer: SeasonValuer,
) -> None:
    avail = season_valuer.availability
    starters = optimal_lineup_points(roster, config.roster_slots)
    season = season_valuer.value(roster, config.roster_slots)
    pct = (season - starters) / starters * 100 if starters else 0.0
    by_pos = Counter(roster["position"].astype(str))
    print(f"\n--- {label}: counts {dict(sorted(by_pos.items()))} ---")
    print(f"    starters metric (best 1-week lineup) : {starters:8.1f}")
    print(f"    season   metric (E[pts] inj + byes)  : {season:8.1f}  ({pct:+.1f}% vs starters)")
    ordered = roster.sort_values(["position", "season_mean_fpts"], ascending=[True, False])
    print(f"  {'PLAYER':<24} {'POS':<4} {'PROJ':>6} {'VORP':>7} {'p_wk':>5} {'bye':>4}")
    for row in ordered.itertuples(index=False):
        gid = str(row.gsis_id)
        name = names.get(gid, gid)[:24]
        bye = avail.bye_week(gid)
        print(
            f"  {name:<24} {row.position:<4} {row.season_mean_fpts:>6.1f} {row.vorp:>7.1f} "
            f"{avail.p_week(gid):>5.2f} {(str(bye) if bye else '-'):>4}"
        )


def _sample_rosters(
    pool: pd.DataFrame,
    config: LeagueConfig,
    names: dict[str, str],
    *,
    my_slot: int,
    sigma: float,
    adp_jitter: float,
    season: int,
    n_sims: int,
    data_root: Path,
) -> None:
    print(
        f"\n=== SAMPLE ROSTERS (slot {my_slot}, seed 0, adp_jitter={adp_jitter}, sigma={sigma}; "
        f"season metric: {season}, n_sims={n_sims}) ==="
    )
    season_valuer = _build_season_valuer(
        pool, season=season, n_sims=n_sims, base_seed=0, data_root=data_root
    )
    non = NowOrNeverStrategy(LogisticSurvival(sigma=sigma))
    raw = RawVorpStrategy()
    r_non = simulate_draft(
        non, my_slot, pool, config, adp_jitter=adp_jitter, rng=np.random.default_rng(0)
    )
    r_raw = simulate_draft(
        raw, my_slot, pool, config, adp_jitter=adp_jitter, rng=np.random.default_rng(0)
    )
    _print_roster("now_or_never", r_non, names, config, season_valuer)
    _print_roster("raw_vorp", r_raw, names, config, season_valuer)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Survival calibration + sample rosters.")
    p.add_argument("--vorp-table", type=Path, required=True)
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--id-map", type=Path, default=Path("data/raw/id_map.parquet"))
    p.add_argument("--my-slot", type=int, default=6)
    p.add_argument(
        "--sigma", type=float, default=None, help="Survival sigma (default ~2/3 of a round)."
    )
    p.add_argument(
        "--adp-jitter", type=float, default=None, help="Bot ADP noise SD (default ~2/3 of a round)."
    )
    p.add_argument(
        "--seeds", type=int, default=400, help="Bot drafts for the calibration estimate."
    )
    p.add_argument(
        "--season", type=int, default=2026, help="Target season for the season-metric byes."
    )
    p.add_argument(
        "--n-sims", type=int, default=4000, help="Monte-Carlo seasons per roster (season metric)."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Store root for the season metric's weekly_stats/schedules/id_map.",
    )
    args = p.parse_args(argv)

    pool = _load_pool(args.vorp_table)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    names = _names(args.id_map)
    sigma = default_sigma(config.n_teams) if args.sigma is None else args.sigma
    jitter = default_sigma(config.n_teams) if args.adp_jitter is None else args.adp_jitter

    _calibration(
        pool, config, my_slot=args.my_slot, sigma=sigma, adp_jitter=jitter, seeds=args.seeds
    )
    _sample_rosters(
        pool,
        config,
        names,
        my_slot=args.my_slot,
        sigma=sigma,
        adp_jitter=jitter,
        season=args.season,
        n_sims=args.n_sims,
        data_root=args.data_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
