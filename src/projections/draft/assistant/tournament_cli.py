"""CLI core for the strategy comparison harness (spec §3.7). scripts/ wraps this.

Two modes over a consensus VORP parquet + a LeagueConfig:
  compare    -- run the registered strategies, print per-strategy CI + winner.
  tune-sigma -- sweep the survival sigma, print the grid + recommended sigma.
The --league-config MUST match the ruleset the VORP table was built under
(the parquet carries no ruleset column to verify it -- spec §3.1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability import build_availability
from projections.draft.assistant.strategy import NowOrNeverStrategy, RawVorpStrategy
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.assistant.tournament import (
    SigmaTuningResult,
    TournamentResult,
    run_tournament,
    tune_sigma,
)
from projections.draft.assistant.valuer import RosterValuer, SeasonValuer, StartersValuer
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.store import read_partition


def _load_pool(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(df)


def _load_config(path: Path) -> LeagueConfig:
    return LeagueConfig.model_validate_json(path.read_text())


def _default_sigma_grid(n_teams: int) -> list[float]:
    base = default_sigma(n_teams)
    return [round(f * base, 3) for f in (1 / 3, 1 / 2, 2 / 3, 1.0, 4 / 3)]


_HISTORY_SEASONS = range(2018, 2025)  # weekly_stats coverage for the availability model


def _build_season_valuer(
    pool: pd.DataFrame, *, season: int, n_sims: int, base_seed: int, data_root: Path
) -> SeasonValuer:
    raw = data_root / "raw"
    frames: list[pd.DataFrame] = []
    for yr in _HISTORY_SEASONS:
        try:
            frames.append(read_partition(raw, "weekly_stats", season=yr))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(f"no weekly_stats partitions under {raw} for {_HISTORY_SEASONS}")
    weekly_stats = pd.concat(frames, ignore_index=True)
    schedules = read_partition(raw, "schedules", season=season)
    id_map = pd.read_parquet(raw / "id_map.parquet")
    availability = build_availability(weekly_stats, schedules, id_map, pool, season=season)
    return SeasonValuer(availability=availability, n_sims=n_sims, base_seed=base_seed)


def _parse_sigma_grid(raw: str) -> list[float]:
    """Parse a comma-separated sigma grid; tolerate whitespace/trailing commas, reject junk.

    Value positivity is enforced by tune_sigma (the engine), so every caller is covered.
    """
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"--sigma-grid had no numeric values: {raw!r}")
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"--sigma-grid has a non-numeric value: {raw!r}") from exc


def format_compare(result: TournamentResult) -> str:
    lines = [
        f"Strategy tournament -- {result.n_seeds} seeds, my_slot={result.my_slot}, "
        f"adp_jitter={result.adp_jitter:.2f}, base_seed={result.base_seed}",
        f"{'STRATEGY':<16} {'MEAN':>9} {'95% CI':>22}",
    ]
    for name, ci in sorted(result.summaries.items(), key=lambda kv: kv[1].point, reverse=True):
        lines.append(f"{name:<16} {ci.point:>9.2f}  [{ci.lo_95:>8.2f}, {ci.hi_95:>8.2f}]")
    if result.diff is not None:
        lines.append(
            f"\nTop-two paired diff: {result.diff.point:+.2f} "
            f"[{result.diff.lo_95:+.2f}, {result.diff.hi_95:+.2f}]"
        )
    lines.append(f"Winner: {result.winner if result.winner else 'no separation (CI brackets 0)'}")
    return "\n".join(lines)


def format_tune(result: SigmaTuningResult) -> str:
    lines = [
        f"Sigma tuning -- {result.n_seeds} seeds, my_slot={result.my_slot}, "
        f"adp_jitter={result.adp_jitter:.2f}",
        f"{'SIGMA':>8} {'MEAN':>9}",
    ]
    for sigma, mean in result.grid:
        lines.append(f"{sigma:>8.3f} {mean:>9.2f}")
    lines.append(f"\nRecommended sigma: {result.best_sigma:.3f}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draft strategy comparison harness.")
    p.add_argument(
        "--vorp-table",
        type=Path,
        required=True,
        help="Consensus VORP parquet (generate_vorp_table.py --source consensus).",
    )
    p.add_argument(
        "--league-config",
        type=Path,
        required=True,
        help="LeagueConfig JSON -- must match the table's ruleset (spec 3.1).",
    )
    p.add_argument("--my-slot", type=int, required=True, help="Hero draft slot (1-based).")
    p.add_argument("--seeds", type=int, default=200, help="Paired draft sims per strategy.")
    p.add_argument(
        "--adp-jitter",
        type=float,
        default=None,
        help="Bot ADP noise SD in picks (default ~2/3 of a draft round, i.e. 2/3*n_teams).",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (reproducibility).")
    p.add_argument(
        "--valuer",
        choices=["starters", "season"],
        default="starters",
        help="Roster metric: 'starters' (optimal single-week lineup) or "
        "'season' (expected points under availability + byes).",
    )
    p.add_argument(
        "--season",
        type=int,
        default=2026,
        help="[--valuer season] target season for byes + availability.",
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=300,
        help="[--valuer season] Monte-Carlo seasons per roster.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="[--valuer season] store root for weekly_stats/schedules/id_map.",
    )
    sub = p.add_subparsers(dest="mode", required=True)
    cmp_p = sub.add_parser("compare", help="Compare now_or_never vs raw_vorp.")
    cmp_p.add_argument(
        "--strategy-sigma",
        type=float,
        default=None,
        help="Survival sigma for now_or_never (default ~2/3 of a round).",
    )
    tune_p = sub.add_parser("tune-sigma", help="Sweep survival sigma for now_or_never.")
    tune_p.add_argument(
        "--sigma-grid",
        type=str,
        default=None,
        help="Comma-separated sigmas (default centered on 2/3*n_teams).",
    )
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pool = _load_pool(args.vorp_table)
    config = _load_config(args.league_config)
    jitter = default_sigma(config.n_teams) if args.adp_jitter is None else args.adp_jitter

    valuer: RosterValuer = (
        StartersValuer()
        if args.valuer == "starters"
        else _build_season_valuer(
            pool,
            season=args.season,
            n_sims=args.n_sims,
            base_seed=args.seed,
            data_root=args.data_root,
        )
    )

    if args.mode == "compare":
        sigma = (
            default_sigma(config.n_teams) if args.strategy_sigma is None else args.strategy_sigma
        )
        result = run_tournament(
            {
                "now_or_never": NowOrNeverStrategy(LogisticSurvival(sigma=sigma)),
                "raw_vorp": RawVorpStrategy(),
            },
            pool=pool,
            config=config,
            my_slot=args.my_slot,
            n_seeds=args.seeds,
            adp_jitter=jitter,
            base_seed=args.seed,
            valuer=valuer,
        )
        print(format_compare(result))
        return 0

    grid = (
        _default_sigma_grid(config.n_teams)
        if args.sigma_grid is None
        else _parse_sigma_grid(args.sigma_grid)
    )
    tuned = tune_sigma(
        grid,
        pool=pool,
        config=config,
        my_slot=args.my_slot,
        n_seeds=args.seeds,
        adp_jitter=jitter,
        base_seed=args.seed,
        valuer=valuer,
    )
    print(format_tune(tuned))
    return 0
