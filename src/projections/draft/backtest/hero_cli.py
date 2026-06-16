"""CLI core for the hero-vs-bots eval. scripts/hero_backtest.py wraps this.

Two subcommands: `run` (resumable seat x seed sweep, manifest-guarded) and `report`
(load cached cells, aggregate, print the seat-averaged headline + write the consolidated
results parquet). The report derives strategies + n_seeds from the run manifest, so it can
never silently recompute at the wrong n_sims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projections.draft.assistant.strategy import (
    _DEFAULT_FLOOR,
    _DEFAULT_FLOOR_WEIGHT,
    STRATEGY_KEYS,
)
from projections.draft.assistant.tournament import Interval
from projections.draft.backtest.checkpoint import verify_or_write_manifest
from projections.draft.backtest.harness import StrategyMetrics
from projections.draft.backtest.hero_harness import (
    bot_baseline,
    collect_hero_cells,
    consolidate_cells,
    load_hero_cells,
    paired_diff,
    seat_averaged_metrics,
)
from projections.draft.backtest.inputs import load_inputs
from projections.draft.league_config import LeagueConfig

_DEFAULT_STRATEGIES = (
    "raw_vorp,now_or_never,now_or_never_floored,season_value,season_value_var,season_value_timing"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hero-vs-bots strategy evaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--league-config", type=Path, required=True)
        sp.add_argument("--season", type=int, default=2025)
        sp.add_argument("--data-root", type=Path, default=Path("data"))
        sp.add_argument("--checkpoint-dir", type=Path, default=Path("_hero_ckpt"))

    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--strategies", default=_DEFAULT_STRATEGIES)  # run-only; report reads manifest
    r.add_argument("--n-seeds", type=int, default=40)
    r.add_argument("--strategy-n-sims", type=int, default=50)
    r.add_argument("--jitter", type=float, default=8.0)
    r.add_argument("--floor", type=float, default=_DEFAULT_FLOOR)
    r.add_argument("--floor-weight", type=float, default=_DEFAULT_FLOOR_WEIGHT)

    rep = sub.add_parser("report")
    _common(rep)
    rep.add_argument("--reference", choices=list(STRATEGY_KEYS), default="now_or_never")
    rep.add_argument(
        "--out-parquet", type=Path, default=Path("data/backtest/hero_eval/results.parquet")
    )
    return p.parse_args(argv)


def _run_key(args: argparse.Namespace) -> dict[str, object]:
    """Manifest run identity (pure -> testable)."""
    return {
        "season": args.season,
        "config": str(args.league_config),
        "n_seeds": args.n_seeds,
        "strategies": args.strategies,
        "jitter": args.jitter,
        "strategy_n_sims": args.strategy_n_sims,
        "floor": args.floor,
        "floor_weight": args.floor_weight,
    }


def _run(args: argparse.Namespace) -> int:
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    verify_or_write_manifest(args.checkpoint_dir, _run_key(args))
    inputs = load_inputs(season=args.season, config=config, data_root=args.data_root)
    cells = collect_hero_cells(
        seed_lo=0,
        seed_hi=args.n_seeds,
        strategies=tuple(args.strategies.split(",")),
        season=args.season,
        pool=inputs.pool,
        config=config,
        availability=inputs.availability,
        proj_lookup=inputs.proj_lookup,
        actual_lookup=inputs.actual_lookup,
        calendar=inputs.calendar,
        jitter=args.jitter,
        strategy_n_sims=args.strategy_n_sims,
        base_seed=0,
        floor=args.floor,
        floor_weight=args.floor_weight,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"[hero] {len(cells)} cells complete in {args.checkpoint_dir}")
    return 0


def _report(args: argparse.Namespace) -> int:
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    inputs = load_inputs(season=args.season, config=config, data_root=args.data_root)
    # The manifest is the run identity: derive strategies + n_seeds from it (NOT from CLI
    # args), and LOAD cells (fail loud on any missing) -- report never recomputes.
    manifest = json.loads((args.checkpoint_dir / "manifest.json").read_text())
    strategies = tuple(str(manifest["strategies"]).split(","))
    cells = load_hero_cells(
        seed_hi=int(manifest["n_seeds"]),
        strategies=strategies,
        season=args.season,
        n_teams=config.n_teams,
        checkpoint_dir=args.checkpoint_dir,
    )
    df = consolidate_cells(cells)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)  # derived report artifact (spec §6, not the store)
    seat_avg = seat_averaged_metrics(df, scoring="actual")
    base = bot_baseline(inputs.calendar, config.n_teams)
    print(_format_headline(seat_avg, base, config.n_teams, reference=args.reference, df=df))
    return 0


def _pct(iv: Interval) -> str:
    return f"{iv.point * 100:5.1f}% [{iv.lo_95 * 100:.1f},{iv.hi_95 * 100:.1f}]"


def _format_headline(
    seat_avg: dict[str, StrategyMetrics],
    base: StrategyMetrics,
    n_teams: int,
    *,
    reference: str,
    df: object,
) -> str:
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    head = (
        f"{'STRATEGY':<22} {'WIN%':>20} {'PLAYOFF%':>20} {'CHAMP%':>20} "
        f"{'PTS FOR':>10} {'dWIN% vs ' + reference:>16}"
    )
    rows = [f"[HERO-VS-BOTS -- ACTUAL, {n_teams} teams; bot = avg team is structural]", head]
    # The paired-diff column needs the reference strategy in the run; if it's absent
    # (e.g. a partial-strategy run) show "n/a" rather than a nan from an empty paired array.
    ref_present = reference in seat_avg
    for name, m in sorted(seat_avg.items()):
        if ref_present:
            d = paired_diff(
                df, scoring="actual", metric="win_pct", strategy=name, reference=reference
            )
            dcol = f"{d.point * 100:>+15.1f}"
        else:
            dcol = f"{'n/a':>16}"
        rows.append(
            f"{name:<22} {_pct(m.win_pct):>20} {_pct(m.playoff):>20} {_pct(m.championship):>20} "
            f"{m.points_for.point:>10.1f} {dcol}"
        )
    rows.append(
        f"{'bot (avg team)':<22} {base.win_pct.point * 100:>5.1f}%{'':>14} "
        f"{base.playoff.point * 100:>5.1f}%{'':>14} {base.championship.point * 100:>5.1f}%{'':>14} "
        f"{'-':>10} {'-':>16}"
    )
    return "\n".join(rows)
