"""CLI core for the live draft assistant (testable; scripts/ wraps this).

Reads the draft-state file + consensus VORP table + id_map, runs a strategy, and
prints a ranked recommendation with player names attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.state import load_draft_state
from projections.draft.assistant.strategy import (
    STRATEGY_KEYS,
    DraftStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema

_DEFAULT_ID_MAP = Path("data/raw/id_map.parquet")


def _load_id_map(path: Path) -> pd.DataFrame:
    """Load + validate id_map. Required — it is the position + name source (spec §3.2)."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"id_map parquet not found at {path}; it is required (position + name source)."
        ) from exc
    return IdMapSchema.validate(df)


def _build_strategy(name: str, n_teams: int, sigma: float | None) -> DraftStrategy:
    from projections.draft.assistant.live import build_session_strategy
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import RosterSlot, Ruleset

    # _build_strategy is only called for the analytic strategies (raw_vorp/now_or_never),
    # which ignore availability/n_sims/base_seed; a minimal league carries n_teams + sigma.
    league = LeagueConfig(
        name="_",
        n_teams=n_teams,
        roster_slots={RosterSlot.QB: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )
    return build_session_strategy(
        name, league=league, sigma=sigma, availability=None, n_sims=1, base_seed=0
    )


def generate_recommendation(
    *,
    state_path: Path,
    vorp_path: Path,
    id_map_path: Path,
    strategy_name: str,
    sigma: float | None,
    season: int = 2026,
    n_sims: int = 300,
    data_root: Path = Path("data"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load inputs, run the chosen strategy.

    Returns `(recommendation, id_map)` — the validated id_map is handed back so
    callers (the CLI display path) reuse it instead of re-reading + re-validating.
    """
    id_map = _load_id_map(id_map_path)
    state, league = load_draft_state(state_path, id_map)

    vorp = pd.read_parquet(vorp_path)
    vorp["gsis_id"] = vorp["gsis_id"].astype(_PYARROW_STR)
    vorp = VorpTableSchema.validate(vorp)

    strategy: DraftStrategy
    if strategy_name in ("season_value", "season_value_var", "season_value_timing"):
        availability = load_store_availability(vorp, season=season, data_root=data_root)
        if strategy_name == "season_value":
            strategy = SeasonValueStrategy(availability, n_sims=n_sims, base_seed=0)
        elif strategy_name == "season_value_var":
            # risk-aware variant; pool here lacks is_rookie (live path), so all-veteran.
            strategy = SeasonValueStrategy(
                availability, n_sims=n_sims, base_seed=0, risk_aware=True
            )
        else:
            spread = default_sigma(league.n_teams) if sigma is None else sigma
            strategy = SeasonValueTimingStrategy(
                availability, n_sims=n_sims, base_seed=0, survival=LogisticSurvival(sigma=spread)
            )
    else:
        strategy = _build_strategy(strategy_name, league.n_teams, sigma)
    return strategy.recommend(state, vorp, league), id_map


def format_table(rec: pd.DataFrame, id_map: pd.DataFrame, top: int) -> str:
    """Render the top-N recommendation as a fixed-width text table."""
    names = dict(zip(id_map["gsis_id"], id_map["full_name"], strict=False))
    header = (
        f"{'#':>3}  {'PLAYER':<24} {'POS':<4} {'VORP':>7} {'ADP':>6} {'P(next)':>8} {'SCORE':>8}"
    )
    lines = [header]
    for row in rec.head(top).itertuples(index=False):
        name = str(names.get(row.gsis_id, "-"))[:24]
        adp = f"{float(row.consensus_adp):.1f}" if pd.notna(row.consensus_adp) else "-"
        p_next = f"{float(row.p_available_next):.2f}" if pd.notna(row.p_available_next) else "-"
        star = "*" if row.fills_starting_slot else " "
        lines.append(
            f"{int(row.rank):>3}  {name:<24} {row.position:<4} {row.vorp:>7.1f} "
            f"{adp:>6} {p_next:>8} {row.score:>7.2f}{star}"
        )
    lines.append("  (* = fills an open starting slot)")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live snake-draft pick recommender.")
    p.add_argument("--state", type=Path, required=True, help="Draft-state JSON path.")
    p.add_argument(
        "--vorp-table",
        type=Path,
        required=True,
        help="Consensus VORP parquet (generate_vorp_table.py --source consensus).",
    )
    p.add_argument(
        "--id-map",
        type=Path,
        default=_DEFAULT_ID_MAP,
        help="IdMap parquet (position + names).",
    )
    p.add_argument(
        "--strategy",
        choices=list(STRATEGY_KEYS),
        default="now_or_never",
        help="Recommendation strategy (default now_or_never).",
    )
    p.add_argument("--top", type=int, default=15, help="Rows to print (default 15).")
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Survival spread in picks (default ~ 2/3 of a round).",
    )
    p.add_argument(
        "--season",
        type=int,
        default=2026,
        help=(
            "[--strategy season_value / season_value_timing] target season for byes + availability."
        ),
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=300,
        help="[--strategy season_value / season_value_timing] Monte-Carlo seasons per candidate.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help=(
            "[--strategy season_value / season_value_timing]"
            " store root for weekly_stats/schedules/id_map."
        ),
    )
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rec, id_map = generate_recommendation(
        state_path=args.state,
        vorp_path=args.vorp_table,
        id_map_path=args.id_map,
        strategy_name=args.strategy,
        sigma=args.sigma,
        season=args.season,
        n_sims=args.n_sims,
        data_root=args.data_root,
    )
    print(format_table(rec, id_map, int(args.top)))
    return 0
