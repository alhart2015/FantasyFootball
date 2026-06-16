"""CLI core for the live draft assistant (testable; scripts/ wraps this).

Reads the draft-state file + consensus VORP table + id_map, runs a strategy, and
prints a ranked recommendation with player names attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.live import MC_STRATEGIES, build_session_strategy
from projections.draft.assistant.state import load_draft_state
from projections.draft.assistant.strategy import (
    _DEFAULT_FLOOR,
    _DEFAULT_FLOOR_WEIGHT,
    STRATEGY_KEYS,
    DraftStrategy,
)
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
    floor: float = _DEFAULT_FLOOR,
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT,
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

    # Single strategy-construction seam (shared with the live board + resume): load
    # availability only for the MC strategies, then build by name.
    availability = None
    if strategy_name in MC_STRATEGIES:
        availability = load_store_availability(vorp, season=season, data_root=data_root)
    strategy: DraftStrategy = build_session_strategy(
        strategy_name,
        league=league,
        sigma=sigma,
        availability=availability,
        n_sims=n_sims,
        base_seed=0,
        floor=floor,
        floor_weight=floor_weight,
    )
    rec = strategy.recommend(state, vorp, league)
    if "full_name" in vorp.columns:
        # Carry the pool's display name (incl. placeholder-gsis rookies absent from id_map)
        # so format_table can prefer it over id_map — consistent with the live board.
        rec = rec.merge(vorp[["gsis_id", "full_name"]], on="gsis_id", how="left")
    return rec, id_map


def format_table(rec: pd.DataFrame, id_map: pd.DataFrame, top: int) -> str:
    """Render the top-N recommendation as a fixed-width text table.

    Display name prefers the pool's full_name (carried on rec by generate_recommendation,
    incl. placeholder-gsis rookies) and falls back to id_map, then '-'.
    """
    names = dict(zip(id_map["gsis_id"], id_map["full_name"], strict=False))
    has_pool_name = "full_name" in rec.columns
    header = (
        f"{'#':>3}  {'PLAYER':<24} {'POS':<4} {'VORP':>7} {'ADP':>6} {'P(next)':>8} {'SCORE':>8}"
    )
    lines = [header]
    for row in rec.head(top).itertuples(index=False):
        pool_name = getattr(row, "full_name", None) if has_pool_name else None
        resolved = (
            pool_name
            if pool_name is not None and pd.notna(pool_name)
            else names.get(row.gsis_id, "-")
        )
        name = str(resolved)[:24]
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
        "--floor",
        type=float,
        default=_DEFAULT_FLOOR,
        help="[--strategy now_or_never_floored] absolute VORP quality bar F.",
    )
    p.add_argument(
        "--floor-weight",
        type=float,
        default=_DEFAULT_FLOOR_WEIGHT,
        help="[--strategy now_or_never_floored] hinge weight (0 = plain now_or_never).",
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
        floor=args.floor,
        floor_weight=args.floor_weight,
    )
    print(format_table(rec, id_map, int(args.top)))
    return 0
