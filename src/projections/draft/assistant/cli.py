"""CLI core for the live draft assistant (testable; scripts/ wraps this).

Reads the draft-state file + consensus VORP table + id_map, runs a strategy, and
prints a ranked recommendation with player names attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.state import load_draft_state
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.schemas import _PYARROW_STR, VorpTableSchema

_DEFAULT_ID_MAP = Path("data/raw/id_map.parquet")


def _load_id_map(path: Path) -> pd.DataFrame:
    """Load id_map and verify required columns exist.

    The CLI uses only gsis_id, position, and full_name (for display + position
    resolution). Full IdMapSchema validation is not applied here because the CLI
    must accept a lightweight fixture (3-column parquet) used in tests and in
    generate_vorp_table pipelines that do not carry all cross-platform ID columns.
    """
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"id_map parquet not found at {path}; it is required (position + name source)."
        ) from exc
    missing = [c for c in ("gsis_id", "position", "full_name") if c not in df.columns]
    if missing:
        raise ValueError(f"id_map is missing required columns: {missing}")
    return df


def _build_strategy(name: str, n_teams: int, sigma: float | None) -> DraftStrategy:
    if name == "raw_vorp":
        return RawVorpStrategy()
    if name == "now_or_never":
        spread = default_sigma(n_teams) if sigma is None else sigma
        return NowOrNeverStrategy(LogisticSurvival(sigma=spread))
    raise ValueError(f"unknown strategy {name!r}")


def generate_recommendation(
    *,
    state_path: Path,
    vorp_path: Path,
    id_map_path: Path,
    strategy_name: str,
    sigma: float | None,
) -> pd.DataFrame:
    """Load inputs, run the chosen strategy, return a RecommendationSchema frame."""
    id_map = _load_id_map(id_map_path)
    state, league = load_draft_state(state_path, id_map)

    vorp = pd.read_parquet(vorp_path)
    vorp["gsis_id"] = vorp["gsis_id"].astype(_PYARROW_STR)
    vorp = VorpTableSchema.validate(vorp)

    strategy = _build_strategy(strategy_name, league.n_teams, sigma)
    return strategy.recommend(state, vorp, league)


def format_table(rec: pd.DataFrame, id_map: pd.DataFrame, top: int) -> str:
    """Render the top-N recommendation as a fixed-width text table."""
    names = dict(zip(id_map["gsis_id"], id_map["full_name"], strict=False))
    header = (
        f"{'#':>3}  {'PLAYER':<24} {'POS':<4} {'VORP':>7} {'ADP':>6} {'P(next)':>8} {'SCORE':>8}"
    )
    lines = [header]
    for row in rec.head(top).itertuples(index=False):
        name = str(names.get(row.gsis_id, "-"))
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
        choices=["now_or_never", "raw_vorp"],
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
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rec = generate_recommendation(
        state_path=args.state,
        vorp_path=args.vorp_table,
        id_map_path=args.id_map,
        strategy_name=args.strategy,
        sigma=args.sigma,
    )
    id_map = _load_id_map(args.id_map)
    print(format_table(rec, id_map, int(args.top)))
    return 0
