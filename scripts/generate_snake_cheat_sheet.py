"""Generate a snake-draft cheat sheet from a VORP parquet.

Reads:
  - --vorp-input    : VorpTableSchema parquet
  - --league-config : LeagueConfig JSON
  - --id-map        : IdMapSchema parquet (optional; warns + uses '—' if missing)

Writes:
  - --out           : .csv or .parquet (sniffed by extension)

Per-position stdout summary printed for eyeball mitigation (see spec §4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.draft.snake_cheat_sheet import generate_snake_cheat_sheet
from projections.schemas import _PYARROW_STR, IdMapSchema, Position

_POSITION_ORDER: tuple[Position, ...] = (
    Position.QB,
    Position.RB,
    Position.WR,
    Position.TE,
    Position.K,
    Position.DST,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a snake-draft cheat sheet.")
    p.add_argument("--season", type=int, required=True, help="Season (metadata only).")
    p.add_argument("--league-config", type=Path, required=True, help="LeagueConfig JSON path.")
    p.add_argument("--vorp-input", type=Path, required=True, help="VORP parquet path.")
    p.add_argument(
        "--id-map",
        type=Path,
        default=Path("data/raw/id_map.parquet"),
        help="IdMap parquet path (optional; warns + falls back to '—' if missing).",
    )
    p.add_argument(
        "--tiers-per-position",
        type=int,
        default=8,
        help="Number of tiers per position (default 8).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output path (.csv or .parquet).")
    return p.parse_args()


def _load_display_names(path: Path) -> pd.DataFrame | None:
    """Read id_map.parquet → (gsis_id, display_name). Returns None on missing."""
    if not path.exists():
        print(
            f"WARNING: id_map parquet not found at {path}; display names will be '—'",
            file=sys.stderr,
        )
        return None
    df = pd.read_parquet(path)
    df = IdMapSchema.validate(df)
    return pd.DataFrame(
        {
            "gsis_id": df["gsis_id"].astype(_PYARROW_STR),
            "display_name": df["full_name"].astype(_PYARROW_STR),
        }
    )


def _write_output(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"--out must be .csv or .parquet (got '{path.suffix}'); use one of these extensions."
        )


def _log_per_position_summary(df: pd.DataFrame, n_tiers: int, ruleset: str, season: int) -> None:
    n_total = len(df)
    print(
        f"Snake cheat sheet written: {n_total} players, season={season}, "
        f"ruleset={ruleset}, tiers_per_position={n_tiers}"
    )
    print()
    print("Position summary (n_in_pool | tier-1 size | top-3):")
    for pos in _POSITION_ORDER:
        sub = df[df["position"] == pos.value]
        if sub.empty:
            continue
        in_pool = sub[sub["is_in_pool"]]
        tier_1_size = int((in_pool["tier"] == 1).sum())
        top3 = sub.head(3)
        top_str = ", ".join(
            f"{row['display_name']} ({pos.value}{int(row['positional_rank'])}, "
            f"T{int(row['tier'])}, VORP{row['vorp']:+.1f})"
            for _, row in top3.iterrows()
            if pd.notna(row["tier"])
        )
        print(f"  {pos.value}  in_pool={len(in_pool):>4}  tier1={tier_1_size:>2}  top: {top_str}")


def main() -> int:
    args = _parse_args()
    cfg = LeagueConfig.model_validate_json(args.league_config.read_text())
    vorp = pd.read_parquet(args.vorp_input)
    display = _load_display_names(args.id_map)
    sheet = generate_snake_cheat_sheet(
        vorp,
        cfg,
        display_names=display,
        tiers_per_position=int(args.tiers_per_position),
    )
    _write_output(sheet, args.out)
    _log_per_position_summary(
        sheet, int(args.tiers_per_position), cfg.ruleset.name, int(args.season)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
