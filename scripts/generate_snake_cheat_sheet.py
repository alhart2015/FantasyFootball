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
from projections.draft.snake_cheat_sheet import (
    DISPLAY_NAME_FALLBACK,
    generate_snake_cheat_sheet,
)
from projections.schemas import _PYARROW_STR, IdMapSchema, Position

#: Column contract for the (gsis_id, display_name) frame the cheat sheet maps names through.
_NAME_COLUMNS = ["gsis_id", "display_name"]


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
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(
            f"WARNING: id_map parquet not found at {path}; falling back to the VORP "
            "table's own names",
            file=sys.stderr,
        )
        return None
    df = IdMapSchema.validate(df)
    return pd.DataFrame(
        {
            "gsis_id": df["gsis_id"].astype(_PYARROW_STR),
            "display_name": df["full_name"].astype(_PYARROW_STR),
        }
    )


def _resolve_display_names(vorp: pd.DataFrame, id_map_names: pd.DataFrame | None) -> pd.DataFrame:
    """id_map names, with the VORP table's own `full_name` filling every gap.

    The id_map is keyed on real GSIS ids, so it cannot cover players who do not have one
    yet — incoming rookies carry a synthetic `99-` id until nfl_data_py issues a real one.
    On the 2026 Critts pool that was 89 of 579 players, 32 of them inside the 13 drafted
    rounds, including a top-15 VORP RB at ADP 22. They rendered as the '—' placeholder,
    which makes a draft-day cheat sheet actively unusable at exactly the picks that matter.

    `full_name` is an optional, nullable column on `VorpTableSchema` (populated on the
    consensus-fed path only), so this degrades to id_map-only when it is absent or null
    rather than assuming it is there.
    """
    if "full_name" not in vorp.columns:
        return id_map_names if id_map_names is not None else pd.DataFrame(columns=_NAME_COLUMNS)

    pool_names = pd.DataFrame(
        {
            "gsis_id": vorp["gsis_id"].astype(_PYARROW_STR),
            "display_name": vorp["full_name"].astype(_PYARROW_STR),
        }
    ).dropna(subset=["display_name"])
    if id_map_names is None or id_map_names.empty:
        return pool_names

    # id_map wins where it has a row; the pool only fills gaps. concat + drop_duplicates
    # keeps the first occurrence, so ordering encodes the precedence.
    merged = pd.concat([id_map_names, pool_names], ignore_index=True)
    return merged.drop_duplicates(subset="gsis_id", keep="first").reset_index(drop=True)


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
    for pos in Position:
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
        if "adp_delta" in sub.columns and sub["adp_delta"].notna().any():
            best_value = sub.loc[sub["adp_delta"].idxmax()]
            biggest_reach = sub.loc[sub["adp_delta"].idxmin()]
            print(
                f"      ADP value: {best_value['display_name']} "
                f"(delta {int(best_value['adp_delta']):+d}); "
                f"reach: {biggest_reach['display_name']} "
                f"(delta {int(biggest_reach['adp_delta']):+d})"
            )


def main() -> int:
    args = _parse_args()
    cfg = LeagueConfig.model_validate_json(args.league_config.read_text())
    vorp = pd.read_parquet(args.vorp_input)
    display = _resolve_display_names(vorp, _load_display_names(args.id_map))
    n_unnamed = int((display["display_name"] == DISPLAY_NAME_FALLBACK).sum())
    missing = len(vorp) - vorp["gsis_id"].astype(_PYARROW_STR).isin(display["gsis_id"]).sum()
    if missing or n_unnamed:
        print(
            f"WARNING: {missing + n_unnamed} of {len(vorp)} players have no resolvable name "
            f"and will render as '{DISPLAY_NAME_FALLBACK}'.",
            file=sys.stderr,
        )
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
