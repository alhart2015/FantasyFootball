"""CLI: convert per-player VORP into per-player auction $ for a given LeagueConfig.

Reads:
    --league-config  Path to LeagueConfig JSON.
    --vorp-input     Path to VORP parquet (gsis_id, position, season_mean_fpts, vorp).
    --reference-prices  Optional CSV with gsis_id, reference_dollars columns.
Writes:
    --out            Output path; .csv and .parquet supported (extension-sniffed).

Spec: docs/superpowers/specs/2026-05-16-auction-values-design.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.schemas import Position


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="Season (metadata only).")
    parser.add_argument("--league-config", type=Path, required=True, help="LeagueConfig JSON path.")
    parser.add_argument("--vorp-input", type=Path, required=True, help="VORP parquet path.")
    parser.add_argument(
        "--reference-prices",
        type=Path,
        default=None,
        help="Optional CSV with gsis_id, reference_dollars columns.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output path (.csv or .parquet).")
    return parser.parse_args()


def _read_vorp(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"vorp-input parquet does not exist: {path}")
    df = pd.read_parquet(path)
    required = {"gsis_id", "position", "season_mean_fpts", "vorp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"vorp-input parquet is missing required columns: {sorted(missing)}")
    df["gsis_id"] = df["gsis_id"].astype(pd.StringDtype("pyarrow"))
    df["position"] = df["position"].astype(pd.StringDtype("pyarrow"))
    df["season_mean_fpts"] = df["season_mean_fpts"].astype("float64")
    df["vorp"] = df["vorp"].astype("float64")
    return df


def _read_reference_prices(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"reference-prices CSV does not exist: {path}")
    df = pd.read_csv(path)
    required = {"gsis_id", "reference_dollars"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reference-prices CSV is missing required columns: {sorted(missing)}")
    df["gsis_id"] = df["gsis_id"].astype(pd.StringDtype("pyarrow"))
    df["reference_dollars"] = df["reference_dollars"].astype(pd.Int64Dtype())
    return df


def _write_output(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        sorted_df = df.sort_values(
            by=["pool_rank", "auction_dollars", "gsis_id"],
            ascending=[True, False, True],
            na_position="last",
            kind="mergesort",
        )
        sorted_df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"unsupported output extension {path.suffix!r}; expected .csv or .parquet."
        )


def _emit_summary(df: pd.DataFrame, season: int) -> None:
    """Per-position summary printed to stdout. Risk mitigation per spec §6."""
    in_pool = df[df["in_pool"]]
    print(
        f"Auction values for season {season}: {len(in_pool)} in-pool players, "
        f"sum auction_dollars = ${int(in_pool['auction_dollars'].sum())}"
    )
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST):
        pos_df = in_pool[in_pool["position"] == pos.value]
        if len(pos_df) == 0:
            continue
        top = pos_df.sort_values("auction_dollars", ascending=False).head(3)
        top_summary = ", ".join(
            f"{row.gsis_id}: ${int(row.auction_dollars)} (vorp {row.vorp:.1f})"
            for row in top.itertuples()
        )
        print(
            f"  {pos.value}: n={len(pos_df)}, "
            f"min/median/max vorp = {pos_df['vorp'].min():.1f} / "
            f"{pos_df['vorp'].median():.1f} / {pos_df['vorp'].max():.1f}; "
            f"top3 = [{top_summary}]"
        )


def main() -> None:
    args = _parse_args()
    league_config = LeagueConfig.model_validate_json(args.league_config.read_text())
    vorp_table = _read_vorp(args.vorp_input)
    reference_prices = _read_reference_prices(args.reference_prices)
    out = generate_auction_values(vorp_table, league_config, reference_prices=reference_prices)
    _emit_summary(out, args.season)
    _write_output(out, args.out)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
