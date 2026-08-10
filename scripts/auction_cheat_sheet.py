"""Printable auction cheat sheet — the `static` bid strategy as a sheet you can read at a table.

`static` (StaticDollarBid, the bake-off winner for an over-bidding room) is one rule: bid up to a
player's auction dollar value and not a cent more. That makes the whole strategy expressible as a
price list, so a live auction needs no software — only this sheet and the discipline to obey it.

`generate_auction_values.py` already computes the dollars, but emits `gsis_id` with no player name,
which is unusable in a draft room. This joins names, ESPN's market price, and ADP, then writes an
overall board plus per-position boards.

Reads the same LeagueConfig + VORP table the bake-off ran on, so the sheet and the simulation agree.

Run:
    python scripts/auction_cheat_sheet.py --league-config configs/my.league.json \
        --vorp-table data/vorp_2026/my.parquet --out reports/cheat_sheet.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, VorpTableSchema

_TXT_POSITIONS: tuple[Position, ...] = (Position.RB, Position.WR, Position.TE, Position.QB)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Printable auction cheat sheet (the `static` plan).")
    p.add_argument("--league-config", type=Path, required=True)
    p.add_argument("--vorp-table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output .csv (a .txt is written too).")
    p.add_argument(
        "--min-dollars",
        type=int,
        default=2,
        help="Omit players whose max bid is below this from the printed sheet (default 2 — "
        "everyone at $1 is interchangeable filler and would swamp the page).",
    )
    p.add_argument(
        "--per-position",
        type=int,
        default=30,
        help="Rows per position in the .txt sheet (default 30).",
    )
    return p.parse_args(argv)


def build_sheet(pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    """Join auction dollars onto names/ADP/ESPN price, sorted by max bid descending."""
    values = generate_auction_values(pool, config)
    cols = ["gsis_id", "auction_dollars", "in_pool", "pool_rank"]
    sheet = pool.merge(values[cols], on="gsis_id", how="left")
    sheet = sheet[sheet["in_pool"]].copy()
    sheet["max_bid"] = sheet["auction_dollars"].astype(int)
    sheet["espn_price"] = sheet["espn_auction_dollars"]
    # value_gap > 0 means ESPN's market is asking MORE than the player is worth to us — the lots to
    # let go. < 0 flags where our board is willing to pay above the room's anchor.
    sheet["value_gap"] = sheet["espn_price"] - sheet["max_bid"]
    sheet["proj_pts"] = sheet["season_mean_fpts"].round(0).astype(int)
    sheet = sheet.rename(columns={"full_name": "player", "consensus_adp": "adp"})
    out = sheet[
        ["player", "position", "max_bid", "espn_price", "value_gap", "proj_pts", "adp", "gsis_id"]
    ]
    return out.sort_values(["max_bid", "proj_pts"], ascending=False).reset_index(drop=True)


def _fmt_money(v: float | int | None) -> str:
    """Dollars, or an em dash for the players ESPN never priced (NA is expected, not an error)."""
    return "—" if v is None or pd.isna(v) else f"${int(v)}"


def render_text(sheet: pd.DataFrame, config: LeagueConfig, *, min_dollars: int, top: int) -> str:
    starters = sum(c for s, c in config.roster_slots.items() if s.value not in ("BENCH", "IR"))
    lines = [
        f"AUCTION CHEAT SHEET — {config.name}",
        f"{config.n_teams} teams · ${config.budget} budget · {config.roster_size} roster spots "
        f"({starters} starters)",
        "",
        "THE RULE: bid up to MAX BID. Never one dollar more. Walk away and take the next name.",
        "",
        f"Total money in the room: ${config.total_budget}.  "
        f"Your even pace: ${config.budget / config.roster_size:.0f} per roster spot.",
        "",
    ]
    for pos in _TXT_POSITIONS:
        block = sheet[(sheet["position"] == pos.value) & (sheet["max_bid"] >= min_dollars)]
        if block.empty:
            continue
        lines.append(f"--- {pos.value} " + "-" * 58)
        lines.append(f"{'MAX BID':>8}  {'player':<26}{'ESPN':>6}{'proj':>7}{'adp':>7}")
        for _, r in block.head(top).iterrows():
            adp = "—" if pd.isna(r["adp"]) else f"{r['adp']:.0f}"
            lines.append(
                f"{_fmt_money(r['max_bid']):>8}  {str(r['player'])[:25]:<26}"
                f"{_fmt_money(r['espn_price']):>6}{r['proj_pts']:>7}{adp:>7}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    pool = pd.read_parquet(args.vorp_table)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = VorpTableSchema.validate(pool)
    sheet = build_sheet(pool, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, index=False)
    txt_path = args.out.with_suffix(".txt")
    txt_path.write_text(
        render_text(sheet, config, min_dollars=args.min_dollars, top=args.per_position),
        encoding="utf-8",
    )
    priced = int((sheet["max_bid"] >= args.min_dollars).sum())
    print(f"{len(sheet)} players in pool ({priced} at >= ${args.min_dollars})")
    print(f"  full board  -> {args.out}")
    print(f"  print sheet -> {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
