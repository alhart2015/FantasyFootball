"""Printable auction cheat sheet — the `overbid_noramp` strategy as a sheet you can read at a table.

`overbid_noramp` (OverbidValueBid with `use_urgency=False`) won the bake-off against an
over-bidding room, and it is two rules: pay up to `stud_multiple` x a player's auction dollar value
for the top `3 x n_teams` by VORP, and straight value for everyone else. Never more, at any point
in the draft — the no-ramp part is what makes it executable by a human, and it also measured
STRONGER than the ramped variant (+0.0118 reg-win%, +0.0124 champ%, both CIs excluding zero).

Because there is no state to track, the whole strategy is a price list: a live auction needs no
software, only this sheet and the discipline to obey it. The MAX BID column already has the stud
multiple applied, so the printed number IS the number to stop at.

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

from projections.draft.assistant.auction.bid_strategy import _vorp_threshold
from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, VorpTableSchema

_TXT_POSITIONS: tuple[Position, ...] = (Position.RB, Position.WR, Position.TE, Position.QB)
# Mirrors OverbidValueBid's defaults: k=1.3 on the top 3*n_teams players by VORP.
_STUD_MULTIPLE = 1.3
_STUDS_PER_TEAM = 3


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Printable auction cheat sheet (the `overbid_noramp` plan)."
    )
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
        "--stud-multiple",
        type=float,
        default=_STUD_MULTIPLE,
        help="Multiple of value to pay for a stud (OverbidValueBid.k; default 1.3).",
    )
    p.add_argument(
        "--per-position",
        type=int,
        default=30,
        help="Rows per position in the .txt sheet (default 30).",
    )
    return p.parse_args(argv)


def build_sheet(
    pool: pd.DataFrame, config: LeagueConfig, *, stud_multiple: float = _STUD_MULTIPLE
) -> pd.DataFrame:
    """Join auction dollars onto names/ADP/ESPN price, sorted by max bid descending.

    `max_bid` is the `overbid_noramp` bid: `stud_multiple x value` for a stud, plain value below.
    The stud cut reuses `_vorp_threshold` on the FULL pool, exactly as OverbidValueBid computes it
    mid-draft, so the sheet and the simulated strategy cannot drift apart.
    """
    values = generate_auction_values(pool, config)
    cols = ["gsis_id", "auction_dollars", "in_pool", "pool_rank"]
    sheet = pool.merge(values[cols], on="gsis_id", how="left")
    sheet = sheet[sheet["in_pool"]].copy()
    threshold = _vorp_threshold(pool, _STUDS_PER_TEAM * config.n_teams)
    sheet["is_stud"] = sheet["vorp"] >= threshold
    sheet["value"] = sheet["auction_dollars"].astype(int)
    sheet["max_bid"] = (
        (sheet["value"] * stud_multiple).round().where(sheet["is_stud"], sheet["value"])
    ).astype(int)
    sheet["espn_price"] = sheet["espn_auction_dollars"]
    # value_gap > 0 means ESPN's market is asking MORE than the player is worth to us — the lots to
    # let go. < 0 flags where our board is willing to pay above the room's anchor.
    sheet["value_gap"] = sheet["espn_price"] - sheet["max_bid"]
    sheet["proj_pts"] = sheet["season_mean_fpts"].round(0).astype(int)
    sheet = sheet.rename(columns={"full_name": "player", "consensus_adp": "adp"})
    out = sheet[
        [
            "player",
            "position",
            "max_bid",
            "is_stud",
            "value",
            "espn_price",
            "value_gap",
            "proj_pts",
            "adp",
            "gsis_id",
        ]
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
        "MAX BID already includes the stud premium — do not add anything on top of it.",
        "A '*' marks a stud (top 36 by value); its MAX BID is 1.3x what the player is worth.",
        "",
        f"Total money in the room: ${config.total_budget}.  "
        f"Your even pace: ${config.budget / config.roster_size:.0f} per roster spot.",
        "Expect most of the budget to go on ~3 studs and most of the roster to cost $1.",
        "That is the plan, not a mistake — do not hold money back for later.",
        "",
    ]
    for pos in _TXT_POSITIONS:
        block = sheet[(sheet["position"] == pos.value) & (sheet["max_bid"] >= min_dollars)]
        if block.empty:
            continue
        lines.append(f"--- {pos.value} " + "-" * 58)
        lines.append(f"{'MAX BID':>8}   {'player':<25}{'worth':>6}{'ESPN':>6}{'proj':>7}{'adp':>7}")
        for _, r in block.head(top).iterrows():
            adp = "—" if pd.isna(r["adp"]) else f"{r['adp']:.0f}"
            star = "*" if bool(r["is_stud"]) else " "
            lines.append(
                f"{_fmt_money(r['max_bid']):>8}{star}  {str(r['player'])[:24]:<25}"
                f"{_fmt_money(r['value']):>6}{_fmt_money(r['espn_price']):>6}"
                f"{r['proj_pts']:>7}{adp:>7}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = LeagueConfig.model_validate_json(args.league_config.read_text())
    pool = pd.read_parquet(args.vorp_table)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = VorpTableSchema.validate(pool)
    sheet = build_sheet(pool, config, stud_multiple=args.stud_multiple)
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
