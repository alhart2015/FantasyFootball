"""Auction before/after for adding the D/ST roster slot — real config, real projections.

Spec §7.4 requires a human to look at this diff before the repricing lands: adding a DST slot
grows `total_pool_size`, which reprices every skill player on the board. A passing test suite
cannot surface that. The 2026-09-06 Critts run is recorded in the spec at §5.3.

Re-run whenever the projections or the league config change.

Usage:
    python scripts/dst_auction_diff.py
    python scripts/dst_auction_diff.py --league-payload data/leagues/other/espn_raw.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.draft.vorp import generate_vorp_table
from projections.ingest.espn_league import build_league_config
from projections.ingest.external_projections import (
    DST_FETCH_LIMIT,
    fetch_espn,
    refresh_dst_projections,
)
from projections.schemas import DST_TEAM_BY_GSIS, RosterSlot
from projections.scoring.dst import score_dst
from projections.store import read_partition

#: Relative to the working directory, not the script: `data/` is gitignored, so a worktree may
#: not carry it. Run from the checkout that holds the league pull.
DEFAULT_PAYLOAD = Path("data/leagues/critts_2025_2026/espn_raw.json")
DEFAULT_SKILL_POOL = Path("data/consensus_vorp_2026.parquet")

GEN_AT = pd.Timestamp("2026-09-06", tz="UTC")


def _season_projections(
    skill: pd.DataFrame, ruleset: Any, season: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    names = dict(zip(skill["gsis_id"], skill["full_name"], strict=True))
    proj = pd.DataFrame(
        {
            "gsis_id": skill["gsis_id"],
            "season": season,
            "position": skill["position"],
            "ruleset": ruleset.name,
            "n_weeks": 17,
            "season_mean": skill["season_mean_fpts"].astype("float64"),
            "season_p10": 0.0,
            "season_p50": 0.0,
            "season_p90": 0.0,
            "model_id": "consensus",
            "generated_at": GEN_AT,
        }
    )
    return proj, names


def _board(projections: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    vorp = generate_vorp_table(projections, config)
    return generate_auction_values(vorp, config).set_index("gsis_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league-payload", type=Path, default=DEFAULT_PAYLOAD)
    ap.add_argument("--skill-pool", type=Path, default=DEFAULT_SKILL_POOL)
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()

    cfg_with = build_league_config(json.loads(args.league_payload.read_text(encoding="utf-8")))
    ruleset = cfg_with.ruleset
    cfg_without = cfg_with.model_copy(
        update={
            "roster_slots": {
                slot: n for slot, n in cfg_with.roster_slots.items() if slot != RosterSlot.DST
            }
        }
    )

    skill = pd.read_parquet(args.skill_pool)
    proj, names = _season_projections(skill, ruleset, args.season)

    # Defenses, scored through the real chain rather than a shortcut, so the diff reflects what
    # the tools will actually see.
    tmp = Path(tempfile.mkdtemp())
    refresh_dst_projections(
        tmp,
        season=args.season,
        asof=date(2026, 9, 6),
        payload=fetch_espn(args.season, limit=DST_FETCH_LIMIT),
    )
    dst_raw = read_partition(
        tmp / "raw", "dst_projections", season=args.season, asof=date(2026, 9, 6)
    )
    dst_rows = []
    for gsis, group in dst_raw.groupby("gsis_id"):
        points = score_dst(dict(zip(group["stat_id"], group["value"], strict=True)), ruleset)
        names[gsis] = f"{DST_TEAM_BY_GSIS[gsis].value} D/ST"
        dst_rows.append(
            {
                "gsis_id": gsis,
                "season": args.season,
                "position": "DST",
                "ruleset": ruleset.name,
                "n_weeks": 17,
                "season_mean": points,
                "season_p10": points,
                "season_p50": points,
                "season_p90": points,
                "model_id": "espn_dst_passthrough",
                "generated_at": GEN_AT,
            }
        )
    proj_with = pd.concat([proj, pd.DataFrame(dst_rows)], ignore_index=True)

    before = _board(proj, cfg_without)
    after = _board(proj_with, cfg_with)

    print("=" * 72)
    print(
        f"POOL:  {cfg_without.total_pool_size}  ->  {cfg_with.total_pool_size}"
        f"   (roster {cfg_without.roster_size} -> {cfg_with.roster_size}, "
        f"budget ${cfg_with.total_budget})"
    )
    print("=" * 72)

    common = before.index.intersection(after.index)
    cmp = pd.DataFrame(
        {
            "name": [names.get(g, g) for g in common],
            "pos": before.loc[common, "position"],
            "before": before.loc[common, "auction_dollars"].astype(float),
            "after": after.loc[common, "auction_dollars"].astype(float),
        }
    )
    cmp["delta"] = cmp["after"] - cmp["before"]

    print("\nTOP 15 SKILL PLAYERS")
    print(f"{'name':<24}{'pos':>4}{'before':>8}{'after':>7}{'delta':>7}")
    for _, row in cmp.sort_values("before", ascending=False).head(15).iterrows():
        print(
            f"{str(row['name'])[:23]:<24}{row['pos']:>4}{row['before']:>8.0f}"
            f"{row['after']:>7.0f}{row['delta']:>+7.0f}"
        )

    print("\nBY POSITION (mean $ change among players priced in both)")
    print(f"{'pos':<5}{'n':>5}{'mean before':>13}{'mean after':>12}{'mean delta':>12}")
    for pos, group in cmp[cmp["before"] > 0].groupby("pos"):
        print(
            f"{pos:<5}{len(group):>5}{group['before'].mean():>13.1f}"
            f"{group['after'].mean():>12.1f}{group['delta'].mean():>+12.1f}"
        )

    dst = after[after["position"] == "DST"].sort_values("auction_dollars", ascending=False)
    print(f"\nDEFENSES NOW PRICED ({len(dst)}), total ${int(dst['auction_dollars'].sum())}")
    print(f"{'name':<14}{'proj':>8}{'vorp':>8}{'$':>5}")
    rows = list(dst.iterrows())
    for gsis, row in rows[:6]:
        print(
            f"{names.get(gsis, gsis):<14}{row['season_mean_fpts']:>8.1f}"
            f"{row['vorp']:>8.1f}{int(row['auction_dollars']):>5}"
        )
    print("  ...")
    for gsis, row in rows[-3:]:
        print(
            f"{names.get(gsis, gsis):<14}{row['season_mean_fpts']:>8.1f}"
            f"{row['vorp']:>8.1f}{int(row['auction_dollars']):>5}"
        )

    print(
        f"\nTotal $ on the board: {int(before['auction_dollars'].sum())} -> "
        f"{int(after['auction_dollars'].sum())}  (budget ${cfg_with.total_budget})"
    )


if __name__ == "__main__":
    main()
