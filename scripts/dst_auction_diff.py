"""Auction before/after for adding the D/ST roster slot — real Critts config, real projections.

Spec §7.4: this diff is the thing the user reviews. A passing test suite cannot surface a
repricing of the whole board.
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from projections.draft.auction import generate_auction_values
from projections.draft.vorp import generate_vorp_table
from projections.ingest.espn_league import build_league_config
from projections.ingest.external_projections import fetch_espn, refresh_dst_projections
from projections.schemas import DST_TEAM_BY_GSIS, RosterSlot
from projections.scoring.dst import score_dst
from projections.store import read_partition

ROOT = Path(r"C:\Users\HartAlden\FantasyFootball")
GEN_AT = pd.Timestamp("2026-09-06", tz="UTC")

cfg_with = build_league_config(
    json.load(open(ROOT / "data/leagues/critts_2025_2026/espn_raw.json"))
)
ruleset = cfg_with.ruleset
cfg_without = cfg_with.model_copy(
    update={"roster_slots": {k: v for k, v in cfg_with.roster_slots.items() if k != RosterSlot.DST}}
)

# --- skill projections (the board as it stands today) ---
skill = pd.read_parquet(ROOT / "data/consensus_vorp_2026.parquet")
names = dict(zip(skill["gsis_id"], skill["full_name"], strict=True))
proj = pd.DataFrame(
    {
        "gsis_id": skill["gsis_id"],
        "season": 2026,
        "position": skill["position"],
        "ruleset": ruleset.name,
        "n_weeks": 17,
        "season_mean": skill["season_mean_fpts"].astype("float64"),
        "season_p10": 0.0,
        "season_p50": 0.0,
        "season_p90": 0.0,
        "model_id": "consensus_2026",
        "generated_at": GEN_AT,
    }
)

# --- defenses, scored through the real chain ---
tmp = Path(tempfile.mkdtemp())
refresh_dst_projections(
    tmp, season=2026, asof=date(2026, 9, 6), payload=fetch_espn(2026, limit=1500)
)
dst_raw = read_partition(tmp / "raw", "dst_projections", season=2026, asof=date(2026, 9, 6))
dst_rows = []
for gsis, g in dst_raw.groupby("gsis_id"):
    pts = score_dst(dict(zip(g["stat_id"], g["value"], strict=True)), ruleset)
    team = DST_TEAM_BY_GSIS[gsis]
    names[gsis] = f"{team.value} D/ST"
    dst_rows.append(
        {
            "gsis_id": gsis,
            "season": 2026,
            "position": "DST",
            "ruleset": ruleset.name,
            "n_weeks": 17,
            "season_mean": pts,
            "season_p10": 0.0,
            "season_p50": 0.0,
            "season_p90": 0.0,
            "model_id": "espn_dst_2026",
            "generated_at": GEN_AT,
        }
    )
proj_with = pd.concat([proj, pd.DataFrame(dst_rows)], ignore_index=True)


def board(projections: pd.DataFrame, cfg) -> pd.DataFrame:
    v = generate_vorp_table(projections, cfg)
    a = generate_auction_values(v, cfg)
    return a.set_index("gsis_id")


before = board(proj, cfg_without)
after = board(proj_with, cfg_with)

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
for _, r in cmp.sort_values("before", ascending=False).head(15).iterrows():
    print(
        f"{str(r['name'])[:23]:<24}{r['pos']:>4}{r['before']:>8.0f}{r['after']:>7.0f}"
        f"{r['delta']:>+7.0f}"
    )

print("\nBY POSITION (mean $ change among players priced in both)")
print(f"{'pos':<5}{'n':>5}{'mean before':>13}{'mean after':>12}{'mean delta':>12}")
for pos, g in cmp[cmp["before"] > 0].groupby("pos"):
    print(
        f"{pos:<5}{len(g):>5}{g['before'].mean():>13.1f}{g['after'].mean():>12.1f}"
        f"{g['delta'].mean():>+12.1f}"
    )

dst = after[after["position"] == "DST"].sort_values("auction_dollars", ascending=False)
print(f"\nDEFENSES NOW PRICED ({len(dst)}), total ${int(dst['auction_dollars'].sum())}")
print(f"{'name':<14}{'proj':>8}{'vorp':>8}{'$':>5}")
for g, r in list(dst.iterrows())[:6]:
    print(
        f"{names.get(g, g):<14}{r['season_mean_fpts']:>8.1f}{r['vorp']:>8.1f}"
        f"{int(r['auction_dollars']):>5}"
    )
print("  ...")
for g, r in list(dst.iterrows())[-3:]:
    print(
        f"{names.get(g, g):<14}{r['season_mean_fpts']:>8.1f}{r['vorp']:>8.1f}"
        f"{int(r['auction_dollars']):>5}"
    )

print(
    f"\nTotal $ on the board: {int(before['auction_dollars'].sum())} -> "
    f"{int(after['auction_dollars'].sum())}  (budget ${cfg_with.total_budget})"
)
