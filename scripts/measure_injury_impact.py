"""Measure what an injury designation actually costs, in points.

Produces the two tables in §4.5 of
`docs/superpowers/specs/2026-08-26-waiver-recommender-design.md`. Kept as a script rather than
as constants alone so the numbers can be re-measured when a season is added, and so the
methodology is inspectable rather than folklore.

    python scripts/measure_injury_impact.py
    python scripts/measure_injury_impact.py --seasons 2022 2023 2024 2025 --min-projection 8

**The two mistakes this script exists to avoid**, both found while first writing it:

1. **Averaging over an unconditioned population.** "54% of Questionable players play" is true
   and useless -- it is dominated by players who were never going to play, injury or not. Split
   by projected volume it runs from 11% (projected under 2 points) to 100% (projected 15+).
   Any aggregate over injury designations that does not condition on projected volume is
   measuring roster churn, not injury. Hence `--by-tier`, which is on by default.

2. **Measuring a discount that has already been applied.** If ESPN downgraded a Questionable
   player's weekly projection, `actual / projected` would look healthy for the wrong reason and
   applying our own multiplier on top would double-count. The `confound` section compares a
   player's projection in his listed weeks against his own median healthy-week projection.
   Measured at 100.4% for Questionable -- no discount -- and near zero for Out, which ESPN does
   zero out.

Inputs, all already on disk or free to pull:
  - `nflreadpy.load_injuries`                      -- the weekly injury report
  - `data/processed/espn_weekly_projections`       -- what he was projected for that week
  - `data/raw/weekly_stats`                        -- what he actually scored
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import nflreadpy
import pandas as pd

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, score
from projections.store import read_partition

#: Skill positions. The pool carries no kickers or defenses, so there is nothing to rank there.
POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})

#: Game-status designations. `report_status` also carries NaN (listed on the practice report
#: but not given a game status) and a rare "Note", neither of which is a designation.
DESIGNATIONS: tuple[str, ...] = ("Questionable", "Doubtful", "Out")

#: Projection tiers for the conditioning split. The bottom tier is the one that makes an
#: unconditioned average meaningless.
TIERS: tuple[tuple[str, float, float], ...] = (
    ("<2", -1.0, 2.0),
    ("2-5", 2.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-15", 10.0, 15.0),
    ("15+", 15.0, 1e9),
)


@dataclass(frozen=True)
class Inputs:
    seasons: tuple[int, ...]
    data_root: Path
    processed_root: Path
    ruleset: Ruleset
    min_projection: float


def weekly_actuals(inputs: Inputs) -> pd.DataFrame:
    """`weekly_stats` scored under the league ruleset, one row per player-week.

    Scored here rather than read from anywhere, because "points" only means something once a
    ruleset says so -- and the ruleset that matters is the league's.
    """
    frames = []
    for season in inputs.seasons:
        frame = read_partition(inputs.data_root, "weekly_stats", season=season)
        frame = frame[frame["position"].isin(POSITIONS)]
        frame = frame.assign(
            actual_points=[
                score(
                    StatLine(
                        passing_yards=float(row["passing_yards"]),
                        passing_tds=int(row["passing_tds"]),
                        interceptions=int(row["interceptions"]),
                        rushing_yards=float(row["rushing_yards"]),
                        rushing_tds=int(row["rushing_tds"]),
                        receptions=int(row["receptions"]),
                        receiving_yards=float(row["receiving_yards"]),
                        receiving_tds=int(row["receiving_tds"]),
                        fumbles_lost=int(row["fumbles_lost"]),
                    ),
                    inputs.ruleset,
                )
                for _, row in frame.iterrows()
            ]
        )
        frames.append(frame[["gsis_id", "season", "week", "position", "actual_points"]])
    return pd.concat(frames, ignore_index=True)


def weekly_projections(inputs: Inputs) -> pd.DataFrame:
    frames = [
        read_partition(inputs.processed_root, "espn_weekly_projections", season=season)[
            ["gsis_id", "season", "week", "position", "projected_points"]
        ]
        for season in inputs.seasons
    ]
    return pd.concat(frames, ignore_index=True)


def designations(inputs: Inputs) -> pd.DataFrame:
    """One row per (player, season, week) that carried a game-status designation."""
    # `nflreadpy`, not `nfl_data_py`: the repo standardised on the nflverse successor
    # (see `ingest/weekly_stats.py`), and it returns polars.
    frame = nflreadpy.load_injuries(seasons=list(inputs.seasons)).to_pandas()
    frame = frame[frame["position"].isin(POSITIONS)]
    frame = frame.dropna(subset=["gsis_id", "season", "week", "report_status"])
    frame = frame.assign(status=frame["report_status"].astype(str).str.strip().str.title())
    frame = frame[frame["status"].isin(DESIGNATIONS)]
    return frame[["gsis_id", "season", "week", "status"]].drop_duplicates(
        ["gsis_id", "season", "week"]
    )


def joined(inputs: Inputs) -> pd.DataFrame:
    """Every projected player-week, labelled with its designation (Healthy when unlisted).

    **Projections are the spine, not actuals.** A player with no stat line scored nothing, and
    that is a real zero rather than a missing row -- joining the other way would silently drop
    exactly the players an injury analysis is about.
    """
    frame = weekly_projections(inputs).merge(
        weekly_actuals(inputs).drop(columns="position"),
        on=["gsis_id", "season", "week"],
        how="left",
    )
    # "Played" is HAS A STAT LINE, not "scored points". A running back who dressed and was
    # stuffed for no gain played; so did a receiver who was targeted twice and caught nothing.
    # Defining it as `> 0` counts those as absences and understates every play rate -- the
    # weekly_stats partitions carry all-zero rows precisely so this distinction survives.
    frame["played"] = frame["actual_points"].notna()
    frame["actual_points"] = frame["actual_points"].fillna(0.0)
    frame = frame.merge(designations(inputs), on=["gsis_id", "season", "week"], how="left")
    frame["status"] = frame["status"].fillna("Healthy")
    return frame


def _rate_table(frame: pd.DataFrame, by: str, value: str) -> pd.DataFrame:
    out = frame.groupby(by, observed=True)[value].agg(n="size", mean="mean")
    out["pct"] = (out["mean"] * 100).round(1)
    return out.drop(columns="mean")


def report(inputs: Inputs) -> None:
    frame = joined(inputs)
    print(f"seasons {inputs.seasons[0]}-{inputs.seasons[-1]} · positions {sorted(POSITIONS)}")
    print(f"projected player-weeks: {len(frame)}")

    print("\n=== 1. play rate by designation, UNCONDITIONED (the misleading version) ===")
    print(_rate_table(frame[frame["status"] != "Healthy"], "status", "played").to_string())

    print("\n=== 2. play rate by projected volume — Questionable only ===")
    print("This is the split that makes the number above meaningless.")
    q = frame[frame["status"] == "Questionable"].copy()
    q["tier"] = pd.cut(
        q["projected_points"],
        [low for _, low, _ in TIERS] + [TIERS[-1][2]],
        labels=[name for name, _, _ in TIERS],
    )
    print(_rate_table(q, "tier", "played").to_string())

    startable = frame[frame["projected_points"] >= inputs.min_projection].copy()
    startable["delivered"] = startable["actual_points"] / startable["projected_points"]
    print(f"\n=== 3. share of projection delivered (projected >= {inputs.min_projection:g}) ===")
    table = startable.groupby("status")["delivered"].agg(n="size", mean="mean", median="median")
    table["mean_pct"] = (table["mean"] * 100).round(1)
    table["median_pct"] = (table["median"] * 100).round(1)
    print(table.drop(columns=["mean", "median"]).to_string())

    print("\n=== 4. CONFOUND CHECK: is ESPN already discounting the projection? ===")
    print("A player's projection in his listed weeks vs his own median healthy-week projection.")
    print("Near 100% means no discount, and our multiplier is ours to apply.")
    healthy = (
        startable[startable["status"] == "Healthy"]
        .groupby("gsis_id")["projected_points"]
        .median()
        .rename("healthy_projection")
    )
    listed = startable[startable["status"] != "Healthy"].merge(healthy, on="gsis_id", how="inner")
    listed["ratio"] = listed["projected_points"] / listed["healthy_projection"]
    print(_rate_table(listed, "status", "ratio").to_string())
    print(
        "\nNote the counts: ESPN gives almost no `Out` player a real projection, which IS the\n"
        "discount. Anything applying an `Out` multiplier on top of this feed double-counts."
    )

    print("\n=== 5. the constants this produces ===")
    baseline = float(table.loc["Healthy", "mean_pct"])
    print(f"weekly multiplier, relative to a healthy baseline of {baseline:.1f}%:")
    for status in DESIGNATIONS:
        if status in table.index:
            got = float(table.loc[status, "mean_pct"])
            n = int(table.loc[status, "n"])
            flag = "  (n too small to trust)" if n < 30 else ""
            print(f"  {status:<14} {got / baseline:.3f}   n={n}{flag}")
    print("\nfor the statuses with too few projected weeks, fall back to the play rate in (1):")
    rates = _rate_table(frame[frame["status"] != "Healthy"], "status", "played")
    for status in DESIGNATIONS:
        if status in rates.index:
            print(
                f"  {status:<14} {float(rates.loc[status, 'pct']) / 100:.3f}   "
                f"n={int(rates.loc[status, 'n'])}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--min-projection",
        type=float,
        default=5.0,
        help=(
            "Only measure the delivered share for player-weeks projected at least this high. "
            "A 0-projected deep-bench player makes the ratio explode and says nothing about "
            "injuries."
        ),
    )
    parser.add_argument(
        "--scoring",
        default="espn_half",
        choices=["espn_half", "espn_ppr", "standard"],
        help="Ruleset the actuals are scored under. Points only mean something under one.",
    )
    args = parser.parse_args(argv)

    rulesets = {
        "espn_half": Ruleset.espn_half(),
        "espn_ppr": Ruleset.espn_ppr(),
        "standard": Ruleset.standard(),
    }
    report(
        Inputs(
            seasons=tuple(sorted(args.seasons)),
            data_root=args.data_root,
            processed_root=args.processed_root,
            ruleset=rulesets[args.scoring],
            min_projection=args.min_projection,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
