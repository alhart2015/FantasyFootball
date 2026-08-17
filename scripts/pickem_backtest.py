"""Pick'em history: is the market calibrated, and what score should I expect?

Two questions, both answered from already-ingested schedules.

1. Calibration. The picker trusts devigged moneylines as truth, which is only
   reasonable if games priced at 63% actually won about 63% of the time.

2. Baseline. Runs the real optimizer over past weeks using closing lines as
   BOTH the organizer's sheet and the consensus - i.e. with the stale-sheet
   edge deliberately removed, since historical line movement is not in this
   data source. The result is a floor, not a forecast: it says what the
   three-underdog rule costs and what a typical week looks like with no edge.

Run:
    python scripts/pickem_backtest.py --seasons 2015-2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.pickem.backtest import (
    baseline_week_scores,
    calibration_table,
    summarize_baseline,
)
from projections.pickem.optimize import DEFAULT_MIN_DOGS
from projections.store import read_partition

# Columns a partition must carry to contribute any games to the backtest.
_SCORE_COLUMNS = ("game_type", "home_score", "away_score")


def _parse_seasons(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in spec.split(",")]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick'em calibration and baseline backtest")
    parser.add_argument("--seasons", default="2015-2025", help="e.g. 2015-2025 or 2023,2024,2025")
    parser.add_argument("--min-dogs", type=int, default=DEFAULT_MIN_DOGS)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args(argv)


def render_calibration(table: pd.DataFrame) -> str:
    lines = [
        "MARKET CALIBRATION - devigged moneyline vs. what actually happened",
        "(one row per game, from the home team's perspective)",
        "",
        f"{'PROBABILITY BIN':<18}{'GAMES':>7}{'PREDICTED':>11}{'ACTUAL':>9}{'ERROR':>8}",
        "-" * 55,
    ]
    for row in table.itertuples():
        lines.append(
            f"{row.bucket:<18}{row.n_games:>7}{row.mean_predicted:>11.1%}"
            f"{row.actual_rate:>9.1%}{row.error:>+8.1%}"
        )
    weighted = float((table["error"].abs() * table["n_games"]).sum() / table["n_games"].sum())
    lines += ["-" * 55, f"Mean absolute error, game-weighted: {weighted:.1%}"]
    return "\n".join(lines)


def render_baseline(weeks: pd.DataFrame, *, min_dogs: int) -> str:
    s = summarize_baseline(weeks)
    lines = [
        f"BASELINE - optimizer over {int(s['weeks'])} weeks "
        f"({int(s['games'])} games), no staleness edge",
        "",
        f"  Games per week           {s['games_per_week']:.1f}",
        f"  Correct per week         {s['actual_per_week']:.2f}  "
        f"(model expected {s['expected_per_week']:.2f})",
        f"  Hit rate                 {s['hit_rate']:.1%}",
        "",
        f"  Cost of the {min_dogs}-dog rule  "
        f"{s['constraint_cost_per_week']:.2f} picks/week expected, "
        f"{s['actual_constraint_cost_per_week']:.2f} actual",
        f"  Free dogs per week       {s['free_dogs_per_week']:.2f}",
        "",
        "Expected vs. actual agreeing is itself a calibration check: it means the",
        "probabilities the picker maximizes line up with real outcomes.",
        "",
        "This is a FLOOR. It assumes the organizer's sheet matches the market",
        "exactly. Your real edge is that it will not.",
    ]
    return "\n".join(lines)


def render_by_season(weeks: pd.DataFrame) -> str:
    by_season = weeks.groupby("season").agg(
        weeks=("week", "size"),
        games=("n_games", "sum"),
        correct=("actual_correct", "sum"),
        expected=("expected_correct", "sum"),
    )
    by_season["per_week"] = by_season["correct"] / by_season["weeks"]
    by_season["hit_rate"] = by_season["correct"] / by_season["games"]

    lines = [
        "BY SEASON",
        "",
        f"{'SEASON':<8}{'WEEKS':>7}{'GAMES':>7}{'CORRECT':>9}{'EXPECTED':>10}"
        f"{'PER WK':>8}{'HIT%':>7}",
        "-" * 56,
    ]
    for season, row in by_season.iterrows():
        lines.append(
            f"{int(season):<8}{int(row['weeks']):>7}{int(row['games']):>7}"
            f"{int(row['correct']):>9}{row['expected']:>10.1f}"
            f"{row['per_week']:>8.2f}{row['hit_rate']:>7.1%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    seasons = _parse_seasons(args.seasons)

    frames = []
    stale: list[tuple[int, list[str]]] = []
    for season in seasons:
        try:
            frame = read_partition(args.data_root / "raw", "schedules", season=season)
        except FileNotFoundError:
            print(f"  (no schedules partition for {season}, skipping)")
            continue
        # Checked PER SEASON, before the concat. `pd.concat` unions columns, so a
        # single re-ingested season is enough to make `home_score` exist on the
        # combined frame - `playable_games`' guard then passes and the notna()
        # filter silently drops every row of the older partitions instead.
        absent = [c for c in _SCORE_COLUMNS if c not in frame.columns]
        if absent:
            stale.append((season, absent))
        frames.append(frame)

    if not frames:
        raise SystemExit("no schedules found; run refresh_schedules first")

    if stale:
        listed = ", ".join(f"{s} (missing {', '.join(cols)})" for s, cols in stale)
        print(
            f"WARNING: {len(stale)} season partition(s) predate the score columns and will "
            f"contribute NO games: {listed}.\n"
            f"         Re-ingest them or the backtest silently covers a shorter span than "
            f"--seasons implies.\n"
        )

    schedules = pd.concat(frames, ignore_index=True)

    print(render_calibration(calibration_table(schedules, n_bins=args.bins)))
    print()
    weeks = baseline_week_scores(schedules, min_dogs=args.min_dogs)
    print(render_baseline(weeks, min_dogs=args.min_dogs))
    print()
    print(render_by_season(weeks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
