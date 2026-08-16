"""Historical checks: is the market calibrated, and what score should I expect?

Two separate questions, both answered from `data/raw/schedules`.

**Calibration.** The whole pipeline trusts devigged moneylines as truth. That is
only reasonable if games the market priced at 63% actually won about 63% of the
time. This bins historical probabilities and compares them to what happened.

**Baseline.** Runs the real optimizer over past weeks using closing lines as
*both* the organizer's sheet and the consensus. That is a deliberate fiction: it
models away the stale-sheet edge entirely, because nflverse stores only closing
lines and historical opening-vs-closing movement is not available here. So the
result is a **floor, not a forecast** — it says what the underdog constraint
costs and what a typical week looks like with no edge at all.
"""

from __future__ import annotations

import pandas as pd

from projections.pickem._validate import require_schedule_columns
from projections.pickem.grade import grade_picks, record
from projections.pickem.optimize import DEFAULT_MIN_DOGS, choose_picks, expected_correct
from projections.pickem.probability import add_win_probs
from projections.pickem.slate import build_slate

_BACKTEST_COLUMNS = (
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "spread_line",
    "home_moneyline",
    "away_moneyline",
    "home_score",
    "away_score",
)


def playable_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """Completed regular-season games that carry a full market price.

    Playoff games are excluded: the pool runs on the regular season, and their
    scheduling quirks are not what we are measuring.
    """
    require_schedule_columns(schedules, _BACKTEST_COLUMNS, needed_for="the pick'em backtest")
    df = schedules
    if "game_type" in df.columns:
        df = df[df["game_type"] == "REG"]
    return df[
        df["spread_line"].notna()
        & df["home_moneyline"].notna()
        & df["away_moneyline"].notna()
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].reset_index(drop=True)


def calibration_table(schedules: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Bin market probabilities and compare each bin to the realized win rate.

    One row per game, taken from the home team's perspective — using both sides
    would double-count, since the two probabilities are one number and its
    complement. Ties count as a loss for the home team, matching how the pool
    grades them.
    """
    games = playable_games(schedules)
    if games.empty:
        raise ValueError("no completed, fully-priced regular-season games to calibrate on")
    priced = add_win_probs(games)

    home_won = (priced["home_score"] > priced["away_score"]).astype(float)
    edges = [i / n_bins for i in range(n_bins + 1)]
    bucket = pd.cut(priced["home_win_prob"], bins=edges, include_lowest=True)

    grouped = pd.DataFrame(
        {
            "predicted": priced["home_win_prob"],
            "actual": home_won,
            "bucket": bucket,
        }
    ).groupby("bucket", observed=True)

    out = grouped.agg(
        n_games=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        actual_rate=("actual", "mean"),
    ).reset_index()
    out["error"] = out["actual_rate"] - out["mean_predicted"]
    out["bucket"] = out["bucket"].astype(str)
    return out


def _sheet_from_closing_lines(week_games: pd.DataFrame) -> pd.DataFrame:
    """Treat the closing line as if it were the organizer's sheet.

    `spread_line` is positive when home is favored; the sheet uses the standard
    convention, so it is negated here exactly as `slate` does for the market.
    """
    return pd.DataFrame(
        {
            "season": week_games["season"],
            "week": week_games["week"],
            "home_team": week_games["home_team"],
            "away_team": week_games["away_team"],
            "home_spread": -week_games["spread_line"].astype(float),
        }
    ).reset_index(drop=True)


def baseline_week_scores(
    schedules: pd.DataFrame, *, min_dogs: int = DEFAULT_MIN_DOGS
) -> pd.DataFrame:
    """Per-week results of running the optimizer with no staleness edge.

    Returns one row per (season, week) with the constrained and unconstrained
    outcomes side by side, so the cost of the underdog rule is directly
    readable rather than inferred.
    """
    games = playable_games(schedules)
    rows: list[dict[str, float | int]] = []

    for (season, week), week_games in games.groupby(["season", "week"], sort=True):
        sheet = _sheet_from_closing_lines(week_games)
        slate = build_slate(sheet, week_games)
        try:
            constrained = choose_picks(slate, min_dogs=min_dogs)
        except ValueError:
            # Too few games with a real underdog to satisfy the rule. Cannot
            # happen on a full NFL slate; skipping beats fabricating a score.
            continue
        unconstrained = choose_picks(slate, min_dogs=0)

        correct, played = record(grade_picks(constrained, week_games))
        free_correct, _ = record(grade_picks(unconstrained, week_games))
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "n_games": played,
                "expected_correct": expected_correct(constrained),
                "actual_correct": correct,
                "expected_unconstrained": expected_correct(unconstrained),
                "actual_unconstrained": free_correct,
                "forced_dogs": int(constrained["forced"].sum()),
                "free_dogs": int((constrained["is_dog_pick"] & ~constrained["forced"]).sum()),
            }
        )

    return pd.DataFrame(rows)


def summarize_baseline(weeks: pd.DataFrame) -> dict[str, float]:
    """Season-agnostic headline numbers from `baseline_week_scores`."""
    if weeks.empty:
        raise ValueError("no weeks to summarize")
    total_games = float(weeks["n_games"].sum())
    return {
        "weeks": float(len(weeks)),
        "games": total_games,
        "actual_per_week": float(weeks["actual_correct"].mean()),
        "expected_per_week": float(weeks["expected_correct"].mean()),
        "games_per_week": float(weeks["n_games"].mean()),
        "hit_rate": float(weeks["actual_correct"].sum() / total_games),
        "constraint_cost_per_week": float(
            (weeks["expected_unconstrained"] - weeks["expected_correct"]).mean()
        ),
        "actual_constraint_cost_per_week": float(
            (weeks["actual_unconstrained"] - weeks["actual_correct"]).mean()
        ),
        "free_dogs_per_week": float(weeks["free_dogs"].mean()),
    }
