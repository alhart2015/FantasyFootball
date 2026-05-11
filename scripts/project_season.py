"""One-off: train BaselineModel per position on 2018..(season-1), project all
weeks of `season`, aggregate weekly mean predictions to a season total per
player, and print top-100 + top-10-per-position rankings.

Reuses the same pipeline as scripts/train_baseline.py + scripts/predict_2024.py
but parametrized on --season so it can run for any historical year.

Usage:
    python scripts/project_season.py --season 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from projections.models import POSITION_DISPATCH, production_model_for
from projections.models.base import Model
from projections.schemas import Position, ProjectionWeeklySchema, Ruleset
from projections.store import read_partition


def _load_draft_picks(raw_root: Path, max_season: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s in range(1980, max_season + 1):
        try:
            frames.append(read_partition(raw_root, "draft_picks", season=s))
        except FileNotFoundError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_features_for_season(
    raw_root: Path,
    position: Position,
    season: int,
    *,
    draft_picks: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Build features for every week of `season`, returning one frame per week.
    Mirrors scripts/train_baseline.py:_build_training_features but for one season."""
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_kwarg = {
        "passing": "ngs_passing",
        "rushing": "ngs_rushing",
        "receiving": "ngs_receiving",
    }[dispatch.ngs_stat_type]
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"

    ws = read_partition(raw_root, "weekly_stats", season=season)
    sc = read_partition(raw_root, "snap_counts", season=season)
    dc = read_partition(raw_root, "depth_charts", season=season)
    ngs = read_partition(raw_root, ngs_table, season=season)
    sch = read_partition(raw_root, "schedules", season=season)
    try:
        pbp = read_partition(raw_root, "pbp", season=season)
    except FileNotFoundError:
        pbp = pd.DataFrame()

    frames: list[pd.DataFrame] = []
    weeks = sorted(int(w) for w in dc["week"].unique())
    for week in weeks:
        kwargs: dict[str, Any] = {
            "weekly_stats": ws,
            "snap_counts": sc,
            "depth_charts": dc,
            "schedules": sch,
            "season": int(season),
            "as_of_week": int(week),
            "pbp": pbp,
            "draft_picks": draft_picks,
            ngs_kwarg: ngs,
        }
        f = builder(**kwargs)
        if not f.empty:
            frames.append(f)
    return frames


def _train_production(
    raw_root: Path,
    position: Position,
    *,
    train_seasons: range,
    draft_picks: pd.DataFrame,
) -> Model:
    """Train the production-default model class for the position (Plan 8 routing):
    QB -> lightgbm-nb, RB -> baseline, TE -> baseline, WR -> ensemble."""
    span = f"{train_seasons.start}-{train_seasons.stop - 1}"
    print(f"  [{position.value}] building training features {span}...", flush=True)
    feature_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    for season in train_seasons:
        truth_frames.append(read_partition(raw_root, "weekly_stats", season=season))
        feature_frames.extend(
            _build_features_for_season(raw_root, position, season, draft_picks=draft_picks)
        )
    features = pd.concat(feature_frames, ignore_index=True)
    weekly_stats = pd.concat(truth_frames, ignore_index=True)
    model_class = POSITION_DISPATCH[position].default_model_class
    print(
        f"  [{position.value}] feature rows: {len(features)}; "
        f"truth rows: {len(weekly_stats)}; fitting {model_class}...",
        flush=True,
    )
    model = production_model_for(position)
    model.fit(features=features, weekly_stats=weekly_stats)
    return model


def _project_season(
    raw_root: Path,
    position: Position,
    season: int,
    *,
    model: Model,
    draft_picks: pd.DataFrame,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Concatenate prior + current season for trailing windows, then predict each week."""
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_kwarg = {
        "passing": "ngs_passing",
        "rushing": "ngs_rushing",
        "receiving": "ngs_receiving",
    }[dispatch.ngs_stat_type]
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"

    ws_prior = read_partition(raw_root, "weekly_stats", season=season - 1)
    sc_prior = read_partition(raw_root, "snap_counts", season=season - 1)
    ngs_prior = read_partition(raw_root, ngs_table, season=season - 1)
    ws_curr = read_partition(raw_root, "weekly_stats", season=season)
    sc_curr = read_partition(raw_root, "snap_counts", season=season)
    dc_curr = read_partition(raw_root, "depth_charts", season=season)
    ngs_curr = read_partition(raw_root, ngs_table, season=season)
    sch_curr = read_partition(raw_root, "schedules", season=season)

    ws_full = pd.concat([ws_prior, ws_curr], ignore_index=True)
    sc_full = pd.concat([sc_prior, sc_curr], ignore_index=True)
    ngs_full = pd.concat([ngs_prior, ngs_curr], ignore_index=True)

    pbp_frames: list[pd.DataFrame] = []
    for s in (season - 1, season):
        try:
            pbp_frames.append(read_partition(raw_root, "pbp", season=s))
        except FileNotFoundError:
            continue
    pbp_full = pd.concat(pbp_frames, ignore_index=True) if pbp_frames else pd.DataFrame()

    weeks = sorted(int(w) for w in dc_curr["week"].unique())
    rows: list[pd.DataFrame] = []
    for week in weeks:
        kwargs: dict[str, Any] = {
            "weekly_stats": ws_full,
            "snap_counts": sc_full,
            "depth_charts": dc_curr,
            "schedules": sch_curr,
            "season": int(season),
            "as_of_week": int(week),
            "pbp": pbp_full,
            "draft_picks": draft_picks,
            ngs_kwarg: ngs_full,
        }
        feats = builder(**kwargs)
        if feats.empty:
            continue
        preds = model.predict_distribution(feats, ruleset=ruleset)
        ProjectionWeeklySchema.validate(preds)
        rows.append(preds)
        print(f"    [{position.value}] week {week}: {len(preds)} rows", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Project a full season retrospectively.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("reports") / "season_projection.csv")
    args = parser.parse_args()

    target_season = args.season
    train_seasons = range(args.train_start, target_season)  # exclusive of target

    ruleset = Ruleset.espn_ppr()
    draft_picks = _load_draft_picks(args.raw_root, target_season)
    print(f"Loaded draft_picks rows: {len(draft_picks)}", flush=True)

    all_preds: list[pd.DataFrame] = []
    for position in (Position.QB, Position.RB, Position.WR, Position.TE):
        model_class = POSITION_DISPATCH[position].default_model_class
        print(
            f"\n[{position.value}] training {args.train_start}-{target_season - 1} "
            f"({model_class})...",
            flush=True,
        )
        model = _train_production(
            args.raw_root, position, train_seasons=train_seasons, draft_picks=draft_picks
        )
        print(f"[{position.value}] model_id={model.model_id}", flush=True)
        print(f"[{position.value}] projecting {target_season}...", flush=True)
        preds = _project_season(
            args.raw_root,
            position,
            target_season,
            model=model,
            draft_picks=draft_picks,
            ruleset=ruleset,
        )
        all_preds.append(preds)

    weekly = pd.concat(all_preds, ignore_index=True)
    print(f"\nTotal weekly projection rows across positions: {len(weekly)}", flush=True)

    # Aggregate weekly mean -> season total per gsis_id.
    season_totals = weekly.groupby(["gsis_id", "position"], as_index=False).agg(
        season_total_mean=("mean", "sum"), n_weeks=("week", "nunique")
    )

    # Lookup full_name via id_map (read once, single file, no per-season partition).
    id_map = read_partition(args.raw_root, "id_map")
    season_totals = season_totals.merge(
        id_map[["gsis_id", "full_name", "team"]], on="gsis_id", how="left"
    )

    season_totals = season_totals.sort_values("season_total_mean", ascending=False).reset_index(
        drop=True
    )
    season_totals.insert(0, "rank", range(1, len(season_totals) + 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    season_totals.to_csv(args.out, index=False)
    print(f"\nWrote season totals CSV: {args.out}", flush=True)

    print(f"\n=== TOP 100 overall ({target_season} ESPN PPR projection) ===")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    print(
        season_totals.head(100)[
            ["rank", "full_name", "position", "team", "season_total_mean", "n_weeks"]
        ].to_string(index=False)
    )

    for pos_str in ("QB", "RB", "WR", "TE"):
        pos_df = season_totals[season_totals["position"] == pos_str].head(10).copy()
        pos_df.insert(0, "pos_rank", range(1, len(pos_df) + 1))
        print(f"\n=== TOP 10 {pos_str} ===")
        print(
            pos_df[
                ["pos_rank", "rank", "full_name", "team", "season_total_mean", "n_weeks"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
