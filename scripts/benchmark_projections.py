# scripts/benchmark_projections.py
"""Spike: benchmark our BaselineModel preseason projection vs ESPN's preseason
projection at predicting actual 2024 fantasy outcomes. Emits a verdict report.

Inputs:
  - data/external_projections/{season}/espn.parquet   (from pull_external_projections.py)
  - data/external_projections/{season}/sleeper_adp.parquet
  - reports/season_projection_{season}.csv            (from project_season.py --out)
  - data/raw weekly_stats + id_map                    (in-house)

Output:
  - reports/external_projection_benchmark_{season}.md

Preseason-vs-preseason only. Every stat line is scored through OUR PPR ruleset so
the comparison is under one scoring rule. Pure transforms are unit-tested; the
end-to-end run is a manual phase.

Usage:
    python scripts/benchmark_projections.py --season 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, score
from projections.store import read_partition

_STAT_FIELDS = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
)
_COUNT_FIELDS = frozenset(
    {
        "passing_tds",
        "interceptions",
        "rushing_tds",
        "receptions",
        "receiving_tds",
        "fumbles_lost",
    }
)


def _score_row(row: pd.Series[object], ruleset: Ruleset) -> float:
    sl = StatLine(
        passing_yards=float(row["passing_yards"]),
        passing_tds=round(float(row["passing_tds"])),
        interceptions=round(float(row["interceptions"])),
        rushing_yards=float(row["rushing_yards"]),
        rushing_tds=round(float(row["rushing_tds"])),
        receptions=round(float(row["receptions"])),
        receiving_yards=float(row["receiving_yards"]),
        receiving_tds=round(float(row["receiving_tds"])),
        fumbles_lost=round(float(row["fumbles_lost"])),
    )
    return score(sl, ruleset)


def actual_season_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Sum each player's weekly stat lines to a season total and score under `ruleset`.
    Position is the modal value across the player's weeks."""
    agg = {f: "sum" for f in _STAT_FIELDS}
    summed = weekly_stats.groupby("gsis_id", as_index=False).agg(agg)
    pos = weekly_stats.groupby("gsis_id")["position"].agg(lambda s: s.mode().iloc[0]).reset_index()
    out = summed.merge(pos, on="gsis_id", how="left")
    out["actual_pts"] = out.apply(lambda r: _score_row(r, ruleset), axis=1)
    return out[["gsis_id", "position", "actual_pts"]]


def our_season_points(csv_df: pd.DataFrame) -> pd.DataFrame:
    """Our model's CSV: season_total_mean is already PPR fantasy points."""
    out = csv_df[["gsis_id", "position", "season_total_mean"]].copy()
    out = out.rename(columns={"season_total_mean": "our_pts"})
    return out


def espn_season_points(espn: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Score ESPN's preseason stat line under `ruleset`, keyed by espn_id."""
    out = espn.copy()
    out["espn_pts"] = out.apply(lambda r: _score_row(r, ruleset), axis=1)
    return out[
        [
            "espn_id",
            "full_name",
            "position",
            "espn_pts",
            "espn_adp",
            "espn_pos_rank",
            "espn_actual_applied_total",
        ]
    ]


def build_benchmark_frame(
    espn: pd.DataFrame,
    ours: pd.DataFrame,
    actuals: pd.DataFrame,
    id_map: pd.DataFrame,
    sleeper: pd.DataFrame,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Join ESPN + our model + actuals on gsis_id (ESPN via id_map.espn_id,
    Sleeper ADP via id_map.sleeper_id). Base universe = actuals (ground truth).
    Position is taken from actuals."""
    espn_scored = espn_season_points(espn, ruleset)
    espn_keyed = espn_scored.merge(
        id_map[["gsis_id", "espn_id"]].dropna(subset=["espn_id"]), on="espn_id", how="left"
    )

    frame = actuals.copy()
    frame = frame.merge(ours[["gsis_id", "our_pts"]], on="gsis_id", how="left")
    # Drop ESPN's own position/full_name/espn_id before the merge: position comes
    # from actuals, full_name is re-attached from id_map below, and keeping any of
    # them here would create _x/_y collisions that break render_report.
    frame = frame.merge(
        espn_keyed.drop(columns=["position", "full_name", "espn_id"]).dropna(subset=["gsis_id"]),
        on="gsis_id",
        how="left",
    )

    sleeper_keyed = sleeper.merge(
        id_map[["gsis_id", "sleeper_id"]].dropna(subset=["sleeper_id"]),
        on="sleeper_id",
        how="left",
    ).dropna(subset=["gsis_id"])
    frame = frame.merge(sleeper_keyed[["gsis_id", "sleeper_adp"]], on="gsis_id", how="left")

    # full_name for readability (from id_map).
    frame = frame.merge(id_map[["gsis_id", "full_name"]], on="gsis_id", how="left")
    return frame


def source_metrics(
    frame: pd.DataFrame, pred_col: str, actual_col: str = "actual_pts"
) -> dict[str, float]:
    """RMSE / MAE / Spearman of pred vs actual over rows where both are present."""
    sub = frame[[pred_col, actual_col]].dropna()
    n = len(sub)
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "spearman": float("nan")}
    resid = sub[pred_col] - sub[actual_col]
    rmse = float((resid**2).mean() ** 0.5)
    mae = float(resid.abs().mean())
    spearman = (
        float(sub[pred_col].corr(sub[actual_col], method="spearman")) if n > 1 else float("nan")
    )
    return {"n": n, "rmse": rmse, "mae": mae, "spearman": spearman}


def top_n_by_rank(frame: pd.DataFrame, rank_col: str, n: int = 20) -> pd.DataFrame:
    """Top-n rows per position by smallest rank (best). Rows with NaN rank dropped."""
    ranked = frame.dropna(subset=[rank_col])
    return (
        ranked.sort_values(rank_col)
        .groupby("position", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_n_hit_rate(frame: pd.DataFrame, rank_col: str, n: int = 20) -> float:
    """Of each position's top-n by preseason rank, the share that finished top-n in actuals."""
    pre = top_n_by_rank(frame, rank_col, n)
    if pre.empty:
        return float("nan")
    actual_top = top_n_by_rank(
        frame.assign(_actual_rank=frame.groupby("position")["actual_pts"].rank(ascending=False)),
        "_actual_rank",
        n,
    )
    hit_keys = set(zip(actual_top["position"], actual_top["gsis_id"], strict=False))
    pre_keys = list(zip(pre["position"], pre["gsis_id"], strict=False))
    return sum(k in hit_keys for k in pre_keys) / len(pre_keys)


_POSITIONS = ("QB", "RB", "WR", "TE")


def _metric_block(frame: pd.DataFrame, label: str) -> list[str]:
    lines = [
        f"### {label}",
        "",
        "| Source | n | RMSE | MAE | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for src, col in (("ESPN", "espn_pts"), ("Ours", "our_pts")):
        m = source_metrics(frame, col)
        lines.append(
            f"| {src} | {m['n']} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['spearman']:.3f} |"
        )
    lines.append("")
    return lines


def render_report(frame: pd.DataFrame, season: int) -> str:
    out: list[str] = [f"# External Projection Benchmark — {season}", ""]
    out += [
        f"Preseason-vs-preseason. Every stat line scored through our ESPN PPR ruleset. "
        f"Base universe = {len(frame)} players with a {season} actual season.",
        "",
        "## Coverage / match rate",
        "",
        f"- ESPN matched: {int(frame['espn_pts'].notna().sum())} / {len(frame)}",
        f"- Ours matched: {int(frame['our_pts'].notna().sum())} / {len(frame)} "
        f"(unmatched are mostly rookies our model cannot project — a real, reportable weakness)",
        "",
        "## Full population",
        "",
    ]
    out += _metric_block(frame, "All QB/RB/WR/TE")
    for pos in _POSITIONS:
        out += _metric_block(frame[frame["position"] == pos], f"{pos} only")

    out += ["## Top-20 per position (by ESPN preseason rank)", ""]
    top = top_n_by_rank(frame, "espn_pos_rank", 20)
    out += _metric_block(top, "Top-20/pos — all")

    out += ["## Veterans-only (rows our model projects)", ""]
    out += _metric_block(frame[frame["our_pts"].notna()], "Veterans — all")

    out += [
        "## ADP rank lens (vs actual finish)",
        "",
        "| Ranking | Spearman vs actual | Top-20 hit rate |",
        "|---|---:|---:|",
    ]
    for label, col in (("ESPN ADP", "espn_adp"), ("Sleeper ADP", "sleeper_adp")):
        sub = frame[[col, "actual_pts"]].dropna()
        # ADP: smaller = better, so correlate negative ADP with actual points.
        sp = (
            float((-sub[col]).corr(sub["actual_pts"], method="spearman"))
            if len(sub) > 1
            else float("nan")
        )
        hit = top_n_hit_rate(frame.assign(_adp_rank=frame[col]), "_adp_rank", 20)
        out.append(f"| {label} | {sp:.3f} | {hit:.2f} |")
    out += [""]

    espn_all = source_metrics(frame, "espn_pts")
    our_all = source_metrics(frame[frame["our_pts"].notna()], "our_pts")
    out += [
        "## Verdict",
        "",
        f"- ESPN full-population RMSE: {espn_all['rmse']:.2f} (n={espn_all['n']})",
        f"- Our model veterans-only RMSE: {our_all['rmse']:.2f} (n={our_all['n']})",
        "",
        "_Reading notes:_ one season (2024) — rerun on 2023 if close. ESPN is one strong "
        "source, not a consensus: losing to ESPN alone strongly implies losing to a consensus; "
        "beating ESPN alone does not yet prove beating a consensus. Our model cannot project "
        "pure rookies (no prior-NFL features); the veterans-only cut is the fairest model-vs-model "
        "comparison.",
        "",
        "**Go/no-go for sub-project #2 (external consensus layer):** _fill in after reading the "
        "numbers above — see plan Task 9._",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark external vs our projections for one season."
    )
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--ext-root", type=Path, default=Path("data/external_projections"))
    ap.add_argument("--our-csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    our_csv = args.our_csv or Path("reports") / f"season_projection_{args.season}.csv"
    out_path = args.out or Path("reports") / f"external_projection_benchmark_{args.season}.md"
    ext_dir = args.ext_root / str(args.season)
    ruleset = Ruleset.espn_ppr()

    espn = pd.read_parquet(ext_dir / "espn.parquet")
    sleeper = pd.read_parquet(ext_dir / "sleeper_adp.parquet")
    ours = our_season_points(pd.read_csv(our_csv))
    weekly = read_partition(args.raw_root, "weekly_stats", season=args.season)
    actuals = actual_season_points(weekly, ruleset)
    id_map = read_partition(args.raw_root, "id_map")

    frame = build_benchmark_frame(espn, ours, actuals, id_map, sleeper, ruleset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(frame, args.season), encoding="utf-8")
    print(f"Wrote {out_path} ({len(frame)} players)", flush=True)


if __name__ == "__main__":
    main()
