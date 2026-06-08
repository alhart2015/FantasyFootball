# scripts/benchmark_projections.py
"""Spike: join external + our-model + actual fantasy projections, score through one
PPR ruleset, and emit per-source / per-position / per-cohort error metrics.

!!! DO NOT USE FOR A PRESEASON / DRAFT VERDICT !!!
-------------------------------------------------
This was built to compare ESPN's PRESEASON projection against our model's
"projection" from `project_season.py`. That comparison is INVALID: `project_season.py`
is NOT a preseason projection. Our model is a weekly, in-season model whose trailing
features read the current season, and it only projects players who are active each
week — so its season totals secretly use the 2024 outcomes we are trying to predict
(e.g. it projected the injured Christian McCaffrey for only 4 weeks). Scoring that
against ESPN's honest preseason forecast flatters our model and is meaningless.
See `reports/external_projection_benchmark_2024.md` (§1) for the full finding.

The machinery here (join + scoring + RMSE/MAE/Spearman/cohorts) is correct and is
reusable for a FAIR comparison at matched information cutoff — specifically the
follow-up weekly start/sit benchmark (our WEEKLY projection vs ESPN's WEEKLY
projection vs weekly actuals). It must not be pointed at `project_season.py` output
and reported as a preseason result.

Inputs:
  - data/external_projections/{season}/espn.parquet   (from pull_external_projections.py)
  - data/external_projections/{season}/sleeper_adp.parquet
  - reports/season_projection_{season}.csv            (from project_season.py --out)
  - data/raw weekly_stats + id_map                    (in-house)

Output:
  - reports/external_projection_benchmark_{season}.md

Every stat line is scored through OUR PPR ruleset so the comparison is under one
scoring rule. Pure transforms are unit-tested.

Usage:
    python scripts/benchmark_projections.py --season 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pull_external_projections import COUNT_FIELDS, STAT_FIELDS, round_count

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, score
from projections.store import read_partition


def _score_row(row: pd.Series[object], ruleset: Ruleset) -> float:
    kwargs = {
        f: (round_count(float(row[f])) if f in COUNT_FIELDS else float(row[f])) for f in STAT_FIELDS
    }
    # dict-unpack into StatLine's typed kwargs (count fields int, yards float); the
    # inferred dict[str, float] is why the narrow arg-type ignore is needed.
    return score(StatLine(**kwargs), ruleset)  # type: ignore[arg-type]


def actual_season_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Sum each player's weekly stat lines to a season total and score under `ruleset`.
    Position is the modal value across the player's weeks."""
    agg = {f: "sum" for f in STAT_FIELDS}
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
        ]
    ]


def _normalize_join_id(s: pd.Series) -> pd.Series:
    """id_map stores espn_id/sleeper_id as float-stringified values ('4374302.0')
    with a string dtype; external pulls write clean int-strings ('4374302') as
    object dtype. Canonicalize both sides to a plain string with any trailing '.0'
    stripped so the merge actually matches (and the dtypes line up). Without this,
    the join silently produces ZERO matches."""
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


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
    espn_scored["espn_id"] = _normalize_join_id(espn_scored["espn_id"])
    # dropna + drop_duplicates: id_map has duplicate espn_id rows; without the dedup
    # a duplicate would multiply a player's rows and inflate every metric.
    id_espn = id_map[["gsis_id", "espn_id"]].dropna(subset=["espn_id"]).copy()
    id_espn["espn_id"] = _normalize_join_id(id_espn["espn_id"])
    id_espn = id_espn.drop_duplicates(subset=["espn_id"])
    espn_keyed = espn_scored.merge(id_espn, on="espn_id", how="left")

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

    sleeper = sleeper.copy()
    sleeper["sleeper_id"] = _normalize_join_id(sleeper["sleeper_id"])
    id_sleeper = id_map[["gsis_id", "sleeper_id"]].dropna(subset=["sleeper_id"]).copy()
    id_sleeper["sleeper_id"] = _normalize_join_id(id_sleeper["sleeper_id"])
    id_sleeper = id_sleeper.drop_duplicates(subset=["sleeper_id"])
    sleeper_keyed = sleeper.merge(id_sleeper, on="sleeper_id", how="left").dropna(
        subset=["gsis_id"]
    )
    frame = frame.merge(sleeper_keyed[["gsis_id", "sleeper_adp"]], on="gsis_id", how="left")

    # full_name for readability (from id_map); dedup to avoid row multiplication
    # when id_map has duplicate gsis_id rows.
    id_name = id_map[["gsis_id", "full_name"]].drop_duplicates(subset=["gsis_id"])
    frame = frame.merge(id_name, on="gsis_id", how="left")
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

    vets = frame[frame["our_pts"].notna()]
    espn_all = source_metrics(frame, "espn_pts")  # ESPN, full population
    espn_vets = source_metrics(vets, "espn_pts")  # ESPN, veterans-only (matched to our model)
    our_vets = source_metrics(vets, "our_pts")  # ours, veterans-only
    out += [
        "## Verdict",
        "",
        f"- **Matched (veterans-only) — the fair head-to-head:** "
        f"ESPN RMSE {espn_vets['rmse']:.2f} vs Ours RMSE {our_vets['rmse']:.2f} "
        f"(n={our_vets['n']})",
        f"- Full-population ESPN RMSE: {espn_all['rmse']:.2f} (n={espn_all['n']}) "
        f"— broader coverage; includes rookies our model cannot project",
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
