"""Offline fitter for the weekly performance-variance model (spec 2026-06-14).

Fits, from historical data, the params consumed by
``projections.draft.assistant.performance_variance``:
  - per-position affine ``std = a*pg + b`` (per-game weekly std vs per-game weekly mean),
  - per-(position, rookie) lognormal log-SD of ``log(realized_pg / projected_pg)``.

Two season ranges (external_projections exists only for 2021+, weekly_stats for 2018+):
the affine uses weekly_stats 2018-2025 (no projection needed); the log-SD uses only seasons
with a projection (2021-2025). Writes ``configs/performance_variance_params.json``.

Usage:
    python scripts/fit_performance_variance.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_MIN_AFFINE = 20
_MIN_LOGSD = 15
_GAMES = 17
# Draft-relevant filters (match the validated brainstorming analysis: only fantasy-relevant
# player-seasons, else deep/marginal players blow up the variance). A player-season counts toward
# the per-game affine if it has >= _MIN_FIT_GAMES active games and mean per-game >= _MIN_MEAN_PG;
# toward the log-SD only if it ALSO has a projected season >= _MIN_PROJ_SEASON.
_MIN_FIT_GAMES = 8
_MIN_MEAN_PG = 6.0
_MIN_PROJ_SEASON = 50.0
_AFFINE_SEASONS = range(2018, 2026)
_PROJECTION_SEASONS = range(2021, 2026)
_OUT_PATH = Path("configs/performance_variance_params.json")
_LEAGUE_CONFIG = Path("configs/league_espn_half_16team.json")


def fit_params(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit variance params from per-player-season rows.

    Each row: ``position`` (str), ``weekly`` (1-D array of active-game points), ``projected_pg``
    (float; <=0 means no projection -> affine only), ``is_rookie`` (bool).
    """
    by_pos: dict[str, list[tuple[float, float]]] = {}
    log_by_cell: dict[str, list[float]] = {}
    all_ms: list[tuple[float, float]] = []
    all_log: dict[str, list[float]] = {"veteran": [], "rookie": []}
    for r in rows:
        w = np.asarray(r["weekly"], dtype=float)
        if w.size < _MIN_FIT_GAMES:  # too few games to estimate anything for this player-season
            continue
        mean_pg, std_pg = float(w.mean()), float(w.std())
        # Affine: real-usage player-seasons (mean per-game >= floor) so scrubs don't flatten it.
        if mean_pg >= _MIN_MEAN_PG:
            by_pos.setdefault(r["position"], []).append((mean_pg, std_pg))
            all_ms.append((mean_pg, std_pg))
        # log-SD: gate on the PROJECTION (draft-relevant), NOT realized mean — busts must stay in
        # the ratio (excluding low-realized seasons would bias the projection-miss SD downward).
        ppg = float(r["projected_pg"])
        if ppg * _GAMES >= _MIN_PROJ_SEASON:
            ratio = mean_pg / ppg
            if np.isfinite(ratio) and ratio > 0:
                tier = "rookie" if r["is_rookie"] else "veteran"
                log_by_cell.setdefault(f"{r['position']}|{tier}", []).append(float(np.log(ratio)))
                all_log[tier].append(float(np.log(ratio)))

    def affine(ms: list[tuple[float, float]]) -> dict[str, float]:
        arr = np.array(ms)
        a, b = np.polyfit(arr[:, 0], arr[:, 1], 1)
        return {"a": float(a), "b": float(b)}

    weekly_std_affine = {"default": affine(all_ms)}
    for pos, ms in by_pos.items():
        weekly_std_affine[pos] = (
            affine(ms) if len(ms) >= _MIN_AFFINE else weekly_std_affine["default"]
        )

    def log_sd(values: list[float]) -> float:
        return float(np.std(values)) if values else 0.0

    mean_mult_log_sd = {
        "default|veteran": log_sd(all_log["veteran"]),
        "default|rookie": log_sd(all_log["rookie"]),
    }
    for cell, logs in log_by_cell.items():
        if len(logs) >= _MIN_LOGSD:
            mean_mult_log_sd[cell] = log_sd(logs)
    return {"weekly_std_affine": weekly_std_affine, "mean_mult_log_sd": mean_mult_log_sd}


def _build_rows() -> list[dict[str, Any]]:
    """Assemble per-player-season rows from the store (network-free; reads local parquet)."""
    from projections.draft.backtest.draft_basis import build_draft_basis
    from projections.draft.backtest.weekly_actuals import build_weekly_actuals
    from projections.draft.league_config import LeagueConfig
    from projections.schemas import ExternalProjectionSchema
    from projections.store import read_latest_partition, read_partition

    cfg = LeagueConfig.model_validate_json(_LEAGUE_CONFIG.read_text())
    seen: set[str] = set()  # gsis_ids appearing in any earlier season -> rookie detection
    rows: list[dict[str, Any]] = []
    for yr in _AFFINE_SEASONS:
        ws = read_partition(Path("data/raw"), "weekly_stats", season=yr)
        ws = ws[ws["position"].isin(["QB", "RB", "WR", "TE"])]
        act = build_weekly_actuals(ws, ruleset=cfg.ruleset)
        act["gsis_id"] = act["gsis_id"].astype(str)
        pos_of = dict(zip(ws["gsis_id"].astype(str), ws["position"].astype(str), strict=False))
        proj_pg: dict[str, float] = {}
        if yr in _PROJECTION_SEASONS:
            ext = ExternalProjectionSchema.validate(
                read_latest_partition(Path("data/raw"), "external_projections", season=yr)
            )
            pool = build_draft_basis(ext, league_config=cfg)
            proj_pg = {
                str(g): float(p) / _GAMES
                for g, p in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=False)
                if p > 0
            }
        appeared_before = set(seen)  # snapshot before adding this season
        for g, sub in act.groupby("gsis_id"):
            gid = str(g)
            rows.append(
                {
                    "gsis_id": gid,
                    "position": pos_of.get(gid, "?"),
                    "season": yr,
                    "weekly": sub["actual_points"].to_numpy(dtype=float),
                    "projected_pg": proj_pg.get(gid, 0.0),
                    "is_rookie": gid not in appeared_before,
                }
            )
        seen.update(act["gsis_id"].tolist())
    return [r for r in rows if r["position"] in ("QB", "RB", "WR", "TE")]


def main() -> None:
    params = fit_params(_build_rows())
    _OUT_PATH.write_text(json.dumps(params, indent=2, sort_keys=True))
    print(f"Wrote {_OUT_PATH}")
    print(json.dumps(params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
