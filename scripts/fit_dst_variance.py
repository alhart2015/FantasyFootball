"""Fit the D/ST entries in configs/performance_variance_params.json.

The season simulators sample weekly points through `performance_variance.sample_weekly_points`,
which is keyed by position with a `"default"` fallback. Before this, a defense fell through to
`default` -- an affine std fitted on skill players, which understates a position whose weekly
score swings on a single return touchdown.

Source: ESPN's own D/ST stat vectors, scored through `scoring.dst.score_dst`. Actuals are
`statSourceId=0, statSplitTypeId=1` (weekly); the preseason season projection is
`statSourceId=1, statSplitTypeId=0`. Using ESPN for both sides keeps the vocabulary identical
and needs no points-allowed / yards-allowed bucketing logic of our own -- the buckets arrive
already computed, in the same stat ids the scoring path consumes.

**Known limitation, stated rather than hidden: ESPN's public endpoint carries ONE completed
season of weekly D/ST actuals (2025, 544 team-weeks).** That is ample for the affine weekly-std
fit (544 observations, 2 parameters) and thin for `mean_mult_log_sd`, which gets 32 -- one per
defense. A multi-season fit needs a real D/ST actuals ingest built on
`nflreadpy.load_team_stats` plus schedule scores for points/yards allowed; that is issue #122's
half of the work. Re-run this script once that exists.

Usage:
    python scripts/fit_dst_variance.py                 # print the fit, write nothing
    python scripts/fit_dst_variance.py --write         # update the params file
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from projections.ingest.espn_league import ESPN_PRO_TEAMS, parse_ruleset
from projections.ingest.external_projections import ESPN_DST_POSITION_ID, fetch_espn
from projections.schemas import Position, normalize_team_code
from projections.scoring.dst import score_dst

REPO = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO / "configs" / "performance_variance_params.json"
#: Relative to the working directory, not the script: `data/` is gitignored, so a worktree
#: does not carry it and the payload lives in the main checkout.
DEFAULT_PAYLOAD = Path("data/leagues/critts_2025_2026/espn_raw.json")

#: The completed season ESPN exposes weekly D/ST actuals for. See the module docstring.
ACTUALS_SEASON = 2025


def collect(payload: dict, ruleset) -> tuple[dict[str, list[float]], dict[str, float]]:
    """-> ({team: [weekly actual points]}, {team: projected season points})."""
    weekly: dict[str, list[float]] = {}
    projected: dict[str, float] = {}
    for entry in payload.get("players", []):
        pl = entry.get("player", {})
        if pl.get("defaultPositionId") != ESPN_DST_POSITION_ID:
            continue
        raw_team = ESPN_PRO_TEAMS.get(int(pl.get("proTeamId", 0) or 0))
        if raw_team is None:
            continue
        team = normalize_team_code(raw_team).value
        for s in pl.get("stats", []):
            if s.get("seasonId") != ACTUALS_SEASON or not s.get("stats"):
                continue
            points = score_dst(s["stats"], ruleset)
            if s.get("statSourceId") == 0 and s.get("statSplitTypeId") == 1:
                weekly.setdefault(team, []).append(points)
            elif s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 0:
                projected[team] = points
    return weekly, projected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="Update the params file in place.")
    ap.add_argument(
        "--league-payload",
        type=Path,
        default=DEFAULT_PAYLOAD,
        help="ESPN league payload supplying the D/ST scoring rules to fit under.",
    )
    args = ap.parse_args()

    ruleset, _ = parse_ruleset(json.loads(args.league_payload.read_text(encoding="utf-8")))
    weekly, projected = collect(fetch_espn(ACTUALS_SEASON + 1, limit=1500), ruleset)

    teams = sorted(set(weekly) & set(projected))
    n_weeks = sum(len(v) for v in weekly.values())
    print(f"{len(teams)} defenses, {n_weeks} team-weeks of {ACTUALS_SEASON} actuals")
    print(f"scored under {ruleset.name} with {len(ruleset.dst_stat_points)} D/ST categories\n")

    # --- weekly_std_affine: per-game std as a function of per-game mean ---
    per_game_mean = np.array([statistics.fmean(weekly[t]) for t in teams])
    per_game_std = np.array([statistics.stdev(weekly[t]) for t in teams])
    a, b = np.polyfit(per_game_mean, per_game_std, 1)
    resid = per_game_std - (a * per_game_mean + b)
    print("weekly_std_affine['DST']:")
    print(f"    a = {a:.6f}   b = {b:.6f}")
    print(
        f"    per-game mean {per_game_mean.min():.2f}..{per_game_mean.max():.2f}, "
        f"std {per_game_std.min():.2f}..{per_game_std.max():.2f}"
    )
    print(f"    fit residual RMS {float(np.sqrt((resid**2).mean())):.3f}\n")

    # --- mean_mult_log_sd: dispersion of actual/projected season totals ---
    ratios = [sum(weekly[t]) / projected[t] for t in teams if projected[t] > 0]
    logs = [float(np.log(r)) for r in ratios]
    log_sd = statistics.stdev(logs)
    print(f"mean_mult_log_sd['DST|veteran'] = {log_sd:.6f}   (n={len(logs)})")
    print(
        f"    actual/projected ratio: min {min(ratios):.2f}  median "
        f"{statistics.median(ratios):.2f}  max {max(ratios):.2f}\n"
    )

    blob = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    print("versus the skill positions already fitted:")
    for pos in ("QB", "RB", "WR", "TE", "default"):
        coef = blob["weekly_std_affine"][pos]
        print(f"    {pos:<8} a={coef['a']:.4f}  b={coef['b']:.4f}")
    print(f"    {'DST':<8} a={a:.4f}  b={b:.4f}   <-- new")

    if not args.write:
        print("\n(dry run — pass --write to update the params file)")
        return

    dst = Position.DST.value
    blob["weekly_std_affine"][dst] = {"a": float(a), "b": float(b)}
    # A defense has no rookie/veteran distinction — a team is not a rookie. Both tiers get the
    # same value so the lookup never falls through to the skill default.
    blob["mean_mult_log_sd"][f"{dst}|veteran"] = log_sd
    blob["mean_mult_log_sd"][f"{dst}|rookie"] = log_sd
    PARAMS_PATH.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {PARAMS_PATH}")


if __name__ == "__main__":
    main()
