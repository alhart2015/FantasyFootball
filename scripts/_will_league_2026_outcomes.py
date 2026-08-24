"""Post-draft outcome sim for Will's 12-team half-PPR auction league (2026).

Takes the completed draft (a TSV of pick / salary / player / nfl_team / pos / fantasy_team),
maps each pick to a gsis_id in the league's VORP pool, then runs the standard forward
predictive sim: sample every rostered player's weekly points from the fitted
performance-variance model, gate them by the availability model (per-player injury Bernoulli
+ team bye), set each week's lineup by projection, and score a 14-week regular season plus a
3-round / 6-team playoff. Repeat over many seasons and many schedule draws.

K and DST are dropped -- the pool carries no kicker or defense projections, so those two
roster slots are outside the model. Everything reported is the 13-slot skill roster.

Usage:
    python scripts/_will_league_2026_outcomes.py --picks <tsv> --sims 2000
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams, sample_weekly_points
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.assistant.season_value import _availability_mask, _bye_indices
from projections.draft.backtest.inputs import DEFAULT_CALENDAR
from projections.draft.backtest.league import score_drafted_league
from projections.draft.backtest.schedule import regular_season_schedule
from projections.draft.league_config import LeagueConfig

_POOL = Path("data/vorp_2026/will_half12.parquet")
_CONFIG = Path("configs/will_half12_pass5.league.json")
_VARIANCE = Path("configs/performance_variance_params.json")
_GAMES = 17  # matches performance_variance._GAMES: season_mean is over a 17-game slate
_SKIP_POSITIONS = frozenset({"K", "DST", "DEF"})

# Draft-app display name -> pool full_name, for the cases normalization can't bridge.
_NAME_OVERRIDES: dict[str, str] = {}


def _norm(name: str) -> str:
    """Fold a display name to a match key: ascii, lowercase, no punctuation, no suffix."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", " ").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_picks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.sort_values("pick").reset_index(drop=True)
    return df


def resolve_ids(picks: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach gsis_id to every skill pick. Returns (matched picks, unmatched display names)."""
    by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for gid, full, pos in zip(pool["gsis_id"], pool["full_name"], pool["position"], strict=True):
        by_key[_norm(str(full))].append((str(gid), str(pos)))

    skill = picks[~picks["pos"].isin(_SKIP_POSITIONS)].copy()
    ids: list[str | None] = []
    unmatched: list[str] = []
    for name, pos in zip(skill["player"], skill["pos"], strict=True):
        key = _norm(_NAME_OVERRIDES.get(str(name), str(name)))
        cands = [c for c in by_key.get(key, []) if c[1] == str(pos)] or by_key.get(key, [])
        if not cands:
            ids.append(None)
            unmatched.append(f"{name} ({pos})")
        else:
            ids.append(cands[0][0])
    skill["gsis_id"] = ids
    return skill[skill["gsis_id"].notna()].copy(), unmatched


def build_rosters(skill: pd.DataFrame) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Seat 1..N per fantasy team, in first-pick order. Returns (rosters, seat -> team name)."""
    order = list(dict.fromkeys(skill.sort_values("pick")["fantasy_team"]))
    seat_of = {team: i + 1 for i, team in enumerate(order)}
    rosters: dict[int, list[str]] = {seat: [] for seat in seat_of.values()}
    for team, gid in zip(skill["fantasy_team"], skill["gsis_id"], strict=True):
        rosters[seat_of[team]].append(str(gid))
    return rosters, {seat: team for team, seat in seat_of.items()}


def run(
    rosters: dict[int, list[str]],
    pool: pd.DataFrame,
    config: LeagueConfig,
    availability: PlayerAvailability,
    params: VarianceParams,
    *,
    n_sims: int,
    seed: int,
    injuries: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    """Sample `n_sims` seasons (fresh schedule + fresh player outcomes each) and score them.

    `injuries=False` keeps every player available every week (byes included) -- the healthy-season
    counterfactual, used to size how much the availability gate costs each roster.
    """
    rng = np.random.default_rng(seed)
    weeks = sorted(set(DEFAULT_CALENDAR.regular_weeks) | set(DEFAULT_CALENDAR.playoff_weeks))
    rostered = sorted({g for r in rosters.values() for g in r})
    sub = pool[pool["gsis_id"].astype(str).isin(rostered)].copy()
    sub["gsis_id"] = sub["gsis_id"].astype(str)
    sub = sub.set_index("gsis_id").loc[rostered].reset_index()

    gsis = sub["gsis_id"].to_numpy(dtype=str)
    positions = sub["position"].astype(str).to_numpy()
    means = sub["season_mean_fpts"].to_numpy(dtype=np.float64)
    rookie = sub["is_rookie"].to_numpy(dtype=bool)
    p = np.array([availability.p_week(str(g)) for g in gsis], dtype=np.float64)
    bye_idx = _bye_indices(availability, gsis, weeks)

    # Lineups are set by projection: a flat per-week mean, the manager's pre-season view.
    weekly_mean = {str(g): float(m) / _GAMES for g, m in zip(gsis, means, strict=True)}

    acc: dict[int, dict[str, list[float]]] = {
        s: {"pf": [], "wins": [], "playoff": [], "champ": []} for s in rosters
    }
    labels = {s: str(s) for s in rosters}

    for i in range(n_sims):
        pts = sample_weekly_points(
            params, positions, means, rookie, n_sims=1, n_weeks=len(weeks), rng=rng
        )[0]  # (n_weeks, n_players)
        uniforms = rng.random((1, len(weeks), len(gsis)))  # drawn either way, so seeds align
        avail = (
            _availability_mask(uniforms, p, bye_idx)[0]
            if injuries
            else np.ones((len(weeks), len(gsis)), dtype=bool)
        )  # (n_weeks, n_players)

        actual_lookup: dict[tuple[str, int], float] = {}
        proj_lookup: dict[tuple[str, int], float] = {}
        for w, wk in enumerate(weeks):
            for j, g in enumerate(gsis):
                if avail[w, j]:
                    actual_lookup[(str(g), wk)] = float(pts[w, j])
                    proj_lookup[(str(g), wk)] = weekly_mean[str(g)]
                # unavailable -> absent from proj_lookup => unstartable that week

        sched = regular_season_schedule(
            n_teams=config.n_teams, n_weeks=len(DEFAULT_CALENDAR.regular_weeks), rng=rng
        )
        outcome = score_drafted_league(
            rosters,
            sub,
            config,
            proj_lookup=proj_lookup,
            actual_lookup=actual_lookup,
            calendar=DEFAULT_CALENDAR,
            strategy_labels=labels,
            sched=sched,
        )
        for r in outcome.actual:
            cell = acc[r.seat]
            cell["pf"].append(r.points_for)
            cell["wins"].append(float(r.wins))
            cell["playoff"].append(1.0 if r.made_playoffs else 0.0)
            cell["champ"].append(1.0 if r.is_champion else 0.0)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n_sims} seasons", file=sys.stderr, flush=True)

    return {s: {k: np.array(v) for k, v in cell.items()} for s, cell in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", type=Path, required=True)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-injury", action="store_true", help="healthy-season counterfactual")
    args = ap.parse_args()

    pool = pd.read_parquet(_POOL)
    pool = attach_is_rookie(pool, season=2026, data_root=Path("data"))
    config = LeagueConfig.model_validate_json(_CONFIG.read_text())
    params = VarianceParams.load(_VARIANCE)

    picks = load_picks(args.picks)
    skill, unmatched = resolve_ids(picks, pool)
    if unmatched:
        print(f"UNMATCHED ({len(unmatched)}): {', '.join(unmatched)}", file=sys.stderr)

    rosters, team_of = build_rosters(skill)
    for seat, r in sorted(rosters.items()):
        print(f"seat {seat:2d} {team_of[seat]:<28} {len(r)} skill players", file=sys.stderr)

    availability = load_store_availability(pool, season=2026, data_root=Path("data"))
    out = run(
        rosters,
        pool,
        config,
        availability,
        params,
        n_sims=args.sims,
        seed=args.seed,
        injuries=not args.no_injury,
    )

    rows = []
    for seat, cell in out.items():
        pf, wins = cell["pf"], cell["wins"]
        rows.append(
            {
                "team": team_of[seat],
                "proj_pf": pf.mean(),
                "pf_sd": pf.std(ddof=1),
                "pf_p10": np.percentile(pf, 10),
                "pf_p90": np.percentile(pf, 90),
                "wins": wins.mean(),
                "wins_sd": wins.std(ddof=1),
                "playoff_pct": 100 * cell["playoff"].mean(),
                "champ_pct": 100 * cell["champ"].mean(),
                "playoff_se": 100 * cell["playoff"].std(ddof=1) / np.sqrt(len(pf)),
                "champ_se": 100 * cell["champ"].std(ddof=1) / np.sqrt(len(pf)),
            }
        )
    res = pd.DataFrame(rows).sort_values("champ_pct", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    if args.out:
        res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
