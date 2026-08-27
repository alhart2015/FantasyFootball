"""Can a roster swap be distinguished from simulation noise?

**A gate, not a diagnostic.** §5 of the waiver-recommender spec proposes reporting a swap as a
change in expected season wins. That is only worth printing if the change is bigger than the
error on it, and writing "use common random numbers" in a design document does not establish
that it is. This measures it.

    python scripts/measure_swap_noise.py
    python scripts/measure_swap_noise.py --n-sims 4000 --repeats 12

Three questions, in order:

1. **How noisy is one run?** Simulate the same unchanged league at many seeds and look at the
   spread of projected wins for one team. That is the floor an unpaired comparison has to clear.

2. **Does pairing help, and by how much?** Simulate baseline and swapped rosters at the SAME
   seed, take the difference, repeat across seeds. Common random numbers shrink the error on the
   difference, because the two runs share their draws and some of the noise cancels.

   **Measured at about 2x, not the order of magnitude textbook CRN suggests** --
   `project_league_standings` reseeds internally, so a roster change perturbs the draw sequence
   and only part of the noise cancels. An earlier version of this docstring claimed "far below
   the error on either estimate", which is not what 2x means.

3. **Is a realistic swap visible?** Apply a swap worth roughly what a real waiver add is worth
   -- a few points a week -- and see whether the measured Δ wins is several times its own
   standard error.

If (3) fails at a tolerable sim count, the tool reports an interval rather than a point
estimate, and this script is what says so.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.midseason.standings import project_league_standings
from projections.schemas import _PYARROW_STR, VorpTableSchema

#: A league big enough to behave like the real one. Ids are non-contiguous on purpose, matching
#: what ESPN actually returns.
TEAM_IDS: tuple[int, ...] = tuple(range(1, 17))
POSITIONS: tuple[str, ...] = ("QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR")


@dataclass(frozen=True)
class Settings:
    n_sims: int
    repeats: int
    swap_points: float
    my_team: int


def _gsis(team_id: int, index: int) -> str:
    return f"00-{team_id:04d}{index:03d}"


def _pool(bump: dict[str, float] | None = None) -> pd.DataFrame:
    """A VORP pool for the whole league, optionally with one player's projection changed."""
    rows: list[dict[str, object]] = []
    for rank, team_id in enumerate(TEAM_IDS):
        for i, pos in enumerate(POSITIONS):
            gsis = _gsis(team_id, i)
            mean = 240.0 - 4.0 * rank - 12.0 * i
            rows.append(
                {
                    "gsis_id": gsis,
                    "full_name": f"Player {team_id}-{i}",
                    "position": pos,
                    "season_mean_fpts": mean + (bump or {}).get(gsis, 0.0),
                    "vorp": mean - 80.0,
                    "replacement_fpts": 80.0,
                    "is_rookie": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame["gsis_id"] = frame["gsis_id"].astype(_PYARROW_STR)
    frame["position"] = frame["position"].astype(_PYARROW_STR)
    frame["full_name"] = frame["full_name"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(frame)


def _payload(played_weeks: int = 0) -> dict[str, object]:
    """A 16-team league with a real round-robin schedule and full rosters."""
    teams = [
        {
            "id": team_id,
            "name": f"Team {team_id}",
            "roster": {
                "entries": [
                    {
                        "lineupSlotId": 20 if i >= 6 else (0, 2, 2, 4, 4, 6)[i],
                        "playerId": 100_000 + team_id * 100 + i,
                        "playerPoolEntry": {
                            "player": {
                                "id": 100_000 + team_id * 100 + i,
                                "fullName": f"Player {team_id}-{i}",
                                "defaultPositionId": {"QB": 1, "RB": 2, "WR": 3, "TE": 4}[pos],
                                "proTeamId": 1,
                            }
                        },
                    }
                    for i, pos in enumerate(POSITIONS)
                ]
            },
        }
        for team_id in TEAM_IDS
    ]
    schedule = []
    half = len(TEAM_IDS) // 2
    order = list(TEAM_IDS)
    for week in range(1, 15):
        for home, away in zip(order[:half], order[half:], strict=True):
            played = week <= played_weeks
            schedule.append(
                {
                    "matchupPeriodId": week,
                    "winner": "HOME" if played else "UNDECIDED",
                    "home": {"teamId": home, "totalPoints": 110.0 if played else 0.0},
                    "away": {"teamId": away, "totalPoints": 100.0 if played else 0.0},
                }
            )
        order = [order[0], order[-1], *order[1:-1]]
    return {
        "teams": teams,
        "schedule": schedule,
        "members": [],
        "settings": {
            "name": "noise probe",
            "scheduleSettings": {
                "matchupPeriodCount": 14,
                "playoffTeamCount": 6,
                "playoffMatchupPeriodLength": 1,
            },
            "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1, "20": 2}},
            # statId 53 is receptions; 0.5 makes it half-PPR, which is this league.
            # A genuinely empty scoringItems is rejected upstream, and rightly -- a league with
            # no scoring rules is a payload that lost its settings view, not a real league.
            "scoringSettings": {"scoringItems": [{"statId": 53, "points": 0.5}]},
        },
    }


def _id_map() -> pd.DataFrame:
    rows = [
        {"espn_id": str(100_000 + team_id * 100 + i), "gsis_id": _gsis(team_id, i)}
        for team_id in TEAM_IDS
        for i in range(len(POSITIONS))
    ]
    frame = pd.DataFrame(rows)
    for column in ("espn_id", "gsis_id"):
        frame[column] = frame[column].astype(_PYARROW_STR)
    return frame


def _projected_wins(pool: pd.DataFrame, seed: int, settings: Settings) -> float:
    run = project_league_standings(
        _payload(),
        pool,
        _id_map(),
        PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
        VarianceParams.load(),
        season=2026,
        n_sims=settings.n_sims,
        rng=np.random.default_rng(seed),
    )
    row = run.standings[run.standings["team_id"] == settings.my_team]
    return float(row.iloc[0]["projected_wins"])


def report(settings: Settings) -> int:
    baseline_pool = _pool()
    # A swap worth roughly what a real waiver add is worth: a few points a week over the season.
    improved_pool = _pool({_gsis(settings.my_team, 7): settings.swap_points})

    print(f"16 teams · {settings.n_sims} sims · {settings.repeats} seeds")
    print(f"swap worth {settings.swap_points:.0f} season points on team {settings.my_team}\n")

    baselines = [_projected_wins(baseline_pool, seed, settings) for seed in range(settings.repeats)]
    improved = [_projected_wins(improved_pool, seed, settings) for seed in range(settings.repeats)]

    print("=== 1. spread of ONE estimate across seeds ===")
    spread = statistics.stdev(baselines)
    print(f"  projected wins: mean {statistics.mean(baselines):.4f}  sd {spread:.4f}")
    print(f"  -> an UNPAIRED difference carries about {spread * (2**0.5):.4f} wins of error")

    print("\n=== 2. the same difference, PAIRED (common random numbers) ===")
    paired = [after - before for before, after in zip(baselines, improved, strict=True)]
    paired_mean = statistics.mean(paired)
    paired_sd = statistics.stdev(paired)
    stderr = paired_sd / (len(paired) ** 0.5)
    print(f"  delta wins: mean {paired_mean:+.4f}  sd {paired_sd:.4f}  stderr {stderr:.4f}")
    reduction = (spread * (2**0.5)) / paired_sd if paired_sd else float("inf")
    print(f"  -> pairing shrinks the error by {reduction:.1f}x")

    print("\n=== 3. is the effect visible? ===")
    ratio = abs(paired_mean) / paired_sd if paired_sd else float("inf")
    print(f"  |delta| / sd = {ratio:.2f}")
    verdict_ok = ratio >= 2.0
    if verdict_ok:
        print("  PASS -- a realistic swap is several times its own noise. Report a point estimate.")
    else:
        print("  FAIL -- the effect is inside the noise at this sim count.")
        print("  The tool must report an interval, or raise n_sims, or lead with lineup gain.")

    print("\n=== 4. determinism ===")
    twice = [_projected_wins(improved_pool, 0, settings) for _ in range(2)]
    same = twice[0] == twice[1]
    print(f"  same pool, same seed, twice: {twice[0]:.6f} vs {twice[1]:.6f} -> {same}")
    if not same:
        print("  FAIL -- an unseeded source is leaking in; pairing cannot work without this.")

    return 0 if verdict_ok and same else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        help=(
            "how many seeds to average over. Six, because that is what the constants in "
            "`midseason.swap_impact` were measured at -- a different default means a "
            "re-run cannot reproduce the stderr they quote."
        ),
    )
    parser.add_argument(
        "--swap-points",
        type=float,
        default=40.0,
        help="season points the swap adds — roughly a few points a week, like a real add",
    )
    parser.add_argument("--my-team", type=int, default=1)
    args = parser.parse_args(argv)
    return report(
        Settings(
            n_sims=args.n_sims,
            repeats=args.repeats,
            swap_points=args.swap_points,
            my_team=args.my_team,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
