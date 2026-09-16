"""Issue #188 step 1: does maximising POINTS ever fail to maximise expected RESULTS?

Under `WIN_BONUS_TOP_HALF` a week hands out two results -- the head-to-head game and a rank
against the whole league -- so the quantity a lineup should maximise is

    ER(lineup) = P(beat your opponent) + P(finish in the top half)

and not its projected point total. This script asks the cheap version of that question before
anyone rebuilds the start/sit objective: **across the real league and every remaining week, how
often does the points-optimal lineup differ from the ER-optimal one, and by how much?**

If the two agree essentially always, #188 closes and the objective stays as it is.

    python scripts/_top_half_objective_disagreement.py
    python scripts/_top_half_objective_disagreement.py --n-sims 80000 --weeks 4

Every remaining regular-season week is swept, not just the next one. The rosters barely move
week to week, but **byes and opponents do**, and those are what reshape a lineup: a week with
two starters on bye forces a different set of candidates into the lineup, which is exactly
where an objective change would show up if it showed up anywhere.

Four things it is careful about, because each one would otherwise fake an answer:

- **Lineups are set EX ANTE, for every team.** `team_weekly_points` fills the optimal lineup
  per simulation using that simulation's realised points, which is clairvoyant -- fine for a
  season projection, wrong here. A manager picks one lineup from projections and lives with
  it. Clairvoyant field scores would also inflate the top-half threshold, and where that
  threshold sits relative to your mean is exactly what decides whether variance helps you.
- **The baseline is optimal on expected STARTABLE points**, not on the raw season projection.
  See `best_lineup_containing`; getting this wrong manufactures disagreements out of bye weeks.
- **Common random numbers.** Every candidate lineup is scored on the SAME draws, so the
  difference between two lineups is driven only by the players that differ. The paired
  standard error is reported; an unpaired comparison at this sample size could not resolve the
  effect at all.
- **The bonus is scored by the shipped `top_half_credit`**, not a reimplementation, so a
  disagreement here cannot be an artifact of this script scoring the rule differently from
  the simulator.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.league_projection import top_half_credit
from projections.draft.assistant.performance_variance import (
    SEASON_GAMES,
    VarianceParams,
    sample_weekly_points,
)
from projections.draft.assistant.season_value import _availability_mask, _bye_indices
from projections.draft.league_calendar import LeagueCalendar
from projections.draft.roster_eligibility import choose_starters_with_slots
from projections.ingest.espn_league import (
    EspnLeagueError,
    build_league_config,
    parse_rosters,
    parse_schedule,
    parse_teams,
    pool_name_index,
)
from projections.midseason.context import build_context
from projections.midseason.standings import SlotMap, first_unplayed_week, rosters_to_slots
from projections.midseason.swap_impact import injury_adjusted_pool
from projections.schemas import RosterSlot


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sims", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--weeks", type=int, default=0, help="cap the sweep; 0 = every remaining week")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="positive control: prove ER responds to variance at fixed mean, then exit",
    )
    p.add_argument("--credentials", type=Path, default=Path("configs/espn_credentials.json"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--league-dir", type=Path, default=None)
    p.add_argument("--pool", type=Path, default=None)
    return p.parse_args(argv)


def best_lineup_containing(
    frame: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    on_bye: np.ndarray,
    *,
    force: int | None,
) -> list[int]:
    """Indices of the points-optimal lineup, optionally forced to include row `force`.

    **Selection runs on expected STARTABLE points, not the raw projection.** `season_mean_fpts`
    is a full-season number: a player on bye this week still carries it, and an injury-risky
    one carries all of it rather than the share he is likely to play. The simulation applies a
    bye mask and an availability Bernoulli on top, so ranking on the raw projection builds a
    baseline that starts players who score zero in every single simulation -- and then any
    alternative beats it, for a reason that has nothing to do with the bonus. The `d pts`
    column is what caught that: the "better" lineups had MORE expected points, which is
    impossible against a genuinely points-optimal baseline.

    `weekly_mean * p_week` is exactly `E[realised points]` under the model, since the draw and
    the availability mask are independent. A bye is unstartable rather than zero-valued, which
    is `startable_points`' distinction: a slot nobody else can fill should still take a real
    zero, but a manager never chooses a bye player over a healthy one.

    Forcing is done by value, not by a special case in the greedy: the forced player is handed
    a value nothing can beat, so he takes the most restrictive slot he is eligible for and the
    remaining slots fill optimally around him. That is precisely "the best lineup a manager
    could set if he insists on starting this guy", which is the alternative worth comparing
    against.
    """
    values = frame["expected_startable"].to_numpy(dtype=np.float64).copy()
    benched = on_bye.copy()
    if force is not None:
        values[force] = float("inf")
        benched[force] = False
    positions = frame["position"].astype(str).tolist()
    filled = choose_starters_with_slots(
        list(range(len(frame))),
        roster_slots,
        value=lambda i: None if benched[i] else float(values[i]),
        position=lambda i: positions[i],
    )
    return sorted(index for index, _slot in filled)


class WeekResult:
    """One week's sweep: how many teams disagreed, and the largest real gain."""

    def __init__(self, week: int) -> None:
        self.week = week
        self.rows: list[tuple[str, float, float, float, bool]] = []

    @property
    def disagreements(self) -> int:
        return sum(1 for *_rest, real in self.rows if real)

    @property
    def biggest(self) -> float:
        return max((delta for _n, _er, delta, _se, real in self.rows if real), default=0.0)


def variance_self_test(
    field: np.ndarray,
    field_cols: dict[int, int],
    opponent: dict[int, int],
    ordered: list[int],
    names: Mapping[int, str],
    slots: SlotMap,
) -> None:
    """Positive control: does ER move when only the SPREAD changes, mean held fixed?

    A sweep that reports "no disagreement anywhere" is worthless if the instrument could not
    have detected one. This proves it could. Each team's own simulated total is rescaled about
    its mean -- `mean + (x - mean) * k` -- which leaves E[points] exactly unchanged and
    multiplies the spread by `k`. Any ER difference is therefore attributable to variance alone.

    It also checks the direction the theory predicts, which is the correction #188 needed: more
    variance helps a team that is BELOW the thresholds it is shooting at and hurts one that is
    above. If safe-vs-risky came out the same sign for everybody, the model of the mechanism
    would be wrong and the null result would mean something different.
    """
    print("Positive control: same mean, spread x0.5 vs x1.5. ER must move, and the sign")
    print("must follow whether the team is above or below the league's median score.\n")
    median_score = float(np.median(field))
    header = (
        f"{'TEAM':<26}{'mean pts':>10}{'vs median':>11}"
        f"{'ER safe':>10}{'ER risky':>10}{'risky-safe':>12}"
    )
    print(header)
    print("-" * len(header))
    moved = 0
    for slot in ordered:
        if slot not in opponent:
            continue
        mine = field[:, field_cols[slot]]
        mean = float(mine.mean())
        safe = mean + (mine - mean) * 0.5
        risky = mean + (mine - mean) * 1.5

        def er(total: np.ndarray, slot: int = slot) -> float:
            opp = field[:, field_cols[opponent[slot]]]
            h2h = np.where(total > opp, 1.0, np.where(total == opp, 0.5, 0.0))
            board = field.copy()
            board[:, field_cols[slot]] = total
            return float((h2h + top_half_credit(board)[:, field_cols[slot]]).mean())

        er_safe, er_risky = er(safe), er(risky)
        gap = er_risky - er_safe
        moved += int(abs(gap) > 0.002)
        print(
            f"{str(names.get(slots.team_id(slot), slot))[:25]:<26}{mean:>10.1f}"
            f"{mean - median_score:>+11.1f}{er_safe:>10.4f}{er_risky:>10.4f}{gap:>+12.4f}"
        )
    print()
    print(
        f"{moved} of {len(opponent)} teams moved by more than 0.002 results. A team below the "
        "median gains from variance and one above it loses -- if the instrument were blind to "
        "spread, every gap here would be zero and the sweep's null result would prove nothing."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx_args = argparse.Namespace(
        league_id=None,
        season=None,
        team_id=None,
        pool=args.pool,
        league_dir=args.league_dir,
        credentials=args.credentials,
        data_root=args.data_root,
        week=None,
    )
    try:
        ctx = build_context(ctx_args, require_team_id=False, require_config=False)
    except (ValueError, EspnLeagueError, FileNotFoundError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 1

    payload = ctx.payload
    config = build_league_config(dict(payload))
    settings = payload.get("settings", {}) or {}
    calendar = LeagueCalendar.from_espn_settings(settings.get("scheduleSettings", {}) or {})
    teams = parse_teams(dict(payload))
    schedule = parse_schedule(dict(payload), teams)
    start_week = first_unplayed_week(schedule, calendar)

    pool = injury_adjusted_pool(ctx.pool, payload, ctx.id_map, week=ctx.schedule_week)
    slots = SlotMap.from_team_ids(list(teams["team_id"]))
    by_slot, _dropped = rosters_to_slots(
        parse_rosters(dict(payload)),
        ctx.id_map,
        slots,
        set(pool["gsis_id"].astype(str)),
        name_index=pool_name_index(ctx.pool),
    )
    names = dict(zip(teams["team_id"], teams["team_name"], strict=False))
    availability = ctx.availability()
    params = VarianceParams.load()
    rng = np.random.default_rng(args.seed)
    roster_slots = dict(config.roster_slots)

    sub = pool.set_index("gsis_id")
    frames: dict[int, pd.DataFrame] = {}
    for slot, gsis_ids in by_slot.items():
        frame = sub.loc[gsis_ids].reset_index()
        weekly = frame["season_mean_fpts"].to_numpy(dtype=np.float64) / SEASON_GAMES
        ids = frame["gsis_id"].astype(str).to_numpy()
        p_week = np.array([availability.p_week(g) for g in ids], dtype=np.float64)
        frame["weekly_mean"] = weekly
        # E[realised points]: the draw and the availability mask are independent, so the
        # expectation is their product. This is what a lineup must be chosen on -- see
        # `best_lineup_containing`.
        frame["expected_startable"] = weekly * p_week
        frame["bye_week"] = [availability.bye_week(g) for g in ids]
        frames[slot] = frame

    ordered = sorted(frames)
    everyone = pd.concat([frames[s].assign(_slot=s) for s in ordered], ignore_index=True)
    gsis = everyone["gsis_id"].astype(str).to_numpy()
    p_avail = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)
    offsets: dict[int, int] = {}
    cursor = 0
    for slot in ordered:
        offsets[slot] = cursor
        cursor += len(frames[slot])

    # Column position of each team in the (n_sims, n_teams) score board. Loop-invariant:
    # `ordered` is fixed, so this is built once rather than rebuilt (and re-closed over) weekly.
    field_cols = {s: i for i, s in enumerate(ordered)}

    weeks = [w for w in calendar.reg_week_numbers if w >= start_week]
    if args.weeks > 0:
        weeks = weeks[: args.weeks]

    print(
        f"{config.name} - weeks {weeks[0]}-{weeks[-1]} ({len(weeks)} weeks x {len(frames)} "
        f"teams = {len(weeks) * len(frames)} team-weeks), {args.n_sims:,} sims each.\n"
        f"Lineups set ex ante from projections, for every team. Common random numbers.\n"
        f"ER = P(beat opponent) + P(top {len(frames) // 2} of {len(frames)}), so 0..2.\n"
    )
    header = (
        f"{'WK':>3}  {'disagree':>9}  {'largest dER':>12}  {'cheapest alt':>13}"
        f"  {'its sd shift':>13}  {'widest sd':>10}"
    )
    print(header)
    print("-" * (len(header) + 12))

    results: list[WeekResult] = []
    worst_overall: tuple[float, str] = (0.0, "")
    for week in weeks:
        draws = sample_weekly_points(
            params,
            everyone["position"].astype(str).to_numpy(),
            everyone["season_mean_fpts"].to_numpy(dtype=np.float64),
            everyone["is_rookie"].to_numpy(dtype=bool),
            n_sims=args.n_sims,
            n_weeks=1,
            rng=rng,
        )[:, 0, :]
        bye = _bye_indices(availability, gsis, [week])
        mask = _availability_mask(rng.random((args.n_sims, 1, len(gsis))), p_avail, bye)[:, 0, :]
        points = np.where(mask, draws, 0.0)

        def total_for(slot: int, rows: list[int], pts: np.ndarray = points) -> np.ndarray:
            base = offsets[slot]
            total: np.ndarray = pts[:, [base + r for r in rows]].sum(axis=1)
            return total

        bye_flags = {
            s: np.array([b == week for b in frames[s]["bye_week"]], dtype=bool) for s in ordered
        }
        baseline_rows = {
            s: best_lineup_containing(frames[s], roster_slots, bye_flags[s], force=None)
            for s in ordered
        }
        field = np.stack([total_for(s, baseline_rows[s]) for s in ordered], axis=1)

        opponent: dict[int, int] = {}
        for row in schedule[schedule["week"] == week].itertuples():
            home, away = slots.slot(int(row.home_team_id)), slots.slot(int(row.away_team_id))
            opponent[home], opponent[away] = away, home
        if not opponent:
            continue

        def expected_results(
            slot: int,
            mine: np.ndarray,
            board_base: np.ndarray = field,
            opp_map: Mapping[int, int] = opponent,
        ) -> np.ndarray:
            """Per-sim results won by `mine`: head-to-head (tie = 0.5) + top-half credit."""
            opp = board_base[:, field_cols[opp_map[slot]]]
            h2h = np.where(mine > opp, 1.0, np.where(mine == opp, 0.5, 0.0))
            board = board_base.copy()
            board[:, field_cols[slot]] = mine
            results: np.ndarray = h2h + top_half_credit(board)[:, field_cols[slot]]
            return results

        if args.self_test:
            variance_self_test(field, field_cols, opponent, ordered, names, slots)
            return 0

        week_result = WeekResult(week)
        worst_dpts = 0.0
        week_levers: list[tuple[float, float, float]] = []
        for slot in ordered:
            if slot not in opponent:
                continue
            frame = frames[slot]
            base_rows = baseline_rows[slot]
            base_total = total_for(slot, base_rows)
            base_er = expected_results(slot, base_total)

            best_delta, best_se, best_dpts = 0.0, 0.0, 0.0
            # The LEVER, measured separately from the verdict. The positive control shows ER
            # responds strongly to spread, so a null sweep only means something once we know
            # how much spread a real lineup swap can actually buy. `cheapest` is the
            # alternative a manager would most plausibly take -- the one costing the fewest
            # expected points -- and `sd_shift` is what it does to the team's weekly spread.
            base_sd = float(base_total.std())
            cheapest_cost, cheapest_sd_shift = 0.0, 0.0
            widest_sd_shift = 0.0
            starters = set(base_rows)
            for candidate in range(len(frame)):
                if candidate in starters or bye_flags[slot][candidate]:
                    continue
                alt_rows = best_lineup_containing(
                    frame, roster_slots, bye_flags[slot], force=candidate
                )
                if alt_rows == base_rows:
                    continue
                alt_total = total_for(slot, alt_rows)
                paired = expected_results(slot, alt_total) - base_er
                delta = float(paired.mean())
                cost = float((alt_total - base_total).mean())
                sd_shift = float(alt_total.std()) / base_sd - 1.0 if base_sd > 0 else 0.0
                if cost > cheapest_cost or cheapest_cost == 0.0:
                    cheapest_cost, cheapest_sd_shift = cost, sd_shift
                if abs(sd_shift) > abs(widest_sd_shift):
                    widest_sd_shift = sd_shift
                if delta > best_delta:
                    best_delta = delta
                    best_se = float(paired.std(ddof=1) / np.sqrt(args.n_sims))
                    best_dpts = cost

            real = best_delta > 2.0 * best_se and best_delta > 0.0
            name = str(names.get(slots.team_id(slot), slot))
            week_result.rows.append((name, float(base_er.mean()), best_delta, best_se, real))
            worst_dpts = min(worst_dpts, best_dpts)
            week_levers.append((cheapest_cost, cheapest_sd_shift, widest_sd_shift))
            if real and best_delta > worst_overall[0]:
                worst_overall = (best_delta, f"{name}, week {week}")

        results.append(week_result)
        note = ""
        if week_result.disagreements:
            flagged = [n for n, _er, _d, _se, real in week_result.rows if real]
            note = ", ".join(flagged[:3]) + (" ..." if len(flagged) > 3 else "")
        cheap_cost = min((c for c, _s, _w in week_levers), default=0.0)
        cheap_sd = max((abs(sd) for _c, sd, _w in week_levers), default=0.0)
        widest = max((abs(w) for _c, _s, w in week_levers), default=0.0)
        print(
            f"{week:>3}  {week_result.disagreements:>4}/{len(week_result.rows):<4}  "
            f"{week_result.biggest:>+12.4f}  {cheap_cost:>13.2f}  {cheap_sd:>12.1%}  "
            f"{widest:>9.1%}  {note}"
        )

    team_weeks = sum(len(r.rows) for r in results)
    total_disagreements = sum(r.disagreements for r in results)
    print()
    print(
        f"{total_disagreements} of {team_weeks} team-weeks "
        f"({total_disagreements / team_weeks:.1%}) have an alternative lineup that beats the "
        f"points-optimal one on expected results by more than 2 standard errors."
    )
    if total_disagreements:
        print(f"Largest real gain: {worst_overall[0]:+.4f} results/week ({worst_overall[1]}).")
        print(
            f"Over a {calendar.reg_weeks}-week season that is at most "
            f"{worst_overall[0] * calendar.reg_weeks:+.2f} results, against the "
            f"{2 * calendar.reg_weeks} the season decides."
        )
    else:
        print(
            "Maximising points and maximising expected results pick the same lineup everywhere "
            "measured. The objective does not need to change."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
