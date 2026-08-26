# Projected standings and matchup odds — design

**Status:** design, not yet implemented
**Issue:** [#104](https://github.com/alhart2015/FantasyFootball/issues/104) (53b + 53c)
**Branch:** `feat/projected-standings`
**Date:** 2026-08-25

Mid-season answer to "where am I actually going to finish?" — expected wins, playoff odds,
bye odds, title odds, per-team; plus P(win) on each remaining matchup. Weekly snapshots
persist so the trajectory can be plotted across the season.

---

## 1. What this is, and what it is not

**Is:** a Monte-Carlo run over *this league's real remaining schedule*, from *current
rosters*, with *already-played weeks locked to their actual results*.

**Is not:** a projection-accuracy claim. Like `project_draft`, every team is scored under our
own projections. It measures where the rosters and the schedule point, not whether the
projections are right about 2026.

**Is not:** a waiver or trade tool (#104 53a, #154). Those need the free-agent pool and a
value model; this needs neither.

## 2. Why the existing engine is close but not sufficient

`draft/assistant/league_projection.py` already does the hard part: `simulate_seasons` draws
weekly points per player under the availability + variance models, sets optimal lineups,
plays a season, seeds, runs the bracket, and reports per-seat rates. `LeagueCalendar` (merged
in #155) made the week count and bracket configurable, which was the prerequisite.

Three things it does that are wrong for in-season use:

**2.1 It invents the schedule.** `gauntlet_schedule` is a synthetic round-robin — a
1-factorization of the complete graph. Preseason that is the right choice: it is
strength-of-schedule neutral, so no seat is advantaged by a fixture list that does not exist
yet. **In-season it is the wrong choice, and not by a little.** Who you actually play in
weeks 7–14 is a first-order driver of your record, and ESPN reports it. Probed live on league
856974: `mMatchup` returns 112 entries, 8 per week for weeks 1–14, each carrying `home.teamId`
/ `away.teamId`. That is the real fixture list and it must replace the gauntlet.

**2.2 It has no notion of a week already played.** Every week is simulated. In week 7 the
first six weeks are *facts* — real points, real wins — and simulating them throws away
information and produces a distribution around a record you already know.

**2.3 It projects from preseason season-totals.** `season_mean_fpts` in the VORP pool is a
preseason number. By week 6 it is stale in the way that matters most: it does not know who
tore an ACL and who broke out.

## 3. Design

### 3.1 Rest-of-season player points

**`ros_points = fresh_season_projection − points_scored_to_date`.**

Re-pull `external_projections` in-season (providers revise season totals weekly, so a fresh
pull already reflects injuries, benchings and depth-chart moves), run it through the existing
consensus → season-projection path, and subtract each player's actual points to date.

Chosen over the alternatives because it reuses the ingest path unchanged and is the only
option that reacts to what has actually happened. Prorating the preseason number
(`total × weeks_left / 14`) is blind to the season; a pace/preseason blend is more accurate in
principle but introduces a free weighting parameter that deserves its own backtest, and is
better as a follow-up once this exists to compare against.

**The known risk, stated plainly: this cannot be verified until Week 1 has happened.** The
subtraction assumes ESPN's in-season "season total" is *full-season including games already
played*. If it turns out to be rest-of-season already, subtracting actuals double-counts and
every projection comes out low. Mitigations, both required:

- **A sanity guard.** `ros_points` below zero, or implausibly small for a healthy starter
  early in the season, indicates the assumption is wrong. Clamp at zero, **log loudly with the
  count of affected players**, and never fail silently — a quietly-zeroed roster looks like a
  bad team rather than a bad ingest.
- **A documented switch.** The subtraction lives behind one function with the assumption in
  its docstring, so flipping to "the provider already gives ROS" is a one-line change, not an
  archaeology exercise.

Players with no fresh projection (rookies with synthetic `99-` ids, mid-season pickups) fall
back to the preseason pool value, prorated, and are counted in the same warning.

### 3.2 Locking played weeks

Weeks `1..current_week-1` contribute their **actual** points and **actual** W/L from the
`mMatchup` payload — no simulation. Weeks `current_week..reg_weeks` are simulated over the
real fixture list. Seeding then runs on (actual wins + simulated wins, actual PF + simulated
PF), which is exactly ESPN's rule and already what `simulate_seasons` implements.

This is the whole reason §3.1's staleness matters less than it looks: early in the season most
weeks are simulated and the projection dominates; late in the season most weeks are locked and
the projection barely moves the answer.

### 3.3 Matchup odds fall out for free

53c is not a second engine. Each simulated week already produces both teams' point totals per
simulation, so `P(I beat you in week 8)` is the fraction of simulations where my week-8 total
exceeds yours. Computed from the same run, reported per remaining matchup.

### 3.4 Weekly snapshots

One partition per `(season, week)` under `data/processed/projected_standings/`, written via
`store.write_partition` (the only sanctioned parquet path). One row per team per snapshot:
team id, name, actual W-L-T to date, points for, projected final wins, playoff / bye / title
percentages, mean seed.

Persisting per-week is what makes the trend line possible — "my playoff odds across the
season" is a read of the accumulated partitions, not a separate computation. It produces
nothing visible until several weeks have accumulated, which is expected and worth the schema.

New `ProjectedStandingsSchema` in `schemas.py`, validated with reassignment at the module
boundary per the repo convention.

## 4. Components

| Piece | Where | Notes |
| --- | --- | --- |
| `mMatchup` in the ESPN client | `ingest/espn_league.py` | Add to `DEFAULT_VIEWS`; `parse_schedule` → (week, home_team_id, away_team_id, home_points, away_points, winner) |
| Current rosters | `ingest/espn_league.py` | `parse_rosters` exists |
| `LeagueCalendar.from_espn_settings` | `draft/league_calendar.py` | exists (#155) |
| ROS projection | new | §3.1, one documented function |
| In-season sim | `draft/assistant/league_projection.py` | `simulate_seasons` gains a real schedule and locked weeks |
| Schema + store | `schemas.py`, `store` | §3.4 |
| CLI | `scripts/projected_standings.py` | report + snapshot write |

## 5. Open questions

- **Does ESPN's mid-season season-total fold in actuals?** §3.1. Unanswerable until Week 1;
  guarded rather than assumed.
- **Ties.** ESPN reports a `ties` field. The current sim breaks every matchup with `>=`, so
  ties are impossible in simulated weeks but *can* appear in locked ones. The seeding
  arithmetic needs to carry them.
- **Divisions.** `mTeam` reports `divisionId`. If a league seeds by division the bracket is
  not pure best-record. Critts appears not to; a league that does needs handling before this
  is correct for it.
- **Median scoring / other formats.** Out of scope; noted so it is not assumed away.

## 6. Testing

- **Real schedule replaces gauntlet:** a synthetic 4-team schedule where team A plays the
  weakest opponent every week must give A more expected wins than the gauntlet does. If
  strength of schedule does not move the number, the schedule is not being used.
- **Locked weeks are locked:** with all weeks played, every team's projected wins equal its
  actual wins exactly and playoff odds are 0 or 1. No distribution, no noise.
- **Half-locked:** week 8 of 14, a team 7-0 must project above one that is 0-7 with identical
  rosters — the difference is purely banked wins.
- **ROS subtraction:** a player who has scored 100 of a projected 200 has 100 remaining; a
  player who has *outscored* his projection clamps to zero and is counted in the warning.
- **Matchup odds are consistent with the run:** `P(A beats B)` and `P(B beats A)` sum to 1
  across the same simulations.
- **Snapshot round-trip:** written partition reads back and validates against the schema.

## 7. Plan of attack

1. `mMatchup` ingest + `parse_schedule`, with tests over a synthetic payload.
2. Real-schedule + locked-weeks support in `simulate_seasons` (default stays the gauntlet, so
   every existing caller is unaffected — same discipline as #155).
3. ROS projection with its guard.
4. Schema + store + CLI + snapshot write.
5. Matchup odds read-out.

Each step is its own commit; steps 1–2 are independently useful and testable.
