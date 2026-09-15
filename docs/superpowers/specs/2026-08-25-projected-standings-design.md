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

### 3.5 Scoring enhancements: the top-half bonus win (added 2026-09-15, issue #185)

ESPN reports `settings.scoringSettings.scoringEnhancementType`. The Critts league carries
`WIN_BONUS_TOP_HALF`, confirmed with the commissioner as intentional. Under it **every week
hands each team two results**: the head-to-head game, and a second win (loss) for finishing in
the top (bottom) half of that week's scores league-wide. Sixteen teams, eight matchups,
thirty-two games a week.

Nothing in the repo read the setting, so §3.2's locked weeks and the simulated remainder were
both built on half the games the league actually decides. That is not a fixable-by-offset
error. It is wrong in two independent ways:

- **Banked records.** 14 of 16 week-1 records were wrong. The two that happened to be right
  were right by coincidence — a bonus loss cancelling a head-to-head win.
- **Calibration.** Doubling the games per season roughly halves the spread of final records,
  and playoff / bye / title percentages are read straight off that spread. Simulating a bonus
  league as plain head-to-head reports odds drawn from half the sample the league will produce.

**The setting is carried as an enum on `LeagueConfig`, not a bool and not a bare string.**
`ScoringEnhancement` lives in `schemas.py` beside every other canonical type. `NONE` covers
both "explicitly none" and "key absent", which is every league written before this and every
older ESPN league.

**An unrecognised enhancement type raises.** This is deliberately unlike every other unmappable
ESPN setting in `espn_league.py`, which degrade and log — the right call for a scoring category
`Ruleset` cannot model, because bricking the whole config over one category is worse than
dropping it. It is the wrong call here. An enhancement changes how many results a week decides,
so a silent fallback to plain head-to-head yields a complete, confident standings table built
on the wrong game count: wrong everywhere, wrong-looking nowhere.

**One rule, applied in two places, from one source.** `team_records` credits the bonus for
played weeks; `simulate_seasons` credits it for simulated ones, reading the enum off the
`league_config` it already takes. `project_league_standings` passes the same derived config to
both, so the banked record and the projection beside it can never describe different leagues.

Two boundary rules, both load-bearing:

- **Only fully-played weeks bank a bonus.** The top half of the four teams who have finished
  is not the top half of the league, and banking a partial week means re-deciding it on Monday
  night. `through_week` already excludes partial weeks on the in-season path; the guard is what
  makes the unbounded call safe too.
- **Ties share the contested places rather than breaking arbitrarily.** A team with `above`
  teams strictly ahead and `equal` teams on its exact score occupies places
  `above .. above+equal-1`; its credit is the fraction of those inside the top half. In the
  simulator that is a fractional win; in `team_records`, where the columns are integers, a
  straddling tie is recorded as a **tie**, which `LockedRecord.credited_wins` already scores as
  half a win — the same number. This is not pedantry: `sample_weekly_points` returns exactly
  0.0 for a non-positive projection, so an unresolved roster produces identical all-zero columns
  for every team, and an argsort there would hand the bonus to the eight lowest slot numbers
  and report it as a finding.

A team on a bye in an odd-sized league is simply not in that week's ranking; the half is taken
over the teams that played.

**Calibration re-checked afterwards, as the issue asked.** Seed-to-seed swing in
`make_playoffs_pct` on the live 16-team league, five seeds per setting:

| `--n-sims` | worst-case swing | mean per-team sd |
| --- | --- | --- |
| 200 | 9.5 pp | 2.4 pp |
| 800 | 6.9 pp | 1.4 pp |
| 3000 | 2.7 pp | 0.6 pp |

The 2000 default (CLI and web) sits between the last two rows, so worst-case noise is roughly
3 pp — comfortably below the differences the table is read for. **The default is left alone**;
doubling the games did not push it out of range.

What the bug was costing, same run, bonus versus plain head-to-head at 3000 sims: the largest
movers are exactly the teams that lose a head-to-head while outscoring half the league —
Easy Breecey Beautiful +7.3 pp, Gibbs in a blanket +6.3 pp — against Triple Threat Tracy
−4.9 pp and Certified Beautys −4.2 pp, who win low-scoring games.

**Start/sit and waiver strategy under a top-half bonus is a separate question.** Raw weekly
points matter more and beating your specific opponent matters less, which argues for weighting
upside over floor. That is a strategy change, not this bug; filed separately rather than
smuggled in here.

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
- **Median scoring / other formats.** Partly answered: ESPN's `WIN_BONUS_TOP_HALF` is
  modelled as of #185 (§3.5), and any other `scoringEnhancementType` now raises rather than
  being assumed away. Median scoring proper — a phantom matchup against the league median,
  which is a different rule from a top-half bonus — is still out of scope.

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
- **Top-half bonus reproduces ESPN (§3.5):** all 16 Critts week-1 records, pinned against the
  numbers ESPN published rather than numbers this repo computed. Plus: the bonus is off by
  default, a partially-played week banks none of it, a bye is excluded from the ranking, a tie
  across the cutoff is a tie, the simulator hands out exactly `n // 2` bonus wins a week, and a
  `NONE` league is bit-identical to the pre-#185 simulator.

## 7. Plan of attack

1. `mMatchup` ingest + `parse_schedule`, with tests over a synthetic payload.
2. Real-schedule + locked-weeks support in `simulate_seasons` (default stays the gauntlet, so
   every existing caller is unaffected — same discipline as #155).
3. ROS projection with its guard.
4. Schema + store + CLI + snapshot write.
5. Matchup odds read-out.

Each step is its own commit; steps 1–2 are independently useful and testable.
