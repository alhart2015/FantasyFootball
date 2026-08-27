# Waiver / free-agent recommender — design

**Status:** design, not yet implemented (§4 measured 2026-08-26)
**Branch:** `feat/waiver-recommender`
**Date:** 2026-08-26
**Issue:** #104 53a (the umbrella issue is closed; this is its last unbuilt feature)

**The question the tool answers:** *is anyone on waivers better than someone on my team?*

The motivating case is an injury. A starter goes down Sunday; Tuesday morning you want to know
who to add and who to drop, before waivers clear. Everything below is shaped by that: the tool
has to run in seconds, has to know who is actually hurt, and has to name a specific
drop-this-add-that pair rather than handing back a ranked list to cross-reference by hand.

---

## 1. What exists, and what does not

**Reusable, do not rebuild:**

| Piece | Where | What it gives us |
|---|---|---|
| Rest-of-season points | `midseason.rest_of_season.remaining_totals` | season projection − points scored, clamped, with diagnostics |
| Season simulation | `midseason.standings.project_league_standings` | projected wins, playoff %, bye %, title % |
| My roster, resolved | `midseason.my_team.build_my_team` | roster + `gsis_id` + YTD + ROS, one team |
| Actuals under our scoring | `scoring.actuals.actual_season_total` | `weekly_stats` × league `Ruleset` |
| **Weekly projections** | `draft.backtest.espn_weekly.refresh_espn_weekly_projections` | per-week ESPN projections, already written to the store for 2021-2025 |
| Positional ranking | `rankings.rank_within_position` | integer ranks, no fractional ties |
| ESPN league pull | `ingest.espn_league.fetch_league_payload` | settings, teams, rosters, schedule |
| A `kona_player_info` fetch | `draft.backtest.espn_weekly._fetch_espn_week` | the request shape, including the filter header |

**Does not exist:**

1. **The free-agent list.** We pull rosters; we have never pulled the players *nobody* rosters.
2. **Injury status.** `parse_rosters` reads `entry.playerPoolEntry.player` and throws
   `injuryStatus` away. Without it the tool recommends nothing the morning after an injury,
   because the model still projects the injured player at full strength — the exact case above.

Those two are the work. The comparison itself is arithmetic over things we already compute.

---

## 2. Ingest — the free-agent pool

New in `ingest/espn_league.py`, beside the other ESPN calls:

```python
def fetch_free_agents(league_id, season, *, creds, scoring_period=None, limit=...) -> dict
def parse_free_agents(payload) -> pd.DataFrame
```

**The league endpoint, not `leaguedefaults`.** `external_projections.py` already hits
`kona_player_info` against `leaguedefaults/3`, which is the generic player universe — it has no
idea who is rostered in *your* league. Free agency is a per-league fact, so this call goes to
the same authenticated league URL the rest of `espn_league.py` uses, with the same cookies.

**Filtered server-side**, via the `X-Fantasy-Filter` header:

```json
{"players": {"filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
             "filterSlotIds": {"value": [0, 2, 4, 6]},
             "limit": 400,
             "sortPercOwned": {"sortPriority": 1, "sortAsc": false}}}
```

Sorted by percent-owned descending, so a truncated response keeps the players anyone would
plausibly add and drops the long tail of never-rostered names. **`limit` is a real cap and the
parse reports when it was hit** — a silent truncation here reads as "there is nobody better
available", which is the one answer this tool must never give wrongly.

`FREEAGENT` and `WAIVERS` are separate ESPN statuses and both are returned, carried through as
an `on_waivers` flag. They differ in how you acquire the player, not in whether he would help,
and a recommendation that ignores the distinction sends you to click "Add" on a player who is
actually on a waiver claim until Wednesday.

Columns: `espn_id`, `player`, `pos`, `nfl_team`, `injury_status`, `percent_owned`,
`on_waivers`.

---

## 3. Ingest — injury status

`parse_rosters` gains `injury_status`; `parse_free_agents` produces the same column from the
same field, so both sides of a swap are described identically.

ESPN reports a string per player: `ACTIVE`, `QUESTIONABLE`, `DOUBTFUL`, `OUT`,
`INJURY_RESERVE`, `SUSPENSION`, `DAY_TO_DAY`. Absent means healthy.

**The values are wrapped in an enum, per CLAUDE.md** — `InjuryStatus` in `schemas.py`, never
the bare strings. An unrecognised status maps to `UNKNOWN` and is treated as healthy, with the
raw value carried so it can be reported rather than silently swallowed. ESPN adds statuses.

---

## 4. What an injury designation actually costs — measured

This was going to be a table of guesses. It is not, because the data to measure it is already
on disk: weekly injury reports (`nfl_data_py.import_injuries`), ESPN weekly projections
(`data/processed/espn_weekly_projections`, 2021-2025), and `weekly_stats` scored under the
league ruleset. Everything below is 2021-2025, QB/RB/WR/TE.

### 4.1 A Questionable starter plays. A Questionable deep-bench player does not.

Play rate given a Questionable designation, split by what ESPN projected him for that week:

| Projected | n | Played |
|---|---|---|
| < 2 pts | 864 | 11.0% |
| 2-5 | 251 | 72.9% |
| 5-10 | 444 | 93.7% |
| 10-15 | 308 | 99.4% |
| 15+ | 94 | 100.0% |

**The headline "54% of Questionable players play" is true and useless.** It is an average over
a population dominated by players who were never going to play, injury or not. Conditioned on
the only players a lineup decision is ever about, Questionable means *plays*.

Any aggregate over injury designations that does not condition on projected volume is measuring
roster churn, not injury.

### 4.2 But he plays worse — by about 14%

Share of ESPN's weekly projection delivered, restricted to players projected for 5+ points:

| | n | Mean delivered | Median |
|---|---|---|---|
| Healthy | 13,960 | 96.5% | 84.6% |
| Questionable | 846 | **83.4%** | 72.7% |

Conditional on playing at all: healthy 100.3%, Questionable 89.2%. So most of the gap is "played
hurt and produced less", not "sat".

**Multiplier: 0.86.**

### 4.3 The confound, checked

If ESPN already discounted a Questionable player's projection, §4.2 would be measuring nothing —
the shortfall would be real points against an already-lowered bar, and applying our own
multiplier on top would double-count.

Test: a player's projection in his Questionable weeks against his own median healthy-week
projection. Result: **100.4%** (n=843). ESPN does not discount Questionable. The 0.86 is ours to
apply.

**It does discount `Out`**: of 1,475 `Out` designations reaching the projection feed, exactly one
carries a projection of 5+ points. ESPN zeroes them. **So a weekly view must not apply an `Out`
multiplier on top of an ESPN weekly projection** — that is a double-discount, and its symptom is
a plausible-looking number.

### 4.4 The two horizons need different adjustments

The finding that reshapes this spec: **a Questionable tag covers one week.**

- **Rest-of-season** (the waiver swap): 0.86 applied to one week out of ten remaining is a ~1.4%
  haircut. It will never flip a drop/add decision. What moves a season-long number is a
  *multi-week* absence — IR, or a long injury.
- **This week** (start/sit, streaming): 0.86 on a single week's projection is a 14% cut, which is
  exactly the size that flips a start/sit call between two close players.

Both are real questions and the tool answers both. Collapsing them into one number was the
mistake this section exists to correct.

### 4.5 The tables, per horizon

**Weekly** — multiply the week's projection:

| Status | Multiplier | Source |
|---|---|---|
| Active / absent / unknown | 1.00 | — |
| Questionable | **0.86** | measured, n=846 |
| Doubtful | 0.04 | measured play rate 0.4%, n=269 |
| Out | 0.00 | measured play rate 0.1%, n=1,661 |

Applied only when the source projection is **not** already injury-aware (§4.3).

**Rest-of-season** — expected games missed:

| Status | Games missed | Source |
|---|---|---|
| Active / day-to-day / unknown | 0 | — |
| Questionable | 0.14 | 1 − 0.86, one week |
| Doubtful | 1 | measured |
| Out | 1 | measured; ESPN re-reports weekly |
| Suspension | 1 | assumed — same shape, known absence, unknown length |
| **Injured Reserve** | **4** | **guess.** NFL minimum. Not measurable from the injury report, which carries game status, not roster designation |

**IR is the only number still guessed, and the one that matters most.** Set to the minimum, which
biases toward keeping the player — dropping someone who returns is worse than holding him a week
too long. §5.3 is how the tool compensates for not knowing.

---

## 5. The objective: change in expected season wins

**One number, and it is not points.**

A move that lifts your odds 20% next week while costing 2.5 expected wins over the rest of the
year is a bad move. One that lifts them 20% at no future cost is free. Points cannot express
that trade — this week's points and the rest of the season's points are different currencies,
and a table with a column of each makes the reader do the conversion by eye.

Expected wins is the conversion. Twenty percent for one week **is** 0.2 expected wins. The
drop's future cost and the add's near-term gain land on the same scale automatically, and the
sign of their sum is the recommendation.

```
Δ wins = E[season wins | roster with the swap] − E[season wins | roster as it stands]
```

Both terms come from `project_league_standings`, which already exists and already returns
projected wins, playoff %, bye % and title %. **The swap is simulated as permanent**, which is
the honest model: dropping a player really is irreversible, and a streamer really does occupy
the spot until the next stream displaces him.

### 5.1 The paired-simulation rule, which is load-bearing

Two independent 2,000-sim runs of the same league differ by simulation noise alone, and that
noise is plausibly the same size as the effect being measured — a stream worth five points in
one week out of eight moves expected wins by something like 0.05.

So: **one baseline, computed once, and every candidate simulated against the same seeded
`rng`.** Common random numbers — the difference between two runs sharing draws is far more
precise than either estimate is on its own.

This is the single most likely way for this tool to produce a confident wrong answer, so it
gets two tests: a no-op swap must report **exactly** 0.0, and the same candidate simulated
twice must report the same number to the bit.

**Whether the effect clears the noise at all is an empirical question**, and it is not settled
by writing it down. Step 6 of the plan measures the standard error of Δ wins across repeated
seeds before any recommendation is printed. If a real stream is not distinguishable from zero
at 2,000 sims, the number of sims goes up or the tool reports an interval rather than a point
estimate. It does not print a false precision.

### 5.2 Two stages, because 50 simulations is minutes

Stage 1 is a filter, not a headline. Stage 2 is the answer.

**Stage 1 — does he crack the lineup at all?** `weekly_lineup_points` (already in
`draft/backtest/lineup.py`) sets the best startable lineup by projection. For each free agent:

```
lineup_gain = best_lineup(roster + FA, week) − best_lineup(roster, week)
```

This is cheap, and in a 16-team league with seven starters and five bench spots it eliminates
most candidates immediately — **a genuinely good free agent who still would not start scores
exactly 0.0, which is the truthful answer most weeks.** It also handles for free the things
that actually drive streaming:

- **Byes.** A player on bye has no weekly projection, so he is unstartable and the hole in the
  lineup appears by itself. `parse_espn_weekly` already returns `None` for these.
- **FLEX.** A WR who beats your RB2 into the flex counts; one who does not, does not.
- **Positional scarcity**, without a hand-written rule about it.

Candidates with `lineup_gain <= 0` never reach stage 2. Nothing that cannot play cannot help.

**Stage 2 — what is it worth in wins?** Survivors (expected: a handful) go through the paired
simulation of §5.1. That is the reported number.

### 5.3 Choosing the drop

**The cheapest player not in the optimal lineup**, by rest-of-season value.

Compute the best lineup *including* the free agent; whoever is left over is droppable; drop the
one with the lowest remaining projection. This never suggests dropping the player the add just
displaced into the lineup, and it makes the cost visible rather than assumed.

An open bench or IR slot is used first and costs nothing — which is why the first thing the
tool should tell you, before any swap, is whether you have a free spot.

### 5.4 What the row says

```
ADD                DROP              LINEUP    Δ WINS   Δ PLAYOFF
Jauan Jennings  →  (IR slot open)     +6.2     +0.11      +2.4%
Tyler Boyd      →  R. Johnson         +4.8     +0.07      +1.6%
Rome Odunze     →  B. Corum           +5.1     −0.19      −4.1%
```

`LINEUP` is stage 1 — this week's starting-lineup gain, kept because it is the thing you can
verify by looking at your own roster, and because a Δ wins figure with no visible mechanism is
hard to trust. `Δ WINS` is the recommendation. When they disagree in sign, the third row above,
the drop is what costs you and the tool says so.

---

## 6. Injury status, and what the tool shows instead of pretending to know

§4 measured what a designation costs. Where it lands:

- **Stage 1** uses the weekly multiplier (§4.5): Questionable × 0.86 on a single week is a 14%
  cut, exactly the size that decides a close start/sit. **The `Out` double-discount guard lives
  here**, with a test — ESPN's weekly feed already zeroes `Out` players, and multiplying that by
  zero again is a plausible-looking wrong number.
- **Stage 2** uses expected games missed (§4.5) inside the rest-of-season projection the
  simulator consumes.

For any player on IR or carrying a multi-week absence, the numeric adjustment is a guess and the
write-up is not. ESPN's public athlete endpoint carries it, no auth:

```
status:       Questionable
type:         Groin · Soreness · 2026-08-24
shortComment: "Nacua (groin) isn't practicing Monday, Sarah Barshop of ESPN.com reports."
longComment:  "...ticking toward two weeks off the practice field... the team likely is being
               as cautious as possible with Week 1 against the 49ers in mind."
```

Body part, severity, a beat reporter's read, and a date so staleness is visible.

**The text is shown, never parsed.** A regex for "four to six weeks" is wrong whenever the
sentence is about practice rather than games, or about a different player — and the error would
land inside a projection, where nobody can see it. The number says what it assumed; the text
lets you overrule it.

**Cost and caching.** One call per player, and the league-wide feed is 403. Fetched only for
players whose fantasy status is not Active, cached by player and date. A handful of calls, not
a hundred.

---

## 7. Where it surfaces

**CLI first**, `scripts/waiver_recommender.py`, because that is what you can use at 7am on a
Tuesday without starting a server, and because it makes the tool testable end to end before any
template exists.

**Then a third dashboard page**, reusing the column registry and table macros the season UI
already has — this is a table of rows with a colour scale, which is exactly what
`views/columns.py` and `_table.html` were built for. The page is a thin add precisely because
that work is done.

---

## 8. Out of scope

- **Acquisition cost.** FAAB bidding and waiver priority are how you *get* the player; this tool
  is about whether you want him. Worth its own pass later.
- **Streaming DST/K.** The pool carries neither, so there is nothing to rank.
- **Multi-player moves.** One drop, one add. Joint swaps are combinatorial and #154's trade
  analyzer is where that argument gets had.
- **Auto-execution.** The tool recommends; it never touches the league.

---

## 9. Plan

1. `InjuryStatus` enum + `parse_rosters` carries it. Smallest change, unblocks the rest.
2. `fetch_free_agents` / `parse_free_agents`, with the truncation report.
3. `scripts/measure_injury_impact.py` — the §4 measurement, kept as code so the tables can be
   re-measured when 2026 adds a season. Produces the constants step 4 consumes.
4. The §4.5 tables and both adjustments, with the `Out` double-discount guard and its test.
5. **Stage 1**: `lineup_gain` over `weekly_lineup_points`, plus the drop rule (§5.3).
6. **Measure the noise floor** (§5.1) before printing any Δ wins: standard error of Δ wins
   across repeated seeds, paired and unpaired, at 2,000 sims. This decides whether stage 2
   reports a point estimate or an interval, and it is a gate on step 7 rather than a nicety.
7. **Stage 2**: the paired simulation, one shared baseline, one seed.
8. The injury write-up fetch (§6), cached, non-Active players only.
9. `scripts/waiver_recommender.py`, run live against league 856974.
10. The dashboard page.

Steps 1-9 give the terminal tool. Step 6 is the one that can change the design: if Δ wins does
not clear its own noise at a tolerable number of sims, stage 1 becomes the headline and stage 2
becomes a range.

## 10. Findings worth keeping regardless of this tool

Recorded here because they outlive the feature and were expensive to establish:

- **Any aggregate over injury designations must condition on projected volume**, or it measures
  roster churn. "54% of Questionable players play" and "Questionable starters always play" are
  both true of the same dataset.
- **ESPN discounts `Out` in its weekly projections and does not discount `Questionable`.**
  Anything consuming that feed needs to know which statuses are already priced in.
- **A Questionable player who plays still delivers ~11% under his projection.** The designation is
  informative even when it does not keep him off the field.
- **The scripts that produced these numbers are worth keeping**, not just their outputs — see
  `scripts/measure_injury_impact.py` (step 3), so the tables can be re-measured when the 2026
  season adds a year of data.
- **Points across different horizons are not a common currency; expected wins is.** This is why
  §5 reports Δ wins rather than a this-week column beside a rest-of-season column. The reader
  should not have to convert by eye, and the conversion rate is not obvious.
