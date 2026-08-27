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

## 5. The recommendation

`src/projections/midseason/waivers.py`, one entry point per horizon.

### 5.1 Rest of season — "should I drop him and add this guy"

```python
def recommend_swaps(run: MyTeamRun, free_agents, *, config, id_map, min_gain=...) -> list[Swap]
```

`MyTeamRun` already carries the roster, the pool with `season_mean_fpts` rewritten to remaining
points, and the week. Free agents are scored through **the same `remaining_totals` call as my
roster** — one code path, so a free agent and a rostered player can never be measured
differently. That is why this takes a `MyTeamRun` rather than re-deriving one.

**Swaps are compared within position**, flex-aware from `config.roster_slots`. A WR who
out-projects my worst RB is not a move I can make.

**The drop candidate is my worst player at that position**, by adjusted rest-of-season points,
from the bench and IR first. It will recommend dropping a starter when that is right, but not
because a bench player sorted oddly.

Each `Swap` reports drop, add, rest-of-season gain, waivers-vs-free-agent, percent rostered, and
— when an injury adjustment drove it — the status and the write-up (§5.3).

### 5.2 This week — "who do I start, and is there a better option on the wire"

```python
def rank_this_week(run: MyTeamRun, free_agents, *, week, config, id_map) -> list[WeeklyRow]
```

Same roster, different number: this week's projection with the §4.5 weekly multiplier. This is
where 0.86 earns its keep — a 14% cut on one week is the size that decides a close start/sit, and
it is the case that motivated measuring any of this.

Weekly projections come from `refresh_espn_weekly_projections`, which already exists and already
writes to the store. **The `Out` double-discount guard (§4.3) lives here**, with a test: an `Out`
player must not be zeroed twice.

### 5.3 What the tool shows instead of pretending to know

For any player on IR or carrying a multi-week absence, the numeric adjustment is a guess (§4.5)
and the write-up is not. ESPN's public athlete endpoint carries it, no auth:

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
land inside a projection, where nobody can see it. The number says what it assumed; the text lets
you overrule it.

**Cost and caching.** One call per player, and the league-wide feed is 403. Fetched only for
players whose fantasy status is not Active, cached by player and date. A handful of calls, not a
hundred.

---

## 6. Wins, for the shortlist only

Points answer "is he better". Wins answer "does this help me win the league", and those come
apart: a bench upgrade at a position I already start two good players at is worth real points and
almost no wins.

So the top `k` swaps by rest-of-season gain (default 3) are re-run through
`project_league_standings` with the swap applied, reporting Δ projected wins, Δ playoff %, Δ
title %. **Only the shortlist**, because each is a full Monte-Carlo season and the point of §5.1
is that it returns in seconds.

The baseline is a single run with no swap, shared across all candidates at the same `rng` seed —
two runs at different seeds differ by simulation noise alone, and at 2000 sims that noise is
comparable to the effect being measured. **Same seed, one baseline, differences only.** This is
the failure mode most likely to produce a confident wrong answer here, so it gets a test
asserting a no-op swap reports exactly zero.

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
3. The §4.5 tables and both adjustments, with the `Out` double-discount guard and its test.
4. `recommend_swaps` — rest of season, positional, min-gain filtered.
5. The injury write-up fetch (§5.3), cached, non-Active players only.
6. `scripts/waiver_recommender.py`, run live against league 856974.
7. `rank_this_week` — the weekly horizon.
8. The wins shortlist: top 3 swaps re-simulated, one shared baseline, one seed.
9. The dashboard page.

Steps 1-6 give a terminal tool that answers the Tuesday-morning question. 7 is the Sunday-morning
one. 8 and 9 are upgrades to both.

---

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
