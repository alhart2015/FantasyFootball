# Waiver / free-agent recommender — design

**Status:** design, not yet implemented
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

## 4. The modelling call: turning a status into points

This is the one genuinely new piece of modelling, and it has a free parameter, so it is stated
here rather than buried.

A rest-of-season projection assumes a healthy player. An injured player will miss some games.
So:

```
adjusted_ros = ros_points × (games_remaining − expected_games_missed) / games_remaining
```

`expected_games_missed` per status, as a v1 table:

| Status | Games missed | Why |
|---|---|---|
| `ACTIVE` / absent / `UNKNOWN` | 0 | healthy, or we cannot tell |
| `DAY_TO_DAY` | 0 | practice designation, plays most weeks |
| `QUESTIONABLE` | 0.5 | genuinely a coin flip on game day |
| `DOUBTFUL` | 1 | almost never plays |
| `OUT` | 1 | ruled out for this week only — ESPN re-reports weekly |
| `SUSPENSION` | 1 | same shape: a known absence with an unknown length |
| `INJURY_RESERVE` | 4 | the NFL minimum for a return from IR |

**Every one of these is a guess, and the table is the only place they live.** The IR number is
the most consequential and the least certain: a player can return at four weeks or be gone for
the year, and ESPN's status does not distinguish them. Four is the floor, which biases the tool
toward *keeping* an IR'd player — the conservative direction, since dropping a returning starter
is worse than holding him a week too long.

**What this deliberately does not do** is re-model availability. `availability.py` already
learns per-player injury *rates* from prior seasons and is already applied inside the
simulator. This adjustment is about a specific, known, current absence, which historical rates
cannot see. The two compose: the model says "this player misses 15% of games in a typical
season", the status says "and he is definitely missing this one".

**The tool reports the adjustment rather than applying it silently.** A recommendation that
only exists because of an injury discount says so, so you can disagree with the table.

---

## 5. The recommendation — points first

`src/projections/midseason/waivers.py`:

```python
def recommend_swaps(run: MyTeamRun, free_agents: pd.DataFrame, *, config, id_map, min_gain=...)
    -> list[Swap]
```

`MyTeamRun` already carries the roster, the pool with `season_mean_fpts` rewritten to remaining
points, and the week. The free agents are scored through **the same `remaining_totals` call as
my roster** — one code path, so a free agent and a rostered player are never measured
differently. That is the whole reason this takes a `MyTeamRun` rather than re-deriving.

**Swaps are compared within position.** A WR who out-projects my worst RB is not a swap I can
make without breaking my lineup, and a v1 that ignores positional slots produces confident
nonsense. FLEX is handled by treating RB/WR/TE as mutually comparable *for flex-eligible drop
candidates only*, which is a rule the league config already states.

**The drop candidate is my worst player at that position**, by adjusted ROS, and only from the
bench and IR unless the free agent beats a starter outright. Recommending you drop a starter is
a real recommendation, and the tool should make it when it is right — but it should not make it
by accident because a bench player happened to sort oddly.

Each `Swap` reports:

- who to drop and who to add, both by name
- **ROS gain** — the number the recommendation is made on
- whether the add is on waivers or a straight free agent
- percent rostered, as a proxy for "will this claim actually clear"
- **why**, when an injury discount drove it, naming the status

Output is sorted by ROS gain, with `min_gain` filtering the noise: two points over a rest of a
season is not a roster move, and a list that includes it trains you to ignore the list.

---

## 6. The recommendation — wins, for the shortlist only

Points answer "is he better". Wins answer "does this help me win the league", and those come
apart: a bench upgrade at a position I already start two good players at is worth real points
and almost no wins.

So the top `k` swaps by ROS gain (default 3) are re-run through
`project_league_standings` with the swap applied, reporting Δ projected wins, Δ playoff %,
Δ title %. **Only the shortlist**, because each one is a full Monte-Carlo season and the whole
point of §5 is that it returns in seconds.

The baseline is a single run with no swap, shared across all candidates and the same
`rng` seed — two runs at different seeds differ by simulation noise alone, and at 2000 sims
that noise is comparable to the effect being measured. **Same seed, one baseline, differences
only.** This is the failure mode most likely to produce a confident wrong answer here, so it
gets a test that asserts a no-op swap reports exactly zero.

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

- **Acquisition cost.** FAAB bidding and waiver priority are how you *get* the player; this
  tool is about whether you want him. Worth its own pass later.
- **Streaming.** "Best DST/K this week" is a weekly-projection question, not a
  rest-of-season one, and the pool does not carry kickers or defenses at all.
- **Multi-player moves.** One drop, one add. Joint swaps are a combinatorial problem and
  #154's trade analyzer is the place that argument gets had.
- **Auto-execution.** The tool recommends; it never touches the league.

---

## 9. Plan

1. `InjuryStatus` enum + `parse_rosters` carries it. Smallest change, unblocks everything.
2. `fetch_free_agents` / `parse_free_agents`, with the truncation report.
3. The injury adjustment table and `adjusted_ros`, with the reporting.
4. `recommend_swaps` — points only, positional, min-gain filtered.
5. `scripts/waiver_recommender.py`, run live against league 856974.
6. The wins shortlist, shared baseline, same seed.
7. The dashboard page.

Steps 1–5 are independently useful: a points-only recommender you can run from the terminal is
the thing that answers the Tuesday-morning question. 6 and 7 are upgrades to it.
