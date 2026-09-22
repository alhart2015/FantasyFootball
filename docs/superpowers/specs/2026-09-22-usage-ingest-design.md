# Usage & Availability Ingest — Design

**Date:** 2026-09-22
**Sub-project:** projections-core
**Status:** proposed

## Problem

The projection stack has exactly two external opinions in it — ESPN and Sleeper
(`ingest/external_projections.py`, `ingest/sleeper_weekly_projections.py`). Two market-consensus
feeds that largely agree with each other is not an edge; it is the market. Everything the
consensus blend can currently say, a public site says for free.

What the stack does *not* have is any signal that separates **what a player earned** from **what a
player got**. A 3-catch, 40-yard game that ends in a 60-yard catch-and-run touchdown scores the
same as a 9-target, 95-yard workhorse game. Our weekly stats table records both as ~14 points and
the feature builders regress on that number. Volume-based luck regression is the single
cheapest-to-acquire, highest-signal feature class we are missing, and issue
[#121](https://github.com/alhart2015/FantasyFootball/issues/121) already names it as a leverage
point for the elite-season under-projection problem.

Separately, [#119](https://github.com/alhart2015/FantasyFootball/issues/119) has been open since
July for injury-report ingest. Start/sit and waiver tooling currently has no structured view of
who is Questionable or who missed practice.

## What this buys us that we don't already have

`nflreadpy` is already a direct dependency and already vendored into the ingest layer
(`weekly_stats`, `snap_counts`, `depth_charts`, `ngs`, `pbp`, `schedules`, `draft_picks`,
`id_map`). We use 7 of its 26 loaders. Three unused ones are directly relevant, need no new
credential, no new dependency, and no new vendor relationship:

| Loader | Grain | Why it matters |
|---|---|---|
| `load_ff_opportunity` | player × week | **Expected fantasy points** from a play-by-play model — points a player *should* have scored given their opportunities. The `*_exp` / `*_diff` columns are a direct luck-regression signal. |
| `load_pfr_advstats` | player × game | Yards before/after contact, broken tackles, pressure rate, drop rate. Separates back-quality from blocking, and receiver-quality from scheme. |
| `load_injuries` | player × week | Structured practice participation and game-status report. Closes #119. |

All three are CC-BY-SA licensed nflverse releases; `pfr_advstats` and `ff_opportunity` require
attribution to PFR and the `ffopportunity` package respectively, already covered by our existing
nflverse attribution.

## Scope

Three new ingest sources, each following the established pattern
(`CONTRIBUTING.md` → "Adding a new ingest source"), each with a pandera schema, each registered in
`INGEST_SOURCES`. **Ingest only.** No feature builders, no model changes, no consumption in
projections — those are separate work, gated on this data landing and being inspected.

Explicitly out of scope: Vegas player props (option B, a separate spec), FantasyPros,
`load_ftn_charting`, `load_participation`, and `load_ff_rankings`.

## Upstream shapes (probed against 2025, 2026-09-22)

### `load_ff_opportunity(seasons=[s], stat_type="weekly")`

6,054 rows × **159 columns** for 2025. Column families are `{pass,rec,rush}_<stat>` crossed with
`{"", _exp, _diff}`, plus `total_*` rollups, plus a `*_team` mirror of every one of those.

Four traps, all confirmed by probe:

1. **`season` comes back as `String`, `week` as `Float64`.** Every other ingest source in this repo
   gets int32/int64 from upstream. A naive `astype("int64")` on the season string works, but the
   float week does not survive a `Series[int]` schema without an explicit cast.
2. **`player_id` contains nulls.** Team-level aggregate rows carry no player. These must be dropped
   before the gsis pattern check, not after.
3. **`position` includes `DB`, `LB`, `DL`, `OL`, `P`, and `None`.** The `ffopportunity` model covers
   every player who touched the ball, including a punter on a fake and a lineman on a fumble
   recovery. Filter to the `Position` enum before validation.
4. **Weeks run to 22** — playoffs are included. Consistent with `SnapCountsSchema` (`le=22`) we
   store them and leave the playoff filter to the consumer, but note
   [#123](https://github.com/alhart2015/FantasyFootball/issues/123) is open on exactly this seam.

Team column is `posteam`, not `team`.

**Column selection.** Storing all 159 columns would be the largest table in `raw/` by column count
and most of it is redundant (`_diff` is definitionally `actual - exp`, and the `*_team` mirror is a
groupby away). We store the per-player actual and expected for the stats our scoring layer
consumes, and derive nothing:

```
gsis_id, season, week, team, position,
pass_attempt, rec_attempt, rush_attempt,
pass_completions, pass_completions_exp,
receptions, receptions_exp,
pass_yards_gained, pass_yards_gained_exp,
rec_yards_gained, rec_yards_gained_exp,
rush_yards_gained, rush_yards_gained_exp,
pass_touchdown, pass_touchdown_exp,
rec_touchdown, rec_touchdown_exp,
rush_touchdown, rush_touchdown_exp,
pass_interception, pass_interception_exp,
total_fantasy_points, total_fantasy_points_exp
```

`_diff` columns are omitted deliberately: they are a subtraction, and storing a derived column
invites it drifting from its inputs. `*_team` columns are omitted: a `groupby(["season","week",
"team"])` reproduces them exactly, and `posteam` is retained so that groupby is possible.

`total_fantasy_points` here is **the `ffopportunity` model's own scoring, not ours.** It is stored
for diagnostic comparison only. Per the repo convention that
`src/projections/scoring/` is the only place that knows what a fantasy point is, no downstream
consumer may treat this column as a projection — the `_exp` *stat* columns are the real product,
and our scoring layer converts them.

### `load_pfr_advstats(seasons=[s], stat_type=..., summary_level="week")`

Four stat types with four different shapes, all keyed on **`pfr_player_id`, not `gsis_id`**:

| `stat_type` | 2025 rows | Payload |
|---|---|---|
| `pass` | 684 | bad throws, times blitzed/hurried/hit/pressured, sacks taken |
| `rush` | 2,355 | carries, yards before/after contact (+avg), broken tackles |
| `rec` | 4,533 | drops, drop pct, receiving int, receiving rating |
| `def` | 7,926 | coverage targets/completions/yards allowed, missed tackles, pressures |

The id resolution is the same problem `snap_counts` already solved: inner-join `pfr_player_id`
against `id_map.pfr_id`. That helper is currently private to `snap_counts.py`
(`_resolve_gsis_via_id_map`). Rather than copy it a second time — and a third, once a fourth source
needs it — **it moves to `ingest/identity.py`** as a shared, tested helper and `snap_counts`
imports it. `identity.py` is already the home for id-hygiene concerns
(`drop_placeholder_gsis_rows`, `placeholder_name_key`), so this is where a reader would look.

**We ingest `pass`, `rush`, and `rec` in this work and defer `def`.** The offensive three map onto
positions the projection core already models. The `def` table is individual-defender data; it is
the right raw material for a real D/ST model, which is a design question (team aggregation,
opponent adjustment, matchup) large enough to deserve its own spec. Deferring it is recorded as a
follow-up issue rather than dropped — see "Follow-ups".

One schema per stat type, mirroring how `ngs` already splits into
`NgsPassingSchema` / `NgsRushingSchema` / `NgsReceivingSchema`, and one partition per
`(stat_type, season)`, mirroring `ngs_*`'s registry entries.

Note `rush` and `rec` both carry `rushing_broken_tackles` and `receiving_broken_tackles`, and
`pass` and `rec` both carry `passing_drops`/`receiving_drop`. The overlap is upstream's, not ours;
each table keeps only the columns that are meaningful at its own grain.

### `load_injuries(seasons=[s])`

6,068 rows × 16 columns, and the cleanest of the three: a real `gsis_id`, `season` as `Int32`,
`week` as `Int32`, no placeholder-id problem observed.

Two categorical columns, both nullable, both with verbose upstream labels:

- `report_status` ∈ {`Questionable`, `Doubtful`, `Out`, `None`}. `None` means "on the report but
  without a game designation" — typically a Wednesday/Thursday entry — and is **not** the same as
  "healthy". Players not on the report at all have no row.
- `practice_status` ∈ {`Full Participation in Practice`, `Limited Participation in Practice`,
  `Did Not Participate In Practice`, `None`}.

These become two enums in `schemas.py` — `InjuryStatus` and `PracticeStatus` — per the repo rule
that we reference enums, never the strings they wrap. The verbose practice labels are mapped to
`FULL` / `LIMITED` / `DNP`; storing the upstream sentence in a parquet column and then string-
matching on it downstream is exactly the pattern the enum convention exists to prevent.

`season_type` includes `POST`. Unlike `ff_opportunity`, here the postseason rows are genuinely
mixed in with a `game_type` column, and the `week` numbering restarts — so we **keep** `REG` only
and record the filter, since a Week 1 `POST` row colliding with a Week 1 `REG` row would corrupt
any join on `(gsis_id, season, week)`.

Nullability: `report_status` and the injury-description columns are `nullable=True` in the schema.
This is the first ingest table in the repo where a null is *meaningful data* rather than a gap, so
the schema docstring says so explicitly.

## Design decisions

1. **Three separate sources, not one "usage" table.** Different grains (player×week vs
   player×game), different id systems (gsis vs pfr), different season coverage. Joining them at
   ingest would force the union of their failure modes onto every one of them. Join at the feature
   layer where the grain is chosen deliberately.

2. **Store stats, never fantasy points.** `ff_opportunity` ships its own scoring; we store it as a
   diagnostic and route the `_exp` stat columns through `src/projections/scoring/` like every other
   projection. This is the existing repo rule, and `ff_opportunity` is precisely the kind of source
   that tempts a shortcut past it.

3. **The pfr→gsis crosswalk moves to `identity.py`.** Second caller is the point at which a private
   helper becomes shared infrastructure. `snap_counts` is refactored to import it; behaviour is
   unchanged and its existing tests are the regression guard.

4. **`needs_games_played=True` for all three.** None have rows before kickoff. `ff_opportunity`
   in particular is modelled off play-by-play and publishes on the same lag as `pbp`.

5. **`heavy=False` for all three.** The largest (`ff_opportunity` at 6k rows × 28 kept columns) is
   an order of magnitude smaller than `pbp`.

## Verification

Per-source unit tests over synthetic frames, following `tests/` conventions for the existing ingest
modules — the network call stays behind a `_fetch_raw_*` seam the tests monkey-patch. Each source's
tests must cover its specific trap:

- `ff_opportunity`: string season / float week coercion; null `player_id` dropped; non-enum
  positions (`DB`, `OL`, `None`) filtered; `posteam` → `team` normalization.
- `pfr_advstats`: unmatched `pfr_player_id` dropped; all three stat types normalize.
- `injuries`: `POST` rows filtered; null `report_status` preserved as null, not coerced;
  verbose practice labels mapped to enum values.

Plus the repo's standing gate: `pytest -v`, `mypy src tests`, `ruff check src tests`,
`ruff format --check src tests`, and — because this touches pandera schemas and ingest paths —
`pytest -v -k "ingest or store or schemas"`.

A real-data smoke run against 2024–2025 is run manually before merge and its row counts recorded in
the PR description, since synthetic fixtures cannot catch an upstream shape change.

## Follow-ups (issues, not scope)

- `load_pfr_advstats(stat_type="def")` → D/ST modelling raw material. Needs its own spec covering
  team aggregation and opponent adjustment.
- Feature builders consuming `*_exp` columns for luck regression (extends #121).
- Start/sit and waiver tooling consuming injury status (extends #119).
- `load_ftn_charting`, `load_participation`, `load_ff_rankings` — probed as available, unscoped.
