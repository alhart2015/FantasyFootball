# depth_charts 2025+ — derive (season, week) from snapshot-by-timestamp feed — Design

**Status:** approved
**Date:** 2026-05-15
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Parent:** TODO #34 in `TODO.md`
**Predecessor:** PR #35 (`feat/nflreadpy-migration`) merged at commit `0b52a9d`

---

## 1. Overview

nflverse's 2025+ release of `depth_charts` is a snapshot-by-timestamp feed (one row per `(dt, team, player)`; full-roster on each snapshot). The pre-2025 format was a weekly per-team roster keyed on `(season, week, club_code, depth_team, depth_position)`. Those legacy columns are gone from the new payload; `refresh_depth_charts` in PR #35 deliberately raises `NotImplementedError` for any season whose payload lacks them.

This spec derives the legacy `DepthChartsSchema` shape — `(season, week, team, gsis_id, position, depth_team, depth_rank)` — from the new feed by joining each snapshot's `dt` against `schedules.kickoff` and picking the closest-prior snapshot per team-week.

### 1.1 Scope

- Net effect: `refresh_depth_charts(seasons=[2025])` succeeds and writes a `data/raw/depth_charts/season=2025/` partition validated by the existing `DepthChartsSchema`.
- 2018–2024 partitions still work via the unchanged legacy path. The new derivation runs only when the raw payload lacks legacy columns (presence of `dt` and absence of `season`/`week`/`club_code` is the dispatch signal).
- Downstream feature builders (`build_*_features`) consume `DepthChartsSchema` unchanged — no schema migration, no consumer changes.

### 1.2 Out of scope

- Defensive players and special-teams snapshots beyond the existing Position-enum filter — those rows are dropped during normalization, same as the legacy path.
- Per-day snapshot dt cadence audit beyond a smoke-level "expected ~daily" assertion.
- Historical re-derivation of 2018–2024 from the new format. The legacy partitions are already on disk and serve as the source of truth.

---

## 2. New payload shape (probed 2026-05-15)

`nflreadpy.load_depth_charts(seasons=[2025]).to_pandas()` returns ~554k rows over 12 columns:

| column | dtype | example | semantic |
|---|---|---|---|
| `dt` | str ISO Z | `2025-09-04T13:21:11Z` | snapshot timestamp (UTC) |
| `team` | str | `KC` | already-normalized 3-letter code |
| `player_name` | str | `Patrick Mahomes` | display only |
| `espn_id` | str | `3139477` | id_map join key (unused here) |
| `gsis_id` | str | `00-0033873` | canonical id |
| `pos_grp_id` / `pos_grp` | str | `Base 4-3 D` | position group |
| `pos_id` / `pos_name` / `pos_abb` | str | `WR` | position abbreviation |
| `pos_slot` | int | `1` | slot number within position group (WR1 = 1, WR2 = 2) |
| `pos_rank` | int | `1` | rank within slot (1 = slot's primary, 2 = backup, …) |

221 distinct `dt` values in the 2025 payload — roughly daily.

`schedules.kickoff` is a `DatetimeTZDtype` with `tz="UTC"` per `SchedulesSchema`; the existing `_build_kickoff` in `ingest/schedules.py` localizes ET wall-clock then converts to UTC.

---

## 3. Derivation rule

**For each `(team, season, week)` in the season's schedules:**

1. Take that team's kickoff that week as the reference instant (`ref = max(home_kickoff, away_kickoff)` — the team plays exactly one game per week; bye weeks are absent from schedules and thus skipped).
2. Among raw snapshots, pick the one whose `dt` is **strictly before `ref`** with the **largest `dt`** (closest-prior). If no snapshot satisfies `dt < ref` (e.g., the very first week before any preseason snapshot was published), skip that team-week and log a warning with a count.
3. Filter the chosen snapshot to rows where `team` matches.
4. Filter to `pos_abb ∈ {QB, RB, WR, TE}` via the existing `Position` enum.
5. Synthesize:
   - `depth_rank = clip(pos_rank, 1, 10)` per `DepthChartsSchema.depth_rank` bounds.
   - `depth_team = str(depth_rank)` — matches the legacy on-disk values (verified against `data/raw/depth_charts/season=2024/`: `depth_team` is the pyarrow string column with only `{"1", "2", "3"}` as distinct values).
   - `position = pos_abb`.
6. Emit one row per `(season, week, team, gsis_id, position, depth_team, depth_rank)`. Dedupe on `(gsis_id, season, week, team)` defensively — a player should appear at most once per team-snapshot.

### 3.1 Why `pos_rank` (not `pos_slot`) for `depth_rank`

Probed semantics of the new feed (KC, first 2025 snapshot, WR rows):

| player | `pos_abb` | `pos_slot` | `pos_rank` | legacy `depth_rank` analog |
|---|---|---:|---:|---:|
| Rashee Rice (WR1 starter) | WR | 1 | 1 | 1 |
| Xavier Worthy (WR2 starter) | WR | 2 | 2 | 2 |
| Tyquan Thornton (slot/SWR starter) | WR | 8 | 3 | 3 |
| Nikko Remigio (WR1 backup) | WR | 1 | 4 | 4 |

`pos_slot` is a position-id-like value (QB=9, RB=11, TE=10; WR varies among 1/2/8 for X/Z/slot). `pos_rank` is the team's depth rank across all slots within a position group. The legacy `depth_rank` is "rank within position group, 1 = primary" — `pos_rank` is the direct match.

### 3.2 Why no `pos_rank == 1` filter

Each player is listed at most once per snapshot per team in the new feed (verified on the KC sample: 8 distinct WRs, no duplicates). The legacy partitions list players at every slot occupied during the game (a player rotating between WR1 and WR2 produces two rows with identical `gsis_id`); the new derivation cannot reconstruct that alignment-tracking — and downstream feature builders already dedupe via min `depth_rank`, so the loss is invisible to consumers.

### 3.2 Timezone

`dt` is UTC (ISO ending `Z`). `schedules.kickoff` is `DatetimeTZDtype(tz="UTC")`. Direct comparison after parsing `dt` to a tz-aware UTC pandas Timestamp.

---

## 4. Dispatch

`_normalize_one_season(raw)` already checks for legacy required columns and raises `NotImplementedError` if any are missing. Replace that raise with: if the missing set matches the post-2025 signature (i.e., `dt` is present in raw), call the new derivation; otherwise re-raise so unknown-shape upstreams still fail loud.

`refresh_depth_charts` gains an optional `schedules` parameter. When `None` (the normal CLI invocation), it reads `data/raw/schedules/season=<S>/` for the same season via `store.read_partition`. Tests pass `schedules` directly. If the raw payload is the new format and schedules is unavailable on disk for that season, raise `FileNotFoundError` with a clear "ingest schedules first" message.

The orchestrator (`ingest/refresh.py`) already calls `refresh_schedules` before `refresh_depth_charts`, so the read-from-disk fallback works under normal usage.

---

## 5. Risk register

1. **Snapshot gap before week 1.** If the first 2025 snapshot is published *after* week 1 Thursday's kickoff, no closest-prior snapshot exists for any team playing week 1. Mitigation: probe `dt.min()` against `schedules[week==1].kickoff.min()` during real-data run; if there's a gap, warn loudly and let the partition be missing-rows for that week. (Empirically: probe earlier showed `dt.min() == 2025-08-03`, well before the 2025 season opener — risk is low for the historical pull but documented for in-season refreshes.)
2. **A team's kickoff falls within the same daily snapshot's window.** If snapshot dt is 2025-09-04T10:00Z and kickoff is 2025-09-04T13:00Z, the snapshot is closest-prior — fine. The strict `<` rule handles ties correctly (a snapshot stamped at the kickoff instant is excluded).
3. **Position filter dropping legitimate hybrid roles.** `pos_abb` covers offensive positions for QB/RB/WR/TE distinctly; the existing `Position` filter handles this cleanly. FB and other adjacent roles are dropped, same as the legacy path.
4. **Per-team-week row count vs legacy.** Legacy 2024 has ~7.8 WR rows per team-week (max 18) because nfl_data_py listed each player at every slot they occupied during the game (alignment-tracking — same player can appear at rank 1 and rank 2 in adjacent plays). The new feed lists each player once per snapshot at a single rank. Expected new row count per team-week: 5-8 WR + 2-3 RB + 1-2 QB + 2-3 TE = ~10-16 rows. Downstream feature builders already dedupe via min `depth_rank` so the alignment-tracking loss is invisible to consumers.

---

## 6. Verification

- **Unit tests** (synthetic fixture):
  - Closest-prior snapshot rule (snapshot at `dt < kickoff` with the largest `dt` chosen; strict `<`).
  - `depth_rank` clamp: `pos_rank=15` → `depth_rank=10`.
  - `depth_team = str(depth_rank)` synthesis.
  - Position enum filter (drop defensive rows).
  - Bye-week team absent from schedules → no rows for that team-week.
  - First-week-no-prior-snapshot → row skipped with warning.
- **Real-data smoke**: `refresh_depth_charts(data_root, seasons=[2025])` succeeds, writes a partition validated by `DepthChartsSchema`. Expected row count: 22 weeks × 32 teams × ~12 offensive players ≈ 8,000-10,000.
- **Forced verification checklist** per CLAUDE.md "FORCED VERIFICATION" rule.

---

## 7. Plan-vs-execution deviations

Tracked at PR time. None expected from spec; surfaces in the plan.
