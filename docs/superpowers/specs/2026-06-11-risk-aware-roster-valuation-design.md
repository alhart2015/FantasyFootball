# Risk-Aware Roster Valuation (season availability)

## 1. Purpose

The draft tournament currently scores a roster by `optimal_lineup_points` — the projected points of its
**best single-week starting lineup**. That metric is blind to the bench: a roster with a worthless bench
(validation found `raw_vorp` drafting **10 quarterbacks**) scores the same as one with real RB/WR depth,
because depth never enters the number. But a real fantasy season is 17+ weeks of injuries, byes, and
benchings, and that risk is **position-dependent** — your QB1 is far likelier to play all season than your
RB1, so you need *more* RB depth to cover the higher chance of missing time.

This slice replaces the starters-only metric with a **risk-aware roster valuation**: the expected points a
roster actually scores across a full season under per-player availability, filling the best legal lineup
each week from whoever is healthy. Depth then pays for itself — a backup is worth the points it contributes
when a starter is out, weighted by how often that happens — and it is automatically position-weighted
(RBs are out more, so RB depth gets used more). This makes the tournament reward the right thing and gives a
future depth-aware *strategy* a correct target to optimize against.

**This slice builds the metric, not a new strategy.** The strategy that drafts good benches is the next slice.

## 2. Scope

### In scope
- **`availability.py`** — a per-player season availability model: `p = E[games]/17` from historical
  `weekly_stats`, with per-position defaults for rookies / no-history, plus the bye week per player from
  `schedules`.
- **`season_value.py`** — `expected_season_points(...)`: the Monte-Carlo season valuer.
- **A pluggable `RosterValuer` seam** so the cheap `optimal_lineup_points` and the new
  `expected_season_points` both satisfy one interface; the tournament takes a valuer and can **A/B them**.
- **Tournament integration** — `_strategy_values` / `run_tournament` / `tune_sigma` accept a valuer
  (default unchanged = starters-only; opt into season-value).
- **Validation** — re-run the tournament under the new metric; confirm the 10-QB roster now scores far worse.

### Explicitly out of scope (later slices)
- **Weekly performance variance** ("a healthy player underperforms") — needs real per-player distributions
  (the deferred distribution-wrapping); v1 weekly points are deterministic.
- **Injury clustering / spells** (an injury costs 6 *consecutive* weeks) — v1 is independent per-week
  Bernoulli, which is correct for an *expected*-points metric (clustering moves variance, not the mean).
- **Per-player injury priors beyond games-played history** (age curves, injury-type models).
- **A depth-aware `DraftStrategy`** — the metric is the foundation; the strategy is the next slice.
- **Ceiling/floor or tournament-style (GPP) roster scoring** — that is where SD/clustering would matter.

## 3. Design

### 3.1 Inputs and where the data comes from

- **Roster** — the hero's drafted rows (a `VorpTableSchema` sub-frame): `gsis_id`, `position`,
  `season_mean_fpts`. The metric reads only those three columns.
- **`LeagueConfig.roster_slots`** — the starting-slot shape (league-driven, as everywhere in the Draft Hub).
- **`id_map`** (already loaded by the assistant CLIs) — supplies each player's **current `team`**
  (`gsis_id → team`), required to resolve byes; the roster/VORP frame itself carries no `team` column.
- **Historical `weekly_stats`** (2018–2024, already ingested) — one row per (player, season, week); the
  count of weeks a player appears in a season is their games played. Source of per-player availability and
  the per-position defaults.
- **Target-season `schedules`** — a team's **bye** in `season` (e.g. 2026) is the week (1..18) in which the
  team is neither `home_team` nor `away_team`. This is a dependency on the **target-season partition** (the
  2026 schedule, not the historical 2018–2024 ones); §3.2 step 4 specifies graceful degradation if absent.

No new ingest *source*; all tables exist (the target-season schedule must be ingested for the draft year).
The availability model is derived, not ingested.

### 3.2 Availability model (`availability.py`)

```
build_availability(weekly_stats, schedules, id_map, pool, *, season, lo=0.4, hi=0.97) -> PlayerAvailability
```

`PlayerAvailability` answers two questions per rostered `gsis_id`:
- **`p_week(gsis_id) -> float`** — per-week probability the player is healthy/active (injury + benching),
  independent across non-bye weeks.
- **`bye_week(gsis_id) -> int | None`** — the week the player is forced out (their team's bye), or None.

Construction:
1. **Per-player availability fraction** — from `weekly_stats`, games played per (gsis_id, season) = count of
   weeks appearing. Normalize **per season** by that season's scheduled game count (16 for 2018–2020, 17 for
   2021+) so the 16-game and 17-game eras are comparable: `frac_season = games_played / sched_games`. The
   player's `p_raw` is the mean of `frac_season` across their seasons (recency-weighting is a future
   refinement — v1 uses the simple mean).
2. **Per-position default** — for rookies / players with no `weekly_stats` history, use that position's mean
   `p` across all players with history (RB lower, QB higher — derived from the same data, not hardcoded).
   2026 rookies carry deterministic `99-XXXXXXX` placeholder gsis_ids (per the external-ingest design) that
   match no `weekly_stats` row, so they fall to this default by construction — the common rookie path, not
   an edge case.
3. **Clamp** `p` to `[lo, hi]` so no player is degenerate (0 or 1) — even workhorses miss a game; even the
   injury-prone aren't out half the season in expectation.
4. **Bye week** — resolve the player's `team` from `id_map`, then the team's bye in the target-season
   `schedules` (the week with no game row for that team); the player is forced out that week. A player with
   no resolvable team or bye simply has no forced-out week. If the target-season `schedules` partition is
   **absent entirely**, `build_availability` logs a warning and returns availability with **no byes** (the
   injury model still applies) — a deliberate graceful degradation, not a silent failure.

### 3.3 Per-game points convention (the key modeling decision)

ESPN's preseason projection is a **healthy full-season** number (established by the #52 benchmark spike: it
projected the injury-prone CMC at ~335, a healthy season, not an injury-discounted one). So `season_mean_fpts`
is read as a **17-game healthy total** (2026 is a 17-game NFL season), and the per-game scoring rate is
derived **inside the valuer** from the roster's own column — it is not a separate input:

```
per_game(gsis_id) = season_mean_fpts(gsis_id) / 17
```

The injury discount is then applied by the simulation via `p` (a player available `p` of the time scores, in
expectation, `per_game × 17 × p = season_mean_fpts × p`). This **correctly marks injury-prone players below
their optimistic ESPN number** and lets the bench cover the gap. The rejected alternative — reading the
projection as already injury-adjusted (`per_game = season_mean_fpts / E[games]`, no markdown) — would
*double-count* injury given what the spike showed about ESPN. `season_mean_fpts` is non-null on the consensus
VORP path (schema-enforced), so `per_game` is always defined.

### 3.4 The Monte-Carlo valuer (`season_value.py`)

```
expected_season_points(
    roster, roster_slots, availability, *, n_sims, rng, weeks=range(1, 18)
) -> float
```

`weeks` defaults to the 17-week fantasy season (NFL weeks 1–17; week 18 is typically not a fantasy week —
playoff weighting is a deferred refinement, §6). Conceptually: `E[ Σ_weeks (best legal lineup from the
available players that week) ]`.

**Reuse `optimal_lineup_points` directly — no fill refactor.** Because `per_game = season_mean_fpts / 17`
is a *uniform* scaling (the same divisor for every player), the optimal weekly lineup chosen by `per_game`
is identical to the one `optimal_lineup_points` chooses by `season_mean_fpts`; only the sum scales by 1/17.
So a simulated week is exactly:

```
week_points = optimal_lineup_points(available_subset, roster_slots) / 17
```

where `available_subset` is the roster rows not on bye that week and healthy w.p. `p_week`. No new slot-fill
code is needed — the existing greedy fill (single-position → FLEX → SUPER_FLEX) is reused verbatim. Season
value = the average over `n_sims` of `Σ_weeks week_points`.

**Efficiency — the single-week factorization.** Because weekly points are deterministic and availability is
independent across weeks, **every non-bye week has the identical expected sub-problem**, and by linearity of
expectation the season value is *exactly* `Σ_weeks E[week]` — the factorization is **exact in expectation**;
its only error is MC sampling, identical to a brute-force per-week run. So we Monte-Carlo the expected
best-lineup for a generic non-bye week *once* (`n_sims` availability draws), reuse that estimate for every
non-bye week, and recompute only the weeks in which a roster player is on bye (those players forced out).
This collapses `weeks × n_sims` into roughly `n_sims + (distinct bye weeks among the roster) × n_sims`,
keeping the tournament in the minutes range with `n_sims ≈ 300`.

The valuer is a **pure function** of `(roster, roster_slots, availability, rng, n_sims)` — fully reproducible
by seed — so it works both inside the tournament and as a standalone "value this roster" call.

### 3.5 Pluggable `RosterValuer` seam

```python
class RosterValuer(Protocol):
    def value(self, roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]) -> float: ...
```

Two implementations:
- **`StartersValuer`** — wraps the existing `optimal_lineup_points` (the cheap, deterministic,
  starters-only metric). The current default.
- **`SeasonValuer`** — wraps `expected_season_points` with a bound `availability`, `n_sims`, and a
  `base_seed`. Since the `value(roster, roster_slots)` protocol takes no seed, the valuer derives a
  **deterministic per-roster seed** from `base_seed` + a stable hash of the roster's sorted `gsis_id`s (the
  same sha256-derived-seed idiom `scoring/score_distribution.derive_row_seed` already uses), so the
  tournament stays fully reproducible and identical rosters score identically.

`run_tournament` / `tune_sigma` / `_strategy_values` take a `RosterValuer` (default `StartersValuer`, so
existing behavior and tests are unchanged). The CLI gains a `--valuer {starters,season}` flag. This keeps
both metrics live and lets us **measure the difference** rather than silently swapping.

### 3.6 What changes and what does not
- **Unchanged:** the draft simulation, the strategies, the survival model, `optimal_lineup_points` itself,
  the bootstrap/winner machinery. The valuer is the *only* new thing the tournament calls per scored roster.
- **New modules:** `availability.py`, `season_value.py`, the `RosterValuer` protocol + two impls (co-located
  with `season_value.py` or in a small `valuer.py`).
- **Touched:** `tournament.py` (`_strategy_values`/`run_tournament`/`tune_sigma` thread a valuer),
  `tournament_cli.py` (`--valuer`, build the `SeasonValuer` from `weekly_stats` / `schedules` / `id_map`).

## 4. Testing

Synthetic fixtures (project norm — no network in unit tests):
- **Availability construction** — a tiny `weekly_stats` fixture where player A plays 17/17 across seasons and
  player B plays 9/17; assert `p_A` ≈ 1.0 (clamped to `hi`), `p_B` ≈ 0.53; a rookie (no history) gets the
  position default; a `schedules` fixture pins the bye week.
- **`expected_season_points` closed form** — a 1-slot roster (`{RB:1}`) with a starter `p=0.5` (no bye, no
  bench) over `weeks=2` → `E = 2 × 0.5 × per_game`; matches the MC to tolerance. Add a backup RB and assert
  the value rises by the backup's expected fill-in contribution (closed-form for a 2-player case).
- **Reduces to starters-only** — with every `p=1.0`, no byes, and `weeks = range(1, 18)`,
  `expected_season_points` equals `optimal_lineup_points` exactly: 17 weeks × `per_game` (= `season/17`) sums
  back to the season total of the optimal starting lineup. (The metric degenerates correctly.)
- **Factorization correctness** — on a small roster with at least one bye, the factorized
  `expected_season_points` equals a brute-force per-week season MC (every week simulated independently with
  the same availability draws) to MC tolerance. Guards the week-counting and bye-week handling of the
  optimization (which has no other coverage).
- **Depth is rewarded / QB hoarding is punished** — a hand-built roster with 3 RBs scores higher than an
  equal-projection roster with 1 RB + 2 spare QBs at the same total projection (the core behavior).
- **Determinism** — same seed ⇒ same value; different seed ⇒ generally different.
- **Bye handling** — a roster whose only RB is on bye in week W scores 0 at RB that week (and the FLEX/bench
  covers if eligible); a roster with an RB backup not on bye that week does not.
- **Valuer seam** — `StartersValuer` equals `optimal_lineup_points`; `run_tournament` with the default valuer
  is byte-identical to today (no regression); with `SeasonValuer` it produces a different, sensible ranking.
- **CLI** — `--valuer season` runs end-to-end on a fixture.

Gates per the project bar: `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check`.

## 5. Key decisions

- **5.1 Metric, not strategy, this slice** — you cannot build or trust a depth-aware strategy without a
  metric that rewards depth. Ship the valuation first; the strategy optimizes against it next.
- **5.2 Expected-points metric ⇒ per-week Bernoulli on mean availability** — the SD and injury clustering
  move outcome *variance*, not the *expected* value we score rosters by, so mean availability is sufficient
  and far simpler. Clustering/ceiling is a later ceiling-floor concern.
- **5.3 Availability only; deterministic weekly points** — availability (injury + bye) is the load-bearing
  piece for valuing depth; weekly performance variance needs distributions we do not have and adds
  "best-ball" upside that is not required to fix QB hoarding. Deferred.
- **5.4 `per_game = projection / 17`, injury applied via `p`** — ESPN projects healthy seasons (the #52
  spike), so the projection is not pre-discounted; applying `p` is the value-add, not double-counting.
- **5.5 Pluggable `RosterValuer`, default unchanged** — keep the cheap starters metric and the season metric
  side by side; A/B rather than swap, so existing tournament behavior/tests are untouched and the difference
  is measurable.
- **5.6 Single-week factorization for speed** — deterministic points + week-independent availability make
  every non-bye week identical, collapsing the season MC; without it the metric is too slow at tournament
  scale.

## 6. Open questions / future slices

- **Recency-weighted / age-adjusted availability** — v1 uses a simple historical mean of games played; a
  weighted or age-curve model is a refinement once v1 lands.
- **Weekly performance variance** — unlocks "best-ball" depth value and ceiling/floor scoring; arrives with
  real per-player distributions.
- **Depth-aware strategy** — a `DraftStrategy` that values marginal positional insurance, optimizing directly
  against this metric. The natural next slice.
- **Playoff weighting** — v1 sums all weeks equally; weighting fantasy-playoff weeks is a later option.
