# Vegas Team-Context Feature Family Probe — Design

**Date:** 2026-05-17
**Branch:** `feat/probe-vegas-team-context`
**Status:** spec
**Predecessor:** PR #47 (upside-ranking diagnostic, merged at `302661c`) returned `No greenlight` on TODO #33d — the elite-season under-projection problem does not live in distribution-tail mining of the existing model output. The diagnostic's verdict explicitly recommended **33c next** as the genuinely-unexplored feature class (project_management.md, 2026-05-17 entry). This probe is the cheapest entry into that family.

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether a **forward-looking Vegas team-context feature family**, derived without new ingest from already-available `spread_line` and `total_line` columns in `SchedulesSchema`, carries orthogonal signal beyond v1 + every Track 2 family already shipped (PBP team / red-zone / pressure / trajectory / weather).

The family targets **a mechanism axis the current feature set already partially covers** (`implied_team_total` and `spread` are baseline features in QB/RB/WR/TE, derived in `_shared.build_game_environment`), **but only at per-game granularity**. The two new mechanism axes are:

1. **Preseason team-strength** — the Vegas market's view of the team's quality at week 1 of the season, *before* any in-season trailing data exists. Captured here as the team's week-1 closing `implied_team_total` and `spread`, broadcast across the season. This is the closest pbp-derivable proxy for "Vegas's May view," which is exactly the signal TODO #33c hypothesizes is missing.
2. **As-of-time team-strength rollup** — the season-to-date running average of `implied_team_total` and `spread` across all games this team has played. Distinct from trailing-4-game stat features in that it reflects Vegas's market view rather than the team's actual on-field performance.

### 1.2 Family-level prior framework

Per project_management.md (2026-04-30 "PBP Feature Family Probe"), the canonical workflow is:

1. Bundle 3–4 mechanism-distinct candidates per family probe.
2. Run the probe at BaselineModel + lgb-nb composite × augment + swap modes.
3. SIGNAL → greenlight a per-position integration plan.
4. NULL → close the family; refined-unit candidates unlikely to clear absent independent evidence.

Track 2 verdict pattern through PR #47: 2 of 6 family probes returned SIGNAL — PR #20 (RB PBP team features) and PR #25 (trajectory). Four returned NULL durable (PR #22 receiver air-yards, PR #23 red-zone, PR #24 pressure, PR #28 weather, PR #29 weather-refined). Family-probe SIGNAL hit rate ≈ 25-33%.

**Mechanism prediction for this family** (pre-registered):

- **Most likely SIGNAL at Phase 2:** **RB swap mode** — preseason team-strength is a forward-looking RB-rushing-volume proxy that the per-game `spread` doesn't capture in weeks 1-4 (no trailing performance data exists yet to confirm/deny Vegas's preseason view; the per-game `spread` already partially reflects current expectations).
- **Possible SIGNAL at Phase 1:** per-stat on `passing_yards` / `receiving_yards` at QB / WR — preseason `implied_team_total` is a forward-looking scoring-environment signal independent of trailing performance.
- **Probably NULL at TE:** TE elite-season magnitude is target-share-driven, not scoring-environment-driven (per the 2024 retrospective: Kittle's miss was -70 vs Chase's -143).
- **Tautological at lgb-nb augment:** trees get all this signal from existing continuous `spread` / `implied_team_total` cols. lgb-nb is informative only at Phase 2 (composite), and even there mostly only in swap mode.

### 1.3 Success criteria

The probe **ships** when all four pass. The verdict (ADOPT/NULL) is informational, not a ship gate.

1. **Coverage:** per-(position, season) override coverage ≥90% over the 2021–2024 eval window. The default 0.95 threshold will fail on week-1 rows where `season_avg_*` is NaN by construction (no prior games to average); 0.90 is the cold-start relaxation. PR #25 trajectory's precedent (0.35 relaxation for 5-game trajectories) shows the probe runner accepts a documented relaxation cleanly.
2. **Probe completeness:** 4 reports per family — BaselineModel × augment, BaselineModel × swap, lgb-nb composite × augment, lgb-nb composite × swap. All complete without error. Phase 1 (per-stat pooled) and Phase 2 (composite, via `--force-composite` on lgb-nb) verdicts rendered to per-(model, mode) reports + summary report following the PR #28 template.
3. **Both model classes tested at composite** — Phase 1 is RidgeCV-only regardless of `--model`, so bare lgb-nb Phase 1 verdicts are tautological with baseline. Re-run lgb-nb with `--force-composite` to actually test lgb-nb at composite (standardized since PR #25).
4. **Verification gates green:** mypy strict + ruff + ruff format clean across `src/`, `scripts/`, `tests/`. Relevant pytest subset clean.

### 1.4 Out of scope (deferred)

- **External Vegas data (season win totals, preseason O/U).** Genuine preseason Vegas signals (the *May* line, set before any week-1 games) require an external ingest path with no free historical source we currently have. Deferred to a follow-up plan if this probe returns SIGNAL — at that point we have evidence the mechanism is real and the cost of new ingest is justified.
- **Non-linear encodings of existing Vegas cols** (`is_favored`, `is_heavy_favorite`, `is_high_total`, `spread * implied_team_total` interaction). Tree models (lgb-nb) get these natively from the existing continuous columns; Ridge can't, but adding them to this bundle would muddy the two preseason / as-of-time mechanism axes. Refined-unit candidates revisit-only-on-SIGNAL.
- **Opponent-side Vegas rollups** (opponent's `season_avg_*` to date, opponent's preseason strength). Information-theoretically a separate hypothesis from team-side rollups. Punt to follow-up if team-side returns SIGNAL.
- **Line-movement features** (open-to-close spread movement, sharp-money proxies). No historical open-line data in pbp; would require external ingest.
- **Position-specific encodings.** `preseason_pass_volume_proxy` (`implied_team_total × pass_rate_prior`) would be QB/WR-specific. The probe uses position-agnostic team-context cols only; per-position refinement is follow-up.
- **Schema integration.** A SIGNAL verdict greenlights a separate per-position integration plan (extend `_shared.build_game_environment` to emit the 4 new cols alongside `spread` / `implied_team_total`; extend each `*FeaturesSchema`; refresh caches; run the dual-run adoption gate). This branch is probe-only.
- **2025 eval extension.** Feature partitions under `data/features/{position}/season=*/` stop at 2024. Extending the holdout to 2025 (where the 33d evidence most directly motivates this work) requires `python -m scripts.refresh_features {qb,rb,wr,te} --seasons 2025` plus a pbp-2025 cleanliness check. Defer — adds ~20 min of feature-cache work for a marginal eval-window expansion. If the probe is NULL on 2021-2024, 2025 won't rescue it; if it's SIGNAL, the integration plan does the 2025 work properly.
- **Recurring QB augment regression mitigation.** PRs #23 / #24 / #25 / #28 each saw QB augment regress on context / team / trajectory / weather adds. Probe-only here; if QB augment regresses again, document and route around in any integration follow-up — do not block the probe's verdict on it.

---

## 2. Source data — already ingested

`SchedulesSchema` (in `src/projections/schemas.py`) already declares the two source columns:

| Column | Type | Nullable | Source | Notes |
|---|---|---|---|---|
| `spread_line` | `Series[float]` | yes | `import_schedules.spread_line` | Vegas closing spread, **inverted from sportsbook convention**: positive when home team is favored, negative when away favored. Empirically verified against `nfl_data_py.import_schedules([2023])` (see `_shared.build_game_environment` docstring). |
| `total_line` | `Series[float]` | yes | `import_schedules.total_line` | Vegas closing O/U, integer or half-point. Range 30-65 historically. |

**No new ingest is required.** The existing `refresh_schedules` already writes these columns into `data/raw/schedules/season=YYYY/part.parquet`. Coverage is high (`<1%` of regular-season games have NaN `spread_line` / `total_line` in 2018-2024; verified empirically by the existing `implied_team_total` feature, which is non-null in ~99% of v1 baseline rows).

This is the same simplification PR #28 (weather) enjoyed vs. PR #20 / PR #25 (which required new ingest paths). The probe touches the override script, a thin `vegas_team_context_features.py` compute module, and tests — that's it.

---

## 3. Feature definitions

Four mechanism-distinct features. All `pa.Column(pa.Float64, nullable=True)` in the override schema (boolean encodings deliberately not included — see §1.4 non-linear encodings).

### 3.1 `preseason_implied_team_total`

- Continuous, position-agnostic.
- Definition: for each `(season, team)`, look up the team's **first regular-season game** (lowest `week` where the team appears as `home_team` or `away_team` in `schedules`). Compute that game's team-perspective `implied_team_total` per `build_game_environment`'s formula (`(total_line + spread_line) / 2` for home, `(total_line - spread_line) / 2` for away). Broadcast across all weeks of `(season, team)`.
- Mechanism: Vegas's preseason view of the team's per-game scoring environment. Constant within a team-season → 32 unique values per season × 4 holdout seasons = 128 unique values across the eval window, broadcast across all rows of that team-season.
- Compute: `compute_vegas_team_context_features(schedules)` first applies `build_game_environment` to get per-team-game rows, then groups by `(season, team)`, identifies the min-week row, broadcasts.

### 3.2 `preseason_spread`

- Continuous, position-agnostic.
- Definition: same — team's week-1-game closing `spread` (team's POV, favorite negative, dog positive per `build_game_environment` convention), broadcast across all weeks.
- Mechanism: Vegas's preseason view of the team's strength relative to its week-1 opponent. Imperfect because week-1 opponent quality varies, but the league-wide preseason consensus is reasonably well-captured in opening-week lines (sportsbooks set early lines deliberately tight to consensus).

### 3.3 `season_avg_implied_team_total`

- Continuous, position-agnostic.
- Definition: for each `(season, team)` and target `week`, the mean of `implied_team_total` over all `(season, team, prior_week)` games where `prior_week < week`. Implemented as `df.sort_values(["season", "team", "week"]).groupby(["season", "team"])["implied_team_total"].expanding().mean().shift(1)`.
- NaN at week 1 of each season (no prior games). Defined for all subsequent weeks. Bye-week handling: bye weeks produce no schedule row, so the running mean simply doesn't update across a bye — correct semantics.
- Mechanism: as-of-time team scoring-environment view per the Vegas market. Distinct from trailing-4-game stat features in that it reflects Vegas's *market view* (set before each game, embedding all currently-known info) rather than the team's *actual on-field performance*.

### 3.4 `season_avg_spread`

- Continuous, position-agnostic.
- Definition: same — running mean of team-perspective `spread` for all games the team has played this season prior to the target week.
- NaN at week 1.
- Mechanism: as-of-time team strength per Vegas. Persistent dominance (KC, BAL, BUF) produces large-magnitude negative averages; persistent struggling teams produce large-magnitude positive averages.

### 3.5 Cold-start handling

`season_avg_*` is NaN at week 1 by construction. This drops ~6% of holdout rows on the candidate side under the default coverage check (week 1 is one of ~17 weeks per season; rate slightly higher because all 32 teams play in week 1).

**Decision: relax coverage threshold to 0.90.** Documented in the summary report. Alternative considered and rejected: filling week-1 `season_avg_*` with the `preseason_*` value. The `preseason_*` columns *are* the week-1 game's values; filling `season_avg_*` with them at week 1 would make the two columns identical for ~6% of rows, muddying the two-mechanism-axis structure of the bundle. Explicit relaxation is cleaner.

### 3.6 Sign / convention sanity

The `_shared.build_game_environment` convention is **favorite negative, dog positive** at the team-perspective `spread` level. Both `preseason_spread` and `season_avg_spread` inherit this convention without sign-flip. A team-season-averaged spread of, e.g., `-3.5` means Vegas favored this team by ~3.5 on average across the games observed so far this season.

A spec-level verification check (in tests): for the 2023 KC team-season, `preseason_spread` should be negative (KC favored vs. DET in week 1); for the 2023 ARI team-season, `preseason_spread` should be positive (ARI dog vs. WAS in week 1).

### 3.7 Schema integration deferred

Like every Track 2 probe-only spec, the four columns live in the *override parquet*, not the position schemas yet. A SIGNAL verdict triggers a follow-up plan to:

- Extend `_shared.build_game_environment` (or add a sibling `build_vegas_team_context`) to emit the 4 new cols per team-game row alongside `spread` / `implied_team_total`.
- Add the 4 cols to each `{Qb,Rb,Wr,Te}FeaturesSchema`.
- Update `BaselineModel._{POS}_FEATURE_COLUMNS` (per-position feature-column tuples; lightgbm derives features from the schema dynamically and auto-picks up).
- Refresh feature caches under `data/features/{position}/`.
- Run the dual-run adoption gate (`scripts/adoption_gate.py --baseline-run ... --candidate-run ...`).

---

## 4. Module structure

### 4.1 New files

```
src/projections/features/vegas_team_context_features.py    # compute fn + assembler
scripts/build_vegas_team_context_override.py                # CLI override generator
tests/test_features/test_vegas_team_context_features.py     # ~12 feature unit tests
tests/test_scripts/test_build_vegas_team_context_override_cli.py  # 4 CLI tests
```

### 4.2 Modified files

None. No schema changes (probe-only); no ingest changes (`SchedulesSchema` covers the source columns).

### 4.3 `vegas_team_context_features.py` interface

Pattern matches PR #25 (trajectory) / PR #28 (weather) — each `compute_*` returns every-week-at-once frames; the assembler joins onto the player-team-week index in one pass.

```python
def compute_vegas_team_context_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Returns (season, week, team, preseason_implied_team_total,
    preseason_spread, season_avg_implied_team_total, season_avg_spread) —
    one row per team-game.

    Steps:
    1. Call _shared.build_game_environment(schedules) to get per-team-game
       rows with (season, week, team, spread, implied_team_total).
    2. For each (season, team), identify the min-week row's spread and
       implied_team_total → broadcast as preseason_spread,
       preseason_implied_team_total across all weeks of that team-season.
    3. For each (season, team), sort by week and compute
       expanding().mean().shift(1) of spread and implied_team_total →
       season_avg_spread, season_avg_implied_team_total. NaN at week 1.

    Returned frame is sorted by (season, week, team) for caller convenience.

    Raises:
        ValueError: schedules missing required columns spread_line/total_line.
    """
    ...


def attach_vegas_team_context_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Returns a copy of `index` with 4 nullable-Float64 cols appended:
    preseason_implied_team_total, preseason_spread,
    season_avg_implied_team_total, season_avg_spread.

    Index must have columns (gsis_id, season, week, team, opp, position).
    The merge is left on (season, week, team); rows without a matching
    schedule entry retain NaN. The assembler logs the rate of unmatched rows.
    """
    ...


def build_vegas_team_context_overrides(
    schedules: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Returns the override frame: (gsis_id, season, week, position,
    preseason_implied_team_total, preseason_spread,
    season_avg_implied_team_total, season_avg_spread).

    One row per input index row.

    Raises:
        ValueError: malformed index or duplicate (gsis_id, season, week, position) keys.
    """
    ...
```

### 4.4 `build_vegas_team_context_override.py` CLI

Mirrors `scripts/build_weather_override.py`:

- Args: `--seasons RANGE` (default 2018-2024), `--data-root PATH` (default `data/raw`), `--output PATH` (default `data/features_probe/vegas_team_context.parquet`).
- Reads schedules + weekly_stats via `store.read_partition`.
- Builds the player-team-week index for each season's eval window (positions filtered to QB / RB / WR / TE — same `_FANTASY_POSITIONS` carryover from prior override scripts).
- Calls `build_vegas_team_context_overrides`, writes the output parquet.
- Logs the per-(season, position) coverage rate, week-1 NaN rate (expected ~6%), unique team-season count per season (expected 32), and the histogram bounds (min/max preseason_spread, min/max season_avg_spread) to stdout, with an audit report sidecar (`reports/feature_probe_vegas_team_context_override_audit.md`; §6.7).

Acknowledged carry-over from PR #20 / #23 / #24 / #25 / #28: `_read_concat`, `_FANTASY_POSITIONS`, `_build_player_team_week_index`, `_parse_season_range` are duplicated across six existing override scripts. Extraction to `scripts/_override_common.py` is on the project_management.md housekeeping list. The Vegas-team-context script will copy the same patterns; folding the extraction into a probe-only spec is scope creep.

---

## 5. Probe protocol

### 5.1 Modes & model classes

Four reports per family — same matrix as PR #25 / PR #28:

| Model | Mode | Phase 1 | Phase 2 | Output |
|---|---|---|---|---|
| BaselineModel | augment | yes (RidgeCV per stat) | composite (default; via summary verdict) | `feature_probe_vegas_team_context_baseline_augment.{md,csv}` |
| BaselineModel | swap | yes (RidgeCV per stat) | composite | `feature_probe_vegas_team_context_baseline_swap.{md,csv}` |
| lgb-nb composite | augment | tautological (RidgeCV; informational only) | composite via `--force-composite` | `feature_probe_vegas_team_context_lgbnb_augment.{md,csv}` |
| lgb-nb composite | swap | tautological | composite via `--force-composite` | `feature_probe_vegas_team_context_lgbnb_swap.{md,csv}` |

### 5.2 Probe invocation

For each (model, mode) cell:

```
python -m scripts.probe_feature_signal \
    --candidate-name "vegas_team_context_{baseline,lgbnb}_{augment,swap}" \
    --override data/features_probe/vegas_team_context.parquet \
    --drop implied_team_total,spread \           # swap mode only
    --model {baseline,lightgbm-nb} \
    --seasons 2018-2024 \
    --holdout-years 2021-2024 \
    --coverage-threshold 0.90 \
    [--force-composite] \                        # lgb-nb runs only
    --csv-out reports/feature_probe_vegas_team_context_{baseline,lgbnb}_{augment,swap}.csv \
    > reports/feature_probe_vegas_team_context_{baseline,lgbnb}_{augment,swap}.md
```

**Swap mode drops `implied_team_total` and `spread`** from baseline (the candidate cols are the bundle's preseason + season-to-date encodings, replacing the per-game cols). `opp_allowed_*_fppg_l4` stays in place — that's a different feature family.

`--coverage-threshold 0.90` accommodates the week-1 cold-start NaN on `season_avg_*` (§3.5).

### 5.3 Eval window

- Train: 2018–2020.
- Holdout: 2021–2024.
- Matches all prior probes for cross-comparability. Extension to 2025 deferred per §1.4.

### 5.4 Coverage threshold

- Default: `--coverage-threshold 0.90` (cold-start relaxation; week-1 `season_avg_*` is NaN by construction).
- Expected pooled coverage on `preseason_*`: ≥99% (constant within team-season; only fails for the rare team-season with `spread_line` / `total_line` missing in week 1).
- Expected pooled coverage on `season_avg_*`: ~93-94% (one week of 17 is NaN per team-season; plus the ~1% of subsequent weeks where `spread_line` / `total_line` is missing).
- Fallback: if pooled coverage on `season_avg_*` drops below 0.90 due to higher-than-expected NaN rate in pbp, document the actual rate in the summary report and accept the run — relaxing further than 0.90 risks silent NaN imputation by Ridge biasing the Δ-RMSE.

### 5.5 Summary report template

`reports/feature_probe_vegas_team_context_summary.md` follows the PR #28 weather template:

- Decision log (commits + reasoning).
- Per-(model, mode) verdict table (Phase 1 SIGNAL count + Phase 2 ADOPT/MARGINAL/DO_NOT_ADOPT verdicts).
- Mechanism annotation: did `preseason_*` fire on RB swap (the pre-registered prediction)? Did `season_avg_*` fire on QB/WR pass-volume cells? Did anything fire on TE (predicted NULL)?
- Cold-start coverage relaxation note + actual measured coverage.
- Refined-unit candidates left unexplored (non-linear encodings, opponent-side rollups, line-movement, external season win totals).
- Recurring QB augment regression check — does the pattern from PR #23 / PR #24 / PR #25 / PR #28 recur on Vegas team-context?
- Mechanism reflection on the 33c hypothesis: if NULL, that closes the "preseason Vegas view is the missing signal" hypothesis at this level of approximation (week-1-line proxy + season-to-date rollup); the next step would be genuine external preseason data or pivot to a different forward-looking class (coaching tenure, FA flags).

---

## 6. Testing strategy

### 6.1 Per-feature unit tests (`tests/test_features/test_vegas_team_context_features.py`, ~12 tests)

For each of the 4 features, 3 tests:

1. **Correct computation on a synthetic 2-team / 4-week schedule fixture.** Team A goes 2018-week-1 at home, favored by 7, total 47 (→ `implied_team_total=27, spread=-7`); week-2 away, dog by 3, total 44 (→ `implied_team_total=20.5, spread=+3`); etc. Test that `preseason_implied_team_total = 27` and `preseason_spread = -7` for all of Team A's rows, regardless of week. Test that `season_avg_implied_team_total` at week 3 equals mean of weeks 1+2 values, etc.
2. **NaN propagation at cold-start.** Week-1 row of any team-season has `season_avg_*` = NaN; weeks 2+ are defined.
3. **Sign convention.** A team that was the favorite in its week-1 game has negative `preseason_spread`; a dog has positive. Per §3.6 verification.

### 6.2 Override assembler tests (~3 tests)

- Coverage check on a representative fixture: per-(season, position) coverage matches expected.
- Player-team-week index join produces exactly the expected rows (one per input row, no fan-out, no row loss for cases with present schedule data).
- Schema validation: round-trip through pandera holds (override parquet has the 4 candidate cols + 3 join key cols, all `Float64` / `string` / `Int64` as appropriate).

### 6.3 CLI tests (`tests/test_scripts/test_build_vegas_team_context_override_cli.py`, 4 tests)

- argparse smoke (parse_args with valid args).
- Default invocation writes expected file structure.
- Custom `--seasons` range respected.
- Error on missing `data-root` (clean error, not a stack trace).

### 6.4 No new ingest tests

`SchedulesSchema` already has full test coverage. No new ingest module → no new ingest tests.

### 6.5 Network smoke (no new opt-in)

`SchedulesSchema` already has its column-drift smoke at `tests/test_ingest/test_api_drift.py`. No new network test needed.

### 6.6 Verification gate

Per CLAUDE.md "End-of-effort checklist":

```
pytest -v -k "vegas_team_context or schedules or schemas"
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

Plus, before final report: full `pytest -v` to confirm no cross-module regressions.

### 6.7 Override audit report (one-time, persisted)

`reports/feature_probe_vegas_team_context_override_audit.md` (generated alongside the override parquet by the build script):

- Pooled coverage rate per candidate column.
- Per-(season, position) coverage breakdown.
- Week-1 NaN rate on `season_avg_*` (expected ~6%).
- Unique team-season count per season (expected 32).
- Histogram bounds: min/max/mean of `preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`.
- Sign-convention sanity row: 2023 KC `preseason_spread`, 2023 ARI `preseason_spread` (expected: negative + positive respectively).

This is a convention from PR #25 / PR #28 — not strictly required by the probe runner, but useful for sanity-checking the broadcast logic before staking the probe verdict on the override.

---

## 7. Decision log (in-spec)

| Decision | Rationale |
|---|---|
| Probe-first (not direct integration) | Track 2's 2-of-6 SIGNAL hit rate makes probe-first the disciplined default. PR #21 / PR #26 / PR #27 per-position integrations are the SIGNAL paths. |
| 4-col hybrid bundle: `preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread` | Two mechanism axes (preseason view + as-of-time market view), 2 cols each. Larger bundle than PR #28's 4-col weather, but two distinct hypotheses tested at once. If SIGNAL, the follow-up integration plan can decompose. |
| Week-1 line as preseason proxy (not external May line) | The closest pbp-derivable proxy for Vegas's preseason team view. Saves the cost of new external ingest. Acceptable approximation: week-1 closing line is set 24-48 hours pre-kickoff, after preseason / final-roster effects are priced in. Genuine May line is a follow-up if SIGNAL. |
| `season_avg_*` via `expanding().mean().shift(1)` | Leakage-safe: week-N row sees only weeks 1..N-1 averages. Bye weeks update nothing (no row), which is correct semantics. |
| NaN at week 1 + threshold relaxation to 0.90 (not fill with `preseason_*`) | Filling at week 1 with `preseason_*` would make `preseason_*` and `season_avg_*` identical for ~6% of rows, muddying the two-mechanism-axis structure. Explicit relaxation is cleaner. PR #25's 0.35 relaxation precedent shows the runner accepts documented relaxations. |
| Swap mode drops `implied_team_total` + `spread` (not also `is_home` or `roof_dome`) | The two per-game Vegas cols are the direct mechanism counterparts to the candidate bundle. `is_home` and `roof_dome` are orthogonal mechanism axes (location, environment). |
| Bool features (`is_favored`, `is_high_total`) deliberately excluded from the bundle | Refined-unit territory. Tree models get these natively from the continuous cols; Ridge can't. Adding them dilutes the two preseason / as-of-time mechanism axes. Revisit-only-on-SIGNAL. |
| Eval window 2021–2024 holdout, train 2018–2020 | Matches all prior probes for cross-comparability (PR #20–#29). 2025 extension is a follow-up requiring `refresh_features --seasons 2025`. |
| `--coverage-threshold 0.90` default | Cold-start relaxation; week-1 `season_avg_*` is NaN by construction. PR #28's 0.95 default does not apply here. |
| No new ingest, no schema changes | `SchedulesSchema` already covers `spread_line` / `total_line`. Probe scope is feature compute module + override script + tests + this spec. |
| Override script duplicates `_read_concat` / `_FANTASY_POSITIONS` / etc. from prior probes | Triplicate-extraction is on the project_management.md housekeeping list; folding it into a probe-only spec is scope creep. Replicate the pattern. |
| Audit report (`feature_probe_vegas_team_context_override_audit.md`) generated alongside the override | PR #25 / PR #28 precedent. Sanity-check the broadcast logic + cold-start NaN rate before staking the probe verdict on the override. |

---

## 8. Risks & known limitations

- **Collinearity with baseline cols.** `preseason_implied_team_total` is the week-1 game's `implied_team_total`; for a team's week-1 row, the candidate col equals the baseline col exactly. Ridge will likely split coefficients across the correlated pair. Phase 1 paired-bootstrap ΔRMSE attribution still works (per-row residuals), but augment vs swap may give noticeably different verdicts. Documented; both modes reported.
- **Small effective sample on `preseason_*`.** 32 unique values per season × 4 holdout seasons = 128 unique team-season values, broadcast across ~3000-8000 rows per position. The standard paired bootstrap doesn't account for within-team-season correlation. Same risk as all team-level features (`opp_allowed_*_fppg_l4`); inherit the precedent rather than introduce block-bootstrap-by-team-season for one probe.
- **Cold-start coverage relaxation.** Week-1 `season_avg_*` NaN. Documented; threshold relaxed to 0.90.
- **2025 not in holdout.** The 33d evidence was 2024 *and* 2025. If preseason view is genuinely valuable for elite-season magnitude on 2025, we won't see it on this run. Documented; 2025 extension is a follow-up.
- **Week-1 line as preseason proxy is imperfect.** Closing line on the team's first game is set 24-48 hours pre-kickoff, after all preseason / final-roster effects are priced in. This is *close to* the genuine May view but not identical (a starter going on IR in week 0 would move the week-1 line). Acceptable v1; genuine preseason line is follow-up.
- **Bye-week handling at `season_avg_*`.** Bye weeks produce no schedule row, so the running mean simply doesn't update across a bye. Standard convention, correct semantics. Verified in unit tests.
- **2020 COVID-shortened season.** 2020 is in the training window. Week-1 of 2020 had unusual line conditions (no preseason games → line uncertainty). Not special-cased; treated as a normal training year.
- **Recurring QB augment regression mitigation.** Probe-only; if QB augment regresses again (PRs #23/#24/#25/#28 precedent), document and route around in the integration follow-up — do not block the probe's verdict on it.
- **Wider preseason information loss.** A team's true preseason expectations carry information that doesn't make it into the week-1 closing line — preseason ADP, projected pace, team win total, OC/HC tenure, FA flags. If this probe is NULL, the right next step is *not* "refined unit candidates within this bundle" but rather **external ingest of those non-pbp-derivable signals** (per the 33c TODO).

---

## 9. Predecessor pointers

- `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md` — PR #29 weather-refined spec; most-recent NULL family probe.
- `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md` — PR #28 weather spec; canonical no-new-ingest probe template.
- `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md` — PR #25 spec; most-recent SIGNAL probe.
- `docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md` — feature signal probe infrastructure spec.
- `docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md` — PR #47 spec; the diagnostic whose NO GREENLIGHT verdict motivates this probe.
- `src/projections/features/_shared.py` — `build_game_environment` produces per-team-game `(spread, implied_team_total)` rows; the candidate compute module wraps this.
- `src/projections/features/weather_features.py` — implementation template for the compute-fn / attach / assembler structure.
- `scripts/build_weather_override.py` — CLI script template.
- `scripts/probe_feature_signal.py` — the probe runner (no changes).
- `reports/feature_probe_weather_summary.md` — summary-report template.
- `src/projections/ingest/schedules.py` — source of `spread_line`, `total_line` (no extension required).
