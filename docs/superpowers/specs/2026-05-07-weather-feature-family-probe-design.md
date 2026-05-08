# Weather Feature Family Probe — Design

**Date:** 2026-05-07
**Branch:** `feat/probe-weather`
**Status:** spec
**Predecessor:** PR #25 (trajectory family probe — verdict SIGNAL via WR/TE) → PR #26 (WR trajectory integration, ADOPT) → PR #27 (TE trajectory integration, ADOPT). Trajectory's trailing-8-game-unit branch is closed at all three of PR #25's ADOPT cells. Per project_management.md "Next action" (post-PR-#27), weather is the natural next family-probe slot — sibling to trajectory in the original Track 2 brainstorm; queued under TODO #25.

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether the **weather feature family** carries orthogonal signal beyond v1 + the already-shipped PR #21 RB PBP cols + PR #26/#27 trajectory cols, on the production model classes (BaselineModel + lgb-nb composite).

The weather family targets a **mechanism axis the existing model does not see**: per-game environmental conditions (wind, temperature, surface). Today's per-position feature builders already consume `roof_dome: bool` (derived in `_shared.build_game_environment`, which collapses `roof in {'dome', 'closed'}` to `True`), but the underlying `wind`, `temp`, and `surface` columns from `SchedulesSchema` (already ingested at `src/projections/ingest/schedules.py`) do not reach the per-position feature schemas.

### 1.2 Family-level prior framework

Per project_management.md (2026-04-30 "PBP Feature Family Probe"), the canonical workflow is:

1. Bundle 3–4 mechanism-distinct candidates per family probe.
2. Run the probe at BaselineModel + lgb-nb composite × augment + swap modes.
3. SIGNAL → greenlight a per-position integration plan (analogous to PR #20 → PR #21, PR #25 → PR #26 / PR #27).
4. NULL → close the family at this unit; refined-unit candidates are unlikely to clear absent independent evidence.

Track 2 verdict pattern to date: 2 of 5 family probes returned SIGNAL — PR #20 (RB PBP team features, integrated in PR #21 at -0.0124 fpts) and PR #25 (trajectory, integrated in PR #26 / PR #27 at -0.0371 fpts WR / -0.0090 fpts TE-lgbnb). Three returned NULL durable (PR #22 receiver air-yards/aDOT, PR #23 red-zone, PR #24 pressure). Mean prior for SIGNAL on a single family probe is ~40% — slightly above PR #25's "~25%" framing because the trajectory hit moves the running rate.

Mechanism prediction for weather: **QB / WR / TE pass-volume cells should benefit most**; RB rushing relatively insensitive (rushing is the carry-over fallback in bad weather, so it absorbs less of the suppression). Cold-weather and high-wind games disproportionately hurt downfield passing.

### 1.3 Success criteria

The probe **ships** when all four pass. The verdict (ADOPT/NULL) is informational, not a ship gate.

1. **Coverage:** per-(position, season) override coverage ≥95% over the 2021–2024 eval window. Dome-fill (§3.5) ensures all 4 features are non-null on the ~30% of games played indoors. Outdoor missing-data is empirically rare (`<2%` expected); pooled coverage projected ≥98%, default `--coverage-threshold 0.95` should pass without relaxation.
2. **Probe completeness:** 4 reports per family — BaselineModel × augment, BaselineModel × swap, lgb-nb composite × augment, lgb-nb composite × swap. All complete without error. Phase 1 (per-stat pooled) and Phase 2 (composite via `--force-composite`) verdicts rendered to the per-(model, mode) reports + summary report following the PR #25 template.
3. **Both model classes tested at composite** — Phase 1 is RidgeCV-only regardless of `--model`, so bare lgb-nb runs are tautological with baseline. Re-run lgb-nb with `--force-composite` to actually test lgb-nb at composite (PR #22 §3.2 spec gap; standardized since PR #25).
4. **Verification gates green:** mypy strict + ruff + ruff format clean across `src/`, `scripts/`, `tests/`. Relevant pytest subset clean.

### 1.4 Out of scope (deferred)

- **Production integration.** A SIGNAL verdict greenlights a separate per-position integration plan (extending each `*FeaturesSchema` + builder to consume the 4 weather cols, à la PR #21 / PR #26). This branch is probe-only.
- **Schema-extension precipitation feature.** `nfl_data_py.import_schedules` returns a curated set of weather columns (`temp`, `wind`, `roof`, `surface`); precipitation is not among them. Adding precipitation would require a new ingest source (e.g., NOAA historical hourly keyed on stadium lat/lon) — out of scope for a probe-first family check.
- **Mid-game weather changes.** Source data is game-start only. Wind direction shifts and mid-game rain are not in `import_schedules`.
- **Refined-unit candidates** — kickoff hour / time-of-day, cold-weather threshold (`is_cold_weather`), `surface × position` interactions, week-of-season as a cold-weather proxy, multi-game weather averages, wind-direction encoding. Tree models (lgb-nb) capture non-linearities and threshold-style interactions natively from the continuous `wind_speed_mph` and `temperature_f`; Ridge baseline gets `is_high_wind` as the explicit non-linearity. Refined units are revisit-only-on-SIGNAL territory.
- **Recurring QB augment regression mitigation.** PRs #23, #24, #25 each saw QB augment regress on context / team / trajectory adds; PR #26 / PR #27 chose not to integrate QB-side trajectory cols partly on this evidence. Probe-only here; if QB augment regresses again, document and route around in any integration follow-up — do not block the probe's verdict on it.
- **Stadium altitude / climate baselines** (Mile High effect, hot-weather domes). Not encoded; per-game `temperature_f` partially captures this when outdoors.

---

## 2. Source data — already ingested

`SchedulesSchema` (in `src/projections/schemas.py`) already declares the four columns this probe needs:

| Column | Type | Nullable | Source | Notes |
|---|---|---|---|---|
| `wind` | `Series[int]` | yes | `import_schedules.wind` | Mph, integer. Null for dome / closed-roof games. |
| `temp` | `Series[int]` | yes | `import_schedules.temp` | Fahrenheit, integer. Null for dome / closed-roof games. |
| `roof` | `Series[str]` (`pyarrow`) | yes | `import_schedules.roof` | Values include `dome`, `closed`, `open`, `outdoors`. Null indicates upstream data quality issue. |
| `surface` | `Series[str]` (`pyarrow`) | yes | `import_schedules.surface` | Values include `grass`, `fieldturf`, `a_turf`, `matrixturf`, `sportturf`, etc. Rarely null. |

**No new ingest is required.** The existing `refresh_schedules` already writes these columns into `data/raw/schedules/season=YYYY/part.parquet`. The probe override script reads schedule partitions via `store.read_partition` and computes the four features from these source columns directly.

This is a meaningful simplification vs PR #25 (which added `import_draft_picks` ingest end-to-end) and PR #20 / #23 / #24 (which all relied on the PR #20 PBP plumbing). The weather probe touches the override script, the probe runner, and a thin `weather_features.py` compute module — that's it.

---

## 3. Weather feature definitions

Four mechanism-distinct features, all `pa.Column(pa.Float64, nullable=True)` in the override schema (booleans encoded as `Float64` 0.0/1.0 for ML-compatible dtype consistency, mirroring PR #25's `is_rookie` decision).

### 3.1 `wind_speed_mph`

- Continuous, position-agnostic.
- Definition: `float(schedule.wind)`. Dome / closed roof → `0.0` (per §3.5).
- Mechanism: passing efficiency (and downfield passing in particular) suppressed at high wind. Continuous encoding captures the linear effect at all wind levels; threshold encoding (§3.2) captures the regime-change non-linearity.
- Compute: `compute_weather_features(player_team_week_index, schedules) -> DataFrame` reads `schedule.wind` per `(season, week, home_team, away_team)` and joins to the player-team-week index on `(season, week, team or opp)`.

### 3.2 `is_high_wind`

- Binary (encoded `Float64` 0.0/1.0), position-agnostic.
- Definition: `1.0` if `wind_speed_mph >= 20.0`, `0.0` if `wind_speed_mph < 20.0`, `NaN` if `wind_speed_mph` is NaN (outdoor-data-quality propagation). Dome / closed roof → `0.0` (`wind_speed_mph` is filled to `0.0` on dome rows; the threshold falls out naturally). Implementation note: a naive `(wind_speed_mph >= 20.0).astype(float)` evaluates `NaN >= 20.0` as `False` → 0.0, masking outdoor-NaN rows; the compute fn uses an explicit NaN-preserving branch.
- Threshold rationale: 20 mph matches the QB / kicker rule of thumb for "really windy" — historically the cut at which passing yards drop ~10–15% and FG range tightens materially. The choice trades coverage (fewer True rows at 20 mph) for signal density (the True rows are more clearly affected). Lower cuts (15 mph) increase True-row count but dilute the regime-change signal.
- The bundle's two wind encodings (continuous + threshold) are correlated by construction. This is intentional for a probe — Ridge sees both encodings and picks coefficients; lgb-nb may use either or both as splits. Phase 2 composite verdicts attribute to the 4-column bundle, not to individual columns, so within-mechanism correlation does not muddy the family-level verdict.

### 3.3 `temperature_f`

- Continuous, position-agnostic.
- Definition: `float(schedule.temp)`. Dome / closed roof → `70.0` (per §3.5).
- Mechanism: cold weather suppresses passing efficiency (ball harder to grip, wind chill on grip strength) and lowers scoring overall. Linear encoding is what Ridge can use; lgb-nb can find a `temp < 32` style threshold split natively.
- A cold-threshold companion (`is_cold_weather`, `temp < 32`) is **not** included. The two-encodings-per-mechanism pattern from §3.1 / §3.2 was applied to wind (the dominant headline mechanism) but not to temperature, since temperature's effect is more diffuse and Ridge's linear coefficient should pick up most of it. Refined-unit follow-up territory if SIGNAL.

### 3.4 `is_grass_surface`

- Binary (encoded `Float64` 0.0/1.0), position-agnostic.
- Definition: `1.0` if `surface == 'grass'`, else `0.0`. Null `surface` → `0.0` (informational; expected `<1%` of rows). Dome surfaces are virtually always turf-coded; this is correctly captured.
- Mechanism: surface affects skill-position output via cut-back angles and break-away speed. Grass historically slower; turf historically faster. Effects are typically modest (~1–3% on yards-after-catch metrics) but persistent. Mechanism-distinct from wind/temp.
- Note: `surface` codes vary across seasons (`fieldturf` / `a_turf` / `matrixturf` / `sportturf`); `is_grass_surface` collapses all non-grass codes to 0.0. A multi-class encoding (one bool per surface code) is refined-unit territory.

### 3.5 Dome / closed-roof handling

For any row where `roof in {'dome', 'closed'}`:
- `wind_speed_mph = 0.0`
- `temperature_f = 70.0`
- `is_high_wind = 0.0` (falls out of `wind_speed_mph = 0.0`)
- `is_grass_surface` uses the actual `surface` code (no override)

Rationale: a controlled dome literally has no weather, so `(0, 70, False)` is **semantically correct** for the four cells — not "imputed missing." The model already has `roof_dome` (already in `{Qb,Rb,Wr,Te}FeaturesSchema`) to separate indoor from outdoor rows if the per-position model wants to. Without the fill, ~30% of rows would be NaN on three of four weather columns, baseline can't ingest NaN, and probe coverage would drag to ~70% for no good reason.

For all other `roof` values (typically `open`, `outdoors`, etc.) **and** for null `roof`: treat as outdoor — no fill, NaN propagates if `wind` or `temp` is NaN. This mirrors `build_game_environment`'s exact pattern (`schedules["roof"].isin(["dome", "closed"]).fillna(False)`); we deliberately avoid enumerating outdoor codes so future code changes upstream do not silently miscategorize. Outdoor-NaN rate (data quality issue) is expected `<2%`; the override script logs the actual rate, and the probe's coverage check flags it if material.

### 3.6 Schema integration deferred

Like Track 2A's probe-only specs, the four columns live in the *override parquet*, not the position schemas yet. A SIGNAL verdict triggers a follow-up plan to:
- Add the four cols to each `{Qb,Rb,Wr,Te}FeaturesSchema`.
- Extend `_shared.build_game_environment` to also emit `wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface` per team-game row (it already produces the per-team-game shape: one row per team per game with `roof_dome`). The integration follow-up adds the four columns to that helper's output and to each `*FeaturesSchema`; no new attach helper needed.
- Update `BaselineModel._{POS}_FEATURE_COLUMNS` (the spec-gap fixed in PR #21 — hardcoded tuples per position; lightgbm derives features from the schema dynamically and would auto-pick-up).
- Refresh feature caches under `data/features/{position}/`.
- Run the dual-run adoption gate (`scripts/adoption_gate.py --baseline-run ... --candidate-run ...`).

---

## 4. Module structure

### 4.1 New files

```
src/projections/features/weather_features.py    # compute fn + assembler
scripts/build_weather_override.py                # CLI override generator
tests/test_features/test_weather_features.py     # ~12 feature unit tests
tests/test_scripts/test_build_weather_override_cli.py  # 4 CLI tests
```

### 4.2 Modified files

```
CONTRIBUTING.md                                  # +"Regenerating the weather override" subsection (optional; skip if other override docs are silent)
```

No schema changes (probe-only). No ingest changes (`SchedulesSchema` already covers the source columns).

### 4.3 `weather_features.py` interface

Pattern matches PR #24 / PR #25 (each `compute_*` returns every-week-at-once frames; the assembler merges onto the player-team-week index in one pass).

```python
# Per-(season, week, team_pair) features. Weather is a game-level attribute, so
# each schedule row produces one feature row per team in the matchup. The
# assembler joins these onto the player-team-week index on (season, week, team).
def compute_weather_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Returns (season, week, team, wind_speed_mph, is_high_wind,
    temperature_f, is_grass_surface) — one row per (game, team), so each
    schedule row produces two output rows (home + away).

    Dome / closed-roof rows are filled per §3.5: wind_speed_mph=0.0,
    temperature_f=70.0, is_high_wind=0.0. is_grass_surface uses the actual
    surface code in all cases.

    Outdoor rows with NaN wind/temp leave those cols NaN; the assembler's
    left-merge propagates NaN to the override.
    """
    ...

# Public assembler — joins weather features onto the player-team-week index.
def attach_weather_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Returns a copy of index with 4 nullable-Float64 cols appended:
    wind_speed_mph, is_high_wind, temperature_f, is_grass_surface.

    Index must have columns (gsis_id, season, week, team, opp, position).
    The merge is left on (season, week, team); rows without a matching
    schedule entry retain NaN in all 4 weather cols (the assembler logs
    the rate of unmatched rows).
    """
    ...

# Public assembler — top-level entry point used by the override script.
def build_weather_overrides(
    schedules: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Returns the override frame: (gsis_id, season, week, position,
    wind_speed_mph, is_high_wind, temperature_f, is_grass_surface).
    One row per input index row.

    Raises:
        ValueError: malformed index or duplicate (gsis_id, season, week, position) keys.
    """
    ...
```

### 4.4 `build_weather_override.py` CLI

Mirrors `scripts/build_trajectory_override.py`:
- Args: `--seasons RANGE`, `--data-root PATH`, `--output PATH`.
- Reads schedules + weekly_stats via `store.read_partition`.
- Builds the player-team-week index for each season's eval window (positions filtered to QB / RB / WR / TE).
- Calls `build_weather_overrides`, writes the output parquet.
- Logs the dome-fill rate, outdoor-NaN rate, `is_high_wind` rate, `is_grass_surface` rate to stdout (and optionally to a sidecar audit report; see §6.7).

Acknowledged carry-over from PR #20 / #23 / #24 / #25: `_read_concat`, `_FANTASY_POSITIONS`, `_build_player_team_week_index`, `_parse_season_range` are duplicated across all five existing override scripts. Extraction to `scripts/_override_common.py` is captured in the existing project_management.md housekeeping list. The weather script will copy the same patterns; folding the extraction into a probe-only spec is scope creep.

---

## 5. Probe protocol

### 5.1 Modes & model classes

Four reports per family — same matrix as PR #25:

| Model | Mode | Phase 1 | Phase 2 | Output |
|---|---|---|---|---|
| BaselineModel | augment | yes (RidgeCV per stat) | composite (default; via summary verdict) | `feature_probe_weather_baseline_augment.{md,csv}` |
| BaselineModel | swap | yes (RidgeCV per stat) | composite | `feature_probe_weather_baseline_swap.{md,csv}` |
| lgb-nb composite | augment | tautological (RidgeCV; informational only) | composite via `--force-composite` | `feature_probe_weather_lgbnb_augment.{md,csv}` |
| lgb-nb composite | swap | tautological | composite via `--force-composite` | `feature_probe_weather_lgbnb_swap.{md,csv}` |

### 5.2 Probe invocation

For each (model, mode) cell:
```
python scripts/probe_feature_signal.py \
    --override data/features_probe/weather.parquet \
    --override-cols wind_speed_mph is_high_wind temperature_f is_grass_surface \
    --mode {augment,swap} \
    --model {baseline,lgbnb} \
    --positions QB RB WR TE \
    --seasons 2021 2022 2023 2024 \
    [--coverage-threshold 0.95] \
    [--force-composite]  # only on lgbnb runs
```

In swap mode, `--drop` lists no existing-column counterparts (weather features have no v1 equivalents — this is a brand-new mechanism axis). Swap effectively *adds* the override cols only when they are present in the override (matches PR #25's brand-new-feature swap semantics).

### 5.3 Eval window

- Train: 2018–2020.
- Holdout: 2021–2024.
- Matches all prior probes for cross-comparability.

### 5.4 Coverage threshold

- Default: `--coverage-threshold 0.95`.
- Expected pooled coverage: ≥98% (no cold-start window — weather is per-game, not trailing-N — so the 2018 cold-start drag that affected PR #25 trajectory does not apply here).
- Fallback: relax to 0.90 if pooled coverage drags below 0.95 due to outdoor-missing-data rate exceeding the projected `<2%`. Document any relaxation explicitly in the summary report.

### 5.5 Summary report template

`reports/feature_probe_weather_summary.md` follows the PR #25 template:
- Decision log (commits + reasoning).
- Per-mode verdict table (Phase 1 SIGNAL count + Phase 2 ADOPT/MARGINAL/DO_NOT_ADOPT verdicts).
- Mechanism annotation (which feature, if any, drove signal; or which `target_stat` cells regressed). Specifically: did wind features fire on QB / WR / TE pass-volume cells? Did temperature fire? Did surface fire on yards-after-catch-leaning cells (RB receiving, WR / TE)?
- Coverage note + threshold relaxation if applied.
- Refined-unit candidates left unexplored (precipitation, kickoff hour, multi-class surface, cold-threshold).
- Recurring QB augment regression check — does the pattern from PR #23 / PR #24 / PR #25 recur on weather?

---

## 6. Testing strategy

### 6.1 Per-feature unit tests (`tests/test_features/test_weather_features.py`, ~12 tests)

For each of the 4 features (wind_speed_mph, is_high_wind, temperature_f, is_grass_surface), 3 tests:
1. Correct computation on a representative outdoor synthetic schedule fixture (open roof, known wind / temp / surface).
2. Dome fill: row with `roof='dome'` produces `wind_speed_mph=0.0, is_high_wind=0.0, temperature_f=70.0`; `is_grass_surface` reflects the actual surface code.
3. NaN propagation: row with `roof='outdoors'` and `wind=NaN` produces `wind_speed_mph=NaN, is_high_wind=NaN`; analogous for `temp=NaN` → `temperature_f=NaN`. (`is_grass_surface` defaults to `0.0` on null `surface`.)

### 6.2 Override assembler tests (~3 tests)

- Coverage check on a representative fixture with mixed indoor / outdoor / NaN rows.
- Home + away rows both produced from a single schedule row, and both join correctly to the player-team-week index.
- Schema validation holds (override parquet round-trips through pandera).

### 6.3 CLI tests (`tests/test_scripts/test_build_weather_override_cli.py`, 4 tests)

- argparse smoke (parse_args with valid args).
- Default invocation writes expected file structure.
- Custom `--seasons` range respected.
- Error on missing `data_root` (clean error, not a stack trace).

### 6.4 No new ingest tests

`SchedulesSchema` already has full test coverage. No new ingest module → no new ingest tests.

### 6.5 Network smoke (no new opt-in)

`SchedulesSchema` already has its column-drift smoke at `tests/test_ingest/test_api_drift.py`. No new network test needed for weather.

### 6.6 Verification gate

Per CLAUDE.md "End-of-effort checklist":

```
pytest -v -k "weather or schedules or schemas"
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

Plus, before final report: full `pytest -v` to confirm no cross-module regressions.

### 6.7 Override audit report (one-time, persisted)

`reports/feature_probe_weather_override_audit.md` (generated alongside the override parquet by the build script):
- Pooled dome-fill rate (expected ~30%).
- Pooled outdoor-NaN rate per column (expected `<2%`).
- Pooled `is_high_wind` rate (expected single-digit %).
- Pooled `is_grass_surface` rate (varies by year; ~40–50% expected).
- Per-(season, position) coverage breakdown.

This is a convention from PR #25 (whose override surfaced the `draft_year_inferred` audit column rate); not strictly required by the probe runner, but useful for sanity-checking the dome-fill decision before staking the probe verdict on it.

---

## 7. Decision log (in-spec)

| Decision | Rationale |
|---|---|
| Probe-first (not direct integration) | Track 2's 2-of-5 SIGNAL hit rate makes probe-first the disciplined default. PR #21 / PR #26 / PR #27's per-position integrations are the SIGNAL paths. |
| Hybrid bundle: `wind_speed_mph` (cont) + `is_high_wind` (bool, 20 mph) + `temperature_f` (cont) + `is_grass_surface` (bool) | Two encodings on the dominant mechanism (wind), one each on temp + surface. Continuous encodings let Ridge fit the linear effect; threshold encoding captures the wind regime-change non-linearity that Ridge cannot infer. lgb-nb finds non-linearities natively from any encoding. Skips precipitation (unverified availability + new ingest) and cold-threshold (refined-unit territory). |
| Dome fill: `wind=0, temp=70` for `roof in {dome, closed}` | Semantically correct for a controlled environment, not "imputed missing." Preserves coverage for ~30% of rows that would otherwise be NaN. The model already has `roof_dome` to separate indoor / outdoor if it wants. |
| `is_high_wind` cut at 20 mph (not 15 or 25) | 20 mph matches the QB / kicker rule of thumb for "really windy." Lower cuts (15) increase True-row count but dilute the regime-change signal; higher cuts (25) leave too few True rows for Ridge to learn from. |
| Bool features encoded `Float64` 0.0/1.0 | ML-compatible schema dtype across pandera + LightGBM + RidgeCV without coercion gymnastics. Matches PR #25's `is_rookie` precedent. |
| `is_grass_surface` collapses all non-grass codes to 0.0 | Multi-class surface encoding (one bool per code) is refined-unit territory. The single bool tests the binary grass-vs-everything-else mechanism cleanly. |
| Schema integration deferred (probe-only) | Track 2 pattern; SIGNAL verdict greenlights integration plan with adoption gate. Avoids feature-cache invalidation churn for a NULL-likely family. |
| Eval window 2021–2024 holdout, train 2018–2020 | Matches all prior probes for cross-comparability (PR #20–#25). |
| `--coverage-threshold 0.95` default | No cold-start window for weather (per-game feature, not trailing-N), so PR #25's 0.35 relaxation does not apply. Default 0.95 should pass; relax to 0.90 fallback if outdoor-NaN rate exceeds projection. |
| Override script duplicates `_read_concat` / `_FANTASY_POSITIONS` / etc. from PR #20 / #23 / #24 / #25 | Triplicate-extraction is on the housekeeping list; folding it into a probe-only spec is scope creep. Replicate the pattern. |
| No new ingest, no schema changes | `SchedulesSchema` already covers the source columns. The probe scope is `weather_features.py` + `build_weather_override.py` + tests + this spec. |
| Audit report (`feature_probe_weather_override_audit.md`) generated alongside the override | Sanity-check dome-fill rate and outdoor-NaN rate before staking the probe verdict on the dome-fill design decision. PR #25 precedent (audit on `draft_year_inferred`). |

---

## 8. Open questions / known limitations

- **Wind direction not encoded.** A 20 mph headwind on a downfield-throw drive is more punishing than 20 mph perpendicular to the field. The source data does not include direction; absolute wind speed is the best available proxy. Acceptable v1 limitation; not expected to be material at the probe level (the speed coefficient already absorbs an averaged direction effect).
- **Stadium-specific wind anomalies not encoded.** Heinz Field / Soldier Field / Bills Stadium have known wind-tunnel effects that aren't captured by absolute mph alone. Refined-unit territory if SIGNAL.
- **Cold-weather vs warm-weather team home/away interaction not encoded.** A Vikings-vs-Saints game in January at Lambeau is `temp=20` for both teams; the visiting Saints' players may be more affected than the cold-acclimated Packers (visiting players' bodies). Per-team weather features would split this; not in v1.
- **`closed` roof handling.** Some retractable-roof stadiums report `roof=closed` for cold-weather games (e.g., AT&T Stadium in Dallas closes the roof when temp < 50). We treat `closed` identically to `dome` (fill 0/70). The actual outdoor temperature on those days is informational but not used. Acceptable v1 — `closed` rows will look like dome rows to the model, which is approximately right for game-time conditions.
- **Surface durability across a season not encoded.** Late-season grass fields chew up; the same `surface=grass` code in week 1 vs week 17 may represent very different actual surfaces. Acceptable v1 limitation.
- **No explicit handling of the COVID-shortened 2020 season.** 2020 is in the training window. Weather data for 2020 is consistent with prior years; no special handling needed.
- **No explicit handling of preseason / playoffs.** Probe runs against regular-season weeks only (per the player-team-week index built from `weekly_stats` partitions, which are regular-season).

---

## 9. Predecessor pointers

- `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md` — PR #25 spec (most-recent SIGNAL probe; canonical template).
- `docs/superpowers/specs/2026-05-02-pbp-pressure-feature-family-probe-design.md` — PR #24 spec.
- `docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md` — PR #23 spec.
- `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md` — PR #20 spec (the SIGNAL-returning predecessor for Track 2A).
- `docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md` — feature signal probe infrastructure spec.
- `src/projections/features/trajectory_features.py` — implementation template for the compute-fn / attach / assembler structure.
- `scripts/build_trajectory_override.py` — CLI script template.
- `reports/feature_probe_trajectory_summary.md` — summary-report template.
- `src/projections/ingest/schedules.py` — source of `wind`, `temp`, `roof`, `surface` (no extension required).
- `src/projections/features/_shared.py` — `build_game_environment` already pulls `roof_dome` from the schedules join (collapsing `dome` + `closed` to True); weather-features integration plan would extend this helper to emit wind / temp / surface columns alongside `roof_dome` if SIGNAL.
