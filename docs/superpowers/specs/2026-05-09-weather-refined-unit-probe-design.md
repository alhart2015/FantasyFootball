# Weather Refined-Unit Family Probe — Design

**Date:** 2026-05-09
**Branch:** `feat/probe-weather-refined`
**Status:** spec
**Predecessor:** PR #28 (broad-cut weather family probe — verdict SIGNAL via lgb-nb augment composite, RB + WR ADOPT) → PR #29 (RB+WR weather features integration — verdict ADOPT both binding cells; lgb-nb production routing unchanged). PR #28's PM entry left seven refined-unit candidates open under TODO #25, with recommended priority order **cold-weather threshold → multi-class surface → kickoff hour**. This probe takes the top three and bundles them into one family probe, mirroring the Track 2A workflow.

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether **three refined-unit weather features** carry orthogonal signal beyond v1 + the already-shipped PR #29 RB/WR weather cols (`wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface` in `RbFeaturesSchema` + `WrFeaturesSchema`). The refined unit targets three distinct mechanism axes:

- **Cold-weather threshold (`is_cold_weather`)** — a sharp regime indicator at `temp ≤ 32°F` (freezing point), sibling-shape to `is_high_wind` at 20 mph. Captures the frozen-field / cleats-don't-grip / cold-shifts-toward-rushing regime that linear `temperature_f` and the binary `is_grass_surface` together cannot encode.
- **Multi-class surface one-hot** — replaces v1's binary `is_grass_surface` (grass vs all-other-surfaces) with one bool per distinct `schedules.surface` code. Tests whether different turf types (`fieldturf`, `sportturf`, `matrixturf`, `a_turf`, etc.) carry distinguishable signal that the binary collapse hides.
- **Kickoff hour primetime indicator (`is_primetime`)** — a single bool at `kickoff_hour_et ≥ 18`. Captures the TNF/SNF/MNF cohort (~13% of games) — rested players (extra rest day for Thursday/Monday teams), spotlight/officiating bias, late-season cold-weather correlation.

This is the **first refined-unit family probe in the project** (Track 2A's prior probes have all been broad-cut family probes; refined-unit candidates have remained on the deferred-revisit list under their respective TODOs).

### 1.2 Family-level prior framework

Per project_management.md (2026-04-30 "PBP Feature Family Probe"), the canonical workflow is:

1. Bundle 3–4 mechanism-distinct candidates per family probe.
2. Run the probe at BaselineModel + lgb-nb composite × augment + swap modes.
3. SIGNAL → greenlight a per-position integration plan.
4. NULL → close the family at this unit; refined-unit candidates are unlikely to clear absent independent evidence.

Track 2 verdict pattern to date: 2 of 5 broad-cut family probes returned SIGNAL — PR #20 (RB PBP team features, integrated in PR #21 at -0.0124 fpts) and PR #25 (trajectory, integrated in PR #26 / PR #27 at -0.0371 fpts WR / -0.0090 fpts TE-lgbnb), plus PR #28 (weather, integrated in PR #29 at -0.0077 RB / -0.0104 WR fpts). Three returned NULL durable (PR #22 receiver air-yards/aDOT, PR #23 red-zone, PR #24 pressure).

**Refined-unit-specific decoding** of the augment vs swap matrix:
- **swap ADOPT, augment ADOPT** → strict refinement preferred (replace v1 weather cols entirely).
- **swap NULL, augment ADOPT** → additive refinement (keep v1, add refined cols on top).
- **swap ADOPT, augment NULL** → signal lives only in refined cols, not on top of v1 → replace.
- **swap NULL, augment NULL** → close the refined unit at this cut; no integration.
- **REGRESSION (CI strictly above 0)** in any cell → flag as recurring-pattern follow-up; does not gate the probe verdict.

Mechanism prediction:
- **`is_cold_weather`**: PR #28's PM noted that RB's unexpected weather signal might be driven by cold-weather games shifting offensive balance toward rushing. A sharp 32°F threshold should activate the regime more cleanly than continuous `temperature_f` alone — expected RB augment ADOPT under lgb-nb composite.
- **Multi-class surface**: PR #28's binary `is_grass_surface` returned RB+WR ADOPT under lgb-nb augment composite. If different turf types carry distinguishable footing/cut-back regimes, the multi-class encoding should improve over binary. Expected swap ADOPT (multi-class beats binary on the same axis).
- **`is_primetime`**: weakest mechanism story. The "rested players" hypothesis is plausible but small in magnitude; "primetime crowd/officiating" is folkloric. May null-out across all positions; included as the third bundle slot to give the probe statistical power and to test the kickoff-hour axis as a unit before a separate refined-unit probe scopes it.

### 1.3 Success criteria

The probe **ships** when all four pass. The verdict (ADOPT/NULL) is informational, not a ship gate.

1. **Coverage:** per-(position, season) override coverage on the 2021–2024 eval window. PR #28's `temperature_f` / `wind_speed_mph` coverage was 67–98% per (position, season) — the same outdoor-NaN profile applies here for `is_cold_weather`. `is_primetime` should be ≥99% (kickoff is rarely missing). Multi-class surface should match v1 `is_grass_surface` coverage (≥99%; surface is rarely null even on legacy seasons). Default `--coverage-threshold 0.90` mirrors PR #28's relaxation.
2. **Probe completeness:** 4 reports per family — BaselineModel × augment, BaselineModel × swap, lgb-nb composite × augment, lgb-nb composite × swap. All complete without error. Phase 1 (per-stat pooled RidgeCV) and Phase 2 (composite via `--force-composite` on lgb-nb) verdicts rendered to per-(model, mode) reports + summary report following the PR #28 template.
3. **Both model classes tested at composite** — Phase 1 is RidgeCV-only regardless of `--model`, so bare lgb-nb runs are tautological with baseline. Re-run lgb-nb with `--force-composite` to actually test lgb-nb at composite (PR #22 §3.2 spec gap; standardized since PR #25).
4. **Verification gates green:** mypy strict + ruff + ruff format clean across `src/`, `scripts/`, `tests/`. Relevant pytest subset clean.

### 1.4 Out of scope (deferred)

- **Production integration.** A SIGNAL verdict greenlights a separate per-position integration plan whose shape is conditional on which mode binds (per §1.2 decoding). This branch is probe-only.
- **Other refined-unit candidates from TODO #25**: surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). Out of scope for this probe; revisit under TODO #25 only on independent evidence.
- **Cold-weather threshold variants** (40°F NFL media-convention cut; 50°F broader cut). Explicitly considered and rejected — 32°F is the only true physical regime change (frozen field surface); the others are conventions or dilute the signal toward what continuous `temperature_f` already captures.
- **`wind_chill` composite** (e.g., `temp - 0.7 × wind`). Considered as a 4th bundle slot and rejected — composite arithmetic complicates feature-importance interpretation, and lgb-nb tree splits already model the wind×temp interaction natively from the four weather cols.
- **`is_late_season` (week ≥ 14)** as a seasonal-cold proxy. Considered as a 4th bundle slot and rejected — collinear with `is_cold_weather` to a degree that dilutes per-feature signal without targeting a structurally different axis.
- **Continuous `kickoff_hour_et`** vs binary `is_primetime`. Continuous is more expressive but breaks the regime-indicator aesthetic of the rest of the bundle. Revisit as a refined-unit candidate only if `is_primetime` returns "directionally right but small" point estimates with CIs bracketing zero.
- **`is_london`** (`kickoff_hour_et < 11`, ~1–2% of games). Mechanism-clean (jet-lag) but cohort too small to power its own gate cell at composite.
- **Recurring QB augment regression mitigation.** PRs #23, #24, #25 each saw QB augment regress; PR #28 saw a milder version. Probe-only here — if QB augment regresses again, document and route around in any integration follow-up; do not block the probe's verdict on it.
- **Stadium-specific surface anomalies** (Heinz Field's notoriously chewed-up grass; AT&T Stadium's sometimes-closed retractable roof). Folded into existing `is_grass` / `surface` codes; not separately encoded.

---

## 2. Source data — already ingested

`SchedulesSchema` (in `src/projections/schemas.py:289`) already declares the four columns this probe needs:

| Column | Type | Nullable | Source | Use |
|---|---|---|---|---|
| `temp` | `Series[int]` | yes | `import_schedules.temp` | Drives `is_cold_weather` (and v1 `temperature_f`). |
| `surface` | `Series[str]` (`pyarrow`) | yes | `import_schedules.surface` | Drives multi-class one-hot (and v1 `is_grass_surface`). |
| `roof` | `Series[str]` (`pyarrow`) | yes | `import_schedules.roof` | Dome / closed-roof predicate for `is_cold_weather` fill. |
| `kickoff` | `Series[pd.DatetimeTZDtype]` (UTC, us) | yes | `import_schedules.kickoff` | Drives `is_primetime`. |

**No new ingest is required.** The existing `refresh_schedules` writes these columns into `data/raw/schedules/season=YYYY/part.parquet`. The override script reads schedule partitions via `store.read_partition` and computes the three new feature axes from these source columns directly.

The exact set of distinct `surface` codes is **pinned during implementation Phase 0** by reading `data/raw/schedules` across 2018–2024 and listing every observed value. This produces the canonical column list (e.g., `is_grass`, `is_fieldturf`, `is_sportturf`, ...). Per §3.2 below, an unseen code in production data raises `ValueError` to force a deliberate spec amendment rather than silent column drift.

---

## 3. Refined-unit feature definitions

Three mechanism-distinct feature axes, all encoded `pa.Column(pa.Float64, nullable=True)` in the override schema (booleans encoded as `Float64` 0.0/1.0 for ML-compatible dtype consistency, mirroring PR #25 / PR #28 conventions).

### 3.1 `is_cold_weather`

- Binary (encoded `Float64` 0.0/1.0), position-agnostic.
- Definition: `1.0` if `temperature_f <= 32.0`, `0.0` if `temperature_f > 32.0`, `NaN` if `temperature_f` is NaN. Comparison direction `<=` mirrors `is_high_wind`'s `>=`. Implementation uses pandas Float64 + NA propagation: `(temp_f <= 32.0).astype("Float64")` (NaN comparisons evaluate to NA → NaN in Float64 output, mirroring the PR #28 `is_high_wind` precedent at `weather_features.py:73`).
- Dome / closed-roof handling: `temperature_f` is filled to `70.0` for `roof in {dome, closed}` per the existing v1 convention (`weather_features.py:67-68`); the threshold falls out naturally to `is_cold_weather = 0.0` for domes, which is **semantically correct** (controlled environment is not cold).
- Threshold rationale: 32°F is the only true physical regime change in the candidate set (40°F is media convention; 50°F dilutes toward what continuous `temperature_f` already captures). Pairs cleanly with `is_high_wind` at 20 mph as a sibling sharp-threshold pair — two boolean regime indicators flanking the continuous `wind_speed_mph` and `temperature_f`.
- Mechanism: PR #28's RB unexpected signal hypothesis (cold-weather games shift offensive balance toward rushing) should activate more cleanly at a 32°F threshold split than via the continuous `temperature_f` linear coefficient.

### 3.2 Multi-class surface one-hot

- One `Float64` boolean per distinct `surface` code observed in `data/raw/schedules` across 2018–2024. Exact column set pinned in Phase 0 of the implementation plan by reading the data; this spec assumes ~5–7 columns based on the codes named in `SchedulesSchema`'s docstring and PR #28's spec §3.4 (`grass`, `fieldturf`, `a_turf`, `matrixturf`, `sportturf`, possibly `dessograss`).
- Naming convention: `is_<sanitized_code>` where `sanitized_code = code.lower().replace("-", "_")` (e.g., `is_grass`, `is_fieldturf`, `is_a_turf`). Exact list committed alongside the override generator script.
- Definition for each `is_<code>`: `1.0` if `surface == "<code>"`, `0.0` if `surface` is a different known code, `NaN` if `surface` is NaN. **Not `0.0` on NaN** — preserve the missing-data signal so the probe's coverage check sees it. (This differs from v1 `is_grass_surface`, which fills NaN to `0.0`; the multi-class encoding is more conservative.)
- The new `is_grass` is bit-identical to v1 `is_grass_surface` only on rows where `surface` is non-NaN; on NaN-`surface` rows, `is_grass` is NaN whereas v1 `is_grass_surface` is `0.0`. For swap mode, this is the right shape — swap drops v1 weather cols (including `is_grass_surface`) and replaces with refined cols (including the more conservative `is_grass`); the model sees the better-encoded missingness on rare null-surface rows.
- Validation: the override generator raises `ValueError` if it encounters a `surface` code in `schedules` that is not in the pinned column set. This forces deliberate spec amendments rather than silent column drift on `nfl_data_py` upstream changes.
- Mechanism: PR #28's binary `is_grass_surface` signaled RB+WR ADOPT under lgb-nb augment composite; if different turf compositions carry distinguishable footing/cut-back/heat-retention regimes, the multi-class encoding should improve over binary (especially in swap mode). lgb-nb composite handles wide one-hot natively (tree splits aren't sensitive to column count).

### 3.3 `is_primetime`

- Binary (encoded `Float64` 0.0/1.0), position-agnostic.
- Definition: convert `schedules.kickoff` (UTC `pd.DatetimeTZDtype`) to America/New_York via `zoneinfo.ZoneInfo("America/New_York")` (handles EDT/EST switch automatically across the Sep–Feb season span). Extract local hour: `kickoff_hour_et = kickoff_local.dt.hour + kickoff_local.dt.minute / 60.0`. Then `is_primetime = (kickoff_hour_et >= 18.0).astype("Float64")`. NaN `kickoff` propagates to NaN `is_primetime`.
- Captures TNF (8:15 ET), SNF (8:20 ET), MNF (8:15 ET), historically ~13% of games. Misses the ~1–2% London early-window cohort (would require a separate `is_london` feature; explicitly out of scope per §1.4).
- Threshold rationale: 18:00 ET is the cleanest cut between Sunday late-window (4:25 ET) and primetime games. A 19:00 cut would miss MNF/TNF/SNF starts. A 17:00 cut would falsely include some Sunday late-window games.
- Mechanism: weakest of the three. "Extra rest day before TNF/MNF" is plausible but small; "primetime spotlight bias" is folkloric. Bundle-power justification — combined with `is_cold_weather` and multi-class surface, it gives the probe three distinct mechanism axes to clear the family-level prior threshold.

### 3.4 Dome / closed-roof handling

Inherited from PR #28 / v1: `roof in {"dome", "closed"}` filled to `wind_speed_mph = 0.0`, `temperature_f = 70.0`. Any row where the dome predicate fires:
- `is_cold_weather = 0.0` (falls out of `temperature_f = 70.0`)
- Multi-class surface: uses the actual `surface` code (no override)
- `is_primetime`: uses the actual `kickoff` (no override; primetime status is independent of dome)

For all other `roof` values **and** for null `roof`: outdoor — no fill, NaN propagates. Mirrors the existing predicate in `weather_features.py:62`.

### 3.5 Override schema (pre-Phase-0)

Approximate output frame (exact surface column set pinned in Phase 0):

```
gsis_id        : StringDtype("pyarrow")
season         : Int64
week           : Int64
position       : StringDtype("pyarrow")
is_cold_weather: Float64 (nullable)
is_grass       : Float64 (nullable)
is_fieldturf   : Float64 (nullable)
is_sportturf   : Float64 (nullable)
is_matrixturf  : Float64 (nullable)
is_a_turf      : Float64 (nullable)
[…possibly more, pinned in Phase 0…]
is_primetime   : Float64 (nullable)
```

Total: 4 identity cols + 1 (cold) + ~5–7 (surface) + 1 (primetime) = ~11–13 columns.

---

## 4. Module structure

### 4.1 Modified files

```
src/projections/features/weather_features.py    # extend compute_weather_features + assembler
scripts/build_weather_override.py                # extend to call updated assembler
tests/test_features/test_weather_features.py    # +~10 unit tests for the 3 new axes
tests/test_scripts/test_build_weather_override.py  # update CLI tests for new column set
CONTRIBUTING.md                                  # update "Regenerating the weather override" subsection
```

### 4.2 New files

None. The probe extends the existing `weather_features.py` module and `build_weather_override.py` script in-place. Output path remains `data/features_probe/weather.parquet` (overwriting PR #28's narrower override).

### 4.3 `weather_features.py` extension

Pattern matches PR #28 — pure compute fns + per-game frame + attach helper + public assembler. Extensions:

```python
# New constants
_COLD_WEATHER_TEMP_F = 32.0
_PRIMETIME_HOUR_ET = 18.0
_KICKOFF_TZ = "America/New_York"

# Pinned in Phase 0 by reading data/raw/schedules across 2018-2024
_SURFACE_CODES: Final[tuple[str, ...]] = (
    "grass", "fieldturf", "sportturf", "matrixturf", "a_turf",
    # ...exact set pinned during implementation
)
_SURFACE_COL_NAMES: Final[tuple[str, ...]] = tuple(
    f"is_{c.lower().replace('-', '_')}" for c in _SURFACE_CODES
)

# New private helpers (pure compute on Series)
def _compute_is_cold_weather(temperature_f: pd.Series) -> pd.Series: ...
def _compute_surface_onehot(surface: pd.Series) -> pd.DataFrame: ...
def _compute_is_primetime(kickoff_utc: pd.Series) -> pd.Series: ...

# compute_weather_features extended to emit the union (existing 4 cols + 3 new axes).
# Public signature unchanged; output frame grows from 7 cols to ~14.
def compute_weather_features(schedules: pd.DataFrame) -> pd.DataFrame:
    # ...existing logic (wind / is_high_wind / temp / is_grass_surface)
    # + new logic (is_cold_weather / surface one-hot / is_primetime)
    ...
```

`attach_weather_features` and `build_weather_overrides` signatures unchanged; they pass through whatever columns `compute_weather_features` produces. The `_REQUIRED_INDEX_COLS` invariant and gsis-id format check carry over from PR #28 unchanged.

### 4.4 `build_weather_override.py` extension

Mirrors the existing pattern at `scripts/build_weather_override.py:96-185`. Updates:
- Audit-print loop iterates over the extended column set (dome rate, outdoor-NaN rate, `is_high_wind` rate, `is_cold_weather` rate, per-surface rate, `is_primetime` rate).
- Default `--output` unchanged (`data/features_probe/weather.parquet` — overwrites PR #28's).
- New CLI flag (optional): `--force` to overwrite existing output. Mirrors PR #28's pattern.

### 4.5 Backward compatibility note

The extended `compute_weather_features` returns a superset of PR #28's columns. PR #29's production builders (`build_rb_features`, `build_wr_features`) call `attach_weather_features`, which selects via column list — they will pick up the existing 4 v1 weather cols and ignore the new ones. **No production behavior change.** Verified by §6.6 verification gate (full pytest including production-builder integration tests).

---

## 5. Probe protocol

### 5.1 Modes & model classes

Four reports per family — same matrix as PR #28:

| Model | Mode | Phase 1 | Phase 2 | Output |
|---|---|---|---|---|
| BaselineModel | augment | yes (RidgeCV per stat) | composite (default) | `feature_probe_weather_refined_baseline_augment.{md,csv}` |
| BaselineModel | swap | yes (RidgeCV per stat) | composite | `feature_probe_weather_refined_baseline_swap.{md,csv}` |
| lgb-nb composite | augment | tautological (informational only) | composite via `--force-composite` | `feature_probe_weather_refined_lgbnb_augment.{md,csv}` |
| lgb-nb composite | swap | tautological | composite via `--force-composite` | `feature_probe_weather_refined_lgbnb_swap.{md,csv}` |

### 5.2 Probe invocation

For each (model, mode) cell (column list extends in Phase 0 once surface codes are pinned):

```
python scripts/probe_feature_signal.py \
    --override data/features_probe/weather.parquet \
    --override-cols is_cold_weather is_grass is_fieldturf is_sportturf \
                    is_matrixturf is_a_turf [...] is_primetime \
    --mode {augment,swap} \
    --model {baseline,lgbnb} \
    --positions QB RB WR TE \
    --seasons 2021 2022 2023 2024 \
    --coverage-threshold 0.90 \
    [--force-composite]  # only on lgbnb runs
```

In **swap mode**, the override-script's swap semantics drop v1 weather cols (`wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface`) from the candidate-side feature set and replace with the refined override cols. Unlike PR #28's swap (which was degenerate — no v1 weather cols existed pre-PR-29), this swap is **non-degenerate** — v1 cols are present in `RbFeaturesSchema` / `WrFeaturesSchema` after PR #29's integration. The swap is a true refined-unit comparison: "are these refined cols better than v1's broad-cut cols?"

For QB and TE, swap mode is functionally identical to "drop nothing, add refined" (no v1 weather cols in `QbFeaturesSchema` / `TeFeaturesSchema`) — same as PR #28's swap shape for those positions. Document this asymmetry in the summary report.

### 5.3 Eval window

- Train: 2018–2020.
- Holdout: 2021–2024.
- Matches all prior probes for cross-comparability.

### 5.4 Coverage threshold

- Default: `--coverage-threshold 0.90` (mirrors PR #28's relaxation).
- Rationale: outdoor-NaN profile is identical to PR #28 (same source columns); the refinement adds `is_primetime` (≥99% coverage expected) and multi-class surface (matches PR #28's `is_grass_surface` coverage). Aggregate coverage should not degrade meaningfully vs PR #28.
- Fallback: relax to 0.80 if the multi-class surface's preserve-NaN behavior (vs v1's fill-to-zero) drags coverage below 0.90 unexpectedly. Document any further relaxation explicitly in the summary report.

### 5.5 Summary report template

`reports/feature_probe_weather_refined_summary.md` follows the PR #28 template:
- Decision log (commits + reasoning).
- Per-mode verdict table (Phase 1 SIGNAL count + Phase 2 ADOPT/MARGINAL/DO_NOT_ADOPT verdicts).
- Refined-unit-specific outcome decoding per §1.2 (which mode×model combination binds; what integration shape the probe greenlights).
- Mechanism annotation: did `is_cold_weather` fire on RB? Did multi-class surface fire on RB+WR (where binary `is_grass_surface` already did in PR #28)? Did `is_primetime` fire anywhere?
- Coverage note + threshold relaxation if applied beyond 0.90.
- Recurring QB augment regression check (per PR #23/#24/#25/#28 pattern).
- Refined-unit-of-refined-unit candidates left unexplored (continuous kickoff hour, `is_london`, surface×position interactions, per-team weather acclimation).

---

## 6. Testing strategy

### 6.1 Per-feature unit tests (`tests/test_features/test_weather_features.py`, ~10 new tests)

**`is_cold_weather` (~3 tests):**
1. Boundary: `temp == 32` → 1.0; `temp == 33` → 0.0; `temp == 31` → 1.0.
2. Dome fill: `roof == "dome"` → `temperature_f = 70.0` → `is_cold_weather = 0.0`. Same for `roof == "closed"`.
3. NaN propagation: `roof == "outdoors"` and `temp == NaN` → `is_cold_weather = NaN`.

**Multi-class surface one-hot (~4 tests):**
1. Each pinned surface code produces the correct `is_<code>` column at 1.0 and zeros elsewhere on a representative fixture.
2. Sum of all surface bools per row equals 1.0 on non-NaN-surface rows; equals NaN on NaN-`surface` rows.
3. Unknown / unseen surface code raises `ValueError` from `compute_weather_features`.
4. `is_grass` matches v1 `is_grass_surface` on non-NaN-surface rows; differs on NaN-`surface` rows (refined preserves NaN; v1 fills to 0.0).

**`is_primetime` (~3 tests):**
1. `kickoff_hour_et >= 18` produces 1.0 on a known SNF/MNF/TNF fixture (8:15 PM ET).
2. EDT/EST switch correctness: a Sep game at 1:00 PM ET (EDT, UTC-4) and a Nov game at 1:00 PM ET (EST, UTC-5) both produce `is_primetime = 0.0`. A primetime game in each timezone produces 1.0.
3. NaN propagation: `kickoff == NaT` → `is_primetime = NaN`.

### 6.2 Override assembler tests (~2 new tests)

- Output frame has the expected extended column set (identity + ~9 feature cols).
- Row-count invariant still holds (`build_weather_overrides`'s assert: `len(attached) == len(player_team_week_index)`).

### 6.3 CLI tests (`tests/test_scripts/test_build_weather_override.py`, update existing)

- argparse smoke unchanged.
- Output column set assertion updated to reflect the extended set.
- Audit-output assertion updated to include the new rates (`is_cold_weather` rate, per-surface rates, `is_primetime` rate).

### 6.4 No new ingest tests

`SchedulesSchema` already has full coverage. No new ingest module → no new ingest tests.

### 6.5 Network smoke (no new opt-in)

`SchedulesSchema` column-drift smoke at `tests/test_ingest/test_api_drift.py` covers the source columns. No new network test needed.

### 6.6 Verification gate

Per CLAUDE.md "End-of-effort checklist":

```
pytest -v -k "weather or schedules or schemas"
pytest -v -k "ingest or store or schemas"  # CLAUDE.md mandate: schema/ingest seam coverage
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

Plus, before final report: **full `pytest -v`** to confirm no cross-module regressions (especially production-builder tests, which still call `attach_weather_features` for the existing 4 v1 cols).

### 6.7 Override audit report (one-time, persisted)

`reports/feature_probe_weather_refined_override_audit.md` — generated alongside the override parquet by the build script:
- Pooled dome-fill rate (expected ~30%, unchanged from PR #28).
- Pooled outdoor-NaN rate per source column (expected ~8% on `temp`, matching PR #28).
- Pooled `is_cold_weather` rate (expected single-digit %, depends on 2018–2024 cold-weather frequency).
- Per-surface rates (expected `grass` ~50%, modern turfs distributed across the rest).
- Pooled `is_primetime` rate (expected ~12–15%).
- Per-(season, position) coverage breakdown for the eval window.

This is convention from PR #25 / PR #28; not strictly required by the probe runner, but useful for sanity-checking the bundle composition before staking the verdict on it.

---

## 7. Decision log (in-spec)

| Decision | Rationale |
|---|---|
| Refined-unit family probe (not direct integration) | Track 2A's probe-first norm. Refined-unit candidates have not yet been tested on this codebase; probe-first establishes pattern viability before committing to integration. |
| Bundle of three: `is_cold_weather` + multi-class surface + `is_primetime` | Three mechanism-distinct axes hitting the top of TODO #25's recommended priority order. At lower edge of "3-4 candidates" Track 2 norm; size justified by each candidate having a clean independent mechanism story (no filler 4th). Considered and rejected: `wind_chill` composite (composite arithmetic obscures interpretation), `is_late_season` (collinear with cold), continuous `kickoff_hour_et` (breaks regime-indicator aesthetic). |
| Cold-weather threshold at 32°F (not 40 or 50) | Only true physical regime change (frozen field surface). 40°F is media convention; 50°F dilutes toward what continuous `temperature_f` captures. Pairs cleanly with `is_high_wind` at 20 mph as sibling sharp-threshold pair. |
| Multi-class surface as per-distinct-code one-hot (not aggregated buckets) | Bucketing requires a hand-pinned domain-knowledge mapping (which turf type is "modern" vs "legacy"?) — error-prone; if `fieldturf` and `matrixturf` actually behave differently, lumping them throws away signal. lgb-nb composite handles wide one-hot natively. |
| Multi-class surface preserves NaN (vs v1's fill-to-zero on NaN-`surface`) | More conservative encoding for a probe-only context; the probe coverage check sees missing data accurately. Differs from v1 `is_grass_surface` only on NaN-`surface` rows (rare). |
| `is_primetime` as binary at 18:00 ET (not continuous hour) | Sibling shape to `is_high_wind` / `is_cold_weather` regime indicators. Continuous breaks bundle aesthetic. London-game cohort too small (1–2%) for separate `is_london` feature at composite-cell power. |
| `--coverage-threshold 0.90` default (mirroring PR #28) | Same outdoor-NaN profile inherited from PR #28; new features add minor incremental NaN (multi-class surface preserves NaN; `is_primetime` rarely null). |
| Augment + swap × baseline + lgb-nb (4 reports, mirroring PR #28) | Refined-unit's binding question is "should we replace v1 broad-cut cols with refined ones?" — swap directly tests that. Augment tests "are these additive over v1?". Both modes give optionality on integration shape. |
| ValueError on unseen surface code | Forces deliberate spec amendment on `nfl_data_py` upstream changes; prevents silent column drift. Pinned set committed alongside override generator. |
| Schema integration deferred (probe-only) | Track 2 pattern; SIGNAL verdict greenlights integration plan with adoption gate. Avoids feature-cache invalidation churn for a NULL-likely refined family. |
| `data/features_probe/weather.parquet` overwrites PR #28's | Output is regenerable; no need for separate path. PR #28's narrower override is kept reproducible via this spec's predecessor reference. |

---

## 8. Open questions / known limitations

- **Unseen surface codes in 2025+ data.** This spec pins the surface column set from 2018–2024 data. Future seasons may introduce new codes (rare but possible — Las Vegas Allegiant Stadium uses a hybrid surface that's been recoded twice). The ValueError trigger is the right shape: forces a deliberate spec amendment rather than silent drift.
- **Continuous `kickoff_hour_et` not encoded.** Tree models (lgb-nb) cannot find sub-buckets (e.g., late-window 4:25 vs 4:05) from a single binary `is_primetime`. If `is_primetime` returns "directionally right but small" with CIs bracketing zero, refined-unit-of-refined-unit territory is continuous hour or three-bucket encoding (`is_early`, `is_late`, `is_primetime`).
- **Stadium-specific surface durability.** Late-season grass fields chew up; same `surface=grass` in week 1 vs week 17 may represent very different actual surfaces. v1 / refined limitation; refined-unit territory.
- **Cold-weather × surface interaction.** Frozen turf may behave differently from frozen grass (in fact, modern fieldturf is engineered to retain footing better than grass at sub-freezing). The bundle includes `is_cold_weather` and per-surface bools but not their interaction; lgb-nb tree splits should approximate the interaction natively if material.
- **Closed-roof-as-dome for cold-weather games.** Some retractable-roof stadiums (AT&T Stadium, Lucas Oil Stadium) report `roof=closed` for cold-weather games (the roof is closed *because* it's cold). v1's dome-fill convention treats these identically to true domes (`temperature_f = 70.0`, `is_cold_weather = 0.0`). The "actual outside temperature" on those days is informational but not used. v1 limitation; not addressed here.
- **Wind direction not encoded.** Source data does not include direction; absolute wind speed is the best available proxy. Same v1 limitation.
- **No per-team primetime acclimation.** Some teams (Cowboys, Giants, Chiefs, 49ers) play primetime ~5x more often than others. The model can't distinguish "Chiefs in their 5th primetime game of the year" from "Lions in their 1st." Refined-unit-of-refined-unit territory.

---

## 9. Predecessor pointers

- `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md` — PR #29 integration spec (most-recent integration; coverage caveat lessons applied here).
- `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md` — PR #28 probe spec (canonical broad-cut weather template; this spec's parent).
- `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md` — PR #25 probe spec (most-recent probe before weather).
- `src/projections/features/weather_features.py` — implementation site for compute fns + assembler (extended in-place by this probe).
- `scripts/build_weather_override.py` — CLI override generator (extended in-place by this probe).
- `reports/feature_probe_weather_summary.md` — summary-report template from PR #28.
- `src/projections/schemas.py:289` — `SchedulesSchema` (source columns; no extension required).
- `src/projections/features/_shared.py` — `build_game_environment` (used by production builders; not extended here — probe-only).
