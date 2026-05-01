# PBP Feature Family Probe — Design

**Status:** approved (brainstorming, 2026-04-30). Ready for implementation plan.
**Date:** 2026-04-30
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:** Plan 9 (PR #17, merged at `220298f`) + Feature Signal Probe (PR #18, merged at `51d9aa5`) + Plan-9 lgb-nb option-C re-evaluation (PR #19, merged at `98c2088`). Branched from `main` at `98c2088` onto `feat/probe-pbp-family`.

---

## 1. Overview

Plan 9 tested a single PBP-derived feature (opp-EPA-residual) and the adoption gate returned `DO_NOT_ADOPT × 4` — at the BaselineModel level (Plan 9 main verdict, 2026-04-29) and again at the lightgbm-nb level (Plan 9 option-C re-evaluation via composite probes, PR #19, 2026-04-30). The post-mortem identified a planning miscalibration: the "5–15% RMSE" prior in TODO #3 was a *family*-level prior across six PBP-derived features, but Plan 9 applied it to a single-feature swap and never decomposed.

This spec tests the family at family-level granularity. Four PBP features are bundled into a single override parquet — `pace_l4`, `proe_l4`, `team_ayps_l4`, `team_def_epa_resid_l4` — and the existing feature-signal probe (PR #18) screens them in two modes (augment, swap) at one model class always (`baseline`) and a second model class conditionally (`lightgbm-nb`). The verdict the spec produces is one bit: **`SIGNAL`** (greenlit, write the production-builder plan) or **`NULL`** (family closed across model classes, do not scope a plan).

This spec is **probe-only**. It does not ship production feature builders, does not modify per-position feature schemas, and does not modify any model factory. Its terminal artifacts are (a) a committed override-generation script + tests, (b) committed probe reports, (c) a one-line family verdict + decision-log entry. If the family probes `SIGNAL`, a follow-up production-builder plan gets scoped under TODO #3c (with refined per-position units — player-aDOT for receivers, per-position EPA-residual à la Plan 9, etc.). The probe is the cheap pre-spec screen; production refinement happens only after `SIGNAL`.

### 1.1 Goals (in scope)

- New module `src/projections/features/pbp_team_features.py` with five pure functions:
  - `compute_team_pace(pbp) -> pd.DataFrame`            (`team`, `season`, `week`, `pace_l4`)
  - `compute_team_proe(pbp) -> pd.DataFrame`            (`team`, `season`, `week`, `proe_l4`)
  - `compute_team_ayps(pbp) -> pd.DataFrame`            (`team`, `season`, `week`, `team_ayps_l4`)
  - `compute_team_def_epa_residual(pbp) -> pd.DataFrame`(`team`, `season`, `week`, `team_def_epa_resid_l4`) — `team` here is the **defense's** team code; the joiner attaches each player's *opponent's* row.
  - `build_pbp_family_overrides(pbp, player_team_week_index) -> pd.DataFrame` — public assembler that calls the four computes, joins onto a per-player index, returns `(gsis_id, season, week, pace_l4, proe_l4, team_ayps_l4, team_def_epa_resid_l4)`.
- New script `scripts/build_pbp_family_override.py`. Thin glue: load PBP partitions via `projections.ingest.pbp.read_pbp`, load the player-team-week index from `weekly_stats` + `schedules` ingest layer, call the assembler, write `data/features_probe/pbp_family.parquet`. Manual-invoke; not part of CI; output not committed.
- New tests `tests/test_features/test_pbp_team_features.py`. Synthetic-PBP fixtures, one test per pure function (correctness on hand-crafted small frames), one assembly test (coverage + output schema).
- Two-to-four probe runs (each emitting 4 markdown + 4 CSV files, one per position; 16–32 committed files total under `reports/`):
  - `feature_probe_pbp_family_augment_{QB,RB,WR,TE}.{md,csv}` (always — baseline augment)
  - `feature_probe_pbp_family_swap_{QB,RB,WR,TE}.{md,csv}` (always — baseline swap)
  - `feature_probe_pbp_family_lgbnb_augment_{QB,RB,WR,TE}.{md,csv}` (conditional — only if the two baseline reports together return zero pooled `SIGNAL` cells AND zero Phase 2 `ADOPT/MARGINAL` verdicts)
  - `feature_probe_pbp_family_lgbnb_swap_{QB,RB,WR,TE}.{md,csv}` (conditional — same trigger)
- One committed summary report `reports/feature_probe_pbp_family_summary.md` consolidating the 2-or-4 underlying reports plus the family verdict.
- New helper `family_verdict_from_reports(reports) -> Literal["SIGNAL","NULL"]` in `src/projections/backtest/feature_probe.py` so the family verdict is reproducible from committed CSVs, not eyeballed.
- `TODO.md` #3c update + `project_management.md` decision-log entry recording the verdict and what it greenlights or closes.

### 1.2 Non-goals (deferred)

- **No production feature-builder integration.** If the probe returns `SIGNAL`, a follow-up plan adds per-position builders that may refine units (player aDOT for receivers; per-position EPA-allowed residual à la Plan 9). This spec stops at the probe verdict.
- **No new probe machinery.** The probe (PR #18) and `--force-composite` flag (PR #19) are reused as-is. No new CLI flags.
- **No new ingest source.** PBP ingest from Plan 9 is reused; if `nfl_data_py` columns required by these features turn out to be missing, the spec is blocked, not patched (a separate ingest-extension plan would be required).
- **No PROE-model-fitting work.** PROE here is a simplified bucketed expectation (actual pass% minus league-avg pass% in matched down/distance/score-diff buckets) — not a fitted xpass model. If the probe returns `NULL` and we believe a fuller PROE definition would change the verdict, that is a *separate* spec; do not iterate on the PROE definition pre-probe.
- **No tuning of feature backfill policy beyond the trailing-4-needs-prior-season fallback.** If coverage falls below 95% on any (position, season) pair, the probe rejects with a clear error and the user fixes the override; the spec does not add silent-NaN-impute paths.
- **No persistence under `data/` beyond the override parquet.** The override under `data/features_probe/pbp_family.parquet` is regenerable from PBP partitions; not committed (per probe spec §7.2).
- **No multi-window probes.** Trailing window is fixed at l4 (matches the v1 convention and Plan 9). Sweeping l2/l4/l8 is a separate spec if it ever matters.
- **No widening to LightGBM-tuned or untuned LightGBM model classes.** The probe supports `baseline` and `lightgbm-nb` only (probe spec §1.2); same restriction inherited here. C-tuned and untuned C are strictly dominated by C-NB on RMSE (Plan 5c verdict).

### 1.3 Success criteria for trusting the verdict

The probe (PR #18) is already calibrated against Plan 9 (criteria 1 + 2 in `2026-04-30-feature-signal-probe-design.md` §1.3 — both passed). This spec inherits that calibration. The **spec-level** criteria here are about whether the family verdict produced by this probe run is *meaningful*, not whether the probe's screening rule is sound:

1. **Override coverage ≥ 95%** on every (position, season) pair the probe consumes. The probe's existing coverage check enforces this; if any position falls below threshold, the override generation script must be fixed (e.g., backfill policy widened, additional source rows pulled) before the verdict is read. A probe that runs on 80% coverage produces a verdict against a biased post-dropna row set and is not trustworthy.
2. **Both baseline modes (augment, swap) run successfully** before the family verdict is read. A `SIGNAL` from one mode is sufficient to greenlight the family per the verdict rule (§4); but if either mode errors out (e.g., Ridge fit fails, override join fails), the verdict is *not yet computed* and the spec is blocked on whichever mode failed.
3. **If both baseline modes return `NULL`, both lgb-nb modes must run** before declaring the family closed. The conditional-lgb-nb rule in §3 is the contract; skipping it (e.g., on grounds of cost) means the family verdict is half-computed and the closure is not durable. The lgb-nb runs are the answer to TODO #3c's "model class may dominate over feature class" hypothesis at the *family* level — not running them would re-create exactly the Plan-9 single-feature-failure pattern at the family level.

If any of (1)–(3) cannot be satisfied, the spec stops short of declaring a verdict and the failure mode is logged in the summary report. The decision log records "blocked on \<reason\>" rather than "SIGNAL" or "NULL."

---

## 2. Inputs

### 2.1 PBP source

`projections.ingest.pbp.read_pbp` (Plan 9) for seasons 2018–2024. Curated 27-column subset already covers the upstream columns this spec needs:

- `season`, `week`, `posteam`, `defteam`, `play_type`, `qtr`, `wp`, `down`, `ydstogo`, `score_differential`, `air_yards`, `pass_attempt`, `epa`.

If any of these columns is missing in the loaded PBP, the spec is blocked (no silent imputation). The `--run-network` smoke at `tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema` already guards against upstream column-rename drift; if that smoke is green at spec-execution time, the curated subset is intact.

### 2.2 Player-team-week index

For each (gsis_id, season, week) row in the override, we need to know which `team` the player was on (for pace / proe / team_ayps) and which `team` they faced (for team_def_epa_resid). Source:

- `weekly_stats` ingest gives `(gsis_id, season, week, team)` directly.
- `schedules` ingest gives `(team, season, week, opponent)` directly.

The assembler joins these to produce `(gsis_id, season, week, team, opp)`, then attaches the four feature columns by:
- `pace_l4`, `proe_l4`, `team_ayps_l4` ← join on `(team, season, week)` to the offensive-side computes.
- `team_def_epa_resid_l4` ← join on `(opp, season, week)` to the defensive-side compute.

Bye weeks (no schedule row) are dropped before the join — those rows are not in the player-team-week index in the first place. Players whose team has no schedule row in week W have no override row produced for week W; the probe's existing left-merge on `(gsis_id, season, week)` then leaves them with NaN on the four columns, and the post-dropna step removes them from the candidate-side training set. As long as ≥ 95% of baseline rows survive, the probe accepts.

### 2.3 Trailing-window backfill rule

Trailing-4 means "the team's last 4 *completed regular-season games* prior to week W of season Y." Postseason games are excluded from the window — the model the probe consumes is trained on regular-season weekly_stats only, so the trailing context should match. Concretely:

1. For each `(team, season, week)`, compute the four features over the rolling-4-games window of regular-season games ending at week W-1 of the same season.
2. If fewer than 4 prior regular-season games exist in season Y (early-season weeks 1–4), prepend the last regular-season games of season Y-1 (filling backward through weeks 18, 17, …) to the rolling window so the four-game requirement is met.
3. If season Y is the team's first ingested season (Y == 2018, the start of the curated PBP window), the rows with fewer than 4 prior games are emitted with NaN values. The probe's coverage check then enforces the 95% threshold.

For relocated teams, the canonical-team-code mapping (`normalize_team_code`) already collapses `STL`/`SD`/`OAK`/`WSH`/`LA`/`LAR`/`JAX`/`JAC` history per CLAUDE.md conventions; the rolling window follows the canonical code.

### 2.4 Override parquet shape

```
columns:
  gsis_id                : str (pyarrow)            — required, GSIS-id-format-checked
  season                 : Int64 nullable           — required
  week                   : Int64 nullable           — required
  pace_l4                : float64 nullable
  proe_l4                : float64 nullable
  team_ayps_l4           : float64 nullable
  team_def_epa_resid_l4  : float64 nullable
```

One row per `(gsis_id, season, week)` for which the player's team has a schedule row that week. NaN values are permitted in the four feature columns (early-season + Y-1-history-missing edge cases) but the count must remain below the probe's 5%-coverage-loss threshold per (position, season).

The parquet is written to `data/features_probe/pbp_family.parquet` and is **not** committed (regenerable; matches probe spec §7.2 convention for `data/features_probe/` outputs).

---

## 3. Probe invocation matrix

All probes run with `--seasons 2018-2024` and `--holdout-years 2021-2024` (probe defaults; matches Plan 9 + the existing Plan-9 retro reports).

### 3.1 Always — 2 baseline runs

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_augment \
  --model baseline \
  --override data/features_probe/pbp_family.parquet \
  --csv-out reports/feature_probe_pbp_family_augment.csv

python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_swap \
  --model baseline \
  --override data/features_probe/pbp_family.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_family_swap.csv
```

Each probe writes one markdown + one CSV per position (probe spec §5 convention) under `reports/`. Total: 8 markdown + 8 CSV files at minimum (4 positions × 2 modes).

### 3.2 Conditional — 2 lgb-nb runs

Trigger: both baseline runs return zero pooled `SIGNAL` cells AND zero Phase 2 `ADOPT/MARGINAL` verdicts (per the §4 verdict rule — "family is `NULL` so far"). The trigger is computed by `family_verdict_from_reports([augment, swap])` returning `"NULL"`.

If triggered:

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_lgbnb_augment \
  --model lightgbm-nb \
  --override data/features_probe/pbp_family.parquet \
  --csv-out reports/feature_probe_pbp_family_lgbnb_augment.csv

python -m scripts.probe_feature_signal \
  --candidate-name pbp_family_lgbnb_swap \
  --model lightgbm-nb \
  --override data/features_probe/pbp_family.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_family_lgbnb_swap.csv
```

Runtime: ~1–2 hr per run (probe spec §8). Total worst-case: ~3 hr added to the always-run baseline minutes.

If the trigger does *not* fire (i.e., the baselines already returned `SIGNAL`), the lgb-nb runs are skipped and the family is greenlit on the strength of the baseline result alone. The spec does *not* require lgb-nb confirmation when baseline already says `SIGNAL` — that is gold-plating; the family is open at that point and the production-builder plan owns the model-class question for the refined per-position units.

### 3.3 What the probe sees

For each `(position, mode, model)` cell, the probe:

1. Loads the position's baseline feature parquet via `projections.features.cache.read_features`.
2. Left-merges the override on `(gsis_id, season, week)`. Coverage check fires if < 95% of baseline rows have all 4 override columns valid.
3. In swap mode: drops the four `opp_allowed_*_fppg_l4` columns from `baseline_cols`. In augment mode: keeps them. Either way, the four override columns are appended to `candidate_cols`.
4. Runs Phase 1 (per-stat ΔRMSE bootstrap) and conditionally Phase 2 (composite ΔRMSE + ΔSpearman bootstrap) per the probe's existing logic.

The four override columns appear in `candidate_cols` for *all* position runs because the override parquet has uniform team-level columns. No per-position routing in the override — every position sees the same four columns.

---

## 4. Family verdict rule

Family is **`SIGNAL`** (greenlit; write production-builder plan) iff *any* of the following holds across the executed reports:

- A pooled-across-years Phase 1 verdict on any `(position, stat)` is `"SIGNAL"`, OR
- Any position's Phase 2 verdict is `"ADOPT"` or `"MARGINAL"`.

Otherwise: family is **`NULL`** (closed across executed reports).

If only the two baseline reports have been run, "executed reports" is those two; if the conditional lgb-nb reports also ran, "executed reports" is all four. Per §1.3 criterion 3, when both baseline modes return `NULL`, the lgb-nb modes must also run before `NULL` is durable.

`REGRESSION` cells are surfaced in the summary report but do not flip the family verdict to `REGRESSION` — the family-level question is "is there orthogonal signal?", not "does this hurt." A plain `REGRESSION` × all is treated equivalently to `NULL` for the verdict bit (family closed); the summary report's narrative section flags `REGRESSION` cells for the future production-builder spec to consider if the family is later re-opened.

The verdict is computed by:

```python
def family_verdict_from_reports(reports: list[ProbeReport]) -> Literal["SIGNAL", "NULL"]:
    """Return the family-level verdict across one or more probe reports.

    Family is SIGNAL iff any pooled Phase 1 cell is SIGNAL OR any Phase 2
    PositionVerdict is ADOPT/MARGINAL across the input reports.
    """
```

This helper lives in `src/projections/backtest/feature_probe.py` next to `phase1_should_fire_phase2`. It is unit-tested with synthetic `ProbeReport` instances (one fixture per branch of the rule). The summary report's "Family verdict" line is populated by this helper, never hand-written.

---

## 5. Outputs

### 5.1 Per-probe reports (committed)

The probe already writes one markdown + one CSV per position to `reports/`. Filenames follow the existing convention: `feature_probe_<candidate-name>_<POS>.{md,csv}`.

For this spec, that is at minimum 16 files (8 markdown + 8 CSV) for the always-run baseline modes; up to 32 if lgb-nb is triggered.

### 5.2 Family summary report (committed)

`reports/feature_probe_pbp_family_summary.md`. Hand-written narrative + machine-rendered tables. Sections:

- Header: candidate-name, dates, override mtime, the four feature definitions in one paragraph.
- Per-mode summary table (one row per `(model, mode, position)`), populated from the underlying CSVs by a small parsing helper. Columns: pooled-Phase-1 SIGNAL count, pooled-Phase-1 REGRESSION count, Phase 2 verdict (or "skipped — Phase 1 NULL"), best Phase 1 stat-level effect size.
- Family verdict line: `Family verdict: SIGNAL` or `Family verdict: NULL`, populated by `family_verdict_from_reports`. If `SIGNAL`, the line names the `(position, stat, mode, model)` tuples that lit up.
- Decision log entry: one paragraph stating what the verdict greenlights or closes. If `SIGNAL`, names the candidate production-builder plan and what it would scope (per-position units, etc.). If `NULL`, names the family as closed across the four PBP-derived features at the BaselineModel + lgb-nb level and cross-references to TODO #3c.

The summary report is committed alongside the per-probe reports in the same commit, so the verdict + provenance is one diff.

### 5.3 What does NOT get committed

- `data/features_probe/pbp_family.parquet` — regenerable by `scripts/build_pbp_family_override.py` against the live PBP partitions. Same convention as Plan 9's deleted helper output.

---

## 6. Code shape

### 6.1 New module `src/projections/features/pbp_team_features.py`

Pure pandas. Imports `Team` from `projections.schemas` and `normalize_team_code`, `validate_gsis_id` per CLAUDE.md conventions (canonical IDs, never bare strings at boundaries). No new dataclasses; outputs are plain `pd.DataFrame`.

```python
def compute_team_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level neutral-script plays per 60 min, trailing 4 prior games.

    Neutral script: WP ∈ [0.20, 0.80] and qtr ≤ 3.
    Output: (team, season, week, pace_l4) — one row per (team, season, week)
    where the team has a scheduled game.
    """


def compute_team_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level pass rate over expected, trailing 4 prior games.

    Simplified bucketed expectation: actual pass% in neutral game state minus
    league-avg pass% in matched (down, distance bucket, score-diff bucket)
    buckets. Buckets:
      - down ∈ {1, 2, 3, 4}
      - ydstogo bucketed: short (1-3), medium (4-7), long (8+)
      - score_differential bucketed: trailing big (-15 or worse), trailing
        small (-14..-1), tied/leading small (0..7), leading big (+8 or
        better)
    Output: (team, season, week, proe_l4)
    """


def compute_team_ayps(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level mean air yards per pass attempt, trailing 4 prior games.

    Output: (team, season, week, team_ayps_l4)
    """


def compute_team_def_epa_residual(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive EPA-allowed-per-play residual vs schedule strength,
    trailing 4 prior games.

    Plain OLS residual: per (defteam, season, week), EPA-per-play across all
    plays the defense was on the field for, residualized vs the offensive
    opponents' season-average EPA-per-play (schedule strength). Same
    machinery as Plan 9's per-position EPA-residual, but pooled across all
    plays (not split by play type or downstream position).

    Output: (team, season, week, team_def_epa_resid_l4) where `team` is the
    DEFENSE's team code; the joiner attaches each player's *opponent's* row.
    """


def build_pbp_family_overrides(
    pbp: pd.DataFrame,
    player_team_week_index: pd.DataFrame,  # (gsis_id, season, week, team, opp)
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Validates: GSIS id format on every row; canonical team codes; no
    duplicate (gsis_id, season, week) keys; output schema matches §2.4.
    Schema: (gsis_id, season, week, pace_l4, proe_l4, team_ayps_l4,
    team_def_epa_resid_l4).

    Per-position coverage validation is the probe's responsibility (the
    assembler has no access to the per-position feature parquets); see
    §1.3 criterion 1 + §3.3 step 2.
    """
```

The four computes are independent and unit-testable with fixed-PBP fixtures. The assembler is the integration point and is tested with a small assembled-from-fixtures end-to-end fixture.

### 6.2 New script `scripts/build_pbp_family_override.py`

Argparse + I/O glue. Pattern matches `scripts/refresh_features.py`'s shape:

- `parse_args()` → `--seasons 2018-2024` (default), `--data-root data` (default), `--output data/features_probe/pbp_family.parquet` (default), `--force` (overwrite).
- `main(argv=None)`:
  1. Load PBP via `projections.ingest.pbp.read_pbp(seasons=range(start, end+1), data_root=Path(args.data_root))`.
  2. Load `weekly_stats` + `schedules` via the existing ingest helpers; build `player_team_week_index` by inner-joining.
  3. Call `build_pbp_family_overrides(pbp, player_team_week_index)`.
  4. Write the resulting frame to `args.output` via `pyarrow.parquet`. Refuse to overwrite without `--force`.

The script is not invoked in CI; the user runs it manually before each probe invocation. A short "How to regenerate the override" subsection in `CONTRIBUTING.md` documents the invocation (added in §9).

### 6.3 Helper in `src/projections/backtest/feature_probe.py`

```python
def family_verdict_from_reports(
    reports: list[ProbeReport],
) -> Literal["SIGNAL", "NULL"]:
    """Family-level verdict across executed probe reports.

    SIGNAL iff any pooled Phase 1 PerStatVerdict has verdict == 'SIGNAL'
    OR any Phase 2 PositionVerdict has verdict in ('ADOPT', 'MARGINAL').
    NULL otherwise.
    """
```

Lives next to `phase1_should_fire_phase2`. Unit-tested with synthetic `ProbeReport` fixtures covering the SIGNAL-via-Phase-1, SIGNAL-via-Phase-2, and NULL branches.

### 6.4 Tests

- `tests/test_features/test_pbp_team_features.py`:
  - `test_pace_neutral_script_only`: synthetic PBP frame with mixed-script plays, verify only neutral-script plays count toward pace.
  - `test_pace_trailing_4`: synthetic PBP frame with 6 prior games, verify the rolling window uses exactly the last 4.
  - `test_proe_bucketed_expectation`: synthetic frame with two teams in the same down/distance/score buckets but different actual pass rates, verify PROE direction.
  - `test_ayps_pass_attempts_only`: synthetic frame with mixed pass + run plays, verify air yards is summed only over pass attempts.
  - `test_def_epa_residual_vs_schedule_strength`: synthetic frame with two defenses facing offenses of different season-average EPA-per-play, verify residual direction.
  - `test_assembler_schema`: assembler output matches §2.4 exactly.
  - `test_assembler_coverage_threshold`: synthetic frame engineered to fail 95% coverage on RB rows; assembler raises a clear error.
  - `test_assembler_canonical_teams`: input with `JAC` collapsed to `JAX` per `normalize_team_code`.
- `tests/test_backtest/test_feature_probe.py`:
  - Add `test_family_verdict_signal_via_phase1`, `test_family_verdict_signal_via_phase2`, `test_family_verdict_null` to the existing test module.

### 6.5 No changes to the probe CLI or module

The probe (PR #18) and its `--force-composite` flag (PR #19) are reused as-is. No new flags, no new CLI args, no new feature-probe machinery beyond the `family_verdict_from_reports` helper.

---

## 7. Test plan + execution sequence

### 7.1 Synthetic-fixture tests (committed)

Per §6.4 above. All run in CI under `pytest`; no network, no real data dependencies. These tests cover:
- The four pure compute functions (correctness under controlled inputs).
- The assembler (schema + coverage + canonical-team behavior).
- The new `family_verdict_from_reports` helper (truth table).

### 7.2 Real-data execution sequence (run-once, reports committed)

1. Run `scripts/build_pbp_family_override.py --seasons 2018-2024`. Produces `data/features_probe/pbp_family.parquet`. Inspect coverage by position; fix backfill or escalate to spec-blocked if < 95%.
2. Run baseline augment + baseline swap probes (§3.1). Commit the 8 markdown + 8 CSV reports to `reports/`.
3. Compute family verdict via `family_verdict_from_reports([augment_report, swap_report])`.
4. **If `SIGNAL`**: skip steps 5–6 below. Go directly to step 7: write the family summary report (§5.2), commit, then update docs (step 8).
5. **If `NULL`**: run lgb-nb augment + lgb-nb swap probes (§3.2). Commit those 8 markdown + 8 CSV reports.
6. Recompute family verdict via `family_verdict_from_reports([augment_baseline, swap_baseline, augment_lgbnb, swap_lgbnb])`. The recomputed verdict is durable per §1.3 criterion 3.
7. Write the family summary report (§5.2) and commit alongside the probe reports for the verdict path actually taken (SIGNAL via step 4, or NULL/SIGNAL via step 6).
8. Update `TODO.md` #3c + `project_management.md` decision log per §9.

### 7.3 Standard verification gates

Per CLAUDE.md end-of-effort checklist:

- `pytest -v` — full suite, all passing.
- `mypy src tests` — zero violations.
- `ruff check src tests scripts` — zero violations.
- `ruff format --check src tests scripts` — no drift.
- `pytest -v -k "ingest or store or schemas"` — for any change touching PBP ingest or schema (this spec touches `pbp_team_features.py` which is downstream of ingest, not ingest itself, but the gate is cheap and runs anyway).

---

## 8. Risks

- **Override coverage gap.** If the PBP partitions for any (season, team) are incomplete, the trailing-4 backfill rule may fail to produce 4 prior games for some early-season weeks. The probe's 95%-coverage hard reject catches this; the fix is to widen the backfill source (e.g., reach back two prior seasons), but the spec defaults to one prior season and rejects rather than silently impute.
- **PROE simplification gap.** The bucketed PROE expectation is coarser than a fitted xpass model. A `NULL` family verdict could be partly attributable to PROE-definition noise rather than a true absence of family-level signal. The spec accepts this risk; the screening question is "does *any* of this carry signal?" If `NULL`, a future spec may revisit PROE definition specifically — but only with independent evidence that PROE matters.
- **EPA-residual already DO_NOT_ADOPT alone.** The single-feature EPA-residual probe was DO_NOT_ADOPT × 4 at BaselineModel and × 8 at lgb-nb (TODO #3b). The family probe tests it bundled with three other features — Ridge / lgb-nb can compose the four columns. A bundled `SIGNAL` does not retroactively make EPA-residual alone signal; it means the family carries orthogonal signal that the four-column linear (or tree) combination can extract. This distinction is important for the production-builder follow-up: if the family probes `SIGNAL`, the production plan must verify which subset of the four features actually carries the signal — the family-level prior does not pin the per-feature prior.
- **Naming collision: `team_def_epa_resid_l4` vs Plan 9's per-position EPA-residual columns.** This spec deliberately uses a *new* column name (`team_def_epa_resid_l4`) for the team-level overall EPA-residual to avoid collision with the per-position columns Plan 9 produced (`opp_qb_epa_residual_l4`, etc.) — those columns are not in any production schema today (Plan 9's swap was reverted at `941b96c`), but the override's column names should be unambiguous on inspection.
- **Coverage-check false-positive on traded players.** If a player is traded mid-season, their `(gsis_id, season, week, team)` may shift; the assembler's join on `team` produces correct values but the player-team-week index must reflect the post-trade team. `weekly_stats` is the source of truth here (it already records the team a player played for in week W), so the index inherits correct trade behavior. The synthetic fixtures cover this.
- **Probe verdict drift between override regenerations.** Override is regenerable; if `nfl_data_py` upstream PBP data revisions touch a row, the regenerated override produces subtly different values and the verdict may shift. The summary report's header captures the override mtime + the PBP partition mtimes for traceability.
- **No probe of LightGBM-tuned or untuned C.** Per probe spec §1.2 the probe supports `baseline` and `lightgbm-nb` only. C-tuned + untuned C are strictly dominated by C-NB on RMSE (Plan 5c verdict), so no information lost.

---

## 9. Documentation updates on merge

- **`TODO.md` #3c:**
  - Append a paragraph stating the family probe verdict (SIGNAL or NULL).
  - If `SIGNAL`: cite the production-builder follow-up plan candidate and what it scopes.
  - If `NULL`: state the family closed at BaselineModel + lgb-nb, cross-reference the summary report.

- **`project_management.md`:**
  - Append a "PBP Feature Family Probe" decision-log entry under the standard format. Date, branch, PR, verdict, what it greenlights or closes.

- **`CONTRIBUTING.md`:**
  - Add a one-paragraph "Regenerating the PBP family override" subsection under the existing feature-plan workflow (added in PR #18). One example invocation of `scripts/build_pbp_family_override.py`, note that the output is not committed and is regenerable from the live PBP partitions.

- **`docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md`:**
  - No changes. This spec inherits the probe's calibration; the probe's spec doesn't need to know about specific candidates.

---
