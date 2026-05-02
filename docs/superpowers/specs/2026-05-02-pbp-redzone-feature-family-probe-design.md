# PBP Red-Zone Feature Family Probe — Design

**Status:** approved (brainstorming, 2026-05-02). Ready for implementation plan.
**Date:** 2026-05-02
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:** PBP Family Probe (PR #20, merged at `6120ff1`) + RB PBP Features Integration (PR #21, merged at `bc2dc8c`) + WR/TE PBP Receiver Features Probe (PR #22, merged at `5ebce02`). Branched from `main` at `5ebce02` onto `feat/probe-pbp-redzone`.

---

## 1. Overview

PR #20's PBP family probe (pace/PROE/team-AYPS/def-EPA-residual) returned `SIGNAL` via RB and was integrated into `RbFeaturesSchema` in PR #21 (`(BaselineModel, RB)` adoption gate verdict: ADOPT, composite RMSE delta -0.0124 fpts). PR #22's WR/TE receiver-level probe (player aDOT / deep target share / YAC-per-reception / RZ target share) returned durable `NULL` across baseline + lgb-nb at composite. The remaining unexplored team-level cut listed in TODO #3c is **red-zone context** — RZ pace, RZ pass rate, defensive RZ EPA allowed, defensive RZ pass rate forced — none of which were in PR #20's bundle.

This spec tests that family at family-level granularity. Four red-zone-specific PBP features are bundled into a single override parquet — `team_rz_pace_l4`, `team_rz_pass_rate_l4`, `team_def_rz_epa_allowed_l4`, `team_def_rz_pass_rate_allowed_l4` — and the existing feature-signal probe (PR #18) screens them in two modes (augment, swap) at one model class always (`baseline`) and a second model class conditionally (`lightgbm-nb` with `--force-composite`). The verdict the spec produces is one bit: **`SIGNAL`** (greenlight a follow-up production-builder plan) or **`NULL`** (family closed across model classes; do not scope a plan).

This spec is **probe-only**. It does not ship production feature builders, does not modify per-position feature schemas, and does not modify any model factory. Its terminal artifacts are (a) a committed override-generation script + tests, (b) committed probe reports, (c) a one-line family verdict + decision-log entry. If the family probes `SIGNAL`, a follow-up production-builder plan gets scoped under TODO #3c with refined per-position units (e.g., goal-line `yardline_100 ≤ 5` variants, per-stat splits between rushing/passing TDs).

### 1.1 Goals (in scope)

- New module `src/projections/features/pbp_redzone_features.py` with four pure compute functions plus an attach helper plus a public assembler:
  - `compute_team_rz_pace(pbp) -> pd.DataFrame`                  (`team`, `season`, `week`, `team_rz_pace_l4`)
  - `compute_team_rz_pass_rate(pbp) -> pd.DataFrame`             (`team`, `season`, `week`, `team_rz_pass_rate_l4`)
  - `compute_team_def_rz_epa_allowed(pbp) -> pd.DataFrame`       (`team`, `season`, `week`, `team_def_rz_epa_allowed_l4`) — `team` here is the **defense's** team code; the joiner attaches each player's *opponent's* row.
  - `compute_team_def_rz_pass_rate_allowed(pbp) -> pd.DataFrame` (`team`, `season`, `week`, `team_def_rz_pass_rate_allowed_l4`) — `team` here is also the **defense's** team code.
  - `attach_pbp_redzone_features(index, pbp) -> pd.DataFrame` — joiner that appends the four columns to a `(gsis_id, season, week, team, opp)` index. Offensive cols join on `team`; defensive cols join on `opp`.
  - `build_pbp_redzone_overrides(pbp, player_team_week_index) -> pd.DataFrame` — public assembler that calls the four computes, joins onto a per-player index via `attach_pbp_redzone_features`, returns `(gsis_id, season, week, team_rz_pace_l4, team_rz_pass_rate_l4, team_def_rz_epa_allowed_l4, team_def_rz_pass_rate_allowed_l4)`. Validates GSIS-id format + dup-key absence + row-count invariant after merges.
- New script `scripts/build_pbp_redzone_override.py`. Thin glue: load PBP / weekly_stats / schedules partitions via `projections.store.read_partition`, build the `(gsis_id, season, week, team, opp)` index, call the assembler, write `data/features_probe/pbp_redzone.parquet`. Manual-invoke; not part of CI; output not committed.
- New tests `tests/test_features/test_pbp_redzone_features.py` (synthetic-PBP fixtures, mirror PR #20's pattern) plus `tests/test_scripts/test_build_pbp_redzone_override_cli.py` (mirrors PR #20's CLI tests).
- Two-to-four probe runs (each emitting one CSV via `--csv-out` and one markdown via stdout redirect, with all 4 positions rendered in long format inside each file; 4 always-run files + 4 conditional files = 4–8 committed files total under `reports/`):
  - `feature_probe_pbp_redzone_augment.{md,csv}` (always — baseline augment, all 4 positions)
  - `feature_probe_pbp_redzone_swap.{md,csv}` (always — baseline swap, all 4 positions)
  - `feature_probe_pbp_redzone_lgbnb_augment.{md,csv}` (conditional — only if both baseline reports together return `NULL` per the §4 verdict rule; runs with `--force-composite`)
  - `feature_probe_pbp_redzone_lgbnb_swap.{md,csv}` (conditional — same trigger; runs with `--force-composite`)
- One committed summary report `reports/feature_probe_pbp_redzone_summary.md` consolidating the 2-or-4 underlying reports plus the family verdict and a mechanism-annotation paragraph (see §4).
- `TODO.md` #3c update + `project_management.md` decision-log entry recording the verdict and what it greenlights or closes. `CONTRIBUTING.md` "Regenerating the PBP family override" subsection extended with a sibling "Regenerating the PBP red-zone override" entry.

### 1.2 Non-goals (deferred)

- **No production feature-builder integration.** If the probe returns `SIGNAL`, a follow-up plan adds per-position builders with refined units. This spec stops at the probe verdict.
- **No new probe machinery.** The probe (PR #18) and `--force-composite` flag (PR #19) are reused as-is. No new CLI flags. The `family_verdict_from_reports` helper already exists from PR #20 and is reused verbatim — no new helpers.
- **No new ingest source.** PBP ingest from Plan 9 is reused; the curated 27-column `_KEEP` subset already includes every column this spec needs (see §2.1).
- **No goal-line variant.** This spec uses the standard NFL red-zone definition (`yardline_100 ≤ 20`). A goal-line cut (`yardline_100 ≤ 5`) is a refined-unit follow-up if the family probes `SIGNAL`; if it probes `NULL`, goal-line is unlikely to clear what RZ-broad couldn't.
- **No tuning of feature backfill policy beyond the trailing-4-needs-prior-season fallback.** If coverage falls below 95% on any (position, season) pair, the probe rejects with a clear error and the user fixes the override; the spec does not add silent-NaN-impute paths.
- **No persistence under `data/` beyond the override parquet.** The override under `data/features_probe/pbp_redzone.parquet` is regenerable from PBP partitions; not committed (per probe spec §7.2).
- **No multi-window probes.** Trailing window is fixed at l4 (matches v1 + PR #20). Sweeping l2/l4/l8 is a separate spec if it ever matters.
- **No widening to LightGBM-tuned or untuned LightGBM model classes.** The probe supports `baseline` and `lightgbm-nb` only (probe spec §1.2); same restriction inherited here.

### 1.3 Success criteria for trusting the verdict

The probe (PR #18) is calibrated against Plan 9 (criteria 1 + 2 in `2026-04-30-feature-signal-probe-design.md` §1.3 — both passed). This spec inherits that calibration. The **spec-level** criteria here are about whether the family verdict produced by this probe run is *meaningful*:

1. **Override coverage ≥ 95%** on every (position, season) pair the probe consumes. The probe's existing coverage check enforces this; if any position falls below threshold, the override generation script must be fixed before the verdict is read.
2. **Both baseline modes (augment, swap) run successfully** before the family verdict is read. A `SIGNAL` from one mode is sufficient to greenlight the family per §4; but if either mode errors out, the verdict is *not yet computed* and the spec is blocked on whichever mode failed.
3. **If both baseline modes return `NULL`, both lgb-nb modes (with `--force-composite`) must run** before declaring the family closed. The conditional-lgb-nb rule in §3 is the contract; skipping it means the family verdict is half-computed and the closure is not durable. The `--force-composite` flag is non-negotiable here — bare `--model lightgbm-nb` runs are tautological with baseline because Phase 1 is hardcoded RidgeCV regardless of `--model` (lesson surfaced in PR #22 §3.2 spec gap addendum).

If any of (1)–(3) cannot be satisfied, the spec stops short of declaring a verdict and the failure mode is logged in the summary report. The decision log records "blocked on \<reason\>" rather than "SIGNAL" or "NULL."

---

## 2. Inputs

### 2.1 PBP source

PBP partitions read via `projections.store.read_partition(raw_root, "pbp", season=s)` (the same reader used by `scripts/refresh_features.py` and `scripts/build_pbp_family_override.py`). Seasons 2018–2024. The curated 27-column `PbpSchema` (Plan 9) covers the upstream columns this spec needs:

- `season`, `week`, `posteam`, `defteam`, `play_type`, `pass_attempt`, `epa`, `yardline_100` — all already in the curated subset (`src/projections/ingest/pbp.py:_KEEP`).

If any of these columns is missing in the loaded PBP, the spec is blocked (no silent imputation). The `--run-network` smoke at `tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema` already guards against upstream column-rename drift; if that smoke is green at spec-execution time, the curated subset is intact.

**Red-zone definition.** Standard NFL `yardline_100 ≤ 20`. The PBP `yardline_100` column is "yards from opposing end zone" — `yardline_100 == 1` is goal-to-go from the 1; `yardline_100 == 20` is the 20. The filter is inclusive on both ends.

### 2.2 Player-team-week index

For each (gsis_id, season, week) row in the override, we need to know which `team` the player was on (for offensive features) and which `team` they faced (for defensive features). Source:

- `depth_charts` ingest gives `(gsis_id, season, week, team)` for every rostered player per team-week (including inactive players who didn't accumulate weekly_stats). This is the same source the per-position feature parquets are built from. Filtered to fantasy-relevant positions (`Position.QB`, `Position.RB`, `Position.WR`, `Position.TE`).
- `schedules` ingest gives `(season, week, home_team, away_team)`; pivoted twice to `(team, opp)` rows so each game contributes both teams' perspectives.

**Why depth_charts and not weekly_stats:** the per-position feature parquets at `data/features/{pos}/season=Y/week=W/` are built from `depth_charts` — including backup-QB / inactive-roster rows that never appear in `weekly_stats`. Using `weekly_stats` as the index source would miss those rows and produce a ~50% coverage gap at probe time (per PR #20's `scripts/build_pbp_family_override.py:62-86` rationale). The override must be keyed off the same source as the baseline feature parquet.

The override-builder script inner-joins these to produce `(gsis_id, season, week, team, opp)`, then the assembler attaches the four feature columns by:
- `team_rz_pace_l4`, `team_rz_pass_rate_l4` ← join on `(team, season, week)` to the offensive-side computes.
- `team_def_rz_epa_allowed_l4`, `team_def_rz_pass_rate_allowed_l4` ← join on `(opp, season, week)` to the defensive-side computes.

Bye weeks (no schedule row) drop out of the inner join — those rows are not in the player-team-week index in the first place. As long as ≥ 95% of baseline rows survive at probe time, the probe accepts.

### 2.3 Trailing-window backfill rule

Identical to PR #20. Trailing-4 means "the team's last 4 *completed regular-season games* prior to week W of season Y." Concretely:

1. For each `(team, season, week)`, compute the four features over the rolling-4-games window of regular-season games ending at week W-1 of the same season.
2. If fewer than 4 prior regular-season games exist in season Y (early-season weeks 1–4), prepend the last regular-season games of season Y-1 (filling backward through weeks 18, 17, …) to the rolling window so the four-game requirement is met.
3. If season Y is the team's first ingested season (Y == 2018, the start of the curated PBP window), the rows with fewer than 4 prior games are emitted with NaN values. The probe's coverage check then enforces the 95% threshold.

For relocated teams, the canonical-team-code mapping (`normalize_team_code`) collapses `STL`/`SD`/`OAK`/`WSH`/`LA`/`LAR`/`JAX`/`JAC` history per CLAUDE.md conventions; the rolling window follows the canonical code.

The `_trailing_4_mean` helper that implements steps 1–2 lives in `pbp_team_features.py` (PR #20). Rather than extracting it to a shared helper module, this spec duplicates the 12-line helper inline in the new `pbp_redzone_features.py` module. Rationale: extracting now buys nothing and makes a cross-module dependency that future refactors would need to maintain. Revisit on the third PBP feature module if it arrives.

### 2.4 Override parquet shape

```
columns:
  gsis_id                            : str (pyarrow)            — required, GSIS-id-format-checked
  season                             : Int64 nullable           — required
  week                               : Int64 nullable           — required
  team_rz_pace_l4                    : float64 nullable
  team_rz_pass_rate_l4               : float64 nullable
  team_def_rz_epa_allowed_l4         : float64 nullable
  team_def_rz_pass_rate_allowed_l4   : float64 nullable
```

One row per `(gsis_id, season, week)` for which the player's team has a schedule row that week. NaN values are permitted in the four feature columns (early-season + Y-1-history-missing edge cases) but the count must remain below the probe's 5%-coverage-loss threshold per (position, season).

The parquet is written to `data/features_probe/pbp_redzone.parquet` and is **not** committed (regenerable; matches probe spec §7.2 convention).

---

## 3. Probe invocation matrix

All probes run with `--seasons 2018-2024 --holdout-years 2021-2024` (probe defaults; matches Plan 9 + PR #20 + PR #22).

### 3.1 Always — 2 baseline runs

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_redzone_augment \
  --model baseline \
  --override data/features_probe/pbp_redzone.parquet \
  --csv-out reports/feature_probe_pbp_redzone_augment.csv \
  > reports/feature_probe_pbp_redzone_augment.md

python -m scripts.probe_feature_signal \
  --candidate-name pbp_redzone_swap \
  --model baseline \
  --override data/features_probe/pbp_redzone.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_redzone_swap.csv \
  > reports/feature_probe_pbp_redzone_swap.md
```

Each probe writes one CSV via `--csv-out` and one markdown via stdout redirect, with all 4 positions rendered in long format inside each file (probe CLI emits one markdown table + the CSV-equivalent rows; see `scripts/probe_feature_signal.py:790-804`). Total: 2 markdown + 2 CSV files for the always-run baseline modes (4 files), plus 2 markdown + 2 CSV more if the conditional lgb-nb runs fire (4 more files), for a worst-case total of 8 probe-output files.

**On the swap-mode `--drop` set.** This drops only the v1 schedule-strength features (`opp_allowed_*_fppg_l4`). It does *not* drop PR #20's already-shipped RB columns (`pace_l4`, `proe_l4`, `team_ayps_l4`, `team_def_epa_resid_l4`) — those are in `RbFeaturesSchema` directly and are part of RB's current production feature set. The swap-mode probe therefore tests "does the RZ family carry orthogonal signal *beyond v1 + PR #20's bundle*?" for RB, and "does the RZ family carry orthogonal signal *beyond v1*?" for QB/WR/TE (none of which have the PR #20 columns). This is the right comparison: each position's swap-mode baseline is its current production feature set minus v1 fppg-based opp-strength.

### 3.2 Conditional — 2 lgb-nb runs with `--force-composite`

Trigger: both baseline runs return zero pooled `SIGNAL` cells AND zero Phase 2 `ADOPT/MARGINAL` verdicts (per the §4 verdict rule — "family is `NULL` so far"). The trigger is computed by `family_verdict_from_reports([augment, swap])` returning `"NULL"`.

If triggered:

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_redzone_lgbnb_augment \
  --model lightgbm-nb \
  --force-composite \
  --override data/features_probe/pbp_redzone.parquet \
  --csv-out reports/feature_probe_pbp_redzone_lgbnb_augment.csv \
  > reports/feature_probe_pbp_redzone_lgbnb_augment.md

python -m scripts.probe_feature_signal \
  --candidate-name pbp_redzone_lgbnb_swap \
  --model lightgbm-nb \
  --force-composite \
  --override data/features_probe/pbp_redzone.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_redzone_lgbnb_swap.csv \
  > reports/feature_probe_pbp_redzone_lgbnb_swap.md
```

`--force-composite` is mandatory on the conditional runs. Phase 1 is hardcoded RidgeCV regardless of `--model`, so without `--force-composite` the lgb-nb runs are tautological with the already-completed baseline runs (this gap was caught and worked around in PR #22; this spec fixes it upstream). The flag forces Phase 2 to run unconditionally on the lightgbm-nb model class.

Runtime: ~1–2 hr per run (probe spec §8). Total worst-case: ~3 hr added to the always-run baseline minutes when the trigger fires.

If the trigger does *not* fire (i.e., the baselines already returned `SIGNAL`), the lgb-nb runs are skipped and the family is greenlit on the strength of the baseline result alone. The spec does *not* require lgb-nb confirmation when baseline already says `SIGNAL` — the production-builder follow-up owns the model-class question for the refined per-position units.

### 3.3 What the probe sees

For each `(position, mode, model)` cell, the probe:

1. Loads the position's baseline feature parquet via `projections.features.cache.read_features`.
2. Left-merges the override on `(gsis_id, season, week)`. Coverage check fires if < 95% of baseline rows have all 4 override columns valid.
3. In swap mode: drops the four `opp_allowed_*_fppg_l4` columns from `baseline_cols`. In augment mode: keeps them. Either way, the four override columns are appended to `candidate_cols`.
4. Runs Phase 1 (per-stat ΔRMSE bootstrap) and conditionally Phase 2 (composite ΔRMSE + ΔSpearman bootstrap) per the probe's existing logic. With `--force-composite`, Phase 2 runs unconditionally.

The four override columns appear in `candidate_cols` for *all* position runs because the override parquet has uniform team-level columns. No per-position routing in the override — every position sees the same four columns.

---

## 4. Family verdict rule

Inherits PR #20's rule verbatim via the existing `family_verdict_from_reports` helper. Family is **`SIGNAL`** (greenlit; write production-builder plan) iff *any* of the following holds across the executed reports:

- A pooled-across-years Phase 1 verdict on any `(position, stat)` is `"SIGNAL"`, OR
- Any position's Phase 2 verdict is `"ADOPT"` or `"MARGINAL"`.

Otherwise: family is **`NULL`** (closed across executed reports).

If only the two baseline reports have been run, "executed reports" is those two; if the conditional lgb-nb reports also ran, "executed reports" is all four. Per §1.3 criterion 3, when both baseline modes return `NULL`, the lgb-nb modes (with `--force-composite`) must also run before `NULL` is durable.

`REGRESSION` cells are surfaced in the summary report but do not flip the family verdict to `REGRESSION` — the family-level question is "is there orthogonal signal?", not "does this hurt." A plain `REGRESSION × all` is treated equivalently to `NULL` for the verdict bit (family closed); the summary report flags `REGRESSION` cells for the future production-builder spec to consider if the family is later re-opened.

### 4.1 Mechanism annotation (no impact on verdict bit)

The bundle's predicted mechanism is **TD efficiency** — RZ context drives passing/rushing/receiving TDs across all four positions. The summary report (§5.2) annotates whether SIGNAL (if any) fired on:

- A `*_tds` cell (predicted mechanism — plain confirmation note),
- A non-`*_tds` cell only (unexpected mechanism — narrative paragraph for the production-builder follow-up to consider whether the RZ mechanism actually drove the signal or whether something else did),
- Both (mixed; standard note).

This annotation is informational only — it does not change the SIGNAL/NULL verdict. The summary's "Family verdict" line is populated by `family_verdict_from_reports`; the mechanism paragraph is hand-written by the spec executor based on which cells lit up.

---

## 5. Outputs

### 5.1 Per-probe reports (committed)

Per probe run, the CLI writes one CSV via `--csv-out` and one markdown via stdout (captured with `> path.md`). Both files contain all 4 positions in long format. Filenames follow the existing convention: `feature_probe_<candidate-name>.{md,csv}`.

For this spec: 4 always-run files (2 markdown + 2 CSV across the 2 baseline modes); up to 8 total if the conditional lgb-nb runs fire (4 more files).

### 5.2 Family summary report (committed)

`reports/feature_probe_pbp_redzone_summary.md`. Hand-written narrative + machine-rendered tables. Sections:

- Header: candidate-name, dates, override mtime, the four feature definitions in one paragraph.
- Per-mode summary table (one row per `(model, mode, position)`), populated from the underlying CSVs by a small parsing helper. Columns: pooled-Phase-1 SIGNAL count, pooled-Phase-1 REGRESSION count, Phase 2 verdict (or "skipped — Phase 1 NULL"), best Phase 1 stat-level effect size.
- Family verdict line: `Family verdict: SIGNAL` or `Family verdict: NULL`, populated by `family_verdict_from_reports`. If `SIGNAL`, the line names the `(position, stat, mode, model)` tuples that lit up.
- Mechanism annotation paragraph (§4.1): which mechanism axis the SIGNAL fired on, if any.
- Decision log entry: one paragraph stating what the verdict greenlights or closes. If `SIGNAL`, names the candidate production-builder plan and what it would scope (per-position units, goal-line variants, etc.). If `NULL`, names the family as closed across the four PBP-derived RZ features at the BaselineModel + lgb-nb level and cross-references to TODO #3c.

The summary report is committed alongside the per-probe reports in the same commit, so the verdict + provenance is one diff.

### 5.3 What does NOT get committed

- `data/features_probe/pbp_redzone.parquet` — regenerable by `scripts/build_pbp_redzone_override.py` against the live PBP partitions. Same convention as PR #20's `pbp_family.parquet` and PR #22's `pbp_receiver.parquet`.

---

## 6. Code shape

### 6.1 New module `src/projections/features/pbp_redzone_features.py`

Pure pandas. Imports `GSIS_ID_PATTERN` from `projections.schemas` per CLAUDE.md conventions. No new dataclasses; outputs are plain `pd.DataFrame`. Sibling to `pbp_team_features.py`; shape parallels it but specialized to RZ.

```python
def compute_team_rz_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive RZ plays per game, trailing 4 prior games.

    Plays counted: rows where `posteam == team`, `play_type in
    {'pass', 'run'}`, AND `yardline_100 <= 20`. Excludes kickoffs,
    punts, FGs, no-plays. No neutral-script filter — the curated
    PbpSchema does not include `wp` / `qtr` / `score_differential`.

    Output: (team, season, week, team_rz_pace_l4)
    """


def compute_team_rz_pass_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level RZ pass rate, trailing 4 prior games.

    Mean of `pass_attempt` (1.0/0.0) over rows where `posteam == team`,
    `play_type in {'pass', 'run'}`, AND `yardline_100 <= 20`. The
    play_type filter excludes special-teams plays where pass_attempt
    is undefined.

    Output: (team, season, week, team_rz_pass_rate_l4)
    """


def compute_team_def_rz_epa_allowed(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive RZ EPA-per-play allowed, trailing 4 prior games.

    Per (defteam, season, week): mean of `epa` across rows where
    `defteam == team` AND `epa` is non-NaN AND `yardline_100 <= 20`.
    Plays where `play_type in {'pass', 'run'}` only (excludes kickoffs,
    punts, FGs which have non-NaN EPA but are not RZ pass/run plays).
    Then rolling-4 mean of the per-game series per team, shifted so
    row at week W reflects the last 4 prior games.

    Output: (team, season, week, team_def_rz_epa_allowed_l4) where
    `team` is the DEFENSE's team code; the joiner attaches each
    player's *opponent's* row.
    """


def compute_team_def_rz_pass_rate_allowed(pbp: pd.DataFrame) -> pd.DataFrame:
    """Pass rate the defense forces opposing offenses to use in RZ,
    trailing 4 prior games.

    Mean of `pass_attempt` over rows where `defteam == team`,
    `play_type in {'pass', 'run'}`, AND `yardline_100 <= 20`. Same
    aggregation as `compute_team_rz_pass_rate` but grouped by defense.

    Output: (team, season, week, team_def_rz_pass_rate_allowed_l4)
    where `team` is the DEFENSE's team code.
    """


def attach_pbp_redzone_features(
    index: pd.DataFrame,  # (gsis_id, season, week, team, opp)
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the 4 RZ family features to a player-team-week index.

    Mirrors `attach_pbp_family_features` in pbp_team_features.py.
    Empty `pbp` short-circuits to all-NaN columns. Output columns
    appended in order: team_rz_pace_l4, team_rz_pass_rate_l4,
    team_def_rz_epa_allowed_l4, team_def_rz_pass_rate_allowed_l4.
    """


def build_pbp_redzone_overrides(
    pbp: pd.DataFrame,
    player_team_week_index: pd.DataFrame,  # (gsis_id, season, week, team, opp)
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Validates: GSIS id format on every row; no duplicate
    (gsis_id, season, week) keys; row-count invariant after merges.
    Schema: (gsis_id, season, week, team_rz_pace_l4,
    team_rz_pass_rate_l4, team_def_rz_epa_allowed_l4,
    team_def_rz_pass_rate_allowed_l4).

    Per-position coverage validation is the probe's responsibility;
    see §1.3 criterion 1 + §3.3 step 2.
    """
```

Internal `_trailing_4_mean` helper duplicated inline from `pbp_team_features.py` (12 lines; same semantics — within-team rolling+shift, min_periods=4). Internal `_PBP_COLUMNS_USED` tuple lists the 8 PBP columns the four computes touch.

The four computes are independent and unit-testable with fixed-PBP fixtures. The assembler is the integration point and is tested with a small assembled-from-fixtures end-to-end fixture.

### 6.2 New script `scripts/build_pbp_redzone_override.py`

Argparse + I/O glue. Pattern matches `scripts/build_pbp_family_override.py`'s shape exactly:

- `parse_args()` → `--seasons 2018-2024` (default), `--data-root data` (default), `--output data/features_probe/pbp_redzone.parquet` (default), `--force` (overwrite).
- `main(argv=None)`:
  1. Load PBP via `projections.store.read_partition(args.data_root / "raw", "pbp", season=s)` per-season + `pd.concat`. Span is `range(seasons.start - 1, seasons.stop)` to include the prior season for trailing-4 backfill at weeks 1–4. Skip seasons without a partition (matches PR #20's `_read_concat` helper).
  2. Load `depth_charts` + `schedules` via the same `read_partition` per-season pattern; build `player_team_week_index` via the same `_build_player_team_week_index` shape PR #20 uses (filter depth_charts to fantasy positions, dedupe `(gsis_id, season, week)`, pivot schedules home/away into `(team, opp)` rows, inner-join on `(season, week, team)`).
  3. Call `build_pbp_redzone_overrides(pbp, player_team_week_index)`.
  4. Write the resulting frame to `args.output` via `df.to_parquet(args.output, index=False)`. Refuse to overwrite without `--force`.

The script is not invoked in CI; the user runs it manually before each probe invocation. The "Regenerating the PBP red-zone override" subsection in `CONTRIBUTING.md` documents the invocation (added in §9).

### 6.3 Helper reuse — no new helpers in `feature_probe.py`

`family_verdict_from_reports` from PR #20 (`src/projections/backtest/feature_probe.py`) handles this spec's verdict rule unchanged. No new helper, no new tests on the helper. The summary report's "Family verdict" line is populated by calling this function.

### 6.4 Tests

`tests/test_features/test_pbp_redzone_features.py` (synthetic-PBP fixtures, mirrors PR #20's `test_pbp_team_features.py`):

- `test_rz_pace_filters_yardline_100_gt_20` — synthetic PBP with mixed in-zone + out-of-zone plays; verify only `yardline_100 <= 20` plays count.
- `test_rz_pace_excludes_special_teams` — synthetic PBP with kickoff/punt/FG plays at RZ yardlines; verify excluded.
- `test_rz_pass_rate_basic` — synthetic frame with known pass/run mix in RZ; verify the mean.
- `test_rz_pass_rate_filters_yardline_100_gt_20` — same play set with non-RZ rows added; verify they don't shift the mean.
- `test_def_rz_epa_allowed_basic` — synthetic frame with two defenses facing the same offense at RZ; verify per-defense per-game mean.
- `test_def_rz_epa_allowed_excludes_nan_epa` — frame with NaN EPA rows in RZ; verify excluded from the mean.
- `test_def_rz_pass_rate_allowed_basic` — frame with two defenses' RZ play sets; verify per-defense pass-rate.
- `test_trailing_4_min_periods_4` — frame with 3 prior games for a team; verify NaN for the next-week row (min_periods=4).
- `test_trailing_4_within_team` — two teams stacked in input; verify rolling+shift doesn't leak across team boundary (mirrors PR #20's invariant).
- `test_attach_offensive_join_on_team` + `test_attach_defensive_join_on_opp` — assembly correctness test pair: known offensive feature attached on `team`, known defensive feature attached on `opp`.
- `test_attach_empty_pbp_short_circuits_to_nan` — empty PBP → all-NaN columns, same shape as a successful call.
- `test_assembler_schema` — assembler output matches §2.4 exactly.
- `test_assembler_canonical_teams` — input with `JAC` collapsed to `JAX` per `normalize_team_code` (relies on the ingest schemas validating canonical codes upstream; assembler is a passive consumer).
- `test_assembler_dup_keys_raises` — duplicate `(gsis_id, season, week)` in index → `ValueError`.
- `test_assembler_invalid_gsis_raises` — malformed gsis_id → `ValueError`.
- `test_assembler_row_count_invariant` — mock the merge to violate row count → `AssertionError`.

`tests/test_scripts/test_build_pbp_redzone_override_cli.py` (introduces CLI test coverage for the override-builder pattern; PR #20's `scripts/build_pbp_family_override.py` shipped without CLI tests, so this spec adds the missing coverage. The same tests would apply retroactively to PR #20's script if anyone wants to backfill, but that is out of scope here):

- `test_parse_args_defaults` — happy-path arg parsing produces the documented defaults.
- `test_parse_args_seasons_range_format` — `--seasons 2020-2022` parses to `range(2020, 2023)`; `--seasons 2024` parses to `range(2024, 2025)`.
- `test_main_writes_output` — integration test with monkeypatched `read_partition` returning small synthetic PBP / depth_charts / schedules; verify the parquet is written at the configured path with the expected row count.
- `test_main_refuses_overwrite_without_force` — pre-existing output file → `argparse` parser-error exit (parser.error path).

No new probe tests — `family_verdict_from_reports` already has full coverage from PR #20.

### 6.5 No changes to the probe CLI or module

The probe (PR #18) and its `--force-composite` flag (PR #19) are reused as-is. No new flags, no new CLI args, no new feature-probe machinery.

---

## 7. Test plan + execution sequence

### 7.1 Synthetic-fixture tests (committed)

Per §6.4 above. All run in CI under `pytest`; no network, no real data dependencies. These tests cover:
- The four pure compute functions (correctness under controlled inputs, including the RZ filter and trailing-4 invariants).
- The attach helper (offensive vs defensive join correctness, empty-PBP short-circuit).
- The assembler (schema + canonical-team + dup-key + invalid-gsis + row-count-invariant behavior).
- The override-builder script CLI (arg parsing + write integration).

### 7.2 Real-data execution sequence (run-once, reports committed)

1. Run `scripts/build_pbp_redzone_override.py --seasons 2018-2024`. Produces `data/features_probe/pbp_redzone.parquet`. Inspect coverage by position; fix backfill or escalate to spec-blocked if < 95%.
2. Run baseline augment + baseline swap probes (§3.1). Commit the 8 markdown + 8 CSV reports to `reports/`.
3. Compute family verdict via `family_verdict_from_reports([augment_report, swap_report])`.
4. **If `SIGNAL`**: skip steps 5–6 below. Go directly to step 7: write the family summary report (§5.2), commit, then update docs (step 8).
5. **If `NULL`**: run lgb-nb augment + lgb-nb swap probes with `--force-composite` (§3.2). Commit those 8 markdown + 8 CSV reports.
6. Recompute family verdict via `family_verdict_from_reports([augment_baseline, swap_baseline, augment_lgbnb, swap_lgbnb])`. The recomputed verdict is durable per §1.3 criterion 3.
7. Write the family summary report (§5.2) and commit alongside the probe reports for the verdict path actually taken (SIGNAL via step 4, or NULL/SIGNAL via step 6). Include the mechanism annotation paragraph per §4.1.
8. Update `TODO.md` #3c + `project_management.md` decision log per §9.

### 7.3 Standard verification gates

Per CLAUDE.md end-of-effort checklist:

- `pytest -v` — full suite, all passing.
- `mypy src tests` — zero violations.
- `ruff check src tests scripts` — zero violations.
- `ruff format --check src tests scripts` — no drift.
- `pytest -v -k "ingest or store or schemas"` — for any change touching PBP ingest or schema (this spec touches `pbp_redzone_features.py` which is downstream of ingest, not ingest itself, but the gate is cheap and runs anyway).

---

## 8. Risks

- **Override coverage gap on early-2018 weeks** is mitigated by the 1-prior-season backfill rule; RZ-specific risk is that some teams may have very few RZ plays in some games (lopsided losses, run-heavy game scripts), causing high variance in the per-game means before the trailing-4 smooths them. The 95% probe coverage check still applies to row presence, not per-game stability — but high variance shows up as wide bootstrap CIs in Phase 1, which is the right behavior.
- **Correlation with PR #20's `team_def_epa_resid_l4`** (already shipped to RB): `team_def_rz_epa_allowed_l4` is the RZ-specific subset, so it overlaps in concept but on a different play subset. The swap-mode probe doesn't drop PR #20's columns (those are already in `RbFeaturesSchema`, not in the v1 `opp_allowed_*_fppg_l4` set), so RB swap-mode probes the new features against a v1 + pace + proe + ayps + team_def_epa_resid baseline — strictly the right comparison for "is there incremental RZ-specific signal beyond pace/proe/team-AYPS/full-field-EPA?"
- **No `--force-composite` on baseline runs.** The baseline runs (§3.1) deliberately *do not* use `--force-composite`. Phase 1 + Phase 2 trigger naturally via `phase1_should_fire_phase2` for the baseline model (Phase 2 fires on per-stat SIGNAL). Forcing composite on baseline would skip the Phase 1 gate and make every cell a Phase 2 cell, which is over-spending compute when Phase 1 says NULL. The flag is only justified for the conditional lgb-nb runs because the bare-lgb-nb tautology (Phase 1 = RidgeCV regardless of `--model`) makes Phase 1 information-free for the lgb-nb question.
- **Probe verdict drift between override regenerations.** Override is regenerable; if `nfl_data_py` upstream PBP data revisions touch a row, the regenerated override produces subtly different values and the verdict may shift. The summary report's header captures the override mtime + the PBP partition mtimes for traceability.
- **No probe of LightGBM-tuned or untuned C.** Per probe spec §1.2 the probe supports `baseline` and `lightgbm-nb` only. C-tuned + untuned C are strictly dominated by C-NB on RMSE (Plan 5c verdict), so no information lost.
- **Bundle is non-orthogonal by construction.** `team_rz_pass_rate_l4` and `team_def_rz_pass_rate_allowed_l4` are conceptually related (both are RZ pass-rate signals, on offense and defense sides respectively); the bundled probe is intentional — we test the *family*, not each feature in isolation. A `SIGNAL` does not retroactively pin which of the four features carries the signal; the production-builder follow-up plan is responsible for ablation if needed.

---

## 9. Documentation updates on merge

- **`TODO.md` #3c:**
  - Append a paragraph stating the family probe verdict (SIGNAL or NULL) and date.
  - If `SIGNAL`: cite the production-builder follow-up plan candidate and what it scopes (per-position integration; goal-line variant as refined unit; per-stat splits).
  - If `NULL`: state the family closed at BaselineModel + lgb-nb at the RZ-broad cut, cross-reference the summary report.

- **`project_management.md`:**
  - Append a "PBP Red-Zone Family Probe" decision-log entry under the standard format. Date, branch, PR, verdict, what it greenlights or closes.

- **`CONTRIBUTING.md`:**
  - Add a "Regenerating the PBP red-zone override" subsection sibling to the existing "Regenerating the PBP family override" subsection (added in PR #20). One example invocation of `scripts/build_pbp_redzone_override.py`, note that the output is not committed and is regenerable from the live PBP partitions.

- **`docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md`:**
  - No changes. This spec inherits the probe's calibration; the probe's spec doesn't need to know about specific candidates.

- **`docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`:**
  - No changes. This spec is sibling to PR #20's, not a successor; PR #20's design remains authoritative for its bundle.

---
