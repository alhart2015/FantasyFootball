# PBP Pressure Feature Family Probe — Design

**Status:** approved (brainstorming, 2026-05-02). Ready for implementation plan.
**Date:** 2026-05-02
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:** PBP Family Probe (PR #20, merged at `6120ff1`) + RB PBP Features Integration (PR #21, merged at `bc2dc8c`) + WR/TE PBP Receiver Features Probe (PR #22, merged at `5ebce02`) + PBP Red-Zone Family Probe (PR #23, merged at `3b15915`). Branched from `main` at `3b15915` onto `feat/probe-pbp-pressure`.

---

## 1. Overview

PR #20's PBP family probe (pace / PROE / team-AYPS / def-EPA-residual) returned `SIGNAL` via RB and shipped to `RbFeaturesSchema` in PR #21 (`(BaselineModel, RB)` adoption verdict ADOPT, composite RMSE delta -0.0124 fpts). PR #22's WR/TE receiver-level probe (player aDOT / deep target share / YAC-per-reception / RZ target share) returned durable `NULL` across baseline + lgb-nb. PR #23's red-zone team-level probe (RZ pace / RZ pass rate / def RZ EPA allowed / def RZ pass rate allowed) also returned durable `NULL`. The remaining unexplored team-level cut listed in TODO #3c is **pressure rate allowed by O-line** — sack-rate-allowed and scramble-rate proxies on both offense and defense — none of which were in PR #20's, PR #22's, or PR #23's bundles.

This spec tests that family at family-level granularity. Four pressure-related PBP features are bundled into a single override parquet — `team_sack_rate_allowed_l4`, `team_qb_scramble_rate_l4`, `team_def_sack_rate_l4`, `team_def_scramble_rate_l4` — and the existing feature-signal probe (PR #18) screens them in two modes (augment, swap) at one model class always (`baseline`) and a second model class conditionally (`lightgbm-nb` with `--force-composite`). The verdict the spec produces is one bit: **`SIGNAL`** (greenlight a follow-up production-builder plan) or **`NULL`** (family closed across model classes; do not scope a plan).

This spec is **probe-only**. It does not ship production feature builders, does not modify per-position feature schemas, and does not modify any model factory. Its terminal artifacts are (a) a committed override-generation script + tests, (b) committed probe reports, (c) a one-line family verdict + decision-log entry. If the family probes `SIGNAL`, a follow-up production-builder plan gets scoped under TODO #3c with refined per-position units (most likely QB-first integration given the predicted mechanism axis; see §4.1).

### 1.1 Goals (in scope)

- New module `src/projections/features/pbp_pressure_features.py` with four pure compute functions plus an attach helper plus a public assembler:
  - `compute_team_sack_rate_allowed(pbp) -> pd.DataFrame`        (`team`, `season`, `week`, `team_sack_rate_allowed_l4`)
  - `compute_team_qb_scramble_rate(pbp) -> pd.DataFrame`         (`team`, `season`, `week`, `team_qb_scramble_rate_l4`)
  - `compute_team_def_sack_rate(pbp) -> pd.DataFrame`            (`team`, `season`, `week`, `team_def_sack_rate_l4`) — `team` here is the **defense's** team code; the joiner attaches each player's *opponent's* row.
  - `compute_team_def_scramble_rate(pbp) -> pd.DataFrame`        (`team`, `season`, `week`, `team_def_scramble_rate_l4`) — `team` here is also the **defense's** team code.
  - `attach_pbp_pressure_features(index, pbp) -> pd.DataFrame` — joiner that appends the four columns to a `(gsis_id, season, week, team, opp)` index. Offensive cols join on `team`; defensive cols join on `opp`.
  - `build_pbp_pressure_overrides(pbp, player_team_week_index) -> pd.DataFrame` — public assembler that calls the four computes, joins onto a per-player index via `attach_pbp_pressure_features`, returns `(gsis_id, season, week, team_sack_rate_allowed_l4, team_qb_scramble_rate_l4, team_def_sack_rate_l4, team_def_scramble_rate_l4)`. Validates GSIS-id format + dup-key absence + row-count invariant after merges.
- New script `scripts/build_pbp_pressure_override.py`. Thin glue: load PBP / depth_charts / schedules partitions via `projections.store.read_partition`, build the `(gsis_id, season, week, team, opp)` index, call the assembler, write `data/features_probe/pbp_pressure.parquet`. Manual-invoke; not part of CI; output not committed.
- New tests `tests/test_features/test_pbp_pressure_features.py` (synthetic-PBP fixtures, mirror PR #20 + PR #23's pattern) plus `tests/test_scripts/test_build_pbp_pressure_override_cli.py` (mirrors PR #23's CLI tests).
- Two-to-four probe runs (each emitting one CSV via `--csv-out` and one markdown via stdout redirect, with all 4 positions rendered in long format inside each file; 4 always-run files + 4 conditional files = 4–8 committed files total under `reports/`):
  - `feature_probe_pbp_pressure_augment.{md,csv}` (always — baseline augment, all 4 positions)
  - `feature_probe_pbp_pressure_swap.{md,csv}` (always — baseline swap, all 4 positions)
  - `feature_probe_pbp_pressure_lgbnb_augment.{md,csv}` (conditional — only if both baseline reports together return `NULL` per the §4 verdict rule; runs with `--force-composite`)
  - `feature_probe_pbp_pressure_lgbnb_swap.{md,csv}` (conditional — same trigger; runs with `--force-composite`)
- One committed summary report `reports/feature_probe_pbp_pressure_summary.md` consolidating the 2-or-4 underlying reports plus the family verdict and a mechanism-annotation paragraph (see §4).
- `TODO.md` #3c update + `project_management.md` decision-log entry recording the verdict and what it greenlights or closes. `CONTRIBUTING.md` "Regenerating the PBP red-zone override" subsection extended with a sibling "Regenerating the PBP pressure override" entry.

### 1.2 Non-goals (deferred)

- **No production feature-builder integration.** If the probe returns `SIGNAL`, a follow-up plan adds per-position builders (QB-first the likely scoping, given the predicted mechanism axis). This spec stops at the probe verdict.
- **No new probe machinery.** The probe (PR #18) and `--force-composite` flag (PR #19) are reused as-is. No new CLI flags. The `family_verdict_from_reports` helper already exists from PR #20 and is reused verbatim — no new helpers.
- **No new ingest source.** PBP ingest from Plan 9 is reused; the curated 27-column `_KEEP` subset already includes every column this spec needs (see §2.1).
- **No alternate dropback denominator.** This spec uses `qb_dropback == 1` as the canonical pressure-event denominator. NFL's "official" sack rate uses `pass_attempts + sacks`; the dropback variant additionally includes scrambles, which is the right denominator for a *pressure* family because scrambles are themselves a pressure signal. Switching denominators is a refined-unit follow-up if the family probes `SIGNAL`.
- **No tuning of feature backfill policy beyond the trailing-4-needs-prior-season fallback.** If coverage falls below threshold on any (position, season) pair, the probe rejects with a clear error and the user fixes the override; the spec does not add silent-NaN-impute paths.
- **No persistence under `data/` beyond the override parquet.** The override under `data/features_probe/pbp_pressure.parquet` is regenerable from PBP partitions; not committed (per probe spec §7.2).
- **No multi-window probes.** Trailing window is fixed at l4 (matches v1 + PR #20 + PR #22 + PR #23). Sweeping l2/l4/l8 is a separate spec if it ever matters.
- **No widening to LightGBM-tuned or untuned LightGBM model classes.** The probe supports `baseline` and `lightgbm-nb` only (probe spec §1.2); same restriction inherited here.

### 1.3 Success criteria for trusting the verdict

The probe (PR #18) is calibrated against Plan 9 (criteria 1 + 2 in `2026-04-30-feature-signal-probe-design.md` §1.3 — both passed). This spec inherits that calibration. The **spec-level** criteria here are about whether the family verdict produced by this probe run is *meaningful*:

1. **Override coverage ≥ 95%** on every (position, season) pair the probe consumes. The probe's existing coverage check enforces this; if any position falls below threshold, the override generation script must be fixed before the verdict is read. Pooled-vs-per-season precedent: PR #22 used `--coverage-threshold 0.70`, PR #23 used `0.90` because the structural 2018 cold-start (no Y-1 backfill) drags pooled coverage below 95% even when per-season 2019–2024 is ≥97%. This spec applies the same fallback if needed and documents the exact threshold used in the summary report.
2. **Both baseline modes (augment, swap) run successfully** before the family verdict is read. A `SIGNAL` from one mode is sufficient to greenlight the family per §4; but if either mode errors out, the verdict is *not yet computed* and the spec is blocked on whichever mode failed.
3. **If both baseline modes return `NULL`, both lgb-nb modes (with `--force-composite`) must run** before declaring the family closed. The conditional-lgb-nb rule in §3 is the contract; skipping it means the family verdict is half-computed and the closure is not durable. The `--force-composite` flag is non-negotiable here — bare `--model lightgbm-nb` runs are tautological with baseline because Phase 1 is hardcoded RidgeCV regardless of `--model` (lesson surfaced in PR #22 §3.2 spec gap addendum, fixed upstream by PR #23).

If any of (1)–(3) cannot be satisfied, the spec stops short of declaring a verdict and the failure mode is logged in the summary report. The decision log records "blocked on \<reason\>" rather than "SIGNAL" or "NULL."

---

## 2. Inputs

### 2.1 PBP source

PBP partitions read via `projections.store.read_partition(raw_root, "pbp", season=s)` (the same reader used by `scripts/refresh_features.py` and `scripts/build_pbp_family_override.py`). Seasons 2018–2024. The curated 27-column `PbpSchema` (Plan 9) covers every column this spec needs:

- `season`, `week`, `posteam`, `defteam`, `qb_dropback`, `qb_scramble`, `sack` — all already in the curated subset (`src/projections/ingest/pbp.py:_KEEP`).

If any of these columns is missing in the loaded PBP, the spec is blocked (no silent imputation). The `--run-network` smoke at `tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema` already guards against upstream column-rename drift; if that smoke is green at spec-execution time, the curated subset is intact.

**Pressure-event definition.** `qb_dropback == 1` is the canonical denominator. Per `nfl_data_py` / `nflfastR` documentation, `qb_dropback` is `1` on every play that is either a pass attempt, a sack, OR a QB scramble — i.e., every play where the QB intends to drop back. Plays where `qb_dropback == 0` (handoffs, kneels, spikes, kickoffs, punts, FGs, no-plays) are excluded from both numerator and denominator. NaN `qb_dropback` rows are also excluded.

For each compute:
- `team_sack_rate_allowed_l4`: `sum(sack) / sum(qb_dropback)` over rows where `posteam == team` AND `qb_dropback == 1`.
- `team_qb_scramble_rate_l4`: `sum(qb_scramble) / sum(qb_dropback)` over rows where `posteam == team` AND `qb_dropback == 1`.
- `team_def_sack_rate_l4`: `sum(sack) / sum(qb_dropback)` over rows where `defteam == team` AND `qb_dropback == 1`.
- `team_def_scramble_rate_l4`: `sum(qb_scramble) / sum(qb_dropback)` over rows where `defteam == team` AND `qb_dropback == 1`.

Per-game rates are computed via `groupby([team-key, season, week]).agg(num=(...)/denom=(...))` then divided; the trailing-4 helper rolls those per-game rates. Dropback counts are typically 25–45 per team per game; trailing-4 gives ~120–180 dropback events in the rolling window, which is enough for stable rates absent extreme game scripts.

### 2.2 Player-team-week index

For each (gsis_id, season, week) row in the override, we need to know which `team` the player was on (for offensive features) and which `team` they faced (for defensive features). Source:

- `depth_charts` ingest gives `(gsis_id, season, week, team)` for every rostered player per team-week (including inactive players who didn't accumulate weekly_stats). This is the same source the per-position feature parquets are built from. Filtered to fantasy-relevant positions (`Position.QB`, `Position.RB`, `Position.WR`, `Position.TE`).
- `schedules` ingest gives `(season, week, home_team, away_team)`; pivoted twice to `(team, opp)` rows so each game contributes both teams' perspectives.

**Why depth_charts and not weekly_stats:** the per-position feature parquets at `data/features/{pos}/season=Y/week=W/` are built from `depth_charts` — including backup-QB / inactive-roster rows that never appear in `weekly_stats`. Using `weekly_stats` as the index source would miss those rows and produce a ~50% coverage gap at probe time (per PR #20's `scripts/build_pbp_family_override.py:62-86` rationale). The override must be keyed off the same source as the baseline feature parquet.

The override-builder script inner-joins these to produce `(gsis_id, season, week, team, opp)`, then the assembler attaches the four feature columns by:
- `team_sack_rate_allowed_l4`, `team_qb_scramble_rate_l4` ← join on `(team, season, week)` to the offensive-side computes.
- `team_def_sack_rate_l4`, `team_def_scramble_rate_l4` ← join on `(opp, season, week)` to the defensive-side computes.

Bye weeks (no schedule row) drop out of the inner join — those rows are not in the player-team-week index in the first place. As long as ≥ 95% of baseline rows survive at probe time (or the documented lower threshold per §1.3 criterion 1), the probe accepts.

### 2.3 Trailing-window backfill rule

Identical to PR #20 + PR #23. Trailing-4 means "the team's last 4 *completed regular-season games* prior to week W of season Y." Concretely:

1. For each `(team, season, week)`, compute the four features over the rolling-4-games window of regular-season games ending at week W-1 of the same season.
2. If fewer than 4 prior regular-season games exist in season Y (early-season weeks 1–4), prepend the last regular-season games of season Y-1 (filling backward through weeks 18, 17, …) to the rolling window so the four-game requirement is met.
3. If season Y is the team's first ingested season (Y == 2018, the start of the curated PBP window), the rows with fewer than 4 prior games are emitted with NaN values. The probe's coverage check then enforces the threshold.

For relocated teams, the canonical-team-code mapping (`normalize_team_code`) collapses `STL`/`SD`/`OAK`/`WSH`/`LA`/`LAR`/`JAX`/`JAC` history per CLAUDE.md conventions; the rolling window follows the canonical code.

The `_trailing_4_mean` helper that implements steps 1–2 lives in `pbp_team_features.py` (PR #20) and is duplicated inline in `pbp_redzone_features.py` (PR #23). This spec duplicates it inline a third time in the new `pbp_pressure_features.py` module. Rationale: extracting to a shared helper now would buy nothing and creates a cross-module dependency that future refactors would need to maintain. Revisit on the fourth PBP feature module if it arrives — three is the threshold the project's "rule of three" usually honors, and we are at three with this spec.

### 2.4 Override parquet shape

```
columns:
  gsis_id                       : str (pyarrow)            — required, GSIS-id-format-checked
  season                        : Int64 nullable           — required
  week                          : Int64 nullable           — required
  team_sack_rate_allowed_l4     : float64 nullable
  team_qb_scramble_rate_l4      : float64 nullable
  team_def_sack_rate_l4         : float64 nullable
  team_def_scramble_rate_l4     : float64 nullable
```

One row per `(gsis_id, season, week)` for which the player's team has a schedule row that week. NaN values are permitted in the four feature columns (early-season + Y-1-history-missing edge cases) but the count must remain below the probe's coverage-loss threshold per (position, season).

The parquet is written to `data/features_probe/pbp_pressure.parquet` and is **not** committed (regenerable; matches probe spec §7.2 convention).

---

## 3. Probe invocation matrix

All probes run with `--seasons 2018-2024 --holdout-years 2021-2024` (probe defaults; matches Plan 9 + PR #20 + PR #22 + PR #23).

### 3.1 Always — 2 baseline runs

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_pressure_augment \
  --model baseline \
  --override data/features_probe/pbp_pressure.parquet \
  --csv-out reports/feature_probe_pbp_pressure_augment.csv \
  > reports/feature_probe_pbp_pressure_augment.md

python -m scripts.probe_feature_signal \
  --candidate-name pbp_pressure_swap \
  --model baseline \
  --override data/features_probe/pbp_pressure.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_pressure_swap.csv \
  > reports/feature_probe_pbp_pressure_swap.md
```

Each probe writes one CSV via `--csv-out` and one markdown via stdout redirect, with all 4 positions rendered in long format inside each file (probe CLI emits one markdown table + the CSV-equivalent rows; see `scripts/probe_feature_signal.py:790-804`). Total: 2 markdown + 2 CSV files for the always-run baseline modes (4 files), plus 2 markdown + 2 CSV more if the conditional lgb-nb runs fire (4 more files), for a worst-case total of 8 probe-output files.

**On the swap-mode `--drop` set.** This drops only the v1 schedule-strength features (`opp_allowed_*_fppg_l4`). It does *not* drop PR #20's already-shipped RB columns (`pace_l4`, `proe_l4`, `team_ayps_l4`, `team_def_epa_resid_l4`) — those are in `RbFeaturesSchema` directly and are part of RB's current production feature set. The swap-mode probe therefore tests "does the pressure family carry orthogonal signal *beyond v1 + PR #20's bundle*?" for RB, and "does the pressure family carry orthogonal signal *beyond v1*?" for QB/WR/TE (none of which have the PR #20 columns). This is the right comparison: each position's swap-mode baseline is its current production feature set minus v1 fppg-based opp-strength.

### 3.2 Conditional — 2 lgb-nb runs with `--force-composite`

Trigger: both baseline runs return zero pooled `SIGNAL` cells AND zero Phase 2 `ADOPT/MARGINAL` verdicts (per the §4 verdict rule — "family is `NULL` so far"). The trigger is computed by `family_verdict_from_reports([augment, swap])` returning `"NULL"`.

If triggered:

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_pressure_lgbnb_augment \
  --model lightgbm-nb \
  --force-composite \
  --override data/features_probe/pbp_pressure.parquet \
  --csv-out reports/feature_probe_pbp_pressure_lgbnb_augment.csv \
  > reports/feature_probe_pbp_pressure_lgbnb_augment.md

python -m scripts.probe_feature_signal \
  --candidate-name pbp_pressure_lgbnb_swap \
  --model lightgbm-nb \
  --force-composite \
  --override data/features_probe/pbp_pressure.parquet \
  --drop opp_allowed_qb_fppg_l4,opp_allowed_rb_fppg_l4,opp_allowed_wr_fppg_l4,opp_allowed_te_fppg_l4 \
  --csv-out reports/feature_probe_pbp_pressure_lgbnb_swap.csv \
  > reports/feature_probe_pbp_pressure_lgbnb_swap.md
```

`--force-composite` is mandatory on the conditional runs. Phase 1 is hardcoded RidgeCV regardless of `--model`, so without `--force-composite` the lgb-nb runs are tautological with the already-completed baseline runs (this gap was caught and worked around in PR #22; PR #23 fixed it upstream and this spec inherits the fix). The flag forces Phase 2 to run unconditionally on the lightgbm-nb model class.

Runtime: ~1–2 hr per run (probe spec §8). Total worst-case: ~3 hr added to the always-run baseline minutes when the trigger fires.

If the trigger does *not* fire (i.e., the baselines already returned `SIGNAL`), the lgb-nb runs are skipped and the family is greenlit on the strength of the baseline result alone. The spec does *not* require lgb-nb confirmation when baseline already says `SIGNAL` — the production-builder follow-up owns the model-class question for the refined per-position units.

### 3.3 What the probe sees

For each `(position, mode, model)` cell, the probe:

1. Loads the position's baseline feature parquet via `projections.features.cache.read_features`.
2. Left-merges the override on `(gsis_id, season, week)`. Coverage check fires if < threshold of baseline rows have all 4 override columns valid.
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

The bundle's predicted mechanism is **QB-side pressure exposure**. Sacks subtract from `passing_yards`; sacks taken are themselves a fantasy-relevant stat (negative-scoring in some rulesets, zero in PPR but predictive of next-week role); QB scrambles are part of `rushing_yards`. So the predicted SIGNAL cells are:

- `(QB, passing_yards)` — sack rate allowed depresses passing yards.
- `(QB, sacks)` — direct mechanical relationship.
- `(QB, rushing_yards)` — scramble rate elevates rushing yards.
- Defensive-side cols affect QB cells via opponent-strength routing in the per-game join (the player's QB faces the opposing defense's pressure profile).

Secondary mechanisms — RB / WR / TE effects — are weaker priors:
- RB `rushing_yards` via game-script shift (heavy pressure → more obvious passing situations → fewer RB carries) is plausible but expected to be small.
- WR / TE via shorter throws under pressure is plausible but probably below the noise floor.

The summary report (§5.2) annotates whether SIGNAL (if any) fired on:

- A QB-side cell (predicted mechanism — plain confirmation note),
- A non-QB cell only (unexpected mechanism — narrative paragraph for the production-builder follow-up to consider whether the pressure mechanism actually drove the signal or whether something else did),
- Both (mixed; standard note).

This annotation is informational only — it does not change the SIGNAL/NULL verdict. The summary's "Family verdict" line is populated by `family_verdict_from_reports`; the mechanism paragraph is hand-written by the spec executor based on which cells lit up.

---

## 5. Outputs

### 5.1 Per-probe reports (committed)

Per probe run, the CLI writes one CSV via `--csv-out` and one markdown via stdout (captured with `> path.md`). Both files contain all 4 positions in long format. Filenames follow the existing convention: `feature_probe_<candidate-name>.{md,csv}`.

For this spec: 4 always-run files (2 markdown + 2 CSV across the 2 baseline modes); up to 8 total if the conditional lgb-nb runs fire (4 more files).

### 5.2 Family summary report (committed)

`reports/feature_probe_pbp_pressure_summary.md`. Hand-written narrative + machine-rendered tables. Sections:

- Header: candidate-name, dates, override mtime, the four feature definitions in one paragraph.
- Per-mode summary table (one row per `(model, mode, position)`), populated from the underlying CSVs by a small parsing helper. Columns: pooled-Phase-1 SIGNAL count, pooled-Phase-1 REGRESSION count, Phase 2 verdict (or "skipped — Phase 1 NULL"), best Phase 1 stat-level effect size.
- Family verdict line: `Family verdict: SIGNAL` or `Family verdict: NULL`, populated by `family_verdict_from_reports`. If `SIGNAL`, the line names the `(position, stat, mode, model)` tuples that lit up.
- Mechanism annotation paragraph (§4.1): which mechanism axis the SIGNAL fired on, if any.
- Decision log entry: one paragraph stating what the verdict greenlights or closes. If `SIGNAL`, names the candidate production-builder plan and what it would scope (likely QB-first integration; RB/WR/TE only if they also lit up). If `NULL`, names the family as closed across the four PBP-derived pressure features at the BaselineModel + lgb-nb level and cross-references to TODO #3c.

The summary report is committed alongside the per-probe reports in the same commit, so the verdict + provenance is one diff.

### 5.3 What does NOT get committed

- `data/features_probe/pbp_pressure.parquet` — regenerable by `scripts/build_pbp_pressure_override.py` against the live PBP partitions. Same convention as PR #20's `pbp_family.parquet`, PR #22's `pbp_receiver.parquet`, and PR #23's `pbp_redzone.parquet`.

---

## 6. Code shape

### 6.1 New module `src/projections/features/pbp_pressure_features.py`

Pure pandas. Imports `GSIS_ID_PATTERN` from `projections.schemas` per CLAUDE.md conventions. No new dataclasses; outputs are plain `pd.DataFrame`. Sibling to `pbp_team_features.py` and `pbp_redzone_features.py`; shape parallels them but specialized to pressure.

```python
def compute_team_sack_rate_allowed(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive sack rate allowed, trailing 4 prior games.

    Per (posteam, season, week): sum(sack) / sum(qb_dropback) over rows
    where posteam == team AND qb_dropback == 1. NaN qb_dropback rows are
    excluded. Trailing-4 mean of the per-game series per team, shifted
    so row at week W reflects the last 4 prior games.

    Output: (team, season, week, team_sack_rate_allowed_l4)
    """


def compute_team_qb_scramble_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive QB scramble rate, trailing 4 prior games.

    Per (posteam, season, week): sum(qb_scramble) / sum(qb_dropback)
    over rows where posteam == team AND qb_dropback == 1.

    Output: (team, season, week, team_qb_scramble_rate_l4)
    """


def compute_team_def_sack_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive sack rate forced, trailing 4 prior games.

    Per (defteam, season, week): sum(sack) / sum(qb_dropback) over rows
    where defteam == team AND qb_dropback == 1.

    Output: (team, season, week, team_def_sack_rate_l4) where `team` is
    the DEFENSE's team code; the joiner attaches each player's
    *opponent's* row.
    """


def compute_team_def_scramble_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive scramble rate forced (opposing QBs scramble at this rate
    against this defense), trailing 4 prior games.

    Per (defteam, season, week): sum(qb_scramble) / sum(qb_dropback) over
    rows where defteam == team AND qb_dropback == 1.

    Output: (team, season, week, team_def_scramble_rate_l4) where `team`
    is the DEFENSE's team code.
    """


def attach_pbp_pressure_features(
    index: pd.DataFrame,  # (gsis_id, season, week, team, opp)
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the 4 pressure family features to a player-team-week index.

    Mirrors `attach_pbp_family_features` (PR #20) and
    `attach_pbp_redzone_features` (PR #23). Empty `pbp` short-circuits
    to all-NaN columns. Output columns appended in order:
    team_sack_rate_allowed_l4, team_qb_scramble_rate_l4,
    team_def_sack_rate_l4, team_def_scramble_rate_l4.
    """


def build_pbp_pressure_overrides(
    pbp: pd.DataFrame,
    player_team_week_index: pd.DataFrame,  # (gsis_id, season, week, team, opp)
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Validates: GSIS id format on every row; no duplicate
    (gsis_id, season, week) keys; row-count invariant after merges.
    Schema: (gsis_id, season, week, team_sack_rate_allowed_l4,
    team_qb_scramble_rate_l4, team_def_sack_rate_l4,
    team_def_scramble_rate_l4).

    Per-position coverage validation is the probe's responsibility;
    see §1.3 criterion 1 + §3.3 step 2.
    """
```

Internal `_trailing_4_mean` helper duplicated inline from `pbp_team_features.py` (12 lines; same semantics — within-team rolling+shift, min_periods=4). Internal `_PBP_COLUMNS_USED` tuple lists the 7 PBP columns the four computes touch: `("posteam", "defteam", "season", "week", "qb_dropback", "qb_scramble", "sack")`.

The four computes are independent and unit-testable with fixed-PBP fixtures. The assembler is the integration point and is tested with a small assembled-from-fixtures end-to-end fixture.

### 6.2 New script `scripts/build_pbp_pressure_override.py`

Argparse + I/O glue. Pattern matches `scripts/build_pbp_redzone_override.py`'s shape exactly:

- `parse_args()` → `--seasons 2018-2024` (default), `--data-root data` (default), `--output data/features_probe/pbp_pressure.parquet` (default), `--force` (overwrite).
- `main(argv=None)`:
  1. Load PBP via `projections.store.read_partition(args.data_root / "raw", "pbp", season=s)` per-season + `pd.concat`. Span is `range(seasons.start - 1, seasons.stop)` to include the prior season for trailing-4 backfill at weeks 1–4. Skip seasons without a partition (matches PR #20's `_read_concat` helper).
  2. Load `depth_charts` + `schedules` via the same `read_partition` per-season pattern; build `player_team_week_index` via the same `_build_player_team_week_index` shape PR #20 + PR #23 use (filter depth_charts to fantasy positions, dedupe `(gsis_id, season, week)`, pivot schedules home/away into `(team, opp)` rows, inner-join on `(season, week, team)`).
  3. Call `build_pbp_pressure_overrides(pbp, player_team_week_index)`.
  4. Write the resulting frame to `args.output` via `df.to_parquet(args.output, index=False)`. Refuse to overwrite without `--force`.

The script is not invoked in CI; the user runs it manually before each probe invocation. The "Regenerating the PBP pressure override" subsection in `CONTRIBUTING.md` documents the invocation (added in §9).

### 6.3 Helper reuse — no new helpers in `feature_probe.py`

`family_verdict_from_reports` from PR #20 (`src/projections/backtest/feature_probe.py`) handles this spec's verdict rule unchanged. No new helper, no new tests on the helper. The summary report's "Family verdict" line is populated by calling this function.

### 6.4 Tests

`tests/test_features/test_pbp_pressure_features.py` (synthetic-PBP fixtures, mirrors PR #20's `test_pbp_team_features.py` and PR #23's `test_pbp_redzone_features.py`):

- `test_sack_rate_allowed_basic` — synthetic PBP with known `qb_dropback` and `sack` mix; verify ratio.
- `test_sack_rate_allowed_excludes_non_dropback_plays` — frame with `qb_dropback == 0` rows (handoffs, kneels) at posteam==team; verify excluded from both numerator and denominator.
- `test_sack_rate_allowed_excludes_nan_dropback` — frame with NaN `qb_dropback` rows; verify excluded.
- `test_qb_scramble_rate_basic` — synthetic frame with known scramble/dropback ratio; verify the rate.
- `test_qb_scramble_rate_excludes_non_dropback_plays` — non-dropback rows at posteam==team; verify excluded.
- `test_def_sack_rate_basic` — frame with two defenses facing the same offense; verify per-defense per-game ratio.
- `test_def_sack_rate_groups_by_defteam` — frame with same offense facing two defenses in same week; verify defenses get separate per-game rows.
- `test_def_scramble_rate_basic` — frame with two defenses' dropback play sets; verify per-defense scramble rate.
- `test_trailing_4_min_periods_4` — frame with 3 prior games for a team; verify NaN for the next-week row (min_periods=4).
- `test_trailing_4_within_team` — two teams stacked in input; verify rolling+shift doesn't leak across team boundary (mirrors PR #20's invariant).
- `test_trailing_4_no_leak_across_seasons` — last week of season Y followed by week 1 of Y+1; verify rolling crosses seasons by date order (matches PR #20's backfill semantics — the override-builder script feeds multi-season concat'd PBP which carries history forward).
- `test_attach_offensive_join_on_team` — known offensive feature (`team_sack_rate_allowed_l4`) attached on `team`; verify the right value lands on each row.
- `test_attach_defensive_join_on_opp` — known defensive feature (`team_def_sack_rate_l4`) attached on `opp`; verify the right value lands on each row.
- `test_attach_empty_pbp_short_circuits_to_nan` — empty PBP → all-NaN columns, same shape as a successful call.
- `test_assembler_schema` — assembler output matches §2.4 exactly.
- `test_assembler_canonical_teams` — input with `JAC` collapsed to `JAX` per `normalize_team_code` (relies on the ingest schemas validating canonical codes upstream; assembler is a passive consumer).
- `test_assembler_dup_keys_raises` — duplicate `(gsis_id, season, week)` in index → `ValueError`.
- `test_assembler_invalid_gsis_raises` — malformed gsis_id → `ValueError`.
- `test_assembler_row_count_invariant` — mock the merge to violate row count → `AssertionError`.

`tests/test_scripts/test_build_pbp_pressure_override_cli.py` (mirrors PR #23's CLI-test pattern):

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
- The four pure compute functions (correctness under controlled inputs, including the `qb_dropback == 1` filter and trailing-4 invariants).
- The attach helper (offensive vs defensive join correctness, empty-PBP short-circuit).
- The assembler (schema + canonical-team + dup-key + invalid-gsis + row-count-invariant behavior).
- The override-builder script CLI (arg parsing + write integration).

### 7.2 Real-data execution sequence (run-once, reports committed)

1. Run `scripts/build_pbp_pressure_override.py --seasons 2018-2024`. Produces `data/features_probe/pbp_pressure.parquet`. Inspect coverage by position; fix backfill or escalate to spec-blocked if < threshold.
2. Run baseline augment + baseline swap probes (§3.1). Commit the markdown + CSV reports to `reports/`.
3. Compute family verdict via `family_verdict_from_reports([augment_report, swap_report])`.
4. **If `SIGNAL`**: skip steps 5–6 below. Go directly to step 7: write the family summary report (§5.2), commit, then update docs (step 8).
5. **If `NULL`**: run lgb-nb augment + lgb-nb swap probes with `--force-composite` (§3.2). Commit those markdown + CSV reports.
6. Recompute family verdict via `family_verdict_from_reports([augment_baseline, swap_baseline, augment_lgbnb, swap_lgbnb])`. The recomputed verdict is durable per §1.3 criterion 3.
7. Write the family summary report (§5.2) and commit alongside the probe reports for the verdict path actually taken (SIGNAL via step 4, or NULL/SIGNAL via step 6). Include the mechanism annotation paragraph per §4.1.
8. Update `TODO.md` #3c + `project_management.md` decision log per §9.

### 7.3 Standard verification gates

Per CLAUDE.md end-of-effort checklist:

- `pytest -v` — full suite, all passing.
- `mypy src tests` — zero violations.
- `ruff check src tests scripts` — zero violations.
- `ruff format --check src tests scripts` — no drift.
- `pytest -v -k "ingest or store or schemas"` — for any change touching PBP ingest or schema (this spec touches `pbp_pressure_features.py` which is downstream of ingest, not ingest itself, but the gate is cheap and runs anyway).

---

## 8. Risks

- **Coverage gap from low-volume QB-pressure events.** Some games have very few dropbacks (run-heavy game scripts, lopsided losses, or hurry-up situations limiting one side). Per-game rates are high-variance; trailing-4 smoothing absorbs it, but the bootstrap CIs may widen. Same structural issue as `team_proe_l4` (PR #20). If pre-trailing per-game variance is very high, the probe sees noisier ΔRMSE estimates and is more conservative — biased toward NULL, not SIGNAL. Acceptable.
- **Collinearity with `team_def_epa_resid_l4`** (already shipped in `RbFeaturesSchema`). Sack-rate-forced is a narrow subset of full-field EPA-residual on defense — sack EPA is one of the largest negative-EPA play types. Swap mode does not drop PR #20's RB cols (those are production features now, not v1) — so RB swap-mode probes "incremental signal beyond v1 + PR #20" for the new pressure features, which is the right comparison.
- **2018 cold-start coverage drag.** Same precedent as PR #22 (`--coverage-threshold 0.70`) and PR #23 (`0.90`). The probe's threshold check is *pooled* and 2018's structural no-Y-1-backfill drags pooled to ~94–95% even when per-season 2019–2024 is uniformly ≥97%. If hit, lower the threshold to whatever clears (precedent: 0.70 or 0.90) and document in the summary report. Per-season 2019–2024 coverage is the substantive criterion; the eval window is restricted to 2021–2024 holdout regardless.
- **No `--force-composite` on baseline runs.** The baseline runs (§3.1) deliberately *do not* use `--force-composite`. Phase 1 + Phase 2 trigger naturally via `phase1_should_fire_phase2` for the baseline model (Phase 2 fires on per-stat SIGNAL). Forcing composite on baseline would skip the Phase 1 gate and make every cell a Phase 2 cell, which is over-spending compute when Phase 1 says NULL. The flag is only justified for the conditional lgb-nb runs because the bare-lgb-nb tautology (Phase 1 = RidgeCV regardless of `--model`) makes Phase 1 information-free for the lgb-nb question.
- **Probe verdict drift between override regenerations.** Override is regenerable; if `nfl_data_py` upstream PBP data revisions touch a row, the regenerated override produces subtly different values and the verdict may shift. The summary report's header captures the override mtime + the PBP partition mtimes for traceability.
- **No probe of LightGBM-tuned or untuned C.** Per probe spec §1.2 the probe supports `baseline` and `lightgbm-nb` only. C-tuned + untuned C are strictly dominated by C-NB on RMSE (Plan 5c verdict), so no information lost.
- **Bundle is non-orthogonal by construction.** All four features share the `qb_dropback` denominator and three of four share a sack OR scramble numerator on either offense or defense side. The bundled probe is intentional — we test the *family*, not each feature in isolation. A `SIGNAL` does not retroactively pin which of the four features carries the signal; the production-builder follow-up plan is responsible for ablation if needed.

---

## 9. Documentation updates on merge

- **`TODO.md` #3c:**
  - Append a paragraph stating the family probe verdict (SIGNAL or NULL) and date.
  - If `SIGNAL`: cite the production-builder follow-up plan candidate and what it scopes (likely QB-first integration; RB/WR/TE only if they also lit up; refined-unit alternatives like alternate denominators).
  - If `NULL`: state the family closed at BaselineModel + lgb-nb at the dropback-denominator cut, cross-reference the summary report. Note that this closes the third and final TODO #3c team-level family at the team-level granularity.

- **`project_management.md`:**
  - Append a "PBP Pressure Family Probe" decision-log entry under the standard format. Date, branch, PR, verdict, what it greenlights or closes.
  - If this probe + the prior PR #23 + PR #22 have all returned NULL, the "Next action" section advances to the remaining TODO #3c follow-ups (refined-unit alternatives) or to Track 2B (RB PBP × other model classes).

- **`CONTRIBUTING.md`:**
  - Add a "Regenerating the PBP pressure override" subsection sibling to the existing "Regenerating the PBP family override" (PR #20) and "Regenerating the PBP red-zone override" (PR #23) subsections. One example invocation of `scripts/build_pbp_pressure_override.py`, note that the output is not committed and is regenerable from the live PBP partitions.

- **`docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md`:**
  - No changes. This spec inherits the probe's calibration; the probe's spec doesn't need to know about specific candidates.

- **`docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`:**
  - No changes. This spec is sibling to PR #20's, not a successor; PR #20's design remains authoritative for its bundle.

- **`docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md`:**
  - No changes. Sibling spec; this one inherits structural patterns but is independent.

---
