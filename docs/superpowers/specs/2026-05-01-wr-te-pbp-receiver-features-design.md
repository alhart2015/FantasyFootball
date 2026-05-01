# WR/TE PBP Receiver Family Probe — Design

**Status:** approved (brainstorming, 2026-05-01). Ready for implementation plan.
**Date:** 2026-05-01
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:** PBP Feature Family Probe (PR #20, merged at `6120ff1`) — shipped `pbp_team_features.py` + `family_verdict_from_reports` + the override-driven probe pattern. The team-level probe returned `SIGNAL` on RB only; WR / TE composite was null both modes (WR ±0.004, TE ±0.008). RB PBP Features Integration (PR #21, merged at `bc2dc8c`) — promoted the team-level family into RB production; `(BaselineModel, RB)` ADOPT at -0.0124 fpts. Feature Signal Probe (PR #18, merged at `51d9aa5`) — the probe CLI itself.
**Branch:** `feat/wr-te-pbp-features` cut from `main` at `bc2dc8c`.

---

## 1. Overview

PR #20's team-level PBP family probe greenlit RB but returned null for WR / TE: composite RMSE delta -0.004 (WR augment), -0.003 (WR swap), +0.006 (TE augment), +0.008 (TE swap). The probe's own decision log called this out: team-level pace / PROE / team-AYPS / team-def-EPA-residual is too coarse a unit for receiver outcomes; the natural refined unit is **player-level air-yards / aDOT distributions** computable from PBP's `receiver_player_id` linkage.

This spec tests that hypothesis at family-level granularity. Four player-level PBP features are bundled into a single override parquet and probed in two modes (augment, swap) at one model class always (`baseline`) and a second model class conditionally (`lightgbm-nb`). The verdict the spec produces is one bit: **`SIGNAL`** (greenlit, write the production-builder plan) or **`NULL`** (family closed across model classes for receivers, do not scope a plan).

Like PR #20, this spec is **probe-only**. It does not ship production feature builders, does not modify per-position feature schemas, and does not modify any model factory. Its terminal artifacts are (a) a committed receiver-features module + override-generation script + tests, (b) committed probe reports, (c) a one-line family verdict + decision-log entry. If the family probes `SIGNAL`, a follow-up production-builder plan gets scoped (per-position; the verdict tells us whether WR, TE, or both adopt).

### 1.1 Goals (in scope)

- New module `src/projections/features/pbp_receiver_features.py` with six pure functions (4 computes + assembler + validating wrapper):
  - `compute_receiver_adot(pbp) -> pd.DataFrame` — `(gsis_id, season, week, aDOT_l4)`
  - `compute_receiver_deep_target_share(pbp) -> pd.DataFrame` — `(gsis_id, season, week, deep_target_share_l4)`
  - `compute_receiver_yac_per_reception(pbp) -> pd.DataFrame` — `(gsis_id, season, week, yac_per_reception_l4)`
  - `compute_receiver_red_zone_target_share(pbp) -> pd.DataFrame` — `(gsis_id, season, week, red_zone_target_share_l4)`
  - `attach_pbp_receiver_features(index, pbp) -> pd.DataFrame` — assembler that calls the four computes and left-merges them onto a `(gsis_id, season, week)` index.
  - `build_pbp_receiver_overrides(pbp, receiver_index) -> pd.DataFrame` — public outer wrapper that adds GSIS-format / dup-key validation and the row-count invariant assertion (mirrors `build_pbp_family_overrides` in `pbp_team_features.py`). Returns `(gsis_id, season, week, aDOT_l4, deep_target_share_l4, yac_per_reception_l4, red_zone_target_share_l4)`.
- New script `scripts/build_pbp_receiver_override.py`. Thin glue: load PBP partitions via `projections.store.read_partition`, build the receiver index from `weekly_stats` filtered to `position in {WR, TE}`, call the assembler, write `data/features_probe/pbp_receiver.parquet`. Manual-invoke; not part of CI; output not committed.
- New tests `tests/test_features/test_pbp_receiver_features.py` covering the four pure computes plus assembler integration on synthetic-PBP fixtures.
- Two-to-four probe runs (each emitting 2 markdown + 2 CSV files, one per WR / TE; 4-to-8 committed files total under `reports/`):
  - `feature_probe_pbp_receiver_augment_{WR,TE}.{md,csv}` (always — baseline augment)
  - `feature_probe_pbp_receiver_swap_{WR,TE}.{md,csv}` (always — baseline swap)
  - `feature_probe_pbp_receiver_lgbnb_augment_{WR,TE}.{md,csv}` (conditional — only if both baseline reports together return zero pooled `SIGNAL` cells AND zero Phase 2 `ADOPT/MARGINAL` verdicts)
  - `feature_probe_pbp_receiver_lgbnb_swap_{WR,TE}.{md,csv}` (conditional — same trigger)
- One committed summary report `reports/feature_probe_pbp_receiver_summary.md` consolidating the 2-or-4 underlying reports plus the family verdict.
- `TODO.md` #3c update + `project_management.md` decision-log entry recording the verdict and what it greenlights or closes.

### 1.2 Non-goals (deferred)

- **No production feature-builder integration.** If the probe returns `SIGNAL`, a follow-up plan adds per-position builders. This spec stops at the probe verdict.
- **No new probe machinery.** The probe (PR #18) and its `--position` / `--force-composite` flags are reused as-is. The existing `--position` flag (action="append") scopes the probe to WR + TE without touching CLI code.
- **No new ingest source.** PBP ingest from Plan 9 is reused. The four compute functions consume only columns already in the curated `PbpSchema` (`receiver_player_id`, `season`, `week`, `pass_attempt`, `complete_pass`, `air_yards`, `yards_after_catch`, `yardline_100`).
- **No QB or RB inclusion in this probe.** QB was excluded by PR #20's family probe (augment-mode `passing_yards` regression +0.45 fpts — team-level PBP doesn't transfer to QB at this granularity, and the receiver-level analog is structurally meaningless for QBs since they throw, not catch). RB has team-level PBP shipped (PR #21) and a separate "should pass-catching backs get receiver-level aDOT features?" question is its own spec, not bundled here.
- **No threshold sensitivity sweep.** Deep-target threshold is fixed at `air_yards >= 20` (conventional); red-zone threshold is fixed at `yardline_100 <= 20` (standard). Sweeping at 15 / 25 yards or 10 / 30 yard-lines is a separate spec only if independently warranted.
- **No persistence under `data/` beyond the override parquet.** The override under `data/features_probe/pbp_receiver.parquet` is regenerable from PBP partitions; not committed.
- **No multi-window probes.** Trailing window is fixed at l4 (matches PR #20 + the v1 convention).
- **No widening to LightGBM-tuned or untuned LightGBM model classes.** The probe supports `baseline` and `lightgbm-nb` only (probe spec §1.2); same restriction inherited here.

### 1.3 Success criteria for trusting the verdict

This spec inherits the probe's calibration from PR #18 (criteria 1 + 2 in the feature-signal-probe spec §1.3 — both passed against Plan 9 retro). The **spec-level** criteria here are about whether the family verdict produced by this probe run is *meaningful*, not whether the probe's screening rule is sound:

1. **Override coverage ≥ 95%** on every `(position WR/TE, season)` pair the probe consumes. The probe's existing coverage check enforces this; if either WR or TE falls below threshold for any season, the override generation must be fixed (e.g., widen backfill to two prior seasons, or relax the receiver-active-game definition) before the verdict is read. **TE caveat:** blocking-only TEs may produce few receiver-active games; if TE coverage falls below 0.95 on any season, lower the threshold to 0.90 with `--coverage-threshold 0.90` and document in the summary that the surviving-TE row set is biased toward heavy-receiving TEs. The 95% / 90% cutoff is the same instrument PR #20 used to escape its own coverage edge case.
2. **Both baseline modes (augment, swap) run successfully** before the family verdict is read. A `SIGNAL` from one mode is sufficient to greenlight the family per the verdict rule (§4); but if either mode errors out (e.g., Ridge fit fails, override join fails), the verdict is *not yet computed* and the spec is blocked on whichever mode failed.
3. **If both baseline modes return `NULL`, both lgb-nb modes must run** before declaring the family closed. The conditional-lgb-nb rule in §3 is the contract; skipping it (e.g., on grounds of cost) means the family verdict is half-computed and the closure is not durable. The lgb-nb runs are the answer to "model class may dominate over feature class" at the *family* level for receivers — not running them would re-create exactly the Plan-9 single-feature-failure pattern at the family level.

If any of (1)–(3) cannot be satisfied, the spec stops short of declaring a verdict and the failure mode is logged in the summary report. The decision log records "blocked on \<reason\>" rather than "SIGNAL" or "NULL."

---

## 2. Inputs

### 2.1 PBP source

PBP partitions read via `projections.store.read_partition(raw_root, "pbp", season=s)` for `[seasons.start - 1, seasons.stop)`. Seasons 2018–2024. The curated 27-column `PbpSchema` (Plan 9) covers the upstream columns this spec needs:

- `season`, `week`, `receiver_player_id`, `pass_attempt`, `complete_pass`, `air_yards`, `yards_after_catch`, `yardline_100` — all already in the curated subset.

If any of these columns is missing in the loaded PBP, the spec is blocked (no silent imputation). The `--run-network` smoke at `tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema` already guards against upstream column-rename drift.

### 2.2 Receiver index

For each `(gsis_id, season, week)` row in the override, we need the receiver to have appeared in week W of season Y. Source:

- `weekly_stats` ingest gives `(gsis_id, season, week, position)` directly.
- Filter to `position in {WR, TE}`. One row per `(gsis_id, season, week)` for any WR/TE who appeared in week W of season Y (i.e., has a weekly_stats row).

Duplicate `(gsis_id, season, week)` keys in the index (which would arise from a player listed at both WR and TE in `weekly_stats` for the same week) are detected by the assembler's dup-key validation and raise `ValueError`. The synthetic fixtures cover this guardrail; real-data occurrences are vanishingly rare (a few historical data-entry errors in `nfl_data_py`'s upstream).

### 2.3 Trailing-window backfill rule

Trailing-4 means **"the player's last 4 *receiver-active* games prior to week W of season Y."** A receiver-active game is a `(gsis_id, season, week)` for which the player has at least one PBP row with `receiver_player_id == gsis_id` AND `pass_attempt == 1.0`. Players with zero targets in a week don't get a per-game row in the trailing window — so the trailing-4 is over the player's last 4 *target-receiving* games, not their last 4 calendar games. (Important for low-volume / WR3 / blocking-TE players: a player with sporadic usage may have 4 trailing games stretching across 6+ calendar weeks.)

Concretely:

1. For each `(gsis_id, season, week)`, compute the four features over the rolling-4-games window of receiver-active games ending at the player's last receiver-active game prior to W.
2. If fewer than 4 prior receiver-active games exist in season Y, prepend the last receiver-active games of season Y-1 (filling backward through weeks 18, 17, …) to the rolling window so the four-game requirement is met.
3. If season Y is the player's first ingested season (Y == 2018, the start of the curated PBP window), or if Y-1 + Y combined yields fewer than 4 receiver-active games, the rows are emitted with NaN values. The probe's coverage check then enforces the 95% threshold.

This convention matches NGS's "qualifying weeks" semantics and the rolling-4-of-receiver-active-games is the most defensible window definition for player-level depth profiles.

### 2.4 Override parquet shape

```
columns:
  gsis_id                     : str (pyarrow)            — required, GSIS-id-format-checked
  season                      : Int64 nullable           — required
  week                        : Int64 nullable           — required
  aDOT_l4                     : float64 nullable
  deep_target_share_l4        : float64 nullable
  yac_per_reception_l4        : float64 nullable
  red_zone_target_share_l4    : float64 nullable
```

One row per `(gsis_id, season, week)` for which the receiver appeared in `weekly_stats` (regardless of whether they had any targets that week — a WR who only blocks in week W still gets an override row, just with values populated from their prior receiver-active games). NaN values are permitted in the four feature columns (early-season + Y-1-history-missing edge cases) but the count must remain below the probe's 5%-coverage-loss threshold per (position, season) per §1.3 criterion 1.

The parquet is written to `data/features_probe/pbp_receiver.parquet` and is **not** committed (regenerable; matches PR #20's convention for `data/features_probe/` outputs).

---

## 3. Probe invocation matrix

All probes run with `--seasons 2018-2024` and `--holdout-years 2021-2024` (probe defaults; matches PR #20 + PR #21). Both runs are scoped to `--position WR --position TE` via the existing `--position` flag (action="append"); QB and RB are not exercised.

### 3.1 Always — 2 baseline runs

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_augment \
  --model baseline \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --csv-out reports/feature_probe_pbp_receiver_augment.csv

python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_swap \
  --model baseline \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --drop avg_intended_air_yards_std,avg_yac_above_expectation_std \
  --csv-out reports/feature_probe_pbp_receiver_swap.csv
```

**Swap-mode rationale.** Two of the four PBP receiver columns are direct refinements of NGS columns already on the WR / TE schemas:

- `aDOT_l4` (PBP, trailing-4) ↔ `avg_intended_air_yards_std` (NGS, season-to-date snapshot)
- `yac_per_reception_l4` (PBP, trailing-4) ↔ `avg_yac_above_expectation_std` (NGS, season-to-date snapshot)

Swap mode tests whether the trailing-4 + distribution-shape PBP variants strictly dominate the season-snapshot NGS variants. The other two PBP columns (`deep_target_share_l4`, `red_zone_target_share_l4`) have no direct NGS analog so nothing is dropped on their behalf.

We do **not** drop:
- `avg_separation_std` — no PBP analog (separation requires player tracking, lives only in NGS).
- `percent_share_intended_air_yards_std` — different unit (share vs magnitude); closest weekly-stats analog is the existing `air_yards_share_l4` already on WR (not on TE).

Each baseline run emits 2 markdown + 2 CSV files (one per WR / TE). Total minimum: 4 markdown + 4 CSV under `reports/`.

### 3.2 Conditional — 2 lgb-nb runs

Trigger: both baseline runs return `family_verdict_from_reports([augment, swap]) == "NULL"` (zero pooled `SIGNAL` cells AND zero Phase 2 `ADOPT/MARGINAL` verdicts across both reports). Same rule as PR #20 §3.2.

If triggered:

```bash
python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_lgbnb_augment \
  --model lightgbm-nb \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --csv-out reports/feature_probe_pbp_receiver_lgbnb_augment.csv

python -m scripts.probe_feature_signal \
  --candidate-name pbp_receiver_lgbnb_swap \
  --model lightgbm-nb \
  --position WR --position TE \
  --override data/features_probe/pbp_receiver.parquet \
  --drop avg_intended_air_yards_std,avg_yac_above_expectation_std \
  --csv-out reports/feature_probe_pbp_receiver_lgbnb_swap.csv
```

Runtime: ~1–2 hr per run (probe spec §8). Worst-case total: ~3 hr added to the always-run baseline minutes. Half PR #20's lgb-nb cost since this probe is scoped to 2 positions instead of 4.

If the trigger does *not* fire (i.e., the baselines already returned `SIGNAL`), the lgb-nb runs are skipped and the family is greenlit on the strength of the baseline result alone. The production-builder plan that follows owns the model-class question for the per-position units.

### 3.3 What the probe sees

For each `(position, mode, model)` cell, the probe:

1. Loads the position's baseline feature parquet via `projections.features.cache.read_features`.
2. Left-merges the override on `(gsis_id, season, week)`. Coverage check fires if < 95% of baseline rows have all 4 override columns valid (overridable to 0.90 for TE per §1.3 criterion 1).
3. In swap mode: drops `avg_intended_air_yards_std` and `avg_yac_above_expectation_std` from `baseline_cols`. In augment mode: keeps them. Either way, the four override columns are appended to `candidate_cols`.
4. Runs Phase 1 (per-stat ΔRMSE bootstrap) and conditionally Phase 2 (composite ΔRMSE + ΔSpearman bootstrap) per the probe's existing logic.

The four override columns appear in `candidate_cols` for both WR and TE because the override parquet has uniform receiver-level columns. No per-position routing in the override.

---

## 4. Family verdict rule

Family is **`SIGNAL`** (greenlit; write production-builder plan) iff *any* of the following holds across the executed reports:

- A pooled-across-years Phase 1 verdict on any `(position, stat)` is `"SIGNAL"`, OR
- Any position's Phase 2 verdict is `"ADOPT"` or `"MARGINAL"`.

Otherwise: family is **`NULL`** (closed across executed reports).

If only the two baseline reports have been run, "executed reports" is those two; if the conditional lgb-nb reports also ran, "executed reports" is all four. Per §1.3 criterion 3, when both baseline modes return `NULL`, the lgb-nb modes must also run before `NULL` is durable.

`REGRESSION` cells are surfaced in the summary report but do not flip the family verdict to `REGRESSION` — the family-level question is "is there orthogonal signal?", not "does this hurt." A `REGRESSION` × all is treated equivalently to `NULL` for the verdict bit (family closed); the summary report's narrative section flags `REGRESSION` cells for a future production-builder spec to consider if the family is later re-opened.

The verdict is computed by the existing `family_verdict_from_reports` helper in `src/projections/backtest/feature_probe.py` (added in PR #20). No new code in this spec for the verdict logic.

---

## 5. Outputs

### 5.1 Per-probe reports (committed)

The probe writes one markdown + one CSV per position to `reports/`. Filenames follow the existing convention: `feature_probe_<candidate-name>_<POS>.{md,csv}`.

For this spec, that is at minimum 4 files (2 markdown + 2 CSV) for the always-run baseline modes; up to 8 if lgb-nb is triggered.

### 5.2 Family summary report (committed)

`reports/feature_probe_pbp_receiver_summary.md`. Hand-written narrative + machine-rendered tables. Sections:

- Header: candidate-name, dates, override mtime, the four feature definitions in one paragraph, the swap-mode dropped columns.
- Per-mode summary table (one row per `(model, mode, position)`), populated from the underlying CSVs by a small parsing helper. Columns: pooled-Phase-1 SIGNAL count, pooled-Phase-1 REGRESSION count, Phase 2 verdict (or "skipped — Phase 1 NULL"), best Phase 1 stat-level effect size.
- Family verdict line: `Family verdict: SIGNAL` or `Family verdict: NULL`, populated by `family_verdict_from_reports`. If `SIGNAL`, the line names the `(position, stat, mode, model)` tuples that lit up.
- Decision log entry: one paragraph stating what the verdict greenlights or closes. If `SIGNAL`, names the candidate production-builder plan and what it would scope (per-position routing — WR-only, TE-only, or both). If `NULL`, names the family as closed across the four PBP-derived receiver features at the BaselineModel + lgb-nb level for receivers, and cross-references TODO #3c.

The summary report is committed alongside the per-probe reports in the same commit.

### 5.3 What does NOT get committed

- `data/features_probe/pbp_receiver.parquet` — regenerable by `scripts/build_pbp_receiver_override.py`. Same convention as PR #20's team-level override.

---

## 6. Code shape

### 6.1 New module `src/projections/features/pbp_receiver_features.py`

Pure pandas, mirrors `pbp_team_features.py`'s shape one-for-one. Imports `GSIS_ID_PATTERN` from `projections.schemas` for the assembler's id-format validation. No new dataclasses; outputs are plain `pd.DataFrame`.

```python
def compute_receiver_adot(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver mean depth of target, trailing 4 prior receiver-active games.

    Per (gsis_id, season, week): mean of ``air_yards`` across rows where
    ``receiver_player_id == gsis_id`` AND ``pass_attempt == 1.0`` AND
    ``air_yards.notna()``. NaN ``air_yards`` (sacks, throw-aways, no-plays)
    excluded.

    A receiver-active game is one where the receiver had at least one
    target. The trailing window is over the receiver's last 4 receiver-active
    games, not their last 4 calendar weeks.

    Output: (gsis_id, season, week, aDOT_l4)
    """


def compute_receiver_deep_target_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver share of targets with air_yards >= 20, trailing 4
    receiver-active games.

    Per (gsis_id, season, week):
      targets      = count rows where receiver_player_id == gsis_id AND
                     pass_attempt == 1.0 AND air_yards.notna()
      deep_targets = count rows where receiver_player_id == gsis_id AND
                     pass_attempt == 1.0 AND air_yards >= 20
      share        = deep_targets / targets   (NaN if targets == 0)

    Trailing-4 mean of share across receiver-active games. The 20-yard
    cutoff is the conventional "deep" threshold (PFF, NFL Next Gen Stats).

    Output: (gsis_id, season, week, deep_target_share_l4)
    """


def compute_receiver_yac_per_reception(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver mean yards-after-catch per completion, trailing 4
    receiver-active games.

    Per (gsis_id, season, week): mean of ``yards_after_catch`` across rows
    where ``receiver_player_id == gsis_id`` AND ``complete_pass == 1.0`` AND
    ``yards_after_catch.notna()``. Filtered to completions — YAC only exists
    when the ball is caught.

    Receivers with no catches in a week contribute no per-game row; the
    trailing window stretches further calendar weeks back to satisfy the
    four-game requirement.

    Output: (gsis_id, season, week, yac_per_reception_l4)
    """


def compute_receiver_red_zone_target_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver share of targets at yardline_100 <= 20, trailing 4
    receiver-active games.

    Per (gsis_id, season, week):
      total_targets = count rows where receiver_player_id == gsis_id AND
                      pass_attempt == 1.0
      rz_targets    = count rows where receiver_player_id == gsis_id AND
                      pass_attempt == 1.0 AND yardline_100 <= 20
      share         = rz_targets / total_targets   (NaN if total_targets == 0)

    Trailing-4 mean of share across receiver-active games. yardline_100 = 20
    is the standard NFL red-zone definition.

    This is the receiver's RZ target share, not the team's RZ target rate;
    captures whether the player is the offense's preferred end-zone target.

    Output: (gsis_id, season, week, red_zone_target_share_l4)
    """


def attach_pbp_receiver_features(
    index: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the 4 PBP receiver features to a (gsis_id, season, week) index.

    Args:
        index: ``(gsis_id, season, week)`` — one row per receiver-week.
        pbp: PBP frame matching ``PbpSchema``, projected to or wider than
            the receiver-features column set. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.

    Returns:
        A copy of ``index`` with 4 columns appended in order:
        ``aDOT_l4``, ``deep_target_share_l4``, ``yac_per_reception_l4``,
        ``red_zone_target_share_l4``. Row count equals ``len(index)``;
        all 4 columns are float64 (NaN where trailing-4 has fewer than 4
        prior receiver-active games).

    All four computes key on ``receiver_player_id``; no team / opponent
    join required (receiver-level features are independent of which team
    the player was on, beyond the implicit play-callable role attached to
    the player by their participation).

    Empty ``pbp`` short-circuits to all-NaN columns — same shape as a
    successful call where every row's trailing-4 has fewer than 4 prior
    receiver-active games. Schema ``nullable=True`` covers this.
    """


def build_pbp_receiver_overrides(
    pbp: pd.DataFrame,
    receiver_index: pd.DataFrame,
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Args:
        pbp: PBP frame matching ``PbpSchema``. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.
        receiver_index: ``(gsis_id, season, week)`` — one row per
            receiver-week. Built by the override script from
            ``weekly_stats`` filtered to ``position in {WR, TE}``.

    Returns:
        ``(gsis_id, season, week, aDOT_l4, deep_target_share_l4,
        yac_per_reception_l4, red_zone_target_share_l4)`` — one row per
        input index row.

    Raises:
        ValueError: gsis_id format violations or duplicate
            (gsis_id, season, week) keys in the index.
        AssertionError: row-count mismatch after merges.

    Per-position coverage validation is the probe's responsibility; see
    §1.3 criterion 1 + §3.3 step 2.
    """
```

The four computes share an internal `_trailing_4_per_player_pbp(per_game, *, value_col, out_col)` helper analogous to `pbp_team_features.py`'s `_trailing_4_mean`, but groups by `gsis_id` (receiver) instead of `team`. Same `min_periods=4` + within-group shift-1 semantics.

### 6.2 New script `scripts/build_pbp_receiver_override.py`

Argparse + I/O glue. Pattern matches `scripts/build_pbp_family_override.py`'s shape:

- `parse_args()` → `--seasons 2018-2024` (default), `--data-root data` (default), `--output data/features_probe/pbp_receiver.parquet` (default), `--force` (overwrite).
- `main(argv=None)`:
  1. Load PBP via `projections.store.read_partition(raw_root, "pbp", season=s)` for `[args.seasons.start - 1, args.seasons.stop)` (one prior season for backfill).
  2. Load `weekly_stats` for `[args.seasons.start, args.seasons.stop)`; build the receiver index by filtering to `position in {WR, TE}` and projecting `(gsis_id, season, week)`.
  3. Call `build_pbp_receiver_overrides(pbp, receiver_index)`.
  4. Write the resulting frame to `args.output` via `pyarrow.parquet`. Refuse to overwrite without `--force`.

Not invoked in CI; the user runs it manually before each probe invocation. A short "How to regenerate the receiver override" subsection in `CONTRIBUTING.md` documents the invocation (added in §9).

### 6.3 Tests

`tests/test_features/test_pbp_receiver_features.py`:

- `test_adot_air_yards_only` — synthetic PBP with mixed throwaway / sack / completion plays; assert aDOT averages only over rows with non-NaN `air_yards`.
- `test_adot_trailing_4_within_player` — 6 receiver-active games for player A, 6 for B; assert rolling-4 stays within `gsis_id` (no leakage across player boundary).
- `test_adot_shifted_by_one_game` — assert the trailing-4 at week W reflects games W-4 through W-1, not W.
- `test_deep_target_share_threshold` — synthetic frame with mixed depths {5, 15, 19, 20, 25, 35}; assert exactly the rows with `air_yards >= 20` count as deep.
- `test_deep_target_share_zero_targets_no_per_game_row` — a receiver with 0 targets in a week emits no per-game row.
- `test_yac_completions_only` — frame with mixed completions / incompletions; assert YAC averages only over `complete_pass == 1.0`.
- `test_red_zone_target_share_yardline_threshold` — frame with targets at `yardline_100 ∈ {5, 15, 20, 21, 50}`; assert ≤20 fires (5, 15, 20) and >20 doesn't (21, 50).
- `test_attach_receiver_features_schema` — `attach_pbp_receiver_features` output has the 4 new columns + the input index's columns; row count preserved.
- `test_attach_receiver_features_left_join_semantics` — index row for a receiver with no PBP rows yields NaN on all 4 columns.
- `test_attach_receiver_features_empty_pbp` — empty PBP short-circuits to all-NaN columns (matches the team-level fast path).
- `test_build_receiver_overrides_canonical_gsis` — `build_pbp_receiver_overrides` raises `ValueError` on malformed GSIS in the index.
- `test_build_receiver_overrides_dup_key` — `build_pbp_receiver_overrides` raises `ValueError` on duplicate `(gsis_id, season, week)` in the index.
- `test_build_receiver_overrides_row_count_invariant` — `build_pbp_receiver_overrides` raises `AssertionError` if a future merge regression introduces a row-count mismatch.

`tests/test_backtest/test_feature_probe.py`: no new tests. `family_verdict_from_reports` was added in PR #20 and is unchanged.

`tests/test_scripts/test_probe_feature_signal_cli.py`: no new tests. The `--position` flag was added in PR #18.

### 6.4 No changes to the probe CLI or feature_probe module

The probe (PR #18) and its `--position` / `--force-composite` flags are reused as-is. No new CLI args, no new feature-probe machinery. The `family_verdict_from_reports` helper is reused unchanged from PR #20.

---

## 7. Test plan + execution sequence

### 7.1 Synthetic-fixture tests (committed)

Per §6.3 above. All run in CI under `pytest`; no network, no real data dependencies. These tests cover:
- The four pure compute functions (correctness under controlled inputs, threshold semantics, within-player rolling, shift-1 semantics).
- The assembler (schema + left-join + GSIS validation + dup-key validation + empty-PBP fast path).

### 7.2 Real-data execution sequence (run-once, reports committed)

1. Run `python scripts/build_pbp_receiver_override.py --seasons 2018-2024`. Produces `data/features_probe/pbp_receiver.parquet`. Inspect coverage by position (WR / TE); fix backfill or fall back to `--coverage-threshold 0.90` for TE per §1.3 criterion 1.
2. Run baseline augment + baseline swap probes (§3.1). Commit the 4 markdown + 4 CSV reports to `reports/`.
3. Compute family verdict via `family_verdict_from_reports([augment_report, swap_report])`.
4. **If `SIGNAL`**: skip steps 5–6 below. Go directly to step 7: write the family summary report (§5.2), commit, then update docs (step 8).
5. **If `NULL`**: run lgb-nb augment + lgb-nb swap probes (§3.2). Commit those 4 markdown + 4 CSV reports.
6. Recompute family verdict via `family_verdict_from_reports([augment_baseline, swap_baseline, augment_lgbnb, swap_lgbnb])`. The recomputed verdict is durable per §1.3 criterion 3.
7. Write the family summary report (§5.2) and commit alongside the probe reports for the verdict path actually taken.
8. Update `TODO.md` #3c + `project_management.md` decision log per §9.

### 7.3 Standard verification gates

Per CLAUDE.md end-of-effort checklist:

- `pytest -v` — full suite, all passing.
- `mypy src tests` — zero violations.
- `ruff check src tests scripts` — zero violations.
- `ruff format --check src tests scripts` — no drift.
- `pytest -v -k "ingest or store or schemas"` — for any change touching PBP ingest or schema (this spec touches `pbp_receiver_features.py` which is downstream of ingest, not ingest itself, but the gate is cheap and runs anyway).

---

## 8. Risks

- **Override coverage gap on TE.** Blocking-only TEs may produce few receiver-active games; trailing-4 will be structurally NaN for them. The §1.3 criterion 1 caveat (lower coverage to 0.90) is the documented escape hatch. The risk is that a substantial fraction of TE rows are dropped, and the surviving-TE verdict is biased toward heavy-receiving TEs. If this is observed, the summary report flags it for the production-builder follow-up (which would need to decide whether to ship features that only apply to ~70-80% of TEs, leaving the rest on baseline behavior).
- **NGS-vs-PBP collinearity in swap mode.** Dropping the two NGS columns (`avg_intended_air_yards_std`, `avg_yac_above_expectation_std`) may interact unexpectedly with the remaining NGS columns (`avg_separation_std`, `percent_share_intended_air_yards_std`). The augment mode controls for this — augment SIGNAL means "PBP cols add orthogonal signal on top of NGS"; swap SIGNAL means "PBP cols dominate the NGS season snapshots." Both are valid family-level signals.
- **Receiver-active-game window stretch.** A WR3 with sporadic usage may have trailing-4 stretching across 8+ calendar weeks; the rolling window is over receiver-active games, not calendar weeks. This could pick up features from a different role (different team / scheme post-trade). The spec accepts this — it matches NGS's own "qualifying weeks" convention.
- **Threshold sensitivity.** The 20-yard "deep target" and 20-yard-line "red zone" thresholds are conventional but not unique. A `NULL` verdict could be partly attributable to threshold choice, not absence of signal. The spec accepts the standard thresholds; sensitivity sweeps are a separate spec only if independently warranted.
- **PBP missing data on early-2018 weeks.** Same edge case as PR #20: trailing-4 at 2018 weeks 1–4 may have insufficient prior seasons to backfill (curated PBP starts at 2018). NaN rate elevated; coverage check enforces 95%. If a future ingest brings 2017 PBP online, the NaN rate drops with no schema or builder change required.
- **Probe verdict drift between override regenerations.** Override is regenerable; if `nfl_data_py` upstream PBP data revisions touch a row, the regenerated override produces subtly different values and the verdict may shift. The summary report's header captures the override mtime + the PBP partition mtimes for traceability.
- **Receiver-level vs team-level confounding (negative result interpretation).** If both team-level (PR #20) and receiver-level (this spec) probes return null for WR / TE, the conclusion is *not* "PBP carries no signal for receivers" — it is "PBP carries no signal for receivers *at the units we've tested* (team + receiver-level air-yards / aDOT). Other refined units (e.g., per-route-concept distributions, depth × leverage interactions, target-quality residuals) remain unexplored." The summary report should make this distinction explicit if both probes close.

---

## 9. Documentation updates on merge

- **`TODO.md` #3c:**
  - Append a paragraph stating the family probe verdict (SIGNAL or NULL).
  - If `SIGNAL`: cite the production-builder follow-up plan candidate and what it scopes (WR-only, TE-only, or both per the verdict cells).
  - If `NULL`: state the family closed at BaselineModel + lgb-nb for receivers, cross-reference the summary report, and note that other refined-unit candidates (per-route-concept, etc.) remain unexplored.

- **`project_management.md`:**
  - Append a "PBP Receiver Family Probe" decision-log entry under the standard format. Date, branch, PR, verdict, what it greenlights or closes.

- **`CONTRIBUTING.md`:**
  - Add a one-paragraph "Regenerating the PBP receiver override" subsection under the existing feature-plan workflow. One example invocation of `scripts/build_pbp_receiver_override.py`, note that the output is not committed and is regenerable from the live PBP partitions.

- **`docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`:**
  - No changes. The team-level family-probe spec stays as historical record of the team-level cut.

---
