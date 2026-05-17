# Upside-Sensitive Ranking Diagnostic — Design

**Status:** draft (brainstorming, 2026-05-16). Ready for user review.
**Date:** 2026-05-16
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core (TODO #33d)
**Branch:** `feat/upside-ranking-diagnostic` cut from `main` at `8ffa607` (PR #46 merge commit).

**Builds on:**
- TODO #33a empirical result (2026-05-11). Production routing was wired into `scripts/project_season.py` and the 2024 retrospective re-ran. Two bug classes fixed (Carson Wentz over-projection, backup QB over-projection class, Malik Nabers rookie over-extrapolation). **Did not move the elite-magnitude needle**: Chase −152 (worse), Gibbs unchanged at −132, Henry unchanged at −106, Saquon unchanged at −88. Conclusion in `project_management.md`: the mean-regression compression on elite players lives in *feature signal coverage*, not in model class.
- `src/projections/aggregation/season.py:aggregate_to_season` (Plan 3d). Re-derives per-row seeds, regenerates per-week samples via `score_distribution`, sums positionally to `n_samples` season-total samples, summarizes `season_mean / p10 / p50 / p90`. Already shipped, validated, deterministic. **Not currently called by `scripts/project_season.py`** (which does naïve sum-of-weekly-means).
- `scripts/compare_predictions_to_actuals.py`. Existing 2024/2025 retrospective: prints top-10-per-position with pred vs actual delta. Consumes `reports/season_projection.csv`'s `season_total_mean` column.

---

## 1. Goals & success criteria

### 1.1 Goal

Answer one empirical question before committing to TODO #33b (decomposed targets with factor-appropriate sub-models) or TODO #33c (forward-looking team-context features), both of which are weeks-of-work scoped:

> **Does the model's existing per-week distribution already contain elite-season signal that ranking-by-`season_mean` throws away?**

Concretely: for the 2024 actual top-N finishers at each position, what does the predicted `season_p90` look like? Does ranking by `season_p90` (or a blend, or `P(season ≥ elite threshold)`) recover more of the actual top-N than ranking by `season_mean`?

The diagnostic outputs a markdown report + machine-readable CSV. A `SIGNAL` verdict greenlights Phase 2 (production ranking surface — separate spec). A `NULL` verdict closes the upside-from-existing-distributions hypothesis and confirms the next move is real model work (33b / 33c).

This is explicitly **not** a model-training task. No model architecture changes. No new features. The model output we have is fixed input; only the consumption of that output is examined.

### 1.2 Architectural prior

The hypothesis behind 33d (per TODO #33 §33d): trailing-N features describe a player's recent statistical past. Ridge regularization compresses any single feature's coefficient, so the *mean* prediction for an elite breakout looks like the player's prior baseline. But the model also emits a full per-week distribution. If the model's weekly variance is roughly calibrated, summing 17 weekly distributions Monte-Carlo-style produces a season distribution whose **upper tail** may approach the actual elite season totals, even when the mean does not.

**Magnitude prior.** Genuinely uncertain. Three plausible outcomes:

| Outcome | What we'd see | Implication |
|---|---|---|
| **Strong** | Chase 2024 `season_p90` ≈ 380; rank-by-p90 puts Chase in top-3 WR (actual #1) | Ship Phase 2 ranking surface. Free win. |
| **Weak** | Chase `season_p90` ≈ 320 (better than mean 250 but well short of actual 403); blended metric beats mean on rank-recovery at some positions but not all | Phase 2 might still be worth building as an *optional* secondary metric; less urgent. |
| **Null** | `season_p90` for elites is ~280, barely above their mean; whole distribution is shifted-down | Close 33d; pivot to 33b/33c. The model's distribution is uniformly under-confident on upside, not just its mean. |

The composite [p10, p90] calibration shortfall on RB/TE/WR (Plan 5c / Plan 6: ~6pp under-coverage, mean delta -0.062) is upper-bound evidence for the **Null** outcome — if the season distribution is uniformly compressed, the upper tail is compressed too. But the calibration shortfall lives in *count-stat* upper tails specifically (per Plan 7 Phase 0 diagnostic), and elite seasons are driven by both yards (Gaussian-shaped) and TDs (count). So the diagnostic is genuinely informative.

Conservative null prior: somewhat more likely than the **Strong** outcome but materially less likely than 50/50. The diagnostic is cheap enough that running it is correct regardless.

### 1.3 Success criteria

The diagnostic **ships** when all three pass:

1. **Data plumbing:** `project_season.py` produces a per-(gsis_id, season) frame with `season_mean / p10 / p50 / p90` via `aggregate_to_season` for the target season. Distribution columns persist alongside the existing naïve `season_total_mean` (which stays unchanged for backward compatibility with `compare_predictions_to_actuals.py` and the VORP/cheat-sheet CSV contracts).

2. **Diagnostic completeness:** Markdown report covers `--season 2024` (primary) and `--season 2025` (confirmation). For each season × each position, the report includes:
   - Per-actual-top-12 player table: `actual_total`, `season_mean`, `season_p90`, `season_p_above_elite`, rank under each metric, rank-error under each metric.
   - Per-metric rank-recovery summary: top-K positional overlap for K ∈ {5, 12, 24}; median absolute rank-error for the top-5 actual; Kendall's tau over players with ≥ 6 projected weeks.
   - Verdict per metric per position: SIGNAL (clearly beats mean) / MARGINAL (sometimes beats mean) / NULL (does not beat mean).

3. **Phase 2 decision gate:** A single greenlight / marginal / no-greenlight verdict at the end of the report. Decision rule (committed; revisable in spec review). Uses the cell verdict defined in §3.5 (SIGNAL / MARGINAL / NULL / REGRESSION per (position, season, metric) cell):
   - **Greenlight** iff some single non-mean metric M is SIGNAL at ≥ 3 of 4 positions in **both** 2024 AND 2025. ("Same metric, multiple positions, both years" — the strong-signal bar.)
   - **Marginal** iff (a) some metric M is SIGNAL at ≥ 3 of 4 positions in exactly one of {2024, 2025} OR (b) some metric M is SIGNAL-or-MARGINAL at ≥ 3 of 4 positions in both years. (Phase 2 spec is written but flagged as low-confidence; sweep blend weights / thresholds as part of Phase 2 itself rather than committing to one metric.)
   - **No greenlight** otherwise — close 33d; the upside signal isn't there at the strength needed to justify a Phase 2 ranking-surface build.

The per-position-per-season-per-metric verdict cells defined in §3.5 are the atoms; this gate rolls them up across positions and seasons. The rule is intentionally strict-on-greenlight, lenient-on-marginal — committing to weeks of Phase 2 work needs the strong signal.

4. **Verification gates green:** `mypy src tests scripts` strict + `ruff check src tests scripts` + `ruff format --check src tests scripts` clean. Relevant pytest subset green (new aggregation hook test in `project_season.py` + diagnostic script smoke).

### 1.4 Out of scope

- **Production ranking surface.** Phase 2 only. Adding columns to `VorpTableSchema` / `SnakeCheatSheetSchema`, threading new metrics through `generate_vorp_table.py` / `generate_snake_cheat_sheet.py`, etc., are all conditional on the Phase 1 decision gate and live in a separate spec.
- **Cross-player joint sampling.** The diagnostic treats per-player season distributions as independent. Joint draws (e.g., "P(Chase finishes top-5 WR)") would require correlated sampling across all WRs, which the current model doesn't support (TODO #1). Ranking is by per-player marginal metric only.
- **Model retraining / feature changes.** 33d is explicitly about consumption, not model improvement. If the diagnostic returns NULL, the next steps are 33b/33c — themselves separate specs.
- **Schema persistence of `season_p10/p50/p90` for `ProjectionSeasonSchema`.** That schema *already* has those columns (`src/projections/schemas.py:794-799`); the gap is that `project_season.py` doesn't currently produce a `ProjectionSeasonSchema`-validated parquet. This spec produces a CSV via `aggregate_to_season`'s output but does **not** plumb full `ProjectionSeasonSchema` parquet writes. That's downstream-of-Phase-2 plumbing.
- **K / DST.** Not produced by `project_season.py` today (TODO #10).

---

## 2. Components

### 2.1 Producer / consumer artifact map

| Producer | Artifact | Consumer |
|---|---|---|
| `project_season.py` (extended) | `reports/season_projection.csv` (unchanged) | existing surfaces (`compare_predictions_to_actuals.py`, `generate_vorp_table.py`, `generate_snake_cheat_sheet.py`) |
| `project_season.py` (extended) | `reports/season_projection_weekly_<season>.parquet` (new — `ProjectionWeeklySchema`-validated) | `diagnose_upside_ranking.py` |
| `project_season.py` (extended) | `reports/season_projection_distributions_<season>.csv` (new — quantile summary) | `diagnose_upside_ranking.py` for the `mean / p90 / blend` metrics; ad-hoc analysis |
| `diagnose_upside_ranking.py` (new) | `reports/upside_ranking_diagnostic.md` | user (Phase 2 decision gate) |
| `diagnose_upside_ranking.py` (new) | `reports/upside_ranking_diagnostic_table.csv` (per-player per-metric ranks) | user (drill-down) |

The naïve CSV stays byte-identical so downstream surfaces don't break. The new artifacts are additive.

### 2.2 `scripts/project_season.py` extension

Today the script ends with a `groupby.agg(season_total_mean=("mean", "sum"), ...)` over the in-memory `weekly` frame and writes one CSV. The extension adds three lines after that point:

1. `weekly.to_parquet(<weekly parquet path>)` — `ProjectionWeeklySchema`-validated frame, one row per `(gsis_id, season, week)`, includes the per-row `params` blob needed for `aggregate_to_season`.
2. `season_dist = aggregate_to_season(weekly, ruleset=ruleset, n_samples=10_000)` — proper Monte-Carlo aggregation giving `season_mean / season_p10 / season_p50 / season_p90` per player.
3. Merge in `full_name / team` from `id_map` (mirror of the existing naïve CSV's name lookup) and write `<distributions CSV path>`.

**Determinism.** `aggregate_to_season` is deterministic via per-row seeds (`derive_row_seed`); re-running with the same `weekly` frame produces bit-identical distributions. No model-snapshot rows are touched.

**Cost.** `n_samples=10_000` × ~17 weeks × ~150 players × 4 positions ≈ 100M sample draws per season. `score_distribution` is the same code path the backtest harness uses per-row today; cost is single-digit minutes on top of `project_season.py`'s existing multi-minute model-fitting time. Acceptable for a non-CI script.

**Why persist a weekly parquet at all** (vs computing `P(season ≥ threshold)` inline in `project_season.py`): decouples the expensive predict step from the cheap analysis step. Each diagnostic re-run (sweeping blend weights, changing the elite threshold, swapping K cutoffs) costs seconds instead of minutes, and the same weekly parquet can be re-aggregated under alternate rulesets (half-PPR, standard) without retraining.

### 2.3 `scripts/diagnose_upside_ranking.py` (new)

New CLI. Reads the weekly parquet + the distributions CSV + actuals from `data/raw/weekly_stats/season=<s>`, computes per-metric ranks, writes a markdown report and a per-player CSV.

```
python scripts/diagnose_upside_ranking.py \
  --seasons 2024 2025 \
  --raw-root data/raw \
  --weekly-parquet-template "reports/season_projection_weekly_{season}.parquet" \
  --distributions-csv-template "reports/season_projection_distributions_{season}.csv" \
  --out "reports/upside_ranking_diagnostic.md"
```

**Computed metrics** (each is a single scalar per player; ranks descend):

| Metric ID | Definition | Source | Notes |
|---|---|---|---|
| `mean` | `season_mean` | distributions CSV | Baseline — what every draft surface ships today. |
| `p90` | `season_p90` | distributions CSV | Primary 33d candidate. |
| `blend_70_30` | `0.7 · season_mean + 0.3 · season_p90` | distributions CSV | One blended point as a "Pareto-ish" intermediate. |
| `p_elite` | `P(season ≥ elite_threshold[position])` | re-aggregate weekly parquet via `aggregate_to_season(return_samples=True)`; empirical over per-player MC array | Threshold defined in §2.4. |

Blend coefficient (`0.7 / 0.3`) is a single committed default; the diagnostic doesn't sweep blend weights in Phase 1. Sweep is a Phase 2 optimization if greenlit.

**`return_samples=True` on `aggregate_to_season`** is a small additive change: currently returns only the per-(gsis_id, season) summary frame; with the flag, also returns a dict `(gsis_id, season) → np.ndarray[n_samples]`. Memory cost: 800 players × 10,000 samples × 8 bytes ≈ 64 MB per season; held only for the lifetime of the diagnostic run. Type narrowing: callers without the flag get back exactly `pd.DataFrame` as today (backward-compatible).

**Rank-recovery measurements** (per-metric, per-position, per-season):
- **Top-K positional overlap** for K ∈ {5, 12, 24}: `|predicted_top_K ∩ actual_top_K| / K`. Tightest test on K=5 (true elite), broadest on K=24 (starter universe).
- **Median absolute rank-error for the top-5 actual**: for each player in actual top-5, `|predicted_rank − actual_rank|`; report median. Lower is better. This is the metric that most directly answers "does this metric correctly identify the elite tier."
- **Kendall's tau** over players with ≥ 6 projected weeks (filter prevents noise from injury-shortened seasons swamping the tau).

### 2.4 Elite threshold definition

`P(season ≥ elite_threshold)` requires an explicit threshold per position. Committed default:

> `elite_threshold[position]` = mean over `s in {2019, 2020, 2021, 2022, 2023}` of the 5th-highest actual season ESPN-PPR points at position `s`, computed over players with ≥ 8 games played that season.

Rationale: top-5-at-position is "true elite" — the Chase/McCaffrey/Kelce tier. Five years averages out one-season noise (2020 COVID, 2021 17-game introduction). ≥8-games filter removes injured stars whose pro-rated season would dominate the top-5 but who aren't representative of a full elite season. Using *actual* historical data avoids self-prophecy (our own predictions don't define the threshold).

Per-position thresholds will be computed at script run time from `data/raw/weekly_stats/season=<s>` and printed in the diagnostic report header for transparency. Rough order-of-magnitude expectations (ESPN PPR):
- QB: ~370 (Lamar / Allen / Hurts tier)
- RB: ~330 (CMC / Henry / Gibbs tier)
- WR: ~320 (Chase / Jefferson / Tyreek tier)
- TE: ~210 (Kelce / Bowers / Andrews tier)

These are illustrative; the script computes the actual numbers. Threshold variability across years is itself a check on whether the elite tier is stable; if the per-year top-5 means swing wildly, the threshold's interpretation gets noisier.

### 2.5 Markdown report structure

```
# Upside-Sensitive Ranking Diagnostic — <season1>, <season2>

## Setup
- Predictions: <distributions-csv paths>
- Actuals source: data/raw/weekly_stats/season={season1, season2}
- Elite thresholds (computed): QB=<x>, RB=<y>, WR=<z>, TE=<w>
- Ruleset: ESPN PPR
- MC samples: 10000 per player per season

## <season>: per-position diagnostic

### QB
| actual_rank | full_name | actual | mean | p90 | blend_70_30 | p_elite | rank_mean | rank_p90 | rank_blend_70_30 | rank_p_elite |
| 1 | Lamar Jackson | 469.5 | 427.0 | <x> | <y> | <z> | 1 | <r> | <r> | <r> |
| ... (top 12) ... |

| Metric | top-5 overlap | top-12 overlap | top-24 overlap | median |rank_err| top-5 | Kendall τ |
| mean | 0.6 | 0.83 | 0.79 | 1.5 | 0.71 |
| p90 | 0.8 | 0.92 | 0.83 | 0.5 | 0.72 |
| ... |

Verdict (this position, this season): p90 beats mean | blend marginal | p_elite NULL

### RB / WR / TE (same shape)

## <season>: position-level summary

## Cross-season summary
| position | metric | season1_verdict | season2_verdict |
...

## Phase 2 decision

Greenlight | Marginal | No greenlight

Reasoning: <one paragraph>
```

### 2.6 Shared helper extraction

Lift `_actual_ppr_total` from `scripts/compare_predictions_to_actuals.py` into a sibling module `scripts/_actuals_helper.py` (matches the existing `scripts/_run_single_backtest.py` / `_coverage_check_refined.py` / `_one_off_revive_opp_epa.py` underscore-prefix convention for script-local helpers). Both consumers (`compare_predictions_to_actuals.py` post-refactor and `diagnose_upside_ranking.py`) import it. Avoids duplication of the PPR scoring loop and the `(gsis_id, position)` groupby. No new `src/` module needed; out-of-scope-creep risk minimal — single function, single test, single import-site update.

---

## 3. Algorithm details

### 3.1 `aggregate_to_season(return_samples=True)`

Today's signature:
```python
def aggregate_to_season(weekly: pd.DataFrame, *, ruleset: Ruleset, n_samples: int = 10_000) -> pd.DataFrame
```

Extended:
```python
def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
    return_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]
```

When `return_samples=True`, the function additionally collects the `season_samples` array per `(gsis_id, season)` group inside the existing loop (no extra MC cost — the arrays are already materialized; they're just discarded today after the quantile summary). Memory cost: 4 bytes × `n_samples` × n_players × n_seasons ≈ 32 MB per season for ~800 players × 10k samples × 4 bytes (float32 packing) or 64 MB at float64. Acceptable in-memory for a one-off diagnostic.

Type narrowing: callers that don't pass `return_samples=True` get back exactly `pd.DataFrame` as today. Backward-compatible.

### 3.2 Top-K positional overlap

```python
def top_k_overlap(pred_rank: pd.Series, actual_rank: pd.Series, k: int) -> float:
    pred_top = set(pred_rank.nsmallest(k).index)
    actual_top = set(actual_rank.nsmallest(k).index)
    return len(pred_top & actual_top) / k
```

Indexed by `gsis_id`. `nsmallest` because ranks are 1=best.

### 3.3 Median absolute rank-error for top-5 actual

```python
def top5_rank_err(pred_rank: pd.Series, actual_rank: pd.Series) -> float:
    top5 = actual_rank.nsmallest(5).index
    return float((pred_rank.loc[top5] - actual_rank.loc[top5]).abs().median())
```

### 3.4 Kendall's tau

`scipy.stats.kendalltau(pred_score, actual_total)` over players in the intersection of both with `n_weeks ≥ 6` projected. Existing scipy dependency (per `pyproject.toml`).

### 3.5 Per-metric verdict at the (position, season) cell level

| Condition | Cell verdict |
|---|---|
| Metric's top-12 overlap > mean's top-12 overlap by ≥ 1/12 (i.e., recovers ≥ 1 more player out of 12) AND median |rank_err| top-5 ≤ mean's value | **SIGNAL** |
| Metric beats mean on at least one of {top-12 overlap, top-5 rank-err} but not both | **MARGINAL** |
| Metric is no better than mean on either | **NULL** |
| Metric is strictly worse than mean on both | **REGRESSION** |

Cell verdicts roll up to the cross-season summary per §1.3 #3 decision gate.

### 3.6 Edge cases

- **Players with `n_weeks < 6`.** Excluded from Kendall's tau computation (noise). Included in top-K overlap if they rank into the top K under any metric (correctly penalizes a metric that over-ranks injured stars). Their ranks are still computed and shown in the per-player table.
- **Players in predictions but not actuals (drafted but never played).** Their `actual_total = 0`, `actual_rank = N+1`. Their predicted ranks are still reported — this is exactly how the production draft surfaces would behave on a player who busts.
- **Players in actuals but not predictions (mid-season call-ups; surprise FA signings).** Excluded from all metrics. Caveat noted in the report header. Expected to be rare for top-12 outcomes; nonzero for top-24+.
- **Position mismatches** (player switched positions mid-season). Use the prediction's position for grouping, same as `compare_predictions_to_actuals.py` already does.

---

## 4. CLI surface

```
$ python scripts/project_season.py --season 2024
... (existing output) ...
Wrote season totals CSV: reports/season_projection.csv
Wrote season distributions CSV: reports/season_projection_distributions_2024.csv
Wrote weekly distributions parquet: reports/season_projection_weekly_2024.parquet

$ python scripts/project_season.py --season 2025
... (same shape) ...

$ python scripts/diagnose_upside_ranking.py --seasons 2024 2025
Loading 2024 weekly parquet (1782 rows across 4 positions)
Loading 2025 weekly parquet (1893 rows across 4 positions)
Computing elite thresholds from 2019-2023 actuals:
  QB elite_threshold = 371.4 (top-5 mean, ≥8 games)
  RB elite_threshold = 332.8
  WR elite_threshold = 318.6
  TE elite_threshold = 208.2

=== 2024 / QB ===
... (per-player table + per-metric summary) ...
=== 2024 / RB ===
... (same) ...
=== 2025 / QB ===
... (same) ...

=== Cross-season summary ===
... (verdict table) ...

=== Phase 2 decision ===
GREENLIGHT — p90 beats mean on top-12 overlap at 4/4 positions in 2024 and 3/4 in 2025; blend_70_30 marginal.

Wrote diagnostic report: reports/upside_ranking_diagnostic.md
Wrote per-player CSV: reports/upside_ranking_diagnostic_table.csv
```

---

## 5. Test plan

1. `tests/test_aggregation/test_season_return_samples.py` — `aggregate_to_season(return_samples=True)` returns a tuple with the same summary frame as without, plus a dict whose values are `(n_samples,)` float arrays whose mean ≈ each row's `season_mean`.
2. `tests/test_scripts/test_project_season_artifacts.py` — extract the new "write CSVs + parquet given an in-memory weekly frame" logic into a helper inside `project_season.py` (e.g., `_write_season_artifacts(weekly, ruleset, out_dir, season)`); unit-test the helper directly on a synthetic `ProjectionWeeklySchema`-valid weekly frame. Avoids the multi-minute end-to-end model-fitting cost; tests the artifact-writing contract that's actually new in this spec.
3. `tests/test_scripts/test_upside_ranking_metrics.py` — synthetic per-player MC arrays: verify `top_k_overlap`, `top5_rank_err`, Kendall's tau, and per-metric verdict logic (§3.5) match hand-computed values on a 6-player toy set.
4. `tests/test_scripts/test_diagnose_upside_ranking_cli.py` — integration smoke: write fake distributions CSVs + weekly parquets for 2 synthetic seasons + 4 positions, run the CLI end-to-end, assert markdown report exists, has the Phase-2-decision line, and per-position sections are present.
5. `tests/test_scripts/test_actuals_helper.py` — `_actual_ppr_total` extraction parity test: shared helper produces the same output as the inline version in `compare_predictions_to_actuals.py` did pre-refactor (golden numbers on a small synthetic `weekly_stats`-shaped fixture).

All new tests live under existing directories (`tests/test_aggregation/`, `tests/test_scripts/`). No new test directory created. No model-snapshot updates required (Phase 1 doesn't touch any model). No pandera schema additions (the new CSV/parquet are intermediate artifacts; persistent schemas land if/when Phase 2 ships).

---

## 6. Risks

1. **Independence of weekly draws understates season variance.** `aggregate_to_season` samples each week independently, then sums positionally. Real weeks have correlated outcomes (a healthy player has higher mean *and* higher variance across the season than a same-mean player with injuries). Composite [p10, p90] under-covering by ~6pp (Plan 5c/6) is partly this. The diagnostic's `season_p90` is therefore biased toward the center vs the true season distribution. **Mitigation:** none in Phase 1 — this is the *same* bias the production aggregator already has. If Phase 2 lands, the ranking surface inherits the same bias, which is fine for *relative ranking* even if absolute coverage is off.

2. **The diagnostic's verdict is sensitive to the elite-threshold definition.** Top-5-per-position is a defensible choice but not the only one. Top-3 (truer elite) vs top-12 (starter-level) would shift `p_elite` materially. **Mitigation:** report the threshold value in the report header; the per-K overlap (K ∈ {5, 12, 24}) gives broader coverage; user can change the threshold in spec review.

3. **Multiple-comparison effect.** With 4 metrics × 4 positions × 2 seasons = 32 verdict cells, some cells will look better by chance. The decision gate's "≥ 3 of 4 positions in *both* years" requirement (§1.3 #3) is the multiple-comparison correction. If the decision becomes marginal, treat it as **No greenlight** rather than barely-greenlight — the bar for committing weeks of Phase 2 work should be the strong signal, not the marginal one.

4. **Diagnostic uses post-hoc tuning.** Picking `blend_70_30` after seeing 2024 results would be data-snooping. **Mitigation:** the blend weight (0.7/0.3) is committed in the spec *before* running the diagnostic; not tuned on the diagnostic's outputs. Phase 2 may sweep the weight if greenlit, but that's a Phase 2 decision.

5. **2025 already had a retrospective run.** `compare_prediction_to_actuals_2025_results.txt` exists in the working tree; the user has already seen 2025 mean-ranking output. The diagnostic adds the *distribution* dimension on top of that existing mean ranking; it doesn't conflict, but the user has already formed priors from the existing 2025 retrospective. **Mitigation:** the diagnostic uses 2024 as primary (anchoring on TODO #33's framing); 2025 is secondary confirmation.

6. **Project_season.py extension touches a script downstream surfaces consume.** `generate_vorp_table.py` and `generate_snake_cheat_sheet.py` read `reports/season_projection.csv`. The spec preserves that CSV byte-identical (only *additional* files are written). Regression test: existing snake cheat sheet / VORP CLI integration tests should still pass.

---

## 7. Decision tree post-diagnostic

```
                  diagnostic verdict
                  ┌────────┼────────┐
              GREENLIGHT  MARGINAL  NO GREENLIGHT
                  │         │            │
                  │         │            └─ close 33d; pivot to 33b (decomposed
                  │         │               targets with factor-appropriate sub-
                  │         │               models on WR yards/TDs) or 33c
                  │         │               (forward-looking Vegas team-context
                  │         │               features family probe).
                  │         │
                  │         └─ write Phase 2 spec with explicit low-confidence
                  │            flag; consider sweeping blend weights and elite
                  │            thresholds as part of Phase 2 itself rather than
                  │            committing to a single metric.
                  │
                  └─ write Phase 2 spec: new column on VorpTableSchema +
                     SnakeCheatSheetSchema for the winning metric, plumb through
                     generate_vorp_table.py + generate_snake_cheat_sheet.py,
                     CLI flag to switch ranking mode, update docs. Estimated
                     scope: similar to PR #46 (snake cheat sheet) — one feature
                     spec + plan + execute cycle.
```

The diagnostic is cheap (no model retraining); the decision tree is the load-bearing output.

---

## 8. Workflow

Per project workflow: this spec → implementation plan (via `writing-plans` skill) → execute on branch `feat/upside-ranking-diagnostic` → PR → merge.

Update on merge: TODO #33d entry in `TODO.md` records the diagnostic verdict and the Phase 2 decision; `project_management.md` gets an entry summarizing the run; `draft_ready_checklist.md` notes the diagnostic output as supporting context for §2 (draft rankings) — no checklist item flips on Phase 1 alone.
