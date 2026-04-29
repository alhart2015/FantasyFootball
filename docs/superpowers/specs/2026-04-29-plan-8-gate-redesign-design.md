# Plan 8 — Adoption Gate Redesign — Design

**Status:** approved (brainstorming, 2026-04-29). Ready for implementation plan.
**Date:** 2026-04-29
**Author:** alden + claude
**Builds on:** Plan 6 (PR #15, merged at `995d43f`) — depends on the per-row `results.parquet` schema (with `model_class` column for all five model classes), `POSITION_DISPATCH.factories`, and `tests/backtest/model_metrics.json` cell-level snapshot. Branched from `main` at `995d43f` onto `feat/plan-8-gate-redesign`.

---

## 1. Overview

The PR-10-through-PR-15 model losing streak (Plans 3e / 5 / 5b / 5c / 7 / 6 — all failed §1.3 adoption) traces to two compounding structural problems with the current adoption gate, not to bad data, bugs, or fundamentally bad models:

1. **§1.3 thresholds sit below the per-cell noise floor.** "Composite RMSE strictly lower on ≥12/16 cells AND not worse by >1% on any cell" + "weekly calibration no worse on any cell, mean delta ≥ +0.02" treat sampling variation as systematic regression. There is no significance test gating any criterion; any cell-level wobble within the typical sampling-noise envelope (≥1%) registers as a pass/fail flip. Smoking gun: Plan 6 (Model D ensemble) hit 12/16 RMSE wins (meets the count!) but failed because TE 2024 was +1.24% worse — 0.24pp over the no-regression line, on a single 1081-week cell.

2. **The weekly `[p10, p90]` calibration metric is not load-bearing for any planned downstream consumer.** Plan 5c PM and Plan 6 §96–99 already note this: Draft Hub (rankings, ADP, VORP) consumes mean and rank; start/sit consumes mean and rank; the DFS lineup optimizer consumes mean and ownership. None reads `[p10, p90]` coverage. Plans 3e Phase 2/3, 5, 5b, 5c, and 6 collectively spent five plans optimizing a metric whose failure has no downstream cost. Plan 7 separately showed the assumed mechanism for the calibration miss was wrong (per-stat coverage doesn't decompose to composite coverage by convolution), so several plans were also pulling the wrong end of the distribution.

Plan 8 replaces §1.3 with a gate that (a) treats per-cell deltas as random variables via paired-bootstrap CIs, (b) decides adoption per position rather than globally, (c) gates on RMSE with Spearman as a catastrophic-regression-only check, and (d) treats calibration as informational. It also re-evaluates the four existing peer models (C, C-tuned, C-NB, D) against the most recent backtest run under the new gate, and ships the resulting per-position routing changes in the same PR.

This is strictly an evaluation-machinery + routing-config plan. No model code changes. No feature pipeline changes. No new model classes.

### 1.1 Goals (in scope)

- New pure-stats module `src/projections/backtest/adoption_gate.py` exporting:
  - `BootstrapDelta` dataclass: `(point, lo_95, hi_95, n_paired_rows, n_bootstrap)`.
  - `paired_bootstrap_rmse_delta(residuals_incumbent, residuals_candidate, *, n_bootstrap=1000, seed=42) -> BootstrapDelta`.
  - `paired_bootstrap_spearman_delta(predicted_incumbent, predicted_candidate, actual, grouping, *, n_bootstrap=1000, seed=42) -> BootstrapDelta` — per-group Spearman averaged across groups (year is the group; pooling across years would mix populations because the player set rotates).
  - `verdict_for_position(rmse, spearman, *, spearman_floor=-0.02) -> tuple[Literal["ADOPT", "MARGINAL", "DO_NOT_ADOPT"], str]` — applies the §1.3-replacement rule below and returns a one-line human-readable reason.
  - `PositionVerdict` dataclass bundling the above with a per-year breakdown DataFrame for inspection.
- New CLI `scripts/adoption_gate.py`:
  - Args: `--run <run_dir>` (required), `--candidate <model_class>` (required), `--incumbent <model_class>` (default `baseline`), `--position {QB,RB,TE,WR,all}` (default `all`), `--csv-out <path>` (optional), `--n-bootstrap <int>` (default 1000), `--seed <int>` (default 42).
  - Reads the run's `results.parquet`, filters to `{incumbent, candidate} × position(s)`, pairs rows on `(gsis_id, season, week)`, computes the per-position bootstrap CIs and verdict via the pure-stats module, prints a per-position markdown report to stdout, optionally writes a CSV.
  - Always exits 0; this is informational, humans decide.
- New per-position routing field on `_PositionDispatch`:
  - `default_model_class: str` — name into `factories`. Migrates consumers that hardcode `factories["baseline"]()` to `factories[default_model_class]()`. New helper `production_model_for(position: Position) -> Model` in `src/projections/models/__init__.py`.
- New §1.3 template in `docs/superpowers/specs/_adoption_gate_template.md`. Future model-class specs **copy the template inline** into their own §1.3 (not include / symlink), so each spec carries the gate it was evaluated under as record-of-decision — if we later evolve the gate, old specs still document what they were judged by.
- Re-evaluation of all four existing peer candidates (`lightgbm`, `lightgbm-tuned`, `lightgbm-nb`, `ensemble`) against the most recent backtest run (`data/backtest/run_20260429T003552Z`) using the new CLI. Verdicts recorded in `project_management.md`'s Plan 8 entry. `default_model_class` updated for any (position, candidate) pair where the verdict is `ADOPT` and the candidate is the strongest contender for that position (tie-break: most-negative `rmse_delta.point`).
- Half-hour audit pass on `src/projections/backtest/snapshot.py` tolerance defaults vs the per-cell noise floor measured during re-evaluation. Confirm fine-as-is in the PM doc, or open a follow-up TODO if not.

### 1.2 Non-goals (deferred)

- **No model code changes.** Models A / C / C-tuned / C-NB / D are bit-unchanged.
- **No feature pipeline changes.** Feature builders, feature cache, ingest layer untouched.
- **No new model classes.** Plan 8 evaluates existing peers; new classes are out of scope.
- **No changes to the snapshot regression gate's structure.** `snapshot.py`'s tolerance machinery is for catching code-induced numeric regression on a frozen model; that is a different problem from model-vs-model adoption. Tolerances may be adjusted post-audit if values are also below the noise floor, but the structure (per-row tolerance kinds, fail-closed) stays.
- **No removal of calibration metrics from the harness.** `weekly_calibration_p10p90`, `weekly_calibration_le_p90`, `season_calibration_*` rows continue to be emitted into `model_metrics.json`. Only the *adoption-gate template* stops referencing them. Future consumers that ever care about coverage retain the data.
- **No removal of the §1.3-style three-criteria text from existing shipped specs.** Plans 3e, 5, 5b, 5c, 6, 7 stay as record-of-decision.
- **No retraining, no harness re-run.** Re-evaluation uses the existing per-row parquet from the most recent run.
- **No changes to which years are held out.** Still 2021–2024.
- **No automatic POSITION_DISPATCH updates from the CLI.** The CLI prints a verdict; humans manually edit `default_model_class` based on the report. The CLI never mutates source files or config.
- **No statistical-significance correction for multiple comparisons across positions.** Each position's verdict is an independent decision; we are not testing a global null. (If we ever sweep many candidate classes per position and want family-wise error control, file a follow-up.)
- **No widening of the existing `aggregate_to_season` SAMPLED_SUMMARY-only family gate** (TODO #28). Calibration metrics for QUANTILE / MIXED model classes still skip; not load-bearing here since calibration is no longer in the gate.

### 1.3 Adoption gate (the redesign — replaces all prior §1.3 templates going forward)

Adoption decisions are **per position**. For each `Position P ∈ {QB, RB, TE, WR}`, the adoption gate compares a candidate model class against the incumbent default model class (today: `baseline` for every position; post-Plan-8: per-position).

**Inputs.** Per-row predictions from both classes for the same `(gsis_id, season, week)` rows across all held-out years (2021–2024), pulled from a single backtest run's `results.parquet`. After pairing, position P contributes ~3,000–8,000 paired rows.

**Statistical machinery.** Paired bootstrap with `n_bootstrap=1000`, deterministic seed `42`. Resampling unit is the paired player-week — both candidate and incumbent are scored on the same draw. Within-week and within-player correlations exist but the paired structure cancels most of them; full block bootstrap is overkill for this gate.

**Per-position metrics.**
- **RMSE delta** (`candidate − incumbent`): pooled across all 4 held-out years. Negative = candidate wins. CI is the central 95% of the 1000 bootstrap deltas.
- **Spearman delta**: per-year Spearman computed within each held-out year, then averaged across the 4 years. Negative = candidate worse at ranking. CI is the central 95% of the 1000 bootstrap deltas of the averaged statistic.

Per-cell breakdowns (one row per held-out year) are also emitted for inspection but do **not** gate adoption.

**Verdict rule.**
```
PASS_RMSE      := rmse_delta.hi_95     <  0.0       # 95% CI of (cand-inc) entirely below zero
PASS_SPEARMAN  := spearman_delta.lo_95 > -0.02      # lower CI bound above catastrophic-regression floor

if  PASS_RMSE and PASS_SPEARMAN:   "ADOPT"
elif PASS_RMSE and not PASS_SPEARMAN:   "MARGINAL — RMSE wins, Spearman regresses; investigate before adopting"
elif PASS_SPEARMAN and not PASS_RMSE:   "DO_NOT_ADOPT — RMSE inconclusive"
else:                                   "DO_NOT_ADOPT"
```

Spearman is treated as a catastrophic-regression check only because (a) it is highly correlated with RMSE quality on this data (good RMSE → good rank), so requiring a separate Spearman win is double-counting; (b) the binding production metric for every planned consumer is mean-prediction quality (RMSE), not rank stability beyond "doesn't break"; (c) the floor `-0.02` is ≈ 4× the typical fold-to-fold Model A wobble of 0.005–0.010, so MARGINAL only fires when a model genuinely dislocates rank from mean.

**What the gate does not check** (versus the legacy §1.3):
- No per-cell pass/fail — per-year deltas are informational, only the position-pooled CI gates.
- No Spearman-improvement requirement — only no-catastrophic-regression.
- No calibration check at all. `weekly_calibration_*` and `season_calibration_*` continue to be emitted into the snapshot for monitoring; the adoption decision ignores them.
- No "max worse cell" floor — sampling variation on a single year is not adoption-blocking.

**Tie-breaking** when multiple candidate classes ADOPT for the same position: the candidate with the most-negative `rmse_delta.point` (largest observed RMSE win) is selected. Document the contender chain in the spec applying the gate. (For Plan 8's re-evaluation specifically, the contender chain across the four existing peers is recorded in §6 of this spec and the corresponding `project_management.md` entry.)

**Adoption is manual.** The CLI emits a report. A human reads the verdicts, edits `_PositionDispatch.default_model_class` for any position where the verdict is `ADOPT`, and lands the change as part of the spec's PR. The CLI never writes to source.

---

## 2. Architecture

```
src/projections/
├── models/
│   └── __init__.py                  [+ default_model_class field on _PositionDispatch;
│                                       + production_model_for(position) helper;
│                                       all dispatch entries gain explicit default]
└── backtest/
    └── adoption_gate.py             [NEW: pure-stats module; ~120 LOC; no IO]

scripts/
└── adoption_gate.py                 [NEW: CLI orchestration; ~180 LOC]

docs/superpowers/specs/
└── _adoption_gate_template.md       [NEW: §1.3 template for future specs to inline]

tests/
├── test_backtest/
│   └── test_adoption_gate.py        [NEW: bootstrap correctness, Spearman handling,
│                                      verdict truth table, edge cases; ~25 tests]
└── test_scripts/
    └── test_adoption_gate_cli.py    [NEW: CLI smoke against synthetic parquet; ~6 tests]
```

**Module boundary rationale.** The pure-stats module (`src/projections/backtest/adoption_gate.py`) has no IO, no parquet handling, no string formatting — only numpy and scipy. This makes the bootstrap math trivially testable with synthetic arrays and reusable for any future plan that wants paired CIs (e.g., Plan 9 PBP feature evaluation against a Plan-A-style baseline). The CLI module (`scripts/adoption_gate.py`) is a thin orchestration shell: parquet read → pairing → call into the stats module → format markdown. Mirrors the existing `scripts/diagnose_calibration.py` and `scripts/diagnose_calibration_breakdown.py` shape, so contributors find it where they expect.

**Per-position routing change.** The current `_PositionDispatch` has:

```python
@dataclass(frozen=True)
class _PositionDispatch:
    factories: Mapping[str, Callable[[], Model]]
    feature_builder: ...
    feature_schema: ...
    ngs_stat_type: str
```

After Plan 8:

```python
@dataclass(frozen=True)
class _PositionDispatch:
    factories: Mapping[str, Callable[[], Model]]
    feature_builder: ...
    feature_schema: ...
    ngs_stat_type: str
    default_model_class: str          # NEW — must be a key in factories
```

A new module-level helper:

```python
def production_model_for(position: Position) -> Model:
    """Return a freshly instantiated production-default model for the given position."""
    dispatch = POSITION_DISPATCH[position]
    return dispatch.factories[dispatch.default_model_class]()
```

Migration: any caller currently doing `POSITION_DISPATCH[pos].factories["baseline"]()` for "the production model" switches to `production_model_for(pos)`. Audit `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py`, `scripts/train_baseline.py`, and any other call sites during implementation. Callers that explicitly want a *named* class (the backtest harness's `--model {baseline,lightgbm,...}` arg, sanity-check scripts) keep using `factories[name]()` directly — they are asking for a specific class, not the production default.

**Initial `default_model_class` values.** Every position starts with `default_model_class="baseline"`, identical to today's behavior. Plan 8's re-evaluation step (§6) then updates entries based on the gate verdict — likely QB to `ensemble` or `lightgbm-nb`, RB/TE/WR remaining `baseline`. Whatever the gate says.

## 3. Component contracts

### 3.1 `src/projections/backtest/adoption_gate.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projections.schemas import Position


@dataclass(frozen=True, slots=True)
class BootstrapDelta:
    """Result of a paired bootstrap on a metric delta (candidate - incumbent).

    Negative `point` means the candidate wins (lower error / higher metric is better
    is metric-specific — this dataclass is metric-agnostic).
    """
    point: float           # Observed delta on the full sample, no resampling.
    lo_95: float           # Lower bound of the 95% two-sided CI on the delta.
    hi_95: float           # Upper bound of the 95% two-sided CI on the delta.
    n_paired_rows: int     # Number of paired observations contributing.
    n_bootstrap: int       # Number of bootstrap resamples used.


@dataclass(frozen=True, slots=True)
class PositionVerdict:
    """Per-position adoption verdict bundling RMSE, Spearman, and per-year breakdown."""
    position: Position
    incumbent_class: str
    candidate_class: str
    rmse_delta: BootstrapDelta
    spearman_delta: BootstrapDelta
    verdict: Literal["ADOPT", "MARGINAL", "DO_NOT_ADOPT"]
    reason: str
    per_year_breakdown: pd.DataFrame   # columns: year, rmse_delta_point/lo/hi, spearman_delta_point/lo/hi


def paired_bootstrap_rmse_delta(
    residuals_incumbent: np.ndarray,
    residuals_candidate: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapDelta: ...


def paired_bootstrap_spearman_delta(
    predicted_incumbent: np.ndarray,
    predicted_candidate: np.ndarray,
    actual: np.ndarray,
    grouping: np.ndarray,             # group key per row (year); per-group Spearman, then averaged
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapDelta: ...


def verdict_for_position(
    rmse: BootstrapDelta,
    spearman: BootstrapDelta,
    *,
    spearman_floor: float = -0.02,
) -> tuple[Literal["ADOPT", "MARGINAL", "DO_NOT_ADOPT"], str]: ...
```

**Bootstrap implementation notes.**

- Use `np.random.default_rng(seed)` and `rng.integers(0, n, size=n)` for resampling indices. Same indices for incumbent and candidate within each draw — that's the paired structure.
- RMSE per draw: `sqrt(mean(residuals_X[idx] ** 2))`. The delta per draw is `rmse_cand_draw - rmse_inc_draw`.
- Spearman per draw: split the resampled indices by `grouping[idx]`; for each group, compute `spearmanr(predicted_X[idx_group], actual[idx_group]).statistic`; average across groups (unweighted mean — each year contributes equally regardless of player count). Delta per draw is `mean_spearman_cand - mean_spearman_inc`.
- CI: `np.percentile(deltas, [2.5, 97.5])` for the two-sided 95% interval. The verdict rule's "one-sided 95% upper bound < 0" for RMSE is implemented by checking `rmse.hi_95 < 0` against the two-sided CI's upper bound — equivalent to a one-sided 97.5% upper bound, slightly more conservative than a true one-sided 95%. Acceptable; documented in the spec.

**Edge cases that raise:**
- `n_paired_rows < 100` → `ValueError`. Production positions all have ~3,000–8,000 paired rows; this only fires on a degenerate run worth surfacing.
- `len(residuals_incumbent) != len(residuals_candidate)` → `ValueError` (pairing failure upstream).

**Edge cases that propagate as NaN with a degenerate verdict:**
- Spearman with constant-prediction (one model returns the same value for every row): `spearmanr` returns NaN; downstream `verdict_for_position` returns `("DO_NOT_ADOPT", "degenerate prediction (constant) on candidate")`.

### 3.2 `scripts/adoption_gate.py`

```
python -m scripts.adoption_gate \
  --run data/backtest/run_20260429T003552Z \
  --candidate ensemble \
  [--incumbent baseline] \
  [--position QB|RB|TE|WR|all] \
  [--csv-out reports/adoption_gate_<candidate>_<ts>.csv] \
  [--n-bootstrap 1000] \
  [--seed 42]
```

**Behavior.**
1. Load `<run_dir>/results.parquet`. Validate that it contains both `--incumbent` and `--candidate` in `model_class`; raise listing actually-present classes if not.
2. For each requested position:
   - Filter to `position == P` and `model_class ∈ {incumbent, candidate}`.
   - Pair rows: inner-merge on `(gsis_id, season, week)` between incumbent rows and candidate rows. Warn-and-drop rows that don't pair, with the dropped count in the warning. (Both classes should be scored on identical inputs by the harness; asymmetry indicates an upstream bug worth surfacing.)
   - Compute composite per row: the parquet's `mean` column already holds the per-row composite mean prediction in PPR points (written by `BaselineModel.predict_distribution` / its peers). `actual` is `actual_ppr` from the parquet.
   - Build residuals: `residuals_inc = actual_ppr - mean_inc`; same for cand.
   - Group by `season` (the held-out year) for the Spearman per-group call.
   - Call into `adoption_gate.paired_bootstrap_*` and `verdict_for_position`.
   - Build per-year breakdown via the same bootstrap restricted to one year at a time.
   - Assemble a `PositionVerdict`.
3. Print a per-position markdown table to stdout with columns: position, incumbent, candidate, n_paired, RMSE delta (point/CI), Spearman delta (point/CI), verdict, reason. Per-year breakdown printed below as a sub-table per position.
4. If `--csv-out` is provided, write a long-form CSV with one row per (position, metric, statistic) plus per-year breakdown rows.
5. Exit 0 always.

**No automatic config mutation.** The CLI does not edit `models/__init__.py` or any other source. Adoption is manual.

### 3.3 `docs/superpowers/specs/_adoption_gate_template.md`

A short markdown file containing the exact §1.3 template that future model-class specs inline-copy into their own spec. Inlining (not symlink / include) so each spec carries the gate it was evaluated against as record-of-decision — if we later evolve the gate, old specs still document what they were judged by. Template body is the §1.3 of *this* spec, lightly genericized (`<CANDIDATE_MODEL_CLASS>` placeholder, etc.).

## 4. Error handling + edge cases

| Situation | Behavior |
|---|---|
| Run parquet missing required `model_class` value | CLI raises with list of classes actually present; non-zero exit. |
| `--position X` where X is not one of QB/RB/TE/WR/all | argparse rejects with usage. |
| Asymmetric pairing (row in incumbent without matching candidate, or vice versa) | Warn with dropped-count to stderr; continue with the paired subset. |
| Position has < 100 paired rows | Stats module raises; CLI catches and emits `verdict=DO_NOT_ADOPT, reason="insufficient paired rows (n=<X>)"` for that position; continues with other positions. |
| Either model predicts a constant value for a position | `paired_bootstrap_spearman_delta` propagates NaN; `verdict_for_position` returns `(DO_NOT_ADOPT, "degenerate prediction")`. |
| Bootstrap produces all-zero deltas (impossible in practice) | CI brackets zero exactly; `PASS_RMSE=False`; verdict `DO_NOT_ADOPT`. |
| `production_model_for(position)` called with a position whose `default_model_class` is not in `factories` | Raises `KeyError` at construction time of `_PositionDispatch` via a post-init check (not at lookup time) so config errors fail at import. |
| Existing snapshot regression gate disagrees with new adoption gate verdict | Independent gates; not a conflict. The snapshot gate guards code-induced numeric drift on the frozen baseline; the adoption gate decides which class is the production default. They answer different questions. |

## 5. Testing

### 5.1 `tests/test_backtest/test_adoption_gate.py` (~25 tests)

**`paired_bootstrap_rmse_delta`:**
- Identical residuals → CI brackets zero; `point` = 0.0.
- Candidate residuals = incumbent / 2 → point < 0; CI strictly below zero.
- Candidate residuals = incumbent * 2 → point > 0; CI strictly above zero.
- Determinism: same seed → bit-exact `BootstrapDelta`.
- `n_paired_rows < 100` → raises `ValueError`.
- Length mismatch → raises `ValueError`.
- `n_bootstrap` field stored correctly.

**`paired_bootstrap_spearman_delta`:**
- Synthetic perfect-rank vs reverse-rank → point ≈ -2.0.
- Identical predictions → CI brackets zero.
- Multi-group: year-1 wins +0.1 on cand, year-2 loses -0.1 on cand → averaged delta ≈ 0.
- Constant prediction on candidate → returns NaN BootstrapDelta (no raise).
- Determinism: same seed → bit-exact.

**`verdict_for_position`:**
- Truth table: each of (PASS_RMSE × PASS_SPEARMAN) ∈ {(T,T), (T,F), (F,T), (F,F)} produces the documented verdict + a non-empty `reason`.
- Custom `spearman_floor` parameter respected.
- NaN inputs (degenerate prediction case) → `DO_NOT_ADOPT, reason="degenerate prediction"`.

**Integration on synthetic data:**
- 4-year × 200-row synthetic case where candidate genuinely wins on RMSE → end-to-end ADOPT.
- Same scaffold but candidate is just noisier incumbent → DO_NOT_ADOPT.

### 5.2 `tests/test_scripts/test_adoption_gate_cli.py` (~6 tests)

- Smoke against a 200-row × 2-class synthetic parquet → exits 0; stdout contains "ADOPT" or "DO_NOT_ADOPT".
- `--position QB` filters output to QB only.
- Missing model_class → exits non-zero with helpful error mentioning available classes.
- `--csv-out` writes a CSV with the expected schema.
- Asymmetric pairing produces a warning with a dropped-count.
- `--n-bootstrap 50` runs faster (smoke).

### 5.3 No backtest-gate integration

This script is a one-time-per-candidate decision tool, not a CI gate. Tests run in the default-fast lane (no `@pytest.mark.backtest`). No additions to `tests/backtest/model_metrics.json`. The existing snapshot regression gate is not affected.

### 5.4 `tests/test_models/test_position_dispatch.py` (extend, ~3 new tests)

- Every entry in `POSITION_DISPATCH` has `default_model_class` populated and present in its `factories`.
- `production_model_for(position)` returns an instance of the class the dispatch nominates.
- Constructing a `_PositionDispatch` whose `default_model_class` is not in `factories` raises in `__post_init__`.

## 6. Re-evaluation deliverables (in this same PR)

After the gate is built, run it against `data/backtest/run_20260429T003552Z/results.parquet` for all four candidates × all four positions:

```
candidates = ["lightgbm", "lightgbm-tuned", "lightgbm-nb", "ensemble"]
positions  = [QB, RB, TE, WR]
incumbent  = "baseline"
for cand in candidates:
    python -m scripts.adoption_gate --run data/backtest/run_20260429T003552Z \
        --candidate {cand} --csv-out reports/adoption_gate_{cand}.csv
```

Capture the 16 verdicts + CIs as a table in the Plan 8 entry of `project_management.md`. Determine the contender chain per position (tie-break: most-negative `rmse_delta.point` among ADOPT verdicts). Update `_PositionDispatch[pos].default_model_class` for any position where a candidate ADOPTs and is the strongest contender.

**Strong prior on what the re-evaluation will return** (informed by the per-cell metric tables in Plan 5/5b/5c/6 verdicts; not a pre-commitment):
- **QB**: `ensemble` ADOPTs (Plan 6 showed clean wins on every metric × every fold). `lightgbm-nb` likely also ADOPTs. Tie-break expected to favor `ensemble`.
- **RB**: All four candidates expected to DO_NOT_ADOPT (Plan 5/5b/5c/6 all show ~+0.5–+3% RMSE regression on RB; bootstrap CIs likely include zero or are positive).
- **TE**: Mixed. Plan 6 had 3/4 RMSE wins on TE but the cell deltas are small; pooled bootstrap may or may not clear zero. If MARGINAL, default stays `baseline`.
- **WR**: Plan 6 had 4/4 RMSE wins on WR but each ≤ 0.55%; pooled CI may or may not clear zero. If MARGINAL, default stays `baseline`.

Whatever the gate actually says is what ships. The strong prior is a sanity check on the gate, not a substitute for running it.

**Snapshot-regression-gate audit pass.** Read `src/projections/backtest/snapshot.py` tolerance defaults. Compare against the per-cell RMSE noise floor measured from the bootstrap (the pooled CI half-width / `sqrt(n_years)` is a rough per-year noise estimate). If the snapshot tolerances are within 2× of the noise floor, document "fine as-is" in the Plan 8 PM entry. If they are below the noise floor (i.e., snapshot diff is also at risk of false-positive regressions), file a follow-up TODO; do not change them in this PR (out of scope, would deserve its own diff and reviewer attention).

## 7. Open questions / risks

- **Bootstrap variance under-estimation from rotation between years.** Each held-out year is largely a different player set. Pooling all years' player-weeks and resampling row-wise treats them as exchangeable, which they are not at the player level. The paired structure cancels most of the systematic variation, but bootstrap CIs may still under-estimate true uncertainty by ~10–20%. Mitigation: the verdict rule uses two-sided 95% CI's upper bound (effectively one-sided 97.5%) for RMSE, which is intentionally more conservative than a true one-sided 95%. If we ever discover this matters in practice (ADOPT verdicts that don't survive a year-block bootstrap), file a follow-up to switch to year-block bootstrap.
- **`mean` column semantics across model classes.** For `baseline`, `lightgbm-nb`, `ensemble` (SAMPLED_SUMMARY / MIXED families), `mean` is the per-row Monte Carlo composite mean. For `lightgbm` / `lightgbm-tuned` (QUANTILE family), `mean` is also the composite mean (predicted from p50 + sample-summary aggregation in the harness — confirmed during implementation by inspecting `harness.py`'s mean-emission path). Risk: a future model class that emits `mean` in a different scale would silently break the comparison. Mitigation: a one-row sanity assertion in the CLI that `mean` is in the same range as `actual_ppr` (e.g., within [0, 80] for fantasy points); abort with a clear message if not.
- **Per-position routing breaks consumers that assumed one global default.** Audit during implementation: search the codebase for hardcoded `factories["baseline"]` references and migrate. If any consumer genuinely needs "the same model for every position" (probably none — sanity-check scripts already iterate per position), keep an opt-in helper.
- **Plan 6's QB ensemble wraps Models A and C-NB as children.** If we adopt `ensemble` for QB, the production stack now requires three model artifacts to predict QB (the ensemble + its two children). Storage and load-time cost increase. Acceptable given the QB-only scope; document in the PM entry.
- **What happens to RMSE-only candidates that have great calibration?** The new gate ignores calibration. If a future model class is RMSE-equivalent to baseline but vastly better-calibrated, this gate will say DO_NOT_ADOPT (RMSE PASS criterion fails because CI brackets zero). Acceptable: no current consumer uses calibration. If a future consumer (DFS GPP simulator, late-season uncertainty quantifier) does, that consumer's plan adds calibration back into the gate template at that time.
- **No multiple-comparison correction across the 4 positions or 4 candidates.** Each position/candidate pair is an independent decision. We are not testing a global null. If we ever do an explicit "candidate sweep" plan that fits dozens of model variants and picks the best, that plan adds Holm or BH correction to the gate template at that time.

---

## 8. Rollout

Single PR off `feat/plan-8-gate-redesign`. Phase structure (to be detailed in the implementation plan):

1. Pure-stats module + tests.
2. CLI + tests.
3. `_PositionDispatch.default_model_class` field + helper + consumer migration + dispatch tests.
4. Re-evaluation: run CLI for all 16 (position × candidate) pairs; capture in PM doc; update default_model_class for ADOPT verdicts.
5. Snapshot regression gate audit (read-only pass; document in PM doc).
6. Spec template at `docs/superpowers/specs/_adoption_gate_template.md`.
7. PR, review, merge.

No new dependencies. No data migrations. No CI infrastructure changes. Implementation is bounded by the stats math + the CLI orchestration; estimated 1 working day of implementation + ~30 minutes of compute for re-evaluation.
