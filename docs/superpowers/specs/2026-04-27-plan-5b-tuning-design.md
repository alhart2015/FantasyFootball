# Plan 5b — Optuna Hyperparameter Tuning of Model C — Design

**Status:** approved (brainstorming, 2026-04-27).
**Date:** 2026-04-27
**Author:** alden + claude
**Builds on:** Plan 5 (merged at `6a629a7`, PR #11) for `LightGBMModel`, `QuantileDistribution`, the `POSITION_DISPATCH.factories` dict, and the dual-model walk-forward backtest harness this plan extends.

**Plan-3 / Plan-5 series context:**

- **Plan 3a..3e** — Model A (Ridge) per position + parametric calibration tightening.
- **Plan 5 (merged at `6a629a7`):** Model C (LightGBM quantile regression) shipped as a peer of Model A. Adoption gate failed all three §1.3 criteria — Model A stays the production default.
- **Plan 5b (this design):** Optuna-based hyperparameter tuning of Model C. Tuned variant lands as a third model class **Model C-tuned** that coexists with both A and C. Diagnostic-mode budget. Closes the Plan 5 §10 follow-up "if results justify, tune."
- **Plan 5c (conditional, future):** Per-fold tuning + production-default switch from Model A to Model C-tuned. Only authored if Plan 5b's diagnostic flips the adoption gate.
- **Plan 6 (future):** Model D — ensemble of Model A + best of {C, C-tuned}. Independent of Plan 5b's verdict.

---

## 1. Overview

Plan 5 shipped Model C (per-stat LightGBM quantile regression) with hand-set hyperparameter defaults (`n_estimators=2000, learning_rate=0.05, max_depth=6, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0`). Spec §1.3 explicitly deferred hyperparameter tuning to "a focused follow-up if results justify." The post-Plan-5 adoption-gate result was unambiguous:

- **Composite RMSE:** Model C strictly lower on 1/16 cells; max +3.85% worse (TE 2023); 11/16 cells exceed the 1% regression bound. **FAIL.**
- **Spearman top-N:** 12/16 cells fall outside the ±0.005 tolerance. **FAIL.**
- **Calibration:** Model C no worse on 1/16 cells; mean delta -0.0857 (target ≥ +0.02). **FAIL.**

QB cells are the only positive movers; RB / TE / WR each regress by 2-4% RMSE and 0.09-0.15 calibration. The Plan 5 post-mortem identified four plausible mechanisms for Model C's loss; mechanism #2 (hand-set hyperparameters not tuned) is the cheapest to test and the one most likely to materially close the calibration gap on its own. Mechanisms #1 (no shared prior across sub-models), #3 (5-quantile interpolation too coarse), and #4 (per-stat independent training) are deferred — each is a separate model-class change and outside Plan 5b's scope.

This plan delivers Model C-tuned (LightGBM with Optuna-tuned hyperparameters) as a new peer model class. Both Model C (untuned) and Model C-tuned coexist under the existing `Model` Protocol. The backtest harness regenerates the snapshot with three model classes side-by-side; the per-cell A vs C vs C-tuned comparison is durable and committed.

### 1.1 Goals (in scope)

- New `LightGBMTunedModel` implementing the existing `Model` Protocol, with per-position factories (`qb_lightgbm_tuned`, `rb_lightgbm_tuned`, `te_lightgbm_tuned`, `wr_lightgbm_tuned`) mirroring the `LightGBMModel` factory shape. Implementation reuses `LightGBMModel`'s training + prediction logic; only the hyperparameter source differs.
- New tuned-params JSON file `data/tuned_params/lightgbm.json`, keyed by `(position, stat)`, checked into git. Loaded at fit time; merged into `LGBM_DEFAULTS` to overwrite the eight tuned axes (`learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`) per sub-model.
- New `scripts/tune_lightgbm.py` Optuna driver with `--position`, `--stat`, `--all`, `--trials`, `--out`, `--studies-db`, `--seed` flags. Run manually; not on the gated CI path.
- Optuna study persistence at `data/tuned_params/optuna_studies.db` (SQLite); resumable.
- `POSITION_DISPATCH.factories` extended with the `"lightgbm-tuned"` key per position.
- Backtest harness `--model` selector accepts `lightgbm-tuned`. The compound `--model all` value runs `baseline + lightgbm + lightgbm-tuned` together.
- Snapshot file `tests/backtest/model_metrics.json` extended with 368 new rows under `model_class="lightgbm-tuned"` (skipping the 32 `season_calibration_*` rows per the existing SAMPLED_SUMMARY family gate; matches Plan 5's row pattern). Total: 768 → 1136 rows.
- Default-on smoke test extended to assert `lightgbm-tuned` produces finite metrics on the (WR, 2024) cell.
- Diagnostic report appended to `project_management.md` with the per-cell A vs C vs C-tuned comparison; gate-pass-or-fail call recorded.
- `optuna` added to `pyproject.toml` dependencies.

### 1.2 Non-goals (deferred to later plans)

- **No per-fold tuning.** Hyperparameters are tuned once on the most-recent valid split (train 2018-2022 with 2022 as early-stop val; trial scorer 2023). Tuned params are reused across all four backtest folds (held-out 2021/2022/2023/2024). Per-fold tuning is Plan 5c if the diagnostic warrants it.
- **No production-default switch.** Even if Model C-tuned passes the adoption gate, the default-model selection is a separate config change in Plan 5c. Plan 5b ships the diagnostic and the persistent comparison.
- **No widening of `aggregate_to_season` to QUANTILE family.** TODO #28 stays open. Model C-tuned skips the same 32 `season_calibration_*` rows Model C does.
- **No new features, no target decomposition, no shared-prior or multi-output training, no removal of Model A or Model C.**
- **No new ingest, no PBP / EPA, no calibration-aware fitting.**
- **No quantile-grid widening** (the 5-quantile [0.05, 0.10, 0.50, 0.90, 0.95] grid is preserved). Mechanism #3 is its own change.

### 1.3 Diagnostic exit criteria

After running the 24 Optuna studies, persisting tuned params, and regenerating the snapshot, Plan 5b is **complete on either** of:

- **Tuned C passes the Plan 5 §1.3 adoption gate.** Per-cell deltas (C-tuned vs A) satisfy: RMSE strictly lower on ≥ 12 of 16 cells, no cell worse by > 1%; Spearman top-N within ±0.005 on every cell; weekly mean `[p10, p90]` coverage no worse on any cell, mean across 16 cells improves by ≥ 0.02. → File Plan 5c (per-fold tuning + production-default switch).
- **Tuned C fails the gate.** Document deltas in `project_management.md`; the next-step decision (Plan 6 ensemble vs. TODO #3 PBP feature work vs. TODO #23 target decomposition) goes back to the user.

In either case, Model C-tuned ships as a peer model class. Infrastructure is reused regardless of verdict.

---

## 2. Architecture

```
src/projections/
├── distributions/                  [unchanged]
├── models/
│   ├── base.py                     [unchanged]
│   ├── baseline.py                 [unchanged]
│   ├── lightgbm.py                 [unchanged: Model C / untuned]
│   ├── lightgbm_tuned.py           [NEW: Model C-tuned]
│   └── __init__.py                 [POSITION_DISPATCH.factories += "lightgbm-tuned"]
├── schemas.py                      [unchanged]
└── backtest/
    ├── harness.py                  [+ "lightgbm-tuned" + "all" arg parsing]
    └── snapshot.py                 [unchanged: row keying already covers this]

scripts/
├── backtest.py                     [+ "lightgbm-tuned" / "all" arg parsing]
├── tune_lightgbm.py                [NEW: Optuna search driver]
├── train_baseline.py               [+ "lightgbm-tuned" model selector]
├── predict_2024.py                 [+ "lightgbm-tuned" model selector]
└── sanity_check_baseline.py        [+ "lightgbm-tuned" model selector]

data/tuned_params/
├── lightgbm.json                   [NEW: keyed by (position, stat); checked in]
└── optuna_studies.db               [NEW: SQLite study storage; checked in OR gitignored — see §5.5]
```

`POSITION_DISPATCH.factories` extension:

```python
POSITION_DISPATCH: Mapping[Position, _PositionDispatch] = {
    Position.QB: _PositionDispatch(
        factories={
            "baseline": qb_baseline,
            "lightgbm": qb_lightgbm,
            "lightgbm-tuned": qb_lightgbm_tuned,
        },
        ...
    ),
    ...
}
```

CLI scripts and the backtest harness already consume `factories[model_arg]()`; only the arg-parsing surface changes (the dict lookup is unchanged).

---

## 3. Components

### 3.1 `LightGBMTunedModel` (`src/projections/models/lightgbm_tuned.py`)

Subclass of `LightGBMModel`. Reuses `fit`, `predict_distribution`, `save`, and `load` unchanged. Two responsibilities only:

- Override the hyperparameter source. The base class pulls per-stat hyperparams from the `LGBM_DEFAULTS` constant. The tuned subclass loads `data/tuned_params/lightgbm.json`, looks up `params[position][stat]` per sub-model fit, and merges into `LGBM_DEFAULTS` (tuned values overwrite defaults; untuned axes — `verbose`, `random_state`, `subsample_freq`, `n_estimators` — fall through). The tuned-params JSON is loaded once at instance construction and stored on `self`; not re-read during the per-stat loop.
- Override `model_id` to use the `lightgbm-tuned:` prefix and include the tuned-params file's content hash so changing the JSON invalidates artifacts. Format: `lightgbm-tuned:<pos>:<8-char-code-hash>:<train-start>-<train-end>`.

The base class is refactored to support this cleanly: `LightGBMModel.fit` is updated to consult an instance-level `_hyperparams_for(stat: Stat) -> dict[str, Any]` method that defaults to returning `LGBM_DEFAULTS`. The tuned subclass overrides `_hyperparams_for` to return the merged params. No other code in `LightGBMModel` changes.

**`code_hash_files`** (the source files hashed into `model_id` for invalidation):

- `models/lightgbm_tuned.py` (new)
- `models/lightgbm.py` (parent class)
- `models/base.py`
- `distributions/quantile.py`
- `distributions/codec.py`
- `distributions/parametric.py`
- `features/<pos>.py`
- `features/_shared.py`, `features/_rolling.py`, `features/_opponent.py`
- `scoring/score.py`, `scoring/score_distribution.py`
- `data/tuned_params/lightgbm.json` (new — its hash is the dominant invalidator)

13 files. The JSON file is included so any tuned-params change auto-invalidates artifacts and snapshot rows.

**Disk artifacts:** `models/artifacts/lightgbm-tuned-<pos>-<train-start>-<train-end>-<code-hash>.joblib`.

**Per-position factories** mirror the untuned shape:

```python
def qb_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(config=_LightGBMConfig(...same as qb_lightgbm...))
```

Each factory passes the same `_LightGBMConfig` as its untuned counterpart — only the class differs.

### 3.2 Tuned-params JSON (`data/tuned_params/lightgbm.json`)

Schema:

```json
{
  "qb": {
    "passing_yards":   {"learning_rate": 0.073, "num_leaves": 17, "max_depth": 5, "min_child_samples": 30, "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.012, "reg_lambda": 0.91},
    "passing_tds":     {...},
    "interceptions":   {...},
    "rushing_yards":   {...},
    "rushing_tds":     {...},
    "fumbles_lost":    {...}
  },
  "rb": {...},
  "te": {...},
  "wr": {...}
}
```

Keys are lowercase position values (`Position.QB.value.lower()`) and lowercase stat values (`Stat.PASSING_YARDS.value`, which is already lowercase). One entry per (position, stat) — 24 entries total. Keys for axes that aren't tuned (`n_estimators`, `verbose`, `random_state`, `subsample_freq`) are absent; absent axes inherit from `LGBM_DEFAULTS`.

The file is **checked into git**. It's the source of truth for tuned hyperparameters; reproducible builds need it. Initial Phase 0 scaffold seeds every entry with copies of `LGBM_DEFAULTS`'s tuned-axes values so Phase 0's "model class plumbed but numerically identical to untuned C" smoke test is bit-exact. Phase 2 rewrites the file with the actual Optuna outputs.

A small loader helper in `lightgbm_tuned.py`:

```python
@cache
def _load_tuned_params(path: Path) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    with path.open() as f:
        return json.load(f)
```

`@cache` is fine because the JSON content is small and changes only between deliberate dev iterations; tests that need fresh content can clear the cache or write a new path.

### 3.3 `scripts/tune_lightgbm.py` (Optuna driver)

CLI:

```
usage: tune_lightgbm.py [-h]
                       [--position {qb,rb,te,wr,all}]
                       [--stat STAT | --all-stats]
                       [--trials N]
                       [--out PATH]
                       [--studies-db PATH]
                       [--seed N]
                       [--data-root PATH]
                       [--features-root PATH]
                       [--dry-run]
```

**Defaults:**
- `--position all`
- `--all-stats`
- `--trials 50`
- `--out data/tuned_params/lightgbm.json`
- `--studies-db data/tuned_params/optuna_studies.db`
- `--seed 42`
- `--data-root data` / `--features-root data/features`

**Behavior:**
1. Load weekly_stats + features for seasons 2018-2023 (matches Plan 3c's most-recent-fold training window).
2. Inner-join, validate against the position's feature schema. Same join logic as `LightGBMModel.fit`.
3. For each (position, stat) selected by the args:
   a. Open / resume an Optuna study at `data/tuned_params/optuna_studies.db` named `lightgbm:<pos>:<stat>:v1`.
   b. Sampler: `TPESampler(seed=42)`. Pruner: `MedianPruner(n_startup_trials=10, n_warmup_steps=20)`.
   c. Run `--trials` trials. Each trial:
      - Sample one set of hyperparameters from the search space (§3.4).
      - Carve season slices: train = 2018-2022, early-stop val = 2022, trial scorer = 2023.
      - For each `q in (0.05, 0.10, 0.50, 0.90, 0.95)`: fit `lgb.LGBMRegressor(objective="quantile", alpha=q, **trial_params, **fixed_params)` with early stopping on the 2022 val slice; predict on the 2023 trial scorer slice; compute pinball loss.
      - Trial value = sum of 5 pinball losses on the 2023 slice.
      - The pruner is wired via `LightGBMPruningCallback(trial, "quantile", valid_name="val_2022")` on each per-quantile sub-fit. If a sub-fit's intermediate quantile loss is materially worse than the running median at the same iteration count, the trial is pruned. Optuna's pruner reports a `TrialPruned` exception that the search loop catches and continues.
   d. After all trials complete: extract `study.best_params`. Write into `tuned[pos][stat] = best_params`.
4. After all (position, stat) pairs: write `tuned` to `--out`.
5. `--dry-run` runs all studies but does not write the JSON; useful for smoke tests and dry-run inspection. Without `--dry-run` the JSON at `--out` is overwritten with the final tuned params after all 24 studies complete.

**Validation strategy detail:** The trial uses 2018-2022 training data with 2022 carved as the early-stop val (the same split `LightGBMModel.fit` uses internally when training-end is 2022). The trial-scoring 2023 data is *not* seen by early stopping or by training; it is the held-out trial signal. This makes pinball loss on 2023 a clean per-trial metric. Once the best params are chosen, **the deployed `LightGBMTunedModel.fit` uses 2018-2023 (with 2023 as early-stop val) for backtest fold 2024**, as the harness does — i.e., the deployed model trains on more data than the tuning trials saw, but with the tuned params held fixed.

### 3.4 Search space

Sampled per Optuna trial by `_sample_params(trial)`:

```python
def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 7, 127),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
```

Fixed (not sampled): `n_estimators=4000` (raised from 2000 to give early stopping more room), `subsample_freq=1`, `verbose=-1`, `random_state=42`, `objective="quantile"`, `alpha=q` per sub-model.

`num_leaves` ranges up to 127 but the model implicitly bounds via `2 ** max_depth - 1`; Optuna may produce combinations where `num_leaves > 2 ** max_depth - 1`, in which case LightGBM uses the smaller of the two. Acceptable; not worth a constraint.

### 3.5 Backtest harness wiring

`scripts/backtest.py` and `src/projections/backtest/harness.py` already iterate over `selected_models` parsed from `--model`. Plan 5b extends:

- `--model lightgbm-tuned`: runs only Model C-tuned.
- `--model all`: runs `baseline`, `lightgbm`, `lightgbm-tuned`.
- `--model both` (legacy from Plan 5): preserved as alias for `baseline + lightgbm` (not `+ lightgbm-tuned`); deprecation handled in a follow-up.

The arg-parsing helper `parse_model_arg` is updated to handle the new keys. The fold loop is unchanged — it already factors over a list of model class names.

### 3.6 Snapshot extension

`tests/backtest/model_metrics.json` row keying is unchanged — `(position, year, metric, model_class)`. Plan 5b adds 368 rows under `model_class="lightgbm-tuned"`:

- 16 cells × 23 weekly metrics = 368 rows. Same metric set Model C produces. Skips 32 `season_calibration_*` rows per the existing SAMPLED_SUMMARY family gate (TODO #28).

Pre-Plan-5b row count: 768 (Model A 400 + Model C 368). Post-Plan-5b: 1136. The tolerance classifier in `backtest/snapshot.py` already operates per-row regardless of `model_class`, so no logic changes.

The first snapshot regeneration after Plan 5b lands establishes the Model C-tuned baseline rows; subsequent runs regress against them with the same tolerance rules used today. Phase 0's seeded JSON (copies of `LGBM_DEFAULTS`) produces tuned rows numerically identical to untuned rows; Phase 2's actual tuning produces the new baseline.

---

## 4. Data flow

### 4.1 Tuning (one (position, stat))

```
features parquet                weekly_stats parquet
        │                                │
        └────────────┬───────────────────┘
                     ▼
       inner-join on (gsis_id, season, week); position filter
                     ▼
       split:
         train = season ∈ {2018, 2019, 2020, 2021, 2022}
         early-stop val = season == 2022
         trial scorer = season == 2023
                     ▼
       Optuna study (TPE + median pruner, 50 trials)
                     ▼
       Per trial:
         params = _sample_params(trial)
         For q in QUANTILE_GRID:
           ├─ lgb.LGBMRegressor(objective="quantile", alpha=q,
           │                    n_estimators=4000, **params, ...)
           ├─ .fit(X_train, y_train_stat,
           │       eval_set=[(X_val_2022, y_val_2022)],
           │       callbacks=[lgb.early_stopping(50), LightGBMPruningCallback(...)])
           └─ pinball_loss_q = pinball(y_2023, predict(X_2023, alpha=q))
         trial.report_value = sum(pinball_loss_q for q in QUANTILE_GRID)
                     ▼
       study.best_params → tuned[pos][stat]
                     ▼
       (after all 24 (position, stat) studies)
       json.dump(tuned, "data/tuned_params/lightgbm.json")
```

Wall-time estimate: 24 studies × 50 trials × 5 sub-fits per trial ≈ 6,000 nominal sub-fits. With median pruning aborting ~30-50% of bad trials at intermediate iterations, ~3,000-4,000 effective sub-fits at ~1-3s each → 1.5-3 hours total on a typical dev box.

### 4.2 Training and prediction at backtest time (one position, one fold)

```
1. POSITION_DISPATCH[position].factories["lightgbm-tuned"]() → LightGBMTunedModel
2. instance loads data/tuned_params/lightgbm.json once
3. For fold (held-out year Y):
   a. instance.fit(features, weekly_stats) on seasons 2018..(Y-1):
      - Same join + split logic as LightGBMModel.fit.
      - For each (Stat, q): merge LGBM_DEFAULTS with tuned[pos][stat]
        (tuned axes overwrite defaults); fit LGBMRegressor with merged params;
        early-stop on the last training season.
   b. instance.predict_distribution(features_Y, ruleset)
      - Identical to LightGBMModel: 5-quantile predict per stat,
        sort, clip, wrap in QuantileDistribution, run through score_distribution.
   c. metrics = score_metrics(predictions, weekly_stats_Y)
   d. results.append rows with model_class="lightgbm-tuned"
```

The prediction path beneath `score_distribution` is unchanged. `LightGBMTunedModel` shares `LightGBMModel`'s `predict_distribution` directly; only `_hyperparams_for(stat)` and `model_id` differ.

### 4.3 Snapshot regeneration

`scripts/backtest.py --update-snapshot --model all` runs all three model classes through 4 held-out years × 4 positions = 16 cells per model. Total 1136 rows written to `tests/backtest/model_metrics.json`. The `--check` mode regresses the next run against this baseline.

---

## 5. Error handling

### 5.1 Tuning script failure modes

- **Missing tuned-params JSON on resume from --studies-db.** The script reads existing studies but writes a fresh JSON (no merge from prior state); `--out` is overwritten unless `--dry-run` is set.
- **No trials completed for a (position, stat) study** (e.g., all trials pruned at the warmup phase). Raise `RuntimeError` with the (position, stat) and pruning stats; do not write a partial JSON.
- **Optuna `TrialPruned` exception.** Caught inside the per-trial loop; logged at debug level; the trial is recorded by Optuna as PRUNED and the study continues.
- **LightGBM convergence failure / `best_iteration_ == 0`** inside a trial. The trial completes with `pinball_loss = mean(y_val)` (constant baseline) — high but finite — so TPE down-weights that region. No abort.
- **Ctrl-C during search.** The SQLite study persistence preserves all completed trials; rerunning resumes from where it stopped. The JSON is not written until all studies complete (or `--out` was already chosen at the per-study level — clarified in implementation).

### 5.2 `LightGBMTunedModel` failure modes

- **Tuned-params JSON missing or unreadable.** `FileNotFoundError` / `json.JSONDecodeError` propagates with the path in the message. The factory does not silently fall back to `LGBM_DEFAULTS` — that would mask a misconfigured environment.
- **Tuned params missing for a (position, stat).** `KeyError` with both keys in the message. Indicates the JSON is incomplete for the configured target stats.
- **Tuned param key not in LightGBM's accepted kwargs** (e.g., a renamed axis after a LightGBM upgrade). Caught at fit time by LightGBM itself; raise as-is.

All other failure modes (empty join, insufficient training years, feature column mismatch) are inherited from `LightGBMModel` unchanged.

### 5.3 Snapshot drift handling

The 368 new Model C-tuned rows are *new* on first regeneration; no prior baseline to drift against. The tolerance classifier already handles new rows. After the first `--update-snapshot --model all`, subsequent `--check` runs gate against the established Model C-tuned values.

Phase 0's seeded JSON (default values) produces Model C-tuned numbers bit-identical to Model C numbers. The Phase 0 snapshot commit can therefore be reviewed by diffing — the only changes should be `model_class="lightgbm-tuned"` rows mirroring `lightgbm` rows. Phase 2's actual tuning produces a different snapshot; that diff is the substantive change.

### 5.4 Determinism

- Optuna sampler is seeded (`TPESampler(seed=42)`); pruner is deterministic given trial reports.
- LightGBM sub-fits use `random_state=42` (inherited from `LGBM_DEFAULTS`).
- The combination is reproducible: rerunning `tune_lightgbm.py --seed 42` against the same training data and same study DB resumes from the existing trials and is deterministic onward. A fresh study with the same seed produces identical trials.

The tuning script is therefore safe to rerun for verification. The snapshot regeneration after tuning is deterministic from the tuned JSON.

### 5.5 Optuna study DB and version control

The SQLite study DB at `data/tuned_params/optuna_studies.db` is the artifact-of-search; the tuned JSON is the artifact-of-decision. The DB is large (tens to hundreds of MB after 1200 trials) and not directly load-bearing for downstream consumers.

**Decision:** add the DB to `.gitignore`. The tuned JSON is checked in and is the only on-disk source of truth for tuned hyperparameters. Reproducing a search requires rerunning `tune_lightgbm.py` (deterministic given `--seed`); the DB is incidental scratch space. If the user wants to inspect a specific trial post-hoc, the DB on their machine answers it; if it's lost, the JSON still answers "what were the chosen params."

### 5.6 LightGBM warning suppression

Optuna's `LightGBMPruningCallback` interacts with LightGBM's logging. The callback uses LightGBM's iter-level `evals_result_` to report intermediate metrics; this requires `valid_sets` to be passed to `lgb.train` in a particular shape. If using `lgb.LGBMRegressor.fit(eval_set=...)` (the sklearn wrapper, which `LightGBMModel.fit` currently uses), the callback's metric-name and valid-name strings must match. Implementation will validate this on the synthetic-fixture smoke test before running the full search.

---

## 6. Testing

### 6.1 `LightGBMTunedModel` (`tests/test_models/test_lightgbm_tuned*.py`)

Mirror the Plan 5 `tests/test_models/test_lightgbm*.py` shape, adapted to the tuned variant:

- **`test_lightgbm_tuned.py`** (cross-cutting):
  - `fit` produces 5 sub-models per stat per position; `best_iters` populated; `model_id` present and stable.
  - `model_id` is prefixed `lightgbm-tuned:` (not `lightgbm:`); tuned subclass is distinguishable in artifacts.
  - With Phase 0's seeded JSON (defaults), `predict_distribution` output is bit-exact to the untuned `LightGBMModel` on the same fixture (proves the override is plumbed without changing arithmetic).
  - With a synthetic non-default tuned JSON (override `learning_rate` and `num_leaves` for one stat only), the corresponding sub-model boosters differ from the untuned baseline but predictions still validate against `ProjectionWeeklySchema`.
  - Save / load round-trip preserves `model_id` and predictions.
  - `code_hash` includes the JSON path's content; mutating the JSON between two `LightGBMTunedModel()` instantiations changes `code_hash`.
  - Missing tuned-params for (position, stat) raises `KeyError` at fit time; the JSON is *not* permitted to be sparse.

- **Per-position files** (`test_lightgbm_tuned_qb.py`, `_rb.py`, `_te.py`, `_wr.py`):
  - Smoke fit on synthetic per-position fixtures (reuse Plan 3a / 3b fixtures); assert per-stat sub-models exist; assert prediction output non-empty and validates.

- **Smoke parametrized across positions** (`tests/test_models/test_lightgbm_tuned_smoke.py`): single fit + predict for each of 4 positions, ~30s total.

### 6.2 Backtest harness (`tests/test_backtest/`)

- Existing tests pass unchanged.
- Extend `test_harness_dual_model.py` (or add `test_harness_triple_model.py`) for `--model all`: results.parquet has `model_class` column with all three values present; per-cell metrics for all three.
- `parse_model_arg` unit test: assert `--model all` parses to `["baseline", "lightgbm", "lightgbm-tuned"]` and `--model both` still parses to `["baseline", "lightgbm"]`.

### 6.3 Default-on smoke (`tests/backtest/test_backtest_smoke.py`)

Extend the existing (WR, 2024) smoke to assert all three model classes produce finite metrics. Total runtime budget rises to ~45-60s — Model C-tuned fit on one position is well under 30s on the synthetic fixture.

### 6.4 Optuna driver (`tests/test_scripts/test_tune_lightgbm.py`)

- **End-to-end tiny run.** With the synthetic per-position fixture, run `tune_lightgbm.py --position wr --stat receiving_yards --trials 3 --seed 42 --dry-run`; assert the script completes without error and reports a study `best_params` dict containing all 8 tuned axes.
- **Determinism.** Two `--trials 3 --seed 42 --dry-run` runs against the same fixture and a fresh in-memory study produce identical `best_params`.
- **Pruner integration.** With `--trials 5` on the fixture and the median pruner enabled, assert at least 0 trials are PRUNED (smoke; the assertion is loose because the synthetic fixture is too small for pruning to consistently engage). Real verification happens during the production run, not in CI.
- **Resume.** Run `--trials 2 --seed 42` with a SQLite DB path; rerun `--trials 4 --seed 42` against the same DB; assert the study has 4 (not 6) total trials and `best_params` reflects the union.
- **Search-space coverage.** `_sample_params(MockTrial)` is exercised once with a recording trial that captures every `suggest_*` call; assert all 8 axes are sampled with the expected ranges + log-flags.

The test file uses the in-memory Optuna storage (`storage=None`) for the synthetic-fixture path so tests don't write to disk — the SQLite-resume test is the only one that touches a real DB (in a tmp_path).

### 6.5 Backtest snapshot gate (`pytest -m backtest --run-backtest`)

Full opt-in backtest produces 1136 metric rows (400 Model A + 368 Model C + 368 Model C-tuned). Wall time ≈ Model A 292s + Model C 30-50min + Model C-tuned ~30-50min → 65-110 minutes for `--model all`. For ad-hoc verification, `--model lightgbm-tuned` alone runs only the new model class.

### 6.6 Type / lint conformance

- `mypy src tests` clean — `LightGBMTunedModel` and Optuna imports typed (Optuna ships `py.typed`).
- `ruff check src tests` and `ruff format --check src tests` clean.
- `optuna` added to `pyproject.toml` dependencies as `optuna>=3.0` (matches the `>=` convention used by every other dep).

### 6.7 Tuned-params JSON schema validation

A small loader-plus-validator helper in `lightgbm_tuned.py`:

- Top-level keys must be exactly `{"qb", "rb", "te", "wr"}`.
- For each position, second-level keys must be exactly that position's `target_stats` (same as the untuned model).
- For each (position, stat), third-level keys must be a subset of the 8 tuned axes.
- All values must be the appropriate type (float or int).

The validator runs at instance construction time; a malformed JSON fails fast with a path-and-key-specific error message rather than a downstream LightGBM kwarg surprise.

---

## 7. Phasing

Each phase ≤5 files per CLAUDE.md "PHASED EXECUTION" rule.

### Phase 0 — Tuned-model scaffold (no actual tuning yet)

Files (5):
1. `src/projections/models/lightgbm.py` — add `_hyperparams_for(stat: Stat) -> dict[str, Any]` hook returning `LGBM_DEFAULTS`; replace the `**LGBM_DEFAULTS` literal in `fit` with `**self._hyperparams_for(stat)`. Behavioral no-op.
2. `src/projections/models/lightgbm_tuned.py` (new) — `LightGBMTunedModel` subclass + 4 per-position factories.
3. `src/projections/models/__init__.py` — extend `POSITION_DISPATCH.factories` with `"lightgbm-tuned"` per position; export tuned factories.
4. `data/tuned_params/lightgbm.json` (new) — seeded with copies of the 8 tuned-axis values from `LGBM_DEFAULTS`, dense across all 24 (position, stat) entries.
5. `tests/test_models/test_lightgbm_tuned.py` (new) — cross-cutting tests including the bit-exact-with-seeded-JSON assertion against `LightGBMModel` on a shared synthetic fixture.

The per-position smokes (`test_lightgbm_tuned_{qb,rb,te,wr}.py`) and the parametrized `test_lightgbm_tuned_smoke.py` move to Phase 2's testing pass — Phase 0's exit criterion only requires that the cross-cutting test demonstrates the override is plumbed.

**Exit criterion:** `pytest -v` clean; `mypy src tests` and `ruff check src tests` clean. With the seeded JSON, `LightGBMTunedModel` predictions on the synthetic fixture are bit-exact to `LightGBMModel` predictions. No backtest regeneration in this phase.

### Phase 1 — Optuna search infrastructure

Files: `scripts/tune_lightgbm.py` (new), `tests/test_scripts/test_tune_lightgbm.py` (new), `pyproject.toml` (add `optuna>=3.0`), `.gitignore` (add `data/tuned_params/optuna_studies.db`), `data/tuned_params/.gitkeep` (new — keeps the directory tracked even though the DB is gitignored). 5 files.

**Exit criterion:** `pytest -v -k "tune_lightgbm"` passes; Optuna driver runs end-to-end on synthetic fixture with `--trials 3`; `mypy src tests scripts` clean; `ruff check src tests scripts` clean.

### Phase 2 — Backtest harness wiring + per-position tuned smokes

Files (5):
1. `src/projections/backtest/harness.py` — extend `parse_model_arg` to accept `lightgbm-tuned` and `all`; preserve `both` as `baseline + lightgbm` alias.
2. `scripts/backtest.py` — surface the new `--model` values in argparse `choices`; update help text.
3. `tests/backtest/test_backtest_smoke.py` — extend default-on smoke to assert all three model classes produce finite metrics for (WR, 2024).
4. `tests/test_backtest/test_harness_triple_model.py` (new) — `--model all` end-to-end on synthetic fixture; assert results.parquet has all three `model_class` values.
5. `tests/test_models/test_lightgbm_tuned_smoke.py` (new) — parametrized cross-position smoke for the tuned model class (mirrors `test_lightgbm_smoke.py`).

The standalone CLI scripts (`scripts/train_baseline.py`, `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py`) are *not* touched in Plan 5b. They already consume `POSITION_DISPATCH[position].factories[model_arg]()`; the `factories` dict edit in Phase 0 makes the new key available, but the scripts' `argparse` `choices=` lists still gate which strings pass validation. Updating those is mechanical and lands in a small follow-up commit if any script run actually needs the tuned model — Plan 5b's evaluation runs through the backtest harness only.

**Exit criterion:** harness accepts `--model lightgbm-tuned` and `--model all`; default-on smoke covers all three model classes; `pytest -v -k "harness or smoke"` clean.

### Phase 3 — Run the search and persist tuned params

Files: `data/tuned_params/lightgbm.json` (overwritten by Optuna). Possibly `tests/backtest/model_metrics.json` if we regenerate in this phase. 1-2 files.

**Operations** (not files):
1. Run `python scripts/tune_lightgbm.py --position all --all-stats --trials 50 --seed 42` end-to-end.
2. Inspect the resulting `lightgbm.json` (sanity-check ranges, no NaN, all 24 entries present).
3. Run `python scripts/backtest.py --update-snapshot --model all` to regenerate `model_metrics.json` with all three model classes (~65-110min).
4. Commit both the JSON and the snapshot.

**Exit criterion:** `lightgbm.json` populated for all 24 (position, stat) entries; `model_metrics.json` has 1136 rows; backtest gate (`pytest -m backtest --run-backtest`) passes against the new snapshot.

### Phase 4 — Diagnostic report

Files: `project_management.md` (new section), `TODO.md` (close TODO #26 follow-up notes; possibly file Plan 5c TODO if gate flips). 2 files.

**Operations:**
1. Generate the per-cell A vs C vs C-tuned comparison table (same shape as Plan 5's §1.3 table; 16 rows × 4 metrics × 3 models).
2. Compute the §1.3 adoption-gate verdict for C-tuned vs A. Pass / fail per criterion.
3. Append "Plan 5b — Optuna Tuning of Model C — shipped (run YYYY-MM-DD)" section to `project_management.md` with the table and verdict.
4. If gate passes: file a TODO for Plan 5c (per-fold tuning + production-default switch). If gate fails: document the deltas and surface the next-step decision.

**Exit criterion:** Phase 4 commit is the PR's last commit; PR description includes the gate verdict.

---

## 8. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-27 | New model class `LightGBMTunedModel` (not in-place `LightGBMModel` update) | Plan 5's side-by-side infrastructure was built for exactly this. Preserves Model C through any successor; lets the snapshot itself answer the diagnostic question without needing prior numbers. |
| 2026-04-27 | Tune at per-(position, stat) granularity (24 studies, not 4 or 120) | Per-position is too coarse — different stats have different distributions. Per-quantile is too fine — 5 quantile sub-models for the same (position, stat) share data and only differ in `alpha`; tuning per quantile is mostly noise. Per-(position, stat) captures the real heterogeneity. |
| 2026-04-27 | Trial objective = sum of 5 pinball losses on 2023 val | Pinball loss is what each sub-model is trained on; optimizing it directly addresses the under-confidence pattern that produced both the RMSE and calibration regression vs. Model A. Adoption-gate composite metrics fall out at backtest-evaluation time, but they're not the search target. |
| 2026-04-27 | Tune once on most-recent fold (2018-2022 train, 2022 early-stop val, 2023 trial scorer); reuse tuned params for all 4 backtest folds | Diagnostic-mode budget. Per-fold tuning is 4× the cost; defer to Plan 5c if the diagnostic warrants production adoption. The mild leakage (hyperparams chosen with 2022/2023 visible when scoring 2021/2022 backtest folds) is small in practice — hyperparameters are coarser than model weights. |
| 2026-04-27 | TPE sampler + median pruner; 50 trials/study | TPE needs ~10-20 random startups before it steers; 50 trials gives ~30 informed trials per study. Pruner halves wall time without affecting exploration on the relevant region of the search space. |
| 2026-04-27 | Search 8 axes; fix `n_estimators=4000` and rely on early stopping | `n_estimators` is best treated as a budget cap with early stopping picking the actual count, not a search axis. Raising the cap from 2000 to 4000 gives early stopping more headroom. The 8 sampled axes (`learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`) are the standard LightGBM tuning surface. |
| 2026-04-27 | Tuned params persisted as `data/tuned_params/lightgbm.json` (checked in) | The JSON is the artifact-of-decision; it is small and load-bearing for reproducible model fits. Optuna's SQLite study DB is artifact-of-search and gitignored. |
| 2026-04-27 | `_hyperparams_for(stat)` hook on `LightGBMModel`; tuned subclass overrides | Minimal-surface refactor of the existing model. The base class default returns `LGBM_DEFAULTS` (no behavioral change for Model C); the subclass overrides to return merged tuned params. Avoids parallel copy-pasted training loops. |
| 2026-04-27 | `code_hash_files` for the tuned model includes the JSON path | Changing tuned params should invalidate artifacts and force a snapshot regeneration; including the JSON in the hash makes this automatic. |
| 2026-04-27 | Tuned-params JSON must be dense (no missing (position, stat) entries) | Sparse JSON would silently fall back to defaults for missing entries — masking misconfiguration. Fail fast with a clear error message instead. |
| 2026-04-27 | Plan 5b ships diagnostic + persistent comparison; production adoption is Plan 5c | Keeps each plan's scope sized to a reviewable PR; lets the per-fold tuning cost be avoided if the diagnostic verdict is "tuning isn't the lever." |

---

## 9. Out-of-scope / explicit future work

- **Per-fold tuning** (re-tune for each backtest year). Plan 5c if Plan 5b's diagnostic flips the gate.
- **Production-default switch** from Model A to Model C-tuned. Plan 5c.
- **TODO #28** — widening `aggregate_to_season` to accept the QUANTILE family. Independent of tuning; would benefit both Model C and Model C-tuned. Pursue separately when convenient.
- **Multi-output / shared-prior LightGBM training** (Plan 5 post-mortem mechanisms #1 and #4). Would require structural changes to the per-stat sub-model loop. Plan 6 ensemble territory or its own plan.
- **Quantile-grid widening** (Plan 5 post-mortem mechanism #3). 5 → 9 or 11 quantiles. Independent change; not bundled here so any tuning win is attributable.
- **Calibration-aware fitting / quantile-loss-with-calibration-penalty objectives.** Considered and rejected for Plan 5b — risks distorting the upper tail (load-bearing for DFS GPP). Pinball loss is the natural per-sub-model search target; calibration is a downstream consequence reported at evaluation time.
- **Optuna parameter importance reports** in `project_management.md`. Optuna provides `importance_get_param_importances` cheaply post-search; useful for understanding which axes matter and could inform a narrower future search. Land in Phase 4's diagnostic report if straightforward; otherwise punt.
