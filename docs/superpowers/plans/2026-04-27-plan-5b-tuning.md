# Plan 5b — Optuna Tuning of Model C — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `LightGBMTunedModel` (Model C-tuned) as a third model class coexisting with Models A and C, driven by 24 per-(position, stat) Optuna studies; regenerate the backtest snapshot with all three models side-by-side and report the per-cell A vs C vs C-tuned diagnostic.

**Architecture:** Subclass of `LightGBMModel` overriding only the per-stat hyperparameter source via a new `_hyperparams_for(stat)` hook. Tuned hyperparameters live in `data/tuned_params/lightgbm.json` (checked in, dense across all 24 (position, stat) entries, content-hashed into `model_id`). Tuning script `scripts/tune_lightgbm.py` runs once on the most-recent fold (2018-2022 train / 2022 early-stop val / 2023 trial scorer); resulting hyperparams are reused across all 4 backtest folds. POSITION_DISPATCH gains a `lightgbm-tuned` factory key per position; harness `--model` selector accepts `lightgbm-tuned` and `all`.

**Tech Stack:** Python 3.12, LightGBM ≥4.0, Optuna ≥3.0 (TPE sampler + median pruner via `optuna.integration.LightGBMPruningCallback`), pandera, joblib, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-04-27-plan-5b-tuning-design.md`.

**Branch:** `feat/plan-5b-tuning`.

---

## Phase 0 — Tuned-model scaffold

Goal: model class plumbed end-to-end with bit-exact behavior vs untuned `LightGBMModel` on a synthetic fixture (the seeded JSON contains `LGBM_DEFAULTS` values verbatim). No tuning yet.

### Task 1: Add `_hyperparams_for(stat)` hook on `LightGBMModel`

**Files:**
- Modify: `src/projections/models/lightgbm.py`

The `fit` loop currently hardcodes `**LGBM_DEFAULTS` when constructing `lgb.LGBMRegressor`. Replace with `**self._hyperparams_for(stat)` to give subclasses a one-method extension point. The base method returns a copy of `LGBM_DEFAULTS` so behavior is unchanged.

- [ ] **Step 1: Add the method to `LightGBMModel`**

Open `src/projections/models/lightgbm.py`. Find the `class LightGBMModel:` definition and insert this method after the `model_id` property (immediately before `def fit(...):`):

```python
    def _hyperparams_for(self, stat: Stat) -> dict[str, Any]:
        """Return LightGBM kwargs for the given stat's sub-models.

        Subclasses override to provide tuned per-(position, stat) hyperparameters.
        The base implementation returns a copy of LGBM_DEFAULTS so all sub-models
        share the same baseline settings.
        """
        return dict(LGBM_DEFAULTS)
```

- [ ] **Step 2: Use the hook in `fit`**

In the same file, find the `fit` method's per-quantile loop (around lines 332-337):

```python
            for q in QUANTILE_GRID:
                regressor = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    **LGBM_DEFAULTS,
                )
```

Replace with:

```python
            stat_params = self._hyperparams_for(stat)
            for q in QUANTILE_GRID:
                regressor = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    **stat_params,
                )
```

`stat_params` is hoisted out of the q-loop since it's identical for all 5 quantiles of the same stat — saves a method call per quantile and matches how callers will reason about per-(position, stat) tuning.

- [ ] **Step 3: Run existing LightGBM tests to confirm no regressions**

```bash
. .venv/Scripts/activate
pytest -v tests/test_models/test_lightgbm.py tests/test_models/test_lightgbm_qb.py tests/test_models/test_lightgbm_rb.py tests/test_models/test_lightgbm_te.py tests/test_models/test_lightgbm_smoke.py
```

Expected: all tests PASS. The hook is a no-op; predictions, model_id, save/load, and code_hash are bit-exact identical.

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm.py
git commit -m "$(cat <<'EOF'
refactor(models): _hyperparams_for(stat) hook on LightGBMModel — Plan 5b prep

Replaces the hardcoded **LGBM_DEFAULTS in fit()'s per-quantile loop with a
per-stat lookup. Base class returns a copy of LGBM_DEFAULTS — behavioral
no-op. Subclasses override the hook to inject tuned hyperparameters.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create the seeded tuned-params JSON

**Files:**
- Create: `data/tuned_params/lightgbm.json`

Phase 0's exit criterion is bit-exact predictions vs untuned `LightGBMModel`. To achieve that, the seeded JSON must contain `LGBM_DEFAULTS` values verbatim across all 24 (position, stat) entries.

Per-position target stats (from `src/projections/models/lightgbm.py`):
- QB: `passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds, fumbles_lost`
- RB: `rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fumbles_lost`
- TE: `receptions, receiving_yards, receiving_tds, rushing_yards, rushing_tds, fumbles_lost`
- WR: `receptions, receiving_yards, receiving_tds, rushing_yards, rushing_tds, fumbles_lost`

Per-axis defaults from `LGBM_DEFAULTS`:
- `learning_rate: 0.05, num_leaves: 31, max_depth: 6, min_child_samples: 20, subsample: 0.8, colsample_bytree: 0.8, reg_alpha: 0.0, reg_lambda: 1.0`

- [ ] **Step 1: Write the JSON file**

Create `data/tuned_params/lightgbm.json` with the following content (every (position, stat) entry holds the same 8-key default block):

```json
{
  "qb": {
    "passing_yards":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "passing_tds":     {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "interceptions":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "rushing_yards":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "rushing_tds":     {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "fumbles_lost":    {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0}
  },
  "rb": {
    "rushing_yards":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "rushing_tds":     {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "receptions":      {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "receiving_yards": {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "receiving_tds":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "fumbles_lost":    {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0}
  },
  "te": {
    "receptions":      {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "receiving_yards": {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "receiving_tds":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "rushing_yards":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "rushing_tds":     {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "fumbles_lost":    {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0}
  },
  "wr": {
    "receptions":      {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "receiving_yards": {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "receiving_tds":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "rushing_yards":   {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "rushing_tds":     {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "fumbles_lost":    {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0}
  }
}
```

- [ ] **Step 2: Verify the JSON parses**

```bash
python -c "import json; d = json.load(open('data/tuned_params/lightgbm.json')); print(sorted(d.keys()), [(p, sorted(d[p].keys())) for p in sorted(d.keys())])"
```

Expected output: 4 positions, 6 stats each, exactly matching the per-position target_stats lists.

- [ ] **Step 3: Commit**

```bash
git add data/tuned_params/lightgbm.json
git commit -m "$(cat <<'EOF'
chore(plan-5b): seed tuned-params JSON with LGBM_DEFAULTS — Plan 5b Phase 0

Dense (position, stat) entries with each of the 8 tunable axes set to its
LGBM_DEFAULTS value. Phase 2's tune_lightgbm.py overwrites this file with
Optuna's best_params.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Implement `LightGBMTunedModel` and wire factories

**Files:**
- Create: `src/projections/models/lightgbm_tuned.py`
- Modify: `src/projections/models/__init__.py`

`LightGBMTunedModel` subclasses `LightGBMModel`, overrides `_hyperparams_for`, `code_hash`, and `model_id`, and exposes 4 per-position factories. The `models/__init__.py` registry gains `"lightgbm-tuned"` keys.

- [ ] **Step 1: Write `src/projections/models/lightgbm_tuned.py`**

```python
"""Tuned LightGBM model — Plan 5b.

Subclass of LightGBMModel. Reuses fit / predict_distribution / save / load
unchanged. Two responsibilities:
  1. Override `_hyperparams_for(stat)` to load tuned hyperparameters from
     data/tuned_params/lightgbm.json and merge into LGBM_DEFAULTS.
  2. Override `code_hash` and `model_id` to use the lightgbm-tuned: prefix
     and include the tuned-params JSON in the hash so JSON edits invalidate
     artifacts and force snapshot regeneration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from projections.models.base import compute_code_hash
from projections.models.lightgbm import (
    LGBM_DEFAULTS,
    LightGBMModel,
    _LightGBMConfig,
    _filter_features,
    _QB_FEATURE_COLUMNS,
    _QB_NON_NEGATIVE,
    _QB_TARGET_STATS,
    _RB_FEATURE_COLUMNS,
    _RB_NON_NEGATIVE,
    _RB_TARGET_STATS,
    _TE_FEATURE_COLUMNS,
    _TE_NON_NEGATIVE,
    _TE_TARGET_STATS,
    _WR_FEATURE_COLUMNS,
    _WR_NON_NEGATIVE,
    _WR_TARGET_STATS,
)
from projections.schemas import (
    Position,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Stat,
    TeFeaturesSchema,
    WrFeaturesSchema,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_TUNED_PARAMS_PATH: Final[Path] = (
    _PROJECT_ROOT / "data" / "tuned_params" / "lightgbm.json"
)

# Subset of LGBM_DEFAULTS keys that Optuna is allowed to tune. Any other key
# in the tuned-params JSON is rejected by the validator on load.
_TUNED_AXES: Final[frozenset[str]] = frozenset(
    {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
)

_EXPECTED_POSITIONS: Final[frozenset[str]] = frozenset({"qb", "rb", "te", "wr"})


@lru_cache(maxsize=4)
def _load_tuned_params(
    path: Path,
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    """Load + validate the tuned-params JSON. Cached by path.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        json.JSONDecodeError: ``path`` is not valid JSON.
        ValueError: top-level position keys are not exactly {qb,rb,te,wr}, or
            any (position, stat) entry contains an unknown axis key.
    """
    with path.open() as f:
        raw: dict[str, dict[str, dict[str, float]]] = json.load(f)
    if set(raw.keys()) != _EXPECTED_POSITIONS:
        raise ValueError(
            f"tuned-params JSON {path}: top-level keys must be "
            f"{sorted(_EXPECTED_POSITIONS)}; got {sorted(raw.keys())}"
        )
    for pos_key, stat_map in raw.items():
        for stat_key, axis_map in stat_map.items():
            extras = set(axis_map.keys()) - _TUNED_AXES
            if extras:
                raise ValueError(
                    f"tuned-params JSON {path}: position={pos_key} "
                    f"stat={stat_key} has unknown tuned-axis keys: "
                    f"{sorted(extras)}; allowed: {sorted(_TUNED_AXES)}"
                )
    return raw


def _code_hash_files_tuned(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into the tuned model's model_id.

    Mirrors `lightgbm.py`'s _code_hash_files but adds the tuned-params JSON
    so that JSON edits invalidate cached artifacts and force snapshot
    regeneration.
    """
    src = _PROJECT_ROOT / "src" / "projections"
    feat_module = {
        Position.QB: "qb.py",
        Position.RB: "rb.py",
        Position.TE: "te.py",
        Position.WR: "wr.py",
    }[position]
    return (
        src / "models" / "lightgbm_tuned.py",
        src / "models" / "lightgbm.py",
        src / "models" / "base.py",
        src / "distributions" / "quantile.py",
        src / "distributions" / "codec.py",
        src / "distributions" / "parametric.py",
        src / "features" / feat_module,
        src / "features" / "_shared.py",
        src / "features" / "_rolling.py",
        src / "features" / "_opponent.py",
        src / "scoring" / "score.py",
        src / "scoring" / "score_distribution.py",
        _TUNED_PARAMS_PATH,
    )


class LightGBMTunedModel(LightGBMModel):
    """LightGBM with Optuna-tuned per-(position, stat) hyperparameters.

    Loads tuned params from ``data/tuned_params/lightgbm.json`` at
    construction. Overrides only ``_hyperparams_for(stat)``, ``code_hash``,
    and ``model_id``; everything else (fit, predict_distribution, save,
    load) inherits unchanged.
    """

    def __init__(
        self,
        *,
        config: _LightGBMConfig,
        tuned_params_path: Path = _TUNED_PARAMS_PATH,
    ) -> None:
        super().__init__(config=config)
        self._tuned_params_path = tuned_params_path
        self._tuned = _load_tuned_params(tuned_params_path)

    def _hyperparams_for(self, stat: Stat) -> dict[str, Any]:
        pos_key = self._config.position.value.lower()
        try:
            tuned_axes = self._tuned[pos_key][stat.value]
        except KeyError as e:
            raise KeyError(
                f"tuned-params JSON {self._tuned_params_path} missing entry "
                f"for position={pos_key} stat={stat.value}"
            ) from e
        merged = dict(LGBM_DEFAULTS)
        merged.update(tuned_axes)
        return merged

    @property
    def code_hash(self) -> str:
        return compute_code_hash(_code_hash_files_tuned(self._config.position))

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on "
                "training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"lightgbm-tuned:{self._config.position.value.lower()}:"
            f"{self.code_hash}:{self._train_start}-{self._train_end}"
        )


def qb_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_filter_features(_QB_FEATURE_COLUMNS),
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )


def rb_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            feature_columns=_filter_features(_RB_FEATURE_COLUMNS),
            feature_schema=RbFeaturesSchema,
            non_negative_stats=_RB_NON_NEGATIVE,
        )
    )


def te_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            feature_columns=_filter_features(_TE_FEATURE_COLUMNS),
            feature_schema=TeFeaturesSchema,
            non_negative_stats=_TE_NON_NEGATIVE,
        )
    )


def wr_lightgbm_tuned() -> LightGBMTunedModel:
    return LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        )
    )
```

- [ ] **Step 2: Wire factories into `POSITION_DISPATCH`**

Open `src/projections/models/__init__.py`. Update three sections:

(a) The import block — add `LightGBMTunedModel` and the four tuned factories:

```python
from projections.models.lightgbm_tuned import (
    LightGBMTunedModel,
    qb_lightgbm_tuned,
    rb_lightgbm_tuned,
    te_lightgbm_tuned,
    wr_lightgbm_tuned,
)
```

(b) `__all__` — add 5 entries (alphabetical to match the existing ordering convention):

```python
__all__ = [
    "POSITION_DISPATCH",
    "BaselineModel",
    "LightGBMModel",
    "LightGBMTunedModel",
    "Model",
    "compute_code_hash",
    "qb_baseline",
    "qb_lightgbm",
    "qb_lightgbm_tuned",
    "rb_baseline",
    "rb_lightgbm",
    "rb_lightgbm_tuned",
    "te_baseline",
    "te_lightgbm",
    "te_lightgbm_tuned",
    "wr_baseline",
    "wr_lightgbm",
    "wr_lightgbm_tuned",
]
```

(c) The four per-position `_XX_FACTORIES` dicts — add the `lightgbm-tuned` key to each:

```python
_QB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": qb_baseline,
    "lightgbm": qb_lightgbm,
    "lightgbm-tuned": qb_lightgbm_tuned,
}
_RB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": rb_baseline,
    "lightgbm": rb_lightgbm,
    "lightgbm-tuned": rb_lightgbm_tuned,
}
_TE_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": te_baseline,
    "lightgbm": te_lightgbm,
    "lightgbm-tuned": te_lightgbm_tuned,
}
_WR_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": wr_baseline,
    "lightgbm": wr_lightgbm,
    "lightgbm-tuned": wr_lightgbm_tuned,
}
```

- [ ] **Step 3: Verify the tuned factories construct without error**

```bash
python -c "from projections.models import qb_lightgbm_tuned, rb_lightgbm_tuned, te_lightgbm_tuned, wr_lightgbm_tuned; print('factories ok:', [f().__class__.__name__ for f in (qb_lightgbm_tuned, rb_lightgbm_tuned, te_lightgbm_tuned, wr_lightgbm_tuned)])"
```

Expected output: `factories ok: ['LightGBMTunedModel', 'LightGBMTunedModel', 'LightGBMTunedModel', 'LightGBMTunedModel']`. Construction touches the JSON loader; success here also confirms the JSON parses + validates.

- [ ] **Step 4: Verify `POSITION_DISPATCH` has the new key**

```bash
python -c "from projections.models import POSITION_DISPATCH; from projections.schemas import Position; [print(p.value, sorted(POSITION_DISPATCH[p].factories.keys())) for p in Position if p in POSITION_DISPATCH]"
```

Expected output: each position prints `['baseline', 'lightgbm', 'lightgbm-tuned']`.

- [ ] **Step 5: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/lightgbm_tuned.py src/projections/models/__init__.py
git commit -m "$(cat <<'EOF'
feat(models): LightGBMTunedModel + POSITION_DISPATCH wiring — Plan 5b Phase 0

Subclass overrides _hyperparams_for(stat), code_hash, model_id; loads tuned
params from data/tuned_params/lightgbm.json with up-front schema validation.
Tuned-params JSON path is included in code_hash so JSON edits invalidate
cached artifacts. POSITION_DISPATCH.factories gains "lightgbm-tuned" per
position.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Cross-cutting tests for `LightGBMTunedModel`

**Files:**
- Create: `tests/test_models/test_lightgbm_tuned.py`

Mirrors `tests/test_models/test_lightgbm.py`'s shape. The headline assertion is **bit-exact predictions** vs untuned `LightGBMModel` when the JSON contains `LGBM_DEFAULTS` values verbatim (Phase 0's exit criterion).

- [ ] **Step 1: Write the test file**

Create `tests/test_models/test_lightgbm_tuned.py`:

```python
"""Cross-cutting tests for LightGBMTunedModel (Model C-tuned, Plan 5b).

Per-position smokes live in test_lightgbm_tuned_smoke.py (Phase 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from projections.models.lightgbm import LGBM_DEFAULTS, wr_lightgbm
from projections.models.lightgbm_tuned import (
    LightGBMTunedModel,
    _load_tuned_params,
    qb_lightgbm_tuned,
    rb_lightgbm_tuned,
    te_lightgbm_tuned,
    wr_lightgbm_tuned,
)
from projections.schemas import Ruleset, Stat


# ---------------- Synthetic fixture (re-using the test_lightgbm.py shape) ----------------


def _build_synthetic_wr_features(
    n_seasons: int = 4, n_weeks: int = 17, n_players: int = 30
) -> pd.DataFrame:
    """Synthetic WrFeaturesSchema-shaped DataFrame for fit/predict smoke tests."""
    from projections.schemas import _PYARROW_STR, WrFeaturesSchema

    rng = np.random.default_rng(42)
    rows = []
    for season in range(2018, 2018 + n_seasons):
        for week in range(1, n_weeks + 1):
            for p in range(n_players):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": season,
                        "week": week,
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["season"] = df["season"].astype(np.int64)
    df["week"] = df["week"].astype(np.int64)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    schema_cols = WrFeaturesSchema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        else:
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    return WrFeaturesSchema.validate(df)


def _build_synthetic_wr_weekly_stats(features: pd.DataFrame) -> pd.DataFrame:
    """WeeklyStatsSchema-shaped truth for the synthetic WR features."""
    from projections.schemas import WeeklyStatsSchema

    rng = np.random.default_rng(43)
    n = len(features)
    df = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    df["position"] = "WR"
    df["receptions"] = rng.integers(0, 8, size=n).astype(np.int64)
    df["targets"] = (df["receptions"] + rng.integers(0, 4, size=n)).astype(np.int64)
    df["receiving_yards"] = rng.integers(0, 100, size=n).astype(np.int64)
    df["receiving_tds"] = rng.integers(0, 2, size=n).astype(np.int64)
    df["receiving_air_yards"] = rng.integers(0, 80, size=n).astype(np.int64)
    df["carries"] = rng.integers(0, 2, size=n).astype(np.int64)
    df["rushing_yards"] = rng.integers(0, 20, size=n).astype(np.int64)
    df["rushing_tds"] = rng.integers(0, 1, size=n).astype(np.int64)
    df["passing_yards"] = np.zeros(n, dtype=np.int64)
    df["passing_tds"] = np.zeros(n, dtype=np.int64)
    df["interceptions"] = np.zeros(n, dtype=np.int64)
    df["attempts"] = np.zeros(n, dtype=np.int64)
    df["completions"] = np.zeros(n, dtype=np.int64)
    df["sacks"] = np.zeros(n, dtype=np.int64)
    df["fumbles_lost"] = rng.integers(0, 1, size=n).astype(np.int64)
    return WeeklyStatsSchema.validate(df)


# ---------------- Tests ----------------


def test_factories_construct() -> None:
    for factory in (qb_lightgbm_tuned, rb_lightgbm_tuned, te_lightgbm_tuned, wr_lightgbm_tuned):
        m = factory()
        assert isinstance(m, LightGBMTunedModel)


def test_seeded_json_matches_lgbm_defaults() -> None:
    """Phase 0's seeded JSON contains LGBM_DEFAULTS values verbatim."""
    from projections.models.lightgbm_tuned import _TUNED_PARAMS_PATH, _TUNED_AXES

    raw = json.loads(_TUNED_PARAMS_PATH.read_text())
    for pos_key in ("qb", "rb", "te", "wr"):
        for stat_key, axis_map in raw[pos_key].items():
            assert set(axis_map.keys()) == _TUNED_AXES, (
                f"{pos_key}/{stat_key}: axes {sorted(axis_map.keys())} "
                f"differ from {sorted(_TUNED_AXES)}"
            )
            for axis_name, value in axis_map.items():
                assert value == LGBM_DEFAULTS[axis_name], (
                    f"{pos_key}/{stat_key}/{axis_name}: seeded JSON value "
                    f"{value} differs from LGBM_DEFAULTS value "
                    f"{LGBM_DEFAULTS[axis_name]}"
                )


def test_predictions_bit_exact_vs_untuned_under_seeded_json() -> None:
    """Phase 0 exit criterion: with seeded JSON, tuned predictions equal untuned."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    untuned = wr_lightgbm()
    tuned = wr_lightgbm_tuned()

    untuned.fit(features, weekly)
    tuned.fit(features, weekly)

    pred_untuned = untuned.predict_distribution(features, ruleset=Ruleset.ESPN_PPR)
    pred_tuned = tuned.predict_distribution(features, ruleset=Ruleset.ESPN_PPR)

    # Compare the numeric scoring columns. model_id, generated_at, and family
    # differ by design (different prefix / fresh timestamp / same family but
    # checked separately).
    for col in ("mean", "p10", "p50", "p90"):
        np.testing.assert_array_equal(
            pred_untuned[col].to_numpy(),
            pred_tuned[col].to_numpy(),
            err_msg=f"column {col!r} differs between untuned and tuned",
        )


def test_model_id_uses_tuned_prefix() -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    tuned = wr_lightgbm_tuned()
    tuned.fit(features, weekly)
    assert tuned.model_id.startswith("lightgbm-tuned:wr:")


def test_code_hash_differs_from_untuned() -> None:
    """The tuned subclass hashes a different file set, so model_id differs."""
    untuned = wr_lightgbm()
    tuned = wr_lightgbm_tuned()
    assert untuned.code_hash != tuned.code_hash


def test_save_load_round_trip(tmp_path: Path) -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    original = wr_lightgbm_tuned()
    original.fit(features, weekly)
    artifact = tmp_path / "tuned.joblib"
    original.save(artifact)

    loaded = LightGBMTunedModel.load(artifact)
    assert loaded.model_id == original.model_id

    pred_orig = original.predict_distribution(features, ruleset=Ruleset.ESPN_PPR)
    pred_loaded = loaded.predict_distribution(features, ruleset=Ruleset.ESPN_PPR)
    for col in ("mean", "p10", "p50", "p90"):
        np.testing.assert_array_equal(
            pred_orig[col].to_numpy(), pred_loaded[col].to_numpy()
        )


def test_missing_position_in_json_raises(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"qb": {}}))  # missing rb/te/wr
    _load_tuned_params.cache_clear()
    with pytest.raises(ValueError, match="top-level keys must be"):
        _load_tuned_params(bad_json)


def test_unknown_axis_in_json_raises(tmp_path: Path) -> None:
    payload: dict[str, Any] = {pos: {} for pos in ("qb", "rb", "te", "wr")}
    payload["qb"]["passing_yards"] = {"learning_rate": 0.05, "bogus_axis": 1.0}
    bad_json = tmp_path / "bad_axis.json"
    bad_json.write_text(json.dumps(payload))
    _load_tuned_params.cache_clear()
    with pytest.raises(ValueError, match="unknown tuned-axis keys"):
        _load_tuned_params(bad_json)


def test_missing_stat_entry_raises(tmp_path: Path) -> None:
    """A (position, stat) gap in the JSON surfaces as KeyError at fit time."""
    payload: dict[str, dict[str, dict[str, float]]] = {
        pos: {} for pos in ("qb", "rb", "te", "wr")
    }
    sparse_json = tmp_path / "sparse.json"
    sparse_json.write_text(json.dumps(payload))
    _load_tuned_params.cache_clear()

    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    from projections.models.lightgbm import (
        _filter_features,
        _LightGBMConfig,
        _WR_FEATURE_COLUMNS,
        _WR_NON_NEGATIVE,
        _WR_TARGET_STATS,
    )
    from projections.schemas import Position, WrFeaturesSchema

    sparse_tuned = LightGBMTunedModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        ),
        tuned_params_path=sparse_json,
    )
    with pytest.raises(KeyError, match="missing entry for position=wr"):
        sparse_tuned.fit(features, weekly)


def test_load_tuned_params_missing_file_raises(tmp_path: Path) -> None:
    _load_tuned_params.cache_clear()
    with pytest.raises(FileNotFoundError):
        _load_tuned_params(tmp_path / "does_not_exist.json")
```

- [ ] **Step 2: Run the new tests to confirm they pass**

```bash
pytest -v tests/test_models/test_lightgbm_tuned.py
```

Expected: 9 tests PASS. The bit-exact-vs-untuned test is the headline assertion for Phase 0.

- [ ] **Step 3: Run the full models test suite for regression**

```bash
pytest -v tests/test_models/
```

Expected: all tests PASS. The new test file is additive; existing tests still pass because Task 1's hook is a behavioral no-op for `LightGBMModel`.

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add tests/test_models/test_lightgbm_tuned.py
git commit -m "$(cat <<'EOF'
test(models): cross-cutting tests for LightGBMTunedModel — Plan 5b Phase 0

Verifies (a) seeded JSON contains LGBM_DEFAULTS verbatim, (b) tuned
predictions are bit-exact vs untuned under the seeded JSON, (c) model_id
uses lightgbm-tuned: prefix, (d) save/load round-trip, (e) JSON validator
rejects bad shapes, (f) missing (pos, stat) entry surfaces as KeyError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 — Optuna search infrastructure

Goal: a runnable `scripts/tune_lightgbm.py` that runs 24 per-(position, stat) studies on the synthetic fixture (and against real cached features when invoked). No production tuning yet.

### Task 5: Add `optuna` dependency and gitignore the SQLite DB

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `data/tuned_params/.gitkeep`

- [ ] **Step 1: Add optuna to pyproject.toml**

Open `pyproject.toml`, find the `dependencies = [...]` block, add `optuna>=3.0`. Maintain alphabetical order with surrounding entries.

- [ ] **Step 2: Add the SQLite study DB to .gitignore**

Open `.gitignore`. Add a new section near the bottom:

```
# Optuna study databases — artifact-of-search; tuned JSON is the artifact-of-decision
data/tuned_params/optuna_studies.db
data/tuned_params/optuna_studies.db-journal
```

- [ ] **Step 3: Add a .gitkeep for the tuned_params directory**

```bash
touch data/tuned_params/.gitkeep
```

The directory already contains `lightgbm.json` (from Task 2), so `.gitkeep` is technically redundant. Add it anyway for clarity that this is an intentional checked-in directory.

- [ ] **Step 4: Install the new dependency**

```bash
. .venv/Scripts/activate
pip install -e ".[dev]"
python -c "import optuna; print(optuna.__version__)"
```

Expected: optuna version printed (any 3.x).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore data/tuned_params/.gitkeep
git commit -m "$(cat <<'EOF'
chore(deps): add optuna; gitignore study DB — Plan 5b Phase 1

Tuned hyperparameters live in lightgbm.json (checked in, artifact-of-decision).
The SQLite study DB is artifact-of-search; deterministic from --seed; lost
DBs reproduce by rerunning tune_lightgbm.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Implement `scripts/tune_lightgbm.py`

**Files:**
- Create: `scripts/tune_lightgbm.py`

Single file holding the search-space sampler, the per-(position, stat) study runner, and the CLI driver. Module functions are importable from tests.

- [ ] **Step 1: Write `scripts/tune_lightgbm.py`**

Create `scripts/tune_lightgbm.py`:

```python
"""Optuna-driven hyperparameter tuning for LightGBMModel — Plan 5b.

Runs 24 per-(position, stat) studies. Each study:
  - sampler: TPESampler(seed=<--seed>)
  - pruner:  MedianPruner(n_startup_trials=10, n_warmup_steps=20)
  - trials:  configured by --trials (default 50)
  - objective: sum of 5 pinball losses on the 2023 trial-scorer slice

Train slice = season in [2018..2022]; early-stop val = season == 2022;
trial scorer = season == 2023.

Tuned params are written to --out (default data/tuned_params/lightgbm.json).
Optuna studies persist at --studies-db (default
data/tuned_params/optuna_studies.db); resumable across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.models.lightgbm import LGBM_DEFAULTS, QUANTILE_GRID
from projections.schemas import Position, Stat, WeeklyStatsSchema
from projections.store import read_partition

# Search window: most-recent fold's training data. Train 2018-2022;
# 2022 is the early-stopping val; 2023 is the trial scorer.
_TRAIN_SEASONS: Final[range] = range(2018, 2023)  # 2018..2022 inclusive
_EARLY_STOP_VAL_SEASON: Final[int] = 2022
_TRIAL_SCORER_SEASON: Final[int] = 2023

_FIXED_PARAMS: Final[dict[str, Any]] = {
    "n_estimators": 4000,  # raised from default 2000; early stopping picks actual count
    "subsample_freq": 1,
    "verbose": -1,
    "random_state": 42,
}
_DEFAULT_TRIALS: Final[int] = 50
_DEFAULT_SEED: Final[int] = 42


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one set of LightGBM hyperparameters for a trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss at level alpha."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def _load_join_for_position(
    position: Position,
    *,
    seasons: Iterable[int],
    data_root: Path,
    features_root: Path,
) -> pd.DataFrame:
    """Load features + weekly_stats for the given seasons; inner-join."""
    season_list = list(seasons)
    feat_frames = [read_features(position, s, features_root=features_root) for s in season_list]
    feat = pd.concat(feat_frames, ignore_index=True)
    feat = POSITION_DISPATCH[position].feature_schema.validate(feat)

    ws_frames = [read_partition(data_root, "weekly_stats", season=s) for s in season_list]
    ws = WeeklyStatsSchema.validate(pd.concat(ws_frames, ignore_index=True))
    ws = ws[ws["position"] == position.value].copy()

    target_cols = [s.value for s in POSITION_DISPATCH[position].factories["lightgbm"]().target_stats]
    joined = feat.merge(
        ws[["gsis_id", "season", "week", *target_cols]],
        on=["gsis_id", "season", "week"],
        how="inner",
        validate="one_to_one",
    )
    return joined


def _objective(
    trial: optuna.Trial,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_score: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Trial objective: sum of 5 pinball losses on the trial-scorer slice."""
    params = _sample_params(trial)
    total = 0.0
    for q in QUANTILE_GRID:
        regressor = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            **_FIXED_PARAMS,
            **params,
        )
        try:
            regressor.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    LightGBMPruningCallback(trial, metric="quantile", valid_name="valid_0"),
                ],
            )
        except optuna.TrialPruned:
            raise
        y_pred_score = regressor.predict(x_score)
        total += _pinball_loss(y_score, y_pred_score, q)
    return total


def _run_one_study(
    position: Position,
    stat: Stat,
    *,
    joined: pd.DataFrame,
    feat_cols: Sequence[str],
    n_trials: int,
    seed: int,
    studies_db: Path | None,
) -> dict[str, float]:
    """Run one (position, stat) Optuna study; return best_params (8 axes)."""
    train_mask = joined["season"].isin(list(_TRAIN_SEASONS))
    val_mask = joined["season"] == _EARLY_STOP_VAL_SEASON
    score_mask = joined["season"] == _TRIAL_SCORER_SEASON

    if not train_mask.any() or not val_mask.any() or not score_mask.any():
        raise ValueError(
            f"Insufficient data for {position.value}/{stat.value}: "
            f"train_rows={train_mask.sum()}, val_rows={val_mask.sum()}, "
            f"score_rows={score_mask.sum()}"
        )

    x_all = joined[list(feat_cols)].to_numpy(dtype=np.float64)
    y_all = joined[stat.value].to_numpy(dtype=np.float64)

    # Important: the early-stop val (2022) is *part of* the train mask used by
    # LightGBMRegressor.fit's eval_set. The trial scorer (2023) is held out.
    # Train rows here are 2018-2022 minus the val slice — but LightGBM's
    # early stopping needs the val slice as a separate eval_set, so we pass
    # train = 2018-2021 as fit data and val = 2022 as eval_set. The 2022 val
    # mask matches the harness's existing in-fit early-stop slice for fold 2024.
    inner_train_mask = joined["season"].isin([2018, 2019, 2020, 2021])

    x_train = x_all[inner_train_mask.to_numpy()]
    y_train = y_all[inner_train_mask.to_numpy()]
    x_val = x_all[val_mask.to_numpy()]
    y_val = y_all[val_mask.to_numpy()]
    x_score = x_all[score_mask.to_numpy()]
    y_score = y_all[score_mask.to_numpy()]

    storage_url = f"sqlite:///{studies_db}" if studies_db is not None else None
    study_name = f"lightgbm:{position.value.lower()}:{stat.value}:v1"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: _objective(
            t,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_score=x_score,
            y_score=y_score,
        ),
        n_trials=n_trials,
        catch=(),
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"Study {study_name} produced 0 completed trials "
            f"(all pruned or failed). Check pruner / data."
        )
    return dict(study.best_params)


def run_studies(
    *,
    positions: Sequence[Position],
    stats_per_position: dict[Position, Sequence[Stat]],
    n_trials: int,
    seed: int,
    data_root: Path,
    features_root: Path,
    studies_db: Path | None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run studies for every (position, stat) pair; return tuned-params dict."""
    seasons = list(_TRAIN_SEASONS) + [_TRIAL_SCORER_SEASON]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for position in positions:
        joined = _load_join_for_position(
            position, seasons=seasons, data_root=data_root, features_root=features_root
        )
        feat_cols = list(POSITION_DISPATCH[position].factories["lightgbm"]().feature_columns)
        pos_key = position.value.lower()
        out.setdefault(pos_key, {})
        for stat in stats_per_position[position]:
            print(f"[tune] running study {pos_key}/{stat.value} ({n_trials} trials)…", flush=True)
            best = _run_one_study(
                position,
                stat,
                joined=joined,
                feat_cols=feat_cols,
                n_trials=n_trials,
                seed=seed,
                studies_db=studies_db,
            )
            out[pos_key][stat.value] = best
            print(f"[tune] {pos_key}/{stat.value} best_params: {best}", flush=True)
    return out


def _all_positions() -> tuple[Position, ...]:
    return (Position.QB, Position.RB, Position.TE, Position.WR)


def _stats_for(position: Position) -> tuple[Stat, ...]:
    factory = POSITION_DISPATCH[position].factories["lightgbm"]
    return factory().target_stats  # type: ignore[no-any-return]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position",
        choices=["qb", "rb", "te", "wr", "all"],
        default="all",
        help="Position to tune; 'all' runs every position.",
    )
    parser.add_argument(
        "--stat",
        default=None,
        help=(
            "Single stat to tune (e.g. 'receiving_yards'). "
            "If omitted, every target stat for the selected position(s) is tuned."
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=_DEFAULT_TRIALS, help=f"Trials per study (default {_DEFAULT_TRIALS})."
    )
    parser.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED, help=f"TPE sampler seed (default {_DEFAULT_SEED})."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tuned_params/lightgbm.json"),
        help="Output JSON path for tuned params.",
    )
    parser.add_argument(
        "--studies-db",
        type=Path,
        default=Path("data/tuned_params/optuna_studies.db"),
        help="SQLite path for Optuna study persistence; resumable across runs.",
    )
    parser.add_argument(
        "--in-memory-storage",
        action="store_true",
        help="Use in-memory study storage (overrides --studies-db); not resumable.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Raw data root."
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features"),
        help="Feature cache root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all studies but do not overwrite --out.",
    )
    args = parser.parse_args(argv)

    if args.position == "all":
        positions = list(_all_positions())
    else:
        positions = [Position(args.position.upper())]

    stats_per_position: dict[Position, Sequence[Stat]] = {}
    for p in positions:
        if args.stat is None:
            stats_per_position[p] = _stats_for(p)
        else:
            try:
                stats_per_position[p] = (Stat(args.stat),)
            except ValueError:
                print(f"unknown stat: {args.stat}", file=sys.stderr)
                return 2

    studies_db: Path | None
    if args.in_memory_storage:
        studies_db = None
    else:
        studies_db = args.studies_db
        studies_db.parent.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    tuned = run_studies(
        positions=positions,
        stats_per_position=stats_per_position,
        n_trials=args.trials,
        seed=args.seed,
        data_root=args.data_root,
        features_root=args.features_root,
        studies_db=studies_db,
    )

    if args.dry_run:
        print(f"[tune] --dry-run; not writing {args.out}")
        print(json.dumps(tuned, indent=2, sort_keys=True))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n")
    print(f"[tune] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-import the script to catch typos**

```bash
python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('tune_lightgbm', 'scripts/tune_lightgbm.py'); mod = importlib.util.module_from_spec(spec); sys.modules['tune_lightgbm'] = mod; spec.loader.exec_module(mod); print('ok:', mod._sample_params.__name__)"
```

Expected output: `ok: _sample_params`. Any import-time error (typo, missing dep) surfaces here.

- [ ] **Step 3: Run mypy + ruff against the new file**

```bash
mypy src tests scripts/tune_lightgbm.py
ruff check src tests scripts/tune_lightgbm.py
ruff format --check src tests scripts/tune_lightgbm.py
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add scripts/tune_lightgbm.py
git commit -m "$(cat <<'EOF'
feat(scripts): tune_lightgbm.py — Plan 5b Phase 1

Optuna driver: per-(position, stat) studies with TPE sampler + median pruner
via LightGBMPruningCallback; trial objective = sum of 5 pinball losses on
2023 trial scorer; 2018-2021 train / 2022 early-stop val. Studies persist
at data/tuned_params/optuna_studies.db (resumable). Tuned params written
to data/tuned_params/lightgbm.json on success.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: End-to-end test for `tune_lightgbm.py`

**Files:**
- Create: `tests/test_scripts/test_tune_lightgbm.py`

The synthetic-fixture approach mirrors `tests/test_models/test_lightgbm.py`. Tests run with `--in-memory-storage` to avoid touching disk; the SQLite-resume case uses `tmp_path`.

- [ ] **Step 1: Write the test file**

Create `tests/test_scripts/test_tune_lightgbm.py`:

```python
"""End-to-end tests for scripts/tune_lightgbm.py — Plan 5b Phase 1.

The Optuna driver loads features + weekly_stats from the cache and the raw
parquet store. To exercise it without those caches, the tests monkeypatch
`_load_join_for_position` to return a synthetic in-memory join shaped
exactly like the production output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def tune_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "tune_lightgbm", Path(__file__).resolve().parents[2] / "scripts" / "tune_lightgbm.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tune_lightgbm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_synthetic_joined(seed: int = 42, n_per_season: int = 80) -> pd.DataFrame:
    """Synthetic feature + target frame covering 2018-2023 for one position.

    Shape: (gsis_id, season, week, team, opponent) + a handful of feature
    columns + the WR target stats. Sufficient for the Optuna trial loop.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for season in range(2018, 2024):  # 2018..2023 inclusive
        for week in range(1, 18):
            for p in range(n_per_season):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": np.int64(season),
                        "week": np.int64(week),
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)

    # Synthetic feature columns — enough to fit a model on. Use names that
    # do not collide with WrFeaturesSchema; the schema is bypassed in tests.
    df["feat_a"] = rng.normal(0.0, 1.0, size=len(df))
    df["feat_b"] = rng.normal(0.0, 1.0, size=len(df))
    df["feat_c"] = rng.normal(0.0, 1.0, size=len(df))

    # Synthetic targets with mild signal so trials produce non-degenerate
    # pinball losses (not all-zero predictions).
    df["receiving_yards"] = (
        20.0 + 5.0 * df["feat_a"] + 3.0 * df["feat_b"] + rng.normal(0, 10, size=len(df))
    )
    return df


def test_sample_params_covers_all_axes(tune_module: Any) -> None:
    """_sample_params must call suggest_* for every search-space axis."""
    captured: list[tuple[str, str]] = []

    class RecordingTrial:
        def suggest_float(self, name: str, lo: float, hi: float, *, log: bool = False) -> float:
            captured.append((name, "float-log" if log else "float"))
            return (lo + hi) / 2

        def suggest_int(self, name: str, lo: int, hi: int) -> int:
            captured.append((name, "int"))
            return (lo + hi) // 2

    sampled = tune_module._sample_params(RecordingTrial())
    assert set(sampled.keys()) == {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
    captured_names = {n for n, _ in captured}
    assert captured_names == set(sampled.keys())
    log_axes = {n for n, kind in captured if kind == "float-log"}
    assert log_axes == {"learning_rate", "reg_alpha", "reg_lambda"}


def test_pinball_loss_zero_when_perfect(tune_module: Any) -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert tune_module._pinball_loss(y, y, 0.5) == pytest.approx(0.0)


def test_run_one_study_returns_best_params(
    tune_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run one tiny study end-to-end on synthetic data."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    feat_cols = ["feat_a", "feat_b", "feat_c"]

    best = tune_module._run_one_study(
        Position.WR,
        Stat.RECEIVING_YARDS,
        joined=joined,
        feat_cols=feat_cols,
        n_trials=3,
        seed=42,
        studies_db=None,  # in-memory storage
    )
    assert set(best.keys()) == {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }


def test_run_one_study_determinism_in_memory(tune_module: Any) -> None:
    """Two runs with the same seed against the same synthetic data and a
    fresh in-memory study yield identical best_params."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    feat_cols = ["feat_a", "feat_b", "feat_c"]

    best_a = tune_module._run_one_study(
        Position.WR, Stat.RECEIVING_YARDS,
        joined=joined, feat_cols=feat_cols, n_trials=3, seed=42, studies_db=None,
    )
    best_b = tune_module._run_one_study(
        Position.WR, Stat.RECEIVING_YARDS,
        joined=joined, feat_cols=feat_cols, n_trials=3, seed=42, studies_db=None,
    )
    assert best_a == best_b


def test_run_studies_writes_dense_tuned_dict(
    tune_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_studies returns a dict keyed by (position, stat) with full axes."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    monkeypatch.setattr(
        tune_module, "_load_join_for_position", lambda position, **kwargs: joined
    )

    out = tune_module.run_studies(
        positions=[Position.WR],
        stats_per_position={Position.WR: [Stat.RECEIVING_YARDS]},
        n_trials=3,
        seed=42,
        data_root=Path("data"),
        features_root=Path("data/features"),
        studies_db=None,
    )
    assert "wr" in out
    assert "receiving_yards" in out["wr"]
    assert set(out["wr"]["receiving_yards"].keys()) == {
        "learning_rate", "num_leaves", "max_depth", "min_child_samples",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
    }


def test_run_studies_resume_from_sqlite(
    tune_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming with the same study DB does not duplicate trials."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    monkeypatch.setattr(
        tune_module, "_load_join_for_position", lambda position, **kwargs: joined
    )
    db = tmp_path / "studies.db"

    tune_module.run_studies(
        positions=[Position.WR],
        stats_per_position={Position.WR: [Stat.RECEIVING_YARDS]},
        n_trials=2, seed=42, data_root=Path("data"),
        features_root=Path("data/features"), studies_db=db,
    )
    tune_module.run_studies(
        positions=[Position.WR],
        stats_per_position={Position.WR: [Stat.RECEIVING_YARDS]},
        n_trials=4, seed=42, data_root=Path("data"),
        features_root=Path("data/features"), studies_db=db,
    )

    import optuna
    storage_url = f"sqlite:///{db}"
    study = optuna.load_study(
        study_name="lightgbm:wr:receiving_yards:v1", storage=storage_url
    )
    assert len(study.trials) == 4  # 2 + 2 added on resume = 4 total


def test_main_dry_run(
    tune_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main(['--dry-run', ...])` prints the tuned dict and does not write --out."""
    from projections.schemas import Position, Stat

    joined = _build_synthetic_joined()
    monkeypatch.setattr(
        tune_module, "_load_join_for_position", lambda position, **kwargs: joined
    )
    out_path = tmp_path / "lightgbm.json"
    rc = tune_module.main(
        [
            "--position", "wr",
            "--stat", "receiving_yards",
            "--trials", "3",
            "--seed", "42",
            "--out", str(out_path),
            "--in-memory-storage",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not out_path.exists()
    captured = capsys.readouterr().out
    assert "--dry-run" in captured
```

- [ ] **Step 2: Run the new tests**

```bash
pytest -v tests/test_scripts/test_tune_lightgbm.py
```

Expected: 7 tests PASS. Total runtime ≈ 30-90s (each `--trials 3` study runs 15 sub-fits).

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy src tests scripts/tune_lightgbm.py
ruff check src tests scripts/tune_lightgbm.py
ruff format --check src tests scripts/tune_lightgbm.py
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add tests/test_scripts/test_tune_lightgbm.py
git commit -m "$(cat <<'EOF'
test(scripts): end-to-end tests for tune_lightgbm.py — Plan 5b Phase 1

Search-space coverage; pinball-loss helper; one-study end-to-end on synthetic
data; same-seed determinism on in-memory storage; SQLite resume preserves
trial count; main(--dry-run) does not overwrite --out.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Backtest harness wiring + per-position smokes

Goal: backtest harness accepts `--model lightgbm-tuned` and `--model all`; default-on smoke covers all three model classes; per-position smoke confirms each tuned factory works under POSITION_DISPATCH.

### Task 8: Update backtest harness CLI

**Files:**
- Modify: `scripts/backtest.py`

The CLI currently accepts `{baseline, lightgbm, both}`. Extend to `{baseline, lightgbm, lightgbm-tuned, both, all}`. Preserve the `both = baseline + lightgbm` legacy alias; `all` = all three.

- [ ] **Step 1: Edit `scripts/backtest.py`**

Find the `--model` argparse block (around lines 117-122 in the current file) and update:

```python
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm", "lightgbm-tuned", "both", "all"],
        default="both",
        help=(
            "Which model class(es) to run. "
            "'both' = Model A + Model C (legacy default). "
            "'all' = Model A + Model C + Model C-tuned."
        ),
    )
```

Then replace the `if args.model == "both": ... else: ...` block (around lines 125-128) with:

```python
    if args.model == "both":
        model_classes: tuple[str, ...] = ("baseline", "lightgbm")
    elif args.model == "all":
        model_classes = ("baseline", "lightgbm", "lightgbm-tuned")
    else:
        model_classes = (args.model,)
```

- [ ] **Step 2: Smoke-test the CLI parsing**

```bash
python -c "
import sys
sys.argv = ['backtest.py', '--check', '--model', 'all']
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
parser.add_argument('--model', choices=['baseline', 'lightgbm', 'lightgbm-tuned', 'both', 'all'], default='both')
args = parser.parse_args()
print('parsed model=', args.model)
"
```

Expected output: `parsed model= all`. Smoke confirms the choices list accepts the new values.

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy src tests scripts/backtest.py
ruff check src tests scripts/backtest.py
ruff format --check src tests scripts/backtest.py
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest.py
git commit -m "$(cat <<'EOF'
feat(backtest): --model lightgbm-tuned and --model all — Plan 5b Phase 2

'both' preserves legacy A+C selection. 'all' runs A+C+C-tuned. Default
remains 'both' so existing CI invocations are unaffected; switching to
'all' is opt-in for the Phase 3 snapshot regeneration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Extend default-on smoke to cover all three model classes

**Files:**
- Modify: `tests/backtest/test_backtest_smoke.py`

The smoke calls `run_backtest(...)` with one (WR, 2024) cell. Add `lightgbm-tuned` to the model_classes list.

The current smoke (Plan 5 Task 14 in `tests/backtest/test_backtest_smoke.py`) passes `model_classes=("baseline", "lightgbm")` and asserts the unique-class set is `{"baseline", "lightgbm"}`, then iterates `for model_class in ("baseline", "lightgbm"):` for the core-metrics check, and finally pins the season-calibration asymmetry by asserting `lightgbm` rows are empty for `season_calibration_*`. The tuned model has the same SAMPLED_SUMMARY-vs-QUANTILE asymmetry (TODO #28 still open), so the tuned class also has zero season-calibration rows.

- [ ] **Step 1: Edit `tests/backtest/test_backtest_smoke.py`**

Make four edits in order:

(a) Update the `model_classes` argument to `run_backtest`:

```python
        model_classes=("baseline", "lightgbm", "lightgbm-tuned"),
```

(b) Update the unique-class assertion:

```python
    assert set(out.metrics["model_class"].unique()) == {"baseline", "lightgbm", "lightgbm-tuned"}
```

(c) Update the per-model core-metrics loop:

```python
    for model_class in ("baseline", "lightgbm", "lightgbm-tuned"):
```

(d) Update the QUANTILE-family-asymmetry pin to also assert `lightgbm-tuned` does not emit `season_calibration_*` rows. After the existing `lightgbm_metrics` block, add an analogous block for the tuned class:

```python
    # Plan 5b: lightgbm-tuned shares the SAMPLED_SUMMARY-vs-QUANTILE
    # asymmetry (TODO #28). Pin the same expectation.
    tuned_metrics = out.metrics[out.metrics["model_class"] == "lightgbm-tuned"]
    tuned_season_rows = tuned_metrics[
        tuned_metrics["metric"].isin(["season_calibration_p10p90", "season_calibration_le_p90"])
    ]
    assert tuned_season_rows.empty, (
        "lightgbm-tuned is not expected to emit season_calibration_* rows yet; "
        f"got: {tuned_season_rows.to_dict('records')}"
    )
```

- [ ] **Step 3: Run the smoke**

```bash
pytest -v tests/backtest/test_backtest_smoke.py
```

Expected: PASS. Runtime budget rises from ~30s to ~45-60s (one extra LightGBM fit on (WR, 2024)).

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add tests/backtest/test_backtest_smoke.py
git commit -m "$(cat <<'EOF'
test(backtest): smoke covers all three model classes — Plan 5b Phase 2

Default-on smoke now asserts baseline + lightgbm + lightgbm-tuned all
produce finite metrics on (WR, 2024). Runtime ~45-60s; well within the
<2min CI budget.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Triple-model harness end-to-end test

**Files:**
- Create: `tests/test_backtest/test_harness_triple_model.py`

Mirrors `tests/test_backtest/test_harness_dual_model.py` but adds the tuned model class. Confirms `run_backtest(model_classes=("baseline", "lightgbm", "lightgbm-tuned"))` produces metrics keyed by `model_class` for all three.

The existing dual-model test (`tests/test_backtest/test_harness_dual_model.py`) is gated by `@pytest.mark.backtest` + a `_cache_present()` skipif (looking at `data/features/wr/season=2024`); calls `run_backtest(positions=[Position.WR], held_out_years=[2024], train_start=2018, model_classes=("baseline", "lightgbm"))`; asserts `set(df["model_class"].unique()) == {"baseline", "lightgbm"}`; then asserts the core metric set is a subset of the per-model emitted metrics for both classes.

- [ ] **Step 1: Create `tests/test_backtest/test_harness_triple_model.py`**

```python
"""End-to-end harness fold under all three model classes — Plan 5b Phase 2.

Mirrors test_harness_dual_model.py: single (WR, 2024) fold under
``model_classes=("baseline", "lightgbm", "lightgbm-tuned")``; verifies all
three models contributed rows for the cell and that the same model-class-
agnostic metric set is emitted for each. The season-calibration asymmetry
between SAMPLED_SUMMARY (baseline) and QUANTILE (lightgbm + lightgbm-tuned)
is intentionally NOT asserted here — ``tests/backtest/test_backtest_smoke.py``
pins that contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.backtest.harness import run_backtest
from projections.schemas import Position

_FEATURES_ROOT = Path("data/features")
_RAW_ROOT = Path("data/raw")


def _cache_present() -> bool:
    return (_FEATURES_ROOT / "wr" / "season=2024").exists() and (
        _RAW_ROOT / "weekly_stats" / "season=2024"
    ).exists()


@pytest.mark.backtest
@pytest.mark.skipif(
    not _cache_present(),
    reason="data/features/wr/season=2024 missing — run scripts/refresh_features.py wr",
)
def test_harness_runs_all_three_models_for_one_cell() -> None:
    """Run one (WR, 2024) fold with all three model classes; assert all
    three appear in the long-form metrics frame and that the core metric
    set is the same for each."""
    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("baseline", "lightgbm", "lightgbm-tuned"),
    )
    df = result.metrics
    assert set(df["model_class"].unique()) == {"baseline", "lightgbm", "lightgbm-tuned"}
    core_metrics = {
        "composite_rmse",
        "composite_mae",
        "spearman_topN",
        "calibration_p10p90",
    }
    for model_class in ("baseline", "lightgbm", "lightgbm-tuned"):
        per_model = set(df[df["model_class"] == model_class]["metric"].unique())
        assert core_metrics.issubset(per_model), (
            f"model_class={model_class!r} missing core metrics; got {sorted(per_model)}"
        )
```

- [ ] **Step 3: Run the new test**

```bash
pytest -v tests/test_backtest/test_harness_triple_model.py
```

Expected: PASS. If the test depends on the `--run-backtest` opt-in marker that the dual-model test uses, run with that marker:

```bash
pytest -v -m backtest --run-backtest tests/test_backtest/test_harness_triple_model.py
```

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add tests/test_backtest/test_harness_triple_model.py
git commit -m "$(cat <<'EOF'
test(backtest): triple-model harness end-to-end — Plan 5b Phase 2

Asserts run_backtest with model_classes=(baseline, lightgbm, lightgbm-tuned)
produces metrics for all three. Mirrors test_harness_dual_model.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Per-position parametrized smoke for the tuned model class

**Files:**
- Create: `tests/test_models/test_lightgbm_tuned_smoke.py`

Mirrors `tests/test_models/test_lightgbm_smoke.py` exactly — the only difference is the `factories["lightgbm-tuned"]` lookup.

- [ ] **Step 1: Read the existing lightgbm smoke**

```bash
cat tests/test_models/test_lightgbm_smoke.py
```

- [ ] **Step 2: Copy and adapt to the tuned model class**

Create `tests/test_models/test_lightgbm_tuned_smoke.py`. Verbatim copy of `test_lightgbm_smoke.py` with two single-token changes:
1. Change every `factories["lightgbm"]` → `factories["lightgbm-tuned"]`.
2. Update the module docstring's plan reference from "Plan 5" → "Plan 5b" and "LightGBMModel" → "LightGBMTunedModel".

- [ ] **Step 3: Run the new smoke**

```bash
pytest -v tests/test_models/test_lightgbm_tuned_smoke.py
```

Expected: PASS for all 4 parametrized positions. Runtime ~30s.

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add tests/test_models/test_lightgbm_tuned_smoke.py
git commit -m "$(cat <<'EOF'
test(models): parametrized per-position smoke for LightGBMTunedModel — Plan 5b Phase 2

Verbatim copy of test_lightgbm_smoke.py with the factory key swapped to
'lightgbm-tuned'. Catches dispatch-table regressions across all 4 positions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Run search and persist tuned params

Goal: run the actual 24 Optuna studies on real cached features; overwrite the tuned-params JSON; regenerate the snapshot with all three model classes.

### Task 12: Run the full Optuna search

**Files:**
- Modify: `data/tuned_params/lightgbm.json` (overwritten by the script)
- Create: `data/tuned_params/optuna_studies.db` (gitignored; not committed)

This is operational, not code-editing. Wall time ≈ 1.5-3h.

- [ ] **Step 1: Confirm feature cache is populated for 2018-2023**

```bash
ls data/features/wr/season=2018/ data/features/wr/season=2023/ data/features/qb/season=2018/ data/features/qb/season=2023/ data/features/rb/season=2018/ data/features/rb/season=2023/ data/features/te/season=2018/ data/features/te/season=2023/
```

Each directory should contain at least one `week=*/part.parquet`. If any are missing:

```bash
python scripts/refresh_features.py wr --seasons 2018 2019 2020 2021 2022 2023
python scripts/refresh_features.py qb --seasons 2018 2019 2020 2021 2022 2023
python scripts/refresh_features.py rb --seasons 2018 2019 2020 2021 2022 2023
python scripts/refresh_features.py te --seasons 2018 2019 2020 2021 2022 2023
```

- [ ] **Step 2: Run the Optuna search**

```bash
python scripts/tune_lightgbm.py --position all --trials 50 --seed 42
```

Expected output: per-(position, stat) progress lines and final `[tune] wrote data/tuned_params/lightgbm.json`. Wall time 1.5-3h. Resumable via Ctrl-C → re-run.

- [ ] **Step 3: Inspect the resulting JSON**

```bash
python -c "
import json
d = json.load(open('data/tuned_params/lightgbm.json'))
for pos in sorted(d.keys()):
    for stat in sorted(d[pos].keys()):
        params = d[pos][stat]
        print(f'{pos}/{stat}: lr={params[\"learning_rate\"]:.4f}  leaves={params[\"num_leaves\"]}  depth={params[\"max_depth\"]}  min_child={params[\"min_child_samples\"]}  ss={params[\"subsample\"]:.2f}  cs={params[\"colsample_bytree\"]:.2f}  ra={params[\"reg_alpha\"]:.4f}  rl={params[\"reg_lambda\"]:.4f}')
"
```

Expected output: 24 lines covering 4 positions × 6 stats; values within search-space bounds; no NaNs or `inf`s.

- [ ] **Step 4: Commit the tuned JSON**

```bash
git add data/tuned_params/lightgbm.json
git commit -m "$(cat <<'EOF'
chore(plan-5b): tuned hyperparameters from Optuna 50-trial search — Plan 5b Phase 3

24 per-(position, stat) studies. Per-study sampler=TPE(seed=42) +
median pruner; trial scorer = sum of 5 pinball losses on 2023 val.
Train 2018-2021; early-stop val 2022; trial scorer 2023.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Regenerate the backtest snapshot

**Files:**
- Modify: `tests/backtest/model_metrics.json` (overwritten with all three model classes)

- [ ] **Step 1: Run the full backtest with all three model classes**

```bash
python scripts/backtest.py --update-snapshot --model all
```

Expected: ~65-110 minutes wall time. Final stdout reports `Wrote tests/backtest/model_metrics.json` with the new row count.

- [ ] **Step 2: Verify the new snapshot has 1136 rows**

```bash
python -c "
import json
d = json.load(open('tests/backtest/model_metrics.json'))
classes = sorted(set(r['model_class'] for r in d))
print('row count:', len(d))
print('model classes:', classes)
counts = {c: sum(1 for r in d if r['model_class'] == c) for c in classes}
print('per-class counts:', counts)
"
```

Expected: row count 1136; model classes `['baseline', 'lightgbm', 'lightgbm-tuned']`; per-class counts `{'baseline': 400, 'lightgbm': 368, 'lightgbm-tuned': 368}`.

- [ ] **Step 3: Run the gate against the new snapshot**

```bash
pytest -v tests/backtest/test_backtest_gate.py
```

Expected: PASS. The gate compares against itself on the first run.

- [ ] **Step 4: Re-run the gate to confirm determinism**

```bash
python scripts/backtest.py --check --model all
```

Expected: zero metric drift across all 1136 rows. (This is a second full backtest run — another 65-110 min — and may be skipped at the user's discretion if Phase 3 is already deep into the day. If skipped, document in the commit message.)

- [ ] **Step 5: Commit the snapshot**

```bash
git add tests/backtest/model_metrics.json
git commit -m "$(cat <<'EOF'
chore(backtest): regenerate snapshot with C-tuned rows — Plan 5b Phase 3

1136 rows: 400 baseline + 368 lightgbm + 368 lightgbm-tuned. The 32
season_calibration_* metrics for lightgbm-tuned are skipped per the
existing SAMPLED_SUMMARY family gate (TODO #28).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Diagnostic report

Goal: per-cell A vs C vs C-tuned comparison; gate-pass-or-fail call recorded; project_management.md and TODO.md updated.

### Task 14: Build the per-cell A vs C vs C-tuned comparison table

**Files:**
- Modify: `project_management.md`

The table mirrors Plan 5's `## Plan 5 — LightGBM with Quantile Regression (Model C) — shipped` section. Same per-cell delta computation; three columns per metric (A / C / C-tuned) instead of two.

- [ ] **Step 1: Generate the comparison table from the snapshot**

```bash
python -c "
import json
import collections

d = json.load(open('tests/backtest/model_metrics.json'))

# Group by (position, year, metric) so we can pivot model_class into columns.
grouped: dict[tuple[str, int, str], dict[str, float]] = collections.defaultdict(dict)
for r in d:
    key = (r['position'], r['year'], r['metric'])
    grouped[key][r['model_class']] = r['value']

# Headline metrics for the per-cell table.
metrics_of_interest = ['composite_rmse', 'composite_mae', 'spearman_topN', 'calibration_p10p90']
positions = ['QB', 'RB', 'TE', 'WR']
years = [2021, 2022, 2023, 2024]

print('| Cell | metric | A | C | C-tuned | C−A | C-tuned−A | C-tuned−C |')
print('|---|---|---|---|---|---|---|---|')
for pos in positions:
    for year in years:
        for metric in metrics_of_interest:
            row = grouped.get((pos, year, metric))
            if not row:
                continue
            a = row.get('baseline')
            c = row.get('lightgbm')
            ct = row.get('lightgbm-tuned')
            if a is None or c is None or ct is None:
                continue
            print(f'| {pos} {year} | {metric} | {a:.4f} | {c:.4f} | {ct:.4f} | {(c-a):+.4f} | {(ct-a):+.4f} | {(ct-c):+.4f} |')
" > /tmp/plan-5b-table.md
cat /tmp/plan-5b-table.md
```

Save the resulting table; you'll paste it into project_management.md in Step 3.

- [ ] **Step 2: Compute the §1.3 adoption-gate verdict for C-tuned vs A**

```bash
python -c "
import json
import collections

d = json.load(open('tests/backtest/model_metrics.json'))
grouped = collections.defaultdict(dict)
for r in d:
    key = (r['position'], r['year'], r['metric'])
    grouped[key][r['model_class']] = r['value']

positions = ['QB', 'RB', 'TE', 'WR']
years = [2021, 2022, 2023, 2024]

# Criterion 1: composite_rmse strictly lower on >=12 of 16 cells; not worse by >1% on any cell.
rmse_lower = 0
rmse_max_pct_worse = 0.0
for pos in positions:
    for year in years:
        a = grouped[(pos, year, 'composite_rmse')]['baseline']
        ct = grouped[(pos, year, 'composite_rmse')]['lightgbm-tuned']
        pct = (ct - a) / a * 100
        if pct < 0:
            rmse_lower += 1
        rmse_max_pct_worse = max(rmse_max_pct_worse, pct)
print(f'Criterion 1 — composite_rmse: lower on {rmse_lower}/16 cells; max pct worse {rmse_max_pct_worse:+.2f}%; pass={rmse_lower >= 12 and rmse_max_pct_worse <= 1.0}')

# Criterion 2: spearman within +-0.005 on every cell.
spearman_max_abs_delta = 0.0
spearman_outside_tol = 0
for pos in positions:
    for year in years:
        a = grouped[(pos, year, 'spearman_topN')]['baseline']
        ct = grouped[(pos, year, 'spearman_topN')]['lightgbm-tuned']
        delta = ct - a
        if abs(delta) > 0.005:
            spearman_outside_tol += 1
        spearman_max_abs_delta = max(spearman_max_abs_delta, abs(delta))
print(f'Criterion 2 — spearman: {spearman_outside_tol}/16 outside +-0.005; max abs delta {spearman_max_abs_delta:.4f}; pass={spearman_outside_tol == 0}')

# Criterion 3: calibration_p10p90 no worse on any cell; mean delta >= +0.02.
calib_deltas = []
calib_worse = 0
for pos in positions:
    for year in years:
        a = grouped[(pos, year, 'calibration_p10p90')]['baseline']
        ct = grouped[(pos, year, 'calibration_p10p90')]['lightgbm-tuned']
        delta = ct - a
        calib_deltas.append(delta)
        if delta < -0.005:
            calib_worse += 1
mean_delta = sum(calib_deltas) / len(calib_deltas)
print(f'Criterion 3 — calibration_p10p90: worse on {calib_worse}/16 cells; mean delta {mean_delta:+.4f}; pass={calib_worse == 0 and mean_delta >= 0.02}')
"
```

Capture the three pass/fail verdicts.

- [ ] **Step 3: Append the report to `project_management.md`**

Open `project_management.md`. Insert a new top section (above the existing `## Plan 5 — ...` section) with this template, filling in the table and verdicts from Steps 1-2:

```markdown
## Plan 5b — Optuna Tuning of Model C (Model C-tuned) — shipped (run YYYY-MM-DD)

**Closes:** TODO #26 follow-up "if tuning closes the gap, revisit adoption."

24 per-(position, stat) Optuna studies × 50 trials with TPE + median pruner.
Trial scorer = sum of 5 pinball losses on 2023 val. Train 2018-2021;
early-stop val 2022. Tuned hyperparameters reused across all 4 backtest
folds; per-fold tuning deferred to Plan 5c if the gate flips.

### Per-position model_ids

| Position | Model A model_id | Model C model_id | Model C-tuned model_id |
|---|---|---|---|
| WR | (Plan 3e Phase 1: ...) | (Plan 5: lightgbm:wr:a4dd5a82:2018-2023) | <fill in from artifact> |
| QB | ... | ... | ... |
| RB | ... | ... | ... |
| TE | ... | ... | ... |

### Adoption-gate verdict — Model C-tuned vs Model A

| Criterion | Threshold | Actual (C-tuned) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12/16 cells; not worse by >1% on any | C-tuned <= A on 12+; max +1% worse | <fill from Step 2> | <PASS/FAIL> |
| Spearman top-N within +-0.005 of A on every cell | All 16 within +-0.005 | <fill> | <PASS/FAIL> |
| Weekly mean [p10,p90] coverage no worse on any cell; mean delta >= +0.02 | No regressions; mean delta >= +0.02 | <fill> | <PASS/FAIL> |

### Side-by-side per-cell comparison

<paste the table from Step 1>

### Diagnostic and next steps

<2-3 paragraph analysis: which positions did tuning help; which axes Optuna favored vs LGBM_DEFAULTS; whether the gap closed enough to justify per-fold tuning (Plan 5c) or whether the lever isn't tuning. Use Optuna's importance_get_param_importances if convenient.>

**Verdict:** <PASS / FAIL>. <If PASS: file Plan 5c (per-fold tuning + production-default switch). If FAIL: note next-step decision — Plan 6 ensemble, TODO #3 PBP features, or TODO #23 target decomposition — back to user.>

---
```

Replace the `<fill ... >` placeholders with values from Steps 1-2. The model_ids come from `models/artifacts/lightgbm-tuned-{pos}-2018-2023-{hash}.joblib` filenames after the snapshot regeneration.

- [ ] **Step 4: Commit the report**

```bash
git add project_management.md
git commit -m "$(cat <<'EOF'
docs(plan-5b): record Model C-tuned ship + adoption-gate verdict — Plan 5b Phase 4

Per-cell A vs C vs C-tuned table; the three §1.3 criteria evaluated for
C-tuned vs A; next-step recommendation logged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Update TODO.md

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Edit TODO.md based on the gate verdict**

Open `TODO.md`. Two cases:

**If C-tuned PASSED the gate:**
- Locate TODO #26 ("Plan 5 — LightGBM with quantile regression (Model C) — closed in Plan 5"). Append a new line:
  - "Plan 5b followup: tuning flipped the gate. Filed TODO #29 — Plan 5c (per-fold tuning + production-default switch from Model A to Model C-tuned)."
- Add a new TODO #29 below the highest existing number:
  ```markdown
  ### 29. Plan 5c — per-fold tuning + production-default switch to Model C-tuned

  Plan 5b's tune-once-on-2024-fold experiment passed the §1.3 adoption gate.
  Plan 5c does the strict version: re-tune hyperparameters per held-out year
  (4 separate Optuna runs) and switch the production default from Model A
  to Model C-tuned. See `docs/superpowers/specs/2026-04-27-plan-5b-tuning-design.md`
  §1.2 for scope.
  ```

**If C-tuned FAILED the gate:**
- Locate TODO #26. Append:
  - "Plan 5b followup: tuning did not flip the gate. Pivot to Plan 6 (ensemble) or feature work (TODO #3 / TODO #23). Plan 5c not filed."

In both cases, also clean up Plan 5 §`Followup plans` reference numbering if it no longer matches what's in TODO.md.

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "$(cat <<'EOF'
docs(todo): close Plan 5b followup; record next-step decision — Plan 5b Phase 4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

After Task 15 commits:

- [ ] **Step 1: Run the full test suite**

```bash
. .venv/Scripts/activate
pytest -v
```

Expected: all tests PASS. The default-on smoke and per-position tuned smokes already exercise the new model class; the gated `pytest -m backtest --run-backtest` is opt-in and was already verified in Task 13.

- [ ] **Step 2: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/plan-5b-tuning
gh pr create --title "Plan 5b: Optuna tuning of Model C (Model C-tuned)" --body "$(cat <<'EOF'
## Summary
- Adds `LightGBMTunedModel` (Model C-tuned) as a third model class coexisting with Models A and C.
- 24 per-(position, stat) Optuna studies × 50 trials with TPE + median pruner.
- Tuned hyperparameters checked into `data/tuned_params/lightgbm.json`; SQLite study DB gitignored.
- Backtest snapshot extended from 768 → 1136 rows; per-cell A vs C vs C-tuned comparison logged.
- Adoption-gate verdict: <PASS / FAIL — fill in from project_management.md>.

## Test plan
- [x] `pytest -v` clean (all phases)
- [x] `mypy src tests` clean
- [x] `ruff check src tests` and `ruff format --check src tests` clean
- [x] `pytest -m backtest --run-backtest` (snapshot gate) clean against the regenerated `model_metrics.json`
- [x] Default-on smoke covers all three model classes
- [x] Per-position parametrized smoke covers all three model classes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Summary of files touched

| Phase | New files | Modified files |
|---|---|---|
| 0 | `src/projections/models/lightgbm_tuned.py`, `data/tuned_params/lightgbm.json`, `tests/test_models/test_lightgbm_tuned.py` | `src/projections/models/lightgbm.py`, `src/projections/models/__init__.py` |
| 1 | `scripts/tune_lightgbm.py`, `tests/test_scripts/test_tune_lightgbm.py`, `data/tuned_params/.gitkeep` | `pyproject.toml`, `.gitignore` |
| 2 | `tests/test_backtest/test_harness_triple_model.py`, `tests/test_models/test_lightgbm_tuned_smoke.py` | `scripts/backtest.py`, `tests/backtest/test_backtest_smoke.py` |
| 3 | — | `data/tuned_params/lightgbm.json`, `tests/backtest/model_metrics.json` |
| 4 | — | `project_management.md`, `TODO.md` |

Per-phase file count is ≤5 throughout, satisfying CLAUDE.md's PHASED EXECUTION rule.
