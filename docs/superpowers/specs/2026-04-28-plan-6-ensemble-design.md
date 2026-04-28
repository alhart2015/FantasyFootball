# Plan 6 — Model D ensemble (A + C-NB) — design spec

**Status:** drafted 2026-04-28. Branch: `feat/plan-6-ensemble`.

**Closes:** the per-position split exposed by Plan 5c, where Model C-NB cleanly wins QB on every metric while Model A retains the calibration win on RB / TE / WR. Plan 6 builds a per-(position, stat) calibration-aware weighted mixture of A and C-NB and tests whether the per-position split is exploitable under the §1.3 adoption gate.

**Predecessors referenced:** Plan 5c (Model C-NB shipped as peer; QB clean win, RB/TE/WR calibration regression), Plan 5b (Model C-tuned; established the `LightGBMTunedModel` subclass + `data/tuned_params/` artifact pattern), Plan 5 (Model C; established `LightGBMModel`, `QuantileDistribution`, factories per position), Plan 7 (calibration-aware NB-2 fitting; stopped at Phase 0 because per-stat [p10, p90] coverage doesn't decompose to composite [p10, p90] — relevant cautionary lesson here).

---

## 1. Goals and adoption gate

### 1.1 Mechanism hypothesis

For each (position, stat) cell, neither A's distribution nor C-NB's distribution is uniformly best:

- On QB cells, C-NB strictly wins all four metrics (RMSE, MAE, Spearman, calibration) on all four backtest years.
- On RB / TE / WR cells, C-NB wins RMSE on most cells but regresses on [p10, p90] calibration by 6–13 percentage points because NB-2 dispersion fitted on training residuals is too narrow at the tails for held-out years.

A weighted mixture `w · F_A + (1-w) · F_CNB` per (position, stat) has the property that:

1. Mean is the linear combination of component means, so any RMSE intermediate between A and C-NB is reachable.
2. Variance equals `w·var_A + (1-w)·var_B + w(1-w)(mean_A − mean_B)²` — strictly ≥ either component when their means differ, so calibration generally widens. This is the mechanism that addresses RB / TE / WR's narrow-NB-2 calibration regression.
3. Setting `w = 0` recovers C-NB exactly; `w = 1` recovers A. Optimizing `w ∈ [0, 1]` can never produce something worse than the better of the two components on a strictly-proper scoring rule (modulo overfitting validation noise).

### 1.2 What we are NOT doing in this plan

- Not building per-row gated weights (a meta-learner mapping features → weights). Per-(position, stat) constant weights are the minimum-viable scope; per-row gating is deferred to a future plan if N>2 components or feature-dependent weighting is later motivated.
- Not generalizing to N>2 components. The ensemble is binary by construction (A and C-NB).
- Not addressing TODO #28 (`aggregate_to_season` widening to QUANTILE / MIXED). Ensemble inherits the same season-aggregation skip as C / C-tuned / C-NB.
- Not pruning Model C-tuned. TODO #29 stays open and is re-evaluated after Plan 6 lands.
- Not optimizing weights to directly minimize composite [p10, p90] coverage error. Per-stat pinball at q ∈ {0.10, 0.90} is the loss; composite coverage is a downstream observable, not the optimizer's target. (See §3.3 for the rationale and the explicit risk that this proxy may fail to move composite calibration — Plan 7's lesson.)

### 1.3 Adoption gate (verbatim from prior plans)

Model D must beat Model A on all three criteria to be adopted as production default:

1. **RMSE**: composite RMSE strictly lower on ≥ 12 of 16 (position, year) cells; max +1% worse on any cell.
2. **Spearman**: top-N Spearman correlation within ±0.005 of A on every cell.
3. **Calibration**: weekly mean [p10, p90] coverage no worse than A on any cell; mean coverage delta vs A ≥ +0.02 across all 16 cells.

If all three pass → **adopt**: update `scripts/train_baseline.py`'s canonical default to `ensemble`; update CLAUDE.md and `project_management.md` to call out Model D as production. A / C / C-tuned / C-NB stay registered for backtest comparability.

If any fail → **ship as peer** alongside A / C / C-tuned / C-NB; A stays default. Either landing is valid; the verdict is captured in `project_management.md` after the run.

---

## 2. System architecture

### 2.1 New components

```
src/projections/distributions/mixture.py        # NEW — MixtureDistribution
src/projections/distributions/codec.py          # +MIXTURE branch in pack/unpack
src/projections/distributions/base.py           # +cdf(x) on Distribution Protocol
src/projections/distributions/parametric.py     # +cdf on NORMAL / GAMMA / NB / STUDENT_T
src/projections/distributions/quantile.py       # +cdf on QUANTILE
src/projections/models/ensemble.py              # NEW — EnsembleModel
src/projections/models/__init__.py              # +ensemble factories per position
src/projections/backtest/harness.py             # +EnsembleModel in cast widening
scripts/backtest.py                             # --model ensemble + --model all extension
data/ensemble_weights/                          # NEW directory; per-fold JSON artifacts (checked in)
tests/test_distributions/test_mixture.py        # NEW — MixtureDistribution math
tests/test_distributions/test_cdf.py            # NEW — Protocol cdf parity vs scipy
tests/test_models/test_ensemble_model.py        # NEW — fit/predict cross-cutting
tests/test_models/test_ensemble_weight_fit.py   # NEW — weight optimizer math
tests/test_backtest/                            # extend smoke to include ensemble
tests/backtest/model_metrics.json               # 1504 → 1872 rows after Phase 5
```

### 2.2 `EnsembleModel` class (new)

Implements the existing `Model` Protocol (no Protocol changes — TODO #27's `Fitted[Model]` split stays deferred).

```python
@dataclass(slots=True)
class _EnsembleConfig:
    position: Position
    target_stats: tuple[Stat, ...]
    child_a_factory: Callable[[], BaselineModel]
    child_b_factory: Callable[[], LightGBMNbModel]
    weights_dir: Path  # default: data/ensemble_weights/

class EnsembleModel:
    def __init__(self, *, config: _EnsembleConfig) -> None: ...

    @property
    def position(self) -> Position: ...

    @property
    def code_hash(self) -> str:
        # SHA-256 first 8 hex of:
        #   ensemble.py + child_a's code_hash + child_b's code_hash
        #   + sorted(self._weights.items()) JSON-canonical bytes
        ...

    @property
    def model_id(self) -> str:
        # "ensemble:<pos>:<8-char-code-hash>:<train-start>-<train-end>"
        ...

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        # See §3.1 for the per-fold data flow.
        ...

    def predict_distribution(
        self, features: pd.DataFrame, ruleset: Ruleset
    ) -> pd.DataFrame:
        # See §3.4.
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> EnsembleModel: ...
```

Two factories per position: `qb_ensemble`, `rb_ensemble`, `te_ensemble`, `wr_ensemble`. Each constructs an `EnsembleModel` wired with the matching `BaselineModel` and `LightGBMNbModel` factories.

### 2.3 `MixtureDistribution` class (new)

Implements the `Distribution` Protocol structurally. Holds two child Distributions and a scalar weight.

```python
@dataclass(slots=True, frozen=True)
class MixtureDistribution:
    component_a: Distribution
    component_b: Distribution
    weight: float    # in (0, 1); fraction of mass on component_a

    def mean(self) -> float: ...
    def std(self) -> float: ...
    def cdf(self, x: float) -> float: ...
    def quantile(self, q: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]: ...
```

Math:

```
mean()      = w·F_a.mean() + (1-w)·F_b.mean()
variance()  = w·F_a.var() + (1-w)·F_b.var() + w(1-w)(F_a.mean() - F_b.mean())²
std()       = sqrt(variance)
cdf(x)      = w·F_a.cdf(x) + (1-w)·F_b.cdf(x)
quantile(q) = brentq( cdf(x) - q == 0 ) on a wide initial bracket
sample(n)   = vectorized Bernoulli(w) mask → component_a.sample(...) | component_b.sample(...)
```

Construction validates `0 < weight < 1`. Boundary weights are clipped during fitting (§3.3) so the dataclass never sees `w ∈ {0, 1}`.

### 2.4 Distribution Protocol extension — `cdf`

The `Distribution` Protocol gains one method:

```python
class Distribution(Protocol):
    def mean(self) -> float: ...
    def std(self) -> float: ...
    def quantile(self, q: float) -> float: ...
    def sample(self, n: int, rng=None) -> NDArray[np.float64]: ...
    def cdf(self, x: float) -> float: ...   # NEW
```

Each existing implementer adds an analytic `cdf`. The four parametric implementations are one-line wrappers around scipy; `QuantileDistribution` is a small piecewise-linear inversion (~10–15 LOC):

| Class | Implementation |
|---|---|
| `ParametricNormal` | `stats.norm.cdf(x, loc=mean_, scale=std_)` |
| `ParametricGamma` | `stats.gamma.cdf(x, a=shape, scale=scale)` |
| `ParametricNegativeBinomial` | `stats.nbinom.cdf(x, n=n, p=p)` |
| `ParametricStudentT` | `stats.t.cdf(x, df=df_, loc=loc_, scale=scale_)` |
| `QuantileDistribution` | piecewise-linear inverse of the stored (quantiles, values) with endpoint clamping |

Additive change: no existing caller invokes `cdf`, so nothing breaks. New tests in `tests/test_distributions/test_cdf.py` verify each implementation against scipy at sampled points.

### 2.5 Codec — new `MIXTURE` per-stat family

`DistributionFamily` enum gains `MIXTURE = "MIXTURE"`. The row-level `family` value for ensemble rows stays `MIXED` (semantics identical to C-NB rows: per-stat distributions vary in family per row).

Codec encoding:

```
MIXTURE: {
    "family": "MIXTURE",
    "weight": float,
    "component_a": {"family": "<family>", ...family-specific params...},
    "component_b": {"family": "<family>", ...family-specific params...},
}
```

Implementation refactors `pack_per_stat_params` and `unpack_per_stat_params` to extract per-component encoding into private helpers `_pack_single` / `_unpack_single`. The MIXTURE branch calls these recursively.

`schema_version` bumps from 1 → 2. Old v1 blobs (no MIXTURE entries) decode unchanged.

### 2.6 Ensemble weights persistence

Per-(position, train-span) weights persist to `data/ensemble_weights/{model_id}.json`, checked in. One file per fold per position. Mirrors the Plan 5b `data/tuned_params/lightgbm.json` checked-in artifact pattern.

```json
{
  "model_class": "ensemble",
  "position": "QB",
  "train_seasons": [2018, 2023],
  "calibration_year": 2023,
  "child_a_model_id": "baseline:qb:c98738f3:2018-2023",
  "child_b_model_id": "lightgbm-nb:qb:3ae5b940:2018-2023",
  "weights": {
    "passing_yards": 0.482,
    "passing_tds":   0.731,
    "interceptions": 0.624,
    "rushing_yards": 0.515,
    "rushing_tds":   0.689,
    "fumbles_lost":  0.702
  },
  "loss_per_stat": {
    "passing_yards": 12.341,
    "passing_tds":    0.812,
    ...
  },
  "fitted_at": "2026-04-28T12:34:56Z"
}
```

`EnsembleModel.code_hash` includes the weights file content (sorted JSON-canonical bytes), so any change to weights or to component children's code hashes invalidates `model_id`.

`EnsembleModel.save(path)` joblib-pickles `(child_a, child_b, weights, train_start, train_end)` for the prediction-time artifact. The JSON sidecar is the human-inspectable record; the joblib pickle is the load-time artifact.

---

## 3. Data flow

### 3.1 Per-fold training (`EnsembleModel.fit`)

For a backtest fold with held-out year Y, the harness calls `model.fit(features, weekly_stats)` with `features` and `weekly_stats` spanning `[train_start, Y-1]`. Internally:

```
seasons = sorted(unique(features.season))
assert len(seasons) >= 3   # need [S, Y-2] + Y-1 calibration year + (children need their own internal val slice)

cal_year = seasons[-1]                          # Y-1
weight_fit_seasons = seasons[:-1]               # [S, Y-2]
prediction_seasons = seasons                    # [S, Y-1]

# Stage 1 — weight-fit children
child_a_wf = self._config.child_a_factory()
child_b_wf = self._config.child_b_factory()
child_a_wf.fit(features[features.season.isin(weight_fit_seasons)],
                weekly_stats[weekly_stats.season.isin(weight_fit_seasons)])
child_b_wf.fit(features[features.season.isin(weight_fit_seasons)],
                weekly_stats[weekly_stats.season.isin(weight_fit_seasons)])

# Stage 2 — calibration-year predictions
cal_features = features[features.season == cal_year]
cal_actuals  = weekly_stats[weekly_stats.season == cal_year]
pred_a = child_a_wf.predict_distribution(cal_features, ruleset=Ruleset.espn_ppr())
pred_b = child_b_wf.predict_distribution(cal_features, ruleset=Ruleset.espn_ppr())

# Stage 3 — fit per-(stat) weights via per-stat pinball at q in {0.10, 0.90}
self._weights = _fit_ensemble_weights(
    pred_a=pred_a, pred_b=pred_b, cal_actuals=cal_actuals,
    target_stats=self._config.target_stats,
)

# Stage 4 — re-fit children on the full prediction span
self._child_a = self._config.child_a_factory()
self._child_b = self._config.child_b_factory()
self._child_a.fit(features, weekly_stats)
self._child_b.fit(features, weekly_stats)

# Stage 5 — persist artifacts
self._train_start = int(seasons[0])
self._train_end = int(seasons[-1])
self._calibration_year = int(cal_year)
self._is_fitted = True
_write_weights_json(self)
```

Compute roughly doubles per fold vs C-NB alone (4 child fits instead of 2). C-NB caches its tuned hyperparameters from `data/tuned_params/lightgbm.json` so the regressor fits are seconds; A is also seconds. Net incremental ≈ 5–10 minutes per backtest run.

### 3.2 Walk-forward leakage analysis

For held-out year Y, the prediction children use `[S, Y-1]` and predict Y — same as A / C-NB today. Their internal early-stop validation slice is `Y-1` (the most recent training year), as established in `LightGBMNbModel.fit` line 199.

The weight-fit children use `[S, Y-2]` and predict `Y-1`. Their internal early-stop validation slice is `Y-2`. The calibration year `Y-1` was therefore not seen by the weight-fit children at any stage of training — clean walk-forward separation.

Held-out year Y itself is never used for any decision in this fold — it is only the prediction target.

### 3.3 Weight fitting (`_fit_ensemble_weights`)

For each `stat ∈ target_stats`, fit one scalar `w[stat] ∈ (0, 1)` minimizing summed pinball loss at q ∈ {0.10, 0.90}:

```python
def _fit_ensemble_weights(
    *, pred_a: pd.DataFrame, pred_b: pd.DataFrame, cal_actuals: pd.DataFrame,
    target_stats: tuple[Stat, ...],
) -> dict[Stat, float]:
    """Per-(stat) 1-D bounded brent on summed pinball at q in {0.10, 0.90}."""
    weights: dict[Stat, float] = {}

    # Decode per-row per-stat distributions for both children once up front.
    per_row_per_stat_a = [unpack_per_stat_params(bytes(b)) for b in pred_a["params"]]
    per_row_per_stat_b = [unpack_per_stat_params(bytes(b)) for b in pred_b["params"]]

    # Inner-join cal_actuals onto pred_a/pred_b on (gsis_id, season, week) so
    # row alignment is explicit; raise on misalignment rather than zip silently.
    aligned_actuals = _align_actuals(pred_a, cal_actuals, target_stats)

    for stat in target_stats:
        actuals = aligned_actuals[stat.value].to_numpy(dtype=np.float64)
        components_a = [row_dists[stat] for row_dists in per_row_per_stat_a]
        components_b = [row_dists[stat] for row_dists in per_row_per_stat_b]

        def loss(w: float) -> float:
            total = 0.0
            for a_dist, b_dist, actual in zip(components_a, components_b, actuals, strict=True):
                mix = MixtureDistribution(component_a=a_dist, component_b=b_dist, weight=w)
                q10_pred = mix.quantile(0.10)
                q90_pred = mix.quantile(0.90)
                total += _pinball(actual, q10_pred, 0.10)
                total += _pinball(actual, q90_pred, 0.90)
            return total

        result = minimize_scalar(
            loss, method="bounded", bounds=(0.001, 0.999), options={"xatol": 1e-3}
        )
        if not result.success or not np.isfinite(result.fun):
            # Fall back to a coarse grid search.
            grid = np.linspace(0.001, 0.999, 11)
            losses = [loss(w) for w in grid]
            weights[stat] = float(grid[int(np.argmin(losses))])
        else:
            weights[stat] = float(np.clip(result.x, 0.001, 0.999))

    return weights


def _pinball(actual: float, q_pred: float, q: float) -> float:
    """Standard quantile pinball loss."""
    return max((q - 1.0) * (q_pred - actual), q * (q_pred - actual))
```

Compute per cell: ~50 outer brent iterations × N_rows × 2 quantiles × ~30 inner brentq iterations × analytic CDF eval ≈ a few seconds. 24 cells × 4 folds ≈ 5 minutes total weight-fitting overhead per backtest run.

**Why per-stat pinball, not composite [p10, p90].** Per-stat pinball decouples cleanly — 1-D bounded optimization per cell. Composite-direct optimization would require a joint fit over all per-stat weights via Monte Carlo evaluation per weight update; ~100× slower with added MC noise. Plan 7's Phase 0 diagnostic established that per-stat coverage at [p10, p90] doesn't algebraically decompose to composite coverage at [p10, p90], so per-stat-optimal weights may not produce composite-optimal calibration. We accept this proxy explicitly — if Phase 5's adoption-gate run shows per-stat coverage moving but composite not budging, that is a clean diagnostic for a follow-up plan to optimize composite directly. We do not pre-commit to that follow-up; we run and measure.

**Edge cases and fallbacks.**
- `NB.cdf` at non-integer x is well-defined (scipy treats it as the CDF of the integer ≤ x). `QuantileDistribution.cdf` is piecewise-linear with clamping at endpoints. Both are handled by the brentq inversion in `MixtureDistribution.quantile`.
- If `loss(w)` is non-finite for some w (e.g., a numerical pathology in CDF inversion), the grid-search fallback picks the best of 11 grid points. Logged at `RuntimeWarning` so failures are visible.
- Bounded clip `[0.001, 0.999]` keeps the mixture proper. Empirically, the optimal `w` is rarely at the boundary because the variance-widening property of the mixture means an interior point usually beats either pure component on the validation pinball.

### 3.4 Predict-time mixture construction (`EnsembleModel.predict_distribution`)

```python
def predict_distribution(
    self, features: pd.DataFrame, ruleset: Ruleset
) -> pd.DataFrame:
    if not self._is_fitted:
        raise RuntimeError("predict_distribution requires fit() first")

    pred_a = self._child_a.predict_distribution(features, ruleset)
    pred_b = self._child_b.predict_distribution(features, ruleset)
    # pred_a is SAMPLED_SUMMARY; pred_b is MIXED.

    # Inner-align rows on (gsis_id, season, week). Both children predict from
    # the same features; alignment must be 1:1 — assert that explicitly.
    pred_a = pred_a.set_index(["gsis_id", "season", "week"])
    pred_b = pred_b.set_index(["gsis_id", "season", "week"])
    assert pred_a.index.equals(pred_b.index), \
        "child predictions misaligned — both children should predict on the same features"

    out_rows: list[dict[str, Any]] = []
    generated_at = datetime.now(UTC)

    for row_idx, (key, _row_a) in enumerate(pred_a.iterrows()):
        gsis_id, season, week = key
        per_stat_a = unpack_per_stat_params(bytes(pred_a.iloc[row_idx]["params"]))
        per_stat_b = unpack_per_stat_params(bytes(pred_b.iloc[row_idx]["params"]))

        per_stat_dists: dict[Stat, Distribution] = {}
        for stat in self._config.target_stats:
            per_stat_dists[stat] = MixtureDistribution(
                component_a=per_stat_a[stat],
                component_b=per_stat_b[stat],
                weight=self._weights[stat],
            )

        seed = derive_row_seed(
            gsis_id=str(gsis_id), season=int(season), week=int(week),
            ruleset_name=ruleset.name,
        )
        composite = score_distribution(per_stat_dists, ruleset, seed=seed)

        out_rows.append({
            "gsis_id": str(gsis_id),
            "season": int(season),
            "week": int(week),
            "position": self._config.position.value,
            "team": str(pred_a.iloc[row_idx]["team"]),
            "opponent": str(pred_a.iloc[row_idx]["opponent"]),
            "ruleset": ruleset.name,
            "family": DistributionFamily.MIXED.value,
            "params": pack_per_stat_params(per_stat_dists),
            "mean": composite.mean(),
            "p10":  composite.quantile(0.10),
            "p50":  composite.quantile(0.50),
            "p90":  composite.quantile(0.90),
            "model_id": self.model_id,
            "generated_at": pd.Timestamp(generated_at).as_unit("us"),
        })

    out = pd.DataFrame(out_rows)
    # Standard string-column coercion (mirrors LightGBMNbModel).
    for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    out["position"] = out["position"].astype(_PYARROW_STR)
    return ProjectionWeeklySchema.validate(out)
```

The composite's `mean / p10 / p50 / p90` come from `score_distribution`'s existing Monte Carlo path because `MixtureDistribution.sample()` already implements the right semantics (Bernoulli mask → component sample). Per-row deterministic seeds via `derive_row_seed` keep the snapshot stable across re-runs.

### 3.5 Backtest harness integration

Single touch point in `src/projections/backtest/harness.py` — widen the cast at line 261 to include `EnsembleModel`:

```python
model = cast(
    BaselineModel | LightGBMModel | EnsembleModel,
    dispatch.factories[model_class](),
)
```

Everything else is harness-agnostic: `_per_stat_means_from_predictions` already round-trips through `unpack_per_stat_params`, which works for `MixtureDistribution` once the codec branch is registered. Per-row metrics, naive baseline, snapshot extension all flow through unchanged.

`scripts/backtest.py` `--model` argparse choices add `"ensemble"`; `--model all` expands from 4 → 5 model classes:

```python
elif args.model == "all":
    model_classes = ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb", "ensemble")
```

Snapshot extension: 1504 → 1872 rows (368 new rows: 23 per-cell metrics × 4 positions × 4 years). The 32 `season_calibration_*` rows still skip via the existing `SAMPLED_SUMMARY`-only family gate — Plan 6 inherits the same asymmetry as C / C-tuned / C-NB. TODO #28 stays open.

---

## 4. Testing strategy

### 4.1 Phase 0 (Distribution Protocol cdf extension)

`tests/test_distributions/test_cdf.py` (new):

- For each of `ParametricNormal`, `ParametricGamma`, `ParametricNegativeBinomial`, `ParametricStudentT`: assert `cdf(x)` matches `scipy.<dist>.cdf(x, ...)` at 10 sampled x values per distribution.
- For `QuantileDistribution`: synthetic 5-quantile fixture; assert `cdf` is the piecewise-linear inverse of the stored quantile function; assert clamping at endpoints; assert monotone non-decreasing on a fine x grid.

### 4.2 Phase 1 (MixtureDistribution math)

`tests/test_distributions/test_mixture.py` (new):

- Constructor validation: `weight ∉ (0, 1)` raises `ValueError`.
- Mean: equals `w·F_a.mean() + (1-w)·F_b.mean()` for several (component_a, component_b, w) triples.
- Variance: matches the analytic formula `w·var_A + (1-w)·var_B + w(1-w)(mean_A − mean_B)²`.
- CDF: monotone non-decreasing; `cdf(-1e9) ≈ 0`, `cdf(1e9) ≈ 1` (modulo discrete support for NB).
- Quantile: round-trips through cdf at q ∈ {0.05, 0.10, 0.50, 0.90, 0.95} to within 1e-6.
- Sample: empirical mean and std converge to analytic mean/std with N=20000 (tolerance 1%).
- Component pairs covered: Normal+Normal, Normal+Gamma, Gamma+Quantile, NB+Normal, NB+NB, NB+Quantile.

### 4.3 Phase 2 (EnsembleModel scaffolding)

`tests/test_models/test_ensemble_model.py` (new), parametrized over (QB, RB, TE, WR):

- `EnsembleModel.fit` correctly splits seasons (assert weight-fit children see `[S, Y-2]`, prediction children see `[S, Y-1]`).
- `EnsembleModel.predict_distribution` returns a `ProjectionWeeklySchema`-validated frame.
- Round-trip test: pack → unpack restores the per-stat MixtureDistribution structure (component A's family, component B's family, weight, all preserved).
- `model_id` invalidates when weights change (mutate weights file → re-load → assert different `model_id`).

### 4.4 Phase 3 (weight fitting)

`tests/test_models/test_ensemble_weight_fit.py` (new):

- Synthetic problem: known-best `w*` recovered within 0.05 by `minimize_scalar`.
- Pinball loss formula correct: compare against an explicit reference implementation at known (actual, q_pred, q) tuples.
- Grid-search fallback: stub `loss` to return `nan` for some w; assert fallback fires and picks a finite-loss grid point.
- Edge case: identical components (`F_a == F_b`) → loss is constant in w → optimizer returns any value in (0.001, 0.999); test allows the full range.

### 4.5 Phase 4 (harness end-to-end)

- `tests/test_backtest/test_harness.py`: extend the existing smoke to call `run_backtest(model_classes=("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb", "ensemble"))` and assert the output frame has 5 distinct `model_class` values.
- Snapshot row-count assertion: prior snapshot 1504 rows; after Phase 5 snapshot 1872 rows. Phase 4's smoke uses the synthetic fixtures, not real data — it validates wiring, not numeric values.

### 4.6 Phase 5 (real-data backtest + adoption)

- `python scripts/backtest.py --model all --update-snapshot` on real data. Verify: 368 new rows; per-cell ensemble metrics readable.
- Determinism check: re-run `--check` immediately after `--update-snapshot`; expect zero drift (mirrors Plan 3d's pattern). `MixtureDistribution.quantile` is deterministic (brentq, no randomness); composite quantiles via `score_distribution` use existing per-row seeds; weight fitting is deterministic given fixed inputs.
- Build the per-cell decision table (16 cells × 4 metrics × 5 model classes) and apply §1.3 thresholds; record the verdict in `project_management.md`.

---

## 5. Phased rollout

Branch: `feat/plan-6-ensemble`. Each phase is a separate commit. Each phase closes with the standard verification: `pytest -v`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`. Plus the targeted ingest/store/schemas subset (`pytest -v -k "ingest or store or schemas"`) for any phase that touches a pandera schema.

| Phase | Scope | Net new files | Commit message preview |
|---|---|---|---|
| 0 | Add `cdf(x)` to Distribution Protocol + 5 implementations + test_cdf.py. Pure foundations; no production wiring. | 1 test file | `feat(distributions): add cdf to Distribution Protocol — Plan 6 Phase 0` |
| 1 | `MixtureDistribution` class + codec MIXTURE branch + DistributionFamily.MIXTURE + test_mixture.py + codec round-trip tests. | mixture.py, test_mixture.py | `feat(distributions): MixtureDistribution + MIXTURE codec — Plan 6 Phase 1` |
| 2 | `EnsembleModel` skeleton (fit + predict + save/load) using static weights = 0.5 each; cross-cutting test on synthetic features. | ensemble.py, test_ensemble_model.py | `feat(models): EnsembleModel scaffolding (static weights) — Plan 6 Phase 2` |
| 3 | Wire pinball weight-fitting into `EnsembleModel.fit`; persist weights JSON; tests. | test_ensemble_weight_fit.py | `feat(models): pinball weight fitting + JSON persistence — Plan 6 Phase 3` |
| 4 | POSITION_DISPATCH `"ensemble"` factories per position; backtest harness `cast` widening; CLI `--model ensemble`/`all` extension; smoke test with synthetic fixtures. | (no new files; edits) | `feat(backtest): wire ensemble into POSITION_DISPATCH + CLI — Plan 6 Phase 4` |
| 5 | Real-data backtest run; snapshot extension to 1872 rows; per-cell adoption-gate verdict. Either adopt (update CLI default + CLAUDE.md) or ship as peer. | `data/ensemble_weights/*.json`, snapshot delta | `chore(backtest): regenerate snapshot with ensemble rows — Plan 6 Phase 5` |
| 6 | `project_management.md` Plan 6 entry (verdict, per-cell table, decision); TODO #28/#29/#30 progress notes; final review. | doc edits | `docs(plan-6): record Model D verdict + tables — Plan 6 Phase 6` |

---

## 6. Backwards compatibility and risks

### 6.1 No existing rows or callers touched

- Distribution Protocol gains `cdf` additively. No existing caller invokes `cdf`, so nothing breaks.
- Codec gains a new `MIXTURE` family; v1 blobs without `MIXTURE` entries decode unchanged. `schema_version` bump to 2 is forward-compatible.
- POSITION_DISPATCH gains a new factory key `"ensemble"`. Existing `--model baseline` / `--model lightgbm` / `--model both` invocations behave identically.
- Backtest snapshot grows by 368 rows (additive). Existing 1504 rows pass through unchanged. Tolerance config `tests/backtest/tolerances.json` may need a default tolerance entry for the new ensemble cells; default tolerances already cover unknown rows.

### 6.2 Known risks

- **Per-stat pinball doesn't move composite gate.** Plan 7's Phase 0 diagnostic established that per-stat [p10, p90] coverage doesn't decompose to composite [p10, p90] coverage at the central interval. The §1.3 calibration criterion is on composite coverage, so per-stat pinball is a proxy that may fail to close the gate. Mitigation: monitor composite calibration at the verdict stage; if per-stat moves but composite doesn't, that is a clean diagnostic for a follow-up plan to optimize composite-direct via Monte Carlo.
- **Mixture variance widens too aggressively on yards stats.** Yards-stat means from A (NORMAL/GAMMA) and C-NB (QUANTILE) may differ enough that the between-means term inflates variance past the empirical residual variance. Mitigation: weight optimizer naturally pulls toward the cleaner-fit child; if calibration over-shoots 0.80 we know the mechanism at fit time, not at predict time.
- **Determinism in backtest snapshot.** `MixtureDistribution.quantile` uses brentq (deterministic); composite quantiles use existing per-row seeds; weight fitting is deterministic given fixed inputs. Re-running `--check` after `--update-snapshot` should produce zero drift. Verified explicitly in Phase 5.
- **Compute cost.** ~5–10 minutes incremental per backtest run (4 child fits per fold instead of 2, plus weight optimization). Bounded by cached tuned hyperparameters; not a concern.

### 6.3 Out of scope (re-confirmed)

- Per-row gated weights (meta-learner mapping features → weights).
- N>2 components.
- Composite-direct weight optimization via Monte Carlo.
- Aggregation-to-season widening (TODO #28).
- Pruning Model C-tuned (TODO #29).
- Targeted upper-tail count calibration (TODO #30).

---

## 7. Success criteria

This plan is successful if **either** of the following lands on `main`:

1. **Adoption case.** Model D ensemble passes all three §1.3 criteria; `scripts/train_baseline.py` default updates to `ensemble`; CLAUDE.md and `project_management.md` document the cutover; A / C / C-tuned / C-NB stay registered as peers for backtest comparability.
2. **Peer case.** Model D ensemble fails one or more §1.3 criteria; ships as peer alongside A / C / C-tuned / C-NB; A stays default. Per-cell metrics + verdict + the loss-decomposition (per-stat pinball moved? composite calibration moved?) are recorded in `project_management.md`. The plan is not "stopped" — the experimental result is the deliverable. Three follow-up directions are filed for evaluation: (i) composite-direct weight optimization; (ii) per-row gated weights; (iii) accept calibration shortfall and pivot to feature-class tracks (TODO #3, #23) or Plan 4 (CLI / API).

The infrastructure (MixtureDistribution, codec MIXTURE branch, EnsembleModel, weights persistence, harness wiring) ships in either case as reusable foundation for any future plan that wants to compose multiple model classes.
