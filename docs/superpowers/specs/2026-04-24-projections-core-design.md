# Projections Core — Design

**Status:** approved (brainstorming)
**Date:** 2026-04-24
**Author:** alden + claude
**Sub-project of:** FantasyFootball (parent project encompasses Projections Core, Season-long Draft Hub, Mid-season Manager, DFS Engine — see TODO.md for parent decomposition)

---

## 1. Overview

Projections Core is the foundational sub-project of FantasyFootball. It produces per-player, per-week probability distributions over fantasy points, and aggregates them into season-long distributions. It is the single source of truth that downstream sub-projects (Draft Hub, Mid-season Manager, DFS Engine) consume.

### 1.1 Goals

- For every active NFL player at QB / RB / WR / TE / K / DST, produce a **distribution** over fantasy points for the upcoming week and any week within the current season.
- Produce **rest-of-season** and **full-season** distributions derived from weekly outputs, with proper handling of bye weeks and player availability (not naive sum-of-means).
- Stay **scoring-ruleset-agnostic**: re-score historical and current projections under arbitrary scoring without retraining.
- Provide a **first-class backtest harness** that gates model changes — we never claim a model is better without held-out evidence.
- Use a storage format that is friendly to free-tier web hosting (Streamlit Community Cloud, Hugging Face Spaces, GitHub Pages + DuckDB-WASM).

### 1.2 Non-goals (sub-projects elsewhere)

- Auction values, VORP, draft strategy, ADP analysis — Draft Hub.
- Lineup optimization, ownership projections, slate-aware exposure — DFS Engine.
- Waiver-wire ranking, trade analyzer, weekly start/sit UI — Mid-season Manager.
- Joint correlations between players' outcomes — deferred (TODO #1, "option D").
- Web UI — separate sub-project. Storage choices keep this cheap.
- Live in-game projection updates.
- ESPN/Sleeper league integration. Projections Core is platform-agnostic; ESPN integration belongs in Draft Hub and Mid-season Manager.

### 1.3 Modeling roadmap

We commit to an A → C → D progression:

- **Model A (v1):** position-specific regression baseline with parametric residual variance. Goal: usable system in days, debuggable, establishes the data pipeline / scoring layer / schema / backtest harness — all of which we need regardless.
- **Model C (v2):** gradient-boosted trees (LightGBM/XGBoost) with quantile regression for distribution. Plugs into the same Model interface as A. Adopted only if it beats A on the backtest harness.
- **Model D (v3):** ensemble / stack across A, C, and optionally external consensus projections. Last resort once we have something to ensemble.

Joint correlations across players (true "option D" from brainstorming) live in TODO #1 and are explicitly out of scope here. The schema is designed so per-player marginals stay valid and joint structure is an additive upgrade.

---

## 2. Architecture

```
projections/
├── ingest/           # nfl_data_py wrapper + caching; raw → tidy parquet
├── features/         # per-position feature builders (rolling usage, opp-adj, Vegas)
├── scoring/          # parameterized scoring rules (PPR/half/std/custom)
├── models/
│   ├── base.py       # Model interface — fit, predict_distribution, save/load
│   ├── baseline.py   # Model A: position-specific regressions w/ residual variance
│   └── (future)      # gbm.py for C, ensemble.py for D
├── distributions/    # Distribution objects: .mean(), .quantile(q), .sample(n)
├── aggregate/        # weekly → ROS / season w/ bye + availability adjustment
├── backtest/         # held-out evaluation harness; metric registry
├── store/            # parquet writers/readers + DuckDB views
├── schemas.py        # single source of truth: pandera + pydantic + enums
└── api/              # Python entry points + thin CLI
```

### 2.1 Module boundaries

| Module | Owns | Exposes | Constraints |
|---|---|---|---|
| `ingest` | network I/O against `nfl_data_py`; raw parquet writes; ID translation table | `refresh(seasons=...)`, raw schemas | **Only** module that hits the network. Downstream code never imports `nfl_data_py`. Idempotent. |
| `features` | per-position feature builders | `build(raw_df, as_of_week, season) -> feature_df` | Pure functions. No I/O, no models. Leakage-prevention enforced by signature: features cannot see rows on/after `as_of_week`. |
| `scoring` | parameterized fantasy-point math | `score(stat_line, ruleset)`, `score_distribution(stat_dist, ruleset)` | Pure, table-driven, fully unit-tested. **Only** module that knows what counts as a fantasy point. |
| `models` | Model interface + implementations | `Model.fit(...)`, `Model.predict_distribution(...)`, `save/load` | Downstream depends on the *interface*, not implementations. Lets us swap A → C → D without consumer changes. |
| `distributions` | Distribution value object | `.mean()`, `.std()`, `.quantile(q)`, `.sample(n)`, `.score_under(ruleset)` | Same interface regardless of backing (parametric, quantile, sampled). |
| `aggregate` | weekly → ROS/season aggregation | `aggregate_to_season(weekly_dists, schedule, availability)` | Monte Carlo, not naive sum. Handles byes and per-player availability. |
| `backtest` | held-out evaluation harness | `walk_forward(model, seasons, metrics)` | First-class. CI gate on model changes. |
| `store` | parquet writes/reads, DuckDB views | typed reader/writer functions per table | **Only** module that writes to disk. Everything else returns DataFrames or Distribution objects. |
| `schemas` | canonical schemas, enums, NewTypes | `pandera` schemas, `pydantic` models, `Position` / `Team` / `DistributionFamily` enums, `GsisId` / `EspnId` / etc. NewTypes | Single source of truth. Imported everywhere, owned nowhere else. |
| `api` | public Python + CLI surface | `project_week`, `project_season`, `refresh_data`, CLI verbs | Thin shell. No business logic. |

### 2.2 Key principle

`Model` and `Distribution` are interfaces. Model A is the first implementation; C and D plug in at the same seam. Same for `Distribution`: parametric backing for v1, quantile-based for C, sampled for D.

---

## 3. Data → Projection pipeline

### 3.1 Ingest

- `ingest.refresh(seasons=[...])` calls `nfl_data_py` for: weekly player stats, schedules, rosters, snap counts, depth charts, NGS where available, and team-level weekly stats (used for K and DST).
- Normalizes to canonical schema. Canonical player ID is `GsisId` (`nfl_data_py`'s `player_id`); a translation table maps to `EspnId`, `SleeperId`, `PfrId`.
- Writes to `data/raw/{table}/season=YYYY/part.parquet`. Idempotent: re-running a season overwrites that partition only.
- Maintains `data/manifests/ingest_manifest.parquet` with `(source, table, season, fetched_at, rowcount, checksum)` so we know what's stale.

### 3.2 Feature engineering (per-position)

Each position has its own builder so we can iterate independently. Common backbone: rolling 4-week and season-to-date usage rates, opponent-adjusted EPA/play, Vegas implied team total + spread.

| Position | Feature highlights |
|---|---|
| QB | pass attempts/game, aDOT, sack rate, rush attempts, opp pass-defense EPA/play, implied total |
| RB | snap %, rush att/game, target share, redzone touches, opp run-defense EPA, opp adjusted line yards, implied total |
| WR / TE | route %, target share, aDOT, redzone targets, slot %, opp coverage EPA by alignment, implied total |
| K | team implied total, dome/weather, opp redzone TD allowed % (proxy for FG opportunities), recent FG distance distribution |
| DST | opp implied total (negated), opp pass-block win rate / sack rate allowed, opp turnover-worthy throw rate, home/away |

All builders are pure: `(raw_df, as_of_week, season) -> feature_df`. Leakage prevention is enforced by the function signature — the builder cannot see any row whose `(season, week)` is on/after `as_of_week`. Unit tests assert this.

### 3.3 Scoring engine

- `scoring.score(stat_line: StatLine, ruleset: Ruleset) -> float`
- `scoring.score_distribution(stat_dist: StatDistribution, ruleset: Ruleset) -> Distribution`
- `Ruleset` is a `pydantic.BaseModel`: `passing_yd_per_pt`, `passing_td_pts`, `rec_pt`, `bonus_300_pass_yds`, etc. Defaults match ESPN standard PPR. Custom rulesets override individual fields.
- Lives outside the model layer so we can re-score historical projections under different rules without retraining.
- Built-in named rulesets: `Ruleset.ESPN_PPR`, `Ruleset.ESPN_HALF`, `Ruleset.STANDARD`. Easy to add.

### 3.4 Model A (baseline)

- One regression per position. **Target = underlying stats** (yards, TDs, receptions, etc.), not fantasy points directly. The scoring layer converts. This makes the model rule-agnostic and lets us re-score without refitting.
- Fit positional residual variance from training-set residuals; combined with point estimate, this yields a parametric `Distribution` per player. Distribution family is per-position: gamma for non-negative volume stats, normal for net yardage, etc.
- Persisted via `joblib` to `models/artifacts/baseline-{position}-{trained_through}.joblib`. A model registry tracks `(model_class, code_hash, training_window, ruleset)` → artifact, so backtests are reproducible.

### 3.5 Distribution layer

- `Distribution` interface: `.mean()`, `.std()`, `.quantile(q)`, `.sample(n, rng=None)`, `.score_under(ruleset)`.
- v1 backing: parametric `(family: DistributionFamily, params: dict)`. Same interface will back quantile-regression outputs (Model C) and sampled scenarios (Model D, when joint sampling is layered on top per TODO #1).
- Distributions are first-class storable: serialized to parquet as `(family, params)` columns plus pre-computed `mean / p10 / p50 / p90` for fast filtering without rehydrating.

### 3.6 Aggregation (weekly → ROS / season)

- Not naive sum-of-means. Monte Carlo: for each player, draw N samples per future week, apply a per-player weekly availability factor (1 − injury-game probability, derived from positional base rates × player injury history), zero out bye weeks, sum across the horizon.
- Output is itself a `Distribution`, so season-long projections support `.mean()`, `.quantile(0.1)`, etc., for downstream VORP / ceiling / floor analysis.
- v1 availability model: positional base rate (computed from historical games-missed rates) × recent-history multiplier. Improving it is a future TODO.

---

## 4. Storage

### 4.1 Layout

```
data/
├── raw/                              # ingested from nfl_data_py, untouched
│   ├── weekly_stats/season=YYYY/part.parquet
│   ├── schedules/season=YYYY/part.parquet
│   ├── snap_counts/season=YYYY/part.parquet
│   ├── depth_charts/season=YYYY/part.parquet
│   ├── ngs/season=YYYY/part.parquet
│   └── id_map.parquet                # gsis_id ↔ espn_id ↔ sleeper_id ↔ pfr_id
├── features/                         # leakage-safe feature snapshots
│   └── {position}/season=YYYY/week=WW/part.parquet
├── projections/                      # the published outputs
│   ├── weekly/season=YYYY/week=WW/ruleset=NAME/part.parquet
│   └── season/season=YYYY/as_of_week=WW/ruleset=NAME/part.parquet
├── backtests/
│   └── {model_id}/{run_id}/metrics.parquet + predictions.parquet
└── manifests/
    └── ingest_manifest.parquet       # what was fetched when, with checksums
```

### 4.2 Projection schema (the published contract)

```
gsis_id          GsisId
season           int (year)
week             int (1–22; 18 regular + playoffs)
position         Position (enum)
team             Team (enum)
opponent         Team (enum)
ruleset          str (ruleset name)
family           DistributionFamily (enum)
params           bytes (msgpack-encoded params for the family)
mean             float
p10              float
p50              float
p90              float
model_id         str (model_class:code_hash:training_window)
generated_at     timestamp UTC
```

Pre-computed `mean / p10 / p50 / p90` columns let dashboards filter and rank without rehydrating distributions. The `family + params` blob is there for consumers that need full distribution math.

### 4.3 DuckDB view layer

A small `projections.duckdb` view file stitches the partitioned parquet into one queryable surface (`projections.weekly`, `projections.season`, `projections.id_map`). The CLI's `query` verb wraps this. Same view file works locally and from DuckDB-WASM in browser.

### 4.4 Free-tier hosting friendliness

- A full season of all-position weekly projections is a few MB compressed parquet. No tier-busting file sizes.
- Streamlit Community Cloud / Hugging Face Spaces read parquet directly.
- DuckDB-WASM enables a fully static GitHub Pages site that queries parquet in the user's browser — zero backend cost.
- A GitHub Actions workflow can refresh data on a cron and commit updated parquet to a `data` branch; web apps consume from raw.githubusercontent.com.

---

## 5. Public API

### 5.1 Python

```python
from projections import (
    project_week, project_season,
    refresh_data, list_models,
)
from projections.schemas import Ruleset, Distribution, Position, GsisId

# Canonical (programmatic) form — pass IDs.
dist: Distribution = project_week(
    player_id=GsisId("00-0036322"),
    season=2026,
    week=3,
    ruleset=Ruleset.ESPN_PPR,
)

# Convenience form — name resolution via id_map for CLI / notebook use.
dist = project_week(player="Justin Jefferson", season=2026, week=3, ruleset=Ruleset.ESPN_PPR)

dist.mean(); dist.quantile(0.9); dist.sample(1000)
```

ID handling:
- All internal storage and joins use `GsisId`. Always.
- The API accepts `player=<name>` as syntactic sugar; resolves via `id_map.parquet`.
- Ambiguous names (e.g., "Mike Williams") raise `AmbiguousPlayerError` with a candidate list including position and team.
- Downstream sub-projects (Draft Hub, DFS Engine) always pass IDs — they get them from rosters / slates anyway.

### 5.2 CLI

Thin shell over the Python API; same verbs:

```
python -m projections refresh   --seasons 2020-2026
python -m projections project   --season 2026 --week 3 --ruleset espn_ppr
python -m projections backtest  --model baseline --seasons 2020-2024 --ruleset espn_ppr
python -m projections query     "SELECT * FROM projections.weekly WHERE position='WR' ORDER BY mean DESC LIMIT 30"
```

That's the entire surface area. Downstream tools consume parquet directly or import the API; they never reach into internals.

---

## 6. Backtest harness

First-class, not an afterthought. Backtest is the only way we know if A → C → D is actually winning.

- **Walk-forward.** Train on seasons through Y−1, predict every week of season Y, never look ahead. Repeat for each held-out season.
- **Metric registry**, plug-in style:
  - Point: RMSE, MAE, top-N rank correlation (Spearman against actual top scorers per position).
  - Distribution: pinball loss at p10/p50/p90, calibration (does p90 actually contain 90% of outcomes?), CRPS.
  - Decision-relevant: "would this pick win you the week" — start/sit hit rate at common league depths.
- **CI gate.** Any change to `models/` or `features/` requires a backtest run; metrics get committed to `data/backtests/` and we diff against the previous best. No regressions merge silently.
- **Reproducibility.** `model_id = (model_class, code_hash, training_window, ruleset)`. Every prediction in storage carries it; every backtest run is keyed by it.

---

## 7. Testing strategy

- **Unit tests** for `scoring/` (table-driven; every rule × edge case), `features/` (leakage tests — feature builder with `as_of_week=W` must not change if rows past W are mutated), `distributions/` (mean/quantile/sample contracts), `aggregate/` (bye handling, availability adjustment).
- **Integration tests** for `ingest/` against a small fixture parquet (don't hit network in CI), and an end-to-end smoke test that runs `refresh → features → project → query` against fixtures and checks a known projection.
- **Backtest as test.** `pytest -k backtest_baseline` runs a tiny held-out year and asserts metric thresholds — catches model regressions in CI.
- **Type-check** as a CI step (`mypy --strict` on non-DataFrame code, `pandera` schemas validated at module boundaries).

---

## 8. Typing posture

Strong typing is non-negotiable. Hard rules:

- **Every DataFrame that crosses a module boundary has a `pandera` schema.** Schemas live in `projections/schemas.py` (single source of truth). Public functions are decorated with `@pa.check_io` so violations fail loudly at the boundary, not three modules deep.
- **No raw dicts at module boundaries.** Configs and records are `pydantic.BaseModel` (when validated/serialized) or `@dataclass(slots=True, frozen=True)` (when internal). Typed: `Ruleset`, `Distribution`, `ModelConfig`, `IngestManifest`, `BacktestResult`.
- **`NewType` for every ID flavor.** `GsisId`, `EspnId`, `SleeperId`, `PfrId` are distinct types — passing one where another is expected is a `mypy` error. The `id_map` module is the *only* place conversions happen.
- **Enums for every reused string-keyed concept.** Always reference the constant, never the string literal:
  - `Position`: `QB`, `RB`, `WR`, `TE`, `K`, `DST`. (Future-extensible to `IDP_DL`, `IDP_LB`, `IDP_DB` if a league requires it.)
  - `RosterSlot`: `QB`, `RB`, `WR`, `TE`, `FLEX`, `SUPER_FLEX`, `K`, `DST`, `BENCH`, `IR`. (`SUPER_FLEX` exists from day 1 even though current league doesn't use it — see TODO; superflex-readiness was explicit.)
  - `Team`: 32-team enum. Eliminates `JAX` vs `JAC`, `LA` vs `LAR` confusion. Translation table inside the enum maps known aliases to the canonical code.
  - `DistributionFamily`: `NORMAL`, `GAMMA`, `EMPIRICAL_QUANTILE`, `SAMPLED`.
  - `Ruleset` (named presets): `ESPN_PPR`, `ESPN_HALF`, `STANDARD`. Custom rulesets are still `Ruleset` instances; the enum is a registry of named defaults.
  - `Stat` (column-name enum) for stats referenced in scoring rules and feature builders, so misspellings are caught at type-check time.
- **`mypy --strict` in CI** for everything that isn't a DataFrame. Failing types fail the build.
- **`from __future__ import annotations`** in every module. `Final` for module constants. `Literal` and exhaustive `match` on enums where it helps.
- **No `Any` without a comment explaining why.** Linted via `mypy` config.

What stays untyped: intermediate DataFrames inside a single module's private functions. The boundary contract is what matters; we're not going to schema-validate every step of feature engineering.

Pinned tooling (in `pyproject.toml` from day 1): `pandera`, `pydantic>=2`, `mypy`, `ruff`.

---

## 9. Out of scope (explicit)

- Live in-game projection updates.
- Auction values, VORP, draft strategy — Draft Hub sub-project.
- Lineup optimization, ownership — DFS Engine sub-project.
- Joint correlations across players — TODO #1.
- A web UI — separate sub-project. Storage choices keep this cheap when we get there.
- ESPN/Sleeper league API integration — belongs in Draft Hub / Mid-season Manager.

---

## 10. Open questions / future work

Tracked in `TODO.md` at the repo root. Specifically:

- **TODO #0:** pick lint/format config (`ruff` rules, line length, import order), `pyproject.toml` layout, pre-commit hooks, write `CONTRIBUTING.md`.
- **TODO #1:** explore option D (joint-correlation projections). Approach, storage, optimizer interface, validation plan.

---

## 11. What an MVP delivers

In order:

1. `schemas.py` with all enums, NewTypes, pandera schemas, pydantic models.
2. `ingest.refresh(seasons=[2014..2025])` writes all raw partitions; `id_map.parquet` populated.
3. `scoring.score(...)` and `scoring.score_distribution(...)` for ESPN PPR + standard rulesets, with full unit-test coverage.
4. `features.build(...)` for QB / RB / WR / TE / K / DST.
5. Model A trained per position; predictions written to `data/projections/weekly/...` for the upcoming season.
6. `aggregate.to_season(...)` produces a season distribution per player.
7. `backtest.walk_forward(...)` runs on 2020–2024 and writes metrics; CI asserts threshold.
8. Python API + CLI exposing `project_week`, `project_season`, `refresh`, `backtest`, `query`.

Anything beyond this — Model C, joint correlations, web UI, draft hub — is a separate spec.
