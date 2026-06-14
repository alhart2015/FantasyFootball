# Per-player weekly performance variance model — Design

**Date:** 2026-06-14
**Status:** Approved (brainstorming) → spec review
**Topic:** A fitted weekly fantasy-point variance model, wired into (A) the season-value Monte-Carlo valuation and (B) a predictive league-sim mode for honest post-draft CIs. Implements TODO #45.

## Problem

The season-value Monte-Carlo (`src/projections/draft/assistant/season_value.py`) models only **availability** variance: each simulated week a player is available (Bernoulli `p`) or out (injury/bye), and when available they score a **deterministic** `per_game = season_mean_fpts / 17`. There is no week-to-week or season-level performance variance — an available player scores the exact same points every simulated week. Two consequences:

1. **Valuation is risk-blind.** A boom/bust player and a metronome with the same projected mean value identically, even though weekly lineup-setting rewards upside (you start your best each week) and punishes bust weeks. The MC can't see this.
2. **Post-draft CIs are too tight.** The H2H backtest resamples only draft order + schedule; each player's weekly actuals are a single fixed historical realization. The reported CIs (`reports/post_draft_assessment_2021_2025.md`) reflect draft/schedule luck only — per-season champ% bootstrap CIs are ±2–3% while the *actual* season-to-season spread is 5–18%.

## Evidence (from 2018–2025 weekly_stats, half-PPR, active games; see brainstorming analysis)

These findings parameterize and justify the model:

- **Sample sizes:** fantasy-relevant players with ≥17 weekly games: QB 79 / RB 181 / WR 288 / TE 148; ≥34: QB 52 / RB 128 / WR 181 / TE 90. Many draftable players have ~1 season; rookies have **zero**. → pure per-player variance is not estimable for most of the pool.
- **Position matters:** median weekly CV QB **0.56**, RB **0.71**, WR **0.75**, TE **0.72**. QBs are genuinely steadier.
- **Variance scales with the mean (within position):** `corr(mean, CV) = −0.80`; `std ≈ 0.25·mean + 4.4` (pooled). A flat per-position CV is wrong; variance must be a function of the projected mean.
- **Per-player variance is a weak signal:** split-half (≥30 games) `corr(CV_A, CV_B) = 0.69` vs `corr(mean_A, mean_B) = 0.94`. Real but noisy, and only for data-rich players → per-player shrinkage is **deferred** (not v1).
- **Rookies are ~1.5× less predictable vs projection:** realized/projected per-game SD **0.58 (rookie)** vs **0.39 (veteran)** (P10–P90 0.54–1.89 vs 0.65–1.54). Their *weekly* CV is only mildly higher (0.70 vs 0.64); it's their *projection* that's shaky. → rookies need an inflated season-mean-uncertainty component, not a different weekly shape.

## Goals

1. A fitted, position-aware **weekly performance variance model** with two components (below), parameterized entirely by `(position, projected_mean, is_rookie)` so it covers every player including rookies.
2. **Consumer A:** make the season-value MC risk-aware by sampling weekly points from the model instead of using deterministic `per_game`, while staying live-draft-fast and preserving CRN.
3. **Consumer B:** a predictive league-sim mode that scores on model-sampled outcomes to produce **honest forward post-draft CIs**, complementing (not replacing) the real-outcome historical backtest.
4. Fit the model offline from 2018–2025 and commit the parameters; no runtime fitting.

## Non-goals (explicitly out of scope)

- **Per-player shrinkage** (empirical-Bayes blend of a veteran's own historical variance toward the position model). The 0.69 reliability makes it a real but secondary refinement → a later slice.
- **Same-game correlation** (QB↔pass-catcher stacks, shootouts) — TODO #1 option D.
- **Playoff-week weighting**, **draft-capital rookie refinement**, **recency/age-weighted variance**.
- **Replacing the historical real-outcome backtest.** Consumer B is an additional predictive mode; the existing real-actuals backtest and its point estimates stay.

## Chosen approach

### The two-component model

Each simulated season, for each player, draw in two stages:

1. **Season-mean uncertainty (component ii):** a per-season multiplier `m ~ Lognormal(median = 1, σ = σ_pos_rookie)` giving `true_mean = projected_mean × m`. `σ_pos_rookie` is fit per position × rookie-flag (≈ 0.39 veteran / 0.58 rookie pooled). This does **not** average out over a season — it is what fattens season-outcome tails / fixes the CIs.
2. **Within-season weekly noise (component i):** for each available week, draw points `~ Gamma(mean = true_mean / GAMES, std = f_pos(true_mean) / GAMES)` where `f_pos(mean) = a_pos · mean + b_pos` (per-position affine, fit from data). Gamma is non-negative and right-skewed, and self-adjusts (near-normal for low-CV QBs, skewed for high-CV WRs). This averages out over a season but drives weekly lineup/matchup variance.

Availability is unchanged and applied on top (a player out that week scores 0 regardless). The performance distribution is therefore **conditional on playing** (it is fit on active games only).

`GAMES = 17` (matches the existing `_GAMES` uniform-scaling convention in `season_value.py`).

### Rookie flag

`is_rookie` is derived from NFL history: a player with **no prior-season `weekly_stats` appearance** is a rookie. At draft time the incoming class has no prior history; in the backtest/predictive sim, detection uses the multi-season `weekly_stats` (first appearance == this season). The draft-basis pool must carry an `is_rookie` column (new requirement R6). A player whose rookie status can't be determined defaults to **veteran** (the conservative, narrower σ — documented).

### Fitting (offline → committed constants)

`scripts/fit_performance_variance.py` reads `weekly_stats` 2018–2025 + the blended external-projection snapshots and fits:
- `a_pos, b_pos` per position (regress per-player-season weekly `std` on `mean`, active games),
- `σ_pos_rookie` per position × rookie-flag (SD of `log(realized_pg / projected_pg)` for the lognormal), with a documented fallback to the pooled 0.39/0.58 when a position×rookie cell is too thin (e.g. rookie TE/QB).

Output: a committed params artifact `configs/performance_variance_params.json` (position → `{a, b}`; (position, is_rookie) → `σ`). The model module loads it; **no fitting at runtime.** The fitter is re-runnable and doubles as a regression guard (refit should be stable season-to-season).

### Components / file structure

- `scripts/fit_performance_variance.py` — offline fitter → params JSON. Committed, re-runnable.
- `configs/performance_variance_params.json` — fitted params. Version-controlled.
- `src/projections/draft/assistant/performance_variance.py` — the model: load params; pure **vectorized** samplers. Primary entry: given `(positions, projected_means, is_rookie, rng, n_sims, n_weeks)` return a `(n_sims, n_weeks, n_players)` array of sampled weekly points (component ii drawn per `(sim, player)`, component i per `(sim, week, player)`). Parameterized by the loaded params; no I/O beyond the one-time param load.
- **Consumer A** — `season_value.py`: replace the deterministic `per_game` fill in `_vectorized_lineup_points` / `expected_season_points*` / `marginal_season_values` with model-sampled weekly points. The current fast-path factorizes weeks because `per_game` is uniform; per-week sampled points break that factorization, so the fill must run per `(sim, week)` over the sampled-points array. CRN/antithetic structure (shared draws across base and candidate rosters in `marginal_season_values`) is preserved by sharing the rng draws.
- **Consumer B** — a predictive mode of `simulate_league` (`league.py`) that, instead of a fixed `actual_lookup`, scores each (sim, team, week) on model-sampled weekly points; aggregated over many sims → forward champ%/playoff%/wins distributions. `scripts/post_draft_assessment.py` gains a predictive-CI readout distinct from the historical-actuals tables.

## Requirements

R1. `fit_performance_variance.py` fits and writes `configs/performance_variance_params.json` with: per-position `a_pos, b_pos` (weekly `std = a·mean + b`) and per-(position, is_rookie) `σ` (lognormal log-SD of realized/projected per-game), with the pooled-fallback rule for thin cells. Re-running on the same data is deterministic.
R2. `performance_variance.py` loads the params and provides a vectorized sampler returning `(n_sims, n_weeks, n_players)` non-negative weekly points: component ii `m ~ Lognormal(median 1, σ_pos_rookie)` per (sim, player); component i `~ Gamma(mean = true_mean/GAMES, std = f_pos(true_mean)/GAMES)` per (sim, week, player). Given a fixed seed, output is deterministic. `projected_mean ≤ 0` → all-zero (no draw).
R3. **Consumer A:** `season_value.py`'s expected-season-points and `marginal_season_values` use the sampler instead of deterministic `per_game`. Availability still applied (out → 0 that week). `marginal_season_values` keeps CRN: the same `m` and weekly draws are shared between the base roster and every candidate so marginals stay low-variance. Determinism preserved under a fixed `base_seed`.
R4. **Consumer A speed:** a single pick recommendation at the default `n_sims` stays within the live-draft budget — **≤ 250 ms/pick** at `n_sims=200` on the dev box (today it's ~20–35 ms; the per-week sampling adds the ~17× week dimension, so the fast-path must be re-vectorized, not naively looped).
R5. **Consumer B:** a predictive league-sim path scores on model-sampled weekly points (parameterized by a seed + n_sims) and yields per-team champ%/playoff%/expected-wins distributions. It does **not** modify or replace the existing real-actuals `simulate_league` results. `scripts/post_draft_assessment.py` reports the predictive forward CI alongside the historical tables.
R6. The draft-basis pool (`build_draft_basis` output / the backtest pool) carries an `is_rookie` flag derived from `weekly_stats` history (no prior-season appearance), defaulting to veteran when undeterminable.
R7. **Validation (definition of done):**
   - (a) Fitting reproduces the observed structure: per-position weekly CV within ±0.05 of the measured medians (QB 0.56 / RB 0.71 / WR 0.75 / TE 0.72) and the realized/projected per-game SD within ±0.05 of 0.39 (veteran) / 0.58 (rookie), on held-back/refit data.
   - (b) **Consumer A** does not regress per-pick speed past R4, and the strategy ranking stays sane (season_value remains competitive — does not collapse vs `now_or_never`/bots on the existing H2H real-outcome backtest).
   - (c) **Consumer B**: the predictive forward champ%/playoff%/wins CIs for a season_value draft span ≈ the historical cross-season spread (champ ≈ 5–18%, playoff ≈ 50–78%, wins ≈ 7.7–9.2) — i.e. materially wider than the current draft/schedule-only bootstrap (±2–3% champ), confirming the CIs now reflect player-outcome luck.

## Edge cases / failure modes

- **`projected_mean ≤ 0`** (ADP-only / replacement players in the pool): sampler returns all-zero weekly points; they never start. No draw, no error.
- **Gamma parameterization at low mean:** `std = f_pos(mean)` has a positive floor `b_pos`, so a low-mean player still gets a sensible non-degenerate Gamma; guard `std > 0` and `mean > 0` before computing Gamma shape/scale, else return the deterministic mean.
- **Thin rookie cells** (rookie QB/TE have few historical player-seasons): `σ` falls back to the pooled rookie value (documented in R1).
- **Unknown rookie status:** default veteran (narrower σ) — conservative; documented.
- **Availability interaction:** variance is conditional on playing; an unavailable week is 0 *before* any performance draw is applied (no double-counting of "out" weeks into the performance distribution, which is fit on active games).
- **CRN integrity (A):** if base and candidate rosters drew independent `m`/weekly noise, marginals would be swamped by sampling noise. The shared-draw requirement (R3) is the guard; a test pins that the same seed gives identical base-roster value across candidate evaluations.
- **Determinism:** all sampling is seeded; same seed → same result (pinned by tests), so the backtest/predictive runs are reproducible and the chunked runner stays resumable.

## Testing expectations

TDD; each unit gets a failing test first; synthetic params/inputs (no network).
- **Fitter:** on a synthetic weekly+projection fixture with known variance structure, recovers `a_pos/b_pos` and `σ` close to ground truth; thin-cell fallback fires; output JSON schema/keys correct; deterministic.
- **Sampler:** seeded determinism; output shape `(n_sims, n_weeks, n_players)`; non-negativity; recovers target mean and std in expectation (large-n_sims sampled mean ≈ projected_mean, sampled CV ≈ position target); `projected_mean ≤ 0 → 0`; rookie σ wider than veteran for the same inputs.
- **Consumer A:** sampled MC reduces to ~the deterministic result when variance params are set to ~0 (degenerate check); CRN — same seed yields identical base value across candidates; marginal_season_values still ranks a clearly-better candidate first; speed micro-benchmark within R4.
- **Consumer B:** predictive sim with ~0 variance params ≈ a single deterministic-outcome league; with real params, per-team outcome distributions have non-trivial spread; the historical real-actuals path is byte-identical (unchanged).
- Run the CLAUDE.md gates each phase (`pytest`, `mypy src tests`, `ruff check`, `ruff format --check`).

## Phasing

Three phases, each independently testable and committable:

1. **Model + fitter + params** (R1, R2, R6) — the shared core. Deliverable: `fit_performance_variance.py`, `configs/performance_variance_params.json`, `performance_variance.py`, `is_rookie` on the pool, with unit tests. No consumer wired yet.
2. **Consumer A** (R3, R4, R7b) — risk-aware season_value MC, CRN-preserving, speed-gated.
3. **Consumer B** (R5, R7c) — predictive league-sim mode + post-draft-assessment forward-CI readout.

Validation R7 spans the phases (a in Phase 1, b in Phase 2, c in Phase 3). Per-player shrinkage and same-game correlation remain deferred follow-ons.
