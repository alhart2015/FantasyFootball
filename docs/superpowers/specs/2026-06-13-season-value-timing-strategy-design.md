# Season-Value Timing Strategy — Design

**Date:** 2026-06-13
**Status:** Design (pre-plan)
**Branch:** `feat/season-value-timing-strategy`

## 1. Motivation

`SeasonValueStrategy` is the strongest draft strategy in the test series — it beats `now_or_never` on H2H win% and playoff-make% by ~+6 / ~+16 points, replicated across 2024 and 2025 (Tests 7–8, `reports/draft_strategy_tests.md`). Its one proven weakness is **myopia**: it ranks candidates purely by the marginal expected season points they add to the current roster, with **no pick-timing signal**. Earlier validation showed this costs it the long-wait seat (it lost slot 6 to `now_or_never`, whose opportunity-cost layer dominates when your next pick is far away).

This strategy adds the missing layer: take a player now when comparable season-value won't survive to your next pick, defer when it will — `now_or_never`'s opportunity-cost idea expressed in season-value units instead of VORP.

**This is one more candidate strategy to build, run through the H2H harness, and log.** Per the standing process rule (`reports/draft_strategy_tests.md`), no adopt/reject decision is made here — the single decision across all strategies and seasons comes at the end of the investigation.

## 2. Goal & non-goals

**Goal:** a new `DraftStrategy` (`season_value_timing`) that ranks by `marginal_season_value − opportunity_cost`, where opportunity cost is the expected best surviving marginal at the candidate's position by the manager's next pick. Fast enough to run live (no added Monte-Carlo). Selectable in the live assistant CLI and testable in the H2H harness against the existing strategies.

**Non-goals (deliberate deferrals):**
- **No change to the value function.** Still `expected_season_points` (depth + per-player availability + byes, `season_mean_fpts / 17` flat per week). Variance/ceiling-aware scoring (which needs per-player distributions we don't have) is a separate lever.
- **No change to the projection input.** Still preseason `season_mean_fpts` (the known optimistic bias is TODO #42's concern, shared with the nn fix).
- **No replacement of `season_value`.** It stays the untouched control; this is A/B alongside it.
- **No multi-pick lookahead** (the opportunity cost is single-pick, against the current roster — same approximation level as `now_or_never`).

## 3. The strategy — `SeasonValueTimingStrategy`

A frozen dataclass behind the existing `DraftStrategy` protocol (CLI key `season_value_timing`), in `src/projections/draft/assistant/strategy.py`. It holds the union of `season_value`'s and `now_or_never`'s config:

```
SeasonValueTimingStrategy(
    availability: PlayerAvailability,
    n_sims: int,
    base_seed: int,
    survival: SurvivalModel,
    top_k: int = 8,
)
```

`__post_init__` validates `n_sims >= 1` and `top_k >= 1` (as `SeasonValueStrategy` does).

**`recommend(state, pool, config) -> RecommendationSchema`:**

1. **Marginals (identical to `season_value`, the only MC):** `df, elig = _eligible_subset(...)`; build `base_roster` from `state.my_pick_ids` (warn on pool-absent rostered ids, as today); prune to `top_k`-by-VORP per position; `marginals = marginal_season_values(base_roster, pruned, config.roster_slots, availability, n_sims, rng)` with `rng = default_rng([base_seed, state.current_pick])`. Per-pick cost ≈ today's `season_value` → live-draft-fast.
2. **Timing layer (no new MC):** `next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)`.
   - **`next_pick is None` (last pick):** rank by raw marginal — i.e. exactly today's `season_value` (`score = marginal`, `_finalize(starting_need_tier=False)`). Mirrors nn's raw-VORP fallback.
   - **Otherwise:** per pruned candidate, survival prob `p = survival.p_available(adp, next_pick)` where `adp` is the candidate's `consensus_adp` column (null ADP → `p = 1.0`, nn's convention). Compute, per position, the **expected best surviving marginal** via the shared helper (§4):
     `opp_cost[pos] = E[ max marginal among pos candidates that survive to next_pick ]`.
3. **Score:** `score(c) = marginal(c) − opp_cost[pos(c)]`. Pruned-out candidates keep `marginal = 0` (so their score is `−opp_cost[pos]`) — cosmetic tail; the recommended pick (argmax) is always an evaluated candidate, exactly as in `season_value`. Top-k pruning keeps ≥1 candidate per eligible position (`top_k ≥ 1` is validated), so `opp_cost` has a key for **every** position in the eligible pool — the `opp_cost[pos(c)]` lookup is always defined, including for the tail (no defensive `.get`, no `KeyError`).
4. `_finalize(out, elig, display_p, starting_need_tier=False)` — the marginal already values open starting slots, so no `fills_starting_slot` tier (same as `season_value`). On the timing path `display_p` is the per-candidate survival prob (null where ADP null), matching nn's display convention; on the last-pick fallback it is null (`p_na`), matching `season_value`.

**Approximations (inherited from `now_or_never`, stated for the record):** `opp_cost[pos]` is computed against the *current* roster's marginals (not the roster you'll hold at your next pick), and the surviving pool is the pruned top-k (deeper survivors are treated as ~0 marginal). The candidate itself is included in its position's survivor pool (per-position `opp_cost`, same as nn computes `e_best` once per position). All refinable later; v1 matches nn's fidelity.

## 4. Shared helper — `expected_best_by_position`

`now_or_never` inlines the "expected best survivor at each position" accumulation (perf-tuned `np.lexsort` → `itertools.groupby` over contiguous position runs → sequential prefix-product over `1 − p`). The timing strategy needs the identical accumulation fed marginals. Extract it once into `survival.py`:

```
expected_best_by_position(
    positions: np.ndarray,   # str position per row
    values: np.ndarray,      # float value per row (VORP for nn; marginal for timing)
    probs: np.ndarray,       # float survival prob per row
    tiebreak: np.ndarray,    # str, deterministic intra-(pos,value) order (gsis)
) -> dict[str, float]
```

Sort by `(position, −value, tiebreak)`, walk position runs, accumulate `value_i * p_i * Π_{better j}(1 − p_j)` **sequentially** (not `np.sum`, to preserve bit-identical floats).

- `now_or_never` calls it with `values=vorp, tiebreak=gsis` → **bit-identical to today** (same inputs, same order). Its existing pinned tests (hand-computed reorder + determinism) are the regression guard.
- **Fallback:** if extraction perturbs nn's pinned floats at all, leave nn untouched and inline the (~8-line) accumulation in the new strategy instead. DRY with a safety net on merged code.

## 5. Harness generalization — configurable A/B strategy roles

The H2H harness (`src/projections/draft/backtest/`) hardcodes a `4 now_or_never + 4 season_value + 8 bot` field with nn/sv at fixed mirror-paired even seats. Generalize the two strategy **roles** so any strategy pair can be tested:

- **Strategy selection by key.** A small registry maps a strategy key → a constructed `DraftStrategy`, built from the inputs `collect_results` already has: `now_or_never` and `season_value_timing` get `LogisticSurvival(default_sigma(config.n_teams))` (exactly how the harness builds nn today — no new σ flag in v1); `season_value` and `season_value_timing` get `availability` (from `load_inputs`) + `strategy_n_sims` + `base_seed`; `raw_vorp` needs neither. Supported keys: `now_or_never`, `season_value`, `season_value_timing`, `raw_vorp`. `collect_results` takes the role-A and role-B keys (default `now_or_never`, `season_value`).
- `seat_layout(seed)` keeps the same seat sets (A at `{2,6,10,14}`, B at `{4,8,12,16}`, swapped on the paired even seed; bots at the odd seats) — only *which strategy* fills role A/B changes. The 8-bot field is unchanged regardless of A/B.
- **Labels become dynamic.** `aggregate`/`_table` currently hardcode the tuple `('now_or_never','season_value','bot')`; they must instead use the **A and B strategy keys plus `'bot'`**, so `LeagueResult.strategy` and the result tables read e.g. `season_value_timing` / `season_value` / `bot` when testing this strategy. (The seat-weighted-champion identity still holds: 4·A + 4·B + 8·bot champ-rates sum to 1.)
- **CLI flags** `--strategy-a` / `--strategy-b` on `scripts/h2h_backtest.py` and `h2h_backtest_chunked.py`, accepting the supported keys, defaulting to `now_or_never` / `season_value`. The existing `--strategy-n-sims` feeds the season-value MC depth; survival σ comes from `default_sigma` (matching the current harness), so no new σ flag.
- **Default `(now_or_never, season_value)` reproduces today byte-identically** — existing harness tests, the seat-weighted-champion identity, and `test_chunked_collection_matches_monolithic` stay green.
- The 16-team `seat_layout` guard from the `/code-review` pass stays; nothing here changes the team-count assumption.

To test this strategy: run the harness with `A = season_value_timing`, `B = season_value` → a mirror-paired `timing-vs-sv` field plus both vs the bot reference — the same paired-diff shape the F1 analysis used.

## 6. Live assistant CLI

`scripts/draft_assistant.py` / `assistant/cli.py` gains `--strategy season_value_timing`, constructed from the already-wired `--season`/`--data-root`/`--n-sims` (availability, like `season_value`) plus `--sigma` (survival, like `now_or_never`). Default strategy unchanged. The engine never imports the CLI.

## 7. Testing (TDD throughout)

- **`expected_best_by_position`:** hand-computed expected-best-survivor on a small fixture; nn's existing reorder + determinism tests stay green (bit-identical guard).
- **`SeasonValueTimingStrategy`:** determinism (same seed → identical rec); last-pick fallback equals `season_value`'s raw-marginal ranking; a **hand-built reorder case** where a scarce-position, low-ADP-survival candidate ranks above a higher-marginal candidate whose position survives (the timing flip, analogous to nn's reorder test); `score == marginal − opp_cost` on a fixture (no fudge factor); top-k pruning / cosmetic-tail behavior; null-ADP → `p = 1` handling; `__post_init__` raises on `n_sims < 1` and `top_k < 1`.
- **Harness A/B generalization:** default `(nn, sv)` byte-identical (existing tests + chunked-equivalence stay green); swapping A/B mirrors the seats and the seat-weighted-champion identity still holds; labels propagate to the result tables.
- **Live CLI:** `--strategy season_value_timing` parses and smoke-runs.

## 8. Validation plan (data-gathering, no verdict)

Run `season_value_timing` vs `season_value` (and the bot reference) through the H2H harness for **2024 and 2025** (200 seeds × 200 sims, chunked runner), and **log a new Test entry** in `reports/draft_strategy_tests.md` with the per-strategy / paired-bootstrap numbers. Per the standing process rule, record only what it favors in isolation — **no adopt/reject decision**; that is the single end-of-investigation call across all strategies.

## 9. Open questions / future refinements

None blocking. Deferred refinements (each its own later slice): self-exclusion from the survivor pool; opportunity cost against the *future* roster rather than the current one; deeper survivor pool than top-k; a conditional survival model (an available player has already lasted to now); variance/ceiling-aware value function (needs per-player distributions). `σ` reuses `now_or_never`'s `default_sigma = ⅔·n_teams`; empirical σ-tuning is out of scope for v1.
