# Now-or-Never Floored Strategy — Design

**Date:** 2026-06-16
**Status:** Design (pre-plan)
**Branch:** `feat/now-or-never-floored`

## 1. Motivation

`now_or_never` scores each candidate `score = vorp − E[best survivor at position by my next pick]` (`strategy.py`, the opportunity-cost layer). The subtracted term is *small* for a thin position (a talent cliff leaves a low-value best survivor), so for a scarce position the score barely dents — even for a **mediocre** player at that position. The H2H diagnostic (Test 7, `reports/draft_strategy_tests.md`; PM 2026-06-12) root-caused `now_or_never`'s weakness to exactly this: it **over-invests draft capital in scarce positions** (23% on TE vs the bots' 7% and `season_value`'s 10%) because the wait-cost term **has no absolute floor** — when a position is thin it inflates the score of a best-of-a-bad-tier player, so `now_or_never` reaches for that player instead of a genuinely better one at another position. In H2H, raw weekly RB/WR volume beat positional scarcity (the "elite" TE projected 207, scored 142 like a mid WR). Across two seasons the corrected ranking is `season_value > now_or_never > bot`, with the `now_or_never`-over-bot gap **season-variable** (clear in 2024, ~nil in 2025) — a scarcity floor is the lever to make `now_or_never` *reliably* beat ADP.

This strategy adds the missing guard: an **absolute quality bar** below which a player is demoted, so the dynamic-scarcity term can no longer float a sub-bar player over a better one elsewhere. It is `now_or_never` plus a one-sided hinge penalty — not a replacement.

**This is one more candidate strategy to build, run through the H2H harness, and log.** Per the standing process rule (`reports/draft_strategy_tests.md`), no adopt/reject decision is made here — the single decision across all strategies and seasons comes at the end of the investigation.

## 2. Goal & non-goals

**Goal:** a new `DraftStrategy` (`now_or_never_floored`) that ranks by `now_or_never`'s score minus a hinge penalty below an absolute VORP bar `F`, weighted by `λ`. Analytic (no Monte-Carlo, live-draft-fast, same cost as `now_or_never`). Selectable in the live assistant CLI and testable in the H2H harness against the existing strategies, with a tunable `(F, λ)` grid for the A/B.

**Non-goals (deliberate deferrals):**
- **No change to `now_or_never`.** It stays the untouched control; this is A/B alongside it. `λ = 0` reproduces it bit-for-bit (the baseline invariant, §7).
- **No change to the value basis.** Still preseason `season_mean_fpts`-derived VORP (the known optimistic bias is a separate, shared concern).
- **No board-adaptive floor in v1.** `F` is a fixed absolute VORP constant. A board-relative anchor (a percentile of the remaining startable pool) is the documented fallback **only if** the A/B shows the fixed `F` fails to transfer across the 2024 and 2025 pools (§8).
- **No automated tuning inside the harness.** `(F, λ)` are swept by manual grid runs (like Tests 7–9), not an in-harness optimizer.
- **No new survival model / no σ change.** Reuses `now_or_never`'s `LogisticSurvival(default_sigma(n_teams))`.

## 3. The strategy — `NowOrNeverFlooredStrategy`

A frozen dataclass behind the existing `DraftStrategy` protocol (CLI key `now_or_never_floored`), in `src/projections/draft/assistant/strategy.py`. It holds `now_or_never`'s survival model plus the two floor knobs:

```
NowOrNeverFlooredStrategy(
    survival: SurvivalModel,
    floor: float = 30.0,        # F: quality bar, VORP units (replacement = 0)
    floor_weight: float = 1.0,  # λ: hinge hardness (0 = exactly now_or_never)
)
```

`__post_init__` validates `floor_weight >= 0` (a negative weight would *reward* being below the bar) and that `floor`/`floor_weight` are finite (not NaN/inf — a NaN floor would silently turn every penalty into NaN and collapse the score column to null). `floor` itself is unconstrained in sign: `F = 0` penalizes only below-replacement players; `F = 30` penalizes everyone below 30 VORP.

**`recommend(state, pool, config) -> RecommendationSchema`:**

1. **Eligible subset (identical to `now_or_never`):** `df, elig = _eligible_subset(state, pool, config)`.
2. **Pick timing:** `next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)`.
   - **`next_pick is None` (last pick):** `_raw_vorp_result(df, elig)` — raw VORP, null `p_available`, **floor not applied**. Identical to `now_or_never`'s last-pick fallback (on the final roster spot you take best-available; there is no future pick for scarcity to trade against). This also keeps the `λ = 0 ≡ now_or_never` invariant exact on the last pick.
   - **Otherwise:** survival prob per row `internal_p = survival.p_available(adp, next_pick)` (null ADP → `1.0`, `now_or_never`'s convention); `display_p = internal_p` masked to null where ADP is null. Compute `e_best = expected_best_by_position(pos, vorp, internal_p, gsis)` — the **already-extracted** shared helper (`survival.py`), called with the exact same `(vorp, gsis)` arrays `now_or_never` uses.
3. **Score (the only new line):**
   ```
   penalty = floor_weight * np.maximum(0.0, floor - vorp)     # elementwise, ≥ 0
   score   = vorp − e_best[pos] − penalty
   ```
   The hinge is **continuous at `vorp = F`** (penalty 0 there) — above the bar `score = vorp − e_best` (exactly `now_or_never`); below it the slope on `vorp` steepens to `(1 + λ)` and a constant `−λF` shifts the whole sub-bar group down relative to above-bar players, so a sub-bar scarce player can no longer outrank a clearly-better above-bar player. No discontinuity, no separate tier.
4. `_finalize(df, elig, display_p)` — `starting_need_tier=True`, exactly like `now_or_never` (the score is in VORP units; the starting-need tier still applies first, so the floor operates *within* the fills-starting-slot tiers, as `now_or_never`'s opportunity cost does).

**Approximations (inherited from `now_or_never`, stated for the record):** `e_best` is the expected best survivor at each position against the *current* board (not the board at your next pick); the candidate is included in its own position's survivor pool. The floor adds no new approximation — it is a deterministic function of the candidate's own VORP. All `now_or_never` refinements remain open and shared.

## 4. Shared helper — `expected_best_by_position` (reuse, no change)

`expected_best_by_position(positions, values, probs, tiebreak) -> dict[str, float]` already lives in `survival.py` (extracted in the `season_value_timing` slice) and is what `now_or_never` itself calls. The floored strategy calls it with the identical `values=vorp, tiebreak=gsis` arguments → bit-identical `e_best` to `now_or_never`. **No edit to the helper or to `now_or_never`.** This is the single source of the opportunity-cost term for all three of `now_or_never`, `season_value_timing`, and this strategy.

## 5. Harness A/B — register one key (generalization already shipped)

The H2H harness (`src/projections/draft/backtest/`) already supports arbitrary strategy pairs via `--strategy-a`/`--strategy-b` and a key→`DraftStrategy` registry (`_build_strategy`, shipped in the `season_value_timing` slice). This slice only:

- **Adds `"now_or_never_floored"` to `STRATEGY_KEYS`** (`strategy.py`) and to the harness registry, constructed like `now_or_never` — `survival = LogisticSurvival(default_sigma(config.n_teams))` — plus the new `floor` / `floor_weight`.
- **Threads `floor` / `floor_weight` through the registry** as optional params (defaults `30.0` / `1.0`), consumed only when a role's key is `now_or_never_floored`; ignored for every other key.
- **Adds `--floor` / `--floor-weight` flags** to `scripts/h2h_backtest.py` and `scripts/h2h_backtest_chunked.py`, feeding those params so the `(F, λ)` grid can be swept across runs. Existing `--strategy-n-sims` / σ behavior is untouched.
- **The checkpoint manifest guard** (`checkpoint.verify_or_write_manifest`) must include `floor` / `floor_weight` so a resume with a different floor cannot silently pool mismatched chunks (the same provenance concern TODO #43 closed for the strategy pair).
- **Default `(now_or_never, season_value)` reproduces today byte-identically** — existing harness tests, the seat-weighted-champion identity, and `test_chunked_collection_matches_monolithic` stay green.

To test this strategy: run `A = now_or_never_floored`, `B = now_or_never` (and both vs the bot reference) — the same mirror-paired paired-diff shape the F1 analysis used, isolating the floor's effect against its own parent.

## 6. Live assistant CLI + board

- `scripts/draft_assistant.py` / `assistant/cli.py` gains `--strategy now_or_never_floored` with `--floor` / `--floor-weight` (defaults `30.0` / `1.0`), constructed from the already-wired `--sigma` (survival, like `now_or_never`). Default strategy unchanged.
- The shared `build_session_strategy` seam (used by both the CLI and the live board) gains the key, constructed with `floor` / `floor_weight` parameters that **default** to `30.0` / `1.0`. Adding the key makes the strategy **selectable on the live board** at the default `(F, λ)` with no further UI work. Exposing `floor` / `floor_weight` as **board sliders is a deferred polish** (non-goal for v1) — the board offers the strategy at its validated default.

## 7. Testing (TDD throughout)

- **`λ = 0 ≡ now_or_never` (the baseline invariant):** `NowOrNeverFlooredStrategy(survival, floor=F, floor_weight=0.0).recommend(...)` returns a **byte-identical** `RecommendationSchema` frame to `NowOrNeverStrategy(survival).recommend(...)` for any `F`, on a shared fixture (assert frame equality including `score`/`rank` ordering). This pins "no regression to the parent."
- **Score formula:** `score == vorp − e_best − λ·max(0, F − vorp)` hand-computed on a small fixture (no fudge factor); continuity checked at a candidate with `vorp == F` (penalty 0).
- **The pathology fix (a directed reorder test):** a hand-built board where a scarce-position, sub-`F` candidate ranks **above** a higher-VORP candidate under `now_or_never` (`floor_weight=0`) but **below** it once the floor bites (`floor_weight>0`, `F` between the two VORPs) — the exact best-of-a-bad-tier flip the diagnostic identified.
- **Last-pick fallback:** `next_pick is None` → ranking equals `now_or_never`'s raw-VORP fallback (floor not applied), for any `(F, λ)`.
- **Construction guards:** `__post_init__` raises on `floor_weight < 0` and on non-finite `floor` / `floor_weight`.
- **Null-ADP** → `p = 1` handling (same as `now_or_never`).
- **Harness A/B:** default `(now_or_never, season_value)` byte-identical (existing tests + chunked-equivalence stay green); `now_or_never_floored` as a role constructs and runs; the manifest guard rejects a resume with a changed `floor` / `floor_weight`.
- **Live CLI / board seam:** `--strategy now_or_never_floored --floor … --floor-weight …` parses and smoke-runs; `build_session_strategy("now_or_never_floored", …)` constructs at defaults.

## 8. Validation plan (data-gathering, no verdict)

Run `now_or_never_floored` vs `now_or_never` (and the bot reference) through the H2H harness for **2024 and 2025** (200 seeds × 200 sims, chunked runner in PowerShell with `KMP_DUPLICATE_LIB_OK=TRUE` + single-thread BLAS, per memory `h2h-backtest-native-crash`), over a small grid — starting `F ∈ {0, 20, 40, 60}`, `λ ∈ {0.5, 1, 2}` — and **log a new Test entry** in `reports/draft_strategy_tests.md` with the per-strategy / paired-bootstrap numbers per `(F, λ)`. Watch specifically whether a single `(F, λ)` helps in **both** seasons (transfer); if the best fixed `F` is season-split, that is the trigger to try the board-adaptive anchor (§2 fallback). Per the standing process rule, record only what it favors in isolation — **no adopt/reject decision**; that is the single end-of-investigation call across all strategies.

## 9. Open questions / future refinements

None blocking. Deferred (each its own later slice): board-adaptive floor anchor (percentile of remaining startable pool) if the fixed `F` fails to transfer; `floor` / `floor_weight` as live-board sliders; the same shared `now_or_never` refinements (opportunity cost against the *future* roster, conditional survival model, deeper-than-current survivor pool); applying a floor idea to the season-value-space strategies (`season_value_timing`). Empirical σ-tuning remains out of scope (reuses `default_sigma = ⅔·n_teams`).
