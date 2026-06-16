# Hero-vs-Bots Strategy Evaluation — Design

**Date:** 2026-06-16
**Status:** Design (pre-plan)
**Branch:** `feat/hero-vs-bots-eval` (stacked on `feat/now-or-never-floored` / PR #72 — needs `now_or_never_floored`)

## 1. Motivation

The H2H backtest harness (Tests 7–10, `src/projections/draft/backtest/`) seats **multiple strategies in one league** — 4 of strategy A + 4 of strategy B + 8 bots, mirror-paired. That answers "if A and B share a draft, who wins," but it is **not the question a drafter actually faces**. In a real draft you run **one** strategy against ~15 humans (≈ noisy-ADP drafters), not a field salted with copies of A and B. The mixed field confounds each strategy's measured outcome two ways that never occur in deployment:

- **Pool contention** — 4 A-drafters and 4 B-drafters deplete *each other's* targets, so A's roster quality depends on B's presence. A 1-hero + 15-bot draft has the realistic depletion pattern (15 ADP-ish opponents).
- **Schedule** — A's win% is partly "A vs B and vs other-A matchups," which never happen when you're the only A.

So both roster quality (points-for) and record (win%/playoff%/champ%) are biased, and even the *ranking* can flip (winning the A-vs-B pool contention ≠ each-vs-bots performance). The decision-relevant experiment is: **each strategy as the sole hero against a fixed bot field, scored on the real-outcome H2H season, compared across strategies under common random numbers.** This design adds that as a second evaluation mode; the mixed-field harness stays for the (rarer) shared-pool question.

## 2. Goal & non-goals

**Goal:** a `hero-vs-bots` H2H evaluation that, for each of the 6 production strategies, runs it as the **sole hero** (swept across **every** draft seat) against a noisy-ADP bot field on the existing real-outcome H2H season (schedule → playoffs → champion), **persists the per-`(strategy, seat, seed, season)` results** (resumable, queryable), and reports **seat-averaged** rates plus a retained **per-seat** breakdown, with bootstrap CIs and paired-diff CIs between strategies. Config-driven; headline run = 16-team half-PPR (`configs/league_espn_half_16team.json`), 2024 + 2025.

**Strategies (6):** `raw_vorp`, `now_or_never`, `now_or_never_floored`, `season_value`, `season_value_var`, `season_value_timing`. `season_value_var` is **included** — the prior "no draft benefit" finding (memory `risk-aware-season-value-no-draft-benefit`) is a principled consequence of the variance model being *mean-preserving* (drafting optimizes the expected marginal, which mean-preserving variance leaves ~unchanged), so it should land **≈ `season_value`**; including it re-tests that on the better methodology and serves as a **determinism/noise control** (a divergence would flag MC-noise re-ranking, not a real edge).

**Non-goals (deliberate deferrals):**
- **Not replacing the mixed-field harness.** `collect_results`/`seat_layout`/the Tests 7–10 path stay byte-identical; this is an additive second mode.
- **No new scoring metric.** Reuses `simulate_league`'s ACTUAL (real weekly actuals) + PROJECTED outputs unchanged.
- **No improvement to the opponent model.** Bots stay the noisy-ADP `bot_pick` (the human proxy). A better human model is a separate, larger lever.
- **No auto-tuning / no strategy changes.** Pure evaluation.
- **No cross-strategy adopt/reject verdict here** (standing decide-at-end rule) — the eval produces numbers; the verdict is the end-of-investigation call.

## 3. The field — hero vs bots, under common random numbers

For one **cell** = `(strategy, seat, seed, season)`: build a seat map with the hero strategy at `seat` and bots (`None`) at every other seat, run the existing `draft_mixed_field` → `simulate_league`, and take the **hero seat's** `LeagueResult` (both ACTUAL and PROJECTED). `simulate_league` already accepts an arbitrary `{seat: DraftStrategy | None}` map + `{seat: label}`, so the season/schedule/playoff/scoring machinery is reused **unchanged**.

**Common random numbers (CRN):** for a fixed `(seat, seed)`, every strategy is run with the **same league seed** → identical schedule and identical bot ADP-noise draws. The hero's different picks ripple into bot availability (that *is* the real snake-draft dynamic, not a CRN violation), but the opponent decision process and schedule are held fixed, so cross-strategy differences at a given `(seat, seed)` are **paired**. This is what makes "strategy X vs strategy Y, each vs the same bots" a low-variance comparison.

## 4. Seat sweep + persistence (report the average, keep the per-seat data)

Draft slot materially changes outcomes (Tests show slot 1 vs 6 vs 12 differ), and you don't know your slot in advance — so the headline is the **seat-averaged** rate. But the per-seat structure is informative ("is the floor better at the turn than the wings?"), so it is **retained, not discarded**:

- **Full sweep:** run every seat `1..n_teams` for every `(strategy, seed)`. Dense, cleanly paired per-seat data (all strategies at every seat for the same seed). Cost is `n_teams × N_seeds × 6` cells.
- **Persist every cell** to a results store keyed by `(season, strategy, seat, seed, scoring)` with the `LeagueResult` fields (`wins, losses, made_playoffs, is_champion, points_for`). Long format (one row per scoring), so re-aggregation is a `groupby` with no re-simulation.
- **Headline aggregation:** group by `strategy` (average over seat + seed) → per-strategy seat-averaged rates. **Per-seat aggregation:** group by `(strategy, seat)`. Both via the existing bootstrap (`aggregate`/`Interval`/`_bootstrap_mean`), generalized to group by the requested keys.

## 5. Resumability (in scope, not deferred)

The sweep multiplies cost by `n_teams`, and 3 of the 6 strategies (`season_value`, `season_value_var`, `season_value_timing`) carry the per-pick season-value Monte-Carlo that has crashed/BSOD'd this box (memory `h2h-backtest-native-crash`). So the runner **must be resumable**, reusing `checkpoint.py`'s pattern:

- Each cell's result is written atomically (temp file → rename); a partially-written cell is never seen as complete.
- On restart, **completed cells are skipped** (presence + validity check), so a crash/reboot resumes where it left off.
- A **manifest guard** (`verify_or_write_manifest`) pins the run identity — `{season, config, n_seeds, strategies, jitter, strategy_n_sims, floor, floor_weight}` — and fails loud if a resume reuses a directory built with different parameters (the per-cell check alone can't catch a parameter change).
- Always invoke via the resumable runner in PowerShell with `KMP_DUPLICATE_LIB_OK=TRUE` + single-thread BLAS.

The persisted per-cell results from §4 **are** the resumable checkpoint store — the two requirements converge on one mechanism. A consolidation step reads the per-cell results into one validated long-format parquet (`HeroResultSchema`) for analysis.

## 6. Architecture (additive; mixed-field untouched)

- **`hero_seat_layout(hero_seat, hero_label, n_teams) -> dict[int, str]`** (in `draft_field.py`) — `{s: hero_label if s == hero_seat else "bot"}`. Works for any `n_teams` (not hardcoded 16); validates `1 <= hero_seat <= n_teams`.
- **`simulate_hero_cell(...) -> tuple[LeagueResult, LeagueResult]`** (new `hero_harness.py`) — build the hero seat map + labels, call `simulate_league`, return the hero seat's `(actual, projected)` `LeagueResult`. Strategies built via the **existing** `_build_strategy` registry (already covers all 6 incl. `now_or_never_floored`); MC strategies require non-null `availability` (fail loud).
- **Resumable sweep runner** (`scripts/hero_backtest.py` + a `hero_cli` core) — iterate `(strategy, seat, seed)` over `seed ∈ [0, N)`, `seat ∈ [1, n_teams]`, the 6 strategies; write/skip per-cell checkpoints; manifest guard. Seed-range-parameterized so chunking/parallelism is a later add.
- **Aggregation/report** (`hero_aggregate` + CLI subcommand) — read the per-cell results → validated `HeroResultSchema` frame → seat-averaged per-strategy table (ACTUAL + PROJECTED, bootstrap CIs) + per-seat breakdown + paired-diff CIs vs a reference strategy. Writes a consolidated results parquet + prints the headline table.
- **`HeroResultSchema`** — long-format results contract (`season:int, strategy:str, seat:int, seed:int, scoring:str∈{actual,projected}, wins:int, losses:int, made_playoffs:bool, is_champion:bool, points_for:float`), validated at the consolidation boundary.

Reused unchanged: `simulate_league`, `draft_mixed_field`, `_build_strategy`, `aggregate`/`Interval`/`_bootstrap_mean`, `load_inputs`, `checkpoint.py`.

## 7. Edge cases / failure modes

- **Invalid hero seat** (`< 1` or `> n_teams`) → raise at layout construction.
- **MC strategy without availability** → fail loud (mirrors `build_session_strategy`).
- **Empty / missing results at aggregation** → fail loud (no silent empty table), as `aggregate` already does.
- **Manifest mismatch on resume** → raise (different config/season/strategy-set/MC params can't pool).
- **Partial cell from a crash** → atomic write means it's re-run, not counted.
- **CRN determinism** — same `(seat, seed)` ⇒ identical bot field + schedule across strategies; the only deliberate difference is the hero's picks. Pinned by a test.
- **`season_value_var` ≈ `season_value`** — expected (mean-preserving); a material divergence is a noise/instability signal to investigate, not a feature.

## 8. Testing (TDD throughout)

- **`hero_seat_layout`** — hero at seat `k`, all others `bot`, for several `n_teams`; raises on out-of-range seat.
- **`simulate_hero_cell`** — determinism (same inputs → identical result); **CRN** (same `(seat, seed)`, two different hero strategies → identical bot labels/field, only hero roster differs); MC strategy needs availability (raises if `None`).
- **Resumable runner** — completed cells skipped on a second run (no recompute); manifest rejects a changed parameter; atomic write (a truncated cell file is treated as incomplete).
- **`HeroResultSchema`** round-trips; consolidation builds a valid frame from per-cell results.
- **Aggregation** — seat-averaged vs per-seat grouping both correct on a hand-built results frame; paired-diff CI sign correct on a constructed case.
- **End-to-end (synthetic)** — a hero strategy that drafts the best players beats a deliberately weak bot field on win%, on a tiny synthetic pool/calendar.
- **Mixed-field unchanged** — existing `collect_results`/`seat_layout`/harness + chunked-equivalence tests stay green.

## 9. Phasing

- **Phase 1** — `hero_seat_layout` + `simulate_hero_cell` + tests (the cell primitive).
- **Phase 2** — `HeroResultSchema` + resumable sweep runner (per-cell checkpoints + manifest) + tests.
- **Phase 3** — `hero_aggregate` (seat-avg + per-seat + paired-diff) + `scripts/hero_backtest.py` CLI (run + report subcommands) + tests.
- **Phase 4 (data run)** — run the eval for all 6 strategies on 16-team half-PPR, 2024 + 2025; write a results section in `reports/draft_strategy_tests.md` (seat-averaged headline + notable per-seat findings, incl. whether `season_value_var ≈ season_value`); update PM/TODO. No cross-strategy adopt/reject verdict (decide-at-end rule). Data is present in the main checkout (`weekly_stats`/`schedules`/`external_projections` 2021–2025 verified).

## 10. Open questions / future refinements

None blocking. Deferred: chunked/parallel sweep across processes (the seed-range signature enables it) if single-process proves too slow even resumable; per-slot strategy *recommendations* (which strategy is best at *your* drawn slot) once the per-seat table exists; a better human-opponent model than noisy-ADP (the biggest realism lever, separate effort); applying this eval to settle the standing `sv > nn > bot` tally on a deployment-realistic footing.
