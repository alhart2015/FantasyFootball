# TODO

Running project management list. Add items as they come up; remove or check off when resolved.

## Open

### 49. Auction bid-model data-gathering — decision September 2026 (branch `feat/auction-tournament`; sane-bots branch `feat/auction-sane-bots`)

Harness shipped (`src/projections/draft/assistant/auction/`); tracking doc `reports/auction_tournament_validation_2026.md`. Sane-bots slice shipped: `bot_position_bounds` (league-driven per-position min/max) + `bot_eligible` (shared iteration-domain gate); Run B bake-off recorded (150 seeds, n_sims=500; static playoff 0.10/champ 0.01 vs Run A 0.15/0.01; ranking unchanged, static ≥ marginal ≥ inflation). Run experiments through summer and make the bid-model call in September 2026. Axes still to sweep: `price_jitter` sweep; seat sweep; scoring×size presets. Hero absolute metrics still below uniform baseline even with sane bots — better bot WTP model remains the single biggest realism lever. No winner declared until the September decision. See PM entries (2026-06-17) for decisions and gates.

**Update (2026-06-18, branch `feat/auction-budget-urgency`, PR #81):** budget-urgency slice shipped — shared `_budget_urgency` late-draft deployment factor (`URGENCY_GAIN=3.0`) applied at a single exit by all 7 contestants (empty-roster behavior unchanged) + new 8th contestant `StudsAndDepthBid`. **Run F** (eight-model bake-off vs the realistic market, 40 seeds) recorded: urgency compresses the field by forcing cash-hoarders to deploy (`inflation` playoff 0.03→0.12, `marginal` 0.00→0.08, `vorpshare` 0.05→0.13 vs Run E); `studsdepth` mid-pack, `anchors` last. **Still no winner — September decision unchanged.** Better-calibrated bot WTP (anchored on real published auction values) remains the biggest open realism lever.

### 48. Projected draft eval + scoring/size presets — ✅ DONE (branch `feat/draft-eval-presets`, stacked on `feat/draft-board-ux`)

End-of-draft "how good is this roster?" made first-class + shown in the board, plus configurable scoring/size (spec/plan `docs/superpowers/specs|plans/2026-06-16-projected-draft-eval.*`). **Projected-vs-projected** full-league MC of the completed draft: each sim-week draws injury + over/under performance (variance model), sets the optimal starting lineup for every team, higher projected total wins; reg wks 1-13, playoffs 14-17 (top-6, top-2 bye, championship = wks 16-17). `src/projections/draft/assistant/league_projection.py` (`gauntlet_schedule`/`team_weekly_points`/`project_draft`/`SeatProjection`) + `LiveDraftSession.project_league_outcomes` + board results panel. **Measures roster quality under our projections, NOT projection accuracy** (out of scope; real-outcome eval is the hero-vs-bots harness) — and it's somewhat circular (rewards what season_value optimizes) + slot/ADP-field inflated, so it shows projection-optimality, not real-league wins. **Scoring×size presets:** 3×3 {half-PPR, full PPR, standard} × {10,12,16}, default half-PPR/16, sidebar dropdowns; `scripts/generate_preset_vorp_tables.py` re-scores the raw 2026 snapshot per ruleset (half/std genuinely re-score receptions) → `data/vorp_2026/{scoring}_{n}team.parquet` (**untracked — regenerate with the script after each consensus refresh**). `is_rookie` helper promoted to shared `assistant/rookies.py`. **Deferred:** configurable playoff format (size/byes/week layout); a preset-table regeneration step in the data-refresh workflow.

### 47. Live Draft Board UX — name fix + click-to-pick + best-available filter — ✅ DONE (branch `feat/draft-board-ux`)

Three board usability fixes (spec/plan `docs/superpowers/specs|plans/2026-06-16-draft-board-ux.*`; pure UI + name plumbing, no strategy/engine change): (1) **the "ADP suggests: —" bug** — placeholder-gsis rookies (60/458, absent from id_map) now show real names by carrying `full_name` into the VORP table (Optional `VorpTableSchema.full_name` + consensus-generator merge), with `LiveDraftSession.player_names` overlaying pool name over id_map and `cli.format_table` matching; the `record_pick` my-pick guard was decoupled to a dedicated `_id_map_ids` set so the broadened name map can't let an off-id_map rookie poison my roster (verified `99-8467088 → "Jeremiyah Love"`). (2) **click-to-confirm** — top search box removed; click a row in the center recommender or the right best-available pane → shared **Confirm pick** button (most-recent click wins; opponent ADP suggestion stays a 1-click confirm). (3) **best-available picker** — new `available_for_pick(position, query, top)` (All/QB/RB/WR/TE dropdown + case-insensitive search + vorp-sort + cap-last) rendered as a searchable selectable table. Regenerate `data/consensus_vorp_2026.parquet` after each consensus refresh so the board shows current names. **Deferred follow-up:** resolve my-roster position from the pool's `position` column (via `build_draft_state`) so placeholder-gsis rookies are draftable to *my own* roster, not just recordable as opponents' picks — the one functional limitation this pass leaves open.

### 46. Hero-vs-bots eval — ✅ DONE (harness + run; branch `feat/hero-vs-bots-eval`, stacked on PR #72)

Deployment-realistic strategy eval: each strategy as the **sole hero vs a 15-bot field** (not the mixed A-vs-B field of Tests 7–10, which confounds via shared-pool contention + schedule-vs-other-strategies). Swept across all 16 seats, CRN across strategies, per-seat retained. `src/projections/draft/backtest/hero_harness.py` + `scripts/hero_backtest.py` (`run`/`report`). **Data points (Test 11, FIVE seasons 2021–2025) — data-gathering, no strategy committed (decide-at-end; draft months out):** in isolation the most consistent signal is **`now_or_never_floored` > `now_or_never`** (pooled +2.35 win%, CI-separated, 4/5 seasons) — recorded, not a recommendation. Per-season rankings swing wildly (sv best 2021, raw_vorp best 2025, floored best 2022–24), so the earlier 2-season "sv is bottom-tier / simple value wins" cut was an artifact: pooled, `season_value` is ~neutral vs nn but high-variance, `raw_vorp` is worst on average. `season_value_var ≈ season_value` across all 5 (no draft benefit, confirmed). Mixed-field `sv > nn > bot` ordering doesn't hold solo, but sv is "high-variance/neutral," not "loses." **Follow-ups (open):** (a) **better human-opponent model than noisy-ADP** — the single biggest realism lever, since bots are the human proxy and the whole eval rests on them; (b) chunked/parallel cross-process sweep (the seed-range signature enables it) if higher N is wanted; (c) per-slot strategy *recommendations* (which strategy is best at your drawn seat) from the retained per-seat table; (d) extend N beyond 25 (resumable) to tighten CIs / re-confirm the inversion.

### 45. Per-player weekly performance distribution in the season-value Monte Carlo — ✅ DONE (v1; branch `feat/performance-variance-model`)

**Shipped** (spec/plan `docs/superpowers/specs|plans/2026-06-14-performance-variance-model.*`): a fitted two-component variance model — mean-preserving lognormal season-mean multiplier `m` (E[m]=1, log-SD by position×rookie) + per-game-affine Gamma weekly noise (`std = a_pos·pg + b_pos`). `scripts/fit_performance_variance.py` fits it offline (2018-2025 weekly_stats for the affine; 2021-2025 for the log-SD) → committed `configs/performance_variance_params.json` (R7a: arith ratio SD 0.374 vet / 0.580 rookie; within-season CV QB 0.48 / RB-WR-TE 0.68-0.69). `performance_variance.py` is the vectorized sampler. **Consumer A:** `season_value.py` `expected_season_points_var` / `marginal_season_values_var` make the MC risk-aware (optimal-by-sampled weekly fill, CRN-preserved), wired into the SeasonValue strategies; **26 ms/pick at n_sims=200 (R4 ✓, no fallback needed)**; H2H smoke shows sv stays competitive (R7b ✓). **Consumer B:** `backtest/predictive.py` (`score_drafted_league` extracted from `simulate_league`) drafts once then re-scores model-sampled seasons → `scripts/post_draft_assessment.py --predictive`. R7c ✓: 2025 forward champ 8.3% [5.0,12.1] vs the too-tight historical 5.2% [3.8,6.9] — materially wider, reflecting player-outcome luck. `is_rookie` attached to the pool in `load_inputs`. **Deferred follow-ons:** per-player empirical-Bayes shrinkage (0.69-reliable signal); same-game correlation (TODO #1 option D); playoff-week weighting; draft-capital rookie refinement; realistic lineup-set-by-projection in Consumer A (v1 uses optimal-by-sampled). **Original write-up below for history.**

<details><summary>Original spec rationale</summary>

The season-value MC (`src/projections/draft/assistant/season_value.py`) currently models only **availability** variance — each week a player is available (Bernoulli `p`) or out (injury/bye), and when available they score their **deterministic** `per_game = season_mean_fpts / 17`. There is **no week-to-week performance variance**: an available player scores the exact same points every simulated week. Two consequences:

1. **The MC understates real outcome spread.** Boom/bust weeks, matchup variance, and the fat tails that actually decide H2H matchups are absent — the sim only sees roster construction + availability + schedule, not the week-to-week scoring noise that is most of real fantasy variance.
2. **Post-draft assessment CIs are too tight** (flagged in the corr/post-draft analysis, `scripts/post_draft_assessment.py`): the H2H backtest resamples only **draft order and schedule**; each player's weekly actuals are a single fixed historical realization, so the CIs reflect draft/schedule luck only, not player-outcome luck. An honest CI should be ≈ the cross-season spread (2021-2025), widened for player variance — much wider than the bootstrap currently shows.

**The fix — plug real per-player-per-week distributions into the MC.** Projections Core already produces **per-player, per-week probability distributions over fantasy points** (`src/projections/`, the `Distribution` protocol + scoring layer) — that is exactly the missing piece. In the season-value MC, replace the deterministic `per_game` with a **draw from the player's weekly fantasy-point distribution** (conditional on being available), so each simulated week samples a real boom/bust outcome. Keep CRN/antithetic structure for the marginal-value comparisons. Consider correlation (TODO #1 option D — same-game stacks) as a later refinement; v1 can use marginal weekly distributions (independent draws) which already fixes the spread.

**Why it's its own spec/plan, not an inline change:** (a) it touches the hot path — `_vectorized_lineup_points` and the ~125× fast-path must stay fast with an added per-week sampling dimension (n_sims × weeks × players draws); (b) it needs a source for the weekly distribution at *draft time* — preseason we have a season projection, not 17 weekly distributions, so we need a defensible way to spread a season projection into weekly distributions (variance model by position/role, or reuse the Projections Core weekly model's dispersion); (c) it changes every downstream number (`SeasonValuer`, `SeasonValueStrategy`, `SeasonValueTimingStrategy` marginals, and the tournament/backtest), so it needs the adoption-gate / A-B discipline. **Also widen the post-draft CI methodology** (`scripts/post_draft_assessment.py`) once this lands: resample player outcomes from their distributions, not just draft/schedule.

</details>

### 44. Waiver-wire / undrafted-pool assessment after the 16-team draft sims — which positions are weak vs strong

After a 16-team draft (`draft_mixed_field` over the consensus / real-season pool, 176 players taken), look at the **undrafted pool** (pool − the union of all rosters) and characterize the **waiver wire by position**. For QB/RB/WR/TE report, averaged over many draft seeds (and ideally per season 2024/2025): best-available VORP / projected points still on the board; depth = count of remaining players above replacement (or above a startable threshold); and a relative-strength readout — which positions are **depleted** (waiver is thin → the heavily-drafted/scarce ones) vs **relatively strong** (good startable talent remains → the deep, streamable ones). Output a per-position ranking of "how good is the best guy you can still grab."

**Why it matters:** (a) draft strategy — don't over-invest in a position whose waiver stays deep; prioritize the one that dries up (connects to the scarcity findings in Test 9 / TODO #42 — e.g. does the field drain TE to nothing, making elite-TE scarcity real, or does it stay streamable?); (b) seeds the future mid-season waiver/streaming tool (which positions to stream vs roster). Pure analysis on existing sim machinery — no new strategy needed; could live as a `scripts/` analysis or a small `draft/backtest` reporter.

### 43. H2H chunk-runner — stamp checkpoint provenance (from `/code-review` of PR #64) — ✅ DONE (PR #65)

Resolved by the checkpoint **manifest guard** shipped in PR #65: `scripts/h2h_backtest_chunked.py` now writes a per-dir `manifest.json` of `{season, strategy_a, strategy_b, strategy_n_sims, jitter}` via `checkpoint.verify_or_write_manifest`, and **fails loud on resume if the params differ** — closing the silent mismatched-pool path (the row-count gate alone couldn't catch it). (Original concern below for history.) ~~`scripts/h2h_backtest_chunked.py` checkpoints carry no provenance, and `_valid_chunk_file`'s row-count gate is self-fulfilling (both sides derive from `n_teams`). So reusing a checkpoint dir across a different `--season` / `--strategy-n-sims` / league config silently pools mismatched chunks with no error.~~ (Other `/code-review` findings — 16-team `seat_layout` guard, `regular_weeks`/win% div-by-zero guard, pool-loop TOCTOU, `score_by`↔key coupling comment — were fixed inline in PR #64's review commit.)

### 42. `now_or_never` — scarcity vs raw-value re-weighting with an absolute floor (from the H2H backtest dig)

**Update 2026-06-16 (branch `feat/now-or-never-floored`, PR #72): BUILT + wired + gated + A/B'd — the floor works.** Shipped `NowOrNeverFlooredStrategy` (`score = vorp − E[best survivor] − λ·max(0, F − vorp)`) as a separate, A/B-able strategy (key `now_or_never_floored`) — selectable in the assistant CLI, the live board, and the H2H harness via `--floor`/`--floor-weight`; `λ=0` reproduces `now_or_never` byte-for-byte (nn untouched). **A/B verdict (Test 10 in `reports/draft_strategy_tests.md`):** at F=40/λ=1 the floor gives a big CI-separated win in **2025** (win 55.9% vs nn 50.0; beats the bot 47% — fixes the Test 7/8 `nn ≈ bot` 2025 failure) and is neutral-to-slightly-positive in **2024** (never hurts). Default **F=40/λ=1** (`_DEFAULT_FLOOR`/`_DEFAULT_FLOOR_WEIGHT`). **Remaining follow-up:** a floored-vs-`season_value` A/B (does the floor close nn's gap to sv?) — not run here. The original disproven-leads framing below still stands. **Original framing below.**

The 2025 H2H backtest (Test 7/F1, `reports/draft_strategy_tests.md`) found `now_or_never` has **no meaningful edge over a noisy-ADP bot** on real outcomes, while `season_value` clearly wins. Root-caused (not injuries / variance / QB-stash / TE-replacement — all ruled out): nn's opportunity-cost layer `score = vorp − E[best survivor at position by next pick]` **over-invests draft capital in scarce positions (23% on TE vs bots' 7%)** because the wait-cost term has **no absolute floor** — when a position is thin it inflates the score of a *mediocre* player at that position, so nn reaches for the best-of-a-bad-tier instead of a better player elsewhere. In H2H, raw weekly RB/WR volume beats positional scarcity (the "elite" TE projected 207, scored 142 like a mid WR).

**Idea to brainstorm (own spec/plan):** re-weight scarcity vs raw value in `NowOrNeverStrategy`, adding an **absolute value floor** — below a quality bar a player isn't worth taking no matter the wait-cost. A/B-able behind the existing `DraftStrategy` protocol; validate against the H2H real-outcome metric (now the honest yardstick), not the season metric (confounded). **Disproven leads (don't revisit):** availability/injury discount (nn ≈ bot on availability); QB cap (sv stashes QBs harder and wins). Also possible secondary refinement: the VORP `bench_cushion` is multiplicative (1.3×demand) so it adds a deeper backup buffer to high-demand positions — worth checking whether an additive cushion calibrates cross-position replacement better (pushes toward under-crediting WR, not the TE issue).

### 41. Filter playoff weeks at the ingest boundary (currently per-consumer, and inconsistently)

Surfaced by the `/code-review` + `/simplify` of the risk-aware availability work (PR #60). `weekly_stats` and `schedules` partitions both carry playoff rows (`WeeklyStatsSchema.week`/`SchedulesSchema.week` allow `le=22`; `ingest/weekly_stats.py` and `ingest/schedules.py` keep no `season_type`/`game_type` column and apply no REG filter). Consumers each re-derive "regular season" independently, and they **disagree**:

- `draft/assistant/availability.py` filters via the era-aware `_last_regular_week(season)` (17 pre-2021, 18 after) — correct.
- `preseason/features.py:228` hardcodes `weekly_stats["week"] <= 17` for the `prior_N_season_games_played` aggregation. **This is a latent bug for 2021+:** it drops regular-season week 18, undercounting `games_played` (its own comment mislabels week 18 as "playoff"). Verify and fix to the era-aware cutoff, or delete it once ingest filters.

**Deepest fix:** filter `season_type == "REG"` once in `ingest/weekly_stats.py` / `ingest/schedules.py` (nflreadpy exposes `season_type`) and tighten the schema `week` bound to `le=18`, so every present/future consumer is protected and the invariant is documented. That changes stored partitions (other consumers, e.g. PBP receiver features, tolerate `week<=22`), so it was out of scope for the draft slice. When it lands, also hoist the era split (`_sched_games` / `_last_regular_week`) to a shared schedule/calendar location instead of living in `availability.py`.

### 40. Regenerate the backtest snapshot for the baseline Vegas feature change (`--run-backtest`)

**The 15 tests that were red on `main` are fixed in this PR (`chore/test-suite-speedup`).** Root cause was PR #51 half-completing its Vegas team-context integration: it added the 4 cols (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) to `Qb/WrFeaturesSchema` but left `_WR/_QB_FEATURE_COLUMNS` (baseline.py) and two hardcoded WR fixtures (`test_decomposed_baseline`, `test_tune_lightgbm`) inconsistent. Fixed by bringing the feature lists to schema parity + emitting the cols in those fixtures. (`pyproject.toml` also now requires `tabulate>=0.9` — `pip install -e .` to pick it up.)

**Remaining follow-up:** bringing `_WR/_QB_FEATURE_COLUMNS` to parity means `BaselineModel` now consumes the 4 Vegas features, so its WR/QB predictions change and the checked-in walk-forward snapshot `tests/backtest/model_metrics.json` is stale for the baseline WR/QB cells. The snapshot gate is `--run-backtest`-only (skipped by default, so the default suite is green) — regenerate (`scripts/backtest.py --update-snapshot`) before the next gate run.

### 38. External consensus projection layer for draft (sub-project #2) — **PRIORITY**

**Why now.** The external projection benchmark spike (2026-06-08, branch `feat/external-projection-benchmark`; see `project_management.md` top entry + `reports/external_projection_benchmark_2024.md`) established that our home-grown model is a weekly in-season model and **cannot produce a preseason/draft projection at all** (proven: it projected the injured CMC for only 4 weeks at 63 pts vs ESPN preseason 335). For the draft use case (priority #1), there is nothing on our side to benchmark — pivot to external sources.

**Goal.** Build the external-ingest + consensus layer that becomes the projection basis the downstream tools (Draft Hub first) consume. Spend effort on *how we use* projections, not on the projections themselves.

**Foundation already built** (on the spike branch, reusable): `scripts/pull_external_projections.py` pulls ESPN preseason stat lines + ADP + draft ranks and Sleeper ADP (no auth), crosswalk-ready to `GsisId`. `scripts/benchmark_projections.py` has correct join + PPR-scoring + metric machinery (reusable for the fair weekly benchmark, NOT for a preseason verdict — see its docstring).

**v1 ingest mechanism (#2a) — ✅ DONE** (branch `feat/external-projection-ingest`; spec/plan at `docs/superpowers/specs|plans/2026-06-08-external-projection-ingest.*`). `src/projections/ingest/external_projections.py` + `ExternalProjectionSchema` + a store `asof` (pull-date) partition: a repeatable `refresh_external_projections(season=2026)` writes dated snapshots (ESPN stat line + ADP + draft rank; Sleeper ADP + name/position), `gsis_id`-keyed with deterministic `99-XXXXXXX` placeholders for pre-camp rookies (auto-reconcile on later refreshes via `source_player_id`). The `id_map` float-stringified-id defect is fixed at the source. Verified live for 2026 (3,492 rows; veterans → real gsis, e.g. Chase 00-0036900 from both sources; rookies → 99- placeholders). **Re-run `python -m projections.ingest.external_projections --season 2026` weekly up to the August draft** (regen `id_map` too, to pick up rookies as their real gsis_ids land ~late July).

**v1 consensus blend (#2b slice 1) — ✅ DONE** (branch `feat/external-consensus-blend`; spec/plan at `docs/superpowers/specs|plans/2026-06-09-external-consensus-blend.*`). Pure `build_consensus` (`src/projections/consensus/blend.py`) + `refresh_consensus` orchestrator + CLI (`python -m projections.consensus.refresh --season 2026 [--asof YYYY-MM-DD]`) blend the raw ESPN+Sleeper snapshot into the published `ConsensusProjectionSchema` table at `data/processed/consensus_projections/season=YYYY/asof=YYYY-MM-DD/` (derived; asof mirrors the raw snapshot). Per player: 2-source **consensus_adp** (simple mean) → ordinal **consensus_rank**, **n_adp_sources**, ESPN-derived stat line → **projected_points_ppr** via the new fractional **`scoring.expected_points`** (`score()`'s int `StatLine` can't score fractional projections). **Union coverage** — every player ranked by ≥1 source appears; ADP-only players get null points (`has_points=False`). **Point estimates only — distribution-wrapping deferred** to the scraped-source slice (a single points source has no real cross-source spread). `_placeholder_name_key` promoted to the shared `ingest/identity.py` util (ingest + blend agree by construction). Verified live for 2026 (3,042 players; 458 with points; Bijan Robinson rank 1 at consensus_adp 1.745). **Re-run after each ingest refresh** to regenerate the consensus snapshot.

**Remaining scope (#2b+, still open):**
- **Draft Hub consumption — ✅ DONE** (branch `feat/draft-hub-consensus`; spec/plan `docs/superpowers/specs|plans/2026-06-09-draft-hub-consensus.*`): VORP/auction/cheat-sheet now consume `ConsensusProjectionSchema` via `consensus_to_season_projections` + VORP CLI `--source consensus`; cheat sheet has `adp_delta`; skill-positions-only (K/DST = TODO #10). **Re-run after each consensus refresh:** `python scripts/generate_vorp_table.py --source consensus --season 2026 --league-config configs/league_espn_ppr_12team_skill.json --out <path>`.
- **Live Draft Assistant — engine core (Slice 1) — ✅ DONE** (branch `feat/draft-assistant-engine`; spec/plan `docs/superpowers/specs|plans/2026-06-09-draft-assistant-engine.*`; PM top entry). Headless pluggable-strategy pick recommender over the consensus VORP table: `DraftStrategy` Protocol + `NowOrNeverStrategy` (analytic opportunity-cost / grab-now-vs-wait) + `RawVorpStrategy` control; snake `pick_timing`, ADP `LogisticSurvival`, `DraftState`/`load_draft_state` (id_map-resolved roster), shared `roster_eligibility` (greedy slot allocation, promoted out of `_pool.py`), `RecommendationSchema`, and CLI `scripts/draft_assistant.py`. Usage: `python scripts/draft_assistant.py --state <state.json> --vorp-table <consensus_vorp.parquet> [--strategy now_or_never|raw_vorp] [--sigma N] [--top N]`. The state file is `{"league_config": "<path>", "my_slot": int, "picks": [gsis_id, ...]}`. **Remaining Draft Assistant slices:** Slice 3 — Streamlit live-draft UI over this engine.
- **Live Draft Assistant — strategy comparison harness (Slice 2) — ✅ DONE** (branch `feat/draft-strategy-tournament`; spec/plan `docs/superpowers/specs|plans/2026-06-10-draft-strategy-tournament.*`; PM top entry). CLI tournament over the Slice 1 engine: `simulate_draft` (hero strategy vs noisy-ADP `bot_pick` field), `optimal_lineup_points` (optimal starting-lineup scoring), `run_tournament` (paired-seed bootstrap; winner only when the top-two diff CI excludes 0) + `tune_sigma` (σ-grid argmax). League-driven (roster shape/team count/ruleset from `LeagueConfig`); no new schema. Usage: `python scripts/draft_tournament.py --vorp-table <consensus_vorp.parquet> --league-config <league.json> --my-slot N [--seeds K] [--adp-jitter F] [--seed B] {compare [--strategy-sigma S] | tune-sigma [--sigma-grid "a,b,c"]}`. **σ is now empirically tunable** — run `tune-sigma` on a real consensus VORP table to replace the `⅔·n_teams` default. **Auction tournament** stays a documented future seam (simulate→score→compare split: only the draft-mechanism module swaps). Slice 3 (Streamlit UI) is the last Draft Assistant slice.
- **Live Draft Assistant — risk-aware roster valuation (season availability) — ✅ DONE** (branch `feat/risk-aware-roster-valuation`; spec/plan `docs/superpowers/specs|plans/2026-06-11-risk-aware-roster-valuation.*`; PM top entry). Replaces the starters-only tournament metric with expected season points under per-player availability (injury Bernoulli from `weekly_stats` games-played history + byes from `schedules`), filling the best legal lineup each week — so bench depth and positional risk finally matter. `availability.py` + `season_value.py` (MC valuer reusing `optimal_lineup_points/17`, single-week factorization) + pluggable `RosterValuer` (`StartersValuer` default-unchanged / `SeasonValuer`). Usage: `python scripts/draft_tournament.py ... --valuer season --season 2026 --n-sims 300 {compare|tune-sigma}`. **Validated:** under the season metric now_or_never beats raw_vorp by +286 (vs +75 starters) — the depth-aware metric penalizes raw_vorp's 10-QB bench. **Data dep:** the 2026 `schedules` partition isn't ingested → `--season 2026` degrades to no byes (warn); ingest it for bye coverage. **Deferred follow-ups:** depth-aware *strategy* (the natural next slice — now has a metric to optimize), weekly performance variance (needs real distributions), recency-weighted availability, playoff weighting, numpy fast-path for the weekly fill. Slice 3 (Streamlit UI) remains the last Draft Assistant UI slice.
- **Live Draft Assistant — depth-aware strategy (`SeasonValueStrategy`) — ✅ DONE (shipped *selectable*; validation negative)** (branch `feat/depth-aware-draft-strategy`; spec/plan `docs/superpowers/specs|plans/2026-06-11-depth-aware-draft-strategy.*`; PM top entry; `reports/depth_aware_strategy_validation_2026.md`). First strategy that drafts *to* the season metric: ranks each candidate by marginal expected season points it adds (`V(roster+c)−V(roster)`, CRN), prunes top-k-by-VORP/position, ranks by marginal (no `fills_starting_slot` tier). `SeasonValueStrategy` + `expected_season_points_crn`/`marginal_season_values` + `DraftState.my_pick_ids` + shared `load_store_availability`; wired into both CLIs (`compare --with-season-value`; live `--strategy season_value`). Usage: `python scripts/draft_assistant.py --state ... --vorp-table ... --strategy season_value --season 2026 --n-sims 300 --data-root data`. **Validation verdict: greedy marginal-value does NOT beat `now_or_never` under the season metric** (wins only slot 12/the turn; loses slot 6 +22.8; ties slot 1; worst of three under starters). **Default stays `now_or_never`.** Root cause: greedy is *myopic* (no pick timing); `now_or_never`'s opportunity-cost layer dominates — edge largest at slot 6 (longest wait), gone at slot 12 (back-to-back). **The numpy fast-path is now SHIPPED** (was deferred): `_vectorized_lineup_points` vectorizes the weekly fill, ~125× (527s→4.2s, byte-identical), pinned by an `optimal_lineup_points` equivalence test — without it the tournament was ≈9 hrs/slot (the spec's "minutes" cost estimate ignored the per-pick strategy MC). **Next slice (now empirically load-bearing, not optional): opportunity-cost layer in season-value space** — `score = marginal − E[marginal of best survivor at position by my next pick]`; A/B-able behind the same protocol; tractable now the fast-path exists. (Other deferred: weekly performance variance, recency-weighted/age availability, playoff weighting all still open from the metric slice; ingest the 2026 `schedules` partition for byes.)
- **Live Draft Assistant — live draft board (Slice 3, the UI) — ✅ DONE** (branch `feat/live-draft-board`; spec/plan `docs/superpowers/specs|plans/2026-06-15-live-draft-board.*`; PM top entry). The last Draft Assistant slice: a Streamlit board over the engine with **co-pilot** (log every pick; opponents get a one-click ADP smart-assist = the `bot_pick`) and **mock** (bot-drafted opponents; "advance to my pick" + optimal-lineup scorecard) modes sharing one three-column board. Testable `LiveDraftSession` controller (`src/projections/draft/assistant/live.py`) + thin `scripts/draft_board.py`; `build_draft_state` extracted from `load_draft_state`; shared `build_session_strategy` seam (cli delegates); autosave/resume to `data/draft_sessions/`; `streamlit` under the new `[ui]` extra. Board offers the 4 production strategies (`season_value_var` excluded — no draft benefit). Usage: `pip install -e ".[ui]"` then `streamlit run scripts/draft_board.py`. **Deferred future seams:** ESPN/Sleeper live-draft API auto-sync; auction/keeper modes; season-value MC scorecard in mock mode; tier-cliff viz. **This closes the Draft Assistant sub-project's planned slices.**
- **`generate_vorp_table` ADP-seam debt** (code-review finding #2 on the draft-hub-consensus PR, 2026-06-09): consensus ADP is attached *outside* `generate_vorp_table` — the CLI left-joins `consensus_adp` onto the returned frame and re-casts `gsis_id` (the merge degrades the pyarrow dtype). Because `VorpTableSchema.consensus_adp` is Optional, a consensus VORP frame validates with or without it, so a future second consensus-mode caller that skips the CLI's post-hoc join silently gets an ADP-less frame → the cheat sheet's `adp_delta` is all-NA with no error. Deeper fix: give `generate_vorp_table` an optional `adp: pd.Series | None = None` parameter and do the join *inside* it before the final validate, so the function is the single source of truth for a consensus VORP frame's shape (also removes the CLI's `gsis_id` re-cast). Deliberately out of scope for the first slice (spec kept the VORP math untouched); revisit when a second consensus caller lands. (Code-review finding #1 — silently-ignored source-mismatched CLI flags — was fixed in the same PR.)
- **Multi-source points consensus — ✅ PARTIAL (ESPN+Sleeper shipped)** (branch `feat/multi-source-projection-blend`; spec/plan `docs/superpowers/specs|plans/2026-06-14-multi-source-projection-blend.*`). Sleeper now contributes a real **stat line** (not just ADP) — `_sleeper_stats_to_statline` maps its raw fields into `STAT_FIELDS`, and `build_consensus` blends ESPN+Sleeper as a per-field mean over **stat-bearing** rows (a `_is_stat_bearing` gate of ≥2 non-null, non-zero fields, so degenerate stubs are excluded). So `projected_points_ppr` is now a 2-source average where both exist. **Why it mattered:** ESPN does not retain full historical preseason season projections — 2023 came back as 1-field stubs (pool was 99/514 with points); Sleeper fills the gap (re-ingest verified: 2021–2025 pools now ~100% with points, R7; ESPN↔Sleeper cross-source r≈0.94–0.96, ratio≈1.0, R8). Validated point-in-time (injured players still show preseason-high projections). **Still open:** add 1–2 *scraped* preseason sources (FantasyPros/CBS/NumberFire, user OK'd scraping) for a ≥3-source consensus and the cross-source spread the distribution-wrapping slice needs.
- **Distribution-wrapping** (now unblocked by the published contract): once ≥2 stat-line sources exist, wrap the consensus into the existing `Distribution` types using the real cross-source spread for floor/ceiling, so the scoring/store layers are reused. (Deferred from this slice deliberately — no synthetic spread from one source.)
- **Cross-source rookie-matching refinement:** the `placeholder_name_key` (now shared) still misses spelling/nickname/hyphen divergence (e.g. "Amon-Ra", nicknames) and can collide two distinct same-name+position players (logged, not prevented). The blend currently relies on the gsis_id ingest already assigned; a scraped third source (no gsis at all) will need this refinement.
- **`pd.concat` `FutureWarning` — ✅ RESOLVED** (branch `feat/multi-source-projection-blend`). Sleeper rows now carry real stat values (no longer all-NA), and `_to_canonical` casts every source frame's numeric columns (`adp`, `espn_draft_rank`, `STAT_FIELDS`) to uniform `Float64` so `pd.concat` needs no dtype inference over all-NA columns. Guarded by `test_refresh_emits_no_all_na_concat_futurewarning` (asserts no `FutureWarning`). ~~`pd.concat(frames, ...)` warns because Sleeper rows carry all-NA stat columns...~~
- Spike-graduation cleanup (surfaced by `/simplify` on the benchmark spike — deferred as out-of-scope for the throwaway scripts, do when promoting to `src/`): (1) the naive-totals filename `reports/season_projection.csv` is a bare literal in `project_season.py` (writer), `benchmark_projections.py`, and `compare_predictions_to_actuals.py` — promote to a shared constant before any CI/Makefile depends on the filename. (2) score-weekly-stats-to-season-total exists in several variants (`benchmark.actual_season_points`, `compare_predictions_to_actuals._actual_ppr_total`, `sanity_check_baseline._realized_ppr_points`) — consolidate onto the canonical `projections.scoring.actual_season_total` helper that PR #51 added to the core. (3) `benchmark._attach_gsis_id` should assert the platform-id column exists and warn (not silently drop) on id_map dedup collisions once it serves more than the current 2 sources.

### 39. Fair weekly start/sit benchmark — does our weekly model beat ESPN weekly? (open)

The spike proved our model can't do preseason, NOT that it's a bad weekly model (capability ≠ accuracy). If we want to know whether to keep it for in-season start/sit (use-case #2), run the fair version: pull ESPN's **weekly** projections (stats array `statSplitTypeId=1`, per `scoringPeriodId`), compare our weekly projection vs ESPN weekly vs weekly actuals at the weekly grain. `scripts/benchmark_projections.py`'s machinery is reusable. Until then, our model's in-season value is unmeasured. Lower priority than #38 (draft is the goal); decide after the consensus layer lands.

**Meta-note from the spike:** the Track 2 feature-probe treadmill (0.004–0.04 fpts/week effects) should stop until a downstream consumer exists whose decisions the projection quality actually moves.

### 1. Explore option D: joint-correlation projections

**Context.** During Projections Core brainstorming we picked option C (full per-player distributions, marginal only). Option D would extend C to model how player outcomes *co-move* — same-game stacks, opponent dependencies, game-script effects. We deferred D because it adds storage and modeling complexity we may not need until DFS tournament work; we want C's schema to make D an additive upgrade rather than a rewrite.

**Why it matters.**
- DFS GPPs (top-heavy tournaments) live and die on correlated ceilings; an uncorrelated "stack" model dramatically underestimates QB+WR1 joint upside.
- Cash-game DFS and start/sit decisions can survive on marginal distributions alone.
- Season-long draft and waiver tools mostly want means and ranks; correlations are nice-to-have, not load-bearing.

**Questions to answer when we explore.**
- *Scope:* which correlations actually move the needle? Likely candidates, in priority order:
  - Same-team QB ↔ pass-catchers (typically ρ ≈ 0.4–0.6 for QB↔WR1)
  - Same-game opposing players (shootouts lift everyone)
  - RB ↔ team defense (negative; if you allowed the opposing RB to score, your D suffers)
  - Weather and pace shared across a game
- *Modeling approaches:*
  - Empirical covariance matrix from historical weekly fantasy points (simple, but noisy and assumes stationarity)
  - Scenario / Monte Carlo from simulated game states (richer, much more code — could lean on `nflfastR` win-probability and play-type models)
  - Factor model: shared "game environment" latent variable (pace, total) plus player-specific noise (compromise)
  - Gaussian copula on marginals from C (clean separation: marginals stay as in C, dependence lives in the copula)
- *Storage:* covariance matrices per slate/week are O(N²); scenario tables are O(N · S) for S draws. Need to pick one before DFS optimizer work.
- *Optimizer interface:* most ILP optimizers accept point projections + ownership; correlated upside requires either a sim-based optimizer or a stacking-rule heuristic on top of ILP. Decide which path.
- *Validation:* how do we measure that correlated projections beat uncorrelated? Backtest against historical DK/FD GPP results — compare uncorrelated lineups vs correlated lineups by realized payout percentile, not just RMSE.

**Inputs / references to gather.**
- `nfl_data_py` play-by-play features useful for game-script modeling (EPA, pace, success rate, win prob).
- Historical DK/FD slate salaries + ownership (for backtest target).
- Existing OSS work: `pydfs-lineup-optimizer`, `pulp`/`cvxpy`-based optimizers, any public correlation matrices.
- Blog/academic refs: RotoGrinders/Fantasy Labs on stacking; any published papers on DFS lineup construction under correlation.

**Definition of done for this exploration.**
A short written recommendation: pick one modeling approach (covariance / scenario sim / factor / copula), one storage format, and a concrete API addition to the C-era projections schema. Include a backtest plan so we know whether D is actually paying off before we commit to building it.

### 2. Plan 2b — remaining position feature builders

**Status:** QB/RB/TE complete in Plan 2b (merged). K and DST split out into TODO #10.

### 3a. Play-by-play ingest (PBP plumbing) — closed in Plan 9 (2026-04-29)

Closed. `src/projections/ingest/pbp.py` ships `refresh_pbp` covering `nfl_data_py.import_pbp_data` for 2018–2024 with a curated 27-column subset. `PbpSchema` lives in `schemas.py`. Opt-in `--run-network` smoke at `tests/test_ingest/test_api_drift.py::test_pbp_api_columns_and_schema` guards against upstream column-rename drift. Real-data ingest in Phase 6 surfaced one float32→float64 dtype drift (16 numeric columns), patched in `pbp.py:_FLOAT64_COLS`. `pbp` keyword arg is threaded through 4 direct-builder scripts (`refresh_features.py`, `train_baseline.py`, `predict_2024.py`, `sanity_check_baseline.py`) and through every per-position `build_<pos>_features` signature with `_EMPTY_PBP` default — currently unused by builders, reserved for the next PBP-driven feature plan.

`scripts/adoption_gate.py` extended with `--baseline-run` / `--candidate-run` dual-run mode (cross-run pairing for feature-set vs feature-set comparisons) — load-bearing for every future feature-class plan.

### 3b. Opp-adjusted EPA-residual feature — DO_NOT_ADOPT (Plan 9 verdict, 2026-04-29)

Plan 9's first PBP-derived feature attempt: schedule-of-strength-adjusted EPA-per-play residual, replacing v1 `opp_allowed_<pos>_fppg_l4`. Adoption gate verdict: **all 4 positions DO_NOT_ADOPT**. QB / RB / TE returned null results (RMSE + Spearman CIs bracket zero); WR returned a small but statistically significant **regression** (RMSE +0.0083 fpts, CI strictly above 0; Spearman -0.0013, CI strictly below 0). Per-position feature changes reverted at commit `941b96c`; PBP plumbing kept (see TODO #3a).

Mechanism interpretation (full discussion in spec §6 "mechanism interpretation"): Ridge baseline is feature-saturated — opponent strength is partially captured by `implied_team_total`, `spread`, and v1 fppg. The marginal lift from explicit schedule-of-strength residual is below the per-cell noise floor. Three follow-up directions:

1. **Different feature class**, not opponent-strength refinement. Volume-oriented PBP features (pace, PROE, air-yards / aDOT distributions, redzone usage shares) target a different signal axis and are more likely to move the needle.
2. **Different model class** that benefits from the residual feature (LightGBM, ensemble) — Plan 9's gate only evaluated BaselineModel.
3. **Compound feature** that combines opp-EPA-residual with another opp-strength signal (defensive personnel, Vegas line) so the per-cell signal-to-noise improves.

Each is a separate plan candidate. None is queued.

**Update 2026-04-30 (option C re-evaluation):** Direction (2) — different model class — closed via 8 lightgbm-nb composite probes against the existing override parquets (4 positions × {swap, augment}, `--force-composite` flag). All 8 cells DO_NOT_ADOPT; lightgbm-nb does not systematically extract more signal than baseline. Position-by-position changes are within the per-cell noise floor (~0.08 fpts) — WR swap goes regression→null, TE swap goes null→regression, others are essentially unchanged. EPA-residual feature is closed across model classes; do not revisit. Reports under `reports/feature_probe_plan9_lgbnb_*.{md,csv}`. Directions (1) and (3) remain open as separate candidates.

**Workflow for each candidate:** generate an override parquet (one column per candidate), run `scripts/probe_feature_signal.py`. Only proceed to a full plan if the probe returns pooled SIGNAL on at least one (position, stat) cell. See `docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md`.

### 3c. Remaining PBP-derived feature plans (open)

Planned slices on top of Plan 9's PBP plumbing, each its own (position, builder) extension:

- Team pace (plays per 60 minutes neutral)
- PROE (pass rate over expected, game-state adjusted)
- Player-level air yards / aDOT / target depth distributions
- Pressure rate allowed by O-line (proxy for QB sack risk and rushing-yardage-on-scramble)
- Red-zone usage shares (separate from full-field share)

Brainstorm in a focused session before scoping. Plan 9's negative result on opp-EPA-residual at the BaselineModel level argues for evaluating these against LightGBM / ensemble in addition to BaselineModel, since model class may dominate over feature class for marginal signals.

Estimated cumulative win: still potentially 5-15% RMSE if features compose well; treat each slice's individual gate result as the truth (per Plan 8 + Plan 9's lessons). Apply the 5-15% family-level prior at the family level — bundle 3-4 candidates into one probe + adoption gate, not one feature at a time.

**Workflow for each candidate:** before scoping a plan, generate an override parquet (one column per candidate) and run `scripts/probe_feature_signal.py`. Only proceed to a full plan if the probe returns pooled SIGNAL on at least one (position, stat) cell. See `docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md`.

**Update 2026-04-30 (PBP family probe, branch `feat/probe-pbp-family`):** First family-level probe shipped per `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`. Bundled four PBP team-level features (`pace_l4`, `proe_l4`, `team_ayps_l4`, `team_def_epa_resid_l4`) into a single override and probed in two modes (augment + swap) at the BaselineModel level. **Family verdict: `SIGNAL`** — Phase 1 pooled SIGNAL on `(RB, rushing_yards)` in both modes; Phase 2 ADOPT on RB in both modes (composite RMSE delta ~-0.012 fpts, CI strictly below 0). Signal concentrated on RB; QB regresses on `passing_yards` in augment mode (+0.45 fpts); WR/TE net-zero. Conditional lgb-nb runs skipped per spec §3.2 (baseline already returned SIGNAL). Greenlights a follow-up production-builder plan: RB-only integration first; WR/TE deferred to a separate refined-unit spec (player aDOT for receivers, per-position EPA-residual à la Plan 9). See `reports/feature_probe_pbp_family_summary.md` for the per-mode table and decision.

**Update 2026-05-01 (RB PBP features integration, branch `feat/rb-pbp-features`):** Production integration of the 4 PBP team-level features into `RbFeaturesSchema` + `build_rb_features` per `docs/superpowers/specs/2026-05-01-rb-pbp-features-design.md`. Dual-run gate verdict on `(BaselineModel, RB)`: **`ADOPT`** (composite RMSE delta -0.0124 fpts, CI [-0.0255, -0.0006]). Probe predicted -0.0124 fpts; gate matched the point estimate to 4 decimal places. Shipped. Other model classes (lightgbm-tuned, lightgbm-nb, ensemble) deferred to a follow-up — informational per spec §1.3.5, not gating. WR / TE remain open for a separate refined-unit spec (player aDOT for receivers, per-position EPA-residual). QB explicitly excluded per PR #20's augment-mode regression. **Spec gap caught + fixed:** `baseline.py:_RB_FEATURE_COLUMNS` is hardcoded and was not updated by the spec — the lightgbm family derives feature lists from the schema dynamically, but baseline does not. Fixed at commit `9895dee`. Future "add feature to position X" specs should include a model-feature-list checklist item. See `reports/rb_pbp_features_summary.md`.

**Update 2026-05-01 (WR/TE PBP receiver-level family probe, branch `feat/wr-te-pbp-features`):** Player-level air-yards / aDOT family probe shipped per `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md`. Bundled four player-level PBP features (`aDOT_l4`, `deep_target_share_l4`, `yac_per_reception_l4`, `red_zone_target_share_l4`) into a single override and probed in two modes (augment + swap) at the BaselineModel level + lightgbm-nb level (with `--force-composite`) for WR + TE only. **Family verdict: `NULL` (durable)** — no pooled Phase 1 SIGNAL across any of 4 mode×model reports; no Phase 2 ADOPT or MARGINAL across the 2 lgb-nb composite reports. Closest cell: WR swap composite RMSE -0.0052 fpts CI [-0.012, +0.001] DO_NOT_ADOPT (CI just barely brackets zero). Family closed at the trailing-4-receiver-active-games unit. **Refined-unit candidates beyond air-yards / aDOT remain unexplored:** per-route-concept distributions (data not in curated PBP), target-quality residuals (would need per-throw difficulty modeling), in-line vs flexed for TE (not in PBP). None queued; revisit only with independent evidence the unit choice was the binding constraint. **Spec gap caught:** spec §3.2 prescribed lgb-nb runs without `--force-composite`, but Phase 1 is RidgeCV-only regardless of `--model`, so bare lgb-nb runs were tautological with baseline. Re-ran with `--force-composite` to actually test lgb-nb at composite. Same gap exists in PR #20's spec §3.2; not exercised because RB returned baseline SIGNAL. See `reports/feature_probe_pbp_receiver_summary.md`.

**Update 2026-05-02 (PBP red-zone team-level family probe, branch `feat/probe-pbp-redzone`):** Red-zone-context team-level family probe shipped per `docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md`. Bundled four team-level PBP features (`team_rz_pace_l4`, `team_rz_pass_rate_l4`, `team_def_rz_epa_allowed_l4`, `team_def_rz_pass_rate_allowed_l4`) at the standard `yardline_100 ≤ 20` cut and probed in two modes (augment + swap) at BaselineModel + lgb-nb (`--force-composite`) for all 4 positions. **Family verdict: `NULL` (durable)** — 0 pooled Phase 1 SIGNAL cells across all 4 mode×model reports; 0 Phase 2 ADOPT/MARGINAL across the 2 lgb-nb composite reports (8 verdict cells, all DO_NOT_ADOPT). The only directional cell is **QB augment lgb-nb composite REGRESSION** at RMSE +0.0268 fpts CI [+0.0082, +0.0449] (CI strictly above 0); other 7 Phase 2 cells bracket zero with point estimates near zero. Predicted mechanism (TD efficiency) not observed — no `*_tds` cell fires SIGNAL. Family closed at the RZ-broad cut. **Refined-unit candidates beyond `yardline_100 ≤ 20` remain unexplored** but are unlikely to clear what RZ-broad couldn't: goal-line (`≤ 5`), per-stat splits (un-bundling reverses the family-level prior framework), RZ-restricted EPA-residual (correlates with already-shipped full-field `team_def_epa_resid_l4`). None queued. **Coverage caveat:** `--coverage-threshold 0.90` used because the probe's hardcoded check is pooled and the structural 2018 cold-start drags it to 94.7%; per-season 2019–2024 is uniformly ≥96.9% across all positions, so the eval window itself satisfies spec §1.3 criterion 1. Same precedent gap as PR #22 (which used 0.70). **What remains open in TODO #3c:** the third unexplored team-level family — pressure rate allowed by O-line (sack-rate-allowed and scramble-rate proxies, available in curated PBP without ingest extension). Different mechanism axis (offensive-line proxy, not TD distribution); a future bundled probe is the natural next slot. See `reports/feature_probe_pbp_redzone_summary.md`.

**Update 2026-05-02 (PBP pressure team-level family probe, branch `feat/probe-pbp-pressure`):** Pressure-context team-level family probe shipped per `docs/superpowers/specs/2026-05-02-pbp-pressure-feature-family-probe-design.md`. Bundled four team-level PBP features (`team_sack_rate_allowed_l4`, `team_qb_scramble_rate_l4`, `team_def_sack_rate_l4`, `team_def_scramble_rate_l4`) at the canonical `qb_dropback == 1` denominator and probed in two modes (augment + swap) at BaselineModel + lgb-nb (`--force-composite`) for all 4 positions. **Family verdict: `NULL` (durable)** — 0 pooled Phase 1 SIGNAL cells across all 4 mode×model reports (480 total cells); 0 Phase 2 ADOPT/MARGINAL across the 2 lgb-nb composite reports (8 verdict cells, all DO_NOT_ADOPT). Two directional Phase-2 cells: **QB augment lgb-nb composite REGRESSION** at RMSE +0.0276 fpts CI [+0.0077, +0.0472] (CI strictly above 0 — same pattern as PR #23's QB augment regression at +0.0268), and **TE swap lgb-nb composite Spearman regression** at -0.0034 CI [-0.0067, -0.0002] (rank only; RMSE brackets zero). Predicted mechanism (QB-side pressure exposure → passing_yards / sacks / rushing_yards) not observed. Family closed at the dropback-denominator cut. **Refined-unit candidates beyond `qb_dropback == 1` remain unexplored** but are unlikely to clear what the broad cut couldn't: alternate denominators (`pass_attempts + sacks` only — narrower than dropback), goal-line / 3rd-down / two-minute pressure subsets. None queued. **Coverage:** default `--coverage-threshold 0.95` passed cleanly (pooled 96.6%; 2019–2024 uniformly 100%; 2018 cold-start 24.2% NaN, but eval window 2021–2024 unaffected). No threshold relaxation needed. **This closes Track 2A — all three TODO #3c team-level PBP families have now been probed:** PR #20 (pace/PROE/AYPS/EPA-resid → SIGNAL via RB, integrated in PR #21), PR #23 (red-zone → NULL), this PR (pressure → NULL). Remaining TODO #3c open items are receiver-level / refined-unit candidates (PR #22's air-yards / aDOT family also returned NULL; deeper unit choices like per-route-concept distributions need ingest extensions). See `reports/feature_probe_pbp_pressure_summary.md`.

### 4. Feature parquet storage — closed in Plan 3c

Closed 2026-04-26. `data/features/{position}/season=YYYY/week=WW/part.parquet`
populated by `scripts/refresh_features.py`; read by `src/projections/features/cache.py`
and consumed by the backtest harness. Manual invalidation only — see TODO #21
below for code-hash auto-invalidation.

### 5. NGS missing-data forward-fill policy

v1 leaves NaN. Revisit after a notebook investigation against a recent season quantifying how often qualifying-threshold misses happen and whether forward-fill changes feature distributions materially.

### 6. Opening / week-of Vegas line source

`import_schedules` returns *closing* lines. Closing is fine for backtest. Only worth pursuing if Plan 5 ever projects pre-week selections (e.g., DFS workflow uses lines that change through the week).

### 7. Depth chart slot-label parser refinement

v1 extracts the trailing digit from labels like `WR1`, falling back to `1` for unrankable labels (`LWR`/`RWR`/`SWR`) with a warning. If Plan 3 model fitting shows `depth_rank` is noisy or wrong, build a richer parser using alignment + rank.

### 9. WR feature builder edge cases for production data

Issues flagged during Task 15 / final code review that don't manifest on the synthetic fixtures but could surface in real `nfl_data_py` data:

a) `is_home` and `roof_dome` are non-nullable in `WrFeaturesSchema` but the schedule join is a left-merge — if a depth-chart team has no schedule row in the target week (bye week, missing future game), validation fails. Fix: filter rostered teams to only those with schedule rows, OR mark the columns nullable. Revisit when Plan 3 wires real data.

b) `IdMapSchema.pfr_id` is not marked `unique=True`. The snap_counts ingest does an inner-join on pfr_id; duplicate pfr_ids in id_map would multiply rows. Add `unique=True` to `pfr_id` (and to `espn_id`/`sleeper_id` for symmetry) as defense-in-depth, or add `.drop_duplicates(subset=["pfr_id"])` in the snap_counts join helper.

c) `_trailing_4_share_per_team` in `features/wr.py` groups by `(gsis_id, team)`, which produces two rows for any player traded mid-season. The downstream merge in `build_wr_features` joins on `gsis_id` only, so duplicates would propagate into the output. Not exercised by the synthetic fixtures (no traded players). Fix: filter `last4_player` to only the player's *current* team before computing the share, or merge on `(gsis_id, team)` so only the matching team's row survives.

### 10. Plan 2c — K and DST feature builders

Both positions need data we don't currently ingest:

- **K**: spec calls for "recent FG distance distribution" and "opp redzone TD allowed %." Neither is in `WeeklyStatsSchema`. Need to ingest a new source covering FG attempts by distance and accuracy by range. `nfl_data_py.import_weekly_pfr_data` may have this — verify before designing.
- **DST**: team-level not player-level. Schema's primary key is `Team`, not `GsisId` — fundamentally different from the per-player pattern Plan 2a/2b established. Intended features (opp pass-block win rate, sack rate allowed, turnover-worthy throw rate) all need play-by-play (TODO #3).

Decision before brainstorming Plan 2c: do we ingest the missing data first (extending the ingest layer), or build degraded v0 K/DST features from `implied_team_total` alone? The latter is fast but creates a future rewrite; the former takes longer but yields the right shape.

Plan 3 (Model A baseline) doesn't depend on K/DST, so this can run in parallel.

### 11. Bound `percent_attempts_gte_eight_defenders_std`

Surfaced in PR #4 review (latent issue #5). The field on `RbFeaturesSchema` currently has only `nullable=True` — no `ge`/`le`. NGS reports the metric as a 0–100 percentage, so a `ge=0, le=100` bound would catch a unit-mismatch upstream (e.g., if a future ingest change accidentally emits the same value as a 0–1 fraction). One-line fix; defer until any other RB schema work to keep the diff scoped.

### 12. Lift `rushing_qb` / `passing_down_back` to consumption time

Surfaced in PR #4 review (latent issue #6). Both are thresholded booleans baked into the persisted feature schema (`rushing_qb` = `rushing_attempts_per_game_l4 >= 5.0`, `passing_down_back` = `targets_per_game_l4 >= 4.0`). Decision logged in the Plan 2b plan as "rough heuristics from feel," but the threshold is now fixed at the producer side, so a downstream consumer (a model wanting a different cut, or a categorical instead of a boolean) has to recompute from the underlying `*_l4` column anyway. Two options:

- Drop the boolean from the schema and compute at use time (cleanest; consumer holds the policy).
- Keep but document explicitly that this threshold is the canonical league-wide convention and shouldn't be re-derived elsewhere.

Revisit before Plan 3 model fitting — if the model never uses these booleans, just remove them.

### 13. Per-row seed derivation in BaselineModel.predict_distribution

**Closed in Plan 3d.** `derive_row_seed` in `scoring/score_distribution.py` produces a stable 32-bit seed from `(gsis_id, season, week, ruleset.name)` via sha256; `predict_distribution` and `aggregate_to_season` both consume it. Determinism verified by re-running `--check` immediately after `--update-snapshot` in Phase 6 (closes TODO #19 by demonstration).

### 14. ProjectionWeeklySchema params blob carries summary, not samples

**Closed in Plan 3d.** New `DistributionFamily.SAMPLED_SUMMARY` enum value; `params` now encodes per-stat distribution parameters via `pack_per_stat_params` (codec in `distributions/codec.py`). Three orders of magnitude smaller than persisting full sample arrays; deterministic regeneration via the per-row seed makes samples available on demand.

### 16. Real-data drifts not caught by synthetic-fixture tests

Surfaced during Plan 3a Tasks 14-17. The synthetic fixtures used by 2a/2b/3a's CI tests don't exercise the real `nfl_data_py` API surface. Eight ingest/feature drifts had to be patched live during Plan 3a's first real-data pull:

1. `weekly_stats`: `fumbles_lost` had to be derived from three source-specific columns (no aggregated column upstream).
2. `weekly_stats`/`depth_charts`/`ngs`/`snap_counts`: int32 vs int64 dtype mismatch on `season`/`week`.
3. `depth_charts`/`ngs`/`snap_counts`: NaN season/week rows had to be filtered before int coercion.
4. `ngs`: pro-bowl/all-star weeks (>22) and season-summary rows (week=0) had to be filtered.
5. `id_map`: 16+ pro-football-reference 3-letter team aliases (GBP, KAN, NWE, NOR, SDG, TAM, etc.).
6. `id_map`: "FA"/"FA*" (free agent) team codes had to be handled as None.
7. `id_map`: malformed legacy gsis_ids (PFR-style strings) had to be filtered.
8. `wr.py`: bye-week WRs without schedule rows; duplicate depth-chart entries; negative trailing-mean yardage; share calc going negative.

The opt-in `pytest -m network --run-network` smokes (`tests/test_ingest/test_api_drift.py`, formerly TODO #8 — closed) now guard against the same class of column-rename / column-removal drift after a `nfl_data_py` version bump. They do NOT replace this drift list as historical context, and they will NOT catch every real-data edge case (some — like the `id_map` malformed-legacy-gsis-id rows — are data-quality issues per row, not column-level drift), so keep this entry as a record of what the synthetic fixtures missed and audit it after each `nfl_data_py` upgrade.

**Additional Plan 3b real-data drifts (Phase 6):**

9. `WrFeaturesSchema` / `QbFeaturesSchema` / `RbFeaturesSchema` / `TeFeaturesSchema`: `*_yards_per_game_l4` and `passing_yards_per_game_std` had `ge=0` bounds inconsistent with the underlying weekly_stats schema (which allows negative yards from sacks / TFL / kneels). Trailing-4 means can therefore be negative; bounds dropped in commits `fa864ac` + `e25eb57`.
10. QB/RB/TE feature builders missing the bye-week filter on rostered teams (analogous to WR's TODO #9a). Players on a team with no schedule row in `as_of_week` produced NaN `opponent`/`is_home`/`roof_dome` and failed schema validation. Filter ported from `wr.py` in commit `f79806a`.
11. QB/RB/TE feature builders missing the depth-chart dedupe (analogous to WR's TODO #9c). Players listed under multiple slots or traded mid-week produced duplicate `(gsis_id, season, week)` rows and failed `BaselineModel.fit`'s one-to-one merge. Dedupe ported from `wr.py` in commit `54b6d95`.

The TODO #8 opt-in network smokes confirmed no upstream column-rename drift in this run; the new entries are all schema-bound or builder-edge-case mismatches between WR's already-hardened path and QB/RB/TE's pre-3b path. The four fixes above bring QB/RB/TE feature builders to parity with WR.

### 18. Add `python -m projections.ingest.refresh` CLI entry point

Surfaced during Plan 3b Phase 6 ingest. `src/projections/ingest/refresh.py`
exports a `refresh()` function but has no `if __name__ == "__main__":` block,
so `python -m projections.ingest.refresh ...` doesn't work and ingest must
be invoked via `python -c "from projections.ingest.refresh import refresh;
refresh(data_root=Path('data'), seasons=range(2018, 2025))"`. A small
argparse `main()` (with `--seasons RANGE` / `--data-root PATH` and a
sensible default for `data_root=Path('data')`) would make the canonical
ingest invocation a one-liner. Defer until next ingest-touching plan
(Plan 4's CLI verbs are the natural home).

### 19. Walk-forward gate non-determinism check

**Closed in Plan 3d.** With deterministic per-row seeds, re-running `python scripts/backtest.py --check` immediately after `--update-snapshot` produces zero drift. No `random_state` propagation needed inside `RidgeCV` because the regression itself is deterministic; non-determinism only entered through `score_distribution`'s seed.

### 20. Naive-baseline parquet output for trend tracking

Plan 3c writes naive metrics into the in-memory `BacktestRun` and prints
them in `--report` mode but does not persist them. If we ever want to
track "how much value is Model A adding over naive *over time*", persist
naive metrics to a parquet table at `data/backtest/naive_history/...`
keyed by run timestamp. Not load-bearing for v1.

### 21. Feature cache code-hash auto-invalidation

Plan 3c's feature cache is invalidated manually — the user must re-run
`scripts/refresh_features.py` after touching any feature builder.
Auto-invalidation reads the source files for the feature builder (the
same set `BaselineModel.code_hash_files` already tracks) and refuses
to read stale cache. Deferred until manual invalidation produces a
real-world bug.

### 22. Plan 3e — calibration tightening — closed in Plan 3e Phase 3 (Phase 3 routing subsequently reverted)

Closed 2026-04-27. Phase 0 diagnostic identified 3 root causes; Phases 1-3 implemented:
- Phase 1: ParametricNegativeBinomial for `*_tds` / interceptions / fumbles_lost (10 cells; weekly mean coverage 0.726 → 0.733; season mean 0.461 → 0.428).
- Phase 2: ParametricStudentT for `*_yards` — ATTEMPTED + REVERTED. Student-t with the data's tail shape narrows [p10, p90] coverage vs NORMAL despite Phase 0's AIC preference (AIC measures full-distribution fit, not p10/p90 coverage). Infrastructure preserved for future use.
- Phase 3: per-tertile variance bucketing (cross-cutting; applies to NORMAL/GAMMA/NB).

Final coverage (Phase 3 vs Plan 3d at `fe55d5b`): weekly mean 0.726 → 0.710 (-0.016); season mean 0.461 → 0.399 (-0.062); all-32-cells mean delta -0.039; min cell coverage 0.293 (was 0.313). Spec targets (min ≥ 0.65, mean delta ≥ +0.10): **NOT met**. QB cells gained ~+0.02 weekly across all 4 years (only positive movers); RB/WR/TE regressed because their residuals are sharply heteroscedastic and bucketing narrows the central interval where the actuals don't tighten.

Follow-up plan candidates (pick one in post-merge brainstorming):
1. Revert Phase 3 routing (keep mechanism + tests) — Phase 1 snapshot had better mean coverage than Phase 3.
2. ZIP (zero-inflated Poisson) for count cells if NB still undercovers — handles zero mass directly rather than via dispersion.
3. Cross-week residual correlation modeling for season under-dispersion — independent weekly draws understate season variance.
4. Calibration-aware fitting — fit variance to minimize p10/p90 quantile loss directly rather than maximize residual likelihood.

**Update 2026-04-27 (post-Phase-3 revert):** Took option (1) above. Phase 3 routing reverted in `BaselineModel.fit` + `build_stat_distributions`; bucketing helpers, widened `variance_params` type signature (`float | list[float]`), and unit tests preserved as future infrastructure for quantile-based fitting. Snapshot returns to Phase 1 baseline (commit `0078223`) bit-for-bit. **Final shipped state for Plan 3e: Phase 0 (diagnostic CLI) + Phase 1 (NB for count stats).** Phase 2 + Phase 3 are both attempted-and-reverted with infrastructure preserved. Spec calibration targets remain unmet by the shipped state; the canonical follow-up plans (ZIP, cross-week correlation, calibration-aware fitting) stay open for post-merge brainstorming.

**Update 2026-04-27 (post-merge brainstorm):** TODO #22 stays closed. Post-Plan-3e empirical investigation (lag-k autocorrelation of standardized residuals) showed week-to-week persistence is weak (lag-1 ρ in [+0.02, +0.10]; lag-2+ ≈ noise) and AR(1) only explains 5-10% of the season variance gap. Cross-week correlation modeling is therefore not the high-leverage next step. Calibration-aware fitting risks distorting the upper tail (load-bearing for DFS GPP) while fixing the central interval. **Decision: stop calibration tightening; pivot to mean-prediction improvements via the three model-improvement tracks documented in TODOs #3 / #23 / #26 and Plan 5 in the project_management.md backlog.** The remaining season-coverage shortfall is acknowledged as a known limitation — none of the planned downstream tools (Draft Hub, start/sit, DFS) actually consume a calibrated season `[p10, p90]`.

### 23. Target decomposition (volume × efficiency)

Currently each fantasy-relevant stat is predicted directly (e.g., `receiving_yards` is one regression target). Decompose into volume × efficiency components and predict the factors separately, then multiply for the composed prediction:

- **WR / TE**: `targets × catch_rate × yards_per_reception` for yards; `targets × td_rate_per_target` for TDs
- **RB**: `carries × yards_per_carry` + `receptions × yards_per_reception`; per-touch TD rate
- **QB**: `dropbacks × completion_rate × yards_per_completion` + sack/scramble adjustments; per-attempt TD/INT rate

Each factor has different drivers (volume is team-driven; efficiency is player-driven), so smaller specialized sub-models on each factor often beat one combined model. Notable secondary win: TD modeling becomes much better — currently a noisy 0/1/2 prediction with low signal; decomposed it becomes (red-zone touches) × (RZ TD rate), each with more identifiable signal.

**Refactor scope:** feature builders gain decomposed targets; `BaselineModel` (or its successor) trains per-factor sub-models and multiplies. Distribution composition (product of independent distributions) needs to be re-derived at the scoring layer; calibration metrics still apply to the composed prediction. `ProjectionWeeklySchema` may need optional per-factor params for diagnostics.

**Estimated win:** 3-10% RMSE independent of model class. Best evaluated alongside a stable model class (either current Ridge or Plan 5 LightGBM) so the win is attributable. One of the three model-improvement tracks identified in the post-Plan-3e brainstorm (2026-04-27).

**Update 2026-05-10 (target decomposition probe, branch `feat/probe-target-decomposition`):** First model-architecture probe in the project shipped per `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md`. Bundled three WR receiving stats (`receptions`, `receiving_yards`, `receiving_tds`) decomposed as `targets × {catch_rate, yards_per_target, td_rate_per_target}` and probed via a new lite walk-forward CV harness (`src/projections/backtest/target_decomposition_probe.py`). RidgeCV on every sub-model + direct comparator (matches `BaselineModel.fit`'s algorithmic family) so verdicts are attributable to decomposition itself. **Family verdict: SIGNAL (marginal) — receptions only.** Per-stat: receptions SIGNAL (Δ-RMSE -0.0042 fpts, CI [-0.0079, -0.0004], n=8460); receiving_yards NULL (CI brackets zero by ±0.05 fpts); receiving_tds NULL (essentially zero point estimate). Coverage strictly above 0.95 threshold across all eval years (no relaxation). Factor residual orthogonality clean (|ρ| < 0.05 across all 12 (stat × year) cells, well under the 0.2 §5 risk #2 caveat threshold) — strongest mechanism-level finding from the probe; decomposition cleanly separates volume from efficiency signal axes. Greenlights integration plan with named follow-ups: `DecomposedBaselineModel` peer + `ProductDistribution` + coherent within-row sampling + composite-fpts adoption gate vs production `EnsembleModel`. Factor-appropriate sub-models (logistic / Gamma / Poisson-NB2) deliberately deferred to a separate probe + integration cycle. §5 risk #1 caveat flag on the binding cell: composite-fpts Δ -0.0042 fpts is just below the ~0.005 fpts threshold — integration plan's go/no-go must weight CI strength against the small magnitude. Conservative implementation option: opt in to the SIGNAL stat (receptions) only at the model level, leaving receiving_yards and receiving_tds on direct ridges. RB / QB / TE decomposition probes remain open under this TODO; require independent mechanism evidence before re-probing on those positions. See `reports/feature_probe_target_decomposition_summary.md`.

**Update 2026-05-13 (WR target decomposition integration, branch `feat/wr-target-decomposition`):** Production integration shipped per `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`. `DecomposedBaselineModel` (peer to `BaselineModel`) with per-stat decomposition opt-in via constructor arg; `wr_decomposed_baseline` factory registered for receptions-only v1 config. Binding gate `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)` verdict: **DO_NOT_ADOPT** (composite RMSE Delta +0.0109 fpts, CI [-0.0080, +0.0285]). Informational gate `(DecomposedBaselineModel, WR)` vs `(BaselineModel, WR)` verdict: **ADOPT** (composite RMSE Delta -0.0103 fpts, CI [-0.0145, -0.0060]). §1.3.5 outcome: infrastructure-only ship -- production routing stays on ensemble. Recommended next direction: swap `BaselineModel -> DecomposedBaselineModel` inside `EnsembleModel`'s child A factory and re-fit ensemble weights (spec when ready). See `reports/wr_target_decomposition_summary.md`.

**Update 2026-05-15 (WR ensemble-decomposed-child swap, branch `feat/wr-ensemble-decomposed-child`):** Production WR routing flipped to `ensemble-decomposed` (new `wr_ensemble_decomposed()` factory wires `EnsembleModel`'s child A to `wr_decomposed_baseline`). Spec at `docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md`. Binding gate `(EnsembleModel-with-decomposed-baseline-child, WR)` vs `(EnsembleModel, WR)` verdict: **ADOPT** (composite RMSE Δ -0.0038 fpts, CI [-0.0079, -0.0002] strictly negative; Spearman Δ +0.0002 neutral, lo_95 above -0.020 floor). Informational gate `(EnsembleModel-with-decomposed-baseline-child, WR)` vs `(DecomposedBaselineModel, WR)`: DO_NOT_ADOPT on RMSE (CI [-0.0234, +0.0089] brackets zero) but Spearman Δ +0.0041 strictly positive — ensemble's lgb-nb mixing adds rank-correlation lift even on top of decomposition. §1.3.5 outcome: routing flipped (`_PositionDispatch[Position.WR].default_model_class` → `"ensemble-decomposed"`). Marginal-zone flag fires on the binding magnitude (|Δ| < 0.005 fpts), but CI is strictly negative so the flip is statistically conclusive. Side-effect fix in commit `975cd52`: `QuantileDistribution.cdf` now extrapolates past knots (mirrors `quantile()`), required for `MixtureDistribution.quantile()` brentq inversion to bracket tail q values when one component is a QuantileDistribution. Recommended next direction (carries this TODO forward): **factor-appropriate sub-model classes for `catch_rate`** (logistic-link) — would lift the small adoption magnitude into the comfortable zone, address the [0, 1]-bounded ratio's tail-calibration weakness, and make decomposing `receiving_yards` / `receiving_tds` viable. See `reports/wr_ensemble_decomposed_summary.md`.

**Update 2026-05-16 (logit catch_rate probe, branch `feat/probe-logit-catch-rate`):** Factor-appropriate sub-model class probe shipped per `docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md`. New module `src/projections/backtest/logit_catch_rate_probe.py` walks forward over 2021-2024 on real WR data with two arms: (Incumbent) RidgeCV on `catch_rate` ratio with sample-time clip `[0, 1]` (matches current production via PR #36/#38); (Candidate) `LogisticRegressionCV` via Bernoulli-trial row expansion + `StandardScaler` pipeline. Both arms share the same RidgeCV on `targets`; only the catch_rate efficiency sub-model class differs, so any verdict is attributable to the class swap itself. **Per-stat receptions verdict: NULL** (pooled RMSE Δ -0.0018 receptions, 95% CI [-0.0047, +0.0009], n_paired = 5195). Magnitude flag fires (|Δ| 0.0018 < 0.005 receptions per PR #31 retrospective rule) — but moot because the CI brackets zero on the upper side regardless. Per-year breakdown: 2021 -0.0021 [-0.0091, +0.0046], 2022 -0.0052 [-0.0107, +0.0007] (closest to a per-year SIGNAL but still NULL), 2023 +0.0011 [-0.0047, +0.0062], 2024 -0.0018 [-0.0066, +0.0029]. Coverage 0.989-0.996 across all years, comfortably above the 0.95 threshold — verdict not muddied by the `targets > 0` filter. **Mechanism conclusion:** the [0, 1]-bounded ratio's tail-calibration weakness flagged in the 2026-05-15 update is not large enough on real WR data to justify a logit class swap. Recommended next direction: close the catch_rate factor-appropriate direction; next factor-appropriate-class slot is `yards_per_target` (log-link Gamma / Tweedie family) under the same shared-volume / single-factor-swap design pattern — higher-leverage because `yards_per_target` carries more receiving_yards variance than `catch_rate` carries receptions variance, and a Gaussian-on-ratio Ridge is a worse approximation to a Gamma response than to a Bernoulli mean. See `reports/feature_probe_logit_catch_rate_summary.md`.

**Update 2026-05-16 (Tweedie yards_per_target probe, branch `feat/probe-tweedie-yards-per-target`):** Factor-appropriate sub-model class probe shipped per `docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md`. New module `src/projections/backtest/tweedie_yards_per_target_probe.py` walks forward over 2021-2024 on real WR data with two arms: (Incumbent) RidgeCV on `yards_per_target` ratio with predict-time clip `>=0` (matches the unbounded-efficiency code path in `DecomposedBaselineModel`); (Candidate) `TweedieRegressor(power=1.5, link="log")` with alpha CV-selected via `GridSearchCV` over a 7-point log-spaced grid, wrapped in a `StandardScaler` pipeline (scaling required because Tweedie's L2 penalty is scale-dependent). Both arms share the same RidgeCV on `targets`; only the yards_per_target efficiency sub-model class differs. **Per-stat receiving_yards verdict: NULL** (pooled RMSE Δ -0.0121 yards, 95% CI [-0.0564, +0.0353], n_paired = 5195). Composite-fpts equivalent -0.0012 fpts. Magnitude flag fires (|Δ| 0.0121 < 0.050 yards per PR #31 retrospective rule, equivalent to <0.005 fpts) — moot because the CI brackets zero on the upper side regardless. Per-year breakdown: 2021 +0.034 [-0.062, +0.134], 2022 -0.036 [-0.136, +0.070], 2023 +0.025 [-0.063, +0.112], 2024 -0.075 [-0.163, +0.007] (closest to a per-year SIGNAL but still NULL). Coverage 0.989-0.996 across all years, comfortably above 0.95. **Plan-vs-execution deviation:** Tweedie deviance requires y >= 0; ~0.6% of WR `targets > 0` rows have negative `receiving_yards` (real-data laterals / lost yards on receptions). Efficiency-fit row mask tightened to `(targets > 0) & (yards >= 0.0)` on BOTH arms; eval rows unfiltered. Drops <1% of training rows per fold; bias symmetric, cancels in paired-Δ-RMSE. **Mechanism conclusion:** Tweedie's compound-Poisson-Gamma shape on yards_per_target does NOT materially outperform the Ridge-on-ratio + clip(>=0) approximation on the WR data we have. With both catch_rate (NULL, PR #39) and yards_per_target (NULL, this PR) closed factor-appropriate, the factor-class-swap line of attack on WR receiving has produced two NULLs in a row — the recipe-change axis (decomposition itself, PR #36/#38) carries more weight than the sub-model-class axis on this data. Recommended next direction: close the yards_per_target factor-appropriate direction; next factor-appropriate-class slot is `td_rate_per_target` (Poisson or logistic) on a separate cycle. See `reports/feature_probe_tweedie_yards_per_target_summary.md`.

**Update 2026-05-16 (RB decomposition probe, branch `feat/probe-rb-decomposition`):** RB-position decomposition probe shipped per `docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md`. New module `src/projections/backtest/rb_decomposition_probe.py` walks forward over 2021-2024 on real RB data with two shared volume axes (carries, targets) x 5 composed stats: rushing_yards + rushing_tds on carries, receptions + receiving_yards + receiving_tds on targets. Sub-model = RidgeCV everywhere on both arms (decomposition-only test; factor-appropriate sub-models per spec §1.4 #3 are conditional on a Ridge-vs-Ridge SIGNAL here). **Per-stat verdicts: 5x NULL.** rushing_yards Δ -0.0931 (CI [-0.1915, +0.0058], -0.0093 fpts), rushing_tds Δ +0.0010 (CI [-0.0014, +0.0033], +0.0062 fpts), receptions Δ -0.0004 (CI [-0.0022, +0.0016], -0.0004 fpts MARGINAL), receiving_yards Δ -0.0344 (CI [-0.0850, +0.0150], -0.0034 fpts MARGINAL), receiving_tds Δ -0.0003 (CI [-0.0012, +0.0006], -0.0017 fpts MARGINAL); n_paired = 3291 per stat. **Coverage:** carries > 0 rate 0.9638-0.9761 across 2021-2024 (above 0.95 every year); targets > 0 rate 0.7799-0.8518 (**BELOW THRESHOLD for all 4 eval years** — structural for RBs, a legitimate observation of low-volume / out-of-rotation usage rather than a data-quality flaw; receiving-stat verdicts rest on ~80% of eval rows). **Mechanism conclusion:** decomposition with RidgeCV on every sub-model is statistically indistinguishable from direct RidgeCV on RB rushing AND receiving stats. Two distinct mechanistic stories rule out: (a) volume / efficiency separation does not by itself expose RB-specific signal direct RidgeCV misses, and (b) PR #32's marginal WR-receptions SIGNAL does NOT generalize to RB even on the same stat (RB receiving-volume targets too sparse and too correlated with rushing-volume features). Plan-vs-execution deviation: cached RB features for 2018-2020 predated commit `09e0d76`'s weather columns; ran `scripts/refresh_features.py rb --seasons 2018-2024` once (85 s) before the probe (5.4 s). **Recommended next direction:** close the RB decomposition direction at this Ridge-only unit; no integration plan greenlit per spec §4 "all 5 NULL" branch. Factor-appropriate RB sub-model probes are NOT next per spec §1.4 #3 (conditional on at least one RB Ridge-vs-Ridge SIGNAL, none here). With logit catch_rate (PR #39), Tweedie yards_per_target (PR #44), and now RB decomposition all NULL, the decomposition-and-factor-class axis is empirically exhausted without independent mechanism evidence. Higher-leverage slots: (1) refined-unit feature work under TODOs #24 / #25; (2) different mechanism families. See `reports/feature_probe_rb_decomposition_summary.md`.

### 24. Player-trajectory features (age curves, career arc, trend gradients)

Current model sees only trailing-N levels (e.g., `targets_per_game_l4`). It doesn't see *trajectories* — is the player rising or declining, and where in their career arc are they?

Candidate additions:
- Age (years since draft year + offset for late-bloomers)
- Position-specific aging-curve features (RBs decline ~age 28; WRs peak 25-29; TEs peak later)
- Trend signals: `targets_per_game_l4 - targets_per_game_l8` (rising vs declining usage)
- Year-over-year change indicators: 2nd-year breakout flag, 3rd-year-leap flag for QBs
- Snap-count gradient: % change in snaps over last 4 vs prior 4
- Rookie indicator (no prior season for trailing-N to use)

Several derivable from existing ingest (weekly_stats, depth_charts). Draft year needs new ingest — `nfl_data_py.import_draft_picks` returns it.

Likely modest individual feature wins; cumulative impact comes from adding several. Lands inside the current per-position feature builders. Best paired with PBP work (TODO #3) — both are pure feature-expansion adds that compose with whatever model class is in use.

**Update 2026-05-03 (trajectory family probe, branch `feat/probe-trajectory`):** First family-level probe shipped per `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md`. Bundled four player-level features (`age`, `is_rookie`, `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`) into a single override and probed in two modes (augment + swap) at BaselineModel + lgb-nb (`--force-composite`) for all 4 positions. Override: `data/features_probe/trajectory.parquet`, 56,652 rows; UDFA / pre-1980 fallback fired on 22.6% of rows. New ingest: `refresh_draft_picks` + `DraftPicksSchema`. **Family verdict: `SIGNAL` (durable)** — three ADOPT cells in Phase 2 composite: **WR augment baseline -0.0414 fpts** ([-0.0606, -0.0230]), **WR augment lgb-nb -0.0194 fpts** ([-0.0299, -0.0096]), **TE augment lgb-nb -0.0107 fpts** ([-0.0191, -0.0028]). First SIGNAL family probe since PR #20 (PBP team features bundle, RB-only). **Family closed at the trailing-8-game unit on the bundled probe.** Greenlights a follow-up integration plan analogous to PR #20 → PR #21. **Recommended:** WR-first integration plan (binding cell is `(BaselineModel, WR)` augment, expected ADOPT at ~-0.0414 fpts gate); TE secondary (lgb-nb-only). **Recurring QB augment regression** — this bundle adds two new instances (baseline +0.0382, lgb-nb +0.0233) to a pattern previously seen on lgb-nb only at PR #23 / PR #24. Both model classes now show the QB augment regression on context / team / trajectory feature additions. **Coverage relaxation — `--coverage-threshold 0.35`** (vs spec's 0.95 default; vs PR #22's 0.70 fallback). Trend features are structurally sparse (require 8 prior active games per player; ~50% of player-weeks excluded across all years). NOT silent NaN imputation: bias is symmetric on baseline + candidate sides under the probe's left-merge join. Per-position coverage of override candidates: QB (88.7% / 37.8% / 39.6%), RB (96.6 / 53.7 / 66.6), WR (96.7 / 53.6 / 68.4), TE (95.4 / 44.7 / 71.1). Deepest threshold relaxation in Track 2 history; future re-test or production-builder work must apply the same threshold or scope the cohort explicitly. **Refined-unit candidates beyond trailing-8-game unit remain unexplored:** per-position aging-curve interaction terms (`age²` for older-RB drop), `is_2nd_year` / `is_3rd_year` flags (collinear with age but might unlock breakout-year signal), depth-chart-rank trends, longer trailing windows (l8 vs l16), treating sparsity as a feature (`has_trajectory_history` indicator that flips on at game 8+). None queued. See `reports/feature_probe_trajectory_summary.md`.

**Update 2026-05-04 (TE trajectory features integration, branch `feat/te-trajectory-features`):** Production integration of the 4 trajectory features into `TeFeaturesSchema` + `build_te_features` per `docs/superpowers/specs/2026-05-04-te-trajectory-features-design.md`. Dual-run gate verdict on `(LightGBMNbModel, TE)`: **`ADOPT`** (composite RMSE delta -0.0090 fpts, CI [-0.0171, -0.0013]). Probe predicted -0.0107 fpts; gate matched within ~0.0017 fpts (sharper calibration than WR's 0.0043 gap). **First production integration in the project to bind on a non-default model class** (TE production routes to `baseline`; lgb-nb is where the probe's signal lived). `(BaselineModel, TE)` informational cell DO_NOT_ADOPT at -0.0100 (CI [-0.0280, +0.0093] brackets zero — NOT REGRESSION, so spec §1.3.5 modified-shape branch did NOT fire; ship as designed). **Spec deviation:** only baseline + lgb-nb evaluated (not all 5 classes per spec §1.3.3) — `--model all` aborted after 3hr without writing a run dir; the binding + contingency cells completed in 33 min combined. Other 3 cells (lightgbm, lightgbm-tuned, ensemble) skipped as informational per spec §1.3.4. Coverage 94.8% age / 46.4% volume_trend / 75.6% snap_pct on 2021-2024 eval window — within ~5pp of probe. **Closes TODO #24's TE-cell branch at trailing-8-game unit.** Combined with PR #26's WR integration, the trailing-8-game-unit branch is now closed at all 3 of PR #25's ADOPT cells. **Cross-class TE production routing flip** (lgb-nb-with-trajectory vs baseline-without-trajectory at position level) deferred — naive arithmetic suggests close to break-even (Plan 8's +0.0028 baseline-vs-lgb-nb gap stacked with this PR's -0.0090 lift = -0.0062 fpts, CI likely brackets zero). Not load-bearing for any current consumer; queue alongside next TE-related work. Refined-unit candidates remain unexplored. See `reports/te_trajectory_features_summary.md`.

**Update 2026-05-04 (WR trajectory features integration, branch `feat/wr-trajectory-features`):** Production integration of the 4 trajectory features into `WrFeaturesSchema` + `build_wr_features` per `docs/superpowers/specs/2026-05-03-wr-trajectory-features-design.md`. Dual-run gate verdict on `(BaselineModel, WR)`: **`ADOPT`** (composite RMSE delta -0.0371 fpts, CI [-0.0567, -0.0172]). Probe predicted -0.0414 fpts; gate matched direction and landed within probe CI (gap 0.0043 fpts is bootstrap noise). Shipped. **Other model classes** (informational per spec §1.3.5): lightgbm ADOPT -0.0207, lightgbm-nb ADOPT -0.0171 (cross-checks probe's lgb-nb cell at -0.0194), ensemble ADOPT -0.0242, lightgbm-tuned **DO_NOT_ADOPT** +0.0025 (point near zero — consistent with TODO #29 pruning candidate). Per-year breakdown: 2021 -0.0553 (CI strict), 2022 -0.0295 (CI brackets 0), 2023 -0.0397 (CI strict), 2024 -0.0233 (CI brackets 0); pooled CI strictly negative because every year-point is negative. Coverage 97.2% age / 97.2% is_rookie / 57.5% volume_trend / 73.9% snap_pct on 2021-2024 eval window — all within ~5pp of probe. **Spec gap caught + fixed:** §1.1 Task 5 + §2.3 said pass prior-mask-filtered `ws`/`sc` to `attach_trajectory_features`, but the helper's internal `.shift(1)` makes that a double-filter producing 100% NaN trends. Fix at commit `d1b3092` (pass full unfiltered frames; 5 leakage tests still pass; direct regression test added at `a742d83`). **Three Cluster A test-fixture leftovers caught + fixed** at commits `1f1f415` / `33eea57` / `807f046` (cache fixture, lightgbm/ensemble synthetic random fixtures, tune_lightgbm fixture); defense-in-depth grep confirmed no other missed sites. **Closes** the trailing-8-game-unit branch of TODO #24. Refined-unit candidates remain unexplored under the same TODO. **TE remains open for a separate refined-unit spec — must address per-position routing** (TE only ADOPT'd under lgb-nb, not baseline). See `reports/wr_trajectory_features_summary.md`.

### 25. Weather features in per-position builders

`import_schedules` already returns wind, temperature, and precipitation columns (verified during 2a ingest). Confirm whether these reach the per-position feature builders today; if not, plumb through.

Wind especially tanks passing efficiency (rule of thumb: >20mph wind drops passing yards ~10-15%). Today's model doesn't see this and over-projects passing-heavy lines in dome-vs-Buffalo-November mismatches.

Small feature add; likely small but real win on a subset of games. Same plan slot as TODO #24 (player-trajectory) — both are quick adds inside existing builders.

**Status 2026-05-03:** **Still queued** as the original sibling probe per the brainstorm Track 2 plan. TODO #24 (trajectory) shipped 2026-05-03 with verdict SIGNAL via WR/TE; weather is the natural next family-probe slot, independent of the trajectory integration follow-up. Same probe-first workflow: bundle 3-4 weather features (e.g., `is_dome`, `wind_speed`, `temperature`, `precipitation_indicator`), generate an override parquet, run `scripts/probe_feature_signal.py` in augment + swap × baseline + lgb-nb (`--force-composite`) modes for all 4 positions. Mechanism prediction: QB / WR / TE pass-volume cells should benefit most; RB rushing relatively insensitive.

**Update 2026-05-07 (weather family probe, branch `feat/probe-weather`):** Family-level probe shipped per `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`. Bundled four weather features (`wind_speed_mph`, `is_high_wind` ≥20 mph threshold, `temperature_f`, `is_grass_surface`) sourced from existing `SchedulesSchema` columns (no new ingest, no schema changes). Dome / closed-roof games filled with `(wind=0, temp=70)` per spec §3.5 — semantically correct for controlled environments. Override at `data/features_probe/weather.parquet` (56,652 rows). **Family verdict: `SIGNAL`** — two ADOPT cells in lgb-nb augment composite: **(lgb-nb augment, RB) -0.0081 fpts** (CI [-0.0163, -0.0005]) and **(lgb-nb augment, WR) -0.0110 fpts** (CI [-0.0172, -0.0049]). **First probe where signal lives only in lgb-nb composite, not in BaselineModel** — RidgeCV cannot extract the bundle's non-linear thresholds even with explicit `is_high_wind` boolean encoding; tree splits do. Mechanism prediction was QB/WR/TE pass-volume; observed: WR confirmed (likely wind suppressing downfield passing + grass YAC effect), RB unexpected (likely grass-surface footing + cold-weather pass→rush regime shift), QB no signal (likely already proxied by `roof_dome` + `implied_team_total`), TE no signal (likely sample-size). lgb-nb swap returned the degenerate all-zero composite — weather cols have no v1 counterparts. **Coverage relaxation — `--coverage-threshold 0.90`** (vs spec's 0.95 default) due to 8.39% outdoor-NaN rate (upstream `nfl_data_py` data quality, concentrated in 2018-2019; per-(position, season) coverage in the 2021-2024 eval window is uniformly ≥92%). On par with PR #23's 0.90 precedent. **Recurring QB augment regression check — milder than PR #23/#24/#25 pattern.** Single per-stat regression cell (QB rushing_yards 2023, +0.0812 fpts CI [+0.0133, +0.1515]); pooled QB Phase 2 lgb-nb augment is +0.0077 fpts (brackets zero, NOT REGRESSION). Plausibly weather information already partially captured by `roof_dome` + Vegas-implied `implied_team_total` doesn't trigger the QB-specific overfit pattern as strongly. **Greenlights a follow-up integration plan:** combined RB + WR through `LightGBMNbModel` only (PR #27 precedent for non-default-model-class binding). Do NOT extend `QbFeaturesSchema` or `TeFeaturesSchema` in the same plan. **Refined-unit candidates remain unexplored** (now in scope, none queued): cold-weather threshold (`is_cold_weather = temp < 32`, sibling shape to `is_high_wind`), multi-class surface encoding, kickoff hour / time-of-day, surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). Recommended priority order if a refinement plan is scoped: cold-weather threshold → multi-class surface → kickoff hour. **What this closes:** TODO #25's broad-cut weather family at the in-builder unit, on the RB + WR cells. QB + TE remain DO_NOT_ADOPT at this unit; refined-unit candidates remain open under the same TODO. See `reports/feature_probe_weather_summary.md`.

**Update 2026-05-08 (RB+WR weather features integration, branch `feat/weather-features-rb-wr`):** Production integration of the 4 weather features into `RbFeaturesSchema` + `WrFeaturesSchema` + `build_rb_features` + `build_wr_features` per `docs/superpowers/specs/2026-05-08-weather-features-rb-wr-design.md`. **Per-position dual-run gate verdicts: (lgb-nb, RB) ADOPT** (RMSE Δ -0.0077, CI [-0.0157, -0.0001]) and **(lgb-nb, WR) ADOPT** (RMSE Δ -0.0104, CI [-0.0165, -0.0042]). Both `(baseline, *)` contingency cells DO_NOT_ADOPT but **NOT REGRESSION** (CIs bracket zero) — per spec §1.3.5, **default ship-as-designed branch fired for both positions**. Probe predicted RB -0.0081 / WR -0.0110; gate matched within ~5% on both binding cells (gap 0.0004 / 0.0006 fpts — within probe CIs). **Second integration to bind on a non-default model class** (after PR #27 TE trajectory) and **first to bundle two positions in a single PR** with per-position contingency matrix decided independently. RB and WR production routings unchanged: both stay on `BaselineModel`.

**3 informational classes skipped** (lightgbm, lightgbm-tuned, ensemble) per spec §1.3.4 + PR #27 precedent — back-fillable by a follow-up backtest if the cross-class routing-flip discussion needs them.

**Cross-class production routing follow-ups (per position):** RB and WR each route to `baseline` per Plan 8. With weather cols now in `RbFeaturesSchema` / `WrFeaturesSchema`, separate cross-class re-evals (`scripts/adoption_gate.py --position {RB|WR}` comparing `lightgbm-nb` candidate to `baseline` baseline) could justify flipping `_PositionDispatch[{RB|WR}].default_model_class` to `lightgbm-nb`. Not load-bearing for any current consumer; queue alongside the next RB- or WR-related work. Same shape as the PR #27 TE follow-up.

**Refined-unit candidates remain unexplored under this TODO:** `is_cold_weather` (`temp < 32`, sibling shape to `is_high_wind`), multi-class surface encoding (one bool per surface code), kickoff hour / time-of-day, surface × position interactions, per-team weather acclimation, precipitation (would require new ingest), wind direction (would require new ingest). Recommended priority order if a refined-unit plan is scoped: cold-weather threshold → multi-class surface → kickoff hour. None queued.

**This closes** the broad-cut weather family at the in-builder unit on **both** RB and WR ADOPT cells from PR #28. QB and TE remain DO_NOT_ADOPT at the broad-cut unit. See `reports/weather_features_rb_wr_summary.md`.

**Coverage caveat:** PR #28 PM entry's "uniformly ≥92%" claim was **overstated** — actual production-builder coverage on 2022 RB/WR is ~67% (matches PR #28's probe override byte-perfectly; pooled 91.6% hides the per-season trough). The probe was nonetheless valid on this same data; the gate has now reproduced both ADOPT verdicts on production-pipeline output. Future "coverage uniformly ≥X%" claims should be reported per-(position, season).

**Follow-up — shipped in this PR:** 4 PRs in a row (PR #21 RB PBP, PR #26 WR trajectory, PR #27 TE trajectory, this PR) hit the same `baseline.py:_<POS>_FEATURE_COLUMNS` spec gap. The parametrized regression test pinning `set(_<POS>_FEATURE_COLUMNS) == set(SCHEMA.columns) - identity` was added at `tests/test_models/test_baseline_feature_columns_match_schema.py` (5 cases — one per position plus an identity-cols sanity check). Closes the recurring-bug class structurally.

**Update 2026-05-09 (refined-unit family probe, branch `feat/probe-weather-refined`):** Refined-unit family probe shipped per `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`. Bundle: `is_cold_weather` (temp ≤ 32°F), multi-class surface one-hot (6 cols pinned from data 2018-2024: `is_a_turf`, `is_astroturf`, `is_fieldturf`, `is_grass`, `is_matrixturf`, `is_sportturf`), `is_primetime` (kickoff_hour_et ≥ 18 — required an in-scope `_build_kickoff` ingest fix at commit `56df07f` to correct ET-wall-clock-as-UTC mistagging). **Family verdict: `SIGNAL`** — 3 ADOPT cells in lgb-nb composite: (augment, WR) -0.0051 fpts, (swap, RB) -0.0088 fpts, (swap, WR) -0.0050 fpts; all CIs strictly negative. Refined-unit decoding: WR strict refinement, RB replace-only, QB+TE close at this cut. Greenlights a follow-up integration plan replacing v1 weather cols (`wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface`) with the refined 8-col bundle in `RbFeaturesSchema` + `WrFeaturesSchema`. Recurring QB augment regression sharper than prior probes (composite +0.0099 fpts, CI strictly above 0; reinforces "do not extend `QbFeaturesSchema` with weather features" rule). Refined-unit-of-refined-unit candidates remain open: continuous kickoff hour, `is_london`, surface×position interactions, per-team weather acclimation, precipitation (new ingest), wind direction (new ingest). None queued. See `reports/feature_probe_weather_refined_summary.md`.

**Update 2026-05-09 (refined-unit RB+WR strict-replace integration, branch `feat/weather-refined-rb-wr`):** Production strict-replace integration scoped per `docs/superpowers/specs/2026-05-09-weather-refined-rb-wr-design.md`. Schema swap on `RbFeaturesSchema` + `WrFeaturesSchema` (drop 4 v1 cols, add 8 refined cols), matching `_<POS>_FEATURE_COLUMNS` swap in `baseline.py`. **Both binding cells `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)` returned `DO_NOT_ADOPT`** with point estimates +0.0012 / +0.0060 — opposite-signed from PR #30 probe predictions of -0.0088 / -0.0050. Probe-vs-gate magnitude shift ~+0.011 fpts on both binding cells, with sign flipped — **largest probe-vs-gate divergence in Track 2A history**. Per §1.3.5 contingency, both positions full-revert: 9 files restored to main's state in commit `c4ba548`; PR ships zero net code change vs main, only spec/plan/reports/PM-TODO updates. Probe's small-magnitude binding cells (~0.005 fpts) coincided with `--coverage-threshold 0.90` relaxation (2022 `is_cold_weather` non-NaN 0.66) — small-magnitude lift estimates from low-coverage features are most fragile to bootstrap noise. **Retrospective takeaway: probe binding-cell magnitudes under ~0.005 fpts with coverage relaxation should be treated as MARGINAL, not SIGNAL.** This closes the broad-cut refined-unit weather work at the in-builder unit on the RB and WR cells from PR #30. **Refined-unit-of-refined-unit candidates remain open under this TODO but are deprioritized** — no evidence the refined unit is the binding constraint over v1; PR #30 probe retrospectively suspect for false-positive signal. None queued; future weather-related plans should require independent mechanism evidence before re-probing. **Cross-class production-routing follow-up (RB + WR): closed-without-action** — lgb-nb-with-anything is even less attractive given the +0.0012/+0.0060 vs-baseline-v1 measurements. Plan 8's `BaselineModel` routing remains the right call for both positions. See `reports/weather_refined_rb_wr_summary.md`.

### 26. Plan 5 / 5b / 5c — LightGBM with quantile regression and NB-2 hybrid (Model C / C-tuned / C-NB) — closed in Plan 5 + 5b + 5c

Closed 2026-04-27 (Plan 5), 2026-04-28 (Plan 5b), 2026-04-28 (Plan 5c).

**Plan 5 (Model C):** Per-stat sub-models trained at quantiles [0.05, 0.10, 0.50, 0.90, 0.95];
new QuantileDistribution + codec branch; POSITION_DISPATCH.factories dict;
backtest snapshot extended (400 → 768 rows). Model A unchanged; both coexist.
**Adoption verdict: failed all three §1.3 criteria.**

**Plan 5b (Model C-tuned):** `LightGBMTunedModel` subclass overriding only
`_hyperparams_for(stat)`, `code_hash`, `model_id`. 24 per-(position, stat)
Optuna studies × 50 trials (TPE + median pruner; sum-of-5-pinball-losses
on 2023 trial scorer). Tuned params in `data/tuned_params/lightgbm.json`
(checked in). Snapshot extended (768 → 1136 rows). **Adoption verdict: also
failed all three §1.3 criteria** — but improved on every metric vs untuned
C: RMSE wins moved 1/16 → 4/16, calibration mean delta -0.086 → -0.063,
QB cells now strictly dominate A. Hyperparameter tuning cannot address the
per-stat-sub-model "no shared prior" mechanism that drives the residual gap
on RB / TE / WR.

**Plan 5c (Model C-NB):** `LightGBMNbModel` subclass overriding only count-stat
training and prediction; yards stats inherited unchanged from
`LightGBMTunedModel`. For the 13 zero-inflated count cells (`*_tds`,
`interceptions`, `fumbles_lost` × per-position target_stats), trains one
`lgb.LGBMRegressor(objective="poisson")` per stat using Plan 5b's tuned
hyperparameters; reads predicted mu directly from `regressor.predict(X)`
(lgb's poisson predict returns mean in original scale, no `np.exp`); fits
NB-2 dispersion via `nb_dispersion_from_residuals` on training residuals;
predicts via `ParametricNegativeBinomial`. Per-row family is `MIXED`.
Snapshot extended (1136 → 1504 rows). **Adoption verdict: also failed all
three §1.3 criteria** — but RMSE moved further: NB strictly dominates Tuned
on 16/16 cells; NB beats A on 11/16 (vs Tuned's 4/16); max worse vs A is
+1.69% (vs Tuned's +2.95%). The mean-prediction fix Plan 5b's diagnostic
predicted worked. Calibration regression carried over essentially unchanged
(NB mean delta -0.062 vs Tuned -0.063); the binding constraint is now
calibration, not mean. **QB cells cleanly beat A on every metric** (4/4
RMSE wins, mean calib +0.012, all Spearman within ±0.005); RB/TE/WR show
the same pattern of RMSE-win-paired-with-calib-regression because NB-2
dispersion fitted on training residuals under-disperses on held-out years
where target variance exceeds the training-fit dispersion.

Model A stays the production default. Model C / C-tuned / C-NB all ship as
peers; none is adopted. **Model C-NB strictly dominates Model C-tuned on
RMSE** — Tuned can be pruned once C-NB has soak time. Infrastructure for
all three preserved. See project_management.md for the per-cell A vs C vs
C-tuned vs C-NB comparison table and detailed analysis.

**Next experiments (per project_management.md "Next action"):**
1. Plan 6 — ensemble of A + C-NB with calibration-aware weighting (most
   promising given the per-position split: C-NB's QB win + A's RB/TE/WR
   calibration advantage are exactly what an ensemble exploits).
2. Calibration-aware NB-2 fitting (fit dispersion to optimize p10/p90
   pinball loss directly; preserves C-NB's RMSE wins).
3. Feature-class tracks: TODO #3 (PBP / EPA features), TODO #23 (target
   decomposition).

### 27. Revisit Model Protocol shape (Fitted vs base separation)

Surfaced in Plan 5 Task 11 review (2026-04-27). Task 11 initially widened the
`Model` Protocol with `target_stats`, `train_seasons`, `code_hash` so consumers
could type against `Model` generically, but this leaked `BaselineModel`'s
set-at-fit semantics (`code_hash: str | None`, `train_seasons: tuple[int, int] | None`)
into the contract. The widening was reverted; consumers now use `cast(BaselineModel | LightGBMModel, ...)`
narrowing.

The clean long-term shape is probably a `Fitted[Model]` Protocol split:
- Base `Model` exposes `position`, `fit`, `predict_distribution`, `save`, `load`.
- `FittedModel(Model)` adds `model_id`, `train_seasons: tuple[int, int]` (no None),
  `code_hash: str` (no None) — guaranteed populated post-fit.
- `Model.fit()` returns `FittedModel`-typed self; consumers post-fit type against `FittedModel`.

Revisit when Plan 6 (EnsembleModel) lands — that's the natural moment to redesign
the Protocol surface as more model classes need to share the contract. Until then,
the `cast()` pattern in CLI scripts is the trade-off accepted.

### 28. Widen aggregate_to_season to accept QUANTILE / MIXED family — closed 2026-05-17

**Closed during upside-ranking-diagnostic T11.** Surfaced as a real blocker
(not just a metric-coverage gap): `scripts/project_season.py --season 2024`
raised `ValueError: aggregate_to_season requires family=SAMPLED_SUMMARY,
found ['MIXED']` because the production QB / WR models (`lightgbm-nb`,
`ensemble-decomposed`) emit per-row `family=MIXED`.

The single-line guard in `src/projections/aggregation/season.py:86` was
widened to accept `{SAMPLED_SUMMARY, QUANTILE, MIXED}`. The codec's
`_unpack_single` already handles every per-stat family (NORMAL, GAMMA,
NEGATIVE_BINOMIAL, STUDENT_T, QUANTILE, MIXTURE) regardless of the
row-level tag, and `score_distribution` consumes Distribution Protocol
instances — so composition of weekly samples to season totals works
unchanged for all three allowed row-level families. Coverage added in
`tests/test_aggregation/test_aggregate_mixed_family.py` (MIXED with
NORMAL+GAMMA+NEGATIVE_BINOMIAL per-stat mix; pure QUANTILE row;
determinism check) plus regression for unsupported families
(`test_disallowed_family_raises`, `test_schema_valid_but_disallowed_family_lists_allowed_set`).

### 29. Prune Model C-tuned from POSITION_DISPATCH (deferred until Model C-NB soaks)

Surfaced in Plan 5c (2026-04-28). Model C-NB strictly dominates Model
C-tuned on RMSE (16/16 cells) and is mostly equivalent on calibration
(NB +0.0013 mean delta vs Tuned). Tuned has no remaining advantage over
NB and is therefore pruning candidate.

Concrete tasks (when ready):
- Drop `"lightgbm-tuned"` from each `_<POS>_FACTORIES` dict in
  `src/projections/models/__init__.py`.
- Remove `LightGBMTunedModel` factories from `__all__` (the class itself
  can stay — `LightGBMNbModel` subclasses it).
- Delete or migrate Plan 5b backtest snapshot rows under
  `model_class="lightgbm-tuned"` from `tests/backtest/model_metrics.json`
  (368 rows; backtest gate would shrink to 1136 rows).
- Update `scripts/backtest.py --model all` to expand to 3 classes
  (`baseline`, `lightgbm`, `lightgbm-nb`) instead of 4.
- Possibly drop Model C (untuned) too — it's strictly dominated by both
  Tuned (Plan 5b verdict) and NB (Plan 5c verdict). Decide together.

Defer until Plan 6 (ensemble) lands and we're confident which model
classes the ensemble references — then prune dead classes in the same
housekeeping commit.

**Update 2026-04-29 (Plan 6):** Plan 6 (ensemble) shipped as peer.
EnsembleModel references **Model A and Model C-NB** only — Tuned is not
in the ensemble. Tuned is therefore a clean pruning candidate. Concrete
tasks above remain accurate; the snapshot would shrink from 1872 to 1504
rows after the prune (drop the 368 lightgbm-tuned rows). Defer to next
housekeeping pass; nothing actively references Tuned post-Plan-6.

### 30. Upper-tail count calibration follow-up to Plan 7

Plan 7 (calibration-aware NB-2 fitting at p10/p90) was stopped at Phase 0
because the diagnostic measured per-stat NB-2 count distributions on Plan 5c's
C-NB output and found they are *over*-covering at [p10, p90] by ~16pp (gap
mean -0.169 across 16 cells), not under-covering as Plan 7 assumed. Pinball
fitting at q=0.10/0.90 would narrow count distributions, opposite the
direction needed to close the composite [p10, p90] coverage gap. See
`docs/superpowers/research/2026-04-28-calibration-breakdown.md` and the
Plan 7 entry in `project_management.md` for the full analysis.

The composite calibration shortfall remains real (-0.062 mean vs A) but its
mechanism lives in upper-tail (p95+) count behavior — outside Plan 7's loss
target. Three candidate follow-up mechanisms, in expected-leverage order:

1. **Pinball-loss dispersion fit at upper-tail quantiles.** Same machinery
   as Plan 7 but `quantiles=(0.90, 0.95)` or `(0.95,)` only. Targets the
   actual gap location. Plan 7's diagnostic CLI is reusable. Cheapest if
   it works.
2. **Switch count family to ZIP (zero-inflated Poisson).** Handles zero
   mass / non-zero tail decoupled rather than via NB-2's single
   overdispersion knob. Fundamental distribution-family change. More code
   surface (new `ParametricZeroInflatedPoisson` + codec branch).
3. **Mixture model: explicit point mass at 0 + heavier-tailed integer
   distribution.** Most flexible, most code surface. Defer until 1 and 2
   are tried.

Defer indefinitely if the user accepts the calibration shortfall as a known
limitation (Plan 5c PM's framing). None of the planned downstream consumers
(Draft Hub, start/sit, DFS lineup optimizer) depend on a perfectly calibrated
[p10, p90] interval — they consume mean and rank.

Plan 7's spec, plan, and diagnostic CLI ship to main as record-of-decision
on branch `feat/plan-7-calibration-aware-nb`. Future work re-points the
diagnostic at any new model output to verify the upper-tail mechanism on
the same per-row backtest-output schema.

**Update 2026-04-29 (Plan 6):** Plan 6 (ensemble of A + C-NB with per-stat
pinball at q ∈ {0.10, 0.90}) shipped as peer with **all three §1.3 criteria
failing**. The per-stat pinball optimizer found a clean per-stat optimum
(yards heavily C-NB, TDs moderately A) but the per-stat optimum did NOT
propagate to composite calibration on RB/TE/WR (mean delta -0.097 / -0.080
/ -0.074). This is exactly Plan 7's diagnostic prediction recurring:
per-stat coverage at [p10, p90] does NOT algebraically decompose to
composite [p10, p90] coverage. Plan 6 confirms the mechanism with real
backtest data. The three TODO #30 candidate mechanisms above remain the
right next experiments; the **composite-direct optimization track** (TODO
#30 follow-up #1, but applied at composite level via Monte Carlo rather
than per-stat) is now elevated as the natural successor — same
EnsembleModel infrastructure, replace pinball-on-per-stat with
composite-Brier-on-MC. ~5-10x slower per fold but targets the actual
gate metric directly.

QB cells in Plan 6 *do* improve on every metric (mean calib +0.018; RMSE
strict -1.8% to -2.5% across 4/4 years). Per-position routing
(`POSITION_DISPATCH[QB].factories['default'] = qb_ensemble`) would adopt
Model D for QB only without breaking RB/TE/WR. Open as a follow-up if
QB-specific accuracy ever matters more than uniform routing.

### 31. Preseason full-season projections for the Draft Hub (required, not yet built)

**Context.** Today's Projections Core produces *weekly in-season* projections. Every per-position feature builder consumes trailing-N in-season stats (`*_per_game_l4`, `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`, NGS rolling windows). For Week 1 of a new season the builders fall back to the prior season's trailing data; for any week ≥ 2 they require games already played in the current season.

**The gap.** The Draft Hub (planned sub-project per CLAUDE.md) needs *preseason* season-long projections — a single number per player produced before any 2026 games are played. The current pipeline can't deliver that cleanly:

- 2026 weekly_stats don't exist yet (and won't until games are played).
- 2026 depth charts don't exist until preseason camps open (late July).
- 2026 schedules are released by the NFL in May (likely just now available via `nfl_data_py.import_schedules`).
- Even with all of the above, "Week 1 features × 17 weeks" would whiff on rookies (no prior pro stats), free-agent moves (player on new team with new role), and year-over-year aging effects (RB age curves don't show up in trailing-4 from last season).

**What "good" looks like.**
- Per-player season-total projection: full distribution (mean + intervals), not just a point estimate. Same `Distribution` Protocol the weekly path already uses.
- Handles rookies via draft-pick + college-production priors, not zero-padding.
- Handles team/role changes (FA moves, trades, depth-chart shifts) via a role/usage prior keyed on team-position.
- Handles aging via the trajectory features that PR #26/#27 already shipped — but applied *forward* (project age N+1) rather than measured backward.
- Backtest-able: re-project a historical season's preseason ranking from data available at draft time only, and gate it against actual season totals.

**Candidate mechanisms (none built).**
1. **Prior-season aggregate as the trailing window:** instead of `*_per_game_l4` from games actually played, use `*_per_game_full_season_<year-1>`. New feature class on each position's schema, populated from the prior season's `weekly_stats` aggregate.
2. **Role / depth-chart prior:** for any player with a new team-position assignment, predict their usage from a team-position role distribution (e.g., "team's projected WR1 averages 9.2 targets/game across the league"), then per-player residual.
3. **Rookie prior:** draft-pick-anchored model trained on rookie-year per-game distributions, keyed on position + round + college conference. Probably needs its own ingest of college stats.
4. **Aging-curve forward-pass:** TODO #24's trajectory features but applied as "where will player be at age N+1" rather than "where are they now relative to prior season."
5. **Composition layer:** combine 1-4 into a season-total distribution, multiply by projected games played (16 or 17 minus injury prior).

**Estimated scope.** Substantial — likely a new sub-project parallel to `src/projections/` (e.g., `src/projections/preseason/`) with its own feature schema, its own model classes, and its own backtest harness gated on season-total RMSE/rank-correlation vs actuals. Don't try to retrofit the weekly path.

**Workflow recommendation.** Spec → plan → execute on a dedicated branch. First plan should be the brainstorm + roadmap (which mechanism first? what's the cheapest path to a credible preseason rank?). Treat it as Project-level work, not a single feature plan.

**Status.** **CLOSED 2026-05-17.** Spec + plan + impl shipped on `worktree-feat+preseason-projections`. Sub-package `src/projections/preseason/` (features.py / model.py / project.py / backtest.py); 3 new schemas; 2 new CLI scripts; walk-forward backtest harness gating v1.5+ trained models on ≥6/8 cells ADOPT. See `project_management.md` entry for full status and `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` for design.

### 32. Migrate ingest off `nfl_data_py` to `nflreadpy` (blocks in-season 2026 projections)

**Critical blocker for the in-season-projections premise.** Discovered 2026-05-11 when a 2025 retrospective failed at ingest: `nfl_data_py.import_weekly_data([2025])` returned `HTTP 404`. Investigation:

- `nfl_data_py` v0.3.2 (installed) and v0.3.3 (latest on PyPI, uploaded 2024-09-20) both fetch weekly stats from `https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_<y>.parquet`.
- Direct HEAD probe: `player_stats_2024.parquet` is 200 OK (5,597 rows); `player_stats_2025.parquet` is **404**.
- `nflverse-data` quietly migrated the weekly stats release in 2025 to **`stats_player/stats_player_week_<y>.parquet`** (release tag changed + filename changed + schema expanded to include defensive players and renamed columns: `interceptions` → `passing_interceptions`, `recent_team` → `team`, `sacks` → `sacks_suffered`, `sack_yards` → `sack_yards_lost`).
- `stats_player_week_2024.parquet` is 200 OK and returns **18,981 rows** (vs the old URL's 5,597 — 3.4× more, including DST/IDP).
- `stats_player_week_2025.parquet` is 200 OK with 19,421 rows across all 22 weeks of 2025 (regular + playoffs).
- Other tables (`pbp`, `snap_counts`, `depth_charts`, `schedules`, `ngs_*`, `draft_picks`) still work on `nfl_data_py 0.3.2` for 2025 — only the weekly stats endpoint moved.

**The official `nflverse` org maintains `nflreadpy`** (PyPI `nflreadpy`, latest 0.1.5, uploaded 2025-11-19, repo at https://github.com/nflverse/nflreadpy, docs at https://nflreadpy.nflverse.com). Smoke test: `nflreadpy.load_player_stats(seasons=[2025])` returns the full 19,421 rows. This is the supported successor library; `nfl_data_py` appears effectively abandoned (no PyPI release for 14+ months, missed the URL migration).

**What this means for the project's premise:** mid-season 2026 projections require live ingest of weekly stats during the season. With the current dependency, that ingest cannot pull 2025 weekly stats today, and 2026 weekly stats will hit the same wall when the season opens in September. **This is a hard blocker** — not a nice-to-have refactor.

**Three migration paths (in increasing scope):**

1. **Minimum patch (1-2 hour):** add the new URL template directly to `src/projections/ingest/weekly_stats.py:_fetch_raw_weekly`, with a column-rename map (`passing_interceptions → interceptions`, `team` stays, `sacks_suffered → sacks`, `sack_yards_lost → sack_yards`) and filtering on `position_group in {QB, RB, WR, TE}` to match the existing fantasy-positions filter. Skip the DST/IDP columns. Keeps `nfl_data_py` as the dep for the other 7 tables; adds an in-house URL fallback for weekly_stats only. Lowest risk; doesn't touch the other ingest seams.

2. **Migrate weekly_stats to nflreadpy, leave others on `nfl_data_py`:** wrap `nflreadpy.load_player_stats` behind `_fetch_raw_weekly`. Polars-vs-pandas mismatch — `nflreadpy` returns Polars frames; coerce via `.to_pandas()`. Same column-rename layer as option 1. Adds a second dep (`polars`) but uses a maintained library.

3. **Full migrate every ingest source to nflreadpy:** all 8 tables (weekly_stats, schedules, depth_charts, ngs_{passing,rushing,receiving}, snap_counts, pbp, draft_picks, id_map). Future-proofs against further nflverse URL changes (which we now know happen). Largest scope. Requires column-rename audits on every ingest module + every per-position feature builder downstream.

**Recommended order:** ship option 1 immediately to unblock 2025 retrospective + 2026 in-season ingest; then plan option 3 as a separate "migrate ingest layer to nflreadpy" project before the 2026 season opens (Sept 2026 — ~4 months from today).

**Also worth checking** in the same investigation: are any of the *other* `nfl_data_py` endpoints showing 2025 data that's stale-looking vs nflverse's actual releases? PBP-2025 and snap-counts-2025 fetch successfully today, but the URL paths may have similarly silent renames coming. Audit each `nflverse-data/releases/download/<tag>/<file>` path in `nfl_data_py.__init__.py` against the corresponding accessor in `nflreadpy` and note any divergence.

**Probe URLs captured for the record** (HEAD-probed 2026-05-11):
- `player_stats/player_stats_2024.parquet` → 200 (old format, kept for backward-compat)
- `player_stats/player_stats_2025.parquet` → **404**
- `stats_player/stats_player_week_2024.parquet` → 200 (new format, 18,981 rows)
- `stats_player/stats_player_week_2025.parquet` → 200 (new format, 19,421 rows)
- `pbp/play_by_play_2025.parquet` → 200 (still working)
- `snap_counts/snap_counts_2025.parquet` → 200 (still working)
- `depth_charts/depth_charts_2025.parquet` → 200 (still working)
- `depth_charts/depth_charts_2026.parquet` → 200 (already published for upcoming season — useful for TODO #31's preseason-projection work)

**Status.** Captured 2026-05-11. Option 1 shipped same-day on `fix/weekly-stats-2025-ingest` (PR #34). Option 3 (full migration to `nflreadpy` + historical re-ingest) shipped same-day on `feat/nflreadpy-migration`: all 8 sources (weekly_stats, schedules, depth_charts, ngs_{passing,rushing,receiving}, snap_counts, pbp, draft_picks, id_map) now route through `nflreadpy.load_*().to_pandas()`. `nfl_data_py` dropped from deps; `nflreadpy>=0.1.5` and `polars>=1.0` added. Historical re-ingest covered 2018-2025 for every source except depth_charts. **Carve-out:** depth_charts 2025+ uses a fundamentally different upstream schema — see TODO #34 below for the derivation work needed. **All other follow-ups under this TODO are now closed.**

### 34. depth_charts 2025+ — derive (season, week) from snapshot-by-timestamp feed

**Captured 2026-05-11 during the nflreadpy migration.** nflverse changed the 2025+ depth_charts release from a weekly per-team format to a snapshot-by-timestamp feed. New columns: `dt` (ISO timestamp), `team`, `gsis_id`, `pos_abb` / `pos_grp` / `pos_name` / `pos_slot` / `pos_rank`, `player_name`, `espn_id`. Old columns we depended on (`season`, `week`, `club_code`, `depth_team`, `depth_position`) are gone. nfl_data_py 0.3.x ingested the legacy URL successfully into 2024 (which is why it didn't trip the option-1 spike), but the underlying release was already migrating; from 2025 on the only release path returns the new shape.

**Effect on this codebase:** `refresh_depth_charts` raises `NotImplementedError` for any season whose payload lacks the legacy columns, so 2025 depth_charts partitions are simply not produced. 2018-2024 partitions remain fully up to date. Downstream feature builders that read depth_charts (`build_*_features`) treat a missing 2025 row as missing data — currently surfaces as `NaN` features and would fail pandera validation on 2025 projection runs.

**Approach for the derivation.** Snapshot-by-timestamp can be coerced back to per-week:
1. Read 2025 `schedules`. For each (game_id, kickoff), pick the depth_charts snapshot with the largest `dt` strictly before kickoff (closest-prior snapshot).
2. Map `pos_abb` (e.g., LWR, RWR, SWR, RB1, TE1, QB1) and/or `pos_rank` to our canonical `(position, depth_rank)`.
3. Emit one row per (season=2025, week, team, gsis_id) with the resolved rank.

Two open design questions before the spike: (a) what timezone is `dt` published in (assume UTC; verify); (b) how to handle the ~3,200-row snapshots that include defensive/special-teams players we don't model — same `Position` filter as everywhere else should suffice. Snapshots are emitted roughly daily (221 distinct dt values across 2025), so the closest-prior-snapshot rule is well-conditioned even mid-week.

**Status.** Captured 2026-05-11. Blocking 2025+ projection runs that consume depth_chart features. Defer until next ingest-touching plan.

**Update 2026-05-15 (depth_charts 2025+ derivation, branch `feat/depth-charts-2025`):** Shipped per `docs/superpowers/specs/2026-05-15-depth-charts-2025-derivation-design.md` and plan `docs/superpowers/plans/2026-05-15-depth-charts-2025-derivation.md`. `_derive_weekly_snapshots_from_new_format(raw, schedules)` in `src/projections/ingest/depth_charts.py` resolves each `(season, week, team)` to the closest-prior snapshot (`dt < kickoff`, strict <) via per-team `pd.merge_asof(direction="backward", allow_exact_matches=False)`, filters to `Position` enum values, and synthesizes `depth_rank = clip(pos_rank, 1, 10)` + `depth_team = str(depth_rank)` (matches the legacy on-disk pyarrow string column with `{"1", "2", "3"}` distinct values). `_normalize_one_season` dispatches on `dt in raw.columns`; pre-2025 legacy path unchanged. `refresh_depth_charts` loads schedules from `data/raw/schedules/season=<s>/` when the new-format payload is detected and no schedules frame is passed; raises `FileNotFoundError` with a clear "ingest schedules first" message if the partition is missing. 12 new unit tests (closest-prior rule incl. strict-`<`, position filter, bye-week skip, deep-rank clamp, depth_team synthesis, JAX→JAC / LA→LAR normalization regression, schedules-from-disk path, missing-schedules raise, schema round-trip, dedupe). **2025 partition: 10,389 rows, 32 teams, 22 weeks; 3,976 WR / 2,492 RB / 2,321 TE / 1,600 QB.**

**Plan-vs-execution deviations.** (a) Spec initially proposed `pos_rank == 1` filter + `depth_team = pos_abb + str(pos_slot)`; corrected during real-data inspection — `pos_rank` IS the canonical depth rank (1 = starter, 2 = backup) and `pos_slot` is a position-id-like value, not a slot identifier in the legacy sense. Filtering pos_rank==1 would have dropped 80%+ of rows. (b) Real-data first run produced only 30 teams (JAC and LAR missing) because raw nflverse depth_charts uses `JAX`/`LA` while validated schedules use `JAC`/`LAR`; fixed by normalizing the raw `team` column BEFORE the per-team `groupby`/merge. Regression test added at `test_derive_normalizes_raw_team_codes_jax_la`.

### 33. Elite-season under-projection — four leverage points

**Diagnosis.** Captured 2026-05-11 during a 2024 retrospective. The current pipeline's mean prediction systematically under-projects the actual top-tier finishers by 50-150 fantasy points each, across positions. 2024 examples (predicted - actual): Ja'Marr Chase WR1 -143 (260 pred / 403 actual); Jahmyr Gibbs RB2 -132 (259 / 390); Derrick Henry RB3 -106 (276 / 381); Saquon Barkley RB1 -88 (361 / 450); Terry McLaurin WR4 -77 (246 / 322); Joe Burrow QB6 -72 (300 / 373); George Kittle TE3 -70 (167 / 237); Bijan Robinson RB4 -60 (279 / 340); James Cook RB5 -60 (264 / 324); Justin Jefferson WR3 -55 (273 / 328); Amon-Ra St. Brown WR2 -53 (285 / 338); Josh Allen QB2 -51 (383 / 434); Lamar Jackson QB1 -50 (420 / 470). Pattern is *systematic and predictable*: at every position, the model gets the *rank* of the top tier roughly right but compresses the *magnitude*, so elite seasons land in the 250-300 pred-points band even when they actually score 380-470.

**Why this matters.** The draft tool (TODO #31) needs to differentiate elite from very-good — a fantasy draft turns on whether you can take Chase at WR4 ADP and project him at 400, not 260. Mean-prediction RMSE alone misses this; the model can have low aggregate RMSE while still being useless for the specific decisions that draft-tool consumers make.

**Mechanism (best current read).** The trailing-N features describe the player's recent statistical past. Chase's 2018-2023 averaged ~15 ppg PPR; trailing-4 features in 2024 look like a 15 ppg player; Ridge regularization compresses any single feature's coefficient, so the model has no internal mechanism to push the prediction to 23 ppg. Forward-looking signals that *do* explain breakouts (team-context shifts, role consolidation, scheme changes) are mostly absent from the current feature set. Compounding the issue at the count-stat level: TDs are weighted 6 pts in PPR, but Ridge under-disperses TD predictions hard — Chase's 17 TDs (~1/game) likely projected to ~7-10 (~0.5/game), which alone accounts for ~50 of the 143-point miss.

**Four leverage points, in order of likely impact.**

**33a. Use production model routing, not BaselineModel uniformly.** The 2024 retrospective used `BaselineModel` for all four positions because it was the cheapest to train. Plan 8's production routing is QB→`lightgbm-nb`, RB→`baseline`, TE→`baseline`, WR→`ensemble`. Plan 5c established that lgb-nb cleanly beats baseline on QB cells (4/4 RMSE wins, mean calib +0.012) and Plan 6's ensemble was specifically designed to combine A's calibration on RB/TE/WR with C-NB's mean accuracy on QB. **The Chase miss might be 30-50 points smaller under WR ensemble, and the entire QB top-10 might tighten under lgb-nb.** Cheapest possible experiment (no model changes; just stop hardcoding `"baseline"` in `scripts/project_season.py` and call `production_model_for(position)` instead). Status: re-running the 2024 retrospective with production routing 2026-05-11 to quantify.

**33b. Decomposed targets (volume × efficiency) for WR yards and TDs.** TODO #23 established the mechanism. PR #32's probe signaled marginal SIGNAL on receptions but NULL on yards and TDs at the Ridge-only sub-model class. The natural follow-up (already named in PR #32 §7 follow-ups) is to test factor-appropriate sub-models: logistic for catch_rate / td_rate_per_target, log-link Gamma for yards_per_target, Poisson / NB-2 for targets. The compounding multiplier — high target volume × high YPT × high TD rate, each of which Chase had in 2024 — is exactly what direct per-stat prediction collapses. Decomposed prediction with proper distribution composition (`ProductDistribution` + within-row coherent sampling) might recover 30-50 points on elite WR seasons specifically because it preserves the multiplicative structure. **Existing roadmap; this is the highest-leverage already-scoped follow-up.** Same approach extends to RB rushing (carries × yards_per_carry + TDs per touch). Cross-reference TODO #23.

**33c. Forward-looking team-context features.** Current feature classes (PBP team pace/PROE, opp-EPA, red-zone, pressure, trajectory, weather) all measure *what already happened*. None capture *what will happen next year* (the 2024 Saquon-to-PHI / Henry-to-BAL / Bowers-on-LV signals existed in May 2024 from preseason ADP, team win totals, OC hires, free-agent movement). Candidate features: Vegas season-long win totals at each week's as-of-time, projected snap share from depth-chart rank (refined), preseason ADP, head-coach / OC tenure indicators, free-agent acquisition flags. **Different feature class than anything probed so far.** Also directly load-bearing for the Draft Hub (TODO #31) where there is no in-season trailing data at all. Cheapest probe entry: bundle 3-4 Vegas-derived signals (season win total, season O/U, projected pace, projected passing rate) and run the probe-first workflow. Mechanism prediction: most lift on RB and WR; QB modest; TE small.

**33d. Rank by an upside-sensitive statistic, not season-total mean.** Independent of any model change. The pipeline already emits per-week `p10`, `p50`, `p90` for every player; Chase's *weekly p90* is probably close to his actual 23.5 ppg even when his mean is 15 ppg. For *ranking* purposes (which is what draft-tool consumers actually see), Monte Carlo the per-week distributions, sum to a season-total distribution per player, and rank by E[points above replacement] or P[season total ≥ elite threshold] — a metric that rewards both mean and upside. This doesn't fix mean RMSE but it does fix the *display*. **Cheapest fix of the four if it works; doesn't touch the model.** Verification step: sum Chase's weekly p90s for 2024 and check if it approaches 380-400. If yes, the upside signal already exists and is just being thrown away in aggregation. If no, the model genuinely doesn't see the upside and 33b/33c are required. Cross-reference TODO #30 (upper-tail calibration) — 33d uses the same upper-tail distributions; TODO #30's mechanisms (pinball at upper quantiles, ZIP for counts) directly improve the input to 33d.

**Suggested sequencing.** 33a is essentially free (one-line script change); ship today. 33d is the next-cheapest (single-shot MC aggregation script); validates whether the existing distributions already contain the answer. Run 33d *after* 33a so the production-routed predictions feed the ranking metric. 33b is the natural next integration cycle (PR #32 follow-up already scoped). 33c is a fresh feature-class probe; queue alongside or after TODO #31 (Draft Hub) since both need forward-looking signals.

**33a — empirical result, 2026-05-11.** Re-ran 2024 retrospective with production routing (QB lgb-nb, RB baseline, TE baseline, WR ensemble). **Fixed two real bugs, did not fix the elite-magnitude problem.**

| Issue | Baseline-only routing | Production routing |
|---|---|---|
| Carson Wentz over-projection | 323.0 pred / 4.7 actual / +318 miss (QB #7) | dropped out of top 100 entirely |
| Backup-QB over-projection class | Driskel, Browning, Rudolph, Willis all in top 100 | all gone from top 100 |
| Malik Nabers as WR1 | 296.9 pred / 271.6 actual (predicted #1, actual #8) | 256.4 pred (predicted #5) — rank-realistic |
| Ja'Marr Chase miss | 260.1 pred / 403.0 actual / −143 (predicted #7) | 250.4 pred / −152 (predicted #9) — **slightly worse** |
| Lamar Jackson miss (#1 QB) | 419.6 pred / 469.5 actual / −50 | 427.0 / −43 |
| Josh Allen miss (#2 QB) | 383.0 / −51 | 386.2 / −48 |
| Saquon Barkley miss (#1 RB) | 361.3 / 449.7 / −88 | unchanged (RB stays baseline) |
| Gibbs miss (#2 RB) | 258.7 / 390.4 / −132 | unchanged |
| Henry miss (#3 RB) | 275.6 / 381.4 / −106 | unchanged |

**Conclusion: 33a closes two bug classes but doesn't move the elite-magnitude needle.** The mean-regression compression on elite players lives in *feature signal coverage*, not in model class. lgb-nb's NB-2 dispersion handles the backup-QB defect cleanly (correctly down-weights "depth-chart presence without snaps"); the ensemble's calibration-aware weighting handles the rookie-volume-extrapolation defect cleanly (Nabers's trailing-4 target spike no longer linearly extrapolates). Neither addresses the lack of features that *can* predict a 23 ppg WR season or a 24 ppg RB season. Ship 33a permanently (one-line change in any consumer that previously used `factories["baseline"]`); then move to 33d (upside-sensitive ranking) to test whether the existing distributions already carry the elite signal, before committing to the larger 33b / 33c builds.

**Status.** Diagnosis captured 2026-05-11. 33a executed inline — *necessary but not sufficient*; ship the routing change as the durable fix. 33d/33b/33c queued.

**33d — Phase 1 diagnostic verdict, 2026-05-17: NO GREENLIGHT (durable).** Diagnostic shipped on branch `feat/upside-ranking-diagnostic`. New script `scripts/diagnose_upside_ranking.py` consumes the new `season_projection_weekly_<season>.parquet` + `season_projection_distributions_<season>.csv` artifacts (now emitted by extended `scripts/project_season.py` alongside the unchanged naive CSV) and computes per-(position, season) ranks under four metrics: `mean` (baseline), `season_p90`, `blend_70_30 = 0.7·mean + 0.3·p90`, `p_elite = P(season ≥ elite_threshold)`. Elite thresholds (computed from 2019-2023 actuals, ≥8 games): QB=354.3, RB=290.0, WR=316.6, TE=203.5. Decision gate (spec §1.3 #3) requires the same metric SIGNAL at ≥3/4 positions in BOTH 2024 AND 2025. **Result:** zero SIGNAL cells across all 24 (position, season, non-mean-metric) cells; only 2 MARGINAL cells (WR p_elite 2024, QB p_elite 2024), neither survives to 2025. Decision gate returns `No greenlight`.

**Mechanism finding (the load-bearing insight):** `p90` and `blend_70_30` reshuffle the middle/bottom of each positional cohort (only ~54% / ~77% of all player ranks match `mean` byte-for-byte) but produce **identical top-K sets** at K=5 (across all 8 cells) and K=12 (across 5 of 8 cells) — i.e., none of the actual elite finishers move into the predicted top tier under any of these candidates. The mechanism: `blend = 0.7·μ + 0.3·(μ + k·σ) = μ + 0.3·k·σ` is monotonic in μ whenever σ is monotonic in μ, which is empirically what lgb-nb / ensemble-decomposed / baseline output here. The hypothesis that "the upper tail captures elite signal mean ranking discards" is FALSIFIED in the concrete cases that motivated TODO #33: Chase 2024 mean=250.74, **p90=283.80**, actual=403; Gibbs 2024 mean=258.75, p90=295.29, actual=390.40; Henry 2024 mean=275.64, p90=311.38, actual=381.40. The whole distribution is shifted-down for elites, not just its mean.

**`p_elite` (the only metric that's NOT a monotonic transform of mean)** does change rank order — and where it shows MARGINAL (WR/QB 2024), it tends to correctly promote longer-tail prospects. But its Kendall tau is LOWER than mean across all 8 cells (QB 0.74→0.72, RB 0.77→0.61, WR 0.70→0.42, TE 0.70→0.56), so it's noisier than mean for the broader player pool. The MARGINAL gains on the top of the cohort come at the cost of overall ordering quality. Not a free win.

**This closes 33d.** The elite-magnitude problem lives in *feature signal coverage*, not in distribution-tail mining of the existing model output. **Next direction (one of):**

1. **33b — decomposed factor-appropriate sub-models on WR yards/TDs** (TODO #23 continuation). PR #32 already returned marginal SIGNAL on WR receptions (Ridge-vs-Ridge). PRs #39 (logit catch_rate) and #44 (Tweedie yards_per_target) were both NULL. The remaining factor-class slot is `td_rate_per_target` (Poisson or logistic). PR #38 / PR #41 already shipped WR ensemble-decomposed-child for receptions; extending to yards/TDs would lift the marginal magnitude — if the factor-class swap on td_rate_per_target shows SIGNAL.

2. **33c — forward-looking Vegas team-context features family probe.** Genuinely unexplored feature class (not measured by anything in TODOs #3 / #24 / #25). Candidates: as-of-time season win total, season O/U, projected pace, projected pass rate, OC/HC tenure, FA-acquisition flag. Different mechanism axis from anything probed so far. Also load-bearing for TODO #31 (Draft Hub preseason projections). Cheapest probe entry: bundle 3-4 Vegas signals, generate override parquet, run `scripts/probe_feature_signal.py`. Mechanism prediction: most lift on RB and WR — the exact positions where the elite-season miss is worst.

**Recommendation: 33c next.** Genuinely unexplored axis; TODO #31 prerequisite. 33b is the named follow-up but the factor-class line has produced 2 NULLs in a row on WR receiving; the prior is weaker.

**Side-effect closure: TODO #28 also closed.** `aggregate_to_season` was widened to accept MIXED + QUANTILE family rows in commit `ffdd334` (required by the diagnostic because QB lgb-nb + WR ensemble-decomposed both emit MIXED family rows). Single-line guard widening + 3 new tests + docstring update; no behavior change for SAMPLED_SUMMARY callers.

See `reports/upside_ranking_diagnostic.md` for the full per-(position, season) verdict tables and `reports/upside_ranking_diagnostic_table.csv` for the per-player drill-down.

**33c — Phase 1 family probe complete, 2026-05-17: SIGNAL at lgb-nb swap (QB + WR composite).** Probe shipped on branch `feat/probe-vegas-team-context`. 4-col bundle (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) from already-ingested `spread_line` / `total_line`. Probe matrix: BaselineModel × {augment, swap} + lgb-nb × {augment, swap} with `--force-composite`. Result: **lgb-nb swap composite returns 2/4 ADOPT — QB −0.0587 fpts (CI [−0.092, −0.028]) + WR −0.0130 fpts (CI [−0.022, −0.003]). RB just misses ADOPT; TE NULL.** Ridge runs all REGRESS on QB pooled `passing_yards`; the SIGNAL only emerges under trees when the 4 new cols *replace* per-game `implied_team_total` + `spread` (not when they augment). Mechanism: trees overfit per-game line noise; smoother preseason + season-to-date signals generalize better.

Next step: greenlight a **per-position integration plan for QB + WR only** (lgb-nb / ensemble-decomposed routes). Schema-swap on lgb-nb only (preserves Ridge children's signal in the ensemble). TE skipped; RB deferred for a follow-up probe with `preseason_*`-only.

Caveats: ΔRMSE −0.06 fpts at QB is ~1–2% per-week — the Chase 250→403 gap is not closed by this alone. The integration is necessary but not sufficient for elite-magnitude. If gate confirms ADOPT but elite-magnitude persists, next is **external preseason Vegas data** (genuine May win totals, OC/HC tenure, FA flags).

See `reports/feature_probe_vegas_team_context_summary.md`.

**33c — integration shipped & gate verdict, 2026-05-18: DO_NOT_ADOPT across all 3 gates.** Integration shipped on branch `feat/qb-wr-vegas-team-context-integration` (10 commits: 1 spec, 1 plan, 4 schema+builder, 4 lgb-nb override). Probe's predicted improvements did not generalize. Gate verdicts:

- (lgb-nb, QB) ΔRMSE **+0.1112** fpts (CI [+0.0735, +0.1482]) → **REGRESSION** (probe predicted −0.0587 — sign-flipped, 290% miss).
- (lgb-nb, WR) ΔRMSE +0.0068 fpts (CI [−0.0031, +0.0170]) → null/inconclusive (probe predicted −0.0130).
- (ensemble-decomposed, WR) ΔRMSE +0.0004 fpts (CI [−0.0060, +0.0073]) → null.

**Builder correctness verified before reporting the regression as real:** the integration's QB feature parquet matched the probe's override parquet byte-identically on all 4 Vegas cols across 9,379 QB rows (max abs delta = 0.0). Not a builder bug.

**Root cause: harness-pairing divergence between `probe_composite` and `run_backtest`.** Initial PR #50 data-drift hypothesis was wrong (PR #50 is purely additive logging; pre/post backtest runs have byte-identical row coverage). Verified by re-running the probe on current data — it reproduces −0.0587 ADOPT exactly. The probe pairs predictions with `weekly_stats` on `(gsis_id, season, week)` only; the gate filters `weekly_stats[position == "QB"]` first. The 16 paired-row delta between probe (n=2692) and gate (n=2676) is **all Taysom Hill** in 2023 — listed as QB on the depth chart but recorded as TE in stats. The probe pairs him; the gate (correctly, per production semantics) drops him. Math: those 16 rows account for ~6,624 SSE swing (~20 fpts residual diff per row, plausible for Taysom Hill's high-variance utility profile) — enough to flip the QB verdict on their own.

**This is the first observed case** of a `--force-composite` probe Phase-2 ADOPT failing to replicate in the production gate (prior cases — PR #21 RB PBP, Plan 9 negatives — all replicated cleanly). **Framework follow-up to consider (out of scope for this branch):** align `probe_composite`'s truth-merge to use a position filter matching `run_backtest`'s, so the probe's Phase-2 verdict is a more faithful predictor of the production gate.

**Next direction:**

1. **External preseason Vegas data spec** — genuine May win totals, OC/HC tenure, FA-acquisition flag, projected pace, projected pass rate. Different mechanism axis from re-deriving `spread_line` / `total_line`; not affected by the probe → gate generalization gap encountered here.
2. **RB `preseason_*`-only follow-up probe** still queued, but its prior is now weaker — the probe → gate gap here suggests the RB probe verdict may also fail to generalize. Run with the dual-run gate as the load-bearing decision criterion, not the probe.
3. **TODO #33b (factor-class on WR td_rate_per_target)** continues to be the named alternative, but factor-class line has produced 2 NULLs in a row.

**Branch disposition (pending user decision):** the integration's Phase 0 (schemas + builders) is harmless and may help future re-investigation. Phase 1 (lgb-nb factory swap) is what the gate rejects. Cleanest: merge Phase 0 only, revert Phase 1 factory swap before merge.

See `reports/qb_wr_vegas_team_context_integration_summary.md`.

### 35. v1.5 preseason trained model — required spec elements

**Surfaced 2026-05-17 during PR #48 backtest diagnosis.** Three capability requirements that the v1.5 preseason model spec must address — without them, v1.5 still ships with v1.0's elite-player failure modes (Lamar Jackson at QB14, Burrow not on top-of-board, etc.).

**35a. Multi-prior feature usage.** v1.0 `NaivePreseasonModel` uses `prior_1_per_game × 16` exclusively whenever prior_1 exists, and only falls back to prior_2/prior_3 when prior_1 is fully absent. This discards elite prior-2/prior-3 seasons the moment a depressed prior_1 row exists — Lamar's 2024 MVP-tier season was ignored because 2025's injury-shortened per-game existed. The trained model must consume per-stat prior_1/prior_2/prior_3 columns from `PreseasonFeaturesSchema` (already populated by `build_preseason_features`). Linear feature classes get this for free; non-linear feature classes need to be aware they have access to all three.

**35b. Interaction with `prior_N_season_games_played`.** The features schema carries `games_played` per prior season. A trained model must use these as confidence signals — when `prior_1_games_played` is small AND `prior_1` per-game diverges meaningfully from `prior_2`/`prior_3` per-game, the bias-optimal prediction is closer to a weighted historical mean than to prior_1 alone. Tree models (LightGBM) learn the interaction for free; linear GLMs (Gamma) will not without explicit interaction features (e.g., `prior_1_per_game × prior_1_games_played / 17`) or a regularization scheme that produces the same effect. Spec must either engineer the interaction or pick a model class that learns it.

**35c. Real (non-degenerate) distributions.** v1.0 `PreseasonProjectionSchema` populates `mean / p10 / p50 / p90` to the same value for every player — the distribution carries no uncertainty. The "Lamar coming back from injury" case has fundamentally wider uncertainty than the "Mahomes coming off a normal year" case, and the distribution itself is how an honest model expresses that; the v1.5 trained model should produce real per-stat distributions (likely Gamma for counts, log-Normal for yards) composed via Monte Carlo to a season-total distribution with a proper p10/p90 spread. PR #48 already flagged this as the v1.0→v1.5 gap (spec §4 "v1.0 ships degenerate point-mass").

**Status.** Captured 2026-05-17. These are spec inputs for the v1.5 preseason model, not standalone work items — file them at v1.5 spec-writing time. Cross-reference [[TODO #36]] for the orthogonal injury-feature workstream that would let v1.5 distinguish "depressed because hurt" from "depressed because declining" rather than only widening the band.

### 36. NFL injury report ingest

**Surfaced 2026-05-17 during PR #48 backtest diagnosis; promoted to its own TODO at user request.** The "Lamar / Burrow look low" class of error has two distinct underlying causes that look *identical* in our current feature set: (a) genuine performance decline, (b) injury suppressing per-game numbers and missing weeks. Without an injury data source, no model — naive or trained — can distinguish them at the player level; the best a trained model can do is widen the prediction interval to cover both cases (TODO #35c). This caps v1.5+'s upside even with perfect prior-weighting.

**Why it matters beyond preseason.** Per-week injury status is also load-bearing for:
- In-season start/sit (an "active but questionable" tag should shift the start probability).
- Waiver-wire valuation (out-for-season designations for the player above on the depth chart create immediate value plays).
- Trade evaluation (acquiring a player coming off a recent injury report week-1 is meaningfully different from acquiring a clean-bill-of-health player at the same projected points).
- DFS lineup construction (game-time-decision tags drive late swaps).

**Candidate feature classes the ingest would unlock.**
- Per-week injury designation (questionable / doubtful / out / IR-eligible / activated).
- Career snap-count missed-due-to-injury rate (a stickiness signal).
- Returning-from-surgery flags for the upcoming season (ACL / Achilles / shoulder) — biggest preseason use case.
- Days since most recent injury designation.

**Candidate sources.**
- `nflverse` `injuries` table — should be exposed via `nflreadpy.load_injuries()` if it follows the same pattern as the other 8 sources we migrated. **Check first** before designing alternatives; if present, this is the cheapest path (per-week official NFL designations, same schema shape as our existing ingest seams).
- ESPN injury status API — per-day, requires polling; useful for in-season real-time updates but not preseason.
- Manual curation of offseason surgery news — would need a workflow + a maintained YAML file; only worth doing if `nflreadpy` doesn't cover offseason status.

**Scope.** New ingest source + new pandera schema + new feature class consumed by the preseason builder AND the in-season per-position builders. Substantial — don't bundle with the v1.5 preseason model spec. Sequence: (1) probe `nflreadpy.load_injuries()` availability + column shape; (2) design spec for the ingest module + schema; (3) implement; (4) separate spec to add the feature class to per-position builders (gate via probe-first workflow per [[TODO #3c]]); (5) v1.5+ preseason model spec can then list injury features as a candidate input.

**Status.** Captured 2026-05-17. Not yet scoped. Worth landing before the 2026 in-season period (Sept 2026) when per-week injury data becomes fresh and load-bearing for start/sit and waiver tools.

### 37. Pre-camp rookie ingest — placeholder gsis_ids block 2026 rookies until late July

**Surfaced 2026-05-17.** nflverse upstream sources (`load_draft_picks`, `load_ff_playerids`, `load_depth_charts`) populate `gsis_id` for the current-draft-class rookies with **PFR-style placeholders** (e.g., `MEN516487`, `WIS488223`) rather than real NFL gsis_ids (`00-0040XXX`). NFL.com doesn't issue real gsis_ids to draftees until rosters/depth charts solidify around training camp (typically late July). nflverse uses PFR IDs as a hold-over until then.

**Confirmed scope (probed 2026-05-17 against `nflreadpy.load_draft_picks(seasons=[2026])`):** 0 of 230 2026 rookies have NFL-style gsis_ids; 80 of those are fantasy-position picks (QB/RB/WR/TE) — all blocked. For comparison, 100% of 2024 rookies have proper `00-0039XXX` IDs in the same upstream source.

**Why our ingest filters them.** Every ingest module enforces `GSIS_ID_PATTERN = r'\d{2}-\d{7}'` via `_GSIS_RE.match(...)` before writing a partition. This is load-bearing per CLAUDE.md ("`GsisId` is canonical. All internal storage and joins use it.") — loosening the regex would propagate PFR-style placeholders into every parquet partition and create reconciliation churn when nflverse later replaces them with real IDs.

**Effect.** `data/raw/draft_picks/season=2026/part.parquet` writes as a 0-row partition. The refreshed `id_map.parquet` omits 2026 rookies entirely. `depth_charts/season=2026` likely omits rookies for the same reason (worth confirming separately). Downstream consequence: 2026 preseason projections cannot project any 2026 rookie (Mendoza at pick #1, every other Day 1-3 pick) — they get dropped at the "missing from id_map" gate in `build_preseason_features`.

**Three resolution paths.**

1. **Wait until late July.** Re-run ingest when nflverse propagates real gsis_ids. Zero engineering cost; 2026 rookies are invisible in projections for ~2 months. ETA matches when ESPN / Yahoo / FantasyPros start publishing real preseason ADP and depth charts anyway, so this doesn't actually delay any draft prep step that depends on rookie projections. **Default recommendation.**

2. **Placeholder gsis_id abstraction.** Generate synthetic gsis_ids for pre-camp rookies (e.g., `99-XXXXXXX` derived from a hash of the PFR ID), with a reconciliation step in the ingest that maps placeholder → real gsis_id when nflverse propagates the real ones. Substantial — touches every ingest seam (8 modules), every join path that references gsis_id, the [[validate_gsis_id]] entry point, and ID-hygiene tests. Probably 2-3 days of focused work. Worth doing only if a tool consumer (draft cheat sheet, auction values, snake recommender) needs to show 2026 rookies BEFORE late July.

3. **Loosen the GSIS regex.** Accept PFR-style placeholders as valid gsis_ids directly. **Don't** — breaks the canonical-ID invariant; downstream reconciliation when real IDs arrive becomes ugly (every existing parquet partition's rookie rows need to be rewritten).

**Quick adjacent fix (regardless of choice).** Both `_normalize_one_season` in `src/projections/ingest/draft_picks.py` and the analogous filter in `src/projections/ingest/id_map.py` silently filter PFR-placeholder rows. Add a single WARNING line ("`refresh_draft_picks`: filtered N row(s) with non-GSIS placeholder ids — likely pre-camp rookies; re-ingest after training camps for real ids") so the next user who runs this for a draft-class season immediately sees the situation instead of chasing the 0-row partition. ~5 minutes; do this before close.

**Status.** Captured 2026-05-17. Path 1 (wait) is the default. Path 2 (placeholder abstraction) is a candidate sub-project if early-rookie visibility becomes load-bearing. Path 3 is rejected. The quick warning-line fix should ship soon regardless.
