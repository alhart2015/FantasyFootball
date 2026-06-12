# H2H Draft-Strategy Backtest Harness — Design

**Date:** 2026-06-12
**Status:** Design (pre-plan)
**Branch:** `feat/h2h-backtest-harness`

## 1. Motivation

Tests 1–6 (`reports/draft_strategy_tests.md`) all score a roster by *expected season points under availability* (`SeasonValuer` / `expected_season_points`). That metric **is `season_value`'s own objective**, so `season_value` winning by it is partially circular. Every 16-team test favors `season_value`, but none escapes this caveat.

This harness scores rosters by the **real fantasy objective on real outcomes**: set a lineup each week from the projections a manager actually had, score it against what actually happened, win or lose head-to-head matchups, and crown a champion. The metric — regular-season win %, playoff %, championship % — is **not** any strategy's objective function, so it is the first clean test of whether `season_value`'s expected-points dominance converts into *wins*, or whether `now_or_never`'s timing/scarcity layer wins more matchups.

Everything uses **genuine point-in-time 2025 data** (validated 2026-06-12): the draft basis a manager had at the 2025 draft, the weekly projections they had each week, and the realized results.

## 2. Goal & the question it answers

> In a 16-team half-PPR league replayed on real 2025 outcomes, which draft strategy — `now_or_never`, `season_value`, or `raw_vorp` (control) — wins more head-to-head leagues?

Output: per-strategy **championship %**, **regular-season win %**, **playoff-make %**, and **points-for**, each with a bootstrap CI, pooled over many seeded league simulations.

## 3. Scope & non-goals

**In scope (v1):**
- One real season: **2025**.
- One league shape: **16 teams, half-PPR**, `_league_16_half.json` roster (`QB1/RB2/WR2/TE1/FLEX1/BENCH4`).
- **Mixed field:** each league = 4 `now_or_never` + 4 `season_value` + 8 constrained ADP-bots. **Seat layout (load-bearing — seat position swings the nn↔sv gap +11→+120 in the seat sweep):** strategies are *interleaved* so each spans the seat spectrum, not clustered. Concretely, seat `k` (1-based) is `now_or_never` if `k % 4 == 2`, `season_value` if `k % 4 == 0`, else a bot → nn at {2,6,10,14}, sv at {4,8,12,16}, bots elsewhere. **Paired-seed mirroring:** odd seeds use this layout; the paired even seed swaps the nn and sv seat sets (sv at {2,6,10,14}, nn at {4,8,12,16}), so pooled over paired seeds each strategy occupies an identical set of seats — the seat-position confound cancels exactly (same technique as Test 4 / the seat sweep).
- **Draft-and-hold:** no in-season transactions.
- **Weeks 1–17 only. Week 18 is excluded entirely** — never ingested, scored, or matched. (Regular season weeks 1–14; playoffs weeks 15–17.) The week window is a single config constant so a future league with a different playoff schedule can extend it.

**Non-goals (YAGNI v1), each a deliberate deferral:**
- No waivers / free agency / FAAB / trades (confounds draft quality with in-season management).
- No 2024 (fast follow — spike confirmed 2024 preseason is available; harness is season-parameterized).
- No UI.
- No keeper/dynasty, no custom scoring beyond the league's ruleset.
- No injury *model* — actual availability is encoded by whether ESPN published a weekly projection / `weekly_stats` row that week (bye/inactive ⇒ unstartable).

## 4. Data layer (2025, all point-in-time)

| Input | Source | Notes |
|---|---|---|
| Draft basis (projections) | ESPN preseason projected stat line | scored to **half-PPR**; 460 players validated |
| Draft basis (ADP) | **Sleeper ADP** | ESPN ADP is dead for past seasons (all 170) → **Sleeper-only ADP** for the backtest |
| Weekly start/sit | ESPN **weekly** projected stat line (`scoringPeriodId`, weeks 1–17) | scored to half-PPR; no entry ⇒ bye/inactive ⇒ unstartable |
| Weekly actuals | `weekly_stats 2025` | scored to half-PPR per player-week (the scoring layer) |
| Byes / availability | implicit | a player with no weekly projection AND/OR no actual that week cannot be started |

**The Sleeper-only-ADP divergence (explicit decision).** The live consensus blend averages ESPN + Sleeper ADP. ESPN ADP is unusable for completed seasons (sentinel 170), so the backtest's draft basis is built from **ESPN half-PPR projections + Sleeper ADP only**. This is a backtest-specific assembly (`draft_basis.py`), not a change to the live consensus pipeline. Bots draft by this Sleeper ADP — the real 2025 draft order.

**Fixed-VORP dependency.** The 2025 VORP table is built with the corrected starter-demand replacement (`fix/vorp-replacement-calibration`, commit `462e8c5`); the QB-inflation bug must not be present or the bot field and `raw_vorp` are distorted.

## 5. Architecture — `src/projections/backtest/`

A new durable sub-package (reusable for future projection/strategy evaluation), each module one clear purpose, communicating through validated frames / small dataclasses.

```
backtest/
  espn_weekly.py     pull + store ESPN weekly projected stat lines (wk 1-17) -> half-PPR weekly projection table
  weekly_actuals.py  score weekly_stats 2025 -> half-PPR actual points per (gsis_id, week)
  draft_basis.py     build the 2025 Sleeper-ADP consensus VORP table (fixed VORP, half-PPR)
  lineup.py          set best legal lineup by PROJECTION, score by ACTUAL (project->start, score->real)
  schedule.py        14-week H2H schedule generator + single-elim playoff bracket (wk 15-17)
  league.py          simulate one full league season: draft -> weekly points -> standings -> playoffs -> champion
  harness.py         run many seeds (mirrored seats), aggregate per-strategy win%/playoff%/champ%/PF + bootstrap CIs
scripts/h2h_backtest.py   CLI over harness
```

**Reuse (no duplication):** `DraftStrategy` + `NowOrNeverStrategy`/`SeasonValueStrategy`/`RawVorpStrategy`; the constrained ADP-bot draft logic (promote the validated scratch `_seatsweep.py`/`_mixed_sim.py` constrained-draft loop into a reusable `simulation` helper rather than re-hand-rolling); `roster_eligibility` for slot/eligibility; the scoring layer (`scoring.expected_points` / `score` under `Ruleset.espn_half()`); `store` for partition I/O; `id_map` for espn_id→gsis_id; `LeagueConfig`.

### 5.1 `espn_weekly.py`
- `refresh_espn_weekly_projections(season, *, weeks=range(1,18), ruleset)` → writes a store partition.
- Fetch: `_ESPN_URL.format(season) + "&scoringPeriodId={wk}"`, parse the per-player weekly projected stat line (statSourceId=1, statSplitTypeId=1, scoringPeriodId=wk), crosswalk espn_id→gsis_id, score the (fractional) stat line via **`scoring.expected_points(statline, Ruleset.espn_half())`** → `projected_points`. (Projections are fractional, so `expected_points`, not the integer `score`.)
- Output `WeeklyProjectionSchema`: `gsis_id, season, week, position, projected_points` (nullable — absent ⇒ bye/inactive). **Week 18 excluded by the `weeks` range.**

### 5.2 `weekly_actuals.py`
- `build_weekly_actuals(weekly_stats, *, ruleset, weeks=range(1,18))` → `WeeklyActualSchema`: `gsis_id, season, week, actual_points` (half-PPR), one row per player-week that has a `weekly_stats` row. `weekly_stats` rows are **integer** stat lines, so score via **`scoring.score(StatLine, Ruleset.espn_half())`** (the integer scorer), not `expected_points`. **Does not re-implement scoring.**

### 5.3 `draft_basis.py`
- `build_2025_draft_basis(*, ruleset, league_config, data_root)` → a `VorpTableSchema` frame: ESPN half-PPR projections → season projection → `generate_vorp_table` (fixed VORP) with **Sleeper ADP** attached as `consensus_adp`. Mirrors `_make_half_vorp.py` but Sleeper-ADP-sourced.

### 5.4 `lineup.py`
- `weekly_lineup_points(roster_positions, projections_wk, actuals_wk, roster_slots) -> float`: greedily assign roster slots **by projection** (restrictive-slot-first — the order that maximizes *projected* points for laminar eligibility, i.e. the lineup a rational manager sets), then **sum the assigned players' ACTUAL points**. The fill-by-projection / score-by-actual split is the one genuinely new bit vs `optimal_lineup_points` (fills and scores by the same value).
- **Required edge-case behavior** (not just tested — load-bearing for correctness):
  - A player with **no projection that week** (bye/inactive) is **not startable** — excluded from the fill.
  - A player who is started (had a projection) but has **no actual** that week (didn't play / scratched) **scores 0** — the manager's lineup decision stands, the points don't.
  - If too few startable players exist to fill every slot, **unfilled slots score 0** (a real manager left a hole).
  - A drafted player who has **no projection in any week** is permanently unstartable (accepted — a dead-weight bench pick, which is itself signal about draft quality).

### 5.5 `schedule.py`
- `regular_season_schedule(n_teams, n_weeks, rng) -> list[list[(seat,seat)]]`: a balanced rotating pairing (each week the 16 teams split into 8 matchups), deterministic given rng.
- `playoff_bracket(seeds, playoff_weeks) -> Bracket`: single-elimination over the top-N seeds (N a config constant, default 6 with byes for the top 2), weeks 15–17.

### 5.6 `league.py`
- `simulate_league(seed, seat_strategies, draft_basis, projections, actuals, league_config, *, calendar) -> LeagueResult`:
  1. Snake-draft (mixed field) → 16 rosters.
  2. Weekly points weeks 1–17 via `weekly_lineup_points`.
  3. Regular-season standings (weeks 1–14): W/L per matchup, tiebreak points-for.
  4. Playoffs (weeks 15–17): seed by standings, run the bracket, crown champion.
  - `LeagueResult`: per-seat `strategy` label, `wins`/`losses`, `points_for`, `made_playoffs` (bool), `is_champion` (bool). **No `finish` field in v1** — the headline metrics (champion / playoff / win%) don't need a full 1–16 ordering of non-playoff teams, which would require an arbitrary consolation-bracket rule. Defer if a finish distribution is ever wanted.

### 5.7 `harness.py`
- `run_backtest(*, n_seeds, strategy_n_sims, ...) -> BacktestResult`: for each seed run `simulate_league` with the §3 **mirrored seat layout** (odd seed = base layout, paired even seed = nn↔sv swapped) so each strategy sees identical seat exposure. Aggregate per strategy: championship %, playoff %, regular-season win %, mean points-for — each with a percentile-bootstrap CI (mirrors `tournament._bootstrap_mean`). Headline: championship % and win %.
- **`strategy_n_sims`** (default **200**) is the season_value seats' per-pick Monte-Carlo depth; it **must use the vectorized fast-path** (`_vectorized_lineup_points`) or the draft is intractable. Runtime is dominated by the season_value seats' draft MC: ~`2 × n_seeds × 4` sv-seat-drafts (≈1,600 at 200 paired seeds) — comparable to the ~20-min seat sweep, so **default 200 seeds × 200 sims is tractable (~tens of minutes)**. `n_seeds` and `strategy_n_sims` are CLI flags; the weekly-sim and nn/raw_vorp seats are cheap by comparison.

## 6. Determinism, randomness & confidence

Player weekly projections and actuals are **fixed** (real 2025). The only randomness per seed is (a) draft order via bot ADP jitter and (b) schedule pairing. Many seeds (default 200) yield rate distributions with CIs. **Inherent limitation, stated in outputs:** a single real season carries outcome luck (the players who smashed in 2025 are fixed); the 2024 fast-follow adds cross-season robustness. CRN where applicable (shared player-outcome tables across all seeds).

## 7. Schemas (validated at boundaries, `strict="filter"` + reassignment)

- `WeeklyProjectionSchema`: `gsis_id (pyarrow str), season (Int64), week (Int64, ge=1 le=17), position (pyarrow str), projected_points (Float64, nullable)`.
- `WeeklyActualSchema`: `gsis_id, season, week (ge=1 le=17), actual_points (Float64)`.
- Results are small frozen dataclasses (`LeagueResult`, `BacktestResult`, `Interval`) — no new persisted pandera frame for outputs (mirrors the tournament's dataclass results).

## 8. Testing strategy (TDD throughout)

- `lineup`: fill-by-projection / score-by-actual correctness; bye/no-projection exclusion; FLEX/SUPER_FLEX fill order; a player projected high but absent (no actual) scores 0.
- `schedule`: balanced (every team plays each week, no self-match, roughly even opponent distribution); deterministic given rng; playoff bracket correctness (seeding, byes, elimination).
- `weekly_actuals` / `espn_weekly`: scoring matches the ruleset on a synthetic stat line; week 18 excluded; missing-week ⇒ null projection.
- `league`: full season on a tiny synthetic fixture (4 teams, 2 weeks) → hand-verified standings, champion.
- `harness`: aggregation + mirrored-seat pooling + bootstrap CI shape on a fixture.
- Network pulls (`espn_weekly` live fetch): an opt-in `--run-network` smoke guarding ESPN payload-shape drift (mirrors the existing api-drift test pattern); unit tests parse a saved fixture payload.

## 9. Phasing (for the plan)

1. **Data layer:** `espn_weekly` (+ pull script, weeks 1–17), `weekly_actuals`, schemas. Land the 2025 weekly projection + actual tables.
2. **Draft basis:** `draft_basis` — 2025 Sleeper-ADP half-PPR fixed-VORP table.
3. **Lineup + schedule:** `lineup` (project→start/score→actual), `schedule` (regular + playoffs).
4. **League + harness:** `league` (one full season), `harness` (many seeds + aggregation), CLI.
5. **Run + report:** execute the 2025 backtest; write results into `reports/draft_strategy_tests.md` as Test F1 (no verdict — the process judges at the end).

## 10. Dependencies & risks

- **Branch dependencies:** requires `SeasonValueStrategy` (branch `feat/depth-aware-draft-strategy`) and the VORP fix (`fix/vorp-replacement-calibration`). The implementation branch must be based on a tree containing both (neither is on `main` yet). Sort landing order before execution.
- **ESPN weekly payload drift / soft-block:** mitigated by the saved-fixture unit tests + opt-in network smoke; pulls are read-only public API.
- **Sleeper↔gsis crosswalk coverage:** Sleeper provides ADP keyed by sleeper_id; rookies/edge players may miss the id_map (same risk the consensus blend already handles). Log coverage; ADP-less players fall back like the live path.
- **Single-season outcome luck:** acknowledged; 2024 fast-follow planned.

## 11. Open questions

None blocking. Playoff size (default 6, top-2 bye) and regular-season length (14) are config constants tunable without redesign.
