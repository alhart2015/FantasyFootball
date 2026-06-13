# Draft Strategy Tests — running log

**Purpose.** Accumulate evidence on which draft strategy is best (`now_or_never` vs `season_value` "greedy" vs `raw_vorp`), across league sizes, bot models, metrics, and seats. **No final verdict until enough tests are in** — each entry records only *what it favors in isolation*. The judgment call comes at the end, weighing all rows together.

**Process rule (2026-06-12):** run many tests, log each one's parameters + result here, do **not** flip conclusions or docs after any single test. Decide at the end.

**Mid-stream corrections (2026-06-12) — Tests 1–5 are now superseded on two axes:**
1. **Scoring: full PPR → half PPR.** Tests 1–5 ran under `espn_ppr` (full PPR); the league is **half-PPR** (`espn_half`). Half-PPR = full − 0.5×receptions (rulesets differ only in `reception_pts`). Going forward: `_league_16_half.json` + `data/_vorp_16team_half.parquet`.
2. **VORP QB-inflation bug, found + fixed** (commit `462e8c5`, branch `fix/vorp-replacement-calibration`). The replacement level was the best non-pool player, and `_select_pool`'s uncapped BENCH pass flooded ~33 QBs into a 1-QB pool → QB replacement at ~QB#34 (95 full / 45 half) instead of ~QB#21 → QBs swept VORP, auction gave QBs 27% of budget. Fixed: replacement = `round(bench_cushion × starter_demand)`, bench-independent; default cushion 1.3 (→ QB#21), exposed as `bench_cushion` param. Validated on real 2026 data: QB repl 45.5→247.6; top-16 by VORP went all-QB → 14 RB + 2 WR (now matches ADP). All 211 draft tests pass; 2 regression tests added. **Tables regenerated with the fix.** *(Follow-up: empirically fit the cushion from prior-season draft histories; regenerate committed `data/consensus_vorp_2026.parquet`.)*
3. **Impact on Tests 1–5's nn-vs-sv verdicts:** likely small (season_value prunes per-position + ranks by marginal → robust to QB-VORP inflation; now_or_never is ADP-gated). raw_vorp (Test 2 control) was the main victim and should improve. But all 1–5 are full-PPR + buggy-VORP; **the baseline is re-established under half-PPR + fixed VORP starting with the seat sweep below.**

**Standing decision (2026-06-12):** **bots are ALWAYS constrained-ADP; pure-ADP bots are rejected as unrealistic** (a pure-ADP bot will draft, e.g., 5 RBs and 0 TE — nonsense rosters that inflate how easy the field is to beat). Consequence: **Tests 1 and 3 used pure-ADP bots and are therefore deprecated** — including the *committed 12-team validation* (Test 1), which means the committed "now_or_never wins" verdict rests on a rejected bot model. All future tests use constrained bots. F4 re-runs Test 3 with constrained bots (the corrected version).

## Strategies under test

- **`raw_vorp`** — best-available by VORP (static positional scarcity; blind to pick timing). Control.
- **`now_or_never`** — `score = vorp − E[best survivor at position by my next pick]` (opportunity-cost / dynamic scarcity layer on VORP). Survival = `LogisticSurvival(sigma)`, default `sigma = ⅔·n_teams`.
- **`season_value`** ("greedy") — ranks each candidate by **marginal expected season points** it adds to the current roster (`V(roster+c) − V(roster)`, common random numbers), top-k-by-VORP pruning, no pick-timing layer. Built to maximize the season metric directly.

## Metrics (how a roster is scored)

- **starters** — optimal single-week starting-lineup points (`optimal_lineup_points`). Bench scores nothing.
- **season (paired points)** — expected season points under per-player availability (injury Bernoulli + byes), filling the best legal lineup each week (`expected_season_points_crn`). This is `season_value`'s own objective.
- **league-finish rank** — rank of the hero roster among all teams in one simulated league (1 = beat everyone), rosters scored by the season metric with shared CRN draws.

## Bot models

- **pure-ADP** (shipped `bot_pick`) — noisy ADP, no roster constraints. A bot can over-draft one position (5 RBs, 0 TE). **REJECTED as unrealistic (2026-06-12) — do not use; any test using it is deprecated.**
- **constrained-ADP** (scratch `bot_eligible`) — noisy ADP + per-position min/max (`MINP={QB:1,RB:3,WR:3,TE:1}`, `MAXP={QB:3,RB:6,WR:6,TE:3}`). Realistic balanced rosters. **The only sanctioned bot model.**

## Leagues / data

- **12-team:** `configs/league_espn_ppr_12team_skill.json`, consensus VORP table, 2026, asof 2026-06-09.
- **16-team:** `_league_16.json` (`{QB:1,RB:2,WR:2,TE:1,FLEX:1,BENCH:4}`, roster_size=11, espn_ppr), `data/_vorp_16team.parquet`, 2026, asof 2026-06-09.

---

## Test log

### Test 1 — Committed validation (12-team, paired head-to-head)
- **Source:** `reports/depth_aware_strategy_validation_2026.md`, PM entry 2026-06-11.
- **Setup:** shipped `run_tournament`, paired-seed bootstrap. League 12-team. Bots **pure-ADP**. Metric **season (paired points)** (also ran starters). 80 seeds, n_sims=200. Strategies: now_or_never, raw_vorp, season_value. Hero seats 1, 6, 12.
- **Results (season metric, season_value vs now_or_never paired diff):**
  - slot 1: **tie**
  - slot 6: now_or_never **+22.8** [+13, +32]
  - slot 12 (turn): season_value **+9.1** [+1.7, +16.5]
  - starters metric: season_value **worst** of the three at slots 1/6.
- **Favors (isolated):** **now_or_never** (wins the contested mid/long-wait seat; sv wins only the turn).

### Test 2 — Scratch 16-team league-finish sweep
- **Source:** `_league_sim.py`, `_sweep_results.txt`, `_sim_results.txt`.
- **Setup:** 16-team. Bots **constrained-ADP**. Metric **league-finish rank**. Hero seat rotates over all 16 seats. HERO_NSIMS=200, SCORE_NSIMS=500, 100 drafts/cell. σ ∈ {2,4,8,12} where σ sets **both** the ADP jitter SD **and** now_or_never's survival sigma. Each strategy run as the sole hero vs the bot field.
- **Results (WINs/100, mean finish rank):**

  | σ | raw_vorp | now_or_never | season_value |
  |---|---|---|---|
  | 2 | 0 / 10.15 | 70 / 1.78 | 92 / 1.08 |
  | 4 | 1 / 10.06 | 72 / 1.61 | 92 / 1.09 |
  | 8 | 2 / 7.94 | 89 / 1.22 | 95 / 1.05 |
  | 12 | 7 / 5.82 | 94 / 1.15 | 95 / 1.05 |

- **Favors (isolated):** **season_value** (dominates at every σ; gap to now_or_never narrows as the field gets noisier).
- **Caveats:** (1) the rank metric scores rosters by `season_value`'s **own objective** → partly circular (sv TOP4=100/100 every cell is the tell); (2) constrained bots, not the shipped pure-ADP; (3) averaged over all seats; (4) σ couples jitter and survival sigma. Confounded vs Test 1 on ≥3 axes.

### Test 3 — Clean reconciliation (16-team, shipped paired head-to-head)
- **Source:** `_reconcile.py`, `_reconcile_results.txt`, `_reconcile_slot{1,8,16}.json`.
- **Setup:** shipped `run_tournament`, paired-seed. League 16-team. Bots **pure-ADP**. Metric **season (paired points)**. **jitter SD = 8**; now_or_never survival sigma = **default** ⅔·16 ≈ 10.67 (decoupled from jitter, unlike Test 2). 80 seeds, n_sims=200, base_seed=0. Strategies: now_or_never vs season_value. Seats 1 (wing), 8 (mid), 16 (turn).
- **Results (paired diff, season metric):**

  | Seat | now_or_never | season_value | paired diff | winner |
  |---|---|---|---|---|
  | 1 (wing) | 1579.6 | 1556.5 | nn +23.0 [16.1, 29.6] | now_or_never |
  | 8 (mid) | 1555.3 | 1579.3 | sv +24.0 [11.1, 37.4] | season_value |
  | 16 (turn) | 1460.0 | 1562.4 | sv +102.4 [90.6, 114.3] | season_value |

- **Favors (isolated):** **season_value** (2 of 3 seats; now_or_never only at the extreme wing). now_or_never craters at the turn (1460); season_value is seat-robust (~1556–1579 everywhere).
- **Note:** isolates **league size** (12→16) vs Test 1 — same machinery, metric, and bot model. Residual confound vs Test 2: bot model (pure-ADP here vs constrained there).

### Test 4 — Mixed-field 16-team draft (all strategies drafting together)
- **Source:** `_mixed_sim.py`, `_mixed_{A,B}.json`, `_mixed_results.txt`.
- **Setup:** one 16-team league, 8 ADP bots + 4 now_or_never + 4 season_value drafting in the **same** snake draft. Bots **constrained-ADP**. Metric **season (paired points)**, all 16 rosters scored with shared CRN draws per draft. jitter SD = 8; now_or_never survival sigma = default ≈ 10.67; sv strategy n_sims=150, score n_sims=400. Seat-fair via **mirrored assignments** (A: nn at {2,6,10,14}, sv at {4,8,12,16}; B swapped), pooled. 25 drafts/variant = **50 drafts**.
- **Results (pooled, 16 teams/draft):**

  | strategy | mean finish rank | mean score | win% | top-4% | top-8% |
  |---|---|---|---|---|---|
  | season_value | **3.50** | 1524.5 | 15.5 | 69.0 | 100.0 |
  | now_or_never | 6.47 | 1462.9 | 9.5 | 29.5 | 68.0 |
  | adp bots | 12.01 | 1332.0 | 0.0 | 0.8 | 16.0 |

  Per-draft league-win share (some seat of that group finishing #1): season_value ≈ 62%, now_or_never ≈ 38%, adp ≈ 0%. season_value finished **top-8 in 100% of seat-drafts**; mean-score gap sv − nn ≈ +62 pts.
- **Favors (isolated):** **season_value** (best mean rank, never below 8th; now_or_never clearly 2nd, both crush the ADP bots).
- **Caveats:** (1) season metric is sv's **own objective** → same partial circularity as Test 2; (2) constrained bots (like Test 2), not pure-ADP (Tests 1/3); (3) head-to-head in one field rather than paired counterfactual — measures "do sv rosters out-score nn rosters in the same draft," which they do.

### Test 6 — Full seat sweep, HALF-PPR + FIXED VORP (the corrected baseline)
- **Source:** `_seatsweep.py`, `_seat_results.txt`, `_seat_result_slot{1..16}.json`.
- **Setup:** paired now_or_never vs season_value at **every** seat 1–16. **Half-PPR** (`_league_16_half.json` + `data/_vorp_16team_half.parquet`) with the **fixed VORP** (commit `462e8c5`). Constrained bots, season metric (`SeasonValuer`), jitter=8, sigma=default, 80 seeds, n_sims=200, base_seed=0. First test on the corrected data (right ruleset + un-inflated QBs).
- **Results (sv − nn, season metric):**

  | seat | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | sv−nn | +11 | +9 | +1 | −1 | +14 | +35 | +36 | +46 | +62 | +66 | +83 | +87 | +101 | +108 | +122 | +120 |
  | winner | sv | sv | tie | tie | sv | sv | sv | sv | sv | sv | sv | sv | sv | sv | sv | sv |

- **Favors (isolated):** **season_value** — wins 14/16 seats, ties slots 3–4, loses none. season_value is seat-robust (≈1390–1419 all seats); now_or_never decays with wait-time (peak 1402 @ slot 4 → 1279 @ turn). Gap grows monotonically with seat.
- **Contrast:** *stronger* for sv than the buggy full-PPR tests — Test 5 had now_or_never winning slot 1 (+24.7); here sv wins slot 1 (+11.4) and nn wins no seat outright. Correcting the ruleset + QB-VORP inflation shifted the picture toward season_value.
- **Caveat (unchanged):** season metric is sv's own objective → partial circularity persists. The H2H sim (F1) with real start/sit is the first metric that breaks it.

### Test 5 (F4) — 16-team paired head-to-head, CONSTRAINED bots (corrects/supersedes Test 3)
- **Source:** `_f4_constrained.py`, `_f4_results.txt`, `_f4_result_slot{1,8,16}.json`.
- **Setup:** identical to Test 3 (16-team, paired-seed, season metric via `SeasonValuer`, jitter=8, sigma=default, 80 seeds, n_sims=200, seats 1/8/16) **except bots are constrained-ADP** — each bot pick filters `available` to roster-legal positions, then calls the shipped `bot_pick`. The minimal swap, so the diff vs Test 3 = pure bot-model effect.
- **Results (paired diff sv − nn, season metric):**

  | Seat | now_or_never | season_value | sv − nn | winner | (Test 3 pure-ADP) |
  |---|---|---|---|---|---|
  | 1 (wing) | 1580.2 | 1555.5 | **−24.7** [−31.4, −17.2] | now_or_never | (nn +23.0) |
  | 8 (mid) | 1554.1 | 1578.9 | **+24.8** [+12.8, +38.3] | season_value | (sv +24.0) |
  | 16 (turn) | 1462.6 | 1562.2 | **+99.6** [+87.3, +112.3] | season_value | (sv +102.4) |

- **Favors (isolated):** **season_value** (2 of 3 seats; now_or_never only at the extreme wing) — same as Test 3.
- **Finding:** the bot model (constrained vs pure-ADP) has **negligible effect on the paired head-to-head** (all three seats within ~3 pts of Test 3). In a paired comparison both heroes face the same field, so the bot model cancels in the diff. ⟹ (a) the league-size effect between Test 1 (12-team) and Tests 3/5 (16-team) is **real, not a bot-model artifact**; (b) bot model matters for *absolute* scores and the *rank* metric (Test 2), not paired diffs. **This is the sanctioned 16-team paired result; Test 3 is retired.**

### Test 7 (F1) — Head-to-head 2025 backtest, REAL outcomes (the realistic objective) — **HIGH VALUE**
- **Source:** `scripts/h2h_backtest_chunked.py`, `_h2h_run.txt`, `_h2h_ckpt/` (200 seeds × 200 strategy-n-sims). Harness: `src/projections/draft/backtest/`.
- **Setup:** 16-team **half-PPR** mixed field — **4 now_or_never + 4 season_value + 8 constrained-ADP bots**, interleaved & **mirror-paired** seats (odd seeds: nn at {2,6,10,14}, sv at {4,8,12,16}; even seeds swap nn↔sv; **bots always the 8 odd seats {1,3,…,15}**). **Genuine point-in-time 2025 data:** ESPN preseason half-PPR projections + **Sleeper-only ADP** → fixed-VORP draft basis; **ESPN weekly projections** set each week's lineup *without hindsight*; `weekly_stats 2025` half-PPR **actuals** score it. Draft-and-hold; regular weeks 1–14, playoffs 15–17 (top-6, top-2 bye); week 18 excluded. jitter=8, σ=default, base_seed=0.
- **Two scorings from the SAME drafts** (the point of F1): **projected** = the started lineup's *projected* points → *who drafted better* under the shared projections (no outcome luck / projection error); **actual** = *realized* points → real fantasy (also absorbs projection error + luck). Per-strategy rates with bootstrap CIs (marginal, not paired-diff).

  **PROJECTED points** (draft quality):
  | strategy | champ% | playoff% | win% | PF |
  |---|---|---|---|---|
  | season_value | 5.2 [3.7, 6.9] | **43.0 [39.6, 46.6]** | **53.6 [52.3, 54.9]** | **1106.7** |
  | now_or_never | 3.8 [2.5, 5.1] | 34.8 [31.4, 38.0] | 50.0 [48.7, 51.4] | 1085.6 |
  | bot (ref) | 8.0 [6.8, 9.4] | 36.1 [34.0, 38.6] | 48.2 [47.2, 49.3] | 1086.3 |

  **ACTUAL points** (real outcomes):
  | strategy | champ% | playoff% | win% | PF |
  |---|---|---|---|---|
  | season_value | 6.2 [4.6, 7.9] | **48.2 [45.1, 51.5]** | **54.1 [53.1, 55.0]** | **1068.6** |
  | now_or_never | 5.4 [3.9, 7.0] | 32.6 [29.6, 35.9] | 47.7 [46.6, 48.8] | 1010.2 |
  | bot (ref) | 6.7 [5.4, 8.0] | 34.6 [32.3, 36.9] | 49.1 [48.3, 49.9] | 1019.8 |

- **Clean contrast = now_or_never vs season_value only** (both at even seats, mirror-paired → seat confound cancels). **Bots are seat-confounded** (always the odd seats, incl. the 1.01) so their rates are a *reference field*, not a draft-quality signal — do **not** read "bots win the most titles per seat" as bots drafting well.
- **Favors (isolated):** **season_value on the regular-season axis — but NOT on championships.** In *both* scorings sv beats nn on **win%** (CI-separated: proj 53.6 vs 50.0; actual 54.1 vs 47.7) and **playoff-make%** (proj 43.0 vs 34.8; actual 48.2 vs 32.6), and carries the higher points-for. **Championship% is a statistical tie** (nn/sv CIs overlap in both metrics: proj 5.2 vs 3.8; actual 6.2 vs 5.4).
- **The load-bearing finding:** this is the **first metric that is not season_value's own objective**, so it breaks the circularity caveat — and sv's edge **survives on wins/playoff berths but does not convert into a title-rate advantage**. Consistent with the F1 thesis: the 3-week single-elimination playoff rewards *ceiling/variance*, which sv's high-floor (expected-points-max) rosters don't supply more of than nn or the noisier bot field. sv gets you to the dance more often; it doesn't win the dance more often. Projected and actual agree on the ordering (sv top on wins/playoff in both), so the result is not an artifact of projection error.
- **Caveats:** **single season** (2025 — player outcomes are fixed; outcome luck not yet averaged out; 2024 fast-follow is the cross-season check); bot↔strategy **seat confound** (only nn-vs-sv is mirror-controlled). Run completeness: all 40 checkpoints landed; **20 of ~60 chunk-attempts crashed (native access violation) and auto-recovered** via the resumable runner — results are complete and byte-identical to a monolithic run (pinned equivalence test).

#### Diagnostic — why nn ≈ bots in H2H but ≫ bots on the season metric (root-caused 2026-06-12)

The contradiction: in the **same mixed field, same seats** (Test 4), nn's rosters are worth **+131 season-points** over the bots (1462.9 vs 1332.0, rank 6.5 vs 12) — yet in H2H those *same rosters* tie/lose to the bots. Root cause, proven (`_diag_*.py`):

- **Not the H2H code.** Weekly-table coverage is 100% for nn (11/11 players startable every week; bots are slightly *less* covered). The `season_mean / Σweekly` gap is **uniform across positions** (QB 1.11, RB 1.13, WR 1.13, TE 1.08) → rules out a half-vs-full-PPR scoring bug.
- **The season metric scores rosters by the pool's preseason `season_mean_fpts`; the H2H scores by ESPN *weekly* projections + actuals.** These two ESPN products **disagree**, and the gap is concentrated in preseason-hyped players who busted (injury / role-loss): e.g. QBs projected 327–368 preseason but Σweekly ≈ 110–143 and Σactual ≈ 112–113 — i.e. **the weekly model tracked reality; the preseason number didn't.**
- **now_or_never is the strategy most exposed.** Roster-level `season_mean/Σweekly`: **nn 1.36 > sv 1.24 > bot 1.21.** nn carries **+13% by preseason projection (A) but +0% by weekly projection (B)** vs the bots. Mechanism: nn ranks by **VORP (raw `season_mean`) + ADP opportunity-cost with no risk term**, and its "grab the scarce stud now" logic *front-loads exactly the highest-preseason-projection players* — the ones most prone to regress/get hurt. `season_value` ranks by **`expected_season_points`, which already discounts injury availability** + balances depth → insulated.
- **Paired bootstrap (200 seeds, per-seed nn−bot, ACTUAL):** win% **−1.40 [−2.67, −0.11]**, playoff% −1.94 [−5.88, +2.12] (noise), champ% −1.31 [−3.25, +0.81] (noise). On *projected* the win% gap flips to **+1.85 [+0.43, +3.28]**. ⟹ **nn has no meaningful edge over a noisy-ADP bot** — sub-2-pt win% gap that *flips sign* between projected and actual; playoff/champ indistinguishable. (sv−nn win% **+6.38 [+5.09, +7.68]** actual, sv−bot **+4.99 [+3.85, +6.13]** — sv is the only strategy that genuinely separates.)
- **Implication for Tests 1–6 (the season-metric series):** the season metric is **circular** (scores rosters by the same preseason projections the strategies draft from) **and optimistically biased** (preseason hype the weekly model later marks down). It therefore **systematically overrates VORP/preseason-chasing strategies — now_or_never most of all** (+13% phantom edge). The sv>nn verdict survives (sv won even on the metric that *flattered* nn, and wins again on real outcomes); the **nn>>bots verdict does not** — it was the artifact. Treat F1's real-outcome scoring as the honest yardstick; read the season-metric rows of Tests 1–6 as confounded for any strategy-vs-field comparison.

#### Why nn ≈ bots specifically — decomposition (2026-06-12, `_diag_*.py`)

Bots in fact *edge* nn on actual win% (paired −1.40 [−2.67, −0.11]); the sign flips to +1.85 on projected. Chased down the mechanism and **ruled out**: **injuries** (nn roster availability 0.817 ≈ bot 0.810 — nn does *not* draft hurt-prone players), **week-to-week variance** (nn 18.3 ≈ bot 18.1), **roster concentration** (nn *less* concentrated), **QB-stashing** (real — nn 1.92 QB, 70% carry two — but `season_value` stashes *more*, 2.08 / 100%, and wins, so non-causal; the 2nd QB is a ~pick-148 flyer, not premium capital), and a **miscalibrated TE replacement** (TE repl #21 @ 107 pts is *higher* / more conservative than RB's #47 @ 98; elite-TE VORP 100 < WR1 150 < RB1 208 — TEs are not over-ranked by VORP, and the replacement level shifts every TE's VORP by a constant so it can't change the cliff that drives the chase).

**The actual lever — scarcity vs raw points.** Draft-capital allocation (capital = 176−pick): nn spends **23% on TE (217)** vs bots' 7% (70) / sv's 10% (101), funded by less RB/WR. nn's opportunity-cost layer (`vorp − E[best survivor]`) shifts capital toward TE because the TE talent cliff makes the wait-cost large — *exactly what now-or-never is built to do*. But in H2H/raw weekly points, RB/WR **volume** beats positional scarcity: the "elite" TE projected 207 and scored **142 (like a mid WR)**, so nn pays premium capital for scarcity that doesn't score. **Next lever (own slice): re-weight scarcity vs raw value in now-or-never, with an absolute value floor — below a quality bar a player isn't worth taking *no matter the wait-cost*** (the opportunity-cost term currently has no floor, so it over-recommends the best-of-a-thin-tier). Distinct from the two dead leads (injury discount ❌, QB cap ❌). See TODO #42.

---

## Standing tally (no verdict yet)

| Test | League | Bots | Metric | Favors |
|---|---|---|---|---|
| 1 | 12-team | ~~pure-ADP~~ DEPRECATED | season (paired) | now_or_never |
| 2 | 16-team | constrained | finish rank | season_value |
| 3 | 16-team | ~~pure-ADP~~ DEPRECATED | season (paired) | season_value (2/3 seats) |
| 4 | 16-team | constrained | season (mixed field) | season_value |
| 5 (F4) | 16-team | constrained | season (paired) | season_value (2/3 seats) — retires Test 3 |
| **6** | **16-team half-PPR + fixed VORP** | constrained | season (paired, all seats) | **season_value (14/16 seats, 2 ties)** — corrected baseline |
| **7 (F1)** | **16-team half-PPR, real 2025** | constrained (mixed field) | **H2H real outcomes (projected + actual)** | **season_value on win% + playoff-make% (both scorings, CI-separated); championship% a wash (nn ≈ sv); nn ≈ ADP bots (no edge)** — first non-sv-objective metric |

> **⚠️ Season-metric confound (root-caused by Test 7/F1, 2026-06-12).** Tests 1–6 all score by the **season metric**, which values rosters by the pool's **preseason** `season_mean_fpts` — the *same projections the strategies draft from* (circular), and *optimistically biased* (preseason hype the in-season weekly model later marks down). It **overrates VORP/preseason-chasing strategies, now_or_never most** (+13% phantom roster value vs bots that vanishes under weekly projections / actuals). **Robust across the confound:** `season_value > now_or_never` (sv won even when the metric flattered nn). **Killed by the confound:** `now_or_never ≫ bots` (Tests 2/4) — under real outcomes nn has **no meaningful edge over a noisy-ADP bot**. Use F1's real-outcome scoring as the honest yardstick. See the Test 7 diagnostic.

**Open threads to test before judging:** ~~a non-season metric for the mixed field to break the sv-objective circularity~~ (✅ Test 7/F1 — sv wins regular season, not championships); **2024 H2H backtest** (cross-season — does the championship wash hold, or does one season's outcome luck flip it?); 12-team re-run with the season metric at more seats; mixed-field at 12-team (does sv's edge shrink with fewer teams, mirroring Test 1?); sensitivity to n_sims (does sharpening sv's marginals change Test 1's slot-6 loss?); starters-metric cross-check at 16-team; **paired-diff (not marginal) CIs for the F1 nn-vs-sv championship contrast** (tighter test of the apparent tie).

---

## Future tests (backlog)

### F1 — Head-to-head season simulation (the realistic objective) — ✅ DONE → see **Test 7 (F1)** above. sv wins more games + playoff berths (both scorings); championship a wash (nn ≈ sv).
Every test above scores a roster by *expected season points* under availability, filling the **optimal** lineup each week with perfect hindsight. Real fantasy is won differently: you set a lineup each week from **projections** (imperfect start/sit), you score against **actual** weekly points, and you win or lose **head-to-head matchups** on a real schedule (record → playoffs → champion). A strategy that maximizes expected points may not maximize *wins* (variance/ceiling matters in H2H; a high-floor roster banks wins, a boom/bust roster needs ceiling weeks).

What an H2H sim needs:
1. **Draft basis** — preseason projections + ADP to run the draft (same as now).
2. **Weekly start/sit** — a weekly projection per rostered player each week, to choose the lineup *without hindsight* (this is the new ingredient; today's metric cheats by knowing the realized week).
3. **Weekly actuals** — realized fantasy points to score the chosen lineup (`weekly_stats`, already ingested).
4. **Schedule + byes** — per-team bye weeks and a league H2H matchup schedule (`schedules` partition for byes — 2018–2025 present; the *league* matchup schedule is generated, e.g. round-robin + playoffs).
5. **Metric** — regular-season W/L record, playoff seeding, championship rate — not total points. (Optionally also report points-for as a secondary.)

**Two sourcing options (decision needed):**
- **(a) 2025 backtest** — draft from **2025 preseason** consensus, start/sit from **2025 weekly** projections, score vs **2025 actuals**. Most honest (real outcomes, real variance). **Binding constraint:** we need a *2025 preseason* projection/ADP snapshot for the draft and *2025 weekly* projections for start/sit. The external-projection ingest only began writing dated snapshots for **2026** (built 2026-06-08); 2025 preseason ESPN/Sleeper ADP is **point-in-time** and may not be re-pullable now — check archived sources (FantasyPros historical, ADP archives) before committing. 2025 actuals + schedule we have.
- **(b) Use the home-grown weekly projection model** for start/sit. We built a **weekly** in-season projection model (BaselineModel + features); the external spike proved it *can't* do preseason (TODO #38) but it *is* a weekly model (its in-season accuracy is unmeasured — TODO #39). It could supply the per-week start/sit projections for a 2025 backtest, with the draft still seeded from external 2025 preseason consensus (if obtainable). Alternatively benchmark start/sit using **ESPN weekly** projections (TODO #39, `statSplitTypeId=1` per `scoringPeriodId`) — also point-in-time for 2025.

**Why it matters for the strategy question:** an H2H/wins metric is the first one here that is **not** season_value's own objective, so it tests whether sv's expected-points dominance actually converts to *wins* — or whether a timing/ceiling-aware strategy (now_or_never, or a future variance-aware one) wins more matchups despite fewer expected points. This is the test most likely to overturn or confirm the season-metric picture.

**Data availability — RESOLVED (2026-06-12), F1 fully unblocked for a 2025 backtest with real point-in-time data:**
- **Draft basis:** ESPN serves 2025 **preseason projections** (460 players, stat line + pos rank; verified preseason-valid — CeeDee ranked WR7, scored 200.9, no hindsight) via `pull_external_projections.py --season 2025`. **ESPN ADP is dead for past seasons (all 170)** — use **Sleeper ADP** (valid, 1.0→999, 8625 players). Score ESPN stat line to half-PPR + Sleeper ADP → 2025 consensus → 2025 VORP table (same pipeline as 2026, `--season 2025`).
- **Weekly start/sit:** ESPN serves 2025 **weekly projections** (`&scoringPeriodId={wk}`, statSourceId=1/statSplitTypeId=1): wk5 had 621 projected players, genuine point-in-time (Chase proj 16.1 / actual 29.0; byes→None). This is the signal a real manager had each week — the chosen start/sit source.
- **Actuals + schedule:** `weekly_stats 2025` (score to half-PPR) + `schedules 2025` (byes + matchup-schedule generation). ESPN weekly actuals (statSourceId=0) as cross-check.
- **2024 likely available too** (spike verified 2024 preseason) → a two-season backtest is possible.

**Remaining F1 build pieces:** pull ESPN weekly projections for all 2025 weeks; build 2025 consensus VORP draft basis (Sleeper-only ADP — ESPN ADP excluded); H2H schedule generator (round-robin + playoff bracket); lineup-setting = best legal lineup by ESPN weekly projection among non-bye/healthy each week; score vs half-PPR actuals; metric = regular-season record → playoff seeding → championship rate (NOT total points). Strategies: now_or_never vs season_value (+ raw_vorp control) in the hero seats vs constrained ADP-bot field.

### F2 — 12-team mixed field
Repeat Test 4 at 12 teams. Test 1 (paired, 12-team) favored now_or_never; does the *mixed-field* result also shrink sv's edge at 12 teams? Isolates league size for the mixed-field design.

### F3 — Starters-metric mixed field
Repeat Test 4 scored by the **starters** metric instead of season. Breaks the sv-objective circularity with an already-built metric; cheap.

### F4 — 16-team paired head-to-head, CONSTRAINED bots — ✅ DONE → see Test 5. Bot model washes out of paired diffs; sv wins 2/3 seats (mirrors Test 3).

### F6 — 12-team paired head-to-head, CONSTRAINED bots — **NEXT** (replaces deprecated Test 1)
The committed 12-team validation (Test 1) used rejected pure-ADP bots. Re-run it paired with constrained bots, season metric, seats 1/6/12, to get a sanctioned 12-team result. Prediction from Test 5 (bot model washes out of paired diffs): the constrained 12-team result should land close to Test 1's pure-ADP numbers (now_or_never wins the contested mid seat) — if so, league size really is the lever (now_or_never at 12-team, season_value at 16-team). Trivial variant of `_f4_constrained.py` (swap pool/config to 12-team + seats 1/6/12).

### F5 — n_sims sensitivity
Does raising sv's strategy n_sims (sharper marginals) change Test 1's slot-6 loss? Probes whether that loss is MC noise or structural (PM claims structural).
