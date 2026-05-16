# Draft Ready Checklist — 2026 Season

**Status as of 2026-05-15.** Tracking what we need before our 2026 draft. Update inline as items move.

Legend: `[x]` done · `[~]` partial / works but with gaps · `[ ]` not started

---

## TL;DR — current capability vs draft-day need

The projection **core** is built and produces probabilistic weekly + season distributions for QB / RB / TE / WR on the 2024 data we've ingested. We do **not** today have:

- 2025 season ingested (raw store stops at 2024 — see `data/raw/weekly_stats/`).
- Any pre-season roster source for 2026 (current pipeline keys on `depth_charts`, which only exist after games are played).
- Rookie projections (no first-year-player path; models train on trailing-N history).
- K / DST projections (positions never built — see TODO #10).
- Draft Hub, start/sit tool, waiver tool, trade tool (all in CLAUDE.md "planned" list; none scoped).
- ESPN league API integration (planned, never started).

Everything below decomposes those gaps into actionable items.

---

## 1. Predict stats and fantasy points for each player for 2026

**Goal:** for every 2026-relevant player, a probabilistic forecast of weekly and full-season fantasy points under our scoring ruleset(s).

### 1a. Data the prediction depends on

- [ ] **Ingest 2025 season.** `data/raw/` currently stops at 2024. Trailing-N features and 2025-baseline residuals require this. Action: run `refresh()` for 2025 (and 2026 partials as they appear). Verify the opt-in network smoke (`pytest -m network --run-network`) catches any `nfl_data_py` column drift first — eight ingest patches were needed last time (TODO #16).
- [ ] **Pre-season roster source for 2026.** Today's `predict_2024.py` uses `dc_curr = read_partition("depth_charts", season=2024)` — pre-season 2026 has no depth chart yet. Pick a source and ingest it:
  - ESPN league API rosters (preferred — needed anyway for league-aware tools).
  - Sleeper rosters API (no auth required).
  - Manual import of FantasyPros/RotoWire preseason depth charts.
- [ ] **2026 rookies pipeline.** `draft_picks` ingest exists, but builders join rookies onto trailing weekly_stats and produce all-NaN features. Decide and implement:
  - Analog-based projection (nearest-neighbor match on draft capital + combine).
  - ADP-anchored prior with a wider variance band until games are played.
  - Or: explicitly exclude rookies from v1 draft tooling and warn at the surface.
- [ ] **K and DST.** TODO #10 has been open since project start. Decide: build degraded v0 from `implied_team_total` only, or ingest the missing data (FG-by-distance, team-level PBP for DST). v0 from `implied_team_total` is the fast path and unblocks full-roster drafts.

### 1b. Code surface

- [~] **Per-position weekly projection — QB / RB / TE / WR.** Works today via `scripts/predict_2024.py`. Production routing per position (`POSITION_DISPATCH[*].default_model_class`): QB → `lightgbm-nb`, RB → `baseline`, TE → `baseline`, WR → `ensemble`.
- [ ] **Generalize `predict_2024.py` to `predict_season.py SEASON`.** The constant `_PROJECTION_SEASON = 2024` is hardcoded; the script also assumes `depth_charts` partition exists for the target season. Needs a `--roster-source` flag plus a path that doesn't depend on in-season depth charts for pre-season runs.
- [x] **Season aggregation.** `aggregation/season.py:aggregate_to_season` rebuilds per-week samples from per-row seeds and sums them — gives `season_mean / p10 / p50 / p90`.
- [x] **Multi-ruleset scoring.** `Ruleset.espn_ppr() / espn_half() / standard()` already wired through scoring + season aggregation.
- [ ] **One end-to-end "produce 2026 season projections" entry point.** Currently you'd run `predict_season.py` per position, four times, then concat + aggregate by hand. Wrap into one script.

### 1c. Known model gaps (informational — not blockers, decide if any of these get re-opened before draft)

- WR decomposed-baseline / ensemble-decomposed factories are registered (commits `3250284`, `a70fc4b`) but a parallel session is still finishing the spec at `docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md`. Don't touch.
- Composite [p10, p90] under-covers by ~6pp on RB/TE/WR (Plan 5c / Plan 6 verdicts). Mean and rank are fine — draft, start/sit, and waiver tools consume those. DFS GPP construction would care about the tail; defer.

---

## 2. Draft rankings and recommendations (snake + auction)

**Goal:** given our projection outputs, produce per-player draft values and live pick recommendations for both snake and auction formats.

### 2a. Foundational valuation

- [ ] **Replacement-level / VORP per position.** From season-mean projections, derive `mean_fpts − replacement_fpts(pos, league_size, roster_slots)`. Need league config (size, slots, scoring) as an input.
- [ ] **Tier breaks.** Cluster players within position by season-mean (or by 90th-percentile for upside formats) so the live recommender can flag "last player in tier X" cliffs.
- [ ] **Confidence bands per ranking.** Translate `season_p10 / p90` into a "floor rank" and "ceiling rank" so the UI can show range, not just point estimate.

### 2b. Snake draft

- [ ] **Pre-draft cheat sheet.** Per-position ordered list with VORP, ADP delta, tier, and confidence band. Static export (CSV / markdown) is enough for v1.
- [ ] **Live snake recommender.** Given current roster + remaining player pool + next pick number, recommend top-K picks. Two viable approaches:
  - Greedy: highest-VORP available at a position of need.
  - Lookahead: simulate opponents picking via ADP, optimize expected total VORP across remaining rounds.
- [ ] **ADP source.** Needed for both the cheat sheet and the live lookahead. FantasyPros has a free CSV; Sleeper API exposes it too.

### 2c. Auction draft

- [x] **Dollar value generator.** Convert VORP to $ with a budget-conservation constraint (sum of $ values across rostered players = league total budget − $1-per-slot reserve). Standard SOS algorithm; well-defined once VORP exists. Shipped on `feat/auction-values` (2026-05-16): `src/projections/draft/{league_config,auction}.py` + `scripts/generate_auction_values.py` + `tests/test_draft/` (33 tests) + `tests/test_scripts/test_generate_auction_values_cli.py` (3 tests) + two example `configs/league_*.json`. Strategy-agnostic algorithm A (per spec `2026-05-16-auction-values-design.md`). **Output is gated on VORP spec landing** — script errors clearly if the VORP parquet is missing or shaped wrong.
- [ ] **Live auction bid recommender.** Given current nominated player + your remaining budget + remaining roster slots + remaining player pool, return `max_bid` (the price above which it's a value loss). Needs the dollar-value generator + a max-budget-per-remaining-slot constraint.
- [ ] **Nomination strategy helper.** Suggest who to nominate to drain other rosters' budgets early — secondary priority.

### 2d. Surface

- [ ] **CLI or notebook entry point.** Doesn't need to be a UI for v1 — a function that takes `(league_config, current_roster, available_players)` and returns ranked recommendations is enough. Wrap in a notebook for live use on draft day if a real UI doesn't land in time.

---

## 3. Lineup start/sit recommendations

**Goal:** for any given week, output the optimal starting lineup from a given roster under league lineup rules.

- [ ] **Lineup constraint engine.** Take roster slots (`QB, RB, RB, WR, WR, WR, TE, FLEX, SUPER_FLEX, K, DST, BENCH×N`) and a player pool with weekly projections → solve for the highest-expected-points legal lineup. `RosterSlot` enum already exists in `schemas.py`.
- [ ] **Tie-break policy.** When two players have nearly-identical means, prefer the one with higher floor (p10) for cash / lower variance, or higher ceiling (p90) for DFS-GPP. Make this a flag.
- [ ] **"Confidence in start" output.** Probability that player A outscores player B = Monte Carlo over their joint sample sets (currently independent; correlation modeling is TODO #1 D). Useful UI element.
- [ ] **Bye-week + injury filter.** Need a current injury/inactive source (Sleeper API, ESPN, or NFL official feed) at the per-week boundary.

---

## 4. Waiver wire and trade recommendations

**Goal:** for the user's actual ESPN league, identify pickups worth claiming and evaluate proposed trades.

### 4a. League integration (load-bearing for both)

- [ ] **ESPN league API client.** Read roster, free-agent pool, league settings, FAAB balance, transactions. `espn-api` (python package) is the most-used OSS client. Auth via `SWID` + `espn_s2` cookies.

### 4b. Waiver

- [ ] **Rest-of-season (ROS) projection.** Sum weekly projections from `as_of_week` to season end. Already derivable from existing weekly outputs.
- [ ] **Value-over-roster (VOR).** For each free agent, compute `ROS_VOR = ROS_proj(FA) − ROS_proj(weakest startable on roster at same position)`. Rank descending.
- [ ] **FAAB bid suggester.** Map VOR + league bid history (if available) to a recommended bid range. Heuristic OK for v1.

### 4c. Trade

- [ ] **Trade evaluator.** Given a proposed trade (players_out, players_in), compare ROS_VOR delta on both sides. Account for positional scarcity (a WR3 → WR1 upgrade is worth more than the raw fpts delta in a 3-WR league).
- [ ] **Trade finder.** Scan league rosters for plausible win-win trades (proposing user gains ROS_VOR, target user fills a positional weakness). Secondary priority.

---

## 5. Cross-cutting nice-to-haves (not blockers)

- [ ] **One canonical league config file.** League size, roster slots, scoring rules, draft format. Threaded through all four tools above.
- [ ] **A "what changed" digest.** After each weekly ingest, diff projections vs prior week — surfaces injuries / depth-chart shakeups automatically.
- [ ] **Backtest the draft tool on 2024.** Given 2024 ADP + our 2024-prediction-as-of-week-0 outputs, would our snake recommender have built a top-quartile roster against historical ESPN league outcomes? Not strictly needed but worth doing before trusting the tool in a live draft.

---

## Recommended ordering before draft day (~Aug/Sept 2026)

1. **Ingest 2025 data** (unblocks everything; one-day task assuming `nfl_data_py` doesn't drift hard).
2. **Pre-season roster source + rookie path + K/DST v0** (the three remaining "can we even score every drafted player" gaps).
3. **`predict_season.py 2026` end-to-end** (the single command that has to work).
4. **VORP + cheat sheet + snake recommender** (this is "draft tool v1" — gets you through draft day even without the rest).
5. **Auction $ values + live bid recommender** (auction-draft branch — same VORP, different surface).
6. **Start/sit lineup optimizer** (post-draft, week-of-season).
7. **ESPN API client → waiver + trade tools** (post-draft, ongoing).

Items 1-4 are the genuine pre-draft critical path. 5 is critical only if any of our leagues is auction. 6-7 can land after Week 1.
