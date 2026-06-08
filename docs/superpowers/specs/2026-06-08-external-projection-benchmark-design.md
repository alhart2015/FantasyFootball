# External Projection Benchmark Spike — Design

**Date:** 2026-06-08
**Branch:** `feat/external-projection-benchmark`
**Status:** spec
**Type:** spike (throwaway-tolerant; lives in `scripts/`, not `src/projections/` core)

**Predecessor / motivation:** The projection core (ingest → schemas → distributions → scoring → store → 5 model classes → backtest/adoption-gate) is complete, with `BaselineModel` (RidgeCV) as the production default. Roughly two months of recent work (PRs ~20–35) have been feature-family probes whose measured effects sit at **0.004–0.04 fantasy points per week** — below any threshold a human start/sit or draft decision can feel — while **zero user-facing tools** (Draft Hub, start/sit, DFS) have been built, and the model has **never been benchmarked against freely available projections.** This spike answers the one unasked question before any more modeling effort is spent: **is our model any better than free public projections?** The strategic goal it serves: if free/consensus projections match or beat ours, pivot effort to *how we use projections* (Draft Hub first, then start/sit, then DFS), consuming a free consensus as the input rather than grinding our own model.

---

## 1. Goal & success criteria

### 1.1 Goal

Produce a single verdict report answering: **does our `BaselineModel` preseason projection beat ESPN's preseason projection at predicting actual 2024 fantasy outcomes?** The comparison is strictly **preseason-vs-preseason** (apples-to-apples): both sides are forecasts made with no in-season information from the target year, evaluated against full-season actuals.

The verdict drives a go/no-go on **sub-project #2** (build the full external-ingest + multi-source consensus layer):
- **Ours clearly worse** → retire home-grown modeling; build the consensus layer as the projection basis and move to Draft Hub.
- **Ours clearly better** → keep the model; widen the bake-off to a multi-source consensus before concluding (consensus usually beats any single source, so a one-source win is not yet decisive).
- **Roughly tied** → the model isn't earning its complexity cost; lean toward consensus + tools, note it explicitly.

### 1.2 Why preseason-vs-preseason (the apples-to-apples constraint)

Free APIs serve the *current/latest* value, not historical as-of-date snapshots, so in-season-updated projections would be contaminated by information our static model never sees — making any updated source look unfairly good. The comparison is therefore valid **only** between forecasts that share the same information cutoff (before week 1).

- **Our side is clean preseason by construction.** `scripts/project_season.py --season 2024` trains `BaselineModel` per position on 2018–2023, projects all weeks of 2024, and sums weekly means to a season total. It uses no 2024 in-season data.
- **ESPN's season projection is verified clean preseason** (see §3.1). The probe of ESPN's 2024 season projection shows classic preseason misses in both directions — rookies under-projected (Bucky Irving 112→244, Brian Thomas Jr 172→284, Brock Bowers 144→263), breakouts missed (Ja'Marr Chase 289→403, Saquon 252→355), declines not foreseen (Tyreek Hill 299→218). Contaminated end-of-season numbers would instead hug actuals. Conclusion: ESPN's `statSourceId==1, statSplitTypeId==0, seasonId==2024` season projection is the genuine preseason forecast.

### 1.3 Success criteria

The spike **ships** when all of the following hold:

1. **Data joined and verified:** ESPN preseason projections, our model's projections, and actuals (from `weekly_stats` via the scoring layer) are joined on `GsisId` for 2024, QB/RB/WR/TE. Per-source **ID match rate** and **coverage** are reported (not hidden).
2. **Apples-to-apples scoring:** every projected and actual stat line is scored through **our** PPR ruleset (§3.4) — never each source's own points — so the fantasy-point comparison is under one consistent scoring rule.
3. **Both cohorts computed** (§2.2): full population and top-20-per-position-by-preseason-rank.
4. **Verdict report written** to `reports/external_projection_benchmark_2024.md` with the metric table (§2.3) and a plain-English go/no-go recommendation for sub-project #2.
5. **Verification gates green** for any code under `src/` (none expected — spike code lives in `scripts/`): mypy strict + ruff + ruff format clean on touched files; any new script has at least a smoke-level test for its pure transform functions (ID join, scoring pass-through, metric computation) using synthetic fixtures, so the metric math is guarded even though network pulls are not.

### 1.4 Out of scope (deferred to sub-project #2 or later)

1. **Multi-source consensus / averaging.** Free APIs give exactly **one** clean preseason stat-line source (ESPN); Sleeper exposes only preseason ADP + projected games, not a season stat line (§3.2). A true stat-line consensus needs a scraped second source (FantasyPros/CBS/NumberFire preseason pages), which is the build phase, not the spike.
2. **`ConsensusModel` in the core, distribution-wrapping of external point estimates, new pandera schemas in `schemas.py`.** The spike stays in `scripts/` and writes intermediate parquet ad hoc; it does not extend the core.
3. **Weekly start/sit accuracy and DFS/correlation.** Weekly projections are in-week (contaminated for a preseason test) and serve a different product; not part of the draft-first verdict.
4. **Additional seasons.** 2024 only. A second year (2023) is a fast follow-up if the 2024 verdict is close, but is not required to answer the question.
5. **K / DST.** Our model does not cover them.

---

## 2. Benchmark definition

### 2.1 The single comparison

For every player in the 2024 eval universe, compute a **season-total projected fantasy-point** value from each source and a **season-total actual fantasy-point** value, all under our PPR ruleset:

| Source | How the season-total projection is formed | Preseason-clean? |
|---|---|---|
| **ESPN** | ESPN preseason projected stat line (season split) → our scoring | Yes (§3.1) |
| **Ours** | `BaselineModel` 2024 weekly projected stat lines summed → our scoring | Yes (by construction) |
| **Actual** | `weekly_stats` 2024 actual stat lines summed → our scoring | n/a (ground truth) |

The decisive metric is the error of each *projection* column against the *actual* column.

### 2.2 Cohorts

Reported separately to avoid long-tail noise dominating the headline:

1. **Full population** — every QB/RB/WR/TE with an actual 2024 season and a match in at least one projection source.
2. **Top-20 per position by preseason rank** — the 20 highest-ranked players per position **by ESPN preseason positional rank / ADP** (a pre-known quantity; using *actual* finish to define the cohort would be lookahead bias). This is the draft-relevant cohort.

Coverage caveat to surface in the report: our model **cannot project pure rookies** (no prior-NFL features), while ESPN can. Several 2024 top finishers were rookies. The top-20-by-preseason-rank cohort will include some rookies our model has no row for; we report how many and treat "cannot see rookies" as a verdict-relevant structural weakness, not a row to silently drop.

### 2.3 Metrics

Per (source × position × cohort):

- **RMSE** and **MAE** of season-total projected vs actual fantasy points — the primary verdict.
- **Spearman rank correlation** of projected ranking vs actual finish — "did it get the *order* right," which for fantasy matters as much as absolute points.
- **Top-20 hit rate** (cohort-level) — of each source's preseason top-20-per-position, how many finished top-20 in actuals.

Rank lens (cheap, draft-relevant, reported alongside): Spearman + top-20 hit-rate using **ESPN ADP** and **Sleeper ADP** vs actual finish — "how well did each preseason *ranking* predict who'd be good," independent of stat-line projection quality.

### 2.4 Honest-reading notes the report must include

- The rookie-coverage asymmetry (§2.2).
- That this is one season; a single year can be lucky. A 2023 re-run is the cheap robustness check if the verdict is close.
- That ESPN is one strong source, not a consensus; losing to ESPN alone strongly implies losing to a consensus, but beating ESPN alone does not yet imply beating a consensus.

---

## 3. Data sources & access (verified)

### 3.1 ESPN (primary external source — no auth)

Endpoint (verified returning 2024 data):
```
GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2024/segments/0/leaguedefaults/3?view=kona_player_info
Header: X-Fantasy-Filter: {"players":{"limit":N,"sortPercOwned":{"sortPriority":1,"sortAsc":false}}}
```
Per player, the payload provides:
- `player.stats[]` entries — each tagged `seasonId`, `statSourceId` (1=projected, 0=actual), `statSplitTypeId` (0=season). The preseason **projected** season stat line is `statSourceId==1, statSplitTypeId==0`; the **actual** season is `statSourceId==0`. Each entry carries both `appliedTotal` (ESPN's own points) and a `stats` dict of raw projected stats (we use the raw stats, scored through our ruleset).
- `player.draftRanksByRankType` (STANDARD / PPR rank + auction value) and `player.ownership.averageDraftPosition` — preseason ADP/rank for the rank lens.
- `player.id` is the ESPN player id → crosswalks to `GsisId` via our existing `id_map` (`EspnId`).

Pagination via the `limit`/`offset` filter; pull enough to cover all rostered/owned offensive players (a few hundred per position is ample for the top-20 + full-population cohorts).

### 3.2 Sleeper (rank reference only — no auth)

- `GET https://api.sleeper.com/projections/nfl/2024?season_type=regular` returns season-level **ADP fields + projected `gp`**, *not* a projected stat line — so Sleeper cannot serve the preseason stat-RMSE comparison. Used only for the **ADP rank lens** (§2.3).
- Sleeper `player_id` → `GsisId` via `id_map` (`SleeperId`).

### 3.3 Actuals (in-house)

2024 `weekly_stats` is already ingested. Sum each player's weekly actual stat lines to a season total, then score through our ruleset. `GsisId` is native.

### 3.4 Scoring

Reuse `src/projections/scoring/` with the existing **ESPN PPR** ruleset (the coefficients already referenced in the codebase: 1.0/rec, 0.1/rec-yd, 0.04/pass-yd, 6/rec-TD & rush-TD, 4/pass-TD, etc. — exact coefficients taken from the existing `Ruleset`, not re-derived here). The same ruleset scores ESPN projected, our projected, and actual stat lines, so all three columns are commensurable.

### 3.5 ID mapping

Reuse the existing `id_map` ingest + `IdMapSchema` crosswalk (`GsisId` ↔ `EspnId` ↔ `SleeperId`). Players that fail to map are reported in the match-rate stat and excluded from per-source error (a source is not penalized for a row we couldn't join, but the coverage gap is surfaced).

---

## 4. Components (all in `scripts/`)

1. **`scripts/pull_external_projections.py`** — fetch ESPN `kona_player_info` (2024, paginated) + Sleeper season ADP; extract preseason projected stat lines (ESPN), ADP/rank (ESPN + Sleeper), and ESPN actual season totals as a cross-check; normalize to a tidy frame keyed by source-native id; write an intermediate parquet under `data/external_projections/2024/` (ad hoc; not a core store partition). Network-dependent; not unit-tested against live API. Pure parsing helpers (JSON → tidy rows) are unit-tested with captured fixture payloads.
2. **`scripts/benchmark_projections.py`** — join external + our-model + actuals on `GsisId` via `id_map`; score every stat line through the PPR ruleset; compute §2.3 metrics for both cohorts; render `reports/external_projection_benchmark_2024.md`. Pure functions (join, scoring pass-through, RMSE/MAE/Spearman/hit-rate, cohort selection) are unit-tested with synthetic fixtures.
3. **Our-model projections** — produced by the existing `scripts/project_season.py --season 2024` (already preseason-clean); the benchmark script consumes its output. No change to `project_season.py` expected; if a machine-readable (parquet/CSV) output mode is needed instead of stdout rankings, add it as a thin `--out` flag rather than re-implementing the pipeline.

---

## 5. Risks & mitigations

1. **ID-map coverage gaps** (esp. rookies, who may lack `EspnId`/`SleeperId` crosswalk entries or any `weekly_stats` history). *Mitigation:* report match rate per source; treat unmatched as coverage gaps, not silent drops; call out rookie coverage explicitly.
2. **ESPN stat-key decoding** — ESPN's `stats` dict uses numeric stat ids, not names. *Mitigation:* decode the handful of offensive stat ids we need (pass yds/TD/INT, rush att/yds/TD, rec/rec-yds/rec-TD, fumbles) via a small pinned mapping verified against a known player's line; only the stats our PPR ruleset consumes need decoding.
3. **One-season-lucky verdict.** *Mitigation:* report explicitly that 2024 is a single sample; 2023 re-run is the named cheap robustness check if the result is close.
4. **Our model can't project rookies → structural disadvantage on the top-20 cohort.** *Mitigation:* this is a real, verdict-relevant finding, reported as such; also report a "veterans-only" sub-cut so the comparison on players both sides project is visible separately from the rookie penalty.
5. **Spike code rotting into the core.** *Mitigation:* everything stays in `scripts/` + ad-hoc `data/external_projections/`; no `schemas.py` / store changes. Sub-project #2 re-implements cleanly in the core if the verdict says go.

---

## 6. Definition of done

`reports/external_projection_benchmark_2024.md` exists and contains: the per-source × per-position × per-cohort metric table (RMSE / MAE / Spearman / top-20 hit-rate), the ADP rank lens, per-source match-rate/coverage, the veterans-only sub-cut, the honest-reading notes (§2.4), and a one-paragraph **go/no-go recommendation** for sub-project #2. Touched code passes mypy strict + ruff + ruff format; pure transform functions have synthetic-fixture tests.
