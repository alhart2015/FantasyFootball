# External Projection Benchmark Spike — Verdict (2024)

**Date:** 2026-06-08
**Branch:** `feat/external-projection-benchmark`
**Spec:** `docs/superpowers/specs/2026-06-08-external-projection-benchmark-design.md`
**Plan:** `docs/superpowers/plans/2026-06-08-external-projection-benchmark.md`
**Status:** spike concluded — **the planned preseason RMSE benchmark was NOT run; it is invalid on our side (see §1). The verdict stands on architecture, not on a contaminated metric.**

---

## TL;DR

Our projection model **cannot produce a preseason / draft projection** — it is a weekly, in-season model whose features read trailing windows that include the current season. `scripts/project_season.py` aggregates in-season-informed weekly predictions; its season totals secretly use the very 2024 outcomes we wanted to predict. The planned "preseason ESPN vs preseason ours" comparison is therefore **not apples-to-apples** and would make our model look dramatically (and falsely) better. We did not publish those numbers.

**For drafting (the priority use case), this settles the question: pivot to external sources.** Our model is not in the running — not because it loses, but because it structurally cannot play. ESPN and Sleeper provide genuine preseason draft projections; our stack does not.

---

## 1. The disqualifying finding: our "projection" peeks at the season

`project_season.py` trains on 2018–2023 and then, for each week of 2024, builds features from trailing windows over `concat(2023, 2024)` data `as_of_week=W` and predicts week `W`. Two consequences make the aggregated season total in-season-informed, not preseason:

1. **Trailing features include current-season games.** Week-`W` features (e.g. `targets_per_game_l4`) use 2024 weeks `1..W-1` — real, observed outcomes.
2. **Only active players get projected.** A player who is injured/inactive in week `W` has no feature row that week, so he is silently dropped from that week's projection. The season total therefore reflects who actually played.

### Evidence (2024, PPR fantasy points)

| Player | Our model | n_weeks ours | ESPN **preseason** | Actual |
|---|---:|---:|---:|---:|
| **Christian McCaffrey** | **63.4** | **4** | 335.5 | 47.8 |
| Puka Nacua | 203.0 | 14 | 260.5 | 206.6 |
| Lamar Jackson | 419.0 | 19 | 303.8 | 430.4 |
| Ja'Marr Chase | 249.3 | 17 | 286.3 | 403.0 |

McCaffrey is the smoking gun: he tore his Achilles/PCL early and played ~4 games. ESPN's **preseason** projection (335.5) correctly reflects a healthy-season forecast — they could not have known. Our model projected him for **4 weeks only** and landed at 63.4, near his injured-season actual (47.8). The only way to "know" that in advance is to use in-season data — which our model does. Lamar at 419-vs-430 is the same effect from the other side: tracking an MVP season, not forecasting it.

A preseason projection that nails injuries and MVP seasons isn't a good projection — it's a model reading the answer key. Scoring it against ESPN's honest preseason forecast would be meaningless.

## 2. What this means for the strategy

The original question driving this spike: *is our home-grown model worth the continued effort, or should we use freely available projections and spend effort on tools?*

- **Draft / season-long (priority #1):** **Use external sources.** Our model offers no draft projection at all — it requires in-season actuals to function. ESPN (preseason stat lines + ADP + draft ranks) and Sleeper (ADP) are genuine, free, preseason sources, verified pullable for historical seasons (see §3). This is not a close benchmark call; there is nothing on our side to benchmark.
- **Weekly start/sit (use-case #2):** **Open question, still fairly answerable.** Here our model being in-season is legitimate. A *fair* test would compare our weekly projections against ESPN's *weekly* projections vs weekly actuals (both in-season-informed, same grain). Not run in this spike; tracked as a follow-up.
- **DFS (use-case #3):** downstream of the same projection input; inherits the draft conclusion plus correlation needs (separate effort).

## 3. What this spike *did* produce (reusable)

- **A working free-source preseason puller** — `scripts/pull_external_projections.py` fetches ESPN preseason projected stat lines + ADP + draft ranks (no auth) and Sleeper ADP, normalized and keyed for crosswalk to `GsisId`. The ESPN stat-id decode was verified end-to-end on real 2024 data (Ja'Marr Chase: 105 rec / 1335 yds / 8 TD). This is the seed of the external-consensus ingest layer (sub-project #2).
- **A benchmark scaffold** — `scripts/benchmark_projections.py` (join + PPR scoring + RMSE/MAE/Spearman/cohorts/report). Correct machinery, but **must not be run against `project_season.py` output and called a preseason verdict** — its module docstring now carries that warning. It is directly reusable for the weekly start/sit benchmark.
- **Verified data feasibility** — ESPN's `statSourceId=1, statSplitTypeId=0` season projection is genuine preseason (rookie/breakout/injury misses, not contaminated), and historical projections are retrievable from both ESPN and Sleeper.

## 4. Recommended next steps

1. **Build sub-project #2 (external consensus projection layer) for draft**, using the working ESPN + Sleeper puller as the foundation; add 1–2 scraped sources (FantasyPros/CBS preseason) for a real consensus average; this becomes the projection basis the downstream tools consume.
2. **Then build the Draft Hub** on top of that consensus (the actual goal: spend effort on *how we use* projections).
3. **Optionally**, run the fair weekly start/sit benchmark (our weekly model vs ESPN weekly vs actuals) to decide whether to keep our model for in-season start/sit, or retire it entirely.

## 5. Honest caveats

- This conclusion is about *capability*, not accuracy: we did not measure whether our weekly model is accurate, only that it cannot do preseason. The weekly benchmark (§4.3) is what would judge its in-season value.
- One season (2024) of ESPN preseason data was inspected; the qualitative preseason-vs-contaminated check is robust (injuries/rookies), independent of single-season luck.
