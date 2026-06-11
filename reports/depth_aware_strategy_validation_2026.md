# Depth-Aware Strategy Validation (2026 consensus pool)

**Date:** 2026-06-11
**Spec/plan:** `docs/superpowers/specs|plans/2026-06-11-depth-aware-draft-strategy.*`
**Verdict:** Primary bar **NOT met.** Greedy `SeasonValueStrategy` does **not** beat `now_or_never`
under the season metric. Ship it as a *selectable* strategy; **default stays `now_or_never`.** The
opportunity-cost layer in season-value space (spec §7) is the needed next slice, not an optional refinement.

## What was tested

`SeasonValueStrategy` ranks each available player by the **marginal expected season points** it adds to the
hero's current roster (`V(my_roster + candidate) − V(my_roster)`, CRN), then drafts greedily. The question
(spec §6): does drafting to the season metric beat `now_or_never` (VORP − opportunity-cost) **under that
metric**?

Tournament on the real 2026 consensus pool (`data/consensus_vorp_2026.parquet`,
`configs/league_espn_ppr_12team_skill.json`), **80 seeds, n_sims=200, base_seed=0**, comparing
`season_value` vs `now_or_never` vs `raw_vorp` at slots 1 / 6 / 12, under both the **season** valuer (primary)
and the **starters** valuer (guardrail).

> **Performance note.** The season-value strategy runs a Monte-Carlo at *every* pick, so the spec's §3.7 cost
> estimate ("minutes range") was wrong — it accounted for the season *valuer* but not the per-pick *strategy*
> MC (≈9 hrs/slot at 200 seeds × 300 sims). This validation was unblocked by a numpy fast-path
> (`_vectorized_lineup_points`) that vectorizes the weekly fill across all draws: **3 seeds × 50 sims × 1 slot
> dropped 527s → 4.2s (~125×), byte-identical means**, pinned by an equivalence test vs `optimal_lineup_points`
> over 200 random masks (exhaustively re-verified in review over all 1,024 masks of a 10-player roster). This
> promotes the spec §7 "numpy fast-path" deferral to shipped.

## Results — season valuer (primary bar)

| Slot | now_or_never | season_value | raw_vorp | top-two paired diff | winner |
|------|-------------:|-------------:|---------:|--------------------:|--------|
| 1    | **1854.1**   | 1847.8       | 1611.9   | +6.3 `[-1.0, +13.7]` | none (CI brackets 0) |
| 6    | **1886.9**   | 1864.1       | 1580.2   | +22.8 `[+13.0, +32.2]` | **now_or_never** |
| 12   | 1866.7       | **1875.8**   | 1638.6   | +9.1 `[+1.7, +16.5]` | **season_value** |

`season_value` wins **only at slot 12** (the turn). At slot 6 `now_or_never` wins outright; at slot 1 they
tie. The bar required `season_value` to win at **all three** — **not met.**

## Results — starters valuer (guardrail)

| Slot | now_or_never | raw_vorp | season_value | winner |
|------|-------------:|---------:|-------------:|--------|
| 1    | **2017.9**   | 1941.0   | 1904.3       | now_or_never (+76.9 `[+72.9, +81.1]`) |
| 6    | **2030.0**   | 1956.5   | 1921.6       | now_or_never (+73.5 `[+67.7, +78.9]`) |
| 12   | **1987.6**   | 1977.2   | 1930.9       | now_or_never (+10.4 `[+6.0, +15.6]`) |

Under the starters metric `season_value` is the **worst** of the three at slots 1 and 6 — behind even the
position-blind `raw_vorp`. The guardrail expected *some* starters loss (it optimizes a different metric); this
is a *large* loss, consistent with the season-metric result below.

## Determinism

`--adp-jitter 0 --seeds 1` → point CIs (`season_value` 1852.22 = lo = hi). Same seed ⇒ identical roster. ✓

## Interpretation — why greedy depth loses

The result is not noise (paired diffs are tight; slot-6 `now_or_never` +22.8 `[+13, +32]` is conclusive) and
not a bug (the fast-path is exhaustively verified; the strategy passed its discriminating unit test). It is a
**structural property of greedy marginal-value drafting**:

1. **Greedy marginal value is myopic.** `SeasonValueStrategy` maximizes the *current* pick's marginal season
   points given the *current* roster, with **no pick-timing awareness**. `now_or_never` explicitly subtracts
   the opportunity cost of waiting (the expected best survivor at the position by the next pick), so it grabs
   scarce high-value players before they vanish. Greedy season-value lets them go, over-weighting immediate
   depth/insurance — and ends up with a worse roster **even by the season metric it optimizes.**
2. **The season metric fixed a `raw_vorp` pathology, not a `now_or_never` one.** The 10-QB problem that
   motivated the metric (PR #60) was `raw_vorp`'s positional *blindness*. `now_or_never` was already balanced
   (VORP encodes static positional scarcity; its opportunity-cost layer adds dynamic scarcity). Greedily
   optimizing the metric *without* a scarcity/timing layer reintroduces a different inefficiency.
3. **The edge scales with the turn.** `season_value` only wins at slot 12 (back-to-back picks, where
   pick-timing matters least — the wing's "who survives to my next pick" pressure is near zero). At slot 6
   (a ~22-pick wait) `now_or_never`'s timing layer is most valuable, and it wins by the most. The slot pattern
   is itself evidence that **pick-timing, not depth, is the dominant signal** in this pool.

Higher strategy `n_sims` would sharpen the marginal estimates but cannot supply the missing opportunity-cost
term — the gap is structural, not statistical.

## Decision

- **Do not flip the default.** `now_or_never` remains the default in both CLIs.
- **Ship `season_value` as a selectable strategy** (`--strategy season_value` live;
  `compare --with-season-value` tournament). It is correct, tested, and useful as the A/B baseline for the
  next slice.
- **Next slice (now empirically justified, not optional): the opportunity-cost layer in season-value space**
  (spec §7): `score = marginal − E[marginal of the best survivor at that position by my next pick]`. The
  validation shows the timing layer is *load-bearing* — greedy depth alone is a regression. With the fast-path
  shipped, this layer is now tractable to build and A/B behind the same `DraftStrategy` protocol.

## Reproduce

```
python scripts/draft_tournament.py --vorp-table data/consensus_vorp_2026.parquet \
  --league-config configs/league_espn_ppr_12team_skill.json --my-slot {1,6,12} \
  --seeds 80 --seed 0 --valuer {season,starters} --season 2026 --n-sims 200 --data-root data \
  compare --with-season-value
```

Note: the 2026 `schedules` partition is not ingested, so `--season 2026` runs with **no byes** (a logged
warning); the injury-availability model still applies. Byes would only narrow the season/starters gap
slightly and do not change the verdict (the loss is driven by pick selection, not bye coverage).
