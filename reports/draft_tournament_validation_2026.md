# Draft Assistant — Strategy Tournament Validation (2026 consensus)

First end-to-end run of the Slice 2 strategy-comparison harness against **real 2026 consensus data**
(not synthetic fixtures). The question Slice 1 could pose but not answer: does the analytic
`NowOrNeverStrategy` actually beat plain best-available, and by how much? **Yes — from every draft
slot, with tight confidence intervals.**

## Setup

- **Pool:** consensus VORP table generated from the local consensus snapshot
  (`data/processed/consensus_projections/season=2026/asof=2026-06-09/`) →
  `python scripts/generate_vorp_table.py --season 2026 --source consensus --league-config configs/league_espn_ppr_12team_skill.json --out <path>` → **458 players**.
- **League:** `configs/league_espn_ppr_12team_skill.json` — 12 teams, 1 QB / 2 RB / 3 WR / 1 TE / 1 FLEX / 9 BENCH, ESPN PPR (skill positions only; K/DST folded into BENCH).
- **Field:** `adp_jitter = 8.0` (the default `⅔·n_teams`), `base_seed = 0`.
- **Metric:** optimal starting-lineup projected points of the hero's final roster, scored by `optimal_lineup_points`. Winner declared only when the top-two **paired** per-seed difference CI excludes 0.

## What the two strategies do

**`RawVorpStrategy` (the control)** — score = **VORP**. It takes the best available player by value-over-replacement. VORP already encodes *static* positional scarcity (a top TE is worth more over its replacement than a mid WR over its deeper replacement pool), but it is blind to *when your next pick comes* and *who will still be on the board then*.

**`NowOrNeverStrategy`** — score = **VORP − E[best survivor at that position by my next pick]**. For each position it computes the expected VORP of the best player at that position who will *still be available* at the hero's next pick (ADP fed through a logistic survival model), then subtracts that from each candidate's VORP. The result is **opportunity cost**: how much value you forfeit by waiting one round at this position.

- It **reorders across positions**: it attacks the position where waiting is most expensive. If an elite RB will be gone by your next pick but a comparably-valued WR will still be there, it takes the RB *now* even when the WR's raw VORP is marginally higher — because the WR is replaceable next round and the RB is not.
- **Within a position the order is unchanged** (the offset is a per-position constant), so it never reaches past a clearly better player at the same position; its job is choosing *which position to spend this pick on*.
- At the hero's **last pick it falls back to raw VORP** (nothing survives to a non-existent next pick, so there is no timing signal).

In one line: VORP says *"take the most valuable player."* Now-or-never says *"take the value you can't get back by waiting."* It is the **dynamic-scarcity layer** on top of VORP's static scarcity.

## Results

### `compare` — now_or_never vs raw_vorp, by draft slot

| Slot | seeds | now_or_never (95% CI) | raw_vorp (95% CI) | paired diff (95% CI) | winner |
|------|-------|------------------------|-------------------|----------------------|--------|
| 1 (wing, picks first) | 120 | 2019.4 [2016.1, 2022.9] | 1942.9 [1940.8, 1945.1] | **+76.5 [+73.0, +79.9]** | now_or_never |
| 6 (mid) | 150 | 2031.0 [2027.6, 2034.5] | 1956.4 [1954.2, 1958.7] | **+74.6 [+70.7, +78.7]** | now_or_never |
| 12 (the turn) | 120 | 1985.6 [1979.1, 1993.1] | 1976.6 [1971.8, 1981.7] | **+9.0 [+4.7, +13.5]** | now_or_never |

now_or_never wins at **every** slot (all CIs strictly above 0). The **size** of the edge scales with how much pick-timing matters at your seat:

- **Slot 1 (+76):** you wait ~22 picks until your next turn, so survival is highly uncertain and the now-or-never signal is loud.
- **Slot 12 (+9):** you pick 12th and 13th back-to-back at the turn, so almost nobody disappears before your next pick — the timing signal nearly vanishes and the two strategies converge.

That gradient is exactly the dynamic-scarcity effect the engine was built to capture and that static VORP structurally cannot see.

### Determinism check

`--adp-jitter 0 --seeds 1` → a single deterministic draft; every CI bound collapses to the point estimate (now_or_never 2020.77, raw_vorp 1945.32, diff +75.45). Same seed ⇒ identical result, as designed.

### `tune-sigma` — survival spread σ

100 seeds, slot 6, grid `6,8,11,14,18,24`:

| σ | mean | | σ | mean |
|---|------|-|---|------|
| 6 | 2030.3 | | 14 | 2029.7 |
| 8 | 2030.8 | | 18 | 2025.9 |
| **11** | **2031.2** | | 24 | 2023.3 |

The curve is **unimodal** with a peak at **σ ≈ 11**. The optimum is only **+0.4 pts** over the default `σ = ⅔·n_teams = 8`, so the default is effectively optimal and is left unchanged. (σ governs only the *survival* spread; the cross-position reordering is robust to it across the 8–14 range.)

## Takeaways

1. **The harness works on real data** — bootstrap CIs are tight, the σ curve is well-behaved, results are reproducible by seed. The math validated on synthetic fixtures holds on the live 458-player 2026 pool.
2. **`NowOrNeverStrategy` is empirically justified** — it beats best-available from every seat, decisively where pick-timing matters.
3. **No default change needed** — `σ = ⅔·n_teams` (≈8 for a 12-team league) is within 0.4 pts of the empirical optimum (≈11).
4. **Next refinement, now measurable:** the survival model is *unconditional* (it ignores that an available player has already lasted to now). A conditional survival model is the natural Slice-2+ improvement and can be A/B'd against the current one through this same harness.

*Reproduce:* the two-step recipe in the table setup above; each `compare`/`tune-sigma` run is ~1.5 min on a 32-core box. The generated VORP parquet is gitignored/regenerable.
