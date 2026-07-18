# Waiver-Wire / Undrafted-Pool Assessment — 2026 (16-team half-PPR)

**Issue [#112](https://github.com/alhart2015/FantasyFootball/issues/112) (TODO #44).** After a simulated 16-team draft, what's left on the waiver wire by position? Data-gathering — **no strategy adopt/reject verdict** (standing rule; the single decision is ~Sept 2026).

**Setup.** Hero (`now_or_never_floored`, seat 1) + 15 constrained-ADP bots draft the 2026 consensus pool (`data/vorp_2026/half_16team.parquet`, 578 players), 16 teams × 13 roster spots = **208 taken, ~370 left**. 200 seeds of bot-ADP jitter; per-(position, metric) mean + 95% bootstrap CI. The undrafted pool = pool − every roster. `vorp = season_mean_fpts − replacement_fpts`, so `vorp > 0` ⇔ above the position's startable/replacement line. Core: `src/projections/draft/backtest/waiver_pool.py`; driver: `scripts/waiver_pool_assessment.py`.

## Result — the 16-team wire is barren everywhere except WR

Per position, averaged over 200 drafts (best-available = the top-3 undrafted VORP; `#>repl` = count of undrafted players above replacement; `drain%` = share of the position's above-replacement players that got drafted):

| Position | best-avail VORP (top-1 / 2 / 3) | best-avail proj pts | # above-repl left | drain % | pool above-repl (repl fpts) |
|----------|-------------------------------:|--------------------:|------------------:|--------:|-----------------------------|
| **WR**   | **+15.2** / +6.8 / +2.6        | 100.2               | **4.1**           | 94.3    | 72 of 220 (repl 85.0)       |
| RB       | −1.5 / −2.9 / −11.4            | 71.2                | 0.0               | 100.0   | 51 of 147 (repl 72.7)       |
| TE       | −3.2 / −17.3 / −22.3           | 104.3               | 0.0               | 100.0   | 20 of 131 (repl 107.6)      |
| QB       | −24.1 / −46.7 / −95.6          | 220.3               | 0.0               | 100.0   | 20 of 80 (repl 244.5)       |

Positions ranked by "how good is the best guy you can still grab" (best-available VORP): **WR ≫ RB > TE > QB.**

## Readout — depleted vs streamable

- **WR is the only relatively-streamable position — and even it is thin.** ~4 above-replacement WRs remain, the best worth **+15 VORP**; WR is the deepest position (220 in pool, 72 above replacement) so the field can't quite drain it. This is where waiver value lives in a 16-team league.
- **RB, TE, and QB are drained to zero above-replacement talent in essentially every draft (drain ≈ 100%).** The best player you can still grab at each is **at or below replacement**: RB −1.5 (≈ a replacement-level back), TE −3.2, and QB a steep **−24.1** (the best free-agent QB projects ~24 season points below the startable line — the 16-team QB streaming pool is genuinely poor once ~20 startable QBs are gone).
- **Absolute vs relative depletion agree.** `#>repl` (absolute depth left) and `drain%` (relative to the position's startable supply) tell the same story: QB/RB/TE hit 0 left / 100% drained; WR keeps a sliver.

At N=200 the ordering is unambiguous — the WR-vs-rest gap (best-available +15 vs −1.5/−3.2/−24) dwarfs sampling noise (per-cell 95% CIs computed in `run_assessment`).

## Tie-back to the scarcity thread

The issue asked: *does the field drain TE to nothing (making elite-TE scarcity real), or does it stay streamable?* **Answer: TE drains to nothing (100%) — elite-TE scarcity is real; there is no above-replacement TE on the 16-team wire.** But this is **not TE-specific**: QB and RB drain just as completely. The one exception is WR. So the actionable read (in isolation): in a 16-team league, **you cannot plan to stream QB / RB / TE — the startable supply is entirely consumed in the draft; only WR offers (thin) waiver depth.** Don't punt QB/RB/TE expecting the wire to bail you out; if any position is "safe" to go light on and backfill, it's WR.

This is consistent with the broader investigation's scarcity findings (Tests 7–19): the 16-team format is where positional scarcity bites hardest.

## Caveats

- **Single pool / format** — 2026 consensus, 16-team half-PPR only (per-season historical pools not regenerated; other formats run by re-pointing `--vorp-table`/`--league-config`). Shallower leagues (10–12 team) leave a far deeper wire.
- **Bots are a noisy-ADP human proxy** — the single biggest realism lever (TODO #46; find the migrated issue via `Migrated from TODO #46`); the whole readout rests on how the bot field drains positions.
- **Hero is second-order** — one analytic-strategy seat vs 15 bots barely moves the aggregate wire; the readout is a property of 16-team depth, not the hero.
- **"Replacement" is this pool's per-position line** — "drained" means no *above-replacement* talent left, not literally no rosterable player (a below-replacement QB can still be started; the VORP just quantifies how far below the startable line the wire sits).

**Reproduce:** `python scripts/waiver_pool_assessment.py --vorp-table data/vorp_2026/half_16team.parquet --league-config data/vorp_2026/half_16team.league.json --hero-strategy now_or_never_floored --seeds 200` (PowerShell, `KMP_DUPLICATE_LIB_OK=TRUE`).
