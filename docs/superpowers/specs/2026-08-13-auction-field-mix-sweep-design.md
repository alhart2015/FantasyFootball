# Auction field-mix sweep — `n_patient` design note

**Status:** written **retroactively**, 2026-08-13, after the code and Run Z had already landed on
`exp/auction-field-mix-sweep`. **Branch:** `exp/auction-field-mix-sweep`.

## Why this document says "retroactively" at the top

`CLAUDE.md` mandates spec → plan → execute on a feature branch. This branch did not follow that: a
36-line change to one script was written, run, and reported before any design doc existed. A
`/loop-review` intent pass flagged the deviation and it is recorded here rather than quietly
dropped, because the rule's purpose — a future session being able to find *why* the seam is shaped
the way it is — is not served by pretending the order was different. **This is the record, not a
reconstruction of a plan that was never written.**

## Problem

Every auction run from Run A through Run Y modelled the opponent room as **9 aggressive / 2
conservative** of 11 bots. That ratio was a hard-coded module constant (`_PATIENT_EVERY = 5`),
never a measurement and never swept. So the standing question "how wrong can we be about the room
before the hero recommendation changes?" could not be asked at all.

## The seam

`build_field` gains `n_patient: int | None = None`.

- **`None`** — the historical rule (every `_PATIENT_EVERY`-th seat is a `PatientValueBot`). Every
  prior run **re-simulates identically**. Note this is *not* byte-for-byte at the artifact level:
  `_run_chunk` now emits an extra `"n_patient": null` key, so chunk files differ from pre-existing
  ones. Aggregation is unaffected — `_guard_homogeneous` reads `c.get("n_patient")`, so old chunks
  and new default chunks both yield `None` and pool correctly.
- **An int** — exactly that many hoarder seats, placed at `{(2i+1)·n // 2k for i in range(k)}`.

### Why that placement formula

Evenly spaced rather than the first or last `k`, because clustering the hoarders at one end changes
*where* they sit as well as how many there are, confounding the mix with seat adjacency to the hero.

The half-step offset keeps the set off seat 0 at low densities. **It does not once `2k > n`** — at
those densities nearly every seat is a hoarder and there is no spread left to give; the swept 5/6,
3/8 and 0/11 cells all include seat 0. The code comment says so rather than claiming an invariant
it does not hold.

The arithmetic matters: consecutive exact values differ by `n/k ≥ 1`, so the floors are strictly
increasing and the set has exactly `k` members. A plain `round((i + 0.5)·n / k)` does **not** — it
collides once `k` approaches `n` and returned 6 hoarders for a request of 11. That was caught by
testing before the sweep ran, and the parametrized test now checks every `(n, k)` pair for
`n` in 1..15 rather than spot values.

### The refusal rule (the load-bearing part)

`build_field` has five exit paths and only one reads `n_patient`. Meanwhile `_run_chunk` writes the
value into the chunk payload unconditionally and `_guard_homogeneous` treats it as a config key.
Accepting the knob on a path that ignores it therefore produces **an artifact labelled with a room
that was never simulated** — the same class of failure the script already fails loudly on for ESPN
prices.

So a non-`None` `n_patient` that cannot be honored **raises**
(`_reject_unhonorable_n_patient`): negative, above `n_bots`, a field with a fixed archetype mix
(`realistic`, `overbidder_unpaced`, `balanced_field`), the uniform-cap short-circuit
(`n_bots is None or pace_jitter <= 0`), or a non-zero request against `overbidder_only`, which has
no conservative seats by definition.

The aggregate provenance header prints `n_patient` for the same reason: without it, the `p2` and
`p8` chunk directories produce byte-identical header lines and an operator transcribing seven cells
has no discriminator.

### Reach

The flag is threaded to `auction_field_bakeoff` and to the five sibling diagnostics that share
`build_field` (`auction_draft_trace`, `auction_field_outcomes`, `auction_record_spread`,
`auction_sample_rosters`, `auction_spend_profile`). Without that, a diagnostic run to explain a
Run-Z cell would silently describe the historical 9/2 room — which Run Z itself measures as worth
~0.004 `reg_win_pct` away from the swept 9/2 cell.

## What this deliberately does not do

- **Sweep *how* aggressive the bots are.** `overbid`, `pace`, `pace_jitter` and the archetype itself
  are held fixed; only the count varies. That axis remains unswept.
- **Separate placement from pace assignment.** `_spread_paces` hands its jittered caps to
  non-hoarder seats in seat order, so moving a hoarder also reshuffles which aggressive seat draws
  which cap. The two effects are confounded by construction and Run Z does not separate them.
- **Cover the grid evenly.** Swept values are 0, 2, 3, 5, 6, 8, 11 — steps of 1 to 3 seats. `4/7`
  was never run and sits inside the crossover bracket Run Z reports.

## Follow-ups, in rough priority

1. **Fill `n_patient=7` (4/7)** — the one missing cell inside the reported crossover bracket.
2. **Re-run 9/2 with the historical placement** under the new flag path to isolate placement from
   pace reassignment.
3. **Sweep `overbid`/`pace`** to vary how aggressive rather than how many.
