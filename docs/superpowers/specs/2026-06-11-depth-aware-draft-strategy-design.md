# Depth-Aware Draft Strategy (`SeasonValueStrategy`)

## 1. Purpose

PR #60 gave the draft tournament a risk-aware **season metric** — `expected_season_points`, the points a
roster actually scores across a season under per-player availability (injuries + byes), filling the best
legal lineup each week. Validation proved it rewards the right thing: under the season metric the
position-blind `raw_vorp` roster (with its dead-weight 10-QB bench) drops ~18% while the balanced
`now_or_never` roster drops ~5%.

But **no strategy actually drafts to that metric.** `now_or_never` scores each candidate by
`vorp − E[best survivor at position by my next pick]` — static positional scarcity plus a pick-timing
(opportunity-cost) layer. That is blind to bench depth: once a position's starting slots are notionally
covered, `now_or_never` sees no reason to add insurance, even though the season metric says a second RB is
worth real points because RB1 misses time. The last two slices built and validated a depth-aware *yardstick*;
this slice builds the *player* that drafts to it.

This slice adds **`SeasonValueStrategy`**: a `DraftStrategy` that ranks each available candidate by the
**marginal expected season points** it adds to the hero's current roster —
`V(my_roster + candidate) − V(my_roster)`, where `V` is the PR #60 season metric. It is the first strategy
that values positional insurance, and it optimizes *directly* against the metric the tournament scores by, so
"does drafting to the metric beat `now_or_never`?" becomes an empirical, paired-bootstrap question.

**This slice builds the greedy marginal-value strategy.** A pick-timing/opportunity-cost layer in
season-value space (the `now_or_never` analog) is a named follow-up, not this slice (§7).

## 2. Scope

### In scope
- **`SeasonValueStrategy`** (in `strategy.py`) — a `DraftStrategy` constructed with the MC config it needs
  (`PlayerAvailability`, `n_sims`, `base_seed`, `top_k`). Scores candidates by marginal expected season
  points; ranks purely by that score.
- **A CRN (common-random-numbers) season-value evaluation** so the marginal `V(R+c) − V(R)` is low-variance:
  base and every candidate share one pre-drawn per-player availability matrix within a `recommend()` call.
- **Candidate pruning** — score only the top-`top_k`-by-VORP candidates per eligible position with the MC;
  the deep-bench long-shots (marginal ≈ 0) get a deterministic fallback rank.
- **A finalize path that ranks purely by marginal score** — without `_finalize`'s `fills_starting_slot`
  hard tier (the season metric already encodes starting-slot value; the tier would distort it).
- **CLI wiring** — `season_value` selectable in `tournament_cli.py` (reusing `_build_season_valuer`'s
  partition loading for availability) and in the live assistant `cli.py` (`scripts/draft_assistant.py`).
- **Validation** — a `reports/` writeup: tournament under the **season** valuer comparing `season_value`
  vs `now_or_never` vs `raw_vorp` at slots 1/6/12 on the real 2026 consensus pool.

### Explicitly out of scope (later slices)
- **Opportunity-cost / pick-timing layer in season-value space** — `now_or_never`'s dynamic-scarcity idea
  applied to marginal season value (`marginal − E[marginal of best survivor at that position by next pick]`).
  The natural follow-up; deferred so the greedy depth signal is isolable and attributable (§7).
- **Analytic / closed-form depth surrogate** — rejected (§5.4): a second implementation of the metric that
  can drift from `SeasonValuer`.
- **A numpy fast-path for the weekly fill** — a deferred optimization (§3.6), inherited from PR #60's §3.4.
- **Changes to the season metric, availability model, or survival model** — all reused unchanged.

## 3. Design

### 3.1 Inputs and where the data comes from

`SeasonValueStrategy.recommend(state, pool, config)` matches the `DraftStrategy` protocol signature
(unchanged), and gets the MC config from construction (the substitution-seam pattern `NowOrNeverStrategy`
already uses to hold a `SurvivalModel`):

- **`state: DraftState`** — supplies `my_roster` (gsis_ids the hero already owns) and `drafted_ids`.
- **`pool: pd.DataFrame`** — the consensus VORP frame (`VorpTableSchema`): `gsis_id`, `position`, `vorp`,
  `season_mean_fpts`, `consensus_adp`. `season_mean_fpts` is the per-player points the season metric reads
  (confirmed: `optimal_lineup_points` already reads exactly this column).
- **`config: LeagueConfig`** — `roster_slots` shape.
- **Bound at construction:** `availability: PlayerAvailability` (built once via PR #60's `build_availability`
  from `weekly_stats` / `schedules` / `id_map`), `n_sims: int`, `base_seed: int`, `top_k: int = 8`.

No new data source; the strategy reuses the exact tables and availability model PR #60 established.

### 3.2 The strategy (`strategy.py`)

```python
@dataclass(frozen=True)
class SeasonValueStrategy:
    availability: PlayerAvailability
    n_sims: int
    base_seed: int
    top_k: int = 8

    def recommend(self, state, pool, config) -> pd.DataFrame: ...
```

`recommend`:
1. **Eligible subset** — `_eligible_subset(state, pool, config)` (reuse as-is): drops drafted +
   roster-ineligible rows, guarantees `consensus_adp`.
2. **My roster rows** — select the pool rows for `state.my_roster` (the hero's current players). These are
   the incumbents every marginal is measured against. (Players already on the roster are also already in
   `drafted_ids`, so they are not in the eligible candidate subset — no overlap.)
3. **Prune** — within each eligible position, take the top-`top_k` candidates by `vorp`. A candidate ranked
   below `top_k` at its position is strictly dominated as a *starter* by `top_k` better players there and so
   contributes ≈ 0 marginal season points; spending an MC on it is wasted (§5.3). Pruned candidates are kept
   in the output frame but assigned a deterministic fallback (§3.5).
4. **Score evaluated candidates** — for each pruned-in candidate `c`:
   `score(c) = V_crn(my_roster + c) − V_crn(my_roster)` under the shared CRN draw (§3.3), where
   `V_crn` is the season metric. `V_crn(my_roster)` is computed **once** per `recommend()` and reused.
5. **Finalize** — rank purely by `score` (§3.4); validate `RecommendationSchema`.

### 3.3 CRN-aware season-value evaluation (the one genuinely new mechanism)

The marginal is a difference of two Monte-Carlo estimates. Under `SeasonValuer`'s per-roster sha256 seed,
`V(R)` and `V(R+c)` draw *independent* randomness, so `V(R+c) − V(R)` is dominated by MC noise of order
`σ/√n_sims` on each term while the depth signal it is trying to measure is small — the signal drowns unless
`n_sims` is impractically large.

**Fix: common random numbers (CRN).** Within one `recommend()` call, all evaluations — the base and every
candidate roster — share **one pre-drawn per-player availability realization**, so adding `c` perturbs only
`c`'s own draws and leaves every incumbent's draws identical. The paired difference then isolates `c`'s
contribution at far lower variance (the noise common to both terms cancels).

Concretely, the CRN evaluation:
- Indexes the random draw **by `gsis_id`**, not by roster position, so a player's availability realization is
  the same whether or not `c` is on the roster and regardless of roster size. (PR #60's
  `expected_season_points` draws `rng.random(n)` sized to the roster, so the stream re-aligns differently for
  `n` vs `n+1` players — that is the misalignment CRN must avoid.)
- Pre-draws an `(n_sims × n_players_considered)` uniform matrix keyed on a stable sorted-gsis ordering over
  the union of `my_roster` + all evaluated candidates, once per `recommend()`.
- For each evaluated roster, runs the existing **greedy `optimal_lineup_points` fill** and PR #60's
  **clean-week / bye-week factorization** verbatim — only the *source of the availability draw* changes (from
  a fresh `rng.random` per sim to a slice of the shared matrix). The fill logic and the factorization are
  reused, not reimplemented.
- Uses a fixed seed derived deterministically from `(base_seed, state.current_pick)` so the whole call is
  reproducible and stable across runs.

This lives as a CRN variant alongside `expected_season_points` (e.g. an internal
`_expected_season_points_crn(roster, slots, availability, draw_matrix, gsis_index)` factored so the season
metric and the strategy share the week-expectation kernel). The exact factoring (new private helper vs. a
`draws=` parameter on the existing function) is a plan-level call; the **contract** is: same availability
realization across base + candidates, fill + factorization reused.

### 3.4 Ordering: rank purely by marginal score

The shared `_finalize` sorts by `[fills_starting_slot, score, vorp, gsis_id]` — a **hard** primary tier that
floats any player filling an open starting slot above every bench-depth add. That is right for `now_or_never`
(whose `score` does not by itself know about open slots) but **wrong here**: the season metric already values
an open starting slot (a player who fills an empty starting slot has large marginal season points), so the
hard tier would override the very signal this strategy computes. `SeasonValueStrategy` therefore ranks by
`[score, vorp, gsis_id]` with **no `fills_starting_slot` primary tier**.

Implementation: make the starting-need tier **optional** in `_finalize` (e.g. a
`starting_need_tier: bool = True` parameter; existing callers unchanged), or factor a shared finalize core
that both call with different sort keys. `fills_starting_slot` is still **computed and emitted** in the output
frame (it is a `RecommendationSchema` column and useful display context) — it is just not a sort key for this
strategy. The plan picks the cleaner factoring; the contract is: `now_or_never` / `raw_vorp` ordering is
byte-identical to today, `season_value` orders by marginal score.

### 3.5 Pruned-candidate fallback (output completeness)

`recommend` must return a `RecommendationSchema` frame over the **whole** eligible subset, but only the
top-`top_k`-per-position candidates get an MC score. Pruned-out candidates get a deterministic fallback so the
frame is complete and stable:
- They are ranked **below every MC-evaluated candidate**, ordered among themselves by `vorp` (desc), then
  `gsis_id`. Concretely, evaluated candidates carry their real marginal `score`; pruned candidates are sorted
  after them as a block.
- This tail is **cosmetic**: the strategy only ever acts on the argmax (the #1 recommendation), which is
  always an evaluated candidate (the best add at any position is in that position's top-`top_k` by VORP). The
  tail ordering matters only for the live-assistant display, where a VORP-ordered remainder is the sensible
  default.

### 3.6 What changes and what does not

- **Unchanged:** the draft simulation, `now_or_never` / `raw_vorp` strategies and their exact output,
  `optimal_lineup_points`, `expected_season_points`'s public contract, the availability/survival models, the
  bootstrap/winner machinery, the `RosterValuer` seam.
- **New:** `SeasonValueStrategy` (+ the CRN season-value evaluation it calls).
- **Touched:** `strategy.py` (new strategy + optional finalize tier), `tournament_cli.py` (register
  `season_value`, build its availability), the live assistant `cli.py` / `scripts/draft_assistant.py`
  (`--strategy season_value`), and the season-value module (factor the CRN kernel).

### 3.7 Cost

Per hero pick: `(eligible positions × top_k) + 1` CRN season-value evaluations, each
`≈ n_sims + (distinct roster bye weeks) × n_sims` greedy fills (PR #60's factorization). In the tournament
the strategy runs once per hero pick (`≈ rounds` picks) per simulated draft per seed. With `top_k = 8`,
~4–5 eligible positions, and `n_sims ≈ 200–300`, validation `compare` runs are in the minutes range — heavier
than `now_or_never` but acceptable for an offline validation. A live single-draft `recommend()` is trivially
fast (one pick, a few hundred candidate-evals). A numpy fast-path for the weekly fill (PR #60 §3.4 deferral)
is the lever if season-valuer tournament sweeps are ever wanted; not a v1 requirement.

## 4. Testing

Synthetic fixtures (project norm — no network in unit tests):
- **Discriminating depth test (miniature 10-QB)** — a constructed pool + roster state where a depth pick
  (a second RB on a roster whose RB1 has high injury `p`-risk) yields strictly higher marginal season points
  than a redundant high-VORP starter at an already-deep position. Assert `SeasonValueStrategy` recommends the
  depth player while `NowOrNeverStrategy` (same state/pool) recommends the redundant starter. This is the
  core behavior, in miniature — the discriminating test that proves the strategy does something `now_or_never`
  cannot.
- **CRN low-variance** — the marginal of adding a clearly-useful player is positive and **stable across
  `n_sims`** (e.g. `n_sims=50` vs `200` agree to a tight tolerance), where an independent-seed difference of
  the same two MC estimates would not. Guards that CRN is actually in effect.
- **Hand-computed marginal** — on a tiny roster (1 starting slot, a starter with `p<1`, no bye, `weeks`
  short), the marginal of adding a backup matches the closed-form 2-player fill-in contribution to MC
  tolerance.
- **Determinism** — same `state` + `base_seed` ⇒ byte-identical recommendation; the returned frame validates
  `RecommendationSchema`.
- **Pruning invariance** — with `top_k` ≥ pool depth at every position, the result equals the no-pruning
  result on a small pool (pruning never changes the argmax or the evaluated ordering).
- **Ordering** — `SeasonValueStrategy` does **not** apply the `fills_starting_slot` hard tier (a high-marginal
  bench add can outrank an open-slot filler when the metric says so); `now_or_never` / `raw_vorp` ordering is
  **byte-identical to today** (regression guard on the optional-tier refactor).
- **Last-pick / empty-eligible edges** — a state with one eligible position, and a near-full roster, both
  return a valid frame.
- **CLI** — `--strategy season_value` runs end-to-end on a fixture in both the tournament and live-assistant
  CLIs.

Gates per the project bar: `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check`.

## 5. Key decisions

- **5.1 Greedy marginal value this slice; opportunity-cost layer next** — drafting to maximize marginal
  expected season points is the clean, isolable depth signal, and the metric exists to measure it directly.
  Folding in pick-timing (the `now_or_never` analog in season-value space) at the same time would make it
  impossible to attribute which half is winning. Build greedy first, measure, then add the timing layer
  behind the same protocol (§7).
- **5.2 CRN is load-bearing, not a nicety** — the marginal is a small difference of two noisy MC estimates;
  without common random numbers it is unmeasurable at any practical `n_sims`. Sharing one per-player
  availability realization across base + candidates is what makes the whole approach work.
- **5.3 Exact MC + candidate pruning** — reuse the tested season metric verbatim rather than approximate it;
  prune to the top-`top_k`-by-VORP per position because deeper candidates are dominated starters with ≈ 0
  marginal value. Pruning trades a negligible tail for a large constant-factor speedup and never moves the
  argmax (§3.5).
- **5.4 No analytic surrogate** — a closed-form order-statistic depth value would be faster but is a *second*
  implementation of the metric that can silently drift from `SeasonValuer`, and would need its own correctness
  tests. Reuse beats reimplementation; correctness over speed.
- **5.5 Rank by marginal score, drop the `fills_starting_slot` hard tier** — the season metric already values
  open starting slots, so the hard tier would override the strategy's own signal. The tier stays as an emitted
  column and an opt-in for the other strategies; it is just not a sort key here.
- **5.6 Strategy holds the MC config, protocol unchanged** — `recommend(state, pool, config)` is untouched;
  availability / `n_sims` / `base_seed` / `top_k` are constructor args, mirroring
  `NowOrNeverStrategy(survival=...)`. The tournament and live CLIs build the strategy the same way they build
  the `SeasonValuer`, reusing `_build_season_valuer`'s loader.

## 6. Success criteria

Validation deliverable (`reports/`): tournament under the **season** valuer on the real 2026 consensus pool
(`configs/league_espn_ppr_12team_skill.json`), comparing `season_value` vs `now_or_never` vs `raw_vorp` at
slots 1 / 6 / 12.

- **Primary (ship-the-default bar):** under the season valuer, `season_value` beats `now_or_never` with the
  paired-diff CI excluding 0 at all three slots.
- **Guardrail:** report the **starters**-metric numbers too. `season_value` is expected to give up *some*
  single-week starter ceiling (it optimizes a different, better metric); a modest starters-metric loss is
  acceptable and not disqualifying. A *large* starters regression is a flag to investigate.
- **Determinism:** `--adp-jitter 0 --seeds 1` ⇒ point CIs (same seed ⇒ identical roster).

If the primary bar is met, flip the live-assistant default to `season_value`; if not, ship it as a selectable
strategy and write up the slot-by-slot behavior for a follow-up decision.

## 7. Open questions / future slices

- **Opportunity-cost layer in season-value space** — the `now_or_never` analog on the depth metric:
  `score = marginal − E[marginal of the best survivor at that position by my next pick]`. Captures depth AND
  dynamic scarcity; built behind the same `DraftStrategy` protocol, A/B-able against this slice's greedy
  version. The natural next slice.
- **numpy fast-path for the weekly fill** — if season-valuer tournament sweeps (e.g. `top_k` or `n_sims`
  tuning at high seed counts) are wanted, vectorize the greedy fill (PR #60 §3.4 deferral).
- **Conditional survival in the timing layer** — when the opportunity-cost layer lands, it inherits
  `now_or_never`'s unconditional-survival approximation (ignores that an available player already lasted to
  now); the conditional refinement applies there too.
