# Live Draft Assistant — Engine Core (Slice 1)

**Date:** 2026-06-09
**Sub-project:** #2 (Draft Hub), TODO #38 — live draft assistant
**Branch:** `feat/draft-assistant-engine`
**Predecessors:**
- Draft Hub on consensus (PR #56) — VORP table is now consensus-fed and carries `vorp` + `consensus_adp` (`VorpTableSchema`, `scripts/generate_vorp_table.py --source consensus`).
- External consensus blend (PR #55) — `ConsensusProjectionSchema` (the published preseason contract behind the VORP table).
- Existing draft surfaces — VORP (`2026-05-16-vorp-design.md`), snake cheat sheet (`2026-05-16-snake-cheat-sheet-design.md`), `LeagueConfig` (`src/projections/draft/league_config.py`).

---

## 0. Umbrella: the Draft Assistant sub-project

The cheat sheet and overall VORP board are **static, pre-draft** artifacts. The Draft Assistant is
the **live, stateful** surface: as a real draft unfolds you tell it who's been picked, and it
recommends *your* best pick right now — accounting for when your next pick comes and which players
will be gone by then (dynamic scarcity that static VORP cannot capture, because VORP's
positional-scarcity baseline is set pre-draft and never moves).

The sub-project decomposes into three independently shippable slices, each its own spec → plan →
implement cycle:

- **Slice 1 — Engine core (this spec):** headless, pure-Python recommendation engine + a CLI to
  drive it. The brain. Usable on its own (file-driven) before any UI exists.
- **Slice 2 — Strategy comparison harness:** a CLI tournament that simulates full drafts with
  different strategies in the seats, scores the resulting rosters, and declares an empirical winner.
  Builds on Slice 1's `DraftStrategy` protocol.
- **Slice 3 — Streamlit UI:** the live draft-day surface over the engine; mark a pick → the board
  and recommendation update.

This spec covers **Slice 1 only**. Slices 2 and 3 are designed for but not built here.

---

## 1. Purpose

Turn the existing static draft value signals (VORP + market ADP, already published per-player on the
consensus-fed VORP table) into a **live recommendation**: given the current state of a draft, rank
the available players by how good a pick they are *for me, right now*. The core question the engine
answers is **grab-now-vs-wait** — an elite player likely to be gone before my next pick is urgent;
an equally-valued player who will still be on the board next round is not.

The engine is built around a **pluggable strategy interface** so alternative strategies can be
substituted, unit-tested, and (in Slice 2) compared empirically to find the winner — consistent with
this repo's adoption-gate culture (we don't trust a heuristic until it's measured). Slice 1 ships the
analytic **now-or-never** strategy as the first real implementation plus a trivial **raw-VORP**
control (needed as the tournament baseline anyway).

---

## 2. Scope

### In scope (this slice)
1. A **draft-state model** (`DraftState`): the league, my slot, the ordered picks made so far, and
   the derived available pool + my current roster.
2. Pure **pick-timing** math (snake order): my upcoming pick numbers and `picks_until_next`.
3. A pluggable **survival model**: P(player still available at my next pick) from `consensus_adp`.
4. A **`DraftStrategy` protocol** (the substitution seam) + two concrete strategies:
   - `RawVorpStrategy` — best available by VORP, roster-need filtered (the control).
   - `NowOrNeverStrategy` — analytic opportunity-cost recommendation (Approach A).
5. **Roster-need awareness** (§3.6): a deterministic greedy slot-allocation rule (reusing `_pool.py`'s
   eligibility sets via a promoted shared helper) drops positions I can no longer roster and tiers
   open-starting-slot players ahead of bench-only depth (a scale-free preference, not an additive
   weight).
6. A **CLI** (`scripts/draft_assistant.py`) that reads a draft-state file + the consensus VORP table
   + `id_map`, runs a chosen strategy, and prints the ranked recommendation with names attached.

### Explicitly out of scope (later slices / other work)
- **Streamlit UI** — Slice 3.
- **Strategy comparison / draft simulation** — Slice 2. Slice 1 ships the protocol + two strategies
  but no tournament/scoring harness.
- **Monte Carlo strategy** — a future strategy plugged into the same protocol once the analytic core
  is trusted; not built here.
- **Live ESPN draft-API ingestion** — out of scope across all three slices for now; draft state is
  entered manually.
- **Re-deriving VORP** — the engine consumes the existing consensus-fed `VorpTableSchema`; it does
  not recompute value. Change source, not math (consistent with PR #56).
- **K/DST** — the consensus VORP table is skill-positions-only (TODO #10); the engine inherits that
  boundary.
- **Survival-model parameter tuning** — σ ships with a documented default; empirically tuning it is
  Slice 2 work.

---

## 3. Design

New package: `src/projections/draft/assistant/`. The engine is **gsis-id-native**; player-name
resolution (typing picks during a draft) is a CLI/UI concern, not the engine's.

### 3.1 Inputs

- **Consensus VORP table** — `VorpTableSchema`-validated frame from
  `generate_vorp_table.py --source consensus`. Provides, per player: `gsis_id`, `position`,
  `season_mean_fpts`, `vorp`, `consensus_adp` (nullable — the long tail has no ADP). This is the
  candidate universe. The engine does not re-derive any of these.
- **`id_map`** — `IdMapSchema` (`gsis_id` → `position` (non-nullable) + `full_name`). The **position
  source for *my* drafted players**, which the VORP table cannot supply on its own: the consensus
  VORP table is skill-positions-only and `has_points`-only (PR #56), so a drafted K/DST or a
  `has_points=False` player has no VORP row. `id_map` covers every gsis_id, so roster-need counts
  every pick I make. (The CLI already needs `id_map` for name display — same dependency.)
- **`LeagueConfig`** — existing pydantic model; supplies `n_teams` and `roster_slots`
  (`dict[RosterSlot, int]` — note: keyed by `RosterSlot`, including the shared `FLEX` /
  `SUPER_FLEX` / `BENCH` slots, **not** by `Position`). Drives roster-need (§3.6) and pool sizing.
- **Draft-state file** — see §3.2.

### 3.2 Draft-state model

`DraftState` (frozen dataclass), constructed from a small JSON file the CLI reads. Picks are an
**ordered list of gsis_ids** — a pick's slot is derived from its (1-based) position via snake order,
not stored, so there is nothing redundant to mis-enter:

```jsonc
{
  "league_config": "configs/league_espn_ppr_12team_skill.json",
  "my_slot": 7,                       // 1..n_teams; my seat in round 1
  "picks": [                          // gsis_ids in pick order; pick_number is the index + 1
    "00-0036900",
    "00-0034796"
    // ...
  ]
}
```

Derived (computed, not stored in the file):
- `drafted_ids: frozenset[GsisId]` — every `gsis_id` in `picks`.
- `current_pick: int` — `len(picks) + 1` (the pick about to be made).
- `slot_for(pick_number)` — snake-order slot of any absolute pick (see §3.3).
- `my_roster` — the positions I've filled: each pick whose derived slot equals `my_slot`, resolved
  to a `Position` via **`id_map`** (the universal position source — see §3.1). A pick whose
  `gsis_id` is absent from `id_map` is a hard error (we can't roster-account a player of unknown
  position); the CLI surfaces the offending id so it can be corrected.
- `available pool` — VORP rows whose `gsis_id ∉ drafted_ids`.

The picks list is validated at the boundary (each entry matches the GSIS pattern via
`validate_gsis_id`; the list is order-preserving) so malformed state fails loudly, per repo
convention. Duplicate `gsis_id` across picks is a hard error (a player can't be drafted twice).

**Precondition:** the engine assumes `current_pick` is *my* pick — the recommendation answers "who
should I take, on the clock, right now." (Running it off-turn still produces a ranking, but the
opportunity-cost framing in §3.5 is written for "it's my turn.")

### 3.3 Pick-timing (pure functions)

Snake order: round `r` (1-indexed) goes slot `1→n` on odd rounds and `n→1` on even rounds. From
`my_slot`, `n_teams`, and `current_pick`:

- `slot_for(pick_number)` → which slot owns an absolute pick.
- `my_upcoming_picks(current_pick)` → the ordered list of my absolute pick numbers `≥ current_pick`
  (so the current pick is the first entry when it's my turn).
- `my_next_pick(current_pick)` → my first pick **strictly after** `current_pick` — i.e. the pick
  *after* the one I'm about to make. This is the pick the survival model measures availability *at*:
  a candidate must survive picks `current_pick+1 .. my_next_pick−1` to still be there when I'm next
  up. Returns `None` when I have no pick after the current one (see §3.5 last-pick fallback).
- `picks_until_next(current_pick)` → count of *opponent* picks between this pick and `my_next_pick`
  (informational).

`my_next_pick` deliberately excludes the current pick: I take a player now with certainty; the
opportunity cost is against what survives to my *following* turn.

All pure, no pandas, hand-computable expected values for tests. E.g. slot 7 of 12: my picks are
7, 18, 31, 42 …; standing at pick 7, `my_upcoming_picks` = [7, 18, 31, …], `my_next_pick` = 18, the
intervening opponent picks are 8–17 (`picks_until_next` = 10), and a candidate's availability is
evaluated as "not taken on or before pick 17."

### 3.4 Survival model (pluggable)

`SurvivalModel` — callable interface: `p_available(adp: float, at_pick: int) → float in [0,1]`,
where `at_pick` is the absolute pick number I want the player to still be available *at* (my next
pick `N`).

Default implementation `LogisticSurvival(sigma)` uses the **logistic CDF** (sigmoid) in ADP space:
`P(taken on or before pick n) = σ((n − adp) / sigma)` where `σ(x) = 1/(1+e^−x)`, so
`p_available(adp, N) = 1 − σ((N − 1 − adp) / sigma)` — available *at* `N` means *not taken on or
before pick `N − 1`*. One global spread parameter `sigma` (picks); default documented (≈ two-thirds
of a round, refined empirically in Slice 2). The CDF *shape* (logistic vs normal) is not
load-bearing — both are monotone in ADP and deterministic; we use the logistic to keep the name and
math aligned and avoid a scipy dependency.

Null-ADP players (long tail, no market signal): survival ≈ 1.0 (treated as "won't be taken soon") —
they are never the urgent pick, which is correct.

**Known approximation (v1):** `p_available` is *unconditional* — it does not condition on the fact
that a candidate has already lasted to the current pick. A player who has fallen well past their ADP
(still on the board at pick 7 with ADP 3) is scored as near-certain to be gone, ignoring the
evidence that they're sliding. Strategies only ever evaluate currently-available players, so a
survive-to-now conditioning would be more correct; it's deferred as a Slice-2 refinement (it changes
the survival model, not the strategy, thanks to the injection seam). Acceptable for v1 because ADP
sliders are a small minority and the error is conservative (it never *over*-rates a faller's
availability).

The model is injected into strategies, so survival assumptions can be swapped without touching
strategy logic.

### 3.5 `DraftStrategy` protocol + strategies

```python
@runtime_checkable
class DraftStrategy(Protocol):
    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame: ...   # RecommendationSchema-validated, ranked best-first
```

(Per repo convention: `runtime_checkable` is structural-only; mypy enforces the real contract.)

`RecommendationSchema` (consumer-facing output) columns: `gsis_id`, `position`, `vorp`,
`consensus_adp` (nullable), `p_available_next` (nullable — null for null-ADP players),
`fills_starting_slot` (bool — the §3.6 starting-need tier), `score` (the strategy's ranking value),
`rank` (1-based). Names are attached by the CLI, not the engine.

**Final ordering (both strategies).** After a strategy computes `score`, the engine produces `rank`
by sorting on `(fills_starting_slot desc, score desc, vorp desc, gsis_id asc)` — the §3.6
starting-need tier first, then the strategy's score, then deterministic tie-breaks. Because
`fills_starting_slot` is constant within a position (§3.6), this outer tier never splits a position,
so the within-position property below is preserved.

**`RawVorpStrategy`** (control): filter the pool to roster-eligible positions (§3.6), `score = vorp`,
rank desc. It takes **no survival model** and leaves `p_available_next` null — the trivial
"best-available" baseline, decoupled from any timing assumption.

**`NowOrNeverStrategy`** (Approach A): for each roster-eligible position, sort its available players
by **`vorp` desc, then `gsis_id` asc** (deterministic; ties never make the survivor sum
order-dependent) and compute the **expected VORP of the best player still available at my next pick**:

```
E[best survivor VORP at pos] = Σ_i  vorp_i · p_i · Π_{j<i} (1 − p_j)
```

where `p_i = SurvivalModel.p_available(adp_i, state.my_next_pick)` and the product runs over the
earlier-sorted (higher-VORP) players at the same position (the chance every better one is gone).
`cost_of_waiting(pos) =
vorp(best available now at pos) − E[best survivor VORP at pos]`. The per-player score:

```
score(p) = vorp(p) − E[best survivor VORP at p's position]
```

i.e. the value I lock in *over what I could expect at this position if I pass now*. The top-ranked
player is the most urgent good value. Deterministic; every intermediate (VORP, ADP, `p_available`,
expected-alternative) is surfaced for explainability.

**Ranking property (intended, load-bearing):** the subtracted term `E[best survivor VORP at pos]`
is **constant across all players at a position**. Therefore *within* a position the ranking is
exactly VORP order — the strategy never reorders two same-position players. Its entire effect is
**cross-position**: it decides *which position to attack now* by how depleted that position will be
by my next pick (`cost_of_waiting`), then takes the best player there. This is the intended design
(chosen over the alternative of ranking positions by `cost_of_waiting` and picking best-within,
which produces the same ordering when there is one open slot per position; the additive form
generalizes more cleanly and keeps `score` directly comparable across positions). The §4 "reorders
vs raw VORP" test exercises exactly this cross-position reordering.

**Last-pick fallback:** when `my_next_pick` is `None` (the current pick is my final pick — no future
turn to wait for), there is no survival window and the opportunity-cost framing is undefined.
`NowOrNeverStrategy` falls back to `RawVorpStrategy`'s ranking (take the best available), with
`p_available_next` left null. This is a stated requirement with its own test.

Edge cases: a position with only one available player → `E[best survivor]` over an empty
"better-players" set is just `vorp · p_avail`; a position I can no longer roster is filtered out
before scoring (so it never appears, and never anchors another position's cost).

### 3.6 Roster-need awareness

Because `roster_slots` is keyed by `RosterSlot` with shared `FLEX` / `SUPER_FLEX` / `BENCH` slots,
"how many more of position P can I roster" is not a per-position constant — it depends on how my
existing picks consumed shared slots. We make it well-defined with an explicit **greedy
slot-allocation** rule, reusing the eligibility sets already defined in `_pool.py`
(`_FLEX_ELIGIBLE = {RB, WR, TE}`, `_SUPER_FLEX_ELIGIBLE = {QB, RB, WR, TE}`, and its bench-eligibility
rule that excludes positions the league doesn't roster). To avoid a private-symbol import, those sets
+ the slot-fill priority are promoted to a small shared helper (e.g.
`projections/draft/roster_eligibility.py`) that both `_pool.py` and the assistant import — one source
of truth for slot↔position eligibility.

**Allocation (deterministic):** start from my league's per-team slot counts (`roster_slots`). Place
each of *my* drafted players (from `my_roster`) into the first open slot in priority order — its own
position slot → `FLEX` (if eligible) → `SUPER_FLEX` (if eligible) → `BENCH` (if the league benches
that position). The leftover open slots after placing all my picks define need.

- **Roster-eligible position** P: there exists an open slot P could occupy — P's position slot, a
  `FLEX` (if `P ∈ _FLEX_ELIGIBLE`), a `SUPER_FLEX` (if `P ∈ _SUPER_FLEX_ELIGIBLE`), or `BENCH` (if P
  is benchable and bench capacity remains). Ineligible positions (I've filled every slot they could
  go in) are dropped from the candidate pool before scoring.
- **Starting-need tier (scale-free).** Each eligible player is tagged `fills_starting_slot` = there
  is an open **non-`BENCH`** slot its position could occupy (its position slot, `FLEX`, or
  `SUPER_FLEX`). The recommendation is then ordered **lexicographically**: starters-tier
  (`fills_starting_slot=True`) ahead of bench-depth-tier, and *within each tier* by the strategy's
  `score`. This is a deliberately scale-free preference — it works identically for `RawVorpStrategy`
  and `NowOrNeverStrategy` regardless of their different score magnitudes (no additive constant whose
  size would mean different things to each), so the engine won't rank luxury bench depth over filling
  an open starting slot (e.g. a backup RB over a startable WR when a WR/FLEX start is still open),
  while never reordering two players who share a tier. `fills_starting_slot` is surfaced as an output
  column so the tiering is explainable.

Kept deliberately simple and rule-based for v1; richer roster-construction logic is a future strategy
concern, not an engine primitive.

### 3.7 CLI surface

`scripts/draft_assistant.py`:
- `--state PATH` (draft-state JSON; **required**), `--vorp-table PATH` (consensus VORP table written
  by `generate_vorp_table.py --source consensus --out ...`; **required** — there is no canonical
  partitioned VORP location to default to, the VORP CLI writes to a caller-named `--out`),
  `--strategy {now_or_never,raw_vorp}` (default `now_or_never`), `--top N` (rows to print,
  default 15), `--sigma FLOAT` (override survival spread).
- Resolves player names + positions from `id_map` (`full_name` for display; `position` for
  `my_roster`, per §3.2). Prints a ranked table: rank, name, position, VORP, ADP,
  P(available next pick), score.
- This is the surface Slice 2's harness and our in-session strategy comparisons drive.

---

## 4. Testing

Per the repo's correctness bar, each pure unit gets isolated tests with hand-computed expected values:
- **Pick-timing** — snake math across slots/rounds (incl. wrap at round boundaries); `my_upcoming_picks`
  includes the current pick when it's mine; `my_next_pick` is strictly after `current_pick` (e.g.
  slot 7/12 at pick 7 → 18) and returns `None` at my last pick; `picks_until_next`.
- **Survival model** — monotonicity, boundary (adp ≪ pick → ~0; adp ≫ pick → ~1), null-ADP → 1.0.
- **`NowOrNeverStrategy`** — small hand-built pools where the expected-survivor sum and resulting
  ranking are computed by hand; a case where now-or-never reorders vs raw VORP (urgent scarce
  position attacked over a higher-VORP-but-safe one), proving the strategies differ; a within-position
  case confirming order equals VORP order (the ranking property); the **last-pick fallback** to
  raw-VORP when `my_next_pick is None`.
- **`RawVorpStrategy`** — pure VORP order; roster-eligibility filtering; `p_available_next` stays null.
- **Roster-state / eligibility** — greedy slot allocation: a player consumes its position slot, then
  `FLEX`, then `SUPER_FLEX`, then `BENCH`; a position with every eligible slot filled is dropped;
  starting-need tiering puts an open-starting-slot position ahead of a bench-only one *even when the
  bench-only player has the higher score* (proves the tier dominates score), while two players in the
  same tier keep score order; bench-sharing across positions is respected (the same shared helper
  `_pool.py` uses).
- **`my_roster` / `id_map` resolution** — my picks resolve to positions via `id_map`; a pick whose
  `gsis_id` is missing from `id_map` raises with the offending id; a drafted player absent from the
  VORP table (e.g. off-board) is still position-counted via `id_map`.
- **Protocol conformance** — both strategies satisfy `DraftStrategy` structurally; output validates
  against `RecommendationSchema`.
- **Determinism / tie-breaks** — equal-`vorp` same-position players resolve by `gsis_id` (survivor
  sum is order-independent); equal-`score` players resolve by `(vorp desc, gsis_id asc)` so `rank` is
  reproducible across runs.
- **CLI smoke** — runs end-to-end on a tiny fixture state + VORP table + id_map, prints a stable table.

Gates (per `CLAUDE.md`): `pytest`, `mypy src tests`, `ruff check`, `ruff format --check` all clean.

---

## 5. Key decisions

- **Engine consumes the consensus VORP table; does not re-derive value** — change source, not math
  (consistent with PR #56). VORP already encodes static positional scarcity; the engine adds only the
  *dynamic* (pick-timing) layer.
- **Pluggable `DraftStrategy` protocol from day one** — the substitution seam is the point; it makes
  Slice 2's empirical tournament possible and mirrors the existing `Distribution` Protocol pattern.
- **Analytic now-or-never first, Monte Carlo deferred** — A is deterministic, transparent, testable
  against the correctness bar, and uses only data we have; B rests on unvalidated opponent-modeling
  assumptions and is a later strategy behind the same protocol.
- **Survival via ADP + a single global σ** — simplest defensible model; per-player ADP spread isn't
  available today (consensus ADP is a mean). σ is configurable and gets empirically tuned in Slice 2.
- **Engine is gsis-id-native; `id_map` is the universal position + name source** — the VORP table
  can't supply a position for non-skill / `has_points=False` picks, so `my_roster` resolves positions
  via `id_map` (which the CLI already loads for names). A pick missing from `id_map` is a hard error.
- **Roster-need is a deterministic greedy-allocation rule, not a model** — slot↔position eligibility
  is shared with `_pool.py` (promoted out of its private symbols) so there is one source of truth;
  enough to avoid drafting a 4th QB. Richer construction logic is a future strategy, not an engine
  primitive.
- **Now-or-never reorders only across positions** — the opportunity-cost offset is a per-position
  constant, so within a position the order is plain VORP; the strategy's job is choosing *which
  position to attack now*. Falls back to raw VORP at my last pick.

---

## 6. Open questions / future slices

- **Slice 2 (harness):** simulate full drafts with strategies in seats, score final rosters (by
  projected points / VORP captured), declare an empirical winner; tune σ. The `DraftStrategy` protocol
  and both strategies from this slice are its inputs.
- **Slice 3 (UI):** Streamlit live board over this engine.
- **Per-player ADP spread** — when a scraped multi-source ADP arrives (TODO #38 #2b+), the survival
  model can use a real per-player spread instead of a global σ.
- **K/DST** — inherits TODO #10; unblocked when the consensus layer covers them.
