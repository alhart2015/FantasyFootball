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
5. **Roster-need awareness**: recommendations respect `LeagueConfig` slot maxes and prefer unfilled
   starting slots over bench depth.
6. A **CLI** (`scripts/draft_assistant.py`) that reads a draft-state file + the consensus VORP table,
   runs a chosen strategy, and prints the ranked recommendation with player names attached.

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
- **`LeagueConfig`** — existing dataclass; supplies `n_teams`, roster slot structure (starting slots
  per position, bench size), and drives roster-need + pool sizing.
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
- `my_roster` — positions I've filled: the picks whose derived slot equals `my_slot`, mapped to
  `position` via the VORP table.
- `available pool` — VORP rows whose `gsis_id ∉ drafted_ids`.

The picks list is validated at the boundary (each entry matches the GSIS pattern via
`validate_gsis_id`; the list is order-preserving) so malformed state fails loudly, per repo
convention. Duplicate `gsis_id` across picks is a hard error (a player can't be drafted twice).

### 3.3 Pick-timing (pure functions)

Snake order: round `r` (1-indexed) goes slot `1→n` on odd rounds and `n→1` on even rounds. From
`my_slot`, `n_teams`, and `current_pick`:

- `slot_for(pick_number)` → which slot owns an absolute pick.
- `my_upcoming_picks(current_pick)` → the ordered list of my remaining absolute pick numbers
  (including the current pick if it's mine).
- `my_next_pick(current_pick)` → my next absolute pick number `N` — the pick the survival model
  measures availability *at* (a player must survive picks `current_pick..N−1` to be there for me).
- `picks_until_next(current_pick)` → count of *opponent* picks before my next turn (informational).

All pure, no pandas, hand-computable expected values for tests. E.g. slot 7 of 12: my picks are
7, 18, 31, 42 …; standing at pick 7, `my_next_pick` = 18, the intervening opponent picks are 8–17
(so `picks_until_next` = 10), and a candidate's availability is evaluated as "not taken on or before
pick 17."

### 3.4 Survival model (pluggable)

`SurvivalModel` — callable interface: `p_available(adp: float, at_pick: int) → float in [0,1]`,
where `at_pick` is the absolute pick number I want the player to still be available *at* (my next
pick `N`).

Default implementation `LogisticSurvival(sigma)`:
`P(taken on or before pick n) = Φ((n − adp) / sigma)` (normal CDF), so
`p_available(adp, N) = 1 − Φ((N − 1 − adp) / sigma)` — available *at* `N` means *not taken on or
before pick `N − 1`*. One global spread parameter `sigma` (picks); default documented (≈ two-thirds
of a round, refined empirically in Slice 2). Monotone in ADP, deterministic.

Null-ADP players (long tail, no market signal): survival ≈ 1.0 (treated as "won't be taken soon") —
they are never the urgent pick, which is correct.

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
`score` (the strategy's ranking value), `rank` (1-based). Names are attached by the CLI, not the
engine.

**`RawVorpStrategy`** (control): filter the pool to roster-eligible positions (§3.6), `score = vorp`,
rank desc. `p_available_next` populated for transparency but unused in ranking. The "best available"
baseline.

**`NowOrNeverStrategy`** (Approach A): for each roster-eligible position, sort its available players
by VORP desc and compute the **expected VORP of the best player still available at my next pick**:

```
E[best survivor VORP at pos] = Σ_i  vorp_i · p_i · Π_{j<i} (1 − p_j)
```

where `p_i = SurvivalModel.p_available(adp_i, state.my_next_pick)` and the product runs over higher-VORP
players at the same position (the chance every better one is gone). `cost_of_waiting(pos) =
vorp(best available now at pos) − E[best survivor VORP at pos]`. The per-player score:

```
score(p) = vorp(p) − E[best survivor VORP at p's position]
```

i.e. the value I lock in *over what I could expect at this position if I pass now*. The top-ranked
player is the most urgent good value. Deterministic; every intermediate (VORP, ADP, `p_available`,
expected-alternative) is surfaced for explainability.

Edge cases: a position with only one available player → `E[best survivor]` over an empty
"better-players" set is just `vorp · p_avail`; a position I can no longer roster is filtered out
before scoring (so it never appears, and never anchors another position's cost).

### 3.6 Roster-need awareness

From `LeagueConfig` roster structure and `my_roster`:
- A position is **roster-eligible** if I have not hit its maximum rosterable count (starting slots +
  share of bench + flex eligibility). Ineligible positions are dropped from the candidate pool before
  scoring.
- Among eligible positions, **unfilled starting slots** are preferred over pure bench depth: a small,
  documented starting-need weight nudges the score so the engine won't rank a luxury bench RB over a
  startable WR when WR is still an open starting slot. The weight is a named constant, not magic.

Kept deliberately simple and rule-based for v1; richer roster-construction logic is a future strategy
concern, not an engine primitive.

### 3.7 CLI surface

`scripts/draft_assistant.py`:
- `--state PATH` (draft-state JSON), `--vorp-table PATH` (consensus VORP parquet; default to the
  latest under `data/processed/`), `--strategy {now_or_never,raw_vorp}` (default `now_or_never`),
  `--top N` (rows to print, default 15), `--sigma FLOAT` (override survival spread).
- Resolves player names from `id_map` (+ consensus placeholder names for rookies) for display only;
  prints a ranked table: rank, name, position, VORP, ADP, P(available next pick), score.
- This is the surface Slice 2's harness and our in-session strategy comparisons drive.

---

## 4. Testing

Per the repo's correctness bar, each pure unit gets isolated tests with hand-computed expected values:
- **Pick-timing** — snake math across slots/rounds (incl. wrap at round boundaries); `picks_until_next`.
- **Survival model** — monotonicity, boundary (adp ≪ pick → ~0; adp ≫ pick → ~1), null-ADP → 1.0.
- **`NowOrNeverStrategy`** — small hand-built pools where the expected-survivor sum and resulting
  ranking are computed by hand; a case where now-or-never reorders vs raw VORP (urgent scarce player
  jumps a higher-VORP-but-safe player), proving the strategies differ.
- **`RawVorpStrategy`** — pure VORP order, roster-eligibility filtering.
- **Roster-need** — a filled position is dropped; a startable open slot is preferred over bench depth.
- **Protocol conformance** — both strategies satisfy `DraftStrategy` structurally; output validates
  against `RecommendationSchema`.
- **CLI smoke** — runs end-to-end on a tiny fixture state + VORP table, prints a stable table.

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
- **Engine is gsis-id-native; name resolution lives in the CLI/UI** — keeps the brain pure and the
  display concern at the edge.
- **Roster-need is a simple rule, not a model** — enough to avoid drafting a 4th QB; richer
  construction logic is a future strategy, not an engine primitive.

---

## 6. Open questions / future slices

- **Slice 2 (harness):** simulate full drafts with strategies in seats, score final rosters (by
  projected points / VORP captured), declare an empirical winner; tune σ. The `DraftStrategy` protocol
  and both strategies from this slice are its inputs.
- **Slice 3 (UI):** Streamlit live board over this engine.
- **Per-player ADP spread** — when a scraped multi-source ADP arrives (TODO #38 #2b+), the survival
  model can use a real per-player spread instead of a global σ.
- **K/DST** — inherits TODO #10; unblocked when the consensus layer covers them.
