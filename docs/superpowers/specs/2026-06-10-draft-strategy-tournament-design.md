# Draft Strategy Comparison Harness (Slice 2)

## 0. Umbrella: the Draft Assistant sub-project

The live **Draft Assistant** (sub-project #2, TODO #38) is shipping in three independently-deployable
slices over the consensus VORP table:

1. **Slice 1 — engine core (shipped, PR #57):** the `DraftStrategy` protocol + two concrete strategies
   (`NowOrNeverStrategy`, `RawVorpStrategy`), survival model, pick-timing, draft-state, and a
   single-recommendation CLI. See `docs/superpowers/specs/2026-06-09-draft-assistant-engine-design.md`.
2. **Slice 2 — strategy comparison harness (this spec):** a CLI tournament that simulates full drafts
   with a strategy in the hero seat against an ADP-driven field, scores the resulting rosters, and
   declares an empirical winner — plus a σ-tuning mode for the survival model.
3. **Slice 3 — Streamlit UI (later):** a live draft-day board over the engine.

Slice 1 deliberately shipped the substitution seam (`DraftStrategy` is a `runtime_checkable` Protocol,
mirroring `Distribution`) but **no way to tell which strategy is actually better**. `NowOrNeverStrategy`
is analytically motivated (value locked in over the expected best survivor) but unmeasured, and the
survival model's global σ default (`⅔·n_teams`) was a guess flagged for empirical tuning. This slice
closes both gaps.

## 1. Purpose

Answer, with a reproducible number: **does `NowOrNeverStrategy` beat the best-available `RawVorpStrategy`
control, and at what σ?** Do it by simulating realistic drafts — the hero seat runs the strategy under
test against a field of ADP-following bots — scoring each completed roster by the points it would
actually start, and comparing strategies on paired draft realizations with a bootstrap confidence
interval. The harness is the empirical backbone the Draft Assistant needs before its recommendations
are trusted or its σ is fixed.

A hard constraint, called out by the user: the harness must be **league-driven, not hardcoded**.
Roster shape, team count, and scoring ruleset vary by league; all three come from the `LeagueConfig`
the user already passes. Nothing in the harness assumes a 12-team PPR room.

## 2. Scope

### In scope (this slice)

- **`opponent.py`** — the ADP-bot pick policy (lowest noisy `consensus_adp`, seeded RNG).
- **`simulation.py`** — `simulate_draft`: one full snake draft, hero seat via a `DraftStrategy`, all
  other seats via the ADP-bot; returns the hero's final roster.
- **`roster_score.py`** — `optimal_lineup_points`: value a completed roster by its optimal starting
  lineup's `season_mean_fpts` (starters only; bench scores nothing), with slots read from `LeagueConfig`.
- **`tournament.py`** — `run_tournament` (compare strategies over many seeds, mean + bootstrap CI,
  paired-difference winner test) and `tune_sigma` (sweep σ for `NowOrNeverStrategy`).
- **CLI** — `scripts/draft_tournament.py` → `assistant/cli.py` entry, two modes: `compare`, `tune-sigma`.
- Tests following the project's TDD + synthetic-fixture norms.

### Explicitly out of scope (later slices / other work)

- **Auction draft simulation** — a deliberate future effort (user-confirmed). The slice is structured
  (simulate → score → compare) so the auction equivalent reuses `optimal_lineup_points` and the whole
  tournament/bootstrap stats layer unchanged; only the draft-mechanism module swaps (nominate/bid loop
  instead of snake order). `LeagueConfig` already carries `budget`/`min_bid`, so no config work is
  deferred onto that effort. **§5.7 documents the seam** so the later auction slice inherits it.
- **Hero-slot sweep** — averaging the hero across all draft slots (Q3 option C). A later refinement;
  v1 fixes the hero at a configurable `--my-slot`.
- **Conditional / Monte-Carlo survival refinement** — the survival model's unconditional approximation
  (it ignores that an available player already survived to now) stays as-is; this harness *measures* the
  current model, it doesn't replace it. Improving survival is a separate strategy behind the same Protocol.
- **Opponent strategy modeling** — bots model the field via ADP only. No opponent now-or-never logic.
- **Non-PPR consensus tables** — the upstream re-score (`refresh_consensus` accepting a ruleset) is a
  separate, already-logged backlog item. This harness reads whatever ruleset the table was built under;
  the config seam is ready before the data is.
- **Streamlit UI** — Slice 3.

## 3. Design

### 3.1 Inputs

All three inputs already exist; the harness adds no new ingest or schema-producing path.

- **Pool** — a consensus VORP table (`VorpTableSchema`) parquet, the exact frame Slice 1's strategies
  consume. It carries `gsis_id`, `position`, `season_mean_fpts` (consensus projected PPR points,
  scored upstream under the league ruleset), `vorp`, and optional `consensus_adp`. **`season_mean_fpts`
  is the roster-scoring currency** and **`consensus_adp` drives the bots**; a pool with an all-null
  `consensus_adp` is a hard error in this harness (a tournament with no market signal is meaningless —
  unlike the single-recommendation engine, which degrades gracefully). This check is enforced once at the
  `run_tournament` / `tune_sigma` entry (sharing a small `_validate_pool` helper with the §3.3
  pool-sufficiency check), so both CLI modes inherit it; `simulate_draft` itself assumes a validated pool.
- **`LeagueConfig`** — team count (`n_teams`), roster shape (`roster_slots`), and scoring ruleset
  (`ruleset`). **The documented precondition:** this must be the same config the VORP table was
  generated under. The parquet does not carry a ruleset column, so the harness cannot verify the match
  — it is the caller's responsibility, exactly as `generate_vorp_table` and the cheat sheet already pair
  a config with a table. The CLI mirrors the `generate_vorp_table` `--league-config` convention to make
  the correct pairing the obvious one.
- **`id_map`** — only needed to resolve the hero's roster positions when a drafted player's position
  must be known. Positions are already in the VORP pool for every in-pool player, and the simulation
  only ever drafts in-pool players, so **the harness needs no id_map** (a simplification over Slice 1's
  `load_draft_state`, which had to resolve off-board human picks). Stated here to make the absence
  deliberate, not an oversight.

### 3.2 Opponent model — the ADP-bot (`opponent.py`)

A bot occupies every non-hero seat. Its policy is pure noisy-ADP, deterministic given its RNG:

```
bot_pick(available: DataFrame, rng) -> gsis_id
```

1. Draw a **noisy ADP** per available player: `noisy_adp = consensus_adp + rng.normal(0, adp_jitter)`.
   Players with null `consensus_adp` sit at the back (treated as `+inf` — no market signal, so bots leave
   them to the hero).
2. Pick the player with the **lowest** noisy ADP (deterministic tie-break on `gsis_id`).

**The bot takes no roster argument — realism comes from ADP, not from a positional constraint.**
Consensus ADP already spaces positions the way a real room does (QBs leave the board sparsely), so
noisy-ADP bots produce realistic rosters without any roster-construction rule. A roster-eligibility
filter would be a no-op here anyway: under a shared bench, `eligible_positions` never drops a benchable
position until a team's roster is *full* (its bench is position-agnostic — see `roster_eligibility.py`),
and snake order already gives each seat exactly `roster_size` picks. Pure-ADP bots are therefore the
honest, simplest field; opponent roster-construction modeling is an explicit non-goal (§2). The VORP pool
contains only positions the league rosters (the generator filters to `roster_slots`), so a bot can never
draft an un-rosterable position.

`adp_jitter` (the bot reach/fall spread, in picks) is a harness parameter **distinct from the hero
survival model's σ**. Default `adp_jitter = default_sigma(n_teams)` — reusing the same "≈⅔ of a round"
intuition — but the two never share a variable; conflating them is the kind of bug §4 tests guard.

### 3.3 Draft simulation (`simulation.py`)

```
simulate_draft(strategy: DraftStrategy, my_slot: int, pool, config, *, adp_jitter, rng) -> DataFrame
```

Runs absolute picks `1 .. n_teams * roster_size`. For each pick:

- `slot = pick_timing.slot_for(pick_number, config.n_teams)`.
- If `slot == my_slot`: build a `DraftState` from the picks made so far (the hero's own picks resolve to
  positions straight from the pool — no id_map), call `strategy.recommend(state, pool, config)`, and take
  the rank-1 row. The strategy already filters drafted + roster-ineligible players, so the top row is a
  legal pick.
- Else: `bot_pick(available, rng)`.

The function tracks `available` (pool minus drafted) and the hero's own drafted-position list (for the
`DraftState` it builds at each hero pick); bots need no roster tracking (pure-ADP, §3.2). It returns the
hero's drafted rows (a sub-frame of the pool) — the input to scoring.

**Pool sufficiency.** A full draft needs `n_teams * roster_size` distinct draftable players. If the pool
is smaller, the run raises a clear error rather than running off the end of `available` (a silently short
hero roster would mis-score). This is the second check in the shared `_validate_pool` helper called at the
`run_tournament` / `tune_sigma` entry (alongside the §3.1 all-null-ADP check); `simulate_draft` assumes a
validated pool. Starting slots that *no* pool player can fill — e.g. `K` / `DST` starting slots run
against the skill-only consensus pool, which contains neither — are simply never drafted and score 0 in
§3.4 by design. The intended pairing is the skill-only league config
(`configs/league_espn_ppr_12team_skill.json`, which folds K/DST into BENCH), consistent with the rest of
the consensus Draft Hub.

**Determinism & the paired counterfactual.** One seeded `numpy.random.Generator` per draft drives every
bot's noise. When comparing strategy A vs B at the same seed, the bots draw the *same* noise sequence, so
A and B face the same field realization up to the point the hero's own (differing) picks perturb the
board downstream. That is the honest "same draft, different me" counterfactual and it makes the
paired-difference test in §3.5 far lower-variance than independent draws. Same seed + same strategy ⇒
byte-identical hero roster (a tested invariant).

A `DraftStrategy` is stateless across the draft — it is re-asked from the current `DraftState` at each
hero pick — so swapping strategies between runs is clean and the engine surface from Slice 1 is reused
unchanged.

### 3.4 Roster scoring (`roster_score.py`)

```
optimal_lineup_points(roster_rows: DataFrame, roster_slots: dict[RosterSlot, int]) -> float
```

Value a completed roster by the points it would actually start:

1. Group the roster's `season_mean_fpts` by `position`, each group sorted desc with a deterministic
   `gsis_id`-ascending secondary key — so *which* equal-points player fills a slot (and thus what is left
   for later slots) is reproducible.
2. **Fill slots in ascending eligibility-breadth order:** single-position slots (`RosterSlot.QB →
   Position.QB`, etc.) first, then `FLEX` (RB/WR/TE), then `SUPER_FLEX` (QB/RB/WR/TE) — each slot taking
   the best *unused* player it is eligible for.
3. Sum the `season_mean_fpts` of the filled starters. `BENCH` / `IR` slots contribute nothing.

**The fill order is load-bearing, not cosmetic.** Greedy is optimal for this slot structure *only* when
slots fill most-restrictive-first. The eligibility sets are laminar (`{QB} ⊂ SUPER_FLEX`,
`{RB} ⊂ FLEX ⊂ SUPER_FLEX`, …), which is exactly the condition under which restrictive-first greedy is
optimal — no assignment solver needed. Filling a wider slot before a narrower one can strand a player and
undercount: roster `{RB:100, QB:90}`, slots `{FLEX:1, SUPER_FLEX:1}` scores **190** the correct way
(FLEX←RB 100, SUPER_FLEX←QB 90), but only **100** if `SUPER_FLEX` greedily grabs RB 100 first and `FLEX`
is left with an ineligible QB. §4 pins this case. Slot↔position eligibility reuses
`roster_eligibility`'s `FLEX_ELIGIBLE` / `SUPER_FLEX_ELIGIBLE` sets (one source of truth; no second copy
of "what can play FLEX"). The roster shape is `roster_slots` verbatim from `LeagueConfig`, so any
composition (SUPER_FLEX, 3-WR, K/DST-less skill leagues) scores correctly with no code change.

If a starting slot cannot be filled (roster short at a position), it contributes 0 and the lineup is
simply worth less — a realistic penalty for a strategy that left a starting hole, not an error.

### 3.5 Tournament + winner claim (`tournament.py`)

```
run_tournament(strategies: dict[str, DraftStrategy], pool, config, *, my_slot, n_seeds, adp_jitter, base_seed) -> TournamentResult
```

For each strategy and each seed `s in 0 .. n_seeds-1`:

- `rng = Generator(PCG64(base_seed + s))` — **the same `s` yields the same bot field across strategies**
  (the paired design).
- `roster = simulate_draft(strategy, my_slot, pool, config, adp_jitter=adp_jitter, rng=rng)`.
- `value[strategy][s] = optimal_lineup_points(roster, config.roster_slots)`.

Then:

- **Per-strategy summary:** mean starting-lineup points + a percentile-bootstrap CI over the `n_seeds`
  values (mirrors the adoption-gate bootstrap aesthetic already in the codebase).
- **Winner test (paired):** for the two-strategy head-to-head, bootstrap the **paired per-seed difference**
  `value[A][s] − value[B][s]`; declare a winner only when the CI excludes 0. Pairing on the shared field
  is what gives the test its power. With >2 strategies, report each strategy's mean+CI and the pairwise
  difference of the top two.

`TournamentResult` is a small dataclass (per-strategy means/CIs, the paired-diff CI, the declared winner
or "no separation", and the run parameters for reproducibility). The CLI renders it; no parquet, no
pandera schema (the result is a handful of floats — a schema would be ceremony). If a future consumer
needs the per-seed matrix persisted, that earns a schema then, not now.

### 3.6 σ-tuning mode (`tournament.py`)

```
tune_sigma(sigma_grid: Sequence[float], pool, config, *, my_slot, n_seeds, adp_jitter, base_seed) -> SigmaTuningResult
```

For each σ in the grid, build `NowOrNeverStrategy(LogisticSurvival(σ))` and run it through the same
paired-seed simulation/scoring as §3.5 (same `base_seed`, so every σ faces the same fields — paired
across the grid too). Report σ → mean hero value and pick the argmax. Default grid centered on
`survival.default_sigma(n_teams)` (e.g. `[⅓, ½, ⅔, 1, 1⅓]·n_teams`). Output is a small table + the
recommended σ; the user updates their default from it. This is the empirical tuning Slice 1 deferred.

### 3.7 CLI surface

`scripts/draft_tournament.py` → `assistant/cli.py`:

```
python scripts/draft_tournament.py \
    --vorp-table <consensus_vorp.parquet> \
    --league-config <league.json> \
    --my-slot N \
    [--seeds K] [--adp-jitter F] [--seed BASE] \
    {compare | tune-sigma} [--sigma-grid "a,b,c" | --strategy-sigma S]
```

- `compare` — runs the registered strategies (`now_or_never` with a default/`--strategy-sigma` σ, and
  `raw_vorp`) and prints the per-strategy table + winner line.
- `tune-sigma` — runs the σ grid and prints the σ table + recommended σ.
- `--league-config` is required and is the single source of roster shape + team count + ruleset
  (matching `generate_vorp_table`'s flag). `--my-slot` fixes the hero seat. `--seed` makes any run
  reproducible. All defaults (`--seeds`, `--adp-jitter`, grid) are sensible so a bare invocation works.

The CLI lives behind the same engine/UI separation as Slice 1: `cli.py` imports the engine, never the
reverse.

## 4. Testing

Synthetic fixtures only (project norm — no network, no real parquet in unit tests):

- **Pick order** — on a tiny hand-built pool + a 2-team / few-round config, assert `simulate_draft`
  assigns the right picks to hero vs bot seats under snake order (reuses `slot_for`).
- **Determinism** — same `base_seed` + same strategy ⇒ identical hero roster; different seed ⇒ generally
  different (guards the "every p_available silently NaN" class of RNG/σ wiring bugs).
- **Paired field** — same seed, two strategies: assert the bots' early picks (before the hero diverges
  the board) are identical across the two runs (proves the paired counterfactual actually holds).
- **Bot policy** — `bot_pick` returns the lowest noisy-ADP player; with `adp_jitter → 0` it is exactly
  the min-`consensus_adp` available player, and a null-ADP player is taken only when nothing else remains.
- **Pool exhaustion** — a config whose `n_teams * roster_size` exceeds the pool size raises a clear error
  (not an off-the-end crash); and a config with `K`/`DST` starting slots run on a skill-only pool drafts
  no K/DST and scores those slots 0 (the unfillable-slot path in §3.4).
- **All-null-ADP pool** — `run_tournament` / `tune_sigma` raise a clear error when the pool's
  `consensus_adp` is entirely null (the §3.1 hard constraint), exercised at the entry point that enforces
  it, so both CLI modes are covered.
- **`adp_jitter` vs survival σ are independent knobs** — `RawVorpStrategy` results are invariant to the
  survival σ; `NowOrNeverStrategy` results change with σ at fixed `adp_jitter`; `simulate_draft` passes
  `adp_jitter` to bots only and never to the strategy's σ (guards conflation).
- **`optimal_lineup_points`** — known roster with a SUPER_FLEX and a FLEX; assert the optimal starters are
  chosen (best of each position to its slot, **FLEX before SUPER_FLEX**). Includes the strand case from
  §3.4 (`{RB:100, QB:90}`, `{FLEX:1, SUPER_FLEX:1}` → 190, not 100) to pin the fill order; plus bench
  excluded, a roster short at a position scores the partial lineup, and equal-points tie-breaks
  deterministic.
- **Paired-difference stat** — a synthetic fixture where strategy A captures a constant edge over B every
  seed; assert the paired-diff CI excludes 0 and names A. A zero-edge fixture ⇒ "no separation".
- **σ-tuning** — a fixture where a known σ is best; assert `tune_sigma` recovers it as the argmax.
- **CLI smoke** — both modes run end-to-end on a fixture parquet + config and print a result.
- **League-driven** — run the harness under two different `roster_slots` configs (e.g. a SUPER_FLEX
  config and a skill-only config) on the same pool and confirm both score without code change — the
  guard against any hardcoded position count or slot assumption.

All gates per the project bar: `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check`.

## 5. Key decisions

- **5.1 Hero vs ADP field, not all-strategy round-robin** — a realistic draft room and the only setup
  where the survival probabilities are calibrated against actual ADP behavior (so σ-tuning means
  something). All-strategy rooms confound pool dynamics and have no ADP signal to tune against.
- **5.2 Optimal starting-lineup points as the metric** — fantasy is won by starters; total-roster or
  total-VORP over-credit bench hoarding. Reuses the existing slot↔position eligibility.
- **5.3 Seeded bot noise + paired bootstrap, not a single deterministic draft** — one draft is a point
  estimate that can't separate a real edge from luck and over-fits σ to one realization. Seeded noise +
  many paired seeds gives a CI; pairing on the shared field gives the test its power. Deterministic given
  seed, so fully reproducible.
- **5.4 League-driven, nothing hardcoded** — roster shape, team count, and ruleset all come from
  `LeagueConfig`; the harness never references a literal position count or PPR. Scoring rules are honored
  by consuming a VORP table built under that league's ruleset (a documented precondition the parquet
  can't self-verify).
- **5.5 Bots are pure noisy-ADP, not roster-constrained** — under a shared bench `eligible_positions`
  never drops a position until a team's roster is full, so a roster-eligibility filter would be a no-op;
  realism comes from ADP sparsity itself (consensus ADP already spaces positions like a real room).
  Opponent roster-construction modeling is an explicit non-goal. The pool holds only league-rostered
  positions, so a bot can never draft an un-rosterable player.
- **5.6 No new pandera schema** — the result is a handful of floats rendered by the CLI; a schema would be
  ceremony. Persisting the per-seed matrix earns a schema when a consumer needs it, not pre-emptively.
- **5.7 Simulate → score → compare split, with the auction seam documented** — `optimal_lineup_points`
  and the tournament/bootstrap layer are mechanism-agnostic; the future auction slice swaps only the
  draft-mechanism module (`simulation.py` → an auction nominate/bid loop) and reuses scoring + stats.
  Designing the split now is what makes the later effort "obviously a later effort" rather than a rewrite.

## 6. Open questions / future slices

- **Auction tournament** — the user-confirmed later effort; inherits §5.7's seam.
- **Hero-slot sweep** — average the hero across all draft slots for a slot-robust verdict (Q3 option C).
- **Conditional survival model** — refine `LogisticSurvival` to condition on "already survived to now";
  this harness then re-tunes/re-compares it behind the unchanged Protocol.
- **Persisted tournament history** — if we want riser/faller-style tracking of strategy performance across
  pool snapshots, that's when `TournamentResult` graduates to a stored schema.
