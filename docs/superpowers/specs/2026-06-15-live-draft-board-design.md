# Live Draft Board (Draft Assistant Slice 3) — Design

**Date:** 2026-06-15
**Status:** Approved (brainstorming) → spec
**Sub-project:** Draft Hub / Live Draft Assistant (the final UI slice)

## 1. Problem & Context

The Draft Assistant engine is built and validated headless: a consensus VORP pool, a
`DraftStrategy` protocol (`now_or_never`, `raw_vorp`, `season_value`,
`season_value_timing`), an ADP-bot opponent (`bot_pick`), a risk-aware season metric,
and a strategy tournament. Today the only way to drive it in a real draft is to
hand-edit a JSON draft-state file and re-run `scripts/draft_assistant.py` after every
pick. That is unusable at a live draft's pace.

This slice builds the interactive **live draft board**: a Streamlit app that records
picks as they happen and shows the ranked recommendation for the next pick, the user's
roster, and board state — in real time. It serves two modes that share one board:

- **Co-pilot (live):** the user logs every real pick (theirs and opponents'). On an
  opponent's turn the board pre-highlights the ADP-likely player for one-click confirm;
  on the user's turn it shows the full recommendation.
- **Mock (practice):** opponents are auto-drafted by the same ADP bot, so the user can
  rehearse a draft against the field and see the resulting roster scored.

The engine's `DraftState` derives *which picks are the user's* from snake position via
`id_map`, so the board must record **every** pick, in order — there is no shortcut that
skips opponents' picks while keeping recommendations correct.

## 2. Goals

1. A Streamlit board (layout: three-column "command center") usable on a live draft
   clock — recommendations, pick log, and roster all visible at once.
2. Manual pick entry with ADP smart-assist (one-click confirm of the bot's suggestion
   for opponents); full search-and-record fallback for any pick.
3. Mock mode: bot-driven opponents, "advance to my pick", end-of-draft roster scorecard.
4. The four production strategies selectable live (instant `now_or_never`/`raw_vorp`;
   MC `season_value`/`season_value_timing` with a spinner + result caching).
   `season_value_var` is in `STRATEGY_KEYS` but is **deliberately excluded** from the
   board's dropdown — its A/B showed no draft benefit (memory
   `risk-aware-season-value-no-draft-benefit`); exposing it would mislead. (See §4.5.)
5. Crash-recovery: autosave after every pick; resume an in-progress session on launch.
6. **All draft logic in a pure, tested controller in `src/`; the Streamlit file is a
   thin view.** Gates (`pytest`, `mypy --strict`, `ruff`) apply to the controller.

## 3. Non-Goals

- **No live ESPN/Sleeper draft-API sync** (undocumented, brittle, auth-coupled).
  Manual entry + bot smart-assist only. (Documented future seam.)
- **No auction or keeper drafts** — snake only (matches the engine's `pick_timing`).
- **The UI does not generate the VORP table.** It *consumes* an existing consensus VORP
  parquet from `scripts/generate_vorp_table.py`. Separation preserved.
- **No new or modified `DraftStrategy`** — the board drives existing strategies as-is.
- No mobile / responsive design tuning; target is a desktop browser.

## 4. Chosen Approach — testable controller + thin view

Mirror the repo's established pattern (testable core in `src/`, thin wrapper in
`scripts/`; cf. `cli.py` ↔ `scripts/draft_assistant.py`):

- **`src/projections/draft/assistant/live.py`** — `LiveDraftSession`, a pure,
  fully-typed controller (no Streamlit import). Holds the mutable draft truth and
  delegates every decision to existing engine functions. Unit-tested under all gates.
- **`scripts/draft_board.py`** — the Streamlit entry (`streamlit run scripts/draft_board.py`).
  Thin: owns `st.session_state`, renders the three columns and sidebar, calls the
  controller. Any non-trivial *display-prep* helpers that don't need Streamlit live in
  `live.py` so they stay testable.
- **Dependency:** add `streamlit` under a `[project.optional-dependencies] ui` extra in
  `pyproject.toml` (`pip install -e ".[ui]"`), keeping the core install lean.

### 4.1 `build_draft_state` refactor (reuse, don't duplicate)

`DraftState` is immutable and `load_draft_state` builds it from a *file*. The controller
must rebuild it after every pick from *in-memory* picks. Extract the pure in-memory half:

```python
def build_draft_state(
    picks: Sequence[GsisId], my_slot: int, league: LeagueConfig, id_map: pd.DataFrame
) -> DraftState: ...
```

`load_draft_state` keeps its file/JSON parsing and validation, then **delegates** to
`build_draft_state` for the construction — same validations (my_slot range, duplicate
pick, my-pick-absent-from-id_map), so behavior is byte-identical. Pinned by an
equivalence test (`load_draft_state(file)` == `build_draft_state(parsed inputs)`).

### 4.2 `LiveDraftSession` controller API

State (constructor args): `league: LeagueConfig`, `my_slot: int`, `id_map: pd.DataFrame`
(validated `IdMapSchema`; position + name source), `pool: pd.DataFrame` (validated
`VorpTableSchema`; the rankable universe), `strategy: DraftStrategy`,
`mode: Literal["copilot", "mock"]`, `adp_jitter: float`, `base_seed: int`. Mutable:
`picks: list[GsisId]`.

Methods / properties:

- `state() -> DraftState` — `build_draft_state(self.picks, …)`; memoized per `picks` tuple.
- `current_pick: int`, `round_and_slot() -> tuple[int, int]`, `on_clock_slot: int`
  (`slot_for(current_pick, n_teams)`), `is_my_pick: bool`, `next_pick_number: int | None`
  (wraps the imported `my_next_pick(...)` function — the property is renamed to avoid
  shadowing it), `is_complete: bool` (`len(picks) >= n_teams * roster_size`).
- `available_pool() -> pd.DataFrame` — `pool` minus `state().drafted_ids`.
- `record_pick(gsis_id) -> None` — reject if already drafted or absent from `id_map`
  (can't resolve a position → can't keep roster accounting honest); else append.
- `undo() -> GsisId | None` — pop the last pick (mis-clicks happen).
- `recommendation() -> pd.DataFrame` — `self.strategy.recommend(state, pool, league)`
  (`RecommendationSchema`). Empty frame when the draft is complete.
- `suggested_pick() -> GsisId | None` — `bot_pick(available, rng, adp_jitter=…)` with a
  **deterministic** rng seeded from `(base_seed, current_pick)` so the suggestion is
  stable across Streamlit reruns and reproducible in mock mode; `None` if pool empty.
- `my_roster_view() -> RosterView` — a small dataclass (not a bare frame, to fix its
  shape): `filled: pd.DataFrame` with columns `[slot (RosterSlot), gsis_id, full_name,
  position]` (one row per filled starting/bench slot via the shared `roster_eligibility`
  greedy allocation) and `open_slots: dict[RosterSlot, int]` (unfilled starting slots).
- `best_available_by_position(top: int) -> dict[Position, pd.DataFrame]` — top-N
  available per `Position` by `vorp` (tier-cliff visibility).
- `mock_advance_to_my_pick() -> list[GsisId]` — **(Phase 3)** mock only: repeatedly
  `record_pick(suggested_pick())` until `is_my_pick` or `is_complete`; returns the bot
  picks made. Raises in co-pilot mode (guard).
- `roster_scorecard() -> float` — **(Phase 3)** `optimal_lineup_points(my drafted rows
  joined to `pool` on `gsis_id` for `season_mean_fpts`, league.roster_slots)`; the mock
  end-of-draft "how did I do" number (`VorpTableSchema` carries `season_mean_fpts`,
  verified). Season-value MC scorecard is an optional later add via `SeasonValuer`.

The two mock methods above are part of the §4.2 surface for completeness but land in
**Phase 3** with the mock UI, not in the Phase-1 controller foundation (see §8).

### 4.3 The view (layout B — three-column command center)

Sticky **status bar**: `Pick R.PP · on the clock: <TEAM k | YOU> · your next pick in N`,
with the search/record box. Three columns:

- **Left — Board / pick log:** every pick by round; the user's seat highlighted; current
  pick marked.
- **Center (hero) — Recommendations:** the `RecommendationSchema` table (player name,
  `position`, `vorp`, `consensus_adp`, `p_available_next`, `score`, ★ `fills_starting_slot`),
  with the strategy dropdown above (the four production strategies from §2 goal 4 — not
  `season_value_var`). On an opponent's turn in co-pilot mode this column shows the
  smart-assist suggestion with a **Confirm** button (+ search override). MC strategies
  render inside `st.spinner`; results are cached on **every result-affecting parameter** —
  `(picks tuple, strategy_name, n_sims, sigma)` — so a `sigma` change for
  `season_value_timing` at the same `n_sims` is not served a stale cached result.
- **Right — My Roster + Best-available-by-position:** filled slots, open needs, best
  remaining at each position.

**Sidebar — Setup:** mode toggle (Co-pilot / Mock); `my_slot`; data inputs (consensus
VORP parquet, `id_map` parquet, league-config JSON — sensible defaults pre-filled);
strategy dropdown; `n_sims` (MC strategies); `adp_jitter` slider (mock field / suggestion
noise); New-draft / Resume controls.

**Pick entry:** the search box filters `id_map` by `full_name`; matches render as
clickable rows that **show `position` and `team` alongside the name** (so the user
disambiguates same-name players — `id_map` is not unique on name) and call `record_pick`.
Mock mode adds an **"Advance to my pick"** button.

### 4.4 Strategy construction seam

Both the sidebar dropdown and the resume path turn a `strategy_name` (+ live params) into
a `DraftStrategy`. To avoid duplicating that logic (and the existing inline `cli.py`
version covers only the analytic strategies), name one shared builder in `live.py`:

```python
def build_session_strategy(
    name: str, *, league: LeagueConfig, sigma: float | None,
    availability: PlayerAvailability | None, n_sims: int, base_seed: int,
) -> DraftStrategy: ...
```

It maps each production name to its constructor (`now_or_never`/`season_value_timing` →
`LogisticSurvival(default_sigma(n_teams) if sigma is None else sigma)`;
`season_value*` → require a non-null `availability`, fail loud otherwise). The existing
`cli._build_strategy` is refactored to delegate to it (no behavior change). Both the view
and `LiveDraftSession.load` call `build_session_strategy` — there is no `strategy_factory`
parameter.

### 4.5 Persistence / crash-recovery

The dev box has a history of mid-run BSODs (see memory `h2h-backtest-native-crash`); a
live draft cannot lose state. After **every** pick the session autosaves to a JSON file
under `data/draft_sessions/<timestamp>.json`. **This path is not currently gitignored** —
Phase 4 adds `data/draft_sessions/` to `.gitignore` (sessions are per-draft scratch, not
artifacts). Persistence API:

```python
def to_state_dict(self) -> dict   # CLI-compatible superset
def save(self, path: Path) -> None
@classmethod
def load(cls, path, *, id_map, pool, data_root=Path("data")) -> LiveDraftSession
    # rebuilds strategy via build_session_strategy; MC strategies load availability
    # from `data_root` + the saved `season`
```

`to_state_dict` is a **superset** of the CLI draft-state format — it includes the
required `{league_config, my_slot, picks}` (so `load_draft_state` and
`scripts/draft_assistant.py` read it unchanged) plus `{mode, adp_jitter, strategy_name,
n_sims, sigma, season, vorp_table, id_map}` for full one-click resume. `load` reads those,
re-validates the data inputs, and reconstructs the strategy via `build_session_strategy`
(§4.4) — loading `availability` from `data_root`/`season` when the saved strategy is an
MC one. On launch the app detects the newest autosave and offers **Resume** vs **New draft**.

## 5. Data Flow

1. Sidebar config → load + validate `id_map` (`IdMapSchema`), `pool` (`VorpTableSchema`),
   `league` (`LeagueConfig.model_validate_json`) → construct `LiveDraftSession` into
   `st.session_state`.
2. Each interaction reruns the script; the view reads the single `LiveDraftSession` from
   `st.session_state` (never module globals) and renders from it.
3. Record a pick → `record_pick` → autosave → rerun re-renders status, board,
   recommendation, roster.
4. Co-pilot opponent turn → `suggested_pick` shown; Confirm records it. Mock opponent turn
   → "Advance to my pick" loops `record_pick(suggested_pick())`.
5. Draft complete → recommendation empty; show `roster_scorecard`.

## 6. Edge Cases & Failure Modes

- **Missing/invalid data file** → fail loud with a clear sidebar message; reuse existing
  schema validation (`IdMapSchema`, `VorpTableSchema`, `load_draft_state` checks). The
  board does not render until inputs validate.
- **`my_slot` out of `[1, n_teams]`** → reject at construction (existing check).
- **Record an already-drafted player** → no-op + visible message (don't corrupt order).
- **Record a player absent from `id_map`** → reject (can't resolve position). Search is
  over `id_map` (universal), so the user can only pick resolvable players; off-pool
  players (e.g. K/DST not in a skill-only VORP table) are recordable as long as `id_map`
  knows them — they simply won't appear in the rankable recommendation.
- **MC strategy latency** → `st.spinner` + cache per `(picks, strategy, n_sims)` so plain
  reruns don't recompute.
- **`season_value*` availability data missing** (`weekly_stats`/`schedules` for the
  season) → `load_store_availability` fails loud; surface as a sidebar error, keep the
  board on a fast strategy rather than crashing.
- **2026 `schedules` not ingested** → byes degrade to none with the existing warning
  (documented data dep; ingest before the real draft).
- **Draft complete** → `record_pick`/recommendation guarded; scorecard shown.
- **Streamlit reruns** → all state in `st.session_state`; controller is the single source
  of truth; no reliance on module-level mutable state or a stored live RNG (seeds are
  derived per pick).
- **`mock_advance_to_my_pick` in co-pilot mode** → guarded (raises); the view only shows
  the button in mock mode.

## 7. Testing Expectations

- **Controller (`tests/test_draft/`)** — full coverage:
  - `record_pick` / `undo` ordering; reject already-drafted + absent-from-id_map.
  - **Snake ownership**: `my_roster_view` / `state().my_pick_ids` match `load_draft_state`
    on the same picks.
  - `recommendation()` passthrough via a fake `DraftStrategy`.
  - `suggested_pick()` determinism (stable across calls for one board state) + pool
    exhaustion (`None`).
  - `available_pool`, `best_available_by_position` correctness.
  - `mock_advance_to_my_pick` stops exactly at the user's pick / on completion; raises in
    co-pilot mode.
  - `is_complete`, `round_and_slot`, `on_clock_slot`, `my_next_pick` boundaries.
  - `roster_scorecard` equals `optimal_lineup_points` on the user's drafted rows.
  - Persistence round-trip: `to_state_dict` → `save` → `load` reproduces picks/mode;
    asserts the CLI-required keys are present and `load_draft_state` accepts the file.
- **`build_draft_state`** — equivalence test vs `load_draft_state` on a temp file.
- **`build_session_strategy`** — returns the right concrete type per name; `season_value*`
  with `availability=None` fails loud; `cli._build_strategy` still produces an identical
  `now_or_never`/`raw_vorp` to before the refactor.
- **Optional Streamlit smoke** — `st.testing.v1.AppTest` driving setup + 2-3 picks,
  asserting no exception and key widgets present; guarded by `pytest.importorskip("streamlit")`.
- **Gates:** `pytest -k "draft"` (relevant subset), `mypy src tests`, `ruff check src tests`,
  `ruff format --check src tests`.

## 8. Phasing

1. **Controller foundation** — `build_draft_state` refactor (+ equivalence test);
   `build_session_strategy` seam (+ `cli._build_strategy` delegating to it, no behavior
   change); `LiveDraftSession` core (state/record/undo/available/recommendation/
   suggested_pick/ownership/`my_roster_view`/`best_available_by_position`) + tests.
2. **Streamlit view, co-pilot** — layout B, setup sidebar, status bar, search-and-record,
   three columns, smart-assist confirm; strategy dropdown + MC spinner/cache;
   `[ui]` extra in `pyproject`.
3. **Mock mode** — bot auto-advance ("Advance to my pick"), `adp_jitter` control,
   end-of-draft `roster_scorecard` + `mock_advance_to_my_pick`.
4. **Persistence & polish** — autosave/resume (`to_state_dict`/`save`/`load`);
   add `data/draft_sessions/` to `.gitignore`; optional `AppTest` smoke; docs
   (`CONTRIBUTING.md` run command) + `project_management.md` / `TODO.md` update.

Each phase ≤ 5 files, gated independently (per the project's phased-execution rule).

## 9. Future Seams (out of scope, recorded)

- ESPN/Sleeper live-draft auto-sync.
- Auction / keeper draft modes (the engine's simulate→score split already isolates the
  draft mechanism).
- Season-value MC scorecard in mock mode (latency-tolerant; `SeasonValuer` exists).
- Tier-cliff visualization and an undo/redo stack beyond single-step.
