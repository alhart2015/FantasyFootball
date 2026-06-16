# Draft Board UX — Name Fix + Click-to-Pick + Best-Available Filter — Design

**Date:** 2026-06-16
**Status:** Design (pre-plan)
**Branch:** `feat/draft-board-ux`

## 1. Motivation

Three issues surfaced using the live draft board (`scripts/draft_board.py`) in co-pilot mode:

1. **Missing rookie names ("ADP suggests: —").** 60 of the 458 players in the consensus VORP pool (13%) are 2026 rookies with placeholder gsis IDs (`99-XXXXXXX`) that are absent from `id_map`. The board resolves names only via `id_map` (`LiveDraftSession.player_names`), so `name()` returns "—" for them — in the ADP suggestion, the best-available list (a phantom "— (204)"), and the board log. The names **already exist** in `consensus_projections.full_name` (with an `is_placeholder_gsis` flag); they are simply never carried into the VORP table or surfaced. The ADP logic is correct — `suggested_pick` rightly returns the rookie RB `99-8467088` (ADP 18.6); only its *name* is missing.

2. **Clunky pick recording.** The only way to record a pick is a top-of-page "Record a pick — search a player" box, separate from the recommendation display. The user wants to record picks by **clicking** — from the central recommender and from the right-hand best-available list — with the same flow for their own picks and opponents'.

3. **No position filter on best-available.** Best-available shows a fixed top-3 per position with no way to focus on one position or find a specific player.

## 2. Goals & non-goals

**Goals:**
- Surface real names for **every** pool player (including placeholder-gsis rookies) by carrying `full_name` into the VORP table — the draftable pool becomes self-describing.
- A **unified click-to-pick flow** used identically for your picks and opponents': click a player in the central recommender or the right best-available pane → a `Confirm pick: <name>` button appears → confirm records it and advances.
- The right **best-available pane becomes a searchable + position-filterable, click-selectable picker** — the "find and record any pick" surface (replacing the removed top search box).
- Remove the top-of-page search box.

**Non-goals (deliberate):**
- **No change to draft/strategy logic** — `recommendation`, `suggested_pick`, the engine math, and `BOARD_STRATEGIES` are untouched. Pure UI + name-data plumbing.
- **No change to the consensus ingest / `consensus_projections`** — `full_name` is already there; we only join it into the VORP table.
- **No board-log or roster redesign.** My Roster stays compact at the top of the right pane, above best-available.
- **No new persistence format** — autosave/resume unchanged (sessions store paths + picks; names re-resolve from the pool on load).
- **Not a strategy decision** — this is tool UX; the strategy investigation stays pure data-gathering.

## 3. The name fix — `full_name` in the VORP table

- **`VorpTableSchema`** (`schemas.py`) gains `full_name: Series[str] | None = pa.Field(nullable=True)` — **Optional + nullable**, mirroring `consensus_adp`. The weekly-source path (which has no names) omits it and still validates; the consensus path populates it. Existing consumers (cheat sheet, auction) are unaffected (`strict="filter"`, and the column is optional).
- **`generate_vorp_table.py`** — the `--source consensus` path already merges `consensus_adp` from the consensus frame (`adp = consensus[["gsis_id", "consensus_adp"]]; out_df = out_df.merge(adp, ...)`). Extend that projection to include `full_name`: `consensus[["gsis_id", "consensus_adp", "full_name"]]`. The weekly path is unchanged (no `full_name`).
- **Regenerate** `data/consensus_vorp_2026.parquet` via the generator (a fast table build; needs the consensus snapshot + id_map + league config, all present) so the live board picks up names immediately.
- **`LiveDraftSession.player_names`** (`live.py`) becomes **pool `full_name` overlaid on id_map names** — id_map first, then pool `full_name` (non-null) wins, so placeholder rookies resolve and every drafted player (all drawn from the pool) has a name. A pool player whose `full_name` is null falls back to id_map, then "—". This makes the board robust to id_map gaps for any consensus pool.
- **Decouple the `record_pick` my-pick guard from `player_names`.** `record_pick` currently rejects *my* picks of players absent from id_map via `gid not in self.player_names` (its comment asserts the invariant "absence from `player_names` == absent from id_map"). Broadening `player_names` to pool names **breaks that invariant**: a placeholder rookie would pass the guard, get appended, and then the *next* `build_draft_state` call would raise — my-roster position is resolved from id_map only (`state.py` `build_draft_state`, which raises when one of *my* picks is absent from id_map) — wedging every subsequent render and poisoning the autosave with the bad pick. **Fix:** the guard must test **id_map membership directly**, not the broadened display map. Add a cached `_id_map_ids: frozenset[str]` (a `cached_property` over `id_map["gsis_id"]`) and change the guard to `gid not in self._id_map_ids`. Behavior is then exactly preserved: opponents may be recorded drafting any undrafted pool player (incl. rookies); *my* picks still require id_map presence, raising the existing clear `ValueError` **before** the append (caught → Streamlit warning, no poison). Drafting a placeholder rookie to *my own* roster stays unsupported (documented limitation — see §6/§9); this is strictly no regression, since rookies were entirely unpickable before this change.
- **`cli.py` `format_table`** prefers the pool's `full_name` when present (id_map fallback), keeping the headless CLI consistent with the board. Small, same name-resolution rule.

## 4. The pick flow — unified click-to-confirm

- **Shared pending selection (single source of truth).** `st.session_state["pending_pick"]` holds **one** selected gsis (or absent). Both selectable panes feed this single value; a prominent **`Confirm pick: <name>`** button (rendered once, for `pending_pick`) records it (`record_pick` → autosave → `st.rerun()`); a **`✕ clear`** deselects.
- **Selection mechanism + dual-pane resolution.** Click-selection uses `st.dataframe(..., selection_mode="single-row", on_select=…, key=f"sel_{pane}_{current_pick}")` (native row selection, Streamlit ≥1.35; running 1.58). Because each `st.dataframe` keeps its selection across reruns, **two panes can hold a selection at once** — so the rule is **most-recent click wins**: each pane writes its selected gsis into `pending_pick` via its `on_select` handler (an `on_select` **callback** that sets `pending_pick` from that pane's selection if the Streamlit API accepts a callable; otherwise, on each rerun compare each pane's current selection to its prior value stored in session_state and let the pane that *changed* set `pending_pick`). The selected row index maps to that frame's `gsis_id`.
- **Reset per pick.** Both selectable dataframes are **keyed by `current_pick`** (as the old `_search_box` was), so after a recorded pick the widgets are fresh and no stale selection lingers; `pending_pick` is cleared on a successful record.
- **The opponent ADP shortcut is separate.** In co-pilot on an opponent's turn, the ADP suggestion renders as its **own direct one-click `Confirm pick: <name>` button** (records immediately) — it does **not** route through `pending_pick`. The select→confirm path (best-available) coexists for "they took someone else." (Two distinct confirm buttons may appear; that's intended.)
- **Center — the recommender** (`_recommend_col`):
  - **Your pick** (or mock, your turn): the ranked recommendation frame, click-selectable → sets `pending_pick`.
  - **Opponent** (co-pilot, not your turn): keep the **one-click ADP-suggestion shortcut** (now with a real name) as the common-case fast path; the recommender shows that suggestion. Recording a *different* opponent pick is done in the best-available pane.
- **Right — best available** (`_best_available_col`, replacing the best-available block in `_roster_col`):
  - A **position `selectbox`**: `All / QB / RB / WR / TE`. `All` → top-N available by VORP across positions (capped, e.g. 60, for render perf); a specific position → that position's available players (deeper).
  - A **search `text_input`** filtering the list by case-insensitive name substring.
  - The resulting **click-selectable `st.dataframe`** (name, position, VORP, ADP) → selecting a row sets `pending_pick`. This is the "find and record any pick" surface for both your picks and opponents'.
  - My Roster stays compact above this in the right column.
- **`_search_box` removed** (function + its call in `main()`).

## 5. Controller support — `available_for_pick`

Add `LiveDraftSession.available_for_pick(position: Position | None, query: str, top: int) -> pd.DataFrame` to `live.py`: starts from `available_pool()`, filters to `position` (when not None), filters to rows whose resolved name contains `query` (case-insensitive; empty query = no filter), attaches names (pool `full_name` → id_map), sorts by `vorp` descending, **then** returns the top `top`. The `top` cap is applied **last** — strictly after the position + query filters and the VORP sort — so a position selection or a search query can reach a deep player that an unfiltered cross-position top-N would otherwise hide. Pure + testable; the board renders it. The name-contains filter uses the same name source as `player_names` so search matches what's displayed (incl. rookies).

## 6. Edge cases / failure modes

- **Weekly-source pool (no `full_name`)** → schema Optional handles the absent column; `player_names` falls back entirely to id_map (status quo for that path).
- **Pool player with null `full_name` not in id_map** → resolves to "—" (unchanged); the fix only guarantees names where `consensus_projections` had one (all 60 rookies do).
- **Selected player gets drafted before confirm** (stale pending) → `record_pick` raises `ValueError` (already drafted); caught, surfaced as a warning, pending cleared.
- **Search yields no matches** → empty selectable frame, no crash; confirm button absent.
- **`pending_pick` referencing a drafted/ineligible gsis** → guarded by `record_pick`.
- **Selecting a placeholder rookie for *my own* pick** → `record_pick`'s id_map guard raises a clear `ValueError` (rookie absent from id_map → no resolvable roster position) **before** appending; caught and surfaced as a Streamlit warning, the pick is not recorded (no poison). The same rookie *is* recordable as an opponent's pick. Resolving my-roster position from the pool's `position` column (to make rookies fully draftable to my own team) is deferred — see §9.
- **Selection cleared on rerun** so a confirmed pick doesn't linger as a phantom selection.
- **`All` with a large pool** → capped top-N by VORP to bound render cost; search + position filter narrow it.

## 7. Testing (TDD)

- **Schema:** `VorpTableSchema` validates a frame **with** `full_name` and one **without** (Optional). Existing VORP-consumer tests (cheat sheet / auction) stay green.
- **Generator:** the consensus path emits `full_name`. Cheap unit check (no full generator/store round-trip): build a tiny consensus frame (a few `gsis_id` / `full_name` / `consensus_adp` rows) plus a matching VORP `out_df`, run the consensus merge step, and assert the result carries a `full_name` column with the expected values; assert the weekly path's output omits `full_name`.
- **`player_names`:** a pool row with a `full_name` whose gsis is **absent from id_map** resolves to the real name (not "—"); a pool row with null `full_name` but present in id_map resolves via id_map; pool name wins over a differing id_map name.
- **`record_pick` guard (regression for the name fix):** with `player_names` broadened to pool names, recording an off-id_map placeholder rookie as **an opponent's pick** succeeds (it joins `picks`, leaves the pool, and a subsequent `state()` does not raise); recording the same rookie as **my own pick** raises `ValueError` and does **not** append (the session stays usable — a following `state()`/render does not raise). This pins the id_map-membership guard that the broadened display map would otherwise defeat.
- **`available_for_pick`:** position filter; `None` = all; case-insensitive name `query` filter; sort by VORP; `top` cap; names attached.
- **Board (`AppTest` smoke, `tests/test_scripts/test_draft_board_smoke.py`):** extend — the top search box is gone; the best-available pane renders the position `selectbox` + search + a selectable frame; selecting a row + clicking confirm records a pick (picks count increments); opponent-turn ADP shortcut still records. Existing smoke assertions stay green.
- **Regression:** full `tests/test_draft` + the draft-board smoke green; `mypy src tests` + `ruff` clean.
- **Manual verification:** regenerate the parquet, launch the board, confirm `99-8467088` (and the other rookies) show real names in the ADP suggestion + best-available, and that click-to-confirm records picks from both panes.

## 8. Phasing

- **Phase 1 — name fix:** `VorpTableSchema.full_name` + generator merge + regenerate the parquet + `player_names`/`cli` name resolution + **the `record_pick` id_map-membership guard (`_id_map_ids`)** that must land with the broadened `player_names`. Fixes the "—" bug on its own without regressing pick recording.
- **Phase 2 — controller:** `available_for_pick` + tests.
- **Phase 3 — board UI:** remove the search box; clickable recommender; searchable/filterable click-selectable best-available; shared pending→confirm flow; AppTest smoke.
- **Phase 4 — verify + sync:** run the app to confirm names + click-to-pick on real data; update PM/TODO.

## 9. Open questions / future refinements

None blocking. Deferred: **resolving my-roster position from the pool's `position` column** (via `build_draft_state`) so placeholder-gsis rookies become draftable to *my own* roster, not just recordable as opponents' picks — the one functional limitation this design leaves open (see §6); a full keyboard-driven pick entry; multi-select/undo from the board; surfacing `is_placeholder_gsis` (rookie marker) in the UI; carrying `full_name` through the weekly path; per-pane sort controls. My Roster could move to the left column if the right pane should be best-available only (left as a one-line tweak).
