# Auction board — minimal (live-draft) mode

**Status:** design · 2026-08-12 · branch `feat/auction-minimal-mode`

## Problem

The full auction board (PR #136) is a three-column analyst view: sold log, bid panel,
nomination panel, 40-row bid board, roster table, budget table, projected eval. That is the
right surface for studying an auction. It is the wrong surface for *running* one.

During a live draft the operator is doing something else — listening to the room, tracking a
clock, holding a paddle. The board has seconds of attention at a time, and most of what it
shows is already on the Yahoo draft UI in front of them (draft history, rosters, budgets,
who's on the clock). Duplicating that costs screen space and reading time and returns nothing.

Two things also make the full board awkward mid-draft:

- **Finding the nominated player.** The flow is position filter → search box → scan a 40-row
  table → click the row. That is four interactions to answer "someone just nominated Bijan".
- **Team identity.** Seats render as `Team 6` / `Team 7`. Recording a sale means translating
  a human name into a seat number under time pressure, which is exactly when a mis-record
  happens — and a mis-recorded winner corrupts the purchase log every recommendation derives
  from.

## Goal

A mode that fits in a small window and answers three questions with minimal interaction:

1. **Who was just nominated?** — one text box, type-ahead, no filters to set first.
2. **How much do I bid?** — one number, large enough to read at a glance.
3. **Who bought him, for how much?** — two fields and a confirm, using real team names.

Plus a compact **who to nominate** section, since that is a decision the board makes better
than the operator does and Yahoo cannot show it.

## Non-goals

Everything Yahoo already displays: draft history / sold log, full rosters, every team's
budget, the 40-row bid board, the projected-season eval. These stay in the full board, which
is unchanged. Minimal mode is a second view over the same `LiveAuctionSession`, not a fork of
it — no duplicated state, no second autosave format.

## Design

### Mode switch

A sidebar radio: **Full board** (default, unchanged) / **Minimal (live draft)**. Both render
the same session object, so the operator can flip mid-draft — e.g. to check a budget table —
and flip back without losing anything.

### Step 1: team names

On entering minimal mode with no names set, the board shows a naming form before anything
else: one text input per seat, prefilled `Team N`, the operator's own seat marked. Saving
writes `session.team_names` and autosaves.

This is a gate rather than an optional setting because it is cheap once and expensive never:
every later confirm button reads `Player → <name> for $N`, and that sentence is the only
thing standing between a misheard winner and a corrupted log. A resumed session already has
names, so the gate does not reappear.

An explicit "use Team 1…N" escape exists so the gate can never strand a session.

### Step 2: the lot

- **Nominated player** — a single `st.selectbox` over every available player, `index=None`,
  placeholder `Type a player name…`. Streamlit's selectbox filters as you type, which is the
  autocomplete; options are labelled `Name (POS)` so two players sharing a surname are
  distinguishable. No position filter, no separate search box.
- **The number** — `# 🔨 BID UP TO $X` as the largest thing on screen, with the verdict
  underneath (`i_want` / `uncontested` from `BidAdvice`, already shared with the full board).
  One caption line carries worth-to-us, room price, and best rival ceiling.
- **Record the sale** — winner selectbox (real names, no default, seats with no open slot
  excluded) and price input, then a confirm button spelling out the whole sentence. Same
  safety properties the review put on the full board: widgets keyed to the staged player and
  the chosen winner, so a price typed for one lot cannot survive onto the next.
- **Undo** — one button. Not draft history; the correction path for the mis-record the
  confirm sentence is designed to prevent.

### Step 3: who to nominate

The engine's suggestion by name, plus a short table (`top=5`) of candidates with what each
costs. Collapsed by default when it is not the operator's nomination, expanded when it is.

### Footer

One line: money left, open slots, lots sold. Enough context for the bid; not a budget table.

## What this does not change

`LiveAuctionSession` needs no new behaviour — `team_names` is already a persisted field, and
`advise` / `nomination_board` / `record_purchase` / `undo` already expose everything the view
needs. The only session-layer addition is a helper for the autocomplete option list, which is
a view concern expressed over `available_pool()`.

## Testing

AppTest smokes, in the same file as the existing board smokes (which, as of PR #136, actually
run):

- the naming gate appears with no names, and not after they are set;
- saving names changes what `team_label` returns;
- selecting a player renders a bid number;
- recording a sale through the minimal flow lands in `purchases` with the right seat;
- the sold log / budget table / bid board are *absent* in minimal mode (the point of it).
