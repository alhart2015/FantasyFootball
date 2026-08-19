# Pick'em Hub — Design

**Date:** 2026-08-16
**Status:** Proposed
**Sub-project:** `pickem` (new)

## Problem

I play in a straight-up NFL pick'em league with a twist:

- Pick the **outright winner** of every game on the slate. A -13 favorite only has to win, not cover.
- **At least three of your picks each week must be the underdog** according to the spread on the
  organizer's sheet.

Prizes: a weekly prize for most correct picks, and a much larger season-long prize for most
correct picks overall. League size is 10–15 people.

The organizer distributes a Google Sheet (also emailed) on **Tuesday**. Picks are due **before
Thursday kickoff**. Where he sources his spreads from is unknown.

## The edge

Two observations drive the whole design.

**1. The organizer's sheet is stale by the time picks are due.** It is set Tuesday; picks lock
Thursday. Wednesday is the first injury report of the week, and it is where the largest
in-week line moves happen (a QB ruled out can move a line 5+ points). The sheet cannot know
about them.

**2. The sheet's spread and the true win probability are two different things, from two
different sources.** This is the core insight of the design:

- **The organizer's spread decides _who counts as the underdog_.** That is the constraint we
  must satisfy. Nothing more.
- **Consensus market odds decide _who is likely to win_.** That is the objective we maximize.

Never let the organizer's spread touch a probability. Keeping these two strictly separate is
what makes the tool correct, and conflating them is the single most likely source of a subtle,
plausible-looking bug.

The best case this produces: the sheet lists a team as a dog, but by Thursday the market has
them as the *favorite*. That pick satisfies the three-underdog constraint at **zero cost** —
strictly better than any other dog on the board. We flag these loudly as **free dogs**.

## Non-goals (this iteration)

- **Beating the market.** We treat consensus odds as truth. Using the repo's own projections to
  disagree with Vegas is a much harder bar than beating a two-day-stale sheet, and is deferred
  until the market-only version is running and measurable.
- **Game-theoretic / contrarian play.** With 10–15 entrants and a season-long prize as the
  larger pot, maximizing expected correct picks is very close to optimal. Deliberately adding
  variance to win a single week trades away season equity. Revisit only if far behind late.
- **Automated Google Sheets read/write.** Picks are entered into the organizer's sheet by hand.
  Sheet input is a small CSV the user fills in.

## Data

### Source

`nflreadpy.load_schedules` — already ingested by `src/projections/ingest/schedules.py` into
`data/raw/schedules`. It already carries `spread_line`, `total_line`, `home_moneyline`,
`away_moneyline`. Verified: it also publishes lines for **upcoming** games (2026 weeks 1–4 are
populated as of 2026-08-16), so it doubles as our consensus feed with no new dependency.

**Open risk — refresh cadence.** Our entire edge is freshness. If this feed only updates
weekly, its lines are as stale as the organizer's and the edge evaporates. This is unknowable
until the season starts. Mitigation: the consensus fetch is isolated behind
`slate.build_slate(sheet, schedules)`, which takes an already-loaded schedules frame. Swapping
in a paid odds API means producing that same frame from a different source — no other module
changes. **Week 1 action: verify the feed moves between Tuesday and Thursday.**

### Verified data facts

Checked against `nflreadpy` for 2010–2025 regular season (4,175 games):

| Fact | Value | Consequence |
| --- | --- | --- |
| `spread_line` nulls | 0 | — |
| `home_moneyline` nulls | 1 (2017) | Devigged moneyline is a reliable primary source |
| Rows with a spread but no moneyline | 0 (incl. 2026 upcoming) | **No spread→probability fallback model is needed** |
| `result != home_score - away_score` | 0 rows | Store the two scores; derive the margin |
| Ties (`result == 0`) | 13 (~0.8/season) | Must be handled explicitly, not ignored |

### Sign conventions — read carefully

Two conventions are in play and they are opposites. Getting this wrong silently inverts every
pick, so it is stated once here and enforced at one place in code.

- **`nflreadpy.spread_line`:** *positive means the home team is favored.* (`KC` home vs `BAL`,
  `spread_line = +3.0`, `home_moneyline = -148` → KC favored by 3.)
- **Standard betting convention, used everywhere in our code:** *favorite negative, dog
  positive,* from a named team's perspective. `home_spread = -3.5` means home favored by 3.5.

Therefore `home_spread = -spread_line`. This matches the existing convention in
`src/projections/features/_shared.py` (`home["spread"] = -home["spread_line"]`); we reuse it
rather than inventing a third.

The organizer's sheet is entered in **standard convention** (`home_spread`), because that is
how a human reads a betting line.

## Architecture

New sub-package `src/projections/pickem/`, following the existing sub-package shape
(`draft/`, `dfs/`, `consensus/`).

```
sheet.csv (hand-entered)  ──┐
                            ├── slate.build_slate ──> optimize.choose_picks ──> picks
data/raw/schedules ─────────┘         │                                          │
   (consensus lines)                  │                                          │
                            probability.add_win_probs                   grade.grade_picks
                              (devigged moneylines)                              │
                                                                    data/raw/schedules (scores)
```

| Module | Responsibility |
| --- | --- |
| `probability.py` | American odds → implied probability; devig a two-way market; attach win probabilities to a schedules frame |
| `sheet.py` | Read/validate the organizer's CSV; write a pre-filled template |
| `slate.py` | Join sheet ↔ consensus into one per-game view; label dog/favorite; compute line movement |
| `optimize.py` | Choose picks subject to the ≥3-underdog constraint |
| `grade.py` | Score picks against final results |
| `store.py` | The only place that knows the pick'em partition layout |

### Probability

American odds → implied probability:

- Negative (favorite): `p = -odds / (-odds + 100)`
- Positive (dog): `p = 100 / (odds + 100)`

The two sides sum to more than 1 — that excess is the bookmaker's cut (the "vig"). We remove it
by **proportional normalization**: `p_home_fair = p_home_raw / (p_home_raw + p_away_raw)`.

This is the standard default for a two-way market. More refined methods exist (Shin, power) and
mainly matter for heavy favorites, where proportional devigging slightly overstates the
favorite. Noted as a possible future refinement; not worth the complexity at this stage.

A missing moneyline raises rather than silently falling back — per the data audit it should
never happen, and a wrong pick is worse than a loud failure.

### The optimizer

**Objective:** maximize expected correct picks = `Σ P(chosen team wins)`.
**Constraint:** at least `min_dogs` (default 3) chosen teams are underdogs *per the sheet*.

Algorithm:

1. For each game, the unconstrained best pick is `argmax(home_win_prob, away_win_prob)`.
2. Count how many of those are already sheet-dogs → `k`. (Free dogs land here automatically.)
3. If `k >= min_dogs`, done — no compromise needed.
4. Otherwise, for each game where we currently pick the sheet-favorite, compute
   `switch_cost = P(favorite) - P(dog)`, which is `>= 0`.
5. Switch the `min_dogs - k` games with the **smallest** switch cost.

**Why this is optimal, not just reasonable:** picks are independent across games, and the
constraint is a simple count. Deviating from the per-game argmax can only ever mean taking the
dog (there is no third option), so the total cost of any feasible solution is the sum of the
switch costs of the games it deviates on. Satisfying a count constraint at minimum total cost
means taking the cheapest deviations — a greedy choice is exactly optimal here. No search
needed.

Ties in switch cost are broken by `game_id` so runs are reproducible.

Edge cases:

- **Sheet spread of exactly 0** (a true pick'em): the game has no underdog. It is excluded from
  dog eligibility and can never satisfy the constraint.
- **Fewer eligible dog games than `min_dogs`:** raise, with a message naming the shortfall.
  Cannot happen on a real NFL slate but is cheap to guard.
- **A 50/50 game:** switch cost is 0 — a free dog slot in all but name.

### Grading

Join picks to final scores on `game_id`.

- `winner` = home if `home_score > away_score`, away if less, **NA on a tie**.
- `correct` = `pick == winner`.
- **Ties count as incorrect.** You picked a team to win and it did not. This is the common pool
  rule; it is stated explicitly because the alternative (a push) is also defensible, and 13
  games since 2010 are affected.
- **Unplayed games** are distinguished from ties by `correct` being **NA** rather than `False`.

## Schema changes

### `SchedulesSchema` — extended

Add three nullable columns (nullable because upcoming games have no score yet):

| Column | Type | Note |
| --- | --- | --- |
| `home_score` | `Int64`, nullable | |
| `away_score` | `Int64`, nullable | |
| `game_type` | `str`, nullable | `REG`/`WC`/... — lets analysis restrict to regular season |

`result` is deliberately **not** stored: it is exactly `home_score - away_score` in every one of
4,175 checked games, and storing a derived column invites the two drifting apart.

**Migration:** existing `data/raw/schedules` partitions predate these columns. `refresh_schedules`
is idempotent, so re-running it per season backfills them. This is a required one-time step and
is called out in the PR test plan.

### New schemas

`PickemSheetSchema` — the organizer's Tuesday sheet.

| Column | Type | Note |
| --- | --- | --- |
| `season`, `week` | `int` | |
| `home_team`, `away_team` | `str` | Team enum values, normalized on read |
| `home_spread` | `float` | **Standard convention** — negative means home favored |

`PickemSlateSchema` — sheet joined to consensus, one row per game.

| Column | Type | Note |
| --- | --- | --- |
| `season`, `week`, `game_id` | | |
| `home_team`, `away_team` | `str` | |
| `sheet_home_spread` | `float` | From the organizer |
| `consensus_home_spread` | `float` | `-spread_line`, from the market |
| `home_win_prob`, `away_win_prob` | `float` | Devigged; sum to 1 |
| `sheet_favorite`, `sheet_dog` | `str`, nullable | NA when the sheet spread is 0 |
| `dog_win_prob` | `float`, nullable | Consensus probability the **sheet's** dog wins |
| `dog_line_move` | `float`, nullable | Sheet's dog spread − consensus dog spread. **Positive = the dog is stronger than the sheet thinks** |
| `free_dog` | `bool` | `dog_win_prob > 0.5` — the sheet's dog is actually favored |

`PickemPicksSchema` — our picks, graded in place.

| Column | Type | Note |
| --- | --- | --- |
| `season`, `week`, `game_id` | | |
| `home_team`, `away_team` | `str` | |
| `pick` | `str` | Team we pick |
| `pick_win_prob` | `float` | |
| `is_dog_pick` | `bool` | Is `pick` the sheet's dog |
| `forced` | `bool` | True if this dog pick was a constraint swap, not the max-probability choice |
| `switch_cost` | `float` | Probability given up vs. the unconstrained best; 0.0 when not forced |
| `winner` | `str`, nullable | NA if unplayed **or tied** |
| `correct` | `bool`, nullable | NA if unplayed; False on a tie |

Picks are graded by overwriting the same partition, so one row carries a pick from entry
through result.

## Storage

| Table | Path | Partition |
| --- | --- | --- |
| sheet | `data/pickem/sheet` | season / week |
| picks | `data/pickem/picks` | season / week |

Written via the sanctioned `store.write_partition` / `read_partition` only.

## User-facing surface

### `scripts/pickem_board.py`

Two modes.

**Template** — writes a CSV pre-filled with the week's real matchups so only the numbers are
typed by hand (removing the most likely source of error, mistyped team codes):

```bash
python scripts/pickem_board.py --season 2026 --week 1 --template sheet.csv
```

```csv
away_team,home_team,home_spread
NE,SEA,
SF,LA,
CHI,CAR,
```

**Picks** — fill in `home_spread` from the organizer's email, then:

```bash
python scripts/pickem_board.py --season 2026 --week 1 --sheet sheet.csv
```

Prints the full slate with picks, the three dogs, expected correct total, and a highlighted
section for free dogs and large line moves.

Team codes are passed through `normalize_team_code` on read, so `JAC`/`JAX` and `WSH`/`WAS`
from the organizer are handled.

### `scripts/pickem_backtest.py`

Answers two questions from history:

1. **Is the market calibrated?** Bin devigged probabilities and compare predicted vs. actual win
   rates. If games priced at 63% won ~63% of the time, we can trust the number.
2. **What score should I expect?** Run the optimizer over past seasons using closing lines as
   *both* sheet and consensus (i.e., assuming zero staleness edge) to establish the baseline
   correct-picks-per-week the strategy floors at.

Note (2) is a **floor, not a forecast** of live performance: it deliberately models away the
stale-sheet edge, because historical opening-vs-closing line movement is not in this data
source. It tells us what the constraint costs and what a typical week looks like.

## Testing

- `probability`: known odds → known probabilities; devigged pair sums to exactly 1; favorite
  keeps the larger share; missing moneyline raises.
- `sheet`: round-trips; normalizes team codes; rejects unknown teams; template has one row per
  real game.
- `slate`: sign conventions in both directions (home favored / away favored); `dog_line_move`
  sign; a constructed free dog is flagged; unmatched sheet row raises.
- `optimize`: fewer than 3 natural dogs forces exactly the cheapest swaps; ≥3 natural dogs
  forces none; free dogs are picked without being marked `forced`; a 0.0 sheet spread is never
  used to satisfy the constraint; **an exhaustive brute-force check on a small slate confirms
  the greedy result is the true optimum**.
- `grade`: win/loss; tie is incorrect with NA winner; unplayed is NA.
- `schemas`: happy path plus one rejection per new constraint.

## Decisions log

| Decision | Rationale |
| --- | --- |
| Market-only; no own-model input | Stale-sheet edge is free and near-certain; beating Vegas is neither |
| Devigged moneyline as the probability source, no spread fallback | Availability is identical to spreads across 4,175 games; a fallback would be dead code |
| Proportional devig | Standard for two-way markets; refinements matter mainly for heavy favorites |
| Sheet decides the dog, market decides the winner | The central correctness invariant of the tool |
| Store `home_score`/`away_score`, not `result` | Exactly derivable in all checked rows; avoids drift |
| Ties count as incorrect | Common pool rule; stated explicitly because a push is also defensible |
| Maximize expected correct picks, not win probability | Season prize is the big one and 10–15 entrants is a small field |
| Greedy switch selection | Provably optimal here, and brute-force verified in tests |
