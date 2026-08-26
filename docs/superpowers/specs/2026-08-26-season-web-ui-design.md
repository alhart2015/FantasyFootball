# Season web UI — design

**Status:** implemented on `feat/season-web-ui` (2026-08-26). See §10 for what was cut
and what was learned building it.
**Branch:** `feat/season-web-ui`
**Date:** 2026-08-26
**Model:** the Flask UI in `C:\Users\HartAlden\FantasyBaseball` (studied 2026-08-26)

Two read-only pages over data the repo already computes:

1. **Standings** — current record and projected finish for all 16 teams.
2. **My Team** — my roster's year-to-date stats and rankings beside rest-of-season
   projections and rankings.

---

## 1. Why Flask and not Streamlit

The repo already has two Streamlit apps (`scripts/draft_board.py`, `scripts/auction_board.py`),
so this deliberately adds a second UI stack. The reason is that both pages are **dense tables
that need per-cell affordances**, which is the thing Streamlit is worst at:

- a rank badge sitting next to each stat, with the other bases on hover;
- two independent colour signals in one cell (how a player is doing vs how his projection
  moved) without adding a column;
- a YTD / ROS / Total toggle that re-renders the table body and leaves a linkable URL.

Those are ~12 lines of Jinja each in the baseball app and are either impossible or ugly in
`st.dataframe`. The draft and auction boards stay on Streamlit — they are *interactive
session* tools, which is what Streamlit is good at. This is a *dashboard over batch-computed
data*, which is what server-rendered HTML is good at.

## 2. Architecture — three layers, no exceptions

Copied wholesale from the baseball app, because its own worst problems come from where it
broke this rule.

```
src/projections/web/
    app.py                  create_app(); nothing else
    views/
        standings_view.py   format_standings_page(...) -> StandingsPage
        team_view.py        format_team_page(...) -> TeamPage
    routes/
        standings.py        Blueprint: read data, call formatter, render
        team.py             Blueprint
    templates/  static/
```

**Layer 1 — domain.** Already exists: `midseason.standings`, `midseason.rest_of_season`,
`scoring.actuals`, `draft.snake_cheat_sheet._rank_within_position`. Knows nothing about the
web. **No new domain logic belongs in this branch** beyond the YTD/ranking assembly in §4.

**Layer 2 — view models.** Pure functions, **zero Flask imports**, returning frozen
dataclasses. Modelled on the baseball repo's `trajectory_view.py` (frozen dataclasses, typed
to the template boundary) and explicitly *not* on its `season_data.py` (untyped dicts) — the
brief calls the former "the author's second attempt" and it reads like it. Testable directly,
which is where the test density goes.

**Layer 3 — routes.** Flask **Blueprints, one per page group, from day one.** The baseball app
put ~30 handlers plus nested helpers inside a single 2,307-line `register_routes(app)`, which
makes those helpers unreachable from tests except over HTTP. A route here does three things:
read, format, render.

## 3. The Standings page

Data is already complete — `project_league_standings()` returns every column this page needs,
so **the page computes nothing**.

| Section | Content |
|---|---|
| Header | League name, `week N of 14`, matchups played, a stale-data note if the snapshot is old |
| Standings table | Rank, team, W-L-T, PF, **projected wins**, playoff %, bye %, title %, mean seed |
| My remaining games | Opponent and P(win) per week, from `matchup_odds` |
| Trend | Playoff % across accumulated weekly snapshots, once more than one exists |

My row is highlighted (`.user-team`), as in the baseball standings.

**Percentage cells get one continuous colour scale**, computed server-side as a signed
intensity in `[-1, 1]` and emitted as a CSS custom property, so CSS does the mixing and one
rule covers the gradient. **Ties are omitted from the scale entirely** rather than rendered as
a fake uniform value — the baseball app learned that one.

The trend section reads the accumulated `projected_standings` partitions. It renders nothing
until at least two snapshots exist, and says so, rather than drawing a one-point line.

## 4. The My Team page

The one page that needs new assembly. Columns, per player:

```
Slot | Player | Pos | YTD pts | YTD rank | ROS pts | ROS rank | Δ
```

**YTD points come from our own scoring, not ESPN's** (user's call, 2026-08-26):
`weekly_stats` for the season run through the league `Ruleset` via
`scoring.actuals.actual_season_total`. One number everywhere. It can differ from ESPN's
official total by a rounding hair; showing two numbers for the same player would be worse, and
league-wide rankings are impossible from ESPN's per-matchup data anyway since it only covers
rostered players.

**Rankings use `_rank_within_position`**, already in `snake_cheat_sheet.py`. It is deliberately
not `Series.rank()`, whose default `method="average"` yields fractional ranks — "RB 4.5" reads
as a bug. It needs promoting out of that module to a shared home; that is the only existing
code this branch moves.

**Rank is a badge next to the stat, not its own column**, with the other basis on hover — the
`rank_badge` macro shape from the baseball app, which is ~12 lines of Jinja for a genuinely
useful affordance.

**A basis toggle (YTD / ROS / Total)** re-renders only the table body server-side and swaps it
in, updating the URL with `history.replaceState` so the state is linkable. One template serves
both the full page and the partial. Deferred to a follow-up if it costs more than it looks —
the columns above already show YTD and ROS side by side, so the toggle is an enhancement, not
the feature.

## 5. Stat categories are defined once

The baseball app hardcodes its category list in at least four places — template literals, a JS
`Set`, an enum, a route kwarg — so adding a category means editing templates. The brief calls
this out explicitly as what not to copy.

Here, **one Python structure declares each column: key, label, precision, whether higher is
better**, and both column order and formatting derive from it. Nothing in a template or a
`<script>` block re-states which columns exist.

## 6. What the pages will show before Week 1

**Nothing, and they must say so plainly.** As of 2026-08-26 there is no 2026 `weekly_stats`
partition and the draft has not happened, so:

- Standings renders its "no schedule / no rosters" state — the pipeline already raises
  `ProjectionInputError` with a usable message for both, so the route catches that and renders
  the message rather than a traceback or an empty table.
- My Team has no roster and no YTD data.

This is expected and is not a reason to defer building. The whole in-season pipeline was built
and tested this way and was correct on its first real run. But it does mean **these pages can
be tested, not eyeballed**, until Week 1 — which raises the bar on the view-model tests, not
lowers it.

## 7. Testing

Three levels, mirroring the baseball repo's, which is the strongest thing about it:

1. **View-model unit tests** — the bulk. Pure function in, frozen dataclass out. No app, no
   HTTP, no browser. Every formatting rule, colour bucket, rank, and empty state.
2. **Route tests** — Flask's `app.test_client()` against a fixture data root, asserting the
   rendered HTML contains what it should.
3. **Template-source invariants** — regex the template files for the things only copy-paste can
   break. The baseball repo pins scroll-container a11y attributes this way; here the first one
   is that no template re-declares a stat category (§5).

`create_app()` takes its data root as config so tests never touch `data/`.

## 8. Out of scope

- Auth. The baseball app is one shared password; this runs on localhost for one person.
- Writes of any kind. Both pages are readers.
- Deployment. `python scripts/run_season_dashboard.py` on localhost, as the baseball repo does
  for local work.
- Touching the draft or auction boards.

## 9. Plan

1. `create_app()`, one blueprint, a health route, and the test harness that proves the app
   starts against a fixture data root.
2. Stat-column registry (§5) plus the promoted rank helper.
3. Standings view model + template.
4. YTD/ROS assembly, then the My Team view model + template.
5. Styling pass: design tokens, the colour scale, the rank badge.

Each step is its own commit. Steps 1–3 are independently useful: a working standings page is
worth having before the team page exists.


---

## 10. What was actually built (2026-08-26)

The plan in §9 was followed step for step. **Four** things in this spec did not get built, and
two things it did not anticipate turned out to matter more than anything it did.

### Cut: the Trend section (§3)

The standings page has no trend block. §3 promises "Playoff % across accumulated weekly
snapshots, once more than one exists", reading the accumulated `projected_standings`
partitions, and there is no such code, template block or test — `grep -rn "trend\|snapshot"
src/projections/web/` finds only `run.snapshot_week`.

The reason is the same one as for the rank badge, and stronger: a line chart across weekly
snapshots requires **at least two weekly snapshots**, and as of 2026-08-26 there are zero. It
cannot be built against real data, cannot be eyeballed, and a fixture of synthetic snapshots
would be testing the chart library rather than the feature. It is the first thing to build once
Week 2 has been played and the partitions start accumulating — the writer already runs each
week, so the data will be there without any further work.

(This entry was itself a review finding: §10 originally said three things were cut and listed
the three that were easy to remember.)

### Cut: the rank badge (§4)

Ranks are their own columns (`YTD rk`, `ROS rk`), not badges beside the stat with the other
basis on hover. Two reasons. A badge with a hover affordance is the one piece of this design
that cannot be checked without a browser, and until Week 1 there is nothing to look at — the
whole page is verified by tests, so a feature that tests cannot see is a feature nobody can
see. And the table is seven columns wide; it has room. Worth revisiting once there is real
data on the page and the table is being read rather than asserted on.

### Cut: the YTD / ROS / Total basis toggle (§4)

The spec already called this "an enhancement, not the feature", and the columns show YTD and
ROS side by side as it predicted. Deferred as written. There is no partial-render route and no
`history.replaceState`; adding them later touches only `team.html` and one route.

### Cut: the Δ column (§4)

The header row in §4 ends with `Δ`, which the body of the spec never defines. Rank-change
against what — last week? draft day? Both need a stored history the repo does not keep, which
is a snapshot feature, not a column. Dropped rather than guessed at.

### Added: the two pages have to be one pipeline, not two

The assembly for the My Team page moved twice. It began inline in `routes/team.py`, where
nothing could test it (see the last section). The first fix lifted it into the view model,
which was still the wrong home: a module that both parses ESPN payloads and formats table cells
has two halves, and the seam between them is where the next round of defects sat — a frame the
presenter cleaned was read by the pipeline one step earlier, taking the page down with a 500,
and a docstring in the presenter described the pipeline the *other* page uses.

It lives in `midseason/my_team.py` now, beside `project_league_standings`, which is the same
layer for the standings page. Both pages are then the same shape: a domain function returning a
`*Run`, and a view model that renders it.

### Added: rest-of-season means rest-of-season

Not in the spec at all, and the single most consequential thing on the branch. The pool's
`season_mean_fpts` is a FULL-SEASON figure; printing it under "projected points for the rest
of the season" overstates by roughly the fraction of the season already played — about double
at week 10. `remaining_points` subtracts what a player has already scored.

Note it is deliberately NOT `rest_of_season_pool`, which the simulator uses: that converts to a
full-season-equivalent PACE, because the variance model divides by a fixed `SEASON_GAMES`. A
pace is right for the simulator and wrong to print. The two callers want different things from
the same input, which is exactly the sort of thing worth writing down.

### Added: both rank columns count the same players

`weekly_stats` is NFL-wide; the projection pool is not. Ranking YTD over one and ROS over the
other put a bigger denominator under one column than the other, so a player's YTD rank read
worse than his ROS rank for no reason but the universe — while the two columns sit adjacent
under near-identical tooltips, inviting exactly that subtraction. Both rank over the pool.

### Confirmed: §5 was the right call

The column registry paid for itself. `test_no_template_re_declares_a_column` and its sibling
checking labels both caught real re-declarations during the review, and the stylesheet turned
out to be a fourth place columns could be re-declared — `nth-child(2)` hard-coded which column
holds the name. Columns carry `is_label`; the CSS targets a class.

### Confirmed: §7 was the right call, for a reason it did not give

"These pages can be tested, not eyeballed" was written as a constraint. It behaved like one:
the only code on the branch without a view-model test — the My Team assembly, which lived
inline in a route and so was reachable only through an HTTP request that first made a live
ESPN call — held eight of the defects the review found. Lifting it into `assemble_team_page`
was the first fix of the loop and the one the rest depended on.
