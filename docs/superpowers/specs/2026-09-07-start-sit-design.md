# Start/Sit — design

**Branch** `feat/start-sit` · **Date** 2026-09-07 · **Sub-project** Mid-season Manager

## 1. The question

*Is the lineup I have set the best lineup I can set this week?*

The last unbuilt tool of the mid-season set. `projected_standings`, `waiver_recommender` and
`trade_analyzer` all ship; nothing tells you who to start.

## 2. Why this is not "read the ESPN app"

**The user's constraint, stated plainly: a tool that reports ESPN's weekly projection back to
him has no value, because the ESPN app already shows it.** That is correct and it is the
design's hardest requirement. Three things make this tool say something ESPN cannot:

1. **Two independent weekly sources, blended.** ESPN's weekly feed AND Sleeper's weekly
   endpoint. Verified live 2026-09-07: `api.sleeper.com/projections/nfl/2026/2` returns 948
   scored players including K and DEF, each with a full stat line and an `opponent`. Where the
   two disagree is exactly where a start/sit call is interesting, and the ESPN app cannot show
   disagreement with a source it does not carry.
2. **Scored under this league's actual ruleset, not the source's.** Both sources ship stat
   lines; we blend per-stat and score once through `scoring.expected_points` with the league's
   `Ruleset`. ESPN's app shows ESPN's scoring assumptions.
3. **The optimal lineup, cascade and all.** `choose_starters` solves FLEX/SUPER_FLEX properly.
   A WR beating your WR2 by 0.2 can push WR2 into the flex and the flex RB out, for a lineup
   gain of 1.2 — the pairwise comparison a human eye makes understates every move.

## 3. Blend in stat-line space, not points space

Follows `consensus.blend` and `dfs/blend.blend_statlines`: average per-stat means, assemble
one line, score once.

**Be precise about why, because the obvious reason is wrong here.** `Ruleset` is purely
linear — points-per-yard and points-per-event, no bonus thresholds — so blending in stat
space and blending two scored totals give *the same number* whenever both sources report
the same fields. The difference is entirely in the case where they do not:

- **Stat space weights per field.** ESPN reports receptions, Sleeper omits them: the
  reception count comes from ESPN alone at full weight, every other field blends 50/50.
- **Points space weights per player.** The same case averages a total that includes ESPN's
  receptions against one that silently excludes them, and the blend reads low by half of
  ESPN's reception points with nothing on screen to say so.

That is the live reason. Two more that are real but not yet load-bearing: a source's own
published total (`pts_half_ppr`, ESPN's `appliedTotal`) is scored under *its* rules rather
than the league's, and a ruleset that later gains a yardage bonus would make the two spaces
diverge everywhere rather than only on missing fields.

`espn_weekly.parse_espn_weekly` already builds the full line in `_statline_dict` and scores it
immediately. **Expose the line** (new sibling returning stat columns) rather than re-deriving.

Default `--weight-espn 0.5`. Unmeasured; the honest default until a weekly benchmark exists
(see §9).

## 4. Sources, keys, and what happens when one is missing

ESPN weekly is keyed by **espn_id**; Sleeper by **sleeper_id → gsis_id** via `id_map`. The
blend joins on espn_id, crosswalking Sleeper through the id_map.

**A player only one source can price is priced by that source alone, and the report says so.**
Never silently zero. Concretely:

- `parse_sleeper_weekly` filters to QB/RB/WR/TE, so **K and DST fall back to ESPN-only**. The
  league starts a DST (#166/#168), so this is a real, common case, not a corner.
- A just-signed player absent from `id_map` is ESPN-only. This is the exact failure
  `waivers.weekly_projections_by_espn_id` avoids gsis for; the same care applies here.
- **Absent from both = `None`, not `0.0`.** `choose_starters` reads `None` as unstartable, and
  that is how bye weeks work with no rule about bye weeks.

The `src` column names which sources priced each player, so a spread of `—` is legible rather
than mysterious.

## 5. Injuries: applied here, not delegated to the source

`waivers.adjusted_weekly_points` takes `source_is_injury_aware`, a single boolean. **A blend of
one injury-aware source and one unmeasured source has no honest value for it**, so this tool
applies the adjustment itself from the roster's `injury_status`:

**`weekly_multiplier(status, source_is_injury_aware=False)`, plus one structural exception.**

`False` is the load-bearing half. ESPN zeroes the players it lists as `Out`; Sleeper's
behaviour there is unmeasured. A blended `Out` player therefore carries roughly *half* of
Sleeper's projection, and passing `True` — which tells the helper "the source already priced
this" — would leave that half standing. A healthy-looking number for a player who will not
take a snap is exactly the plausible-wrong-answer class this repo keeps paying for. `False`
applies our own multiplier over the blend and lands him at 0.0.

**The exception is `INJURY_RESERVE` -> `None`, and only that one.** IR is a *roster slot*, not
a game status: ESPN will not let an IR player occupy a lineup slot at all, so he is
structurally unstartable rather than merely projected at nothing. `waivers._row` already
forces `None` on this exact condition via `is_on_ir`.

**Everything else keeps its multiplier and stays startable — an earlier draft of this spec got
that wrong.** It forced `OUT`, `SUSPENSION` and `DOUBTFUL` to `None` too. `choose_starters`
says why that is wrong in its own docstring: `0.0` "is a real projection of nothing and can
still fill a slot no one else is eligible for," while `None` cannot fill it at all. Two
consequences:

- `DOUBTFUL` is **0.04**, a measured number, not zero. When the only alternative at a slot is
  a bye-week player — who genuinely is `None` — starting the doubtful player is correct and
  the tool must be able to say so. Forcing `None` would have left the slot empty.
- `OUT` and `SUSPENSION` are 0.0, which already gives the wanted behaviour with no special
  case: they rank below every healthy player and get slotted only when nobody else is
  eligible, which is what a manager forced to field a body actually does.

The 0.86 on `QUESTIONABLE` applies either way, and is why the adjustment is here at all: on a
one-week horizon that tag is a 14% cut, "the size that decides a close start/sit."

## 6. The output

Two blocks. The second is the answer; the first is the evidence.

```
LINEUP — Week 2                              blend: 50% ESPN / 50% Sleeper

  player               pos  slot   espn   slpr  blend  spread  src
  Josh Allen           QB   QB     22.4   21.8   22.1     0.6  both
  Bijan Robinson       RB   RB1    16.2   15.9   16.1     0.3  both
  ...
  Bears D/ST           DST  DST     7.1      —    7.1       —  espn

SWAPS — 2 changes worth making                            +3.4 pts

  START  Chase Brown        RB   11.4   (bench -> FLEX)
  SIT    Tyjae Spears       RB    9.6   (FLEX -> bench)
         +1.8 pts   P(right) 61%

  START  Jauan Jennings     WR   13.1   (bench -> WR2)
  SIT    Rome Odunze        WR   11.5   (WR2 -> bench)   QUESTIONABLE x0.86
         +1.6 pts   P(right) 57%
```

Empty state is a success, not a failure, and reads as one — the same convention
`waiver_recommender` uses:

```
  Your lineup is already optimal. Nothing to change.
```

## 7. P(right), and why NOT change in expected wins

`waiver_recommender --wins` reports Δ expected season wins. **That instrument cannot resolve a
start/sit swap and this tool must not pretend otherwise.** From `measure_swap_noise.py`:
paired noise is **0.062 wins** at 2,000 sims, and roughly **140 season points make a win**. A
3-point weekly swap is ~0.021 wins — a signal three times smaller than the error on it. The
waiver PM notes already grade a 5-point swap at 0.04 wins as "near the edge of resolution";
start/sit swaps live below that edge.

**`P(right)` is the resolvable question:** the probability the player being started outscores
the player being benched, this week. Drawn from the fitted `VarianceParams` the repo already
carries — `weekly_std(position, per_game_mean)` gives each player's weekly spread — so it costs
no Monte-Carlo season. It answers the thing a manager actually wants: is this a real edge or a
coin flip? A 2-point edge between two RBs with weekly sd ~7 is a ~55% call, and printing 55%
is the difference between a recommendation and a superstition.

**Deliberately excluded, with the reason:** Δ expected wins. If a future weekly benchmark shows
the blend beats ESPN by enough to move a season, revisit.

## 8. Scope cuts

- **No writing lineups back to ESPN.** Every ESPN call in `ingest/espn_league.py` is a GET;
  a POST path is a separate, riskier change.
- **No blend weight tuning.** 50/50 until the weekly benchmark in §9 measures it.
- **No matchup/defense-adjustment of our own.** Both sources already price the opponent; a
  third adjustment on top would double-count.
- **Not wired into the dashboard.** CLI first; the web page is a follow-up.

## 9. Follow-up this makes possible

The fair weekly benchmark `project_management.md` has wanted since the consensus spike: with
ESPN weekly, Sleeper weekly and a blend all scored under one ruleset,
`benchmark_projections.py` can measure all three against weekly actuals and settle the blend
weight — and settle whether our own model has any weekly value at all.

## 10. Reuse — what this does NOT rewrite

- `choose_starters` (`draft/roster_eligibility.py`) — the engine. Fourth caller, not fourth copy.
- `waivers.lineup_points` — the seam. Public, takes any sequence with `.projected`/`.position`.
- `build_my_team` — week derivation, roster, `lineup_slot`, `injury_status`.
- `injuries.weekly_multiplier` — the measured 0.86.
- `league_profile.add_league_arguments` / `resolve_league_target` — the five shared flags.
- `dfs/blend.blend_statlines` — the stat-line blend pattern.

New code is the blend join, the slot labelling `choose_starters` does not return, the
current-vs-optimal diff, and `P(right)`.
