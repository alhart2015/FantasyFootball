# Trade analyzer — design

**Issue [#154](https://github.com/alhart2015/FantasyFootball/issues/154). Fourth and last feature
of the Mid-season Manager (sub-project #3).**

The user's ask, in his words: *identify positions of strength on my roster to trade from, and
opponents who are weak at those positions; and vice versa, weaknesses on my roster and strong
opponents to trade with. Come up with trades that seem fair given ESPN projections but that our
projections say are advantageous for me.*

That is three tools, and the third is the point. #154 as filed evaluates **a proposal you already
have**. This spec keeps that and adds the harder half: **finding** the proposal.

---

## 1. Why this, and why now

The waiver recommender answered "is anyone on the wire better than someone on my team?" On the
post-draft Critts league the answer is **no, and structurally so**. Measured 2026-08-30 over all
764 free agents ESPN returns:

| | Season points |
| --- | --- |
| Best available RB (Tyrone Tracy Jr.) | 67.8 |
| The user's **worst rostered** RB (Keaton Mitchell) | 85.6 |

In a 16-team league with 208 players rostered, the wire is worse than the back of a bench. The
waiver tool's own output says so: *"Nothing on the wire would change your starting lineup this
week. In a 16-team league that is the usual answer, not a failure."*

**So trade is the only lever that moves anything**, and the motivating case is live: Josh Jacobs
is reportedly facing a season-long suspension, which costs **−1.22 projected wins** (playoff
52.0% → 31.8%, title 10.0% → 4.5%). Nothing on waivers recovers a single point of it.

### The asymmetry that makes trading work

Two paired simulations, 3000 sims each, same league:

| Losing… | Costs |
| --- | --- |
| Josh Jacobs (RB) | **−1.22 wins** |
| Lamar Jackson (QB) | **−0.65 wins** |

Lamar outprojects Jacobs by 106 points and is worth **half as much** to this roster. The reason is
positional, not personal: one QB starts, and **Bryce Young (228.6 projected) is sitting on waivers
at 9.8% rostered**. Losing Lamar costs the gap to a free replacement. Losing Jacobs costs the gap
to Jordan Mason plus the bye/injury cover he provided.

**That gap between a player's value and his value *to this roster* is what the tool has to find,
for all 16 teams, in both directions.** Everything below is machinery for measuring it.

---

## 2. What exists, and what does not

**Exists and is reused, not rebuilt:**

| Piece | Where | Used for |
| --- | --- | --- |
| `choose_starters` | `draft/roster_eligibility.py` | which players actually start |
| `optimal_lineup_points` | `draft/assistant/roster_score.py` | lineup value of a roster |
| `simulate_seasons` / `project_league_standings` | `draft/assistant/league_projection.py`, `midseason/standings.py` | Δ wins / playoff / title |
| `swap_impact.py` | `midseason/` | paired Δ-wins for one roster change — built for waivers, the same shape |
| `parse_rosters`, `espn_to_gsis`, `pool_name_index` | `ingest/espn_league.py` | live rosters for all 16 teams |
| `InjuryStatus`, `season_multiplier` | `midseason/injuries.py` | a suspended/injured player is not worth his projection |
| `external_projections` (ESPN rows) | `data/raw/external_projections/` | the market's valuation |
| `Ruleset` scoring | `scoring/` | both valuations scored the same way |

**Does not exist, and this spec builds it:**

- Per-`(team, position)` **surplus** and **need**, for all 16 teams.
- A **market valuation** distinct from ours, and the gap between them.
- Trade **generation** and ranking (the issue only covered evaluation).

**Verified available before designing on it** (2026-08-30): ESPN publishes a full stat line for
**460** players and auction values for 277; **192 of 192 rostered skill players have an ESPN
projection**, all resolving by gsis with no name fallback needed. The market half is not a guess.

---

## 3. Strength and need, defined so they can be measured

Raw points cannot answer "am I strong at RB". A fourth running back in a 2-RB-plus-flex league is
worth zero to the lineup and a lot on the trade market. The definitions must be **marginal and
roster-aware**, and both fall out of `choose_starters`, which the waiver work already extracted
for exactly this reason.

### 3.1 Surplus — what a player is worth *to his own team*

```
surplus(p, t) = best_lineup(roster_t) − best_lineup(roster_t − p)
```

The lineup points team `t` loses by giving up `p`, after the next man steps in and the whole
lineup re-optimises. **Low surplus = tradeable.** This is the same cascade the waiver spec found
mattered — remove an RB2 and the flex RB slides up and a bench player enters the flex, so
comparing a player only against his direct backup understates every move.

A player can have high season points and near-zero surplus. That player is the trade chip.

### 3.2 Need — what an upgrade is worth

```
need(q, t) = best_lineup(roster_t + ref_q) − best_lineup(roster_t)
```

where `ref_q` is a **reference player at position `q`**, fixed across all teams so the number is
comparable. `ref_q` = the median *starter* at `q` across the 16 teams — deliberately not a star
(which would make every team look needy) and not replacement level (which would make none of
them). **High need = they will pay.**

### 3.3 The screen

A trade is worth generating when it moves a player from **low surplus on the sender** to **high
need on the receiver**, in both directions at once. This is a cheap arithmetic screen over
16 teams × ~12 players, and it exists to keep the expensive stage from running on 2,500 pairs.

---

## 4. Two valuations, and the gap between them

The user's request turns on holding **two** numbers per player:

| | Symbol | Source |
| --- | --- | --- |
| **Market** | `espn_pts(p)` | ESPN's own stat-line projection, scored under **this league's `Ruleset`** |
| **Ours** | `our_pts(p)` | the pool's `season_mean_fpts` (ESPN + Sleeper consensus) |

```
edge(p) = our_pts(p) − espn_pts(p)
```

**Positive edge: we like him more than the market does — a player to acquire. Negative edge: a
player to send.** Both are scored through the same `Ruleset`, because comparing a
half-PPR-scored projection to a full-PPR one is a units error that would look like an edge.

**A caveat the tool must print, not bury:** our consensus *contains* ESPN, so the two are
correlated and `edge` is the disagreement between ESPN and Sleeper weighted by our blend, not an
independent opinion. It is a real signal about market divergence; it is not proof we are right.
The tool reports the edge and its source, and never claims more.

### 4.1 Two different reasons a trade can be good, and they must be told apart

1. **Positional fit** — I have surplus RB, you have surplus WR, we both start more of what we
   lack. This gain is real *even if both sides value every player identically*, and it is the
   kind an opponent can be shown and will agree with.
2. **Edge** — we think a player is worth more than the market does. This gain exists only if our
   projections are better than ESPN's.

**The output attributes the gain to each**, because they carry completely different confidence. A
trade that is all fit is one I would send with a straight face; a trade that is all edge is a bet
on our model.

---

## 5. What makes a trade proposable

A trade nobody accepts is not a trade. Two conditions, and the asymmetry between them *is* the
feature:

- **Acceptable to them, by their numbers.** `Σ espn_pts(they receive) ≥ Σ espn_pts(they send)`,
  within a tolerance, **and** their lineup improves under `espn_pts`. They evaluate on the market
  view; the proposal must look good *there*.
- **Good for me, by ours.** Positive Δ expected wins under `our_pts`.

This is a deliberate, narrow departure from #154's line — *"no opponent-preference model … do not
pretend to model what they will say yes to."* That warning is about modelling **psychology**, and
it stands: nothing here knows that the guy in slot 12 never trades within his division. What this
does is score their side on a **stated, checkable proxy** — the public projections both managers
can see — and report it as such. **The tool says "this looks fair on ESPN's numbers", never "he
will accept".**

The honest framing of the whole feature: *find trades that are genuinely good for both sides on
the shared public view, and additionally good for me on ours.*

---

## 6. Objective: Δ expected wins, two stages

Straight from the waiver recommender, including why:

- **Stage 1 — lineup arithmetic.** Δ optimal starting lineup points, mine and theirs, over every
  candidate that clears the §3.3 screen. Cheap, exhaustive, verifiable by hand against a roster.
  It decides *what gets simulated*; it is not the answer.
- **Stage 2 — Δ expected wins.** Paired Monte-Carlo over the real remaining schedule for the
  shortlist only. **Both rosters change**, so unlike a waiver add this perturbs the league:
  the opponent's own record moves, and if they are a playoff rival that matters on top of the
  points.

**Pairing and the noise floor.** `measure_swap_noise.py` measured a paired difference at
**sd ≈ 0.062 wins** over 2,000 sims (unpaired 0.127 — pairing is worth ~2x, not the order of
magnitude first assumed, because `project_league_standings` reseeds internally). **Nothing below
~0.15 wins is reported as a difference**, and the threshold is printed beside the number rather
than left for the reader to infer. Rosters are modified **in place, in slot order** — the waiver
work found that appending a player instead of replacing him in the departing player's slot shifts
every later player onto a different draw and silently unpairs the comparison, making a no-op swap
read as 0.05 wins.

---

## 7. Output

Three sections, matching the three things asked for.

**A. My roster, by position** — surplus per player, need per slot, with the league distribution
beside it so "strong" is relative to the league and not to a feeling.

**B. The market map** — a 16 × 4 grid of need by team and position, with my own row highlighted.
This is the "who should I call" answer on its own, before any specific trade.

**C. Ranked proposals** — per trade:

| Column | Meaning |
| --- | --- |
| out / in | players, with position and injury status |
| `Δ lineup (me)` | stage-1 points, my optimal lineup |
| `Δ wins (me)`, `Δ playoff`, `Δ title` | stage 2, paired |
| `P(good for me)` | share of simulations where my season improves — the issue asked for uncertainty, not a point estimate |
| `ESPN balance` | `Σ espn_pts(in) − Σ espn_pts(out)`; **negative or near zero is the sellable trade** |
| `Δ lineup (them, ESPN)` | their gain on their own numbers — the acceptability proxy |
| `from fit` / `from edge` | the §4.1 attribution |

Sorted by Δ wins. A trade that is +wins for me and ESPN-negative for them is printed **and
flagged as a lowball** rather than silently ranked first.

---

## 8. Failure modes this must not have

Each is a plausible-looking wrong answer, which is the only kind that matters here.

1. **Valuing a suspended or injured player at his projection.** Josh Jacobs projects 212.2 and may
   play zero games. Both valuations pass through `season_multiplier` on `InjuryStatus`, and any
   player whose status is doing real work in a recommendation gets the beat-reporter note printed
   under it, as the waiver tool does. **A trade proposal is the single easiest place to be
   defrauded by a stale injury tag**, in both directions.
2. **Scoring the two valuations under different rulesets.** A units error that looks exactly like
   an edge. Both go through the league's own `Ruleset`; a test pins that a player with identical
   ESPN and consensus stat lines has `edge == 0`.
3. **Ignoring the lineup cascade.** Comparing a player to his direct backup rather than
   re-optimising the whole lineup understates every trade. `choose_starters`, not arithmetic.
4. **Unpairing the simulation by roster order.** See §6. Pinned by an exact-zero assertion on a
   no-op trade — the only test that catches it.
5. **Roster-size deltas.** A 2-for-1 leaves the receiver a man short and the sender a man over.
   #154 flags this as an open question. **Decision: v1 generates only size-neutral trades**
   (1-for-1, 2-for-2). It sidesteps crediting a trade with a free-agent add, and given §1 the FA
   pool is worthless here anyway. 2-for-1 is deferred to §10 with the reason.
6. **Rookies dropped from opponent rosters.** Fixed 2026-08-30 (PR #163) — synthetic `99-` gsis
   ids have no id_map edge, and 16 rostered rookies were silently vanishing, three of them
   starters. The trade analyzer reads the same rosters and would have inherited it. Regression
   pinned by the existing pipeline test.
7. **Treating my own team as special in the machinery.** Surplus and need are computed identically
   for all 16 teams; "mine" is a display flag. A separate code path for my roster is how the two
   sides of a trade come to be scored by subtly different rules.

---

## 9. Out of scope for v1

- **2-for-1 and larger.** Needs a roster-spot-filling policy, and the FA pool that would fill it
  is measurably worthless in this league (§1). Revisit if the wire improves.
- **Draft-pick / FAAB components.** Not modelled anywhere in the repo.
- **Multi-team trades.** Combinatorially large, rare in practice.
- **Actually sending the trade.** Read-only, like every other ESPN-facing tool here.
- **Predicting acceptance.** §5 — a stated proxy, never a claim about a person.

---

## 10. Plan

Phased per `CLAUDE.md`, each phase verified before the next.

| Phase | What | Verification |
| --- | --- | --- |
| **1** | `midseason/valuation.py` — `espn_pts` / `our_pts` / `edge`, both through the league `Ruleset` | unit tests incl. the `edge == 0` identity |
| **2** | `midseason/roster_shape.py` — `surplus`, `need`, the 16×4 market map | hand-checked against the user's real roster; Lamar-vs-Jacobs asymmetry must reproduce |
| **3** | `midseason/trades.py` — generation, the §5 filters, stage-1 ranking | no-op trade scores exactly 0.0 |
| **4** | Stage-2 Δ wins over the shortlist, paired | no-op trade reads 0.0 wins, not 0.05 |
| **5** | `scripts/trade_analyzer.py` CLI + the three output sections | end-to-end on league 856974 |

Optional Phase 6: a dashboard page, deferred on the same reasoning as the waiver page — it is a
thin presentation over the same objects and cannot be judged until the tables have been read once
in anger.

---

## 11. The question this tool is actually being built to answer

*Josh Jacobs may miss the season. The wire cannot replace him. Who in this league has a running
back they do not need, what do they need that I have, and is there a deal that ESPN's own numbers
call fair?*

If v1 answers that with a specific, sendable proposal, it has earned its place.
