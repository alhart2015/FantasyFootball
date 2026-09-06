# D/ST projections — design

**Issue [#166](https://github.com/alhart2015/FantasyFootball/issues/166). Related: [#122](https://github.com/alhart2015/FantasyFootball/issues/122).**

The user's ask, in his words: *not accounting for defenses in the 2026 draft was a costly mistake,
and roster decisions cannot keep ignoring them.*

---

## 1. What #166 assumed, and what is actually true

#166 framed the choice as **degraded v0 from implied team totals** vs. **ingest the real inputs
from play-by-play (slow, right shape)**. Both premises were wrong, and the investigation on
2026-09-06 replaced them with measured facts.

### 1.1 We are not missing defense projections. We are discarding them.

Both external sources we already ingest publish full D/ST projections. Four lines drop them:

| Line | What it does |
| --- | --- |
| `src/projections/ingest/external_projections.py:79` | `_SKILL_POSITIONS = frozenset(ESPN_POSITIONS.values())` — QB/RB/WR/TE only |
| `src/projections/ingest/external_projections.py:217` | `if position not in _SKILL_POSITIONS: continue` |
| `src/projections/ingest/sleeper_weekly_projections.py:32` | the same four-position set |
| `src/projections/ingest/sleeper_weekly_projections.py:51` | `if position not in _SKILL_POSITIONS: continue` |

Measured live against the 2026 endpoints:

- **ESPN** (`kona_player_info`, `leaguedefaults/3`): 32 rows at `defaultPositionId: 16`, each
  carrying a full per-scoring-period projected stat vector and an `appliedTotal`.
- **Sleeper** (`/projections/nfl/2026/1`): 32 rows at `position: "DEF"`, with named fields —
  `sack`, `int`, `ff`, `fum_rec`, `def_td`, `safe`, `blk_kick`, `pts_allow`, `yds_allow`,
  `st_td`, `pr_td`, `tkl_loss`, plus bucket indicators and a pre-scored `pts_half_ppr`.

The play-by-play derivation #166 contemplated is unnecessary for v1. `nflreadpy.load_team_stats`
also exists and carries every actual D/ST counting stat — that is the right source for *actuals*
and backtesting (§8), not for the projection itself.

### 1.2 Critts *does* score D/ST, and `parse_ruleset` cannot see it

`data/leagues/critts_2025_2026/espn_raw.json` carries 26 D/ST scoring categories. Every one has
base `points: 0` and its real value in `pointsOverrides["16"]` (16 = the D/ST position id):

```
 89 = 0  {'16':  5.0}      99 = 0  {'16':  1.0}     128 = 0  {'16':  5.0}
 90 = 0  {'16':  4.0}     101 = 6  {'16':  6.0}     132 = 0  {'16': -1.0}
 ...
```

`parse_ruleset` (`src/projections/ingest/espn_league.py:601`) collects unmodelled categories under
`elif points != 0.0`. Because the base value is `0`, **these 26 rules are not merely unmodelled —
they never reach the "unmodelled categories" warning at all.** The tool cannot currently report
that the league has D/ST scoring.

This is worse than #166 states and is the first thing to fix.

### 1.3 D/ST scoring is a dot product, verified exactly

The decisive finding. ESPN's D/ST score is:

```
appliedTotal = Σ over statIds:  raw_stat[id] × points_for_position_16[id]
```

where `points_for_position_16[id]` is `pointsOverrides["16"][id]` when present, else the base
`points[id]`.

**Verification run 2026-09-06** — reconstructed `appliedTotal` from the raw stat vector and the
default league's scoring settings, over every D/ST projection ESPN publishes for 2026:

```
rows checked: 1215        (32 defenses × ~38 scoring periods)
WORST absolute error: 0.00000001
```

Float noise. Not one row disagrees.

**Consequence: the implementation never needs to know what a statId means.** No hand-written
`{99: "sacks", 89: "zero_points_allowed", ...}` table, and therefore no class of bug where a
mis-transcribed id produces a plausible-looking wrong projection.

### 1.4 The Critts point values are confirmed against the league UI

The user supplied League Info screenshots on 2026-09-06. Diffed against the payload:

```
matched 26/26   mismatches 0
screenshot labels with no payload id: none
```

Verified id → label map (for **display and diagnostics only** — never the scoring path):

| id | label | Critts | id | label | Critts |
| --- | --- | --- | --- | --- | --- |
| 89 | PAO (0 pts allowed) | 5 | 104 | INTTD | 6 |
| 90 | PA1 (1–6) | 4 | 123 | PA28 (28–34) | −1 |
| 91 | PA7 (7–13) | 3 | 124 | PA35 (35–45) | −3 |
| 92 | PA14 (14–17) | 1 | 125 | PA46 (46+) | −5 |
| 93 | BLKKRTD | 6 | 128 | YA100 (<100) | 5 |
| 95 | INT | 2 | 129 | YA199 (100–199) | 3 |
| 96 | FR | 2 | 130 | YA299 (200–299) | 2 |
| 97 | BLKK | 2 | 132 | YA399 (350–399) | −1 |
| 98 | SF | 2 | 133 | YA449 (400–449) | −3 |
| 99 | SK | 1 | 134 | YA499 (450–499) | −5 |
| 101 | KRTD | 6 | 135 | YA549 (500–549) | −6 |
| 102 | PRTD | 6 | 136 | YA550 (550+) | −7 |
| 103 | FRTD | 6 | 206 | 2PTRET | 2 |

Note the 18–27 points-allowed band is absent from both sources: it is worth 0, correctly omitted.

Note also `FUML` (statId 72, −2) carries a **base** value and no position override, so under the
dot product it applies to D/ST as well. That is real ESPN behaviour (a defense can lose a fumble
on a return) and the rule handles it without a special case.

---

## 2. What exists and is reused, and what this spec builds

**Reused, not rebuilt:**

| Piece | Where | Used for |
| --- | --- | --- |
| ESPN + Sleeper preseason ingest | `ingest/external_projections.py` | the projection source; positions are filtered, the fetch already returns D/ST |
| Sleeper weekly ingest | `ingest/sleeper_weekly_projections.py` | in-season weekly D/ST |
| `parse_ruleset` / `build_league_config` | `ingest/espn_league.py` | league scoring + roster slots |
| `Position.DST`, `RosterSlot.DST` | `schemas.py:42`, `schemas.py:154` | already exist; nothing to add |
| `store.write_partition` / `read_partition` | `store/` | all parquet I/O |
| `generate_vorp_table` | `draft/vorp.py:129` | replacement level, once DST is in `roster_slots` |
| `normalize_team_code` | `schemas.py` | Sleeper's DEF id is a raw team code |

**Built here:** a `Team → GsisId` mapping, D/ST scoring on `Ruleset`, the position filters opened,
and the downstream schemas widened.

---

## 3. Identity: 32 hardcoded synthetic gsis_ids

D/ST is team-level. Its natural key is `Team`, not `GsisId`. Every storage and join path in this
repo keys on `GsisId` (CLAUDE.md: *"`GsisId` is canonical"*), and rewriting that for one position
is not proportionate.

**Decision (user, 2026-09-06): a hardcoded 32-team mapping. There are 32 teams and they will not
change.**

Synthetic ids follow the existing placeholder convention (`external_projections` already mints
`99-XXXXXXX` for pre-camp rookies) and must satisfy `GSIS_ID_PATTERN = r"\d{2}-\d{7}"`:

```python
# schemas.py — single source of truth, mirroring the Team enum.
DST_GSIS_IDS: Final[Mapping[Team, GsisId]] = {
    Team.ARI: GsisId("98-0000001"),
    Team.ATL: GsisId("98-0000002"),
    ...  # 32 entries, ordered as Team is ordered
}
DST_TEAM_BY_GSIS: Final[Mapping[GsisId, Team]] = {v: k for k, v in DST_GSIS_IDS.items()}
```

`98-` is chosen deliberately: distinct from real ids and from the rookie placeholder block
(`99-`), so a defense id is recognisable on sight and in a parquet dump.

**Constraints, enforced by test:**

- The map covers every member of `Team` exactly once and is injective.
- Every value matches `GSIS_ID_PATTERN` and round-trips through `validate_gsis_id`.
- No value collides with the `99-` placeholder block.
- **The ids are frozen.** They are persisted in parquet; changing one silently orphans history. The
  test pins the literal values, so a reorder or renumber fails loudly.

`is_placeholder_gsis` stays `False` for these — they are stable and canonical, not awaiting
reconciliation. The `id_map` gains a row per defense (`espn_id` = ESPN's D/ST player id,
`sleeper_id` = the team code, `position` = `DST`, `team` = the team).

---

## 4. Scoring: `Ruleset.dst_points_by_stat_id`

`Ruleset` models named skill categories (`passing_td_pts`, `reception_pts`). D/ST does not fit
that shape, and §1.3 proved it does not need to.

```python
class Ruleset(BaseModel):
    ...
    #: ESPN statId -> points, for position 16 only. Empty for a league that does not
    #: roster a D/ST. Populated from pointsOverrides["16"], falling back to the base
    #: `points` value. Scoring is a dot product against the projected stat vector --
    #: see docs/superpowers/specs/2026-09-06-dst-projections-design.md §1.3 for the
    #: 1215-row exactness proof. Deliberately keyed by id, not by name: a hand-written
    #: id->name table is a wrong-projection-that-looks-right waiting to happen.
    dst_points_by_stat_id: Mapping[str, float] = Field(default_factory=dict)
```

`Ruleset` is a frozen model, so the value must be an immutable mapping type at construction and
must hash — the plan pins how (the model is hashed/cached per its own docstring).

**Rejected alternative:** ~27 named fields plus an id→name translation table. It reads better and
reintroduces exactly the failure mode §1.3 eliminated. The verified table in §1.4 is carried as a
**display-only** constant (`DST_STAT_LABELS`) so output can say "sacks" without the scoring path
ever consulting it.

`score_dst(stat_vector, ruleset) -> float` lives in `src/projections/scoring/dst.py` and is the
only place that converts a D/ST stat vector to points — same rule as the rest of the scoring
layer (CLAUDE.md: *"the scoring layer is the only place that knows what counts as a fantasy
point"*).

### 4.1 `parse_ruleset` changes

1. Read `pointsOverrides["16"]` into `dst_points_by_stat_id`, base `points` where no override.
2. **Stop treating a `0` base value as "not scored."** The current `elif points != 0.0` guard is
   why 26 live rules are invisible. An item carrying any position override is scored somewhere.
3. Keep the unmodelled-category note for genuinely unmapped ids — kicker categories remain
   unmodelled and must keep saying so (§7.1).

### 4.2 Sleeper

Sleeper publishes named fields, not ESPN statIds, so the dot product does not apply directly.
v1 maps the Sleeper D/ST fields onto the same **stat vector** the ESPN path produces, using the
§1.4 label map in reverse — this is the one place a name↔id translation exists, it is confined to
one source adapter, and it is covered by a test that scores a known Sleeper line under Critts
rules and compares against the ESPN line for the same team-week within tolerance.

**Known mismatch to handle explicitly:** Sleeper's points-allowed buckets do not align with
ESPN's (`pts_allow_14_20` vs ESPN's `14–17` + `18–27`). Where a Sleeper bucket straddles an ESPN
boundary, v1 **drops the bucket and re-derives it from Sleeper's continuous `pts_allow` /
`yds_allow`**, which are unambiguous. This is stated in the module docstring, not silently done.

---

## 5. Opening the filters

| File | Change |
| --- | --- |
| `ingest/external_projections.py:73,79,217` | add `16: Position.DST.value` to `ESPN_POSITIONS`; admit DST |
| `ingest/sleeper_weekly_projections.py:32,51` | admit `DEF` and map it to `Position.DST` |
| `ingest/espn_league.py:690` | stop dropping `RosterSlot.DST` (§5.1) |
| `schemas.py:387` | `_SKILL_POSITION_VALUES` — widen the two schemas in §5.2 |

### 5.1 `build_league_config` and replacement level

The docstring at `espn_league.py:660` argues dropping DST is *correct* replacement math because a
D/ST pick "does not consume a skill player." That was true only while D/ST was unprojectable. It
is now the bug #166 names: a 16-team league drains 16 × 13, not 16 × 12, and the 2026 draft is the
evidence.

`RosterSlot.K` **stays dropped.** Critts rosters no kicker, kicker categories remain unmodelled,
and widening that too would ship an unvalidated position. The drop becomes K-only and the log
message says so.

### 5.2 Schemas to widen

`_SKILL_POSITION_VALUES` gates five schemas. Each is widened to include `Position.DST` only if
D/ST rows genuinely flow through it:

| Schema | Line | Verdict |
| --- | --- | --- |
| `ConsensusProjectionSchema` | 1019 | **widen** — the blend must carry defenses |
| `WaiverPoolSchema` | 1273 | **widen** — streaming a defense is the point (#166) |
| `PreseasonFeaturesSchema` | 1307 | **leave** — feature builders are skill-only; no DST features in v1 |
| `PreseasonProjectionSchema` | 1413 | **leave** — same |
| `PreseasonBacktestSchema` | 1478 | **leave** — same |

Leaving three narrow is deliberate: D/ST in v1 is an **external projection passed through**, not a
modelled position. §8 covers what it would take to change that.

---

## 6. Distributions

`generate_vorp_table` and the season simulators need a distribution, not a point estimate. v1
fits the same family used elsewhere from the external projection's mean plus a variance estimated
from `nflreadpy.load_team_stats` history (2018–2025 actual weekly D/ST scores under the league
ruleset). D/ST is high-variance and the sims must see that — treating a 7.9-point projection as
near-deterministic would understate both the upside of a good streaming call and the downside of
a bad one.

Variance is fit **once, offline**, and stored as a per-team-strength-tier parameter, not per team.
32 teams × 17 weeks is too little data to fit per-team spread without overfitting.

---

## 7. Failure modes this must not have

1. **A silent kicker hole.** Widening DST must not imply K works. The log line and the unmodelled
   note must keep naming K explicitly.
2. **A wrong id that looks right.** Addressed structurally in §4 — the scoring path has no
   id→name table to get wrong. The one translation (Sleeper, §4.2) is isolated and cross-checked
   against ESPN.
3. **Renumbered gsis_ids orphaning stored data.** Pinned by test (§3).
4. **Replacement level moving without anyone noticing.** Adding DST to `roster_slots` changes VORP
   for *every* position. The plan must include a before/after VORP diff on the real Critts config,
   reviewed by the user, not just a passing test suite.
5. **A defense projection that is silently zero.** If `dst_points_by_stat_id` is empty (league has
   no D/ST scoring) but a DST roster slot exists, that is a contradiction — raise, do not project 0.
6. **Sleeper bucket misalignment applied silently.** §4.2 requires it be documented and derived,
   never fudged.

---

## 8. Out of scope for v1

- **Kickers.** #122 keeps that half.
- **A modelled D/ST projection.** v1 passes through and blends external projections. A real model
  (opponent implied total, opponent sack rate allowed, turnover-worthy throw rate) is the natural
  follow-up and is what #122's feature-builder half is for. `nflreadpy.load_team_stats` is the
  actuals source when that happens.
- **IDP.** `Position` reserves it; nothing here touches it.
- **DFS D/ST.** Different scoring, different optimizer constraint. Separate issue.

---

## 9. Plan

Phased per CLAUDE.md; each phase ends green with the full end-of-effort checklist run.

1. **Identity.** `DST_GSIS_IDS` + inverse + frozen-value tests. No behaviour change.
2. **Scoring.** `Ruleset.dst_points_by_stat_id`, `scoring/dst.py`, `parse_ruleset` reads
   `pointsOverrides["16"]` and stops hiding zero-base items. Test: reconstruct the 1215-row ESPN
   fixture exactly (§1.3) and the Critts 26-category map (§1.4).
3. **ESPN ingest.** Admit DST through `external_projections`; store the stat vector; `id_map` rows.
4. **Sleeper ingest.** Admit DEF; the isolated name→id adapter; the cross-source agreement test.
5. **League config + schemas.** Stop dropping `RosterSlot.DST`; widen the two schemas in §5.2.
6. **Distributions + VORP.** §6, then the before/after VORP diff for user review (§7.4).
7. **Downstream verification.** Re-run `projected_standings`, `trade_analyzer`,
   `waiver_recommender` on live Critts data and confirm the three "N players not valued" notes
   drop by the expected count.

---

## 10. The question this is being built to answer

*What is a defense worth, to this roster, right now — as a draft pick, as a trade chip, and as a
weekly streaming add?* v1 answers all three at external-projection quality. #122 makes the answer
ours rather than ESPN's.
