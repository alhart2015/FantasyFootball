# D/ST projections — implementation plan

Spec: `docs/superpowers/specs/2026-09-06-dst-projections-design.md`. Issue
[#166](https://github.com/alhart2015/FantasyFootball/issues/166).

Each task ends green on the full end-of-effort checklist (`pytest`, `mypy src tests`,
`ruff check src tests`, `ruff format --check src tests`). Phases 1–4 are **additive**: no existing
behaviour changes. Phase 5 is the first one that moves numbers, and it stops for user review.

---

## Phase 1 — Identity (additive)

**1.1** `schemas.py`: add `DST_GSIS_IDS: Final[Mapping[Team, GsisId]]`, 32 entries in `Team`
declaration order, `98-0000001` … `98-0000032`. Add the inverse `DST_TEAM_BY_GSIS`.

**1.2** `tests/test_schemas/test_dst_ids.py`:
- covers every `Team` exactly once; injective
- every value round-trips `validate_gsis_id`
- no value starts `99-` (rookie placeholder block)
- **literal values pinned** — the ids are persisted in parquet, so a renumber must fail loudly

---

## Phase 2 — Scoring (additive)

**2.1** `Ruleset.dst_stat_points: tuple[tuple[str, float], ...] = ()`, sorted by `int(stat_id)`.

Stored as a tuple, not a dict, because `Ruleset` is `frozen=True` and its docstring promises
"we can hash/cache them" — a dict field makes `hash(ruleset)` raise. Nothing hashes it today, so
a dict would install a latent break of a documented property. A `dst_points_by_stat_id` property
returns the `Mapping` view for readers.

**2.2** `scoring/dst.py`: `score_dst(stat_vector, ruleset) -> float`, the dot product of §1.3.
The only place a D/ST stat vector becomes points.

**2.3** `DST_STAT_LABELS` in `scoring/dst.py` — the verified §1.4 id→label map, **display only**,
never consulted by `score_dst`.

**2.4** Tests:
- `hash(Ruleset(...))` works with `dst_stat_points` populated
- `score_dst` reproduces the Critts 26-category values
- a recorded ESPN fixture reconstructs `appliedTotal` to 1e-6

**2.5** `parse_ruleset`: read `pointsOverrides["16"]` into `dst_stat_points`; stop hiding items
whose base `points` is `0` but which carry a position override. Keep the K note.

---

## Phase 3 — ESPN ingest (additive)

**3.1** `ESPN_POSITIONS[16] = Position.DST.value`; admit DST past `external_projections.py:217`.
**3.2** Store the raw stat vector for DST rows; `gsis_id` from `DST_GSIS_IDS`.
**3.3** `id_map` rows per defense.

---

## Phase 4 — Sleeper ingest (additive)

**4.1** Admit `DEF`, map to `Position.DST`, `gsis_id` via `normalize_team_code` → `DST_GSIS_IDS`.
**4.2** The isolated name→id adapter (§4.2), including the straddling-bucket re-derivation.
**4.3** Cross-source agreement test against the ESPN line for the same team-week.

---

## Phase 5 — League config + schemas (**moves numbers — stop here for review**)

**5.1** `build_league_config` drops K only, not DST. Log message updated.
**5.2** Widen `ConsensusProjectionSchema` and `WaiverPoolSchema` to admit `Position.DST`.
**5.3** Regression test pinning the VORP invariant: skill `replacement_fpts` identical with and
without the DST slot (spec §5.1).
**5.4** **Produce the auction-value before/after diff on the real Critts config and stop for user
review** (spec §7.4).

---

## Phase 6 — Distributions + VORP

**6.1** Fit D/ST weekly variance from `nflreadpy.load_team_stats` (2018–2025) under the ruleset.
**6.2** Per-strength-tier parameter, not per team.
**6.3** Wire into the season projection path so `generate_vorp_table` sees a real distribution.

---

## Phase 7 — Downstream verification

Re-run `projected_standings`, `trade_analyzer`, `waiver_recommender` on live Critts data; confirm
the "N players not valued" counts drop by the expected number.
