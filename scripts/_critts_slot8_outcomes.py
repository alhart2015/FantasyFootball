"""Season outcomes for the slot-8 draft plan: win %, playoff %, championship %.

`draft_tournament.py compare` scores a roster; it does not say how often that roster wins
anything. This chains the two halves that were previously run apart:

  1. Simulate a full 16-team snake draft -- hero on `RawVorpStrategy` (the tournament
     winner), the other 15 seats on market ADP -- and reconstruct all 16 rosters.
  2. Run projected-vs-projected Monte-Carlo seasons over those rosters
     (`league_projection.simulate_seasons`) and record what each seat actually did.

Both halves are random, and they have to be sampled together. A single simulated draft
gives one roster, and its title odds are conditional on that roster; the question "how often
does the *plan* win" needs the draft resampled too. So the run is `--drafts` drafts x
`--sims` seasons each, and every rate below pools all of them.

Reported for the hero seat, plus the field mean as the no-skill baseline: with 16 teams and
6 playoff spots the field averages 37.5% playoffs and 6.25% titles by construction, so a
number is only meaningful against that.

Usage:
    python scripts/_critts_slot8_outcomes.py --drafts 12 --sims 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_projection import (
    N_BYES,
    PLAYOFF_SIZE,
    REG_WEEKS,
    simulate_seasons,
)
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.rookies import attach_is_rookie
from projections.draft.assistant.simulation import _draft_picks
from projections.draft.assistant.strategy import RawVorpStrategy
from projections.draft.assistant.survival import default_sigma
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema

_LEAGUE = Path("data/leagues/critts_2025_2026")
_POOL = Path("data/vorp_2026/critts_half16_snake.parquet")
_MY_SLOT = 8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--my-slot", type=int, default=_MY_SLOT)
    p.add_argument("--drafts", type=int, default=12, help="Distinct simulated drafts.")
    p.add_argument("--sims", type=int, default=200, help="MC seasons per draft.")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = LeagueConfig.model_validate_json((_LEAGUE / "league_config.json").read_text())
    pool = pd.read_parquet(_POOL)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    validated = VorpTableSchema.validate(pool)
    names = pool.set_index("gsis_id")["full_name"].to_dict()
    pos = pool.set_index("gsis_id")["position"].to_dict()
    vorp = pool.set_index("gsis_id")["vorp"].to_dict()

    enriched = attach_is_rookie(validated.copy(), season=args.season, data_root=Path("data"))
    availability = load_store_availability(enriched, season=args.season, data_root=Path("data"))
    params = VarianceParams.load()

    strategy = RawVorpStrategy()
    jitter = default_sigma(config.n_teams)
    n_teams = config.n_teams

    hero_win, hero_seed = [], []
    hero_playoff, hero_bye, hero_champ = [], [], []
    field_playoff, field_champ, field_win = [], [], []
    # One exemplar per outcome: (label -> (draft_idx, sim_idx, roster, wins, seed, pf))
    examples: dict[str, tuple[int, int, list[str], float, int, float]] = {}
    fixed_roster_spread: dict[str, float] = {}

    for d in range(args.drafts):
        rng = np.random.default_rng(args.seed + d)
        picks = _draft_picks(strategy, args.my_slot, validated, config, adp_jitter=jitter, rng=rng)
        rosters: dict[int, list[str]] = {s: [] for s in range(1, n_teams + 1)}
        for i, gid in enumerate(picks):
            rosters[slot_for(i + 1, n_teams)].append(gid)

        # `enriched`, not `validated`: VorpTableSchema.validate uses strict="filter", so it
        # strips the `is_rookie` column attach_is_rookie adds — and team_weekly_points needs
        # it. The draft half wants the validated frame; the season half wants the enriched one.
        res = simulate_seasons(
            rosters,
            enriched,
            availability,
            params,
            league_config=config,
            n_sims=args.sims,
            rng=np.random.default_rng(args.seed + 10_000 + d),
        )
        col = res.slots.index(args.my_slot)
        seeds = res.seed[:, col]
        hero_win.append(res.wins[:, col] / len(REG_WEEKS))
        hero_seed.append(seeds)
        hero_playoff.append(seeds <= PLAYOFF_SIZE)
        hero_bye.append(seeds <= N_BYES)
        hero_champ.append(res.champion == args.my_slot)
        field_playoff.append(res.seed <= PLAYOFF_SIZE)
        field_win.append(res.wins / len(REG_WEEKS))
        field_champ.append(np.stack([res.champion == s for s in res.slots], axis=1))

        labels = res.outcome_labels(args.my_slot)
        # Prefer an exemplar from a draft not already used, so the sample seasons show
        # different *rosters*. Without this every label fills from draft #0 -- its seasons
        # already contain all four outcomes -- and the samples would differ only by luck.
        used = {dd for dd, *_ in examples.values()}
        for i, label in enumerate(labels):
            if label in examples and (d in used or examples[label][0] not in used):
                continue
            examples[label] = (
                d,
                i,
                rosters[args.my_slot],
                float(res.wins[i, col]),
                int(seeds[i]),
                float(res.points_for[i, col]),
            )
            used.add(d)

        if d == 0:
            for lab in set(labels):
                fixed_roster_spread[lab] = labels.count(lab) / len(labels)

    hw = np.concatenate(hero_win)
    hs = np.concatenate(hero_seed)
    n = hw.size
    print(f"Slot {args.my_slot} of {n_teams} -- RawVorpStrategy")
    print(f"{args.drafts} simulated drafts x {args.sims} MC seasons = {n} seasons\n")
    print(f"{'metric':<24}{'hero':>10}{'field avg':>12}")
    rows = [
        ("regular-season win %", 100 * hw.mean(), 100 * np.concatenate(field_win).mean()),
        (
            f"make playoffs % (top {PLAYOFF_SIZE})",
            100 * np.concatenate(hero_playoff).mean(),
            100 * np.concatenate(field_playoff).mean(),
        ),
        (
            f"first-round bye % (top {N_BYES})",
            100 * np.concatenate(hero_bye).mean(),
            100 * N_BYES / n_teams,
        ),
        (
            "championship %",
            100 * np.concatenate(hero_champ).mean(),
            100 * np.concatenate(field_champ).mean(),
        ),
    ]
    for label, hero, field in rows:
        print(f"{label:<24}{hero:>9.1f}%{field:>11.1f}%")
    print(f"{'mean final seed':<24}{hs.mean():>10.2f}{(n_teams + 1) / 2:>12.2f}")

    if fixed_roster_spread:
        print()
        print("One FIXED roster (draft #0) -- every outcome it produced:")
        for lab, frac in sorted(fixed_roster_spread.items(), key=lambda kv: -kv[1]):
            print(f"  {lab:<28}{100 * frac:>6.1f}%")
        print("  (the draft sets your odds; it does not decide the season)")

    print("\n" + "=" * 78)
    print("SAMPLE SEASONS -- three outcomes, three different drafts")
    print("=" * 78)
    order = [
        "won championship",
        "lost championship",
        "made playoffs, eliminated",
        "missed playoffs",
    ]
    for label in order:
        if label not in examples:
            continue
        d, i, roster, wins, seed, pf = examples[label]
        print(f"\n--- {label.upper()} --- (draft #{d}, season #{i})")
        record = f"{wins:.0f}-{len(REG_WEEKS) - wins:.0f}"
        print(f"    {record} regular season, seed {seed}, {pf:.0f} pts")
        ranked = sorted(roster, key=lambda g: -vorp.get(g, 0.0))
        # `pid`, not `gid`: the draft loop above already binds `gid` as a GsisId, and reusing
        # the name here makes mypy infer this plain-str loop variable as GsisId too.
        for rnd, pid in enumerate(roster, start=1):
            print(
                f"    R{rnd:<3} {pos.get(pid, '?')!s:<3} {names.get(pid, pid)!s:<24}"
                f"VORP {vorp.get(pid, float('nan')):+7.1f}"
            )
        print(f"    roster VORP total: {sum(vorp.get(g, 0.0) for g in ranked):+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
