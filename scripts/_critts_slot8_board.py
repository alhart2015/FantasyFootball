"""Pick-by-pick availability board for one snake draft slot.

For each of the hero's own picks, reports which players the market is likely to
leave on the board, using the same `LogisticSurvival` model the draft assistant's
`NowOrNeverStrategy` scores with -- so this board and the live assistant agree
about who survives to when.

Two numbers per player, and they answer different questions:

  p_here  P(still on the board AT this pick). What you can plan to take now.
  p_next  P(still on the board at my NEXT pick). What you can afford to wait on.

The gap between them is the whole point of a snake plan. At slot 8 of 16 the
wait between picks is 15 or 17 opponent selections -- more than a full round of
attrition -- so a player at p_here 0.9 / p_next 0.2 is a now-or-never pick even
though he looks safely available.

Usage:
    python scripts/_critts_slot8_board.py --top 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.draft.assistant.pick_timing import my_upcoming_picks
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.league_config import LeagueConfig

_LEAGUE = Path("data/leagues/critts_2025_2026")
_POOL = Path("data/vorp_2026/critts_half16_snake.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--my-slot", type=int, default=8, help="1-based draft slot.")
    p.add_argument("--top", type=int, default=8, help="Players to show per pick.")
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Survival spread in picks (default ~2/3 of a round).",
    )
    p.add_argument("--rounds", type=int, default=13, help="Rounds in the draft.")
    p.add_argument("--out", type=Path, default=_LEAGUE / "slot8_board.csv")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = LeagueConfig.model_validate_json((_LEAGUE / "league_config.json").read_text())
    n_teams = config.n_teams
    sigma = default_sigma(n_teams) if args.sigma is None else args.sigma
    survival = LogisticSurvival(sigma=sigma)

    sheet = pd.read_csv(_LEAGUE / "cheat_sheet.csv")
    # Null-ADP players are undrafted-in-market: the survival model has no opinion
    # on them, and treating a missing ADP as pick 0 would call them all gone.
    sheet = sheet[sheet["consensus_adp"].notna()].copy()

    picks = my_upcoming_picks(1, args.my_slot, n_teams, args.rounds)
    print(f"Slot {args.my_slot} of {n_teams}, {args.rounds} rounds, sigma={sigma:.2f} picks")
    print(f"My picks: {', '.join(str(p) for p in picks)}\n")

    rows: list[dict[str, object]] = []
    for i, pick in enumerate(picks):
        nxt = picks[i + 1] if i + 1 < len(picks) else None
        board = sheet.copy()
        # Bind `pick` / `nxt` as defaults: the lambdas close over the loop variables
        # otherwise, and every column would silently take the *last* pick's value if the
        # map were ever deferred (ruff B023).
        board["p_here"] = board["consensus_adp"].map(lambda a, at=pick: survival.p_available(a, at))
        board["p_next"] = (
            board["consensus_adp"].map(lambda a, at=nxt: survival.p_available(a, at))
            if nxt is not None
            else float("nan")
        )
        # Rank by VORP among players with a real chance of being there. A 0.5 floor
        # keeps the board to players it is worth *planning* around, rather than
        # listing elites who will be long gone.
        live = board[board["p_here"] >= 0.5].sort_values("vorp", ascending=False)
        gap = (nxt - pick - 1) if nxt is not None else 0
        header = f"Pick {pick} (round {i + 1})"
        header += f" -- {gap} opponent picks until pick {nxt}" if nxt else " -- last pick"
        print(header)
        if live.empty:
            print("  (nothing at p_here >= 0.50)\n")
            continue

        # Best-by-position first. Ranking the whole board by raw VORP returns an all-RB
        # list at every pick in this league, which is a true statement about VORP and a
        # useless one at the table: you start 2 RB and one FLEX, so a fourth RB cannot
        # enter your lineup at all. VORP ranks players; it has no idea what your roster
        # already holds. Showing each position's best survivor is what makes the real
        # trade-off visible at the pick.
        best = live.groupby("position").head(1).sort_values("vorp", ascending=False)
        cells = [
            f"{r.position} {r.display_name} ({r.vorp:+.0f}, p{r.p_here:.2f})"
            for r in best.itertuples()
        ]
        print("  best by position: " + " | ".join(cells))
        for row in live.head(args.top).itertuples():
            wait = "" if nxt is None else f"  p_next {row.p_next:.2f}"
            flag = ""
            if nxt is not None and row.p_here >= 0.6 and row.p_next < 0.25:
                flag = "  <- NOW OR NEVER"
            print(
                f"  {row.display_name:<24} {row.position:<3} "
                f"VORP {row.vorp:>6.1f}  ADP {row.consensus_adp:>6.1f}  "
                f"p_here {row.p_here:.2f}{wait}{flag}"
            )
            rows.append(
                {
                    "pick": pick,
                    "round": i + 1,
                    "display_name": row.display_name,
                    "position": row.position,
                    "vorp": row.vorp,
                    "consensus_adp": row.consensus_adp,
                    "p_here": row.p_here,
                    "p_next": row.p_next,
                }
            )
        print()

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Board written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
