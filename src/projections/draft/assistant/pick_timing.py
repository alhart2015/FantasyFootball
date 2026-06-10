"""Pure snake-draft pick-timing math. No pandas; hand-computable.

Picks are 1-based absolute pick numbers. Rounds are 0-based internally. Odd
0-based rounds run the slot order in reverse (the "snake").
"""

from __future__ import annotations


def slot_for(pick_number: int, n_teams: int) -> int:
    """Which 1-based slot owns an absolute pick under snake order."""
    if pick_number < 1:
        raise ValueError(f"pick_number must be >= 1; got {pick_number}")
    round_idx = (pick_number - 1) // n_teams
    offset = (pick_number - 1) % n_teams
    if round_idx % 2 == 0:
        return offset + 1
    return n_teams - offset


def _pick_number(round_idx: int, my_slot: int, n_teams: int) -> int:
    """Absolute pick number my slot holds in a given 0-based round."""
    if round_idx % 2 == 0:
        return round_idx * n_teams + my_slot
    return round_idx * n_teams + (n_teams - my_slot + 1)


def my_upcoming_picks(current_pick: int, my_slot: int, n_teams: int, rounds: int) -> list[int]:
    """My absolute pick numbers `>= current_pick` (current included if it's mine)."""
    return [p for r in range(rounds) if (p := _pick_number(r, my_slot, n_teams)) >= current_pick]


def my_next_pick(current_pick: int, my_slot: int, n_teams: int, rounds: int) -> int | None:
    """My first pick strictly after `current_pick`, or None if none remain."""
    later = [p for r in range(rounds) if (p := _pick_number(r, my_slot, n_teams)) > current_pick]
    return min(later) if later else None


def picks_until_next(current_pick: int, my_slot: int, n_teams: int, rounds: int) -> int | None:
    """Count of opponent picks strictly between this pick and my next one."""
    nxt = my_next_pick(current_pick, my_slot, n_teams, rounds)
    if nxt is None:
        return None
    return nxt - current_pick - 1
