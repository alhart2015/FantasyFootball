import numpy as np
import pandas as pd

from projections.draft.assistant.auction.snake_bot import (
    DEFAULT_BROKE_ADP_JITTER,
    SnakeBoard,
    adp_usable,
)
from projections.schemas import Position


def _pool() -> pd.DataFrame:
    # gsis_ids MUST be canonical (\d{2}-\d{7}) — SnakeBoard.best_available routes through
    # _best_by_noisy_adp, which calls validate_gsis_id on the winner. Non-canonical ids raise.
    return pd.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000002", "00-0000003", "00-0000004", "00-0000005"],
            "position": ["RB", "RB", "WR", "QB", "TE"],
            "consensus_adp": pd.array([2.0, 8.0, 1.0, 40.0, 90.0], dtype="Float64"),
        }
    )


def test_adp_usable() -> None:
    assert adp_usable(_pool()) is True
    no_col = _pool().drop(columns=["consensus_adp"])
    assert adp_usable(no_col) is False
    all_null = _pool().assign(consensus_adp=pd.array([None] * 5, dtype="Float64"))
    assert adp_usable(all_null) is False


def test_best_available_respects_eligibility() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(0), adp_jitter=0.0)  # no noise -> pure ADP
    # WR (00-0000003) has lowest ADP overall, but if only RB is eligible we get the lowest-ADP RB.
    assert str(board.best_available(frozenset(), frozenset({Position.RB}))) == "00-0000001"
    assert str(board.best_available(frozenset(), frozenset({Position.WR}))) == "00-0000003"


def test_best_available_excludes_drafted() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(0), adp_jitter=0.0)
    pick = board.best_available(frozenset({"00-0000001"}), frozenset({Position.RB}))
    assert str(pick) == "00-0000002"


def test_best_available_none_when_empty() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(0), adp_jitter=0.0)
    # All RBs drafted, only RB eligible -> nothing left.
    assert (
        board.best_available(frozenset({"00-0000001", "00-0000002"}), frozenset({Position.RB}))
        is None
    )
    assert board.best_available(frozenset(), frozenset()) is None


def test_noise_is_fixed_per_board() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(7), adp_jitter=20.0)
    elig = frozenset({Position.RB, Position.WR, Position.QB, Position.TE})
    first = board.best_available(frozenset(), elig)
    # Same board, same query -> same answer every call (no re-draw).
    assert all(board.best_available(frozenset(), elig) == first for _ in range(5))


def test_order_independent_and_default_jitter() -> None:
    shuffled = _pool().iloc[::-1].reset_index(drop=True)
    b1 = SnakeBoard(_pool(), np.random.default_rng(3), adp_jitter=5.0)
    b2 = SnakeBoard(shuffled, np.random.default_rng(3), adp_jitter=5.0)
    elig = frozenset({Position.RB, Position.WR})
    assert b1.best_available(frozenset(), elig) == b2.best_available(frozenset(), elig)
    assert DEFAULT_BROKE_ADP_JITTER == 8.0
