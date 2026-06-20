"""Tests for the ADP-bot pick policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import _best_by_noisy_adp, bot_pick
from projections.schemas import _PYARROW_STR


def _available(rows: list[tuple[str, float | None]]) -> pd.DataFrame:
    """rows = [(gsis_id, consensus_adp_or_None), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "consensus_adp": pd.array(
                [r[1] if r[1] is not None else pd.NA for r in rows], dtype=pd.Float64Dtype()
            ),
        }
    )


def _available_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def test_zero_jitter_picks_lowest_adp() -> None:
    avail = _available([("00-0000001", 10.0), ("00-0000002", 3.0), ("00-0000003", 7.0)])
    rng = np.random.default_rng(0)
    assert bot_pick(avail, rng, adp_jitter=0.0) == "00-0000002"


def test_null_adp_left_for_hero_until_nothing_else() -> None:
    avail = _available([("00-0000001", None), ("00-0000002", 50.0)])
    # Even with big jitter, a finite ADP always beats +inf.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert bot_pick(avail, rng, adp_jitter=10.0) == "00-0000002"


def test_all_null_falls_back_to_gsis_order() -> None:
    avail = _available([("00-0000002", None), ("00-0000001", None)])
    rng = np.random.default_rng(0)
    # All +inf -> deterministic gsis_id tie-break (ascending).
    assert bot_pick(avail, rng, adp_jitter=5.0) == "00-0000001"


def test_deterministic_given_seed() -> None:
    avail = _available([("00-0000001", 5.0), ("00-0000002", 5.5), ("00-0000003", 6.0)])
    a = bot_pick(_available_copy(avail), np.random.default_rng(42), adp_jitter=3.0)
    b = bot_pick(_available_copy(avail), np.random.default_rng(42), adp_jitter=3.0)
    assert a == b


def test_result_independent_of_row_order() -> None:
    rows: list[tuple[str, float | None]] = [
        ("00-0000001", 5.0),
        ("00-0000002", 5.1),
        ("00-0000003", 4.9),
    ]
    avail_ab = _available(rows)
    avail_ba = _available(list(reversed(rows)))
    for seed in range(50):
        a = bot_pick(avail_ab.copy(), np.random.default_rng(seed), adp_jitter=3.0)
        b = bot_pick(avail_ba.copy(), np.random.default_rng(seed), adp_jitter=3.0)
        assert a == b, f"seed {seed}: row order changed result"


def test_single_player_available() -> None:
    avail = _available([("00-0000001", 5.0)])
    assert bot_pick(avail, np.random.default_rng(0), adp_jitter=99.0) == "00-0000001"


def test_bot_pick_characterization_stable_across_refactor() -> None:
    # Pins bot_pick's exact picks for fixed seeds so the Task 1 extraction is proven byte-identical.
    avail = pd.DataFrame(
        {
            "gsis_id": ["00-0000005", "00-0000001", "00-0000003", "00-0000002", "00-0000004"],
            "consensus_adp": pd.array([12.0, 3.0, None, 3.0, 50.0], dtype="Float64"),
        }
    )
    picks = [str(bot_pick(avail, np.random.default_rng(seed), adp_jitter=2.0)) for seed in range(6)]
    expected = [
        "00-0000002",
        "00-0000001",
        "00-0000002",
        "00-0000002",
        "00-0000001",
        "00-0000002",
    ]
    assert picks == expected


def test_best_by_noisy_adp_argmin_and_tiebreak() -> None:
    # Canonical gsis_ids (validate_gsis_id requires \d{2}-\d{7}); ascending order is a < b < c.
    a, b, c = "00-0000001", "00-0000002", "00-0000003"
    gsis = np.array([c, a, b], dtype=str)
    noisy = np.array([5.0, 5.0, 2.0], dtype=float)
    assert str(_best_by_noisy_adp(gsis, noisy)) == b  # lowest noisy
    tie = np.array([2.0, 2.0, 2.0], dtype=float)
    assert str(_best_by_noisy_adp(gsis, tie)) == a  # gsis-ascending tiebreak
    inf = np.array([np.inf, 1.0, np.inf], dtype=float)
    assert str(_best_by_noisy_adp(gsis, inf)) == a  # finite beats inf
