from __future__ import annotations

import pandas as pd

from projections.draft.backtest.inputs import _attach_is_rookie


def test_attach_is_rookie() -> None:
    pool = pd.DataFrame({"gsis_id": ["00-0000001", "00-0000002"], "position": ["WR", "RB"]})
    out = _attach_is_rookie(pool, prior_gsis={"00-0000001"})
    by = dict(zip(out["gsis_id"], out["is_rookie"], strict=True))
    assert bool(by["00-0000001"]) is False  # appeared before -> veteran
    assert bool(by["00-0000002"]) is True  # never appeared -> rookie


def test_attach_is_rookie_does_not_mutate_input() -> None:
    pool = pd.DataFrame({"gsis_id": ["00-0000003"], "position": ["TE"]})
    _attach_is_rookie(pool, prior_gsis=set())
    assert "is_rookie" not in pool.columns  # returns a new frame
