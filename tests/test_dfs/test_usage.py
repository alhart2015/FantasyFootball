import pandas as pd

from projections.dfs.usage import build_usage


def test_touches_targets_is_carries_plus_targets() -> None:
    ws = pd.DataFrame(
        {
            "gsis_id": ["a", "b"],
            "season": [2023, 2023],
            "week": [1, 1],
            "carries": [10, 0],
            "targets": [2, 7],
        }
    )
    out = build_usage(ws).set_index("gsis_id")
    assert float(out.loc["a", "touches_targets"]) == 12.0
    assert float(out.loc["b", "touches_targets"]) == 7.0
