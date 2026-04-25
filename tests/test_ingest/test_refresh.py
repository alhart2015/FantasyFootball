"""ingest.refresh orchestrator unit test -- verifies it fans out to every
per-source refresh function. We mock the source functions so this test does
not hit the network.

The plan text described `(seasons=, raw_root=)` signatures for every per-
source function, but the real signatures are `(data_root, *, seasons)`
positional + kwarg, and `refresh_ngs` additionally takes `stat_type` and so
must be invoked once per stat type. `build_id_map` does not take seasons at
all. The orchestrator therefore adapts to those real signatures.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

from projections.ingest import refresh


def test_refresh_calls_every_per_source_refresh_function(tmp_path: Path) -> None:
    seasons = [2018, 2019]

    with (
        patch("projections.ingest.refresh.refresh_weekly_stats") as ws,
        patch("projections.ingest.refresh.refresh_schedules") as sch,
        patch("projections.ingest.refresh.refresh_snap_counts") as sc,
        patch("projections.ingest.refresh.refresh_depth_charts") as dc,
        patch("projections.ingest.refresh.refresh_ngs") as ngs,
        patch("projections.ingest.refresh.build_id_map") as id_map,
    ):
        refresh(seasons=seasons, data_root=tmp_path)

        # Each per-season-table source: one call with the materialized list.
        ws.assert_called_once_with(tmp_path, seasons=seasons)
        sch.assert_called_once_with(tmp_path, seasons=seasons)
        sc.assert_called_once_with(tmp_path, seasons=seasons)
        dc.assert_called_once_with(tmp_path, seasons=seasons)

        # id_map has no seasons argument.
        id_map.assert_called_once_with(tmp_path)

        # NGS is invoked once per stat_type. Order doesn't matter; presence does.
        # `_Call` is not hashable, so we compare sorted lists of args/kwargs.
        assert ngs.call_count == 3
        expected_ngs_calls = [
            call(tmp_path, stat_type="passing", seasons=seasons),
            call(tmp_path, stat_type="rushing", seasons=seasons),
            call(tmp_path, stat_type="receiving", seasons=seasons),
        ]
        for expected in expected_ngs_calls:
            assert expected in ngs.call_args_list, (
                f"Missing expected NGS call: {expected}; got {ngs.call_args_list}"
            )


def test_refresh_builds_id_map_before_snap_counts(tmp_path: Path) -> None:
    """snap_counts ingest requires id_map.parquet to already exist on disk
    (it raises FileNotFoundError otherwise). The orchestrator must therefore
    invoke build_id_map before refresh_snap_counts."""
    call_order: list[str] = []

    with (
        patch("projections.ingest.refresh.refresh_weekly_stats") as ws,
        patch("projections.ingest.refresh.refresh_schedules") as sch,
        patch("projections.ingest.refresh.refresh_snap_counts") as sc,
        patch("projections.ingest.refresh.refresh_depth_charts") as dc,
        patch("projections.ingest.refresh.refresh_ngs") as ngs,
        patch("projections.ingest.refresh.build_id_map") as id_map,
    ):
        ws.side_effect = lambda *a, **kw: call_order.append("weekly_stats")
        sch.side_effect = lambda *a, **kw: call_order.append("schedules")
        sc.side_effect = lambda *a, **kw: call_order.append("snap_counts")
        dc.side_effect = lambda *a, **kw: call_order.append("depth_charts")
        ngs.side_effect = lambda *a, **kw: call_order.append("ngs")
        id_map.side_effect = lambda *a, **kw: call_order.append("id_map")

        refresh(seasons=[2024], data_root=tmp_path)

        assert call_order.index("id_map") < call_order.index("snap_counts"), (
            f"build_id_map must precede refresh_snap_counts; got order: {call_order}"
        )
