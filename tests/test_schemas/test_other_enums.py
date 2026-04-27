"""Tests for RosterSlot, DistributionFamily, Stat."""

from __future__ import annotations

import pytest

from projections.schemas import DistributionFamily, RosterSlot, Stat


def test_roster_slot_includes_super_flex() -> None:
    # Spec calls out superflex-readiness from day 1.
    assert RosterSlot.SUPER_FLEX.value == "SUPER_FLEX"
    assert RosterSlot.FLEX.value == "FLEX"


def test_roster_slot_has_bench_and_ir() -> None:
    assert RosterSlot.BENCH.value == "BENCH"
    assert RosterSlot.IR.value == "IR"


def test_distribution_family_options() -> None:
    assert {f.value for f in DistributionFamily} == {
        "NORMAL",
        "GAMMA",
        "NEGATIVE_BINOMIAL",
        "EMPIRICAL_QUANTILE",
        "SAMPLED",
        "SAMPLED_SUMMARY",
    }


@pytest.mark.parametrize(
    "stat",
    [
        Stat.PASSING_YARDS,
        Stat.PASSING_TDS,
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
        Stat.INTERCEPTIONS,
        Stat.FUMBLES_LOST,
    ],
)
def test_core_stats_exist(stat: Stat) -> None:
    assert isinstance(stat.value, str)
    assert stat.value.islower()  # column-name style
