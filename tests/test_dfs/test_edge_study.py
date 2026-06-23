import numpy as np
import pandas as pd

from projections.dfs import config
from projections.dfs import edge_study as es


def _frame(
    our: list[float],
    slp: list[float],
    actual: list[float],
    player_seasons: list[str] | None = None,
    positions: list[str] | None = None,
) -> pd.DataFrame:
    n = len(our)
    return pd.DataFrame(
        {
            "gsis_id": [f"g{i}" for i in range(n)],
            "season": [2023] * n,
            "week": list(range(1, n + 1)),
            "player_season": player_seasons or [f"g{i}-2023" for i in range(n)],
            "position": positions or ["WR"] * n,
            "our_pts": our,
            "sleeper_pts": slp,
            "actual_points": actual,
            "actual_points_with_bonus": actual,
        }
    )


def test_head_to_head_always_closer_is_one() -> None:
    # ours always closer to actual; disagreement large
    df = _frame(our=[10, 20, 30], slp=[0, 0, 0], actual=[11, 21, 31])
    assert es.head_to_head_fraction(df) == 1.0


def test_head_to_head_identical_sources_drops_all_ties() -> None:
    df = _frame(our=[10, 20], slp=[10, 20], actual=[5, 5])
    # no disagreement cells -> empty subset -> NaN (handled, not div-by-zero)
    assert np.isnan(es.head_to_head_fraction(df))


def test_clustered_bootstrap_wider_than_iid_on_correlated_data() -> None:
    # 10 player-seasons x 10 weeks. WITHIN a cluster the same source is closer
    # every week (perfect within-cluster correlation); ACROSS clusters it is 50/50.
    # The i.i.d. cell bootstrap sees ~100 "independent" cells (narrow CI); the
    # clustered bootstrap sees only 10 effective units (wide CI). Pins spec §7.2.3.
    rows = []
    for p in range(10):
        favor_ours = p % 2 == 0
        our, slp = (
            (51.0, 58.0) if favor_ours else (58.0, 51.0)
        )  # |err|=1 vs 8; disagreement=7>delta
        for w in range(10):
            rows.append(
                dict(
                    player_season=f"p{p}",
                    gsis_id=f"p{p}",
                    season=2023,
                    week=w + 1,
                    position="WR",
                    our_pts=our,
                    sleeper_pts=slp,
                    actual_points=50.0,
                )
            )
    df = pd.DataFrame(rows)

    clustered = es.clustered_bootstrap_fraction(df, seed=1)
    assert 0.0 <= clustered.lo_95 <= clustered.point <= clustered.hi_95 <= 1.0

    # i.i.d. cell bootstrap of the same head-to-head statistic, for comparison.
    closer = (
        (
            (df["our_pts"] - df["actual_points"]).abs()
            < (df["sleeper_pts"] - df["actual_points"]).abs()
        )
        .astype(float)
        .to_numpy()
    )
    rngb = np.random.default_rng(1)
    boot = np.array(
        [
            closer[rngb.integers(0, len(closer), len(closer))].mean()
            for _ in range(config.N_BOOTSTRAP)
        ]
    )
    iid_halfwidth = float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2
    clustered_halfwidth = (clustered.hi_95 - clustered.lo_95) / 2
    assert clustered_halfwidth > iid_halfwidth


def test_verdict_inconclusive_when_too_few_clusters() -> None:
    df = _frame(our=[11], slp=[0], actual=[10])  # 1 cluster << N_MIN
    res = es.run_edge_study_from_universe(df)
    assert res.verdict == "INCONCLUSIVE"


def test_ranking_diff_ci_clears_when_ours_strictly_better() -> None:
    # Across 12 player-seasons x 6 weeks, ours reproduces the actual ranking
    # monotonically (Spearman 1.0) while Sleeper inverts it (Spearman -1.0).
    # The clustered-bootstrap ranking-diff CI should sit strictly above 0.
    rows = []
    for p in range(12):
        for w in range(6):
            actual = float(w)
            rows.append(
                dict(
                    gsis_id=f"g{p}",
                    season=2023,
                    week=w + 1,
                    player_season=f"g{p}-2023",
                    position="WR",
                    our_pts=actual,  # perfectly tracks actual rank
                    sleeper_pts=-actual,  # inverted
                    actual_points=actual,
                    actual_points_with_bonus=actual,
                )
            )
    df = pd.DataFrame(rows)
    ci = es.ranking_skill_diff_ci(df, seed=1)
    assert ci.lo_95 >= 0
    assert ci.point > 0


def test_ranking_diff_ci_does_not_clear_with_no_signal() -> None:
    # Both sources are pure noise w.r.t. actual; the ranking-diff CI must not
    # exclude 0 on the low side (lo_95 < 0), so the gate condition fails.
    rng = np.random.default_rng(7)
    rows = []
    for p in range(20):
        for w in range(6):
            rows.append(
                dict(
                    gsis_id=f"g{p}",
                    season=2023,
                    week=w + 1,
                    player_season=f"g{p}-2023",
                    position="WR",
                    our_pts=float(rng.normal()),
                    sleeper_pts=float(rng.normal()),
                    actual_points=float(rng.normal()),
                    actual_points_with_bonus=0.0,
                )
            )
    df = pd.DataFrame(rows)
    ci = es.ranking_skill_diff_ci(df, seed=2)
    assert not (ci.lo_95 >= 0)  # the ranking condition does not clear


def test_ranking_diff_ci_degenerate_returns_nan_ci() -> None:
    # A single-cluster, <3-row universe makes Spearman undefined -> NaN CI,
    # which conservatively fails the gate (NaN >= 0 is False).
    df = _frame(our=[11.0, 12.0], slp=[0.0, 1.0], actual=[10.0, 11.0])
    ci = es.ranking_skill_diff_ci(df, seed=3)
    assert np.isnan(ci.lo_95)
    assert not (ci.lo_95 >= 0)


def test_inclusion_disagreement_counts_one_source_cells() -> None:
    key = ["gsis_id", "season", "week"]
    usage = pd.DataFrame(
        {
            "gsis_id": ["a", "b", "c"],
            "season": [2023] * 3,
            "week": [1, 1, 1],
            "touches_targets": [10, 10, 10],  # all above floor
        }
    )
    ours = pd.DataFrame({"gsis_id": ["a", "b"], "season": [2023, 2023], "week": [1, 1]})
    sleeper = pd.DataFrame({"gsis_id": ["b", "c"], "season": [2023, 2023], "week": [1, 1]})
    out = es.inclusion_disagreement(ours[key], sleeper[key], usage=usage)
    assert out == {"ours_only": 1, "sleeper_only": 1, "both": 1}  # a-only, c-only, b-both
