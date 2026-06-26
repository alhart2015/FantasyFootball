# RB usage-cut diagnostic — does benchwarmer depth dilute the STOP? (#55)

**Date:** 2026-06-25. Follow-up to the #55 STOP verdict, testing the natural objection:
*"We lose to Sleeper / the features don't adopt only because the evaluation is diluted by
low-usage RBs — focus on top-tier RBs and the signal appears."* **Both forms refuted.**

## Cut 1 — our model vs Sleeper, by usage (edge-study fraction)

Re-cut RB's disagreement head-to-head fraction (share of |ours−Sleeper|>3 DK-pt cells where our
projection is closer to actual; >0.50 = we beat Sleeper) from the persisted universe
`data/dfs_universe_2021-2024_floor3_*.parquet`, at rising usage/projection thresholds.
Clustered-bootstrap 95% CIs (seed 20260623).

| restrict RB to | cells | fraction | 95% CI | verdict |
|---|---:|---:|---|---|
| touches ≥ 3 (current floor) | 4272 | 0.430 | [0.397, 0.463] | deficit |
| touches ≥ 8 | 2988 | 0.464 | [0.424, 0.504] | n.s. (best, still <0.50) |
| touches ≥ 12 | 2235 | 0.443 | [0.398, 0.484] | deficit |
| touches ≥ 15 | 1703 | 0.407 | [0.357, 0.457] | deficit |
| touches ≥ 18 (bellcows) | 1135 | 0.386 | [0.325, 0.449] | deficit |
| proj_max ≥ 12 (rosterable) | 1760 | 0.406 | [0.356, 0.453] | deficit |
| our_pts ≥ 12 | 1496 | 0.415 | [0.355, 0.471] | deficit |

**Refuted.** The fraction does not climb toward 0.50 as we focus on startable RBs — it stays
~0.40–0.46 and gets *worse* at the high-volume end (bellcows 0.386). We lose to Sleeper **most**
on exactly the RBs you'd roster. Hypothesis (consensus has the most situational edge — game
script, goal-line role, vulture upside — on bellcows, which our box-score model can't see).

## Cut 2 — do the trajectory-trend features help the high-volume case? (feature gate)

The G2 DO_NOT_ADOPT was composite RMSE over all RBs. Tested whether the two trend features help
high-volume RBs specifically, by emitting RB-baseline per-row predictions on the old cache vs the
augmented cache (`scripts/_rb_perrow.py`, throwaway; aug run via a temporary `git checkout 44bfb5f`
of the schema/builder/`_RB_FEATURE_COLUMNS`) and stratifying composite error by actual touches.
Δ = aug − old (negative = trend features help):

| RB band | n | RMSE old → aug | ΔRMSE | ΔMAE |
|---|---:|---|---:|---:|
| ALL | 5261 | 6.568 → 6.566 | −0.002 | −0.028 |
| touches ≥ 8 | 2988 | 7.600 → 7.616 | **+0.016 (worse)** | +0.016 |
| touches ≥ 12 | 2235 | 8.133 → 8.138 | +0.004 (worse) | +0.005 |
| touches ≥ 15 | 1703 | 8.629 → 8.623 | −0.006 | −0.006 |
| touches ≥ 18 (bellcows) | 1135 | 9.354 → 9.327 | −0.027 | −0.020 |
| touches < 8 | 1279 | 4.859 → 4.856 | −0.003 | **−0.049** |

**Refuted.** The trend features do not help the high-volume case: mid-volume (8–14 touches) is
slightly *worse*, and the bellcow help (−0.027 RMSE on a 9.35 base, ~0.3%) is far below the
0.05-fpt effect floor. The tiny aggregate MAE gain runs the *opposite* way — it's concentrated in
the **low-usage** band (touches <8: ΔMAE −0.049). So the no-go was not a depth-dilution artifact.
Mechanistically: `carries_per_game_l4`/`targets_per_game_l4` already capture recent volume, so
`volume_trend` is a noisy second-derivative that adds little, swamped by the dropna training
penalty and the five box-score stats that NULL'd.

## Conclusion

Both dilution hypotheses are refuted. RB has no high-volume path via the current box-score baseline
or the candidate trajectory-trend features — the deficit vs Sleeper is real and concentrated on the
RBs that matter, and the new features don't move it there. **STOP stands for RB.** The only untried
lever is Option C — a model that ingests opportunity/role/game-script signal the box-score baseline
is blind to — a research bet, logged in #55, deprioritized vs the #52 draft work.
