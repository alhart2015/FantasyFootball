# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_pre_te_traj_baseline` candidate=`data\backtest\run_post_te_traj_baseline`
n_bootstrap: 1000, seed: 42

### TE — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.028, +0.009] brackets / exceeds zero_

- n_paired: 4257; n_bootstrap: 1000
- RMSE delta: -0.0100 (95% CI [-0.0280, +0.0093])
- Spearman delta: +0.0018 (95% CI [-0.0033, +0.0071])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1030         -0.016948      -0.068580       0.034428              0.006234          -0.008082           0.020099
 2022      1088         -0.013825      -0.048554       0.021000             -0.001416          -0.012713           0.009168
 2023      1058         -0.021964      -0.048921       0.003221              0.004823          -0.002407           0.012670
 2024      1081          0.012288      -0.021922       0.042093             -0.002371          -0.009514           0.005787

Wrote CSV: reports\adoption_gate_te_trajectory_baseline.csv
