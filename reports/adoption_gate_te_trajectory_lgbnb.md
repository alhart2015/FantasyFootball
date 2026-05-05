# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_pre_te_traj_lightgbm_nb` candidate=`data\backtest\run_post_te_traj_lightgbm_nb`
n_bootstrap: 1000, seed: 42

### TE — _candidate_run vs _baseline_run: **ADOPT**

_RMSE delta -0.009 (95% CI [-0.017, -0.001]); Spearman lo_95 +0.0001 > floor -0.020_

- n_paired: 4257; n_bootstrap: 1000
- RMSE delta: -0.0090 (95% CI [-0.0171, -0.0013])
- Spearman delta: +0.0028 (95% CI [+0.0001, +0.0055])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1030          0.003592      -0.011364       0.017650             -0.000097          -0.005598           0.005554
 2022      1088         -0.013904      -0.031766       0.002006              0.006666           0.000006           0.012698
 2023      1058         -0.015123      -0.028864      -0.001273              0.003019          -0.001779           0.008396
 2024      1081         -0.010640      -0.028270       0.005311              0.001720          -0.003300           0.007648

Wrote CSV: reports\adoption_gate_te_trajectory_lgbnb.csv
