# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_baseline_baseline` candidate=`data\backtest\run_candidate_baseline`
n_bootstrap: 1000, seed: 42

### RB — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.012, +0.016] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: +0.0020 (95% CI [-0.0117, +0.0156])
- Spearman delta: -0.0015 (95% CI [-0.0040, +0.0010])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315         -0.000346      -0.033209       0.033215             -0.001069          -0.006498           0.003991
 2022      1331          0.008754      -0.016569       0.036139             -0.005389          -0.010936          -0.000625
 2023      1311         -0.007905      -0.028131       0.017088              0.001797          -0.002588           0.005687
 2024      1316          0.007180      -0.016367       0.031136             -0.001441          -0.006033           0.003264

Wrote CSV: reports\adoption_gate_weather_refined_baseline_rb.csv
