# Adoption gate report — _candidate_run vs _baseline_run

Run: baseline=`data\backtest\run_pre_plan9_baseline` candidate=`data\backtest\run_post_plan9`
n_bootstrap: 1000, seed: 42

### QB — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.012, +0.014] brackets / exceeds zero_

- n_paired: 2676; n_bootstrap: 1000
- RMSE delta: +0.0005 (95% CI [-0.0125, +0.0144])
- Spearman delta: -0.0001 (95% CI [-0.0029, +0.0024])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021       665         -0.003400      -0.027906       0.020518              0.001884          -0.003036           0.007129
 2022       657          0.005338      -0.025201       0.039698             -0.002814          -0.009282           0.003146
 2023       670          0.000018      -0.025505       0.024751              0.000488          -0.004763           0.005776
 2024       684          0.000323      -0.025819       0.025125              0.000119          -0.005380           0.005723

### RB — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.011, +0.011] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: +0.0001 (95% CI [-0.0110, +0.0111])
- Spearman delta: +0.0006 (95% CI [-0.0014, +0.0027])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315          0.011198      -0.014228       0.034775             -0.001033          -0.005589           0.003786
 2022      1331         -0.010195      -0.035607       0.012747              0.003819          -0.000388           0.008380
 2023      1311          0.000789      -0.018733       0.022225             -0.000652          -0.004463           0.002837
 2024      1316         -0.001633      -0.018346       0.013271              0.000167          -0.002702           0.003255

### TE — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [-0.012, +0.005] brackets / exceeds zero_

- n_paired: 4257; n_bootstrap: 1000
- RMSE delta: -0.0037 (95% CI [-0.0121, +0.0050])
- Spearman delta: +0.0000 (95% CI [-0.0028, +0.0027])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1030          0.016593      -0.001699       0.033981             -0.003525          -0.009352           0.002386
 2022      1088         -0.014582      -0.033131       0.004972             -0.000163          -0.006976           0.006073
 2023      1058         -0.011881      -0.026999       0.001999              0.000994          -0.003336           0.005796
 2024      1081         -0.005072      -0.022013       0.012954              0.002798          -0.001800           0.007498

### WR — _candidate_run vs _baseline_run: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.004, +0.012] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: +0.0083 (95% CI [+0.0043, +0.0124])
- Spearman delta: -0.0013 (95% CI [-0.0021, -0.0004])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109          0.018670       0.008141       0.030359             -0.002711          -0.004886          -0.000482
 2022      2102          0.002714      -0.005502       0.012172             -0.000663          -0.002379           0.000961
 2023      2201          0.006280       0.000050       0.012376             -0.001285          -0.002515          -0.000017
 2024      2048          0.005332       0.000547       0.010457             -0.000387          -0.001490           0.000614

Wrote CSV: reports\adoption_gate_plan9.csv
