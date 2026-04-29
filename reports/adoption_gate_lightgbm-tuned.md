# Adoption gate report — lightgbm-tuned vs baseline

Run: `data\backtest\run_20260429T003552Z`
n_bootstrap: 1000, seed: 42

### QB — lightgbm-tuned vs baseline: **ADOPT**

_RMSE delta -0.119 (95% CI [-0.206, -0.031]); Spearman lo_95 +0.0046 > floor -0.020_

- n_paired: 2676; n_bootstrap: 1000
- RMSE delta: -0.1189 (95% CI [-0.2063, -0.0310])
- Spearman delta: +0.0177 (95% CI [+0.0046, +0.0304])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021       665         -0.102185      -0.278090       0.091177              0.034518           0.007691           0.061961
 2022       657         -0.104105      -0.275694       0.064750              0.010699          -0.016662           0.037198
 2023       670         -0.048998      -0.237404       0.148976             -0.000385          -0.025576           0.022605
 2024       684         -0.215055      -0.354270      -0.070605              0.025926           0.003653           0.047294

### RB — lightgbm-tuned vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.080, +0.152] brackets / exceeds zero_

- n_paired: 5273; n_bootstrap: 1000
- RMSE delta: +0.1144 (95% CI [+0.0798, +0.1520])
- Spearman delta: -0.0043 (95% CI [-0.0098, +0.0009])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1315          0.140875       0.058088       0.222567             -0.023104          -0.034343          -0.010544
 2022      1331          0.067278      -0.006352       0.145153              0.009485          -0.001896           0.021353
 2023      1311          0.111278       0.050464       0.175530             -0.000880          -0.011950           0.009335
 2024      1316          0.138169       0.079492       0.198048             -0.002507          -0.011846           0.005899

### TE — lightgbm-tuned vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.047, +0.132] brackets / exceeds zero_

- n_paired: 4257; n_bootstrap: 1000
- RMSE delta: +0.0879 (95% CI [+0.0468, +0.1322])
- Spearman delta: +0.0082 (95% CI [-0.0003, +0.0170])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      1030          0.047018      -0.035375       0.134718              0.019971          -0.001517           0.040207
 2022      1088          0.069891      -0.014148       0.159471             -0.000282          -0.018133           0.019221
 2023      1058          0.084973       0.004861       0.160450              0.018130           0.003787           0.034015
 2024      1081          0.149793       0.074783       0.232220             -0.005030          -0.020143           0.009987

### WR — lightgbm-tuned vs baseline: **DO_NOT_ADOPT**

_RMSE inconclusive: 95% CI [+0.040, +0.105] brackets / exceeds zero_

- n_paired: 8460; n_bootstrap: 1000
- RMSE delta: +0.0711 (95% CI [+0.0397, +0.1046])
- Spearman delta: +0.0044 (95% CI [-0.0012, +0.0099])

Per-year breakdown (informational):

 year  n_paired  rmse_delta_point  rmse_delta_lo  rmse_delta_hi  spearman_delta_point  spearman_delta_lo  spearman_delta_hi
 2021      2109          0.069178       0.004512       0.132821              0.009710          -0.001279           0.020424
 2022      2102          0.089402       0.019538       0.155606              0.004592          -0.006079           0.014825
 2023      2201          0.076294       0.008699       0.145611              0.000656          -0.010582           0.011810
 2024      2048          0.049090      -0.005311       0.108258              0.002714          -0.008208           0.012770

Wrote CSV: reports\adoption_gate_lightgbm-tuned.csv
