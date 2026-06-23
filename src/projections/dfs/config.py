"""Pre-registered constants for the DFS edge study. Fixed BEFORE computing the
verdict; never tuned to the outcome. Calibrated from a prior-year (e.g. 2020)
projection-difference + usage distribution — see the plan's Task 11 calibration.
"""

DELTA: float = 3.0  # DK-base-point disagreement threshold
USAGE_FLOOR_TOUCHES_TARGETS: int = 3  # actual (carries + targets) floor per cell
MARGIN_M: float = 0.05  # anti-masking: no position below 0.50 - m
N_MIN_CLUSTERS: int = 100  # min player-seasons in the disagreement subset
TARGET_CI_HALFWIDTH: float = 0.05  # else INCONCLUSIVE
N_BOOTSTRAP: int = 2000
BOOTSTRAP_SEED: int = 20260623
