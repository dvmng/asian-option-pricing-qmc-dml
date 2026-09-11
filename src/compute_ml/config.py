from __future__ import annotations

from typing import Tuple

INPUT_COLUMNS: Tuple[str, ...] = (
    "spot_K",
    "fixed_avg_contrib_K",
    "tau",
    "r",
    "kappa_q",
    "theta_q_log_K",
    "sigma_q",
)
PRICE_COLUMN = "price_K"
DELTA_COLUMN = "delta"
SCENARIO_ID_COLUMN = "scenario_id"
SPOT_INPUT_INDEX = 0

# Final architecture and optimizer configuration shared by MLP and DML.
SHARED_MODEL_PARAMS = {
    "backbone": "residual",
    "width": 128,
    "n_blocks": 3,
    "activation": "silu",
    "optimizer": "adamw",
    "learning_rate": 0.0019615987173439097,
    "scheduler": "cosine",
    "weight_decay": 6.634059887871314e-07,
}

EXPECTED_PARAMETER_COUNT = 100_225

# Candidate derivative-loss weights used for DML selection.
DML_LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDA_SEEDS = (1701, 3701, 5701)

# Lambda screening uses a fixed architecture and optimizer.
LAMBDA_TRAIN_SIZE = 2**13
LAMBDA_MAX_EPOCHS = 1500
BATCH_SIZE = 1024

# Fixed nested-sample ordering seed used across final model fits.
TRAIN_ORDER_SEED = 20260822

# Validation is split into disjoint checkpoint-selection and lambda-selection subsets.
VAL_ROLE_SPLIT_SEED = 20260826

FINAL_TRAIN_SIZES = (2**10, 2**12, 2**14, 2**16)
FINAL_TRAINING_SEEDS = (1701, 2701, 3701, 4701, 5701)
FINAL_MAX_EPOCHS = 3000
CHECKPOINT_METRIC = "val_price_mse_std"
