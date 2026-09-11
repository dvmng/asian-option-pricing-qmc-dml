from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ComputeDatasetConfig:
    # Contract specification
    K_ref_usd_per_gpu_hour: float = 2.698470674716595
    T: float = 1.0
    n_fixings: int = 12

    # Physical-process calibration under P
    kappa_p: float = 32.66452619190628
    theta_p_log_K: float = 0.0
    sigma_p: float = 0.6247403836973323

    # Simulation design domain
    S0_K_min: float = 0.85
    S0_K_max: float = 1.175
    spot_K_min: float = 0.70
    spot_K_max: float = 1.30
    past_avg_K_min: float = 0.70
    past_avg_K_max: float = 1.30

    tau_min: float = 0.05
    tau_max: float = 1.0
    r_min: float = 0.0
    r_max: float = 0.08

    kappa_q_min: float = 25.0
    kappa_q_max: float = 87.0
    theta_q_log_K_min: float = -0.16413337819792415
    theta_q_log_K_max: float = 0.16104639212511673
    sigma_q_min: float = 0.55
    sigma_q_max: float = 0.71

    # Dataset sizes
    n_train: int = 2**16
    n_val: int = 2**13
    n_test: int = 2**13
    n_reference: int = 2**10

    # Pricing-label simulation budgets
    paths_per_scenario: int = 2**13
    label_replications: int = 1
    label_engine: str = "rqmc"

    reference_paths_per_replication: int = 2**13
    reference_replications: int = 16
    reference_engine: str = "rqmc"

    # Independent seed namespaces for each data split
    seed_train_outer: int = 41001
    seed_train_history: int = 41101
    seed_train_pricing: int = 41201

    seed_val_outer: int = 42001
    seed_val_history: int = 42101
    seed_val_pricing: int = 42201

    seed_test_outer: int = 43001
    seed_test_history: int = 43101
    seed_test_pricing: int = 43201

    seed_reference_outer: int = 44001
    seed_reference_history: int = 44101
    seed_reference_pricing: int = 44201

    @property
    def fixing_times(self):
        return [self.T * j / self.n_fixings for j in range(1, self.n_fixings + 1)]

    def as_dict(self) -> Dict:
        return asdict(self)


def smoke_config() -> ComputeDatasetConfig:
    """Small deterministic pricing-dataset smoke preset."""
    return ComputeDatasetConfig(
        n_train=256,
        n_val=64,
        n_test=64,
        n_reference=32,
        paths_per_scenario=2**10,
        label_replications=1,
        reference_paths_per_replication=2**10,
        reference_replications=4,
    )


INPUT_COLUMNS = (
    "spot_K",
    "fixed_avg_contrib_K",
    "tau",
    "r",
    "kappa_q",
    "theta_q_log_K",
    "sigma_q",
)

TARGET_PRICE = "price_K"
TARGET_DELTA = "delta"
SPOT_INPUT_INDEX = 0
