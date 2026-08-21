from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TFMConfig:
    # Contract
    K: float = 1.0
    T: float = 1.0
    n_fixings: int = 12
    q: float = 0.0

    # State-generation domain
    S0_K_min: float = 0.70
    S0_K_max: float = 1.30
    S_t_K_min: float = 0.60
    S_t_K_max: float = 1.40
    A_t_K_min: float = 0.60
    A_t_K_max: float = 1.40

    tau_min: float = 0.05
    tau_max: float = 1.00
    r_min: float = 0.00
    r_max: float = 0.08
    sigma_min: float = 0.10
    sigma_max: float = 0.40

    # Main planned dataset sizes
    n_train: int = 2**16
    n_val: int = 2**13
    n_test: int = 2**13
    n_reference: int = 2**10

    # Ordinary labels
    paths_per_scenario: int = 2**13
    label_replications: int = 1
    label_engine: str = "rqmc"  # "mc" or "rqmc"

    # Reference labels: 8 x 8192 = 65536 paths per state
    reference_paths_per_replication: int = 2**13
    reference_replications: int = 16
    reference_engine: str = "rqmc"

    # Reproducibility
    seed_train_outer: int = 11001
    seed_val_outer: int = 22001
    seed_test_outer: int = 33001
    seed_reference_outer: int = 44001

    seed_train_history: int = 51001
    seed_val_history: int = 52001
    seed_test_history: int = 53001
    seed_reference_history: int = 54001

    seed_train_pricing: int = 61001
    seed_val_pricing: int = 62001
    seed_test_pricing: int = 63001
    seed_reference_pricing: int = 64001

    @property
    def fixing_times(self):
        return [self.T * j / self.n_fixings for j in range(1, self.n_fixings + 1)]

    def as_dict(self):
        return asdict(self)


def pilot_config() -> TFMConfig:
    return TFMConfig(
        n_train=2**10,
        n_val=2**8,
        n_test=2**8,
        n_reference=2**6,
        paths_per_scenario=2**10,
        reference_paths_per_replication=2**11,
        reference_replications=4,
    )
