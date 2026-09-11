from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from .asian_logou import ComputeAsianConfig, ComputeAsianState
from .logou import LogOUParams, exact_step_log
from .phase4_config import ComputeDatasetConfig
from .seed_utils import state_seed as make_state_seed


_TIME_TOL = 1e-12


@dataclass(frozen=True)
class ComputeDatasetState:
    scenario_id: str
    split: str
    S0_K: float
    spot_K: float
    fixed_avg_contrib_K: float
    past_avg_K: float
    t: float
    tau: float
    n_fix_past: int
    n_fix_future: int
    r: float
    kappa_q: float
    theta_q_log_K: float
    sigma_q: float
    kappa_p: float
    theta_p_log_K: float
    sigma_p: float
    state_seed: int
    pricing_seed: int

    def to_dict(self) -> Dict:
        return asdict(self)

    def pricing_state(self) -> ComputeAsianState:
        return ComputeAsianState(
            scenario_id=self.scenario_id,
            spot_K=self.spot_K,
            fixed_avg_contrib_K=self.fixed_avg_contrib_K,
            t=self.t,
            tau=self.tau,
            n_fix_past=self.n_fix_past,
            n_fix_future=self.n_fix_future,
            r=self.r,
            kappa_q=self.kappa_q,
            theta_q_log_K=self.theta_q_log_K,
            sigma_q=self.sigma_q,
            pricing_seed=self.pricing_seed,
        )


def _uniform(u: float, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * float(u))


def simulate_physical_history_to_t(
    S0_K: float,
    t: float,
    cfg: ComputeDatasetConfig,
    seed: int,
) -> Tuple[float, np.ndarray]:
    """One coherent realized history under the frozen physical Log-OU process."""
    fixings = np.asarray(cfg.fixing_times, dtype=np.float64)
    past_times = fixings[fixings <= t + _TIME_TOL]

    params = LogOUParams(
        kappa=cfg.kappa_p,
        theta_log_K=cfg.theta_p_log_K,
        sigma=cfg.sigma_p,
    )
    rng = np.random.default_rng(seed)

    x = float(np.log(S0_K))
    prev_t = 0.0
    past_values = []

    for fixing_t in past_times:
        dt = float(fixing_t - prev_t)
        z = float(rng.standard_normal())
        x = float(exact_step_log(x, dt, params, z))
        past_values.append(float(np.exp(x)))
        prev_t = float(fixing_t)

    if t > prev_t + _TIME_TOL:
        dt = float(t - prev_t)
        z = float(rng.standard_normal())
        x = float(exact_step_log(x, dt, params, z))

    return float(np.exp(x)), np.asarray(past_values, dtype=np.float64)


def build_state_from_candidate(
    u6: np.ndarray,
    split: str,
    candidate_index: int,
    accepted_index: int,
    cfg: ComputeDatasetConfig,
    history_seed_base: int,
    pricing_seed_base: int,
) -> Optional[ComputeDatasetState]:
    # Sobol coordinates cover only exogenous scenario variables.
    S0_K = _uniform(u6[0], cfg.S0_K_min, cfg.S0_K_max)
    tau = _uniform(u6[1], cfg.tau_min, cfg.tau_max)
    r = _uniform(u6[2], cfg.r_min, cfg.r_max)
    kappa_q = _uniform(u6[3], cfg.kappa_q_min, cfg.kappa_q_max)
    theta_q_log_K = _uniform(
        u6[4], cfg.theta_q_log_K_min, cfg.theta_q_log_K_max
    )
    sigma_q = _uniform(u6[5], cfg.sigma_q_min, cfg.sigma_q_max)

    t = cfg.T - tau
    state_seed = make_state_seed(history_seed_base, candidate_index)
    pricing_seed = make_state_seed(pricing_seed_base, accepted_index)

    spot_K, past_fixings_K = simulate_physical_history_to_t(
        S0_K=S0_K,
        t=t,
        cfg=cfg,
        seed=state_seed,
    )

    n_past = int(len(past_fixings_K))
    n_future = int(cfg.n_fixings - n_past)
    fixed_avg_contrib_K = float(past_fixings_K.sum() / cfg.n_fixings)
    past_avg_K = (
        float(past_fixings_K.mean()) if n_past > 0 else float("nan")
    )

    # Apply acceptance filters only after constructing a coherent simulated history.
    if not (cfg.spot_K_min <= spot_K <= cfg.spot_K_max):
        return None
    if n_past > 0 and not (
        cfg.past_avg_K_min <= past_avg_K <= cfg.past_avg_K_max
    ):
        return None

    state = ComputeDatasetState(
        scenario_id=f"{split}_{accepted_index:07d}",
        split=split,
        S0_K=S0_K,
        spot_K=spot_K,
        fixed_avg_contrib_K=fixed_avg_contrib_K,
        past_avg_K=past_avg_K,
        t=t,
        tau=tau,
        n_fix_past=n_past,
        n_fix_future=n_future,
        r=r,
        kappa_q=kappa_q,
        theta_q_log_K=theta_q_log_K,
        sigma_q=sigma_q,
        kappa_p=cfg.kappa_p,
        theta_p_log_K=cfg.theta_p_log_K,
        sigma_p=cfg.sigma_p,
        state_seed=state_seed,
        pricing_seed=pricing_seed,
    )

    state.pricing_state().validate(
        ComputeAsianConfig(T=cfg.T, n_fixings=cfg.n_fixings)
    )
    return state


def generate_states(
    n_states: int,
    split: str,
    cfg: ComputeDatasetConfig,
    outer_seed: int,
    history_seed_base: int,
    pricing_seed_base: int,
    oversampling_factor: float = 1.35,
) -> list[ComputeDatasetState]:
    if n_states <= 0:
        raise ValueError("n_states must be positive.")

    sobol = qmc.Sobol(d=6, scramble=True, seed=int(outer_seed))
    states: list[ComputeDatasetState] = []
    candidate_index = 0

    while len(states) < n_states:
        remaining = n_states - len(states)
        target = max(256, int(np.ceil(remaining * oversampling_factor)))
        batch_n = 1 << int(np.ceil(np.log2(target)))

        # Acceptance filtering breaks digital-net balance; use Sobol random() here.
        U = sobol.random(batch_n)

        for u6 in U:
            state = build_state_from_candidate(
                u6=u6,
                split=split,
                candidate_index=candidate_index,
                accepted_index=len(states),
                cfg=cfg,
                history_seed_base=history_seed_base,
                pricing_seed_base=pricing_seed_base,
            )
            candidate_index += 1

            if state is not None:
                states.append(state)
                if len(states) >= n_states:
                    break

    return states
