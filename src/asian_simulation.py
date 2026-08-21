from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from config import TFMConfig


_EPS_U = 1e-12
_TIME_TOL = 1e-12


@dataclass
class AsianState:
    """
    State of a discretely monitored arithmetic Asian call.

    All price quantities are normalized by K. K=1 in the main experiment.
    C_t is the contribution already fixed in the final arithmetic average:

        C_t = (1/N) * sum_{fixing <= t} (S_fixing / K)

    Therefore the final normalized average is:

        A_T / K = C_t + (1/N) * sum_future (S_fixing / K)
    """
    scenario_id: str
    split: str
    S0_K: float
    S_t_K: float
    C_t: float
    A_t_K: float
    t: float
    tau: float
    n_fix_past: int
    n_fix_future: int
    r: float
    sigma: float
    q: float
    state_seed: int
    pricing_seed: int

    def to_dict(self) -> Dict:
        return asdict(self)


def _uniform_map(u: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(u)


def _gbm_step(
    s: float,
    dt: float,
    r: float,
    sigma: float,
    z: float,
    q_yield: float = 0.0,
) -> float:
    if dt <= 0.0:
        return s
    return float(
        s
        * np.exp(
            (r - q_yield - 0.5 * sigma * sigma) * dt
            + sigma * np.sqrt(dt) * z
        )
    )


def simulate_history_to_t(
    S0_K: float,
    t: float,
    r: float,
    sigma: float,
    cfg: TFMConfig,
    seed: int,
) -> Tuple[float, np.ndarray]:
    """
    Simulate a single economically consistent past trajectory under GBM.

    The path is advanced chronologically through the already observed fixing
    dates and then, if needed, from the last fixing to the current valuation
    time t. No floating-point time is used as a dictionary key.
    """
    fixings = np.asarray(cfg.fixing_times, dtype=float)
    past_times = fixings[fixings <= t + _TIME_TOL]

    rng = np.random.default_rng(seed)
    s = float(S0_K)
    prev_t = 0.0
    past_values = []

    # Simulate every fixing already observed.
    for fixing_t in past_times:
        dt = float(fixing_t - prev_t)
        z = float(rng.standard_normal())
        s = _gbm_step(s, dt, r, sigma, z, cfg.q)
        past_values.append(s)
        prev_t = float(fixing_t)

    # If t lies between fixing dates, simulate the remaining partial interval.
    if t > prev_t + _TIME_TOL:
        dt = float(t - prev_t)
        z = float(rng.standard_normal())
        s = _gbm_step(s, dt, r, sigma, z, cfg.q)

    S_t_K = float(s)
    past_fixings_K = np.asarray(past_values, dtype=float)
    return S_t_K, past_fixings_K


def _namespaced_seed(base_seed: int, index: int) -> int:
    """
    Create deterministic, non-overlapping seed namespaces across dataset splits.

    As long as index < 1e9 (far above the planned experiment size), two
    different base_seed values cannot generate the same integer seed.
    """
    if index < 0 or index >= 1_000_000_000:
        raise ValueError("Seed index must satisfy 0 <= index < 1e9.")
    return int(base_seed) * 1_000_000_000 + int(index)


def build_state_from_candidate(
    u4: np.ndarray,
    split: str,
    candidate_index: int,
    accepted_index: int,
    cfg: TFMConfig,
    history_seed_base: int,
    pricing_seed_base: int,
) -> Optional[AsianState]:
    """
    Convert one 4-D design point into an economically consistent Asian state.

    Sobol is used only to cover exogenous parameters:
        S0/K, tau, r, sigma.

    S_t/K and the accumulated average are obtained from the SAME simulated
    historical trajectory. They are never sampled independently.
    """
    S0_K = _uniform_map(u4[0], cfg.S0_K_min, cfg.S0_K_max)
    tau = _uniform_map(u4[1], cfg.tau_min, cfg.tau_max)
    r = _uniform_map(u4[2], cfg.r_min, cfg.r_max)
    sigma = _uniform_map(u4[3], cfg.sigma_min, cfg.sigma_max)

    t = cfg.T - tau
    state_seed = _namespaced_seed(history_seed_base, candidate_index)
    pricing_seed = _namespaced_seed(pricing_seed_base, accepted_index)

    S_t_K, past_fixings_K = simulate_history_to_t(
        S0_K=S0_K,
        t=t,
        r=r,
        sigma=sigma,
        cfg=cfg,
        seed=state_seed,
    )

    n_past = int(len(past_fixings_K))
    n_future = cfg.n_fixings - n_past

    C_t = float(past_fixings_K.sum() / cfg.n_fixings)
    A_t_K = float(past_fixings_K.mean()) if n_past > 0 else np.nan

    # Domain filters. They do not create the state; they only decide whether
    # an economically generated state is inside the chosen experimental domain.
    if not (cfg.S_t_K_min <= S_t_K <= cfg.S_t_K_max):
        return None

    if n_past > 0 and not (cfg.A_t_K_min <= A_t_K <= cfg.A_t_K_max):
        return None

    return AsianState(
        scenario_id=f"{split}_{accepted_index:07d}",
        split=split,
        S0_K=float(S0_K),
        S_t_K=float(S_t_K),
        C_t=float(C_t),
        A_t_K=float(A_t_K),
        t=float(t),
        tau=float(tau),
        n_fix_past=n_past,
        n_fix_future=n_future,
        r=float(r),
        sigma=float(sigma),
        q=float(cfg.q),
        state_seed=state_seed,
        pricing_seed=pricing_seed,
    )


def generate_states(
    n_states: int,
    split: str,
    cfg: TFMConfig,
    outer_seed: int,
    history_seed_base: int,
    pricing_seed_base: int,
    oversampling_factor: float = 1.35,
) -> list[AsianState]:
    """
    Generate states with an independently scrambled Sobol design for each split.

    Important:
    Sobol here is DESIGN OF EXPERIMENTS over exogenous parameters.
    It is NOT the QMC pricing experiment.
    """
    sobol = qmc.Sobol(d=4, scramble=True, seed=outer_seed)

    states: list[AsianState] = []
    candidate_index = 0

    while len(states) < n_states:
        remaining = n_states - len(states)
        target_batch = max(256, int(np.ceil(remaining * oversampling_factor)))
        batch_n = 1 << int(np.ceil(np.log2(target_batch)))

        # random() is used because accept/reject filtering destroys the exact
        # digital-net balance anyway. Pricing QMC uses random_base2 separately.
        U = sobol.random(batch_n)

        for u4 in U:
            state = build_state_from_candidate(
                u4=u4,
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


def _future_fixing_times(state: AsianState, cfg: TFMConfig) -> np.ndarray:
    fixings = np.asarray(cfg.fixing_times, dtype=float)
    return fixings[fixings > state.t + _TIME_TOL]


def _normal_draws(
    n_paths: int,
    dimension: int,
    engine: str,
    seed: int,
) -> np.ndarray:
    if dimension <= 0:
        return np.empty((n_paths, 0), dtype=float)

    engine = engine.lower()

    if engine == "mc":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_paths, dimension))

    if engine in {"rqmc", "qmc", "sobol"}:
        # Main experiments use n_paths = 2^m.
        m_float = np.log2(n_paths)
        if abs(m_float - round(m_float)) > 1e-12:
            raise ValueError("For Sobol QMC, n_paths must be a power of 2.")

        sobol = qmc.Sobol(d=dimension, scramble=True, seed=seed)
        U = sobol.random_base2(m=int(round(m_float)))
        U = np.clip(U, _EPS_U, 1.0 - _EPS_U)
        return ndtri(U)

    raise ValueError("engine must be 'mc' or 'rqmc'.")


def _future_paths_from_z(
    state: AsianState,
    cfg: TFMConfig,
    z: np.ndarray,
    spot_override: Optional[float] = None,
) -> np.ndarray:
    """
    Build normalized future fixing prices from supplied N(0,1) innovations.
    """
    future_times = _future_fixing_times(state, cfg)
    d = len(future_times)

    if z.shape[1] != d:
        raise ValueError(f"Expected z dimension {d}, got {z.shape[1]}.")

    if d == 0:
        return np.empty((z.shape[0], 0), dtype=float)

    start_spot = float(state.S_t_K if spot_override is None else spot_override)
    dt = np.diff(np.r_[state.t, future_times])

    drift = (state.r - state.q - 0.5 * state.sigma**2) * dt
    diffusion = state.sigma * np.sqrt(dt) * z
    log_increments = drift[None, :] + diffusion
    log_growth = np.cumsum(log_increments, axis=1)

    return start_spot * np.exp(log_growth)


def payoff_and_pathwise_delta_samples(
    state: AsianState,
    cfg: TFMConfig,
    z: np.ndarray,
    spot_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discounted pathwise samples for price and Delta.

    Delta is with respect to S_t while the already observed average contribution
    C_t is held fixed.
    """
    spot = float(state.S_t_K if spot_override is None else spot_override)
    future = _future_paths_from_z(state, cfg, z, spot_override=spot)

    final_avg_K = state.C_t + future.sum(axis=1) / cfg.n_fixings
    intrinsic = np.maximum(final_avg_K - 1.0, 0.0)

    discount = np.exp(-state.r * state.tau)
    price_samples = discount * intrinsic

    if future.shape[1] == 0:
        delta_samples = np.zeros_like(price_samples)
    else:
        dA_dS = (future / spot).sum(axis=1) / cfg.n_fixings
        delta_samples = discount * (final_avg_K > 1.0).astype(float) * dA_dS

    return price_samples, delta_samples


def price_state(
    state: AsianState,
    cfg: TFMConfig,
    n_paths: int,
    engine: str,
    seed: int,
) -> Dict[str, float]:
    d = state.n_fix_future
    z = _normal_draws(n_paths=n_paths, dimension=d, engine=engine, seed=seed)
    p, delta = payoff_and_pathwise_delta_samples(state, cfg, z)

    # For RQMC, these within-scramble SEs are only naive diagnostics.
    # Formal RQMC uncertainty is estimated across independent scramblings.
    return {
        "price_K": float(np.mean(p)),
        "delta": float(np.mean(delta)),
        "price_se_naive": float(np.std(p, ddof=1) / np.sqrt(n_paths)),
        "delta_se_naive": float(np.std(delta, ddof=1) / np.sqrt(n_paths)),
    }


def price_state_replicated(
    state: AsianState,
    cfg: TFMConfig,
    n_paths_per_replication: int,
    n_replications: int,
    engine: str,
    seed_base: int,
) -> Dict[str, float]:
    """
    Average over independent MC seeds or independent Sobol scramblings.

    Between-replication SE is the preferred uncertainty estimate for randomized
    QMC.
    """
    price_estimates = []
    delta_estimates = []

    for rep in range(n_replications):
        out = price_state(
            state=state,
            cfg=cfg,
            n_paths=n_paths_per_replication,
            engine=engine,
            seed=int(seed_base + 1_000_003 * rep),
        )
        price_estimates.append(out["price_K"])
        delta_estimates.append(out["delta"])

    prices = np.asarray(price_estimates, dtype=float)
    deltas = np.asarray(delta_estimates, dtype=float)

    price_rep_se = (
        float(prices.std(ddof=1) / np.sqrt(n_replications))
        if n_replications > 1
        else np.nan
    )
    delta_rep_se = (
        float(deltas.std(ddof=1) / np.sqrt(n_replications))
        if n_replications > 1
        else np.nan
    )

    return {
        "price_K": float(prices.mean()),
        "delta": float(deltas.mean()),
        "price_se_rep": price_rep_se,
        "delta_se_rep": delta_rep_se,
        "n_paths": int(n_paths_per_replication * n_replications),
        "n_replications": int(n_replications),
        "pricing_engine": engine.lower(),
    }


def finite_difference_delta(
    state: AsianState,
    cfg: TFMConfig,
    n_paths: int = 2**14,
    engine: str = "rqmc",
    seed: int = 999,
    relative_bump: float = 1e-3,
) -> float:
    """
    Central finite-difference Delta using COMMON RANDOM NUMBERS.

    Because v = V/K and s = S/K:
        d(V/K)/d(S/K) = dV/dS = Delta.
    """
    d = state.n_fix_future
    z = _normal_draws(n_paths=n_paths, dimension=d, engine=engine, seed=seed)

    h = max(relative_bump * state.S_t_K, 1e-6)
    s_up = state.S_t_K + h
    s_dn = max(state.S_t_K - h, 1e-8)

    p_up, _ = payoff_and_pathwise_delta_samples(
        state, cfg, z, spot_override=s_up
    )
    p_dn, _ = payoff_and_pathwise_delta_samples(
        state, cfg, z, spot_override=s_dn
    )

    return float((p_up.mean() - p_dn.mean()) / (s_up - s_dn))
