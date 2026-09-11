from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .logou import LogOUParams, path_from_z, spot_jacobian_path
from .seed_utils import replication_seed


_EPS_U = 1e-12
_TIME_TOL = 1e-12


@dataclass(frozen=True)
class ComputeAsianConfig:
    """Contract-only configuration; pricing-measure parameters live in each state."""

    T: float = 1.0
    n_fixings: int = 12

    @property
    def fixing_times(self) -> np.ndarray:
        return np.asarray(
            [self.T * j / self.n_fixings for j in range(1, self.n_fixings + 1)],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ComputeAsianState:
    """State of an arithmetic Asian call on normalized compute rental price.

    ``spot_K`` is the current spot divided by strike and
    ``fixed_avg_contrib_K`` is the normalized contribution of past fixings.

    ``theta_q_log_K`` is the long-run log(P/K) level under the pricing measure
    specified for the state; it is not inferred from historical P dynamics.
    """

    scenario_id: str
    spot_K: float
    fixed_avg_contrib_K: float
    t: float
    tau: float
    n_fix_past: int
    n_fix_future: int
    r: float
    kappa_q: float
    theta_q_log_K: float
    sigma_q: float
    pricing_seed: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def logou_params(self) -> LogOUParams:
        return LogOUParams(
            kappa=float(self.kappa_q),
            theta_log_K=float(self.theta_q_log_K),
            sigma=float(self.sigma_q),
        )

    def validate(self, cfg: ComputeAsianConfig) -> None:
        if self.spot_K <= 0.0:
            raise ValueError("spot_K must be positive.")
        if not (0.0 <= self.t <= cfg.T + _TIME_TOL):
            raise ValueError("t outside contract horizon.")
        if abs(self.tau - (cfg.T - self.t)) > 1e-10:
            raise ValueError("tau must equal T - t.")
        if self.n_fix_past + self.n_fix_future != cfg.n_fixings:
            raise ValueError("past + future fixings must equal N.")
        if self.fixed_avg_contrib_K < 0.0:
            raise ValueError("fixed_avg_contrib_K cannot be negative.")
        self.logou_params.validate()


def _future_fixing_times(state: ComputeAsianState, cfg: ComputeAsianConfig) -> np.ndarray:
    return cfg.fixing_times[cfg.fixing_times > state.t + _TIME_TOL]


def _future_dt(state: ComputeAsianState, cfg: ComputeAsianConfig) -> np.ndarray:
    future = _future_fixing_times(state, cfg)
    if len(future) == 0:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.r_[state.t, future]).astype(np.float64)


def _elapsed_from_valuation(state: ComputeAsianState, cfg: ComputeAsianConfig) -> np.ndarray:
    return (_future_fixing_times(state, cfg) - state.t).astype(np.float64)


def normal_draws(n_paths: int, dimension: int, engine: str, seed: int) -> np.ndarray:
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if dimension <= 0:
        return np.empty((n_paths, 0), dtype=np.float64)

    engine = engine.lower()
    if engine == "mc":
        return np.random.default_rng(seed).standard_normal((n_paths, dimension))

    if engine in {"rqmc", "qmc", "sobol"}:
        m = math.log2(n_paths)
        if abs(m - round(m)) > 1e-12:
            raise ValueError("For Sobol RQMC, n_paths must be a power of 2.")
        sobol = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
        u = sobol.random_base2(m=int(round(m)))
        u = np.clip(u, _EPS_U, 1.0 - _EPS_U)
        return ndtri(u)

    raise ValueError("engine must be 'mc' or 'rqmc'.")


def future_paths_from_z(
    state: ComputeAsianState,
    cfg: ComputeAsianConfig,
    z: np.ndarray,
    spot_override: Optional[float] = None,
) -> np.ndarray:
    state.validate(cfg)
    spot = float(state.spot_K if spot_override is None else spot_override)
    return path_from_z(
        spot_K=spot,
        dt=_future_dt(state, cfg),
        params=state.logou_params,
        z=z,
    )


def payoff_and_pathwise_delta_samples(
    state: ComputeAsianState,
    cfg: ComputeAsianConfig,
    z: np.ndarray,
    spot_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discounted Asian payoff and pathwise Delta samples under log-OU.

    Delta holds the already-fixed Asian contribution constant. Because price and
    spot are normalized by K, d(V/K)/d(P_t/K) equals the usual dV/dP_t Delta.
    """
    state.validate(cfg)
    spot = float(state.spot_K if spot_override is None else spot_override)
    future = future_paths_from_z(state, cfg, z, spot_override=spot)

    final_avg_K = state.fixed_avg_contrib_K + future.sum(axis=1) / cfg.n_fixings
    intrinsic = np.maximum(final_avg_K - 1.0, 0.0)
    discount = math.exp(-state.r * state.tau)
    price_samples = discount * intrinsic

    if future.shape[1] == 0:
        return price_samples, np.zeros_like(price_samples)

    jac = spot_jacobian_path(
        future_K=future,
        spot_K=spot,
        elapsed_from_valuation=_elapsed_from_valuation(state, cfg),
        kappa=state.kappa_q,
    )
    dA_dspot = jac.sum(axis=1) / cfg.n_fixings
    delta_samples = discount * (final_avg_K > 1.0).astype(np.float64) * dA_dspot
    return price_samples, delta_samples


def price_state(
    state: ComputeAsianState,
    cfg: ComputeAsianConfig,
    n_paths: int,
    engine: str,
    seed: int,
) -> Dict[str, float]:
    z = normal_draws(n_paths, state.n_fix_future, engine, seed)
    p, d = payoff_and_pathwise_delta_samples(state, cfg, z)
    return {
        "price_K": float(np.mean(p)),
        "delta": float(np.mean(d)),
        "price_se_naive": float(np.std(p, ddof=1) / math.sqrt(n_paths)),
        "delta_se_naive": float(np.std(d, ddof=1) / math.sqrt(n_paths)),
    }


def price_state_replicated(
    state: ComputeAsianState,
    cfg: ComputeAsianConfig,
    n_paths_per_replication: int,
    n_replications: int,
    engine: str,
    seed_base: int,
) -> Dict[str, float]:
    prices, deltas = [], []
    for rep in range(n_replications):
        out = price_state(
            state=state,
            cfg=cfg,
            n_paths=n_paths_per_replication,
            engine=engine,
            seed=replication_seed(seed_base, rep),
        )
        prices.append(out["price_K"])
        deltas.append(out["delta"])

    prices = np.asarray(prices, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    return {
        "price_K": float(prices.mean()),
        "delta": float(deltas.mean()),
        "price_se_rep": float(prices.std(ddof=1) / math.sqrt(n_replications)) if n_replications > 1 else float("nan"),
        "delta_se_rep": float(deltas.std(ddof=1) / math.sqrt(n_replications)) if n_replications > 1 else float("nan"),
        "n_paths": int(n_paths_per_replication * n_replications),
        "n_replications": int(n_replications),
        "pricing_engine": engine.lower(),
    }


def finite_difference_delta_crn(
    state: ComputeAsianState,
    cfg: ComputeAsianConfig,
    n_paths: int,
    engine: str,
    seed: int,
    rel_bump: float = 1e-4,
) -> float:
    """Central finite-difference Delta with common random numbers."""
    h = rel_bump * state.spot_K
    if h <= 0.0 or state.spot_K - h <= 0.0:
        raise ValueError("Invalid finite-difference bump.")
    z = normal_draws(n_paths, state.n_fix_future, engine, seed)
    p_up, _ = payoff_and_pathwise_delta_samples(state, cfg, z, spot_override=state.spot_K + h)
    p_dn, _ = payoff_and_pathwise_delta_samples(state, cfg, z, spot_override=state.spot_K - h)
    return float((p_up.mean() - p_dn.mean()) / (2.0 * h))
