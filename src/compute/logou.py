from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogOUParams:
    """One-factor log mean-reverting dynamics for a positive normalized price.

    X_t = log(P_t / K)
    dX_t = kappa * (theta_log_K - X_t) dt + sigma dW_t

    Time units must be consistent across kappa, sigma and dt. In the TFM engine,
    contract time is expressed in years.
    """

    kappa: float
    theta_log_K: float
    sigma: float

    def validate(self) -> None:
        if not (self.kappa > 0.0):
            raise ValueError("kappa must be strictly positive.")
        if not (self.sigma > 0.0):
            raise ValueError("sigma must be strictly positive.")
        if not np.isfinite(self.theta_log_K):
            raise ValueError("theta_log_K must be finite.")


def transition_coefficients(kappa: float, sigma: float, dt: np.ndarray | float):
    """Exact Gaussian transition coefficients for the OU log process."""
    if kappa <= 0.0:
        raise ValueError("kappa must be strictly positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")

    dt_arr = np.asarray(dt, dtype=np.float64)
    if np.any(dt_arr < 0.0):
        raise ValueError("dt must be non-negative.")

    phi = np.exp(-kappa * dt_arr)
    # Exact conditional standard deviation computed with expm1 for numerical stability.
    variance = (sigma * sigma) * (-np.expm1(-2.0 * kappa * dt_arr)) / (2.0 * kappa)
    std = np.sqrt(np.maximum(variance, 0.0))
    return phi, std


def exact_step_log(
    x: np.ndarray | float,
    dt: np.ndarray | float,
    params: LogOUParams,
    z: np.ndarray | float,
):
    """Exact OU transition in log normalized-price space."""
    params.validate()
    phi, std = transition_coefficients(params.kappa, params.sigma, dt)
    return params.theta_log_K + (np.asarray(x) - params.theta_log_K) * phi + std * np.asarray(z)


def path_from_z(
    spot_K: float,
    dt: np.ndarray,
    params: LogOUParams,
    z: np.ndarray,
) -> np.ndarray:
    """Generate positive future normalized prices from supplied N(0,1) shocks.

    Parameters
    ----------
    spot_K:
        Current compute rental price divided by strike K.
    dt:
        Step sizes from valuation time through future fixing dates, shape [D].
    z:
        Standard normal shocks, shape [P, D].
    """
    params.validate()
    if spot_K <= 0.0:
        raise ValueError("spot_K must be positive.")

    dt = np.asarray(dt, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must have shape [n_paths, n_steps].")
    if z.shape[1] != len(dt):
        raise ValueError(f"Expected {len(dt)} shock columns, got {z.shape[1]}.")
    if len(dt) == 0:
        return np.empty((z.shape[0], 0), dtype=np.float64)

    phi, step_std = transition_coefficients(params.kappa, params.sigma, dt)
    x = np.full(z.shape[0], math.log(float(spot_K)), dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    for j in range(len(dt)):
        x = params.theta_log_K + (x - params.theta_log_K) * phi[j] + step_std[j] * z[:, j]
        out[:, j] = np.exp(x)

    return out


def spot_jacobian_path(
    future_K: np.ndarray,
    spot_K: float,
    elapsed_from_valuation: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Pathwise d(P_{t+u}/K)/d(P_t/K) for the log-OU process.

    For X=log(P/K), dX_{t+u}/dX_t = exp(-kappa*u), therefore
      dP_{t+u}/dP_t = exp(-kappa*u) * P_{t+u}/P_t.
    """
    future_K = np.asarray(future_K, dtype=np.float64)
    elapsed = np.asarray(elapsed_from_valuation, dtype=np.float64)
    if spot_K <= 0.0:
        raise ValueError("spot_K must be positive.")
    if future_K.ndim != 2 or future_K.shape[1] != len(elapsed):
        raise ValueError("future_K and elapsed_from_valuation have incompatible shapes.")
    attenuation = np.exp(-kappa * elapsed)
    return (future_K / float(spot_K)) * attenuation[None, :]
