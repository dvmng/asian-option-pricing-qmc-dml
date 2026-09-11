from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import statsmodels.api as sm


@dataclass(frozen=True)
class PhysicalLogOUCalibration:
    """Exact-discretization AR(1) mapping to continuous-time log-OU under P."""

    n_obs: int
    dt_years: float
    intercept: float
    phi: float
    residual_std: float
    kappa_p: float
    theta_p_log: float
    long_run_price_p: float
    sigma_p: float
    half_life_days: float

    def as_dict(self):
        return asdict(self)


def calibrate_logou_physical(prices: np.ndarray, day_count: float = 365.0) -> PhysicalLogOUCalibration:
    """Calibrate historical P dynamics from equally spaced daily prices.

    This function deliberately does NOT convert P parameters into pricing-measure Q
    parameters. That is a separate economic modelling decision.
    """
    prices = np.asarray(prices, dtype=np.float64)
    prices = prices[np.isfinite(prices)]
    if len(prices) < 20:
        raise ValueError("At least 20 observations are required.")
    if np.any(prices <= 0.0):
        raise ValueError("Prices must be positive.")

    x = np.log(prices)
    y = x[1:]
    X = sm.add_constant(x[:-1])
    res = sm.OLS(y, X).fit()
    a, phi = map(float, res.params)
    if not (0.0 < phi < 1.0):
        raise ValueError(f"Stationary AR(1) mapping requires 0 < phi < 1; got {phi:.6f}.")

    dt = 1.0 / float(day_count)
    kappa = -math.log(phi) / dt
    theta = a / (1.0 - phi)
    # Two estimated AR(1) parameters imply ddof=2 for the residual standard deviation.
    eta = float(np.std(np.asarray(res.resid), ddof=2))
    sigma = eta * math.sqrt(2.0 * kappa / (1.0 - phi * phi))
    half_life_days = math.log(2.0) / (kappa / day_count)

    return PhysicalLogOUCalibration(
        n_obs=int(len(prices)),
        dt_years=dt,
        intercept=a,
        phi=phi,
        residual_std=eta,
        kappa_p=kappa,
        theta_p_log=theta,
        long_run_price_p=math.exp(theta),
        sigma_p=sigma,
        half_life_days=half_life_days,
    )
