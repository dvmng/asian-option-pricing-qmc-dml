from __future__ import annotations

import numpy as np


def logou_forward_K(
    spot_K,
    dt,
    kappa_q,
    theta_q_log_K,
    sigma_q,
):
    """Model-implied cash-settled forward mark E^Q[P_T/K | P_t/K].

    This is a model-implied forward mark for a non-storable compute-price index,
    not a cash-and-carry commodity forward.
    """
    spot = np.asarray(spot_K, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    kappa = np.asarray(kappa_q, dtype=np.float64)
    theta = np.asarray(theta_q_log_K, dtype=np.float64)
    sigma = np.asarray(sigma_q, dtype=np.float64)

    if np.any(spot <= 0):
        raise ValueError("spot_K must be positive.")
    if np.any(dt < 0):
        raise ValueError("dt must be non-negative.")
    if np.any(kappa <= 0):
        raise ValueError("kappa_q must be positive.")
    if np.any(sigma <= 0):
        raise ValueError("sigma_q must be positive.")

    phi = np.exp(-kappa * dt)
    var_log = (
        sigma**2
        * (-np.expm1(-2.0 * kappa * dt))
        / (2.0 * kappa)
    )
    mean_log = theta + (np.log(spot) - theta) * phi
    return np.exp(mean_log + 0.5 * var_log)


def logou_forward_spot_jacobian(
    spot_K,
    dt,
    kappa_q,
    theta_q_log_K,
    sigma_q,
):
    """d F(t,t+dt)/d spot for the exact one-factor Log-OU forward mark."""
    spot = np.asarray(spot_K, dtype=np.float64)
    fwd = logou_forward_K(
        spot, dt, kappa_q, theta_q_log_K, sigma_q
    )
    phi = np.exp(-np.asarray(kappa_q, dtype=np.float64) * np.asarray(dt, dtype=np.float64))
    return phi * fwd / spot


def spot_delta_to_forward_units(
    delta_spot,
    spot_K,
    dt,
    kappa_q,
    theta_q_log_K,
    sigma_q,
):
    """Reproduce the historical Delta/J policy (delivery-price sensitivity).

    This omits settlement discounting. For present-value Delta neutrality,
    use spot_delta_to_forward_units_pv with the interest rate explicitly provided.
    """
    jac = logou_forward_spot_jacobian(
        spot_K, dt, kappa_q, theta_q_log_K, sigma_q
    )
    if np.any(np.abs(jac) < 1e-12):
        raise FloatingPointError("Forward spot Jacobian is too small.")
    return np.asarray(delta_spot, dtype=np.float64) / jac


def spot_delta_to_forward_units_pv(
    delta_spot, spot_K, dt, kappa_q, theta_q_log_K, sigma_q, r,
):
    """Units neutralizing option PV Delta against a cash-settled forward.

    G = exp(-r*dt)*(F-L), where the existing delivery price L is fixed.
    Hence dG/dP = exp(-r*dt)*dF/dP and q = exp(r*dt)*Delta/(dF/dP).
    This opt-in correction does not change frozen historical experiment runners.
    """
    return np.exp(np.asarray(r, dtype=np.float64) * np.asarray(dt, dtype=np.float64)) * spot_delta_to_forward_units(
        delta_spot, spot_K, dt, kappa_q, theta_q_log_K, sigma_q
    )
