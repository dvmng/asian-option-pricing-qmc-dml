from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from compute.compute_state_generation import ComputeDatasetState
from compute.logou import LogOUParams, exact_step_log
from compute.logou_gpu import price_states_replicated_gpu
from compute.phase4_config import ComputeDatasetConfig, INPUT_COLUMNS

from .forward import (
    logou_forward_K,
    logou_forward_spot_jacobian,
    spot_delta_to_forward_units,
)
from .surrogate import FrozenSurrogate, discover_final_surrogates


@dataclass(frozen=True)
class HedgeScenario:
    S0_K: float
    r: float
    kappa_q: float
    theta_q_log_K: float
    sigma_q: float
    name: str = "Q0_ATM"

    def validate_against_training_domain(self, cfg: ComputeDatasetConfig) -> None:
        if not (cfg.S0_K_min <= self.S0_K <= cfg.S0_K_max):
            raise ValueError("S0_K outside frozen training design.")
        if not (cfg.r_min <= self.r <= cfg.r_max):
            raise ValueError("r outside frozen training design.")
        if not (cfg.kappa_q_min <= self.kappa_q <= cfg.kappa_q_max):
            raise ValueError("kappa_q outside frozen training design.")
        if not (
            cfg.theta_q_log_K_min
            <= self.theta_q_log_K
            <= cfg.theta_q_log_K_max
        ):
            raise ValueError("theta_q_log_K outside frozen training design.")
        if not (cfg.sigma_q_min <= self.sigma_q <= cfg.sigma_q_max):
            raise ValueError("sigma_q outside frozen training design.")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto/cuda/cpu.")


def simulate_physical_fixing_paths(
    *,
    n_paths: int,
    S0_K: float,
    cfg: ComputeDatasetConfig,
    seed: int,
) -> np.ndarray:
    """Exact physical-P Log-OU paths sampled at the 12 monthly fixing dates."""
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if S0_K <= 0:
        raise ValueError("S0_K must be positive.")

    params = LogOUParams(
        kappa=cfg.kappa_p,
        theta_log_K=cfg.theta_p_log_K,
        sigma=cfg.sigma_p,
    )
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, cfg.n_fixings))

    dt = cfg.T / cfg.n_fixings
    x = np.full(n_paths, np.log(S0_K), dtype=np.float64)
    out = np.empty((n_paths, cfg.n_fixings + 1), dtype=np.float64)
    out[:, 0] = S0_K

    for j in range(cfg.n_fixings):
        x = exact_step_log(x, dt, params, z[:, j])
        out[:, j+1] = np.exp(x)

    return out


def raw_features_at_time(
    paths: np.ndarray,
    j: int,
    scenario: HedgeScenario,
    cfg: ComputeDatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return seven-input raw features, validity mask and past-average diagnostic.

    Hedge time j corresponds to t=j/N after observing the j-th fixing.
    j=0 is inception before any fixing.
    """
    n_paths = paths.shape[0]
    if not (0 <= j < cfg.n_fixings):
        raise ValueError("j must be 0,...,N-1.")

    t = cfg.T * j / cfg.n_fixings
    tau = cfg.T - t
    spot = paths[:, j]

    if j == 0:
        fixed = np.zeros(n_paths, dtype=np.float64)
        past_avg = np.full(n_paths, np.nan)
    else:
        observed = paths[:, 1:j+1]
        fixed = observed.sum(axis=1) / cfg.n_fixings
        past_avg = observed.mean(axis=1)

    x = np.column_stack([
        spot,
        fixed,
        np.full(n_paths, tau),
        np.full(n_paths, scenario.r),
        np.full(n_paths, scenario.kappa_q),
        np.full(n_paths, scenario.theta_q_log_K),
        np.full(n_paths, scenario.sigma_q),
    ]).astype(np.float64)

    valid = (
        (spot >= cfg.spot_K_min)
        & (spot <= cfg.spot_K_max)
        & (tau >= cfg.tau_min)
        & (tau <= cfg.tau_max)
    )
    if j > 0:
        valid &= (
            (past_avg >= cfg.past_avg_K_min)
            & (past_avg <= cfg.past_avg_K_max)
        )

    return x, valid, past_avg


def pricing_states_at_time(
    *,
    paths: np.ndarray,
    j: int,
    scenario: HedgeScenario,
    cfg: ComputeDatasetConfig,
    pricing_seed_base: int,
) -> list[ComputeDatasetState]:
    x, _, past_avg = raw_features_at_time(paths, j, scenario, cfg)
    t = cfg.T * j / cfg.n_fixings
    tau = cfg.T - t

    states = []
    for i in range(len(paths)):
        states.append(
            ComputeDatasetState(
                scenario_id=f"hedge_j{j:02d}_p{i:07d}",
                split="hedging",
                S0_K=float(paths[i, 0]),
                spot_K=float(x[i, 0]),
                fixed_avg_contrib_K=float(x[i, 1]),
                past_avg_K=float(past_avg[i]),
                t=float(t),
                tau=float(tau),
                n_fix_past=int(j),
                n_fix_future=int(cfg.n_fixings-j),
                r=float(scenario.r),
                kappa_q=float(scenario.kappa_q),
                theta_q_log_K=float(scenario.theta_q_log_K),
                sigma_q=float(scenario.sigma_q),
                kappa_p=float(cfg.kappa_p),
                theta_p_log_K=float(cfg.theta_p_log_K),
                sigma_p=float(cfg.sigma_p),
                state_seed=0,
                pricing_seed=int(pricing_seed_base + j * 1_000_000 + i),
            )
        )
    return states


def predict_surrogate_ensemble(
    models: list[FrozenSurrogate],
    x_raw: np.ndarray,
) -> dict:
    price_matrix, delta_matrix = [], []
    for m in models:
        p, d = m.predict_price_delta(x_raw)
        price_matrix.append(p)
        delta_matrix.append(d)
    P = np.vstack(price_matrix)
    D = np.vstack(delta_matrix)
    return {
        "price_mean": P.mean(axis=0),
        "price_sd_seed": P.std(axis=0, ddof=1) if len(models) > 1 else np.zeros(P.shape[1]),
        "delta_mean": D.mean(axis=0),
        "delta_sd_seed": D.std(axis=0, ddof=1) if len(models) > 1 else np.zeros(D.shape[1]),
        "price_by_seed": P,
        "delta_by_seed": D,
    }


def initial_oracle_price(
    scenario: HedgeScenario,
    cfg: ComputeDatasetConfig,
    *,
    n_paths_per_replication: int,
    n_replications: int,
    device: str,
) -> dict:
    dummy_path = np.full((1, cfg.n_fixings+1), scenario.S0_K)
    state = pricing_states_at_time(
        paths=dummy_path,
        j=0,
        scenario=scenario,
        cfg=cfg,
        pricing_seed_base=7_000_000,
    )[0]
    return price_states_replicated_gpu(
        [state],
        cfg,
        n_paths_per_replication=n_paths_per_replication,
        n_replications=n_replications,
        engine="rqmc",
        device=device,
        dtype="float64",
        batch_size=1,
    )[0]


def terminal_metrics(errors: np.ndarray) -> dict:
    e = np.asarray(errors, dtype=np.float64)
    loss = -e  # Seller shortfall is defined as the negative hedging error.
    var95 = float(np.quantile(loss, 0.95))
    var99 = float(np.quantile(loss, 0.99))
    tail95 = loss[loss >= var95]
    tail99 = loss[loss >= var99]

    return {
        "n_paths": int(len(e)),
        "mean_error": float(e.mean()),
        "std_error": float(e.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "p95_abs_error": float(np.quantile(np.abs(e), 0.95)),
        "p99_abs_error": float(np.quantile(np.abs(e), 0.99)),
        "max_abs_error": float(np.max(np.abs(e))),
        "loss_var95": var95,
        "loss_cvar95": float(tail95.mean()),
        "loss_var99": var99,
        "loss_cvar99": float(tail99.mean()),
        "prob_shortfall": float(np.mean(e < 0.0)),
    }


def run_monthly_next_fixing_forward_hedge(
    *,
    n_outer_paths: int,
    outer_seed: int,
    scenario: HedgeScenario,
    cfg: ComputeDatasetConfig,
    final_models_dir: Path,
    device: str = "cuda",
    oracle_inner_paths: int = 2048,
    oracle_replications: int = 1,
    premium_paths_per_replication: int = 8192,
    premium_replications: int = 8,
    model_seeds: Iterable[int] = (1701,2701,3701,4701,5701),
    train_size: int = 65536,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Monthly next-fixing forward hedge engine.

    Each hedge entered at t_j is a zero-cost model-implied forward maturing at
    the next monthly Asian fixing t_{j+1}. All strategies receive the same
    high-accuracy RQMC option premium, isolating hedge-ratio quality.
    """
    scenario.validate_against_training_domain(cfg)
    dev = resolve_device(device)

    paths = simulate_physical_fixing_paths(
        n_paths=n_outer_paths,
        S0_K=scenario.S0_K,
        cfg=cfg,
        seed=outer_seed,
    )

    validity = np.ones(n_outer_paths, dtype=bool)
    for j in range(cfg.n_fixings):
        _, valid_j, _ = raw_features_at_time(paths, j, scenario, cfg)
        validity &= valid_j

    valid_idx = np.flatnonzero(validity)
    ood_idx = np.flatnonzero(~validity)
    if len(valid_idx) == 0:
        raise RuntimeError("No physical paths remain inside the frozen ML domain.")

    common_paths = paths[valid_idx]

    mlp_models = [
        FrozenSurrogate.load(p, dev)
        for p in discover_final_surrogates(
            final_models_dir, "mlp", train_size, model_seeds
        )
    ]
    dml_models = [
        FrozenSurrogate.load(p, dev)
        for p in discover_final_surrogates(
            final_models_dir, "dml", train_size, model_seeds
        )
    ]

    premium = initial_oracle_price(
        scenario,
        cfg,
        n_paths_per_replication=premium_paths_per_replication,
        n_replications=premium_replications,
        device=str(dev),
    )
    V0 = float(premium["price_K"])

    strategy_names = ["oracle_rqmc", "mlp_ensemble", "dml_ensemble"]
    settlements = {
        name: np.zeros(len(common_paths), dtype=np.float64)
        for name in strategy_names
    }
    hedge_rows = []

    dt = cfg.T / cfg.n_fixings

    for j in range(cfg.n_fixings):
        x_raw, _, past_avg = raw_features_at_time(
            common_paths, j, scenario, cfg
        )

        states = pricing_states_at_time(
            paths=common_paths,
            j=j,
            scenario=scenario,
            cfg=cfg,
            pricing_seed_base=9_000_000,
        )
        oracle = price_states_replicated_gpu(
            states,
            cfg,
            n_paths_per_replication=oracle_inner_paths,
            n_replications=oracle_replications,
            engine="rqmc",
            device=str(dev),
            dtype="float32",
            batch_size=128,
        )
        oracle_delta = np.asarray([r["delta"] for r in oracle], dtype=np.float64)

        mlp = predict_surrogate_ensemble(mlp_models, x_raw)
        dml = predict_surrogate_ensemble(dml_models, x_raw)

        spot = common_paths[:, j]
        next_spot = common_paths[:, j+1]

        F = logou_forward_K(
            spot, dt,
            scenario.kappa_q,
            scenario.theta_q_log_K,
            scenario.sigma_q,
        )
        J = logou_forward_spot_jacobian(
            spot, dt,
            scenario.kappa_q,
            scenario.theta_q_log_K,
            scenario.sigma_q,
        )

        deltas = {
            "oracle_rqmc": oracle_delta,
            "mlp_ensemble": mlp["delta_mean"],
            "dml_ensemble": dml["delta_mean"],
        }

        settlement_time = cfg.T * (j+1) / cfg.n_fixings
        accumulation = math.exp(scenario.r * (cfg.T - settlement_time))

        for name, delta in deltas.items():
            # Frozen historical Delta/J policy; PV-corrected diagnostic is separate.
            h = delta / J
            cashflow = h * (next_spot - F)
            settlements[name] += cashflow * accumulation

            hedge_rows.append(
                pd.DataFrame({
                    "outer_path_index": valid_idx,
                    "hedge_step": j,
                    "t": cfg.T * j / cfg.n_fixings,
                    "tau": cfg.T - cfg.T * j / cfg.n_fixings,
                    "n_fix_future": cfg.n_fixings-j,
                    "strategy": name,
                    "spot_K": spot,
                    "past_avg_K": past_avg,
                    "fixed_avg_contrib_K": x_raw[:, 1],
                    "forward_K": F,
                    "forward_spot_jacobian": J,
                    "delta": delta,
                    "forward_units": h,
                    "next_spot_K": next_spot,
                    "forward_cashflow_K": cashflow,
                    "forward_cashflow_terminal_K": cashflow * accumulation,
                })
            )

    payoff = np.maximum(
        common_paths[:, 1:].mean(axis=1) - 1.0,
        0.0,
    )
    premium_terminal = V0 * math.exp(scenario.r * cfg.T)

    errors = {
        "no_hedge": premium_terminal - payoff,
        **{
            name: premium_terminal + settlements[name] - payoff
            for name in strategy_names
        },
    }

    terminal = pd.DataFrame({
        "outer_path_index": valid_idx,
        "payoff_K": payoff,
        "premium_K": V0,
        "premium_terminal_K": premium_terminal,
    })
    for name, e in errors.items():
        terminal[f"hedge_error_{name}_K"] = e

    metric_rows = []
    for name, e in errors.items():
        row = {
            "strategy": name,
            **terminal_metrics(e),
        }
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    audit = {
        "status": "PHASE5A_INFRASTRUCTURE_DIAGNOSTIC",
        "scenario": scenario.__dict__,
        "n_outer_paths_requested": int(n_outer_paths),
        "n_paths_in_ml_domain": int(len(valid_idx)),
        "n_paths_outside_ml_domain": int(len(ood_idx)),
        "ml_domain_coverage": float(len(valid_idx)/n_outer_paths),
        "outer_seed": int(outer_seed),
        "rebalancing": "monthly, immediately after each scheduled fixing",
        "hedge_instrument": (
            "zero-cost one-period model-implied compute forward settling at "
            "the next Asian fixing"
        ),
        "premium_policy": (
            "same high-accuracy RQMC option premium for no-hedge/oracle/MLP/DML; "
            "isolates hedge-ratio quality"
        ),
        "oracle_delta": {
            "engine": "rqmc",
            "paths_per_state": int(oracle_inner_paths),
            "replications": int(oracle_replications),
        },
        "premium_rqmc": premium,
        "model_seeds": [int(s) for s in model_seeds],
        "train_size": int(train_size),
        "warning": (
            "Phase 5A is infrastructure/smoke only. Final hedge instrument, "
            "rebalance frequency, physical scenario grid and Q calibration are "
            "not frozen here."
        ),
    }

    hedges = pd.concat(hedge_rows, ignore_index=True)
    return terminal, hedges, metrics, audit
