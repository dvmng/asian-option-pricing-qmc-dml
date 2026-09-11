from __future__ import annotations

"""
Local economic hedge-effectiveness engine.

This module evaluates the first-order local risk offset delivered by a Delta
estimate without pretending that compute spot is a storable/tradable asset.

For a shocked spot P' at the SAME contract time/state:

    dV = V(P') - V(P)
    residual(Delta_hat) = dV - Delta_hat * (P' - P)

The smaller the residual, the better the local first-order hedge/sensitivity.

This is not a self-financing trading strategy. It is a local economic
risk-offset diagnostic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from compute.compute_state_generation import ComputeDatasetState
from compute.logou import LogOUParams, exact_step_log
from compute.logou_gpu import price_states_replicated_gpu
from compute.phase4_config import ComputeDatasetConfig

from .surrogate import FrozenSurrogate, discover_final_surrogates


@dataclass(frozen=True)
class LocalHedgeScenario:
    S0_K: float
    r: float
    kappa_q: float
    theta_q_log_K: float
    sigma_q: float
    name: str = "Q0_ATM"

    def validate(self, cfg: ComputeDatasetConfig) -> None:
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


def interior_evaluation_times(
    cfg: ComputeDatasetConfig,
    tau_floor_margin: float = 0.01,
) -> list[dict]:
    """
    Return one structurally interior evaluation time per monitoring interval.

    Intervals 0,...,N-2 use their midpoint. In the final interval, the
    evaluation time is adjusted to remain within the ML domain tau >= 0.05
    while staying strictly inside the interval.
    """
    out = []

    for j in range(cfg.n_fixings):
        left = cfg.T * j / cfg.n_fixings
        right = cfg.T * (j + 1) / cfg.n_fixings
        t_mid = 0.5 * (left + right)
        tau_mid = cfg.T - t_mid

        if tau_mid < cfg.tau_min:
            tau = cfg.tau_min + tau_floor_margin
            t_eval = cfg.T - tau
            if not (left < t_eval < right):
                raise ValueError(
                    "Adjusted final interior time is not inside the final interval."
                )
        else:
            t_eval = t_mid
            tau = tau_mid

        distance_days = (
            min(t_eval - left, right - t_eval)
            * 365.0
        )

        out.append(
            {
                "interval_index": j,
                "t": float(t_eval),
                "tau": float(tau),
                "n_fix_past": int(j),
                "n_fix_future": int(cfg.n_fixings - j),
                "distance_to_nearest_fixing_days": float(distance_days),
            }
        )

    return out


def simulate_interior_states_under_p(
    *,
    n_paths: int,
    scenario: LocalHedgeScenario,
    cfg: ComputeDatasetConfig,
    seed: int,
    shock_sizes: Iterable[float],
) -> pd.DataFrame:
    """
    Simulate fresh physical-P paths and record one state inside every
    scheduled-fixing interval.

    The already-fixed Asian contribution is kept coherent with the same
    physical path.
    """
    scenario.validate(cfg)

    eval_meta = interior_evaluation_times(cfg)
    fixing_times = {
        round(cfg.T * k / cfg.n_fixings, 12): k
        for k in range(1, cfg.n_fixings + 1)
    }
    eval_times = {
        round(x["t"], 12): x
        for x in eval_meta
    }

    event_times = sorted(set(fixing_times) | set(eval_times))

    params_p = LogOUParams(
        kappa=cfg.kappa_p,
        theta_log_K=cfg.theta_p_log_K,
        sigma=cfg.sigma_p,
    )

    rng = np.random.default_rng(seed)
    x = np.full(n_paths, np.log(scenario.S0_K), dtype=np.float64)

    prev_t = 0.0
    fixing_prices: dict[int, np.ndarray] = {}
    rows = []

    max_shock = max(float(abs(s)) for s in shock_sizes)

    for event_t in event_times:
        dt = float(event_t - prev_t)
        if dt <= 0:
            raise RuntimeError("Non-increasing event grid.")

        z = rng.standard_normal(n_paths)
        x = exact_step_log(x, dt, params_p, z)
        spot = np.exp(x)
        prev_t = event_t

        key = round(event_t, 12)

        # Evaluation times are strictly interior, so fixing and evaluation events cannot coincide.
        if key in eval_times and key in fixing_times:
            raise RuntimeError("Evaluation time unexpectedly equals a fixing time.")

        if key in eval_times:
            meta = eval_times[key]
            j = int(meta["n_fix_past"])

            if j == 0:
                fixed_contrib = np.zeros(n_paths, dtype=np.float64)
                past_avg = np.full(n_paths, np.nan, dtype=np.float64)
            else:
                observed = np.column_stack(
                    [fixing_prices[k] for k in range(1, j + 1)]
                )
                fixed_contrib = observed.sum(axis=1) / cfg.n_fixings
                past_avg = observed.mean(axis=1)

            valid = (
                (spot >= cfg.spot_K_min)
                & (spot <= cfg.spot_K_max)
                & (meta["tau"] >= cfg.tau_min)
                & (meta["tau"] <= cfg.tau_max)
                & (spot * (1.0 - max_shock) >= cfg.spot_K_min)
                & (spot * (1.0 + max_shock) <= cfg.spot_K_max)
            )

            if j > 0:
                valid &= (
                    (past_avg >= cfg.past_avg_K_min)
                    & (past_avg <= cfg.past_avg_K_max)
                )

            for path_idx in np.flatnonzero(valid):
                rows.append(
                    {
                        "physical_path_index": int(path_idx),
                        "interval_index": int(meta["interval_index"]),
                        "t": float(meta["t"]),
                        "tau": float(meta["tau"]),
                        "n_fix_past": int(meta["n_fix_past"]),
                        "n_fix_future": int(meta["n_fix_future"]),
                        "distance_to_nearest_fixing_days": float(
                            meta["distance_to_nearest_fixing_days"]
                        ),
                        "S0_K": float(scenario.S0_K),
                        "spot_K": float(spot[path_idx]),
                        "fixed_avg_contrib_K": float(fixed_contrib[path_idx]),
                        "past_avg_K": (
                            float(past_avg[path_idx])
                            if j > 0
                            else np.nan
                        ),
                        "r": float(scenario.r),
                        "kappa_q": float(scenario.kappa_q),
                        "theta_q_log_K": float(scenario.theta_q_log_K),
                        "sigma_q": float(scenario.sigma_q),
                    }
                )

        if key in fixing_times:
            fixing_prices[int(fixing_times[key])] = spot.copy()

    states = pd.DataFrame(rows)

    if states.empty:
        raise RuntimeError("No valid interior states generated.")

    states.insert(
        0,
        "local_state_id",
        [f"local_{i:07d}" for i in range(len(states))],
    )

    return states


def states_to_raw_features(states: pd.DataFrame) -> np.ndarray:
    return states[
        [
            "spot_K",
            "fixed_avg_contrib_K",
            "tau",
            "r",
            "kappa_q",
            "theta_q_log_K",
            "sigma_q",
        ]
    ].to_numpy(dtype=np.float32)


def to_pricing_state(
    row: pd.Series,
    *,
    spot_K: float,
    scenario_suffix: str,
    cfg: ComputeDatasetConfig,
    pricing_seed: int,
) -> ComputeDatasetState:
    return ComputeDatasetState(
        scenario_id=f"{row['local_state_id']}_{scenario_suffix}",
        split="local_hedge_oos",
        S0_K=float(row["S0_K"]),
        spot_K=float(spot_K),
        fixed_avg_contrib_K=float(row["fixed_avg_contrib_K"]),
        past_avg_K=float(row["past_avg_K"]),
        t=float(row["t"]),
        tau=float(row["tau"]),
        n_fix_past=int(row["n_fix_past"]),
        n_fix_future=int(row["n_fix_future"]),
        r=float(row["r"]),
        kappa_q=float(row["kappa_q"]),
        theta_q_log_K=float(row["theta_q_log_K"]),
        sigma_q=float(row["sigma_q"]),
        kappa_p=float(cfg.kappa_p),
        theta_p_log_K=float(cfg.theta_p_log_K),
        sigma_p=float(cfg.sigma_p),
        state_seed=0,
        pricing_seed=int(pricing_seed),
    )


def load_ensemble(
    *,
    final_models_dir: Path,
    model_type: str,
    train_size: int,
    seeds: Iterable[int],
    device: torch.device,
) -> list[FrozenSurrogate]:
    return [
        FrozenSurrogate.load(p, device)
        for p in discover_final_surrogates(
            final_models_dir,
            model_type,
            train_size,
            seeds,
        )
    ]


def ensemble_prediction(
    models: list[FrozenSurrogate],
    x_raw: np.ndarray,
) -> dict:
    p_all = []
    d_all = []

    for model in models:
        p, d = model.predict_price_delta(x_raw)
        p_all.append(p)
        d_all.append(d)

    P = np.vstack(p_all)
    D = np.vstack(d_all)

    return {
        "price_mean": P.mean(axis=0),
        "price_sd_seed": P.std(axis=0, ddof=1),
        "delta_mean": D.mean(axis=0),
        "delta_sd_seed": D.std(axis=0, ddof=1),
    }


def price_local_triplets(
    *,
    states: pd.DataFrame,
    shock_sizes: Iterable[float],
    cfg: ComputeDatasetConfig,
    device: str,
    n_paths_per_replication: int,
    n_replications: int,
    seed_base: int,
) -> pd.DataFrame:
    """
    Price base/up/down state triplets with common pricing seeds.

    The same RQMC seed is assigned to base, up and down members of each
    state/shock triplet to reduce numerical noise in local value differences.
    """
    shock_sizes = tuple(float(s) for s in shock_sizes)
    out_rows = []

    for shock_idx, shock in enumerate(shock_sizes):
        pricing_states = []
        metadata = []

        for i, row in states.iterrows():
            common_seed = (
                int(seed_base)
                + shock_idx * 10_000_000
                + int(i)
            )

            p0 = float(row["spot_K"])
            levels = {
                "down": p0 * (1.0 - shock),
                "base": p0,
                "up": p0 * (1.0 + shock),
            }

            for direction, shocked_spot in levels.items():
                pricing_states.append(
                    to_pricing_state(
                        row,
                        spot_K=shocked_spot,
                        scenario_suffix=f"s{shock:.4f}_{direction}",
                        cfg=cfg,
                        pricing_seed=common_seed,
                    )
                )
                metadata.append(
                    {
                        "local_state_id": row["local_state_id"],
                        "shock_size": shock,
                        "direction": direction,
                        "spot_base_K": p0,
                        "spot_scenario_K": shocked_spot,
                    }
                )

        priced = price_states_replicated_gpu(
            pricing_states,
            cfg,
            n_paths_per_replication=n_paths_per_replication,
            n_replications=n_replications,
            engine="rqmc",
            device=device,
            dtype="float64",
            batch_size=128,
        )

        if len(priced) != len(metadata):
            raise RuntimeError("Pricing output length mismatch.")

        for meta, res in zip(metadata, priced):
            out_rows.append(
                {
                    **meta,
                    "oracle_price_K": float(res["price_K"]),
                    "oracle_delta": float(res["delta"]),
                    "oracle_price_se_rep": float(
                        res.get("price_se_rep", np.nan)
                    ),
                    "oracle_delta_se_rep": float(
                        res.get("delta_se_rep", np.nan)
                    ),
                }
            )

    return pd.DataFrame(out_rows)


def build_local_residuals(
    *,
    states: pd.DataFrame,
    oracle_triplets: pd.DataFrame,
    mlp: dict,
    dml: dict,
) -> pd.DataFrame:
    prediction = states[
        [
            "local_state_id",
            "interval_index",
            "t",
            "tau",
            "n_fix_past",
            "n_fix_future",
            "distance_to_nearest_fixing_days",
            "spot_K",
            "fixed_avg_contrib_K",
            "past_avg_K",
        ]
    ].copy()

    prediction["delta_mlp"] = mlp["delta_mean"]
    prediction["delta_dml"] = dml["delta_mean"]
    prediction["delta_mlp_seed_sd"] = mlp["delta_sd_seed"]
    prediction["delta_dml_seed_sd"] = dml["delta_sd_seed"]

    wide_price = oracle_triplets.pivot(
        index=["local_state_id", "shock_size"],
        columns="direction",
        values="oracle_price_K",
    ).reset_index()

    wide_delta = (
        oracle_triplets.loc[
            oracle_triplets["direction"].eq("base"),
            ["local_state_id", "shock_size", "oracle_delta"],
        ]
        .drop_duplicates(["local_state_id", "shock_size"])
    )

    trip = wide_price.merge(
        wide_delta,
        on=["local_state_id", "shock_size"],
        how="inner",
        validate="one_to_one",
    ).merge(
        prediction,
        on="local_state_id",
        how="inner",
        validate="many_to_one",
    )

    rows = []

    for _, row in trip.iterrows():
        p0 = float(row["spot_K"])
        shock = float(row["shock_size"])

        # Optional finite-difference check around the same base state.
        fd_delta = (
            float(row["up"]) - float(row["down"])
        ) / (2.0 * shock * p0)

        for direction, shocked_price in (
            ("down", p0 * (1.0 - shock)),
            ("up", p0 * (1.0 + shock)),
        ):
            dP = shocked_price - p0
            dV = float(row[direction]) - float(row["base"])

            deltas = {
                "no_hedge": 0.0,
                "oracle_rqmc": float(row["oracle_delta"]),
                "mlp_ensemble": float(row["delta_mlp"]),
                "dml_ensemble": float(row["delta_dml"]),
            }

            base_common = {
                "local_state_id": row["local_state_id"],
                "interval_index": int(row["interval_index"]),
                "t": float(row["t"]),
                "tau": float(row["tau"]),
                "n_fix_past": int(row["n_fix_past"]),
                "n_fix_future": int(row["n_fix_future"]),
                "distance_to_nearest_fixing_days": float(
                    row["distance_to_nearest_fixing_days"]
                ),
                "spot_K": p0,
                "fixed_avg_contrib_K": float(
                    row["fixed_avg_contrib_K"]
                ),
                "past_avg_K": float(row["past_avg_K"]),
                "shock_size": shock,
                "direction": direction,
                "dP_K": float(dP),
                "oracle_dV_K": float(dV),
                "oracle_delta": float(row["oracle_delta"]),
                "central_fd_delta": float(fd_delta),
                "oracle_vs_fd_delta_abs_diff": float(
                    abs(float(row["oracle_delta"]) - fd_delta)
                ),
                "mlp_delta": float(row["delta_mlp"]),
                "dml_delta": float(row["delta_dml"]),
                "mlp_delta_abs_error_vs_oracle": float(
                    abs(float(row["delta_mlp"]) - float(row["oracle_delta"]))
                ),
                "dml_delta_abs_error_vs_oracle": float(
                    abs(float(row["delta_dml"]) - float(row["oracle_delta"]))
                ),
                "mlp_delta_seed_sd": float(row["delta_mlp_seed_sd"]),
                "dml_delta_seed_sd": float(row["delta_dml_seed_sd"]),
            }

            for strategy, delta_hat in deltas.items():
                hedge_offset = delta_hat * dP
                residual = dV - hedge_offset

                rows.append(
                    {
                        **base_common,
                        "strategy": strategy,
                        "delta_used": float(delta_hat),
                        "local_hedge_offset_K": float(hedge_offset),
                        "local_residual_K": float(residual),
                        "abs_local_residual_K": float(abs(residual)),
                    }
                )

    return pd.DataFrame(rows)


def metric_row(g: pd.DataFrame) -> dict:
    e = g["local_residual_K"].to_numpy(dtype=float)
    abs_e = np.abs(e)

    return {
        "n_observations": int(len(e)),
        "mean_residual": float(np.mean(e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(abs_e)),
        "median_abs": float(np.median(abs_e)),
        "p95_abs": float(np.quantile(abs_e, 0.95)),
        "p99_abs": float(np.quantile(abs_e, 0.99)),
        "max_abs": float(np.max(abs_e)),
    }


def summarize_metrics(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (shock, strategy), g in residuals.groupby(
        ["shock_size", "strategy"],
        sort=True,
    ):
        row = {
            "shock_size": float(shock),
            "strategy": strategy,
            **metric_row(g),
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    nohedge = (
        out.loc[out["strategy"].eq("no_hedge")]
        .set_index("shock_size")
    )

    for idx, row in out.iterrows():
        shock = row["shock_size"]
        benchmark = nohedge.loc[shock]

        out.loc[idx, "rmse_improvement_vs_nohedge_pct"] = (
            100.0 * (benchmark["rmse"] - row["rmse"])
            / benchmark["rmse"]
        )
        out.loc[idx, "mae_improvement_vs_nohedge_pct"] = (
            100.0 * (benchmark["mae"] - row["mae"])
            / benchmark["mae"]
        )

    return out


def summarize_by_fixing(
    residuals: pd.DataFrame,
    primary_shock: float = 0.01,
) -> pd.DataFrame:
    x = residuals.loc[
        np.isclose(residuals["shock_size"], primary_shock)
    ].copy()

    rows = []

    for (nfuture, strategy), g in x.groupby(
        ["n_fix_future", "strategy"],
        sort=False,
    ):
        rows.append(
            {
                "n_fix_future": int(nfuture),
                "tau": float(g["tau"].iloc[0]),
                "distance_to_nearest_fixing_days": float(
                    g["distance_to_nearest_fixing_days"].iloc[0]
                ),
                "strategy": strategy,
                **metric_row(g),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["n_fix_future", "strategy"], ascending=[False, True])
        .reset_index(drop=True)
    )


def paired_win_rates(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    key = [
        "local_state_id",
        "shock_size",
        "direction",
    ]

    wide = residuals.pivot(
        index=key,
        columns="strategy",
        values="abs_local_residual_K",
    ).reset_index()

    rows = []

    for shock, g in wide.groupby("shock_size", sort=True):
        rows.append(
            {
                "shock_size": float(shock),
                "dml_lower_abs_residual_than_mlp": float(
                    (
                        g["dml_ensemble"]
                        < g["mlp_ensemble"]
                    ).mean()
                ),
                "dml_lower_abs_residual_than_nohedge": float(
                    (
                        g["dml_ensemble"]
                        < g["no_hedge"]
                    ).mean()
                ),
                "mlp_lower_abs_residual_than_nohedge": float(
                    (
                        g["mlp_ensemble"]
                        < g["no_hedge"]
                    ).mean()
                ),
                "oracle_lower_abs_residual_than_nohedge": float(
                    (
                        g["oracle_rqmc"]
                        < g["no_hedge"]
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)
