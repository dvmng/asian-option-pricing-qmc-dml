"""Dynamic monthly compute-forward hedge stress engine.

Rebalancing occurs at scheduled Asian fixing boundaries. Each hedge uses a
one-period model-implied compute forward settling at the next fixing, followed
by immediate rebalancing after the fixing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from compute.logou_gpu import price_states_replicated_gpu
from compute.phase4_config import ComputeDatasetConfig

from .engine import (
    HedgeScenario,
    initial_oracle_price,
    pricing_states_at_time,
    raw_features_at_time,
    resolve_device,
    simulate_physical_fixing_paths,
    terminal_metrics,
)
from .forward import logou_forward_K, logou_forward_spot_jacobian
from .surrogate import FrozenSurrogate, discover_final_surrogates


def load_frozen_ensembles(
    *,
    final_models_dir: Path,
    device: torch.device,
    train_size: int,
    model_seeds: Iterable[int],
) -> tuple[list[FrozenSurrogate], list[FrozenSurrogate]]:
    mlp = [
        FrozenSurrogate.load(p, device)
        for p in discover_final_surrogates(
            final_models_dir,
            "mlp",
            train_size,
            model_seeds,
        )
    ]
    dml = [
        FrozenSurrogate.load(p, device)
        for p in discover_final_surrogates(
            final_models_dir,
            "dml",
            train_size,
            model_seeds,
        )
    ]
    return mlp, dml


def predict_ensemble(
    models: list[FrozenSurrogate],
    x_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    prices = []
    deltas = []

    for model in models:
        p, d = model.predict_price_delta(x_raw)
        prices.append(p)
        deltas.append(d)

    P = np.vstack(prices)
    D = np.vstack(deltas)

    return {
        "price_mean": P.mean(axis=0),
        "price_sd_seed": P.std(axis=0, ddof=1),
        "delta_mean": D.mean(axis=0),
        "delta_sd_seed": D.std(axis=0, ddof=1),
    }


def run_forward_stress_block(
    *,
    block_id: int,
    n_outer_paths: int,
    outer_seed: int,
    oracle_pricing_seed_base: int,
    scenario: HedgeScenario,
    cfg: ComputeDatasetConfig,
    mlp_models: list[FrozenSurrogate],
    dml_models: list[FrozenSurrogate],
    common_premium_K: float,
    device: torch.device,
    oracle_inner_paths: int,
    oracle_replications: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    scenario.validate_against_training_domain(cfg)

    paths = simulate_physical_fixing_paths(
        n_paths=n_outer_paths,
        S0_K=scenario.S0_K,
        cfg=cfg,
        seed=outer_seed,
    )

    validity = np.ones(n_outer_paths, dtype=bool)
    for j in range(cfg.n_fixings):
        _, valid_j, _ = raw_features_at_time(
            paths,
            j,
            scenario,
            cfg,
        )
        validity &= valid_j

    valid_idx = np.flatnonzero(validity)
    ood_idx = np.flatnonzero(~validity)

    if len(valid_idx) == 0:
        raise RuntimeError(
            f"Block {block_id}: no physical paths remain inside ML domain."
        )

    common_paths = paths[valid_idx]
    n_valid = len(common_paths)

    strategy_names = (
        "oracle_rqmc",
        "mlp_ensemble",
        "dml_ensemble",
    )

    terminal_forward_cash = {
        name: np.zeros(n_valid, dtype=np.float64)
        for name in strategy_names
    }

    hedge_rows = []
    dt = cfg.T / cfg.n_fixings

    for j in range(cfg.n_fixings):
        x_raw, _, past_avg = raw_features_at_time(
            common_paths,
            j,
            scenario,
            cfg,
        )

        pricing_states = pricing_states_at_time(
            paths=common_paths,
            j=j,
            scenario=scenario,
            cfg=cfg,
            pricing_seed_base=(
                int(oracle_pricing_seed_base)
                + j * 1_000_000
            ),
        )

        oracle = price_states_replicated_gpu(
            pricing_states,
            cfg,
            n_paths_per_replication=oracle_inner_paths,
            n_replications=oracle_replications,
            engine="rqmc",
            device=str(device),
            dtype="float32",
            batch_size=128,
        )

        oracle_delta = np.asarray(
            [r["delta"] for r in oracle],
            dtype=np.float64,
        )

        mlp = predict_ensemble(mlp_models, x_raw)
        dml = predict_ensemble(dml_models, x_raw)

        spot = common_paths[:, j]
        next_spot = common_paths[:, j + 1]

        forward = logou_forward_K(
            spot,
            dt,
            scenario.kappa_q,
            scenario.theta_q_log_K,
            scenario.sigma_q,
        )
        jacobian = logou_forward_spot_jacobian(
            spot,
            dt,
            scenario.kappa_q,
            scenario.theta_q_log_K,
            scenario.sigma_q,
        )

        if np.any(jacobian <= 0.0):
            raise RuntimeError(
                f"Block {block_id}, step {j}: non-positive forward Jacobian."
            )

        deltas = {
            "oracle_rqmc": oracle_delta,
            "mlp_ensemble": mlp["delta_mean"],
            "dml_ensemble": dml["delta_mean"],
        }

        delta_seed_sd = {
            "oracle_rqmc": np.zeros(n_valid, dtype=np.float64),
            "mlp_ensemble": mlp["delta_sd_seed"],
            "dml_ensemble": dml["delta_sd_seed"],
        }

        settlement_time = cfg.T * (j + 1) / cfg.n_fixings
        terminal_accumulation = math.exp(
            scenario.r * (cfg.T - settlement_time)
        )

        for strategy, delta in deltas.items():
            # Frozen historical Delta/J policy; PV-corrected diagnostic is separate.
            units = delta / jacobian
            cashflow = units * (next_spot - forward)
            terminal_cashflow = cashflow * terminal_accumulation

            terminal_forward_cash[strategy] += terminal_cashflow

            hedge_rows.append(
                pd.DataFrame(
                    {
                        "block_id": block_id,
                        "outer_seed": outer_seed,
                        "outer_path_index": valid_idx,
                        "path_uid": [
                            f"b{block_id}_p{int(i):05d}"
                            for i in valid_idx
                        ],
                        "hedge_step": j,
                        "t": cfg.T * j / cfg.n_fixings,
                        "tau": cfg.T - cfg.T * j / cfg.n_fixings,
                        "n_fix_future": cfg.n_fixings - j,
                        "strategy": strategy,
                        "spot_K": spot,
                        "past_avg_K": past_avg,
                        "fixed_avg_contrib_K": x_raw[:, 1],
                        "forward_K": forward,
                        "forward_spot_jacobian": jacobian,
                        "inverse_forward_spot_jacobian": 1.0 / jacobian,
                        "delta": delta,
                        "delta_seed_sd": delta_seed_sd[strategy],
                        "forward_units": units,
                        "next_spot_K": next_spot,
                        "forward_cashflow_K": cashflow,
                        "forward_cashflow_terminal_K": terminal_cashflow,
                    }
                )
            )

    payoff = np.maximum(
        common_paths[:, 1:].mean(axis=1) - 1.0,
        0.0,
    )

    premium_terminal = (
        float(common_premium_K)
        * math.exp(scenario.r * cfg.T)
    )

    errors = {
        "no_hedge": premium_terminal - payoff,
        **{
            strategy: (
                premium_terminal
                + terminal_forward_cash[strategy]
                - payoff
            )
            for strategy in strategy_names
        },
    }

    terminal = pd.DataFrame(
        {
            "block_id": block_id,
            "outer_seed": outer_seed,
            "outer_path_index": valid_idx,
            "path_uid": [
                f"b{block_id}_p{int(i):05d}"
                for i in valid_idx
            ],
            "payoff_K": payoff,
            "premium_K": float(common_premium_K),
            "premium_terminal_K": premium_terminal,
        }
    )

    for strategy, error in errors.items():
        terminal[f"hedge_error_{strategy}_K"] = error

    metric_rows = []
    for strategy, error in errors.items():
        metric_rows.append(
            {
                "block_id": block_id,
                "outer_seed": outer_seed,
                "strategy": strategy,
                **terminal_metrics(error),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    hedge_details = pd.concat(hedge_rows, ignore_index=True)

    audit = {
        "block_id": int(block_id),
        "outer_seed": int(outer_seed),
        "oracle_pricing_seed_base": int(oracle_pricing_seed_base),
        "n_outer_paths_requested": int(n_outer_paths),
        "n_paths_in_ml_domain": int(len(valid_idx)),
        "n_paths_outside_ml_domain": int(len(ood_idx)),
        "ml_domain_coverage": float(len(valid_idx) / n_outer_paths),
        "valid_outer_path_indices": valid_idx.tolist(),
        "ood_outer_path_indices": ood_idx.tolist(),
    }

    return terminal, hedge_details, metrics, audit


def squared_error_concentration(
    terminal: pd.DataFrame,
    fractions: Iterable[float] = (0.01, 0.05),
) -> pd.DataFrame:
    rows = []

    for strategy in (
        "no_hedge",
        "oracle_rqmc",
        "mlp_ensemble",
        "dml_ensemble",
    ):
        error = terminal[f"hedge_error_{strategy}_K"].to_numpy(dtype=float)
        sq = np.square(error)
        order = np.argsort(sq)[::-1]
        total = float(sq.sum())

        row = {
            "strategy": strategy,
            "n_paths": int(len(error)),
            "total_squared_error": total,
        }

        for fraction in fractions:
            n_top = max(1, int(np.ceil(len(error) * float(fraction))))
            top_share = (
                float(sq[order[:n_top]].sum() / total)
                if total > 0.0
                else np.nan
            )
            pct = int(round(100.0 * fraction))
            row[f"top_{pct}pct_n_paths"] = int(n_top)
            row[f"top_{pct}pct_share_squared_error"] = top_share

        rows.append(row)

    return pd.DataFrame(rows)


def paired_terminal_win_rates(
    terminal: pd.DataFrame,
) -> dict:
    abs_err = {
        strategy: np.abs(
            terminal[f"hedge_error_{strategy}_K"].to_numpy(dtype=float)
        )
        for strategy in (
            "no_hedge",
            "oracle_rqmc",
            "mlp_ensemble",
            "dml_ensemble",
        )
    }

    return {
        "n_paths": int(len(terminal)),
        "dml_lower_abs_error_than_mlp": float(
            np.mean(abs_err["dml_ensemble"] < abs_err["mlp_ensemble"])
        ),
        "dml_lower_abs_error_than_nohedge": float(
            np.mean(abs_err["dml_ensemble"] < abs_err["no_hedge"])
        ),
        "mlp_lower_abs_error_than_nohedge": float(
            np.mean(abs_err["mlp_ensemble"] < abs_err["no_hedge"])
        ),
        "oracle_lower_abs_error_than_nohedge": float(
            np.mean(abs_err["oracle_rqmc"] < abs_err["no_hedge"])
        ),
        "dml_lower_abs_error_than_oracle": float(
            np.mean(abs_err["dml_ensemble"] < abs_err["oracle_rqmc"])
        ),
    }


def surrogate_step_diagnostic(
    hedge_details: pd.DataFrame,
) -> pd.DataFrame:
    key = [
        "block_id",
        "path_uid",
        "hedge_step",
        "n_fix_future",
        "tau",
    ]

    delta = hedge_details.pivot(
        index=key,
        columns="strategy",
        values="delta",
    ).reset_index()

    pnl = hedge_details.pivot(
        index=key,
        columns="strategy",
        values="forward_cashflow_terminal_K",
    ).reset_index()

    base = (
        hedge_details.loc[
            hedge_details["strategy"].eq("oracle_rqmc"),
            key + [
                "forward_spot_jacobian",
                "inverse_forward_spot_jacobian",
            ],
        ]
        .drop_duplicates(key)
    )

    x = (
        base
        .merge(
            delta,
            on=key,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pnl,
            on=key,
            suffixes=("_delta", "_pnl"),
            how="inner",
            validate="one_to_one",
        )
    )

    rows = []

    for (nfuture, step, tau), g in x.groupby(
        ["n_fix_future", "hedge_step", "tau"],
        sort=False,
    ):
        row = {
            "n_fix_future": int(nfuture),
            "hedge_step": int(step),
            "tau": float(tau),
            "n_path_blocks": int(len(g)),
            "mean_forward_spot_jacobian": float(
                g["forward_spot_jacobian"].mean()
            ),
            "mean_inverse_forward_spot_jacobian": float(
                g["inverse_forward_spot_jacobian"].mean()
            ),
        }

        for model in ("mlp_ensemble", "dml_ensemble"):
            delta_err = (
                g[f"{model}_delta"]
                - g["oracle_rqmc_delta"]
            ).to_numpy(dtype=float)

            pnl_err = (
                g[f"{model}_pnl"]
                - g["oracle_rqmc_pnl"]
            ).to_numpy(dtype=float)

            row[f"delta_mae_{model}_vs_oracle"] = float(
                np.mean(np.abs(delta_err))
            )
            row[f"delta_rmse_{model}_vs_oracle"] = float(
                np.sqrt(np.mean(delta_err**2))
            )
            row[f"pnl_diff_mae_{model}_vs_oracle"] = float(
                np.mean(np.abs(pnl_err))
            )
            row[f"pnl_diff_rmse_{model}_vs_oracle"] = float(
                np.sqrt(np.mean(pnl_err**2))
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("n_fix_future", ascending=False)
        .reset_index(drop=True)
    )
