from __future__ import annotations

from tfm_project.paths import ProjectPaths

P = ProjectPaths.discover()

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from compute.phase4_config import ComputeDatasetConfig
from compute_hedging.local_hedge import (
    LocalHedgeScenario,
    build_local_residuals,
    ensemble_prediction,
    load_ensemble,
    paired_win_rates,
    price_local_triplets,
    resolve_device,
    simulate_interior_states_under_p,
    states_to_raw_features,
    summarize_by_fixing,
    summarize_metrics,
)

PRIMARY_SHOCK = 0.01
SHOCKS = (0.005, 0.01, 0.02)

PHYSICAL_SEEDS = (9101, 9201, 9301, 9401)
PRICING_SEED_BASES = (
    21_000_000,
    31_000_000,
    41_000_000,
    51_000_000,
)

PHYSICAL_PATHS_PER_BLOCK = 128
RQMC_PATHS = 4096
RQMC_REPS = 2

MODEL_SEEDS = (1701, 2701, 3701, 4701, 5701)
TRAIN_SIZE = 65536


def metric_row(g: pd.DataFrame) -> dict:
    e = g["local_residual_K"].to_numpy(dtype=float)
    a = np.abs(e)
    return {
        "n_observations": int(len(e)),
        "mean_residual": float(np.mean(e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(a)),
        "median_abs": float(np.median(a)),
        "p95_abs": float(np.quantile(a, 0.95)),
        "p99_abs": float(np.quantile(a, 0.99)),
        "max_abs": float(np.max(a)),
    }


def pooled_metrics(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (shock, strategy), g in residuals.groupby(
        ["shock_size", "strategy"],
        sort=True,
    ):
        rows.append(
            {
                "shock_size": float(shock),
                "strategy": strategy,
                **metric_row(g),
            }
        )

    out = pd.DataFrame(rows)

    nohedge = (
        out.loc[out["strategy"].eq("no_hedge")]
        .set_index("shock_size")
    )

    for idx, row in out.iterrows():
        b = nohedge.loc[row["shock_size"]]
        out.loc[idx, "rmse_improvement_vs_nohedge_pct"] = (
            100.0 * (b["rmse"] - row["rmse"]) / b["rmse"]
        )
        out.loc[idx, "mae_improvement_vs_nohedge_pct"] = (
            100.0 * (b["mae"] - row["mae"]) / b["mae"]
        )

    return out


def block_dispersion(
    block_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (shock, strategy), g in block_metrics_df.groupby(
        ["shock_size", "strategy"],
        sort=True,
    ):
        rows.append(
            {
                "shock_size": float(shock),
                "strategy": strategy,
                "n_blocks": int(len(g)),
                "rmse_block_mean": float(g["rmse"].mean()),
                "rmse_block_sd": float(g["rmse"].std(ddof=1)),
                "rmse_block_min": float(g["rmse"].min()),
                "rmse_block_max": float(g["rmse"].max()),
                "mae_block_mean": float(g["mae"].mean()),
                "mae_block_sd": float(g["mae"].std(ddof=1)),
                "p95_abs_block_mean": float(g["p95_abs"].mean()),
                "p95_abs_block_sd": float(g["p95_abs"].std(ddof=1)),
            }
        )

    return pd.DataFrame(rows)


def paired_rates_from_residuals(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    key = [
        "block_id",
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
                "n_pairs": int(len(g)),
                "dml_lower_abs_residual_than_mlp": float(
                    (g["dml_ensemble"] < g["mlp_ensemble"]).mean()
                ),
                "dml_lower_abs_residual_than_nohedge": float(
                    (g["dml_ensemble"] < g["no_hedge"]).mean()
                ),
                "mlp_lower_abs_residual_than_nohedge": float(
                    (g["mlp_ensemble"] < g["no_hedge"]).mean()
                ),
                "oracle_lower_abs_residual_than_nohedge": float(
                    (g["oracle_rqmc"] < g["no_hedge"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_rates_by_block(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    parts = []

    for block_id, g in residuals.groupby("block_id", sort=True):
        x = paired_rates_from_residuals(g.copy())
        x.insert(0, "block_id", int(block_id))
        parts.append(x)

    return pd.concat(parts, ignore_index=True)


def fd_summary(block_id: int, residuals: pd.DataFrame) -> dict:
    fd = (
        residuals.loc[
            np.isclose(residuals["shock_size"], min(SHOCKS))
            & residuals["direction"].eq("up")
        ][
            [
                "local_state_id",
                "oracle_delta",
                "central_fd_delta",
                "oracle_vs_fd_delta_abs_diff",
            ]
        ]
        .drop_duplicates("local_state_id")
    )

    return {
        "block_id": int(block_id),
        "n_states": int(len(fd)),
        "mean_abs_diff": float(
            fd["oracle_vs_fd_delta_abs_diff"].mean()
        ),
        "median_abs_diff": float(
            fd["oracle_vs_fd_delta_abs_diff"].median()
        ),
        "p95_abs_diff": float(
            fd["oracle_vs_fd_delta_abs_diff"].quantile(0.95)
        ),
        "max_abs_diff": float(
            fd["oracle_vs_fd_delta_abs_diff"].max()
        ),
    }


def save_figures(
    pooled: pd.DataFrame,
    by_fixing: pd.DataFrame,
    block_metrics_df: pd.DataFrame,
    out: Path,
) -> list[str]:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    pvt = pooled.pivot(
        index="shock_size",
        columns="strategy",
        values="rmse",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for strategy in (
        "no_hedge",
        "oracle_rqmc",
        "mlp_ensemble",
        "dml_ensemble",
    ):
        ax.plot(
            100.0 * pvt.index.to_numpy(dtype=float),
            pvt[strategy].to_numpy(dtype=float),
            marker="o",
            label=strategy,
        )
    ax.set_title("Final local hedge RMSE by spot-shock size")
    ax.set_xlabel("Spot shock magnitude (%)")
    ax.set_ylabel("Local residual RMSE (K units)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "01_final_local_hedge_rmse_by_shock.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(str(path))

    x = by_fixing.loc[
        np.isclose(by_fixing["shock_size"], PRIMARY_SHOCK)
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, g in x.groupby("strategy", sort=False):
        g = g.sort_values("n_fix_future", ascending=False)
        ax.plot(
            g["n_fix_future"],
            g["rmse"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Final local hedge RMSE by contract interval — 1% shock")
    ax.set_xlabel("Future fixings remaining")
    ax.set_ylabel("Local residual RMSE (K units)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "02_final_local_hedge_rmse_by_fixing.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(str(path))

    y = block_metrics_df.loc[
        np.isclose(block_metrics_df["shock_size"], PRIMARY_SHOCK)
        & block_metrics_df["strategy"].isin(
            ["mlp_ensemble", "dml_ensemble"]
        )
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for strategy, g in y.groupby("strategy", sort=False):
        ax.plot(
            g["block_id"],
            g["rmse"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Physical-block stability — 1% local hedge RMSE")
    ax.set_xlabel("Independent physical block")
    ax.set_ylabel("Local residual RMSE (K units)")
    ax.set_xticks(sorted(y["block_id"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "03_final_local_hedge_block_stability.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(str(path))

    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )
    ap.add_argument(
        "--models-dir",
        default=str(P.models_final),
        help="Directory containing the MLP/DML model ensemble.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(P.reproduced_results / "hedging" / "local"),
    )
    args = ap.parse_args()

    cfg = ComputeDatasetConfig()
    dev = resolve_device(args.device)

    scenario = LocalHedgeScenario(
        S0_K=1.0,
        r=0.03,
        kappa_q=cfg.kappa_p,
        theta_q_log_K=0.0,
        sigma_q=cfg.sigma_p,
        name="Q0_ATM_LOCAL_HEDGE_FINAL",
    )

    final_models_dir = P.resolve(args.models_dir)

    print(f"Loading MLP/DML ensembles from: {final_models_dir}")
    mlp_models = load_ensemble(
        final_models_dir=final_models_dir,
        model_type="mlp",
        train_size=TRAIN_SIZE,
        seeds=MODEL_SEEDS,
        device=dev,
    )
    dml_models = load_ensemble(
        final_models_dir=final_models_dir,
        model_type="dml",
        train_size=TRAIN_SIZE,
        seeds=MODEL_SEEDS,
        device=dev,
    )

    all_states = []
    all_oracle = []
    all_residuals = []
    all_block_metrics = []
    fd_rows = []

    for block_id, (physical_seed, pricing_seed_base) in enumerate(
        zip(PHYSICAL_SEEDS, PRICING_SEED_BASES),
        start=1,
    ):
        print(
            f"\n[Block {block_id}/4] "
            f"physical_seed={physical_seed}, "
            f"pricing_seed_base={pricing_seed_base}"
        )

        states = simulate_interior_states_under_p(
            n_paths=PHYSICAL_PATHS_PER_BLOCK,
            scenario=scenario,
            cfg=cfg,
            seed=physical_seed,
            shock_sizes=SHOCKS,
        )

        # Keep state identifiers unique across independent simulation blocks.
        states = states.copy()
        states["local_state_id"] = (
            "b"
            + str(block_id)
            + "_"
            + states["local_state_id"].astype(str)
        )
        states.insert(0, "block_id", block_id)
        states["physical_seed"] = physical_seed

        x_raw = states_to_raw_features(states)
        mlp = ensemble_prediction(mlp_models, x_raw)
        dml = ensemble_prediction(dml_models, x_raw)

        oracle = price_local_triplets(
            states=states,
            shock_sizes=SHOCKS,
            cfg=cfg,
            device=str(dev),
            n_paths_per_replication=RQMC_PATHS,
            n_replications=RQMC_REPS,
            seed_base=pricing_seed_base,
        )
        oracle.insert(0, "block_id", block_id)
        oracle["physical_seed"] = physical_seed
        oracle["pricing_seed_base"] = pricing_seed_base

        residuals = build_local_residuals(
            states=states,
            oracle_triplets=oracle,
            mlp=mlp,
            dml=dml,
        )
        residuals.insert(0, "block_id", block_id)
        residuals["physical_seed"] = physical_seed

        bm = summarize_metrics(residuals)
        bm.insert(0, "block_id", block_id)
        bm["physical_seed"] = physical_seed
        bm["valid_local_states"] = len(states)

        fd_rows.append(fd_summary(block_id, residuals))

        all_states.append(states)
        all_oracle.append(oracle)
        all_residuals.append(residuals)
        all_block_metrics.append(bm)

        primary = bm.loc[
            np.isclose(bm["shock_size"], PRIMARY_SHOCK)
            & bm["strategy"].isin(
                ["mlp_ensemble", "dml_ensemble", "oracle_rqmc", "no_hedge"]
            ),
            ["strategy", "rmse", "mae"],
        ]
        print(primary.to_string(index=False))

    states_all = pd.concat(all_states, ignore_index=True)
    oracle_all = pd.concat(all_oracle, ignore_index=True)
    residuals_all = pd.concat(all_residuals, ignore_index=True)
    block_metrics_df = pd.concat(all_block_metrics, ignore_index=True)
    fd_df = pd.DataFrame(fd_rows)

    pooled = pooled_metrics(residuals_all)
    dispersion = block_dispersion(block_metrics_df)
    paired_pooled = paired_rates_from_residuals(residuals_all)
    paired_block = paired_rates_by_block(residuals_all)

    # Preserve the shock dimension in the pooled fixing summary.
    by_fixing_parts = []
    for shock in SHOCKS:
        x = residuals_all.loc[
            np.isclose(residuals_all["shock_size"], shock)
        ]
        for (nfuture, strategy), g in x.groupby(
            ["n_fix_future", "strategy"],
            sort=False,
        ):
            by_fixing_parts.append(
                {
                    "shock_size": float(shock),
                    "n_fix_future": int(nfuture),
                    "tau": float(g["tau"].iloc[0]),
                    "distance_to_nearest_fixing_days": float(
                        g["distance_to_nearest_fixing_days"].iloc[0]
                    ),
                    "strategy": strategy,
                    **metric_row(g),
                }
            )

    by_fixing = (
        pd.DataFrame(by_fixing_parts)
        .sort_values(
            ["shock_size", "n_fix_future", "strategy"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )

    out = P.assert_not_frozen_write(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    states_all.to_csv(
        out / "final_local_oos_states.csv",
        index=False,
    )
    oracle_all.to_csv(
        out / "final_local_oracle_repricing.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    residuals_all.to_csv(
        out / "final_local_hedge_residuals.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    block_metrics_df.to_csv(
        out / "final_local_hedge_metrics_by_block.csv",
        index=False,
    )
    pooled.to_csv(
        out / "final_local_hedge_metrics_pooled.csv",
        index=False,
    )
    dispersion.to_csv(
        out / "final_local_hedge_block_dispersion.csv",
        index=False,
    )
    paired_pooled.to_csv(
        out / "final_local_hedge_paired_win_rates_pooled.csv",
        index=False,
    )
    paired_block.to_csv(
        out / "final_local_hedge_paired_win_rates_by_block.csv",
        index=False,
    )
    by_fixing.to_csv(
        out / "final_local_hedge_metrics_by_fixing.csv",
        index=False,
    )
    fd_df.to_csv(
        out / "final_oracle_delta_fd_check_by_block.csv",
        index=False,
    )

    figures = save_figures(
        pooled,
        by_fixing,
        block_metrics_df,
        out,
    )

    protocol = {
        "status": "FROZEN_BEFORE_PHASE5B1_FINAL",
        "experiment": "Phase 5B.1 final local economic hedge effectiveness",
        "primary_shock": PRIMARY_SHOCK,
        "robustness_shocks": [0.005, 0.02],
        "directions": ["down", "up"],
        "physical_blocks": len(PHYSICAL_SEEDS),
        "physical_paths_per_block": PHYSICAL_PATHS_PER_BLOCK,
        "physical_seeds": list(PHYSICAL_SEEDS),
        "pricing_seed_bases": list(PRICING_SEED_BASES),
        "rqmc_paths_per_state": RQMC_PATHS,
        "rqmc_replications": RQMC_REPS,
        "pricing_measure": "Q0 provisional",
        "physical_measure": "frozen physical-P Log-OU",
        "state_rule": (
            "one deterministic interior state per fixing interval; midpoint "
            "except final interval at tau_min+0.01"
        ),
        "model_train_size": TRAIN_SIZE,
        "model_seeds": list(MODEL_SEEDS),
        "mlp_dml_aggregation": "five-seed ensemble mean",
        "common_random_numbers_base_up_down": True,
        "no_clipping": True,
        "uses_phase4_test_or_reference_states": False,
        "retraining_allowed": False,
        "post_test_calendar_feature_addition_allowed": False,
    }
    payload = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    protocol_hash = hashlib.sha256(payload).hexdigest()

    audit = {
        "status": "PHASE5B1_FINAL_COMPLETED",
        "protocol_hash_sha256": protocol_hash,
        "protocol": protocol,
        "scenario": scenario.__dict__,
        "total_valid_local_states": int(len(states_all)),
        "total_local_residual_observations": int(len(residuals_all)),
        "valid_states_by_block": {
            str(int(k)): int(v)
            for k, v in states_all.groupby("block_id").size().items()
        },
        "fd_check_by_block": fd_rows,
        "figures": figures,
        "interpretation": (
            "Primary economic validation of local first-order risk offset. "
            "Not a self-financing spot hedge and not the dynamic forward stress test."
        ),
        "frozen_phase4_note": (
            "No Phase-4 model, dataset, lambda, architecture, checkpoint, "
            "test/reference result or feature set was modified."
        ),
    }

    (out / "phase5b1_final_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n=== PHASE 5B.1 FINAL — POOLED METRICS ===")
    print(
        pooled[
            [
                "shock_size",
                "strategy",
                "n_observations",
                "rmse",
                "mae",
                "median_abs",
                "p95_abs",
                "p99_abs",
                "rmse_improvement_vs_nohedge_pct",
                "mae_improvement_vs_nohedge_pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== PHASE 5B.1 FINAL — BLOCK DISPERSION ===")
    print(dispersion.to_string(index=False))

    print("\n=== PHASE 5B.1 FINAL — PAIRED WIN RATES ===")
    print(paired_pooled.to_string(index=False))

    print("\n=== PHASE 5B.1 FINAL — FD CHECK BY BLOCK ===")
    print(fd_df.to_string(index=False))

    print("\n=== FINAL AUDIT ===")
    print(json.dumps(audit, indent=2))

    if not np.isfinite(pooled["rmse"]).all():
        raise SystemExit("Non-finite pooled RMSE.")

    if len(fd_df) != len(PHYSICAL_SEEDS):
        raise SystemExit("Missing FD diagnostic block.")

    if fd_df["max_abs_diff"].max() > 0.01:
        raise SystemExit(
            "Unexpectedly large pathwise-vs-FD Delta discrepancy."
        )

    print("\nPHASE 5B.1 FINAL PASSED.")


if __name__ == "__main__":
    main()
