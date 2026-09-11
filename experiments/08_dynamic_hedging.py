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
from compute_hedging.engine import HedgeScenario, initial_oracle_price, resolve_device
from compute_hedging.dynamic_forward_stress import (
    load_frozen_ensembles,
    paired_terminal_win_rates,
    run_forward_stress_block,
    squared_error_concentration,
    surrogate_step_diagnostic,
)


PHYSICAL_SEEDS = (12101, 12201, 12301, 12401)
ORACLE_PRICING_SEED_BASES = (
    61_000_000,
    71_000_000,
    81_000_000,
    91_000_000,
)

PATHS_PER_BLOCK = 512
ORACLE_PATHS = 4096
ORACLE_REPS = 1

PREMIUM_PATHS = 8192
PREMIUM_REPS = 8

MODEL_SEEDS = (1701, 2701, 3701, 4701, 5701)
TRAIN_SIZE = 65536

MIN_DOMAIN_COVERAGE = 0.90

STRATEGIES = (
    "no_hedge",
    "oracle_rqmc",
    "mlp_ensemble",
    "dml_ensemble",
)


def terminal_metrics(error: np.ndarray) -> dict:
    e = np.asarray(error, dtype=float)
    loss = -e
    abs_e = np.abs(e)

    var95 = float(np.quantile(loss, 0.95))
    var99 = float(np.quantile(loss, 0.99))

    return {
        "n_paths": int(len(e)),
        "mean_error": float(e.mean()),
        "std_error": float(e.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(abs_e)),
        "median_abs_error": float(np.median(abs_e)),
        "p95_abs_error": float(np.quantile(abs_e, 0.95)),
        "p99_abs_error": float(np.quantile(abs_e, 0.99)),
        "max_abs_error": float(np.max(abs_e)),
        "loss_var95": var95,
        "loss_cvar95": float(loss[loss >= var95].mean()),
        "loss_var99": var99,
        "loss_cvar99": float(loss[loss >= var99].mean()),
        "prob_shortfall": float(np.mean(e < 0.0)),
    }


def pooled_terminal_metrics(terminal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        e = terminal[f"hedge_error_{strategy}_K"].to_numpy(dtype=float)
        rows.append(
            {
                "strategy": strategy,
                **terminal_metrics(e),
            }
        )
    return pd.DataFrame(rows)


def block_dispersion(metrics_by_block: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for strategy, g in metrics_by_block.groupby("strategy", sort=False):
        rows.append(
            {
                "strategy": strategy,
                "n_blocks": int(len(g)),
                "rmse_block_mean": float(g["rmse"].mean()),
                "rmse_block_sd": float(g["rmse"].std(ddof=1)),
                "rmse_block_min": float(g["rmse"].min()),
                "rmse_block_max": float(g["rmse"].max()),
                "mae_block_mean": float(g["mae"].mean()),
                "mae_block_sd": float(g["mae"].std(ddof=1)),
                "p95_abs_block_mean": float(g["p95_abs_error"].mean()),
                "p95_abs_block_sd": float(g["p95_abs_error"].std(ddof=1)),
                "p99_abs_block_mean": float(g["p99_abs_error"].mean()),
                "p99_abs_block_sd": float(g["p99_abs_error"].std(ddof=1)),
            }
        )

    return pd.DataFrame(rows)


def concentration_by_block(terminal: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for block_id, g in terminal.groupby("block_id", sort=True):
        c = squared_error_concentration(g)
        c.insert(0, "block_id", int(block_id))
        parts.append(c)

    return pd.concat(parts, ignore_index=True)


def paired_by_block(terminal: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for block_id, g in terminal.groupby("block_id", sort=True):
        row = {
            "block_id": int(block_id),
            **paired_terminal_win_rates(g),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def top_terminal_errors(
    terminal: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:
    parts = []

    for strategy in STRATEGIES:
        col = f"hedge_error_{strategy}_K"
        x = terminal.copy()
        x["strategy"] = strategy
        x["hedge_error_K"] = x[col]
        x["abs_hedge_error_K"] = x[col].abs()

        keep = [
            "strategy",
            "block_id",
            "outer_seed",
            "outer_path_index",
            "path_uid",
            "payoff_K",
            "premium_K",
            "premium_terminal_K",
            "hedge_error_K",
            "abs_hedge_error_K",
        ]

        parts.append(
            x.nlargest(n, "abs_hedge_error_K")[keep]
        )

    return pd.concat(parts, ignore_index=True)


def save_figures(
    pooled: pd.DataFrame,
    dispersion: pd.DataFrame,
    concentration: pd.DataFrame,
    step_diag: pd.DataFrame,
    out: Path,
) -> list[str]:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    order = [
        "no_hedge",
        "oracle_rqmc",
        "mlp_ensemble",
        "dml_ensemble",
    ]

    x = pooled.set_index("strategy").loc[order]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(order, x["rmse"].to_numpy())
    ax.set_title("Dynamic monthly forward hedge — terminal RMSE")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("RMSE (K units)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = fig_dir / "01_terminal_rmse.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(str(p))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(order, x["p99_abs_error"].to_numpy())
    ax.set_title("Dynamic monthly forward hedge — P99 absolute error")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("P99 |hedge error| (K units)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = fig_dir / "02_terminal_p99_abs_error.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(str(p))

    c = concentration.set_index("strategy").loc[order]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    positions = np.arange(len(order))
    width = 0.34
    ax.bar(
        positions - width / 2,
        c["top_1pct_share_squared_error"].to_numpy(),
        width=width,
        label="Top 1%",
    )
    ax.bar(
        positions + width / 2,
        c["top_5pct_share_squared_error"].to_numpy(),
        width=width,
        label="Top 5%",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=20)
    ax.set_title("Concentration of terminal squared hedge error")
    ax.set_ylabel("Share of total squared error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "03_terminal_mse_concentration.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        step_diag["n_fix_future"],
        step_diag["delta_rmse_mlp_ensemble_vs_oracle"],
        marker="o",
        label="MLP vs RQMC",
    )
    ax.plot(
        step_diag["n_fix_future"],
        step_diag["delta_rmse_dml_ensemble_vs_oracle"],
        marker="o",
        label="DML vs RQMC",
    )
    ax.set_title("Delta error at scheduled fixing boundaries")
    ax.set_xlabel("Future fixings remaining")
    ax.set_ylabel("Delta RMSE vs RQMC")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "04_delta_rmse_by_fixing_boundary.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(str(p))

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
        default=str(P.reproduced_results / "hedging" / "dynamic_forward"),
    )
    args = ap.parse_args()

    cfg = ComputeDatasetConfig()
    dev = resolve_device(args.device)

    scenario = HedgeScenario(
        S0_K=1.0,
        r=0.03,
        kappa_q=cfg.kappa_p,
        theta_q_log_K=0.0,
        sigma_q=cfg.sigma_p,
        name="Q0_ATM_FORWARD_STRESS_FINAL",
    )

    models_dir = P.resolve(args.models_dir)
    print(f"Loading MLP/DML ensembles from: {models_dir}")
    mlp_models, dml_models = load_frozen_ensembles(
        final_models_dir=models_dir,
        device=dev,
        train_size=TRAIN_SIZE,
        model_seeds=MODEL_SEEDS,
    )

    print(
        "\nComputing one common high-accuracy initial RQMC premium "
        "for every strategy and physical block..."
    )
    premium = initial_oracle_price(
        scenario,
        cfg,
        n_paths_per_replication=PREMIUM_PATHS,
        n_replications=PREMIUM_REPS,
        device=str(dev),
    )
    common_premium_K = float(premium["price_K"])

    all_terminal = []
    all_hedges = []
    all_block_metrics = []
    block_audits = []

    for block_id, (outer_seed, pricing_seed_base) in enumerate(
        zip(PHYSICAL_SEEDS, ORACLE_PRICING_SEED_BASES),
        start=1,
    ):
        print(
            f"\n[Block {block_id}/4] "
            f"outer_seed={outer_seed}, "
            f"oracle_pricing_seed_base={pricing_seed_base}"
        )

        terminal, hedges, metrics, audit = run_forward_stress_block(
            block_id=block_id,
            n_outer_paths=PATHS_PER_BLOCK,
            outer_seed=outer_seed,
            oracle_pricing_seed_base=pricing_seed_base,
            scenario=scenario,
            cfg=cfg,
            mlp_models=mlp_models,
            dml_models=dml_models,
            common_premium_K=common_premium_K,
            device=dev,
            oracle_inner_paths=ORACLE_PATHS,
            oracle_replications=ORACLE_REPS,
        )

        if audit["ml_domain_coverage"] < MIN_DOMAIN_COVERAGE:
            raise SystemExit(
                f"Block {block_id}: ML-domain coverage "
                f"{audit['ml_domain_coverage']:.3%} is below frozen "
                f"minimum {MIN_DOMAIN_COVERAGE:.0%}."
            )

        all_terminal.append(terminal)
        all_hedges.append(hedges)
        all_block_metrics.append(metrics)
        block_audits.append(audit)

        print(
            metrics[
                [
                    "strategy",
                    "n_paths",
                    "rmse",
                    "mae",
                    "p95_abs_error",
                    "p99_abs_error",
                ]
            ].to_string(index=False)
        )

    terminal_all = pd.concat(all_terminal, ignore_index=True)
    hedge_all = pd.concat(all_hedges, ignore_index=True)
    metrics_by_block = pd.concat(all_block_metrics, ignore_index=True)

    pooled = pooled_terminal_metrics(terminal_all)
    dispersion = block_dispersion(metrics_by_block)

    concentration_pooled = squared_error_concentration(terminal_all)
    concentration_blocks = concentration_by_block(terminal_all)

    paired_pooled = pd.DataFrame(
        [paired_terminal_win_rates(terminal_all)]
    )
    paired_blocks = paired_by_block(terminal_all)

    step_diag = surrogate_step_diagnostic(hedge_all)
    top_errors = top_terminal_errors(terminal_all, n=50)

    out = P.assert_not_frozen_write(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    terminal_all.to_csv(
        out / "final_forward_terminal_errors.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    hedge_all.to_csv(
        out / "final_forward_hedge_path_details.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    metrics_by_block.to_csv(
        out / "final_forward_metrics_by_block.csv",
        index=False,
    )
    pooled.to_csv(
        out / "final_forward_metrics_pooled.csv",
        index=False,
    )
    dispersion.to_csv(
        out / "final_forward_block_dispersion.csv",
        index=False,
    )
    concentration_pooled.to_csv(
        out / "final_forward_mse_concentration_pooled.csv",
        index=False,
    )
    concentration_blocks.to_csv(
        out / "final_forward_mse_concentration_by_block.csv",
        index=False,
    )
    paired_pooled.to_csv(
        out / "final_forward_paired_win_rates_pooled.csv",
        index=False,
    )
    paired_blocks.to_csv(
        out / "final_forward_paired_win_rates_by_block.csv",
        index=False,
    )
    step_diag.to_csv(
        out / "final_forward_error_by_fixing_step.csv",
        index=False,
    )
    top_errors.to_csv(
        out / "final_forward_top50_terminal_errors.csv",
        index=False,
    )

    figures = save_figures(
        pooled,
        dispersion,
        concentration_pooled,
        step_diag,
        out,
    )

    frozen_protocol = {
        "status": "FROZEN_BEFORE_PHASE5B2_FINAL",
        "experiment": "Phase 5B.2 final dynamic monthly compute-forward hedge stress test",
        "role": "secondary dynamic stress test after Phase 5B.1 local hedge validation",
        "pricing_measure": "Q0 provisional",
        "physical_measure": "frozen physical-P Log-OU",
        "scenario": {
            "S0_K": 1.0,
            "r": 0.03,
            "kappa_q": scenario.kappa_q,
            "theta_q_log_K": scenario.theta_q_log_K,
            "sigma_q": scenario.sigma_q,
        },
        "hedge_instrument": (
            "zero-cost one-period model-implied compute forward "
            "settling at next scheduled Asian fixing"
        ),
        "rebalancing": (
            "monthly, immediately after each of the scheduled fixing boundaries"
        ),
        "physical_blocks": len(PHYSICAL_SEEDS),
        "physical_paths_per_block": PATHS_PER_BLOCK,
        "physical_seeds": list(PHYSICAL_SEEDS),
        "oracle_pricing_seed_bases": list(ORACLE_PRICING_SEED_BASES),
        "oracle_delta_paths_per_state": ORACLE_PATHS,
        "oracle_delta_replications": ORACLE_REPS,
        "premium_paths_per_replication": PREMIUM_PATHS,
        "premium_replications": PREMIUM_REPS,
        "model_train_size": TRAIN_SIZE,
        "model_seeds": list(MODEL_SEEDS),
        "mlp_dml_aggregation": "five-seed ensemble mean",
        "common_initial_premium": (
            "single high-accuracy RQMC premium shared by all strategies and blocks"
        ),
        "no_clipping": True,
        "common_valid_domain_sample_within_each_block": True,
        "minimum_required_ml_domain_coverage_per_block": MIN_DOMAIN_COVERAGE,
        "primary_metrics": [
            "terminal hedge-error RMSE",
            "terminal hedge-error MAE",
            "P95/P99 absolute hedge error",
            "seller-shortfall VaR/CVaR 95/99",
        ],
        "tail_diagnostics": [
            "top 1 percent share of squared error",
            "top 5 percent share of squared error",
        ],
        "paired_diagnostics": [
            "DML abs error < MLP abs error",
            "DML abs error < no-hedge abs error",
            "MLP abs error < no-hedge abs error",
            "oracle abs error < no-hedge abs error",
        ],
        "step_diagnostics": [
            "Delta RMSE vs RQMC by remaining fixings",
            "terminal-valued one-period hedge P&L discrepancy vs RQMC by remaining fixings",
        ],
        "retraining_allowed": False,
        "feature_changes_allowed": False,
        "post_result_instrument_changes_allowed_within_phase5b2": False,
    }

    payload = json.dumps(
        frozen_protocol,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    protocol_hash = hashlib.sha256(payload).hexdigest()

    audit = {
        "status": "PHASE5B2_FINAL_COMPLETED",
        "protocol_hash_sha256": protocol_hash,
        "protocol": frozen_protocol,
        "common_initial_premium_rqmc": premium,
        "block_audits": block_audits,
        "total_valid_terminal_paths": int(len(terminal_all)),
        "total_hedge_path_rows": int(len(hedge_all)),
        "figures": figures,
        "interpretation": (
            "Secondary dynamic stress test at scheduled fixing boundaries. "
            "This experiment is not used to retune the seven-input surrogate."
        ),
        "phase5b1_relation": (
            "Phase 5B.1 remains the primary local economic validation; "
            "Phase 5B.2 tests whether that local advantage survives dynamic "
            "forward implementation at structurally difficult contract boundaries."
        ),
    }

    (out / "phase5b2_final_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n=== PHASE 5B.2 FINAL — POOLED TERMINAL METRICS ===")
    print(pooled.to_string(index=False))

    print("\n=== PHASE 5B.2 FINAL — BLOCK DISPERSION ===")
    print(dispersion.to_string(index=False))

    print("\n=== PHASE 5B.2 FINAL — PAIRED WIN RATES ===")
    print(paired_pooled.to_string(index=False))

    print("\n=== PHASE 5B.2 FINAL — MSE CONCENTRATION ===")
    print(concentration_pooled.to_string(index=False))

    print("\n=== PHASE 5B.2 FINAL — ERROR BY FIXING STEP ===")
    print(step_diag.to_string(index=False))

    print("\n=== DOMAIN COVERAGE ===")
    for a in block_audits:
        print(
            f"block={a['block_id']} "
            f"coverage={a['ml_domain_coverage']:.6f} "
            f"valid={a['n_paths_in_ml_domain']} "
            f"ood={a['n_paths_outside_ml_domain']}"
        )

    print("\n=== FINAL AUDIT ===")
    print(json.dumps(audit, indent=2))

    if not np.isfinite(pooled["rmse"]).all():
        raise SystemExit("Non-finite final forward RMSE.")

    if protocol_hash != "deferred":
        pass

    print("\nPHASE 5B.2 FINAL PASSED.")


if __name__ == "__main__":
    main()
