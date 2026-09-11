from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^(?P<model>mlp|dml)_n(?P<n>\d+)_seed(?P<seed>\d+)__(?P<split>test|reference)\.csv\.gz$"
)


def load_config(data_dir: Path) -> dict:
    return json.loads((data_dir / "config.json").read_text(encoding="utf-8"))


def expected_future_asian_state(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add conditional expected normalized final average under each row's Q scenario."""
    out = df.copy()
    T = float(cfg.get("T", 1.0))
    N = int(cfg.get("n_fixings", 12))
    fixing_times = np.asarray(
        cfg.get("fixing_times") or [T * j / N for j in range(1, N + 1)],
        dtype=float,
    )

    exp_final = np.empty(len(out), dtype=float)
    required_future_avg = np.empty(len(out), dtype=float)

    for i, row in enumerate(out.itertuples(index=False)):
        t = float(row.t)
        spot = float(row.spot_K)
        fixed = float(row.fixed_avg_contrib_K)
        kappa = float(row.kappa_q)
        theta = float(row.theta_q_log_K)
        sigma = float(row.sigma_q)

        future = fixing_times[fixing_times > t + 1e-12]
        n_future = len(future)
        if n_future == 0:
            ef = fixed
            req = np.nan
        else:
            u = future - t
            phi = np.exp(-kappa * u)
            mean_log = theta + (np.log(spot) - theta) * phi
            var_log = sigma**2 * (1.0 - np.exp(-2.0*kappa*u)) / (2.0*kappa)
            expected_spot = np.exp(mean_log + 0.5 * var_log)
            ef = fixed + float(expected_spot.sum()) / N
            req = N * (1.0 - fixed) / n_future

        exp_final[i] = ef
        required_future_avg[i] = req

    out["expected_final_avg_K_Q"] = exp_final
    out["expected_moneyness_Q"] = exp_final - 1.0
    out["abs_expected_moneyness_Q"] = np.abs(exp_final - 1.0)
    out["required_future_avg_K_for_strike"] = required_future_avg
    out["spot_minus_required_future_avg"] = (
        out["spot_K"] - out["required_future_avg_K_for_strike"]
    )
    return out


def discover_predictions(pred_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(pred_dir.glob("*.csv.gz")):
        m = RUN_RE.match(p.name)
        if not m:
            continue
        rows.append({
            "path": p,
            "model_type": m.group("model"),
            "train_size": int(m.group("n")),
            "seed": int(m.group("seed")),
            "split": m.group("split"),
        })
    if not rows:
        raise FileNotFoundError(f"No final prediction files found in {pred_dir}")
    return pd.DataFrame(rows)


def load_prediction_panel(files: pd.DataFrame, split: str, train_size: int) -> pd.DataFrame:
    subset = files[
        (files["split"] == split) & (files["train_size"] == train_size)
    ]
    chunks = []
    for meta in subset.itertuples(index=False):
        d = pd.read_csv(meta.path)
        d["model_type"] = meta.model_type
        d["train_size"] = meta.train_size
        d["seed"] = meta.seed
        d["split"] = meta.split
        chunks.append(d)
    if not chunks:
        raise ValueError(f"No predictions for split={split}, train_size={train_size}")
    return pd.concat(chunks, ignore_index=True)


def scenario_aggregate(panel: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    g = (
        panel.groupby(["scenario_id", "model_type"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            price_true=("price_true", "first"),
            delta_true=("delta_true", "first"),
            price_abs_error_mean=("price_abs_error", "mean"),
            price_abs_error_max_seed=("price_abs_error", "max"),
            price_error_mean=("price_error", "mean"),
            price_pred_sd_seed=("price_pred", "std"),
            delta_abs_error_mean=("delta_abs_error", "mean"),
            delta_abs_error_max_seed=("delta_abs_error", "max"),
            delta_error_mean=("delta_error", "mean"),
            delta_pred_sd_seed=("delta_pred", "std"),
        )
    )
    return g.merge(states, on="scenario_id", how="left", validate="many_to_one")


def concentration(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), g in panel.groupby(["model_type", "seed"]):
        for target in ("price", "delta"):
            sq = np.square(g[f"{target}_error"].to_numpy(dtype=float))
            total = sq.sum()
            order = np.sort(sq)[::-1]
            n = len(order)
            row = {
                "model_type": model,
                "seed": int(seed),
                "target": target,
                "n_eval": n,
            }
            for frac in (0.001, 0.005, 0.01, 0.05):
                k = max(1, int(np.ceil(frac*n)))
                row[f"mse_share_top_{frac:g}"] = (
                    float(order[:k].sum()/total) if total > 0 else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def feature_spearman(scen: pd.DataFrame, error_col: str) -> pd.DataFrame:
    numeric = [
        "spot_K", "fixed_avg_contrib_K", "past_avg_K", "tau", "r",
        "kappa_q", "theta_q_log_K", "sigma_q", "n_fix_past", "n_fix_future",
        "expected_final_avg_K_Q", "expected_moneyness_Q",
        "abs_expected_moneyness_Q", "required_future_avg_K_for_strike",
        "spot_minus_required_future_avg",
    ]
    rows = []
    for model, g in scen.groupby("model_type"):
        for feature in numeric:
            if feature not in g.columns:
                continue
            x = g[[feature, error_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < 5 or x[feature].nunique() < 2:
                continue
            rho = x[feature].corr(x[error_col], method="spearman")
            rows.append({
                "model_type": model,
                "error_metric": error_col,
                "feature": feature,
                "spearman_rho": float(rho),
                "abs_spearman_rho": float(abs(rho)),
                "n": int(len(x)),
            })
    return pd.DataFrame(rows).sort_values(
        ["model_type", "abs_spearman_rho"], ascending=[True, False]
    )


def boundary_bins(scen: pd.DataFrame) -> pd.DataFrame:
    edges = [-np.inf, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, np.inf]
    labels = [
        "<=0.0025", "(0.0025,0.005]", "(0.005,0.01]", "(0.01,0.025]",
        "(0.025,0.05]", "(0.05,0.10]", ">0.10",
    ]
    d = scen.copy()
    d["boundary_distance_bin"] = pd.cut(
        d["abs_expected_moneyness_Q"], bins=edges, labels=labels
    )
    return (
        d.groupby(["model_type", "boundary_distance_bin"], observed=True)
        .agg(
            n=("scenario_id", "count"),
            price_abs_error_mean=("price_abs_error_mean", "mean"),
            price_abs_error_p95=("price_abs_error_mean", lambda x: x.quantile(.95)),
            delta_abs_error_mean=("delta_abs_error_mean", "mean"),
            delta_abs_error_p95=("delta_abs_error_mean", lambda x: x.quantile(.95)),
        )
        .reset_index()
    )


def paired_scenario(scen: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "scenario_id", "model_type",
        "price_abs_error_mean", "delta_abs_error_mean",
    ]
    w = scen[keep].pivot(
        index="scenario_id", columns="model_type",
        values=["price_abs_error_mean", "delta_abs_error_mean"],
    )
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    if {
        "price_abs_error_mean_mlp", "price_abs_error_mean_dml"
    }.issubset(w.columns):
        w["dml_minus_mlp_price_abs_error"] = (
            w["price_abs_error_mean_dml"] - w["price_abs_error_mean_mlp"]
        )
    if {
        "delta_abs_error_mean_mlp", "delta_abs_error_mean_dml"
    }.issubset(w.columns):
        w["dml_minus_mlp_delta_abs_error"] = (
            w["delta_abs_error_mean_dml"] - w["delta_abs_error_mean_mlp"]
        )
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/phase4_final")
    ap.add_argument(
        "--evaluation-dir", default="results/compute_ml_evaluation"
    )
    ap.add_argument("--output-dir", default="results/phase4d_extreme_errors")
    ap.add_argument("--train-size", type=int, default=65536)
    ap.add_argument(
        "--splits", nargs="+", choices=["test", "reference"],
        default=["test", "reference"],
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    eval_dir = Path(args.evaluation_dir)
    pred_dir = eval_dir / "predictions"
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(data_dir)
    files = discover_predictions(pred_dir)

    audit = {
        "status": "PHASE4D_DIAGNOSTIC_ONLY_NO_RETRAINING",
        "train_size": args.train_size,
        "splits": args.splits,
        "prediction_files_discovered": int(len(files)),
        "policy": (
            "Test/reference are already consumed. This phase diagnoses frozen final "
            "models only; it must not be used to retune architecture, lambda, epochs "
            "or other training hyperparameters."
        ),
    }

    all_top = []
    all_conc = []
    all_corr = []
    all_bins = []
    all_pair = []

    for split in args.splits:
        states = pd.read_csv(data_dir / f"{split}.csv.gz")
        states = expected_future_asian_state(states, cfg)

        panel = load_prediction_panel(files, split, args.train_size)
        scen = scenario_aggregate(panel, states)

        # Top scenarios by average error across the five seeds.
        for target in ("price", "delta"):
            col = f"{target}_abs_error_mean"
            top = (
                scen.sort_values(col, ascending=False)
                .groupby("model_type", group_keys=False)
                .head(50)
                .copy()
            )
            top["target_ranked"] = target
            top["split"] = split
            all_top.append(top)

            corr = feature_spearman(scen, col)
            corr["split"] = split
            all_corr.append(corr)

        conc = concentration(panel)
        conc["split"] = split
        all_conc.append(conc)

        bins = boundary_bins(scen)
        bins["split"] = split
        all_bins.append(bins)

        pair = paired_scenario(scen)
        pair = pair.merge(
            states[
                [
                    "scenario_id", "spot_K", "fixed_avg_contrib_K", "tau",
                    "r", "kappa_q", "theta_q_log_K", "sigma_q",
                    "n_fix_past", "n_fix_future",
                    "expected_final_avg_K_Q", "expected_moneyness_Q",
                    "abs_expected_moneyness_Q",
                ]
            ],
            on="scenario_id", how="left", validate="one_to_one",
        )
        pair["split"] = split
        all_pair.append(pair)

        scen.to_csv(
            outdir / f"scenario_errors_{split}_n{args.train_size}.csv",
            index=False,
        )

    top_df = pd.concat(all_top, ignore_index=True)
    conc_df = pd.concat(all_conc, ignore_index=True)
    corr_df = pd.concat(all_corr, ignore_index=True)
    bins_df = pd.concat(all_bins, ignore_index=True)
    pair_df = pd.concat(all_pair, ignore_index=True)

    top_df.to_csv(outdir / "top50_extreme_scenarios.csv", index=False)
    conc_df.to_csv(outdir / "mse_error_concentration.csv", index=False)
    corr_df.to_csv(outdir / "feature_error_spearman.csv", index=False)
    bins_df.to_csv(outdir / "error_by_exercise_boundary_distance.csv", index=False)
    pair_df.to_csv(outdir / "scenario_paired_dml_minus_mlp.csv", index=False)
    (outdir / "diagnostic_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("\n=== PHASE 4D: FROZEN-MODEL EXTREME ERROR DIAGNOSTICS ===")
    print(json.dumps(audit, indent=2))

    print("\n=== STRONGEST FEATURE ASSOCIATIONS ===")
    show = (
        corr_df.sort_values(
            ["split", "model_type", "error_metric", "abs_spearman_rho"],
            ascending=[True, True, True, False],
        )
        .groupby(["split", "model_type", "error_metric"], group_keys=False)
        .head(5)
    )
    print(show.to_string(index=False))

    print("\n=== MSE CONCENTRATION (mean across seeds) ===")
    mean_conc = (
        conc_df.groupby(["split", "model_type", "target"], as_index=False)
        .mean(numeric_only=True)
    )
    print(mean_conc.to_string(index=False))

    print("\n=== ERROR BY EXPECTED EXERCISE-BOUNDARY DISTANCE ===")
    print(bins_df.to_string(index=False))

    print(f"\nSaved diagnostics -> {outdir}")
    print("PHASE 4D DIAGNOSTIC COMPLETED (NO RETRAINING).")


if __name__ == "__main__":
    main()
