from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def add_fixing_boundary_distance(df: pd.DataFrame, n_fixings: int = 12) -> pd.DataFrame:
    out = df.copy()
    boundaries = np.arange(1, n_fixings + 1, dtype=float) / n_fixings
    tau = out["tau"].to_numpy(dtype=float)
    dist = np.abs(tau[:, None] - boundaries[None, :])
    nearest_idx = dist.argmin(axis=1)

    out["nearest_tau_fixing_boundary"] = boundaries[nearest_idx]
    out["distance_to_fixing_boundary_years"] = dist[np.arange(len(out)), nearest_idx]
    out["distance_to_fixing_boundary_days"] = (
        365.0 * out["distance_to_fixing_boundary_years"]
    )
    return out


def summarize_bins(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 0.25, 0.5, 1, 2, 3, 7, 15, np.inf]
    labels = [
        "<=0.25d", "(0.25,0.5]d", "(0.5,1]d", "(1,2]d",
        "(2,3]d", "(3,7]d", "(7,15]d", ">15d",
    ]
    d = df.copy()
    d["fixing_boundary_distance_bin"] = pd.cut(
        d["distance_to_fixing_boundary_days"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return (
        d.groupby(
            ["split", "model_type", "fixing_boundary_distance_bin"],
            observed=True,
        )
        .agg(
            n=("scenario_id", "count"),
            price_abs_error_mean=("price_abs_error_mean", "mean"),
            price_abs_error_p95=("price_abs_error_mean", lambda x: x.quantile(.95)),
            delta_abs_error_mean=("delta_abs_error_mean", "mean"),
            delta_abs_error_p95=("delta_abs_error_mean", lambda x: x.quantile(.95)),
            delta_abs_error_max=("delta_abs_error_mean", "max"),
        )
        .reset_index()
    )


def top_error_summary(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    rows = []
    for (split, model), g in df.groupby(["split", "model_type"]):
        baseline_median = float(g["distance_to_fixing_boundary_days"].median())
        for target in ("price_abs_error_mean", "delta_abs_error_mean"):
            top = g.nlargest(min(top_n, len(g)), target)
            rows.append({
                "split": split,
                "model_type": model,
                "error_metric": target,
                "top_n": int(len(top)),
                "all_scenarios_median_distance_days": baseline_median,
                "top_median_distance_days": float(
                    top["distance_to_fixing_boundary_days"].median()
                ),
                "top_frac_within_0_5d": float(
                    (top["distance_to_fixing_boundary_days"] <= 0.5).mean()
                ),
                "top_frac_within_1d": float(
                    (top["distance_to_fixing_boundary_days"] <= 1.0).mean()
                ),
                "top_frac_within_2d": float(
                    (top["distance_to_fixing_boundary_days"] <= 2.0).mean()
                ),
                "top_frac_within_3d": float(
                    (top["distance_to_fixing_boundary_days"] <= 3.0).mean()
                ),
                "spearman_error_vs_distance": float(
                    g[[target, "distance_to_fixing_boundary_days"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
            })
    return pd.DataFrame(rows)


def n_future_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "model_type", "n_fix_future"], as_index=False)
        .agg(
            n=("scenario_id", "count"),
            price_abs_error_mean=("price_abs_error_mean", "mean"),
            price_abs_error_p95=("price_abs_error_mean", lambda x: x.quantile(.95)),
            delta_abs_error_mean=("delta_abs_error_mean", "mean"),
            delta_abs_error_p95=("delta_abs_error_mean", lambda x: x.quantile(.95)),
            delta_abs_error_max=("delta_abs_error_mean", "max"),
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase4d-dir",
        default="results/phase4d_extreme_errors",
    )
    ap.add_argument(
        "--output-dir",
        default="results/phase4d_fixing_boundary",
    )
    ap.add_argument("--n-fixings", type=int, default=12)
    args = ap.parse_args()

    phase4d = Path(args.phase4d_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "test": phase4d / "scenario_errors_test_n65536.csv",
        "reference": phase4d / "scenario_errors_reference_n65536.csv",
    }

    chunks = []
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        d = pd.read_csv(path)
        d["split"] = split
        chunks.append(add_fixing_boundary_distance(d, args.n_fixings))

    panel = pd.concat(chunks, ignore_index=True)

    bins = summarize_bins(panel)
    top = top_error_summary(panel)
    nfut = n_future_summary(panel)

    panel.to_csv(outdir / "scenario_errors_with_fixing_distance.csv", index=False)
    bins.to_csv(outdir / "error_by_fixing_boundary_distance.csv", index=False)
    top.to_csv(outdir / "top50_fixing_boundary_summary.csv", index=False)
    nfut.to_csv(outdir / "error_by_n_fix_future.csv", index=False)

    summary = {
        "status": "PHASE4D1_DIAGNOSTIC_ONLY",
        "n_fixings": args.n_fixings,
        "interpretation": (
            "Distance is measured from valuation tau to the nearest scheduled "
            "monthly fixing boundary k/N. This is a post-hoc diagnostic of frozen "
            "models and must not be used to retune them."
        ),
        "key_test_dml_delta_top50": top[
            (top["split"].eq("test"))
            & (top["model_type"].eq("dml"))
            & (top["error_metric"].eq("delta_abs_error_mean"))
        ].to_dict(orient="records")[0],
    }

    (outdir / "fixing_boundary_diagnostic.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== TOP-50 EXTREME ERROR PROXIMITY TO MONTHLY FIXING BOUNDARIES ===")
    print(top.to_string(index=False))
    print("\n=== ERROR BY NUMBER OF FUTURE FIXINGS ===")
    print(nfut.to_string(index=False))
    print(f"\nSaved -> {outdir}")
    print("PHASE 4D.1 COMPLETED (NO RETRAINING).")


if __name__ == "__main__":
    main()
