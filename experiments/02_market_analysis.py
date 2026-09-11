"""Descriptive robustness analysis of the public multi-GPU Ornn panel.

The analysis keeps GPU benchmarks separate and does not alter the H100
calibration, pricing-domain design or trained MLP/DML models.
"""

from __future__ import annotations

from tfm_project.paths import ProjectPaths

P = ProjectPaths.discover()



import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_GPUS = (
    "H100 SXM",
    "H200",
    "A100 SXM4",
    "RTX 5090",
    "B200",
)

PRIMARY_GPU = "H100 SXM"
ANNUALIZATION_DAYS = 365.0


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"date", "gpu_model", "price_avg", "currency", "unit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price_avg"] = pd.to_numeric(df["price_avg"], errors="coerce")
    df["gpu_model"] = df["gpu_model"].astype(str)
    df = df.dropna(subset=["date", "gpu_model", "price_avg"])

    if (df["price_avg"] <= 0).any():
        raise ValueError("Non-positive Ornn prices detected.")

    dup = df.duplicated(["gpu_model", "date"], keep=False)
    if dup.any():
        raise ValueError(
            "Duplicate GPU/date observations detected:\n"
            + df.loc[dup, ["gpu_model", "date", "price_avg"]]
            .sort_values(["gpu_model", "date"])
            .to_string(index=False)
        )

    if df["currency"].dropna().nunique() != 1:
        raise ValueError("Multiple currencies detected in Ornn panel.")
    if df["unit"].dropna().nunique() != 1:
        raise ValueError("Multiple units detected in Ornn panel.")

    return df.sort_values(["gpu_model", "date"]).reset_index(drop=True)


def max_drawdown(price: pd.Series) -> float:
    x = price.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(x)
    drawdown = x / running_max - 1.0
    return float(drawdown.min())


def ols_ar1(log_price: np.ndarray) -> dict:
    x = np.asarray(log_price, dtype=float)
    lag = x[:-1]
    y = x[1:]

    if len(y) < 5:
        return {
            "ar1_intercept": np.nan,
            "ar1_phi": np.nan,
            "ar1_r2": np.nan,
            "ar1_resid_sd": np.nan,
            "ou_kappa_per_year_exploratory": np.nan,
            "ou_half_life_days_exploratory": np.nan,
            "ou_theta_log_exploratory": np.nan,
            "ou_long_run_price_exploratory": np.nan,
            "ou_sigma_per_sqrt_year_exploratory": np.nan,
        }

    X = np.column_stack([np.ones_like(lag), lag])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    alpha = float(beta[0])
    phi = float(beta[1])
    pred = X @ beta
    resid = y - pred

    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    resid_sd = float(np.std(resid, ddof=2)) if len(resid) > 2 else np.nan

    out = {
        "ar1_intercept": alpha,
        "ar1_phi": phi,
        "ar1_r2": r2,
        "ar1_resid_sd": resid_sd,
        "ou_kappa_per_year_exploratory": np.nan,
        "ou_half_life_days_exploratory": np.nan,
        "ou_theta_log_exploratory": np.nan,
        "ou_long_run_price_exploratory": np.nan,
        "ou_sigma_per_sqrt_year_exploratory": np.nan,
    }

    if 0.0 < phi < 1.0:
        dt = 1.0 / ANNUALIZATION_DAYS
        kappa = -math.log(phi) / dt
        half_life_days = math.log(2.0) / kappa * ANNUALIZATION_DAYS
        theta = alpha / (1.0 - phi)
        long_run = math.exp(theta)

        resid_var = float(np.var(resid, ddof=2)) if len(resid) > 2 else np.nan
        sigma = (
            math.sqrt(resid_var * 2.0 * kappa / (1.0 - phi**2))
            if np.isfinite(resid_var) and (1.0 - phi**2) > 0
            else np.nan
        )

        out.update(
            {
                "ou_kappa_per_year_exploratory": float(kappa),
                "ou_half_life_days_exploratory": float(half_life_days),
                "ou_theta_log_exploratory": float(theta),
                "ou_long_run_price_exploratory": float(long_run),
                "ou_sigma_per_sqrt_year_exploratory": float(sigma),
            }
        )

    return out


def ar1_vs_random_walk_oos(
    log_price: np.ndarray,
    train_fraction: float = 0.8,
) -> dict:
    x = np.asarray(log_price, dtype=float)
    lag = x[:-1]
    y = x[1:]

    if len(y) < 10:
        return {
            "oos_n": 0,
            "ar1_oos_rmse": np.nan,
            "rw_oos_rmse": np.nan,
            "ar1_vs_rw_rmse_improvement_pct": np.nan,
            "ar1_oos_mae": np.nan,
            "rw_oos_mae": np.nan,
            "ar1_vs_rw_mae_improvement_pct": np.nan,
        }

    n_train = int(math.floor(len(y) * train_fraction))
    n_train = min(max(n_train, 3), len(y) - 2)

    X_train = np.column_stack([np.ones(n_train), lag[:n_train]])
    beta, *_ = np.linalg.lstsq(X_train, y[:n_train], rcond=None)

    lag_test = lag[n_train:]
    y_test = y[n_train:]

    ar1_pred = beta[0] + beta[1] * lag_test
    rw_pred = lag_test

    ar1_err = y_test - ar1_pred
    rw_err = y_test - rw_pred

    ar1_rmse = float(np.sqrt(np.mean(ar1_err**2)))
    rw_rmse = float(np.sqrt(np.mean(rw_err**2)))
    ar1_mae = float(np.mean(np.abs(ar1_err)))
    rw_mae = float(np.mean(np.abs(rw_err)))

    return {
        "oos_n": int(len(y_test)),
        "ar1_oos_rmse": ar1_rmse,
        "rw_oos_rmse": rw_rmse,
        "ar1_vs_rw_rmse_improvement_pct": (
            float(100.0 * (rw_rmse - ar1_rmse) / rw_rmse)
            if rw_rmse > 0
            else np.nan
        ),
        "ar1_oos_mae": ar1_mae,
        "rw_oos_mae": rw_mae,
        "ar1_vs_rw_mae_improvement_pct": (
            float(100.0 * (rw_mae - ar1_mae) / rw_mae)
            if rw_mae > 0
            else np.nan
        ),
    }


def build_series(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    transformed_parts = []

    for gpu, g in panel.groupby("gpu_model", sort=True, observed=True):
        g = g.sort_values("date").copy()

        price = g["price_avg"].astype(float)
        log_price = np.log(price)
        log_return = log_price.diff()
        unchanged = price.diff().eq(0)
        base100 = 100.0 * price / float(price.iloc[0])

        transformed = g[["date", "gpu_model", "price_avg"]].copy()
        transformed["log_price"] = log_price
        transformed["log_return"] = log_return
        transformed["base100"] = base100
        transformed["unchanged_from_previous"] = unchanged
        transformed_parts.append(transformed)

        r = log_return.dropna()

        summary_rows.append(
            {
                "gpu_model": gpu,
                "observations": int(len(g)),
                "first_date": g["date"].min().date().isoformat(),
                "last_date": g["date"].max().date().isoformat(),
                "unique_dates": int(g["date"].nunique()),
                "mean_price_usd_per_gpu_hour": float(price.mean()),
                "min_price_usd_per_gpu_hour": float(price.min()),
                "max_price_usd_per_gpu_hour": float(price.max()),
                "price_range_pct_of_mean": float(
                    100.0 * (price.max() - price.min()) / price.mean()
                ),
                "daily_log_return_mean": float(r.mean()),
                "daily_log_return_sd": float(r.std(ddof=1)),
                "annualized_log_return_volatility": float(
                    r.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)
                ),
                "daily_log_return_skew": float(r.skew()),
                "daily_log_return_excess_kurtosis": float(r.kurt()),
                "daily_log_return_acf1": float(r.autocorr(lag=1)),
                "unchanged_day_share": float(unchanged.iloc[1:].mean()),
                "max_drawdown": max_drawdown(price),
                **ols_ar1(log_price.to_numpy()),
                **ar1_vs_random_walk_oos(log_price.to_numpy()),
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("gpu_model")
        .reset_index(drop=True)
    )

    transformed = pd.concat(transformed_parts, ignore_index=True)
    return summary, transformed


def aligned_wide(
    transformed: pd.DataFrame,
    value: str,
) -> pd.DataFrame:
    return (
        transformed
        .pivot(index="date", columns="gpu_model", values=value)
        .sort_index()
        .dropna(how="any")
    )


def save_figures(
    transformed: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for gpu, g in transformed.groupby("gpu_model", sort=True, observed=True):
        ax.plot(g["date"], g["price_avg"], label=gpu)
    ax.set_title("Ornn GPU compute rental benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD per GPU-hour")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = fig_dir / "01_ornn_gpu_price_levels.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(p)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for gpu, g in transformed.groupby("gpu_model", sort=True, observed=True):
        ax.plot(g["date"], g["base100"], label=gpu)
    ax.axhline(100.0, linewidth=1.0, alpha=0.6)
    ax.set_title("Ornn GPU benchmarks — Base 100 at sample start")
    ax.set_xlabel("Date")
    ax.set_ylabel("Base 100")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = fig_dir / "02_ornn_gpu_base100.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(p)

    vol = summary.sort_values(
        "annualized_log_return_volatility",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(
        vol["gpu_model"],
        vol["annualized_log_return_volatility"],
    )
    ax.set_title("Annualized volatility of daily log returns")
    ax.set_xlabel("GPU benchmark")
    ax.set_ylabel("Annualized volatility")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = fig_dir / "03_ornn_gpu_annualized_volatility.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(p)

    returns = aligned_wide(transformed, "log_return")
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation of daily log returns")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    p = fig_dir / "04_ornn_gpu_return_correlation.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    saved.append(p)

    return saved


def write_findings(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    s = summary.copy()

    highest_price = s.loc[
        s["mean_price_usd_per_gpu_hour"].idxmax()
    ]
    highest_vol = s.loc[
        s["annualized_log_return_volatility"].idxmax()
    ]

    support = s.loc[
        (s["ar1_phi"] > 0)
        & (s["ar1_phi"] < 1)
        & (s["ar1_vs_rw_rmse_improvement_pct"] > 0),
        "gpu_model",
    ].tolist()

    lines = [
        "# Phase 4E — Ornn multi-GPU descriptive findings",
        "",
        "## Scope",
        "",
        "This phase is descriptive robustness only. It does not replace the frozen H100 calibration, Q bands, ML datasets or trained neural networks.",
        "",
        f"- GPU benchmarks analysed: {len(s)}.",
        f"- Highest mean rental-price benchmark: **{highest_price['gpu_model']}** ({highest_price['mean_price_usd_per_gpu_hour']:.4f} USD/GPU-hour).",
        f"- Highest annualized daily-log-return volatility: **{highest_vol['gpu_model']}** ({highest_vol['annualized_log_return_volatility']:.4f}).",
        "",
        "## Exploratory mean-reversion diagnostic",
        "",
        "Descriptive support is noted only when `0 < phi < 1` and the fixed-parameter AR(1) improves OOS RMSE relative to a log-price random walk.",
        "",
    ]

    if support:
        lines.append(
            "Benchmarks meeting both descriptive conditions: "
            + ", ".join(f"**{x}**" for x in support)
            + "."
        )
    else:
        lines.append(
            "No benchmark meets both descriptive conditions in this short sample."
        )

    lines.extend(
        [
            "",
            "This is not a formal model-selection test and must not be described as proof of mean reversion.",
            "",
            "## H100",
            "",
        ]
    )

    h = s.loc[s["gpu_model"].eq(PRIMARY_GPU)]
    if not h.empty:
        h = h.iloc[0]
        lines.append(f"- AR(1) phi: {h['ar1_phi']:.6f}.")
        if np.isfinite(h["ou_half_life_days_exploratory"]):
            lines.append(
                f"- Exploratory half-life: {h['ou_half_life_days_exploratory']:.3f} days."
            )
        else:
            lines.append(
                "- Exploratory half-life: not defined because fitted phi is outside `(0,1)`."
            )
        lines.append(
            f"- OOS AR(1) vs random-walk RMSE improvement: {h['ar1_vs_rw_rmse_improvement_pct']:.3f}%."
        )
        lines.append(
            f"- Annualized log-return volatility: {h['annualized_log_return_volatility']:.4f}."
        )
        lines.append("")
        lines.append(
            "These H100 numbers use the NEW rolling multi-GPU window and must not replace the earlier frozen H100 calibration used for ML data generation."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The five GPU benchmarks are compared separately. Base-100 normalization and return correlations are used for co-movement analysis, not to assert economic interchangeability or construct a synthetic GPU index.",
        ]
    )

    path = output_dir / "phase4e_findings.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(P.data_processed / "compute_market" / "global" / "ornn_gpu_panel_long.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(P.reproduced_results / "market" / "ornn_multigpu"),
    )
    args = parser.parse_args()

    input_path = P.resolve(args.input)
    output_dir = P.assert_not_frozen_write(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(input_path)

    observed = tuple(sorted(panel["gpu_model"].unique()))
    missing_expected = sorted(set(EXPECTED_GPUS).difference(observed))
    unexpected = sorted(set(observed).difference(EXPECTED_GPUS))

    summary, transformed = build_series(panel)

    price_wide = aligned_wide(transformed, "price_avg")
    base100_wide = aligned_wide(transformed, "base100")
    return_wide = aligned_wide(transformed, "log_return")
    return_corr = return_wide.corr()

    summary.to_csv(
        output_dir / "ornn_gpu_dynamics_summary.csv",
        index=False,
    )
    transformed.to_csv(
        output_dir / "ornn_gpu_transformed_long.csv",
        index=False,
    )
    price_wide.to_csv(
        output_dir / "ornn_gpu_price_wide_common_dates.csv",
    )
    base100_wide.to_csv(
        output_dir / "ornn_gpu_base100_wide_common_dates.csv",
    )
    return_wide.to_csv(
        output_dir / "ornn_gpu_log_returns_wide_common_dates.csv",
    )
    return_corr.to_csv(
        output_dir / "ornn_gpu_return_correlation.csv",
    )

    figures = save_figures(transformed, summary, output_dir)
    findings = write_findings(summary, output_dir)

    audit = {
        "status": "PHASE4E_DESCRIPTIVE_ONLY_NO_RETRAINING",
        "input_file": str(input_path),
        "primary_gpu": PRIMARY_GPU,
        "expected_gpus": list(EXPECTED_GPUS),
        "observed_gpus": list(observed),
        "missing_expected_gpus": missing_expected,
        "unexpected_gpus": unexpected,
        "total_rows": int(len(panel)),
        "rows_by_gpu": {
            str(k): int(v)
            for k, v in panel.groupby("gpu_model").size().items()
        },
        "common_price_dates": int(len(price_wide)),
        "common_return_dates": int(len(return_wide)),
        "annualization_days": ANNUALIZATION_DAYS,
        "ar1_definition": "log(P_t)=alpha+phi*log(P_{t-1})+epsilon_t",
        "oos_comparison": (
            "fit first 80% of AR1 pairs once; evaluate last 20%; "
            "benchmark is log-price random walk"
        ),
        "frozen_protocol_note": (
            "No Phase 2-4 calibration, bootstrap band, dataset, lambda, "
            "architecture, checkpoint, test result or prediction is changed."
        ),
        "figures": [str(x) for x in figures],
        "findings_file": str(findings),
    }

    (output_dir / "phase4e_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "gpu_model",
        "observations",
        "mean_price_usd_per_gpu_hour",
        "annualized_log_return_volatility",
        "daily_log_return_acf1",
        "unchanged_day_share",
        "ar1_phi",
        "ou_half_life_days_exploratory",
        "ar1_vs_rw_rmse_improvement_pct",
        "max_drawdown",
    ]

    print("\n=== PHASE 4E — ORNN MULTI-GPU DYNAMICS ===")
    print(summary[display_cols].to_string(index=False))

    print("\n=== DAILY LOG-RETURN CORRELATION ===")
    print(return_corr.round(4).to_string())

    print("\n=== AUDIT ===")
    print(json.dumps(audit, indent=2))

    if missing_expected:
        raise SystemExit(
            "Phase 4E audit failed: missing expected GPUs: "
            + ", ".join(missing_expected)
        )

    if len(panel) != sum(audit["rows_by_gpu"].values()):
        raise SystemExit("Phase 4E audit failed: row-count inconsistency.")

    if return_corr.shape != (len(observed), len(observed)):
        raise SystemExit(
            "Phase 4E audit failed: incomplete correlation matrix."
        )

    print("\nPHASE 4E PASSED — DESCRIPTIVE ROBUSTNESS ONLY.")


if __name__ == "__main__":
    main()
