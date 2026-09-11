from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import jarque_bera


def fit_ar1(x: np.ndarray):
    y = x[1:]
    X = sm.add_constant(x[:-1])
    res = sm.OLS(y, X).fit()
    a, phi = map(float, res.params)
    return res, a, phi


def stationarity_rows(x: np.ndarray, name: str):
    rows = []
    for reg in ("c", "ct"):
        try:
            adf = adfuller(x, regression=reg, autolag="AIC")
            rows.append({"series": name, "test": "ADF", "regression": reg,
                         "statistic": float(adf[0]), "p_value": float(adf[1]),
                         "null": "unit root"})
        except Exception as e:
            rows.append({"series": name, "test": "ADF", "regression": reg,
                         "statistic": np.nan, "p_value": np.nan, "null": f"ERROR: {e}"})
        try:
            kp = kpss(x, regression=reg, nlags="auto")
            rows.append({"series": name, "test": "KPSS", "regression": reg,
                         "statistic": float(kp[0]), "p_value": float(kp[1]),
                         "null": "stationary" if reg == "c" else "trend-stationary"})
        except Exception as e:
            rows.append({"series": name, "test": "KPSS", "regression": reg,
                         "statistic": np.nan, "p_value": np.nan, "null": f"ERROR: {e}"})
    return rows


def expanding_one_step(logp: np.ndarray, initial_train: int = 45):
    rows = []
    for t in range(initial_train, len(logp)):
        train = logp[:t]
        actual = float(logp[t])
        rw = float(train[-1])
        _, a, phi = fit_ar1(train)
        ar1 = float(a + phi * train[-1])
        rows.append({
            "t": t,
            "actual": actual,
            "rw_pred": rw,
            "ar1_pred": ar1,
            "rw_error": rw - actual,
            "ar1_error": ar1 - actual,
            "rw_abs_error": abs(rw - actual),
            "ar1_abs_error": abs(ar1 - actual),
            "rw_sq_error": (rw - actual) ** 2,
            "ar1_sq_error": (ar1 - actual) ** 2,
            "ar1_phi": phi,
        })
    return pd.DataFrame(rows)


def hac_mean_test(d: np.ndarray, maxlags: int = 3):
    # Tests H0: mean(loss_RW - loss_AR1) = 0 using HAC covariance.
    X = np.ones((len(d), 1))
    res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.params[0]), float(res.tvalues[0]), float(res.pvalues[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/global/ornn_h100.csv")
    p.add_argument("--output-dir", default="results/phase2_model_evidence")
    p.add_argument("--initial-train", type=int, default=45)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["price_avg"])
    pval = df["price_avg"].astype(float).to_numpy()
    logp = np.log(pval)
    ret = np.diff(logp)

    pd.DataFrame(stationarity_rows(logp, "log_price") + stationarity_rows(ret, "log_return")).to_csv(
        out / "stationarity_extended.csv", index=False
    )

    res, a, phi = fit_ar1(logp)
    ci = res.conf_int(alpha=0.05)
    phi_lo, phi_hi = map(float, ci[1])
    stable = bool(0 < phi < 1)
    kappa = -math.log(phi) if stable else float("nan")
    half_life = math.log(2) / kappa if stable and kappa > 0 else float("nan")
    theta_log = a / (1 - phi) if abs(1 - phi) > 1e-12 else float("nan")

    resid = np.asarray(res.resid, dtype=float)
    arch = het_arch(resid, nlags=min(5, max(1, len(resid)//10)))
    lb = acorr_ljungbox(ret, lags=[1, 5, 10], return_df=True)
    jb = jarque_bera(ret)

    cv = expanding_one_step(logp, initial_train=args.initial_train)
    cv.to_csv(out / "expanding_one_step.csv", index=False)

    rw_rmse = float(np.sqrt(cv["rw_sq_error"].mean()))
    ar_rmse = float(np.sqrt(cv["ar1_sq_error"].mean()))
    rw_mae = float(cv["rw_abs_error"].mean())
    ar_mae = float(cv["ar1_abs_error"].mean())

    d_sq = (cv["rw_sq_error"] - cv["ar1_sq_error"]).to_numpy(float)
    d_abs = (cv["rw_abs_error"] - cv["ar1_abs_error"]).to_numpy(float)
    sq_mean, sq_t, sq_p = hac_mean_test(d_sq)
    abs_mean, abs_t, abs_p = hac_mean_test(d_abs)

    evidence = {
        "n_obs": int(len(df)),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "ar1_full_sample": {
            "intercept": a,
            "phi": phi,
            "phi_95ci": [phi_lo, phi_hi],
            "phi_ci_includes_unit_root": bool(phi_hi >= 1.0),
            "kappa_daily_if_stationary": kappa,
            "half_life_days_if_stationary": half_life,
            "long_run_log_level_if_stationary": theta_log,
            "long_run_price_if_stationary": math.exp(theta_log) if np.isfinite(theta_log) else float("nan"),
            "r2": float(res.rsquared),
        },
        "return_distribution": {
            "mean": float(ret.mean()),
            "std": float(ret.std(ddof=1)),
            "skew": float(pd.Series(ret).skew()),
            "excess_kurtosis": float(pd.Series(ret).kurt()),
            "jarque_bera_stat": float(jb.statistic),
            "jarque_bera_p": float(jb.pvalue),
        },
        "arch_lm_on_ar1_residuals": {
            "lm_stat": float(arch[0]), "lm_p": float(arch[1]),
            "f_stat": float(arch[2]), "f_p": float(arch[3]),
        },
        "one_step_expanding_cv": {
            "initial_train": int(args.initial_train),
            "n_forecasts": int(len(cv)),
            "rw_rmse": rw_rmse,
            "ar1_rmse": ar_rmse,
            "rmse_improvement_ar1_pct": 100 * (rw_rmse - ar_rmse) / rw_rmse,
            "rw_mae": rw_mae,
            "ar1_mae": ar_mae,
            "mae_improvement_ar1_pct": 100 * (rw_mae - ar_mae) / rw_mae,
            "hac_test_sq_loss_diff_rw_minus_ar1": {"mean": sq_mean, "t": sq_t, "p": sq_p},
            "hac_test_abs_loss_diff_rw_minus_ar1": {"mean": abs_mean, "t": abs_t, "p": abs_p},
        },
        "interpretation_guardrail": (
            "Do not select a pricing measure from these historical P diagnostics alone. "
            "They only inform the physical-dynamics baseline/model-risk choice."
        ),
    }
    (out / "phase2_model_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    lb.to_csv(out / "ljung_box_returns.csv")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
