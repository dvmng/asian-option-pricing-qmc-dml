from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SeriesDiagnostics:
    n_obs: int
    start_date: str
    end_date: str
    n_missing_calendar_days: int
    price_min: float
    price_max: float
    price_mean: float
    log_return_mean: float
    log_return_std: float
    log_return_skew: float
    log_return_excess_kurtosis: float
    log_return_acf1: float


def _acf1(x: pd.Series) -> float:
    return float(x.autocorr(lag=1)) if len(x.dropna()) >= 3 else float("nan")


def summarize_series(df: pd.DataFrame, value_col: str = "price_avg") -> Dict[str, object]:
    x = df[["date", value_col]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    x = x.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if len(x) < 3:
        raise ValueError("At least 3 observations are required for diagnostics.")
    if (x[value_col] <= 0).any():
        raise ValueError("Positive prices required for log-return diagnostics.")

    lr = np.log(x[value_col]).diff().dropna()
    full_days = (x["date"].max() - x["date"].min()).days + 1
    d = SeriesDiagnostics(
        n_obs=len(x),
        start_date=x["date"].min().date().isoformat(),
        end_date=x["date"].max().date().isoformat(),
        n_missing_calendar_days=max(full_days - len(x), 0),
        price_min=float(x[value_col].min()),
        price_max=float(x[value_col].max()),
        price_mean=float(x[value_col].mean()),
        log_return_mean=float(lr.mean()),
        log_return_std=float(lr.std(ddof=1)),
        log_return_skew=float(stats.skew(lr, bias=False)) if len(lr) >= 3 else float("nan"),
        log_return_excess_kurtosis=float(stats.kurtosis(lr, fisher=True, bias=False)) if len(lr) >= 4 else float("nan"),
        log_return_acf1=_acf1(lr),
    )
    return asdict(d)


def stationarity_tests(df: pd.DataFrame, value_col: str = "price_avg") -> pd.DataFrame:
    """ADF/KPSS on log-levels and log returns; requires statsmodels."""
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError as exc:
        raise ImportError("Install statsmodels>=0.14 to run stationarity tests.") from exc

    x = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if (x <= 0).any():
        raise ValueError("Positive prices required.")
    logp = np.log(x)
    lr = logp.diff().dropna()
    rows = []
    for label, s in (("log_price", logp), ("log_return", lr)):
        if len(s) < 10:
            rows.append({"series": label, "test": "ADF", "statistic": np.nan, "p_value": np.nan, "note": "too_few_observations"})
            rows.append({"series": label, "test": "KPSS", "statistic": np.nan, "p_value": np.nan, "note": "too_few_observations"})
            continue
        adf = adfuller(s, autolag="AIC")
        rows.append({"series": label, "test": "ADF", "statistic": float(adf[0]), "p_value": float(adf[1]), "note": "H0: unit root"})
        try:
            kp = kpss(s, regression="c", nlags="auto")
            rows.append({"series": label, "test": "KPSS", "statistic": float(kp[0]), "p_value": float(kp[1]), "note": "H0: level stationary"})
        except Exception as e:
            rows.append({"series": label, "test": "KPSS", "statistic": np.nan, "p_value": np.nan, "note": f"failed: {e}"})
    return pd.DataFrame(rows)


def arch_lm_test(df: pd.DataFrame, value_col: str = "price_avg", nlags: int = 5) -> pd.DataFrame:
    try:
        from statsmodels.stats.diagnostic import het_arch
    except ImportError as exc:
        raise ImportError("Install statsmodels>=0.14 to run ARCH diagnostics.") from exc
    x = pd.to_numeric(df[value_col], errors="coerce").dropna()
    lr = np.log(x).diff().dropna()
    if len(lr) <= nlags + 3:
        return pd.DataFrame([{"test": "ARCH-LM", "lags": nlags, "lm_stat": np.nan, "lm_pvalue": np.nan, "f_stat": np.nan, "f_pvalue": np.nan, "note": "too_few_observations"}])
    lm, lmp, f, fp = het_arch(lr - lr.mean(), nlags=nlags)
    return pd.DataFrame([{"test": "ARCH-LM", "lags": nlags, "lm_stat": float(lm), "lm_pvalue": float(lmp), "f_stat": float(f), "f_pvalue": float(fp), "note": "H0: no ARCH effects"}])


def compare_gbm_logou_one_step(
    df: pd.DataFrame,
    value_col: str = "price_avg",
    train_fraction: float = 0.8,
) -> pd.DataFrame:
    """Simple temporal OOS comparison used only to inform model selection.

    GBM candidate: random walk in log price with constant drift.
    Log-OU candidate: AR(1) discretization of the log price.

    This is diagnostic evidence, not derivative-pricing calibration.
    """
    x = df[["date", value_col]].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    x = x.dropna().sort_values("date").drop_duplicates("date", keep="last")
    y = np.log(x[value_col].to_numpy(dtype=float))
    if len(y) < 20:
        raise ValueError("At least 20 observations recommended for temporal model comparison.")
    split = max(10, min(len(y) - 5, int(math.floor(train_fraction * len(y)))))

    train = y[:split]
    mu = float(np.diff(train).mean())

    lag = train[:-1]
    nxt = train[1:]
    X = np.column_stack([np.ones_like(lag), lag])
    a, b = np.linalg.lstsq(X, nxt, rcond=None)[0]

    rows = []
    for model in ("GBM_log_random_walk", "LogOU_AR1"):
        preds, actual = [], []
        for t in range(split, len(y)):
            prev = y[t - 1]
            pred = prev + mu if model == "GBM_log_random_walk" else a + b * prev
            preds.append(pred)
            actual.append(y[t])
        err = np.asarray(preds) - np.asarray(actual)
        rows.append({
            "model": model,
            "train_n": split,
            "test_n": len(err),
            "log_price_rmse": float(np.sqrt(np.mean(err**2))),
            "log_price_mae": float(np.mean(np.abs(err))),
            "ar1_intercept": float(a) if model == "LogOU_AR1" else np.nan,
            "ar1_phi": float(b) if model == "LogOU_AR1" else np.nan,
            "mean_reversion_kappa_discrete": float(-np.log(b)) if model == "LogOU_AR1" and 0 < b < 1 else np.nan,
        })
    return pd.DataFrame(rows)
