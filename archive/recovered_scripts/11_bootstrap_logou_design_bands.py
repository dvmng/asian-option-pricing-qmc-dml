from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORNN = ROOT / "data" / "raw" / "global" / "ornn_h100.csv"
OUT = ROOT / "results" / "phase37_bootstrap_bands"


def fit_ar1_logou(prices: np.ndarray, dt_years: float) -> dict:
    x = np.log(np.asarray(prices, dtype=float))
    y = x[1:]
    X = np.column_stack([np.ones(len(x)-1), x[:-1]])
    intercept, phi = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ np.array([intercept, phi])

    if not (0.0 < phi < 1.0):
        raise ValueError(f"Point estimate is not stationary: phi={phi}")

    kappa = -np.log(phi) / dt_years
    theta = intercept / (1.0 - phi)

    # OLS residual standard error: sqrt(SSE / (n_pairs - 2))
    residual_std = np.sqrt(np.sum(resid**2) / (len(resid) - 2))
    sigma = residual_std * np.sqrt(2.0 * kappa / (1.0 - phi**2))

    return {
        "x": x,
        "intercept": float(intercept),
        "phi": float(phi),
        "resid": resid,
        "kappa": float(kappa),
        "theta_log": float(theta),
        "long_run_price": float(np.exp(theta)),
        "residual_std": float(residual_std),
        "sigma": float(sigma),
        "half_life_days": float(np.log(2.0) / kappa * 365.0),
    }


def residual_bootstrap(
    prices: np.ndarray,
    dt_years: float,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    fit = fit_ar1_logou(prices, dt_years)
    x = fit["x"]
    a = fit["intercept"]
    phi = fit["phi"]
    residuals = np.asarray(fit["resid"], dtype=float)
    residuals = residuals - residuals.mean()

    rng = np.random.default_rng(seed)
    n = len(x)
    rows = []
    n_nonstationary = 0

    for b in range(n_boot):
        eps = rng.choice(residuals, size=n-1, replace=True)

        xb = np.empty(n, dtype=float)
        xb[0] = x[0]
        for t in range(1, n):
            xb[t] = a + phi * xb[t-1] + eps[t-1]

        yb = xb[1:]
        Xb = np.column_stack([np.ones(n-1), xb[:-1]])
        ab, phib = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        rb = yb - Xb @ np.array([ab, phib])

        if not (0.0 < phib < 1.0):
            n_nonstationary += 1
            continue

        kappab = -np.log(phib) / dt_years
        thetab = ab / (1.0 - phib)
        residual_std_b = np.sqrt(np.sum(rb**2) / (len(rb) - 2))
        sigmab = residual_std_b * np.sqrt(
            2.0 * kappab / (1.0 - phib**2)
        )

        rows.append(
            {
                "bootstrap_id": b,
                "phi": phib,
                "kappa": kappab,
                "theta_log": thetab,
                "long_run_price": np.exp(thetab),
                "sigma": sigmab,
                "half_life_days": np.log(2.0) / kappab * 365.0,
            }
        )

    boot = pd.DataFrame(rows)

    meta = {
        "n_boot_requested": int(n_boot),
        "n_stationary_bootstrap": int(len(boot)),
        "n_nonstationary_discarded": int(n_nonstationary),
        "nonstationary_share": float(n_nonstationary / n_boot),
        "seed": int(seed),
        "bootstrap_type": (
            "residual bootstrap of fitted AR(1) in log price; centered residuals; "
            "AR(1) re-estimated on every bootstrap path"
        ),
        "important_interpretation": (
            "Percentile bands are used as surrogate-model DESIGN RANGES, not as "
            "formal confidence intervals for pricing-measure Q parameters."
        ),
    }
    return boot, {**fit, **meta}


def percentile_table(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in (0.80, 0.90, 0.95):
        alpha = 1.0 - level
        qlo = 100.0 * alpha / 2.0
        qhi = 100.0 * (1.0 - alpha / 2.0)

        for parameter in (
            "phi", "kappa", "sigma", "theta_log",
            "long_run_price", "half_life_days"
        ):
            vals = boot[parameter].to_numpy(dtype=float)
            lo, med, hi = np.percentile(vals, [qlo, 50.0, qhi])
            rows.append(
                {
                    "central_probability": level,
                    "lower_percentile": qlo,
                    "upper_percentile": qhi,
                    "parameter": parameter,
                    "lower": lo,
                    "median": med,
                    "upper": hi,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ornn", type=Path, default=DEFAULT_ORNN)
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--dt-days", type=float, default=1.0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ornn)
    prices = df["price_avg"].astype(float).to_numpy()
    dt_years = args.dt_days / 365.0

    boot, meta = residual_bootstrap(
        prices=prices,
        dt_years=dt_years,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    bands = percentile_table(boot)

    point = {
        k: v for k, v in meta.items()
        if k not in {"x", "resid"}
    }

    point["n_obs"] = int(len(prices))
    point["start"] = str(df["date"].iloc[0])
    point["end"] = str(df["date"].iloc[-1])
    point["dt_years"] = dt_years

    # Frozen design band is P10-P90 of bootstrap kappa/sigma.
    b80 = bands[bands["central_probability"].eq(0.80)].set_index("parameter")
    design = {
        "status": "FROZEN_PHASE4_LOGOU_DESIGN_BAND",
        "kappa_q": {
            "raw_p10": float(b80.loc["kappa", "lower"]),
            "raw_p90": float(b80.loc["kappa", "upper"]),
            "rounded_training_lower": 25.0,
            "rounded_training_upper": 87.0,
        },
        "sigma_q": {
            "raw_p10": float(b80.loc["sigma", "lower"]),
            "raw_p90": float(b80.loc["sigma", "upper"]),
            "rounded_training_lower": 0.55,
            "rounded_training_upper": 0.71,
        },
        "theta_q_design": {
            "basis": (
                "pricing-measure/model-risk scenario range based on the observed "
                "Ornn H100 price range, not the bootstrap distribution"
            ),
            "reference_strike": float(meta["long_run_price"]),
            "low_long_run_price": float(df["price_avg"].min()),
            "central_long_run_price": float(meta["long_run_price"]),
            "high_long_run_price": float(df["price_avg"].max()),
            "theta_log_K_low": float(
                np.log(float(df["price_avg"].min()) / meta["long_run_price"])
            ),
            "theta_log_K_central": 0.0,
            "theta_log_K_high": float(
                np.log(float(df["price_avg"].max()) / meta["long_run_price"])
            ),
        },
        "interpretation": (
            "P10-P90 is a central bootstrap DESIGN BAND chosen to concentrate "
            "Sobol/DML training density on the bulk of empirically supported "
            "stationary parameter uncertainty. P5-P95 and P2.5-P97.5 are retained "
            "as sensitivity diagnostics, not as the main surrogate training domain."
        ),
    }

    boot.to_csv(OUT / "bootstrap_draws.csv", index=False)
    bands.to_csv(OUT / "bootstrap_band_sensitivity.csv", index=False)
    (OUT / "point_estimate_and_bootstrap_meta.json").write_text(
        json.dumps(point, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "frozen_phase4_design_band.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== POINT ESTIMATE ===")
    print(json.dumps(point, ensure_ascii=False, indent=2))

    print("\n=== BOOTSTRAP BAND SENSITIVITY ===")
    show = bands[
        bands["parameter"].isin(["kappa", "sigma", "half_life_days"])
    ].copy()
    print(show.to_string(index=False))

    print("\n=== FROZEN PHASE-4 DESIGN BAND ===")
    print(json.dumps(design, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
