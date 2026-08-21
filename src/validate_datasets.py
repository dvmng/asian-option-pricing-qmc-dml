from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from asian_simulation import AsianState, finite_difference_delta
from config import TFMConfig, pilot_config


REQUIRED_COLUMNS = {
    "scenario_id",
    "split",
    "S0_K",
    "S_t_K",
    "C_t",
    "A_t_K",
    "t",
    "tau",
    "n_fix_past",
    "n_fix_future",
    "r",
    "sigma",
    "q",
    "state_seed",
    "pricing_seed",
    "price_K",
    "delta",
    "pricing_engine",
    "n_paths",
    "n_replications",
}


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_checks(df: pd.DataFrame, cfg: TFMConfig, name: str) -> list[str]:
    issues = []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        issues.append(f"{name}: missing columns {sorted(missing)}")
        return issues

    if df["scenario_id"].duplicated().any():
        issues.append(f"{name}: duplicated scenario_id")

    if (df["S_t_K"] <= 0).any():
        issues.append(f"{name}: non-positive S_t/K")

    if (df["sigma"] <= 0).any():
        issues.append(f"{name}: non-positive sigma")

    if not ((df["tau"] >= cfg.tau_min - 1e-12) & (df["tau"] <= cfg.tau_max + 1e-12)).all():
        issues.append(f"{name}: tau outside configured range")

    if not ((df["S_t_K"] >= cfg.S_t_K_min - 1e-12) & (df["S_t_K"] <= cfg.S_t_K_max + 1e-12)).all():
        issues.append(f"{name}: S_t/K outside configured range")

    has_past = df["n_fix_past"] > 0
    if has_past.any():
        a = df.loc[has_past, "A_t_K"]
        if not ((a >= cfg.A_t_K_min - 1e-12) & (a <= cfg.A_t_K_max + 1e-12)).all():
            issues.append(f"{name}: A_t/K outside configured range")

        implied_c = (
            df.loc[has_past, "n_fix_past"]
            / cfg.n_fixings
            * df.loc[has_past, "A_t_K"]
        )
        if not np.allclose(
            implied_c.to_numpy(),
            df.loc[has_past, "C_t"].to_numpy(),
            atol=1e-10,
            rtol=1e-10,
        ):
            issues.append(f"{name}: C_t != (n_fix_past/N) * A_t/K")

    if not (df["n_fix_past"] + df["n_fix_future"] == cfg.n_fixings).all():
        issues.append(f"{name}: n_fix_past + n_fix_future != N")

    if (df["price_K"] < -1e-12).any():
        issues.append(f"{name}: negative normalized price")

    # For a call on a normalized underlying under GBM, pathwise Delta should
    # normally lie in [0,1] up to Monte Carlo noise / numerical tolerance.
    if ((df["delta"] < -0.02) | (df["delta"] > 1.02)).any():
        issues.append(f"{name}: suspicious Delta outside [-0.02,1.02]")

    if df["state_seed"].duplicated().any():
        issues.append(f"{name}: duplicated state_seed within split")

    if df["pricing_seed"].duplicated().any():
        issues.append(f"{name}: duplicated pricing_seed within split")

    return issues


def leakage_checks(dfs: dict[str, pd.DataFrame]) -> list[str]:
    issues = []
    names = list(dfs)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]

            overlap_ids = set(dfs[a]["scenario_id"]).intersection(dfs[b]["scenario_id"])
            if overlap_ids:
                issues.append(f"Leakage: scenario_id overlap between {a} and {b}")

            overlap_state_seeds = set(dfs[a]["state_seed"]).intersection(dfs[b]["state_seed"])
            if overlap_state_seeds:
                issues.append(f"Leakage: state_seed overlap between {a} and {b}")

            overlap_price_seeds = set(dfs[a]["pricing_seed"]).intersection(dfs[b]["pricing_seed"])
            if overlap_price_seeds:
                issues.append(f"Leakage: pricing_seed overlap between {a} and {b}")

    return issues


def row_to_state(row: pd.Series) -> AsianState:
    return AsianState(
        scenario_id=str(row.scenario_id),
        split=str(row.split),
        S0_K=float(row.S0_K),
        S_t_K=float(row.S_t_K),
        C_t=float(row.C_t),
        A_t_K=float(row.A_t_K) if pd.notna(row.A_t_K) else np.nan,
        t=float(row.t),
        tau=float(row.tau),
        n_fix_past=int(row.n_fix_past),
        n_fix_future=int(row.n_fix_future),
        r=float(row.r),
        sigma=float(row.sigma),
        q=float(row.q),
        state_seed=int(row.state_seed),
        pricing_seed=int(row.pricing_seed),
    )


def finite_difference_spot_check(
    df: pd.DataFrame,
    cfg: TFMConfig,
    n_rows: int = 20,
    paths: int = 2**14,
) -> pd.DataFrame:
    sample = df.sample(n=min(n_rows, len(df)), random_state=12345).copy()

    out = []
    for _, row in sample.iterrows():
        state = row_to_state(row)
        delta_fd = finite_difference_delta(
            state,
            cfg,
            n_paths=paths,
            engine="rqmc",
            seed=state.pricing_seed + 777_777,
        )
        out.append(
            {
                "scenario_id": state.scenario_id,
                "delta_pathwise_label": float(row.delta),
                "delta_finite_difference": delta_fd,
                "abs_diff": abs(float(row.delta) - delta_fd),
            }
        )

    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_simulated")
    parser.add_argument("--preset", choices=["pilot", "tfm"], default="pilot")
    parser.add_argument("--fd-check", action="store_true")
    args = parser.parse_args()

    cfg = pilot_config() if args.preset == "pilot" else TFMConfig()
    data_dir = Path(args.data_dir)

    dfs = {}
    all_issues = []

    for split in ["train", "val", "test", "reference"]:
        path = data_dir / f"{split}.csv.gz"
        if path.exists():
            df = load_split(path)
            dfs[split] = df
            all_issues.extend(basic_checks(df, cfg, split))

    all_issues.extend(leakage_checks(dfs))

    if all_issues:
        print("VALIDATION ISSUES:")
        for item in all_issues:
            print(" -", item)
    else:
        print("All basic consistency and anti-leakage checks passed.")

    if args.fd_check and "reference" in dfs:
        fd = finite_difference_spot_check(dfs["reference"], cfg)
        out = data_dir / "delta_pathwise_vs_fd.csv"
        fd.to_csv(out, index=False)
        print("\nPathwise vs finite-difference check:")
        print(fd.describe(include="all"))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
