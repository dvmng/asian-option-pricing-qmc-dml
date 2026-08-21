from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from asian_simulation import generate_states, price_state_replicated
from config import TFMConfig, pilot_config


def benchmark_settings(preset: str):
    if preset == "pilot":
        return {
            "n_states": 12,
            "powers": list(range(8, 13)),
            "replications": 5,
            "reference_paths_per_rep": 2**12,
            "reference_replications": 8,
        }
    return {
        "n_states": 128,
        "powers": list(range(8, 16)),
        "replications": 20,
        "reference_paths_per_rep": 2**14,
        "reference_replications": 16,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Homogeneous MC vs randomized Sobol-QMC benchmark."
    )
    parser.add_argument("--preset", choices=["pilot", "tfm"], default="pilot")
    parser.add_argument("--output", default="results_mc_qmc")
    args = parser.parse_args()

    cfg = pilot_config() if args.preset == "pilot" else TFMConfig()
    s = benchmark_settings(args.preset)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Independent benchmark states: not train/val/test/reference.
    states = generate_states(
        n_states=s["n_states"],
        split="benchmark",
        cfg=cfg,
        outer_seed=991001,
        history_seed_base=992001,
        pricing_seed_base=993001,
    )

    # High-precision RQMC reference.
    references = {}
    print("Building high-precision references...")
    for i, state in enumerate(states, 1):
        ref = price_state_replicated(
            state=state,
            cfg=cfg,
            n_paths_per_replication=s["reference_paths_per_rep"],
            n_replications=s["reference_replications"],
            engine="rqmc",
            seed_base=state.pricing_seed + 50_000_000,
        )
        references[state.scenario_id] = ref
        print(f"  reference {i}/{len(states)}")

    rows = []
    for engine in ["mc", "rqmc"]:
        for p in s["powers"]:
            n_paths = 2**p
            for state in states:
                for rep in range(s["replications"]):
                    seed = state.pricing_seed + rep * 1_000_003 + (0 if engine == "mc" else 700_000_000)

                    tic = time.perf_counter()
                    est = price_state_replicated(
                        state=state,
                        cfg=cfg,
                        n_paths_per_replication=n_paths,
                        n_replications=1,
                        engine=engine,
                        seed_base=seed,
                    )
                    elapsed = time.perf_counter() - tic

                    ref = references[state.scenario_id]

                    rows.append(
                        {
                            "scenario_id": state.scenario_id,
                            "engine": engine,
                            "n_paths": n_paths,
                            "replication": rep,
                            "price_K": est["price_K"],
                            "delta": est["delta"],
                            "reference_price_K": ref["price_K"],
                            "reference_delta": ref["delta"],
                            "abs_price_error": abs(est["price_K"] - ref["price_K"]),
                            "abs_delta_error": abs(est["delta"] - ref["delta"]),
                            "time_seconds": elapsed,
                            "S_t_K": state.S_t_K,
                            "A_t_K": state.A_t_K,
                            "C_t": state.C_t,
                            "tau": state.tau,
                            "r": state.r,
                            "sigma": state.sigma,
                            "n_fix_future": state.n_fix_future,
                        }
                    )

            print(f"Finished engine={engine}, N=2^{p}={n_paths:,}")

    raw = pd.DataFrame(rows)
    raw_path = outdir / "mc_qmc_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby(["engine", "n_paths"], as_index=False)
        .agg(
            mean_abs_price_error=("abs_price_error", "mean"),
            median_abs_price_error=("abs_price_error", "median"),
            mean_abs_delta_error=("abs_delta_error", "mean"),
            median_abs_delta_error=("abs_delta_error", "median"),
            mean_time_seconds=("time_seconds", "mean"),
        )
    )
    summary_path = outdir / "mc_qmc_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary)
    print(f"\nSaved:\n - {raw_path}\n - {summary_path}")


if __name__ == "__main__":
    main()
