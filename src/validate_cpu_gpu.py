from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from asian_simulation import generate_states, price_state_replicated
from asian_simulation_gpu import (
    device_summary,
    price_states_replicated_gpu,
    resolve_device,
)
from config import pilot_config


def main():
    parser = argparse.ArgumentParser(
        description="Validate PyTorch/CUDA pricing against the validated NumPy/SciPy CPU engine."
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--n-states", type=int, default=32)
    parser.add_argument("--paths", type=int, default=2**11)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = pilot_config()
    dev = resolve_device(args.device)

    print("Compute environment:")
    for k, v in device_summary(dev).items():
        print(f"  {k}: {v}")

    states = generate_states(
        n_states=args.n_states,
        split="cpu_gpu_check",
        cfg=cfg,
        outer_seed=771001,
        history_seed_base=772001,
        pricing_seed_base=773001,
    )

    print("\nRunning validated CPU engine...")
    tic = time.perf_counter()
    cpu = [
        price_state_replicated(
            state=s,
            cfg=cfg,
            n_paths_per_replication=args.paths,
            n_replications=1,
            engine="rqmc",
            seed_base=s.pricing_seed,
        )
        for s in states
    ]
    cpu_seconds = time.perf_counter() - tic

    print("Running PyTorch engine...")
    tic = time.perf_counter()
    gpu = price_states_replicated_gpu(
        states=states,
        cfg=cfg,
        n_paths_per_replication=args.paths,
        n_replications=1,
        engine="rqmc",
        device=str(dev),
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    gpu_seconds = time.perf_counter() - tic

    df = pd.DataFrame(
        {
            "scenario_id": [s.scenario_id for s in states],
            "price_cpu": [x["price_K"] for x in cpu],
            "price_torch": [x["price_K"] for x in gpu],
            "delta_cpu": [x["delta"] for x in cpu],
            "delta_torch": [x["delta"] for x in gpu],
        }
    )
    df["abs_price_diff"] = (df["price_cpu"] - df["price_torch"]).abs()
    df["abs_delta_diff"] = (df["delta_cpu"] - df["delta_torch"]).abs()

    print("\nCPU vs PyTorch agreement:")
    print(
        df[
            [
                "abs_price_diff",
                "abs_delta_diff",
            ]
        ].describe()
    )

    print(f"\nCPU time:      {cpu_seconds:.6f} s")
    print(f"PyTorch time:  {gpu_seconds:.6f} s")
    if gpu_seconds > 0:
        print(f"CPU/Torch:     {cpu_seconds / gpu_seconds:.3f}x")

    max_price = float(df["abs_price_diff"].max())
    max_delta = float(df["abs_delta_diff"].max())

    # float32 should normally be far inside these tolerances for this problem.
    tol = 1e-5 if args.dtype == "float32" else 1e-10

    print(f"\nMax price difference: {max_price:.3e}")
    print(f"Max Delta difference: {max_delta:.3e}")
    print(f"Validation tolerance: {tol:.1e}")

    if max_price <= tol and max_delta <= tol:
        print("\nCPU/GPU NUMERICAL VALIDATION PASSED.")
    else:
        print(
            "\nCPU/GPU VALIDATION NEEDS REVIEW. "
            "Do not generate the final dataset yet."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
