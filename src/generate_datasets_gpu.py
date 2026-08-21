from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from asian_simulation import generate_states
from asian_simulation_gpu import (
    device_summary,
    price_states_replicated_gpu,
    resolve_device,
)
from config import TFMConfig, pilot_config


def _split_spec(cfg: TFMConfig) -> Dict[str, tuple]:
    return {
        "train": (
            cfg.n_train,
            cfg.seed_train_outer,
            cfg.seed_train_history,
            cfg.seed_train_pricing,
            cfg.paths_per_scenario,
            cfg.label_replications,
            cfg.label_engine,
        ),
        "val": (
            cfg.n_val,
            cfg.seed_val_outer,
            cfg.seed_val_history,
            cfg.seed_val_pricing,
            cfg.paths_per_scenario,
            cfg.label_replications,
            cfg.label_engine,
        ),
        "test": (
            cfg.n_test,
            cfg.seed_test_outer,
            cfg.seed_test_history,
            cfg.seed_test_pricing,
            cfg.paths_per_scenario,
            cfg.label_replications,
            cfg.label_engine,
        ),
        "reference": (
            cfg.n_reference,
            cfg.seed_reference_outer,
            cfg.seed_reference_history,
            cfg.seed_reference_pricing,
            cfg.reference_paths_per_replication,
            cfg.reference_replications,
            cfg.reference_engine,
        ),
    }


def generate_one_split_gpu(
    split: str,
    cfg: TFMConfig,
    output_dir: Path,
    device: str,
    dtype: str,
    batch_size: int,
    override_engine: str | None = None,
):
    (
        n_states,
        outer_seed,
        history_seed,
        pricing_seed,
        paths_per_rep,
        n_reps,
        engine,
    ) = _split_spec(cfg)[split]

    if override_engine is not None:
        engine = override_engine

    print(
        f"\n[{split}] states={n_states:,} | engine={engine} | "
        f"paths/rep={paths_per_rep:,} | reps={n_reps} | "
        f"device={device} | dtype={dtype} | batch={batch_size}"
    )

    tic = time.perf_counter()
    states = generate_states(
        n_states=n_states,
        split=split,
        cfg=cfg,
        outer_seed=outer_seed,
        history_seed_base=history_seed,
        pricing_seed_base=pricing_seed,
    )
    state_seconds = time.perf_counter() - tic
    print(f"State generation: {state_seconds:.3f} s")

    total_work = n_states * n_reps
    pbar = tqdm(total=total_work, desc=f"Pricing {split}", unit="state")

    def update_progress(n: int):
        pbar.update(n)

    tic = time.perf_counter()
    priced_rows = price_states_replicated_gpu(
        states=states,
        cfg=cfg,
        n_paths_per_replication=paths_per_rep,
        n_replications=n_reps,
        engine=engine,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        progress_callback=update_progress,
    )
    pricing_seconds = time.perf_counter() - tic
    pbar.close()

    rows = []
    for state, priced in zip(states, priced_rows):
        row = state.to_dict()
        row.update(priced)
        rows.append(row)

    df = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{split}.csv.gz"
    df.to_csv(out_csv, index=False, compression="gzip")

    print(f"Pricing time:    {pricing_seconds:.3f} s")
    print(f"Total split:     {state_seconds + pricing_seconds:.3f} s")
    print(f"Saved:           {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Asian-option datasets with PyTorch/CUDA pricing."
    )
    parser.add_argument(
        "--preset",
        choices=["pilot", "tfm"],
        default="pilot",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "reference", "all"],
        default="all",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Use cuda to require the GPU; auto falls back to CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="float32 is recommended for RTX 4060 speed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of scenarios priced together on GPU.",
    )
    parser.add_argument(
        "--engine",
        choices=["mc", "rqmc"],
        default=None,
    )
    parser.add_argument(
        "--output",
        default="data_simulated_gpu",
    )
    args = parser.parse_args()

    cfg = pilot_config() if args.preset == "pilot" else TFMConfig()
    dev = resolve_device(args.device)

    print("Compute environment:")
    for key, value in device_summary(dev).items():
        print(f"  {key}: {value}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = cfg.as_dict()
    metadata["gpu_run"] = {
        "requested_device": args.device,
        "resolved_device": str(dev),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    splits = (
        ["train", "val", "test", "reference"]
        if args.split == "all"
        else [args.split]
    )

    for split in splits:
        generate_one_split_gpu(
            split=split,
            cfg=cfg,
            output_dir=output_dir,
            device=str(dev),
            dtype=args.dtype,
            batch_size=args.batch_size,
            override_engine=args.engine,
        )


if __name__ == "__main__":
    main()
