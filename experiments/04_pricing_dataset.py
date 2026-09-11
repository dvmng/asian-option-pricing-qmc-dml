from __future__ import annotations

from tfm_project.paths import ProjectPaths

P = ProjectPaths.discover()

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


from compute.compute_state_generation import generate_states
from compute.logou_gpu import (
    device_summary,
    price_states_replicated_gpu,
    resolve_device,
)
from compute.phase4_config import ComputeDatasetConfig, smoke_config


def split_spec(cfg):
    return {
        "train": (
            cfg.n_train, cfg.seed_train_outer, cfg.seed_train_history,
            cfg.seed_train_pricing, cfg.paths_per_scenario,
            cfg.label_replications, cfg.label_engine,
        ),
        "val": (
            cfg.n_val, cfg.seed_val_outer, cfg.seed_val_history,
            cfg.seed_val_pricing, cfg.paths_per_scenario,
            cfg.label_replications, cfg.label_engine,
        ),
        "test": (
            cfg.n_test, cfg.seed_test_outer, cfg.seed_test_history,
            cfg.seed_test_pricing, cfg.paths_per_scenario,
            cfg.label_replications, cfg.label_engine,
        ),
        "reference": (
            cfg.n_reference, cfg.seed_reference_outer, cfg.seed_reference_history,
            cfg.seed_reference_pricing, cfg.reference_paths_per_replication,
            cfg.reference_replications, cfg.reference_engine,
        ),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_split(split, cfg, output_dir, device, dtype, batch_size):
    n_states, outer, hist, pricing, paths, reps, engine = split_spec(cfg)[split]
    print(
        f"\n[{split}] states={n_states:,} | {engine} | paths/rep={paths:,} | "
        f"reps={reps} | device={device} | dtype={dtype}"
    )

    tic = time.perf_counter()
    states = generate_states(
        n_states=n_states,
        split=split,
        cfg=cfg,
        outer_seed=outer,
        history_seed_base=hist,
        pricing_seed_base=pricing,
    )
    print(f"State generation: {time.perf_counter()-tic:.3f}s")

    pbar = tqdm(total=n_states * reps, desc=f"Pricing {split}", unit="state")
    tic = time.perf_counter()
    labels = price_states_replicated_gpu(
        states=states,
        cfg=cfg,
        n_paths_per_replication=paths,
        n_replications=reps,
        engine=engine,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        progress_callback=pbar.update,
    )
    pbar.close()
    print(f"Pricing: {time.perf_counter()-tic:.3f}s")

    rows = []
    for s, y in zip(states, labels):
        row = s.to_dict()
        row.update(y)
        rows.append(row)
    df = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{split}.csv.gz"
    df.to_csv(
        out,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(f"Saved {len(df):,} rows -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "tfm"], default="smoke")
    ap.add_argument(
        "--split", choices=["train", "val", "test", "reference", "all"],
        default="all"
    )
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = smoke_config() if args.preset == "smoke" else ComputeDatasetConfig()
    output = (
        P.assert_not_frozen_write(args.output)
        if args.output
        else P.reproduced_data / (
            "pricing_smoke" if args.preset == "smoke" else "pricing_dataset"
        )
    )

    dev = resolve_device(args.device)
    print("Compute environment:")
    for k, v in device_summary(dev).items():
        print(f"  {k}: {v}")

    output.mkdir(parents=True, exist_ok=True)
    meta = cfg.as_dict()
    meta.update({
        "preset": args.preset,
        "requested_device": args.device,
        "resolved_device": str(dev),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "state_design": (
            "6-D scrambled Sobol over S0/K,tau,r,kappa_q,theta_q_log_K,sigma_q; "
            "spot and fixed Asian contribution derived from one physical-P history"
        ),
        "pricing_history_measure": "P",
        "future_pricing_measure": "Q scenario parameters",
        "seed_scheme": (
            "collision_free_int64_v2: seed=(namespace<<32)|row_index; "
            "replication_seed=pricing_seed+replication*2**56"
        ),
    })
    config_path = output / "config.json"
    config_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    splits = ["train", "val", "test", "reference"] if args.split == "all" else [args.split]
    files = [generate_split(s, cfg, output, str(dev), args.dtype, args.batch_size) for s in splits]

    manifest = {
        "config_sha256": sha256(config_path),
        "files": {p.name: sha256(p) for p in files},
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nManifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
