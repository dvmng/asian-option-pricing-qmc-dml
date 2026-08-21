from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from asian_simulation import generate_states, price_state_replicated
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


def generate_one_split(
    split: str,
    cfg: TFMConfig,
    output_dir: Path,
    override_engine: str | None = None,
):
    spec = _split_spec(cfg)[split]
    (
        n_states,
        outer_seed,
        history_seed,
        pricing_seed,
        paths_per_rep,
        n_reps,
        engine,
    ) = spec

    if override_engine is not None:
        engine = override_engine

    print(
        f"[{split}] states={n_states:,} | engine={engine} | "
        f"paths/rep={paths_per_rep:,} | reps={n_reps}"
    )

    states = generate_states(
        n_states=n_states,
        split=split,
        cfg=cfg,
        outer_seed=outer_seed,
        history_seed_base=history_seed,
        pricing_seed_base=pricing_seed,
    )

    rows = []
    for i, state in enumerate(states, start=1):
        priced = price_state_replicated(
            state=state,
            cfg=cfg,
            n_paths_per_replication=paths_per_rep,
            n_replications=n_reps,
            engine=engine,
            seed_base=state.pricing_seed,
        )

        row = state.to_dict()
        row.update(priced)
        rows.append(row)

        if i % max(1, n_states // 20) == 0 or i == n_states:
            print(f"  priced {i:,}/{n_states:,}")

    df = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{split}.csv.gz"
    df.to_csv(out_csv, index=False, compression="gzip")

    print(f"Saved: {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate economically consistent Asian-option datasets."
    )
    parser.add_argument(
        "--preset",
        choices=["pilot", "tfm"],
        default="pilot",
        help="pilot runs quickly; tfm uses the planned final sizes.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "reference", "all"],
        default="all",
    )
    parser.add_argument(
        "--engine",
        choices=["mc", "rqmc"],
        default=None,
        help="Optional override of the configured label engine.",
    )
    parser.add_argument(
        "--output",
        default="data_simulated",
        help="Output directory.",
    )
    args = parser.parse_args()

    cfg = pilot_config() if args.preset == "pilot" else TFMConfig()
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.as_dict(), f, indent=2)

    splits = ["train", "val", "test", "reference"] if args.split == "all" else [args.split]
    for split in splits:
        generate_one_split(split, cfg, output_dir, override_engine=args.engine)


if __name__ == "__main__":
    main()
