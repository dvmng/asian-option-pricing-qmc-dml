from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from compute_ml.config import INPUT_COLUMNS, SPOT_INPUT_INDEX
from compute_ml.models import build_model, spot_derivative


@dataclass
class FrozenSurrogate:
    model_type: str
    train_size: int
    seed: int
    model: torch.nn.Module
    scaler: dict
    device: torch.device

    @classmethod
    def load(
        cls,
        run_dir: Path,
        device: torch.device,
    ) -> "FrozenSurrogate":
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        state = torch.load(run_dir / "model.pt", map_location=device)

        model = build_model(
            len(INPUT_COLUMNS),
            meta["model_params"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()

        return cls(
            model_type=str(meta["model_type"]),
            train_size=int(meta["train_size"]),
            seed=int(meta["seed"]),
            model=model,
            scaler=meta["scaler"],
            device=device,
        )

    def predict_price_delta(
        self,
        x_raw: np.ndarray,
        batch_size: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_raw = np.asarray(x_raw, dtype=np.float32)
        if x_raw.ndim != 2 or x_raw.shape[1] != len(INPUT_COLUMNS):
            raise ValueError(
                f"Expected [n,{len(INPUT_COLUMNS)}] raw features, got {x_raw.shape}."
            )

        x_mean = np.asarray(self.scaler["x_mean"], dtype=np.float32)
        x_std = np.asarray(self.scaler["x_std"], dtype=np.float32)
        y_mean = float(self.scaler["y_mean"])
        y_std = float(self.scaler["y_std"])
        x_std_spot = float(self.scaler["x_std"][SPOT_INPUT_INDEX])

        xs = (x_raw - x_mean) / x_std
        prices, deltas = [], []

        for start in range(0, len(xs), batch_size):
            xb = torch.from_numpy(xs[start:start+batch_size]).to(
                self.device
            ).requires_grad_(True)
            pred_std, grad_std = spot_derivative(
                self.model,
                xb,
                SPOT_INPUT_INDEX,
                create_graph=False,
            )

            price = pred_std.detach().cpu().numpy().astype(np.float64) * y_std + y_mean
            delta = (
                grad_std.detach().cpu().numpy().astype(np.float64)
                * y_std / x_std_spot
            )
            prices.append(price)
            deltas.append(delta)

        return np.concatenate(prices), np.concatenate(deltas)


def discover_final_surrogates(
    results_dir: Path,
    model_type: str,
    train_size: int = 65536,
    seeds: Iterable[int] = (1701, 2701, 3701, 4701, 5701),
) -> list[Path]:
    paths = []
    for seed in seeds:
        p = results_dir / f"{model_type}_n{train_size}_seed{int(seed)}"
        if not (p / "COMPLETED").exists():
            raise FileNotFoundError(f"Missing completed run: {p}")
        paths.append(p)
    return paths
