from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    BATCH_SIZE,
    DELTA_COLUMN,
    EXPECTED_PARAMETER_COUNT,
    INPUT_COLUMNS,
    PRICE_COLUMN,
    SPOT_INPUT_INDEX,
)
from .models import build_model, parameter_count, spot_derivative


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto/cuda/cpu.")


def fit_scaler(train_df: pd.DataFrame) -> Dict[str, object]:
    x = train_df[list(INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    y = train_df[PRICE_COLUMN].to_numpy(dtype=np.float64)
    delta = train_df[DELTA_COLUMN].to_numpy(dtype=np.float64)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))

    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 1e-12):
        raise ValueError(f"Degenerate input scaler. std={x_std.tolist()}")
    if not np.isfinite(y_std) or y_std <= 1e-12:
        raise ValueError("Degenerate price scaler.")

    # Standardized derivative: dy_std/dx_spot_std = (sigma_x_spot / sigma_y) * Delta.
    delta_std = delta * (x_std[SPOT_INPUT_INDEX] / y_std)
    delta_rms = float(np.sqrt(np.mean(delta_std**2)))
    derivative_weight = 1.0 / max(delta_rms, 1e-12)

    return {
        "input_columns": list(INPUT_COLUMNS),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
        "delta_rms_std": delta_rms,
        "derivative_weight": derivative_weight,
        "normalization": (
            "delta_std = delta * sigma_x_spot / sigma_y; "
            "derivative_weight = 1/RMS(delta_std)"
        ),
    }


def transform_df(df: pd.DataFrame, scaler: Dict[str, object]):
    x = df[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32)
    y = df[PRICE_COLUMN].to_numpy(dtype=np.float32)
    delta = df[DELTA_COLUMN].to_numpy(dtype=np.float32)

    x_mean = np.asarray(scaler["x_mean"], dtype=np.float32)
    x_std = np.asarray(scaler["x_std"], dtype=np.float32)
    y_mean = np.float32(scaler["y_mean"])
    y_std = np.float32(scaler["y_std"])

    return (
        ((x - x_mean) / x_std).astype(np.float32, copy=False),
        ((y - y_mean) / y_std).astype(np.float32, copy=False),
        (delta * (x_std[SPOT_INPUT_INDEX] / y_std)).astype(np.float32, copy=False),
    )


def make_loader(x, y, d, batch_size, shuffle, seed, pin):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(d))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )


def build_optimizer(model, params):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )


def build_scheduler(optimizer, params, max_epochs):
    if str(params["scheduler"]).lower() != "cosine":
        raise ValueError("Phase 4B expects the frozen cosine scheduler.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=float(params["learning_rate"]) * 1e-2,
    )


def evaluate(model, loader, device, scaler):
    model.eval()
    y_std = float(scaler["y_std"])
    x_std_spot = float(scaler["x_std"][SPOT_INPUT_INDEX])
    dw = float(scaler["derivative_weight"])

    n = 0
    price_sq_std = price_sq = price_abs = 0.0
    delta_sq = delta_abs = diff_sq_weighted = 0.0

    for xb, yb, db in loader:
        xb = xb.to(device, non_blocking=True).requires_grad_(True)
        yb = yb.to(device, non_blocking=True)
        db = db.to(device, non_blocking=True)

        pred, grad = spot_derivative(model, xb, SPOT_INPUT_INDEX, False)
        ep_std = pred - yb
        ed_std = grad - db

        ep = ep_std * y_std
        ed = ed_std * (y_std / x_std_spot)

        price_sq_std += float(ep_std.square().sum().detach().item())
        price_sq += float(ep.square().sum().detach().item())
        price_abs += float(ep.abs().sum().detach().item())
        delta_sq += float(ed.square().sum().detach().item())
        delta_abs += float(ed.abs().sum().detach().item())
        diff_sq_weighted += float(((dw * ed_std) ** 2).sum().detach().item())
        n += int(xb.shape[0])

    return {
        "val_price_mse_std": price_sq_std / n,
        "val_price_rmse": math.sqrt(price_sq / n),
        "val_price_mae": price_abs / n,
        "val_delta_rmse": math.sqrt(delta_sq / n),
        "val_delta_mae": delta_abs / n,
        "val_diff_mse_weighted": diff_sq_weighted / n,
        "val_balanced_score": (
            0.5 * (price_sq_std / n) + 0.5 * (diff_sq_weighted / n)
        ),
    }


def train_dml_lambda_run(
    *,
    params: Dict[str, object],
    train_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    lambda_select_df: pd.DataFrame,
    seed: int,
    max_epochs: int,
    lambda_delta: float,
    device: torch.device,
    save_dir: Path,
):
    set_seed(seed)
    scaler = fit_scaler(train_df)

    xtr, ytr, dtr = transform_df(train_df, scaler)
    xcp, ycp, dcp = transform_df(checkpoint_df, scaler)
    xls, yls, dls = transform_df(lambda_select_df, scaler)

    pin = device.type == "cuda"
    train_loader = make_loader(
        xtr, ytr, dtr, BATCH_SIZE, True, seed, pin
    )
    checkpoint_loader = make_loader(
        xcp, ycp, dcp, max(BATCH_SIZE, 2048), False, seed, pin
    )
    lambda_loader = make_loader(
        xls, yls, dls, max(BATCH_SIZE, 2048), False, seed, pin
    )

    model = build_model(len(INPUT_COLUMNS), params).to(device)
    n_params = parameter_count(model)
    if n_params != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT:,} params, got {n_params:,}."
        )

    optimizer = build_optimizer(model, params)
    scheduler = build_scheduler(optimizer, params, max_epochs)
    mse = nn.MSELoss()

    dw = float(scaler["derivative_weight"])
    alpha = 1.0 / (1.0 + lambda_delta)
    beta = lambda_delta / (1.0 + lambda_delta)

    best_price_mse = float("inf")
    best_epoch = -1
    best_state = None
    best_checkpoint_metrics = None
    history = []
    tic = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for xb, yb, db in train_loader:
            xb = xb.to(device, non_blocking=True).requires_grad_(True)
            yb = yb.to(device, non_blocking=True)
            db = db.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred, grad = spot_derivative(model, xb, SPOT_INPUT_INDEX, True)
            value_loss = mse(pred, yb)
            diff_loss = mse(dw * grad, dw * db)
            loss = alpha * value_loss + beta * diff_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss lambda={lambda_delta}, seed={seed}, epoch={epoch}"
                )
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(xb)
            seen += len(xb)

        scheduler.step()
        cp_metrics = evaluate(model, checkpoint_loader, device, scaler)

        history.append({
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": running / seen,
            **{f"checkpoint_{k}": v for k, v in cp_metrics.items()},
        })

        # Select checkpoints using price MSE only.
        if cp_metrics["val_price_mse_std"] < best_price_mse:
            best_price_mse = cp_metrics["val_price_mse_std"]
            best_epoch = epoch
            best_checkpoint_metrics = copy.deepcopy(cp_metrics)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 100 == 0 or epoch == max_epochs:
            print(
                f"[lambda={lambda_delta:g} seed={seed}] "
                f"epoch={epoch:4d}/{max_epochs} "
                f"checkpoint_price_mse={cp_metrics['val_price_mse_std']:.4e} "
                f"best_epoch={best_epoch}"
            )

    if best_state is None:
        raise RuntimeError("No checkpoint selected.")

    model.load_state_dict(best_state)
    lambda_metrics = evaluate(model, lambda_loader, device, scaler)
    elapsed = time.perf_counter() - tic

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_dir / "model.pt")
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)

    result = {
        "lambda_delta": float(lambda_delta),
        "seed": int(seed),
        "alpha": float(alpha),
        "beta": float(beta),
        "max_epochs": int(max_epochs),
        "selected_epoch": int(best_epoch),
        "parameter_count": int(n_params),
        "training_seconds": float(elapsed),
        "scaler": scaler,
        "model_params": dict(params),
        "checkpoint_metrics": best_checkpoint_metrics,
        "lambda_select_metrics": lambda_metrics,
        **{f"select_{k}": v for k, v in lambda_metrics.items()},
    }
    (save_dir / "metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
