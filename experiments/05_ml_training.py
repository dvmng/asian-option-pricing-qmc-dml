from __future__ import annotations

from tfm_project.paths import ProjectPaths

P = ProjectPaths.discover()

import argparse
import gzip
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from compute_ml.config import (
    DELTA_COLUMN,
    EXPECTED_PARAMETER_COUNT,
    INPUT_COLUMNS,
    PRICE_COLUMN,
    SPOT_INPUT_INDEX,
    TRAIN_ORDER_SEED,
)
from compute_ml.models import build_model, parameter_count, spot_derivative
from compute_ml.training import fit_scaler, transform_df


DEFAULT_PROTOCOL = P.protocols / "ml_protocol.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_payload_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reproduced_split(path: Path, canonical: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not canonical.exists():
        raise FileNotFoundError(canonical)

    if gzip_payload_sha256(path) == gzip_payload_sha256(canonical):
        return

    reference = pd.read_csv(canonical)
    candidate = pd.read_csv(path)
    if list(reference.columns) != list(candidate.columns):
        raise RuntimeError(f"{label}: column mismatch vs frozen dataset.")
    if len(reference) != len(candidate):
        raise RuntimeError(f"{label}: row-count mismatch vs frozen dataset.")

    for column in reference.columns:
        if (
            pd.api.types.is_numeric_dtype(reference[column])
            and pd.api.types.is_numeric_dtype(candidate[column])
        ):
            a = reference[column].to_numpy(dtype=np.float64)
            b = candidate[column].to_numpy(dtype=np.float64)
            if not np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True):
                raise RuntimeError(
                    f"{label}: numeric mismatch in column {column!r} "
                    "vs frozen dataset."
                )
        elif not reference[column].astype(str).equals(candidate[column].astype(str)):
            raise RuntimeError(
                f"{label}: non-numeric mismatch in column {column!r} "
                "vs frozen dataset."
            )


def set_reproducible_seed(seed: int) -> None:
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


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def load_protocol(path: Path) -> tuple[dict, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    p = load_json(path)
    if p.get("status") != "FROZEN_BEFORE_COMPUTE_FINAL_TRAINING":
        raise RuntimeError(f"Unexpected protocol status: {p.get('status')!r}")
    if p.get("checkpoint_metric") != "val_price_mse_std":
        raise RuntimeError("Final training requires val_price_mse_std checkpoint.")
    if p.get("no_early_stopping") is not True:
        raise RuntimeError("Final training requires no early stopping.")
    if p.get("paired_initialization_and_shuffle") is not True:
        raise RuntimeError("Paired design is not frozen.")
    if int(p.get("expected_parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Frozen parameter count mismatch.")
    return p, sha256_file(path)


def verify_training_inputs(protocol: dict, data_dir: Path) -> None:
    source = protocol["source_hashes"]
    canonical_dir = P.pricing_dataset.resolve()

    for split in ("train", "val"):
        path = data_dir / f"{split}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)

        if data_dir.resolve() == canonical_dir:
            expected = source[split]["sha256"]
            if sha256_file(path) != expected:
                raise RuntimeError(f"{split}: SHA256 mismatch vs frozen protocol.")
        else:
            verify_reproduced_split(
                path,
                canonical_dir / f"{split}.csv.gz",
                split,
            )


def load_or_create_train_order(
    protocol: dict,
    prepared_dir: Path,
    n_train: int,
) -> np.ndarray:
    order_path = prepared_dir / "train_order.npy"
    canonical_path = P.ml_prepared.resolve() / "train_order.npy"

    if not order_path.exists():
        if prepared_dir.resolve() == P.ml_prepared.resolve():
            raise FileNotFoundError(order_path)
        prepared_dir.mkdir(parents=True, exist_ok=True)
        order = np.random.default_rng(TRAIN_ORDER_SEED).permutation(n_train).astype(np.int64)
        np.save(order_path, order)
        print(f"Generated deterministic train order -> {order_path}")

    order = np.load(order_path)
    if len(order) != n_train or not np.array_equal(np.sort(order), np.arange(n_train)):
        raise RuntimeError("train_order.npy is not a full permutation of training rows.")

    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    canonical_order = np.load(canonical_path)
    if not np.array_equal(order, canonical_order):
        raise RuntimeError("train_order.npy differs from the frozen nested-sample ordering.")

    if prepared_dir.resolve() == P.ml_prepared.resolve():
        expected = protocol["source_hashes"]["train_order"]["sha256"]
        if sha256_file(order_path) != expected:
            raise RuntimeError("train_order: SHA256 mismatch vs frozen protocol.")

    return order


def make_loader(x, y, d, batch_size, shuffle, seed, pin_memory):
    ds = TensorDataset(
        torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(d)
    )
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=min(int(batch_size), len(ds)),
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_optimizer(model, params):
    if str(params["optimizer"]).lower() != "adamw":
        raise RuntimeError("Frozen compute protocol expects AdamW.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )


def build_scheduler(optimizer, params, max_epochs):
    if str(params["scheduler"]).lower() != "cosine":
        raise RuntimeError("Frozen compute protocol expects cosine scheduler.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(max_epochs),
        eta_min=float(params["learning_rate"]) * 1e-2,
    )


def validation_metrics(model, loader, device, scaler) -> Dict[str, float]:
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
        pred, grad = spot_derivative(
            model, xb, SPOT_INPUT_INDEX, create_graph=False
        )
        ep_std = pred - yb
        ed_std = grad - db
        ep = ep_std * y_std
        ed = ed_std * (y_std / x_std_spot)

        price_sq_std += float(ep_std.square().sum().detach().item())
        price_sq += float(ep.square().sum().detach().item())
        price_abs += float(ep.abs().sum().detach().item())
        delta_sq += float(ed.square().sum().detach().item())
        delta_abs += float(ed.abs().sum().detach().item())
        diff_sq_weighted += float(
            ((dw * ed_std) ** 2).sum().detach().item()
        )
        n += int(xb.shape[0])

    p_mse_std = price_sq_std / n
    d_mse_w = diff_sq_weighted / n
    return {
        "val_price_mse_std": p_mse_std,
        "val_price_rmse": math.sqrt(price_sq / n),
        "val_price_mae": price_abs / n,
        "val_delta_rmse": math.sqrt(delta_sq / n),
        "val_delta_mae": delta_abs / n,
        "val_diff_mse_weighted": d_mse_w,
        "val_balanced_score": 0.5 * p_mse_std + 0.5 * d_mse_w,
    }


def train_one(
    *,
    model_type: str,
    train_size: int,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_order: np.ndarray,
    output_root: Path,
    device: torch.device,
    protocol: dict,
    protocol_path: Path,
    protocol_hash: str,
    max_epochs: int,
    smoke: bool,
    force: bool,
) -> Path:
    if model_type not in {"mlp", "dml"}:
        raise ValueError("model_type must be mlp/dml.")

    name = f"{model_type}_n{train_size}_seed{seed}"
    if smoke:
        name = f"SMOKE_{name}"
    run_dir = output_root / name
    done = run_dir / "COMPLETED"

    if done.exists() and not force:
        print(f"[skip] {name}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    subset = train_df.iloc[train_order[:train_size]].copy()
    scaler = fit_scaler(subset)
    xtr, ytr, dtr = transform_df(subset, scaler)
    xva, yva, dva = transform_df(val_df, scaler)

    batch = int(protocol["batch_size"])
    pin = device.type == "cuda"
    train_loader = make_loader(
        xtr, ytr, dtr, batch, True, seed, pin
    )
    val_loader = make_loader(
        xva, yva, dva, max(batch, 2048), False, seed, pin
    )

    # Reset seeds before each paired MLP/DML fit for matched initialization and shuffling.
    set_reproducible_seed(seed)
    params = dict(protocol["shared_model_params"])
    model = build_model(len(INPUT_COLUMNS), params).to(device)
    n_params = parameter_count(model)
    if n_params != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_PARAMETER_COUNT}, got {n_params}."
        )

    optimizer = build_optimizer(model, params)
    scheduler = build_scheduler(optimizer, params, max_epochs)
    mse = nn.MSELoss()

    lam = float(protocol["dml_lambda_delta"])
    alpha = 1.0 / (1.0 + lam)
    beta = lam / (1.0 + lam)
    dw = float(scaler["derivative_weight"])

    best_val = float("inf")
    best_epoch = -1
    best_metrics = None
    best_state = None
    history: List[Dict[str, float]] = []
    tic = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = price_total = d_raw_total = d_w_total = 0.0
        seen = 0

        for xb, yb, db in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            db = db.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if model_type == "dml":
                xb = xb.requires_grad_(True)
                pred, grad = spot_derivative(
                    model, xb, SPOT_INPUT_INDEX, create_graph=True
                )
                lp = mse(pred, yb)
                ld_raw = mse(grad, db)
                ld_w = mse(dw * grad, dw * db)
                loss = alpha * lp + beta * ld_w
            else:
                pred = model(xb).squeeze(-1)
                lp = mse(pred, yb)
                ld_raw = torch.zeros((), device=device)
                ld_w = torch.zeros((), device=device)
                loss = lp

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{name}: non-finite loss at epoch {epoch}."
                )
            loss.backward()
            optimizer.step()

            bn = int(xb.shape[0])
            seen += bn
            total += float(loss.detach().item()) * bn
            price_total += float(lp.detach().item()) * bn
            d_raw_total += float(ld_raw.detach().item()) * bn
            d_w_total += float(ld_w.detach().item()) * bn

        scheduler.step()
        val = validation_metrics(model, val_loader, device, scaler)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total / seen,
            "train_price_mse_std": price_total / seen,
            "train_delta_mse_std": d_raw_total / seen,
            "train_delta_mse_weighted": d_w_total / seen,
            **val,
        }
        history.append(row)

        if val["val_price_mse_std"] < best_val:
            best_val = val["val_price_mse_std"]
            best_epoch = epoch
            best_metrics = dict(val)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 100 == 0 or epoch == max_epochs:
            print(
                f"[{name}] epoch={epoch:4d}/{max_epochs} "
                f"train={row['train_loss']:.4e} "
                f"val_price={val['val_price_mse_std']:.4e} "
                f"val_delta_rmse={val['val_delta_rmse']:.4e} "
                f"best={best_epoch}"
            )

    elapsed = time.perf_counter() - tic
    if best_state is None or best_metrics is None:
        raise RuntimeError(f"{name}: no checkpoint selected.")

    torch.save(best_state, run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    metadata = {
        "run_name": name,
        "model_type": model_type,
        "train_size": int(train_size),
        "seed": int(seed),
        "smoke_test": bool(smoke),
        "input_columns": list(INPUT_COLUMNS),
        "spot_input_index": int(SPOT_INPUT_INDEX),
        "parameter_count": int(n_params),
        "scaler": scaler,
        "model_params": params,
        "batch_size": batch,
        "max_epochs_used": int(max_epochs),
        "no_early_stopping": True,
        "checkpoint_metric": "val_price_mse_std",
        "checkpoint_split": "full val",
        "selected_epoch": int(best_epoch),
        "best_epoch": int(best_epoch),
        "best_val_price_mse_std": float(best_val),
        "selected_validation_metrics": best_metrics,
        "lambda_delta": lam,
        "dml_alpha": alpha,
        "dml_beta": beta,
        "canonical_derivative_rms_weighting": True,
        "dml_target_definition": (
            "(sigma_spot_K / sigma_price_K) * delta"
        ),
        "dml_loss_definition": (
            "alpha*price_MSE_std + beta*RMS_weighted_Delta_MSE_std"
        ),
        "training_seconds": float(elapsed),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "git_commit": git_commit(),
        "frozen_protocol_path": str(protocol_path),
        "frozen_protocol_sha256": protocol_hash,
        "frozen_at_utc": protocol["frozen_at_utc"],
        "test_reference_read_during_training": False,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    done.write_text("ok\n", encoding="utf-8")
    print(
        f"[done] {name} | selected_epoch={best_epoch} | "
        f"time={elapsed:.1f}s"
    )
    return run_dir


def validate_subset(requested, allowed, label):
    invalid = [v for v in requested if v not in allowed]
    if invalid:
        raise ValueError(
            f"{label} invalid values {invalid}; frozen allowed={allowed}."
        )
    return requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(P.pricing_dataset))
    ap.add_argument("--prepared-dir", default=str(P.ml_prepared))
    ap.add_argument("--output-dir", default=str(P.reproduced_models))
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--models", nargs="+", choices=["mlp", "dml"], default=None)
    ap.add_argument("--train-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--full-grid", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    protocol_path = P.resolve(args.protocol)
    protocol, phash = load_protocol(protocol_path)
    data_dir = P.resolve(args.data_dir)
    prep = P.resolve(args.prepared_dir)
    verify_training_inputs(protocol, data_dir)

    train_df = pd.read_csv(data_dir / "train.csv.gz")
    val_df = pd.read_csv(data_dir / "val.csv.gz")
    order = load_or_create_train_order(protocol, prep, len(train_df))

    frozen_sizes = [int(x) for x in protocol["final_train_sizes"]]
    frozen_seeds = [int(x) for x in protocol["final_training_seeds"]]

    if args.smoke:
        sizes = [frozen_sizes[0]]
        seeds = [frozen_seeds[0]]
        models = ["mlp", "dml"]
        epochs = 5
    elif args.full_grid:
        sizes = frozen_sizes
        seeds = frozen_seeds
        models = ["mlp", "dml"]
        epochs = int(protocol["final_max_epochs"])
    else:
        sizes = (
            frozen_sizes if args.train_sizes is None
            else validate_subset(args.train_sizes, frozen_sizes, "train-sizes")
        )
        seeds = (
            frozen_seeds if args.seeds is None
            else validate_subset(args.seeds, frozen_seeds, "seeds")
        )
        models = args.models or ["mlp", "dml"]
        epochs = int(protocol["final_max_epochs"])

    device = resolve_device(args.device)
    output = P.assert_not_frozen_write(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FINAL COMPUTE MLP/DML TRAINING")
    print("=" * 72)
    print(f"Protocol SHA256: {phash}")
    print(f"Device:          {device}")
    print(f"Train sizes:     {sizes}")
    print(f"Seeds:           {seeds}")
    print(f"Models:          {models}")
    print(f"Epochs:          {epochs}")
    print(f"Lambda DML:      {protocol['dml_lambda_delta']}")
    print(f"Expected runs:   {len(sizes)*len(seeds)*len(models)}")
    print("=" * 72)

    for n in sizes:
        for seed in seeds:
            for model_type in models:
                train_one(
                    model_type=model_type,
                    train_size=n,
                    seed=seed,
                    train_df=train_df,
                    val_df=val_df,
                    train_order=order,
                    output_root=output,
                    device=device,
                    protocol=protocol,
                    protocol_path=protocol_path,
                    protocol_hash=phash,
                    max_epochs=epochs,
                    smoke=args.smoke,
                    force=args.force,
                )

    print("\nTraining request completed.")


if __name__ == "__main__":
    main()
