from __future__ import annotations

from tfm_project.paths import ProjectPaths

P = ProjectPaths.discover()

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


from compute_ml.config import (
    DELTA_COLUMN,
    EXPECTED_PARAMETER_COUNT,
    INPUT_COLUMNS,
    PRICE_COLUMN,
    SPOT_INPUT_INDEX,
)
from compute_ml.models import build_model, parameter_count, spot_derivative


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


def load_protocol(path: Path) -> tuple[dict, str]:
    p = load_json(path)
    if p.get("status") != "FROZEN_BEFORE_COMPUTE_FINAL_TRAINING":
        raise RuntimeError("Unexpected compute protocol status.")
    return p, sha256_file(path)


def verify_eval_data(protocol: dict, data_dir: Path, split: str) -> None:
    path = data_dir / f"{split}.csv.gz"
    canonical_dir = P.pricing_dataset.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if data_dir.resolve() == canonical_dir:
        expected = protocol["source_hashes"][split]["sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"{split}: SHA256 mismatch vs frozen protocol.")
    else:
        verify_reproduced_split(
            path,
            canonical_dir / f"{split}.csv.gz",
            split,
        )


def transform_inputs(df, scaler):
    x = df[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32)
    mean = np.asarray(scaler["x_mean"], dtype=np.float32)
    std = np.asarray(scaler["x_std"], dtype=np.float32)
    return ((x - mean) / std).astype(np.float32, copy=False)


def predict_price_delta(model, x_scaled, scaler, device, batch_size):
    model.eval()
    y_mean = float(scaler["y_mean"])
    y_std = float(scaler["y_std"])
    x_std_spot = float(scaler["x_std"][SPOT_INPUT_INDEX])
    prices, deltas = [], []

    for start in range(0, len(x_scaled), batch_size):
        xb = torch.from_numpy(
            x_scaled[start:start+batch_size]
        ).to(device).requires_grad_(True)
        pred_std, grad_std = spot_derivative(
            model, xb, SPOT_INPUT_INDEX, create_graph=False
        )
        prices.append(
            pred_std.detach().cpu().numpy().astype(np.float64) * y_std + y_mean
        )
        deltas.append(
            grad_std.detach().cpu().numpy().astype(np.float64)
            * (y_std / x_std_spot)
        )
    return np.concatenate(prices), np.concatenate(deltas)


def metrics(y, pred, prefix):
    err = pred - y
    mse = float(np.mean(err**2))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": float(np.sqrt(mse)),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_r2": (
            float("nan") if ss_tot <= 0 else 1.0 - ss_res/ss_tot
        ),
        f"{prefix}_max_abs_error": float(np.max(np.abs(err))),
        f"{prefix}_p95_abs_error": float(np.quantile(np.abs(err), 0.95)),
        f"{prefix}_p99_abs_error": float(np.quantile(np.abs(err), 0.99)),
    }


def discover_runs(results_dir: Path) -> List[Path]:
    return [
        p for p in sorted(results_dir.iterdir())
        if p.is_dir()
        and not p.name.startswith("SMOKE_")
        and (p/"COMPLETED").exists()
        and (p/"metadata.json").exists()
        and (p/"model.pt").exists()
    ]


def evaluate_run(
    run_dir, split, df, protocol, phash, device, batch_size,
    save_predictions, pred_dir
):
    meta = load_json(run_dir/"metadata.json")
    if meta.get("frozen_protocol_sha256") != phash:
        raise RuntimeError(f"{run_dir.name}: protocol hash mismatch.")
    if int(meta.get("parameter_count",-1)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"{run_dir.name}: parameter count mismatch.")
    if meta.get("no_early_stopping") is not True:
        raise RuntimeError(f"{run_dir.name}: early stopping mismatch.")
    if int(meta.get("max_epochs_used",-1)) != int(protocol["final_max_epochs"]):
        raise RuntimeError(f"{run_dir.name}: epoch budget mismatch.")

    params = dict(meta["model_params"])
    model = build_model(len(INPUT_COLUMNS), params).to(device)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"{run_dir.name}: rebuilt architecture mismatch.")
    state = torch.load(
        run_dir/"model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)

    x = transform_inputs(df, meta["scaler"])
    pp, pdlt = predict_price_delta(
        model, x, meta["scaler"], device, batch_size
    )
    yp = df[PRICE_COLUMN].to_numpy(dtype=np.float64)
    yd = df[DELTA_COLUMN].to_numpy(dtype=np.float64)

    out = {
        "run_name": meta["run_name"],
        "model_type": meta["model_type"],
        "train_size": int(meta["train_size"]),
        "seed": int(meta["seed"]),
        "selected_epoch": int(meta["selected_epoch"]),
        "training_seconds": float(meta["training_seconds"]),
        "parameter_count": int(meta["parameter_count"]),
        "lambda_delta": float(meta["lambda_delta"]),
        "split": split,
        "n_eval": int(len(df)),
        "protocol_sha256": phash,
    }
    out.update(metrics(yp, pp, "price"))
    out.update(metrics(yd, pdlt, "delta"))

    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
        ids = (
            df["scenario_id"].astype(str).to_numpy()
            if "scenario_id" in df.columns
            else np.arange(len(df)).astype(str)
        )
        pd.DataFrame({
            "scenario_id": ids,
            "price_true": yp,
            "price_pred": pp,
            "price_error": pp-yp,
            "price_abs_error": np.abs(pp-yp),
            "delta_true": yd,
            "delta_pred": pdlt,
            "delta_error": pdlt-yd,
            "delta_abs_error": np.abs(pdlt-yd),
        }).to_csv(
            pred_dir/f"{meta['run_name']}__{split}.csv.gz",
            index=False, compression={"method": "gzip", "mtime": 0}
        )
    return out


def aggregate(per_run):
    metric_cols = [
        "price_mse","price_rmse","price_mae","price_r2",
        "price_max_abs_error","price_p95_abs_error","price_p99_abs_error",
        "delta_mse","delta_rmse","delta_mae","delta_r2",
        "delta_max_abs_error","delta_p95_abs_error","delta_p99_abs_error",
        "selected_epoch","training_seconds",
    ]
    rows = []
    for (split,n,model), g in per_run.groupby(
        ["split","train_size","model_type"], sort=True
    ):
        row = {
            "split": split, "train_size": int(n),
            "model_type": model, "n_seeds": int(len(g))
        }
        for col in metric_cols:
            vals = g[col].astype(float)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_sd"] = (
                float(vals.std(ddof=1)) if len(vals)>1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["split","train_size","model_type"]
    )


def paired(per_run):
    metrics_list = [
        "price_mse","price_rmse","price_mae",
        "delta_mse","delta_rmse","delta_mae"
    ]
    rows=[]
    for split in sorted(per_run.split.unique()):
        sdf=per_run[per_run.split==split]
        for n in sorted(sdf.train_size.unique()):
            g=sdf[sdf.train_size==n]
            m=g[g.model_type=="mlp"].set_index("seed")
            d=g[g.model_type=="dml"].set_index("seed")
            common=sorted(set(m.index)&set(d.index))
            if not common:
                continue
            for metric in metrics_list:
                a=m.loc[common,metric].astype(float).to_numpy()
                b=d.loc[common,metric].astype(float).to_numpy()
                diff=b-a
                imp=100*(a-b)/np.maximum(np.abs(a),1e-15)
                rows.append({
                    "split":split,"train_size":int(n),"metric":metric,
                    "n_pairs":len(common),
                    "mlp_mean":float(a.mean()),
                    "dml_mean":float(b.mean()),
                    "dml_minus_mlp_mean":float(diff.mean()),
                    "dml_minus_mlp_sd":(
                        float(diff.std(ddof=1)) if len(diff)>1 else float("nan")
                    ),
                    "dml_improvement_pct_mean":float(imp.mean()),
                    "dml_improvement_pct_sd":(
                        float(imp.std(ddof=1)) if len(imp)>1 else float("nan")
                    ),
                    "dml_better_pairs":int(np.sum(b<a)),
                    "mlp_better_pairs":int(np.sum(a<b)),
                })
    return pd.DataFrame(rows).sort_values(
        ["split","train_size","metric"]
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default=str(P.pricing_dataset))
    ap.add_argument("--results-dir",default=str(P.models_final))
    ap.add_argument("--output-dir",default=str(P.reproduced_results / "ml_evaluation"))
    ap.add_argument("--protocol",default=str(DEFAULT_PROTOCOL))
    ap.add_argument(
        "--splits",nargs="+",choices=["test","reference"],
        default=["test","reference"]
    )
    ap.add_argument("--device",choices=["auto","cuda","cpu"],default="cuda")
    ap.add_argument("--batch-size",type=int,default=4096)
    ap.add_argument("--save-predictions",action="store_true")
    args=ap.parse_args()

    protocol_path=P.resolve(args.protocol)
    protocol,phash=load_protocol(protocol_path)
    results_dir=P.resolve(args.results_dir)
    runs=discover_runs(results_dir)
    expected=len(protocol["final_train_sizes"])*len(
        protocol["final_training_seeds"]
    )*2
    if len(runs)!=expected:
        raise RuntimeError(
            f"Expected exactly {expected} completed final runs, found {len(runs)}."
        )

    # Require the complete expected model grid before evaluation.
    expected_names = {
        f"{m}_n{n}_seed{s}"
        for n in protocol["final_train_sizes"]
        for s in protocol["final_training_seeds"]
        for m in ("mlp","dml")
    }
    actual_names={p.name for p in runs}
    if actual_names!=expected_names:
        raise RuntimeError(
            f"Final run grid mismatch. Missing={sorted(expected_names-actual_names)} "
            f"extra={sorted(actual_names-expected_names)}"
        )

    device=resolve_device(args.device)
    outdir=P.assert_not_frozen_write(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    pred_dir=outdir/"predictions"
    rows=[]

    print("="*72)
    print("FINAL COMPUTE TEST / REFERENCE EVALUATION")
    print("="*72)
    print(f"Protocol SHA256: {phash}")
    print(f"Runs: {len(runs)} | Device: {device} | Splits: {args.splits}")
    print("="*72)

    data_dir=P.resolve(args.data_dir)
    for split in args.splits:
        verify_eval_data(protocol,data_dir,split)
        df=pd.read_csv(data_dir/f"{split}.csv.gz")
        required=set(INPUT_COLUMNS)|{PRICE_COLUMN,DELTA_COLUMN}
        missing=required-set(df.columns)
        if missing:
            raise RuntimeError(f"{split}: missing {sorted(missing)}")
        arr=df[list(INPUT_COLUMNS)+[PRICE_COLUMN,DELTA_COLUMN]].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{split}: non-finite data.")

        print(f"\n[{split}] N={len(df):,}")
        for i,run in enumerate(runs,1):
            row=evaluate_run(
                run,split,df,protocol,phash,device,args.batch_size,
                args.save_predictions,pred_dir
            )
            rows.append(row)
            print(
                f"{i:02d}/{len(runs)} {row['run_name']} "
                f"price_MAE={row['price_mae']:.3e} "
                f"delta_MAE={row['delta_mae']:.3e}"
            )

    per_run=pd.DataFrame(rows).sort_values(
        ["split","train_size","seed","model_type"]
    )
    agg=aggregate(per_run)
    pair=paired(per_run)

    per_run.to_csv(outdir/"per_run_metrics.csv",index=False)
    agg.to_csv(outdir/"aggregate_metrics.csv",index=False)
    pair.to_csv(outdir/"paired_mlp_dml.csv",index=False)

    sample=agg[[
        "split","train_size","model_type",
        "price_mae_mean","price_mae_sd","price_rmse_mean","price_rmse_sd",
        "delta_mae_mean","delta_mae_sd","delta_rmse_mean","delta_rmse_sd",
        "training_seconds_mean","training_seconds_sd",
        "selected_epoch_mean","selected_epoch_sd",
    ]].copy()
    sample.to_csv(outdir/"sample_efficiency.csv",index=False)

    audit={
        "protocol_path":str(protocol_path),
        "protocol_sha256":phash,
        "results_dir":str(results_dir),
        "data_dir":str(data_dir),
        "splits":args.splits,
        "n_completed_runs":len(runs),
        "expected_runs":expected,
        "expected_parameter_count":EXPECTED_PARAMETER_COUNT,
        "evaluation_device":str(device),
        "torch_version":torch.__version__,
        "save_predictions":bool(args.save_predictions),
    }
    (outdir/"evaluation_audit.json").write_text(
        json.dumps(audit,indent=2),encoding="utf-8"
    )

    print("\nPrimary aggregate metrics:")
    print(agg[[
        "split","train_size","model_type",
        "price_mae_mean","price_mae_sd",
        "delta_mae_mean","delta_mae_sd"
    ]].to_string(index=False))
    print("\nPaired DML improvement (%), positive = DML better:")
    print(pair[pair.metric.isin(["price_mae","delta_mae"])][[
        "split","train_size","metric","dml_improvement_pct_mean",
        "dml_improvement_pct_sd","dml_better_pairs","mlp_better_pairs"
    ]].to_string(index=False))
    print(f"\nSaved -> {outdir}")


if __name__=="__main__":
    main()
