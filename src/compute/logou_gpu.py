from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import qmc

from .compute_state_generation import ComputeDatasetState
from .phase4_config import ComputeDatasetConfig
from .seed_utils import replication_seed


_EPS_U = 1e-12
_TIME_TOL = 1e-12
def resolve_device(device: str = "auto") -> torch.device:
    device = device.lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto/cuda/cpu.")


def resolve_dtype(dtype: str = "float32") -> torch.dtype:
    if dtype.lower() == "float32":
        return torch.float32
    if dtype.lower() == "float64":
        return torch.float64
    raise ValueError("dtype must be float32/float64.")


def device_summary(device: torch.device) -> Dict[str, object]:
    out = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        out.update(
            gpu_name=p.name,
            total_vram_gb=p.total_memory / 1024**3,
            compute_capability=f"{p.major}.{p.minor}",
        )
    return out


def _check_power_two(n: int) -> int:
    m = np.log2(n)
    if n <= 0 or abs(m - round(m)) > 1e-12:
        raise ValueError("RQMC paths must be a positive power of 2.")
    return int(round(m))


def _future_fixing_times(s: ComputeDatasetState, cfg: ComputeDatasetConfig):
    ft = np.asarray(cfg.fixing_times, dtype=np.float64)
    return ft[ft > s.t + _TIME_TOL]


def _future_dt(s: ComputeDatasetState, cfg: ComputeDatasetConfig):
    future = _future_fixing_times(s, cfg)
    return np.diff(np.r_[s.t, future]).astype(np.float64)


def _elapsed(s: ComputeDatasetState, cfg: ComputeDatasetConfig):
    return (_future_fixing_times(s, cfg) - s.t).astype(np.float64)


def _normal_draws_independent(
    states: Sequence[ComputeDatasetState],
    n_paths: int,
    engine: str,
    replication: int,
    np_dtype,
) -> np.ndarray:
    if not states:
        return np.empty((0, n_paths, 0), dtype=np_dtype)
    d = states[0].n_fix_future
    if any(s.n_fix_future != d for s in states):
        raise ValueError("GPU batch must share n_fix_future.")
    if d == 0:
        return np.empty((len(states), n_paths, 0), dtype=np_dtype)

    z = np.empty((len(states), n_paths, d), dtype=np_dtype)
    engine = engine.lower()

    if engine in {"rqmc", "qmc", "sobol"}:
        m = _check_power_two(n_paths)
        for i, s in enumerate(states):
            seed = replication_seed(s.pricing_seed, replication)
            sobol = qmc.Sobol(d=d, scramble=True, seed=seed)
            u = np.clip(sobol.random_base2(m=m), _EPS_U, 1.0 - _EPS_U)
            z[i] = ndtri(u).astype(np_dtype, copy=False)
        return z

    if engine == "mc":
        for i, s in enumerate(states):
            seed = replication_seed(s.pricing_seed, replication)
            z[i] = np.random.default_rng(seed).standard_normal(
                (n_paths, d)
            ).astype(np_dtype, copy=False)
        return z

    raise ValueError("engine must be mc/rqmc.")


@torch.inference_mode()
def _price_same_dimension_batch(
    states: Sequence[ComputeDatasetState],
    cfg: ComputeDatasetConfig,
    n_paths: int,
    engine: str,
    replication: int,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    if not states:
        return np.empty(0), np.empty(0)

    d = states[0].n_fix_future
    if d == 0:
        return np.zeros(len(states)), np.zeros(len(states))

    np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64
    z_np = _normal_draws_independent(
        states, n_paths, engine, replication, np_dtype
    )
    z_cpu = torch.from_numpy(z_np)
    if device.type == "cuda":
        z_cpu = z_cpu.pin_memory()
    z = z_cpu.to(device=device, dtype=torch_dtype, non_blocking=device.type == "cuda")

    spot = torch.tensor([s.spot_K for s in states], device=device, dtype=torch_dtype)
    fixed = torch.tensor(
        [s.fixed_avg_contrib_K for s in states], device=device, dtype=torch_dtype
    )
    r = torch.tensor([s.r for s in states], device=device, dtype=torch_dtype)
    tau = torch.tensor([s.tau for s in states], device=device, dtype=torch_dtype)
    kappa = torch.tensor([s.kappa_q for s in states], device=device, dtype=torch_dtype)
    theta = torch.tensor(
        [s.theta_q_log_K for s in states], device=device, dtype=torch_dtype
    )
    sigma = torch.tensor([s.sigma_q for s in states], device=device, dtype=torch_dtype)

    dt = torch.as_tensor(
        np.stack([_future_dt(s, cfg) for s in states]),
        device=device,
        dtype=torch_dtype,
    )
    elapsed = torch.as_tensor(
        np.stack([_elapsed(s, cfg) for s in states]),
        device=device,
        dtype=torch_dtype,
    )

    phi = torch.exp(-kappa[:, None] * dt)
    variance = (
        sigma[:, None].square()
        * (-torch.expm1(-2.0 * kappa[:, None] * dt))
        / (2.0 * kappa[:, None])
    )
    step_std = torch.sqrt(torch.clamp_min(variance, 0.0))

    x = torch.log(spot)[:, None].expand(-1, n_paths).clone()
    future = torch.empty(
        (len(states), n_paths, d), device=device, dtype=torch_dtype
    )

    for j in range(d):
        x = (
            theta[:, None]
            + (x - theta[:, None]) * phi[:, j, None]
            + step_std[:, j, None] * z[:, :, j]
        )
        future[:, :, j] = torch.exp(x)

    final_avg_K = fixed[:, None] + future.sum(dim=2) / cfg.n_fixings
    discount = torch.exp(-r * tau)
    price_samples = discount[:, None] * torch.clamp_min(final_avg_K - 1.0, 0.0)

    attenuation = torch.exp(-kappa[:, None] * elapsed)
    jac = future / spot[:, None, None] * attenuation[:, None, :]
    dA_dspot = jac.sum(dim=2) / cfg.n_fixings
    delta_samples = (
        discount[:, None]
        * (final_avg_K > 1.0).to(torch_dtype)
        * dA_dspot
    )

    prices = price_samples.mean(dim=1)
    deltas = delta_samples.mean(dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return (
        prices.detach().cpu().double().numpy(),
        deltas.detach().cpu().double().numpy(),
    )


def price_states_replicated_gpu(
    states: Sequence[ComputeDatasetState],
    cfg: ComputeDatasetConfig,
    n_paths_per_replication: int,
    n_replications: int,
    engine: str,
    device: str = "auto",
    dtype: str = "float32",
    batch_size: int = 128,
    progress_callback: Callable[[int], None] | None = None,
) -> List[Dict[str, float]]:
    if batch_size <= 0 or n_replications <= 0:
        raise ValueError("batch_size and n_replications must be positive.")

    dev = resolve_device(device)
    tdtype = resolve_dtype(dtype)
    n_states = len(states)
    if n_states == 0:
        return []

    price_rep = np.empty((n_replications, n_states), dtype=np.float64)
    delta_rep = np.empty((n_replications, n_states), dtype=np.float64)

    groups = defaultdict(list)
    for i, s in enumerate(states):
        groups[s.n_fix_future].append(i)

    for rep in range(n_replications):
        for d in sorted(groups):
            idxs = groups[d]
            for start in range(0, len(idxs), batch_size):
                batch_idx = idxs[start:start + batch_size]
                batch = [states[i] for i in batch_idx]
                p, de = _price_same_dimension_batch(
                    batch, cfg, n_paths_per_replication, engine,
                    rep, dev, tdtype
                )
                price_rep[rep, batch_idx] = p
                delta_rep[rep, batch_idx] = de
                if progress_callback:
                    progress_callback(len(batch_idx))

    mean_p = price_rep.mean(axis=0)
    mean_d = delta_rep.mean(axis=0)
    if n_replications > 1:
        se_p = price_rep.std(axis=0, ddof=1) / np.sqrt(n_replications)
        se_d = delta_rep.std(axis=0, ddof=1) / np.sqrt(n_replications)
    else:
        se_p = np.full(n_states, np.nan)
        se_d = np.full(n_states, np.nan)

    backend = "pytorch_cuda" if dev.type == "cuda" else "pytorch_cpu"
    return [
        {
            "price_K": float(mean_p[i]),
            "delta": float(mean_d[i]),
            "price_se_rep": float(se_p[i]),
            "delta_se_rep": float(se_d[i]),
            "n_paths": int(n_paths_per_replication * n_replications),
            "n_replications": int(n_replications),
            "pricing_engine": engine.lower(),
            "compute_backend": backend,
            "compute_dtype": dtype.lower(),
        }
        for i in range(n_states)
    ]
