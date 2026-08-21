from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import qmc

from asian_simulation import AsianState
from config import TFMConfig


_EPS_U = 1e-12
_TIME_TOL = 1e-12
_REP_SEED_STEP = 1_000_003


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve the execution device.

    Parameters
    ----------
    device : {"auto", "cuda", "cpu"}
        - auto: CUDA when available, otherwise CPU.
        - cuda: require CUDA; raise if unavailable.
        - cpu: force CPU.
    """
    device = device.lower()

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
            )
        return torch.device("cuda")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be 'auto', 'cuda', or 'cpu'.")


def resolve_dtype(dtype: str = "float32") -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'.")


def device_summary(device: torch.device) -> Dict[str, object]:
    info: Dict[str, object] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_gb": props.total_memory / 1024**3,
            }
        )

    return info


def _check_power_of_two(n: int) -> int:
    if n <= 0:
        raise ValueError("n_paths must be positive.")
    m = np.log2(n)
    if abs(m - round(m)) > 1e-12:
        raise ValueError("For Sobol RQMC, n_paths must be a power of 2.")
    return int(round(m))


def _future_fixing_times(state: AsianState, cfg: TFMConfig) -> np.ndarray:
    fixings = np.asarray(cfg.fixing_times, dtype=np.float64)
    return fixings[fixings > state.t + _TIME_TOL]


def _future_dt(state: AsianState, cfg: TFMConfig) -> np.ndarray:
    future = _future_fixing_times(state, cfg)
    if len(future) == 0:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.r_[state.t, future]).astype(np.float64)


def _normal_draws_independent(
    states: Sequence[AsianState],
    n_paths: int,
    engine: str,
    replication: int,
    np_dtype: np.dtype,
) -> np.ndarray:
    """
    Generate one independent randomization per scenario.

    RQMC deliberately uses the SAME SciPy scrambled-Sobol construction as the
    validated CPU implementation. The resulting normal variates are then moved
    to the GPU for vectorized path construction and payoff/Greek evaluation.

    This preserves the methodology:
        one scenario -> one independent Sobol scramble.
    """
    if not states:
        return np.empty((0, n_paths, 0), dtype=np_dtype)

    d = states[0].n_fix_future
    if any(s.n_fix_future != d for s in states):
        raise ValueError("All states in a GPU pricing batch must have the same n_fix_future.")

    if d == 0:
        return np.empty((len(states), n_paths, 0), dtype=np_dtype)

    engine = engine.lower()
    z = np.empty((len(states), n_paths, d), dtype=np_dtype)

    if engine in {"rqmc", "qmc", "sobol"}:
        m = _check_power_of_two(n_paths)

        for i, state in enumerate(states):
            seed = int(state.pricing_seed + _REP_SEED_STEP * replication)
            sobol = qmc.Sobol(d=d, scramble=True, seed=seed)
            u = sobol.random_base2(m=m)
            u = np.clip(u, _EPS_U, 1.0 - _EPS_U)
            # ndtri computes in float64; cast only after the transform.
            z[i] = ndtri(u).astype(np_dtype, copy=False)

        return z

    if engine == "mc":
        for i, state in enumerate(states):
            seed = int(state.pricing_seed + _REP_SEED_STEP * replication)
            rng = np.random.default_rng(seed)
            z[i] = rng.standard_normal((n_paths, d)).astype(np_dtype, copy=False)
        return z

    raise ValueError("engine must be 'mc' or 'rqmc'.")


@torch.inference_mode()
def _price_same_dimension_batch(
    states: Sequence[AsianState],
    cfg: TFMConfig,
    n_paths: int,
    engine: str,
    replication: int,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Price a batch of states sharing the same number of future fixings.

    Tensor shapes:
        B = scenarios in batch
        P = paths per scenario
        D = future fixings

        z       : [B, P, D]
        dt      : [B, D]
        future  : [B, P, D]
        payoff  : [B, P]
    """
    if not states:
        return np.empty(0), np.empty(0)

    d = states[0].n_fix_future
    if any(s.n_fix_future != d for s in states):
        raise ValueError("Batch states do not share the same number of future fixings.")

    np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

    if d == 0:
        # Generic edge case. Under the current tau_min > 0 design there is
        # always at least one future fixing.
        zeros = np.zeros(len(states), dtype=np.float64)
        return zeros, zeros

    z_np = _normal_draws_independent(
        states=states,
        n_paths=n_paths,
        engine=engine,
        replication=replication,
        np_dtype=np_dtype,
    )

    # Host -> GPU transfer. pin_memory is only useful when CUDA is active.
    z_cpu = torch.from_numpy(z_np)
    if device.type == "cuda":
        z_cpu = z_cpu.pin_memory()
    z = z_cpu.to(device=device, dtype=torch_dtype, non_blocking=(device.type == "cuda"))

    spot = torch.tensor(
        [s.S_t_K for s in states], device=device, dtype=torch_dtype
    )
    c_t = torch.tensor(
        [s.C_t for s in states], device=device, dtype=torch_dtype
    )
    r = torch.tensor(
        [s.r for s in states], device=device, dtype=torch_dtype
    )
    sigma = torch.tensor(
        [s.sigma for s in states], device=device, dtype=torch_dtype
    )
    q_yield = torch.tensor(
        [s.q for s in states], device=device, dtype=torch_dtype
    )
    tau = torch.tensor(
        [s.tau for s in states], device=device, dtype=torch_dtype
    )

    dt_np = np.stack([_future_dt(s, cfg) for s in states], axis=0)
    dt = torch.as_tensor(dt_np, device=device, dtype=torch_dtype)

    drift = (
        (r - q_yield - 0.5 * sigma.square()).unsqueeze(1) * dt
    )  # [B, D]

    diffusion_scale = sigma.unsqueeze(1) * torch.sqrt(dt)  # [B, D]

    log_increments = (
        drift[:, None, :] + diffusion_scale[:, None, :] * z
    )  # [B, P, D]

    log_growth = torch.cumsum(log_increments, dim=2)
    future = spot[:, None, None] * torch.exp(log_growth)

    final_avg_k = c_t[:, None] + future.sum(dim=2) / cfg.n_fixings
    intrinsic = torch.clamp_min(final_avg_k - 1.0, 0.0)

    discount = torch.exp(-r * tau)
    price_samples = discount[:, None] * intrinsic

    # Pathwise derivative:
    # d S_future / d S_t = S_future / S_t
    dA_dS = (
        (future / spot[:, None, None]).sum(dim=2) / cfg.n_fixings
    )
    delta_samples = (
        discount[:, None]
        * (final_avg_k > 1.0).to(torch_dtype)
        * dA_dS
    )

    prices = price_samples.mean(dim=1)
    deltas = delta_samples.mean(dim=1)

    # Synchronize before returning results so timings outside this function are
    # meaningful.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return (
        prices.detach().cpu().double().numpy(),
        deltas.detach().cpu().double().numpy(),
    )


def price_states_replicated_gpu(
    states: Sequence[AsianState],
    cfg: TFMConfig,
    n_paths_per_replication: int,
    n_replications: int,
    engine: str,
    device: str = "auto",
    dtype: str = "float32",
    batch_size: int = 128,
    progress_callback: Callable[[int], None] | None = None,
) -> List[Dict[str, float]]:
    """
    Vectorized pricing for many Asian-option states using PyTorch/CUDA.

    Important methodological choice
    -------------------------------
    The GPU implementation does NOT replace independent scenario scramblings
    with a single shared high-dimensional Sobol net. Every scenario keeps its
    own scrambled Sobol sequence and its own pricing_seed, matching the
    validated CPU methodology. CUDA accelerates the GBM path construction,
    payoff calculation, and pathwise Delta calculation.

    Returns one dictionary per input state, in the original state order.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_replications <= 0:
        raise ValueError("n_replications must be positive.")

    dev = resolve_device(device)
    tdtype = resolve_dtype(dtype)
    n_states = len(states)

    if n_states == 0:
        return []

    # Store replication estimates. This is small:
    # train = 1 x 65,536; reference = 16 x 1,024.
    price_rep = np.empty((n_replications, n_states), dtype=np.float64)
    delta_rep = np.empty((n_replications, n_states), dtype=np.float64)

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, state in enumerate(states):
        groups[state.n_fix_future].append(idx)

    for rep in range(n_replications):
        for d in sorted(groups):
            indices = groups[d]

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_states = [states[i] for i in batch_idx]

                prices, deltas = _price_same_dimension_batch(
                    states=batch_states,
                    cfg=cfg,
                    n_paths=n_paths_per_replication,
                    engine=engine,
                    replication=rep,
                    device=dev,
                    torch_dtype=tdtype,
                )

                price_rep[rep, batch_idx] = prices
                delta_rep[rep, batch_idx] = deltas

                if progress_callback is not None:
                    progress_callback(len(batch_idx))

    mean_price = price_rep.mean(axis=0)
    mean_delta = delta_rep.mean(axis=0)

    if n_replications > 1:
        price_se = price_rep.std(axis=0, ddof=1) / np.sqrt(n_replications)
        delta_se = delta_rep.std(axis=0, ddof=1) / np.sqrt(n_replications)
    else:
        price_se = np.full(n_states, np.nan, dtype=np.float64)
        delta_se = np.full(n_states, np.nan, dtype=np.float64)

    results: List[Dict[str, float]] = []
    for i in range(n_states):
        results.append(
            {
                "price_K": float(mean_price[i]),
                "delta": float(mean_delta[i]),
                "price_se_rep": float(price_se[i]),
                "delta_se_rep": float(delta_se[i]),
                "n_paths": int(n_paths_per_replication * n_replications),
                "n_replications": int(n_replications),
                "pricing_engine": engine.lower(),
                "compute_backend": "pytorch_cuda" if dev.type == "cuda" else "pytorch_cpu",
                "compute_dtype": dtype.lower(),
            }
        )

    return results
