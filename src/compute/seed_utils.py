from __future__ import annotations

# Integer ranges reserve disjoint namespaces for deterministic seed construction.
_MAX_NAMESPACE = 2**31
_MAX_ROW_INDEX = 2**32
_REPLICATION_STRIDE = 1 << 56
_MAX_REPLICATIONS = 64


def state_seed(namespace: int, row_index: int) -> int:
    """Injective signed-int64 encoding of (namespace, row_index)."""
    namespace = int(namespace)
    row_index = int(row_index)
    if not (0 <= namespace < _MAX_NAMESPACE):
        raise ValueError("namespace must satisfy 0 <= namespace < 2**31.")
    if not (0 <= row_index < _MAX_ROW_INDEX):
        raise ValueError("row_index must satisfy 0 <= row_index < 2**32.")
    seed = (namespace << 32) | row_index
    if seed >= 2**63:
        raise ValueError("Encoded seed exceeds signed int64 range.")
    return seed


def replication_seed(base_seed: int, replication: int) -> int:
    """Injective encoding of (base_seed, replication), preserving rep-0 seed.

    `base_seed` is produced by :func:`state_seed` and is below 2**48 for the
    frozen namespaces.  Replication buckets are separated by 2**56, so they
    cannot overlap.  Replication 0 deliberately equals `base_seed`, which makes
    single-rep CPU and Torch validation use identical Sobol scrambles.
    """
    base_seed = int(base_seed)
    replication = int(replication)
    if not (0 <= base_seed < _REPLICATION_STRIDE):
        raise ValueError("base_seed must be smaller than the replication stride.")
    if not (0 <= replication < _MAX_REPLICATIONS):
        raise ValueError(f"replication must be in [0, {_MAX_REPLICATIONS - 1}].")
    seed = base_seed + replication * _REPLICATION_STRIDE
    if seed >= 2**63:
        raise ValueError("Replication seed exceeds signed int64 range.")
    return seed
