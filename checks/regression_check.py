from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare_csv(reference: Path, candidate: Path, rtol: float, atol: float) -> None:
    a = pd.read_csv(reference)
    b = pd.read_csv(candidate)
    if list(a.columns) != list(b.columns):
        raise SystemExit("Column mismatch.")
    if len(a) != len(b):
        raise SystemExit(f"Row-count mismatch: {len(a)} != {len(b)}")

    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x = a[col].to_numpy(dtype=float)
            y = b[col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any() and not np.allclose(x[mask], y[mask], rtol=rtol, atol=atol):
                diff = np.max(np.abs(x[mask] - y[mask]))
                raise SystemExit(f"Numeric mismatch in {col}: max abs diff={diff:.3e}")
        else:
            if not a[col].astype(str).equals(b[col].astype(str)):
                raise SystemExit(f"Non-numeric mismatch in {col}.")

    print("REGRESSION CSV CHECK PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare a reproduced CSV with frozen thesis evidence.")
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    args = p.parse_args()
    compare_csv(args.reference, args.candidate, args.rtol, args.atol)


if __name__ == "__main__":
    main()
