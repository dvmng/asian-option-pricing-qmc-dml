from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def validate_price_series(df: pd.DataFrame, id_col: str = "index_id") -> None:
    required = {"date", id_col, "price_avg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Empty price dataset.")
    if (pd.to_numeric(df["price_avg"], errors="coerce") <= 0).any():
        raise ValueError("All price_avg observations must be strictly positive.")
    if df.duplicated([id_col, "date"]).any():
        raise ValueError(f"Duplicate ({id_col}, date) observations.")


def base100_each_series(
    df: pd.DataFrame,
    id_col: str = "index_id",
    base_value: float = 100.0,
) -> pd.DataFrame:
    """Create a base-100 series per raw benchmark; no cross-GPU aggregation."""
    validate_price_series(df, id_col=id_col)
    out = df.copy().sort_values([id_col, "date"])
    out["price_avg"] = pd.to_numeric(out["price_avg"], errors="raise")
    first = out.groupby(id_col)["price_avg"].transform("first")
    out["index_base100"] = base_value * out["price_avg"] / first
    return out


def build_candidate_composite(
    df: pd.DataFrame,
    series_ids: Sequence[str],
    method: str,
    value_col: str = "index_base100",
    min_coverage: float = 1.0,
) -> pd.DataFrame:
    """Build an explicitly labelled *candidate* composite.

    No method is chosen automatically because index methodology is a material
    research decision. Supported candidates:
      - equal_arithmetic: equal-weight arithmetic mean of constituent indices.
      - equal_geometric: equal-weight geometric mean of constituent indices.

    The function never forward-fills missing constituent observations.
    """
    if method not in {"equal_arithmetic", "equal_geometric"}:
        raise ValueError("method must be 'equal_arithmetic' or 'equal_geometric'")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1].")

    part = df[df["index_id"].isin(series_ids)].copy()
    if part.empty:
        raise ValueError("None of the requested series_ids are present.")
    missing_ids = set(series_ids) - set(part["index_id"].unique())
    if missing_ids:
        raise ValueError(f"Missing requested series: {sorted(missing_ids)}")

    wide = part.pivot(index="date", columns="index_id", values=value_col).sort_index()
    required_n = int(np.ceil(min_coverage * len(series_ids)))
    coverage_n = wide.notna().sum(axis=1)

    if method == "equal_arithmetic":
        composite = wide.mean(axis=1, skipna=True)
    else:
        if (wide <= 0).any().any():
            raise ValueError("Geometric composite requires positive values.")
        composite = np.exp(np.log(wide).mean(axis=1, skipna=True))

    out = pd.DataFrame({
        "date": wide.index,
        "candidate_index": composite,
        "n_constituents": coverage_n,
        "coverage": coverage_n / len(series_ids),
        "method": method,
        "constituents": " | ".join(series_ids),
    })
    out.loc[out["n_constituents"] < required_n, "candidate_index"] = np.nan
    return out.reset_index(drop=True)
