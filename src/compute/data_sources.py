"""Acquisition and normalization utilities for the compute-market dataset.

The public Ornn panel keeps each GPU series as a separate hardware-specific
benchmark. H100 SXM is the primary global underlying used by the TFM. SMM
utilities provide the complementary China-market data used in the empirical
context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import re
from typing import Optional, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Data sources

ORNN_BASE_URL = "https://api.ornnai.com"
SMM_COMPUTE_URL = "https://aidc.smm.cn/"


# Public Ornn GPU universe used for the market panel.
ORNN_FREE_GPUS: tuple[str, ...] = (
    "H100 SXM",
    "H200",
    "A100 SXM4",
    "RTX 5090",
    "B200",
)

# Primary global underlying used in the empirical analysis.
ORNN_PRIMARY_GPU = "H100 SXM"


# Leave memory as NaN when the public product label does not identify it unambiguously.
ORNN_GPU_METADATA: dict[str, dict[str, object]] = {
    "H100 SXM": {
        "memory": np.nan,
    },
    "H200": {
        "memory": np.nan,
    },
    "A100 SXM4": {
        "memory": np.nan,
    },
    "RTX 5090": {
        "memory": np.nan,
    },
    "B200": {
        "memory": np.nan,
    },
}


STANDARD_COLUMNS = [
    "date",
    "index_id",
    "market",
    "gpu_model",
    "memory",
    "region",
    "delivery_form",
    "contract_type",
    "price_low",
    "price_high",
    "price_avg",
    "currency",
    "unit",
    "observation_type",
    "source",
    "source_url",
    "retrieved_at_utc",
]


# Generic helpers

@dataclass(frozen=True)
class DataProvenance:
    source: str
    source_url: str
    observation_type: str


@dataclass(frozen=True)
class OrnnExtractionManifest:
    retrieved_at_utc: str
    requested_gpus: tuple[str, ...]
    successful_gpus: tuple[str, ...]
    failed_gpus: tuple[str, ...]
    total_observations: int
    primary_gpu: str
    source: str
    base_url: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[STANDARD_COLUMNS]

    if not out.empty:
        out["date"] = (
            pd.to_datetime(out["date"], errors="coerce")
            .dt.date
            .astype("string")
        )

        for c in ("price_low", "price_high", "price_avg"):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _build_retry_session(
    total_retries: int = 3,
    backoff_factor: float = 0.5,
) -> requests.Session:
    """
    Build a requests session with conservative retry behaviour.

    Retries only cover transient server/rate-limit failures. They do not alter
    or manufacture data.
    """
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "TFMResearch/1.0",
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        }
    )

    return session


def _safe_filename(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


# Ornn data

def fetch_ornn_history(
    gpu: str = ORNN_PRIMARY_GPU,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeout: int = 30,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download ONE public Ornn daily compute-price history.

    The anonymous/free surface currently exposes a trailing daily history for
    the public GPU set. Values are USD per GPU-hour.

    Use fetch_ornn_panel() to download all public GPU series.
    """

    url = (
        f"{ORNN_BASE_URL}/api/gpu/"
        f"{quote(gpu, safe='')}/index-history"
    )

    params: dict[str, str] = {}

    if start_date:
        params["startDate"] = start_date

    if end_date:
        params["endDate"] = end_date

    owns_session = session is None
    client = session or _build_retry_session()

    try:
        response = client.get(
            url,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

        payload = response.json()

    finally:
        if owns_session:
            client.close()

    if not payload.get("success", False):
        raise RuntimeError(
            f"Ornn API returned an unsuccessful payload for {gpu!r}: "
            f"{payload}"
        )

    rows = payload.get("data", [])

    if not rows:
        raise RuntimeError(
            f"No Ornn history returned for {gpu!r}."
        )

    raw = pd.DataFrame(rows)

    required = {"timestamp", "index_value"}
    missing = required.difference(raw.columns)

    if missing:
        raise RuntimeError(
            f"Unexpected Ornn schema for {gpu!r}. "
            f"Missing columns: {sorted(missing)}. "
            f"Available columns: {list(raw.columns)}"
        )

    raw["date"] = pd.to_datetime(
        raw["timestamp"],
        utc=True,
        errors="coerce",
    ).dt.date

    raw["price_avg"] = pd.to_numeric(
        raw["index_value"],
        errors="coerce",
    )

    retrieved_at = _utc_now()

    metadata = ORNN_GPU_METADATA.get(gpu, {})

    out = pd.DataFrame(
        {
            "date": raw["date"],
            "index_id": f"ORNN-{gpu}",
            "market": "global",
            "gpu_model": gpu,
            "memory": metadata.get("memory", np.nan),
            "region": "global",
            "delivery_form": "rental_index",

            # "daily" is the observation frequency; the benchmark contract is on-demand rental.
            "contract_type": "on_demand",

            "price_low": np.nan,
            "price_high": np.nan,
            "price_avg": raw["price_avg"],
            "currency": "USD",
            "unit": "USD/GPU-hour",
            "observation_type": "transaction_based_index",
            "source": "Ornn Compute Price Index (OCPI)",
            "source_url": url,
            "retrieved_at_utc": retrieved_at,
        }
    )

    out = _ensure_standard_columns(out)

    out = (
        out
        .dropna(subset=["date", "price_avg"])
        .drop_duplicates(
            subset=["date", "index_id"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if out.empty:
        raise RuntimeError(
            f"Ornn returned rows for {gpu!r}, but none remained "
            "after date/price validation."
        )

    if (out["price_avg"] <= 0).any():
        raise RuntimeError(
            f"Non-positive Ornn prices detected for {gpu!r}."
        )

    return out


def summarize_ornn_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact audit table for the Ornn long-format panel.
    """
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "gpu_model",
                "observations",
                "first_date",
                "last_date",
                "unique_dates",
                "min_price",
                "max_price",
                "mean_price",
            ]
        )

    summary = (
        panel
        .groupby("gpu_model", observed=True)
        .agg(
            observations=("price_avg", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            unique_dates=("date", "nunique"),
            min_price=("price_avg", "min"),
            max_price=("price_avg", "max"),
            mean_price=("price_avg", "mean"),
        )
        .reset_index()
        .sort_values("gpu_model")
        .reset_index(drop=True)
    )

    return summary


def fetch_ornn_panel(
    gpus: Sequence[str] = ORNN_FREE_GPUS,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeout: int = 30,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, OrnnExtractionManifest]:
    """
    Download all requested Ornn GPU indices.

    Returns
    -------
    panel
        Long-format standardized observations.
    summary
        One audit row per successfully downloaded GPU.
    manifest
        Extraction-level provenance and success/failure metadata.

    Notes
    -----
    strict=True:
        Any failed GPU aborts the entire extraction.

    strict=False:
        Successful GPU histories are retained and failures are reported.
        This should only be used for exploratory acquisition, not silently for
        a final frozen TFM dataset.
    """

    requested = tuple(str(g) for g in gpus)

    if not requested:
        raise ValueError("At least one Ornn GPU must be requested.")

    if len(set(requested)) != len(requested):
        raise ValueError(
            "Duplicate GPU names supplied to fetch_ornn_panel()."
        )

    frames: list[pd.DataFrame] = []
    successful: list[str] = []
    errors: dict[str, str] = {}

    with _build_retry_session() as session:

        for gpu in requested:

            print(f"[Ornn] Downloading {gpu} ...")

            try:
                df_gpu = fetch_ornn_history(
                    gpu=gpu,
                    start_date=start_date,
                    end_date=end_date,
                    timeout=timeout,
                    session=session,
                )

                frames.append(df_gpu)
                successful.append(gpu)

                print(
                    f"       {len(df_gpu):,} observations | "
                    f"{df_gpu['date'].min()} -> "
                    f"{df_gpu['date'].max()}"
                )

            except Exception as exc:
                errors[gpu] = repr(exc)

                if strict:
                    raise RuntimeError(
                        f"Failed downloading Ornn GPU {gpu!r}: {exc}"
                    ) from exc

                print(
                    f"       WARNING: {gpu} failed: {exc}"
                )

    if not frames:
        raise RuntimeError(
            "No Ornn GPU series could be downloaded."
        )

    panel = pd.concat(
        frames,
        ignore_index=True,
    )

    panel = (
        panel
        .sort_values(["gpu_model", "date"])
        .reset_index(drop=True)
    )

    duplicated = panel.duplicated(
        subset=["gpu_model", "date"],
        keep=False,
    )

    if duplicated.any():
        duplicated_rows = panel.loc[
            duplicated,
            ["gpu_model", "date", "price_avg"]
        ]

        raise RuntimeError(
            "Duplicate Ornn GPU/date observations detected:\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    summary = summarize_ornn_panel(panel)

    print("\n=== ORNN PANEL SUMMARY ===")
    print(summary.to_string(index=False))

    if errors:
        print("\nFailed GPU downloads:")
        for gpu, err in errors.items():
            print(f"  {gpu}: {err}")

    manifest = OrnnExtractionManifest(
        retrieved_at_utc=_utc_now(),
        requested_gpus=requested,
        successful_gpus=tuple(successful),
        failed_gpus=tuple(errors.keys()),
        total_observations=int(len(panel)),
        primary_gpu=ORNN_PRIMARY_GPU,
        source="Ornn Compute Price Index (OCPI)",
        base_url=ORNN_BASE_URL,
    )

    return panel, summary, manifest


def select_ornn_primary(
    panel: pd.DataFrame,
    primary_gpu: str = ORNN_PRIMARY_GPU,
) -> pd.DataFrame:
    """
    Select the canonical Ornn underlying without changing the stored panel.
    """

    out = (
        panel.loc[
            panel["gpu_model"].eq(primary_gpu)
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if out.empty:
        raise RuntimeError(
            f"Primary Ornn GPU {primary_gpu!r} is absent from the panel."
        )

    return out


def save_ornn_panel(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: OrnnExtractionManifest,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save the standardized Ornn panel plus per-GPU extracts and provenance.
    """

    output_dir = Path(output_dir)

    global_dir = output_dir / "global"
    per_gpu_dir = global_dir / "ornn_by_gpu"

    global_dir.mkdir(parents=True, exist_ok=True)
    per_gpu_dir.mkdir(parents=True, exist_ok=True)

    panel_path = global_dir / "ornn_gpu_panel_long.csv"
    summary_path = global_dir / "ornn_gpu_panel_summary.csv"
    manifest_path = global_dir / "ornn_extraction_manifest.json"
    primary_path = global_dir / "ornn_h100_primary.csv"

    panel.to_csv(panel_path, index=False)
    summary.to_csv(summary_path, index=False)

    select_ornn_primary(panel).to_csv(
        primary_path,
        index=False,
    )

    per_gpu_paths: dict[str, Path] = {}

    for gpu, df_gpu in panel.groupby(
        "gpu_model",
        sort=True,
        observed=True,
    ):
        safe = _safe_filename(gpu)

        path = per_gpu_dir / f"ornn_{safe}.csv"

        (
            df_gpu
            .sort_values("date")
            .reset_index(drop=True)
            .to_csv(path, index=False)
        )

        per_gpu_paths[str(gpu)] = path

    manifest_dict = asdict(manifest)
    manifest_dict["requested_gpus"] = list(manifest.requested_gpus)
    manifest_dict["successful_gpus"] = list(manifest.successful_gpus)
    manifest_dict["failed_gpus"] = list(manifest.failed_gpus)
    manifest_dict["files"] = {
        "panel": str(panel_path),
        "summary": str(summary_path),
        "primary_h100": str(primary_path),
        "per_gpu": {
            gpu: str(path)
            for gpu, path in per_gpu_paths.items()
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest_dict,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "panel": panel_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "primary_h100": primary_path,
        **{
            f"gpu::{gpu}": path
            for gpu, path in per_gpu_paths.items()
        },
    }


# SMM data

def _parse_smm_name(name: str) -> dict:
    """
    Parse the public SMM product label without translating away its meaning.
    """

    text = str(name).strip()
    tokens = [t for t in text.split("-") if t]

    if tokens and tokens[0].upper() == "SMM":
        tokens = tokens[1:]

    gpu_model = tokens[0] if tokens else np.nan

    memory = (
        tokens[1]
        if len(tokens) > 1
        and re.fullmatch(
            r"\d+G",
            tokens[1],
            flags=re.I,
        )
        else np.nan
    )

    pos = 2 if isinstance(memory, str) else 1

    delivery_form = (
        tokens[pos]
        if len(tokens) > pos
        else np.nan
    )

    region = (
        tokens[pos + 1]
        if len(tokens) > pos + 1
        else np.nan
    )

    contract_type = (
        "-".join(tokens[pos + 2:])
        if len(tokens) > pos + 2
        else np.nan
    )

    return {
        "gpu_model": gpu_model,
        "memory": memory,
        "delivery_form": delivery_form,
        "region": region,
        "contract_type": contract_type,
    }


def _parse_price_range(
    value: object,
) -> tuple[float, float]:
    text = str(value).replace(" ", "")

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*[~～-]\s*"
        r"(-?\d+(?:\.\d+)?)",
        text,
    )

    if not match:
        return np.nan, np.nan

    return float(match.group(1)), float(match.group(2))


def fetch_smm_current_snapshot(
    url: str = SMM_COMPUTE_URL,
    year: Optional[int] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Capture the public SMM compute tables visible without login.

    This is a snapshot collector, not a historical-data bypass.
    Full SMM history may require login/licensing.

    The function records only values genuinely present in the public HTML at
    retrieval time.
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; TFMResearch/1.0)"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text)
    )

    current_year = int(
        year or datetime.now().year
    )

    captured: list[dict] = []

    for table in tables:

        cols = {
            str(c).strip(): c
            for c in table.columns
        }

        if (
            "名称" not in cols
            or "均价" not in cols
            or "日期" not in cols
        ):
            continue

        for _, row in table.iterrows():

            name = str(
                row[cols["名称"]]
            ).strip()

            if not name.startswith("SMM-"):
                continue

            parsed = _parse_smm_name(name)

            if "价格范围" in cols:
                low, high = _parse_price_range(
                    row[cols["价格范围"]]
                )
            else:
                low, high = np.nan, np.nan

            avg = pd.to_numeric(
                pd.Series(
                    [row[cols["均价"]]]
                ),
                errors="coerce",
            ).iloc[0]

            unit = (
                str(
                    row[cols["单位"]]
                ).strip()
                if "单位" in cols
                else "元/卡·时"
            )

            date_text = str(
                row[cols["日期"]]
            ).strip()

            date = pd.to_datetime(
                f"{current_year}-{date_text}",
                errors="coerce",
            )

            captured.append(
                {
                    "date": (
                        date.date()
                        if not pd.isna(date)
                        else pd.NaT
                    ),
                    "index_id": name,
                    "market": "china",
                    **parsed,
                    "price_low": low,
                    "price_high": high,
                    "price_avg": avg,
                    "currency": "CNY",
                    "unit": unit,
                    "observation_type": (
                        "SMM_public_compute_price"
                    ),
                    "source": (
                        "Shanghai Metals Market (SMM)"
                    ),
                    "source_url": url,
                    "retrieved_at_utc": _utc_now(),
                }
            )

    if not captured:
        raise RuntimeError(
            "No public SMM compute-price rows could be parsed. "
            "The site layout may have changed or prices may "
            "require authentication in this session."
        )

    out = _ensure_standard_columns(
        pd.DataFrame(captured)
    )

    return (
        out
        .dropna(
            subset=["date", "price_avg"]
        )
        .drop_duplicates(
            subset=["date", "index_id"],
            keep="last",
        )
        .sort_values(
            ["gpu_model", "index_id"]
        )
        .reset_index(drop=True)
    )


def load_smm_official_history_csv(
    path: str | Path,
) -> pd.DataFrame:
    """
    Normalize an officially exported/manual SMM history file.

    Expected minimum fields
    -----------------------
    - date
    - product name/index id
    - average price

    Chinese and English aliases are accepted.
    Low/high prices are optional.

    This function is intentionally strict:
    it does not interpolate or manufacture missing dates.
    """

    path = Path(path)

    df = pd.read_csv(path)

    aliases = {
        "date": [
            "date",
            "日期",
        ],
        "name": [
            "name",
            "index_id",
            "名称",
        ],
        "avg": [
            "avg",
            "average",
            "price_avg",
            "均价",
        ],
        "low": [
            "low",
            "price_low",
            "最低价",
        ],
        "high": [
            "high",
            "price_high",
            "最高价",
        ],
    }

    def pick(
        key: str,
        required: bool,
    ) -> Optional[str]:

        for c in aliases[key]:
            if c in df.columns:
                return c

        if required:
            raise ValueError(
                f"SMM history file missing required "
                f"field {key!r}; tried {aliases[key]}"
            )

        return None

    date_c = pick("date", True)
    name_c = pick("name", True)
    avg_c = pick("avg", True)
    low_c = pick("low", False)
    high_c = pick("high", False)

    rows: list[dict] = []

    retrieved = _utc_now()

    for _, row in df.iterrows():

        name = str(
            row[name_c]
        ).strip()

        parsed = _parse_smm_name(name)

        rows.append(
            {
                "date": pd.to_datetime(
                    row[date_c],
                    errors="coerce",
                ),
                "index_id": name,
                "market": "china",
                **parsed,
                "price_low": (
                    pd.to_numeric(
                        pd.Series(
                            [row[low_c]]
                        ),
                        errors="coerce",
                    ).iloc[0]
                    if low_c
                    else np.nan
                ),
                "price_high": (
                    pd.to_numeric(
                        pd.Series(
                            [row[high_c]]
                        ),
                        errors="coerce",
                    ).iloc[0]
                    if high_c
                    else np.nan
                ),
                "price_avg": pd.to_numeric(
                    pd.Series(
                        [row[avg_c]]
                    ),
                    errors="coerce",
                ).iloc[0],
                "currency": "CNY",
                "unit": "CNY/GPU-hour",
                "observation_type": (
                    "official_history_import"
                ),
                "source": (
                    "Shanghai Metals Market (SMM)"
                ),
                "source_url": str(path),
                "retrieved_at_utc": retrieved,
            }
        )

    out = _ensure_standard_columns(
        pd.DataFrame(rows)
    )

    return (
        out
        .dropna(
            subset=["date", "price_avg"]
        )
        .sort_values(
            ["index_id", "date"]
        )
        .reset_index(drop=True)
    )


# Command-line interface

def main() -> None:
    """
    Default action:
        download the full public Ornn five-GPU panel and save it locally.

    This deliberately does NOT fetch SMM automatically because the SMM
    historical dataset used in the TFM is an official user export and should
    remain a separate provenance path.
    """

    output_dir = Path(
        "data/processed/compute_market"
    )

    print(
        "Downloading public Ornn compute-price panel:"
    )

    for gpu in ORNN_FREE_GPUS:
        marker = " [PRIMARY]" if gpu == ORNN_PRIMARY_GPU else ""
        print(f"  - {gpu}{marker}")

    panel, summary, manifest = fetch_ornn_panel(
        gpus=ORNN_FREE_GPUS,
        strict=True,
    )

    paths = save_ornn_panel(
        panel=panel,
        summary=summary,
        manifest=manifest,
        output_dir=output_dir,
    )

    print("\n=== SAVED FILES ===")
    for label, path in paths.items():
        print(f"{label:24s} {path}")

    print(
        "\nH100 remains the canonical pricing/DML "
        "underlying; the other GPU series form the "
        "global multi-GPU market panel."
    )


if __name__ == "__main__":
    main()
