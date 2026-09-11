from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from compute.data_sources import fetch_ornn_panel, save_ornn_panel
from tfm_project.paths import ProjectPaths


P = ProjectPaths.discover()


def describe_csv(path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(path)
    out = {"path": str(path), "exists": True, "rows": int(len(df)), "columns": list(df.columns)}
    if "date" in df:
        d = pd.to_datetime(df["date"], errors="coerce")
        out["first_date"] = str(d.min().date()) if d.notna().any() else None
        out["last_date"] = str(d.max().date()) if d.notna().any() else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-data entry point for the thesis repository.")
    parser.add_argument("--refresh-ornn", action="store_true", help="Download a fresh public Ornn panel to data/staging; never overwrite frozen data.")
    args = parser.parse_args()

    files = [
        P.data_raw / "global" / "ornn_h100.csv",
        P.data_raw / "china" / "smm_history_full.csv",
        P.data_processed / "compute_market" / "global" / "ornn_gpu_panel_long.csv",
    ]
    report = {"frozen_inputs": [describe_csv(p) for p in files]}
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.refresh_ornn:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging = P.data / "staging" / f"ornn_{stamp}"
        panel, summary, manifest = fetch_ornn_panel(strict=True)
        saved = save_ornn_panel(panel, summary, manifest, staging)
        print("\nFresh Ornn data staged (not frozen):")
        for name, path in saved.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
