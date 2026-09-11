from __future__ import annotations

from tfm_project.paths import ProjectPaths

P = ProjectPaths.discover()

import argparse
import json
from pathlib import Path

import pandas as pd


from compute.logou_calibration import calibrate_logou_physical


def main():
    p = argparse.ArgumentParser(description="Calibrate historical H100 log-OU dynamics under P only.")
    p.add_argument("--input", default=str(P.data_raw / "global" / "ornn_h100.csv"))
    p.add_argument("--output", default=str(P.reproduced_results / "calibration" / "logou_physical_calibration.json"))
    p.add_argument("--day-count", type=float, default=365.0)
    args = p.parse_args()

    input_path = P.resolve(args.input)
    df = pd.read_csv(input_path)
    if "date" not in df or "price_avg" not in df:
        raise ValueError("Input must contain date and price_avg columns.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["price_avg"])

    # Calibration assumes equally spaced calendar-daily observations.
    gaps = df["date"].diff().dropna().dt.days
    if len(gaps) and not (gaps == 1).all():
        raise ValueError("Historical calibration assumes equally spaced daily observations; gaps detected.")

    cal = calibrate_logou_physical(df["price_avg"].to_numpy(float), day_count=args.day_count)
    payload = {
        **cal.as_dict(),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "measure": "P (physical/historical)",
        "guardrail": "These parameters are not automatically pricing-measure Q parameters.",
    }

    out = P.assert_not_frozen_write(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
