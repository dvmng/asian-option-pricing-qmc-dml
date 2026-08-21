from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


HKMA_HIBOR_URL = (
    "https://api.hkma.gov.hk/public/"
    "market-data-and-statistics/monthly-statistical-bulletin/"
    "er-ir/hk-interbank-ir-daily"
)


def download_hk_equity_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download a Hong Kong-listed equity/index proxy through yfinance.

    Examples:
        0700.HK  Tencent
        1810.HK  Xiaomi
        9988.HK  Alibaba
        0005.HK  HSBC Holdings
        1299.HK  AIA
        ^HSI     Hang Seng Index

    This is convenient/free, but it is NOT the official HKEX historical feed.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Install yfinance first: pip install yfinance"
        ) from exc

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=True,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    # yfinance can return MultiIndex columns in recent versions.
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df["Ticker"] = ticker
    return df


def realized_volatility(
    price_series: pd.Series,
    annualization: int = 252,
) -> float:
    """
    Annualized historical volatility from daily log returns.

    Prefer adjusted prices for this calculation to avoid artificial jumps from
    corporate actions. Historical volatility is a calibration/diagnostic input;
    it is not the same object as option-implied volatility.
    """
    x = pd.to_numeric(price_series, errors="coerce").dropna()
    log_returns = np.log(x / x.shift(1)).dropna()
    return float(log_returns.std(ddof=1) * np.sqrt(annualization))


def download_hkma_hibor(
    start: str,
    end: str,
    segment: str = "hibor.fixing",
    pagesize: int = 1000,
) -> pd.DataFrame:
    """
    Download official HKMA daily HIBOR/HONIA data.

    segment:
        'hibor.fixing'  -> HKD Interest Settlement Rates by tenor
        'honia'         -> HONIA
    """
    all_records = []
    offset = 0

    while True:
        params = {
            "segment": segment,
            "choose": "end_of_day",
            "from": start,
            "to": end,
            "pagesize": pagesize,
            "offset": offset,
        }

        r = requests.get(HKMA_HIBOR_URL, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        header = payload.get("header", {})
        if header and not header.get("success", True):
            raise RuntimeError(
                f"HKMA API error: {header.get('err_code')} "
                f"{header.get('err_msg')}"
            )

        result = payload.get("result", {})
        records = result.get("records", [])
        all_records.extend(records)

        if len(records) < pagesize:
            break
        offset += pagesize

    df = pd.DataFrame(all_records)
    if not df.empty and "end_of_day" in df.columns:
        df["end_of_day"] = pd.to_datetime(df["end_of_day"])
        df = df.sort_values("end_of_day").drop_duplicates("end_of_day")

    return df


def choose_rate_proxy(
    hibor: pd.DataFrame,
    tenor: str = "ir_3m",
) -> pd.Series:
    """
    Convert a quoted annual percentage rate from HKMA into decimal form.

    For this TFM, this is best treated as a REAL-DATA plausibility/calibration
    proxy, not as an exact continuously compounded zero-coupon curve.
    """
    if tenor not in hibor.columns:
        raise KeyError(f"{tenor} not found. Available: {list(hibor.columns)}")
    return pd.to_numeric(hibor[tenor], errors="coerce") / 100.0


def main():
    parser = argparse.ArgumentParser(
        description="Download optional real Hong Kong market data for calibration/context."
    )
    parser.add_argument("--ticker", default="0700.HK")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-08-21")
    parser.add_argument("--output", default="data_real_hk")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    stock = download_hk_equity_yfinance(args.ticker, args.start, args.end)
    stock_path = outdir / f"{args.ticker.replace('^','INDEX_').replace('.','_')}_daily.csv"
    stock.to_csv(stock_path, index=False)

    adj_col = "Adj Close" if "Adj Close" in stock.columns else "Close"
    sigma_hist = realized_volatility(stock[adj_col])

    hibor = download_hkma_hibor(args.start, args.end, segment="hibor.fixing")
    hibor_path = outdir / "hkma_hibor_daily.csv"
    hibor.to_csv(hibor_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "ticker": args.ticker,
                "start": args.start,
                "end": args.end,
                "annualized_realized_volatility": sigma_hist,
                "note": (
                    "Use as calibration/context only. Core Asian-option labels "
                    "remain simulated under the controlled GBM experiment."
                ),
            }
        ]
    )
    summary_path = outdir / "real_data_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved stock/index data: {stock_path}")
    print(f"Saved HKMA HIBOR:       {hibor_path}")
    print(f"Saved summary:          {summary_path}")
    print(f"Annualized realized volatility ({adj_col}): {sigma_hist:.4f}")


if __name__ == "__main__":
    main()
