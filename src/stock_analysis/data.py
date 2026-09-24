from __future__ import annotations

import io
import os
import re
import warnings
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
import nsepython
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


def get_optional_ai_summary(stock_name: str, summary: dict, profile: dict | None = None, fii_df: pd.DataFrame | None = None, bulk_df: pd.DataFrame | None = None) -> str:
    """Use OpenAI only when the user has configured a local API key. The key remains outside the repo."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "Heuristic recommendation available: "
            f"{stock_name} is showing {summary.get('percent_change', 0):.2f}% change with "
            f"momentum and institutional flow signals as the primary decision inputs."
        )

    try:
        from openai import OpenAI
    except Exception:
        return "OpenAI support is configured but the client library is not installed in the current environment."

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    sector = (profile or {}).get("sector") or "N/A"
    fii_net = float((fii_df["netValue"].sum()) if fii_df is not None and not fii_df.empty and "netValue" in fii_df.columns else 0.0)
    bulk_buy = int((bulk_df["Buy/Sell"].str.upper() == "BUY").sum()) if bulk_df is not None and not bulk_df.empty and "Buy/Sell" in bulk_df.columns else 0
    bulk_sell = int((bulk_df["Buy/Sell"].str.upper() == "SELL").sum()) if bulk_df is not None and not bulk_df.empty and "Buy/Sell" in bulk_df.columns else 0

    prompt = (
        f"Give a concise market view for {stock_name} in sector {sector}. "
        f"Current price change is {summary.get('percent_change', 0):.2f}%. "
        f"FII/DII net flow is {fii_net:.2f} crore. Bulk buy count is {bulk_buy} and bulk sell count is {bulk_sell}. "
        "Answer with 3 short bullets covering trend, risk, and action. Keep it factual and brief."
    )

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text.strip() if hasattr(response, "output_text") else str(response)


def normalize_stock_symbol(raw_symbol: str) -> str:
    """Normalize a stock ticker or company name into an NSE-style Yahoo Finance symbol."""
    if raw_symbol is None:
        raise ValueError("Stock symbol or name is required.")

    cleaned = str(raw_symbol).strip()
    if not cleaned:
        raise ValueError("Stock symbol or name is required.")

    if "." in cleaned:
        candidate = cleaned.upper()
        return candidate if candidate.endswith(".NS") else f"{candidate}.NS"

    candidate = re.sub(r"[^A-Z0-9\s\-]", "", cleaned.upper())
    base = re.split(r"[\s\-/]+", candidate.strip())[0]
    if not base:
        raise ValueError("Could not infer a stock symbol from the provided value.")
    return f"{base}.NS"


def _build_sample_data(symbol: str) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=120, freq="D")
    start_price = 100.0
    drift = 0.7 / 100
    noise = 0.03
    prices = [start_price]

    for index in range(1, len(dates)):
        last = prices[-1]
        random_step = noise * (0.5 - (index % 2))
        next_price = max(last * (1 + drift + random_step), 10.0)
        prices.append(float(next_price))

    sample = pd.DataFrame({
        "date": dates,
        "open": [p * 0.995 for p in prices],
        "high": [p * 1.015 for p in prices],
        "low": [p * 0.985 for p in prices],
        "close": prices,
        "volume": [1000000 + i * 500 for i in range(len(prices))],
    })
    sample["symbol"] = symbol.upper()
    return sample


def load_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Load NSE stock OHLC data and fall back to generated sample data when the market feed is unavailable."""
    ticker = normalize_stock_symbol(symbol)

    try:
        history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if history.empty:
            raise ValueError(f"No data found for symbol {ticker}.")

        data = history.reset_index()
        if "Date" in data.columns:
            data = data.rename(columns={"Date": "date"})
        data["symbol"] = ticker
        return data[["symbol", "date", "Open", "High", "Low", "Close", "Volume"]].rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
    except Exception:
        sample = _build_sample_data(ticker)
        return sample


def fetch_stock_profile(symbol: str) -> dict:
    """Retrieve the latest market profile and key stats for a stock symbol from Yahoo Finance."""
    ticker = normalize_stock_symbol(symbol)
    try:
        market_ticker = yf.Ticker(ticker)
        info = market_ticker.info or {}
        if not info:
            return {
                "symbol": ticker,
                "shortName": ticker.replace(".NS", ""),
                "sector": "N/A",
                "industry": "N/A",
                "marketCap": 0,
                "currency": "INR",
            }
        return {
            "symbol": ticker,
            "shortName": info.get("shortName") or ticker.replace(".NS", ""),
            "longName": info.get("longName") or info.get("shortName") or ticker.replace(".NS", ""),
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "marketCap": info.get("marketCap") or 0,
            "currency": info.get("currency") or "INR",
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "regularMarketPrice": info.get("regularMarketPrice"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "priceToBook": info.get("priceToBook"),
            "averageVolume": info.get("averageVolume"),
            "website": info.get("website"),
            "exchange": info.get("exchange") or "NSE",
        }
    except Exception:
        return {
            "symbol": ticker,
            "shortName": ticker.replace(".NS", ""),
            "sector": "N/A",
            "industry": "N/A",
            "marketCap": 0,
            "currency": "INR",
        }


def fetch_fii_data(limit: int = 10) -> pd.DataFrame:
    """Return recent FII/DII net investment flow from the NSE institutional flow endpoint."""
    try:
        fii_df = nsepython.nse_fiidii()
        if fii_df.empty:
            return pd.DataFrame(columns=["date", "category", "buyValue", "sellValue", "netValue"])

        fii_df = fii_df.copy()
        fii_df["date"] = pd.to_datetime(fii_df["date"], format="%d-%b-%Y", errors="coerce")
        fii_df["buyValue"] = pd.to_numeric(fii_df["buyValue"], errors="coerce")
        fii_df["sellValue"] = pd.to_numeric(fii_df["sellValue"], errors="coerce")
        fii_df["netValue"] = pd.to_numeric(fii_df["netValue"], errors="coerce")
        fii_df = fii_df.sort_values("date", ascending=False).head(limit).reset_index(drop=True)
        return fii_df
    except Exception:
        fallback = pd.DataFrame(
            {
                "date": [pd.Timestamp(datetime.today())],
                "category": ["FII/FPI"],
                "buyValue": [0.0],
                "sellValue": [0.0],
                "netValue": [0.0],
            }
        )
        return fallback


def fetch_bulk_deals(limit: int = 10) -> pd.DataFrame:
    """Return recent bulk deals from the NSE archive CSV."""
    url = "https://archives.nseindia.com/content/equities/bulk.csv"
    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        payload = pd.read_csv(io.StringIO(response.text), low_memory=False)
        payload = payload.head(limit).copy()
        if "Date" in payload.columns:
            payload["Date"] = pd.to_datetime(payload["Date"], format="%d-%b-%Y", errors="coerce")
        return payload.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(
            {
                "Date": [pd.Timestamp(datetime.today())],
                "Symbol": ["N/A"],
                "Security Name": ["N/A"],
                "Client Name": ["N/A"],
                "Buy/Sell": ["BUY"],
                "Quantity Traded": [0],
                "Trade Price / Wght. Avg. Price": [0.0],
                "Remarks": ["Fallback"],
            }
        )
