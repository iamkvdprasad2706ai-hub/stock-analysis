from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf


US_SECTOR_UNIVERSES = {
    "Technology": ("MSFT", "NVDA", "AVGO", "AAPL", "GOOGL"),
    "Banking": ("JPM", "BAC", "GS", "WFC", "C"),
    "Quantum Computing": ("IONQ", "RGTI", "QBTS", "QUBT", "ARQQ"),
    "Other sectors": ("COST", "V", "LLY", "UNH", "PG"),
}

US_STOCK_COLUMNS = [
    "Rank",
    "Sector",
    "Ticker",
    "Company",
    "Setup status",
    "Recommendation",
    "CMP ($)",
    "Entry zone ($)",
    "P/E",
    "ROE (%)",
    "Profit margin (%)",
    "Revenue growth (%)",
    "Debt/equity",
    "CMP vs 50-day avg (%)",
    "CMP vs 200-day avg (%)",
    "52-week range position (%)",
    "Confidence score (%)",
    "Verification",
]


def _number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _ticker_snapshot(ticker: str) -> tuple[dict, pd.DataFrame]:
    ticker_object = yf.Ticker(ticker)
    try:
        info = ticker_object.info or {}
    except Exception:
        info = {}
    try:
        history = ticker_object.history(period="2y", interval="1d", auto_adjust=True)
        if history.empty:
            history = pd.DataFrame()
        else:
            history = history.reset_index()
            if "Date" in history.columns:
                history = history.rename(columns={"Date": "date"})
            history = history.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            history.attrs["synthetic"] = False
            history.attrs["source"] = "yahoo_finance"
    except Exception:
        history = pd.DataFrame()
    return info, history


def fetch_us_buying_zone() -> pd.DataFrame:
    """Build a 20-stock US research table using Yahoo fundamentals and price history."""
    tickers = [ticker for sector_tickers in US_SECTOR_UNIVERSES.values() for ticker in sector_tickers]

    with ThreadPoolExecutor(max_workers=10) as executor:
        snapshots = dict(zip(tickers, executor.map(_ticker_snapshot, tickers)))

    rows = []
    for sector, sector_tickers in US_SECTOR_UNIVERSES.items():
        sector_rows = []
        for ticker in sector_tickers:
            info, history = snapshots[ticker]
            if history.empty or len(history) < 200 or "close" not in history.columns:
                sector_rows.append({
                    "Ticker": ticker,
                    "Company": info.get("shortName") or info.get("longName") or ticker,
                    "Sector": sector,
                    "Setup status": "Unverified — insufficient data",
                    "Recommendation": "HOLD",
                    "Confidence score (%)": 0,
                    "_sector": sector,
                })
                continue

            close = pd.to_numeric(history["close"], errors="coerce").dropna()
            if len(close) < 200:
                continue
            price = float(close.iloc[-1])
            average_50 = float(close.tail(50).mean())
            average_200 = float(close.tail(200).mean())
            distance_50 = (price / average_50 - 1) * 100
            distance_200 = (price / average_200 - 1) * 100

            year_close = close.tail(252)
            year_low = float(year_close.min())
            year_high = float(year_close.max())
            range_position = ((price - year_low) / (year_high - year_low)) if year_high > year_low else 1.0

            pe = _number(info.get("trailingPE"))
            roe = _number(info.get("returnOnEquity"))
            if roe is not None:
                roe *= 100
            profit_margin = _number(info.get("profitMargins"))
            if profit_margin is not None:
                profit_margin *= 100
            revenue_growth = _number(info.get("revenueGrowth"))
            if revenue_growth is not None:
                revenue_growth *= 100
            debt_equity = _number(info.get("debtToEquity"))
            if debt_equity is not None:
                debt_equity /= 100
            if debt_equity is None:
                debt_equity = _number(info.get("debtEquity"))

            score = 0.0
            if roe is not None and roe >= 15:
                score += 20
            elif roe is not None and roe >= 10:
                score += 10
            if profit_margin is not None and profit_margin >= 10:
                score += 20
            elif profit_margin is not None and profit_margin > 0:
                score += 10
            if revenue_growth is not None and revenue_growth >= 5:
                score += 20
            elif revenue_growth is not None and revenue_growth > 0:
                score += 10
            if pe is not None and 0 < pe <= 40:
                score += 15
            elif pe is not None and pe <= 60:
                score += 8
            if debt_equity is not None:
                if sector == "Banking" and debt_equity <= 15:
                    score += 10
                elif debt_equity <= 2:
                    score += 10
            score += max(0.0, 15 * (1 - abs(distance_50) / 15)) if abs(distance_50) <= 15 else 0
            score += max(0.0, 10 * (1 - range_position)) if 0 <= range_position <= 0.85 else 0

            entry_candidate = (
                abs(distance_50) <= 15
                and abs(distance_200) <= 35
                and 0 <= range_position <= 0.85
                and (roe is None or roe >= 10)
                and (profit_margin is None or profit_margin > 0)
                and score >= 45
            )
            status = "Buying-zone candidate" if entry_candidate else "Watchlist — wait for pullback/verification"

            sector_rows.append({
                "Ticker": ticker,
                "Company": info.get("shortName") or info.get("longName") or ticker,
                "Sector": sector,
                "Setup status": status,
                "Recommendation": "BUY" if entry_candidate else "HOLD",
                "CMP ($)": round(price, 2),
                "Entry zone ($)": f"{min(average_50, average_200):.2f}–{max(average_50, average_200):.2f}",
                "P/E": None if pe is None else round(pe, 2),
                "ROE (%)": None if roe is None else round(roe, 2),
                "Profit margin (%)": None if profit_margin is None else round(profit_margin, 2),
                "Revenue growth (%)": None if revenue_growth is None else round(revenue_growth, 2),
                "Debt/equity": None if debt_equity is None else round(debt_equity, 2),
                "CMP vs 50-day avg (%)": round(distance_50, 2),
                "CMP vs 200-day avg (%)": round(distance_200, 2),
                "52-week range position (%)": round(range_position * 100, 2),
                "Confidence score (%)": round(min(100, score)),
                "Verification": (
                    "Fundamentals and 2-year price history loaded"
                    if info and not close.empty
                    else "Incomplete verification"
                ),
                "_sector": sector,
            })

        sector_frame = pd.DataFrame(sector_rows)
        if sector_frame.empty:
            continue
        sector_frame["_buying"] = sector_frame["Setup status"].eq("Buying-zone candidate")
        sector_frame = sector_frame.sort_values(
            ["_buying", "Confidence score (%)"],
            ascending=[False, False],
        )
        sector_frame = sector_frame.head(5).drop(columns="_buying")
        sector_frame.insert(0, "Rank", range(1, len(sector_frame) + 1))
        rows.extend(sector_frame.to_dict("records"))

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=US_STOCK_COLUMNS)
    return result.drop(columns="_sector")[US_STOCK_COLUMNS]
