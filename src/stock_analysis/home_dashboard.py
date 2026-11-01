from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_STOCK_COLUMNS = [
    "Symbol",
    "Company",
    "Sector",
    "Latest price (₹)",
    "3-month return (%)",
    "1-year return (%)",
    "vs 50-day average (%)",
    "vs 200-day average (%)",
]

SECTOR_COLUMNS = ["Sector", "Market read", "Stocks tracked", "Average 1-year return (%)"]


def build_market_overview(
    screen_data: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    sectors: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize price performance for the supplied, explicitly scoped stock universe."""
    stock_rows = []
    for _, candidate in screen_data.iterrows():
        symbol = str(candidate.get("Symbol", "")).upper()
        history = price_data.get(symbol)
        if (
            not symbol
            or history is None
            or history.empty
            or "close" not in history.columns
            or history.attrs.get("synthetic")
        ):
            continue

        close = pd.to_numeric(history["close"], errors="coerce").dropna()
        if len(close) < 2:
            continue

        price = float(close.iloc[-1])
        average_50 = float(close.tail(50).mean())
        average_200 = float(close.tail(200).mean())
        year_close = close.tail(252)
        year_return = (
            (price / float(year_close.iloc[0]) - 1) * 100 if len(year_close) > 1 else np.nan
        )
        quarter_close = close.tail(63)
        quarter_return = (
            (price / float(quarter_close.iloc[0]) - 1) * 100
            if len(quarter_close) > 1
            else np.nan
        )
        stock_rows.append(
            {
                "Symbol": symbol,
                "Company": candidate.get("Company", symbol),
                "Sector": sectors.get(symbol, "Unclassified"),
                "Latest price (₹)": round(price, 2),
                "3-month return (%)": round(quarter_return, 2)
                if pd.notna(quarter_return)
                else np.nan,
                "1-year return (%)": round(year_return, 2) if pd.notna(year_return) else np.nan,
                "vs 50-day average (%)": round((price / average_50 - 1) * 100, 2)
                if average_50 > 0
                else np.nan,
                "vs 200-day average (%)": round((price / average_200 - 1) * 100, 2)
                if average_200 > 0
                else np.nan,
            }
        )

    stocks = pd.DataFrame(stock_rows, columns=MARKET_STOCK_COLUMNS)
    if stocks.empty:
        return stocks, pd.DataFrame(columns=SECTOR_COLUMNS)

    classified = stocks[
        (stocks["Sector"].notna())
        & (stocks["Sector"].astype(str).str.strip() != "")
        & (stocks["Sector"].astype(str).str.lower() != "unclassified")
    ]
    if classified.empty:
        return stocks, pd.DataFrame(columns=SECTOR_COLUMNS)

    sector_summary = (
        classified.groupby("Sector", as_index=False)
        .agg(
            **{
                "Stocks tracked": ("Symbol", "count"),
                "Average 1-year return (%)": ("1-year return (%)", "mean"),
            }
        )
        .dropna(subset=["Average 1-year return (%)"])
    )
    sector_summary["Average 1-year return (%)"] = sector_summary[
        "Average 1-year return (%)"
    ].round(2)
    sector_summary["Market read"] = np.select(
        [
            sector_summary["Average 1-year return (%)"] >= 5,
            sector_summary["Average 1-year return (%)"] <= -5,
        ],
        ["Doing well", "Down"],
        default="Stagnating",
    )
    sector_summary = sector_summary.sort_values(
        "Average 1-year return (%)", ascending=False
    ).reset_index(drop=True)
    return stocks, sector_summary[SECTOR_COLUMNS]
