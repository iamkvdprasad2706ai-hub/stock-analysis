from __future__ import annotations

import pandas as pd

from stock_analysis.analysis import generate_recommendations


WATCHLIST_COLUMNS = [
    "Symbol",
    "Exchange",
    "Previous close",
    "Last close",
    "View",
    "Target 1",
    "Target 2",
    "Target 3",
    "Stop loss",
    "Market sentiment",
    "Signal date",
    "Data status",
]


def build_watchlist_table(
    stocks: list[dict[str, str]],
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a watchlist snapshot from real daily closes and existing recommendation signals."""
    rows = []
    for stock in stocks:
        symbol = stock["symbol"]
        exchange = stock["exchange"]
        history = price_data.get(f"{exchange}:{symbol}")
        row = {
            "Symbol": symbol,
            "Exchange": exchange,
            "Previous close": None,
            "Last close": None,
            "View": "HOLD",
            "Target 1": None,
            "Target 2": None,
            "Target 3": None,
            "Stop loss": None,
            "Market sentiment": "Unavailable",
            "Signal date": None,
            "Data status": "Price data unavailable",
        }

        if history is None or history.empty or "close" not in history.columns or history.attrs.get("synthetic"):
            rows.append(row)
            continue

        ordered = history.sort_values("date") if "date" in history.columns else history
        closes = pd.to_numeric(ordered["close"], errors="coerce").dropna()
        if closes.empty:
            rows.append(row)
            continue

        row["Last close"] = float(closes.iloc[-1])
        if len(closes) > 1:
            row["Previous close"] = float(closes.iloc[-2])
        if "date" in ordered.columns:
            dates = pd.to_datetime(ordered.loc[closes.index, "date"], errors="coerce").dropna()
            if not dates.empty:
                row["Signal date"] = dates.iloc[-1].strftime("%Y-%m-%d")
        row["Data status"] = "Daily price history" if len(closes) > 1 else "Limited price history"

        recommendation = generate_recommendations(ordered)
        action = str(recommendation.get("action", "HOLD")).upper()
        row["View"] = action
        row["Market sentiment"] = {
            "BUY": "Bullish",
            "HOLD": "Neutral",
            "SELL": "Bearish",
        }.get(action, "Unavailable")
        targets = recommendation.get("target_prices") or []
        if len(targets) >= 3:
            row["Target 1"], row["Target 2"], row["Target 3"] = targets[:3]
        row["Stop loss"] = recommendation.get("stop_loss")
        if recommendation.get("entry_price") is None:
            row["Data status"] = "Limited price history"
        rows.append(row)

    return pd.DataFrame(rows, columns=WATCHLIST_COLUMNS)
