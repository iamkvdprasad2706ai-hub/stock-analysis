from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_moving_average(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    result = data.copy()
    result[f"ma_{window}"] = result["close"].rolling(window=window, min_periods=1).mean()
    return result


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    if data.empty or "close" not in data.columns:
        return pd.Series(dtype=float)

    close = pd.to_numeric(data["close"], errors="coerce").dropna()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_daily_returns(data: pd.DataFrame) -> pd.Series:
    if "close" not in data.columns:
        raise ValueError("Data must include a 'close' column.")
    return data["close"].pct_change().dropna()


def summarize_stock(data: pd.DataFrame) -> dict:
    if data.empty:
        raise ValueError("No stock data available for summary.")

    closes = pd.to_numeric(data["close"], errors="coerce").dropna()
    returns = compute_daily_returns(data)

    latest_close = float(closes.iloc[-1])
    percent_change = (closes.iloc[-1] / closes.iloc[0] - 1.0) * 100 if len(closes) > 1 else 0.0
    avg_daily_return = float(returns.mean() * 100) if not returns.empty else 0.0
    volatility = float(returns.std() * 100) if not returns.empty else 0.0

    return {
        "latest_close": latest_close,
        "percent_change": percent_change,
        "average_daily_return": avg_daily_return,
        "volatility": volatility,
        "latest_volume": int(data["volume"].iloc[-1]) if "volume" in data.columns else 0,
    }


def generate_recommendations(data: pd.DataFrame, profile: dict | None = None, fii_df: pd.DataFrame | None = None, bulk_df: pd.DataFrame | None = None) -> dict:
    if data.empty:
        return {
            "action": "HOLD",
            "confidence": 0,
            "risk_level": "Low",
            "reasons": ["No valid stock data available."],
            "entry_price": None,
            "entry_partial_price": None,
            "target_prices": [None, None, None],
            "target_price": None,
            "stop_loss": None,
            "summary": "Not enough market data to provide a trading action.",
        }

    closes = pd.to_numeric(data["close"], errors="coerce").dropna()
    if closes.empty:
        return {
            "action": "HOLD",
            "confidence": 0,
            "risk_level": "Low",
            "reasons": ["Close price data is missing."],
            "entry_price": None,
            "entry_partial_price": None,
            "target_prices": [None, None, None],
            "target_price": None,
            "stop_loss": None,
            "summary": "No usable close-price series was found.",
        }

    price = float(closes.iloc[-1])
    start_price = float(closes.iloc[0])
    percent_change = (price / start_price - 1.0) * 100 if start_price else 0.0
    ma20 = float(data["close"].rolling(window=20, min_periods=1).mean().iloc[-1])
    ma50 = float(data["close"].rolling(window=50, min_periods=1).mean().iloc[-1])
    rsi = float(calculate_rsi(data).iloc[-1]) if not calculate_rsi(data).empty else 50.0
    latest_vol = float(data["volume"].iloc[-1]) if "volume" in data.columns else 0.0
    avg_vol = float(data["volume"].mean()) if "volume" in data.columns else 0.0
    volume_ratio = (latest_vol / avg_vol) if avg_vol else 1.0

    fii_net = 0.0
    if fii_df is not None and not fii_df.empty and "netValue" in fii_df.columns:
        fii_net = float(fii_df["netValue"].sum())

    bulk_buy = 0
    bulk_sell = 0
    if bulk_df is not None and not bulk_df.empty and "Buy/Sell" in bulk_df.columns:
        bulk_buy = int((bulk_df["Buy/Sell"].str.upper() == "BUY").sum())
        bulk_sell = int((bulk_df["Buy/Sell"].str.upper() == "SELL").sum())

    reasons: list[str] = []
    score = 50

    if price > ma20 > ma50:
        reasons.append("Price is trading above both short- and medium-term moving averages.")
        score += 20
    elif price < ma20 < ma50:
        reasons.append("Price is below the moving averages, which suggests weak trend momentum.")
        score -= 20
    else:
        reasons.append("Price is near the moving-average zone, so the trend is balanced.")

    if rsi >= 60:
        reasons.append("RSI is in a healthy bullish zone, suggesting improving momentum.")
        score += 15
    elif rsi <= 40:
        reasons.append("RSI is weak, so sentiment is timid and momentum may be fading.")
        score -= 15
    else:
        reasons.append("RSI is neutral, so momentum is stable but not strongly directional.")

    if volume_ratio >= 1.2:
        reasons.append("Trading volume is above the recent average, which supports the ongoing move.")
        score += 10
    elif volume_ratio < 0.8:
        reasons.append("Volume is weak relative to the recent average, which reduces conviction.")
        score -= 10

    if fii_net > 0:
        reasons.append("Institutional flows are positive, which is supportive of the stock.")
        score += 12
    elif fii_net < 0:
        reasons.append("Institutional flows are negative, which increases downside risk.")
        score -= 12

    if bulk_buy > bulk_sell:
        reasons.append("Bulk deals are skewed toward buying, suggesting interest from larger investors.")
        score += 8
    elif bulk_sell > bulk_buy:
        reasons.append("Bulk deals lean toward selling, which may signal distribution.")
        score -= 8

    if percent_change > 12:
        reasons.append("The stock has posted a strong recent gain, which can increase valuation risk.")
        score -= 5
    elif percent_change < -12:
        reasons.append("The stock has weakened materially, so caution is warranted.")
        score -= 10

    if profile and profile.get("sector"):
        reasons.append(f"Sector context: {profile.get('sector')}.")

    if score >= 65:
        action = "BUY"
    elif score <= 35:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = max(0, min(100, int(score)))
    risk_level = "Low" if score >= 60 else "Medium" if score >= 40 else "High"

    entry_price = round(price, 2)
    entry_partial_price = round(price * 0.95, 2)
    target_1 = round(price * 1.04, 2)
    target_2 = round(price * 1.08, 2)
    target_3 = round(price * 1.12, 2)
    target_price = target_3
    stop_loss = round(price * 0.92, 2)

    summary = (
        f"{action} view: {price:.2f} with {percent_change:.2f}% price change, "
        f"RSI {rsi:.1f}, and institutional flow support is {'positive' if fii_net > 0 else 'mixed' if fii_net == 0 else 'negative'}."
    )

    return {
        "action": action,
        "confidence": confidence,
        "risk_level": risk_level,
        "reasons": reasons[:6],
        "entry_price": entry_price,
        "entry_partial_price": entry_partial_price,
        "target_prices": [target_1, target_2, target_3],
        "target_price": target_price,
        "stop_loss": stop_loss,
        "summary": summary,
    }
