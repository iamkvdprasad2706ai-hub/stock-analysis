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


def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD, signal, and histogram series for a close-price dataframe."""
    result = data.copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    result["macd"] = fast_ema - slow_ema
    result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False, min_periods=1).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    return result


def _historical_metrics(data: pd.DataFrame) -> dict[str, float]:
    close = pd.to_numeric(data.get("close"), errors="coerce")
    valid = close.dropna()
    if valid.empty:
        return {"history_years": 0.0, "cagr": np.nan, "one_year_return": np.nan, "max_drawdown": np.nan, "annualized_volatility": np.nan}

    if "date" in data.columns:
        dates = pd.to_datetime(data.loc[valid.index, "date"], errors="coerce")
    elif isinstance(data.index, pd.DatetimeIndex):
        dates = pd.Series(data.index, index=data.index).reindex(valid.index)
    else:
        dates = pd.Series(dtype="datetime64[ns]")

    dated = pd.DataFrame({"date": dates, "close": valid}).dropna().sort_values("date")
    if not dated.empty:
        if dated["date"].dt.tz is not None:
            dated["date"] = dated["date"].dt.tz_localize(None)
        history_years = max((dated["date"].iloc[-1] - dated["date"].iloc[0]).days / 365.25, 0.0)
        latest_date = dated["date"].iloc[-1]
        one_year_start = dated[dated["date"] <= latest_date - pd.DateOffset(years=1)]
        one_year_return = (
            (float(dated["close"].iloc[-1]) / float(one_year_start["close"].iloc[-1]) - 1) * 100
            if not one_year_start.empty and one_year_start["close"].iloc[-1] > 0
            else np.nan
        )
    else:
        history_years = max((len(valid) - 1) / 252, 0.0)
        one_year_return = np.nan

    first_close = float(valid.iloc[0])
    last_close = float(valid.iloc[-1])
    cagr = ((last_close / first_close) ** (1 / history_years) - 1) * 100 if history_years >= 1 and first_close > 0 else np.nan
    drawdown = valid / valid.cummax() - 1
    daily_returns = valid.pct_change().dropna()
    annualized_volatility = float(daily_returns.std() * np.sqrt(252) * 100) if not daily_returns.empty else np.nan
    return {
        "history_years": history_years,
        "cagr": cagr,
        "one_year_return": one_year_return,
        "max_drawdown": float(drawdown.min() * 100),
        "annualized_volatility": annualized_volatility,
    }


def analyze_institutional_trend(fii_data: pd.DataFrame, recent_window: int = 5) -> pd.DataFrame:
    """Summarize recent FII/DII direction and project the next net-flow reading."""
    columns = ["category", "recent_average", "previous_average", "change", "direction", "projected_next"]
    if fii_data.empty or not {"category", "netValue"}.issubset(fii_data.columns):
        return pd.DataFrame(columns=columns)

    clean = fii_data[["category", "netValue"]].copy()
    clean["netValue"] = pd.to_numeric(clean["netValue"], errors="coerce")
    clean = clean.dropna(subset=["netValue"])
    rows = []
    for category, group in clean.groupby("category"):
        values = group["netValue"].reset_index(drop=True)
        if values.empty:
            continue
        window = min(recent_window, len(values))
        recent = float(values.iloc[-window:].mean())
        previous_values = values.iloc[-2 * window:-window] if len(values) > window else values.iloc[:-window]
        previous = float(previous_values.mean()) if not previous_values.empty else recent
        change = recent - previous
        direction = "Increasing buying" if change > 0 else "Increasing selling" if change < 0 else "Stable"
        projected = recent + (change * 0.5)
        rows.append(
            {
                "category": category,
                "recent_average": round(recent, 2),
                "previous_average": round(previous, 2),
                "change": round(change, 2),
                "direction": direction,
                "projected_next": round(projected, 2),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_market_ideas(
    screen_data: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    macro_risk: str = "Neutral",
    market_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank sectors by trend and return the three strongest long-history candidates per sector."""
    columns = [
        "Sector rank", "Sector", "Sector trend", "Rank", "Symbol", "Company", "Score", "View",
        "Entry", "Stop loss", "Target 1", "Target 2", "Target 3", "One-year return (%)",
        "CAGR (available history, %)", "Annualized volatility (%)", "History years", "Max drawdown (%)", "Horizon", "Drivers",
    ]
    if screen_data.empty:
        return pd.DataFrame(columns=columns)

    ideas = []
    macro_penalty = {"Neutral": 0, "Elevated": 8, "High": 15}.get(macro_risk, 0)
    market_adjustment = 0
    if market_context is not None and not market_context.empty:
        broad_indices = market_context[market_context["Market"].isin(["NIFTY 50", "S&P 500", "NASDAQ Composite"])]
        for _, index_row in broad_indices.iterrows():
            if index_row.get("Trend vs 200-day average") == "Bullish":
                market_adjustment += 1
            elif index_row.get("Trend vs 200-day average") == "Bearish":
                market_adjustment -= 3
            index_return = pd.to_numeric(index_row.get("One-year return (%)"), errors="coerce")
            if pd.notna(index_return) and index_return < -10:
                market_adjustment -= 2
        cross_asset = market_context.set_index("Market")
        if "CBOE VIX" in cross_asset.index:
            vix = pd.to_numeric(cross_asset.loc["CBOE VIX"].get("Latest"), errors="coerce")
            if pd.notna(vix) and vix >= 25:
                market_adjustment -= 3
        if "Crude oil" in cross_asset.index:
            crude_return = pd.to_numeric(cross_asset.loc["Crude oil"].get("One-year return (%)"), errors="coerce")
            if pd.notna(crude_return) and crude_return >= 20:
                market_adjustment -= 2
        if "USD/INR" in cross_asset.index:
            currency_return = pd.to_numeric(cross_asset.loc["USD/INR"].get("One-year return (%)"), errors="coerce")
            if pd.notna(currency_return) and currency_return >= 5:
                market_adjustment -= 2
        market_adjustment = max(-15, min(3, market_adjustment))

    unclassified_count = 0
    rejected_history_count = 0
    for _, row in screen_data.iterrows():
        symbol = str(row.get("Symbol", "")).upper()
        history = price_data.get(symbol)
        sector = str(row.get("Sector", "Unclassified")).strip()
        if not sector or sector.lower() == "unclassified":
            unclassified_count += 1
            continue
        if history is None or history.empty or "close" not in history.columns or history.attrs.get("synthetic"):
            rejected_history_count += 1
            continue
        close = pd.to_numeric(history["close"], errors="coerce").dropna()
        if close.empty:
            rejected_history_count += 1
            continue

        price = float(close.iloc[-1])
        ma20 = float(close.rolling(20, min_periods=1).mean().iloc[-1])
        ma50 = float(close.rolling(50, min_periods=1).mean().iloc[-1])
        ma200 = float(close.rolling(200, min_periods=1).mean().iloc[-1])
        rsi = float(calculate_rsi(history).iloc[-1])
        volume = pd.to_numeric(history.get("volume"), errors="coerce")
        volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean()) if volume is not None and volume.tail(20).mean() else 1.0
        historical = _historical_metrics(history)
        score = 50 - macro_penalty + market_adjustment
        drivers = []
        if price > ma20 > ma50:
            score += 12
            drivers.append("bullish trend")
        elif price < ma20 < ma50:
            score -= 12
            drivers.append("weak trend")
        if price > ma50 > ma200:
            score += 8
            drivers.append("long-term trend confirmation")
        elif price < ma50 < ma200:
            score -= 8
            drivers.append("long-term trend weakness")
        if 45 <= rsi <= 68:
            score += 8
            drivers.append("healthy RSI")
        elif rsi > 75 or rsi < 30:
            score -= 8
            drivers.append("extreme RSI")
        if volume_ratio >= 1.2:
            score += 12
            drivers.append("volume confirmation")
        else:
            drivers.append("normal volume")
        fii_value = pd.to_numeric(row.get("FII change (%)"), errors="coerce")
        dii_value = pd.to_numeric(row.get("DII change (%)"), errors="coerce")
        fii_change = float(fii_value) if pd.notna(fii_value) else 0.0
        dii_change = float(dii_value) if pd.notna(dii_value) else 0.0
        score += min(12, max(-5, fii_change + dii_change))
        if fii_change > 0 and dii_change > 0:
            drivers.append("FII/DII accumulation")
        cagr = historical["cagr"]
        if pd.notna(cagr):
            score += 10 if cagr >= 12 else 5 if cagr > 0 else -10
            drivers.append(f"{historical['history_years']:.1f}y CAGR {cagr:+.1f}%")
        if pd.notna(historical["max_drawdown"]) and historical["max_drawdown"] <= -60:
            score -= 5
            drivers.append("high historical drawdown")
        volatility = historical["annualized_volatility"]
        if pd.notna(volatility):
            if volatility >= 60:
                score -= 5
                drivers.append(f"high annualized volatility {volatility:.1f}%")
            elif volatility >= 40:
                score -= 2
                drivers.append(f"elevated annualized volatility {volatility:.1f}%")
        if market_adjustment:
            drivers.append("broad-market trend overlay")
        if macro_penalty:
            drivers.append(f"{macro_risk.lower()} macro overlay")
        score = max(0, min(100, round(score, 1)))
        view = "BUY" if score >= 68 else "HOLD" if score >= 48 else "SELL"
        stop_loss = round(price * (0.90 if macro_risk == "High" else 0.92), 2)
        target_1 = round(price * 1.04, 2)
        target_2 = round(price * 1.08, 2)
        target_3 = round(price * 1.12, 2)
        one_year_return = historical["one_year_return"]
        sector_strength = (
            (one_year_return + cagr) / 2
            if pd.notna(one_year_return) and pd.notna(cagr)
            else one_year_return if pd.notna(one_year_return) else cagr
        )
        ideas.append(
            {
                "Sector": sector,
                "Symbol": symbol,
                "Company": row.get("Company", symbol),
                "Score": score,
                "View": view,
                "One-year return (%)": round(one_year_return, 2) if pd.notna(one_year_return) else np.nan,
                "CAGR (available history, %)": round(cagr, 2) if pd.notna(cagr) else np.nan,
                "Annualized volatility (%)": round(volatility, 2) if pd.notna(volatility) else np.nan,
                "History years": round(historical["history_years"], 1),
                "Max drawdown (%)": round(historical["max_drawdown"], 2) if pd.notna(historical["max_drawdown"]) else np.nan,
                "_sector_strength": sector_strength,
                "Entry": round(price, 2),
                "Stop loss": stop_loss,
                "Target 1": target_1,
                "Target 2": target_2,
                "Target 3": target_3,
                "Horizon": "2-6 weeks",
                "Drivers": ", ".join(drivers),
            }
        )

    if not ideas:
        result = pd.DataFrame(columns=columns)
        result.attrs["unclassified_count"] = unclassified_count
        result.attrs["rejected_history_count"] = rejected_history_count
        return result

    candidates = pd.DataFrame(ideas)
    sector_counts = candidates.groupby("Sector")["Symbol"].count()
    underfilled_sectors = [f"{sector} ({count}/3)" for sector, count in sector_counts.items() if count < 3]
    sector_summaries = candidates.groupby("Sector").agg(
        candidate_count=("Symbol", "count"),
        strength=("_sector_strength", "mean"),
        one_year_return=("One-year return (%)", "mean"),
        cagr=("CAGR (available history, %)", "mean"),
    )
    sector_summaries["strength"] = sector_summaries["strength"].fillna(-np.inf)
    sector_summaries = sector_summaries[sector_summaries["candidate_count"] >= 3].sort_values("strength", ascending=False)

    ranked_rows = []
    for sector_rank, (sector, sector_summary) in enumerate(sector_summaries.iterrows(), start=1):
        sector_rows = candidates[candidates["Sector"] == sector].sort_values(["Score", "View"], ascending=[False, True]).head(3).copy()
        one_year_return = sector_summary["one_year_return"]
        trend_return = one_year_return if pd.notna(one_year_return) else sector_summary["cagr"]
        if pd.notna(trend_return):
            sector_trend = "Strong" if trend_return >= 15 else "Positive" if trend_return > 0 else "Weak" if trend_return <= -10 else "Mixed"
        else:
            sector_trend = "Positive" if sector_summary["strength"] > 0 else "Weak"
        sector_rows.insert(0, "Sector rank", sector_rank)
        sector_rows.insert(2, "Sector trend", sector_trend)
        sector_rows.insert(3, "Rank", range(1, len(sector_rows) + 1))
        ranked_rows.append(sector_rows)

    result = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else pd.DataFrame(columns=columns)
    result = result[columns]
    result.attrs["unclassified_count"] = unclassified_count
    result.attrs["rejected_history_count"] = rejected_history_count
    result.attrs["underfilled_sectors"] = underfilled_sectors
    return result


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
    if data.empty or data.attrs.get("synthetic"):
        reason = "Real historical price data is unavailable; no recommendation was generated." if data.attrs.get("synthetic") else "No valid stock data available."
        return {
            "action": "HOLD",
            "confidence": 0,
            "risk_level": "Low",
            "reasons": [reason],
            "entry_price": None,
            "entry_partial_price": None,
            "target_prices": [None, None, None],
            "target_price": None,
            "stop_loss": None,
            "price_change": None,
            "rsi": None,
            "institutional_flow": "unknown",
            "summary": reason,
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
            "price_change": None,
            "rsi": None,
            "institutional_flow": "unknown",
            "summary": "No usable close-price series was found.",
        }

    price = float(closes.iloc[-1])
    start_price = float(closes.iloc[0])
    historical = _historical_metrics(data)
    total_return = (price / start_price - 1.0) * 100 if start_price else 0.0
    percent_change = historical["one_year_return"] if pd.notna(historical["one_year_return"]) else total_return
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

    cagr = historical["cagr"]
    if pd.notna(cagr):
        reasons.append(f"Available long-term history: {historical['history_years']:.1f} years at {cagr:+.1f}% annualized growth.")
        score += 10 if cagr >= 12 else 5 if cagr > 0 else -10
    if pd.notna(historical["max_drawdown"]) and historical["max_drawdown"] <= -60:
        reasons.append(f"Long-term maximum drawdown reached {historical['max_drawdown']:.1f}%, indicating elevated historical risk.")
        score -= 5

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
        f"{action} view: {price:.2f} with {percent_change:.2f}% one-year return, "
        f"RSI {rsi:.1f}, and institutional flow support is {'positive' if fii_net > 0 else 'mixed' if fii_net == 0 else 'negative'}."
    )

    return {
        "action": action,
        "confidence": confidence,
        "risk_level": risk_level,
        "reasons": reasons[:8],
        "entry_price": entry_price,
        "entry_partial_price": entry_partial_price,
        "target_prices": [target_1, target_2, target_3],
        "target_price": target_price,
        "stop_loss": stop_loss,
        "price_change": round(percent_change, 2),
        "rsi": round(rsi, 1),
        "institutional_flow": "positive" if fii_net > 0 else "mixed" if fii_net == 0 else "negative",
        "summary": summary,
    }
