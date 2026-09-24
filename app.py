from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stock_analysis.analysis import (  # noqa: E402
    analyze_institutional_trend,
    calculate_macd,
    calculate_moving_average,
    calculate_rsi,
    compute_daily_returns,
    generate_recommendations,
    summarize_stock,
)
from stock_analysis.data import (  # noqa: E402
    fetch_bulk_deals,
    fetch_fii_data,
    fetch_stock_profile,
    fetch_screener_fii_dii_top_stocks,
    load_stock_data,
    normalize_stock_symbol,
)

st.set_page_config(page_title="NSE Stock Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .streamlit-expanderHeader { font-size: 16px; }
    .stMetric { background: #eaf4fb; border-radius: 8px; padding: 8px 10px; }
    .stMetric > div { font-size: 15px !important; }
    .stMetric .stMetricValue { font-size: 2.2rem !important; line-height: 1.2; }
    .dataframe { font-size: 15px; }
    .snapshot-card {
        background: #eaf4fb;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
        border: 1px solid #bfd9ee;
    }
    .snapshot-label {
        font-size: 14px;
        color: #3f4d5a;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .snapshot-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2a37;
        line-height: 1.2;
    }
    .positive { color: #0f9d58; }
    .negative { color: #d93025; }
    .neutral { color: #1f2a37; }
    .recommendation-header {
        border: 1px solid #d6dee7;
        border-radius: 10px;
        padding: 14px 16px;
        background: #f8fafc;
        margin: 8px 0 14px;
    }
    .recommendation-label {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #52606d;
        margin-bottom: 5px;
    }
    .recommendation-action {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .action-buy { color: #087f5b; }
    .action-hold { color: #b26a00; }
    .action-sell { color: #c92a2a; }
    .trade-card {
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 112px;
        margin: 6px 0 14px;
        border: 1px solid;
    }
    .entry-card { background: #eef7ff; border-color: #b8d8f2; }
    .target-card { background: #effaf3; border-color: #b7e3c7; }
    .stop-card { background: #fff1f1; border-color: #f1b8b8; }
    .trade-label {
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #3f4d5a;
        margin-bottom: 8px;
    }
    .trade-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1f2a37;
        line-height: 1.3;
    }
    .trade-note {
        font-size: 13px;
        color: #52606d;
        margin-top: 5px;
    }
    .summary-card {
        border: 1px solid #d6dee7;
        border-left: 5px solid #c92a2a;
        border-radius: 10px;
        padding: 14px 16px;
        background: #fffafa;
        margin: 10px 0 16px;
    }
    .summary-title {
        font-size: 15px;
        font-weight: 800;
        color: #263238;
        margin-bottom: 10px;
    }
    .summary-text {
        font-size: 16px;
        color: #3f4d5a;
        line-height: 1.5;
    }
    .status-strip {
        border: 1px solid #d6dee7;
        border-radius: 8px;
        padding: 8px 12px;
        background: #f8fafc;
        color: #52606d;
        font-size: 13px;
        margin: 8px 0 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_currency(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        numeric = float(value)
        if pd.isna(numeric) or numeric == 0:
            return "N/A"
        return f"₹{numeric:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_market_cap(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        numeric = float(value)
        if pd.isna(numeric) or numeric <= 0:
            return "N/A"
        if numeric >= 1_00_00_00_00_000:
            return f"₹{numeric / 1_00_00_00_00_000:.2f} lakh crore"
        if numeric >= 1_00_00_000:
            return f"₹{numeric / 1_00_00_000:.2f} crore"
        return f"₹{numeric:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def display_value(value: Any, formatter=format_currency) -> str:
    if value is None or value == "":
        return "N/A"
    return formatter(value)


st.title("NSE Stock Analysis Dashboard")
st.caption("Enter a stock name or ticker to load its NSE market data and key indicators.")

workspace = st.sidebar.radio("Workspace", ["Stock workspace", "Market screener"], index=0)
st.sidebar.caption("Use Stock workspace for chart and trade ideas. Use Market screener for FII/DII discovery.")

symbol_input = st.text_input("Stock name or ticker", value="Reliance")
period = st.selectbox("Time range", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

if symbol_input:
    try:
        ticker = normalize_stock_symbol(symbol_input)
        profile = fetch_stock_profile(symbol_input)
        data = load_stock_data(symbol=ticker, period=period)
        data = data.sort_values("date").reset_index(drop=True)
        data = calculate_moving_average(data, window=20)
        data = calculate_moving_average(data, window=50)
        data = calculate_macd(data)
        data["rsi_14"] = calculate_rsi(data).reindex(data.index).fillna(50)
        returns = compute_daily_returns(data)
        summary = summarize_stock(data)

        top_stocks = fetch_screener_fii_dii_top_stocks(limit=10)
        if workspace == "Market screener":
            st.subheader("Market screener")
            st.caption("Screener.in screen: FII holding change > 0.3%, DII holding change > 0.3%, market cap > ₹1,000 crore.")
            screener_filter = st.text_input("Filter companies", placeholder="Search company name")
            if top_stocks.empty:
                st.info("The Screener.in top-10 list is temporarily unavailable.")
            else:
                screener_view = top_stocks.copy()
                if screener_filter:
                    screener_view = screener_view[screener_view["Company"].str.contains(screener_filter, case=False, na=False)]
                st.dataframe(screener_view, width="stretch", hide_index=True)
            st.caption("Source: Screener.in screen 1340210. Values may be delayed and should be verified before trading.")
            st.stop()

        st.subheader("Stock overview")
        if top_stocks.empty:
            st.info("The Screener.in top-10 list is temporarily unavailable.")
        else:
            with st.expander("Top 10 FII and DII buying stocks", expanded=False):
                st.dataframe(top_stocks, width="stretch", hide_index=True)

        st.subheader(f"{profile.get('longName') or profile.get('shortName') or ticker}")
        st.caption(f"{ticker} • {profile.get('sector') or 'N/A'} • {profile.get('industry') or 'N/A'}")
        st.markdown(
            f"<div class='status-strip'><strong>Data status:</strong> Yahoo Finance price/profile feed &nbsp;|&nbsp; "
            f"<strong>Period:</strong> {period} &nbsp;|&nbsp; <strong>Updated:</strong> {datetime.now().strftime('%d-%b-%Y %H:%M')}</div>",
            unsafe_allow_html=True,
        )

        def color_value(raw_value: str, numeric_value: float | None = None) -> str:
            if numeric_value is None:
                return raw_value
            css_class = "neutral"
            if numeric_value > 0:
                css_class = "positive"
            elif numeric_value < 0:
                css_class = "negative"
            return f'<span class="{css_class}">{raw_value}</span>'

        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Latest close</div><div class='snapshot-value'>{format_currency(summary['latest_close'])}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Avg daily return</div><div class='snapshot-value {('positive' if summary['average_daily_return'] > 0 else 'negative' if summary['average_daily_return'] < 0 else 'neutral')}'>{summary['average_daily_return']:.2f}%</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>52W High</div><div class='snapshot-value'>{format_currency(profile.get('fiftyTwoWeekHigh'))}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Market cap</div><div class='snapshot-value'>{format_market_cap(profile.get('marketCap'))}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>P/E</div><div class='snapshot-value'>{display_value(profile.get('trailingPE'))}</div></div>", unsafe_allow_html=True)

        with right_col:
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Change</div><div class='snapshot-value {('positive' if summary['percent_change'] > 0 else 'negative' if summary['percent_change'] < 0 else 'neutral')}'>{summary['percent_change']:.2f}%</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Volatility</div><div class='snapshot-value'>{summary['volatility']:.2f}%</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>52W Low</div><div class='snapshot-value'>{format_currency(profile.get('fiftyTwoWeekLow'))}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Book value</div><div class='snapshot-value'>{format_currency(profile.get('bookValue'))}</div></div>", unsafe_allow_html=True)
            dividend_yield = profile.get('dividendYield')
            dividend_text = f"{float(dividend_yield) * 100:.2f}%" if dividend_yield is not None else "N/A"
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Dividend yield</div><div class='snapshot-value'>{dividend_text}</div></div>", unsafe_allow_html=True)

        with st.expander("Portfolio position", expanded=False):
            portfolio_col1, portfolio_col2 = st.columns(2)
            with portfolio_col1:
                shares = st.number_input("Quantity", min_value=0.0, value=0.0, step=1.0)
            with portfolio_col2:
                average_buy_price = st.number_input("Average buy price (₹)", min_value=0.0, value=0.0, step=0.05)

            if shares > 0 and average_buy_price > 0:
                invested_value = shares * average_buy_price
                current_value = shares * summary["latest_close"]
                profit_loss = current_value - invested_value
                return_pct = (profit_loss / invested_value) * 100
                portfolio_df = pd.DataFrame(
                    {
                        "Portfolio item": ["Invested value", "Current value", "P/L", "Return"],
                        "Value": [
                            format_currency(invested_value),
                            format_currency(current_value),
                            format_currency(profit_loss),
                            f"{return_pct:+.2f}%",
                        ],
                    }
                )
                st.dataframe(portfolio_df, width="stretch", hide_index=True)
            else:
                st.info("Enter quantity and average buy price to calculate your real position value and return.")

        tab_prices, tab_profile, tab_stats, tab_table, tab_fii, tab_bulk, tab_reco = st.tabs(["Price charts", "Company profile", "Valuation & stats", "Data table", "FII / DII", "Bulk deals", "Recommendations"])

        with tab_prices:
            technical_chart = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.025,
                row_heights=[0.52, 0.14, 0.17, 0.17],
                subplot_titles=("Price / trend", "Volume", "RSI 14", "MACD 12 26 9"),
            )
            technical_chart.add_trace(
                go.Candlestick(
                    x=data["date"],
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name="Price",
                    increasing_line_color="#0f9d58",
                    decreasing_line_color="#d93025",
                ),
                row=1,
                col=1,
            )
            technical_chart.add_trace(go.Scatter(x=data["date"], y=data["ma_20"], name="EMA 20", line={"color": "#f59e0b", "width": 2}), row=1, col=1)
            technical_chart.add_trace(go.Scatter(x=data["date"], y=data["ma_50"], name="EMA 50", line={"color": "#7c3aed", "width": 2}), row=1, col=1)

            bullish_cross = (data["ma_20"] > data["ma_50"]) & (data["ma_20"].shift(1) <= data["ma_50"].shift(1))
            bearish_cross = (data["ma_20"] < data["ma_50"]) & (data["ma_20"].shift(1) >= data["ma_50"].shift(1))
            technical_chart.add_trace(go.Scatter(x=data.loc[bullish_cross, "date"], y=data.loc[bullish_cross, "close"], mode="markers+text", text=["BUY"] * int(bullish_cross.sum()), textposition="bottom center", marker={"symbol": "triangle-up", "size": 11, "color": "#0f9d58"}, name="BUY signal"), row=1, col=1)
            technical_chart.add_trace(go.Scatter(x=data.loc[bearish_cross, "date"], y=data.loc[bearish_cross, "close"], mode="markers+text", text=["SELL"] * int(bearish_cross.sum()), textposition="top center", marker={"symbol": "triangle-down", "size": 11, "color": "#d93025"}, name="SELL signal"), row=1, col=1)

            volume_colors = ["#0f9d58" if close >= open_price else "#d93025" for close, open_price in zip(data["close"], data["open"])]
            technical_chart.add_trace(go.Bar(x=data["date"], y=data["volume"], marker_color=volume_colors, name="Volume", showlegend=False), row=2, col=1)
            technical_chart.add_trace(go.Scatter(x=data["date"], y=data["rsi_14"], name="RSI", line={"color": "#7c3aed", "width": 2}), row=3, col=1)
            technical_chart.add_trace(go.Scatter(x=data["date"], y=[70] * len(data), name="RSI 70", line={"color": "#9ca3af", "dash": "dash", "width": 1}, showlegend=False), row=3, col=1)
            technical_chart.add_trace(go.Scatter(x=data["date"], y=[30] * len(data), name="RSI 30", line={"color": "#9ca3af", "dash": "dash", "width": 1}, showlegend=False), row=3, col=1)
            histogram_colors = ["#0f9d58" if value >= 0 else "#d93025" for value in data["macd_histogram"]]
            technical_chart.add_trace(go.Bar(x=data["date"], y=data["macd_histogram"], marker_color=histogram_colors, name="MACD histogram"), row=4, col=1)
            technical_chart.add_trace(go.Scatter(x=data["date"], y=data["macd"], name="MACD", line={"color": "#2563eb", "width": 2}), row=4, col=1)
            technical_chart.add_trace(go.Scatter(x=data["date"], y=data["macd_signal"], name="Signal", line={"color": "#f97316", "width": 2}), row=4, col=1)
            technical_chart.update_yaxes(title_text="Price", row=1, col=1)
            technical_chart.update_yaxes(title_text="Volume", row=2, col=1)
            technical_chart.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
            technical_chart.update_yaxes(title_text="MACD", row=4, col=1)
            technical_chart.update_layout(
                title=f"{ticker} Technical Analysis",
                template="plotly_white",
                height=1050,
                hovermode="x unified",
                xaxis_rangeslider_visible=False,
                legend={"orientation": "h", "y": 1.02, "x": 0},
                margin={"l": 55, "r": 25, "t": 90, "b": 35},
            )
            st.plotly_chart(technical_chart, width="stretch")

            candlestick = go.Figure(
                data=[
                    go.Candlestick(
                        x=data["date"],
                        open=data["open"],
                        high=data["high"],
                        low=data["low"],
                        close=data["close"],
                        increasing_line_color="#22c55e",
                        decreasing_line_color="#ef4444",
                        name="Price",
                    )
                ]
            )
            candlestick.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(candlestick, width="stretch")

            price_chart = go.Figure()
            price_chart.add_trace(go.Scatter(x=data["date"], y=data["close"], mode="lines", name="Close", line=dict(color="#2563eb", width=2)))
            price_chart.add_trace(go.Scatter(x=data["date"], y=data["ma_20"], mode="lines", name="20-day MA", line=dict(color="#f59e0b", dash="dash")))
            price_chart.add_trace(go.Scatter(x=data["date"], y=data["ma_50"], mode="lines", name="50-day MA", line=dict(color="#10b981", dash="dot")))
            price_chart.update_layout(title=f"{ticker} Price Trend", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(price_chart, width="stretch")

            volume_chart = px.bar(data, x="date", y="volume", title=f"{ticker} Trading Volume", template="plotly_white")
            volume_chart.update_xaxes(title="Date")
            volume_chart.update_yaxes(title="Volume")
            st.plotly_chart(volume_chart, width="stretch")

        with tab_profile:
            st.markdown(
                f"""
                **Company**: {profile.get('longName') or profile.get('shortName') or ticker}

                **Sector**: {profile.get('sector') or 'N/A'}

                **Industry**: {profile.get('industry') or 'N/A'}

                **Exchange**: {profile.get('exchange') or 'NSE'}

                **Company / NSE website**: {profile.get('website') or 'NSE quote page'}
                """
            )

            if profile.get("website"):
                st.markdown(f"[Open company / NSE website]({profile.get('website')})")

            st.subheader("Daily return distribution")
            returns_pct = returns * 100
            hist = px.histogram(returns_pct, nbins=25, title=f"{ticker} Daily Return Distribution", template="plotly_white")
            hist.update_xaxes(title="Daily return (%)")
            hist.update_yaxes(title="Count")
            st.plotly_chart(hist, width="stretch")

        with tab_stats:
            metrics = {
                "Regular Market Price": profile.get("regularMarketPrice"),
                "Market Cap": profile.get("marketCap"),
                "52 Week High": profile.get("fiftyTwoWeekHigh"),
                "52 Week Low": profile.get("fiftyTwoWeekLow"),
                "Trailing P/E": profile.get("trailingPE"),
                "Forward P/E": profile.get("forwardPE"),
                "Price/Book": profile.get("priceToBook"),
                "Dividend Yield": profile.get("dividendYield"),
                "Beta": profile.get("beta"),
                "Average Volume": profile.get("averageVolume"),
            }
            stats_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
            st.dataframe(stats_df, width="stretch")

            fig = px.scatter(
                x=data["close"],
                y=data["volume"],
                title=f"{ticker} Price vs Volume",
                labels={"x": "Close Price", "y": "Volume"},
                template="plotly_white",
            )
            st.plotly_chart(fig, width="stretch")

        with tab_table:
            display = data.tail(20)[["date", "open", "high", "low", "close", "volume"]].copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display, width="stretch")

        with tab_fii:
            fii_df = fetch_fii_data(limit=30)
            if fii_df.empty:
                st.info("No recent FII/DII data is available right now.")
            else:
                fii_df = fii_df.sort_values("date").reset_index(drop=True)
                recent_dates = fii_df["date"].dropna().drop_duplicates().nlargest(5)
                recent_fii_df = fii_df[fii_df["date"].isin(recent_dates)].copy()
                recent_fii_df = recent_fii_df.sort_values("date").reset_index(drop=True)
                st.caption("Showing the latest five reported working days of FII/DII activity.")

                fii_view = recent_fii_df[["date", "category", "buyValue", "sellValue", "netValue"]].copy()
                fii_view["date"] = fii_view["date"].dt.strftime("%d-%b-%Y")
                st.dataframe(fii_view, width="stretch")

                trend_summary = analyze_institutional_trend(recent_fii_df, recent_window=3)
                st.subheader("Short-term flow outlook")
                if trend_summary.empty:
                    st.info("There is not enough institutional history to estimate a trend.")
                else:
                    outlook = trend_summary.rename(
                        columns={
                            "category": "Investor type",
                            "recent_average": "Recent avg (₹ cr)",
                            "previous_average": "Previous avg (₹ cr)",
                            "change": "Change (₹ cr)",
                            "direction": "Direction",
                            "projected_next": "Projected next (₹ cr)",
                        }
                    )
                    st.dataframe(outlook, width="stretch", hide_index=True)
                    st.caption("Projection uses the change between the recent and previous five reported observations. It is a statistical signal, not a guaranteed forecast.")

                total_net = recent_fii_df["netValue"].sum()
                st.metric("Overall net allocation", f"₹{total_net:,.2f} crore")
                st.caption("Positive values suggest net buying; negative values suggest net selling.")

        with tab_bulk:
            bulk_df = fetch_bulk_deals(limit=12)
            if bulk_df.empty:
                st.info("There are no recent bulk deals available right now.")
            else:
                bulk_view = bulk_df.copy()
                bulk_view["Date"] = bulk_view["Date"].dt.strftime("%d-%b-%Y")
                st.dataframe(bulk_view[["Date", "Symbol", "Security Name", "Client Name", "Buy/Sell", "Quantity Traded", "Trade Price / Wght. Avg. Price", "Remarks"]], width="stretch")

                buy_sell = bulk_df["Buy/Sell"].value_counts().reset_index()
                buy_sell.columns = ["Action", "Count"]
                bulk_chart = px.bar(
                    buy_sell,
                    x="Action",
                    y="Count",
                    color="Action",
                    title="Recent Bulk Deal Direction",
                    template="plotly_white",
                )
                bulk_chart.update_layout(showlegend=False)
                st.plotly_chart(bulk_chart, width="stretch")

        with tab_reco:
            fii_df = fetch_fii_data(limit=12)
            bulk_df = fetch_bulk_deals(limit=12)
            recommendation = generate_recommendations(data, profile=profile, fii_df=fii_df, bulk_df=bulk_df)

            st.subheader(f"{profile.get('longName') or profile.get('shortName') or ticker}")

            action = str(recommendation.get("action", "HOLD")).upper()
            action_class = {"BUY": "action-buy", "SELL": "action-sell"}.get(action, "action-hold")
            entry_price = recommendation.get("entry_price") or summary["latest_close"]
            partial_entry = recommendation.get("entry_partial_price") or round(entry_price * 0.95, 2)
            targets = recommendation.get("target_prices") or []
            if len(targets) < 3 or any(level is None for level in targets[:3]):
                legacy_target = recommendation.get("target_price")
                target_2 = legacy_target or round(entry_price * 1.08, 2)
                targets = [round(entry_price * 1.04, 2), target_2, round(entry_price * 1.12, 2)]
            stop_loss = recommendation.get("stop_loss") or round(entry_price * 0.92, 2)
            risk_per_share = entry_price - stop_loss
            reward_per_share = targets[2] - entry_price
            risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
            holding_style = "Swing / positional" if period in {"3mo", "6mo", "1y", "2y", "5y"} else "Short-term"

            st.markdown(
                f"<div class='recommendation-header'>"
                f"<div class='recommendation-label'>Trading view</div>"
                f"<div class='recommendation-action {action_class}'>{action}</div>"
                f"<div class='trade-note'>Confidence: {recommendation.get('confidence', 0)}% &nbsp; | &nbsp; Risk: {recommendation.get('risk_level', 'Unknown')} &nbsp; | &nbsp; Style: {holding_style}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if entry_price is not None and len(targets) >= 3 and stop_loss is not None:
                entry_html = (
                    f"<div class='trade-value'>CMP {entry_price:.0f}</div>"
                    f"<div class='trade-note'><strong>80%</strong> at {entry_price:.0f} &nbsp; | &nbsp; <strong>20%</strong> at {partial_entry:.0f}</div>"
                )
                target_html = (
                    f"<div class='trade-value'>{targets[0]:.0f} &nbsp; {targets[1]:.0f} &nbsp; {targets[2]:.0f}+</div>"
                    f"<div class='trade-note'>Book partial profits at each level</div>"
                )
                stop_html = (
                    f"<div class='trade-value'>{stop_loss:.0f}</div>"
                    f"<div class='trade-note'>Exit if price closes below this level</div>"
                )
                entry_col, target_col, stop_col = st.columns(3)
                with entry_col:
                    st.markdown(f"<div class='trade-card entry-card'><div class='trade-label'>Entry</div>{entry_html}</div>", unsafe_allow_html=True)
                with target_col:
                    st.markdown(f"<div class='trade-card target-card'><div class='trade-label'>Targets</div>{target_html}</div>", unsafe_allow_html=True)
                with stop_col:
                    st.markdown(f"<div class='trade-card stop-card'><div class='trade-label'>Stop loss</div>{stop_html}</div>", unsafe_allow_html=True)
            else:
                st.warning("Entry, target, and stop-loss levels are unavailable because the market data is incomplete.")

            summary_change = recommendation.get("price_change")
            summary_change_text = f"{summary_change:+.2f}%" if summary_change is not None else "N/A"
            summary_rsi = recommendation.get("rsi")
            summary_rsi_text = f"{summary_rsi:.1f}" if summary_rsi is not None else "N/A"
            summary_flow = str(recommendation.get("institutional_flow", "unknown")).capitalize()
            summary_price = recommendation.get("entry_price")
            summary_price_text = format_currency(summary_price) if summary_price is not None else "N/A"
            st.markdown(
                f"<div class='summary-card'>"
                f"<div class='summary-title'>Summary</div>"
                f"<div class='summary-text'><strong class='{action_class}'>{action}</strong> view at <strong>{summary_price_text}</strong>.</div>"
                f"<div class='trade-note'><strong>Price change:</strong> {summary_change_text} &nbsp; | &nbsp; <strong>RSI:</strong> {summary_rsi_text} &nbsp; | &nbsp; <strong>Institutional flow:</strong> {summary_flow} &nbsp; | &nbsp; <strong>Risk/reward:</strong> 1:{risk_reward:.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Why this view?")
            for reason in recommendation["reasons"]:
                st.markdown(f"- {reason}")

    except ValueError as exc:
        st.error(str(exc))
else:
    st.info("Enter a stock symbol or company name to view the company profile and charts.")
