from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stock_analysis.analysis import (  # noqa: E402
    calculate_moving_average,
    compute_daily_returns,
    generate_recommendations,
    summarize_stock,
)
from stock_analysis.data import (  # noqa: E402
    fetch_bulk_deals,
    fetch_fii_data,
    fetch_stock_profile,
    get_optional_ai_summary,
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
    </style>
    """,
    unsafe_allow_html=True,
)


def format_currency(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        numeric = float(value)
        return f"₹{numeric:,.2f}"
    except (TypeError, ValueError):
        return str(value)


st.title("NSE Stock Analysis Dashboard")
st.caption("Enter a stock name or ticker to load its NSE market data and key indicators.")

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
        returns = compute_daily_returns(data)
        summary = summarize_stock(data)

        st.subheader(f"{profile.get('longName') or profile.get('shortName') or ticker}")
        st.caption(f"{ticker} • {profile.get('sector') or 'N/A'} • {profile.get('industry') or 'N/A'}")

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
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Market cap</div><div class='snapshot-value'>{format_currency(float(profile.get('marketCap') or 0))}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>P/E</div><div class='snapshot-value'>{profile.get('trailingPE') or 'N/A'}</div></div>", unsafe_allow_html=True)

        with right_col:
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Change</div><div class='snapshot-value {('positive' if summary['percent_change'] > 0 else 'negative' if summary['percent_change'] < 0 else 'neutral')}'>{summary['percent_change']:.2f}%</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Volatility</div><div class='snapshot-value'>{summary['volatility']:.2f}%</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>52W Low</div><div class='snapshot-value'>{format_currency(profile.get('fiftyTwoWeekLow'))}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Book value</div><div class='snapshot-value'>{profile.get('priceToBook') or 'N/A'}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='snapshot-card'><div class='snapshot-label'>Dividend yield</div><div class='snapshot-value'>{((profile.get('dividendYield') or 0) * 100):.2f}%</div></div>", unsafe_allow_html=True)

        portfolio_df = pd.DataFrame(
            {
                "Portfolio Item": ["Invested value", "Current value", "P/L", "Action"],
                "Value": [
                    "₹1,00,000",
                    f"₹{(summary['latest_close'] * 80):,.0f}",
                    f"₹{(summary['latest_close'] * 80 - 100000):,.0f}",
                    "Hold",
                ],
            }
        )
        st.markdown("### Compact portfolio summary")
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

        tab_prices, tab_profile, tab_stats, tab_table, tab_fii, tab_bulk, tab_reco = st.tabs(["Price charts", "Company profile", "Valuation & stats", "Data table", "FII / DII", "Bulk deals", "Recommendations"])

        with tab_prices:
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
            st.plotly_chart(candlestick, use_container_width=True)

            price_chart = go.Figure()
            price_chart.add_trace(go.Scatter(x=data["date"], y=data["close"], mode="lines", name="Close", line=dict(color="#2563eb", width=2)))
            price_chart.add_trace(go.Scatter(x=data["date"], y=data["ma_20"], mode="lines", name="20-day MA", line=dict(color="#f59e0b", dash="dash")))
            price_chart.add_trace(go.Scatter(x=data["date"], y=data["ma_50"], mode="lines", name="50-day MA", line=dict(color="#10b981", dash="dot")))
            price_chart.update_layout(title=f"{ticker} Price Trend", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(price_chart, use_container_width=True)

            volume_chart = px.bar(data, x="date", y="volume", title=f"{ticker} Trading Volume", template="plotly_white")
            volume_chart.update_xaxes(title="Date")
            volume_chart.update_yaxes(title="Volume")
            st.plotly_chart(volume_chart, use_container_width=True)

        with tab_profile:
            st.markdown(
                f"""
                **Company**: {profile.get('longName') or profile.get('shortName') or ticker}

                **Sector**: {profile.get('sector') or 'N/A'}

                **Industry**: {profile.get('industry') or 'N/A'}

                **Exchange**: {profile.get('exchange') or 'NSE'}

                **Website**: {profile.get('website') or 'N/A'}
                """
            )

            if profile.get("website"):
                st.markdown(f"[Official website]({profile.get('website')})")

            st.subheader("Daily return distribution")
            returns_pct = returns * 100
            hist = px.histogram(returns_pct, nbins=25, title=f"{ticker} Daily Return Distribution", template="plotly_white")
            hist.update_xaxes(title="Daily return (%)")
            hist.update_yaxes(title="Count")
            st.plotly_chart(hist, use_container_width=True)

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
            st.dataframe(stats_df, use_container_width=True)

            fig = px.scatter(
                x=data["close"],
                y=data["volume"],
                title=f"{ticker} Price vs Volume",
                labels={"x": "Close Price", "y": "Volume"},
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_table:
            display = data.tail(20)[["date", "open", "high", "low", "close", "volume"]].copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display, use_container_width=True)

        with tab_fii:
            fii_df = fetch_fii_data(limit=12)
            if fii_df.empty:
                st.info("No recent FII/DII data is available right now.")
            else:
                fii_view = fii_df[["date", "category", "buyValue", "sellValue", "netValue"]].copy()
                fii_view["date"] = fii_view["date"].dt.strftime("%d-%b-%Y")
                st.dataframe(fii_view, use_container_width=True)

                net_flow = fii_df.groupby("category", as_index=False)["netValue"].sum()
                flow_chart = px.bar(
                    net_flow,
                    x="category",
                    y="netValue",
                    color="category",
                    title="Recent FII / DII Net Flow",
                    template="plotly_white",
                    labels={"category": "Investor Type", "netValue": "Net Investment (₹ crore)"},
                )
                flow_chart.update_layout(showlegend=False)
                st.plotly_chart(flow_chart, use_container_width=True)

                total_net = fii_df["netValue"].sum()
                st.metric("Overall net allocation", f"₹{total_net:,.2f} crore")
                st.caption("Positive values suggest net buying; negative values suggest net selling.")

        with tab_bulk:
            bulk_df = fetch_bulk_deals(limit=12)
            if bulk_df.empty:
                st.info("There are no recent bulk deals available right now.")
            else:
                bulk_view = bulk_df.copy()
                bulk_view["Date"] = bulk_view["Date"].dt.strftime("%d-%b-%Y")
                st.dataframe(bulk_view[["Date", "Symbol", "Security Name", "Client Name", "Buy/Sell", "Quantity Traded", "Trade Price / Wght. Avg. Price", "Remarks"]], use_container_width=True)

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
                st.plotly_chart(bulk_chart, use_container_width=True)

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

            st.markdown(
                f"<div class='recommendation-header'>"
                f"<div class='recommendation-label'>Trading view</div>"
                f"<div class='recommendation-action {action_class}'>{action}</div>"
                f"<div class='trade-note'>Confidence: {recommendation.get('confidence', 0)}% &nbsp; | &nbsp; Risk: {recommendation.get('risk_level', 'Unknown')}</div>"
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

            st.markdown(f"**Summary:** {recommendation.get('summary', 'No recommendation summary is available.')}")

            st.markdown("### Why this view?")
            for reason in recommendation["reasons"]:
                st.markdown(f"- {reason}")

            ai_summary = get_optional_ai_summary(symbol_input, summary, profile, fii_df, bulk_df)
            st.markdown("### Optional AI enhancement")
            st.caption("Set OPENAI_API_KEY locally in your environment to enable a richer narrative. The key stays outside this repo and is never stored in source control.")
            st.info(ai_summary)

    except ValueError as exc:
        st.error(str(exc))
else:
    st.info("Enter a stock symbol or company name to view the company profile and charts.")
