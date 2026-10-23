from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
    build_market_ideas,
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
    fetch_market_context,
    fetch_market_news,
    fetch_market_news_queries,
    fetch_stock_profile,
    fetch_stock_sector,
    fetch_screener_fii_dii_top_stocks,
    load_stock_data,
    normalize_stock_symbol,
)
from stock_analysis.long_term import (  # noqa: E402
    LONG_TERM_COLUMNS,
    build_long_term_ideas,
    select_long_term_candidates,
)
from stock_analysis.ipo_analysis import (  # noqa: E402
    IPOFeed,
    IPOFeedError,
    NSE_IPO_URL,
    fetch_ipo_issues,
    listing_price_scenarios,
    retail_allotment_proxy,
)
from stock_analysis.ipo_research import get_issuer_research  # noqa: E402
from stock_analysis.home_dashboard import (  # noqa: E402
    build_market_overview,
)

st.set_page_config(page_title="Indian Stock Dashboard", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #52606d;
        --line: #d8e2ea;
        --surface: rgba(255, 255, 255, 0.92);
        --blue-soft: #edf6ff;
        --green-soft: #eef9f3;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(155deg, #f4f9fd 0%, #ffffff 48%, #f1faf6 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.72);
    }
    [data-testid="stSidebar"] {
        background: #edf6f3;
        border-right: 1px solid #d5e5df;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    h1 {
        font-weight: 800;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }
    [data-baseweb="tab"] {
        color: var(--muted);
        font-weight: 700;
        padding: 0.7rem 1rem;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        color: #1261a0;
        border-bottom-color: #2383c4;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        background: var(--surface);
    }
    .streamlit-expanderHeader { font-size: 16px; }
    .stMetric { background: #eaf4fb; border-radius: 8px; padding: 8px 10px; }
    .stMetric > div { font-size: 15px !important; }
    .stMetric .stMetricValue { font-size: 2.2rem !important; line-height: 1.2; }
    .dataframe { font-size: 15px; }
    .snapshot-card {
        background: var(--blue-soft);
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
        border: 1px solid #c6ddec;
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


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_history(symbol: str, period: str, exchange: str) -> pd.DataFrame:
    return load_stock_data(symbol=symbol, period=period, exchange=exchange)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_stock_sector(symbol: str, exchange: str) -> str:
    return fetch_stock_sector(symbol, exchange=exchange)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_screener_page(limit: int, page: int) -> pd.DataFrame:
    return fetch_screener_fii_dii_top_stocks(limit=limit, page=page)


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_context() -> pd.DataFrame:
    return fetch_market_context()


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_news() -> pd.DataFrame:
    return fetch_market_news()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_long_term_ideas(market_cap_limit: float, exchange: str) -> pd.DataFrame:
    pages = [cached_screener_page(limit=100, page=page) for page in range(1, 5)]
    screen_data = pd.concat(pages, ignore_index=True).drop_duplicates("Symbol")
    candidates = select_long_term_candidates(screen_data, market_cap_limit)
    if candidates.empty:
        return pd.DataFrame(columns=LONG_TERM_COLUMNS)

    symbols = candidates["Symbol"].dropna().astype(str).str.upper().tolist()
    with ThreadPoolExecutor(max_workers=8) as executor:
        histories = dict(
            zip(
                symbols,
                executor.map(
                    lambda symbol: cached_stock_history(symbol, "3y", exchange),
                    symbols,
                ),
            )
        )
    return build_long_term_ideas(candidates, histories, market_cap_limit)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_home_market_data(
    exchange: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pages = [cached_screener_page(limit=100, page=page) for page in range(1, 5)]
    screen_data = pd.concat(pages, ignore_index=True).drop_duplicates("Symbol")
    symbols = screen_data["Symbol"].dropna().astype(str).str.upper().tolist()
    with ThreadPoolExecutor(max_workers=8) as executor:
        histories = dict(
            zip(
                symbols,
                executor.map(lambda symbol: cached_stock_history(symbol, "3y", exchange), symbols),
            )
        )
        sector_values = dict(
            zip(
                symbols,
                executor.map(lambda symbol: cached_stock_sector(symbol, exchange), symbols),
            )
        )
    stocks, sectors = build_market_overview(screen_data, histories, sector_values)
    market_context = cached_market_context()

    sector_names = sectors["Sector"].astype(str).tolist()
    sector_queries = {
        f"Sector news · {sector}": (
            f"India {sector} industry stocks earnings demand policy global market"
        )
        for sector in sector_names
    }
    recent_news_filter = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    news_queries = {
        "Global markets and geopolitics": (
            f"global markets economy geopolitics trade policy oil after:{recent_news_filter}"
        ),
        "India markets and policy": (
            f"India stock market economy government policy after:{recent_news_filter}"
        ),
        **{
            topic: f"{query} after:{recent_news_filter}"
            for topic, query in sector_queries.items()
        },
    }
    news = fetch_market_news_queries(news_queries, limit=2)
    return stocks, sectors, market_context, news


@st.cache_data(ttl=900, show_spinner=False)
def cached_ipo_issues() -> IPOFeed:
    return fetch_ipo_issues()


st.title("Indian Stock Analysis Dashboard")
st.caption("Analyze listed stocks on the National Stock Exchange or Bombay Stock Exchange.")

workspace = st.sidebar.radio(
    "Workspace",
    [
        "Home / Market overview",
        "Stock workspace",
        "Market screener",
        "Market ideas",
        "Long-term ideas",
        "IPO analysis",
    ],
    index=0,
)
st.sidebar.caption("Home: market snapshot. Stock workspace: analysis. Market screener: discovery. Market ideas: ranked setups. Long-term ideas: independent small-cap screen. IPO analysis: current and forthcoming NSE issues.")

if workspace == "Home / Market overview":
    exchange = "NSE"
    symbol_input = ""
    period = "1y"
elif workspace == "IPO analysis":
    exchange = ""
    symbol_input = ""
    period = ""
else:
    exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=0)
    symbol_input = st.text_input("Stock name or ticker", value="Reliance")
    period = st.selectbox("Time range", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "15y"], index=3)

if workspace == "IPO analysis":
    st.subheader("Current and forthcoming IPOs")
    st.caption(
        "Issue terms and live subscription are from NSE. Fundamentals and "
        "listing-price scenarios are evidence-limited; read the linked offer "
        "documents before making an investment decision."
    )
    st.link_button("Open official NSE IPO page", NSE_IPO_URL)
    try:
        with st.spinner("Loading current and forthcoming public issues from NSE..."):
            ipo_feed = cached_ipo_issues()
    except IPOFeedError as exc:
        st.error(f"Unable to load NSE IPO data: {exc}")
        st.stop()

    if ipo_feed.issues.empty:
        st.info("NSE currently reports no current or forthcoming IPO issues.")
    else:
        issues = ipo_feed.issues
        st.caption(
            f"NSE feed refreshed {ipo_feed.fetched_at.strftime('%d %b %Y, %H:%M %Z')}. "
            f"{len(issues)} issue(s) found."
        )
        for issue_index, issue in enumerate(issues.to_dict("records")):
            name = str(issue.get("company") or issue.get("symbol") or "IPO")
            status = str(issue.get("status") or "Status unavailable")
            with st.expander(f"{name} · {status}", expanded=issue_index == 0):
                price_low = issue.get("price_low")
                price_high = issue.get("price_high")
                price_band_text = (
                    f"₹{price_low:,.2f}–₹{price_high:,.2f}"
                    if pd.notna(price_low) and pd.notna(price_high)
                    else str(issue.get("price_band") or "Not published")
                )
                start_date = issue.get("issue_start_date")
                end_date = issue.get("issue_end_date")
                date_text = (
                    f"{start_date:%d %b %Y} – {end_date:%d %b %Y}"
                    if pd.notna(start_date) and pd.notna(end_date)
                    else "Dates not published"
                )
                summary_cols = st.columns(5)
                summary_cols[0].metric("Issue status", status)
                summary_cols[1].metric("Offer price band", price_band_text)
                summary_cols[2].metric("Bidding period", date_text)
                total_subscription = issue.get("total_subscription")
                summary_cols[3].metric(
                    "Total subscription",
                    f"{total_subscription:.2f}×"
                    if pd.notna(total_subscription)
                    else "Not available yet",
                )
                retail_subscription = issue.get("retail_subscription")
                summary_cols[4].metric(
                    "Retail subscription",
                    f"{retail_subscription:.2f}×"
                    if pd.notna(retail_subscription)
                    else "Not available yet",
                )

                st.markdown("#### Issuer fundamentals and offer terms")
                issue_size = issue.get("issue_size_description")
                if issue_size:
                    st.markdown(f"- **Issue structure:** {issue_size}")
                else:
                    st.markdown("- **Issue structure:** Not provided in the NSE summary feed.")
                sector = issue.get("sector")
                if sector:
                    st.markdown(f"- **Sector / industry:** {sector}")
                lot_size = issue.get("lot_size")
                if lot_size:
                    st.markdown(f"- **Minimum bid lot:** {lot_size}")
                face_value = issue.get("face_value")
                if face_value:
                    st.markdown(f"- **Face value:** {face_value}")
                business_description = issue.get("business_description")
                if business_description:
                    st.markdown(f"- **Business description:** {business_description}")
                recent_expansion = issue.get("recent_expansion")
                if recent_expansion:
                    st.markdown(f"- **Expansion / capex disclosed:** {recent_expansion}")
                order_book = issue.get("order_book")
                if order_book:
                    st.markdown(f"- **Order book disclosed:** {order_book}")

                research = get_issuer_research(str(issue.get("symbol") or ""))
                if research:
                    st.markdown("#### Issuer research")
                    st.caption(
                        f"Prospectus-based snapshot · {research.researched_on}. "
                        "Reported amounts retain the source units and periods."
                    )
                    st.markdown(f"- **Sector:** {research.sector}")
                    st.markdown(f"- **Business:** {research.business}")
                    if research.financials:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        key: (
                                                "—"
                                                if pd.isna(value)
                                                else (
                                                    f"{value:,.2f}"
                                                    if isinstance(value, (int, float))
                                                    and not isinstance(value, bool)
                                                    else str(value)
                                                )
                                            )
                                            for key, value in row.items()
                                        }
                                    for row in research.financials
                                ]
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.info(
                            "Financial figures are not shown because they have "
                            "not yet been verified against the issuer's RHP."
                        )
                    st.markdown(f"- **Peer context:** {research.peer_context}")
                    st.markdown(
                        f"- **Expansion / investment:** "
                        f"{research.expansion_and_investment}"
                    )
                    st.markdown(f"- **Order book / backlog:** {research.order_book}")
                    st.markdown("**Key disclosed risks**")
                    for risk in research.risks:
                        st.markdown(f"- {risk}")
                    st.info(f"**Preliminary view:** {research.view}")
                    st.caption(research.coverage_note)

                if status.lower() == "active" and pd.notna(total_subscription):
                    if total_subscription >= 1:
                        demand_read = (
                            f"Current aggregate demand is {total_subscription:.2f}× "
                            "at the NSE observation time; demand is not evidence "
                            "that the offer is fairly valued."
                        )
                    else:
                        demand_read = (
                            f"Current aggregate bids cover {total_subscription:.2f}× "
                            "the issue; demand may change before closing."
                        )
                else:
                    demand_read = (
                        "NSE has not published live bid data for this forthcoming issue."
                    )
                st.caption(
                    f"Live demand context: {demand_read} Subscription reflects "
                    "bids at the NSE observation time and is not a valuation "
                    "signal or personal allotment probability."
                )

                ratios_url = issue.get("ratios_url")
                rhp_url = issue.get("rhp_url")
                company_website = (
                    research.issuer_url if research else issue.get("company_website")
                )
                investor_relations_url = (
                    research.investor_relations_url
                    if research
                    else issue.get("investor_relations_url")
                )
                annual_report_url = issue.get("annual_report_url")
                document_cols = st.columns(5)
                if (
                    isinstance(company_website, str)
                    and company_website.startswith("https://")
                ):
                    document_cols[0].link_button(
                        "Issuer website",
                        company_website,
                        width="stretch",
                    )
                if (
                    isinstance(investor_relations_url, str)
                    and investor_relations_url.startswith("https://")
                ):
                    document_cols[1].link_button(
                        "Issuer investor relations",
                        investor_relations_url,
                        width="stretch",
                    )
                if (
                    isinstance(annual_report_url, str)
                    and annual_report_url.startswith("https://")
                ):
                    document_cols[2].link_button(
                        "Issuer annual report / results",
                        annual_report_url,
                        width="stretch",
                    )
                if isinstance(ratios_url, str) and ratios_url.startswith("https://"):
                    document_cols[3].link_button(
                        "NSE price / valuation basis",
                        ratios_url,
                        width="stretch",
                    )
                else:
                    document_cols[3].caption("NSE valuation document unavailable")
                rhp_url = research.rhp_url if research else rhp_url
                if isinstance(rhp_url, str) and rhp_url.startswith("https://"):
                    document_cols[4].link_button(
                        "NSE Red Herring Prospectus",
                        rhp_url,
                        width="stretch",
                    )

                st.markdown("#### Listing-price sensitivity (not a prediction)")
                scenarios = listing_price_scenarios(price_low, price_high)
                if scenarios:
                    scenario_cols = st.columns(3)
                    for column, (label, value) in zip(
                        scenario_cols, scenarios.items(), strict=False
                    ):
                        column.metric(label, f"₹{value:,.2f}")
                    st.caption(
                        "Sensitivity uses a 10% downside/upside around the published "
                        "price band; the midpoint is only a neutral reference. "
                        "These are not probability-weighted forecasts, and NSE does "
                        "not publish a reliable listing-price prediction."
                    )
                else:
                    st.caption("Price scenarios are unavailable until NSE publishes a price band.")

                st.markdown("#### Retail allotment context")
                allotment_proxy = retail_allotment_proxy(
                    float(retail_subscription)
                    if pd.notna(retail_subscription)
                    else None
                )
                if allotment_proxy is None:
                    if pd.notna(retail_subscription):
                        st.caption(
                            f"Retail category is at {retail_subscription:.2f}× "
                            "subscription and is not oversubscribed at this NSE "
                            "update. This does not guarantee allotment; demand "
                            "may change before the issue closes."
                        )
                    else:
                        st.caption(
                            "Retail category subscription is not available yet; "
                            "allotment context will appear when NSE publishes bids."
                        )
                else:
                    st.metric(
                        "Inverse retail-subscription proxy",
                        f"≈{allotment_proxy:.1f}%",
                    )
                    st.caption(
                        "This is 1 ÷ retail oversubscription, not a personal "
                        "allotment probability. Actual allotment depends on valid "
                        "applications, lot rules, category allocation, and the "
                        "registrar’s lottery where applicable."
                    )
                category_rows = issue.get("categories")
                if isinstance(category_rows, list) and category_rows:
                    st.markdown("##### NSE category-wise bid data")
                    st.dataframe(
                        pd.DataFrame(category_rows).rename(
                            columns={
                                "category": "Category",
                                "shares_offered": "Shares offered",
                                "shares_bid": "Shares bid",
                                "subscription": "Subscription (×)",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                if issue.get("details_error"):
                    st.warning(str(issue["details_error"]))
                st.caption(f"Sources: [NSE issue details]({issue['issue_info_url']}) · [NSE IPO feed]({NSE_IPO_URL})")

    st.warning(
        "For information only, not investment advice. IPO demand can change until "
        "the issue closes; listing gains and allotment are never guaranteed."
    )
    st.stop()
elif workspace == "Home / Market overview":
    st.subheader("Market pulse")
    st.caption(
        f"Live market snapshot · {datetime.now().strftime('%d %b %Y, %H:%M')} IST · "
        "index levels and available India/global indicators are shown below."
    )
    with st.spinner("Loading index trends, screened-stock performance, sector data, and current headlines..."):
        home_stocks, home_sectors, home_context, home_news = cached_home_market_data("NSE")

    if home_context.empty:
        st.warning("Market index data is currently unavailable. Check your connection and try again.")
    else:
        index_lookup = home_context.set_index("Market")
        index_names = ["NIFTY 50", "S&P 500", "NASDAQ Composite", "CBOE VIX", "Crude oil", "Gold", "USD/INR"]
        available_indices = [name for name in index_names if name in index_lookup.index]
        index_cards = st.columns(min(4, len(available_indices)))
        for position, name in enumerate(available_indices):
            row = index_lookup.loc[name]
            value = row.get("Latest")
            change = row.get("One-year return (%)")
            with index_cards[position % len(index_cards)]:
                st.metric(
                    name,
                    f"{float(value):,.2f}" if pd.notna(value) else "—",
                    f"{float(change):+.1f}% 1Y" if pd.notna(change) else None,
                )

        nifty = index_lookup.loc["NIFTY 50"] if "NIFTY 50" in index_lookup.index else None
        if nifty is not None:
            nifty_return = pd.to_numeric(nifty.get("One-year return (%)"), errors="coerce")
            nifty_trend = nifty.get("Trend vs 200-day average", "unavailable")
            if pd.notna(nifty_return):
                if nifty_return >= 10 and nifty_trend == "Bullish":
                    pulse = "Indian large-cap trend is positive"
                    explanation = f"NIFTY 50 is {nifty_trend.lower()} versus its 200-day average and up {nifty_return:.1f}% over one year."
                elif nifty_return <= -10 or nifty_trend == "Bearish":
                    pulse = "Indian market trend is under pressure"
                    explanation = f"NIFTY 50 is {nifty_trend.lower()} versus its 200-day average with a {nifty_return:+.1f}% one-year return."
                else:
                    pulse = "Indian market trend is mixed"
                    explanation = f"NIFTY 50 has returned {nifty_return:+.1f}% over one year and is {nifty_trend.lower()} versus its 200-day average."
                st.info(f"**{pulse}.** {explanation} Index direction is a snapshot, not a forecast.")

    st.subheader("Best and weakest performers")
    st.caption(
        "Ranked by one-year price return from the available four-page Screener.in FII/DII screen universe "
        "(not the full NSE/BSE listed universe). Synthetic/fallback prices are excluded."
    )
    if home_stocks.empty:
        st.warning("Insufficient real price-history data to rank stocks right now.")
    else:
        ranked_stocks = home_stocks.dropna(subset=["1-year return (%)"])
        st.markdown("#### Top five · strongest 1Y return")
        st.dataframe(
            ranked_stocks.sort_values("1-year return (%)", ascending=False).head(5),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Bottom five · weakest 1Y return")
        st.dataframe(
            ranked_stocks.sort_values("1-year return (%)", ascending=True).head(5),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Sector heat check")
    st.caption(
        "One-year average return by sector in the screened-stock sample. "
        "Green is positive, amber is near flat, and red is negative."
    )
    if home_sectors.empty:
        st.info("Sector classifications or return history are unavailable.")
    else:
        heatmap_data = home_sectors.sort_values(
            "Average 1-year return (%)", ascending=True
        ).copy()
        heatmap_colors = heatmap_data["Market read"].map(
            {
                "Doing well": "#16835d",
                "Stagnating": "#d79a16",
                "Down": "#c94c4c",
            }
        )
        sector_chart = go.Figure(
            go.Bar(
                x=heatmap_data["Average 1-year return (%)"],
                y=heatmap_data["Sector"],
                orientation="h",
                marker_color=heatmap_colors,
                text=heatmap_data["Average 1-year return (%)"].map(lambda value: f"{value:+.1f}%"),
                textposition="outside",
                cliponaxis=False,
                customdata=heatmap_data[["Market read", "Stocks tracked"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>Average 1Y return: %{x:+.1f}%"
                    "<br>Market read: %{customdata[0]}"
                    "<br>Stocks tracked: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        sector_chart.add_vline(x=0, line_color="#64748b", line_width=1)
        sector_chart.update_layout(
            height=max(300, 48 * len(heatmap_data) + 90),
            margin={"l": 12, "r": 55, "t": 12, "b": 45},
            xaxis_title="Average one-year return (%)",
            yaxis_title=None,
            template="plotly_white",
            showlegend=False,
            bargap=0.32,
            hoverlabel={"bgcolor": "white", "font_size": 13},
        )
        sector_chart.update_xaxes(
            zeroline=False,
            ticksuffix="%",
            gridcolor="#e9eef3",
        )
        sector_chart.update_yaxes(
            automargin=True,
            tickfont={"size": 13},
        )
        st.plotly_chart(sector_chart, width="stretch", config={"displayModeBar": False})

        doing_well, stagnating, down = st.columns(3)
        for column, (market_read, label) in zip(
            (doing_well, stagnating, down),
            (
                ("Doing well", "Positive"),
                ("Stagnating", "Near flat"),
                ("Down", "Negative"),
            ),
        ):
            group = home_sectors[home_sectors["Market read"] == market_read]
            names = ", ".join(group["Sector"].astype(str)) if not group.empty else "None in this sample"
            column.markdown(f"**{label}**")
            column.write(names)

        st.markdown("#### Sector context")
        for _, sector_row in home_sectors.iterrows():
            sector = str(sector_row["Sector"])
            market_read = str(sector_row["Market read"])
            return_value = float(sector_row["Average 1-year return (%)"])
            matches = home_news[home_news["Topic"] == f"Sector news · {sector}"]
            with st.expander(
                f"{sector} · {market_read} · {return_value:+.1f}% 1Y",
                expanded=False,
            ):
                st.write(
                    f"Average return across {int(sector_row['Stocks tracked'])} tracked stocks: "
                    f"{return_value:+.1f}% over one year. This measures price performance, not its cause. "
                    "Earnings, industry demand, commodity or climate exposure, regulation, and geopolitical events "
                    "are possible factors to investigate."
                )
                if matches.empty:
                    st.caption("No recent sector-specific headline was retrieved; no causal explanation is inferred.")
                else:
                    for _, story in matches.head(2).iterrows():
                        st.markdown(
                            f"- [{story['Headline']}]({story['URL']}) · {story['Source']} · {story['Published']}"
                        )

    st.subheader("Global context and latest headlines")
    st.caption("Headlines are links for context, not proof that an event caused a price move.")
    global_news = home_news[home_news["Topic"].isin(["Global markets and geopolitics", "India markets and policy"])]
    if global_news.empty:
        st.info("Current global and India market headlines are temporarily unavailable.")
    else:
        for topic, stories in global_news.groupby("Topic", sort=False):
            st.markdown(f"**{topic}**")
            for _, story in stories.iterrows():
                st.markdown(
                    f"- [{story['Headline']}]({story['URL']}) · {story['Source']} · {story['Published']}"
                )
    st.caption(
        "Coverage note: market breadth here is based on an FII/DII-focused stock screen, not every listed company. "
        "Sector stories and macro headlines are recent search results and do not establish causation; verify company filings "
        "and official sources before making investment decisions."
    )
    st.stop()

if symbol_input:
    try:
        if workspace == "Long-term ideas":
            st.subheader("Long-term small-cap ideas")
            st.caption(
                "Independent long-term screening engine. It uses available Screener.in fundamentals and Yahoo Finance price history; "
                "it does not reuse the market-ideas event-risk recommendations."
            )
            market_cap_preset = st.radio(
                "Choose maximum market cap",
                ["₹5,000 Cr", "₹10,000 Cr", "₹15,000 Cr", "₹25,000 Cr"],
                index=3,
                horizontal=True,
                help="Choose a larger cap to include more companies in the candidate screen.",
            )
            market_cap_limit = int(market_cap_preset.replace("₹", "").replace(",", "").replace(" Cr", ""))
            st.caption(f"Showing eligible companies with market cap up to ₹{market_cap_limit:,} Cr.")
            st.info(
                "Order-book strength and legal implications are not verified by the available data feeds. "
                "The confidence score measures screening evidence, not the probability of returns. "
                "Targets and timelines are illustrative scenarios, not a multibagger promise or personalized financial advice."
            )
            with st.spinner("Screening fundamentals and loading eligible price histories in parallel..."):
                long_term_ideas = cached_long_term_ideas(market_cap_limit, exchange)
            if long_term_ideas.empty:
                st.warning(
                    "No stocks currently met the available fundamental, price-history, moving-average, and 52-week-range filters. "
                    "This screen does not relax its criteria or substitute synthetic prices."
                )
            else:
                st.dataframe(long_term_ideas, width="stretch", hide_index=True)
                st.caption(
                    "Screen criteria: market cap within selected limit, ROCE ≥ 15%, positive FII and DII holding changes, "
                    "P/E between 0 and 60, at least 200 real daily closes, CMP within 15% of its 50-day average and 35% "
                    "of its 200-day average, and within the lower 85% of its 52-week range. “Entry candidate” is reserved "
                    "for stocks within 15% of both averages and the lower 40% of their 52-week range; all other rows are "
                    "watchlist-only pending a pullback. T1/T2/T3 are scenario levels of +15%/+30%/+50% from the "
                    "50/200-day average entry zone; timelines are tentative. The feed may provide fewer than 5 qualifying "
                    "stocks, so results are not padded with weak or synthetic candidates. Independently check balance-sheet "
                    "quality, order book, litigation, liquidity, and company filings before acting."
                )
            st.stop()

        ticker = normalize_stock_symbol(symbol_input, exchange=exchange)
        profile = fetch_stock_profile(symbol_input, exchange=exchange)
        data = cached_stock_history(ticker, period, exchange)
        data = data.sort_values("date").reset_index(drop=True)
        data = calculate_moving_average(data, window=20)
        data = calculate_moving_average(data, window=50)
        data = calculate_macd(data)
        data["rsi_14"] = calculate_rsi(data).reindex(data.index).fillna(50)
        returns = compute_daily_returns(data)
        summary = summarize_stock(data)

        top_stocks = fetch_screener_fii_dii_top_stocks(limit=10)
        if workspace == "Market ideas":
            st.subheader("Sector-ranked market ideas")
            st.caption("Top three eligible stocks per sector from the Screener.in FII/DII universe, ranked using current market conditions and up to 15 years of price history.")
            macro_risk = st.selectbox("Manual event-risk overlay", ["Neutral", "Elevated", "High"], help="Neutral ranks by trend and momentum, Elevated favors a more balanced risk profile, and High favors lower relative volatility and historical drawdown. This changes candidate rankings; headlines are not automatically scored.")
            if top_stocks.empty or "Symbol" not in top_stocks.columns:
                st.info("The market-ideas universe is temporarily unavailable.")
            else:
                with st.spinner("Loading sector classifications and long-term stock histories..."):
                    screen_pages = [cached_screener_page(limit=100, page=page) for page in range(1, 5)]
                    idea_candidates = pd.concat(screen_pages, ignore_index=True).drop_duplicates("Symbol")
                    idea_candidates["Sector"] = [
                        cached_stock_sector(symbol, exchange)
                        for symbol in idea_candidates["Symbol"].dropna().astype(str)
                    ]
                    idea_history = {
                        symbol.upper(): cached_stock_history(symbol, "15y", exchange)
                        for symbol in idea_candidates["Symbol"].dropna().astype(str)
                    }
                    market_context = cached_market_context()
                    market_ideas = build_market_ideas(
                        idea_candidates,
                        idea_history,
                        macro_risk=macro_risk,
                        market_context=market_context,
                    )
                if market_ideas.empty:
                    st.warning("No sectors had three or more classified stocks with real price history. Try again later or review the screen source availability.")
                else:
                    for sector in market_ideas["Sector"].drop_duplicates():
                        sector_rows = market_ideas[market_ideas["Sector"] == sector]
                        sector_rank = int(sector_rows["Sector rank"].iloc[0])
                        sector_trend = sector_rows["Sector trend"].iloc[0]
                        st.markdown(f"#### {sector_rank}. {sector} · {sector_trend} trend")
                        st.dataframe(
                            sector_rows.drop(columns=["Sector rank", "Sector", "Sector trend"]),
                            width="stretch",
                            hide_index=True,
                        )
                    if market_ideas.attrs.get("underfilled_sectors"):
                        st.warning("Omitted sectors with fewer than three eligible candidates: " + ", ".join(market_ideas.attrs["underfilled_sectors"]))
                    excluded = market_ideas.attrs.get("unclassified_count", 0) + market_ideas.attrs.get("rejected_history_count", 0)
                    if excluded:
                        st.caption(f"Excluded {excluded} candidates with missing sector classification or genuine price history.")
                    risk_guidance = {
                        "Neutral": "uses the standard trend-and-momentum ranking.",
                        "Elevated": "favors a balanced volatility and drawdown profile.",
                        "High": "favors lower-volatility, lower-drawdown stocks.",
                    }[macro_risk]
                    st.caption(f"{macro_risk} event-risk setting {risk_guidance} History coverage varies by listing date and data availability. Entry, targets, and stop loss are model levels, not personalized financial advice.")

                st.subheader("Indian and global market context")
                if market_context.empty:
                    st.info("Market indicators are temporarily unavailable.")
                else:
                    st.dataframe(market_context.drop(columns=["Ticker"]), width="stretch", hide_index=True)

                st.subheader("India policy and global developments")
                market_news = cached_market_news()
                if market_news.empty:
                    st.info("Headline feed is temporarily unavailable. Political and geopolitical event risk remains a manual input above.")
                else:
                    for topic, headlines in market_news.groupby("Topic", sort=False):
                        st.markdown(f"**{topic}**")
                        for _, headline in headlines.iterrows():
                            st.markdown(
                                f"- [{headline['Headline']}]({headline['URL']}) · {headline['Source']} · {headline['Published']}"
                            )
                st.caption("Market indicators affect the ranking mechanically. Headlines are linked for review and are not treated as verified causal signals or automatically scored sentiment.")

                st.subheader("Lower-priced sector ideas")
                if "CMP (Rs.)" in idea_candidates.columns:
                    penny_stocks = idea_candidates.dropna(subset=["CMP (Rs.)"]).nsmallest(30, "CMP (Rs.)")
                    penny_history = {
                        symbol.upper(): idea_history[symbol.upper()]
                        for symbol in penny_stocks["Symbol"].astype(str)
                        if symbol.upper() in idea_history
                    }
                    penny_ideas = build_market_ideas(
                        penny_stocks,
                        penny_history,
                        macro_risk=macro_risk,
                        market_context=market_context,
                    )
                    if penny_ideas.empty:
                        st.info("Fewer than three eligible lower-priced stocks were available in each classified sector.")
                    else:
                        for sector in penny_ideas["Sector"].drop_duplicates():
                            sector_rows = penny_ideas[penny_ideas["Sector"] == sector]
                            sector_rank = int(sector_rows["Sector rank"].iloc[0])
                            sector_trend = sector_rows["Sector trend"].iloc[0]
                            st.markdown(f"#### {sector_rank}. {sector} · {sector_trend} trend")
                            st.dataframe(
                                sector_rows.drop(columns=["Sector rank", "Sector", "Sector trend"]),
                                width="stretch",
                                hide_index=True,
                            )
                        st.caption("Lower-priced stocks can carry elevated liquidity, spread, and gap risks.")
            st.stop()

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

                **Company / exchange website**: {profile.get('website') or 'Exchange quote page'}
                """
            )

            if profile.get("website"):
                st.markdown(f"[Open company / exchange website]({profile.get('website')})")

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
            recommendation_history = cached_stock_history(ticker, "15y", exchange)
            recommendation = generate_recommendations(recommendation_history, profile=profile, fii_df=fii_df, bulk_df=bulk_df)
            st.caption("Recommendation signals use up to 15 years of available price history; the chart range above does not change this lookback.")

            st.subheader(f"{profile.get('longName') or profile.get('shortName') or ticker}")

            action = str(recommendation.get("action", "HOLD")).upper()
            action_class = {"BUY": "action-buy", "SELL": "action-sell"}.get(action, "action-hold")
            entry_price = recommendation.get("entry_price")
            partial_entry = recommendation.get("entry_partial_price")
            targets = recommendation.get("target_prices") or []
            if entry_price is not None and (len(targets) < 3 or any(level is None for level in targets[:3])):
                legacy_target = recommendation.get("target_price")
                target_2 = legacy_target or round(entry_price * 1.08, 2)
                targets = [round(entry_price * 1.04, 2), target_2, round(entry_price * 1.12, 2)]
            stop_loss = recommendation.get("stop_loss")
            risk_per_share = entry_price - stop_loss if entry_price is not None and stop_loss is not None else 0.0
            reward_per_share = targets[2] - entry_price if entry_price is not None and len(targets) >= 3 else 0.0
            risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
            holding_style = "Long-term analysis"

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
                st.warning("Entry, target, and stop-loss levels are unavailable because genuine 15-year market data could not be loaded.")

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
                f"<div class='trade-note'><strong>One-year return:</strong> {summary_change_text} &nbsp; | &nbsp; <strong>RSI:</strong> {summary_rsi_text} &nbsp; | &nbsp; <strong>Institutional flow:</strong> {summary_flow} &nbsp; | &nbsp; <strong>Risk/reward:</strong> 1:{risk_reward:.2f}</div>"
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
