import numpy as np
import pandas as pd
import pytest

import stock_analysis.data as stock_data
from stock_analysis.analysis import _historical_metrics, analyze_institutional_trend, build_market_ideas, generate_recommendations, summarize_stock
from stock_analysis.data import fetch_bulk_deals, fetch_fii_data, fetch_market_context, fetch_market_news, load_stock_data, normalize_stock_symbol


def test_normalize_stock_symbol_converts_to_nse_format():
    assert normalize_stock_symbol("reliance") == "RELIANCE.NS"
    assert normalize_stock_symbol("TCS") == "TCS.NS"
    assert normalize_stock_symbol("RELIANCE.NS") == "RELIANCE.NS"


def test_normalize_stock_symbol_supports_bse_and_explicit_exchange_suffixes():
    assert normalize_stock_symbol("reliance", exchange="BSE") == "RELIANCE.BO"
    assert normalize_stock_symbol("TCS.BO") == "TCS.BO"
    assert normalize_stock_symbol("TCS.NS", exchange="BSE") == "TCS.NS"


def test_load_stock_data_maps_fifteen_year_lookback_to_start_date(monkeypatch):
    calls = {}

    class FakeTicker:
        def history(self, **kwargs):
            calls.update(kwargs)
            index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Date")
            return pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [102.0, 103.0],
                    "Low": [99.0, 100.0],
                    "Close": [101.0, 102.0],
                    "Volume": [1000, 1100],
                },
                index=index,
            )

    monkeypatch.setattr(stock_data.yf, "Ticker", lambda ticker: FakeTicker())
    history = load_stock_data("TCS", period="15y")
    assert "period" not in calls
    assert calls["start"] <= pd.Timestamp.today() - pd.DateOffset(years=15)
    assert len(history) == 2
    assert not history.attrs.get("synthetic", False)


def test_historical_metrics_measure_fifteen_year_cagr():
    dates = pd.date_range("2010-01-01", "2025-01-01", freq="B")
    history = pd.DataFrame({"date": dates, "close": np.geomspace(100, 100 * 1.1**15, len(dates))})
    metrics = _historical_metrics(history)
    assert metrics["history_years"] >= 14.9
    assert metrics["cagr"] == pytest.approx(10.0, abs=0.1)


def test_fetch_market_context_returns_index_trend_signals(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            close = [100.0, 110.0, 120.0] if self.ticker != "^VIX" else [30.0, 25.0, 20.0]
            return pd.DataFrame({"Close": close})

    monkeypatch.setattr(stock_data.yf, "Ticker", FakeTicker)
    context = fetch_market_context()
    assert {"NIFTY 50", "S&P 500", "NASDAQ Composite"}.issubset(set(context["Market"]))
    assert context.loc[context["Market"] == "NIFTY 50", "Trend vs 200-day average"].iloc[0] == "Bullish"


def test_fetch_market_news_keeps_headline_sources_and_links(monkeypatch):
    class FakeResponse:
        content = b"""<rss><channel><item><title>India policy update</title><link>https://example.com/story</link><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate><source>Example News</source></item></channel></rss>"""

        def raise_for_status(self):
            pass

    monkeypatch.setattr(stock_data.requests, "get", lambda *args, **kwargs: FakeResponse())
    news = fetch_market_news(limit=1)
    assert len(news) == 2
    assert news.iloc[0]["Source"] == "Example News"
    assert news.iloc[0]["URL"] == "https://example.com/story"


def test_summarize_stock_handles_basic_dataframe():
    df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 105.0],
            "volume": [1000, 1200, 1100, 1400],
        }
    )
    summary = summarize_stock(df)
    assert summary["latest_close"] == 105.0
    assert summary["latest_volume"] == 1400
    assert "percent_change" in summary


def test_fetch_fii_data_returns_investment_rows():
    fii_df = fetch_fii_data(limit=2)
    assert isinstance(fii_df, pd.DataFrame)
    assert not fii_df.empty
    assert {"category", "netValue"}.issubset(set(fii_df.columns))


def test_fetch_bulk_deals_returns_market_rows():
    bulk = fetch_bulk_deals(limit=2)
    assert isinstance(bulk, pd.DataFrame)
    assert not bulk.empty
    assert {"Date", "Symbol", "Buy/Sell"}.issubset(set(bulk.columns))


def test_generate_recommendations_returns_action_and_reasons():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "open": [100 + i * 0.5 for i in range(30)],
            "high": [110 + i * 0.5 for i in range(30)],
            "low": [90 + i * 0.5 for i in range(30)],
            "close": [100 + i * 0.8 for i in range(30)],
            "volume": [1000 + i * 100 for i in range(30)],
        }
    )
    rec = generate_recommendations(df, profile={"sector": "Technology"})
    assert rec["action"] in {"BUY", "HOLD", "SELL"}
    assert isinstance(rec["reasons"], list)
    assert rec["confidence"] >= 0
    assert rec["confidence"] <= 100


def test_analyze_institutional_trend_projects_direction():
    flow = pd.DataFrame(
        {
            "category": ["FII/FPI"] * 6,
            "netValue": [-100, -80, -60, 20, 40, 60],
        }
    )
    trend = analyze_institutional_trend(flow, recent_window=3)
    assert trend.iloc[0]["direction"] == "Increasing buying"
    assert trend.iloc[0]["projected_next"] > trend.iloc[0]["recent_average"]


def test_build_market_ideas_returns_trade_levels():
    dates = pd.date_range("2010-01-01", periods=16 * 252, freq="B")
    screen_rows = []
    histories = {}
    for sector, symbol_prefix, annual_growth, count in [
        ("Technology", "TECH", 0.20, 3),
        ("Consumer Staples", "STAP", -0.05, 3),
        ("Industrials", "IND", 0.05, 2),
    ]:
        close = 100 * (1 + annual_growth) ** (np.arange(len(dates)) / 252)
        for stock_number in range(count):
            symbol = f"{symbol_prefix}{stock_number}"
            screen_rows.append(
                {"Symbol": symbol, "Company": symbol, "Sector": sector, "FII change (%)": 1.0, "DII change (%)": 1.5}
            )
            histories[symbol] = pd.DataFrame({"date": dates, "close": close, "volume": [1000] * len(dates)})

    ideas = build_market_ideas(pd.DataFrame(screen_rows), histories)
    assert len(ideas) == 6
    assert ideas.groupby("Sector").size().to_dict() == {"Consumer Staples": 3, "Technology": 3}
    assert ideas["Sector"].iloc[0] == "Technology"
    assert "Industrials (2/3)" in ideas.attrs["underfilled_sectors"]
    assert "Annualized volatility (%)" in ideas.columns
    assert ideas.iloc[0]["View"] in {"BUY", "HOLD", "SELL"}
    assert ideas.iloc[0]["Stop loss"] < ideas.iloc[0]["Entry"] < ideas.iloc[0]["Target 1"]

    bearish_context = pd.DataFrame(
        [
            {"Market": "NIFTY 50", "One-year return (%)": -20.0, "Trend vs 200-day average": "Bearish"},
            {"Market": "CBOE VIX", "Latest": 30.0},
            {"Market": "Crude oil", "One-year return (%)": 25.0},
            {"Market": "USD/INR", "One-year return (%)": 6.0},
        ]
    )
    adjusted = build_market_ideas(pd.DataFrame(screen_rows), histories, market_context=bearish_context)
    strong_stock_score = ideas.loc[ideas["Symbol"] == "TECH0", "Score"].iloc[0]
    adjusted_score = adjusted.loc[adjusted["Symbol"] == "TECH0", "Score"].iloc[0]
    assert adjusted_score < strong_stock_score


def test_generate_recommendations_refuses_synthetic_fallback_prices():
    history = pd.DataFrame({"close": [100.0, 105.0], "volume": [1000, 1200]})
    history.attrs["synthetic"] = True
    recommendation = generate_recommendations(history)
    assert recommendation["action"] == "HOLD"
    assert recommendation["entry_price"] is None
    assert "Real historical price data is unavailable" in recommendation["reasons"][0]
