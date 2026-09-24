import pandas as pd

from stock_analysis.analysis import analyze_institutional_trend, build_market_ideas, generate_recommendations, summarize_stock
from stock_analysis.data import fetch_bulk_deals, fetch_fii_data, normalize_stock_symbol


def test_normalize_stock_symbol_converts_to_nse_format():
    assert normalize_stock_symbol("reliance") == "RELIANCE.NS"
    assert normalize_stock_symbol("TCS") == "TCS.NS"
    assert normalize_stock_symbol("RELIANCE.NS") == "RELIANCE.NS"


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
    screen = pd.DataFrame(
        [{"Symbol": "TEST", "Company": "Test Co", "FII change (%)": 1.0, "DII change (%)": 1.5}]
    )
    history = {"TEST": pd.DataFrame({"close": [100 + i for i in range(60)], "volume": [1000] * 40 + [2500] * 20})}
    ideas = build_market_ideas(screen, history)
    assert len(ideas) == 1
    assert ideas.iloc[0]["View"] in {"BUY", "HOLD", "SELL"}
    assert ideas.iloc[0]["Stop loss"] < ideas.iloc[0]["Entry"] < ideas.iloc[0]["Target 1"]
