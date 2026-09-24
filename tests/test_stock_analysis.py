import pandas as pd

from stock_analysis.analysis import generate_recommendations, summarize_stock
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
