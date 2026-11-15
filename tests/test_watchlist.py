import pandas as pd
import pytest

from stock_analysis.analysis import generate_recommendations
from stock_analysis.browser_storage import parse_watchlist_payload
from stock_analysis.watchlist import build_watchlist_analysis_link, build_watchlist_table


def test_watchlist_analysis_link_preserves_symbol_and_exchange():
    link = build_watchlist_analysis_link("BRK.B", "US")

    assert link == "?analysis_symbol=BRK.B&analysis_exchange=US"


def test_watchlist_payload_restores_valid_stock_entries():
    assert parse_watchlist_payload('[{"symbol":"TCS.NS","exchange":"NSE"}]') == [
        {"symbol": "TCS.NS", "exchange": "NSE"}
    ]
    assert parse_watchlist_payload(None) == []


def test_watchlist_payload_rejects_invalid_saved_data():
    with pytest.raises(ValueError, match="unsupported exchange"):
        parse_watchlist_payload('[{"symbol":"TCS.NS","exchange":"OTHER"}]')


def test_watchlist_includes_closes_recommendation_levels_and_sentiment():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]),
            "close": [105.0, 100.0, 102.0],
            "volume": [1200, 1000, 1100],
        }
    )
    stock = {"symbol": "TEST.NS", "exchange": "NSE"}

    table = build_watchlist_table([stock], {"NSE:TEST.NS": history})
    recommendation = generate_recommendations(history.sort_values("date"))
    row = table.iloc[0]

    assert row["Previous close"] == 102.0
    assert row["Last close"] == 105.0
    assert row["View"] == recommendation["action"]
    assert row["Market sentiment"] == {
        "BUY": "Bullish",
        "HOLD": "Neutral",
        "SELL": "Bearish",
    }[recommendation["action"]]
    assert row["Target 1"] == recommendation["target_prices"][0]
    assert row["Target 2"] == recommendation["target_prices"][1]
    assert row["Target 3"] == recommendation["target_prices"][2]
    assert row["Stop loss"] == recommendation["stop_loss"]
    assert row["Signal date"] == "2025-01-03"


def test_watchlist_does_not_show_synthetic_prices_or_model_levels():
    history = pd.DataFrame({"close": [100.0, 105.0]})
    history.attrs["synthetic"] = True

    table = build_watchlist_table(
        [{"symbol": "TEST", "exchange": "US"}],
        {"US:TEST": history},
    )

    row = table.iloc[0]
    assert row["Previous close"] is None
    assert row["Last close"] is None
    assert row["Target 1"] is None
    assert row["Stop loss"] is None
    assert row["Market sentiment"] == "Unavailable"
    assert row["Data status"] == "Price data unavailable"
