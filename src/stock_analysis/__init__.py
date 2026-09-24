"""Stock analysis package."""

from .analysis import analyze_institutional_trend, build_market_ideas, calculate_macd, calculate_moving_average, compute_daily_returns, generate_recommendations, summarize_stock
from .data import fetch_bulk_deals, fetch_fii_data, load_stock_data, normalize_stock_symbol

__all__ = [
    "load_stock_data",
    "normalize_stock_symbol",
    "fetch_fii_data",
    "fetch_bulk_deals",
    "calculate_moving_average",
    "calculate_macd",
    "analyze_institutional_trend",
    "build_market_ideas",
    "compute_daily_returns",
    "generate_recommendations",
    "summarize_stock",
]
