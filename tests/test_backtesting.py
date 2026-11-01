import numpy as np
import pandas as pd
import pytest

from stock_analysis.backtesting import BacktestResult, run_backtest


def test_signal_is_executed_one_bar_later():
    prices = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5, 6], dtype=float)

    result = run_backtest(prices, fast_window=2, slow_window=3)

    assert result.equity_curve["signal"].iloc[6] == 1
    assert result.equity_curve["position"].iloc[6] == 0
    assert result.equity_curve["position"].iloc[7] == 1
    assert result.equity_curve["strategy_returns"].iloc[7] == 0
    assert result.equity_curve["strategy_returns"].iloc[8] == pytest.approx(1 / 4)


def test_fees_and_slippage_are_charged_on_entry_and_exit():
    prices = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2], dtype=float)

    result = run_backtest(prices, fast_window=2, slow_window=3, fee_bps=10, slippage_bps=20)
    curve = result.equity_curve

    assert curve["fees"].sum() > 0
    assert curve["slippage"].sum() == pytest.approx(curve["fees"].sum() * 2)
    assert result.summary["total_costs"] == pytest.approx(
        result.summary["fees_paid"] + result.summary["slippage_cost"]
    )
    assert result.summary["trade_count"] == 1
    assert (curve["costs"] > 0).sum() == 2


def test_benchmark_dataframe_is_aligned_and_return_is_reported():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5, 6], index=index)
    benchmark = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=index)

    result = run_backtest(prices, fast_window=2, slow_window=3, benchmark=benchmark)

    assert result.summary["benchmark_return"] == pytest.approx(0.1)
    assert "benchmark_equity" in result.equity_curve
    assert result.equity_curve["benchmark_equity"].iloc[-1] == pytest.approx(1.1)
    assert result.equity_curve["benchmark_equity"].iloc[0] == pytest.approx(1.0)


def test_result_includes_expected_metrics_and_zero_risk_free_assumption():
    prices = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5, 6], dtype=float)

    result = run_backtest(prices, fast_window=2, slow_window=3)

    assert isinstance(result, BacktestResult)
    assert {
        "total_return",
        "benchmark_return",
        "max_drawdown",
        "annualized_volatility",
        "sharpe_ratio",
        "trade_count",
        "fees_paid",
        "slippage_cost",
        "total_costs",
    } <= result.summary.keys()
    assert result.summary["benchmark_return"] is None
    assert result.equity_curve["equity"].iloc[0] == pytest.approx(1.0)
    assert result.metrics is result.summary


@pytest.mark.parametrize(
    ("fast_window", "slow_window"),
    [(0, 3), (-1, 3), (True, 3), (3, 3), (4, 3)],
)
def test_invalid_windows_are_rejected(fast_window, slow_window):
    with pytest.raises(ValueError):
        run_backtest(pd.Series([1, 2, 3, 4, 5]), fast_window, slow_window)


def test_insufficient_data_is_reported():
    with pytest.raises(ValueError, match="Insufficient data"):
        run_backtest(pd.Series([1.0, 2.0, 3.0]), fast_window=2, slow_window=3)


@pytest.mark.parametrize(
    "prices",
    [
        pd.Series([], dtype=float),
        pd.Series([1.0, 0.0, 2.0]),
        pd.Series([1.0, np.nan, 2.0]),
        pd.Series([1.0, np.inf, 2.0]),
    ],
)
def test_invalid_price_inputs_are_rejected(prices):
    with pytest.raises(ValueError):
        run_backtest(prices, fast_window=1, slow_window=2)


def test_duplicate_price_dates_are_rejected():
    prices = pd.Series([1.0, 2.0, 3.0], index=[1, 1, 2])

    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(prices, fast_window=1, slow_window=2)


def test_synthetic_flag_is_rejected_for_prices_and_benchmark():
    prices = pd.Series([1.0, 2.0, 3.0])
    prices.attrs["synthetic"] = True
    valid = pd.Series([1.0, 2.0, 3.0])
    synthetic_benchmark = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    synthetic_benchmark.attrs["synthetic"] = True

    with pytest.raises(ValueError, match="synthetic"):
        run_backtest(prices, fast_window=1, slow_window=2)
    with pytest.raises(ValueError, match="synthetic"):
        run_backtest(valid, fast_window=1, slow_window=2, benchmark=synthetic_benchmark)


def test_price_dataframe_with_date_and_close_columns_is_supported():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "close": [5.0, 4.0, 3.0, 4.0, 5.0],
        }
    )

    result = run_backtest(prices, fast_window=1, slow_window=2)

    assert result.equity_curve.index.equals(pd.DatetimeIndex(prices["date"]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fee_bps": -1},
        {"slippage_bps": np.inf},
        {"fee_bps": 10_000},
        {"risk_free_rate": -1},
        {"periods_per_year": 0},
    ],
)
def test_invalid_cost_or_annualization_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        run_backtest(pd.Series([1.0, 2.0, 3.0]), fast_window=1, slow_window=2, **kwargs)
