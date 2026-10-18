"""Small, deterministic backtesting utilities for long-only strategies.

Returns are close-to-close. A signal observed at a close can only affect the
position from the following bar onward. Sharpe uses a zero annual risk-free
rate by default; a different annual rate can be supplied to :func:`run_backtest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Backtest time series and aggregate performance metrics.

    ``summary`` reports returns, volatility, and drawdown as decimal fractions.
    ``fees_paid``, ``slippage_cost``, and ``total_costs`` are measured in units
    of the initial portfolio value (which is 1.0).
    """

    equity_curve: pd.DataFrame
    summary: dict[str, float | int | None]

    @property
    def metrics(self) -> dict[str, float | int | None]:
        """Alias for ``summary`` for callers that prefer metrics terminology."""
        return self.summary


def _as_prices(data: pd.Series | pd.DataFrame, label: str) -> pd.Series:
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError(f"{label} must be a pandas Series or DataFrame of prices.")
    if data.attrs.get("synthetic", False):
        raise ValueError(f"{label} is marked synthetic and cannot be used for backtesting.")

    if isinstance(data, pd.DataFrame):
        columns = {str(column).strip().lower(): column for column in data.columns}
        price_column = next(
            (columns[key] for key in ("close", "adj close", "adjusted close") if key in columns),
            None,
        )
        if price_column is None and len(data.columns) == 1:
            price_column = data.columns[0]
        if price_column is None:
            raise ValueError(
                f"{label} DataFrame must contain a close/adjusted-close column "
                "or exactly one price column."
            )

        date_column = next(
            (columns[key] for key in ("date", "datetime") if key in columns),
            None,
        )
        series = data[price_column].copy()
        if date_column is not None:
            try:
                series.index = pd.to_datetime(data[date_column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} contains invalid dates.") from exc
    else:
        series = data.copy()

    if series.empty:
        raise ValueError(f"{label} must contain at least one price.")
    if series.index.hasnans:
        raise ValueError(f"{label} index must not contain missing values.")
    if series.index.has_duplicates:
        raise ValueError(f"{label} index must not contain duplicate dates.")

    try:
        series = series.sort_index()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} index must be sortable in chronological order.") from exc
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain only finite numeric prices.")
    if (values <= 0).any():
        raise ValueError(f"{label} prices must be greater than zero.")
    values.name = "close"
    return values.astype(float)


def _validate_window(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def run_backtest(
    prices: pd.Series | pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    benchmark: pd.Series | pd.DataFrame | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Backtest a long-only moving-average crossover strategy.

    The position is 1 when the fast simple moving average is above the slow
    average and 0 otherwise. Signals are shifted by one bar, so a signal
    calculated from a closing price cannot earn that same bar's return.
    Fees and slippage are each charged in basis points on traded portfolio
    value at every entry and exit. The Sharpe ratio uses ``risk_free_rate``
    as an annual decimal rate (default 0.0); returns are close-to-close and
    annualized using ``periods_per_year`` (default 252).

    ``trade_count`` counts entries into a long position, including a position
    still open at the end of the test. The initial portfolio value is 1.0.
    """
    fast_window = _validate_window("fast_window", fast_window)
    slow_window = _validate_window("slow_window", slow_window)
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")
    periods_per_year = _validate_window("periods_per_year", periods_per_year)

    try:
        fee_bps = float(fee_bps)
        slippage_bps = float(slippage_bps)
        risk_free_rate = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fees, slippage, and risk_free_rate must be finite numbers.") from exc
    if not np.isfinite([fee_bps, slippage_bps, risk_free_rate]).all():
        raise ValueError("Fees, slippage, and risk_free_rate must be finite numbers.")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps and slippage_bps must be non-negative.")
    if fee_bps + slippage_bps >= 10_000:
        raise ValueError("Combined fee and slippage must be less than 10,000 bps.")
    if risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1.")

    close = _as_prices(prices, "prices")
    if len(close) < slow_window + 1:
        raise ValueError(
            f"Insufficient data: at least slow_window + 1 ({slow_window + 1}) "
            "prices are required."
        )

    fast_ma = close.rolling(fast_window, min_periods=fast_window).mean()
    slow_ma = close.rolling(slow_window, min_periods=slow_window).mean()
    signal = (fast_ma > slow_ma).astype(int)
    position = signal.shift(1, fill_value=0).astype(int)

    gross_returns = close.pct_change(fill_method=None).fillna(0.0)
    turnover = position.diff().abs().fillna(position.iloc[0]).astype(float)
    active_position = position.shift(1, fill_value=0)
    invested_returns = gross_returns * active_position
    prior_equity = (1.0 + invested_returns).cumprod().shift(1, fill_value=1.0)
    pre_cost_equity = prior_equity * (1.0 + invested_returns)
    fees = pre_cost_equity * turnover * (fee_bps / 10_000.0)
    slippage = pre_cost_equity * turnover * (slippage_bps / 10_000.0)
    cost_fraction = turnover * ((fee_bps + slippage_bps) / 10_000.0)
    strategy_returns = (1.0 + invested_returns) * (1.0 - cost_fraction) - 1.0
    equity = (1.0 + strategy_returns).cumprod()

    curve = pd.DataFrame(
        {
            "close": close,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "signal": signal,
            "position": position,
            "gross_returns": invested_returns,
            "strategy_returns": strategy_returns,
            "fees": fees,
            "slippage": slippage,
            "costs": fees + slippage,
            "equity": equity,
        },
        index=close.index,
    )

    benchmark_return: float | None = None
    if benchmark is not None:
        benchmark_prices = _as_prices(benchmark, "benchmark")
        aligned_benchmark = benchmark_prices.reindex(close.index).dropna()
        if not aligned_benchmark.empty:
            benchmark_return = float(aligned_benchmark.iloc[-1] / aligned_benchmark.iloc[0] - 1.0)
            benchmark_curve = benchmark_prices.reindex(close.index).ffill()
            benchmark_curve = benchmark_curve / aligned_benchmark.iloc[0]
            benchmark_curve.loc[benchmark_curve.index < aligned_benchmark.index[0]] = np.nan
            curve["benchmark_equity"] = benchmark_curve

    returns_std = float(strategy_returns.std(ddof=1)) if len(strategy_returns) > 1 else 0.0
    annualized_volatility = returns_std * np.sqrt(periods_per_year)
    periodic_risk_free = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    sharpe_ratio = (
        float((strategy_returns - periodic_risk_free).mean() / returns_std * np.sqrt(periods_per_year))
        if returns_std > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    summary: dict[str, float | int | None] = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "benchmark_return": benchmark_return,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": sharpe_ratio,
        "trade_count": int((position.diff().fillna(position.iloc[0]) > 0).sum()),
        "fees_paid": float(fees.sum()),
        "slippage_cost": float(slippage.sum()),
        "total_costs": float((fees + slippage).sum()),
    }
    return BacktestResult(equity_curve=curve, summary=summary)
