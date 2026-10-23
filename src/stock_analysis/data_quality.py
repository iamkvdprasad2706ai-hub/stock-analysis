"""Quality checks for stock price history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import pandas as pd

DataQualityStatus = Literal["fresh", "stale", "synthetic", "unverified", "empty", "malformed"]


@dataclass(frozen=True)
class StockDataQuality:
    """A compact quality report for a stock price history."""

    is_real: bool
    is_synthetic: bool
    source: str
    latest_observation_date: date | None
    age_days: int | None
    row_count: int
    status: DataQualityStatus
    message: str


def inspect_stock_data_quality(
    history: pd.DataFrame | None,
    *,
    as_of: date | datetime | pd.Timestamp | None = None,
    max_age_days: int = 3,
) -> StockDataQuality:
    """Assess price-history provenance, recency, and basic structural validity.

    Histories must contain date and close columns (case-insensitive). Yahoo
    Finance's ``Date``/``Close`` names and this project's normalized names are
    both supported. Source is read from DataFrame.attrs and never inferred from
    the presence of price rows.
    """
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative.")

    attrs = getattr(history, "attrs", {}) if isinstance(history, pd.DataFrame) else {}
    source_value = attrs.get("source", attrs.get("data_source", ""))
    source = str(source_value).strip().lower().replace("-", "_").replace(" ", "_")
    synthetic = bool(attrs.get("synthetic", False)) or source in {
        "synthetic",
        "synthetic_fallback",
    }
    real = not synthetic and source in {"yahoo", "yahoo_finance", "real"}
    source = "synthetic_fallback" if synthetic else ("yahoo_finance" if real else "unknown")
    row_count = len(history) if isinstance(history, pd.DataFrame) else 0

    def report(
        status: DataQualityStatus,
        message: str,
        latest_date: date | None = None,
        age: int | None = None,
    ) -> StockDataQuality:
        return StockDataQuality(
            is_real=real,
            is_synthetic=synthetic,
            source=source,
            latest_observation_date=latest_date,
            age_days=age,
            row_count=row_count,
            status=status,
            message=message,
        )

    if not isinstance(history, pd.DataFrame):
        return report("malformed", "History must be provided as a pandas DataFrame.")
    if history.empty:
        return report("empty", "No price rows are available; retry the data fetch or verify the stock symbol.")

    columns = {str(column).strip().lower(): column for column in history.columns}
    missing = [name for name in ("date", "close") if name not in columns]
    if missing:
        return report(
            "malformed",
            f"History is missing required column(s): {', '.join(missing)}; provide date and close values.",
        )

    dates = pd.to_datetime(history[columns["date"]], errors="coerce", utc=True)
    closes = pd.to_numeric(history[columns["close"]], errors="coerce")
    valid = dates.notna() & closes.gt(0)
    if not valid.any():
        return report(
            "malformed",
            "History contains no valid observations; check that dates parse and closing prices are positive numbers.",
        )

    latest_timestamp = dates.loc[valid].max().tz_convert(None).normalize()
    latest_date = latest_timestamp.date()
    reference = pd.Timestamp(as_of if as_of is not None else datetime.now().astimezone())
    if reference.tzinfo is not None:
        reference = reference.tz_convert("UTC").tz_localize(None)
    age = (reference.normalize().date() - latest_date).days
    if age < 0:
        return report(
            "malformed",
            f"Latest observation ({latest_date.isoformat()}) is in the future; check the history's date values.",
            latest_date,
            age,
        )

    if synthetic:
        return report(
            "synthetic",
            "Synthetic fallback prices are being used; treat them as illustrative only and check Yahoo Finance connectivity and the symbol.",
            latest_date,
            age,
        )
    if age > max_age_days:
        provenance = "Yahoo Finance data" if real else "Unverified-source data"
        return report(
            "stale",
            f"{provenance} is {age} days old (latest {latest_date.isoformat()}); refresh the history before making decisions.",
            latest_date,
            age,
        )
    if not real:
        return report(
            "unverified",
            f"History is current through {latest_date.isoformat()}, but its source is unmarked; verify provenance before relying on it.",
            latest_date,
            age,
        )
    return report(
        "fresh",
        f"Yahoo Finance price data is current through {latest_date.isoformat()} ({age} days old).",
        latest_date,
        age,
    )
