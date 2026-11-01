from datetime import date

import pandas as pd

from stock_analysis.data_quality import inspect_stock_data_quality


def _history(observation_date: str) -> pd.DataFrame:
    history = pd.DataFrame({"date": [observation_date], "close": [123.45]})
    history.attrs["source"] = "yahoo_finance"
    return history


def test_inspect_stock_data_quality_reports_recent_real_history():
    report = inspect_stock_data_quality(_history("2026-10-17"), as_of=date(2026, 10, 18))

    assert report.is_real
    assert not report.is_synthetic
    assert report.source == "yahoo_finance"
    assert report.latest_observation_date == date(2026, 10, 17)
    assert report.age_days == 1
    assert report.row_count == 1
    assert report.status == "fresh"
    assert "current" in report.message


def test_inspect_stock_data_quality_marks_old_history_stale():
    report = inspect_stock_data_quality(_history("2026-10-10"), as_of=date(2026, 10, 18))

    assert report.status == "stale"
    assert report.age_days == 8
    assert "refresh" in report.message


def test_inspect_stock_data_quality_identifies_synthetic_fallback():
    history = _history("2026-10-17")
    history.attrs["synthetic"] = True
    history.attrs["source"] = "synthetic_fallback"

    report = inspect_stock_data_quality(history, as_of=date(2026, 10, 18))

    assert not report.is_real
    assert report.is_synthetic
    assert report.source == "synthetic_fallback"
    assert report.status == "synthetic"
    assert "illustrative only" in report.message


def test_inspect_stock_data_quality_handles_empty_history():
    report = inspect_stock_data_quality(pd.DataFrame(), as_of=date(2026, 10, 18))

    assert report.status == "empty"
    assert report.row_count == 0
    assert report.latest_observation_date is None
    assert report.age_days is None
    assert "retry" in report.message


def test_inspect_stock_data_quality_reports_missing_columns():
    report = inspect_stock_data_quality(pd.DataFrame({"close": [123.45]}))

    assert report.status == "malformed"
    assert report.row_count == 1
    assert report.latest_observation_date is None
    assert report.age_days is None
    assert "date" in report.message


def test_inspect_stock_data_quality_rejects_unusable_observations():
    report = inspect_stock_data_quality(pd.DataFrame({"date": ["not-a-date"], "close": ["bad"]}))

    assert report.status == "malformed"
    assert "no valid observations" in report.message
