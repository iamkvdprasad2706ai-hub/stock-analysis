import pytest
import requests
import pandas as pd

import stock_analysis.ipo_analysis as ipo_analysis
from stock_analysis.ipo_analysis import (
    _issue_record,
    IPOFeedError,
    fetch_ipo_issues,
    listing_price_scenarios,
    retail_allotment_proxy,
)


def test_issue_record_extracts_official_terms_and_retail_subscription():
    issue = {
        "companyName": "Example Limited",
        "symbol": "EXAMPLE",
        "series": "EQ",
        "status": "Active",
        "issuePrice": "Rs.665 to Rs.700",
        "issueSize": "1000000",
        "issueStartDate": "15-Oct-2026",
        "issueEndDate": "19-Oct-2026",
        "noOfTime": "2.5",
    }
    detail = {
        "companyName": "EXAMPLE",
        "issueInfo": {
            "dataList": [
                {"title": "Issue Size", "value": "Fresh issue of equity shares"},
                {"title": "Bid Lot", "value": "21 Equity Shares"},
                {"title": "Sector", "value": "Technology"},
                {"title": "Order Book", "value": "₹1,200 crore"},
                {"title": "Red Herring Prospectus", "value": "https://example.com/rhp.zip"},
                {"title": "Ratios / Basis of Issue Price", "value": "https://example.com/ratios.zip"},
            ]
        },
        "activeCat": {
            "dataList": [
                {
                    "category": "Retail Individual Investors(RIIs)",
                    "noOfShareOffered": "10000",
                    "noOfSharesBid": "25000",
                    "noOfTotalMeant": "2.5",
                }
            ]
        },
    }

    result = _issue_record(issue, detail)

    assert result["company"] == "Example Limited"
    assert (result["price_low"], result["price_high"]) == (665, 700)
    assert result["total_subscription"] == 2.5
    assert result["retail_subscription"] == 2.5
    assert result["retail_shares_offered"] == 10_000
    assert result["retail_shares_bid"] == 25_000
    assert result["lot_size"] == "21 Equity Shares"
    assert result["issue_size_description"] == "Fresh issue of equity shares"
    assert result["sector"] == "Technology"
    assert result["order_book"] == "₹1,200 crore"
    assert result["rhp_url"] == "https://example.com/rhp.zip"
    assert result["ratios_url"] == "https://example.com/ratios.zip"
    assert "symbol=EXAMPLE" in result["issue_info_url"]


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (100, 120, {
            "Downside sensitivity": 90,
            "Offer-band midpoint reference": 110,
            "Upside sensitivity": 132,
        }),
        (100, 100, {
            "Downside sensitivity": 90,
            "Offer-band midpoint reference": 100,
            "Upside sensitivity": 110,
        }),
    ],
)
def test_listing_price_scenarios_are_band_sensitivities(low, high, expected):
    scenarios = listing_price_scenarios(low, high)

    assert scenarios == pytest.approx(expected)


def test_listing_price_scenarios_validate_inputs():
    assert listing_price_scenarios(None, 100) is None
    assert listing_price_scenarios(120, 100) is None
    assert listing_price_scenarios(float("nan"), 100) is None
    with pytest.raises(ValueError, match="shock_pct"):
        listing_price_scenarios(100, 120, shock_pct=101)
    with pytest.raises(ValueError, match="shock_pct"):
        listing_price_scenarios(100, 120, shock_pct=float("nan"))


@pytest.mark.parametrize(
    ("subscription", "expected"),
    [
        (0.5, None),
        (1, None),
        (4, 25),
        (12.5, 8),
        (None, None),
        (-2, None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_retail_allotment_proxy_is_inverse_subscription(subscription, expected):
    assert retail_allotment_proxy(subscription) == expected


def test_fetch_ipo_issues_merges_current_and_forthcoming_official_records(monkeypatch):
    class FakeResponse:
        def __init__(self, data=None):
            self.data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self.data

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout):
            if url == ipo_analysis.NSE_IPO_URL:
                return FakeResponse()
            if url.endswith("/api/all-upcoming-issues?category=ipo"):
                return FakeResponse(
                    [
                        {
                            "companyName": "Active Limited",
                            "symbol": "ACTIVE",
                            "status": "Active",
                            "issuePrice": "Rs. 100 to Rs. 110",
                            "issueStartDate": "15-Oct-2026",
                            "issueEndDate": "19-Oct-2026",
                            "noOfTime": "2",
                        },
                        {
                            "companyName": "Future Limited",
                            "symbol": "FUTURE",
                            "status": "Forthcoming",
                            "issuePrice": "Rs. 200 to Rs. 220",
                            "issueStartDate": "21-Oct-2026",
                            "issueEndDate": "23-Oct-2026",
                        },
                    ]
                )
            if url.endswith("/api/ipo-current-issue"):
                return FakeResponse(
                    [
                        {
                            "companyName": "Active Limited",
                            "symbol": "ACTIVE",
                            "status": "Active",
                            "noOfTime": "2.5",
                        }
                    ]
                )
            if "symbol=ACTIVE" in url:
                return FakeResponse(
                    {
                        "activeCat": {
                            "dataList": [
                                {
                                    "category": "Retail Individual Investors(RIIs)",
                                    "noOfShareOffered": "100",
                                    "noOfSharesBid": "500",
                                    "noOfTotalMeant": "5",
                                }
                            ]
                        },
                        "issueInfo": {
                            "dataList": [
                                {"title": "Red Herring Prospectus", "value": "https://example.com/active-rhp.zip"},
                                {"title": "Issuer Website", "value": "https://issuer.example.com"},
                            ]
                        },
                    }
                )
            return FakeResponse(
                {
                    "issueInfo": {
                        "dataList": [
                            {"title": "Company Website", "value": "https://example.com"}
                        ]
                    }
                }
            )

    monkeypatch.setattr(ipo_analysis.requests, "Session", FakeSession)

    result = fetch_ipo_issues()

    assert len(result.issues) == 2
    active, forthcoming = (
        result.issues.set_index("symbol").loc[["ACTIVE", "FUTURE"]].to_dict("records")
    )
    assert active["total_subscription"] == 2.5
    assert active["retail_subscription"] == 5
    assert active["price_low"] == 100
    assert active["issue_start_date"] == pd.Timestamp("2026-10-15")
    assert active["rhp_url"] == "https://example.com/active-rhp.zip"
    assert active["company_website"] == "https://issuer.example.com"
    assert pd.isna(forthcoming["retail_subscription"])
    assert forthcoming["company_website"] == "https://example.com"
    assert result.source_url == ipo_analysis.NSE_IPO_URL


def test_fetch_ipo_issues_surfaces_main_feed_network_errors(monkeypatch):
    class FailedSession:
        headers = {}

        def get(self, url, timeout):
            raise requests.ConnectionError("offline")

    monkeypatch.setattr(ipo_analysis.requests, "Session", FailedSession)

    with pytest.raises(IPOFeedError, match="Could not fetch current IPO data"):
        fetch_ipo_issues()
