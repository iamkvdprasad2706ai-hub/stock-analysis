"""NSE public-issue data and transparent IPO scenario helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

NSE_IPO_URL = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"
_NSE_BASE_URL = "https://www.nseindia.com"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)


class IPOFeedError(RuntimeError):
    """Raised when the official NSE IPO issue feed cannot be read."""


@dataclass(frozen=True)
class IPOFeed:
    """Current and forthcoming NSE public issues with an observation time."""

    issues: pd.DataFrame
    fetched_at: datetime
    source_url: str = NSE_IPO_URL


def _get_json(session: requests.Session, path: str) -> Any:
    response = session.get(f"{_NSE_BASE_URL}{path}", timeout=25)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise IPOFeedError(f"NSE returned invalid JSON for {path}.") from exc


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _issue_price_band(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    values = re.findall(r"\d[\d,]*(?:\.\d+)?", str(value))
    parsed = [float(item.replace(",", "")) for item in values[:2]]
    if len(parsed) == 1:
        return parsed[0], parsed[0]
    if len(parsed) == 2:
        return min(parsed), max(parsed)
    return None, None


def _issue_info(detail: dict[str, Any]) -> dict[str, str]:
    raw = detail.get("issueInfo", {})
    entries = raw.get("dataList", []) if isinstance(raw, dict) else []
    return {
        str(item["title"]).strip(): str(item.get("value") or "").strip().strip('"')
        for item in entries
        if isinstance(item, dict) and item.get("title")
    }


def _first_info_value(info: dict[str, str], *keys: str) -> str | None:
    return next((info[key] for key in keys if info.get(key)), None)


def _category_details(detail: dict[str, Any]) -> list[dict[str, Any]]:
    active = detail.get("activeCat")
    if isinstance(active, dict):
        entries = active.get("dataList", [])
        if isinstance(entries, list):
            return [
                {
                    "category": item.get("category"),
                    "shares_offered": _number(item.get("noOfShareOffered")),
                    "shares_bid": _number(item.get("noOfSharesBid")),
                    "subscription": _number(item.get("noOfTotalMeant")),
                }
                for item in entries
                if isinstance(item, dict)
                and item.get("category")
                and item.get("category") != "Category"
            ]

    bids = detail.get("bidDetails", [])
    if isinstance(bids, list):
        return [
            {
                "category": item.get("category"),
                "shares_offered": _number(item.get("noOfSharesOffered")),
                "shares_bid": _number(item.get("noOfsharesBid")),
                "subscription": _number(item.get("noOfTime")),
            }
            for item in bids
            if isinstance(item, dict) and item.get("category")
        ]
    return []


def _issue_record(issue: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    info = _issue_info(detail)
    categories = _category_details(detail)
    retail = next(
        (
            item
            for item in categories
            if "retail individual investors" in str(item["category"]).lower()
            or str(item["category"]).strip().lower() in {"rii", "retail"}
        ),
        {},
    )
    price = issue.get("issuePrice") or info.get("Price Range")
    price_low, price_high = _issue_price_band(price)
    symbol = str(issue.get("symbol") or "").strip().upper()
    series = str(issue.get("series") or "EQ").strip().upper()
    status = str(issue.get("status") or "Unknown").strip()
    issue_info_url = (
        f"{_NSE_BASE_URL}/market-data/issue-information?"
        f"{urlencode({'symbol': symbol, 'series': series, 'type': status})}"
    )
    return {
        "company": str(issue.get("companyName") or detail.get("companyName") or symbol),
        "symbol": symbol,
        "series": series,
        "status": status,
        "issue_start_date": issue.get("issueStartDate"),
        "issue_end_date": issue.get("issueEndDate"),
        "price_band": price,
        "price_low": price_low,
        "price_high": price_high,
        "issue_size_shares": _number(issue.get("issueSize")),
        "total_subscription": _number(issue.get("noOfTime")),
        "retail_subscription": retail.get("subscription"),
        "retail_shares_offered": retail.get("shares_offered"),
        "retail_shares_bid": retail.get("shares_bid"),
        "categories": categories,
        "issue_size_description": info.get("Issue Size"),
        "issue_type": info.get("Issue Type"),
        "face_value": info.get("Face Value"),
        "lot_size": info.get("Bid Lot") or info.get("Minimum Order Quantity"),
        "company_website": _first_info_value(
            info,
            "Company Website",
            "Issuer Website",
            "Official Website",
            "Website",
        ),
        "investor_relations_url": _first_info_value(
            info,
            "Investor Relations",
            "Investor Relations Website",
            "Investor Relations URL",
        ),
        "annual_report_url": _first_info_value(
            info,
            "Annual Report",
            "Latest Annual Report",
            "Financial Results",
        ),
        "sector": _first_info_value(info, "Sector", "Industry", "Industry/Sector"),
        "business_description": _first_info_value(
            info,
            "Business Overview",
            "Nature of Business",
            "Business of the Company",
            "Company Description",
        ),
        "recent_expansion": _first_info_value(
            info, "Recent Expansion", "Expansion", "Capital Expenditure"
        ),
        "order_book": _first_info_value(
            info, "Order Book", "Orderbook", "Order Book Position"
        ),
        "ratios_url": info.get("Ratios / Basis of Issue Price"),
        "rhp_url": info.get("Red Herring Prospectus"),
        "issue_info_url": issue_info_url,
        "details_error": None,
    }


def fetch_ipo_issues() -> IPOFeed:
    """Fetch current and forthcoming IPOs and their NSE issue-detail records.

    NSE's page is opened first to establish the session cookies used by its
    public JSON endpoints. A failed main feed raises ``IPOFeedError``; an
    individual detail failure is retained on that issue so partial data stays
    visible and is not mistaken for a complete analysis.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Referer": NSE_IPO_URL,
        }
    )
    try:
        page_response = session.get(NSE_IPO_URL, timeout=25)
        page_response.raise_for_status()
        upcoming = _get_json(session, "/api/all-upcoming-issues?category=ipo")
        current = _get_json(session, "/api/ipo-current-issue")
    except requests.RequestException as exc:
        raise IPOFeedError(f"Could not fetch current IPO data from NSE: {exc}") from exc

    if not isinstance(upcoming, list) or not isinstance(current, list):
        raise IPOFeedError("NSE returned an unexpected IPO issue-feed shape.")

    current_by_symbol = {
        str(row.get("symbol", "")).strip().upper(): row
        for row in current
        if isinstance(row, dict) and row.get("symbol")
    }
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for row in [*upcoming, *current]:
        if not isinstance(row, dict) or not row.get("symbol"):
            continue
        symbol = str(row["symbol"]).strip().upper()
        series = str(row.get("series") or "EQ").strip().upper()
        combined[(symbol, series)] = {**combined.get((symbol, series), {}), **row}

    records = []
    for (symbol, series), issue in combined.items():
        status = str(issue.get("status") or "Unknown").strip()
        detail_path = (
            f"/api/ipo-detail?{urlencode({'symbol': symbol, 'series': series})}"
        )
        try:
            detail = _get_json(session, detail_path)
            if not isinstance(detail, dict):
                raise IPOFeedError(f"NSE returned unexpected issue details for {symbol}.")
            record = _issue_record(issue, detail)
        except requests.RequestException as exc:
            record = _issue_record(issue, {})
            record["details_error"] = f"NSE issue details unavailable: {exc}"
        except IPOFeedError as exc:
            record = _issue_record(issue, {})
            record["details_error"] = str(exc)
        if status.lower() == "active":
            subscription = current_by_symbol.get(symbol, {})
            record["total_subscription"] = _number(
                subscription.get("noOfTime", record["total_subscription"])
            )
            record["retail_subscription"] = (
                record["retail_subscription"]
                if record["retail_subscription"] is not None
                else None
            )
        records.append(record)

    issues = pd.DataFrame(records)
    if not issues.empty:
        issues["issue_start_date"] = pd.to_datetime(
            issues["issue_start_date"], format="%d-%b-%Y", errors="coerce"
        )
        issues["issue_end_date"] = pd.to_datetime(
            issues["issue_end_date"], format="%d-%b-%Y", errors="coerce"
        )
        issues = issues.sort_values(
            ["issue_start_date", "company"], na_position="last"
        ).reset_index(drop=True)
    return IPOFeed(issues=issues, fetched_at=datetime.now().astimezone())


def listing_price_scenarios(
    price_low: float | None,
    price_high: float | None,
    *,
    shock_pct: float = 10.0,
) -> dict[str, float] | None:
    """Return illustrative price-band sensitivity cases, not a price forecast."""
    if (
        price_low is None
        or price_high is None
        or not math.isfinite(price_low)
        or not math.isfinite(price_high)
        or price_low <= 0
        or price_high < price_low
    ):
        return None
    if not math.isfinite(shock_pct) or not 0 <= shock_pct <= 100:
        raise ValueError("shock_pct must be between 0 and 100.")
    return {
        "Downside sensitivity": price_low * (1 - shock_pct / 100),
        "Offer-band midpoint reference": (price_low + price_high) / 2,
        "Upside sensitivity": price_high * (1 + shock_pct / 100),
    }


def retail_allotment_proxy(retail_subscription: float | None) -> float | None:
    """Return the inverse oversubscription share-cover proxy, not allotment odds."""
    if (
        retail_subscription is None
        or not math.isfinite(retail_subscription)
        or retail_subscription <= 1
    ):
        return None
    return 100.0 / retail_subscription
