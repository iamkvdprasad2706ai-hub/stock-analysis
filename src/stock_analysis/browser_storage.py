from __future__ import annotations

import json
from pathlib import Path

import streamlit.components.v1 as components


WATCHLIST_STORAGE_KEY = "stock-analysis.watchlist.v1"

_browser_storage = components.declare_component(
    "stock_analysis_browser_storage",
    path=str(Path(__file__).with_name("browser_storage")),
)


def browser_storage(
    command: str,
    storage_key: str,
    request_id: str,
    payload: str | None = None,
) -> dict | None:
    return _browser_storage(
        command=command,
        storage_key=storage_key,
        request_id=request_id,
        payload=payload,
        default=None,
        key="watchlist_browser_storage",
    )


def parse_watchlist_payload(payload: str | None) -> list[dict[str, str]]:
    if payload is None:
        return []
    try:
        entries = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Saved watchlist data is not valid JSON.") from exc
    if not isinstance(entries, list):
        raise ValueError("Saved watchlist data must be a list.")

    validated = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Saved watchlist contains an invalid stock entry.")
        symbol = entry.get("symbol")
        exchange = entry.get("exchange")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Saved watchlist contains a stock with no symbol.")
        if exchange not in {"NSE", "BSE", "US"}:
            raise ValueError("Saved watchlist contains an unsupported exchange.")
        validated.append({"symbol": symbol.strip(), "exchange": exchange})
    return validated
