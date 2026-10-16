from __future__ import annotations

import numpy as np
import pandas as pd


LONG_TERM_COLUMNS = [
    "Rank",
    "Setup status",
    "Symbol",
    "Company",
    "Market cap (₹ Cr)",
    "ROCE (%)",
    "P/E",
    "CMP (₹)",
    "Entry zone (₹)",
    "Target 1 (₹)",
    "Target 2 (₹)",
    "Target 3 (₹)",
    "Tentative timeline",
    "Confidence score (%)",
    "CMP vs 50-day average (%)",
    "CMP vs 200-day average (%)",
    "52-week range position (%)",
    "Order book",
    "Legal screening",
    "Screening evidence",
]


def select_long_term_candidates(
    screen_data: pd.DataFrame,
    market_cap_limit: float = 25_000,
) -> pd.DataFrame:
    """Select small-cap candidates with available baseline fundamental evidence."""
    if screen_data.empty:
        return pd.DataFrame(columns=screen_data.columns)

    required = ["Market Cap (Rs. Cr)", "ROCE (%)", "P/E", "FII change (%)", "DII change (%)"]
    if not set(required).issubset(screen_data.columns):
        return pd.DataFrame(columns=screen_data.columns)

    candidates = screen_data.copy()
    for column in required:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates = candidates.dropna(subset=required)
    candidates = candidates[
        (candidates["Market Cap (Rs. Cr)"] > 0)
        & (candidates["Market Cap (Rs. Cr)"] <= market_cap_limit)
        & (candidates["ROCE (%)"] >= 15)
        & (candidates["P/E"] > 0)
        & (candidates["P/E"] <= 60)
        & (candidates["FII change (%)"] > 0)
        & (candidates["DII change (%)"] > 0)
    ]
    return candidates.sort_values(
        ["ROCE (%)", "FII change (%)", "DII change (%)"],
        ascending=False,
    ).drop_duplicates("Symbol").reset_index(drop=True)


def build_long_term_ideas(
    screen_data: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    market_cap_limit: float = 25_000,
) -> pd.DataFrame:
    """Build an independent small-cap shortlist from fundamentals and price history."""
    candidates = select_long_term_candidates(screen_data, market_cap_limit)
    ideas: list[dict] = []

    for _, row in candidates.iterrows():
        symbol = str(row["Symbol"]).upper()
        history = price_data.get(symbol)
        if (
            history is None
            or history.empty
            or "close" not in history.columns
            or history.attrs.get("synthetic")
        ):
            continue

        close = pd.to_numeric(history["close"], errors="coerce").dropna()
        if len(close) < 200:
            continue

        price = float(close.iloc[-1])
        average_50 = float(close.tail(50).mean())
        average_200 = float(close.tail(200).mean())
        if price <= 0 or average_50 <= 0 or average_200 <= 0:
            continue

        distance_50 = (price / average_50 - 1) * 100
        distance_200 = (price / average_200 - 1) * 100
        if abs(distance_50) > 15 or abs(distance_200) > 35:
            continue

        year_close = close.tail(252)
        year_low = float(year_close.min())
        year_high = float(year_close.max())
        if year_high <= year_low:
            continue
        range_position = (price - year_low) / (year_high - year_low)
        if not 0 <= range_position <= 0.85:
            continue

        strict_entry = max(abs(distance_50), abs(distance_200)) <= 15 and range_position <= 0.4
        roce = float(row["ROCE (%)"])
        pe = float(row["P/E"])
        market_cap = float(row["Market Cap (Rs. Cr)"])
        institutional_change = min(float(row["FII change (%)"]), float(row["DII change (%)"]))
        proximity_score = max(0.0, 1 - max(abs(distance_50), abs(distance_200)) / 15)
        bottom_score = max(0.0, 1 - range_position / 0.4)
        roce_score = min(25.0, 15 + max(0.0, roce - 15) * 0.5)
        valuation_score = 10.0 if pe <= 30 else 5.0 if pe <= 45 else 2.0
        ownership_score = min(10.0, 5 + institutional_change)
        size_score = max(0.0, 5 * (1 - market_cap / market_cap_limit))
        confidence = round(
            min(
                100.0,
                roce_score
                + valuation_score
                + ownership_score
                + 20 * proximity_score
                + 20 * bottom_score
                + 10
                + size_score,
            )
        )

        entry = (average_50 + average_200) / 2
        ideas.append(
            {
                "Symbol": symbol,
                "Setup status": (
                    "Entry candidate"
                    if strict_entry
                    else "Watchlist — wait for pullback"
                ),
                "Company": row.get("Company", symbol),
                "Market cap (₹ Cr)": round(market_cap, 2),
                "ROCE (%)": round(roce, 2),
                "P/E": round(pe, 2),
                "CMP (₹)": round(price, 2),
                "Entry zone (₹)": f"{min(average_50, average_200):.2f}–{max(average_50, average_200):.2f}",
                "Target 1 (₹)": round(entry * 1.15, 2),
                "Target 2 (₹)": round(entry * 1.30, 2),
                "Target 3 (₹)": round(entry * 1.50, 2),
                "Tentative timeline": "T1: 6–12 mo · T2: 12–18 mo · T3: 18–24 mo",
                "Confidence score (%)": confidence,
                "CMP vs 50-day average (%)": round(distance_50, 2),
                "CMP vs 200-day average (%)": round(distance_200, 2),
                "52-week range position (%)": round(range_position * 100, 2),
                "Order book": "Not verified",
                "Legal screening": "Not verified",
                "Screening evidence": (
                    f"ROCE {roce:.1f}%; FII/DII holding changes positive; "
                    f"CMP {distance_50:+.1f}% vs 50-day and {distance_200:+.1f}% vs 200-day average; "
                    f"at {range_position * 100:.0f}% of 52-week range"
                ),
            }
        )

    result = pd.DataFrame(ideas)
    if result.empty:
        return pd.DataFrame(columns=LONG_TERM_COLUMNS)
    result["_entry_priority"] = result["Setup status"].eq("Entry candidate")
    result = result.sort_values(
        ["_entry_priority", "Confidence score (%)", "ROCE (%)"],
        ascending=[False, False, False],
    ).drop(columns="_entry_priority").reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result[LONG_TERM_COLUMNS]
