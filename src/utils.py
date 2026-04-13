"""
utils.py — Indian market version
"""

import pandas as pd


def format_inr(value: float) -> str:
    """Format number in Indian currency system (crores/lakhs)."""
    if not value or value == 0:
        return "N/A"
    if value >= 1e12:
        return f"₹{value/1e7:.0f} Cr"   # lakh crores
    if value >= 1e9:
        return f"₹{value/1e7:.0f} Cr"   # crores
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value/1e5:.2f} L"    # lakhs
    return f"₹{value:,.0f}"


def format_market_cap(value: float) -> str:
    return format_inr(value)


def sentiment_color(label: str) -> str:
    return {
        "Bullish": "#1D9E75",
        "Bearish": "#D85A30",
        "Neutral": "#888780",
    }.get(label, "#888780")


def sentiment_emoji(label: str) -> str:
    return {"Bullish": "▲", "Bearish": "▼", "Neutral": "—"}.get(label, "—")


def pct_change_color(pct: float) -> str:
    if pct > 0:  return "#1D9E75"
    if pct < 0:  return "#D85A30"
    return "#888780"


def clean_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace("$", "").replace("#", "").replace(".NS", "").replace(".BO", "")


def safe_divide(a, b, default=0):
    try:
        return a / b if b != 0 else default
    except Exception:
        return default
