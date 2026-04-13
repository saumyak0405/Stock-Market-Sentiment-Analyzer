"""
stock_data.py
-------------
Fetches Indian stock data from NSE/BSE via yfinance.
Ticker format: RELIANCE.NS (NSE), RELIANCE.BO (BSE)
Includes Nifty 50 and Sensex index support.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Popular Indian stocks with display names
INDIAN_TICKERS = {
    "RELIANCE":   {"name": "Reliance Industries",     "sector": "Energy / Telecom"},
    "TCS":        {"name": "Tata Consultancy Services","sector": "IT"},
    "INFY":       {"name": "Infosys",                  "sector": "IT"},
    "HDFCBANK":   {"name": "HDFC Bank",                "sector": "Banking"},
    "ICICIBANK":  {"name": "ICICI Bank",               "sector": "Banking"},
    "SBIN":       {"name": "State Bank of India",      "sector": "Banking"},
    "WIPRO":      {"name": "Wipro",                    "sector": "IT"},
    "TATAMOTORS": {"name": "Tata Motors",              "sector": "Auto"},
    "BAJFINANCE": {"name": "Bajaj Finance",            "sector": "NBFC"},
    "MARUTI":     {"name": "Maruti Suzuki",            "sector": "Auto"},
    "SUNPHARMA":  {"name": "Sun Pharmaceutical",       "sector": "Pharma"},
    "ADANIENT":   {"name": "Adani Enterprises",        "sector": "Conglomerate"},
    "KOTAKBANK":  {"name": "Kotak Mahindra Bank",      "sector": "Banking"},
    "AXISBANK":   {"name": "Axis Bank",                "sector": "Banking"},
    "HINDUNILVR": {"name": "Hindustan Unilever",       "sector": "FMCG"},
    "ONGC":       {"name": "ONGC",                     "sector": "Oil & Gas"},
    "LTIM":       {"name": "LTIMindtree",              "sector": "IT"},
    "TITAN":      {"name": "Titan Company",            "sector": "Consumer"},
    "POWERGRID":  {"name": "Power Grid Corp",          "sector": "Utilities"},
    "NTPC":       {"name": "NTPC",                     "sector": "Utilities"},
    "NIFTY":      {"name": "Nifty 50 Index",           "sector": "Index"},
    "SENSEX":     {"name": "BSE Sensex",               "sector": "Index"},
}

INDEX_TICKERS = {
    "NIFTY":  "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTYIT":"^CNXIT",
    "NIFTYBANK": "^NSEBANK",
}


def resolve_ticker(ticker: str) -> str:
    """Convert short NSE ticker to yfinance format."""
    ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
    if ticker in INDEX_TICKERS:
        return INDEX_TICKERS[ticker]
    return ticker + ".NS"


def get_price_history(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Download historical OHLCV data for Indian stocks/indices.

    Args:
        ticker:   NSE symbol (e.g. 'RELIANCE', 'TCS', 'NIFTY')
        period:   '1mo', '3mo', '6mo', '1y'
        interval: '1d', '1wk'

    Returns:
        DataFrame with price data and derived features.
        Prices are in INR (₹).
    """
    yf_ticker = resolve_ticker(ticker)
    logger.info(f"Fetching price history for {yf_ticker} ({period})")

    try:
        tk = yf.Ticker(yf_ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data for {yf_ticker}")
            return pd.DataFrame()

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"

        # Derived features
        df["daily_return"]  = df["Close"].pct_change()
        df["log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
        df["volatility_5d"] = df["daily_return"].rolling(5).std()
        df["sma_20"]        = df["Close"].rolling(20).mean()
        df["sma_5"]         = df["Close"].rolling(5).mean()
        df["rsi"]           = _compute_rsi(df["Close"])

        logger.info(f"Fetched {len(df)} bars for {yf_ticker}")
        return df

    except Exception as e:
        logger.error(f"Price fetch failed: {e}")
        return pd.DataFrame()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def get_ticker_info(ticker: str) -> dict:
    """Get company metadata. Falls back to local dict for Indian stocks."""
    base = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Try local lookup first (faster, always works)
    local = INDIAN_TICKERS.get(base, {})

    try:
        yf_ticker = resolve_ticker(ticker)
        tk   = yf.Ticker(yf_ticker)
        info = tk.info

        return {
            "name":          info.get("longName", local.get("name", base)),
            "sector":        info.get("sector", local.get("sector", "N/A")),
            "industry":      info.get("industry", "N/A"),
            "market_cap":    info.get("marketCap", 0),
            "pe_ratio":      info.get("trailingPE", None),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", None)),
            "52w_high":      info.get("fiftyTwoWeekHigh", None),
            "52w_low":       info.get("fiftyTwoWeekLow", None),
            "avg_volume":    info.get("averageVolume", None),
            "currency":      info.get("currency", "INR"),
            "exchange":      info.get("exchange", "NSI"),
        }
    except Exception as e:
        logger.error(f"Ticker info fetch failed: {e}")
        return {
            "name":     local.get("name", base),
            "sector":   local.get("sector", "N/A"),
            "currency": "INR",
        }


def compute_technical_signals(df: pd.DataFrame) -> dict:
    """Compute technical signals from price data."""
    if df.empty or len(df) < 5:
        return {}

    latest = df.iloc[-1]

    return {
        "price":            round(latest["Close"], 2),
        "daily_change_pct": round(latest["daily_return"] * 100, 2) if pd.notna(latest["daily_return"]) else 0,
        "volume":           int(latest["Volume"]),
        "volatility":       round(latest["volatility_5d"] * 100, 2) if pd.notna(latest["volatility_5d"]) else 0,
        "rsi":              round(latest["rsi"], 1) if pd.notna(latest.get("rsi", float("nan"))) else None,
        "above_sma20":      bool(latest["Close"] > latest["sma_20"]) if pd.notna(latest["sma_20"]) else None,
        "golden_cross":     bool(latest["sma_5"] > latest["sma_20"]) if pd.notna(latest["sma_5"]) and pd.notna(latest["sma_20"]) else None,
        "recent_trend":     "up" if df["Close"].iloc[-5:].is_monotonic_increasing else
                            "down" if df["Close"].iloc[-5:].is_monotonic_decreasing else "mixed",
    }


if __name__ == "__main__":
    df = get_price_history("RELIANCE")
    print(df.tail())
    print(get_ticker_info("RELIANCE"))
