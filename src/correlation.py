"""
correlation.py
--------------
Statistical analysis module: correlates NLP sentiment signals
with stock price movements using pandas and scipy.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def build_sentiment_series(news_df: pd.DataFrame) -> pd.Series:
    """
    Aggregate per-article sentiment scores into a daily time series.

    Args:
        news_df: DataFrame with columns 'published' and 'sentiment_score'

    Returns:
        pd.Series indexed by date with daily mean sentiment score
    """
    if news_df.empty or "sentiment_score" not in news_df.columns:
        return pd.Series(dtype=float)

    df = news_df.copy()
    df["date"] = pd.to_datetime(df["published"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    daily = df.groupby("date")["sentiment_score"].mean()
    daily.name = "sentiment"
    return daily


def correlate_sentiment_returns(
    sentiment_series: pd.Series,
    price_df: pd.DataFrame,
    lag_days: int = 1,
) -> dict:
    """
    Compute Pearson correlation between daily sentiment and stock returns.

    Args:
        sentiment_series: Daily sentiment scores (output of build_sentiment_series)
        price_df:         OHLCV DataFrame from stock_data.get_price_history()
        lag_days:         Number of days to lag returns (1 = next-day returns)

    Returns:
        dict with:
            pearson_r     — correlation coefficient (-1 to 1)
            p_value       — statistical significance
            n_overlap     — number of matched days used
            interpretation — human-readable interpretation
            lag_days      — lag used
    """
    if sentiment_series.empty or price_df.empty:
        return {"pearson_r": None, "p_value": None, "n_overlap": 0,
                "interpretation": "Insufficient data", "lag_days": lag_days}

    returns = price_df["daily_return"].copy()
    returns.index = pd.to_datetime(returns.index).normalize()

    if lag_days > 0:
        returns = returns.shift(-lag_days)

    aligned = pd.concat([sentiment_series, returns], axis=1, join="inner").dropna()
    aligned.columns = ["sentiment", "return"]

    if len(aligned) < 5:
        return {"pearson_r": None, "p_value": None, "n_overlap": len(aligned),
                "interpretation": "Too few overlapping days", "lag_days": lag_days}

    r, p = stats.pearsonr(aligned["sentiment"], aligned["return"])

    if abs(r) < 0.2:
        interp = "Weak or no linear correlation"
    elif abs(r) < 0.4:
        interp = "Moderate correlation"
    else:
        interp = "Strong correlation"

    direction = "positive" if r > 0 else "negative"
    sig = "statistically significant (p < 0.05)" if p < 0.05 else f"not significant (p = {p:.3f})"

    return {
        "pearson_r":      round(r, 4),
        "p_value":        round(p, 4),
        "n_overlap":      len(aligned),
        "interpretation": f"{interp} — {direction} direction, {sig}",
        "lag_days":       lag_days,
        "aligned_df":     aligned,
    }


def compute_rolling_correlation(
    sentiment_series: pd.Series,
    price_df: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """
    Compute rolling Pearson correlation over a sliding window.

    Args:
        sentiment_series: Daily sentiment scores
        price_df:         OHLCV DataFrame
        window:           Rolling window in trading days

    Returns:
        pd.Series of rolling correlations indexed by date
    """
    returns = price_df["daily_return"].copy()
    returns.index = pd.to_datetime(returns.index).normalize()

    aligned = pd.concat([sentiment_series, returns], axis=1, join="inner").dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    aligned.columns = ["sentiment", "return"]
    rolling_corr = aligned["sentiment"].rolling(window).corr(aligned["return"])
    return rolling_corr.dropna()


def compute_all_correlations(sentiment_series: pd.Series, price_df: pd.DataFrame) -> dict:
    """
    Run correlation analysis across multiple lag windows.

    Returns:
        dict with correlation results for lag 0, 1, 2 days
    """
    results = {}
    for lag in [0, 1, 2]:
        key = f"lag_{lag}d"
        results[key] = correlate_sentiment_returns(sentiment_series, price_df, lag_days=lag)

    return results


def sentiment_backtest(news_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """
    Simple backtest: if sentiment was Bullish, would buying next day have been profitable?

    Args:
        news_df:  DataFrame with 'published' and 'sentiment' columns
        price_df: OHLCV DataFrame

    Returns:
        dict with accuracy, bullish_accuracy, bearish_accuracy, n_signals
    """
    if news_df.empty or price_df.empty:
        return {"accuracy": None, "n_signals": 0}

    df = news_df.copy()
    df["date"] = pd.to_datetime(df["published"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "sentiment"])

    daily_sentiment = df.groupby("date")["sentiment"].agg(
        lambda x: x.value_counts().index[0]
    )

    returns = price_df["daily_return"].copy()
    returns.index = pd.to_datetime(returns.index).normalize()
    next_day_returns = returns.shift(-1)

    aligned = pd.concat(
        [daily_sentiment, next_day_returns], axis=1, join="inner"
    ).dropna()
    aligned.columns = ["sentiment", "next_return"]

    if aligned.empty:
        return {"accuracy": None, "n_signals": 0}

    bullish_days  = aligned[aligned["sentiment"] == "Bullish"]
    bearish_days  = aligned[aligned["sentiment"] == "Bearish"]

    bull_acc = (bullish_days["next_return"] > 0).mean() if len(bullish_days) > 0 else None
    bear_acc = (bearish_days["next_return"] < 0).mean() if len(bearish_days) > 0 else None

    correct = (
        ((aligned["sentiment"] == "Bullish") & (aligned["next_return"] > 0)) |
        ((aligned["sentiment"] == "Bearish") & (aligned["next_return"] < 0))
    ).sum()
    total_acc = correct / len(aligned) if len(aligned) > 0 else None

    return {
        "accuracy":          round(float(total_acc) * 100, 1) if total_acc is not None else None,
        "bullish_accuracy":  round(float(bull_acc) * 100, 1) if bull_acc is not None else None,
        "bearish_accuracy":  round(float(bear_acc) * 100, 1) if bear_acc is not None else None,
        "n_signals":         len(aligned),
        "n_bullish":         len(bullish_days),
        "n_bearish":         len(bearish_days),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from stock_data import get_price_history

    price_df = get_price_history("AAPL", period="3mo")

    # Simulate some sentiment scores
    dates = price_df.index[:20]
    sent_series = pd.Series(
        np.random.uniform(-0.5, 0.5, len(dates)), index=dates, name="sentiment"
    )

    result = correlate_sentiment_returns(sent_series, price_df)
    print("Correlation result:", result)
