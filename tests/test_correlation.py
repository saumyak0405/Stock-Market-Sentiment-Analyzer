"""
tests/test_correlation.py
-------------------------
Unit tests for the correlation analysis module.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np

from correlation import (
    build_sentiment_series,
    correlate_sentiment_returns,
    sentiment_backtest,
)


@pytest.fixture
def sample_price_df():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    np.random.seed(42)
    close = 150 + np.random.randn(30).cumsum()
    df = pd.DataFrame({
        "Close":  close,
        "Open":   close * 0.999,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Volume": np.random.randint(1e6, 5e6, 30),
    }, index=dates)
    df["daily_return"] = df["Close"].pct_change()
    return df


@pytest.fixture
def sample_news_df():
    dates = pd.date_range("2024-01-02", periods=15, freq="B")
    return pd.DataFrame({
        "published":      [d.strftime("%Y-%m-%d 09:00") for d in dates],
        "sentiment_score": np.random.uniform(-0.5, 0.5, 15),
        "sentiment":      np.random.choice(["Bullish", "Bearish", "Neutral"], 15),
        "title":          [f"Headline {i}" for i in range(15)],
    })


class TestBuildSentimentSeries:

    def test_returns_series(self, sample_news_df):
        result = build_sentiment_series(sample_news_df)
        assert isinstance(result, pd.Series)

    def test_empty_df(self):
        result = build_sentiment_series(pd.DataFrame())
        assert result.empty

    def test_daily_aggregation(self):
        df = pd.DataFrame({
            "published":      ["2024-01-02 09:00", "2024-01-02 14:00", "2024-01-03 10:00"],
            "sentiment_score": [0.5, -0.1, 0.3],
        })
        result = build_sentiment_series(df)
        assert len(result) == 2
        assert abs(result.iloc[0] - 0.2) < 0.01


class TestCorrelateReturns:

    def test_returns_dict(self, sample_news_df, sample_price_df):
        series = build_sentiment_series(sample_news_df)
        result = correlate_sentiment_returns(series, sample_price_df)
        assert isinstance(result, dict)
        assert "pearson_r" in result
        assert "p_value" in result

    def test_empty_inputs(self):
        result = correlate_sentiment_returns(pd.Series(), pd.DataFrame())
        assert result["pearson_r"] is None

    def test_r_range(self, sample_news_df, sample_price_df):
        series = build_sentiment_series(sample_news_df)
        result = correlate_sentiment_returns(series, sample_price_df)
        if result["pearson_r"] is not None:
            assert -1.0 <= result["pearson_r"] <= 1.0


class TestBacktest:

    def test_returns_dict(self, sample_news_df, sample_price_df):
        result = sentiment_backtest(sample_news_df, sample_price_df)
        assert isinstance(result, dict)
        assert "n_signals" in result

    def test_accuracy_range(self, sample_news_df, sample_price_df):
        result = sentiment_backtest(sample_news_df, sample_price_df)
        if result.get("accuracy") is not None:
            assert 0 <= result["accuracy"] <= 100

    def test_empty_inputs(self):
        result = sentiment_backtest(pd.DataFrame(), pd.DataFrame())
        assert result["n_signals"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
