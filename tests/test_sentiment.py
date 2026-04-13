"""
tests/test_sentiment.py
-----------------------
Unit tests for the sentiment analysis module.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from sentiment import analyze_text, compute_aggregate_sentiment, LABEL_MAP, SCORE_MAP


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_nlp():
    """Mock FinBERT pipeline that returns predictable outputs."""
    def _nlp(text):
        text_lower = text.lower()
        if any(w in text_lower for w in ["beats", "record", "strong", "surge"]):
            return [[
                {"label": "positive", "score": 0.85},
                {"label": "neutral",  "score": 0.10},
                {"label": "negative", "score": 0.05},
            ]]
        elif any(w in text_lower for w in ["miss", "decline", "bankrupt", "plummet"]):
            return [[
                {"label": "negative", "score": 0.80},
                {"label": "neutral",  "score": 0.12},
                {"label": "positive", "score": 0.08},
            ]]
        else:
            return [[
                {"label": "neutral",  "score": 0.65},
                {"label": "positive", "score": 0.20},
                {"label": "negative", "score": 0.15},
            ]]
    return _nlp


@pytest.fixture
def sample_news_df():
    return pd.DataFrame([
        {"title": "Apple beats earnings, revenue surges",       "published": "2024-01-15 09:00", "source": "Yahoo"},
        {"title": "Stock declines on weak revenue miss",        "published": "2024-01-16 10:00", "source": "Finviz"},
        {"title": "Company files routine quarterly disclosure", "published": "2024-01-17 11:00", "source": "Yahoo"},
    ])


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAnalyzeText:

    def test_bullish_headline(self, mock_nlp):
        result = analyze_text("Apple beats earnings, revenue surges", nlp=mock_nlp)
        assert result["label"] == "Bullish"
        assert result["confidence"] > 0.5
        assert result["score"] > 0

    def test_bearish_headline(self, mock_nlp):
        result = analyze_text("Revenue miss causes stock to plummet", nlp=mock_nlp)
        assert result["label"] == "Bearish"
        assert result["score"] < 0

    def test_neutral_headline(self, mock_nlp):
        result = analyze_text("Company files annual report with SEC", nlp=mock_nlp)
        assert result["label"] == "Neutral"

    def test_empty_string(self, mock_nlp):
        result = analyze_text("", nlp=mock_nlp)
        assert result["label"] == "Neutral"
        assert result["score"] == 0.0

    def test_long_text_truncated(self, mock_nlp):
        long_text = "Apple " * 200
        result = analyze_text(long_text, nlp=mock_nlp)
        assert "label" in result
        assert "score" in result

    def test_result_keys(self, mock_nlp):
        result = analyze_text("Test headline", nlp=mock_nlp)
        assert set(result.keys()) == {"label", "confidence", "score", "raw"}

    def test_score_range(self, mock_nlp):
        result = analyze_text("Earnings growth exceeds expectations", nlp=mock_nlp)
        assert -1.0 <= result["score"] <= 1.0

    def test_confidence_range(self, mock_nlp):
        result = analyze_text("Stock market update", nlp=mock_nlp)
        assert 0.0 <= result["confidence"] <= 1.0


class TestComputeAggregateSentiment:

    def test_empty_dataframe(self):
        result = compute_aggregate_sentiment(pd.DataFrame())
        assert result["overall_sentiment"] == "Neutral"
        assert result["article_count"] == 0

    def test_bullish_aggregate(self):
        df = pd.DataFrame({
            "sentiment":       ["Bullish", "Bullish", "Bullish", "Neutral"],
            "sentiment_score": [0.8, 0.6, 0.7, 0.0],
            "confidence":      [0.9, 0.8, 0.85, 0.7],
        })
        result = compute_aggregate_sentiment(df)
        assert result["overall_sentiment"] == "Bullish"
        assert result["bullish_pct"] == 75.0

    def test_bearish_aggregate(self):
        df = pd.DataFrame({
            "sentiment":       ["Bearish", "Bearish", "Neutral"],
            "sentiment_score": [-0.7, -0.5, 0.05],
            "confidence":      [0.9, 0.8, 0.6],
        })
        result = compute_aggregate_sentiment(df)
        assert result["overall_sentiment"] == "Bearish"

    def test_pct_sum(self):
        df = pd.DataFrame({
            "sentiment":       ["Bullish", "Bearish", "Neutral", "Bullish"],
            "sentiment_score": [0.5, -0.5, 0.0, 0.4],
            "confidence":      [0.8, 0.8, 0.7, 0.9],
        })
        result = compute_aggregate_sentiment(df)
        total = result["bullish_pct"] + result["bearish_pct"] + result["neutral_pct"]
        assert abs(total - 100) < 0.5

    def test_article_count(self):
        df = pd.DataFrame({
            "sentiment":       ["Bullish"] * 5,
            "sentiment_score": [0.5] * 5,
            "confidence":      [0.8] * 5,
        })
        result = compute_aggregate_sentiment(df)
        assert result["article_count"] == 5


class TestLabelMap:

    def test_all_labels_mapped(self):
        assert "positive" in LABEL_MAP
        assert "negative" in LABEL_MAP
        assert "neutral"  in LABEL_MAP

    def test_score_map_signs(self):
        assert SCORE_MAP["positive"] > 0
        assert SCORE_MAP["negative"] < 0
        assert SCORE_MAP["neutral"]  == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
