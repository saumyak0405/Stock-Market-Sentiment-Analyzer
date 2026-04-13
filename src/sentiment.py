"""
sentiment.py
------------
FinBERT-powered sentiment analysis pipeline.
Uses ProsusAI/finbert — a BERT model fine-tuned on financial text.
"""

import torch
import pandas as pd
import numpy as np
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP = {
    "positive": "Bullish",
    "negative": "Bearish",
    "neutral":  "Neutral",
}
SCORE_MAP = {
    "positive": 1.0,
    "neutral":  0.0,
    "negative": -1.0,
}


@lru_cache(maxsize=1)
def load_finbert():
    """
    Load FinBERT model and tokenizer (cached — loads only once per session).

    Returns:
        HuggingFace pipeline for text classification
    """
    logger.info("Loading FinBERT model (first run downloads ~440MB)...")
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    logger.info(f"Running inference on {device_name}")

    nlp = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        top_k=None,       # return all class scores
    )
    logger.info("FinBERT loaded successfully.")
    return nlp


def analyze_text(text: str, nlp=None) -> dict:
    """
    Run FinBERT sentiment inference on a single text snippet.

    Args:
        text: News headline or financial text (max ~512 tokens)
        nlp:  Optional pre-loaded pipeline (avoids reloading)

    Returns:
        dict with keys:
            label       — 'Bullish' | 'Bearish' | 'Neutral'
            confidence  — float 0-1 (probability of predicted class)
            score       — float -1 to 1 (signed sentiment score)
            raw         — full probability dict from model
    """
    if nlp is None:
        nlp = load_finbert()

    text = text.strip()
    if not text:
        return {"label": "Neutral", "confidence": 0.0, "score": 0.0, "raw": {}}

    # FinBERT max token length is 512; truncate long text
    text = text[:512]

    try:
        results = nlp(text)[0]   # list of {label, score} dicts
        probs = {r["label"].lower(): r["score"] for r in results}

        best_label = max(probs, key=probs.get)
        confidence = probs[best_label]
        sentiment_score = sum(
            SCORE_MAP.get(lbl, 0) * prob for lbl, prob in probs.items()
        )

        return {
            "label":      LABEL_MAP.get(best_label, "Neutral"),
            "confidence": round(confidence, 4),
            "score":      round(sentiment_score, 4),
            "raw":        {LABEL_MAP.get(k, k): round(v, 4) for k, v in probs.items()},
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"label": "Neutral", "confidence": 0.0, "score": 0.0, "raw": {}}


def analyze_dataframe(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    """
    Run FinBERT sentiment on every row of a DataFrame.

    Args:
        df:        DataFrame containing news articles
        text_col:  Column to run inference on (default: 'title')

    Returns:
        Original DataFrame with added columns:
            sentiment, confidence, sentiment_score,
            prob_bullish, prob_bearish, prob_neutral
    """
    if df.empty:
        return df

    nlp = load_finbert()
    logger.info(f"Running FinBERT inference on {len(df)} articles...")

    results = []
    for text in df[text_col].tolist():
        results.append(analyze_text(text, nlp=nlp))

    result_df = pd.DataFrame(results)
    df = df.copy()
    df["sentiment"]       = result_df["label"].values
    df["confidence"]      = result_df["confidence"].values
    df["sentiment_score"] = result_df["score"].values
    df["prob_bullish"]    = result_df["raw"].apply(lambda x: x.get("Bullish", 0)).values
    df["prob_bearish"]    = result_df["raw"].apply(lambda x: x.get("Bearish", 0)).values
    df["prob_neutral"]    = result_df["raw"].apply(lambda x: x.get("Neutral", 0)).values

    logger.info("Inference complete.")
    return df


def compute_aggregate_sentiment(df: pd.DataFrame) -> dict:
    """
    Aggregate per-article sentiment into a single ticker-level summary.

    Args:
        df: DataFrame output from analyze_dataframe()

    Returns:
        dict with keys:
            overall_sentiment, overall_score,
            bullish_pct, bearish_pct, neutral_pct,
            avg_confidence, article_count
    """
    if df.empty or "sentiment_score" not in df.columns:
        return {
            "overall_sentiment": "Neutral",
            "overall_score": 0.0,
            "bullish_pct": 0,
            "bearish_pct": 0,
            "neutral_pct": 100,
            "avg_confidence": 0.0,
            "article_count": 0,
        }

    counts = df["sentiment"].value_counts()
    total = len(df)
    bullish_pct = round(counts.get("Bullish", 0) / total * 100, 1)
    bearish_pct = round(counts.get("Bearish", 0) / total * 100, 1)
    neutral_pct = round(100 - bullish_pct - bearish_pct, 1)

    # Confidence-weighted average score
    weights = df["confidence"].values
    scores  = df["sentiment_score"].values
    if weights.sum() > 0:
        overall_score = float(np.average(scores, weights=weights))
    else:
        overall_score = float(scores.mean())

    if overall_score > 0.15:
        overall_sentiment = "Bullish"
    elif overall_score < -0.15:
        overall_sentiment = "Bearish"
    else:
        overall_sentiment = "Neutral"

    return {
        "overall_sentiment": overall_sentiment,
        "overall_score":     round(overall_score, 4),
        "bullish_pct":       bullish_pct,
        "bearish_pct":       bearish_pct,
        "neutral_pct":       neutral_pct,
        "avg_confidence":    round(float(df["confidence"].mean()), 4),
        "article_count":     total,
    }


if __name__ == "__main__":
    # Quick smoke test
    sample_headlines = [
        "Apple reports record quarterly earnings, beats analyst expectations",
        "iPhone sales decline as competition intensifies in China market",
        "Apple announces new AI features for iOS 18 at WWDC",
    ]
    nlp = load_finbert()
    for h in sample_headlines:
        result = analyze_text(h, nlp=nlp)
        print(f"[{result['label']:8s}] ({result['score']:+.2f}) {h}")
