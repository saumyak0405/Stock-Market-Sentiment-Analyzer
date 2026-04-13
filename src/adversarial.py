"""
adversarial.py
--------------
Adversarial robustness analysis for the FinBERT sentiment model.
Unique angle: tests whether injected fake headlines can manipulate
the sentiment score — relevant for financial misinformation & market manipulation.

Relevant to CSE Information Security background.
"""

import re
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# --- Injection patterns that might flip sentiment ---
BULLISH_INJECTIONS = [
    "beats earnings expectations by wide margin",
    "record revenue growth reported",
    "strong buy rating from analysts",
    "bullish outlook as profits surge",
    "massive dividend increase announced",
]

BEARISH_INJECTIONS = [
    "massive earnings miss disappoints investors",
    "revenue decline raises bankruptcy concerns",
    "SEC investigation launched into accounting fraud",
    "CEO resigns amid scandal, stock plummets",
    "credit downgrade signals financial distress",
]

NEUTRAL_OBFUSCATIONS = [
    "company files standard quarterly report",
    "routine management restructuring announced",
    "board meets to discuss operational updates",
]


def inject_phrase(headline: str, injection: str, position: str = "append") -> str:
    """
    Inject a phrase into a headline to test model robustness.

    Args:
        headline:  Original headline
        injection: Phrase to inject
        position:  'append', 'prepend', or 'middle'

    Returns:
        Modified headline string
    """
    if position == "append":
        return f"{headline}; {injection}"
    elif position == "prepend":
        return f"{injection}: {headline}"
    elif position == "middle":
        words = headline.split()
        mid = len(words) // 2
        inject_words = injection.split()
        combined = words[:mid] + inject_words + words[mid:]
        return " ".join(combined)
    return headline


def character_perturbation(headline: str, char_swap_rate: float = 0.05) -> str:
    """
    Apply subtle character-level perturbations (typos, zero-width spaces).
    Tests if model is sensitive to minor text noise.

    Args:
        headline:       Input text
        char_swap_rate: Fraction of characters to perturb

    Returns:
        Perturbed headline
    """
    chars = list(headline)
    n_swaps = max(1, int(len(chars) * char_swap_rate))
    indices = np.random.choice(len(chars), n_swaps, replace=False)

    confusables = {
        'a': 'а',   # Cyrillic 'а' looks like Latin 'a'
        'e': 'е',   # Cyrillic 'е'
        'o': 'о',   # Cyrillic 'о'
        'p': 'р',   # Cyrillic 'р'
        'c': 'с',   # Cyrillic 'с'
    }

    for idx in indices:
        ch = chars[idx].lower()
        if ch in confusables:
            chars[idx] = confusables[ch]

    return "".join(chars)


def run_adversarial_tests(headline: str, nlp=None) -> dict:
    """
    Run a battery of adversarial tests on a single headline.

    Args:
        headline: Original news headline
        nlp:      FinBERT pipeline (loaded externally to avoid reloading)

    Returns:
        dict with original result, adversarial results, and flip_rate
    """
    from sentiment import analyze_text

    original = analyze_text(headline, nlp=nlp)
    original_label = original["label"]
    original_score = original["score"]

    tests = []

    # Test 1: Bullish phrase injection
    for phrase in BULLISH_INJECTIONS[:3]:
        for pos in ["append", "prepend"]:
            modified = inject_phrase(headline, phrase, position=pos)
            result   = analyze_text(modified, nlp=nlp)
            tests.append({
                "test_type":     f"bullish_injection_{pos}",
                "modified_text": modified,
                "label":         result["label"],
                "score":         result["score"],
                "score_delta":   round(result["score"] - original_score, 4),
                "label_flipped": result["label"] != original_label,
            })

    # Test 2: Bearish phrase injection
    for phrase in BEARISH_INJECTIONS[:3]:
        for pos in ["append", "prepend"]:
            modified = inject_phrase(headline, phrase, position=pos)
            result   = analyze_text(modified, nlp=nlp)
            tests.append({
                "test_type":     f"bearish_injection_{pos}",
                "modified_text": modified,
                "label":         result["label"],
                "score":         result["score"],
                "score_delta":   round(result["score"] - original_score, 4),
                "label_flipped": result["label"] != original_label,
            })

    # Test 3: Character perturbation
    perturbed = character_perturbation(headline)
    result    = analyze_text(perturbed, nlp=nlp)
    tests.append({
        "test_type":     "char_perturbation",
        "modified_text": perturbed,
        "label":         result["label"],
        "score":         result["score"],
        "score_delta":   round(result["score"] - original_score, 4),
        "label_flipped": result["label"] != original_label,
    })

    test_df   = pd.DataFrame(tests)
    flip_rate = test_df["label_flipped"].mean()
    max_delta = test_df["score_delta"].abs().max()

    return {
        "original_headline": headline,
        "original_label":    original_label,
        "original_score":    original_score,
        "n_tests":           len(tests),
        "flip_rate":         round(float(flip_rate), 3),
        "max_score_delta":   round(float(max_delta), 4),
        "robustness_rating": _rate_robustness(flip_rate, max_delta),
        "tests":             test_df,
    }


def _rate_robustness(flip_rate: float, max_delta: float) -> str:
    """Classify model robustness based on adversarial metrics."""
    if flip_rate < 0.1 and max_delta < 0.3:
        return "High — model is robust to adversarial perturbations"
    elif flip_rate < 0.3 and max_delta < 0.6:
        return "Medium — model is moderately sensitive to injections"
    else:
        return "Low — model is vulnerable to sentiment manipulation attacks"


def batch_adversarial_report(headlines: list[str], nlp=None) -> pd.DataFrame:
    """
    Run adversarial tests across a list of headlines.

    Args:
        headlines: List of news headline strings
        nlp:       Pre-loaded FinBERT pipeline

    Returns:
        Summary DataFrame
    """
    rows = []
    for h in headlines:
        result = run_adversarial_tests(h, nlp=nlp)
        rows.append({
            "headline":         h[:80],
            "original_label":   result["original_label"],
            "flip_rate":        result["flip_rate"],
            "max_score_delta":  result["max_score_delta"],
            "robustness":       result["robustness_rating"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    test_headline = "Apple reports strong quarterly earnings, beats Wall Street estimates"
    print(f"Testing adversarial robustness on: '{test_headline}'")
    print("(FinBERT model required — skipping inference in standalone mode)")
    print("Run via the Streamlit dashboard for full adversarial analysis.")
