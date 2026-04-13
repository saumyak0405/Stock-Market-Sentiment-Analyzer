"""
scraper.py
----------
Collects financial news for Indian stocks from:
- MoneyControl RSS
- Economic Times Markets RSS
- Investing.com India RSS
- Yahoo Finance RSS (supports NSE tickers like RELIANCE.NS)
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timezone
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Indian financial news RSS feeds
RSS_FEEDS = {
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "MoneyControl":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "ET Markets":     "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
}

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN"


def fetch_yahoo_rss(ticker: str, max_articles: int = 15) -> list[dict]:
    """
    Fetch headlines from Yahoo Finance RSS for NSE/BSE tickers.
    Ticker format: RELIANCE.NS (NSE) or RELIANCE.BO (BSE)
    """
    # Auto-append .NS if no exchange suffix
    if "." not in ticker:
        ticker_ns = ticker + ".NS"
    else:
        ticker_ns = ticker

    url = YAHOO_RSS.format(ticker=ticker_ns)
    logger.info(f"Fetching Yahoo RSS for {ticker_ns}")

    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_articles]:
            try:
                pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pub_str = entry.get("published", "")[:16]

            articles.append({
                "title":     entry.get("title", "").strip(),
                "summary":   BeautifulSoup(entry.get("summary", ""), "lxml").get_text()[:300],
                "published": pub_str,
                "source":    "Yahoo Finance",
                "url":       entry.get("link", ""),
                "ticker":    ticker.upper(),
            })

        logger.info(f"Yahoo RSS: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.error(f"Yahoo RSS failed: {e}")
        return []


def fetch_indian_rss(ticker: str, max_articles: int = 20) -> list[dict]:
    """
    Fetch from Indian financial news RSS feeds and filter by ticker/company name.
    """
    # Map common tickers to company name keywords for filtering
    TICKER_KEYWORDS = {
        "RELIANCE":  ["reliance", "ril", "jio"],
        "TCS":       ["tcs", "tata consultancy"],
        "INFY":      ["infosys", "infy"],
        "HDFCBANK":  ["hdfc bank", "hdfc"],
        "WIPRO":     ["wipro"],
        "ICICIBANK": ["icici bank", "icici"],
        "SBIN":      ["sbi", "state bank"],
        "TATAMOTORS":["tata motors", "tatamotors"],
        "ADANIENT":  ["adani", "adani enterprises"],
        "HINDUNILVR":["hindustan unilever", "hul"],
        "BAJFINANCE":["bajaj finance"],
        "MARUTI":    ["maruti", "suzuki"],
        "SUNPHARMA": ["sun pharma", "sun pharmaceutical"],
        "LTIM":      ["ltimindtree", "lti"],
        "KOTAKBANK": ["kotak bank", "kotak mahindra"],
        "AXISBANK":  ["axis bank"],
        "ONGC":      ["ongc", "oil and natural gas"],
        "POWERGRID": ["power grid"],
        "NTPC":      ["ntpc"],
        "TITAN":     ["titan"],
        "NIFTY":     ["nifty", "sensex", "market"],
        "SENSEX":    ["sensex", "nifty", "market"],
    }

    base_ticker = ticker.replace(".NS", "").replace(".BO", "").upper()
    keywords = TICKER_KEYWORDS.get(base_ticker, [base_ticker.lower()])

    all_articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            logger.info(f"Fetching {source_name} RSS...")
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:40]:
                title = entry.get("title", "").strip()
                title_lower = title.lower()

                # Include if ticker keyword found OR if it's a broad market article
                is_relevant = (
                    any(kw in title_lower for kw in keywords) or
                    base_ticker in ["NIFTY", "SENSEX"] or
                    any(w in title_lower for w in ["nifty", "sensex", "market", "bse", "nse"])
                )

                if not is_relevant:
                    continue

                try:
                    pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pub_str = entry.get("published", "")[:16]

                all_articles.append({
                    "title":     title,
                    "summary":   BeautifulSoup(entry.get("summary", ""), "lxml").get_text()[:300],
                    "published": pub_str,
                    "source":    source_name,
                    "url":       entry.get("link", ""),
                    "ticker":    base_ticker,
                })

            time.sleep(0.3)

        except Exception as e:
            logger.error(f"{source_name} fetch failed: {e}")
            continue

    logger.info(f"Indian RSS total: {len(all_articles)} relevant articles")
    return all_articles[:max_articles]


def get_all_news(ticker: str, max_total: int = 30) -> pd.DataFrame:
    """
    Aggregate news from Yahoo Finance + Indian RSS sources.

    Args:
        ticker:    NSE ticker (e.g. 'RELIANCE', 'TCS', 'INFY')
        max_total: Max articles to return

    Returns:
        DataFrame with news headlines
    """
    articles = []

    yahoo = fetch_yahoo_rss(ticker, max_articles=15)
    articles.extend(yahoo)
    time.sleep(0.5)

    indian = fetch_indian_rss(ticker, max_articles=20)
    articles.extend(indian)

    if not articles:
        logger.warning(f"No articles found for {ticker}")
        return pd.DataFrame(columns=["title", "summary", "published", "source", "url", "ticker"])

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset=["title"])
    df = df[df["title"].str.len() > 10]
    df = df.head(max_total).reset_index(drop=True)

    logger.info(f"Total unique articles: {len(df)}")
    return df


if __name__ == "__main__":
    df = get_all_news("RELIANCE")
    print(df[["title", "source", "published"]].to_string())
