# 🇮🇳 Indian Stock Market Sentiment Analyzer

> **FinBERT-powered NLP pipeline for NSE/BSE stocks — analyzes sentiment from Economic Times, MoneyControl & Yahoo Finance India.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)](https://streamlit.io)
[![FinBERT](https://img.shields.io/badge/Model-FinBERT-yellow?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
[![NSE](https://img.shields.io/badge/Exchange-NSE%2FBSE-orange?style=flat-square)](https://nseindia.com)

---

## 🧠 What This Project Does

End-to-end automated pipeline that:

1. **Scrapes** financial news from Economic Times Markets RSS, MoneyControl RSS, and Yahoo Finance India
2. **Classifies** each headline as *Bullish / Bearish / Neutral* using **FinBERT** (BERT fine-tuned on financial text)
3. **Fetches** NSE stock prices in ₹ INR via `yfinance` with OHLCV data + RSI + SMA
4. **Correlates** daily NLP sentiment scores with next-day stock returns using Pearson r
5. **Backtests** whether the sentiment signal predicted price direction correctly
6. **Tests adversarial robustness** — can fake headlines manipulate the model? (InfoSec angle)
7. **Visualizes** everything in a dark-themed Streamlit dashboard with Plotly candlestick charts

---

## 📈 Supported Stocks

| Ticker     | Company                  | Sector      |
|------------|--------------------------|-------------|
| RELIANCE   | Reliance Industries      | Energy/Telecom |
| TCS        | Tata Consultancy Services | IT          |
| INFY       | Infosys                  | IT          |
| HDFCBANK   | HDFC Bank                | Banking     |
| ICICIBANK  | ICICI Bank               | Banking     |
| SBIN       | State Bank of India      | Banking     |
| WIPRO      | Wipro                    | IT          |
| TATAMOTORS | Tata Motors              | Auto        |
| BAJFINANCE | Bajaj Finance            | NBFC        |
| MARUTI     | Maruti Suzuki            | Auto        |
| ADANIENT   | Adani Enterprises        | Conglomerate |
| NIFTY      | Nifty 50 Index           | Index       |
| SENSEX     | BSE Sensex               | Index       |
| ...and any NSE listed stock! |      |             |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   dashboard.py (Streamlit)                │
└────────┬──────────────┬──────────────┬───────────────────┘
         │              │              │
   ┌─────▼──────┐ ┌─────▼──────┐ ┌───▼────────────┐
   │ scraper.py  │ │sentiment.py│ │ stock_data.py  │
   │             │ │            │ │                │
   │ ET Markets  │ │  FinBERT   │ │ yfinance NSE   │
   │ MoneyControl│ │ Inference  │ │ ₹ INR prices   │
   │ Yahoo India │ │            │ │ RSI, SMA       │
   └─────┬───────┘ └─────┬──────┘ └───┬────────────┘
         └───────────────▼────────────┘
                         │
               ┌──────────▼──────────┐
               │   correlation.py    │
               │  Pearson r, p-val   │
               │  Rolling corr       │
               │  Backtest engine    │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   adversarial.py    │  ← InfoSec angle
               │  Injection attacks  │
               │  Robustness score   │
               └─────────────────────┘
```

---

## ⚙️ Setup & Installation

### Step 1: Clone
```bash
git clone https://github.com/YOUR_USERNAME/indian-stock-sentiment-analyzer.git
cd indian-stock-sentiment-analyzer
```

### Step 2: Virtual environment (Mac/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install torch
pip install -r requirements.txt --ignore-installed torch
```

### Step 4: Run
```bash
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser.

> First run downloads FinBERT (~440MB, one time only).

---

## 🚀 Deploy Free on Streamlit Cloud

```bash
git init && git add . && git commit -m "feat: Indian stock sentiment analyzer"
git remote add origin https://github.com/YOUR_USERNAME/indian-stock-sentiment-analyzer.git
git push -u origin main
```
Then deploy at [share.streamlit.io](https://share.streamlit.io) — free hosting, live URL for your resume.

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Component      | Library                        |
|----------------|--------------------------------|
| NLP Model      | FinBERT (ProsusAI/finbert)     |
| ML Framework   | HuggingFace Transformers       |
| News Scraping  | BeautifulSoup4, feedparser     |
| Price Data     | yfinance (NSE/BSE)             |
| Data Analysis  | pandas, numpy                  |
| Statistics     | scipy.stats (Pearson r)        |
| Dashboard      | Streamlit                      |
| Charts         | Plotly                         |
| Testing        | pytest                         |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.
It is not financial advice. Always consult a SEBI-registered advisor before investing.

---

## 👤 Author

**[Your Name]** · B.Tech CSE (Information Security) · 3rd Year  
[Your College] · Chennai, Tamil Nadu

