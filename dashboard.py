"""
dashboard.py — Indian Stock Market Sentiment Analyzer
Run: streamlit run dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from scraper     import get_all_news
from sentiment   import analyze_dataframe, compute_aggregate_sentiment, load_finbert
from stock_data  import (get_price_history, get_ticker_info,
                          compute_technical_signals, INDIAN_TICKERS)
from correlation import (build_sentiment_series, compute_all_correlations,
                          compute_rolling_correlation, sentiment_backtest)
from utils       import (format_inr, format_market_cap, sentiment_color,
                          sentiment_emoji, clean_ticker)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Market Sentiment Analyzer",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; }
h1,h2,h3 { font-family:'Space Mono',monospace; }
.news-card {
    background: #161b22;
    border-left: 3px solid #30363d;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin: 6px 0;
}
.news-bullish { border-left-color: #1D9E75; }
.news-bearish { border-left-color: #D85A30; }
.news-neutral { border-left-color: #888780; }
.index-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇮🇳 NSE/BSE ANALYZER")
    st.markdown("*FinBERT · Indian Markets*")
    st.divider()

    ticker_input = st.text_input(
        "NSE Ticker",
        value="RELIANCE",
        placeholder="e.g. RELIANCE, TCS, INFY",
        help="Enter NSE ticker symbol. .NS suffix added automatically."
    ).upper()

    st.caption("🔵 Nifty 50 Stocks:")
    nifty_row1 = st.columns(3)
    nifty_tickers_1 = ["RELIANCE", "TCS", "INFY"]
    nifty_tickers_2 = ["HDFCBANK", "SBIN", "WIPRO"]
    nifty_tickers_3 = ["TATAMOTORS", "ADANIENT", "MARUTI"]
    nifty_tickers_4 = ["ICICIBANK", "BAJFINANCE", "TITAN"]

    for row_tickers in [nifty_tickers_1, nifty_tickers_2, nifty_tickers_3, nifty_tickers_4]:
        cols = st.columns(3)
        for i, t in enumerate(row_tickers):
            if cols[i].button(t, key=f"qt_{t}", use_container_width=True):
                ticker_input = t

    st.caption("📊 Indices:")
    idx_cols = st.columns(2)
    for i, t in enumerate(["NIFTY", "SENSEX"]):
        if idx_cols[i].button(t, key=f"idx_{t}", use_container_width=True):
            ticker_input = t

    st.divider()
    max_articles   = st.slider("Max Articles", 10, 40, 20)
    price_period   = st.selectbox("Price Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    show_adversarial = st.checkbox("Adversarial Robustness", value=False)
    st.divider()
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.markdown("""
    ---
    **Data Sources**
    - Economic Times Markets RSS
    - MoneyControl RSS
    - Yahoo Finance (NSE)

    **Models**
    - FinBERT (ProsusAI)
    - Pearson Correlation
    - Backtest Engine
    """)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🇮🇳 Indian Stock Market Sentiment Analyzer")
st.caption("FinBERT NLP · NSE/BSE · Economic Times · MoneyControl · Real-time sentiment")

if not analyze_btn:
    # Show market overview cards when idle
    st.divider()
    st.markdown("### Popular Nifty 50 Stocks")
    cols = st.columns(4)
    showcase = [
        ("RELIANCE", "Reliance Industries", "Energy/Telecom"),
        ("TCS",      "Tata Consultancy",    "IT"),
        ("HDFCBANK", "HDFC Bank",           "Banking"),
        ("INFY",     "Infosys",             "IT"),
    ]
    for i, (t, name, sector) in enumerate(showcase):
        with cols[i]:
            st.markdown(f"""
            <div class="index-card">
                <div style='font-size:18px;font-weight:700;font-family:monospace;color:#e6edf3;'>{t}</div>
                <div style='font-size:12px;color:#8b949e;margin-top:4px;'>{name}</div>
                <div style='font-size:11px;color:#1D9E75;margin-top:2px;'>{sector}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#8b949e;'>
        <div style='font-size:42px;margin-bottom:12px;'>📊</div>
        <div style='font-size:16px;font-family:monospace;'>Select a ticker and click Analyze</div>
        <div style='font-size:13px;margin-top:8px;'>
            Supports all NSE listed stocks · Nifty 50 · Sensex
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

ticker = clean_ticker(ticker_input)

# ── Load FinBERT ──────────────────────────────────────────────────────────────
with st.spinner("Loading FinBERT model..."):
    nlp = load_finbert()

# ── Fetch & Analyze ───────────────────────────────────────────────────────────
with st.status(f"Analyzing {ticker} (NSE)...", expanded=True) as status:
    st.write("📰 Scraping Economic Times, MoneyControl, Yahoo Finance...")
    news_df = get_all_news(ticker, max_total=max_articles)

    st.write("🤖 Running FinBERT inference on headlines...")
    if news_df.empty:
        st.error(f"No news found for '{ticker}'. Try: RELIANCE, TCS, INFY, HDFCBANK")
        st.stop()
    news_df = analyze_dataframe(news_df, text_col="title")

    st.write("📈 Fetching NSE price data (₹ INR)...")
    price_df     = get_price_history(ticker, period=price_period)
    ticker_info  = get_ticker_info(ticker)
    tech_signals = compute_technical_signals(price_df) if not price_df.empty else {}

    st.write("📐 Computing Pearson correlation & backtest...")
    agg              = compute_aggregate_sentiment(news_df)
    sentiment_series = build_sentiment_series(news_df)
    corr_results     = compute_all_correlations(sentiment_series, price_df)
    backtest         = sentiment_backtest(news_df, price_df)

    status.update(label="✅ Analysis complete!", state="complete")

# ── Section 1: Key Metrics ────────────────────────────────────────────────────
st.divider()
company_name = ticker_info.get("name", ticker)
sector       = ticker_info.get("sector", "N/A")
st.subheader(f"{company_name} — {ticker}.NS")
st.caption(f"Sector: {sector} · Exchange: NSE · Currency: ₹ INR")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Ticker", f"{ticker}.NS")
m2.metric("Signal", f"{sentiment_emoji(agg['overall_sentiment'])} {agg['overall_sentiment']}")
m3.metric("Sentiment Score", f"{agg['overall_score']:+.3f}", help="-1 = very bearish, +1 = very bullish")
m4.metric("Confidence", f"{agg['avg_confidence']*100:.1f}%")
m5.metric("Articles", agg["article_count"])

if tech_signals and tech_signals.get("price"):
    chg = tech_signals.get("daily_change_pct", 0)
    m6.metric(
        "LTP (₹)",
        f"₹{tech_signals['price']:,.2f}",
        delta=f"{chg:+.2f}%",
        delta_color="normal" if chg >= 0 else "inverse",
    )

# Market cap + fundamentals
info_parts = []
if ticker_info.get("market_cap"):
    info_parts.append(f"MCap: {format_market_cap(ticker_info['market_cap'])}")
if ticker_info.get("pe_ratio"):
    info_parts.append(f"P/E: {ticker_info['pe_ratio']:.1f}")
if ticker_info.get("52w_high"):
    info_parts.append(f"52W High: ₹{ticker_info['52w_high']:,.0f}")
if ticker_info.get("52w_low"):
    info_parts.append(f"52W Low: ₹{ticker_info['52w_low']:,.0f}")
if tech_signals.get("rsi"):
    rsi = tech_signals["rsi"]
    rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    info_parts.append(f"RSI: {rsi} ({rsi_label})")

if info_parts:
    st.caption(" · ".join(info_parts))


# ── Section 2: Charts ─────────────────────────────────────────────────────────
st.divider()
col_sent, col_price = st.columns([1, 2])

with col_sent:
    st.subheader("Sentiment Breakdown")

    labels = ["Bullish", "Neutral", "Bearish"]
    values = [agg["bullish_pct"], agg["neutral_pct"], agg["bearish_pct"]]
    colors = ["#1D9E75", "#888780", "#D85A30"]

    fig_donut = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker_colors=colors, textinfo="label+percent", textfont_size=12,
    ))
    fig_donut.update_layout(
        showlegend=False, height=240,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
    )
    fig_donut.add_annotation(
        text=f"<b>{agg['overall_sentiment']}</b>",
        x=0.5, y=0.5, font_size=14, showarrow=False,
        font_color=sentiment_color(agg["overall_sentiment"]),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # Technical signals summary
    if tech_signals:
        st.markdown("**Technical Signals**")
        signals_data = {
            "Above SMA 20":    "✅ Yes" if tech_signals.get("above_sma20") else "❌ No",
            "Golden Cross":    "✅ Yes" if tech_signals.get("golden_cross") else "❌ No",
            "5-Day Trend":     tech_signals.get("recent_trend", "N/A").title(),
            "Volatility (5d)": f"{tech_signals.get('volatility', 0):.2f}%",
        }
        for k, v in signals_data.items():
            col_a, col_b = st.columns([1.2, 1])
            col_a.caption(k)
            col_b.caption(f"**{v}**")

with col_price:
    st.subheader(f"Price Chart — ₹ INR ({price_period})")
    if not price_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=price_df.index,
            open=price_df["Open"], high=price_df["High"],
            low=price_df["Low"],   close=price_df["Close"],
            name=ticker,
            increasing_line_color="#1D9E75",
            decreasing_line_color="#D85A30",
        ))
        if "sma_20" in price_df.columns:
            fig.add_trace(go.Scatter(
                x=price_df.index, y=price_df["sma_20"],
                name="SMA 20", line=dict(color="#378ADD", width=1, dash="dot"),
            ))
        if "sma_5" in price_df.columns:
            fig.add_trace(go.Scatter(
                x=price_df.index, y=price_df["sma_5"],
                name="SMA 5", line=dict(color="#EF9F27", width=1, dash="dot"),
            ))

        # Add volume as subplot
        fig.add_trace(go.Bar(
            x=price_df.index, y=price_df["Volume"],
            name="Volume", yaxis="y2",
            marker_color="rgba(55,138,221,0.2)",
        ))

        fig.update_layout(
            height=420,
            margin=dict(t=20, b=40, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#8b949e",
            xaxis=dict(gridcolor="#30363d", rangeslider_visible=False),
            yaxis=dict(gridcolor="#30363d", title="Price (₹)"),
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                       bgcolor="rgba(0,0,0,0)", font_color="#8b949e"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Price data unavailable. Try .NS suffix e.g. RELIANCE.NS")

# ── Section 3: RSI Chart ──────────────────────────────────────────────────────
if not price_df.empty and "rsi" in price_df.columns:
    st.subheader("RSI — Relative Strength Index")
    rsi_clean = price_df["rsi"].dropna()
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=rsi_clean.index, y=rsi_clean.values,
        line=dict(color="#EF9F27", width=1.5), name="RSI 14",
    ))
    fig_rsi.add_hline(y=70, line_color="#D85A30", line_dash="dash", line_width=0.8,
                      annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_color="#1D9E75", line_dash="dash", line_width=0.8,
                      annotation_text="Oversold (30)")
    fig_rsi.update_layout(
        height=180, margin=dict(t=10, b=30, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8b949e", yaxis=dict(gridcolor="#30363d", range=[0,100]),
        xaxis=dict(gridcolor="#30363d"),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)


# ── Section 4: News Feed ──────────────────────────────────────────────────────
st.divider()
st.subheader(f"📰 News Signal Feed — {len(news_df)} Headlines")
st.caption("Sources: Economic Times · MoneyControl · Yahoo Finance India")

tab_all, tab_bull, tab_bear, tab_neutral = st.tabs(["All", "🟢 Bullish", "🔴 Bearish", "⚪ Neutral"])

def render_news(df_slice):
    if df_slice.empty:
        st.info("No articles in this category.")
        return
    for _, row in df_slice.head(20).iterrows():
        sentiment = row.get("sentiment", "Neutral")
        score     = row.get("sentiment_score", 0)
        conf      = row.get("confidence", 0)
        color     = sentiment_color(sentiment)
        css_cls   = sentiment.lower()
        st.markdown(f"""
        <div class="news-card news-{css_cls}">
            <div style='font-size:13px;color:#e6edf3;margin-bottom:4px;line-height:1.5;'>{row['title']}</div>
            <div style='font-size:11px;color:#8b949e;display:flex;gap:12px;flex-wrap:wrap;'>
                <span style='color:{color};font-weight:700;'>{sentiment_emoji(sentiment)} {sentiment}</span>
                <span>Score: {score:+.3f}</span>
                <span>Confidence: {conf*100:.0f}%</span>
                <span>📰 {row.get('source','')}</span>
                <span>🕐 {row.get('published','')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_all:     render_news(news_df)
with tab_bull:    render_news(news_df[news_df["sentiment"] == "Bullish"])
with tab_bear:    render_news(news_df[news_df["sentiment"] == "Bearish"])
with tab_neutral: render_news(news_df[news_df["sentiment"] == "Neutral"])


# ── Section 5: Correlation ────────────────────────────────────────────────────
st.divider()
st.subheader("📐 Statistical Correlation — Sentiment vs Returns")
st.caption("Pearson r between FinBERT daily sentiment score and NSE price returns")

corr_cols = st.columns(3)
lag_labels = {"lag_0d": "Same Day", "lag_1d": "Next Day (Lag 1)", "lag_2d": "2-Day Lag"}
for i, (key, label) in enumerate(lag_labels.items()):
    res = corr_results.get(key, {})
    r   = res.get("pearson_r")
    p   = res.get("p_value")
    n   = res.get("n_overlap", 0)
    with corr_cols[i]:
        st.metric(label, f"r = {r:.3f}" if r is not None else "N/A")
        if r is not None:
            sig = "✓ Significant (p<0.05)" if p is not None and p < 0.05 else f"✗ p = {p:.3f}"
            st.caption(f"{n} days · {sig}")
            st.caption(res.get("interpretation", ""))

# Rolling correlation
rolling_corr = compute_rolling_correlation(sentiment_series, price_df, window=10)
if not rolling_corr.empty:
    st.markdown("**Rolling 10-Day Correlation**")
    fig_corr = go.Figure(go.Scatter(
        x=rolling_corr.index, y=rolling_corr.values,
        fill="tozeroy", line=dict(color="#378ADD", width=1.5),
        fillcolor="rgba(55,138,221,0.1)", name="Rolling r",
    ))
    fig_corr.add_hline(y=0, line_color="#8b949e", line_dash="dash", line_width=0.8)
    fig_corr.update_layout(
        height=180, margin=dict(t=10, b=30, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8b949e",
        yaxis=dict(gridcolor="#30363d", range=[-1.1, 1.1]),
        xaxis=dict(gridcolor="#30363d"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ── Section 6: Backtest ───────────────────────────────────────────────────────
st.divider()
st.subheader("🔁 Signal Backtest")
st.caption("Did the FinBERT signal predict next-day NSE price direction?")

bt1, bt2, bt3, bt4 = st.columns(4)
bt1.metric("Overall Accuracy",  f"{backtest.get('accuracy')}%" if backtest.get("accuracy") else "N/A")
bt2.metric("Bullish Accuracy",  f"{backtest.get('bullish_accuracy')}%" if backtest.get("bullish_accuracy") else "N/A")
bt3.metric("Bearish Accuracy",  f"{backtest.get('bearish_accuracy')}%" if backtest.get("bearish_accuracy") else "N/A")
bt4.metric("Total Signals",     backtest.get("n_signals", 0))
st.caption("⚠️ Educational only. Not financial advice. SEBI regulations apply.")


# ── Section 7: Adversarial (InfoSec) ─────────────────────────────────────────
if show_adversarial:
    st.divider()
    st.subheader("🔐 Adversarial Robustness — NLP Security Analysis")
    st.caption(
        "Tests whether injected fake bullish/bearish phrases can manipulate the model's sentiment. "
        "Relevant to financial misinformation and AI security research."
    )
    from adversarial import run_adversarial_tests
    sample_headlines = news_df["title"].head(3).tolist()
    adv_rows = []
    with st.spinner("Running adversarial injection tests..."):
        for h in sample_headlines:
            res = run_adversarial_tests(h, nlp=nlp)
            adv_rows.append({
                "Headline":        (h[:70] + "...") if len(h) > 70 else h,
                "Original Label":  res["original_label"],
                "Flip Rate":       f"{res['flip_rate']*100:.0f}%",
                "Max Score Δ":     f"{res['max_score_delta']:+.3f}",
                "Robustness":      res["robustness_rating"].split("—")[0].strip(),
            })
    st.dataframe(pd.DataFrame(adv_rows), use_container_width=True)
    st.info(
        "A high flip rate = model vulnerable to financial misinformation injection. "
        "Relevant to AI security, adversarial NLP, and market manipulation research."
    )


# ── Section 8: Export ─────────────────────────────────────────────────────────
st.divider()
with st.expander("📥 Export Data"):
    export_df = news_df[["title", "sentiment", "sentiment_score", "confidence", "source", "published"]].round(4)
    st.dataframe(export_df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        data=news_df.to_csv(index=False),
        file_name=f"{ticker}_NSE_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "🇮🇳 Indian Stock Market Sentiment Analyzer · "
    "FinBERT (ProsusAI) · NSE/BSE via yfinance · "
    "Economic Times · MoneyControl · Streamlit · Plotly"
)
