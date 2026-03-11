"""
SentimentAlpha — Real-Time Streamlit Dashboard
================================================
Features:
  1. Live news feed with color-coded sentiment tags
  2. Sentiment heatmap across Nifty 50 stocks
  3. Alert system — flag stocks with sudden sentiment shift
  4. "Ask the AI" chatbot — stock research summaries
  5. Backtesting results viewer
  6. ML model predictions & feature importance
"""

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Must be the first Streamlit command ────────────────────────────────────
st.set_page_config(
    page_title="SentimentAlpha — AI Trading Engine",
    page_icon="SA",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import NIFTY50_STOCKS, STOCK_TICKERS, LLM_PROVIDER
from database import DatabaseManager

logger = logging.getLogger("sentimentalpha.dashboard")

# Initialize database
db = DatabaseManager()


# ── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .sentiment-bullish { color: #00c853; font-weight: bold; }
    .sentiment-bearish { color: #ff1744; font-weight: bold; }
    .sentiment-neutral { color: #ffc107; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px; padding: 20px; margin: 5px;
        border: 1px solid #0f3460;
    }
    .alert-high { border-left: 4px solid #ff1744; padding-left: 10px; }
    .alert-medium { border-left: 4px solid #ffc107; padding-left: 10px; }
    .alert-low { border-left: 4px solid #00c853; padding-left: 10px; }
    .stMetric { background-color: #0e1117; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────

def sentiment_color(sentiment: str) -> str:
    """Return color for sentiment label."""
    return {"bullish": "#00c853", "bearish": "#ff1744", "neutral": "#ffc107"}.get(
        sentiment, "#888888"
    )


def sentiment_emoji(sentiment: str) -> str:
    return {"bullish": "[+]", "bearish": "[-]", "neutral": "[~]"}.get(sentiment, "[?]")


def safe_json_loads(text, default=None):
    if default is None:
        default = []
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-share.png", width=60)
        st.title("SentimentAlpha")
        st.caption("AI-Powered News Sentiment Trading Engine")

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Live News Feed",
                "Sentiment Heatmap",
                "Alerts",
                "Backtesting",
                "ML Models",
                "Ask the AI",
                "Settings",
            ],
            captions=[
                "Overview and key metrics",
                "Browse all scraped news articles",
                "Visual map of sentiment across stocks",
                "Sudden sentiment shift warnings",
                "Strategy performance comparison",
                "ML model predictions and features",
                "Type a stock name, get a research report",
                "Database stats and pipeline commands",
            ],
        )

        st.divider()

        # Quick stats
        stats = db.get_stats()
        st.metric("Total Articles", f"{stats.get('news_articles', 0):,}")
        st.metric("Analyzed", f"{stats.get('sentiment_scores', 0):,}")
        st.metric("Price Records", f"{stats.get('stock_prices', 0):,}")
        st.metric("Active Alerts", f"{stats.get('alerts', 0):,}")

        st.divider()
        st.caption(f"LLM Provider: **{LLM_PROVIDER}**")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return page


# ── Page: Dashboard ────────────────────────────────────────────────────────

def page_dashboard():
    st.title("SentimentAlpha Dashboard")
    st.markdown("**AI-Powered News Sentiment Trading Engine for Nifty 50**")

    # Getting Started guide (collapsible)
    with st.expander("How to Use This App — Click to expand", expanded=False):
        st.markdown("""
**What is SentimentAlpha?**

This app scrapes financial news about Indian Nifty 50 stocks, uses an AI model (FinBERT) to determine whether each article is **bullish**, **bearish**, or **neutral**, then builds trading signals from that sentiment data.

---

**Pages explained:**

| Page | What it does |
|---|---|
| **Dashboard** (this page) | Shows key numbers, latest news with sentiment tags, alerts, and top bullish/bearish stocks |
| **Live News Feed** | Browse all scraped articles — filter by stock, sentiment, or source |
| **Sentiment Heatmap** | A treemap showing which stocks have the most positive or negative news sentiment |
| **Alerts** | Flags stocks where sentiment has suddenly shifted (e.g. a company went from bullish to bearish overnight) |
| **Backtesting** | Shows how trading strategies based on sentiment would have performed historically |
| **ML Models** | Displays predictions from Random Forest and XGBoost models trained on sentiment + price data |
| **Ask the AI** | Type any stock name (e.g. "Reliance" or "TCS") and get a full research summary with sentiment breakdown |
| **Settings** | View database statistics and find terminal commands to re-run the pipeline |

---

**How to refresh data (run in terminal):**
```bash
source venv/bin/activate
python main.py --full
```
This re-scrapes news, re-analyzes sentiment, rebuilds factors, and re-runs backtests.
        """)

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    stats = db.get_stats()
    with col1:
        st.metric("Articles Ingested", f"{stats.get('news_articles', 0):,}")
    with col2:
        st.metric("Sentiments Scored", f"{stats.get('sentiment_scores', 0):,}")
    with col3:
        st.metric("Price Points", f"{stats.get('stock_prices', 0):,}")
    with col4:
        st.metric("Daily Scores", f"{stats.get('daily_sentiment', 0):,}")
    with col5:
        st.metric("Alerts", f"{stats.get('alerts', 0):,}")

    st.divider()

    # Recent articles with sentiment
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Latest Analyzed News")
        articles = db.get_recent_articles(limit=15)
        if articles:
            for article in articles:
                sentiment = article.get("sentiment", "neutral")
                emoji = sentiment_emoji(sentiment)
                confidence = article.get("confidence", 0)
                title = article.get("title", "No title")
                source = article.get("source", "unknown")
                pub = article.get("published_at", "")

                st.markdown(
                    f"{emoji} **{title}**  \n"
                    f"<small>{source} | {str(pub)[:16]} | "
                    f"<span style='color:{sentiment_color(sentiment)}'>"
                    f"{sentiment.upper()}</span> "
                    f"({confidence:.0%} conf)</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")
        else:
            st.info("No articles yet. Run the ingestion pipeline first!")

    with col_right:
        st.subheader("Recent Alerts")
        alerts = db.get_unread_alerts(limit=10)
        if alerts:
            for alert in alerts:
                severity = alert.get("severity", "medium")
                icon = {"high": "[!!]", "medium": "[!]", "low": "[i]"}.get(severity, "[-]")
                st.markdown(
                    f"{icon} **{alert.get('ticker', '')}** — "
                    f"{alert.get('message', '')}",
                )
        else:
            st.success("No active alerts")

        st.divider()

        # Quick sentiment overview
        st.subheader("Sentiment Overview")
        all_sent = db.get_all_daily_sentiment()
        if all_sent:
            df = pd.DataFrame(all_sent)
            latest = df.groupby("ticker")["rolling_7d"].last().sort_values(ascending=False)

            if not latest.empty:
                top_bull = latest.head(5)
                st.markdown("**Most Bullish:**")
                for ticker, score in top_bull.items():
                    name = NIFTY50_STOCKS.get(ticker, {}).get("name", ticker)
                    st.markdown(f"[+] {name}: {score:+.3f}")

                st.markdown("")
                top_bear = latest.tail(5)
                st.markdown("**Most Bearish:**")
                for ticker, score in top_bear.items():
                    name = NIFTY50_STOCKS.get(ticker, {}).get("name", ticker)
                    st.markdown(f"[-] {name}: {score:+.3f}")


# ── Page: Live News Feed ──────────────────────────────────────────────────

def page_news_feed():
    st.title("Live News Feed")
    st.caption("All scraped news articles with AI sentiment scores. Use the filters below to narrow down.")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_sentiment = st.multiselect(
            "Filter by Sentiment",
            ["bullish", "bearish", "neutral"],
            default=["bullish", "bearish", "neutral"],
        )
    with col2:
        filter_source = st.multiselect(
            "Filter by Source",
            ["moneycontrol", "economic_times", "livemint", "reddit", "earnings_transcript"],
        )
    with col3:
        filter_stock = st.selectbox(
            "Filter by Stock",
            ["All"] + [f"{v['name']} ({k})" for k, v in NIFTY50_STOCKS.items()],
        )

    st.divider()

    articles = db.get_recent_articles(limit=200)
    if not articles:
        st.info("No articles found. Run the ingestion pipeline first!")
        return

    # Apply filters
    filtered = articles
    if filter_sentiment:
        filtered = [a for a in filtered if a.get("sentiment") in filter_sentiment]
    if filter_source:
        filtered = [a for a in filtered if a.get("source") in filter_source]
    if filter_stock and filter_stock != "All":
        ticker = filter_stock.split("(")[-1].strip(")")
        filtered = [
            a for a in filtered
            if ticker in str(a.get("affected_stocks", ""))
        ]

    st.caption(f"Showing {len(filtered)} articles")

    for article in filtered[:50]:
        sentiment = article.get("sentiment", "neutral")
        color = sentiment_color(sentiment)
        emoji = sentiment_emoji(sentiment)
        confidence = article.get("confidence", 0)
        signal = article.get("signal_type", "")

        with st.container():
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(
                    f"### {emoji} {article.get('title', 'No title')}\n"
                    f"**Source:** {article.get('source', 'unknown')} | "
                    f"**Published:** {str(article.get('published_at', ''))[:16]}"
                )
                if article.get("summary"):
                    st.markdown(f"_{article['summary'][:300]}_")

            with col_b:
                st.markdown(
                    f"<div style='text-align:center; padding:10px; "
                    f"border-radius:8px; background:{color}22; border:1px solid {color}'>"
                    f"<span style='color:{color}; font-size:1.2em; font-weight:bold'>"
                    f"{sentiment.upper()}</span><br>"
                    f"<small>{confidence:.0%} confidence</small><br>"
                    f"<small>{signal}</small></div>",
                    unsafe_allow_html=True,
                )

            affected = safe_json_loads(article.get("affected_stocks"))
            if affected:
                st.markdown(f"**Stocks:** {', '.join(affected)}")

        st.divider()


# ── Page: Sentiment Heatmap ────────────────────────────────────────────────

def page_heatmap():
    st.title("Sentiment Heatmap — Nifty 50")

    score_col = st.selectbox(
        "Score Type", ["rolling_7d", "rolling_14d", "raw_score"],
        index=0,
    )

    all_sent = db.get_all_daily_sentiment()
    if not all_sent:
        st.warning("No sentiment data available. Run the pipeline first!")
        return

    df = pd.DataFrame(all_sent)
    df["factor_date"] = pd.to_datetime(df["factor_date"])

    # Current heatmap — latest score per stock
    latest = df.groupby("ticker")[score_col].last().reset_index()
    latest["name"] = latest["ticker"].map(
        lambda t: NIFTY50_STOCKS.get(t, {}).get("name", t)
    )
    latest["sector"] = latest["ticker"].map(
        lambda t: NIFTY50_STOCKS.get(t, {}).get("sector", "Other")
    )
    latest = latest.sort_values(score_col, ascending=False)

    # Treemap
    st.subheader("Current Sentiment Treemap")
    if not latest.empty:
        fig = px.treemap(
            latest, path=["sector", "name"], values=latest[score_col].abs() + 0.01,
            color=score_col, color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Nifty 50 Sentiment Heatmap",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # Time series heatmap
    st.subheader("Sentiment Over Time")
    pivot = df.pivot_table(
        index="factor_date", columns="ticker", values=score_col
    ).fillna(0)

    if not pivot.empty:
        # Show last 30 days
        pivot_recent = pivot.tail(30)

        fig = px.imshow(
            pivot_recent.T, aspect="auto",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels=dict(color=score_col),
            title="Daily Sentiment Scores (Last 30 Days)",
        )
        fig.update_layout(height=max(400, len(pivot_recent.columns) * 18))
        st.plotly_chart(fig, use_container_width=True)

    # Individual stock drill-down
    st.subheader("Stock Deep Dive")
    selected = st.selectbox(
        "Select Stock",
        [f"{v['name']} ({k})" for k, v in NIFTY50_STOCKS.items()],
    )
    ticker = selected.split("(")[-1].strip(")")

    stock_sent = df[df["ticker"] == ticker].sort_values("factor_date")
    if not stock_sent.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_sent["factor_date"], y=stock_sent["raw_score"],
            mode="lines", name="Raw Score", line=dict(color="#888", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=stock_sent["factor_date"], y=stock_sent["rolling_7d"],
            mode="lines", name="7-Day Rolling", line=dict(color="#2196f3", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=stock_sent["factor_date"], y=stock_sent["rolling_14d"],
            mode="lines", name="14-Day Rolling", line=dict(color="#ff9800", width=2),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"Sentiment Trend — {NIFTY50_STOCKS.get(ticker, {}).get('name', ticker)}",
            yaxis_title="Sentiment Score",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Article count bar chart
        fig2 = px.bar(
            stock_sent, x="factor_date", y="article_count",
            title="Daily Article Count",
            color="raw_score", color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Alerts ───────────────────────────────────────────────────────────

def page_alerts():
    st.title("Sentiment Alerts")

    alerts = db.get_unread_alerts(limit=100)

    if not alerts:
        st.success("No active alerts — all quiet on the sentiment front!")
        return

    # Summary metrics
    high = sum(1 for a in alerts if a.get("severity") == "high")
    medium = sum(1 for a in alerts if a.get("severity") == "medium")
    low = sum(1 for a in alerts if a.get("severity") == "low")

    col1, col2, col3 = st.columns(3)
    col1.metric("High Severity", high)
    col2.metric("Medium Severity", medium)
    col3.metric("Low Severity", low)

    st.divider()

    for alert in alerts:
        severity = alert.get("severity", "medium")
        icon = {"high": "[!!]", "medium": "[!]", "low": "[i]"}.get(severity, "[-]")
        ticker = alert.get("ticker", "")
        stock_name = NIFTY50_STOCKS.get(ticker, {}).get("name", ticker)

        with st.expander(f"{icon} {stock_name} ({ticker}) — {alert.get('alert_type', '')}", expanded=(severity == "high")):
            st.markdown(f"**Message:** {alert.get('message', '')}")
            st.markdown(f"**Created:** {alert.get('created_at', '')}")
            st.markdown(f"**Severity:** {severity.upper()}")

            if st.button(f"Mark as read", key=f"alert_{alert.get('id')}"):
                db.mark_alert_read(alert["id"])
                st.rerun()


# ── Page: Backtesting ─────────────────────────────────────────────────────

def page_backtesting():
    st.title("Backtesting Results")

    backtests = db.get_all_backtests()

    if not backtests:
        st.info("No backtest results yet. Run `python main.py --backtest` in the terminal to generate results.")

        st.markdown("""
        ### How to run backtests:
        ```bash
        python backtester.py
        ```
        Or use the main pipeline:
        ```bash
        python main.py --backtest
        ```
        """)
        return

    # Comparison table
    st.subheader("Strategy Comparison")
    bt_df = pd.DataFrame(backtests)
    display_cols = [
        "strategy_name", "start_date", "end_date",
        "total_return", "annualized_return", "sharpe_ratio",
        "max_drawdown", "win_rate", "num_trades",
    ]
    available_cols = [c for c in display_cols if c in bt_df.columns]

    formatted = bt_df[available_cols].copy()
    for col in ["total_return", "annualized_return", "max_drawdown", "win_rate"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    if "sharpe_ratio" in formatted.columns:
        formatted["sharpe_ratio"] = formatted["sharpe_ratio"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

    st.dataframe(formatted, use_container_width=True)

    # Performance charts
    st.subheader("Performance Metrics Comparison")

    numeric_cols = ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown"]
    available_numeric = [c for c in numeric_cols if c in bt_df.columns]

    if available_numeric:
        fig = go.Figure()
        for col in available_numeric:
            fig.add_trace(go.Bar(
                name=col.replace("_", " ").title(),
                x=bt_df["strategy_name"],
                y=bt_df[col],
            ))
        fig.update_layout(
            barmode="group", title="Strategy Metrics Comparison",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page: ML Models ────────────────────────────────────────────────────────

def page_ml_models():
    st.title("ML Model Results")

    # Predictions
    st.subheader("Latest Predictions")
    predictions = db.get_predictions()
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df["stock_name"] = pred_df["ticker"].map(
            lambda t: NIFTY50_STOCKS.get(t, {}).get("name", t)
        )

        # Latest predictions per model
        for model_name in pred_df["model_name"].unique():
            st.markdown(f"### Model: {model_name}")
            model_preds = pred_df[pred_df["model_name"] == model_name].copy()
            latest = model_preds.groupby("ticker").last().reset_index()
            latest = latest.sort_values("predicted_return", ascending=False)

            # Top picks
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Buy Signals:**")
                top5 = latest.head(5)
                for _, row in top5.iterrows():
                    name = NIFTY50_STOCKS.get(row["ticker"], {}).get("name", row["ticker"])
                    st.markdown(f"  [+] {name}: {row['predicted_return']:+.2%}")

            with col2:
                st.markdown("**Top Avoid Signals:**")
                bottom5 = latest.tail(5)
                for _, row in bottom5.iterrows():
                    name = NIFTY50_STOCKS.get(row["ticker"], {}).get("name", row["ticker"])
                    st.markdown(f"  [-] {name}: {row['predicted_return']:+.2%}")

            # Prediction bar chart
            fig = px.bar(
                latest, x="stock_name", y="predicted_return",
                color="predicted_return", color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title=f"Predicted {model_name} Returns",
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet. Run ML models first!")
        st.code("python ml_models.py", language="bash")

    # Feature importance from saved files
    st.subheader("Feature Importance")
    import pickle
    from config import MODELS_DIR

    for model_file in MODELS_DIR.glob("*.pkl"):
        try:
            with open(model_file, "rb") as f:
                data = pickle.load(f)
            model = data.get("model")
            if hasattr(model, "feature_importances_") and hasattr(model, "_feature_names"):
                fi_df = pd.DataFrame({
                    "feature": model._feature_names[:len(model.feature_importances_)],
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=True)

                fig = px.bar(
                    fi_df, x="importance", y="feature",
                    orientation="h",
                    title=f"Feature Importance — {model_file.stem}",
                    color="importance", color_continuous_scale="Viridis",
                )
                fig.update_layout(height=max(300, len(fi_df) * 25))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load {model_file.stem}: {e}")


# ── Page: Ask the AI ───────────────────────────────────────────────────────

def page_ask_ai():
    st.title("Ask the AI — Stock Research Assistant")

    st.markdown(
        "Type a stock name or ticker to get an AI-generated research summary "
        "based on recent news and sentiment analysis."
    )

    st.info(
        "**Try it:** Type a stock name like **Reliance**, **TCS**, **Infosys**, **HDFC Bank**, "
        "or any Nifty 50 company. You can also use tickers like **INFY.NS**."
    )

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask about a stock (e.g., 'Reliance', 'TCS', 'INFY.NS')...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Find the stock
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                response = _handle_ai_query(user_input)
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})


def _handle_ai_query(query: str) -> str:
    """Process a user query and return an AI-generated response."""
    query_lower = query.lower().strip()

    # Try to match a stock
    matched_ticker = None

    # Direct ticker match
    for ticker in NIFTY50_STOCKS:
        if ticker.lower().replace(".ns", "") in query_lower:
            matched_ticker = ticker
            break

    # Name match
    if not matched_ticker:
        for ticker, info in NIFTY50_STOCKS.items():
            if info["name"].lower() in query_lower:
                matched_ticker = ticker
                break

    # Partial match
    if not matched_ticker:
        for ticker, info in NIFTY50_STOCKS.items():
            name_parts = info["name"].lower().split()
            if any(part in query_lower for part in name_parts if len(part) > 3):
                matched_ticker = ticker
                break

    if not matched_ticker:
        return (
            "I couldn't identify a specific Nifty 50 stock from your query. "
            "Try asking about a specific stock like 'Reliance', 'TCS', or 'HDFC Bank'.\n\n"
            "**Available stocks:** " +
            ", ".join(v["name"] for v in list(NIFTY50_STOCKS.values())[:10]) + "..."
        )

    # Get stock info
    stock_info = NIFTY50_STOCKS[matched_ticker]
    stock_name = stock_info["name"]
    sector = stock_info["sector"]

    # Get articles
    articles = db.get_articles_for_stock(matched_ticker, days=30)

    # Get sentiment data
    sent_data = db.get_daily_sentiment(matched_ticker, days=30)

    # Get latest brief
    brief = db.get_latest_brief(matched_ticker)

    # Build response
    response = f"## {stock_name} ({matched_ticker})\n**Sector:** {sector}\n\n"

    if sent_data:
        df = pd.DataFrame(sent_data)
        latest_score = df["rolling_7d"].iloc[-1] if "rolling_7d" in df.columns else 0
        sentiment_label = "Bullish" if latest_score > 0.1 else "Bearish" if latest_score < -0.1 else "Neutral"

        response += f"### Current Sentiment: {sentiment_label}\n"
        response += f"- **Rolling 7-Day Score:** {latest_score:+.3f}\n"
        response += f"- **Data points (30d):** {len(df)}\n\n"

    if articles:
        sentiments = [a.get("sentiment") for a in articles if a.get("sentiment")]
        total = len(sentiments) or 1
        response += f"### News Analysis ({len(articles)} articles, last 30 days)\n"
        response += f"- Bullish: {sentiments.count('bullish')}/{total} ({sentiments.count('bullish')/total:.0%})\n"
        response += f"- Bearish: {sentiments.count('bearish')}/{total} ({sentiments.count('bearish')/total:.0%})\n"
        response += f"- Neutral: {sentiments.count('neutral')}/{total} ({sentiments.count('neutral')/total:.0%})\n\n"

        response += "### Recent Headlines\n"
        for a in articles[:5]:
            emoji = sentiment_emoji(a.get("sentiment", "neutral"))
            response += f"- {emoji} {a.get('title', 'No title')}\n"
        response += "\n"

    if brief and brief.get("brief"):
        response += f"### Research Brief\n{brief['brief']}\n\n"
    elif articles:
        response += (
            "### Research Brief\n"
            "*No AI-generated brief available yet. Run the sentiment pipeline to generate one.*\n\n"
        )

    if not articles and not sent_data:
        response += (
            "*No data available for this stock yet. "
            "Run the full pipeline (`python main.py --full`) to ingest news and compute sentiment.*\n"
        )

    return response


# ── Page: Settings ─────────────────────────────────────────────────────────

def page_settings():
    st.title("Settings & Pipeline Control")

    st.subheader("Pipeline Status")

    stats = db.get_stats()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Database Statistics")
        for table, count in stats.items():
            st.markdown(f"- **{table.replace('_', ' ').title()}:** {count:,} records")

    with col2:
        st.markdown("### Configuration")
        st.markdown(f"- **LLM Provider:** {LLM_PROVIDER}")
        st.markdown(f"- **Database:** {db.db_path}")

    st.divider()

    st.subheader("Run Pipeline Components")
    st.markdown("Use the terminal to run pipeline components:")

    st.code("""
# Full pipeline (ingest → analyze → build factors → backtest → train ML)
python main.py --full

# Individual components
python ingestion.py          # Scrape news
python price_data.py         # Fetch stock prices
python sentiment_analyzer.py # Run LLM sentiment analysis
python factor_builder.py     # Build sentiment factors
python backtester.py         # Run backtests
python ml_models.py          # Train ML models

# Start the dashboard
streamlit run dashboard.py
    """, language="bash")


# ── Main App Router ────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if "Dashboard" in page:
        page_dashboard()
    elif "News Feed" in page:
        page_news_feed()
    elif "Heatmap" in page:
        page_heatmap()
    elif "Alerts" in page:
        page_alerts()
    elif "Backtesting" in page:
        page_backtesting()
    elif "ML Models" in page:
        page_ml_models()
    elif "Ask the AI" in page:
        page_ask_ai()
    elif "Settings" in page:
        page_settings()


if __name__ == "__main__":
    main()
